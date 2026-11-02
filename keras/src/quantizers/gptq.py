import functools

from keras.src import ops
from keras.src.ops import linalg
from keras.src.quantizers.calibrator import Calibrator
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_zero_point


def _stable_permutation(metric):
    """Return a stable permutation that sorts `metric` in descending order.
    Uses an index-based jitter to break ties deterministically."""
    n = ops.shape(metric)[0]
    idx = ops.arange(0, n, dtype="int32")
    # tiny jitter = (idx / n) * 1e-12 so it never flips a real strict ordering
    jitter = ops.divide(ops.cast(idx, "float32"), ops.cast(n, "float32"))
    metric_jittered = ops.add(metric, ops.multiply(jitter, 1e-12))
    # argsort by negative to get descending
    return ops.argsort(ops.negative(metric_jittered))


def _stored_scale_zero(weights, *, scale_dtype, **kwargs):
    """`compute_quantization_parameters`, with the scale the layer stores.

    The parameters are computed in float32, then the scale is rounded to
    `scale_dtype`, the dtype of the variable that holds it. The solve then
    chooses the codes, and corrects their error, against the scale the
    forward pass dequantizes with.
    """
    scale, zero, maxq = compute_quantization_parameters(weights, **kwargs)
    scale = ops.cast(ops.cast(scale, scale_dtype), "float32")
    return scale, zero, maxq


def gptq_quantize_matrix(
    weights_transpose,
    hessian,
    *,
    blocksize=128,
    group_size=-1,
    activation_order=False,
    order_metric=None,
    compute_scale_zero=compute_quantization_parameters,
):
    """
    Implements the GPTQ error correction updates.

    For a single column update (column j):
        e = invH[j, j] * (w_j - q_j)
        W[:, j+1:] -= e * invH[j, j+1:]
    where:
    - w_j is the original column,
    - q_j is the quantized column,
    - invH is the inverse Hessian,
    - e is the propagated error term.

    Across entire blocks:
        W[:, future] -= E_block * invH[block, future]
    where:
    - E_block is the quantization error accumulated for the current block,
    - invH[block, future] denotes the cross-block slice of the inverse Hessian,
    - W[:, future] are the columns yet to be quantized.

    The inverse Hessian used for error propagation is derived from the
    (already dampened) Hessian using the reference GPTQ/AutoGPTQ Cholesky
    formulation rather than a dense matrix inverse: the upper-triangular
    Cholesky factor of `H^-1` is computed via triangular solves and indexed
    directly inside the error-correction loop.

    Args:
        weights_transpose: Transposed weight matrix [out_features, in_features]
         to quantize.
        hessian: Dampened Hessian matrix [in_features, in_features]. The upper
         Cholesky factor of its inverse drives error propagation.
        blocksize: Size of the blocks to process (default: 128).
        group_size: Size of the groups for parameter reuse
         (default: -1, no grouping).
        activation_order: Whether to apply activation-order permutation
         (default: False).
        order_metric: Metric for ordering features
         (default: None, uses diag(H)).
        compute_scale_zero: Function to compute scale and zero for
         quantization.

    Returns:
        quantized_weights: Quantized weight matrix [out_features, in_features].
        scale: float32. Scale parameters for quantization
         [out_features, num_groups].
        zero: Zero-point parameters for quantization [out_features, num_groups].
        g_idx: int32. Group indices for each feature [in_features].
    """
    in_features = ops.shape(weights_transpose)[1]

    if activation_order:
        # Use diag(H) as the importance proxy by default (as in AutoGPTQ).
        if order_metric is None:
            order_metric = ops.diagonal(hessian)
        else:
            # sanitize provided metric
            order_metric = ops.cast(order_metric, "float32")
            order_metric = ops.where(
                ops.isfinite(order_metric),
                order_metric,
                ops.zeros_like(order_metric),
            )
        # Sort in descending order by importance
        perm = _stable_permutation(order_metric)
        inv_perm = ops.argsort(perm)

        weights_transpose = ops.take(weights_transpose, perm, axis=1)
        # Permute the Hessian *before* factorization. Cholesky does not commute
        # with permutation, so the factor must be computed on the reordered H.
        hessian = ops.take(ops.take(hessian, perm, axis=0), perm, axis=1)
    else:
        perm = inv_perm = None

    # Reference GPTQ/AutoGPTQ inverse-Hessian factorization. Compute `H^-1`
    # from its lower Cholesky factor using triangular solves (no dense inverse),
    # then take the UPPER Cholesky factor of `H^-1`. Indexing this factor inside
    # the loop below reproduces AutoGPTQ's error-propagation exactly.
    cholesky_lower = linalg.cholesky(hessian)
    inverse_hessian = linalg.cholesky_inverse(cholesky_lower)
    inv_hessian = linalg.cholesky(inverse_hessian, upper=True)

    # weights_buffer: [out_features, in_features]
    weights_buffer = weights_transpose
    # Buffer for the final quantized matrix: [out_features, in_features]
    quantized_weights_buffer = ops.zeros_like(weights_transpose, dtype="int32")

    scale_chunks = []
    zero_chunks = []

    # Compute effective group size
    effective_group = in_features if group_size == -1 else group_size

    # Per-group cached params, reused until the column index crosses into
    # the next group. The cache must live across processing blocks: a group
    # can span several blocks (`group_size == -1` covers the whole matrix,
    # and `group_size > blocksize` covers more than one block). Resetting it
    # per block would recompute and re-append the same group's params once
    # per block, corrupting the [out_features, n_groups] scale/zero layout.
    cached_scale = None
    cached_zero = None
    cached_maxq = None
    cached_group_start = -1

    # Process features in blocks
    for block_start in range(0, in_features, blocksize):
        block_end = min(block_start + blocksize, in_features)
        block_size = block_end - block_start

        # Block views
        # block_weights: [out_features, block_size]
        block_weights = weights_buffer[:, block_start:block_end]
        # block_error: [out_features, block_size]
        block_error = ops.zeros_like(block_weights)
        # block_inv_hessian: [block_size, block_size]
        block_inv_hessian = inv_hessian[
            block_start:block_end, block_start:block_end
        ]

        for block_idx in range(block_size):
            # Current global column index, represents the original column
            # in the weight matrix
            global_idx = block_start + block_idx
            # weight_column: [out_features,]
            weight_column = block_weights[:, block_idx]
            # Group-wise parameter reuse (compute once per group)
            if not effective_group == in_features:  # group_size != -1
                # Determine the group start index for the current column
                group_start = (global_idx // effective_group) * effective_group
                if group_start != cached_group_start:
                    # New group encountered, compute & cache params
                    # for this group
                    group_end = min(group_start + effective_group, in_features)
                    group_slice = weights_buffer[:, group_start:group_end]
                    cached_scale, cached_zero, cached_maxq = compute_scale_zero(
                        group_slice
                    )
                    # Store params once per group (in the order encountered).
                    scale_chunks.append(cached_scale)
                    zero_chunks.append(cached_zero)
                    cached_group_start = group_start
                scale, zero, maxq = cached_scale, cached_zero, cached_maxq
            else:
                # Single global group covering all columns.
                if cached_scale is None:
                    cached_scale, cached_zero, cached_maxq = compute_scale_zero(
                        weights_buffer
                    )
                    scale_chunks.append(cached_scale)
                    zero_chunks.append(cached_zero)
                    cached_group_start = 0
                scale, zero, maxq = cached_scale, cached_zero, cached_maxq

            # Quantize column and store it.
            # quantized_column: [out_features, 1]
            quantized_column = quantize_with_zero_point(
                ops.expand_dims(weight_column, 1), scale, zero, maxq
            )

            # Store quantized column in the buffer.
            quantized_weights_buffer = ops.slice_update(
                quantized_weights_buffer,
                (0, global_idx),
                ops.cast(quantized_column, "int32"),
            )
            # Dequantize column to compute error.
            # dequantized_col: [out_features,]
            dequantized_col = dequantize_with_zero_point(
                quantized_column, scale, zero
            )[:, 0]
            # Error feedback for remaining columns within the block
            # block_inv_hessian_diag: scalar
            current_block_influence = block_inv_hessian[block_idx, block_idx]
            # We divide by current_block_influence to get the
            # correct scaling of the error term. Prevent division by zero.
            err = ops.divide_no_nan(
                ops.subtract(weight_column, dequantized_col),
                current_block_influence,
            )
            # Record error for propagation to future blocks
            block_error = ops.slice_update(
                block_error, (0, block_idx), ops.expand_dims(err, 1)
            )

            # Update remaining columns in the current block
            # (those before the current column have already been quantized)
            # Propagate error to remaining columns in the block.
            if block_idx < block_size - 1:
                # update: [out_features, block_size - block_idx - 1]
                update = ops.matmul(
                    ops.expand_dims(err, 1),
                    ops.expand_dims(
                        block_inv_hessian[block_idx, block_idx + 1 :], 0
                    ),
                )
                # tail is a view of the remaining columns in the block
                # to be updated
                # tail: [out_features, block_size - block_idx - 1]
                tail = block_weights[:, block_idx + 1 :]
                block_weights = ops.slice_update(
                    block_weights,
                    (0, block_idx + 1),
                    ops.subtract(tail, update),
                )

        # Propagate block errors to future features (beyond the block)
        if block_end < in_features:
            # Total update for all future columns, based on the
            # accumulated error in this block. This is calculated
            # as the matrix product of the block_error and the
            # relevant slice of the inverse Hessian.
            # total_update: [out_features, in_features - block_end]
            total_update = ops.matmul(
                block_error, inv_hessian[block_start:block_end, block_end:]
            )
            # Update the remaining weights in the buffer. This is done
            # by subtracting the total_update from the remaining columns.
            weights_buffer = ops.concatenate(
                [
                    weights_buffer[:, :block_end],
                    ops.subtract(weights_buffer[:, block_end:], total_update),
                ],
                axis=1,
            )

    # Build group indices for each (possibly permuted) column
    # base_group = effective_group (int)
    base_group = effective_group

    # g_idx in permuted domain. It is integer group metadata, kept as int32.
    g_idx = ops.floor_divide(
        ops.arange(0, in_features, dtype="int32"), base_group
    )

    # Map group indices and quantized weights back to original column order
    if activation_order:
        g_idx = ops.take(g_idx, inv_perm, axis=0)
        quantized_weights_buffer = ops.take(
            quantized_weights_buffer, inv_perm, axis=1
        )

    # Concatenate recorded group params
    if len(scale_chunks) == 0:
        # Edge case: no groups recorded (empty input); fall back to whole matrix
        s, z, _ = compute_scale_zero(weights_transpose)
        scale = s
        zero = z
    else:
        scale = ops.concatenate(scale_chunks, axis=1)
        zero = ops.concatenate(zero_chunks, axis=1)

    return quantized_weights_buffer, scale, zero, g_idx


class GPTQCalibrator(Calibrator):
    """GPTQ calibrator for one layer: the Hessian of its inputs.

    Args:
        layer: A layer with a projection geometry (`Dense`, `EinsumDense`)
            that supports the `gptq` mode.
        config: `GPTQConfig` instance with quantization parameters.
    """

    mode = "gptq"

    # GPTQ's Hessian is close to singular with fewer calibration tokens
    # than this per input feature.
    warn_tokens_per_row = 4

    def __init__(self, layer, config):
        super().__init__(layer, config)
        # `kernel_scale` is stored in the layer's variable dtype.
        self.compute_scale_zero = functools.partial(
            _stored_scale_zero,
            scale_dtype=layer.variable_dtype,
            bits=config.weight_bits,
            symmetric=config.symmetric,
            per_channel=config.per_channel,
            group_size=config.group_size,
        )
        self.hessian = ops.zeros((self.rows, self.rows), dtype="float32")

    @classmethod
    def undersampling_warning(cls, layers):
        """The warning for layers calibrated on too few tokens.

        Args:
            layers: List of `(name, tokens, rows)` for every undersampled
                layer, as tallied by the calibration driver.
        """
        worst = min(tokens / rows for _, tokens, rows in layers)
        examples = ", ".join(
            f"{name} ({tokens} tokens for {rows} input features)"
            for name, tokens, rows in layers[:3]
        )
        return (
            f"GPTQ calibration is undersampled for {len(layers)} layer(s): "
            f"fewer than {cls.warn_tokens_per_row} calibration tokens per "
            f"input feature (worst ratio: {worst:.1f}). With this little "
            "data the Hessian is close to singular and GPTQ's error "
            "correction can overfit the calibration set and produce worse "
            "results than plain round-to-nearest. Increase `num_samples` "
            "and/or "
            "`sequence_length` in `GPTQConfig` (8 or more tokens per input "
            f"feature is recommended). Examples: {examples}."
        )

    def observe(self, inputs):
        """Updates the running mean of the Hessian `2 X^T X / N`."""
        x = self._flatten_inputs(inputs)
        num_new_samples = int(ops.shape(x)[0])
        num_prev_samples = self.num_samples
        total_samples = num_prev_samples + num_new_samples

        # gram_matrix: [features, features]
        gram_matrix = ops.matmul(ops.transpose(x), x)
        # Ensures numerical stability and symmetry in case of large floating
        # point activations.
        gram_matrix = ops.divide(
            ops.add(gram_matrix, ops.transpose(gram_matrix)), 2.0
        )

        # Decay previous mean and add current per-sample contribution
        # (factor 2/N)
        if self.num_samples > 0:
            self.hessian = ops.multiply(
                self.hessian, ops.divide(num_prev_samples, total_samples)
            )

        self.hessian = ops.add(
            self.hessian,
            ops.multiply(ops.divide(2.0, total_samples), gram_matrix),
        )

        self.num_samples = total_samples

    def _solve(self, weights):
        # Dampen the Hessian for Stability
        hessian_diagonal = ops.diagonal(self.hessian)
        dead_diagonal = ops.equal(hessian_diagonal, 0.0)
        hessian_diagonal = ops.where(dead_diagonal, 1.0, hessian_diagonal)
        hessian_matrix = ops.add(
            self.hessian,
            ops.diag(
                ops.where(dead_diagonal, 1.0, ops.zeros_like(hessian_diagonal))
            ),
        )

        # Add dampening factor to the Hessian diagonal
        damping_factor = ops.multiply(
            self.config.hessian_damping, ops.mean(hessian_diagonal)
        )
        hessian_diagonal = ops.add(hessian_diagonal, damping_factor)
        hessian_matrix = ops.add(
            ops.subtract(
                hessian_matrix, ops.diag(ops.diagonal(hessian_matrix))
            ),
            ops.diag(hessian_diagonal),
        )

        # The inverse Hessian used for error correction is derived inside
        # `gptq_quantize_matrix` from the dampened Hessian using a numerically
        # stable Cholesky formulation (triangular solves, no dense inverse).
        codes, scale, zero, g_idx = gptq_quantize_matrix(
            weights,
            hessian=hessian_matrix,
            group_size=self.config.group_size,
            activation_order=self.config.activation_order,
            order_metric=ops.diagonal(hessian_matrix),
            compute_scale_zero=self.compute_scale_zero,
        )
        return codes, scale, zero, g_idx, None
