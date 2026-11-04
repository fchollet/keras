"""AWQ (Activation-aware Weight Quantization) algorithm implementation.

AWQ protects salient weights by finding optimal per-channel scales based on
activation magnitudes, then applies those scales before quantization.

Reference: https://arxiv.org/abs/2306.00978
"""

from keras.src import ops
from keras.src.quantizers.calibrator import Calibrator
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_zero_point

# Maximum number of activation rows stashed per layer for the AutoAWQ-style
# clipping search. Bounds calibration memory; a few hundred rows is enough to
# estimate per-group reconstruction error (matches AutoAWQ's ``n_sample_token``
# default of 512).
MAX_CLIP_SAMPLE_ROWS = 512


def _get_weight_scale(weights, group_size):
    """Per-in-channel weight magnitude used in the AWQ scale formula.

    Mirrors llm-awq's ``get_weight_scale`` (and AutoAWQ's weight term): the
    weights are normalized by their per-group maximum so each group lives on a
    0-1 scale, then averaged over the output channels to obtain a single
    statistic per input channel.

    Args:
        weights: Weight matrix ``[out_features, in_features]``.
        group_size: Quantization group size (``-1`` for per-channel).

    Returns:
        Per-in-channel weight statistic ``[in_features]``.
    """
    weights = ops.cast(weights, "float32")
    out_features, in_features = ops.shape(weights)
    w_abs = ops.abs(weights)
    if group_size and group_size > 0 and in_features % group_size == 0:
        n_groups = in_features // group_size
        w_grouped = ops.reshape(w_abs, (out_features, n_groups, group_size))
        group_max = ops.max(w_grouped, axis=2, keepdims=True)
        w_norm = ops.divide(w_grouped, ops.add(group_max, 1e-6))
        w_norm = ops.reshape(w_norm, (out_features, in_features))
    else:
        group_max = ops.max(w_abs, axis=1, keepdims=True)
        w_norm = ops.divide(w_abs, ops.add(group_max, 1e-6))
    return ops.mean(w_norm, axis=0)


def _fake_quantize_weights(weights_scaled, in_features, group_size):
    """Quantize then dequantize a weight matrix (4-bit asymmetric).

    Shared by the scale search and the clipping search so both evaluate the
    exact same quantizer that is used to produce the final packed weights.

    Args:
        weights_scaled: Weight matrix ``[out_features, in_features]``.
        in_features: Number of input features (columns).
        group_size: Quantization group size (``-1`` for per-channel).

    Returns:
        The dequantized weight matrix, same shape as ``weights_scaled``.
    """
    if group_size == -1:
        scale_q, zero_q, maxq = compute_quantization_parameters(
            weights_scaled,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
            compute_dtype="float32",
        )
        quantized = quantize_with_zero_point(
            weights_scaled, scale_q, zero_q, maxq
        )
        return dequantize_with_zero_point(quantized, scale_q, zero_q)

    scale_q, zero_q, maxq = compute_quantization_parameters(
        weights_scaled,
        bits=4,
        symmetric=False,
        per_channel=True,
        group_size=group_size,
        compute_dtype="float32",
    )
    g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")
    quantized = quantize_with_sz_map(
        weights_scaled, scale_q, zero_q, g_idx, maxq
    )
    return dequantize_with_sz_map(quantized, scale_q, zero_q, g_idx)


def awq_search_optimal_scales(
    weights,
    activation_magnitudes,
    *,
    num_grid_points=20,
    group_size=-1,
):
    """Search for optimal AWQ scales using grid search.

    The AWQ algorithm finds scaling factors that protect salient weights.
    For each channel, we search for an optimal ratio in [0, 1] that minimizes
    the activation-weighted quantization error.

    The key insight: we MULTIPLY weights by scales before quantization to
    expand salient weights. This ensures quantization noise is small relative
    to the expanded weight magnitude. During inference, we divide by scales
    to restore the original magnitude.

    Scale formula (reference llm-awq / AutoAWQ):
        scales = (x_stat**ratio / w_stat**(1 - ratio)).clamp(min=1e-4)
    where ``x_stat`` is the per-channel activation magnitude and ``w_stat`` is
    the per-in-channel weight magnitude from :func:`_get_weight_scale`. Scales
    are then normalized by ``sqrt(max * min)``.
    Loss function: Activation-weighted MSE (approximates output error)

    Args:
        weights: Weight tensor [out_features, in_features] (transposed kernel).
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points. Defaults to 20.
        group_size: Group size for quantization (-1 for per-channel).

    Returns:
        best_scales: Optimal per-channel scales [in_features].
    """
    in_features = ops.shape(weights)[1]

    # Per-channel activation statistic (reference AWQ uses mean(|x|)).
    x_stat = ops.cast(activation_magnitudes, "float32")
    # Avoid zero or very small values.
    x_stat = ops.where(ops.less(x_stat, 1e-8), ops.ones_like(x_stat), x_stat)

    # Per-in-channel weight statistic (llm-awq get_weight_scale). This is the
    # term the previous implementation dropped.
    w_stat = _get_weight_scale(weights, group_size)
    w_stat = ops.where(ops.less(w_stat, 1e-8), ops.ones_like(w_stat), w_stat)

    best_loss = None
    best_scales = ops.ones((in_features,), dtype="float32")

    # Grid search over ratio values from 0 to 1
    for i in range(num_grid_points + 1):
        ratio = i / num_grid_points

        # Reference scale formula: balance activation and weight magnitudes.
        scales = ops.divide(
            ops.power(x_stat, ratio), ops.power(w_stat, 1.0 - ratio)
        )
        scales = ops.maximum(scales, 1e-4)

        # Normalize scales to avoid extreme values
        scale_mean = ops.sqrt(ops.multiply(ops.max(scales), ops.min(scales)))
        scale_mean = ops.maximum(scale_mean, 1e-8)
        scales = ops.divide(scales, scale_mean)

        # Apply scales to weights by MULTIPLYING (expand salient weights)
        # weights_scaled: [out_features, in_features]
        weights_scaled = ops.multiply(weights, scales)

        dequantized = _fake_quantize_weights(
            weights_scaled, in_features, group_size
        )

        # Scale back down by DIVIDING to restore original magnitude
        reconstructed = ops.divide(dequantized, scales)

        # Compute activation-weighted MSE loss
        # This approximates the output error: ||W*X - W_hat*X||^2
        # by weighting each channel's error by x_stat^2
        weight_error = ops.square(ops.subtract(weights, reconstructed))
        # Weight by activation magnitudes squared (broadcast over out_features)
        weighted_error = ops.multiply(weight_error, ops.square(x_stat))
        loss = ops.mean(weighted_error)

        # Track best
        if best_loss is None:
            best_loss = loss
            best_scales = scales
        else:
            is_better = ops.less(loss, best_loss)
            if is_better:
                best_loss = loss
                best_scales = scales

    return best_scales


def awq_search_best_clip(
    weights_scaled,
    activation_sample,
    awq_scales,
    *,
    group_size=-1,
    n_grid=20,
    max_shrink=0.5,
    oc_batch_size=64,
):
    """Search per-group weight clipping bounds (AutoAWQ ``best_clip``).

    After the AWQ scales have been applied, quantization error can be further
    reduced by clipping the weight magnitudes. For each output channel and
    quantization group this grid-searches a shrink factor on the per-group max
    and keeps the value that minimizes the reconstruction error of the layer
    output against a stashed activation sample.

    The reconstruction uses the *scaled* input (``x / awq_scales``) so that it
    matches the effective inference computation ``(x / s) @ (W * s)^T``.

    Args:
        weights_scaled: Scaled weight matrix ``[out_features, in_features]``
            (``W * awq_scales``).
        activation_sample: Raw activation sample ``[rows, in_features]``.
        awq_scales: Per-in-channel AWQ scales ``[in_features]``.
        group_size: Quantization group size (``-1`` for per-channel).
        n_grid: Number of shrink factors to try. Defaults to 20.
        max_shrink: Maximum fractional shrink of the per-group max (the search
            spans ``[1, 1 - max_shrink]``). Defaults to 0.5.
        oc_batch_size: Output-channel batch size, to bound peak memory.

    Returns:
        best_max: Per-group clipping bound ``[out_features, n_group, 1]``.
        gs: Effective group size used for reshaping.
        n_group: Number of groups.
    """
    out_features, in_features = ops.shape(weights_scaled)
    awq_scales = ops.cast(awq_scales, "float32")

    x = ops.cast(activation_sample, "float32")
    if ops.ndim(x) > 2:
        x = ops.reshape(x, (-1, in_features))
    # Effective input to the scaled weights (see docstring).
    x_scaled = ops.divide(x, awq_scales)

    if group_size and group_size > 0 and in_features % group_size == 0:
        gs = group_size
    else:
        # Per-channel, or a group size that does not evenly divide the input:
        # fall back to a single group per output channel for the clip search.
        gs = in_features
    n_group = in_features // gs

    x_grouped = ops.reshape(x_scaled, (-1, n_group, gs))  # [rows, n_group, gs]
    w_grouped = ops.reshape(weights_scaled, (out_features, n_group, gs))
    org_max = ops.max(
        ops.abs(w_grouped), axis=-1, keepdims=True
    )  # [oc, n_group, 1]

    n_shrink = max(1, int(n_grid))
    step = max_shrink / n_grid
    fq_group_size = gs if n_group > 1 else -1

    best_max_parts = []
    n_batches = (out_features + oc_batch_size - 1) // oc_batch_size
    for b in range(n_batches):
        start = b * oc_batch_size
        end = min(start + oc_batch_size, out_features)
        ocb = end - start
        w = w_grouped[start:end]  # [ocb, n_group, gs]
        omax = org_max[start:end]  # [ocb, n_group, 1]

        # Reference output for this output-channel batch: [ocb, rows, n_group].
        org_out = ops.einsum("rng,ong->orn", x_grouped, w)

        best_max = omax
        min_err = None
        for i_s in range(n_shrink):
            max_val = ops.multiply(omax, 1.0 - i_s * step)  # [ocb, n_group, 1]
            w_clamped = ops.clip(w, ops.negative(max_val), max_val)
            w_deq = _fake_quantize_weights(
                ops.reshape(w_clamped, (ocb, in_features)),
                in_features,
                fq_group_size,
            )
            w_deq = ops.reshape(w_deq, (ocb, n_group, gs))
            cur_out = ops.einsum("rng,ong->orn", x_grouped, w_deq)
            err = ops.mean(
                ops.square(ops.subtract(cur_out, org_out)), axis=1
            )  # [ocb, n_group]
            err = ops.expand_dims(err, axis=-1)  # [ocb, n_group, 1]
            if min_err is None:
                min_err = err
                best_max = max_val
            else:
                better = ops.less(err, min_err)
                min_err = ops.where(better, err, min_err)
                best_max = ops.where(better, max_val, best_max)
        best_max_parts.append(best_max)

    best_max = ops.concatenate(best_max_parts, axis=0)  # [oc, n_group, 1]
    return best_max, gs, n_group


def awq_quantize_matrix(
    weights_transpose,
    activation_magnitudes,
    *,
    num_grid_points=20,
    group_size=-1,
    apply_clip=False,
    activation_sample=None,
    clip_n_grid=20,
    clip_max_shrink=0.5,
):
    """Quantize a weight matrix using AWQ.

    This function performs the complete AWQ quantization process:
    1. Find optimal per-channel scales via grid search
    2. Apply scales to weights
    3. (Optional) Search and apply per-group clipping bounds
    4. Compute quantization parameters
    5. Quantize weights

    Args:
        weights_transpose: Weight matrix [out_features, in_features].
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points.
        group_size: Group size for quantization.
        apply_clip: Whether to run the AutoAWQ-style clipping search. Requires
            ``activation_sample`` to be provided.
        activation_sample: Optional raw activation sample [rows, in_features]
            used only for the clipping search.
        clip_n_grid: Number of shrink factors for the clipping search.
        clip_max_shrink: Maximum fractional shrink for the clipping search.

    Returns:
        quantized_weights: Quantized weights [out_features, in_features].
        scales: Quantization scales [out_features, num_groups].
        zeros: Zero points [out_features, num_groups].
        awq_scales: AWQ per-channel scales [in_features].
        g_idx: Group indices [in_features].
    """
    out_features, in_features = ops.shape(weights_transpose)

    # Step 1: Find optimal AWQ scales via grid search
    awq_scales = awq_search_optimal_scales(
        weights_transpose,
        activation_magnitudes,
        num_grid_points=num_grid_points,
        group_size=group_size,
    )

    # Step 2: Apply AWQ scales by MULTIPLYING (expand salient weights)
    # weights_scaled: [out_features, in_features]
    weights_scaled = ops.multiply(weights_transpose, awq_scales)

    # Step 3: (Optional) Search and apply per-group clipping bounds.
    if apply_clip and activation_sample is not None:
        best_max, gs, n_group = awq_search_best_clip(
            weights_scaled,
            activation_sample,
            awq_scales,
            group_size=group_size,
            n_grid=clip_n_grid,
            max_shrink=clip_max_shrink,
        )
        w_grouped = ops.reshape(weights_scaled, (out_features, n_group, gs))
        w_grouped = ops.clip(w_grouped, ops.negative(best_max), best_max)
        weights_scaled = ops.reshape(w_grouped, (out_features, in_features))

    if group_size == -1:
        # Per-channel quantization (no grouping)
        scale_q, zero_q, maxq = compute_quantization_parameters(
            weights_scaled,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
            compute_dtype="float32",
        )

        # Quantize
        quantized = quantize_with_zero_point(
            weights_scaled, scale_q, zero_q, maxq
        )

        # Build group indices (all 0s for per-channel). Integer group
        # metadata, kept as int32.
        g_idx = ops.zeros((in_features,), dtype="int32")
    else:
        # Grouped quantization - use proper per-row grouping
        scale_q, zero_q, maxq = compute_quantization_parameters(
            weights_scaled,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
            compute_dtype="float32",
        )

        # Compute group indices: maps each input feature to its group
        g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")

        # Quantize using group index mapping
        quantized = quantize_with_sz_map(
            weights_scaled, scale_q, zero_q, g_idx, maxq
        )

    return quantized, scale_q, zero_q, awq_scales, g_idx


class AWQCalibrator(Calibrator):
    """AWQ calibrator for one layer: the activation magnitudes of its inputs.

    It accumulates the per-channel mean of `|x|` and, for the clipping
    search, a bounded sample of input rows.

    Args:
        layer: A layer with a projection geometry (`Dense`, `EinsumDense`)
            that supports the `awq` mode.
        config: `AWQConfig` instance with quantization parameters.
    """

    mode = "awq"

    def __init__(self, layer, config):
        super().__init__(layer, config)
        # Running per-channel mean of |x|, as in the reference AWQ
        # implementations.
        self.activation_magnitudes = ops.zeros((self.rows,), dtype="float32")
        # Bounded stash of raw activation rows for the clipping search.
        self._clip_samples = []
        self._clip_sample_rows = 0

    def observe(self, inputs):
        """Updates the running mean of `|x|` and the clipping sample."""
        x = self._flatten_inputs(inputs)
        num_new_samples = int(ops.shape(x)[0])
        total_samples = self.num_samples + num_new_samples

        # Running per-channel mean of |x| via a stable weighted update:
        #   mean <- mean + (batch_mean - mean) * n / (count + n)
        batch_mean = ops.mean(ops.abs(x), axis=0)
        delta = ops.subtract(batch_mean, self.activation_magnitudes)
        self.activation_magnitudes = ops.add(
            self.activation_magnitudes,
            ops.multiply(delta, num_new_samples / total_samples),
        )
        self.num_samples = total_samples

        if (
            self.config.apply_clip
            and self._clip_sample_rows < MAX_CLIP_SAMPLE_ROWS
        ):
            take = min(
                num_new_samples, MAX_CLIP_SAMPLE_ROWS - self._clip_sample_rows
            )
            # A copy: on torch a slice is a view that would keep the whole
            # batch alive.
            self._clip_samples.append(ops.copy(x[:take]))
            self._clip_sample_rows += take

    def _solve(self, weights):
        activation_sample = None
        if self._clip_samples:
            activation_sample = ops.concatenate(self._clip_samples, axis=0)
        codes, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights,
            self.activation_magnitudes,
            num_grid_points=self.config.num_grid_points,
            group_size=self.config.group_size,
            apply_clip=activation_sample is not None,
            activation_sample=activation_sample,
        )
        return codes, scale, zero, g_idx, awq_scales
