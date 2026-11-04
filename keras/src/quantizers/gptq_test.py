import functools
import os
from collections.abc import Callable

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.calibration_run import find_layers_in_block
from keras.src.quantizers.gptq import _stable_permutation
from keras.src.quantizers.gptq import gptq_quantize_matrix
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import tiny_calibration_model
from keras.src.quantizers.quantization_test_utils import token_dataset
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_zero_point
from keras.src.testing.test_utils import named_product

VOCAB_SIZE = 1000
SEQ_LEN = 128
NUM_SAMPLES = 16
W_BITS = 4
NUM_CLASSES = 32

CALIBRATION_TEXT = r"""
GPTQ (Generative Pre-trained Transformer Quantization) is an advanced 
post-training quantization (PTQ) algorithm designed to compress large 
language models with minimal accuracy degradation. It addresses the 
challenge of reducing model size from high-precision formats like 
FP16 to low-bit integers (e.g., INT4, INT3) without the need for
expensive retraining. The algorithm operates on a layer-by-layer basis, 
treating the quantization of each weight matrix $W$ as a 
reconstruction problem. Its objective is to find a quantized weight 
matrix $\hat{W}$ that minimizes the mean squared error of the layer's 
output, formulated as $\arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$, 
where $X$ is a set of calibration inputs. GPTQ's primary innovation 
is its greedy, error-compensating quantization process, based on the 
Optimal Brain Quantizer (OBQ) framework. It quantizes weights one by 
one (or in small groups). After quantizing a single weight $w_q$ to 
its discrete value $\hat{w}_q$, it introduces a quantization error of 
$\delta = w_q - \hat{w}_q$. This error is then immediately compensated 
for by updating all remaining, unquantized weights in the layer. 
The update step is guided by second-order information, specifically 
the inverse of the Hessian matrix ($\mathbf{H}^{-1}$) of the layer's 
reconstruction loss. This inverse Hessian provides a measure of weight 
saliency and inter-dependencies. The update applied to the remaining 
weights is calculated based on $\delta$ and the corresponding entries 
in $\mathbf{H}^{-1}$, effectively propagating the error to less 
sensitive weights. This sequential compensation minimizes the 
cumulative error across the entire layer, allowing GPTQ to maintain 
high model fidelity, as measured by perplexity, even at aggressive 
bit-rates.
"""


def _get_test_layer(layer_type, kernel_shape):
    if layer_type == "Dense":
        layer = layers.Dense(units=kernel_shape[1])
        layer.build(input_shape=(None, kernel_shape[0]))
    elif layer_type == "EinsumDense":
        output_shape = (kernel_shape[1], kernel_shape[2])
        layer = layers.EinsumDense(
            equation="...h,hio->...io", output_shape=output_shape
        )
        layer.build(input_shape=(None, kernel_shape[0]))
    else:
        layer = layers.Layer()
    return layer


def _hessian_calibrator(layer):
    """A GPTQ calibrator of `layer` with the default config, unsolved."""
    return calibrate_layer(layer, calibration_config("gptq"), solve=False)


def _scale_zero_fn(config, compute_dtype="float32"):
    """The scale and zero rule `GPTQ` binds for a layer of `compute_dtype`."""
    return functools.partial(
        compute_quantization_parameters,
        bits=config.weight_bits,
        symmetric=config.symmetric,
        per_channel=config.per_channel,
        group_size=config.group_size,
        compute_dtype=compute_dtype,
    )


@pytest.mark.requires_trainable_backend
class GPTQTest(testing.TestCase):
    def test_initialization_with_dense_layer(self):
        mock_layer = _get_test_layer("Dense", kernel_shape=(64, 128))

        calibrator = _hessian_calibrator(mock_layer)
        self.assertEqual(calibrator.rows, 64)
        self.assertEqual(calibrator.columns, 128)
        self.assertEqual(calibrator.hessian.shape, (64, 64))

    def test_initialization_with_einsumdense_3d(self):
        mock_layer = _get_test_layer("EinsumDense", kernel_shape=(64, 4, 32))
        calibrator = _hessian_calibrator(mock_layer)
        self.assertEqual(calibrator.rows, 64)
        self.assertEqual(calibrator.columns, 4 * 32)
        self.assertEqual(calibrator.hessian.shape, (64, 64))

    def test_update_hessian(self):
        dense = _get_test_layer("Dense", kernel_shape=(16, 32))
        calibrator = _hessian_calibrator(dense)

        rng = np.random.default_rng(seed=42)
        batch1 = rng.standard_normal(size=(8, 16)).astype("float32")

        calibrator.observe(batch1)
        self.assertEqual(calibrator.num_samples, 8)
        H1 = calibrator.hessian

        batch2 = rng.standard_normal(size=(4, 16)).astype("float32")

        calibrator.observe(batch2)
        self.assertEqual(calibrator.num_samples, 12)

        H2 = calibrator.hessian

        self.assertNotAllClose(H1, H2)

    def test_gptq_on_single_layer(self):
        rng = np.random.default_rng(seed=42)
        dense = _get_test_layer("Dense", kernel_shape=(16, 32))

        config = GPTQConfig(
            dataset=None,
            tokenizer=None,
            weight_bits=4,
            symmetric=False,
            group_size=-1,
        )

        calibration_data = rng.standard_normal(size=(128, 16)).astype("float32")
        calibrate_layer(dense, config, calibration_data)

        self.assertEqual(standardize_dtype(dense.kernel.dtype), "uint8")

    def _calibrate_gptq_dense(self, kernel_shape, weight_bits, group_size):
        rng = np.random.default_rng(seed=7)
        dense = _get_test_layer("Dense", kernel_shape=kernel_shape)
        config = GPTQConfig(
            dataset=None,
            tokenizer=None,
            weight_bits=weight_bits,
            symmetric=False,
            group_size=group_size,
        )
        calibrate_layer(
            dense,
            config,
            rng.standard_normal((128, kernel_shape[0])).astype("float32"),
        )
        return dense

    def test_gptq_2bit_packing_end_to_end(self):
        """2-bit GPTQ packs four values per byte and round-trips through
        `load_own_variables`."""
        in_dim, out_dim, group_size = 64, 32, 32
        dense = self._calibrate_gptq_dense((in_dim, out_dim), 2, group_size)

        # The quantized kernel keeps the kernel's `[in, out]` orientation and
        # packs four 2-bit values per byte along the output axis: shape
        # (in, ceil(out/4)), dtype uint8.
        packed_cols = (out_dim + 3) // 4
        self.assertEqual(
            tuple(dense.quantized_kernel.shape), (in_dim, packed_cols)
        )
        self.assertEqual(
            standardize_dtype(dense.quantized_kernel.dtype), "uint8"
        )
        self.assertEqual(standardize_dtype(dense.g_idx.dtype), "float32")

        rng = np.random.default_rng(seed=123)
        x = rng.standard_normal((4, in_dim)).astype("float32")
        y_ref = ops.convert_to_numpy(dense(x))
        self.assertTrue(np.isfinite(y_ref).all())

        # Storage: the packed kernel is a quarter of the unpacked byte count.
        packed_bytes = int(np.prod(dense.quantized_kernel.shape))
        unpacked_bytes = out_dim * in_dim
        self.assertEqual(packed_bytes * 4, unpacked_bytes)

        # Rebuild the serialized store (gptq spec order) and reload it into a
        # fresh layer. `save_own_variables` is not used because the GPTQ save
        # path has a separate, pre-existing limitation around kernel_zero.
        store = {
            "0": ops.convert_to_numpy(dense.bias),
            "1": ops.convert_to_numpy(dense.quantized_kernel),
            "2": ops.convert_to_numpy(dense.kernel_scale),
            "3": ops.convert_to_numpy(dense.kernel_zero),
            "4": ops.convert_to_numpy(dense.g_idx),
        }
        reloaded = layers.Dense(
            units=out_dim, dtype=f"gptq/2/{group_size}_from_float32"
        )
        reloaded.build((None, in_dim))
        reloaded.load_own_variables(store)
        self.assertEqual(
            tuple(reloaded.quantized_kernel.shape), (in_dim, packed_cols)
        )
        self.assertAllClose(reloaded(x), y_ref)

    def test_gptq_2bit_storage_reduction_256(self):
        """A 256x256 kernel packs from 65536 to 16384 bytes at 2-bit."""
        dense = self._calibrate_gptq_dense((256, 256), 2, 128)
        self.assertEqual(tuple(dense.quantized_kernel.shape), (256, 64))
        packed_bytes = int(np.prod(dense.quantized_kernel.shape))
        self.assertEqual(packed_bytes, 16384)
        self.assertEqual(256 * 256, 65536)  # unpacked one value per byte

    def test_codes_are_solved_against_the_stored_scale(self):
        """Under a `bfloat16` policy, the codes match the stored scale.

        `kernel_scale` is stored in `bfloat16`. With an identity Hessian,
        GPTQ rounds each weight to nearest, so the stored codes must be the
        weights rounded on the stored scale, not on the float32 scale the
        parameters were computed in.
        """
        rng = np.random.default_rng(seed=3)
        layer = layers.Dense(16, use_bias=False, dtype="bfloat16")
        layer.build((None, 32))
        layer.set_weights([rng.standard_normal((32, 16)).astype("float32")])
        weights = ops.cast(layer.kernel, "float32")
        config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=8, group_size=-1
        )
        layer.quantize("gptq", config=config)
        calibrator = calibrate_layer(layer, config, solve=False)
        calibrator.hessian = ops.eye(32, dtype="float32")
        calibrator.quantize()

        expected = quantize_with_zero_point(
            weights,
            ops.cast(layer.kernel_scale, "float32"),
            layer.kernel_zero,
            255.0,
        )
        self.assertAllEqual(layer.quantized_kernel, ops.cast(expected, "uint8"))

    def test_initialization_errors(self):
        # A 4-D einsum kernel has no 2-D calibration view.
        four_d = layers.EinsumDense(
            "abc,cdef->abdef", output_shape=(3, 2, 3, 2)
        )
        four_d.build((None, 3, 4))
        with self.assertRaisesRegex(ValueError, "only supports 2D or 3D"):
            _hessian_calibrator(four_d)
        # An unbuilt layer is refused by the check `Layer.quantize` runs.
        with self.assertRaisesRegex(ValueError, "did not set kernel_shape"):
            _hessian_calibrator(layers.Dense(4))

    def test_update_hessian_invalid_input(self):
        rng = np.random.default_rng(seed=42)
        dense = _get_test_layer("Dense", kernel_shape=(16, 32))
        calibrator = _hessian_calibrator(dense)
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            calibrator.observe(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            calibrator.observe(np.empty((0, 16)))
        with self.assertRaisesRegex(ValueError, "match input features"):
            bad_input = rng.standard_normal(size=(8, 99))
            calibrator.observe(bad_input)

    def test_streaming_equals_big_batch(self):
        """Tests that streaming updates match big batch updates."""
        # dummy inputs
        x = ops.array(np.random.randn(100, 7), "float32")

        # One-shot hessian update
        layer_1 = layers.Dense(5, use_bias=False)
        layer_1.build(input_shape=(None, 7))

        g1 = _hessian_calibrator(layer_1)
        g1.observe(x)

        # Streamed hessian update
        layer_2 = layers.Dense(5, use_bias=False)
        layer_2.build(input_shape=(None, 7))
        g2 = _hessian_calibrator(layer_2)
        g2.observe(x[:50])
        g2.observe(x[50:])

        # Both the one-shot and streamed hessian updates should match
        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_hessian_matches_closed_form(self):
        """Tests that the Hessian matches the closed-form solution."""
        x = ops.array(np.random.randn(128, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = _hessian_calibrator(layer)
        g.observe(x)

        expected = ops.multiply(
            ops.divide(2.0, x.shape[0]), ops.matmul(ops.transpose(x), x)
        )
        self.assertAllClose(g.hessian, expected, rtol=1e-6, atol=1e-6)

    def test_higher_rank_inputs_are_reshaped(self):
        """Tests that higher-rank inputs are reshaped correctly."""
        # x: [batch, time, feat]
        x = ops.array(np.random.randn(10, 4, 7), "float32")
        x_flat = ops.reshape(x, (-1, ops.shape(x)[-1]))

        layer1 = layers.Dense(5, use_bias=False)
        layer1.build((None, 7))
        g1 = _hessian_calibrator(layer1)
        g1.observe(x)

        layer2 = layers.Dense(5, use_bias=False)
        layer2.build((None, 7))
        g2 = _hessian_calibrator(layer2)
        g2.observe(x_flat)

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_raises_on_feature_mismatch(self):
        x = ops.array(np.random.randn(8, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 6))  # wrong in_features
        g = _hessian_calibrator(layer)

        with self.assertRaisesRegex(ValueError, "do not match input features"):
            g.observe(x)

        with self.assertRaisesRegex(ValueError, "cannot be None"):
            g.observe(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            g.observe(ops.array(np.empty((0, 7), dtype="float32")))

    def test_num_samples_accumulates_correctly(self):
        """Tests that the number of samples is accumulated correctly when
        streaming updates are used."""
        x = ops.array(np.random.randn(64, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = _hessian_calibrator(layer)

        g.observe(x[:5])
        g.observe(x[5:30])
        g.observe(x[30:])

        self.assertEqual(g.num_samples, 64)

    def test_numeric_stability_large_values(self):
        """Tests numeric stability of hessian update with large input values."""
        x = ops.multiply(ops.array(np.random.randn(32, 7), "float32"), 1e6)
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))

        g = _hessian_calibrator(layer)
        g.observe(x)

        # Should be finite and symmetric
        self.assertTrue(ops.all(ops.isfinite(g.hessian)))
        self.assertTrue(ops.all(ops.equal(g.hessian, ops.transpose(g.hessian))))

    def test_einsumdense_2d_kernel_hessian_shape(self):
        x = layers.Input((7,))
        y = layers.EinsumDense("ab,bc->ac", output_shape=(5,))(x)
        model = keras.Model(x, y)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        g = _hessian_calibrator(einsum_dense_layer)

        # should infer rows==7
        self.assertEqual(ops.shape(g.hessian), (7, 7))

    def test_einsumdense_3d_kernel_streaming_equals_big_batch(self):
        """Tests that streaming updates to the Hessian are equivalent to a big
        batch update."""
        # Construct a tiny attention-like einsum with 3D kernel
        x = layers.Input((7,))
        qkv = layers.EinsumDense("bf,fhk->bhk", output_shape=(2, 3))(
            x
        )  # heads=2, head_dim=3
        model = keras.Model(x, qkv)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        x = ops.array(np.random.randn(50, 7), "float32")

        g1 = _hessian_calibrator(einsum_dense_layer)
        g1.observe(x)

        g2 = _hessian_calibrator(einsum_dense_layer)
        g2.observe(x[:20])
        g2.observe(x[20:])

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_identity_hessian_matches_direct_quantization(self):
        """Tests that the matrix quantization without error correction
        matches the direct implementation."""
        in_features, out_features = 16, 8
        weights = ops.reshape(
            ops.linspace(
                -0.9, 1.1, in_features * out_features, dtype="float32"
            ),
            (in_features, out_features),
        )
        weights_transpose = ops.transpose(weights)

        # hessian = identity => inverse Hessian is identity; no cross-feature
        # correction (since all off-diagonal elements are zero), which means
        # there is no interaction between different features.
        hessian = ops.eye(in_features, dtype="float32")

        quantized_weights, scale_map, zero_map, g_idx = gptq_quantize_matrix(
            weights_transpose,
            hessian,
            blocksize=128,
            group_size=1,  # per-column quantization
            activation_order=False,
            compute_scale_zero=_compute_scale_zero,
        )

        dequantized_weights = dequantize_with_sz_map(
            quantized_weights, scale_map, zero_map, g_idx
        )

        # Compare function output with columnwise direct application
        # of quantization.
        out = ops.zeros_like(weights_transpose)
        for j in range(ops.shape(weights_transpose)[1]):
            column = weights_transpose[:, j : j + 1]
            scale, zero, maxq = _compute_scale_zero(column)
            quantized_col = quantize_with_zero_point(column, scale, zero, maxq)
            dequantized = dequantize_with_zero_point(quantized_col, scale, zero)
            out = ops.slice_update(
                out, (0, j), ops.expand_dims(dequantized[:, 0], 1)
            )

        self.assertAllClose(dequantized_weights, out, atol=1e-6)

    def test_activation_order_produces_equivalent_weights(self):
        """
        Tests that quantizing with `activation_order=True` yields the same
        final weights as `activation_order=False`, because the internal
        permutation should be undone.
        """
        # Set up shared inputs and a non-trivial permutation.
        in_features, out_features = 8, 6
        initial_weights = ops.array(
            np.random.randn(in_features, out_features), "float32"
        )

        # Generate a Hessian that creates a non-trivial permutation.
        hessian_diag = random.shuffle(
            ops.linspace(10.0, 1.0, in_features, dtype="float32")
        )
        hessian_matrix = ops.diag(hessian_diag)

        # Sanity check: ensure the permutation is not the identity.
        perm = _stable_permutation(hessian_diag)
        self.assertFalse(ops.all(ops.equal(perm, ops.arange(in_features))))

        def create_and_quantize(use_activation_order):
            layer = layers.Dense(out_features, use_bias=False)
            layer.build((None, in_features))
            layer.set_weights([ops.copy(initial_weights)])

            config = GPTQConfig(
                dataset=None,
                tokenizer=None,
                group_size=-1,
                activation_order=use_activation_order,
            )
            layer.quantize("gptq", config=config)

            calibrator = calibrate_layer(layer, config, solve=False)
            calibrator.hessian = hessian_matrix
            calibrator.quantize()
            return layer

        # Quantize two layers, one with and one without activation ordering.
        ordered_layer = create_and_quantize(use_activation_order=True)
        unordered_layer = create_and_quantize(use_activation_order=False)

        self.assertAllClose(
            ordered_layer.get_weights()[0],
            unordered_layer.get_weights()[0],
            msg="Weights should be identical as the permutation is undone.",
        )

    def test_non_positive_definite_hessian_raises(self):
        """A non-positive-definite Hessian is rejected with a clear error.

        `gptq_quantize_matrix` takes an already dampened Hessian, and
        `GPTQ.quantize` guarantees positive definiteness by adding
        `hessian_damping * mean(diag(H))` to the diagonal before calling.
        A zero or negative diagonal entry breaks that contract, and the
        Cholesky factorization must surface it as a `ValueError` on every
        backend rather than silently propagating NaNs.
        """
        out_features, in_features = 4, 4
        weights = ops.ones((out_features, in_features), dtype="float32")
        config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=4, group_size=-1
        )
        compute_scale_zero = _scale_zero_fn(config)

        for bad_diagonal in (0.0, -1.0):
            hessian = np.eye(in_features, dtype=np.float32)
            hessian[2, 2] = bad_diagonal
            with self.assertRaisesRegex(ValueError, "Cholesky"):
                gptq_quantize_matrix(
                    weights,
                    ops.convert_to_tensor(hessian),
                    blocksize=2,
                    group_size=-1,
                    compute_scale_zero=compute_scale_zero,
                )

    def test_ill_conditioned_hessian_produces_finite_weights(self):
        """Severe ill-conditioning must not produce NaNs or infinities.

        The per-column error is divided by `inv_hessian[j, j]`, the diagonal
        of the upper Cholesky factor of `H^-1`, which satisfies
        `inv_hessian[j, j] ** 2 = det(H[j+1:, j+1:]) / det(H[j:, j:])`. That
        is the reciprocal of the trailing Schur pivot of `H` at `j`, which is
        bounded above by `H[j, j]`. For any float32 positive-definite `H` the
        divisor is therefore at least `1 / sqrt(float32 max)`, about `5e-20`,
        so it never underflows to zero.

        The Hessian below is the reachable extreme: one feature is weighted by
        a power of two, which is exact in binary floating point, so `H` stays
        exactly positive definite (a congruence of a well-conditioned positive
        definite matrix) while `inv_hessian[2, 2]` falls to roughly `6e-19`.
        The off-diagonal entries keep the error-propagation path active.
        """
        base = np.array(
            [
                [4.0, 1.0, 1.0, 0.5],
                [1.0, 3.0, 0.5, 1.0],
                [1.0, 0.5, 2.0, 0.5],
                [0.5, 1.0, 0.5, 3.0],
            ],
            dtype=np.float32,
        )
        feature_scale = np.array([1.0, 1.0, 2.0**60, 1.0], dtype=np.float32)
        hessian = base * feature_scale[:, None] * feature_scale[None, :]

        rng = np.random.default_rng(seed=42)
        weights = ops.convert_to_tensor(
            rng.normal(size=(6, 4)).astype("float32")
        )
        config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=W_BITS, group_size=-1
        )
        compute_scale_zero = _scale_zero_fn(config)

        # blocksize=2 puts the ill-conditioned feature at the start of the
        # second block, so the cross-block error propagation is covered too.
        for blocksize in (2, 4):
            quantized, scale, zero, _ = gptq_quantize_matrix(
                weights,
                ops.convert_to_tensor(hessian),
                blocksize=blocksize,
                group_size=-1,
                compute_scale_zero=compute_scale_zero,
            )
            for name, tensor in (
                ("quantized", quantized),
                ("scale", scale),
                ("zero", zero),
            ):
                values = ops.convert_to_numpy(tensor)
                self.assertTrue(
                    np.isfinite(values).all(),
                    msg=f"{name} is not finite for blocksize={blocksize}.",
                )
            quantized_values = ops.convert_to_numpy(quantized)
            self.assertGreaterEqual(quantized_values.min(), 0)
            self.assertLessEqual(quantized_values.max(), 2**W_BITS - 1)

    def test_find_layers_in_block_includes_layers_with_sub_layers(self):
        """`Dense`/`EinsumDense` are collected even when they own sub-layers.

        A `Dense` whose activation is a `Layer` owns that `Layer`, so a leaf
        filter would wrongly hide it from calibration. `find_layers_in_block`
        must return both such a `Dense` and a plain `Dense`.
        """
        block = models.Sequential(
            [
                layers.Dense(8, activation=layers.ReLU()),
                layers.Dense(8),
            ]
        )
        block.build((None, 8))

        found = find_layers_in_block(block)

        self.assertEqual(len(found), 2)
        for dense in block.layers:
            self.assertIn(dense.path, found)
            self.assertIs(found[dense.path], dense)

    @parameterized.named_parameters(
        ("per_channel", -1, 1),
        ("group_gt_blocksize", 256, 2),
        ("group_eq_blocksize", 128, 4),
        ("group_lt_blocksize", 64, 8),
    )
    def test_gptq_quantize_matrix_group_param_shapes(
        self, group_size, expected_groups
    ):
        """Scale/zero must have one column per quantization group.

        Regression test: the per-group parameter cache was reset at every
        processing block, so a group spanning several blocks (`group_size`
        of -1, or larger than `blocksize`) had its params recomputed and
        re-appended once per block, producing `[out, n_blocks]`-shaped
        scales that could not be assigned to the layer's
        `[out, n_groups]` variables.
        """
        rng = np.random.default_rng(0)
        in_features, out_features = 512, 16
        w = ops.convert_to_tensor(
            rng.standard_normal((out_features, in_features)).astype("float32")
        )
        x = rng.standard_normal((2048, in_features)).astype("float32")
        hessian = ops.convert_to_tensor(
            (2.0 / 2048) * (x.T @ x)
            + 0.01 * np.eye(in_features, dtype="float32")
        )
        config = GPTQConfig(
            tokenizer=None, dataset=None, weight_bits=4, group_size=group_size
        )
        _, scale, zero, g_idx = gptq_quantize_matrix(
            w,
            hessian=hessian,
            blocksize=128,
            group_size=group_size,
            compute_scale_zero=_scale_zero_fn(config),
        )
        self.assertEqual(tuple(scale.shape), (out_features, expected_groups))
        self.assertEqual(tuple(zero.shape), (out_features, expected_groups))
        self.assertEqual(
            int(ops.convert_to_numpy(ops.max(g_idx))), expected_groups - 1
        )

    def test_gptq_model_quantize_per_channel_group_size(self):
        """`model.quantize("gptq")` with `group_size=-1` must not crash.

        Regression test: with per-channel quantization the layer builds
        `[out, 1]` scale variables, but the solver used to emit one scale
        chunk per 128-column processing block, crashing the assignment for
        any layer with more than 128 input features.
        """
        vocab_size, seq_len, embed_dim = 64, 8, 256
        model, structure = tiny_calibration_model(
            [layers.Dense(4)],
            vocab_size=vocab_size,
            sequence_length=seq_len,
            embed_dim=embed_dim,
            head_units=2,
        )
        rng = np.random.default_rng(seed=5)
        config = calibration_config(
            "gptq",
            dataset=token_dataset(2, seq_len, vocab_size, rng),
            tokenizer=lambda text: text,
            weight_bits=4,
            num_samples=2,
            sequence_length=seq_len,
            group_size=-1,
            quantization_layer_structure=structure,
        )
        model.quantize("gptq", config=config)

        dense = structure["sequential_blocks"][0].layers[0]
        self.assertFalse(dense.calibration_pending)
        self.assertEqual(tuple(dense.kernel_scale.shape), (1, 4))
        self.assertEqual(tuple(dense.kernel_zero.shape), (1, 4))

    def test_gptq_warns_on_undersampled_calibration(self):
        """Fewer than 4 calibration tokens per input feature must warn.

        With a near-singular Hessian, GPTQ's error correction overfits the
        calibration set and can produce worse results than plain
        round-to-nearest; the user should be told to increase
        `num_samples`/`sequence_length`.
        """
        vocab_size, seq_len, embed_dim = 64, 8, 256
        model, structure = tiny_calibration_model(
            [layers.Dense(4)],
            vocab_size=vocab_size,
            sequence_length=seq_len,
            embed_dim=embed_dim,
            head_units=2,
        )
        rng = np.random.default_rng(seed=5)
        config = calibration_config(
            "gptq",
            dataset=token_dataset(2, seq_len, vocab_size, rng),
            tokenizer=lambda text: text,
            weight_bits=4,
            num_samples=2,
            sequence_length=seq_len,
            group_size=-1,
            quantization_layer_structure=structure,
        )
        # 2 samples x 8 tokens = 16 tokens for a 256-feature layer.
        with self.assertWarnsRegex(UserWarning, "undersampled"):
            model.quantize("gptq", config=config)


def _compute_scale_zero(x, **_):
    # Per-column asymmetric int4 example
    # scale = (max-min)/maxq, zero = round(-min/scale)
    maxq = 15.0
    xmin = ops.min(x, axis=0, keepdims=True)
    xmax = ops.max(x, axis=0, keepdims=True)
    scale = ops.divide(ops.subtract(xmax, xmin), ops.add(maxq, 1e-8))
    zero = ops.round(ops.divide(ops.negative(xmin), ops.add(scale, 1e-8)))
    return scale, zero, maxq


def _get_sequence_classifier():
    """Transformer-based sequence classifier

    tokens -> Embedding -> Transformer -> GAP -> Dense(num_classes).
    """
    embed_dim = 32
    num_heads = 4
    ff_dim = 32

    class SimpleTransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
            super().__init__(**kwargs)

            self.att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.ffn = models.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            attention_output = self.att(inputs, inputs)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            return self.layernorm2(out1 + ffn_output)

    inputs = layers.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, embed_dim)(inputs)
    x = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return models.Model(inputs, outputs)


def _get_simple_model():
    return models.Sequential([layers.Dense(10, input_shape=(5,))])


def _mean_kl(p, q):
    # Add small epsilon for numerical stability
    eps = 1e-8
    p = ops.clip(p, eps, 1.0)
    q = ops.clip(q, eps, 1.0)
    # Compute KL divergence
    # D_KL(P || Q) = sum(P * log(P / Q))
    return ops.mean(
        ops.sum(ops.multiply(p, ops.subtract(ops.log(p), ops.log(q))), axis=-1)
    )


def _top1_match_rate(a_logits, b_logits):
    """Calculates the top-1 match rate between two sets of logits.

    Formula: T = 1/N * sum(1{argmax(a_i) == argmax(b_i)})
    """
    return ops.mean(
        ops.equal(ops.argmax(a_logits, axis=-1), ops.argmax(b_logits, axis=-1))
    )


DATASETS = {
    "string_dataset": lambda: _string_dataset(
        CALIBRATION_TEXT, NUM_SAMPLES, SEQ_LEN
    ),
    "token_dataset": lambda: _token_dataset(NUM_SAMPLES, SEQ_LEN),
}

CONFIGS = {
    "default": {},
    "per_channel": {"group_size": -1, "per_channel": True},
    "act_order": {"activation_order": True},
    "symmetric": {"symmetric": True},
    "group_wise": {"group_size": 8},
    "group_wise_act_order": {"group_size": 8, "activation_order": True},
    "symmetric_act_order": {"symmetric": True, "activation_order": True},
    "symmetric_per_channel": {"symmetric": True, "per_channel": True},
    "group_wise_symmetric_8bit": {
        "group_size": 8,
        "symmetric": True,
        "weight_bits": 8,
    },
}


def _pad_or_trim_1d(ids, length):
    """Pads or trims a 1D array to a specified length."""
    ids = ops.ravel(ops.array(ids, "int64"))
    if len(ids) < length:
        ids = ops.concatenate(
            [ids, ops.zeros(length - len(ids), dtype=ids.dtype)]
        )
    else:
        ids = ids[:length]
    return ids


def _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Tokenizes strings to char-IDs or passes through int arrays;
    outputs shape (1, seq_len)."""

    def _tok(x):
        if isinstance(x, str):
            ids = ops.convert_to_tensor(
                np.fromiter((ord(c) % vocab_size for c in x), dtype=np.int64)
            )
        else:
            ids = np.asarray(x, dtype=np.int64)
        ids = _pad_or_trim_1d(ids, seq_len)
        return ids[None, :]

    _tok.tokenize = _tok
    return _tok


def _string_dataset(
    long_text, num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN
):
    """Yields string slices"""
    rng = np.random.default_rng(seed=0)
    L = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = rng.integers(0, L) if L > 1 else 0
        yield long_text[start : start + sequence_length]


def _token_dataset(
    num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN, vocab_size=VOCAB_SIZE
):
    """Yields tokenized samples."""
    rng = np.random.default_rng(seed=0)
    for _ in range(num_samples):
        yield rng.integers(
            low=0, high=vocab_size, size=(1, sequence_length), dtype=np.int64
        )


@pytest.mark.requires_trainable_backend
@pytest.mark.skipif(
    backend.backend() == "torch",
    reason="torch gives low accuracy on CI, but works well locally",
)
class TestModelQuantization(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": dataset_id, "dataset": dataset}
                for dataset_id, dataset in DATASETS.items()
            ],
            [
                {"testcase_name": config_id, "config": config}
                for config_id, config in CONFIGS.items()
            ],
        )
    )
    def test_quantize_gptq_combinations(self, dataset, config):
        """Tests GPTQ quantization on a tiny transformer classifier.

        Validates classification performance of the quantized model
        with respect to the full-precision baseline.
        """
        rng = np.random.default_rng(seed=321)
        keras.utils.set_random_seed(123)

        # Build the calibration set.
        calibration_set = list(
            dataset() if isinstance(dataset, Callable) else dataset
        )
        self.assertNotEmpty(calibration_set)

        # Build classifier and tokenizer
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build an eval batch drawn from the SAME distribution as calibration
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[rng.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Baseline logits
        y_ref = model.predict(x_eval)

        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]

        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        base_cfg = dict(
            dataset=calibration_set,
            tokenizer=tokenizer,
            weight_bits=W_BITS,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=32,
            symmetric=False,
            activation_order=False,
            quantization_layer_structure=layer_structure,
        )
        gptq_cfg = GPTQConfig(**{**base_cfg, **config})

        # Quantize
        model.quantize("gptq", config=gptq_cfg)

        # Post-quant logits
        y_q = model.predict(x_eval)

        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref, p_q = ops.softmax(y_ref), ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        self.assertGreaterEqual(
            top1_match, 0.5, f"Top-1 agreement too low: {top1_match:.3f}"
        )
        self.assertLessEqual(kl, 0.30, f"KL divergence too high: {kl:.3f}")

    @parameterized.named_parameters(
        {
            "testcase_name": "gptq_with_invalid_config_type",
            "mode": "gptq",
            "config": {"weight_bits": 4},
            "expected_exception": ValueError,
            "error_msg": "Argument `config` must be an instance of "
            "`QuantizationConfig`",
        },
        {
            "testcase_name": "gptq_with_none_config",
            "mode": "gptq",
            "config": None,
            "expected_exception": ValueError,
            "error_msg": "For GPTQ, the `config` argument must be of "
            "type `GPTQConfig`.",
        },
        {
            "testcase_name": "gptq_with_base_quantization_config",
            "mode": "gptq",
            "config": QuantizationConfig(),
            "expected_exception": NotImplementedError,
            "error_msg": "Do not instantiate QuantizationConfig directly.",
        },
        {
            "testcase_name": "gptq_missing_structure",
            "mode": "gptq",
            "config": GPTQConfig(dataset=["a"], tokenizer=lambda x: x),
            "expected_exception": ValueError,
            "error_msg": "For mode='gptq', a valid quantization structure",
        },
    )
    def test_quantize_scenarios(
        self, mode, config, expected_exception, error_msg
    ):
        model = _get_simple_model()
        with self.assertRaisesRegex(expected_exception, error_msg):
            model.quantize(mode, config=config)

    def test_gptq_filtering(self):
        """Tests that filters argument works for GPTQ."""
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Structure
        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]
        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        config = GPTQConfig(
            dataset=[np.zeros((1, SEQ_LEN), dtype="int32")],
            tokenizer=tokenizer,
            quantization_layer_structure=layer_structure,
            weight_bits=4,
            group_size=32,
        )

        target_layer = transformer_block.ffn.layers[0]

        def filter_fn(layer):
            return layer.name != target_layer.name

        model.quantize("gptq", config=config, filters=filter_fn)

        # Check that target_layer is NOT quantized.
        self.assertIsNone(getattr(target_layer, "quantization_mode", None))
        self.assertFalse(hasattr(target_layer, "quantized_kernel"))

        # Check that other dense layers ARE quantized.
        other_dense = transformer_block.ffn.layers[1]
        self.assertEqual(
            getattr(other_dense, "quantization_mode", None), "gptq"
        )
        self.assertTrue(hasattr(other_dense, "quantized_kernel"))

    def test_gptq_multi_filtering(self):
        """Tests that list of regex filters works for GPTQ."""
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]
        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        config = GPTQConfig(
            dataset=[np.zeros((1, SEQ_LEN), dtype="int32")],
            tokenizer=tokenizer,
            quantization_layer_structure=layer_structure,
            weight_bits=4,
            group_size=32,
        )

        layer0 = transformer_block.ffn.layers[0]
        layer1 = transformer_block.ffn.layers[1]

        # We want to quantize only layer0.
        filters = [f"^{layer0.name}$"]

        model.quantize("gptq", config=config, filters=filters)

        # Check that layer0 is quantized.
        self.assertEqual(getattr(layer0, "quantization_mode", None), "gptq")
        self.assertTrue(hasattr(layer0, "quantized_kernel"))

        # Check that layer1 is not quantized.
        self.assertIsNone(getattr(layer1, "quantization_mode", None))
        self.assertFalse(hasattr(layer1, "quantized_kernel"))

    def test_gptq_save_load_round_trip(self):
        """Full GPTQ quantize -> save -> load round trip.

        Only the Dense layers inside the structure's ``sequential_blocks``
        are quantized; the embedding and classifier head stay untouched,
        and predictions are reproduced after a save/load cycle.
        """
        keras.utils.set_random_seed(123)
        embed_dim = 8
        model, structure = tiny_calibration_model(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(embed_dim),
            ],
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=embed_dim,
            head_units=NUM_CLASSES,
        )
        (embedding,) = structure["pre_block_layers"]
        (block,) = structure["sequential_blocks"]
        head = model.layers[-1]

        rng = np.random.default_rng(seed=7)
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
        config = calibration_config(
            "gptq",
            dataset=token_dataset(4, SEQ_LEN, VOCAB_SIZE, rng),
            tokenizer=tokenizer,
            weight_bits=4,
            group_size=8,
            num_samples=4,
            sequence_length=SEQ_LEN,
            quantization_layer_structure=structure,
        )

        model.quantize("gptq", config=config)

        # In-structure Dense layers are quantized and calibrated.
        for dense in block.layers:
            self.assertEqual(dense.quantization_mode, "gptq")
            self.assertFalse(dense.calibration_pending)

        # Out-of-structure layers must stay completely untouched.
        self.assertIsNone(getattr(head, "quantization_mode", None))
        self.assertFalse(hasattr(head, "quantized_kernel"))
        self.assertIsNone(getattr(embedding, "quantization_mode", None))
        self.assertFalse(hasattr(embedding, "quantized_kernel"))

        # Predictions survive a save/load round trip.
        eval_rng = np.random.default_rng(seed=99)
        x_eval = eval_rng.integers(
            0, VOCAB_SIZE, size=(4, SEQ_LEN), dtype=np.int32
        )
        y_before = model.predict(x_eval)

        path = os.path.join(self.get_temp_dir(), "gptq_model.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        y_after = reloaded.predict(x_eval)

        self.assertAllClose(y_before, y_after)

    def test_gptq_calibrates_dense_with_layer_activation(self):
        """A `Dense` with a `Layer` activation inside a block is calibrated.

        The activation `Layer` makes the `Dense` a non-leaf, but GPTQ must
        still discover and calibrate it: after `quantize("gptq")` the layer
        is in `gptq` mode with its calibration no longer pending.
        """
        keras.utils.set_random_seed(123)
        embed_dim = 8
        model, structure = tiny_calibration_model(
            [layers.Dense(embed_dim, activation=layers.ReLU())],
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=embed_dim,
            head_units=NUM_CLASSES,
        )

        rng = np.random.default_rng(seed=7)
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
        config = calibration_config(
            "gptq",
            dataset=token_dataset(4, SEQ_LEN, VOCAB_SIZE, rng),
            tokenizer=tokenizer,
            weight_bits=4,
            group_size=8,
            num_samples=4,
            sequence_length=SEQ_LEN,
            quantization_layer_structure=structure,
        )

        model.quantize("gptq", config=config)

        act_dense = structure["sequential_blocks"][0].layers[0]
        self.assertEqual(act_dense.quantization_mode, "gptq")
        self.assertFalse(act_dense.calibration_pending)

    def test_gptq_missing_structure_leaves_model_unmodified(self):
        """A config without a structure raises before any layer is mutated."""
        model = _get_simple_model()
        dense = model.layers[0]

        config = GPTQConfig(dataset=["a"], tokenizer=lambda x: x)

        with self.assertRaisesRegex(
            ValueError, "a valid quantization structure"
        ):
            model.quantize("gptq", config=config)

        # The model must be left unmodified when the structure is missing.
        self.assertIsNone(getattr(dense, "quantization_mode", None))
        self.assertFalse(hasattr(dense, "quantized_kernel"))

    def test_gptq_save_load_round_trip_einsum_dense_block(self):
        """Regression test for a RecursionError when saving a GPTQ model.

        Saving a GPTQ-quantized model used to raise a RecursionError because
        `GPTQConfig.get_config` serialized `quantization_layer_structure`,
        which holds live model layers, forming a reference cycle
        (layer -> config -> layer). This builds a tiny model whose quantized
        block contains both `Dense` and `EinsumDense` (unlike the Dense-only
        round trip above), saves it, reloads it, and checks the predictions
        are preserved exactly.
        """
        vocab_size, seq_len, embed_dim = 32, 8, 4
        model, structure = tiny_calibration_model(
            [
                layers.Dense(embed_dim, activation="relu"),
                layers.EinsumDense(
                    "abc,cd->abd", output_shape=(seq_len, embed_dim)
                ),
            ],
            vocab_size=vocab_size,
            sequence_length=seq_len,
            embed_dim=embed_dim,
            head_units=2,
        )
        (embedding,) = structure["pre_block_layers"]
        head = model.layers[-1]

        rng = np.random.default_rng(seed=13)
        config = calibration_config(
            "gptq",
            dataset=token_dataset(3, seq_len, vocab_size, rng),
            tokenizer=lambda text: text,
            weight_bits=4,
            num_samples=2,
            sequence_length=seq_len,
            group_size=4,
            quantization_layer_structure=structure,
        )

        # Layers outside the structure (embedding, pooling, head) are not
        # quantized at all, so the round-trip can be compared exactly.
        model.quantize("gptq", config=config)
        self.assertIsNone(getattr(head, "quantization_mode", None))

        # The embedding only supports int8/int4; `quantize` must reject the
        # unsupported mode without stashing a stale GPTQ config on it.
        self.assertIsNone(embedding.quantization_config)

        x_eval = rng.integers(0, vocab_size, size=(2, seq_len)).astype("int32")
        y_quantized = model.predict(x_eval)

        # This `save` used to raise a RecursionError.
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        restored = saving.load_model(path)
        y_restored = restored.predict(x_eval)
        self.assertAllClose(y_quantized, y_restored)

        # The quantized block state survives the round-trip.
        restored_block = next(
            l for l in restored.layers if isinstance(l, models.Sequential)
        )
        restored_dense = restored_block.layers[0]
        self.assertEqual(
            getattr(restored_dense, "quantization_mode", None), "gptq"
        )
        self.assertTrue(hasattr(restored_dense, "quantized_kernel"))
        self.assertIsNone(
            restored_dense.quantization_config.quantization_layer_structure
        )
