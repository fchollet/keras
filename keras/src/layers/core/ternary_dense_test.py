import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.common import standardize_dtype


class TernaryDenseTest(testing.TestCase):
    def test_output_shape(self):
        layer = layers.TernaryDense(8)
        x = np.random.rand(4, 16).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (4, 8))

    def test_effective_kernel_is_ternary(self):
        layer = layers.TernaryDense(16)
        x = np.ones((1, 8), dtype="float32")
        layer(x)
        k, _ = layer._ternary_kernel()
        k = ops.convert_to_numpy(k)
        # STE: round to nearest int for fp tolerance
        rounded = set(np.round(k).astype(np.int32).flat)
        self.assertTrue(
            rounded <= {-1, 0, 1},
            f"Expected values near {{-1, 0, 1}}, got {np.unique(k)}",
        )

    def test_threshold_zero_no_zeros(self):
        # threshold=0.0: |w| > 0 for all non-zero glorot weights → all ±1
        layer = layers.TernaryDense(16, threshold=0.0)
        x = np.ones((1, 8), dtype="float32")
        layer(x)
        k, _ = layer._ternary_kernel()
        k = ops.convert_to_numpy(k)
        self.assertNotIn(0.0, np.unique(k).tolist())

    def test_threshold_large_all_zeros(self):
        layer = layers.TernaryDense(4, threshold=1e9)
        x = np.ones((1, 4), dtype="float32")
        y = layer(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y),
            np.zeros((1, 4), dtype="float32"),
            atol=1e-6,
        )

    def test_forward_matches_manual(self):
        # 2-input, 2-output layer, no bias.
        # kernel = [[1, -1], [0, 1]]  (will be forced via direct assignment)
        # input  = [[2, 3]]
        # expected output = [[2*1 + 3*0, 2*(-1) + 3*1]] = [[2, 1]]
        layer = layers.TernaryDense(2, use_bias=False)
        layer.build((None, 2))
        layer.kernel.assign(
            np.array([[1.0, -1.0], [0.0, 1.0]], dtype="float32")
        )
        layer.threshold = 0.5  # only |w|>0.5 survive: 0.0 → 0, ±1.0 → ±1
        x = np.array([[2.0, 3.0]], dtype="float32")
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(y, [[2.0, 1.0]], atol=1e-6)

    def test_no_bias(self):
        layer = layers.TernaryDense(4, use_bias=False)
        layer.build((None, 4))
        self.assertIsNone(layer.bias)

    def test_activation(self):
        layer = layers.TernaryDense(8, activation="relu")
        x = np.random.randn(2, 4).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        self.assertTrue(np.all(y >= 0), "relu output should be non-negative")

    def test_nd_input(self):
        # Works on 3D input (batch, seq, dim)
        layer = layers.TernaryDense(16)
        x = np.random.rand(2, 5, 8).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 16))

    def test_get_config_roundtrip(self):
        layer = layers.TernaryDense(
            32,
            threshold=0.01,
            activation="relu",
            use_bias=False,
        )
        config = layer.get_config()
        self.assertEqual(config["units"], 32)
        self.assertAlmostEqual(config["threshold"], 0.01)
        self.assertEqual(config["use_bias"], False)

        restored = layers.TernaryDense.from_config(config)
        self.assertEqual(restored.units, 32)
        self.assertAlmostEqual(restored.threshold, 0.01)

    def test_serialization(self):
        layer = layers.TernaryDense(8)
        x = np.random.rand(2, 4).astype("float32")
        layer(x)
        config = layer.get_config()
        restored = layers.TernaryDense.from_config(config)
        self.assertEqual(restored.units, layer.units)

    def test_invalid_units(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.TernaryDense(0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.TernaryDense(-4)

    def test_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            layers.TernaryDense(4, threshold=-0.1)

    def test_beta_scaling_applied_when_threshold_none(self):
        # threshold=None → output *= beta=mean(|kernel|) per BitNet b1.58 §3.1
        layer = layers.TernaryDense(2, use_bias=False)
        layer.build((None, 2))
        kernel = np.array([[0.5, -0.5], [0.5, 0.5]], dtype="float32")
        layer.kernel.assign(kernel)
        # t=0.25; all |w|=0.5 > t → all ±1; beta=0.5
        # k_ternary = [[1, -1], [1, 1]]; beta = mean(|kernel|) = 0.5
        # x=[[1,1]] @ k_ternary = [[2, 0]]; scaled = [[1.0, 0.0]]
        x = np.array([[1.0, 1.0]], dtype="float32")
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(y, [[1.0, 0.0]], atol=1e-5)

    def test_compute_output_shape(self):
        layer = layers.TernaryDense(10)
        self.assertEqual(layer.compute_output_shape((None, 5)), (None, 10))
        self.assertEqual(layer.compute_output_shape((2, 3, 5)), (2, 3, 10))

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_call_on_a_different_device_than_build(self):
        # The kernel lives on the device the layer was built on. A call
        # under another device runs there, and the gradient still reaches
        # the kernel.
        layer = layers.TernaryDense(8)
        layer.build((None, 16))
        build_device = layer._kernel.value.device
        x = np.random.rand(4, 16).astype("float32")

        with backend.device("cpu"):
            y = layer(x)
        self.assertEqual(y.device.type, "cpu")

        ops.sum(y).backward()
        grad = layer._kernel.value.grad
        self.assertEqual(grad.device, build_device)
        self.assertGreater(ops.convert_to_numpy(ops.sum(ops.abs(grad))), 0)

    # Quantization: real ternary export + inference.

    def test_quantize_matches_float_forward(self):
        # Freezing the ternary kernel must not change the layer's outputs:
        # the packed kernel holds exactly the values the float path produces.
        layer = layers.TernaryDense(16)
        layer.build((None, 11))
        x = np.random.rand(4, 11).astype("float32")
        y_float = layer(x)

        layer.quantize("ternary")
        self.assertEqual(layer.quantization_mode, "ternary")
        # `kernel` is now the unpacked `{-1, 0, +1}` view of the packed codes.
        k = ops.convert_to_numpy(layer.kernel)
        self.assertEqual(k.shape, (11, 16))
        self.assertTrue(set(np.unique(k).tolist()) <= {-1, 0, 1})
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "uint8")

        y_quantized = layer(x)
        self.assertAllClose(y_float, y_quantized)

    @parameterized.named_parameters(
        ("mixed_float16", "mixed_float16", True, False, 1e-3),
        ("mixed_float16_no_autocast", "mixed_float16", False, False, 1e-3),
        (
            "mixed_float16_no_autocast_nested",
            "mixed_float16",
            False,
            True,
            1e-3,
        ),
        ("mixed_bfloat16", "mixed_bfloat16", True, False, 1e-2),
        ("bfloat16", "bfloat16", True, False, 1e-2),
    )
    def test_quantize_matches_mixed_precision_forward(
        self, dtype, autocast, nested, tolerance
    ):
        # The forward pass decides the trits on the stored float32 kernel,
        # whatever autocast scope it runs in, and the freeze uses the same
        # rule, so the packed codes are exactly the forward's trits. Every
        # fourth weight lies just above the float32 threshold and rounds
        # below it in float16 and bfloat16.
        layer = layers.TernaryDense(16, dtype=dtype, autocast=autocast)
        layer.build((None, 12))
        pattern = np.array([[0.5], [-0.5], [0.5], [0.21429]])
        layer._kernel.assign(np.tile(pattern, (3, 16)))
        x = np.random.rand(4, 12).astype("float32")
        if nested:
            with AutocastScope(layer.compute_dtype):
                y_float = layer(x)
        else:
            y_float = layer(x)
        with AutocastScope(layer.compute_dtype):
            k_ste, _ = layer._ternary_kernel()
        k_ste = ops.convert_to_numpy(ops.cast(k_ste, "float32"))

        layer.quantize("ternary")
        codes = ops.convert_to_numpy(ops.cast(layer.kernel, "float32"))
        self.assertAllEqual(codes, k_ste)
        if dtype != "bfloat16":
            # Decided on the float32 variable, not a rounded copy.
            self.assertAllEqual(codes[3], np.ones(16))
        self.assertAllClose(y_float, layer(x), atol=tolerance, rtol=tolerance)

    @parameterized.named_parameters(
        ("float32", "float32"),
        ("mixed_float16", "mixed_float16"),
        ("mixed_bfloat16", "mixed_bfloat16"),
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
    )
    def test_forward_runs_in_compute_dtype(self, dtype):
        # The kernel is created with `autocast=False`, so it is the layer
        # that casts it for the matmul. The output is in the compute dtype
        # and the stored kernel keeps the variable dtype.
        layer = layers.TernaryDense(8, dtype=dtype)
        layer.build((None, 16))
        x = np.random.rand(4, 16).astype("float32")
        y = layer(x)
        self.assertEqual(standardize_dtype(y.dtype), layer.compute_dtype)
        self.assertEqual(
            standardize_dtype(layer._kernel.dtype), layer.variable_dtype
        )

    def test_quantize_fixed_threshold_matches_float_forward(self):
        # Fixed-threshold mode applies no beta rescaling (scale == 1.0).
        layer = layers.TernaryDense(8, threshold=0.02, use_bias=False)
        layer.build((None, 9))
        x = np.random.rand(3, 9).astype("float32")
        y_float = layer(x)

        layer.quantize("ternary")
        self.assertAllClose(ops.convert_to_numpy(layer.kernel_scale), 1.0)
        self.assertAllClose(y_float, layer(x))

    @parameterized.named_parameters(
        ("mixed_bfloat16", "mixed_bfloat16"), ("bfloat16", "bfloat16")
    )
    def test_quantize_bfloat16(self, dtype):
        # The threshold and the scale come from a float32 mean of the
        # kernel on every backend, however many weights it has. Every
        # fourth weight lies just above the float32 threshold, which
        # rounds up onto it in bfloat16.
        layer = layers.TernaryDense(64, dtype=dtype)
        layer.build((None, 256))
        pattern = np.array([[0.5625], [-0.5625], [0.5625], [0.2412109375]])
        layer._kernel.assign(np.tile(pattern, (64, 64)))
        kernel = ops.cast(layer._kernel, "float32")
        beta = float(ops.convert_to_numpy(ops.mean(ops.abs(kernel))))
        kernel = ops.convert_to_numpy(kernel)
        forward = ops.convert_to_numpy(
            ops.cast(layer._ternary_kernel()[0], "float32")
        )

        layer.quantize("ternary")
        scale = ops.convert_to_numpy(ops.cast(layer.kernel_scale, "float32"))
        self.assertAllClose(scale, beta, rtol=1e-2)
        expected = np.sign(kernel) * (np.abs(kernel) > 0.5 * beta)
        codes = ops.convert_to_numpy(ops.cast(layer.kernel, "float32"))
        self.assertAllClose(codes, expected)
        # The frozen codes are the forward value of the training kernel.
        self.assertAllEqual(codes, forward)

    @parameterized.named_parameters(
        ("float16", "float16"), ("mixed_float16", "mixed_float16")
    )
    def test_quantize_small_kernel_float16(self, dtype):
        # `mean(|W|)` is about 1e-5, so `1 / mean(|W|)` would overflow
        # float16. The multiplier scale keeps the outputs' magnitude.
        rng = np.random.RandomState(0)
        layer = layers.TernaryDense(32, use_bias=False, dtype=dtype)
        layer.build((None, 64))
        layer._kernel.assign(rng.randn(64, 32) * 1e-5)
        kernel = ops.convert_to_numpy(ops.cast(layer._kernel, "float32"))
        beta = np.mean(np.abs(kernel))
        codes = np.sign(kernel) * (np.abs(kernel) > 0.5 * beta)
        x = rng.randn(4, 64).astype("float32")
        layer.quantize("ternary")

        y = ops.convert_to_numpy(ops.cast(layer(x), "float32"))
        self.assertAllClose(y, np.matmul(x, codes) * beta, rtol=1e-2, atol=1e-6)

    def test_quantized_kernel_is_packed_at_floor(self):
        # units=40 -> ceil(40/5)=8 bytes per row; each byte holds 5 trits.
        layer = layers.TernaryDense(40)
        layer.build((None, 32))
        layer.quantize("ternary")

        self.assertEqual(tuple(layer._kernel.shape), (32, 8))

        n_weights = 32 * 40
        n_bytes = int(np.prod(layer._kernel.shape))
        bits_per_weight = 8 * n_bytes / n_weights
        self.assertEqual(bits_per_weight, 1.6)
        # Strictly denser than int4 (would need n_weights / 2 bytes).
        self.assertLess(n_bytes, n_weights // 2)

    def test_quantized_model_save_load(self):
        layer = layers.TernaryDense(16)
        layer.build((None, 8))
        x = np.random.random((2, 8))
        y_float = layer(x)
        layer.quantize("ternary")
        y_quantized = layer(x)
        self.assertAllClose(y_float, y_quantized)

        # Full model save / load round-trip.
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_ternary_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertEqual(new_model.layers[0].quantization_mode, "ternary")
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Weights-only save / load round-trip.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_ternary_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.TernaryDense(16)])
        new_model.build((None, 8))
        new_model.quantize("ternary")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_model_quantize_ternary(self):
        model = models.Sequential([layers.TernaryDense(8)])
        model.build((None, 6))
        x = np.random.rand(3, 6).astype("float32")
        y_float = model.predict(x)
        model.quantize("ternary")
        self.assertEqual(model.layers[0].quantization_mode, "ternary")
        self.assertAllClose(y_float, model.predict(x))

    def test_quantize_unbuilt_raises(self):
        layer = layers.TernaryDense(4)
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize("ternary")

    def test_quantize_twice_raises(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 6))
        layer.quantize("ternary")
        with self.assertRaisesRegex(ValueError, "already quantized"):
            layer.quantize("ternary")

    def test_quantize_invalid_mode_raises(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 6))
        with self.assertRaises(NotImplementedError):
            layer.quantize("int8")

    def test_quantize_subclass_type_check_raises(self):
        # type_check=True (default) must reject subclasses with a different
        # kernel layout to prevent silent weight mismatches on save/load.
        class MyTernaryDense(layers.TernaryDense):
            pass

        layer = MyTernaryDense(4)
        layer.build((None, 6))
        with self.assertRaises(NotImplementedError):
            layer.quantize("ternary", type_check=True)

    def test_quantize_subclass_type_check_false(self):
        # type_check=False bypasses the class guard — useful for subclasses that
        # haven't changed the kernel layout and explicitly opt in.
        class MyTernaryDense(layers.TernaryDense):
            pass

        layer = MyTernaryDense(4)
        layer.build((None, 6))
        x = np.random.rand(2, 6).astype("float32")
        y_float = layer(x)
        layer.quantize("ternary", type_check=False)
        self.assertEqual(layer.quantization_mode, "ternary")
        self.assertAllClose(y_float, layer(x))

    def test_quantized_build_invalid_mode_raises(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 6))
        with self.assertRaises(NotImplementedError):
            layer.quantized_build((6, 4), mode="int8")

    def test_variable_serialization_spec(self):
        layer = layers.TernaryDense(8)
        spec = layer.variable_serialization_spec
        self.assertIn(None, spec)
        self.assertIn("ternary", spec)
        self.assertEqual(spec[None], ["kernel", "bias"])
        self.assertIn("kernel_scale", spec["ternary"])

    def test_save_load_no_bias(self):
        # Covers the `if name == "bias" and self.bias is None: continue` branch
        # in save_own_variables / load_own_variables.
        layer = layers.TernaryDense(16, use_bias=False)
        layer.build((None, 8))
        x = np.random.rand(3, 8).astype("float32")
        y_float = layer(x)
        layer.quantize("ternary")
        y_quantized = layer(x)
        self.assertAllClose(y_float, y_quantized)

        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "ternary_nobias_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        temp_weights = os.path.join(
            self.get_temp_dir(), "ternary_nobias_model.weights.h5"
        )
        model.save_weights(temp_weights)
        new_model = models.Sequential([layers.TernaryDense(16, use_bias=False)])
        new_model.build((None, 8))
        new_model.quantize("ternary")
        new_model.load_weights(temp_weights)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_quantize_skips_dtype_policy_update_when_already_set(self):
        # When dtype_policy.quantization_mode is already set after the first
        # quantize(), a second call hits the "already quantized" guard before
        # reaching the policy block — verifies the policy guard is not None.
        layer = layers.TernaryDense(8)
        layer.build((None, 6))
        layer.quantize("ternary")
        self.assertIsNotNone(layer.dtype_policy.quantization_mode)
        with self.assertRaisesRegex(ValueError, "already quantized"):
            layer.quantize("ternary")
