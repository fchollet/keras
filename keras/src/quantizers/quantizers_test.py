import inspect
import itertools
import math

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import quantizers
from keras.src import random
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_test_utils import host_data_ops
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_zero_point
from keras.src.quantizers.quantizers import ternarize
from keras.src.testing.test_utils import named_product


class QuantizersTest(testing.TestCase):
    def test_get_method(self):
        quantizer = quantizers.get("abs_max_quantizer")
        self.assertTrue(quantizer, quantizers.AbsMaxQuantizer)

        quantizer = quantizers.get(None)
        self.assertEqual(quantizer, None)

        with self.assertRaises(ValueError):
            quantizers.get("typo")

    def test_get_from_string_identifier(self):
        # A string identifier resolves to a fresh instance whose constructor
        # ran with no positional garbage (previously the kwargs dict was passed
        # positionally into `axis`).
        quantizer = quantizers.get("abs_max_quantizer")
        self.assertIsInstance(quantizer, quantizers.AbsMaxQuantizer)
        self.assertIsNone(quantizer.axis)
        self.assertEqual(quantizer.value_range, (-127, 127))

    def test_get_from_bare_class(self):
        quantizer = quantizers.get(quantizers.AbsMaxQuantizer)
        self.assertIsInstance(quantizer, quantizers.AbsMaxQuantizer)
        self.assertIsNone(quantizer.axis)

    def test_get_from_class_with_kwargs(self):
        # kwargs must be forwarded by keyword, not positionally.
        quantizer = quantizers.get(
            quantizers.AbsMaxQuantizer,
            value_range=(-8, 7),
            output_dtype="int8",
        )
        self.assertIsInstance(quantizer, quantizers.AbsMaxQuantizer)
        self.assertEqual(quantizer.value_range, (-8, 7))
        self.assertEqual(quantizer.output_dtype, "int8")
        # `axis` stays at its default rather than being clobbered by kwargs.
        self.assertIsNone(quantizer.axis)

    def test_get_from_config_dict(self):
        # A serialized `{"class_name": ..., "config": ...}` dict deserializes to
        # an equivalent instance.
        original = quantizers.AbsMaxQuantizer(value_range=(-8, 7))
        config = quantizers.serialize(original)
        self.assertIn("class_name", config)
        self.assertIn("config", config)
        quantizer = quantizers.get(config)
        self.assertIsInstance(quantizer, quantizers.AbsMaxQuantizer)
        self.assertEqual(quantizer.value_range, (-8, 7))
        self.assertIsNone(quantizer.axis)

    @parameterized.named_parameters(
        (mode, mode) for mode in strategy_registry.registered_modes()
    )
    def test_mode_config_round_trips(self, mode):
        # `deserialize` resolves the config class of every registered mode.
        config_cls = strategy_registry.get_strategy(mode).config_cls
        if config_cls is None:
            self.skipTest(f"Mode '{mode}' defines no config class.")
        # Calibration configs take `dataset` and `tokenizer` with no default.
        # Neither is serialized, so `None` is enough here.
        required = {
            name: None
            for name, param in inspect.signature(config_cls).parameters.items()
            if param.default is param.empty
        }
        serialized = quantizers.serialize(config_cls(**required))
        reloaded = quantizers.deserialize(serialized)
        self.assertIsInstance(reloaded, config_cls)
        self.assertEqual(quantizers.serialize(reloaded), serialized)

    def test_abs_max_quantizer(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float32")
        quantizer = quantizers.AbsMaxQuantizer()

        # Test quantizing
        quantized_values, scale = quantizer(values, axis=-1)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "float32")
        self.assertEqual(tuple(quantized_values.shape), (3, 4, 5))
        self.assertEqual(tuple(scale.shape), (3, 4, 1))
        self.assertLessEqual(ops.max(quantized_values), 127)
        self.assertGreaterEqual(ops.min(quantized_values), -127)

        # Test dequantizing
        dequantized_values = ops.divide(quantized_values, scale)
        rmse = ops.sqrt(
            ops.mean(ops.square(ops.subtract(values, dequantized_values)))
        )
        self.assertLess(rmse, 1e-1)  # loose assertion

        # Test serialization
        self.run_class_serialization_test(quantizer)

        # Test bfloat16 & float16 dtype
        values = random.uniform(
            [3, 4, 5], minval=-1, maxval=1, dtype="bfloat16"
        )
        quantized_values, scale = quantizer(values, axis=-1)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "bfloat16")
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float16")
        quantized_values, scale = quantizer(values, axis=-1)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "float16")

    def test_abs_max_quantizer_to_numpy(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float32")
        quantized_values, scale = quantizers.abs_max_quantize(
            values, axis=-1, to_numpy=True
        )
        ref_quantized_values, ref_scale = quantizers.abs_max_quantize(
            values, axis=-1
        )
        self.assertAllClose(quantized_values, ref_quantized_values)
        self.assertAllClose(scale, ref_scale)

    def test_pack_int4_layout_is_pinned(self):
        # The packed bytes are a checkpoint format: value `2i` sits in the
        # low nibble of byte `i` and value `2i + 1` in the high nibble, and
        # an odd length is padded with a zero nibble.
        codes = np.array([[-3, 7], [2, -8], [1, 0]], "int8")
        packed, _, orig_len = quantizers.pack_int4(codes, axis=0)
        self.assertEqual(orig_len, 3)
        self.assertAllEqual(packed, [[45, -121], [1, 0]])
        codes = np.array([[-3, 7, 2], [-8, 1, 0]], "int8")
        packed, _, _ = quantizers.pack_int4(codes, axis=1)
        self.assertAllEqual(packed, [[125, 2], [24, 0]])

    @parameterized.named_parameters(
        ("int4_int8", 4, "int8"),
        ("int4_uint8", 4, "uint8"),
        ("int2_int8", 2, "int8"),
        ("int2_uint8", 2, "uint8"),
    )
    def test_unpack_every_byte(self, bits, dtype):
        # Every byte value unpacks to its fields, lowest bits first,
        # sign-extended for `int8`.
        every_byte = np.arange(256, dtype="uint8").astype(dtype).reshape(256, 1)
        unpack = quantizers.unpack_int4 if bits == 4 else quantizers.unpack_int2
        fields = 8 // bits
        unpacked = unpack(every_byte, fields, axis=1, dtype=dtype)
        values = np.arange(256)[:, None] >> (bits * np.arange(fields))
        values = values & ((1 << bits) - 1)
        if dtype == "int8":
            half = 1 << (bits - 1)
            values = (values ^ half) - half
        self.assertAllEqual(unpacked, values)

    SHAPE_AXIS_SCENARIOS = [
        # 1. 2D Tensors
        # Axis 0, both parities
        {"testcase_name": "2d_axis0_odd", "shape": (5, 8), "axis": 0},
        {"testcase_name": "2d_axis0_even", "shape": (4, 8), "axis": 0},
        # A middle axis and a negative axis of a 2D tensor
        {"testcase_name": "2d_axis1_odd", "shape": (8, 7), "axis": 1},
        {"testcase_name": "2d_axis_neg1_even", "shape": (8, 6), "axis": -1},
        # 2. Higher-Rank Tensors
        # Covers a middle axis for a complex shape with both parities
        {"testcase_name": "4d_axis1_odd", "shape": (2, 5, 4, 6), "axis": 1},
        {"testcase_name": "4d_axis2_even", "shape": (2, 4, 8, 6), "axis": 2},
        # Covers the last axis of a complex shape with a negative index
        {
            "testcase_name": "4d_axis_neg1_odd",
            "shape": (2, 4, 6, 7),
            "axis": -1,
        },
    ]

    DTYPE_PARAMS = [
        {"testcase_name": "int8", "dtype": "int8", "minval": -8, "maxval": 8},
        {"testcase_name": "uint8", "dtype": "uint8", "minval": 0, "maxval": 16},
    ]

    @parameterized.named_parameters(
        named_product(SHAPE_AXIS_SCENARIOS, DTYPE_PARAMS)
    )
    def test_pack_unpack_int4(self, shape, axis, dtype, minval, maxval):
        # Create a random tensor with int4 values in the specified range and
        # dtype
        arr = ops.cast(
            ops.floor(random.uniform(shape, minval=minval, maxval=maxval)),
            dtype,
        )

        # Pack the tensor using the specified dtype
        packed, packed_shape, orig_len = quantizers.pack_int4(
            arr, axis=axis, dtype=dtype
        )

        # Unpack the tensor using the specified dtype
        unpacked = quantizers.unpack_int4(
            packed, orig_len, axis=axis, dtype=dtype
        )

        # Verify that the packed tensor has the correct dtype
        self.assertDType(packed, dtype)

        # Verify that the unpacked tensor has the correct dtype
        self.assertDType(unpacked, dtype)

        # The unpacked tensor should be the same as the original tensor
        self.assertAllClose(unpacked, arr)

        # Test the packed shape
        expected_packed_shape = list(shape)
        expected_packed_shape[axis] = (expected_packed_shape[axis] + 1) // 2
        self.assertEqual(
            list(ops.convert_to_numpy(packed_shape)), expected_packed_shape
        )

    @parameterized.named_parameters(SHAPE_AXIS_SCENARIOS)
    def test_pack_unpack_ternary(self, shape, axis):
        # Random ternary values in {-1, 0, +1}.
        arr = ops.cast(
            ops.floor(random.uniform(shape, minval=-1, maxval=2)), "int8"
        )

        packed, packed_shape, orig_len = quantizers.pack_ternary(arr, axis=axis)
        unpacked = quantizers.unpack_ternary(packed, orig_len, axis=axis)

        # Packed bytes are uint8, with five trits packed along the pack axis.
        ax = axis % len(shape)
        self.assertDType(packed, "uint8")
        self.assertEqual(packed_shape[ax], (shape[ax] + 4) // 5)

        # The round-trip is lossless back to {-1, 0, +1}.
        self.assertDType(unpacked, "int8")
        self.assertAllClose(unpacked, arr)

    def test_pack_ternary_bits_per_value(self):
        # 100 trits packed along axis 0 (length 25 -> 5 bytes) for 4 columns.
        arr = ops.cast(
            ops.floor(random.uniform((25, 4), minval=-1, maxval=2)), "int8"
        )
        packed, packed_shape, _ = quantizers.pack_ternary(arr, axis=0)

        n_bytes = 1
        for dim in packed_shape:
            n_bytes *= dim
        # 5 bytes/column * 4 columns = 20 bytes for 100 ternary values: the
        # log2(3) ~= 1.58-bit floor, denser than int4 (50 bytes) or int8.
        self.assertEqual(n_bytes, 20)
        self.assertEqual(8 * n_bytes / 100, 1.6)

    # int2 packs four values per byte, so the packing axis must exercise every
    # padding remainder (0, 1, 2, 3) along both the fast path (axis=0, rank 2)
    # and the general transpose path.
    DTYPE_PARAMS_INT2 = [
        {"testcase_name": "int8", "dtype": "int8", "minval": -2, "maxval": 2},
        {"testcase_name": "uint8", "dtype": "uint8", "minval": 0, "maxval": 4},
    ]

    @parameterized.named_parameters(
        named_product(SHAPE_AXIS_SCENARIOS, DTYPE_PARAMS_INT2)
    )
    def test_pack_unpack_int2(self, shape, axis, dtype, minval, maxval):
        arr = ops.cast(
            ops.floor(random.uniform(shape, minval=minval, maxval=maxval)),
            dtype,
        )

        packed, packed_shape, orig_len = quantizers.pack_int2(
            arr, axis=axis, dtype=dtype
        )
        unpacked = quantizers.unpack_int2(
            packed, orig_len, axis=axis, dtype=dtype
        )

        self.assertDType(packed, dtype)
        self.assertDType(unpacked, dtype)
        self.assertAllClose(unpacked, arr)

        # Four values are packed per byte along `axis`.
        expected_packed_shape = list(shape)
        expected_packed_shape[axis] = (expected_packed_shape[axis] + 3) // 4
        self.assertEqual(
            list(ops.convert_to_numpy(packed_shape)), expected_packed_shape
        )

    @parameterized.named_parameters(
        {"testcase_name": "int8", "dtype": "int8", "values": [-2, -1, 0, 1]},
        {"testcase_name": "uint8", "dtype": "uint8", "values": [0, 1, 2, 3]},
    )
    def test_pack_unpack_int2_exhaustive(self, dtype, values):
        # Every possible arrangement of four 2-bit values within a byte
        # (4**4 = 256 quadruples) must round-trip bit-exactly.
        combos = np.array(
            list(itertools.product(values, repeat=4)), dtype=dtype
        )
        arr = ops.convert_to_tensor(combos.reshape(-1, 1))  # (1024, 1)

        packed, _, orig_len = quantizers.pack_int2(arr, axis=0, dtype=dtype)
        unpacked = quantizers.unpack_int2(packed, orig_len, axis=0, dtype=dtype)

        # 1024 values -> 256 packed bytes (4 per byte).
        self.assertEqual(tuple(ops.shape(packed)), (256, 1))
        self.assertAllClose(unpacked, arr)

    @parameterized.named_parameters(
        ("block_32", 32),
        ("block_64", 64),
        ("block_128", 128),
    )
    def test_grouped_quantize_dequantize_roundtrip(self, block_size):
        """Test that grouped quantize/dequantize has low error."""
        input_dim, output_dim = 256, 128
        kernel = random.uniform(
            (input_dim, output_dim), minval=-1, maxval=1, dtype="float32"
        )

        quantized, scale, zero = (
            quantizers.abs_max_quantize_grouped_with_zero_point(
                kernel,
                block_size=block_size,
                value_range=(-8, 7),
                dtype="int8",
            )
        )

        # Use dequantize_with_sz_map with generated g_idx
        g_idx = ops.arange(input_dim) // block_size
        dequantized = ops.transpose(
            quantizers.dequantize_with_sz_map(
                ops.transpose(ops.cast(quantized, scale.dtype)),
                ops.transpose(scale),
                ops.transpose(zero),
                g_idx,
            )
        )

        rmse = ops.sqrt(ops.mean(ops.square(kernel - dequantized)))
        # Grouped quantization should have reasonable error
        self.assertLess(rmse, 0.15)

    def test_grouped_quantize_with_padding(self):
        """Test grouped quantization when input_dim is not divisible."""

        # 500 is not divisible by 128, so padding will be needed
        input_dim, output_dim, block_size = 500, 256, 128
        kernel = random.uniform(
            (input_dim, output_dim), minval=-1, maxval=1, dtype="float32"
        )

        quantized, scale, zero = (
            quantizers.abs_max_quantize_grouped_with_zero_point(
                kernel,
                block_size=block_size,
                value_range=(-8, 7),
                dtype="int8",
            )
        )

        n_groups = math.ceil(input_dim / block_size)  # 4 groups
        self.assertEqual(quantized.shape, (input_dim, output_dim))
        self.assertEqual(scale.shape, (n_groups, output_dim))
        self.assertEqual(zero.shape, (n_groups, output_dim))

    def test_grouped_vs_perchannel_accuracy(self):
        """Test that grouped quantization has lower error than per-channel."""
        input_dim, output_dim, block_size = 512, 256, 128
        # Use a specific seed for reproducibility
        kernel = random.uniform(
            (input_dim, output_dim),
            minval=-1,
            maxval=1,
            dtype="float32",
            seed=42,
        )

        # Per-channel quantization (one scale per output channel)
        quantizer = quantizers.AbsMaxQuantizer(
            axis=0, value_range=(-8, 7), output_dtype="int8"
        )
        pc_quantized, pc_scale = quantizer(kernel)
        pc_dequantized = ops.cast(pc_quantized, "float32") / pc_scale
        pc_rmse = ops.sqrt(ops.mean(ops.square(kernel - pc_dequantized)))

        # Grouped (sub-channel) quantization with zero point
        grouped_quantized, grouped_scale, grouped_zero = (
            quantizers.abs_max_quantize_grouped_with_zero_point(
                kernel, block_size=block_size, value_range=(-8, 7), dtype="int8"
            )
        )

        # Use dequantize_with_sz_map with generated g_idx
        g_idx = ops.arange(input_dim) // block_size
        grouped_dequantized = ops.transpose(
            quantizers.dequantize_with_sz_map(
                ops.transpose(ops.cast(grouped_quantized, grouped_scale.dtype)),
                ops.transpose(grouped_scale),
                ops.transpose(grouped_zero),
                g_idx,
            )
        )

        grouped_rmse = ops.sqrt(
            ops.mean(ops.square(kernel - grouped_dequantized))
        )

        # Grouped should have lower or similar error
        # (in most cases it should be lower due to finer granularity)
        self.assertLessEqual(float(grouped_rmse), float(pc_rmse) + 0.01)

    def test_grouped_quantize_various_block_sizes(self):
        """Test grouped quantization with various block sizes."""

        input_dim, output_dim = 512, 128
        kernel = random.uniform(
            (input_dim, output_dim), minval=-1, maxval=1, dtype="float32"
        )

        for block_size in [32, 64, 128, 256]:
            quantized, scale, zero = (
                quantizers.abs_max_quantize_grouped_with_zero_point(
                    kernel,
                    block_size=block_size,
                    value_range=(-8, 7),
                    dtype="int8",
                )
            )

            n_groups = math.ceil(input_dim / block_size)
            self.assertEqual(quantized.shape, (input_dim, output_dim))
            self.assertEqual(scale.shape, (n_groups, output_dim))
            self.assertEqual(zero.shape, (n_groups, output_dim))

    @parameterized.named_parameters(("tensor", False), ("numpy", True))
    def test_grouped_quantize_one_signed_groups(self, to_numpy):
        # The first group is all positive in column 0, all negative in
        # column 1 and all small positive in column 2. Its range is widened
        # to include zero, so the zero point stays inside the code range
        # and every value lands within half a step of its input instead
        # of clipping to one end of the group's range.
        block_size = 4
        kernel = np.array(
            [
                [0.5, -0.6, 0.02],
                [0.7, -0.8, 0.03],
                [0.6, -0.7, 0.01],
                [0.9, -0.5, 0.04],
                [-0.5, 0.5, -0.9],
                [0.5, -0.5, 0.9],
                [0.2, 0.1, 0.0],
                [-0.2, -0.1, 0.0],
            ],
            "float32",
        )
        quantized, scale, zero = (
            quantizers.abs_max_quantize_grouped_with_zero_point(
                kernel, block_size=block_size, to_numpy=to_numpy
            )
        )
        quantized = ops.convert_to_numpy(quantized).astype("float32")
        scale = ops.convert_to_numpy(scale)
        zero = ops.convert_to_numpy(zero).astype("float32")
        self.assertTrue(np.all(zero >= -8) and np.all(zero <= 7))
        g_idx = np.arange(kernel.shape[0]) // block_size
        dequantized = (quantized - zero[g_idx]) * scale[g_idx]
        half_step = scale[g_idx] / 2 + 1e-6
        self.assertTrue(np.all(np.abs(dequantized - kernel) <= half_step))

    @parameterized.named_parameters(("tensor", False), ("numpy", True))
    def test_grouped_zero_point_exact_values(self, to_numpy):
        # One group of three rows: column 0 is all positive and column 1
        # all negative, so each range is widened to include zero.
        kernel = np.array([[0.5, -1.5], [1.0, -1.0], [1.5, -0.5]], "float32")
        _, scale, zero = quantizers.abs_max_quantize_grouped_with_zero_point(
            kernel, block_size=3, to_numpy=to_numpy
        )
        self.assertAllClose(scale, [[1.5 / 15, 1.5 / 15]])
        self.assertAllClose(zero, [[-8, 7]])

    @parameterized.named_parameters(
        ("int4", (-8, 7)),
        ("uint4", (0, 15)),
        ("int3", (-4, 3)),
        ("int2", (-2, 1)),
        ("int8", (-128, 127)),
    )
    def test_grouped_paths_agree(self, value_range):
        # The NumPy path and the backend path apply one formula, for any
        # code range, with a padded last group and an all-zero group.
        rng = np.random.default_rng(0)
        kernel = rng.standard_normal((11, 3)).astype("float32")
        kernel[4:8, 1] = 0.0
        results = []
        for to_numpy in (True, False):
            outputs = quantizers.abs_max_quantize_grouped_with_zero_point(
                kernel,
                block_size=4,
                value_range=value_range,
                to_numpy=to_numpy,
            )
            results.append([ops.convert_to_numpy(t) for t in outputs])
        for numpy_result, tensor_result in zip(*results):
            self.assertAllClose(numpy_result, tensor_result)
        codes, _, zero = results[0]
        low, high = value_range
        self.assertTrue(low <= codes.min() and codes.max() <= high)
        self.assertTrue(low <= zero.min() and zero.max() <= high)


class Int4QuantizationConfigTest(testing.TestCase):
    def test_default_block_size(self):
        """Test that default block_size is 128."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig()
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.mode, "int4")

    @parameterized.named_parameters(
        ("block_32", 32),
        ("block_64", 64),
        ("block_128", 128),
        ("block_256", 256),
    )
    def test_custom_block_size(self, block_size):
        """Test setting custom block_size values."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=block_size)
        self.assertEqual(config.block_size, block_size)

    def test_per_channel_mode_with_none(self):
        """Test per-channel mode with block_size=None."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=None)
        self.assertIsNone(config.block_size)

    def test_per_channel_mode_with_negative_one(self):
        """Test per-channel mode with block_size=-1."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=-1)
        self.assertEqual(config.block_size, -1)

    def test_invalid_block_size_raises(self):
        """Test that invalid block_size values raise ValueError."""
        from keras.src.quantizers import Int4QuantizationConfig

        with self.assertRaisesRegex(ValueError, "block_size must be"):
            Int4QuantizationConfig(block_size=0)

        with self.assertRaisesRegex(ValueError, "block_size must be"):
            Int4QuantizationConfig(block_size=-2)

    def test_get_config_includes_block_size(self):
        """Test that get_config includes block_size."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=64)
        serialized = config.get_config()
        self.assertEqual(serialized["block_size"], 64)

    def test_from_config_restores_block_size(self):
        """Test that from_config restores block_size."""
        from keras.src.quantizers import Int4QuantizationConfig

        original = Int4QuantizationConfig(block_size=64)
        serialized = original.get_config()
        restored = Int4QuantizationConfig.from_config(serialized)
        self.assertEqual(restored.block_size, 64)

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=128)
        serialized = quantizers.serialize(config)
        deserialized = quantizers.deserialize(serialized)
        self.assertEqual(deserialized.block_size, 128)
        self.assertEqual(deserialized.mode, "int4")

    def test_serialization_with_per_channel(self):
        """Test serialization with per-channel mode."""
        from keras.src.quantizers import Int4QuantizationConfig

        config = Int4QuantizationConfig(block_size=None)
        serialized = quantizers.serialize(config)
        deserialized = quantizers.deserialize(serialized)
        self.assertIsNone(deserialized.block_size)


class ZeroPointPrimitivesTest(testing.TestCase):
    @parameterized.named_parameters(
        ("bits_2_sym_False", 2, False),
        ("bits_4_sym_False", 4, False),
        ("bits_8_sym_False", 8, False),
        ("bits_2_sym_True", 2, True),
        ("bits_4_sym_True", 4, True),
        ("bits_8_sym_True", 8, True),
    )
    def test_quantize_dequantize_roundtrip_error_bound_per_tensor(
        self, bits, symmetric
    ):
        """
        For finite inputs and positive scales, the reconstruction error
        |x_hat - clip(x)| is bounded by 0.5 * scale elementwise.
        """
        rng = np.random.default_rng(0)
        x = ops.array(rng.standard_normal((64, 32)), "float32")
        scale = ops.array(0.05)  # per-tensor scale
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(maxq / 2.0 if symmetric else 3.0, "float32")

        quantized = quantize_with_zero_point(x, scale, zero, maxq)
        dequantized = dequantize_with_zero_point(quantized, scale, zero)

        # Representable dequantization range:
        # [scale*(0 - zero), scale*(maxq - zero)]
        lo = ops.multiply(scale, ops.subtract(ops.array(0.0), zero))
        hi = ops.multiply(scale, ops.subtract(maxq, zero))
        x_clipped = ops.clip(x, lo, hi)

        err = ops.abs(dequantized - x_clipped)
        self.assertTrue(
            ops.all(err <= (ops.add(ops.multiply(0.5, scale), 1e-7)))
        )

    def test_quantize_clipping_behavior_extremes(self):
        """
        Very negative q == 0 ; very positive q == maxq.
        """
        maxq = ops.array(15.0)
        scale = ops.array(0.1)
        zero = ops.array(7.0)

        x = ops.array([[-1e6, 1e6]], "float32")
        quantized = quantize_with_zero_point(x, scale, zero, maxq)

        self.assertEqual(quantized.shape, (1, 2))
        self.assertEqual(quantized[0, 0], 0.0)
        self.assertEqual(quantized[0, 1], maxq)

    def test_zero_scale_guard_no_nans_for_finite_inputs(self):
        """
        If scale == 0, quantize should not produce NaNs (uses epsilon
        replacement).
        """
        x = ops.array([[0.0, 1.0, -2.0]])
        scale = ops.array(0.0)  # triggers epsilon path
        zero = ops.array(5.0)
        maxq = ops.array(15.0)

        q = quantize_with_zero_point(x, scale, zero, maxq)
        self.assertFalse(ops.any(ops.isnan(q)))

        # Dequantize should also be finite
        x_hat = dequantize_with_zero_point(q, scale, zero)
        self.assertTrue(ops.all(ops.isfinite(x_hat)))

    @parameterized.parameters(4, 8)
    def test_idempotent_quantize_when_input_is_already_levels(self, bits):
        """
        If input is already exactly on representable dequantized grid,
        quantize→dequantize should return the same values (within float eps).
        """
        scale = ops.array(0.125)
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(ops.divide(maxq, 2.0))

        # Build dequantized grid points: x = scale * (k - zero), k in [0..maxq]
        ks = ops.arange(0, ops.add(maxq, 1))
        x_vals = ops.multiply(scale, ops.subtract(ks, zero))
        x = ops.reshape(x_vals, (1, -1))

        q = quantize_with_zero_point(x, scale, zero, maxq)
        x_hat = dequantize_with_zero_point(q, scale, zero)
        self.assertAllClose(x_hat, x, rtol=0, atol=1e-6)


class ComputeScaleZeroTest(testing.TestCase):
    def test_error_when_x_is_none(self):
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            compute_quantization_parameters(None, bits=4)

    def test_error_when_x_is_empty(self):
        x = ops.array([[], []], "float32")  # 2D empty tensor
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            compute_quantization_parameters(x, bits=4)

    def test_error_when_weight_rank_too_low(self):
        x = ops.array([1.0, 2.0], "float32")  # rank-1
        with self.assertRaisesRegex(ValueError, "rank of at least 2"):
            compute_quantization_parameters(x, bits=4)

    @parameterized.named_parameters(
        ("bits2_asym", 2, False),
        ("bits4_asym", 4, False),
        ("bits8_asym", 8, False),
        ("bits2_sym", 2, True),
        ("bits4_sym", 4, True),
        ("bits8_sym", 8, True),
    )
    def test_per_tensor_shapes_and_basic_invariants(self, bits, symmetric):
        """Test per-tensor shapes and basic invariants."""
        x = ops.array(
            np.random.default_rng(0).standard_normal((7, 5), dtype="float32")
        )
        scale, zero, maxq = compute_quantization_parameters(
            x, bits=bits, symmetric=symmetric, per_channel=False
        )

        # Shapes (per-tensor with weight semantics): (out_features, 1)
        self.assertEqual(scale.shape, (7, 1))
        self.assertEqual(zero.shape, (7, 1))

        # Scale must be strictly positive
        self.assertTrue(ops.all(scale > 0.0))

        # All elements in the scale and zero tensors must be equal due to
        # tiling for per-tensor quantization
        self.assertTrue(ops.all(scale == scale[0, 0]))
        self.assertTrue(ops.all(zero == zero[0, 0]))

    def test_per_tensor_symmetric_on_constant_input_uses_safe_range(self):
        """Ensures safe range adjustment if entries are equal"""
        x = ops.array(np.full((3, 4), 0.0, dtype=np.float32))
        scale, zero, maxq = compute_quantization_parameters(
            x, bits=4, symmetric=True, per_channel=False
        )
        # With symmetric=True and constant input, zero = (maxq+1)/2
        # Shape is now (3, 1) due to weight semantics
        expected_zero = ops.array((float(maxq) + 1.0) / 2.0)
        self.assertAllClose(zero[0, 0], expected_zero)
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

    def test_weight_per_tensor_tiles_rows(self):
        """Tests that scales/zeros tensors are properly tiled when
        per-channel quantization is not used."""
        x = ops.array(
            np.random.default_rng(1).standard_normal((8, 16)), "float32"
        )
        scale, zero, _ = compute_quantization_parameters(
            x, bits=4, symmetric=False, per_channel=False
        )
        # With per_channel=False, shapes are (rows, 1)
        self.assertEqual(scale.shape, (8, 1))
        self.assertEqual(zero.shape, (8, 1))

        # All elements in the scale and zero tensors must be equal due to
        # tiling.
        self.assertTrue(ops.all(scale == scale[0, 0]))
        self.assertTrue(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_ungrouped_shapes(self):
        """Tests that scales/zeros tensors have the correct shape when
        per-channel quantization is used without grouping."""
        x = ops.array(
            np.random.default_rng(2).standard_normal((6, 10)), "float32"
        )
        scale, zero, _ = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
        )
        # Per-channel (ungrouped): one scale per output row -> (rows, 1)
        self.assertEqual(scale.shape, (6, 1))
        self.assertEqual(zero.shape, (6, 1))
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

        # Each channel should have roughly unique scales and zeros
        self.assertFalse(ops.all(scale == scale[0, 0]))
        self.assertFalse(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_grouped_shapes_and_count(self):
        """Tests that scales/zeros have the correct shape and count when
        per-channel quantization is used with grouping."""
        out_features, in_features, group_size = 8, 16, 4
        x = ops.array(
            np.random.default_rng(3).standard_normal(
                (out_features, in_features)
            ),
            "float32",
        )
        scale, zero, _ = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
        )
        # Grouped path produces [out_features, n_groups] shape
        n_groups = in_features // group_size
        self.assertEqual(scale.shape, (out_features, n_groups))
        self.assertEqual(zero.shape, (out_features, n_groups))
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

    @parameterized.named_parameters(
        ("bits2", 2),
        ("bits4", 4),
        ("bits8", 8),
    )
    def test_unsigned_asymmetric_zero_point_stays_in_range(self, bits):
        """Zero point must stay in [0, maxq] for one-sided weight groups.

        Regression test: the unsigned asymmetric branch used to compute
        `zero = round(-min / scale)` without clamping the range to include
        zero, so all-negative groups produced zero points far above `maxq`
        (unrepresentable in `bits`-bit packed export formats), and the
        quantized grid could not represent 0. Reference GPTQ/AWQ clamp with
        `xmin = min(xmin, 0)`, `xmax = max(xmax, 0)`.
        """
        maxq = 2**bits - 1
        rng = np.random.default_rng(5)

        # All-negative groups: without the range clamp, zero overflows.
        x_negative = ops.array((-1.0 - rng.random((8, 16))).astype("float32"))
        scale, zero, _ = compute_quantization_parameters(
            x_negative,
            bits=bits,
            symmetric=False,
            per_channel=True,
            group_size=8,
        )
        zero_np = ops.convert_to_numpy(zero)
        self.assertTrue(np.all(zero_np >= 0))
        self.assertTrue(np.all(zero_np <= maxq))
        # With max clamped to 0, the top of the grid is exactly 0:
        # zero == maxq and (maxq - zero) * scale == 0.
        self.assertTrue(np.all(zero_np == maxq))

        # All-positive groups: zero must be 0 (bottom of the grid is 0).
        x_positive = ops.array((1.0 + rng.random((8, 16))).astype("float32"))
        _, zero, _ = compute_quantization_parameters(
            x_positive,
            bits=bits,
            symmetric=False,
            per_channel=True,
            group_size=8,
        )
        zero_np = ops.convert_to_numpy(zero)
        self.assertTrue(np.all(zero_np == 0))

        # The clamped range must still cover the original values.
        scale_np = ops.convert_to_numpy(scale)
        min_np = ops.convert_to_numpy(
            ops.min(ops.reshape(x_negative, (8, 2, 8)), axis=2)
        )
        # Lowest representable value (q=0): (0 - zero) * scale <= min.
        self.assertTrue(np.all(-maxq * scale_np <= min_np + 1e-6))

    @parameterized.named_parameters(
        ("sym_true", True),
        ("sym_false", False),
    )
    def test_dtype_and_finiteness(self, symmetric):
        x = ops.array(
            np.random.default_rng(4).standard_normal((5, 7)).astype("float32")
        )
        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=8,
            symmetric=symmetric,
            per_channel=True,
            group_size=-1,
        )
        # All outputs should be all finite
        self.assertTrue(ops.all(ops.isfinite(scale)))
        self.assertTrue(ops.all(ops.isfinite(zero)))
        self.assertTrue(ops.all(ops.isfinite(maxq)))

    def test_dequantize_with_sz_map_logic(self):
        """Validates the vectorized dequantization logic against a
        manual implementation."""
        out_features, in_features, group_size = 4, 16, 4
        n_groups = in_features // group_size

        # Create dummy quantized weights
        q_weights = ops.cast(
            ops.array(
                np.random.randint(0, 15, size=(out_features, in_features))
            ),
            "uint8",
        )

        # Create dummy scales and zeros
        scale = ops.abs(
            ops.array(
                np.random.random((out_features, n_groups)).astype("float32")
            )
        )
        zero = ops.cast(
            ops.array(np.random.randint(0, 15, size=(out_features, n_groups))),
            "uint8",
        )

        # Create group index mapping
        g_idx = ops.array(np.arange(in_features) // group_size, dtype="int32")

        # Get the result from the function under test
        dequantized_result = dequantize_with_sz_map(
            q_weights, scale, zero, g_idx
        )

        # Manually compute the expected result
        expected_dequantized = np.zeros(
            (out_features, in_features), dtype="float32"
        )

        for i in range(out_features):
            for j in range(in_features):
                group = g_idx[j]
                s = scale[i, group]
                z = zero[i, group]
                # Dequantization formula: (q_val - z) * s
                expected_dequantized[i, j] = ops.multiply(
                    ops.subtract(q_weights[i, j], ops.cast(z, "float32")), s
                )

        self.assertAllClose(dequantized_result, expected_dequantized)

    def test_quantize_with_sz_map_logic(self):
        """Validates the vectorized quantization logic against a
        manual implementation."""
        out_features, in_features, group_size = 4, 16, 4
        n_groups = in_features // group_size

        # Create dummy float weights
        weights = ops.array(
            np.random.default_rng(5).standard_normal(
                (out_features, in_features)
            ),
            "float32",
        )

        # Create dummy scales and zeros
        scale = ops.abs(
            ops.array(
                np.random.random((out_features, n_groups)).astype("float32")
            )
        )
        zero = ops.cast(
            ops.array(np.random.randint(0, 15, size=(out_features, n_groups))),
            "uint8",
        )

        maxq = ops.array(15.0)

        # Create group index mapping
        g_idx = ops.array(np.arange(in_features) // group_size, dtype="int32")

        # Get the result from the function under test
        quantized_result = quantize_with_sz_map(
            weights, scale, zero, g_idx, maxq
        )

        # Manually compute the expected result
        expected_quantized = np.zeros(
            (out_features, in_features), dtype="uint8"
        )

        for i in range(out_features):
            for j in range(in_features):
                group = g_idx[j]
                s = scale[i, group]
                z = zero[i, group]
                # Quantization formula: clip(round(x/s + z), 0, maxq)
                q_val = ops.round(ops.add(ops.divide(weights[i, j], s), z))
                q_val_clipped = ops.clip(q_val, 0.0, maxq)
                expected_quantized[i, j] = ops.cast(q_val_clipped, "uint8")

        self.assertAllClose(quantized_result, expected_quantized)


class GroupedQuantizationParametersTest(testing.TestCase):
    """Test grouped weight quantization in compute_quantization_parameters."""

    def test_grouped_weight_shapes_divisible(self):
        """Test grouped quantization with divisible dimensions."""
        out_features, in_features, group_size = 64, 128, 32
        n_groups = in_features // group_size  # 4

        x = ops.array(
            np.random.randn(out_features, in_features).astype("float32")
        )

        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
        )

        self.assertEqual(scale.shape, (out_features, n_groups))
        self.assertEqual(zero.shape, (out_features, n_groups))
        self.assertEqual(float(maxq), 15.0)

    def test_grouped_weight_shapes_non_divisible(self):
        """Test grouped quantization with non-divisible dimensions."""
        out_features, in_features, group_size = 32, 100, 32
        n_groups = (in_features + group_size - 1) // group_size  # 4

        x = ops.array(
            np.random.randn(out_features, in_features).astype("float32")
        )

        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
        )

        self.assertEqual(scale.shape, (out_features, n_groups))
        self.assertEqual(zero.shape, (out_features, n_groups))

    def test_grouped_returns_3_values(self):
        """Test that grouped quantization returns exactly 3 values."""
        x = ops.array(np.random.randn(32, 64).astype("float32"))

        result = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=16,
        )

        # Should return exactly 3 values
        self.assertEqual(len(result), 3)
        scale, zero, maxq = result
        self.assertEqual(scale.shape, (32, 4))
        self.assertEqual(zero.shape, (32, 4))

    def test_single_group_per_channel_semantics(self):
        """Test that single group slice uses per-channel semantics."""
        out_features, in_features = 32, 16
        group_size = 16  # in_features == group_size

        x = ops.array(
            np.random.randn(out_features, in_features).astype("float32")
        )

        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
        )

        # Single group should produce per-channel output shape
        # n_groups = 1, so shape is [out_features, 1]
        self.assertEqual(scale.shape, (out_features, 1))
        self.assertEqual(zero.shape, (out_features, 1))

    def test_grouped_no_nan_inf(self):
        """Test grouped quantization produces no NaN/Inf."""
        x = ops.array(np.random.randn(64, 128).astype("float32"))

        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=4,
            per_channel=True,
            group_size=32,
        )

        self.assertFalse(ops.any(ops.isnan(scale)))
        self.assertFalse(ops.any(ops.isinf(scale)))

    def test_grouped_various_group_sizes(self):
        """Test grouped quantization with various group sizes."""
        out_features, in_features = 64, 128

        for group_size in [8, 16, 32, 64]:
            n_groups = (in_features + group_size - 1) // group_size
            x = ops.array(
                np.random.randn(out_features, in_features).astype("float32")
            )

            scale, zero, maxq = compute_quantization_parameters(
                x,
                bits=4,
                per_channel=True,
                group_size=group_size,
            )

            self.assertEqual(
                scale.shape,
                (out_features, n_groups),
                f"Failed for group_size={group_size}",
            )


class TernarizeTest(testing.TestCase):
    def test_default_threshold_returns_mean(self):
        kernel = ops.array([[0.5, -0.5], [0.1, -1.5]], "float32")
        codes, scale = ternarize(kernel)
        # threshold = 0.5 * mean(|W|) = 0.325, so 0.1 maps to 0.
        self.assertAllClose(codes, [[1.0, -1.0], [0.0, -1.0]])
        self.assertAllClose(scale, 0.65)

    def test_fixed_threshold_returns_unit_scale(self):
        kernel = ops.array([[0.5, -0.5], [0.1, -1.5]], "float32")
        codes, scale = ternarize(kernel, threshold=0.2)
        self.assertAllClose(codes, [[1.0, -1.0], [0.0, -1.0]])
        self.assertEqual(scale, 1.0)

    def test_bfloat16_kernel_takes_mean_in_float32(self):
        # A bfloat16 mean over many weights saturates when it accumulates
        # in bfloat16 (as NumPy does), and a 0-d bfloat16 tensor does not
        # convert to NumPy on TensorFlow.
        values = np.random.RandomState(0).randn(512, 512)
        kernel = ops.cast(values, "bfloat16")
        codes, scale = ternarize(kernel)
        exact = ops.convert_to_numpy(ops.cast(kernel, "float32"))
        beta = np.mean(np.abs(exact), dtype="float64")
        self.assertAllClose(scale, beta, rtol=1e-5)
        expected = np.sign(exact) * (np.abs(exact) > 0.5 * beta)
        self.assertAllClose(np.asarray(codes, "float32"), expected)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Records the torch dispatch."
)
class HostDataTest(testing.TestCase):
    """The helpers that quantized layers run on each call use no host data.

    `host_data_ops` finds each op that makes a tensor from host data, on
    every torch device, and each op that reads or returns a CPU tensor, on
    a GPU or MPS device. Torch copies host data to the device on each
    call, and a CUDA graph capture refuses the copy.
    """

    @parameterized.named_parameters(
        ("int4_int8", 4, "int8"),
        ("int4_uint8", 4, "uint8"),
        ("int2_int8", 2, "int8"),
        ("int2_uint8", 2, "uint8"),
    )
    def test_unpack_uses_no_host_data(self, bits, dtype):
        packed = ops.convert_to_tensor(
            np.arange(256, dtype="uint8").astype(dtype).reshape(16, 16)
        )
        unpack = quantizers.unpack_int4 if bits == 4 else quantizers.unpack_int2
        # An odd length also trims the padding.
        orig_len = 16 * (8 // bits) - 1
        found = host_data_ops(
            lambda: unpack(packed, orig_len, axis=0, dtype=dtype)
        )
        self.assertEqual(found, [])

    def test_unpack_ternary_uses_no_host_data(self):
        packed = ops.convert_to_tensor(
            np.arange(243, dtype="uint8").reshape(27, 9)
        )
        found = host_data_ops(
            lambda: quantizers.unpack_ternary(packed, 27 * 5 - 2, axis=0)
        )
        self.assertEqual(found, [])

    @parameterized.named_parameters(
        ("float32", "float32"), ("bfloat16", "bfloat16"), ("float16", "float16")
    )
    def test_activation_quantizer_uses_no_host_data(self, dtype):
        # The int8 layers quantize their inputs with this quantizer.
        values = ops.convert_to_tensor(
            np.random.RandomState(0).randn(4, 32), dtype
        )
        found = host_data_ops(
            lambda: quantizers.AbsMaxQuantizer()(values, axis=-1)
        )
        self.assertEqual(found, [])

    def test_float8_training_step_uses_no_host_data(self):
        # The forward and the backward pass update the scales and the amax
        # histories.
        if ops.convert_to_tensor(0).device.type == "mps":
            self.skipTest("MPS has no float8 dtypes.")
        layer = layers.Dense(8)
        layer.build((None, 16))
        layer.quantize("float8")
        x = ops.convert_to_tensor(
            np.random.RandomState(0).randn(4, 16), "float32"
        )

        def step():
            ops.sum(layer(x, training=True)).backward()

        self.assertEqual(host_data_ops(step), [])
