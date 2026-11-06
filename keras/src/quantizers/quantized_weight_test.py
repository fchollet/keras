from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import packing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.int8 import Int8Strategy
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantized_weight import Int2Quads
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import TernaryTrits
from keras.src.quantizers.quantized_weight import WeightScheme

DIVISOR = WeightScheme(code_range=(-127, 127), scale_form="divisor")
MULTIPLIER = WeightScheme(code_range=(-127, 127), scale_form="multiplier")
GROUPED = WeightScheme(
    code_range=(0, 15),
    scale_form="multiplier",
    has_zero_point=True,
    group_size=2,
)


def _grouped_view(**kwargs):
    # Two groups of two rows; group 1 has a non-zero zero point.
    return QuantizedWeight(
        codes=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], "uint8"),
        scale=np.array([[1.0, 0.5], [2.0, 0.25]], "float32"),
        layout=NoPack(),
        scheme=GROUPED,
        shape=(4, 2),
        axis=0,
        zero_point=np.array([[0, 0], [1, 2]], "uint8"),
        g_idx=np.array([0, 0, 1, 1], "float32"),
        **kwargs,
    )


class WeightSchemeTest(testing.TestCase):
    def test_scale_form_is_required_and_validated(self):
        with self.assertRaises(TypeError):
            WeightScheme(code_range=(-8, 7))
        with self.assertRaisesRegex(ValueError, "scale_form"):
            WeightScheme(code_range=(-8, 7), scale_form="reciprocal")

    def test_grouped_scheme_needs_a_multiplier_and_a_zero_point(self):
        # The grouped dequantization multiplies by the scale and subtracts
        # the zero point.
        for scale_form, has_zero_point in (
            ("divisor", True),
            ("multiplier", False),
        ):
            with self.assertRaisesRegex(ValueError, "grouped scheme"):
                WeightScheme(
                    code_range=(0, 15),
                    scale_form=scale_form,
                    has_zero_point=has_zero_point,
                    group_size=4,
                )


class PackLayoutTest(testing.TestCase):
    @parameterized.named_parameters(
        ("int4_axis0", Int4Pairs, packing.pack_int4, 0, (-8, 7), "int8"),
        ("int4_axis1", Int4Pairs, packing.pack_int4, 1, (-8, 7), "int8"),
        ("int4_uint8", Int4Pairs, packing.pack_int4, 0, (0, 15), "uint8"),
        ("int2_axis0", Int2Quads, packing.pack_int2, 0, (-2, 1), "int8"),
        ("int2_uint8", Int2Quads, packing.pack_int2, 1, (0, 3), "uint8"),
    )
    def test_unpack_inverts_the_pack_function(
        self, layout_cls, pack, axis, value_range, dtype
    ):
        rng = np.random.default_rng(0)
        # Odd lengths exercise the padding on every axis.
        codes = rng.integers(value_range[0], value_range[1] + 1, (5, 7))
        codes = codes.astype(dtype)
        packed, _, _ = pack(codes, axis=axis, dtype=dtype)
        layout = layout_cls(axis=axis, orig_len=codes.shape[axis])
        self.assertEqual(
            packed.shape[axis], layout.packed_length(codes.shape[axis])
        )
        self.assertAllClose(layout.unpack(packed), codes)

    def test_ternary_round_trip(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(-1, 2, (11, 3)).astype("int8")
        packed, _, _ = packing.pack_ternary(codes, axis=0)
        layout = TernaryTrits(axis=0, orig_len=11)
        self.assertEqual(tuple(packed.shape), (3, 3))
        self.assertEqual(layout.packed_length(11), 3)
        self.assertAllClose(layout.unpack(packed), codes)

    def test_ternary_pack_inverts_unpack(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(-1, 2, (3, 11)).astype("int8")
        layout = TernaryTrits(axis=-1, orig_len=11)
        packed = layout.pack(codes)
        expected, _, _ = packing.pack_ternary(codes, axis=-1)
        self.assertAllEqual(packed, expected)
        self.assertAllEqual(layout.unpack(packed), codes)


class QuantizedWeightTest(testing.TestCase):
    def test_validates_variables_against_scheme(self):
        codes = np.zeros((4, 2), "int8")
        scale = np.ones((2,), "float32")
        with self.assertRaisesRegex(ValueError, "zero_point"):
            QuantizedWeight(
                codes=codes,
                scale=scale,
                layout=NoPack(),
                scheme=DIVISOR,
                shape=(4, 2),
                axis=0,
                zero_point=np.zeros((2,), "int8"),
            )
        with self.assertRaisesRegex(ValueError, "g_idx"):
            QuantizedWeight(
                codes=codes,
                scale=scale,
                layout=NoPack(),
                scheme=GROUPED,
                shape=(4, 2),
                axis=0,
                zero_point=np.zeros((2,), "int8"),
            )
        with self.assertRaisesRegex(ValueError, "align_scale"):
            QuantizedWeight(
                codes=codes,
                scale=scale,
                layout=NoPack(),
                scheme=DIVISOR,
                shape=(4, 2),
                axis=0,
                align_scale=ops.transpose,
            )

    def test_grouped_view_needs_one_integer_axis(self):
        # `g_idx` maps the positions along one axis to their groups.
        view = _grouped_view()
        for axis in (None, (0,)):
            with self.assertRaisesRegex(ValueError, "axis"):
                QuantizedWeight(
                    codes=view.codes,
                    scale=view.scale,
                    layout=NoPack(),
                    scheme=GROUPED,
                    shape=(4, 2),
                    axis=axis,
                    zero_point=view.zero_point,
                    g_idx=view.g_idx,
                )

    def test_input_scales_need_one_integer_axis(self):
        # `input_scales` has one entry per position along one axis.
        for axis in (None, (0,)):
            with self.assertRaisesRegex(ValueError, "input_scales"):
                QuantizedWeight(
                    codes=np.zeros((4, 2), "int8"),
                    scale=np.ones((2,), "float32"),
                    layout=NoPack(),
                    scheme=DIVISOR,
                    shape=(4, 2),
                    axis=axis,
                    input_scales=np.ones((4,), "float32"),
                )

    @parameterized.named_parameters(
        ("shared_along_rows", 0, [[1.0, 1.0], [3.0, 2.0]]),
        ("shared_along_columns", -1, [[1.0, 2.0], [1.5, 2.0]]),
    )
    def test_divisor_scale(self, axis, expected):
        # Scale `[2, 4]`: along rows it divides each column, along columns
        # each row.
        view = QuantizedWeight(
            codes=np.array([[2, 4], [6, 8]], "int8"),
            scale=np.array([2.0, 4.0], "float32"),
            layout=NoPack(),
            scheme=DIVISOR,
            shape=(2, 2),
            axis=axis,
        )
        self.assertAllClose(view.dequantize("float32"), expected)

    @parameterized.named_parameters(
        ("divisor", "divisor", [[-0.5, 0.0, 0.5]]),
        ("multiplier", "multiplier", [[-2.0, 0.0, 2.0]]),
    )
    def test_scalar_scale(self, scale_form, expected):
        view = QuantizedWeight(
            codes=np.array([[-1, 0, 1]], "int8"),
            scale=np.float32(2.0),
            layout=NoPack(),
            scheme=WeightScheme(code_range=(-1, 1), scale_form=scale_form),
            shape=(1, 3),
        )
        self.assertAllClose(view.dequantize("float32"), expected)

    def test_align_scale_lays_out_a_stored_scale(self):
        # Stored as a column; the layer's rule turns it into a row.
        view = QuantizedWeight(
            codes=np.array([[2, 4, 6], [8, 10, 12]], "int8"),
            scale=np.array([[2.0], [4.0], [6.0]], "float32"),
            layout=NoPack(),
            scheme=DIVISOR,
            shape=(2, 3),
            align_scale=ops.transpose,
        )
        self.assertAllClose(
            view.dequantize("float32"), [[1.0, 1.0, 1.0], [4.0, 2.5, 2.0]]
        )

    def test_grouped_multiplier_scale(self):
        # `(code - zero_point) * scale`, with the group of each row.
        self.assertAllClose(
            _grouped_view().dequantize("float32"),
            [[1.0, 1.0], [3.0, 2.0], [8.0, 1.0], [12.0, 1.5]],
        )

    def test_input_scales_divide_each_position_along_axis(self):
        view = _grouped_view(
            input_scales=np.array([1.0, 2.0, 4.0, 8.0], "float32")
        )
        self.assertAllClose(
            view.dequantize("float32"),
            [[1.0, 1.0], [1.5, 1.0], [2.0, 0.25], [1.5, 0.1875]],
        )

    @parameterized.product(
        case=(
            "divisor",
            "scalar",
            "scalar_multiplier",
            "align_scale",
            "grouped",
            "input_scales",
        ),
        dtype=("float32", "bfloat16", "float16"),
    )
    def test_dequantize_returns_the_requested_dtype(self, case, dtype):
        # The stored scales are float32, as under a mixed precision policy.
        codes = np.array([[2, 4], [6, 8], [1, 3], [5, 7]], "int8")
        if case == "grouped":
            view = _grouped_view()
        elif case == "input_scales":
            view = _grouped_view(input_scales=np.ones((4,), "float32"))
        else:
            view = QuantizedWeight(
                codes=codes,
                scale={
                    "divisor": np.ones((2,), "float32"),
                    "scalar": np.float32(2.0),
                    "scalar_multiplier": np.float32(2.0),
                    "align_scale": np.ones((2, 1), "float32"),
                }[case],
                layout=NoPack(),
                scheme=MULTIPLIER if case == "scalar_multiplier" else DIVISOR,
                shape=(4, 2),
                axis=0 if case == "divisor" else None,
                align_scale=ops.transpose if case == "align_scale" else None,
            )
        self.assertDType(view.dequantize(dtype), dtype)

    def test_shape_restores_an_nd_weight(self):
        view = QuantizedWeight(
            codes=np.arange(12, dtype="int8").reshape(4, 3),
            scale=np.ones((3,), "float32"),
            layout=NoPack(),
            scheme=DIVISOR,
            shape=(2, 2, 3),
            axis=0,
        )
        self.assertEqual(tuple(view.unpack().shape), (2, 2, 3))
        self.assertEqual(tuple(view.dequantize("float32").shape), (2, 2, 3))
        self.assertIn("shape=(2, 2, 3)", repr(view))

    def test_with_tensors_takes_read_tensors_in_order(self):
        view = _grouped_view(
            input_scales=np.array([1.0, 2.0, 4.0, 8.0], "float32")
        )
        tensors = view.read_tensors()
        self.assertEqual(len(tensors), 5)
        for tensor, stored in zip(
            tensors,
            (
                view.codes,
                view.scale,
                view.zero_point,
                view.g_idx,
                view.input_scales,
            ),
        ):
            self.assertAllEqual(tensor, stored)
        rebuilt = view.with_tensors(tensors)
        names = ("codes", "scale", "zero_point", "g_idx", "input_scales")
        for name, tensor in zip(names, tensors):
            self.assertIs(getattr(rebuilt, name), tensor)
        self.assertEqual(repr(rebuilt), repr(view))
        self.assertAllEqual(
            rebuilt.dequantize("float32"), view.dequantize("float32")
        )
        with self.assertRaises(ValueError):
            view.with_tensors(tensors[:4])

    def test_read_tensors_reads_variables_in_the_autocast_scope(self):
        # A forward pass reads the tensors it hands to a custom gradient
        # inside the autocast scope, where a float variable reads as the
        # compute dtype.
        view = QuantizedWeight(
            codes=backend.Variable(
                np.ones((4, 2)), dtype="int8", trainable=False
            ),
            scale=backend.Variable(
                np.ones((2,)), dtype="float32", trainable=False
            ),
            layout=NoPack(),
            scheme=DIVISOR,
            shape=(4, 2),
            axis=0,
        )
        with AutocastScope("bfloat16"):
            codes, scale = view.read_tensors()
        self.assertDType(codes, "int8")
        self.assertDType(scale, "bfloat16")
        self.assertNotIsInstance(scale, backend.Variable)


def _quantized(layer, build_shape, mode, config=None):
    """Builds `layer`, sets its weight in `[-1, 1]` and quantizes it."""
    layer.build(build_shape)
    if isinstance(layer, layers.Embedding):
        weight = layer._embeddings
    else:
        weight = layer._kernel
    rng = np.random.default_rng(0)
    reference = rng.uniform(-1, 1, tuple(weight.shape)).astype("float32")
    weight.assign(reference)
    layer.quantize(mode, config=config)
    return layer, reference


class LayerViewTest(testing.TestCase):
    """The view each mode builds reads the layer's stored variables."""

    @parameterized.named_parameters(
        # Half the largest code step of a kernel in `[-1, 1]`.
        ("dense_int8", "dense", "int8", -1, 0.5 / 127),
        ("dense_int4_per_channel", "dense", "int4", -1, 0.5 / 7),
        ("dense_int4_grouped", "dense", "int4", 4, 1 / 15),
        ("einsum_int8", "einsum", "int8", -1, 0.5 / 127),
        ("einsum_int4_grouped", "einsum", "int4", 2, 1 / 15),
        ("embedding_int8", "embedding", "int8", -1, 0.5 / 127),
        ("embedding_int4_per_channel", "embedding", "int4", -1, 0.5 / 7),
        ("embedding_int4_grouped", "embedding", "int4", 2, 1 / 15),
    )
    def test_dequantize_reconstructs_the_float_weight(
        self, kind, mode, block_size, atol
    ):
        if kind == "dense":
            layer, build_shape = layers.Dense(5), (None, 7)
        elif kind == "einsum":
            layer = layers.EinsumDense(
                "btd,dnh->btnh", output_shape=(None, 2, 3), bias_axes=None
            )
            build_shape = (None, 4, 6)
        else:
            layer, build_shape = layers.Embedding(9, 4), None
        config = None
        if mode == "int4":
            config = Int4QuantizationConfig(block_size=block_size)
        layer, reference = _quantized(layer, build_shape, mode, config)
        weight = layer._quantized_weight().dequantize("float32")
        self.assertEqual(tuple(weight.shape), reference.shape)
        self.assertAllClose(weight, reference, atol=atol * 1.001, rtol=0)

    @parameterized.named_parameters(
        ("int8", "int8", None),
        ("int4_per_channel", "int4", Int4QuantizationConfig(block_size=-1)),
        ("int4_grouped", "int4", Int4QuantizationConfig(block_size=4)),
    )
    def test_encode_is_what_quantize_stores(self, mode, config):
        # `encode` is the exact quantization math behind `quantize()`: fed
        # the float kernel, it must produce the stored form byte for byte.
        layer = layers.Dense(5)
        layer.build((None, 7))
        float_kernel = ops.convert_to_numpy(layer._kernel)
        strategy = strategy_registry.get_strategy(mode)
        codes, scale, zero = strategy.encode(layer, float_kernel, config)

        layer.quantize(mode, config=config)
        self.assertDType(codes, layer._kernel.dtype)
        self.assertAllClose(codes, layer._kernel)
        self.assertAllClose(scale, layer.kernel_scale)
        if zero is None:
            self.assertIsNone(layer._quantized_weight().zero_point)
        else:
            self.assertAllClose(zero, layer.kernel_zero)

    def test_int8_projection_scale_layout(self):
        # A matmul kernel's scale is shared along its input axis; an einsum
        # kernel's is stored for the outputs and the geometry lays it out.
        layer, _ = _quantized(layers.Dense(5), (None, 7), "int8")
        view = layer._quantized_weight()
        self.assertEqual(view.axis, 0)
        self.assertIsNone(view.align_scale)

        layer = layers.EinsumDense(
            "btd,dnh->btnh", output_shape=(None, 2, 3), bias_axes=None
        )
        layer, _ = _quantized(layer, (None, 4, 6), "int8")
        view = layer._quantized_weight()
        self.assertIsNone(view.axis)
        self.assertIsNotNone(view.align_scale)

    @parameterized.named_parameters(
        ("int8", "int8", None),
        ("int4_grouped", "int4", Int4QuantizationConfig(block_size=2)),
    )
    def test_tied_reverse_view_is_the_forward_table_transposed(
        self, mode, config
    ):
        layer = layers.ReversibleEmbedding(9, 4, tie_weights=True)
        layer, _ = _quantized(layer, None, mode, config)
        strategy = strategy_registry.get_strategy(mode)
        reverse = strategy._get_reverse_lookup_quantized_weight(layer, None)
        self.assertEqual(reverse.shape, (4, 9))
        self.assertAllEqual(
            reverse.dequantize("float32"),
            ops.transpose(layer._quantized_weight().dequantize("float32")),
        )

    def test_ternary_view(self):
        layer, reference = _quantized(
            layers.TernaryDense(4), (None, 11), "ternary"
        )
        view = layer._quantized_weight()
        self.assertIsInstance(view.layout, TernaryTrits)
        self.assertEqual(view.scheme.scale_form, "multiplier")
        codes = ops.convert_to_numpy(view.unpack())
        self.assertEqual(codes.shape, (11, 4))
        self.assertTrue(np.isin(codes, [-1, 0, 1]).all())
        # BitNet b1.58: each code stands for `sign * mean(|W|)`.
        beta = np.abs(reference).mean()
        self.assertAllClose(view.dequantize("float32"), codes * beta)

    def test_modes_without_codes_have_no_view(self):
        layer = layers.Dense(3)
        layer.build((None, 4))
        self.assertIsNone(layer._quantized_weight())
        layer.quantize("float8")
        self.assertIsNone(layer._quantized_weight())

    def test_uncalibrated_calibration_mode_has_no_view(self):
        layer = layers.Dense(3)
        layer.build((None, 4))
        layer.quantize(
            "gptq",
            config=GPTQConfig(dataset=None, tokenizer=None, group_size=2),
        )
        self.assertIsNone(layer._quantized_weight())
        # The float kernel is still what the property exposes.
        self.assertEqual(standardize_dtype(layer.kernel.dtype), "float32")


class QuantizationSummaryTest(testing.TestCase):
    def test_summary_counts_logical_parameters_exactly(self):
        # An odd output dim pads the packed int4 kernel; the summary must
        # count the 8 x 5 weights the codes stand for, not the padding.
        inputs = layers.Input([8])
        outputs = layers.Dense(5, name="d")(inputs)
        model = models.Model(inputs, outputs)
        model.quantize(
            "int4", config=Int4QuantizationConfig(block_size=4), verbose=False
        )
        summary = model.quantization_summary(verbose=False)
        self.assertIn("Quantized params : 40", summary)
        # Codes 8 x 3 (24), a 2 x 5 float32 scale (40), a 2 x 5 int8 zero
        # point (10) and an 8-entry float32 `g_idx` (32).
        self.assertIn("weight store : int8 (106 bytes)", summary)
        self.assertIn("160 bytes float32", summary)

    @parameterized.named_parameters(
        # Two 60-byte tables, each with a 10-entry float32 scale (40).
        ("int8", "int8", None, 200),
        # Two packed 30-byte tables, each with a 3 x 10 float32 scale (120)
        # and int8 zero point (30), and one shared 6-entry float32 `g_idx`
        # (24).
        ("int4", "int4", Int4QuantizationConfig(block_size=2), 384),
    )
    def test_summary_counts_both_tables_of_an_untied_lookup(
        self, mode, config, stored_bytes
    ):
        inputs = layers.Input([3], dtype="int32")
        outputs = layers.ReversibleEmbedding(10, 6, tie_weights=False)(inputs)
        model = models.Model(inputs, outputs)
        model.quantize(mode, config=config, verbose=False)
        summary = model.quantization_summary(verbose=False)
        # Two 10 x 6 tables.
        self.assertIn("Quantized params : 120", summary)
        self.assertIn(f"({stored_bytes} bytes)", summary)

    @parameterized.named_parameters(
        # Codes 16 x 4 (64) and an 8-entry float32 scale (32).
        ("int4_per_channel", "int4/-1_from_float32", "int8 (96 bytes)"),
        # Codes (64), a 2 x 8 float32 scale (64), a 2 x 8 zero point (16)
        # and a 16-entry float32 `g_idx` (64).
        ("int4_grouped", "int4/8_from_float32", "int8 (208 bytes)"),
        ("gptq", "gptq/4/8_from_float32", "uint8 (208 bytes)"),
        # The GPTQ store and 16 float32 input scales (64).
        ("awq", "awq/4/8_from_float32", "uint8 (272 bytes)"),
        # The 16 x 8 float32 kernel; not its 1024-entry amax histories.
        (
            "float8",
            "float8_from_float32",
            "float32 (512 bytes, float weight kept)",
        ),
    )
    def test_summary_counts_every_stored_tensor(self, policy, store):
        inputs = layers.Input([16])
        outputs = layers.Dense(8, use_bias=False, dtype=policy)(inputs)
        model = models.Model(inputs, outputs)
        summary = model.quantization_summary(verbose=False)
        self.assertIn(f"weight store : {store}", summary)
        self.assertIn("Quantized params : 128", summary)

    def test_summary_lists_a_layer_that_owns_a_sublayer(self):
        # A `Dense` with a layer activation owns a sub-layer; its own
        # variables still make it a quantized layer.
        model = models.Sequential(
            [
                layers.Input([16]),
                layers.Dense(8, activation=layers.ReLU(), name="d1"),
                layers.Dense(4, name="d2"),
            ]
        )
        model.quantize("int8", verbose=False)
        summary = model.quantization_summary(verbose=False)
        self.assertRegex(summary, r"Layer: \S*d1\n")
        self.assertIn("Quantized layers : 2", summary)
        self.assertIn("Quantized params : 160", summary)

    def test_summary_names_the_policy_of_a_dtype_policy_map_entry(self):
        inputs = layers.Input([6])
        outputs = layers.Dense(
            4, name="d", dtype=dtype_policies.DTypePolicyMap()
        )(inputs)
        model = models.Model(inputs, outputs)
        model.quantize("int8", verbose=False)
        summary = model.quantization_summary(verbose=False)
        self.assertIn("dtype policy : int8_from_float32", summary)
        self.assertNotIn("map_", summary)

    def test_summary_counts_largest_weight_of_a_layer_without_geometry(self):
        class OwnInt8Kernel(layers.Layer):
            # Stores its own int8 kernel and defines no quantization
            # geometry.
            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], 4),
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )
                self.scale = self.add_weight(
                    shape=(4,), initializer="ones", trainable=False
                )

            def call(self, inputs):
                kernel = ops.cast(self.kernel, self.compute_dtype)
                return ops.divide(ops.matmul(inputs, kernel), self.scale)

        inputs = layers.Input([8])
        outputs = OwnInt8Kernel(dtype="int8_from_float32")(inputs)
        model = models.Model(inputs, outputs)
        summary = model.quantization_summary(verbose=False)
        self.assertIn("weight store : int8 (32 bytes)", summary)
        self.assertIn("Quantized params : 32", summary)


class QuantizedProjectionGradientTest(testing.TestCase):
    @parameterized.named_parameters(
        ("dense_int8", "dense", "int8", -1),
        ("dense_int4_per_channel", "dense", "int4", -1),
        ("dense_int4_grouped", "dense", "int4", 4),
        ("einsum_int8", "einsum", "int8", -1),
        ("einsum_int4_per_channel", "einsum", "int4", -1),
        ("einsum_int4_grouped", "einsum", "int4", 4),
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only TensorFlow runs the eager gradient outside autocast.",
    )
    def test_eager_input_gradient_matches_graph(self, kind, mode, block_size):
        # TensorFlow eager runs the custom gradient after the autocast scope
        # closes, so the forward pass must read the scale inside it.
        import tensorflow as tf  # Only this backend runs the test.

        rng = np.random.default_rng(0)
        if kind == "dense":
            layer = layers.Dense(8, dtype="mixed_bfloat16")
            layer.build((None, 16))
            x = rng.standard_normal((3, 16))
        else:
            layer = layers.EinsumDense(
                "btd,ndh->btnh",
                output_shape=(None, 2, 4),
                dtype="mixed_bfloat16",
            )
            layer.build((None, 3, 16))
            x = rng.standard_normal((2, 3, 16))
        config = None
        if mode == "int4":
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        x = tf.constant(x.astype("float32"))

        def input_gradient(inputs):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                y = tf.reduce_sum(tf.cast(layer(inputs), "float32"))
            return tape.gradient(y, inputs)

        self.assertAllEqual(input_gradient(x), tf.function(input_gradient)(x))


class GeometryDispatchTest(testing.TestCase):
    def test_untied_reversible_layer_without_codes_has_no_views(self):
        class NoCodes(Int8Strategy):
            def _get_lookup_quantized_weight(self, layer, geometry):
                return None

            def _get_reverse_lookup_quantized_weight(self, layer, geometry):
                return None

        layer = layers.ReversibleEmbedding(10, 4, tie_weights=False)
        layer.build((None,))
        self.assertEqual(NoCodes().quantized_weights(layer), ())

    @parameterized.named_parameters(
        ("build", "_build_projection"),
        ("call", "_call_projection"),
        ("quantized_weight", "_get_projection_quantized_weight"),
    )
    def test_missing_handler_refuses_before_the_layer_changes(self, handler):
        incomplete = type("Incomplete", (Int8Strategy,), {handler: None})
        layer = layers.Dense(4)
        layer.build((None, 3))
        kernel = ops.convert_to_numpy(layer._kernel)
        modes = {"int8": incomplete()}
        with mock.patch.dict(strategy_registry._MODE_TO_STRATEGY, modes):
            with self.assertRaisesRegex(NotImplementedError, handler):
                layer.quantize("int8")
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertAllEqual(layer._kernel, kernel)

    def test_overridden_verb_needs_no_handler(self):
        class OwnCall(Int8Strategy):
            _call_projection = None

            def call(self, layer, *args, **kwargs):
                return Int8Strategy._call_projection(
                    self, layer, *args, **kwargs
                )

        layer = layers.Dense(4)
        layer.build((None, 3))
        modes = {"int8": OwnCall()}
        with mock.patch.dict(strategy_registry._MODE_TO_STRATEGY, modes):
            layer.quantize("int8")
        self.assertEqual(layer.quantization_mode, "int8")
