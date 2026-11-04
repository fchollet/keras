import math
import os
import re

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import quantizers
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.layers.core.einsum_dense import _analyze_einsum_string
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.saving.saving_api import load_model
from keras.src.testing import test_utils
from keras.src.utils.rng_utils import set_random_seed


class EinsumDenseTest(testing.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", {"axis": 0}, {"axis": -1}),
        (
            "int4",
            "int4",
            {"axis": 0, "value_range": (-8, 7), "output_dtype": "int8"},
            {"axis": -1},
        ),
        ("int8_weight_only", "int8", {"axis": 0}, None),
        (
            "int4_weight_only",
            "int4",
            {"axis": 0, "value_range": (-8, 7), "output_dtype": "int8"},
            None,
        ),
    )
    def test_einsum_dense_quantize(
        self, mode, weight_quantizer_args, activation_quantizer_args
    ):
        """Test EinsumDense quantization with QuantizationConfig."""
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))

        weight_quantizer = AbsMaxQuantizer(**weight_quantizer_args)
        if activation_quantizer_args is not None:
            activation_quantizer = AbsMaxQuantizer(**activation_quantizer_args)
        else:
            activation_quantizer = None

        if mode == "int8":
            config = Int8QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
            )
        elif mode == "int4":
            # Custom quantizers require per-channel mode (block_size=None)
            config = Int4QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
                block_size=None,
            )

        layer.quantize(mode, config=config)

        if activation_quantizer_args is not None:
            # Verify inputs_quantizer is set correctly
            self.assertIsInstance(layer.inputs_quantizer, AbsMaxQuantizer)
        else:
            # Verify inputs_quantizer is None
            self.assertIsNone(layer.inputs_quantizer)

        # Verify call works
        x = np.random.random((2, 3)).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 8, 32))

        if mode == "int4":
            # Verify kernel is int8 (packed int4)
            self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")

    @parameterized.named_parameters(
        {
            "testcase_name": "_1d_end_weight",
            "equation": "ab,b->a",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (),
            "expected_kernel_shape": (32,),
            "expected_bias_shape": None,
            "expected_output_shape": (2,),
        },
        {
            "testcase_name": "_2d_middle_weight",
            "equation": "ab,bc->ac",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (64),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 64),
        },
        {
            "testcase_name": "_3d_bert",
            "equation": "abc,cde->abde",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_3_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (4,),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_2_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 1),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_1_3_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "be",
            "input_shape": (2, 7, 2),
            "output_shape": (7, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (7, 1, 4),
            "expected_output_shape": (2, 7, 3, 4),
        },
        {
            "testcase_name": "_3d_bert_projection",
            "equation": "BFNH,NHD->BFD",
            "bias_axes": None,
            "input_shape": (2, 1, 2, 3),
            "output_shape": (1, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 4),
        },
        {
            "testcase_name": "_2d_bert",
            "equation": "abc,cd->abd",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (1, 4),
            "expected_kernel_shape": (2, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 4),
        },
        {
            "testcase_name": "_embedding_1d",
            "equation": "i,d->id",
            "bias_axes": None,
            "input_shape": (2,),
            "output_shape": (2,),
            "expected_kernel_shape": (2,),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 2),
        },
        {
            "testcase_name": "_xlnet_lm",
            "equation": "ibd,nd->ibn",
            "bias_axes": None,
            "input_shape": (2, 2, 1),
            "output_shape": (2, 2),
            "expected_kernel_shape": (2, 1),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 2, 2),
        },
        {
            "testcase_name": "_2d_precast",
            "equation": "...b,bc->...c",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (64,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 64),
        },
        {
            "testcase_name": "_2d_precast_elided_input_used_in_output",
            "equation": "...bc,bc->...b",
            "bias_axes": None,
            "input_shape": (2, 32, 64),
            "output_shape": (32,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 32),
        },
        {
            "testcase_name": "_2d_precast_multiple_elided_dims",
            "equation": "...b,bc->...c",
            "bias_axes": None,
            "input_shape": (2, 3, 32),
            "output_shape": (64,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 3, 64),
        },
        {
            "testcase_name": "_3d_precast",
            "equation": "...c,cde->...de",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_3_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (4,),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_2_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 1),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_2_3_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "de",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 4),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_2d_postcast",
            "equation": "bc...,cd->bd...",
            "bias_axes": None,
            "input_shape": (2, 1, 2, 3),
            "output_shape": (4,),
            "expected_kernel_shape": (1, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 4, 2, 3),
        },
        {
            "testcase_name": "_3d_postcast",
            "equation": "bc...,cde->bde...",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_1_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (3, 1, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_2_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (4, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_1_2_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "de",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (3, 4, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
    )
    def test_einsum_dense_basics(
        self,
        equation,
        bias_axes,
        input_shape,
        output_shape,
        expected_kernel_shape,
        expected_bias_shape,
        expected_output_shape,
    ):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": equation,
                "output_shape": output_shape,
                "bias_axes": bias_axes,
            },
            input_shape=input_shape,
            expected_output_shape=expected_output_shape,
            expected_num_trainable_weights=(
                2 if expected_bias_shape is not None else 1
            ),
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        layer = layers.EinsumDense(
            equation, output_shape=output_shape, bias_axes=bias_axes
        )
        layer.build(input_shape)
        self.assertEqual(layer.kernel.shape, expected_kernel_shape)
        if expected_bias_shape is not None:
            self.assertEqual(layer.bias.shape, expected_bias_shape)

    def test_compute_output_shape_before_build(self):
        # `compute_output_shape` must work without the layer being built first.
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(7,))
        self.assertEqual(layer.compute_output_shape((None, 5)), (None, 7))
        # And it stays consistent after the layer is built.
        layer.build((None, 5))
        self.assertEqual(layer.compute_output_shape((None, 5)), (None, 7))

        # A higher-rank equation with an explicitly-unknown output axis;
        # `compute_output_shape` must agree with the actual call.
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 4), bias_axes="d"
        )
        x = layers.Input(shape=(3, 8))
        self.assertEqual(
            layer.compute_output_shape((None, 3, 8)), tuple(layer(x).shape)
        )

    @parameterized.named_parameters(
        # Fewer axes than the equation has input labels.
        ("plain_too_short", "ab,bc->ac", 4, (None,), "ndim=2"),
        ("left_elided_too_short", "...ab,bc->...ac", 4, (None,), "min_ndim=2"),
        ("right_elided_too_short", "ab...,bc->ac...", 4, (None,), "min_ndim=2"),
        ("scalar_input", "ab,bc->ac", 4, (), "ndim=2"),
        # A repeated leading label keeps the label map in range, so only the
        # ndim check rejects these equations before they are called.
        (
            "left_elided_repeated_label_ndim_1",
            "...aa,ac->...ac",
            (),
            (4,),
            "min_ndim=2",
        ),
        (
            "left_elided_repeated_label_ndim_2",
            "...aab,bc->...ac",
            (),
            (3, 3),
            "min_ndim=3",
        ),
        # More axes than labels, without an ellipsis to absorb them.
        ("plain_too_long", "ab,bc->ac", 4, (None, 3, 5), "ndim=2"),
        (
            "plain_three_labels_too_long",
            "abc,cd->abd",
            (3, 4),
            (None, 3, 5, 6),
            "ndim=3",
        ),
    )
    def test_input_ndim_must_fit_equation(
        self, equation, output_shape, input_shape, expected_ndim
    ):
        message = re.escape(
            f"The einsum equation '{equation}' expects an input with "
            f"{expected_ndim}, but received input shape {input_shape} with "
            f"ndim={len(input_shape)}."
        )
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        with self.assertRaisesRegex(ValueError, message):
            layer.build(input_shape)
        self.assertFalse(layer.built)

        layer = layers.EinsumDense(equation, output_shape=output_shape)
        with self.assertRaisesRegex(ValueError, message):
            layer.compute_output_shape(input_shape)

    def test_einsum_dense_constraints(self):
        layer = layers.EinsumDense(
            "abc,cde->abde", (1, 3, 4), kernel_constraint="non_neg"
        )
        layer.build((2, 1, 2))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.EinsumDense(
            "ab,b->a", (1, 3, 4), bias_axes="a", bias_constraint="non_neg"
        )
        layer.build((2, 1))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)

    def test_analyze_einsum_string_axes(self):
        # Test case 1: Standard dense
        equation = "ab,bc->ac"
        input_shape = (None, 32)
        output_shape = (64,)
        kernel_shape, bias_shape, full_output_shape, input_axes, output_axes = (
            _analyze_einsum_string(
                equation,
                bias_axes=None,
                input_shape=input_shape,
                output_shape=output_shape,
            )
        )
        self.assertEqual(input_axes, [0])
        self.assertEqual(output_axes, [1])

        # Test case 2: Attention QKV projection
        equation = "abc,cde->abde"
        input_shape = (None, 10, 32)
        output_shape = (10, 3, 4)
        kernel_shape, bias_shape, full_output_shape, input_axes, output_axes = (
            _analyze_einsum_string(
                equation,
                bias_axes=None,
                input_shape=input_shape,
                output_shape=output_shape,
            )
        )
        self.assertEqual(input_axes, [0])
        self.assertEqual(output_axes, [1, 2])

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes=None,
        )
        layer.build((None, 3))
        layer.enable_lora(2)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 3)
        self.assertDType(layer.lora_kernel_a, "float32")
        self.assertDType(layer.lora_kernel_b, "float32")

        # Try eager call
        x = np.random.random((64, 3))
        y = np.random.random((64, 8, 32))
        _ = layer(x[:2])

        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                layers.EinsumDense(
                    equation="ab,bcd->acd",
                    output_shape=(8, 32),
                    bias_axes=None,
                ),
            ]
        )
        new_model.build((None, 3))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_lora_with_alpha(self):
        # Use a simple equation that mimics a `Dense` layer behavior.
        equation = "ab,bc->ac"
        output_shape = 3  # This means the kernel shape will be (input_dim, 3).
        bias_axes = None

        # Create and build the `EinsumDense` layer
        # with an input shape (None, 2).
        layer = layers.EinsumDense(
            equation=equation, output_shape=output_shape, bias_axes=bias_axes
        )
        # Build the layer with an input shape of (batch, 2).
        layer.build((None, 2))

        # Set the base kernel weights to a known value.
        base_kernel = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        layer._kernel.assign(base_kernel)

        # Enable LoRA with `rank`=2 and a custom `lora_alpha`=3.0.
        layer.enable_lora(rank=2, lora_alpha=3.0)
        self.assertEqual(layer.lora_rank, 2)
        self.assertEqual(layer.lora_alpha, 3.0)

        # The expected shapes are:
        #   `base_kernel`: (2, 3)
        #   `lora_kernel_a`: (2, 2) and `lora_kernel_b`: (2, 3)
        a_val = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        b_val = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
        layer.lora_kernel_a.assign(a_val)
        layer.lora_kernel_b.assign(b_val)

        # Compute expected effective kernel.
        # Scaling factor is `lora_alpha / lora_rank` = 3.0 / 2 = 1.5
        expected_delta = 1.5 * np.matmul(a_val, b_val)
        expected_kernel = base_kernel + expected_delta

        # Verify that the effective kernel property returns the expected value.
        actual_kernel = ops.convert_to_numpy(layer.kernel)
        self.assertAllClose(
            actual_kernel, expected_kernel, tpu_atol=1e-3, tpu_rtol=1e-3
        )

    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": "ab,bcd->acd",
                "output_shape": (8, 32),
                "bias_axes": None,
                "lora_rank": 2,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 8, 32),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.requires_trainable_backend
    def test_enable_dora(self):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes=None,
        )
        layer.build((None, 3))
        layer.enable_dora(2)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)
        self.assertDType(layer.dora_kernel_a, "float32")
        self.assertDType(layer.dora_kernel_b, "float32")
        self.assertDType(layer.dora_magnitude, "float32")

        # Try eager call
        x = np.random.random((64, 3))
        y = np.random.random((64, 8, 32))
        _ = layer(x[:2])

        init_dora_a_kernel_value = layer.dora_kernel_a.numpy()
        init_dora_b_kernel_value = layer.dora_kernel_b.numpy()
        init_dora_magnitude_value = layer.dora_magnitude.numpy()

        # Try calling fit()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_dora_a_kernel_value = layer.dora_kernel_a.numpy()
        final_dora_b_kernel_value = layer.dora_kernel_b.numpy()
        final_dora_magnitude_value = layer.dora_magnitude.numpy()

        diff_a = np.max(
            np.abs(init_dora_a_kernel_value - final_dora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_dora_b_kernel_value - final_dora_b_kernel_value)
        )
        diff_m = np.max(
            np.abs(init_dora_magnitude_value - final_dora_magnitude_value)
        )

        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)
        self.assertGreater(diff_m, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "dora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].dora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "dora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-dora model
        new_model = models.Sequential(
            [
                layers.EinsumDense(
                    equation="ab,bcd->acd",
                    output_shape=(8, 32),
                    bias_axes=None,
                ),
            ]
        )
        new_model.build((None, 3))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a dora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_dora_with_alpha(self):
        equation = "ab,bc->ac"
        output_shape = 3
        bias_axes = None

        layer = layers.EinsumDense(
            equation=equation, output_shape=output_shape, bias_axes=bias_axes
        )
        layer.build((None, 2))

        base_kernel = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        layer._kernel.assign(base_kernel)

        layer.enable_dora(rank=2, dora_alpha=3.0)
        self.assertEqual(layer.dora_rank, 2)
        self.assertEqual(layer.dora_alpha, 3.0)

        a_val = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        b_val = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
        layer.dora_kernel_a.assign(a_val)
        layer.dora_kernel_b.assign(b_val)

        # Manually compute expected effective kernel.
        expected_delta = 1.5 * np.matmul(a_val, b_val)
        W_combined = base_kernel + expected_delta
        norm = np.sqrt(np.sum(np.square(W_combined), axis=0))
        expected_kernel = ops.convert_to_numpy(layer.dora_magnitude) * (
            W_combined / norm
        )

        actual_kernel = ops.convert_to_numpy(layer.kernel)
        self.assertAllClose(
            actual_kernel, expected_kernel, tpu_atol=1e-3, tpu_rtol=1e-3
        )

    def test_dora_rank_argument(self):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": "ab,bcd->acd",
                "output_shape": (8, 32),
                "bias_axes": None,
                "dora_rank": 2,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 8, 32),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_enable_dora_with_kernel_constraint(self):
        layer = layers.EinsumDense(
            "ab,bc->ac", output_shape=4, kernel_constraint="max_norm"
        )
        with self.assertRaisesRegex(
            ValueError, "DoRA is incompatible with kernel constraints"
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_on_unbuilt_layer(self):
        layer = layers.EinsumDense("ab,bc->ac", output_shape=4)
        with self.assertRaisesRegex(
            ValueError, "Cannot enable dora on a layer that isn't yet built"
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_when_already_enabled(self):
        layer = layers.EinsumDense("ab,bc->ac", output_shape=4)
        layer.build((None, 2))
        layer.enable_dora(rank=2)
        with self.assertRaisesRegex(ValueError, "dora is already enabled"):
            layer.enable_dora(rank=2)

    def test_enable_dora_with_gptq_quantization(self):
        layer = layers.EinsumDense("ab,bc->ac", output_shape=4)
        layer.build((None, 2))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=2
            ),
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "DoRA is not currently supported with quantized layers.",
        ):
            layer.enable_dora(rank=2)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_enable_dora_int4_kernel_shape(self):
        layer = layers.EinsumDense("ab,bcd->acd", output_shape=(2, 4))
        layer.build((None, 3))
        # original_kernel_shape should be (3, 2, 4)
        layer.quantize("int4")
        with self.assertRaisesRegex(
            NotImplementedError,
            "DoRA is not currently supported with quantized layers.",
        ):
            layer.enable_dora(rank=2)

    def test_dora_lora_mutual_exclusivity(self):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes=None,
        )
        layer.build((None, 3))
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "LoRA is already enabled. Cannot enable DoRA."
        ):
            layer.enable_dora(rank=2)

        layer2 = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes=None,
        )
        layer2.build((None, 3))
        layer2.enable_dora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "dora is already enabled. Cannot enable LoRA."
        ):
            layer2.enable_lora(rank=2)

    # Test quantization-related methods.

    @parameterized.named_parameters(
        ("int8", "int8", 2e-2),
        ("int4", "int4", 3e-3),
    )
    def test_quantize_int(self, mode, error_threshold):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        x = np.random.random((2, 3))
        y_float = layer(x)
        layer.quantize(mode)

        # Verify weights dtype
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            standardize_dtype(layer.kernel_scale.dtype),
            layer.variable_dtype,
        )

        # Try eager call and verify output correctness
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

        # Try saving and reloading the model
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Try building with quantized dtype policy
        layer = layers.EinsumDense(
            equation="abcde,afce->acdbf",  # Test reduce and transpose
            output_shape=(2, 4, 8, 16),
            bias_axes="d",
            dtype=f"{mode}_from_mixed_bfloat16",
        )
        layer.build((1, 8, 2, 4, 32))
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(standardize_dtype(layer.kernel_scale.dtype), "float32")
        layer = layers.EinsumDense(
            equation="a,b->ab",  # Test expand
            output_shape=(4,),
            dtype=f"{mode}_from_float32",
        )
        layer.build((None,))
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(standardize_dtype(layer.kernel_scale.dtype), "float32")
        layer = layers.EinsumDense(
            equation="ab,ab->a",  # Test squeeze
            output_shape=(2,),
            dtype="int8_from_float32",
        )
        layer.build((2, 4))
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(standardize_dtype(layer.kernel_scale.dtype), "float32")

    @parameterized.named_parameters(
        (
            "int8_btnh,nhd->btd",
            "int8",
            "btnh,nhd->btd",
            (None, 8),
            (1, 2, 2, 4),
            1e-3,
        ),
        (
            "int8_btd,ndh->btnh",
            "int8",
            "btd,ndh->btnh",
            (None, 2, 8),
            (1, 2, 4),
            1e-3,
        ),
        ("int8_btd,df->btf", "int8", "btd,df->btf", (None, 4), (1, 2, 4), 1e-3),
        (
            "int4_btnh,nhd->btd",
            "int4",
            "btnh,nhd->btd",
            (None, 8),
            (1, 2, 2, 4),
            4e-3,
        ),
        (
            "int4_btd,ndh->btnh",
            "int4",
            "btd,ndh->btnh",
            (None, 2, 8),
            (1, 2, 4),
            4e-3,
        ),
        (
            "int4_btd,df->btf",
            "int4",
            "btd,df->btf",
            (None, 4),
            (1, 2, 4),
            4e-3,
        ),
    )
    def test_quantize_with_specific_equations(
        self,
        quantization_mode,
        equation,
        output_shape,
        input_shape,
        error_threshold,
    ):
        set_random_seed(1337)
        layer = layers.EinsumDense(equation=equation, output_shape=output_shape)
        layer.build(input_shape)
        x = random.uniform(input_shape)
        y_float = layer(x)

        layer.quantize(quantization_mode)
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

    @parameterized.named_parameters(
        ("abc,bdc->ad", "abc,bdc->ad", (3, 4), (5,)),
        ("ab,b->a", "ab,b->a", (3,), ()),
        ("abc,c->ab", "abc,c->ab", (3, 4), (3,)),
        ("abc,cb->ab", "abc,cb->ab", (3, 4), (3,)),
        ("abc,bc->a", "abc,bc->a", (3, 4), ()),
        ("abcd,cd->ab", "abcd,cd->ab", (3, 4, 5), (3,)),
        ("size_one_free_axis", "acd,gecd->aeg", (4, 5), (6, 1)),
        ("size_one_inputs_axis", "acdb,bc->acd", (4, 1, 3), (4, 1)),
    )
    def test_quantize_int8_scale_alignment(
        self, equation, input_shape, output_shape
    ):
        # These equations reduce more axes than the output adds, so the
        # scales need a squeeze to line up with the outputs. In the size-one
        # cases, the squeeze must not remove another axis of size 1. The
        # int8 output must equal an einsum of the dequantized inputs and the
        # dequantized kernel.
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build((None,) + input_shape)
        rng = np.random.default_rng(0)
        kernel = rng.standard_normal(layer.kernel.shape).astype("float32")
        layer.kernel.assign(kernel)
        x = rng.standard_normal((2,) + input_shape).astype("float32")
        y_float = layer(x)

        layer.quantize("int8")
        y_quantized = ops.convert_to_numpy(layer(x))

        inputs_spec, rest = equation.split(",")
        kernel_spec, output_spec = rest.split("->")
        inputs_axes = [
            i for i, c in enumerate(inputs_spec) if c not in output_spec
        ]
        kernel_axes = [
            i for i, c in enumerate(kernel_spec) if c not in output_spec
        ]
        x_q, x_scale = quantizers.abs_max_quantize(
            x, axis=inputs_axes, to_numpy=True
        )
        kernel_q, kernel_scale = quantizers.abs_max_quantize(
            kernel, axis=kernel_axes, to_numpy=True
        )
        self.assertAllEqual(layer._kernel, kernel_q)
        expected = np.einsum(
            equation,
            ops.convert_to_numpy(x_q) / ops.convert_to_numpy(x_scale),
            ops.convert_to_numpy(kernel_q) / ops.convert_to_numpy(kernel_scale),
        )
        self.assertEqual(y_quantized.shape, tuple(y_float.shape))
        self.assertAllClose(y_quantized, expected, atol=1e-5, rtol=1e-5)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_on_unbuilt_layer(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize(mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_on_subclass(self, mode):
        class MyEinsumDense(layers.EinsumDense):
            pass

        layer = MyEinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode)

        layer.quantize(mode, type_check=False)  # No error

    def test_quantize_subclass_skipping_super_build_raises(self):
        # Without `super().build()` the layer records neither `kernel_shape`
        # nor `einsum_axes`; `quantize` refuses before it changes anything.
        class MyEinsumDense(layers.EinsumDense):
            def build(self, input_shape):
                self._kernel = self.add_weight(name="kernel", shape=(3, 8))
                self.bias = None

        layer = MyEinsumDense(equation="ab,bc->ac", output_shape=8)
        layer.build((None, 3))
        with self.assertRaisesRegex(ValueError, "kernel_shape, einsum_axes"):
            layer.quantize("int8", type_check=False)
        self.assertIsNone(layer.quantization_mode)
        self.assertEqual(tuple(layer._kernel.shape), (3, 8))

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_when_already_quantized(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 16),
            bias_axes="d",
        )
        layer.build((None, 3))
        layer.quantize(mode)
        for m in ["int8", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 16),
            bias_axes="d",
            dtype=f"{mode}_from_float32",
        )
        layer.build((None, 3))
        for m in ["int8", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 3),
        ("float8", "float8_from_float32", 8),
        ("int4", "int4_from_float32", 3),  # kernel + bias + scale
    )
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
        ("int3", "int3"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        x = np.random.random((1, 3))
        # dtype_policy should not be altered by failed quantization
        original_dtype_policy = layer.dtype_policy

        # Test quantize
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode."):
            layer.quantize(mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_build
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            layer.quantized_build((None, 2), mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_call
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            # Explicitly set quantization_mode
            layer._dtype_policy._quantization_mode = mode
            layer.quantized_call(x)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

    @parameterized.named_parameters(
        ("int8", "int8_from_mixed_bfloat16", 1, 2),
        ("float8", "float8_from_mixed_bfloat16", 8, 0),
        ("int4", "int4_from_mixed_bfloat16", 1, 2),
    )
    def test_quantize_dtype_argument(
        self, dtype, num_trainable_weights, num_non_trainable_weights
    ):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": "ab,bcd->acd",
                "output_shape": (8, 32),
                "bias_axes": "d",
                "dtype": dtype,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 8, 32),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.named_parameters(
        ("int8_ab,bcd->acd", "int8", "ab,bcd->acd", (64, 3), (64, 8, 32), 2, 4),
        (
            "int8_btd,ndh->btnh",
            "int8",
            "btd,ndh->btnh",
            (1, 4, 32),
            (1, 4, 8, 16),
            2,
            4,
        ),
        # int4 has +1 for kernel_zero and +1 for g_idx
        ("int4_ab,bcd->acd", "int4", "ab,bcd->acd", (64, 3), (64, 8, 32), 4, 6),
        (
            "int4_btd,ndh->btnh",
            "int4",
            "btd,ndh->btnh",
            (1, 4, 32),
            (1, 4, 8, 16),
            4,
            6,
        ),
    )
    @pytest.mark.requires_trainable_backend
    def test_quantize_lora_integration(
        self,
        quantization_mode,
        equation,
        input_shape,
        output_shape,
        expected_non_trainable,
        expected_torch_params,
    ):
        config = dict(
            equation=equation, output_shape=output_shape[1:], bias_axes=None
        )
        layer = layers.EinsumDense(**config)
        layer.build(input_shape)
        layer.enable_lora(2)
        layer.quantize(quantization_mode)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, expected_non_trainable)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, expected_torch_params)

        # Try calling fit()
        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        x = np.random.random(input_shape)
        y = np.random.random(output_shape)
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.EinsumDense(**config)])
        new_model.build(input_shape)
        new_model.quantize(quantization_mode)
        new_model.load_weights(temp_filepath)
        self.assertFalse(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal(input_shape)
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(
                reloaded_layer(ref_input), ref_output, atol=1e-7
            )
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8(self):
        import ml_dtypes

        from keras.src import quantizers

        layer = layers.EinsumDense(
            "ab,bc->ac",
            output_shape=[32],
            bias_axes="c",
        )
        layer.build((None, 16))
        layer.quantize("float8")
        optimizer = optimizers.AdamW(learning_rate=0.1)
        optimizer.build(layer.trainable_variables)

        def stateless_loss_fn(trainable_variables, x, dy):
            y = layer.stateless_call(trainable_variables, [], x, training=True)[
                0
            ]
            return y * ops.cast(dy, y.dtype)

        grad_fn = ops.grad(stateless_loss_fn)

        def train_one_step(x, dy):
            trainable_variables = [v.value for v in layer.trainable_variables]
            grads = grad_fn(trainable_variables, x, dy)
            optimizer.apply(grads, layer.trainable_variables)

        scale_x, amax_history_x = ops.ones(()), ops.zeros((1024,))
        scale_k, amax_history_k = ops.ones(()), ops.zeros((1024,))
        scale_g, amax_history_g = ops.ones(()), ops.zeros((1024,))
        e4m3_max = ops.cast(
            float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
        )
        e5m2_max = ops.cast(
            float(ml_dtypes.finfo("float8_e5m2").max), "float32"
        )

        for _ in range(3):
            x = random.normal((16, 16), dtype="float32")
            g = random.normal((16, 32), dtype="float32")
            k = ops.convert_to_tensor(layer._kernel)

            # Manually compute the expected amax history and scaling factors.
            amax_from_history_x = ops.max(amax_history_x)
            amax_from_history_k = ops.max(amax_history_k)
            amax_from_history_g = ops.max(amax_history_g)
            scale_x = quantizers.compute_float8_scale(
                amax_from_history_x, scale_x, e4m3_max
            )
            scale_k = quantizers.compute_float8_scale(
                amax_from_history_k, scale_k, e4m3_max
            )
            scale_g = quantizers.compute_float8_scale(
                amax_from_history_g, scale_g, e5m2_max
            )
            amax_history_x = quantizers.compute_float8_amax_history(
                x, amax_history_x
            )
            amax_history_k = quantizers.compute_float8_amax_history(
                k, amax_history_k
            )
            amax_history_g = quantizers.compute_float8_amax_history(
                g, amax_history_g
            )

            train_one_step(x, g)

            self.assertAllClose(layer.inputs_amax_history, amax_history_x)
            self.assertAllClose(layer.kernel_amax_history, amax_history_k)
            self.assertAllClose(layer.outputs_grad_amax_history, amax_history_g)
            self.assertAllClose(layer.inputs_scale, scale_x)
            self.assertAllClose(layer.kernel_scale, scale_k)
            self.assertAllClose(layer.outputs_grad_scale, scale_g)

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8_fitting(self):
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize("float8")
        self.assertLen(layer.trainable_weights, 8)
        self.assertLen(layer.non_trainable_weights, 0)

        # Try calling fit()
        x = np.random.random((64, 3))
        y = np.random.random((64, 8, 32))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.EinsumDense(**config)])
        new_model.build((None, 3))
        new_model.quantize("float8")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((2, 3))
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(reloaded_layer(ref_input), ref_output)
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    def test_quantize_float8_inference(self):
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize("float8")

        # Try calling with `training=False` and the result must match
        # `training=True` because there is no update.
        x = np.random.random((64, 3))
        y_inference = layer(x, training=False)
        y_training = layer(x, training=True)
        self.assertAllClose(y_inference, y_training)

    @parameterized.named_parameters(
        ("ab_bc_ac", "ab,bc->ac", (16,), "c", (None, 8)),
        ("btd_dnh_btnh", "btd,dnh->btnh", (5, 2, 4), "nh", (None, 5, 8)),
        (
            "btd_dnh_btnh_no_bias",
            "btd,dnh->btnh",
            (5, 2, 4),
            None,
            (None, 5, 8),
        ),
    )
    def test_quantize_float8_weights_order(
        self, equation, output_shape, bias_axes, input_shape
    ):
        # A layer quantized in place and a layer built from its policy hold
        # the same variables in the same order.
        config = dict(
            equation=equation, output_shape=output_shape, bias_axes=bias_axes
        )
        layer = layers.EinsumDense(**config)
        layer.build(input_shape)
        layer.quantize("float8")
        for v in layer.weights:
            v.assign(np.random.uniform(0.5, 1.5, v.shape))
        new_layer = layers.EinsumDense(**config, dtype=layer.dtype_policy)
        new_layer.build(input_shape)
        self.assertEqual(
            [(v.name, v.shape) for v in new_layer.weights],
            [(v.name, v.shape) for v in layer.weights],
        )
        new_layer.set_weights(layer.get_weights())
        x = np.random.random((2,) + input_shape[1:])
        self.assertAllClose(
            new_layer(x, training=False), layer(x, training=False)
        )

    def test_gptq_serialization(self):
        """Test that a GPTQ-quantized layer can be serialized and deserialized
        correctly."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        config = layer.get_config()
        new_layer = layers.EinsumDense.from_config(config)
        new_layer.build((None, 3))
        self.assertEqual(new_layer.quantization_mode, "gptq")

    def test_awq_serialization(self):
        """Test that an AWQ-quantized layer can be serialized and deserialized
        correctly."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )
        layer_config = layer.get_config()
        new_layer = layers.EinsumDense.from_config(layer_config)
        new_layer.build((None, 3))
        self.assertEqual(new_layer.quantization_mode, "awq")

    def test_gptq_uncalibrated_save_raises(self):
        """Saving a GPTQ layer that was never calibrated must raise."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        with self.assertRaisesRegex(ValueError, "never been calibrated"):
            layer.save_own_variables({})

    def test_awq_uncalibrated_save_raises(self):
        """Saving an AWQ layer that was never calibrated must raise."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )
        with self.assertRaisesRegex(ValueError, "never been calibrated"):
            layer.save_own_variables({})

    def test_int4_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 kernel."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(2,),
        )
        layer.build((None, 2))
        layer.quantize("int4")
        packed_kernel = layer._kernel
        # Unpack [rows, ceil(columns/2)] -> [rows, columns],
        # then reshape to original shape
        unpacked = quantizers.unpack_int4(
            packed_kernel, layer.kernel_scale.shape[-1], axis=-1
        )
        expected = ops.reshape(unpacked, layer.kernel_shape)
        self.assertAllClose(layer.kernel, expected)

    def test_legacy_load_own_variables(self):
        # In previous versions, `load_own_variables` accepted a store with
        # numeric keys.
        float32_store = {
            "0": np.random.random((3, 8, 32)).astype("float32"),
            "1": np.random.random((32,)).astype("float32"),
        }
        int8_store = {
            "0": np.random.randint(-128, 127, size=(3, 8, 32), dtype="int8"),
            "1": np.random.random((32,)).astype("float32"),
            "2": np.random.random((1, 8, 32)).astype("float32"),
        }
        int4_store = {
            # int4 layout: kernel is [rows, ceil(columns/2)] = [3, 128]
            # where rows=3, columns=8*32=256
            "0": np.random.randint(-128, 127, size=(3, 128), dtype="int8"),
            "1": np.random.random((32,)).astype("float32"),
            "2": np.random.random((256,)).astype("float32"),  # per-channel
        }
        float8_store = {
            "0": np.random.random((3, 8, 32)).astype("float32"),
            "1": np.random.random((32,)).astype("float32"),
            # inputs_scale.
            "2": np.random.random(()).astype("float32"),
            # inputs_amax_history.
            "3": np.random.random((1024,)).astype("float32"),
            # kernel_scale.
            "4": np.random.random(()).astype("float32"),
            # kernel_amax_history.
            "5": np.random.random((1024,)).astype("float32"),
            # outputs_grad_scale.
            "6": np.random.random(()).astype("float32"),
            # outputs_grad_amax_history.
            "7": np.random.random((1024,)).astype("float32"),
        }
        gptq_store = {
            # bias
            "0": np.random.random((32,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(24, 16), dtype="uint8"),
            # kernel_scale.
            "2": np.random.random((3, 32)).astype("float32"),
            # kernel_zero
            "3": np.random.random((3, 32)).astype("uint8"),
            # g_idx: legacy checkpoints stored the integer group indices as
            # float32; they load into the float32 g_idx variable unchanged.
            "4": (np.arange(24) // 8).astype("float32"),
        }
        # kernel shape (3, 8, 32), packed: (16, 24) for 4-bit
        awq_store = {
            "0": np.random.random((32,)).astype("float32"),  # bias
            "1": np.random.randint(0, 16, size=(24, 16), dtype="uint8"),
            "2": np.random.random((3, 32)).astype("float32"),  # scale
            "3": np.random.random((3, 32)).astype("uint8"),  # zero
            "4": np.random.random((24,)).astype("float32"),  # awq_scales
            # g_idx saved as int32 by a newer checkpoint; the cast on load
            # brings it into the float32 storage variable (see above).
            "5": (np.arange(24) // 8).astype("int32"),
        }
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )

        # Test float32 layer.
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.load_own_variables(float32_store)
        self.assertAllClose(layer._kernel, float32_store["0"])
        self.assertAllClose(layer.bias, float32_store["1"])

        # Test int8-quantized layer.
        layer = layers.EinsumDense(**config, dtype="int8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(int8_store)
        self.assertAllClose(layer._kernel, int8_store["0"])
        self.assertAllClose(layer.bias, int8_store["1"])
        self.assertAllClose(layer.kernel_scale, int8_store["2"])

        # Test int4-quantized layer.
        layer = layers.EinsumDense(**config, dtype="int4_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(int4_store)
        self.assertAllClose(layer._kernel, int4_store["0"])
        self.assertAllClose(layer.bias, int4_store["1"])
        self.assertAllClose(layer.kernel_scale, int4_store["2"])

        # Test float8-quantized layer.
        layer = layers.EinsumDense(**config, dtype="float8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(float8_store)
        self.assertAllClose(layer._kernel, float8_store["0"])
        self.assertAllClose(layer.bias, float8_store["1"])
        self.assertAllClose(layer.inputs_scale, float8_store["2"])
        self.assertAllClose(layer.inputs_amax_history, float8_store["3"])
        self.assertAllClose(layer.kernel_scale, float8_store["4"])
        self.assertAllClose(layer.kernel_amax_history, float8_store["5"])
        self.assertAllClose(layer.outputs_grad_scale, float8_store["6"])
        self.assertAllClose(layer.outputs_grad_amax_history, float8_store["7"])

        # Test gptq-quantized layer.
        layer = layers.EinsumDense(**config, dtype="gptq/4/8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(gptq_store)
        self.assertFalse(layer.calibration_pending)
        self.assertAllClose(layer.bias, gptq_store["0"])
        self.assertAllClose(layer.quantized_kernel, gptq_store["1"])
        self.assertAllClose(layer.kernel_scale, gptq_store["2"])
        self.assertAllClose(layer.kernel_zero, gptq_store["3"])
        self.assertAllClose(layer.g_idx, gptq_store["4"])
        # g_idx is stored as float32; the legacy float32 store loads as-is.
        self.assertDType(layer.g_idx, "float32")

        # Test awq-quantized layer.
        layer = layers.EinsumDense(**config, dtype="awq/4/8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(awq_store)
        self.assertFalse(layer.calibration_pending)
        self.assertAllClose(layer.bias, awq_store["0"])
        self.assertAllClose(layer.quantized_kernel, awq_store["1"])
        self.assertAllClose(layer.kernel_scale, awq_store["2"])
        self.assertAllClose(layer.kernel_zero, awq_store["3"])
        self.assertAllClose(layer.awq_scales, awq_store["4"])
        self.assertAllClose(layer.g_idx, awq_store["5"])
        # The int32-saved g_idx is cast to the float32 variable on load.
        self.assertDType(layer.g_idx, "float32")

    @staticmethod
    def _build_einsum_for_mode(mode, input_dim=256):
        """Builds an `EinsumDense` populated for `mode`'s serialization spec."""
        cfg = dict(equation="ab,bcd->acd", output_shape=(8, 32), bias_axes="d")
        if mode == "none":
            layer = layers.EinsumDense(**cfg)
            layer.build((None, input_dim))
        elif mode == "int8":
            layer = layers.EinsumDense(**cfg, dtype="int8_from_float32")
            layer.build((None, input_dim))
        elif mode == "int4_per_channel":
            layer = layers.EinsumDense(**cfg, dtype="int4_from_float32")
            layer.build((None, input_dim))
        elif mode == "int4_grouped":
            layer = layers.EinsumDense(**cfg)
            layer.build((None, input_dim))
            layer.quantize(
                "int4", config=Int4QuantizationConfig(block_size=128)
            )
        elif mode == "float8":
            layer = layers.EinsumDense(**cfg, dtype="float8_from_float32")
            layer.build((None, input_dim))
        elif mode == "gptq":
            layer = layers.EinsumDense(**cfg, dtype="gptq/4/32_from_float32")
            layer.build((None, input_dim))
        elif mode == "awq":
            layer = layers.EinsumDense(**cfg, dtype="awq/4/32_from_float32")
            layer.build((None, input_dim))
        else:
            raise ValueError(f"Unhandled test mode: {mode}")
        return layer

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_serialization_round_trip(self):
        # Modes whose `save_own_variables` writes a self-consistent store.
        for mode in (
            "none",
            "int8",
            "int4_per_channel",
            "int4_grouped",
            "float8",
        ):
            with self.subTest(mode=mode):
                source = self._build_einsum_for_mode(mode)
                test_utils.randomize_serialized_variables(source)

                store = {}
                source.save_own_variables(store)
                # Variables are keyed by consecutive integer positions in the
                # mode's serialization spec -- the on-disk contract.
                n_expected = len(test_utils.serialized_variable_names(source))
                self.assertEqual(
                    set(store.keys()), {str(i) for i in range(n_expected)}
                )

                target = self._build_einsum_for_mode(mode)
                target.load_own_variables(store)
                test_utils.assert_serialized_variables_equal(
                    self, source, target
                )

    def test_gptq_awq_load_from_store(self):
        # GPTQ/AWQ checkpoints carry variables (e.g. `kernel_zero`) that
        # `save_own_variables` does not emit for a freshly built layer, so
        # they are validated through the load path with a fully-populated
        # store.
        for mode in ("gptq", "awq"):
            with self.subTest(mode=mode):
                source = self._build_einsum_for_mode(mode)
                test_utils.randomize_serialized_variables(source)

                target = self._build_einsum_for_mode(mode)
                target.load_own_variables(test_utils.positional_store(source))
                self.assertFalse(target.calibration_pending)
                test_utils.assert_serialized_variables_equal(
                    self, source, target
                )
                # The codes come back in the kernel's own 3-D shape.
                self.assertEqual(tuple(target.kernel.shape), (256, 8, 32))

    def test_load_own_variables_reports_clear_errors(self):
        # int8 spec order: kernel ("0"), bias ("1"), kernel_scale ("2").
        source = self._build_einsum_for_mode("int8")
        test_utils.randomize_serialized_variables(source)
        store = test_utils.positional_store(source)

        # A missing variable trips the variable-count check with a clear error.
        missing = dict(store)
        del missing["2"]
        with self.assertRaisesRegex(ValueError, "expected 3 variables"):
            self._build_einsum_for_mode("int8").load_own_variables(missing)

        # A renumbered/corrupted key keeps the count but fails on the exact
        # missing position.
        corrupted = dict(store)
        corrupted["9"] = corrupted.pop("2")
        with self.assertRaises(KeyError):
            self._build_einsum_for_mode("int8").load_own_variables(corrupted)

    def test_int4_gptq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 GPTQ
        kernel."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(2,),
        )
        layer.build((None, 2))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        layer.calibration_pending = False  # Bypass calibration check
        packed_kernel = layer.quantized_kernel
        self.assertAllClose(
            layer.kernel,
            quantizers.unpack_int4(packed_kernel, 2, axis=-1, dtype="uint8"),
        )

    def test_gptq_kernel_packing(self):
        """Validates that 4-bit GPTQ packing reduces the kernel size."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(
            quantized_kernel_params,
            original_kernel_params // 2,
        )

    def test_from_config_ignores_legacy_gptq_unpacked_column_size(self):
        # Configs written by earlier releases carried the unpacked column
        # count of a GPTQ kernel; it is read from the stored group
        # parameters now, so the key is accepted and dropped.
        layer = layers.EinsumDense(
            "ab,bc->ac", output_shape=(4,), dtype="gptq/4/8_from_float32"
        )
        config = layer.get_config()
        self.assertNotIn("gptq_unpacked_column_size", config)
        config["gptq_unpacked_column_size"] = 4
        restored = layers.EinsumDense.from_config(config)
        restored.build((None, 8))
        self.assertEqual(restored.quantization_mode, "gptq")
        self.assertEqual(tuple(restored.kernel_scale.shape), (1, 4))

    def test_int4_awq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 AWQ
        kernel."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(2,),
        )
        layer.build((None, 2))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )
        layer.calibration_pending = False  # Bypass calibration check
        packed_kernel = layer.quantized_kernel
        self.assertAllClose(
            layer.kernel,
            quantizers.unpack_int4(packed_kernel, 2, axis=-1, dtype="uint8"),
        )

    def test_awq_kernel_packing(self):
        """Validates that 4-bit AWQ packing reduces the kernel size."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(
            quantized_kernel_params,
            original_kernel_params // 2,
        )

    def test_einsum_dense_int8_custom_quantizer(self):
        """
        Test custom quantizer serialization for einsum dense layer with
        int8 quantization.
        """
        # Setup
        weight_range = (-10, 10)
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0, value_range=weight_range),
            activation_quantizer=None,
        )

        # Build & Quantize
        layer = layers.EinsumDense("ab,bc->ac", output_shape=10)
        layer.build((None, 5))
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.EinsumDense.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )

        quantizer = new_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertEqual(quantizer.axis, (0,))
        self.assertAllEqual(quantizer.value_range, weight_range)

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
        ("per_channel_none", None),
        ("per_channel_neg1", -1),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_quantization_block_size(self, block_size):
        """Test int4 quantization with different block_size configurations."""
        # Use simple 2D equation: ab,bc->ac (like Dense)
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(64,),
            bias_axes="c",
        )
        layer.build((None, 256))

        x = np.random.random((2, 256)).astype("float32")
        y_float = layer(x)

        # Create config with specified block_size
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Both spellings of per-channel resolve to an ungrouped scheme.
        expected = None if block_size in (None, -1) else block_size
        self.assertEqual(layer._quantized_weight().scheme.group_size, expected)

        # Verify kernel_scale shape (GPTQ layout)
        if block_size is None or block_size == -1:
            # Per-channel: one scale per output unit
            expected_scale_shape = (64,)
        else:
            # Sub-channel: (n_groups, columns) = (n_groups, out)
            n_groups = math.ceil(256 / block_size)
            expected_scale_shape = (n_groups, 64)

        self.assertEqual(layer.kernel_scale.shape, expected_scale_shape)

        # Verify outputs are reasonable
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 0.01)  # Reasonable accuracy

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
        ("per_channel_none", None),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_block_size_serialization(self, block_size):
        """Test that block_size is preserved through serialization."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(32,),
            bias_axes="c",
        )
        layer.build((None, 128))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Get output before serialization
        x = np.random.random((2, 128)).astype("float32")
        y_before = layer(x)

        # Save and load model to test full serialization roundtrip
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "int4_block_size_einsum_model.keras"
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify block_size is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertIsInstance(
            loaded_layer.quantization_config, Int4QuantizationConfig
        )
        self.assertEqual(
            loaded_layer.quantization_config.block_size, block_size
        )

        # Verify outputs match after deserialization
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("per_channel", None),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_block_size_with_lora(self, block_size):
        """Test int4 quantization with LoRA and different block_size."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(64,),
            bias_axes="c",
        )
        layer.build((None, 128))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)
        layer.enable_lora(rank=4)

        x = np.random.random((2, 128)).astype("float32")

        # Should run without error
        y = layer(x)
        self.assertEqual(y.shape, (2, 64))

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_grouped_merged_save_keeps_lora_update(self):
        # See `DenseTest.test_int4_grouped_merged_save_keeps_lora_update`.
        input_dim, units, block_size = 12, 16, 4
        inputs = layers.Input((input_dim,))
        layer = layers.EinsumDense(
            "ab,bc->ac", output_shape=units, bias_axes=None, name="target"
        )
        model = models.Model(inputs, layer(inputs))
        layer.quantize(
            "int4", config=Int4QuantizationConfig(block_size=block_size)
        )
        eye = np.eye(input_dim, dtype="float32")
        quantized = ops.convert_to_numpy(layer(eye))
        layer.enable_lora(2)
        rng = np.random.RandomState(0)
        layer.lora_kernel_a.assign(
            rng.randn(input_dim, 2).astype("float32") * 0.5
        )
        layer.lora_kernel_b.assign(rng.randn(2, units).astype("float32") * 0.5)
        with_update = ops.convert_to_numpy(layer(eye))
        path = os.path.join(self.get_temp_dir(), "merged.keras")
        model.save(path)
        merged_layer = saving.load_model(path).get_layer("target")
        merged = ops.convert_to_numpy(merged_layer(eye))
        scale = ops.convert_to_numpy(merged_layer.kernel_scale)
        half_step = 0.5 * scale[np.arange(input_dim) // block_size]
        self.assertGreater(
            np.abs(with_update - quantized).max(), 4 * half_step.max()
        )
        # A float32 matmul runs at bfloat16 precision on TPU.
        atol = 1e-2 if testing.uses_tpu() else 1e-6
        self.assertTrue(
            np.all(np.abs(merged - with_update) <= half_step * 1.001 + atol)
        )

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_grouped_vs_perchannel_scale_shapes(self):
        """Test that grouped and per-channel have different scale shapes."""
        input_dim, output_dim = 256, 64
        block_size = 64

        # Per-channel layer
        layer_pc = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(output_dim,),
        )
        layer_pc.build((None, input_dim))
        config_pc = Int4QuantizationConfig(block_size=None)
        layer_pc.quantize("int4", config=config_pc)

        # Grouped layer
        layer_grouped = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(output_dim,),
        )
        layer_grouped.build((None, input_dim))
        config_grouped = Int4QuantizationConfig(block_size=block_size)
        layer_grouped.quantize("int4", config=config_grouped)

        # Per-channel: (output_dim,),
        # Grouped: (n_groups, output_dim)
        self.assertEqual(layer_pc.kernel_scale.shape, (output_dim,))
        self.assertEqual(
            layer_grouped.kernel_scale.shape,
            (input_dim // block_size, output_dim),
        )

    @parameterized.named_parameters(
        ("btd_df_btf_grouped", "btd,df->btf", (8, 32), (None, 8, 256), 64),
        ("btd_df_btf_pc", "btd,df->btf", (8, 32), (None, 8, 256), None),
        ("ab_bcd_acd_grouped", "ab,bcd->acd", (8, 32), (None, 64), 32),
        ("ab_bcd_acd_pc", "ab,bcd->acd", (8, 32), (None, 64), None),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_block_size_various_equations(
        self, equation, output_shape, input_shape, block_size
    ):
        """Test int4 quantization with different equations and block_size."""
        layer = layers.EinsumDense(
            equation=equation,
            output_shape=output_shape,
        )
        layer.build(input_shape)

        batch_input_shape = (2,) + input_shape[1:]
        x = np.random.random(batch_input_shape).astype("float32")
        y_float = layer(x)

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify quantization works
        y_quantized = layer(x)
        self.assertEqual(y_float.shape, y_quantized.shape)

        # Verify reasonable accuracy
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 0.02)

    @parameterized.named_parameters(
        # Attention output: bnh,nhd->bd (two reduced axes: n, h)
        # Kernel shape: (n=4, h=32, d=64), reduced axes: n, h
        ("attn_output_grouped", "bnh,nhd->bd", (64,), (None, 4, 32), 64),
        ("attn_output_pc", "bnh,nhd->bd", (64,), (None, 4, 32), None),
        # Multi-head attention value projection: ab,bcd->acd (one reduced: b)
        ("mha_value_grouped", "ab,bcd->acd", (8, 32), (None, 64), 32),
        ("mha_value_pc", "ab,bcd->acd", (8, 32), (None, 64), None),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_grouped_multi_reduced_axes(
        self, equation, output_shape, input_shape, block_size
    ):
        """Test int4 grouped quantization with multiple reduced axes."""
        layer = layers.EinsumDense(
            equation=equation,
            output_shape=output_shape,
        )
        layer.build(input_shape)

        batch_input_shape = (2,) + input_shape[1:]
        x = np.random.random(batch_input_shape).astype("float32")
        y_float = layer(x)

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify quantization works
        y_quantized = layer(x)
        self.assertEqual(y_float.shape, y_quantized.shape)

        # Verify reasonable accuracy
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 0.1)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_multi_reduced_axes_scale_shape(self):
        """Test that scale shape is correct for multi-reduced-axis equations."""
        # Equation: bnh,nhd->bd (two reduced axes: n, h)
        # Kernel shape: (n=4, h=32, d=64)
        # Reduced axes: n, h (total reduced dim = 4 * 32 = 128)
        layer = layers.EinsumDense(
            equation="bnh,nhd->bd",
            output_shape=(64,),
        )
        layer.build((None, 4, 32))

        block_size = 64
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Total reduced dim = 4 * 32 = 128, n_groups = ceil(128/64) = 2
        # scale shape is (n_groups, columns) =
        # (n_groups, output_dim)
        # For this equation, output_dim=64, n_groups=ceil(128/64)=2
        kernel_scale = layer.kernel_scale
        self.assertIsNotNone(kernel_scale)
        # (n_groups, columns) where n_groups is at index 0
        self.assertEqual(kernel_scale.shape[0], 2)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_multi_reduced_axes_serialization(self):
        """Test serialization with multi-reduced-axis equations."""
        # Equation: bnh,nhd->bd (two reduced axes: n, h)
        layer = layers.EinsumDense(
            equation="bnh,nhd->bd",
            output_shape=(64,),
        )
        layer.build((None, 4, 32))

        config = Int4QuantizationConfig(block_size=64)
        layer.quantize("int4", config=config)

        x = np.random.random((2, 4, 32)).astype("float32")
        y_before = layer(x)

        # Save and load model
        model = models.Sequential([layer])
        path = f"{self.get_temp_dir()}/model.keras"
        model.save(path)
        loaded_model = load_model(path)

        # Verify outputs match
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after, atol=1e-5)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_multi_reduced_vs_single_reduced(self):
        """Compare grouped quantization: single vs multi-reduced axes."""
        block_size = 64

        # Single reduced axis: bd,df->bf (reduced: d)
        layer_single = layers.EinsumDense(
            equation="bd,df->bf",
            output_shape=(32,),
        )
        layer_single.build((None, 128))

        # Multi reduced axes: bnh,nhd->bd (reduced: n, h)
        # n=4, h=32 gives total reduced dim = 128
        layer_multi = layers.EinsumDense(
            equation="bnh,nhd->bd",
            output_shape=(128,),
        )
        layer_multi.build((None, 4, 32))

        # Quantize both with grouped quantization
        config = Int4QuantizationConfig(block_size=block_size)
        layer_single.quantize("int4", config=config)
        layer_multi.quantize("int4", config=config)

        # Both should use grouped quantization (block_size stored)
        self.assertEqual(
            layer_single._quantized_weight().scheme.group_size, block_size
        )
        self.assertEqual(
            layer_multi._quantized_weight().scheme.group_size, block_size
        )

        # Verify forward pass works for both
        x_single = np.random.random((2, 128)).astype("float32")
        x_multi = np.random.random((2, 4, 32)).astype("float32")

        y_single = layer_single(x_single)
        y_multi = layer_multi(x_multi)

        self.assertEqual(y_single.shape, (2, 32))
        self.assertEqual(y_multi.shape, (2, 128))

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_subchannel_g_idx_created(self, block_size):
        """Test that g_idx is created for sub-channel int4 quantization."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(32,),
            bias_axes="c",
        )
        layer.build((None, 128))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify g_idx is created
        self.assertIsNotNone(layer.g_idx)

        # Verify g_idx shape (128 = input_dim = reduced dimension)
        self.assertEqual(layer.g_idx.shape, (128,))

        # Verify g_idx values (should map each row to its group)
        expected_g_idx = np.arange(128) // block_size
        self.assertAllClose(layer.g_idx, expected_g_idx)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_perchannel_no_g_idx(self):
        """Test that per-channel int4 does NOT create g_idx."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(32,),
            bias_axes="c",
        )
        layer.build((None, 64))

        config = Int4QuantizationConfig(block_size=None)  # Per-channel
        layer.quantize("int4", config=config)

        # Verify g_idx is NOT created for per-channel
        self.assertIsNone(layer.g_idx)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_subchannel_g_idx_serialization(self):
        """Test that g_idx is properly serialized and deserialized."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(32,),
            bias_axes="c",
        )
        layer.build((None, 128))
        block_size = 64

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        x = np.random.random((2, 128)).astype("float32")
        y_before = layer(x)
        g_idx_before = layer.g_idx

        # Save and load
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "einsum_int4_g_idx_model.keras"
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify g_idx is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertIsNotNone(loaded_layer.g_idx)
        self.assertAllClose(loaded_layer.g_idx, g_idx_before)

        # Verify outputs match
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_string_matches_default_config(self):
        """`quantize("int4")` must resolve to the exact same scheme as the
        default `Int4QuantizationConfig()` (grouped, block_size=128): identical
        variables (names/shapes/dtypes/values), dtype policy, and outputs."""
        input_dim, output_dim = 256, 64
        kernel = np.random.RandomState(0).randn(input_dim, output_dim)
        kernel = kernel.astype("float32")

        def build_quantized(config):
            layer = layers.EinsumDense(
                equation="ab,bc->ac", output_shape=(output_dim,), bias_axes="c"
            )
            layer.build((None, input_dim))
            layer._kernel.assign(kernel)
            layer.quantize("int4", config=config)
            return layer

        layer_str = build_quantized(config=None)
        layer_cfg = build_quantized(config=Int4QuantizationConfig())

        self.assertEqual(layer_str.dtype_policy.name, "int4/128_from_float32")
        self.assertEqual(
            layer_str.dtype_policy.name, layer_cfg.dtype_policy.name
        )

        vars_str = {v.name: v for v in layer_str.weights}
        vars_cfg = {v.name: v for v in layer_cfg.weights}
        self.assertEqual(set(vars_str.keys()), set(vars_cfg.keys()))
        for name, v_str in vars_str.items():
            v_cfg = vars_cfg[name]
            self.assertEqual(tuple(v_str.shape), tuple(v_cfg.shape))
            self.assertEqual(
                standardize_dtype(v_str.dtype),
                standardize_dtype(v_cfg.dtype),
            )
            self.assertAllClose(v_str, v_cfg)

        x = np.random.RandomState(1).randn(4, input_dim).astype("float32")
        self.assertAllClose(layer_str(x), layer_cfg(x))

    @parameterized.named_parameters(
        ("none", None),
        ("neg1", -1),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_per_channel_escape_hatch(self, block_size):
        """`block_size=None` and `block_size=-1` both select the per-channel
        escape hatch: no zero-point / g_idx and an `int4/-1` dtype policy."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac", output_shape=(64,), bias_axes="c"
        )
        layer.build((None, 256))
        layer.quantize(
            "int4", config=Int4QuantizationConfig(block_size=block_size)
        )
        self.assertIsNone(layer.kernel_zero)
        self.assertIsNone(layer.g_idx)
        self.assertEqual(layer.dtype_policy.name, "int4/-1_from_float32")

    @parameterized.named_parameters(
        ("block_none", "None", True),
        ("block_neg1", "-1", True),
        ("block_128", "128", False),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_policy_string_reload_builds_right_variables(
        self, block_token, per_channel
    ):
        """Old checkpoints identified only by their int4 dtype-policy string
        must deserialize and rebuild the correct variables. Covers the legacy
        `int4/None` spelling of per-channel plus `int4/-1` and `int4/128`."""
        input_dim, output_dim = 256, 64
        policy = f"int4/{block_token}_from_float32"
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(output_dim,),
            bias_axes="c",
            dtype=policy,
        )
        layer.build((None, input_dim))

        self.assertEqual(layer.quantization_mode, "int4")
        if per_channel:
            self.assertIsNone(layer.g_idx)
        else:
            self.assertIsNotNone(layer.g_idx)
            self.assertEqual(tuple(layer.g_idx.shape), (input_dim,))
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")

        x = np.random.random((2, input_dim)).astype("float32")
        self.assertEqual(tuple(layer(x).shape), (2, output_dim))

    def test_int8_weight_only_matches_dense(self):
        """Weight-only int8 `EinsumDense` and `Dense` agree on every backend.

        Both contract the float inputs against the int8 kernel and de-scale
        the outputs, so the two layers must produce the same values from the
        same float kernel and bias."""
        input_dim, units = 16, 8
        dense = layers.Dense(units)
        dense.build((None, input_dim))
        dense.bias.assign(np.random.random((units,)).astype("float32"))
        einsum_dense = layers.EinsumDense(
            "ab,bc->ac", output_shape=(units,), bias_axes="c"
        )
        einsum_dense.build((None, input_dim))
        einsum_dense.kernel.assign(dense.kernel)
        einsum_dense.bias.assign(dense.bias)

        dense.quantize(
            "int8", config=Int8QuantizationConfig(activation_quantizer=None)
        )
        einsum_dense.quantize(
            "int8", config=Int8QuantizationConfig(activation_quantizer=None)
        )

        x = np.random.random((4, input_dim)).astype("float32")
        self.assertAllClose(einsum_dense(x), dense(x), atol=1e-6, rtol=1e-6)

    @parameterized.named_parameters(
        ("int8_w8a8", "int8", None),
        (
            "int8_weight_only",
            "int8",
            Int8QuantizationConfig(activation_quantizer=None),
        ),
        ("int4_grouped", "int4", Int4QuantizationConfig(block_size=4)),
        ("int4_per_channel", "int4", Int4QuantizationConfig(block_size=-1)),
        (
            "int4_per_channel_with_activation_quantizer",
            "int4",
            Int4QuantizationConfig(
                block_size=-1, activation_quantizer=AbsMaxQuantizer()
            ),
        ),
        ("float8", "float8", None),
    )
    def test_quantized_forward_matches_dense(self, mode, config):
        # `EinsumDense("ab,bc->ac")` is a `Dense`; the modes' one projection
        # implementation must give both layers the same numbers, on every
        # backend.
        units, input_dim = 6, 12
        dense = layers.Dense(units)
        dense.build((None, input_dim))
        einsum = layers.EinsumDense(
            "ab,bc->ac", output_shape=(units,), bias_axes="c"
        )
        einsum.build((None, input_dim))
        einsum._kernel.assign(dense._kernel)
        bias = np.random.random((units,)).astype("float32")
        dense.bias.assign(bias)
        einsum.bias.assign(bias)

        dense.quantize(mode, config=config)
        einsum.quantize(mode, config=config)

        x = np.random.random((4, input_dim)).astype("float32")
        self.assertAllClose(
            dense(x, training=False),
            einsum(x, training=False),
            atol=1e-6,
            rtol=1e-6,
        )


class EinsumDenseLoRATest(testing.TestCase):
    def test_float8_refuses_lora(self):
        # Refused in either order, before the layer changes.
        message = "lora is not currently supported with FLOAT8"
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(4,))
        layer.build((None, 3))
        layer.quantize("float8")
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.enable_lora(2)
        self.assertFalse(layer.lora_enabled)

        layer = layers.EinsumDense("ab,bc->ac", output_shape=(4,))
        layer.build((None, 3))
        layer.enable_lora(2)
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.quantize("float8")
        self.assertIsNone(layer.quantization_mode)


class EinsumDenseLoRAEquationsTest(testing.TestCase):
    @parameterized.named_parameters(
        ("precast_int8", "...b,bc->...c", (4, 3, 8), (8,), "int8"),
        ("postcast_int8", "bc...,cd->bd...", (2, 8, 2, 3), (4,), "int8"),
        ("permuted_int8", "abc,cde->abed", (4, 3, 8), (3, 5, 4), "int8"),
        ("reduced_last_int8", "ibnd,hnd->ibh", (2, 3, 4, 8), (3, 6), "int8"),
        ("postcast_int4", "bc...,cd->bd...", (2, 8, 2, 3), (4,), "int4"),
        ("permuted_int4", "abc,cde->abed", (4, 3, 8), (3, 5, 4), "int4"),
    )
    def test_quantized_lora_delta_matches_float(
        self, equation, input_shape, output_shape, mode
    ):
        # The LoRA update is applied on top of the quantized contraction in
        # low-rank form. Its contribution must match the float layer's for
        # every equation, including the ones where the kernel's last axis
        # is not the output's last axis.
        x = np.random.random(input_shape).astype("float32")
        config = (
            Int4QuantizationConfig(block_size=-1) if mode == "int4" else None
        )

        def lora_delta(quantize):
            layer = layers.EinsumDense(equation, output_shape=output_shape)
            layer.build(input_shape)
            layer.kernel.assign(
                np.random.default_rng(0).random(layer.kernel.shape) - 0.5
            )
            if quantize:
                layer.quantize(mode, config=config)
            before = layer(x)
            layer.enable_lora(2)
            rng = np.random.default_rng(1)
            layer.lora_kernel_a.assign(
                rng.random(layer.lora_kernel_a.shape) - 0.5
            )
            layer.lora_kernel_b.assign(
                rng.random(layer.lora_kernel_b.shape) - 0.5
            )
            return ops.convert_to_numpy(layer(x)) - ops.convert_to_numpy(before)

        # TPU matmuls run at bfloat16 precision by default, which moves the
        # two paths apart by up to about 5e-3.
        self.assertAllClose(
            lora_delta(quantize=True),
            lora_delta(quantize=False),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
