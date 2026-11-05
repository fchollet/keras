import math
import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import quantizers
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing import test_utils


class DenseTest(testing.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", {"axis": 0}, {}),
        (
            "int4",
            "int4",
            {"axis": 0, "value_range": (-8, 7), "output_dtype": "int8"},
            {"axis": -1},
        ),
        ("int8_weight_only", "int8", {"axis": 0}, None),
    )
    def test_dense_quantize_config(
        self, mode, weight_quantizer_args, activation_quantizer_args
    ):
        """Test Dense quantization with QuantizationConfig."""
        layer = layers.Dense(units=32)
        layer.build((None, 8))

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
        x = np.random.random((2, 8)).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 32))

        if mode == "int4":
            # Verify kernel is int8 (packed int4)
            self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")

    def test_dense_basics(self):
        # 2D case, no bias.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
                "activation": "relu",
                "kernel_initializer": "random_uniform",
                "bias_initializer": "ones",
                "use_bias": False,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # 3D case, some regularizers.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "bias_regularizer": "l2",
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    @parameterized.named_parameters(
        ("zero", 0),
        ("negative", -3),
        ("float", 2.5),
        ("none", None),
        ("string", "64"),
    )
    def test_dense_invalid_units_raises(self, units):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.Dense(units)

    def test_dense_correctness(self):
        # With bias and activation.
        layer = layers.Dense(units=2, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
                np.array([5.0, -6.0]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertAllClose(layer(inputs), [[10.0, 0.0]])

        # Just a kernel matmul.
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_errors(self):
        with self.assertRaisesRegex(ValueError, "incompatible with the layer"):
            layer = layers.Dense(units=2, activation="relu")
            layer(keras_tensor.KerasTensor((1, 2)))
            layer(keras_tensor.KerasTensor((1, 3)))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_dense_sparse(self):
        import tensorflow as tf

        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
            },
            input_shape=(2, 3),
            input_sparse=True,
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
        )

        inputs = 4 * backend.random.uniform((10, 10))
        inputs = tf.sparse.from_dense(tf.nn.dropout(inputs, 0.8))

        inputs = np.random.random((10, 10)).astype("float32")
        inputs = np.multiply(inputs, inputs >= 0.8)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            inputs = tf.sparse.from_dense(inputs)
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            inputs = jax_sparse.BCOO.fromdense(inputs)
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        layer = layers.Dense(units=10)
        outputs = layer(inputs)

        # Verify the computation is the same as if it had been a dense tensor
        expected_outputs = ops.add(
            ops.matmul(
                backend.ops.convert_to_tensor(inputs, sparse=False),
                layer.kernel,
            ),
            layer.bias,
        )
        self.assertAllClose(
            outputs, expected_outputs, tpu_atol=1e-2, tpu_rtol=1e-2
        )

        # Verify the gradient is sparse
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape() as g:
                outputs = layer(inputs)

            self.assertIsInstance(
                g.gradient(outputs, layer.kernel), tf.IndexedSlices
            )

    def test_dense_no_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation=None)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_without_activation_set(self):
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        layer.activation = None
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_with_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )

        inputs = np.array(
            [[-1.0, 2.0]],
        )
        output = layer(inputs)
        expected_output = np.array([[5.0, 0.0]])
        self.assertAllClose(output, expected_output)

    def test_dense_constraints(self):
        layer = layers.Dense(units=2, kernel_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.Dense(units=2, bias_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.enable_lora(4)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)
        self.assertDType(layer.lora_kernel_a, "float32")
        self.assertDType(layer.lora_kernel_b, "float32")

        # Try eager call
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
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
        model.fit(x, y)

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
                layers.Dense(units=16),
            ]
        )
        new_model.build((None, 8))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_lora_with_alpha(self):
        # Create a `Dense` layer and build it.
        layer = layers.Dense(units=8)
        layer.build((None, 4))

        # Enable LoRA with `rank`=2 and `lora_alpha`=3.0.
        layer.enable_lora(2, lora_alpha=3.0)
        self.assertEqual(layer.lora_rank, 2)
        self.assertEqual(layer.lora_alpha, 3.0)

        # Manually compute the expected effective kernel:
        # `effective_kernel_expected` = `base_kernel` +
        # `lora_alpha / lora_rank` * `lora_kernel_a @ lora_kernel_b`
        base_kernel = ops.convert_to_numpy(layer._kernel)
        lora_update = np.matmul(
            ops.convert_to_numpy(layer.lora_kernel_a),
            ops.convert_to_numpy(layer.lora_kernel_b),
        )
        effective_kernel_expected = base_kernel + (3.0 / 2) * lora_update

        # Verify that the effective kernel matches expectation.
        self.assertAllClose(
            ops.convert_to_numpy(layer.kernel), effective_kernel_expected
        )

    def test_lora_weight_name(self):
        class MyModel(models.Model):
            def __init__(self):
                super().__init__(name="mymodel")
                self.dense = layers.Dense(16, name="dense")

            def build(self, input_shape):
                self.dense.build(input_shape)

            def call(self, x):
                return self.dense(x)

        model = MyModel()
        model.build((None, 8))
        model.dense.enable_lora(4)
        self.assertEqual(
            model.dense.lora_kernel_a.path, "mymodel/dense/lora_kernel_a"
        )

    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "lora_rank": 2,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    def test_lora_alpha_argument(self):
        layer = layers.Dense(units=4, lora_rank=2, lora_alpha=16)
        layer.build((None, 3))
        self.assertEqual(layer.lora_alpha, 16)
        self.assertEqual(layer.get_config()["lora_alpha"], 16)

        # The forward pass scales the update by `lora_alpha / lora_rank`.
        # Every value, product and sum is exact in bfloat16, so the check
        # also holds where matmuls round their inputs to bfloat16 (TPU).
        kernel = np.arange(12, dtype="float32").reshape((3, 4)) / 4 - 1
        lora_a = np.array([[1, 0], [0, 1], [1, 1]], dtype="float32") / 2
        lora_b = np.ones((2, 4), dtype="float32")
        layer._kernel.assign(kernel)
        layer.lora_kernel_a.assign(lora_a)
        layer.lora_kernel_b.assign(lora_b)
        x = np.array([[1, 2, 3], [0.5, -1, 2]], dtype="float32")
        bias = ops.convert_to_numpy(layer.bias)
        expected = x @ (kernel + (16 / 2) * lora_a @ lora_b) + bias
        self.assertAllClose(layer(x), expected)

        # A layer built from the config keeps `lora_alpha`.
        restored = layers.Dense.from_config(layer.get_config())
        restored.build((None, 3))
        self.assertEqual(restored.lora_alpha, 16)
        restored.set_weights(layer.get_weights())
        self.assertAllClose(restored(x), expected)

    def test_enable_lora_with_kernel_constraint(self):
        layer = layers.Dense(units=2, kernel_constraint="max_norm")
        with self.assertRaisesRegex(
            ValueError, "incompatible with kernel constraints"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_on_unbuilt_layer(self):
        layer = layers.Dense(units=2)
        with self.assertRaisesRegex(
            ValueError, "Cannot enable lora on a layer that isn't yet built"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_when_already_enabled(self):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(ValueError, "lora is already enabled"):
            layer.enable_lora(rank=2)

    @pytest.mark.requires_trainable_backend
    def test_enable_dora(self):
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.enable_dora(4)
        self.assertLen(layer.trainable_weights, 4)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 5)
        self.assertDType(layer.dora_kernel_a, "float32")
        self.assertDType(layer.dora_kernel_b, "float32")
        self.assertDType(layer.dora_magnitude, "float32")

        # Try eager call
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        _ = layer(x[:2])

        init_dora_a_kernel_value = layer.dora_kernel_a.numpy()
        init_dora_b_kernel_value = layer.dora_kernel_b.numpy()
        init_dora_magnitude_value = layer.dora_magnitude.numpy()

        # Try calling fit()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

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
        new_model = models.Sequential([layers.Dense(units=16)])
        new_model.build((None, 8))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a dora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_dora_with_alpha(self):
        layer = layers.Dense(units=8)
        layer.build((None, 4))

        # Enable DoRA with `rank`=2 and `dora_alpha`=3.0.
        layer.enable_dora(2, dora_alpha=3.0)
        self.assertEqual(layer.dora_rank, 2)
        self.assertEqual(layer.dora_alpha, 3.0)

        # Manually compute the expected effective kernel
        base_kernel = ops.convert_to_numpy(layer._kernel)
        dora_update = np.matmul(
            ops.convert_to_numpy(layer.dora_kernel_a),
            ops.convert_to_numpy(layer.dora_kernel_b),
        )
        W_combined = base_kernel + (3.0 / 2) * dora_update
        norm = np.sqrt(np.sum(np.square(W_combined), axis=0))
        effective_kernel_expected = ops.convert_to_numpy(
            layer.dora_magnitude
        ) * (W_combined / norm)

        # Verify that the effective kernel matches expectation.
        self.assertAllClose(
            ops.convert_to_numpy(layer.kernel), effective_kernel_expected
        )

    def test_dora_rank_argument(self):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "dora_rank": 2,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=3,
            supports_masking=True,
        )

    def test_enable_dora_with_kernel_constraint(self):
        layer = layers.Dense(units=2, kernel_constraint="max_norm")
        with self.assertRaisesRegex(
            ValueError, "DoRA is incompatible with kernel constraints"
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_on_unbuilt_layer(self):
        layer = layers.Dense(units=2)
        with self.assertRaisesRegex(
            ValueError, "Cannot enable dora on a layer that isn't yet built"
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_when_already_enabled(self):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.enable_dora(rank=2)
        with self.assertRaisesRegex(ValueError, "dora is already enabled"):
            layer.enable_dora(rank=2)

    def test_dora_lora_mutual_exclusivity(self):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "LoRA is already enabled. Cannot enable DoRA."
        ):
            layer.enable_dora(rank=2)

        layer2 = layers.Dense(units=2)
        layer2.build((None, 2))
        layer2.enable_dora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "dora is already enabled. Cannot enable LoRA."
        ):
            layer2.enable_lora(rank=2)

    # Test quantization-related methods.

    @parameterized.named_parameters(
        ("int8", "int8", 1e-3),
        ("int4", "int4", 5e-3),
    )
    def test_quantize_int(self, mode, error_threshold):
        if mode == "int4" and testing.tensorflow_uses_gpu():
            self.skipTest("Segfault")
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        x = np.random.random((2, 8))
        y_float = layer(x)
        layer.quantize(mode)

        # Verify the dtype of the weights.
        # The kernel's data type is int8, despite the int4 quantization, because
        # we pack the int4 values into int8.
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            standardize_dtype(layer.kernel_scale.dtype),
            layer.variable_dtype,
        )

        # Verify the correctness of the outputs.
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

        # Check model save / load round-trip.
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Check weights-only save / load round-trip.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.Dense(units=16)])
        new_model.build((None, 8))
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_on_unbuilt_layer(self, mode):
        layer = layers.Dense(units=2)
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize(mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_on_subclass(self, mode):
        class MyDense(layers.Dense):
            pass

        layer = MyDense(units=16)
        layer.build((None, 8))
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode)

        layer.quantize(mode, type_check=False)  # No error

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
        ("ternary", "ternary"),
    )
    def test_quantize_subclass_skipping_super_build_raises(self, mode):
        # Without `super().build()` the layer records no `kernel_shape`;
        # `quantize` refuses before it changes anything.
        class MyDense(layers.Dense):
            def build(self, input_shape):
                self._kernel = self.add_weight(
                    name="kernel", shape=(input_shape[-1], self.units)
                )
                self.bias = None

        layer = MyDense(units=4)
        layer.build((None, 8))
        with self.assertRaisesRegex(ValueError, "kernel_shape"):
            layer.quantize(mode, type_check=False)
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(tuple(layer._kernel.shape), (8, 4))

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
    )
    def test_quantize_when_already_quantized(self, mode):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize(mode)
        for m in ["int8", "int4", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

        layer = layers.Dense(units=2, dtype=f"{mode}_from_float32")
        layer.build((None, 2))
        for m in ["int8", "int4", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 3),
        ("int4", "int4_from_float32", 3),  # kernel + bias + scale
        ("float8", "float8_from_float32", 8),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_by_setting_dtype_policy_forwards_block_size(self):
        # Regression test: assigning a parameterized int4 policy to a built
        # layer must quantize with the policy's block size. Previously the
        # setter forwarded only the bare mode string, so "int4/32" quantized
        # with the default block size (128) while keeping a name that said
        # 32 -- the checkpoint's policy string contradicted its stored
        # weights.
        layer = layers.Dense(units=2)
        layer.build((None, 64))
        layer.dtype_policy = "int4/32_from_float32"
        self.assertEqual(layer._quantized_weight().scheme.group_size, 32)
        # ceil(64 / 32) = 2 groups, one scale row per group.
        self.assertEqual(tuple(layer.kernel_scale.shape), (2, 2))
        self.assertEqual(layer.dtype_policy.name, "int4/32_from_float32")

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_by_setting_dtype_policy_per_channel(self):
        # The per-channel escape hatch ("int4/-1") must also be honored by
        # the dtype-policy setter instead of falling back to the grouped
        # default.
        layer = layers.Dense(units=2)
        layer.build((None, 64))
        layer.dtype_policy = "int4/-1_from_float32"
        self.assertIsNone(layer._quantized_weight().scheme.group_size)
        # Per-channel: one scale per output unit, no zero point, no g_idx.
        self.assertEqual(tuple(layer.kernel_scale.shape), (2,))
        self.assertIsNone(layer.kernel_zero)
        self.assertEqual(layer.dtype_policy.name, "int4/-1_from_float32")

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        x = np.random.random((1, 2))
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
        ("int4", "int4_from_mixed_bfloat16", 1, 2),
        ("float8", "float8_from_mixed_bfloat16", 8, 0),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_dtype_argument(
        self, dtype, num_trainable_weights, num_non_trainable_weights
    ):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={"units": 5, "dtype": dtype},
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.named_parameters(
        ("int8", "int8", 3, 2, 5),
        ("int4", "int4", 3, 4, 7),  # +2 non-trainable for zero and g_idx
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_lora_integration(
        self,
        mode,
        num_trainable_weights,
        num_non_trainable_weights,
        num_torch_params,
    ):
        # Note that saving and loading with lora_enabled and quantized are
        # lossy, so we use a weak correctness test for model outputs (atol=0.5).
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.enable_lora(4)
        layer.quantize(mode)
        self.assertLen(layer.trainable_weights, num_trainable_weights)
        self.assertLen(layer.non_trainable_weights, num_non_trainable_weights)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, num_torch_params)

        # Try calling fit()
        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
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
        new_model = models.Sequential([layers.Dense(**config)])
        new_model.build((None, 8))
        new_model.quantize(mode)
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
            ref_input = tf.random.normal((2, 8))
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
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_quantize_float8(self):
        import ml_dtypes

        from keras.src import quantizers

        layer = layers.Dense(units=32)
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
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.quantize("float8")
        self.assertLen(layer.trainable_weights, 8)
        self.assertLen(layer.non_trainable_weights, 0)

        # Try calling fit()
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_float8_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_float8_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.Dense(**config)])
        new_model.build((None, 8))
        new_model.quantize("float8")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((2, 8))
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
        config = dict(units=16)
        layer = layers.Dense(**config)
        layer.build((None, 8))
        layer.quantize("float8")

        # Try calling with `training=False` and the result must match
        # `training=True` because there is no update.
        x = np.random.random((64, 8))
        y_inference = layer(x, training=False)
        y_training = layer(x, training=True)
        self.assertAllClose(y_inference, y_training)

    @parameterized.named_parameters(("bias", True), ("no_bias", False))
    def test_quantize_float8_weights_order(self, use_bias):
        # A layer quantized in place and a layer built from its policy hold
        # the same variables in the same order.
        layer = layers.Dense(units=16, use_bias=use_bias)
        layer.build((None, 8))
        layer.quantize("float8")
        for v in layer.weights:
            v.assign(np.random.uniform(0.5, 1.5, v.shape))
        new_layer = layers.Dense(
            units=16, use_bias=use_bias, dtype=layer.dtype_policy
        )
        new_layer.build((None, 8))
        self.assertEqual(
            [(v.name, v.shape) for v in new_layer.weights],
            [(v.name, v.shape) for v in layer.weights],
        )
        new_layer.set_weights(layer.get_weights())
        x = np.random.random((2, 8))
        self.assertAllClose(
            new_layer(x, training=False), layer(x, training=False)
        )

    def test_quantize_float8_rebuilt_model_takes_weights(self):
        inputs = layers.Input((8,))
        model = models.Model(inputs, layers.Dense(16)(inputs))
        model.layers[1].quantize("float8")
        for v in model.weights:
            v.assign(np.random.uniform(0.5, 1.5, v.shape))
        x = np.random.random((2, 8))
        y = model(x, training=False)

        # `from_config` builds the layer from its policy.
        revived = models.Model.from_config(model.get_config())
        revived.set_weights(model.get_weights())
        self.assertAllClose(revived(x, training=False), y)

        # The legacy `.h5` format stores the weights in `weights` order.
        temp_filepath = os.path.join(self.get_temp_dir(), "float8_model.h5")
        model.save(temp_filepath)
        reloaded = saving.load_model(temp_filepath)
        self.assertAllClose(reloaded(x, training=False), y)

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8_trained_model_reloads_optimizer_state(self):
        # Gradient accumulators are stored by position, one per trainable
        # variable, in the order of `trainable_variables`.
        inputs = layers.Input((8,))
        model = models.Model(inputs, layers.Dense(16)(inputs))
        model.layers[1].quantize("float8")
        model.compile(
            optimizer=optimizers.SGD(gradient_accumulation_steps=2),
            loss="mse",
        )
        x = np.random.random((4, 8))
        model.fit(x, np.random.random((4, 16)), batch_size=4, verbose=0)

        temp_filepath = os.path.join(self.get_temp_dir(), "float8_model.keras")
        model.save(temp_filepath)
        reloaded = saving.load_model(temp_filepath)
        self.assertEqual(
            [v.shape for v in reloaded.optimizer.variables],
            [v.shape for v in model.optimizer.variables],
        )
        for v, ref in zip(
            reloaded.optimizer.variables, model.optimizer.variables
        ):
            self.assertAllClose(v, ref)

    def test_gptq_serialization(self):
        """Test that a GPTQ-quantized layer can be serialized and deserialized
        correctly."""
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        config = layer.get_config()
        new_layer = layers.Dense.from_config(config)
        new_layer.build((None, 8))
        self.assertEqual(new_layer.quantization_mode, "gptq")

    def test_awq_serialization(self):
        """Test that an AWQ-quantized layer can be serialized and deserialized
        correctly."""
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )
        config = layer.get_config()
        new_layer = layers.Dense.from_config(config)
        new_layer.build((None, 8))
        self.assertEqual(new_layer.quantization_mode, "awq")

    def test_gptq_uncalibrated_save_raises(self):
        """Saving a GPTQ layer that was never calibrated must raise."""
        layer = layers.Dense(units=16)
        layer.build((None, 8))
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
        layer = layers.Dense(units=16)
        layer.build((None, 8))
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
        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize("int4")
        packed_kernel = layer._kernel
        # unpack [in, ceil(out/2)] -> [in, out]
        expected = quantizers.unpack_int4(packed_kernel, layer.units, axis=-1)
        self.assertAllClose(layer.kernel, expected)

    @parameterized.named_parameters(
        ("float8", "float8"), ("ternary", "ternary")
    )
    def test_modes_without_lora_support_refuse_lora(self, mode):
        # The float8 forward pass has no term for a LoRA update, and
        # re-ternarizing a merged save is not idempotent. LoRA is refused
        # in either order, before the layer changes.
        message = f"lora is not currently supported with {mode.upper()}"
        layer = layers.Dense(4)
        layer.build((None, 3))
        layer.quantize(mode)
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.enable_lora(2)
        self.assertFalse(layer.lora_enabled)

        layer = layers.Dense(4)
        layer.build((None, 3))
        layer.enable_lora(2)
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.quantize(mode)
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)

        # `model.quantize` skips such a layer as unsupported.
        model = models.Sequential([layers.Input((3,)), layer])
        report = model.quantize(mode, verbose=False)
        self.assertIsNone(layer.quantization_mode)
        self.assertEqual(report.skipped, [(layer.path, report.SKIP_NO_SUPPORT)])

    def test_legacy_load_own_variables(self):
        # In previous versions, `load_own_variables` accepted a store with
        # numeric keys.
        float32_store = {
            "0": np.random.random((8, 16)).astype("float32"),
            "1": np.random.random((16,)).astype("float32"),
        }
        int8_store = {
            "0": np.random.randint(-128, 127, size=(8, 16), dtype="int8"),
            "1": np.random.random((16,)).astype("float32"),
            "2": np.random.random((16,)).astype("float32"),  # kernel_scale.
        }
        int4_store = {
            # kernel is [in, ceil(out/2)] = [8, 8]
            "0": np.random.randint(-128, 127, size=(8, 8), dtype="int8"),
            "1": np.random.random((16,)).astype("float32"),
            "2": np.random.random((16,)).astype("float32"),  # kernel_scale.
        }
        float8_store = {
            "0": np.random.random((8, 16)).astype("float32"),
            "1": np.random.random((16,)).astype("float32"),
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
            "0": np.random.random((16,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(8, 8), dtype="uint8"),
            # kernel_scale: [n_groups, out].
            "2": np.random.random((1, 16)).astype("float32"),
            # kernel_zero: [n_groups, out].
            "3": np.random.random((1, 16)).astype("uint8"),
            # g_idx: legacy checkpoints stored the integer group indices as
            # float32; they load into the float32 g_idx variable unchanged.
            "4": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="float32"),
        }
        awq_store = {
            "0": np.random.random((16,)).astype("float32"),  # bias
            "1": np.random.randint(0, 16, size=(8, 8), dtype="uint8"),  # kernel
            "2": np.random.random((1, 16)).astype("float32"),  # scale
            "3": np.random.random((1, 16)).astype("uint8"),  # zero
            "4": np.random.random((8,)).astype("float32"),  # awq_scales
            # g_idx saved as int32 by a newer checkpoint; the cast on load
            # brings it into the float32 storage variable (see above).
            "5": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int32"),
        }

        # Test float32 layer.
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.load_own_variables(float32_store)
        self.assertAllClose(layer._kernel, float32_store["0"])
        self.assertAllClose(layer.bias, float32_store["1"])

        # Test int8-quantized layer.
        layer = layers.Dense(units=16, dtype="int8_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(int8_store)
        self.assertAllClose(layer._kernel, int8_store["0"])
        self.assertAllClose(layer.bias, int8_store["1"])
        self.assertAllClose(layer.kernel_scale, int8_store["2"])

        # Test int4-quantized layer.
        layer = layers.Dense(units=16, dtype="int4_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(int4_store)
        self.assertAllClose(layer._kernel, int4_store["0"])
        self.assertAllClose(layer.bias, int4_store["1"])
        self.assertAllClose(layer.kernel_scale, int4_store["2"])

        # Test float8-quantized layer.
        layer = layers.Dense(units=16, dtype="float8_from_float32")
        layer.build((None, 8))
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
        layer = layers.Dense(units=16, dtype="gptq/4/8_from_float32")
        layer.build((None, 8))
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
        layer = layers.Dense(units=16, dtype="awq/4/8_from_float32")
        layer.build((None, 8))
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
    def _build_dense_for_mode(mode, input_dim=256, units=64):
        """Builds a `Dense` layer populated for `mode`'s serialization spec."""
        if mode == "none":
            layer = layers.Dense(units=units)
            layer.build((None, input_dim))
        elif mode == "int8":
            layer = layers.Dense(units=units, dtype="int8_from_float32")
            layer.build((None, input_dim))
        elif mode == "int4_per_channel":
            layer = layers.Dense(units=units, dtype="int4_from_float32")
            layer.build((None, input_dim))
        elif mode == "int4_grouped":
            layer = layers.Dense(units=units)
            layer.build((None, input_dim))
            layer.quantize(
                "int4", config=Int4QuantizationConfig(block_size=128)
            )
        elif mode == "float8":
            layer = layers.Dense(units=units, dtype="float8_from_float32")
            layer.build((None, input_dim))
        elif mode == "ternary":
            layer = layers.Dense(units=units, dtype="ternary_from_float32")
            layer.build((None, input_dim))
        elif mode == "gptq":
            layer = layers.Dense(units=units, dtype="gptq/4/32_from_float32")
            layer.build((None, input_dim))
        elif mode == "awq":
            layer = layers.Dense(units=units, dtype="awq/4/32_from_float32")
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
            "ternary",
        ):
            with self.subTest(mode=mode):
                source = self._build_dense_for_mode(mode)
                test_utils.randomize_serialized_variables(source)

                store = {}
                source.save_own_variables(store)
                # Variables are keyed by consecutive integer positions in the
                # mode's serialization spec -- the on-disk contract.
                n_expected = len(test_utils.serialized_variable_names(source))
                self.assertEqual(
                    set(store.keys()), {str(i) for i in range(n_expected)}
                )

                target = self._build_dense_for_mode(mode)
                target.load_own_variables(store)
                test_utils.assert_serialized_variables_equal(
                    self, source, target
                )

    @parameterized.parameters("gptq", "awq")
    def test_gptq_awq_load_from_store(self, mode):
        # GPTQ/AWQ checkpoints carry variables (e.g. `kernel_zero`) that
        # `save_own_variables` does not emit for a freshly built layer, so
        # they are validated through the load path with a fully-populated
        # store.
        source = self._build_dense_for_mode(mode)
        test_utils.randomize_serialized_variables(source)

        target = self._build_dense_for_mode(mode)
        target.load_own_variables(test_utils.positional_store(source))
        self.assertFalse(target.calibration_pending)
        test_utils.assert_serialized_variables_equal(self, source, target)
        # A layer built from its policy runs from its variables.
        x = np.random.rand(2, target.kernel_shape[0]).astype("float32")
        weight = target._quantized_weight().dequantize("float32")
        self.assertAllClose(target(x), ops.matmul(x, weight) + target.bias)

    @parameterized.parameters("gptq", "awq")
    def test_gptq_awq_unbuilt_layer_reports_no_variables(self, mode):
        source = self._build_dense_for_mode(mode)
        test_utils.randomize_serialized_variables(source)

        # An unbuilt layer reports that it has no variables.
        unbuilt = layers.Dense(units=64, dtype=source.dtype_policy)
        with self.assertRaisesRegex(ValueError, "was never built"):
            unbuilt.load_own_variables(test_utils.positional_store(source))

    def test_load_own_variables_reports_clear_errors(self):
        # int8 spec order: kernel ("0"), bias ("1"), kernel_scale ("2").
        source = self._build_dense_for_mode("int8")
        test_utils.randomize_serialized_variables(source)
        store = test_utils.positional_store(source)

        # A missing variable trips the variable-count check with a clear error.
        missing = dict(store)
        del missing["2"]
        with self.assertRaisesRegex(ValueError, "expected 3 variables"):
            self._build_dense_for_mode("int8").load_own_variables(missing)

        # A renumbered/corrupted key keeps the count but fails on the exact
        # missing position.
        corrupted = dict(store)
        corrupted["9"] = corrupted.pop("2")
        with self.assertRaises(KeyError):
            self._build_dense_for_mode("int8").load_own_variables(corrupted)

    def test_int4_gptq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 GPTQ
        kernel."""
        layer = layers.Dense(units=2)
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
        layer = layers.Dense(units=16, use_bias=False)
        layer.build((None, 8))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(quantized_kernel_params, original_kernel_params // 2)

    def test_int4_awq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 AWQ
        kernel."""
        layer = layers.Dense(units=2)
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
        layer = layers.Dense(units=16, use_bias=False)
        layer.build((None, 8))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, num_grid_points=10
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(quantized_kernel_params, original_kernel_params // 2)

    def _check_quantizer_config(
        self, quantizer, valid_class, axis, value_range
    ):
        self.assertIsInstance(quantizer, valid_class)
        self.assertEqual(quantizer.axis, axis)

        # Normalize value_range to list
        if value_range is not None:
            self.assertAllEqual(quantizer.value_range, value_range)

    def test_dense_int8_custom_quantizer(self):
        """
        Test custom quantizer serialization for dense layer.
        """
        # Setup
        weight_range = (-127, 127)
        act_range = (-5, 5)
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0, value_range=weight_range),
            activation_quantizer=AbsMaxQuantizer(
                axis=-1, value_range=act_range
            ),
        )

        # Build & Quantize
        layer = layers.Dense(10)
        layer.build((None, 5))
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.Dense.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )
        self._check_quantizer_config(
            new_layer.quantization_config.weight_quantizer,
            AbsMaxQuantizer,
            axis=(0,),
            value_range=weight_range,
        )
        self._check_quantizer_config(
            new_layer.quantization_config.activation_quantizer,
            AbsMaxQuantizer,
            axis=(-1,),
            value_range=act_range,
        )

    def test_dense_int8_weight_only_quantizer(self):
        """
        Test custom quantizer serialization for dense layer with
        weight-only quantization.
        """
        # Setup
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0),
            activation_quantizer=None,
        )

        # Build & Quantize
        layer = layers.Dense(10)
        layer.build((None, 5))
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.Dense.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )
        self.assertIsInstance(
            new_layer.quantization_config.weight_quantizer, AbsMaxQuantizer
        )
        self.assertIsNone(new_layer.quantization_config.activation_quantizer)

    def test_dense_int4_custom_quantizer(self):
        """
        Test custom quantizer serialization for dense layer with
        int4 quantization.
        """
        # Setup - custom quantizers require per-channel mode (block_size=None)
        weight_range = (-8, 7)
        act_range = (-2, 2)
        config = Int4QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0, value_range=weight_range),
            activation_quantizer=AbsMaxQuantizer(
                axis=-1, value_range=act_range
            ),
            block_size=None,
        )

        # Build & Quantize
        layer = layers.Dense(10)
        layer.build((None, 5))
        layer.quantize("int4", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.Dense.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int4QuantizationConfig
        )
        self._check_quantizer_config(
            new_layer.quantization_config.weight_quantizer,
            AbsMaxQuantizer,
            axis=(0,),
            value_range=weight_range,
        )
        self._check_quantizer_config(
            new_layer.quantization_config.activation_quantizer,
            AbsMaxQuantizer,
            axis=(-1,),
            value_range=act_range,
        )

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
        input_dim, output_dim = 256, 64
        layer = layers.Dense(units=output_dim)
        layer.build((None, input_dim))

        x = np.random.random((2, input_dim)).astype("float32")
        y_float = layer(x)

        # Create config with specified block_size
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify block_size is stored
        self.assertEqual(
            layer._quantized_weight().scheme.group_size,
            None if block_size in (None, -1) else block_size,
        )

        # Verify kernel_scale shape
        if block_size is None or block_size == -1:
            # Per-channel: one scale per output unit
            expected_scale_shape = (output_dim,)
        else:
            # Sub-channel: (n_groups, out_features)
            n_groups = math.ceil(input_dim / block_size)
            expected_scale_shape = (n_groups, output_dim)

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
        layer = layers.Dense(units=32)
        layer.build((None, 128))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Get output before serialization
        x = np.random.random((2, 128)).astype("float32")
        y_before = layer(x)

        # Save and load model to test full serialization roundtrip
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "int4_block_size_model.keras"
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
        input_dim, output_dim = 128, 64
        layer = layers.Dense(units=output_dim)
        layer.build((None, input_dim))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)
        layer.enable_lora(rank=4)

        x = np.random.random((2, input_dim)).astype("float32")

        # Should run without error
        y = layer(x)
        self.assertEqual(y.shape, (2, output_dim))

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_grouped_merged_save_keeps_lora_update(self):
        # A merged save re-quantizes the dequantized kernel plus the LoRA
        # update, so every stored weight must land within half a code step
        # of that sum. A zero update cannot tell this from a merge that
        # drops the update, so the update here is as large as the kernel.
        input_dim, units, block_size = 12, 16, 4
        inputs = layers.Input((input_dim,))
        layer = layers.Dense(units, use_bias=False, name="target")
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
        # Half a step of the re-quantized grid, per group and column.
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
        layer_pc = layers.Dense(units=output_dim)
        layer_pc.build((None, input_dim))
        config_pc = Int4QuantizationConfig(block_size=None)
        layer_pc.quantize("int4", config=config_pc)

        # Grouped layer
        layer_grouped = layers.Dense(units=output_dim)
        layer_grouped.build((None, input_dim))
        config_grouped = Int4QuantizationConfig(block_size=block_size)
        layer_grouped.quantize("int4", config=config_grouped)

        # Verify different scale shapes
        self.assertEqual(layer_pc.kernel_scale.shape, (output_dim,))
        # scale shape is (n_groups, out_features)
        self.assertEqual(
            layer_grouped.kernel_scale.shape,
            (input_dim // block_size, output_dim),
        )

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_subchannel_g_idx_created(self, block_size):
        """Test that g_idx is created for sub-channel int4 quantization."""
        input_dim, output_dim = 256, 64
        layer = layers.Dense(units=output_dim)
        layer.build((None, input_dim))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify g_idx is created
        self.assertIsNotNone(layer.g_idx)

        # Verify g_idx shape
        self.assertEqual(layer.g_idx.shape, (input_dim,))

        # Verify g_idx values (should map each row to its group)
        expected_g_idx = np.arange(input_dim) // block_size
        self.assertAllClose(layer.g_idx, expected_g_idx)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_perchannel_no_g_idx(self):
        """Test that per-channel int4 does NOT create g_idx."""
        layer = layers.Dense(units=32)
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
        input_dim, output_dim = 128, 32
        block_size = 64

        layer = layers.Dense(units=output_dim)
        layer.build((None, input_dim))

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        x = np.random.random((2, input_dim)).astype("float32")
        y_before = layer(x)
        g_idx_before = layer.g_idx

        # Save and load
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "int4_g_idx_model.keras"
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

    @parameterized.named_parameters(("int4", "int4/2"), ("gptq", "gptq/4/2"))
    def test_more_than_256_groups_under_mixed_bfloat16(self, policy):
        # 300 groups of two rows. bfloat16 holds integers exactly only up
        # to 256, so an autocast `g_idx` would send later rows to another
        # group's scale and zero point.
        input_dim = 600
        source = layers.Dense(4, dtype=f"{policy}_from_float32")
        source.build((None, input_dim))
        test_utils.randomize_serialized_variables(source)
        source.g_idx.assign(np.arange(input_dim) // 2)
        store = test_utils.positional_store(source)
        x = np.random.default_rng(0).standard_normal((2, input_dim))
        outputs = []
        for dtype in ("float32", "mixed_bfloat16"):
            layer = layers.Dense(4, dtype=f"{policy}_from_{dtype}")
            layer.build((None, input_dim))
            layer.load_own_variables(store)
            y = layer(x.astype("float32"))
            outputs.append(ops.convert_to_numpy(ops.cast(y, "float32")))
        expected, y = outputs
        atol = 0.02 * np.abs(expected).max()
        self.assertAllClose(y, expected, rtol=0.02, atol=atol)

    def test_quantize_by_setting_dtype_policy_map_uses_layer_entry(self):
        # A map answers `quantization_mode` for its default policy, which is
        # not quantized here; the layer's own entry decides, with its
        # parameters.
        layer = layers.Dense(units=8, name="target")
        layer.build((None, 8))
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["target"] = dtype_policies.get("int4/4_from_float32")
        layer.dtype_policy = policy_map
        self.assertEqual(layer.quantization_mode, "int4")
        self.assertEqual(tuple(layer.kernel_scale.shape), (2, 8))

    def test_float8_reads_its_dtype_policy_map_entry(self):
        # The amax history length comes from the layer's own entry, as it
        # does from a policy passed directly, so the two stores agree.
        policy = dtype_policies.QuantizedFloat8DTypePolicy(
            "float8", "float32", amax_history_length=16
        )
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["dense"] = policy
        direct = layers.Dense(units=4, name="dense", dtype=policy)
        direct.build((None, 6))
        layer = layers.Dense(units=4, name="dense", dtype=policy_map)
        layer.build((None, 6))
        self.assertEqual(tuple(layer.inputs_amax_history.shape), (16,))
        history = np.random.rand(16).astype("float32")
        direct.inputs_amax_history.assign(history)
        store = {}
        direct.save_own_variables(store)
        layer.load_own_variables(store)
        self.assertAllClose(layer.inputs_amax_history, history)

        # A model whose layer holds the map saves and loads the same length.
        inputs = layers.Input((6,))
        outputs = layers.Dense(4, name="dense", dtype=policy_map)(inputs)
        model = models.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), "float8_map.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        reloaded_layer = reloaded.get_layer("dense")
        self.assertEqual(tuple(reloaded_layer.inputs_amax_history.shape), (16,))
        x = np.random.rand(2, 6).astype("float32")
        self.assertAllClose(reloaded(x), model(x))

    def test_already_quantized_error_names_the_dtype_policy_map_entry(self):
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["dense"] = dtype_policies.get("int8_from_float32")
        layer = layers.Dense(units=4, name="dense", dtype=policy_map)
        layer.build((None, 6))
        with self.assertRaisesRegex(
            ValueError, "already quantized with dtype_policy='int8_from_"
        ):
            layer.quantize("int8")

    def test_stateless_call_uses_dtype_policy_map_entry(self):
        # A layer quantized through its `DTypePolicyMap` entry (the map's
        # default policy is not quantized) must dispatch `stateless_call` to
        # the quantized forward exactly as `__call__` does.
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["dense"] = dtype_policies.get("int8_from_float32")
        layer = layers.Dense(units=4, name="dense", dtype=policy_map)
        layer.build((None, 6))
        self.assertEqual(layer.quantization_mode, "int8")
        # A freshly built layer holds zero codes and unit scales, for which
        # the float and quantized forwards coincide; make them distinct.
        layer._kernel.assign(
            np.random.randint(-127, 128, size=(6, 4)).astype("int8")
        )
        layer.kernel_scale.assign(
            np.full(tuple(layer.kernel_scale.shape), 0.37, dtype="float32")
        )
        x = np.random.rand(2, 6).astype("float32")
        y = layer(x)
        y_stateless, _ = layer.stateless_call(
            layer.trainable_variables, layer.non_trainable_variables, x
        )
        self.assertAllClose(y, y_stateless)

    # Ternary quantization tests for Dense.quantize("ternary").

    def test_dense_quantize_ternary_matches_float(self):
        layer = layers.Dense(units=16)
        layer.build((None, 11))
        x = np.random.rand(4, 11).astype("float32")
        y_float = layer(x)

        layer.quantize("ternary")

        self.assertEqual(layer.quantization_mode, "ternary")
        # The float kernel is replaced by the packed codes.
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "uint8")
        # Packed along the output axis: ceil(16 / 5) = 4 bytes per row.
        self.assertEqual(tuple(layer._kernel.shape), (11, 4))

        y_quantized = layer(x)
        # Dense.quantize("ternary") is lossy: float kernel → {-1,0,+1}×beta.
        # Only verify output shape, not numerical equality.
        self.assertEqual(tuple(y_quantized.shape), tuple(y_float.shape))

    def test_dense_quantize_ternary_packed_density(self):
        # units=40 → ceil(40/5)=8 bytes per row; each byte holds 5 trits.
        layer = layers.Dense(units=40)
        layer.build((None, 32))
        layer.quantize("ternary")

        self.assertEqual(tuple(layer._kernel.shape), (32, 8))
        n_bytes = int(np.prod(layer._kernel.shape))
        n_weights = 32 * 40
        bits_per_weight = 8 * n_bytes / n_weights
        self.assertEqual(bits_per_weight, 1.6)
        self.assertLess(n_bytes, n_weights // 2)

    def test_dense_quantize_ternary_kernel_property(self):
        # Dense.kernel property should return an unpacked {-1, 0, +1} tensor.
        layer = layers.Dense(units=8)
        layer.build((None, 6))
        layer.quantize("ternary")

        k = ops.convert_to_numpy(layer.kernel)
        self.assertEqual(k.shape, (6, 8))
        unique_vals = set(np.round(k).astype(np.int32).flat)
        self.assertTrue(
            unique_vals <= {-1, 0, 1},
            f"Expected kernel values in {{-1, 0, 1}}, got {np.unique(k)}",
        )

    def test_dense_quantize_ternary_save_load(self):
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        x = np.random.rand(3, 8).astype("float32")
        y_float = layer(x)
        layer.quantize("ternary")
        y_quantized = layer(x)
        # Dense.quantize("ternary") is lossy: verify shape not exact equality.
        self.assertEqual(tuple(y_quantized.shape), tuple(y_float.shape))

        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "dense_ternary_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertEqual(new_model.layers[0].quantization_mode, "ternary")
        self.assertAllClose(model.predict(x), new_model.predict(x))

        temp_weights = os.path.join(
            self.get_temp_dir(), "dense_ternary_model.weights.h5"
        )
        model.save_weights(temp_weights)
        new_model = models.Sequential([layers.Dense(units=16)])
        new_model.build((None, 8))
        new_model.quantize("ternary")
        new_model.load_weights(temp_weights)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_dense_quantize_ternary_beta_scale(self):
        # With default threshold (None), beta = mean(|W|) is stored in
        # kernel_scale and applied by the ternary forward pass.
        layer = layers.Dense(units=4, use_bias=False)
        layer.build((None, 4))
        kernel = np.array(
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5, 0.5],
            ],
            dtype="float32",
        )
        layer._kernel.assign(kernel)
        beta_expected = float(np.mean(np.abs(kernel)))
        layer.quantize("ternary")

        self.assertAllClose(
            float(ops.convert_to_numpy(layer.kernel_scale)),
            beta_expected,
            atol=1e-5,
        )

    @parameterized.named_parameters(
        ("float16", "float16"), ("mixed_float16", "mixed_float16")
    )
    def test_dense_quantize_ternary_small_kernel_float16(self, dtype):
        # `mean(|W|)` is about 1e-5, so `1 / mean(|W|)` would overflow
        # float16. The multiplier scale keeps the outputs' magnitude.
        rng = np.random.RandomState(0)
        layer = layers.Dense(units=32, use_bias=False, dtype=dtype)
        layer.build((None, 64))
        layer._kernel.assign(rng.randn(64, 32) * 1e-5)
        kernel = ops.convert_to_numpy(ops.cast(layer._kernel, "float32"))
        beta = np.mean(np.abs(kernel))
        codes = np.sign(kernel) * (np.abs(kernel) > 0.5 * beta)
        x = rng.randn(4, 64).astype("float32")
        layer.quantize("ternary")

        y = ops.convert_to_numpy(ops.cast(layer(x), "float32"))
        self.assertAllClose(y, np.matmul(x, codes) * beta, rtol=1e-2, atol=1e-6)

    def test_dense_quantize_ternary_no_bias(self):
        layer = layers.Dense(units=8, use_bias=False)
        layer.build((None, 6))
        x = np.random.rand(2, 6).astype("float32")
        y_float = layer(x)
        layer.quantize("ternary")
        self.assertIsNone(layer.bias)
        y_quantized = layer(x)
        # Dense.quantize("ternary") is lossy: verify shape not exact equality.
        self.assertEqual(tuple(y_quantized.shape), tuple(y_float.shape))

    def test_dense_prebuilt_ternary_mode(self):
        # Dense built with dtype="ternary_from_float32" routes to the
        # ternary build during build — the float kernel never exists.
        layer = layers.Dense(units=8, dtype="ternary_from_float32")
        layer.build((None, 10))
        self.assertTrue(layer.built)
        self.assertEqual(layer.quantization_mode, "ternary")
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "uint8")
        # ceil(8 / 5) = 2 bytes per row
        self.assertEqual(tuple(layer._kernel.shape), (10, 2))
        x = np.random.rand(3, 10).astype("float32")
        y = layer(x)
        self.assertEqual(tuple(y.shape), (3, 8))

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
            layer = layers.Dense(units=output_dim)
            layer.build((None, input_dim))
            layer._kernel.assign(kernel)
            layer.quantize("int4", config=config)
            return layer

        layer_str = build_quantized(config=None)
        layer_cfg = build_quantized(config=Int4QuantizationConfig())

        # Same dtype policy: grouped, block_size=128.
        self.assertEqual(layer_str.dtype_policy.name, "int4/128_from_float32")
        self.assertEqual(
            layer_str.dtype_policy.name, layer_cfg.dtype_policy.name
        )
        self.assertEqual(layer_str._quantized_weight().scheme.group_size, 128)

        # Same variables: names, shapes, dtypes, and values.
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

        # Same outputs.
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
        escape hatch: one scale per output channel, no zero-point / g_idx, and
        an `int4/-1` dtype policy."""
        input_dim, output_dim = 256, 64
        layer = layers.Dense(units=output_dim)
        layer.build((None, input_dim))
        layer.quantize(
            "int4", config=Int4QuantizationConfig(block_size=block_size)
        )
        self.assertEqual(tuple(layer.kernel_scale.shape), (output_dim,))
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
        layer = layers.Dense(units=output_dim, dtype=policy)
        layer.build((None, input_dim))

        self.assertEqual(layer.quantization_mode, "int4")
        if per_channel:
            self.assertEqual(tuple(layer.kernel_scale.shape), (output_dim,))
            self.assertIsNone(layer.g_idx)
        else:
            block_size = int(block_token)
            n_groups = math.ceil(input_dim / block_size)
            self.assertEqual(
                tuple(layer.kernel_scale.shape), (n_groups, output_dim)
            )
            self.assertIsNotNone(layer.g_idx)
            self.assertEqual(tuple(layer.g_idx.shape), (input_dim,))

        # The packed kernel is always [input_dim, ceil(output_dim / 2)] int8.
        self.assertEqual(
            tuple(layer._kernel.shape), (input_dim, (output_dim + 1) // 2)
        )
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "int8")

        # A forward pass runs without error.
        x = np.random.random((2, input_dim)).astype("float32")
        self.assertEqual(tuple(layer(x).shape), (2, output_dim))
