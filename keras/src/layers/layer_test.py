import pickle
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import Input
from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import global_state
from keras.src.backend.common import set_keras_mask
from keras.src.backend.common.remat import RematScope
from keras.src.models import Model
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantization_config import Int4QuantizationConfig


class MockRemat:
    """Mock remat by returning a wrapper Mock calling the original function"""

    def __init__(self):
        self.rematted_functions = {}

    def __call__(self, func):
        if func in self.rematted_functions:
            return self.rematted_functions[func]

        wrapped_func = mock.Mock(wraps=func)
        self.rematted_functions[func] = wrapped_func
        return wrapped_func


class LayerTest(testing.TestCase):
    def test_compute_output_spec(self):
        # Test that implementing compute_output_shape
        # is enough to make compute_output_spec work.

        # Case: single output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return input_shape

        layer = TestLayer()
        self.assertEqual(
            layer.compute_output_spec(backend.KerasTensor((2, 3))).shape, (2, 3)
        )

        # Case: tuple output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return (input_shape, input_shape)

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))

        # Case: list output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return [input_shape, input_shape]

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))

        # Case: dict output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return {"1": input_shape, "2": input_shape}

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].shape, (2, 3))
        self.assertEqual(out["2"].shape, (2, 3))

        # Case: nested tuple output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return (
                    input_shape,
                    (input_shape, input_shape),
                    (input_shape, input_shape),
                )

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertIsInstance(out[1], tuple)
        self.assertEqual(len(out[1]), 2)
        self.assertEqual(out[1][0].shape, (2, 3))
        self.assertEqual(out[1][1].shape, (2, 3))
        self.assertIsInstance(out[2], tuple)
        self.assertEqual(len(out[2]), 2)
        self.assertEqual(out[2][0].shape, (2, 3))
        self.assertEqual(out[2][1].shape, (2, 3))

        # Case: nested dict output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return {
                    "1": input_shape,
                    "2": {"11": input_shape, "22": input_shape},
                }

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].shape, (2, 3))
        self.assertIsInstance(out["2"], dict)
        self.assertEqual(len(out["2"]), 2)
        self.assertEqual(out["2"]["11"].shape, (2, 3))
        self.assertEqual(out["2"]["22"].shape, (2, 3))

    def test_positional_arg_error(self):
        class SomeLayer(layers.Layer):
            def call(self, x, bool_arg):
                if bool_arg:
                    return x
                return x + 1

        x = backend.KerasTensor(shape=(2, 3), name="x")
        with self.assertRaisesRegex(
            ValueError, "Only input tensors may be passed as"
        ):
            SomeLayer()(x, True)

        # This works
        SomeLayer()(x, bool_arg=True)

    @parameterized.named_parameters(
        ("call", "call", None),
        ("compute_output_shape", "compute_output_shape", None),
        (
            "quantized_build",
            "quantized_build",
            {"input_shape": None, "mode": None},
        ),
        ("quantize", "quantize", {"mode": "int8"}),
    )
    def test_not_implemented_error(self, method, args):
        layer = layers.Layer()
        layer.built = True

        with self.assertRaisesRegex(
            NotImplementedError,
            f"does not have a `{method}` method implemented.",
        ):
            if isinstance(args, dict):
                getattr(layer, method)(**args)
            else:
                getattr(layer, method)(args)

    def test_layer_with_remat(self):
        """Test rematerialization on a simple layer."""
        # Create a mock to track calls to remat
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class SomeLayer(layers.Layer):
                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((2, 4))
            layer = SomeLayer()
            # Case 1: Without rematerialization
            output_no_remat = layer(input_tensor)

            # Case 2: With rematerialization
            with RematScope(mode="full"):
                layer = SomeLayer()
                output_with_remat = layer(input_tensor)

            # Assert outputs are the same
            self.assertAllClose(output_no_remat, output_with_remat)

        # Ensure remat was applied in the second case
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_rematerialized_call_none(self):
        layer = layers.Dense(4)
        layer.rematerialized_call(layer.call, ops.ones((2, 3)))

    @parameterized.product(
        mode=("full", "list_of_layers", "larger_than"),
        rate_and_training=((0.0, False), (0.0, True), (0.5, False)),
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_GRADIENT,
        reason="remat requires gradient support",
    )
    def test_remat_dropout_training(self, mode, rate_and_training):
        rate, training = rate_and_training
        with RematScope(
            mode=mode, layer_names=["dropout"], output_size_threshold=0
        ):
            layer = layers.Dropout(rate, name="dropout")
        inputs = ops.ones((2, 3))
        self.assertAllClose(layer(inputs, training=training), inputs)

    @parameterized.parameters("full", "list_of_layers", "larger_than")
    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX-specific test")
    def test_remat_nested_training_gradients(self, mode):
        import jax

        class TrainingLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dropout = layers.Dropout(0.0)

            def build(self, input_shape):
                self.scale = self.add_weight(
                    shape=(input_shape[-1],), initializer="ones"
                )

            def call(self, inputs, mask=None, training=False):
                outputs = self.dropout(inputs, training=training) * self.scale
                outputs = outputs * mask
                return outputs * 2 if training else outputs

        plain = TrainingLayer()
        with RematScope(
            mode=mode, layer_names=["block"], output_size_threshold=0
        ):
            rematerialized = TrainingLayer(name="block")
        inputs = ops.ones((2, 3))
        mask = ops.convert_to_tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        plain.build(inputs.shape)
        rematerialized.build(inputs.shape)
        weights = [variable.value for variable in plain.trainable_variables]

        # Reuse the layers across modes to ensure the flag is captured per call.
        for training in (False, True, False):

            def value_and_grad(layer):
                def loss(weights, inputs, mask):
                    outputs, _ = layer.stateless_call(
                        weights, [], inputs, mask=mask, training=training
                    )
                    return ops.sum(outputs)

                return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(
                    weights, inputs, mask
                )

            expected_value, expected_grads = value_and_grad(plain)
            actual_value, actual_grads = value_and_grad(rematerialized)
            self.assertAllClose(actual_value, expected_value)
            self.assertAllClose(actual_grads[0][0], expected_grads[0][0])
            self.assertAllClose(actual_grads[1], expected_grads[1])
            self.assertAllClose(actual_grads[2], expected_grads[2])

    def test_quantized_layer_with_remat(self):
        """Test rematerialization on a quantized layer."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):
            input_tensor = backend.random.uniform((2, 4))

            # Case 2: With rematerialization
            with RematScope(mode="full"):
                layer = layers.Dense(3)
                layer.build((2, 4))
                layer.quantize("float8")
                layer(input_tensor)

        # Ensure remat was applied
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_calibration_quantization_by_setting_dtype(self, mode):
        """Tests error being raised when dtype is set to GPTQ or AWQ."""
        with self.assertRaisesRegex(
            ValueError,
            f"Implicitly enabling {mode.upper()} quantization.*is not "
            "supported.*requires a calibration dataset",
        ):
            layer = layers.Dense(3)
            layer.build((2, 4))
            layer.dtype_policy = f"{mode}/4/-1_from_float32"

    @pytest.mark.skipif(
        not backend.SUPPORTS_GRADIENT,
        reason="remat requires gradient support",
    )
    def test_functional_model_with_remat(self):
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):
            # Define model inputs
            inputs = Input(shape=(32, 32, 3))

            # just one layer in remat scope
            with RematScope(mode="activations"):
                layer = layers.Dense(64, activation="relu")
                output = layer(layers.Flatten()(inputs))

            # Build the functional model
            model = Model(inputs=inputs, outputs=output)

            # Compile the model
            model.compile(optimizer="adam", loss="mse")

            # Generate dummy data for testing
            x_train = np.random.random((10, 32, 32, 3)).astype(np.float32)
            y_train = np.random.random((10, 64)).astype(np.float32)

            # Run training to ensure `RematScope` is applied correctly
            model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)

        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_wrapper_list_of_layers(self):
        """Test rematerialization using list_of_layers mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def call(self, x):
                    return x + 1

            class OtherLayer(layers.Layer):
                def call(self, x):
                    return x * 2

            remat_layers = ["test_layer"]
            input_tensor = backend.random.uniform((4, 4))

            with RematScope(mode="list_of_layers", layer_names=remat_layers):
                test_layer = TestLayer(name="test_layer")
                other_layer = OtherLayer(name="other_layer")
                output_test = test_layer(input_tensor)
                output_other = other_layer(input_tensor)

            self.assertAllClose(output_test, input_tensor + 1)
            self.assertAllClose(output_other, input_tensor * 2)

        # Ensure remat was applied to the correct layer
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_larger_than_mode(self):
        """Test rematerialization using larger_than mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def compute_output_shape(self, input_shape):
                    return input_shape

                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((100, 100))  # Large tensor

            with RematScope(mode="larger_than", output_size_threshold=5000):
                layer = TestLayer()
                output = layer(input_tensor)

            self.assertAllClose(output, input_tensor + 1)

        # Ensure remat was applied
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_larger_than_mode_high_threshold(self):
        """Test rematerialization using larger_than mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def compute_output_shape(self, input_shape):
                    return input_shape

                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((100, 100))  # Large tensor

            with RematScope(mode="larger_than", output_size_threshold=50000):
                layer = TestLayer()
                output = layer(input_tensor)

            self.assertAllClose(output, input_tensor + 1)

        # Ensure remat was not applied
        self.assertLen(mock_remat.rematted_functions, 0)

    def test_rng_seed_tracking(self):
        class RNGLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_gen = backend.random.SeedGenerator(seed=1337)

            def call(self, x):
                return x * backend.random.normal(x.shape, seed=self.seed_gen)

        layer = RNGLayer()
        self.assertEqual(layer.variables, [layer.seed_gen.state])
        self.assertAllClose(layer.variables[0], [1337, 0])
        layer(np.ones((3, 4)))
        self.assertAllClose(layer.variables[0], [1337, 1])

        # Test tracking in list attributes.
        class RNGListLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_gens = []
                self.seed_gens.append(backend.random.SeedGenerator(seed=1))
                self.seed_gens.append(backend.random.SeedGenerator(seed=10))

            def call(self, x):
                x = x * backend.random.normal(x.shape, seed=self.seed_gens[0])
                x = x * backend.random.normal(x.shape, seed=self.seed_gens[1])
                return x

        layer = RNGListLayer()
        self.assertEqual(
            layer.variables,
            [layer.seed_gens[0].state, layer.seed_gens[1].state],
        )
        self.assertAllClose(layer.variables[0], [1, 0])
        self.assertAllClose(layer.variables[1], [10, 0])
        layer(np.ones((3, 4)))
        self.assertAllClose(layer.variables[0], [1, 1])
        self.assertAllClose(layer.variables[1], [10, 1])

    def test_layer_tracking(self):
        class LayerWithDenseLayers(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)
                self.layer_dict = {
                    "dense2": layers.Dense(units),
                }
                self.layer_list = [layers.Dense(units)]
                self.units = units
                self.seed_generator = backend.random.SeedGenerator(seed=1)

            def build(self, input_shape):
                self.layer_list.append(layers.Dense(self.units))

            def call(self, x):
                x = self.dense1(x)
                x = self.layer_dict["dense2"](x)
                x = self.layer_list[0](x)
                x = self.layer_list[1](x)
                return x

        class ParentLayer(layers.Layer):
            def __init__(self, inner_layer):
                super().__init__()
                self.inner_layer = inner_layer

            def call(self, x):
                return self.inner_layer(x)

        layer = LayerWithDenseLayers(3)
        layer.build((1, 3))
        self.assertLen(layer._layers, 4)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

        layer = ParentLayer(LayerWithDenseLayers(3))
        self.assertLen(layer._layers, 1)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

        layer = ParentLayer(ParentLayer(LayerWithDenseLayers(3)))
        self.assertLen(layer._layers, 1)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

    def test_metric_tracking(self):
        class LayerWithMetric(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)
                self.metric = metrics.MeanSquaredError(name="my_metric")

            def build(self, input_shape):
                self.dense.build(input_shape)

            def call(self, x):
                return self.dense(x)

        class ParentLayerWithMetric(layers.Layer):
            def __init__(self, inner_layer):
                super().__init__()
                self.inner_layer = inner_layer
                self.metric = metrics.MeanSquaredError(name="my_metric")

            def build(self, input_shape):
                self.inner_layer.build(input_shape)

            def call(self, x):
                return self.inner_layer(x)

        layer = LayerWithMetric(3)
        layer.build((1, 3))

        self.assertLen(layer.metrics, 1)
        self.assertLen(layer.metrics_variables, 2)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

        layer = ParentLayerWithMetric(LayerWithMetric(3))
        layer.build((1, 3))

        self.assertLen(layer.metrics, 2)
        self.assertLen(layer.metrics_variables, 4)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

        layer = ParentLayerWithMetric(ParentLayerWithMetric(LayerWithMetric(3)))
        layer.build((1, 3))

        self.assertLen(layer.metrics, 3)
        self.assertLen(layer.metrics_variables, 6)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

    def test_build_on_call(self):
        class LayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)

            def call(self, x):
                return self.dense1(x)

        layer = LayerWithUnbuiltState(2)
        layer(backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 2)

        class KwargsLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)
                self.dense2 = layers.Dense(units)

            def call(self, x1, x2):
                return self.dense1(x1) + self.dense2(x2)

        layer = KwargsLayerWithUnbuiltState(2)
        layer(backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 4)

        layer = KwargsLayerWithUnbuiltState(2)
        layer(x1=backend.KerasTensor((3, 4)), x2=backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 4)

        class DictLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, xs):
                result = self.dense(xs["x1"])
                if xs.get("x2", None) is not None:
                    result += self.dense(xs["x2"])
                return result

        layer = DictLayerWithUnbuiltState(2)
        layer(
            {
                "x1": backend.KerasTensor((3, 4)),
                "x2": backend.KerasTensor((3, 4)),
            }
        )
        self.assertLen(layer.weights, 2)

        layer = DictLayerWithUnbuiltState(2)
        layer({"x1": backend.KerasTensor((3, 4)), "x2": None})
        self.assertLen(layer.weights, 2)

        class ListLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, xs):
                result = self.dense(xs[0])
                if xs[1] is not None:
                    result += self.dense(xs[1])
                return result

        layer = ListLayerWithUnbuiltState(2)
        layer([backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))])
        self.assertLen(layer.weights, 2)

        layer = ListLayerWithUnbuiltState(2)
        layer([backend.KerasTensor((3, 4)), None])
        self.assertLen(layer.weights, 2)

    def test_activity_regularization(self):
        class ActivityRegularizer(layers.Layer):
            def call(self, x):
                return x

        layer = ActivityRegularizer(activity_regularizer="l1")
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 0.01)

        # losses are reset upon call
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 0.01)

        # KerasTensors are no op
        layer = ActivityRegularizer(activity_regularizer="l1")
        layer(layers.Input(batch_shape=(2, 2)))
        self.assertLen(layer.losses, 0)

    @parameterized.named_parameters(
        ("batch_size_0", 0),
        ("batch_size_1", 1),
        ("batch_size_5", 5),
        ("batch_size_10", 10),
    )
    def test_activity_regularization_batch_normalization(self, batch_size):
        class SimpleLayer(layers.Layer):
            def call(self, x):
                return x

        layer = SimpleLayer(activity_regularizer="l2")
        layer(ops.ones((batch_size, 5)) * 2.0)
        self.assertLen(layer.losses, 1)
        expected_loss = 0.0 if batch_size == 0 else 0.2
        self.assertAllClose(layer.losses[0], expected_loss)

    def test_add_loss(self):
        class LossLayer(layers.Layer):
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x

        layer = LossLayer()
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 1.0)

        # losses are reset upon call
        layer = LossLayer()
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 1.0)

        # It works inside a model
        model = models.Sequential([layer])
        model(np.ones((1,)))
        self.assertLen(model.losses, 1)
        self.assertAllClose(model.losses[0], 1.0)

        # It works recursively in nested models
        model = models.Sequential([model])
        model(np.ones((1,)))
        self.assertLen(model.losses, 1)
        self.assertAllClose(model.losses[0], 1.0)

    def test_training_arg_value_resolution(self):
        # Check that even if `training` is not passed
        # to an inner layer, the outer value gets propagated
        # in __call__.
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = layers.Dropout(0.9)

            def call(self, x, training=False):
                return self.dp(x)

        layer = TrainingLayer()
        x = np.ones((4, 4))
        y = layer(x)
        self.assertEqual(ops.min(y), 1)
        y = layer(x, training=True)
        self.assertEqual(ops.min(y), 0)

        # Check that it still works one level deeper.
        class WrappedTrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = TrainingLayer()

            def call(self, x, training=False):
                return self.dp(x)

        layer = WrappedTrainingLayer()
        x = np.ones((4, 4))
        y = layer(x)
        self.assertEqual(ops.min(y), 1)
        y = layer(x, training=True)
        self.assertEqual(ops.min(y), 0)

        # Check that if `training` is passed
        # to an inner layer in call(), the explicitly
        # passed value is what the layer sees.
        class TrainingLayerExplicit(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = layers.Dropout(0.9)

            def call(self, x, training=False):
                return self.dp(x, training=True)

        layer = TrainingLayerExplicit()
        x = np.ones((4, 4))
        y = layer(x, training=False)
        self.assertEqual(ops.min(y), 0)

        # Test that layer interruption does not cause
        # the call context to linger
        class BadLayer(layers.Layer):
            def call(self, x, training=False):
                raise RuntimeError("oops!")

        x = np.ones((4, 4))
        layer = BadLayer()
        try:
            # training=True will be recorded
            # in the call context
            layer(x, training=True)
        except RuntimeError:
            pass
        layer = TrainingLayer()
        # But this layer call should not see it
        y = layer(x)
        self.assertEqual(ops.min(y), 1)

    def test_signature_default_training_does_not_leak(self):
        """A `training=True` call() signature default (as on `Resizing`/
        `CenterCrop`) stays local to that layer: it is not propagated
        through the shared call context to sibling or downstream layers.
        """

        class DefaultsTrainingTrue(layers.Layer):
            def call(self, x, training=True):  # mirrors `Resizing.call`
                return x

        x = np.ones((4, 4))

        # Functional graph: the downstream layer defaults training=None so
        # any value propagated from the upstream layer is directly visible.
        seen = {}

        class RecordTraining(layers.Layer):
            def call(self, x, training=None):
                seen["training"] = training
                return x

        inp = Input((4,))
        out = RecordTraining()(DefaultsTrainingTrue()(inp))
        model = Model(inp, out)
        model(x)
        self.assertIsNone(seen["training"])
        model(x, training=True)
        self.assertTrue(seen["training"])

        # Imperative call: same invariant inside another layer's call().
        class Wrapper(layers.Layer):
            def __init__(self):
                super().__init__()
                self.pre = DefaultsTrainingTrue()
                self.dp = layers.Dropout(0.9)

            def call(self, x):
                return self.dp(self.pre(x))

        layer = Wrapper()
        self.assertEqual(ops.min(layer(x)), 1)
        self.assertEqual(ops.min(layer(x, training=True)), 0)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Some torch ops not implemented for float16 on CPU.",
    )
    def test_mixed_precision(self):
        x = np.ones((4, 4))

        layer = layers.Dense(2, dtype="float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float16")
        self.assertDType(y, "float16")

        layer = layers.Dense(2, dtype="mixed_float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float32")
        self.assertDType(y, "float16")
        self.assertEqual(layer.kernel.dtype, "float32")

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Some torch ops not implemented for float16 on CPU.",
    )
    def test_autocast(self):
        assertDType = self.assertDType

        # A layer with a int dtype (some preprocessing layers do this).
        class InnerLayerOne(layers.Layer):
            def __init__(self):
                super().__init__(dtype="int")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                    dtype="float32",
                )
                self._build_at_init()

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return ops.add(ops.cast(x, "float32"), self.v)

        # A layer that is explicitly full precision.
        class InnerLayerTwo(layers.Layer):
            def __init__(self):
                super().__init__(dtype="float32")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self._build_at_init()

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return ops.add(x, self.v)

        # A layer that is explicitly mixed precision but with autocast=False
        # weight.
        class InnerLayerThree(layers.Layer):
            def __init__(self):
                super().__init__(dtype="mixed_float16")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                    autocast=False,
                )
                self._build_at_init()

            def call(self, x):
                # Should not autocast `self.v`.
                assertDType(self.v, "float32")
                return ops.add(x, self.v)

        # A layer that is explicitly mixed precision with inner layers.
        class MixedPrecisionLayer(layers.Layer):
            def __init__(self):
                super().__init__(dtype="mixed_float16")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.inner_one = InnerLayerOne()
                self.inner_two = InnerLayerTwo()
                self.inner_three = InnerLayerThree()
                self._build_at_init()

            def call(self, x):
                # Should autocast.
                assertDType(self.v, "float16")
                return self.inner_three(
                    self.inner_two(self.inner_one(ops.add(x, self.v)))
                )

        layer = MixedPrecisionLayer()
        y = layer(np.array(0.0))
        self.assertEqual(y, 4.0)

    def test_autocast_with_np_array(self):
        assertDType = self.assertDType

        class CustomLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, x):
                # Here are the assertions.
                assertDType(x[0], "float32")  # Cast to compute_dtype
                assertDType(x[1], "int32")  # Untouched

        x = [np.zeros(1, dtype="float64"), np.zeros(1, dtype="int32")]
        CustomLayer()(x)

    def test_keras_mask_with_autocast(self):
        test_obj = self

        class CustomLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                test_obj.assertDType(x, "float16")
                return x

        x = ops.zeros((1, 2), dtype="float32")
        mask = ops.array([True, False])
        set_keras_mask(x, mask)
        y = CustomLayer(dtype="float16")(x)
        self.assertAllEqual(
            get_keras_mask(y),
            mask,
            msg="Masking is not propagated by Autocast",
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="compute_output_spec not supported with numpy",
    )
    def test_end_to_end_masking(self):
        # Check that masking survives compilation
        model = models.Sequential(
            [
                layers.Embedding(
                    2, 2, mask_zero=True, embeddings_initializer="ones"
                ),
            ]
        )
        model.compile(loss="mse")
        targets = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]])
        loss = model.evaluate(np.array([[1, 0, 0, 1]]), targets, verbose=0)
        self.assertAllClose(loss, 0.0)

    def test_masking(self):
        test_obj = self

        class BasicMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                return x

        layer = BasicMaskedLayer()
        x = backend.ops.numpy.ones((4, 4))
        mask = backend.ops.numpy.ones((4,))
        set_keras_mask(x, mask)
        layer(x)

        layer(backend.ops.numpy.ones((4, 4)), mask=backend.ops.numpy.ones((4,)))

        class NestedInputMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsInstance(x, list)
                test_obj.assertLen(x, 2)
                test_obj.assertIsInstance(mask, list)
                test_obj.assertLen(mask, 2)
                return x

        layer = NestedInputMaskedLayer()
        x1 = backend.ops.numpy.ones((4, 4))
        mask1 = backend.ops.numpy.ones((4,))
        set_keras_mask(x1, mask1)
        x2 = backend.ops.numpy.ones((4, 4))
        mask2 = backend.ops.numpy.ones((4,))
        set_keras_mask(x2, mask2)
        layer([x1, x2])

        layer(
            [backend.ops.numpy.ones((4, 4)), backend.ops.numpy.ones((4, 4))],
            mask=[backend.ops.numpy.ones((4,)), backend.ops.numpy.ones((4,))],
        )

        class PositionalInputsMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x1, x2, x1_mask=None, x2_mask=None):
                test_obj.assertIsNotNone(x1_mask)
                test_obj.assertIsNotNone(x2_mask)
                return x1 + x2

        layer = PositionalInputsMaskedLayer()
        layer(x1, x2)
        layer(x1=x1, x2=x2)

        class PositionalNestedInputsMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x1, x2, x1_mask=None, x2_mask=None):
                test_obj.assertIsInstance(x1, tuple)
                test_obj.assertIsNotNone(x1_mask)
                test_obj.assertIsNotNone(x2_mask)
                test_obj.assertIsInstance(x1_mask, tuple)
                return x1[0] + x1[1] + x2

        layer = PositionalNestedInputsMaskedLayer()
        x1_1 = backend.ops.numpy.ones((4, 4))
        mask1 = backend.ops.numpy.ones((4,))
        set_keras_mask(x1_1, mask1)
        x1_2 = backend.ops.numpy.ones((4, 4))
        mask2 = backend.ops.numpy.ones((4,))
        set_keras_mask(x1_2, mask2)
        x2 = backend.ops.numpy.ones((4, 4))
        mask2 = backend.ops.numpy.ones((4,))
        set_keras_mask(x2, mask2)
        layer((x1_1, x1_2), x2)
        layer(x1=(x1_1, x1_2), x2=x2)

        class MaskUnsetDuringCallLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                set_keras_mask(x, None)  # Unset mask
                return x

        layer = MaskUnsetDuringCallLayer()
        x = backend.ops.numpy.ones((4, 4))
        mask = backend.ops.numpy.ones((4,))
        set_keras_mask(x, mask)
        y = layer(x)
        self.assertAllClose(get_keras_mask(y), mask)

    def test_masking_with_explicit_kwarg_propagation(self):
        """This test validates that an explicit `mask` kwarg is correctly
        used to compute the output mask.
        """

        class PassthroughMaskLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                # The layer itself can use the mask.
                self.used_mask = mask is not None
                return x

        layer = PassthroughMaskLayer()
        # Create an input tensor WITHOUT an attached mask.
        x = backend.ops.numpy.ones((4, 4))
        self.assertIsNone(get_keras_mask(x))

        # Create a mask to be passed explicitly.
        explicit_mask = backend.ops.numpy.array([True, True, False, False])

        # Call the layer, passing the mask as a keyword argument.
        y = layer(x, mask=explicit_mask)

        # Assert that the layer's internal call received the mask.
        self.assertTrue(layer.used_mask)

        # Assert that the output tensor 'y' now has the explicit mask attached
        # for propagation to the next layer.
        self.assertAllClose(get_keras_mask(y), explicit_mask)

    def test_stateless_call(self):
        class TestLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self._seed_generator = backend.random.SeedGenerator(1337)
                self.ntw = self.add_weight(
                    shape=(),
                    initializer="zeros",
                    trainable=False,
                )
                self.tw = self.add_weight(
                    shape=(),
                    initializer="zeros",
                    trainable=True,
                    regularizer="l1",
                )
                self._build_at_init()

            def call(self, x):
                x = backend.ops.convert_to_tensor(x, dtype="float32")
                self.add_loss(ops.sum(x))
                self.ntw.assign(ops.sum(x))
                x = x + backend.random.normal(
                    shape=(), seed=self._seed_generator
                )
                return ops.add(x, ops.add(self.tw, self.ntw))

        data = np.random.random((3, 4))
        layer = TestLayer()
        out = layer(data)
        layer1 = TestLayer()
        out1 = layer1(data)
        # Check that the layer is in fact deterministic
        self.assertAllClose(out, out1)

        # Test stateless_call correctness
        layer2 = TestLayer()
        trainable_variables = layer2.trainable_variables
        non_trainable_variables = layer2.non_trainable_variables
        out2, non_trainable_variables = layer2.stateless_call(
            trainable_variables, non_trainable_variables, data
        )
        self.assertAllClose(out1, out2)
        self.assertEqual(
            len(layer1.non_trainable_variables), len(non_trainable_variables)
        )
        for ref_v, v in zip(
            layer1.non_trainable_variables, non_trainable_variables
        ):
            self.assertAllClose(v, ref_v)

        # Test with loss collection
        layer3 = TestLayer()
        trainable_variables = layer3.trainable_variables
        non_trainable_variables = layer3.non_trainable_variables
        out3, non_trainable_variables, losses = layer3.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
            return_losses=True,
        )
        self.assertAllClose(out1, out3)
        for ref_v, v in zip(
            layer1.non_trainable_variables, non_trainable_variables
        ):
            self.assertAllClose(v, ref_v)
        self.assertLen(losses, 2)
        for ref_loss, loss in zip(layer1.losses, losses):
            self.assertAllClose(loss, ref_loss)

    def test_trainable_setting(self):
        class NonTrainableWeightsLayer(layers.Layer):
            def build(self, _):
                self.w1 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.w2 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=False,
                )
                self.seed = backend.random.SeedGenerator(123)

            def call(self, inputs):
                return inputs

        class NestedNonTrainableWeightsLayer(layers.Layer):
            def build(self, _):
                self.w1 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.w2 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=False,
                )
                self.nested = NonTrainableWeightsLayer()
                self.nested.build(None)

            def call(self, inputs):
                return inputs

        layer = NestedNonTrainableWeightsLayer()
        layer.build(None)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)

        layer.trainable = False
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.trainable_variables), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)
        self.assertEqual(len(layer.non_trainable_variables), 5)
        self.assertFalse(layer.w1.trainable)
        self.assertFalse(layer.nested.w1.trainable)

        layer.trainable = True
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)
        self.assertTrue(layer.w1.trainable)
        self.assertTrue(layer.nested.w1.trainable)

        layer = NestedNonTrainableWeightsLayer(trainable=False)
        layer.build(None)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.trainable_variables), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)
        self.assertEqual(len(layer.non_trainable_variables), 5)

        layer.trainable = True
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)

    def test_build_signature_errors(self):
        class NoShapeSuffix(layers.Layer):
            def build(self, foo_shape, bar):
                self.built = True

            def call(self, foo, bar):
                return foo + bar

        class NonMatchingArgument(layers.Layer):
            def build(self, foo_shape, baz_shape):
                self.built = True

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        class MatchingArguments(layers.Layer):
            def build(self, bar_shape, foo_shape):
                self.foo_shape = foo_shape
                self.bar_shape = bar_shape

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        class SubsetArguments(layers.Layer):
            def build(self, baz_shape, foo_shape):
                self.foo_shape = foo_shape
                self.baz_shape = baz_shape

            def call(self, foo, bar=None, baz=None):
                return foo[:, 0] + bar[:, 0] + baz[:, 0]

        class SingleArgument(layers.Layer):
            def build(self, anything_whatsoever):
                self.foo_shape = anything_whatsoever

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        foo = backend.ops.numpy.ones((4, 1))
        bar = backend.ops.numpy.ones((4, 2))
        baz = backend.ops.numpy.ones((4, 3))
        with self.assertRaisesRegex(
            ValueError,
            r"argument `bar`, which does not end in `_shape`",
        ):
            layer = NoShapeSuffix()
            layer(foo, bar)

        with self.assertRaisesRegex(
            ValueError,
            r"`baz_shape`, but `call\(\)` does not have argument `baz`",
        ):
            layer = NonMatchingArgument()
            layer(foo, bar)

        # Align by name when build and call arguments match.
        layer = MatchingArguments()
        layer(foo, bar)
        self.assertEqual(layer.foo_shape, foo.shape)
        self.assertEqual(layer.bar_shape, bar.shape)

        # Align by name when build supports a subset of call arguments.
        layer = SubsetArguments()
        layer(foo, bar, baz)
        self.assertEqual(layer.foo_shape, foo.shape)
        self.assertEqual(layer.baz_shape, baz.shape)

        # When build has only one argument, match the first call argument.
        layer = SingleArgument()
        layer(foo, bar)
        self.assertEqual(layer.foo_shape, foo.shape)

    def test_training_arg_not_specified(self):
        class NoTrainingSpecified(layers.Layer):
            def __init__(self):
                super().__init__()

            def build(self, input_shape):
                self.activation = layers.Activation("linear")

            def call(self, inputs):
                return self.activation(inputs)

        layer = NoTrainingSpecified()
        inputs = random.uniform(shape=(1, 100, 100, 3))
        layer(inputs, training=True)

    def test_tracker_locking(self):
        class BadLayer(layers.Layer):
            def call(self, x):
                self.w = self.add_weight(initializer="zeros", shape=())
                return x

        layer = BadLayer()
        with self.assertRaisesRegex(
            ValueError,
            "cannot add new elements of state",
        ):
            layer(np.random.random((3, 2)))

    def test_init_after_state_tracking(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                self.some_attr = True
                self.w = backend.Variable(np.random.random((2,)))
                super().__init__()

        layer = MyLayer()
        self.assertEqual(len(layer.weights), 1)

    def test_add_weight_defaults(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight()
                self.w2 = self.add_weight(dtype="int32", trainable=False)
                self.w3 = self.add_weight(dtype="bool", trainable=False)
                self.w4 = self.add_weight(
                    dtype="int32", shape=(2, 2), trainable=False
                )
                self.w5 = self.add_weight(initializer="ones", shape=(2, 2))

        layer = MyLayer()
        self.assertEqual(layer.w1.shape, ())
        self.assertEqual(layer.w1.dtype, "float32")

        self.assertEqual(layer.w2.shape, ())
        self.assertEqual(layer.w2.dtype, "int32")
        self.assertAllClose(backend.ops.convert_to_numpy(layer.w2), 0)

        self.assertEqual(layer.w3.shape, ())
        self.assertEqual(layer.w3.dtype, "bool")
        self.assertAllClose(backend.ops.convert_to_numpy(layer.w3), False)

        self.assertEqual(layer.w4.shape, (2, 2))
        self.assertEqual(layer.w4.dtype, "int32")
        self.assertAllClose(
            backend.ops.convert_to_numpy(layer.w4), np.zeros((2, 2))
        )

        self.assertEqual(layer.w5.shape, (2, 2))
        self.assertEqual(layer.w5.dtype, "float32")
        self.assertAllClose(
            backend.ops.convert_to_numpy(layer.w5), np.ones((2, 2))
        )

    def test_add_weight_string_as_first_positional_arg(self):
        """Test that passing a string as first positional arg to add_weight
        raises a clear error guiding users to use name= keyword."""

        # Case 1: String as only positional arg (e.g. add_weight("matrix"))
        class MyLayer1(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight("my_weight")

        with self.assertRaisesRegex(
            ValueError,
            "name.*keyword argument",
        ):
            MyLayer1()

        # Case 2: String positional + shape kwarg — the exact bug from
        # https://github.com/keras-team/keras/issues/22265
        # In Keras 2 this was valid: add_weight("matrix", shape=(3, 4))
        class MyLayer2(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(
                    "matrix", shape=(3, 4), initializer="zeros"
                )

        with self.assertRaisesRegex(
            ValueError,
            "name.*keyword argument",
        ):
            MyLayer2()

        # Case 3: shape passed both positionally and as keyword
        class MyLayer3(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), shape=(3, 4))

        with self.assertRaisesRegex(
            ValueError,
            "`shape` was passed both positionally and as a keyword argument",
        ):
            MyLayer3()

        # Case 4: positional shape / initializer / dtype must remain valid.
        class MyLayer4(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), "zeros", "float32")

        layer = MyLayer4()
        self.assertEqual(layer.w.shape, (3, 4))
        self.assertEqual(layer.w.dtype, "float32")
        self.assertAllClose(
            backend.ops.convert_to_numpy(layer.w), np.zeros((3, 4))
        )

        # Case 5: too many positional arguments
        class MyLayer5(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), "zeros", "float32", "name")

        with self.assertRaisesRegex(
            TypeError,
            "takes at most 3 positional arguments",
        ):
            MyLayer5()

    def test_remove_weight(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight()

            def custom_remove_w(self):
                self.w = self._untrack_variable(self.w)

            def custom_change_dtype(self):
                self.w = self._untrack_variable(self.w)
                self.w = self.add_weight(
                    initializer="zeros", dtype="int8", trainable=False
                )

        layer = MyLayer()
        self.assertEqual(len(layer.weights), 1)
        layer.custom_remove_w()
        self.assertEqual(len(layer.weights), 0)
        self.assertEqual(layer.w, None)

        layer = MyLayer()
        self.assertEqual(layer.w.dtype, "float32")
        self.assertEqual(layer.w.trainable, True)
        layer.custom_change_dtype()
        self.assertEqual(layer.w.dtype, "int8")
        self.assertEqual(layer.w.trainable, False)

    def test_trainable_init_arg_validation(self):
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable="yes")
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable=1)
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable=None)

    def test_trainable_init_arg(self):
        inputs = layers.Input(shape=(1,))
        layer = layers.Dense(2, trainable=False)
        outputs = layer(inputs)
        model = models.Model(inputs, outputs)

        self.assertFalse(layer.trainable)
        self.assertLen(layer._trainable_variables, 2)
        self.assertLen(layer._non_trainable_variables, 0)
        self.assertLen(layer.trainable_weights, 0)
        self.assertLen(model.trainable_weights, 0)
        self.assertLen(model.non_trainable_weights, 2)

        layer.trainable = True
        self.assertTrue(layer.trainable)
        self.assertLen(layer._trainable_variables, 2)
        self.assertLen(layer._non_trainable_variables, 0)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(model.trainable_weights, 2)
        self.assertLen(model.non_trainable_weights, 0)

    def test_dtype_policy_setter(self):
        layer = layers.Dense(2)
        # Set by string
        layer.dtype_policy = "mixed_bfloat16"
        self.assertEqual(layer.dtype_policy.name, "mixed_bfloat16")
        self.assertEqual(layer.dtype_policy.compute_dtype, "bfloat16")
        self.assertEqual(layer.dtype_policy.variable_dtype, "float32")
        # Set by DTypePolicy
        layer.dtype_policy = dtype_policies.DTypePolicy("mixed_float16")
        self.assertEqual(layer.dtype_policy.name, "mixed_float16")
        self.assertEqual(layer.dtype_policy.compute_dtype, "float16")
        self.assertEqual(layer.dtype_policy.variable_dtype, "float32")
        # Set with DTypePolicyMap
        dtype_policy_map = dtype_policies.DTypePolicyMap()
        layer = layers.Dense(2, dtype=dtype_policy_map)
        layer.build([None, 1])
        layer.dtype_policy = "mixed_bfloat16"
        self.assertIsInstance(
            layer._dtype_policy, dtype_policies.DTypePolicyMap
        )
        self.assertEqual(
            layer._dtype_policy[layer.path],
            dtype_policies.DTypePolicy("mixed_bfloat16"),
        )

    def test_pickle_layer(self):
        layer = layers.Dense(2)
        reloaded = pickle.loads(pickle.dumps(layer))
        self.assertEqual(layer.get_config(), reloaded.get_config())

    def test_serialize_dtype(self):
        assertIsNone = self.assertIsNone
        assertIsNotNone = self.assertIsNotNone

        class AssertionDense(layers.Dense):
            def __init__(self, *args, **kwargs):
                dtype = kwargs["dtype"]
                if isinstance(dtype, str):
                    # `dtype` is a plain string, it should be the `name` from a
                    # `DTypePolicy`
                    dtype = dtype_policies.get(dtype)
                    assertIsNone(dtype.quantization_mode)
                else:
                    # `dtype` is a DTypePolicy instance, it should be an
                    # instance of `QuantizedDTypePolicy`
                    assertIsNotNone(dtype.quantization_mode)
                super().__init__(*args, **kwargs)

        # Test floating dtype serialization
        layer = layers.Dense(2, dtype="bfloat16")
        config = layer.get_config()
        self.assertIn("dtype", config)
        self.assertEqual(
            config["dtype"],
            dtype_policies.serialize(dtype_policies.DTypePolicy("bfloat16")),
        )
        AssertionDense.from_config(config)  # Assertion inside

        # Test quantized dtype serialization
        layer = layers.Dense(2, dtype="int8_from_bfloat16")
        config = layer.get_config()
        self.assertIn("dtype", config)
        self.assertEqual(
            config["dtype"],
            dtype_policies.serialize(dtype_policies.get("int8_from_bfloat16")),
        )
        AssertionDense.from_config(config)  # Assertion inside

    def test_serialize_activity_regularizer(self):
        layer = layers.Dense(2, activity_regularizer="l2")
        config = layer.get_config()
        self.assertIn("activity_regularizer", config)
        new_layer = layers.Dense.from_config(config)
        self.assertEqual(
            new_layer.activity_regularizer.__class__.__name__, "L2"
        )

        layer = layers.Dense(2)
        config = layer.get_config()
        self.assertNotIn("activity_regularizer", config)

    def test_custom_layer_add_weight_in_init_name(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerLayer()

        class InnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )
                self.inner = InnerInnerLayer()

        class InnerInnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

        layer = TrainingLayer()
        layer.build(None)
        self.assertEqual(len(layer.variables), 2)
        variable_paths = set(v.path for v in layer.variables)
        self.assertTrue("inner_layer/inner" in variable_paths)
        self.assertTrue("inner_inner_layer/inner" in variable_paths)
        if backend.backend() == "torch":
            parameter_names = set(
                param_name.replace("_torch_params.", "")
                for param_name, _ in layer.named_parameters()
            )
            self.assertSetEqual(variable_paths, parameter_names)

    def test_custom_layer_add_weight_in_build_name(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerLayer()

            def call(self, input):
                return self.inner(input)

        class InnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerInnerLayer()

            def build(self, _):
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

            def call(self, input):
                return self.var + self.inner(input)

        class InnerInnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()

            def build(self, _):
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

            def call(self, input):
                return self.var + input

        layer = TrainingLayer()
        output = layer(
            backend.KerasTensor(
                (4, 1),
            )
        )
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(len(layer.variables), 2)
        variable_paths = set(v.path for v in layer.variables)
        self.assertTrue("training_layer/inner_layer/inner" in variable_paths)
        self.assertTrue(
            "training_layer/inner_layer/inner_inner_layer/inner"
            in variable_paths
        )
        if backend.backend() == "torch":
            parameter_names = set(
                param_name.replace("_torch_params.", "")
                for param_name, _ in layer.named_parameters()
            )
            self.assertSetEqual(variable_paths, parameter_names)

    def test_layer_variable_tracking_correct(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.post_build_modify_layer = PostBuildModifyLayer()

            def call(self, input):
                return self.post_build_modify_layer(input)

        class PostBuildModifyLayer(layers.Layer):
            def call(self, input):
                return self.var + input

            def build(self, _):
                self.var = self.add_weight(
                    shape=(2,),
                    name="var",
                )

            def post_build_add(self):
                self._tracker.unlock()
                self.additional_var = self.add_weight(
                    shape=(2,),
                    name="var2",
                )
                self._tracker.lock()

            def post_build_remove(self):
                self._tracker.unlock()
                self._untrack_variable(self.var)
                del self.var
                self._tracker.lock()

        layer = TrainingLayer()
        output = layer(backend.KerasTensor((4, 2)))

        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(len(layer.variables), 1)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var",
        )
        if backend.backend() == "torch":
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )

        layer.post_build_modify_layer.post_build_add()
        self.assertEqual(len(layer.variables), 2)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var",
        )
        self.assertEqual(
            layer.variables[1].path,
            "training_layer/post_build_modify_layer/var2",
        )
        if backend.backend() == "torch":
            # TODO (haohuanw, fchollet): Needs further discussion on how to
            # properly manage torch params. Post build modification cannot
            # propagate to parent torch params.
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            # Below check should have 2 parameters instead of 1.
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 2)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )
            self.assertEqual(
                parameter_names[1],
                "_torch_params.training_layer/post_build_modify_layer/var2",
            )

        layer.post_build_modify_layer.post_build_remove()
        self.assertEqual(len(layer.variables), 1)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var2",
        )
        if backend.backend() == "torch":
            # TODO (haohuanw, fchollet): Needs further discussion on how to
            # properly manage torch params. Post build modification cannot
            # propagate to parent torch params.
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            # Below check should have 1 parameters instead of 2, torch_params
            # in parent layer is wrong.
            self.assertEqual(len(parameter_names), 2)
            self.assertEqual(
                parameter_names[0],
                "post_build_modify_layer._torch_params.training_layer/"
                "post_build_modify_layer/var2",
            )
            self.assertEqual(
                parameter_names[1],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var2",
            )

    @pytest.mark.skipif(backend.backend() != "torch", reason="Torch only test.")
    def test_torch_params_create_deterministic(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight()
                self.w2 = self.add_weight(dtype="int32", trainable=False)
                self.w3 = self.add_weight(dtype="bool", trainable=False)
                self.w4 = self.add_weight(
                    dtype="int32", shape=(2, 2), trainable=False
                )
                self.w5 = self.add_weight(initializer="ones", shape=(2, 2))

        layer1 = MyLayer()
        layer1.build(None)
        layer1_names = list(pname for pname, _ in layer1.named_parameters())
        global_state.clear_session()
        layer2 = MyLayer()
        layer2.build(None)
        layer2_names = list(pname for pname, _ in layer2.named_parameters())
        self.assertListEqual(layer1_names, layer2_names)

    @pytest.mark.skipif(
        not backend.SUPPORTS_COMPLEX_DTYPES,
        reason=f"{backend.backend()} backend doesn't support complex dtypes.",
    )
    def test_complex_dtype_support(self):
        class MyDenseLayer(layers.Layer):
            def __init__(self, num_outputs):
                super(MyDenseLayer, self).__init__()
                self.num_outputs = num_outputs

            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=[int(input_shape[-1]), self.num_outputs],
                )

            def call(self, inputs):
                kernel = ops.cast(self.kernel, "complex64")
                return ops.matmul(inputs, kernel)

        inputs = ops.zeros([10, 5], dtype="complex64")
        layer = MyDenseLayer(10)
        output = layer(inputs)
        self.assertEqual(output.shape, (10, 10))

    def test_call_context_args_with_custom_layers(self):
        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")
                self.inner = Inner()

            def call(self, x):
                # Outer doesn’t even need to re‑inject explicitly:
                # our base class will propagate foo_mode automatically
                return self.inner(x)

        layer = Outer()
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 1)
        self.assertEqual(int(layer(np.array(0))), 0)

    def test_register_call_context_arguments(self):
        """Validate that registering call-context args works as expected."""

        class MyLayer(layers.Layer):
            def call(self, x):
                return x

        layer = MyLayer()

        layer._register_call_context_args("foo_mode")

        self.assertCountEqual(
            layer._call_context_args, ("foo_mode", "training")
        )

    def test_register_call_context_arguments_after_call(self):
        """Validate that registering call-context args after the layer has
        been called raises an error."""

        class MyLayer(layers.Layer):
            def call(self, x):
                return x

        layer = MyLayer()
        layer(np.array(0))
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot add call-context args after the layer has been called.",
        ):
            layer._register_call_context_args("foo_mode")

    def test_context_args_with_triple_nesting_and_priority(self):
        """Validate that call-context args are propagated correctly
        through multiple layers, and that the most specific value is used
        when multiple values are passed down the call-stack.
        """

        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Middle(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                return self.inner(x)

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.middle = Middle()

            def call(self, x):
                # Outer explicitly sets foo_mode=False when calling Inner,
                # so the value being passed here should be ignored.
                return self.middle(x)

        layer = Outer()
        layer._register_call_context_args("foo_mode")

        # The value of foo_mode is set to True in the call to Outer,
        # so it should automatically propagate to Inner through Middle.
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 1)
        self.assertEqual(int(layer(np.array(0))), 0)

    def test_context_arg_propagation_without_declaration(self):
        """Validate that layer does not resolve a propagated arg if it is not
        declared as a call-context arg in the layer itself."""

        class Inner(layers.Layer):
            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Wrapper(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                return self.inner(x)

        layer = Wrapper()
        layer._register_call_context_args("foo_mode")

        # The value of foo_mode is set to True in the call to Wrapper,
        # However, it is not declared as a call-context arg in Inner,
        # so it should not resolve to True inside Inner (and instead
        # default to False).
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 0)

    def test_call_context_args_with_func_seq_models_as_layers(self):
        """Validate that call-context args are propagated correctly
        through functional and sequential models when used as layers.
        """

        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=False):
                # If foo_mode=True add 1, otherwise add 0
                add_val = ops.where(foo_mode, 1.0, 0.0)
                return x + add_val

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                # We don’t explicitly pass foo_mode here—Base Layer.__call__
                # should inject it into `self.inner`
                return self.inner(x)

        sample_input = np.array([[1.0], [2.0]])

        # Sequential model
        seq = models.Sequential([layers.Identity(), Outer()])
        # Tell the Sequential model to propagate foo_mode down
        # the call-stack
        seq._register_call_context_args("foo_mode")

        # foo_mode=True -> input + 1
        out_true = seq(sample_input, foo_mode=True)
        self.assertAllClose(out_true, sample_input + 1.0)

        # foo_mode omitted -> foo_mode defaults to False -> no change
        out_false = seq(sample_input)
        self.assertAllClose(out_false, sample_input)

        # Functional model
        inp = Input(shape=(1,))
        out = layers.Identity()(inp)
        out = Outer()(out)
        model = models.Model(inp, out)
        # Tell the Functional model to propagate foo_mode down
        # the call-stack
        model._register_call_context_args("foo_mode")

        # foo_mode=True -> input + 1
        y1 = model(sample_input, foo_mode=True)
        self.assertAllClose(y1, sample_input + 1.0)

        # foo_mode omitted -> foo_mode defaults to False -> no change
        y2 = model(sample_input)
        self.assertAllClose(y2, sample_input)

    def test_layer_build_with_attention_mask_arg(self):
        test = self

        class CustomLayer(layers.Layer):
            def call(self, inputs, attention_mask=None):
                if attention_mask is not None:
                    return inputs * ops.cast(attention_mask, x.dtype)
                return inputs

            def build(self, inputs_shape, attention_mask_shape=None):
                test.assertIsNotNone(attention_mask_shape)
                self.built = True

        layer = CustomLayer()
        x = np.ones((2, 3), dtype="float32")
        mask = np.ones((2, 1), dtype="float32")
        y = layer(x, attention_mask=mask)
        self.assertEqual(y.shape, (2, 3))

    def test_name_scope_opened_once_per_call_when_built(self):
        # `_maybe_build` opens the name scope itself, after its `built` check,
        # so a layer that is already built opens it once per call (for the
        # call itself) rather than twice.
        opens = []

        class CountingLayer(layers.Layer):
            def _open_name_scope(self):
                opens.append(1)
                return super()._open_name_scope()

            def call(self, x):
                return x

        layer = CountingLayer()
        x = np.ones((2, 4), dtype="float32")
        layer(x)

        opens.clear()
        layer(x)
        layer(x)
        self.assertEqual(len(opens), 2)

    def test_called_and_built_flags_set_once(self):
        # Verify that built and _called are True after the first call and
        # remain True on repeated calls, that build() is invoked exactly
        # once regardless of how many times the layer is called, and that
        # `built`/`_called` are not reassigned once already True (the
        # optimization this test guards against regressing).
        build_count = []

        class CountingLayer(layers.Layer):
            def __init__(self, *args, **kwargs):
                # Set up before calling `super().__init__()` since that
                # call already triggers `__setattr__` for `built` and
                # `_called`.
                object.__setattr__(self, "setattr_calls", [])
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value):
                if name in ("built", "_called"):
                    self.setattr_calls.append((name, value))
                super().__setattr__(name, value)

            def build(self, input_shape):
                build_count.append(1)
                self.built = True

            def call(self, x):
                return x

        layer = CountingLayer()
        x = np.ones((2, 4), dtype="float32")
        layer(x)

        self.assertTrue(layer.built)
        self.assertTrue(layer._called)
        self.assertEqual(len(build_count), 1)

        # Reset the recorded setattr traffic and call the already
        # built/called layer a couple more times: `built` and `_called`
        # must not be reassigned since they are already True.
        layer.setattr_calls.clear()
        layer(x)
        layer(x)

        self.assertTrue(layer.built)
        self.assertTrue(layer._called)
        self.assertEqual(len(build_count), 1)
        self.assertEqual(len(layer.setattr_calls), 0)

    @parameterized.named_parameters(
        ("true", True), ("false", False), ("none", None)
    )
    def test_symbolic_call_records_training_kwarg_not_taken_by_call(
        self, training
    ):
        # `CallSpec` pops call-context args that `call()` does not accept out
        # of the `kwargs` dict it is handed. That dict must not be the one
        # `Node` records, or the recorded call loses `training` and replaying
        # the model runs the wrapped layer in the wrong mode.
        class Wrapper(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.bn = layers.BatchNormalization(momentum=0.0)

            def call(self, x):  # Deliberately does not accept `training`.
                return self.bn(x)

        inputs = Input(shape=(4,), batch_size=4)
        layer = Wrapper()
        model = Model(inputs, layer(inputs, training=training))

        self.assertEqual(
            dict(model.layers[1]._inbound_nodes[0].arguments.kwargs),
            {"training": training},
        )

        x = np.arange(16, dtype="float32").reshape(4, 4)
        y = ops.convert_to_numpy(model(x))
        if training is True:
            # `momentum=0.0` snaps the running stats to the batch seen in
            # training mode, so replaying normalizes with them: column `j`
            # of `x` is `[0, 4, 8, 12] + j`, i.e. mean `6 + j`, variance 20.
            column_means = 6.0 + np.arange(4, dtype="float32")
            self.assertAllClose(
                y, (x - column_means) / np.sqrt(20.0 + 1e-3), atol=1e-3
            )
        else:
            # `training=False`/`None` never update the running stats, so
            # they are still mean 0 / variance 1.
            self.assertAllClose(y, x / np.sqrt(1.0 + 1e-3), atol=1e-3)


class _LoraLessProjection(layers.Layer):
    """A third-party projection that opts into quantization without LoRA."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = None

    def build(self, input_shape):
        self.kernel_shape = (input_shape[-1], self.units)
        if self.quantization_mode:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if not self._strategy_owns_weight_storage():
            self._kernel = self.add_weight(
                name="kernel", shape=self.kernel_shape
            )
        self.bias = None

    def call(self, inputs):
        return ops.matmul(inputs, self._kernel)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        return {
            None: ["kernel"],
            "int8": ["kernel", "kernel_scale"],
            "int4": ["kernel", "kernel_scale", "kernel_zero", "g_idx"],
            "float8": [
                "kernel",
                "inputs_scale",
                "inputs_amax_history",
                "kernel_scale",
                "kernel_amax_history",
                "outputs_grad_scale",
                "outputs_grad_amax_history",
            ],
        }

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")


class LoraLessQuantizationTest(testing.TestCase):
    def test_layer_does_not_enable_lora_by_default(self):
        self.assertFalse(layers.Layer().lora_enabled)

    @parameterized.named_parameters(
        ("int8", "int8", None),
        ("int4_per_channel", "int4", -1),
        ("int4_grouped", "int4", 4),
    )
    def test_quantize_call_and_round_trip(self, mode, block_size):
        config = None
        if mode == "int4":
            config = Int4QuantizationConfig(block_size=block_size)
        inputs = Input((8,))
        layer = _LoraLessProjection(6)
        model = Model(inputs, layer(inputs))
        x = np.random.default_rng(0).standard_normal((3, 8)).astype("float32")
        model.quantize(mode, config=config)
        self.assertEqual(layer.quantization_mode, mode)
        y = layer(x)

        store = {}
        layer.save_own_variables(store)
        restored = _LoraLessProjection(6, dtype=layer.dtype_policy.name)
        restored.build((None, 8))
        restored.load_own_variables(store)
        self.assertAllClose(restored(x), y)

    def test_float8_quantize_and_call(self):
        layer = _LoraLessProjection(6)
        layer.build((None, 8))
        layer.quantize("float8")
        layer(np.ones((3, 8), dtype="float32"))


class _DenseListingExtra(layers.Dense):
    """A `Dense` whose spec lists a name that the layer may not define."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        return {mode: names + ["extra"] for mode, names in spec.items()}


class SpecNameWithoutAttributeTest(testing.TestCase):
    def _layer(self, extra_is_none=False):
        layer = _DenseListingExtra(4)
        layer.build((None, 3))
        if extra_is_none:
            layer.extra = None
        return layer

    def test_save_raises_for_a_spec_name_without_attribute(self):
        layer = self._layer()
        with self.assertRaisesRegex(
            AttributeError, "lists 'extra' in its `variable_serialization_spec`"
        ):
            layer.save_own_variables({})

    def test_load_raises_for_a_spec_name_without_attribute(self):
        layer = self._layer()
        store = {"0": np.ones((3, 4), "float32"), "1": np.ones((4,), "float32")}
        with self.assertRaisesRegex(
            AttributeError, "lists 'extra' in its `variable_serialization_spec`"
        ):
            layer.load_own_variables(store)

    def test_spec_name_holding_none_is_skipped(self):
        layer = self._layer(extra_is_none=True)
        store = {}
        layer.save_own_variables(store)
        self.assertEqual(sorted(store), ["0", "1"])

        restored = self._layer(extra_is_none=True)
        restored.load_own_variables(store)
        self.assertAllClose(restored.kernel, layer.kernel)
        self.assertAllClose(restored.bias, layer.bias)


def _int4_setter_case(kind, dtype=None):
    if kind == "dense":
        layer = layers.Dense(6, dtype=dtype)
        layer.build((None, 256))
        x = np.random.default_rng(0).standard_normal((2, 256))
    elif kind == "einsum":
        layer = layers.EinsumDense(
            "ab,bcd->acd", output_shape=(8, 32), bias_axes="d", dtype=dtype
        )
        layer.build((None, 256))
        x = np.random.default_rng(0).standard_normal((2, 256))
    elif kind == "embedding":
        layer = layers.Embedding(300, 16, dtype=dtype)
        layer.build()
        x = np.array([[1, 2, 299]])
    else:
        layer = layers.ReversibleEmbedding(300, 16, dtype=dtype)
        layer.build()
        x = np.array([[1, 2, 299]])
    return layer, x


class LegacyInt4PolicySetterTest(testing.TestCase):
    @parameterized.named_parameters(
        ("dense", "dense"),
        ("einsum", "einsum"),
        ("embedding", "embedding"),
        ("reversible_embedding", "reversible_embedding"),
    )
    def test_bare_int4_policy_builds_what_the_name_builds(self, kind):
        # The legacy bare "int4" name holds no block size, and a layer built
        # from it is per-channel. Setting it on a built layer quantizes
        # per-channel too, so a layer built from the name alone (a
        # `DTypePolicyMap` reload) loads the store.
        layer, x = _int4_setter_case(kind)
        layer.dtype_policy = "int4_from_float32"
        self.assertEqual(layer.dtype_policy.name, "int4_from_float32")
        store = {}
        layer.save_own_variables(store)

        restored, _ = _int4_setter_case(kind, dtype="int4_from_float32")
        restored.load_own_variables(store)
        self.assertAllEqual(restored(x), layer(x))
