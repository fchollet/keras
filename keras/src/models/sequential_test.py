import pickle

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import saving
from keras.src import testing
from keras.src.layers.core.input_layer import Input
from keras.src.models.functional import Functional
from keras.src.models.model import Model
from keras.src.models.sequential import Sequential
from keras.src.testing.test_utils import named_product


class SequentialTest(testing.TestCase):
    def test_basic_flow_with_input(self):
        model = Sequential(name="seq")
        model.add(Input(shape=(2,), batch_size=3))
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        model.summary()

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)

        # Test eager call
        x = np.random.random((3, 2))
        y = model(x)

        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(y.shape, (3, 5))

        # Test symbolic call
        x = backend.KerasTensor((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test `layers` constructor arg
        model = Sequential(
            layers=[
                Input(shape=(2,), batch_size=3),
                layers.Dense(4),
                layers.Dense(5),
            ]
        )
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test pop
        model.pop()
        self.assertEqual(len(model.layers), 1)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 2)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 4))

    def test_legacy_flow_with_input_shape(self):
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_shape=(2,)))
        model.add(layers.Dense(5))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

        # Input_dim works too
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_dim=2))
        model.add(layers.Dense(5))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

        # Subsequent input_shapes are ignored
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_shape=(2,)))
        model.add(layers.Dense(5, input_shape=(3, 4)))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

    def test_basic_flow_deferred(self):
        model = Sequential(name="seq")
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        model.summary()

        self.assertEqual(len(model.layers), 2)

        # Test eager call
        x = np.random.random((3, 2))
        y = model(x)
        self.assertTrue(model.built)
        model.summary()

        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(y.shape, (3, 5))

        # Test symbolic call
        x = backend.KerasTensor((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test `layers` constructor arg
        model = Sequential(
            layers=[
                layers.Dense(4),
                layers.Dense(5),
            ]
        )
        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test pop
        model.pop()
        self.assertEqual(len(model.layers), 1)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 2)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 4))

    def test_basic_flow_as_a_submodel(self):
        # Build submodel
        submodel = Sequential()
        submodel.add(layers.Flatten())
        self.assertFalse(submodel.built)

        inputs = Input((None, 4))
        outputs = layers.TimeDistributed(submodel)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        x = np.random.random((2, 3, 4))
        y = model(x)
        self.assertEqual(y.shape, (2, 3, 4))

    def test_basic_flow_with_functional_model_as_first_layer(self):
        # Build functional model
        inputs = Input((16, 16, 3))
        outputs = layers.Conv2D(4, 3, padding="same")(inputs)
        functional_model = Model(inputs=inputs, outputs=outputs)

        model = Sequential(
            [functional_model, layers.Flatten(), layers.Dense(1)]
        )
        model.summary()
        self.assertEqual(len(model.layers), 3)
        self.assertTrue(model.built)
        for layer in model.layers:
            self.assertTrue(layer.built)

        # Test eager call
        x = np.random.random((1, 16, 16, 3))
        y = model(x)
        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(tuple(y.shape), (1, 1))

        # Test symbolic call
        x = backend.KerasTensor((1, 16, 16, 3))
        y = model(x)
        self.assertEqual(y.shape, (1, 1))

    def test_basic_flow_with_sequential_model_as_first_layer(self):
        # Build sequential model
        sequential_model = Sequential(
            [Input((16, 16, 3)), layers.Conv2D(4, 3, padding="same")]
        )

        model = Sequential(
            [sequential_model, layers.Flatten(), layers.Dense(1)]
        )
        model.summary()
        self.assertEqual(len(model.layers), 3)
        self.assertTrue(model.built)
        for layer in model.layers:
            self.assertTrue(layer.built)

        # Test eager call
        x = np.random.random((1, 16, 16, 3))
        y = model(x)
        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(tuple(y.shape), (1, 1))

        # Test symbolic call
        x = backend.KerasTensor((1, 16, 16, 3))
        y = model(x)
        self.assertEqual(y.shape, (1, 1))

    def test_dict_inputs(self):
        test_obj = self

        class DictLayer(layers.Layer):
            def call(self, inputs):
                test_obj.assertIsInstance(inputs, dict)
                return inputs

        model = Sequential([DictLayer()])
        x = {"a": np.random.random((3, 2)), "b": np.random.random((3, 2))}
        y = model(x)
        self.assertEqual(type(y), dict)
        model.summary()

    def test_list_inputs(self):
        test_obj = self

        class ListLayer(layers.Layer):
            def call(self, inputs):
                test_obj.assertIsInstance(inputs, list)
                return inputs

        model = Sequential([ListLayer()])
        x = [np.random.random((3, 2)), np.random.random((3, 2))]
        y = model(x)
        self.assertEqual(type(y), list)
        model.summary()

    def test_nested_sequential(self):
        # https://github.com/keras-team/keras/issues/20203
        model = Sequential()
        model.add(Input(shape=(16,)))
        Sequential([model])

    def test_errors(self):
        # Trying to pass 2 Inputs
        model = Sequential()
        model.add(Input(shape=(2,), batch_size=3))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.add(Input(shape=(2,), batch_size=3))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.add(layers.InputLayer(shape=(2,), batch_size=3))

        # Same name 2x
        model = Sequential()
        model.add(layers.Dense(2, name="dense"))
        with self.assertRaisesRegex(ValueError, "should have unique names"):
            model.add(layers.Dense(2, name="dense"))

        # No layers
        model = Sequential()
        x = np.random.random((3, 2))
        with self.assertRaisesRegex(ValueError, "no layers"):
            model(x)

        # Build conflict
        model = Sequential()
        model.add(Input(shape=(2,), batch_size=3))
        model.add(layers.Dense(2))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.build((3, 4))
        # But this works
        model.build((3, 2))

    def test_shape_inference_failure(self):
        class DynamicLayer(layers.Layer):
            def call(self, inputs):
                return inputs + 1.0

            def compute_output_spec(self, *args, **kwargs):
                raise NotImplementedError

        model = Sequential([DynamicLayer()])
        x = np.random.random((3, 2))
        y = model(x)
        self.assertAllClose(y, x + 1)
        model.summary()

    def test_serialization(self):
        test_obj = self

        # Using add
        model = Sequential(name="seq")
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 2)

        # Using list
        model = Sequential(
            [layers.Dense(4), layers.Dense(5), layers.Dense(6)], name="seq"
        )
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 3)

        # Weird
        class DictLayer(layers.Layer):
            def call(self, inputs):
                test_obj.assertIsInstance(inputs, dict)
                return inputs

        model = Sequential([DictLayer()])
        revived = self.run_class_serialization_test(
            model, custom_objects={"DictLayer": DictLayer}
        )
        self.assertLen(revived.layers, 1)

    @parameterized.named_parameters(named_product(input_layer=[True, False]))
    def test_from_config_keeps_variable_paths(self, input_layer):
        # Regression test for #23251: deserializing builds the layers, and
        # their variables must get the same `<model>/<layer>/...` paths as in
        # a model built from scratch, so state trees can be restored.
        model_layers = [layers.Dense(4), layers.Dense(5)]
        if input_layer:
            model_layers = [Input(shape=(3,))] + model_layers
        model = Sequential(model_layers)
        model.build((None, 3))
        model.compile(optimizer="sgd", loss="mse")
        config = saving.serialize_keras_object(model)
        revived = saving.deserialize_keras_object(config)
        self.assertEqual(
            [v.path for v in revived.weights],
            [v.path for v in model.weights],
        )
        state = model.get_state_tree()
        revived.set_state_tree(
            {"trainable_variables": state["trainable_variables"]}
        )
        for orig, new in zip(model.get_weights(), revived.get_weights()):
            self.assertAllClose(orig, new)

    def test_serialization_with_lambda_layer(self):
        # https://github.com/keras-team/keras/issues/20074
        inputs = np.random.random(size=(1, 10, 4)).astype("float32")
        CONV_WIDTH = 3
        model = Sequential([layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :])])
        outputs = model(inputs)

        temp = self.get_temp_dir()
        save_path = f"{temp}/model.keras"
        model.save(save_path)
        revived = saving.load_model(save_path, safe_mode=False)
        revived_outputs = revived(inputs)
        self.assertLen(revived.layers, 1)
        self.assertAllClose(revived_outputs, outputs)

    def test_functional_properties(self):
        model = Sequential(name="seq")
        inputs = Input(shape=(2,))
        model.add(inputs)
        model.add(layers.Dense(4))

        self.assertEqual(model.inputs, [inputs])
        self.assertEqual(model.outputs, [model.layers[-1].output])
        self.assertEqual(model.input_shape, (None, 2))
        self.assertEqual(model.output_shape, (None, 4))

    def test_training_arg(self):
        class CustomLayer(layers.Layer):
            def call(self, inputs, training=False):
                return inputs * 0.0 if training else inputs

        model = Sequential([CustomLayer()])
        self.assertAllClose(model(np.ones((4, 20))), np.ones((4, 20)))
        self.assertAllClose(
            model(np.ones((4, 20)), training=False), np.ones((4, 20))
        )
        self.assertAllClose(
            model(np.ones((4, 20)), training=True), np.zeros((4, 20))
        )

    def test_pickleable(self):
        model = Sequential(name="seq")
        model.add(layers.Dense(4))

        result = pickle.loads(pickle.dumps(model))
        self.assertLen(result.layers, 1)

    def test_bad_layer(self):
        model = Sequential(name="seq")
        with self.assertRaisesRegex(ValueError, "Only instances of"):
            model.add({})

        model = Sequential(name="seq")

        class BadLayer(layers.Layer):
            def call(self, inputs, training):
                return inputs

        model.add(BadLayer())
        with self.assertRaisesRegex(
            ValueError, "can only have a single.*positional"
        ):
            model.build((None, 2))

    def test_compute_output_shape(self):
        layer = Sequential([layers.Dense(4), layers.Dense(8)])
        output_shape = layer.compute_output_shape((1, 2))
        self.assertEqual(output_shape, (1, 8))

    def test_hasattr(self):
        model = Sequential()
        self.assertFalse(hasattr(model, "input_shape"))
        self.assertFalse(hasattr(model, "output_shape"))
        self.assertFalse(hasattr(model, "inputs"))
        self.assertFalse(hasattr(model, "outputs"))

        model = Sequential([layers.Input((4,)), layers.Dense(8)])
        self.assertTrue(hasattr(model, "input_shape"))
        self.assertTrue(hasattr(model, "output_shape"))
        self.assertTrue(hasattr(model, "inputs"))
        self.assertTrue(hasattr(model, "outputs"))

    def test_layers_setter(self):
        model = Sequential()
        with self.assertRaisesRegex(
            AttributeError, r"Use `add\(\)` and `pop\(\)`"
        ):
            model.layers = [layers.Dense(4)]
