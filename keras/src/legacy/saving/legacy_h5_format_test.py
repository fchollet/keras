import os

import numpy as np
import pytest

import keras
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.legacy.saving import legacy_h5_format
from keras.src.saving import object_registration
from keras.src.saving import serialization_lib

# TODO: more thorough testing. Correctness depends
# on exact weight ordering for each layer, so we need
# to test across all types of layers.

try:
    import tf_keras
except:
    tf_keras = None


def get_sequential_model(keras):
    return keras.Sequential(
        [
            keras.layers.Input((3,), batch_size=2),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.BatchNormalization(
                moving_mean_initializer="uniform", gamma_initializer="uniform"
            ),
            keras.layers.Dense(5, activation="softmax"),
        ]
    )


def get_functional_model(keras):
    inputs = keras.Input((3,), batch_size=2)
    x = keras.layers.Dense(4, activation="relu")(inputs)
    residual = x
    x = keras.layers.BatchNormalization(
        moving_mean_initializer="uniform", gamma_initializer="uniform"
    )(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.add([x, residual])
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_subclassed_model(keras):
    class MyModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(3, activation="relu")
            self.dense_2 = keras.layers.Dense(1, activation="sigmoid")

            # top_level_model_weights
            self.bias = self.add_weight(
                name="bias",
                shape=[1],
                trainable=True,
                initializer=keras.initializers.Zeros(),
            )

        def call(self, x):
            x = self.dense_1(x)
            x = self.dense_2(x)

            # top_level_model_weights
            x += ops.cast(self.bias, x.dtype)
            return x

    model = MyModel()
    model(np.random.random((2, 3)))
    return model


@pytest.mark.requires_trainable_backend
@pytest.mark.skipif(tf_keras is None, reason="Test requires tf_keras")
class LegacyH5WeightsTest(testing.TestCase):
    def _check_reloading_weights(self, ref_input, model, tf_keras_model):
        ref_output = tf_keras_model(ref_input)
        initial_weights = model.get_weights()
        # Check weights only file
        temp_filepath = os.path.join(self.get_temp_dir(), "weights.h5")
        tf_keras_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(output, ref_output, atol=1e-5)
        model.set_weights(initial_weights)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(output, ref_output, atol=1e-5)

    def test_sequential_model_weights(self):
        model = get_sequential_model(keras)
        tf_keras_model = get_sequential_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_functional_model_weights(self):
        model = get_functional_model(keras)
        tf_keras_model = get_functional_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_subclassed_model_weights(self):
        model = get_subclassed_model(keras)
        tf_keras_model = get_subclassed_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)


@pytest.mark.requires_trainable_backend
class LegacyH5WholeModelTest(testing.TestCase):
    def _check_reloading_model(self, ref_input, model):
        # Whole model file
        ref_output = model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(output, ref_output, atol=1e-5)

    def test_sequential_model(self):
        model = get_sequential_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_functional_model(self):
        model = get_functional_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_compiled_model_with_various_layers(self):
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))

        model.compile(optimizer="rmsprop", loss="mean_squared_error")
        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model)

    def test_saving_preserves_compile_config(self):
        model = models.Sequential([layers.Input((3,)), layers.Dense(2)])
        model.compile(optimizer="adam", loss="mean_squared_error")
        compile_config = model.get_compile_config()

        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)

        self.assertEqual(model.get_compile_config(), compile_config)

    def test_saving_lambda(self):
        mean = random.uniform((4, 2, 3))
        std = ops.abs(random.uniform((4, 2, 3))) + 1e-5
        inputs = layers.Input(shape=(4, 2, 3))
        output = layers.Lambda(
            lambda image, mu, std: (image - mu) / std,
            arguments={"mu": mean, "std": std},
        )(inputs)
        model = models.Model(inputs, output)
        model.compile(
            loss="mean_squared_error", optimizer="sgd", metrics=["acc"]
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "lambda_model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)

        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            legacy_h5_format.load_model_from_hdf5(temp_filepath)

        # `safe_mode=None` reaches the scope unchanged; only an explicit
        # `False` may opt out.
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            legacy_h5_format.load_model_from_hdf5(temp_filepath, safe_mode=None)

        loaded = legacy_h5_format.load_model_from_hdf5(
            temp_filepath, safe_mode=False
        )
        self.assertAllClose(mean, loaded.layers[1].arguments["mu"])
        self.assertAllClose(std, loaded.layers[1].arguments["std"])

    def test_safe_mode_none_opens_safe_scope(self):
        seen = []

        @object_registration.register_keras_serializable(package="my_package")
        class ScopeRecorder(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                seen.append(serialization_lib.in_safe_mode())

            def call(self, inputs):
                return inputs

        model = models.Sequential([layers.Input((3,)), ScopeRecorder()])
        temp_filepath = os.path.join(self.get_temp_dir(), "recorder.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)

        # A `Sequential` model opens no scope of its own, so the layer sees
        # the loader's scope. `safe_mode=None` must not act as an opt-out.
        seen.clear()
        legacy_h5_format.load_model_from_hdf5(temp_filepath, safe_mode=None)
        self.assertEqual(seen, [True])

    def test_saving_include_optimizer_false(self):
        model = models.Sequential()
        model.add(layers.Dense(1))
        model.compile("adam", loss="mean_squared_error")
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model.fit(x, y)
        ref_output = model(x)

        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(
            model, temp_filepath, include_optimizer=False
        )
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(x)

        # Assert that optimizer does not exist in loaded model
        with self.assertRaises(AttributeError):
            _ = loaded.optimizer

        # Compare output
        self.assertAllClose(output, ref_output, atol=1e-5)

    def test_custom_sequential_registered_no_scope(self):
        @object_registration.register_keras_serializable(package="my_package")
        class MyDense(layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = layers.Input(shape=[1])
        custom_layer = MyDense(1)
        model = models.Sequential(layers=[inputs, custom_layer])

        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model)

    def test_custom_functional_registered_no_scope(self):
        @object_registration.register_keras_serializable(package="my_package")
        class MyDense(layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        model = models.Model(inputs, outputs)

        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model)

    def test_custom_function_from_custom_objects_no_registration(self):
        from keras.src.saving import custom_object_scope

        def custom_fn(x):
            return x * 2.0

        class MyDense(layers.Dense):
            def call(self, inputs):
                return custom_fn(super().call(inputs))

        inputs = layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        model = models.Model(inputs, outputs)

        ref_input = np.array([[5.0]])
        with custom_object_scope({"MyDense": MyDense, "custom_fn": custom_fn}):
            self._check_reloading_model(ref_input, model)

    def test_nested_layers(self):
        class MyLayer(layers.Layer):
            def __init__(self, sublayers, **kwargs):
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                config = super().get_config()
                config["sublayers"] = serialization_lib.serialize_keras_object(
                    self.sublayers
                )
                return config

            @classmethod
            def from_config(cls, config):
                config["sublayers"] = (
                    serialization_lib.deserialize_keras_object(
                        config["sublayers"]
                    )
                )
                return cls(**config)

        @object_registration.register_keras_serializable(package="Foo")
        class RegisteredSubLayer(layers.Layer):
            pass

        layer = MyLayer(
            [
                layers.Dense(2, name="MyDense"),
                RegisteredSubLayer(name="MySubLayer"),
            ]
        )
        model = models.Sequential([layer])
        with self.subTest("test_JSON"):
            from keras.src.models.model import model_from_json

            model_json = model.to_json()
            self.assertIn("Foo>RegisteredSubLayer", model_json)

            loaded_model = model_from_json(
                model_json, custom_objects={"MyLayer": MyLayer}
            )
            loaded_layer = loaded_model.layers[0]

            self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
            self.assertEqual(loaded_layer.sublayers[0].name, "MyDense")
            self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
            self.assertEqual(loaded_layer.sublayers[1].name, "MySubLayer")

        with self.subTest("test_H5"):
            temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
            legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
            loaded_model = legacy_h5_format.load_model_from_hdf5(
                temp_filepath, custom_objects={"MyLayer": MyLayer}
            )
            loaded_layer = loaded_model.layers[0]

            self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
            self.assertEqual(loaded_layer.sublayers[0].name, "MyDense")
            self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
            self.assertEqual(loaded_layer.sublayers[1].name, "MySubLayer")

    def test_model_loading_with_axis_arg(self):
        input1 = layers.Input(shape=(1, 4), name="input1")
        input2 = layers.Input(shape=(1, 4), name="input2")
        concat1 = layers.Concatenate(axis=1)([input1, input2])
        output = layers.Dense(1, activation="sigmoid")(concat1)
        model = models.Model(inputs=[input1, input2], outputs=output)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        temp_filepath = os.path.join(
            self.get_temp_dir(), "model_with_axis_arg.h5"
        )
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        legacy_h5_format.load_model_from_hdf5(temp_filepath)


@pytest.mark.requires_trainable_backend
@pytest.mark.skipif(tf_keras is None, reason="Test requires tf_keras")
class LegacyH5BackwardsCompatTest(testing.TestCase):
    def _check_reloading_model(self, ref_input, model, tf_keras_model):
        # Whole model file
        ref_output = tf_keras_model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        tf_keras_model.save(temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(output, ref_output, atol=1e-5)

    def test_sequential_model(self):
        model = get_sequential_model(keras)
        tf_keras_model = get_sequential_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_functional_model(self):
        tf_keras_model = get_functional_model(tf_keras)
        model = get_functional_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_compiled_model_with_various_layers(self):
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))
        model.compile(optimizer="rmsprop", loss="mse")

        tf_keras_model = tf_keras.Sequential()
        tf_keras_model.add(tf_keras.layers.Dense(2, input_shape=(3,)))
        tf_keras_model.add(tf_keras.layers.RepeatVector(3))
        tf_keras_model.add(
            tf_keras.layers.TimeDistributed(tf_keras.layers.Dense(3))
        )
        tf_keras_model.compile(optimizer="rmsprop", loss="mean_squared_error")

        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_saving_lambda(self):
        mean = np.random.random((4, 2, 3))
        std = np.abs(np.random.random((4, 2, 3))) + 1e-5
        inputs = tf_keras.layers.Input(shape=(4, 2, 3))
        output = tf_keras.layers.Lambda(
            lambda image, mu, std: (image - mu) / std,
            arguments={"mu": mean, "std": std},
            output_shape=inputs.shape,
        )(inputs)
        tf_keras_model = tf_keras.Model(inputs, output)
        tf_keras_model.compile(
            loss="mean_squared_error", optimizer="sgd", metrics=["acc"]
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "lambda_model.h5")
        tf_keras_model.save(temp_filepath)

        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            legacy_h5_format.load_model_from_hdf5(temp_filepath)

        loaded = legacy_h5_format.load_model_from_hdf5(
            temp_filepath, safe_mode=False
        )
        self.assertAllClose(mean, loaded.layers[1].arguments["mu"])
        self.assertAllClose(std, loaded.layers[1].arguments["std"])

    def test_saving_include_optimizer_false(self):
        tf_keras_model = tf_keras.Sequential()
        tf_keras_model.add(tf_keras.layers.Dense(1))
        tf_keras_model.compile("adam", loss="mse")
        x, y = np.ones((10, 10)), np.ones((10, 1))
        tf_keras_model.fit(x, y)
        ref_output = tf_keras_model(x)

        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        tf_keras_model.save(temp_filepath, include_optimizer=False)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(x)

        # Assert that optimizer does not exist in loaded model
        with self.assertRaises(AttributeError):
            _ = loaded.optimizer

        # Compare output
        self.assertAllClose(output, ref_output, atol=1e-5)

    @pytest.mark.skipif(tf_keras is None, reason="Test requires tf_keras")
    def test_custom_sequential_registered_no_scope(self):
        @tf_keras.saving.register_keras_serializable(package="my_package")
        class MyDense(tf_keras.layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = tf_keras.layers.Input(shape=[1])
        custom_layer = MyDense(1)
        tf_keras_model = tf_keras.Sequential(layers=[inputs, custom_layer])

        # Re-implement and re-register in Keras 3
        @object_registration.register_keras_serializable(package="my_package")
        class MyDense(layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = layers.Input(shape=[1])
        custom_layer = MyDense(1)
        model = models.Sequential(layers=[inputs, custom_layer])

        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model, tf_keras_model)

    @pytest.mark.skipif(tf_keras is None, reason="Test requires tf_keras")
    def test_custom_functional_registered_no_scope(self):
        @tf_keras.saving.register_keras_serializable(package="my_package")
        class MyDense(tf_keras.layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = tf_keras.layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        tf_keras_model = tf_keras.Model(inputs, outputs)

        # Re-implement and re-register in Keras 3
        @object_registration.register_keras_serializable(package="my_package")
        class MyDense(layers.Dense):
            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)

        inputs = layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        model = models.Model(inputs, outputs)

        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_nested_layers(self):
        class MyLayer(tf_keras.layers.Layer):
            def __init__(self, sublayers, **kwargs):
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                config = super().get_config()
                config["sublayers"] = tf_keras.saving.serialize_keras_object(
                    self.sublayers
                )
                return config

            @classmethod
            def from_config(cls, config):
                config["sublayers"] = tf_keras.saving.deserialize_keras_object(
                    config["sublayers"]
                )
                return cls(**config)

        @tf_keras.saving.register_keras_serializable(package="Foo")
        class RegisteredSubLayer(layers.Layer):
            def call(self, x):
                return x

        layer = MyLayer(
            [
                tf_keras.layers.Dense(2, name="MyDense"),
                RegisteredSubLayer(name="MySubLayer"),
            ]
        )
        tf_keras_model = tf_keras.Sequential([layer])

        x = np.random.random((4, 2))
        ref_output = tf_keras_model(x)

        # Save TF Keras model to H5 file
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        tf_keras_model.save(temp_filepath)

        # Re-implement in Keras 3
        class MyLayer(layers.Layer):
            def __init__(self, sublayers, **kwargs):
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                config = super().get_config()
                config["sublayers"] = serialization_lib.serialize_keras_object(
                    self.sublayers
                )
                return config

            @classmethod
            def from_config(cls, config):
                config["sublayers"] = (
                    serialization_lib.deserialize_keras_object(
                        config["sublayers"]
                    )
                )
                return cls(**config)

        # Re-implement and re-register in Keras 3
        @object_registration.register_keras_serializable(package="Foo")
        class RegisteredSubLayer(layers.Layer):
            def call(self, x):
                return x

        # Load in Keras 3
        loaded_model = legacy_h5_format.load_model_from_hdf5(
            temp_filepath, custom_objects={"MyLayer": MyLayer}
        )
        loaded_layer = loaded_model.layers[0]
        output = loaded_model(x)

        # Ensure nested layer structure
        self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
        self.assertEqual(loaded_layer.sublayers[0].name, "MyDense")
        self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
        self.assertEqual(loaded_layer.sublayers[1].name, "MySubLayer")

        # Compare output
        self.assertAllClose(output, ref_output, atol=1e-5)


@pytest.mark.requires_trainable_backend
class DirectoryCreationTest(testing.TestCase):
    def test_directory_creation_on_save(self):
        """Test if directory is created on model save."""
        model = get_sequential_model(keras)
        nested_dirpath = os.path.join(
            self.get_temp_dir(), "dir1", "dir2", "dir3"
        )
        filepath = os.path.join(nested_dirpath, "model.h5")
        self.assertFalse(os.path.exists(nested_dirpath))
        legacy_h5_format.save_model_to_hdf5(model, filepath)
        self.assertTrue(os.path.exists(nested_dirpath))
        loaded_model = legacy_h5_format.load_model_from_hdf5(filepath)
        self.assertEqual(model.to_json(), loaded_model.to_json())
