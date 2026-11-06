"""Tests for serialization_lib."""

import json

import numpy as np

import keras
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.saving import deserialize_keras_object
from keras.src.saving import object_registration
from keras.src.saving import serialization_lib


def custom_fn(x):
    return x**2


class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


class NestedCustomLayer(keras.layers.Layer):
    def __init__(self, factor, dense=None, activation=None):
        super().__init__()
        self.factor = factor

        if dense is None:
            self.dense = keras.layers.Dense(1, activation=custom_fn)
        else:
            self.dense = serialization_lib.deserialize_keras_object(dense)
        self.activation = serialization_lib.deserialize_keras_object(activation)

    def call(self, x):
        return self.dense(x * self.factor)

    def get_config(self):
        return {
            "factor": self.factor,
            "dense": self.dense,
            "activation": self.activation,
        }


class WrapperLayer(keras.layers.Wrapper):
    def call(self, x):
        return self.layer(x)


class SerializationLibTest(testing.TestCase):
    def roundtrip(self, obj, custom_objects=None, safe_mode=True):
        serialized = serialization_lib.serialize_keras_object(obj)
        json_data = json.dumps(serialized)
        json_data = json.loads(json_data)
        deserialized = serialization_lib.deserialize_keras_object(
            json_data, custom_objects=custom_objects, safe_mode=safe_mode
        )
        reserialized = serialization_lib.serialize_keras_object(deserialized)
        return serialized, deserialized, reserialized

    def test_simple_objects(self):
        for obj in [
            "hello",
            b"hello",
            np.array([0, 1]),
            np.array([0.0, 1.0]),
            np.float32(1.0),
            ["hello", 0, "world", 1.0, True],
            {"1": "hello", "2": 0, "3": True},
            {"1": "hello", "2": [True, False]},
            slice(None, 20, 1),
            slice(None, np.array([0, 1]), 1),
        ]:
            serialized, _, reserialized = self.roundtrip(obj)
            self.assertEqual(serialized, reserialized)

    def test_builtin_layers(self):
        layer = keras.layers.Dense(
            3,
            name="foo",
            trainable=False,
            dtype="float16",
        )
        serialized, restored, reserialized = self.roundtrip(layer)
        self.assertEqual(serialized, reserialized)
        self.assertEqual(layer.name, restored.name)
        self.assertEqual(layer.trainable, restored.trainable)
        self.assertEqual(layer.compute_dtype, restored.compute_dtype)

    def test_builtin_layers_with_alias_name(self):
        obj = serialization_lib.deserialize_keras_object(
            {
                "class_name": "AveragePooling1D",
                "module": "keras.layers",
                "registered_name": "",
                "config": {"pool_size": 2},
            }
        )
        self.assertIsInstance(obj, keras.layers.AveragePooling1D)

        obj = serialization_lib.deserialize_keras_object(
            {
                "class_name": "AvgPool1D",
                "module": "keras.layers",
                "registered_name": "",
                "config": {"pool_size": 2},
            }
        )
        self.assertIsInstance(obj, keras.layers.AveragePooling1D)

    def test_missing_module(self):
        with self.assertRaisesRegex(
            TypeError, "'keras.foo' cannot be imported"
        ):
            serialization_lib.deserialize_keras_object(
                {
                    "class_name": "Bar",
                    "module": "keras.foo",
                    "registered_name": "",
                    "config": {},
                }
            )

    def test_missing_attribute(self):
        with self.assertRaisesRegex(
            ValueError, "'keras.layers' has no attribute 'Foo'"
        ):
            serialization_lib.deserialize_keras_object(
                {
                    "class_name": "Foo",
                    "module": "keras.layers",
                    "registered_name": "",
                    "config": {},
                }
            )

    def test_numpy_get_item_layer(self):
        def tuples_to_lists_str(x):
            return str(x).replace("(", "[").replace(")", "]")

        input = keras.layers.Input(shape=(2,))
        layer = input[:, 1]
        model = keras.Model(input, layer)
        serialized, _, reserialized = self.roundtrip(model)
        # Anticipate JSON roundtrip mapping tuples to lists:
        serialized_str = tuples_to_lists_str(serialized)
        reserialized_str = tuples_to_lists_str(reserialized)
        self.assertEqual(serialized_str, reserialized_str)

    def test_serialize_ellipsis(self):
        _, deserialized, _ = self.roundtrip(Ellipsis)
        self.assertEqual(..., deserialized)

    def test_tensors_and_shapes(self):
        x = random.normal((2, 2), dtype="float64")
        obj = {"x": x}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertAllClose(x, new_obj["x"], atol=1e-5)

        obj = {"x.shape": x.shape}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertEqual(tuple(x.shape), tuple(new_obj["x.shape"]))

    def test_custom_fn(self):
        obj = {"activation": custom_fn}
        serialized, _, reserialized = self.roundtrip(
            obj, custom_objects={"custom_fn": custom_fn}
        )
        self.assertEqual(serialized, reserialized)

        # Test inside layer
        dense = keras.layers.Dense(1, activation=custom_fn)
        dense.build((None, 2))
        _, new_dense, _ = self.roundtrip(
            dense, custom_objects={"custom_fn": custom_fn}
        )
        x = random.normal((2, 2))
        y1 = dense(x)
        _ = new_dense(x)
        new_dense.set_weights(dense.get_weights())
        y2 = new_dense(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_custom_layer(self):
        layer = CustomLayer(factor=2)
        x = random.normal((2, 2))
        y1 = layer(x)
        _, new_layer, _ = self.roundtrip(
            layer, custom_objects={"CustomLayer": CustomLayer}
        )
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

        layer = NestedCustomLayer(factor=2)
        x = random.normal((2, 2))
        y1 = layer(x)
        _, new_layer, _ = self.roundtrip(
            layer,
            custom_objects={
                "NestedCustomLayer": NestedCustomLayer,
                "custom_fn": custom_fn,
            },
        )
        _ = new_layer(x)
        new_layer.set_weights(layer.get_weights())
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_lambda_fn(self):
        obj = {"activation": lambda x: x**2}
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            self.roundtrip(obj, safe_mode=True)

        _, new_obj, _ = self.roundtrip(obj, safe_mode=False)
        self.assertEqual(obj["activation"](3), new_obj["activation"](3))

    def test_lambda_layer(self):
        lmbda = keras.layers.Lambda(lambda x: x**2)
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            self.roundtrip(lmbda, safe_mode=True)

        _, new_lmbda, _ = self.roundtrip(lmbda, safe_mode=False)
        x = random.normal((2, 2))
        y1 = lmbda(x)
        y2 = new_lmbda(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_deserialize_invalid_type_config(self):
        bad_config = {"class_name": "Dense", "config": [1, 2, 3]}
        with self.assertRaisesRegex(
            TypeError, "Expected 'config' to be a dict"
        ):
            serialization_lib.deserialize_keras_object(bad_config)

    def test_deserialize_invalid_typespec_config(self):
        bad_config = {
            "class_name": "__typespec__",
            "config": "not_a_list_nor_a_tuple",
        }
        with self.assertRaisesRegex(
            TypeError, "Expected 'config' to be a list or a tuple"
        ):
            serialization_lib.deserialize_keras_object(bad_config)

    def test_deserialize_invalid_function_config(self):
        bad_config = {
            "class_name": "function",
            "config": ["not", "a", "string"],
        }
        with self.assertRaisesRegex(
            TypeError, "Expected 'config' to be a non-null str"
        ):
            serialization_lib.deserialize_keras_object(bad_config)

    def test_safe_mode_scope(self):
        lmbda = keras.layers.Lambda(lambda x: x**2)
        with serialization_lib.SafeModeScope(safe_mode=True):
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                self.roundtrip(lmbda)
        with serialization_lib.SafeModeScope(safe_mode=False):
            _, new_lmbda, _ = self.roundtrip(lmbda)
        x = random.normal((2, 2))
        y1 = lmbda(x)
        y2 = new_lmbda(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_dict_inputs_outputs(self):
        input_foo = keras.Input((2,), name="foo")
        input_bar = keras.Input((2,), name="bar")
        dense = keras.layers.Dense(1)
        output_foo = dense(input_foo)
        output_bar = dense(input_bar)
        model = keras.Model(
            {"foo": input_foo, "bar": input_bar},
            {"foo": output_foo, "bar": output_bar},
        )
        _, new_model, _ = self.roundtrip(model)
        original_output = model(
            {"foo": np.zeros((2, 2)), "bar": np.zeros((2, 2))}
        )
        restored_output = model(
            {"foo": np.zeros((2, 2)), "bar": np.zeros((2, 2))}
        )
        self.assertAllClose(original_output["foo"], restored_output["foo"])
        self.assertAllClose(original_output["bar"], restored_output["bar"])

    def test_shared_inner_layer(self):
        with serialization_lib.ObjectSharingScope():
            input_1 = keras.Input((2,))
            input_2 = keras.Input((2,))
            shared_layer = keras.layers.Dense(1)
            output_1 = shared_layer(input_1)
            wrapper_layer = WrapperLayer(shared_layer)
            output_2 = wrapper_layer(input_2)
            model = keras.Model([input_1, input_2], [output_1, output_2])
            _, new_model, _ = self.roundtrip(
                model, custom_objects={"WrapperLayer": WrapperLayer}
            )

            self.assertIs(model.layers[2], model.layers[3].layer)
            self.assertIs(new_model.layers[2], new_model.layers[3].layer)

    def test_functional_subclass(self):
        class PlainFunctionalSubclass(keras.Model):
            pass

        inputs = keras.Input((2,), batch_size=3)
        outputs = keras.layers.Dense(1)(inputs)
        model = PlainFunctionalSubclass(inputs, outputs)
        x = random.normal((2, 2))
        y1 = model(x)
        _, new_model, _ = self.roundtrip(
            model,
            custom_objects={"PlainFunctionalSubclass": PlainFunctionalSubclass},
        )
        new_model.set_weights(model.get_weights())
        y2 = new_model(x)
        self.assertAllClose(y1, y2, atol=1e-5)
        self.assertIsInstance(new_model, PlainFunctionalSubclass)

        class FunctionalSubclassWCustomInit(keras.Model):
            def __init__(self, num_units=2):
                inputs = keras.Input((2,), batch_size=3)
                outputs = keras.layers.Dense(num_units)(inputs)
                super().__init__(inputs, outputs)
                self.num_units = num_units

            def get_config(self):
                return {"num_units": self.num_units}

        model = FunctionalSubclassWCustomInit(num_units=3)
        x = random.normal((2, 2))
        y1 = model(x)
        _, new_model, _ = self.roundtrip(
            model,
            custom_objects={
                "FunctionalSubclassWCustomInit": FunctionalSubclassWCustomInit
            },
        )
        new_model.set_weights(model.get_weights())
        y2 = new_model(x)
        self.assertAllClose(y1, y2, atol=1e-5)
        self.assertIsInstance(new_model, FunctionalSubclassWCustomInit)

    def test_shared_object(self):
        class MyLayer(keras.layers.Layer):
            def __init__(self, activation, **kwargs):
                super().__init__(**kwargs)
                if isinstance(activation, dict):
                    self.activation = (
                        serialization_lib.deserialize_keras_object(activation)
                    )
                else:
                    self.activation = activation

            def call(self, x):
                return self.activation(x)

            def get_config(self):
                config = super().get_config()
                config["activation"] = self.activation
                return config

        class SharedActivation:
            def __call__(self, x):
                return x**2

            def get_config(self):
                return {}

            @classmethod
            def from_config(cls, config):
                return cls()

        shared_act = SharedActivation()
        layer_1 = MyLayer(activation=shared_act)
        layer_2 = MyLayer(activation=shared_act)
        layers = [layer_1, layer_2]

        with serialization_lib.ObjectSharingScope():
            serialized, new_layers, reserialized = self.roundtrip(
                layers,
                custom_objects={
                    "MyLayer": MyLayer,
                    "SharedActivation": SharedActivation,
                },
            )
        self.assertIn("shared_object_id", serialized[0]["config"]["activation"])
        obj_id = serialized[0]["config"]["activation"]
        self.assertIn("shared_object_id", serialized[1]["config"]["activation"])
        self.assertEqual(obj_id, serialized[1]["config"]["activation"])
        self.assertIs(layers[0].activation, layers[1].activation)
        self.assertIs(new_layers[0].activation, new_layers[1].activation)

    def test_layer_sharing(self):
        seq = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(5),
                keras.layers.Softmax(),
            ],
        )
        func = keras.Model(inputs=seq.inputs, outputs=seq.outputs)
        serialized, deserialized, reserialized = self.roundtrip(func)
        self.assertLen(deserialized.layers, 3)

    def test_keras36_custom_function_reloading(self):
        @object_registration.register_keras_serializable(package="serial_test")
        def custom_registered_fn(x):
            return x**2

        config36 = {
            "module": "builtins",
            "class_name": "function",
            "config": "custom_registered_fn",
            "registered_name": "function",
        }
        obj = serialization_lib.deserialize_keras_object(config36)
        self.assertIs(obj, custom_registered_fn)

        config = {
            "module": "builtins",
            "class_name": "function",
            "config": "serial_test>custom_registered_fn",
            "registered_name": "function",
        }
        obj = serialization_lib.deserialize_keras_object(config)
        self.assertIs(obj, custom_registered_fn)

    def test_layer_instance_as_activation(self):
        """Tests serialization when activation is a Layer instance."""

        # Dense layer with ReLU layer as activation
        layer_dense_relu = keras.layers.Dense(
            units=4, activation=keras.layers.ReLU(name="my_relu")
        )
        # Build the layer to ensure weights/state are initialized if needed
        layer_dense_relu.build(input_shape=(None, 8))
        _, restored_dense_relu, _ = self.roundtrip(layer_dense_relu)

        # Verify the activation is correctly deserialized as a ReLU layer
        self.assertIsInstance(restored_dense_relu.activation, keras.layers.ReLU)
        # Verify properties are preserved
        self.assertEqual(restored_dense_relu.activation.name, "my_relu")

    def test_layer_instance_with_config_as_activation(self):
        """
        Tests serialization when activation is a Layer instance with config.
        """

        # Conv1D layer with LeakyReLU layer (with config) as activation
        leaky_activation = keras.layers.LeakyReLU(
            negative_slope=0.15, name="my_leaky"
        )
        layer_conv_leaky = keras.layers.Conv1D(
            filters=2, kernel_size=3, activation=leaky_activation
        )
        # Build the layer
        layer_conv_leaky.build(input_shape=(None, 10, 4))
        _, restored_conv_leaky, _ = self.roundtrip(layer_conv_leaky)

        # Verify the activation is correctly deserialized as LeakyReLU
        self.assertIsInstance(
            restored_conv_leaky.activation, keras.layers.LeakyReLU
        )
        # Verify configuration of the activation layer is preserved
        self.assertEqual(restored_conv_leaky.activation.negative_slope, 0.15)
        self.assertEqual(restored_conv_leaky.activation.name, "my_leaky")

    def test_layer_string_as_activation(self):
        """Tests serialization when activation is a string."""

        layer = keras.layers.Dense(units=4, activation="gelu")
        layer.build(input_shape=(None, 8))
        _, restored_layer, _ = self.roundtrip(layer)
        self.assertIs(restored_layer.activation, keras.activations.gelu)

    def test_layer_function_as_activation(self):
        """Tests serialization when activation is an activation function."""

        layer = keras.layers.Dense(units=4, activation=keras.activations.gelu)
        layer.build(input_shape=(None, 8))
        _, restored_layer, _ = self.roundtrip(layer)
        self.assertIs(restored_layer.activation, keras.activations.gelu)

    def test_layer_op_as_activation(self):
        """Tests serialization when activation is an op."""

        layer = keras.layers.Dense(units=4, activation=keras.ops.gelu)
        layer.build(input_shape=(None, 8))
        _, restored_layer, _ = self.roundtrip(layer)
        self.assertIs(restored_layer.activation, keras.ops.gelu)
        self.assertIsNot(restored_layer.activation, keras.activations.gelu)

    def test_layer_registered_function_as_activation(self):
        """Tests serialization when activation is a registered function."""

        @keras.saving.register_keras_serializable(package="serial_test")
        def gelu(x):
            return x

        layer = keras.layers.Dense(units=4, activation=gelu)
        layer.build(input_shape=(None, 8))
        _, restored_layer, _ = self.roundtrip(layer)
        self.assertIs(restored_layer.activation, gelu)

    def test_layer_unregistered_function_as_activation(self):
        """Tests serialization when activation is an unregistered function."""

        def gelu(x):
            return x

        layer = keras.layers.Dense(units=4, activation=gelu)
        layer.build(input_shape=(None, 8))
        _, restored_layer, _ = self.roundtrip(
            layer, custom_objects={"gelu": gelu}
        )
        self.assertIs(restored_layer.activation, gelu)

    def test_missing_layers_for_sequential_model(self):
        """Tests serialization when sequential model has no name."""
        serialized = {
            "class_name": "Sequential",
            "module": "keras",
            "config": {"name": "sequential_model"},
        }
        with self.assertRaisesRegex(
            ValueError,
            "A Sequential model configuration must be a "
            "dictionary containing the 'layers' key",
        ):
            deserialize_keras_object(serialized)

    def test_malformed_layer_for_sequential_model(self):
        serialized = {
            "class_name": "Sequential",
            "module": "keras",
            "config": {
                "name": "sequential_model",
                "layers": [
                    {
                        "class_name": "Dense",
                        "module": "keras.layers",
                        "config": {"units": 4},
                    },
                    {},
                ],
            },
        }
        with self.assertRaisesRegex(
            ValueError, "Expected a dictionary containing a 'config' key."
        ):
            deserialize_keras_object(serialized)

    def test_registered_name_for_keras_builtin_raises(self):
        for bad_name in (
            "builtins.eval",
            "subprocess.Popen",
            "not.a.real.name",
            "my_pkg>MyDense",
        ):
            with self.assertRaisesRegex(
                ValueError, "resolved to a Keras built-in"
            ):
                serialization_lib.deserialize_keras_object(
                    {
                        "class_name": "Dense",
                        "module": "keras.layers",
                        "registered_name": bad_name,
                        "config": {"units": 1},
                    }
                )

    def test_registered_name_matching_class_name_passes_through(self):
        obj = serialization_lib.deserialize_keras_object(
            {
                "class_name": "Dense",
                "module": "keras.layers",
                "registered_name": "Dense",
                "config": {"units": 1},
            }
        )
        self.assertIsInstance(obj, keras.layers.Dense)

    def test_none_or_missing_registered_name_passes_through(self):
        for config in (
            {
                "class_name": "Dense",
                "module": "keras.layers",
                "registered_name": None,
                "config": {"units": 1},
            },
            {
                "class_name": "Dense",
                "module": "keras.layers",
                "config": {"units": 1},
            },
        ):
            obj = serialization_lib.deserialize_keras_object(config)
            self.assertIsInstance(obj, keras.layers.Dense)

    def test_empty_string_registered_name_passes_through(self):
        obj = serialization_lib.deserialize_keras_object(
            {
                "class_name": "Dense",
                "module": "keras.layers",
                "registered_name": "",
                "config": {"units": 1},
            }
        )
        self.assertIsInstance(obj, keras.layers.Dense)

    def test_registered_name_for_keras_builtin_function_raises(self):
        with self.assertRaisesRegex(ValueError, "resolved to a Keras built-in"):
            serialization_lib.deserialize_keras_object(
                {
                    "class_name": "function",
                    "module": "builtins",
                    "registered_name": "some_other_name",
                    "config": "relu",
                }
            )

    def test_registered_name_matching_function_name_passes_through(self):
        for registered_name in ("function", "kl_divergence"):
            obj = serialization_lib.deserialize_keras_object(
                {
                    "class_name": "function",
                    "module": "builtins",
                    "registered_name": registered_name,
                    "config": "kl_divergence",
                }
            )
            self.assertIs(obj, keras.metrics.kl_divergence)

    def test_registered_custom_object_still_resolves(self):
        layer = MyDense(units=4)
        config = serialization_lib.serialize_keras_object(layer)
        restored = serialization_lib.deserialize_keras_object(config)
        self.assertIsInstance(restored, MyDense)


@keras.saving.register_keras_serializable()
class MyDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        *,
        kernel_regularizer=None,
        kernel_initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._units = units
        self._kernel_regularizer = kernel_regularizer
        self._kernel_initializer = kernel_initializer

    def get_config(self):
        return dict(
            units=self._units,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            **super().get_config(),
        )

    def build(self, input_shape):
        _, input_units = input_shape
        self._kernel = self.add_weight(
            name="kernel",
            shape=[input_units, self._units],
            dtype="float32",
            regularizer=self._kernel_regularizer,
            initializer=self._kernel_initializer,
        )

    def call(self, inputs):
        return ops.matmul(inputs, self._kernel)


@keras.saving.register_keras_serializable()
class MyWrapper(keras.layers.Layer):
    def __init__(self, wrapped, **kwargs):
        super().__init__(**kwargs)
        self._wrapped = wrapped

    def get_config(self):
        return dict(wrapped=self._wrapped, **super().get_config())

    @classmethod
    def from_config(cls, config):
        config["wrapped"] = keras.saving.deserialize_keras_object(
            config["wrapped"]
        )
        return cls(**config)

    def call(self, inputs):
        return self._wrapped(inputs)
