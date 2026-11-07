import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src import tree
from keras.src import utils
from keras.src.dtype_policies.dtype_policy import DTypePolicy
from keras.src.saving import object_registration
from keras.src.utils.jax_layer import FlaxLayer
from keras.src.utils.jax_layer import JaxLayer

try:
    import flax
except ImportError:
    flax = None

num_classes = 10
input_shape = (28, 28, 1)  # Excluding batch_size


@object_registration.register_keras_serializable()
def jax_stateless_function(inputs):
    x = jax.numpy.sum(inputs, axis=(1, 2, 3))
    x = jax.numpy.expand_dims(x, axis=1)
    x = jax.numpy.tile(x, (1, 10))
    return x


@object_registration.register_keras_serializable()
def jax_model_no_state_init(rng, inputs):
    layer_sizes = [784, 300, 100, 10]
    params = []
    w_init = jax.nn.initializers.glorot_normal()
    b_init = jax.nn.initializers.normal(0.1)
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, w_rng = jax.random.split(rng)
        rng, b_rng = jax.random.split(rng)
        params.append([w_init(w_rng, (m, n)), b_init(b_rng, (n,))])
    return params


@object_registration.register_keras_serializable()
def jax_model_no_state_apply(params, inputs):
    activations = inputs.reshape((inputs.shape[0], -1))  # flatten
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return jax.nn.softmax(logits, axis=-1)


@object_registration.register_keras_serializable()
def jax_model_with_state_init(rng, inputs, training):
    params = jax_model_no_state_init(rng, inputs)
    state = jnp.zeros([], jnp.float32)
    return params, state


@object_registration.register_keras_serializable()
def jax_model_with_state_apply(params, state, inputs, training):
    outputs = jax_model_no_state_apply(params, inputs)
    if training:
        state = state + 1
    return outputs, state


if flax is not None:

    @object_registration.register_keras_serializable()
    class FlaxTrainingIndependentModel(flax.linen.Module):
        @flax.linen.compact
        def forward(self, inputs):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

        def get_config(self):
            return {}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    @object_registration.register_keras_serializable()
    class FlaxDropoutModel(flax.linen.Module):
        @flax.linen.compact
        def my_apply(self, inputs, training):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.Dropout(rate=0.3, deterministic=not training)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

        def get_config(self):
            return {}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    @object_registration.register_keras_serializable()
    def flax_dropout_wrapper(module, x, training):
        return module.my_apply(x, training)

    @object_registration.register_keras_serializable()
    class FlaxBatchNormModel(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, inputs, training=False):
            ura = not training
            x = inputs
            x = flax.linen.Conv(
                features=12, kernel_size=(3, 3), use_bias=False
            )(x)
            x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(
                x
            )
            x = flax.linen.relu(x)
            x = flax.linen.Conv(
                features=24, kernel_size=(6, 6), strides=(2, 2)
            )(x)
            x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(
                x
            )
            x = flax.linen.relu(x)
            x = flax.linen.Conv(
                features=32, kernel_size=(6, 6), strides=(2, 2)
            )(x)
            x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(
                x
            )
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200, use_bias=True)(x)
            x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(
                x
            )
            x = flax.linen.Dropout(rate=0.3, deterministic=not training)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

        def get_config(self):
            return {}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    FLAX_OBJECTS = {
        "FlaxTrainingIndependentModel": FlaxTrainingIndependentModel,
        "FlaxBatchNormModel": FlaxBatchNormModel,
        "FlaxDropoutModel": FlaxDropoutModel,
        "flax_dropout_wrapper": flax_dropout_wrapper,
    }


@pytest.mark.skipif(
    backend.backend() not in ["jax", "tensorflow"],
    reason="JaxLayer and FlaxLayer are only supported with JAX and TF backend",
)
class TestJaxLayer(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.init_kwargs = {}
        if testing.tensorflow_uses_gpu():
            self.init_kwargs = {
                "native_serialization_platforms": ("cpu", "cuda")
            }

    def _test_layer(
        self,
        model_name,
        layer_class,
        layer_init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        layer_init_kwargs.update(self.init_kwargs)

        # Fake MNIST data
        x_train = random.uniform(shape=(320, 28, 28, 1))
        y_train_indices = ops.cast(
            random.uniform(shape=(320,), minval=0, maxval=num_classes),
            dtype="int32",
        )
        y_train = ops.one_hot(y_train_indices, num_classes, dtype="int32")
        x_test = random.uniform(shape=(32, 28, 28, 1))

        def _count_params(weights):
            count = 0
            for weight in weights:
                count = count + math.prod(ops.shape(weight))
            return count

        def verify_weights_and_params(layer):
            self.assertEqual(trainable_weights, len(layer.trainable_weights))
            self.assertEqual(
                trainable_params,
                _count_params(layer.trainable_weights),
            )
            self.assertEqual(
                non_trainable_weights, len(layer.non_trainable_weights)
            )
            self.assertEqual(
                non_trainable_params,
                _count_params(layer.non_trainable_weights),
            )

        # functional model
        layer1 = layer_class(**layer_init_kwargs)
        inputs1 = layers.Input(shape=input_shape)
        outputs1 = layer1(inputs1)
        model1 = models.Model(
            inputs=inputs1, outputs=outputs1, name=f"{model_name}1"
        )
        model1.summary()

        verify_weights_and_params(layer1)

        model1.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[metrics.CategoricalAccuracy()],
        )

        tw1_before_fit = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.trainable_weights
        )
        ntw1_before_fit = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.non_trainable_weights
        )
        model1.fit(x_train, y_train, epochs=1, steps_per_epoch=10)
        tw1_after_fit = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.trainable_weights
        )
        ntw1_after_fit = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.non_trainable_weights
        )

        # verify both trainable and non-trainable weights did change after fit
        for before, after in zip(tw1_before_fit, tw1_after_fit):
            self.assertNotAllClose(before, after)
        for before, after in zip(ntw1_before_fit, ntw1_after_fit):
            self.assertNotAllClose(before, after)

        expected_output_shape = (ops.shape(x_test)[0], num_classes)
        output1 = model1(x_test)
        self.assertEqual(output1.shape, expected_output_shape)
        predict1 = model1.predict(x_test, steps=1)
        self.assertEqual(predict1.shape, expected_output_shape)

        # verify both trainable and non-trainable weights did not change
        tw1_after_call = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.trainable_weights
        )
        ntw1_after_call = tree.map_structure(
            backend.ops.convert_to_numpy, layer1.non_trainable_weights
        )
        for after_fit, after_call in zip(tw1_after_fit, tw1_after_call):
            self.assertAllClose(after_fit, after_call)
        for after_fit, after_call in zip(ntw1_after_fit, ntw1_after_call):
            self.assertAllClose(after_fit, after_call)

        exported_params = jax.tree_util.tree_map(
            backend.ops.convert_to_numpy, layer1.params
        )
        if layer1.state is not None:
            exported_state = jax.tree_util.tree_map(
                backend.ops.convert_to_numpy, layer1.state
            )
        else:
            exported_state = None

        def verify_identical_model(model):
            output = model(x_test)
            self.assertAllClose(output1, output)

            predict = model.predict(x_test, steps=1)
            self.assertAllClose(predict1, predict)

        # sequential model to compare results
        layer2 = layer_class(
            params=exported_params,
            state=exported_state,
            input_shape=input_shape,
            **layer_init_kwargs,
        )
        model2 = models.Sequential([layer2], name=f"{model_name}2")
        model2.summary()
        verify_weights_and_params(layer2)
        model2.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[metrics.CategoricalAccuracy()],
        )
        verify_identical_model(model2)

        # save, load back and compare results
        path = os.path.join(self.get_temp_dir(), "jax_layer_model.keras")
        model2.save(path)

        model3 = saving.load_model(path)
        layer3 = model3.layers[0]
        model3.summary()
        verify_weights_and_params(layer3)
        verify_identical_model(model3)

        # export, load back and compare results
        path = os.path.join(self.get_temp_dir(), "jax_layer_export")
        export_kwargs = {}
        if testing.jax_uses_gpu():
            export_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "cuda")
                }
            }
        elif testing.jax_uses_tpu():
            export_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "tpu")
                }
            }
        model2.export(path, format="tf_saved_model", **export_kwargs)
        model4 = tf.saved_model.load(path)
        output4 = model4.serve(x_test)
        self.assertAllClose(output1, output4, atol=1e-2, rtol=1e-3)

        # test subclass model building without a build method
        class TestModel(models.Model):
            def __init__(self, layer):
                super().__init__()
                self._layer = layer

            def call(self, inputs):
                return self._layer(inputs)

        layer5 = layer_class(**layer_init_kwargs)
        model5 = TestModel(layer5)
        output5 = model5(x_test)
        self.assertNotAllClose(output5, 0.0)

    @parameterized.named_parameters(
        {
            "testcase_name": "stateless",
            "init_kwargs": {
                "call_fn": jax_stateless_function,
                "init_fn": None,
            },
            "trainable_weights": 0,
            "trainable_params": 0,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_independent",
            "init_kwargs": {
                "call_fn": jax_model_no_state_apply,
                "init_fn": jax_model_no_state_init,
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_state",
            "init_kwargs": {
                "call_fn": jax_model_with_state_apply,
                "init_fn": jax_model_with_state_init,
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 1,
            "non_trainable_params": 1,
        },
        {
            "testcase_name": "training_state_dtype_policy",
            "init_kwargs": {
                "call_fn": jax_model_with_state_apply,
                "init_fn": jax_model_with_state_init,
                "dtype": DTypePolicy("mixed_float16"),
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 1,
            "non_trainable_params": 1,
        },
        {
            "testcase_name": "native_serialization",
            "init_kwargs": {
                "call_fn": jax_model_no_state_apply,
                "init_fn": jax_model_no_state_init,
                "native_serialization_platforms": ("cpu", "tpu"),
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
    )
    def test_jax_layer(
        self,
        init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        self._test_layer(
            init_kwargs["call_fn"].__name__,
            JaxLayer,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "training_independent_bound_method",
            "flax_model_class": "FlaxTrainingIndependentModel",
            "flax_model_method": "forward",
            "init_kwargs": {},
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_rng_unbound_method",
            "flax_model_class": "FlaxDropoutModel",
            "flax_model_method": None,
            "init_kwargs": {
                "method": "flax_dropout_wrapper",
            },
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_rng_state_no_method",
            "flax_model_class": "FlaxBatchNormModel",
            "flax_model_method": None,
            "init_kwargs": {},
            "trainable_weights": 13,
            "trainable_params": 354258,
            "non_trainable_weights": 8,
            "non_trainable_params": 536,
        },
        {
            "testcase_name": "training_rng_unbound_method_dtype_policy",
            "flax_model_class": "FlaxDropoutModel",
            "flax_model_method": None,
            "init_kwargs": {
                "method": "flax_dropout_wrapper",
                "dtype": DTypePolicy("mixed_float16"),
            },
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
    )
    @pytest.mark.skipif(flax is None, reason="Flax library is not available.")
    def test_flax_layer(
        self,
        flax_model_class,
        flax_model_method,
        init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        flax_model_class = FLAX_OBJECTS.get(flax_model_class)
        if "method" in init_kwargs:
            init_kwargs["method"] = FLAX_OBJECTS.get(init_kwargs["method"])

        def create_wrapper(**kwargs):
            params = kwargs.pop("params") if "params" in kwargs else None
            state = kwargs.pop("state") if "state" in kwargs else None
            if params and state:
                variables = {**params, **state}
            elif params:
                variables = params
            elif state:
                variables = state
            else:
                variables = None
            kwargs["variables"] = variables
            flax_model = flax_model_class()
            if flax_model_method:
                kwargs["method"] = getattr(flax_model, flax_model_method)
            return FlaxLayer(flax_model_class(), **kwargs)

        self._test_layer(
            flax_model_class.__name__,
            create_wrapper,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    def test_with_no_init_fn_and_no_params(self):
        def jax_fn(params, inputs):
            return inputs

        with self.assertRaises(ValueError):
            JaxLayer(jax_fn)

    def test_with_training_in_call_fn_but_not_init_fn(self):
        def jax_call_fn(params, state, rng, inputs, training):
            return inputs, {}

        def jax_init_fn(rng, inputs):
            return {}, {}

        layer = JaxLayer(jax_call_fn, jax_init_fn, **self.init_kwargs)
        layer(np.ones((1,)))

    def test_with_different_argument_order(self):
        def jax_call_fn(training, inputs, rng, state, params):
            return inputs, {}

        def jax_init_fn(training, inputs, rng):
            return {}, {}

        layer = JaxLayer(jax_call_fn, jax_init_fn, **self.init_kwargs)
        layer(np.ones((1,)))

    def test_with_minimal_arguments(self):
        def jax_call_fn(inputs):
            return inputs

        def jax_init_fn(inputs):
            return {}

        layer = JaxLayer(jax_call_fn, jax_init_fn, **self.init_kwargs)
        layer(np.ones((1,)))

    def test_with_missing_inputs_in_call_fn(self):
        def jax_call_fn(params, rng, training):
            return jnp.ones((1,))

        def jax_init_fn(rng, inputs):
            return {}

        with self.assertRaisesRegex(ValueError, "`call_fn`.*`inputs`"):
            JaxLayer(jax_call_fn, jax_init_fn)

    def test_with_missing_inputs_in_init_fn(self):
        def jax_call_fn(params, rng, inputs, training):
            return jnp.ones((1,))

        def jax_init_fn(rng, training):
            return {}

        with self.assertRaisesRegex(ValueError, "`init_fn`.*`inputs`"):
            JaxLayer(jax_call_fn, jax_init_fn)

    def test_with_unsupported_argument_in_call_fn(self):
        def jax_call_fn(params, rng, inputs, mode):
            return jnp.ones((1,))

        def jax_init_fn(rng, inputs):
            return {}

        with self.assertRaisesRegex(ValueError, "`call_fn`.*`mode`"):
            JaxLayer(jax_call_fn, jax_init_fn)

    def test_with_unsupported_argument_in_init_fn(self):
        def jax_call_fn(params, rng, inputs, training):
            return inputs

        def jax_init_fn(rng, inputs, mode):
            return {}

        with self.assertRaisesRegex(ValueError, "`init_fn`.*`mode`"):
            JaxLayer(jax_call_fn, jax_init_fn)

    def test_with_structures_as_inputs_and_outputs(self):
        def jax_fn(params, inputs):
            a = inputs["a"]
            b = inputs["b"]
            output1 = jnp.concatenate([a, b], axis=1)
            output2 = jnp.concatenate([b, a], axis=1)
            return output1, output2

        layer = JaxLayer(jax_fn, params={}, **self.init_kwargs)
        inputs = {
            "a": layers.Input((None, 3)),
            "b": layers.Input((None, 3)),
        }
        outputs = layer(inputs)
        model = models.Model(inputs, outputs)

        test_inputs = {
            "a": np.ones((2, 6, 3)),
            "b": np.ones((2, 7, 3)),
        }
        test_outputs = model(test_inputs)
        self.assertAllClose(test_outputs[0], np.ones((2, 13, 3)))
        self.assertAllClose(test_outputs[1], np.ones((2, 13, 3)))

    def test_with_polymorphic_shape_more_than_26_dimension_names(self):
        def jax_fn(params, inputs):
            return jnp.concatenate(inputs, axis=1)

        layer = JaxLayer(jax_fn, params=(), **self.init_kwargs)
        inputs = [layers.Input((None, 3)) for _ in range(60)]
        output = layer(inputs)
        model = models.Model(inputs, output)

        test_inputs = [np.ones((2, 1, 3))] * 60
        test_output = model(test_inputs)
        self.assertAllClose(test_output, np.ones((2, 60, 3)))

    @pytest.mark.skipif(flax is None, reason="Flax library is not available.")
    def test_with_flax_state_no_params(self):
        class MyFlaxLayer(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x):
                def zeros_init(shape):
                    return jnp.zeros(shape, jnp.int32)

                count = self.variable("a", "b", zeros_init, [])
                count.value = count.value + 1
                return x

        layer = FlaxLayer(
            MyFlaxLayer(), variables={"a": {"b": 0}}, **self.init_kwargs
        )
        layer(np.ones((1,)))
        self.assertLen(layer.params, 0)
        self.assertEqual(layer.state["a"]["b"].value, 1)

    def test_with_state_none_leaves(self):
        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(jax_fn, state={"foo": None}, **self.init_kwargs)
        self.assertIsNone(layer.state["foo"])
        layer(np.ones((1,)))

    def test_with_state_non_tensor_leaves(self):
        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(jax_fn, state={"foo": "bar"}, **self.init_kwargs)
        self.assertEqual(layer.state["foo"], "bar")
        # layer cannot be invoked as jax2tf will fail on strings

    def test_with_state_jax_registered_node_class(self):
        @jax.tree_util.register_pytree_node_class
        class NamedPoint:
            def __init__(self, x, y, name):
                self.x = x
                self.y = y
                self.name = name

            def tree_flatten(self):
                return ((self.x, self.y), self.name)

            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(*children, aux_data)

        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(
            jax_fn, state=[NamedPoint(1.0, 2.0, "foo")], **self.init_kwargs
        )
        layer(np.ones((1,)))

    @parameterized.named_parameters(
        {
            "testcase_name": "sequence_instead_of_mapping",
            "init_state": [0.0],
        },
        {
            "testcase_name": "mapping_instead_of_sequence",
            "init_state": {"state": {"foo": 0.0}},
        },
        {
            "testcase_name": "sequence_instead_of_variable",
            "init_state": {"state": [[0.0]]},
        },
        {
            "testcase_name": "no_initial_state",
            "init_state": None,
        },
        {
            "testcase_name": "missing_dict_key",
            "init_state": {"state": {}},
        },
        {
            "testcase_name": "missing_variable_in_list",
            "init_state": {"state": {"foo": [2.0]}},
        },
    )
    def test_state_mismatch_during_update(self, init_state):
        def jax_fn(params, state, inputs):
            return inputs, {"state": [jnp.ones([])]}

        layer = JaxLayer(
            jax_fn, params={}, state=init_state, **self.init_kwargs
        )
        with self.assertRaises(ValueError):
            layer(np.ones((1,)))

    def test_rng_seeding(self):
        def jax_init(rng, inputs):
            return [jax.nn.initializers.normal(1.0)(rng, inputs.shape)]

        def jax_apply(params, inputs):
            return jnp.dot(inputs, params[0])

        shape = (2, 2)

        utils.set_random_seed(0)
        layer1 = JaxLayer(jax_apply, jax_init, **self.init_kwargs)
        layer1.build(shape)
        utils.set_random_seed(0)
        layer2 = JaxLayer(jax_apply, jax_init, **self.init_kwargs)
        layer2.build(shape)
        self.assertAllClose(layer1.params[0], layer2.params[0])

    def test_with_keras_variable_leaves(self):
        def jax_call_fn(params, state, inputs):
            return jnp.dot(inputs, params["w"]) + state["b"], state

        w = backend.Variable(initializer="ones", shape=(3, 2), trainable=True)
        b = backend.Variable(initializer="zeros", shape=(2,), trainable=False)
        layer = JaxLayer(
            jax_call_fn, params={"w": w}, state={"b": b}, **self.init_kwargs
        )

        # The variables are used as-is, not copied.
        self.assertIs(layer.params["w"], w)
        self.assertIs(layer.state["b"], b)
        self.assertLen(layer.trainable_weights, 1)
        self.assertIs(layer.trainable_weights[0], w)
        self.assertLen(layer.non_trainable_weights, 1)
        self.assertIs(layer.non_trainable_weights[0], b)

        # The layer reads the current value of the variables.
        w.assign(np.full((3, 2), 2.0, dtype="float32"))
        out = layer(np.ones((1, 3), dtype="float32"))
        self.assertAllClose(out, np.full((1, 2), 6.0, dtype="float32"))

    def test_with_keras_variable_leaves_uninitialized(self):
        def jax_call_fn(params, inputs):
            return jnp.dot(inputs, params["w"])

        with backend.StatelessScope(initialize_variables=False):
            w = backend.Variable(initializer="ones", shape=(3, 2))
            layer = JaxLayer(jax_call_fn, params={"w": w}, **self.init_kwargs)

        # Creating the layer does not initialize or copy the variable.
        self.assertIsNone(w._value)
        self.assertIs(layer.params["w"], w)

        w._deferred_initialize()
        out = layer(np.ones((1, 3), dtype="float32"))
        self.assertAllClose(out, np.full((1, 2), 3.0, dtype="float32"))
