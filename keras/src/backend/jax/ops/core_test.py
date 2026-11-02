import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.config import is_nnx_enabled
from keras.src.backend.jax.ops.core import Variable

if is_nnx_enabled():
    from flax import nnx


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not is_nnx_enabled(),
    reason="Test requires NNX backend to be enabled by default for setup.",
)
class NnxVariableTest(testing.TestCase):
    def setUp(self):
        super().setUp()

        class NNXModel(nnx.Module):
            def __init__(self, rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)
                # `Variable` is automatically the NNX specific sub-class.
                self.custom_variable = Variable(jnp.ones((1, 3)))

            def __call__(self, x):
                return self.linear(x) + self.custom_variable

        self.nnx_model = NNXModel(rngs=nnx.Rngs(0))
        self.keras_nnx_model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=(10,))]
        )
        self.single_dummy_input = np.random.rand(1, 10)

    def test_variable_in_nnx_module(self):
        self.assertTrue(hasattr(self.nnx_model.custom_variable, "_trace_state"))
        self.assertIsNotNone(self.nnx_model.custom_variable._trace_state)
        self.assertAllClose(self.nnx_model.custom_variable.value, [[1, 1, 1]])
        self.assertTrue(
            isinstance(self.nnx_model.custom_variable, nnx.Variable)
        )

    def test_model_saving(self):
        path = os.path.join(self.get_temp_dir(), "model.keras")
        original_outputs = self.keras_nnx_model(self.single_dummy_input)
        self.keras_nnx_model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)
        restored_outputs = restored_model(self.single_dummy_input)
        self.assertAllClose(restored_outputs, original_outputs)

    def test_keras_variable_nnx_split_merge_sync(self):
        variable1 = keras.Variable(jnp.array(1.0))
        graphdef, state = nnx.split(variable1)
        state = jax.tree.map(lambda x: x + 1, state)
        variable2 = nnx.merge(graphdef, state)
        self.assertEqual(variable2._value, variable2.value)

    def test_lazy_build_within_symbolic_trace(self):
        class InnerNorm(keras.layers.Layer):
            def build(self, input_shape):
                self.weight = self.add_weight(
                    name="weight",
                    shape=(input_shape[-1],),
                    initializer="ones",
                )
                self.built = True

            def call(self, x):
                return self.weight * x

        class OuterBlock(keras.layers.Layer):
            # No `build()`: triggers build-by-run inside `jax.eval_shape`.
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.norm = InnerNorm()

            def call(self, x):
                return self.norm(x) + x

        inputs = keras.Input(shape=(4,))
        outputs = OuterBlock()(inputs)
        model = keras.Model(inputs, outputs)

        y = model(np.ones((2, 4), "float32"))
        self.assertEqual(y.shape, (2, 4))

        # Variables created inside the trace must stay mutable after it.
        inner = model.layers[1].norm
        inner.weight.assign(np.full((4,), 2.0, "float32"))
        self.assertAllClose(inner.weight.value, np.full((4,), 2.0))

        # They must also look like eagerly created objects to NNX, or its
        # transforms cannot write back into them once tracing is over.
        # The layer's trace state lives on its NNX state object, which is
        # named differently across flax versions.
        self.assertTrue(inner.weight._trace_state.is_valid())
        layer_state = next(
            state
            for state in vars(inner).values()
            if hasattr(state, "_trace_state")
        )
        self.assertTrue(layer_state._trace_state.is_valid())

        # Training exercises `_losses_override` assignment in the jitted
        # train step and end-to-end gradient flow.
        model.compile(optimizer="sgd", loss="mse")
        model.fit(
            np.ones((8, 4), "float32"),
            np.zeros((8, 4), "float32"),
            epochs=1,
            verbose=0,
        )
