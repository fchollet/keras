import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.random import random
from keras.src.random import seed_generator
from keras.src.testing.test_utils import named_product
from keras.src.utils.rng_utils import set_random_seed


class RandomCorrectnessTest(testing.TestCase):
    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
    )
    def test_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.normal(shape, mean=mean, stddev=stddev, seed=seed)
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "minval": 0, "maxval": 1},
        {"seed": 10, "shape": (2, 3), "minval": 0, "maxval": 1},
        {"seed": 10, "shape": (2, 3, 4), "minval": 0, "maxval": 2},
        {"seed": 10, "shape": (2, 3), "minval": -1, "maxval": 1},
        {"seed": 10, "shape": (2, 3), "minval": 1, "maxval": 3},
    )
    def test_uniform(self, seed, shape, minval, maxval):
        np.random.seed(seed)
        np_res = np.random.uniform(low=minval, high=maxval, size=shape)
        res = random.uniform(shape, minval=minval, maxval=maxval, seed=seed)
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), maxval)
        self.assertGreaterEqual(ops.max(res), minval)

    @parameterized.parameters(
        {"seed": 10, "num_samples": 1, "batch_size": 1},
        {"seed": 10, "num_samples": 5, "batch_size": 2},
        {"seed": 10, "num_samples": 10, "batch_size": 4},
        {"seed": 10, "num_samples": 15, "batch_size": 8},
    )
    def test_categorical(self, seed, num_samples, batch_size):
        np.random.seed(seed)
        # Create logits that definitely favors the batch index after a softmax
        # is applied. Without a softmax, this would be close to random.
        logits = np.eye(batch_size) * 1e5 + 1e6
        res = random.categorical(logits, num_samples, seed=seed)
        # Outputs should have shape `(batch_size, num_samples)`, where each
        # output index matches the batch index.
        self.assertEqual(res.shape, (batch_size, num_samples))
        expected = np.tile(np.arange(batch_size)[:, None], (1, num_samples))
        self.assertAllClose(res, expected)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "min": 0, "max": 10, "dtype": "uint16"},
        {"seed": 10, "shape": (2, 3), "min": 0, "max": 10, "dtype": "uint32"},
        {"seed": 10, "shape": (2, 3, 4), "min": 0, "max": 2, "dtype": "int8"},
        {"seed": 10, "shape": (2, 3), "min": -1, "max": 1, "dtype": "int16"},
        {"seed": 10, "shape": (2, 3), "min": 1, "max": 3, "dtype": "int32"},
    )
    def test_randint(self, seed, shape, min, max, dtype):
        np.random.seed(seed)
        np_res = np.random.randint(low=min, high=max, size=shape)
        res = random.randint(
            shape, minval=min, maxval=max, seed=seed, dtype=dtype
        )
        self.assertEqual(res.shape, shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), max)
        self.assertGreaterEqual(ops.max(res), min)
        # Torch has incomplete dtype support for uints; will remap some dtypes.
        if keras.backend.backend() != "torch":
            self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.parameters(
        {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
        {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
        # Test list shapes.
        {"seed": 10, "shape": [2, 3], "mean": 10, "stddev": 3},
    )
    def test_truncated_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.truncated_normal(
            shape, mean=mean, stddev=stddev, seed=seed
        )
        self.assertEqual(res.shape, tuple(shape))
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), mean + 2 * stddev)
        self.assertGreaterEqual(ops.max(res), mean - 2 * stddev)

    def test_dropout(self):
        x = ops.ones((10, 10))
        self.assertAllClose(random.dropout(x, rate=0, seed=0), x)
        x_res = random.dropout(x, rate=0.5, seed=0)
        self.assertGreater(ops.max(x_res), ops.max(x))
        self.assertAllClose(ops.max(x_res), 2.0)
        self.assertGreater(ops.cast(ops.sum(x_res == 0), "int32"), 2)
        x_res = random.dropout(x, rate=1.0, seed=0)
        self.assertAllClose(x_res, ops.zeros((10, 10)))

    def test_dropout_noise_shape(self):
        inputs = ops.ones((2, 3, 5, 7))
        x = random.dropout(
            inputs, rate=0.3, noise_shape=[None, 3, 5, None], seed=0
        )
        self.assertEqual(x.shape, (2, 3, 5, 7))

    def test_global_seed_generator(self):
        # Check that unseeded RNG calls use and update global_rng_state()

        def random_numbers(seed):
            rng_state = seed_generator.global_seed_generator().state
            rng_state.assign(seed)
            x = random.normal((), seed=None)
            y = random.normal((), seed=None)
            return x, y, rng_state.value

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            random_numbers = tf.function(jit_compile=True)(random_numbers)

        seed = ops.zeros((2,))
        seed0 = ops.convert_to_numpy(seed)
        x1, y1, seed = random_numbers(seed)
        x1 = ops.convert_to_numpy(x1)
        y1 = ops.convert_to_numpy(y1)
        seed1 = ops.convert_to_numpy(seed)
        x2, y2, seed = random_numbers(seed)
        x2 = ops.convert_to_numpy(x2)
        y2 = ops.convert_to_numpy(y2)
        seed2 = ops.convert_to_numpy(seed)
        x3, y3, seed = random_numbers(seed)
        x3 = ops.convert_to_numpy(x3)
        y3 = ops.convert_to_numpy(y3)
        seed3 = ops.convert_to_numpy(seed)

        self.assertNotEqual(seed0[1], seed1[1])
        self.assertNotEqual(seed1[1], seed2[1])
        self.assertNotEqual(seed2[1], seed3[1])

        self.assertGreater(np.abs(x1 - y1), 1e-4)
        self.assertGreater(np.abs(x1 - y1), 1e-4)
        self.assertGreater(np.abs(x2 - y2), 1e-4)
        self.assertGreater(np.abs(x3 - y3), 1e-4)
        self.assertGreater(np.abs(x1 - x2), 1e-4)
        self.assertGreater(np.abs(x1 - x3), 1e-4)
        self.assertGreater(np.abs(x2 - x3), 1e-4)
        self.assertGreater(np.abs(y1 - y2), 1e-4)
        self.assertGreater(np.abs(y1 - y3), 1e-4)
        self.assertGreater(np.abs(y2 - y3), 1e-4)

        seed_generator.global_seed_generator().state.assign(seed)

    def test_shuffle(self):
        x = np.arange(100).reshape(10, 10)

        # Test axis=0
        y = random.shuffle(x, seed=0)

        self.assertFalse(np.all(x == ops.convert_to_numpy(y)))
        self.assertAllClose(np.sum(x, axis=0), ops.sum(y, axis=0))
        self.assertNotAllClose(np.sum(x, axis=1), ops.sum(y, axis=1))

        # Test axis=1
        y = random.shuffle(x, axis=1, seed=0)

        self.assertFalse(np.all(x == ops.convert_to_numpy(y)))
        self.assertAllClose(np.sum(x, axis=1), ops.sum(y, axis=1))
        self.assertNotAllClose(np.sum(x, axis=0), ops.sum(y, axis=0))

        # Test axis=-1
        y = random.shuffle(x, axis=-1, seed=0)

        self.assertFalse(np.all(x == ops.convert_to_numpy(y)))
        self.assertAllClose(np.sum(x, axis=-1), ops.sum(y, axis=-1))
        self.assertNotAllClose(np.sum(x, axis=0), ops.sum(y, axis=0))

    @parameterized.parameters(
        {"seed": 10, "shape": (5, 2), "alpha": 2.0, "dtype": "float16"},
        {"seed": 10, "shape": (2,), "alpha": 1.5, "dtype": "float32"},
        {"seed": 10, "shape": (2, 3), "alpha": 0.5, "dtype": "float32"},
    )
    def test_gamma(self, seed, shape, alpha, dtype):
        values = random.gamma(shape, alpha=alpha, seed=seed, dtype=dtype)
        self.assertEqual(ops.shape(values), shape)
        self.assertEqual(standardize_dtype(values.dtype), dtype)
        self.assertGreater(np.min(ops.convert_to_numpy(values)), 0.0)

    @parameterized.parameters(
        {
            "seed": 10,
            "shape": (5, 2),
            "counts": 5e4,
            "probabilities": 0.5,
            "dtype": "float16",
        },
        {
            "seed": 10,
            "shape": (2,),
            "counts": 1e5,
            "probabilities": 0.5,
            "dtype": "float32",
        },
        {
            "seed": 10,
            "shape": (2, 3),
            "counts": [[1e5, 2e5, 3e5], [4e5, 5e5, 6e5]],
            "probabilities": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "dtype": "float32",
        },
    )
    def test_binomial(self, seed, shape, counts, probabilities, dtype):
        set_random_seed(1337)
        values = random.binomial(
            shape=shape,
            counts=counts,
            probabilities=probabilities,
            seed=seed,
            dtype=dtype,
        )
        self.assertEqual(ops.shape(values), shape)
        self.assertEqual(standardize_dtype(values.dtype), dtype)

        # The following test that ensures that the number of time
        # each event occurs doesn't exceed the total input count specified
        # by the user for that event.
        # Hence, we do an element wise comparison between `counts` array
        # and the (generated) `values` array.
        values_np = ops.convert_to_numpy(values)
        self.assertTrue(np.greater_equal(np.array(counts), values_np).all())

        # Following test computes the probabilities of each event
        # by dividing number of times an event occurs (which is the generated
        # value) by the corresponding value in the (total) counts array.
        # and then makes sure that the computed probabilities approximate
        # the input probabilities
        generated_probabilities = values_np / np.array(counts)
        probabilities = np.ones(shape) * np.array(probabilities)
        self.assertAllClose(
            probabilities, generated_probabilities, rtol=0.005, atol=0.005
        )

    @parameterized.parameters(
        {
            "seed": 10,
            "shape": (10000,),
            "alpha": 3.0,
            "beta": 2.0,
            "dtype": "float16",
        },
        {
            "seed": 10,
            "shape": (10000, 3),
            "alpha": [[7.0, 0.5, 1.5]],
            "beta": [[15.0, 0.9, 4.5]],
            "dtype": "float32",
        },
        {
            "seed": 10,
            "shape": (10000, 30),
            "alpha": 1.0,
            "beta": 1.0,
            "dtype": "float32",
        },
    )
    def test_beta(self, seed, shape, alpha, beta, dtype):
        set_random_seed(1337)
        values = random.beta(
            shape=shape, alpha=alpha, beta=beta, seed=seed, dtype=dtype
        )
        self.assertEqual(ops.shape(values), shape)
        self.assertEqual(standardize_dtype(values.dtype), dtype)
        values_np = ops.convert_to_numpy(values)
        self.assertGreaterEqual(np.min(values_np), b=0.0)
        self.assertLessEqual(np.max(values_np), b=1.0)

        _alpha_is_an_array = False
        if isinstance(alpha, list):
            alpha = np.array(alpha)
            beta = np.array(beta)
            _alpha_is_an_array = True

        # Mean check:
        # For a beta distributed random variable,
        # mean = alpha / (alpha + beta)
        expected_mean = alpha / (alpha + beta)

        if _alpha_is_an_array:
            actual_mean = np.mean(values_np, axis=0)
            self.assertAllClose(
                expected_mean.flatten(), actual_mean, atol=0.005, rtol=0.005
            )
        else:
            actual_mean = np.mean(values_np.flatten())
            self.assertAlmostEqual(actual_mean, expected_mean, decimal=2)

        # Variance check:
        # For a beta distributed random variable,
        # variance = (alpha * beta) / ((alpha + beta)^2)(alpha + beta + 1)
        expected_variance = (alpha * beta) / (
            np.square(alpha + beta) * (alpha + beta + 1)
        )
        if _alpha_is_an_array:
            actual_variance = np.var(values_np, axis=0)
            self.assertAllClose(
                expected_variance.flatten(),
                actual_variance,
                atol=0.005,
                rtol=0.005,
            )
        else:
            actual_variance = np.var(values_np.flatten())
            self.assertAlmostEqual(
                actual_variance, expected_variance, decimal=2
            )


class RandomBehaviorTest(testing.TestCase):
    def test_beta_tf_data_compatibility(self):
        import tensorflow as tf

        from keras.src.layers.preprocessing.data_layer import DataLayer
        from keras.src.random.seed_generator import SeedGenerator

        class BetaLayer(DataLayer):
            def __init__(self, seed=None, **kwargs):
                super().__init__(**kwargs)
                self.seed = seed
                self.generator = SeedGenerator(seed)

            def compute_output_shape(self, input_shape):
                return input_shape

            def call(self, inputs):
                seed_generator = self._get_seed_generator(self.backend._backend)
                noise = self.backend.random.beta(
                    self.backend.ops.shape(inputs),
                    alpha=0.5,
                    beta=0.5,
                    seed=seed_generator,
                )
                inputs = inputs + noise
                return inputs

        layer = BetaLayer()
        input_data = np.random.random([2, 4, 4, 3])
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            self.assertEqual(output.shape, (2, 4, 4, 3))

    def test_categorical_errors(self):
        with self.assertRaises(ValueError):
            random.categorical(np.ones((5,)), 5)
        with self.assertRaises(ValueError):
            random.categorical(np.ones((5, 5, 5)), 5)

    def test_randint_dtype_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`keras.random.randint` requires an integer `dtype`."
        ):
            random.randint((3, 4), minval=0, maxval=10, dtype="float64")

    def test_uniform_dtype_validation(self):
        with self.assertRaisesRegex(
            ValueError,
            "`keras.random.uniform` requires a floating point `dtype`.",
        ):
            random.uniform((3, 4), minval=0, maxval=10, dtype="int64")

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_dropout_jax_jit_stateless(self):
        import jax
        import jax.numpy as jnp

        x = ops.ones(3)

        @jax.jit
        def train_step(x):
            with keras.src.backend.StatelessScope():
                x = keras.layers.Dropout(rate=0.1)(x, training=True)
            return x

        x = train_step(x)
        self.assertIsInstance(x, jnp.ndarray)

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_jax_rngkey_seed(self):
        import jax
        import jax.numpy as jnp

        seed = 1234
        rng = jax.random.PRNGKey(seed)
        self.assertEqual(rng.shape, (2,))
        self.assertEqual(rng.dtype, jnp.uint32)
        x = random.randint((3, 5), 0, 10, seed=rng)
        self.assertIsInstance(x, jnp.ndarray)

    @pytest.mark.skipif(
        keras.backend.backend() != "jax",
        reason="This test requires `jax` as the backend.",
    )
    def test_jax_unseed_disallowed_during_tracing(self):
        import jax

        @jax.jit
        def jit_fn():
            return random.randint((2, 2), 0, 10, seed=None)

        with self.assertRaisesRegex(
            ValueError, "you should only use seeded random ops"
        ):
            jit_fn()

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="This test requires `tensorflow` as the backend.",
    )
    def test_tf_cast_seed(self):
        import tensorflow as tf

        inputs = tf.ones([2, 3], dtype="float32")
        seed = tf.int32.max + 1000  # Test floormod operation
        outputs_mod = random.categorical(inputs, 2, seed=seed)
        outputs_nomod = random.categorical(inputs, 2, seed=1001)
        self.assertAllClose(outputs_mod, outputs_nomod)


class RandomDTypeTest(testing.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]
    if backend.backend() == "torch":
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_normal(self, dtype):
        res = random.normal((2, 3), dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=INT_DTYPES))
    def test_categorical(self, dtype):
        logits = np.eye(4) * 1e5 + 1e6
        res = random.categorical(logits, 10, dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_uniform(self, dtype):
        res = random.uniform((2, 3), dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=INT_DTYPES))
    def test_randint(self, dtype):
        res = random.randint((2, 3), 0, 10, dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_truncated_normal(self, dtype):
        res = random.truncated_normal((2, 3), dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_dropout(self, dtype):
        x = ops.ones((3, 5), dtype=dtype)
        res = random.dropout(x, rate=0.8, seed=0)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_gamma(self, dtype):
        res = random.gamma((2, 3), 2.0, dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_binomial(self, dtype):
        res = random.binomial((2,), 1e5, 0.5, dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_beta(self, dtype):
        res = random.beta((2, 3), 2.0, 3.0, dtype=dtype)
        self.assertEqual(standardize_dtype(res.dtype), dtype)
