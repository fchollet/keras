import random as python_random

import numpy as np
import pytest

from keras.src import backend
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.random import seed_generator
from keras.src.utils import rng_utils


class SeedGeneratorTest(testing.TestCase):
    def test_seed_generator_initialization(self):
        gen = seed_generator.SeedGenerator()
        self.assertIsNotNone(gen.state)

        seed = 12345
        gen = seed_generator.SeedGenerator(seed=seed)
        self.assertEqual(ops.convert_to_numpy(gen.state)[0], seed)

        with self.assertRaisesRegex(
            ValueError, "Argument `seed` must be an integer"
        ):
            seed_generator.SeedGenerator(seed="invalid_seed")

    def test_seed_generator_next(self):
        gen = seed_generator.SeedGenerator(seed=42)
        seed1 = ops.convert_to_numpy(gen.next())
        seed2 = ops.convert_to_numpy(gen.next())
        self.assertFalse(np.array_equal(seed1, seed2))

    def test_global_seed_generator(self):
        gen1 = seed_generator.global_seed_generator()
        gen2 = seed_generator.global_seed_generator()
        self.assertEqual(gen1, gen2)

    def test_make_default_seed(self):
        seed1 = seed_generator.make_default_seed()
        seed2 = seed_generator.make_default_seed()
        self.assertNotEqual(seed1, seed2)

    def test_make_default_seed_repeated_after_set_random_seed(self):
        rng_utils.set_random_seed(42)
        seed1 = seed_generator.make_default_seed()
        seed2 = seed_generator.make_default_seed()
        self.assertNotEqual(seed1, seed2)
        rng_utils.set_random_seed(42)
        self.assertEqual(seed_generator.make_default_seed(), seed1)
        self.assertEqual(seed_generator.make_default_seed(), seed2)

    def test_make_default_seed_isolated_from_global_rng_after_set_random_seed(
        self,
    ):
        rng_utils.set_random_seed(42)
        seed1 = seed_generator.make_default_seed()
        seed2 = seed_generator.make_default_seed()

        rng_utils.set_random_seed(42)
        python_random.randint(1, 2**31 - 1)
        np.random.randint(1, 2**31)

        self.assertEqual(seed_generator.make_default_seed(), seed1)
        self.assertEqual(seed_generator.make_default_seed(), seed2)

    def test_seed_generator_dtype(self):
        gen = seed_generator.SeedGenerator(seed=42)
        self.assertEqual(gen.state.dtype, backend.ops.random_seed_dtype())
        seed = gen.next()
        self.assertEqual(gen.state.dtype, backend.ops.random_seed_dtype())
        self.assertEqual(
            standardize_dtype(seed.dtype),
            backend.ops.random_seed_dtype(),
        )

    def test_draw_seed_from_seed_generator(self):
        gen = seed_generator.SeedGenerator(seed=42)
        seed1 = seed_generator.draw_seed(gen)
        self.assertTrue(backend.ops.is_tensor(seed1))

    def test_draw_seed_from_integer(self):
        seed2 = seed_generator.draw_seed(12345)
        self.assertTrue(backend.ops.is_tensor(seed2))
        self.assertEqual(
            standardize_dtype(seed2.dtype),
            backend.ops.random_seed_dtype(),
        )

    def test_draw_seed_from_large_integer(self):
        # Seeds at int32 boundaries must not cause overflow or sign-flip errors.
        # 2**31 is where signed-int32 wraps; 2**32 - 1 is the max uint32 value.
        for s in [2**31 - 1, 2**31, 2**32 - 1]:
            seed = seed_generator.draw_seed(s)
            self.assertTrue(backend.ops.is_tensor(seed))
            self.assertEqual(
                standardize_dtype(seed.dtype),
                backend.ops.random_seed_dtype(),
            )
            # Verify the seed can be consumed by a random op without error.
            res = random.uniform((2, 2), seed=s)
            self.assertEqual(res.shape, (2, 2))

    def test_draw_seed_from_none(self):
        seed3 = seed_generator.draw_seed(None)
        self.assertTrue(backend.ops.is_tensor(seed3))

    def test_draw_seed_invalid(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `seed` must be either an integer"
        ):
            seed_generator.draw_seed("invalid_seed")

    def test_seed_generator_unexpected_kwargs(self):
        with self.assertRaisesRegex(
            ValueError, "Unrecognized keyword arguments"
        ):
            seed_generator.SeedGenerator(invalid_arg="unexpected_value")

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="This test requires the JAX backend"
    )
    def test_jax_tracing_with_global_seed_generator(self):
        import jax

        @jax.jit
        def traced_function():
            return seed_generator.global_seed_generator().next()

        with self.assertRaisesRegex(
            ValueError,
            "When tracing a JAX function, you should only use seeded random",
        ):
            traced_function()

    def test_seed_generator_serialization(self):
        random_generator = seed_generator.SeedGenerator(seed=42, name="sg")
        self.run_class_serialization_test(random_generator)
