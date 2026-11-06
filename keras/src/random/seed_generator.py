import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
from keras.src.utils.naming import auto_name

GLOBAL_SEED_GENERATOR = "global_seed_generator"


@keras_export("keras.random.SeedGenerator")
class SeedGenerator:
    """Generates variable seeds upon each call to a function generating
    random numbers.

    In Keras, all random number generators (such as
    `keras.random.normal()`) are stateless, meaning that if you pass an
    integer seed to them (such as `seed=42`), they will return the same
    values for repeated calls. To get different values for each
    call, a `SeedGenerator` providing the state of the random generator
    has to be used.

    Note that all the random number generators have a default seed of None,
    which implies that an internal global SeedGenerator is used.
    If you need to decouple the RNG from the global state you can provide
    a local `StateGenerator` with either a deterministic or random initial
    state.

    Remark concerning the JAX backend: Note that the use of a local
    `StateGenerator` as seed argument is required for JIT compilation of
    RNG with the JAX backend, because the use of global state is not
    supported.

    Example:

    ```python
    seed_gen = keras.random.SeedGenerator(seed=42)
    values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    ```

    Usage in a layer:

    ```python
    class Dropout(keras.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, x, training=False):
            if training:
                return keras.random.dropout(
                    x, rate=0.5, seed=self.seed_generator
                )
            return x
    ```
    """

    def __init__(self, seed=None, name=None, **kwargs):
        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name

        custom_backend = kwargs.pop("backend", None)
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")
        if custom_backend is not None:
            self.backend = custom_backend
        else:
            self.backend = backend

        self._initial_seed = seed
        if seed is None:
            seed = make_default_seed()

        if not isinstance(seed, int):
            raise ValueError(
                f"Argument `seed` must be an integer. Received: seed={seed}"
            )

        def seed_initializer(*args, **kwargs):
            dtype = kwargs.get("dtype", None)
            return self.backend.ops.convert_to_tensor([seed, 0], dtype=dtype)

        with self.backend.name_scope(self.name, caller=self):
            self.state = self.backend.Variable(
                seed_initializer,
                shape=(2,),
                dtype=self.backend.ops.random_seed_dtype(),
                trainable=False,
                aggregation="none",
                name="seed_generator_state",
            )

    def next(self, ordered=True):
        seed_state = self.state
        # Use * 1 to create a copy
        new_seed_value = seed_state.value * 1
        if ordered:
            increment = self.backend.ops.convert_to_tensor(
                np.array([0, 1]), dtype=seed_state.dtype
            )
            self.state.assign(self.backend.ops.numpy.add(seed_state, increment))
        else:
            # This produces a sequence of near-unique numbers
            # between 0 and 1M
            self.state.assign((seed_state + 1) * 5387 % 933199)
        return new_seed_value

    def get_config(self):
        return {"seed": self._initial_seed, "name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def global_seed_generator():
    if jax_utils.is_in_jax_tracing_scope():
        raise ValueError(
            "[JAX RNG] When tracing a JAX function, "
            "you should only use seeded random ops, e.g. "
            "you should create a `SeedGenerator` instance, attach it "
            "to your layer/model, and pass the instance as the `seed` "
            "argument when calling random ops. Unseeded random ops "
            "would get incorrectly traced by JAX and would become constant "
            "after tracing. Example:\n\n"
            "```\n"
            "# Make sure to set the seed generator as a layer attribute\n"
            "self.seed_generator = keras.random.SeedGenerator(seed=1337)\n"
            "...\n"
            "out = keras.random.normal(shape=(1,), seed=self.seed_generator)\n"
            "```"
        )
    gen = global_state.get_global_attribute(GLOBAL_SEED_GENERATOR)
    if gen is None:
        gen = SeedGenerator()
        global_state.set_global_attribute(GLOBAL_SEED_GENERATOR, gen)
    return gen


def make_default_seed():
    """Return the default integer seed used by random initializers."""
    from keras.src.utils import rng_utils

    # Keep implicit initializer seeds deterministic after
    # `keras.utils.set_random_seed()` calls.
    return rng_utils.consume_default_initializer_seed()


def draw_seed(seed):
    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        dtype = backend.ops.random_seed_dtype()
        # Seeds are conceptually uint32 values but some backends declare
        # their seed dtype as a signed type (e.g. "int32"). np.array(x,
        # dtype="int32") raises OverflowError on NumPy >= 1.24 for values
        # >= 2**31. Perform an explicit 2's-complement bit-cast via uint32
        # so that the integer passed to convert_to_tensor is always in the
        # representable range of the declared dtype while preserving full
        # 32-bit entropy.
        if dtype == "int32":
            # Re-interpret the bits of a uint32 as an int32.
            seed = (seed & 0xFFFFFFFF ^ 0x80000000) - 0x80000000
        return backend.ops.convert_to_tensor([seed, 0], dtype=dtype)
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(
        "Argument `seed` must be either an integer "
        "or an instance of `SeedGenerator`. "
        f"Received: seed={seed} (of type {type(seed)})"
    )
