import inspect

import numpy as np

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common import standardize_shape
from keras.src.initializers.constant_initializers import STFT
from keras.src.initializers.constant_initializers import Constant
from keras.src.initializers.constant_initializers import Identity
from keras.src.initializers.constant_initializers import Ones
from keras.src.initializers.constant_initializers import Zeros
from keras.src.initializers.initializer import Initializer
from keras.src.initializers.random_initializers import GlorotNormal
from keras.src.initializers.random_initializers import GlorotUniform
from keras.src.initializers.random_initializers import HeNormal
from keras.src.initializers.random_initializers import HeUniform
from keras.src.initializers.random_initializers import LecunNormal
from keras.src.initializers.random_initializers import LecunUniform
from keras.src.initializers.random_initializers import Orthogonal
from keras.src.initializers.random_initializers import RandomNormal
from keras.src.initializers.random_initializers import RandomUniform
from keras.src.initializers.random_initializers import TruncatedNormal
from keras.src.initializers.random_initializers import VarianceScaling
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

ALL_OBJECTS = {
    Initializer,
    Constant,
    Identity,
    Ones,
    STFT,
    Zeros,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    Orthogonal,
    RandomNormal,
    RandomUniform,
    TruncatedNormal,
    VarianceScaling,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
# Aliases
ALL_OBJECTS_DICT.update(
    {
        "IdentityInitializer": Identity,  # For compatibility
        "normal": RandomNormal,
        "one": Ones,
        "STFTInitializer": STFT,  # For compatibility
        "OrthogonalInitializer": Orthogonal,  # For compatibility
        "uniform": RandomUniform,
        "zero": Zeros,
    }
)


@keras_export("keras.initializers.serialize")
def serialize(initializer):
    """Returns the initializer configuration as a Python dict."""
    return serialization_lib.serialize_keras_object(initializer)


@keras_export("keras.initializers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras initializer object via its configuration."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.initializers.get")
def get(identifier):
    """Retrieves a Keras initializer object via an identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> keras.initializers.get(identifier)
    <...keras.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> keras.initializers.get(cfg)
    <...keras.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    You may also pass a callable function with a signature that includes `shape`
    and `dtype=None` as an identifier.

    >>> fn = lambda shape, dtype=None: ops.ones(shape, dtype)
    >>> keras.initializers.get(fn)
    <function <lambda> at ...>

    Alternatively, you can pass a backend tensor or numpy array as the
    `identifier` to define the initializer values directly. Note that when
    calling the initializer, the specified `shape` argument must be the same as
    the shape of the tensor.

    >>> tensor = ops.ones(shape=(5, 5))
    >>> keras.initializers.get(tensor)
    <function get.<locals>.initialize_fn at ...>

    Args:
        identifier: A string, dict, callable function, or tensor specifying
            the initializer. If a string, it should be the name of an
            initializer. If a dict, it should contain the configuration of an
            initializer. Callable functions or predefined tensors are also
            accepted.

    Returns:
        Initializer instance base on the input identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    elif ops.is_tensor(identifier) or isinstance(
        identifier, (np.generic, np.ndarray)
    ):

        def initialize_fn(shape, dtype=None):
            dtype = standardize_dtype(dtype)
            if standardize_shape(shape) != standardize_shape(identifier.shape):
                raise ValueError(
                    f"Expected `shape` to be {identifier.shape} for direct "
                    f"tensor as initializer. Received shape={shape}"
                )
            return ops.cast(identifier, dtype)

        obj = initialize_fn
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret initializer identifier: {identifier}"
        )
