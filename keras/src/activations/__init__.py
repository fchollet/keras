from keras.src.activations.activations import celu
from keras.src.activations.activations import elu
from keras.src.activations.activations import exponential
from keras.src.activations.activations import gelu
from keras.src.activations.activations import glu
from keras.src.activations.activations import hard_shrink
from keras.src.activations.activations import hard_sigmoid
from keras.src.activations.activations import hard_silu
from keras.src.activations.activations import hard_tanh
from keras.src.activations.activations import leaky_relu
from keras.src.activations.activations import linear
from keras.src.activations.activations import log_sigmoid
from keras.src.activations.activations import log_softmax
from keras.src.activations.activations import mish
from keras.src.activations.activations import relu
from keras.src.activations.activations import relu6
from keras.src.activations.activations import selu
from keras.src.activations.activations import sigmoid
from keras.src.activations.activations import silu
from keras.src.activations.activations import soft_shrink
from keras.src.activations.activations import softmax
from keras.src.activations.activations import softplus
from keras.src.activations.activations import softsign
from keras.src.activations.activations import sparse_plus
from keras.src.activations.activations import sparse_sigmoid
from keras.src.activations.activations import sparsemax
from keras.src.activations.activations import squareplus
from keras.src.activations.activations import tanh
from keras.src.activations.activations import tanh_shrink
from keras.src.activations.activations import threshold
from keras.src.api_export import keras_export
from keras.src.saving import serialization_lib

ALL_OBJECTS = {
    relu,
    leaky_relu,
    relu6,
    softmax,
    celu,
    elu,
    selu,
    softplus,
    softsign,
    squareplus,
    soft_shrink,
    sparse_plus,
    silu,
    gelu,
    glu,
    tanh,
    tanh_shrink,
    threshold,
    sigmoid,
    sparse_sigmoid,
    exponential,
    hard_sigmoid,
    hard_silu,
    hard_tanh,
    hard_shrink,
    linear,
    mish,
    log_softmax,
    log_sigmoid,
    sparsemax,
}

ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}
# Additional aliases
ALL_OBJECTS_DICT["swish"] = silu
ALL_OBJECTS_DICT["hard_swish"] = hard_silu


@keras_export("keras.activations.serialize")
def serialize(activation):
    if activation in ALL_OBJECTS:
        # Case for keras.activations builtins (simply return name)
        return activation.__name__
    return serialization_lib.serialize_keras_object(activation)


@keras_export("keras.activations.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras activation function via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.activations.get")
def get(identifier):
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:
        return linear
    if isinstance(identifier, dict):
        obj = serialization_lib.deserialize_keras_object(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier
    if callable(obj):
        return obj
    raise ValueError(
        f"Could not interpret activation function identifier: {identifier}"
    )
