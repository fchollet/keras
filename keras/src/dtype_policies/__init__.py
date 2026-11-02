from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.dtype_policies import dtype_policy
from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy
from keras.src.dtype_policies.dtype_policy import DTypePolicy
from keras.src.dtype_policies.dtype_policy import FloatDTypePolicy
from keras.src.dtype_policies.dtype_policy import GPTQDTypePolicy
from keras.src.dtype_policies.dtype_policy import Int4DTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedFloat8DTypePolicy
from keras.src.dtype_policies.dtype_policy import (
    _get_quantized_dtype_policy_by_str,
)
from keras.src.dtype_policies.dtype_policy import _is_quantized_policy_string
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap

ALL_OBJECTS = {
    AWQDTypePolicy,
    DTypePolicy,
    FloatDTypePolicy,
    QuantizedDTypePolicy,
    QuantizedFloat8DTypePolicy,
    DTypePolicyMap,
    GPTQDTypePolicy,
    Int4DTypePolicy,
}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


@keras_export("keras.dtype_policies.serialize")
def serialize(dtype_policy):
    """Serializes `DTypePolicy` instance.

    Args:
        dtype_policy: A Keras `DTypePolicy` instance.

    Returns:
        `DTypePolicy` configuration dictionary.
    """
    from keras.src.saving import serialization_lib

    return serialization_lib.serialize_keras_object(dtype_policy)


@keras_export("keras.dtype_policies.deserialize")
def deserialize(config, custom_objects=None):
    """Deserializes a serialized `DTypePolicy` instance.

    Args:
        config: `DTypePolicy` configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras `DTypePolicy` instance.
    """
    from keras.src.saving import serialization_lib

    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.dtype_policies.get")
def get(identifier):
    """Retrieves a Keras `DTypePolicy` instance.

    The `identifier` may be the string name of a `DTypePolicy` class.

    >>> policy = dtype_policies.get("mixed_bfloat16")
    >>> type(policy)
    <class '...DTypePolicy'>

    You can also specify `config` of the dtype policy to this function by
    passing dict containing `class_name` and `config` as an identifier. Also
    note that the `class_name` must map to a `DTypePolicy` class

    >>> identifier = {"class_name": "DTypePolicy",
    ...               "config": {"name": "float32"}}
    >>> policy = dtype_policies.get(identifier)
    >>> type(policy)
    <class '...DTypePolicy'>

    Args:
        identifier: A dtype policy identifier. One of `None` or string name of a
            `DTypePolicy` or `DTypePolicy` configuration dictionary or a
            `DTypePolicy` instance.

    Returns:
        A Keras `DTypePolicy` instance.
    """
    if identifier is None:
        return dtype_policy.dtype_policy()
    if isinstance(identifier, DTypePolicy):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        if _is_quantized_policy_string(identifier):
            return _get_quantized_dtype_policy_by_str(identifier)
        else:
            return DTypePolicy(identifier)
    try:
        return DTypePolicy(standardize_dtype(identifier))
    except:
        raise ValueError(
            "Cannot interpret `dtype` argument. Expected a string "
            f"or an instance of DTypePolicy. Received: dtype={identifier}"
        )
