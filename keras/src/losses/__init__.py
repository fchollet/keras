import inspect

from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.losses import CTC
from keras.src.losses.losses import BinaryCrossentropy
from keras.src.losses.losses import BinaryFocalCrossentropy
from keras.src.losses.losses import CategoricalCrossentropy
from keras.src.losses.losses import CategoricalFocalCrossentropy
from keras.src.losses.losses import CategoricalGeneralizedCrossEntropy
from keras.src.losses.losses import CategoricalHinge
from keras.src.losses.losses import Circle
from keras.src.losses.losses import CosineSimilarity
from keras.src.losses.losses import Dice
from keras.src.losses.losses import Hinge
from keras.src.losses.losses import Huber
from keras.src.losses.losses import KLDivergence
from keras.src.losses.losses import LogCosh
from keras.src.losses.losses import LossFunctionWrapper
from keras.src.losses.losses import MeanAbsoluteError
from keras.src.losses.losses import MeanAbsolutePercentageError
from keras.src.losses.losses import MeanSquaredError
from keras.src.losses.losses import MeanSquaredLogarithmicError
from keras.src.losses.losses import Poisson
from keras.src.losses.losses import SparseCategoricalCrossentropy
from keras.src.losses.losses import SparseCategoricalFocalCrossentropy
from keras.src.losses.losses import SquaredHinge
from keras.src.losses.losses import Tversky
from keras.src.losses.losses import binary_crossentropy
from keras.src.losses.losses import binary_focal_crossentropy
from keras.src.losses.losses import categorical_crossentropy
from keras.src.losses.losses import categorical_focal_crossentropy
from keras.src.losses.losses import categorical_generalized_cross_entropy
from keras.src.losses.losses import categorical_hinge
from keras.src.losses.losses import circle
from keras.src.losses.losses import cosine_similarity
from keras.src.losses.losses import ctc
from keras.src.losses.losses import dice
from keras.src.losses.losses import hinge
from keras.src.losses.losses import huber
from keras.src.losses.losses import kl_divergence
from keras.src.losses.losses import log_cosh
from keras.src.losses.losses import mean_absolute_error
from keras.src.losses.losses import mean_absolute_percentage_error
from keras.src.losses.losses import mean_squared_error
from keras.src.losses.losses import mean_squared_logarithmic_error
from keras.src.losses.losses import poisson
from keras.src.losses.losses import sparse_categorical_crossentropy
from keras.src.losses.losses import sparse_categorical_focal_crossentropy
from keras.src.losses.losses import squared_hinge
from keras.src.losses.losses import tversky
from keras.src.saving import serialization_lib

ALL_OBJECTS = {
    # Base
    Loss,
    LossFunctionWrapper,
    # Probabilistic
    KLDivergence,
    Poisson,
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
    CategoricalFocalCrossentropy,
    CategoricalGeneralizedCrossEntropy,
    SparseCategoricalCrossentropy,
    SparseCategoricalFocalCrossentropy,
    # Regression
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    CosineSimilarity,
    LogCosh,
    Huber,
    # Hinge
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    # Image segmentation
    Dice,
    Tversky,
    # Similarity
    Circle,
    # Sequence
    CTC,
    # Probabilistic
    kl_divergence,
    poisson,
    binary_crossentropy,
    binary_focal_crossentropy,
    categorical_crossentropy,
    categorical_focal_crossentropy,
    categorical_generalized_cross_entropy,
    sparse_categorical_crossentropy,
    sparse_categorical_focal_crossentropy,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    cosine_similarity,
    log_cosh,
    huber,
    # Hinge
    hinge,
    squared_hinge,
    categorical_hinge,
    # Image segmentation
    dice,
    tversky,
    # Similarity
    circle,
    # Sequence
    ctc,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {
        "bce": binary_crossentropy,
        "BCE": binary_crossentropy,
        "kld": kl_divergence,
        "KLD": kl_divergence,
        "mae": mean_absolute_error,
        "MAE": mean_absolute_error,
        "mse": mean_squared_error,
        "MSE": mean_squared_error,
        "mape": mean_absolute_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "msle": mean_squared_logarithmic_error,
        "MSLE": mean_squared_logarithmic_error,
    }
)


@keras_export("keras.losses.serialize")
def serialize(loss):
    """Serializes loss function or `Loss` instance.

    Args:
        loss: A Keras `Loss` instance or a loss function.

    Returns:
        Loss configuration dictionary.
    """
    return serialization_lib.serialize_keras_object(loss)


@keras_export("keras.losses.deserialize")
def deserialize(name, custom_objects=None):
    """Deserializes a serialized loss class/function instance.

    Args:
        name: Loss configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras `Loss` instance or a loss function.
    """
    return serialization_lib.deserialize_keras_object(
        name,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.losses.get")
def get(identifier):
    """Retrieves a Keras loss as a `function`/`Loss` class instance.

    The `identifier` may be the string name of a loss function or `Loss` class.

    >>> loss = losses.get("categorical_crossentropy")
    >>> type(loss)
    <class 'function'>
    >>> loss = losses.get("CategoricalCrossentropy")
    >>> type(loss)
    <class '...CategoricalCrossentropy'>

    You can also specify `config` of the loss to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Loss` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> loss = losses.get(identifier)
    >>> type(loss)
    <class '...CategoricalCrossentropy'>

    Args:
        identifier: A loss identifier. One of None or string name of a loss
            function/class or loss configuration dictionary or a loss function
            or a loss class instance.

    Returns:
        A Keras loss as a `function`/ `Loss` class instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret loss identifier: {identifier}")
