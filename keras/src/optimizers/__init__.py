import inspect

from keras.src.api_export import keras_export
from keras.src.optimizers.adadelta import Adadelta
from keras.src.optimizers.adafactor import Adafactor
from keras.src.optimizers.adagrad import Adagrad
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.adamax import Adamax
from keras.src.optimizers.adamw import AdamW
from keras.src.optimizers.ftrl import Ftrl
from keras.src.optimizers.lamb import Lamb
from keras.src.optimizers.lion import Lion
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.optimizers.multi_optimizer import MultiOptimizer
from keras.src.optimizers.multi_optimizer import OptimizerMap
from keras.src.optimizers.muon import Muon
from keras.src.optimizers.nadam import Nadam
from keras.src.optimizers.optimizer import Optimizer
from keras.src.optimizers.rmsprop import RMSprop
from keras.src.optimizers.schedule_free_adamw import ScheduleFreeAdamW
from keras.src.optimizers.sgd import SGD
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

ALL_OBJECTS = {
    Optimizer,
    Adam,
    SGD,
    RMSprop,
    Adadelta,
    AdamW,
    Adagrad,
    Adamax,
    Adafactor,
    Muon,
    MultiOptimizer,
    OptimizerMap,
    Nadam,
    Ftrl,
    Lamb,
    Lion,
    LossScaleOptimizer,
    ScheduleFreeAdamW,
}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
ALL_OBJECTS_DICT.update({cls.__name__.lower(): cls for cls in ALL_OBJECTS})


@keras_export("keras.optimizers.serialize")
def serialize(optimizer):
    """Returns the optimizer configuration as a Python dict.

    Args:
        optimizer: An `Optimizer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the optimizer.
    """
    return serialization_lib.serialize_keras_object(optimizer)


@keras_export("keras.optimizers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras optimizer object via its configuration.

    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras Optimizer instance.
    """
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.optimizers.get")
def get(identifier):
    """Retrieves a Keras Optimizer instance.

    Args:
        identifier: Optimizer identifier, one of:
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).

    Returns:
        A Keras Optimizer instance.
    """
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if inspect.isclass(obj):
        obj = obj()
    if isinstance(obj, Optimizer):
        return obj
    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")


# We will add this temporarily so that tensorflow packages that depend on
# estimators will continue to import (there are a large number). Note that
# Keras 3 will not work with the estimators API.
@keras_export(
    [
        "keras.optimizers.legacy.Adagrad",
        "keras.optimizers.legacy.Adam",
        "keras.optimizers.legacy.Ftrl",
        "keras.optimizers.legacy.RMSprop",
        "keras.optimizers.legacy.SGD",
        "keras.optimizers.legacy.Optimizer",
    ]
)
class LegacyOptimizerWarning:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "`keras.optimizers.legacy` is not supported in Keras 3. When using "
            "`tf.keras`, to continue using a `tf.keras.optimizers.legacy` "
            "optimizer, you can install the `tf_keras` package (Keras 2) and "
            "set the environment variable `TF_USE_LEGACY_KERAS=True` to "
            "configure TensorFlow to use `tf_keras` when accessing `tf.keras`."
        )
