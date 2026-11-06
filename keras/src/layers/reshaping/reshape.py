from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.ops import operation_utils
from keras.src.ops.operation_utils import validate_reshape_shape


@keras_export("keras.layers.Reshape")
class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

    Args:
        target_shape: Target shape. Tuple of integers, does not include the
            samples dimension (batch size). One element of the `target_shape`
            can be -1 in which case the missing value is inferred from the
            size of the array and remaining dimensions.

    Input shape:
        Arbitrary, but required to be compatible with `target_shape`.

    Output shape:
        `(batch_size, *target_shape)`

    Example:

    >>> x = keras.Input(shape=(12,))
    >>> y = keras.layers.Reshape((3, 4))(x)
    >>> y.shape
    (None, 3, 4)

    >>> # another example with shape inference using `-1` as dimension
    >>> y = keras.layers.Reshape((-1, 2, 2))(x)
    >>> y.shape
    (None, 3, 2, 2)
    """

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        target_shape = tuple(target_shape)
        validate_reshape_shape(target_shape, newshape_arg_name="target_shape")
        self.target_shape = target_shape
        self.built = True

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            *operation_utils.compute_reshape_output_shape(
                input_shape[1:], self.target_shape, "target_shape"
            ),
        )

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            shape=output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def call(self, inputs):
        potentially_resolved_target_shape = (
            operation_utils.compute_reshape_output_shape(
                tuple(inputs.shape)[1:], self.target_shape, "target_shape"
            )
        )
        potentially_resolved_target_shape = tuple(
            -1 if d is None else d for d in potentially_resolved_target_shape
        )
        return backend.ops.numpy.reshape(
            inputs,
            (backend.ops.shape(inputs)[0],) + potentially_resolved_target_shape,
        )

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
