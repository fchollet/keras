import math

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Flatten")
class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Note: If inputs are shaped `(batch,)` without a feature axis, then
    flattening adds an extra channel dimension and output shape is `(batch, 1)`.

    Args:
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Example:

    >>> x = keras.Input(shape=(10, 64))
    >>> y = keras.layers.Flatten()(x)
    >>> y.shape
    (None, 640)
    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)
        self._channels_first = self.data_format == "channels_first"

    def call(self, inputs):
        input_shape = backend.ops.shape(inputs)
        rank = len(input_shape)

        if self._channels_first and rank > 1:
            # Switch to channels-last format.
            inputs = backend.ops.numpy.transpose(
                inputs, axes=(0, *range(2, rank), 1)
            )

        non_batch_dims = input_shape[1:]
        if len(non_batch_dims) == 0:
            flattened_dim = 1
        elif any(not isinstance(d, int) for d in non_batch_dims):
            flattened_dim = -1
        else:
            flattened_dim = math.prod(non_batch_dims)

        return backend.ops.numpy.reshape(
            inputs, (input_shape[0], flattened_dim)
        )

    def compute_output_shape(self, input_shape):
        non_batch_dims = input_shape[1:]
        if len(non_batch_dims) == 0:
            flattened_dim = 1
        elif any(d is None for d in non_batch_dims):
            # NB: we cannot use the shorter `None in non_batch_dims` here b/c
            # torchdynamo errors when calling `__contains__` op with
            # a constant (in this case `None`) operand since it assumes
            # that the elements in the collection are also `ConstantVariable`s
            # but tensor shapes can be `SymNodeVariable`s (e.g. `SymInt`)
            flattened_dim = None
        else:
            flattened_dim = math.prod(non_batch_dims)
        return (input_shape[0], flattened_dim)

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            shape=output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def get_config(self):
        config = {"data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
