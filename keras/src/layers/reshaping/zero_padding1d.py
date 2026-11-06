from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.ZeroPadding1D")
class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Example:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras.layers.ZeroPadding1D(padding=2)(x)
    >>> y
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]]

    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int: how many zeros to add at the beginning and end of
              the padding dimension (axis 1).
            - If tuple of 2 ints: how many zeros to add at the beginning and the
              end of the padding dimension (`(left_pad, right_pad)`).
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, axis_to_pad, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, axis_to_pad)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, axis_to_pad, features)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, features, axis_to_pad)`

    Output shape:
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, padded_axis, features)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, features, padded_axis)`
    """

    def __init__(self, padding=1, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.padding = argument_validation.standardize_tuple(
            padding, 2, "padding", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        padding_dim = 2 if self.data_format == "channels_first" else 1
        if output_shape[padding_dim] is not None:
            output_shape[padding_dim] += self.padding[0] + self.padding[1]
        return tuple(output_shape)

    def call(self, inputs):
        if self.data_format == "channels_first":
            all_dims_padding = ((0, 0), (0, 0), self.padding)
        else:
            all_dims_padding = ((0, 0), self.padding, (0, 0))
        return backend.ops.numpy.pad(inputs, all_dims_padding)

    def get_config(self):
        config = {"padding": self.padding, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
