from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.ZeroPadding2D")
class ZeroPadding2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of zeros at the top, bottom, left and
    right side of an image tensor.

    Example:

    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[[0 1]
       [2 3]]]]
    >>> y = keras.layers.ZeroPadding2D(padding=1)(x)
    >>> y
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]]

    Args:
        padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding is applied to height and width.
            - If tuple of 2 ints: interpreted as two different symmetric padding
              values for height and width:
              `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints: interpreted as
             `((top_pad, bottom_pad), (left_pad, right_pad))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, height, width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, padded_height, padded_width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_height, padded_width)`
    """

    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(
                    "`padding` should have two elements. "
                    f"Received: padding={padding}."
                )
            height_padding = argument_validation.standardize_tuple(
                padding[0], 2, "1st entry of padding", allow_zero=True
            )
            width_padding = argument_validation.standardize_tuple(
                padding[1], 2, "2nd entry of padding", allow_zero=True
            )
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError(
                "`padding` should be either an int, a tuple of 2 ints "
                "(symmetric_height_crop, symmetric_width_crop), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_crop, bottom_crop), (left_crop, right_crop)). "
                f"Received: padding={padding}."
            )
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        spatial_dims_offset = 2 if self.data_format == "channels_first" else 1
        for index in range(0, 2):
            if output_shape[index + spatial_dims_offset] is not None:
                output_shape[index + spatial_dims_offset] += (
                    self.padding[index][0] + self.padding[index][1]
                )
        return tuple(output_shape)

    def call(self, inputs):
        if self.data_format == "channels_first":
            all_dims_padding = ((0, 0), (0, 0), *self.padding)
        else:
            all_dims_padding = ((0, 0), *self.padding, (0, 0))
        return backend.ops.numpy.pad(inputs, all_dims_padding)

    def get_config(self):
        config = {"padding": self.padding, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
