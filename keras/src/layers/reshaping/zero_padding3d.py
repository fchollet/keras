from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.ZeroPadding3D")
class ZeroPadding3D(Layer):
    """Zero-padding layer for 3D data (spatial or spatio-temporal).

    Example:

    >>> input_shape = (1, 1, 2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.ZeroPadding3D(padding=2)(x)
    >>> y.shape
    (1, 5, 6, 6, 3)

    Args:
        padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric padding is applied to depth, height,
              and width.
            - If tuple of 3 ints: interpreted as three different symmetric
              padding values for depth, height, and width:
              `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
            - If tuple of 3 tuples of 2 ints: interpreted as
              `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
              right_dim2_pad), (left_dim3_pad, right_dim3_pad))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad, depth)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_padded_axis, second_padded_axis,
          third_axis_to_pad, depth)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad)`
    """

    def __init__(
        self, padding=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = (
                (padding, padding),
                (padding, padding),
                (padding, padding),
            )
        elif hasattr(padding, "__len__"):
            if len(padding) != 3:
                raise ValueError(
                    f"`padding` should have 3 elements. Received: {padding}."
                )
            dim1_padding = argument_validation.standardize_tuple(
                padding[0], 2, "1st entry of padding", allow_zero=True
            )
            dim2_padding = argument_validation.standardize_tuple(
                padding[1], 2, "2nd entry of padding", allow_zero=True
            )
            dim3_padding = argument_validation.standardize_tuple(
                padding[2], 2, "3rd entry of padding", allow_zero=True
            )
            self.padding = (dim1_padding, dim2_padding, dim3_padding)
        else:
            raise ValueError(
                "`padding` should be either an int, a tuple of 3 ints "
                "(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), "
                "or a tuple of 3 tuples of 2 ints "
                "((left_dim1_pad, right_dim1_pad),"
                " (left_dim2_pad, right_dim2_pad),"
                " (left_dim3_pad, right_dim2_pad)). "
                f"Received: padding={padding}."
            )
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        spatial_dims_offset = 2 if self.data_format == "channels_first" else 1
        for index in range(0, 3):
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
