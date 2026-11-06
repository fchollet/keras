from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.UpSampling3D")
class UpSampling3D(Layer):
    """Upsampling layer for 3D inputs.

    Repeats the 1st, 2nd and 3rd dimensions
    of the data by `size[0]`, `size[1]` and `size[2]` respectively.

    Example:

    >>> input_shape = (2, 1, 2, 1, 3)
    >>> x = np.ones(input_shape)
    >>> y = keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    >>> y.shape
    (2, 2, 4, 2, 3)

    Args:
        size: Int, or tuple of 3 integers.
            The upsampling factors for dim1, dim2 and dim3.
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses
            `image_data_format` value found in your Keras config file at
             `~/.keras/keras.json` (if exists) else `"channels_last"`.
            Defaults to `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, dim1, dim2, dim3, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, dim1, dim2, dim3)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
            channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, upsampled_dim1, upsampled_dim2,
            upsampled_dim3)`
    """

    def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.size = argument_validation.standardize_tuple(size, 3, "size")
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            dim1 = (
                self.size[0] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            dim2 = (
                self.size[1] * input_shape[3]
                if input_shape[3] is not None
                else None
            )
            dim3 = (
                self.size[2] * input_shape[4]
                if input_shape[4] is not None
                else None
            )
            return (input_shape[0], input_shape[1], dim1, dim2, dim3)
        else:
            dim1 = (
                self.size[0] * input_shape[1]
                if input_shape[1] is not None
                else None
            )
            dim2 = (
                self.size[1] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            dim3 = (
                self.size[2] * input_shape[3]
                if input_shape[3] is not None
                else None
            )
            return (input_shape[0], dim1, dim2, dim3, input_shape[4])

    def call(self, inputs):
        return self._resize_volumes(
            inputs, self.size[0], self.size[1], self.size[2], self.data_format
        )

    def get_config(self):
        config = {"size": self.size, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}

    def _resize_volumes(
        self, x, depth_factor, height_factor, width_factor, data_format
    ):
        """Resizes the volume contained in a 5D tensor.

        Args:
            x: Tensor or variable to resize.
            depth_factor: Positive integer.
            height_factor: Positive integer.
            width_factor: Positive integer.
            data_format: One of `"channels_first"`, `"channels_last"`.

        Returns:
            Resized tensor.
        """
        if data_format == "channels_first":
            output = backend.ops.numpy.repeat(x, depth_factor, axis=2)
            output = backend.ops.numpy.repeat(output, height_factor, axis=3)
            output = backend.ops.numpy.repeat(output, width_factor, axis=4)
            return output
        elif data_format == "channels_last":
            output = backend.ops.numpy.repeat(x, depth_factor, axis=1)
            output = backend.ops.numpy.repeat(output, height_factor, axis=2)
            output = backend.ops.numpy.repeat(output, width_factor, axis=3)
            return output
        else:
            raise ValueError(f"Invalid data_format: {data_format}")
