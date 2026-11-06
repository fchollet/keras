from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.UpSampling2D")
class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.

    The implementation uses interpolative resizing, given the resize method
    (specified by the `interpolation` argument). Use `interpolation=nearest`
    to repeat the rows and columns of the data.

    Example:

    >>> input_shape = (2, 2, 1, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[ 0  1  2]]
      [[ 3  4  5]]]
     [[[ 6  7  8]]
      [[ 9 10 11]]]]
    >>> y = keras.layers.UpSampling2D(size=(1, 2))(x)
    >>> print(y)
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]]

    Args:
        size: Int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json` (if exists) else `"channels_last"`.
            Defaults to `"channels_last"`.
        interpolation: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
            `"lanczos5"`, `"nearest"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, rows, cols)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, upsampled_rows, upsampled_cols)`
    """

    def __init__(
        self, size=(2, 2), data_format=None, interpolation="nearest", **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.size = argument_validation.standardize_tuple(size, 2, "size")
        self.interpolation = interpolation.lower()
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            height = (
                self.size[0] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[3]
                if input_shape[3] is not None
                else None
            )
            return (input_shape[0], input_shape[1], height, width)
        else:
            height = (
                self.size[0] * input_shape[1]
                if input_shape[1] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        return self._resize_images(
            inputs,
            self.size[0],
            self.size[1],
            self.data_format,
            interpolation=self.interpolation,
        )

    def get_config(self):
        config = {
            "size": self.size,
            "data_format": self.data_format,
            "interpolation": self.interpolation,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _resize_images(
        self,
        x,
        height_factor,
        width_factor,
        data_format,
        interpolation="nearest",
    ):
        """Resizes the images contained in a 4D tensor.

        Args:
            x: Tensor or variable to resize.
            height_factor: Positive integer.
            width_factor: Positive integer.
            data_format: One of `"channels_first"`, `"channels_last"`.
            interpolation: A string, one of `"bicubic"`, `"bilinear"`,
            `"lanczos3"`, `"lanczos5"`, or `"nearest"`.

        Returns:
            A tensor.
        """
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Invalid `data_format` argument: {data_format}")

        if data_format == "channels_first":
            x = backend.ops.numpy.transpose(x, [0, 2, 3, 1])
        # https://github.com/keras-team/keras/issues/294
        # Use `ops.repeat` for `nearest` interpolation to enable XLA
        if interpolation == "nearest":
            x = backend.ops.numpy.repeat(x, height_factor, axis=1)
            x = backend.ops.numpy.repeat(x, width_factor, axis=2)
        else:
            # multiply the height and width factor on each dim
            # by hand (versus using element-wise multiplication
            # by np.array([height_factor, width_factor]) then
            # list-ifying the tensor by calling `.tolist()`)
            # since when running under torchdynamo, `new_shape`
            # will be traced as a symbolic variable (specifically
            # a `FakeTensor`) which does not have a `tolist()` method.
            shape = backend.ops.shape(x)
            new_shape = (
                shape[1] * height_factor,
                shape[2] * width_factor,
            )
            x = ops.image.resize(
                x,
                new_shape,
                data_format="channels_last",
                interpolation=interpolation,
            )
        if data_format == "channels_first":
            x = backend.ops.numpy.transpose(x, [0, 3, 1, 2])

        return x
