from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.UpSampling1D")
class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.

    Repeats each temporal step `size` times along the time axis.

    Example:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras.layers.UpSampling1D(size=2)(x)
    >>> y
    [[[ 0.  1.  2.]
      [ 0.  1.  2.]
      [ 3.  4.  5.]
      [ 3.  4.  5.]]
     [[ 6.  7.  8.]
      [ 6.  7.  8.]
      [ 9. 10. 11.]
      [ 9. 10. 11.]]]

    Args:
        size: Integer. Upsampling factor.

    Input shape:
        3D tensor with shape: `(batch_size, steps, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, upsampled_steps, features)`.
    """

    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(
                f"Expected an integer value for `size`, got {type(size)}."
            )
        if size <= 0:
            raise ValueError(
                "Argument `size` should be a positive integer. "
                f"Received: size={size}"
            )
        self.size = size
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        size = (
            self.size * input_shape[1] if input_shape[1] is not None else None
        )
        return [input_shape[0], size, input_shape[2]]

    def call(self, inputs):
        return backend.ops.numpy.repeat(x=inputs, repeats=self.size, axis=1)

    def get_config(self):
        config = {"size": self.size}
        base_config = super().get_config()
        return {**base_config, **config}
