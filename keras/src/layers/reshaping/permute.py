from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Permute")
class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful e.g. connecting RNNs and convnets.

    Args:
        dims: Tuple of integers. Permutation pattern does not include the
            batch dimension. Indexing starts at 1.
            For instance, `(1, 3, 2)` permutes the second and third dimensions
            of the input.

    Input shape:
        Arbitrary.

    Output shape:
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.

    Example:

    >>> x = keras.Input(shape=(10, 64))
    >>> y = keras.layers.Permute((2, 1))(x)
    >>> y.shape
    (None, 64, 10)
    """

    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)
        self.dims = tuple(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError(
                "Invalid permutation argument `dims` for Permute Layer. "
                "The set of indices in `dims` must be consecutive and start "
                f"from 1. Received dims={dims}"
            )
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        for dim in self.dims:
            output_shape.append(input_shape[dim])
        return tuple(output_shape)

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            shape=output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def call(self, inputs):
        return backend.ops.numpy.transpose(inputs, axes=(0,) + self.dims)

    def get_config(self):
        config = {"dims": self.dims}
        base_config = super().get_config()
        return {**base_config, **config}
