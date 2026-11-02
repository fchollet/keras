from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.merging.base_merge import Merge


@keras_export("keras.layers.Average")
class Average(Merge):
    """Averages a list of inputs element-wise..

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Average()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.average([x1, x2])`
    >>> y = keras.layers.Average()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = backend.ops.numpy.add(output, inputs[i])
        return output / len(inputs)


@keras_export("keras.layers.average")
def average(inputs, **kwargs):
    """Functional interface to the `keras.layers.Average` layer.

    Args:
        inputs: A list of input tensors , all of the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the element-wise product of the inputs with the same
        shape as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.average([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> y = keras.layers.average([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Average(**kwargs)(inputs)
