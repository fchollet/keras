from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.UnitNormalization")
class UnitNormalization(Layer):
    """Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:

    >>> data = np.arange(6).reshape(2, 3)
    >>> normalized_data = keras.layers.UnitNormalization()(data)
    >>> np.sum(normalized_data[0, :] ** 2)
    1.0

    Args:
        axis: Integer or list/tuple. The axis or axes to normalize across.
            Typically, this is the features axis or axes. The left-out axes are
            typically the batch axis or axes. `-1` is the last dimension
            in the input. Defaults to `-1`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Invalid value for `axis` argument: "
                "expected an int or a list/tuple of ints. "
                f"Received: axis={axis}"
            )
        self.supports_masking = True

        self._build_at_init()

    def call(self, inputs):
        # torch compilation cannot trace the `TrackedList` that `self.axis`
        # becomes, so pass a plain tuple to avoid a graph break
        axis = tuple(self.axis) if isinstance(self.axis, list) else self.axis
        return ops.normalize(inputs, axis=axis, order=2, epsilon=1e-12)

    def compute_output_shape(self, input_shape):
        # Ensure axis is always treated as a list
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = self.axis

        for axis in axes:
            if axis >= len(input_shape) or axis < -len(input_shape):
                raise ValueError(
                    f"Axis {self.axis} is out of bounds for "
                    f"input shape {input_shape}."
                )
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
