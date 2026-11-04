from keras.src import activations
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.layers.layer import Layer


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


@keras_export("keras.layers.Softmax")
class Softmax(Layer):
    """Softmax activation layer.

    Formula:
    ``` python
    exp_x = exp(x - max(x))
    f(x) = exp_x / sum(exp_x)
    ```

    Example:
    >>> softmax_layer = keras.layers.Softmax()
    >>> input = np.array([1.0, 2.0, 1.0])
    >>> result = softmax_layer(input)
    >>> result
    [0.21194157, 0.5761169, 0.21194157]


    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Call arguments:
        inputs: The inputs (logits) to the softmax layer.
        mask: A boolean mask that is broadcastable to `inputs`. The mask
            specifies 1 to keep and 0 to mask. Each dimension of the mask
            must either be 1 or match the corresponding dimension of
            `inputs`; it must not be larger. Defaults to `None`.

    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

        self._build_at_init()

    def call(self, inputs, mask=None):
        if mask is not None:
            if len(mask.shape) > len(inputs.shape):
                raise ValueError(
                    "The `mask` must be broadcastable to `inputs` "
                    "and must not have more dimensions. "
                    f"Received: inputs.shape={inputs.shape}, "
                    f"mask.shape={mask.shape}"
                )
            for m_dim, i_dim in zip(mask.shape[::-1], inputs.shape[::-1]):
                if m_dim is not None and i_dim is not None:
                    if m_dim != 1 and m_dim != i_dim:
                        raise ValueError(
                            "The `mask` must be broadcastable to "
                            "`inputs`. Each mask dimension must be 1 "
                            "or match the corresponding input "
                            "dimension. Received: "
                            f"inputs.shape={inputs.shape}, "
                            f"mask.shape={mask.shape}"
                        )
            # We keep the positions where the mask is True or > 0.5, and set the
            # other (masked) positions to -1e.9.
            if standardize_dtype(mask.dtype) != "bool":
                mask = backend.ops.numpy.greater(
                    mask, backend.ops.cast(0.5, dtype=mask.dtype)
                )
            inputs = backend.ops.numpy.where(
                mask, inputs, _large_negative_number(inputs.dtype)
            )
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                outputs = backend.ops.numpy.exp(
                    inputs
                    - backend.ops.math.logsumexp(
                        inputs, axis=self.axis, keepdims=True
                    )
                )
            else:
                outputs = activations.softmax(inputs, axis=self.axis[0])
        else:
            outputs = activations.softmax(inputs, axis=self.axis)

        # Free pre-softmax masked inputs to reduce peak memory.
        # Without this, the masked inputs, softmax outputs, and
        # post-masked outputs all exist simultaneously.
        del inputs

        if mask is not None:
            # Zero out masked positions in case the entire axis is masked
            # (where softmax would output a uniform distribution).
            outputs = backend.ops.numpy.where(mask, outputs, 0.0)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
