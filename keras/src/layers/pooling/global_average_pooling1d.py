from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling


@keras_export(
    [
        "keras.layers.GlobalAveragePooling1D",
        "keras.layers.GlobalAvgPool1D",
    ]
)
class GlobalAveragePooling1D(BaseGlobalPooling):
    """Global average pooling operation for temporal data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            temporal dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Call arguments:
        inputs: A 3D tensor.
        mask: Binary tensor of shape `(batch_size, steps)` indicating whether
            a given step should be masked (excluded from the average).
            Fully masked samples return zeros.

    Input shape:

    - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, features)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            3D tensor with shape `(batch_size, 1, features)`
        - If `data_format="channels_first"`:
            3D tensor with shape `(batch_size, features, 1)`

    Example:

    >>> x = np.random.rand(2, 3, 4)
    >>> y = keras.layers.GlobalAveragePooling1D()(x)
    >>> y.shape
    (2, 4)
    """

    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super().__init__(
            pool_dimensions=1,
            data_format=data_format,
            keepdims=keepdims,
            **kwargs,
        )
        self.supports_masking = True

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask_expanded = ops.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            mask_expanded = ops.cast(mask_expanded, "bool")
            inputs = ops.where(mask_expanded, inputs, 0)
            float_mask = ops.cast(mask_expanded, inputs.dtype)
            mask_count = ops.sum(
                float_mask, axis=steps_axis, keepdims=self.keepdims
            )
            mask_count = ops.where(mask_count == 0, 1, mask_count)
            return (
                ops.sum(inputs, axis=steps_axis, keepdims=self.keepdims)
                / mask_count
            )
        else:
            return ops.mean(inputs, axis=steps_axis, keepdims=self.keepdims)

    def compute_mask(self, inputs, mask=None):
        return None
