from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling


@keras_export(
    [
        "keras.layers.GlobalMaxPooling2D",
        "keras.layers.GlobalMaxPool2D",
    ]
)
class GlobalMaxPooling2D(BaseGlobalPooling):
    """Global max pooling operation for 2D data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, height, weight)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            spatial dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        4D tensor with shape:
        `(batch_size, height, width, channels)`
    - If `data_format='channels_first'`:
        4D tensor with shape:
        `(batch_size, channels, height, width)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, channels)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            4D tensor with shape `(batch_size, 1, 1, channels)`
        - If `data_format="channels_first"`:
            4D tensor with shape `(batch_size, channels, 1, 1)`

    Example:

    >>> x = np.random.rand(2, 4, 5, 3)
    >>> y = keras.layers.GlobalMaxPooling2D()(x)
    >>> y.shape
    (2, 3)
    """

    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super().__init__(
            pool_dimensions=2,
            data_format=data_format,
            keepdims=keepdims,
            **kwargs,
        )

    def call(self, inputs):
        if self.data_format == "channels_last":
            return backend.ops.numpy.max(
                inputs, axis=[1, 2], keepdims=self.keepdims
            )
        return backend.ops.numpy.max(
            inputs, axis=[2, 3], keepdims=self.keepdims
        )
