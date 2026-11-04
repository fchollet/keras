from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import set_keras_mask
from keras.src.layers.layer import Layer
from keras.src.saving.serialization_lib import deserialize_keras_object


@keras_export("keras.layers.Masking")
class Masking(Layer):
    """Masks a sequence by using a mask value to skip timesteps.

    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    Example:

    Consider a NumPy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
    lack data for these timesteps. You can:

    - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
    - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
    samples, timesteps, features = 32, 10, 8
    inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
    inputs[:, 3, :] = 0.
    inputs[:, 5, :] = 0.

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0.0))
    model.add(keras.layers.LSTM(32))
    output = model(inputs)
    # The time step 3 and 5 will be skipped from LSTM calculation.
    ```

    Note: in the Keras masking convention, a masked timestep is denoted by
    a mask value of `False`, while a non-masked (i.e. usable) timestep
    is denoted by a mask value of `True`.
    """

    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        # `mask_value` can be a serialized tensor, hence verify it
        if isinstance(mask_value, dict) and mask_value.get("config", None):
            mask_value = deserialize_keras_object(mask_value)
        self.mask_value = mask_value
        self.supports_masking = True

        self._build_at_init()

    def compute_mask(self, inputs, mask=None):
        return ops.any(ops.not_equal(inputs, self.mask_value), axis=-1)

    def call(self, inputs):
        boolean_mask = ops.any(
            ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True
        )
        # Set masked outputs to 0
        outputs = ops.where(boolean_mask, inputs, 0)
        # Compute the mask and outputs simultaneously.
        set_keras_mask(outputs, mask=ops.squeeze(boolean_mask, axis=-1))
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {"mask_value": self.mask_value}
        return {**base_config, **config}
