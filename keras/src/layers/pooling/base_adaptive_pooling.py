"""Base classes for adaptive pooling layers."""

from keras.src import backend
from keras.src.backend import config
from keras.src.layers.layer import Layer


class BaseAdaptivePooling(Layer):
    """Base class shared by all adaptive pooling layers."""

    def __init__(self, output_size, data_format=None, **kwargs):
        """Initialize base adaptive pooling layer.

        Args:
            output_size: Normalized spatial output size as a tuple
                (for example, (32,), (32, 32), or (32, 32, 32)).
            data_format: Either "channels_last" or "channels_first".
            **kwargs: Additional layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.output_size = output_size
        self.data_format = data_format or config.image_data_format()
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"Invalid data_format: {self.data_format}. "
                "Expected 'channels_first' or 'channels_last'."
            )

    def compute_output_shape(self, input_shape):
        """Return the output shape tensor after pooling."""
        batch_size = input_shape[0]
        if self.data_format == "channels_last":
            channels = input_shape[-1]
            return (batch_size, *self.output_size, channels)
        else:
            channels = input_shape[1]
            return (batch_size, channels, *self.output_size)

    def get_config(self):
        config_dict = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config_dict}


class BaseAdaptiveAveragePooling(BaseAdaptivePooling):
    """Base class for adaptive average pooling in 1D, 2D, and 3D."""

    def call(self, inputs):
        return backend.ops.nn.adaptive_average_pool(
            inputs, output_size=self.output_size, data_format=self.data_format
        )


class BaseAdaptiveMaxPooling(BaseAdaptivePooling):
    """Base class for adaptive max pooling in 1D, 2D, and 3D."""

    def call(self, inputs):
        return backend.ops.nn.adaptive_max_pool(
            inputs, output_size=self.output_size, data_format=self.data_format
        )
