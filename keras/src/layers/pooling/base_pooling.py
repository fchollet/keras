from keras.src import backend
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.ops.operation_utils import compute_pooling_output_shape
from keras.src.utils import argument_validation


class BasePooling(Layer):
    """Base pooling layer."""

    def __init__(
        self,
        pool_size,
        strides,
        pool_dimensions,
        pool_mode="max",
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.pool_size = argument_validation.standardize_tuple(
            pool_size, pool_dimensions, "pool_size"
        )
        strides = pool_size if strides is None else strides
        self.strides = argument_validation.standardize_tuple(
            strides, pool_dimensions, "strides", allow_zero=True
        )
        self.pool_mode = pool_mode
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

        self.input_spec = InputSpec(ndim=pool_dimensions + 2)

        self._build_at_init()

    def call(self, inputs):
        if self.pool_mode == "max":
            return backend.ops.nn.max_pool(
                inputs,
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        elif self.pool_mode == "average":
            return backend.ops.nn.average_pool(
                inputs,
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        else:
            raise ValueError(
                "`pool_mode` must be either 'max' or 'average'. Received: "
                f"{self.pool_mode}."
            )

    def compute_output_shape(self, input_shape):
        return compute_pooling_output_shape(
            input_shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "padding": self.padding,
                "strides": self.strides,
                "data_format": self.data_format,
            }
        )
        return config
