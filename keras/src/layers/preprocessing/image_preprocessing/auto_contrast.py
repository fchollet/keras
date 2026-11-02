from keras.src.api_export import keras_export
from keras.src.backend.common import is_int_dtype
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.ops.core import _saturate_cast


@keras_export("keras.layers.AutoContrast")
class AutoContrast(BaseImagePreprocessingLayer):
    """Performs the auto-contrast operation on an image.

    Auto contrast stretches the values of an image across the entire available
    `value_range`. This makes differences between pixels more obvious. An
    example of this is if an image only has values `[0, 1]` out of the range
    `[0, 255]`, auto contrast will change the `1` values to be `255`.

    This layer is active at both training and inference time.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        value_range: Range of values the incoming images will have.
            Represented as a two number tuple written `(low, high)`.
            This is typically either `(0, 1)` or `(0, 255)` depending
            on how your preprocessing pipeline is set up.
            Defaults to `(0, 255)`.
    """

    _USE_BASE_FACTOR = False
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    def __init__(
        self,
        value_range=(0, 255),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._set_value_range(value_range)

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        self.value_range = sorted(value_range)

    def transform_images(self, images, transformation=None, training=True):
        original_images = images
        images = self._transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )

        images = self.backend.ops.cast(images, self.compute_dtype)
        low = self.backend.ops.numpy.min(images, axis=(1, 2), keepdims=True)
        high = self.backend.ops.numpy.max(images, axis=(1, 2), keepdims=True)
        scale = 255.0 / (high - low)
        offset = -low * scale

        images = images * scale + offset
        results = self.backend.ops.numpy.clip(images, 0.0, 255.0)
        results = self._transform_value_range(
            results,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        # don't process NaN channels
        results = self.backend.ops.numpy.where(
            self.backend.ops.numpy.isnan(results), original_images, results
        )
        if results.dtype == images.dtype:
            return results
        if is_int_dtype(images.dtype):
            results = self.backend.ops.numpy.round(results)
        return _saturate_cast(results, images.dtype, self.backend)

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
