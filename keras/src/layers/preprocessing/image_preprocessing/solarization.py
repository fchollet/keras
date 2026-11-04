from keras.src.api_export import keras_export
from keras.src.backend.common import is_int_dtype
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.ops.core import _saturate_cast
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.Solarization")
class Solarization(BaseImagePreprocessingLayer):
    """Applies `(max_value - pixel + min_value)` for each pixel in the image.

    When created without `threshold` parameter, the layer performs solarization
    to all values. When created with specified `threshold` the layer only
    augments pixels that are above the `threshold` value.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        addition_factor: (Optional)  A tuple of two floats or a single float,
            between 0 and 1.
            For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, addition_factor)`. If specified, this value
            (times the value range of input images, e.g. 255), is
            added to each pixel before solarization and thresholding.
            Defaults to 0.0.
        threshold_factor: (Optional)  A tuple of two floats or a single float.
            For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, threshold_factor)`. If specified, only pixel
            values above this threshold will be solarized.
        value_range: a tuple or a list of two elements. The first value
            represents the lower bound for values in input images, the second
            represents the upper bound. Images passed to the layer should have
            values within `value_range`. Typical values to pass
            are `(0, 255)` (RGB image) or `(0., 1.)` (scaled image).
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:

    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    print(images[0, 0, 0])
    # [59 62 63]
    # Note that images are Tensor with values in the range [0, 255]
    solarization = Solarization(value_range=(0, 255))
    images = solarization(images)
    print(images[0, 0, 0])
    # [196, 193, 192]
    ```
    """

    _USE_BASE_FACTOR = False
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )
    _FACTOR_VALIDATION_ERROR = (
        "The `addition_factor` and `threshold_factor` arguments "
        "should be a number (or a list of two numbers) "
        "in the range [0, 1]. "
    )

    def __init__(
        self,
        addition_factor=0.0,
        threshold_factor=0.0,
        value_range=(0, 255),
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.addition_factor = self._set_factor(
            addition_factor, "addition_factor"
        )
        self.threshold_factor = self._set_factor(
            threshold_factor, "threshold_factor"
        )
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

    def _set_factor(self, factor, factor_name):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR
                    + f"Received: {factor_name}={factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            lower, upper = [0, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: {factor_name}={factor}"
            )
        return lower, upper

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < 0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: input_number={input_number}"
            )

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        images_shape = self.backend.ops.shape(images)
        if len(images_shape) == 4:
            factor_shape = (images_shape[0], 1, 1, 1)
        else:
            factor_shape = (1, 1, 1)

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        return {
            "additions": self.backend.random.uniform(
                minval=self.addition_factor[0],
                maxval=self.addition_factor[1] * 255,
                shape=factor_shape,
                seed=seed,
                dtype=self.compute_dtype,
            ),
            "thresholds": self.backend.random.uniform(
                minval=self.threshold_factor[0],
                maxval=self.threshold_factor[1] * 255,
                shape=factor_shape,
                seed=seed,
                dtype=self.compute_dtype,
            ),
        }

    def transform_images(self, images, transformation, training=True):
        images = self.backend.ops.cast(images, self.compute_dtype)

        if training:
            if transformation is None:
                return images

            thresholds = transformation["thresholds"]
            additions = transformation["additions"]
            images = self._transform_value_range(
                images,
                original_range=self.value_range,
                target_range=(0, 255),
                dtype=self.compute_dtype,
            )
            results = images + additions
            results = self.backend.ops.numpy.clip(results, 0, 255)
            results = self.backend.ops.numpy.where(
                results < thresholds, results, 255 - results
            )
            results = self._transform_value_range(
                results,
                original_range=(0, 255),
                target_range=self.value_range,
                dtype=self.compute_dtype,
            )
            if results.dtype == images.dtype:
                return results
            if is_int_dtype(images.dtype):
                results = self.backend.ops.numpy.round(results)
            return _saturate_cast(results, images.dtype, self.backend)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def get_config(self):
        base_config = super().get_config()
        config = {
            "value_range": self.value_range,
            "addition_factor": self.addition_factor,
            "threshold_factor": self.threshold_factor,
            "seed": self.seed,
        }
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
