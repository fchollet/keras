from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    base_image_preprocessing_transform_example,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils


@keras_export("keras.layers.RandomShear")
class RandomShear(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly applies shear transformations to
    images.

    This layer shears the input images along the x-axis and/or y-axis by a
    randomly selected factor within the specified range. The shear
    transformation is applied to each image independently in a batch. Empty
    regions created during the transformation are filled according to the
    `fill_mode` and `fill_value` parameters.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        x_factor: A tuple of two floats. For each augmented image, a value
            is sampled from the provided range. If a float is passed, the
            range is interpreted as `(0, x_factor)`. Values represent a
            percentage of the image to shear over. For example, 0.3 shears
            pixels up to 30% of the way across the image. All provided values
            should be positive.
        y_factor: A tuple of two floats. For each augmented image, a value
            is sampled from the provided range. If a float is passed, the
            range is interpreted as `(0, y_factor)`. Values represent a
            percentage of the image to shear over. For example, 0.3 shears
            pixels up to 30% of the way across the image. All provided values
            should be positive.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the
                last pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond the edge
                with the same constant value `k` specified by `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
            Note that when using torch backend, `"reflect"` is redirected to
            `"mirror"` `(c d c b | a b c d | c b a b)` because torch does
            not support `"reflect"`.
            Note that torch backend does not support `"wrap"`.
        fill_value: A float representing the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        seed: Integer. Used to create a random seed.

    Example:

    {{base_image_preprocessing_transform_example}}
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)
    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [0, 1.0]. "
    )
    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        x_factor=0.0,
        y_factor=0.0,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.x_factor = x_factor
        self.x_lower, self.x_upper = self._set_factor_with_name(
            x_factor, "x_factor"
        )
        self.y_factor = y_factor
        self.y_lower, self.y_upper = self._set_factor_with_name(
            y_factor, "y_factor"
        )

        if fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.supports_jit = False

    def _set_factor_with_name(self, factor, factor_name):
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
            factor = abs(factor)
            lower, upper = [-factor, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: {factor_name}={factor}"
            )
        return lower, upper

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < 0.0:
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
        if len(images_shape) == 3:
            batch_size = 1
        else:
            batch_size = images_shape[0]

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        invert = self.backend.random.uniform(
            minval=0,
            maxval=1,
            shape=[batch_size, 1],
            seed=seed,
            dtype=self.compute_dtype,
        )
        invert = self.backend.ops.numpy.where(
            invert > 0.5,
            -self.backend.ops.numpy.ones_like(invert),
            self.backend.ops.numpy.ones_like(invert),
        )

        shear_y = self.backend.random.uniform(
            minval=self.y_lower,
            maxval=self.y_upper,
            shape=[batch_size, 1],
            seed=seed,
            dtype=self.compute_dtype,
        )
        shear_x = self.backend.random.uniform(
            minval=self.x_lower,
            maxval=self.x_upper,
            shape=[batch_size, 1],
            seed=seed,
            dtype=self.compute_dtype,
        )
        shear_factor = (
            self.backend.ops.cast(
                self.backend.ops.numpy.concatenate([shear_x, shear_y], axis=1),
                dtype=self.compute_dtype,
            )
            * invert
        )
        return {"shear_factor": shear_factor, "input_shape": images_shape}

    def _transform_images(self, images, transformation, interpolation):
        if transformation is None:
            return images

        inputs_shape = self.backend.ops.shape(images)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            images = self.backend.ops.numpy.expand_dims(images, axis=0)

        shear_factor = transformation["shear_factor"]
        outputs = self.backend.ops.image.affine_transform(
            images,
            transform=self._get_shear_matrix(shear_factor),
            interpolation=interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        if unbatched:
            outputs = self.backend.ops.numpy.squeeze(outputs, axis=0)
        return outputs

    def _get_shear_matrix(self, shear_factors):
        num_shear_factors = self.backend.ops.shape(shear_factors)[0]

        # The shear matrix looks like:
        # [[1   s_x  0]
        #  [s_y  1   0]
        #  [0    0   1]]

        return self.backend.ops.numpy.stack(
            [
                self.backend.ops.numpy.ones((num_shear_factors,)),
                shear_factors[:, 0],
                self.backend.ops.numpy.zeros((num_shear_factors,)),
                shear_factors[:, 1],
                self.backend.ops.numpy.ones((num_shear_factors,)),
                self.backend.ops.numpy.zeros((num_shear_factors,)),
                self.backend.ops.numpy.zeros((num_shear_factors,)),
                self.backend.ops.numpy.zeros((num_shear_factors,)),
            ],
            axis=1,
        )

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def get_transformed_x_y(self, x, y, transform):
        a0, a1, a2, b0, b1, b2, c0, c1 = self.backend.ops.numpy.split(
            transform, 8, axis=-1
        )

        k = c0 * x + c1 * y + 1
        x_transformed = (a0 * x + a1 * y + a2) / k
        y_transformed = (b0 * x + b1 * y + b2) / k
        return x_transformed, y_transformed

    def get_shifted_bbox(self, bounding_boxes, w_shift_factor, h_shift_factor):
        bboxes = bounding_boxes["boxes"]
        x1, x2, x3, x4 = self.backend.ops.numpy.split(bboxes, 4, axis=-1)

        w_shift_factor = self.backend.ops.convert_to_tensor(
            w_shift_factor, dtype=x1.dtype
        )
        h_shift_factor = self.backend.ops.convert_to_tensor(
            h_shift_factor, dtype=x1.dtype
        )

        if len(bboxes.shape) == 3:
            w_shift_factor = self.backend.ops.numpy.expand_dims(
                w_shift_factor, -1
            )
            h_shift_factor = self.backend.ops.numpy.expand_dims(
                h_shift_factor, -1
            )

        bounding_boxes["boxes"] = self.backend.ops.numpy.concatenate(
            [
                x1 - w_shift_factor,
                x2 - h_shift_factor,
                x3 - w_shift_factor,
                x4 - h_shift_factor,
            ],
            axis=-1,
        )
        return bounding_boxes

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        def _get_height_width(transformation):
            if self.data_format == "channels_first":
                height_axis = -2
                width_axis = -1
            else:
                height_axis = -3
                width_axis = -2
            input_height, input_width = (
                transformation["input_shape"][height_axis],
                transformation["input_shape"][width_axis],
            )
            return input_height, input_width

        if training:
            if backend_utils.in_tf_graph():
                self.backend.set_backend("tensorflow")

            input_height, input_width = _get_height_width(transformation)

            bounding_boxes = convert_format(
                bounding_boxes,
                source=self.bounding_box_format,
                target="rel_xyxy",
                height=input_height,
                width=input_width,
                dtype=self.compute_dtype,
            )

            bounding_boxes = self._shear_bboxes(bounding_boxes, transformation)

            bounding_boxes = clip_to_image_size(
                bounding_boxes=bounding_boxes,
                height=input_height,
                width=input_width,
                bounding_box_format="rel_xyxy",
            )

            bounding_boxes = convert_format(
                bounding_boxes,
                source="rel_xyxy",
                target=self.bounding_box_format,
                height=input_height,
                width=input_width,
                dtype=self.compute_dtype,
            )

            self.backend.reset()

        return bounding_boxes

    def _shear_bboxes(self, bounding_boxes, transformation):
        shear_factor = self.backend.ops.cast(
            transformation["shear_factor"], dtype=self.compute_dtype
        )
        shear_x_amount, shear_y_amount = self.backend.ops.numpy.split(
            shear_factor, 2, axis=-1
        )

        x1, y1, x2, y2 = self.backend.ops.numpy.split(
            bounding_boxes["boxes"], 4, axis=-1
        )
        x1 = self.backend.ops.numpy.squeeze(x1, axis=-1)
        y1 = self.backend.ops.numpy.squeeze(y1, axis=-1)
        x2 = self.backend.ops.numpy.squeeze(x2, axis=-1)
        y2 = self.backend.ops.numpy.squeeze(y2, axis=-1)

        if shear_x_amount is not None:
            x1_top = x1 - (shear_x_amount * y1)
            x1_bottom = x1 - (shear_x_amount * y2)
            x1 = self.backend.ops.numpy.where(
                shear_x_amount < 0, x1_top, x1_bottom
            )

            x2_top = x2 - (shear_x_amount * y1)
            x2_bottom = x2 - (shear_x_amount * y2)
            x2 = self.backend.ops.numpy.where(
                shear_x_amount < 0, x2_bottom, x2_top
            )

        if shear_y_amount is not None:
            y1_left = y1 - (shear_y_amount * x1)
            y1_right = y1 - (shear_y_amount * x2)
            y1 = self.backend.ops.numpy.where(
                shear_y_amount > 0, y1_right, y1_left
            )

            y2_left = y2 - (shear_y_amount * x1)
            y2_right = y2 - (shear_y_amount * x2)
            y2 = self.backend.ops.numpy.where(
                shear_y_amount > 0, y2_left, y2_right
            )

        boxes = self.backend.ops.numpy.concatenate(
            [
                self.backend.ops.numpy.expand_dims(x1, axis=-1),
                self.backend.ops.numpy.expand_dims(y1, axis=-1),
                self.backend.ops.numpy.expand_dims(x2, axis=-1),
                self.backend.ops.numpy.expand_dims(y2, axis=-1),
            ],
            axis=-1,
        )
        bounding_boxes["boxes"] = boxes

        return bounding_boxes

    def get_config(self):
        base_config = super().get_config()
        config = {
            "x_factor": self.x_factor,
            "y_factor": self.y_factor,
            "fill_mode": self.fill_mode,
            "interpolation": self.interpolation,
            "seed": self.seed,
            "fill_value": self.fill_value,
            "data_format": self.data_format,
        }
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


if RandomShear.__doc__ is not None:
    RandomShear.__doc__ = RandomShear.__doc__.replace(
        "{{base_image_preprocessing_transform_example}}",
        base_image_preprocessing_transform_example.replace(
            "{LayerName}", "RandomShear"
        ),
    )
