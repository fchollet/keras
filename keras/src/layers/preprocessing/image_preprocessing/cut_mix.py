from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    base_image_preprocessing_color_example,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.CutMix")
class CutMix(BaseImagePreprocessingLayer):
    """CutMix data augmentation technique.

    CutMix is a data augmentation method where patches are cut and pasted
    between two images in the dataset, while the labels are also mixed
    proportionally to the area of the patches.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    References:
       - [CutMix paper]( https://arxiv.org/abs/1905.04899).

    Args:
        factor: A single float or a tuple of two floats between 0 and 1.
            For each batch, a concentration `alpha` is sampled uniformly
            between the two values (between 0 and `factor` if a single float
            is passed). Each image's mixing weight `lambda`, the fraction of
            the image and of its label that is kept, is then sampled from
            `Beta(alpha, alpha)`. `lambda` always ranges over (0, 1) and has
            mean 0.5 for every `factor`. A smaller factor makes extreme patch
            sizes (patches covering almost none or almost all of the image)
            more likely; a larger factor concentrates the patch sizes around
            half of the image. The CutMix paper uses `alpha = 1` (a uniform
            mixing weight), i.e. `factor=(1.0, 1.0)`. Defaults to 1, which
            samples `alpha` uniformly between 0 and 1 (not the paper's
            setting).
        seed: Integer. Used to create a random seed.

    Example:

    {{base_image_preprocessing_color_example}}
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(self, factor=1.0, seed=None, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.seed = seed
        self.generator = SeedGenerator(seed)

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
            self.channel_axis = -3
        else:
            self.height_axis = -3
            self.width_axis = -2
            self.channel_axis = -1

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.ops.shape(images)
        if len(images_shape) == 3:
            return None

        batch_size = images_shape[0]
        image_height = images_shape[self.height_axis]
        image_width = images_shape[self.width_axis]

        seed = seed or self._get_seed_generator(self.backend._backend)

        mix_weight = self._generate_mix_weight(batch_size, seed)
        ratio = self.backend.ops.numpy.sqrt(1.0 - mix_weight)

        x0, x1 = self._compute_crop_bounds(batch_size, image_width, ratio, seed)
        y0, y1 = self._compute_crop_bounds(
            batch_size, image_height, ratio, seed
        )

        batch_masks, mix_weight = self._generate_batch_mask(
            images_shape,
            (x0, x1, y0, y1),
        )

        permutation_order = self.backend.random.shuffle(
            self.backend.ops.numpy.arange(0, batch_size, dtype="int32"),
            seed=seed,
        )

        return {
            "permutation_order": permutation_order,
            "batch_masks": batch_masks,
            "mix_weight": mix_weight,
        }

    def _generate_batch_mask(self, images_shape, box_corners):
        def _generate_grid_xy(image_height, image_width):
            grid_y, grid_x = self.backend.ops.numpy.meshgrid(
                self.backend.ops.numpy.arange(
                    image_height, dtype=self.compute_dtype
                ),
                self.backend.ops.numpy.arange(
                    image_width, dtype=self.compute_dtype
                ),
                indexing="ij",
            )
            if self.data_format == "channels_last":
                grid_y = self.backend.ops.cast(
                    grid_y[None, :, :, None], dtype=self.compute_dtype
                )
                grid_x = self.backend.ops.cast(
                    grid_x[None, :, :, None], dtype=self.compute_dtype
                )
            else:
                grid_y = self.backend.ops.cast(
                    grid_y[None, None, :, :], dtype=self.compute_dtype
                )
                grid_x = self.backend.ops.cast(
                    grid_x[None, None, :, :], dtype=self.compute_dtype
                )
            return grid_x, grid_y

        image_height, image_width = (
            images_shape[self.height_axis],
            images_shape[self.width_axis],
        )
        grid_x, grid_y = _generate_grid_xy(image_height, image_width)

        x0, x1, y0, y1 = box_corners

        x0 = x0[:, None, None, None]
        y0 = y0[:, None, None, None]
        x1 = x1[:, None, None, None]
        y1 = y1[:, None, None, None]

        batch_masks = (
            (grid_x >= x0) & (grid_x < x1) & (grid_y >= y0) & (grid_y < y1)
        )
        batch_masks = self.backend.ops.numpy.repeat(
            batch_masks, images_shape[self.channel_axis], axis=self.channel_axis
        )
        mix_weight = 1.0 - (x1 - x0) * (y1 - y0) / (image_width * image_height)
        return batch_masks, mix_weight

    def _compute_crop_bounds(self, batch_size, image_length, crop_ratio, seed):
        crop_length = self.backend.ops.cast(
            crop_ratio * image_length, dtype=self.compute_dtype
        )

        start_pos = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=1,
            dtype=self.compute_dtype,
            seed=seed,
        ) * (image_length - crop_length)

        end_pos = start_pos + crop_length

        return start_pos, end_pos

    def _generate_mix_weight(self, batch_size, seed):
        alpha = (
            self.backend.random.uniform(
                shape=(),
                minval=self.factor[0],
                maxval=self.factor[1],
                dtype=self.compute_dtype,
                seed=seed,
            )
            + 1e-6
        )
        mix_weight = self.backend.random.beta(
            (batch_size,), alpha, alpha, seed=seed, dtype=self.compute_dtype
        )
        return mix_weight

    def transform_images(self, images, transformation=None, training=True):
        if training and transformation is not None:
            images = self.backend.ops.cast(images, self.compute_dtype)

            permutation_order = transformation["permutation_order"]
            batch_masks = transformation["batch_masks"]

            images = self.backend.ops.numpy.where(
                batch_masks,
                self.backend.ops.numpy.take(images, permutation_order, axis=0),
                images,
            )
        images = self.backend.ops.cast(images, self.compute_dtype)
        return images

    def transform_labels(self, labels, transformation, training=True):
        if training and transformation is not None:
            permutation_order = transformation["permutation_order"]
            mix_weight = transformation["mix_weight"]

            cutout_labels = self.backend.ops.numpy.take(
                labels, permutation_order, axis=0
            )
            mix_weight = self.backend.ops.numpy.reshape(mix_weight, [-1, 1])
            labels = mix_weight * labels + (1.0 - mix_weight) * cutout_labels

        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        raise NotImplementedError()

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        # CutMix pastes a rectangular region from another sample. For masks
        # this is an exact copy (no blending/interpolation), so keep the
        # original class-index dtype instead of casting to `compute_dtype`.
        if training and transformation is not None:
            segmentation_masks = self.backend.ops.convert_to_tensor(
                segmentation_masks
            )
            permutation_order = transformation["permutation_order"]
            batch_masks = transformation["batch_masks"]
            segmentation_masks = self.backend.ops.numpy.where(
                batch_masks,
                self.backend.ops.numpy.take(
                    segmentation_masks, permutation_order, axis=0
                ),
                segmentation_masks,
            )
        return segmentation_masks

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


if CutMix.__doc__ is not None:
    CutMix.__doc__ = CutMix.__doc__.replace(
        "{{base_image_preprocessing_color_example}}",
        base_image_preprocessing_color_example.replace("{LayerName}", "CutMix"),
    )
