from keras.src.api_export import keras_export
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.Mosaic")
class Mosaic(BaseImagePreprocessingLayer):
    """Mosaic implements the Mosaic data augmentation technique.

    For every image in the batch, Mosaic picks 3 other images at random from
    the same batch and arranges the 4 images in a 2x2 grid. A crop the size of
    a single image is then taken from the grid around a randomly sampled mosaic
    center, so that each output image contains a piece of each of the 4 images.

    Labels are mixed in proportion to the area that each of the 4 images covers
    in the output. Bounding boxes are moved along with their images and clipped
    to the output image; boxes that end up entirely outside of it get a label
    of `-1`. Segmentation masks are cropped exactly like the images.

    A single (unbatched) image is processed as a batch of one image. All 4
    images of its grid are then that same image, so the output is the image
    shifted with wrap-around. To combine different images, call the layer on
    batches, for example after `tf.data.Dataset.batch()`.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Input shape:
        A 3D (unbatched) or 4D (batched) tensor of images, or a dictionary
        with an `"images"` key and optional `"labels"`, `"bounding_boxes"`
        and `"segmentation_masks"` keys. Batched images have shape
        `(batch, height, width, channels)` in `"channels_last"` format, or
        `(batch, channels, height, width)` in `"channels_first"` format.
        Labels have shape `(batch, num_classes)`, or `(num_classes,)` for a
        single image, and segmentation masks have the same layout as the
        images.

    Output shape:
        Same as the input, except that each image's bounding boxes become
        4 times as many, since an output image can contain boxes from 4
        images.

    References:
        - [YOLOv4 paper](https://arxiv.org/abs/2004.10934).
        - [YOLOv5 implementation](https://github.com/ultralytics/yolov5).
        - [YOLOX implementation](https://github.com/Megvii-BaseDetection/YOLOX).

    Args:
        offset: A float or a tuple of two floats in the range `[0, 1]`.
            Controls where the mosaic center can be placed, as a fraction of
            the image width and height measured from the top-left corner. If a
            tuple `(lower, upper)` is passed, the x and y coordinates of the
            center are each sampled uniformly between `lower` and `upper`. If a
            single float is passed, they are sampled between `0.0` and that
            value. To always use the same center, pass two identical values,
            for example `(0.5, 0.5)`. Defaults to `(0.25, 0.75)`.
        seed: Integer. Used to create a random seed.
        bounding_box_format: A string specifying the format of the bounding
            boxes, for example `"xyxy"`. Required if bounding boxes are passed
            to the layer. Defaults to `None`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:

    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    images, labels = images[:8], labels[:8]
    labels = keras.ops.cast(keras.ops.one_hot(labels.flatten(), 10), "float32")
    mosaic = keras.layers.Mosaic(seed=42)
    output = mosaic({"images": images, "labels": labels})
    ```
    """

    _USE_BASE_FACTOR = False
    _OFFSET_BOUNDS = (0.0, 1.0)

    def __init__(
        self,
        offset=(0.25, 0.75),
        seed=None,
        bounding_box_format=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(
            data_format=data_format,
            bounding_box_format=bounding_box_format,
            **kwargs,
        )
        self.offset = self._parse_offset(offset)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.interpolation = "nearest"

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
            self.channel_axis = -3
        else:
            self.height_axis = -3
            self.width_axis = -2
            self.channel_axis = -1

    def _parse_offset(self, offset):
        error_msg = (
            "The `offset` argument should be a number "
            "(or a list of two numbers) "
            "in the range "
            f"[{self._OFFSET_BOUNDS[0]}, {self._OFFSET_BOUNDS[1]}]. "
            f"Received: offset={offset}"
        )

        if isinstance(offset, (tuple, list)):
            if len(offset) != 2:
                raise ValueError(
                    "If a tuple is passed to `offset`, it must be of length 2."
                )

            if (
                offset[0] < self._OFFSET_BOUNDS[0]
                or offset[1] > self._OFFSET_BOUNDS[1]
            ):
                raise ValueError(error_msg)

            if offset[0] > offset[1]:
                raise ValueError(
                    "The first value of the `offset` tuple must be less than "
                    f"or equal to the second value. Received: offset={offset}"
                )
            return tuple(offset)

        elif isinstance(offset, (float, int)):
            if (
                offset < self._OFFSET_BOUNDS[0]
                or offset > self._OFFSET_BOUNDS[1]
            ):
                raise ValueError(error_msg)
            return (self._OFFSET_BOUNDS[0], float(offset))
        else:
            raise ValueError(
                "The `offset` argument must be either a float or a tuple of "
                f"two floats. Received: offset={offset}"
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

        seed = seed or self._get_seed_generator(self.backend._backend)

        permutation_order = self.backend.random.randint(
            (batch_size, 3), minval=0, maxval=batch_size, seed=seed
        )

        permutation_order = self.backend.ops.numpy.concatenate(
            [
                self.backend.ops.numpy.expand_dims(
                    self.backend.ops.numpy.arange(batch_size), axis=-1
                ),
                permutation_order,
            ],
            axis=-1,
        )

        # Centers are always float32: float16 can't represent fractions like
        # `k / 640` exactly, which would move them off the pixel grid.
        mosaic_centers_x = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.offset[0],
            maxval=self.offset[1],
            seed=seed,
            dtype="float32",
        )

        mosaic_centers_y = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.offset[0],
            maxval=self.offset[1],
            seed=seed,
            dtype="float32",
        )

        # Snap the centers to whole pixels so that images, labels and bounding
        # boxes all use exactly the same split. They stay stored as fractions.
        input_height = images_shape[self.height_axis]
        input_width = images_shape[self.width_axis]
        mosaic_centers_x = (
            self.backend.ops.numpy.round(mosaic_centers_x * input_width)
            / input_width
        )
        mosaic_centers_y = (
            self.backend.ops.numpy.round(mosaic_centers_y * input_height)
            / input_height
        )

        mosaic_centers = self.backend.ops.numpy.stack(
            [mosaic_centers_x, mosaic_centers_y], axis=-1
        )

        return {
            "permutation_order": permutation_order,
            "mosaic_centers": mosaic_centers,
            "input_shape": images_shape,
        }

    def transform_labels(self, labels, transformation, training=True):
        if not training or transformation is None:
            return labels

        labels_shape = self.backend.ops.shape(labels)
        if len(labels_shape) != 2 or not is_float_dtype(labels.dtype):
            raise ValueError(
                "Mosaic mixes labels in proportion to the area of each image, "
                "so `labels` must be one-hot encoded (or soft) float labels of "
                "shape `(batch_size, num_classes)`. Received: labels.shape="
                f"{labels_shape}, "
                f"labels.dtype={standardize_dtype(labels.dtype)}"
            )

        permutation_order = transformation["permutation_order"]

        labels = self.backend.ops.cast(labels, dtype=self.compute_dtype)
        labels_for_mosaic = self.backend.ops.numpy.take(
            labels, permutation_order, axis=0
        )

        mosaic_centers = self.backend.ops.cast(
            transformation["mosaic_centers"], dtype=self.compute_dtype
        )
        center_x = mosaic_centers[..., 0]
        center_y = mosaic_centers[..., 1]

        top_left_ratio = center_x * center_y
        top_right_ratio = (1 - center_x) * center_y
        bottom_left_ratio = center_x * (1 - center_y)
        bottom_right_ratio = (1 - center_x) * (1 - center_y)

        labels = (
            labels_for_mosaic[:, 0] * top_left_ratio[:, None]
            + labels_for_mosaic[:, 1] * top_right_ratio[:, None]
            + labels_for_mosaic[:, 2] * bottom_left_ratio[:, None]
            + labels_for_mosaic[:, 3] * bottom_right_ratio[:, None]
        )

        return labels

    def _transform_images(self, images, transformation, interpolation):
        image_shape = self.backend.ops.shape(images)
        input_height = image_shape[self.height_axis]
        input_width = image_shape[self.width_axis]

        permutation_order = transformation["permutation_order"]
        mosaic_images = self.backend.ops.numpy.take(
            images, permutation_order, axis=0
        )

        tops = self.backend.ops.numpy.concatenate(
            [mosaic_images[:, 0], mosaic_images[:, 1]], axis=self.width_axis
        )
        bottoms = self.backend.ops.numpy.concatenate(
            [mosaic_images[:, 2], mosaic_images[:, 3]], axis=self.width_axis
        )
        grid = self.backend.ops.numpy.concatenate(
            [tops, bottoms], axis=self.height_axis
        )

        mosaic_centers = transformation["mosaic_centers"]
        mosaic_centers_x = self.backend.ops.cast(
            self.backend.ops.numpy.round(mosaic_centers[..., 0] * input_width),
            "int32",
        )
        mosaic_centers_y = self.backend.ops.cast(
            self.backend.ops.numpy.round(mosaic_centers[..., 1] * input_height),
            "int32",
        )

        start_x = input_width - mosaic_centers_x
        start_y = input_height - mosaic_centers_y

        rows = (
            start_y[:, None]
            + self.backend.ops.numpy.arange(input_height, dtype="int32")[
                None, :
            ]
        )
        cols = (
            start_x[:, None]
            + self.backend.ops.numpy.arange(input_width, dtype="int32")[None, :]
        )

        # take_along_axis needs indices with the same rank as the grid, with
        # size 1 on every axis except the one being indexed.
        if self.data_format == "channels_first":
            row_indices = rows[:, None, :, None]  # (B, 1, H, 1)
            col_indices = cols[:, None, None, :]  # (B, 1, 1, W)
        else:
            row_indices = rows[:, :, None, None]  # (B, H, 1, 1)
            col_indices = cols[:, None, :, None]  # (B, 1, W, 1)

        outputs = self.backend.ops.numpy.take_along_axis(
            grid, row_indices, axis=self.height_axis
        )
        outputs = self.backend.ops.numpy.take_along_axis(
            outputs, col_indices, axis=self.width_axis
        )

        return outputs

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        if not training or transformation is None:
            return bounding_boxes

        input_height = transformation["input_shape"][self.height_axis]
        input_width = transformation["input_shape"][self.width_axis]

        bounding_boxes = convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=input_height,
            width=input_width,
        )
        boxes = self.backend.ops.cast(bounding_boxes["boxes"], dtype="float32")
        labels = bounding_boxes["labels"]

        permutation_order = transformation["permutation_order"]
        boxes = self.backend.ops.numpy.take(boxes, permutation_order, axis=0)
        labels = self.backend.ops.numpy.take(labels, permutation_order, axis=0)

        mosaic_centers = transformation["mosaic_centers"]
        center_x = mosaic_centers[:, 0] * input_width
        center_y = mosaic_centers[:, 1] * input_height

        shift_x = self.backend.ops.numpy.stack(
            [
                center_x - input_width,
                center_x,
                center_x - input_width,
                center_x,
            ],
            axis=-1,
        )
        shift_y = self.backend.ops.numpy.stack(
            [
                center_y - input_height,
                center_y - input_height,
                center_y,
                center_y,
            ],
            axis=-1,
        )
        shift = self.backend.ops.numpy.stack(
            [shift_x, shift_y, shift_x, shift_y], axis=-1
        )
        boxes = boxes + shift[:, :, None, :]

        batch_size = self.backend.ops.shape(boxes)[0]
        boxes = self.backend.ops.numpy.reshape(boxes, (batch_size, -1, 4))
        labels = self.backend.ops.numpy.reshape(labels, (batch_size, -1))

        bounding_boxes = clip_to_image_size(
            {"boxes": boxes, "labels": labels},
            height=input_height,
            width=input_width,
            bounding_box_format="xyxy",
        )
        bounding_boxes = convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=input_height,
            width=input_width,
        )
        return bounding_boxes

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "offset": self.offset,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
