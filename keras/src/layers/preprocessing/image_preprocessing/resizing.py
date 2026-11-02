from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import is_int_dtype
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.ops.core import _saturate_cast


@keras_export("keras.layers.Resizing")
class Resizing(BaseImagePreprocessingLayer):
    """A preprocessing layer which resizes images.

    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`).

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
            `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
        pad_to_aspect_ratio: If `True`, pad the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            evenly padded on the short side.
        fill_mode: When using `pad_to_aspect_ratio=True`, padded areas
            are filled according to the given mode. Only `"constant"` is
            supported at this time
            (fill with constant value, equal to `fill_value`).
        fill_value: Float. Padding value to use when `pad_to_aspect_ratio=True`.
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
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    image = x_train[0]
    resizer = keras.layers.Resizing(128, 128)
    resized_image = resizer(image)
    print("original:", image.shape, "resized:", resized_image.shape)
    ```
    """

    _USE_BASE_FACTOR = False

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        fill_mode="constant",
        fill_value=0.0,
        antialias=False,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.data_format = backend.standardize_data_format(data_format)
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.antialias = bool(antialias)
        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
        elif self.data_format == "channels_last":
            self.height_axis = -3
            self.width_axis = -2

    def _transform_images(self, images, transformation, interpolation):
        size = (self.height, self.width)
        resized = self.backend.ops.image.resize(
            images,
            size=size,
            interpolation=interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio,
            pad_to_aspect_ratio=self.pad_to_aspect_ratio,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        if resized.dtype == images.dtype:
            return resized
        if is_int_dtype(images.dtype):
            resized = self.backend.ops.numpy.round(resized)
        return _saturate_cast(resized, images.dtype, self.backend)

    def transform_images(self, images, transformation=None, training=True):
        return self._transform_images(
            images, transformation, self.interpolation
        )

    def transform_segmentation_masks(
        self, segmentation_masks, transformation=None, training=True
    ):
        return self._transform_images(
            segmentation_masks, transformation, "nearest"
        )

    def transform_labels(self, labels, transformation=None, training=True):
        return labels

    def get_random_transformation(self, data, training=True, seed=None):
        if isinstance(data, dict):
            input_shape = self.backend.ops.shape(data["images"])
        else:
            input_shape = self.backend.ops.shape(data)

        input_height, input_width = (
            input_shape[self.height_axis],
            input_shape[self.width_axis],
        )

        return input_height, input_width

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        ops = self.backend.ops
        input_height, input_width = transformation
        mask_negative_1s = ops.numpy.all(bounding_boxes["boxes"] == -1, axis=-1)
        mask_zeros = ops.numpy.all(bounding_boxes["boxes"] == 0, axis=-1)
        boxes_mask = ops.numpy.logical_or(mask_negative_1s, mask_zeros)

        bounding_boxes = convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=input_height,
            width=input_width,
        )

        bounding_boxes["boxes"] = self._transform_xyxy(
            bounding_boxes["boxes"],
            input_height=input_height,
            input_width=input_width,
        )

        bounding_boxes = clip_to_image_size(
            bounding_boxes=bounding_boxes,
            height=self.height,
            width=self.width,
        )

        bounding_boxes["boxes"] = ops.numpy.where(
            ops.numpy.expand_dims(boxes_mask, axis=-1),
            ops.convert_to_tensor(
                [0.0, 0.0, 0.0, 0.0], dtype=bounding_boxes["boxes"].dtype
            ),
            bounding_boxes["boxes"],
        )

        bounding_boxes = convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=self.height,
            width=self.width,
        )

        return bounding_boxes

    def _transform_xyxy(self, boxes, input_height, input_width):
        ops = self.backend.ops
        input_height = ops.cast(input_height, dtype=boxes.dtype)
        input_width = ops.cast(input_width, dtype=boxes.dtype)

        if self.pad_to_aspect_ratio:
            return self._transform_boxes_pad_to_aspect_ratio(
                boxes, input_height, input_width
            )
        elif self.crop_to_aspect_ratio:
            return self._transform_boxes_crop_to_aspect_ratio(
                boxes, input_height, input_width
            )
        else:
            return self._transform_boxes_stretch(
                boxes, input_height, input_width
            )

    def _transform_boxes_pad_to_aspect_ratio(
        self, boxes, input_height, input_width
    ):
        """Transforms bounding boxes for padding to aspect ratio."""
        ops = self.backend.ops
        height_ratio = ops.cast(self.height / input_height, dtype=boxes.dtype)
        width_ratio = ops.cast(self.width / input_width, dtype=boxes.dtype)
        min_aspect_ratio = ops.numpy.minimum(height_ratio, width_ratio)
        y_offset = (self.height - input_height * min_aspect_ratio) // 2
        x_offset = (self.width - input_width * min_aspect_ratio) // 2
        return ops.numpy.stack(
            [
                boxes[..., 0] * min_aspect_ratio + x_offset,
                boxes[..., 1] * min_aspect_ratio + y_offset,
                boxes[..., 2] * min_aspect_ratio + x_offset,
                boxes[..., 3] * min_aspect_ratio + y_offset,
            ],
            axis=-1,
        )

    def _transform_boxes_crop_to_aspect_ratio(
        self, boxes, input_height, input_width
    ):
        """Transforms bounding boxes for cropping to aspect ratio."""
        ops = self.backend.ops
        source_aspect_ratio = input_width / input_height
        target_aspect_ratio = self.width / self.height
        new_width = ops.numpy.where(
            source_aspect_ratio > target_aspect_ratio,
            self.height * source_aspect_ratio,
            self.width,
        )
        new_height = ops.numpy.where(
            source_aspect_ratio > target_aspect_ratio,
            self.height,
            self.width / source_aspect_ratio,
        )
        scale_x = new_width / input_width
        scale_y = new_height / input_height
        crop_left = (new_width - self.width) // 2
        crop_top = (new_height - self.height) // 2
        return ops.numpy.stack(
            [
                boxes[..., 0] * scale_x - crop_left,
                boxes[..., 1] * scale_y - crop_top,
                boxes[..., 2] * scale_x - crop_left,
                boxes[..., 3] * scale_y - crop_top,
            ],
            axis=-1,
        )

    def _transform_boxes_stretch(self, boxes, input_height, input_width):
        """Transforms bounding boxes by simple stretching."""
        ops = self.backend.ops
        height_ratio = ops.cast(self.height / input_height, dtype=boxes.dtype)
        width_ratio = ops.cast(self.width / input_width, dtype=boxes.dtype)
        return ops.numpy.stack(
            [
                boxes[..., 0] * width_ratio,
                boxes[..., 1] * height_ratio,
                boxes[..., 2] * width_ratio,
                boxes[..., 3] * height_ratio,
            ],
            axis=-1,
        )

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if len(input_shape) == 4:
            if self.data_format == "channels_last":
                input_shape[1] = self.height
                input_shape[2] = self.width
            else:
                input_shape[2] = self.height
                input_shape[3] = self.width
        else:
            if self.data_format == "channels_last":
                input_shape[0] = self.height
                input_shape[1] = self.width
            else:
                input_shape[1] = self.height
                input_shape[2] = self.width
        return tuple(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "height": self.height,
            "width": self.width,
            "interpolation": self.interpolation,
            "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
            "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "antialias": self.antialias,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
