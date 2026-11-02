import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import is_int_dtype
from keras.src.backend.common import standardize_dtype


@keras_export("keras.visualization.draw_segmentation_masks")
def draw_segmentation_masks(
    images,
    segmentation_masks,
    num_classes=None,
    color_mapping=None,
    alpha=0.8,
    blend=True,
    ignore_index=-1,
    data_format=None,
):
    """Draws segmentation masks on images.

    The function overlays segmentation masks on the input images.
    The masks are blended with the images using the specified alpha value.

    Args:
        images: A batch of images as a 4D tensor or NumPy array. Shape
            should be (batch_size, height, width, channels).
        segmentation_masks: A batch of segmentation masks as a 3D or 4D tensor
            or NumPy array.  Shape should be (batch_size, height, width) or
            (batch_size, height, width, 1). The values represent class indices
            from 0 up to `num_classes - 1`.
        num_classes: The number of segmentation classes. If `None`, it is
            inferred as the maximum non-ignored value in `segmentation_masks`
            plus 1.
        color_mapping: A dictionary mapping class indices to RGB colors.
            If `None`, a default color palette is generated. The keys should be
            integers from 0 up to `num_classes - 1`.
        alpha: The opacity of the segmentation masks. Must be in the range
            `[0, 1]`.
        blend: Whether to blend the masks with the input image using the
            `alpha` value. If `False`, the masks are drawn directly on the
            images without blending. Defaults to `True`.
        ignore_index: The class index to ignore. Mask pixels with this value
            will not be drawn.  Defaults to -1.
        data_format: Image data format, either `"channels_last"` or
            `"channels_first"`. Defaults to the `image_data_format` value found
            in your Keras config file at `~/.keras/keras.json`. If you never
            set it, then it will be `"channels_last"`.

    Returns:
        A NumPy array of the images with the segmentation masks overlaid.

    Raises:
        ValueError: If the input `images` is not a 4D tensor or NumPy array.
        TypeError: If the input `segmentation_masks` is not an integer type.
    """
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if data_format == "channels_first":
        images = ops.transpose(images, (0, 2, 3, 1))
        segmentation_masks = ops.transpose(segmentation_masks, (0, 2, 3, 1))
    images = ops.convert_to_tensor(images, dtype="float32")
    segmentation_masks = ops.convert_to_tensor(segmentation_masks)

    if not is_int_dtype(segmentation_masks.dtype):
        dtype = standardize_dtype(segmentation_masks.dtype)
        raise TypeError(
            "`segmentation_masks` must be in integer dtype. "
            f"Received: segmentation_masks.dtype={dtype}"
        )

    # Infer num_classes
    if num_classes is None:
        valid_segmentation_masks = ops.where(
            ops.not_equal(segmentation_masks, ignore_index),
            segmentation_masks,
            -1,
        )
        num_classes = (
            int(ops.convert_to_numpy(ops.max(valid_segmentation_masks))) + 1
        )
        num_classes = max(1, num_classes)
    if color_mapping is None:
        colors = _generate_color_palette(num_classes)
    else:
        colors = [color_mapping[i] for i in range(num_classes)]
    valid_masks = ops.not_equal(segmentation_masks, ignore_index)
    valid_masks = ops.squeeze(valid_masks, axis=-1)
    segmentation_masks = ops.one_hot(segmentation_masks, num_classes)
    segmentation_masks = segmentation_masks[..., 0, :]
    segmentation_masks = ops.convert_to_numpy(segmentation_masks)

    # Replace class with color
    masks = segmentation_masks
    masks = np.transpose(masks, axes=(3, 0, 1, 2)).astype("bool")
    images_to_draw = ops.convert_to_numpy(images).copy()
    for mask, color in zip(masks, colors):
        color = np.array(color, dtype=images_to_draw.dtype)
        images_to_draw[mask, ...] = color[None, :]
    images_to_draw = ops.convert_to_tensor(images_to_draw)
    outputs = ops.cast(images_to_draw, dtype="float32")

    if blend:
        outputs = images * (1 - alpha) + outputs * alpha
        outputs = ops.where(valid_masks[..., None], outputs, images)
        outputs = ops.cast(outputs, dtype="uint8")
        outputs = ops.convert_to_numpy(outputs)
    return outputs


def _generate_color_palette(num_classes):
    palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [((i * palette) % 255).tolist() for i in range(num_classes)]
