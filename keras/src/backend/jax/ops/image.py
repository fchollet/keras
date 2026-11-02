import functools

import jax
import jax.numpy as jnp

from keras.src import backend
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.jax.ops.core import convert_to_tensor
from keras.src.random.seed_generator import draw_seed

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)
AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
SCALE_AND_TRANSLATE_METHODS = {
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5",
}


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if images.shape[channels_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if images.shape[channels_axis] == 1:
        return images
    # Convert to floats
    original_dtype = images.dtype
    compute_dtype = backend.result_type(images.dtype, float)
    images = images.astype(compute_dtype)

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = convert_to_tensor(
        [0.2989, 0.5870, 0.1140], dtype=images.dtype
    )
    images = jnp.tensordot(images, rgb_weights, axes=(channels_axis, -1))
    images = jnp.expand_dims(images, axis=channels_axis)
    return images.astype(original_dtype)


def rgb_to_hsv(images, data_format=None):
    # Ref: dm_pix
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={standardize_dtype(dtype)}"
        )
    eps = jnp.finfo(dtype).eps
    images = jnp.where(jnp.abs(images) < eps, 0.0, images)
    red, green, blue = jnp.split(images, 3, channels_axis)
    red = jnp.squeeze(red, channels_axis)
    green = jnp.squeeze(green, channels_axis)
    blue = jnp.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = jnp.maximum(jnp.maximum(r, g), b)
        minimum = jnp.minimum(jnp.minimum(r, g), b)
        range_ = value - minimum

        safe_value = jnp.where(value > 0, value, 1.0)
        safe_range = jnp.where(range_ > 0, range_, 1.0)

        saturation = jnp.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = jnp.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = jnp.where(value == r, norm * (g - b), hue)
        hue = jnp.where(range_ > 0, hue, 0.0) + (hue < 0.0).astype(hue.dtype)
        return hue, saturation, value

    images = jnp.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images


def hsv_to_rgb(images, data_format=None):
    # Ref: dm_pix
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={standardize_dtype(dtype)}"
        )
    hue, saturation, value = jnp.split(images, 3, channels_axis)
    hue = jnp.squeeze(hue, channels_axis)
    saturation = jnp.squeeze(saturation, channels_axis)
    value = jnp.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = jnp.mod(hue, 1.0) * 6.0
        dr = jnp.clip(jnp.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = jnp.clip(2.0 - jnp.abs(dh - 2.0), 0.0, 1.0)
        db = jnp.clip(2.0 - jnp.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = jnp.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images


def resize(
    images,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    target_height, target_width = size
    if len(images.shape) == 4:
        if data_format == "channels_last":
            size = (images.shape[0],) + size + (images.shape[-1],)
        else:
            size = (images.shape[0], images.shape[1]) + size
        batch_size = images.shape[0]
    elif len(images.shape) == 3:
        if data_format == "channels_last":
            size = size + (images.shape[-1],)
        else:
            size = (images.shape[0],) + size
    else:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if crop_to_aspect_ratio:
        shape = images.shape
        if data_format == "channels_last":
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        if data_format == "channels_last":
            if len(images.shape) == 4:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
            else:
                images = images[
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
        else:
            if len(images.shape) == 4:
                images = images[
                    :,
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
            else:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
    elif pad_to_aspect_ratio:
        shape = images.shape
        if data_format == "channels_last":
            height, width, channels = shape[-3], shape[-2], shape[-1]
        else:
            height, width, channels = shape[-2], shape[-1], shape[-3]

        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)
        if data_format == "channels_last":
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (batch_size, img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (batch_size, img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=1,
                    )
                else:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=0,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (batch_size, height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (batch_size, height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=2,
                    )
                else:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=1,
                    )
            else:
                padded_img = images
        else:
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (batch_size, channels, img_box_hstart, width)
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (batch_size, channels, img_box_hstart, width)
                            )
                            * fill_value,
                        ],
                        axis=2,
                    )
                else:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones((channels, img_box_hstart, width))
                            * fill_value,
                            images,
                            jnp.ones((channels, img_box_hstart, width))
                            * fill_value,
                        ],
                        axis=1,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones(
                                (batch_size, channels, height, img_box_wstart)
                            )
                            * fill_value,
                            images,
                            jnp.ones(
                                (batch_size, channels, height, img_box_wstart)
                            )
                            * fill_value,
                        ],
                        axis=3,
                    )
                else:
                    padded_img = jnp.concatenate(
                        [
                            jnp.ones((channels, height, img_box_wstart))
                            * fill_value,
                            images,
                            jnp.ones((channels, height, img_box_wstart))
                            * fill_value,
                        ],
                        axis=2,
                    )
            else:
                padded_img = images
        images = padded_img

    return jax.image.resize(
        images, size, method=interpolation, antialias=antialias
    )


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )

    transform = convert_to_tensor(transform)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )
    if isinstance(transform.shape[-1], int) and transform.shape[-1] != 8:
        raise ValueError(
            "Invalid transform shape: expected the last dimension to be 8. "
            f"Received: transform.shape={transform.shape}"
        )

    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = jnp.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = jnp.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))

    batch_size = images.shape[0]

    # get indices
    meshgrid = jnp.meshgrid(
        *[jnp.arange(size) for size in images.shape[1:]], indexing="ij"
    )
    indices = jnp.concatenate(
        [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = jnp.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0]
    a2 = transform[:, 2]
    b1 = transform[:, 4]
    b2 = transform[:, 5]
    transform = transform.at[:, 0].set(b1)
    transform = transform.at[:, 2].set(b2)
    transform = transform.at[:, 4].set(a0)
    transform = transform.at[:, 5].set(a2)

    # deal with transform
    transform = jnp.pad(
        transform, pad_width=[[0, 0], [0, 1]], constant_values=1
    )
    transform = jnp.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2]
    offset = jnp.pad(offset, pad_width=[[0, 0], [0, 1]])
    transform = transform.at[:, 0:2, 2].set(0)

    # transform the indices
    coordinates = jnp.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = jnp.moveaxis(coordinates, source=-1, destination=1)
    coordinates += jnp.reshape(offset, shape=(*offset.shape, 1, 1, 1))

    # apply affine transformation
    _map_coordinates = functools.partial(
        jax.scipy.ndimage.map_coordinates,
        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        mode=fill_mode,
        cval=fill_value,
    )
    affined = jax.vmap(_map_coordinates)(images, coordinates)

    if data_format == "channels_first":
        affined = jnp.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = jnp.squeeze(affined, axis=0)
    return affined


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.shape[-2:] != (4, 2) or start_points.ndim not in (2, 3):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.shape[-2:] != (4, 2) or end_points.ndim not in (2, 3):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )
    if start_points.shape != end_points.shape:
        raise ValueError(
            "start_points and end_points must have the same shape."
            f" Received start_points.shape={start_points.shape}, "
            f"end_points.shape={end_points.shape}"
        )

    need_squeeze = False
    if len(images.shape) == 3:
        images = jnp.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = jnp.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = jnp.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))

    _, height, width, _ = images.shape
    transforms = compute_homography_matrix(
        jnp.asarray(start_points, dtype="float32"),
        jnp.asarray(end_points, dtype="float32"),
    )

    x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="xy")
    grid = jnp.stack([x.ravel(), y.ravel(), jnp.ones_like(x).ravel()], axis=0)

    def transform_coordinates(transform):
        denom = transform[6] * grid[0] + transform[7] * grid[1] + 1.0
        x_in = (
            transform[0] * grid[0] + transform[1] * grid[1] + transform[2]
        ) / denom
        y_in = (
            transform[3] * grid[0] + transform[4] * grid[1] + transform[5]
        ) / denom
        return jnp.stack([y_in, x_in], axis=0)

    transformed_coords = jax.vmap(transform_coordinates)(transforms)

    def interpolate_image(image, coords):
        def interpolate_channel(channel_img):
            return jax.scipy.ndimage.map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                mode="constant",
                cval=fill_value,
            ).reshape(height, width)

        return jax.vmap(interpolate_channel, in_axes=0)(
            jnp.moveaxis(image, -1, 0)
        )

    output = jax.vmap(interpolate_image, in_axes=(0, 0))(
        images, transformed_coords
    )
    output = jnp.moveaxis(output, 1, -1)

    if data_format == "channels_first":
        output = jnp.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = jnp.squeeze(output, axis=0)

    return output


def compute_homography_matrix(start_points, end_points):
    start_x, start_y = start_points[..., 0], start_points[..., 1]
    end_x, end_y = end_points[..., 0], end_points[..., 1]

    zeros = jnp.zeros_like(end_x)
    ones = jnp.ones_like(end_x)

    x_rows = jnp.stack(
        [
            end_x,
            end_y,
            ones,
            zeros,
            zeros,
            zeros,
            -start_x * end_x,
            -start_x * end_y,
        ],
        axis=-1,
    )
    y_rows = jnp.stack(
        [
            zeros,
            zeros,
            zeros,
            end_x,
            end_y,
            ones,
            -start_y * end_x,
            -start_y * end_y,
        ],
        axis=-1,
    )

    coefficient_matrix = jnp.concatenate([x_rows, y_rows], axis=1)

    target_vector = jnp.expand_dims(
        jnp.concatenate([start_x, start_y], axis=-1), axis=-1
    )

    homography_matrix = jnp.linalg.solve(coefficient_matrix, target_vector)

    return homography_matrix.squeeze(-1)


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    inputs = convert_to_tensor(inputs)
    coordinates = convert_to_tensor(coordinates)
    if coordinates.shape[0] != len(inputs.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {inputs.shape} and coordinate "
            f"leading dim of {coordinates.shape[0]}"
        )
    if len(coordinates.shape) < 2:
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {coordinates.shape}"
        )
    if fill_mode not in MAP_COORDINATES_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(MAP_COORDINATES_FILL_MODES)}. Received: "
            f"fill_mode={fill_mode}"
        )
    if order not in range(2):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"{[0, 1]}. Received: order={order}"
        )
    return jax.scipy.ndimage.map_coordinates(
        inputs, coordinates, order, fill_mode, fill_value
    )


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = jnp.arange(size, dtype=dtype) - jnp.array(
                (size - 1) / 2, dtype=dtype
            )
            kernel1d = jnp.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / jnp.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            kernel1d_height = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_width = _get_gaussian_kernel1d(size[1], sigma[1])
            return jnp.outer(kernel1d_height, kernel1d_width)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)[
            jnp.newaxis, jnp.newaxis, :, :
        ]
        return kernel

    data_format = backend.standardize_data_format(data_format)
    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)
    sigma = convert_to_tensor(sigma, dtype=dtype)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if images.ndim == 3:
        images = images[jnp.newaxis, ...]
        need_squeeze = True

    if data_format == "channels_last":
        images = jnp.transpose(images, (0, 3, 1, 2))

    num_channels = images.shape[1]
    kernel = _create_gaussian_kernel(kernel_size, sigma, dtype)

    kernel = jnp.tile(kernel, (num_channels, 1, 1, 1))

    blurred_images = jax.lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=num_channels,
    )

    if data_format == "channels_last":
        blurred_images = jnp.transpose(blurred_images, (0, 2, 3, 1))

    if need_squeeze:
        blurred_images = blurred_images.squeeze(axis=0)

    return blurred_images


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    images = convert_to_tensor(images)
    alpha = convert_to_tensor(alpha)
    sigma = convert_to_tensor(sigma)
    input_dtype = images.dtype
    kernel_size = (int(6 * sigma) | 1, int(6 * sigma) | 1)

    need_squeeze = False
    if len(images.shape) == 3:
        images = jnp.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    seed = draw_seed(seed)
    dx = (
        jax.random.normal(
            seed, shape=(batch_size, height, width), dtype=input_dtype
        )
        * sigma
    )
    dy = (
        jax.random.normal(
            seed, shape=(batch_size, height, width), dtype=input_dtype
        )
        * sigma
    )

    dx = gaussian_blur(
        jnp.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )
    dy = gaussian_blur(
        jnp.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )

    dx = jnp.squeeze(dx)
    dy = jnp.squeeze(dy)

    x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    x, y = x[None, :, :], y[None, :, :]

    distorted_x = x + alpha * dx
    distorted_y = y + alpha * dy

    transformed_images = jnp.zeros_like(images)

    if data_format == "channels_last":
        for i in range(channels):
            transformed_images = transformed_images.at[..., i].set(
                jnp.stack(
                    [
                        map_coordinates(
                            images[b, ..., i],
                            [distorted_y[b], distorted_x[b]],
                            order=AFFINE_TRANSFORM_INTERPOLATIONS[
                                interpolation
                            ],
                            fill_mode=fill_mode,
                            fill_value=fill_value,
                        )
                        for b in range(batch_size)
                    ]
                )
            )
    else:
        for i in range(channels):
            transformed_images = transformed_images.at[:, i, :, :].set(
                jnp.stack(
                    [
                        map_coordinates(
                            images[b, i, ...],
                            [distorted_y[b], distorted_x[b]],
                            order=AFFINE_TRANSFORM_INTERPOLATIONS[
                                interpolation
                            ],
                            fill_mode=fill_mode,
                            fill_value=fill_value,
                        )
                        for b in range(batch_size)
                    ]
                )
            )

    if need_squeeze:
        transformed_images = jnp.squeeze(transformed_images, axis=0)
    transformed_images = transformed_images.astype(input_dtype)

    return transformed_images


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    if method not in SCALE_AND_TRANSLATE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{SCALE_AND_TRANSLATE_METHODS}. Received: method={method}"
        )
    images = convert_to_tensor(images)
    scale = convert_to_tensor(scale)
    translation = convert_to_tensor(translation)
    return jax.image.scale_and_translate(
        images,
        output_shape,
        spatial_dims,
        scale,
        translation,
        method,
        antialias,
    )


def sobel_edges(images, data_format=None):
    images = convert_to_tensor(images)

    # Convert to channels_first for convolution (like gaussian_blur)
    if data_format != "channels_first":
        images = jnp.transpose(images, (0, 3, 1, 2))

    num_channels = images.shape[1]

    # Sobel kernels - shape (1, 1, 3, 3) then tile to (C, 1, 3, 3) for OIHW
    sobel_x = jnp.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=images.dtype
    )[jnp.newaxis, jnp.newaxis, :, :]
    sobel_y = jnp.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=images.dtype
    )[jnp.newaxis, jnp.newaxis, :, :]

    kernel_x = jnp.tile(sobel_x, (num_channels, 1, 1, 1))
    kernel_y = jnp.tile(sobel_y, (num_channels, 1, 1, 1))

    # Apply convolutions using depthwise conv with NCHW format
    edges_x = jax.lax.conv_general_dilated(
        images,
        kernel_x,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=num_channels,
    )
    edges_y = jax.lax.conv_general_dilated(
        images,
        kernel_y,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=num_channels,
    )

    # Convert back to channels_last: (N, C, H, W) -> (N, H, W, C)
    if data_format != "channels_first":
        edges_x = jnp.transpose(edges_x, (0, 2, 3, 1))
        edges_y = jnp.transpose(edges_y, (0, 2, 3, 1))

    # Stack to get (..., 2) where last dim contains [dy, dx]
    edges = jnp.stack([edges_y, edges_x], axis=-1)

    return edges
