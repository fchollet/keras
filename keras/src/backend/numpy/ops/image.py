import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.numpy.ops.core import convert_to_tensor
from keras.src.random.seed_generator import draw_seed
from keras.src.utils.module_utils import scipy

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
        return images.copy()
    # Convert to floats
    original_dtype = images.dtype
    compute_dtype = backend.result_type(images.dtype, float)
    images = images.astype(compute_dtype)

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=images.dtype)
    grayscales = np.tensordot(images, rgb_weights, axes=(channels_axis, -1))
    grayscales = np.expand_dims(grayscales, axis=channels_axis)
    return grayscales.astype(original_dtype)


def rgb_to_hsv(images, data_format=None):
    # Ref: dm_pix
    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)
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
            f"Received: images.dtype={dtype}"
        )
    eps = ml_dtypes.finfo(dtype).eps
    images = np.where(np.abs(images) < eps, 0.0, images)
    red, green, blue = np.split(images, 3, channels_axis)
    red = np.squeeze(red, channels_axis)
    green = np.squeeze(green, channels_axis)
    blue = np.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = np.maximum(np.maximum(r, g), b)
        minimum = np.minimum(np.minimum(r, g), b)
        range_ = value - minimum

        safe_value = np.where(value > 0, value, 1.0)
        safe_range = np.where(range_ > 0, range_, 1.0)

        saturation = np.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = np.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = np.where(value == r, norm * (g - b), hue)
        hue = np.where(range_ > 0, hue, 0.0) + (hue < 0.0).astype(hue.dtype)
        return hue, saturation, value

    images = np.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images.astype(dtype)


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
    hue, saturation, value = np.split(images, 3, channels_axis)
    hue = np.squeeze(hue, channels_axis)
    saturation = np.squeeze(saturation, channels_axis)
    value = np.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = np.mod(hue, 1.0) * 6.0
        dr = np.clip(np.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = np.clip(2.0 - np.abs(dh - 2.0), 0.0, 1.0)
        db = np.clip(2.0 - np.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = np.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images.astype(dtype)


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
        batch_size = images.shape[0]
        if data_format == "channels_last":
            height, width, channels = shape[-3], shape[-2], shape[-1]
        else:
            channels, height, width = shape[-3], shape[-2], shape[-1]
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        if data_format == "channels_last":
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (batch_size, img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            np.ones(
                                (batch_size, img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=1,
                    )
                else:
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            np.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=0,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (batch_size, height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            np.ones(
                                (batch_size, height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=2,
                    )
                else:
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            np.ones(
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
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (batch_size, channels, img_box_hstart, width)
                            )
                            * fill_value,
                            images,
                            np.ones(
                                (batch_size, channels, img_box_hstart, width)
                            )
                            * fill_value,
                        ],
                        axis=2,
                    )
                else:
                    padded_img = np.concatenate(
                        [
                            np.ones((channels, img_box_hstart, width))
                            * fill_value,
                            images,
                            np.ones((channels, img_box_hstart, width))
                            * fill_value,
                        ],
                        axis=1,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = np.concatenate(
                        [
                            np.ones(
                                (batch_size, channels, height, img_box_wstart)
                            )
                            * fill_value,
                            images,
                            np.ones(
                                (batch_size, channels, height, img_box_wstart)
                            )
                            * fill_value,
                        ],
                        axis=3,
                    )
                else:
                    padded_img = np.concatenate(
                        [
                            np.ones((channels, height, img_box_wstart))
                            * fill_value,
                            images,
                            np.ones((channels, height, img_box_wstart))
                            * fill_value,
                        ],
                        axis=2,
                    )
            else:
                padded_img = images
        images = padded_img

    return _resize(images, size, method=interpolation, antialias=antialias)


def _compute_weight_mat(
    input_size, output_size, scale, translation, kernel, antialias
):
    dtype = np.result_type(scale, translation)
    inv_scale = 1.0 / scale
    kernel_scale = np.maximum(inv_scale, 1.0) if antialias else 1.0

    sample_f = (
        (np.arange(output_size, dtype=dtype) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )

    x = (
        np.abs(
            sample_f[np.newaxis, :]
            - np.arange(input_size, dtype=dtype)[:, np.newaxis]
        )
        / kernel_scale
    )

    weights = kernel(x)

    total_weight_sum = np.sum(weights, axis=0, keepdims=True)
    weights = np.where(
        np.abs(total_weight_sum) > 1000.0 * np.finfo(np.float32).eps,
        np.divide(
            weights, np.where(total_weight_sum != 0, total_weight_sum, 1)
        ),
        0,
    )

    input_size_minus_0_5 = input_size - 0.5
    return np.where(
        np.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)[
            np.newaxis, :
        ],
        weights,
        0,
    )


def _resize(image, shape, method, antialias):
    if method == "nearest":
        return _resize_nearest(image, shape)
    else:
        kernel = _kernels.get(method, None)
    if kernel is None:
        raise ValueError("Unknown resize method")

    spatial_dims = tuple(
        i for i in range(len(shape)) if image.shape[i] != shape[i]
    )
    scale = [
        shape[d] / image.shape[d] if image.shape[d] != 0 else 1.0
        for d in spatial_dims
    ]

    return _scale_and_translate(
        image,
        shape,
        spatial_dims,
        scale,
        [0.0] * len(spatial_dims),
        kernel,
        antialias,
    )


def _resize_nearest(x, output_shape):
    input_shape = x.shape
    spatial_dims = tuple(
        i for i in range(len(input_shape)) if input_shape[i] != output_shape[i]
    )

    for d in spatial_dims:
        m, n = input_shape[d], output_shape[d]
        offsets = (np.arange(n, dtype=np.float32) + 0.5) * m / n
        offsets = np.floor(offsets).astype(np.int32)
        indices = [slice(None)] * len(input_shape)
        indices[d] = offsets
        x = x[tuple(indices)]
    return x


def _fill_triangle_kernel(x):
    return np.maximum(0, 1 - np.abs(x))


def _fill_keys_cubic_kernel(x):
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = np.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return np.where(x >= 2.0, 0.0, out)


def _fill_lanczos_kernel(radius, x):
    y = radius * np.sin(np.pi * x) * np.sin(np.pi * x / radius)
    out = np.where(
        x > 1e-3, np.divide(y, np.where(x != 0, np.pi**2 * x**2, 1)), 1
    )
    return np.where(x > radius, 0.0, out)


_kernels = {
    "linear": _fill_triangle_kernel,
    "bilinear": _fill_triangle_kernel,  # For `resize`.
    "cubic": _fill_keys_cubic_kernel,
    "bicubic": _fill_keys_cubic_kernel,  # For `resize`.
    "lanczos3": lambda x: _fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _fill_lanczos_kernel(5.0, x),
}


def _scale_and_translate(
    x, output_shape, spatial_dims, scale, translation, kernel, antialias
):
    input_shape = x.shape

    if len(spatial_dims) == 0:
        return x

    if np.issubdtype(x.dtype, np.integer):
        output = x.astype(np.float32)
        use_rounding = True
    else:
        output = x.copy()
        use_rounding = False

    for i, d in enumerate(spatial_dims):
        d = d % x.ndim
        m, n = input_shape[d], output_shape[d]

        w = _compute_weight_mat(
            m, n, scale[i], translation[i], kernel, antialias
        ).astype(output.dtype)
        output = np.tensordot(output, w, axes=(d, 0))
        output = np.moveaxis(output, -1, d)

    if use_rounding:
        output = np.clip(np.round(output), x.min(), x.max())
        output = output.astype(x.dtype)
    return output


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

    images = convert_to_tensor(images)
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

    # `scipy.ndimage.map_coordinates` lacks support for float16 and bfloat16.
    input_dtype = standardize_dtype(images.dtype)
    compute_dtype = backend.result_type(input_dtype, "float32")
    images = images.astype(compute_dtype)
    transform = transform.astype(compute_dtype)

    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = np.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch_size = images.shape[0]

    # get indices
    meshgrid = np.meshgrid(
        *[np.arange(size) for size in images.shape[1:]], indexing="ij"
    )
    indices = np.concatenate(
        [np.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = np.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0].copy()
    a2 = transform[:, 2].copy()
    b1 = transform[:, 4].copy()
    b2 = transform[:, 5].copy()
    transform[:, 0] = b1
    transform[:, 2] = b2
    transform[:, 4] = a0
    transform[:, 5] = a2

    # deal with transform
    transform = np.pad(transform, pad_width=[[0, 0], [0, 1]], constant_values=1)
    transform = np.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2].copy()
    offset = np.pad(offset, pad_width=[[0, 0], [0, 1]])
    transform[:, 0:2, 2] = 0

    # transform the indices
    coordinates = np.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = np.moveaxis(coordinates, source=-1, destination=1)
    coordinates += np.reshape(offset, (*offset.shape, 1, 1, 1))

    # apply affine transformation
    affined = np.stack(
        [
            map_coordinates(
                images[i],
                coordinates[i],
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(batch_size)
        ],
        axis=0,
    )

    if data_format == "channels_first":
        affined = np.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = np.squeeze(affined, axis=0)
    return affined.astype(input_dtype)


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.ndim not in (2, 3) or start_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.ndim not in (2, 3) or end_points.shape[-2:] != (4, 2):
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

    input_dtype = images.dtype
    if input_dtype == "float16":
        images = images.astype("float32")

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = np.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = np.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, channels = images.shape

    transforms = compute_homography_matrix(start_points, end_points)

    if len(transforms.shape) == 1:
        transforms = np.expand_dims(transforms, axis=0)
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = np.tile(transforms, (batch_size, 1))

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )

    output = np.empty((batch_size, height, width, channels))

    for i in range(batch_size):
        a0, a1, a2, a3, a4, a5, a6, a7 = transforms[i]
        denom = a6 * x + a7 * y + 1.0
        x_in = (a0 * x + a1 * y + a2) / denom
        y_in = (a3 * x + a4 * y + a5) / denom

        coords = np.stack([y_in.ravel(), x_in.ravel()], axis=0)

        mapped_channels = []
        for channel in range(channels):
            channel_img = images[i, :, :, channel]

            mapped_channel = map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mapped_channel.reshape(height, width))

        output[i] = np.stack(mapped_channels, axis=-1)

    if data_format == "channels_first":
        output = np.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = np.squeeze(output, axis=0)
    output = output.astype(input_dtype)

    return output


def compute_homography_matrix(start_points, end_points):
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    dtype = backend.result_type(start_points.dtype, end_points.dtype, float)
    # `np.linalg.solve` lacks support for float16 and bfloat16. Build the
    # coefficient matrix in float32 so low-precision start/end points do not
    # create an ill-conditioned (or singular) homography due to rounding.
    compute_dtype = backend.result_type(dtype, "float32")
    start_points = start_points.astype(compute_dtype)
    end_points = end_points.astype(compute_dtype)

    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    coefficient_matrix = np.stack(
        [
            np.stack(
                [
                    end_x1,
                    end_y1,
                    np.ones_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    -start_x1 * end_x1,
                    -start_x1 * end_y1,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    end_x1,
                    end_y1,
                    np.ones_like(end_x1),
                    -start_y1 * end_x1,
                    -start_y1 * end_y1,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x2,
                    end_y2,
                    np.ones_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    -start_x2 * end_x2,
                    -start_x2 * end_y2,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    end_x2,
                    end_y2,
                    np.ones_like(end_x2),
                    -start_y2 * end_x2,
                    -start_y2 * end_y2,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x3,
                    end_y3,
                    np.ones_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    -start_x3 * end_x3,
                    -start_x3 * end_y3,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    end_x3,
                    end_y3,
                    np.ones_like(end_x3),
                    -start_y3 * end_x3,
                    -start_y3 * end_y3,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x4,
                    end_y4,
                    np.ones_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    -start_x4 * end_x4,
                    -start_x4 * end_y4,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    end_x4,
                    end_y4,
                    np.ones_like(end_x4),
                    -start_y4 * end_x4,
                    -start_y4 * end_y4,
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )

    target_vector = np.stack(
        [
            start_x1,
            start_y1,
            start_x2,
            start_y2,
            start_x3,
            start_y3,
            start_x4,
            start_y4,
        ],
        axis=-1,
    )
    target_vector = np.expand_dims(target_vector, axis=-1)
    homography_matrix = np.linalg.solve(coefficient_matrix, target_vector)
    homography_matrix = np.reshape(homography_matrix, [-1, 8])
    return homography_matrix.astype(compute_dtype)


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
            f"{set(MAP_COORDINATES_FILL_MODES.keys())}. Received: "
            f"fill_mode={fill_mode}"
        )
    if order not in range(2):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"{[0, 1]}. Received: order={order}"
        )
    # SciPy's implementation of map_coordinates handles boundaries incorrectly,
    # unless mode='reflect'. For order=1, this only affects interpolation
    # outside the bounds of the original array.
    # https://github.com/scipy/scipy/issues/2640
    padding = [
        (
            max(-np.floor(c.min()).astype(int) + 1, 0),
            max(np.ceil(c.max()).astype(int) + 1 - size, 0),
        )
        for c, size in zip(coordinates, inputs.shape)
    ]
    shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
    pad_mode = {
        "nearest": "edge",
        "mirror": "reflect",
        "reflect": "symmetric",
    }.get(fill_mode, fill_mode)
    if fill_mode == "constant":
        padded = np.pad(
            inputs, padding, mode=pad_mode, constant_values=fill_value
        )
    else:
        padded = np.pad(inputs, padding, mode=pad_mode)

    # `scipy.ndimage.map_coordinates` lacks support for float16 and bfloat16.
    if is_float_dtype(padded.dtype):
        padded = padded.astype("float32")
    result = scipy.ndimage.map_coordinates(
        padded, shifted_coords, order=order, mode=fill_mode, cval=fill_value
    )
    return result.astype(inputs.dtype)


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = np.arange(size, dtype=dtype) - (size - 1) / 2
            kernel1d = np.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / np.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            size = np.asarray(size, dtype)
            kernel1d_height = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_width = _get_gaussian_kernel1d(size[1], sigma[1])
            return np.outer(kernel1d_height, kernel1d_width)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel[:, :, np.newaxis]
        kernel = np.tile(kernel, (1, 1, num_channels))
        return kernel.astype(dtype)

    data_format = backend.standardize_data_format(data_format)
    images = convert_to_tensor(images)
    kernel_size = convert_to_tensor(kernel_size)
    sigma = convert_to_tensor(sigma)
    input_dtype = standardize_dtype(images.dtype)
    # `scipy.signal.convolve2d` lacks support for float16 and bfloat16.
    compute_dtype = backend.result_type(input_dtype, "float32")
    images = images.astype(compute_dtype)
    sigma = sigma.astype(compute_dtype)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, num_channels = images.shape

    kernel = _create_gaussian_kernel(
        kernel_size, sigma, num_channels, input_dtype
    )

    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    pad_h = (kernel_h - 1) // 2
    pad_h_after = kernel_h - 1 - pad_h
    pad_w = (kernel_w - 1) // 2
    pad_w_after = kernel_w - 1 - pad_w

    blurred_images = np.empty_like(images)

    for b in range(batch_size):
        for ch in range(num_channels):
            padded = np.pad(
                images[b, :, :, ch],
                ((pad_h, pad_h_after), (pad_w, pad_w_after)),
                mode="constant",
            )
            blurred_images[b, :, :, ch] = scipy.signal.convolve2d(
                padded, kernel[:, :, ch], mode="valid"
            )

    if data_format == "channels_first":
        blurred_images = np.transpose(blurred_images, (0, 3, 1, 2))
    if need_squeeze:
        blurred_images = np.squeeze(blurred_images, axis=0)
    return blurred_images.astype(input_dtype)


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
    input_dtype = images.dtype

    alpha = convert_to_tensor(alpha, dtype=input_dtype)
    sigma = convert_to_tensor(sigma, dtype=input_dtype)

    kernel_size = (int(6 * sigma) | 1, int(6 * sigma) | 1)

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    dx = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0).astype(
            input_dtype
        )
        * sigma
    )
    dy = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0).astype(
            input_dtype
        )
        * sigma
    )

    dx = gaussian_blur(
        np.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )
    dy = gaussian_blur(
        np.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )

    dx = np.squeeze(dx)
    dy = np.squeeze(dy)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x[None, :, :], y[None, :, :]

    distorted_x = x + alpha * dx
    distorted_y = y + alpha * dy

    transformed_images = np.zeros_like(images)

    if data_format == "channels_last":
        for i in range(channels):
            transformed_images[..., i] = np.stack(
                [
                    map_coordinates(
                        images[b, ..., i],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                    for b in range(batch_size)
                ]
            )
    else:
        for i in range(channels):
            transformed_images[:, i, :, :] = np.stack(
                [
                    map_coordinates(
                        images[b, i, ...],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                    for b in range(batch_size)
                ]
            )

    if need_squeeze:
        transformed_images = np.squeeze(transformed_images, axis=0)
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
    if method in ("linear", "bilinear", "trilinear", "triangle"):
        method = "linear"
    elif method in ("cubic", "bicubic", "tricubic"):
        method = "cubic"

    images = convert_to_tensor(images)
    scale = convert_to_tensor(scale)
    translation = convert_to_tensor(translation)
    kernel = _kernels[method]
    dtype = backend.result_type(scale.dtype, translation.dtype)
    scale = scale.astype(dtype)
    translation = translation.astype(dtype)
    return _scale_and_translate(
        images,
        output_shape,
        spatial_dims,
        scale,
        translation,
        kernel,
        antialias,
    )


def sobel_edges(images, data_format=None):
    from scipy import ndimage

    images = convert_to_tensor(images)
    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch, height, width, channels = images.shape

    # Output shape: (batch, height, width, channels, 2)
    edges = np.zeros((batch, height, width, channels, 2), dtype=images.dtype)

    for b in range(batch):
        for c in range(channels):
            # axis=0 is vertical (y), axis=1 is horizontal (x)
            edges[b, :, :, c, 0] = ndimage.sobel(images[b, :, :, c], axis=0)
            edges[b, :, :, c, 1] = ndimage.sobel(images[b, :, :, c], axis=1)

    if data_format == "channels_first":
        edges = np.transpose(edges, (0, 3, 1, 2, 4))

    return edges
