from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import is_int_dtype
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape


class RGBToGrayscale(Operation):
    def __init__(self, data_format=None, *, name=None):
        super().__init__(name=name)
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.rgb_to_grayscale(
            images, data_format=self.data_format
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        channels_axis = -1 if self.data_format == "channels_last" else -3
        channels = images_shape[channels_axis]
        if channels is not None and channels not in (1, 3):
            raise ValueError(
                "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
                f"Received input with shape: images.shape={tuple(images_shape)}"
            )
        images_shape[channels_axis] = 1
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.rgb_to_grayscale")
def rgb_to_grayscale(images, data_format=None):
    """Convert RGB images to grayscale.

    This function converts RGB images to grayscale images. It supports both
    3D and 4D tensors.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Grayscale image or batch of grayscale images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.rgb_to_grayscale(x)
    >>> y.shape
    (2, 4, 4, 1)

    >>> x = np.random.random((4, 4, 3)) # Single RGB image
    >>> y = ops.image.rgb_to_grayscale(x)
    >>> y.shape
    (4, 4, 1)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.rgb_to_grayscale(x, data_format="channels_first")
    >>> y.shape
    (2, 1, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return RGBToGrayscale(data_format=data_format).symbolic_call(images)
    return backend.ops.image.rgb_to_grayscale(images, data_format=data_format)


class RGBToHSV(Operation):
    def __init__(self, data_format=None, *, name=None):
        super().__init__(name=name)
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.rgb_to_hsv(
            images, data_format=self.data_format
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        dtype = images.dtype
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        if not is_float_dtype(dtype):
            raise ValueError(
                "Invalid images dtype: expected float dtype. "
                f"Received: images.dtype={dtype}"
            )
        channels_axis = -1 if self.data_format == "channels_last" else -3
        channels = images_shape[channels_axis]
        if channels is not None and channels != 3:
            raise ValueError(
                "Input images must have 3 channels, but received images with "
                f"{channels} channels."
            )
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.rgb_to_hsv")
def rgb_to_hsv(images, data_format=None):
    """Convert RGB images to HSV.

    `images` must be of float dtype, and the output is only well defined if the
    values in `images` are in `[0, 1]`.

    All HSV values are in `[0, 1]`. A hue of `0` corresponds to pure red, `1/3`
    is pure green, and `2/3` is pure blue.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        HSV image or batch of HSV images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.rgb_to_hsv(x)
    >>> y.shape
    (2, 4, 4, 3)

    >>> x = np.random.random((4, 4, 3)) # Single RGB image
    >>> y = ops.image.rgb_to_hsv(x)
    >>> y.shape
    (4, 4, 3)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.rgb_to_hsv(x, data_format="channels_first")
    >>> y.shape
    (2, 3, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return RGBToHSV(data_format=data_format).symbolic_call(images)
    return backend.ops.image.rgb_to_hsv(images, data_format=data_format)


class HSVToRGB(Operation):
    def __init__(self, data_format=None, *, name=None):
        super().__init__(name=name)
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.hsv_to_rgb(
            images, data_format=self.data_format
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        dtype = images.dtype
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        if not is_float_dtype(dtype):
            raise ValueError(
                "Invalid images dtype: expected float dtype. "
                f"Received: images.dtype={dtype}"
            )
        channels_axis = -1 if self.data_format == "channels_last" else -3
        channels = images_shape[channels_axis]
        if channels is not None and channels != 3:
            raise ValueError(
                "Input images must have 3 channels, but received images with "
                f"{channels} channels."
            )
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.hsv_to_rgb")
def hsv_to_rgb(images, data_format=None):
    """Convert HSV images to RGB.

    `images` must be of float dtype, and the output is only well defined if the
    values in `images` are in `[0, 1]`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        RGB image or batch of RGB images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.hsv_to_rgb(x)
    >>> y.shape
    (2, 4, 4, 3)

    >>> x = np.random.random((4, 4, 3)) # Single HSV image
    >>> y = ops.image.hsv_to_rgb(x)
    >>> y.shape
    (4, 4, 3)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.hsv_to_rgb(x, data_format="channels_first")
    >>> y.shape
    (2, 3, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return HSVToRGB(data_format=data_format).symbolic_call(images)
    return backend.ops.image.hsv_to_rgb(images, data_format=data_format)


class Resize(Operation):
    def __init__(
        self,
        size,
        interpolation="bilinear",
        antialias=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        fill_mode="constant",
        fill_value=0.0,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.size = tuple(size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _resize(
            images,
            self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio,
            pad_to_aspect_ratio=self.pad_to_aspect_ratio,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"images.shape={images.shape}"
            )
        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
        else:
            height_axis, width_axis = -2, -1
        images_shape[height_axis] = self.size[0]
        images_shape[width_axis] = self.size[1]
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.resize")
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
    """Resize images to size using the specified interpolation method.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        size: Size of output image in `(height, width)` format.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            `"bilinear"`, `"bicubic"`, `"lanczos3"`, and `"lanczos5"`.
            Defaults to `"bilinear"`.
        antialias: Whether to use an antialiasing filter when downsampling an
            image. Defaults to `False`.
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
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Resized image or batch of images.

    Examples:

    >>> x = np.random.random((2, 4, 4, 3)) # batch of 2 RGB images
    >>> y = keras.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 2, 3)

    >>> x = np.random.random((4, 4, 3)) # single RGB image
    >>> y = keras.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 3)

    >>> x = np.random.random((2, 3, 4, 4)) # batch of 2 RGB images
    >>> y = keras.ops.image.resize(x, (2, 2),
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 2, 2)
    """
    if len(size) != 2:
        raise ValueError(
            "Expected `size` to be a tuple of 2 integers. "
            f"Received: size={size}"
        )
    if (isinstance(size[0], int) and size[0] <= 0) or (
        isinstance(size[1], int) and size[1] <= 0
    ):
        raise ValueError(
            f"`size` must have positive height and width. Received: size={size}"
        )
    if len(images.shape) < 3 or len(images.shape) > 4:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if any_symbolic_tensors((images,)):
        return Resize(
            size,
            interpolation=interpolation,
            antialias=antialias,
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            fill_mode=fill_mode,
            fill_value=fill_value,
        ).symbolic_call(images)
    return _resize(
        images,
        size,
        interpolation=interpolation,
        antialias=antialias,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        data_format=data_format,
        pad_to_aspect_ratio=pad_to_aspect_ratio,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )


def _resize(
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
    resized = backend.ops.image.resize(
        images,
        size,
        interpolation=interpolation,
        antialias=antialias,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        data_format=data_format,
        pad_to_aspect_ratio=pad_to_aspect_ratio,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    if resized.dtype == images.dtype:
        # Only `torch` backend will cast result to original dtype with
        # correct rounding and without dtype overflow
        return resized
    if is_int_dtype(images.dtype):
        resized = ops.round(resized)
    return ops.saturate_cast(resized, images.dtype)


class AffineTransform(Operation):
    def __init__(
        self,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images, transform):
        return backend.ops.image.affine_transform(
            images,
            transform,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images, transform):
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
        return KerasTensor(images.shape, dtype=images.dtype)


@keras_export("keras.ops.image.affine_transform")
def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    """Applies the given transform(s) to the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        transform: Projective transform matrix/matrices. A vector of length 8 or
            tensor of size N x 8. If one row of transform is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the output point
            `(x, y)` to a transformed input point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`. The transform is inverted compared to
            the transform mapping input points to output points. Note that
            gradients are not backpropagated into transformation parameters.
            Note that `c0` and `c1` are only effective when using TensorFlow
            backend and will be considered as `0` when using other backends.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            and `"bilinear"`. Defaults to `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        fill_value: Value used for points outside the boundaries of the input if
            `fill_mode="constant"`. Defaults to `0`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Applied affine transform image or batch of images.

    Examples:

    >>> x = np.random.random((2, 64, 80, 3)) # batch of 2 RGB images
    >>> transform = np.array(
    ...     [
    ...         [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
    ...         [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ...     ]
    ... )
    >>> y = keras.ops.image.affine_transform(x, transform)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3)) # single RGB image
    >>> transform = np.array([1.0, 0.5, -20, 0.5, 1.0, -16, 0, 0])  # shear
    >>> y = keras.ops.image.affine_transform(x, transform)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80)) # batch of 2 RGB images
    >>> transform = np.array(
    ...     [
    ...         [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
    ...         [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ...     ]
    ... )
    >>> y = keras.ops.image.affine_transform(x, transform,
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((images, transform)):
        return AffineTransform(
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
            data_format=data_format,
        ).symbolic_call(images, transform)
    return backend.ops.image.affine_transform(
        images,
        transform,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format,
    )


class ExtractPatches(Operation):
    def __init__(
        self,
        size,
        strides=None,
        dilation_rate=1,
        padding="valid",
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.is_3d = len(self.size) == 3
        if strides is None:
            strides = size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _extract_patches(
            images=images,
            size=self.size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        original_ndim = len(images_shape)
        if self.data_format == "channels_last":
            channels_in = images_shape[-1]
        else:
            channels_in = images_shape[-4] if self.is_3d else images_shape[-3]

        if self.is_3d:
            # 3D patch extraction
            if original_ndim == 4:
                images_shape = [1] + images_shape
            filters = self.size[0] * self.size[1] * self.size[2] * channels_in
            kernel_size = (self.size[0], self.size[1], self.size[2])
        else:
            # 2D patch extraction
            if original_ndim == 3:
                images_shape = [1] + images_shape
            filters = self.size[0] * self.size[1] * channels_in
            kernel_size = (self.size[0], self.size[1])

        out_shape = compute_conv_output_shape(
            images_shape,
            filters,
            kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if self.is_3d:
            if original_ndim == 4:
                out_shape = out_shape[1:]
        else:
            if original_ndim == 3:
                out_shape = out_shape[1:]
        return KerasTensor(shape=out_shape, dtype=images.dtype)

    def get_config(self):
        return {
            "size": self.size,
            "strides": self.strides,
            "dilation_rate": self.dilation_rate,
            "padding": self.padding,
            "data_format": self.data_format,
        }


@keras_export("keras.ops.image.extract_patches")
def extract_patches(
    images,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    """Extracts patches from the image(s) or volume(s).

    This function supports both 2D and 3D patch extraction based on the
    `size` argument length, similar to how `keras.ops.conv` handles
    different dimensions.

    Args:
        images: Input image/volume or batch of images/volumes.
            For 2D patches: 3D `(H, W, C)` or 4D `(N, H, W, C)`.
            For 3D patches: 4D `(D, H, W, C)` or 5D `(N, D, H, W, C)`.
        size: Patch size as int or tuple.
            Length 2 tuple `(patch_height, patch_width)` or int for 2D patches.
            Length 3 tuple `(patch_depth, patch_height, patch_width)` for
            3D patches.
        strides: Strides for patch extraction. If not specified, defaults
            to `size` (non-overlapping patches).
        dilation_rate: Dilation rate for patch extraction. Note that
            `dilation_rate > 1` is not supported with `strides > 1`.
        padding: The type of padding algorithm to use: `"same"` or `"valid"`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, defaults to `keras.config.image_data_format`.

    Returns:
        Extracted patches with shape depending on input and `size`:
        - 2D patches: 3D (unbatched) or 4D (batched)
        - 3D patches: 4D (unbatched) or 5D (batched)

    Examples:

    >>> # 2D patches from batch of images
    >>> image = np.random.random(
    ...     (2, 20, 20, 3)
    ... ).astype("float32")
    >>> patches = keras.ops.image.extract_patches(image, (5, 5))
    >>> patches.shape
    (2, 4, 4, 75)

    >>> # 2D patches from single image
    >>> image = np.random.random((20, 20, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches(image, (3, 3), (1, 1))
    >>> patches.shape
    (18, 18, 27)

    >>> # 3D patches from batch of volumes
    >>> volumes = np.random.random(
    ...     (2, 10, 10, 10, 3)
    ... ).astype("float32")
    >>> patches = keras.ops.image.extract_patches(volumes, (3, 3, 3))
    >>> patches.shape
    (2, 3, 3, 3, 81)

    >>> # 3D patches from single volume
    >>> volume = np.random.random((10, 10, 10, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches(volume, (3, 3, 3))
    >>> patches.shape
    (3, 3, 3, 81)
    """
    # Validate size argument
    if not isinstance(size, int):
        if not isinstance(size, (tuple, list)):
            raise TypeError(
                "Invalid `size` argument. Expected an int or a tuple. "
                f"Received: size={size} of type {type(size).__name__}"
            )
        if len(size) not in (2, 3):
            raise ValueError(
                "Invalid `size` argument. Expected a tuple of length 2 or 3. "
                f"Received: size={size} with length {len(size)}"
            )

    # 2D patch extraction (default)
    if any_symbolic_tensors((images,)):
        return ExtractPatches(
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=data_format,
        ).symbolic_call(images)

    return _extract_patches(
        images, size, strides, dilation_rate, padding, data_format=data_format
    )


def _extract_patches(
    images,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    if not isinstance(size, int) and len(size) == 3:
        return _extract_patches_3d(
            images, size, strides, dilation_rate, padding, data_format
        )
    return _extract_patches_2d(
        images, size, strides, dilation_rate, padding, data_format
    )


def _extract_patches_2d(
    images,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    if isinstance(size, int):
        patch_h = patch_w = size
    elif len(size) == 2:
        patch_h, patch_w = size[0], size[1]
    else:
        raise TypeError(
            "Invalid `size` argument. Expected an "
            f"int or a tuple of length 2. Received: size={size}"
        )
    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        channels_in = images.shape[-1]
    elif data_format == "channels_first":
        channels_in = images.shape[-3]
    if not strides:
        strides = size
    out_dim = patch_h * patch_w * channels_in
    kernel = backend.ops.numpy.eye(out_dim, dtype=images.dtype)
    kernel = backend.ops.numpy.reshape(
        kernel, (patch_h, patch_w, channels_in, out_dim)
    )
    _unbatched = False
    if len(images.shape) == 3:
        _unbatched = True
        images = backend.ops.numpy.expand_dims(images, axis=0)
    patches = backend.ops.nn.conv(
        inputs=images,
        kernel=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )
    if _unbatched:
        patches = backend.ops.numpy.squeeze(patches, axis=0)
    return patches


def _extract_patches_3d(
    volumes,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    if isinstance(size, int):
        patch_d = patch_h = patch_w = size
    elif len(size) == 3:
        patch_d, patch_h, patch_w = size
    else:
        raise TypeError(
            "Invalid `size` argument. Expected an "
            f"int or a tuple of length 3. Received: size={size}"
        )
    if strides is None:
        strides = size
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    if len(strides) != 3:
        raise ValueError(f"Invalid `strides` argument. Got: {strides}")
    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        channels_in = volumes.shape[-1]
    elif data_format == "channels_first":
        channels_in = volumes.shape[-4]
    out_dim = patch_d * patch_w * patch_h * channels_in
    kernel = backend.ops.numpy.eye(out_dim, dtype=volumes.dtype)
    kernel = backend.ops.numpy.reshape(
        kernel, (patch_d, patch_h, patch_w, channels_in, out_dim)
    )
    _unbatched = False
    if len(volumes.shape) == 4:
        _unbatched = True
        volumes = backend.ops.numpy.expand_dims(volumes, axis=0)
    patches = backend.ops.nn.conv(
        inputs=volumes,
        kernel=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )
    if _unbatched:
        patches = backend.ops.numpy.squeeze(patches, axis=0)
    return patches


@keras_export("keras.ops.image.extract_patches_3d")
def extract_patches_3d(
    volumes,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    """Extracts patches from the volume(s).

    Args:
        volumes: Input volume or batch of volumes. Must be 4D or 5D.
        size: Patch size int or tuple (patch_depth, patch_height, patch_width)
        strides: strides along depth, height, and width. If not specified, or
            if `None`, it defaults to the same value as `size`.
        dilation_rate: This is the input stride, specifying how far two
            consecutive patch samples are in the input. Note that using
            `dilation_rate > 1` is not supported in conjunction with
            `strides > 1` on the TensorFlow backend.
        padding: The type of padding algorithm to use: `"same"` or `"valid"`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, depth, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, depth, height, width)`. If not specified,
             the value will default to `keras.config.image_data_format()`.

    Returns:
        Extracted patches 4D (if not batched) or 5D (if batched)

    Examples:

    >>> import numpy as np
    >>> import keras
    >>> # Batched case
    >>> volumes = np.random.random(
    ...     (2, 10, 10, 10, 3)
    ... ).astype("float32") # batch of 2 volumes
    >>> patches = keras.ops.image.extract_patches_3d(volumes, (3, 3, 3))
    >>> patches.shape
    (2, 3, 3, 3, 81)
    >>> # Unbatched case
    >>> volume = np.random.random((10, 10, 10, 3)).astype("float32") # 1 volume
    >>> patches = keras.ops.image.extract_patches_3d(volume, (3, 3, 3))
    >>> patches.shape
    (3, 3, 3, 81)
    """
    # Convert int to 3-tuple for 3D
    if isinstance(size, int):
        size = (size, size, size)
    if any_symbolic_tensors((volumes,)):
        return ExtractPatches(
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=data_format,
        ).symbolic_call(volumes)

    return _extract_patches_3d(
        volumes, size, strides, dilation_rate, padding, data_format=data_format
    )


class ReconstructPatches(Operation):
    def __init__(
        self,
        size,
        output_size=None,
        strides=None,
        padding="valid",
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        if isinstance(size, int):
            size = (size, size)
        self.size = tuple(size)
        self.output_size = (
            tuple(output_size) if output_size is not None else None
        )
        self.is_3d = len(self.size) == 3
        if strides is None:
            strides = self.size
        if isinstance(strides, int):
            strides = (strides,) * len(self.size)
        self.strides = tuple(strides)
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        if self.output_size is None and self.padding != "valid":
            raise ValueError(
                "`output_size=None` (auto-infer) is only supported for "
                "`padding='valid'`. For `padding='same'`, the original "
                "size is ambiguous from patches alone — pass `output_size`."
            )

    def call(self, patches):
        return _reconstruct_patches(
            patches=patches,
            size=self.size,
            output_size=self.output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_spec(self, patches):
        patches_shape = list(patches.shape)
        original_ndim = len(patches_shape)
        flat = (
            patches_shape[-1]
            if self.data_format == "channels_last"
            else (patches_shape[-4] if self.is_3d else patches_shape[-3])
        )
        patch_volume = 1
        for s in self.size:
            patch_volume *= s
        channels_out = None if flat is None else flat // patch_volume

        if self.is_3d:
            expected_ndim_batched = 5
            expected_ndim_unbatched = 4
        else:
            expected_ndim_batched = 4
            expected_ndim_unbatched = 3

        if original_ndim == expected_ndim_batched:
            batch = patches_shape[0]
        elif original_ndim == expected_ndim_unbatched:
            batch = None
        else:
            raise ValueError(
                f"`patches` has unexpected rank for "
                f"{'3D' if self.is_3d else '2D'} reconstruction. "
                f"Expected {expected_ndim_unbatched} (unbatched) or "
                f"{expected_ndim_batched} (batched). "
                f"Received shape: {patches.shape}"
            )

        if flat is not None and flat % patch_volume != 0:
            raise ValueError(
                f"`patches` last dim ({flat}) is not divisible by "
                f"prod(size) ({patch_volume})."
            )

        if self.data_format == "channels_last":
            grid_offset = 1 if original_ndim == expected_ndim_batched else 0
            grid = patches_shape[grid_offset : grid_offset + len(self.size)]
        else:
            # channels_first: the grid dims are the trailing dims,
            # (B, flat, *grid) batched or (flat, *grid) unbatched.
            grid = patches_shape[-len(self.size) :]

        # Resolve the output spatial shape: use `output_size` if given
        # (validating it against static grid dims), else auto-infer from
        # the grid — `__init__` guarantees `padding="valid"` when
        # `output_size` is None. Unknown grid dims stay None.
        if self.output_size is None:
            spatial = tuple(
                (g - 1) * s + k if isinstance(g, int) else None
                for g, s, k in zip(grid, self.strides, self.size)
            )
        else:
            spatial = self.output_size
            dim_names = ("depth", "height", "width")[-len(self.size) :]
            for g, p, s, o, dim_name in zip(
                grid, self.size, self.strides, self.output_size, dim_names
            ):
                if not isinstance(g, int):
                    continue
                if self.padding == "valid":
                    if (g - 1) * s + p != o:
                        raise ValueError(
                            f"`padding='valid'` requires output_size to "
                            f"equal (grid - 1) * stride + size. Got "
                            f"output_size={self.output_size}, grid "
                            f"{dim_name}={g}, stride={s}, size={p}."
                        )
                elif not (g * p - p < o <= g * p):
                    raise ValueError(
                        f"For `padding='same'`, `output_size` {dim_name} "
                        f"({o}) must be in the range ((g-1)*p, g*p], i.e. "
                        f"({g * p - p}, {g * p}]. Got: grid={g}, patch={p}."
                    )

        if self.data_format == "channels_last":
            out_shape = list(spatial) + [channels_out]
        else:
            out_shape = [channels_out] + list(spatial)

        if original_ndim == expected_ndim_batched:
            out_shape = [batch] + out_shape
        return KerasTensor(shape=tuple(out_shape), dtype=patches.dtype)

    def get_config(self):
        return {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }


@keras_export("keras.ops.image.reconstruct_patches")
def reconstruct_patches(
    patches,
    size,
    output_size=None,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Reconstructs image(s) or volume(s) from non-overlapping patches.

    Inverse of `keras.ops.image.extract_patches` for the non-overlapping case
    (`strides == size`).

    Args:
        patches: Patches tensor as produced by `extract_patches`.
            For 2D patches: 3D `(gH, gW, pH*pW*C)` or
            4D `(N, gH, gW, pH*pW*C)`.
            For 3D patches: 4D `(gD, gH, gW, pD*pH*pW*C)` or
            5D `(N, gD, gH, gW, pD*pH*pW*C)`.
            With `data_format="channels_first"` the flat patch dim comes
            first instead: `(pH*pW*C, gH, gW)` / `(N, pH*pW*C, gH, gW)`
            for 2D patches, `(pD*pH*pW*C, gD, gH, gW)` /
            `(N, pD*pH*pW*C, gD, gH, gW)` for 3D patches.
        size: Patch size, matching the `size` used for extraction.
            Length 2 tuple for 2D, length 3 tuple for 3D, or int.
        output_size: Target spatial shape of the reconstruction. Length 2
            tuple `(H, W)` for 2D, length 3 tuple `(D, H, W)` for 3D. With
            `padding="valid"` this may be omitted (`None`) and is then
            inferred from the patch grid; if given, it must equal
            `(grid - 1) * stride + size` per dim — the region covered by
            the extracted patches, i.e. the original size cropped down to
            a multiple of `size` in the non-overlapping case.
            With `padding="same"` it is required: pass the original
            spatial shape so the padding added during extraction can be
            unambiguously removed.
        strides: Currently must equal `size` (non-overlapping). Defaults
            to `size`.
        padding: `"same"` or `"valid"`, matching the extraction.
        data_format: A string specifying the data format of the input tensor.
            Either `"channels_last"` or `"channels_first"`. With
            `"channels_first"` the reconstruction is returned in
            `(C, H, W)` / `(N, C, H, W)` layout.
            If not specified, defaults to `keras.config.image_data_format`.

    Returns:
        Reconstructed image/volume:
            2D: 3D (unbatched) or 4D (batched).
            3D: 4D (unbatched) or 5D (batched).

    Examples:

    >>> image = np.random.random((2, 20, 20, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches(image, (5, 5))
    >>> patches.shape
    (2, 4, 4, 75)
    >>> recon = keras.ops.image.reconstruct_patches(
    ...     patches, size=(5, 5), output_size=(20, 20)
    ... )
    >>> recon.shape
    (2, 20, 20, 3)

    >>> # 3D patches: pass a length-3 `size`
    >>> volume = np.random.random((2, 9, 9, 9, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches(volume, (3, 3, 3))
    >>> recon = keras.ops.image.reconstruct_patches(
    ...     patches, size=(3, 3, 3), output_size=(9, 9, 9)
    ... )
    >>> recon.shape
    (2, 9, 9, 9, 3)
    """
    if not isinstance(size, int):
        if not isinstance(size, (tuple, list)):
            raise TypeError(
                "Invalid `size` argument. Expected an int or a tuple. "
                f"Received: size={size} of type {type(size).__name__}"
            )
        if len(size) not in (2, 3):
            raise ValueError(
                "Invalid `size` argument. Expected a tuple of length 2 or 3. "
                f"Received: size={size} with length {len(size)}"
            )
    if output_size is not None and not isinstance(output_size, (tuple, list)):
        raise TypeError(
            "Invalid `output_size` argument. Expected a tuple or list. "
            f"Received: output_size={output_size} of type "
            f"{type(output_size).__name__}"
        )

    if any_symbolic_tensors((patches,)):
        return ReconstructPatches(
            size=size,
            output_size=output_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        ).symbolic_call(patches)

    return _reconstruct_patches(
        patches,
        size,
        output_size,
        strides,
        padding,
        data_format=data_format,
    )


def _reconstruct_patches(
    patches,
    size,
    output_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    if not isinstance(size, int) and len(size) == 3:
        return _reconstruct_patches_3d(
            patches,
            size,
            output_size,
            strides,
            padding,
            data_format,
        )
    return _reconstruct_patches_2d(
        patches,
        size,
        output_size,
        strides,
        padding,
        data_format,
    )


def _validate_reconstruct_strides(size, strides, fn_name):
    if strides is None:
        return tuple(size)
    if isinstance(strides, int):
        strides = (strides,) * len(size)
    if tuple(strides) != tuple(size):
        raise NotImplementedError(
            f"`{fn_name}` currently supports only non-overlapping "
            f"reconstruction (strides == size). Got strides={strides} "
            f"and size={size}. Overlap-add reconstruction is planned for "
            f"a follow-up."
        )
    return tuple(strides)


def _infer_output_size_valid(patches, size, strides, data_format):
    """Infer `output_size` from the patch grid for `padding='valid'`.

    The forward `extract_patches` with `padding='valid'` produces a grid of
    `g = (input - size) // stride + 1`, so the smallest input that yields
    grid `g` is `(g - 1) * stride + size`. Requires statically-known grid
    dims; raises otherwise so the caller can pass `output_size` explicitly.
    """
    rank = len(patches.shape)
    n = len(size)
    if data_format == "channels_last":
        grid_start = 0 if rank == n + 1 else 1
    else:
        grid_start = 1 if rank == n + 1 else 2
    grid = patches.shape[grid_start : grid_start + n]
    if any(not isinstance(g, int) for g in grid):
        raise ValueError(
            "Cannot auto-infer `output_size` for `padding='valid'`: at "
            "least one patch-grid dimension is unknown "
            f"(patches.shape={patches.shape}). Pass `output_size` explicitly."
        )
    return tuple((g - 1) * s + k for g, s, k in zip(grid, strides, size))


def _reconstruct_patches_2d(
    patches,
    size,
    output_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError(
            "Invalid `size`. Expected length 2 for 2D reconstruction. "
            f"Got: size={size}"
        )
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    if len(patches.shape) not in (3, 4):
        raise ValueError(
            "`patches` has unexpected rank for 2D reconstruction. "
            "Expected 3 (unbatched) or 4 (batched). "
            f"Received shape: {patches.shape}"
        )
    strides = _validate_reconstruct_strides(
        size, strides, "reconstruct_patches"
    )
    data_format = backend.standardize_data_format(data_format)
    if output_size is None:
        if padding != "valid":
            raise ValueError(
                "`output_size=None` (auto-infer) is only supported for "
                "`padding='valid'`. For `padding='same'`, the original "
                "size is ambiguous from patches alone — pass `output_size`."
            )
        output_size = _infer_output_size_valid(
            patches, size, strides, data_format
        )
    if len(output_size) != 2:
        raise ValueError(
            "Invalid `output_size`. Expected length 2 (H, W). "
            f"Got: output_size={output_size}"
        )
    if data_format == "channels_first":
        # Reconstruct in channels_last layout, then move channels back.
        # Patches are (flat, gH, gW) unbatched or (B, flat, gH, gW) batched.
        if len(patches.shape) == 3:
            patches = backend.ops.numpy.transpose(patches, axes=(1, 2, 0))
        else:
            patches = backend.ops.numpy.transpose(patches, axes=(0, 2, 3, 1))
        result = _reconstruct_patches_2d(
            patches, size, output_size, strides, padding, "channels_last"
        )
        if len(result.shape) == 3:
            return backend.ops.numpy.transpose(result, axes=(2, 0, 1))
        return backend.ops.numpy.transpose(result, axes=(0, 3, 1, 2))

    pH, pW = size
    H, W = output_size

    _unbatched = False
    if len(patches.shape) == 3:
        _unbatched = True
        patches = backend.ops.numpy.expand_dims(patches, axis=0)

    shp = ops.shape(patches)
    B, gH, gW = shp[0], shp[1], shp[2]
    static_flat = patches.shape[-1]
    if static_flat is None:
        C = shp[3] // (pH * pW)
    else:
        if static_flat % (pH * pW) != 0:
            raise ValueError(
                f"`patches` last dim ({static_flat}) is not divisible by "
                f"prod(size) ({pH * pW})."
            )
        C = static_flat // (pH * pW)

    x = backend.ops.numpy.reshape(patches, (B, gH, gW, pH, pW, C))
    x = backend.ops.numpy.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    x = backend.ops.numpy.reshape(x, (B, gH * pH, gW * pW, C))

    if padding == "same":
        static_gH = patches.shape[1]
        static_gW = patches.shape[2]
        if isinstance(static_gH, int) and not (
            static_gH * pH - pH < H <= static_gH * pH
        ):
            raise ValueError(
                f"For `padding='same'`, `output_size` height ({H}) must "
                f"be in the range ((gH-1)*pH, gH*pH], i.e. "
                f"({static_gH * pH - pH}, {static_gH * pH}]. "
                f"Got: gH={static_gH}, pH={pH}."
            )
        if isinstance(static_gW, int) and not (
            static_gW * pW - pW < W <= static_gW * pW
        ):
            raise ValueError(
                f"For `padding='same'`, `output_size` width ({W}) must "
                f"be in the range ((gW-1)*pW, gW*pW], i.e. "
                f"({static_gW * pW - pW}, {static_gW * pW}]. "
                f"Got: gW={static_gW}, pW={pW}."
            )
        pad_total_h = gH * pH - H
        pad_total_w = gW * pW - W
        begin = [0, pad_total_h // 2, pad_total_w // 2, 0]
        out_shape = [B, H, W, C]
        x = ops.slice(x, begin, out_shape)
    else:
        sH, sW = strides
        if (gH - 1) * sH + pH != H or (gW - 1) * sW + pW != W:
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"(grid - 1) * stride + size. Got output_size=({H},{W}), "
                f"grid=({gH},{gW}), strides=({sH},{sW}), "
                f"size=({pH},{pW})."
            )

    if _unbatched:
        x = backend.ops.numpy.squeeze(x, axis=0)
    return x


def _reconstruct_patches_3d(
    patches,
    size,
    output_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    if len(patches.shape) not in (4, 5):
        raise ValueError(
            "`patches` has unexpected rank for 3D reconstruction. "
            "Expected 4 (unbatched) or 5 (batched). "
            f"Received shape: {patches.shape}"
        )
    strides = _validate_reconstruct_strides(
        size, strides, "reconstruct_patches"
    )
    data_format = backend.standardize_data_format(data_format)
    if output_size is None:
        if padding != "valid":
            raise ValueError(
                "`output_size=None` (auto-infer) is only supported for "
                "`padding='valid'`. For `padding='same'`, the original "
                "size is ambiguous from patches alone — pass `output_size`."
            )
        output_size = _infer_output_size_valid(
            patches, size, strides, data_format
        )
    if len(output_size) != 3:
        raise ValueError(
            "Invalid `output_size`. Expected length 3 (D, H, W). "
            f"Got: output_size={output_size}"
        )
    if data_format == "channels_first":
        # Reconstruct in channels_last layout, then move channels back.
        # Patches are (flat, gD, gH, gW) unbatched or (B, flat, gD, gH, gW).
        if len(patches.shape) == 4:
            patches = backend.ops.numpy.transpose(patches, axes=(1, 2, 3, 0))
        else:
            patches = backend.ops.numpy.transpose(patches, axes=(0, 2, 3, 4, 1))
        result = _reconstruct_patches_3d(
            patches, size, output_size, strides, padding, "channels_last"
        )
        if len(result.shape) == 4:
            return backend.ops.numpy.transpose(result, axes=(3, 0, 1, 2))
        return backend.ops.numpy.transpose(result, axes=(0, 4, 1, 2, 3))

    pD, pH, pW = size
    D, H, W = output_size

    _unbatched = False
    if len(patches.shape) == 4:
        _unbatched = True
        patches = backend.ops.numpy.expand_dims(patches, axis=0)

    shp = ops.shape(patches)
    B, gD, gH, gW = shp[0], shp[1], shp[2], shp[3]
    static_flat = patches.shape[-1]
    if static_flat is None:
        C = shp[4] // (pD * pH * pW)
    else:
        if static_flat % (pD * pH * pW) != 0:
            raise ValueError(
                f"`patches` last dim ({static_flat}) is not divisible by "
                f"prod(size) ({pD * pH * pW}). Are `size` and the patches "
                f"tensor consistent?"
            )
        C = static_flat // (pD * pH * pW)

    x = backend.ops.numpy.reshape(patches, (B, gD, gH, gW, pD, pH, pW, C))
    x = backend.ops.numpy.transpose(x, axes=(0, 1, 4, 2, 5, 3, 6, 7))
    x = backend.ops.numpy.reshape(x, (B, gD * pD, gH * pH, gW * pW, C))

    if padding == "same":
        static_gD = patches.shape[1]
        static_gH = patches.shape[2]
        static_gW = patches.shape[3]
        if isinstance(static_gD, int) and not (
            static_gD * pD - pD < D <= static_gD * pD
        ):
            raise ValueError(
                f"For `padding='same'`, `output_size` depth ({D}) must "
                f"be in the range ((gD-1)*pD, gD*pD], i.e. "
                f"({static_gD * pD - pD}, {static_gD * pD}]. "
                f"Got: gD={static_gD}, pD={pD}."
            )
        if isinstance(static_gH, int) and not (
            static_gH * pH - pH < H <= static_gH * pH
        ):
            raise ValueError(
                f"For `padding='same'`, `output_size` height ({H}) must "
                f"be in the range ((gH-1)*pH, gH*pH], i.e. "
                f"({static_gH * pH - pH}, {static_gH * pH}]. "
                f"Got: gH={static_gH}, pH={pH}."
            )
        if isinstance(static_gW, int) and not (
            static_gW * pW - pW < W <= static_gW * pW
        ):
            raise ValueError(
                f"For `padding='same'`, `output_size` width ({W}) must "
                f"be in the range ((gW-1)*pW, gW*pW], i.e. "
                f"({static_gW * pW - pW}, {static_gW * pW}]. "
                f"Got: gW={static_gW}, pW={pW}."
            )
        pad_total_d = gD * pD - D
        pad_total_h = gH * pH - H
        pad_total_w = gW * pW - W
        begin = [
            0,
            pad_total_d // 2,
            pad_total_h // 2,
            pad_total_w // 2,
            0,
        ]
        out_shape = [B, D, H, W, C]
        x = ops.slice(x, begin, out_shape)
    else:
        sD, sH, sW = strides
        if (
            (gD - 1) * sD + pD != D
            or (gH - 1) * sH + pH != H
            or (gW - 1) * sW + pW != W
        ):
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"(grid - 1) * stride + size. Got output_size=({D},{H},{W}), "
                f"grid=({gD},{gH},{gW}), strides=({sD},{sH},{sW}), "
                f"size=({pD},{pH},{pW})."
            )

    if _unbatched:
        x = backend.ops.numpy.squeeze(x, axis=0)
    return x


class MapCoordinates(Operation):
    def __init__(self, order, fill_mode="constant", fill_value=0, *, name=None):
        super().__init__(name=name)
        self.order = order
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def call(self, inputs, coordinates):
        return backend.ops.image.map_coordinates(
            inputs,
            coordinates,
            order=self.order,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

    def compute_output_spec(self, inputs, coordinates):
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
        return KerasTensor(coordinates.shape[1:], dtype=inputs.dtype)


@keras_export("keras.ops.image.map_coordinates")
def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    """Map the input array to new coordinates by interpolation.

    Note that interpolation near boundaries differs from the scipy function,
    because we fixed an outstanding bug
    [scipy/issues/2640](https://github.com/scipy/scipy/issues/2640).

    Args:
        inputs: The input array.
        coordinates: The coordinates at which inputs is evaluated.
        order: The order of the spline interpolation. The order must be `0` or
            `1`. `0` indicates the nearest neighbor and `1` indicates the linear
            interpolation.
        fill_mode: Points outside the boundaries of the inputs are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"mirror"` and `"reflect"`. Defaults to
            `"constant"`.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The inputs is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The inputs is extended by the nearest pixel.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The inputs is extended by wrapping around to the opposite edge.
            - `"mirror"`: `(c d c b | a b c d | c b a b)`
                The inputs is extended by mirroring about the edge.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The inputs is extended by reflecting about the edge of the last
                pixel.
        fill_value: Value used for points outside the boundaries of the inputs
            if `fill_mode="constant"`. Defaults to `0`.

    Returns:
        Output input or batch of inputs.

    """
    if any_symbolic_tensors((inputs, coordinates)):
        return MapCoordinates(
            order,
            fill_mode,
            fill_value,
        ).symbolic_call(inputs, coordinates)
    return backend.ops.image.map_coordinates(
        inputs,
        coordinates,
        order,
        fill_mode,
        fill_value,
    )


def _validate_non_negative(value, name):
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0. Received: {name}={value}")


def _validate_pad_images_args(
    top_padding,
    left_padding,
    bottom_padding,
    right_padding,
    target_height,
    target_width,
):
    if [top_padding, bottom_padding, target_height].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "top_padding, bottom_padding, target_height. "
            f"Received: top_padding={top_padding}, "
            f"bottom_padding={bottom_padding}, "
            f"target_height={target_height}"
        )
    if [left_padding, right_padding, target_width].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "left_padding, right_padding, target_width. "
            f"Received: left_padding={left_padding}, "
            f"right_padding={right_padding}, "
            f"target_width={target_width}"
        )

    _validate_non_negative(top_padding, "top_padding")
    _validate_non_negative(bottom_padding, "bottom_padding")
    _validate_non_negative(target_height, "target_height")
    _validate_non_negative(left_padding, "left_padding")
    _validate_non_negative(right_padding, "right_padding")
    _validate_non_negative(target_width, "target_width")


def _validate_crop_images_args(
    top_cropping,
    left_cropping,
    bottom_cropping,
    right_cropping,
    target_height,
    target_width,
):
    if [top_cropping, bottom_cropping, target_height].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "top_cropping, bottom_cropping, target_height. "
            f"Received: top_cropping={top_cropping}, "
            f"bottom_cropping={bottom_cropping}, "
            f"target_height={target_height}"
        )
    if [left_cropping, right_cropping, target_width].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "left_cropping, right_cropping, target_width. "
            f"Received: left_cropping={left_cropping}, "
            f"right_cropping={right_cropping}, "
            f"target_width={target_width}"
        )

    _validate_non_negative(top_cropping, "top_cropping")
    _validate_non_negative(bottom_cropping, "bottom_cropping")
    _validate_non_negative(target_height, "target_height")
    _validate_non_negative(left_cropping, "left_cropping")
    _validate_non_negative(right_cropping, "right_cropping")
    _validate_non_negative(target_width, "target_width")


class PadImages(Operation):
    def __init__(
        self,
        top_padding=None,
        left_padding=None,
        bottom_padding=None,
        right_padding=None,
        target_height=None,
        target_width=None,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.top_padding = top_padding
        self.left_padding = left_padding
        self.bottom_padding = bottom_padding
        self.right_padding = right_padding
        self.target_height = target_height
        self.target_width = target_width
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _pad_images(
            images,
            self.top_padding,
            self.left_padding,
            self.bottom_padding,
            self.right_padding,
            self.target_height,
            self.target_width,
            self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        _validate_pad_images_args(
            self.top_padding,
            self.left_padding,
            self.bottom_padding,
            self.right_padding,
            self.target_height,
            self.target_width,
        )

        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
            height, width = images_shape[height_axis], images_shape[width_axis]
        else:
            height_axis, width_axis = -2, -1
            height, width = images_shape[height_axis], images_shape[width_axis]

        target_height = self.target_height
        if target_height is None and height is not None:
            target_height = self.top_padding + height + self.bottom_padding
        target_width = self.target_width
        if target_width is None and width is not None:
            target_width = self.left_padding + width + self.right_padding

        if height is not None:
            top_padding = self.top_padding
            bottom_padding = self.bottom_padding
            if top_padding is None:
                top_padding = target_height - height - bottom_padding
            if bottom_padding is None:
                bottom_padding = target_height - height - top_padding
            if top_padding < 0:
                raise ValueError(
                    "top_padding must be >= 0. "
                    f"Received: top_padding={top_padding}"
                )
            if bottom_padding < 0:
                raise ValueError(
                    "bottom_padding must be >= 0. "
                    f"Received: bottom_padding={bottom_padding}"
                )
            if target_height < 0:
                raise ValueError(
                    "target_height must be >= 0. "
                    f"Received: target_height={target_height}"
                )

        if width is not None:
            left_padding = self.left_padding
            right_padding = self.right_padding
            if left_padding is None:
                left_padding = target_width - width - right_padding
            if right_padding is None:
                right_padding = target_width - width - left_padding
            if left_padding < 0:
                raise ValueError(
                    "left_padding must be >= 0. "
                    f"Received: left_padding={left_padding}"
                )
            if right_padding < 0:
                raise ValueError(
                    "right_padding must be >= 0. "
                    f"Received: right_padding={right_padding}"
                )
            if target_width < 0:
                raise ValueError(
                    "target_width must be >= 0. "
                    f"Received: target_width={target_width}"
                )

        images_shape[height_axis] = target_height
        images_shape[width_axis] = target_width
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.pad_images")
def pad_images(
    images,
    top_padding=None,
    left_padding=None,
    bottom_padding=None,
    right_padding=None,
    target_height=None,
    target_width=None,
    data_format=None,
):
    """Pad `images` with zeros to the specified `height` and `width`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        top_padding: Number of rows of zeros to add on top.
        left_padding: Number of columns of zeros to add on the left.
        bottom_padding: Number of rows of zeros to add at the bottom.
        right_padding: Number of columns of zeros to add on the right.
        target_height: Height of output images.
        target_width: Width of output images.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Padded image or batch of images.

    Example:

    >>> images = np.random.random((15, 25, 3))
    >>> padded_images = keras.ops.image.pad_images(
    ...     images, 2, 3, target_height=20, target_width=30
    ... )
    >>> padded_images.shape
    (20, 30, 3)

    >>> batch_images = np.random.random((2, 15, 25, 3))
    >>> padded_batch = keras.ops.image.pad_images(
    ...     batch_images, 2, 3, target_height=20, target_width=30
    ... )
    >>> padded_batch.shape
    (2, 20, 30, 3)"""

    if any_symbolic_tensors((images,)):
        return PadImages(
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
            data_format,
        ).symbolic_call(images)

    return _pad_images(
        images,
        top_padding,
        left_padding,
        bottom_padding,
        right_padding,
        target_height,
        target_width,
        data_format,
    )


def _pad_images(
    images,
    top_padding,
    left_padding,
    bottom_padding,
    right_padding,
    target_height,
    target_width,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    images = backend.ops.convert_to_tensor(images)
    images_shape = ops.shape(images)

    # Check
    if len(images_shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). "
            f"Received: images.shape={images_shape}"
        )
    _validate_pad_images_args(
        top_padding,
        left_padding,
        bottom_padding,
        right_padding,
        target_height,
        target_width,
    )

    is_batch = False if len(images_shape) == 3 else True
    if data_format == "channels_last":
        height, width = images_shape[-3], images_shape[-2]
    else:
        height, width = images_shape[-2], images_shape[-1]

    # Infer padding
    if top_padding is None:
        top_padding = target_height - bottom_padding - height
    if bottom_padding is None:
        bottom_padding = target_height - top_padding - height
    if left_padding is None:
        left_padding = target_width - right_padding - width
    if right_padding is None:
        right_padding = target_width - left_padding - width

    if top_padding < 0:
        raise ValueError(
            f"top_padding must be >= 0. Received: top_padding={top_padding}"
        )
    if left_padding < 0:
        raise ValueError(
            f"left_padding must be >= 0. Received: left_padding={left_padding}"
        )
    if right_padding < 0:
        raise ValueError(
            "right_padding must be >= 0. "
            f"Received: right_padding={right_padding}"
        )
    if bottom_padding < 0:
        raise ValueError(
            "bottom_padding must be >= 0. "
            f"Received: bottom_padding={bottom_padding}"
        )

    # Compute pad_width
    pad_width = [[top_padding, bottom_padding], [left_padding, right_padding]]
    if data_format == "channels_last":
        pad_width = pad_width + [[0, 0]]
    else:
        pad_width = [[0, 0]] + pad_width
    if is_batch:
        pad_width = [[0, 0]] + pad_width

    padded_images = backend.ops.numpy.pad(images, pad_width)
    return padded_images


class CropImages(Operation):
    def __init__(
        self,
        top_cropping=None,
        left_cropping=None,
        bottom_cropping=None,
        right_cropping=None,
        target_height=None,
        target_width=None,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.top_cropping = top_cropping
        self.bottom_cropping = bottom_cropping
        self.left_cropping = left_cropping
        self.right_cropping = right_cropping
        self.target_height = target_height
        self.target_width = target_width
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _crop_images(
            images,
            self.top_cropping,
            self.left_cropping,
            self.bottom_cropping,
            self.right_cropping,
            self.target_height,
            self.target_width,
            self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        _validate_crop_images_args(
            self.top_cropping,
            self.left_cropping,
            self.bottom_cropping,
            self.right_cropping,
            self.target_height,
            self.target_width,
        )

        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
        else:
            height_axis, width_axis = -2, -1
        height, width = images_shape[height_axis], images_shape[width_axis]

        if height is None and self.target_height is None:
            raise ValueError(
                "When the height of the images is unknown, `target_height` "
                "must be specified."
                f"Received images.shape={images_shape} and "
                f"target_height={self.target_height}"
            )
        if width is None and self.target_width is None:
            raise ValueError(
                "When the width of the images is unknown, `target_width` "
                "must be specified."
                f"Received images.shape={images_shape} and "
                f"target_width={self.target_width}"
            )

        target_height = self.target_height
        if target_height is None:
            target_height = height - self.top_cropping - self.bottom_cropping
        target_width = self.target_width
        if target_width is None:
            target_width = width - self.left_cropping - self.right_cropping

        if height is not None:
            top_cropping = self.top_cropping
            bottom_cropping = self.bottom_cropping
            if top_cropping is None:
                top_cropping = height - target_height - bottom_cropping
            if bottom_cropping is None:
                bottom_cropping = height - target_height - top_cropping
            if top_cropping < 0:
                raise ValueError(
                    "top_cropping must be >= 0. "
                    f"Received: top_cropping={top_cropping}"
                )
            if bottom_cropping < 0:
                raise ValueError(
                    "bottom_cropping must be >= 0. "
                    f"Received: bottom_cropping={bottom_cropping}"
                )
            if target_height < 0:
                raise ValueError(
                    "target_height must be >= 0. "
                    f"Received: target_height={target_height}"
                )

        if width is not None:
            left_cropping = self.left_cropping
            right_cropping = self.right_cropping
            if left_cropping is None:
                left_cropping = width - target_width - right_cropping
            if right_cropping is None:
                right_cropping = width - target_width - left_cropping
            if left_cropping < 0:
                raise ValueError(
                    "left_cropping must be >= 0. "
                    f"Received: left_cropping={left_cropping}"
                )
            if right_cropping < 0:
                raise ValueError(
                    "right_cropping must be >= 0. "
                    f"Received: right_cropping={right_cropping}"
                )
            if target_width < 0:
                raise ValueError(
                    "target_width must be >= 0. "
                    f"Received: target_width={target_width}"
                )

        images_shape[height_axis] = target_height
        images_shape[width_axis] = target_width
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.crop_images")
def crop_images(
    images,
    top_cropping=None,
    left_cropping=None,
    bottom_cropping=None,
    right_cropping=None,
    target_height=None,
    target_width=None,
    data_format=None,
):
    """Crop `images` to a specified `height` and `width`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        top_cropping: Number of columns to crop from the top.
        left_cropping: Number of columns to crop from the left.
        bottom_cropping: Number of columns to crop from the bottom.
        right_cropping: Number of columns to crop from the right.
        target_height: Height of the output images.
        target_width: Width of the output images.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Cropped image or batch of images.

    Example:

    >>> images = np.reshape(np.arange(1, 28, dtype="float32"), [3, 3, 3])
    >>> images[:,:,0] # print the first channel of the images
    array([[ 1.,  4.,  7.],
           [10., 13., 16.],
           [19., 22., 25.]], dtype=float32)
    >>> cropped_images = keras.image.crop_images(images, 0, 0, 2, 2)
    >>> cropped_images[:,:,0] # print the first channel of the cropped images
    array([[ 1.,  4.],
           [10., 13.]], dtype=float32)"""

    if any_symbolic_tensors((images,)):
        return CropImages(
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
            data_format,
        ).symbolic_call(images)

    return _crop_images(
        images,
        top_cropping,
        left_cropping,
        bottom_cropping,
        right_cropping,
        target_height,
        target_width,
        data_format,
    )


def _crop_images(
    images,
    top_cropping,
    left_cropping,
    bottom_cropping,
    right_cropping,
    target_height,
    target_width,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    images = backend.ops.convert_to_tensor(images)
    images_shape = ops.shape(images)

    # Check
    if len(images_shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). "
            f"Received: images.shape={images_shape}"
        )
    _validate_crop_images_args(
        top_cropping,
        left_cropping,
        bottom_cropping,
        right_cropping,
        target_height,
        target_width,
    )

    is_batch = False if len(images_shape) == 3 else True
    if data_format == "channels_last":
        height, width = images_shape[-3], images_shape[-2]
        channels = images_shape[-1]
    else:
        height, width = images_shape[-2], images_shape[-1]
        channels = images_shape[-3]

    # Infer padding
    if top_cropping is None:
        top_cropping = height - target_height - bottom_cropping
    if target_height is None:
        target_height = height - bottom_cropping - top_cropping
    if left_cropping is None:
        left_cropping = width - target_width - right_cropping
    if target_width is None:
        target_width = width - right_cropping - left_cropping

    if top_cropping < 0:
        raise ValueError(
            f"top_cropping must be >= 0. Received: top_cropping={top_cropping}"
        )
    if target_height < 0:
        raise ValueError(
            "target_height must be >= 0. "
            f"Received: target_height={target_height}"
        )
    if left_cropping < 0:
        raise ValueError(
            "left_cropping must be >= 0. "
            f"Received: left_cropping={left_cropping}"
        )
    if target_width < 0:
        raise ValueError(
            f"target_width must be >= 0. Received: target_width={target_width}"
        )

    # Compute start_indices and shape
    start_indices = [top_cropping, left_cropping]
    shape = [target_height, target_width]
    if data_format == "channels_last":
        start_indices = start_indices + [0]
        shape = shape + [channels]
    else:
        start_indices = [0] + start_indices
        shape = [channels] + shape
    if is_batch:
        batch_size = images_shape[0]
        start_indices = [0] + start_indices
        shape = [batch_size] + shape

    cropped_images = ops.slice(images, start_indices, shape)
    return cropped_images


class PerspectiveTransform(Operation):
    def __init__(
        self,
        interpolation="bilinear",
        fill_value=0,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images, start_points, end_points):
        return backend.ops.image.perspective_transform(
            images,
            start_points,
            end_points,
            interpolation=self.interpolation,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images, start_points, end_points):
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
        return KerasTensor(images.shape, dtype=images.dtype)


@keras_export("keras.ops.image.perspective_transform")
def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    """Applies a perspective transformation to the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        start_points: A tensor of shape `(N, 4, 2)` or `(4, 2)`,
            representing the source points in the original image
            that define the transformation.
        end_points: A tensor of shape `(N, 4, 2)` or `(4, 2)`,
            representing the target points in the output image
            after transformation.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            and `"bilinear"`. Defaults to `"bilinear"`.
        fill_value: Value used for points outside the boundaries of the input if
            extrapolation is needed. Defaults to `0`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Applied perspective transform image or batch of images.

    Examples:

    >>> x = np.random.random((2, 64, 80, 3))  # batch of 2 RGB images
    >>> start_points = np.array(
    ...     [
    ...         [[0, 0], [0, 64], [80, 0], [80, 64]],
    ...         [[0, 0], [0, 64], [80, 0], [80, 64]],
    ...     ]
    ... )
    >>> end_points = np.array(
    ...     [
    ...         [[3, 5], [7, 64], [76, -10], [84, 61]],
    ...         [[8, 10], [10, 61], [65, 3], [88, 43]],
    ...     ]
    ... )
    >>> y = keras.ops.image.perspective_transform(x, start_points, end_points)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3))  # single RGB image
    >>> start_points = np.array([[0, 0], [0, 64], [80, 0], [80, 64]])
    >>> end_points = np.array([[3, 5], [7, 64], [76, -10], [84, 61]])
    >>> y = keras.ops.image.perspective_transform(x, start_points, end_points)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80))  # batch of 2 RGB images
    >>> start_points = np.array(
    ...     [
    ...         [[0, 0], [0, 64], [80, 0], [80, 64]],
    ...         [[0, 0], [0, 64], [80, 0], [80, 64]],
    ...     ]
    ... )
    >>> end_points = np.array(
    ...     [
    ...         [[3, 5], [7, 64], [76, -10], [84, 61]],
    ...         [[8, 10], [10, 61], [65, 3], [88, 43]],
    ...     ]
    ... )
    >>> y = keras.ops.image.perspective_transform(
    ...     x, start_points, end_points, data_format="channels_first"
    ... )
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((images, start_points, end_points)):
        return PerspectiveTransform(
            interpolation=interpolation,
            fill_value=fill_value,
            data_format=data_format,
        ).symbolic_call(images, start_points, end_points)
    return backend.ops.image.perspective_transform(
        images,
        start_points,
        end_points,
        interpolation=interpolation,
        fill_value=fill_value,
        data_format=data_format,
    )


class GaussianBlur(Operation):
    def __init__(
        self,
        kernel_size=(3, 3),
        sigma=(1.0, 1.0),
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.gaussian_blur(
            images,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images):
        if len(images.shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"images.shape={images.shape}"
            )
        return KerasTensor(images.shape, dtype=images.dtype)


@keras_export("keras.ops.image.gaussian_blur")
def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    """Applies a Gaussian blur to the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        kernel_size: A tuple of two integers, specifying the height and width
            of the Gaussian kernel.
        sigma: A tuple of two floats, specifying the standard deviation of
            the Gaussian kernel along height and width.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Blurred image or batch of images.

    Examples:

    >>> x = np.random.random((2, 64, 80, 3))  # batch of 2 RGB images
    >>> y = keras.ops.image.gaussian_blur(x)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3))  # single RGB image
    >>> y = keras.ops.image.gaussian_blur(x)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80))  # batch of 2 RGB images
    >>> y = keras.ops.image.gaussian_blur(
    ...     x, data_format="channels_first")
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((images,)):
        return GaussianBlur(
            kernel_size=kernel_size,
            sigma=sigma,
            data_format=data_format,
        ).symbolic_call(images)
    return backend.ops.image.gaussian_blur(
        images,
        kernel_size=kernel_size,
        sigma=sigma,
        data_format=data_format,
    )


class ElasticTransform(Operation):
    def __init__(
        self,
        alpha=20.0,
        sigma=5.0,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        seed=None,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.elastic_transform(
            images,
            alpha=self.alpha,
            sigma=self.sigma,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            seed=self.seed,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images):
        if len(images.shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"images.shape={images.shape}"
            )
        return KerasTensor(images.shape, dtype=images.dtype)


@keras_export("keras.ops.image.elastic_transform")
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
    """Applies elastic deformation to the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        alpha: Scaling factor that controls the intensity of the deformation.
        sigma: Standard deviation of the Gaussian filter used for
            smoothing the displacement fields.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            and `"bilinear"`. Defaults to `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"reflect"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        fill_value: Value used for points outside the boundaries of the input if
            `fill_mode="constant"`. Defaults to `0`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Transformed image or batch of images with elastic deformation.

    Examples:

    >>> x = np.random.random((2, 64, 80, 3))  # batch of 2 RGB images
    >>> y = keras.ops.image.elastic_transform(x)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3))  # single RGB image
    >>> y = keras.ops.image.elastic_transform(x)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80))  # batch of 2 RGB images
    >>> y = keras.ops.image.elastic_transform(
    ...     x, data_format="channels_first")
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((images,)):
        return ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
            seed=seed,
            data_format=data_format,
        ).symbolic_call(images)
    return backend.ops.image.elastic_transform(
        images,
        alpha=alpha,
        sigma=sigma,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        seed=seed,
        data_format=data_format,
    )


class ScaleAndTranslate(Operation):
    def __init__(self, spatial_dims, method, antialias=True, *, name=None):
        super().__init__(name=name)
        self.spatial_dims = spatial_dims
        self.method = method
        self.antialias = antialias

    def call(self, images, output_shape, scale, translation):
        return backend.ops.image.scale_and_translate(
            images,
            output_shape=output_shape,
            scale=scale,
            translation=translation,
            spatial_dims=self.spatial_dims,
            method=self.method,
            antialias=self.antialias,
        )

    def compute_output_spec(self, images, output_shape, scale, translation):
        return KerasTensor(output_shape, dtype=images.dtype)


@keras_export("keras.ops.image.scale_and_translate")
def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    """Apply a scale and translation to the images.

    Generates a new image of `output_shape` by resampling from the input image
    using the sampling method corresponding to method. For 2D images, this
    operation transforms a location in the input images, (x, y), to a location
    in the output image according to:

    `(x * scale[1] + translation[1], y * scale[0] + translation[0])`.

    (Note the inverse warp is used to generate the sample locations.) Assumes
    half-centered pixels, i.e the pixel at integer location row, col has
    coordinates y, x = row + 0.5, col + 0.5, and similarly for other input image
    dimensions.

    If an output location(pixel) maps to an input sample location that is
    outside the input boundaries then the value for the output location will be
    set to zero.

    The `method` argument expects one of the following resize methods:

    - `"linear"`, `"bilinear"`, `"trilinear"`, `"triangle"`: Linear
        interpolation. If `antialias` is True, uses a triangular filter when
        downsampling.
    - `"cubic"`, `"bicubic"`, `"tricubic"`: Cubic interpolation, using the Keys
        cubic kernel.
    - `"lanczos3"`: Lanczos resampling, using a kernel of radius 3.
    - `"lanczos5"`: Lanczos resampling, using a kernel of radius 5.

    Args:
        images: The input array.
        output_shape: The output shape, as a sequence of integers with length
            equal to the number of dimensions of image.
        scale: A [K] array with the same number of dimensions as `images`,
            containing the scale to apply in each dimension.
        translation: A [K] array with the same number of dimensions as `images`,
            containing the translation to apply in each dimension.
        spatial_dims: A length K tuple specifying the spatial dimensions that
            the passed `scale` and `translation` should be applied to.
        method: A string specifying the resizing method to use. Available
            methods are `"linear"`, `"bilinear"`, `"trilinear"`, `"triangle"`,
            `"cubic"`, `"bicubic"`, `"tricubic"`, `"lanczos3"` and `"lanczos5"`.
        antialias: Whether an antialiasing filter should be applied when
            downsampling. Has no effect when upsampling. Defaults to `True`.

    Returns:
        The scale and translated images.

    Example:

    >>> images = np.arange(9, dtype="float32").reshape((3, 3))
    >>> scale = np.array([2.0, 2.0]).astype("float32")
    >>> translation = -(scale / 2.0 - 0.5)
    >>> resized_images = keras.image.scale_and_translate(
    ...     images, (5, 5), scale, translation, (0, 1), "linear"
    ... )
    >>> resized_images
    array([[0.0 0.5 1.0 1.5 2.0]
           [1.5 2.0 2.5 3.0 3.5]
           [3.0 3.5 4.0 4.5 5.0]
           [4.5 5.0 5.5 6.0 6.5]
           [6.0 6.5 7.0 7.5 8.0]], dtype=float32)
    """
    if any_symbolic_tensors((images, scale, translation)):
        return ScaleAndTranslate(spatial_dims, method, antialias).symbolic_call(
            images, output_shape, scale, translation
        )
    return backend.ops.image.scale_and_translate(
        images,
        output_shape,
        scale,
        translation,
        spatial_dims,
        method,
        antialias,
    )


class SobelEdges(Operation):
    def __init__(self, data_format=None, *, name=None):
        super().__init__(name=name)
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.ops.image.sobel_edges(
            images, data_format=self.data_format
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) != 4:
            raise ValueError(
                "Invalid images rank: expected rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        # Output adds an extra dimension of size 2 for [dy, dx]
        output_shape = images_shape + [2]
        return KerasTensor(shape=output_shape, dtype=images.dtype)


@keras_export("keras.ops.image.sobel_edges")
def sobel_edges(images, data_format=None):
    """Computes Sobel edge detection on images.

    The Sobel operator computes the gradient of the image intensity at each
    pixel, giving the direction of the largest increase from light to dark
    and the rate of change in that direction.

    Args:
        images: Input tensor of shape `(batch, height, width, channels)` if
            `data_format="channels_last"`, or
            `(batch, channels, height, width)` if
            `data_format="channels_first"`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        A tensor containing the Sobel edges. For `data_format="channels_last"`,
        the output shape is `(batch, height, width, channels, 2)` where the
        last dimension contains `[dy, dx]` representing the vertical and
        horizontal gradients. For `data_format="channels_first"`, the output
        shape is `(batch, channels, height, width, 2)`.

    Example:

    >>> import numpy as np
    >>> from keras import ops
    >>> # Create image with vertical edge
    >>> image = np.zeros((1, 8, 8, 1), dtype="float32")
    >>> image[0, :, 4:, 0] = 1.0
    >>> edges = ops.image.sobel_edges(image)
    >>> edges.shape
    (1, 8, 8, 1, 2)
    """
    if any_symbolic_tensors((images,)):
        return SobelEdges(data_format=data_format).symbolic_call(images)
    return backend.ops.image.sobel_edges(
        images, data_format=backend.standardize_data_format(data_format)
    )


class SSIM(Operation):
    def __init__(
        self,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, image1, image2):
        return _ssim(
            image1,
            image2,
            max_val=self.max_val,
            filter_size=self.filter_size,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2,
            data_format=self.data_format,
        )

    def compute_output_spec(self, image1, image2):
        if len(image1.shape) not in (3, 4):
            raise ValueError(
                "Invalid image1 rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image1.shape={image1.shape}"
            )
        if len(image2.shape) not in (3, 4):
            raise ValueError(
                "Invalid image2 rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image2.shape={image2.shape}"
            )
        # Output is a scalar per image in the batch
        if len(image1.shape) == 3:
            output_shape = ()
        else:
            output_shape = (image1.shape[0],)
        return KerasTensor(shape=output_shape, dtype=image1.dtype)


@keras_export("keras.ops.image.ssim")
def ssim(
    image1,
    image2,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    data_format=None,
):
    """Computes the Structural Similarity Index (SSIM) between two images.

    The SSIM index is a method for measuring the similarity between two images.
    It is based on the comparison of luminance, contrast, and structure
    between the images. The resulting SSIM index is a value between -1 and 1,
    where 1 indicates identical images.

    This implementation is based on the original SSIM paper:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    "Image quality assessment: from error visibility to structural similarity."
    IEEE Transactions on Image Processing.

    Args:
        image1: First image or batch of images. Must be 3D or 4D.
        image2: Second image or batch of images. Must have the same shape
            as `image1`.
        max_val: The maximum possible pixel value of the images. Defaults to
            `1.0` (for normalized images). Use `255.0` for images with pixel
            values in `[0, 255]`.
        filter_size: Size of the Gaussian filter used for computing SSIM.
            Defaults to `11`. Must be an odd integer >= 1.
        filter_sigma: Standard deviation of the Gaussian filter. Defaults to
            `1.5`.
        k1: First stabilization constant. Defaults to `0.01`.
        k2: Second stabilization constant. Defaults to `0.03`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        A tensor of SSIM values. For batched inputs, returns a 1D tensor
        with one SSIM value per image pair. For unbatched inputs, returns
        a scalar tensor.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> # Two identical images should have SSIM = 1
    >>> image = np.random.random((32, 32, 3)).astype("float32")
    >>> ssim_value = ops.image.ssim(image, image, max_val=1.0)
    >>> float(ssim_value) > 0.99
    True

    >>> # Batched images
    >>> images1 = np.random.random((2, 32, 32, 3)).astype("float32")
    >>> images2 = np.random.random((2, 32, 32, 3)).astype("float32")
    >>> ssim_values = ops.image.ssim(images1, images2, max_val=1.0)
    >>> ssim_values.shape
    (2,)
    """
    if any_symbolic_tensors((image1, image2)):
        return SSIM(
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
            data_format=data_format,
        ).symbolic_call(image1, image2)
    return _ssim(
        image1,
        image2,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
        data_format=data_format,
    )


def _ssim(
    image1,
    image2,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    data_format=None,
):
    """Backend-agnostic SSIM implementation using Keras ops."""
    data_format = backend.standardize_data_format(data_format)
    image1 = backend.ops.convert_to_tensor(image1)
    image2 = backend.ops.convert_to_tensor(image2)

    # Ensure float dtype for computation
    original_dtype = image1.dtype
    compute_dtype = backend.result_type(image1.dtype, float)
    image1 = ops.cast(image1, compute_dtype)
    image2 = ops.cast(image2, compute_dtype)

    # Validate inputs
    if len(image1.shape) not in (3, 4):
        raise ValueError(
            "Invalid image1 rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image1.shape={image1.shape}"
        )
    if len(image2.shape) not in (3, 4):
        raise ValueError(
            "Invalid image2 rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image2.shape={image2.shape}"
        )

    # Handle unbatched images
    unbatched = len(image1.shape) == 3
    if unbatched:
        image1 = ops.expand_dims(image1, axis=0)
        image2 = ops.expand_dims(image2, axis=0)

    # Convert to channels_last if needed
    if data_format == "channels_first":
        image1 = ops.moveaxis(image1, 1, -1)
        image2 = ops.moveaxis(image2, 1, -1)

    # Get image dimensions
    shape = ops.shape(image1)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    # Create Gaussian kernel
    kernel = _create_gaussian_kernel(filter_size, filter_sigma, compute_dtype)
    # Reshape for depthwise convolution: (H, W, 1, 1)
    kernel = ops.reshape(kernel, (filter_size, filter_size, 1, 1))

    # SSIM constants
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Vectorized implementation: reshape to treat channels as batch items
    # This avoids Python loops and supports dynamic channel dimensions
    # Shape: (B, H, W, C) -> (B, C, H, W) -> (B*C, H, W, 1)
    image1_transposed = ops.transpose(image1, (0, 3, 1, 2))
    image2_transposed = ops.transpose(image2, (0, 3, 1, 2))
    image1_ch = ops.reshape(image1_transposed, [-1, height, width, 1])
    image2_ch = ops.reshape(image2_transposed, [-1, height, width, 1])

    # Compute means using depthwise convolution
    mu1 = _depthwise_conv(image1_ch, kernel)
    mu2 = _depthwise_conv(image2_ch, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = _depthwise_conv(image1_ch * image1_ch, kernel) - mu1_sq
    sigma2_sq = _depthwise_conv(image2_ch * image2_ch, kernel) - mu2_sq
    sigma12 = _depthwise_conv(image1_ch * image2_ch, kernel) - mu1_mu2

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    # Average SSIM over spatial dimensions: (B*C, H', W', 1) -> (B*C,)
    ssim_val = ops.mean(ssim_map, axis=[1, 2, 3])

    # Reshape back to (B, C) and average over channels
    ssim_per_image = ops.reshape(ssim_val, [batch_size, channels])
    ssim_result = ops.mean(ssim_per_image, axis=-1)

    # Remove batch dimension for unbatched input
    if unbatched:
        ssim_result = ops.squeeze(ssim_result, axis=0)

    return ops.cast(ssim_result, original_dtype)


def _create_gaussian_kernel(size, sigma, dtype):
    """Create a 2D Gaussian kernel."""
    # Create 1D Gaussian
    x = ops.arange(size, dtype=dtype)
    x = x - (size - 1) / 2.0
    gauss_1d = ops.exp(-(x**2) / (2.0 * sigma**2))
    gauss_1d = gauss_1d / ops.sum(gauss_1d)

    # Create 2D kernel via outer product
    kernel = ops.outer(gauss_1d, gauss_1d)
    return kernel


def _depthwise_conv(images, kernel):
    """Apply depthwise convolution with valid padding."""
    # Expand kernel for conv: (H, W, in_channels=1, out_channels=1)
    return backend.ops.nn.conv(
        images,
        kernel,
        strides=1,
        padding="valid",
        data_format="channels_last",
    )
