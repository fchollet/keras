import io
import pathlib

import numpy as np

from keras.src.api_export import keras_export
from keras.src.backend.config import standardize_data_format
from keras.src.utils import dataset_utils
from keras.src.utils import file_utils
from keras.src.utils import image_utils
from keras.src.utils.grain_utils import make_batch
from keras.src.utils.module_utils import grain
from keras.src.utils.module_utils import tensorflow as tf

try:
    from PIL import Image as pil_image

    try:
        pil_image_resampling = pil_image.Resampling
    except AttributeError:
        pil_image_resampling = pil_image
except ImportError:
    pil_image = None
    pil_image_resampling = None

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


@keras_export(
    [
        "keras.utils.image_dataset_from_directory",
        "keras.preprocessing.image_dataset_from_directory",
    ]
)
def image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
):
    """Generates a dataset from image files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataset_from_directory(main_directory,
    labels='inferred')` will return a dataset that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.
    Animated gifs are truncated to the first frame.

    By default, this function will return a `tf.data.Dataset` object. You can
    set `format="grain"` to return a `grain.IterDataset` object instead, which
    removes the TensorFlow dependency.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing images for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            image files found in the directory. Labels should be sorted
            according to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `labels` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`.
            Whether the images will be converted to
            have 1, 3, or 4 channels. Defaults to `"rgb"`.
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        image_size: Size to resize images to after they are read from disk,
            specified as `(height, width)`.
            Since the pipeline processes batches of images that must all have
            the same size, this must be provided. Defaults to `(256, 256)`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"`, or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        interpolation: String, the interpolation method used when
            resizing images.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `"bilinear"`.
        follow_links: Whether to visit subdirectories pointed to by symlinks.
            Defaults to `False`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the
            target aspect ratio, the output image will be cropped so as to
            return the largest possible window in the image
            (of size `image_size`) that matches the target aspect ratio. By
            default (`crop_to_aspect_ratio=False`), aspect ratio may not be
            preserved.
        pad_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the
            target aspect ratio, the output image will be padded so as to
            return the largest possible window in the image
            (of size `image_size`) that matches the target aspect ratio. By
            default (`pad_to_aspect_ratio=False`), aspect ratio may not be
            preserved.
        data_format: If None uses keras.config.image_data_format()
            otherwise either 'channel_last' or 'channel_first'.
        format: The format of the return object. Defaults to `"tf"`. Available
            options are:
            - `"tf"`: returns a `tf.data.Dataset` object. Requires
                TensorFlow to be installed.
            - `"grain"`: returns a `grain.IterDataset` object. Requires
                Grain to be installed.
        verbose: Whether to display number information on classes and
            number of files found. Defaults to `True`.

    Returns:

    A `tf.data.Dataset` (`format="tf"`) or `grain.IterDataset`
    (`format="grain"`) object.

    - If `label_mode` is `None`, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
    - Otherwise, it yields a tuple `(images, labels)`, where `images` has
        shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

    Rules regarding labels format:

    - if `label_mode` is `"int"`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `"binary"`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `"categorical"`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

    - if `color_mode` is `"grayscale"`,
        there's 1 channel in the image tensors.
    - if `color_mode` is `"rgb"`,
        there are 3 channels in the image tensors.
    - if `color_mode` is `"rgba"`,
        there are 4 channels in the image tensors.
    """

    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "`labels` argument should be a list/tuple of integer labels, "
                "of the same size as the number of image files in the target "
                "directory. If you wish to infer the labels from the "
                "subdirectory "
                'names in the target directory, pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains images "
                f"(no labels), pass `labels=None`. Received: labels={labels}"
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if "
                f'`labels="inferred"`. Received: labels={labels}, and '
                f"class_names={class_names}"
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", '
            '"categorical", "binary", '
            f"or None. Received: label_mode={label_mode}"
        )
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    elif not isinstance(image_size, (list, tuple)) or not len(image_size) == 2:
        raise ValueError(
            "Invalid `image_size` value. Expected a tuple of 2 integers. "
            f"Received: image_size={image_size}"
        )

    interpolation = interpolation.lower()
    supported_interpolations = (
        "bilinear",
        "nearest",
        "bicubic",
        "area",
        "lanczos3",
        "lanczos5",
        "gaussian",
        "mitchellcubic",
    )
    if interpolation not in supported_interpolations:
        raise ValueError(
            "Argument `interpolation` should be one of "
            f"{supported_interpolations}. "
            f"Received: interpolation={interpolation}"
        )
    if format not in ("tf", "grain"):
        raise ValueError(
            '`format` should be either "tf" or "grain". '
            f"Received: format={format}"
        )

    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=ALLOWLIST_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
        verbose=verbose,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary"`, there must be exactly 2 '
            f"class_names. Received: class_names={class_names}"
        )

    data_format = standardize_data_format(data_format=data_format)
    if batch_size is not None:
        shuffle_buffer_size = batch_size * 8
    else:
        shuffle_buffer_size = 1024

    if subset == "both":
        (
            image_paths_train,
            labels_train,
        ) = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, "training"
        )
        (
            image_paths_val,
            labels_val,
        ) = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, "validation"
        )
        if not image_paths_train:
            raise ValueError(
                f"No training images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )
        if not image_paths_val:
            raise ValueError(
                f"No validation images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )
        train_dataset = paths_and_labels_to_dataset(
            image_paths=image_paths_train,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels_train,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            data_format=data_format,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            format=format,
        )

        val_dataset = paths_and_labels_to_dataset(
            image_paths=image_paths_val,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels_val,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            data_format=data_format,
            shuffle=False,
            format=format,
        )

        if format == "tf":
            if batch_size is not None:
                train_dataset = train_dataset.batch(batch_size)
                val_dataset = val_dataset.batch(batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = train_dataset.to_iter_dataset()
            val_dataset = val_dataset.to_iter_dataset()
            if batch_size is not None:
                train_dataset = train_dataset.batch(
                    batch_size, batch_fn=make_batch
                )
                val_dataset = val_dataset.batch(batch_size, batch_fn=make_batch)

        # Users may need to reference `class_names`.
        train_dataset.class_names = class_names
        val_dataset.class_names = class_names

        # Include file paths for images as attribute.
        train_dataset.file_paths = image_paths_train
        val_dataset.file_paths = image_paths_val

        dataset = [train_dataset, val_dataset]
    else:
        image_paths, labels = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, subset
        )
        if not image_paths:
            raise ValueError(
                f"No images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )

        dataset = paths_and_labels_to_dataset(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            data_format=data_format,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            format=format,
        )

        if format == "tf":
            if batch_size is not None:
                dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.to_iter_dataset()
            if batch_size is not None:
                dataset = dataset.batch(batch_size, batch_fn=make_batch)

        # Users may need to reference `class_names`.
        dataset.class_names = class_names

        # Include file paths for images as attribute.
        dataset.file_paths = image_paths

    return dataset


def paths_and_labels_to_dataset(
    image_paths,
    image_size,
    num_channels,
    labels,
    label_mode,
    num_classes,
    interpolation,
    data_format,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
    format="tf",
):
    """Constructs a dataset of images and labels."""
    if format == "tf":
        return _paths_and_labels_to_dataset_tf(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=num_classes,
            interpolation=interpolation,
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
    elif format == "grain":
        return _paths_and_labels_to_dataset_grain(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=num_classes,
            interpolation=interpolation,
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        raise ValueError(
            '`format` should be either "tf" or "grain". '
            f"Received: format={format}"
        )


def _paths_and_labels_to_dataset_tf(
    image_paths,
    image_size,
    num_channels,
    labels,
    label_mode,
    num_classes,
    interpolation,
    data_format,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
):
    """Constructs a dataset of images and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_tf(
            labels, label_mode, num_classes
        )
        ds = tf.data.Dataset.zip((path_ds, label_ds))
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size or 1024, seed=seed)

    args = (
        image_size,
        num_channels,
        interpolation,
        data_format,
        crop_to_aspect_ratio,
        pad_to_aspect_ratio,
    )
    if label_mode:
        ds = ds.map(
            lambda x, y: (_load_image_tf(x, *args), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x: _load_image_tf(x, *args),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds


def _load_image_tf(
    path,
    image_size,
    num_channels,
    interpolation,
    data_format,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )

    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio`, `crop_to_aspect_ratio`"
            " can be set to `True`."
        )

    if crop_to_aspect_ratio:
        from keras.src.backend import tensorflow as tf_backend

        if data_format == "channels_first":
            img = tf.transpose(img, (2, 0, 1))
        img = image_utils.smart_resize(
            img,
            image_size,
            interpolation=interpolation,
            data_format=data_format,
            backend_module=tf_backend,
        )
    elif pad_to_aspect_ratio:
        img = tf.image.resize_with_pad(
            img, image_size[0], image_size[1], method=interpolation
        )
        if data_format == "channels_first":
            img = tf.transpose(img, (2, 0, 1))
    else:
        img = tf.image.resize(img, image_size, method=interpolation)
        if data_format == "channels_first":
            img = tf.transpose(img, (2, 0, 1))

    if data_format == "channels_last":
        img.set_shape((image_size[0], image_size[1], num_channels))
    else:
        img.set_shape((num_channels, image_size[0], image_size[1]))
    return img


def _paths_and_labels_to_dataset_grain(
    image_paths,
    image_size,
    num_channels,
    labels,
    label_mode,
    num_classes,
    interpolation,
    data_format,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    shuffle=False,
    seed=None,
):
    """Constructs a dataset of images and labels."""
    path_ds = grain.MapDataset.source(image_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_grain(
            labels, label_mode, num_classes
        )
        ds = grain.experimental.ZipMapDataset([path_ds, label_ds])
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(seed=seed)

    args = (
        image_size,
        num_channels,
        interpolation,
        data_format,
        crop_to_aspect_ratio,
        pad_to_aspect_ratio,
    )
    if label_mode:
        ds = ds.map(lambda data: (_load_image_grain(data[0], *args), data[1]))
    else:
        ds = ds.map(lambda x: _load_image_grain(x, *args))

    return ds


def _load_image_grain(
    path,
    image_size,
    num_channels,
    interpolation,
    data_format,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
):
    """Load an image from a path and resize it."""
    from keras.src import backend
    from keras.src import ops

    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. The use of `load_img` requires PIL."
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio`, `crop_to_aspect_ratio`"
            " can be set to `True`."
        )

    if isinstance(path, io.BytesIO):
        img = pil_image.open(path)
    elif isinstance(path, (pathlib.Path, bytes, str)):
        if isinstance(path, pathlib.Path):
            path = str(path.resolve())
        with file_utils.File(path, "rb") as f:
            img = pil_image.open(f)
            img.load()
    else:
        raise TypeError(
            f"path should be path-like or io.BytesIO, not {type(path)}"
        )
    if num_channels == 1:
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif num_channels == 4:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif num_channels == 3:
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError(
            "num_channels must be 1, 3 or 4. "
            f"Received: num_channels={num_channels}"
        )

    with backend.device_scope("cpu"):
        img = ops.convert_to_tensor(np.array(img), dtype="float32")
        if len(img.shape) == 2:
            # If the image is grayscale, expand dims to add channel axis.
            # The reason is that `ops.image.resize` expects 3D or 4D tensors.
            img = ops.expand_dims(img, axis=-1)
        if data_format == "channels_first":
            img = ops.transpose(img, (2, 0, 1))
        img = ops.image.resize(
            img,
            size=image_size,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            data_format=data_format,
        )
        if backend.backend() == "tensorflow":
            if data_format == "channels_last":
                img.set_shape((image_size[0], image_size[1], num_channels))
            else:
                img.set_shape((num_channels, image_size[0], image_size[1]))
    return img
