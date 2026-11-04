import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils import dataset_utils
from keras.src.utils import file_utils
from keras.src.utils.grain_utils import make_string_batch
from keras.src.utils.module_utils import grain
from keras.src.utils.module_utils import tensorflow as tf


@keras_export(
    [
        "keras.utils.text_dataset_from_directory",
        "keras.preprocessing.text_dataset_from_directory",
    ]
)
def text_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    format="tf",
    verbose=True,
):
    """Generates a dataset from text files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    ```

    Then calling `text_dataset_from_directory(main_directory,
    labels='inferred')` will return a dataset that yields batches of
    texts from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.txt` files are supported at this time.

    By default, this function will return a `tf.data.Dataset` object. You can
    set `format="grain"` to return a `grain.IterDataset` object instead, which
    removes the TensorFlow dependency.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing text files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            text files found in the directory. Labels should be sorted according
            to the alphanumeric order of the text file paths
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
        class_names: Only valid if `"labels"` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
            Defaults to `32`.
        max_length: Maximum size of a text string. Texts longer than this will
            be truncated to `max_length`.
        shuffle: Whether to shuffle the data.
            If set to `False`, sorts the data in alphanumeric order.
            Defaults to `True`.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"` or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.
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

    When `format="tf"`:
    - If `label_mode` is `None`, it yields `string` tensors of shape
        `(batch_size,)`, containing the contents of a batch of text files.
    - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        has shape `(batch_size,)` and `labels` follows the format described
        below.

    When `format="grain"`:
    - If `label_mode` is `None`, it yields a list of Python strings containing
        the contents of a batch of text files.
    - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        is a list of Python strings and `labels` follows the format described
        below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.
    """
    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "`labels` argument should be a list/tuple of integer labels, "
                "of the same size as the number of text files in the target "
                "directory. If you wish to infer the labels from the "
                "subdirectory names in the target directory, "
                'pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains text samples "
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
    if format not in ("tf", "grain"):
        raise ValueError(
            '`format` should be either "tf" or "grain". '
            f"Received: format={format}"
        )
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)
    file_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=(".txt",),
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
    if batch_size is not None:
        shuffle_buffer_size = batch_size * 8
    else:
        shuffle_buffer_size = 1024

    if subset == "both":
        (
            file_paths_train,
            labels_train,
        ) = dataset_utils.get_training_or_validation_split(
            file_paths, labels, validation_split, "training"
        )
        (
            file_paths_val,
            labels_val,
        ) = dataset_utils.get_training_or_validation_split(
            file_paths, labels, validation_split, "validation"
        )
        if not file_paths_train:
            raise ValueError(
                f"No training text files found in directory {directory}. "
                "Allowed format: .txt"
            )
        if not file_paths_val:
            raise ValueError(
                f"No validation text files found in directory {directory}. "
                "Allowed format: .txt"
            )
        train_dataset = paths_and_labels_to_dataset(
            file_paths=file_paths_train,
            labels=labels_train,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            max_length=max_length,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            format=format,
        )
        val_dataset = paths_and_labels_to_dataset(
            file_paths=file_paths_val,
            labels=labels_val,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            max_length=max_length,
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
                    batch_size, batch_fn=make_string_batch
                )
                val_dataset = val_dataset.batch(
                    batch_size, batch_fn=make_string_batch
                )

        # Users may need to reference `class_names`.
        train_dataset.class_names = class_names
        val_dataset.class_names = class_names
        dataset = [train_dataset, val_dataset]
    else:
        file_paths, labels = dataset_utils.get_training_or_validation_split(
            file_paths, labels, validation_split, subset
        )
        if not file_paths:
            raise ValueError(
                f"No text files found in directory {directory}. "
                "Allowed format: .txt"
            )
        dataset = paths_and_labels_to_dataset(
            file_paths=file_paths,
            labels=labels,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            max_length=max_length,
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
                dataset = dataset.batch(batch_size, batch_fn=make_string_batch)

        # Users may need to reference `class_names`.
        dataset.class_names = class_names
    return dataset


def paths_and_labels_to_dataset(
    file_paths,
    labels,
    label_mode,
    num_classes,
    max_length,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
    format="tf",
):
    """Constructs a dataset of text strings and labels."""
    if format == "tf":
        return _paths_and_labels_to_dataset_tf(
            file_paths,
            labels,
            label_mode,
            num_classes,
            max_length,
            shuffle,
            shuffle_buffer_size,
            seed,
        )
    elif format == "grain":
        return _paths_and_labels_to_dataset_grain(
            file_paths,
            labels,
            label_mode,
            num_classes,
            max_length,
            shuffle,
            shuffle_buffer_size,
            seed,
        )


def _paths_and_labels_to_dataset_tf(
    file_paths,
    labels,
    label_mode,
    num_classes,
    max_length,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
):
    """Constructs a dataset of text strings and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_tf(
            labels, label_mode, num_classes
        )
        ds = tf.data.Dataset.zip((path_ds, label_ds))
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size or 1024, seed=seed)

    if label_mode:
        ds = ds.map(
            lambda x, y: (_path_to_string_content_tf(x, max_length), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x: _path_to_string_content_tf(x, max_length),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds


def _path_to_string_content_tf(path, max_length):
    txt = tf.io.read_file(path)
    if max_length is not None:
        txt = tf.strings.substr(txt, 0, max_length)
    return txt


def _paths_and_labels_to_dataset_grain(
    file_paths,
    labels,
    label_mode,
    num_classes,
    max_length,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
):
    """Constructs a dataset of text strings and labels."""
    path_ds = grain.MapDataset.source(file_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_grain(
            labels, label_mode, num_classes
        )
        ds = grain.experimental.ZipMapDataset([path_ds, label_ds])
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if label_mode:
        ds = ds.map(
            lambda data: (
                _path_to_string_content_grain(data[0], max_length),
                data[1],
            ),
        )
    else:
        ds = ds.map(lambda x: _path_to_string_content_grain(x, max_length))
    return ds


def _path_to_string_content_grain(path, max_length):
    with file_utils.File(path, "r") as f:
        txt = f.read()
    if max_length is not None:
        txt = txt[:max_length]
    return txt
