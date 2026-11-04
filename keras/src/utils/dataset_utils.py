import os
import random
import time
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.utils import file_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import grain


@keras_export("keras.utils.split_dataset")
def split_dataset(
    dataset,
    left_size=None,
    right_size=None,
    shuffle=False,
    seed=None,
    preferred_backend=None,
):
    """Splits a dataset into a left half and a right half (e.g. train / test).

    Args:
        dataset:
            A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays with the same length.
        left_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the left dataset. If integer, it
            signifies the number of samples to pack in the left dataset. If
            `None`, defaults to the complement to `right_size`.
            Defaults to `None`.
        right_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the right dataset.
            If integer, it signifies the number of samples to pack
            in the right dataset.
            If `None`, defaults to the complement to `left_size`.
            Defaults to `None`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.
        preferred_backend: String, specifying which backend
            (e.g.; "tensorflow", "torch") to use. If `None`, the
            backend is inferred from the type of `dataset` - if
            `dataset` is a `tf.data.Dataset`, "tensorflow" backend
            is used, if `dataset` is a `torch.utils.data.Dataset`,
            "torch" backend is used, and if `dataset` is a list/tuple/np.array
            the current Keras backend is used. Defaults to `None`.

    Returns:
        A tuple of two dataset objects, the left and right splits. The exact
        type of the returned objects depends on the `preferred_backend`.
        For example, with a "tensorflow" backend,
        `tf.data.Dataset` objects are returned. With a "torch" backend,
        `torch.utils.data.Dataset` objects are returned.
    Example:

    >>> data = np.random.random(size=(1000, 4))
    >>> left_ds, right_ds = keras.utils.split_dataset(data, left_size=0.8)
    >>> # For a tf.data.Dataset, you can use .cardinality()
    >>> # >>> int(left_ds.cardinality())
    >>> # 800
    >>> # For a torch.utils.data.Dataset, you can use len()
    >>> # >>> len(left_ds)
    >>> # 800
    """
    preferred_backend = preferred_backend or _infer_preferred_backend(dataset)
    if preferred_backend != "torch":
        return _split_dataset_tf(
            dataset,
            left_size=left_size,
            right_size=right_size,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        return _split_dataset_torch(
            dataset,
            left_size=left_size,
            right_size=right_size,
            shuffle=shuffle,
            seed=seed,
        )


def _split_dataset_tf(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
):
    """Splits a dataset into a left half and a right half (e.g. train / test).

    Args:
        dataset:
            A `tf.data.Dataset` object,
            or a list/tuple of arrays with the same length.
        left_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the left dataset. If integer, it
            signifies the number of samples to pack in the left dataset. If
            `None`, defaults to the complement to `right_size`.
            Defaults to `None`.
        right_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the right dataset.
            If integer, it signifies the number of samples to pack
            in the right dataset.
            If `None`, defaults to the complement to `left_size`.
            Defaults to `None`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.

    Returns:
        A tuple of two `tf.data.Dataset` objects:
        the left and right splits.
    """
    from keras.src.utils.module_utils import tensorflow as tf

    dataset_type_spec = _get_type_spec(dataset)

    if dataset_type_spec is None:
        raise TypeError(
            "The `dataset` argument must be either"
            "a `tf.data.Dataset` object, or"
            "a list/tuple of arrays. "
            f"Received: dataset={dataset} of type {type(dataset)}"
        )

    if right_size is None and left_size is None:
        raise ValueError(
            "At least one of the `left_size` or `right_size` "
            "must be specified. Received: left_size=None and "
            "right_size=None"
        )

    dataset_as_list = _convert_dataset_to_list(dataset, dataset_type_spec)

    if shuffle:
        if seed is None:
            seed = random.randint(0, int(1e6))
        random.seed(seed)
        random.shuffle(dataset_as_list)

    total_length = len(dataset_as_list)

    left_size, right_size = _rescale_dataset_split_sizes(
        left_size, right_size, total_length
    )
    left_split = list(dataset_as_list[:left_size])
    right_split = list(dataset_as_list[-right_size:])

    left_split = _restore_dataset_from_list(
        left_split, dataset_type_spec, dataset
    )
    right_split = _restore_dataset_from_list(
        right_split, dataset_type_spec, dataset
    )

    left_split = tf.data.Dataset.from_tensor_slices(left_split)
    right_split = tf.data.Dataset.from_tensor_slices(right_split)

    # apply batching to the splits if the dataset is batched
    if dataset_type_spec is tf.data.Dataset and is_batched(dataset):
        batch_size = get_batch_size(dataset)
        if batch_size is not None:
            left_split = left_split.batch(batch_size)
            right_split = right_split.batch(batch_size)

    left_split = left_split.prefetch(tf.data.AUTOTUNE)
    right_split = right_split.prefetch(tf.data.AUTOTUNE)
    return left_split, right_split


def _split_dataset_torch(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
):
    """Splits a dataset into a left half and a right half (e.g. train / test).

    Args:
        dataset:
            A `torch.utils.data.Dataset` object,
            or a list/tuple of arrays with the same length.
        left_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the left dataset. If integer, it
            signifies the number of samples to pack in the left dataset. If
            `None`, defaults to the complement to `right_size`.
            Defaults to `None`.
        right_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the right dataset.
            If integer, it signifies the number of samples to pack
            in the right dataset.
            If `None`, defaults to the complement to `left_size`.
            Defaults to `None`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.

    Returns:
        A tuple of two `torch.utils.data.Dataset` objects:
        the left and right splits.
    """
    import torch
    from torch.utils.data import TensorDataset
    from torch.utils.data import random_split

    dataset_type_spec = _get_type_spec(dataset)
    if dataset_type_spec is None:
        raise TypeError(
            "The `dataset` argument must be a `torch.utils.data.Dataset`"
            " object, or a list/tuple of arrays."
            f" Received: dataset={dataset} of type {type(dataset)}"
        )

    if not isinstance(dataset, torch.utils.data.Dataset):
        if dataset_type_spec is np.ndarray:
            dataset = TensorDataset(torch.from_numpy(dataset))
        elif dataset_type_spec in (list, tuple):
            tensors = [torch.from_numpy(x) for x in dataset]
            dataset = TensorDataset(*tensors)
        elif is_tf_dataset(dataset):
            dataset_as_list = _convert_dataset_to_list(
                dataset, dataset_type_spec
            )
            tensors = [
                torch.from_numpy(np.array(sample))
                for sample in zip(*dataset_as_list)
            ]
            dataset = TensorDataset(*tensors)

    if right_size is None and left_size is None:
        raise ValueError(
            "At least one of the `left_size` or `right_size` "
            "must be specified. "
            "Received: left_size=None and right_size=None"
        )

    # Calculate total length and rescale split sizes
    total_length = len(dataset)
    left_size, right_size = _rescale_dataset_split_sizes(
        left_size, right_size, total_length
    )

    # Shuffle the dataset if required
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
    else:
        generator = None

    left_split, right_split = random_split(
        dataset, [left_size, right_size], generator=generator
    )

    return left_split, right_split


def _infer_preferred_backend(dataset):
    """Infer the backend from the dataset type."""
    if isinstance(dataset, (list, tuple, np.ndarray)):
        return backend.backend()
    if is_tf_dataset(dataset):
        return "tensorflow"
    elif is_torch_dataset(dataset):
        return "torch"
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def _convert_dataset_to_list(
    dataset,
    dataset_type_spec,
    data_size_warning_flag=True,
    ensure_shape_similarity=True,
):
    """Convert `dataset` object to a list of samples.

    Args:
        dataset: A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays.
        dataset_type_spec: the type of the dataset.
        data_size_warning_flag: If set to `True`, a warning will
            be issued if the dataset takes longer than 10 seconds to iterate.
            Defaults to `True`.
        ensure_shape_similarity: If set to `True`, the shape of
            the first sample will be used to validate the shape of rest of the
            samples. Defaults to `True`.

    Returns:
        List: A list of samples.
    """
    dataset_iterator = _get_data_iterator_from_dataset(
        dataset, dataset_type_spec
    )
    dataset_as_list = []

    start_time = time.time()
    for sample in _get_next_sample(
        dataset_iterator,
        ensure_shape_similarity,
        data_size_warning_flag,
        start_time,
    ):
        dataset_as_list.append(sample)

    return dataset_as_list


def _get_data_iterator_from_dataset(dataset, dataset_type_spec):
    """Get the iterator from a dataset.

    Args:
        dataset: A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays.
        dataset_type_spec: The type of the dataset.

    Returns:
        iterator: An `iterator` object.
    """
    if dataset_type_spec is list:
        if len(dataset) == 0:
            raise ValueError(
                "Received an empty list dataset. "
                "Please provide a non-empty list of arrays."
            )

        expected_shape = None
        for i, element in enumerate(dataset):
            if not isinstance(element, np.ndarray):
                raise ValueError(
                    "Expected a list of `numpy.ndarray` objects,"
                    f"Received: {type(element)} at index {i}."
                )
            if expected_shape is None:
                expected_shape = element.shape
            elif element.shape[0] != expected_shape[0]:
                raise ValueError(
                    "Received a list of NumPy arrays with different lengths."
                    f"Mismatch found at index {i}, "
                    f"Expected shape={expected_shape} "
                    f"Received shape={np.array(element).shape}."
                    "Please provide a list of NumPy arrays of the same length."
                )

        return iter(zip(*dataset))
    elif dataset_type_spec is tuple:
        if len(dataset) == 0:
            raise ValueError(
                "Received an empty list dataset."
                "Please provide a non-empty tuple of arrays."
            )

        expected_shape = None
        for i, element in enumerate(dataset):
            if not isinstance(element, np.ndarray):
                raise ValueError(
                    "Expected a tuple of `numpy.ndarray` objects,"
                    f"Received: {type(element)} at index {i}."
                )
            if expected_shape is None:
                expected_shape = element.shape
            elif element.shape[0] != expected_shape[0]:
                raise ValueError(
                    "Received a tuple of NumPy arrays with different lengths."
                    f"Mismatch found at index {i}, "
                    f"Expected shape={expected_shape} "
                    f"Received shape={np.array(element).shape}."
                    "Please provide a tuple of NumPy arrays of the same length."
                )

        return iter(zip(*dataset))
    elif is_tf_dataset(dataset):
        if is_batched(dataset):
            dataset = dataset.unbatch()
        return iter(dataset)

    elif is_torch_dataset(dataset):
        return iter(dataset)
    elif dataset_type_spec is np.ndarray:
        return iter(dataset)
    raise ValueError(f"Invalid dataset_type_spec: {dataset_type_spec}")


def _get_next_sample(
    dataset_iterator,
    ensure_shape_similarity,
    data_size_warning_flag,
    start_time,
):
    """Yield data samples from the `dataset_iterator`.

    Args:
        dataset_iterator: An `iterator` object.
        ensure_shape_similarity: If set to `True`, the shape of
            the first sample will be used to validate the shape of rest of the
            samples. Defaults to `True`.
        data_size_warning_flag: If set to `True`, a warning will
            be issued if the dataset takes longer than 10 seconds to iterate.
            Defaults to `True`.
        start_time (float): the start time of the dataset iteration. this is
            used only if `data_size_warning_flag` is set to true.

    Yields:
        data_sample: The next sample.
    """
    from keras.src.trainers.data_adapters.data_adapter_utils import (
        is_tensorflow_tensor,
    )
    from keras.src.trainers.data_adapters.data_adapter_utils import (
        is_torch_tensor,
    )

    try:
        dataset_iterator = iter(dataset_iterator)
        first_sample = next(dataset_iterator)
        if (
            isinstance(first_sample, np.ndarray)
            or is_tensorflow_tensor(first_sample)
            or is_torch_tensor(first_sample)
        ):
            first_sample_shape = np.array(first_sample).shape
        else:
            first_sample_shape = None
            ensure_shape_similarity = False
        yield first_sample
    except StopIteration:
        raise ValueError(
            "Received an empty dataset. Argument `dataset` must "
            "be a non-empty list/tuple of `numpy.ndarray` objects "
            "or `tf.data.Dataset` objects."
        )

    for i, sample in enumerate(dataset_iterator):
        if ensure_shape_similarity:
            if first_sample_shape != np.array(sample).shape:
                raise ValueError(
                    "All `dataset` samples must have same shape, "
                    f"Expected shape: {np.array(first_sample).shape} "
                    f"Received shape: {np.array(sample).shape} at index "
                    f"{i}."
                )
        if data_size_warning_flag:
            if i % 10 == 0:
                cur_time = time.time()
                # warns user if the dataset is too large to iterate within 10s
                if int(cur_time - start_time) > 10 and data_size_warning_flag:
                    warnings.warn(
                        "The dataset is taking longer than 10 seconds to "
                        "iterate over. This may be due to the size of the "
                        "dataset. Keep in mind that the `split_dataset` "
                        "utility is only for small in-memory dataset "
                        "(e.g. < 10,000 samples).",
                        category=ResourceWarning,
                        source="split_dataset",
                    )
                    data_size_warning_flag = False
        yield sample


def is_tf_dataset(dataset):
    return _mro_matches(
        dataset,
        class_names=("DatasetV2", "Dataset"),
        module_substrings=(
            "tensorflow.python.data",  # TF classic
            "tensorflow.data",  # newer TF paths
        ),
    )


def is_grain_dataset(dataset):
    return _mro_matches(
        dataset,
        class_names=("MapDataset", "IterDataset"),
        module_prefixes=("grain._src.python",),
    )


def is_torch_dataset(dataset):
    return _mro_matches(dataset, ("Dataset",), ("torch.utils.data",))


def _mro_matches(
    dataset, class_names, module_prefixes=(), module_substrings=()
):
    if not hasattr(dataset, "__class__"):
        return False
    for parent in dataset.__class__.__mro__:
        if parent.__name__ in class_names:
            mod = str(parent.__module__)
            if any(mod.startswith(pref) for pref in module_prefixes):
                return True
            if any(subs in mod for subs in module_substrings):
                return True
    return False


def _rescale_dataset_split_sizes(left_size, right_size, total_length):
    """Rescale the dataset split sizes.

    We want to ensure that the sum of
    the split sizes is equal to the total length of the dataset.

    Args:
        left_size: The size of the left dataset split.
        right_size: The size of the right dataset split.
        total_length: The total length of the dataset.

    Returns:
        tuple: A tuple of rescaled `left_size` and `right_size` integers.
    """
    left_size_type = type(left_size)
    right_size_type = type(right_size)

    # check both left_size and right_size are integers or floats
    if (left_size is not None and left_size_type not in [int, float]) and (
        right_size is not None and right_size_type not in [int, float]
    ):
        raise TypeError(
            "Invalid `left_size` and `right_size` Types. Expected: "
            "integer or float or None, Received: type(left_size)="
            f"{left_size_type} and type(right_size)={right_size_type}"
        )

    # check left_size is a integer or float
    if left_size is not None and left_size_type not in [int, float]:
        raise TypeError(
            "Invalid `left_size` Type. Expected: int or float or None, "
            f"Received: type(left_size)={left_size_type}.  "
        )

    # check right_size is a integer or float
    if right_size is not None and right_size_type not in [int, float]:
        raise TypeError(
            "Invalid `right_size` Type. "
            "Expected: int or float or None,"
            f"Received: type(right_size)={right_size_type}."
        )

    # check left_size and right_size are non-zero
    if left_size == 0 and right_size == 0:
        raise ValueError(
            "Both `left_size` and `right_size` are zero. "
            "At least one of the split sizes must be non-zero."
        )

    # check left_size is non-negative and less than 1 and less than total_length
    if (
        left_size_type is int
        and (left_size <= 0 or left_size >= total_length)
        or left_size_type is float
        and (left_size <= 0 or left_size >= 1)
    ):
        raise ValueError(
            "`left_size` should be either a positive integer "
            f"smaller than {total_length}, or a float "
            "within the range `[0, 1]`. Received: left_size="
            f"{left_size}"
        )

    # check right_size is non-negative and less than 1 and less than
    # total_length
    if (
        right_size_type is int
        and (right_size <= 0 or right_size >= total_length)
        or right_size_type is float
        and (right_size <= 0 or right_size >= 1)
    ):
        raise ValueError(
            "`right_size` should be either a positive integer "
            f"and smaller than {total_length} or a float "
            "within the range `[0, 1]`. Received: right_size="
            f"{right_size}"
        )

    # check sum of left_size and right_size is less than or equal to
    # total_length
    if (
        right_size_type is left_size_type is float
        and right_size + left_size > 1
    ):
        raise ValueError(
            "The sum of `left_size` and `right_size` is greater "
            "than 1. It must be less than or equal to 1."
        )

    if left_size_type is float:
        left_size = round(left_size * total_length)
    elif left_size_type is int:
        left_size = float(left_size)

    if right_size_type is float:
        right_size = round(right_size * total_length)
    elif right_size_type is int:
        right_size = float(right_size)

    if left_size is None:
        left_size = total_length - right_size
    elif right_size is None:
        right_size = total_length - left_size

    if left_size + right_size > total_length:
        raise ValueError(
            "The sum of `left_size` and `right_size` should "
            f"be smaller than the {total_length}. "
            f"Received: left_size + right_size = {left_size + right_size}"
            f"and total_length = {total_length}"
        )

    for split, side in [(left_size, "left"), (right_size, "right")]:
        if split == 0:
            raise ValueError(
                f"With `dataset` of length={total_length}, `left_size`="
                f"{left_size} and `right_size`={right_size}."
                f"Resulting {side} side dataset split will be empty. "
                "Adjust any of the aforementioned parameters"
            )

    left_size, right_size = int(left_size), int(right_size)
    return left_size, right_size


def _restore_dataset_from_list(
    dataset_as_list, dataset_type_spec, original_dataset
):
    """Restore the dataset from the list of arrays."""
    if (
        dataset_type_spec in [tuple, list]
        or is_tf_dataset(original_dataset)
        or is_torch_dataset(original_dataset)
    ):
        # Save structure by taking the first element.
        element_spec = dataset_as_list[0]
        # Flatten each element.
        dataset_as_list = [tree.flatten(sample) for sample in dataset_as_list]
        # Combine respective elements at all indices.
        dataset_as_list = [np.array(sample) for sample in zip(*dataset_as_list)]
        # Recreate the original structure of elements.
        dataset_as_list = tree.pack_sequence_as(element_spec, dataset_as_list)
        # Turn lists to tuples as tf.data will fail on lists.
        return tree.traverse(
            lambda x: tuple(x) if isinstance(x, list) else x,
            dataset_as_list,
            top_down=False,
        )

    return dataset_as_list


def is_batched(dataset):
    """Check if the `tf.data.Dataset` is batched."""
    return hasattr(dataset, "_batch_size")


def get_batch_size(dataset):
    """Get the batch size of the dataset."""
    if is_batched(dataset):
        return dataset._batch_size
    else:
        return None


def _get_type_spec(dataset):
    """Get the type spec of the dataset."""
    if isinstance(dataset, tuple):
        return tuple
    elif isinstance(dataset, list):
        return list
    elif isinstance(dataset, np.ndarray):
        return np.ndarray
    elif is_tf_dataset(dataset):
        from keras.src.utils.module_utils import tensorflow as tf

        return tf.data.Dataset
    elif is_torch_dataset(dataset):
        from torch.utils.data import Dataset as TorchDataset

        return TorchDataset
    elif is_grain_dataset(dataset):
        from grain import MapDataset

        return MapDataset
    else:
        return None


def index_directory(
    directory,
    labels,
    formats,
    class_names=None,
    shuffle=True,
    seed=None,
    follow_links=False,
    verbose=True,
):
    """List all files in `directory`, with their labels.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number
            of valid files found in the directory.
            Labels should be sorted according
            to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
        formats: Allowlist of file extensions to index
            (e.g. `".jpg"`, `".txt"`).
        class_names: Only valid if `labels="inferred"`. This is the explicit
            list of class names (must match names of subdirectories). Used
            to control the order of the classes
            (otherwise alphanumerical order is used).
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
        verbose: Whether the function prints number of files found and classes.
            Defaults to `True`.

    Returns:
        tuple (file_paths, labels, class_names).
        - file_paths: list of file paths (strings).
        - labels: list of matching integer labels (same length as file_paths)
        - class_names: names of the classes corresponding to these labels, in
        order.
    """
    if labels == "inferred":
        subdirs = []
        for subdir in sorted(file_utils.listdir(directory)):
            if file_utils.isdir(file_utils.join(directory, subdir)):
                if not subdir.startswith("."):
                    if subdir.endswith("/"):
                        subdir = subdir[:-1]
                    subdirs.append(subdir)
        if class_names is not None:
            if not set(class_names).issubset(set(subdirs)):
                raise ValueError(
                    "The `class_names` passed did not match the "
                    "names of the subdirectories of the target directory. "
                    f"Expected: {subdirs} (or a subset of it), "
                    f"but received: class_names={class_names}"
                )
            subdirs = class_names  # Keep provided order.
    else:
        # In the explicit/no-label cases, index from the parent directory down.
        subdirs = [""]
        if class_names is not None:
            if labels is None:
                raise ValueError(
                    "When `labels=None` (no labels), argument `class_names` "
                    "cannot be specified."
                )
            else:
                raise ValueError(
                    "When argument `labels` is specified, argument "
                    "`class_names` cannot be specified (the `class_names` "
                    "will be the sorted list of labels)."
                )
    class_names = subdirs
    class_indices = dict(zip(class_names, range(len(class_names))))

    # Build an index of the files
    # in the different class subfolders.
    pool = ThreadPool()
    results = []
    filenames = []

    for dirpath in (file_utils.join(directory, subdir) for subdir in subdirs):
        results.append(
            pool.apply_async(
                index_subdirectory,
                (dirpath, class_indices, follow_links, formats),
            )
        )
    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res.get()
        labels_list.append(partial_labels)
        filenames += partial_filenames

    if labels == "inferred":
        # Inferred labels.
        i = 0
        labels = np.zeros((len(filenames),), dtype="int32")
        for partial_labels in labels_list:
            labels[i : i + len(partial_labels)] = partial_labels
            i += len(partial_labels)
    elif labels is None:
        class_names = None
    else:
        # Manual labels.
        if len(labels) != len(filenames):
            raise ValueError(
                "Expected the lengths of `labels` to match the number "
                "of files in the target directory. len(labels) is "
                f"{len(labels)} while we found {len(filenames)} files "
                f"in directory {directory}."
            )
        class_names = [str(label) for label in sorted(set(labels))]
    if verbose:
        if labels is None:
            io_utils.print_msg(f"Found {len(filenames)} files.")
        else:
            io_utils.print_msg(
                f"Found {len(filenames)} files belonging "
                f"to {len(class_names)} classes."
            )
    pool.close()
    pool.join()
    file_paths = [file_utils.join(directory, fname) for fname in filenames]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        if labels is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(labels)
    return file_paths, labels, class_names


def iter_valid_files(directory, follow_links, formats):
    walk = file_utils.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname


def index_subdirectory(directory, class_indices, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.

    Args:
        directory: string, target directory.
        class_indices: dict mapping class names to their index.
        follow_links: boolean, whether to recursively follow subdirectories
            (if False, we only list top-level images in `directory`).
        formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").

    Returns:
        tuple `(filenames, labels)`. `filenames` is a list of relative file
            paths, and `labels` is a list of integer labels corresponding
            to these files.
    """
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = file_utils.join(root, fname)
        relative_path = file_utils.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)
    return filenames, labels


def get_training_or_validation_split(samples, labels, validation_split, subset):
    """Potentially restrict samples & labels to a training or validation split.

    Args:
        samples: List of elements.
        labels: List of corresponding labels.
        validation_split: Float, fraction of data to reserve for validation.
        subset: Subset of the data to return.
            Either `"training"`, `"validation"`, or `None`.
            If `None`, we return all of the data.

    Returns:
        tuple (samples, labels), potentially restricted to the specified subset.
    """
    if not validation_split:
        return samples, labels

    num_val_samples = int(validation_split * len(samples))
    num_train_samples = len(samples) - num_val_samples
    if subset == "training":
        io_utils.print_msg(f"Using {num_train_samples} files for training.")
        samples = samples[:num_train_samples]
        if labels is not None:
            labels = labels[:num_train_samples]
    elif subset == "validation":
        io_utils.print_msg(f"Using {num_val_samples} files for validation.")
        samples = samples[num_train_samples:]
        if labels is not None:
            labels = labels[num_train_samples:]
    else:
        raise ValueError(
            '`subset` must be either "training" '
            f'or "validation", received: {subset}'
        )
    return samples, labels


def labels_to_dataset_tf(labels, label_mode, num_classes):
    """Create a `tf.data.Dataset` from the list/tuple of labels.

    Args:
        labels: list/tuple of labels to be converted into a `tf.data.Dataset`.
        label_mode: String describing the encoding of `labels`. Options are:
        - `"binary"` indicates that the labels (there can be only 2) are encoded
            as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - `"categorical"` means that the labels are mapped into a categorical
            vector.  (e.g. for `categorical_crossentropy` loss).
        num_classes: number of classes of labels.

    Returns:
        A `tf.data.Dataset` instance.
    """
    from keras.src.utils.module_utils import tensorflow as tf

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    if label_mode == "binary":
        label_ds = label_ds.map(
            lambda x: tf.expand_dims(tf.cast(x, "float32"), axis=-1),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif label_mode == "categorical":
        label_ds = label_ds.map(
            lambda x: tf.one_hot(x, num_classes),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return label_ds


def labels_to_dataset_grain(labels, label_mode, num_classes):
    """Create a `grain.MapDataset` from the list/tuple of labels.

    Args:
        labels: list/tuple of labels to be converted into a `grain.MapDataset`.
        label_mode: String describing the encoding of `labels`. Options are:
        - `"binary"` indicates that the labels (there can be only 2) are encoded
            as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - `"categorical"` means that the labels are mapped into a categorical
            vector.  (e.g. for `categorical_crossentropy` loss).
        num_classes: number of classes of labels.

    Returns:
        A `grain.MapDataset` instance.
    """
    from keras.src import backend
    from keras.src import ops

    if label_mode not in ("binary", "categorical", "int"):
        raise ValueError(
            f"Invalid `label_mode`: {label_mode}. "
            "Expected one of: 'binary', 'categorical', 'int'."
        )

    def preprocess_labels_in_cpu(label_mode, x, num_classes):
        with backend.device_scope("cpu"):
            if label_mode == "binary":
                return ops.expand_dims(
                    ops.convert_to_tensor(x, dtype="float32"), axis=-1
                )
            elif label_mode == "categorical":
                return ops.one_hot(
                    ops.convert_to_tensor(x, dtype="int32"), num_classes
                )
            else:
                return ops.convert_to_tensor(x, dtype="int32")

    label_ds = grain.MapDataset.source(labels)
    label_ds = label_ds.map(
        lambda x: preprocess_labels_in_cpu(label_mode, x, num_classes),
    )
    return label_ds


def check_validation_split_arg(validation_split, subset, shuffle, seed):
    """Raise errors in case of invalid argument values.

    Args:
        validation_split: float between 0 and 1, fraction of data to reserve for
            validation.
        subset: One of `"training"`, `"validation"`, or `"both"`. Only used if
            `validation_split` is set.
        shuffle: Whether to shuffle the data. Either `True` or `False`.
        seed: random seed for shuffling and transformations.
    """
    if validation_split and not 0 < validation_split < 1:
        raise ValueError(
            "`validation_split` must be between 0 and 1, "
            f"received: {validation_split}"
        )
    if (validation_split or subset) and not (validation_split and subset):
        raise ValueError(
            "If `subset` is set, `validation_split` must be set, and inversely."
        )
    if subset not in ("training", "validation", "both", None):
        raise ValueError(
            '`subset` must be either "training", '
            f'"validation" or "both", received: {subset}'
        )
    if validation_split and shuffle and seed is None:
        raise ValueError(
            "If using `validation_split` and shuffling the data, you must "
            "provide a `seed` argument, to make sure that there is no "
            "overlap between the training and validation subset."
        )
