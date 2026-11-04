import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype

NUM_BATCHES_FOR_TENSOR_SPEC = 2


@keras_export("keras.utils.unpack_x_y_sample_weight")
def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple.

    This is a convenience utility to be used when overriding
    `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
    This utility makes it easy to support data of the form `(x,)`,
    `(x, y)`, or `(x, y, sample_weight)`.

    Example:

    >>> features_batch = ops.ones((10, 5))
    >>> labels_batch = ops.zeros((10, 5))
    >>> data = (features_batch, labels_batch)
    >>> # `y` and `sample_weight` will default to `None` if not provided.
    >>> x, y, sample_weight = unpack_x_y_sample_weight(data)
    >>> sample_weight is None
    True

    Args:
        data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

    Returns:
        The unpacked tuple, with `None`s for `y` and `sample_weight` if they are
        not provided.
    """
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    error_msg = (
        "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
        f"or `(x, y, sample_weight)`, found: {data}"
    )
    raise ValueError(error_msg)


@keras_export("keras.utils.pack_x_y_sample_weight")
def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.

    This is a convenience utility for packing data into the tuple formats
    that `Model.fit()` uses.

    Example:

    >>> x = ops.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x)
    >>> isinstance(data, ops.Tensor)
    True
    >>> y = ops.ones((10, 1))
    >>> data = pack_x_y_sample_weight(x, y)
    >>> isinstance(data, tuple)
    True
    >>> x, y = data

    Args:
        x: Features to pass to `Model`.
        y: Ground-truth targets to pass to `Model`.
        sample_weight: Sample weight for each element.

    Returns:
        Tuple in the format used in `Model.fit()`.
    """
    if y is None:
        # For single x-input, we do no tuple wrapping since in this case
        # there is no ambiguity. This also makes NumPy and Dataset
        # consistent in that the user does not have to wrap their Dataset
        # data in an unnecessary tuple.
        if not isinstance(x, (tuple, list)):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def list_to_tuple(maybe_list):
    """Datasets will stack any list of tensors, so we convert them to tuples."""
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list


def check_data_cardinality(data):
    num_samples = set(
        int(i.shape[0]) for i in tree.flatten(data) if i is not None
    )
    if len(num_samples) > 1:
        msg = (
            "Data cardinality is ambiguous. "
            "Make sure all arrays contain the same number of samples."
        )
        for label, single_data in zip(["x", "y", "sample_weight"], data):
            sizes = ", ".join(
                str(i.shape[0]) for i in tree.flatten(single_data)
            )
            msg += f"'{label}' sizes: {sizes}\n"
        raise ValueError(msg)


def class_weight_to_sample_weights(y, class_weight):
    # Validate and normalize class_weight keys to integer class indices.
    # Without this check, invalid keys are silently ignored, and string keys
    # (e.g., "0") would fail to match integer lookups later, silently
    # defaulting sample_weight to 1.0.
    if not hasattr(class_weight, "items") or not hasattr(class_weight, "get"):
        raise ValueError(
            "`class_weight` must be a dict-like mapping of class indices "
            f"to weights. Received: class_weight={class_weight} of type "
            f"{type(class_weight)}."
        )

    cleaned_class_weight = {}
    for key, val in class_weight.items():
        try:
            cleaned_class_weight[int(key)] = val
        except (TypeError, ValueError):
            raise ValueError(
                "`class_weight` must be a dictionary with integer keys "
                "representing class indices. Received an invalid key: "
                f"{key!r} of type {type(key)}."
            )
    class_weight = cleaned_class_weight

    # Convert to numpy to ensure consistent handling of operations
    # (e.g., np.round()) across frameworks like TensorFlow, JAX, and PyTorch

    y_numpy = ops.convert_to_numpy(y)
    sample_weight = np.ones(shape=(y_numpy.shape[0],), dtype=backend.floatx())
    if len(y_numpy.shape) > 1:
        if y_numpy.shape[-1] != 1:
            y_numpy = np.argmax(y_numpy, axis=-1)
        else:
            y_numpy = np.squeeze(y_numpy, axis=-1)
    y_numpy = np.round(y_numpy).astype("int32")

    for i in range(y_numpy.shape[0]):
        sample_weight[i] = class_weight.get(int(y_numpy[i]), 1.0)
    return sample_weight


def get_keras_tensor_spec(batches):
    """Return the KerasTensor spec for a list of batches.

    The spec is represented using `KerasTensor` which could handle dense, sparse
    or ragged tensors.

    Args:
        batches: list of structures of tensors. The structures must be
            identical, but the shape at each leaf may be different.

    Returns:
        A nested structure of `KerasTensor`.
    """

    def get_single_tensor_spec(*tensors):
        x = tensors[0]
        if not hasattr(x, "shape"):
            # Try to convert to a numpy array.
            x = np.array(x)
        rank = len(x.shape)
        if rank < 1:
            raise ValueError(
                "When passing a dataset to a Keras model, the arrays must "
                f"be at least rank 1. Received: {x} of rank {len(x.shape)}."
            )
        for t in tensors:
            if len(t.shape) != rank:
                raise ValueError(
                    "When passing a dataset to a Keras model, the "
                    "corresponding arrays in each batch must have the same "
                    f"rank. Received: {x} and {t}"
                )
        shape = []
        # Merge shapes: go through each dimension one by one and keep the
        # common values
        for dims in zip(*[list(x.shape) for x in tensors]):
            dims_set = set(dims)
            shape.append(dims_set.pop() if len(dims_set) == 1 else None)

        dtype = standardize_dtype(x.dtype)
        if is_tensorflow_ragged(x):
            return backend.KerasTensor(
                shape=shape,
                dtype=dtype,
                ragged=True,
                ragged_rank=x.ragged_rank,
                row_splits_dtype=x.row_splits.dtype,
            )
        if is_tensorflow_sparse(x) or is_scipy_sparse(x) or is_jax_sparse(x):
            return backend.KerasTensor(shape=shape, dtype=dtype, sparse=True)
        else:
            return backend.KerasTensor(shape=shape, dtype=dtype)

    return tree.map_structure(
        get_single_tensor_spec, *batches, none_is_leaf=False
    )


def convert_to_tf_tensor_spec(keras_tensor, batch_axis_to_none=True):
    """Convert a KerasTensor to a TensorSpec.

    Args:
        keras_tensor: A KerasTensor instance.
        batch_axis_to_none: If `True`, the batch axis of the returned
            tensor spec will be set to None. Defaults to `True`.
    """
    from keras.src.utils.module_utils import tensorflow as tf

    if keras_tensor is None:
        return tf.OptionalSpec(None)
    if not isinstance(keras_tensor, backend.KerasTensor):
        raise TypeError(
            f"Expected a KerasTensor, but got {keras_tensor} of type "
            f"{type(keras_tensor)}."
        )
    shape = list(keras_tensor.shape)
    if batch_axis_to_none:
        shape[0] = None
    if keras_tensor.ragged:
        return tf.RaggedTensorSpec(
            shape=shape,
            dtype=keras_tensor.dtype,
            ragged_rank=keras_tensor.ragged_rank,
            row_splits_dtype=keras_tensor.row_splits_dtype,
        )
    elif keras_tensor.sparse:
        return tf.SparseTensorSpec(shape=shape, dtype=keras_tensor.dtype)
    else:
        return tf.TensorSpec(shape=shape, dtype=keras_tensor.dtype)


def get_tensor_spec(batches):
    """Return the common tensor spec for a list of batches.

    The spec is represented using `tf.TensorSpec`, `tf.SparseTensorSpec` and
    `tf.RaggedTensorSpec`.

    Args:
        batches: list of structures of tensors. The structures must be
            identical, but the shape at each leaf may be different.

    Returns:
        A common tensor spec.
    """
    tensor_specs = get_keras_tensor_spec(batches)
    return tree.map_structure(convert_to_tf_tensor_spec, tensor_specs)


def get_jax_iterator(iterable):
    import jax
    import jax.experimental.sparse as jax_sparse

    def convert_to_jax_compatible(x):
        if isinstance(x, (jax.Array, jax_sparse.JAXSparse, np.ndarray)):
            return x
        elif is_scipy_sparse(x):
            return scipy_sparse_to_jax_sparse(x)
        elif is_tensorflow_sparse(x):
            return tf_sparse_to_jax_sparse(x)
        else:
            return np.asarray(x)

    for batch in iterable:
        yield tree.map_structure(
            convert_to_jax_compatible, batch, none_is_leaf=False
        )


def get_numpy_iterator(iterable):
    def convert_to_numpy(x):
        if not isinstance(x, np.ndarray):
            if backend.ops.is_tensor(x):
                # The backend knows how to convert its own tensors, including
                # the ones `__array__` cannot, e.g. a bfloat16 MLX array.
                x = ops.convert_to_numpy(x)
            elif hasattr(x, "__array__"):
                # Using `__array__` should handle `tf.Tensor`,
                # `jax.np.ndarray`, `torch.Tensor`, as well as any other
                # tensor-like object that has added numpy support.
                if is_torch_tensor(x):
                    x = x.cpu()
                x = np.asarray(x)
        return x

    for batch in iterable:
        yield tree.map_structure(convert_to_numpy, batch, none_is_leaf=False)


def get_torch_dataloader(iterable):
    import torch.utils.data as torch_data

    from keras.src.backend.torch.ops.core import convert_to_tensor

    class ConverterIterableDataset(torch_data.IterableDataset):
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            for batch in self.iterable:
                yield tree.map_structure(
                    convert_to_tensor, batch, none_is_leaf=False
                )

    dataset = ConverterIterableDataset(iterable)
    # `batch_size=None` indicates that we should not re-batch
    return torch_data.DataLoader(dataset, batch_size=None)


def is_tensorflow_tensor(value):
    if hasattr(value, "__class__"):
        if value.__class__.__name__ in ("RaggedTensor", "SparseTensor"):
            return "tensorflow.python." in str(value.__class__.__module__)
        for parent in value.__class__.__mro__:
            if parent.__name__ in ("Tensor") and "tensorflow.python." in str(
                parent.__module__
            ):
                return True
    return False


def is_tensorflow_ragged(value):
    if hasattr(value, "__class__"):
        return (
            value.__class__.__name__ == "RaggedTensor"
            and "tensorflow.python." in str(value.__class__.__module__)
        )
    return False


def is_tensorflow_sparse(value):
    if hasattr(value, "__class__"):
        return (
            value.__class__.__name__ == "SparseTensor"
            and "tensorflow.python." in str(value.__class__.__module__)
        )
    return False


def is_jax_array(value):
    if hasattr(value, "__class__"):
        for parent in value.__class__.__mro__:
            if parent.__name__.endswith("Array") and str(
                parent.__module__
            ).startswith("jax"):
                return True
    return is_jax_sparse(value)  # JAX sparse arrays do not extend jax.Array


def is_jax_sparse(value):
    if hasattr(value, "__class__"):
        return str(value.__class__.__module__).startswith(
            "jax.experimental.sparse"
        )
    return False


def is_torch_tensor(value):
    if hasattr(value, "__class__"):
        for parent in value.__class__.__mro__:
            if parent.__name__ == "Tensor" and str(parent.__module__).endswith(
                "torch"
            ):
                return True
    return False


def is_pandas_data_frame(value):
    return (
        hasattr(value, "__class__")
        and value.__class__.__name__ == "DataFrame"
        and str(value.__class__.__module__).startswith("pandas")
    )


def is_pandas_series(value):
    return (
        hasattr(value, "__class__")
        and value.__class__.__name__ == "Series"
        and str(value.__class__.__module__).startswith("pandas")
    )


class DistributedBatchSampler:
    """Sampler that shards a batch sampler.

    This sampler is useful for sharding an existing batch sampler (an iterable
    of batches) across multiple data shards.

    This is intentionally a batch-level sharding helper for cases where the
    underlying iterator already produces full batches. In `ArrayDataAdapter`, we
    build batch samplers over the dataset and then may optionally shuffle within
    each returned batch.

    Args:
        batch_sampler: An iterable of batches. For example, a
            `torch.utils.data.BatchSampler` instance.
        num_data_shards: Total number of data shards.
        data_shard_id: ID of the current data shard.
        drop_last: Whether to drop the last partial batch.
    """

    def __init__(
        self, batch_sampler, num_data_shards, data_shard_id, drop_last=False
    ):
        """Initialize the DistributedBatchSampler.

        Args:
            batch_sampler: An iterable of batches.
            num_data_shards: Total number of data shards.
            data_shard_id: ID of the current data shard.
            drop_last: Whether to drop the last partial batch.
        """
        if num_data_shards < 1:
            raise ValueError(
                f"num_data_shards must be >= 1. Received: {num_data_shards}"
            )
        if not (0 <= data_shard_id < num_data_shards):
            raise ValueError(
                f"data_shard_id must be in [0, {num_data_shards - 1}]. "
                f"Received: {data_shard_id}"
            )
        self.batch_sampler = batch_sampler
        self.num_data_shards = num_data_shards
        self.data_shard_id = data_shard_id
        self.drop_last = drop_last

    def __iter__(self):
        sampler_len = len(self.batch_sampler)

        limit = (
            (sampler_len // self.num_data_shards) * self.num_data_shards
            if self.drop_last
            else sampler_len
        )

        import torch

        # Optimization: if the batch_sampler is a standard BatchSampler
        # with an indexable sampler, we can skip indices efficiently.
        if isinstance(self.batch_sampler, torch.utils.data.BatchSampler):
            sampler = self.batch_sampler.sampler
            if hasattr(sampler, "__getitem__") and not isinstance(
                sampler, torch.utils.data.Sampler
            ):
                for i in range(self.data_shard_id, limit, self.num_data_shards):
                    # Manual index-based access is more efficient than filtering
                    # the iterator.
                    yield sampler[
                        i * self.batch_sampler.batch_size : (i + 1)
                        * self.batch_sampler.batch_size
                    ]
                return

        for i, batch in enumerate(self.batch_sampler):
            if i >= limit:
                break
            if i % self.num_data_shards == self.data_shard_id:
                yield batch

    def __len__(self):
        sampler_len = len(self.batch_sampler)

        if self.drop_last:
            return sampler_len // self.num_data_shards
        return (
            sampler_len - self.data_shard_id + self.num_data_shards - 1
        ) // self.num_data_shards


def is_scipy_sparse(x):
    return str(x.__class__.__module__).startswith("scipy.sparse") and hasattr(
        x, "tocoo"
    )


def scipy_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf

    coo = x.tocoo()
    indices = np.concatenate(
        (np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)), axis=1
    )
    return tf.SparseTensor(indices, coo.data, coo.shape)


def scipy_sparse_to_jax_sparse(x):
    import jax
    import jax.experimental.sparse as jax_sparse

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        return jax_sparse.BCOO.from_scipy_sparse(x)


def tf_sparse_to_jax_sparse(x):
    import jax
    import jax.experimental.sparse as jax_sparse

    values = np.asarray(x.values)
    indices = np.asarray(x.indices)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        return jax_sparse.BCOO((values, indices), shape=x.shape)


def jax_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf

    return tf.SparseTensor(x.indices, x.data, x.shape)


class PartialBatchList(list):
    """Container for unstacked remainder batches from super_batch_iterator."""

    pass


def super_batch_iterator(iterator, super_batch, stack_fn=np.stack):
    """Wraps an iterator to stack batches into super-batches along axis 0."""
    while True:
        batches = []
        try:
            for _ in range(super_batch):
                batches.append(next(iterator))
        except StopIteration:
            pass

        if not batches:
            break

        for leaf in tree.flatten(batches[0]):
            if (
                is_scipy_sparse(leaf)
                or is_tensorflow_sparse(leaf)
                or is_jax_sparse(leaf)
            ):
                raise ValueError(
                    "`steps_per_execution > 1` is not supported with sparse "
                    "tensors."
                )

        if len(batches) == super_batch:
            try:

                def _stack_leaf(*xs):
                    if xs[0] is None:
                        return None
                    return stack_fn(xs)

                yield tree.map_structure(_stack_leaf, *batches)
                continue
            except (ValueError, RuntimeError):
                pass

        yield PartialBatchList(batches)
