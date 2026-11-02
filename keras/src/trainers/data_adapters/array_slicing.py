import collections
import math

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils.module_utils import tensorflow as tf


class Sliceable:
    """`Sliceable` wrapping a tensor.

    A `Sliceable` implements the subscript operator to slice or index against
    the first dimension of the array. It also has conversion methods for each
    one of the backends.

    Args:
        array: the native array or tensor to wrap.

    Attributes:
        shape: the shape of the full dense native array.
    """

    def __init__(self, array):
        self.array = array

    def __getitem__(self, indices):
        """Select elements in the 0th dimension.

        Args:
            indices: the indices to select. Only needs to support one dimension,
                the 0th dimension. Should support a `slice` or a list, tuple,
                `np.array` or 1D tensor.
        Returns: A slice of `self.array`.
        """
        return self.array[indices]

    @classmethod
    def cast(cls, x, dtype):
        """Cast a tensor to a different dtype.

        Only called on a full array as provided by the user.

        Args:
            x: the tensor to cast.
        Returns: the cast tensor.
        """
        return x.astype(dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        """Convert a tensor to a NumPy array.

        Only called after slicing using `__getitem__`.

        Args:
            x: the tensor to convert.
        Returns: the converted tensor.
        """
        return x

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        """Convert a tensor to something compatible with `tf.data.Dataset`.

        This can be a NumPy array, `tf.Tensor` or any other type of tensor that
        `tf.data.Dataset.from_tensors` can consume.
        Only called on a full array as provided by the user.

        Args:
            x: the tensor to convert.
        Returns: converted version tensor.
        """
        return x

    @classmethod
    def convert_to_jax_compatible(cls, x):
        """Convert a tensor to something that the JAX backend can consume.

        This can be a `JAX` array, `JAXSparse` or a NumPy array.
        Only called after slicing using `__getitem__`.
        Used to convert sparse tensors and densify ragged tensors.

        Args:
            x: the tensor to convert.
        Returns: the converted tensor.
        """
        return x

    @classmethod
    def convert_to_torch_compatible(cls, x):
        """Convert a tensor to something that the Torch backend can consume.

        This can be a Torch tensor, NumPy array or any other type of tensor that
        `keras.backend.torch.core.convert_to_tensor()` can consume.
        Only called after slicing using `__getitem__`.
        Used to densify sparse tensors and ragged tensors.

        Args:
            x: the tensor to convert.
        Returns: the converted tensor.
        """
        return x

    @classmethod
    def convert_to_native_compatible(cls, x):
        """Convert a tensor to something that the native backend can consume.

        Only called after slicing using `__getitem__`.
        Used to densify sparse tensors and ragged tensors.

        Args:
            x: the tensor to convert.
        Returns: the converted tensor.
        """
        return x


class NumpySliceable(Sliceable):
    pass


class NativeArraySliceable(Sliceable):
    def __getitem__(self, indices):
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        return self.array[indices]

    @classmethod
    def cast(cls, x, dtype):
        return backend.ops.cast(x, dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        return backend.ops.convert_to_numpy(x)


class TensorflowSliceable(Sliceable):
    def __getitem__(self, indices):
        from keras.src.utils.module_utils import tensorflow as tf

        if isinstance(indices, slice):
            return self.array[indices]
        else:
            return tf.gather(self.array, indices, axis=0)

    @classmethod
    def cast(cls, x, dtype):
        from keras.src.backend.tensorflow.ops.core import cast

        return cast(x, dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.tensorflow.ops.core import convert_to_numpy

        return convert_to_numpy(x)


class TensorflowRaggedSliceable(TensorflowSliceable):
    @classmethod
    def convert_to_jax_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return x.to_tensor()

    @classmethod
    def convert_to_native_compatible(cls, x):
        return x.to_tensor()


class TensorflowSparseSliceable(TensorflowSliceable):
    def __init__(self, array):
        super().__init__(to_tensorflow_sparse_wrapper(array))

    @property
    def shape(self):
        return self.array.sparse.shape

    def __getitem__(self, indices):
        return slice_tensorflow_sparse_wrapper(self.array, indices)

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return to_tensorflow_sparse_wrapper(x)

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return data_adapter_utils.tf_sparse_to_jax_sparse(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        from keras.src.backend.tensorflow import sparse as tf_sparse

        return tf_sparse.sparse_to_dense(x)

    @classmethod
    def convert_to_native_compatible(cls, x):
        from keras.src.backend.tensorflow import sparse as tf_sparse

        return tf_sparse.sparse_to_dense(x)


class JaxSparseSliceable(Sliceable):
    def __getitem__(self, indices):
        return self.array[indices, ...]

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.jax.ops.core import convert_to_numpy

        return convert_to_numpy(x)

    @classmethod
    def convert_to_tf_dataset_compatible(cls, array):
        return to_tensorflow_sparse_wrapper(
            data_adapter_utils.jax_sparse_to_tf_sparse(array)
        )

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return x.todense()

    @classmethod
    def convert_to_native_compatible(cls, x):
        return x.todense()


class TorchSliceable(Sliceable):
    @classmethod
    def cast(cls, x, dtype):
        from keras.src.backend.torch.ops.core import cast

        return cast(x, dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.torch.ops.core import convert_to_numpy

        return convert_to_numpy(x)


class PandasSliceable(Sliceable):
    def __getitem__(self, indices):
        return self.array.iloc[indices]

    @classmethod
    def convert_to_numpy(cls, x):
        return x.to_numpy()

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_native_compatible(cls, x):
        return cls.convert_to_numpy(x)


class PandasDataFrameSliceable(PandasSliceable):
    pass


class PandasSeriesSliceable(PandasSliceable):
    @classmethod
    def convert_to_numpy(cls, x):
        return np.expand_dims(x.to_numpy(), axis=-1)


class ScipySparseSliceable(Sliceable):
    def __init__(self, array):
        # The COO representation is not indexable / sliceable and does not lend
        # itself to it. Use the CSR representation instead, which is sliceable.
        super().__init__(array.tocsr())

    @classmethod
    def convert_to_numpy(cls, x):
        return x.todense()

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return to_tensorflow_sparse_wrapper(
            data_adapter_utils.scipy_sparse_to_tf_sparse(x)
        )

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return data_adapter_utils.scipy_sparse_to_jax_sparse(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return x.todense()

    @classmethod
    def convert_to_native_compatible(cls, x):
        return x.todense()


# `tf.SparseTensor` does not support indexing or `tf.gather`. The COO
# representation it uses does not lend itself to indexing. We add some
# intermediary tensors to ease the indexing and slicing. We put both indices and
# values in `RaggedTensor`s where each row corresponds to a row in the sparse
# tensor. This is because the number of values per row is not fixed.
# `RaggedTensor`s do support indexing and `tf.gather`, although on CPU only.
# We then reconstruct a `SparseTensor` from extracted rows. In theory, there is
# no duplication of data for the indices and values, only the addition of row
# splits for the ragged representation.
# `TensorflowSparseWrapper` is a named tuple which combines the original
# `SparseTensor` (used for the shape) and the ragged representations of indices
# and values for indexing / slicing. We use a named tuple and not a `Sliceable`
# to be able to ingest it in `tf.data.Dataset.from_tensors()` and map it.

TensorflowSparseWrapper = collections.namedtuple(
    "TensorflowSparseWrapper", ["sparse", "ragged_indices", "ragged_values"]
)


def to_tensorflow_sparse_wrapper(sparse):
    from keras.src.utils.module_utils import tensorflow as tf

    row_ids = sparse.indices[:, 0]
    row_splits = tf.experimental.RowPartition.from_value_rowids(
        row_ids, nrows=sparse.dense_shape[0]
    ).row_splits()

    ragged_indices = tf.cast(
        tf.RaggedTensor.from_row_splits(sparse.indices, row_splits), tf.int64
    )
    ragged_values = tf.RaggedTensor.from_row_splits(sparse.values, row_splits)
    return TensorflowSparseWrapper(sparse, ragged_indices, ragged_values)


def slice_tensorflow_sparse_wrapper(sparse_wrapper, indices):
    from keras.src.utils.module_utils import tensorflow as tf

    if isinstance(indices, slice):
        sparse_indices = sparse_wrapper.ragged_indices[indices]
        sparse_values = sparse_wrapper.ragged_values[indices]
        batch_dim = indices.stop - indices.start
    else:
        sparse_indices = tf.gather(sparse_wrapper.ragged_indices, indices)
        sparse_values = tf.gather(sparse_wrapper.ragged_values, indices)
        if isinstance(indices, list):
            batch_dim = len(indices)
        else:
            batch_dim = indices.shape[0]
            if batch_dim is None:
                batch_dim = tf.shape(indices)[0]

    row_ids = sparse_indices.value_rowids()
    sparse_indices = sparse_indices.flat_values[:, 1:]  # remove first value
    sparse_indices = tf.concat(
        [tf.expand_dims(row_ids, -1), sparse_indices], axis=1
    )

    sparse_values = sparse_values.flat_values
    sparse_shape = (batch_dim,) + tuple(
        sparse_wrapper.sparse.shape.as_list()[1:]
    )
    return tf.SparseTensor(sparse_indices, sparse_values, sparse_shape)


def can_slice_array(x):
    return (
        x is None
        or isinstance(x, np.ndarray)
        or data_adapter_utils.is_tensorflow_tensor(x)
        or data_adapter_utils.is_jax_array(x)
        or data_adapter_utils.is_torch_tensor(x)
        or data_adapter_utils.is_scipy_sparse(x)
        or data_adapter_utils.is_pandas_data_frame(x)
        or data_adapter_utils.is_pandas_series(x)
        or backend.ops.is_tensor(x)
        or hasattr(x, "__array__")
    )


def convert_to_sliceable(arrays, target_backend=None):
    """Convert a structure of arrays into `Sliceable` instances

    Args:
        arrays: the arrays to convert.
        target_backend: the target backend for the output:
            - `None` indicates that `arrays` will be wrapped into `Sliceable`s
              as-is without using a different representation. This is used by
              `train_validation_split()`.
            - `tensorflow` indicates that
              `Sliceable.convert_to_tf_dataset_compatible` will be called. The
              returned structure therefore contains arrays, not `Sliceable`s.
            - `numpy`, `jax` or `torch` indices that the arrays will eventually
              be converted to this backend type after slicing. In this case,
              the intermediary `Sliceable`s may use a different representation
              from the input `arrays` for better performance.
    Returns: the same structure with `Sliceable` instances or arrays.
    """

    def convert_single_array(x):
        if x is None:
            return x

        # Special case: handle np "object" arrays containing strings
        if (
            isinstance(x, np.ndarray)
            and str(x.dtype) == "object"
            and backend.backend() == "tensorflow"
            and all(isinstance(e, str) for e in x)
        ):
            x = tf.convert_to_tensor(x, dtype="string")

        # Step 1. Determine which Sliceable class to use.
        if isinstance(x, np.ndarray):
            sliceable_class = NumpySliceable
        elif data_adapter_utils.is_tensorflow_tensor(x):
            if data_adapter_utils.is_tensorflow_ragged(x):
                sliceable_class = TensorflowRaggedSliceable
            elif data_adapter_utils.is_tensorflow_sparse(x):
                sliceable_class = TensorflowSparseSliceable
            else:
                sliceable_class = TensorflowSliceable
        elif data_adapter_utils.is_jax_array(x):
            if data_adapter_utils.is_jax_sparse(x):
                sliceable_class = JaxSparseSliceable
            else:
                x = np.asarray(x)
                sliceable_class = NumpySliceable
        elif data_adapter_utils.is_torch_tensor(x):
            sliceable_class = TorchSliceable
        elif data_adapter_utils.is_pandas_data_frame(x):
            sliceable_class = PandasDataFrameSliceable
        elif data_adapter_utils.is_pandas_series(x):
            sliceable_class = PandasSeriesSliceable
        elif data_adapter_utils.is_scipy_sparse(x):
            sliceable_class = ScipySparseSliceable
        elif backend.ops.is_tensor(x):
            sliceable_class = NativeArraySliceable
        elif hasattr(x, "__array__"):
            x = np.asarray(x)
            sliceable_class = NumpySliceable
        else:
            raise ValueError(
                "Expected a NumPy array, tf.Tensor, tf.RaggedTensor, "
                "tf.SparseTensor, jax.np.ndarray, "
                "jax.experimental.sparse.JAXSparse, torch.Tensor, "
                "Pandas Dataframe, or Pandas Series. Received invalid input: "
                f"{x} (of type {type(x)})"
            )

        # Step 2. Normalize floats to floatx.
        def is_non_floatx_float(dtype):
            return (
                dtype is not object
                and is_float_dtype(dtype)
                and not standardize_dtype(dtype) == backend.floatx()
            )

        cast_dtype = None
        if data_adapter_utils.is_pandas_data_frame(x):
            if any(is_non_floatx_float(d) for d in x.dtypes.values):
                cast_dtype = backend.floatx()
        else:
            if is_non_floatx_float(x.dtype):
                cast_dtype = backend.floatx()

        if cast_dtype is not None:
            x = sliceable_class.cast(x, cast_dtype)

        # Step 3. Apply target backend specific logic and optimizations.
        if target_backend is None:
            return sliceable_class(x)

        if target_backend == "tensorflow":
            return sliceable_class.convert_to_tf_dataset_compatible(x)

        # With dense arrays and JAX as output, it is faster to use NumPy as an
        # intermediary representation, so wrap input array in a NumPy array,
        # which should not use extra memory.
        # See https://github.com/google/jax/issues/1276 for an explanation of
        # why slicing a NumPy array is faster than slicing a JAX array.
        if target_backend == "jax" and sliceable_class in (
            TensorflowSliceable,
            TorchSliceable,
        ):
            x = np.asarray(x)
            sliceable_class = NumpySliceable

        return sliceable_class(x)

    return tree.map_structure(convert_single_array, arrays)


def train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

    The last part of data will become validation data.

    Args:
        arrays: Tensors to split. Allowed inputs are arbitrarily nested
            structures of Tensors and NumPy arrays.
        validation_split: Float between 0 and 1. The proportion of the dataset
            to include in the validation split. The rest of the dataset will be
            included in the training split.

    Returns:
        `(train_arrays, validation_arrays)`
    """

    flat_arrays = tree.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not can_slice_array(t)]
    if unsplitable:
        raise ValueError(
            "Argument `validation_split` is only supported "
            "for tensors or NumPy arrays."
            f"Found incompatible type in the input: {unsplitable}"
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            f"Training data contains {batch_dim} samples, which is not "
            "sufficient to split it into a validation and training set as "
            f"specified by `validation_split={validation_split}`. Either "
            "provide more data, or a different value for the "
            "`validation_split` argument."
        )

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]

    sliceables = convert_to_sliceable(arrays)
    train_arrays = tree.map_structure(
        lambda x: _split(x, start=0, end=split_at), sliceables
    )
    val_arrays = tree.map_structure(
        lambda x: _split(x, start=split_at, end=batch_dim), sliceables
    )
    return train_arrays, val_arrays
