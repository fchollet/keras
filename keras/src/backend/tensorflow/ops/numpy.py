import builtins
import collections
import functools
import math
import string
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.ops.math_ops import is_nan

from keras.src import tree
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import normalize_shift_and_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.common.backend_utils import vectorize_impl
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.ops.core import cast
from keras.src.backend.tensorflow.ops.core import convert_to_tensor
from keras.src.backend.tensorflow.ops.core import shape as shape_op


def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the specified plane.

    Args:
        array: Input tensor
        k: Number of 90-degree rotations (default=1)
        axes: Tuple of two axes that define the plane of rotation.
        Defaults to (0, 1).

    Returns:
        Rotated tensor with correct shape transformation
    """
    array = convert_to_tensor(array)

    if array.shape.rank < 2:
        raise ValueError(
            f"Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.shape.rank}"
        )

    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple of "
            "two different dimensions."
        )

    k = k % 4
    if k == 0:
        return array

    axes = tuple(
        axis if axis >= 0 else array.shape.rank + axis for axis in axes
    )

    other_axes = [i for i in range(array.shape.rank) if i not in axes]
    perm = other_axes + list(axes)
    array = tf.transpose(array, perm)

    if k == 1:
        array = tf.reverse(array, axis=[-1])
        array = swapaxes(array, -1, -2)
    elif k == 2:
        array = tf.reverse(array, axis=[-2, -1])
    elif k == 3:
        array = tf.reverse(array, axis=[-2])
        array = swapaxes(array, -1, -2)

    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    array = tf.transpose(array, inv_perm)

    return array


@sparse.elementwise_binary_union(tf.sparse.add)
def add(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    if (
        x1.shape.rank is None
        or x2.shape.rank is None
        or dtype == "string"
        or isinstance(x1, tf.RaggedTensor)
        or isinstance(x2, tf.RaggedTensor)
    ):
        return tf.add(x1, x2)

    x1_shape = x1.shape.as_list()
    x1_rank = len(x1_shape)
    x2_shape = x2.shape.as_list()
    x2_rank = len(x2_shape)

    # Special case of `tf.add`: `tf.nn.bias_add`
    # `BiasAdd` can be fused with `MatMul` and `Conv*` kernels
    # Expecting `x1` to be `inputs` and `x2` to be `bias` (no swapping)
    if x1_rank > 1 and x2_rank == 1:
        # Detect the dense layer case (`MatMul` + `BiasAdd` fusion)
        if x2_shape[0] is not None and x2_shape[0] == x1_shape[-1]:
            return tf.nn.bias_add(x1, x2)
    elif x1_rank >= 3 and x1_rank == x2_rank:
        # Detect the conv layer case (`Conv` + `BiasAdd` fusion)
        channels_idx = None
        dim_one_count = 0
        for i, d in enumerate(x2_shape):
            if d == 1:
                dim_one_count += 1
            else:
                channels_idx = i

        if dim_one_count == x2_rank - 1:
            # channels_idx points to the one dimension that is not 1.
            channels_dim = x2_shape[channels_idx]
        elif dim_one_count == x2_rank:
            # All dimensions are 1, solve the ambiguity by checking x1 shape.
            channels_dim = 1
            if x1_shape[-1] == 1:
                channels_idx = x1_rank - 1
            elif x1_shape[1] == 1:
                channels_idx = 1
        else:
            # Note: channels_idx is bogus and unused.
            channels_dim = None

        if (
            channels_dim is not None
            and channels_idx in (1, x2_rank - 1)
            and x1_shape[channels_idx] == channels_dim
        ):
            x2 = tf.squeeze(
                x2, axis=[i for i in range(x2_rank) if i != channels_idx]
            )
            if channels_idx == 1:
                return tf.nn.bias_add(x1, x2, data_format="NCHW")
            elif channels_idx == x2_rank - 1:
                return tf.nn.bias_add(x1, x2, data_format="NHWC")

    return tf.add(x1, x2)


def bartlett(x):
    x = convert_to_tensor(x, dtype=config.floatx())
    if x == 0:
        return tf.constant([])
    if x == 1:
        return tf.ones([1])

    n = tf.range(x)
    half = (x - 1) / 2

    window = tf.where(n <= half, 2.0 * n / (x - 1), 2.0 - 2.0 * n / (x - 1))

    return window


def hamming(x):
    x = convert_to_tensor(x, dtype=tf.int32)
    return tf.signal.hamming_window(x, periodic=False)


def hanning(x):
    x = convert_to_tensor(x, dtype=tf.int32)
    return tf.signal.hann_window(x, periodic=False)


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"

    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    return tf.where(
        x1 < 0,
        tf.zeros_like(x1),
        tf.where(x1 > 0, tf.ones_like(x1), x2),
    )


def kaiser(x, beta):
    x = convert_to_tensor(x, dtype=tf.int32)
    return tf.signal.kaiser_window(x, beta=beta)


def bincount(x, weights=None, minlength=0, sparse=False):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype]
    if standardize_dtype(x.dtype) not in ["int32", "int64"]:
        x = tf.cast(x, tf.int32)
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
        if standardize_dtype(weights.dtype) not in [
            "int32",
            "int64",
            "float32",
            "float64",
        ]:
            if "int" in standardize_dtype(weights.dtype):
                weights = tf.cast(weights, tf.int32)
            else:
                weights = tf.cast(weights, tf.float32)
    else:
        dtype = "int32"
    if sparse or isinstance(x, tf.SparseTensor):
        output = tf.sparse.bincount(
            x,
            weights=weights,
            minlength=minlength,
            axis=-1,
        )
        actual_length = output.shape[-1]
        if actual_length is None:
            actual_length = tf.shape(output)[-1]
        output = cast(output, dtype)
        if x.shape.rank == 1:
            output_shape = (actual_length,)
        else:
            batch_size = output.shape[0]
            if batch_size is None:
                batch_size = tf.shape(output)[0]
            output_shape = (batch_size, actual_length)
        return tf.SparseTensor(
            indices=output.indices,
            values=output.values,
            dense_shape=output_shape,
        )
    return tf.cast(
        tf.math.bincount(x, weights=weights, minlength=minlength, axis=-1),
        dtype,
    )


@functools.lru_cache(512)
def _normalize_einsum_subscripts(subscripts):
    # string.ascii_letters
    mapping = {}
    normalized_subscripts = ""
    for c in subscripts:
        if c in string.ascii_letters:
            if c not in mapping:
                mapping[c] = string.ascii_letters[len(mapping)]
            normalized_subscripts += mapping[c]
        else:
            normalized_subscripts += c
    return normalized_subscripts


def einsum(subscripts, *operands, **kwargs):
    operands = tree.map_structure(convert_to_tensor, operands)
    subscripts = _normalize_einsum_subscripts(subscripts)

    def is_valid_for_custom_ops(subscripts, *operands):
        # Check that `subscripts` is supported and the shape of operands is not
        # `None`.
        if subscripts in [
            "a,b->ab",
            "ab,b->a",
            "ab,bc->ac",
            "ab,cb->ac",
            "abc,cd->abd",
            "abc,dc->abd",
            "abcd,abde->abce",
            "abcd,abed->abce",
            "abcd,acbe->adbe",
            "abcd,adbe->acbe",
            "abcd,aecd->acbe",
            "abcd,aecd->aceb",
        ]:
            # These subscripts don't require the shape information
            return True
        elif subscripts == "abc,cde->abde":
            _, b1, c1 = operands[0].shape
            c2, d2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abc,dce->abde":
            _, b1, c1 = operands[0].shape
            d2, c2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abc,dec->abde":
            _, b1, c1 = operands[0].shape
            d2, e2, c2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,cde->abe":
            _, b1, c1, d1 = operands[0].shape
            c2, d2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,ced->abe":
            _, b1, c1, d1 = operands[0].shape
            c2, e2, d2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,ecd->abe":
            _, b1, c1, d1 = operands[0].shape
            e2, c2, d2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcde,aebf->adbcf":
            _, b1, c1, d1, e1 = operands[0].shape
            _, e2, b2, f2 = operands[1].shape
            b, c, d, e, f = b1 or b2, c1, d1, e1 or e2, f2
            if None in (b, c, d, e, f):
                return False
            return True
        elif subscripts == "abcde,afce->acdbf":
            _, b1, c1, d1, e1 = operands[0].shape
            _, f2, c2, e2 = operands[1].shape
            b, c, d, e, f = b1, c1 or c2, d1, e1 or e2, f2
            if None in (b, c, d, e, f):
                return False
            return True
        else:
            # No match in subscripts
            return False

    def use_custom_ops(subscripts, *operands, output_type):
        # Replace tf.einsum with custom ops to utilize hardware-accelerated
        # matmul
        x, y = operands[0], operands[1]
        if subscripts == "a,b->ab":
            x = tf.expand_dims(x, axis=-1)
            y = tf.expand_dims(y, axis=0)
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "ab,b->a":
            y = tf.expand_dims(y, axis=-1)
            result = tf.matmul(x, y, output_type=output_type)
            return tf.squeeze(result, axis=-1)
        elif subscripts == "ab,bc->ac":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "ab,cb->ac":
            y = tf.transpose(y, [1, 0])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,cd->abd":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,cde->abde":
            _, b1, c1 = x.shape
            c2, d2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abc,dc->abd":
            y = tf.transpose(y, [1, 0])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,dce->abde":
            _, b1, c1 = x.shape
            d2, c2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.transpose(y, [1, 0, 2])  # cde
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abc,dec->abde":
            _, b1, c1 = x.shape
            d2, e2, c2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.transpose(y, [2, 0, 1])  # cde
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abcd,abde->abce":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,abed->abce":
            y = tf.transpose(y, [0, 1, 3, 2])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,acbe->adbe":
            x = tf.transpose(x, [0, 1, 3, 2])
            y = tf.transpose(y, [0, 2, 1, 3])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.transpose(result, [0, 2, 1, 3])
        elif subscripts == "abcd,adbe->acbe":
            y = tf.transpose(y, [0, 2, 1, 3])  # abde
            result = tf.matmul(x, y, output_type=output_type)  # abce
            return tf.transpose(result, [0, 2, 1, 3])
        elif subscripts == "abcd,aecd->acbe":
            x = tf.transpose(x, [0, 2, 1, 3])  # acbd
            y = tf.transpose(y, [0, 2, 3, 1])  # acde
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,aecd->aceb":
            x = tf.transpose(x, [0, 2, 1, 3])
            y = tf.transpose(y, [0, 2, 3, 1])
            result = tf.matmul(x, y, output_type=output_type)  # acbe
            return tf.transpose(result, [0, 1, 3, 2])
        elif subscripts == "abcd,cde->abe":
            _, b1, c1, d1 = x.shape
            c2, d2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,ced->abe":
            _, b1, c1, d1 = x.shape
            c2, e2, d2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.transpose(y, [0, 2, 1])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,ecd->abe":
            _, b1, c1, d1 = x.shape
            e2, c2, d2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.transpose(y, [1, 2, 0])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcde,aebf->adbcf":
            _, b1, c1, d1, e1 = x.shape
            _, e2, b2, f2 = y.shape
            b, c, d, e, f = b1 or b2, c1, d1, e1 or e2, f2
            x = tf.reshape(x, [-1, b, c * d, e])  # ab(cd)e
            y = tf.transpose(y, [0, 2, 1, 3])  # abef
            result = tf.matmul(x, y, output_type=output_type)  # ab(cd)f
            result = tf.reshape(result, [-1, b, c, d, f])  # abcdf
            return tf.transpose(result, [0, 3, 1, 2, 4])
        elif subscripts == "abcde,afce->acdbf":
            _, b1, c1, d1, e1 = x.shape
            _, f2, c2, e2 = y.shape
            b, c, d, e, f = b1, c1 or c2, d1, e1 or e2, f2
            x = tf.transpose(x, [0, 2, 3, 1, 4])  # acdbe
            x = tf.reshape(x, [-1, c, d * b, e])  # ac(db)e
            y = tf.transpose(y, [0, 2, 3, 1])  # acef
            result = tf.matmul(x, y, output_type=output_type)  # ac(db)f
            return tf.reshape(result, [-1, c, d, b, f])
        else:
            raise NotImplementedError

    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    # When operands are of int8, we cast the result to int32 to align with
    # the behavior of jax.
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        compute_dtype = "int8"
        result_dtype = "int32"
        output_type = "int32"
    else:
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
        compute_dtype = result_dtype
        output_type = None

    # TODO: Remove the condition once `tf.einsum` supports int8xint8->int32
    if is_valid_for_custom_ops(subscripts, *operands) and not kwargs:
        # TODO: tf.matmul doesn't support integer dtype if not specifying
        # output_type="int32"
        if "int" in compute_dtype and output_type is None:
            compute_dtype = config.floatx()
        operands = tree.map_structure(
            lambda x: tf.cast(x, compute_dtype), operands
        )
        result = use_custom_ops(subscripts, *operands, output_type=output_type)
    else:
        # TODO: tf.einsum doesn't support integer dtype with gpu
        if "int" in compute_dtype:
            compute_dtype = config.floatx()
        operands = tree.map_structure(
            lambda x: tf.cast(x, compute_dtype), operands
        )
        result = tf.einsum(subscripts, *operands, **kwargs)
    return tf.cast(result, result_dtype)


@sparse.elementwise_binary_union(sparse.sparse_subtract)
def subtract(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_shape = x1.shape
    x2_shape = x2.shape
    x1_sparse = isinstance(x1, tf.SparseTensor)
    x2_sparse = isinstance(x2, tf.SparseTensor)
    # When both x1 and x2 are of int8 and dense tensor, specifying `output_type`
    # as int32 to enable hardware-accelerated matmul
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if (
        x1_dtype == "int8"
        and x2_dtype == "int8"
        and not x1_sparse
        and not x2_sparse
        and x1_shape.rank != 1  # TODO: support tf.tensordot
        and x2_shape.rank != 1  # TODO: support tf.tensordot
    ):
        compute_dtype = "int8"
        result_dtype = "int32"
        output_type = result_dtype
    else:
        # TODO: Typically, GPU and XLA only support float types
        compute_dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
        result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
        output_type = None
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)

    def with_combined_batch_dimensions(a, b, output_shape, fn_3d):
        a_sparse = isinstance(a, tf.SparseTensor)
        b_sparse = isinstance(b, tf.SparseTensor)
        batch_shape = b.shape[:-2] if b_sparse else a.shape[:-2]
        batch_size = math.prod(batch_shape)
        a3d_shape = [batch_size] + a.shape[-2:]
        a_3d = (
            tf.sparse.reshape(a, a3d_shape)
            if a_sparse
            else tf.reshape(a, a3d_shape)
        )
        b3d_shape = [batch_size] + b.shape[-2:]
        b_3d = (
            tf.sparse.reshape(b, b3d_shape)
            if b_sparse
            else tf.reshape(b, b3d_shape)
        )
        result_3d = fn_3d(a_3d, b_3d)
        return (
            tf.sparse.reshape(result_3d, output_shape)
            if isinstance(result_3d, tf.SparseTensor)
            else tf.reshape(result_3d, output_shape)
        )

    def sparse_sparse_matmul(a, b):
        dtype = a.values.dtype
        # Convert SparseTensors to CSR SparseMatrix.
        a_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a.indices, a.values, a.dense_shape
        )
        b_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b.indices, b.values, b.dense_shape
        )
        # Compute the CSR SparseMatrix matrix multiplication.
        result_csr = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a_csr, b_csr, dtype
        )
        # Convert the CSR SparseMatrix to a SparseTensor.
        res = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            result_csr, dtype
        )
        return tf.SparseTensor(res.indices, res.values, res.dense_shape)

    def embedding_lookup_sparse_dense_matmul(a, b):
        # We need at least one id per rows for embedding_lookup_sparse,
        # otherwise there will be missing rows in the output.
        a, _ = tf.sparse.fill_empty_rows(a, 0)
        # We need to split x1 into separate ids and weights tensors. The ids
        # should be the column indices of x1 and the values of the weights
        # can continue to be the actual x1. The column arrangement of ids
        # and weights does not matter as we sum over columns. See details in
        # the documentation for sparse_ops.sparse_tensor_dense_matmul.
        ids = tf.SparseTensor(
            indices=a.indices,
            values=a.indices[:, 1],
            dense_shape=a.dense_shape,
        )
        return tf.nn.embedding_lookup_sparse(b, ids, a, combiner="sum")

    # Either a or b is sparse
    def sparse_dense_matmul_3d(a, b):
        return tf.map_fn(
            lambda x: tf.sparse.sparse_dense_matmul(x[0], x[1]),
            elems=(a, b),
            fn_output_signature=a.dtype,
        )

    if x1_sparse or x2_sparse:
        from keras.src.ops.operation_utils import compute_matmul_output_shape

        output_shape = compute_matmul_output_shape(x1_shape, x2_shape)
        if x1_sparse and x2_sparse:
            if x1_shape.rank <= 3:
                output = sparse_sparse_matmul(x1, x2)
            else:
                output = with_combined_batch_dimensions(
                    x1, x2, output_shape, sparse_sparse_matmul
                )
        else:
            # Sparse * dense or dense * sparse
            sparse_rank = x1_shape.rank if x1_sparse else x2_shape.rank

            # Special case: embedding_lookup_sparse for sparse * dense, rank 2
            if x1_sparse and sparse_rank == 2:
                output = embedding_lookup_sparse_dense_matmul(x1, x2)
            elif sparse_rank == 2:
                output = tf.sparse.sparse_dense_matmul(x1, x2)
            elif sparse_rank == 3:
                output = sparse_dense_matmul_3d(x1, x2)
            else:
                output = with_combined_batch_dimensions(
                    x1, x2, output_shape, sparse_dense_matmul_3d
                )
        output = tf.cast(output, result_dtype)
        output.set_shape(output_shape)
        return output
    else:
        if x1_shape.rank == 2 and x2_shape.rank == 2:
            output = tf.matmul(x1, x2, output_type=output_type)
        elif x2_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=1)
        elif x1_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=[[0], [-2]])
        elif (
            x2_shape.rank == 2
            and x1_shape.rank is not None
            and x1_shape.rank > 2
            and output_type is None
        ):
            output = tf.tensordot(x1, x2, axes=[[-1], [0]])
        else:
            output = tf.matmul(x1, x2, output_type=output_type)
        return tf.cast(output, result_dtype)


@sparse.elementwise_binary_intersection
def multiply(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    if isinstance(x, tf.IndexedSlices):
        if axis is None:
            # Reduce against all axes, result is a single value and dense.
            # The denominator has to account for `dense_shape`.
            sum = tf.reduce_sum(x.values, keepdims=keepdims)
            return sum / tf.cast(tf.reduce_prod(x.dense_shape), dtype=sum.dtype)

        axis = to_tuple_or_list(axis)
        if not axis:
            # Empty axis tuple, this is a no-op
            return x

        dense_shape = tf.convert_to_tensor(x.dense_shape)
        rank = tf.shape(dense_shape)[0]
        # Normalize axis: convert negative values and sort
        axis = [canonicalize_axis(a, rank) for a in axis]
        axis.sort()

        if axis == [0]:
            # Reduce against `axis=0` only, result is dense.
            # The denominator has to account for `dense_shape[0]`.
            sum = tf.reduce_sum(x.values, axis=0, keepdims=keepdims)
            return sum / tf.cast(dense_shape[0], dtype=sum.dtype)
        elif axis[0] == 0:
            # Reduce against axis 0 and other axes, result is dense.
            # We do `axis=0` separately first. The denominator has to account
            # for `dense_shape[0]`.
            # We use `keepdims=True` in `reduce_sum`` so that we can leave the
            # 0 in axis and do `reduce_mean` with `keepdims` to apply it for all
            # axes.
            sum = tf.reduce_sum(x.values, axis=0, keepdims=True)
            axis_0_mean = sum / tf.cast(dense_shape[0], dtype=sum.dtype)
            return tf.reduce_mean(axis_0_mean, axis=axis, keepdims=keepdims)
        elif keepdims:
            # With `keepdims=True`, result is an `IndexedSlices` with the same
            # indices since axis 0 is not touched. The only thing to do is to
            # correct `dense_shape` to account for dimensions that became 1.
            new_values = tf.reduce_mean(x.values, axis=axis, keepdims=True)
            new_dense_shape = tf.concat(
                [dense_shape[0:1], new_values.shape[1:]], axis=0
            )
            return tf.IndexedSlices(new_values, x.indices, new_dense_shape)
        elif rank == len(axis) + 1:
            # `keepdims=False` and reducing against all axes except 0, result is
            # a 1D tensor, which cannot be `IndexedSlices`. We have to scatter
            # the computed means to construct the correct dense tensor.
            return tf.scatter_nd(
                tf.expand_dims(x.indices, axis=1),
                tf.reduce_mean(x.values, axis=axis),
                [dense_shape[0]],
            )
        else:
            # `keepdims=False`, not reducing against axis 0 and there is at
            # least one other axis we are not reducing against. We simply need
            # to fix `dense_shape` to remove dimensions that were reduced.
            gather_indices = [i for i in range(rank) if i not in axis]
            return tf.IndexedSlices(
                tf.reduce_mean(x.values, axis=axis),
                x.indices,
                tf.gather(x.dense_shape, gather_indices, axis=0),
            )
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "float16":
        # The CPU implementation of reduce_mean for float16 is broken, but the
        # XLA implementation is correct.
        @tf.function(jit_compile=True)
        def xla_reduce_mean(t):
            return tf.reduce_mean(t, axis=axis, keepdims=keepdims)

        return xla_reduce_mean(x)
    if "int" in dtype or dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)

    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    if initial is not None:
        if standardize_dtype(x.dtype) == "bool":
            x = tf.reduce_any(x, axis=axis, keepdims=keepdims)
            x = tf.math.maximum(tf.cast(x, "int32"), tf.cast(initial, "int32"))
            return tf.cast(x, "bool")
        else:
            x = tf.reduce_max(x, axis=axis, keepdims=keepdims)
            return tf.math.maximum(x, initial)

    # TensorFlow returns -inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = tf.size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the max of an empty tensor.",
        )

    if standardize_dtype(x.dtype) == "bool":
        return tf.reduce_any(x, axis=axis, keepdims=keepdims)
    else:
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.zeros(shape, dtype=dtype)


@sparse.elementwise_unary
def absolute(x):
    x = convert_to_tensor(x)
    # uintx and bool are always non-negative
    dtype = standardize_dtype(x.dtype)
    if "uint" in dtype or dtype == "bool":
        return x
    return tf.abs(x)


def abs(x):
    return absolute(x)


@sparse.elementwise_unary
def fabs(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.abs(x)


def all(x, axis=None, keepdims=False):
    x = tf.cast(x, "bool")
    return tf.reduce_all(x, axis=axis, keepdims=keepdims)


def angle(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.angle(x)


def any(x, axis=None, keepdims=False):
    x = tf.cast(x, "bool")
    return tf.reduce_any(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if axis is None:
        return tf.concat([tf.reshape(x1, [-1]), tf.reshape(x2, [-1])], axis=0)
    else:
        return tf.concat([x1, x2], axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [getattr(start, "dtype", type(start))]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = standardize_dtype(dtype)
    if step is None:
        step = 1
    try:
        out = tf.range(start, stop, delta=step, dtype=dtype)
    except tf.errors.NotFoundError:
        # Some dtypes may not work in eager mode on CPU or GPU.
        out = tf.range(start, stop, delta=step, dtype="float32")
        out = tf.cast(out, dtype)
    return out


@sparse.densifying_unary(0.5 * np.pi)
def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.acos(x)


@sparse.densifying_unary(np.nan)
def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.acosh(x)


@sparse.elementwise_unary
def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.asin(x)


@sparse.elementwise_unary
def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.asinh(x)


@sparse.elementwise_unary
def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.atan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.math.atan2(x1, x2)


@sparse.elementwise_unary
def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.atanh(x)


def _keepdims(x, y, axis):
    if axis is None:
        shape = [1 for _ in range(len(x.shape))]
    else:
        shape = list(shape_op(x))
        for axis in tree.flatten(axis):
            shape[axis] = 1
    y = tf.reshape(y, shape)
    return y


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        _x = x
        if axis is None:
            x = tf.reshape(x, [-1])
        y = tf.argmax(x, axis=axis, output_type="int32")
        if keepdims:
            y = _keepdims(_x, y, axis)
        return y

    # Fix the flush-to-zero (FTZ) issue based on this issue:
    # https://github.com/jax-ml/jax/issues/24280
    dtype = dtypes.result_type(dtype, "float32")
    x = cast(x, dtype)
    is_negative_zero = tf.logical_and(tf.equal(x, 0.0), signbit(x))
    x = tf.where(
        is_negative_zero, -np.finfo(standardize_dtype(x.dtype)).tiny, x
    )
    _x = x
    if axis is None:
        x = tf.reshape(x, [-1])
    y = tf.argmax(x, axis=axis, output_type="int32")
    if keepdims:
        y = _keepdims(_x, y, axis)
    return y


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        _x = x
        if axis is None:
            x = tf.reshape(x, [-1])
        y = tf.argmin(x, axis=axis, output_type="int32")
        if keepdims:
            y = _keepdims(_x, y, axis)
        return y

    # Fix the flush-to-zero (FTZ) issue based on this issue:
    # https://github.com/jax-ml/jax/issues/24280
    dtype = dtypes.result_type(dtype, "float32")
    x = cast(x, dtype)
    is_negative_zero = tf.logical_and(tf.equal(x, 0.0), signbit(x))
    x = tf.where(
        is_negative_zero, -np.finfo(standardize_dtype(x.dtype)).tiny, x
    )
    _x = x
    if axis is None:
        x = tf.reshape(x, [-1])
    y = tf.argmin(x, axis=axis, output_type="int32")
    if keepdims:
        y = _keepdims(_x, y, axis)
    return y


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "uint8")

    x_shape = x.shape
    if x_shape.rank == 0:
        return tf.cast([0], "int32")

    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.argsort(x, axis=axis)


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def view(x, dtype=None):
    x = convert_to_tensor(x)
    old_dtype = tf.as_dtype(standardize_dtype(x.dtype))
    new_dtype = tf.as_dtype(standardize_dtype(dtype if dtype else x.dtype))

    old_itemsize = old_dtype.size
    new_itemsize = new_dtype.size

    old_shape = list(shape_op(x))
    last_dim_size = old_shape[-1] if len(old_shape) > 0 else -1
    if (last_dim_size == -1 and old_itemsize != new_itemsize) or (
        last_dim_size * old_itemsize % new_itemsize != 0
    ):
        raise ValueError(
            f"Cannot view array of shape {x.shape} and dtype {old_dtype} "
            f"as dtype {new_dtype} because the total number of bytes "
            f"is not divisible by the new itemsize."
        )

    if old_itemsize == new_itemsize:
        return tf.bitcast(x, type=new_dtype)
    elif old_itemsize > new_itemsize:
        ratio = old_itemsize // new_itemsize
        new_shape = list(shape_op(x))
        new_shape[-1] *= ratio
        flat_tensor = tf.reshape(x, [-1])
        cast_tensor = tf.bitcast(flat_tensor, type=new_dtype)
        return tf.reshape(cast_tensor, new_shape)
    else:
        ratio = new_itemsize // old_itemsize
        if isinstance(last_dim_size, int) and last_dim_size % ratio != 0:
            raise ValueError(
                f"Cannot view dtype. Last dimension size ({last_dim_size}) "
                f"must be divisible by the ratio of new/old item sizes "
                f"({ratio})."
            )
        intermediate_shape = old_shape[:-1] + [last_dim_size // ratio, ratio]
        reshaped_tensor = tf.reshape(x, intermediate_shape)
        return tf.bitcast(reshaped_tensor, new_dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)

    if weights is None:  # Treat all weights as 1
        dtype = dtypes.result_type(x.dtype, float)
        x = tf.cast(x, dtype)
        avg = tf.reduce_mean(x, axis=axis)
    else:
        weights = convert_to_tensor(weights)
        if len(weights.shape) == 1 and len(x.shape) > 1:
            if axis is None or (
                isinstance(axis, (list, tuple)) and len(axis) != 1
            ):
                raise ValueError(
                    "Axis must be specified when shapes of a and weights "
                    "differ."
                )
            axis_val = axis[0] if isinstance(axis, (list, tuple)) else axis
            axis_val = canonicalize_axis(axis_val, len(x.shape))
            if weights.shape[0] != x.shape[axis_val]:
                raise ValueError(
                    "Shape of weights must be consistent with shape of a "
                    "along specified axis."
                )
        elif x.shape != weights.shape:
            if axis is None:
                raise ValueError(
                    "Axis must be specified when shapes of a and weights "
                    "differ."
                )
            raise ValueError(
                "Shape of weights must be consistent with shape of a "
                "along specified axis."
            )
        dtype = dtypes.result_type(x.dtype, weights.dtype, float)
        x = tf.cast(x, dtype)
        weights = tf.cast(weights, dtype)

        def _rank_equal_case():
            weights_sum = tf.reduce_sum(weights, axis=axis)
            return tf.reduce_sum(x * weights, axis=axis) / weights_sum

        def _rank_not_equal_case():
            weights_sum = tf.reduce_sum(weights)
            a = axis[0] if isinstance(axis, (tuple, list)) else axis
            axes = tf.convert_to_tensor([[a], [0]])
            return tf.tensordot(x, weights, axes) / weights_sum

        if axis is None:
            avg = _rank_equal_case()
        else:
            if len(x.shape) == len(weights.shape):
                avg = _rank_equal_case()
            else:
                avg = _rank_not_equal_case()
    return avg


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return tf.bitwise.invert(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = tf.cast(x, dtype)
        y = tf.cast(y, dtype)
    return tf.bitwise.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = tf.cast(x, dtype)
        y = tf.cast(y, dtype)
    return tf.bitwise.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def blackman(x):
    dtype = config.floatx()
    compute_dtype = dtypes.result_type(dtype, "float32")

    x = tf.cast(x, "int32")
    n = tf.range(x, dtype=compute_dtype)
    x_minus_1 = tf.cast(x - 1, compute_dtype)
    phase = np.pi * (2 * n - x_minus_1) / tf.maximum(x_minus_1, 1.0)
    window = 0.42 + 0.5 * tf.cos(phase) + 0.08 * tf.cos(2 * phase)
    return tf.cast(window, dtype)


def broadcast_to(x, shape):
    return tf.broadcast_to(x, shape)


def cbrt(x):
    x = convert_to_tensor(x)

    dtype = standardize_dtype(x.dtype)
    if dtype == "int64":
        x = tf.cast(x, "float64")
    elif dtype not in ["bfloat16", "float16", "float64"]:
        x = tf.cast(x, config.floatx())

    return tf.sign(x) * tf.pow(tf.abs(x), 1.0 / 3.0)


@sparse.elementwise_unary
def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.ceil(x)


def clip(x, x_min, x_max):
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        x = tf.cast(x, "int32")
    return tf.clip_by_value(x, x_min, x_max)


def concatenate(xs, axis=0):
    sparse_count = builtins.sum(isinstance(x, tf.SparseTensor) for x in xs)
    if sparse_count:
        if sparse_count == len(xs):
            return tf.sparse.concat(axis=axis, sp_inputs=xs)
        else:
            xs = [
                (
                    convert_to_tensor(x, sparse=False)
                    if isinstance(x, tf.SparseTensor)
                    else x
                )
                for x in xs
            ]
    xs = tree.map_structure(convert_to_tensor, xs)
    dtype_set = set([x.dtype for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(lambda x: tf.cast(x, dtype), xs)
    return tf.concat(xs, axis=axis)


@sparse.elementwise_unary
def conjugate(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def conj(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def copy(x):
    x = convert_to_tensor(x)
    return tf.identity(x)


def copysign(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    magnitude = tf.abs(x1)
    return tf.where(signbit(x2), -magnitude, magnitude)


@sparse.densifying_unary(1)
def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.cos(x)


@sparse.densifying_unary(1)
def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.cosh(x)


def count_nonzero(x, axis=None):
    return tf.math.count_nonzero(x, axis=axis, dtype="int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    if axis is not None:
        axisa = axis
        axisb = axis
        axisc = axis
    x1 = moveaxis(x1, axisa, -1)
    x2 = moveaxis(x2, axisb, -1)

    def maybe_pad_zeros(x, size_of_last_dim):
        def pad_zeros(x):
            return tf.pad(
                x,
                tf.concat(
                    [
                        tf.zeros([tf.rank(x) - 1, 2], "int32"),
                        tf.constant([[0, 1]], "int32"),
                    ],
                    axis=0,
                ),
            )

        if isinstance(size_of_last_dim, int):
            if size_of_last_dim == 2:
                return pad_zeros(x)
            return x

        return tf.cond(
            tf.equal(size_of_last_dim, 2), lambda: pad_zeros(x), lambda: x
        )

    x1_dim = shape_op(x1)[-1]
    x2_dim = shape_op(x2)[-1]

    x1 = maybe_pad_zeros(x1, x1_dim)
    x2 = maybe_pad_zeros(x2, x2_dim)

    # Broadcast each other
    shape = shape_op(x1)

    shape = tf.broadcast_dynamic_shape(shape, shape_op(x2))
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)

    c = tf.linalg.cross(x1, x2)

    if isinstance(x1_dim, int) and isinstance(x2_dim, int):
        if (x1_dim == 2) & (x2_dim == 2):
            return c[..., 2]
        return moveaxis(c, -1, axisc)

    return tf.cond(
        (x1_dim == 2) & (x2_dim == 2),
        lambda: c[..., 2],
        lambda: moveaxis(c, -1, axisc),
    )


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    # tf.math.cumprod doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.math.cumprod(x, axis=axis)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    # tf.math.cumprod doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.math.cumsum(x, axis=axis)


def deg2rad(x):
    x = convert_to_tensor(x)

    dtype = x.dtype
    if standardize_dtype(dtype) in [
        "bool",
        "int8",
        "int16",
        "int32",
        "uint8",
        "uint16",
        "uint32",
    ]:
        dtype = config.floatx()
    elif standardize_dtype(dtype) in ["int64"]:
        dtype = "float64"
    x = tf.cast(x, dtype)

    pi = tf.constant(math.pi, dtype=dtype)
    return x * (pi / tf.constant(180.0, dtype=dtype))


def rad2deg(x):
    x = convert_to_tensor(x)

    dtype = x.dtype
    if standardize_dtype(dtype) in [
        "bool",
        "int8",
        "int16",
        "int32",
        "uint8",
        "uint16",
        "uint32",
    ]:
        dtype = config.floatx()
    elif standardize_dtype(dtype) in ["int64"]:
        dtype = "float64"
    x = tf.cast(x, dtype)

    pi = tf.constant(math.pi, dtype=dtype)
    return x * (tf.constant(180.0, dtype=dtype) / pi)


def diag(x, k=0):
    x = convert_to_tensor(x)
    if len(x.shape) == 1:
        return tf.linalg.diag(x, k=k)
    elif len(x.shape) == 2:
        return diagonal(x, offset=k)
    else:
        raise ValueError(f"`x` must be 1d or 2d. Received: x.shape={x.shape}")


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    return diag(tf.reshape(x, [-1]), k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    x_rank = x.ndim
    if (
        offset == 0
        and (axis1 == x_rank - 2 or axis1 == -2)
        and (axis2 == x_rank - 1 or axis2 == -1)
    ):
        return tf.linalg.diag_part(x)

    x = moveaxis(x, (axis1, axis2), (-2, -1))
    x_shape = shape_op(x)

    def _zeros():
        return tf.zeros(tf.concat([x_shape[:-1], [0]], 0), dtype=x.dtype)

    if isinstance(x_shape[-1], int) and isinstance(x_shape[-2], int):
        if offset <= -1 * x_shape[-2] or offset >= x_shape[-1]:
            x = _zeros()
    else:
        x = tf.cond(
            tf.logical_or(
                tf.less_equal(offset, -1 * x_shape[-2]),
                tf.greater_equal(offset, x_shape[-1]),
            ),
            lambda: _zeros(),
            lambda: x,
        )
    return tf.linalg.diag_part(x, k=offset)


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    if n == 0:
        return a
    elif n < 0:
        raise ValueError(f"Order `n` must be non-negative. Received n={n}")
    elif a.ndim == 0:
        raise ValueError(
            "`diff` requires input that is at least one dimensional. "
            f"Received: a={a}"
        )
    axis = canonicalize_axis(axis, a.ndim)
    slice1 = [slice(None)] * a.ndim
    slice2 = [slice(None)] * a.ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1_tuple = tuple(slice1)
    slice2_tuple = tuple(slice2)
    for _ in range(n):
        if standardize_dtype(a.dtype) == "bool":
            a = tf.not_equal(a[slice1_tuple], a[slice2_tuple])
        else:
            a = tf.subtract(a[slice1_tuple], a[slice2_tuple])
    return a


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = list(bins)

    # bins must be float type
    bins = tree.map_structure(lambda x: float(x), bins)

    # TODO: tf.raw_ops.Bucketize doesn't support bool, bfloat16, float16, int8
    # int16, uint8, uint16, uint32
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype in ("bool", "int8", "int16", "uint8", "uint16"):
        x = cast(x, "int32")
    elif ori_dtype == "uint32":
        x = cast(x, "int64")
    elif ori_dtype in ("bfloat16", "float16"):
        x = cast(x, "float32")

    if isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(
            lambda y: tf.raw_ops.Bucketize(input=y, boundaries=bins), x
        )
    elif isinstance(x, tf.SparseTensor):
        output = tf.SparseTensor(
            indices=tf.identity(x.indices),
            values=tf.raw_ops.Bucketize(input=x.values, boundaries=bins),
            dense_shape=tf.identity(x.dense_shape),
        )
        output.set_shape(x.shape)
        return output
    return tf.raw_ops.Bucketize(input=x, boundaries=bins)


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # GPU only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)

    x_shape = x1.shape
    y_shape = x2.shape
    if x_shape.rank == 0 or y_shape.rank == 0:
        output = x1 * x2
    elif y_shape.rank == 1:
        output = tf.tensordot(x1, x2, axes=[[-1], [-1]])
    else:
        output = tf.tensordot(x1, x2, axes=[[-1], [-2]])
    return tf.cast(output, result_dtype)


def dstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    if len(xs) > 1:
        unique_dtypes = {x.dtype for x in xs}
        if len(unique_dtypes) > 1:
            dtype = dtypes.result_type(*[x.dtype for x in xs])
            xs = [cast(x, dtype) for x in xs]
    xs_reshaped = []
    for x in xs:
        shape = x.shape
        if len(shape) == 0:
            x = tf.reshape(x, (1, 1, 1))
        elif len(shape) == 1:
            x = tf.expand_dims(x, axis=0)
            x = tf.expand_dims(x, axis=2)
        elif len(shape) == 2:
            x = tf.expand_dims(x, axis=2)
        xs_reshaped.append(x)
    return tf.concat(xs_reshaped, axis=2)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return tf.zeros_like(x, dtype=dtype)


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.equal(x1, x2)


@sparse.densifying_unary(1)
def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.exp(x)


@sparse.densifying_unary(1)
def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.pow(2.0, x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    out_ndim = len(x.shape) + len(axis)
    axis = sorted([canonicalize_axis(a, out_ndim) for a in axis])
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import (
            compute_expand_dims_output_shape,
        )

        output_shape = compute_expand_dims_output_shape(x.shape, axis)
        for a in axis:
            x = tf.sparse.expand_dims(x, a)
        x.set_shape(output_shape)
        return x
    for a in axis:
        x = tf.expand_dims(x, a)
    return x


@sparse.elementwise_unary
def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.expm1(x)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    rank = x.shape.rank
    if axis is None:
        return tf.reverse(x, tf.range(rank))
    else:
        axis = to_tuple_or_list(axis)
        axis = [canonicalize_axis(a, rank) for a in axis]
        return tf.reverse(x, axis)


def fliplr(x):
    x = convert_to_tensor(x)
    return tf.reverse(x, [1])


def flipud(x):
    x = convert_to_tensor(x)
    return tf.reverse(x, [0])


@sparse.elementwise_unary
def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    fill_value = convert_to_tensor(fill_value, dtype)
    return tf.broadcast_to(fill_value, shape)


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    fill_value = convert_to_tensor(fill_value, dtype)
    return tf.broadcast_to(fill_value, tf.shape(x))


def gcd(x1, x2):
    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    if not x1.dtype.is_integer:
        raise TypeError("Arguments to gcd must be integers.")

    target_shape = tf.broadcast_static_shape(x1.shape, x2.shape)
    x1 = tf.broadcast_to(x1, target_shape)
    x2 = tf.broadcast_to(x2, target_shape)

    def cond(a, b):
        return tf.reduce_any(b != 0)

    def body(a, b):
        b_safe = tf.where(tf.equal(b, 0), tf.ones_like(b), b)
        return (
            tf.where(tf.not_equal(b, 0), b, a),
            tf.where(
                tf.not_equal(b, 0),
                tf.math.floormod(a, b_safe),
                tf.zeros_like(b),
            ),
        )

    if dtype not in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        x1 = tf.abs(x1)
        x2 = tf.abs(x2)

    gcd_val, _ = tf.while_loop(cond, body, [x1, x2])
    return gcd_val


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    start = convert_to_tensor(start)
    stop = convert_to_tensor(stop)
    log_start = tf.math.log(tf.cast(tf.abs(start), dtype or config.floatx()))
    log_stop = tf.math.log(tf.cast(tf.abs(stop), dtype or config.floatx()))
    log_base = tf.math.log(tf.constant(10.0, dtype=log_start.dtype))
    result = logspace(
        log_start / log_base,
        log_stop / log_base,
        num=num,
        endpoint=endpoint,
        base=10,
        dtype=dtype,
        axis=axis,
    )
    # Handle sign: start and stop must have the same sign (or be zero)
    start_sign = tf.cast(tf.sign(tf.cast(start, log_start.dtype)), result.dtype)
    return result * start_sign


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.greater_equal(x1, x2)


def hstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    if len(xs[0].shape) == 1:
        return tf.concat(xs, axis=0)
    return tf.concat(xs, axis=1)


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim == 1:
        return split(x, indices_or_sections, axis=0)
    return split(x, indices_or_sections, axis=1)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"

    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    x1_abs = tf.abs(x1)
    x2_abs = tf.abs(x2)
    max_val = tf.maximum(x1_abs, x2_abs)
    both_zero = tf.equal(max_val, 0)
    scale = tf.stop_gradient(
        tf.where(both_zero, tf.ones_like(max_val), max_val)
    )
    sum_sq = tf.square(x1 / scale) + tf.square(x2 / scale)
    safe_sum_sq = tf.where(both_zero, tf.ones_like(sum_sq), sum_sq)
    result = tf.where(
        both_zero,
        tf.zeros_like(max_val),
        scale * tf.sqrt(safe_sum_sq),
    )
    return tf.where(
        tf.math.is_inf(x1_abs) | tf.math.is_inf(x2_abs),
        tf.constant(float("inf"), dtype=result.dtype),
        result,
    )


def identity(n, dtype=None):
    return eye(N=n, M=n, dtype=dtype)


@sparse.elementwise_unary
def imag(x):
    return tf.math.imag(x)


def i0(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ["int64", "float64"]:
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16"]:
        dtype = config.floatx()
    x = tf.cast(x, dtype)
    return tf.math.bessel_i0(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if "float" in dtype:
        finite = tf.math.is_finite(x1) & tf.math.is_finite(x2)
        close = tf.abs(x1 - x2) <= (atol + rtol * tf.abs(x2))
        result = (finite & close) | tf.equal(x1, x2)
        if equal_nan:
            result = result | (is_nan(x1) & is_nan(x2))
        return result
    else:
        return tf.equal(x1, x2)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    if "float" in standardize_dtype(dtype):
        finite = tf.math.is_finite(x1) & tf.math.is_finite(x2)
        close = tf.abs(x1 - x2) <= (atol + rtol * tf.abs(x2))
        result = (finite & close) | tf.equal(x1, x2)
        if equal_nan:
            result = result | (is_nan(x1) & is_nan(x2))
        return tf.reduce_all(result)
    else:
        return tf.reduce_all(tf.equal(x1, x2))


@sparse.densifying_unary(True)
def isfinite(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.ones(x.shape, tf.bool)
    return tf.math.is_finite(x)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    output_shape = tf.shape(x1)

    x1 = tf.reshape(x1, [-1])
    x2 = tf.reshape(x2, [-1])

    if not assume_unique:
        x2 = tf.unique(x2)[0]

    if tf.size(x1) == 0 or tf.size(x2) == 0:
        return tf.zeros(output_shape, dtype=tf.bool)

    cmp = tf.equal(tf.expand_dims(x1, 1), tf.expand_dims(x2, 0))
    result_flat = tf.reduce_any(cmp, axis=1)

    if invert:
        result_flat = tf.logical_not(result_flat)

    return tf.reshape(result_flat, output_shape)


def isinf(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_inf(x)


def isnan(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_nan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros_like(x, dtype=tf.bool)
    return tf.math.equal(x, -tf.constant(float("inf"), dtype=x.dtype))


def isposinf(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros_like(x, dtype=tf.bool)
    return tf.math.equal(x, tf.constant(float("inf"), dtype=x.dtype))


def isreal(x):
    x = convert_to_tensor(x)
    if x.dtype.is_complex:
        return tf.equal(tf.math.imag(x), 0)
    else:
        return tf.ones_like(x, dtype=tf.bool)


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    ndim_x1 = tf.rank(x1)
    ndim_x2 = tf.rank(x2)

    def expand_front(x, num):
        for _ in range(num):
            x = tf.expand_dims(x, axis=0)
        return x

    x1 = tf.cond(
        ndim_x1 < ndim_x2,
        lambda: expand_front(x1, ndim_x2 - ndim_x1),
        lambda: x1,
    )
    x2 = tf.cond(
        ndim_x2 < ndim_x1,
        lambda: expand_front(x2, ndim_x1 - ndim_x2),
        lambda: x2,
    )

    x1_reshaped = tf.reshape(
        x1,
        tf.reshape(
            tf.stack([tf.shape(x1), tf.ones_like(tf.shape(x1))], axis=1), [-1]
        ),
    )
    x2_reshaped = tf.reshape(
        x2,
        tf.reshape(
            tf.stack([tf.ones_like(tf.shape(x2)), tf.shape(x2)], axis=1), [-1]
        ),
    )

    out = tf.multiply(x1_reshaped, x2_reshaped)
    out_shape = tf.multiply(tf.shape(x1), tf.shape(x2))
    out = tf.reshape(out, out_shape)
    return out


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    if not (x1.dtype.is_integer and x2.dtype.is_integer):
        raise TypeError(
            f"Arguments to lcm must be integers. "
            f"Received: x1.dtype={x1.dtype.name}, x2.dtype={x2.dtype.name}"
        )

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    if dtype not in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        x1 = tf.math.abs(x1)
        x2 = tf.math.abs(x2)

    divisor = gcd(x1, x2)
    divisor_safe = tf.where(
        divisor == 0, tf.constant(1, dtype=divisor.dtype), divisor
    )

    result = x1 * (x2 // divisor_safe)
    result = tf.where(divisor == 0, tf.zeros_like(result), result)

    return result


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)

    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )

    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, x1.dtype)
    pow_val = tf.pow(tf.constant(2.0, dtype=x1.dtype), x2)
    result = tf.where(tf.math.is_inf(pow_val) & (x1 == 0), x1, x1 * pow_val)
    return tf.cast(tf.where(tf.math.is_inf(x1), x1, result), dtype)


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if num < 0:
        raise ValueError(
            f"`num` must be a non-negative integer. Received: num={num}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = standardize_dtype(dtype)
    start = convert_to_tensor(start, dtype=dtype)
    stop = convert_to_tensor(stop, dtype=dtype)
    step = convert_to_tensor(np.nan)
    if endpoint:
        result = tf.linspace(start, stop, num, axis=axis)
        if num > 1:
            step = (stop - start) / (tf.cast(num, dtype) - 1)
    else:
        # tf.linspace doesn't support endpoint=False, so we manually handle it
        if num > 0:
            step = (stop - start) / tf.cast(num, dtype)
        if num > 1:
            new_stop = tf.cast(stop, step.dtype) - step
            start = tf.cast(start, new_stop.dtype)
            result = tf.linspace(start, new_stop, num, axis=axis)
        else:
            result = tf.linspace(start, stop, num, axis=axis)
    if dtype is not None:
        if "int" in dtype:
            result = tf.floor(result)
        result = tf.cast(result, dtype)
    if retstep:
        return (result, step)
    else:
        return result


@sparse.densifying_unary(-np.inf)
def log(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x)


@sparse.densifying_unary(-np.inf)
def log10(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x) / tf.math.log(tf.constant(10, x.dtype))


@sparse.elementwise_unary
def log1p(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log1p(x)


@sparse.densifying_unary(-np.inf)
def log2(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x) / tf.math.log(tf.constant(2, x.dtype))


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    amax = tf.maximum(x1, x2)
    amin = tf.minimum(x1, x2)
    finite_max = tf.math.is_finite(amax)
    # Dummy center is 0 when amax is non-finite so exp/log stay finite.
    safe_max = tf.stop_gradient(tf.where(finite_max, amax, tf.zeros_like(amax)))
    # Unused branch: both operands = safe_max -> log(2) finite (WhereGrad).
    use_lse = finite_max & tf.math.is_finite(amin)
    x1_masked = tf.where(use_lse, x1, safe_max)
    x2_masked = tf.where(use_lse, x2, safe_max)
    lse = safe_max + tf.math.log(
        tf.math.exp(x1_masked - safe_max) + tf.math.exp(x2_masked - safe_max)
    )
    # NaN in, NaN out; non-finite inputs resolve via amax (including inf pairs).
    return tf.where(
        tf.math.is_nan(x1) | tf.math.is_nan(x2),
        x1 + x2,
        tf.where(use_lse, lse, amax),
    )


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    amax = tf.maximum(x1, x2)
    amin = tf.minimum(x1, x2)
    finite_max = tf.math.is_finite(amax)
    safe_max = tf.stop_gradient(tf.where(finite_max, amax, tf.zeros_like(amax)))
    use_lse = finite_max & tf.math.is_finite(amin)
    log2 = tf.cast(tf.math.log(2.0), dtype)
    x1_masked = tf.where(use_lse, x1, safe_max)
    x2_masked = tf.where(use_lse, x2, safe_max)
    lse = (
        safe_max
        + tf.math.log(
            tf.math.exp((x1_masked - safe_max) * log2)
            + tf.math.exp((x2_masked - safe_max) * log2)
        )
        / log2
    )
    return tf.where(
        tf.math.is_nan(x1) | tf.math.is_nan(x2),
        x1 + x2,
        tf.where(use_lse, lse, amax),
    )


def logical_and(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.logical_and(x1, x2)


def logical_not(x):
    x = tf.cast(x, "bool")
    return tf.logical_not(x)


def logical_or(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    result = linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
        axis=axis,
    )
    return tf.pow(tf.cast(base, result.dtype), result)


@sparse.elementwise_binary_union(tf.sparse.maximum, densify_mixed=True)
def maximum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.maximum(x1, x2)


def fmax(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)

    if "float" not in standardize_dtype(dtype):
        return tf.maximum(x1, x2)

    nan_x1 = tf.math.is_nan(x1)
    nan_x2 = tf.math.is_nan(x2)

    res = tf.maximum(x1, x2)

    res = tf.where(tf.logical_and(nan_x1, tf.logical_not(nan_x2)), x2, res)

    res = tf.where(tf.logical_and(nan_x2, tf.logical_not(nan_x1)), x1, res)
    return res


def median(x, axis=None, keepdims=False):
    return quantile(x, 0.5, axis=axis, keepdims=keepdims)


def meshgrid(*x, indexing="xy"):
    return tf.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)

    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    if initial is not None:
        if standardize_dtype(x.dtype) == "bool":
            x = tf.reduce_all(x, axis=axis, keepdims=keepdims)
            x = tf.math.minimum(tf.cast(x, "int32"), tf.cast(initial, "int32"))
            return tf.cast(x, "bool")
        else:
            x = tf.reduce_min(x, axis=axis, keepdims=keepdims)
        return tf.math.minimum(x, initial)

    # TensorFlow returns inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = tf.size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the min of an empty tensor.",
        )

    if standardize_dtype(x.dtype) == "bool":
        return tf.reduce_all(x, axis=axis, keepdims=keepdims)
    else:
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)


@sparse.elementwise_binary_union(tf.sparse.minimum, densify_mixed=True)
def minimum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.minimum(x1, x2)


def fmin(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)

    if "float" not in standardize_dtype(dtype):
        return tf.minimum(x1, x2)

    nan_x1 = tf.math.is_nan(x1)
    nan_x2 = tf.math.is_nan(x2)

    res = tf.minimum(x1, x2)

    res = tf.where(tf.logical_and(nan_x1, tf.logical_not(nan_x2)), x2, res)

    res = tf.where(tf.logical_and(nan_x2, tf.logical_not(nan_x1)), x1, res)
    return res


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.math.mod(x1, x2)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    # tf.math.floormod does not support uint8/uint16; compute in int32
    compute_dtype = dtype
    if dtype in ("uint8", "uint16", "uint32"):
        compute_dtype = "int32"
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    result = tf.sign(x1) * tf.math.floormod(tf.abs(x1), tf.abs(x2))
    return tf.cast(result, dtype)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)

    _source = to_tuple_or_list(source)
    _destination = to_tuple_or_list(destination)
    _source = tuple(canonicalize_axis(i, x.ndim) for i in _source)
    _destination = tuple(canonicalize_axis(i, x.ndim) for i in _destination)
    if len(_source) != len(_destination):
        raise ValueError(
            "Inconsistent number of `source` and `destination`. "
            f"Received: source={source}, destination={destination}"
        )
    # Directly return x if no movement is required
    if _source == _destination:
        return x
    perm = [i for i in range(x.ndim) if i not in _source]
    for dest, src in sorted(zip(_destination, _source)):
        perm.insert(dest, src)
    return tf.transpose(x, perm)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not x.dtype.is_floating:
        return argmax(x, axis=axis, keepdims=keepdims)

    nan_mask = tf.math.is_nan(x)

    return tf.where(
        tf.reduce_all(nan_mask, axis=axis, keepdims=keepdims),
        tf.constant(-1, dtype=tf.int32),
        argmax(
            tf.where(nan_mask, tf.constant(float("-inf"), dtype=x.dtype), x),
            axis=axis,
            keepdims=keepdims,
        ),
    )


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not x.dtype.is_floating:
        return argmin(x, axis=axis, keepdims=keepdims)

    nan_mask = tf.math.is_nan(x)

    return tf.where(
        tf.reduce_all(nan_mask, axis=axis, keepdims=keepdims),
        tf.constant(-1, dtype=tf.int32),
        argmin(
            tf.where(nan_mask, tf.constant(float("inf"), dtype=x.dtype), x),
            axis=axis,
            keepdims=keepdims,
        ),
    )


def nancumsum(x, axis=None, dtype=None):
    x = nan_to_num(x, posinf=float("inf"), neginf=float("-inf"))
    return cumsum(x, axis=axis, dtype=dtype)


def nancumprod(x, axis=None, dtype=None):
    x = nan_to_num(x, nan=1.0, posinf=float("inf"), neginf=float("-inf"))
    return cumprod(x, axis=axis, dtype=dtype)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not x.dtype.is_floating:
        dtype = standardize_dtype(x.dtype)
        if dtype == "bool":
            return tf.reduce_any(x, axis=axis, keepdims=keepdims)
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    x_clean = tf.where(
        tf.math.is_nan(x), tf.constant(float("-inf"), dtype=x.dtype), x
    )

    return tf.where(
        tf.reduce_all(tf.math.is_nan(x), axis=axis, keepdims=keepdims),
        tf.constant(float("nan"), dtype=x.dtype),
        tf.reduce_max(x_clean, axis=axis, keepdims=keepdims),
    )


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if axis == () or axis == []:
        return x

    if not x.dtype.is_floating:
        return tf.reduce_mean(
            tf.cast(x, "float32"), axis=axis, keepdims=keepdims
        )

    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    total_sum = cast(nansum(x, axis=axis, keepdims=keepdims), dtype)
    normalizer = tf.reduce_sum(
        cast(~tf.math.is_nan(x), dtype),
        axis=axis,
        keepdims=keepdims,
    )
    return tf.divide(total_sum, normalizer)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if axis == () or axis == []:
        return x

    return nanquantile(x, q=0.5, axis=axis, keepdims=keepdims)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not x.dtype.is_floating:
        dtype = standardize_dtype(x.dtype)
        if dtype == "bool":
            return tf.reduce_all(x, axis=axis, keepdims=keepdims)
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    x_clean = tf.where(
        tf.math.is_nan(x), tf.constant(float("inf"), dtype=x.dtype), x
    )

    return tf.where(
        tf.reduce_all(tf.math.is_nan(x), axis=axis, keepdims=keepdims),
        tf.constant(float("nan"), dtype=x.dtype),
        tf.reduce_min(x_clean, axis=axis, keepdims=keepdims),
    )


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, dtype=config.floatx()) / 100.0
    return nanquantile(x, q, axis=axis, method=method, keepdims=keepdims)


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not x.dtype.is_floating:
        return prod(x, axis=axis, keepdims=keepdims)

    x_safe = tf.where(tf.math.is_nan(x), tf.ones((), dtype=x.dtype), x)
    return prod(x_safe, axis=axis, keepdims=keepdims)


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, dtype=config.floatx())

    def _nanquantile_1d(v):
        valid = tf.boolean_mask(v, ~tf.math.is_nan(cast(v, config.floatx())))
        return tf.cond(
            tf.size(valid) > 0,
            lambda: quantile(valid, q, method=method, keepdims=False),
            lambda: tf.constant(float("nan"), dtype=x.dtype),
        )

    if axis is None:
        x_flat = tf.reshape(x, [-1])
        result = _nanquantile_1d(x_flat)

        if keepdims:
            new_shape = tf.concat(
                [tf.shape(result), tf.ones(tf.rank(x), dtype=tf.int32)], axis=0
            )
            result = tf.reshape(result, new_shape)

        return result

    if isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)

    ndims = x.shape.rank
    axis = [a if a >= 0 else a + ndims for a in axis]
    other_axes = [i for i in range(ndims) if i not in axis]

    perm = other_axes + axis
    x_t = tf.transpose(x, perm)

    shape = tf.shape(x_t)
    other_rank = len(other_axes)
    other_shape = shape[:other_rank]
    reduce_shape = shape[other_rank:]

    batch_size = tf.reduce_prod(other_shape)
    reduction_size = tf.reduce_prod(reduce_shape)
    x_flat = tf.reshape(x_t, [batch_size, reduction_size])

    q_shape = tf.shape(q)

    results = tf.map_fn(
        _nanquantile_1d,
        x_flat,
        fn_output_signature=tf.TensorSpec(shape=q_shape, dtype=x.dtype),
    )

    if tf.rank(q) > 0:
        results = tf.transpose(results)
        results = tf.reshape(results, tf.concat([q_shape, other_shape], axis=0))
    else:
        results = tf.reshape(results, other_shape)

    if keepdims:
        for ax in sorted(axis):
            results = tf.expand_dims(results, axis=ax + tf.rank(q))

    return results


def nanstd(x, axis=None, keepdims=False):
    var_val = nanvar(x, axis=axis, keepdims=keepdims)
    return tf.sqrt(var_val)


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    x_clean = tf.where(
        tf.math.is_nan(cast(x, config.floatx())), tf.zeros((), dtype=dtype), x
    )

    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x_clean = cast(x_clean, dtype)

    return tf.reduce_sum(x_clean, axis=axis, keepdims=keepdims)


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, result_dtype)

    mean = nanmean(x, axis=axis, keepdims=True)

    valid = ~tf.math.is_nan(x)

    centered = tf.where(valid, x - mean, tf.zeros_like(x))

    if centered.dtype.is_complex:
        centered = tf.math.real(centered * tf.math.conj(centered))
    else:
        centered = tf.square(centered)

    count = tf.reduce_sum(
        tf.cast(valid, centered.dtype), axis=axis, keepdims=keepdims
    )

    var = tf.reduce_sum(centered, axis=axis, keepdims=keepdims) / count

    return var


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)

    dtype = x.dtype
    dtype_as_dtype = tf.as_dtype(dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return x

    # Replace NaN with `nan`
    x = tf.where(tf.math.is_nan(x), tf.constant(nan, dtype), x)

    # Replace positive infinity with `posinf` or `dtype.max`
    if posinf is None:
        posinf = dtype.max
    x = tf.where(tf.math.is_inf(x) & (x > 0), tf.constant(posinf, dtype), x)

    # Replace negative infinity with `neginf` or `dtype.min`
    if neginf is None:
        neginf = dtype.min
    x = tf.where(tf.math.is_inf(x) & (x < 0), tf.constant(neginf, dtype), x)

    return x


def ndim(x):
    x = convert_to_tensor(x)
    return x.shape.rank


def nonzero(x):
    x = convert_to_tensor(x)
    result = tf.unstack(tf.where(tf.cast(x, "bool")), x.shape.rank, axis=1)
    return tree.map_structure(lambda indices: tf.cast(indices, "int32"), result)


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return tf.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return tf.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.reshape(x1, [-1, 1]) * tf.reshape(x2, [-1])


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    kwargs = {}
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        kwargs["constant_values"] = constant_values
    if len(pad_width) == 1:
        # A single `(before, after)` pair broadcasts to every axis, matching
        # `np.pad`. `tf.pad` requires an explicit `[rank(x), 2]` spec, so
        # expand it here.
        pad_width = [pad_width[0]] * len(x.shape)
    pad_width = convert_to_tensor(pad_width, "int32")
    return tf.pad(x, pad_width, mode.upper(), **kwargs)


def percentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, dtype=config.floatx()) / 100.0
    return quantile(x, q, axis=axis, method=method, keepdims=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = dtypes.result_type(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        elif dtype in ("int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
        x = tf.cast(x, dtype)
    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return tf.reduce_max(x, axis=axis, keepdims=keepdims) - tf.reduce_min(
        x, axis=axis, keepdims=keepdims
    )


def _quantile(x, q, axis=None, method="linear", keepdims=False):
    # ref: tfp.stats.percentile
    # float64 is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = tf.cast(q, "float64")

    # Move `axis` dims of `x` to the rightmost, call it `y`.
    if axis is None:
        y = tf.reshape(x, [-1])
    else:
        x_ndims = len(x.shape)
        # _make_static_axis_non_negative_list
        axis = [canonicalize_axis(a, x_ndims) for a in axis]

        # _move_dims_to_flat_end
        other_dims = sorted(set(range(x_ndims)).difference(axis))
        perm = other_dims + list(axis)
        x_permed = tf.transpose(a=x, perm=perm)
        if None not in x.shape:
            x_shape = list(x.shape)
            other_shape = [x_shape[i] for i in other_dims]
            end_shape = [math.prod([x_shape[i] for i in axis])]
            full_shape = other_shape + end_shape
        else:
            other_shape = tf.gather(tf.shape(x), tf.cast(other_dims, tf.int64))
            full_shape = tf.concat([other_shape, [-1]], axis=0)
        y = tf.reshape(x_permed, shape=full_shape)

    # Sort (in ascending order) everything which allows multiple calls to sort
    # only once (under the hood) and use CSE.
    sorted_y = tf.sort(y, axis=-1, direction="ASCENDING")

    d = tf.cast(tf.shape(y)[-1], "float64")

    def _get_indices(method):
        """Get values of y at the indices implied by method."""
        if method == "lower":
            indices = tf.math.floor((d - 1) * q)
        elif method == "higher":
            indices = tf.math.ceil((d - 1) * q)
        elif method == "nearest":
            indices = tf.round((d - 1) * q)
        # d - 1 will be distinct from d in int32, but not necessarily double.
        # So clip to avoid out of bounds errors.
        return tf.clip_by_value(
            tf.cast(indices, "int32"), 0, tf.shape(y)[-1] - 1
        )

    if method in ["nearest", "lower", "higher"]:
        gathered_y = tf.gather(sorted_y, _get_indices(method), axis=-1)
    elif method == "midpoint":
        gathered_y = 0.5 * (
            tf.gather(sorted_y, _get_indices("lower"), axis=-1)
            + tf.gather(sorted_y, _get_indices("higher"), axis=-1)
        )
    elif method == "linear":
        larger_y_idx = _get_indices("higher")
        exact_idx = (d - 1) * q
        # preserve_gradients
        smaller_y_idx = tf.maximum(larger_y_idx - 1, 0)
        larger_y_idx = tf.minimum(smaller_y_idx + 1, tf.shape(y)[-1] - 1)
        fraction = tf.cast(larger_y_idx, tf.float64) - exact_idx
        fraction = tf.cast(fraction, y.dtype)
        gathered_y = (
            tf.gather(sorted_y, larger_y_idx, axis=-1) * (1 - fraction)
            + tf.gather(sorted_y, smaller_y_idx, axis=-1) * fraction
        )

    # Propagate NaNs
    if x.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
        # Apparently tf.is_nan doesn't like other dtypes
        nan_batch_members = tf.reduce_any(tf.math.is_nan(x), axis=axis)
        right_rank_matched_shape = tf.pad(
            tf.shape(nan_batch_members),
            paddings=[[0, tf.rank(q)]],
            constant_values=1,
        )
        nan_batch_members = tf.reshape(
            nan_batch_members, shape=right_rank_matched_shape
        )
        nan_value = tf.constant(float("NaN"), dtype=x.dtype)
        gathered_y = tf.where(nan_batch_members, nan_value, gathered_y)

    # Expand dimensions if requested
    if keepdims:
        if axis is None:
            ones_vec = tf.ones(shape=[tf.rank(x) + tf.rank(q)], dtype="int32")
            gathered_y *= tf.ones(ones_vec, dtype=gathered_y.dtype)
        else:
            for i in sorted(axis):
                gathered_y = tf.expand_dims(gathered_y, axis=i)

    # rotate_transpose
    shift_value_static = tf.get_static_value(tf.rank(q))
    ndims = tf.TensorShape(gathered_y.shape).rank
    if ndims < 2:
        return gathered_y
    shift_value_static = int(
        math.copysign(1, shift_value_static)
        * (builtins.abs(shift_value_static) % ndims)
    )
    if shift_value_static == 0:
        return gathered_y
    perm = collections.deque(range(ndims))
    perm.rotate(shift_value_static)
    return tf.transpose(a=gathered_y, perm=list(perm))


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    axis = to_tuple_or_list(axis)
    compute_dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, compute_dtype)
    return _quantile(x, q, axis=axis, method=method, keepdims=keepdims)


def ravel(x):
    x = convert_to_tensor(x)
    return tf.reshape(x, [-1])


def unravel_index(indices, shape):
    indices = tf.convert_to_tensor(indices)
    input_dtype = indices.dtype

    if None in shape:
        raise ValueError(
            f"`shape` argument cannot contain `None`. Received: shape={shape}"
        )

    coords = []
    for dim in reversed(shape):
        coords.append(tf.cast(indices % dim, input_dtype))
        indices = indices // dim

    return tuple(reversed(coords))


@sparse.elementwise_unary
def real(x):
    x = convert_to_tensor(x)
    return tf.math.real(x)


@sparse.densifying_unary(np.inf)
def reciprocal(x):
    x = convert_to_tensor(x)
    return tf.math.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    # TODO: tf.repeat doesn't support uint16
    if standardize_dtype(x.dtype) == "uint16":
        x = tf.cast(x, "uint32")
        return tf.cast(tf.repeat(x, repeats, axis=axis), "uint16")
    return tf.repeat(x, repeats, axis=axis)


def reshape(x, newshape):
    x = convert_to_tensor(x)
    if isinstance(newshape, (int, np.integer)):
        newshape = (newshape,)
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import compute_reshape_output_shape

        output_shape = compute_reshape_output_shape(
            x.shape, newshape, "newshape"
        )
        output = tf.sparse.reshape(x, newshape)
        output.set_shape(output_shape)
        return output
    return tf.reshape(x, newshape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    if axis is not None:
        # `tf.roll` requires `shift` and `axis` to have the same length,
        # while numpy broadcasts them against each other.
        shifts, axes = normalize_shift_and_axis(shift, axis)
        return tf.roll(x, shift=shifts, axis=axes)

    # If axis is None, the roll happens as a 1-d tensor.
    original_shape = tf.shape(x)
    x = tf.roll(tf.reshape(x, [-1]), shift, 0)
    return tf.reshape(x, original_shape)


def searchsorted(sorted_sequence, values, side="left"):
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    sequence_len = sorted_sequence.shape[0]
    out_type = (
        "int32"
        if sequence_len is not None and sequence_len <= np.iinfo(np.int32).max
        else "int64"
    )
    values_shape = shape_op(values)
    result = tf.searchsorted(
        sorted_sequence, tf.reshape(values, [-1]), side=side, out_type=out_type
    )
    return tf.reshape(result, values_shape)


@sparse.elementwise_unary
def sign(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # TODO: tf.sign doesn't support uint8, uint16, uint32
    if ori_dtype in ("uint8", "uint16", "uint32"):
        x = tf.cast(x, "int32")
        return tf.cast(tf.sign(x), ori_dtype)
    return tf.sign(x)


@sparse.elementwise_unary
def signbit(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        return tf.fill(tf.shape(x), False)
    elif "int" in ori_dtype:
        return x < 0
    else:
        x = cast(x, "float32")
        return tf.less(
            tf.bitwise.bitwise_and(
                tf.bitcast(x, tf.int32),
                # tf.float32 sign bit
                tf.constant(tf.int32.min, dtype=tf.int32),
            ),
            0,
        )


@sparse.elementwise_unary
def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.sin(x)


def sinc(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    pi_x = x * tf.constant(np.pi, dtype=x.dtype)
    return tf.where(
        tf.equal(x, 0),
        tf.ones_like(x),
        tf.math.sin(pi_x) / pi_x,
    )


@sparse.elementwise_unary
def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    # A dynamic shape has no static size, so return a tensor instead.
    if x.shape.is_fully_defined():
        return x.shape.num_elements()
    return tf.size(x)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # TODO: tf.sort doesn't support bool
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0

    if ori_dtype == "bool":
        x = tf.cast(x, "int8")
        return tf.cast(tf.sort(x, axis=axis), ori_dtype)
    return tf.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    if not isinstance(indices_or_sections, int):
        # `tf.split` requires `num_or_size_splits`, so we need to convert
        # `indices_or_sections` to the appropriate format.
        indices_or_sections = convert_to_tensor(indices_or_sections)
        total_size = shape_op(x)[axis]
        # Like NumPy, wrap negative indices and clip them to the axis length.
        indices_or_sections = tf.where(
            indices_or_sections < 0,
            indices_or_sections + total_size,
            indices_or_sections,
        )
        indices_or_sections = tf.clip_by_value(
            indices_or_sections, 0, total_size
        )
        start_size = indices_or_sections[0:1]
        end_size = total_size - indices_or_sections[-1:]
        num_or_size_splits = tf.concat(
            [start_size, diff(indices_or_sections), end_size], axis=0
        )
    else:
        num_or_size_splits = indices_or_sections
    return tf.split(x, num_or_size_splits, axis=axis)


def array_split(x, indices_or_sections, axis=0):
    x = tf.convert_to_tensor(x)
    num_splits = indices_or_sections
    total_size = shape_op(x)[axis]
    avg_size = total_size // num_splits
    remainder = total_size % num_splits
    sizes = [avg_size + 1] * remainder + [avg_size] * (num_splits - remainder)

    return tf.split(x, sizes, axis=axis)


def stack(x, axis=0):
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tree.map_structure(lambda a: convert_to_tensor(a, dtype), x)
    return tf.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)

    if (
        x.shape.rank is not None
        and isinstance(axis1, int)
        and isinstance(axis2, int)
    ):
        # This branch makes sure `perm` is statically known, to avoid a
        # not-compile-time-constant XLA error.
        axis1 = canonicalize_axis(axis1, x.ndim)
        axis2 = canonicalize_axis(axis2, x.ndim)

        # Directly return x if no movement is required
        if axis1 == axis2:
            return x

        perm = list(range(x.ndim))
        perm[axis1] = axis2
        perm[axis2] = axis1
    else:
        x_rank = tf.rank(x)
        axis1 = tf.where(axis1 < 0, tf.add(axis1, x_rank), axis1)
        axis2 = tf.where(axis2 < 0, tf.add(axis2, x_rank), axis2)
        perm = tf.range(x_rank)
        perm = tf.tensor_scatter_nd_update(
            perm, [[axis1], [axis2]], [axis2, axis1]
        )
    return tf.transpose(x, perm)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = tf.reshape(x, (-1,))
        axis = 0

    def fix_negative_indices(i):
        # Correct the indices using "fill" mode which is the same as in jax
        return tf.where(i < 0, i + tf.cast(tf.shape(x)[axis], i.dtype), i)

    if isinstance(indices, tf.SparseTensor):
        if x.dtype not in (tf.float16, tf.float32, tf.float64, tf.bfloat16):
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`x.dtype={x.dtype}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            indices = convert_to_tensor(indices, sparse=False)
        elif axis != 0:
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`axis={axis}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            indices = convert_to_tensor(indices, sparse=False)
        else:
            indices = sparse.sparse_with_values(
                indices, fix_negative_indices(indices.values)
            )
            # `expand_dims` on `indices` prevents combiner from being applied.
            output = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=tf.convert_to_tensor(x),
                sparse_ids=tf.sparse.expand_dims(indices, axis=-1),
                default_id=0,
            )
            output.set_shape(indices.shape + output.shape[len(indices.shape) :])
            return output
    elif isinstance(indices, tf.RaggedTensor):
        indices = indices.with_values(fix_negative_indices(indices.values))
        if axis == 0:
            return tf.nn.embedding_lookup(x, indices)
        else:
            return tf.gather(x, indices, axis=axis)

    indices = fix_negative_indices(convert_to_tensor(indices))
    return tf.gather(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    from keras.src.ops import operation_utils

    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, "int64")
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                "`indices` must be 1D if axis=None. "
                f"Received: indices.shape={indices.shape}"
            )
        return take_along_axis(tf.reshape(x, [-1]), indices, 0)

    # Compute the static output shape as later on, all shapes manipulations
    # use dynamic shapes.
    static_output_shape = operation_utils.compute_take_along_axis_output_shape(
        x.shape, indices.shape, axis
    )
    rank = x.ndim
    static_axis = axis
    axis = axis + rank if axis < 0 else axis

    if axis >= rank:
        raise ValueError(f"Invalid axis: {static_axis} for input rank: {rank}")

    x_original_shape = shape_op(x)
    indices_original_shape = shape_op(indices)

    # Broadcast the static shapes first, but not for the `axis` dimension.
    x_static_shape = list(x.shape)
    indices_static_shape = list(indices.shape)
    x_static_shape[axis] = 1
    indices_static_shape[axis] = 1
    broadcast_shape = operation_utils.broadcast_shapes(
        x_static_shape, indices_static_shape
    )
    if None in broadcast_shape:
        # Dynamic broadcast case. Note that `tf.broadcast_dynamic_shape` is
        # not always XLA compilable with dynamic dimensions.
        # We replace `None`s with the dynamic dimensions.
        # `maximum` is the correct formula only when shapes are broadcastable,
        # we rely on the broacast itself to fail in the incorrect case rather
        # than make some expensive dynamic checks here.
        broadcast_shape = [
            tf.maximum(x_original_shape[i], indices_original_shape[i])
            if dim is None
            else dim
            for i, dim in enumerate(broadcast_shape)
        ]

    x_shape = list(broadcast_shape)
    x_shape[axis] = x_original_shape[axis]
    indices_shape = list(broadcast_shape)
    indices_shape[axis] = indices_original_shape[axis]
    x = tf.broadcast_to(x, x_shape)
    indices = tf.broadcast_to(indices, indices_shape)

    # Correct the indices using "fill" mode which is the same as in jax
    indices = tf.where(
        indices < 0,
        indices + tf.cast(x_shape[static_axis], dtype=indices.dtype),
        indices,
    )

    x = swapaxes(x, static_axis, -1)
    indices = swapaxes(indices, static_axis, -1)

    x_shape = tf.shape(x)
    x = tf.reshape(x, [-1, x_shape[-1]])
    indices_shape = tf.shape(indices)
    indices = tf.reshape(indices, [-1, indices_shape[-1]])

    result = tf.gather(x, indices, batch_dims=1)
    result = tf.reshape(result, indices_shape)
    result = swapaxes(result, static_axis, -1)
    result.set_shape(static_output_shape)
    return result


@sparse.elementwise_unary
def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.tan(x)


@sparse.elementwise_unary
def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)

    if result_dtype in [
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
    ]:
        compute_dtype = result_dtype
    else:
        compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    return tf.cast(tf.tensordot(x1, x2, axes=axes), dtype=result_dtype)


@sparse.elementwise_unary
def round(x, decimals=0):
    if decimals == 0:
        return tf.round(x)
    x_dtype = x.dtype
    if tf.as_dtype(x_dtype).is_integer:
        # int
        if decimals > 0:
            return x
        # temporarily convert to floats
        factor = tf.cast(math.pow(10, decimals), config.floatx())
        x = tf.cast(x, config.floatx())
    else:
        # float
        factor = tf.cast(math.pow(10, decimals), x.dtype)
    x = tf.multiply(x, factor)
    x = tf.round(x)
    x = tf.divide(x, factor)
    return tf.cast(x, x_dtype)


def tile(x, repeats):
    x = convert_to_tensor(x)

    # Convert repeats to a list (works for both sequences and 1D tensors)
    if isinstance(repeats, int):
        repeats = [repeats]
    else:
        repeats = [v for v in repeats]

    # Process list elements: convert concrete scalar tensors to Python ints
    processed_repeats = []
    for r in repeats:
        if hasattr(r, "numpy") and r.shape == ():
            processed_repeats.append(int(r.numpy()))
        else:
            processed_repeats.append(r)
    repeats = processed_repeats

    # Get x rank
    x_rank = x.shape.rank

    # Pad repeats if needed
    if len(repeats) < x_rank:
        repeats = [1] * (x_rank - len(repeats)) + repeats

    # Add dimensions to x if needed using tf.expand_dims
    while len(repeats) > x.shape.rank:
        x = tf.expand_dims(x, 0)

    return tf.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = tf.cast(x, dtype)
    x_shape = tf.shape(x)
    x = moveaxis(x, (axis1, axis2), (-2, -1))
    # Mask out the diagonal and reduce.
    x = tf.where(
        eye(x_shape[axis1], x_shape[axis2], k=offset, dtype="bool"),
        x,
        tf.zeros_like(x),
    )
    return tf.reduce_sum(x, axis=(-2, -1))


def tri(N, M=None, k=0, dtype=None):
    M = M if M is not None else N
    dtype = standardize_dtype(dtype or config.floatx())
    if k < 0:
        lower = -k - 1
        if lower > N:
            r = tf.zeros([N, M], dtype=dtype)
        else:
            o = tf.ones([N, M], dtype="bool")
            r = tf.cast(
                tf.logical_not(tf.linalg.band_part(o, lower, -1)), dtype=dtype
            )
    else:
        o = tf.ones([N, M], dtype=dtype)
        if k > M:
            r = o
        else:
            r = tf.linalg.band_part(o, -1, k)
    return r


def tril(x, k=0):
    x = convert_to_tensor(x)

    def _negative_k_branch():
        shape = tf.shape(x)
        rows, cols = shape[-2], shape[-1]
        i, j = tf.meshgrid(tf.range(rows), tf.range(cols), indexing="ij")
        mask = i >= j - k
        return tf.where(tf.broadcast_to(mask, shape), x, tf.zeros_like(x))

    if isinstance(k, int):
        if k >= 0:
            return tf.linalg.band_part(x, -1, k)
        return _negative_k_branch()

    # when `k` is a tensor
    return tf.cond(
        tf.greater_equal(k, 0),
        lambda: tf.linalg.band_part(x, -1, k),
        _negative_k_branch,
    )


def triu(x, k=0):
    x = convert_to_tensor(x)

    def _positive_k_branch():
        shape = tf.shape(x)
        rows, cols = shape[-2], shape[-1]
        i, j = tf.meshgrid(tf.range(rows), tf.range(cols), indexing="ij")
        mask = i <= j - k
        return tf.where(tf.broadcast_to(mask, shape), x, tf.zeros_like(x))

    if isinstance(k, int):
        if k <= 0:
            return tf.linalg.band_part(x, -k, -1)
        return _positive_k_branch()

    # when `k` is a tensor
    return tf.cond(
        tf.less_equal(k, 0),
        lambda: tf.linalg.band_part(x, -k, -1),
        _positive_k_branch,
    )


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool" or "int" in dtype:
        return x
    return tf.where(x < 0, tf.math.ceil(x), tf.math.floor(x))


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    x1 = tf.reshape(x1, [-1])
    x2 = tf.reshape(x2, [-1])
    return tf.cast(dot(x1, x2), result_dtype)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    x = tf.cond(
        tf.math.logical_or(
            tf.math.equal(tf.rank(x1), 0),
            tf.math.equal(tf.rank(x2), 0),
        ),
        lambda: x1 * x2,
        lambda: tf.tensordot(x1, x2, axes=[[-1], [-1]]),
    )
    return tf.cast(x, result_dtype)


def vstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    return tf.concat(xs, axis=0)


def vsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=0)


def _vmap_fn(fn, in_axes=0):
    if in_axes != 0:
        raise ValueError(
            "Not supported with `vectorize()` with the TensorFlow backend."
        )

    @functools.wraps(fn)
    def wrapped(x):
        return tf.vectorized_map(fn, x)

    return wrapped


def vectorize(pyfunc, *, excluded=None, signature=None):
    return vectorize_impl(
        pyfunc, _vmap_fn, excluded=excluded, signature=signature
    )


def where(condition, x1=None, x2=None):
    condition = tf.cast(condition, "bool")
    if x1 is not None and x2 is not None:
        if not isinstance(x1, (int, float)):
            x1 = convert_to_tensor(x1)
        if not isinstance(x2, (int, float)):
            x2 = convert_to_tensor(x2)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1 = convert_to_tensor(x1, dtype)
        x2 = convert_to_tensor(x2, dtype)
        return tf.where(condition, x1, x2)
    if x1 is None and x2 is None:
        return nonzero(condition)
    raise ValueError(
        "`x1` and `x2` either both should be `None`"
        " or both should have non-None value."
    )


@sparse.elementwise_division
def divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.divide(x1, x2)


def divide_no_nan(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.math.divide_no_nan(x1, x2)


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    # TODO: tf.pow doesn't support uint* types
    if "uint" in dtype:
        x1 = convert_to_tensor(x1, "int32")
        x2 = convert_to_tensor(x2, "int32")
        return tf.cast(tf.pow(x1, x2), dtype)
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.pow(x1, x2)


def float_power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.pow(x1, x2)


@sparse.elementwise_unary
def negative(x):
    return tf.negative(x)


def _nextafter_16bit(x1, x2, dtype):
    """`nextafter` for 16-bit floats, which `tf.math.nextafter` rejects.

    Steps the IEEE 754 bit pattern by one, which is exactly a one-ulp step.
    Computing in float32 and casting back does not work: a float32 step is
    much smaller than a 16-bit ulp, so the result rounds back to `x1`.
    """
    b1 = tf.bitcast(x1, tf.int16)
    b2 = tf.bitcast(x2, tf.int16)
    zero_i = tf.constant(0, tf.int16)
    one_i = tf.constant(1, tf.int16)
    # Zero is tested on the bit pattern rather than with `==`, because
    # bfloat16 subnormals compare equal to zero in tf's comparison ops.
    abs_mask = tf.constant(32767, tf.int16)
    is_zero_1 = tf.equal(tf.bitwise.bitwise_and(b1, abs_mask), zero_i)
    is_zero_2 = tf.equal(tf.bitwise.bitwise_and(b2, abs_mask), zero_i)
    same = tf.equal(b1, b2) | (is_zero_1 & is_zero_2)
    # The bit pattern grows with magnitude for both signs, so stepping away
    # from zero is +1 for a non-negative x1 and -1 for a negative one.
    step = tf.where(tf.equal(x2 > x1, b1 >= zero_i), one_i, -one_i)
    stepped = tf.bitcast(b1 + step, dtype)
    # Zero has no bit predecessor, so its neighbour is the smallest subnormal
    # carrying the sign of the direction travelled.
    tiny = tf.bitcast(
        tf.where(b2 < zero_i, tf.constant(-32767, tf.int16), one_i), dtype
    )
    out = tf.where(is_zero_1, tiny, stepped)
    out = tf.where(same, x2, out)
    return tf.where(
        tf.math.is_nan(x1) | tf.math.is_nan(x2),
        tf.constant(float("nan"), dtype),
        out,
    )


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    # tf.math.nextafter supports only float32 and float64.
    if dtype in ("float16", "bfloat16"):
        return _nextafter_16bit(x1, x2, dtype)
    return tf.math.nextafter(x1, x2)


@sparse.elementwise_unary
def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    return tf.square(x)


@sparse.elementwise_unary
def sqrt(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.sqrt(x)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    static_shape = x.shape.as_list()
    if axis is not None:
        for a in axis:
            if static_shape[a] != 1:
                raise ValueError(
                    f"Cannot squeeze axis={a}, because the dimension is not 1."
                )
        axis = sorted([canonicalize_axis(a, len(static_shape)) for a in axis])
    if isinstance(x, tf.SparseTensor):
        dynamic_shape = tf.shape(x)
        new_shape = []
        gather_indices = []
        for i, dim in enumerate(static_shape):
            if not (dim == 1 if axis is None else i in axis):
                new_shape.append(dim if dim is not None else dynamic_shape[i])
                gather_indices.append(i)
        new_indices = tf.gather(x.indices, gather_indices, axis=1)
        return tf.SparseTensor(new_indices, x.values, tuple(new_shape))
    return tf.squeeze(x, axis=axis)


def transpose(x, axes=None):
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import compute_transpose_output_shape

        output = tf.sparse.transpose(x, perm=axes)
        output.set_shape(compute_transpose_output_shape(x.shape, axes))
        return output
    if axes:
        axes = tuple(canonicalize_axis(axis, len(x.shape)) for axis in axes)
        # `tf.transpose` raises a low-level `InvalidArgumentError` for
        # duplicate axes (e.g. "1 is missing from {0,0,0}"). The other
        # backends already surface a clear permutation error, so only TF
        # needs the explicit check here. JAX/Torch/NumPy validate natively.
        if len(set(axes)) != len(axes):
            raise ValueError(
                "`axes` must be a valid permutation of the input dimensions "
                f"(no duplicates). Received: axes={list(axes)} for input of "
                f"rank {len(x.shape)}."
            )
    return tf.transpose(x, perm=axes)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    def _move_axis_to_last(tensor, axis):
        if axis == -1:
            return tensor
        rank = tf.rank(tensor)
        if axis < 0:
            axis = rank + axis
        perm = tf.concat(
            [
                tf.range(axis, dtype=tf.int32),
                tf.range(axis + 1, rank, dtype=tf.int32),
                tf.constant([axis], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.transpose(tensor, perm=perm)

    y = convert_to_tensor(y)
    dtype = dtypes.result_type(y.dtype, float)
    y = tf.cast(y, dtype)

    if x is None:
        dx_array = tf.cast(dx, dtype)
    else:
        x = convert_to_tensor(x, dtype=dtype)
        dx_array = diff(x, axis=axis)
        dx_array = _move_axis_to_last(dx_array, axis)

    y = _move_axis_to_last(y, axis)

    avg_heights = 0.5 * (y[..., 1:] + y[..., :-1])
    result = tf.reduce_sum(avg_heights * dx_array, axis=-1)

    return result


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)

    if N is None:
        N = shape_op(x)[0]

    if increasing:
        powers = tf.range(N)
    else:
        powers = tf.range(N - 1, -1, -1)

    x_exp = tf.expand_dims(x, axis=-1)

    compute_dtype = dtypes.result_type(x.dtype, "float32")
    vander = tf.math.pow(
        tf.cast(x_exp, compute_dtype), tf.cast(powers, compute_dtype)
    )
    return tf.cast(vander, result_dtype)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, compute_dtype)
    return tf.cast(
        tf.math.reduce_variance(x, axis=axis, keepdims=keepdims),
        result_dtype,
    )


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    # follow jax's rule
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = cast(x, dtype)
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.reduce_sum(
            x, axis=axis, keepdims=keepdims, output_is_sparse=True
        )
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    M = N if M is None else M
    if isinstance(k, int) and k == 0:
        return tf.eye(N, M, dtype=dtype)
    # Create a smaller square eye and pad appropriately.
    return tf.pad(
        tf.eye(tf.minimum(M - k, N + k), dtype=dtype),
        paddings=(
            (tf.maximum(-k, 0), tf.maximum(N - M + k, 0)),
            (tf.maximum(k, 0), tf.maximum(M - N - k, 0)),
        ),
    )


def floor_divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    if "float" not in dtype:
        return tf.math.floordiv(x1, x2)
    # `tf.math.floordiv` takes the floor of the rounded quotient, which can
    # reach the next whole number: in float32, 4.2 / 1.4 rounds to 3.0, so
    # 4.2 // 1.4 gave 3 instead of 2. Divide the exact multiple
    # `x1 - fmod(x1, x2)` instead, as NumPy and JAX do.
    # `tf.truncatemod` has no float16 or bfloat16 kernel.
    compute_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    mod = tf.truncatemod(x1, x2)
    div = (x1 - mod) / x2
    # `fmod` has the sign of `x1`; floor division rounds toward -inf.
    div = tf.where(
        tf.logical_and(
            tf.not_equal(mod, 0),
            tf.math.logical_xor(tf.less(mod, 0), tf.less(x2, 0)),
        ),
        div - 1,
        div,
    )
    # A zero divisor gives `x1 / x2` (inf, -inf or nan), as before.
    result = tf.where(tf.equal(x2, 0), x1 / x2, tf.round(div))
    return tf.stop_gradient(tf.cast(result, dtype))


def logical_xor(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.math.logical_xor(x1, x2)


def corrcoef(x):
    dtype = x.dtype
    if dtype in ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    x = convert_to_tensor(x, dtype)

    if tf.rank(x) == 0:
        return tf.constant(float("nan"), dtype=config.floatx())
    if x.shape.rank == 1:
        return tf.reshape(corrcoef(x[tf.newaxis, :]), [])

    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x_centered = x - mean

    num_samples = tf.cast(tf.shape(x)[-1], x.dtype)
    cov_matrix = tf.matmul(x_centered, x_centered, adjoint_b=True) / (
        num_samples - 1
    )

    diag = tf.linalg.diag_part(cov_matrix)
    stddev = tf.sqrt(tf.math.real(diag))

    outer_std = tf.tensordot(stddev, stddev, axes=0)
    outer_std = tf.cast(outer_std, cov_matrix.dtype)
    correlation = cov_matrix / outer_std

    correlation_clipped = tf.clip_by_value(tf.math.real(correlation), -1.0, 1.0)
    if correlation.dtype.is_complex:
        imag_clipped = tf.clip_by_value(tf.math.imag(correlation), -1.0, 1.0)
        return tf.complex(correlation_clipped, imag_clipped)
    else:
        return correlation_clipped


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == tf.int64:
        dtype = tf.float64
    elif dtype not in [tf.bfloat16, tf.float16, tf.float64]:
        dtype = tf.float32

    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    def _pack(a, b):
        # a: input [N] -> [1,N,1];
        # b: filter [M] -> [M,1,1]
        return (
            tf.reshape(a, (1, shape_op(a)[0], 1)),
            tf.reshape(b, (shape_op(b)[0], 1, 1)),
        )

    def _full_corr(x1, x2):
        """Compute 'full' correlation result (length = n + m - 1)."""
        m = shape_op(x2)[0]
        pad = (
            builtins.max(m - 1, 0)
            if isinstance(m, int)
            else tf.maximum(m - 1, 0)
        )
        x1 = tf.pad(x1, [[pad, pad]])  # pad input with zeros
        x1, x2 = _pack(x1, x2)
        out = tf.nn.conv1d(x1, x2, stride=1, padding="VALID")
        return tf.squeeze(out, axis=[0, 2])

    n = shape_op(x1)[0]
    m = shape_op(x2)[0]

    if mode == "full":
        return _full_corr(x1, x2)
    elif mode == "same":
        # unfortunately we can't leverage 'SAME' padding directly like
        # we can with "valid"
        # it works fine for odd-length filters, but for even-length filters
        # the output is off by 1 compared to numpy, due to how
        # tf handles centering
        full_corr = _full_corr(x1, x2)
        full_len = n + m - 1
        out_len = (
            max([n, m])
            if isinstance(n, int) and isinstance(m, int)
            else tf.maximum(n, m)
        )
        start = (full_len - out_len + tf.cast(tf.less(n, m), "int32")) // 2
        return tf.slice(full_corr, [start], [out_len])
    elif mode == "valid":
        full_corr = _full_corr(x1, x2)
        return full_corr[tf.minimum(n, m) - 1 : tf.maximum(n, m)]
    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Mode must be one of:"
            f" 'full', 'same', 'valid'."
        )


def select(condlist, choicelist, default=0):
    return tf.experimental.numpy.select(condlist, choicelist, default=default)


def slogdet(x):
    x = convert_to_tensor(x)
    return tuple(tf.linalg.slogdet(x))


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x, tf.int32)

    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
        original_axis = None
    else:
        original_axis = axis

    x = swapaxes(x, axis, -1)

    n = tf.shape(x)[-1]
    kth = tf.clip_by_value(tf.cast(kth, tf.int32), 0, n - 1)

    bottom_ind = tf.math.top_k(-x, kth + 1).indices

    mask = tf.reduce_sum(tf.one_hot(bottom_ind, n, dtype=tf.int32), axis=0)
    indices = tf.where(mask)
    updates = tf.zeros(tf.shape(indices)[0], dtype=tf.int32)
    final_mask = tf.tensor_scatter_nd_update(x, indices, updates)

    top_ind = tf.math.top_k(final_mask, n - kth - 1).indices

    out = tf.concat([bottom_ind, top_ind], axis=-1)

    if original_axis is None:
        return out

    return swapaxes(out, -1, axis)


def histogram(x, bins=10, range=None):
    """Computes a histogram of the data tensor `x`.

    Note: the `tf.histogram_fixed_width()` and
    `tf.histogram_fixed_width_bins()` functions
    yield slight numerical differences for some edge cases.
    """

    x = tf.convert_to_tensor(x, dtype=x.dtype)

    # Handle the range argument
    if range is None:
        min_val = tf.reduce_min(x)
        max_val = tf.reduce_max(x)
    else:
        min_val, max_val = range

    x = tf.boolean_mask(x, (x >= min_val) & (x <= max_val))
    bin_edges = tf.linspace(min_val, max_val, bins + 1)
    bin_edges = tf.cast(bin_edges, x.dtype)
    bin_indices = tf.searchsorted(bin_edges[1:-1], x, side="right")

    # tf.math.bincount does not work with XLA in this case. So, we use
    # `scatter_nd`.
    bin_counts = tf.scatter_nd(
        indices=tf.expand_dims(bin_indices, axis=-1),
        updates=tf.ones_like(bin_indices, dtype=x.dtype),
        shape=(bins,),
    )
    return bin_counts, bin_edges


def unique(
    x,
    sorted=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    size=None,
    fill_value=None,
):
    x = tf.convert_to_tensor(x)
    is_flatten = axis is None
    original_shape = tf.shape(x)

    if is_flatten:
        x = tf.reshape(x, [-1])
        axis = 0
        if return_counts:
            y, inverse, counts = tf.unique_with_counts(x)
        else:
            y, inverse = tf.unique(x)

    else:
        axis = canonicalize_axis(axis, x.shape.rank)
        if return_counts:
            y, inverse, counts = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[axis])
        else:
            y, inverse = tf.raw_ops.UniqueV2(x=x, axis=[axis])

    if return_index:
        num_unique = tf.shape(y)[axis]
        axis_dim = tf.shape(x)[0] if is_flatten else original_shape[axis]
        unique_indices = tf.math.unsorted_segment_min(
            tf.range(axis_dim, dtype=tf.int32), inverse, num_unique
        )

    if sorted:
        num_unique = tf.shape(y)[axis]
        if is_flatten or y.shape.rank == 1:
            sort_order = tf.argsort(y)
        else:
            # Multi-D lexicographical sort
            perm = list(range(y.shape.rank))
            perm[0], perm[axis] = perm[axis], perm[0]
            y_transposed = tf.transpose(y, perm)
            y_2d = tf.reshape(y_transposed, [num_unique, -1])
            num_cols = tf.shape(y_2d)[1]

            sort_order = tf.range(num_unique, dtype=tf.int32)

            def body(i, current_indices):
                col = tf.gather(y_2d[:, i], current_indices)
                perm_sort = tf.argsort(col, stable=True)
                return i - 1, tf.gather(current_indices, perm_sort)

            def cond(i, current_indices):
                return i >= 0

            _, sort_order = tf.while_loop(
                cond, body, [num_cols - 1, sort_order], parallel_iterations=1
            )

        y = tf.gather(y, sort_order, axis=axis)
        if return_index:
            unique_indices = tf.gather(unique_indices, sort_order)
        if return_counts:
            counts = tf.gather(counts, sort_order)
        if return_inverse:
            # Must invert permutation to map inverse indices correctly
            inv_perm = tf.math.invert_permutation(sort_order)
            inverse = tf.gather(inv_perm, inverse)

    # Static size padding/truncation (branchless logic for graph mode safety)
    if size is not None:
        values_count = tf.shape(y)[axis]

        # 1. Truncate using gather
        truncate_size = tf.minimum(values_count, size)
        y = tf.gather(y, tf.range(truncate_size), axis=axis)
        if return_index:
            unique_indices = tf.gather(unique_indices, tf.range(truncate_size))
        if return_counts:
            counts = tf.gather(counts, tf.range(truncate_size))

        # 2. Pad using tf.pad (pad_amount = 0 makes it a no-op)
        pad_amount = tf.maximum(0, size - values_count)
        paddings = tf.zeros([tf.rank(y), 2], dtype=tf.int32)
        paddings = tf.tensor_scatter_nd_update(
            paddings, [[axis, 1]], [pad_amount]
        )

        fill = tf.cast(0 if fill_value is None else fill_value, y.dtype)
        y = tf.pad(y, paddings, constant_values=fill)

        if return_index:
            unique_indices = tf.pad(
                unique_indices, [[0, pad_amount]], constant_values=1
            )

        if return_counts:
            counts = tf.pad(counts, [[0, pad_amount]], constant_values=0)

        # 3. Enforce static shape for XLA compatibility
        if isinstance(size, int):
            static_shape = y.shape.as_list()
            static_shape[axis] = size
            y.set_shape(static_shape)
            if return_index:
                unique_indices.set_shape([size])
            if return_counts:
                counts.set_shape([size])

    if return_inverse and is_flatten:
        inverse = tf.reshape(inverse, original_shape)

    results = [y]
    if return_index:
        results.append(unique_indices)
    if return_inverse:
        results.append(inverse)
    if return_counts:
        results.append(counts)

    return tuple(results) if len(results) > 1 else results[0]


def dsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=2)


def column_stack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    dtype_set = set([x.dtype for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = [tf.cast(x, dtype) for x in xs]
    xs = [tf.expand_dims(x, axis=-1) if len(x.shape) == 1 else x for x in xs]
    return tf.concat(xs, axis=1)


def cov(x):
    x = convert_to_tensor(x)
    if len(x.shape) > 2:
        raise ValueError(
            "Input tensor must have at most 2 dimensions. "
            f"Received: x.shape={x.shape}"
        )

    dtype = standardize_dtype(x.dtype)
    if dtype in ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"
    x = cast(x, dtype)

    # A 0D input has no observations to vary over, as in `np.cov`.
    if len(x.shape) == 0:
        return tf.constant(float("nan"), dtype=x.dtype)

    # `np.cov` squeezes the result when there is only one variable.
    is_scalar = len(x.shape) < 2 or x.shape[0] == 1
    if len(x.shape) == 1:
        x = tf.reshape(x, (1, -1))

    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x_centered = x - mean

    num_samples = tf.cast(tf.shape(x)[-1], x.dtype)
    result = tf.matmul(x_centered, x_centered, adjoint_b=True) / (
        num_samples - 1
    )
    return tf.reshape(result, ()) if is_scalar else result
