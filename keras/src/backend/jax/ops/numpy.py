import builtins
import math
import re
import string

import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
from jax import export as jax_export

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.jax import sparse
from keras.src.backend.jax.ops import nn
from keras.src.backend.jax.ops.core import cast
from keras.src.backend.jax.ops.core import convert_to_tensor


def _uses_cpu(x):
    if hasattr(x, "device"):
        device = x.device
        if not isinstance(device, jax.Device):
            # Array is sharded.
            return False
        return device.platform == "cpu"
    else:
        # This is a Tracer, not a concrete Array.
        return jax.default_backend() == "cpu"


def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the specified plane."""
    if array.ndim < 2:
        raise ValueError(
            f"Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple of "
            "two different dimensions."
        )
    return jnp.rot90(array, k=k, axes=axes)


@sparse.elementwise_binary_union(linear=True, use_sparsify=True)
def add(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.add(x1, x2)


def bartlett(x):
    x = convert_to_tensor(x)
    return cast(jnp.bartlett(x), config.floatx())


def hamming(x):
    x = convert_to_tensor(x)
    return cast(jnp.hamming(x), config.floatx())


def hanning(x):
    x = convert_to_tensor(x)
    return cast(jnp.hanning(x), config.floatx())


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.heaviside(x1, x2)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.hypot(x1, x2)


def kaiser(x, beta):
    x = convert_to_tensor(x)
    return cast(jnp.kaiser(x, beta), config.floatx())


def bincount(x, weights=None, minlength=0, sparse=False):
    # Note: bincount is never traceable / jittable because the output shape
    # depends on the values in x.
    if sparse or isinstance(x, jax_sparse.BCOO):
        if isinstance(x, jax_sparse.BCOO):
            if weights is not None:
                if not isinstance(weights, jax_sparse.BCOO):
                    raise ValueError("`x` and `weights` must both be BCOOs")
                if x.indices is not weights.indices:
                    # This test works in eager mode only
                    if not jnp.all(jnp.equal(x.indices, weights.indices)):
                        raise ValueError(
                            "`x` and `weights` BCOOs must have the same indices"
                        )
                weights = weights.data
            x = x.data
        reduction_axis = 1 if len(x.shape) > 1 else 0
        maxlength = jnp.maximum(jnp.max(x) + 1, minlength)
        one_hot_encoding = nn.one_hot(x, maxlength, sparse=True)
        if weights is not None:
            expanded_weights = jnp.expand_dims(weights, reduction_axis + 1)
            one_hot_encoding = one_hot_encoding * expanded_weights

        outputs = jax_sparse.bcoo_reduce_sum(
            one_hot_encoding,
            axes=(reduction_axis,),
        )
        return outputs
    if len(x.shape) == 2:
        if weights is None:

            def bincount_fn(arr):
                return jnp.bincount(arr, minlength=minlength)

            bincounts = list(map(bincount_fn, x))
        else:

            def bincount_fn(arr_w):
                return jnp.bincount(
                    arr_w[0], weights=arr_w[1], minlength=minlength
                )

            bincounts = list(map(bincount_fn, zip(x, weights)))

        return jnp.stack(bincounts)
    return jnp.bincount(x, weights=weights, minlength=minlength)


def _one_row_int8_einsum_subscripts(subscripts, x1, x2):
    """Returns the subscripts of a one-row int8 `einsum` padded to two rows.

    A one-row `einsum` is a `dot_general` of `x1` with `x2`. Each label of
    `x1` is contracted with `x2` or is a free label of size 1, and the output
    is the free labels of `x1` and then those of `x2`. The subscripts add two
    leading labels, of sizes 1 and 2, to `x1` and to the output. They name
    the labels of `x2` first, in the axis order of `x2`, and then the other
    labels of `x1`. `jnp.einsum` sorts the contracted labels, and a sorted
    order that differs from the axis order of `x2` makes XLA write a
    transposed int8 copy of `x2`. Returns `None` for any other `einsum`.
    """
    if not isinstance(subscripts, str):
        return None
    match = re.fullmatch(
        r"([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]*)", subscripts.replace(" ", "")
    )
    if match is None:
        return None
    lhs, rhs, output = match.groups()
    names = dict(zip(dict.fromkeys(rhs + lhs), string.ascii_letters))
    if (
        set(lhs).isdisjoint(rhs)
        or len(set(lhs)) < len(lhs)
        or len(set(rhs)) < len(rhs)
        or len(lhs) != x1.ndim
        or len(rhs) != x2.ndim
        or len(names) > len(string.ascii_letters) - 2
    ):
        return None
    free1 = "".join(c for c in lhs if c not in rhs)
    free2 = "".join(c for c in rhs if c not in lhs)
    # With this output, `jnp.einsum` adds no transpose after the dot. A
    # transpose after a two-row int8 dot can give wrong values on XLA CPU for
    # jax 0.9.1 to 0.11.2.
    if output != free1 + free2:
        return None
    sizes2 = dict(zip(rhs, x2.shape))
    for label, size in zip(lhs, x1.shape):
        # A symbolic size, as in `jax.export`, keeps the unpadded `einsum`.
        if label in free1 and not (isinstance(size, int) and size == 1):
            return None
        # A contracted size that differs keeps the unpadded `einsum`, so its
        # error names the subscripts of the user.
        if label not in free1 and size != sizes2[label]:
            return None
    new = string.ascii_letters[len(names) : len(names) + 2]
    table = str.maketrans(names)
    return (
        f"{new}{lhs.translate(table)},{rhs.translate(table)}"
        f"->{new}{output.translate(table)}"
    )


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    # When all operands are of int8, specifying `preferred_element_type` as
    # int32 to enable hardware-accelerated einsum
    dtypes = list(set(standardize_dtype(x.dtype) for x in operands))
    if len(dtypes) == 1 and dtypes[0] == "int8":
        preferred_element_type = "int32"
    else:
        preferred_element_type = None
    kwargs["preferred_element_type"] = preferred_element_type
    if (
        preferred_element_type == "int32"
        and len(operands) == 2
        and kwargs.get("out_sharding") is None
        and _uses_cpu(operands[0])
    ):
        x1, x2 = operands
        padded = _one_row_int8_einsum_subscripts(subscripts, x1, x2)
        if padded is not None:
            # XLA CPU runs a one-row int8 dot as an int32 dot on an int32 copy
            # of `x2`. Two rows keep the int8 dot, so pad `x1` to two rows and
            # keep the first row of the result. The padded `x1` has rank 3 or
            # more: with a rank-2 `x1`, a transposed `x2` also gets the int32
            # dot.
            x1 = jnp.pad(x1[None, None], ((0, 0), (0, 1)) + ((0, 0),) * x1.ndim)
            return jnp.einsum(padded, x1, x2, **kwargs)[0, 0]
    return jnp.einsum(subscripts, *operands, **kwargs)


@sparse.elementwise_binary_union(linear=True, use_sparsify=True)
def subtract(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # When both x1 and x2 are of int8, specifying `preferred_element_type` as
    # int32 to enable hardware-accelerated matmul
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if x1_dtype == "int8" and x2_dtype == "int8":
        preferred_element_type = "int32"
    else:
        preferred_element_type = None
    if isinstance(x1, jax_sparse.JAXSparse) or isinstance(
        x2, jax_sparse.JAXSparse
    ):
        if not hasattr(matmul, "sparse_matmul"):
            matmul.sparse_matmul = jax_sparse.sparsify(jnp.matmul)
        if isinstance(x1, jax_sparse.BCOO):
            x1 = jax_sparse.bcoo_update_layout(
                x1, n_batch=len(x1.shape) - 2, on_inefficient="warn"
            )
        if isinstance(x2, jax_sparse.BCOO):
            x2 = jax_sparse.bcoo_update_layout(
                x2, n_batch=len(x2.shape) - 2, on_inefficient="warn"
            )
        return matmul.sparse_matmul(
            x1, x2, preferred_element_type=preferred_element_type
        )

    if (
        preferred_element_type == "int32"
        and x1.ndim >= 1
        and x2.ndim == 2
        and math.prod(x1.shape[:-1]) == 1
        and _uses_cpu(x1)
    ):
        # XLA CPU runs a one-row int8 dot as an int32 dot on an int32 copy
        # of `x2`. Two rows keep the int8 dot, so pad `x1` to two rows and
        # keep the first row of the result. The padded `x1` is rank 3: with
        # a rank-2 `x1`, a transposed `x2` also gets the int32 dot.
        outputs = jnp.matmul(
            jnp.pad(jnp.reshape(x1, (1, 1, -1)), ((0, 0), (0, 1), (0, 0))),
            x2,
            preferred_element_type=preferred_element_type,
        )
        return jnp.reshape(outputs[:, :1], x1.shape[:-1] + x2.shape[-1:])
    return jnp.matmul(x1, x2, preferred_element_type=preferred_element_type)


def multiply(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if isinstance(x1, jax_sparse.BCOO):
        if isinstance(x2, jax_sparse.BCOO):
            # x1 is sparse, x2 is sparse.
            if x1.indices is x2.indices:
                # `bcoo_multiply_sparse` will not detect that the indices are
                # the same, optimize this case here.
                if not x1.unique_indices:
                    x1 = jax_sparse.bcoo_sum_duplicates(x1)
                    x2 = jax_sparse.bcoo_sum_duplicates(x2)
                return jax_sparse.BCOO(
                    (jnp.multiply(x1.data, x2.data), x1.indices),
                    shape=x1.shape,
                    indices_sorted=True,
                    unique_indices=True,
                )
            else:
                return jax_sparse.bcoo_multiply_sparse(x1, x2)
        else:
            # x1 is sparse, x2 is dense.
            out_data = jax_sparse.bcoo_multiply_dense(x1, x2)
            return jax_sparse.BCOO(
                (out_data, x1.indices),
                shape=x1.shape,
                indices_sorted=x1.indices_sorted,
                unique_indices=x1.unique_indices,
            )
    elif isinstance(x2, jax_sparse.BCOO):
        # x1 is dense, x2 is sparse.
        out_data = jax_sparse.bcoo_multiply_dense(x2, x1)
        return jax_sparse.BCOO(
            (out_data, x2.indices),
            shape=x2.shape,
            indices_sorted=x2.indices_sorted,
            unique_indices=x2.unique_indices,
        )
    return jnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # `jnp.mean` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype
    if isinstance(x, jax_sparse.BCOO):
        if axis is None:
            axis = tuple(range(len(x.shape)))
        (
            canonical_axis,
            keep_dims_shape,
            broadcast_dimensions,
        ) = sparse.axis_shape_dims_for_broadcast_in_dim(
            axis, x.shape, insert_dims=False
        )
        divisor = math.prod(x.shape[i] for i in canonical_axis)
        output = jax_sparse.bcoo_reduce_sum(x, axes=canonical_axis)
        output = jax_sparse.BCOO(
            (output.data.astype(result_dtype) / divisor, output.indices),
            shape=output.shape,
        )
        if keepdims:
            # `bcoo_reduce_sum` does not support keepdims, neither does
            # sparsify(jnp.sum), so we recreate the empty dimensions.
            output = jax_sparse.bcoo_broadcast_in_dim(
                output,
                shape=keep_dims_shape,
                broadcast_dimensions=broadcast_dimensions,
            )
        return output
    else:
        output = jnp.mean(x, axis=axis, keepdims=keepdims, dtype=compute_dtype)
        return cast(output, result_dtype)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    return jnp.max(x, axis=axis, keepdims=keepdims, initial=initial)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.zeros(shape, dtype=dtype)


@sparse.elementwise_unary(linear=False)
def absolute(x):
    x = convert_to_tensor(x)
    return jnp.absolute(x)


def abs(x):
    return absolute(x)


def fabs(x):
    x = convert_to_tensor(x)
    return jnp.fabs(x)


def all(x, axis=None, keepdims=False):
    return jnp.all(x, axis=axis, keepdims=keepdims)


def angle(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.angle(x)


def any(x, axis=None, keepdims=False):
    return jnp.any(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.append(x1, x2, axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    def get_dtype(x):
        if hasattr(x, "dtype"):
            return x.dtype
        if jax_export.is_symbolic_dim(x):
            return int
        return type(x)

    if dtype is None:
        dtypes_to_resolve = [get_dtype(start)]
        if stop is not None:
            dtypes_to_resolve.append(get_dtype(stop))
        if step is not None:
            dtypes_to_resolve.append(get_dtype(step))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = standardize_dtype(dtype)
    return jnp.arange(start, stop, step=step, dtype=dtype)


@sparse.densifying_unary
def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arccos(x)


@sparse.densifying_unary
def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arccosh(x)


@sparse.elementwise_unary(linear=False)
def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arcsin(x)


@sparse.elementwise_unary(linear=False)
def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arcsinh(x)


@sparse.elementwise_unary(linear=False)
def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return jnp.arctan2(x1, x2)


@sparse.elementwise_unary(linear=False)
def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0 or not _uses_cpu(x):
        return jnp.argmax(x, axis=axis, keepdims=keepdims)

    # Fix the flush-to-zero (FTZ) issue based on this issue:
    # https://github.com/jax-ml/jax/issues/24280
    dtype = dtypes.result_type(dtype, "float32")
    x = cast(x, dtype)
    is_negative_zero = (x == 0.0) & jnp.signbit(x)
    x = jnp.where(is_negative_zero, -jnp.finfo(x.dtype).tiny, x)
    return jnp.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0 or not _uses_cpu(x):
        return jnp.argmin(x, axis=axis, keepdims=keepdims)

    # Fix the flush-to-zero (FTZ) issue based on this issue:
    # https://github.com/jax-ml/jax/issues/24280
    dtype = dtypes.result_type(dtype, "float32")
    x = cast(x, dtype)
    is_negative_zero = (x == 0.0) & jnp.signbit(x)
    x = jnp.where(is_negative_zero, -jnp.finfo(x.dtype).tiny, x)
    return jnp.argmin(x, axis=axis, keepdims=keepdims)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        return jnp.argsort(x, axis=None)
    return jnp.argsort(x, axis=axis)


def array(x, dtype=None):
    return jnp.array(x, dtype=dtype)


def view(x, dtype=None):
    x = convert_to_tensor(x)
    return x.view(dtype=dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = cast(x, dtype)
    if weights is not None:
        weights = cast(weights, dtype)
    return jnp.average(x, weights=weights, axis=axis)


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return jnp.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return jnp.invert(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return jnp.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return jnp.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
    return jnp.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
    return jnp.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def blackman(x):
    x = convert_to_tensor(x)
    return cast(jnp.blackman(x), config.floatx())


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return jnp.broadcast_to(x, shape)


def cbrt(x):
    x = convert_to_tensor(x)
    return jnp.cbrt(x)


@sparse.elementwise_unary(linear=False)
def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.ceil(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "int32")
    return jnp.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    bcoo_count = builtins.sum(isinstance(x, jax_sparse.BCOO) for x in xs)
    if bcoo_count == len(xs):
        axis = canonicalize_axis(axis, len(xs[0].shape))
        return jax_sparse.bcoo_concatenate(xs, dimension=axis)
    elif bcoo_count:
        xs = [
            x.todense()
            if isinstance(x, jax_sparse.JAXSparse)
            else convert_to_tensor(x)
            for x in xs
        ]
    else:
        xs = [convert_to_tensor(x) for x in xs]
    return jnp.concatenate(xs, axis=axis)


@sparse.elementwise_unary(linear=True)
def conjugate(x):
    x = convert_to_tensor(x)
    return jnp.conjugate(x)


@sparse.elementwise_unary(linear=True)
def conj(x):
    x = convert_to_tensor(x)
    return jnp.conjugate(x)


@sparse.elementwise_unary(linear=True)
def copy(x):
    x = convert_to_tensor(x)
    return jnp.copy(x)


def copysign(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.copysign(x1, x2)


@sparse.densifying_unary
def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.cos(x)


@sparse.densifying_unary
def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.cosh(x)


def count_nonzero(x, axis=None):
    return cast(jnp.count_nonzero(x, axis=axis), "int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    axisa = canonicalize_axis(axisa if axis is None else axis, x1.ndim)
    axisb = canonicalize_axis(axisb if axis is None else axis, x2.ndim)

    x1_dim = x1.shape[axisa]
    x2_dim = x2.shape[axisb]

    if x1_dim not in (2, 3) or x2_dim not in (2, 3):
        raise ValueError(
            "Both input arrays must be (arrays of) 2 or 3-dimensional "
            f"vectors, but they are {x1_dim} and {x2_dim} dimensional "
            "instead."
        )

    if x1_dim == 3 and x2_dim == 3:
        return jnp.cross(
            x1,
            x2,
            axisa=axisa,
            axisb=axisb,
            axisc=axisc,
            axis=axis,
        )

    # Newer versions of JAX removed support for 2-dimensional vectors in
    # `jnp.cross`, so pad them to 3 dimensions ourselves (with an implicit zero
    # z-component) before delegating the cross product of the resulting
    # 3-dimensional vectors to `jnp.cross`.
    def _pad_2d_vector_to_3d(x):
        if x.shape[-1] == 2:
            return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 1)])
        return x

    x1 = jnp.moveaxis(x1, axisa if axis is None else axis, -1)
    x2 = jnp.moveaxis(x2, axisb if axis is None else axis, -1)
    x1 = _pad_2d_vector_to_3d(x1)
    x2 = _pad_2d_vector_to_3d(x2)

    c = jnp.cross(x1, x2)

    if x1_dim == 2 and x2_dim == 2:
        return c[..., 2]
    axisc = canonicalize_axis(axisc if axis is None else axis, c.ndim)
    return jnp.moveaxis(c, -1, axisc)


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    return jnp.cumprod(x, axis=axis, dtype=dtype)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    return jnp.cumsum(x, axis=axis, dtype=dtype)


def deg2rad(x):
    x = convert_to_tensor(x)
    return jnp.deg2rad(x)


def rad2deg(x):
    x = convert_to_tensor(x)
    return jnp.rad2deg(x)


def diag(x, k=0):
    x = convert_to_tensor(x)
    return jnp.diag(x, k=k)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    return jnp.diagflat(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return jnp.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    return jnp.diff(a, n=n, axis=axis)


@sparse.elementwise_unary(linear=False)
def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    return jnp.digitize(x, bins)


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.dot(x1, x2)


def dstack(xs):
    return jnp.dstack(xs)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.empty(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return jnp.empty_like(x, dtype=dtype)


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.equal(x1, x2)


@sparse.densifying_unary
def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return jnp.exp(x)


@sparse.densifying_unary
def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return jnp.exp2(x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    if isinstance(x, jax_sparse.BCOO):
        (
            _,
            result_shape,
            broadcast_dimensions,
        ) = sparse.axis_shape_dims_for_broadcast_in_dim(
            axis, x.shape, insert_dims=True
        )
        return jax_sparse.bcoo_broadcast_in_dim(
            x, shape=result_shape, broadcast_dimensions=broadcast_dimensions
        )
    return jnp.expand_dims(x, axis)


@sparse.elementwise_unary(linear=False)
def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return jnp.expm1(x)


def flip(x, axis=None):
    return jnp.flip(x, axis=axis)


def fliplr(x):
    x = convert_to_tensor(x)
    return jnp.fliplr(x)


def flipud(x):
    x = convert_to_tensor(x)
    return jnp.flipud(x)


@sparse.elementwise_unary(linear=False)
def floor(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return jnp.full_like(x, fill_value, dtype=dtype)


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.gcd(x1, x2)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    return jnp.geomspace(
        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
    )


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.greater_equal(x1, x2)


def hstack(xs):
    return jnp.hstack(xs)


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    return jnp.hsplit(x, indices_or_sections)


def identity(n, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.identity(n, dtype=dtype)


@sparse.elementwise_unary(linear=True)
def imag(x):
    x = convert_to_tensor(x)
    return jnp.imag(x)


def i0(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ["int64", "float64"]:
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16"]:
        dtype = config.floatx()
    x = cast(x, dtype)
    return jnp.i0(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.isclose(x1, x2, rtol, atol, equal_nan)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.allclose(x1, x2, rtol, atol, equal_nan)


@sparse.densifying_unary
def isfinite(x):
    x = convert_to_tensor(x)
    return jnp.isfinite(x)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.isin(x1, x2, assume_unique=assume_unique, invert=invert)


@sparse.elementwise_unary(linear=False)
def isinf(x):
    x = convert_to_tensor(x)
    return jnp.isinf(x)


@sparse.elementwise_unary(linear=False)
def isnan(x):
    x = convert_to_tensor(x)
    return jnp.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    return jnp.isneginf(x)


def isposinf(x):
    x = convert_to_tensor(x)
    return jnp.isposinf(x)


def isreal(x):
    x = convert_to_tensor(x)
    return jnp.isreal(x)


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.kron(x1, x2)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.lcm(x1, x2)


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )

    return jnp.ldexp(x1, x2)


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    return jnp.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


@sparse.densifying_unary
def log(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log(x)


@sparse.densifying_unary
def log10(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log10(x)


@sparse.elementwise_unary(linear=False)
def log1p(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log1p(x)


@sparse.densifying_unary
def log2(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log2(x)


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return jnp.logaddexp(x1, x2)


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return jnp.logaddexp2(x1, x2)


def logical_and(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return jnp.logical_not(x)


def logical_or(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    return jnp.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


@sparse.elementwise_binary_union(linear=False, use_sparsify=False)
def maximum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.maximum(x1, x2)


def fmax(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.fmax(x1, x2)


def median(x, axis=None, keepdims=False):
    # axis of jnp.median must be hashable
    if isinstance(axis, list):
        axis = tuple(axis)
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())

    result = jnp.median(x, axis=axis, keepdims=keepdims)

    # TODO: with jax < 0.4.26 jnp.median failed to keepdims when axis is None
    if keepdims is True and axis is None:
        while result.ndim < x.ndim:
            result = jnp.expand_dims(result, axis=-1)
    return result


def meshgrid(*x, indexing="xy"):
    return jnp.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    return jnp.min(x, axis=axis, keepdims=keepdims, initial=initial)


@sparse.elementwise_binary_union(linear=False, use_sparsify=False)
def minimum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.minimum(x1, x2)


def fmin(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.fmin(x1, x2)


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.mod(x1, x2)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.fmod(x1, x2)


def moveaxis(x, source, destination):
    return jnp.moveaxis(x, source=source, destination=destination)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanargmax(x, axis=axis, keepdims=keepdims)


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanargmin(x, axis=axis, keepdims=keepdims)


def nancumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    return jnp.nancumsum(x, axis=axis, dtype=dtype)


def nancumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    return jnp.nancumprod(x, axis=axis, dtype=dtype)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanmax(x, axis=axis, keepdims=keepdims)


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanmean(x, axis=axis, keepdims=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanmedian(x, axis=axis, keepdims=keepdims)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanmin(x, axis=axis, keepdims=keepdims)


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    return jnp.nanpercentile(x, q, axis=axis, method=method, keepdims=keepdims)


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanprod(x, axis=axis, keepdims=keepdims)


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    return jnp.nanquantile(x, q, axis=axis, method=method, keepdims=keepdims)


def nanstd(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanstd(x, axis=axis, keepdims=keepdims)


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nansum(x, axis=axis, keepdims=keepdims)


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.nanvar(x, axis=axis, keepdims=keepdims)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    return jnp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def ndim(x):
    return jnp.ndim(x)


def nonzero(x):
    return jnp.nonzero(x)


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return jnp.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return jnp.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    return jnp.outer(x1, x2)


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
    return jnp.pad(x, pad_width, mode=mode, **kwargs)


def percentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    return jnp.percentile(x, q, axis=axis, method=method, keepdims=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    return jnp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.ptp(x, axis=axis, keepdims=keepdims)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())

    result = jnp.quantile(x, q, axis=axis, method=method, keepdims=keepdims)

    # TODO: with jax < 0.4.26 jnp.quantile failed to keepdims when axis is None
    if keepdims is True and axis is None:
        result_ndim = x.ndim + (1 if len(q.shape) > 0 else 0)
        while result.ndim < result_ndim:
            result = jnp.expand_dims(result, axis=-1)
    return result


def ravel(x):
    x = convert_to_tensor(x)
    return jnp.ravel(x)


def unravel_index(indices, shape):
    indices = convert_to_tensor(indices)
    return jnp.unravel_index(indices, shape)


@sparse.elementwise_unary(linear=True)
def real(x):
    x = convert_to_tensor(x)
    return jnp.real(x)


@sparse.densifying_unary
def reciprocal(x):
    x = convert_to_tensor(x)
    return jnp.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    return jnp.repeat(x, repeats, axis=axis)


def reshape(x, newshape):
    if isinstance(x, jax_sparse.BCOO):
        from keras.src.ops import operation_utils

        # Resolve the -1 in `new_shape` if applicable and possible
        output_shape = operation_utils.compute_reshape_output_shape(
            x.shape, newshape, "new_shape"
        )
        if None not in output_shape:
            newshape = output_shape
        return jax_sparse.bcoo_reshape(x, new_sizes=newshape)
    x = convert_to_tensor(x)
    return jnp.reshape(x, newshape)


def roll(x, shift, axis=None):
    return jnp.roll(x, shift, axis=axis)


def searchsorted(sorted_sequence, values, side="left"):
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    return jnp.searchsorted(sorted_sequence, values, side=side)


@sparse.elementwise_unary(linear=False)
def sign(x):
    x = convert_to_tensor(x)
    return jnp.sign(x)


@sparse.elementwise_unary(linear=False)
def signbit(x):
    x = convert_to_tensor(x)
    return jnp.signbit(x)


@sparse.elementwise_unary(linear=False)
def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.sin(x)


def sinc(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.sinc(x)


@sparse.elementwise_unary(linear=False)
def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.sinh(x)


def size(x):
    return jnp.size(x)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    return jnp.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    return jnp.split(x, indices_or_sections, axis=axis)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    return jnp.array_split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    x = [convert_to_tensor(t) for t in x]
    return jnp.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return jnp.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, sparse=False)
    return jnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, sparse=False)
    return jnp.take_along_axis(x, indices, axis=axis)


@sparse.elementwise_unary(linear=False)
def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.tan(x)


@sparse.elementwise_unary(linear=False)
def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.tensordot(x1, x2, axes=axes)


@sparse.elementwise_unary(linear=False)
def round(x, decimals=0):
    x = convert_to_tensor(x)

    # jnp.round doesn't support decimals < 0 for integers
    x_dtype = standardize_dtype(x.dtype)
    if "int" in x_dtype and decimals < 0:
        factor = cast(math.pow(10, decimals), config.floatx())
        x = cast(x, config.floatx())
        x = jnp.multiply(x, factor)
        x = jnp.round(x)
        x = jnp.divide(x, factor)
        return cast(x, x_dtype)
    else:
        return jnp.round(x, decimals=decimals)


def tile(x, repeats):
    return jnp.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    x = convert_to_tensor(x)
    return jnp.tril(x, k=k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    return jnp.triu(x, k=k)


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or "bool" == dtype:
        return x
    return jnp.trunc(x)


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.vdot(x1, x2)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.inner(x1, x2)


def vstack(xs):
    return jnp.vstack(xs)


def vsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    return jnp.vsplit(x, indices_or_sections)


def vectorize(pyfunc, *, excluded=None, signature=None):
    if excluded is None:
        excluded = set()
    return jnp.vectorize(pyfunc, excluded=excluded, signature=signature)


def where(condition, x1=None, x2=None):
    return jnp.where(condition, x1, x2)


@sparse.elementwise_division
def divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.divide(x1, x2)


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    safe_x2 = jnp.where(x2 == 0, 1, x2)
    return jnp.where(x2 == 0, 0, jnp.divide(x1, safe_x2))


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.power(x1, x2)


def float_power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.float_power(x1, x2)


@sparse.elementwise_unary(linear=True)
def negative(x):
    x = convert_to_tensor(x)
    return jnp.negative(x)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.nextafter(x1, x2)


@sparse.elementwise_unary(linear=False)
def square(x):
    x = convert_to_tensor(x)
    return jnp.square(x)


@sparse.elementwise_unary(linear=False)
def sqrt(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.sqrt(x)


def squeeze(x, axis=None):
    if isinstance(x, jax_sparse.BCOO):
        if axis is None:
            axis = tuple(i for i, d in enumerate(x.shape) if d == 1)
        axis = to_tuple_or_list(axis)
        return jax_sparse.bcoo_squeeze(x, dimensions=axis)
    x = convert_to_tensor(x)
    return jnp.squeeze(x, axis=axis)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if isinstance(x, jax_sparse.BCOO):
        num_dims = len(x.shape)
        if axes is None:
            permutation = tuple(range(num_dims)[::-1])
        else:
            permutation = []
            for a in axes:
                a = canonicalize_axis(a, num_dims)
                permutation.append(a)
        return jax_sparse.bcoo_transpose(x, permutation=permutation)
    return jnp.transpose(x, axes=axes)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    if x is not None:
        x = convert_to_tensor(x)
    dx = convert_to_tensor(dx)
    return jnp.trapezoid(y, x, dx=dx, axis=axis)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    return jnp.vander(x, N=N, increasing=increasing)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # `jnp.var` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    return cast(
        jnp.var(x, axis=axis, keepdims=keepdims, dtype=compute_dtype),
        result_dtype,
    )


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(x, jax_sparse.BCOO):
        if axis is None:
            axis = tuple(range(len(x.shape)))
        (
            canonical_axis,
            keep_dims_shape,
            broadcast_dimensions,
        ) = sparse.axis_shape_dims_for_broadcast_in_dim(
            axis, x.shape, insert_dims=False
        )
        output = jax_sparse.bcoo_reduce_sum(x, axes=canonical_axis)
        if keepdims:
            # `bcoo_reduce_sum` does not support keepdims, neither does
            # sparsify(jnp.sum), so we recreate the empty dimensions.
            output = jax_sparse.bcoo_broadcast_in_dim(
                output,
                shape=keep_dims_shape,
                broadcast_dimensions=broadcast_dimensions,
            )
        return output
    return jnp.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.eye(N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.floor_divide(x1, x2)


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_xor(x1, x2)


def corrcoef(x):
    x = convert_to_tensor(x)
    return jnp.corrcoef(x)


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.correlate(x1, x2, mode)


def select(condlist, choicelist, default=0):
    return jnp.select(condlist, choicelist, default=default)


def slogdet(x):
    x = convert_to_tensor(x)
    return tuple(jnp.linalg.slogdet(x))


def argpartition(x, kth, axis=-1):
    if axis is None:
        x = jnp.reshape(x, (-1,))
        return jnp.argpartition(x, kth, axis=0)

    return jnp.argpartition(x, kth, axis=axis)


def histogram(x, bins=10, range=None):
    return jnp.histogram(x, bins=bins, range=range)


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
    return jnp.unique(
        x,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=False,
        size=size,
        sorted=sorted,
        fill_value=fill_value,
    )


def dsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    return jnp.dsplit(x, indices_or_sections)


def column_stack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return jnp.column_stack(xs)


def cov(x):
    x = convert_to_tensor(x)
    return jnp.cov(x)
