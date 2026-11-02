import builtins
import math

import numpy as np
import torch

from keras.src.backend import KerasTensor
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import normalize_shift_and_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.common.backend_utils import vectorize_impl
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.torch.ops.core import cast
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.backend.torch.ops.core import is_tensor
from keras.src.backend.torch.ops.core import to_torch_dtype

TORCH_INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def convert_to_tensors_of_same_dtype(x1, x2, *additional_dtypes):
    if not hasattr(x1, "dtype") and not isinstance(
        x1, (bool, int, float, complex)
    ):
        x1 = convert_to_tensor(x1)
    if not hasattr(x2, "dtype") and not isinstance(
        x2, (bool, int, float, complex)
    ):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        *additional_dtypes,
    )
    return convert_to_tensor(x1, dtype), convert_to_tensor(x2, dtype), dtype


def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the specified plane using PyTorch.

    Args:
        array: Input tensor
        k: Number of 90-degree rotations (default=1)
        axes: Tuple of two axes that define the
            plane of rotation (defaults to `(0, 1)`).

    Returns:
        Rotated tensor
    """
    array = convert_to_tensor(array)

    if array.ndim < 2:
        raise ValueError(
            "Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple "
            "of two different dimensions."
        )

    axes = tuple(axis if axis >= 0 else array.ndim + axis for axis in axes)

    if not builtins.all(0 <= axis < array.ndim for axis in axes):
        raise ValueError(
            f"Invalid axes {axes} for tensor with {array.ndim} dimensions"
        )

    rotated = torch.rot90(array, k=k, dims=axes)
    if isinstance(array, np.ndarray):
        rotated = rotated.cpu().numpy()

    return rotated


def add(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(operand) for operand in operands]
    # When all operands are of int8, we cast the result to int32 to align with
    # the behavior of jax.
    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        compute_dtype = "int32"
        if get_device() == "cuda":
            # TODO: torch.einsum doesn't support int32 when using cuda
            compute_dtype = config.floatx()
        # prevent overflow
        operands = [cast(operand, compute_dtype) for operand in operands]
        return cast(torch.einsum(subscripts, *operands), "int32")
    if len(dtypes_to_resolve) > 1:
        # `torch.einsum` does not promote mixed dtypes (e.g. a float input
        # against an int8 kernel), unlike `matmul` above and the other
        # backends. Resolve the result dtype the way the op's output spec
        # does and cast the operands to it.
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
        compute_dtype = result_dtype
        # TODO: torch.einsum doesn't support integer types with cuda
        if get_device() == "cuda" and "int" in compute_dtype:
            compute_dtype = config.floatx()
        operands = [cast(operand, compute_dtype) for operand in operands]
        return cast(torch.einsum(subscripts, *operands), result_dtype)
    return torch.einsum(subscripts, *operands)


def subtract(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    # TODO: torch.subtract doesn't support bool
    if standardize_dtype(x1.dtype) == "bool":
        x1 = cast(x1, x2.dtype)
    if standardize_dtype(x2.dtype) == "bool":
        x2 = cast(x2, x1.dtype)
    return torch.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)

    def can_use_int_matmul(x1, x2):
        # torch._int_mm only accepts the following conditions:
        # 1. cuda
        # 2. both inputs must have int8 dtype
        # 3. both inputs must be 2d
        # 4. x1.shape must be [>16, >= 16 and a multiplier of 8]
        # 5. x2.shape must be [>= 16 and a multiplier of 8, multiplier of 8]
        if get_device() != "cuda":
            return False
        if x1_dtype != "int8" or x2_dtype != "int8":
            return False
        x1_shape = x1.shape
        x2_shape = x2.shape
        if x1.ndim != 2 or x2.ndim != 2:
            return False
        if x1_shape[0] <= 16 or x1_shape[1] < 16 or x1_shape[1] % 8 != 0:
            return False
        if x2_shape[0] < 16 or x2_shape[0] % 8 != 0 or x2_shape[1] % 8 != 0:
            return False
        return True

    # Shortcut for torch._int_mm
    # TODO: Loosen the restriction of the usage of torch._int_mm
    # TODO: We should replace torch._int_mm with the public api if possible
    # Not yet supported with CUDA 13 without `use_transpose`
    # https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py#L7876
    if can_use_int_matmul(x1, x2):
        try:
            return torch._int_mm(x1, x2)
        except RuntimeError:
            pass

    if x1_dtype == "int8" and x2_dtype == "int8":
        result_dtype = "int32"
    else:
        result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    compute_dtype = result_dtype

    # TODO: torch.matmul doesn't support bool
    if compute_dtype == "bool":
        compute_dtype = config.floatx()
    # TODO: torch.matmul doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"
    # TODO: torch.matmul doesn't support integer types with cuda
    if get_device() == "cuda" and "int" in compute_dtype:
        compute_dtype = config.floatx()

    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    return cast(torch.matmul(x1, x2), result_dtype)


def multiply(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    if isinstance(x, (list, tuple)):
        x = stack(x)
    x = convert_to_tensor(x)
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        return x
    axis = to_tuple_or_list(axis)  # see [NB] below

    ori_dtype = standardize_dtype(x.dtype)
    # torch.mean only supports floating point inputs
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype

    # [NB] the python torch op torch.mean() is generated into
    # `torch._C._VariableFunctions.pyi`, and the method
    # signature is overloaded.
    # Dynamo won't actually find the correct signature of
    # `torch.mean()` if arguments are passed via kwargs
    # So we have to pass the arguments via positional args
    # EXCEPT for those that are forced as kwargs via the `*`
    # delimiter in the overloaded method signatures.
    # Additionally, we have to create a singleton-tuple
    # when `axis` is an int to match the existing fn signature

    # Cast input to compute dtype before mean to avoid dtype kwarg
    # which causes issues with ONNX export (dtype kwarg not supported)
    x = cast(x, compute_dtype)
    result = torch.mean(x, axis, keepdims)
    return cast(result, result_dtype)


def _reduced_axes(x, axis):
    if axis is None:
        return list(range(x.ndim))
    return [canonicalize_axis(a, x.ndim) for a in to_tuple_or_list(axis)]


def _full_like_reduction(x, axes, keepdims, fill_value):
    """Returns `fill_value` in the shape, dtype and device of a reduction."""
    shape = [
        1 if i in axes else d
        for i, d in enumerate(x.shape)
        if keepdims or i not in axes
    ]
    return torch.full(shape, fill_value, dtype=x.dtype, device=x.device)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    # Only a reduction over an empty axis has no result. A zero-size tensor
    # reduced over axes that are not empty gives an empty result.
    axes = _reduced_axes(x, axis)
    if builtins.any(x.shape[a] == 0 for a in axes):
        if initial is None:
            raise ValueError("Cannot compute the max of an empty tensor.")
        return _full_like_reduction(x, axes, keepdims, initial)

    if axis is None:
        # Reduce over every dimension explicitly, since older torch versions
        # do not accept `dim=None` for `torch.amax`.
        axis = tuple(range(x.ndim))
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        result = x
    else:
        result = torch.amax(x, dim=axis, keepdim=keepdims)

    if initial is not None:
        dtype = to_torch_dtype(result.dtype)
        initial = convert_to_tensor(initial, dtype=dtype)
        return torch.maximum(result, initial)
    return result


def ones(shape, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())
    if isinstance(shape, int):
        shape = (shape,)
    return torch.ones(size=shape, dtype=dtype, device=get_device())


def zeros(shape, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())
    if isinstance(shape, int):
        shape = (shape,)
    return torch.zeros(size=shape, dtype=dtype, device=get_device())


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_torch_dtype(dtype or x.dtype)
    return torch.zeros_like(x, dtype=dtype)


def absolute(x):
    x = convert_to_tensor(x)
    # bool are always non-negative
    if standardize_dtype(x.dtype) == "bool":
        return x
    return torch.abs(x)


def abs(x):
    return absolute(x)


def fabs(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = cast(x, config.floatx())
    return torch.abs(x)


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    if axis == () or axis == []:
        return cast(x, "bool")
    if isinstance(axis, list):
        axis = tuple(axis)
    return cast(torch.all(x, dim=axis, keepdim=keepdims), "bool")


def angle(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)

    # torch.angle doesn't support float16 with cuda
    if get_device() != "cpu" and ori_dtype == "float16":
        x = cast(x, "float32")
        return cast(torch.angle(x), "float16")
    return torch.angle(x)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    if axis == () or axis == []:
        return cast(x, "bool")
    if isinstance(axis, list):
        axis = tuple(axis)
    return cast(torch.any(x, dim=axis, keepdim=keepdims), "bool")


def append(x1, x2, axis=None):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    if axis is None:
        return torch.cat((x1.flatten(), x2.flatten()))
    return torch.cat((x1, x2), dim=axis)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [getattr(start, "dtype", type(start))]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = to_torch_dtype(dtype)
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    return torch.arange(
        start, stop, step=step, dtype=dtype, device=get_device()
    )


def arccos(x):
    x = convert_to_tensor(x)
    return torch.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    return torch.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    return torch.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    return torch.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    return torch.arctan(x)


def arctan2(x1, x2):
    x1, x2, result_dtype = convert_to_tensors_of_same_dtype(x1, x2, float)
    compute_dtype = result_dtype
    # TODO: torch.arctan2 doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"
    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    return cast(torch.arctan2(x1, x2), result_dtype)


def arctanh(x):
    x = convert_to_tensor(x)
    return torch.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    # TODO: torch.argmax doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "uint8")

    return cast(torch.argmax(x, dim=axis, keepdim=keepdims), dtype="int32")


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    # TODO: torch.argmin doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "uint8")

    return cast(torch.argmin(x, dim=axis, keepdim=keepdims), dtype="int32")


def argsort(x, axis=-1):
    x = convert_to_tensor(x)

    # TODO: torch.argsort doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "uint8")

    if axis is None:
        axis = -1
        x = x.reshape(-1)
    indices = cast(torch.argsort(x, dim=axis, stable=True), dtype="int32")
    if indices.ndim == 0:
        return indices.unsqueeze(0)
    return indices


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def view(x, dtype=None):
    x = convert_to_tensor(x)
    # `to_torch_dtype(None)` resolves to `floatx()`, so the default has to be
    # resolved to the dtype of `x` to keep it a no-op.
    dtype = x.dtype if dtype is None else to_torch_dtype(dtype)
    return x.view(dtype=dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
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
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = cast(x, dtype)
    if weights is not None:
        weights = cast(weights, dtype)
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        return x
    if weights is not None:
        if len(weights.shape) == 1 and len(x.shape) > 1 and axis is not None:
            a = axis[0] if isinstance(axis, (tuple, list)) else axis
            if isinstance(a, int):
                a = a if a >= 0 else len(x.shape) + a
                w_shape = [1] * len(x.shape)
                w_shape[a] = weights.shape[0]
                weights = torch.reshape(weights, w_shape)

        if axis is not None:
            return torch.sum(torch.mul(x, weights), dim=axis) / torch.sum(
                weights, dim=axis
            )
        return torch.sum(torch.mul(x, weights)) / torch.sum(weights)
    return torch.mean(x, axis)


def bartlett(x):
    x = convert_to_tensor(x)
    return torch.signal.windows.bartlett(x)


def hamming(x):
    x = convert_to_tensor(x)
    return torch.signal.windows.hamming(x)


def hanning(x):
    x = convert_to_tensor(x)
    return torch.signal.windows.hann(x)


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"

    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)

    return torch.heaviside(x1, x2)


def kaiser(x, beta):
    x = convert_to_tensor(x)
    return torch.signal.windows.kaiser(x, beta=beta)


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = "int32"
    if len(x.shape) == 2:
        if weights is None:

            def bincount_fn(arr):
                return torch.bincount(arr, minlength=minlength)

            bincounts = list(map(bincount_fn, x))
        else:

            def bincount_fn(arr_w):
                return torch.bincount(
                    arr_w[0], weights=arr_w[1], minlength=minlength
                )

            bincounts = list(map(bincount_fn, zip(x, weights)))

        return cast(torch.stack(bincounts), dtype)
    return cast(torch.bincount(x, weights, minlength), dtype)


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return torch.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return torch.bitwise_not(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return torch.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return torch.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
    return torch.bitwise_left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
    return torch.bitwise_right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def blackman(x):
    x = convert_to_tensor(x)
    return torch.signal.windows.blackman(x)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return torch.broadcast_to(x, shape)


def cbrt(x):
    x = convert_to_tensor(x)

    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        x = cast(x, "int32")
    elif dtype == "int64":
        x = cast(x, "float64")

    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def ceil(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)

    # TODO: torch.ceil doesn't support bool
    if ori_dtype == "bool":
        x = cast(x, "uint8")
    # TODO: torch.ceil doesn't support float16 with cpu
    elif get_device() == "cpu" and ori_dtype == "float16":
        x = cast(x, config.floatx())

    if ori_dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(ori_dtype, float)
    return cast(torch.ceil(x), dtype=dtype)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    x_min = convert_to_tensor(x_min)
    x_max = convert_to_tensor(x_max)
    ori_dtype = standardize_dtype(x.dtype)

    # TODO: torch.clip doesn't support float16 with cpu
    if get_device() == "cpu" and ori_dtype == "float16":
        x = cast(x, "float32")
        return cast(torch.clip(x, min=x_min, max=x_max), "float16")

    if ori_dtype == "bool":
        x = cast(x, "int32")
    return torch.clip(x, min=x_min, max=x_max)


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.cat(xs, dim=axis)


def conjugate(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.conj(x).resolve_conj()


def conj(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.conj(x).resolve_conj()


def copy(x):
    x = convert_to_tensor(x)
    return torch.clone(x)


def copysign(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return torch.copysign(x1, x2)


def cos(x):
    x = convert_to_tensor(x)
    return torch.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    return torch.cosh(x)


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        return cast(torch.ne(x, 0), "int32")
    return cast(torch.count_nonzero(x, dim=axis).T, "int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axisa != -1 or axisb != -1 or axisc != -1:
        raise ValueError(
            "Torch backend does not support `axisa`, `axisb`, or `axisc`. "
            f"Received: axisa={axisa}, axisb={axisb}, axisc={axisc}. Please "
            "use `axis` arg in torch backend."
        )
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    compute_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    result_dtype = compute_dtype
    # TODO: torch.cross doesn't support bfloat16 with gpu
    if get_device() == "cuda" and compute_dtype == "bfloat16":
        compute_dtype = "float32"
    # TODO: torch.cross doesn't support float16 with cpu
    elif get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"
    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    if axis is None:
        # match numpy default (last axis)
        axis = -1

    x1_dim = x1.shape[axis]
    x2_dim = x2.shape[axis]
    if x1_dim not in (2, 3) or x2_dim not in (2, 3):
        raise ValueError(
            "Both input arrays must be (arrays of) 2 or 3-dimensional "
            f"vectors, but they are {x1_dim} and {x2_dim} dimensional "
            "instead."
        )

    if x1_dim == 3 and x2_dim == 3:
        return cast(torch.linalg.cross(x1, x2, dim=axis), result_dtype)

    # `torch.linalg.cross` only supports 3-dimensional vectors, so pad
    # 2-dimensional vectors to 3 dimensions ourselves (with an implicit zero
    # z-component) before delegating the cross product of the resulting
    # 3-dimensional vectors to `torch.linalg.cross`.
    def _pad_2d_vector_to_3d(x):
        if x.shape[-1] == 2:
            return torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)
        return x

    x1 = _pad_2d_vector_to_3d(torch.movedim(x1, axis, -1))
    x2 = _pad_2d_vector_to_3d(torch.movedim(x2, axis, -1))

    c = torch.linalg.cross(x1, x2, dim=-1)

    if x1_dim == 2 and x2_dim == 2:
        return cast(c[..., 2], result_dtype)
    return cast(torch.movedim(c, -1, axis), result_dtype)


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    # TODO: torch.cumprod doesn't support float16 with cpu
    elif get_device() == "cpu" and dtype == "float16":
        return cast(
            torch.cumprod(x, dim=axis, dtype=to_torch_dtype("float32")),
            "float16",
        )
    return torch.cumprod(x, dim=axis, dtype=to_torch_dtype(dtype))


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    # TODO: torch.cumsum doesn't support float16 with cpu
    elif get_device() == "cpu" and dtype == "float16":
        return cast(
            torch.cumsum(x, dim=axis, dtype=to_torch_dtype("float32")),
            "float16",
        )
    return torch.cumsum(x, dim=axis, dtype=to_torch_dtype(dtype))


def deg2rad(x):
    x = convert_to_tensor(x)

    if standardize_dtype(x.dtype) == "int64":
        return cast(torch.deg2rad(x), "float64")

    return torch.deg2rad(x)


def rad2deg(x):
    x = convert_to_tensor(x)

    if standardize_dtype(x.dtype) == "int64":
        return cast(torch.rad2deg(x), "float64")

    return torch.rad2deg(x)


def diag(x, k=0):
    x = convert_to_tensor(x)
    return torch.diag(x, diagonal=k)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    return torch.diagflat(x, offset=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return torch.diagonal(
        x,
        offset=offset,
        dim1=axis1,
        dim2=axis2,
    )


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    return torch.diff(a, n=n, dim=axis)


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "uint8")
    return cast(torch.bucketize(x, bins, right=True), "int32")


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # GPU only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)

    # TODO: torch.matmul doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"

    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    if x1.ndim == 0 or x2.ndim == 0:
        return cast(torch.multiply(x1, x2), result_dtype)
    if x2.ndim >= 2:
        return cast(
            torch.tensordot(x1, x2, dims=([x1.ndim - 1], [x2.ndim - 2])),
            result_dtype,
        )
    return cast(torch.matmul(x1, x2), result_dtype)


def dstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.dstack(xs)


def empty(shape, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())
    return torch.empty(size=shape, dtype=dtype, device=get_device())


def empty_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_torch_dtype(dtype or x.dtype)
    return torch.empty_like(x, dtype=dtype, device=get_device())


def equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.eq(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return torch.exp(x)


def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return torch.exp2(x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    out_ndim = len(x.shape) + len(axis)
    axis = sorted([canonicalize_axis(a, out_ndim) for a in axis])
    for a in axis:
        x = torch.unsqueeze(x, dim=a)
    return x


def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return torch.expm1(x)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        axis = tuple(range(x.ndim))
    axis = to_tuple_or_list(axis)
    return torch.flip(x, dims=axis)


def fliplr(x):
    x = convert_to_tensor(x)
    return torch.fliplr(x)


def flipud(x):
    x = convert_to_tensor(x)
    return torch.flipud(x)


def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = cast(x, dtype)
    return torch.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = to_torch_dtype(dtype)
    fill_value = convert_to_tensor(fill_value, dtype=dtype)
    if len(fill_value.shape) > 0:
        # `torch.full` only supports scala `fill_value`.
        expand_size = len(shape) - len(fill_value.shape)
        tile_shape = tuple(shape[:expand_size]) + (1,) * len(fill_value.shape)
        return torch.tile(fill_value, tile_shape)
    return torch.full(
        size=shape, fill_value=fill_value, dtype=dtype, device=get_device()
    )


def full_like(x, fill_value, dtype=None):
    dtype = dtype or x.dtype
    return full(shape=x.shape, fill_value=fill_value, dtype=dtype)


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return torch.gcd(x1, x2)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
            "torch does not support an `axis` argument for geomspace. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = to_torch_dtype(dtype)

    start = convert_to_tensor(start, dtype=dtype)
    stop = convert_to_tensor(stop, dtype=dtype)

    log_start = torch.log10(torch.abs(start))
    log_stop = torch.log10(torch.abs(stop))

    result = logspace(
        log_start, log_stop, num=num, endpoint=endpoint, base=10, dtype=dtype
    )
    return result * torch.sign(start)


def greater(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.greater(x1, x2)


def greater_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.greater_equal(x1, x2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.hstack(xs)


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if not isinstance(indices_or_sections, int):
        indices_or_sections = convert_to_tensor(indices_or_sections).tolist()
    return list(torch.hsplit(x, indices_or_sections))


def hypot(x1, x2):
    x1, x2, dtype = convert_to_tensors_of_same_dtype(x1, x2)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"

    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)

    return torch.hypot(x1, x2)


def identity(n, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())

    # TODO: torch.eye doesn't support bfloat16 with cpu
    if get_device() == "cpu" and dtype == torch.bfloat16:
        return cast(
            torch.eye(n, dtype=to_torch_dtype("float32"), device=get_device()),
            dtype,
        )
    return torch.eye(n, dtype=dtype, device=get_device())


def imag(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.imag(x)


def i0(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ["int64", "float64"]:
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16"]:
        dtype = config.floatx()
    x = cast(x, dtype)
    return torch.i0(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = cast(x1, result_dtype)
    x2 = cast(x2, result_dtype)
    if "float" in standardize_dtype(result_dtype):
        return torch.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return torch.eq(x1, x2)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = cast(x1, result_dtype)
    x2 = cast(x2, result_dtype)
    if "float" in standardize_dtype(result_dtype):
        return torch.all(
            torch.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )
    return torch.all(torch.eq(x1, x2))


def isfinite(x):
    x = convert_to_tensor(x)
    return torch.isfinite(x)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        x1 = cast(x1, "int32")
        x2 = cast(x2, "int32")

    if standardize_dtype(x1.dtype) == "bool":
        x1 = cast(x1, x2.dtype)
    if standardize_dtype(x2.dtype) == "bool":
        x2 = cast(x2, x1.dtype)

    return torch.isin(x1, x2, assume_unique=assume_unique, invert=invert)


def isinf(x):
    x = convert_to_tensor(x)
    return torch.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    return torch.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    return torch.isneginf(x)


def isposinf(x):
    x = convert_to_tensor(x)
    return torch.isposinf(x)


def isreal(x):
    x = convert_to_tensor(x)
    return torch.isreal(x)


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return torch.kron(x1, x2)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return torch.lcm(x1, x2)


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)

    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )

    return cast(torch.ldexp(x1, x2), dtype)


def less(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.less(x1, x2)


def less_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if axis != 0:
        raise ValueError(
            "torch.linspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = to_torch_dtype(dtype)

    step = convert_to_tensor(torch.nan)
    if endpoint:
        if num > 1:
            step = (stop - start) / (num - 1)
    else:
        if num > 0:
            step = (stop - start) / num
        if num > 1:
            stop = stop - ((stop - start) / num)
    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start = convert_to_tensor(start, dtype=dtype)
        stop = convert_to_tensor(stop, dtype=dtype)
        steps = torch.arange(num, dtype=dtype, device=get_device()) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # increments from `start` to `stop` in each dimension
        linspace = start[None] + steps * (stop - start)[None]
    else:
        linspace = torch.linspace(
            start=start,
            end=stop,
            steps=num,
            dtype=dtype,
            device=get_device(),
        )
    if retstep is True:
        return (linspace, step)
    return linspace


def log(x):
    x = convert_to_tensor(x)
    return torch.log(x)


def log10(x):
    x = convert_to_tensor(x)
    return torch.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    return torch.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    return torch.log2(x)


def logaddexp(x1, x2):
    x1, x2, dtype = convert_to_tensors_of_same_dtype(x1, x2, float)

    # TODO: torch.logaddexp doesn't support float16 with cpu
    if get_device() == "cpu" and dtype == "float16":
        x1 = cast(x1, "float32")
        x2 = cast(x2, "float32")
        return cast(torch.logaddexp(x1, x2), dtype)
    return torch.logaddexp(x1, x2)


def logaddexp2(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2, float)
    return torch.logaddexp2(x1, x2)


def logical_and(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return torch.logical_not(x)


def logical_or(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
            "torch.logspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = to_torch_dtype(dtype)

    if endpoint is False:
        stop = stop - ((stop - start) / num)
    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start = convert_to_tensor(start, dtype=dtype)
        stop = convert_to_tensor(stop, dtype=dtype)
        steps = torch.arange(num, dtype=dtype, device=get_device()) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # increments from `start` to `stop` in each dimension
        linspace = start[None] + steps * (stop - start)[None]
        logspace = base**linspace
    else:
        compute_dtype = dtype
        # TODO: torch.logspace doesn't support float16 with cpu
        if get_device() == "cpu" and dtype == torch.float16:
            compute_dtype = torch.float32
        logspace = cast(
            torch.logspace(
                start=start,
                end=stop,
                steps=num,
                base=base,
                dtype=compute_dtype,
                device=get_device(),
            ),
            dtype,
        )
    return logspace


def maximum(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.maximum(x1, x2)


def fmax(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.fmax(x1, x2)


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    # `torch.median` returns the lower of the two middle values for an
    # even-length reduction, whereas numpy and the other backends average
    # them. Delegate to `quantile(q=0.5)`, which averages (and returns a
    # float result, including for an empty `axis`), matching numpy.
    return quantile(x, q=0.5, axis=axis, keepdims=keepdims)


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(sc_tensor) for sc_tensor in x]
    return torch.meshgrid(x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    # Only a reduction over an empty axis has no result. A zero-size tensor
    # reduced over axes that are not empty gives an empty result.
    axes = _reduced_axes(x, axis)
    if builtins.any(x.shape[a] == 0 for a in axes):
        if initial is None:
            raise ValueError("Cannot compute the min of an empty tensor.")
        return _full_like_reduction(x, axes, keepdims, initial)

    if axis is None:
        # Reduce over every dimension explicitly, since older torch versions
        # do not accept `dim=None` for `torch.amin`.
        axis = tuple(range(x.ndim))
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        result = x
    else:
        result = torch.amin(x, dim=axis, keepdim=keepdims)

    if initial is not None:
        dtype = to_torch_dtype(result.dtype)
        initial = convert_to_tensor(initial, dtype=dtype)
        return torch.minimum(result, initial)
    return result


def minimum(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.minimum(x1, x2)


def fmin(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.fmin(x1, x2)


def mod(x1, x2):
    x1, x2, dtype = convert_to_tensors_of_same_dtype(x1, x2)
    if dtype == "bool":
        x1 = cast(x1, "int32")
        x2 = cast(x2, "int32")
    return torch.remainder(x1, x2)


def fmod(x1, x2):
    x1, x2, dtype = convert_to_tensors_of_same_dtype(x1, x2)
    if dtype == "bool":
        x1 = cast(x1, "int32")
        x2 = cast(x2, "int32")
    return torch.fmod(x1, x2)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    return torch.moveaxis(x, source=source, destination=destination)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not torch.is_floating_point(x):
        return argmax(x, axis=axis, keepdims=keepdims)

    x_clean = torch.where(torch.isnan(x), float("-inf"), x)

    return torch.where(
        torch.isnan(x).all(dim=axis, keepdim=keepdims),
        torch.tensor(-1, dtype=torch.int32, device=get_device()),
        argmax(x_clean, axis=axis, keepdims=keepdims),
    )


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if not torch.is_floating_point(x):
        return argmin(x, axis=axis, keepdims=keepdims)

    x_clean = torch.where(torch.isnan(x), float("inf"), x)

    return torch.where(
        torch.isnan(x).all(dim=axis, keepdim=keepdims),
        torch.tensor(-1, dtype=torch.int32, device=get_device()),
        argmin(x_clean, axis=axis, keepdims=keepdims),
    )


def nancumsum(x, axis=None, dtype=None):
    x = nan_to_num(x, posinf=float("inf"), neginf=float("-inf"))
    return cumsum(x, axis=axis, dtype=dtype)


def nancumprod(x, axis=None, dtype=None):
    x = nan_to_num(x, nan=1.0, posinf=float("inf"), neginf=float("-inf"))
    return cumprod(x, axis=axis, dtype=dtype)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if not torch.is_floating_point(x):
        return torch.amax(x, dim=axis, keepdim=keepdims)

    if axis == () or axis == []:
        return x

    x_clean = torch.where(torch.isnan(x), float("-inf"), x)
    out = torch.amax(x_clean, dim=axis, keepdim=keepdims)

    return torch.where(
        torch.isnan(x).all(dim=axis, keepdim=keepdims),
        torch.tensor(float("nan"), dtype=x.dtype, device=get_device()),
        out,
    )


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if axis == () or axis == []:
        return x

    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    return torch.nanmean(cast(x, dtype), dim=axis, keepdim=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if axis == () or axis == []:
        return x

    return nanquantile(x, q=0.5, axis=axis, keepdims=keepdims)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if not torch.is_floating_point(x):
        return torch.amin(x, dim=axis, keepdim=keepdims)

    if axis == () or axis == []:
        return x

    x_clean = torch.where(torch.isnan(x), float("inf"), x)
    out = torch.amin(x_clean, dim=axis, keepdim=keepdims)

    return torch.where(
        torch.isnan(x).all(dim=axis, keepdim=keepdims),
        torch.tensor(float("nan"), dtype=x.dtype, device=get_device()),
        out,
    )


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, dtype=config.floatx()) / 100.0
    return nanquantile(x, q, axis=axis, method=method, keepdims=keepdims)


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    if axis == () or axis == []:
        return torch.nan_to_num(x, nan=1)

    if isinstance(axis, (list, tuple)):
        axis = sorted(axis, reverse=True)

    if not torch.is_floating_point(x):
        return prod(x, axis=axis, keepdims=keepdims)

    return prod(
        torch.where(torch.isnan(x), torch.ones((), dtype=x.dtype), x),
        axis=axis,
        keepdims=keepdims,
    )


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    axis = to_tuple_or_list(axis)

    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)

    x = cast(x, compute_dtype)
    if x.dtype != q.dtype:
        q = cast(q, x.dtype)

    if axis is None:
        y = reshape(x, [-1])
    else:
        axis = [canonicalize_axis(a, x.ndim) for a in axis]
        other_dims = sorted(set(range(x.ndim)).difference(axis))
        x_permed = torch.permute(x, dims=(other_dims + list(axis)))

        x_shape = list(x.shape)
        other_shape = [x_shape[i] for i in other_dims]
        end_shape = [math.prod([x_shape[i] for i in axis])]
        full_shape = other_shape + end_shape
        y = reshape(x_permed, full_shape)

    y = torch.nanquantile(y, q, dim=-1, interpolation=method)

    if keepdims:
        if axis is None:
            for _ in range(x.ndim):
                y = expand_dims(y, axis=-1)
        else:
            for i in sorted(axis):
                i = i + 1 if q.ndim > 0 else i
                y = expand_dims(y, axis=i)

    return cast(y, result_dtype)


def nanstd(x, axis=None, keepdims=False):
    var_val = nanvar(x, axis=axis, keepdims=keepdims)
    return torch.sqrt(var_val)


def nansum(x, axis=None, keepdims=False):
    if isinstance(x, (list, tuple)):
        x = stack(x)
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)

    if dtype in ("bool", "uint8", "int8", "int16"):
        dtype = "int32"

    if axis == () or axis == []:
        return cast(torch.nan_to_num(x, nan=0), dtype)
    return cast(torch.nansum(x, dim=axis, keepdim=keepdims), dtype)


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)

    result_dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, result_dtype)

    if axis == () or axis == []:
        return torch.where(torch.isnan(x), x, torch.zeros(()))

    mean = nanmean(x, axis=axis, keepdims=True)

    valid = ~torch.isnan(x)
    centered = torch.where(valid, x - mean, torch.zeros_like(x))

    if torch.is_complex(centered):
        centered = centered.real * centered.real + centered.imag * centered.imag
    else:
        centered = centered.square()

    count = valid.sum(dim=axis, keepdim=keepdims)
    var = centered.sum(dim=axis, keepdim=keepdims) / count
    return var


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    x = convert_to_tensor(x)
    return cast(torch.nonzero(x).T, "int32")


def not_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.not_equal(x1, x2)


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_torch_dtype(dtype or x.dtype)
    return torch.ones_like(x, dtype=dtype)


def outer(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.outer(x1.flatten(), x2.flatten())


def pad(x, pad_width, mode="constant", constant_values=None):
    kwargs = {}
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        kwargs["value"] = constant_values
    x = convert_to_tensor(x)
    if len(pad_width) == 1:
        # A single `(before, after)` pair broadcasts to every axis, matching
        # `np.pad`. `torch.nn.functional.pad` needs a per-axis spec, so expand
        # it here.
        pad_width = [pad_width[0]] * x.ndim
    if mode == "symmetric":
        # torch has no "symmetric" pad, so mirror manually (including the edge
        # value). `replicate` only repeats the edge and is numpy's "edge" mode.
        pw = list(pad_width)
        if len(pw) == 1:
            pw = pw * x.ndim
        for axis, (left, right) in enumerate(pw):
            if left == 0 and right == 0:
                continue
            size = x.shape[axis]
            period = 2 * size
            pos = torch.arange(-left, size + right, device=x.device)
            m = torch.remainder(pos, period)
            idx = torch.where(m < size, m, period - 1 - m)
            x = torch.index_select(x, axis, idx)
        return x
    pad_sum = []
    pad_width = list(pad_width)[::-1]  # torch uses reverse order
    pad_width_sum = 0
    for pad in pad_width:
        pad_width_sum += pad[0] + pad[1]
    for pad in pad_width:
        pad_sum += pad
        pad_width_sum -= pad[0] + pad[1]
        if pad_width_sum == 0:  # early break when no padding in higher order
            break
    if mode == "constant":
        return torch.nn.functional.pad(x, pad=pad_sum, mode=mode, **kwargs)
    # TODO: reflect and symmetric padding are implemented for padding the
    # last 3 dimensions of a 4D or 5D input tensor, the last 2 dimensions of a
    # 3D or 4D input tensor, or the last dimension of a 2D or 3D input tensor.
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    ori_dtype = x.dtype
    ori_ndim = x.ndim
    need_squeeze = False
    if x.ndim < 3:
        need_squeeze = True
        new_dims = [1] * (3 - x.ndim)
        x = x.view(*new_dims, *x.shape)
    need_cast = False
    if x.dtype not in (torch.float32, torch.float64):
        # TODO: reflect and symmetric padding are only supported with float32/64
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        x = cast(x, torch.float32)
    x = torch.nn.functional.pad(x, pad=pad_sum, mode=mode)
    if need_cast:
        x = cast(x, ori_dtype)
    if need_squeeze:
        x = torch.squeeze(x, dim=tuple(range(3 - ori_ndim)))
    return x


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
        # TODO: torch.prod doesn't support uint32
        elif dtype == "uint8":
            dtype = "int32"
    compute_dtype = dtype
    # TODO: torch.prod doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"
    if axis is None:
        # `torch.prod` reduces a single axis at a time and does not accept
        # `dim=None`, so reduce everything and restore the rank if needed.
        ndim = x.ndim
        x = cast(torch.prod(x, dtype=to_torch_dtype(compute_dtype)), dtype)
        if keepdims:
            x = torch.reshape(x, (1,) * ndim)
        return x
    axis = to_tuple_or_list(axis)
    axis = [canonicalize_axis(a, x.ndim) for a in axis]
    for a in sorted(axis, reverse=True):
        # `torch.prod` does not handle multiple axes.
        x = cast(
            torch.prod(
                x, dim=a, keepdim=keepdims, dtype=to_torch_dtype(compute_dtype)
            ),
            dtype,
        )
    return x


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        result = x.max() - x.min()
        if keepdims:
            result = result.reshape((1,) * x.ndim)
        return result
    elif axis == ():
        return torch.zeros_like(x)
    else:
        return torch.amax(x, dim=axis, keepdim=keepdims) - torch.amin(
            x, dim=axis, keepdim=keepdims
        )


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    axis = to_tuple_or_list(axis)

    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)

    x = cast(x, compute_dtype)
    # q must be same dtype as x
    if x.dtype != q.dtype:
        q = cast(q, x.dtype)

    # support multiple axes
    if axis is None:
        y = reshape(x, [-1])
    else:
        # transpose
        axis = [canonicalize_axis(a, x.ndim) for a in axis]
        other_dims = sorted(set(range(x.ndim)).difference(axis))
        perm = other_dims + list(axis)
        x_permed = torch.permute(x, dims=perm)
        # reshape
        x_shape = list(x.shape)
        other_shape = [x_shape[i] for i in other_dims]
        end_shape = [math.prod([x_shape[i] for i in axis])]
        full_shape = other_shape + end_shape
        y = reshape(x_permed, full_shape)

    y = torch.quantile(y, q, dim=-1, interpolation=method)

    if keepdims:
        if axis is None:
            for _ in range(x.ndim):
                y = expand_dims(y, axis=-1)
        else:
            for i in sorted(axis):
                i = i + 1 if q.ndim > 0 else i
                y = expand_dims(y, axis=i)

    return cast(y, result_dtype)


def ravel(x):
    x = convert_to_tensor(x)
    return torch.ravel(x)


def unravel_index(indices, shape):
    indices = convert_to_tensor(indices)
    dtype = dtypes.result_type(indices.dtype)
    return tuple(
        cast(idx, dtype) for idx in torch.unravel_index(indices, shape)
    )


def real(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.real(x)


def reciprocal(x):
    x = convert_to_tensor(x)
    return torch.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)

    if get_device() == "meta":
        x = KerasTensor(x.shape, standardize_dtype(x.dtype))
        outputs = repeat(x, repeats, axis=axis)

        return torch.empty(
            size=outputs.shape,
            dtype=to_torch_dtype(outputs.dtype),
            device=get_device(),
        )

    # When repeats is a scalar int and axis is specified, use
    # unsqueeze + expand + reshape instead of repeat_interleave.
    # repeat_interleave introduces unbacked symbolic integers during
    # torch.export tracing, preventing static shape inference. This
    # alternative preserves static output shapes.
    if isinstance(repeats, int) and axis is not None:
        if axis < 0:
            axis = x.ndim + axis
        shape = list(x.shape)
        x = x.unsqueeze(axis + 1)
        expand_shape = [-1] * x.ndim
        expand_shape[axis + 1] = repeats
        x = x.expand(expand_shape)
        new_shape = list(shape)
        new_shape[axis] = shape[axis] * repeats
        return x.reshape(new_shape)

    # When repeats is a list/tuple of ints, pass it directly to
    # repeat_interleave without converting to a tensor. The output
    # size (sum(repeats)) is statically known, so torch.export can
    # infer shapes without introducing unbacked symbols.
    if (
        isinstance(repeats, (list, tuple))
        and all(isinstance(r, int) for r in repeats)
        and axis is not None
    ):
        return torch.repeat_interleave(x, repeats, dim=axis)

    repeats = convert_to_tensor(repeats, dtype=int)

    return torch.repeat_interleave(x, repeats, dim=axis)


def reshape(x, newshape):
    if not isinstance(newshape, (list, tuple)):
        newshape = (newshape,)
    x = convert_to_tensor(x)
    return torch.reshape(x, newshape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    if axis is not None:
        # `torch.roll` requires `shifts` and `dims` to have the same length,
        # while numpy broadcasts them against each other.
        shifts, axes = normalize_shift_and_axis(shift, axis)
        return torch.roll(x, tuple(shifts), dims=tuple(axes))
    return torch.roll(x, shift)


def searchsorted(sorted_sequence, values, side="left"):
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    out_int32 = sorted_sequence.shape[0] <= np.iinfo(np.int32).max
    return torch.searchsorted(
        sorted_sequence, values, side=side, out_int32=out_int32
    )


def sign(x):
    x = convert_to_tensor(x)
    return torch.sign(x)


def signbit(x):
    x = convert_to_tensor(x)
    return torch.signbit(x)


def sin(x):
    x = convert_to_tensor(x)
    return torch.sin(x)


def sinc(x):
    x = convert_to_tensor(x)
    return torch.sinc(x)


def sinh(x):
    x = convert_to_tensor(x)
    return torch.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    return math.prod(x.shape)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    # TODO: torch.sort doesn't support bool with cuda
    if get_device() == "cuda" and standardize_dtype(x.dtype) == "bool":
        x = cast(x, "uint8")
        return cast(torch.sort(x, dim=axis).values, "bool")
    return torch.sort(x, dim=axis).values


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    dim = x.shape[axis]
    if not isinstance(indices_or_sections, int):
        indices_or_sections = convert_to_tensor(indices_or_sections)
        start_size = indices_or_sections[0:1]
        end_size = dim - indices_or_sections[-1:]
        chunk_sizes = torch.concat(
            [start_size, torch.diff(indices_or_sections), end_size], dim=0
        )
        # torch.split doesn't support tensor input for `split_size_or_sections`
        chunk_sizes = chunk_sizes.tolist()
    else:
        if dim % indices_or_sections != 0:
            raise ValueError(
                f"Received indices_or_sections={indices_or_sections} "
                f"(interpreted as a number of sections) and axis={axis}, "
                f"but input dimension x.shape[{axis}]={x.shape[axis]} "
                f"is not divisible by {indices_or_sections}. "
                f"Full input shape: x.shape={x.shape}"
            )
        chunk_sizes = dim // indices_or_sections
    out = torch.split(
        tensor=x,
        split_size_or_sections=chunk_sizes,
        dim=axis,
    )
    if dim == 0 and isinstance(indices_or_sections, int):
        out = [out[0].clone() for _ in range(indices_or_sections)]
    return list(out)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    out = torch.tensor_split(x, indices_or_sections, dim=axis)
    return list(out)


def stack(x, axis=0):
    x = [convert_to_tensor(elem) for elem in x]
    return torch.stack(x, dim=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, "float32")
    # Remove Bessel correction to align with numpy
    return torch.std(x, dim=axis, keepdim=keepdims, unbiased=False)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return torch.swapaxes(x, axis0=axis1, axis1=axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    # Correct the indices using "fill" mode which is the same as in jax
    x_dim = x.shape[axis] if axis is not None else x.shape[0]
    indices = torch.where(
        indices < 0,
        indices + x_dim,
        indices,
    )
    if x.ndim == 2 and axis == 0:
        # This case is equivalent to embedding lookup.
        return torch.nn.functional.embedding(indices, x)
    if axis is None:
        x = torch.reshape(x, (-1,))
        axis = 0
    if axis is not None:
        axis = canonicalize_axis(axis, x.ndim)
        shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
        # ravel the `indices` since `index_select` expects `indices`
        # to be a vector (1-D tensor).
        indices = indices.ravel()
        out = torch.index_select(x, dim=axis, index=indices).squeeze(axis)
        return out.reshape(shape)
    return torch.take(x, index=indices)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    # Correct the indices using "fill" mode which is the same as in jax
    x_dim = x.shape[axis] if axis is not None else x.shape[0]
    indices = torch.where(
        indices < 0,
        indices + x_dim,
        indices,
    )
    return torch.take_along_dim(x, indices, dim=axis)


def tan(x):
    x = convert_to_tensor(x)
    return torch.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    return torch.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # TODO: torch.tensordot only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)
    # TODO: torch.tensordot doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"
    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    # torch only handles dims=((0,), (1,)), numpy accepts axes=(0, 1).
    if isinstance(axes, (list, tuple)):
        first, second = axes
        if not isinstance(first, (list, tuple)):
            first = (first,)
        if not isinstance(second, (list, tuple)):
            second = (second,)
        axes = (first, second)
    return cast(torch.tensordot(x1, x2, dims=axes), result_dtype)


def round(x, decimals=0):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # TODO: torch.round doesn't support int8, int16, int32, int64, uint8
    if "int" in ori_dtype:
        x = cast(x, config.floatx())
        return cast(torch.round(x, decimals=decimals), ori_dtype)
    return torch.round(x, decimals=decimals)


def tile(x, repeats):
    if is_tensor(repeats):
        repeats = tuple(repeats.int().numpy())
    if isinstance(repeats, int):
        repeats = (repeats,)
    x = convert_to_tensor(x)
    return torch.tile(x, dims=repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16", "uint8"):
        # Torch backend doesn't support uint32 dtype.
        dtype = "int32"
    return torch.sum(
        torch.diagonal(x, offset, axis1, axis2),
        dim=-1,
        dtype=to_torch_dtype(dtype),
    )


def tri(N, M=None, k=0, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())
    M = M or N
    x = torch.ones((N, M), dtype=dtype, device=get_device())
    return torch.tril(x, diagonal=k)


def tril(x, k=0):
    x = convert_to_tensor(x)
    return torch.tril(x, diagonal=k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    return torch.triu(x, diagonal=k)


def trunc(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        return x
    return torch.trunc(x)


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # TODO: torch.vdot only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)

    # TODO: torch.vdot doesn't support float16 with cpu
    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"

    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    return cast(torch.vdot(x1, x2), result_dtype)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    compute_dtype = dtypes.result_type(result_dtype, float)

    if get_device() == "cpu" and compute_dtype == "float16":
        compute_dtype = "float32"

    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    return cast(torch.inner(x1, x2), result_dtype)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.vstack(xs)


def vsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if not isinstance(indices_or_sections, int):
        indices_or_sections = convert_to_tensor(indices_or_sections).tolist()
    return list(torch.vsplit(x, indices_or_sections))


def vectorize(pyfunc, *, excluded=None, signature=None):
    return vectorize_impl(
        pyfunc, torch.vmap, excluded=excluded, signature=signature
    )


def where(condition, x1=None, x2=None):
    condition = convert_to_tensor(condition, dtype=bool)
    if x1 is not None and x2 is not None:
        x1 = convert_to_tensor(x1)
        x2 = convert_to_tensor(x2)
        return torch.where(condition, x1, x2)
    else:
        # `torch.where(condition)` returns a tuple of tensors.
        return torch.stack(torch.where(condition), dim=0)


def divide(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2, float)
    return torch.divide(x1, x2)


def divide_no_nan(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2, float)
    safe_x2 = torch.where(x2 == 0, torch.ones_like(x2), x2)
    return torch.where(x2 == 0, 0, torch.divide(x1, safe_x2))


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    x1, x2, _ = convert_to_tensors_of_same_dtype(x1, x2)
    return torch.pow(x1, x2)


def float_power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    # `torch.float_power` always computes in float64, which the MPS device
    # does not support, so raise the operands to a float dtype instead.
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return torch.pow(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return torch.negative(x)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return torch.nextafter(x1, x2)


def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "int32")
    return torch.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return torch.sqrt(x)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    if axis is not None:
        return torch.squeeze(x, dim=axis)
    return torch.squeeze(x)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if axes is not None:
        return torch.permute(x, dims=axes)
    return x.T


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    if standardize_dtype(y.dtype) == "bool":
        y = cast(y, config.floatx())
    if x is not None:
        x = convert_to_tensor(x)
        return torch.trapz(y, x=x, dim=axis)
    else:
        dx = convert_to_tensor(dx)
        return torch.trapz(y, dx=dx, dim=axis)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)
    return cast(torch.vander(x, N=N, increasing=increasing), result_dtype)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    if axis == [] or axis == ():
        # Torch handles the empty axis case differently from numpy.
        return zeros_like(x, result_dtype)
    # Bessel correction removed for numpy compatibility
    x = cast(x, compute_dtype)
    return cast(
        torch.var(x, dim=axis, keepdim=keepdims, correction=0), result_dtype
    )


def sum(x, axis=None, keepdims=False):
    if isinstance(x, (list, tuple)):
        x = stack(x)
    x = convert_to_tensor(x)
    if axis == () or axis == []:
        # Torch handles the empty axis case differently from numpy.
        return x
    dtype = standardize_dtype(x.dtype)
    # follow jax's rule
    # TODO: torch doesn't support uint32
    if dtype in ("bool", "uint8", "int8", "int16"):
        dtype = "int32"
    return cast(torch.sum(x, dim=axis, keepdim=keepdims), dtype)


def eye(N, M=None, k=0, dtype=None):
    dtype = to_torch_dtype(dtype or config.floatx())
    M = N if M is None else M
    k = 0 if k is None else k
    if k == 0:
        # TODO: torch.eye doesn't support bfloat16 with cpu
        if get_device() == "cpu" and dtype == torch.bfloat16:
            return cast(
                torch.eye(
                    N, M, dtype=to_torch_dtype("float32"), device=get_device()
                ),
                dtype,
            )
        return torch.eye(N, M, dtype=dtype, device=get_device())
    diag_length = builtins.max(N, M)
    diag = torch.ones(diag_length, dtype=dtype, device=get_device())
    return torch.diag(diag, diagonal=k)[:N, :M]


def floor_divide(x1, x2):
    x1, x2, dtype = convert_to_tensors_of_same_dtype(x1, x2)
    return cast(torch.floor_divide(x1, x2), dtype)


def logical_xor(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.logical_xor(x1, x2)


def corrcoef(x):
    x = convert_to_tensor(x)

    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, config.floatx())
    elif standardize_dtype(x.dtype) == "int64":
        x = cast(x, "float64")

    return torch.corrcoef(x)


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == "int64":
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16", "float64"]:
        dtype = "float32"

    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)

    x1_len, x2_len = x1.size(0), x2.size(0)

    if x1.shape[:-1] != x2.shape[:-1]:
        new_shape = [max(i, j) for i, j in zip(x1.shape[:-1], x2.shape[:-1])]
        x1 = torch.broadcast_to(x1, new_shape + [x1.shape[-1]])
        x2 = torch.broadcast_to(x2, new_shape + [x2.shape[-1]])

    num_signals = torch.tensor(x1.shape[:-1]).prod()
    x1 = torch.reshape(x1, (int(num_signals), x1.size(-1)))
    x2 = torch.reshape(x2, (int(num_signals), x2.size(-1)))

    output = torch.nn.functional.conv1d(
        x1, x2.unsqueeze(1), groups=x1.size(0), padding=x2.size(-1) - 1
    )
    output_shape = x1.shape[:-1] + (-1,)
    result = output.reshape(output_shape)

    if mode == "valid":
        target_length = (
            builtins.max(x1_len, x2_len) - builtins.min(x1_len, x2_len) + 1
        )
        start_idx = (result.size(-1) - target_length) // 2
        result = result[..., start_idx : start_idx + target_length]

    if mode == "same":
        start_idx = (result.size(-1) - x1_len) // 2
        result = result[..., start_idx : start_idx + x1_len]

    return torch.squeeze(result, dim=0)


def select(condlist, choicelist, default=0):
    condlist = [convert_to_tensor(c) for c in condlist]
    choicelist = [convert_to_tensor(c) for c in choicelist]
    out = convert_to_tensor(default)
    for c, v in reversed(list(zip(condlist, choicelist))):
        out = torch.where(c, v, out)
    return out


def slogdet(x):
    x = convert_to_tensor(x)
    return tuple(torch.linalg.slogdet(x))


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x, "int32")

    if axis is None:
        x = torch.flatten(x)
        axis = 0
        original_axis = None
    else:
        original_axis = axis

    x = torch.transpose(x, axis, -1)

    bottom_ind = torch.topk(-x, kth + 1)[1]

    def set_to_zero(a, i):
        a = a.clone()
        a[i] = 0
        return a

    for _ in range(x.dim() - 1):
        set_to_zero = torch.vmap(set_to_zero)

    proxy = set_to_zero(torch.ones_like(x, dtype=torch.int32), bottom_ind)

    top_ind = torch.topk(proxy, x.shape[-1] - kth - 1)[1]

    out = torch.cat([bottom_ind, top_ind], dim=-1)

    if original_axis is None:
        return cast(out, "int32")

    return cast(torch.transpose(out, -1, axis), "int32")


def histogram(x, bins=10, range=None):
    hist_result = torch.histogram(x, bins=bins, range=range)
    return hist_result.hist, hist_result.bin_edges


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
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    output = torch.unique(
        x,
        sorted=sorted,  # Added sorted parameter here
        return_inverse=(return_inverse or return_index),
        return_counts=return_counts,
        dim=axis,
    )

    if not (return_inverse or return_counts or return_index):
        output = [output]
    else:
        output = list(output)

    values = output[0]

    if axis is None:
        dim = 0
    else:
        dim = canonicalize_axis(axis, x.ndim)

    if return_index:
        inverse = output[1]
        perm = torch.arange(
            inverse.size(0), dtype=inverse.dtype, device=inverse.device
        )
        unique_indices = inverse.new_empty(values.size(dim)).scatter_reduce_(
            0, inverse, perm, reduce="min", include_self=False
        )

    if size is not None:
        values_count = values.shape[dim]

        if values_count > size:
            # Truncate
            indices = [slice(None)] * values.ndim
            indices[dim] = slice(0, size)
            values = values[tuple(indices)]
            if return_counts:
                output[-1] = output[-1][indices[dim]]
            if return_index:
                unique_indices = unique_indices[indices[dim]]

        elif values_count < size:
            # Pad
            diff = size - values_count
            pad_width = [0, 0] * values.ndim
            # F.pad expects padding from last dim to first
            idx = (values.ndim - 1 - dim) * 2
            pad_width[idx + 1] = diff
            fill = 0 if fill_value is None else fill_value
            values = torch.nn.functional.pad(values, pad_width, value=fill)
            if return_counts:
                output[-1] = torch.nn.functional.pad(
                    output[-1], pad_width[idx : idx + 2], value=0
                )
            if return_index:
                unique_indices = torch.nn.functional.pad(
                    unique_indices, pad_width[idx : idx + 2], value=1
                )

    output[0] = values
    if return_index and return_inverse:
        output.insert(1, unique_indices)
    elif return_index:
        output[1] = unique_indices
    return output[0] if len(output) == 1 else tuple(output)


def dsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if not isinstance(indices_or_sections, int):
        indices_or_sections = convert_to_tensor(indices_or_sections).tolist()
    return list(torch.dsplit(x, indices_or_sections))


def column_stack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    dtype = dtypes.result_type(*(x.dtype for x in xs))
    xs = [cast(x, dtype) for x in xs]
    return torch.column_stack(xs)


def cov(x):
    x = convert_to_tensor(x)
    if len(x.shape) > 2:
        raise ValueError(
            "Input tensor must have at most 2 dimensions. "
            f"Received: x.shape={tuple(x.shape)}"
        )

    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, config.floatx())
    elif standardize_dtype(x.dtype) == "int64":
        x = cast(x, "float64")

    return torch.cov(x)
