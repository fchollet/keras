import builtins
import math as python_math
import re

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import normalize_shift_and_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape


def _compute_binary_output_spec(x1, x2):
    x1_shape = getattr(x1, "shape", [])
    x2_shape = getattr(x2, "shape", [])
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    return KerasTensor(broadcast_shapes(x1_shape, x2_shape), dtype=dtype)


class Rot90(Operation):
    def __init__(self, k=1, axes=(0, 1), *, name=None):
        super().__init__(name=name)
        self.k = k
        self.axes = axes

    def call(self, array):
        return backend.ops.numpy.rot90(array, k=self.k, axes=self.axes)

    def compute_output_spec(self, array):
        array_shape = list(array.shape)
        if len(array_shape) < 2:
            raise ValueError(
                "Input array must have at least 2 dimensions. "
                f"Received: array.shape={array_shape}"
            )
        if len(self.axes) != 2 or self.axes[0] == self.axes[1]:
            raise ValueError(
                f"Invalid axes: {self.axes}. "
                "Axes must be a tuple of two different dimensions."
            )
        if self.k % 2 == 1:
            axis1, axis2 = self.axes
            array_shape[axis1], array_shape[axis2] = (
                array_shape[axis2],
                array_shape[axis1],
            )
        return KerasTensor(shape=array_shape, dtype=array.dtype)


@keras_export(["keras.ops.rot90", "keras.ops.numpy.rot90"])
def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes.

    This function rotates an array counterclockwise
    by 90 degrees `k` times in the plane specified by `axes`.
    Supports arrays of two or more dimensions.

    Args:
        array: Input array to rotate.
        k: Number of times the array is rotated by 90 degrees.
        axes: A tuple of two integers specifying the
            plane of rotation (defaults to `(0, 1)`).

    Returns:
        Rotated array.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> m = np.array([[1, 2], [3, 4]])
    >>> rotated = ops.rot90(m)
    >>> rotated
    array([[2, 4],
           [1, 3]])

    >>> m = np.arange(8).reshape((2, 2, 2))
    >>> rotated = ops.rot90(m, k=1, axes=(1, 2))
    >>> rotated
    array([[[1, 3],
            [0, 2]],
           [[5, 7],
            [4, 6]]])
    """
    if any_symbolic_tensors((array,)):
        return Rot90(k=k, axes=axes).symbolic_call(array)
    return backend.ops.numpy.rot90(array, k=k, axes=axes)


def shape_equal(shape1, shape2, axis=None, allow_none=True):
    """Check if two shapes are equal.

    Args:
        shape1: A list or tuple of integers for first shape to be compared.
        shape2: A list or tuple of integers for second shape to be compared.
        axis: An integer, list, or tuple of integers (optional):
            Axes to ignore during comparison. Defaults to `None`.
        allow_none (bool, optional): If `True`, allows `None` in a shape
            to match any value in the corresponding position of the other shape.
            Defaults to `True`.

    Returns:
        bool: `True` if shapes are considered equal based on the criteria,
        `False` otherwise.

    Examples:

    >>> shape_equal((32, 64, 128), (32, 64, 128))
    True
    >>> shape_equal((32, 64, 128), (32, 64, 127))
    False
    >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=True)
    True
    >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=False)
    False
    >>> shape_equal((32, 64, 128), (32, 63, 128), axis=1)
    True
    >>> shape_equal((32, 64, 128), (32, 63, 127), axis=(1, 2))
    True
    >>> shape_equal((32, 64, 128), (32, 63, 127), axis=[1,2])
    True
    >>> shape_equal((32, 64), (32, 64, 128))
    False
    """
    if len(shape1) != len(shape2):
        return False

    shape1 = list(shape1)
    shape2 = list(shape2)

    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        for ax in axis:
            shape1[ax] = -1
            shape2[ax] = -1

    if allow_none:
        for i in range(len(shape1)):
            if shape1[i] is None:
                shape1[i] = shape2[i]
            if shape2[i] is None:
                shape2[i] = shape1[i]

    return shape1 == shape2


class Absolute(Operation):
    def call(self, x):
        return backend.ops.numpy.absolute(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.absolute", "keras.ops.numpy.absolute"])
def absolute(x):
    """Compute the absolute value element-wise.

    `keras.ops.abs` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        An array containing the absolute value of each element in `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-1.2, 1.2])
    >>> keras.ops.absolute(x)
    array([1.2, 1.2], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.ops.numpy.absolute(x)


class Abs(Absolute):
    pass


@keras_export(["keras.ops.abs", "keras.ops.numpy.abs"])
def abs(x):
    """Shorthand for `keras.ops.absolute`."""
    return absolute(x)


class Fabs(Operation):
    def call(self, x):
        return backend.ops.numpy.fabs(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        dtype = standardize_dtype(getattr(x, "dtype", type(x)))
        if "complex" in dtype:
            raise TypeError(
                f"fabs does not support complex inputs. Received: dtype={dtype}"
            )
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.fabs", "keras.ops.numpy.fabs"])
def fabs(x):
    """Compute the absolute values element-wise.

    Integer or boolean inputs are automatically promoted to the
    default floating-point type.
    Complex values are not supported.

    Args:
        x: Input tensor.

    Returns:
        An array containing the absolute value of each element in `x`.

    Example:

     >>> x = keras.ops.convert_to_tensor([-1, 2], dtype="int32")
    >>> keras.ops.fabs(x)
    array([1., 2.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Fabs().symbolic_call(x)

    x = backend.ops.convert_to_tensor(x)
    if "complex" in standardize_dtype(x.dtype):
        raise TypeError(
            f"fabs does not support complex inputs. Received: x.dtype={x.dtype}"
        )

    return backend.ops.numpy.fabs(x)


class Add(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.add(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.add", "keras.ops.numpy.add"])
def add(x1, x2):
    """Add arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        The tensor containing the element-wise sum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 4])
    >>> x2 = keras.ops.convert_to_tensor([5, 6])
    >>> keras.ops.add(x1, x2)
    array([6, 10], dtype=int32)

    `keras.ops.add` also broadcasts shapes:
    >>> x1 = keras.ops.convert_to_tensor(
    ...     [[5, 4],
    ...      [5, 6]]
    ... )
    >>> x2 = keras.ops.convert_to_tensor([5, 6])
    >>> keras.ops.add(x1, x2)
    array([[10 10]
           [10 12]], shape=(2, 2), dtype=int32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Add().symbolic_call(x1, x2)
    return backend.ops.numpy.add(x1, x2)


class All(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.all(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_export(["keras.ops.all", "keras.ops.numpy.all"])
def all(x, axis=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical AND reduction is performed. The default
            (`axis=None`) is to perform a logical AND over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Defaults to `False`.

    Returns:
        The tensor containing the logical AND reduction over the `axis`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([True, False])
    >>> keras.ops.all(x)
    array(False, shape=(), dtype=bool)

    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, axis=0)
    array([ True False], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.all(x, axis=axis, keepdims=keepdims)


class AllClose(Operation):
    def __init__(self, rtol=1e-5, atol=1e-8, equal_nan=False, *, name=None):
        super().__init__(name=name)
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def call(self, x1, x2):
        return _allclose(
            x1,
            x2,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=self.equal_nan,
        )

    def compute_output_spec(self, x1, x2):
        return KerasTensor([], dtype="bool")


@keras_export(["keras.ops.allclose", "keras.ops.numpy.allclose"])
def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol * abs(b)`) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        rtol: The relative tolerance parameter (see Notes).
        atol: The absolute tolerance parameter (see Notes).
        equal_nan: Whether to compare NaN's as equal.  If True, NaN's in
            `a` will be considered equal to NaN's in `b` in the output array.

    Returns:
        True if the two arrays are equal within the given tolerance;
        False otherwise.
    """
    if any_symbolic_tensors((x1, x2)):
        return AllClose(
            rtol=rtol, atol=atol, equal_nan=equal_nan
        ).symbolic_call(x1, x2)
    return _allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "allclose"
    ):
        return backend.ops.numpy.allclose(
            x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    return backend.ops.numpy.all(
        backend.ops.numpy.isclose(
            x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    )


class Angle(Operation):
    def call(self, x):
        return _angle(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.angle", "keras.ops.numpy.angle"])
def angle(x):
    """Element-wise angle of a complex tensor.

    Arguments:
        x: Input tensor. Can be real or complex.

    Returns:
        Output tensor of same shape as x. containing the angle of each element
        (in radians).

    Example:
    >>> x = keras.ops.convert_to_tensor([[1 + 3j, 2 - 5j], [4 - 3j, 3 + 2j]])
    >>> keras.ops.angle(x)
    array([[ 1.2490457, -1.19029  ],
       [-0.6435011,  0.5880026]], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Angle().symbolic_call(x)
    return _angle(x)


def _angle(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "angle"
    ):
        return backend.ops.numpy.angle(x)
    x = backend.ops.convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in dtypes.COMPLEX_TYPES:
        x_imag = backend.ops.numpy.imag(x)
        x_real = backend.ops.numpy.real(x)
    else:
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        x_real = backend.ops.cast(x, dtype)
        x_imag = backend.ops.numpy.zeros_like(x_real)
    return backend.ops.numpy.arctan2(x_imag, x_real)


class Any(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.any(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_export(["keras.ops.any", "keras.ops.numpy.any"])
def any(x, axis=None, keepdims=False):
    """Test whether any array element along a given axis evaluates to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical OR reduction is performed. The default
            (`axis=None`) is to perform a logical OR over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Defaults to `False`.

    Returns:
        The tensor containing the logical OR reduction over the `axis`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([True, False])
    >>> keras.ops.any(x)
    array(True, shape=(), dtype=bool)

    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.any(x, axis=0)
    array([ True  True], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return Any(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.any(x, axis=axis, keepdims=keepdims)


class Append(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x1, x2):
        return backend.ops.numpy.append(x1, x2, axis=self.axis)

    def compute_output_spec(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if self.axis is None:
            if None in x1_shape or None in x2_shape:
                output_shape = (None,)
            else:
                output_shape = (
                    python_math.prod(x1_shape) + python_math.prod(x2_shape),
                )
            return KerasTensor(output_shape, dtype=dtype)

        if not shape_equal(x1_shape, x2_shape, [self.axis]):
            raise ValueError(
                "`append` requires inputs to have the same shape except the "
                f"`axis={self.axis}`, but received shape {x1_shape} and "
                f"{x2_shape}."
            )

        output_shape = list(x1_shape)
        output_shape[self.axis] = x1_shape[self.axis] + x2_shape[self.axis]
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.append", "keras.ops.numpy.append"])
def append(
    x1,
    x2,
    axis=None,
):
    """Append tensor `x2` to the end of tensor `x1`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        axis: Axis along which tensor `x2` is appended to tensor `x1`.
            If `None`, both tensors are flattened before use.

    Returns:
        A tensor with the values of `x2` appended to `x1`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([[4, 5, 6], [7, 8, 9]])
    >>> keras.ops.append(x1, x2)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

    When `axis` is specified, `x1` and `x2` must have compatible shapes.
    >>> x1 = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> x2 = keras.ops.convert_to_tensor([[7, 8, 9]])
    >>> keras.ops.append(x1, x2, axis=0)
    array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=int32)
    >>> x3 = keras.ops.convert_to_tensor([7, 8, 9])
    >>> keras.ops.append(x1, x3, axis=0)
    Traceback (most recent call last):
        ...
    TypeError: Cannot concatenate arrays with different numbers of
    dimensions: got (2, 3), (3,).
    """
    if any_symbolic_tensors((x1, x2)):
        return Append(axis=axis).symbolic_call(x1, x2)
    return backend.ops.numpy.append(x1, x2, axis=axis)


class Arange(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, start, stop=None, step=None):
        return backend.ops.numpy.arange(
            start, stop, step=step, dtype=self.dtype
        )

    def compute_output_spec(self, start, stop=None, step=None):
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        output_shape = [int(np.ceil((stop - start) / step))]
        dtype = self.dtype
        if dtype is None:
            dtypes_to_resolve = [getattr(start, "dtype", type(start))]
            if stop is not None:
                dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
            if step is not None:
                dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
            dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.arange", "keras.ops.numpy.arange"])
def arange(start, stop=None, step=None, dtype=None):
    """Return evenly spaced values within a given interval.

    `arange` can be called with a varying number of positional arguments:
    * `arange(stop)`: Values are generated within the half-open interval
        `[0, stop)` (in other words, the interval including start but excluding
        stop).
    * `arange(start, stop)`: Values are generated within the half-open interval
        `[start, stop)`.
    * `arange(start, stop, step)`: Values are generated within the half-open
        interval `[start, stop)`, with spacing between values given by step.

    Args:
        start: Integer or real, representing the start of the interval. The
            interval includes this value.
        stop: Integer or real, representing the end of the interval. The
            interval does not include this value, except in some cases where
            `step` is not an integer and floating point round-off affects the
            length of `out`. Defaults to `None`.
        step: Integer or real, represent the spacing between values. For any
            output `out`, this is the distance between two adjacent values,
            `out[i+1] - out[i]`. The default step size is 1. If `step` is
            specified as a position argument, `start` must also be given.
        dtype: The type of the output array. If `dtype` is not given, infer the
            data type from the other input arguments.

    Returns:
        Tensor of evenly spaced values.
        For floating point arguments, the length of the result is
        `ceil((stop - start)/step)`. Because of floating point overflow, this
        rule may result in the last element of out being greater than stop.

    Examples:
    >>> keras.ops.arange(3)
    array([0, 1, 2], dtype=int32)

    >>> keras.ops.arange(3.0)
    array([0., 1., 2.], dtype=float32)

    >>> keras.ops.arange(3, 7)
    array([3, 4, 5, 6], dtype=int32)

    >>> keras.ops.arange(3, 7, 2)
    array([3, 5], dtype=int32)
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((start, stop, step)):
        return Arange(dtype=dtype).symbolic_call(start, stop, step=step)
    return backend.ops.numpy.arange(start, stop, step=step, dtype=dtype)


class Arccos(Operation):
    def call(self, x):
        return backend.ops.numpy.arccos(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.arccos", "keras.ops.numpy.arccos"])
def arccos(x):
    """Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the angle of the ray intersecting the unit circle at the given
        x-coordinate in radians `[0, pi]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arccos(x)
    array([0.0, 3.1415927], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccos().symbolic_call(x)
    return backend.ops.numpy.arccos(x)


class Arccosh(Operation):
    def call(self, x):
        return backend.ops.numpy.arccosh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.arccosh", "keras.ops.numpy.arccosh"])
def arccosh(x):
    """Inverse hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as x.

    Example:
    >>> x = keras.ops.convert_to_tensor([10, 100])
    >>> keras.ops.arccosh(x)
    array([2.993223, 5.298292], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccosh().symbolic_call(x)
    return backend.ops.numpy.arccosh(x)


class Arcsin(Operation):
    def call(self, x):
        return backend.ops.numpy.arcsin(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arcsin", "keras.ops.numpy.arcsin"])
def arcsin(x):
    """Inverse sine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse sine of each element in `x`, in radians and in
        the closed interval `[-pi/2, pi/2]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1, 0])
    >>> keras.ops.arcsin(x)
    array([ 1.5707964, -1.5707964,  0.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsin().symbolic_call(x)
    return backend.ops.numpy.arcsin(x)


class Arcsinh(Operation):
    def call(self, x):
        return backend.ops.numpy.arcsinh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arcsinh", "keras.ops.numpy.arcsinh"])
def arcsinh(x):
    """Inverse hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1, 0])
    >>> keras.ops.arcsinh(x)
    array([0.88137364, -0.88137364, 0.0], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsinh().symbolic_call(x)
    return backend.ops.numpy.arcsinh(x)


class Arctan(Operation):
    def call(self, x):
        return backend.ops.numpy.arctan(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arctan", "keras.ops.numpy.arctan"])
def arctan(x):
    """Trigonometric inverse tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse tangent of each element in `x`, in the interval
        `[-pi/2, pi/2]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([0, 1])
    >>> keras.ops.arctan(x)
    array([0., 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arctan().symbolic_call(x)
    return backend.ops.numpy.arctan(x)


class Arctan2(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.arctan2(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        outputs_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_dtype = standardize_dtype(getattr(x1, "dtype", backend.floatx()))
        x2_dtype = standardize_dtype(getattr(x2, "dtype", backend.floatx()))
        dtype = dtypes.result_type(x1_dtype, x2_dtype, float)
        return KerasTensor(outputs_shape, dtype=dtype)


@keras_export(["keras.ops.arctan2", "keras.ops.numpy.arctan2"])
def arctan2(x1, x2):
    """Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the
    signed angle in radians between the ray ending at the origin and passing
    through the point `(1, 0)`, and the ray ending at the origin and passing
    through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate"
    is the first function parameter, the "x-coordinate" is the second.) By IEEE
    convention, this function is defined for `x2 = +/-0` and for either or both
    of `x1` and `x2` `= +/-inf`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Tensor of angles in radians, in the range `[-pi, pi]`.

    Examples:
    Consider four points in different quadrants:
    >>> x = keras.ops.convert_to_tensor([-1, +1, +1, -1])
    >>> y = keras.ops.convert_to_tensor([-1, -1, +1, +1])
    >>> keras.ops.arctan2(y, x) * 180 / numpy.pi
    array([-135., -45., 45., 135.], dtype=float32)

    Note the order of the parameters. `arctan2` is defined also when x2=0 and
    at several other points, obtaining values in the range `[-pi, pi]`:
    >>> keras.ops.arctan2(
    ...     keras.ops.array([1., -1.]),
    ...     keras.ops.array([0., 0.]),
    ... )
    array([ 1.5707964, -1.5707964], dtype=float32)
    >>> keras.ops.arctan2(
    ...     keras.ops.array([0., 0., numpy.inf]),
    ...     keras.ops.array([+0., -0., numpy.inf]),
    ... )
    array([0., 3.1415925, 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Arctan2().symbolic_call(x1, x2)
    return backend.ops.numpy.arctan2(x1, x2)


class Arctanh(Operation):
    def call(self, x):
        return backend.ops.numpy.arctanh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arctanh", "keras.ops.numpy.arctanh"])
def arctanh(x):
    """Inverse hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.

    Example:
    >>> x = keras.ops.convert_to_tensor([0, -0.5])
    >>> keras.ops.arctanh(x)
    array([ 0.        , -0.54930615], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arctanh().symbolic_call(x)
    return backend.ops.numpy.arctanh(x)


class Argmax(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.argmax(
            x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype="int32",
        )


@keras_export(["keras.ops.argmax", "keras.ops.numpy.argmax"])
def argmax(x, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras.ops.argmax(x)
    array(5, dtype=int32)
    >>> keras.ops.argmax(x, axis=0)
    array([1, 1, 1], dtype=int32)
    >>> keras.ops.argmax(x, axis=1)
    array([2, 2], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.argmax(x, axis=axis, keepdims=keepdims)


class Argmin(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.argmin(
            x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype="int32",
        )


@keras_export(["keras.ops.argmin", "keras.ops.numpy.argmin"])
def argmin(x, axis=None, keepdims=False):
    """Returns the indices of the minimum values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras.ops.argmin(x)
    array(0, dtype=int32)
    >>> keras.ops.argmin(x, axis=0)
    array([0, 0, 0], dtype=int32)
    >>> keras.ops.argmin(x, axis=1)
    array([0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.argmin(x, axis=axis, keepdims=keepdims)


class Argsort(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.argsort(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                return KerasTensor((None,), dtype="int32")
            return KerasTensor((python_math.prod(x.shape),), dtype="int32")
        canonicalize_axis(self.axis, len(x.shape))
        return KerasTensor(x.shape, dtype="int32")


@keras_export(["keras.ops.argsort", "keras.ops.numpy.argsort"])
def argsort(x, axis=-1):
    """Returns the indices that would sort a tensor.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. Defaults to `-1` (the last axis). If
            `None`, the flattened tensor is used.

    Returns:
        Tensor of indices that sort `x` along the specified `axis`.

    Examples:
    One dimensional array:
    >>> x = keras.ops.array([3, 1, 2])
    >>> keras.ops.argsort(x)
    array([1, 2, 0], dtype=int32)

    Two-dimensional array:
    >>> x = keras.ops.array([[0, 3], [3, 2], [4, 5]])
    >>> x
    array([[0, 3],
           [3, 2],
           [4, 5]], dtype=int32)
    >>> keras.ops.argsort(x, axis=0)
    array([[0, 1],
           [1, 0],
           [2, 2]], dtype=int32)
    >>> keras.ops.argsort(x, axis=1)
    array([[0, 1],
           [1, 0],
           [0, 1]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argsort(axis=axis).symbolic_call(x)
    return backend.ops.numpy.argsort(x, axis=axis)


class Array(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.array(x, dtype=self.dtype)

    def compute_output_spec(self, x, dtype=None):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.array", "keras.ops.numpy.array"])
def array(x, dtype=None):
    """Create a tensor.

    Args:
        x: Input tensor.
        dtype: The desired data-type for the tensor.

    Returns:
        A tensor.

    Examples:
    >>> keras.ops.array([1, 2, 3])
    array([1, 2, 3], dtype=int32)

    >>> keras.ops.array([1, 2, 3], dtype="float32")
    array([1., 2., 3.], dtype=float32)
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return Array(dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.array(x, dtype=dtype)


class View(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.view(x, dtype=self.dtype)

    def compute_output_spec(self, x):
        old_dtype = standardize_dtype(x.dtype)
        new_dtype = standardize_dtype(self.dtype if self.dtype else x.dtype)

        old_itemsize = np.dtype(old_dtype).itemsize
        new_itemsize = np.dtype(new_dtype).itemsize

        if old_itemsize == new_itemsize:
            return KerasTensor(x.shape, dtype=new_dtype)

        if not x.shape:
            raise ValueError(
                "Cannot view a scalar as a different dtype if item sizes "
                "are different."
            )

        output_shape = list(x.shape)
        if output_shape[-1] is not None:
            if (output_shape[-1] * old_itemsize) % new_itemsize != 0:
                raise ValueError(
                    f"Cannot view array of shape {x.shape} and dtype {x.dtype} "
                    f"as dtype {new_dtype} because the total number of bytes "
                    "is not divisible by the new itemsize."
                )
            output_shape[-1] = output_shape[-1] * old_itemsize // new_itemsize
        return KerasTensor(tuple(output_shape), dtype=new_dtype)


@keras_export(["keras.ops.view", "keras.ops.numpy.view"])
def view(x, dtype=None):
    """Create a new bitwise view of the same data with the specified dtype.

    Args:
        x: Input tensor.
        dtype: Data-type descriptor of the returned view,
            e.g., float32 or int16.

    Returns:
        View of a tensor with data type dtype.

    Examples:
    >>> x = keras.ops.array([1, 2, 3])
    >>> x
    array([1, 2, 3], dtype=int32)
    >>> keras.ops.view(x, dtype="float32")
    array([1.0e-45, 3.0e-45, 4.0e-45], dtype=float32)
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return View(dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.view(x, dtype=dtype)


class Average(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x, weights=None):
        return backend.ops.numpy.average(x, weights=weights, axis=self.axis)

    def compute_output_spec(self, x, weights=None):
        dtypes_to_resolve = [getattr(x, "dtype", type(x)), float]
        if weights is not None:
            shape_match = shape_equal(x.shape, weights.shape, allow_none=True)
            if self.axis is not None:
                if isinstance(self.axis, (tuple, list)):
                    if len(self.axis) == 1:
                        shape_match_on_axis = shape_equal(
                            [x.shape[self.axis[0]]],
                            weights.shape,
                            allow_none=True,
                        )
                    else:
                        shape_match_on_axis = False
                else:
                    shape_match_on_axis = shape_equal(
                        [x.shape[self.axis]], weights.shape, allow_none=True
                    )
            dtypes_to_resolve.append(getattr(weights, "dtype", type(weights)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
        if self.axis is None:
            if weights is None or shape_match:
                return KerasTensor([], dtype=dtype)
            else:
                raise ValueError(
                    "`weights` must have the same shape as `x` when "
                    f"`axis=None`, but received `weights.shape={weights.shape}`"
                    f" and `x.shape={x.shape}`."
                )

        if weights is None or shape_match_on_axis or shape_match:
            return KerasTensor(
                reduce_shape(x.shape, axis=self.axis), dtype=dtype
            )
        else:
            # `weights` can either be a 1D array of length `x.shape[axis]` or
            # of the same shape as `x`.
            axis_repr = (
                f"axis={self.axis[0]}"
                if isinstance(self.axis, (tuple, list)) and len(self.axis) == 1
                else f"axis={self.axis}"
            )
            axis_val = (
                self.axis[0]
                if isinstance(self.axis, (tuple, list)) and len(self.axis) == 1
                else self.axis
            )
            try:
                x_shape_at_axis = x.shape[axis_val]
            except TypeError:
                x_shape_at_axis = "multiple axes"

            raise ValueError(
                f"`weights` must have the same size as `x` at {axis_repr} "
                f"but received `weights.shape={weights.shape}` while x.shape "
                f"at {axis_repr} is `{x_shape_at_axis}`."
            )


@keras_export(["keras.ops.average", "keras.ops.numpy.average"])
def average(x, axis=None, weights=None):
    """Compute the weighted average along the specified axis.

    Args:
        x: Input tensor.
        axis: Integer or tuple of integers along which to average `x`.
            The default, `axis=None`, will average over all of the
            elements of the input tensor. If axis is negative it counts
            from the last to the first axis.
        weights: Tensor of weights associated with the values in `x`. Each
            value in `x` contributes to the average according to its
            associated weight. The weights array can either be 1-D (in which
            case its length must be the size of a along the given axis) or of
            the same shape as `x`. If `weights=None` (default), then all data
            in `x` are assumed to have a weight equal to one.

            The 1-D calculation is: `avg = sum(a * weights) / sum(weights)`.
            The only constraint on weights is that `sum(weights)` must not be 0.

    Returns:
        Return the average along the specified axis.

    Examples:
    >>> data = keras.ops.arange(1, 5)
    >>> data
    array([1, 2, 3, 4], dtype=int32)
    >>> keras.ops.average(data)
    array(2.5, dtype=float32)
    >>> keras.ops.average(
    ...     keras.ops.arange(1, 11),
    ...     weights=keras.ops.arange(10, 0, -1)
    ... )
    array(4., dtype=float32)

    >>> data = keras.ops.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]], dtype=int32)
    >>> keras.ops.average(
    ...     data,
    ...     axis=1,
    ...     weights=keras.ops.array([1./4, 3./4])
    ... )
    array([0.75, 2.75, 4.75], dtype=float32)
    >>> keras.ops.average(
    ...     data,
    ...     weights=keras.ops.array([1./4, 3./4])
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Axis must be specified when shapes of a and weights differ.
    """
    if any_symbolic_tensors((x, weights)):
        return Average(axis=axis).symbolic_call(x, weights=weights)
    return backend.ops.numpy.average(x, axis=axis, weights=weights)


@keras_export(["keras.ops.bartlett", "keras.ops.numpy.bartlett"])
def bartlett(x):
    """Bartlett window function.
    The Bartlett window is a triangular window that rises then falls linearly.

    Args:
        x: Scalar or 1D Tensor. Window length.

    Returns:
        A 1D tensor containing the Bartlett window values.

    Example:
    >>> x = keras.ops.convert_to_tensor(5)
    >>> keras.ops.bartlett(x)
    array([0. , 0.5, 1. , 0.5, 0. ], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        raise TypeError(
            f"Bartlett operation does not support symbolic tensors. "
            f"Received input x = {x} of type {type(x)}"
        )
    return backend.ops.numpy.bartlett(x)


@keras_export(["keras.ops.hamming", "keras.ops.numpy.hamming"])
def hamming(x):
    """Hamming window function.

    The Hamming window is defined as:
    `w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))` for `0 <= n <= N - 1`.

    Args:
        x: Scalar or 1D Tensor. The window length.

    Returns:
        A 1D tensor containing the Hamming window values.

    Example:
    >>> x = keras.ops.convert_to_tensor(5)
    >>> keras.ops.hamming(x)
    array([0.08, 0.54, 1.  , 0.54, 0.08], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        raise TypeError(
            f"Hamming operation does not support symbolic tensors. "
            f"Received input x = {x} of type {type(x)}"
        )
    return backend.ops.numpy.hamming(x)


@keras_export(["keras.ops.hanning", "keras.ops.numpy.hanning"])
def hanning(x):
    """Hanning window function.

    The Hanning window is defined as:
    `w[n] = 0.5 - 0.5 * cos(2 * pi * n / (N - 1))` for `0 <= n <= N - 1`.

    Args:
        x: Scalar or 1D Tensor. The window length.

    Returns:
        A 1D tensor containing the Hanning window values.

    Example:
    >>> x = keras.ops.convert_to_tensor(5)
    >>> keras.ops.hanning(x)
    array([0. , 0.5, 1. , 0.5, 0. ], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        raise TypeError(
            f"Hanning operation does not support symbolic tensors. "
            f"Received input x = {x} of type {type(x)}"
        )
    return backend.ops.numpy.hanning(x)


class Heaviside(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.heaviside(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
        if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
            dtype = backend.floatx()
        elif dtype == "int64":
            dtype = "float64"
        return KerasTensor(broadcast_shapes(x1.shape, x2.shape), dtype=dtype)


@keras_export(["keras.ops.heaviside", "keras.ops.numpy.heaviside"])
def heaviside(x1, x2):
    """Heaviside step function.

    The Heaviside step function is defined as:
    `heaviside(x1, x2) = 0 if x1 < 0, 1 if x1 > 0, x2 if x1 == 0`

    Args:
        x1: A tensor input.
        x2: A scalar or tensor, the value to return when `x1 == 0`.

    Returns:
        A tensor with a shape determined by broadcasting `x1` and `x2`.

    Example:
    >>> x1 = keras.ops.convert_to_tensor([-2.0, 0.0, 3.0])
    >>> x2 = 0.5
    >>> keras.ops.heaviside(x1, x2)
    array([0. , 0.5, 1. ], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Heaviside().symbolic_call(x1, x2)
    return backend.ops.numpy.heaviside(x1, x2)


@keras_export(["keras.ops.kaiser", "keras.ops.numpy.kaiser"])
def kaiser(x, beta):
    """Kaiser window function.

    The Kaiser window is defined as:
    `w[n] = I0(beta * sqrt(1 - (2n / (N - 1) - 1)^2)) / I0(beta)`
    where I0 is the modified zeroth-order Bessel function of the first kind.

    Args:
        x: Scalar or 1D Tensor. The window length.
        beta: Float. Shape parameter for the Kaiser window.

    Returns:
        A 1D tensor containing the Kaiser window values.

    Example:
    >>> x = keras.ops.convert_to_tensor(5)
    >>> keras.ops.kaiser(x, beta=14.0)
    array([7.7268669e-06, 1.6493219e-01, 1.0000000e+00, 1.6493219e-01,
       7.7268669e-06], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        raise TypeError(
            f"Kaiser operation does not support symbolic tensors. "
            f"Received input x = {x} of type {type(x)}"
        )
    return backend.ops.numpy.kaiser(x, beta)


class Bincount(Operation):
    def __init__(self, weights=None, minlength=0, sparse=False, *, name=None):
        super().__init__(name=name)
        self.weights = weights
        self.minlength = minlength
        self.sparse = sparse

    def call(self, x):
        return backend.ops.numpy.bincount(
            x,
            weights=self.weights,
            minlength=self.minlength,
            sparse=self.sparse,
        )

    def compute_output_spec(self, x):
        dtypes_to_resolve = [x.dtype]
        if self.weights is not None:
            weights = backend.ops.convert_to_tensor(self.weights)
            dtypes_to_resolve.append(weights.dtype)
            dtype = dtypes.result_type(*dtypes_to_resolve)
        else:
            dtype = "int32"
        x_sparse = getattr(x, "sparse", False)
        return KerasTensor(
            list(x.shape[:-1]) + [None],
            dtype=dtype,
            sparse=x_sparse or self.sparse,
        )


@keras_export(["keras.ops.bincount", "keras.ops.numpy.bincount"])
def bincount(x, weights=None, minlength=0, sparse=False):
    """Count the number of occurrences of each value in a tensor of integers.

    Args:
        x: Input tensor.
            It must be of dimension 1, and it must only contain non-negative
            integer(s).
        weights: Weight tensor.
            It must have the same length as `x`. The default value is `None`.
            If specified, `x` is weighted by it, i.e. if `n = x[i]`,
            `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
        minlength: An integer.
            The default value is 0. If specified, there will be at least
            this number of bins in the output tensor. If greater than
            `max(x) + 1`, each value of the output at an index higher than
            `max(x)` is set to 0.
        sparse: Whether to return a sparse tensor; for backends that support
            sparse tensors.

    Returns:
        1D tensor where each element gives the number of occurrence(s) of its
        index value in x. Its length is the maximum between `max(x) + 1` and
        minlength.

    Examples:
    >>> x = keras.ops.array([1, 2, 2, 3], dtype="uint8")
    >>> keras.ops.bincount(x)
    array([0, 1, 2, 1], dtype=int32)
    >>> weights = x / 2
    >>> weights
    array([0.5, 1., 1., 1.5], dtype=float64)
    >>> keras.ops.bincount(x, weights=weights)
    array([0., 0.5, 2., 1.5], dtype=float64)
    >>> minlength = (keras.ops.max(x).numpy() + 1) + 2 # 6
    >>> keras.ops.bincount(x, minlength=minlength)
    array([0, 1, 2, 1, 0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Bincount(
            weights=weights, minlength=minlength, sparse=sparse
        ).symbolic_call(x)
    return backend.ops.numpy.bincount(
        x, weights=weights, minlength=minlength, sparse=sparse
    )


class BitwiseAnd(Operation):
    def call(self, x, y):
        return backend.ops.numpy.bitwise_and(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(["keras.ops.bitwise_and", "keras.ops.numpy.bitwise_and"])
def bitwise_and(x, y):
    """Compute the bit-wise AND of two arrays element-wise.

    Computes the bit-wise AND of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `&`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseAnd().symbolic_call(x, y)
    return backend.ops.numpy.bitwise_and(x, y)


class BitwiseInvert(Operation):
    def call(self, x):
        return backend.ops.numpy.bitwise_invert(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.bitwise_invert", "keras.ops.numpy.bitwise_invert"])
def bitwise_invert(x):
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `~`.

    Args:
        x: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x,)):
        return BitwiseInvert().symbolic_call(x)
    return backend.ops.numpy.bitwise_invert(x)


class BitwiseNot(Operation):
    def call(self, x):
        return backend.ops.numpy.bitwise_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.bitwise_not", "keras.ops.numpy.bitwise_not"])
def bitwise_not(x):
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `~`.

    Args:
        x: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x,)):
        return BitwiseNot().symbolic_call(x)
    return backend.ops.numpy.bitwise_not(x)


class BitwiseOr(Operation):
    def call(self, x, y):
        return backend.ops.numpy.bitwise_or(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(["keras.ops.bitwise_or", "keras.ops.numpy.bitwise_or"])
def bitwise_or(x, y):
    """Compute the bit-wise OR of two arrays element-wise.

    Computes the bit-wise OR of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `|`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseOr().symbolic_call(x, y)
    return backend.ops.numpy.bitwise_or(x, y)


class BitwiseXor(Operation):
    def call(self, x, y):
        return backend.ops.numpy.bitwise_xor(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(["keras.ops.bitwise_xor", "keras.ops.numpy.bitwise_xor"])
def bitwise_xor(x, y):
    """Compute the bit-wise XOR of two arrays element-wise.

    Computes the bit-wise XOR of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `^`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseXor().symbolic_call(x, y)
    return backend.ops.numpy.bitwise_xor(x, y)


class BitwiseLeftShift(Operation):
    def call(self, x, y):
        return backend.ops.numpy.bitwise_left_shift(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(
    ["keras.ops.bitwise_left_shift", "keras.ops.numpy.bitwise_left_shift"]
)
def bitwise_left_shift(x, y):
    """Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `y` 0s at the right of `x`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x` by `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseLeftShift().symbolic_call(x, y)
    return backend.ops.numpy.bitwise_left_shift(x, y)


class LeftShift(Operation):
    def call(self, x, y):
        return backend.ops.numpy.left_shift(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(["keras.ops.left_shift", "keras.ops.numpy.left_shift"])
def left_shift(x, y):
    """Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `y` 0s at the right of `x`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x` by `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return LeftShift().symbolic_call(x, y)
    return backend.ops.numpy.left_shift(x, y)


class BitwiseRightShift(Operation):
    def call(self, x, y):
        return backend.ops.numpy.bitwise_right_shift(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(
    ["keras.ops.bitwise_right_shift", "keras.ops.numpy.bitwise_right_shift"]
)
def bitwise_right_shift(x, y):
    """Shift the bits of an integer to the right.

    Bits are shifted to the right `y`. Because the internal representation of
    numbers is in binary format, this operation is equivalent to dividing `x` by
    `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseRightShift().symbolic_call(x, y)
    return backend.ops.numpy.bitwise_right_shift(x, y)


class RightShift(Operation):
    def call(self, x, y):
        return backend.ops.numpy.right_shift(x, y)

    def compute_output_spec(self, x, y):
        return _compute_binary_output_spec(x, y)


@keras_export(["keras.ops.right_shift", "keras.ops.numpy.right_shift"])
def right_shift(x, y):
    """Shift the bits of an integer to the right.

    Bits are shifted to the right `y`. Because the internal representation of
    numbers is in binary format, this operation is equivalent to dividing `x` by
    `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return RightShift().symbolic_call(x, y)
    return backend.ops.numpy.right_shift(x, y)


@keras_export(["keras.ops.blackman", "keras.ops.numpy.blackman"])
def blackman(x):
    """Blackman window function.
    The Blackman window is a taper formed by using a weighted cosine.

    Args:
        x: Scalar or 1D tensor. Window length.

    Returns:
        A 1D tensor containing the Blackman window values.

    Example:
    >>> keras.ops.blackman(x)
    array([-1.4901161e-08,  3.4000003e-01,  9.9999994e-01,  3.3999997e-01,
       -1.4901161e-08], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        raise TypeError(
            f"Blackman operation does not support symbolic tensors. "
            f"Received input x = {x} of type {type(x)}"
        )
    return _blackman(x)


def _blackman(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "blackman"
    ):
        return backend.ops.numpy.blackman(x)
    dtype = config.floatx()
    compute_dtype = dtypes.result_type(dtype, "float32")
    x = backend.ops.cast(x, "int32")
    n = backend.ops.numpy.arange(x, dtype=compute_dtype)
    x_minus_1 = backend.ops.cast(x - 1, compute_dtype)
    phase = (
        python_math.pi
        * (2 * n - x_minus_1)
        / backend.ops.numpy.maximum(x_minus_1, 1.0)
    )
    window = (
        0.42
        + 0.5 * backend.ops.numpy.cos(phase)
        + 0.08 * backend.ops.numpy.cos(2 * phase)
    )
    return backend.ops.cast(window, dtype)


class BroadcastTo(Operation):
    def __init__(self, shape, *, name=None):
        super().__init__(name=name)
        self.shape = shape

    def call(self, x):
        return backend.ops.numpy.broadcast_to(x, self.shape)

    def compute_output_spec(self, x):
        # Catch broadcasting errors for clear error messages.
        broadcast_shapes(x.shape, self.shape)
        return KerasTensor(self.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.broadcast_to",
        "keras.ops.numpy.broadcast_to",
    ]
)
def broadcast_to(x, shape):
    """Broadcast a tensor to a new shape.

    Args:
        x: The tensor to broadcast.
        shape: The shape of the desired tensor. A single integer `i` is
            interpreted as `(i,)`.

    Returns:
        A tensor with the desired shape.

    Examples:
    >>> x = keras.ops.array([1, 2, 3])
    >>> keras.ops.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    if any_symbolic_tensors((x,)):
        return BroadcastTo(shape=shape).symbolic_call(x)
    return backend.ops.numpy.broadcast_to(x, shape)


class Cbrt(Operation):
    def call(self, x):
        return _cbrt(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if dtype in [
            "bool",
            "int8",
            "int16",
            "int32",
            "uint8",
            "uint16",
            "uint32",
        ]:
            dtype = backend.floatx()
        elif dtype == "int64":
            dtype = "float64"

        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.cbrt", "keras.ops.numpy.cbrt"])
def cbrt(x):
    """Computes the cube root of the input tensor, element-wise.

    This operation returns the real-valued cube root of `x`, handling
    negative numbers properly in the real domain.

    Args:
        x: Input tensor.

    Returns:
        A tensor containing the cube root of each element in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Cbrt().symbolic_call(x)
    return _cbrt(x)


def _cbrt(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "cbrt"
    ):
        return backend.ops.numpy.cbrt(x)
    x = backend.ops.convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"):
        dtype = backend.floatx()
    elif dtype == "int64":
        dtype = "float64"
    x = backend.ops.cast(x, dtype)
    y = backend.ops.numpy.sign(x) * backend.ops.numpy.power(
        backend.ops.numpy.absolute(x), 1.0 / 3.0
    )
    return backend.ops.cast(y, dtype)


class Ceil(Operation):
    def call(self, x):
        return backend.ops.numpy.ceil(x)

    def compute_output_spec(self, x):
        if standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.ceil", "keras.ops.numpy.ceil"])
def ceil(x):
    """Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.

    Args:
        x: Input tensor.

    Returns:
        The ceiling of each element in `x`, with float dtype.
    """
    if any_symbolic_tensors((x,)):
        return Ceil().symbolic_call(x)
    return backend.ops.numpy.ceil(x)


class Clip(Operation):
    def __init__(self, x_min, x_max, *, name=None):
        super().__init__(name=name)
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return backend.ops.numpy.clip(x, self.x_min, self.x_max)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.clip", "keras.ops.numpy.clip"])
def clip(x, x_min, x_max):
    """Clip (limit) the values in a tensor.

    Given an interval, values outside the interval are clipped to the
    interval edges. For example, if an interval of `[0, 1]` is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.

    Args:
        x: Input tensor.
        x_min: Minimum value.
        x_max: Maximum value.
    Returns:
        The clipped tensor.
    """
    if any_symbolic_tensors((x,)):
        return Clip(x_min, x_max).symbolic_call(x)
    return backend.ops.numpy.clip(x, x_min, x_max)


class Concatenate(Operation):
    def __init__(self, axis=0, *, name=None):
        super().__init__(name=name)
        if axis is None:
            raise ValueError("`axis` cannot be None for `concatenate`.")
        self.axis = axis

    def call(self, xs):
        return backend.ops.numpy.concatenate(xs, axis=self.axis)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        axis = canonicalize_axis(self.axis, len(first_shape))
        total_size_on_axis = 0
        all_sparse = True
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(
                x.shape, first_shape, axis=[axis], allow_none=True
            ):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[axis] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[axis]
            all_sparse = all_sparse and getattr(x, "sparse", False)
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[axis] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype, sparse=all_sparse)


@keras_export(
    [
        "keras.ops.concatenate",
        "keras.ops.numpy.concatenate",
    ]
)
def concatenate(xs, axis=0):
    """Join a sequence of tensors along an existing axis.

    Args:
        xs: The sequence of tensors to concatenate.
        axis: The axis along which the tensors will be joined. Defaults to `0`.

    Returns:
        The concatenated tensor.
    """
    if any_symbolic_tensors(xs):
        return Concatenate(axis=axis).symbolic_call(xs)
    return backend.ops.numpy.concatenate(xs, axis=axis)


class Conjugate(Operation):
    def call(self, x):
        return backend.ops.numpy.conjugate(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.conjugate", "keras.ops.numpy.conjugate"])
def conjugate(x):
    """Returns the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the sign
    of its imaginary part.

    `keras.ops.conj` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        The complex conjugate of each element in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Conjugate().symbolic_call(x)
    return backend.ops.numpy.conjugate(x)


class Conj(Conjugate):
    pass


@keras_export(["keras.ops.conj", "keras.ops.numpy.conj"])
def conj(x):
    """Shorthand for `keras.ops.conjugate`."""
    return conjugate(x)


class Copy(Operation):
    def call(self, x):
        return backend.ops.numpy.copy(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.copy", "keras.ops.numpy.copy"])
def copy(x):
    """Returns a copy of `x`.

    Args:
        x: Input tensor.

    Returns:
        A copy of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Copy().symbolic_call(x)
    return backend.ops.numpy.copy(x)


class Copysign(Operation):
    def call(self, x1, x2):
        return _copysign(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.copysign", "keras.ops.numpy.copysign"])
def copysign(x1, x2):
    """Compose a value from the magnitude of `x1` and the sign of `x2`.

    The sign of zero is taken into account, so an `x2` of `-0.0` gives a
    negative result and an `x2` of `0.0` gives a positive one.

    Args:
        x1: Input tensor providing the magnitude.
        x2: Input tensor providing the sign.

    Returns:
        Output tensor with the magnitude of `x1` and the sign of `x2`.

    Example:
    >>> x1 = keras.ops.convert_to_tensor([-1.0, 2.0, -3.0])
    >>> x2 = keras.ops.convert_to_tensor([1.0, -1.0, -0.0])
    >>> keras.ops.copysign(x1, x2)
    array([ 1., -2., -3.], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Copysign().symbolic_call(x1, x2)
    return _copysign(x1, x2)


def _copysign(x1, x2):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "copysign"
    ):
        return backend.ops.numpy.copysign(x1, x2)
    x1 = backend.ops.convert_to_tensor(x1)
    x2 = backend.ops.convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = backend.ops.cast(x1, dtype)
    x2 = backend.ops.cast(x2, dtype)
    # Negate `x1` exactly when its sign differs from the sign of `x2`.
    flip = backend.ops.numpy.logical_xor(
        backend.ops.numpy.signbit(x1), backend.ops.numpy.signbit(x2)
    )
    return backend.ops.numpy.where(flip, backend.ops.numpy.negative(x1), x1)


class Cos(Operation):
    def call(self, x):
        return backend.ops.numpy.cos(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.cos", "keras.ops.numpy.cos"])
def cos(x):
    """Cosine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        The corresponding cosine values.
    """
    if any_symbolic_tensors((x,)):
        return Cos().symbolic_call(x)
    return backend.ops.numpy.cos(x)


class Cosh(Operation):
    def call(self, x):
        return backend.ops.numpy.cosh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.cosh", "keras.ops.numpy.cosh"])
def cosh(x):
    """Hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Cosh().symbolic_call(x)
    return backend.ops.numpy.cosh(x)


class CountNonzero(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = axis

    def call(self, x):
        return backend.ops.numpy.count_nonzero(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis),
            dtype="int32",
        )


@keras_export(
    [
        "keras.ops.count_nonzero",
        "keras.ops.numpy.count_nonzero",
    ]
)
def count_nonzero(x, axis=None):
    """Counts the number of non-zero values in `x` along the given `axis`.

    If no axis is specified then all non-zeros in the tensor are counted.

    Args:
        x: Input tensor.
        axis: Axis or tuple of axes along which to count the number of
            non-zeros. Defaults to `None`.

    Returns:
        int or tensor of ints.

    Examples:
    >>> x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
    >>> keras.ops.count_nonzero(x)
    5
    >>> keras.ops.count_nonzero(x, axis=0)
    array([1, 1, 2, 1], dtype=int64)
    >>> keras.ops.count_nonzero(x, axis=1)
    array([2, 3], dtype=int64)
    """
    if any_symbolic_tensors((x,)):
        return CountNonzero(axis=axis).symbolic_call(x)
    return backend.ops.numpy.count_nonzero(x, axis=axis)


class Cross(Operation):
    def __init__(self, axisa=-1, axisb=-1, axisc=-1, axis=None, *, name=None):
        super().__init__(name=name)
        if axis is not None:
            self.axisa = axis
            self.axisb = axis
            self.axisc = axis
        else:
            self.axisa = axisa
            self.axisb = axisb
            self.axisc = axisc

    def call(self, x1, x2):
        return backend.ops.numpy.cross(
            x1, x2, self.axisa, self.axisb, self.axisc
        )

    def compute_output_spec(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)

        x1_value_size = x1_shape[self.axisa]
        x2_value_size = x2_shape[self.axisb]
        del x1_shape[self.axisa]
        del x2_shape[self.axisb]
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        if x1_value_size is not None and x1_value_size not in (2, 3):
            raise ValueError(
                "`x1`'s dim on `axis={axisa}` must be either 2 or 3, but "
                f"received: {x1_value_size}"
            )
        if x2_value_size is not None and x2_value_size not in (2, 3):
            raise ValueError(
                "`x2`'s dim on `axis={axisb}` must be either 2 or 3, but "
                f"received: {x2_value_size}"
            )

        if x1_value_size == 3 or x2_value_size == 3:
            value_size = [3]
        else:
            value_size = []

        axisc = canonicalize_axis(
            self.axisc, len(output_shape) + len(value_size)
        )
        output_shape = output_shape[:axisc] + value_size + output_shape[axisc:]

        dtype = dtypes.result_type(x1.dtype, x2.dtype)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.cross", "keras.ops.numpy.cross"])
def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Returns the cross product of two (arrays of) vectors.

    The cross product of `x1` and `x2` in R^3 is a vector
    perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of
    vectors, the vectors are defined by the last axis of `x1` and `x2`
    by default, and these axes can have dimensions 2 or 3.

    Where the dimension of either `x1` or `x2` is 2, the third component of
    the input vector is assumed to be zero and the cross product calculated
    accordingly.

    In cases where both input vectors have dimension 2, the z-component of
    the cross product is returned.

    Args:
        x1: Components of the first vector(s).
        x2: Components of the second vector(s).
        axisa: Axis of `x1` that defines the vector(s). Defaults to `-1`.
        axisb: Axis of `x2` that defines the vector(s). Defaults to `-1`.
        axisc: Axis of the result containing the cross product vector(s).
            Ignored if both input vectors have dimension 2, as the return is
            scalar. By default, the last axis.
        axis: If defined, the axis of `x1`, `x2` and the result that
            defines the vector(s) and cross product(s). Overrides `axisa`,
            `axisb` and `axisc`.

    Note:
        Torch backend does not support two dimensional vectors, or the
        arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.

    Returns:
        Vector cross product(s).
    """
    if any_symbolic_tensors((x1, x2)):
        return Cross(
            axisa=axisa, axisb=axisb, axisc=axisc, axis=axis
        ).symbolic_call(x1, x2)
    return backend.ops.numpy.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


class Cumprod(Operation):
    def __init__(self, axis=None, dtype=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.cumprod(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (python_math.prod(x.shape),)
        else:
            canonicalize_axis(self.axis, len(x.shape))
            output_shape = x.shape
        output_dtype = (
            standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        )
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.cumprod", "keras.ops.numpy.cumprod"])
def cumprod(x, axis=None, dtype=None):
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative product is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return Cumprod(axis=axis, dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.cumprod(x, axis=axis, dtype=dtype)


class Cumsum(Operation):
    def __init__(self, axis=None, dtype=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.cumsum(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (python_math.prod(x.shape),)
        else:
            canonicalize_axis(self.axis, len(x.shape))
            output_shape = x.shape
        output_dtype = (
            standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        )
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.cumsum", "keras.ops.numpy.cumsum"])
def cumsum(x, axis=None, dtype=None):
    """Returns the cumulative sum of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return Cumsum(axis=axis, dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.cumsum(x, axis=axis, dtype=dtype)


class Deg2rad(Operation):
    def call(self, x):
        return _deg2rad(x)

    def compute_output_spec(self, x):
        if standardize_dtype(x.dtype) == "int64":
            dtype = config.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(x.shape, dtype)


@keras_export(["keras.ops.deg2rad", "keras.ops.numpy.deg2rad"])
def deg2rad(x):
    """Convert angles from degrees to radians.

    The conversion is defined as:
    `rad = deg * (π / 180)`

    Args:
        x: Input tensor of angles in degrees.

    Returns:
        A tensor containing angles converted to radians.

    Examples:
    >>> from keras import ops
    >>> ops.deg2rad(180.0)
    3.141592653589793
    >>> ops.deg2rad([0.0, 90.0, 180.0])
    array([0., 1.57079633, 3.14159265])
    """
    if any_symbolic_tensors((x,)):
        return Deg2rad().symbolic_call(x)
    return _deg2rad(x)


def _deg2rad(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "deg2rad"
    ):
        return backend.ops.numpy.deg2rad(x)
    x = backend.ops.convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = backend.ops.cast(x, dtype)
    return backend.ops.numpy.multiply(x, python_math.pi / 180.0)


class Rad2deg(Operation):
    def call(self, x):
        return _rad2deg(x)

    def compute_output_spec(self, x):
        if standardize_dtype(x.dtype) == "int64":
            dtype = config.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(x.shape, dtype)


@keras_export(["keras.ops.rad2deg", "keras.ops.numpy.rad2deg"])
def rad2deg(x):
    """Convert angles from radians to degrees.

    The conversion is defined as:
    `deg = rad * (180 / π)`

    Args:
        x: Input tensor of angles in radians.

    Returns:
        A tensor containing angles converted to degrees.

    Examples:
    >>> from keras import ops
    >>> ops.rad2deg(3.141592653589793)
    180.0
    >>> ops.rad2deg([0.0, 1.57079633, 3.14159265])
    array([0., 90., 180.])
    """
    if any_symbolic_tensors((x,)):
        return Rad2deg().symbolic_call(x)
    return _rad2deg(x)


def _rad2deg(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "rad2deg"
    ):
        return backend.ops.numpy.rad2deg(x)
    x = backend.ops.convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = backend.ops.cast(x, dtype)
    return backend.ops.numpy.multiply(x, 180.0 / python_math.pi)


class Diag(Operation):
    def __init__(self, k=0, *, name=None):
        super().__init__(name=name)
        self.k = k

    def call(self, x):
        return backend.ops.numpy.diag(x, k=self.k)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if len(x_shape) == 1:
            if x_shape[0] is None:
                output_shape = [None, None]
            else:
                output_shape = [
                    x_shape[0] + int(np.abs(self.k)),
                    x_shape[0] + int(np.abs(self.k)),
                ]
        elif len(x_shape) == 2:
            if None in x_shape:
                output_shape = [None]
            else:
                shorter_side = np.minimum(x_shape[0], x_shape[1])
                if self.k > 0:
                    remaining = x_shape[1] - self.k
                else:
                    remaining = x_shape[0] + self.k
                output_shape = [
                    int(np.maximum(0, np.minimum(remaining, shorter_side)))
                ]
        else:
            raise ValueError(
                f"`x` must be 1-D or 2-D, but received shape {x.shape}."
            )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.diag", "keras.ops.numpy.diag"])
def diag(x, k=0):
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
            If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
        k: The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals
            above the main diagonal, and `k < 0` for diagonals below
            the main diagonal.

    Returns:
        The extracted diagonal or constructed diagonal tensor.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> ops.diag(x)
    array([0, 4, 8])
    >>> ops.diag(x, k=1)
    array([1, 5])
    >>> ops.diag(x, k=-1)
    array([3, 7])

    >>> ops.diag(ops.diag(x)))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    if any_symbolic_tensors((x,)):
        return Diag(k=k).symbolic_call(x)
    return backend.ops.numpy.diag(x, k=k)


class Diagflat(Operation):
    def __init__(self, k=0, *, name=None):
        super().__init__(name=name)
        self.k = k

    def call(self, x):
        return backend.ops.numpy.diagflat(x, k=self.k)

    def compute_output_spec(self, x):
        x_shape = x.shape

        if len(x_shape) == 0:
            flat_size = 1
        elif len(x_shape) == 1:
            flat_size = x_shape[0] if x_shape[0] is not None else None
        else:
            flat_size = None
            for s in x_shape:
                if s is None:
                    flat_size = None
                    break
                elif flat_size is None:
                    flat_size = s
                else:
                    flat_size *= s

        if flat_size is None:
            output_shape = [None, None]
        else:
            output_shape = [
                flat_size + int(np.abs(self.k)),
                flat_size + int(np.abs(self.k)),
            ]

        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.diagflat", "keras.ops.numpy.diagflat"])
def diagflat(x, k=0):
    """Create a two-dimensional array with the flattened input on
       the k-th diagonal.

    Args:
        x: Input tensor to be flattened and placed on the diagonal.
        k: The diagonal to place the flattened input. Defaults to `0`.
           Use `k > 0` for diagonals above the main diagonal,
           and `k < 0` for diagonals below the main diagonal.

    Returns:
        A 2-D tensor with the flattened input on the specified diagonal.
    """
    if any_symbolic_tensors((x,)):
        return Diagflat(k=k).symbolic_call(x)
    return backend.ops.numpy.diagflat(x, k=k)


class Diagonal(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1, *, name=None):
        super().__init__(name=name)
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.ops.numpy.diagonal(
            x,
            offset=self.offset,
            axis1=self.axis1,
            axis2=self.axis2,
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if len(x_shape) < 2:
            raise ValueError(
                "`diagonal` requires an array of at least two dimensions, but "
                f"`x` is of shape {x.shape}."
            )

        shape_2d = [x_shape[self.axis1], x_shape[self.axis2]]
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = [d for d in x_shape if d != -1]
        if None in shape_2d:
            diag_shape = [None]
        else:
            shorter_side = np.minimum(shape_2d[0], shape_2d[1])
            if self.offset > 0:
                remaining = shape_2d[1] - self.offset
            else:
                remaining = shape_2d[0] + self.offset
            diag_shape = [
                int(np.maximum(0, np.minimum(remaining, shorter_side)))
            ]
        output_shape = output_shape + diag_shape
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.diagonal", "keras.ops.numpy.diagonal"])
def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
    collection of elements of the form `x[i, i+offset]`.

    If `x` has more than two dimensions, the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal
    is returned.

    The shape of the resulting array can be determined by removing `axis1`
    and `axis2` and appending an index to the right equal to the size of
    the resulting diagonals.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to `0`.(main diagonal).
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to `0`.(first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to `1` (second axis).

    Returns:
        Tensor of diagonals.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.arange(4).reshape((2, 2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.diagonal()
    array([0, 3])
    >>> x.diagonal(1)
    array([1])

    >>> x = ops.arange(8).reshape((2, 2, 2))
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> x.diagonal(0, 0, 1)
    array([[0, 6],
           [1, 7]])
    """
    x_ndim = len(x.shape)
    ax1 = canonicalize_axis(axis1, x_ndim)
    ax2 = canonicalize_axis(axis2, x_ndim)
    if ax1 == ax2:
        raise ValueError(
            "`axis1` and `axis2` cannot be the same. "
            f"Received: axis1={axis1}, axis2={axis2}"
        )
    if any_symbolic_tensors((x,)):
        return Diagonal(
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        ).symbolic_call(x)
    return backend.ops.numpy.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


class Diff(Operation):
    def __init__(self, n=1, axis=-1, *, name=None):
        super().__init__(name=name)
        self.n = n
        self.axis = axis

    def call(self, a):
        return backend.ops.numpy.diff(a, n=self.n, axis=self.axis)

    def compute_output_spec(self, a):
        axis = canonicalize_axis(self.axis, len(a.shape))
        shape = list(a.shape)
        size = shape[axis]
        if size is not None:
            shape[axis] = builtins.max(size - self.n, 0)
        return KerasTensor(shape, dtype=a.dtype)


@keras_export(["keras.ops.diff", "keras.ops.numpy.diff"])
def diff(a, n=1, axis=-1):
    """Calculate the n-th discrete difference along the given axis.

    The first difference is given by `out[i] = a[i+1] - a[i]` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Args:
        a: Input tensor.
        n: The number of times values are differenced. Defaults to `1`.
        axis: Axis to compute discrete difference(s) along.
            Defaults to `-1`.(last axis).

    Returns:
        Tensor of diagonals.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.convert_to_tensor([1, 2, 4, 7, 0])
    >>> ops.diff(x)
    array([ 1,  2,  3, -7])
    >>> ops.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = ops.convert_to_tensor([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> ops.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> ops.diff(x, axis=0)
    array([[-1,  2,  0, -2]])
    """
    return Diff(n=n, axis=axis)(a)


class Digitize(Operation):
    def call(self, x, bins):
        return backend.ops.numpy.digitize(x, bins)

    def compute_output_spec(self, x, bins):
        bins_shape = bins.shape
        if len(bins_shape) > 1:
            raise ValueError(
                f"`bins` must be a 1D array. Received: bins={bins} "
                f"with shape bins.shape={bins_shape}"
            )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype="int32", sparse=sparse)


@keras_export(["keras.ops.digitize", "keras.ops.numpy.digitize"])
def digitize(x, bins):
    """Returns the indices of the bins to which each value in `x` belongs.

    Args:
        x: Input array to be binned.
        bins: Array of bins. It has to be one-dimensional and monotonically
            increasing.

    Returns:
        Output array of indices, of same shape as `x`.

    Example:
    >>> x = np.array([0.0, 1.0, 3.0, 1.6])
    >>> bins = np.array([0.0, 3.0, 4.5, 7.0])
    >>> keras.ops.digitize(x, bins)
    array([1, 1, 2, 1])
    """
    if any_symbolic_tensors((x, bins)):
        return Digitize().symbolic_call(x, bins)
    return backend.ops.numpy.digitize(x, bins)


class Dot(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.dot(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if x1_shape == [] or x2_shape == []:
            return multiply(x1, x2)
        if len(x1_shape) == 1 and len(x2_shape) == 1:
            return KerasTensor([], dtype=dtype)
        if len(x2_shape) == 1:
            if x1_shape[-1] != x2_shape[0]:
                raise ValueError(
                    "Shape must match on the last axis of `x1` and `x2` when "
                    "`x1` is N-d array while `x2` is 1-D, but receive shape "
                    f"`x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
                )
            return KerasTensor(x1_shape[:-1], dtype=dtype)

        if (
            x1_shape[-1] is None
            or x2_shape[-2] is None
            or x1_shape[-1] == x2_shape[-2]
        ):
            del x1_shape[-1]
            del x2_shape[-2]
            return KerasTensor(x1_shape + x2_shape, dtype=dtype)

        raise ValueError(
            "Shape must match on the last axis of `x1` and second last "
            "axis of `x2` when `x1` is N-d array while `x2` is M-D, but "
            f"received `x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
        )


@keras_export(["keras.ops.dot", "keras.ops.numpy.dot"])
def dot(x1, x2):
    """Dot product of two tensors.

    - If both `x1` and `x2` are 1-D tensors, it is inner product of vectors
      (without complex conjugation).
    - If both `x1` and `x2` are 2-D tensors, it is matrix multiplication.
    - If either `x1` or `x2` is 0-D (scalar), it is equivalent to `x1 * x2`.
    - If `x1` is an N-D tensor and `x2` is a 1-D tensor, it is a sum product
      over the last axis of `x1` and `x2`.
    - If `x1` is an N-D tensor and `x2` is an M-D tensor (where `M>=2`),
      it is a sum product over the last axis of `x1` and the second-to-last
      axis of `x2`: `dot(x1, x2)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`.

    Args:
        x1: First argument.
        x2: Second argument.

    Note:
        Torch backend does not accept 0-D tensors as arguments.

    Returns:
        Dot product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Dot().symbolic_call(x1, x2)
    return backend.ops.numpy.dot(x1, x2)


class Dstack(Operation):
    def call(self, xs):
        return backend.ops.numpy.dstack(xs)

    def compute_output_spec(self, xs):
        dtypes_to_resolve = []
        out_shapes = []
        for x in xs:
            shape = list(x.shape)
            if len(shape) == 0:
                shape = [1, 1, 1]
            elif len(shape) == 1:
                shape = [1, shape[0], 1]
            elif len(shape) == 2:
                shape = shape + [1]
            out_shapes.append(shape)
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))

        first_shape = out_shapes[0]
        total_depth = 0
        for shape in out_shapes:
            if not shape_equal(shape, first_shape, axis=[2], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_depth is None or shape[2] is None:
                total_depth = None
            else:
                total_depth += shape[2]

        output_shape = list(first_shape)
        output_shape[2] = total_depth
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.dstack", "keras.ops.numpy.dstack"])
def dstack(xs):
    """Stack tensors in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D tensors
    of shape `(M, N)` have been reshaped to `(M, N, 1)` and 1-D tensors of shape
    `(N,)` have been reshaped to `(1, N, 1)`.

    Args:
        xs: Sequence of tensors.

    Returns:
        The tensor formed by stacking the given tensors.

    Examples:
    >>> import keras
    >>> x = keras.ops.array([1, 2, 3])
    >>> y = keras.ops.array([4, 5, 6])
    >>> keras.ops.dstack([x, y])
    array([[[1, 4],
            [2, 5],
            [3, 6]]])

    >>> x = keras.ops.array([[1], [2], [3]])
    >>> y = keras.ops.array([[4], [5], [6]])
    >>> keras.ops.dstack([x, y])
    array([[[1, 4]],
           [[2, 5]],
           [[3, 6]]])
    """
    if any_symbolic_tensors((xs,)):
        return Dstack().symbolic_call(xs)
    return backend.ops.numpy.dstack(xs)


class Einsum(Operation):
    def __init__(self, subscripts, *, name=None):
        super().__init__(name=name)
        self.subscripts = subscripts

    def call(self, *operands, **kwargs):
        return backend.ops.numpy.einsum(self.subscripts, *operands, **kwargs)

    def compute_output_spec(self, *operands):
        """Compute the output shape of `einsum`.

        The shape computation follows the steps below:
        1. Find all letters in the input specs (left part of "->"), and
            break them into two categories: letters appearing more than once
            go to `reduced_dims`, otherwise go to `kept_dims`.
        2. Adjust `reduced_dims` and `kept_dims` based on the output spec
            (right part of "->"). The rule is if the letter appears in the
            output spec, then move it to `kept_dims`, otherwise move it to
            `reduced_dims`.
        3. Compute the target output shape. If no output spec is set, then
            the target output shape will be "...{kept_dims}", e.g., "...ijk",
            else it will be the same as output spec. "..." is a wildcard that
            could map shape of arbitrary length.
        4. For each operand in `operands`, map the shape specified in the input
            spec to the output target, e.g, if operand is of shape [2,3,4],
            input spec is "i..." and output target is "i...jk", then 2 will go
            the index 0. For dims not represented by any letter, insert to the
            wildcard part. For each letter in output target not appearing in
            input spec, the dim will be 1 for broadcasting. After 4, each
            operand should have a target shape containing only number and
            `None`.
        5. Broadcast all shapes computed from 4, and the result is the output
            shape.

        Let's take an example to illustrate the steps above. Let's define:
        ```python
        x = KerasTensor([None, 3, 4])
        y = KerasTensor(2, 4, 3)
        z = knp.einsum("...ij, kji->...k", x, y)
        ```

        1. `reduced_dims` is {"i", "j"}, `kept_dims` is {"k"}.
        2. `reduced_dims` is still {"i", "j"}, and `kept_dims` is {"k"}.
        3. Output target is "...k".
        4. For `x`, the input spec is "...ij", and the output target is "...k".
            "i" and "j" do not appear in the output target, so no replacement
            happens, and [None] goes to wildcard. Afterwards, "k" is replaced
            by 1, so we get shape [None, 1]. Applying the same logic to `y`, we
            get shape [2].
        5. Broadcast [None, 1] and [2], and we get [None, 2], which is the
            output shape.
        """
        split_subscripts = self.subscripts.split("->")
        if len(split_subscripts) > 2:
            raise ValueError(
                "At most one '->' is supported in `einsum` subscripts, but "
                f"received {self.subscripts}."
            )
        if len(split_subscripts) == 2:
            subscripts = split_subscripts[0]
            output_spec = split_subscripts[1]
        else:
            subscripts = self.subscripts
            output_spec = None
        input_specs = subscripts.split(",")
        if len(input_specs) != len(operands):
            raise ValueError(
                f"Number of operands ({len(operands)}) does not match the "
                f"number of input specs ({len(input_specs)}) in `einsum`, "
                f"received subscripts={self.subscripts}."
            )
        reduced_dims = set()
        kept_dims = set()
        for s in subscripts:
            if not s.isalpha():
                continue
            if s not in reduced_dims and s not in kept_dims:
                kept_dims.add(s)
            elif s in kept_dims:
                kept_dims.remove(s)
                reduced_dims.add(s)

        if output_spec is not None:
            # The output spec changes the rule of kept_dims and reduced_dims.
            # In short, dims appearing in the output spec will be kept, and
            # dims not appearing in the output spec will be reduced.
            kept_dims_copy = kept_dims.copy()
            reduced_dims_copy = reduced_dims.copy()
            for dim in kept_dims:
                if dim not in output_spec:
                    kept_dims_copy.remove(dim)
                    reduced_dims_copy.add(dim)
            for dim in reduced_dims:
                if dim in output_spec:
                    reduced_dims_copy.remove(dim)
                    kept_dims_copy.add(dim)
            kept_dims = kept_dims_copy
            reduced_dims = reduced_dims_copy

        reduced_dims = sorted(reduced_dims)
        kept_dims = sorted(kept_dims)

        if output_spec is None:
            target_broadcast_spec = f"...{''.join(kept_dims)}"
        else:
            target_broadcast_spec = output_spec

        expanded_operands_shapes = []
        for x, spec in zip(operands, input_specs):
            x_shape = getattr(x, "shape", [])
            x_shape = [-1 if size is None else size for size in x_shape]
            split_spec = spec.split("...")
            expanded_shape = target_broadcast_spec
            if len(split_spec) == 1:
                # In this case, the input spec is just a string of letters,
                # e.g., "ijk".
                if len(x_shape) != len(split_spec[0]):
                    raise ValueError(
                        "Number of dimensions in the subscript does not "
                        "match the number of dimensions in the operand, "
                        f"received subscript `{spec}` and operand of shape "
                        f"{x_shape}."
                    )
                for size, s in zip(x_shape, split_spec[0]):
                    # Replace the letter with the right shape.
                    expanded_shape = expanded_shape.replace(s, f"{str(size)} ")
                expanded_shape = expanded_shape.replace("...", "")
            else:
                # In this case, the input spec has "...", e.g., "i...j", "i...",
                # or "...j".
                for i in range(len(split_spec[0])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[0][i], f"{x_shape[i]} "
                    )
                for i in range(len(split_spec[1])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[1][-i - 1], f"{x_shape[-i - 1]} "
                    )
                # Shape matched by "..." will be inserted to the position of
                # "...".
                wildcard_shape_start_index = len(split_spec[0])
                wildcard_shape_end_index = (
                    len(x_shape)
                    if len(split_spec[1]) == 0
                    else -len(split_spec[1])
                )
                wildcard_shape = x_shape[
                    wildcard_shape_start_index:wildcard_shape_end_index
                ]
                wildcard_shape_str = (
                    f"{' '.join([str(size) for size in wildcard_shape])} "
                )
                expanded_shape = expanded_shape.replace(
                    "...", wildcard_shape_str
                )
            # Replace all letters not yet handled with "1" for broadcasting.
            expanded_shape = re.sub("[a-z]", "1 ", expanded_shape)
            expanded_shape = expanded_shape.split()
            expanded_shape = [
                None if size == "-1" else int(size) for size in expanded_shape
            ]
            expanded_operands_shapes.append(expanded_shape)

        output_shape = expanded_operands_shapes[0]
        for shape in expanded_operands_shapes[1:]:
            output_shape = broadcast_shapes(output_shape, shape)
        dtypes_to_resolve = list(
            set(
                standardize_dtype(getattr(x, "dtype", type(x)))
                for x in operands
            )
        )
        if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
            dtype = "int32"
        else:
            dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.einsum", "keras.ops.numpy.einsum"])
def einsum(subscripts, *operands, **kwargs):
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts: Specifies the subscripts for summation as comma separated
            list of subscript labels. An implicit (classical Einstein
            summation) calculation is performed unless the explicit indicator
            `->` is included as well as subscript labels of the precise
            output form.
        operands: The operands to compute the Einstein sum of.

    Returns:
        The calculation based on the Einstein summation convention.

    Example:
    >>> from keras.src import ops
    >>> a = ops.arange(25).reshape(5, 5)
    >>> b = ops.arange(5)
    >>> c = ops.arange(6).reshape(2, 3)

    Trace of a matrix:

    >>> ops.einsum("ii", a)
    60
    >>> ops.einsum(a, [0, 0])
    60
    >>> ops.trace(a)
    60

    Extract the diagonal:

    >>> ops.einsum("ii -> i", a)
    array([ 0,  6, 12, 18, 24])
    >>> ops.einsum(a, [0, 0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> ops.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis:

    >>> ops.einsum("ij -> i", a)
    array([ 10,  35,  60,  85, 110])
    >>> ops.einsum(a, [0, 1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> ops.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional tensors summing a single axis can be done
    with ellipsis:

    >>> ops.einsum("...j -> ...", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [..., 1], [...])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose or reorder any number of axes:

    >>> ops.einsum("ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum("ij -> ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum(c, [1, 0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Matrix vector multiplication:

    >>> ops.einsum("ij, j", a, b)
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum(a, [0, 1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum("...j, j", a, b)
    array([ 30,  80, 130, 180, 230])
    """
    if any_symbolic_tensors(operands):
        return Einsum(subscripts).symbolic_call(*operands, **kwargs)
    return backend.ops.numpy.einsum(subscripts, *operands, **kwargs)


@keras_export(["keras.ops.empty", "keras.ops.numpy.empty"])
def empty(shape, dtype=None):
    """Return a tensor of given shape and type filled with uninitialized data.

    Args:
        shape: Shape of the empty tensor.
        dtype: Desired data type of the empty tensor.

    Returns:
        The empty tensor.
    """
    return backend.ops.numpy.empty(shape, dtype=dtype)


class EmptyLike(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.empty_like(x, dtype=self.dtype)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.empty_like", "keras.ops.numpy.empty_like"])
def empty_like(x, dtype=None):
    """Return a new uninitialized tensor with the same shape and dtype as `x`.

    Args:
        x: Input tensor to mimic shape and dtype.
        dtype: Optional data type. If None, uses `x.dtype`.

    Returns:
        A tensor with the same shape and dtype as `x`, with arbitrary contents.

    Example:
    >>> from keras import ops
    >>> x = ops.ones((2, 3), dtype="float32")
    >>> y = ops.empty_like(x)
    >>> y.shape
    (2, 3)
    >>> y.dtype
    dtype('float32')
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return EmptyLike(dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.empty_like(x, dtype=dtype)


class Equal(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.equal", "keras.ops.numpy.equal"])
def equal(x1, x2):
    """Returns `(x1 == x2)` element-wise.

    Args:
        x1: Tensor to compare.
        x2: Tensor to compare.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Equal().symbolic_call(x1, x2)
    return backend.ops.numpy.equal(x1, x2)


class Exp(Operation):
    def call(self, x):
        return backend.ops.numpy.exp(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.exp", "keras.ops.numpy.exp"])
def exp(x):
    """Calculate the exponential of all elements in the input tensor.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise exponential of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Exp().symbolic_call(x)
    return backend.ops.numpy.exp(x)


class Exp2(Operation):
    def call(self, x):
        return backend.ops.numpy.exp2(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.exp2", "keras.ops.numpy.exp2"])
def exp2(x):
    """Calculate the base-2 exponential of all elements in the input tensor.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base-2 exponential of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Exp2().symbolic_call(x)
    return backend.ops.numpy.exp2(x)


class ExpandDims(Operation):
    def __init__(self, axis, *, name=None):
        super().__init__(name=name)
        if not isinstance(axis, (int, tuple, list)):
            raise ValueError(
                "The `axis` argument to `expand_dims` should be an integer, "
                f"tuple or list. Received axis={axis}"
            )
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.expand_dims(x, self.axis)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_expand_dims_output_shape(
            x.shape, self.axis
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(
    [
        "keras.ops.expand_dims",
        "keras.ops.numpy.expand_dims",
    ]
)
def expand_dims(x, axis):
    """Expand the shape of a tensor.

    Insert a new axis at the `axis` position in the expanded tensor shape.

    Args:
        x: Input tensor.
        axis: Position in the expanded axes where the new axis
            (or axes) is placed.

    Returns:
        Output tensor with the number of dimensions increased.
    """
    if any_symbolic_tensors((x,)):
        return ExpandDims(axis=axis).symbolic_call(x)
    return backend.ops.numpy.expand_dims(x, axis)


class Expm1(Operation):
    def call(self, x):
        return backend.ops.numpy.expm1(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.expm1", "keras.ops.numpy.expm1"])
def expm1(x):
    """Calculate `exp(x) - 1` for all elements in the tensor.

    Args:
        x: Input values.

    Returns:
        Output tensor, element-wise exponential minus one.
    """
    if any_symbolic_tensors((x,)):
        return Expm1().symbolic_call(x)
    return backend.ops.numpy.expm1(x)


class Flip(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.flip(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is not None:
            canonicalize_axes(self.axis, len(x.shape))
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.flip", "keras.ops.numpy.flip"])
def flip(x, axis=None):
    """Reverse the order of elements in the tensor along the given axis.

    The shape of the tensor is preserved, but the elements are reordered.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to flip the tensor. The default,
            `axis=None`, will flip over all of the axes of the input tensor.

    Returns:
        Output tensor with entries of `axis` reversed.
    """
    if any_symbolic_tensors((x,)):
        return Flip(axis=axis).symbolic_call(x)
    return backend.ops.numpy.flip(x, axis=axis)


class Fliplr(Operation):
    def call(self, x):
        return backend.ops.numpy.fliplr(x)

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                "`fliplr` requires input with 2 or more dimensions. "
                f"Received: x.shape={x.shape}"
            )
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.fliplr", "keras.ops.numpy.fliplr"])
def fliplr(x):
    """Reverse the order of elements along axis 1 (left/right).

    For a 2-D tensor, this flips the entries in each row in the
    left/right direction. Columns are preserved, but appear in a
    different order than before.

    Args:
        x: Input tensor, must be at least 2-D.

    Returns:
        Output tensor with columns reversed.
    """
    if any_symbolic_tensors((x,)):
        return Fliplr().symbolic_call(x)
    return backend.ops.numpy.fliplr(x)


class Flipud(Operation):
    def call(self, x):
        return backend.ops.numpy.flipud(x)

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                "`flipud` requires input with 1 or more dimensions. "
                f"Received: x.shape={x.shape}"
            )
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.flipud", "keras.ops.numpy.flipud"])
def flipud(x):
    """Reverse the order of elements along axis 0 (up/down).

    For a 2-D tensor, this flips the entries in each column in the
    up/down direction. Rows are preserved, but appear in a different
    order than before.

    Args:
        x: Input tensor, must be at least 1-D.

    Returns:
        Output tensor with rows reversed.
    """
    if any_symbolic_tensors((x,)):
        return Flipud().symbolic_call(x)
    return backend.ops.numpy.flipud(x)


class Floor(Operation):
    def call(self, x):
        return backend.ops.numpy.floor(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.floor", "keras.ops.numpy.floor"])
def floor(x):
    """Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise floor of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Floor().symbolic_call(x)
    return backend.ops.numpy.floor(x)


class Full(Operation):
    def __init__(self, shape, dtype=None, *, name=None):
        super().__init__(name=name)
        self.shape = shape
        self.dtype = dtype

    def call(self, fill_value):
        return backend.ops.numpy.full(self.shape, fill_value, dtype=self.dtype)

    def compute_output_spec(self, fill_value):
        dtype = backend.floatx() if self.dtype is None else self.dtype
        return KerasTensor(self.shape, dtype=dtype)


@keras_export(["keras.ops.full", "keras.ops.numpy.full"])
def full(shape, fill_value, dtype=None):
    """Return a new tensor of given shape and type, filled with `fill_value`.

    Args:
        shape: Shape of the new tensor.
        fill_value: Fill value.
        dtype: Desired data type of the tensor.

    Returns:
        Output tensor.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((fill_value,)):
        return Full(shape=shape, dtype=dtype).symbolic_call(fill_value)
    return backend.ops.numpy.full(shape, fill_value, dtype=dtype)


class FullLike(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x, fill_value):
        return backend.ops.numpy.full_like(x, fill_value, dtype=self.dtype)

    def compute_output_spec(self, x, fill_value):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.full_like", "keras.ops.numpy.full_like"])
def full_like(x, fill_value, dtype=None):
    """Return a full tensor with the same shape and type as the given tensor.

    Args:
        x: Input tensor.
        fill_value: Fill value.
        dtype: Overrides data type of the result.

    Returns:
        Tensor of `fill_value` with the same shape and type as `x`.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x, fill_value)):
        return FullLike(dtype=dtype).symbolic_call(x, fill_value)
    return backend.ops.numpy.full_like(x, fill_value, dtype=dtype)


class Gcd(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.gcd(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.gcd", "keras.ops.numpy.gcd"])
def gcd(x1, x2):
    """Greatest common divisor of `x1` and `x2`, element-wise.

    Args:
        x1: First input tensor (integer type).
        x2: Second input tensor (integer type).

    Returns:
        Output tensor, element-wise greatest common divisor of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Gcd().symbolic_call(x1, x2)
    return backend.ops.numpy.gcd(x1, x2)


class GetItem(Operation):
    def call(self, x, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[key]

    def compute_output_spec(self, x, key):
        remaining_shape = list(x.shape)
        new_shape = []
        if isinstance(key, int):
            remaining_key = [key]
        elif isinstance(key, tuple):
            remaining_key = list(key)
        elif isinstance(key, list):
            remaining_key = key.copy()
        else:
            raise ValueError(
                f"Unsupported key type for array slice. Received: `{key}`"
            )
        num_ellipses = remaining_key.count(Ellipsis)
        if num_ellipses > 1:
            raise ValueError(
                f"Slice should only have one ellipsis. Received: `{key}`"
            )
        elif num_ellipses == 0:
            # Add an implicit final ellipsis.
            remaining_key.append(Ellipsis)
        # Consume slice key element by element.
        while True:
            if not remaining_key:
                break
            subkey = remaining_key.pop(0)
            # Check for `newaxis` and `Ellipsis`.
            if subkey == Ellipsis:
                # Keep as many slices remain in our key, omitting `newaxis`.
                needed = len(remaining_key) - remaining_key.count(np.newaxis)
                consumed = len(remaining_shape) - needed
                new_shape += remaining_shape[:consumed]
                remaining_shape = remaining_shape[consumed:]
                continue
            # All frameworks follow numpy for newaxis. `np.newaxis == None`.
            if subkey == np.newaxis:
                new_shape.append(1)
                continue
            # At this point, we need to consume a new axis from the shape.
            if not remaining_shape:
                raise ValueError(
                    f"Array has shape {x.shape} but slice "
                    f"has to many indices. Received: `{key}`"
                )
            length = remaining_shape.pop(0)
            if isinstance(subkey, int):
                if length is not None:
                    index = subkey if subkey >= 0 else subkey + length
                    if index < 0 or index >= length:
                        raise ValueError(
                            f"Array has shape {x.shape} but out-of-bounds "
                            f"index {key} was requested."
                        )
            elif isinstance(subkey, slice):
                if length is not None:
                    # python3 friendly way to compute a slice length.
                    new_length = len(range(*subkey.indices(length)))
                    new_shape.append(new_length)
                else:
                    new_shape.append(length)
            else:
                raise ValueError(
                    f"Unsupported key type for array slice. Received: `{key}`"
                )
        return KerasTensor(tuple(new_shape), dtype=x.dtype)


@keras_export(["keras.ops.get_item", "keras.ops.numpy.get_item"])
def get_item(x, key):
    """Return `x[key]`."""
    if any_symbolic_tensors((x,)):
        return GetItem().symbolic_call(x, key)
    return x[key]


class Geomspace(Operation):
    def __init__(self, num=50, endpoint=True, dtype=None, axis=0, *, name=None):
        super().__init__(name=name)
        self.num = num
        self.endpoint = endpoint
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.ops.numpy.geomspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        axis = canonicalize_axis(self.axis, len(output_shape) + 1)
        output_shape = list(output_shape)
        output_shape.insert(axis, self.num)
        dtype = (
            self.dtype
            if self.dtype is not None
            else standardize_dtype(getattr(start, "dtype", type(start)))
        )
        dtype = backend.result_type(dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.geomspace", "keras.ops.numpy.geomspace"])
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """Returns numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly
    instead of as logarithms. Each output sample is a constant multiple of
    the previous.

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless `endpoint` is `False`.
            In that case, `num + 1` values are spaced over the interval in
            log-space, of which all but the last (a sequence of length `num`)
            are returned.
        num: Number of samples to generate. Defaults to `50`.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is not
            included. Defaults to `True`.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of `num` samples, evenly spaced on a log scale.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((start, stop)):
        return Geomspace(num, endpoint, dtype, axis)(start, stop)
    return backend.ops.numpy.geomspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
        axis=axis,
    )


class Greater(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.greater(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.greater", "keras.ops.numpy.greater"])
def greater(x1, x2):
    """Return the truth value of `x1 > x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Greater().symbolic_call(x1, x2)
    return backend.ops.numpy.greater(x1, x2)


class GreaterEqual(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.greater_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.greater_equal",
        "keras.ops.numpy.greater_equal",
    ]
)
def greater_equal(x1, x2):
    """Return the truth value of `x1 >= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return GreaterEqual().symbolic_call(x1, x2)
    return backend.ops.numpy.greater_equal(x1, x2)


class Hstack(Operation):
    def call(self, xs):
        return backend.ops.numpy.hstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[1], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[1] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[1]
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[1] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.hstack", "keras.ops.numpy.hstack"])
def hstack(xs):
    """Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the first axis for 1-D tensors,
    and along the second axis for all other tensors.

    Args:
        xs: Sequence of tensors.

    Returns:
        The tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Hstack().symbolic_call(xs)
    return backend.ops.numpy.hstack(xs)


class Hsplit(Operation):
    def __init__(self, indices_or_sections, *, name=None):
        super().__init__(name=name)
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections

    def call(self, x):
        return _hsplit(x, self.indices_or_sections)

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                "`hsplit` only works on arrays of at least 1 dimension. "
                f"Received array with shape {x.shape}."
            )

        axis = 0 if len(x.shape) == 1 else 1
        return _compute_split_output_spec(x, self.indices_or_sections, axis)


@keras_export(["keras.ops.hsplit", "keras.ops.numpy.hsplit"])
def hsplit(x, indices_or_sections):
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        x: Input tensor.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along axis 1 (if ndim >= 2) or axis 0 (if ndim == 1).
            If a 1-D array of sorted integers, the entries indicate indices at
            which the tensor will be split along the axis.

    Returns:
        A list of sub-arrays.

    Example:

    >>> x = keras.ops.arange(16.0).reshape((4, 4))
    >>> keras.ops.hsplit(x, 2)
    [array([[ 0.,  1.],
           [ 4.,  5.],
           [ 8.,  9.],
           [12., 13.]]),
     array([[ 2.,  3.],
           [ 6.,  7.],
           [10., 11.],
           [14., 15.]])]
    >>> keras.ops.hsplit(x, [1, 3])
    [array([[0.],
        [4.],
        [8.],
        [12.]]),
    array([[1., 2.],
        [5., 6.],
        [9., 10.],
        [13., 14.]]),
    array([[3.],
        [7.],
        [11.],
        [15.]])]
    """
    if any_symbolic_tensors((x,)):
        return Hsplit(indices_or_sections).symbolic_call(x)
    return _hsplit(x, indices_or_sections)


def _hsplit(x, indices_or_sections):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "hsplit"
    ):
        return backend.ops.numpy.hsplit(x, indices_or_sections)
    x = backend.ops.convert_to_tensor(x)
    # 1D inputs are split along axis=0. Inputs with 2 or more dimensions are
    # split along axis=1.
    axis = 0 if len(x.shape) == 1 else 1
    return backend.ops.numpy.split(x, indices_or_sections, axis=axis)


class Hypot(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.hypot(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
        if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
            dtype = backend.floatx()
        elif dtype == "int64":
            dtype = "float64"
        return KerasTensor(broadcast_shapes(x1.shape, x2.shape), dtype=dtype)


@keras_export(["keras.ops.hypot", "keras.ops.numpy.hypot"])
def hypot(x1, x2):
    """Element-wise hypotenuse of right triangles with legs `x1` and `x2`.

    This is equivalent to computing `sqrt(x1**2 + x2**2)` element-wise,
    with shape determined by broadcasting.

    Args:
        x1: A tensor, representing the first leg of the right triangle.
        x2: A tensor, representing the second leg of the right triangle.

    Returns:
        A tensor with a shape determined by broadcasting `x1` and `x2`.

    Example:
    >>> x1 = keras.ops.convert_to_tensor([3.0, 4.0, 5.0])
    >>> x2 = keras.ops.convert_to_tensor([4.0, 3.0, 12.0])
    >>> keras.ops.hypot(x1, x2)
    array([5., 5., 13.], dtype=float32)

    >>> x1 = keras.ops.convert_to_tensor([[1, 2], [3, 4]])
    >>> x2 = keras.ops.convert_to_tensor([1, 1])
    >>> keras.ops.hypot(x1, x2)
    array([[1.41421356 2.23606798],
          [3.16227766 4.12310563]], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Hypot().symbolic_call(x1, x2)
    return backend.ops.numpy.hypot(x1, x2)


@keras_export(["keras.ops.identity", "keras.ops.numpy.identity"])
def identity(n, dtype=None):
    """Return the identity tensor.

    The identity tensor is a square tensor with ones on the main diagonal and
    zeros elsewhere.

    Args:
        n: Number of rows (and columns) in the `n x n` output tensor.
        dtype: Data type of the output tensor.

    Returns:
        The identity tensor.
    """
    return backend.ops.numpy.identity(n, dtype=dtype)


class Imag(Operation):
    def call(self, x):
        return backend.ops.numpy.imag(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.imag", "keras.ops.numpy.imag"])
def imag(x):
    """Return the imaginary part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The imaginary component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Imag().symbolic_call(x)
    return backend.ops.numpy.imag(x)


class I0(Operation):
    def call(self, x):
        return backend.ops.numpy.i0(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        if dtype in ["int64", "float64"]:
            dtype = "float64"
        elif dtype not in ["bfloat16", "float16"]:
            dtype = backend.floatx()
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.i0", "keras.ops.numpy.i0"])
def i0(x):
    """Modified Bessel function of the first kind, order 0.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise modified Bessel function of the
        first kind, order 0 of `x`.
    """
    if any_symbolic_tensors((x,)):
        return I0().symbolic_call(x)
    return backend.ops.numpy.i0(x)


class Isclose(Operation):
    def __init__(self, equal_nan=False, *, name=None):
        super().__init__(name=name)
        self.equal_nan = equal_nan

    def call(self, x1, x2, rtol=1e-5, atol=1e-8):
        return backend.ops.numpy.isclose(x1, x2, rtol, atol, self.equal_nan)

    def compute_output_spec(self, x1, x2, rtol=1e-5, atol=1e-8):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.isclose", "keras.ops.numpy.isclose"])
def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return whether two tensors are element-wise almost equal.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If `True`, element-wise NaNs are considered equal.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Isclose(equal_nan=equal_nan).symbolic_call(x1, x2, rtol, atol)
    return backend.ops.numpy.isclose(x1, x2, rtol, atol, equal_nan)


class Isfinite(Operation):
    def call(self, x):
        return backend.ops.numpy.isfinite(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isfinite", "keras.ops.numpy.isfinite"])
def isfinite(x):
    """Return whether a tensor is finite, element-wise.

    Real values are finite when they are not NaN, not positive infinity, and
    not negative infinity. Complex values are finite when both their real
    and imaginary parts are finite.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isfinite().symbolic_call(x)
    return backend.ops.numpy.isfinite(x)


class IsIn(Operation):
    def __init__(
        self,
        assume_unique=False,
        invert=False,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.assume_unique = assume_unique
        self.invert = invert

    def call(self, x1, x2):
        return backend.ops.numpy.isin(
            x1, x2, assume_unique=self.assume_unique, invert=self.invert
        )

    def compute_output_spec(self, x1, x2):
        return KerasTensor(x1.shape, dtype="bool")


@keras_export(["keras.ops.isin", "keras.ops.numpy.isin"])
def isin(x1, x2, assume_unique=False, invert=False):
    """Test whether each element of `x1` is present in `x2`.

    This operation performs element-wise checks to determine if each value
    in `x1` is contained within `x2`. The result is a boolean tensor with
    the same shape as `x1`, where each entry is `True` if the corresponding
    element in `x1` is in `x2`, and `False` otherwise.

    Args:
        x1: Input tensor or array-like structure to test.
        x2: Values against which each element of `x1` is tested.
            Can be a tensor, list, or scalar.
        assume_unique: Boolean (default: False).
            If True, assumes both `x1` and `x2` contain only unique elements.
            This can speed up the computation. If False, duplicates will be
            handled correctly but may impact performance.
        invert: A boolean (default: False).
            If True, inverts the result. Entries will be `True`
            where `x1` elements are not in `x2`.

    Returns:
        A boolean tensor of the same shape as `x1` indicating element-wise
        membership in `x2`.

    Example:
    >>> from keras import ops
    >>> x1 = ops.array([0, 1, 2, 5])
    >>> x2 = ops.array([0, 2])
    >>> result = ops.isin(x1, x2)
    array([ True, False,  True, False])
    """
    if any_symbolic_tensors((x1, x2)):
        return IsIn(assume_unique=assume_unique, invert=invert).symbolic_call(
            x1, x2
        )
    return backend.ops.numpy.isin(
        x1, x2, assume_unique=assume_unique, invert=invert
    )


class Isinf(Operation):
    def call(self, x):
        return backend.ops.numpy.isinf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isinf", "keras.ops.numpy.isinf"])
def isinf(x):
    """Test element-wise for positive or negative infinity.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isinf().symbolic_call(x)
    return backend.ops.numpy.isinf(x)


class Isnan(Operation):
    def call(self, x):
        return backend.ops.numpy.isnan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isnan", "keras.ops.numpy.isnan"])
def isnan(x):
    """Test element-wise for NaN and return result as a boolean tensor.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isnan().symbolic_call(x)
    return backend.ops.numpy.isnan(x)


class Isneginf(Operation):
    def call(self, x):
        return backend.ops.numpy.isneginf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isneginf", "keras.ops.numpy.isneginf"])
def isneginf(x):
    """Test element-wise for negative infinity.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isneginf().symbolic_call(x)
    return backend.ops.numpy.isneginf(x)


class Isposinf(Operation):
    def call(self, x):
        return backend.ops.numpy.isposinf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isposinf", "keras.ops.numpy.isposinf"])
def isposinf(x):
    """Test element-wise for positive infinity.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isposinf().symbolic_call(x)
    return backend.ops.numpy.isposinf(x)


class Isreal(Operation):
    def call(self, x):
        return backend.ops.numpy.isreal(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isreal", "keras.ops.numpy.isreal"])
def isreal(x):
    """Test element-wise for real numbers.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.

    Example:
    >>> from keras import ops
    >>> x = ops.array([1+1j, 1+0j, 4.5, 3, 2, 2j], dtype="complex64")
    >>> ops.isreal(x)
    array([False,  True,  True,  True,  True, False])
    """
    if any_symbolic_tensors((x,)):
        return Isreal().symbolic_call(x)
    return backend.ops.numpy.isreal(x)


class Kron(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.kron(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])

        def _mul_shape_dim(a, b):
            if a is None or b is None:
                return None
            return a * b

        output_shape = tuple(
            _mul_shape_dim(a, b) for a, b in zip(x1_shape, x2_shape)
        )

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.kron", "keras.ops.numpy.kron"])
def kron(x1, x2):
    """Kronecker product of `x1` and `x2`.

    Computes the Kronecker product of two input tensors. If `x1` has shape
    `(a0, a1, ..., an)` and `x2` has shape `(b0, b1, ..., bn)`, then the
    output will have shape `(a0*b0, a1*b1, ..., an*bn)`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        A tensor representing the Kronecker product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Kron().symbolic_call(x1, x2)
    return backend.ops.numpy.kron(x1, x2)


class Lcm(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.lcm(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.lcm", "keras.ops.numpy.lcm"])
def lcm(x1, x2):
    """Least common multiple of `x1` and `x2`, element-wise.

    Args:
        x1: First input tensor (integer type).
        x2: Second input tensor (integer type).

    Returns:
        Output tensor, element-wise least common multiple of `x1` and `x2`.

    Example:
    >>> x1 = keras.ops.convert_to_tensor([2, 3, 4])
    >>> x2 = keras.ops.convert_to_tensor([5, 6, 7])
    >>> keras.ops.lcm(x1, x2)
    array([10,  6, 28], dtype=int32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Lcm().symbolic_call(x1, x2)
    return backend.ops.numpy.lcm(x1, x2)


class Ldexp(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.ldexp(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.ldexp", "keras.ops.numpy.ldexp"])
def ldexp(x1, x2):
    """Multiply `x1` by 2 raised to the power of `x2`, element-wise.

    This function computes:
        ldexp(x1, x2) = x1 * 2**x2

    Args:
        x1: Float input tensor.
        x2: Integer exponent tensor.

    Returns:
        Output tensor

    Example:
    >>> x1 = keras.ops.convert_to_tensor([0.75, 1.5])
    >>> x2 = keras.ops.convert_to_tensor([1, 2])
    >>> keras.ops.ldexp(x1, x2)
    array([1.5, 6. ], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Ldexp().symbolic_call(x1, x2)
    return backend.ops.numpy.ldexp(x1, x2)


class Less(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.less(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.less", "keras.ops.numpy.less"])
def less(x1, x2):
    """Return the truth value of `x1 < x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Less().symbolic_call(x1, x2)
    return backend.ops.numpy.less(x1, x2)


class LessEqual(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.less_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.less_equal",
        "keras.ops.numpy.less_equal",
    ]
)
def less_equal(x1, x2):
    """Return the truth value of `x1 <= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return LessEqual().symbolic_call(x1, x2)
    return backend.ops.numpy.less_equal(x1, x2)


class Linspace(Operation):
    def __init__(
        self,
        num=50,
        endpoint=True,
        retstep=False,
        dtype=None,
        axis=0,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.num = num
        self.endpoint = endpoint
        self.retstep = retstep
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.ops.numpy.linspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            retstep=self.retstep,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        output_shape = list(output_shape)
        axis = canonicalize_axis(self.axis, len(output_shape) + 1)
        output_shape.insert(axis, self.num)

        dtype = (
            self.dtype
            if self.dtype is not None
            else standardize_dtype(getattr(start, "dtype", type(start)))
        )
        dtype = backend.result_type(dtype, float)
        if self.retstep:
            return (KerasTensor(output_shape, dtype=dtype), None)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.linspace", "keras.ops.numpy.linspace"])
def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    """Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the interval
    `[start, stop]`.

    The endpoint of the interval can optionally be excluded.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence, unless `endpoint` is set to
            `False`. In that case, the sequence consists of all but the last
            of `num + 1` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is `False`.
        num: Number of samples to generate. Defaults to `50`. Must be
            non-negative.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is
            not included. Defaults to `True`.
        retstep: If `True`, return `(samples, step)`, where `step` is the
            spacing between samples.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only if
            start or stop are array-like. Defaults to `0`.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced numbers.
        If `retstep` is `True`, returns `(samples, step)`
    """
    if any_symbolic_tensors((start, stop)):
        return Linspace(num, endpoint, retstep, dtype, axis)(start, stop)
    return backend.ops.numpy.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


class Log(Operation):
    def call(self, x):
        return backend.ops.numpy.log(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log", "keras.ops.numpy.log"])
def log(x):
    """Natural logarithm, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log().symbolic_call(x)
    return backend.ops.numpy.log(x)


class Log10(Operation):
    def call(self, x):
        return backend.ops.numpy.log10(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log10", "keras.ops.numpy.log10"])
def log10(x):
    """Return the base 10 logarithm of the input tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base 10 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log10().symbolic_call(x)
    return backend.ops.numpy.log10(x)


class Log1p(Operation):
    def call(self, x):
        return backend.ops.numpy.log1p(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.log1p", "keras.ops.numpy.log1p"])
def log1p(x):
    """Returns the natural logarithm of one plus the `x`, element-wise.

    Calculates `log(1 + x)`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `1 + x`.
    """
    if any_symbolic_tensors((x,)):
        return Log1p().symbolic_call(x)
    return backend.ops.numpy.log1p(x)


class Log2(Operation):
    def call(self, x):
        return backend.ops.numpy.log2(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log2", "keras.ops.numpy.log2"])
def log2(x):
    """Base-2 logarithm of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base-2 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log2().symbolic_call(x)
    return backend.ops.numpy.log2(x)


class Logaddexp(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.logaddexp(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.logaddexp", "keras.ops.numpy.logaddexp"])
def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs.

    Calculates `log(exp(x1) + exp(x2))`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logarithm of the sum of exponentiations
        of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Logaddexp().symbolic_call(x1, x2)
    return backend.ops.numpy.logaddexp(x1, x2)


class Logaddexp2(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.logaddexp2(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.logaddexp2", "keras.ops.numpy.logaddexp2"])
def logaddexp2(x1, x2):
    """Base-2 logarithm of the sum of exponentiations of the inputs.

    Calculates `log2(2**x1 + 2**x2)`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise log base 2 of the sum of 2**x1 and 2**x2.

    Example:
    >>> from keras import ops
    >>> x1 = ops.array([1, 2, 3])
    >>> x2 = ops.array([1, 2, 3])
    >>> ops.logaddexp2(x1, x2)
    array([2., 3., 4.], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Logaddexp2().symbolic_call(x1, x2)
    return backend.ops.numpy.logaddexp2(x1, x2)


class LogicalAnd(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.logical_and(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_and",
        "keras.ops.numpy.logical_and",
    ]
)
def logical_and(x1, x2):
    """Computes the element-wise logical AND of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical AND of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalAnd().symbolic_call(x1, x2)
    return backend.ops.numpy.logical_and(x1, x2)


class LogicalNot(Operation):
    def call(self, x):
        return backend.ops.numpy.logical_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_not",
        "keras.ops.numpy.logical_not",
    ]
)
def logical_not(x):
    """Computes the element-wise NOT of the given input tensor.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise logical NOT of the input.
    """
    if any_symbolic_tensors((x,)):
        return LogicalNot().symbolic_call(x)
    return backend.ops.numpy.logical_not(x)


class LogicalOr(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.logical_or(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_or",
        "keras.ops.numpy.logical_or",
    ]
)
def logical_or(x1, x2):
    """Computes the element-wise logical OR of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical OR of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalOr().symbolic_call(x1, x2)
    return backend.ops.numpy.logical_or(x1, x2)


class Logspace(Operation):
    def __init__(
        self, num=50, endpoint=True, base=10, dtype=None, axis=0, *, name=None
    ):
        super().__init__(name=name)
        self.num = num
        self.endpoint = endpoint
        self.base = base
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.ops.numpy.logspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            base=self.base,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        output_shape = list(output_shape)
        axis = canonicalize_axis(self.axis, len(output_shape) + 1)
        output_shape.insert(axis, self.num)
        dtype = (
            self.dtype
            if self.dtype is not None
            else standardize_dtype(getattr(start, "dtype", type(start)))
        )
        dtype = backend.result_type(dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.logspace", "keras.ops.numpy.logspace"])
def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    """Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at `base ** start` and ends with
    `base ** stop` (see `endpoint` below).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless `endpoint` is `False`.
            In that case, `num + 1` values are spaced over the interval in
            log-space, of which all but the last (a sequence of length `num`)
            are returned.
        num: Number of samples to generate. Defaults to `50`.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is not
            included. Defaults to `True`.
        base: The base of the log space. Defaults to `10`.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced samples on a log scale.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((start, stop)):
        return Logspace(num, endpoint, base, dtype, axis)(start, stop)
    return backend.ops.numpy.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.matmul(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = operation_utils.compute_matmul_output_shape(
            x1_shape, x2_shape
        )
        x1_sparse = getattr(x1, "sparse", True)
        x2_sparse = getattr(x2, "sparse", True)
        output_sparse = x1_sparse and x2_sparse
        x1_dtype = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_dtype = standardize_dtype(getattr(x2, "dtype", type(x2)))
        if x1_dtype == "int8" and x2_dtype == "int8":
            dtype = "int32"
        else:
            dtype = dtypes.result_type(x1_dtype, x2_dtype)
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.matmul", "keras.ops.numpy.matmul"])
def matmul(x1, x2):
    """Matrix product of two tensors.

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If either tensor is N-D, N > 2, it is treated as a stack of matrices
      residing in the last two indexes and broadcast accordingly.
    - If the first tensor is 1-D, it is promoted to a matrix by prepending
      a 1 to its dimensions. After matrix multiplication the prepended
      1 is removed.
    - If the second tensor is 1-D, it is promoted to a matrix by appending a 1
      to its dimensions. After matrix multiplication the appended 1 is removed.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, matrix product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Matmul().symbolic_call(x1, x2)
    return backend.ops.numpy.matmul(x1, x2)


class Max(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.ops.numpy.max(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


class Amax(Max):
    """Kept for backward compatibility of saved models using `amax`."""

    pass


@keras_export(
    [
        "keras.ops.max",
        "keras.ops.amax",
        "keras.ops.numpy.max",
        "keras.ops.numpy.amax",
    ]
)
def max(x, axis=None, keepdims=False, initial=None):
    """Return the maximum of a tensor or maximum along an axis.

    `keras.ops.amax` is an alias for this function.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.
        initial: The minimum value of an output element. Defaults to `None`.

    Returns:
        Maximum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Max(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.ops.numpy.max(
        x, axis=axis, keepdims=keepdims, initial=initial
    )


class Maximum(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.maximum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.maximum", "keras.ops.numpy.maximum"])
def maximum(x1, x2):
    """Element-wise maximum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise maximum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Maximum().symbolic_call(x1, x2)
    return backend.ops.numpy.maximum(x1, x2)


class Fmax(Operation):
    def call(self, x1, x2):
        return _fmax(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.fmax", "keras.ops.numpy.fmax"])
def fmax(x1, x2):
    """Element-wise maximum of tensor elements, ignoring NaNs.

    Compare two tensors element-wise and return the maximum. If one of the
    elements being compared is a NaN, the non-NaN element is returned.
    If both elements are NaNs, the NaN is returned.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise maximum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([2.0, float("nan")])
    >>> x2 = keras.ops.convert_to_tensor([1.0, 4.0])
    >>> keras.ops.fmax(x1, x2)
    array([2.0, 4.0], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Fmax().symbolic_call(x1, x2)
    return _fmax(x1, x2)


def _fmax(x1, x2):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "fmax"
    ):
        return backend.ops.numpy.fmax(x1, x2)
    if not isinstance(x1, (int, float)):
        x1 = backend.ops.convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = backend.ops.convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = backend.ops.convert_to_tensor(x1, dtype)
    x2 = backend.ops.convert_to_tensor(x2, dtype)
    res = backend.ops.numpy.maximum(x1, x2)
    if "float" not in dtype:
        return res

    res = backend.ops.numpy.where(backend.ops.numpy.isnan(x2), x1, res)
    return backend.ops.numpy.where(backend.ops.numpy.isnan(x1), x2, res)


class Median(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.median(
            x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )
        if standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.median", "keras.ops.numpy.median"])
def median(x, axis=None, keepdims=False):
    """Compute the median along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the medians are computed. Defaults to
            `axis=None` which is to compute the median(s) along a flattened
            version of the array.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.

    Returns:
        The output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Median(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.median(x, axis=axis, keepdims=keepdims)


class Meshgrid(Operation):
    def __init__(self, indexing="xy", *, name=None):
        super().__init__(name=name)
        if indexing not in ("xy", "ij"):
            raise ValueError(
                "Valid values for `indexing` are 'xy' and 'ij', "
                "but received {index}."
            )
        self.indexing = indexing

    def call(self, *x):
        return backend.ops.numpy.meshgrid(*x, indexing=self.indexing)

    def compute_output_spec(self, *x):
        output_shape = []
        for xi in x:
            if len(xi.shape) == 0:
                size = 1
            else:
                if None in xi.shape:
                    size = None
                else:
                    size = python_math.prod(xi.shape)
            output_shape.append(size)
        if self.indexing == "ij":
            return [KerasTensor(output_shape) for _ in range(len(x))]
        tmp = output_shape[0]
        output_shape[0] = output_shape[1]
        output_shape[1] = tmp
        return [
            KerasTensor(output_shape, dtype=xi.dtype) for _ in range(len(x))
        ]


@keras_export(["keras.ops.meshgrid", "keras.ops.numpy.meshgrid"])
def meshgrid(*x, indexing="xy"):
    """Creates grids of coordinates from coordinate vectors.

    Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
    lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
    `G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
    `Gi` is constructed by expanding `Ti` to the result shape.

    Args:
        x: 1-D tensors representing the coordinates of a grid.
        indexing: `"xy"` or `"ij"`. "xy" is cartesian; `"ij"` is matrix
            indexing of output. Defaults to `"xy"`.

    Returns:
        Sequence of N tensors.

    Example:
    >>> from keras.src import ops
    >>> x = ops.array([1, 2, 3])
    >>> y = ops.array([4, 5, 6])

    >>> grid_x, grid_y = ops.meshgrid(x, y, indexing="ij")
    >>> grid_x
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> grid_y
    array([[4, 5, 6],
           [4, 5, 6],
           [4, 5, 6]])
    """
    if any_symbolic_tensors(x):
        return Meshgrid(indexing=indexing).symbolic_call(*x)
    return backend.ops.numpy.meshgrid(*x, indexing=indexing)


class Min(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.ops.numpy.min(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


class Amin(Min):
    """Kept for backward compatibility of saved models using `amin`."""

    pass


@keras_export(
    [
        "keras.ops.min",
        "keras.ops.amin",
        "keras.ops.numpy.min",
        "keras.ops.numpy.amin",
    ]
)
def min(x, axis=None, keepdims=False, initial=None):
    """Return the minimum of a tensor or minimum along an axis.

    `keras.ops.amin` is an alias for this function.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.
        initial: The maximum value of an output element. Defaults to `None`.

    Returns:
        Minimum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Min(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.ops.numpy.min(
        x, axis=axis, keepdims=keepdims, initial=initial
    )


class Minimum(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.minimum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.minimum", "keras.ops.numpy.minimum"])
def minimum(x1, x2):
    """Element-wise minimum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise minimum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Minimum().symbolic_call(x1, x2)
    return backend.ops.numpy.minimum(x1, x2)


class Fmin(Operation):
    def call(self, x1, x2):
        return _fmin(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.fmin", "keras.ops.numpy.fmin"])
def fmin(x1, x2):
    """Element-wise minimum of tensor elements, ignoring NaNs.

    Compare two tensors element-wise and return the minimum. If one of the
    elements being compared is a NaN, the non-NaN element is returned.
    If both elements are NaNs, the NaN is returned.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise minimum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([2.0, float("nan")])
    >>> x2 = keras.ops.convert_to_tensor([1.0, 4.0])
    >>> keras.ops.fmin(x1, x2)
    array([1.0, 4.0], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Fmin().symbolic_call(x1, x2)
    return _fmin(x1, x2)


def _fmin(x1, x2):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "fmin"
    ):
        return backend.ops.numpy.fmin(x1, x2)
    if not isinstance(x1, (int, float)):
        x1 = backend.ops.convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = backend.ops.convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = backend.ops.convert_to_tensor(x1, dtype)
    x2 = backend.ops.convert_to_tensor(x2, dtype)
    res = backend.ops.numpy.minimum(x1, x2)
    if "float" not in dtype:
        return res

    res = backend.ops.numpy.where(backend.ops.numpy.isnan(x2), x1, res)
    return backend.ops.numpy.where(backend.ops.numpy.isnan(x1), x2, res)


class Mod(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.mod(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.mod", "keras.ops.numpy.mod"])
def mod(x1, x2):
    """Returns the element-wise remainder of division.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise remainder of division.
    """
    if any_symbolic_tensors((x1, x2)):
        return Mod().symbolic_call(x1, x2)
    return backend.ops.numpy.mod(x1, x2)


class Fmod(Operation):
    def call(self, x1, x2):
        return _fmod(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.fmod", "keras.ops.numpy.fmod"])
def fmod(x1, x2):
    """Returns the element-wise remainder of division with truncation.

    Computes the remainder complementary to the `floor_divide` function,
    equivalent to the C library function `fmod`. The result has the same
    sign as the dividend `x1`. This is different from `keras.ops.mod`
    which has the same sign as the divisor `x2`.

    Args:
        x1: First tensor, the dividend.
        x2: Second tensor, the divisor.

    Returns:
        Output tensor, element-wise remainder with truncation.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([-5.5, 5.5])
    >>> x2 = keras.ops.convert_to_tensor([2.0, -2.0])
    >>> keras.ops.fmod(x1, x2)
    array([-1.5,  1.5], dtype=float32)
    >>> keras.ops.mod(x1, x2)
    array([ 0.5, -0.5], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Fmod().symbolic_call(x1, x2)
    return _fmod(x1, x2)


def _fmod(x1, x2):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "fmod"
    ):
        return backend.ops.numpy.fmod(x1, x2)
    if not isinstance(x1, (int, float)):
        x1 = backend.ops.convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = backend.ops.convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == "bool":
        dtype = "int32"
    x1 = backend.ops.convert_to_tensor(x1, dtype)
    x2 = backend.ops.convert_to_tensor(x2, dtype)
    return backend.ops.numpy.multiply(
        backend.ops.numpy.sign(x1),
        backend.ops.numpy.mod(
            backend.ops.numpy.abs(x1), backend.ops.numpy.abs(x2)
        ),
    )


class Moveaxis(Operation):
    def __init__(self, source, destination, *, name=None):
        super().__init__(name=name)
        if isinstance(source, int):
            self.source = [source]
        else:
            self.source = source
        if isinstance(destination, int):
            self.destination = [destination]
        else:
            self.destination = destination

        if len(self.source) != len(self.destination):
            raise ValueError(
                "`source` and `destination` arguments must have the same "
                f"number of elements, but received `source={source}` and "
                f"`destination={destination}`."
            )

    def call(self, x):
        return backend.ops.numpy.moveaxis(x, self.source, self.destination)

    def compute_output_spec(self, x):
        ndim = len(x.shape)
        sources = [canonicalize_axis(a, ndim) for a in self.source]
        destinations = [canonicalize_axis(a, ndim) for a in self.destination]
        x_shape = list(x.shape)
        output_shape = [-1 for _ in range(ndim)]
        for sc, dst in zip(sources, destinations):
            output_shape[dst] = x_shape[sc]
            x_shape[sc] = -1
        i, j = 0, 0
        while i < len(output_shape):
            while i < len(output_shape) and output_shape[i] != -1:
                # Find the first dim unset.
                i += 1
            while j < len(output_shape) and x_shape[j] == -1:
                # Find the first dim not being passed.
                j += 1
            if i == len(output_shape):
                break
            output_shape[i] = x_shape[j]
            i += 1
            j += 1
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.moveaxis", "keras.ops.numpy.moveaxis"])
def moveaxis(x, source, destination):
    """Move axes of a tensor to new positions.

    Other axes remain in their original order.

    Args:
        x: Tensor whose axes should be reordered.
        source: Original positions of the axes to move. These must be unique.
        destination: Destinations positions for each of the original axes.
            These must also be unique.

    Returns:
        Tensor with moved axes.
    """
    if any_symbolic_tensors((x,)):
        return Moveaxis(source, destination).symbolic_call(x)
    return backend.ops.numpy.moveaxis(x, source=source, destination=destination)


class Nanargmax(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanargmax(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        axis = [self.axis] if self.axis is not None else None
        return KerasTensor(
            reduce_shape(x.shape, axis=axis, keepdims=self.keepdims),
            dtype="int32",
        )


@keras_export(["keras.ops.nanargmax", "keras.ops.numpy.nanargmax"])
def nanargmax(x, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed. NaN values are ignored when computing the maximum.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 0.0]])

    >>> ops.nanargmax(x)
    array(2, dtype=int32)

    >>> ops.nanargmax(x, axis=0)
    array([0, 1, 0], dtype=int32)

    >>> ops.nanargmax(x, axis=1)
    array([2, 1], dtype=int32)

    >>> ops.nanargmax(x, axis=1, keepdims=True)
    array([[2],
           [1]], dtype=int32)
    """

    if any_symbolic_tensors((x,)):
        return Nanargmax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanargmax(x, axis=axis, keepdims=keepdims)


def _nanargmax(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanargmax"
    ):
        return backend.ops.numpy.nanargmax(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if not is_float_dtype(x.dtype):
        return backend.ops.numpy.argmax(x, axis=axis, keepdims=keepdims)
    nan_mask = backend.ops.numpy.isnan(x)
    neg_inf = backend.ops.cast(float("-inf"), x.dtype)
    result = backend.ops.numpy.argmax(
        backend.ops.numpy.where(nan_mask, neg_inf, x),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.where(
        backend.ops.numpy.all(nan_mask, axis=axis, keepdims=keepdims),
        backend.ops.cast(-1, result.dtype),
        result,
    )


class Nanargmin(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanargmin(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        axis = [self.axis] if self.axis is not None else None
        return KerasTensor(
            reduce_shape(x.shape, axis=axis, keepdims=self.keepdims),
            dtype="int32",
        )


@keras_export(["keras.ops.nanargmin", "keras.ops.numpy.nanargmin"])
def nanargmin(x, axis=None, keepdims=False):
    """Returns the indices of the minimum values along an axis, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
                in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed. NaN values are ignored when computing the minimum.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 0.0]])

    >>> ops.nanargmin(x)
    array(5, dtype=int32)

    >>> ops.nanargmin(x, axis=0)
    array([0, 1, 1], dtype=int32)

    >>> ops.nanargmin(x, axis=1)
    array([0, 2], dtype=int32)

    >>> ops.nanargmin(x, axis=1, keepdims=True)
    array([[0],
           [2]], dtype=int32)
    """

    if any_symbolic_tensors((x,)):
        return Nanargmin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanargmin(x, axis=axis, keepdims=keepdims)


def _nanargmin(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanargmin"
    ):
        return backend.ops.numpy.nanargmin(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if not is_float_dtype(x.dtype):
        return backend.ops.numpy.argmin(x, axis=axis, keepdims=keepdims)
    nan_mask = backend.ops.numpy.isnan(x)
    pos_inf = backend.ops.cast(float("inf"), x.dtype)
    result = backend.ops.numpy.argmin(
        backend.ops.numpy.where(nan_mask, pos_inf, x),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.where(
        backend.ops.numpy.all(nan_mask, axis=axis, keepdims=keepdims),
        backend.ops.cast(-1, result.dtype),
        result,
    )


class Nancumsum(Operation):
    def __init__(self, axis=None, dtype=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return _nancumsum(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (python_math.prod(x.shape),)
        else:
            output_shape = x.shape

        output_dtype = (
            standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        )

        if output_dtype == "bool":
            output_dtype = "int32"

        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.nancumsum", "keras.ops.numpy.nancumsum"])
def nancumsum(x, axis=None, dtype=None):
    """Returns the cumulative sum of elements along a given axis,
    treating NaNs as zero.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nancumsum(x)
    array([1., 1., 4., 4., 6., 7.])

    >>> ops.nancumsum(x, axis=1)
    array([[1., 1., 4.],
           [0., 2., 3.]])
    """
    if any_symbolic_tensors((x,)):
        return Nancumsum(axis=axis, dtype=dtype).symbolic_call(x)
    return _nancumsum(x, axis=axis, dtype=dtype)


def _nancumsum(x, axis=None, dtype=None):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nancumsum"
    ):
        return backend.ops.numpy.nancumsum(x, axis=axis, dtype=dtype)
    x = backend.ops.convert_to_tensor(x)
    if is_float_dtype(x.dtype):
        x = backend.ops.numpy.where(
            backend.ops.numpy.isnan(x), backend.ops.cast(0, x.dtype), x
        )
    return backend.ops.numpy.cumsum(x, axis=axis, dtype=dtype)


class Nancumprod(Operation):
    def __init__(self, axis=None, dtype=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return _nancumprod(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (python_math.prod(x.shape),)
        else:
            output_shape = x.shape

        output_dtype = (
            standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        )

        if output_dtype == "bool":
            output_dtype = "int32"

        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.nancumprod", "keras.ops.numpy.nancumprod"])
def nancumprod(x, axis=None, dtype=None):
    """Returns the cumulative product of elements along a given axis,
    treating NaNs as one.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative product is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nancumprod(x)
    array([1., 1., 3., 3., 6., 6.])

    >>> ops.nancumprod(x, axis=1)
    array([[1., 1., 3.],
           [1., 2., 2.]])
    """
    if any_symbolic_tensors((x,)):
        return Nancumprod(axis=axis, dtype=dtype).symbolic_call(x)
    return _nancumprod(x, axis=axis, dtype=dtype)


def _nancumprod(x, axis=None, dtype=None):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nancumprod"
    ):
        return backend.ops.numpy.nancumprod(x, axis=axis, dtype=dtype)
    x = backend.ops.convert_to_tensor(x)
    if is_float_dtype(x.dtype):
        x = backend.ops.numpy.where(
            backend.ops.numpy.isnan(x), backend.ops.cast(1, x.dtype), x
        )
    return backend.ops.numpy.cumprod(x, axis=axis, dtype=dtype)


class Nanmax(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanmax(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(getattr(x, "dtype", backend.floatx()))

        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"

        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.nanmax", "keras.ops.numpy.nanmax"])
def nanmax(x, axis=None, keepdims=False):
    """Maximum of a tensor over the given axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the maximum is computed.
            The default is to compute the maximum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults
            to `False`.

    Returns:
        Output tensor containing the maximum, with NaN values ignored. If all
        values along a reduced axis are NaN, the result is NaN.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nanmax(x)
    3.0

    >>> ops.nanmax(x, axis=1)
    array([3., 2.])

    >>> ops.nanmax(x, axis=1, keepdims=True)
    array([[3.],
           [2.]])
    """
    if any_symbolic_tensors((x,)):
        return Nanmax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanmax(x, axis=axis, keepdims=keepdims)


def _nanmax(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanmax"
    ):
        return backend.ops.numpy.nanmax(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if not is_float_dtype(x.dtype):
        return backend.ops.numpy.max(x, axis=axis, keepdims=keepdims)
    nan_mask = backend.ops.numpy.isnan(x)
    neg_inf = backend.ops.cast(float("-inf"), x.dtype)
    nan = backend.ops.cast(float("nan"), x.dtype)
    result = backend.ops.numpy.max(
        backend.ops.numpy.where(nan_mask, neg_inf, x),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.where(
        backend.ops.numpy.all(nan_mask, axis=axis, keepdims=keepdims),
        nan,
        result,
    )


class Nanmean(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanmean(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.nanmean", "keras.ops.numpy.nanmean"])
def nanmean(x, axis=None, keepdims=False):
    """Mean of a tensor over the given axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the mean is computed.
            The default is to compute the mean of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults
            to `False`.

    Returns:
        Output tensor containing the mean, with NaN values ignored.
        If all values along a reduced axis are NaN, the result is NaN.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nanmean(x)
    1.75

    >>> ops.nanmean(x, axis=1)
    array([2., 1.5])

    >>> ops.nanmean(x, axis=1, keepdims=True)
    array([[2. ],
           [1.5]])
    """
    if any_symbolic_tensors((x,)):
        return Nanmean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanmean(x, axis=axis, keepdims=keepdims)


def _nanmean(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanmean"
    ):
        return backend.ops.numpy.nanmean(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    x = backend.ops.cast(x, dtype)
    if axis == () or axis == []:
        return x
    total = _nansum(x, axis=axis, keepdims=keepdims)
    count = backend.ops.numpy.sum(
        backend.ops.cast(
            backend.ops.numpy.logical_not(backend.ops.numpy.isnan(x)), dtype
        ),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.divide(total, count)


class Nanmedian(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanmedian(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.nanmedian", "keras.ops.numpy.nanmedian"])
def nanmedian(x, axis=None, keepdims=False):
    """Median of a tensor over the given axes, ignoring NaNs.

    This function computes the median along the specified axis or axes,
    skipping any NaN values. If all values along a reduced axis are NaN,
    the result is NaN.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the median is computed.
            If None (default), the median of the flattened tensor is returned.
        keepdims: If True, the reduced axes are retained as dimensions
            with size one. Defaults to False.

    Returns:
        Tensor with the median values, ignoring NaNs.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nanmedian(x)
    1.5

    >>> ops.nanmedian(x, axis=1)
    array([2., 1.5])

    >>> ops.nanmedian(x, axis=1, keepdims=True)
    array([[2. ],
           [1.5]])
    """
    if any_symbolic_tensors((x,)):
        return Nanmedian(axis=axis, keepdims=keepdims).symbolic_call(x)

    return _nanmedian(x, axis=axis, keepdims=keepdims)


def _nanmedian(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanmedian"
    ):
        return backend.ops.numpy.nanmedian(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if axis == () or axis == []:
        return backend.ops.cast(x, dtypes.result_type(x.dtype, float))
    return _nanquantile(x, 0.5, axis=axis, keepdims=keepdims)


class Nanmin(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanmin(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(getattr(x, "dtype", backend.floatx()))

        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"

        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.nanmin", "keras.ops.numpy.nanmin"])
def nanmin(x, axis=None, keepdims=False):
    """Minimum of a tensor over the given axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the minimum is computed.
            The default is to compute the minimum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the minimum, with NaN values ignored. If all
        values along a reduced axis are NaN, the result is NaN.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nanmin(x)
    1.0

    >>> ops.nanmin(x, axis=1)
    array([1., 1.])

    >>> ops.nanmin(x, axis=1, keepdims=True)
    array([[1.],
           [1.]])
    """
    if any_symbolic_tensors((x,)):
        return Nanmin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanmin(x, axis=axis, keepdims=keepdims)


def _nanmin(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanmin"
    ):
        return backend.ops.numpy.nanmin(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if not is_float_dtype(x.dtype):
        return backend.ops.numpy.min(x, axis=axis, keepdims=keepdims)
    nan_mask = backend.ops.numpy.isnan(x)
    pos_inf = backend.ops.cast(float("inf"), x.dtype)
    nan = backend.ops.cast(float("nan"), x.dtype)
    result = backend.ops.numpy.min(
        backend.ops.numpy.where(nan_mask, pos_inf, x),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.where(
        backend.ops.numpy.all(nan_mask, axis=axis, keepdims=keepdims),
        nan,
        result,
    )


class Nanpercentile(Operation):
    def __init__(
        self, axis=None, method="linear", keepdims=False, *, name=None
    ):
        super().__init__(name=name)
        self.axis = axis
        self.method = method
        self.keepdims = keepdims

    def call(self, x, q):
        return _nanpercentile(
            x, q, axis=self.axis, method=self.method, keepdims=self.keepdims
        )

    def compute_output_spec(self, x, q):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )

        if hasattr(q, "shape"):
            if len(q.shape) > 0:
                output_shape = (q.shape[0],) + output_shape
        elif isinstance(q, (list, tuple)):
            output_shape = (len(q),) + output_shape

        if standardize_dtype(x.dtype) == "int64":
            dtype = config.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)

        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.nanpercentile", "keras.ops.numpy.nanpercentile"])
def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th percentile(s) of the data along the specified axis,
    while ignoring NaNs.

    Args:
        x: Input tensor.
        q: Percentile or sequence of percentiles to compute.
            Values must be between 0 and 100 inclusive.
        axis: Axis or axes along which the percentiles are computed.
        method: A string specifies the method to use for estimating the
            percentile. Available methods are `"linear"`, `"lower"`, `"higher"`,
            `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
            If the desired percentile lies between two data points `i < j`:
            - `"linear"`: `i + (j - i) * fraction`, where fraction is the
                fractional part of the index surrounded by `i` and `j`.
            - `"lower"`: `i`.
            - `"higher"`: `j`.
            - `"midpoint"`: `(i + j) / 2`
            - `"nearest"`: `i` or `j`, whichever is nearest.
        keepdims: If True, reduced axes are kept with size 1.

    Returns:
        The percentile(s) ignoring NaNs.

    Examples:
    >>> import keras
    >>> from keras import ops
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.nanpercentile(x, 50)
    2.5
    >>> x = keras.ops.array([1., 2., float("nan"), 4.])
    >>> keras.ops.nanpercentile(x, 50)
    2.0
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.nanpercentile(x, [25, 75])
    array([1.75, 3.25])
    >>> x = keras.ops.array([[1., 2., float("nan")],
    ...                      [4., 5., 6.]])
    >>> keras.ops.nanpercentile(x, 50, axis=1)
    array([1.5, 5.0])
    """
    if any_symbolic_tensors((x, q)):
        return Nanpercentile(
            axis=axis, method=method, keepdims=keepdims
        ).symbolic_call(x, q)

    return _nanpercentile(x, q, axis=axis, method=method, keepdims=keepdims)


def _nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(
            "`method` must be one of 'linear', 'lower', 'higher', "
            f"'midpoint', or 'nearest'. Received: method={method}"
        )
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanpercentile"
    ):
        return backend.ops.numpy.nanpercentile(
            x, q, axis=axis, method=method, keepdims=keepdims
        )
    q = backend.ops.numpy.divide(
        backend.ops.convert_to_tensor(q, dtype=config.floatx()), 100.0
    )
    return _nanquantile(x, q, axis=axis, method=method, keepdims=keepdims)


class Nanprod(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanprod(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)

        if dtype == "bool":
            dtype = "int32"
        elif dtype in ("int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"

        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"

        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.nanprod", "keras.ops.numpy.nanprod"])
def nanprod(x, axis=None, keepdims=False):
    """Product of a tensor over the given axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the product is computed. The default is
            to compute the product of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the product, with NaN values ignored.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nanprod(x)
    6.0

    >>> ops.nanprod(x, axis=1)
    array([3., 2.])

    >>> ops.nanprod(x, axis=1, keepdims=True)
    array([[3.],
           [2.]])
    """
    if any_symbolic_tensors((x,)):
        return Nanprod(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanprod(x, axis=axis, keepdims=keepdims)


def _nanprod(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanprod"
    ):
        return backend.ops.numpy.nanprod(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if is_float_dtype(x.dtype):
        x = backend.ops.numpy.where(
            backend.ops.numpy.isnan(x), backend.ops.cast(1, x.dtype), x
        )
    return backend.ops.numpy.prod(x, axis=axis, keepdims=keepdims)


class Nanquantile(Operation):
    def __init__(
        self, axis=None, method="linear", keepdims=False, *, name=None
    ):
        super().__init__(name=name)
        self.axis = axis
        self.method = method
        self.keepdims = keepdims

    def call(self, x, q):
        return _nanquantile(
            x, q, axis=self.axis, method=self.method, keepdims=self.keepdims
        )

    def compute_output_spec(self, x, q):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )
        if hasattr(q, "shape"):
            if len(q.shape) > 0:
                output_shape = (q.shape[0],) + output_shape
        elif isinstance(q, (list, tuple)) and len(q) > 1:
            output_shape = (len(q),) + output_shape

        if standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)

        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.nanquantile", "keras.ops.numpy.nanquantile"])
def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th quantile(s) of the data along the specified axis,
    while ignoring NaNs.

    Args:
        x: Input tensor.
        q: Probability or sequence of probabilities for the quantiles to
            compute. Values must be between 0 and 1 inclusive.
        axis: Axis or axes along which the quantiles are computed. Defaults to
            `axis=None` which is to compute the quantile(s) along a flattened
            version of the array.
        method: A string specifies the method to use for estimating the
            quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,
            `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
            If the desired quantile lies between two data points `i < j`:
            - `"linear"`: `i + (j - i) * fraction`, where fraction is the
                fractional part of the index surrounded by `i` and `j`.
            - `"lower"`: `i`.
            - `"higher"`: `j`.
            - `"midpoint"`: `(i + j) / 2`
            - `"nearest"`: `i` or `j`, whichever is nearest.
        keepdims: If this is set to `True`, the axes which are reduced
            are left in the result as dimensions with size one.

    Returns:
        The quantile(s) ignoring NaNs.

    Examples:
    >>> import keras
    >>> from keras import ops
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.nanquantile(x, 0.5)
    2.5
    >>> x = keras.ops.array([1., 2., float("nan"), 4.])
    >>> keras.ops.nanquantile(x, 0.5)
    2.0
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.nanquantile(x, [0.25, 0.75])
    array([1.75, 3.25])
    >>> x = keras.ops.array([[1., 2., float("nan")],
    ...                      [4., 5., 6.]])
    >>> keras.ops.nanquantile(x, 0.5, axis=1)
    array([1.5, 5.0])
    """
    if any_symbolic_tensors((x, q)):
        return Nanquantile(
            axis=axis, method=method, keepdims=keepdims
        ).symbolic_call(x, q)

    return _nanquantile(x, q, axis=axis, method=method, keepdims=keepdims)


def _nanquantile(x, q, axis=None, method="linear", keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanquantile"
    ):
        return backend.ops.numpy.nanquantile(
            x, q, axis=axis, method=method, keepdims=keepdims
        )
    if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(
            "`method` must be one of 'linear', 'lower', 'higher', "
            f"'midpoint', or 'nearest'. Received: method={method}"
        )
    x = backend.ops.convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = backend.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = backend.ops.cast(x, dtype)
    q = backend.ops.convert_to_tensor(q, dtype=dtype)
    x_shape = backend.ops.shape(x)
    ndim = len(x_shape)
    output_shape = tuple(q.shape) + tuple(
        reduce_shape(x_shape, axis=axis, keepdims=keepdims)
    )

    # Move the reduced axes to the end and flatten them into one axis.
    axes = tuple(range(ndim)) if axis is None else canonicalize_axes(axis, ndim)
    y = backend.ops.numpy.moveaxis(x, axes, tuple(range(-len(axes), 0)))
    y = backend.ops.numpy.reshape(y, reduce_shape(x_shape, axis=axes) + (-1,))

    # Replace NaNs with +inf so they sort to the end of each slice, then
    # count the valid (non-NaN) entries at the front of each slice.
    nan_mask = backend.ops.numpy.isnan(y)
    inf = backend.ops.cast(float("inf"), dtype)
    sorted_y = backend.ops.numpy.sort(
        backend.ops.numpy.where(nan_mask, inf, y), axis=-1
    )
    valid_count = backend.ops.numpy.sum(
        backend.ops.cast(backend.ops.numpy.logical_not(nan_mask), dtype),
        axis=-1,
        keepdims=True,
    )

    last_valid_index = backend.ops.numpy.maximum(
        backend.ops.numpy.subtract(valid_count, 1), 0
    )
    position = backend.ops.numpy.multiply(
        last_valid_index, backend.ops.numpy.reshape(q, (-1,))
    )
    position = backend.ops.numpy.minimum(
        backend.ops.numpy.maximum(position, 0), last_valid_index
    )

    def take(indices):
        indices = backend.ops.cast(indices, "int32")
        return backend.ops.numpy.take_along_axis(sorted_y, indices, axis=-1)

    if method == "lower":
        result = take(backend.ops.numpy.floor(position))
    elif method == "higher":
        result = take(backend.ops.numpy.ceil(position))
    elif method == "nearest":
        result = take(backend.ops.numpy.round(position))
    else:
        lower_index = backend.ops.numpy.floor(position)
        lower = take(lower_index)
        upper = take(backend.ops.numpy.ceil(position))
        if method == "midpoint":
            result = backend.ops.numpy.multiply(
                backend.ops.numpy.add(lower, upper), 0.5
            )
        else:  # linear
            fraction = backend.ops.numpy.subtract(position, lower_index)
            interpolated = backend.ops.numpy.add(
                lower,
                backend.ops.numpy.multiply(
                    backend.ops.numpy.subtract(upper, lower), fraction
                ),
            )
            result = backend.ops.numpy.where(
                backend.ops.numpy.equal(lower, upper), lower, interpolated
            )

    # Slices with no valid entries yield NaN.
    nan = backend.ops.cast(float("nan"), dtype)
    result = backend.ops.numpy.where(
        backend.ops.numpy.equal(valid_count, 0), nan, result
    )

    result = backend.ops.numpy.moveaxis(result, -1, 0)
    return backend.ops.numpy.reshape(result, output_shape)


class Nanstd(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanstd(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = backend.result_type(getattr(x, "dtype", type(x)), float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.nanstd", "keras.ops.numpy.nanstd"])
def nanstd(x, axis=None, keepdims=False):
    """Compute the standard deviation along the specified axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the standard deviation is computed.
            The default is to compute the std of the flattened tensor.
        keepdims: If `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Output tensor containing the standard deviation ignoring NaNs.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])

    >>> ops.nanstd(x)
    0.8291562

    >>> ops.nanstd(x, axis=1)
    array([1. , 0.5])

    >>> ops.nanstd(x, axis=1, keepdims=True)
    array([[1. ],
           [0.5]])
    """
    if any_symbolic_tensors((x,)):
        return Nanstd(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanstd(x, axis=axis, keepdims=keepdims)


def _nanstd(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanstd"
    ):
        return backend.ops.numpy.nanstd(x, axis=axis, keepdims=keepdims)
    return backend.ops.numpy.sqrt(_nanvar(x, axis=axis, keepdims=keepdims))


class Nansum(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nansum(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(getattr(x, "dtype", backend.floatx()))

        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"

        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"
        sparse = getattr(x, "sparse", False)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
            sparse=sparse,
        )


@keras_export(["keras.ops.nansum", "keras.ops.numpy.nansum"])
def nansum(x, axis=None, keepdims=False):
    """Sum of a tensor over the given axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the sum is computed. The default is to
            compute the sum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the sum, with NaN values ignored.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])
    >>> ops.nansum(x)
    7.0

    >>> ops.nansum(x, axis=1)
    array([4., 3.])

    >>> ops.nansum(x, axis=1, keepdims=True)
    array([[4.],
           [3.]])
    """
    if any_symbolic_tensors((x,)):
        return Nansum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nansum(x, axis=axis, keepdims=keepdims)


def _nansum(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nansum"
    ):
        return backend.ops.numpy.nansum(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    if is_float_dtype(x.dtype):
        x = backend.ops.numpy.where(backend.ops.numpy.isnan(x), 0, x)
    return backend.ops.numpy.sum(x, axis=axis, keepdims=keepdims)


class Nanvar(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return _nanvar(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = backend.result_type(getattr(x, "dtype", type(x)), float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.nanvar", "keras.ops.numpy.nanvar"])
def nanvar(x, axis=None, keepdims=False):
    """Compute the variance along the specified axes, ignoring NaNs.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed. The default
            is to compute the variance of the flattened tensor.
        keepdims: If `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Output tensor containing the variance ignoring NaNs.

    Examples:
    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.array([[1.0, np.nan, 3.0],
    ...               [np.nan, 2.0, 1.0]])

    >>> ops.nanvar(x)
    0.6875

    >>> ops.nanvar(x, axis=1)
    array([1.  , 0.25])

    >>> ops.nanvar(x, axis=1, keepdims=True)
    array([[1.  ],
           [0.25]])
    """
    if any_symbolic_tensors((x,)):
        return Nanvar(axis=axis, keepdims=keepdims).symbolic_call(x)
    return _nanvar(x, axis=axis, keepdims=keepdims)


def _nanvar(x, axis=None, keepdims=False):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "nanvar"
    ):
        return backend.ops.numpy.nanvar(x, axis=axis, keepdims=keepdims)
    x = backend.ops.convert_to_tensor(x)
    dtype = backend.result_type(x.dtype, float)
    x = backend.ops.cast(x, dtype)
    nan_mask = backend.ops.numpy.isnan(x)
    mean = _nanmean(x, axis=axis, keepdims=True)
    centered = backend.ops.numpy.where(
        nan_mask,
        backend.ops.cast(0, dtype),
        backend.ops.numpy.subtract(x, mean),
    )
    sum_sq = backend.ops.numpy.sum(
        backend.ops.numpy.square(centered), axis=axis, keepdims=keepdims
    )
    count = backend.ops.numpy.sum(
        backend.ops.cast(backend.ops.numpy.logical_not(nan_mask), dtype),
        axis=axis,
        keepdims=keepdims,
    )
    return backend.ops.numpy.divide(sum_sq, count)


class NanToNum(Operation):
    def __init__(self, nan=0.0, posinf=None, neginf=None, *, name=None):
        super().__init__(name=name)
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def call(self, x):
        return backend.ops.numpy.nan_to_num(
            x, nan=self.nan, posinf=self.posinf, neginf=self.neginf
        )

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.nan_to_num",
        "keras.ops.numpy.nan_to_num",
    ]
)
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers.

    Args:
        x: Input data.
        nan: Optional float or int. Value to replace `NaN` entries with.
        posinf: Optional float or int.
            Value to replace positive infinity with.
        neginf: Optional float or int.
            Value to replace negative infinity with.

    Returns:
        `x`, with non-finite values replaced.
    """
    if any_symbolic_tensors((x,)):
        return NanToNum(nan=nan, posinf=posinf, neginf=neginf).symbolic_call(x)
    return backend.ops.numpy.nan_to_num(
        x, nan=nan, posinf=posinf, neginf=neginf
    )


@keras_export(["keras.ops.ndim", "keras.ops.numpy.ndim"])
def ndim(x):
    """Return the number of dimensions of a tensor.

    Args:
        x: Input tensor.

    Returns:
        The number of dimensions in `x`.
    """
    if any_symbolic_tensors((x,)):
        # Unlike individual dimension sizes, the rank of a symbolic tensor
        # is always statically known, so this never needs a backend op.
        return len(x.shape)
    return backend.ops.numpy.ndim(x)


class Nonzero(Operation):
    def call(self, x):
        return backend.ops.numpy.nonzero(x)

    def compute_output_spec(self, x):
        return tuple(
            [KerasTensor((None,), dtype="int32") for _ in range(len(x.shape))]
        )


@keras_export(["keras.ops.nonzero", "keras.ops.numpy.nonzero"])
def nonzero(x):
    """Return the indices of the elements that are non-zero.

    Args:
        x: Input tensor.

    Returns:
        Indices of elements that are non-zero.
    """
    if any_symbolic_tensors((x,)):
        return Nonzero().symbolic_call(x)
    return backend.ops.numpy.nonzero(x)


class NotEqual(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.not_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.not_equal", "keras.ops.numpy.not_equal"])
def not_equal(x1, x2):
    """Return `(x1 != x2)` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return NotEqual().symbolic_call(x1, x2)
    return backend.ops.numpy.not_equal(x1, x2)


class OnesLike(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.ones_like(x, dtype=self.dtype)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.ones_like", "keras.ops.numpy.ones_like"])
def ones_like(x, dtype=None):
    """Return a tensor of ones with the same shape and type of `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of ones with the same shape and type as `x`.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return OnesLike(dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.ones_like(x, dtype=dtype)


class ZerosLike(Operation):
    def __init__(self, dtype=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.zeros_like(x, dtype=self.dtype)

    def compute_output_spec(self, x, dtype=None):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(
    [
        "keras.ops.zeros_like",
        "keras.ops.numpy.zeros_like",
    ]
)
def zeros_like(x, dtype=None):
    """Return a tensor of zeros with the same shape and type as `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of zeros with the same shape and type as `x`.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return ZerosLike(dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.zeros_like(x, dtype=dtype)


class Outer(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.outer(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [1])
        x2_shape = getattr(x2, "shape", [1])
        if None in x1_shape:
            x1_flatten_shape = None
        else:
            x1_flatten_shape = python_math.prod(x1_shape)
        if None in x2_shape:
            x2_flatten_shape = None
        else:
            x2_flatten_shape = python_math.prod(x2_shape)
        output_shape = (x1_flatten_shape, x2_flatten_shape)
        output_dtype = backend.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.outer", "keras.ops.numpy.outer"])
def outer(x1, x2):
    """Compute the outer product of two vectors.

    Given two vectors `x1` and `x2`, the outer product is:

    ```
    out[i, j] = x1[i] * x2[j]
    ```

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Outer product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Outer().symbolic_call(x1, x2)
    return backend.ops.numpy.outer(x1, x2)


class Pad(Operation):
    def __init__(self, pad_width, mode="constant", *, name=None):
        super().__init__(name=name)
        self.pad_width = self._process_pad_width(pad_width)
        self.mode = mode

    def _process_pad_width(self, pad_width):
        if isinstance(pad_width, int):
            return ((pad_width, pad_width),)
        if isinstance(pad_width, (tuple, list)) and isinstance(
            pad_width[0], int
        ):
            if len(pad_width) == 1:
                # A single `(pad,)` means pad before and after, like `np.pad`.
                return ((pad_width[0], pad_width[0]),)
            return (tuple(pad_width),)
        first_len = len(pad_width[0])
        for i, pw in enumerate(pad_width):
            if len(pw) != first_len:
                raise ValueError(
                    "`pad_width` should be a list of tuples of length "
                    f"1 or 2. Received: pad_width={pad_width}"
                )
            if len(pw) == 1:
                pad_width[i] = (pw[0], pw[0])
        return pad_width

    def call(self, x, constant_values=None):
        if len(self.pad_width) > 1 and len(self.pad_width) != len(x.shape):
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`. "
                f"Received: pad_width={self.pad_width} "
                f"(of length {len(self.pad_width)}) and x.shape={x.shape} "
                f"(of length {len(x.shape)})"
            )
        return backend.ops.numpy.pad(
            x,
            pad_width=self.pad_width,
            mode=self.mode,
            constant_values=constant_values,
        )

    def compute_output_spec(self, x, constant_values=None):
        output_shape = list(x.shape)
        if len(self.pad_width) == 1:
            pad_width = [self.pad_width[0] for _ in range(len(output_shape))]
        elif len(self.pad_width) == len(output_shape):
            pad_width = self.pad_width
        else:
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`. "
                f"Received: pad_width={self.pad_width} "
                f"(of length {len(self.pad_width)}) and x.shape={x.shape} "
                f"(of length {len(x.shape)})"
            )

        for i in range(len(output_shape)):
            if output_shape[i] is None:
                output_shape[i] = None
            else:
                output_shape[i] += pad_width[i][0] + pad_width[i][1]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.pad", "keras.ops.numpy.pad"])
def pad(x, pad_width, mode="constant", constant_values=None):
    """Pad a tensor.

    Args:
        x: Tensor to pad.
        pad_width: Number of values padded to the edges of each axis.
            `((before_1, after_1), ...(before_N, after_N))` unique pad
            widths for each axis.
            `((before, after),)` yields same before and after pad for
            each axis.
            `(pad,)` or `int` is a shortcut for `before = after = pad`
            width for all axes.
        mode: One of `"constant"`, `"edge"`, `"linear_ramp"`,
            `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
            `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
            `"circular"`. Defaults to `"constant"`.
        constant_values: value to pad with if `mode == "constant"`.
            Defaults to `0`. A `ValueError` is raised if not None and
            `mode != "constant"`.

    Note:
        Torch backend only supports modes `"constant"`, `"reflect"`,
        `"symmetric"` and `"circular"`.
        Only Torch backend supports `"circular"` mode.

    Note:
        Tensorflow backend only supports modes `"constant"`, `"reflect"`
        and `"symmetric"`.

    Returns:
        Padded tensor.
    """
    return Pad(pad_width, mode=mode)(x, constant_values=constant_values)


class Percentile(Operation):
    def __init__(
        self, axis=None, method="linear", keepdims=False, *, name=None
    ):
        super().__init__(name=name)
        self.axis = axis
        self.method = method
        self.keepdims = keepdims

    def call(self, x, q):
        return backend.ops.numpy.percentile(
            x, q, axis=self.axis, method=self.method, keepdims=self.keepdims
        )

    def compute_output_spec(self, x, q):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )

        if hasattr(q, "shape"):
            if len(q.shape) > 0:
                output_shape = (q.shape[0],) + output_shape
        elif isinstance(q, (list, tuple)):
            output_shape = (len(q),) + output_shape

        if not is_float_dtype(x.dtype):
            dtype = backend.floatx()
        else:
            dtype = x.dtype
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.percentile", "keras.ops.numpy.percentile"])
def percentile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th percentile(s) of the data along the specified axis.

    Args:
        x: Input tensor.
        q: Percentile or sequence of percentiles to compute.
            Values must be between 0 and 100 inclusive.
        axis: Axis or axes along which the percentiles are computed.
        method: A string specifies the method to use for estimating the
            percentile. Available methods are `"linear"`, `"lower"`, `"higher"`,
            `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
            If the desired percentile lies between two data points `i < j`:
            - `"linear"`: `i + (j - i) * fraction`, where fraction is the
                fractional part of the index surrounded by `i` and `j`.
            - `"lower"`: `i`.
            - `"higher"`: `j`.
            - `"midpoint"`: `(i + j) / 2`
            - `"nearest"`: `i` or `j`, whichever is nearest.
        keepdims: If True, reduced axes are kept with size 1.

    Returns:
        The percentile(s). If `q` is a single percentile and `axis=None`, then
        the result is a scalar. If multiple percentile levels are given, the
        first axis of the result corresponds to the percentiles. The other axes
        are the axes that remain after the reduction of `x`.

    Examples:
    >>> import keras
    >>> from keras import ops
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.percentile(x, 50)
    2.5
    >>> x = keras.ops.array([1., 2., 3., 4.])
    >>> keras.ops.percentile(x, [25, 75])
    array([1.75, 3.25])
    >>> x = keras.ops.array([[1., 2., 3.],
    ...                      [4., 5., 6.]])
    >>> keras.ops.percentile(x, 50, axis=1)
    array([2., 5.])
    """

    if any_symbolic_tensors((x, q)):
        return Percentile(
            axis=axis, method=method, keepdims=keepdims
        ).symbolic_call(x, q)

    return backend.ops.numpy.percentile(
        x, q, axis=axis, method=method, keepdims=keepdims
    )


class Prod(Operation):
    def __init__(self, axis=None, keepdims=False, dtype=None, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype

    def call(self, x):
        return backend.ops.numpy.prod(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
            dtype=self.dtype,
        )

    def compute_output_spec(self, x):
        if self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = standardize_dtype(x.dtype)
            if dtype == "bool":
                dtype = "int32"
            elif dtype in ("int8", "int16"):
                dtype = "int32"
            elif dtype in ("uint8", "uint16"):
                dtype = "uint32"
        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.prod", "keras.ops.numpy.prod"])
def prod(x, axis=None, keepdims=False, dtype=None):
    """Return the product of tensor elements over a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which a product is performed. The default,
            `axis=None`, will compute the product of all elements
            in the input tensor.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.
        dtype: Data type of the returned tensor.

    Returns:
        Product of elements of `x` over the given axis or axes.
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return Prod(axis=axis, keepdims=keepdims, dtype=dtype).symbolic_call(x)
    return backend.ops.numpy.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


class Ptp(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.ptp(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.ptp", "keras.ops.numpy.ptp"])
def ptp(x, axis=None, keepdims=False):
    """Return the peak-to-peak (max - min) value of tensor elements
    over a given axis.

    The peak-to-peak value is defined as the difference between the
    maximum and minimum values along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the peak-to-peak value is computed.
            The default, `axis=None`, will compute the peak-to-peak value
            over all elements in the input tensor.
        keepdims: If this is set to `True`, the axes which are reduced
            are left in the result as dimensions with size one.

    Returns:
        A tensor containing the peak-to-peak values of `x` over the
        given axis or axes.

    Examples:
    >>> x = keras.ops.array([[1., 3., 2.],
    ...                      [4., 0., 5.]])

    >>> # Peak-to-peak over all elements
    >>> keras.ops.ptp(x)
    5.0

    >>> # Peak-to-peak along axis 1
    >>> keras.ops.ptp(x, axis=1)
    array([2., 5.], dtype=float32)

    >>> # Peak-to-peak over multiple axes
    >>> x = keras.ops.reshape(x, (1, 2, 3))
    >>> keras.ops.ptp(x, axis=(1, 2))
    array([5.], dtype=float32)

    >>> # Keep reduced dimensions
    >>> keras.ops.ptp(x, axis=2, keepdims=True)
    array([[[2.],
            [5.]]], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Ptp(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.ptp(x, axis=axis, keepdims=keepdims)


class Quantile(Operation):
    def __init__(
        self, axis=None, method="linear", keepdims=False, *, name=None
    ):
        super().__init__(name=name)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.method = method
        self.keepdims = keepdims

    def call(self, x, q):
        return backend.ops.numpy.quantile(
            x, q, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x, q):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )
        if hasattr(q, "shape"):
            if len(q.shape) > 0:
                output_shape = (q.shape[0],) + output_shape
        elif isinstance(q, (list, tuple)) and len(q) > 1:
            output_shape = (len(q),) + output_shape
        if standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.quantile", "keras.ops.numpy.quantile"])
def quantile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th quantile(s) of the data along the specified axis.

    Args:
        x: Input tensor.
        q: Probability or sequence of probabilities for the quantiles to
            compute. Values must be between 0 and 1 inclusive.
        axis: Axis or axes along which the quantiles are computed. Defaults to
            `axis=None` which is to compute the quantile(s) along a flattened
            version of the array.
        method: A string specifies the method to use for estimating the
            quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,
            `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
            If the desired quantile lies between two data points `i < j`:
            - `"linear"`: `i + (j - i) * fraction`, where fraction is the
                fractional part of the index surrounded by `i` and `j`.
            - `"lower"`: `i`.
            - `"higher"`: `j`.
            - `"midpoint"`: `(i + j) / 2`
            - `"nearest"`: `i` or `j`, whichever is nearest.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.

    Returns:
        The quantile(s). If `q` is a single probability and `axis=None`, then
        the result is a scalar. If multiple probabilities levels are given,
        first axis of the result corresponds to the quantiles. The other axes
        are the axes that remain after the reduction of `x`.
    """
    if any_symbolic_tensors((x, q)):
        return Quantile(
            axis=axis, method=method, keepdims=keepdims
        ).symbolic_call(x, q)
    return backend.ops.numpy.quantile(
        x, q, axis=axis, method=method, keepdims=keepdims
    )


class Ravel(Operation):
    def call(self, x):
        return backend.ops.numpy.ravel(x)

    def compute_output_spec(self, x):
        if None in x.shape:
            output_shape = (None,)
        else:
            output_shape = (python_math.prod(x.shape),)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.ravel", "keras.ops.numpy.ravel"])
def ravel(x):
    """Return a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Ravel().symbolic_call(x)
    return backend.ops.numpy.ravel(x)


class UnravelIndex(Operation):
    def __init__(self, shape, *, name=None):
        super().__init__(name=name)
        self.shape = shape

    def call(self, indices):
        return backend.ops.numpy.unravel_index(indices, self.shape)

    def compute_output_spec(self, indices):
        if None in self.shape:
            output_shapes = [[None] for _ in self.shape]
        else:
            if isinstance(indices, int):
                output_shapes = [[1] for _ in self.shape]
            elif hasattr(indices, "shape"):
                output_shapes = [list(indices.shape) for _ in self.shape]
            else:
                try:
                    indices_shape = np.shape(indices)
                    output_shapes = [list(indices_shape) for _ in self.shape]
                except Exception:
                    output_shapes = [[None] for _ in self.shape]

        return [
            KerasTensor(shape, dtype=indices.dtype) for shape in output_shapes
        ]


@keras_export(["keras.ops.unravel_index", "keras.ops.numpy.unravel_index"])
def unravel_index(indices, shape):
    """Convert flat indices to coordinate arrays in a given array shape.

    Args:
        indices: An integer or array of integers representing flat indices.
        shape: The shape of the array to unravel into.

    Returns:
        Tuple of arrays for each dimension with unraveled indices.

    Example:
    >>> indices = 5
    >>> shape = (3, 3)
    >>> unravel_index(indices, shape)
    (1, 2)  # 5 is at row 1, column 2 in a 3x3 array
    """
    if any_symbolic_tensors((indices,)):
        return UnravelIndex(shape).symbolic_call(indices)

    return backend.ops.numpy.unravel_index(indices, shape)


class Real(Operation):
    def call(self, x):
        return backend.ops.numpy.real(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.real", "keras.ops.numpy.real"])
def real(x):
    """Return the real part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The real component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Real().symbolic_call(x)
    return backend.ops.numpy.real(x)


class Reciprocal(Operation):
    def call(self, x):
        return backend.ops.numpy.reciprocal(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_export(
    [
        "keras.ops.reciprocal",
        "keras.ops.numpy.reciprocal",
    ]
)
def reciprocal(x):
    """Return the reciprocal of the argument, element-wise.

    Calculates `1/x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise reciprocal of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Reciprocal().symbolic_call(x)
    return backend.ops.numpy.reciprocal(x)


class Repeat(Operation):
    def __init__(self, repeats, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.repeats = repeats

    def call(self, x):
        return backend.ops.numpy.repeat(x, self.repeats, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        repeats = self.repeats
        if isinstance(repeats, int):
            repeats = [repeats]
        repeats_size = len(repeats)
        broadcast = repeats_size == 1

        if self.axis is None:
            if None in x_shape:
                return KerasTensor([None], dtype=x.dtype)

            x_flatten_size = python_math.prod(x_shape)
            if broadcast:
                output_shape = (x_flatten_size * repeats[0],)
            elif repeats_size != x_flatten_size:
                raise ValueError(
                    "Size of `repeats` and "
                    "dimensions of `x` after flattening should be compatible. "
                    f"Received: {repeats_size} and {x_flatten_size}"
                )
            else:
                output_shape = (builtins.sum(repeats),)
            return KerasTensor(output_shape, dtype=x.dtype)

        size_on_ax = x_shape[self.axis]
        if size_on_ax is None:
            return KerasTensor(x_shape, dtype=x.dtype)

        output_shape = x_shape
        if broadcast:
            output_shape[self.axis] = size_on_ax * repeats[0]
        elif size_on_ax != repeats_size:
            raise ValueError(
                "Size of `repeats` and "
                f"dimensions of `axis {self.axis} of x` should be compatible. "
                f"Received: {repeats_size} and {x_shape}"
            )
        else:
            output_shape[self.axis] = builtins.sum(repeats)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.repeat", "keras.ops.numpy.repeat"])
def repeat(x, repeats, axis=None):
    """Repeat each element of a tensor after themselves.

    Args:
        x: Input tensor.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values. By default, use
            the flattened input array, and return a flat output array.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Repeat(repeats, axis=axis).symbolic_call(x)
    return backend.ops.numpy.repeat(x, repeats, axis=axis)


class Reshape(Operation):
    def __init__(self, newshape, *, name=None):
        super().__init__(name=name)
        self.newshape = newshape

    def call(self, x):
        return backend.ops.numpy.reshape(x, self.newshape)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_reshape_output_shape(
            x.shape, self.newshape, "newshape"
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.reshape", "keras.ops.numpy.reshape"])
def reshape(x, newshape):
    """Gives a new shape to a tensor without changing its data.

    Args:
        x: Input tensor.
        newshape: The new shape should be compatible with the original shape.
            One shape dimension can be -1 in which case the value is
            inferred from the length of the array and remaining dimensions.

    Returns:
        The reshaped tensor.
    """
    newshape = operation_utils.standardize_reshape_shape(newshape)
    if any_symbolic_tensors((x, newshape)):
        return Reshape(newshape).symbolic_call(x)
    return backend.ops.numpy.reshape(x, newshape)


class Roll(Operation):
    def __init__(self, shift, axis=None, *, name=None):
        super().__init__(name=name)
        self.shift = shift
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.roll(x, self.shift, self.axis)

    def compute_output_spec(self, x):
        if self.axis is not None:
            _, axes = normalize_shift_and_axis(self.shift, self.axis)
            canonicalize_axes(axes, len(x.shape))
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.roll", "keras.ops.numpy.roll"])
def roll(x, shift, axis=None):
    """Roll tensor elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        x: Input tensor.
        shift: The number of places by which elements are shifted.
        axis: The axis along which elements are shifted. By default, the
            array is flattened before shifting, after which the original
            shape is restored.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Roll(shift, axis=axis).symbolic_call(x)
    return backend.ops.numpy.roll(x, shift, axis=axis)


class Round(Operation):
    def __init__(self, decimals=0, *, name=None):
        super().__init__(name=name)
        self.decimals = decimals

    def call(self, x):
        return backend.ops.numpy.round(x, self.decimals)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.round", "keras.ops.numpy.round"])
def round(x, decimals=0):
    """Evenly round to the given number of decimals.

    Args:
        x: Input tensor.
        decimals: Number of decimal places to round to. Defaults to `0`.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Round(decimals).symbolic_call(x)
    return backend.ops.numpy.round(x, decimals)


class SearchSorted(Operation):
    def __init__(self, side="left", *, name=None):
        super().__init__(name=name)
        self.side = side

    def call(self, sorted_sequence, values):
        sorted_sequence = backend.ops.convert_to_tensor(sorted_sequence)
        values = backend.ops.convert_to_tensor(values)
        return backend.ops.numpy.searchsorted(
            sorted_sequence, values, side=self.side
        )

    def compute_output_spec(self, sorted_sequence, values):
        if len(sorted_sequence.shape) != 1:
            raise ValueError(
                "searchsorted only supports 1-D sorted sequences. Use"
                "keras.ops.vectorized_map to extend to N-D sequences."
            )
        sequence_len = sorted_sequence.shape[0]
        out_type = (
            "int32"
            if sequence_len is not None
            and sequence_len <= np.iinfo(np.int32).max
            else "int64"
        )
        return KerasTensor(values.shape, dtype=out_type)


@keras_export(["keras.ops.searchsorted", "keras.ops.numpy.searchsorted"])
def searchsorted(sorted_sequence, values, side="left"):
    """Perform a binary search, returning indices for insertion of `values`
    into `sorted_sequence` that maintain the sorting order.

    Args:
        sorted_sequence: 1-D input tensor, sorted along the innermost
            dimension.
        values: N-D tensor of query insertion values.
        side: 'left' or 'right', specifying the direction in which to insert
            for the equality case (tie-breaker).

    Returns:
        Tensor of insertion indices of same shape as `values`.
    """
    if any_symbolic_tensors((sorted_sequence, values)):
        return SearchSorted(side=side).symbolic_call(sorted_sequence, values)

    sorted_sequence = backend.ops.convert_to_tensor(sorted_sequence)
    values = backend.ops.convert_to_tensor(values)
    return backend.ops.numpy.searchsorted(sorted_sequence, values, side=side)


class Sign(Operation):
    def call(self, x):
        return backend.ops.numpy.sign(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.sign", "keras.ops.numpy.sign"])
def sign(x):
    """Returns a tensor with the signs of the elements of `x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sign().symbolic_call(x)
    return backend.ops.numpy.sign(x)


class Signbit(Operation):
    def call(self, x):
        return backend.ops.numpy.signbit(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype="bool", sparse=sparse)


@keras_export(["keras.ops.signbit", "keras.ops.numpy.signbit"])
def signbit(x):
    """Return the sign bit of the elements of `x`.

    The output boolean tensor contains `True` where the sign of `x` is negative,
    and `False` otherwise.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Signbit().symbolic_call(x)
    return backend.ops.numpy.signbit(x)


class Sin(Operation):
    def call(self, x):
        return backend.ops.numpy.sin(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sin", "keras.ops.numpy.sin"])
def sin(x):
    """Trigonometric sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sin().symbolic_call(x)
    return backend.ops.numpy.sin(x)


class Sinc(Operation):
    def call(self, x):
        return backend.ops.numpy.sinc(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.sinc", "keras.ops.numpy.sinc"])
def sinc(x):
    """Return the normalized sinc function.

    The sinc function is equal to `sin(pi*x) / (pi*x)` for any argument
    `x != 0`, and `sinc(0)` takes the limit value 1, making `sinc` not
    just everywhere continuous but also infinitely differentiable.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0])
    >>> keras.ops.sinc(x)
    array([1., 0., 0.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Sinc().symbolic_call(x)
    return backend.ops.numpy.sinc(x)


class Sinh(Operation):
    def call(self, x):
        return backend.ops.numpy.sinh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sinh", "keras.ops.numpy.sinh"])
def sinh(x):
    """Hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sinh().symbolic_call(x)
    return backend.ops.numpy.sinh(x)


@keras_export(["keras.ops.size", "keras.ops.numpy.size"])
def size(x):
    """Return the number of elements in a tensor.

    Note: On the TensorFlow backend, when `x` is a `tf.Tensor` with dynamic
    shape, the size in the context of a compiled function will be a
    `tf.Tensor` instead of a static integer value.

    Args:
        x: Input tensor.

    Returns:
        An integer, the number of elements in `x`. When `x` is a symbolic
        tensor with an unknown dimension, returns `None`.
    """
    if any_symbolic_tensors((x,)):
        if None in x.shape:
            return None
        return python_math.prod(x.shape)
    return backend.ops.numpy.size(x)


class Sort(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.sort(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (python_math.prod(x.shape),)
            return KerasTensor(output_shape, x.dtype)
        canonicalize_axis(self.axis, len(x.shape))
        return KerasTensor(x.shape, x.dtype)


@keras_export(["keras.ops.sort", "keras.ops.numpy.sort"])
def sort(x, axis=-1):
    """Sorts the elements of `x` along a given axis in ascending order.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. If `None`, the tensor is flattened
            before sorting. Defaults to `-1`; the last axis.

    Returns:
        Sorted tensor.
    """
    if any_symbolic_tensors((x,)):
        return Sort(axis=axis).symbolic_call(x)
    return backend.ops.numpy.sort(x, axis=axis)


def _compute_split_output_spec(x, indices_or_sections, axis):
    x_shape = list(x.shape)
    axis = canonicalize_axis(axis, len(x_shape))
    x_size_on_axis = x_shape[axis]
    if isinstance(indices_or_sections, int):
        if x_size_on_axis is None:
            x_shape[axis] = None
            return [
                KerasTensor(x_shape, dtype=x.dtype)
                for _ in range(indices_or_sections)
            ]

        if np.mod(x_size_on_axis, indices_or_sections) != 0:
            raise ValueError(
                "`x` size on given `axis` must be divisible by "
                "`indices_or_sections` when `indices_or_sections` is an "
                f"int. But received {x_size_on_axis} and "
                f"{indices_or_sections}."
            )

        size = x_size_on_axis // indices_or_sections
        x_shape[axis] = size
        return [
            KerasTensor(x_shape, dtype=x.dtype)
            for _ in range(indices_or_sections)
        ]

    all_indices = [0] + list(indices_or_sections) + [x_size_on_axis]
    outputs = []

    for i in range(len(all_indices) - 1):
        start = all_indices[i]
        end = all_indices[i + 1]
        if start is None or end is None:
            output_size = None
        else:
            output_size = end - start
        output_shape = list(x_shape)
        output_shape[axis] = output_size
        outputs.append(KerasTensor(output_shape, dtype=x.dtype))

    return outputs


class Split(Operation):
    def __init__(self, indices_or_sections, axis=0, *, name=None):
        super().__init__(name=name)
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.split(
            x, self.indices_or_sections, axis=self.axis
        )

    def compute_output_spec(self, x):
        return _compute_split_output_spec(
            x, self.indices_or_sections, self.axis
        )


@keras_export(["keras.ops.split", "keras.ops.numpy.split"])
def split(x, indices_or_sections, axis=0):
    """Split a tensor into chunks.

    Args:
        x: Input tensor.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along `axis`. If a 1-D array of sorted integers,
            the entries indicate indices at which the tensor will be split
            along `axis`.
        axis: Axis along which to split. Defaults to `0`.

    Note:
        A split does not have to result in equal division when using
        Torch backend.

    Returns:
        A list of tensors.
    """
    if any_symbolic_tensors((x,)):
        return Split(indices_or_sections, axis=axis).symbolic_call(x)
    return backend.ops.numpy.split(x, indices_or_sections, axis=axis)


class Stack(Operation):
    def __init__(self, axis=0, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.stack(x, axis=self.axis)

    def compute_output_spec(self, x):
        first_shape = x[0].shape
        # `stack` adds a new axis, so the valid range is one larger than
        # the input rank.
        axis = canonicalize_axis(self.axis, len(first_shape) + 1)
        dtypes_to_resolve = []
        for a in x:
            if not shape_equal(a.shape, first_shape, axis=[], allow_none=True):
                raise ValueError(
                    "Every value in `x` must have the same shape. But found "
                    f"element of shape {a.shape},  which is different from the "
                    f"first element's shape {first_shape}."
                )
            dtypes_to_resolve.append(getattr(a, "dtype", type(a)))

        size_on_axis = len(x)
        output_shape = list(first_shape)
        output_shape.insert(axis, size_on_axis)
        output_dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.stack", "keras.ops.numpy.stack"])
def stack(x, axis=0):
    """Join a sequence of tensors along a new axis.

    The `axis` parameter specifies the index of the new axis in the
    dimensions of the result.

    Args:
        x: A sequence of tensors.
        axis: Axis along which to stack. Defaults to `0`.

    Returns:
        The stacked tensor.
    """
    if any_symbolic_tensors((x,)):
        return Stack(axis=axis).symbolic_call(x)
    return backend.ops.numpy.stack(x, axis=axis)


class Std(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.std(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = standardize_dtype(x.dtype)
        if "int" in output_dtype or output_dtype == "bool":
            output_dtype = backend.floatx()
        elif output_dtype == "complex64":
            output_dtype = "float32"
        elif output_dtype == "complex128":
            output_dtype = "float64"
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.std", "keras.ops.numpy.std"])
def std(x, axis=None, keepdims=False):
    """Compute the standard deviation along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute standard deviation.
            Default is to compute the standard deviation of the
            flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the standard deviation values.
    """
    if any_symbolic_tensors((x,)):
        return Std(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.std(x, axis=axis, keepdims=keepdims)


class Swapaxes(Operation):
    def __init__(self, axis1, axis2, *, name=None):
        super().__init__(name=name)

        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.ops.numpy.swapaxes(x, self.axis1, self.axis2)

    def compute_output_spec(self, x):
        ndim = len(x.shape)
        ax1 = canonicalize_axis(self.axis1, ndim)
        ax2 = canonicalize_axis(self.axis2, ndim)
        x_shape = list(x.shape)
        x_shape[ax1], x_shape[ax2] = x_shape[ax2], x_shape[ax1]
        return KerasTensor(x_shape, dtype=x.dtype)


@keras_export(["keras.ops.swapaxes", "keras.ops.numpy.swapaxes"])
def swapaxes(x, axis1, axis2):
    """Interchange two axes of a tensor.

    Args:
        x: Input tensor.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        A tensor with the axes swapped.
    """
    if any_symbolic_tensors((x,)):
        return Swapaxes(axis1, axis2).symbolic_call(x)
    return backend.ops.numpy.swapaxes(x, axis1=axis1, axis2=axis2)


class Take(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x, indices):
        return backend.ops.numpy.take(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        if isinstance(indices, KerasTensor):
            indices_shape = list(indices.shape)
            ragged = indices.ragged
        else:
            indices_shape = list(getattr(np.array(indices), "shape", []))
            ragged = False
        if self.axis is None:
            return KerasTensor(indices_shape, dtype=x.dtype)

        axis = canonicalize_axis(self.axis, len(x_shape))
        output_shape = x_shape[:axis] + indices_shape + x_shape[axis + 1 :]
        return KerasTensor(output_shape, dtype=x.dtype, ragged=ragged)


@keras_export(["keras.ops.take", "keras.ops.numpy.take"])
def take(x, indices, axis=None):
    """Take elements from a tensor along an axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the
            flattened input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return Take(axis=axis).symbolic_call(x, indices)
    return backend.ops.numpy.take(x, indices, axis=axis)


class TakeAlongAxis(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x, indices):
        return backend.ops.numpy.take_along_axis(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        output_shape = operation_utils.compute_take_along_axis_output_shape(
            x.shape, indices.shape, self.axis
        )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.take_along_axis",
        "keras.ops.numpy.take_along_axis",
    ]
)
def take_along_axis(x, indices, axis=None):
    """Select values from `x` at the 1-D `indices` along the given axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the flattened
            input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return TakeAlongAxis(axis=axis).symbolic_call(x, indices)
    return backend.ops.numpy.take_along_axis(x, indices, axis=axis)


class Tan(Operation):
    def call(self, x):
        return backend.ops.numpy.tan(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.tan", "keras.ops.numpy.tan"])
def tan(x):
    """Compute tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tan().symbolic_call(x)
    return backend.ops.numpy.tan(x)


class Tanh(Operation):
    def call(self, x):
        return backend.ops.numpy.tanh(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.tanh", "keras.ops.numpy.tanh"])
def tanh(x):
    """Hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tanh().symbolic_call(x)
    return backend.ops.numpy.tanh(x)


class Tensordot(Operation):
    def __init__(self, axes=2, *, name=None):
        super().__init__(name=name)
        self.axes = axes

    def call(self, x1, x2):
        return backend.ops.numpy.tensordot(x1, x2, axes=self.axes)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if not isinstance(self.axes, int):
            x1_select_shape = [x1_shape[ax] for ax in self.axes[0]]
            x2_select_shape = [x2_shape[ax] for ax in self.axes[1]]
            if not shape_equal(
                x1_select_shape, x2_select_shape, allow_none=True
            ):
                raise ValueError(
                    "Shape mismatch on `x1[axes[0]]` and `x2[axes[1]]`, "
                    f"received {x1_select_shape} and {x2_select_shape}."
                )

            for ax in self.axes[0]:
                x1_shape[ax] = -1
            for ax in self.axes[1]:
                x2_shape[ax] = -1

            x1_shape = list(filter((-1).__ne__, x1_shape))
            x2_shape = list(filter((-1).__ne__, x2_shape))

            output_shape = x1_shape + x2_shape
            return KerasTensor(output_shape, dtype=dtype)

        if self.axes <= 0:
            output_shape = x1_shape + x2_shape
        else:
            output_shape = x1_shape[: -self.axes] + x2_shape[self.axes :]

        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.tensordot", "keras.ops.numpy.tensordot"])
def tensordot(x1, x2, axes=2):
    """Compute the tensor dot product along specified axes.

    Args:
        x1: First tensor.
        x2: Second tensor.
        axes: - If an integer, N, sum over the last N axes of `x1` and the
                first N axes of `x2` in order. The sizes of the corresponding
                axes must match.
              - Or, a list of axes to be summed over, first sequence applying
                to `x1`, second to `x2`. Both sequences must be of the
                same length.

    Returns:
        The tensor dot product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Tensordot(axes=axes).symbolic_call(x1, x2)
    return backend.ops.numpy.tensordot(x1, x2, axes=axes)


class Tile(Operation):
    def __init__(self, repeats, *, name=None):
        super().__init__(name=name)
        self.repeats = repeats

    def call(self, x):
        return backend.ops.numpy.tile(x, self.repeats)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        repeats = self.repeats
        if isinstance(repeats, int):
            repeats = [repeats]
        else:
            repeats = list(repeats)

        if len(x_shape) > len(repeats):
            repeats = [1] * (len(x_shape) - len(repeats)) + repeats
        else:
            x_shape = [1] * (len(repeats) - len(x_shape)) + x_shape

        output_shape = []
        for x_size, repeat in zip(x_shape, repeats):
            if isinstance(x_size, int):
                output_shape.append(x_size * repeat)
            else:
                output_shape.append(None)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.tile", "keras.ops.numpy.tile"])
def tile(x, repeats):
    """Repeat `x` the number of times given by `repeats`.

    If `repeats` has length `d`, the result will have dimension of
    `max(d, x.ndim)`.

    If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending
    new axes.

    If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.

    Args:
        x: Input tensor.
        repeats: The number of repetitions of `x` along each axis.

    Returns:
        The tiled output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Tile(
            repeats,
        ).symbolic_call(x)
    return backend.ops.numpy.tile(x, repeats)


class Trace(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1, *, name=None):
        super().__init__(name=name)
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.ops.numpy.trace(
            x, offset=self.offset, axis1=self.axis1, axis2=self.axis2
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        axis1 = canonicalize_axis(self.axis1, len(x_shape))
        axis2 = canonicalize_axis(self.axis2, len(x_shape))
        if axis1 == axis2:
            raise ValueError(
                "`axis1` and `axis2` must be different. "
                f"Received: axis1={self.axis1}, axis2={self.axis2}"
            )
        x_shape[axis1] = -1
        x_shape[axis2] = -1
        output_shape = [d for d in x_shape if d != -1]
        output_dtype = standardize_dtype(x.dtype)
        if output_dtype in ("bool", "int8", "int16"):
            output_dtype = "int32"
        elif output_dtype in ("uint8", "uint16"):
            output_dtype = "uint32"
        if output_dtype == "uint32" and backend.backend() == "torch":
            # Torch backend doesn't support uint32 dtype.
            output_dtype = "int32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.trace", "keras.ops.numpy.trace"])
def trace(x, offset=0, axis1=0, axis2=1):
    """Return the sum along diagonals of the tensor.

    If `x` is 2-D, the sum along its diagonal with the given offset is
    returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.

    If a has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-arrays whose traces are
    returned.

    The shape of the resulting tensor is the same as that of `x` with `axis1`
    and `axis2` removed.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal. Can be
            both positive and negative. Defaults to `0`.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to `0`.(first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to `1` (second axis).

    Returns:
        If `x` is 2-D, the sum of the diagonal is returned. If `x` has
        larger dimensions, then a tensor of sums along diagonals is
        returned.
    """
    if any_symbolic_tensors((x,)):
        return Trace(offset, axis1, axis2).symbolic_call(x)
    return backend.ops.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


@keras_export(["keras.ops.tri", "keras.ops.numpy.tri"])
def tri(N, M=None, k=0, dtype=None):
    """Return a tensor with ones at and below a diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the tensor.
        M: Number of columns in the tensor.
        k: The sub-diagonal at and below which the array is filled.
            `k = 0` is the main diagonal, while `k < 0` is below it, and
            `k > 0` is above. The default is 0.
        dtype: Data type of the returned tensor. The default is "float32".

    Returns:
        Tensor with its lower triangle filled with ones and zeros elsewhere.
        `T[i, j] == 1` for `j <= i + k`, 0 otherwise.
    """
    return backend.ops.numpy.tri(N, M=M, k=k, dtype=dtype)


class Tril(Operation):
    def __init__(self, k=0, *, name=None):
        super().__init__(name=name)
        self.k = k

    def call(self, x):
        return backend.ops.numpy.tril(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.tril", "keras.ops.numpy.tril"])
def tril(x, k=0):
    """Return lower triangle of a tensor.

    For tensors with `ndim` exceeding 2, `tril` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal above which to zero elements. Defaults to `0`. the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Lower triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tril(k=k).symbolic_call(x)
    return backend.ops.numpy.tril(x, k=k)


class Triu(Operation):
    def __init__(self, k=0, *, name=None):
        super().__init__(name=name)
        self.k = k

    def call(self, x):
        return backend.ops.numpy.triu(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.triu", "keras.ops.numpy.triu"])
def triu(x, k=0):
    """Return upper triangle of a tensor.

    For tensors with `ndim` exceeding 2, `triu` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal below which to zero elements. Defaults to `0`. the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Upper triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Triu(k=k).symbolic_call(x)
    return backend.ops.numpy.triu(x, k=k)


class Trunc(Operation):
    def call(self, x):
        return backend.ops.numpy.trunc(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.trunc", "keras.ops.numpy.trunc"])
def trunc(x):
    """Return the truncated value of the input, element-wise.

    The truncated value of the scalar `x` is the nearest integer `i` which is
    closer to zero than `x` is. In short, the fractional part of the signed
    number `x` is discarded.

    Args:
        x: Input tensor.

    Returns:
        The truncated value of each element in `x`.

    Example:
    >>> x = ops.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> ops.trunc(x)
    array([-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0])
    """
    if any_symbolic_tensors((x,)):
        return Trunc().symbolic_call(x)
    return backend.ops.numpy.trunc(x)


class Vdot(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.vdot(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor([], dtype=dtype)


@keras_export(["keras.ops.vdot", "keras.ops.numpy.vdot"])
def vdot(x1, x2):
    """Return the dot product of two vectors.

    If the first argument is complex, the complex conjugate of the first
    argument is used for the calculation of the dot product.

    Multidimensional tensors are flattened before the dot product is taken.

    Args:
        x1: First input tensor. If complex, its complex conjugate is taken
            before calculation of the dot product.
        x2: Second input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Vdot().symbolic_call(x1, x2)
    return backend.ops.numpy.vdot(x1, x2)


class Inner(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.inner(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_shape = tuple(getattr(x1, "shape", ()))
        x2_shape = tuple(getattr(x2, "shape", ()))
        # A scalar operand acts as scalar multiplication, yielding the other
        # operand's full shape (matches `np.inner`).
        if not x1_shape:
            output_shape = x2_shape
        elif not x2_shape:
            output_shape = x1_shape
        else:
            # `inner` sums over the last axis, so the last dimensions must
            # match; the output is x1.shape[:-1] + x2.shape[:-1].
            d1, d2 = x1_shape[-1], x2_shape[-1]
            if isinstance(d1, int) and isinstance(d2, int) and d1 != d2:
                raise ValueError(
                    "`inner` requires the last dimension of `x1` and `x2` "
                    f"to match. Received: x1.shape={x1_shape}, "
                    f"x2.shape={x2_shape}."
                )
            output_shape = x1_shape[:-1] + x2_shape[:-1]
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.inner", "keras.ops.numpy.inner"])
def inner(x1, x2):
    """Return the inner product of two tensors.

    Ordinary inner product of vectors for 1-D tensors
    (without complex conjugation), in higher dimensions
    a sum product over the last axes.

    Multidimensional arrays are treated as vectors by flattening
    all but their last axes. The resulting dot product is performed
    over their last axes.

    Args:
        x1: First input tensor.
        x2: Second input tensor. The last dimension of `x1` and `x2`
            must match.

    Returns:
        Output tensor. The shape of the output is determined by
        broadcasting the shapes of `x1` and `x2` after removing
        their last axes.
    """
    if any_symbolic_tensors((x1, x2)):
        return Inner().symbolic_call(x1, x2)
    return backend.ops.numpy.inner(x1, x2)


@keras_export(["keras.ops.vectorize", "keras.ops.numpy.vectorize"])
def vectorize(pyfunc, *, excluded=None, signature=None):
    """Turn a function into a vectorized function.

    Example:

    ```python
    def myfunc(a, b):
        return a + b

    vfunc = keras.ops.vectorize(myfunc)
    y = vfunc([1, 2, 3, 4], 2)  # Returns Tensor([3, 4, 5, 6])
    ```

    Args:
        pyfunc: Callable of a single tensor argument.
        excluded: Optional set of integers representing
            positional arguments for which the function
            will not be vectorized.
            These will be passed directly to `pyfunc` unmodified.
        signature: Optional generalized universal function signature,
            e.g., `"(m,n),(n)->(m)"` for vectorized
            matrix-vector multiplication. If provided,
            `pyfunc` will be called with (and expected to return)
            arrays with shapes given by the size of corresponding
            core dimensions. By default, `pyfunc` is assumed
            to take scalars tensors as input and output.

    Returns:
        A new function that applies `pyfunc` to every element
        of its input along axis 0 (the batch axis).
    """
    if not callable(pyfunc):
        raise ValueError(
            "Expected argument `pyfunc` to be a callable. "
            f"Received: pyfunc={pyfunc}"
        )
    return backend.ops.numpy.vectorize(
        pyfunc, excluded=excluded, signature=signature
    )


class Vstack(Operation):
    def call(self, xs):
        return backend.ops.numpy.vstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[0], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[0] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[0]
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[0] = total_size_on_axis
        output_dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.vstack", "keras.ops.numpy.vstack"])
def vstack(xs):
    """Stack tensors in sequence vertically (row wise).

    Args:
        xs: Sequence of tensors.

    Returns:
        Tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Vstack().symbolic_call(xs)
    return backend.ops.numpy.vstack(xs)


class Vsplit(Operation):
    def __init__(self, indices_or_sections, *, name=None):
        super().__init__(name=name)
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections

    def call(self, x):
        return _vsplit(x, self.indices_or_sections)

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                "`vsplit` only works on arrays of at least 2 dimensions. "
                f"Received array with shape {x.shape}."
            )
        return _compute_split_output_spec(x, self.indices_or_sections, 0)


@keras_export(["keras.ops.vsplit", "keras.ops.numpy.vsplit"])
def vsplit(x, indices_or_sections):
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        x: Input tensor.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along axis 0. If a 1-D array of sorted integers,
            the entries indicate indices at which the tensor will be split
            along axis 0.

    Returns:
        A list of sub-arrays.

    Example:

    >>> x = keras.ops.arange(12).reshape((4, 3))
    >>> keras.ops.vsplit(x, 2)
    [array([[0, 1, 2],
           [3, 4, 5]]),
     array([[ 6,  7,  8],
           [ 9, 10, 11]])]
    >>> keras.ops.vsplit(x, [1, 3])
    [array([[0, 1, 2]]),
     array([[3, 4, 5],
           [6, 7, 8]]),
     array([[ 9, 10, 11]])]
    """
    if any_symbolic_tensors((x,)):
        return Vsplit(indices_or_sections).symbolic_call(x)
    return _vsplit(x, indices_or_sections)


def _vsplit(x, indices_or_sections):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "vsplit"
    ):
        return backend.ops.numpy.vsplit(x, indices_or_sections)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.numpy.split(x, indices_or_sections, axis=0)


class Where(Operation):
    def call(self, condition, x1=None, x2=None):
        return backend.ops.numpy.where(condition, x1, x2)

    def compute_output_spec(self, condition, x1, x2):
        condition_shape = getattr(condition, "shape", [])
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(condition_shape, x1_shape)
        output_shape = broadcast_shapes(output_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1) if x1 is not None else "int"),
            getattr(x2, "dtype", type(x2) if x2 is not None else "int"),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.where", "keras.ops.numpy.where"])
def where(condition, x1=None, x2=None):
    """Return elements chosen from `x1` or `x2` depending on `condition`.

    Args:
        condition: Where `True`, yield `x1`, otherwise yield `x2`.
        x1: Values from which to choose when `condition` is `True`.
        x2: Values from which to choose when `condition` is `False`.

    Returns:
        A tensor with elements from `x1` where `condition` is `True`, and
        elements from `x2` where `condition` is `False`.
    """
    if (x1 is None and x2 is not None) or (x1 is not None and x2 is None):
        raise ValueError(
            "`x1` and `x2` either both should be `None`"
            " or both should have non-None value."
        )
    if any_symbolic_tensors((condition, x1, x2)):
        return Where().symbolic_call(condition, x1, x2)
    return backend.ops.numpy.where(condition, x1, x2)


class Subtract(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.subtract(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.subtract", "keras.ops.numpy.subtract"])
def subtract(x1, x2):
    """Subtract arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise difference of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.ops.numpy.subtract(x1, x2)


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.multiply(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_sparse = getattr(x1, "sparse", True)
        x2_sparse = getattr(x2, "sparse", True)
        output_sparse = x1_sparse or x2_sparse
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.multiply", "keras.ops.numpy.multiply"])
def multiply(x1, x2):
    """Multiply arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.ops.numpy.multiply(x1, x2)


class Divide(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.divide", "keras.ops.numpy.divide"])
def divide(x1, x2):
    """Divide arguments element-wise.

    `keras.ops.true_divide` is an alias for this function.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, the quotient `x1/x2`, element-wise.
    """
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.ops.numpy.divide(x1, x2)


class DivideNoNan(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.divide_no_nan(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.divide_no_nan", "keras.ops.numpy.divide_no_nan"])
def divide_no_nan(x1, x2):
    """Safe element-wise division which returns 0 where the denominator is 0.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        The quotient `x1/x2`, element-wise, with zero where x2 is zero.
    """
    if any_symbolic_tensors((x1, x2)):
        return DivideNoNan().symbolic_call(x1, x2)
    return backend.ops.numpy.divide_no_nan(x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.true_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(
    [
        "keras.ops.true_divide",
        "keras.ops.numpy.true_divide",
    ]
)
def true_divide(x1, x2):
    """Alias for `keras.ops.divide`."""
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.ops.numpy.true_divide(x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.power(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.power", "keras.ops.numpy.power"])
def power(x1, x2):
    """First tensor elements raised to powers from second tensor, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        Output tensor, the bases in `x1` raised to the exponents in `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.ops.numpy.power(x1, x2)


class FloatPower(Operation):
    def call(self, x1, x2):
        return _float_power(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.float_power", "keras.ops.numpy.float_power"])
def float_power(x1, x2):
    """First tensor elements raised to powers from second tensor, in floats.

    This is `power` with the operands promoted to a float dtype first, so a
    negative exponent has a well defined result for integer inputs, where
    `power` either raises an error or returns an integer.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        Output tensor, the bases in `x1` raised to the exponents in `x2`.

    Example:
    >>> x1 = keras.ops.convert_to_tensor([2, 3, 4])
    >>> x2 = keras.ops.convert_to_tensor([-1, -2, 2])
    >>> keras.ops.float_power(x1, x2)
    array([ 0.5       ,  0.11111111, 16.        ], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return FloatPower().symbolic_call(x1, x2)
    return _float_power(x1, x2)


def _float_power(x1, x2):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "float_power"
    ):
        return backend.ops.numpy.float_power(x1, x2)
    x1 = backend.ops.convert_to_tensor(x1)
    x2 = backend.ops.convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = backend.ops.cast(x1, dtype)
    x2 = backend.ops.cast(x2, dtype)
    return backend.ops.numpy.power(x1, x2)


class Negative(Operation):
    def call(self, x):
        return backend.ops.numpy.negative(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.negative", "keras.ops.numpy.negative"])
def negative(x):
    """Numerical negative, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, `y = -x`.
    """
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.ops.numpy.negative(x)


class Nextafter(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.nextafter(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        x1_type = standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_type = standardize_dtype(getattr(x2, "dtype", type(x2)))
        dtype = dtypes.result_type(x1_type, x2_type, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.nextafter", "keras.ops.numpy.nextafter"])
def nextafter(x1, x2):
    """
    Return the next representable floating-point value after `x1` towards `x2`.

    This function computes the next floating-point value
    following `x1` in the direction of `x2`, element-wise.

    Args:
        x1: Input tensor whose values will be moved to the next
            representable floating-point value.
        x2: Input tensor indicating the direction toward which
            `x1` is moved.

    Returns:
        Output tensor

    Example:
    >>> x1 = keras.ops.convert_to_tensor([1.0, 1.0])
    >>> x2 = keras.ops.convert_to_tensor([2.0, 0.0])
    >>> keras.ops.nextafter(x1, x2)
    array([1.0000001, 0.99999994], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Nextafter().symbolic_call(x1, x2)
    return backend.ops.numpy.nextafter(x1, x2)


class Square(Operation):
    def call(self, x):
        return backend.ops.numpy.square(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        dtype = standardize_dtype(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.square", "keras.ops.numpy.square"])
def square(x):
    """Return the element-wise square of the input.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the square of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.ops.numpy.square(x)


class Sqrt(Operation):
    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        return backend.ops.numpy.sqrt(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sqrt", "keras.ops.numpy.sqrt"])
def sqrt(x):
    """Return the non-negative square root of a tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the non-negative square root of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sqrt().symbolic_call(x)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.numpy.sqrt(x)


class Squeeze(Operation):
    def __init__(self, axis=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.numpy.squeeze(x, axis=self.axis)

    def compute_output_spec(self, x):
        input_shape = list(x.shape)
        sparse = getattr(x, "sparse", False)
        axis = to_tuple_or_list(self.axis)
        if axis is None:
            output_shape = [d for d in input_shape if d != 1]
            return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)
        else:
            for a in axis:
                if input_shape[a] != 1:
                    raise ValueError(
                        f"Cannot squeeze axis {a}, because the dimension "
                        "is not 1."
                    )
            axis = [canonicalize_axis(a, len(input_shape)) for a in axis]
            for a in sorted(axis, reverse=True):
                del input_shape[a]
            return KerasTensor(input_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.squeeze", "keras.ops.numpy.squeeze"])
def squeeze(x, axis=None):
    """Remove axes of length one from `x`.

    Args:
        x: Input tensor.
        axis: Select a subset of the entries of length one in the shape.

    Returns:
        The input tensor with all or a subset of the dimensions of
        length 1 removed.
    """
    if any_symbolic_tensors((x,)):
        return Squeeze(axis=axis).symbolic_call(x)
    return backend.ops.numpy.squeeze(x, axis=axis)


class Transpose(Operation):
    def __init__(self, axes=None, *, name=None):
        super().__init__(name=name)
        self.axes = axes

    def call(self, x):
        return backend.ops.numpy.transpose(x, axes=self.axes)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_transpose_output_shape(
            x.shape, self.axes
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.transpose", "keras.ops.numpy.transpose"])
def transpose(x, axes=None):
    """Returns a tensor with `axes` transposed.

    Args:
        x: Input tensor.
        axes: Sequence of integers. Permutation of the dimensions of `x`.
            By default, the order of the axes are reversed.

    Returns:
        `x` with its axes permuted.
    """
    if any_symbolic_tensors((x,)):
        return Transpose(axes=axes).symbolic_call(x)
    return backend.ops.numpy.transpose(x, axes=axes)


class Trapezoid(Operation):
    def __init__(self, x=None, dx=1.0, axis=-1, *, name=None):
        super().__init__(name=name)
        self.x = x
        self.dx = dx
        self.axis = axis

    def call(self, y):
        return backend.ops.numpy.trapezoid(
            y, x=self.x, dx=self.dx, axis=self.axis
        )

    def compute_output_spec(self, y):
        out_shape = list(y.shape)
        if self.axis is not None and len(out_shape) > 0:
            out_shape.pop(self.axis % len(out_shape))
        dtype = backend.result_type(getattr(y, "dtype", type(y)), float)
        return KerasTensor(tuple(out_shape), dtype=dtype)


@keras_export(["keras.ops.trapezoid", "keras.ops.numpy.trapezoid"])
def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y: Input tensor.
        x: Optional tensor specifying sample points corresponding to `y`.
           If `None`, spacing is assumed to be `dx`.
        dx: Spacing between sample points when `x` is `None`.
        axis: Axis along which to integrate. Default is the last axis.

    Returns:
        The approximate integral of `y` along the given axis.

    Example:
    >>> y = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> keras.ops.trapezoid(y, axis=1)
    array([ 4., 10.], dtype=float32)
    """
    if any_symbolic_tensors((y,)):
        return Trapezoid(x=x, dx=dx, axis=axis).symbolic_call(y)
    return backend.ops.numpy.trapezoid(y, x=x, dx=dx, axis=axis)


class Mean(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.mean(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        result_dtype = standardize_dtype(x.dtype)
        if "int" in result_dtype or result_dtype == "bool":
            result_dtype = config.floatx()
        sparse = getattr(x, "sparse", False)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=result_dtype,
            sparse=sparse,
        )


@keras_export(["keras.ops.mean", "keras.ops.numpy.mean"])
def mean(x, axis=None, keepdims=False):
    """Compute the arithmetic mean along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the means are computed. The default
            is to compute the mean of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the mean values.
    """
    if any_symbolic_tensors((x,)):
        return Mean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.mean(x, axis=axis, keepdims=keepdims)


class Vander(Operation):
    def __init__(self, N=None, increasing=False, *, name=None):
        super().__init__(name=name)
        self.N = N
        self.increasing = increasing

    def call(self, x):
        return backend.ops.numpy.vander(x, self.N, self.increasing)

    def compute_output_spec(self, x):
        if self.N is None:
            N = x.shape[0]
        else:
            N = self.N

        out_shape = x.shape + (N,)
        return KerasTensor(tuple(out_shape), dtype=x.dtype)


@keras_export(["keras.ops.vander", "keras.ops.numpy.vander"])
def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix.

    Args:
        x: 1D input tensor.
        N: Number of columns. If `None`, `N` = `len(x)`.
        increasing: Order of powers. If True, powers increase left to right.

    Returns:
        Output tensor, Vandermonde matrix of shape `(len(x), N)`.

    Example:
    >>> import numpy as np
    >>> import keras
    >>> x = np.array([1, 2, 3, 5])
    >>> keras.ops.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])
    """

    if len(x.shape) != 1:
        raise ValueError(
            "Input tensor must be 1-dimensional. "
            f"Received: input.shape={x.shape}"
        )

    if N is not None:
        if not isinstance(N, int):
            raise TypeError(
                f"Argument `N` must be of type `int`. "
                f"Received: N={N} of type {type(N)}"
            )

        if N < 0:
            raise ValueError(
                f"Argument 'N' must be nonnegative. Received: N={N}"
            )

    if not isinstance(increasing, bool):
        raise TypeError(
            f"Argument `increasing` must be of type `bool`. "
            f"Received: increasing={increasing} of type {type(increasing)}"
        )

    if any_symbolic_tensors((x,)):
        return Vander(N=N, increasing=increasing).symbolic_call(x)
    return backend.ops.numpy.vander(x, N=N, increasing=increasing)


class Var(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.var(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = backend.result_type(getattr(x, "dtype", type(x)), float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.var", "keras.ops.numpy.var"])
def var(x, axis=None, keepdims=False):
    """Compute the variance along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed. The default
            is to compute the variance of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the variance.
    """
    if any_symbolic_tensors((x,)):
        return Var(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.var(x, axis=axis, keepdims=keepdims)


class Sum(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.ops.numpy.sum(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(getattr(x, "dtype", backend.floatx()))
        # follow jax's rule
        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"
        sparse = getattr(x, "sparse", False)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
            sparse=sparse,
        )


@keras_export(["keras.ops.sum", "keras.ops.numpy.sum"])
def sum(x, axis=None, keepdims=False):
    """Sum of a tensor over the given axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the sum is computed. The default is to
            compute the sum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the sum.
    """
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.ops.numpy.sum(x, axis=axis, keepdims=keepdims)


@keras_export(["keras.ops.zeros", "keras.ops.numpy.zeros"])
def zeros(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with zeros.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of zeros with the given shape and dtype.
    """
    return backend.ops.numpy.zeros(shape, dtype=dtype)


@keras_export(["keras.ops.ones", "keras.ops.numpy.ones"])
def ones(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with ones.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of ones with the given shape and dtype.
    """
    return backend.ops.numpy.ones(shape, dtype=dtype)


@keras_export(["keras.ops.eye", "keras.ops.numpy.eye"])
def eye(N, M=None, k=0, dtype=None):
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If `None`, defaults to `N`.
        k: Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal,
            and a negative value to a lower diagonal.
        dtype: Data type of the returned tensor.

    Returns:
        Tensor with ones on the k-th diagonal and zeros elsewhere.
    """

    def is_floating_type(v):
        return (
            isinstance(v, float)
            or getattr(v, "dtype", None) in dtypes.FLOAT_TYPES
        )

    if is_floating_type(N):
        raise TypeError("Argument `N` must be an integer or an integer tensor.")
    if is_floating_type(M):
        raise TypeError(
            "Argument `M` must be an integer, an integer tensor, or `None`."
        )
    return backend.ops.numpy.eye(N, M=M, k=k, dtype=dtype)


class FloorDivide(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.floor_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.floor_divide", "keras.ops.numpy.floor_divide"])
def floor_divide(x1, x2):
    """Returns the largest integer smaller or equal to the division of inputs.

    Args:
        x1: Numerator.
        x2: Denominator.

    Returns:
        Output tensor, `y = floor(x1/x2)`
    """
    if any_symbolic_tensors((x1, x2)):
        return FloorDivide().symbolic_call(x1, x2)
    return backend.ops.numpy.floor_divide(x1, x2)


class LogicalXor(Operation):
    def call(self, x1, x2):
        return backend.ops.numpy.logical_xor(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.logical_xor", "keras.ops.numpy.logical_xor"])
def logical_xor(x1, x2):
    """Compute the truth value of `x1 XOR x2`, element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalXor().symbolic_call(x1, x2)
    return backend.ops.numpy.logical_xor(x1, x2)


class Corrcoef(Operation):
    def call(self, x):
        return backend.ops.numpy.corrcoef(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = "float64"
        else:
            dtype = dtypes.result_type(dtype, float)
        if len(x.shape) > 2:
            raise ValueError(
                "Input tensor must have at most 2 dimensions. "
                f"Received: x.shape={x.shape}"
            )
        # The correlation matrix of the `N` variables of a 2D input of shape
        # `(N, D)` has shape `(N, N)`. A 1D input yields a scalar.
        output_shape = (x.shape[0], x.shape[0]) if len(x.shape) == 2 else ()
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.corrcoef", "keras.ops.numpy.corrcoef"])
def corrcoef(x):
    """Compute the Pearson correlation coefficient matrix.

    Args:
        x: A 2D tensor of shape `(N, D)`, where N is the number of variables
           and D is the number of observations.

    Returns:
        A tensor of shape `(N, N)` representing the correlation matrix.
    """
    if any_symbolic_tensors((x,)):
        return Corrcoef().symbolic_call(x)
    return backend.ops.numpy.corrcoef(x)


class Correlate(Operation):
    def __init__(self, mode="valid", *, name=None):
        super().__init__(name=name)
        self.mode = mode

    def call(self, x1, x2):
        return backend.ops.numpy.correlate(x1, x2, mode=self.mode)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        if len(x1_shape) != 1:
            raise ValueError(
                "`x1` must be a 1-dimensional tensor, but received"
                + f"shape {x1_shape}"
            )
        if len(x2_shape) != 1:
            raise ValueError(
                "`x2` must be a 1-dimensional tensor, but received"
                + f"shape {x2_shape}"
            )
        x1_len, x2_len = x1_shape[0], x2_shape[0]
        output_shape = (
            np.maximum(x1_len, x2_len) - np.minimum(x1_len, x2_len) + 1,
        )
        if self.mode == "same":
            output_shape = (np.maximum(x1_len, x2_len),)
        elif self.mode == "full":
            output_shape = (x1_len + x2_len - 1,)
        if self.mode not in ("valid", "same", "full"):
            raise ValueError(
                "`mode` must be either `valid`, `same`, or `full`, but"
                f"received: {self.mode}"
            )
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if output_dtype == "int64":
            output_dtype = "float64"
        elif output_dtype not in ["bfloat16", "float16", "float64"]:
            output_dtype = "float32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.correlate", "keras.ops.numpy.correlate"])
def correlate(x1, x2, mode="valid"):
    """Compute the cross-correlation of two 1-dimensional tensors.

    Args:
        x1: First 1-dimensional input tensor of length M.
        x2: Second 1-dimensional input tensor of length N.
        mode: Either `valid`, `same` or `full`.
            By default the mode is set to `valid`, which returns
            an output of length max(M, N) - min(M, N) + 1.
            `same` returns an output of length max(M, N).
            `full` mode returns the convolution at each point of
            overlap, with an output length of N+M-1

    Returns:
        Output tensor, cross-correlation of `x1` and `x2`.

    Notes:
        Complex-valued inputs are currently not fully supported on the
        TensorFlow and PyTorch backends. When complex tensors are passed,
        they are cast to floating-point types and the imaginary component
        is discarded.

        This behavior is documented for clarity and may change in the
        future. See discussion in issue #21617.
    """
    if any_symbolic_tensors((x1, x2)):
        return Correlate(mode=mode).symbolic_call(x1, x2)
    return backend.ops.numpy.correlate(x1, x2, mode=mode)


class Select(Operation):
    def call(self, condlist, choicelist, default=0):
        return backend.ops.numpy.select(condlist, choicelist, default)

    def compute_output_spec(self, condlist, choicelist, default=0):
        # `select` broadcasts every array in `condlist` and `choicelist` (and
        # `default`) to a common shape and promotes the dtype across all
        # choices, like `where` above -- rather than assuming the first
        # choice's shape/dtype, which is wrong when a condition, a later
        # choice, or `default` is broader than the first choice.
        output_shape = ()
        for x in list(condlist) + list(choicelist) + [default]:
            output_shape = broadcast_shapes(
                output_shape, getattr(x, "shape", [])
            )
        dtypes_to_resolve = [
            getattr(choice, "dtype", type(choice)) for choice in choicelist
        ]
        dtypes_to_resolve.append(getattr(default, "dtype", type(default)))
        output_dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.select", "keras.ops.numpy.select"])
def select(condlist, choicelist, default=0):
    """Return elements from `choicelist`, based on conditions in `condlist`.

    Args:
        condlist: List of boolean tensors.
            The list of conditions which determine from which array
            in choicelist the output elements are taken.
            When multiple conditions are satisfied,
            the first one encountered in condlist is used.
        choicelist: List of tensors.
            The list of tensors from which the output elements are taken.
            This list has to be of the same length as `condlist`.
        default: Optional scalar value.
            The element inserted in the output
            when all conditions evaluate to `False`.

    Returns:
        Tensor where the output at position `m` is the `m`-th element
        of the tensor in `choicelist` where the `m`-th element of the
        corresponding tensor in `condlist` is `True`.

    Example:

    ```python
    from keras import ops

    x = ops.arange(6)
    condlist = [x<3, x>3]
    choicelist = [x, x**2]
    ops.select(condlist, choicelist, 42)
    # Returns: tensor([0,  1,  2, 42, 16, 25])
    ```
    """
    if not isinstance(condlist, (list, tuple)) or not isinstance(
        choicelist, (list, tuple)
    ):
        raise ValueError(
            "condlist and choicelist must be lists. Received: "
            f"type(condlist) = {type(condlist)}, "
            f"type(choicelist) = {type(choicelist)}"
        )
    condlist = list(condlist)
    choicelist = list(choicelist)
    if not condlist or not choicelist:
        raise ValueError(
            "condlist and choicelist must not be empty. Received: "
            f"condlist = {condlist}, "
            f"choicelist = {choicelist}"
        )
    if any_symbolic_tensors(condlist + choicelist + [default]):
        return Select().symbolic_call(condlist, choicelist, default)
    return backend.ops.numpy.select(condlist, choicelist, default)


class Slogdet(Operation):
    def call(self, x):
        return backend.ops.numpy.slogdet(x)

    def compute_output_spec(self, x):
        sign = KerasTensor(x.shape[:-2], dtype=x.dtype)
        logabsdet = KerasTensor(x.shape[:-2], dtype=x.dtype)
        return (sign, logabsdet)


@keras_export(["keras.ops.slogdet", "keras.ops.numpy.slogdet"])
def slogdet(x):
    """Compute the sign and natural logarithm of the determinant of a matrix.

    Args:
        x: Input matrix. It must 2D and square.

    Returns:
        A tuple `(sign, logabsdet)`. `sign` is a number representing
        the sign of the determinant. For a real matrix, this is 1, 0, or -1.
        For a complex matrix, this is a complex number with absolute value 1
        (i.e., it is on the unit circle), or else 0.
        `logabsdet` is the natural log of the absolute value of the determinant.
    """
    if any_symbolic_tensors((x,)):
        return Slogdet().symbolic_call(x)
    return backend.ops.numpy.slogdet(x)


class Argpartition(Operation):
    def __init__(self, kth, axis=-1, *, name=None):
        super().__init__(name=name)
        if not isinstance(kth, int):
            raise ValueError(f"kth must be an integer. Received:kth = {kth}")
        self.kth = kth
        self.axis = axis

    def call(self, x):
        return _argpartition(x, kth=self.kth, axis=self.axis)

    def compute_output_spec(self, x):
        shape = x.shape

        if self.axis is None:
            size = 1
            for dim in shape:
                if dim is None:
                    size = None
                    break
                size *= dim
            shape = (size,)

        return KerasTensor(shape, dtype="int32")


@keras_export(["keras.ops.argpartition", "keras.ops.numpy.argpartition"])
def argpartition(x, kth, axis=-1):
    """Performs an indirect partition along the given axis.

    It returns an array
    of indices of the same shape as `x` that index data along the given axis
    in partitioned order.

    Args:
        x: Array to sort.
        kth: Element index to partition by.
            The k-th element will be in its final sorted position and all
            smaller elements will be moved before it and all larger elements
            behind it. The order of all elements in the partitions is undefined.
            If provided with a sequence of k-th it will partition all of them
            into their sorted position at once.
        axis: Axis along which to sort. The default is -1 (the last axis).
            If `None`, the flattened array is used.

    Returns:
        Array of indices that partition `x` along the specified `axis`.
    """
    if any_symbolic_tensors((x,)):
        return Argpartition(kth, axis).symbolic_call(x)
    return _argpartition(x, kth, axis)


def _argpartition(x, kth, axis=-1):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "argpartition"
    ):
        return backend.ops.numpy.argpartition(x, kth, axis)
    x = backend.ops.convert_to_tensor(x)
    if axis is None:
        x = backend.ops.numpy.reshape(x, (-1,))
        axis = -1
    size = x.shape[axis]
    if size is not None and not -size <= kth < size:
        raise ValueError(
            f"kth must be in the range [-{size}, {size}) for an axis of size "
            f"{size}. Received: kth={kth}"
        )
    # A full sort is a valid partition for every kth.
    indices = backend.ops.numpy.argsort(x, axis=axis)
    return backend.ops.cast(indices, "int32")


class Histogram(Operation):
    def __init__(self, bins=10, range=None, *, name=None):
        super().__init__(name=name)

        if not isinstance(bins, int):
            raise TypeError("bins must be of type `int`")
        if bins < 0:
            raise ValueError("`bins` should be a non-negative integer")

        if range:
            if len(range) < 2 or not isinstance(range, tuple):
                raise ValueError("range must be a tuple of two elements")

            if range[1] < range[0]:
                raise ValueError(
                    "The second element of range must be greater than the first"
                )

        self.bins = bins
        self.range = range

    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        if len(x.shape) > 1:
            raise ValueError("Input tensor must be 1-dimensional")
        return backend.ops.numpy.histogram(x, bins=self.bins, range=self.range)

    def compute_output_spec(self, x):
        return (
            KerasTensor(shape=(self.bins,), dtype=x.dtype),
            KerasTensor(shape=(self.bins + 1,), dtype=x.dtype),
        )


@keras_export(["keras.ops.histogram", "keras.ops.numpy.histogram"])
def histogram(x, bins=10, range=None):
    """Computes a histogram of the data tensor `x`.

    Args:
        x: Input tensor.
        bins: An integer representing the number of histogram bins.
            Defaults to 10.
        range: A tuple representing the lower and upper range of the bins.
            If not specified, it will use the min and max of `x`.

    Returns:
        A tuple containing:
        - A tensor representing the counts of elements in each bin.
        - A tensor representing the bin edges.

    Example:
    >>> input_tensor = np.random.rand(8)
    >>> keras.ops.histogram(input_tensor)
    (array([1, 1, 1, 0, 0, 1, 2, 1, 0, 1], dtype=int32),
    array([0.0189519 , 0.10294958, 0.18694726, 0.27094494, 0.35494262,
        0.43894029, 0.52293797, 0.60693565, 0.69093333, 0.77493101,
        0.85892869]))
    """
    if not isinstance(bins, int):
        raise TypeError(
            f"Argument `bins` must be of type `int`. Received: bins={bins}"
        )
    if bins < 0:
        raise ValueError(
            "Argument `bins` should be a non-negative integer. "
            f"Received: bins={bins}"
        )

    if range:
        if len(range) < 2 or not isinstance(range, tuple):
            raise ValueError(
                "Argument `range` must be a tuple of two elements. "
                f"Received: range={range}"
            )

        if range[1] < range[0]:
            raise ValueError(
                "The second element of `range` must be greater than the first. "
                f"Received: range={range}"
            )

    if any_symbolic_tensors((x,)):
        return Histogram(bins=bins, range=range).symbolic_call(x)

    x = backend.ops.convert_to_tensor(x)
    if len(x.shape) > 1:
        raise ValueError(
            "Input tensor must be 1-dimensional. "
            f"Received: input.shape={x.shape}"
        )
    return backend.ops.numpy.histogram(x, bins=bins, range=range)


class ArraySplit(Operation):
    def __init__(self, indices_or_sections, axis=0, *, name=None):
        super().__init__(name=name)

        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def call(self, x):
        return _array_split(
            x,
            indices_or_sections=self.indices_or_sections,
            axis=self.axis,
        )

    def compute_output_spec(self, x):
        num_splits = self.indices_or_sections

        axis = self.axis
        if axis < 0:
            axis += len(x.shape)

        total_size = x.shape[axis]

        if total_size is None:
            output_specs = []
            base_shape = list(x.shape)
            base_shape[axis] = None
            for _ in range(num_splits):
                output_specs.append(
                    KerasTensor(shape=tuple(base_shape), dtype=x.dtype)
                )
            return tuple(output_specs)

        split_size = total_size // num_splits
        remainder = total_size % num_splits

        output_specs = []
        base_shape = list(x.shape)
        for i in range(num_splits):
            size = split_size + (1 if i < remainder else 0)
            shape = base_shape.copy()
            shape[axis] = size
            output_specs.append(KerasTensor(shape=tuple(shape), dtype=x.dtype))

        return list(output_specs)


@keras_export(["keras.ops.array_split", "keras.ops.numpy.array_split"])
def array_split(x, indices_or_sections, axis=0):
    """Splits an array into multiple sub-arrays (unevenly).

    This is similar to `keras.ops.split`, but it allows for
    unequal splits. `indices_or_sections` must be an integer
    that indicates the total number of sub-arrays to create.
    If the tensor cannot be divided evenly, the first `remainder`
    splits will have size `quotient + 1`, and the rest will
    have size `quotient`.

    Args:
        x: Input tensor.
        indices_or_sections: An integer indicating the number of
            sub-arrays to create.
        axis: The axis along which to split. Defaults to 0.

    Returns:
        A list of sub-tensors.

    Example:
    >>> x = keras.ops.arange(10)
    >>> keras.ops.array_split(x, 3)
    (array([0, 1, 2, 3], dtype=int32),
     array([4, 5, 6], dtype=int32),
     array([7, 8, 9], dtype=int32))
    """
    if not isinstance(indices_or_sections, int):
        raise TypeError(
            "Argument `indices_or_sections` must be of type `int`. "
            f"Received: indices_or_sections={indices_or_sections}"
        )

    if indices_or_sections <= 0:
        raise ValueError(
            "Argument `indices_or_sections` must be a positive integer. "
            f"Received: indices_or_sections={indices_or_sections}"
        )

    if not isinstance(axis, int):
        raise TypeError(
            f"Argument `axis` must be of type `int`. Received: {axis}"
        )

    if any_symbolic_tensors((x,)):
        return ArraySplit(
            indices_or_sections=indices_or_sections, axis=axis
        ).symbolic_call(x)

    return _array_split(x, indices_or_sections=indices_or_sections, axis=axis)


def _array_split(x, indices_or_sections, axis=0):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "array_split"
    ):
        return backend.ops.numpy.array_split(
            x, indices_or_sections=indices_or_sections, axis=axis
        )
    x = backend.ops.convert_to_tensor(x)
    total_size = backend.ops.shape(x)[axis]
    num_splits = indices_or_sections
    if num_splits == 1:
        return [x]
    indices = []
    offset = 0
    for i in range(num_splits - 1):
        offset += (total_size + num_splits - 1 - i) // num_splits
        indices.append(offset)
    return backend.ops.numpy.split(x, indices, axis=axis)


class Unique(Operation):
    def __init__(
        self,
        sorted=True,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        axis=None,
        size=None,
        fill_value=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.sorted = sorted
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis
        self.size = size
        self.fill_value = fill_value

    def call(self, x):
        return backend.ops.numpy.unique(
            x,
            sorted=self.sorted,
            return_index=self.return_index,
            return_inverse=self.return_inverse,
            return_counts=self.return_counts,
            axis=self.axis,
            size=self.size,
            fill_value=self.fill_value,
        )

    def compute_output_spec(self, x):
        input_shape = list(x.shape)
        # Determine the dimension of the unique values
        # If size is provided, the shape becomes static for JIT compatibility
        u_dim = self.size if self.size is not None else None

        if self.axis is None:
            values_shape = (u_dim,)
        else:
            axis = canonicalize_axis(self.axis, len(input_shape))
            values_shape = list(input_shape)
            values_shape[axis] = u_dim

        outputs = [KerasTensor(tuple(values_shape), dtype=x.dtype)]

        # Indices of first occurrences shape is 1D
        # with length matching the unique values
        if self.return_index:
            outputs.append(KerasTensor((u_dim,), dtype="int32"))

        # Inverse indices shape matches the original input size along axis
        if self.return_inverse:
            if self.axis is None:
                inv_shape = x.shape
            else:
                axis = canonicalize_axis(self.axis, len(input_shape))
                inv_shape = (input_shape[axis],)
            outputs.append(KerasTensor(tuple(inv_shape), dtype="int32"))

        # Counts shape is 1D with length matching the unique values
        if self.return_counts:
            outputs.append(KerasTensor((u_dim,), dtype="int32"))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


@keras_export(["keras.ops.unique", "keras.ops.numpy.unique"])
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
    """Finds the unique elements of a tensor.

    This function returns the unique elements of an array, optionally sorted.
    When `size` is provided, the output shape is fixed, making it compatible
    with JIT compilation (e.g., JAX, TensorFlow Graph Mode).

    Args:
        x: Input tensor.
        sorted: Whether to sort the unique elements in ascending order.
            Defaults to `True`.
        return_index: If `True`, also return the indices of the first
            occurrence of each unique element in `x`. Defaults to `False`.
        return_inverse: If `True`, also return the indices of the unique tensor
            (for the specified axis, if provided) that can be used to
            reconstruct `x`. Defaults to `False`.
        return_counts: If `True`, also return the number of times each
            unique item appears in `x`. Defaults to `False`.
        axis: The axis to operate on. If `None`, `x` will be flattened.
            If an integer, the subarrays indexed by the given axis will be
            treated as the elements. Defaults to `None`.
        size: Optional integer. If provided, the output will always have this
            fixed size. If the number of unique elements is less than `size`,
            the output will be padded with `fill_value`. If greater, the
            output will be truncated.
        fill_value: The value used to pad the output when `size` is specified
            and there are fewer unique elements than `size`. Defaults to `0`.

    Returns:
        A tensor or a tuple of tensors.
        - `unique_values`: The unique values. Shape is `(size,)` if `size` is
          provided and `axis=None`.
        - `unique_index` (optional): The indices of the first occurrence of each
          unique element in `x`. Only provided if `return_index` is `True`.
        - `unique_inverse` (optional): The indices to reconstruct the original
          array. Only provided if `return_inverse` is `True`.
        - `unique_counts` (optional): The number of occurrences of each unique
          value. Only provided if `return_counts` is `True`.

    Examples:

    >>> x = keras.ops.convert_to_tensor([3, 1, 2, 1, 3, 2])
    >>> keras.ops.unique(x)
    array([1, 2, 3], dtype=int32)

    Fixed size output with padding (JIT-compatible):

    >>> x = keras.ops.convert_to_tensor([3, 1, 2, 1])
    >>> values, counts = keras.ops.unique(x, size=5, fill_value=0,
    ...                                   return_counts=True)
    >>> values
    array([1, 2, 3, 0, 0], dtype=int32)
    >>> counts
    array([2, 1, 1, 0, 0], dtype=int32)

    Fixed size output with truncation:

    >>> x = keras.ops.convert_to_tensor([3, 1, 2, 1])
    >>> keras.ops.unique(x, size=2)
    array([1, 2], dtype=int32)

    Return unique rows of a 2D array:

    >>> x = keras.ops.convert_to_tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> keras.ops.unique(x, axis=0)
    array([[0, 1, 0],
           [1, 0, 0]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Unique(
            sorted=sorted,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
            axis=axis,
            size=size,
            fill_value=fill_value,
        ).symbolic_call(x)
    return backend.ops.numpy.unique(
        backend.ops.convert_to_tensor(x),
        sorted=sorted,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        size=size,
        fill_value=fill_value,
    )


class Dsplit(Operation):
    def __init__(self, indices_or_sections, *, name=None):
        super().__init__(name=name)
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections

    def call(self, x):
        return _dsplit(x, self.indices_or_sections)

    def compute_output_spec(self, x):
        if len(x.shape) < 3:
            raise ValueError(
                "`dsplit` only works on arrays of at least 3 dimensions. "
                f"Received array with shape {x.shape}."
            )
        return _compute_split_output_spec(x, self.indices_or_sections, 2)


@keras_export(["keras.ops.dsplit", "keras.ops.numpy.dsplit"])
def dsplit(x, indices_or_sections):
    """Split an array into multiple sub-arrays depth-wise.

    Args:
        x: Input tensor.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along axis 2. If a 1-D array of sorted integers,
            the entries indicate indices at which the tensor will be split
            along axis 2.
    Returns:
        A list of sub-arrays.

    Example:

    >>> x = keras.ops.arange(16.0).reshape((2, 2, 4))
    >>> keras.ops.dsplit(x, 2)
    [array([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]),
     array([[[ 2.,  3.],
            [ 6.,  7.]],
           [[10., 11.],
            [14., 15.]]])]
    """
    if any_symbolic_tensors((x,)):
        return Dsplit(indices_or_sections).symbolic_call(x)
    return _dsplit(x, indices_or_sections)


def _dsplit(x, indices_or_sections):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "dsplit"
    ):
        return backend.ops.numpy.dsplit(x, indices_or_sections)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.numpy.split(x, indices_or_sections, axis=2)


class ColumnStack(Operation):
    def call(self, xs):
        return backend.ops.numpy.column_stack(xs)

    def compute_output_spec(self, xs):
        if len(xs) == 0:
            raise ValueError(
                "`column_stack` requires at least one tensor, but received "
                "an empty sequence."
            )

        first_shape = xs[0].shape
        if len(first_shape) < 1:
            raise ValueError(
                "`column_stack` requires tensors of at least 1 dimension. "
                f"Received tensor with shape {first_shape}."
            )

        # 1-D tensors are treated as columns, i.e. reshaped to (N, 1).
        first_effective_shape = (
            (first_shape[0], 1) if len(first_shape) == 1 else first_shape
        )

        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            x_shape = x.shape
            if len(x_shape) < 1:
                raise ValueError(
                    "`column_stack` requires tensors of at least 1 "
                    f"dimension. Received tensor with shape {x_shape}."
                )
            x_effective_shape = (
                (x_shape[0], 1) if len(x_shape) == 1 else x_shape
            )
            if not shape_equal(
                x_effective_shape,
                first_effective_shape,
                axis=1,
                allow_none=True,
            ):
                raise ValueError(
                    "Every value in xs must have the same shape except on "
                    "the column axis (axis 1) after 1-D tensors are reshaped "
                    "to 2-D columns. But found element of shape "
                    f"{x_shape}, which is different from the first "
                    f"element's shape {first_shape}."
                )
            if total_size_on_axis is None or x_effective_shape[1] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x_effective_shape[1]
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))

        output_shape = list(first_effective_shape)
        output_shape[1] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.column_stack", "keras.ops.numpy.column_stack"])
def column_stack(xs):
    """Stack 1-D tensors as columns into a 2-D tensor.

    Take a sequence of 1-D tensors and stack them as columns to make a
    single 2-D tensor. 2-D tensors are stacked as-is, like with
    `hstack`. Tensors with more than 2 dimensions are concatenated
    along the second axis.

    Args:
        xs: Sequence of tensors.

    Returns:
        The tensor formed by stacking the given tensors.

    Example:

    >>> a = keras.ops.array([1, 2, 3])
    >>> b = keras.ops.array([4, 5, 6])
    >>> keras.ops.column_stack([a, b])
    array([[1, 4],
           [2, 5],
           [3, 6]])
    """
    if any_symbolic_tensors((xs,)):
        return ColumnStack().symbolic_call(xs)
    return backend.ops.numpy.column_stack(xs)


class Cov(Operation):
    def call(self, x):
        return _cov(x)

    def compute_output_spec(self, x):
        dtype = standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = "float64"
        else:
            dtype = dtypes.result_type(dtype, float)
        if len(x.shape) > 2:
            raise ValueError(
                "Input tensor must have at most 2 dimensions. "
                f"Received: x.shape={x.shape}"
            )
        # The covariance matrix of a 2D input of shape `(N, D)` has shape
        # `(N, N)`. A 1D input, or a single variable, yields a scalar.
        if len(x.shape) == 2 and x.shape[0] != 1:
            output_shape = (x.shape[0], x.shape[0])
        else:
            output_shape = ()
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.cov", "keras.ops.numpy.cov"])
def cov(x):
    """Estimate the covariance matrix of the variables in `x`.

    The covariance is normalized by `D - 1`, where `D` is the number of
    observations.

    Args:
        x: A 2D tensor of shape `(N, D)`, where N is the number of variables
           and D is the number of observations.

    Returns:
        A tensor of shape `(N, N)` representing the covariance matrix.
    """
    if any_symbolic_tensors((x,)):
        return Cov().symbolic_call(x)
    return _cov(x)


def _cov(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "cov"
    ):
        return backend.ops.numpy.cov(x)
    x = backend.ops.convert_to_tensor(x)
    if len(x.shape) > 2:
        raise ValueError(
            "Input tensor must have at most 2 dimensions. "
            f"Received: x.shape={x.shape}"
        )
    dtype = standardize_dtype(x.dtype)
    if dtype == "int64":
        dtype = "float64"
    else:
        dtype = dtypes.result_type(dtype, float)
    x = backend.ops.cast(x, dtype)
    # A 0D input has no observations to vary over, as in `np.cov`.
    if len(x.shape) == 0:
        return backend.ops.numpy.full((), float("nan"), dtype=dtype)
    # `np.cov` squeezes the result when there is only one variable.
    is_scalar = len(x.shape) < 2 or x.shape[0] == 1
    if len(x.shape) == 1:
        x = backend.ops.numpy.reshape(x, (1, -1))
    mean = backend.ops.numpy.mean(x, axis=-1, keepdims=True)
    x_centered = backend.ops.numpy.subtract(x, mean)
    num_samples = backend.ops.cast(backend.ops.shape(x)[-1], dtype)
    result = backend.ops.numpy.divide(
        backend.ops.numpy.matmul(
            x_centered,
            backend.ops.numpy.transpose(
                backend.ops.numpy.conjugate(x_centered)
            ),
        ),
        backend.ops.numpy.subtract(num_samples, 1),
    )
    return backend.ops.numpy.reshape(result, ()) if is_scalar else result
