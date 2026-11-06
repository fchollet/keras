"""Commonly-used neural network operations not included in NumPy."""

import warnings

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import config
from keras.src.backend import standardize_data_format
from keras.src.backend.common import dtypes
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import check_conv_input_channels
from keras.src.backend.common.backend_utils import (
    check_conv_transpose_input_channels,
)
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
from keras.src.utils.argument_validation import standardize_tuple


class Relu(Operation):
    def call(self, x):
        return backend.ops.nn.relu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.relu", "keras.ops.nn.relu"])
def relu(x):
    """Rectified linear unit activation function.

    It is defined as `f(x) = max(0, x)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x1 = keras.ops.convert_to_tensor([-1.0, 0.0, 1.0, 0.2])
    >>> keras.ops.relu(x1)
    array([0.0, 0.0, 1.0, 0.2], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Relu().symbolic_call(x)
    return backend.ops.nn.relu(x)


class Relu6(Operation):
    def call(self, x):
        return backend.ops.nn.relu6(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.relu6", "keras.ops.nn.relu6"])
def relu6(x):
    """Rectified linear unit activation function with upper bound of 6.

    It is defined as `f(x) = np.clip(x, 0, 6)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-3.0, -2.0, 0.1, 0.2, 6.0, 8.0])
    >>> keras.ops.relu6(x)
    array([0.0, 0.0, 0.1, 0.2, 6.0, 6.0], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Relu6().symbolic_call(x)
    return backend.ops.nn.relu6(x)


class Sigmoid(Operation):
    def call(self, x):
        return backend.ops.nn.sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.sigmoid", "keras.ops.nn.sigmoid"])
def sigmoid(x):
    """Sigmoid activation function.

    It is defined as `f(x) = 1 / (1 + exp(-x))`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0])
    >>> keras.ops.sigmoid(x)
    array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Sigmoid().symbolic_call(x)
    return backend.ops.nn.sigmoid(x)


class SparseSigmoid(Operation):
    def call(self, x):
        return backend.ops.nn.sparse_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(["keras.ops.sparse_sigmoid", "keras.ops.nn.sparse_sigmoid"])
def sparse_sigmoid(x):
    """Sparse sigmoid activation function.

    It is defined as

    `f(x) = 0` for `x <= -1`,
    `f(x) = 0.5 * (x + 1)` for `-1 < x < 1`,
    `f(x) = 1` for `x >= 1`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0])
    >>> keras.ops.sparse_sigmoid(x)
    array([0. , 1. , 0.5, 1. , 1. ], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return SparseSigmoid().symbolic_call(x)
    return backend.ops.nn.sparse_sigmoid(x)


class Softplus(Operation):
    def call(self, x):
        return backend.ops.nn.softplus(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.softplus", "keras.ops.nn.softplus"])
def softplus(x):
    """Softplus activation function.

    It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
    logarithm and `exp` is the exponential function.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-0.555, 0.0, 0.555])
    >>> keras.ops.softplus(x)
    array([0.45366603, 0.6931472, 1.008666], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Softplus().symbolic_call(x)
    return backend.ops.nn.softplus(x)


class Softsign(Operation):
    def call(self, x):
        return backend.ops.nn.softsign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.softsign", "keras.ops.nn.softsign"])
def softsign(x):
    """Softsign activation function.

    It is defined as `f(x) = x / (abs(x) + 1)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-0.100, -10.0, 1.0, 0.0, 100.0])
    >>> keras.ops.softsign(x)
    Array([-0.09090909, -0.90909094, 0.5, 0.0, 0.990099], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Softsign().symbolic_call(x)
    return backend.ops.nn.softsign(x)


class SoftShrink(Operation):
    def __init__(self, threshold=0.5, *, name=None):
        super().__init__(name=name)
        self.threshold = threshold

    def call(self, x):
        return backend.ops.nn.soft_shrink(x, self.threshold)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(["keras.ops.soft_shrink", "keras.ops.nn.soft_shrink"])
def soft_shrink(x, threshold=0.5):
    """Soft Shrink activation function.

    It is defined as

    `f(x) = x - threshold` if `x > threshold`,
    `f(x) = x + threshold` if `x < -threshold`,
    `f(x) = 0` otherwise.

    Args:
        x: Input tensor.
        threshold: Threshold value. Defaults to 0.5.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> x_soft_shrink = keras.ops.soft_shrink(x)
    >>> print(x_soft_shrink)
    array([-0.5  0.   0.5], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return SoftShrink(threshold).symbolic_call(x)
    return backend.ops.nn.soft_shrink(x, threshold)


class SparsePlus(Operation):
    def call(self, x):
        return backend.ops.nn.sparse_plus(x)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(["keras.ops.sparse_plus", "keras.ops.nn.sparse_plus"])
def sparse_plus(x):
    """SparsePlus activation function.

    It is defined as

    `f(x) = 0` for `x <= -1`.
    `f(x) = (1/4) * (x + 1)^2` for `-1 < x < 1`.
    `f(x) = x` for `x >= 1`.


    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> x_sparse_plus = keras.ops.sparse_plus(x)
    >>> print(x_sparse_plus)
    Array([0.   0.25 1.  ], shape=(3,), dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return SparsePlus().symbolic_call(x)
    return backend.ops.nn.sparse_plus(x)


class Silu(Operation):
    def call(self, x):
        return backend.ops.nn.silu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.silu",
        "keras.ops.nn.silu",
        "keras.ops.swish",
        "keras.ops.nn.swish",
    ]
)
def silu(x):
    """Sigmoid Linear Unit (SiLU) activation function, also known as Swish.

    The SiLU activation function is computed by the sigmoid function multiplied
    by its input. It is defined as `f(x) = x * sigmoid(x)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0])
    >>> keras.ops.sigmoid(x)
    array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32)
    >>> keras.ops.silu(x)
    array([-0.0148357, 0.7310586, 0.0, 0.7310586, 5.9851646], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Silu().symbolic_call(x)
    return backend.ops.nn.silu(x)


class Squareplus(Operation):
    def __init__(self, b=4, *, name=None):
        super().__init__(name=name)
        self.b = b

    def call(self, x):
        return _squareplus(x, self.b)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.squareplus", "keras.ops.nn.squareplus"])
def squareplus(x, b=4):
    """Squareplus activation function.

    The Squareplus activation function is defined as:

    `f(x) = (x + sqrt(x^2 + b)) / 2`

    Args:
        x: Input tensor.
        b: Smoothness parameter. Defaults to 4.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> x_squareplus = keras.ops.squareplus(x)
    >>> print(x_squareplus)
    array([0.6180, 1.0000, 1.6180], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Squareplus(b).symbolic_call(x)
    return _squareplus(x, b)


def _squareplus(x, b):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.nn, "squareplus"
    ):
        return backend.ops.nn.squareplus(x, b)
    x = backend.ops.convert_to_tensor(x)
    b = backend.ops.convert_to_tensor(b, dtype=x.dtype)
    y = (x + backend.ops.numpy.sqrt(backend.ops.numpy.square(x) + b)) / 2.0
    return backend.ops.cast(y, dtype=x.dtype)


class LogSigmoid(Operation):
    def call(self, x):
        return backend.ops.nn.log_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.log_sigmoid",
        "keras.ops.nn.log_sigmoid",
    ]
)
def log_sigmoid(x):
    """Logarithm of the sigmoid activation function.

    It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-0.541391, 0.0, 0.50, 5.0])
    >>> keras.ops.log_sigmoid(x)
    array([-1.0000418, -0.6931472, -0.474077, -0.00671535], dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return LogSigmoid().symbolic_call(x)
    return backend.ops.nn.log_sigmoid(x)


class LeakyRelu(Operation):
    def __init__(self, negative_slope=0.2, *, name=None):
        super().__init__(name=name)
        self.negative_slope = negative_slope

    def call(self, x):
        return backend.ops.nn.leaky_relu(x, self.negative_slope)

    def compute_output_spec(self, x):
        # Integer and bool input is promoted to `floatx`, matching what the
        # backends return. Same rule as `Exp.compute_output_spec`.
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(["keras.ops.leaky_relu", "keras.ops.nn.leaky_relu"])
def leaky_relu(x, negative_slope=0.2):
    """Leaky version of a Rectified Linear Unit activation function.

    It allows a small gradient when the unit is not active, it is defined as:

    `f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.

    Args:
        x: Input tensor.
        negative_slope: Slope of the activation function at x < 0.
            Defaults to `0.2`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_leaky_relu = keras.ops.leaky_relu(x)
    >>> print(x_leaky_relu)
    array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return LeakyRelu(negative_slope).symbolic_call(x)
    return backend.ops.nn.leaky_relu(x, negative_slope=negative_slope)


class HardSigmoid(Operation):
    def call(self, x):
        return backend.ops.nn.hard_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(
    [
        "keras.ops.hard_sigmoid",
        "keras.ops.nn.hard_sigmoid",
    ]
)
def hard_sigmoid(x):
    """Hard sigmoid activation function.

    It is defined as:

    `0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_hard_sigmoid = keras.ops.hard_sigmoid(x)
    >>> print(x_hard_sigmoid)
    array([0.3, 0.5, 0.7], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return HardSigmoid().symbolic_call(x)
    return backend.ops.nn.hard_sigmoid(x)


class HardSilu(Operation):
    def call(self, x):
        return backend.ops.nn.hard_silu(x)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(
    [
        "keras.ops.hard_silu",
        "keras.ops.nn.hard_silu",
        "keras.ops.hard_swish",
        "keras.ops.nn.hard_swish",
    ]
)
def hard_silu(x):
    """Hard SiLU activation function, also known as Hard Swish.

    It is defined as:

    - `0` if `if x < -3`
    - `x` if `x > 3`
    - `x * (x + 3) / 6` if `-3 <= x <= 3`

    It's a faster, piecewise linear approximation of the silu activation.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> keras.ops.hard_silu(x)
    array([-0.0, -0.3333333, 0.0, 0.6666667, 3.0], shape=(5,), dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return HardSilu().symbolic_call(x)
    return backend.ops.nn.hard_silu(x)


class Elu(Operation):
    def __init__(self, alpha=1.0, *, name=None):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, x):
        return backend.ops.nn.elu(x, alpha=self.alpha)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.elu", "keras.ops.nn.elu"])
def elu(x, alpha=1.0):
    """Exponential Linear Unit activation function.

    It is defined as:

    `f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.

    Args:
        x: Input tensor.
        alpha: A scalar, slope of positive section. Defaults to `1.0`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_elu = keras.ops.elu(x)
    >>> print(x_elu)
    array([-0.63212055, 0., 1.], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Elu(alpha).symbolic_call(x)
    return backend.ops.nn.elu(x, alpha=alpha)


class Selu(Operation):
    def call(self, x):
        return backend.ops.nn.selu(x)

    def compute_output_spec(self, x):
        return KerasTensor(
            x.shape, dtype=dtypes.promote_to_float_dtype(x.dtype)
        )


@keras_export(["keras.ops.selu", "keras.ops.nn.selu"])
def selu(x):
    """Scaled Exponential Linear Unit (SELU) activation function.

    It is defined as:

    `f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = scale * x for x >= 0`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_selu = keras.ops.selu(x)
    >>> print(x_selu)
    array([-1.11133055, 0., 1.05070098], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Selu().symbolic_call(x)
    return backend.ops.nn.selu(x)


class Gelu(Operation):
    def __init__(self, approximate=True, *, name=None):
        super().__init__(name=name)
        self.approximate = approximate

    def call(self, x):
        return backend.ops.nn.gelu(x, self.approximate)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.gelu", "keras.ops.nn.gelu"])
def gelu(x, approximate=True):
    """Gaussian Error Linear Unit (GELU) activation function.

    If `approximate` is `True`, it is defined as:
    `f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`

    Or if `approximate` is `False`, it is defined as:
    `f(x) = x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
    where `P(X) ~ N(0, 1)`.

    Args:
        x: Input tensor.
        approximate: Approximate version of GELU activation. Defaults to `True`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_gelu = keras.ops.gelu(x)
    >>> print(x_gelu)
    array([-0.15865525, 0., 0.84134475], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Gelu(approximate).symbolic_call(x)
    return backend.ops.nn.gelu(x, approximate)


class Celu(Operation):
    def __init__(self, alpha=1.0, *, name=None):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, x):
        return backend.ops.nn.celu(x, self.alpha)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.celu", "keras.ops.nn.celu"])
def celu(x, alpha=1.0):
    """Continuously-differentiable exponential linear unit.

    It is defined as:

    `f(x) =  alpha * (exp(x / alpha) - 1) for x < 0`, `f(x) = x for x >= 0`.

    Args:
        x: Input tensor.
        alpha: the α value for the CELU formulation. Defaults to `1.0`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_celu = keras.ops.celu(x)
    >>> print(x_celu)
    array([-0.63212056, 0. , 1. ], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Celu(alpha).symbolic_call(x)
    return backend.ops.nn.celu(x, alpha)


class Glu(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.nn.glu(x, axis=self.axis)

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        canonicalize_axis(self.axis, len(output_shape))
        if output_shape[self.axis] is not None:
            if output_shape[self.axis] % 2 != 0:
                raise ValueError(
                    "axis size must be divisible by 2. "
                    f"Received: x.shape={x.shape} with axis={self.axis}"
                )
            output_shape[self.axis] = output_shape[self.axis] // 2
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.glu", "keras.ops.nn.glu"])
def glu(x, axis=-1):
    """Gated Linear Unit (GLU) activation function.

    It is defined as:

    `f(x) = a * sigmoid(b)`
    where `x` is split into `a` and `b` along the given axis.

    Args:
        x: Input tensor.
        axis: The axis along which to split the input tensor. Defaults to `-1`.

    Returns:
        A tensor with the same shape as half of the input.

    Example:

    >>> x = np.array([-1., 0., 1. , 1.])
    >>> x_glu = keras.ops.glu(x)
    >>> print(x_glu)
    array([-0.73105858, 0. ], shape=(2,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Glu(axis).symbolic_call(x)
    return backend.ops.nn.glu(x, axis=axis)


class TanhShrink(Operation):
    def call(self, x):
        return backend.ops.nn.tanh_shrink(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.tanh_shrink", "keras.ops.nn.tanh_shrink"])
def tanh_shrink(x):
    """Applies the tanh shrink function element-wise.

    It is defined as:

    `f(x) = x - tanh(x)`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of the same shape as `x`, where each element is
        transformed according to the tanh shrink operation.

    Example:

    >>> x = np.array([ -1., 0., 1.])
    >>> x_tanh_shrink = keras.ops.tanh_shrink(x)
    >>> print(x_tanh_shrink)
    array([-0.23840584  0.  0.23840584], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return TanhShrink().symbolic_call(x)
    return backend.ops.nn.tanh_shrink(x)


class HardTanh(Operation):
    def call(self, x):
        return backend.ops.nn.hard_tanh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.hard_tanh", "keras.ops.nn.hard_tanh"])
def hard_tanh(x):
    """Applies the HardTanh function element-wise.

    It is defined as:

    `f(x) = -1 for x < -1`, `f(x) = x for -1 <= x <= 1`, `f(x) = 1 for x > 1`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`
        where values are clamped between -1 and 1.

    Example:

    >>> x = np.array([-2., -1., 0., 1., 2.])
    >>> x_hard_tanh = keras.ops.hard_tanh(x)
    >>> print(x_hard_tanh)
    array([-1. -1.  0.  1.  1.], shape=(5,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return HardTanh().symbolic_call(x)
    return backend.ops.nn.hard_tanh(x)


class HardShrink(Operation):
    def __init__(self, threshold=0.5, *, name=None):
        super().__init__(name=name)
        self.threshold = threshold

    def call(self, x):
        return backend.ops.nn.hard_shrink(x, self.threshold)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.hard_shrink", "keras.ops.nn.hard_shrink"])
def hard_shrink(x, threshold=0.5):
    """Hard Shrink activation function.

    The Hard Shrink function is a thresholding operation defined as:

    `f(x) = x` if `|x| > threshold`,
    `f(x) = 0` otherwise.

    Args:
        x: Input tensor.
        threshold: Threshold value. Defaults to 0.5.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-0.5, 0., 1.])
    >>> x_hard_shrink = keras.ops.hard_shrink(x)
    >>> print(x_hard_shrink)
    array([0. 0. 1.], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return HardShrink(threshold).symbolic_call(x)
    return backend.ops.nn.hard_shrink(x, threshold)


class Threshold(Operation):
    def __init__(self, threshold, default_value, *, name=None):
        super().__init__(name=name)
        self.threshold = threshold
        self.default_value = default_value

    def call(self, x):
        return backend.ops.nn.threshold(x, self.threshold, self.default_value)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.threshold", "keras.ops.nn.threshold"])
def threshold(x, threshold, default_value):
    """Threshold activation function.

    The function thresholds the input `x` as follows:
    `f(x) = x` if `x > threshold`,
    `f(x) = default_value` otherwise.

    Args:
        x: Input tensor.
        threshold: The value that decides when to retain or replace x.
        default_value: Value to assign when `x <= threshold`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1.0, 0.0, 1.0, 2.0])
    >>> x_threshold = keras.ops.threshold(x, 1, 0)
    >>> print(x_threshold)
    array([0., 0., 0., 2.], shape=(4,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Threshold(threshold, default_value).symbolic_call(x)
    return backend.ops.nn.threshold(x, threshold, default_value)


class Softmax(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.nn.softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is not None:
            canonicalize_axes(self.axis, len(x.shape))
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.softmax", "keras.ops.nn.softmax"])
def softmax(x, axis=-1):
    """Softmax activation function.

    The elements of the output vector lie within the range `(0, 1)`, and their
    total sum is exactly 1 (excluding the floating point rounding error).

    Each vector is processed independently. The `axis` argument specifies the
    axis along which the function is applied within the input.

    It is defined as:
    `f(x) = exp(x) / sum(exp(x))`

    Args:
        x: Input tensor.
        axis: Integer, axis along which the softmax is applied.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_softmax = keras.ops.softmax(x)
    >>> print(x_softmax)
    array([0.09003057, 0.24472847, 0.66524096], shape=(3,), dtype=float64)

    """
    # Don't use `backend.ops.shape` since TensorFlow returns
    # symbolic tensors for unknown shape which can trigger
    # an error in TensorFlow graph execution.
    if (
        isinstance(axis, int)
        and -len(x.shape) <= axis < len(x.shape)
        and x.shape[axis] == 1
    ):
        warnings.warn(
            f"You are using a softmax over axis {axis} "
            f"of a tensor of shape {x.shape}. This axis "
            "has size 1. The softmax operation will always return "
            "the value 1, which is likely not what you intended. "
            "Did you mean to use a sigmoid instead?"
        )
    if any_symbolic_tensors((x,)):
        return Softmax(axis).symbolic_call(x)
    if isinstance(axis, tuple):
        axis_to_keep = [v for v in range(len(x.shape)) if v not in axis]

        x_transposed = backend.ops.numpy.transpose(
            x, axes=(*axis_to_keep, *axis)
        )
        x_reshaped = backend.ops.numpy.reshape(
            x_transposed, (*[x.shape[v] for v in axis_to_keep], -1)
        )

        x = backend.ops.nn.softmax(x_reshaped, axis=-1)

        x = backend.ops.numpy.reshape(x, x_transposed.shape)
        combined = [*axis_to_keep, *axis]
        x = backend.ops.numpy.transpose(
            x,
            axes=sorted(range(len(combined)), key=combined.__getitem__),
        )
        return x
    else:
        return backend.ops.nn.softmax(x, axis=axis)


class LogSoftmax(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.nn.log_softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is not None:
            canonicalize_axes(self.axis, len(x.shape))
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.log_softmax",
        "keras.ops.nn.log_softmax",
    ]
)
def log_softmax(x, axis=-1):
    """Log-softmax activation function.

    It is defined as:
    `f(x) = x - max(x) - log(sum(exp(x - max(x))))`

    Args:
        x: Input tensor.
        axis: Integer, axis along which the log-softmax is applied.
            Defaults to `-1`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_log_softmax = keras.ops.log_softmax(x)
    >>> print(x_log_softmax)
    array([-2.40760596, -1.40760596, -0.40760596], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return LogSoftmax(axis).symbolic_call(x)
    if isinstance(axis, tuple):
        axis_to_keep = [v for v in range(len(x.shape)) if v not in axis]

        x_transposed = backend.ops.numpy.transpose(
            x, axes=(*axis_to_keep, *axis)
        )
        x_reshaped = backend.ops.numpy.reshape(
            x_transposed, (*[x.shape[v] for v in axis_to_keep], -1)
        )

        x = backend.ops.nn.log_softmax(x_reshaped, axis=-1)

        x = backend.ops.numpy.reshape(x, x_transposed.shape)
        combined = [*axis_to_keep, *axis]
        x = backend.ops.numpy.transpose(
            x,
            axes=sorted(range(len(combined)), key=combined.__getitem__),
        )
        return x
    else:
        return backend.ops.nn.log_softmax(x, axis=axis)


class Sparsemax(Operation):
    def __init__(self, axis=-1, *, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        return backend.ops.nn.sparsemax(x, axis=self.axis)

    def compute_output_spec(self, x):
        canonicalize_axis(self.axis, len(x.shape))
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.sparsemax", "keras.ops.nn.sparsemax"])
def sparsemax(x, axis=-1):
    """Sparsemax activation function.

    For each batch `i`, and class `j`,
    sparsemax activation function is defined as:

    `sparsemax(x)[i, j] = max(x[i, j] - τ(x[i, :]), 0).`

    Args:
        x: Input tensor.
        axis: `int`, axis along which the sparsemax operation is applied.

    Returns:
        A tensor, output of sparsemax transformation. Has the same type and
        shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_sparsemax = keras.ops.sparsemax(x)
    >>> print(x_sparsemax)
    array([0., 0., 1.], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Sparsemax(axis).symbolic_call(x)
    return backend.ops.nn.sparsemax(x, axis=axis)


class MaxPool(Operation):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format

    def call(self, inputs):
        return backend.ops.nn.max_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        output_shape = operation_utils.compute_pooling_output_shape(
            inputs.shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(["keras.ops.max_pool", "keras.ops.nn.max_pool"])
def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Max pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        pool_size: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`, specifying the size of the pooling
            window for each spatial dimension of the input tensor. If
            `pool_size` is int, then every spatial dimension shares the same
            `pool_size`.
        strides: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`. The stride of the sliding window for
            each spatial dimension of the input tensor. If `strides` is int,
            then every spatial dimension shares the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.

    Returns:
        A tensor of rank N+2, the result of the max pooling operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return MaxPool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(inputs)
    return backend.ops.nn.max_pool(
        inputs, pool_size, strides, padding, data_format
    )


class AdaptiveMaxPool(Operation):
    """Adaptive max pooling operation."""

    def __init__(self, output_size, data_format=None, *, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.data_format = standardize_data_format(data_format)

    def call(self, inputs):
        return backend.ops.nn.adaptive_max_pool(
            inputs, output_size=self.output_size, data_format=self.data_format
        )

    def compute_output_spec(self, inputs):
        num_spatial_dims = len(inputs.shape) - 2
        spatial_dims = standardize_tuple(
            self.output_size, num_spatial_dims, "output_size"
        )
        if self.data_format == "channels_last":
            output_shape = (
                (inputs.shape[0],) + spatial_dims + (inputs.shape[-1],)
            )
        else:
            output_shape = (inputs.shape[0], inputs.shape[1]) + spatial_dims
        return backend.KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(["keras.ops.adaptive_max_pool", "keras.ops.nn.adaptive_max_pool"])
def adaptive_max_pool(
    inputs,
    output_size,
    data_format=None,
):
    """Adaptive max pooling operation.

    Applies an adaptive max pooling operation that automatically computes the
    kernel size and stride to pool the input to the specified `output_size`.
    This operation is useful when you want a fixed output size regardless of
    input size, commonly used in models like ResNet for global feature
    extraction.
    Args:
        inputs: Tensor of rank 4. Input tensor of shape:
            - If `data_format="channels_last"`:
                `(batch_size, height, width, channels)`.
            - If `data_format="channels_first"`:
                `(batch_size, channels, height, width)`.
        output_size: Integer or tuple/list of 2 integers, specifying the target
            output spatial dimensions `(output_height, output_width)`. If a
            single
            integer is provided, the same value is used for both dimensions.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, defaults to `"channels_last"`.

    Returns:
        A tensor of rank 4 representing the adaptive max pooled result.

    Example:

    >>> x = np.random.rand(2, 64, 64, 3)
    >>> y = keras.ops.adaptive_max_pool(x, output_size=(32, 32))
    >>> y.shape
    (2, 32, 32, 3)

    >>> # Works with any input size
    >>> x = np.random.rand(2, 100, 80, 3)
    >>> y = keras.ops.adaptive_max_pool(x, output_size=7)
    >>> y.shape
    (2, 7, 7, 3)
    """
    data_format = standardize_data_format(data_format)

    if any_symbolic_tensors((inputs,)):
        return AdaptiveMaxPool(output_size, data_format).symbolic_call(inputs)

    return backend.ops.nn.adaptive_max_pool(
        inputs, output_size=output_size, data_format=data_format
    )


class AveragePool(Operation):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format

    def call(self, inputs):
        return backend.ops.nn.average_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        output_shape = operation_utils.compute_pooling_output_shape(
            inputs.shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(
    [
        "keras.ops.average_pool",
        "keras.ops.nn.average_pool",
    ]
)
def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Average pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        pool_size: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`, specifying the size of the pooling
            window for each spatial dimension of the input tensor. If
            `pool_size` is int, then every spatial dimension shares the same
            `pool_size`.
        strides: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`. The stride of the sliding window for
            each spatial dimension of the input tensor. If `strides` is int,
            then every spatial dimension shares the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.

    Returns:
        A tensor of rank N+2, the result of the average pooling operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return AveragePool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(inputs)
    return backend.ops.nn.average_pool(
        inputs, pool_size, strides, padding, data_format
    )


class AdaptiveAveragePool(Operation):
    """Adaptive average pooling operation."""

    def __init__(self, output_size, data_format=None, *, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.data_format = standardize_data_format(data_format)

    def call(self, inputs):
        return backend.ops.nn.adaptive_average_pool(
            inputs, output_size=self.output_size, data_format=self.data_format
        )

    def compute_output_spec(self, inputs):
        num_spatial_dims = len(inputs.shape) - 2
        spatial_dims = standardize_tuple(
            self.output_size, num_spatial_dims, "output_size"
        )
        if self.data_format == "channels_last":
            output_shape = (
                (inputs.shape[0],) + spatial_dims + (inputs.shape[-1],)
            )
        else:
            output_shape = (inputs.shape[0], inputs.shape[1]) + spatial_dims
        return backend.KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(
    ["keras.ops.adaptive_average_pool", "keras.ops.nn.adaptive_average_pool"]
)
def adaptive_average_pool(
    inputs,
    output_size,
    data_format=None,
):
    """Adaptive average pooling operation.

    Applies an adaptive average pooling operation that automatically
    computes the kernel size and stride to pool the input to the
    specified `output_size`. This operation is useful when you want a
    fixed output size regardless of input size, commonly used in models
    like ResNet for global feature extraction.

    Args:
        inputs: Tensor of rank 4. Input tensor of shape:
            - If `data_format="channels_last"`:
                `(batch_size, height, width, channels)`.
            - If `data_format="channels_first"`:
                `(batch_size, channels, height, width)`.
        output_size: Integer or tuple/list of 2 integers, specifying the target
            output spatial dimensions `(output_height, output_width)`. If a
            single
            integer is provided, the same value is used for both dimensions.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, defaults to `"channels_last"`.

    Returns:
        A tensor of rank 4 representing the adaptive average pooled result.

    Example:

    >>> x = np.random.rand(2, 64, 64, 3)
    >>> y = keras.ops.adaptive_average_pool(x, output_size=(32, 32))
    >>> y.shape
    (2, 32, 32, 3)

    >>> # Works with any input size
    >>> x = np.random.rand(2, 100, 80, 3)
    >>> y = keras.ops.adaptive_average_pool(x, output_size=7)
    >>> y.shape
    (2, 7, 7, 3)
    """
    data_format = standardize_data_format(data_format)

    if any_symbolic_tensors((inputs,)):
        return AdaptiveAveragePool(output_size, data_format).symbolic_call(
            inputs
        )

    return backend.ops.nn.adaptive_average_pool(
        inputs, output_size=output_size, data_format=data_format
    )


class Conv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, kernel):
        return backend.ops.nn.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        data_format = standardize_data_format(self.data_format)
        check_conv_input_channels(inputs, kernel, data_format)
        output_shape = operation_utils.compute_conv_output_shape(
            inputs.shape,
            kernel.shape[-1],
            kernel.shape[:-2],
            self.strides,
            self.padding,
            data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(["keras.ops.conv", "keras.ops.nn.conv"])
def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D convolution.

    This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            `(kernel_spatial_shape, num_input_channels, num_output_channels)`.
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return Conv(strides, padding, data_format, dilation_rate).symbolic_call(
            inputs, kernel
        )
    return backend.ops.nn.conv(
        inputs, kernel, strides, padding, data_format, dilation_rate
    )


class DepthwiseConv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, kernel):
        return backend.ops.nn.depthwise_conv(
            inputs,
            kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        data_format = standardize_data_format(self.data_format)
        check_conv_input_channels(inputs, kernel, data_format)
        output_shape = operation_utils.compute_conv_output_shape(
            inputs.shape,
            kernel.shape[-1] * kernel.shape[-2],
            kernel.shape[:-2],
            self.strides,
            self.padding,
            data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(
    [
        "keras.ops.depthwise_conv",
        "keras.ops.nn.depthwise_conv",
    ]
)
def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D depthwise convolution.

    This ops supports 1D and 2D depthwise convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs, kernel)):
        return DepthwiseConv(
            strides, padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return backend.ops.nn.depthwise_conv(
        inputs,
        kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )


class SeparableConv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, depthwise_kernel, pointwise_kernel):
        return backend.ops.nn.separable_conv(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, depthwise_kernel, pointwise_kernel):
        data_format = standardize_data_format(self.data_format)
        check_conv_input_channels(inputs, depthwise_kernel, data_format)
        output_shape = list(
            depthwise_conv(
                inputs,
                depthwise_kernel,
                self.strides,
                self.padding,
                data_format,
                self.dilation_rate,
            ).shape
        )
        if data_format == "channels_last":
            output_shape[-1] = pointwise_kernel.shape[-1]
        else:
            output_shape[1] = pointwise_kernel.shape[-1]
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(
    [
        "keras.ops.separable_conv",
        "keras.ops.nn.separable_conv",
    ]
)
def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D separable convolution.

    This ops supports 1D and 2D separable convolution. `separable_conv` is
    a depthwise conv followed by a pointwise conv.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        depthwise_kernel: Tensor of rank N+2. `depthwise_kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        pointwise_kernel: Tensor of rank N+2. `pointwise_kernel` has shape
            `(*ones_like(kernel_spatial_shape),
            num_input_channels * num_channels_multiplier, num_output_channels)`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return SeparableConv(
            strides,
            padding,
            data_format,
            dilation_rate,
        ).symbolic_call(inputs, depthwise_kernel, pointwise_kernel)
    return backend.ops.nn.separable_conv(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )


class ConvTranspose(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.strides = strides
        self.output_padding = output_padding
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(
        self,
        inputs,
        kernel,
    ):
        return backend.ops.nn.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        data_format = standardize_data_format(self.data_format)
        check_conv_transpose_input_channels(inputs, kernel, data_format)
        kernel_size = kernel.shape[:-2]
        filters = kernel.shape[-2]
        output_shape = compute_conv_transpose_output_shape(
            inputs.shape,
            kernel_size,
            filters,
            self.strides,
            self.padding,
            self.output_padding,
            data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_export(
    [
        "keras.ops.conv_transpose",
        "keras.ops.nn.conv_transpose",
    ]
)
def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    """General N-D convolution transpose.

    Also known as de-convolution. This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_output_channels, num_input_channels],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        output_padding: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the amount of padding along the height and width of
            the output tensor. Can be a single integer to specify the same
            value for all spatial dimensions. The amount of output padding
            along a given dimension must be lower than the stride along that
            same dimension. If set to `None` (default), the output shape is
            inferred.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
    data_format = standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return ConvTranspose(
            strides, padding, output_padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return backend.ops.nn.conv_transpose(
        inputs,
        kernel,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )


class OneHot(Operation):
    def __init__(
        self, num_classes, axis=-1, dtype=None, sparse=False, *, name=None
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.axis = axis
        self.dtype = dtype
        self.sparse = sparse

    def call(self, x):
        return backend.ops.nn.one_hot(
            x,
            self.num_classes,
            axis=self.axis,
            dtype=self.dtype,
            sparse=self.sparse,
        )

    def compute_output_spec(self, x):
        x_shape = list(getattr(x, "shape", []))
        axis = canonicalize_axis(self.axis, len(x_shape) + 1)
        x_shape.insert(axis, self.num_classes)
        return KerasTensor(x_shape, dtype=self.dtype, sparse=self.sparse)


@keras_export(["keras.ops.one_hot", "keras.ops.nn.one_hot"])
def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    """Converts integer tensor `x` into a one-hot tensor.

    The one-hot encoding is a representation where each integer value is
    converted into a binary vector with a length equal to `num_classes`,
    and the index corresponding to the integer value is marked as 1, while
    all other indices are marked as 0.

    Args:
        x: Integer tensor to be encoded. The shape can be
            arbitrary, but the dtype should be integer.
        num_classes: Number of classes for the one-hot encoding.
        axis: Axis along which the encoding is performed.
            `-1` represents the last axis. Defaults to `-1`.
        dtype: (Optional) Data type of the output tensor. If not
            provided, it defaults to the default data type of the backend.
        sparse: Whether to return a sparse tensor; for backends that support
            sparse tensors.

    Returns:
        Integer tensor: One-hot encoded tensor with the same shape as `x`
        except for the specified `axis` dimension, which will have
        a length of `num_classes`. The dtype of the output tensor
        is determined by `dtype` or the default data type of the backend.

    Example:

    >>> x = keras.ops.convert_to_tensor([1, 3, 2, 0])
    >>> one_hot(x, num_classes=4)
    array([[0. 1. 0. 0.]
           [0. 0. 0. 1.]
           [0. 0. 1. 0.]
           [1. 0. 0. 0.]], shape=(4, 4), dtype=float32)
    """
    dtype = standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return OneHot(
            num_classes, axis=axis, dtype=dtype, sparse=sparse
        ).symbolic_call(x)
    return backend.ops.nn.one_hot(
        x,
        num_classes,
        axis=axis,
        dtype=dtype or backend.floatx(),
        sparse=sparse,
    )


class BinaryCrossentropy(Operation):
    def __init__(self, from_logits=False, *, name=None):
        super().__init__(name=name)
        self.from_logits = from_logits

    def call(self, target, output):
        return backend.ops.nn.binary_crossentropy(
            target, output, from_logits=self.from_logits
        )

    def compute_output_spec(self, target, output):
        if target.shape != output.shape:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        dtype = standardize_dtype(output.dtype)
        if not is_float_dtype(dtype):
            dtype = backend.floatx()
        return KerasTensor(output.shape, dtype=dtype)


@keras_export(
    [
        "keras.ops.binary_crossentropy",
        "keras.ops.nn.binary_crossentropy",
    ]
)
def binary_crossentropy(target, output, from_logits=False):
    """Computes binary cross-entropy loss between target and output tensor.

    The binary cross-entropy loss is commonly used in binary
    classification tasks where each input sample belongs to one
    of the two classes. It measures the dissimilarity between the
    target and output probabilities or logits.

    Args:
        target: The target tensor representing the true binary labels.
            Its shape should match the shape of the `output` tensor.
        output: The output tensor representing the predicted probabilities
            or logits. Its shape should match the shape of the
            `target` tensor.
        from_logits: (optional) Whether `output` is a tensor of logits or
            probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Defaults to `False`.

    Returns:
        Floating-point tensor: The computed binary cross-entropy loss between
        `target` and `output`.

    Example:

    >>> target = keras.ops.convert_to_tensor([0, 1, 1, 0])
    >>> output = keras.ops.convert_to_tensor([0.1, 0.9, 0.8, 0.2])
    >>> binary_crossentropy(target, output)
    array([0.10536054 0.10536054 0.22314355 0.22314355],
          shape=(4,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return BinaryCrossentropy(from_logits=from_logits).symbolic_call(
            target, output
        )
    return backend.ops.nn.binary_crossentropy(
        target, output, from_logits=from_logits
    )


class CategoricalCrossentropy(Operation):
    def __init__(self, from_logits=False, axis=-1, *, name=None):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.axis = axis

    def call(self, target, output):
        return backend.ops.nn.categorical_crossentropy(
            target, output, from_logits=self.from_logits, axis=self.axis
        )

    def compute_output_spec(self, target, output):
        if target.shape != output.shape:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        if len(target.shape) < 1:
            raise ValueError(
                "Arguments `target` and `output` must be at least rank 1. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        axis = canonicalize_axis(self.axis, len(output.shape))
        return KerasTensor(
            output.shape[:axis] + output.shape[axis + 1 :], dtype=output.dtype
        )


@keras_export(
    [
        "keras.ops.categorical_crossentropy",
        "keras.ops.nn.categorical_crossentropy",
    ]
)
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Computes categorical cross-entropy loss between target and output tensor.

    The categorical cross-entropy loss is commonly used in multi-class
    classification tasks where each input sample can belong to one of
    multiple classes. It measures the dissimilarity
    between the target and output probabilities or logits.

    Args:
        target: The target tensor representing the true categorical labels.
            Its shape should match the shape of the `output` tensor
            except for the last dimension.
        output: The output tensor representing the predicted probabilities
            or logits. Its shape should match the shape of the `target`
            tensor except for the last dimension.
        from_logits: (optional) Whether `output` is a tensor of logits or
            probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Defaults to `False`.
        axis: (optional) The axis along which the categorical cross-entropy
            is computed.
            Defaults to `-1`, which corresponds to the last dimension of
            the tensors.

    Returns:
        Integer tensor: The computed categorical cross-entropy loss between
        `target` and `output`.

    Example:

    >>> target = keras.ops.convert_to_tensor(
    ... [[1, 0, 0],
    ...  [0, 1, 0],
    ...  [0, 0, 1]])
    >>> output = keras.ops.convert_to_tensor(
    ... [[0.9, 0.05, 0.05],
    ...  [0.1, 0.8, 0.1],
    ...  [0.2, 0.3, 0.5]])
    >>> categorical_crossentropy(target, output)
    array([0.10536054 0.22314355 0.6931472 ], shape=(3,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return CategoricalCrossentropy(
            from_logits=from_logits, axis=axis
        ).symbolic_call(target, output)
    return backend.ops.nn.categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis
    )


class SparseCategoricalCrossentropy(Operation):
    def __init__(self, from_logits=False, axis=-1, *, name=None):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.axis = axis

    def call(self, target, output):
        return backend.ops.nn.sparse_categorical_crossentropy(
            target, output, from_logits=self.from_logits, axis=self.axis
        )

    def compute_output_spec(self, target, output):
        if len(output.shape) < 1:
            raise ValueError(
                "Argument `output` must be at least rank 1. "
                "Received: "
                f"output.shape={output.shape}"
            )
        axis = canonicalize_axis(self.axis, len(output.shape))
        target_shape = target.shape
        if len(target_shape) == len(output.shape) and target_shape[-1] == 1:
            target_shape = target_shape[:-1]
        output_shape_without_class = (
            output.shape[:axis] + output.shape[axis + 1 :]
        )
        if target_shape != output_shape_without_class:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape "
                "up until the last dimension: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        return KerasTensor(output_shape_without_class, dtype=output.dtype)


@keras_export(
    [
        "keras.ops.sparse_categorical_crossentropy",
        "keras.ops.nn.sparse_categorical_crossentropy",
    ]
)
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Computes sparse categorical cross-entropy loss.

    The sparse categorical cross-entropy loss is similar to categorical
    cross-entropy, but it is used when the target tensor contains integer
    class labels instead of one-hot encoded vectors. It measures the
    dissimilarity between the target and output probabilities or logits.

    Args:
        target: The target tensor representing the true class labels as
            integers. Its shape should match the shape of the `output`
            tensor except for the last dimension.
        output: The output tensor representing the predicted probabilities
            or logits.
            Its shape should match the shape of the `target` tensor except
            for the last dimension.
        from_logits: (optional) Whether `output` is a tensor of logits
            or probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Defaults to `False`.
        axis: (optional) The axis along which the sparse categorical
            cross-entropy is computed.
            Defaults to `-1`, which corresponds to the last dimension
            of the tensors.

    Returns:
        Integer tensor: The computed sparse categorical cross-entropy
        loss between `target` and `output`.

    Example:

    >>> target = keras.ops.convert_to_tensor([0, 1, 2], dtype=int32)
    >>> output = keras.ops.convert_to_tensor(
    ... [[0.9, 0.05, 0.05],
    ...  [0.1, 0.8, 0.1],
    ...  [0.2, 0.3, 0.5]])
    >>> sparse_categorical_crossentropy(target, output)
    array([0.10536056 0.22314355 0.6931472 ], shape=(3,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return SparseCategoricalCrossentropy(
            from_logits=from_logits, axis=axis
        ).symbolic_call(target, output)
    return backend.ops.nn.sparse_categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis
    )


class MultiHot(Operation):
    def __init__(
        self,
        num_classes=None,
        axis=-1,
        dtype=None,
        sparse=False,
        *,
        name=None,
        **kwargs,
    ):
        if num_classes is None and "num_tokens" in kwargs:
            num_classes = kwargs.pop("num_tokens")
        if num_classes is None:
            raise ValueError("Argument `num_classes` must be specified.")
        super().__init__(name=name)
        self.num_classes = num_classes
        self.axis = axis
        self.dtype = dtype or backend.floatx()
        self.sparse = sparse

    def call(self, inputs):
        return backend.ops.nn.multi_hot(
            inputs,
            num_classes=self.num_classes,
            axis=self.axis,
            dtype=self.dtype,
            sparse=self.sparse,
        )

    def compute_output_spec(self, inputs):
        x_shape = list(getattr(inputs, "shape", []))
        input_rank = len(x_shape)
        axis = canonicalize_axis(self.axis, input_rank + 1)
        x_shape.insert(axis, self.num_classes)
        reduction_axis = 1 if input_rank > 1 else 0
        del x_shape[reduction_axis]

        return KerasTensor(x_shape, dtype=self.dtype, sparse=self.sparse)


@keras_export(
    [
        "keras.ops.multi_hot",
        "keras.ops.nn.multi_hot",
    ]
)
def multi_hot(
    inputs, num_classes=None, axis=-1, dtype=None, sparse=False, **kwargs
):
    """Encodes integer labels as multi-hot vectors.

    This function encodes integer labels as multi-hot vectors, where each label
    is mapped to a binary value in the resulting vector.

    Args:
        inputs: Tensor of integer labels to be converted to multi-hot vectors.
        num_classes: Integer, the total number of unique classes.
        axis: (optional) Axis along which the multi-hot encoding should be
            added. Defaults to `-1`, which corresponds to the last dimension.
        dtype: (optional) The data type of the resulting tensor. Default
            is backend's float type.
        sparse: Whether to return a sparse tensor; for backends that support
            sparse tensors.

    Returns:
        Tensor: The multi-hot encoded tensor.

    Example:

    >>> data = keras.ops.convert_to_tensor([0, 4])
    >>> keras.ops.multi_hot(data, num_classes=5)
    array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=float32)

    """
    if num_classes is None and "num_tokens" in kwargs:
        num_classes = kwargs.pop("num_tokens")
    if num_classes is None:
        raise ValueError("Argument `num_classes` must be specified.")

    dtype = standardize_dtype(dtype)
    if any_symbolic_tensors((inputs,)):
        return MultiHot(num_classes, axis, dtype, sparse).symbolic_call(inputs)

    return backend.ops.nn.multi_hot(inputs, num_classes, axis, dtype, sparse)


class Moments(Operation):
    def __init__(self, axes, keepdims=False, synchronized=False, *, name=None):
        super().__init__(name=name)
        self.axes = axes
        self.keepdims = keepdims
        self.synchronized = synchronized

    def call(self, x):
        return backend.ops.nn.moments(
            x,
            axes=self.axes,
            keepdims=self.keepdims,
            synchronized=self.synchronized,
        )

    def compute_output_spec(self, x):
        return (
            KerasTensor(
                reduce_shape(x.shape, axis=self.axes, keepdims=self.keepdims),
                dtype=x.dtype,
            ),
            KerasTensor(
                reduce_shape(x.shape, axis=self.axes, keepdims=self.keepdims),
                dtype=x.dtype,
            ),
        )


@keras_export(
    [
        "keras.ops.moments",
        "keras.ops.nn.moments",
    ]
)
def moments(x, axes, keepdims=False, synchronized=False):
    """Calculates the mean and variance of `x`.

    The mean and variance are calculated by aggregating the contents of `x`
    across `axes`. If `x` is 1-D and `axes = [0]` this is just the mean and
    variance of a vector.

    Args:
        x: Input tensor.
        axes: A list of axes which to compute mean and variance.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.
        synchronized: Only applicable with the TensorFlow backend.
            If `True`, synchronizes the global batch statistics (mean and
            variance) across all devices at each training step in a
            distributed training strategy. If `False`, each replica uses its own
            local batch statistics.

    Returns:
        A tuple containing two tensors - mean and variance.

    Example:

    >>> x = keras.ops.convert_to_tensor([0, 1, 2, 3, 100], dtype="float32")
    >>> keras.ops.moments(x, axes=[0])
    (array(21.2, dtype=float32), array(1553.3601, dtype=float32))

    """
    if any_symbolic_tensors((x,)):
        return Moments(axes, keepdims, synchronized=synchronized).symbolic_call(
            x
        )

    return backend.ops.nn.moments(x, axes, keepdims, synchronized=synchronized)


class BatchNorm(Operation):
    def __init__(self, axis, epsilon=1e-3, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x, mean, variance, offset=None, scale=None):
        return backend.ops.nn.batch_normalization(
            x,
            mean,
            variance,
            axis=self.axis,
            offset=offset,
            scale=scale,
            epsilon=self.epsilon,
        )

    def _check_shape(self, name, shape, expected_shape):
        if shape != expected_shape:
            raise ValueError(
                f"Arguments `{name}` must be a vector of length "
                f"`x.shape[axis]`. Expected: `{expected_shape}`. "
                f"Received: `{shape}."
            )

    def compute_output_spec(self, x, mean, variance, offset, scale):
        shape = (x.shape[self.axis],)
        self._check_shape("mean", tuple(mean.shape), shape)
        self._check_shape("variance", tuple(variance.shape), shape)
        if offset is not None:
            self._check_shape("offset", tuple(offset.shape), shape)
        if scale is not None:
            self._check_shape("scale", tuple(scale.shape), shape)
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.batch_normalization",
        "keras.ops.nn.batch_normalization",
    ]
)
def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    """Normalizes `x` by `mean` and `variance`.

    This op is typically used by the batch normalization step in a neural
    network. It normalizes the input tensor along the given axis.

    Args:
        x: Input tensor.
        mean: A mean vector of the same length as the `axis` dimension of the
            input thensor.
        variance: A variance vector of the same length as the `axis` dimension
            of the input tensor.
        axis: Integer, the axis that should be normalized.
        offset: An offset vector of the same length as the `axis` dimension of
            the input tensor. If not `None`, `offset` is added to the normalized
            tensor. Defaults to `None`.
        scale: A scale vector of the same length as the `axis` dimension of the
            input tensor. If not `None`, the normalized tensor is multiplied by
            `scale`. Defaults to `None`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.

    Returns:
        The normalized tensor.

    Example:

    >>> x = keras.ops.convert_to_tensor(
    ...     [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ... )
    >>> keras.ops.batch_normalization(
    ...     x,
    ...     mean=[0.4, 0.5, 0.6],
    ...     variance=[0.67, 0.67, 0.67],
    ...     axis=-1
    ... )
    array([[-3.6624e-01, -3.6624e-01, -3.6624e-01],
           [-4.6445e-09,  0.0000e+00, -1.8578e-08],
           [ 3.6624e-01,  3.6624e-01,  3.6624e-01]])

    """
    if any_symbolic_tensors((x, mean, variance, offset, scale)):
        return BatchNorm(axis, epsilon).symbolic_call(
            x, mean, variance, offset, scale
        )

    return backend.ops.nn.batch_normalization(
        x, mean, variance, axis, offset, scale, epsilon
    )


class CTCLoss(Operation):
    def __init__(self, mask_index=0, *, name=None):
        super().__init__(name=name)
        self.mask_index = mask_index

    def call(self, target, output, target_length, output_length):
        return backend.ops.nn.ctc_loss(
            target, output, target_length, output_length, self.mask_index
        )

    def _check_shape_first_dim(self, name1, shape1, name2, shape2):
        if shape1[0] != shape2[0]:
            raise ValueError(
                f"Arguments `{name1}` and `{name2}` must have the same "
                "first dimension. "
                f"Received shapes: `{shape1}` and `{shape2}`."
            )

    def compute_output_spec(self, target, output, target_length, output_length):
        self._check_shape_first_dim(
            "target", target.shape, "output", output.shape
        )
        self._check_shape_first_dim(
            "target_length", target_length.shape, "target", target.shape
        )
        self._check_shape_first_dim(
            "output_length", output_length.shape, "output", output.shape
        )
        dtype = backend.result_type(output.dtype, "float32")
        return KerasTensor((target.shape[0],), dtype=dtype)


@keras_export(
    [
        "keras.ops.ctc_loss",
        "keras.ops.nn.ctc_loss",
    ]
)
def ctc_loss(target, output, target_length, output_length, mask_index=0):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        target: A tensor of shape `(batch_size, max_length)` containing
            the true labels in integer format.
        output: A tensor of shape `(batch_size, max_length, num_classes)`
            containing logits (the output of your model).
        target_length: A tensor of shape `(batch_size,)` containing the
            true label lengths.
        output_length: A tensor of shape `(batch_size,)` containing the
            output lengths.
        mask_index: The index of the mask character in the vocabulary.
            Defaults to `0`.
    """

    if any_symbolic_tensors((target, output, target_length, output_length)):
        return CTCLoss(mask_index).symbolic_call(
            target, output, target_length, output_length
        )
    return backend.ops.nn.ctc_loss(
        target, output, target_length, output_length, mask_index
    )


class CTCDecode(Operation):
    def __init__(
        self,
        strategy="greedy",
        beam_width=100,
        top_paths=1,
        merge_repeated=True,
        mask_index=0,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.strategy = strategy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.merge_repeated = merge_repeated
        self.mask_index = mask_index

    def call(self, inputs, sequence_lengths):
        return backend.ops.nn.ctc_decode(
            inputs,
            sequence_lengths,
            strategy=self.strategy,
            beam_width=self.beam_width,
            top_paths=self.top_paths,
            merge_repeated=self.merge_repeated,
            mask_index=self.mask_index,
        )

    def compute_output_spec(self, inputs, sequence_lengths):
        inputs_shape = inputs.shape
        if self.strategy == "greedy":
            top_paths = 1
        else:
            top_paths = self.top_paths
        dtype = backend.result_type(inputs.dtype, "float32")
        return (
            KerasTensor(
                (top_paths, inputs_shape[0], inputs_shape[1]), dtype="int32"
            ),
            KerasTensor((inputs_shape[0], top_paths), dtype=dtype),
        )


@keras_export(
    [
        "keras.ops.ctc_decode",
        "keras.ops.nn.ctc_decode",
    ]
)
def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    """Decodes the output of a CTC model.

    Args:
        inputs: A tensor of shape `(batch_size, max_length, num_classes)`
            containing the logits (the output of the model).
            They should *not* be normalized via softmax.
        sequence_lengths: A tensor of shape `(batch_size,)` containing the
            sequence lengths for the batch.
        strategy: A string for the decoding strategy. Supported values are
            `"greedy"` and `"beam_search"`.
        beam_width: An integer scalar beam width used in beam search.
            Defaults to 100.
        top_paths: An integer scalar, the number of top paths to return.
            Defaults to 1.
        merge_repeated: A boolean scalar, whether to merge repeated
            labels in the output. Defaults to `True`.
        mask_index: An integer scalar, the index of the mask character in
            the vocabulary. Defaults to `0`.

    Returns:
        A tuple containing:
        - The tensor representing the list of decoded sequences. If
            `strategy="greedy"`, the shape is `(1, batch_size, max_length)`. If
            `strategy="beam_search"`, the shape is
            `(top_paths, batch_size, max_length)`. Note that: `-1` indicates the
            blank label.
        - If `strategy="greedy"`, a tensor of shape `(batch_size, 1)`
            representing the negative of the sum of the probability logits for
            each sequence. If `strategy="beam_seatch"`, a tensor of shape
            `(batch_size, top_paths)` representing the log probability for each
            sequence.
    """

    if any_symbolic_tensors((inputs, sequence_lengths)):
        return CTCDecode(
            strategy=strategy,
            beam_width=beam_width,
            top_paths=top_paths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        ).symbolic_call(inputs, sequence_lengths)
    return backend.ops.nn.ctc_decode(
        inputs=inputs,
        sequence_lengths=sequence_lengths,
        strategy=strategy,
        beam_width=beam_width,
        top_paths=top_paths,
        merge_repeated=merge_repeated,
        mask_index=mask_index,
    )


class Normalize(Operation):
    def __init__(self, axis=-1, order=2, epsilon=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.order = order
        self.epsilon = epsilon

    def compute_output_spec(self, x):
        if self.axis is not None:
            canonicalize_axes(self.axis, len(x.shape))
        return KerasTensor(shape=x.shape)

    def call(self, x):
        return _normalize(
            x, axis=self.axis, order=self.order, epsilon=self.epsilon
        )


@keras_export(
    [
        "keras.ops.normalize",
        "keras.ops.nn.normalize",
    ]
)
def normalize(x, axis=-1, order=2, epsilon=None):
    """Normalizes `x` over the specified axis.

    It is defined as: `normalize(x) = x / max(norm(x), epsilon)`.

    Args:
        x: Input tensor.
        axis: The axis or axes along which to perform normalization.
            Default to -1.
        order: The exponent value in the norm formulation.
            Defaults to 2.
        epsilon: A lower bound value for the norm.
            Defaults to `backend.epsilon()`.

    Returns:
        The normalized array.

    Example:

    >>> x = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> x_norm = keras.ops.math.normalize(x)
    >>> print(x_norm)
    array([[0.26726124 0.5345225  0.8017837 ]
           [0.45584232 0.5698029  0.68376344]], shape=(2, 3), dtype=float32)

    """
    if any_symbolic_tensors((x,)):
        return Normalize(axis=axis, order=order, epsilon=epsilon).symbolic_call(
            x
        )
    return _normalize(x, axis=axis, order=order, epsilon=epsilon)


def _normalize(x, axis=-1, order=2, epsilon=None):
    if not isinstance(order, int) or not order >= 1:
        raise ValueError(
            f"Argument `order` must be an int >= 1. Received: order={order}"
        )
    x = backend.ops.convert_to_tensor(x)
    if len(x.shape) == 0:
        x = backend.ops.numpy.expand_dims(x, axis=0)
    if epsilon is None:
        epsilon = backend.epsilon()
    if 2 == order:
        # A special case: L2 normalization with `x * rsqrt(...)`
        # instead of `x / sqrt(...)`. Clamp the squared norm before the
        # rsqrt so zero vectors get a finite gradient.
        square_sum = backend.ops.numpy.sum(
            backend.ops.numpy.square(x), axis=axis, keepdims=True
        )
        inv_norm = backend.ops.math.rsqrt(
            backend.ops.numpy.maximum(square_sum, epsilon * epsilon)
        )
        return x * inv_norm
    norm = backend.ops.linalg.norm(x, ord=order, axis=axis, keepdims=True)
    denom = backend.ops.numpy.maximum(norm, epsilon)
    return backend.ops.numpy.divide(x, denom)


class PSNR(Operation):
    def __init__(
        self,
        max_val,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.max_val = max_val

    def call(self, x1, x2):
        return backend.ops.nn.psnr(
            x1=x1,
            x2=x2,
            max_val=self.max_val,
        )

    def compute_output_spec(self, x1, x2):
        if len(x1.shape) != len(x2.shape):
            raise ValueError("Inputs must have the same rank")

        return KerasTensor(shape=())


@keras_export(
    [
        "keras.ops.psnr",
        "keras.ops.nn.psnr",
    ]
)
def psnr(
    x1,
    x2,
    max_val,
):
    """Peak Signal-to-Noise Ratio (PSNR) function.

    This function computes the Peak Signal-to-Noise Ratio between two signals,
    `x1` and `x2`. PSNR is a measure of the quality of a reconstructed signal.
    The higher the PSNR, the closer the reconstructed signal is to the original
    signal. Note that it can become negative when the signal power is
    smaller that the noise power.

    Args:
        x1: The first input signal.
        x2: The second input signal. Must have the same shape as `x1`.
        max_val: The maximum possible value in the signals.

    Returns:
        float: The PSNR value between `x1` and `x2`.

    Examples:

    >>> x1 = keras.random.normal((2, 4, 4, 3))
    >>> x2 = keras.random.normal((2, 4, 4, 3))
    >>> max_val = 1.0
    >>> keras.ops.nn.psnr(x1, x2, max_val)
    -3.1697404
    """
    if any_symbolic_tensors(
        (
            x1,
            x2,
        )
    ):
        return PSNR(
            max_val,
        ).symbolic_call(x1, x2)
    return backend.ops.nn.psnr(
        x1,
        x2,
        max_val,
    )


class DotProductAttention(Operation):
    def __init__(
        self,
        is_causal=False,
        flash_attention=None,
        attn_logits_soft_cap=None,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.is_causal = is_causal
        self.flash_attention = flash_attention
        self.attn_logits_soft_cap = attn_logits_soft_cap

    def call(
        self,
        query,
        key,
        value,
        bias=None,
        mask=None,
        scale=None,
    ):
        return backend.ops.nn.dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
            is_causal=self.is_causal,
            flash_attention=self.flash_attention,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
        )

    def compute_output_spec(
        self,
        query,
        key,
        value,
        bias=None,
        mask=None,
        scale=None,
    ):
        dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
        return KerasTensor(query.shape, dtype=dtype)


@keras_export(
    ["keras.ops.dot_product_attention", "keras.ops.nn.dot_product_attention"]
)
def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    """Scaled dot product attention function.

    Computes the attention function on Q (`query`), K (`key`), and V(`value`):
    `attention(Q, K, V) = softmax(Q * K / sqrt(d)) * V`. If we define `logits`
    as the output of `Q * K` and the `probs` as the output of `softmax`.

    Throughout this function, we utilize the following notation to represent the
    shape of array:
    - B: batch size
    - S: length of the key/value
    - T: length of the query
    - N: number of attention heads
    - H: dimensions of each attention head
    - K: number of key/value heads
    - G: number of groups, which equals to `N // K`

    Args:
        query: The query array with the shape of `(B, T, N, H)`.
        key: The key array with the shape of `(B, S, K, H)`. When `K` equals
            `N`, multi-headed attention (MHA) is performed. Otherwise, grouped
            query attention (GQA) is performed if `N` is a multiple of `K`. and
            multi-query attention (MQA) is performed if `K==1` (a special case
            of GQA).
        value: The value array with the same shape of `key`.
        bias: Optional bias array to be added to logits. The shape must be
            broadcastable to `(B, N, T, S)`.
        mask: Optional mask array used to filter out logits. It is a boolean
            mask where `True` indicates the element should take part in
            attention. For an additive mask, users should pass it to bias. The
            shape must be broadcastable to `(B, N, T, S)`.
        scale: Optional scale for the logits. If `None`, the scale will be set
            to `1.0 / sqrt(H)`.
        is_causal: Whether to apply causal mask.
        flash_attention: Whether to use flash attention. If `None`, it will
            attempt to use flash attention if the required conditions are met.
            Typically, the inputs must be in float16 and bfloat16 dtype and the
            input layout requirements may vary depending on the backend.
        attn_logits_soft_cap: The value limit for maximum value of the
            attention logits before the softmax function is applied. This is
            only supported in JAX TPU backend. Defaults to None.

    Returns:
        An array of the attention output with the same shape of `query`.

    Example:

    >>> query = keras.random.normal((2, 4, 8, 16))
    >>> key = keras.random.normal((2, 6, 8, 16))
    >>> value = keras.random.normal((2, 6, 8, 16))
    >>> keras.ops.nn.dot_product_attention(query, key, value).shape
    (2, 4, 8, 16)
    """
    if attn_logits_soft_cap is not None:
        if backend.backend() == "jax":
            import jax

            if jax.devices()[0].platform != "tpu":
                raise ValueError(
                    "attn_logits_soft_cap is only supported for JAX on TPU. "
                    "Set attn_logits_soft_cap=None when not using JAX on TPU."
                )
        else:
            raise ValueError(
                "attn_logits_soft_cap is only supported for JAX on TPU. "
                "Set attn_logits_soft_cap=None when not using JAX on TPU."
            )

    if any_symbolic_tensors((query, key, value)):
        return DotProductAttention(
            is_causal=is_causal,
            flash_attention=flash_attention,
            attn_logits_soft_cap=attn_logits_soft_cap,
        ).symbolic_call(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
        )
    return backend.ops.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        flash_attention=flash_attention,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )


class RMSNorm(Operation):
    def __init__(self, axis=-1, epsilon=None, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.epsilon = epsilon

    def compute_output_spec(self, x, scale):
        return KerasTensor(shape=x.shape, dtype=x.dtype)

    def call(self, x, scale=None):
        return _rms_normalization(
            x, scale=scale, axis=self.axis, epsilon=self.epsilon
        )


@keras_export(
    [
        "keras.ops.rms_normalization",
        "keras.ops.nn.rms_normalization",
    ]
)
def rms_normalization(x, scale=None, axis=-1, epsilon=None):
    """Performs Root Mean Square (RMS) normalization on `x`.

    The Keras operation implements the operation as described in
    [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
    by Biao Zhang et al.

    The operation is different from LayerNormalization with RMS scaling.

    It is defined as `rms_normalization(x) = x * rsqrt(mean(square(x))) * scale`

    Args:
        x: Input tensor.
        scale: Optional scaling factor for the normalization.
        axis: The axis or axes along which to perform normalization. Defaults
            to `-1`.
        epsilon: A lower bound value for the norm. Defaults to
            `backend.epsilon()`.

    Returns:
        The normalized array.

    Example:

    >>> x = keras.random.normal((1, 10))
    >>> keras.ops.rms_normalization(x)
    array([[0.69384296, 0.94444374, 0.16551171, 0.05749961, 1.11008865,
            0.52475186, 1.57686807, 1.69893307, 1.27292764, 0.30819128]])
    """
    if any_symbolic_tensors((x, scale)):
        return RMSNorm(axis=axis, epsilon=epsilon).symbolic_call(x, scale=scale)
    return _rms_normalization(x, scale=scale, axis=axis, epsilon=epsilon)


def _rms_normalization(x, scale=None, axis=-1, epsilon=None):
    if epsilon is None:
        epsilon = backend.epsilon()
    original_dtype = standardize_dtype(x.dtype)
    # Computes in at least float32 precision for stability in half precision
    # training.
    compute_dtype = backend.result_type(x.dtype, "float32")

    x = backend.ops.convert_to_tensor(x, dtype=compute_dtype)
    if scale is not None:
        scale = backend.ops.convert_to_tensor(scale, x.dtype)

    if isinstance(axis, (tuple, list)):
        axis = sorted(axis)
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.nn, "rms_normalization"
    ):
        outputs = backend.ops.nn.rms_normalization(
            x, scale=scale, axis=axis, epsilon=epsilon
        )
        return backend.ops.cast(outputs, original_dtype)
    if len(x.shape) == 0:
        x = backend.ops.numpy.expand_dims(x, axis=0)
    rrms = backend.ops.math.rsqrt(
        backend.ops.numpy.mean(
            backend.ops.numpy.square(x), axis=axis, keepdims=True
        )
        + epsilon
    )
    outputs = backend.ops.numpy.multiply(x, rrms)
    if scale is not None:
        outputs = backend.ops.numpy.multiply(outputs, scale)
    return backend.ops.cast(outputs, original_dtype)


class LayerNorm(Operation):
    def __init__(self, axis=-1, epsilon=None, rms_scaling=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.epsilon = epsilon
        self.rms_scaling = rms_scaling

    def compute_output_spec(self, x, gamma, beta):
        return KerasTensor(shape=x.shape, dtype=x.dtype)

    def call(self, x, gamma=None, beta=None):
        return _layer_normalization(
            x,
            gamma=gamma,
            beta=beta,
            axis=self.axis,
            epsilon=self.epsilon,
            rms_scaling=self.rms_scaling,
        )


@keras_export(
    [
        "keras.ops.layer_normalization",
        "keras.ops.nn.layer_normalization",
    ]
)
def layer_normalization(
    x, gamma=None, beta=None, axis=-1, epsilon=None, **kwargs
):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    Args:
        x: Input tensor.
        gamma: Optional scaling factor for the normalization.
        beta: Optional add offset for the normalized tensor.
        axis: The axis or axes along which to perform normalization. Default to
            `-1`.
        epsilon: A lower bound value for the norm.
            Defaults to `backend.epsilon()`.

    Returns:
        The normalized array.

    Example:

    >>> x = keras.ops.arange(5, dtype="float32")
    >>> keras.ops.layer_normalization(x)
    array([-1.4142135, -0.70710677, 0.0, 0.7071067, 1.4142135])
    """
    rms_scaling = kwargs.pop("rms_scaling", False)
    if rms_scaling:
        warnings.warn(
            "You passed `rms_scaling=True`, which is deprecated. This argument "
            "incorrectly scales the input by the variance, not the root mean "
            "square. To correctly use RMS Normalization, please use "
            "`keras.ops.rms_normalization` / `keras.ops.nn.rms_normalization` "
            "instead."
        )

    if any_symbolic_tensors((x, gamma, beta)):
        return LayerNorm(
            axis=axis, epsilon=epsilon, rms_scaling=rms_scaling
        ).symbolic_call(x, gamma, beta)
    return _layer_normalization(
        x,
        gamma=gamma,
        beta=beta,
        axis=axis,
        epsilon=epsilon,
        rms_scaling=rms_scaling,
    )


def _layer_normalization(
    x, gamma=None, beta=None, axis=-1, epsilon=None, rms_scaling=False
):
    if epsilon is None:
        epsilon = backend.epsilon()
    original_dtype = standardize_dtype(x.dtype)
    # Computes in at least float32 precision for stability in half precision
    # training.
    compute_dtype = backend.result_type(x.dtype, "float32")

    x = backend.ops.convert_to_tensor(x, dtype=compute_dtype)
    if gamma is not None:
        gamma = backend.ops.convert_to_tensor(gamma, x.dtype)
    if beta is not None:
        beta = backend.ops.convert_to_tensor(beta, x.dtype)

    # Compute the axes along which to reduce the mean / variance
    input_shape = x.shape
    ndims = len(input_shape)

    # Broadcasting only necessary for norm when the axis is not just
    # the last dimension
    broadcast_shape = [1] * ndims
    if isinstance(axis, int):
        axis = [axis]
    axis = sorted(axis)
    for dim in axis:
        broadcast_shape[dim] = input_shape[dim]

    def _broadcast(v):
        if v is not None and len(v.shape) != ndims and axis != [ndims - 1]:
            return backend.ops.numpy.reshape(v, broadcast_shape)
        return v

    if rms_scaling:
        variance = backend.ops.numpy.var(x, axis=axis, keepdims=True)
        inv = backend.ops.math.rsqrt(variance + epsilon)
        outputs = x * inv
        if gamma is not None:
            outputs = outputs * backend.ops.cast(_broadcast(gamma), x.dtype)
        return backend.ops.cast(outputs, original_dtype)

    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.nn, "layer_normalization"
    ):
        outputs = backend.ops.nn.layer_normalization(
            x, gamma=gamma, beta=beta, axis=axis, epsilon=epsilon
        )
        return backend.ops.cast(outputs, original_dtype)

    # Calculate the mean & variance along self.axis (layer activations).
    mean, variance = moments(x, axes=axis, keepdims=True)
    gamma, beta = _broadcast(gamma), _broadcast(beta)
    inv = backend.ops.math.rsqrt(variance + epsilon)
    if gamma is not None:
        inv = inv * gamma

    # Center before scaling. Folding the mean and beta into a single offset,
    # as `x * inv + (beta - mean * inv)`, loses beta whenever `mean * inv` is
    # large enough that adding beta to it rounds away: for a constant row the
    # two large terms then cancel exactly and the result is 0 instead of beta.
    outputs = (x - mean) * inv
    if beta is not None:
        outputs = outputs + beta
    return backend.ops.cast(outputs, original_dtype)


class Polar(Operation):
    def compute_output_spec(self, abs_, angle):
        dtype = standardize_dtype(
            dtypes.result_type(
                getattr(abs_, "dtype", backend.floatx()),
                getattr(angle, "dtype", backend.floatx()),
            )
        )
        # `polar` combines two real tensors into a complex one, so the output
        # is the complex dtype of matching width.
        dtype = "complex128" if dtype == "float64" else "complex64"
        output_shape = operation_utils.broadcast_shapes(abs_.shape, angle.shape)
        return KerasTensor(shape=output_shape, dtype=dtype)

    def call(self, abs_, angle):
        return _polar(abs_, angle)


@keras_export(["keras.ops.polar", "keras.ops.nn.polar"])
def polar(abs_, angle):
    """Constructs a complex tensor whose elements are Cartesian
    coordinates corresponding to the polar coordinates
    with absolute value `abs` and angle `angle`.

    The operation is numerically equivalent to `torch.polar()`.
    It is not equivalent to `scipy.lingalg.polar()` which performs
    Singular Value Decomposition.

    Given the magnitude (`abs_`) and angle (`angle`), this function computes the
    corresponding complex number in the form of `real + imaginary * 1j`, where:
    - `real = abs_ * cos(angle)`
    - `imaginary = abs_ * sin(angle)`

    Args:
        abs_: The magnitude (absolute value) of the complex number.
        angle: The angle (in radians) of the complex number.

    Returns:
        A complex number (or array of complex numbers) with the same shape as
        `abs_` and `angle`.

    Example:

    >>> abs_ = keras.random.normal((1, 2))
    >>> angle = keras.random.normal((1, 2))
    >>> keras.ops.nn.polar(abs_, angle).shape
    (1, 2)
    >>> keras.ops.nn.polar(abs_, angle)
    Array([[0.63185346-0.59370506j, 0.48960376-0.31677645j]], dtype=complex64)
    """
    if any_symbolic_tensors((abs_, angle)):
        return Polar().symbolic_call(abs_, angle)
    return _polar(abs_, angle)


def _polar(abs_, angle):
    """Internal implementation of the polar function.

    Args:
        abs_: The magnitude (absolute value) of the complex number.
        angle: The angle (in radians) of the complex number.

    Returns:
        A complex number (or array of complex numbers) with the same shape as
        `abs_` and `angle`.
    """
    abs_ = backend.ops.convert_to_tensor(abs_)
    angle = backend.ops.convert_to_tensor(angle)

    real = abs_ * backend.ops.numpy.cos(angle)
    imaginary = abs_ * backend.ops.numpy.sin(angle)

    result = backend.ops.math._get_complex_tensor_from_tuple((real, imaginary))

    return result


class Unfold(Operation):
    def __init__(
        self, kernel_size, dilation=1, padding=0, stride=1, *, name=None
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def compute_output_spec(self, x):
        N, C, H, W = x.shape

        def _pair(x):
            return (x, x) if isinstance(x, int) else x

        kH, kW = _pair(self.kernel_size)
        dH, dW = _pair(self.dilation)
        pH, pW = _pair(self.padding)
        sH, sW = _pair(self.stride)

        def out_size(L, k, d, p, s):
            return (L + 2 * p - d * (k - 1) - 1) // s + 1

        outH = out_size(H, kH, dH, pH, sH)
        outW = out_size(W, kW, dW, pW, sW)
        return KerasTensor(shape=(N, C * kH * kW, outH * outW), dtype=x.dtype)

    def call(self, x):
        return _unfold(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        )


@keras_export(["keras.ops.unfold", "keras.ops.nn.unfold"])
def unfold(x, kernel_size, dilation=1, padding=0, stride=1):
    """Extract sliding local blocks from a 4-D input (batched image).

    This operation is known as **im2col** when used with convolution.
    It rearranges the image into overlapping or non-overlapping patches
    and returns a tensor whose *depth* (last axis) contains the flattened
    patches.

    Args:
        x: A 4-D tensor of shape `(N, C, H, W)` (**channels-first** format).
        kernel_size: int or tuple of two ints, the size of the sliding window
            `(kH, kW)`.  If a single int is given, it is used for both
            dimensions.
        dilation: int or tuple of two ints, the spacing between kernel points
            (a.k.a. **dilation** or **atrous** convolution). Default: 1.
        padding: int or tuple of two ints, the amount of zero-padding to apply
            to both spatial dimensions. Default: 0.
        stride: int or tuple of two ints, the step size of the sliding window.
            Default: 1.

    Returns:
        A 3-D tensor of shape `(N, C * kH * kW, L)` where
        `L = num_patches_H * num_patches_W` is the total number of patches
        extracted.

    Example:

    >>> x = keras.ops.ones((1, 2, 4, 4))
    >>> patches = keras.ops.unfold(x, kernel_size=2, stride=2)
    >>> patches.shape
    (1, 8, 4)

    """
    input_shape = x.shape
    ndims = len(input_shape)
    if ndims != 4:
        raise ValueError(
            f"Input must be a 4D tensor. Received: input.shape={input_shape}"
        )
    if any_symbolic_tensors((x,)):
        return Unfold(kernel_size, dilation, padding, stride).symbolic_call(x)
    return _unfold(x, kernel_size, dilation, padding, stride)


def _unfold(x, kernel_size, dilation=1, padding=0, stride=1):
    """Internal implementation of unfold."""
    return backend.ops.nn.unfold(
        x,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


class Fold(Operation):
    def __init__(
        self,
        output_size,
        kernel_size,
        dilation=1,
        padding=0,
        stride=1,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def compute_output_spec(self, x):
        N, CKK, L = x.shape

        def _pair(v):
            return (v, v) if isinstance(v, int) else v

        kH, kW = _pair(self.kernel_size)
        oH, oW = _pair(self.output_size)

        if CKK is not None and CKK % (kH * kW) != 0:
            raise ValueError(
                f"The second dimension of the input ({CKK}) must be "
                f"divisible by kernel_size product ({kH * kW})."
            )

        C = CKK // (kH * kW) if CKK is not None else None
        return KerasTensor(shape=(N, C, oH, oW), dtype=x.dtype)

    def call(self, x):
        return backend.ops.nn.fold(
            x,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )


@keras_export(["keras.ops.fold", "keras.ops.nn.fold"])
def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Combines an array of sliding local blocks into a large containing
    tensor (reverses `unfold`).

    This operation is known as **col2im** when used with convolution.
    It takes a 3-D tensor of flattened patches and reconstructs a 4-D
    image tensor by summing overlapping patches.

    Args:
        x: A 3-D tensor of shape `(N, C * kH * kW, L)` where `L` is
            the total number of blocks.
        output_size: int or tuple of two ints `(oH, oW)`, the spatial
            shape of the output tensor.
        kernel_size: int or tuple of two ints, the size of the sliding
            window `(kH, kW)`.  If a single int is given, it is used
            for both dimensions.
        dilation: int or tuple of two ints, the spacing between kernel
            points. Default: 1.
        padding: int or tuple of two ints, the amount of zero-padding
            that was applied to the input of `unfold`. Default: 0.
        stride: int or tuple of two ints, the step size of the sliding
            window. Default: 1.

    Returns:
        A 4-D tensor of shape `(N, C, oH, oW)`.

    Example:

    >>> x = keras.ops.ones((1, 2, 4, 4))
    >>> patches = keras.ops.unfold(x, kernel_size=2, stride=2)
    >>> patches.shape
    (1, 8, 4)
    >>> y = keras.ops.fold(patches, output_size=(4, 4),
    ...     kernel_size=2, stride=2)
    >>> y.shape
    (1, 2, 4, 4)

    """
    input_shape = x.shape
    ndims = len(input_shape)
    if ndims != 3:
        raise ValueError(
            f"Input must be a 3D tensor. Received: input.shape={input_shape}"
        )
    if any_symbolic_tensors((x,)):
        return Fold(
            output_size, kernel_size, dilation, padding, stride
        ).symbolic_call(x)
    return backend.ops.nn.fold(
        x,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


class DepthToSpace(Operation):
    def __init__(self, block_size, data_format="channels_last", *, name=None):
        super().__init__(name=name)
        self.block_size = block_size
        self.data_format = standardize_data_format(data_format)

    def compute_output_spec(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                "`depth_to_space` requires a 4D input tensor. "
                f"Received: x.shape={x.shape}"
            )
        if self.data_format == "channels_last":
            b, h, w, c = x.shape
        else:
            b, c, h, w = x.shape

        if c is not None and c % (self.block_size**2) != 0:
            raise ValueError(
                f"The number of channels ({c}) must be divisible by "
                f"block_size**2 ({self.block_size**2})."
            )

        new_c = c // (self.block_size**2) if c is not None else None
        new_h = h * self.block_size if h is not None else None
        new_w = w * self.block_size if w is not None else None

        if self.data_format == "channels_last":
            output_shape = (b, new_h, new_w, new_c)
        else:
            output_shape = (b, new_c, new_h, new_w)

        return KerasTensor(output_shape, dtype=x.dtype)

    def call(self, x):
        return backend.ops.nn.depth_to_space(
            x, self.block_size, data_format=self.data_format
        )


@keras_export(["keras.ops.depth_to_space", "keras.ops.nn.depth_to_space"])
def depth_to_space(x, block_size, data_format="channels_last"):
    """Rearranges data from depth into blocks of spatial data.

    This operation is useful for resizing the activations between convolutions
    (but keeping all data), e.g., instead of pooling. It is also useful for
    training purely convolutional models.

    Also known as pixel shuffle, this operation rearranges elements in a tensor
    of shape `(N, H, W, C * r^2)` to `(N, H * r, W * r, C)` where `r` is the
    `block_size` for `data_format="channels_last"`, or from
    `(N, C * r^2, H, W)` to `(N, C, H * r, W * r)` for
    `data_format="channels_first"`.

    This is the reverse transformation of `space_to_depth`.

    Args:
        x: Input tensor. Must be 4D.
        block_size: An integer specifying the size of the spatial block.
            The depth (number of channels) must be divisible by
            `block_size ** 2`.
        data_format: A string specifying the data format of the input tensor.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            Defaults to `"channels_last"`.

    Returns:
        A tensor with the same dtype as `x`, with shape
        `(N, H * block_size, W * block_size, C // block_size ** 2)` for
        `data_format="channels_last"` or
        `(N, C // block_size ** 2, H * block_size, W * block_size)` for
        `data_format="channels_first"`.

    Example:

    >>> x = keras.ops.reshape(keras.ops.arange(1 * 2 * 2 * 12), (1, 2, 2, 12))
    >>> keras.ops.depth_to_space(x, block_size=2).shape
    (1, 4, 4, 3)

    >>> # channels_first example
    >>> x = keras.ops.reshape(keras.ops.arange(1 * 12 * 2 * 2), (1, 12, 2, 2))
    >>> keras.ops.depth_to_space(x, block_size=2,
    ...                          data_format="channels_first").shape
    (1, 3, 4, 4)
    """
    data_format = standardize_data_format(data_format)
    if block_size < 2:
        raise ValueError(
            "`block_size` must be at least 2. "
            f"Received: block_size={block_size}"
        )
    if any_symbolic_tensors((x,)):
        return DepthToSpace(block_size, data_format=data_format).symbolic_call(
            x
        )
    return backend.ops.nn.depth_to_space(x, block_size, data_format=data_format)


class SpaceToDepth(Operation):
    def __init__(self, block_size, data_format="channels_last", *, name=None):
        super().__init__(name=name)
        self.block_size = block_size
        self.data_format = standardize_data_format(data_format)

    def compute_output_spec(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                "`space_to_depth` requires a 4D input tensor. "
                f"Received: x.shape={x.shape}"
            )
        if self.data_format == "channels_last":
            b, h, w, c = x.shape
        else:
            b, c, h, w = x.shape

        if h is not None and h % self.block_size != 0:
            raise ValueError(
                f"Height ({h}) must be divisible by block_size "
                f"({self.block_size})."
            )
        if w is not None and w % self.block_size != 0:
            raise ValueError(
                f"Width ({w}) must be divisible by block_size "
                f"({self.block_size})."
            )

        new_c = c * (self.block_size**2) if c is not None else None
        new_h = h // self.block_size if h is not None else None
        new_w = w // self.block_size if w is not None else None

        if self.data_format == "channels_last":
            output_shape = (b, new_h, new_w, new_c)
        else:
            output_shape = (b, new_c, new_h, new_w)

        return KerasTensor(output_shape, dtype=x.dtype)

    def call(self, x):
        return backend.ops.nn.space_to_depth(
            x, self.block_size, data_format=self.data_format
        )


@keras_export(["keras.ops.space_to_depth", "keras.ops.nn.space_to_depth"])
def space_to_depth(x, block_size, data_format="channels_last"):
    """Rearranges blocks of spatial data into depth.

    This operation is useful for resizing the activations between convolutions
    (but keeping all data). It is also useful for training purely convolutional
    models.

    This operation rearranges elements in a tensor of shape
    `(N, H * block_size, W * block_size, C)` to a tensor of shape
    `(N, H, W, C * block_size ** 2)` (for `data_format="channels_last"`)
    or `(N, C, H * block_size, W * block_size)` to
    `(N, C * block_size ** 2, H, W)` (for `data_format="channels_first"`).

    This is the reverse transformation of `depth_to_space`.

    Args:
        x: Input tensor. Must be 4D.
        block_size: An integer specifying the size of the spatial block.
            The height and width of the input must be divisible by
            `block_size`.
        data_format: A string specifying the data format of the input tensor.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            Defaults to `"channels_last"`.

    Returns:
        A tensor with the same dtype as `x`, with shape
        `(N, H // block_size, W // block_size, C * block_size ** 2)` for
        `data_format="channels_last"` or
        `(N, C * block_size ** 2, H // block_size, W // block_size)` for
        `data_format="channels_first"`.

    Example:

    >>> x = keras.ops.reshape(keras.ops.arange(1 * 4 * 4 * 3), (1, 4, 4, 3))
    >>> keras.ops.space_to_depth(x, block_size=2).shape
    (1, 2, 2, 12)

    >>> # channels_first example
    >>> x = keras.ops.reshape(keras.ops.arange(1 * 3 * 4 * 4), (1, 3, 4, 4))
    >>> keras.ops.space_to_depth(x, block_size=2,
    ...                          data_format="channels_first").shape
    (1, 12, 2, 2)
    """
    data_format = standardize_data_format(data_format)
    if block_size < 2:
        raise ValueError(
            "`block_size` must be at least 2. "
            f"Received: block_size={block_size}"
        )
    if any_symbolic_tensors((x,)):
        return SpaceToDepth(block_size, data_format=data_format).symbolic_call(
            x
        )
    return backend.ops.nn.space_to_depth(x, block_size, data_format=data_format)
