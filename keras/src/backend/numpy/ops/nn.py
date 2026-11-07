import math

import jax
import numpy as np
from jax import lax

from keras.src import backend
from keras.src.backend.common import dtypes
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import check_conv_input_channels
from keras.src.backend.common.backend_utils import (
    check_conv_transpose_input_channels,
)
from keras.src.backend.common.backend_utils import (
    compute_adaptive_pooling_window_sizes,
)
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.numpy.ops.core import cast
from keras.src.backend.numpy.ops.core import convert_to_tensor
from keras.src.backend.numpy.ops.core import is_tensor
from keras.src.utils.module_utils import scipy


def relu(x):
    x = convert_to_tensor(x)
    return np.maximum(x, np.array(0.0, x.dtype))


def relu6(x):
    x = convert_to_tensor(x)
    # np.clip incorrectly promote bfloat16 to float32, so we replace it with
    # np.minimum and np.maximum here
    return np.minimum(
        np.maximum(x, np.array(0.0, x.dtype)), np.array(6.0, x.dtype)
    )


def sigmoid(x):
    x = convert_to_tensor(x)
    return np.array(1.0, x.dtype) / (np.array(1.0, x.dtype) + np.exp(-x))


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    # `0.5` below truncates to 0 under an integer dtype, which drops the
    # midpoint so every input in (-1, 1) maps to 0.
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return np.where(
        x <= -1,
        np.array(0.0, x.dtype),
        np.where(
            x >= 1, np.array(1.0, x.dtype), np.array(0.5 * (x + 1), x.dtype)
        ),
    )


def tanh(x):
    return np.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return x - np.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return np.logaddexp(x, np.array(0.0, x.dtype))


def softsign(x):
    x = convert_to_tensor(x)
    return x / (np.array(1.0, x.dtype) + np.abs(x))


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return np.where(
        x > threshold,
        np.array(x - threshold, dtype=x.dtype),
        np.where(
            x < -threshold,
            np.array(x + threshold, dtype=x.dtype),
            np.array(0.0, dtype=x.dtype),
        ),
    )


def sparse_plus(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return np.where(
        x <= -1,
        np.zeros_like(x, dtype=x.dtype),
        np.where(x < 1, np.array((1 / 4) * (x + 1) ** 2, dtype=x.dtype), x),
    )


def silu(x):
    x = convert_to_tensor(x)
    return x * sigmoid(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + np.sqrt(x**2 + b)
    return y / 2


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return -softplus(-x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    # `negative_slope` truncates to 0 under an integer dtype, which leaves
    # `maximum(x, 0)`, i.e. `relu`.
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return np.maximum(x, np.array(negative_slope, x.dtype) * x)


def hard_sigmoid(x):
    # `0.5` below truncates to 0 under an integer dtype, which drops the
    # offset and turns this into `clip(x / 6, 0, 1)`.
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    # python numbers will be promoted to float64 by np, so it's necessary to
    # first convert the python numbers to np scalars
    x = x / np.array(6.0, x.dtype) + np.array(0.5, x.dtype)
    return np.where(
        x <= 0.0,
        np.array(0.0, x.dtype),
        np.where(x >= 1.0, np.array(1.0, x.dtype), x),
    )


def hard_silu(x):
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return np.where(
        x >= np.array(0.0, x.dtype), x, np.array(alpha, x.dtype) * np.expm1(x)
    )


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return np.array(scale, x.dtype) * elu(x, alpha)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    # followed by JAX's implementation
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = np.array(0.5, x.dtype) * (
            np.array(1.0, x.dtype)
            + np.tanh(
                sqrt_2_over_pi
                * (x + np.array(0.044715, x.dtype) * (x**3).astype(x.dtype))
            )
        )
        return x * cdf
    else:
        sqrt_2 = np.sqrt(2).astype(x.dtype)
        return (
            x
            * (scipy.special.erf(x / sqrt_2) + 1).astype(x.dtype)
            / np.array(2, x.dtype)
        )


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    alpha = np.array(alpha, x.dtype)
    return np.maximum(x, np.array(0.0, dtype=x.dtype)) + alpha * np.expm1(
        np.minimum(x, np.array(0.0, dtype=x.dtype)) / alpha
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = x.dtype
    canonicalize_axis(axis, len(x.shape))
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = np.split(x, 2, axis)
    return (x1 * sigmoid(x2)).astype(dtype)


def hard_tanh(x):
    x = convert_to_tensor(x)
    min_val = np.asarray(-1.0, x.dtype)
    max_val = np.asarray(1.0, x.dtype)
    return np.array(np.clip(x, min_val, max_val), dtype=x.dtype)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    threshold = np.asarray(threshold, x.dtype)
    return np.array(
        np.where(np.abs(x) > threshold, x, np.array(0.0, dtype=x.dtype)),
        dtype=x.dtype,
    )


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return np.where(x > threshold, x, np.array(default_value, dtype=x.dtype))


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
    return x - max_x - logsumexp


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    # The ranks in `r` and the counts in `k` are integers, and dividing by them
    # promotes the result to `float64`. The Python float constants promote
    # `bfloat16` as well. Both are materialized in the dtype of `logits` so the
    # computation stays in it.
    zero = np.array(0.0, logits.dtype)
    logits_sorted = -np.sort(-logits, axis=axis)
    logits_cumsum = np.cumsum(logits_sorted, axis=axis)
    r = np.arange(1, logits.shape[axis] + 1)
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = r.reshape(r_shape).astype(logits.dtype)
    support = logits_sorted - (logits_cumsum - 1) / r > zero
    # Find the threshold
    k = np.sum(support, axis=axis, keepdims=True).astype(logits.dtype)
    logits_sorted_safe = np.where(support, logits_sorted, zero)
    tau = (np.sum(logits_sorted_safe, axis=axis, keepdims=True) - 1) / k
    output = np.maximum(logits - tau, zero)
    return output


def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _pool(
    inputs,
    initial_value,
    reduce_fn,
    pool_size,
    strides=None,
    padding="valid",
):
    """Helper function to define pooling functions.

    Args:
        inputs: input data of shape `N+2`.
        initial_value: the initial value for the reduction.
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        pool_size: a sequence of `N` integers, representing the window size to
            reduce over.
        strides: a sequence of `N` integers, representing the inter-window
            strides (default: `(1, ..., 1)`).
        padding: either the string `same` or `valid`.

    Returns:
        The output of the reduction for each window slice.
    """
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid padding '{padding}', must be 'same' or 'valid'."
        )
    padding = padding.upper()
    return np.array(
        lax.reduce_window(
            inputs,
            initial_value,
            reduce_fn,
            pool_size,
            strides,
            padding,
        )
    )


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )
    return _pool(inputs, -np.inf, lax.max, pool_size, strides, padding)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )

    pooled = _pool(inputs, 0.0, lax.add, pool_size, strides, padding)
    if padding == "valid":
        # Avoid the extra reduce_window.
        return pooled / math.prod(pool_size)
    else:
        # Count the number of valid entries at each input point, then use that
        # for computing average. Assumes that any two arrays of same shape will
        # be padded the same. Avoid broadcasting on axis where pooling is
        # skipped.
        shape = [
            (a if b != 1 else 1) for (a, b) in zip(inputs.shape, pool_size)
        ]
        window_counts = _pool(
            np.ones(shape, inputs.dtype),
            0.0,
            lax.add,
            pool_size,
            strides,
            padding,
        )
        return pooled / window_counts


def _compute_adaptive_pooling_gather_indices(
    input_dim, output_size, big_window
):
    window_starts = np.floor(
        (np.arange(output_size) * input_dim) / output_size
    ).astype(np.int32)

    window_ends = np.ceil(
        (np.arange(1, output_size + 1) * input_dim) / output_size
    ).astype(np.int32)

    window_sizes = window_ends - window_starts
    is_big = window_sizes == big_window

    small_window = big_window - 1
    small_pool_len = input_dim - small_window + 1

    small_indices = window_starts
    big_indices = window_starts + small_pool_len

    gather = np.where(is_big, big_indices, small_indices)
    return gather.astype(np.int32)


def _strided_view_1d(x, window_size):
    n, l, c = x.shape
    out = l - window_size + 1

    strides = x.strides
    shape = (n, out, window_size, c)
    new_strides = (strides[0], strides[1], strides[1], strides[2])

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=new_strides)


def _adaptive_pool1d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size,)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 1))

    n, l, c = inputs.shape
    out_l = output_size[0]

    small, big = compute_adaptive_pooling_window_sizes(l, out_l)
    gather = _compute_adaptive_pooling_gather_indices(l, out_l, big)

    sv_small = _strided_view_1d(inputs, small)
    small_pool = (
        np.mean(sv_small, axis=2)
        if mode == "average"
        else np.max(sv_small, axis=2)
    )

    sv_big = _strided_view_1d(inputs, big)
    big_pool = (
        np.mean(sv_big, axis=2) if mode == "average" else np.max(sv_big, axis=2)
    )

    combined = np.concatenate([small_pool, big_pool], axis=1)
    out = combined[:, gather, :]

    if data_format == "channels_first":
        out = np.transpose(out, (0, 2, 1))

    return out


def _adaptive_pool2d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 3, 1))

    n, h, w, c = inputs.shape
    out_h, out_w = output_size

    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = np.transpose(inputs, (0, 2, 1, 3)).reshape(n * w, h, c)

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        np.mean(sv_small_h, axis=2)
        if mode == "average"
        else np.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        np.mean(sv_big_h, axis=2)
        if mode == "average"
        else np.max(sv_big_h, axis=2)
    )

    combined_h = np.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :]

    pooled_h = pooled_h.reshape(n, w, out_h, c)
    pooled_h = np.transpose(pooled_h, (0, 2, 1, 3))

    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        np.mean(sv_small_w, axis=2)
        if mode == "average"
        else np.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        np.mean(sv_big_w, axis=2)
        if mode == "average"
        else np.max(sv_big_w, axis=2)
    )

    combined_w = np.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_h, out_w, c)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 3, 1, 2))

    return out


def _adaptive_pool3d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 3, 4, 1))

    n, d, h, w, c = inputs.shape
    out_d, out_h, out_w = output_size

    small_d, big_d = compute_adaptive_pooling_window_sizes(d, out_d)
    gather_d = _compute_adaptive_pooling_gather_indices(d, out_d, big_d)

    x_d = np.transpose(inputs, (0, 2, 3, 1, 4)).reshape(n * h * w, d, c)

    sv_small_d = _strided_view_1d(x_d, small_d)
    small_pool_d = (
        np.mean(sv_small_d, axis=2)
        if mode == "average"
        else np.max(sv_small_d, axis=2)
    )

    sv_big_d = _strided_view_1d(x_d, big_d)
    big_pool_d = (
        np.mean(sv_big_d, axis=2)
        if mode == "average"
        else np.max(sv_big_d, axis=2)
    )

    combined_d = np.concatenate([small_pool_d, big_pool_d], axis=1)
    pooled_d = combined_d[:, gather_d, :].reshape(n, h, w, out_d, c)
    pooled_d = np.transpose(pooled_d, (0, 3, 1, 2, 4))

    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = np.transpose(pooled_d, (0, 1, 3, 2, 4)).reshape(n * out_d * w, h, c)

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        np.mean(sv_small_h, axis=2)
        if mode == "average"
        else np.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        np.mean(sv_big_h, axis=2)
        if mode == "average"
        else np.max(sv_big_h, axis=2)
    )

    combined_h = np.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :].reshape(n, out_d, w, out_h, c)
    pooled_h = np.transpose(pooled_h, (0, 1, 3, 2, 4))

    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_d * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        np.mean(sv_small_w, axis=2)
        if mode == "average"
        else np.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        np.mean(sv_big_w, axis=2)
        if mode == "average"
        else np.max(sv_big_w, axis=2)
    )

    combined_w = np.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_d, out_h, out_w, c)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 4, 1, 2, 3))

    return out


def adaptive_average_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 2:
        return _adaptive_pool2d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 3:
        return _adaptive_pool3d_impl(
            inputs, output_size, "average", data_format
        )
    raise ValueError("adaptive_average_pool supports only 1D/2D/3D")


def adaptive_max_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(inputs, output_size, "max", data_format)
    if dims == 2:
        return _adaptive_pool2d_impl(inputs, output_size, "max", data_format)
    if dims == 3:
        return _adaptive_pool3d_impl(inputs, output_size, "max", data_format)
    raise ValueError("adaptive_max_pool supports only 1D/2D/3D")


def _convert_to_lax_conv_dimension_numbers(
    num_spatial_dims,
    data_format="channels_last",
    transpose=False,
):
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if data_format == "channels_last":
        spatial_dims = tuple(range(1, num_dims - 1))
        inputs_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        inputs_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(
        lhs_spec=inputs_dn, rhs_spec=kernel_dn, out_spec=inputs_dn
    )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[-2]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    feature_group_count = channels // kernel_in_channels
    result = np.array(
        jax.lax.conv_general_dilated(
            inputs,
            kernel if is_tensor(kernel) else kernel.numpy(),
            strides,
            padding,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
        )
    )
    if result.size == 0 and inputs.size != 0:
        raise ValueError(
            "The convolution operation resulted in an empty output. "
            "This can happen if the input is too small for the given "
            "kernel size, strides, dilation rate, and padding mode. "
            "Please check the input shape and convolution parameters."
        )
    return result


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    check_conv_input_channels(inputs, kernel, data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    feature_group_count = (
        inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
    )
    kernel = np.reshape(
        kernel if is_tensor(kernel) else kernel.numpy(),
        kernel.shape[:-2] + (1, feature_group_count * kernel.shape[-1]),
    )
    return np.array(
        jax.lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
        )
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
    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    depthwise_kernel = convert_to_tensor(depthwise_kernel)
    pointwise_kernel = convert_to_tensor(pointwise_kernel)
    check_conv_input_channels(inputs, depthwise_kernel, data_format)
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
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
    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    check_conv_transpose_input_channels(inputs, kernel, data_format)
    num_spatial_dims = inputs.ndim - 2
    padding_values = compute_conv_transpose_padding_args_for_jax(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )

    return np.array(
        jax.lax.conv_transpose(
            inputs,
            kernel if is_tensor(kernel) else kernel.numpy(),
            strides,
            padding=padding_values,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            transpose_kernel=True,
        )
    )


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    if dtype is None:
        dtype = "float32"
    x = convert_to_tensor(x)
    input_shape = x.shape

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1

    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes), dtype=dtype)
    valid_indices = x >= 0
    categorical[np.arange(batch_size)[valid_indices], x[valid_indices]] = 1

    # First, reshape the array with the extra dimension at the end
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    # Then, move this new dimension to the right place (according to axis)
    if axis != -1:
        categorical = np.moveaxis(categorical, -1, axis)

    return categorical


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = np.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target)
    output = np.array(output)
    if not backend.is_float_dtype(output.dtype):
        output = cast(output, backend.floatx())
    target = cast(target, output.dtype)
    dtype = output.dtype

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

    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = np.log(output)
    return (-np.sum(target * log_prob, axis=axis)).astype(dtype)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target, dtype="int32")
    output = np.array(output)
    if not backend.is_float_dtype(output.dtype):
        output = cast(output, backend.floatx())
    dtype = output.dtype
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = np.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = np.log(output)
    target = one_hot(target, output.shape[axis], axis=axis, dtype=dtype)
    return (-np.sum(target * log_prob, axis=axis)).astype(dtype)


def binary_crossentropy(target, output, from_logits=False):
    target = np.array(target)
    output = np.array(output)
    if not is_float_dtype(output.dtype):
        output = cast(output, backend.floatx())
    target = cast(target, output.dtype)
    dtype = output.dtype

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = sigmoid(output)

    output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * np.log(output)
    bce += (1.0 - target) * np.log(1.0 - output)
    return (-bce).astype(dtype)


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with NumPy."
        )
    axes = tuple(axes) if isinstance(axes, list) else axes
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = np.mean(x, axes, keepdims=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    variance = np.mean(np.square(x), axis=axes, keepdims=True) - np.square(mean)

    if not keepdims:
        mean = np.squeeze(mean, axes)
        variance = np.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = np.clip(mean, np.finfo(np.float16).min, np.finfo(np.float16).max)
        variance = np.clip(
            variance, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = np.reshape(mean, shape)
    variance = np.reshape(variance, shape)

    inv = 1.0 / np.sqrt(variance + epsilon)
    if scale is not None:
        scale = np.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = np.reshape(offset, shape)
        res = res + offset

    return x * inv + res


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    # Ref: https://github.com/google-deepmind/optax
    # optax.ctc_loss_with_forward_probs
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, "int32")
    output_length = convert_to_tensor(output_length, "int32")
    batch_size, max_input_length, num_classes = output.shape
    batch_size, max_label_length = target.shape
    log_epsilon = -1e5

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(output.dtype, "float32")
    output = output.astype(dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = np.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = np.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return np.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)
    target_paddings = target_paddings.astype(output.dtype)
    output_paddings = output_paddings.astype(output.dtype)

    logprobs = log_softmax(output, axis=-1)
    label_lengths = max_label_length - np.sum(target_paddings, axis=1).astype(
        np.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target[:, :-1] == target[:, 1:]).astype(np.float32)
    repeat = np.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = np.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = np.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = np.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        np.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    logalpha_phi_init[:, 0] = 0.0
    logalpha_emit_init = (
        np.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return np.concatenate(
            [phi[:, :1], np.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = np.logaddexp(
            prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label is repetition
        next_phi = update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
        )

        pad = pad.reshape((batch_size, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    def np_scan(f, init, xs):
        carry = init
        ys = []
        for x in zip(*xs):
            carry, y = f(carry, x)
            ys.append(y)
        result = []
        for i in range(len(ys[0])):
            result.append(np.stack([y[i] for y in ys]))
        return carry, result

    xs = (logprobs_emit, logprobs_phi, output_paddings.transpose((1, 0)))
    _, (logalpha_phi, logalpha_emit) = np_scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi[-1] = logalpha_phi_last

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -np.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
    return per_seq_loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = np.argmax(inputs, axis=-1).astype("int32")
    scores = np.max(inputs, axis=-1)

    seqlen_mask = np.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = np.where(seqlen_mask, mask_index, indices)
    scores = np.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = np.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = np.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = np.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = np.expand_dims(np.arange(max_length), axis=0)  # [1, N]
    order = np.tile(order, (batch_size, 1))  # [B, N]
    order = np.where(invalid_mask, max_length, order)
    order = np.argsort(order, axis=-1)
    indices = np.take_along_axis(indices, order, axis=-1)

    scores = -np.sum(scores, axis=1)[:, None]
    indices = np.expand_dims(indices, axis=0)
    return indices, scores


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths)

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = log_softmax(inputs, axis=-1)
    seqlen_mask = np.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that np.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = np.flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = np.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=np.int32
    )

    num_init_paths = np.min(np.array([num_classes, beam_width]))
    max_classes = np.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = np.where(max_classes == mask_index, _pad, max_classes)
    init_paths[:, :num_init_paths, 0] = init_classes

    init_scores = np.full(
        (batch_size, 2 * beam_width), -np.inf, dtype=inputs.dtype
    )
    init_scores[:, :num_init_paths] = np.take_along_axis(
        inputs[:, 0], max_classes, axis=1
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = np.repeat(paths, num_classes, axis=0)
        scores = np.repeat(scores, num_classes)
        masked = np.repeat(masked, num_classes)

        path_tail_index = np.argmax(paths == _pad, axis=1)
        paths_arange = np.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = np.where(path_tail_index == 0, _pad, path_tails)

        classes = np.arange(num_classes)
        classes[mask_index] = _pad
        classes = np.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = np.where(masked_repeat, _pad, classes)
        paths[paths_arange, path_tail_index] = classes

        x = np.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = np.max(scores)
        scores_exp = np.exp(scores - scores_max)
        new_scores = np.zeros_like(scores)
        np.add.at(new_scores, unique_inverse, scores_exp)
        return np.log(new_scores) + scores_max

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)

        emit_scores = np.where(masked, -np.inf, scores)
        mask_scores = np.where(masked, scores, -np.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = np.logaddexp(emit_scores, mask_scores)
        top_indices = np.argsort(total_scores, kind="stable")[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = np.tile(paths, (2, 1))
        scores = np.concatenate([emit_scores, mask_scores])
        masked = np.concatenate(
            [np.zeros(beam_width, bool), np.ones(beam_width, bool)]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x
        if not seqlen_mask:
            paths, scores, masked = _decode_step(paths, scores, masked, x)
        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        def np_scan_only_carry(f, init, xs):
            carry = init
            for x in zip(*xs):
                carry, y = f(carry, x)
            return carry, None

        (paths, scores, masked), _ = np_scan_only_carry(
            _step,
            (init_paths, init_scores, init_masked),
            (inputs[1:], seqlen_mask[1:]),
        )

        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = np.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    results = [
        _decode_batch(p, s, m, i, sm)
        for p, s, m, i, sm in zip(
            init_paths, init_scores, init_masked, inputs, seqlen_mask
        )
    ]
    paths = np.stack([r[0] for r in results])
    scores = np.stack([r[1] for r in results])

    # convert classes back to the correct indices
    paths = np.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = np.transpose(paths, [1, 0, 2])
    return paths, scores


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )


def psnr(x1, x2, max_val):
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = np.mean(np.square(x1 - x2))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return np.asarray(val * -0.7, dtype=dtype)


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = np.ones_like(logits, dtype=np.bool_)
    if mask is not None:
        combined_mask = np.logical_and(combined_mask, mask)

    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        mask = np.tril(np.ones((T, S), dtype=np.bool_))
        mask = mask[None, None, :, :]
        combined_mask = np.logical_and(combined_mask, mask)

    padded_logits = np.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )
    return padded_logits


def _dot_product_attention_xla(query, key, value, bias, mask, is_causal, scale):
    original_dtype = key.dtype
    logits_dtype = np.promote_types(query.dtype, np.float32)
    if standardize_dtype(key.dtype) == "bfloat16":
        # `np.einsum` doesn't support bfloat16
        key = key.astype("float32")
        value = value.astype("float32")
    logits = np.einsum("BTNH,BSNH->BNTS", query, key)
    logits = logits.astype(logits_dtype)
    logits *= np.array(scale, dtype=logits.dtype)

    if bias is not None:
        logits = (logits + bias).astype(logits.dtype)

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax and it is always carried out in fp32.
    padded_logits = padded_logits.astype(np.float32)
    probs = softmax(padded_logits, axis=-1).astype(original_dtype)
    encoded_dtype = probs.dtype
    if standardize_dtype(probs.dtype) == "bfloat16":
        # `np.einsum` doesn't support bfloat16
        probs = probs.astype("float32")
        value = value.astype("float32")
    encoded = np.einsum("BNTS,BSNH->BTNH", probs, value)
    encoded = encoded.astype(encoded_dtype)
    return encoded


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
    if flash_attention is None:
        flash_attention = False
    if flash_attention:
        raise ValueError("Flash attention is not supported in numpy backend.")

    # Ref: jax.nn.dot_product_attention
    # https://github.com/jax-ml/jax/blob/jax-v0.4.32/jax/_src/nn/functions.py#L828
    # Not support `query_seq_lengths` and `key_value_seq_lengths` args
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = cast(query, compute_dtype)
    key = cast(key, compute_dtype)
    value = cast(value, compute_dtype)
    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)

    _, _, _, H = key.shape
    scale = (1.0 / np.sqrt(H)) if scale is None else scale
    return _dot_product_attention_xla(
        query, key, value, bias, mask, is_causal, scale
    )


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """NumPy implementation of Unfold.
    Extract sliding local blocks from a **NCHW** batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W)  **required**.
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    k = _pair(kernel_size)
    d = _pair(dilation)
    p = _pair(padding)
    s = _pair(stride)

    N, C, H, W = input.shape

    # ---- padding ----
    if any(_ > 0 for _ in p):
        input = np.pad(
            input, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode="constant"
        )

    # ----  spatial size ----
    oH = (input.shape[2] - (k[0] - 1) * d[0] - 1) // s[0] + 1
    oW = (input.shape[3] - (k[1] - 1) * d[1] - 1) // s[1] + 1

    i0 = np.arange(0, oH) * s[0]
    j0 = np.arange(0, oW) * s[1]
    i, j = np.meshgrid(i0, j0, indexing="ij")  # shape (oH, oW)
    i = i.reshape(-1)
    j = j.reshape(-1)

    # ---- flatten patches ----
    patches = np.empty((N, C, k[0], k[1], oH * oW), dtype=input.dtype)
    for idx in range(k[0]):
        for jdx in range(k[1]):
            patches[:, :, idx, jdx, :] = input[
                :, :, i + idx * d[0], j + jdx * d[1]
            ]

    # ---- reshape -> (N, C*kH*kW, L) ----
    return patches.reshape(N, C * k[0] * k[1], -1)


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """NumPy implementation of Fold (col2im).
    Combine an array of sliding local blocks into a large tensor.

    Args:
        x: 3-D tensor, shape (N, C*kH*kW, L)  **required**.
        output_size: int or (oH, oW)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        4-D tensor, shape (N, C, oH, oW)
    """

    def _pair(val):
        return (val, val) if isinstance(val, int) else val

    oH, oW = _pair(output_size)
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    pH, pW = _pair(padding)
    sH, sW = _pair(stride)

    N, CKK, L = x.shape
    C = CKK // (kH * kW)

    # Number of output patches along each dimension
    nH = (oH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    nW = (oW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # Reshape: (N, C*kH*kW, L) -> (N, C, kH, kW, nH, nW)
    x = np.reshape(x, (N, C, kH, kW, nH, nW))

    # Padded output size
    oH_pad = oH + 2 * pH
    oW_pad = oW + 2 * pW

    output = np.zeros((N, C, oH_pad, oW_pad), dtype=x.dtype)

    for i in range(kH):
        for j in range(kW):
            h_start = i * dH
            w_start = j * dW
            h_indices = h_start + np.arange(nH) * sH
            w_indices = w_start + np.arange(nW) * sW
            h_ix, w_ix = np.ix_(h_indices, w_indices)
            output[:, :, h_ix, w_ix] += x[:, :, i, j, :, :]

    # Remove padding
    if pH > 0 or pW > 0:
        output = output[:, :, pH : oH_pad - pH, pW : oW_pad - pW]

    return output


def depth_to_space(x, block_size, data_format="channels_last"):
    """NumPy implementation of depth_to_space (pixel shuffle).

    Rearranges data from depth into blocks of spatial data.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H*block_size, W*block_size, C/block_size**2)
        for channels_last or (N, C/block_size**2, H*block_size, W*block_size)
        for channels_first.
    """
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, H, W, C) -> (N, H, W, block_size, block_size, new_C)
        x = np.reshape(x, (n, h, w, block_size, block_size, new_c))
        # Transpose to (N, H, bH, W, bW, new_C) to interleave spatial blocks.
        x = np.transpose(x, (0, 1, 3, 2, 4, 5))
        # Reshape to the final spatial dimensions.
        x = np.reshape(x, (n, h * block_size, w * block_size, new_c))
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, C, H, W) -> (N, new_C, block_size, block_size, H, W)
        x = np.reshape(x, (n, new_c, block_size, block_size, h, w))
        # Transpose: (N, C, bH, bW, H, W) -> (N, C, H, bH, W, bW)
        x = np.transpose(x, (0, 1, 4, 2, 5, 3))
        # Reshape: (N, C, H, bH, W, bW) -> (N, C, H*bH, W*bW)
        x = np.reshape(x, (n, new_c, h * block_size, w * block_size))
    return x


def space_to_depth(x, block_size, data_format="channels_last"):
    """NumPy implementation of space_to_depth (pixel unshuffle).

    Rearranges blocks of spatial data into depth.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H/block_size, W/block_size, C*block_size**2)
        for channels_last or (N, C*block_size**2, H/block_size, W/block_size)
        for channels_first.
    """
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, H, W, C) -> (N, new_H, bH, new_W, bW, C)
        x = np.reshape(x, (n, new_h, block_size, new_w, block_size, c))
        # Transpose: -> (N, new_H, new_W, bH, bW, C)
        x = np.transpose(x, (0, 1, 3, 2, 4, 5))
        # Reshape: -> (N, new_H, new_W, C*bH*bW)
        x = np.reshape(x, (n, new_h, new_w, c * block_size**2))
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, C, H, W) -> (N, C, new_H, bH, new_W, bW)
        x = np.reshape(x, (n, c, new_h, block_size, new_w, block_size))
        # Transpose: -> (N, C, bH, bW, new_H, new_W)
        x = np.transpose(x, (0, 1, 3, 5, 2, 4))
        # Reshape: -> (N, C*bH*bW, new_H, new_W)
        x = np.reshape(x, (n, c * block_size**2, new_h, new_w))
    return x
