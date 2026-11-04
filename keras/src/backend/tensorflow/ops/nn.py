import functools
import math
import numbers
import warnings

import tensorflow as tf

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
    compute_conv_transpose_output_shape,
)
from keras.src.backend.tensorflow.ops.core import cast
from keras.src.backend.tensorflow.ops.core import convert_to_tensor


@functools.lru_cache(maxsize=None)
def _cpu_only():
    """Whether the TF runtime sees only CPU devices.

    Cached because the device list does not change during a run and
    `tf.config.list_logical_devices()` is comparatively expensive to call on
    every conv invocation in eager mode.
    """
    return all(d.device_type == "CPU" for d in tf.config.list_logical_devices())


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def sigmoid(x):
    logits = x
    output = tf.nn.sigmoid(x)
    output._keras_logits = logits
    return output


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tf.where(
        x <= -1,
        tf.constant(0.0, dtype=x.dtype),
        tf.where(x >= 1, tf.constant(1.0, dtype=x.dtype), 0.5 * (x + 1)),
    )


def tanh(x):
    return tf.nn.tanh(x)


def tanh_shrink(x):
    return x - tf.math.tanh(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tf.where(
        x > threshold,
        x - threshold,
        tf.where(x < -threshold, x + threshold, tf.zeros_like(x)),
    )


def sparse_plus(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tf.where(
        x <= -1,
        tf.zeros_like(x),
        tf.where(x < 1, (1 / 4) * tf.pow(x + 1, 2), x),
    )


def silu(x):
    return tf.nn.silu(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + tf.sqrt(tf.square(x) + b)
    return y / 2


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return relu6(x + tf.constant(3.0, x.dtype)) / tf.constant(6.0, x.dtype)


def hard_silu(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)


def selu(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tf.nn.selu(x)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    return tf.nn.gelu(x, approximate=approximate)


def celu(x, alpha=1.0):
    return tf.maximum(x, 0.0) + alpha * tf.math.expm1(
        tf.minimum(x, 0.0) / alpha
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    canonicalize_axis(axis, len(x.shape))
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=axis)
    return x1 * tf.sigmoid(x2)


def hard_tanh(x):
    return tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)


def hard_shrink(x, threshold=0.5):
    return tf.where(tf.abs(x) > threshold, x, tf.zeros_like(x))


def threshold(x, threshold, default_value):
    return tf.where(x > threshold, x, default_value)


def softmax(x, axis=-1):
    logits = x
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.softmax(output, axis=-1)
        output = tf.reshape(output, tf.shape(x))
    else:
        output = tf.nn.softmax(x, axis=axis)
    output._keras_logits = logits
    return output


def log_softmax(x, axis=-1):
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.log_softmax(output, axis=-1)
        return tf.reshape(output, tf.shape(x))
    return tf.nn.log_softmax(x, axis=axis)


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    logits_sorted = tf.sort(logits, direction="DESCENDING", axis=axis)
    logits_cumsum = tf.cumsum(logits_sorted, axis=axis)
    r = tf.range(1, tf.shape(logits)[axis] + 1, dtype=logits.dtype)
    r_shape = [1] * len(logits.shape)
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = tf.reshape(r, r_shape)  # Reshape for broadcasting
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    # Find the threshold
    logits_sorted_safe = tf.where(support, logits_sorted, 0.0)
    k = tf.reduce_sum(tf.cast(support, logits.dtype), axis=axis, keepdims=True)
    tau = (tf.reduce_sum(logits_sorted_safe, axis=axis, keepdims=True) - 1) / k
    output = tf.maximum(logits - tau, 0.0)
    return output


def _transpose_spatial_inputs(inputs):
    num_spatial_dims = len(inputs.shape) - 2
    # Tensorflow pooling does not support `channels_first` format, so
    # we need to transpose to `channels_last` format.
    if num_spatial_dims == 1:
        inputs = tf.transpose(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
    else:
        raise ValueError(
            "Pooling inputs's shape must be 3, 4 or 5, corresponding to 1D, 2D "
            f"and 3D inputs. But received shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the transpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = tf.transpose(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = tf.transpose(outputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        outputs = tf.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.max_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.avg_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def _compute_static_gather_indices(
    input_dim, output_size, small_window, big_window
):
    """Compute gather indices for Two-Pool Gather method (corrected)."""
    window_starts = tf.cast(
        tf.floor(
            tf.cast(tf.range(output_size), tf.float32)
            * tf.cast(input_dim, tf.float32)
            / tf.cast(output_size, tf.float32)
        ),
        tf.int32,
    )
    window_ends = tf.cast(
        tf.math.ceil(
            tf.cast(tf.range(1, output_size + 1), tf.float32)
            * tf.cast(input_dim, tf.float32)
            / tf.cast(output_size, tf.float32)
        ),
        tf.int32,
    )

    window_ends = tf.minimum(window_ends, input_dim)
    window_starts = tf.minimum(window_starts, input_dim - 1)

    window_sizes = window_ends - window_starts
    is_big_window = tf.equal(window_sizes, big_window)

    small_pool_len = max(1, input_dim - small_window + 1)

    small_indices = window_starts
    big_indices = window_starts + small_pool_len

    gather_indices = tf.where(is_big_window, big_indices, small_indices)
    return tf.cast(gather_indices, tf.int32)


def _adaptive_average_pool1d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size,)
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 1))

    static_shape = inputs.shape.as_list()
    l_static = static_shape[1]
    out_l = output_size[0]

    if l_static is None:
        raise ValueError(
            "Input length must be statically known for adaptive pooling"
        )

    small_l, big_l = compute_adaptive_pooling_window_sizes(l_static, out_l)
    gather_l = _compute_static_gather_indices(l_static, out_l, small_l, big_l)

    small_pool_l = tf.nn.pool(
        inputs,
        window_shape=(small_l,),
        pooling_type="AVG",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )
    big_pool_l = tf.nn.pool(
        inputs,
        window_shape=(big_l,),
        pooling_type="AVG",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )

    combined_l = tf.concat([small_pool_l, big_pool_l], axis=1)
    pooled_l = tf.gather(combined_l, gather_l, axis=1)

    if data_format == "channels_first":
        pooled_l = tf.transpose(pooled_l, (0, 2, 1))
    return pooled_l


def _adaptive_max_pool1d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size,)
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 1))

    static_shape = inputs.shape.as_list()
    l_static = static_shape[1]
    out_l = output_size[0]

    if l_static is None:
        raise ValueError(
            "Input length must be statically known for adaptive pooling"
        )

    small_l, big_l = compute_adaptive_pooling_window_sizes(l_static, out_l)
    gather_l = _compute_static_gather_indices(l_static, out_l, small_l, big_l)

    small_pool_l = tf.nn.pool(
        inputs,
        window_shape=(small_l,),
        pooling_type="MAX",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )
    big_pool_l = tf.nn.pool(
        inputs,
        window_shape=(big_l,),
        pooling_type="MAX",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )

    combined_l = tf.concat([small_pool_l, big_pool_l], axis=1)
    pooled_l = tf.gather(combined_l, gather_l, axis=1)

    if data_format == "channels_first":
        pooled_l = tf.transpose(pooled_l, (0, 2, 1))
    return pooled_l


def _adaptive_average_pool2d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 1))

    static_shape = inputs.shape.as_list()
    h_static = static_shape[1]
    w_static = static_shape[2]
    out_h, out_w = output_size

    if h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_h = tf.nn.pool(
        inputs,
        window_shape=(small_h, 1),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_h = tf.nn.pool(
        inputs,
        window_shape=(big_h, 1),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=1)
    pooled_h = tf.gather(combined_h, gather_h, axis=1)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, small_w),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, big_w),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=2)
    pooled_w = tf.gather(combined_w, gather_w, axis=2)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 3, 1, 2))

    return pooled_w


def _adaptive_max_pool2d(inputs, output_size, data_format="channels_first"):
    """Adaptive Max Pooling 2D using Two-Pool Gather method."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 1))

    static_shape = inputs.shape.as_list()
    h_static = static_shape[1]
    w_static = static_shape[2]
    out_h, out_w = output_size

    if h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_h = tf.nn.pool(
        inputs,
        window_shape=(small_h, 1),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_h = tf.nn.pool(
        inputs,
        window_shape=(big_h, 1),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=1)
    pooled_h = tf.gather(combined_h, gather_h, axis=1)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, small_w),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, big_w),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=2)
    pooled_w = tf.gather(combined_w, gather_w, axis=2)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 3, 1, 2))

    return pooled_w


def _adaptive_average_pool3d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

    static_shape = inputs.shape.as_list()
    d_static = static_shape[1]
    h_static = static_shape[2]
    w_static = static_shape[3]
    out_d, out_h, out_w = output_size

    if d_static is None or h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_d, big_d = compute_adaptive_pooling_window_sizes(d_static, out_d)
    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_d = _compute_static_gather_indices(d_static, out_d, small_d, big_d)
    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_d = tf.nn.pool(
        inputs,
        window_shape=(small_d, 1, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_d = tf.nn.pool(
        inputs,
        window_shape=(big_d, 1, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_d = tf.concat([small_pool_d, big_pool_d], axis=1)
    pooled_d = tf.gather(combined_d, gather_d, axis=1)

    small_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, small_h, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, big_h, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=2)
    pooled_h = tf.gather(combined_h, gather_h, axis=2)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, small_w),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, big_w),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=3)
    pooled_w = tf.gather(combined_w, gather_w, axis=3)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 4, 1, 2, 3))

    return pooled_w


def _adaptive_max_pool3d(inputs, output_size, data_format="channels_first"):
    """Adaptive Max Pooling 3D using Two-Pool Gather method."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

    static_shape = inputs.shape.as_list()
    d_static = static_shape[1]
    h_static = static_shape[2]
    w_static = static_shape[3]
    out_d, out_h, out_w = output_size

    if d_static is None or h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_d, big_d = compute_adaptive_pooling_window_sizes(d_static, out_d)
    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_d = _compute_static_gather_indices(d_static, out_d, small_d, big_d)
    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_d = tf.nn.pool(
        inputs,
        window_shape=(small_d, 1, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_d = tf.nn.pool(
        inputs,
        window_shape=(big_d, 1, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_d = tf.concat([small_pool_d, big_pool_d], axis=1)
    pooled_d = tf.gather(combined_d, gather_d, axis=1)

    small_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, small_h, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, big_h, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=2)
    pooled_h = tf.gather(combined_h, gather_h, axis=2)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, small_w),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, big_w),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=3)
    pooled_w = tf.gather(combined_w, gather_w, axis=3)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 4, 1, 2, 3))

    return pooled_w


def adaptive_average_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    data_format = backend.standardize_data_format(data_format)
    ndims = len(inputs.shape) - 2
    if ndims == 1:
        return _adaptive_average_pool1d(inputs, output_size, data_format)
    elif ndims == 2:
        return _adaptive_average_pool2d(inputs, output_size, data_format)
    elif ndims == 3:
        return _adaptive_average_pool3d(inputs, output_size, data_format)
    else:
        raise ValueError(
            "adaptive_average_pool supports 1D, 2D, or 3D inputs only."
        )


def adaptive_max_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    data_format = backend.standardize_data_format(data_format)
    ndims = len(inputs.shape) - 2
    if ndims == 1:
        return _adaptive_max_pool1d(inputs, output_size, data_format)
    elif ndims == 2:
        return _adaptive_max_pool2d(inputs, output_size, data_format)
    elif ndims == 3:
        return _adaptive_max_pool3d(inputs, output_size, data_format)
    else:
        raise ValueError(
            "adaptive_max_pool supports 1D, 2D, or 3D inputs only."
        )


def _convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
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
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = len(inputs.shape) - 2
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    else:
        strides = tuple(strides)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    else:
        dilation_rate = tuple(dilation_rate)
    # `tf.nn.convolution` rejects `strides > 1` together with
    # `dilation_rate > 1`. Decompose into a dilated stride-1 conv plus
    # subsampling, matching the JAX/Torch backends and the symbolic path.
    if _needs_depthwise_stride_dilation_decomposition(strides, dilation_rate):
        return _conv_stride_dilation_decomposition(
            inputs, kernel, strides, padding, data_format, dilation_rate
        )

    def _conv():
        tf_data_format = _convert_data_format(data_format, len(inputs.shape))
        result = tf.nn.convolution(
            inputs,
            kernel,
            strides,
            padding.upper(),
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        result_shape = result.shape
        if (
            result_shape.is_fully_defined()
            and math.prod(result_shape.as_list()) == 0
            and math.prod(inputs.shape.as_list()) != 0
        ):
            raise ValueError(
                "The convolution operation resulted in an empty output. "
                "Output shape:"
                f" {result_shape}. This can happen if the input is too small "
                "for the given kernel size, strides, dilation rate, and "
                "padding mode. Please check the input shape and convolution "
                "parameters."
            )
        return result

    # Certain ops are are broken in Tensorflow on CPU only.
    # We can work around by compiling the op with XLA.
    @tf.function(jit_compile=True)
    def _conv_xla():
        return _conv()

    # Channels first "NCW", NCHW" and "NCDHW" (1d, 2d and 3d convolutions) are
    # broken on CPU without XLA.
    needs_xla = (
        data_format == "channels_first"
        and len(inputs.shape) in (3, 4, 5)
        and _cpu_only()  # fmt: skip
    )
    # grouped convolutions are broken on CPU without XLA.
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    needs_xla = needs_xla or channels != kernel.shape[-2]
    if needs_xla:
        return _conv_xla()
    else:
        return _conv()


def _needs_depthwise_stride_dilation_decomposition(strides, dilation_rate):
    # `tf.nn.depthwise_conv2d` / `tf.nn.separable_conv2d` (and
    # `tf.nn.convolution`) do not support `strides > 1` together with
    # `dilation_rate > 1`. When both are requested we decompose instead.
    if dilation_rate is None:
        return False
    return any(s > 1 for s in strides) and any(d > 1 for d in dilation_rate)


def _pad_same_spatial_channels_last(
    inputs, kernel_size, strides, dilation_rate
):
    # Emulate `padding="same"` with explicit padding so the conv can run as
    # `padding="valid"`. Matches TF SAME: extra padding goes after (bottom).
    spatial_shape = tf.shape(inputs, out_type=tf.int32)[1:-1]

    paddings = [tf.constant([0, 0], dtype=tf.int32)]
    for i, k in enumerate(kernel_size):
        n = spatial_shape[i]
        s = strides[i]
        d = dilation_rate[i]
        effective_kernel_size = (k - 1) * d + 1
        out_size = (n + s - 1) // s
        total_pad = tf.maximum(
            (out_size - 1) * s + effective_kernel_size - n, 0
        )
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        paddings.append(tf.stack([pad_before, pad_after]))
    paddings.append(tf.constant([0, 0], dtype=tf.int32))

    return tf.pad(inputs, tf.stack(paddings))


def _depthwise_conv_stride_dilation_decomposition(
    inputs,
    depthwise_kernel,
    strides,
    padding,
    data_format,
    dilation_rate,
    pointwise_kernel=None,
):
    # Compute a depthwise (and optionally separable) conv with both stride > 1
    # and dilation > 1 by running the dilated conv at stride 1 and subsampling,
    # which TF supports. Always computed in `channels_last` layout.
    num_spatial_dims = len(inputs.shape) - 2
    padding = padding.upper()

    if data_format == "channels_first":
        inputs = _transpose_spatial_inputs(inputs)

    if num_spatial_dims == 1:
        inputs = tf.expand_dims(inputs, axis=1)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        if pointwise_kernel is not None:
            pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        spatial_strides = (1, strides[0])
        spatial_dilation_rate = (1, dilation_rate[0])
    else:
        spatial_strides = strides
        spatial_dilation_rate = dilation_rate

    kernel_size = tuple(int(i) for i in depthwise_kernel.shape[:2])

    if padding == "SAME":
        inputs = _pad_same_spatial_channels_last(
            inputs, kernel_size, spatial_strides, spatial_dilation_rate
        )
    elif padding != "VALID":
        raise ValueError(
            f"`padding` must be 'valid' or 'same'. Received: padding={padding}"
        )

    outputs = tf.nn.depthwise_conv2d(
        inputs,
        depthwise_kernel,
        strides=(1, 1, 1, 1),
        padding="VALID",
        data_format="NHWC",
        dilations=spatial_dilation_rate,
    )
    outputs = outputs[:, :: spatial_strides[0], :: spatial_strides[1], :]

    if pointwise_kernel is not None:
        outputs = tf.nn.conv2d(
            outputs,
            pointwise_kernel,
            strides=(1, 1, 1, 1),
            padding="VALID",
            data_format="NHWC",
        )

    if num_spatial_dims == 1:
        outputs = tf.squeeze(outputs, [1])

    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)

    return outputs


def _conv_stride_dilation_decomposition(
    inputs, kernel, strides, padding, data_format, dilation_rate
):
    # Compute a regular conv with both stride > 1 and dilation > 1 by running
    # the dilated conv at stride 1 (which `tf.nn.convolution` supports) and
    # subsampling the result. Always computed in `channels_last` layout.
    num_spatial_dims = len(inputs.shape) - 2
    padding = padding.upper()

    if data_format == "channels_first":
        inputs = _transpose_spatial_inputs(inputs)

    kernel_size = tuple(int(i) for i in kernel.shape[:num_spatial_dims])

    if padding == "SAME":
        inputs = _pad_same_spatial_channels_last(
            inputs, kernel_size, strides, dilation_rate
        )
    elif padding != "VALID":
        raise ValueError(
            f"`padding` must be 'valid' or 'same'. Received: padding={padding}"
        )

    def _run_conv():
        return tf.nn.convolution(
            inputs,
            kernel,
            strides=1,
            padding="VALID",
            data_format=_convert_data_format(
                "channels_last", num_spatial_dims + 2
            ),
            dilations=dilation_rate,
        )

    # Grouped convolutions are broken on CPU without XLA (see `conv`).
    # `inputs` is already `channels_last` at this point.
    if inputs.shape[-1] != kernel.shape[-2]:
        outputs = tf.function(_run_conv, jit_compile=True)()
    else:
        outputs = _run_conv()

    slices = [slice(None)]
    slices.extend(slice(None, None, s) for s in strides)
    slices.append(slice(None))
    outputs = outputs[tuple(slices)]

    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)

    return outputs


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
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`inputs` rank must be 3 (1D conv) or 4 (2D conv). Received: "
            f"{inputs.ndim}."
        )
    check_conv_input_channels(inputs, kernel, data_format)
    # Because we use `tf.nn.depthwise_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if _needs_depthwise_stride_dilation_decomposition(strides, dilation_rate):
        return _depthwise_conv_stride_dilation_decomposition(
            inputs, kernel, strides, padding, data_format, dilation_rate
        )
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        # `tf.nn.depthwise_conv2d` does not support `channels_first` with
        # dilations on CPU. Transpose to `channels_last`, compute, and
        # transpose back to avoid the limitation.
        need_transpose = data_format == "channels_first" and _cpu_only()
        if need_transpose:
            inputs = _transpose_spatial_inputs(inputs)
        if need_transpose or data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        kernel = tf.expand_dims(kernel, axis=0)

        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        if need_transpose or data_format == "channels_last":
            conv_data_format = _convert_data_format("channels_last", 4)
        else:
            conv_data_format = _convert_data_format("channels_first", 4)
        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides,
            padding,
            data_format=conv_data_format,
            dilations=dilation_rate,
        )
        outputs = tf.squeeze(outputs, [spatial_start_dim])
        if need_transpose:
            outputs = _transpose_spatial_outputs(outputs)
        return outputs

    # `tf.nn.depthwise_conv2d` does not support `channels_first` on CPU.
    # Transpose to `channels_last`, compute, and transpose back to avoid the
    # limitation. On GPU, `channels_first` (NCHW) is supported natively, so we
    # keep it to avoid unnecessary copies.
    need_transpose = data_format == "channels_first" and _cpu_only()
    if need_transpose:
        inputs = _transpose_spatial_inputs(inputs)
    if need_transpose or data_format == "channels_last":
        strides = (1,) + strides + (1,)
        conv_data_format = _convert_data_format("channels_last", 4)
    else:
        strides = (1, 1) + strides
        conv_data_format = tf_data_format
    outputs = tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides,
        padding,
        data_format=conv_data_format,
        dilations=dilation_rate,
    )
    if need_transpose:
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


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
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`num_spatial_dims` must be 1 or 2. Received: "
            f"num_spatial_dims={num_spatial_dims}."
        )
    check_conv_input_channels(inputs, depthwise_kernel, data_format)
    # Because we use `tf.nn.separable_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if _needs_depthwise_stride_dilation_decomposition(strides, dilation_rate):
        return _depthwise_conv_stride_dilation_decomposition(
            inputs,
            depthwise_kernel,
            strides,
            padding,
            data_format,
            dilation_rate,
            pointwise_kernel=pointwise_kernel,
        )
    if num_spatial_dims == 1:
        # 1D separable conv.
        # `tf.nn.separable_conv2d` does not support `channels_first` on CPU.
        # Transpose to `channels_last`, compute, and transpose back.
        need_transpose = data_format == "channels_first" and _cpu_only()
        if need_transpose:
            inputs = _transpose_spatial_inputs(inputs)
        if need_transpose or data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
            conv_data_format = _convert_data_format("channels_last", 4)
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
            conv_data_format = _convert_data_format("channels_first", 4)
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding,
            data_format=conv_data_format,
            dilations=dilation_rate,
        )
        outputs = tf.squeeze(outputs, [spatial_start_dim])
        if need_transpose:
            outputs = _transpose_spatial_outputs(outputs)
        return outputs

    # `tf.nn.separable_conv2d` does not support `channels_first` on CPU.
    # Transpose to `channels_last`, compute, and transpose back. On GPU,
    # NCHW is supported natively, so we keep it to avoid unnecessary copies.
    need_transpose = data_format == "channels_first" and _cpu_only()
    if need_transpose:
        inputs = _transpose_spatial_inputs(inputs)
    if need_transpose or data_format == "channels_last":
        strides = (1,) + strides + (1,)
        conv_data_format = _convert_data_format("channels_last", 4)
    else:
        strides = (1, 1) + strides
        conv_data_format = tf_data_format
    outputs = tf.nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format=conv_data_format,
        dilations=dilation_rate,
    )
    if need_transpose:
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


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
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    kernel_size = kernel.shape[:-2]
    filters = kernel.shape[-2]
    input_shape = list(inputs.shape)
    symbolic_shape = tf.shape(inputs)
    for i, e in enumerate(input_shape):
        if e is None:
            input_shape[i] = symbolic_shape[i]
    output_shape = compute_conv_transpose_output_shape(
        input_shape,
        kernel_size,
        filters,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )

    return tf.nn.conv_transpose(
        inputs,
        kernel,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    x = convert_to_tensor(x, dtype="int64")
    if dtype is None:
        dtype = "float32"
    else:
        dtype = standardize_dtype(dtype)
    axis = canonicalize_axis(axis, len(x.shape) + 1)
    if sparse:
        # We don't use `tf.sparse.bincount`, it doesn't handle negative indices
        # and only support rank 1 and 2 tensors (`one_hot` adds a dimension).
        values_count = math.prod(x.shape)
        values = tf.reshape(x, (values_count,))
        # We deal with negative inputs by having zeros in the output although
        # it's useless. It makes shapes static.
        values = tf.cast(tf.greater_equal(values, 0), dtype=dtype)
        indices = [tf.range(dim) for dim in x.shape]
        indices = tf.meshgrid(*indices, indexing="ij")
        indices.insert(axis, tf.maximum(x, 0))  # Deal with negative indices
        indices = [tf.reshape(a, (values_count, 1)) for a in indices]
        indices = [tf.cast(a, tf.int64) for a in indices]
        indices = tf.concat(indices, axis=1)
        shape = list(x.shape)
        shape.insert(axis, num_classes)
        return tf.SparseTensor(indices, values, shape)
    on_value, off_value = (True, False) if dtype == "bool" else (None, None)
    return tf.one_hot(
        x,
        num_classes,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
    )


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    reduction_axis = 1 if len(x.shape) > 1 else 0
    if standardize_dtype(dtype) == "bool":
        if sparse:
            # `tf.sparse.reduce_max` doesn't work on bool and there is no
            # `tf.sparse.reduce_any`.
            outputs = one_hot(
                x, num_classes, axis=axis, dtype="int8", sparse=True
            )
            outputs = tf.sparse.reduce_max(
                outputs, axis=reduction_axis, output_is_sparse=True
            )
            outputs_shape = outputs.shape
            outputs = tf.cast(outputs, dtype)
            outputs.set_shape(outputs_shape)
            return outputs
        else:
            outputs = one_hot(x, num_classes, axis=axis, dtype=dtype)
            return tf.reduce_any(outputs, axis=reduction_axis)
    else:
        if sparse:
            # We don't use `tf.sparse.bincount`, it doesn't handle negative
            # indices and has a rank limitation.
            outputs = one_hot(
                x, num_classes, axis=axis, dtype=dtype, sparse=True
            )
            return tf.sparse.reduce_max(
                outputs, axis=reduction_axis, output_is_sparse=True
            )
        else:
            outputs = one_hot(x, num_classes, axis=axis, dtype=dtype)
            return tf.reduce_max(outputs, axis=reduction_axis)


def _get_logits(output, from_logits, op_type, fn_name):
    """Retrieves logits tensor from maybe-softmax or maybe-sigmoid tensor."""
    output_ = output
    from_logits_ = from_logits

    has_keras_logits = hasattr(output, "_keras_logits")
    if has_keras_logits:
        output_ = output._keras_logits
        from_logits_ = True

    from_expected_op_type = (
        hasattr(output, "op")
        and not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable))
        and output.op.type == op_type
    ) and not has_keras_logits

    if from_expected_op_type:
        # When softmax activation function is used for output operation, we
        # use logits from the softmax function directly to compute loss in order
        # to prevent collapsing zero when training.
        if len(output.op.inputs) != 1:
            raise ValueError(f"Expected 1 input for {op_type}.")
        output_ = output.op.inputs[0]
        from_logits_ = True

    if from_logits and (has_keras_logits or from_expected_op_type):
        warnings.warn(
            f'"`{fn_name}` received `from_logits=True`, but '
            f"the `output` argument was produced by a {op_type} "
            "activation and thus does not represent logits. "
            "Was this intended?",
            stacklevel=2,
        )
    return output_, from_logits_


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is `True`, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.

    Example:

    >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0.]
       [0. 1. 0.]
       [0. 0. 1.]], shape=(3, 3), dtype=float32)
    >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94],
    ...                 shape=[3, 3])
    >>> print(b)
    tf.Tensor(
      [[0.9  0.05 0.05]
       [0.05 0.89 0.06]
       [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
    >>> loss = categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.11653 0.06188]
    >>> loss = categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0.]
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) != len(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same rank "
            "(ndim). Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "categorical_crossentropy"
    )
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, keepdims=True)

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(
        output, backend.epsilon(), 1.0 - backend.epsilon()
    )
    return -tf.reduce_sum(target * tf.math.log(output), axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    Args:
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.
    """
    if axis != -1 and axis != len(output.shape) - 1:
        raise ValueError(
            f"Only axis=-1 is currently supported. Received: axis={axis}"
        )
    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "sparse_categorical_crossentropy"
    )

    target = tf.convert_to_tensor(target)
    target = tf.cast(target, dtype="int64")
    output = tf.convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = tf.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if len(target.shape) != len(output.shape[:-1]):
        raise ValueError(
            "Argument `output` must have rank (ndim) `target.ndim - 1`. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape[:-1]):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape "
                "up until the last dimension: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    if not from_logits:
        output = tf.clip_by_value(
            output, backend.epsilon(), 1 - backend.epsilon()
        )
        output = tf.math.log(output)

    result = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=output
    )
    return result


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    if not is_float_dtype(output.dtype):
        output = tf.cast(output, backend.floatx())
    target = tf.cast(target, output.dtype)

    if len(target.shape) != len(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same rank "
            "(ndim). Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    output, from_logits = _get_logits(
        output, from_logits, "Sigmoid", "binary_crossentropy"
    )

    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(
        output, backend.epsilon(), 1.0 - backend.epsilon()
    )
    bce = target * tf.math.log(output)
    bce += (1 - target) * tf.math.log(1 - output)
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype in ("float16", "bfloat16"):
        need_cast = True
        x = cast(x, "float32")

    if synchronized:
        mean, variance = _compute_moments_sync(x, axes, keepdims)
    else:
        mean, variance = _compute_moments(x, axes, keepdims)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = tf.clip_by_value(mean, tf.float16.min, tf.float16.max)
        variance = tf.clip_by_value(variance, tf.float16.min, tf.float16.max)
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def _compute_moments_sync(x, axes, keepdims):
    replica_ctx = tf.distribute.get_replica_context()
    if not replica_ctx:
        return _compute_moments(x, axes, keepdims)

    local_count = tf.ones_like(x, name="count")

    local_sum = tf.reduce_sum(x, axis=axes, keepdims=True)
    local_squared_sum = tf.reduce_sum(tf.square(x), axis=axes, keepdims=True)
    local_count = tf.reduce_sum(local_count, axis=axes, keepdims=True)

    # TODO(b/163099951): batch the all-reduces once we sort out the
    # ordering issue for NCCL. We don't have a mechanism to launch
    # NCCL in the same order in each replica nowadays, so we limit
    # NCCL to batch all-reduces.
    y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
    y_squared_sum = replica_ctx.all_reduce(
        tf.distribute.ReduceOp.SUM, local_squared_sum
    )
    count_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_count)

    mean = tf.math.divide_no_nan(y_sum, count_sum)
    y_squared_mean = tf.math.divide_no_nan(y_squared_sum, count_sum)
    # var = E(x^2) - E(x)^2
    variance = tf.maximum(y_squared_mean - tf.square(mean), 0.0)
    if not keepdims:
        mean = tf.squeeze(mean, axes)
        variance = tf.squeeze(variance, axes)

    return mean, variance


def _compute_moments(x, axes, keepdims):
    return tf.nn.moments(x, axes, keepdims=keepdims)


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    if axis != -1:
        shape = [1] * len(x.shape)
        shape[axis] = mean.shape[0]
        mean = tf.reshape(mean, shape)
        variance = tf.reshape(variance, shape)
        if offset is not None:
            offset = tf.reshape(offset, shape)
        if scale is not None:
            scale = tf.reshape(scale, shape)

    return tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=offset,
        scale=scale,
        variance_epsilon=epsilon,
    )


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    target = tf.cast(target, dtype="int32")

    # `tf.nn.ctc_loss` will internally cast to float32 when the input is float16
    # or bfloat16. Additionally, it will raise an error when the input is
    # float64. As a result, we perform the casting externally and add support
    # for float64.
    result_dtype = backend.result_type(output.dtype, "float32")
    compute_dtype = "float32" if result_dtype == "float64" else result_dtype
    output = tf.cast(output, compute_dtype)
    loss = tf.nn.ctc_loss(
        labels=target,
        logits=output,
        label_length=target_length,
        logit_length=output_length,
        blank_index=mask_index,
        logits_time_major=False,
    )
    return tf.cast(loss, result_dtype)


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
    input_shape = tf.shape(inputs)
    num_samples, num_steps = input_shape[0], input_shape[1]
    inputs = tf.transpose(inputs, (1, 0, 2))

    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = tf.cast(inputs, dtype)

    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    if strategy == "greedy":
        (decoded, scores) = tf.nn.ctc_greedy_decoder(
            inputs=inputs,
            sequence_length=sequence_lengths,
            merge_repeated=merge_repeated,
            blank_index=mask_index,
        )
    elif strategy == "beam_search":
        # Move `mask_index` column to the last position since this is the
        # default for `tf.nn.ctc_beam_search_decoder`
        if mask_index is not None:
            inputs_before = inputs[..., :mask_index]
            inputs_mask = inputs[..., mask_index : mask_index + 1]
            inputs_after = inputs[..., mask_index + 1 :]
            inputs = tf.concat(
                [inputs_before, inputs_after, inputs_mask], axis=-1
            )
        (decoded, scores) = tf.nn.ctc_beam_search_decoder(
            inputs=inputs,
            sequence_length=sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )

    # Postprocess sparse tensor
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    decoded_dense = tf.stack(decoded_dense, axis=0)
    decoded_dense = tf.cast(decoded_dense, "int32")

    # We need to recover the labels because we swapped the indices earlier
    if strategy == "beam_search" and mask_index is not None:
        if mask_index < 0:
            mask_index = mask_index + input_shape[-1]
        decoded_dense = tf.where(
            decoded_dense >= mask_index, decoded_dense + 1, decoded_dense
        )
    return decoded_dense, scores


def psnr(x1, x2, max_val):
    from keras.src.backend.tensorflow.ops.numpy import log10

    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = tf.reduce_mean(tf.square(x1 - x2))
    psnr = 20 * log10(max_val) - 10 * log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return tf.constant(val * -0.7, dtype=dtype)


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = tf.ones_like(logits, dtype="bool")
    if mask is not None:
        combined_mask = tf.logical_and(combined_mask, mask)

    if is_causal:
        logits_shape = tf.shape(logits)
        T, S = logits_shape[2], logits_shape[3]
        mask = tf.linalg.band_part(tf.ones((T, S), "bool"), -1, 0)
        mask = mask[None, None, :, :]
        combined_mask = tf.logical_and(combined_mask, mask)

    padded_logits = tf.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )
    return padded_logits


def _dot_product_attention_xla(query, key, value, bias, mask, is_causal, scale):
    logits_dtype = backend.result_type(query.dtype, "float32")
    logits = tf.einsum("BTNH,BSNH->BNTS", query, key, optimize="optimal")
    logits = tf.cast(logits, logits_dtype)
    logits = tf.multiply(logits, tf.cast(scale, logits.dtype))

    if bias is not None:
        logits = tf.add(logits, tf.cast(bias, logits.dtype))

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax is always carried out in high precision.
    probs_dtype = backend.result_type(padded_logits.dtype, "float32")
    probs = tf.cast(
        tf.nn.softmax(tf.cast(padded_logits, probs_dtype), axis=-1), key.dtype
    )
    return tf.einsum("BNTS,BSNH->BTNH", probs, value, optimize="optimal")


def _check_flash_attention_support(
    query, bias, mask, scale, attn_logits_soft_cap
):
    """Raises a ValueError if `tf.nn.flash_attention` can't run these inputs."""

    def fail(message):
        raise ValueError(
            f"Flash attention is not supported for these inputs: {message}"
        )

    if not hasattr(tf.nn, "flash_attention"):
        fail("this TensorFlow build does not provide `tf.nn.flash_attention`.")
    if bias is not None or mask is not None:
        fail(
            "`bias` and `mask` are not supported. Use `is_causal=True` for "
            "causal masking."
        )
    if attn_logits_soft_cap is not None:
        fail("`attn_logits_soft_cap` is not supported.")
    if query.dtype not in (tf.float16, tf.bfloat16, tf.float32):
        fail(
            "inputs must be float16, bfloat16 or float32. "
            f"Received: dtype={query.dtype.name}."
        )
    if scale is not None and not isinstance(scale, numbers.Real):
        fail(f"`scale` must be a Python number. Received: {scale}.")
    if scale is None and query.shape[-1] is None:
        fail("the head dim must be statically known when `scale` is None.")


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
    dropout_rate=0.0,
    seed=None,
):
    # `dropout_rate` and `seed` are TensorFlow-only extensions used by the
    # attention layers: `tf.nn.flash_attention` applies dropout to the
    # attention weights inside the kernel, which the layers can't do
    # themselves since the weights are never materialized.
    if flash_attention is None:
        flash_attention = False
    if dropout_rate > 0.0 and not flash_attention:
        raise ValueError(
            "`dropout_rate` is only supported with `flash_attention=True`. "
            f"Received: dropout_rate={dropout_rate}"
        )

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

    num_query_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    if (
        num_query_heads is not None
        and num_kv_heads is not None
        and num_query_heads > num_kv_heads
        and num_kv_heads > 1
    ):
        groups = num_query_heads // num_kv_heads
        key = tf.repeat(key, repeats=groups, axis=2)
        value = tf.repeat(value, repeats=groups, axis=2)

    if flash_attention:
        _check_flash_attention_support(
            query, bias, mask, scale, attn_logits_soft_cap
        )
        return tf.nn.flash_attention(
            query,
            key,
            value,
            scale=None if scale is None else float(scale),
            is_causal=is_causal,
            dropout_rate=float(dropout_rate),
            seed=seed,
        )

    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)

    H = tf.shape(key)[-1]
    scale = (1.0 / tf.sqrt(tf.cast(H, "float32"))) if scale is None else scale
    return _dot_product_attention_xla(
        query, key, value, bias, mask, is_causal, scale
    )


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Tensorflow implementation of Unfold.
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
    k = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    d = (dilation, dilation) if isinstance(dilation, int) else dilation
    p = (padding, padding) if isinstance(padding, int) else padding
    s = (stride, stride) if isinstance(stride, int) else stride
    N, C, H, W = input.shape

    # ---- padding ----
    if any(_ > 0 for _ in p):
        input = tf.pad(input, [[0, 0], [0, 0], [p[0], p[0]], [p[1], p[1]]])
    x = tf.transpose(input, [0, 2, 3, 1])  # (N, H, W, C)
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, k[0], k[1], 1],
        strides=[1, s[0], s[1], 1],
        rates=[1, d[0], d[1], 1],
        padding="VALID",
    )  # (N, nH, nW, kH*kW*C)

    N, nH, nW, D = patches.shape
    patches = tf.reshape(
        patches, [N, nH, nW, k[0], k[1], C]
    )  # (N, nH, nW, kH, kW, C)
    patches = tf.transpose(
        patches, [0, 5, 3, 4, 1, 2]
    )  # (N, C, kH, kW, nH, nW)
    patches = tf.reshape(patches, [N, C * k[0] * k[1], nH * nW])
    return patches


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """TensorFlow implementation of Fold (col2im).
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
    k = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    o = (
        (output_size, output_size)
        if isinstance(output_size, int)
        else output_size
    )
    d = (dilation, dilation) if isinstance(dilation, int) else dilation
    p = (padding, padding) if isinstance(padding, int) else padding
    s = (stride, stride) if isinstance(stride, int) else stride

    N = tf.shape(x)[0]
    CKK = x.shape[1]
    kH, kW = k
    oH, oW = o
    C = CKK // (kH * kW)

    # Number of output patches along each dimension
    nH = (oH + 2 * p[0] - d[0] * (kH - 1) - 1) // s[0] + 1
    nW = (oW + 2 * p[1] - d[1] * (kW - 1) - 1) // s[1] + 1

    # Reshape: (N, C*kH*kW, L) -> (N, C, kH, kW, nH, nW)
    x = tf.reshape(x, [N, C, kH, kW, nH, nW])

    # Padded output size
    oH_pad = oH + 2 * p[0]
    oW_pad = oW + 2 * p[1]

    # Build scatter indices for all kernel positions
    # Process one sample at a time using vectorized_map
    def _fold_single(x_single):
        # x_single: (C, kH, kW, nH, nW)
        output = tf.zeros([C, oH_pad, oW_pad], dtype=x.dtype)
        for i in range(kH):
            for j in range(kW):
                h_start = i * d[0]
                w_start = j * d[1]
                h_indices = tf.range(nH) * s[0] + h_start
                w_indices = tf.range(nW) * s[1] + w_start
                # x_single[:, i, j, :, :] has shape (C, nH, nW)
                patch = x_single[:, i, j, :, :]
                # Build indices for scatter
                c_idx = tf.repeat(tf.range(C), nH * nW)  # (C*nH*nW,)
                h_idx = tf.tile(tf.repeat(h_indices, nW), [C])  # (C*nH*nW,)
                w_idx = tf.tile(tf.tile(w_indices, [nH]), [C])  # (C*nH*nW,)
                indices = tf.stack(
                    [c_idx, h_idx, w_idx], axis=1
                )  # (C*nH*nW, 3)
                values = tf.reshape(patch, [-1])  # (C*nH*nW,)
                output = tf.tensor_scatter_nd_add(output, indices, values)
        return output

    output = tf.vectorized_map(_fold_single, x)  # (N, C, oH_pad, oW_pad)

    # Remove padding
    if p[0] > 0 or p[1] > 0:
        output = output[:, :, p[0] : oH_pad - p[0], p[1] : oW_pad - p[1]]

    return output


def depth_to_space(x, block_size, data_format="channels_last"):
    """TensorFlow implementation of depth_to_space.

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
        return tf.nn.depth_to_space(x, block_size, data_format="NHWC")
    else:
        # NCHW format is not supported on CPU, so we transpose manually
        # NCHW -> NHWC
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.depth_to_space(x, block_size, data_format="NHWC")
        # NHWC -> NCHW
        x = tf.transpose(x, [0, 3, 1, 2])
        return x


def space_to_depth(x, block_size, data_format="channels_last"):
    """TensorFlow implementation of space_to_depth.

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
        return tf.nn.space_to_depth(x, block_size, data_format="NHWC")
    else:
        # NCHW format is not supported on CPU, so we transpose manually
        # NCHW -> NHWC
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.space_to_depth(x, block_size, data_format="NHWC")
        # NHWC -> NCHW
        x = tf.transpose(x, [0, 3, 1, 2])
        return x
