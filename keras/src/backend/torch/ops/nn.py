import math

import torch
import torch.nn.functional as tnn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

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
    compute_conv_transpose_output_crops_for_torch,
)
from keras.src.backend.torch.ops.core import cast
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.backend.torch.ops.numpy import expand_dims
from keras.src.backend.torch.ops.numpy import where
from keras.src.utils.argument_validation import standardize_tuple


def relu(x):
    x = convert_to_tensor(x)
    return tnn.relu(x)


def relu6(x):
    x = convert_to_tensor(x)
    return tnn.relu6(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.sigmoid(x)


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return torch.where(
        x <= -1,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        torch.where(
            x >= 1,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            0.5 * (x + 1),
        ),
    )


def tanh(x):
    x = convert_to_tensor(x)
    return tnn.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return tnn.tanhshrink(x)


def softplus(x):
    x = convert_to_tensor(x)
    return tnn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return tnn.softsign(x)


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tnn.softshrink(x, lambd=threshold)


def sparse_plus(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return torch.where(
        x <= -1,
        torch.zeros_like(x),
        torch.where(x < 1, (1 / 4) * (x + 1) ** 2, x),
    )


def silu(x):
    x = convert_to_tensor(x)
    return tnn.silu(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b)
    y = x + torch.sqrt(x**2 + b)
    return y / 2


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.logsigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    # `tnn.leaky_relu` is not implemented for integer or bool dtypes.
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    # `tnn.hardsigmoid` is not implemented for integer or bool dtypes.
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tnn.hardsigmoid(x)


def hard_silu(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tnn.hardswish(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return tnn.elu(x, alpha)


def selu(x):
    x = convert_to_tensor(x)
    float_dtype = dtypes.promote_to_float_dtype(x.dtype)
    if float_dtype != standardize_dtype(x.dtype):
        x = cast(x, float_dtype)
    return tnn.selu(x)


def gelu(x, approximate=True):
    # TODO: torch.nn.gelu expects string approximate of `"none"` or `"tanh"`
    x = convert_to_tensor(x)
    if approximate:
        return tnn.gelu(x, approximate="tanh")
    return tnn.gelu(x)


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return tnn.celu(x, alpha=alpha)


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    canonicalize_axis(axis, len(x.shape))
    return tnn.glu(x, dim=axis)


def hard_tanh(x):
    x = convert_to_tensor(x)
    return tnn.hardtanh(x, min_val=-1.0, max_val=1.0)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return tnn.hardshrink(x, lambd=threshold)


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return tnn.threshold(x, threshold=threshold, value=default_value)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    # TODO: tnn.softmax doesn't support float16 using cpu
    if get_device() == "cpu" and dtype == "float16":
        x = cast(x, "float32")
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = torch.reshape(x, [-1])
        output = tnn.softmax(output, dim=-1)
        output = torch.reshape(output, x.shape)
    else:
        output = tnn.softmax(x, dim=axis)
    return cast(output, dtype)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    # TODO: tnn.log_softmax doesn't support float16 using cpu
    if get_device() == "cpu" and dtype == "float16":
        x = cast(x, "float32")
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = torch.reshape(x, [-1])
        output = tnn.log_softmax(output, dim=-1)
        output = torch.reshape(output, x.shape)
    else:
        output = tnn.log_softmax(x, dim=axis)
    return cast(output, dtype)


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    logits_sorted, _ = torch.sort(logits, dim=axis, descending=True)
    logits_cumsum = torch.cumsum(logits_sorted, dim=axis)
    r = torch.arange(
        1, logits.size(axis) + 1, device=logits.device, dtype=logits.dtype
    )
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = r.view(r_shape)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    # Find the threshold
    k = torch.sum(support, dim=axis, keepdim=True)
    logits_sorted_safe = torch.where(
        support, logits_sorted, torch.tensor(0.0, device=logits.device)
    )
    tau = (torch.sum(logits_sorted_safe, dim=axis, keepdim=True) - 1) / k
    output = torch.clamp(logits - tau, min=0.0)
    return output


def _compute_padding_length(
    input_length, kernel_length, stride, dilation_rate=1
):
    """Compute padding length along one dimension with support
    for asymmetric padding."""
    effective_k_size = (kernel_length - 1) * dilation_rate + 1

    # calc. needed padding for case with stride involved
    output_size = (input_length + stride - 1) // stride
    total_padding = max(
        0, (output_size - 1) * stride + effective_k_size - input_length
    )

    # divide padding evenly, with extra pixel going at the end if needed
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return (left_padding, right_padding)


def _apply_same_padding(
    inputs,
    kernel_size,
    strides,
    data_format,
    padding_mode="constant",
    dilation_rate=1,
):
    """Apply same padding to the input tensor.

    This function will evaluate if the padding value is compatible with torch
    functions. To avoid calling `pad()` as much as possible, which may cause
    performance or memory issues, when compatible, it does not apply the padding
    to the tensor, but returns the input tensor and the padding value to pass to
    the torch functions. If not compatible, it returns the padded tensor and 0
    as the padding value.

    Returns:
        tensor: A padded tensor or the inputs.
        padding: The padding value, ready to pass to the torch functions.
    """
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = []

    dilation_rate = standardize_tuple(
        dilation_rate, num_spatial_dims, "dilation_rate"
    )

    for i in range(num_spatial_dims):
        pad = _compute_padding_length(
            spatial_shape[i], kernel_size[i], strides[i], dilation_rate[i]
        )
        padding.append(pad)

    # convert padding to torch format
    if all(left == right for left, right in padding):
        return inputs, [left for left, _ in padding]

    # else, need to pad manually
    flattened_padding = []
    for pad in reversed(padding):
        flattened_padding.extend(pad)

    return tnn.pad(inputs, pad=tuple(flattened_padding), mode=padding_mode), 0


def _transpose_spatial_inputs(inputs, channels_last_memory_format=False):
    """Transpose inputs from channels_last to channels_first format."""
    # Torch pooling does not support `channels_last` format, so
    # we need to transpose to `channels_first` format.
    # After permute, the tensor may be non-contiguous which causes
    # failures in view-based ops (e.g., conv2d, torch.export) that
    # require contiguous memory. Adding .contiguous() ensures
    # compatible memory layout.
    # `channels_last_memory_format` selects torch's channels_last
    # `memory_format` for the result instead of the default contiguous one.
    # It does not change the dimension order, which is always channels_first
    # here: a permuted NHWC->NCHW view of a contiguous tensor already has
    # channels_last strides, so requesting that format copies nothing.
    ndim = inputs.ndim - 2
    if ndim == 1:  # 1D case
        return torch.permute(inputs, (0, 2, 1)).contiguous()
    elif ndim == 2:  # 2D case
        inputs = torch.permute(inputs, (0, 3, 1, 2))
        if channels_last_memory_format:
            return inputs.contiguous(memory_format=torch.channels_last)
        return inputs.contiguous()
    elif ndim == 3:  # 3D case
        inputs = torch.permute(inputs, (0, 4, 1, 2, 3))
        if channels_last_memory_format:
            return inputs.contiguous(memory_format=torch.channels_last_3d)
        return inputs.contiguous()
    raise ValueError(
        "Inputs must have ndim=3, 4 or 5, "
        "corresponding to 1D, 2D and 3D inputs. "
        f"Received input shape: {inputs.shape}."
    )


def _transpose_spatial_outputs(outputs):
    # Undo the transpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = torch.permute(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = torch.permute(outputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        outputs = torch.permute(outputs, (0, 2, 3, 4, 1))
    return outputs


def _transpose_conv_kernel(kernel):
    # Torch requires conv kernel of format
    # `(out_channels, in_channels, spatial_dims)`, we need to transpose.
    num_spatial_dims = len(kernel.shape) - 2
    if num_spatial_dims == 1:
        kernel = torch.permute(kernel, (2, 1, 0))
    elif num_spatial_dims == 2:
        kernel = torch.permute(kernel, (3, 2, 0, 1))
    elif num_spatial_dims == 3:
        kernel = torch.permute(kernel, (4, 3, 0, 1, 2))
    return kernel


def _get_channels_last_memory_format(ndim):
    if ndim == 4:
        return torch.channels_last
    elif ndim == 5:
        return torch.channels_last_3d
    return None


def _maybe_convert_to_channels_last(tensor):
    mem_fmt = _get_channels_last_memory_format(tensor.ndim)
    if mem_fmt is not None and not tensor.is_contiguous(memory_format=mem_fmt):
        return tensor.contiguous(memory_format=mem_fmt)
    return tensor


def _is_pointwise_kernel(kernel):
    return all(dim == 1 for dim in kernel.shape[:-2])


def _conv_pointwise_channels_last(inputs, kernel, strides):
    if any(stride != 1 for stride in strides):
        spatial_slices = tuple(slice(None, None, stride) for stride in strides)
        inputs = inputs[(slice(None), *spatial_slices, slice(None))]
    kernel = torch.reshape(kernel, (kernel.shape[-2], kernel.shape[-1]))
    return torch.matmul(inputs, kernel)


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Fixed max pooling implementation."""
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    if strides is None:
        strides = pool_size
    else:
        strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(
            inputs, channels_last_memory_format=True
        )

    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs, padding = _apply_same_padding(
            inputs, pool_size, strides, data_format, padding_mode="replicate"
        )
    else:
        padding = 0

    device = get_device()
    # Torch max pooling ops do not support symbolic tensors.
    # Create a real tensor to execute the ops.
    if device == "meta":
        inputs = torch.empty(
            size=inputs.shape, dtype=inputs.dtype, device="cpu"
        )

    if num_spatial_dims == 1:
        outputs = tnn.max_pool1d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    elif num_spatial_dims == 2:
        outputs = tnn.max_pool2d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    elif num_spatial_dims == 3:
        outputs = tnn.max_pool3d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    else:
        raise ValueError(
            "Inputs to pooling op must have ndim=3, 4 or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )

    outputs = outputs.to(device)
    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Fixed average pooling with correct padding calculation."""
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    strides = (
        pool_size
        if strides is None
        else standardize_tuple(strides, num_spatial_dims, "strides")
    )

    data_format = backend.standardize_data_format(data_format)
    orig_format = data_format

    if data_format == "channels_last":
        # 2D only: `avg_pool3d` has no `channels_last_3d` kernel, so 3D gains
        # nothing and its uneven `same` padding path gets slower.
        inputs = _transpose_spatial_inputs(
            inputs, channels_last_memory_format=num_spatial_dims == 2
        )

    orig_inputs = inputs
    manual_padded = False

    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs, padding = _apply_same_padding(
            inputs,
            pool_size,
            strides,
            "channels_first",  # we're in channels_first here
        )
        if padding == 0:
            manual_padded = True
            ones = torch.ones_like(orig_inputs)
            ones_padded, _ = _apply_same_padding(
                ones,
                pool_size,
                strides,
                "channels_first",
            )
    else:
        padding = 0

    # apply pooling
    if num_spatial_dims == 1:
        outputs = tnn.avg_pool1d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
        if manual_padded:
            outputs_ones = tnn.avg_pool1d(
                ones_padded,
                kernel_size=pool_size,
                stride=strides,
                padding=padding,
                count_include_pad=False,
            )
    elif num_spatial_dims == 2:
        outputs = tnn.avg_pool2d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
        if manual_padded:
            outputs_ones = tnn.avg_pool2d(
                ones_padded,
                kernel_size=pool_size,
                stride=strides,
                padding=padding,
                count_include_pad=False,
            )
    elif num_spatial_dims == 3:
        outputs = tnn.avg_pool3d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
        if manual_padded:
            outputs_ones = tnn.avg_pool3d(
                ones_padded,
                kernel_size=pool_size,
                stride=strides,
                padding=padding,
                count_include_pad=False,
            )
    else:
        raise ValueError(
            "Inputs to pooling op must have ndim=3, 4 or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )

    if manual_padded:
        outputs = outputs / outputs_ones

    if orig_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)

    return outputs


def adaptive_average_pool(inputs, output_size, data_format=None):
    """Adaptive average pooling(1D/2D/3D) with channels_last support."""
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2

    data_format = backend.standardize_data_format(data_format)
    orig_format = data_format
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(
            inputs, channels_last_memory_format=True
        )

    if isinstance(output_size, int):
        torch_output_size = (
            output_size
            if num_spatial_dims == 1
            else (output_size,) * num_spatial_dims
        )
    else:
        torch_output_size = standardize_tuple(
            output_size, num_spatial_dims, "output_size"
        )

    if get_device() == "meta":
        inputs = torch.empty(
            size=inputs.shape, dtype=inputs.dtype, device="cpu"
        )

    if num_spatial_dims == 1:
        outputs = tnn.adaptive_avg_pool1d(inputs, output_size=torch_output_size)
    elif num_spatial_dims == 2:
        outputs = tnn.adaptive_avg_pool2d(inputs, output_size=torch_output_size)
    elif num_spatial_dims == 3:
        outputs = tnn.adaptive_avg_pool3d(inputs, output_size=torch_output_size)
    else:
        raise ValueError(
            "Inputs to adaptive average pooling must have ndim=3, 4 or 5, "
            f"Received input shape: {inputs.shape}."
        )

    if orig_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def adaptive_max_pool(inputs, output_size, data_format=None):
    """Adaptive max pooling(1D/2D/3D) with channels_last support."""
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2

    data_format = backend.standardize_data_format(data_format)
    orig_format = data_format
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)

    if isinstance(output_size, int):
        torch_output_size = (
            output_size
            if num_spatial_dims == 1
            else (output_size,) * num_spatial_dims
        )
    else:
        torch_output_size = standardize_tuple(
            output_size, num_spatial_dims, "output_size"
        )

    if get_device() == "meta":
        inputs = torch.empty(
            size=inputs.shape, dtype=inputs.dtype, device="cpu"
        )

    if num_spatial_dims == 1:
        res = tnn.adaptive_max_pool1d(inputs, output_size=torch_output_size)
    elif num_spatial_dims == 2:
        res = tnn.adaptive_max_pool2d(inputs, output_size=torch_output_size)
    elif num_spatial_dims == 3:
        res = tnn.adaptive_max_pool3d(inputs, output_size=torch_output_size)
    else:
        raise ValueError(
            "Inputs to adaptive max pooling must have ndim=3, 4 or 5, "
            f"Received input shape: {inputs.shape}."
        )

    outputs = res[0] if isinstance(res, tuple) else res

    if orig_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """Convolution with fixed group handling."""
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = backend.standardize_data_format(data_format)
    # Fast path for pointwise channels_last conv: matmul avoids the
    # channels_first transpose and torch conv dispatch.
    if (
        data_format == "channels_last"
        and padding in {"valid", "same"}
        and _is_pointwise_kernel(kernel)
        and inputs.shape[-1] == kernel.shape[-2]
    ):
        return _conv_pointwise_channels_last(inputs, kernel, strides)

    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(
            inputs, channels_last_memory_format=True
        )

    kernel = _transpose_conv_kernel(kernel)

    if data_format == "channels_last":
        kernel = _maybe_convert_to_channels_last(kernel)

    # calc. groups snippet
    in_channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[1]
    if in_channels % kernel_in_channels != 0:
        raise ValueError(
            f"Input channels ({in_channels}) must be divisible by "
            f"kernel input channels ({kernel_in_channels})"
        )
    groups = in_channels // kernel_in_channels

    # handle padding
    if padding == "same":
        inputs, padding = _apply_same_padding(
            inputs,
            kernel.shape[2:],
            strides,
            data_format,
            padding_mode="constant",
            dilation_rate=dilation_rate,
        )
    else:
        padding = 0

    # apply convolution
    if num_spatial_dims == 1:
        outputs = tnn.conv1d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=groups,
        )
    elif num_spatial_dims == 2:
        outputs = tnn.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=groups,
        )
    elif num_spatial_dims == 3:
        outputs = tnn.conv3d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=groups,
        )
    else:
        raise ValueError(
            "Inputs to conv operation should have ndim=3, 4, or 5,"
            "corresponding to 1D, 2D and 3D inputs. Received input "
            f"shape: {inputs.shape}."
        )

    if data_format == "channels_last":
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
    check_conv_input_channels(inputs, kernel, data_format)
    kernel = torch.reshape(
        kernel, kernel.shape[:-2] + (1, kernel.shape[-2] * kernel.shape[-1])
    )
    return conv(inputs, kernel, strides, padding, data_format, dilation_rate)


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
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = backend.standardize_data_format(data_format)
    check_conv_transpose_input_channels(inputs, kernel, data_format)

    # Torch's `conv_transpose*d` only takes a symmetric `padding` plus a
    # right-side `output_padding`, which cannot reproduce the asymmetric
    # padding Keras's "same" mode requires when stride > 1 with an odd
    # kernel. We call torch with `padding=0, output_padding=0` (giving the
    # largest "natural" output) and asymmetrically slice the spatial dims to
    # the same window JAX would compute. Negative crops mean we extend with
    # zero-padding to match JAX's left/right pad semantics.
    crops = compute_conv_transpose_output_crops_for_torch(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )

    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)
    # Transpose kernel from keras format to torch format.
    kernel = _transpose_conv_kernel(kernel)

    if isinstance(dilation_rate, int):
        dilation_rate = [dilation_rate] * num_spatial_dims

    if num_spatial_dims == 1:
        outputs = tnn.conv_transpose1d(
            inputs,
            kernel,
            stride=strides,
            padding=0,
            output_padding=0,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 2:
        outputs = tnn.conv_transpose2d(
            inputs,
            kernel,
            stride=strides,
            padding=0,
            output_padding=0,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 3:
        outputs = tnn.conv_transpose3d(
            inputs,
            kernel,
            stride=strides,
            padding=0,
            output_padding=0,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(
            "Inputs to conv transpose operation should have ndim=3, 4, or 5,"
            "corresponding to 1D, 2D and 3D inputs. Received input "
            f"shape: {inputs.shape}."
        )

    # Apply asymmetric crop (or zero-pad if a crop amount is negative) to
    # each spatial dim. `outputs` is NCHW-style here; spatial dims start at
    # axis 2.
    slices = [slice(None), slice(None)]
    needs_zero_pad = any(cl < 0 or cr < 0 for cl, cr in crops)
    for crop_left, crop_right in crops:
        start = max(0, crop_left)
        end = -crop_right if crop_right > 0 else None
        slices.append(slice(start, end))
    outputs = outputs[tuple(slices)]
    if needs_zero_pad:
        # torch's F.pad takes pads in REVERSE spatial order: last dim first.
        pads = []
        for crop_left, crop_right in reversed(crops):
            pads.extend(
                [
                    -crop_left if crop_left < 0 else 0,
                    -crop_right if crop_right < 0 else 0,
                ]
            )
        outputs = tnn.pad(outputs, pads)

    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    # Axis is the output axis. By default, PyTorch, outputs to last axis.
    # If axis is not last, change output to axis and shift remaining elements.
    x = convert_to_tensor(x, dtype=torch.long)
    zero = convert_to_tensor(0, dtype=torch.long)

    # Torch one_hot does not natively handle negative or out of range values,
    # so we clamp the input to `[0, num_classes - 1]`. The output will have
    # some invalid results, so we set them back to 0 using `where` afterwards.
    valid = ((x >= 0) & (x < num_classes)).unsqueeze(-1)
    output = tnn.one_hot(torch.clamp(x, 0, num_classes - 1), num_classes)
    output = torch.where(valid, output, zero)
    output = convert_to_tensor(output, dtype=dtype)
    dims = output.dim()
    axis = canonicalize_axis(axis, dims)
    if axis != dims - 1:
        new_axes_order = list(range(dims))
        new_axes_order[axis] = -1  # Shifts output to axis position
        # Shift remaining axes with offset by 1 since output moved to `axis`.
        for ax in range(axis + 1, dims):
            new_axes_order[ax] -= 1
        output = output.permute(new_axes_order)
    return output


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = torch.amax(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        dim=reduction_axis,
    )
    return outputs


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

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
        log_prob = tnn.log_softmax(output, dim=axis)
    else:
        output = output / torch.sum(output, dim=axis, keepdim=True)
        output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = torch.log(output)
    return -torch.sum(target * log_prob, dim=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype=torch.long)
    output = convert_to_tensor(output)

    if len(target.shape) == len(output.shape) and target.shape[axis] == 1:
        target = torch.squeeze(target, dim=axis)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    output_shape_without_class_dim = list(output.shape)
    del output_shape_without_class_dim[axis]

    if list(target.shape) != output_shape_without_class_dim:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    # Use PyTorch native cross-entropy ops to avoid allocating a full
    # one-hot matrix of shape (batch, ..., num_classes).  For large
    # vocabularies this saves gigabytes of GPU memory per step.
    # F.cross_entropy / F.nll_loss expect the class dim at position 1,
    if output.dim() == 1:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
        class_axis = axis % output.dim()
        if class_axis != 1:
            output = output.movedim(class_axis, 1)

    if from_logits:
        result = tnn.cross_entropy(output, target, reduction="none")
    else:
        output = output / torch.sum(output, dim=1, keepdim=True)
        output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = torch.log(output)
        result = tnn.nll_loss(log_prob, target, reduction="none")

    if squeeze:
        result = result.squeeze(0)
    return result


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    if not is_float_dtype(output.dtype):
        output = cast(output, backend.floatx())
    target = cast(target, output.dtype)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    # By default, PyTorch, does reduction of `sum` over all rows,
    # change reduction to `none` to keep dim
    if from_logits:
        return tnn.binary_cross_entropy_with_logits(
            output, target, reduction="none"
        )

    output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    # Before torch 2.10, the MPS `binary_cross_entropy` kernel squeezes every
    # size-1 dimension of its inputs but not of `grad_output`, so the backward
    # pass aborts or returns wrong gradients. Computing the loss on 1-D
    # tensors avoids this on every device.
    # See https://github.com/pytorch/pytorch/issues/166746.
    loss = tnn.binary_cross_entropy(
        output.reshape(-1), target.reshape(-1), reduction="none"
    )
    return loss.reshape(output.shape)


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with PyTorch."
        )
    x = convert_to_tensor(x)
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = torch.mean(x, dim=axes, keepdim=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    # Note: stop_gradient does not change the gradient to the mean, because that
    # gradient is zero.
    variance = torch.mean(
        torch.square(x), dim=axes, keepdim=True
    ) - torch.square(mean)

    if not keepdims:
        mean = torch.squeeze(mean, axes)
        variance = torch.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = torch.clip(
            mean,
            torch.finfo(torch.float16).min,
            torch.finfo(torch.float16).max,
        )
        variance = torch.clip(
            variance,
            torch.finfo(torch.float16).min,
            torch.finfo(torch.float16).max,
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)

    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = torch.reshape(mean, shape)
    variance = torch.reshape(variance, shape)

    if offset is not None:
        offset = convert_to_tensor(offset)
        offset = torch.reshape(offset, shape)
    else:
        offset = torch.zeros_like(mean)
    if scale is not None:
        scale = convert_to_tensor(scale)
        scale = torch.reshape(scale, shape)
    else:
        scale = torch.ones_like(variance)

    return (
        x.subtract(mean)
        .mul_(variance.add(epsilon).rsqrt_().mul(scale))
        .add_(offset)
    )


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length)
    output_length = convert_to_tensor(output_length)

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(output.dtype, "float32")
    output = cast(output, dtype)

    output = torch.transpose(output, 1, 0)
    logits = tnn.log_softmax(output, dim=-1)
    loss = tnn.ctc_loss(
        logits,
        target,
        output_length,
        target_length,
        blank=mask_index,
        reduction="none",
    )
    return loss


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

    indices = torch.argmax(inputs, axis=-1)
    indices = cast(indices, "int32")
    scores = torch.max(inputs, axis=-1)[0]

    seqlen_mask = torch.arange(max_length, device=indices.device)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = torch.where(seqlen_mask, mask_index, indices)
    scores = torch.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat = indices[:, 1:] == indices[:, :-1]
        repeat = tnn.pad(repeat, (1, 0, 0, 0))
        indices = torch.where(repeat, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = torch.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = torch.unsqueeze(
        torch.arange(max_length, device=indices.device), dim=0
    )  # [1, N]
    order = torch.tile(order, (batch_size, 1))  # [B, N]
    order = torch.where(invalid_mask, max_length, order)
    order = torch.argsort(order, dim=-1)
    indices = torch.take_along_dim(indices, order, dim=-1)

    scores = -torch.sum(scores, axis=1)[:, None]
    indices = torch.unsqueeze(indices, dim=0)
    return indices, scores


# Knuth's multiplicative hash constant. Used as the polynomial base in
# `_unique_padded`; int64 overflow during exponentiation is intentional
# since we only need a deterministic path -> hash mapping.
_KNUTH_HASH_CONSTANT = 2654435769


def _unique_padded(paths, size, pad):
    """Hash-based row-dedup, padded to a fixed leading size with `pad` rows.

    Mirrors `jax.numpy.unique(..., size=size, fill_value=pad, axis=0,
    return_inverse=True)` in observable behavior: the unique rows come
    first, followed by `(size - n_unique)` rows filled with `pad`, and
    `inverse` maps each input row to its index in the unique output.

    Internally avoids `torch.unique(dim=0)` (which sorts the full row
    tensor every call and dominates beam-search runtime) by hashing each
    path to int64 and deduplicating along that 1D axis. Collisions are
    detected by an explicit row-equality check on adjacent same-hash
    entries, so correctness does not rely on a collision-free hash.
    """
    n, t = paths.shape
    device = paths.device

    # Polynomial hash. Shift values to non-negative so all-`pad` rows
    # don't collapse to zero. Adjacent same-hash rows are verified for
    # equality below, so collisions stay correct.
    p = paths.to(torch.int64) + 1
    powers = _KNUTH_HASH_CONSTANT ** torch.arange(
        t, dtype=torch.int64, device=device
    )
    hashes = (p * powers).sum(dim=1)

    order = torch.argsort(hashes, stable=True)
    sorted_paths = paths[order]
    sorted_hashes = hashes[order]

    adj_hash_eq = sorted_hashes[1:] == sorted_hashes[:-1]
    adj_row_eq = (sorted_paths[1:] == sorted_paths[:-1]).all(dim=1)
    is_dup = adj_hash_eq & adj_row_eq
    is_first = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=device), ~is_dup]
    )

    cum_first = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
    inverse = torch.empty(n, dtype=torch.int64, device=device)
    inverse[order] = cum_first

    unique = sorted_paths[is_first]
    n_unique = unique.shape[0]
    if n_unique < size:
        pad_rows = torch.full(
            (size - n_unique, t),
            pad,
            dtype=unique.dtype,
            device=device,
        )
        unique = torch.cat([unique, pad_rows], dim=0)
    return unique, inverse


def _merge_scores(inverse, scores, num_uniques):
    """Log-space scatter-add of `scores` into `num_uniques` buckets."""
    scores_max = scores.max()
    scores_exp = torch.exp(scores - scores_max)
    out = torch.zeros(num_uniques, dtype=scores.dtype, device=scores.device)
    out.scatter_add_(0, inverse, scores_exp)
    return torch.log(out) + scores_max


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    """Beam search CTC decoding for the torch backend.

    Direct port of `keras/src/backend/jax/nn.py::_ctc_beam_search_decode`.
    The semantics (tie-breaking, log-space score merging, blank/emit
    score tracks) match the JAX reference implementation; correctness is
    prioritized over throughput, so the batch dimension is iterated in
    Python rather than vmapped.
    """
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = tnn.log_softmax(inputs, dim=-1)

    if mask_index is None:
        mask_index = num_classes - 1

    # Tie-breaking: jnp.argsort doesn't accept an `order` parameter, so
    # the JAX impl flips classes so the desired ordering falls out of the
    # default ascending argsort. We mirror that here for parity.
    inputs = torch.flip(inputs, dims=[2])
    mask_index = num_classes - mask_index - 1

    _pad = -1
    device = inputs.device

    # Precompute per-batch seqlen masks on CPU so the inner loop avoids
    # a GPU->CPU sync on every timestep.
    seqlen_cpu = sequence_lengths.detach().cpu().tolist()
    num_init_paths = min(num_classes, beam_width)

    paths_per_batch = []
    scores_per_batch = []
    for b in range(batch_size):
        x = inputs[b]
        seq_len_b = int(seqlen_cpu[b])

        init_paths = torch.full(
            (2 * beam_width, max_seq_len),
            _pad,
            dtype=torch.int32,
            device=device,
        )

        max_classes = torch.argsort(x[0], stable=True)[-num_init_paths:]
        init_classes = torch.where(
            max_classes == mask_index, _pad, max_classes.to(torch.int32)
        )
        init_paths[:num_init_paths, 0] = init_classes

        init_scores = torch.full(
            (2 * beam_width,),
            float("-inf"),
            dtype=inputs.dtype,
            device=device,
        )
        init_scores[:num_init_paths] = x[0, max_classes]

        paths = init_paths
        scores = init_scores
        masked = paths[:, 0] == _pad

        # Only iterate timesteps within this sequence's length.
        for t in range(1, seq_len_b):
            paths, scores, masked = _ctc_beam_extend(
                paths, scores, masked, x[t], num_classes, mask_index, _pad
            )
            paths, scores, masked = _ctc_beam_prune(
                paths, scores, masked, num_classes, beam_width, _pad
            )

        # Final dedup + top_paths selection.
        paths_unique, inverse = _unique_padded(
            paths, size=2 * num_classes * beam_width, pad=_pad
        )
        scores = _merge_scores(inverse, scores, paths_unique.shape[0])

        top_indices = torch.argsort(scores, stable=True)[-top_paths:].flip(0)
        paths_per_batch.append(paths_unique[top_indices])
        scores_per_batch.append(scores[top_indices])

    paths = torch.stack(paths_per_batch, dim=0)
    scores = torch.stack(scores_per_batch, dim=0)

    # Convert classes back from the flipped representation.
    paths = torch.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = paths.permute(1, 0, 2)
    return paths, scores


def _ctc_beam_extend(paths, scores, masked, x, num_classes, mask_index, _pad):
    """Extend each beam with every possible class for a single timestep."""
    paths = paths.repeat_interleave(num_classes, dim=0)
    scores = scores.repeat_interleave(num_classes)
    masked = masked.repeat_interleave(num_classes)

    is_pad = paths == _pad
    path_tail_index = is_pad.to(torch.int32).argmax(dim=1)
    arange = torch.arange(paths.shape[0], device=paths.device)
    path_tails = paths[arange, path_tail_index - 1]
    path_tails = torch.where(path_tail_index == 0, _pad, path_tails)

    classes = torch.arange(num_classes, device=paths.device, dtype=paths.dtype)
    classes = classes.clone()
    classes[mask_index] = _pad
    classes = classes.repeat(paths.shape[0] // num_classes)

    prev_masked = masked
    masked = classes == _pad

    masked_repeat = (~prev_masked) & (path_tails == classes)
    classes = torch.where(masked_repeat, _pad, classes)
    paths[arange, path_tail_index] = classes

    scores = scores + x.repeat(paths.shape[0] // num_classes)
    return paths, scores, masked


def _ctc_beam_prune(paths, scores, masked, num_classes, beam_width, _pad):
    """Dedup + score-merge + keep top `beam_width` (emit, blank) tracks."""
    paths_unique, inverse = _unique_padded(
        paths, size=2 * num_classes * beam_width, pad=_pad
    )

    emit_scores = torch.where(masked, float("-inf"), scores)
    mask_scores = torch.where(masked, scores, float("-inf"))

    n_uniques = paths_unique.shape[0]
    emit_scores = _merge_scores(inverse, emit_scores, n_uniques)
    mask_scores = _merge_scores(inverse, mask_scores, n_uniques)

    total_scores = torch.logaddexp(emit_scores, mask_scores)
    top_indices = torch.argsort(total_scores, stable=True)[-beam_width:]

    paths_top = paths_unique[top_indices]
    emit_scores_top = emit_scores[top_indices]
    mask_scores_top = mask_scores[top_indices]

    paths = paths_top.repeat(2, 1)
    scores = torch.cat([emit_scores_top, mask_scores_top])
    masked_out = torch.cat(
        [
            torch.zeros(beam_width, dtype=torch.bool, device=paths.device),
            torch.ones(beam_width, dtype=torch.bool, device=paths.device),
        ]
    )
    return paths, scores, masked_out


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

    x1, x2 = (
        convert_to_tensor(x1),
        convert_to_tensor(x2),
    )
    max_val = convert_to_tensor(max_val, dtype=x1.dtype)
    mse = torch.mean((x1 - x2) ** 2)
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = standardize_dtype(dtype)
    if dtype == "float16":
        val = 65500.0
    else:
        val = 3.38953e38
    return convert_to_tensor(val * -0.7, dtype=dtype)


def _can_use_flash_attention(
    query, key, value, mask=None, is_causal=False, raise_error=False
):
    """Verify the availability of flash attention."""
    if (
        not raise_error
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    ):
        # The probe below constructs a pybind11 `SDPAParams` object, which
        # dynamo cannot trace, so it breaks the graph at every attention call.
        # Skipping it is safe: `scaled_dot_product_attention` still selects the
        # flash kernel when the inputs allow. Auto-detection only, so an
        # explicit `flash_attention=True` still reports an unsupported input.
        return False

    try:
        from torch.backends.cuda import SDPAParams
        from torch.backends.cuda import can_use_flash_attention
    except ImportError:
        if raise_error:
            raise ImportError(
                "Flash attention is not supported in your current PyTorch "
                "version. Please update it by following the official guide: "
                "https://pytorch.org/get-started/locally/"
            )
        return False

    try:
        spda_params = SDPAParams(
            query,
            key,
            value,
            mask,
            0.0,  # dropout_p
            is_causal,
            False,  # enable_gqa
        )
    except TypeError:
        # The old function signature for the older version of PyTorch
        spda_params = SDPAParams(
            query,
            key,
            value,
            mask,
            0.0,  # dropout_p
            is_causal,
        )
    if raise_error and can_use_flash_attention(spda_params, True) is False:
        raise RuntimeError(
            "Flash attention is not supported with the provided inputs. "
            "Please check the warnings for more details."
        )
    return can_use_flash_attention(spda_params, False)


def _causal_mask(query_length, kv_length, like):
    """Lower-triangular boolean mask placed like the tensor `like`."""
    causal_mask = torch.tril(
        torch.ones(
            (query_length, kv_length), dtype=torch.bool, device=like.device
        )
    )
    if isinstance(like, DTensor):
        causal_mask = DTensor.from_local(
            causal_mask,
            like.device_mesh,
            [Replicate()] * like.device_mesh.ndim,
        )
    return causal_mask


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
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    if bias is not None and mask is not None:
        raise ValueError(
            "Only one of `bias` and `mask` can be provided. Received both."
        )
    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = cast(query, compute_dtype)
    key = cast(key, compute_dtype)
    value = cast(value, compute_dtype)

    mask = mask if mask is None else convert_to_tensor(mask, dtype="bool")
    if mask is not None:
        if is_causal:
            # `scaled_dot_product_attention` treats `attn_mask` and
            # `is_causal` as mutually exclusive, so a caller-provided mask
            # would otherwise silently drop the causal constraint. Fold the
            # causal mask into the explicit mask to honor both.
            mask = torch.logical_and(
                mask, _causal_mask(query.shape[1], key.shape[1], mask)
            )
        # Explicitly set `is_causal` to `False` when `mask` is not `None`.
        is_causal = False
        mask = torch.where(mask, 0.0, _get_large_negative(query.dtype))
    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)
        if is_causal:
            # The same exclusivity applies to a float `attn_mask` (the bias).
            bias = torch.where(
                _causal_mask(query.shape[1], key.shape[1], bias),
                bias,
                _get_large_negative(bias.dtype).to(bias.device),
            )
            is_causal = False
        mask = bias  # Use `bias` as `mask` for scaled_dot_product_attention.

    axis0, axis1 = 1, 2
    query = torch.transpose(query, axis0, axis1)
    key = torch.transpose(key, axis0, axis1)
    value = torch.transpose(value, axis0, axis1)

    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads > num_kv_heads and num_kv_heads > 1:
        groups = num_query_heads // num_kv_heads
        key = torch.repeat_interleave(key, repeats=groups, dim=1)
        value = torch.repeat_interleave(value, repeats=groups, dim=1)

    is_dtensor = isinstance(query, DTensor)
    if is_dtensor:
        device_mesh = query.device_mesh
        placements = query.placements
        query = query.to_local()
        key = key.to_local() if hasattr(key, "to_local") else key
        value = value.to_local() if hasattr(value, "to_local") else value
        if mask is not None:
            mask = mask.to_local() if hasattr(mask, "to_local") else mask

    if flash_attention is None:
        flash_attention = _can_use_flash_attention(
            query, key, value, mask, is_causal
        )
    elif flash_attention is True:
        # Use `raise_error=True` to provide more details if the inputs failed to
        # use flash attention
        _can_use_flash_attention(
            query, key, value, mask, is_causal, raise_error=True
        )
    if flash_attention:
        with torch.nn.attention.sdpa_kernel(
            backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION],
        ):
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=is_causal,
                scale=scale,
            )
    else:
        if mask is not None:
            mask = mask.contiguous()
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=mask,
            is_causal=is_causal,
            scale=scale,
        )

    if is_dtensor:
        attention_output = DTensor.from_local(
            attention_output, device_mesh, placements
        )

    return torch.transpose(attention_output, axis1, axis0)


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Native PyTorch implementation of Unfold.
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
    return tnn.unfold(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Native PyTorch implementation of Fold.
    Combine an array of sliding local blocks into a large tensor (col2im).

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
    return tnn.fold(
        x,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


def depth_to_space(x, block_size, data_format="channels_last"):
    """PyTorch implementation of depth_to_space.

    Rearranges data from depth into blocks of spatial data.
    Matches TensorFlow's depth_to_space behavior.

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
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, H, W, C) -> (N, H, W, block_size, block_size, new_C)
        x = x.reshape(n, h, w, block_size, block_size, new_c)
        # Permute to (N, H, bH, W, bW, new_C) to interleave spatial blocks.
        x = x.permute(0, 1, 3, 2, 4, 5)
        # Reshape to the final spatial dimensions.
        x = x.reshape(n, h * block_size, w * block_size, new_c)
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, C, H, W) -> (N, new_C, block_size, block_size, H, W)
        x = x.reshape(n, new_c, block_size, block_size, h, w)
        # Permute: (N, C, bH, bW, H, W) -> (N, C, H, bH, W, bW)
        x = x.permute(0, 1, 4, 2, 5, 3)
        # Reshape: (N, C, H, bH, W, bW) -> (N, C, H*bH, W*bW)
        x = x.reshape(n, new_c, h * block_size, w * block_size)
    return x


def space_to_depth(x, block_size, data_format="channels_last"):
    """PyTorch implementation of space_to_depth.

    Rearranges blocks of spatial data into depth.
    Matches TensorFlow's space_to_depth behavior.

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
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, H, W, C) -> (N, new_H, bH, new_W, bW, C)
        x = x.reshape(n, new_h, block_size, new_w, block_size, c)
        # Permute: (N, new_H, bH, new_W, bW, C) -> (N, new_H, new_W, bH, bW, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        # Reshape: (N, new_H, new_W, bH, bW, C) -> (N, new_H, new_W, C*bH*bW)
        x = x.reshape(n, new_h, new_w, c * block_size**2)
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, C, H, W) -> (N, C, new_H, bH, new_W, bW)
        x = x.reshape(n, c, new_h, block_size, new_w, block_size)
        # Permute: (N, C, new_H, bH, new_W, bW) -> (N, C, bH, bW, new_H, new_W)
        x = x.permute(0, 1, 3, 5, 2, 4)
        # Reshape: (N, C, bH, bW, new_H, new_W) -> (N, C*bH*bW, new_H, new_W)
        x = x.reshape(n, c * block_size**2, new_h, new_w)
    return x


def _canonical_axes(x, axis):
    if isinstance(axis, int):
        axis = [axis]
    return sorted(canonicalize_axis(a, x.dim()) for a in axis)


def _normalization_operands(x, weights, axis):
    """Prepare `x` and `weights` for the torch normalization kernels.

    `tnn.rms_norm` and `tnn.layer_norm` normalize over the trailing axes of
    the input and take weights shaped like those axes, so the axes are moved
    to the end here and the caller moves the output back. Returns None when
    a weight is not shaped like the axes it scales, in which case the caller
    composes the normalization from elementary ops instead.
    """
    normalized_shape = tuple(x.shape[a] for a in axis)
    size = math.prod(normalized_shape)
    reshaped = []
    for weight in weights:
        if weight is None:
            reshaped.append(None)
        elif weight.numel() == size:
            reshaped.append(weight.reshape(normalized_shape))
        else:
            return None
    kept = [d for d in range(x.dim()) if d not in axis]
    perm = kept + list(axis)
    return x.permute(perm), normalized_shape, reshaped, perm


def _inverse_permutation(perm):
    return [perm.index(d) for d in range(len(perm))]


def rms_normalization(x, scale=None, axis=-1, epsilon=None):
    if epsilon is None:
        epsilon = backend.epsilon()
    if x.dim() == 0:
        # A scalar is normalized as a single element, like the composed op.
        x = x.unsqueeze(0)
    axis = _canonical_axes(x, axis)
    operands = _normalization_operands(x, (scale,), axis) if axis else None
    if operands is None:
        rrms = torch.rsqrt(
            torch.mean(torch.square(x), dim=axis, keepdim=True) + epsilon
        )
        outputs = x * rrms
        if scale is not None:
            outputs = outputs * scale
        return outputs
    x, normalized_shape, (scale,), perm = operands
    outputs = tnn.rms_norm(x, normalized_shape, scale, epsilon)
    return outputs.permute(_inverse_permutation(perm))


def layer_normalization(x, gamma=None, beta=None, axis=-1, epsilon=None):
    if epsilon is None:
        epsilon = backend.epsilon()
    axis = _canonical_axes(x, axis)
    operands = _normalization_operands(x, (gamma, beta), axis) if axis else None
    if operands is None:
        mean = torch.mean(x, dim=axis, keepdim=True)
        variance = torch.var(x, dim=axis, keepdim=True, unbiased=False)
        inv = torch.rsqrt(variance + epsilon)
        if gamma is not None:
            inv = inv * gamma
        # Center before scaling; see the note in keras/src/ops/nn.py.
        # Folding the mean and beta into one offset loses beta when
        # `mean * inv` is large enough that adding beta to it rounds away.
        outputs = (x - mean) * inv
        if beta is not None:
            outputs = outputs + beta
        return outputs
    x, normalized_shape, (gamma, beta), perm = operands
    outputs = tnn.layer_norm(x, normalized_shape, gamma, beta, epsilon)
    return outputs.permute(_inverse_permutation(perm))
