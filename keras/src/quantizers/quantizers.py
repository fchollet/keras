import math

import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy


@keras_export(["keras.Quantizer", "keras.quantizers.Quantizer"])
class Quantizer:
    def __init__(self, output_dtype="int8"):
        self.output_dtype = output_dtype

    def __call__(self, x):
        """Compute a quantized output from an input tensor."""
        return x

    @classmethod
    def from_config(cls, config):
        """Creates a quantizer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same quantizer from the config
        dictionary.

        This method is used by Keras `model_to_estimator`, saving and
        loading models to HDF5 formats, Keras model cloning, some visualization
        utilities, and exporting models to and from JSON.

        Args:
            config: A Python dictionary, typically the output of get_config.

        Returns:
            A quantizer instance.
        """
        return cls(**config)

    def get_config(self):
        """Returns the config of the quantizer.

        A quantizer config is a Python dictionary (serializable)
        containing all configuration parameters of the quantizer.
        The same quantizer can be reinstantiated later
        (without any saved state) from this configuration.

        This method is optional if you are just training and executing models,
        exporting to and from SavedModels, or using weight checkpoints.

        This method is required for Keras `model_to_estimator`, saving and
        loading models to HDF5 formats, Keras model cloning, some visualization
        utilities, and exporting models to and from JSON.

        Returns:
            Python dictionary.
        """
        raise NotImplementedError(f"{self} does not implement get_config()")


@keras_export("keras.quantizers.abs_max_quantize")
def abs_max_quantize(
    inputs,
    axis,
    value_range=(-127, 127),
    dtype="int8",
    epsilon=backend.epsilon(),
    to_numpy=False,
):
    """
    Quantizes the input tensor using the absolute maximum quantization scheme.

    Args:
        inputs: Input tensor to quantize.
        axis: Axis along which to compute the quantization range.
        value_range: Tuple of the minimum and maximum values of the quantization
            range.
        dtype: Data type of the quantized output.
        epsilon: Small value to avoid division by zero.
        to_numpy: Whether to perform the quantization in numpy. This performs
            the computation on the host CPU and can be useful for saving memory
            on the device. If False, the computation is performed on the device.

    Returns:
        A tuple of the quantized tensor and the scale.
    """
    if to_numpy:
        # Save memory on the device using numpy
        original_dtype = standardize_dtype(inputs.dtype)
        inputs = ops.convert_to_numpy(inputs)
        axis = standardize_axis_for_numpy(axis)
        scale = np.divide(
            value_range[1],
            np.add(np.max(np.abs(inputs), axis=axis, keepdims=True), epsilon),
        )
        outputs = np.multiply(inputs, scale)
        outputs = np.clip(np.round(outputs), value_range[0], value_range[1])
        outputs = outputs.astype(dtype)
        return ops.convert_to_tensor(outputs), ops.convert_to_tensor(
            scale, dtype=original_dtype
        )

    inputs = ops.convert_to_tensor(inputs)
    scale = ops.divide(
        value_range[1],
        ops.add(ops.max(ops.abs(inputs), axis=axis, keepdims=True), epsilon),
    )
    scale = ops.cast(scale, standardize_dtype(inputs.dtype))
    outputs = ops.multiply(inputs, scale)
    outputs = ops.clip(ops.round(outputs), value_range[0], value_range[1])
    outputs = ops.cast(outputs, dtype)
    return outputs, scale


@keras_export("keras.quantizers.abs_max_quantize_grouped_with_zero_point")
def abs_max_quantize_grouped_with_zero_point(
    inputs,
    block_size,
    value_range=(-8, 7),
    dtype="int8",
    epsilon=backend.epsilon(),
    to_numpy=False,
):
    """Quantizes a 2D tensor using grouped asymmetric quantization with
    zero point.

    Groups are formed along axis 0 (the input/contracting dimension).
    Each group of `block_size` rows gets its own scale factor and zero point
    per column. This is useful for weight distributions that are not centered
    around zero. A group's range always includes zero, so its zero point is
    representable in `value_range` and the grid holds an exact zero.

    Args:
        inputs: Input tensor to quantize. Shape: `(input_dim, output_dim)`.
        block_size: Number of elements per group along axis 0.
        value_range: Tuple of `(min, max)` quantization range.
        dtype: Data type of quantized output.
        epsilon: Small value to avoid division by zero.
        to_numpy: Whether to compute in NumPy, which keeps the weight off
            the accelerator during quantization, rather than in backend
            ops. Both paths apply the same formula; `bfloat16` inputs can
            differ between them at the ulp level.

    Returns:
        A tuple `(quantized_tensor, scale, zero_point)` where:
            - `quantized_tensor`: Same shape as inputs, dtype=`dtype`.
            - `scale`: Shape `(n_groups, output_dim)` where
              `n_groups = ceil(input_dim / block_size)`.
            - `zero_point`: Shape `(n_groups, output_dim)`, dtype=`int8`.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import abs_max_quantize_grouped_with_zero_point
    >>> kernel = np.random.randn(512, 256).astype("float32")
    >>> quantized, scale, zero_point = abs_max_quantize_grouped_with_zero_point(
    ...     kernel, block_size=128, value_range=(-8, 7)
    ... )
    >>> quantized.shape
    (512, 256)
    >>> scale.shape  # 512 / 128 = 4 groups
    (4, 256)
    >>> zero_point.shape
    (4, 256)
    ```
    """
    if to_numpy:
        return _abs_max_quantize_grouped_with_zero_point_numpy(
            inputs, block_size, value_range, dtype, epsilon
        )
    return _abs_max_quantize_grouped_with_zero_point_tensor(
        inputs, block_size, value_range, dtype, epsilon
    )


def _abs_max_quantize_grouped_with_zero_point_numpy(
    inputs, block_size, value_range, dtype, epsilon
):
    """NumPy implementation of grouped asymmetric quantization.

    Uses NumPy for computation to reduce GPU memory usage during
    model quantization.
    """
    original_dtype = standardize_dtype(inputs.dtype)
    inputs = ops.convert_to_numpy(inputs)

    input_dim, output_dim = inputs.shape
    n_groups = math.ceil(input_dim / block_size)
    qmin, qmax = value_range

    # Zero-pad rows so input_dim is divisible by block_size
    padded_input_dim = n_groups * block_size
    if padded_input_dim > input_dim:
        padding = np.zeros(
            (padded_input_dim - input_dim, output_dim), dtype=inputs.dtype
        )
        inputs_padded = np.concatenate([inputs, padding], axis=0)
    else:
        inputs_padded = inputs

    inputs_reshaped = inputs_padded.reshape(n_groups, block_size, output_dim)

    # Per-group min/max, widened to include zero: the zero point then lands
    # inside `[qmin, qmax]` and the grid represents 0 exactly. A group whose
    # values are all one sign would otherwise clip to one end of the range.
    min_val = np.minimum(np.min(inputs_reshaped, axis=1, keepdims=True), 0.0)
    max_val = np.maximum(np.max(inputs_reshaped, axis=1, keepdims=True), 0.0)

    # Scale maps the [min, max] range to [qmin, qmax]; the floor keeps an
    # all-zero group finite when `epsilon` underflows in the input dtype.
    scale = np.divide(np.subtract(max_val, min_val) + epsilon, qmax - qmin)
    scale = np.maximum(scale, ml_dtypes.finfo(scale.dtype).tiny)

    # Zero point shifts the quantized range to include the original zero
    zero_point = np.round(np.divide(-min_val, scale)) + qmin
    zero_point = np.clip(zero_point, qmin, qmax)

    # Quantize: q = round(input / scale) + zero_point
    outputs = np.round(np.divide(inputs_reshaped, scale)) + zero_point
    outputs = np.clip(outputs, qmin, qmax)
    outputs = outputs.astype(dtype)

    # Remove padding and squeeze to (n_groups, output_dim)
    outputs = outputs.reshape(padded_input_dim, output_dim)[:input_dim, :]
    scale = np.squeeze(scale, axis=1)
    zero_point = np.squeeze(zero_point, axis=1).astype("int8")

    return (
        ops.convert_to_tensor(outputs),
        ops.convert_to_tensor(scale, dtype=original_dtype),
        ops.convert_to_tensor(zero_point),
    )


def _abs_max_quantize_grouped_with_zero_point_tensor(
    inputs, block_size, value_range, dtype, epsilon
):
    """Backend-ops implementation of grouped asymmetric quantization.

    The same formula as `_abs_max_quantize_grouped_with_zero_point_numpy`,
    for callers that keep the weight on the accelerator.
    """
    original_dtype = standardize_dtype(inputs.dtype)
    inputs = ops.convert_to_tensor(inputs)

    input_shape = ops.shape(inputs)
    input_dim, output_dim = int(input_shape[0]), int(input_shape[1])
    n_groups = math.ceil(input_dim / block_size)
    qmin, qmax = value_range

    # Zero-pad rows so input_dim is divisible by block_size
    padded_input_dim = n_groups * block_size
    if padded_input_dim > input_dim:
        padding = ops.zeros(
            (padded_input_dim - input_dim, output_dim), dtype=inputs.dtype
        )
        inputs = ops.concatenate([inputs, padding], axis=0)

    inputs_reshaped = ops.reshape(inputs, (n_groups, block_size, output_dim))

    # Per-group min/max, widened to include zero (see the NumPy path).
    min_val = ops.minimum(ops.min(inputs_reshaped, axis=1, keepdims=True), 0.0)
    max_val = ops.maximum(ops.max(inputs_reshaped, axis=1, keepdims=True), 0.0)

    # Scale maps the [min, max] range to [qmin, qmax]; the floor keeps an
    # all-zero group finite when `epsilon` underflows in the input dtype.
    scale = ops.divide(
        ops.add(ops.subtract(max_val, min_val), epsilon), qmax - qmin
    )
    scale = ops.maximum(scale, float(ml_dtypes.finfo(original_dtype).tiny))

    # Zero point shifts the quantized range to include the original zero
    zero_point = ops.add(
        ops.round(ops.divide(ops.negative(min_val), scale)), qmin
    )
    zero_point = ops.clip(zero_point, qmin, qmax)

    # Quantize: q = round(input / scale) + zero_point
    outputs = ops.add(ops.round(ops.divide(inputs_reshaped, scale)), zero_point)
    outputs = ops.cast(ops.clip(outputs, qmin, qmax), dtype)

    # Remove padding and squeeze to (n_groups, output_dim)
    outputs = ops.reshape(outputs, (padded_input_dim, output_dim))
    outputs = outputs[:input_dim, :]
    scale = ops.cast(ops.squeeze(scale, axis=1), original_dtype)
    zero_point = ops.cast(ops.squeeze(zero_point, axis=1), "int8")

    return outputs, scale, zero_point


@keras_export("keras.quantizers.AbsMaxQuantizer")
class AbsMaxQuantizer(Quantizer):
    def __init__(
        self,
        axis=None,  # Deprecated, provide axis in __call__ instead.
        value_range=(-127, 127),
        epsilon=backend.epsilon(),
        output_dtype="int8",
    ):
        Quantizer.__init__(self, output_dtype=output_dtype)
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            self.axis = tuple(axis)
        else:
            self.axis = None
        self.value_range = value_range
        self.epsilon = epsilon
        if output_dtype == "int8":
            if value_range[0] < -128 or value_range[1] > 127:
                raise ValueError(
                    f"Quantizer with output_dtype='int8' requires value_range "
                    f"to be within the interval [-128, 127]. Received: "
                    f"value_range={value_range}"
                )

    def __call__(self, x, axis=None, to_numpy=False):
        """
        Quantizes the input tensor.

        Args:
            x: Input tensor to quantize.
            axis: Axis along which to compute the quantization range. If None,
                uses the axis specified in the constructor. If None and no axis
                was specified in the constructor, defaults to -1.
            to_numpy: Whether to perform the quantization in numpy. This
                performs the computation on the host CPU and can be useful for
                saving memory on the device. If False, the computation is
                performed on the device.

        Returns:
            A tuple of the quantized tensor and the scale.
        """
        if axis is None:
            axis = self.axis
        if axis is None:
            # Default to -1 if no axis is specified
            axis = -1
        quantized_x, scale = abs_max_quantize(
            x,
            axis,
            self.value_range,
            self.output_dtype,
            self.epsilon,
            to_numpy,
        )
        return quantized_x, scale

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "epsilon": self.epsilon,
            "output_dtype": self.output_dtype,
        }
        if self.axis is not None:
            config["axis"] = self.axis
        return config


def compute_quantization_parameters(
    x,
    *,
    bits,
    symmetric=False,
    per_channel=False,
    group_size=-1,
    compute_dtype="float32",
    epsilon=0.0,
):
    """
    Computes the scale and zero-point for quantizing weight tensors.

    This function calculates the scale and zero-point required for quantizing
    a given weight tensor `x` based on the specified parameters. It supports
    grouped, per-channel, per-tensor, symmetric, and asymmetric quantization.

    For grouped quantization (per_channel=True, group_size > 0), the output
    shapes are [out_features, n_groups] where n_groups is the number of groups
    along the in_features dimension.

    Args:
        x: KerasTensor. The weight tensor to quantize with shape
            [out_features, in_features].
        bits: int. The number of bits to quantize to (e.g., 4).
        symmetric: bool. Whether to use symmetric quantization.
        per_channel: bool. Whether to quantize per channel.
        group_size: int. The group size for quantization. -1 means no grouping.
        compute_dtype: str. The dtype for computation. Defaults to "float32".
        epsilon: float. Small value added to (max - min) before computing
            scale to avoid division by zero. Defaults to 0.0.

    Returns:
        scale: KerasTensor. The scale tensor for quantization.
        zero: KerasTensor. The `uint8` zero tensor for quantization.
        maxq: scalar. The maximum quantization value.
    """
    # Input validation
    if x is None:
        raise ValueError(f"Input tensor {x} cannot be None.")
    if len(x.shape) < 2:
        raise ValueError(
            f"Input weight tensor {x} must have a rank of at "
            f"least 2, but got rank {len(x.shape)}."
        )
    if ops.size(x) == 0:
        raise ValueError("Input tensor 'x' cannot be empty.")

    out_features, in_features = x.shape[0], x.shape[1]

    # Determine number of groups for quantization
    if per_channel and group_size > 0:
        n_groups = (in_features + group_size - 1) // group_size
    else:
        n_groups = 1

    # Compute min/max values based on quantization mode
    if n_groups > 1:
        # Grouped quantization: output shape [out_features, n_groups]
        remainder = in_features % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            x = ops.pad(x, [[0, 0], [0, pad_size]], constant_values=0.0)

        x_grouped = ops.reshape(x, [out_features, n_groups, group_size])
        min_values = ops.min(x_grouped, axis=2)
        max_values = ops.max(x_grouped, axis=2)
    else:
        # Per-channel or per-tensor: compute stats along rows
        reduction_shape = [out_features, -1] if per_channel else [1, -1]
        x_reshaped = ops.reshape(x, reduction_shape)
        min_values = ops.min(x_reshaped, axis=1)
        max_values = ops.max(x_reshaped, axis=1)

    # Asymmetric quantization: clamp the range to include zero, matching
    # reference GPTQ/AWQ (`xmin = min(xmin, 0)`, `xmax = max(xmax, 0)`).
    # This guarantees the zero point lands in `[0, maxq]`, so it is
    # representable in `bits`-bit packed formats, and that the quantized
    # grid can represent 0 exactly, even for groups whose values are
    # all-negative or all-positive.
    if not symmetric:
        min_values = ops.minimum(min_values, 0.0)
        max_values = ops.maximum(max_values, 0.0)

    # Symmetric quantization: make range symmetric around zero
    if symmetric:
        max_abs = ops.maximum(ops.abs(min_values), max_values)
        min_values = ops.where(
            ops.less(min_values, 0), ops.negative(max_abs), min_values
        )
        max_values = max_abs

    # Ensure non-zero range to avoid division errors
    zero_range = ops.equal(min_values, max_values)
    min_values = ops.where(zero_range, ops.subtract(min_values, 1), min_values)
    max_values = ops.where(zero_range, ops.add(max_values, 1), max_values)

    # Compute scale and zero-point
    maxq = ops.cast(ops.subtract(ops.power(2, bits), 1), compute_dtype)
    range_values = ops.subtract(max_values, min_values)
    if epsilon > 0:
        range_values = ops.add(range_values, epsilon)
    scale = ops.divide(range_values, maxq)
    scale = ops.where(ops.less_equal(scale, 0), 1e-8, scale)

    # Zero point in the unsigned range [0, 2^bits-1], e.g., [0, 15] for 4-bit
    if symmetric:
        zero = ops.full_like(scale, ops.divide(ops.add(maxq, 1), 2))
    else:
        zero = ops.round(ops.divide(ops.negative(min_values), scale))
    zero = ops.clip(zero, 0, maxq)

    # Reshape output to [out_features, n_groups] or [out_features, 1]
    if n_groups > 1:
        pass  # Already [out_features, n_groups]
    elif per_channel:
        scale = ops.reshape(scale, [-1, 1])
        zero = ops.reshape(zero, [-1, 1])
    else:
        # Per-tensor: tile single value to [out_features, 1]
        scale = ops.tile(ops.reshape(scale, (1, 1)), (out_features, 1))
        zero = ops.tile(ops.reshape(zero, (1, 1)), (out_features, 1))

    return scale, ops.cast(zero, "uint8"), maxq


def quantize_with_zero_point(input_tensor, scale, zero, maxq):
    """Quantize a float tensor into discrete levels [0, maxq] using
    per-tensor/per-channel/grouped scaling.

    Returns `q` (same dtype as inputs/scales; float is fine) where values are in
    [0, maxq].

    Args:
        input_tensor: KerasTensor. The input tensor to quantize.
        scale: KerasTensor. The scale tensor for quantization.
        zero: KerasTensor. The zero tensor for quantization.
        maxq: KerasTensor. The maximum quantization value.

    Returns:
        KerasTensor. The quantized tensor.
    """
    # Guard against divide-by-zero
    epsilon = ops.cast(1e-8, dtype=scale.dtype)
    safe_scale = ops.where(ops.equal(scale, 0), epsilon, scale)

    quantized_tensor = ops.round(
        ops.add(
            ops.divide(input_tensor, safe_scale), ops.cast(zero, scale.dtype)
        )
    )
    quantized_tensor = ops.clip(quantized_tensor, 0, maxq)
    return quantized_tensor


def dequantize_with_zero_point(input_tensor, scale, zero):
    """
    Dequantizes a quantized tensor using the provided scale and zero tensors.

    Args:
        input_tensor: KerasTensor. The quantized tensor to dequantize.
        scale: KerasTensor. The scale tensor for dequantization.
        zero: KerasTensor. The zero tensor for dequantization.

    Returns:
        KerasTensor. The dequantized tensor.
    """
    return ops.multiply(
        scale, ops.subtract(input_tensor, ops.cast(zero, scale.dtype))
    )


def _take_group_params(scale, zero, g_idx, group_axis):
    """Gathers each position's group scale and zero point.

    `g_idx` is a 1-D integer tensor with one entry per position along the
    quantized dimension, naming that position's group (`0` to
    `n_groups - 1`; with 128 columns and `group_size=32` it is
    `[0] * 32 + [1] * 32 + [2] * 32 + [3] * 32`). `group_axis` is the axis
    of `scale` and `zero` that holds the per-group values. The gathered
    zero point is cast to the scale's dtype.
    """
    groups = ops.cast(g_idx, "int32")
    scales = ops.take(scale, groups, axis=group_axis)
    zeros = ops.cast(ops.take(zero, groups, axis=group_axis), scales.dtype)
    return scales, zeros


def quantize_with_sz_map(
    weights_matrix, scale, zero, g_idx, maxq, group_axis=-1
):
    """Quantizes `weights_matrix` with per-group multiplier scales.

    See `_take_group_params` for `g_idx` and `group_axis`; `maxq` is the
    largest code, `2**bits - 1`.
    """
    scales, zeros = _take_group_params(scale, zero, g_idx, group_axis)
    return quantize_with_zero_point(weights_matrix, scales, zeros, maxq)


def dequantize_with_sz_map(weights_matrix, scale, zero, g_idx, group_axis=-1):
    """Dequantizes codes with per-group multiplier scales.

    The real value is `(code - zero) * scale`; see `_take_group_params`
    for `g_idx` and `group_axis`.
    """
    scales, zeros = _take_group_params(scale, zero, g_idx, group_axis)
    return dequantize_with_zero_point(weights_matrix, scales, zeros)


def ternarize(kernel, threshold=None):
    """Ternarizes `kernel` to `{-1, 0, +1}` codes with their scale.

    The BitNet b1.58 rule: a weight becomes `sign(w)` when `|w|` exceeds
    the threshold and 0 otherwise. With `threshold=None` the threshold is
    `0.5 * mean(|W|)` and the scale is `mean(|W|)`, so `codes * scale`
    carries the kernel's magnitude. The mean is taken in float32 whatever
    the kernel's dtype. With a fixed threshold the scale is 1.0.

    Args:
        kernel: The float weight to ternarize.
        threshold: Optional fixed, non-negative threshold on `|w|`.

    Returns:
        `(codes, scale)`: the codes as a NumPy array in the kernel's dtype
        and the scale as a Python float.
    """
    abs_kernel = ops.abs(ops.cast(kernel, "float32"))
    beta = ops.mean(abs_kernel)
    t = ops.multiply(beta, 0.5) if threshold is None else threshold
    codes = ops.multiply(
        ops.sign(kernel), ops.cast(ops.greater(abs_kernel, t), kernel.dtype)
    )
    scale = 1.0 if threshold is not None else ops.convert_to_numpy(beta)
    return ops.convert_to_numpy(codes), float(scale)
