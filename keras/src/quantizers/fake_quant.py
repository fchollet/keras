"""Fake quantization with learnable ranges, for quantization-aware training."""

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.ops.operation import Operation


def adjust_and_nudge(min_range, max_range, num_bits, narrow_range):
    """Adjusts and nudges the quantization range for better accuracy."""
    # Use higher precision for the computation.
    compute_dtype = backend.result_type(min_range.dtype, "float32")
    min_range = ops.cast(min_range, compute_dtype)
    max_range = ops.cast(max_range, compute_dtype)

    quant_max = (1 << num_bits) - 1
    quant_min = 0 if not narrow_range else 1
    diff_range = ops.subtract(max_range, min_range)

    # Calculate the scale and ensure it's positive
    scale = ops.divide(diff_range, quant_max - quant_min)

    # Re-calculate the inverse to avoid loss of precision
    inv_scale = ops.divide(quant_max - quant_min, diff_range)

    # Calculate the zero point from the min range
    zero_point_from_min = quant_min - ops.divide(min_range, scale)

    # Ensure zero point is within valid range [0, quant_max]
    zero_point = ops.clip(zero_point_from_min, quant_min, quant_max)

    # Nudge zero point if it's very close to an integer
    nudged_zero_point = ops.round(zero_point)

    # Calculate nudged limits
    nudged_min = ops.multiply(ops.subtract(quant_min, nudged_zero_point), scale)
    nudged_max = ops.multiply(ops.subtract(quant_max, nudged_zero_point), scale)

    return nudged_min, nudged_max, scale, inv_scale


class FakeQuantWithMinMaxVars(Operation):
    def __init__(self, num_bits=8, narrow_range=False, axis=None):
        super().__init__()
        self.num_bits = num_bits
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs, min_vals, max_vals):
        return fake_quant_with_min_max_vars(
            inputs,
            min_vals,
            max_vals,
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
            axis=self.axis,
        )

    def compute_output_spec(self, inputs, min_vals, max_vals):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.quantizers.fake_quant_with_min_max_vars")
def fake_quant_with_min_max_vars(
    inputs,
    min_vals,
    max_vals,
    num_bits=8,
    narrow_range=False,
    axis=None,
):
    """Perform per-tensor or per-channel fake quantization.

    `[min_vals, max_vals]` define the clamping range for the `inputs`.

    The `inputs` are quantized into the quantization range:
    - `[0, 2^num_bits - 1]` when `narrow_range=False`
    - `[1, 2^num_bits - 1]` when `narrow_range=True`

    After quantization, the values are dequantized and output as floats within
    the `[min_vals, max_vals]` interval.

    This operation supports gradient computation, allowing `min_vals` and
    `max_vals` to be trained.

    Args:
        inputs: Input Keras tensor of float dtype.
        min_vals: A global minimum scalar or a per-channel minimum tensor.
        max_vals: A global maximum scalar or a per-channel maximum tensor.
        num_bits: Quantization bit width (e.g., `8` for int8). Defaults to `8`.
        narrow_range: Whether to use narrow quantization range. Defaults to
            `False`.
        axis: Axis along which to perform per-channel quantization. If `None`,
              per-tensor quantization is performed. Defaults to `None`.


    Returns:
        Tensor: A Keras tensor with fake quantization applied.
    """
    if any_symbolic_tensors((inputs, min_vals, max_vals)):
        if axis is not None:
            # Validate the axis while the graph is built.
            axis = canonicalize_axis(axis, len(ops.shape(inputs)))
        return FakeQuantWithMinMaxVars(
            num_bits=num_bits, narrow_range=narrow_range, axis=axis
        ).symbolic_call(inputs, min_vals, max_vals)

    inputs = ops.convert_to_tensor(inputs)
    min_vals = ops.convert_to_tensor(min_vals)
    max_vals = ops.convert_to_tensor(max_vals)
    num_bits = int(num_bits)

    if axis is not None:
        axis = canonicalize_axis(axis, inputs.ndim)

    # Shortcut for TensorFlow backend by using `tf.quantization.fake_quant_*`
    # apis. This is necessary to be recognizable for the TFLite converter.
    if backend.backend() == "tensorflow":
        import tensorflow as tf

        # `tf.quantization.fake_quant_*` only supports float32.
        dtype = standardize_dtype(inputs.dtype)
        if axis is None:
            outputs = tf.quantization.fake_quant_with_min_max_vars(
                ops.cast(inputs, "float32"),
                ops.cast(ops.reshape(min_vals, ()), "float32"),
                ops.cast(ops.reshape(max_vals, ()), "float32"),
                num_bits=num_bits,
                narrow_range=narrow_range,
            )
            return ops.cast(outputs, dtype=dtype)
        else:
            # `tf.quantization.fake_quant_with_min_max_vars_per_channel` only
            # supports the last channel for the per-channel quantization. We
            # use `ops.swapaxes` for the pre- and post-processing.
            last_axis = inputs.ndim - 1
            inputs = ops.swapaxes(inputs, axis, last_axis)
            outputs = tf.quantization.fake_quant_with_min_max_vars_per_channel(
                ops.cast(inputs, "float32"),
                ops.cast(min_vals, "float32"),
                ops.cast(max_vals, "float32"),
                num_bits=num_bits,
                narrow_range=narrow_range,
            )
            outputs = ops.cast(outputs, dtype=dtype)
            return ops.swapaxes(outputs, last_axis, axis)

    @ops.custom_gradient
    def _fake_quant_with_min_max_vars_per_channel(x, min_val, max_val):
        dtype = standardize_dtype(x.dtype)

        # Calculate quantization parameters for all channels at once
        nudged_min, nudged_max, scale, inv_scale = adjust_and_nudge(
            min_val, max_val, num_bits, narrow_range
        )

        quant_zero = ops.floor(
            ops.add(ops.multiply(-nudged_min, inv_scale), 0.5)
        )
        x_clamped = ops.clip(
            ops.cast(x, nudged_min.dtype), nudged_min, nudged_max
        )
        x_clamped_shifted = ops.subtract(x_clamped, nudged_min)
        result = ops.multiply(
            ops.floor(
                ops.add(
                    ops.subtract(
                        ops.multiply(x_clamped_shifted, inv_scale), quant_zero
                    ),
                    0.5,
                )
            ),
            scale,
        )
        result = ops.cast(result, dtype=dtype)

        # Create gradient mask for all channels
        masks = ops.logical_and(
            ops.greater_equal(x, nudged_min), ops.less_equal(x, nudged_max)
        )

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args

            # Gradient for x
            dx = ops.where(masks, upstream, 0.0)
            axes = [i for i in range(len(dx.shape)) if i != axis]

            # Gradient for min_val
            # When x is clipped to min, the gradient flows to min_val
            min_mask = ops.less_equal(x, nudged_min)
            grad_min = ops.where(min_mask, upstream, 0.0)
            if axis is not None:
                grad_min = ops.sum(grad_min, axis=axes)
            else:
                grad_min = ops.sum(grad_min)
            grad_min = ops.reshape(grad_min, ops.shape(min_val))

            # Gradient for max_val
            # When x is clipped to max, the gradient flows to max_val
            max_mask = ops.greater_equal(x, nudged_max)
            grad_max = ops.where(max_mask, upstream, 0.0)
            if axis is not None:
                grad_max = ops.sum(grad_max, axis=axes)
            else:
                grad_max = ops.sum(grad_max)
            grad_max = ops.reshape(grad_max, ops.shape(max_val))

            return dx, grad_min, grad_max

        return result, grad

    return _fake_quant_with_min_max_vars_per_channel(inputs, min_vals, max_vals)
