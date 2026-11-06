"""int4 handlers for the projection family (`Dense`, `EinsumDense`)."""

import math

from keras.src import ops
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.int4.block_size import is_grouped
from keras.src.quantizers.modes.int4.block_size import is_per_channel
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)


class Int4ProjectionHandlers:
    """The int4 build, forward, encode and view of a projection kernel.

    The kernel is stored as 2-D `[rows, columns]`, a plain reshape of the
    kernel (`geometry.rows_columns`), so `rows` are the contracted axes
    only when those lead the kernel. The codes are packed two per byte
    along the columns, and the scale runs per column (per-channel) or per
    group of rows (grouped, with a zero point and a group index). The
    forward pass dequantizes through the `QuantizedWeight` view and
    contracts in float.
    """

    def _build_projection(self, layer, geometry, kernel_shape, config):
        geometry.prepare()
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        rows, columns = geometry.rows_columns(kernel_shape)
        block_size = self.resolve_block_size(layer, config)

        # Codes packed two per byte along the columns.
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(rows, Int4Pairs.packed_length(columns)),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        if is_per_channel(block_size):
            scale_shape = (columns,)
        else:
            scale_shape = (math.ceil(rows / block_size), columns)
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )
        if is_grouped(block_size):
            # Grouped quantization is asymmetric: a zero point per group and
            # the row-to-group index.
            def idx_initializer(shape, dtype):
                return ops.floor_divide(
                    ops.arange(rows, dtype=dtype), block_size
                )

            layer.kernel_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            # `g_idx` is stored as `float32` because TF has no GPU kernel for
            # int32 resource variables (would pin the variable to CPU and
            # break jit_compile on GPU); consumers cast to int32 on-device.
            # Not autocast: bfloat16 holds integers exactly only up to 256.
            layer.g_idx = layer.add_weight(
                name="g_idx",
                shape=(rows,),
                initializer=idx_initializer,
                dtype="float32",
                trainable=False,
                autocast=False,
            )
        else:
            layer.kernel_zero = None
            layer.g_idx = None

    def _get_projection_quantized_weight(self, layer, geometry):
        _, columns = geometry.rows_columns(geometry.weight_shape)
        return QuantizedWeight(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=Int4Pairs(axis=-1, orig_len=columns),
            scheme=int4_scheme(
                self.resolve_block_size(layer, layer.quantization_config)
            ),
            shape=geometry.weight_shape,
            axis=0,
            zero_point=layer.kernel_zero,
            g_idx=layer.g_idx,
        )

    def _call_projection(self, layer, inputs, training=None):
        geometry = layer._quantization_geometry()
        view = self._get_projection_quantized_weight(layer, geometry)

        @ops.custom_gradient
        def contract_with_inputs_gradient(inputs, *tensors):
            """Dequantizes the int4 kernel and contracts in float.

            `tensors` are the view's stored tensors: the codes and the scale,
            then the zero point and the group index of a grouped scheme.
            Autodiff cannot differentiate through the packed kernel, so the
            gradient with respect to the inputs is taken through the
            dequantized kernel.
            """
            quantized_weight = view.with_tensors(tensors)

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = quantized_weight.dequantize(layer.compute_dtype)
                inputs_grad = geometry.contract_grad(upstream, float_kernel)
                return (inputs_grad,) + (None,) * len(tensors)

            float_kernel = quantized_weight.dequantize(layer.compute_dtype)
            if layer.inputs_quantizer:
                inputs_q, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                x = geometry.contract(inputs_q, float_kernel)
                x = ops.cast(x, layer.compute_dtype)
                x = ops.divide(x, geometry.align_inputs_scale(inputs_scale))
            else:
                x = geometry.contract(inputs, float_kernel)
            return x, grad_fn

        # Read inside the autocast scope: on TensorFlow eager the gradient
        # runs after it, and the scale variable would then read float32.
        x = contract_with_inputs_gradient(inputs, *view.read_tensors())
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        geometry.prepare()
        # `Int4Strategy.resolve_block_size` is the single source of truth for
        # the group size, shared with the build path and the dtype-policy
        # naming, so the quantized values, the built variables, and the saved
        # policy string can never disagree. A bare `quantize("int4")` reaches
        # here with the canonical `Int4QuantizationConfig()` (grouped,
        # block_size=128); a `block_size` of `None` or `-1` selects the
        # per-channel escape hatch.
        block_size = self.resolve_block_size(layer, config)
        rows, columns = geometry.rows_columns(weight.shape)
        flat_kernel = ops.reshape(weight, (rows, columns))

        if is_per_channel(block_size):
            # Symmetric codes with one scale per column.
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=0, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            kernel_value_int4, kernel_scale = weight_quantizer(
                flat_kernel, to_numpy=True
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            # Asymmetric codes per group of rows: scale and zero point are
            # `[n_groups, columns]`.
            kernel_value_int4, kernel_scale, kernel_zero = (
                abs_max_quantize_grouped_with_zero_point(
                    flat_kernel, block_size=block_size, to_numpy=True
                )
            )

        # Pack two int4 values per int8 byte along the columns.
        packed_kernel_value, _, _ = pack_int4(kernel_value_int4, axis=-1)
        return packed_kernel_value, kernel_scale, kernel_zero

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, kernel_zero = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, "int4", config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)
        if kernel_zero is not None:
            layer.kernel_zero.assign(kernel_zero)
