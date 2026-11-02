import ml_dtypes

from keras.src import ops
from keras.src.dtype_policies.dtype_policy import QuantizedFloat8DTypePolicy
from keras.src.quantizers.float8_scaling import compute_float8_amax_history
from keras.src.quantizers.float8_scaling import compute_float8_scale
from keras.src.quantizers.float8_scaling import quantize_and_dequantize
from keras.src.quantizers.quantization_config import Float8QuantizationConfig
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class Float8Strategy(QuantizationStrategy):
    """Float8 QDQ mixed-precision training.

    Quantizing only allocates the scale/amax-history variables; the float
    kernel is kept and the fp8 casts happen dynamically during training.
    """

    name = "float8"
    config_cls = Float8QuantizationConfig
    # The float kernel is kept; only auxiliary variables are added.
    owns_weight_storage = False
    geometry_families = ("projection",)
    # The fp8 forward has no term for a LoRA update.
    supports_lora = False

    def policy_from_string(self, mode_str, source_name):
        return QuantizedFloat8DTypePolicy(mode_str, source_name)

    def quantize(self, layer, config):
        # The quantized values arrive later, so this only allocates the
        # mode's variables from the layer's current weight shape.
        geometry = self.require_geometry(layer)
        layer.quantized_build(geometry.weight_shape, self.name, config)

    def build(self, layer, input_shape, config):
        # The scale/amax variables are shape-independent and the float
        # kernel is left in place.
        del input_shape, config
        self.require_geometry(layer)
        # A float layer being quantized has no float8 policy yet and uses
        # the default length.
        policy = layer._own_dtype_policy
        amax_history_length = (
            policy.amax_history_length
            if isinstance(policy, QuantizedFloat8DTypePolicy)
            else QuantizedFloat8DTypePolicy.default_amax_history_length
        )
        # We set `trainable=True` because we will use the gradients to
        # overwrite these variables
        scale_kwargs = {
            "shape": (),
            "initializer": "ones",
            "dtype": "float32",  # Always be float32
            "trainable": True,
            "autocast": False,
            "overwrite_with_gradient": True,
        }
        amax_history_kwargs = {
            "shape": (amax_history_length,),
            "initializer": "zeros",
            "dtype": "float32",  # Always be float32
            "trainable": True,
            "autocast": False,
            "overwrite_with_gradient": True,
        }
        layer.inputs_scale = layer.add_weight(
            name="inputs_scale", **scale_kwargs
        )
        layer.inputs_amax_history = layer.add_weight(
            name="inputs_amax_history", **amax_history_kwargs
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale", **scale_kwargs
        )
        layer.kernel_amax_history = layer.add_weight(
            name="kernel_amax_history", **amax_history_kwargs
        )
        layer.outputs_grad_scale = layer.add_weight(
            name="outputs_grad_scale", **scale_kwargs
        )
        layer.outputs_grad_amax_history = layer.add_weight(
            name="outputs_grad_amax_history", **amax_history_kwargs
        )

    def call(self, layer, inputs, training=None):
        geometry = self.require_geometry(layer)
        if layer.lora_enabled:
            raise NotImplementedError(
                "Currently, float8 quantization doesn't support LoRA"
            )

        @ops.custom_gradient
        def quantized_dequantize_inputs(inputs, scale, amax_history):
            if training:
                new_scale = compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
                    ),
                )
                new_amax_history = compute_float8_amax_history(
                    inputs, amax_history
                )
            else:
                new_scale = None
                new_amax_history = None
            qdq_inputs = quantize_and_dequantize(
                inputs, scale, "float8_e4m3fn", layer.compute_dtype
            )

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                return upstream, new_scale, new_amax_history

            return qdq_inputs, grad

        @ops.custom_gradient
        def quantized_dequantize_outputs(outputs, scale, amax_history):
            """Quantize-dequantize the output gradient but not the output."""

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                new_scale = compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e5m2").max), "float32"
                    ),
                )
                qdq_upstream = quantize_and_dequantize(
                    upstream, scale, "float8_e5m2", layer.compute_dtype
                )
                new_amax_history = compute_float8_amax_history(
                    upstream, amax_history
                )
                return qdq_upstream, new_scale, new_amax_history

            return outputs, grad

        x = geometry.contract(
            quantized_dequantize_inputs(
                inputs,
                ops.convert_to_tensor(layer.inputs_scale),
                ops.convert_to_tensor(layer.inputs_amax_history),
            ),
            quantized_dequantize_inputs(
                ops.convert_to_tensor(layer._kernel),
                ops.convert_to_tensor(layer.kernel_scale),
                ops.convert_to_tensor(layer.kernel_amax_history),
            ),
        )
        # `quantized_dequantize_outputs` is placed immediately after the
        # contraction for the sake of pattern matching in gemm_rewrite. That
        # way, the qdq will be adjacent to the corresponding matmul_bprop in
        # the bprop.
        x = quantized_dequantize_outputs(
            x,
            ops.convert_to_tensor(layer.outputs_grad_scale),
            ops.convert_to_tensor(layer.outputs_grad_amax_history),
        )
        if layer.bias is not None:
            # Under non-mixed precision cases, F32 bias has to be converted
            # to BF16 first to get the biasAdd fusion support. ref. PR
            # https://github.com/tensorflow/tensorflow/pull/60306
            bias = layer.bias
            if layer.compute_dtype == "float32":
                bias_bf16 = ops.cast(bias, "bfloat16")
                bias = ops.cast(bias_bf16, bias.dtype)
            x = ops.add(x, bias)
        if layer.activation is not None:
            x = layer.activation(x)
        return x
