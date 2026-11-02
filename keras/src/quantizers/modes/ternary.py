from keras.src import initializers
from keras.src import ops
from keras.src.quantizers.geometry import EinsumProjectionGeometry
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantization_config import TernaryQuantizationConfig
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import TernaryTrits
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class TernaryStrategy(QuantizationStrategy):
    """Ternary (BitNet b1.58) quantization: weights in `{-1, 0, +1}`.

    The ternarization rule (threshold and scale) is owned by the layer's
    geometry: the default is the BitNet b1.58 rule applied to the float
    kernel, and `TernaryDense` supplies its straight-through-estimator
    values instead.
    """

    name = "ternary"
    config_cls = TernaryQuantizationConfig
    # A LoRA update cannot survive a merged save: re-ternarizing the merged
    # weight shrinks the scale on every save, and rounding it onto the
    # stored three-level grid drops all but the largest updates.
    supports_lora = False
    geometry_families = ("projection",)

    def check_quantizable(self, layer):
        geometry = self.require_geometry(layer)
        # The ternary math is written for the 2-D kernel of a `Dense`.
        if isinstance(geometry, EinsumProjectionGeometry):
            raise NotImplementedError(
                "Quantization mode 'ternary' supports only a `Dense` kernel, "
                "not the einsum kernel of layer "
                f"{layer.__class__.__name__}."
            )

    def build(self, layer, input_shape, config):
        del config
        self.check_quantizable(layer)
        input_dim, units = input_shape
        # Stored as `[in, packed(out)]` like every other packed projection:
        # five trits per byte (3^5 == 243 <= 256) along the output axis.
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(input_dim, TernaryTrits.packed_length(units)),
            # 121 = 1+3+9+27+81: byte whose five base-3 digits are all 0,
            # decoding to trit 0 (neutral). "zeros" (byte 0) has the same
            # digits but maps to trit -1, giving an all-minus-one kernel.
            initializer=initializers.Constant(121),
            dtype="uint8",
            trainable=False,
        )
        # Scalar BitNet b1.58 beta scale; 1.0 in fixed-threshold mode.
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=(),
            initializer="ones",
            trainable=False,
        )

    def quantized_weight(self, layer):
        # The scale is the scalar multiplier `beta`. The forward pass
        # applies it to the matmul output rather than to the codes.
        return QuantizedWeight(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=TernaryTrits(axis=-1, orig_len=layer.units),
            scheme=WeightScheme(code_range=(-1, 1), scale_form="multiplier"),
            shape=self.require_geometry(layer).weight_shape,
        )

    def call(self, layer, inputs, **kwargs):
        # A storage format, not a compute win: the packed kernel is unpacked
        # to `{-1, 0, +1}` on every call and fed to a standard matmul, so
        # inference is slightly slower than a float `Dense` call. A native
        # ternary kernel reading the packed format would be needed for a
        # speedup.
        kernel = ops.cast(
            self.quantized_weight(layer).unpack(), layer.compute_dtype
        )
        x = ops.matmul(inputs, kernel)
        x = ops.multiply(x, ops.cast(layer.kernel_scale, layer.compute_dtype))
        return apply_bias_activation(layer, x)

    def quantize(self, layer, config):
        del config
        geometry = self.require_geometry(layer)
        kernel_shape = geometry.weight_shape
        # The geometry owns the ternarization rule: the BitNet b1.58 rule by
        # default, or the layer's own values (`TernaryDense` freezes exactly
        # the forward value of its straight-through kernel, so quantizing
        # does not change the layer's outputs).
        kernel_ternary, beta = geometry.ternary_values()
        layout = TernaryTrits(axis=-1, orig_len=kernel_shape[-1])
        packed_kernel = layout.pack(kernel_ternary)
        del layer._kernel
        layer.quantized_build(kernel_shape, "ternary")
        layer._kernel.assign(packed_kernel)
        layer.kernel_scale.assign(beta)
