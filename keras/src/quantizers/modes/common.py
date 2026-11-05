"""Shared building blocks for the built-in quantization modes.

The helpers here are pure code motion: each one emits exactly the op
sequence its call sites emitted inline, so modes that adopt them keep
producing identical traced programs.
"""

from keras.src import ops
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class GeometryDispatchStrategy(QuantizationStrategy):
    """A mode whose math is written once per geometry family.

    Each verb resolves the layer's geometry and calls the mode's handler
    for that verb and the geometry's family. For each family it
    supports, a mode implements:

    - `_build_<family>`, `_call_<family>` and `_quantize_<family>`: the
      variables, the forward pass and the conversion.
    - `_get_<family>_quantized_weight`: the `QuantizedWeight` view, or
      `None` when the mode holds no integer codes. The weight property,
      saving, `enable_lora` and `Model.quantization_summary` read it.
    - `_encode_<family>`, when the view is not `None`: the LoRA-merged
      save re-quantizes the merged weight with it.
    - `_get_reverse_<family>_quantized_weight`, for a reversible family
      whose layer is untied and has a view: the reverse table's view.

    A mode implements every handler of a family it supports.
    `check_quantizable` resolves the quantize, build, call and view
    handlers before `Layer.quantize` changes the layer, so a mode missing
    one of them is refused with `NotImplementedError` and the layer stays
    as it was. A missing encode or reverse view handler raises only when
    its verb runs. A subclass that overrides a verb itself needs no
    handlers for it.
    """

    # The handler that each verb calls, by geometry family.
    _handler_names = {
        "build": "_build_{family}",
        "call": "_call_{family}",
        "quantize": "_quantize_{family}",
        "encode": "_encode_{family}",
        "quantized_weight": "_get_{family}_quantized_weight",
        "reverse_quantized_weight": "_get_reverse_{family}_quantized_weight",
    }

    def build(self, layer, input_shape, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("build", geometry.family, layer)
        handler(layer, geometry, input_shape, config)

    def call(self, layer, *args, **kwargs):
        geometry = self.require_geometry(layer)
        handler = self._handler("call", geometry.family, layer)
        return handler(layer, *args, **kwargs)

    def quantize(self, layer, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("quantize", geometry.family, layer)
        handler(layer, geometry, config)

    def check_quantizable(self, layer):
        geometry = self.require_geometry(layer)
        for verb in ("quantize", "build", "call", "quantized_weight"):
            # A subclass that overrides the verb itself needs no handler.
            if getattr(type(self), verb) is getattr(
                GeometryDispatchStrategy, verb
            ):
                self._handler(verb, geometry.family, layer)

    def encode(self, layer, weight, config=None):
        geometry = self.require_geometry(layer)
        handler = self._handler("encode", geometry.family, layer)
        return handler(layer, geometry, weight, config)

    def quantized_weight(self, layer):
        geometry = self.require_geometry(layer)
        handler = self._handler("quantized_weight", geometry.family, layer)
        return handler(layer, geometry)

    def quantized_weights(self, layer):
        views = super().quantized_weights(layer)
        geometry = self.require_geometry(layer)
        if views and geometry.reversible and not layer.tie_weights:
            # An untied reversible lookup holds a second table.
            handler = self._handler(
                "reverse_quantized_weight", geometry.family, layer
            )
            views += (handler(layer, geometry),)
        return views

    def _handler(self, verb, family, layer):
        """Returns this mode's implementation of `verb` for one family."""
        name = self._handler_names[verb].format(family=family)
        handler = getattr(self, name, None)
        if handler is None:
            raise NotImplementedError(
                f"Quantization mode '{self.name}' does not implement "
                f"`{name}` for the '{family}' quantization geometry of "
                f"layer {layer.__class__.__name__}."
            )
        return handler


def cast_lookup_inputs(inputs):
    """Casts embedding-lookup indices to `int32` unless already integral."""
    if standardize_dtype(inputs.dtype) not in ("int32", "int64"):
        inputs = ops.cast(inputs, "int32")
    return inputs


def apply_bias_activation(layer, x):
    """Adds the layer's bias and applies its activation, when present."""
    if layer.bias is not None:
        x = ops.add(x, layer.bias)
    if layer.activation is not None:
        x = layer.activation(x)
    return x


def add_lookup_lora_delta(layer, inputs, outputs):
    """Adds the LoRA or DoRA update to gathered embeddings."""
    if layer.lora_enabled:
        lora_outputs = ops.take(layer.lora_embeddings_a, inputs, axis=0)
        lora_outputs = ops.matmul(lora_outputs, layer.lora_embeddings_b)
        outputs = ops.add(
            outputs, (layer.lora_alpha / layer.lora_rank) * lora_outputs
        )
        outputs = ops.cast(outputs, dtype=layer.compute_dtype)
    elif getattr(layer, "dora_enabled", False):
        quantized_weight = layer._quantized_weight()
        embeddings = (
            layer._embeddings
            if quantized_weight is None
            else quantized_weight.dequantize(layer.compute_dtype)
        )
        delta = (layer.dora_alpha / layer.dora_rank) * ops.matmul(
            layer.dora_embeddings_a, layer.dora_embeddings_b
        )
        combined = ops.add(embeddings, delta)
        norm = ops.sqrt(ops.sum(ops.square(combined), axis=0))
        embeddings = layer.dora_magnitude * ops.divide_no_nan(combined, norm)
        outputs = ops.take(embeddings, inputs, axis=0)
        outputs = ops.cast(outputs, dtype=layer.compute_dtype)
    return outputs


def apply_logit_soft_cap(layer, logits):
    """Applies the reverse-projection logit soft cap, when configured."""
    if layer.logit_soft_cap is not None:
        soft_cap = layer.logit_soft_cap
        logits = ops.multiply(ops.tanh(ops.divide(logits, soft_cap)), soft_cap)
    return logits


def reverse_lookup_params(layer, with_zero_point=False):
    """The stored table, scale and zero point of the reverse projection.

    An untied layer stores the reverse table in its own layout. A tied layer
    stores only the forward table, `(input_dim, ...)`, so its tensors are
    transposed into the reverse layout (transposing the 1-D per-channel
    scale is a no-op, so per-channel and grouped take the same path). The
    zero point is `None` unless `with_zero_point`.
    """
    if not layer.tie_weights:
        return (
            layer.reverse_embeddings,
            layer.reverse_embeddings_scale,
            layer.reverse_embeddings_zero if with_zero_point else None,
        )
    return (
        ops.transpose(layer._embeddings),
        ops.transpose(layer.embeddings_scale),
        ops.transpose(layer.embeddings_zero) if with_zero_point else None,
    )


def reverse_lookup_dtype(layer):
    """The dtype the reverse projection computes in.

    Mirrors the float layer, which casts the inputs and the kernel to
    `reverse_dtype` when it is set and otherwise runs in `compute_dtype`.
    """
    return layer.reverse_dtype or layer.compute_dtype


def add_reverse_lookup_lora_delta(layer, inputs, logits):
    """Adds the LoRA update to reverse-projection logits, when enabled.

    Only a tied layer projects back through the adapted table (an untied
    layer's reverse table has no adapter), and the delta is taken from the
    float inputs, before any activation quantization.
    """
    if layer.tie_weights and layer.lora_enabled:
        lora_logits = ops.matmul(inputs, ops.transpose(layer.lora_embeddings_b))
        lora_logits = ops.matmul(
            lora_logits, ops.transpose(layer.lora_embeddings_a)
        )
        logits = ops.add(
            logits,
            ops.cast(
                (layer.lora_alpha / layer.lora_rank) * lora_logits, logits.dtype
            ),
        )
    return logits


def encode_reverse_lookup(strategy, layer, geometry, config):
    """Quantizes the reverse table by the forward rule on its transpose.

    The reverse table is the forward layout transposed, so encoding its
    transpose and transposing the results back applies exactly the rule the
    forward table gets, including a user-supplied `weight_quantizer`.
    """
    codes, scale, zero_point = strategy._encode_lookup(
        layer, geometry, ops.transpose(layer.reverse_embeddings), config
    )
    return (
        ops.transpose(codes),
        ops.transpose(scale),
        None if zero_point is None else ops.transpose(zero_point),
    )
