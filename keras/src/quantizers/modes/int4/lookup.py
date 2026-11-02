"""int4 handlers for the lookup family (`Embedding`, `ReversibleEmbedding`)."""

import math

from keras.src import ops
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import add_reverse_lookup_lora_delta
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.modes.common import encode_reverse_lookup
from keras.src.quantizers.modes.common import reverse_lookup_dtype
from keras.src.quantizers.modes.common import reverse_lookup_params
from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.int4.block_size import is_grouped
from keras.src.quantizers.modes.int4.block_size import is_per_channel
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.packing import unpack_int4
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)
from keras.src.quantizers.quantizers import dequantize_with_sz_map


class Int4LookupHandlers:
    """The int4 build, forward, encode and views of an embeddings table.

    The table is packed two codes per byte along `output_dim`. The scale
    runs per row (per-channel) or per row and group of columns (grouped,
    with a zero point and a group index).
    """

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        """Build variables for int4 quantization of an embeddings table.

        Args:
            layer: The layer being built.
            geometry: The layer's `LookupGeometry`.
            embeddings_shape: Original shape `(input_dim, output_dim)`.
            config: Optional quantization config specifying block_size.
        """
        input_dim, output_dim = embeddings_shape
        block_size = self.resolve_block_size(layer, config)

        # The table is packed two int4 values per byte along `output_dim`;
        # the scale runs per row (per channel) or per row and group.
        packed_output_dim = Int4Pairs.packed_length(output_dim)
        layer._embeddings = layer.add_weight(
            name="embeddings",
            shape=(input_dim, packed_output_dim),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        if is_per_channel(block_size):
            scale_shape = (input_dim,)
        else:
            scale_shape = (input_dim, math.ceil(output_dim / block_size))
        layer.embeddings_scale = layer.add_weight(
            name="embeddings_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )
        if is_grouped(block_size):
            # Grouped quantization is asymmetric: a zero point per row and
            # group, and the column-to-group index.
            def idx_initializer(shape, dtype):
                return ops.floor_divide(
                    ops.arange(output_dim, dtype=dtype), block_size
                )

            layer.embeddings_zero = layer.add_weight(
                name="embeddings_zero",
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
                shape=(output_dim,),
                initializer=idx_initializer,
                dtype="float32",
                trainable=False,
                autocast=False,
            )
        else:
            layer.embeddings_zero = None
            layer.g_idx = None

        if geometry.reversible:
            # Weight-only by default, like an int4 projection; a config may
            # add an activation quantizer for the reverse projection.
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(config, None)
            )
            if not layer.tie_weights:
                # The reverse table is the forward layout transposed: packed
                # along `output_dim` (axis 0), with the scale and zero point
                # per column or per group of rows and column.
                reverse_scale_shape = tuple(reversed(scale_shape))
                layer.reverse_embeddings = layer.add_weight(
                    name="reverse_embeddings",
                    shape=(packed_output_dim, input_dim),
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )
                layer.reverse_embeddings_scale = layer.add_weight(
                    name="reverse_embeddings_scale",
                    shape=reverse_scale_shape,
                    initializer="ones",
                    trainable=False,
                )
                if is_grouped(block_size):
                    layer.reverse_embeddings_zero = layer.add_weight(
                        name="reverse_embeddings_zero",
                        shape=reverse_scale_shape,
                        initializer="zeros",
                        dtype="int8",
                        trainable=False,
                    )
                else:
                    layer.reverse_embeddings_zero = None

    def _call_lookup(self, layer, inputs, reverse=False):
        """Forward pass for an int4 quantized embeddings lookup."""
        if reverse:
            return self._reverse_lookup(layer, inputs)
        inputs = cast_lookup_inputs(inputs)

        # Gather the packed rows first, then unpack only those.
        rows = ops.take(layer._embeddings, inputs, axis=0)
        outputs = unpack_int4(rows, layer.output_dim, axis=-1)

        block_size = self.resolve_block_size(layer, layer.quantization_config)

        if is_per_channel(block_size):
            embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
            outputs = ops.divide(
                ops.cast(outputs, dtype=layer.compute_dtype),
                ops.expand_dims(embeddings_scale, axis=-1),
            )
        else:
            # Sub-channel: look up scale/zero for each input token,
            # then dequantize using g_idx to expand groups
            embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
            embeddings_zero = ops.take(layer.embeddings_zero, inputs, axis=0)

            # Scale/zero are [batch..., n_groups], g_idx is [output_dim]
            outputs = dequantize_with_sz_map(
                ops.cast(outputs, dtype=layer.compute_dtype),
                embeddings_scale,
                embeddings_zero,
                layer.g_idx,
                group_axis=-1,
            )

        return add_lookup_lora_delta(layer, inputs, outputs)

    def _reverse_lookup(self, layer, inputs):
        """Reverse projection through an int4 quantized table."""
        block_size = self.resolve_block_size(layer, layer.quantization_config)
        per_channel = is_per_channel(block_size)
        dtype = reverse_lookup_dtype(layer)
        inputs = ops.cast(inputs, dtype)
        table = self._get_reverse_lookup_quantized_weight(
            layer, layer._quantization_geometry()
        )
        unpacked_embeddings = table.unpack()

        if layer.inputs_quantizer:
            inputs_q, inputs_scale = layer.inputs_quantizer(inputs)
        else:
            inputs_q, inputs_scale = inputs, ops.ones((1,), dtype=dtype)

        if per_channel:
            # Symmetric: matmul on the int values, then fold both scales
            # into the logits.
            logits = ops.matmul(inputs_q, unpacked_embeddings)
            logits = ops.cast(logits, dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, table.scale))
        else:
            # Asymmetric sub-channel: the zero point cannot be pulled out of
            # the matmul, so dequantize the embeddings first.
            float_embeddings = dequantize_with_sz_map(
                ops.cast(unpacked_embeddings, dtype),
                table.scale,
                table.zero_point,
                table.g_idx,
                group_axis=0,
            )
            logits = ops.matmul(inputs_q, float_embeddings)
            logits = ops.divide(logits, inputs_scale)

        # The scales are float32 variables; the projection reports its own
        # dtype, as the float layer does.
        logits = ops.cast(logits, dtype)
        logits = add_reverse_lookup_lora_delta(layer, inputs, logits)
        return apply_logit_soft_cap(layer, logits)

    def _encode_lookup(self, layer, geometry, weight, config):
        # `Int4Strategy.resolve_block_size` is the single source of truth for
        # the group size, shared with the build path and the dtype-policy
        # naming. A bare `quantize("int4")` resolves to the canonical
        # `Int4QuantizationConfig()` (grouped, block_size=128); `None`/`-1`
        # selects per-channel.
        block_size = self.resolve_block_size(layer, config)

        if is_per_channel(block_size):
            # Per-channel quantization
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=-1, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                weight, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            embeddings_zero = None
        else:
            # Sub-channel quantization with asymmetric zero point
            # Transpose to put output_dim first for grouped quantization
            embeddings_t = ops.transpose(weight)

            embeddings_value_t, scale_t, zero_t = (
                abs_max_quantize_grouped_with_zero_point(
                    embeddings_t,
                    block_size=block_size,
                    value_range=(-8, 7),
                    dtype="int8",
                    to_numpy=True,
                )
            )
            # Transpose back to (input_dim, output_dim) layout
            embeddings_value = ops.transpose(embeddings_value_t)
            embeddings_scale = ops.transpose(scale_t)
            embeddings_zero = ops.transpose(zero_t)

        packed_embeddings_value, _, _ = pack_int4(embeddings_value, axis=-1)
        return packed_embeddings_value, embeddings_scale, embeddings_zero

    def _get_lookup_quantized_weight(self, layer, geometry):
        block_size = self.resolve_block_size(layer, layer.quantization_config)
        return QuantizedWeight(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            layout=Int4Pairs(axis=-1, orig_len=layer.output_dim),
            scheme=int4_scheme(block_size),
            shape=(layer.input_dim, layer.output_dim),
            axis=-1,
            zero_point=layer.embeddings_zero,
            g_idx=layer.g_idx,
        )

    def _get_reverse_lookup_quantized_weight(self, layer, geometry):
        # A tied layer's reverse table is its forward table transposed.
        block_size = self.resolve_block_size(layer, layer.quantization_config)
        grouped = is_grouped(block_size)
        codes, scale, zero_point = reverse_lookup_params(
            layer, with_zero_point=grouped
        )
        return QuantizedWeight(
            codes=codes,
            scale=scale,
            layout=Int4Pairs(axis=0, orig_len=layer.output_dim),
            scheme=int4_scheme(block_size),
            shape=(layer.output_dim, layer.input_dim),
            axis=0,
            zero_point=zero_point,
            g_idx=layer.g_idx,
        )

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = geometry.weight_shape
        grouped = is_grouped(self.resolve_block_size(layer, config))
        packed_embeddings_value, embeddings_scale, embeddings_zero = (
            self._encode_lookup(layer, geometry, layer._embeddings, config)
        )
        del layer._embeddings
        untied = geometry.reversible and not layer.tie_weights
        if untied:
            reverse_value, reverse_scale, reverse_zero = encode_reverse_lookup(
                self, layer, geometry, config
            )
            del layer.reverse_embeddings

        layer.quantized_build(embeddings_shape, "int4", config)
        layer._embeddings.assign(packed_embeddings_value)
        layer.embeddings_scale.assign(embeddings_scale)
        if grouped:
            layer.embeddings_zero.assign(embeddings_zero)
        if untied:
            layer.reverse_embeddings.assign(reverse_value)
            layer.reverse_embeddings_scale.assign(reverse_scale)
            if grouped:
                layer.reverse_embeddings_zero.assign(reverse_zero)
