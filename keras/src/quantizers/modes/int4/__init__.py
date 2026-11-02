"""W4A16 weight-only quantization (packed int4 weights).

The mode is one strategy assembled from one handler class per geometry
family: `projection` (`Dense`, `EinsumDense`) and `lookup` (`Embedding`,
`ReversibleEmbedding`). `Int4Strategy` itself holds what is shared: the
policy-string codec and the block-size resolution.
"""

from keras.src.dtype_policies.dtype_policy import Int4DTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.int4.lookup import Int4LookupHandlers
from keras.src.quantizers.modes.int4.projection import Int4ProjectionHandlers
from keras.src.quantizers.quantization_config import Int4QuantizationConfig


class Int4Strategy(
    Int4ProjectionHandlers, Int4LookupHandlers, GeometryDispatchStrategy
):
    """W4A16 weight-only quantization (packed int4 weights)."""

    name = "int4"
    config_cls = Int4QuantizationConfig
    geometry_families = ("projection", "lookup")

    def resolve_block_size(self, layer, config):
        """Determine the block size for int4 quantization.

        The block size can be specified either through the `config` argument
        or through the `dtype_policy` if it is of type `Int4DTypePolicy`.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the block size should be specified in the
        `dtype_policy`.

        Args:
            layer: The layer being quantized.
            config: An optional configuration object that may contain the
                `block_size` attribute.
        Returns:
            int or None. The determined block size for int4 quantization.
            Returns `None` or `-1` for per-channel quantization.
        """
        if isinstance(config, Int4QuantizationConfig):
            return config.block_size
        policy = layer._own_dtype_policy
        if isinstance(policy, Int4DTypePolicy):
            # Convert -1 to None for consistency
            return None if policy.block_size == -1 else policy.block_size
        # A legacy bare "int4" policy (no `Int4DTypePolicy`) is per-channel.
        return None

    def policy_from_string(self, mode_str, source_name):
        # Legacy bare "int4" policies carry no block size and stay generic
        # (they resolve to per-channel quantization on reload).
        if "/" in mode_str:
            return Int4DTypePolicy(mode_str, source_name)
        else:
            return QuantizedDTypePolicy(mode_str, source_name)

    def config_from_policy(self, policy):
        # The config builds what a layer built from the policy name holds,
        # so a legacy bare "int4" name, which a build reads as per-channel,
        # gives a per-channel config.
        if isinstance(policy, Int4DTypePolicy):
            return Int4QuantizationConfig(block_size=policy.block_size)
        return Int4QuantizationConfig(block_size=-1)

    def policy_suffix(self, layer, config):
        # Include block_size in policy name for sub-channel quantization.
        block_size = self.resolve_block_size(layer, config)
        # Use -1 for per-channel, otherwise use block_size
        block_size_value = -1 if block_size is None else block_size
        return f"int4/{block_size_value}"
