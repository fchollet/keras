"""The calibration-based quantization modes, GPTQ and AWQ.

GPTQ and AWQ allocate the same family of variables, run the same
dequantize-and-contract forward pass, and speak the same three-part policy
grammar. They differ in the code bit-width (which fixes how the kernel is
packed), in the dtype policy class, in the calibrator class, in the driver
that calibrates a model and in one extra AWQ variable of input scales that
the quantized weight divides out. `GPTQStrategy` and `AWQStrategy`, at the
end of this module, declare these differences.
"""

import math

from keras.src import ops
from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy
from keras.src.dtype_policies.dtype_policy import GPTQDTypePolicy
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantized_weight import Int2Quads
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class CalibrationStrategy(QuantizationStrategy):
    """A post-training strategy whose values arrive from a calibration pass."""

    geometry_families = ("projection",)
    requires_config = True
    requires_layer_structure = True
    # The mode's dedicated dtype policy class.
    policy_cls = None

    def quantize(self, layer, config):
        # The quantized values arrive later, so this only allocates the
        # mode's variables from the layer's current weight shape.
        geometry = self.require_geometry(layer)
        layer.quantized_build(geometry.weight_shape, self.name, config)
        # A live float layer keeps its float kernel as the weight until
        # `write_back` installs the calibrated codes.
        layer.calibration_pending = True

    # --- Config and policy-string surface ---------------------------------

    def policy_from_string(self, mode_str, source_name):
        return self.policy_cls(mode_str, source_name)

    def config_from_policy(self, policy):
        name = self.name.upper()
        raise ValueError(
            f"Implicitly enabling {name} quantization by setting "
            f"`dtype_policy` to '{policy.name}' is not supported. "
            f"{name} requires a calibration dataset and a config object "
            f"(`{self.config_cls.__name__}`).\n\n"
            f"Please use the `.quantize('{self.name}', config=...)` method "
            "on the layer or model instead."
        )

    def _missing_config_error(self):
        return (
            f"For {self.name.upper()}, the `config` argument must be of "
            f"type `{self.config_cls.__name__}`."
        )

    def policy_suffix(self, layer, config):
        del layer
        return config.dtype_policy_string()

    def resolve_group_size(self, layer, config):
        """Determine the group size from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "group_size")

    def resolve_weight_bits(self, layer, config):
        """Determine the weight bits from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "weight_bits")

    def _resolve_from_config_or_policy(self, layer, config, attr):
        """Resolves a hyperparameter with config-over-policy precedence.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the value comes from the mode's dtype policy.
        """
        if isinstance(config, self.config_cls):
            return getattr(config, attr)
        policy = layer._own_dtype_policy
        if policy.quantization_mode == self.name:
            return getattr(policy, attr)
        raise ValueError(
            f"For {self.name.upper()} quantization, the {attr} must be "
            "specified either through a `dtype_policy` of type "
            f"`{self.policy_cls.__name__}` or the `config` argument. "
            f"Received: dtype_policy={policy!r}"
        )

    # --- Variables --------------------------------------------------------

    # Name of the variable of per-input-row scales that multiplied the
    # weights before quantization, or `None`. The quantized weight divides
    # them back out (`QuantizedWeight.input_scales`).
    input_scales_name = None

    def build(self, layer, input_shape, config):
        """Allocates the quantized kernel and quantization parameters.

        The variables hold uninitialized values until the calibration pass
        (run by `Model.quantize`) writes the quantized weights back.
        """
        geometry = self.require_geometry(layer)
        # Allocation alone leaves nothing pending: a layer built under a
        # calibration policy loads its codes from a checkpoint. `quantize`
        # marks a live float layer pending after this returns.
        layer.calibration_pending = False

        rows, columns = geometry.calibration_rows_columns(input_shape)

        bits = self.resolve_weight_bits(layer, config)
        kernel_columns = self._get_pack_layout(bits, columns).packed_length(
            columns
        )
        group_size = self.resolve_group_size(layer, config)
        n_groups = 1 if group_size == -1 else math.ceil(rows / group_size)

        # Stored in the kernel's own `[in, out]` orientation and packed
        # along the output axis, like the int4 layout, so the forward pass
        # unpacks and dequantizes without a transpose.
        layer.quantized_kernel = layer.add_weight(
            name="kernel",
            shape=(rows, kernel_columns),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=(n_groups, columns),
            initializer="ones",
            trainable=False,
        )
        layer.kernel_zero = layer.add_weight(
            name="zero_point",
            shape=(n_groups, columns),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        if self.input_scales_name is not None:
            setattr(
                layer,
                self.input_scales_name,
                layer.add_weight(
                    name=self.input_scales_name,
                    shape=(rows,),
                    initializer="ones",
                    trainable=False,
                ),
            )
        # `g_idx` is stored as `float32` because TF has no GPU kernel for
        # int32 resource variables (would pin the variable to CPU and break
        # jit_compile on GPU); consumers cast to int32 on-device.
        # Not autocast: bfloat16 holds integers exactly only up to 256.
        layer.g_idx = layer.add_weight(
            name="g_idx",
            shape=(rows,),
            initializer="zeros",
            dtype="float32",
            trainable=False,
            autocast=False,
        )

    # --- Calibration state ------------------------------------------------

    def write_back(
        self, layer, codes, scale, zero_point, g_idx, input_scales=None
    ):
        """Installs the calibrated values and retires the float kernel.

        `codes` are the unpacked codes in the kernel's `[in, out]`
        orientation; they are packed here as `build` lays out the variable.
        `input_scales` go to the `input_scales_name` variable of a mode
        that has one.
        """
        bits = self.resolve_weight_bits(layer, layer.quantization_config)
        codes = ops.cast(codes, layer.quantized_kernel.dtype)
        codes = self._get_pack_layout(bits, codes.shape[-1]).pack(codes)
        del layer._kernel
        layer.quantized_kernel.assign(codes)
        layer.kernel_scale.assign(scale)
        layer.kernel_zero.assign(zero_point)
        layer.g_idx.assign(g_idx)
        if self.input_scales_name is not None:
            getattr(layer, self.input_scales_name).assign(input_scales)
        layer.calibration_pending = False

    def check_saveable(self, layer):
        if layer.calibration_pending:
            raise ValueError(
                f"Cannot save layer '{layer.name}' because it is quantized "
                f"with mode '{self.name}' but has never been calibrated. Its "
                "quantized weights are uninitialized, so saving would "
                "produce a corrupted model. Run calibration first, e.g. via "
                "`model.quantize(...)` with a quantization layer structure "
                "that covers this layer, or exclude the layer from "
                "quantization with `filters`."
            )

    def variables_loaded(self, layer):
        # A stored calibration checkpoint is always calibrated: loading
        # completes the transition and retires the float kernel a live
        # `quantize()` left in place.
        if layer.calibration_pending:
            del layer._kernel
            layer.calibration_pending = False

    @staticmethod
    def _get_pack_layout(bits, columns):
        """How `columns` codes of `bits` bits pack along the output axis."""
        if bits == 4:
            return Int4Pairs(axis=-1, orig_len=columns)
        if bits == 2:
            return Int2Quads(axis=-1, orig_len=columns)
        # 3-bit codes are not packed densely (3 does not divide 8) and
        # 8-bit codes need no packing: one code per byte.
        return NoPack()

    # --- Quantized weight view --------------------------------------------

    def quantized_weight(self, layer):
        if layer.calibration_pending:
            # The codes are uninitialized; the float kernel is still the
            # layer's weight.
            return None
        geometry = self.require_geometry(layer)
        config = layer.quantization_config
        bits = self.resolve_weight_bits(layer, config)
        group_size = self.resolve_group_size(layer, config)
        # The group parameters are stored as `[n_groups, out]`, so their
        # axes give the unpacked column count the packed codes stand for
        # and, with the group index, the row count.
        columns = int(layer.kernel_scale.shape[1])
        rows = int(layer.g_idx.shape[0])
        return QuantizedWeight(
            codes=layer.quantized_kernel,
            scale=layer.kernel_scale,
            zero_point=layer.kernel_zero,
            g_idx=layer.g_idx,
            layout=self._get_pack_layout(bits, columns),
            scheme=WeightScheme(
                code_range=(0, 2**bits - 1),
                scale_form="multiplier",
                has_zero_point=True,
                # `-1` means one group spanning every input row.
                group_size=rows if group_size == -1 else group_size,
            ),
            shape=geometry.weight_shape,
            axis=0,
            input_scales=(
                getattr(layer, self.input_scales_name)
                if self.input_scales_name is not None
                else None
            ),
        )

    # --- Forward pass -----------------------------------------------------

    def call(self, layer, inputs, training=False):
        geometry = self.require_geometry(layer)
        quantized_weight = self.quantized_weight(layer)
        W = (
            layer._kernel
            if quantized_weight is None
            else quantized_weight.dequantize(layer.compute_dtype)
        )
        y = geometry.contract(inputs, W)
        return apply_bias_activation(layer, y)


class GPTQStrategy(CalibrationStrategy):
    """GPTQ post-training quantization (calibration-based, 2/3/4/8-bit).

    GPTQ quantizes the kernel one column at a time and corrects the columns
    still to come with the inverse Hessian of the layer's inputs.
    """

    name = "gptq"
    config_cls = GPTQConfig
    policy_cls = GPTQDTypePolicy

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.calibration_run import gptq_quantize

        del model
        gptq_quantize(config, structure, filters=filters)


class AWQStrategy(CalibrationStrategy):
    """AWQ post-training quantization (activation-aware, 4-bit).

    AWQ uses 4-bit quantization with per-channel AWQ scales that protect
    salient weights based on activation magnitudes.
    """

    name = "awq"
    config_cls = AWQConfig
    policy_cls = AWQDTypePolicy
    # Per-input-row scales from the activation magnitudes.
    input_scales_name = "awq_scales"

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.awq_core import awq_quantize

        del model
        awq_quantize(config, structure, filters=filters)
