from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype


def _strategy_registry():
    """Returns the quantization mode registry, imported on first use.

    The import must be deferred: `keras.src.quantizers` registers the
    built-in modes on import, and those mode modules import the policy
    classes defined below, so a module-level import here would be a cycle.
    """
    from keras.src.quantizers import strategy_registry

    return strategy_registry


def _parse_int4_mode(mode_str):
    """Parses the `"int4/<block_size>"` grammar into policy parameters."""
    parts = mode_str.split("/")
    expected_format = "'int4/<block_size>'"

    # Validate format
    if len(parts) != 2 or parts[0] != "int4":
        raise ValueError(
            "Invalid mode for Int4DTypePolicy. Expected format "
            f"{expected_format}, but got '{mode_str}'."
        )

    # Validate and cast block_size. Older checkpoints encoded per-channel
    # int4 as the literal "None" in the policy string (e.g.
    # "int4/None_from_float32"); normalize it to -1 so those checkpoints
    # still deserialize. Both "None" and -1 mean per-channel quantization.
    if parts[1] == "None":
        block_size = -1
    else:
        try:
            block_size = int(parts[1])
        except ValueError:
            raise ValueError(
                "Invalid mode for Int4DTypePolicy. <block_size> must be "
                "an integer. Expected format "
                f"{expected_format}, but got '{mode_str}'."
            )

    # Validate supported values
    if block_size < -1 or block_size == 0:
        raise ValueError(
            "Invalid block_size in mode. Supported values are "
            "-1 (per-channel) or a positive integer (sub-channel), "
            f"but got {block_size} from '{mode_str}'."
        )
    return {"block_size": block_size}


def _parse_bits_and_group_mode(
    mode_str, name, policy_name, validate_weight_bits, group_size_semantics
):
    """Parses the `"<name>/<weight_bits>/<group_size>"` grammar.

    Shared by the two calibration policies, which differ only in the
    bit-widths they accept and in how `group_size=-1` reads.
    """
    parts = mode_str.split("/")
    expected_format = f"'{name}/<weight_bits>/<group_size>'"

    # Validate format
    if len(parts) != 3 or parts[0] != name:
        raise ValueError(
            f"Invalid mode for {policy_name}. Expected format "
            f"{expected_format}, but got '{mode_str}'."
        )

    # Validate and cast weight_bits and group_size
    try:
        weight_bits = int(parts[1])
        group_size = int(parts[2])
    except ValueError:
        raise ValueError(
            f"Invalid mode for {policy_name}. <weight_bits> and "
            "<group_size> must be integers. Expected format "
            f"{expected_format}, but got '{mode_str}'."
        )

    validate_weight_bits(weight_bits, mode_str)

    if group_size < -1 or group_size == 0:
        raise ValueError(
            "Invalid group_size in mode. Supported values are "
            f"-1 ({group_size_semantics}) or a positive integer, "
            f"but got {group_size} from '{mode_str}'."
        )
    return {"weight_bits": weight_bits, "group_size": group_size}


def _validate_gptq_weight_bits(weight_bits, mode_str):
    if weight_bits not in [2, 3, 4, 8]:
        raise ValueError(
            "Invalid weight_bits in mode. Supported values are "
            f"2, 3, 4, and 8, but got {weight_bits} from '{mode_str}'."
        )


def _validate_awq_weight_bits(weight_bits, mode_str):
    # AWQ presently only supports 4-bit quantization.
    if weight_bits != 4:
        raise ValueError(
            "Invalid weight_bits in mode. AWQ only supports 4-bit "
            f"quantization, but got {weight_bits} from '{mode_str}'."
        )


def _parse_gptq_mode(mode_str):
    return _parse_bits_and_group_mode(
        mode_str,
        "gptq",
        "GPTQDTypePolicy",
        _validate_gptq_weight_bits,
        "whole-tensor",
    )


def _parse_awq_mode(mode_str):
    return _parse_bits_and_group_mode(
        mode_str,
        "awq",
        "AWQDTypePolicy",
        _validate_awq_weight_bits,
        "per-channel",
    )


@keras_export(
    [
        "keras.DTypePolicy",
        "keras.dtype_policies.DTypePolicy",
        "keras.mixed_precision.DTypePolicy",  # Legacy
        "keras.mixed_precision.Policy",  # Legacy
    ]
)
class DTypePolicy:
    """A dtype policy for a Keras layer.

    A dtype policy determines a layer's computation and variable dtypes. Each
    layer has a policy. Policies can be passed to the `dtype` argument of layer
    constructors, or a global policy can be set with
    `keras.config.set_dtype_policy`.

    Args:
        name: The policy name, which determines the compute and variable dtypes.
            Can be any dtype name, such as `"float32"` or `"float64"`,
            which causes both the compute and variable dtypes
            will be that dtype.
            Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
            which causes the compute dtype to be `float16` or `bfloat16`
            and the variable dtype to be `float32`.

    Typically you only need to interact with dtype policies when using mixed
    precision, which is the use of float16 or bfloat16 for computations and
    float32 for variables. This is why the term `mixed_precision` appears in the
    API name. Mixed precision can be enabled by passing `"mixed_float16"` or
    `"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

    >>> keras.config.set_dtype_policy("mixed_float16")
    >>> layer1 = keras.layers.Dense(10)
    >>> layer1.dtype_policy  # layer1 will automatically use mixed precision
    <DTypePolicy "mixed_float16">
    >>> # Can optionally override layer to use float32
    >>> # instead of mixed precision.
    >>> layer2 = keras.layers.Dense(10, dtype="float32")
    >>> layer2.dtype_policy
    <DTypePolicy "float32">
    >>> # Set policy back to initial float32.
    >>> keras.config.set_dtype_policy('float32')

    In the example above, passing `dtype="float32"` to the layer is
    equivalent to passing
    `dtype=keras.config.DTypePolicy("float32")`.
    In general, passing a dtype policy name to a layer is equivalent
    to passing the corresponding policy, so it is never necessary
    to explicitly construct a `DTypePolicy` object.
    """

    def __init__(self, name=None):
        # Use the global dtype policy if `name` is not specified
        if name is None:
            name = dtype_policy().name
        self._name = name
        self._compute_dtype, self._variable_dtype = self._parse_name(name)
        self._quantization_mode = None

    def _parse_name(self, name):
        """Parses a `DTypePolicy` name into a compute and variable dtype.

        Args:
            name: The name of the policy.

        Returns:
            The `(compute_dtype, variable_dtype)` pair.
        """
        if not isinstance(name, str):
            raise TypeError(
                "'name' must be a string, such as 'mixed_float16'. "
                f"Received: name={name} (of type {type(name)})"
            )
        if name == "mixed_float16":
            return "float16", "float32"
        elif name == "mixed_bfloat16":
            return "bfloat16", "float32"
        try:
            dtype = standardize_dtype(name)
            return dtype, dtype
        except ValueError:
            raise ValueError(
                f"Cannot convert '{name}' to a mixed precision "
                "DTypePolicy. Valid policies include 'mixed_float16', "
                "'mixed_bfloat16', and the name of any float dtype such as "
                f"'float32'.{_quantized_policy_hint(name)}"
            )

    @property
    def variable_dtype(self):
        """The variable dtype of this policy.

        This is the dtype layers will create their variables in, unless a layer
        explicitly chooses a different dtype. If this is different than
        `DTypePolicy.compute_dtype`, Layers will cast variables to
        the compute dtype to avoid type errors.

        Variable regularizers are run in the variable dtype, not the compute
        dtype.

        Returns:
            The variable dtype of this policy, as a string.
        """
        return self._variable_dtype

    @property
    def compute_dtype(self):
        """The compute dtype of this policy.

        This is the dtype layers will do their computations in. Typically layers
        output tensors with the compute dtype as well.

        Note that even if the compute dtype is float16 or bfloat16, hardware
        devices may not do individual adds, multiplies, and other fundamental
        operations in float16 or bfloat16, but instead may do some of them in
        float32 for numeric stability. The compute dtype is the dtype of the
        inputs and outputs of the ops that the layer executes.
        Internally, many ops will do certain internal calculations in
        float32 or some other device-internal intermediate format with higher
        precision than float16/bfloat16, to increase numeric stability.

        Returns:
            The compute dtype of this policy, as a string.
        """
        return self._compute_dtype

    @property
    def name(self):
        """Returns the name of this policy."""
        return self._name

    @property
    def quantization_mode(self):
        """The quantization mode of this policy.

        Returns:
            The quantization mode of this policy, as a string. If this policy is
            not quantized, it will return `None`.
        """
        return self._quantization_mode

    def convert_input(self, x, autocast, dtype):
        """Converts the input dtype based on `autocast` and `dtype`.

        Note that `x` can be a tensor, symbolic tensor or numpy array, and this
        method will keep integer inputs untouched and only apply casting to
        floats.
        """

        dtype = standardize_dtype(dtype)
        if backend.ops.is_tensor(x):
            if self._should_cast(x, autocast, dtype):
                x = backend.ops.cast(x, dtype=dtype)
            return x
        elif backend.is_keras_tensor(x):
            if self._should_cast(x, autocast, dtype):
                x = ops.cast(x, dtype=dtype)
            return x
        elif hasattr(x, "__array__"):
            try:
                x = backend.ops.convert_to_tensor(x)
            except TypeError:
                x = backend.ops.convert_to_tensor(x, dtype=dtype)
            if self._should_cast(x, autocast, dtype):
                x = backend.ops.cast(x, dtype=dtype)
            return x
        return x

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        class_name = self.__class__.__name__
        if class_name == "FloatDTypePolicy":
            class_name = "DTypePolicy"
        return f'<{class_name} "{self._name}">'

    def __eq__(self, other):
        if self.__class__ in (DTypePolicy, FloatDTypePolicy):
            if type(other) not in (DTypePolicy, FloatDTypePolicy):
                return False
        else:
            if type(other) is not self.__class__:
                return False
        return self._name == other._name

    def _should_cast(self, x, autocast, dtype):
        x_dtype = standardize_dtype(x.dtype)
        if autocast and is_float_dtype(x_dtype) and x_dtype != dtype:
            return True
        else:
            return False


@keras_export(
    ["keras.FloatDTypePolicy", "keras.dtype_policies.FloatDTypePolicy"]
)
class FloatDTypePolicy(DTypePolicy):
    # An alias for `DTypePolicy`
    pass


@keras_export("keras.dtype_policies.QuantizedDTypePolicy")
class QuantizedDTypePolicy(DTypePolicy):
    def __init__(self, mode, source_name=None):
        # Use the global dtype policy if `source_name` is not specified
        if source_name is None:
            source_name = dtype_policy().name
        name = f"{mode}_from_{source_name}"
        self._compute_dtype, self._variable_dtype = self._parse_name(
            source_name
        )
        self._check_quantization_mode(mode, self._compute_dtype)

        self._name = name
        self._source_name = source_name
        self._quantization_mode = mode

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return (
            self._quantization_mode == other._quantization_mode
            and self._source_name == other._source_name
        )

    def get_config(self):
        return {
            "mode": self._quantization_mode,
            "source_name": self._source_name,
        }

    def _check_quantization_mode(self, mode, compute_dtype):
        if not _strategy_registry().is_registered(mode):
            raise ValueError(
                "Invalid quantization mode. "
                f"Expected one of {_strategy_registry().registered_modes()}. "
                f"Received: mode={mode}"
            )
        if compute_dtype == "float16" and mode == "int8":
            raise ValueError(
                f"Quantization mode='{mode}' doesn't work well with "
                "compute_dtype='float16'."
            )


@keras_export("keras.dtype_policies.QuantizedFloat8DTypePolicy")
class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    default_amax_history_length = 1024

    def __init__(self, mode, source_name=None, amax_history_length=1024):
        super().__init__(mode=mode, source_name=source_name)
        if not isinstance(amax_history_length, int):
            raise TypeError(
                "`amax_history_length` must be an integer. "
                f"Received: amax_history_length={amax_history_length}"
            )
        self._amax_history_length = amax_history_length

    @property
    def amax_history_length(self):
        """The length of the amax history window.

        This property is used for scaling factor computation in float8 training.
        """
        return self._amax_history_length

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return self._amax_history_length == other._amax_history_length

    def get_config(self):
        config = super().get_config()
        config.update({"amax_history_length": self.amax_history_length})
        return config


@keras_export("keras.dtype_policies.Int4DTypePolicy")
class Int4DTypePolicy(QuantizedDTypePolicy):
    """Quantized dtype policy for int4 quantization.

    This policy helps propagate quantization settings for int4 sub-channel
    quantization when loading a quantized model in Keras format.

    Args:
        mode: The quantization mode. This should be a string in the format
            `"int4/<block_size>"`.
            -   `"int4"`: The identifier for the quantization algorithm.
            -   `<block_size>`: The block size for sub-channel quantization.
                Use -1 for per-channel (legacy) quantization. Any positive
                integer enables sub-channel quantization with that block size.
            Example: `"int4/128"` for sub-channel with 128-element groups.
        source_name: The source dtype policy name, e.g. "float32".
    """

    def __init__(
        self,
        mode,
        source_name=None,
    ):
        params = _parse_int4_mode(mode)
        block_size = params["block_size"]

        base_mode = "int4"
        super().__init__(
            mode=base_mode,
            source_name=source_name,
        )

        # Use the normalized block_size so a loaded "int4/None" policy reports
        # the canonical "int4/-1" name (identical for all other block sizes).
        self._name = f"{base_mode}/{block_size}_from_{self._source_name}"
        self.mode = base_mode
        self.block_size = block_size

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return self.block_size == other.block_size

    def get_config(self):
        config = super().get_config()
        # Reconstruct the full mode string for serialization
        mode = f"{self.mode}/{self.block_size}"
        config.update({"mode": mode})
        return config


@keras_export("keras.dtype_policies.GPTQDTypePolicy")
class GPTQDTypePolicy(QuantizedDTypePolicy):
    """Quantized dtype policy for GPTQ quantization.

    This policy helps propagate quantization settings for GPTQ
    when loading a GPTQ quantized model in Keras format.

    Args:
        mode: The quantization mode. This should be a string in the format
            `"gptq/<weight_bits>/<group_size>"`.
            -   `"gptq"`: The identifier for the quantization algorithm.
            -   `<weight_bits>`: Number of bits to quantize weights to.
                Supported values are 2, 3, 4, and 8.
            -   `<group_size>`: The group size for quantization. Supported
                values are -1 (for whole-tensor quantization) or any
                positive integer. Typically a smaller group size leads
                to better accuracy but slower speed.
            Example: `"gptq/4/128"`.
        source_name: The source dtype policy name, e.g. "float32".
    """

    def __init__(
        self,
        mode,
        source_name=None,
    ):
        params = _parse_gptq_mode(mode)

        base_mode = "gptq"
        super().__init__(
            mode=base_mode,
            source_name=source_name,
        )

        self._name = f"{mode}_from_{self._source_name}"
        self.mode = base_mode
        self.weight_bits = params["weight_bits"]
        self.group_size = params["group_size"]

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return (
            self.weight_bits == other.weight_bits
            and self.group_size == other.group_size
        )

    def get_config(self):
        config = super().get_config()
        # Reconstruct the full mode string for serialization
        mode = f"{self.mode}/{self.weight_bits}/{self.group_size}"
        config.update({"mode": mode})
        return config


@keras_export("keras.dtype_policies.AWQDTypePolicy")
class AWQDTypePolicy(QuantizedDTypePolicy):
    """Quantized dtype policy for AWQ quantization.

    This policy helps propagate quantization settings for AWQ
    when loading an AWQ quantized model in Keras format.

    Args:
        mode: The quantization mode. This should be a string in the format
            `"awq/<weight_bits>/<group_size>"`.
            -   `"awq"`: The identifier for the quantization algorithm.
            -   `<weight_bits>`: Number of bits to quantize weights to.
                AWQ presently only supports 4-bit quantization.
            -   `<group_size>`: The group size for quantization. Supported
                values are -1 (for per-channel quantization) or any
                positive integer.
            Example: `"awq/4/128"`.
        source_name: The source dtype policy name, e.g. "float32".
    """

    def __init__(
        self,
        mode,
        source_name=None,
    ):
        params = _parse_awq_mode(mode)

        base_mode = "awq"
        super().__init__(
            mode=base_mode,
            source_name=source_name,
        )

        self._name = f"{mode}_from_{self._source_name}"
        self.mode = base_mode
        self.weight_bits = params["weight_bits"]
        self.group_size = params["group_size"]

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return (
            self.weight_bits == other.weight_bits
            and self.group_size == other.group_size
        )

    def get_config(self):
        config = super().get_config()
        # Reconstruct the full mode string for serialization
        mode = f"{self.mode}/{self.weight_bits}/{self.group_size}"
        config.update({"mode": mode})
        return config


@keras_export(
    [
        "keras.config.set_dtype_policy",
        "keras.mixed_precision.set_dtype_policy",  # Legacy
        "keras.mixed_precision.set_global_policy",  # Legacy
    ]
)
def set_dtype_policy(policy):
    """Sets the default dtype policy globally.

    The policy must not be quantized. Every layer and model created
    afterwards takes the global policy, including the layers that cannot be
    quantized, so a quantized global policy would fail their calls. To
    quantize a layer or a model, call its `quantize()` method.

    Example:

    >>> keras.config.set_dtype_policy("mixed_float16")
    """
    if not isinstance(policy, DTypePolicy):
        if isinstance(policy, str):
            if _is_quantized_policy_string(policy):
                policy = _get_quantized_dtype_policy_by_str(policy)
            else:
                policy = DTypePolicy(policy)
        else:
            raise ValueError(
                "Invalid `policy` argument. "
                "Expected the string name of a policy "
                "(such as 'mixed_float16') or a `DTypePolicy` "
                f"instance. Received: policy={policy} "
                f"(of type {type(policy)})"
            )
    if policy.quantization_mode is not None:
        raise ValueError(
            "The global dtype policy cannot be quantized: every layer and "
            "model created afterwards takes it, including the layers that "
            "cannot be quantized. To quantize a layer or a model, call its "
            f"`quantize()` method. Received: policy='{policy.name}'"
        )
    global_state.set_global_attribute("dtype_policy", policy)


@keras_export(
    [
        "keras.config.dtype_policy",
        "keras.mixed_precision.dtype_policy",  # Legacy
        "keras.mixed_precision.global_policy",  # Legacy
    ]
)
def dtype_policy():
    """Returns the current default dtype policy object."""
    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy is None:
        policy = DTypePolicy(backend.floatx())
        set_dtype_policy(policy)
    return policy


def _quantized_mode_of(policy):
    """Returns the quantization mode a policy string names, or `None`.

    A quantized policy string is `<mode>[/<params>]_from_<source>`, where
    `<mode>` is a registered mode. Every other string is a plain dtype
    policy name, including a bare mode name such as `"int8"`.
    """
    if "_from_" not in policy:
        return None
    mode = policy.split("_from_")[0].split("/")[0]
    return mode if _strategy_registry().is_registered(mode) else None


def _quantized_policy_hint(policy):
    """Returns an error hint when `policy` is a quantized policy name."""
    if not _is_quantized_policy_string(policy):
        return ""
    return (
        " You passed a quantized dtype policy name, which is not allowed to "
        "create a `DTypePolicy`. To quantize a layer or a model, call its "
        "`quantize()` method instead."
    )


def _is_quantized_policy_string(policy):
    """Whether a policy string names a registered quantization mode."""
    return _quantized_mode_of(policy) is not None


def _get_quantized_dtype_policy_by_str(policy):
    if not isinstance(policy, str):
        raise TypeError(f"`policy` must be a string. Received: policy={policy}")
    name = _quantized_mode_of(policy)
    if name is None:
        raise ValueError(
            "`policy` is incompatible with the current supported quantization."
        )
    split_name = policy.split("_from_")
    if len(split_name) != 2:
        raise ValueError(
            "Cannot convert `policy` into a valid pair (`mode`, `source_name`) "
            "to instantiate `QuantizedDTypePolicy`. "
            f"Received: policy={policy}"
        )
    mode, source_name = split_name
    if not source_name:
        raise ValueError(
            f"Cannot convert '{policy}' to a quantized DTypePolicy. The "
            "source dtype policy name after `_from_` must not be empty."
        )
    if source_name == "None":
        # Older checkpoints carry a literal "None" source; use the default.
        source_name = None
    strategy = _strategy_registry().get_strategy(name)
    return strategy.policy_from_string(mode, source_name)
