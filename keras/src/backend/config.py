import json
import os
import warnings

from keras.src.api_export import keras_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"

# Default backend: TensorFlow.
_BACKEND = "tensorflow"

BUILT_IN_BACKENDS = frozenset(["jax", "numpy", "tensorflow", "torch"])

PLUGGABLE_BACKENDS = frozenset(["mlx", "openvino", "paddle"])

# Whether NNX is enabled.
_NNX_ENABLED = False

# Cap run duration for debugging.
_MAX_EPOCHS = None
_MAX_STEPS_PER_EPOCH = None

# This flag forces Keras to use backend-agnostic ops even if there is
# backend-specific implementation available.
_USE_BACKEND_AGNOSTIC_OPS = False

# How Keras handles 64-bit dtypes, see `set_x64_mode()`.
_X64_MODE = "off"
_X64_MODES = ("off", "explicit_x64", "enable_x64")
# TODO: Add mlx and paddle, and let pluggable backends declare support.
_X64_BACKENDS = frozenset(["jax", "numpy", "torch"])


@keras_export(["keras.config.floatx", "keras.backend.floatx"])
def floatx():
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:

    >>> keras.config.floatx()
    'float32'

    """
    return _FLOATX


@keras_export(["keras.config.set_floatx", "keras.backend.set_floatx"])
def set_floatx(value):
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"` for training,
    as this will likely cause numeric stability issues.
    Instead, mixed precision, which leverages
    a mix of `float16` and `float32`. It can be configured by calling
    `keras.mixed_precision.set_dtype_policy('mixed_float16')`.

    Args:
        value: String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.

    Examples:
    >>> keras.config.floatx()
    'float32'

    >>> keras.config.set_floatx('float64')
    >>> keras.config.floatx()
    'float64'

    >>> # Set it back to float32
    >>> keras.config.set_floatx('float32')

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"bfloat16", "float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. "
            f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


@keras_export("keras.config.x64_mode")
def x64_mode():
    """Returns how Keras handles 64-bit dtypes, as a string.

    One of `"off"`, `"explicit_x64"` or `"enable_x64"`, see
    `keras.config.set_x64_mode()`.

    Example:

    >>> keras.config.x64_mode()
    'off'

    """
    return _X64_MODE


@keras_export("keras.config.set_x64_mode")
def set_x64_mode(mode):
    """Sets how Keras handles 64-bit dtypes.

    64-bit dtypes are `int64`, `uint64`, `float64` and `complex128`.

    Args:
        mode: One of:
            - `"off"` (default): the result dtype of every op is capped to
              32-bit, e.g. adding two `int64` tensors gives an `int32` tensor.
            - `"explicit_x64"`: a 64-bit input gives a 64-bit output. Inputs
              that are all 32-bit, Python scalars and default dtypes stay
              32-bit. This mirrors JAX's `jax_explicit_x64_dtypes="allow"`,
              which Keras turns on for you on the JAX backend.
            - `"enable_x64"`: not supported yet.

    The mode can also be set with the `KERAS_X64_MODE` environment variable
    or the `"x64_mode"` field of `~/.keras/keras.json`. Set it before
    creating tensors or models. Supported on the `jax`, `numpy` and `torch`
    backends. On the JAX backend, Keras turns on the matching JAX flag. If
    JAX's own 64-bit flags are already on and don't match `mode`, Keras
    warns and keeps `mode`.

    Example:

    >>> keras.config.set_x64_mode("explicit_x64")
    >>> x = keras.ops.ones((2,), dtype="int64")
    >>> keras.ops.add(x, x).dtype
    'int64'

    Raises:
        ValueError: If `mode` is unknown, not supported yet, or not supported
            on the current backend.
    """
    global _X64_MODE
    _validate_x64_mode(mode)
    if backend() == "jax":
        _reconcile_jax_x64_options(mode)
    _X64_MODE = mode


def _validate_x64_mode(mode):
    """Raises if `mode` is unknown or unsupported (yet, or on this backend)."""
    if mode not in _X64_MODES:
        raise ValueError(
            f"Unknown `x64_mode`: {mode!r}. Expected one of {_X64_MODES}."
        )
    if mode == "enable_x64":
        raise ValueError(
            "`x64_mode='enable_x64'` is not supported yet. Use 'explicit_x64'."
        )
    if mode != "off" and backend() not in _X64_BACKENDS:
        raise ValueError(
            f"`x64_mode={mode!r}` is not supported on the {backend()} "
            f"backend yet. Supported backends: {sorted(_X64_BACKENDS)}."
        )


def _reconcile_jax_x64_options(mode):
    """Puts JAX's 64-bit flags in `mode`, or warns if they disagree with it.

    Nothing changes if the flags already match `mode`. If they are off, the
    flag for `mode` is turned on. Otherwise the user set them to another
    mode, which Keras does not override.
    """
    import jax

    explicit = getattr(jax.config, "jax_explicit_x64_dtypes", None)
    if jax.config.jax_enable_x64:
        jax_mode, flag = "enable_x64", "`jax.config.jax_enable_x64` is True"
    elif getattr(explicit, "name", None) == "ALLOW":
        jax_mode = "explicit_x64"
        flag = "`jax.config.jax_explicit_x64_dtypes` is 'allow'"
    else:
        jax_mode, flag = "off", None
    if jax_mode == mode:
        return
    if flag:
        warnings.warn(
            f"{flag}, which does not match Keras's `x64_mode={mode!r}`. "
            "It is recommended to set JAX's 64-bit flags through Keras's "
            "`x64_mode` rather than directly. Otherwise, make sure JAX and "
            "Keras have matching 64-bit behavior."
        )
        return
    # Only "explicit_x64" gets here until "enable_x64" is supported.
    if explicit is None:
        raise ValueError(
            "`x64_mode='explicit_x64'` requires JAX 0.8.0 or newer, which "
            f"adds `jax_explicit_x64_dtypes`. Found JAX {jax.__version__}."
        )
    jax.config.update("jax_explicit_x64_dtypes", "allow")


@keras_export(["keras.config.epsilon", "keras.backend.epsilon"])
def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:

    >>> keras.config.epsilon()
    1e-07

    """
    return _EPSILON


@keras_export(["keras.config.set_epsilon", "keras.backend.set_epsilon"])
def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Examples:
    >>> keras.config.epsilon()
    1e-07

    >>> keras.config.set_epsilon(1e-5)
    >>> keras.config.epsilon()
    1e-05

    >>> # Set it back to the default value.
    >>> keras.config.set_epsilon(1e-7)

    """
    global _EPSILON
    _EPSILON = value


@keras_export(
    [
        "keras.config.image_data_format",
        "keras.backend.image_data_format",
    ]
)
def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:

    >>> keras.config.image_data_format()
    'channels_last'

    """
    return _IMAGE_DATA_FORMAT


@keras_export(
    [
        "keras.config.set_image_data_format",
        "keras.backend.set_image_data_format",
    ]
)
def set_image_data_format(data_format):
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Examples:

    >>> keras.config.image_data_format()
    'channels_last'

    >>> keras.config.set_image_data_format('channels_first')
    >>> keras.config.image_data_format()
    'channels_first'

    >>> # Set it back to `'channels_last'`
    >>> keras.config.set_image_data_format('channels_last')

    """
    global _IMAGE_DATA_FORMAT
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    _IMAGE_DATA_FORMAT = data_format


@keras_export("keras.config.enable_flash_attention")
def enable_flash_attention():
    """Enable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once enabled, supported layers like `MultiHeadAttention` will **attempt** to
    use flash attention for faster computations. By default, this feature is
    enabled.

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.
    """
    from keras.src.backend.common import global_state

    global_state.set_global_attribute("flash_attention", None)


@keras_export("keras.config.disable_flash_attention")
def disable_flash_attention():
    """Disable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once disabled, supported layers like `MultiHeadAttention` will not
    use flash attention for faster computations.
    """
    from keras.src.backend.common import global_state

    global_state.set_global_attribute("flash_attention", False)


@keras_export("keras.config.is_flash_attention_enabled")
def is_flash_attention_enabled():
    """Checks whether flash attention is globally enabled in Keras.

    Flash attention is a performance-optimized method for computing attention
    in large models, such as transformers, allowing for faster and more
    memory-efficient operations. This function checks the global Keras
    configuration to determine if flash attention is enabled for compatible
    layers (e.g., `MultiHeadAttention`).

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.

    Returns:
        `False` if disabled; otherwise, it indicates that it is enabled.
    """
    from keras.src.backend.common import global_state

    return global_state.get_global_attribute("flash_attention", default=None)


@keras_export("keras.config.is_nnx_enabled")
def is_nnx_enabled():
    """Checks whether NNX specific features are enabled for the JAX backend.

    Returns:
        bool: `True` if NNX backend features are enabled, `False` otherwise.
        Defaults to `False`.
    """
    return _NNX_ENABLED


def set_nnx_enabled(value):
    global _NNX_ENABLED
    from keras.src.backend.common import global_state

    _NNX_ENABLED = bool(value)
    if _NNX_ENABLED:
        try:
            from flax import nnx  # noqa F401
        except ImportError:
            raise ImportError(
                "To use NNX with the JAX backend, you must install `flax`."
            )
    global_state.set_global_attribute("nnx_enabled", bool(value))


def standardize_data_format(data_format):
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def tmp_keras_dir_name():
    # Private helper returning the name of the `.keras` directory used when
    # falling back to a tmp location. That location is shared by all local
    # users, so the name is made user specific to give each user their own.
    if hasattr(os, "getuid"):
        return f".keras-{os.getuid()}"
    return ".keras"


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp/.keras-<uid>.
if "KERAS_HOME" in os.environ:
    _KERAS_DIR = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    _KERAS_DIR = os.path.join(_keras_base_dir, ".keras")
    if not os.access(_keras_base_dir, os.W_OK):
        _KERAS_DIR = os.path.join("/tmp", tmp_keras_dir_name())


def keras_home():
    # Private accessor for the keras home location.
    return _KERAS_DIR


# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_KERAS_DIR, "keras.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get("floatx", floatx())
    if _floatx not in {"float16", "float32", "float64"}:
        raise ValueError(
            "Invalid `floatx` configuration. "
            "Expected one of {'float16', 'float32', 'float64'}. "
            f"Received: floatx={_floatx}"
        )
    _epsilon = _config.get("epsilon", epsilon())
    if not isinstance(_epsilon, float):
        raise ValueError(
            "Invalid `epsilon` configuration. "
            "Expected a float. "
            f"Received: epsilon={_epsilon}"
        )
    _backend = _config.get("backend", _BACKEND)
    _image_data_format = _config.get("image_data_format", image_data_format())
    if _image_data_format not in {"channels_last", "channels_first"}:
        raise ValueError(
            "Invalid `image_data_format` configuration. "
            "Expected one of {'channels_last', 'channels_first'}. "
            f"Received: image_data_format={_image_data_format}"
        )
    _nnx_enabled_config = _config.get("nnx_enabled", _NNX_ENABLED)
    _x64_mode_config = _config.get("x64_mode", _X64_MODE)

    # Apply basic configs that don't cause circular import
    set_floatx(_floatx)
    _NNX_ENABLED = _nnx_enabled_config
    _X64_MODE = _x64_mode_config
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_KERAS_DIR):
    try:
        os.makedirs(_KERAS_DIR)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        "floatx": floatx(),
        "epsilon": epsilon(),
        "backend": _BACKEND,
        "image_data_format": image_data_format(),
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on KERAS_BACKEND flag, if applicable.
if "KERAS_BACKEND" in os.environ:
    _backend = os.environ["KERAS_BACKEND"]
    if _backend:
        _BACKEND = _backend
if "KERAS_MAX_EPOCHS" in os.environ:
    _MAX_EPOCHS = int(os.environ["KERAS_MAX_EPOCHS"])
if "KERAS_MAX_STEPS_PER_EPOCH" in os.environ:
    _MAX_STEPS_PER_EPOCH = int(os.environ["KERAS_MAX_STEPS_PER_EPOCH"])


if _BACKEND != "tensorflow":
    # If we are not running on the tensorflow backend, we should stop tensorflow
    # from using all available GPU memory. See
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


@keras_export(
    [
        "keras.config.backend",
        "keras.backend.backend",
    ]
)
def backend():
    """Publicly accessible method for determining the current backend.

    Returns:
        String, the name of the backend Keras is currently using. One of
            `"tensorflow"`, `"torch"`, or `"jax"`.

    Example:

    >>> keras.config.backend()
    'tensorflow'

    """
    return _BACKEND


@keras_export(["keras.config.set_max_epochs"])
def set_max_epochs(max_epochs):
    """Limit the maximum number of epochs for any call to fit.

    This will cap the number of epochs for any training run using `model.fit()`.
    This is purely for debugging, and can also be set via the `KERAS_MAX_EPOCHS`
    environment variable to quickly run a script without modifying its source.

    Args:
        max_epochs: The integer limit on the number of epochs or `None`. If
            `None`, no limit is applied.
    """
    global _MAX_EPOCHS
    _MAX_EPOCHS = max_epochs


@keras_export(["keras.config.set_max_steps_per_epoch"])
def set_max_steps_per_epoch(max_steps_per_epoch):
    """Limit the maximum number of steps for any call to fit/evaluate/predict.

    This will cap the number of steps for single epoch of a call to `fit()`,
    `evaluate()`, or `predict()`. This is purely for debugging, and can also be
    set via the `KERAS_MAX_STEPS_PER_EPOCH` environment variable to quickly run
    a scrip without modifying its source.

    Args:
        max_steps_per_epoch: The integer limit on the number of steps per
            epoch or `None`. If `None`, no limit is applied.
    """
    global _MAX_STEPS_PER_EPOCH
    _MAX_STEPS_PER_EPOCH = max_steps_per_epoch


@keras_export(["keras.config.max_epochs"])
def max_epochs():
    """Get the maximum number of epochs for any call to fit.

    Retrieves the limit on the number of epochs set by
    `keras.config.set_max_epochs` or the `KERAS_MAX_EPOCHS` environment
    variable.

    Returns:
        The integer limit on the number of epochs or `None`, if no limit has
        been set.
    """
    return _MAX_EPOCHS


@keras_export(["keras.config.max_steps_per_epoch"])
def max_steps_per_epoch():
    """Get the maximum number of steps for any call to fit/evaluate/predict.

    Retrieves the limit on the number of steps per epoch set by
    `keras.config.set_max_steps_per_epoch` or the `KERAS_MAX_STEPS_PER_EPOCH`
    environment variable.

    Returns:
        The integer limit on the number of steps per epoch or `None`, if no
        limit has been set.
    """
    return _MAX_STEPS_PER_EPOCH


if "KERAS_NNX_ENABLED" in os.environ:
    env_val = os.environ["KERAS_NNX_ENABLED"].lower()
    if env_val == "true" or env_val == "1":
        _NNX_ENABLED = True
    else:
        _NNX_ENABLED = False

set_nnx_enabled(_NNX_ENABLED)

if os.environ.get("KERAS_X64_MODE"):
    _X64_MODE = os.environ["KERAS_X64_MODE"]

set_x64_mode(_X64_MODE)  # Validates the configured mode and syncs JAX.

if "KERAS_USE_BACKEND_AGNOSTIC_OPS" in os.environ:
    env_val = os.environ["KERAS_USE_BACKEND_AGNOSTIC_OPS"].lower()
    _USE_BACKEND_AGNOSTIC_OPS = env_val in ("true", "1", "yes")


def _use_backend_agnostic_ops():
    """Returns the flag that checks if backend-agnostic ops are forced."""
    return _USE_BACKEND_AGNOSTIC_OPS


def _set_use_backend_agnostic_ops(value):
    """Internal helper to force backend-agnostic ops for testing/benchmarks."""
    global _USE_BACKEND_AGNOSTIC_OPS
    _USE_BACKEND_AGNOSTIC_OPS = bool(value)
