from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib
from keras.src.utils.module_utils import scipy


@keras_export(["keras.initializers.Constant", "keras.initializers.constant"])
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Constant(10.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Constant(10.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        value: A Python scalar.
    """

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, shape, dtype=None):
        dtype = standardize_dtype(dtype)
        return ops.cast(self.value, dtype=dtype) * ops.ones(
            shape=shape, dtype=dtype
        )

    def get_config(self):
        return {"value": serialization_lib.serialize_keras_object(self.value)}

    @classmethod
    def from_config(cls, config):
        value = serialization_lib.deserialize_keras_object(config["value"])
        return cls(value)


@keras_export(["keras.initializers.Zeros", "keras.initializers.zeros"])
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(units=3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        dtype = standardize_dtype(dtype)
        return ops.zeros(shape, dtype=dtype)


@keras_export(["keras.initializers.Ones", "keras.initializers.ones"])
class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `ones`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Ones()
    >>> layer = Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        dtype = standardize_dtype(dtype)
        return ops.ones(shape, dtype=dtype)


@keras_export(
    [
        "keras.initializers.Identity",
        "keras.initializers.identity",
        "keras.initializers.IdentityInitializer",
    ]
)
class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices. "
                f"Received: shape={shape} of rank {len(shape)}."
            )
        dtype = standardize_dtype(dtype)
        return self.gain * ops.eye(*shape, dtype=dtype)

    def get_config(self):
        return {"gain": self.gain}


@keras_export(
    [
        "keras.initializers.STFT",
        "keras.initializers.stft",
        "keras.initializers.STFTInitializer",
    ]
)
class STFT(Initializer):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT).

    Since the formula involves complex numbers, this class compute either the
    real or the imaginary components of the final output.

    Additionally, this initializer supports windowing functions across the time
    dimension as commonly used in STFT. Windowing functions from the module
    `scipy.signal.windows` are supported, including the common `hann` and
    `hamming` windowing functions. This layer supports periodic windows and
    scaling-based normalization.

    This is primarily intended for use in the `STFTSpectrogram` layer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = STFTInitializer("real", "hann", "density", False)
    >>> values = initializer(shape=(128, 1, 513))

    Args:
        side: String, `"real"` or `"imag"` deciding if the kernel will compute
            the real side or the imaginary side of the output. Defaults to
            `"real"`.
        window: String for the name of the windowing function in the
            `scipy.signal.windows` module, or array_like for the window values,
            or `None` for no windowing.
        scaling: String, `"density"` or `"spectrum"` for scaling of the window
            for normalization, either L2 or L1 normalization.
            `None` for no scaling.
        periodic: Boolean, if True, the window function will be treated as
            periodic. Defaults to `False`.
    """

    def __init__(
        self, side="real", window="hann", scaling="density", periodic=False
    ):
        if side not in ["real", "imag"]:
            raise ValueError(f"side should be 'real' or 'imag', not {side}")
        if isinstance(window, str):
            # throws an exception for invalid window function
            scipy.signal.get_window(window, 1)
        if scaling is not None and scaling not in ["density", "spectrum"]:
            raise ValueError(
                "Scaling is invalid, it must be `None`, 'density' "
                f"or 'spectrum'. Received scaling={scaling}"
            )
        self.side = side
        self.window = window
        self.scaling = scaling
        self.periodic = periodic

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        The shape is assumed to be `(T, 1, F // 2 + 1)`, where `T` is the size
        of the given window, and `F` is the number of frequency bands. Only half
        the frequency bands are used, which is a common practice in STFT,
        because the second half are the conjugates of the first half in
        a reversed order.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        dtype = standardize_dtype(dtype)
        frame_length, input_channels, fft_length = shape

        win = None
        scaling = 1
        if self.window is not None:
            win = self.window
            if isinstance(win, str):
                # Using SciPy since it provides more windowing functions,
                # easier to be compatible with multiple backends.
                win = scipy.signal.get_window(win, frame_length, self.periodic)
            win = ops.convert_to_tensor(win, dtype=dtype)
            if len(win.shape) != 1 or win.shape[-1] != frame_length:
                raise ValueError(
                    "The shape of `window` must be equal to [frame_length]."
                    f"Received: window shape={win.shape}"
                )
            win = ops.reshape(win, [frame_length, 1, 1])
            if self.scaling == "density":
                scaling = ops.sqrt(ops.sum(ops.square(win)))
            elif self.scaling == "spectrum":
                scaling = ops.sum(ops.abs(win))

        _fft_length = (fft_length - 1) * 2
        freq = ops.divide(
            ops.reshape(
                ops.arange(fft_length, dtype=dtype), (1, 1, fft_length)
            ),
            _fft_length,
        )
        time = ops.reshape(
            ops.arange(frame_length, dtype=dtype), (frame_length, 1, 1)
        )
        args = ops.multiply(ops.multiply(-2, time), freq) * ops.arccos(
            ops.cast(-1, dtype)
        )

        if self.side == "real":
            kernel = ops.cast(ops.cos(args), dtype)
        else:
            kernel = ops.cast(ops.sin(args), dtype)

        if win is not None:
            kernel = ops.divide(ops.multiply(kernel, win), scaling)
        return kernel

    def get_config(self):
        return {
            "side": self.side,
            "window": self.window,
            "periodic": self.periodic,
            "scaling": self.scaling,
        }
