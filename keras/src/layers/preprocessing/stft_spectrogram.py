import math
import warnings

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.utils.module_utils import scipy


@keras_export("keras.layers.STFTSpectrogram")
class STFTSpectrogram(layers.Layer):
    """Layer to compute the Short-Time Fourier Transform (STFT) on a 1D signal.

    A layer that computes Spectrograms of the input signal to produce
    a spectrogram. This layers utilizes Short-Time Fourier Transform (STFT) by
    The layer computes Spectrograms based on STFT by utilizing convolution
    kernels, which allows parallelization on GPUs and trainable kernels for
    fine-tuning support. This layer allows different modes of output
    (e.g., log-scaled magnitude, phase, power spectral density, etc.) and
    provides flexibility in windowing, padding, and scaling options for the
    STFT calculation.

    Examples:

    Apply it as a non-trainable preprocessing layer on 3 audio tracks of
    1 channel, 10 seconds and sampled at 16 kHz.

    >>> layer = keras.layers.STFTSpectrogram(
    ...     mode='log',
    ...     frame_length=256,
    ...     frame_step=128,   # 50% overlap
    ...     fft_length=512,
    ...     window="hann",
    ...     padding="valid",
    ...     trainable=False,  # non-trainable, preprocessing only
    ... )
    >>> layer(keras.random.uniform(shape=(3, 160000, 1))).shape
    (3, 1249, 257)

    Apply it as a trainable processing layer on 3 stereo audio tracks of
    2 channels, 10 seconds and sampled at 16 kHz. This is initialized as the
    non-trainable layer, but then can be trained jointly within a model.

    >>> layer = keras.layers.STFTSpectrogram(
    ...     mode='log',
    ...     frame_length=256,
    ...     frame_step=128,    # 50% overlap
    ...     fft_length=512,
    ...     window="hamming",  # hamming windowing function
    ...     padding="same",    # padding to preserve the time dimension
    ...     trainable=True,    # trainable, this is the default in keras
    ... )
    >>> layer(keras.random.uniform(shape=(3, 160000, 2))).shape
    (3, 1250, 514)

    Similar to the last example, but add an extra dimension so the output is
    an image to be used with image models. We apply this here on a signal of
    3 input channels to output an image tensor, hence is directly applicable
    with an image model.

    >>> layer = keras.layers.STFTSpectrogram(
    ...     mode='log',
    ...     frame_length=256,
    ...     frame_step=128,
    ...     fft_length=512,
    ...     padding="same",
    ...     expand_dims=True,  # this adds the extra dimension
    ... )
    >>> layer(keras.random.uniform(shape=(3, 160000, 3))).shape
    (3, 1250, 257, 3)

    Args:
        mode: String, the output type of the spectrogram. Can be one of
            `"log"`, `"magnitude`", `"psd"`, `"real`", `"imag`", `"angle`",
            `"stft`". Defaults to `"log`".
        frame_length: Integer, The length of each frame (window) for STFT in
            samples. Defaults to 256.
        frame_step: Integer, the step size (hop length) between
            consecutive frames. If not provided, defaults to half the
            frame_length. Defaults to `frame_length // 2`.
        fft_length: Integer, the size of frequency bins used in the Fast-Fourier
            Transform (FFT) to apply to each frame. Should be greater than or
            equal to `frame_length`.  Recommended to be a power of two. Defaults
            to the smallest power of two that is greater than or equal
            to `frame_length`.
        window: (String or array_like), the windowing function to apply to each
            frame. Can be `"hann`" (default), `"hamming`", or a custom window
            provided as an array_like.
        periodic: Boolean, if True, the window function will be treated as
            periodic. Defaults to `False`.
        scaling: String, type of scaling applied to the window. Can be
            `"density`", `"spectrum`", or None. Default is `"density`".
        padding: String, padding strategy. Can be `"valid`" or `"same`".
            Defaults to `"valid"`.
        expand_dims: Boolean, if True, will expand the output into spectrograms
            into two dimensions to be compatible with image models.
            Defaults to `False`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, weight)`. Defaults to `"channels_last"`.

    Raises:
        ValueError: If an invalid value is provided for `"mode`", `"scaling`",
            `"padding`", or other input arguments.
        TypeError: If the input data type is not one of `"float16`",
            `"float32`", or `"float64`".

    Input shape:
        A 3D tensor of shape `(batch_size, time_length, input_channels)`, if
        `data_format=="channels_last"`, and of shape
        `(batch_size, input_channels, time_length)` if
        `data_format=="channels_first"`, where `time_length` is the length of
        the input signal, and `input_channels` is the number of input channels.
        The same kernels are applied to each channel independently.

    Output shape:
        If `data_format=="channels_first" and not expand_dims`, a 3D tensor:
            `(batch_size, input_channels * freq_channels, new_time_length)`
        If `data_format=="channels_last" and not expand_dims`, a 3D tensor:
            `(batch_size, new_time_length, input_channels * freq_channels)`
        If `data_format=="channels_first" and expand_dims`, a 4D tensor:
            `(batch_size, input_channels, new_time_length, freq_channels)`
        If `data_format=="channels_last" and expand_dims`, a 4D tensor:
            `(batch_size, new_time_length, freq_channels, input_channels)`

        where `new_time_length` depends on the padding, and `freq_channels` is
        the number of FFT bins `(fft_length // 2 + 1)`.
    """

    def __init__(
        self,
        mode="log",
        frame_length=256,
        frame_step=None,
        fft_length=None,
        window="hann",
        periodic=False,
        scaling="density",
        padding="valid",
        expand_dims=False,
        data_format=None,
        **kwargs,
    ):
        if frame_step is not None and (
            frame_step > frame_length or frame_step < 1
        ):
            raise ValueError(
                "`frame_step` should be a positive integer not greater than "
                f"`frame_length`. Received frame_step={frame_step}, "
                f"frame_length={frame_length}"
            )

        if fft_length is not None and fft_length < frame_length:
            raise ValueError(
                "`fft_length` should be not less than `frame_length`. "
                f"Received fft_length={fft_length}, frame_length={frame_length}"
            )

        if fft_length is not None and (fft_length & -fft_length) != fft_length:
            warnings.warn(
                "`fft_length` is recommended to be a power of two. "
                f"Received fft_length={fft_length}"
            )

        all_modes = ["log", "magnitude", "psd", "real", "imag", "angle", "stft"]

        if mode not in all_modes:
            raise ValueError(
                "Output mode is invalid, it must be one of "
                f"{', '.join(all_modes)}. Received: mode={mode}"
            )

        if scaling is not None and scaling not in ["density", "spectrum"]:
            raise ValueError(
                "Scaling is invalid, it must be `None`, 'density' "
                f"or 'spectrum'. Received scaling={scaling}"
            )

        if padding not in ["valid", "same"]:
            raise ValueError(
                "Padding is invalid, it should be 'valid', 'same'. "
                f"Received: padding={padding}"
            )

        if isinstance(window, str):
            # throws an exception for invalid window function
            scipy.signal.get_window(window, 1)

        super().__init__(**kwargs)

        self.mode = mode

        self.frame_length = frame_length
        self.frame_step = frame_step
        self._frame_step = frame_step or self.frame_length // 2
        self.fft_length = fft_length
        self._fft_length = fft_length or (
            2 ** int(math.ceil(math.log2(frame_length)))
        )

        self.window = window
        self.periodic = periodic
        self.scaling = scaling
        self.padding = padding
        self.expand_dims = expand_dims
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = layers.input_spec.InputSpec(ndim=3)

    def build(self, input_shape):
        shape = (self.frame_length, 1, self._fft_length // 2 + 1)

        if self.mode != "imag":
            self.real_kernel = self.add_weight(
                name="real_kernel",
                shape=shape,
                initializer=initializers.STFT(
                    "real", self.window, self.scaling, self.periodic
                ),
            )
        if self.mode != "real":
            self.imag_kernel = self.add_weight(
                name="imag_kernel",
                shape=shape,
                initializer=initializers.STFT(
                    "imag", self.window, self.scaling, self.periodic
                ),
            )

    def _adjust_shapes(self, outputs):
        _, channels, freq_channels, time_seq = ops.shape(outputs)
        batch_size = -1
        if self.data_format == "channels_last":
            if self.expand_dims:
                outputs = ops.transpose(outputs, [0, 3, 2, 1])
                # [batch_size, time_seq, freq_channels, input_channels]
            else:
                outputs = ops.reshape(
                    outputs,
                    [batch_size, channels * freq_channels, time_seq],
                )
                # [batch_size, input_channels * freq_channels, time_seq]
                outputs = ops.transpose(outputs, [0, 2, 1])
        else:
            if self.expand_dims:
                outputs = ops.transpose(outputs, [0, 1, 3, 2])
                # [batch_size, channels, time_seq, freq_channels]
            else:
                outputs = ops.reshape(
                    outputs,
                    [batch_size, channels * freq_channels, time_seq],
                )
        return outputs

    def _apply_conv(self, inputs, kernel):
        if self.data_format == "channels_last":
            _, time_seq, channels = ops.shape(inputs)
            inputs = ops.transpose(inputs, [0, 2, 1])
            inputs = ops.reshape(inputs, [-1, time_seq, 1])
        else:
            _, channels, time_seq = ops.shape(inputs)
            inputs = ops.reshape(inputs, [-1, 1, time_seq])

        outputs = ops.conv(
            inputs,
            ops.cast(kernel, standardize_dtype(inputs.dtype)),
            padding=self.padding,
            strides=self._frame_step,
            data_format=self.data_format,
        )
        batch_size = -1
        if self.data_format == "channels_last":
            _, time_seq, freq_channels = ops.shape(outputs)
            outputs = ops.transpose(outputs, [0, 2, 1])
            outputs = ops.reshape(
                outputs,
                [batch_size, channels, freq_channels, time_seq],
            )
        else:
            _, freq_channels, time_seq = ops.shape(outputs)
            outputs = ops.reshape(
                outputs,
                [batch_size, channels, freq_channels, time_seq],
            )
        return outputs

    def call(self, inputs):
        dtype = inputs.dtype
        if standardize_dtype(dtype) not in {
            "float16",
            "float32",
            "float64",
        }:
            raise TypeError(
                "Invalid input type. Expected `float16`, `float32` or "
                f"`float64`. Received: input type={dtype}"
            )

        real_signal = None
        imag_signal = None
        power = None

        if self.mode != "imag":
            real_signal = self._apply_conv(inputs, self.real_kernel)
        if self.mode != "real":
            imag_signal = self._apply_conv(inputs, self.imag_kernel)

        if self.mode == "real":
            return self._adjust_shapes(real_signal)
        elif self.mode == "imag":
            return self._adjust_shapes(imag_signal)
        elif self.mode == "angle":
            return self._adjust_shapes(ops.arctan2(imag_signal, real_signal))
        elif self.mode == "stft":
            return self._adjust_shapes(
                ops.concatenate([real_signal, imag_signal], axis=2)
            )
        else:
            power = ops.square(real_signal) + ops.square(imag_signal)

        if self.mode == "psd":
            return self._adjust_shapes(
                power
                + ops.pad(
                    power[:, :, 1:-1, :], [[0, 0], [0, 0], [1, 1], [0, 0]]
                )
            )
        linear_stft = self._adjust_shapes(
            ops.sqrt(ops.maximum(power, backend.epsilon()))
        )

        if self.mode == "magnitude":
            return linear_stft
        else:
            return ops.log(ops.maximum(linear_stft, backend.epsilon()))

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            channels = input_shape[-1]
        else:
            channels = input_shape[1]
        freq_channels = self._fft_length // 2 + 1
        if self.mode == "stft":
            freq_channels *= 2
        shape = ops.operation_utils.compute_conv_output_shape(
            input_shape,
            freq_channels * channels,
            (self.frame_length,),
            strides=self._frame_step,
            padding=self.padding,
            data_format=self.data_format,
        )
        if self.data_format == "channels_last":
            batch_size, time_seq, _ = shape
        else:
            batch_size, _, time_seq = shape
        if self.expand_dims:
            if self.data_format == "channels_last":
                return (batch_size, time_seq, freq_channels, channels)
            else:
                return (batch_size, channels, time_seq, freq_channels)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "window": self.window,
                "periodic": self.periodic,
                "scaling": self.scaling,
                "padding": self.padding,
                "data_format": self.data_format,
                "expand_dims": self.expand_dims,
            }
        )
        return config
