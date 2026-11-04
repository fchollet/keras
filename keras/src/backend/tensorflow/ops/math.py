import numpy as np
import tensorflow as tf

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.tensorflow.ops.core import cast
from keras.src.backend.tensorflow.ops.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_sum(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_max(data, segment_ids)
    else:
        if num_segments is None:
            num_segments = tf.cast(tf.reduce_max(segment_ids) + 1, tf.int32)
        return tf.math.unsorted_segment_max(data, segment_ids, num_segments)


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_min(data, segment_ids)
    else:
        if num_segments is None:
            num_segments = tf.cast(tf.reduce_max(segment_ids) + 1, tf.int32)
        return tf.math.unsorted_segment_min(data, segment_ids, num_segments)


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_prod(data, segment_ids)
    else:
        if num_segments is None:
            num_segments = tf.cast(tf.reduce_max(segment_ids) + 1, tf.int32)
        return tf.math.unsorted_segment_prod(data, segment_ids, num_segments)


def top_k(x, k, sorted=True, is_stable=True):
    return tf.math.top_k(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    # Use `np.ndim` (rather than `predictions.shape`) so this also works for
    # plain lists/tuples, which don't have a `.shape` attribute.
    if np.ndim(predictions) > 2:
        targets = convert_to_tensor(targets)
        predictions = convert_to_tensor(predictions)
        original_shape = tf.shape(targets)
        predictions = tf.reshape(predictions, [-1, tf.shape(predictions)[-1]])
        targets = tf.reshape(targets, [-1])
        result = tf.math.in_top_k(targets, predictions, k)
        return tf.reshape(result, original_shape)
    return tf.math.in_top_k(targets, predictions, k)


def logsumexp(x, axis=None, keepdims=False):
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def cdist(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype, float)
    x = cast(x, dtype)
    y = cast(y, dtype)
    if x.shape.rank < 2 or y.shape.rank < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    diff = tf.expand_dims(x, -2) - tf.expand_dims(y, -3)
    return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))


def extract_sequences(x, sequence_length, sequence_stride):
    return tf.signal.frame(
        x,
        frame_length=sequence_length,
        frame_step=sequence_stride,
        axis=-1,
        pad_end=False,
    )


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not real.dtype.is_floating or not imag.dtype.is_floating:
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = tf.dtypes.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft2d(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def ifft2(x):
    real, imag = x
    h = cast(tf.shape(real)[-2], real.dtype)
    w = cast(tf.shape(real)[-1], real.dtype)
    real_conj, imag_conj = real, -imag
    fft_real, fft_imag = fft2((real_conj, imag_conj))
    return fft_real / (h * w), -fft_imag / (h * w)


def rfft(x, fft_length=None):
    if fft_length is not None:
        fft_length = [fft_length]
    complex_output = tf.signal.rfft(x, fft_length=fft_length)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    if fft_length is not None:
        fft_length = [fft_length]
    return tf.signal.irfft(complex_input, fft_length)


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )
    x = convert_to_tensor(x)

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = tf.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=x.dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])

        def win(frame_step, dtype):
            return win_array

    else:
        win = None

    result = tf.signal.stft(
        x,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )
    return tf.math.real(result), tf.math.imag(result)


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    complex_input = _get_complex_tensor_from_tuple(x)
    dtype = tf.math.real(complex_input).dtype

    expected_output_len = fft_length + sequence_stride * (
        tf.shape(complex_input)[-2] - 1
    )
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])
        win = tf.signal.inverse_stft_window_fn(
            sequence_stride, lambda frame_step, dtype: win_array
        )
    else:
        win = None

    x = tf.signal.inverse_stft(
        complex_input,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = expected_output_len - (fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def rsqrt(x):
    x = convert_to_tensor(x)
    return tf.math.rsqrt(x)


def erf(x):
    return tf.math.erf(x)


def erfc(x):
    x = convert_to_tensor(x)
    return tf.math.erfc(x)


def erfinv(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)

    if dtype in ["bfloat16", "float16"]:
        return tf.cast(
            tf.math.erfinv(tf.cast(x, tf.float32)),
            dtype,
        )

    if dtype == "float32":
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        clamped_x = tf.clip_by_value(x, -1.0 + epsilon, 1.0 - epsilon)
        x = tf.where(tf.abs(x) < 1.0, clamped_x, x)

    return tf.math.erfinv(x)


def logdet(x):
    x = convert_to_tensor(x)
    return tf.linalg.logdet(x)


def gammainc(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)

    compute_dtype = dtype
    if standardize_dtype(dtype) in ("float16", "bfloat16"):
        compute_dtype = "float32"
    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)

    return cast(tf.math.igamma(x1, x2), dtype)


def lgamma(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    if standardize_dtype(dtype) == "bfloat16":
        return cast(tf.math.lgamma(cast(x, "float32")), dtype)
    if not tf.as_dtype(x.dtype).is_floating:
        x = cast(x, dtype)
    return tf.math.lgamma(x)
