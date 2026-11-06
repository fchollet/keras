import numpy as np

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.jax.ops.math import fft as jax_fft
from keras.src.backend.jax.ops.math import fft2 as jax_fft2
from keras.src.backend.numpy.ops.core import convert_to_tensor
from keras.src.utils.module_utils import scipy


def _segment_reduction_fn(
    data, segment_ids, reduction_method, num_segments, sorted
):
    if num_segments is None:
        num_segments = np.amax(segment_ids) + 1

    valid_indices = segment_ids >= 0  # Ignore segment_ids that are -1
    valid_data = data[valid_indices]
    valid_segment_ids = segment_ids[valid_indices]

    data_shape = list(valid_data.shape)
    data_shape[0] = (
        num_segments  # Replace first dimension (which corresponds to segments)
    )

    if reduction_method == np.maximum:
        result = np.ones(data_shape, dtype=valid_data.dtype) * -np.inf
    elif reduction_method == np.minimum:
        result = np.ones(data_shape, dtype=valid_data.dtype) * np.inf
    elif reduction_method == np.multiply:
        result = np.ones(data_shape, dtype=valid_data.dtype)
    else:
        result = np.zeros(data_shape, dtype=valid_data.dtype)

    if sorted:
        reduction_method.at(result, valid_segment_ids, valid_data)
    else:
        sort_indices = np.argsort(valid_segment_ids)
        sorted_segment_ids = valid_segment_ids[sort_indices]
        sorted_data = valid_data[sort_indices]

        reduction_method.at(result, sorted_segment_ids, sorted_data)

    return result


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, np.add, num_segments, sorted
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, np.maximum, num_segments, sorted
    )


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, np.minimum, num_segments, sorted
    )


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, np.multiply, num_segments, sorted
    )


def top_k(x, k, sorted=True, is_stable=True):
    if not is_stable and not sorted:
        # Partition the array such that all values larger than the k-th
        # largest value are to the right of it.
        top_k_indices = np.argpartition(x, -k, axis=-1)[..., -k:]
        top_k_values = np.take_along_axis(x, top_k_indices, axis=-1)
        return top_k_values, top_k_indices
    if not is_stable and sorted:
        # Take the k largest values.
        sorted_indices = np.argsort(x, axis=-1)[..., ::-1]
        sorted_values = np.take_along_axis(x, sorted_indices, axis=-1)
        return sorted_values[..., :k], sorted_indices[..., :k]
    # Universal Reverse-Argsort-Reverse stable sort:
    x_rev = np.flip(x, axis=-1)
    rev_indices = np.argsort(x_rev, axis=-1, kind="stable")[..., ::-1]
    orig_indices = x.shape[-1] - 1 - rev_indices
    values = np.take_along_axis(x, orig_indices, axis=-1)
    return values[..., :k], orig_indices[..., :k]


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets)
    predictions = convert_to_tensor(predictions)
    targets = targets[..., None]
    topk_values = top_k(predictions, k, sorted=False, is_stable=False)[0]
    targets_values = np.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return np.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    return scipy.special.logsumexp(x, axis=axis, keepdims=keepdims).astype(
        dtype
    )


def cdist(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    dtype = dtypes.result_type(x.dtype, y.dtype, float)
    x = x.astype(dtype)
    y = y.astype(dtype)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    diff = x[..., :, None, :] - y[..., None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def extract_sequences(x, sequence_length, sequence_stride):
    *batch_shape, _ = x.shape
    batch_shape = list(batch_shape)
    shape = x.shape[:-1] + (
        (x.shape[-1] - (sequence_length - sequence_stride)) // sequence_stride,
        sequence_length,
    )
    strides = x.strides[:-1] + (
        sequence_stride * x.strides[-1],
        x.strides[-1],
    )
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.reshape(x, (*batch_shape, *x.shape[-2:]))


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not np.issubdtype(real.dtype, np.floating) or not np.issubdtype(
        imag.dtype, np.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = real + 1j * imag
    return complex_input


def fft(x):
    real, imag = jax_fft(x)
    return np.array(real), np.array(imag)


def fft2(x):
    real, imag = jax_fft2(x)
    return np.array(real), np.array(imag)


def ifft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = np.fft.ifft2(complex_input)
    return np.real(complex_output), np.imag(complex_output)


def rfft(x, fft_length=None):
    complex_output = np.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    # numpy always outputs complex128, so we need to recast the dtype
    return (
        np.real(complex_output).astype(x.dtype),
        np.imag(complex_output).astype(x.dtype),
    )


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    # numpy always outputs float64, so we need to recast the dtype
    return np.fft.irfft(
        complex_input, n=fft_length, axis=-1, norm="backward"
    ).astype(x[0].dtype)


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
    ori_dtype = x.dtype

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = np.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
            )
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])
    else:
        win = np.ones((sequence_length + l_pad + r_pad), dtype=x.dtype)

    x = scipy.signal.stft(
        x,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=None,
        padded=False,
    )[-1]

    # scale and swap to (..., num_sequences, fft_bins)
    x = x / np.sqrt(1.0 / win.sum() ** 2)
    x = np.swapaxes(x, -2, -1)
    return np.real(x).astype(ori_dtype), np.imag(x).astype(ori_dtype)


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    x = _get_complex_tensor_from_tuple(x)
    dtype = np.real(x).dtype

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=dtype
            )
        else:
            win = convert_to_tensor(window, dtype=dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])
    else:
        win = np.ones((sequence_length + l_pad + r_pad), dtype=dtype)

    x = scipy.signal.istft(
        x,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=False,
        time_axis=-2,
        freq_axis=-1,
    )[-1]

    # scale
    x = x / win.sum() if window is not None else x / sequence_stride

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = expected_output_len - (fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def rsqrt(x):
    dtype = dtypes.result_type(x.dtype)
    return (1.0 / np.sqrt(x)).astype(dtype)


def erf(x):
    dtype = dtypes.result_type(x.dtype)
    return scipy.special.erf(x).astype(dtype)


def erfc(x):
    dtype = dtypes.result_type(x.dtype, float)
    return scipy.special.erfc(x).astype(dtype)


def erfinv(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    return scipy.special.erfinv(x).astype(dtype)


def logdet(x):
    from keras.src.backend.numpy.ops.numpy import slogdet

    # In NumPy slogdet is more stable than `np.log(np.linalg.det(x))`. See
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html
    return slogdet(x)[1]


def gammainc(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return scipy.special.gammainc(x1, x2).astype(dtype)


def lgamma(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    return scipy.special.gammaln(x).astype(dtype)
