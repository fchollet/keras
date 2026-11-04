import math

import torch

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.torch.ops.core import cast
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.backend.torch.ops.numpy import pad


def _segment_reduction_fn(data, segment_ids, reduction_method, num_segments):
    num_repeats = torch.prod(
        torch.tensor(data.shape[1:], device=get_device())
    ).long()
    # To use `scatter_add` in torch, we need to replicate `segment_ids` into the
    # shape of `data`.
    segment_ids = (
        segment_ids.repeat_interleave(num_repeats)
        .view(*data.shape)
        .type(torch.int64)
    )
    num_segments = num_segments or torch.max(segment_ids) + 1

    # .scatter_add does not support -1 in the indices.
    # Add all out-of-bound indices value to an extra dimension after
    # num_segments, which is removed before returning the result.

    # Replacing the out-of-bound indices.
    segment_ids = torch.where(segment_ids >= 0, segment_ids, num_segments)
    segment_ids = torch.where(
        segment_ids < num_segments, segment_ids, num_segments
    )

    # Add one more dimension to the result shape with the "+1".
    shape = (num_segments + 1,) + tuple(data.shape[1:])

    if reduction_method == "amax":
        result = torch.ones(*shape, device=get_device()) * -float("Inf")
    elif reduction_method == "amin":
        result = torch.ones(*shape, device=get_device()) * float("Inf")
    elif reduction_method == "prod":
        result = torch.ones(*shape, device=get_device())
    else:
        result = torch.zeros(*shape, device=get_device())

    result = result.scatter_reduce(
        0, segment_ids, data.float(), reduction_method
    )

    # Removing the extra dimension.
    result = result[:-1, ...]

    return result.type(data.dtype)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "sum", num_segments)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "amax", num_segments)


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "amin", num_segments)


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "prod", num_segments)


def top_k(x, k, sorted=True, is_stable=True):
    x = convert_to_tensor(x)
    if not is_stable:
        return torch.topk(x, k, sorted=sorted)
    sorted_values, sorted_indices = torch.sort(
        x, dim=-1, descending=True, stable=True
    )
    return sorted_values[..., :k], sorted_indices[..., :k]


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets).type(torch.int64)
    targets = targets.unsqueeze(-1)
    predictions = convert_to_tensor(predictions)
    # Every class is in the top-k when `k` exceeds the number of classes.
    k = min(k, predictions.shape[-1])
    topk_values = top_k(predictions, k, sorted=False, is_stable=False)[0]
    targets_values = torch.take_along_dim(predictions, targets, dim=-1)
    mask = targets_values >= topk_values
    return torch.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.logsumexp(x, dim=axis, keepdim=keepdims)


def cdist(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    # torch.cdist exists but does NOT broadcast batch dims the same way
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    return torch.sqrt(torch.sum(diff * diff, dim=-1))


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    return torch.unfold_copy(
        x, dimension=-1, size=sequence_length, step=sequence_stride
    )


def _overlap_sequences(x, sequence_stride):
    # Ref: https://github.com/google/jax/blob/main/jax/_src/scipy/signal.py
    x = convert_to_tensor(x)
    *batch_shape, num_sequences, sequence_length = x.shape
    if sequence_stride > sequence_length:
        raise ValueError(
            "`sequence_stride` must equal or less than x.shape[-1]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}"
        )
    if sequence_stride < (sequence_length / num_sequences):
        raise ValueError(
            "`sequence_stride` must equal or greater than "
            "x.shape[-1] / x.shape[-2]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}, x.shape[-2]={num_sequences}"
        )
    flat_batchsize = math.prod(batch_shape)
    x = torch.reshape(x, (flat_batchsize, num_sequences, sequence_length))
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
    # Here, we use shorter notation for axes.
    # B: batch_size, N: num_sequences, S: nstep_per_segment,
    # T: sequence_length divided by S
    padded_segment_len = nstep_per_segment * sequence_stride
    x = torch.nn.functional.pad(
        x, (0, padded_segment_len - sequence_length, 0, 0, 0, 0)
    )
    x = torch.reshape(
        x, (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride)
    )
    # For obtaining shifted signals, this routine reinterprets flattened array
    # with a shrinked axis.  With appropriate truncation/ padding, this
    # operation pushes the last padded elements of the previous row to the head
    # of the current row.
    # See implementation of `overlap_and_add` in Tensorflow for details.
    x = torch.permute(x, (0, 2, 1, 3))  # x: (B, S, N, T)
    x = torch.nn.functional.pad(x, (0, 0, 0, num_sequences, 0, 0, 0, 0))
    # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = torch.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = torch.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    # Finally, sum shifted segments, and truncate results to the output_size.
    x = torch.sum(x, dim=1)[:, :output_size]
    return torch.reshape(x, tuple(batch_shape) + (-1,))


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
    # Check shape.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not torch.is_floating_point(real) or not torch.is_floating_point(imag):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )

    complex_input = torch.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = torch.fft.fft(complex_input)
    return complex_output.real, complex_output.imag


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = torch.fft.fft2(complex_input)
    return complex_output.real, complex_output.imag


def ifft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = torch.fft.ifft2(complex_input)
    return complex_output.real, complex_output.imag


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = torch.fft.rfft(x, n=fft_length, dim=-1, norm="backward")
    return complex_output.real, complex_output.imag


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return torch.fft.irfft(complex_input, n=fft_length, dim=-1, norm="backward")


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

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win = torch.hann_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
            else:
                win = torch.hamming_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
    else:
        win = torch.ones((sequence_length,), dtype=x.dtype, device=get_device())

    need_unpack = False
    *batch_shape, samples = x.shape
    if len(x.shape) > 2:
        need_unpack = True
        flat_batchsize = math.prod(batch_shape)
        x = torch.reshape(x, (flat_batchsize, samples))

    x = torch.stft(
        x,
        n_fft=fft_length,
        hop_length=sequence_stride,
        win_length=sequence_length,
        window=win,
        center=center,
        return_complex=True,
    )
    if need_unpack:
        fft_unique_bins, num_sequences = x.shape[-2:]
        x = torch.reshape(x, (*batch_shape, fft_unique_bins, num_sequences))

    x = torch.swapaxes(x, -2, -1)
    return x.real, x.imag


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
    dtype = complex_input.real.dtype
    win = None
    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win = torch.hann_window(
                    sequence_length,
                    periodic=True,
                    dtype=dtype,
                    device=get_device(),
                )
            else:
                win = torch.hamming_window(
                    sequence_length,
                    periodic=True,
                    dtype=dtype,
                    device=get_device(),
                )
        else:
            win = convert_to_tensor(window, dtype=dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )

    if (
        sequence_length == fft_length
        and center is True
        and win is not None
        and fft_length > 1
    ):
        # can be fallen back to torch.istft
        need_unpack = False
        *batch_shape, num_sequences, fft_unique_bins = complex_input.shape
        if len(complex_input.shape) > 3:
            need_unpack = True
            flat_batchsize = math.prod(batch_shape)
            complex_input = torch.reshape(
                complex_input, (flat_batchsize, num_sequences, fft_unique_bins)
            )
        complex_input = torch.swapaxes(complex_input, -2, -1)
        x = torch.istft(
            complex_input,
            n_fft=fft_length,
            hop_length=sequence_stride,
            win_length=sequence_length,
            window=win,
            center=center,
            length=length,
            return_complex=False,
        )
        if need_unpack:
            samples = x.shape[-1]
            x = torch.reshape(x, (*batch_shape, samples))
        return x

    # custom implementation with irfft and _overlap_sequences
    # references:
    # torch: aten/src/ATen/native/SpectralOps.cpp
    # tf: tf.signal.inverse_stft_window_fn
    x = irfft(x, fft_length)

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)

    if win is not None:
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = pad(win, [[l_pad, r_pad]], "constant")

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = torch.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = pad(denom, [(0, overlaps * sequence_stride - _sequence_length)])
        denom = torch.reshape(denom, [overlaps, sequence_stride])
        denom = torch.sum(denom, 0, keepdims=True)
        denom = torch.tile(denom, [overlaps, 1])
        denom = torch.reshape(denom, [overlaps * sequence_stride])
        win = torch.divide(win, denom[:_sequence_length])
        x = torch.multiply(x, win)

    x = _overlap_sequences(x, sequence_stride)

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
    return torch.rsqrt(x)


def erf(x):
    x = convert_to_tensor(x)
    return torch.erf(x)


def erfc(x):
    x = convert_to_tensor(x)
    return torch.erfc(x)


def erfinv(x):
    x = convert_to_tensor(x)
    return torch.erfinv(x)


def logdet(x):
    x = convert_to_tensor(x)
    return torch.logdet(x)


def gammainc(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)

    compute_dtype = dtype
    if standardize_dtype(dtype) in ("float16", "bfloat16"):
        compute_dtype = "float32"
    x1 = cast(x1, compute_dtype)
    x2 = cast(x2, compute_dtype)
    return cast(torch.special.gammainc(x1, x2), dtype)


def lgamma(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    compute_dtype = dtype
    if standardize_dtype(dtype) in ("float16", "bfloat16"):
        compute_dtype = "float32"
    x = cast(x, compute_dtype)
    return cast(torch.lgamma(x), dtype)
