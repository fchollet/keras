import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
import scipy.special
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import math as kmath
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product


def _stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    # pure numpy version of stft that matches librosa's implementation
    x = np.array(x)
    ori_dtype = x.dtype

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = np.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            window = scipy.signal.get_window(window, sequence_length)
        win = np.array(window, dtype=x.dtype)
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


def _istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    # pure numpy version of istft that matches librosa's implementation
    complex_input = x[0] + 1j * x[1]
    x = np.fft.irfft(
        complex_input, n=fft_length, axis=-1, norm="backward"
    ).astype(x[0].dtype)

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)

    if window is not None:
        if isinstance(window, str):
            win = np.array(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
            )
        else:
            win = np.array(window, dtype=x.dtype)
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = np.pad(win, [[l_pad, r_pad]])

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = np.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = np.pad(
            denom, [(0, overlaps * sequence_stride - _sequence_length)]
        )
        denom = np.reshape(denom, [overlaps, sequence_stride])
        denom = np.sum(denom, 0, keepdims=True)
        denom = np.tile(denom, [overlaps, 1])
        denom = np.reshape(denom, [overlaps * sequence_stride])
        win = np.divide(win, denom[:_sequence_length])
        x = np.multiply(x, win)

    # overlap_sequences
    def _overlap_sequences(x, sequence_stride):
        *batch_shape, num_sequences, sequence_length = x.shape
        flat_batchsize = math.prod(batch_shape)
        x = np.reshape(x, (flat_batchsize, num_sequences, sequence_length))
        output_size = sequence_stride * (num_sequences - 1) + sequence_length
        nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
        padded_segment_len = nstep_per_segment * sequence_stride
        x = np.pad(
            x, ((0, 0), (0, 0), (0, padded_segment_len - sequence_length))
        )
        x = np.reshape(
            x,
            (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride),
        )
        x = x.transpose((0, 2, 1, 3))
        x = np.pad(x, ((0, 0), (0, 0), (0, num_sequences), (0, 0)))
        shrinked = x.shape[2] - 1
        x = np.reshape(x, (flat_batchsize, -1))
        x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
        x = np.reshape(
            x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
        )
        x = np.sum(x, axis=1)[:, :output_size]
        return np.reshape(x, tuple(batch_shape) + (-1,))

    x = _overlap_sequences(x, sequence_stride)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center:
        end = expected_output_len - (fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def _sum_reduce(left, right):
    return left + right


def _max_reduce(left, right):
    return np.max(np.stack([left, right]), axis=0)


class MathOpsDynamicShapeTest(testing.TestCase):
    def test_cdist(self):
        x = KerasTensor((None, 2, 3))
        y = KerasTensor((None, 4, 3))

        z = kmath.cdist(x, y)
        self.assertEqual(z.shape, (None, 2, 4))

    def test_erf(self):
        x = KerasTensor((None, 2, 3))
        y = kmath.erf(x)
        self.assertEqual(y.shape, (None, 2, 3))

    def test_erfc(self):
        x = KerasTensor((None, 2, 3))
        y = kmath.erfc(x)
        self.assertEqual(y.shape, (None, 2, 3))

    def test_erfinv(self):
        x = KerasTensor((None, 2, 3))
        y = kmath.erfinv(x)
        self.assertEqual(y.shape, (None, 2, 3))

    def test_gammainc(self):
        x1 = KerasTensor((None, 2, 3))
        x2 = KerasTensor((None, 2, 3))

        z = kmath.gammainc(x1, x2)
        self.assertEqual(z.shape, (None, 2, 3))

        x1 = KerasTensor((None, 2, 3))
        x2 = KerasTensor((1, 2, 3))

        z = kmath.gammainc(x1, x2)
        self.assertEqual(z.shape, (None, 2, 3))

    def test_lgamma(self):
        x = KerasTensor((None, 2, 3))
        y = kmath.lgamma(x)
        self.assertEqual(y.shape, (None, 2, 3))

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    def test_segment_reduce(self, segment_reduce_op):
        # 1D case
        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = segment_reduce_op(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

        data = KerasTensor((None, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = segment_reduce_op(data, segment_ids, num_segments=5)
        self.assertEqual(outputs.shape, (5, 4))

        data = KerasTensor((10,), dtype="float32")
        segment_ids = KerasTensor(
            (10,),
            dtype="int32",
        )
        outputs = segment_reduce_op(data, segment_ids)
        self.assertEqual(outputs.shape, (None,))

    def test_top_k(self):
        x = KerasTensor((None, 2, 3))
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (None, 2, 1))
        self.assertEqual(indices.shape, (None, 2, 1))

    def test_in_top_k(self):
        targets = KerasTensor((None,))
        predictions = KerasTensor((None, 10))
        self.assertEqual(
            kmath.in_top_k(targets, predictions, k=1).shape, (None,)
        )

    def test_logsumexp(self):
        x = KerasTensor((None, 2, 3), dtype="float32")
        self.assertEqual(kmath.logsumexp(x).shape, ())
        self.assertEqual(kmath.logsumexp(x, axis=1).shape, (None, 3))
        self.assertEqual(kmath.logsumexp(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(kmath.logsumexp(x, keepdims=True).shape, (1, 1, 1))

    def test_extract_sequences(self):
        # Defined dimension
        x = KerasTensor((None, 32), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        self.assertEqual(outputs.shape, (None, num_sequences, sequence_length))

        # Undefined dimension
        x = KerasTensor((None, None), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        self.assertEqual(outputs.shape, (None, None, sequence_length))

    def test_fft(self):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    def test_fft2(self):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    def test_ifft2(self):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.ifft2((real, imag))
        ref = np.fft.ifft2(np.ones((2, 4, 3)))
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    @parameterized.parameters([(None,), (1,), (5,)])
    def test_rfft(self, fft_length):
        x = KerasTensor((None, 4, 3), dtype="float32")
        real_output, imag_output = kmath.rfft(x, fft_length=fft_length)
        ref = np.fft.rfft(np.ones((2, 4, 3)), n=fft_length)
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(real_output.shape, ref_shape)
        self.assertEqual(imag_output.shape, ref_shape)

    @parameterized.parameters([(None,), (1,), (5,)])
    def test_irfft(self, fft_length):
        real = KerasTensor((None, 4, 3), dtype="float32")
        imag = KerasTensor((None, 4, 3), dtype="float32")
        output = kmath.irfft((real, imag), fft_length=fft_length)
        ref = np.fft.irfft(np.ones((2, 4, 3)), n=fft_length)
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(output.shape, ref_shape)

    def test_stft(self):
        x = KerasTensor((None, 32), dtype="float32")
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length
        )
        real_ref, imag_ref = _stft(
            np.ones((2, 32)), sequence_length, sequence_stride, fft_length
        )
        real_ref_shape = (None,) + real_ref.shape[1:]
        imag_ref_shape = (None,) + imag_ref.shape[1:]
        self.assertEqual(real_output.shape, real_ref_shape)
        self.assertEqual(imag_output.shape, imag_ref_shape)

    def test_istft(self):
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real = KerasTensor((None, 32), dtype="float32")
        imag = KerasTensor((None, 32), dtype="float32")
        output = kmath.istft(
            (real, imag), sequence_length, sequence_stride, fft_length
        )
        ref = _istft(
            (np.ones((5, 32)), np.ones((5, 32))),
            sequence_length,
            sequence_stride,
            fft_length,
        )
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(output.shape, ref_shape)

    def test_istft_with_length(self):
        sequence_length = 4
        sequence_stride = 1
        fft_length = 8
        length = 1
        window = "hann"
        center = False
        real = KerasTensor((None, 10, 5), dtype="float32")
        imag = KerasTensor((None, 10, 5), dtype="float32")

        output = kmath.istft(
            (real, imag),
            sequence_length,
            sequence_stride,
            fft_length,
            length,
            window,
            center,
        )

        ref = _istft(
            (np.ones((2, 10, 5)), np.ones((2, 10, 5))),
            sequence_length,
            sequence_stride,
            fft_length,
            length,
            window,
            center,
        )
        ref_shape = (None,) + ref.shape[1:]
        self.assertEqual(output.shape, ref_shape)

        real_dyn = KerasTensor((None, None, 5), dtype="float32")
        imag_dyn = KerasTensor((None, None, 5), dtype="float32")

        output_dyn = kmath.istft(
            (real_dyn, imag_dyn),
            sequence_length,
            sequence_stride,
            fft_length,
            length,
            window,
            center,
        )
        self.assertEqual(output_dyn.shape, (None, 1))

    def test_rsqrt(self):
        x = KerasTensor([None, 3])
        self.assertEqual(kmath.rsqrt(x).shape, (None, 3))

    def test_logdet(self):
        x = KerasTensor((None, 3, 3))
        out = kmath.logdet(x)
        self.assertEqual(out.shape, (None,))


class MathOpsStaticShapeTest(testing.TestCase):
    def test_cdist(self):
        x = KerasTensor((1, 2, 3))
        y = KerasTensor((1, 4, 3))

        z = kmath.cdist(x, y)
        self.assertEqual(z.shape, (1, 2, 4))

    def test_erf(self):
        x = KerasTensor((1, 2, 3))
        y = kmath.erf(x)
        self.assertEqual(y.shape, (1, 2, 3))

    def test_erfc(self):
        x = KerasTensor((1, 2, 3))
        y = kmath.erfc(x)
        self.assertEqual(y.shape, (1, 2, 3))

    def test_erfinv(self):
        x = KerasTensor((1, 2, 3))
        y = kmath.erfinv(x)
        self.assertEqual(y.shape, (1, 2, 3))

    def test_gammainc(self):
        x1 = KerasTensor((1, 2, 3))
        x2 = KerasTensor((1, 2, 3))

        z = kmath.gammainc(x1, x2)
        self.assertEqual(z.shape, (1, 2, 3))

    def test_lgamma(self):
        x = KerasTensor((1, 2, 3))
        y = kmath.lgamma(x)
        self.assertEqual(y.shape, (1, 2, 3))

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_reduce(self, segment_reduce_op):
        # 1D case
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = segment_reduce_op(data, segment_ids)
        self.assertEqual(outputs.shape, (None, 4))

        data = KerasTensor((10,), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = segment_reduce_op(data, segment_ids)
        self.assertEqual(outputs.shape, (None,))

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    def test_segment_reduce_explicit_num_segments(self, segment_reduce_op):
        # 1D case
        data = KerasTensor((10, 4), dtype="float32")
        segment_ids = KerasTensor((10,), dtype="int32")
        outputs = segment_reduce_op(data, segment_ids, num_segments=5)
        self.assertEqual(outputs.shape, (5, 4))

        data = KerasTensor((6,), dtype="float32")
        segment_ids = KerasTensor(
            (6,),
            dtype="int32",
        )
        outputs = segment_reduce_op(data, segment_ids, num_segments=5)
        self.assertEqual(outputs.shape, (5,))

    def test_topk(self):
        x = KerasTensor((1, 2, 3))
        values, indices = kmath.top_k(x, k=1)
        self.assertEqual(values.shape, (1, 2, 1))
        self.assertEqual(indices.shape, (1, 2, 1))

    def test_in_top_k(self):
        targets = KerasTensor((5,))
        predictions = KerasTensor((5, 10))
        self.assertEqual(kmath.in_top_k(targets, predictions, k=1).shape, (5,))

    def test_logsumexp(self):
        x = KerasTensor((1, 2, 3), dtype="float32")
        result = kmath.logsumexp(x)
        self.assertEqual(result.shape, ())

    def test_extract_sequences(self):
        x = KerasTensor((10, 16), dtype="float32")
        sequence_length = 3
        sequence_stride = 2
        outputs = kmath.extract_sequences(x, sequence_length, sequence_stride)
        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        self.assertEqual(outputs.shape, (10, num_sequences, sequence_length))

    def test_fft(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_fft2(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_ifft2(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.ifft2((real, imag))
        ref = np.fft.ifft2(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_rfft(self):
        x = KerasTensor((2, 4, 3), dtype="float32")
        real_output, imag_output = kmath.rfft(x)
        ref = np.fft.rfft(np.ones((2, 4, 3)))
        self.assertEqual(real_output.shape, ref.shape)
        self.assertEqual(imag_output.shape, ref.shape)

    def test_irfft(self):
        real = KerasTensor((2, 4, 3), dtype="float32")
        imag = KerasTensor((2, 4, 3), dtype="float32")
        output = kmath.irfft((real, imag))
        ref = np.fft.irfft(np.ones((2, 4, 3)))
        self.assertEqual(output.shape, ref.shape)

    def test_rsqrt(self):
        x = KerasTensor([4, 3], dtype="float32")
        self.assertEqual(kmath.rsqrt(x).shape, (4, 3))

    def test_stft(self):
        x = KerasTensor((2, 32), dtype="float32")
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length
        )
        real_ref, imag_ref = _stft(
            np.ones((2, 32)), sequence_length, sequence_stride, fft_length
        )
        self.assertEqual(real_output.shape, real_ref.shape)
        self.assertEqual(imag_output.shape, imag_ref.shape)

    def test_istft(self):
        # sequence_stride must <= x[0].shape[-1]
        # sequence_stride must >= fft_length / num_sequences
        sequence_length = 10
        sequence_stride = 3
        fft_length = 15
        num_sequences = fft_length // sequence_stride + 1
        real = KerasTensor((num_sequences, 32), dtype="float32")
        imag = KerasTensor((num_sequences, 32), dtype="float32")
        output = kmath.istft(
            (real, imag), sequence_length, sequence_stride, fft_length
        )
        ref = _istft(
            (np.ones((num_sequences, 32)), np.ones((num_sequences, 32))),
            sequence_length,
            sequence_stride,
            fft_length,
        )
        self.assertEqual(output.shape, ref.shape)

    def test_logdet(self):
        x = KerasTensor((3, 3))
        out = kmath.logdet(x)
        self.assertEqual(out.shape, ())

        x = KerasTensor((2, 4, 3, 3))
        out = kmath.logdet(x)
        self.assertEqual(out.shape, (2, 4))


BACKEND_AGNOSTIC_OPS = [
    {"testcase_name": "backend_specific", "backend_agnostic_ops": False},
    {"testcase_name": "backend_agnostic", "backend_agnostic_ops": True},
]


class MathOpsCorrectnessTest(testing.TestCase):
    def run_segment_reduce_test(
        self,
        segment_reduce_op,
        element_wise_reduce_method,
        num_indices,
        indices_high,
        data_dims=tuple(),
        num_segments=None,
        add_neg1_to_indices=False,
        sorted_indices=False,
    ):
        if num_segments is not None and indices_high >= num_segments:
            raise ValueError("Indices high cannot be more than num segments")
        indices_dims = (num_indices,)
        full_data_dims = indices_dims + data_dims
        data = np.random.rand(*full_data_dims).astype(np.float32)
        segment_ids = np.concatenate(
            [
                np.arange(indices_high),
                np.random.randint(
                    low=0,
                    high=indices_high,
                    size=(indices_dims[0] - indices_high),
                ),
            ]
        ).astype(np.int32)
        if sorted_indices:
            segment_ids = np.sort(segment_ids, axis=-1)
        if add_neg1_to_indices:
            segment_ids[0] = -1
        outputs = segment_reduce_op(
            data, segment_ids, num_segments, sorted=sorted_indices
        )
        if num_segments is None:
            num_segments = np.max(segment_ids).item() + 1
        expected_shape = (num_segments,) + data_dims
        if segment_reduce_op == kmath.segment_max:
            if backend.backend() == "tensorflow":
                empty_fill_value = -np.finfo(np.float32).max
            else:
                empty_fill_value = -np.inf
            expected = np.full(expected_shape, empty_fill_value)
        else:
            expected = np.zeros(expected_shape)

        for idx in range(num_indices):
            segment_id = segment_ids[idx]
            if segment_id == -1:
                continue
            expected[segment_id] = element_wise_reduce_method(
                expected[segment_id], data[idx]
            )
        self.assertAllClose(outputs, expected)

    @parameterized.product(
        (
            dict(
                segment_reduce_op=kmath.segment_sum,
                element_wise_reduce_method=_sum_reduce,
            ),
            dict(
                segment_reduce_op=kmath.segment_max,
                element_wise_reduce_method=_max_reduce,
            ),
        ),
        sorted_indices=(True, False),
    )
    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support `num_segments=None`.",
    )
    def test_segment_reduce(
        self,
        segment_reduce_op,
        element_wise_reduce_method,
        sorted_indices,
    ):
        # Test 1D case.
        self.run_segment_reduce_test(
            segment_reduce_op,
            element_wise_reduce_method,
            num_indices=9,
            indices_high=3,
            sorted_indices=sorted_indices,
        )

        # Test ND data case.
        self.run_segment_reduce_test(
            segment_reduce_op,
            element_wise_reduce_method,
            num_indices=9,
            indices_high=3,
            data_dims=(
                3,
                3,
            ),
            sorted_indices=sorted_indices,
        )

    @parameterized.product(
        (
            dict(
                segment_reduce_op=kmath.segment_sum,
                element_wise_reduce_method=_sum_reduce,
            ),
            dict(
                segment_reduce_op=kmath.segment_max,
                element_wise_reduce_method=_max_reduce,
            ),
        ),
        (
            dict(
                contains_neg1_in_indices=True,
                sorted_indices=False,
            ),
            dict(
                contains_neg1_in_indices=False,
                sorted_indices=False,
            ),
            dict(
                contains_neg1_in_indices=False,
                sorted_indices=True,
            ),
        ),
    )
    def test_segment_reduce_explicit_num_segments(
        self,
        segment_reduce_op,
        element_wise_reduce_method,
        contains_neg1_in_indices,
        sorted_indices,
    ):
        if backend.backend() == "tensorflow" and sorted_indices:
            pytest.skip(
                "Num segments and sorted_indices=True doesn't work for "
                "tensorflow."
            )
        # Test 1D case.
        self.run_segment_reduce_test(
            segment_reduce_op,
            element_wise_reduce_method,
            num_indices=9,
            indices_high=3,
            num_segments=4,
            add_neg1_to_indices=contains_neg1_in_indices,
            sorted_indices=sorted_indices,
        )

        # Test ND data case.
        self.run_segment_reduce_test(
            segment_reduce_op,
            element_wise_reduce_method,
            num_indices=9,
            indices_high=3,
            data_dims=(
                3,
                3,
            ),
            num_segments=4,
            add_neg1_to_indices=contains_neg1_in_indices,
            sorted_indices=sorted_indices,
        )

    def test_top_k(self):
        x = np.array([0, 4, 2, 1, 3, -1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=2)
        self.assertAllClose(values, [4, 3])
        self.assertAllClose(indices, [1, 4])

        x = np.array([0, 4, 2, 1, 3, -1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=2, sorted=False)
        # Any order ok when `sorted=False`.
        self.assertEqual(set(backend.ops.convert_to_numpy(values)), set([4, 3]))
        self.assertEqual(
            set(backend.ops.convert_to_numpy(indices)), set([1, 4])
        )

        x = np.random.rand(5, 5)
        outputs = kmath.top_k(x, k=2)

        expected_values = np.zeros((5, 2))
        expected_indices = np.zeros((5, 2), dtype=np.int32)

        for i in range(x.shape[0]):
            top_k_indices = np.argsort(x[i])[-2:][::-1]
            expected_values[i] = x[i, top_k_indices]
            expected_indices[i] = top_k_indices

        self.assertAllClose(outputs[0], expected_values)
        self.assertAllClose(outputs[1], expected_indices)

    def check_stability(self, values, indices):
        """Helper function to check stability of top_k."""
        values_np = backend.ops.convert_to_numpy(values)
        indices_np = backend.ops.convert_to_numpy(indices)
        is_equal = values_np[..., :-1] == values_np[..., 1:]
        index_increasing = indices_np[..., :-1] < indices_np[..., 1:]
        self.assertTrue(np.all(np.logical_or(~is_equal, index_increasing)))

    # Below tests are specific to stable top_k implementation.
    # We have `top_k` operation defined with `is_stable` argument
    def test_top_k_stability_small(self):
        x = np.array([3, 5, 5, 2, 5, 1], dtype=np.float32)
        values, indices = kmath.top_k(x, k=4, is_stable=True)
        self.assertAllClose(values, [5, 5, 5, 3])
        self.assertAllClose(indices, [1, 2, 4, 0])

    def test_top_k_stability_large_1d(self):
        x = np.random.randint(0, 10, size=5000).astype(np.float32)
        values, indices = kmath.top_k(x, k=2500, is_stable=True)
        self.check_stability(values, indices)

    def test_top_k_stability_large_2d(self):
        x = np.random.randint(0, 5, size=(10, 500)).astype(np.int32)
        values, indices = kmath.top_k(x, k=200, is_stable=True)
        self.check_stability(values, indices)

    def test_top_k_unstable_large(self):
        x = np.random.randint(0, 10, size=1000).astype(np.float32)
        values, indices = kmath.top_k(x, k=500, sorted=True, is_stable=False)
        self.assertEqual(values.shape, (500,))
        self.assertEqual(indices.shape, (500,))

    def test_in_top_k(self):
        targets = np.array([1, 0, 2])
        predictions = np.array(
            [
                [0.1, 0.9, 0.8, 0.8],
                [0.05, 0.95, 0, 1],
                [0.1, 0.8, 0.3, 1],
            ]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=1), [True, False, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2), [True, False, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=3), [True, True, True]
        )

        # Test tie cases.
        targets = np.array([1, 0, 2])
        predictions = np.array(
            [
                [0.1, 0.9, 0.8, 0.8],
                [0.95, 0.95, 0, 0.95],
                [0.1, 0.8, 0.8, 0.95],
            ]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=1), [True, True, False]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2), [True, True, True]
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=3), [True, True, True]
        )

        # Test multi-dimensional targets
        targets = np.array([[1, 0]])
        predictions = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype="float32")
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=1), [[False, False]]
        )
        targets = np.array([[1, 2], [0, 3]])
        predictions = np.array(
            [
                [[0.1, 0.9, 0.8, 0.8], [0.05, 0.95, 0, 1]],
                [[0.9, 0.1, 0.8, 0.8], [0.1, 0.8, 0.3, 1]],
            ],
            dtype="float32",
        )
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2),
            [[True, False], [True, True]],
        )

        # Test `nan` in predictions
        # https://github.com/keras-team/keras/issues/19995
        targets = np.array([1, 0])
        predictions = np.array([[0.1, np.nan, 0.5], [0.3, 0.2, 0.5]])
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2), [False, True]
        )

        # Test multi-dimensional list (not array/tensor) inputs.
        targets = [[1, 2], [0, 3]]
        predictions = [
            [[0.1, 0.9, 0.8, 0.8], [0.05, 0.95, 0, 1]],
            [[0.9, 0.1, 0.8, 0.8], [0.1, 0.8, 0.3, 1]],
        ]
        self.assertAllEqual(
            kmath.in_top_k(targets, predictions, k=2),
            [[True, False], [True, True]],
        )

    def test_logsumexp(self):
        x = np.random.rand(5, 5)
        outputs = kmath.logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        self.assertAllClose(outputs, expected)

        outputs = kmath.logsumexp(x, axis=1)
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertAllClose(outputs, expected)

    def test_extract_sequences(self):
        # Test 1D case.
        x = np.random.random((10,))
        sequence_length = 3
        sequence_stride = 2
        output = kmath.extract_sequences(x, sequence_length, sequence_stride)

        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        expected = np.zeros(shape=(num_sequences, sequence_length))
        pos = 0
        for i in range(num_sequences):
            expected[i] = x[pos : pos + sequence_length]
            pos += sequence_stride
        self.assertAllClose(output, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

        # Test N-D case.
        x = np.random.random((4, 8))
        sequence_length = 3
        sequence_stride = 2
        output = kmath.extract_sequences(x, sequence_length, sequence_stride)

        num_sequences = 1 + (x.shape[-1] - sequence_length) // sequence_stride
        expected = np.zeros(shape=(4, num_sequences, sequence_length))
        pos = 0
        for i in range(num_sequences):
            expected[:, i] = x[:, pos : pos + sequence_length]
            pos += sequence_stride
        self.assertAllClose(output, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    def test_fft(self):
        real = np.random.random((2, 4, 3))
        imag = np.random.random((2, 4, 3))
        complex_arr = real + 1j * imag

        real_output, imag_output = kmath.fft((real, imag))
        ref = np.fft.fft(complex_arr)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

    def test_fft2(self):
        real = np.random.random((2, 4, 3))
        imag = np.random.random((2, 4, 3))
        complex_arr = real + 1j * imag

        real_output, imag_output = kmath.fft2((real, imag))
        ref = np.fft.fft2(complex_arr)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

    def test_ifft2(self):
        real = np.random.random((2, 4, 3)).astype(np.float32)
        imag = np.random.random((2, 4, 3)).astype(np.float32)
        complex_arr = real + 1j * imag

        real_output, imag_output = kmath.ifft2((real, imag))
        ref = np.fft.ifft2(complex_arr)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

    @parameterized.parameters([(None,), (3,), (15,)])
    def test_rfft(self, n):
        # Test 1D.
        x = np.random.random((10,))
        real_output, imag_output = kmath.rfft(x, fft_length=n)
        ref = np.fft.rfft(x, n=n)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 10))
        real_output, imag_output = kmath.rfft(x, fft_length=n)
        ref = np.fft.rfft(x, n=n)
        real_ref = np.real(ref)
        imag_ref = np.imag(ref)
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

    @parameterized.parameters([(None,), (3,), (15,)])
    def test_irfft(self, n):
        # Test 1D.
        real = np.random.random((10,))
        imag = np.random.random((10,))
        complex_arr = real + 1j * imag
        output = kmath.irfft((real, imag), fft_length=n)
        ref = np.fft.irfft(complex_arr, n=n)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        real = np.random.random((2, 3, 10))
        imag = np.random.random((2, 3, 10))
        complex_arr = real + 1j * imag
        output = kmath.irfft((real, imag), fft_length=n)
        ref = np.fft.irfft(complex_arr, n=n)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        [
            (32, 8, 32, "hann", True),
            (8, 8, 16, "hann", True),
            (4, 4, 7, "hann", True),
            (32, 8, 32, "hamming", True),
            (32, 8, 32, "hann", False),
            (32, 8, 32, np.ones((32,)), True),
            (32, 8, 32, None, True),
        ]
    )
    def test_stft(
        self, sequence_length, sequence_stride, fft_length, window, center
    ):
        # Test 1D case.
        x = np.random.random((32,))
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        real_ref, imag_ref = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 32))
        real_output, imag_output = kmath.stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        real_ref, imag_ref = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        self.assertAllClose(real_output, real_ref, atol=1e-5, rtol=1e-5)
        self.assertAllClose(imag_output, imag_ref, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        [
            (32, 8, 32, "hann", True),
            (8, 8, 16, "hann", True),
            (4, 4, 7, "hann", True),
            (32, 8, 32, "hamming", True),
            (8, 4, 8, "hann", False),
            (32, 8, 32, np.ones((32,)), True),
            (32, 8, 32, None, True),
            (1, 1, 1, "hann", True),
            (1, 1, 1, None, True),
        ]
    )
    def test_istft(
        self, sequence_length, sequence_stride, fft_length, window, center
    ):
        # sequence_stride must <= x[0].shape[-1]
        # sequence_stride must >= fft_length / num_sequences
        # Test 1D case.
        x = np.random.random((256,))
        real_x, imag_x = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        output = kmath.istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        ref = _istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        if backend.backend() in ("numpy", "jax", "torch"):
            # these backends have different implementation for the boundary of
            # the output, so we need to truncate 5% before assertAllClose
            truncated_len = int(output.shape[-1] * 0.05)
            output = output[..., truncated_len:-truncated_len]
            ref = ref[..., truncated_len:-truncated_len]
        # Nans are handled differently in different backends, so zero them out.
        output = np.nan_to_num(backend.ops.convert_to_numpy(output), nan=0.0)
        ref = np.nan_to_num(ref, nan=0.0)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

        # Test N-D case.
        x = np.random.random((2, 3, 256))
        real_x, imag_x = _stft(
            x, sequence_length, sequence_stride, fft_length, window, center
        )
        output = kmath.istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        ref = _istft(
            (real_x, imag_x),
            sequence_length,
            sequence_stride,
            fft_length,
            window=window,
            center=center,
        )
        if backend.backend() in ("numpy", "jax", "torch"):
            # these backends have different implementation for the boundary of
            # the output, so we need to truncate 5% before assertAllClose
            truncated_len = int(output.shape[-1] * 0.05)
            output = output[..., truncated_len:-truncated_len]
            ref = ref[..., truncated_len:-truncated_len]
        # Nans are handled differently in different backends, so zero them out.
        output = np.nan_to_num(backend.ops.convert_to_numpy(output), nan=0.0)
        ref = np.nan_to_num(ref, nan=0.0)
        self.assertAllClose(output, ref, atol=1e-5, rtol=1e-5)

    def test_rsqrt(self):
        x = np.array([[1, 4, 9], [16, 25, 36]], dtype="float32")
        self.assertAllClose(kmath.rsqrt(x), 1 / np.sqrt(x))
        self.assertAllClose(kmath.Rsqrt()(x), 1 / np.sqrt(x))

    def test_erf_operation_basic(self):
        # Sample values for testing
        sample_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        # Expected output using numpy's approximation of the error function
        expected_output = scipy.special.erf(sample_values)

        # Output from the erf operation in keras_core
        output_from_erf_op = kmath.erf(sample_values)

        # Assert that the outputs are close
        self.assertAllClose(output_from_erf_op, expected_output, atol=1e-4)

    def test_erf_operation_dtype(self):
        # Test for float32 and float64 data types
        for dtype in ("float32", "float64"):
            sample_values = np.array(
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype
            )
            expected_output = scipy.special.erf(sample_values)
            output_from_erf_op = kmath.erf(sample_values)
            self.assertAllClose(output_from_erf_op, expected_output, atol=1e-4)

    def test_erf_operation_edge_cases(self):
        # Test for edge cases
        edge_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float64)
        expected_output = scipy.special.erf(edge_values)
        output_from_edge_erf_op = kmath.erf(edge_values)
        self.assertAllClose(output_from_edge_erf_op, expected_output, atol=1e-4)

    def test_erfc_operation_basic(self):
        sample_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        expected_output = scipy.special.erfc(sample_values)
        output_from_erfc_op = kmath.erfc(sample_values)

        self.assertAllClose(output_from_erfc_op, expected_output, atol=1e-4)

    def test_erfc_operation_edge_cases(self):
        edge_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float64)
        expected_output = scipy.special.erfc(edge_values)
        output_from_edge_erfc_op = kmath.erfc(edge_values)
        self.assertAllClose(
            output_from_edge_erfc_op, expected_output, atol=1e-4
        )

    def test_erfinv_operation_basic(self):
        # Sample values for testing
        sample_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        # Expected output using numpy's approximation of the error function
        expected_output = scipy.special.erfinv(sample_values)

        # Output from the erf operation in keras_core
        output_from_erfinv_op = kmath.erfinv(sample_values)

        # Assert that the outputs are close
        self.assertAllClose(output_from_erfinv_op, expected_output, atol=1e-4)

    def test_erfinv_operation_dtype(self):
        # Test for float32 and float64 data types
        for dtype in ("float32", "float64"):
            sample_values = np.array(
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype
            )
            expected_output = scipy.special.erfinv(sample_values)
            output_from_erfinv_op = kmath.erfinv(sample_values)
            self.assertAllClose(
                output_from_erfinv_op, expected_output, atol=1e-4
            )

    def test_erfinv_operation_edge_cases(self):
        # Test for edge cases
        edge_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float64)
        expected_output = scipy.special.erfinv(edge_values)
        output_from_edge_erfinv_op = kmath.erfinv(edge_values)
        self.assertAllClose(
            output_from_edge_erfinv_op, expected_output, atol=1e-4
        )

        # Inputs extremely close to (but not equal to) `1.0` and `-1.0` must
        # produce a finite value (see keras-team/keras#23133).
        near_ones = np.array(
            [
                np.nextafter(np.float32(1.0), np.float32(0.0)),
                np.nextafter(np.float32(-1.0), np.float32(0.0)),
            ],
            dtype="float32",
        )
        expected_output = scipy.special.erfinv(near_ones.astype("float64"))
        output_from_edge_erfinv_op = kmath.erfinv(near_ones)
        self.assertAllClose(
            output_from_edge_erfinv_op, expected_output, atol=0.1
        )

    def test_logdet(self):
        x = np.array(
            [
                [4.42, -1.18, 0.06, 0.74],
                [-1.18, 1.77, -0.84, -1.16],
                [0.06, -0.84, 5.84, 0.55],
                [0.74, -1.16, 0.55, 0.77],
            ],
            dtype="float32",
        )
        out = kmath.logdet(x)
        self.assertAllClose(out, -1.1178946, atol=1e-3)

    def test_cdist_basic(self):
        x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype="float32")
        y = np.array([[1.0, 0.0]], dtype="float32")
        out = kmath.cdist(x, y)
        expected = np.array([[1.0], [1.0]], dtype="float32")
        self.assertAllClose(out, expected)
        self.assertEqual(out.shape, (2, 1))

    def test_cdist_invalid_last_dim(self):
        x = np.random.rand(3, 4)
        y = np.random.rand(5, 5)
        with self.assertRaises(ValueError):
            kmath.cdist(x, y)

    def test_cdist_symbolic(self):
        x = KerasTensor(shape=(None, 2), dtype="float32")
        y = KerasTensor(shape=(1, 2), dtype="float32")
        out = kmath.cdist(x, y)
        self.assertEqual(out.shape, (None, 1))

    def test_gammainc(self):
        import scipy.special

        x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        x2 = np.array([[0.5, 1.0], [2.0, 5.0]], dtype="float32")

        out = kmath.gammainc(x1, x2)
        expected = scipy.special.gammainc(x1, x2)

        self.assertAllClose(out, expected)
        self.assertEqual(out.shape, (2, 2))

    @parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))
    def test_lgamma_operation_dtype(self, backend_agnostic_ops):
        backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
        self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
        for dtype in ("float32", "float64"):
            sample_values = np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 10.0],
                dtype=dtype,
            )
            expected_output = scipy.special.gammaln(sample_values)
            output_from_lgamma_op = kmath.lgamma(sample_values)
            self.assertAllClose(
                output_from_lgamma_op, expected_output, atol=1e-4
            )

    @parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))
    def test_lgamma_operation_edge_cases(self, backend_agnostic_ops):
        backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
        self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
        edge_values = np.array(
            [-0.5, -1.5, -2.5, 1e-5, 50.0, 100.0, float("inf")],
            dtype=np.float64,
        )
        expected_output = scipy.special.gammaln(edge_values)
        output_from_edge_lgamma_op = kmath.lgamma(edge_values)
        self.assertAllClose(
            output_from_edge_lgamma_op, expected_output, atol=1e-4
        )

    @parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))
    def test_lgamma_operation_basic(self, backend_agnostic_ops):
        backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
        self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
        sample_values = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 10.0]
        )
        expected_output = scipy.special.gammaln(sample_values)
        output_from_lgamma_op = kmath.lgamma(sample_values)
        self.assertAllClose(output_from_lgamma_op, expected_output, atol=1e-4)


class MathDtypeTest(testing.TestCase):
    """Test the floating dtype to verify that the behavior matches JAX."""

    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x
        not in (
            "string",
            "complex64",
            "complex128",
            # Remove 64-bit dtypes.
            "float64",
            "uint64",
            "int64",
        )
        + dtypes.FLOAT8_TYPES  # Remove float8 dtypes for the following tests
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]

    if backend.backend() == "torch":
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint16", "uint32")]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]

    @parameterized.named_parameters(
        named_product(
            dtypes=list(itertools.combinations(FLOAT_DTYPES + INT_DTYPES, 2))
        )
    )
    def test_cdist(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes

        x = knp.ones((2, 3), dtype=dtype1)
        y = knp.ones((4, 3), dtype=dtype2)

        x_jax = jnp.ones((2, 3), dtype=dtype1)
        y_jax = jnp.ones((4, 3), dtype=dtype2)

        expected_dtype = standardize_dtype(
            jnp.linalg.norm(
                x_jax[:, None, :] - y_jax[None, :, :],
                axis=-1,
            ).dtype
        )

        self.assertEqual(
            standardize_dtype(kmath.cdist(x, y).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.CDist().symbolic_call(x, y).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_erf(self, dtype):
        import jax.lax as lax
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(lax.erf(x_jax).dtype)

        self.assertEqual(standardize_dtype(kmath.erf(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(kmath.Erf().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_erfc(self, dtype):
        import jax.numpy as jnp
        import jax.scipy.special as special

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(special.erfc(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(kmath.erfc(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.Erfc().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_erfinv(self, dtype):
        import jax.numpy as jnp
        import jax.scipy.special as special

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(special.erfinv(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(kmath.erfinv(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.Erfinv().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            dtypes=list(itertools.combinations(FLOAT_DTYPES + INT_DTYPES, 2))
        )
    )
    def test_gammainc(self, dtypes):
        import jax.numpy as jnp
        import jax.scipy.special

        dtype1, dtype2 = dtypes

        x1 = knp.ones((2, 3), dtype=dtype1)
        x2 = knp.ones((2, 3), dtype=dtype2)

        x1_jax = jnp.ones((2, 3), dtype=dtype1)
        x2_jax = jnp.ones((2, 3), dtype=dtype2)

        expected_dtype = standardize_dtype(
            jax.scipy.special.gammainc(x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(kmath.gammainc(x1, x2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.Gammainc().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(BACKEND_AGNOSTIC_OPS, dtype=FLOAT_DTYPES)
    )
    def test_lgamma(self, backend_agnostic_ops, dtype):
        import jax.lax as lax
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(lax.lgamma(x_jax).dtype)

        backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
        self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
        self.assertEqual(
            standardize_dtype(kmath.lgamma(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(kmath.Lgamma().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_logsumexp(self, dtype):
        import jax.numpy as jnp
        import jax.scipy.special as jsp

        x = knp.ones((2, 3), dtype=dtype)
        x_jax = jnp.ones((2, 3), dtype=dtype)

        expected_dtype = standardize_dtype(jsp.logsumexp(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(kmath.logsumexp(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.Logsumexp().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_rsqrt(self, dtype):
        import jax.lax as lax

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(lax.rsqrt(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(kmath.rsqrt(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(kmath.Rsqrt().symbolic_call(x).dtype),
            expected_dtype,
        )


class ExtractSequencesOpTest(testing.TestCase):
    def test_extract_sequences_init_length_1_stride_1(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=1, sequence_stride=1
        )
        self.assertIsNotNone(extract_op)
        self.assertEqual(extract_op.sequence_length, 1)
        self.assertEqual(extract_op.sequence_stride, 1)

    def test_extract_sequences_init_length_5_stride_2(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=5, sequence_stride=2
        )
        self.assertIsNotNone(extract_op)
        self.assertEqual(extract_op.sequence_length, 5)
        self.assertEqual(extract_op.sequence_stride, 2)

    def test_compute_output_spec_low_rank(self):
        extract_op = kmath.ExtractSequences(
            sequence_length=5, sequence_stride=1
        )
        low_rank_input = np.array(42)
        error_message = r"Input should have rank >= 1. Received: .*"
        with self.assertRaisesRegex(ValueError, error_message):
            extract_op.compute_output_spec(low_rank_input)

    def test_extract_sequences_call(self):
        sequence_length, sequence_stride = 5, 2
        extract_op = kmath.ExtractSequences(sequence_length, sequence_stride)
        test_input = np.random.rand(10, 20)
        result = extract_op.call(test_input)

        expected_shape = self.calculate_expected_shape(
            test_input.shape, sequence_length, sequence_stride
        )
        self.assertEqual(result.shape, expected_shape)

    def calculate_expected_shape(
        self, input_shape, sequence_length, sequence_stride
    ):
        num_sequences = (
            (input_shape[1] - sequence_length) // sequence_stride
        ) + 1
        return (input_shape[0], num_sequences, sequence_length)


class SegmentSumTest(testing.TestCase):
    def test_segment_sum_call(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)
        num_segments = 2
        sorted_segments = False
        segment_sum_op = kmath.SegmentSum(
            num_segments=num_segments, sorted=sorted_segments
        )
        output = segment_sum_op.call(data, segment_ids)
        expected_output = np.array([[5, 7, 9], [7, 8, 9]], dtype=np.float32)
        self.assertAllClose(output, expected_output)


class SegmentMaxTest(testing.TestCase):
    def test_segment_max_call(self):
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)
        num_segments = 2
        sorted_segments = False
        segment_max_op = kmath.SegmentMax(
            num_segments=num_segments, sorted=sorted_segments
        )
        output = segment_max_op.call(data, segment_ids)
        expected_output = np.array([[2, 5, 8], [3, 6, 9]], dtype=np.float32)
        self.assertAllClose(output, expected_output)


class SegmentMinTest(testing.TestCase):
    def test_segment_min_call(self):
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)
        num_segments = 2
        sorted_segments = False

        segment_min_op = kmath.SegmentMin(
            num_segments=num_segments, sorted=sorted_segments
        )

        output = segment_min_op.call(data, segment_ids)
        expected_output = np.array([[1, 4, 7], [3, 6, 9]], dtype=np.float32)
        self.assertAllClose(output, expected_output)


class SegmentProdTest(testing.TestCase):
    def test_segment_prod_call(self):
        data = np.array([[1, 4, 7], [3, 6, 9], [2, 5, 8]], dtype=np.float32)
        segment_ids = np.array([0, 1, 0], dtype=np.int32)

        segment_prod_op = kmath.SegmentProd(num_segments=2, sorted=False)

        output = segment_prod_op.call(data, segment_ids)
        expected_output = np.array(
            [[2, 20, 56], [3, 6, 9]],
            dtype=np.float32,
        )
        self.assertAllClose(output, expected_output)

    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="Argument `num_segments` cannot be set when sorted is True "
        f"when using the {backend.backend()}",
    )
    def test_segment_prod_call_sorted(self):
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)

        segment_prod_op = kmath.SegmentProd(num_segments=2, sorted=True)

        output = segment_prod_op.call(data, segment_ids)
        expected_output = np.array(
            [[2, 20, 56], [3, 6, 9]],
            dtype=np.float32,
        )
        self.assertAllClose(output, expected_output)

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="Argument `num_segments` must be set "
        f"when using the {backend.backend()}",
    )
    def test_segment_prod_call_sorted_without_num_segments(self):
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int32)

        segment_prod_op = kmath.SegmentProd(sorted=True)

        output = segment_prod_op.call(data, segment_ids)
        expected_output = np.array(
            [[2, 20, 56], [3, 6, 9]],
            dtype=np.float32,
        )
        self.assertAllClose(output, expected_output)


class TopKTest(testing.TestCase):
    def test_top_k_call_values(self):
        data = np.array([[1, 3, 2], [4, 6, 5]], dtype=np.float32)
        k = 2
        sorted_flag = True
        top_k_op = kmath.TopK(k=k, sorted=sorted_flag)
        values, _ = top_k_op.call(data)
        expected_values = np.array([[3, 2], [6, 5]], dtype=np.float32)
        self.assertAllClose(values, expected_values)

    def test_top_k_call_indices(self):
        data = np.array([[1, 3, 2], [4, 6, 5]], dtype=np.float32)
        k = 2
        sorted_flag = True
        top_k_op = kmath.TopK(k=k, sorted=sorted_flag)
        _, indices = top_k_op.call(data)
        expected_indices = np.array([[1, 2], [1, 2]], dtype=np.int32)
        self.assertAllClose(indices, expected_indices)

    def test_top_k_operation_config_and_symbolic_call(self):
        top_k_op = kmath.TopK(k=2, sorted=True, is_stable=True)
        self.assertEqual(top_k_op.get_config()["is_stable"], True)
        data = np.array([3, 5, 5, 2, 5, 1], dtype=np.float32)
        values, indices = top_k_op.call(data)
        self.assertAllClose(values, [5, 5])
        self.assertAllClose(indices, [1, 2])


class InTopKTest(testing.TestCase):
    def test_in_top_k_call(self):
        targets = np.array([2, 0, 1], dtype=np.int32)
        predictions = np.array(
            [[0.1, 0.2, 0.7], [1.0, 0.2, 0.3], [0.2, 0.6, 0.2]],
            dtype=np.float32,
        )
        k = 2
        in_top_k_op = kmath.InTopK(k=k)
        output = in_top_k_op.call(targets, predictions)
        expected_output = np.array([True, True, True], dtype=bool)
        self.assertAllEqual(output, expected_output)


class LogsumexpTest(testing.TestCase):
    def test_logsumexp_call(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        axis = 0
        keepdims = True
        logsumexp_op = kmath.Logsumexp(axis=axis, keepdims=keepdims)
        output = logsumexp_op.call(x)
        expected_output = np.log(
            np.sum(np.exp(x), axis=axis, keepdims=keepdims)
        )
        self.assertAllClose(output, expected_output)

    def test_logsumexp_list_input(self):
        x = [[1.0, 2.0], [3.0, 4.0]]
        logsumexp_op = kmath.Logsumexp()
        output = logsumexp_op.call(x)
        expected_output = np.log(np.sum(np.exp(x)))
        self.assertAllClose(output, expected_output)


class FFTTest(testing.TestCase):
    def test_fft_input_not_tuple_or_list(self):
        fft_op = kmath.FFT()
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            fft_op.compute_output_spec(np.array([1, 2, 3]))

    def test_fft_input_parts_different_shapes(self):
        fft_op = kmath.FFT()
        real = np.array([1, 2, 3])
        imag = np.array([1, 2])
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            fft_op.compute_output_spec((real, imag))

    def test_fft_input_not_1d(self):
        fft_op = kmath.FFT()
        real = np.array(1)
        imag = np.array(1)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            fft_op.compute_output_spec((real, imag))

    def test_fft_last_axis_not_fully_defined(self):
        fft_op = kmath.FFT()
        real = KerasTensor(shape=(None,), dtype="float32")
        imag = KerasTensor(shape=(None,), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "Input should have its last dimension fully-defined"
        ):
            fft_op.compute_output_spec((real, imag))


class FFT2Test(testing.TestCase):
    def test_fft2_correct_input(self):
        fft2_op = kmath.FFT2()
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3, 4)
        # This should not raise any errors
        fft2_op.compute_output_spec((real_part, imag_part))

    def test_fft2_incorrect_input_type(self):
        fft2_op = kmath.FFT2()
        incorrect_input = np.array([1, 2, 3])  # Not a tuple or list
        with self.assertRaisesRegex(
            ValueError, "should be a tuple of two tensors"
        ):
            fft2_op.compute_output_spec(incorrect_input)

    def test_fft2_mismatched_shapes(self):
        fft2_op = kmath.FFT2()
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3)  # Mismatched shape
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            fft2_op.compute_output_spec((real_part, imag_part))

    def test_fft2_low_rank(self):
        fft2_op = kmath.FFT2()
        low_rank_input = np.random.rand(3)  # Rank of 1
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 2"):
            fft2_op.compute_output_spec((low_rank_input, low_rank_input))

    def test_fft2_undefined_dimensions(self):
        fft2_op = kmath.FFT2()
        real_part = KerasTensor(shape=(None, None, 3), dtype="float32")
        imag_part = KerasTensor(shape=(None, None, 3), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "Input should have its .* axes fully-defined"
        ):
            fft2_op.compute_output_spec((real_part, imag_part))


class RFFTTest(testing.TestCase):
    def test_rfft_low_rank_input(self):
        rfft_op = kmath.RFFT()
        low_rank_input = np.array(5)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            rfft_op.compute_output_spec(low_rank_input)

    def test_rfft_defined_fft_length(self):
        fft_length = 10
        rfft_op = kmath.RFFT(fft_length=fft_length)
        input_tensor = np.random.rand(3, 8)

        expected_last_dimension = fft_length // 2 + 1
        expected_shape = input_tensor.shape[:-1] + (expected_last_dimension,)

        output_tensors = rfft_op.compute_output_spec(input_tensor)
        for output_tensor in output_tensors:
            self.assertEqual(output_tensor.shape, expected_shape)

        def test_rfft_undefined_fft_length_defined_last_dim(self):
            rfft_op = kmath.RFFT()
            input_tensor = np.random.rand(3, 8)
            expected_last_dimension = input_tensor.shape[-1] // 2 + 1
            expected_shape = input_tensor.shape[:-1] + (
                expected_last_dimension,
            )
            output_tensors = rfft_op.compute_output_spec(input_tensor)
            for output_tensor in output_tensors:
                self.assertEqual(output_tensor.shape, expected_shape)

    def test_rfft_undefined_fft_length_undefined_last_dim(self):
        rfft_op = kmath.RFFT()
        input_tensor = KerasTensor(shape=(None, None), dtype="float32")
        expected_shape = input_tensor.shape[:-1] + (None,)
        output_tensors = rfft_op.compute_output_spec(input_tensor)
        for output_tensor in output_tensors:
            self.assertEqual(output_tensor.shape, expected_shape)


class ISTFTTest(testing.TestCase):
    def test_istft_incorrect_input_type(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        incorrect_input = np.array([1, 2, 3])
        with self.assertRaisesRegex(
            ValueError, "should be a tuple of two tensors"
        ):
            istft_op.compute_output_spec(incorrect_input)

    def test_istft_mismatched_shapes(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        real_part = np.random.rand(2, 3, 4)
        imag_part = np.random.rand(2, 3)
        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            istft_op.compute_output_spec((real_part, imag_part))

    def test_istft_low_rank_input(self):
        istft_op = kmath.ISTFT(
            sequence_length=5, sequence_stride=2, fft_length=10
        )
        low_rank_input = np.random.rand(3)
        with self.assertRaisesRegex(ValueError, "Input should have rank >= 2"):
            istft_op.compute_output_spec((low_rank_input, low_rank_input))

    def test_input_not_tuple_or_list_raises_error(self):
        irfft_op = kmath.IRFFT()
        invalid_input = np.array([1, 2, 3])
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(invalid_input)

    def test_input_tuple_with_less_than_two_elements_raises_error(self):
        irfft_op = kmath.IRFFT()
        too_short_input = (np.array([1, 2, 3]),)
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(too_short_input)

    def test_input_tuple_with_more_than_two_elements_raises_error(self):
        irfft_op = kmath.IRFFT()
        too_long_input = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        )
        with self.assertRaisesRegex(
            ValueError, "Input `x` should be a tuple of two tensors"
        ):
            irfft_op.compute_output_spec(too_long_input)

    def test_mismatched_shapes_input_validation(self):
        irfft_op = kmath.IRFFT()

        # Create real and imaginary parts with mismatched shapes
        real_part = np.array([1, 2, 3])
        imag_part = np.array([[1, 2], [3, 4]])

        with self.assertRaisesRegex(
            ValueError,
            "Both the real and imaginary parts should have the same shape",
        ):
            irfft_op.compute_output_spec((real_part, imag_part))

    def test_insufficient_rank_input_validation(self):
        irfft_op = kmath.IRFFT()

        # Create real and imaginary parts with insufficient rank (0D)
        real_part = np.array(1)
        imag_part = np.array(1)

        with self.assertRaisesRegex(ValueError, "Input should have rank >= 1"):
            irfft_op.compute_output_spec((real_part, imag_part))

    def test_with_specified_fft_length(self):
        fft_length = 10
        irfft_op = kmath.IRFFT(fft_length=fft_length)

        real_part = np.random.rand(4, 8)
        imag_part = np.random.rand(4, 8)

        expected_shape = real_part.shape[:-1] + (fft_length,)
        output_shape = irfft_op.compute_output_spec(
            (real_part, imag_part)
        ).shape

        self.assertEqual(output_shape, expected_shape)

    def test_inferred_fft_length_with_defined_last_dimension(self):
        irfft_op = kmath.IRFFT()

        real_part = np.random.rand(4, 8)
        imag_part = np.random.rand(4, 8)

        inferred_fft_length = 2 * (real_part.shape[-1] - 1)
        expected_shape = real_part.shape[:-1] + (inferred_fft_length,)
        output_shape = irfft_op.compute_output_spec(
            (real_part, imag_part)
        ).shape

        self.assertEqual(output_shape, expected_shape)

    def test_undefined_fft_length_and_last_dimension(self):
        irfft_op = kmath.IRFFT()

        real_part = KerasTensor(shape=(4, None), dtype="float32")
        imag_part = KerasTensor(shape=(4, None), dtype="float32")

        output_spec = irfft_op.compute_output_spec((real_part, imag_part))
        expected_shape = real_part.shape[:-1] + (None,)

        self.assertEqual(output_spec.shape, expected_shape)


class TestMathErrors(testing.TestCase):
    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Testing Jax errors only"
    )
    def test_segment_reduce_no_num_segments(self, segment_reduce_op):
        data = jnp.array([1, 2, 3, 4])
        segment_ids = jnp.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` must be set when using the JAX backend.",
        ):
            segment_reduce_op(data, segment_ids)

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Tensorflow error only"
    )
    def test_segment_reduce_sort_and_num_segments(self, segment_reduce_op):
        data = np.array([1, 2, 3, 4])
        segment_ids = np.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `num_segments` cannot be set when sorted is True when "
            "using the tensorflow backend.",
        ):
            segment_reduce_op(data, segment_ids, num_segments=2, sorted=True)

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    def test_segment_reduce_multi_dim_segment_ids(self, segment_reduce_op):
        data = np.array([1, 2, 3, 4])
        segment_ids = np.array([0, 0, 1, 1]).reshape((2, 2))
        with self.assertRaisesRegex(
            ValueError,
            "Argument `segment_ids` should be an 1-D vector,",
        ):
            segment_reduce_op(data, segment_ids)

    @parameterized.parameters([(kmath.segment_sum,), (kmath.segment_max,)])
    def test_segment_reduce_leading_not_match(self, segment_reduce_op):
        data = np.array([])
        segment_ids = np.array([0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `segment_ids` and `data` should have same leading "
            "dimension.",
        ):
            segment_reduce_op(data, segment_ids)

        output_tensor = segment_reduce_op(
            KerasTensor(shape=(None, 4)), KerasTensor(shape=(5,))
        )
        self.assertEqual(output_tensor.shape, (None, 4))

        output_tensor = segment_reduce_op(
            KerasTensor(shape=(5, 4)), KerasTensor(shape=(None,))
        )
        self.assertEqual(output_tensor.shape, (None, 4))

        output_tensor = segment_reduce_op(
            KerasTensor(shape=(None, 4)), KerasTensor(shape=(None,))
        )
        self.assertEqual(output_tensor.shape, (None, 4))

    def test_stft_invalid_input_type(self):
        # backend agnostic error message
        x = np.array([1, 2, 3, 4])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        with self.assertRaisesRegex(TypeError, "`float32` or `float64`"):
            kmath.stft(x, sequence_length, sequence_stride, fft_length)

    def test_invalid_fft_length(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 4
        sequence_stride = 1
        fft_length = 2
        with self.assertRaisesRegex(ValueError, "`fft_length` must equal or"):
            kmath.stft(x, sequence_length, sequence_stride, fft_length)

    def test_stft_invalid_window(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = "invalid_window"
        with self.assertRaisesRegex(ValueError, "If a string is passed to"):
            kmath.stft(
                x, sequence_length, sequence_stride, fft_length, window=window
            )

    def test_stft_invalid_window_shape(self):
        # backend agnostic error message
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        window = np.ones((sequence_length + 1))
        with self.assertRaisesRegex(ValueError, "The shape of `window` must"):
            kmath.stft(
                x, sequence_length, sequence_stride, fft_length, window=window
            )

    @parameterized.parameters([0, -5, 1.5])
    def test_stft_invalid_sequence_stride(self, sequence_stride):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sequence_length = 2
        fft_length = 4
        with self.assertRaisesRegex(
            ValueError, "`sequence_stride` must be a positive integer"
        ):
            kmath.stft(x, sequence_length, sequence_stride, fft_length)

    @parameterized.parameters([0, -5, 1.5])
    def test_istft_invalid_sequence_stride(self, sequence_stride):
        x = (np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        sequence_length = 2
        fft_length = 4
        with self.assertRaisesRegex(
            ValueError, "`sequence_stride` must be a positive integer"
        ):
            kmath.istft(x, sequence_length, sequence_stride, fft_length)

    def test_istft_invalid_window_shape_2D_inputs(self):
        # backend agnostic error message
        x = (np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        sequence_length = 2
        sequence_stride = 1
        fft_length = 4
        incorrect_window = np.ones((sequence_length + 1,))
        with self.assertRaisesRegex(
            ValueError, "The shape of `window` must be equal to"
        ):
            kmath.istft(
                x,
                sequence_length,
                sequence_stride,
                fft_length,
                window=incorrect_window,
            )


@pytest.mark.skipif(
    not backend.SUPPORTS_COMPLEX_DTYPES,
    reason="Complex dtype is not supported on this backend.",
)
class ViewAsComplexRealTest(testing.TestCase):
    def test_view_as_complex_basic(self):
        real_imag = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)

        result = kmath.view_as_complex(real_imag)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(standardize_dtype(result.dtype), expected.dtype)
        self.assertAllClose(result, expected)

    def test_view_as_real_basic(self):
        complex_tensor = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = kmath.view_as_real(complex_tensor)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(standardize_dtype(result.dtype), expected.dtype)
        self.assertAllClose(result, expected)

    def test_view_as_complex_invalid_shape(self):
        bad_input = np.array([1.0, 2.0, 3.0])  # Last dimension not size 2
        with self.assertRaisesRegex(
            ValueError, "Last dimension of input must be size 2"
        ):
            kmath.view_as_complex(bad_input)

    def test_view_as_complex_symbolic_input(self):
        x = KerasTensor(shape=(None, 2), dtype="float32")
        result = kmath.view_as_complex(x)

        self.assertEqual(result.shape, (None,))
        self.assertEqual(standardize_dtype(result.dtype), "complex64")

    def test_view_as_real_symbolic_input(self):
        x = KerasTensor(shape=(None,), dtype="complex64")
        result = kmath.view_as_real(x)

        self.assertEqual(result.shape, (None, 2))
        self.assertEqual(standardize_dtype(result.dtype), "float32")

    def test_view_as_complex_multi_dimensional(self):
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        expected = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)

        result = kmath.view_as_complex(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(standardize_dtype(result.dtype), expected.dtype)
        self.assertAllClose(result, expected)

    def test_view_as_real_multi_dimensional(self):
        x = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
        expected = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

        result = kmath.view_as_real(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(standardize_dtype(result.dtype), expected.dtype)
        self.assertAllClose(result, expected)
