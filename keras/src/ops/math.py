"""Commonly used math operations not included in NumPy."""

import math as python_math

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import config
from keras.src.backend.common.dtypes import result_type
from keras.src.ops.operation import ElementwiseUnaryOperation
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape


def _segment_reduce_validation(data, segment_ids):
    data_shape = data.shape
    segment_ids_shape = segment_ids.shape
    if len(segment_ids_shape) > 1:
        raise ValueError(
            "Argument `segment_ids` should be an 1-D vector, got shape: "
            f"{len(segment_ids_shape)}. Consider either flatten input with "
            "segment_ids.reshape((-1)) and "
            "data.reshape((-1, ) + data.shape[len(segment_ids.shape):]) or "
            "vectorize with vmap."
        )
    if (
        segment_ids_shape[0] is not None
        and data_shape[0] is not None
        and segment_ids_shape[0] != data_shape[0]
    ):
        raise ValueError(
            "Argument `segment_ids` and `data` should have same leading "
            f"dimension. Got {segment_ids_shape} v.s. "
            f"{data_shape}."
        )


class SegmentReduction(Operation):
    def __init__(self, num_segments=None, sorted=False, *, name=None):
        super().__init__(name=name)
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, _):
        output_shape = (self.num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)


class SegmentSum(SegmentReduction):
    def call(self, data, segment_ids):
        _segment_reduce_validation(data, segment_ids)
        return backend.ops.math.segment_sum(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_export("keras.ops.segment_sum")
def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    """Computes the sum of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A N-D tensor containing segment indices for each
            element in `data`. Num dims for segment ids should be strictly
            smaller or equal to number of dims in data.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Defaults to `False`.

    Returns:
        A tensor containing the sum of segments, where each element
        represents the sum of the corresponding segment in `data`.

    Example:

    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_sum(data, segment_ids,num_segments)
    array([3, 30, 300], dtype=int32)
    """
    _segment_reduce_validation(data, segment_ids)
    if any_symbolic_tensors((data,)):
        return SegmentSum(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.ops.math.segment_sum(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class SegmentMax(SegmentReduction):
    def call(self, data, segment_ids):
        _segment_reduce_validation(data, segment_ids)
        return backend.ops.math.segment_max(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_export("keras.ops.segment_max")
def segment_max(data, segment_ids, num_segments=None, sorted=False):
    """Computes the max of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A N-D tensor containing segment indices for each
            element in `data`. data.shape[:len(segment_ids.shape)] should match.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Defaults to `False`.

    Returns:
        A tensor containing the max of segments, where each element
        represents the max of the corresponding segment in `data`.

    Example:

    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_max(data, segment_ids, num_segments)
    array([2, 20, 200], dtype=int32)
    """
    _segment_reduce_validation(data, segment_ids)
    if any_symbolic_tensors((data,)):
        return SegmentMax(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.ops.math.segment_max(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class SegmentMin(SegmentReduction):
    def call(self, data, segment_ids):
        _segment_reduce_validation(data, segment_ids)
        return backend.ops.math.segment_min(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_export("keras.ops.segment_min")
def segment_min(data, segment_ids, num_segments=None, sorted=False):
    """Computes the min of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A N-D tensor containing segment indices for each
            element in `data`. data.shape[:len(segment_ids.shape)] should match.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Defaults to `False`.

    Returns:
        A tensor containing the min of segments, where each element
        represents the min of the corresponding segment in `data`.

    Example:

    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_min(data, segment_ids, num_segments)
    array([1, 10, 100], dtype=int32)
    """
    _segment_reduce_validation(data, segment_ids)
    if any_symbolic_tensors((data,)):
        return SegmentMin(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.ops.math.segment_min(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class SegmentProd(SegmentReduction):
    def call(self, data, segment_ids):
        _segment_reduce_validation(data, segment_ids)
        return backend.ops.math.segment_prod(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_export("keras.ops.segment_prod")
def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    """Computes the product of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A 1-D tensor containing segment indices for each
            element in `data`. `data.shape[0]` should match
            `segment_ids.shape[0]`.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Defaults to `False`.

    Returns:
        A tensor containing the product of segments, where each element
        represents the product of the corresponding segment in `data`.

    Example:
    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_prod(data, segment_ids, num_segments)
    array([2, 200, 20000], dtype=int32)
    """
    _segment_reduce_validation(data, segment_ids)

    if any_symbolic_tensors((data,)):
        return SegmentProd(
            num_segments,
            sorted,
        ).symbolic_call(data, segment_ids)
    return backend.ops.math.segment_prod(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class TopK(Operation):
    def __init__(self, k, sorted=True, is_stable=True, *, name=None):
        super().__init__(name=name)
        self.k = k
        self.sorted = sorted
        self.is_stable = is_stable

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        output_shape[-1] = self.k
        # Return a tuple (values, indices).
        return (
            KerasTensor(shape=output_shape, dtype=x.dtype),
            KerasTensor(shape=output_shape, dtype="int32"),
        )

    def call(self, x):
        return backend.ops.math.top_k(
            x, self.k, sorted=self.sorted, is_stable=self.is_stable
        )


@keras_export("keras.ops.top_k")
def top_k(x, k, sorted=True, is_stable=True):
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Input tensor.
        k: An integer representing the number of top elements to retrieve.
        sorted: A boolean indicating whether to sort the output in
            descending order. Defaults to `True`.
        is_stable: Optional boolean indicating whether to preserve the relative
            order of equal elements. Defaults to `True`.

    Returns:
        A tuple containing two tensors. The first tensor contains the
        top-k values, and the second tensor contains the indices of the
        top-k values in the input tensor.

    Example:

    >>> x = keras.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
    >>> values, indices = top_k(x, k=3)
    >>> print(values)
    array([9 7 5], shape=(3,), dtype=int32)
    >>> print(indices)
    array([4 2 0], shape=(3,), dtype=int32)

    """
    if any_symbolic_tensors((x,)):
        return TopK(k, sorted=sorted, is_stable=is_stable).symbolic_call(x)
    return backend.ops.math.top_k(x, k, sorted=sorted, is_stable=is_stable)


class InTopK(Operation):
    def __init__(self, k, *, name=None):
        super().__init__(name=name)
        self.k = k

    def compute_output_spec(self, targets, predictions):
        return KerasTensor(shape=targets.shape, dtype="bool")

    def call(self, targets, predictions):
        return backend.ops.math.in_top_k(targets, predictions, self.k)


@keras_export("keras.ops.in_top_k")
def in_top_k(targets, predictions, k):
    """Checks if the targets are in the top-k predictions.

    Args:
        targets: A tensor of true labels.
        predictions: A tensor of predicted labels.
        k: An integer representing the number of predictions to consider.

    Returns:
        A boolean tensor of the same shape as `targets`, where each element
        indicates whether the corresponding target is in the top-k predictions.

    Example:

    >>> targets = keras.ops.convert_to_tensor([2, 5, 3])
    >>> predictions = keras.ops.convert_to_tensor(
    ... [[0.1, 0.4, 0.6, 0.9, 0.5],
    ...  [0.1, 0.7, 0.9, 0.8, 0.3],
    ...  [0.1, 0.6, 0.9, 0.9, 0.5]])
    >>> in_top_k(targets, predictions, k=3)
    array([ True False  True], shape=(3,), dtype=bool)
    """
    if any_symbolic_tensors((targets, predictions)):
        return InTopK(k).symbolic_call(targets, predictions)
    return backend.ops.math.in_top_k(targets, predictions, k)


class Logsumexp(Operation):
    def __init__(self, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output_spec(self, x):
        output_shape = reduce_shape(x.shape, self.axis, self.keepdims)
        return KerasTensor(
            shape=output_shape, dtype=result_type(x.dtype, float)
        )

    def call(self, x):
        return backend.ops.math.logsumexp(
            x, axis=self.axis, keepdims=self.keepdims
        )


@keras_export("keras.ops.logsumexp")
def logsumexp(x, axis=None, keepdims=False):
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Input tensor.
        axis: An integer or a tuple of integers specifying the axis/axes
            along which to compute the sum. If `None`, the sum is computed
            over all elements. Defaults to `None`.
        keepdims: A boolean indicating whether to keep the dimensions of
            the input tensor when computing the sum. Defaults to `False`.

    Returns:
        A tensor containing the logarithm of the sum of exponentials of
        elements in `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([1., 2., 3.])
    >>> logsumexp(x)
    3.407606
    """
    if any_symbolic_tensors((x,)):
        return Logsumexp(axis, keepdims).symbolic_call(x)
    return backend.ops.math.logsumexp(x, axis=axis, keepdims=keepdims)


class CDist(Operation):
    def call(self, x, y):
        diff = backend.ops.numpy.expand_dims(
            x, -2
        ) - backend.ops.numpy.expand_dims(y, -3)
        return backend.ops.numpy.sqrt(
            backend.ops.numpy.sum(backend.ops.numpy.square(diff), axis=-1)
        )

    def compute_output_spec(self, x, y):
        if x.ndim < 2 or y.ndim < 2:
            raise ValueError(
                "Inputs to `cdist` must have rank >= 2. "
                f"Received shapes: x.shape={x.shape}, y.shape={y.shape}"
            )

        if (
            x.shape[-1] is not None
            and y.shape[-1] is not None
            and x.shape[-1] != y.shape[-1]
        ):
            raise ValueError(
                "The last dimension of inputs to `cdist` must match. "
                f"Received shapes: x.shape={x.shape}, y.shape={y.shape}"
            )

        try:
            batch_shape = broadcast_shapes(x.shape[:-2], y.shape[:-2])
        except ValueError:
            raise ValueError(
                "Batch dimensions of inputs to `cdist` must be broadcastable. "
                f"Received shapes: x.shape={x.shape}, y.shape={y.shape}"
            )

        output_shape = tuple(batch_shape + [x.shape[-2], y.shape[-2]])
        dtype = result_type(x.dtype, y.dtype, float)
        return KerasTensor(shape=output_shape, dtype=dtype)


@keras_export("keras.ops.cdist")
def cdist(x, y):
    """Computes pairwise distances between two collections of vectors.

    This function computes the Euclidean distance between each pair of the two
    collections of inputs.

    Args:
        x: Tensor of shape `(..., m, d)`.
        y: Tensor of shape `(..., n, d)`.

    Returns:
        A tensor of shape `(..., m, n)` with the pairwise distances.

    Example:
    >>> x = keras.ops.convert_to_tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> keras.ops.cdist(x, y)
    array([[1.       , 1.       ],
           [1.       , 1.4142135]], dtype=float32)
    """
    if any_symbolic_tensors((x, y)):
        return CDist().symbolic_call(x, y)
    return backend.ops.math.cdist(x, y)


class ExtractSequences(Operation):
    def __init__(self, sequence_length, sequence_stride, *, name=None):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )
        if x.shape[-1] is not None:
            num_sequences = (
                1 + (x.shape[-1] - self.sequence_length) // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.sequence_length)
        return KerasTensor(shape=new_shape, dtype=x.dtype)

    def call(self, x):
        return backend.ops.math.extract_sequences(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )


@keras_export("keras.ops.extract_sequences")
def extract_sequences(x, sequence_length, sequence_stride):
    """Expands the dimension of last axis into sequences of `sequence_length`.

    Slides a window of size `sequence_length` over the last axis of the input
    with a stride of `sequence_stride`, replacing the last axis with
    `[num_sequences, sequence_length]` sequences.

    If the dimension along the last axis is N, the number of sequences can be
    computed by:

    `num_sequences = 1 + (N - sequence_length) // sequence_stride`

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequences length.
        sequence_stride: An integer representing the sequences hop size.

    Returns:
        A tensor of sequences with shape [..., num_sequences, sequence_length].

    Example:

    >>> x = keras.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> extract_sequences(x, 3, 2)
    array([[1, 2, 3],
       [3, 4, 5]])
    """
    if any_symbolic_tensors((x,)):
        return ExtractSequences(sequence_length, sequence_stride).symbolic_call(
            x
        )
    return backend.ops.math.extract_sequences(
        x, sequence_length, sequence_stride
    )


class FFT(Operation):
    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )

        # We are calculating 1D FFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        # The axis along which we are calculating FFT should be fully-defined.
        m = real.shape[-1]
        if m is None:
            raise ValueError(
                f"Input should have its last dimension fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.ops.math.fft(x)


@keras_export("keras.ops.fft")
def fft(x):
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output tensor.

    Example:

    >>> x = (
    ...     keras.ops.convert_to_tensor([1., 2.]),
    ...     keras.ops.convert_to_tensor([0., 1.]),
    ... )
    >>> fft(x)
    (array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT().symbolic_call(x)
    return backend.ops.math.fft(x)


class FFT2(Operation):
    def compute_output_spec(self, x):
        axes = (-2, -1)
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[axes[0]]
        n = real.shape[axes[1]]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.ops.math.fft2(x)


@keras_export("keras.ops.fft2")
def fft2(x):
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Example:

    >>> x = (
    ...     keras.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
    ...     keras.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
    ... )
    >>> fft2(x)
    (array([[ 6.,  0.],
        [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
        [ 0., -2.]], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT2().symbolic_call(x)
    return backend.ops.math.fft2(x)


class Gammainc(Operation):
    def compute_output_spec(self, x1, x2):
        output_shape = broadcast_shapes(x1.shape, x2.shape)
        return KerasTensor(
            shape=output_shape,
            dtype=result_type(x1.dtype, x2.dtype, float),
        )

    def call(self, x1, x2):
        return backend.ops.math.gammainc(x1, x2)


@keras_export("keras.ops.gammainc")
def gammainc(x1, x2):
    """Computes the regularized lower incomplete gamma function.
    The regularized lower incomplete gamma function is defined as:
        P(x1, x2) = 1 / Γ(x1) * ∫₀ˣ² t^(x1 - 1) e^(-t) dt
    where `Γ(x1)` is the gamma function.

    Args:
        x1: A tensor containing the shape parameter.
        x2: A tensor containing the upper limit of integration. Must be
            broadcast-compatible with `x1`.

    Returns:
        A tensor containing the regularized lower incomplete gamma function
        evaluated elementwise.

    Example:

    >>> x1 = keras.ops.convert_to_tensor([1.0, 2.0, 3.0])
    >>> x2 = keras.ops.convert_to_tensor([0.5, 1.0, 2.0])
    >>> keras.ops.gammainc(x1, x2)
    array([0.39346936, 0.26424113, 0.3233236 ], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Gammainc().symbolic_call(x1, x2)
    return backend.ops.math.gammainc(x1, x2)


class IFFT2(Operation):
    def compute_output_spec(self, x):
        axes = (-2, -1)
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 2D IFFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes along which we are calculating IFFT should be fully-defined.
        m = real.shape[axes[0]]
        n = real.shape[axes[1]]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.ops.math.ifft2(x)


@keras_export("keras.ops.ifft2")
def ifft2(x):
    """Computes the 2D Inverse Fast Fourier Transform along the last two axes of
        input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Example:

    >>> x = (
    ...     keras.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
    ...     keras.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
    ... )
    >>> ifft2(x)
    (array([[ 6.,  0.],
        [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
        [ 0., -2.]], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return IFFT2().symbolic_call(x)
    return backend.ops.math.ifft2(x)


class RFFT(Operation):
    def __init__(self, fft_length=None, *, name=None):
        super().__init__(name=name)
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        # We are calculating 1D RFFT. Hence, rank >= 1.
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )

        if self.fft_length is not None:
            new_last_dimension = self.fft_length // 2 + 1
        else:
            if x.shape[-1] is not None:
                new_last_dimension = x.shape[-1] // 2 + 1
            else:
                new_last_dimension = None
        new_shape = x.shape[:-1] + (new_last_dimension,)

        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.ops.math.rfft(x, fft_length=self.fft_length)


@keras_export("keras.ops.rfft")
def rfft(x, fft_length=None):
    """Real-valued Fast Fourier Transform along the last axis of the input.

    Computes the 1D Discrete Fourier Transform of a real-valued signal over the
    inner-most dimension of input.

    Since the Discrete Fourier Transform of a real-valued signal is
    Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
    components of the FFT: the zero-frequency term, followed by the
    `fft_length / 2` positive-frequency terms.

    Along the axis RFFT is computed on, if `fft_length` is smaller than the
    corresponding dimension of the input, the dimension is cropped. If it is
    larger, the dimension is padded with zeros.

    Args:
        x: Input tensor.
        fft_length: An integer representing the number of the fft length. If not
            specified, it is inferred from the length of the last axis of `x`.
            Defaults to `None`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Examples:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> rfft(x)
    (array([10.0, -2.5, -2.5]), array([0.0, 3.4409548, 0.81229924]))

    >>> rfft(x, 3)
    (array([3.0, -1.5]), array([0.0, 0.8660254]))
    """
    if any_symbolic_tensors((x,)):
        return RFFT(fft_length).symbolic_call(x)
    return backend.ops.math.rfft(x, fft_length)


class IRFFT(Operation):
    def __init__(self, fft_length=None, *, name=None):
        super().__init__(name=name)
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )
        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 1D IRFFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        if self.fft_length is not None:
            new_last_dimension = self.fft_length
        else:
            if real.shape[-1] is not None:
                new_last_dimension = 2 * (real.shape[-1] - 1)
            else:
                new_last_dimension = None
        new_shape = real.shape[:-1] + (new_last_dimension,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.ops.math.irfft(x, fft_length=self.fft_length)


@keras_export("keras.ops.irfft")
def irfft(x, fft_length=None):
    """Inverse real-valued Fast Fourier transform along the last axis.

    Computes the inverse 1D Discrete Fourier Transform of a real-valued signal
    over the inner-most dimension of input.

    The inner-most dimension of the input is assumed to be the result of RFFT:
    the `fft_length / 2 + 1` unique components of the DFT of a real-valued
    signal. If `fft_length` is not provided, it is computed from the size of the
    inner-most dimension of the input `(fft_length = 2 * (inner - 1))`. If the
    FFT length used to compute is odd, it should be provided since it cannot
    be inferred properly.

    Along the axis IRFFT is computed on, if `fft_length / 2 + 1` is smaller than
    the corresponding dimension of the input, the dimension is cropped. If it is
    larger, the dimension is padded with zeros.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.
        fft_length: An integer representing the number of the fft length. If not
            specified, it is inferred from the length of the last axis of `x`.
            Defaults to `None`.

    Returns:
        A tensor containing the inverse real-valued Fast Fourier Transform
        along the last axis of `x`.

    Examples:

    >>> real = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> imag = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> irfft((real, imag))
    array([0.66666667, -0.9106836, 0.24401694])

    >>> irfft(rfft(real, 5), 5)
    array([0.0, 1.0, 2.0, 3.0, 4.0])
    """
    if any_symbolic_tensors(x):
        return IRFFT(fft_length).symbolic_call(x)
    return backend.ops.math.irfft(x, fft_length)


class STFT(Operation):
    def __init__(
        self,
        sequence_length,
        sequence_stride,
        fft_length,
        window="hann",
        center=True,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if x.shape[-1] is not None:
            padded = 0 if self.center is False else (self.fft_length // 2) * 2
            num_sequences = (
                1
                + (x.shape[-1] + padded - self.fft_length)
                // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.fft_length // 2 + 1)
        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.ops.math.stft(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            window=self.window,
            center=self.center,
        )


@keras_export("keras.ops.stft")
def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    """Short-Time Fourier Transform along the last axis of the input.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This giving frequency components of the signal as they change over
    time.

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequence length.
        sequence_stride: An integer representing the sequence hop size.
        fft_length: An integer representing the size of the FFT to apply. If not
            specified, uses the smallest power of 2 enclosing `sequence_length`.
        window: A string, a tensor of the window or `None`. If `window` is a
            string, available values are `"hann"` and `"hamming"`. If `window`
            is a tensor, it will be used directly as the window and its length
            must be `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        center: Whether to pad `x` on both sides so that the t-th sequence is
            centered at time `t * sequence_stride`. Otherwise, the t-th sequence
            begins at time `t * sequence_stride`. Defaults to `True`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        STFT output.

    Example:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> stft(x, 3, 2, 3)
    (array([[0.75, -0.375],
       [3.75, -1.875],
       [5.25, -2.625]]), array([[0.0, 0.64951905],
       [0.0, 0.64951905],
       [0.0, -0.64951905]]))
    """
    if not isinstance(sequence_stride, int) or sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be a positive integer. "
            f"Received: sequence_stride={sequence_stride}"
        )
    if any_symbolic_tensors((x,)):
        return STFT(
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            fft_length=fft_length,
            window=window,
            center=center,
        ).symbolic_call(x)
    return backend.ops.math.stft(
        x,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        fft_length=fft_length,
        window=window,
        center=center,
    )


class ISTFT(Operation):
    def __init__(
        self,
        sequence_length,
        sequence_stride,
        fft_length,
        length=None,
        window="hann",
        center=True,
        *,
        name=None,
    ):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.length = length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )
        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )
        if real.shape[-2] is not None:
            output_size = (
                real.shape[-2] - 1
            ) * self.sequence_stride + self.fft_length
            if self.length is not None:
                output_size = self.length
            elif self.center:
                output_size = output_size - (self.fft_length // 2) * 2
        else:
            output_size = None
            if self.length is not None:
                output_size = self.length
        new_shape = real.shape[:-2] + (output_size,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.ops.math.istft(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            length=self.length,
            window=self.window,
            center=self.center,
        )


@keras_export("keras.ops.istft")
def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    """Inverse Short-Time Fourier Transform along the last axis of the input.

    To reconstruct an original waveform, the parameters should be the same in
    `stft`.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.
        sequence_length: An integer representing the sequence length.
        sequence_stride: An integer representing the sequence hop size.
        fft_length: An integer representing the size of the FFT that produced
            `stft`. Should be of type `int32`.
        length: An integer representing the output is clipped to exactly length.
            If not specified, no padding or clipping take place. Defaults to
            `None`.
        window: A string, a tensor of the window or `None`. If `window` is a
            string, available values are `"hann"` and `"hamming"`. If `window`
            is a tensor, it will be used directly as the window and its length
            must be `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        center: Whether `x` was padded on both sides so that the t-th sequence
            is centered at time `t * sequence_stride`. Defaults to `True`.

    Returns:
        A tensor containing the inverse Short-Time Fourier Transform along the
        last axis of `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> istft(stft(x, 1, 1, 1), 1, 1, 1)
    array([0.0, 1.0, 2.0, 3.0, 4.0])
    """
    if not isinstance(sequence_stride, int) or sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be a positive integer. "
            f"Received: sequence_stride={sequence_stride}"
        )
    if any_symbolic_tensors(x):
        return ISTFT(
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            fft_length=fft_length,
            length=length,
            window=window,
            center=center,
        ).symbolic_call(x)
    return backend.ops.math.istft(
        x,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        fft_length=fft_length,
        length=length,
        window=window,
        center=center,
    )


class Rsqrt(ElementwiseUnaryOperation):
    call_fn = backend.ops.math.rsqrt


@keras_export("keras.ops.rsqrt")
def rsqrt(x):
    """Computes reciprocal of square root of x element-wise.

    Args:
        x: input tensor

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])
    >>> keras.ops.rsqrt(x)
    array([1.0, 0.31622776, 0.1], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Rsqrt().symbolic_call(x)
    return backend.ops.math.rsqrt(x)


class Erf(ElementwiseUnaryOperation):
    call_fn = backend.ops.math.erf


@keras_export("keras.ops.erf")
def erf(x):
    """Computes the error function of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
    >>> keras.ops.erf(x)
    array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Erf().symbolic_call(x)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.math.erf(x)


class Erfc(Operation):
    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape, dtype=result_type(x.dtype, float))

    def call(self, x):
        return backend.ops.math.erfc(x)


@keras_export("keras.ops.erfc")
def erfc(x):
    """Computes the complementary error function of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same dtype as `x`.

    Example:
    >>> x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
    >>> keras.ops.erfc(x)
    array([1.999978 , 1.9953222, 1.8427008, 1. , 0.1572992],
      dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Erfc().symbolic_call(x)
    return backend.ops.math.erfc(x)


class Erfinv(ElementwiseUnaryOperation):
    call_fn = backend.ops.math.erfinv


@keras_export("keras.ops.erfinv")
def erfinv(x):
    """Computes the inverse error function of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = np.array([-0.5, -0.2, -0.1, 0.0, 0.3])
    >>> keras.ops.erfinv(x)
    array([-0.47694, -0.17914, -0.08886,  0. ,  0.27246], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Erfinv().symbolic_call(x)
    return backend.ops.math.erfinv(x)


class Logdet(Operation):
    def call(self, x):
        return backend.ops.math.logdet(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape[:-2], dtype=x.dtype)


@keras_export(["keras.ops.logdet"])
def logdet(x):
    """Computes log of the determinant of a hermitian positive definite matrix.

    Args:
        x: Input matrix. It must 2D and square.

    Returns:
        The natural log of the determinant of matrix.
    """
    if any_symbolic_tensors((x,)):
        return Logdet().symbolic_call(x)
    return backend.ops.math.logdet(x)


class ViewAsComplex(Operation):
    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        if len(x.shape) < 1 or x.shape[-1] != 2:
            raise ValueError(
                "Input tensor's last dimension must be 2 (real and imaginary)."
            )
        return x[..., 0] + 1j * x[..., 1]

    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape[:-1], dtype="complex64")


class ViewAsReal(Operation):
    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        real_part = backend.ops.numpy.real(x)
        imag_part = backend.ops.numpy.imag(x)
        return backend.ops.numpy.stack((real_part, imag_part), axis=-1)

    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape + (2,), dtype="float32")


@keras_export("keras.ops.view_as_complex")
def view_as_complex(x):
    """Converts a real tensor with shape `(..., 2)` to a complex tensor,
    where the last dimension represents the real and imaginary components
    of a complex tensor.

    Args:
        x: A real tensor with last dimension of size 2.

    Returns:
        A complex tensor with shape `x.shape[:-1]`.

    Example:

    ```
    >>> import numpy as np
    >>> from keras import ops

    >>> real_imag = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> complex_tensor = ops.view_as_complex(real_imag)
    >>> complex_tensor
    array([1.+2.j, 3.+4.j])
    ```
    """
    if any_symbolic_tensors((x,)):
        return ViewAsComplex().symbolic_call(x)

    x = backend.ops.convert_to_tensor(x)
    if len(x.shape) < 1 or x.shape[-1] != 2:
        raise ValueError(
            "Last dimension of input must be size 2 (real and imaginary). "
            f"Received shape: {x.shape}"
        )
    real_part = x[..., 0]
    imag_part = x[..., 1]

    return backend.ops.cast(
        real_part, dtype="complex64"
    ) + 1j * backend.ops.cast(imag_part, dtype="complex64")


@keras_export("keras.ops.view_as_real")
def view_as_real(x):
    """Converts a complex tensor to a real tensor with shape `(..., 2)`,
    where the last dimension represents the real and imaginary components.

    Args:
        x: A complex tensor.

    Returns:
        A real tensor where the last dimension contains the
        real and imaginary parts.

    Example:
    ```
    >>> import numpy as np
    >>> from keras import ops

    >>> complex_tensor = np.array([1 + 2j, 3 + 4j])
    >>> real = ops.view_as_real(complex_tensor)
    >>> real
    array([[1., 2.],
           [3., 4.]])
    ```
    """
    if any_symbolic_tensors((x,)):
        return ViewAsReal().symbolic_call(x)

    x = backend.ops.convert_to_tensor(x)
    real_part = backend.ops.numpy.real(x)
    imag_part = backend.ops.numpy.imag(x)
    return backend.ops.numpy.stack((real_part, imag_part), axis=-1)


class Lgamma(Operation):
    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape, dtype=result_type(x.dtype, float))

    def call(self, x):
        return _lgamma(x)


@keras_export("keras.ops.lgamma")
def lgamma(x):
    """Computes the natural log of the absolute value of the Gamma function.

    `lgamma(x) = log(|Gamma(x)|)`

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape and floating-point dtype as `x`.

    Example:

    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> keras.ops.lgamma(x)
    array([0.       , 0.       , 0.6931472, 1.7917595], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Lgamma().symbolic_call(x)
    return _lgamma(x)


# Lanczos approximation parameters
_LANCZOS_GAMMA = 7.0
_BASE_LANCZOS_COEFF = 0.99999999999980993227684700473478
_LANCZOS_COEFFICIENTS = (
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7,
)
_PI = 3.14159265358979323846
_LOG_SQRT_TWO_PI = (python_math.log(2.0) + python_math.log(_PI)) / 2.0
_LANCZOS_GAMMA_PLUS_HALF = _LANCZOS_GAMMA + 0.5
_LOG_LANCZOS_GAMMA_PLUS_HALF = python_math.log(_LANCZOS_GAMMA_PLUS_HALF)


def _lgamma(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.math, "lgamma"
    ):
        return backend.ops.math.lgamma(x)

    x = backend.ops.convert_to_tensor(x)
    orig_dtype = x.dtype
    target_dtype = result_type(orig_dtype, float)
    compute_dtype = (
        "float32" if target_dtype in ("float16", "bfloat16") else target_dtype
    )
    x = backend.ops.cast(x, compute_dtype)

    # If the input is less than 0.5 use Euler's reflection formula:
    # gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
    need_to_reflect = backend.ops.numpy.less(x, 0.5)
    z = backend.ops.numpy.where(need_to_reflect, -x, x - 1.0)

    series = backend.ops.cast(_BASE_LANCZOS_COEFF, compute_dtype)
    for i, coeff in enumerate(_LANCZOS_COEFFICIENTS):
        series = series + backend.ops.cast(coeff, compute_dtype) / (
            z + float(i + 1)
        )

    lanczos_gamma_plus_half = backend.ops.cast(
        _LANCZOS_GAMMA_PLUS_HALF, compute_dtype
    )
    log_lanczos_gamma_plus_half = backend.ops.cast(
        _LOG_LANCZOS_GAMMA_PLUS_HALF, compute_dtype
    )
    pi = backend.ops.cast(_PI, compute_dtype)
    t = z + lanczos_gamma_plus_half
    log_t = log_lanczos_gamma_plus_half + backend.ops.numpy.log1p(
        z / lanczos_gamma_plus_half
    )

    log_sqrt_two_pi = backend.ops.cast(_LOG_SQRT_TWO_PI, compute_dtype)
    log_y = (
        log_sqrt_two_pi
        + (z + 0.5 - t / log_t) * log_t
        + backend.ops.numpy.log(series)
    )

    abs_x = backend.ops.numpy.abs(x)
    abs_frac_x = abs_x - backend.ops.numpy.floor(abs_x)
    reduced_frac_x = backend.ops.numpy.where(
        backend.ops.numpy.greater(abs_frac_x, 0.5), 1.0 - abs_frac_x, abs_frac_x
    )
    reflection_denom = backend.ops.numpy.log(
        backend.ops.numpy.sin(pi * reduced_frac_x)
    )

    reflection = backend.ops.numpy.where(
        backend.ops.numpy.isfinite(reflection_denom),
        backend.ops.cast(backend.ops.numpy.log(pi), compute_dtype)
        - reflection_denom
        - log_y,
        -reflection_denom,
    )
    result = backend.ops.numpy.where(need_to_reflect, reflection, log_y)

    # Handle +/-inf edge cases: lgamma(+/-inf) = +inf
    inf_val = backend.ops.cast(float("inf"), compute_dtype)
    result = backend.ops.numpy.where(
        backend.ops.numpy.isinf(x), inf_val, result
    )

    if compute_dtype != target_dtype:
        result = backend.ops.cast(result, target_dtype)
    return result
