from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import standardize_dtype
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape


class Cholesky(Operation):
    def __init__(self, upper=False, *, name=None):
        super().__init__(name=name)
        self.upper = upper

    def call(self, x):
        return _cholesky(x, self.upper)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export(["keras.ops.cholesky", "keras.ops.linalg.cholesky"])
def cholesky(x, upper=False):
    """Computes the Cholesky decomposition of a positive semi-definite matrix.

    Args:
        x: Input tensor of shape `(..., M, M)`.
        upper (bool): If True, returns the upper-triangular Cholesky factor.
            If False (default), returns the lower-triangular Cholesky factor.

    Returns:
        A tensor of shape `(..., M, M)` representing the Cholesky factor of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Cholesky(upper=upper).symbolic_call(x)
    return _cholesky(x, upper=upper)


def _cholesky(x, upper=False):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    try:
        return backend.ops.linalg.cholesky(x, upper=upper)
    except Exception as e:
        raise ValueError(f"Cholesky decomposition failed: {e}")


class CholeskyInverse(Operation):
    def __init__(self, upper=False, *, name=None):
        super().__init__(name=name)
        self.upper = upper

    def call(self, x):
        return _cholesky_inverse(x, self.upper)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export(
    ["keras.ops.cholesky_inverse", "keras.ops.linalg.cholesky_inverse"]
)
def cholesky_inverse(x, upper=False):
    """Computes the inverse of a symmetric positive-definite matrix.

    Args:
        x: Input tensor of shape `(..., M, M)`.
        upper (bool): Determines whether to use the upper- or lower-triangular
            factor for the internal computation. Defaults to False.

    Returns:
        A tensor of shape `(..., M, M)` representing the inverse of `x`.

    Raises:
        ValueError: If `x` is not a symmetric positive-definite matrix.
    """
    if any_symbolic_tensors((x,)):
        return CholeskyInverse(upper=upper).symbolic_call(x)
    return _cholesky_inverse(x, upper=upper)


def _cholesky_inverse(x, upper=False):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    try:
        return backend.ops.linalg.cholesky_inverse(x, upper=upper)
    except Exception as e:
        raise ValueError(f"Cholesky inverse failed: {e}")


class Det(Operation):
    def call(self, x):
        return _det(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape[:-2], x.dtype)


@keras_export(["keras.ops.det", "keras.ops.linalg.det"])
def det(x):
    """Computes the determinant of a square tensor.

    Args:
        x: Input tensor of shape `(..., M, M)`.

    Returns:
        A tensor of shape `(...,)` representing the determinant of `x`.

    """
    if any_symbolic_tensors((x,)):
        return Det().symbolic_call(x)
    return _det(x)


def _det(x):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    return backend.ops.linalg.det(x)


class Eig(Operation):
    def call(self, x):
        return _eig(x)

    def compute_output_spec(self, x):
        _assert_square(x)
        _assert_2d(x)
        return (
            KerasTensor(x.shape[:-1], x.dtype),
            KerasTensor(x.shape, x.dtype),
        )


@keras_export(["keras.ops.eig", "keras.ops.linalg.eig"])
def eig(x):
    """Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        x: Input tensor of shape `(..., M, M)`.

    Returns:
        A tuple of two tensors: a tensor of shape `(..., M)` containing
        eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.
    """
    if any_symbolic_tensors((x,)):
        return Eig().symbolic_call(x)
    return _eig(x)


def _eig(x):
    x = backend.ops.convert_to_tensor(x)
    _assert_square(x)
    _assert_2d(x)
    return backend.ops.linalg.eig(x)


class Eigh(Operation):
    def call(self, x):
        return _eigh(x)

    def compute_output_spec(self, x):
        _assert_square(x)
        _assert_2d(x)
        return (
            KerasTensor(x.shape[:-1], x.dtype),
            KerasTensor(x.shape, x.dtype),
        )


@keras_export(["keras.ops.eigh", "keras.ops.linalg.eigh"])
def eigh(x):
    """Computes the eigenvalues and eigenvectors of a complex Hermitian.

    Args:
        x: Input tensor of shape `(..., M, M)`.

    Returns:
        A tuple of two tensors: a tensor of shape `(..., M)` containing
        eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.

    """
    if any_symbolic_tensors((x,)):
        return Eigh().symbolic_call(x)
    return _eigh(x)


def _eigh(x):
    x = backend.ops.convert_to_tensor(x)
    _assert_square(x)
    _assert_2d(x)
    return backend.ops.linalg.eigh(x)


class Inv(Operation):
    def call(self, x):
        return _inv(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export(["keras.ops.inv", "keras.ops.linalg.inv"])
def inv(x):
    """Computes the inverse of a square tensor.

    Args:
        x: Input tensor of shape `(..., M, M)`.

    Returns:
        A tensor of shape `(..., M, M)` representing the inverse of `x`.

    """
    if any_symbolic_tensors((x,)):
        return Inv().symbolic_call(x)
    return _inv(x)


def _inv(x):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    return backend.ops.linalg.inv(x)


class LuFactor(Operation):
    def call(self, x):
        return _lu_factor(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        batch_shape = x.shape[:-2]
        m, n = x.shape[-2:]
        k = min(m, n)
        return (
            KerasTensor(batch_shape + (m, n), x.dtype),
            # The pivots are indices, not values drawn from `x`.
            KerasTensor(batch_shape + (k,), "int32"),
        )


@keras_export(["keras.ops.lu_factor", "keras.ops.linalg.lu_factor"])
def lu_factor(x):
    """Computes the lower-upper decomposition of a square matrix.

    Args:
        x: A tensor of shape `(..., M, M)`.

    Returns:
        A tuple of two tensors: a tensor of shape `(..., M, M)` containing the
        lower and upper triangular matrices and a tensor of shape `(..., M)`
        containing the pivots.

    """
    if any_symbolic_tensors((x,)):
        return LuFactor().symbolic_call(x)
    return _lu_factor(x)


def _lu_factor(x):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    if backend.backend() == "tensorflow":
        try:
            _assert_square(x)
        except ValueError as e:
            raise ValueError(
                f"LU decomposition failed: {e}. LU decomposition is only "
                "supported for square matrices in Tensorflow."
            )
    return backend.ops.linalg.lu_factor(x)


class Norm(Operation):
    def __init__(self, ord=None, axis=None, keepdims=False, *, name=None):
        super().__init__(name=name)
        if isinstance(ord, str):
            if ord not in ("fro", "nuc"):
                raise ValueError(
                    "Invalid `ord` argument. "
                    "Expected one of {'fro', 'nuc'} when using string. "
                    f"Received: ord={ord}"
                )
        self.ord = ord
        self.axis = axis
        self.keepdims = keepdims

    def compute_output_spec(self, x):
        output_dtype = standardize_dtype(x.dtype)
        if "int" in output_dtype or output_dtype == "bool":
            output_dtype = backend.floatx()
        if self.axis is None:
            num_axes = len(x.shape)
        elif isinstance(self.axis, int):
            num_axes = 1
        else:
            num_axes = len(self.axis)
        if num_axes == 1 and isinstance(self.ord, str):
            raise ValueError(
                "Invalid `ord` argument for vector norm. "
                f"Received: ord={self.ord}"
            )
        elif num_axes == 2 and self.ord not in (
            None,
            "fro",
            "nuc",
            float("inf"),
            float("-inf"),
            1,
            -1,
            2,
            -2,
        ):
            raise ValueError(
                "Invalid `ord` argument for matrix norm. "
                f"Received: ord={self.ord}"
            )
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )

    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        return backend.ops.linalg.norm(
            x, ord=self.ord, axis=self.axis, keepdims=self.keepdims
        )


@keras_export(["keras.ops.norm", "keras.ops.linalg.norm"])
def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm.

    This function is able to return one of eight different matrix norms, or one
    of an infinite number of vector norms (described below), depending on the
    value of the `ord` parameter.

    Args:
        x: Input tensor.
        ord: Order of the norm (see table under Notes). The default is `None`.
        axis: If `axis` is an integer, it specifies the axis of `x` along which
            to compute the vector norms. If `axis` is a 2-tuple, it specifies
            the axes that hold 2-D matrices, and the matrix norms of these
            matrices are computed.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Note:
        For values of `ord < 1`, the result is, strictly speaking, not a
        mathematical 'norm', but it may still be useful for various numerical
        purposes. The following norms can be calculated:
        - For matrices:
            - `ord=None`: Frobenius norm
            - `ord="fro"`: Frobenius norm
            - `ord="nuc"`: nuclear norm
            - `ord=np.inf`: `max(sum(abs(x), axis=1))`
            - `ord=-np.inf`: `min(sum(abs(x), axis=1))`
            - `ord=0`: not supported
            - `ord=1`: `max(sum(abs(x), axis=0))`
            - `ord=-1`: `min(sum(abs(x), axis=0))`
            - `ord=2`: 2-norm (largest sing. value)
            - `ord=-2`: smallest singular value
            - other: not supported
        - For vectors:
            - `ord=None`: 2-norm
            - `ord="fro"`: not supported
            - `ord="nuc"`: not supported
            - `ord=np.inf`: `max(abs(x))`
            - `ord=-np.inf`: `min(abs(x))`
            - `ord=0`: `sum(x != 0)`
            - `ord=1`: as below
            - `ord=-1`: as below
            - `ord=2`: as below
            - `ord=-2`: as below
            - other: `sum(abs(x)**ord)**(1./ord)`

    Returns:
        Norm of the matrix or vector(s).

    Example:

    >>> x = keras.ops.reshape(keras.ops.arange(9, dtype="float32") - 4, (3, 3))
    >>> keras.ops.linalg.norm(x)
    7.7459664
    """
    if any_symbolic_tensors((x,)):
        return Norm(ord=ord, axis=axis, keepdims=keepdims).symbolic_call(x)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


class Qr(Operation):
    def __init__(self, mode="reduced", *, name=None):
        super().__init__(name=name)
        if mode not in {"reduced", "complete"}:
            raise ValueError(
                "`mode` argument value not supported. "
                "Expected one of {'reduced', 'complete'}. "
                f"Received: mode={mode}"
            )
        self.mode = mode

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                "Input should have rank >= 2. Received: "
                f"input.shape = {x.shape}"
            )
        m = x.shape[-2]
        n = x.shape[-1]
        if m is None or n is None:
            raise ValueError(
                "Input should have its last 2 dimensions "
                "fully-defined. Received: "
                f"input.shape = {x.shape}"
            )
        k = min(m, n)
        base = tuple(x.shape[:-2])
        if self.mode == "reduced":
            return (
                KerasTensor(shape=base + (m, k), dtype=x.dtype),
                KerasTensor(shape=base + (k, n), dtype=x.dtype),
            )
        # 'complete' mode.
        return (
            KerasTensor(shape=base + (m, m), dtype=x.dtype),
            KerasTensor(shape=base + (m, n), dtype=x.dtype),
        )

    def call(self, x):
        x = backend.ops.convert_to_tensor(x)
        return backend.ops.linalg.qr(x, mode=self.mode)


@keras_export(["keras.ops.qr", "keras.ops.linalg.qr"])
def qr(x, mode="reduced"):
    """Computes the QR decomposition of a tensor.

    Args:
        x: Input tensor of shape `(..., M, N)`.
        mode: A string specifying the mode of the QR decomposition.
            - 'reduced': Returns the reduced QR decomposition. (default)
            - 'complete': Returns the complete QR decomposition.

    Returns:
        A tuple containing two tensors. The first tensor of shape `(..., M, K)`
        is the orthogonal matrix `q` and the second tensor of shape
        `(..., K, N)` is the upper triangular matrix `r`, where `K = min(M, N)`.

    Example:

    >>> x = keras.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> q, r = qr(x)
    >>> print(q)
    array([[-0.16903079  0.897085]
           [-0.5070925   0.2760267 ]
           [-0.8451542  -0.34503305]], shape=(3, 2), dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Qr(mode=mode).symbolic_call(x)
    x = backend.ops.convert_to_tensor(x)
    return backend.ops.linalg.qr(x, mode=mode)


class Solve(Operation):
    def call(self, a, b):
        return _solve(a, b)

    def compute_output_spec(self, a, b):
        _assert_2d(a)
        _assert_square(a)
        _assert_1d(b)
        _assert_a_b_compat(a, b)
        return KerasTensor(b.shape, b.dtype)


@keras_export(["keras.ops.solve", "keras.ops.linalg.solve"])
def solve(a, b):
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: A tensor of shape `(..., M, M)` representing the coefficients matrix.
        b: A tensor of shape `(..., M)` or `(..., M, N)` representing the
        right-hand side or "dependent variable" matrix.

    Returns:
        A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
        of the linear system. Returned shape is identical to `b`.

    """
    if any_symbolic_tensors((a, b)):
        return Solve().symbolic_call(a, b)
    return _solve(a, b)


def _solve(a, b):
    a = backend.ops.convert_to_tensor(a)
    b = backend.ops.convert_to_tensor(b)
    _assert_2d(a)
    _assert_square(a)
    _assert_1d(b)
    _assert_a_b_compat(a, b)
    return backend.ops.linalg.solve(a, b)


class SolveTriangular(Operation):
    def __init__(self, lower=False, *, name=None):
        super().__init__(name=name)
        self.lower = lower

    def call(self, a, b):
        return _solve_triangular(a, b, self.lower)

    def compute_output_spec(self, a, b):
        _assert_2d(a)
        _assert_square(a)
        _assert_1d(b)
        _assert_a_b_compat(a, b)
        return KerasTensor(b.shape, b.dtype)


@keras_export(
    ["keras.ops.solve_triangular", "keras.ops.linalg.solve_triangular"]
)
def solve_triangular(a, b, lower=False):
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: A tensor of shape `(..., M, M)` representing the coefficients matrix.
        b: A tensor of shape `(..., M)` or `(..., M, N)` representing the
        right-hand side or "dependent variable" matrix.

    Returns:
        A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
        of the linear system. Returned shape is identical to `b`.

    """
    if any_symbolic_tensors((a, b)):
        return SolveTriangular(lower).symbolic_call(a, b)
    return _solve_triangular(a, b, lower)


def _solve_triangular(a, b, lower=False):
    a = backend.ops.convert_to_tensor(a)
    b = backend.ops.convert_to_tensor(b)
    _assert_2d(a)
    _assert_square(a)
    _assert_1d(b)
    _assert_a_b_compat(a, b)
    return backend.ops.linalg.solve_triangular(a, b, lower)


class SVD(Operation):
    def __init__(self, full_matrices=True, compute_uv=True, *, name=None):
        super().__init__(name=name)
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def call(self, x):
        return _svd(x, self.full_matrices, self.compute_uv)

    def compute_output_spec(self, x):
        _assert_2d(x)
        rows, columns = x.shape[-2:]
        batches = x.shape[:-2]
        s_shape = batches + (min(rows, columns),)
        if self.full_matrices:
            u_shape = batches + (rows, rows)
            v_shape = batches + (columns, columns)
        else:
            u_shape = batches + (rows, min(rows, columns))
            v_shape = batches + (min(rows, columns), columns)

        if self.compute_uv:
            return (
                KerasTensor(u_shape, x.dtype),
                KerasTensor(s_shape, x.dtype),
                KerasTensor(v_shape, x.dtype),
            )
        return KerasTensor(s_shape, x.dtype)


@keras_export(["keras.ops.svd", "keras.ops.linalg.svd"])
def svd(x, full_matrices=True, compute_uv=True):
    """Computes the singular value decomposition of a matrix.

    Args:
        x: Input tensor of shape `(..., M, N)`.

    Returns:
        A tuple of three tensors: a tensor of shape `(..., M, M)` containing the
        left singular vectors, a tensor of shape `(..., M, N)` containing the
        singular values and a tensor of shape `(..., N, N)` containing the
        right singular vectors.

    """
    if any_symbolic_tensors((x,)):
        return SVD(full_matrices, compute_uv).symbolic_call(x)
    return _svd(x, full_matrices, compute_uv)


def _svd(x, full_matrices=True, compute_uv=True):
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    return backend.ops.linalg.svd(x, full_matrices, compute_uv)


class Lstsq(Operation):
    def __init__(self, rcond=None, *, name=None):
        super().__init__(name=name)
        self.rcond = rcond

    def call(self, a, b):
        return backend.ops.linalg.lstsq(a, b, rcond=self.rcond)

    def compute_output_spec(self, a, b):
        if len(a.shape) != 2:
            raise ValueError(
                f"Expected a to have rank 2. Received: a.shape={a.shape}"
            )
        if len(b.shape) not in (1, 2):
            raise ValueError(
                f"Expected b to have rank 1 or 2. Received: b.shape={b.shape}"
            )
        m, n = a.shape
        if b.shape[0] != m:
            raise ValueError(
                "Expected b.shape[0] to be equal to "
                "a.shape[0]. Received: "
                f"a.shape={a.shape}, b.shape={b.shape}"
            )
        if len(b.shape) == 2:
            k = b.shape[1]
            x = KerasTensor((n, k), dtype=a.dtype)
        else:
            x = KerasTensor((n,), dtype=a.dtype)
        return x


@keras_export(["keras.ops.lstsq", "keras.ops.linalg.lstsq"])
def lstsq(a, b, rcond=None):
    """Return the least-squares solution to a linear matrix equation.

    Computes the vector x that approximately solves the equation
    `a @ x = b`. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of a can be less than,
    equal to, or greater than its number of linearly independent columns).
    If a is square and of full rank, then `x` (but for round-off error)
    is the exact solution of the equation. Else, `x` minimizes the
    L2 norm of `b - a * x`.

    If there are multiple minimizing solutions,
    the one with the smallest L2 norm  is returned.

    Args:
        a: "Coefficient" matrix of shape `(M, N)`.
        b: Ordinate or "dependent variable" values,
            of shape `(M,)` or `(M, K)`.
            If `b` is two-dimensional, the least-squares solution
            is calculated for each of the K columns of `b`.
        rcond: Cut-off ratio for small singular values of `a`.
            For the purposes of rank determination,
            singular values are treated as zero if they are
            smaller than rcond times the largest
            singular value of `a`.

    Returns:
        Tensor with shape `(N,)` or `(N, K)` containing
        the least-squares solutions.

    **NOTE:** The output differs from `numpy.linalg.lstsq`.
    NumPy returns a tuple with four elements, the first of which
    being the least-squares solutions and the others
    being essentially never used.
    Keras only returns the first value. This is done both
    to ensure consistency across backends (which cannot be achieved
    for the other values) and to simplify the API.
    """
    if any_symbolic_tensors((a, b)):
        return Lstsq(rcond=rcond).symbolic_call(a, b)
    return backend.ops.linalg.lstsq(a, b, rcond=rcond)


class MatrixRank(Operation):
    def __init__(self, tol=None, *, name=None):
        super().__init__(name=name)
        self.tol = tol

    def call(self, x):
        return backend.ops.linalg.matrix_rank(x, tol=self.tol)

    def compute_output_spec(self, x):
        _assert_2d(x)
        return KerasTensor(x.shape[:-2], dtype="int32")


@keras_export(["keras.ops.matrix_rank", "keras.ops.linalg.matrix_rank"])
def matrix_rank(x, tol=None):
    """Return the matrix rank of one or more matrices using SVD.

    The rank is the number of singular values that exceed `tol`. If `tol`
    is `None`, each backend uses its own default threshold derived from
    the largest singular value and the matrix dimensions.

    Args:
        x: Input tensor of shape `(..., M, N)`.
        tol: Absolute threshold below which singular values are treated
            as zero. If `None` (default), the backend's default is used.

    Returns:
        An integer tensor of shape `(...,)` with the rank of each matrix
        in the batch.

    Example:

    >>> a = keras.ops.convert_to_tensor([[1., 2.], [2., 4.]])
    >>> keras.ops.matrix_rank(a)
    1
    """
    if any_symbolic_tensors((x,)):
        return MatrixRank(tol=tol).symbolic_call(x)
    return backend.ops.linalg.matrix_rank(x, tol=tol)


class MatrixPower(Operation):
    def __init__(self, n, *, name=None):
        super().__init__(name=name)
        if not isinstance(n, int):
            raise TypeError(
                f"n must be an integer. Received: n={n} of type {type(n)}"
            )
        self.n = n

    def call(self, x):
        return _matrix_power(x, self.n)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export(["keras.ops.matrix_power", "keras.ops.linalg.matrix_power"])
def matrix_power(x, n):
    """Raise a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If `n == 0`, the identity matrix
    of the same shape as `x` is returned. If `n < 0`, the inverse is
    computed and then raised to the `abs(n)`.

    Args:
        x: Input tensor of shape `(..., M, M)`.
        n: Exponent (integer).

    Returns:
        A tensor of shape `(..., M, M)` representing `x**n`.

    Example:

    >>> x = keras.ops.convert_to_tensor([[1., 2.], [3., 4.]])
    >>> matrix_power(x, 3)
    array([[ 37.,  54.],
           [ 81., 118.]], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return MatrixPower(n).symbolic_call(x)
    return _matrix_power(x, n)


def _matrix_power(x, n):
    if not isinstance(n, int):
        raise TypeError(
            f"n must be an integer. Received: n={n} of type {type(n)}"
        )
    x = backend.ops.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    return backend.ops.linalg.matrix_power(x, n)


class Pinv(Operation):
    def __init__(self, rcond=None, *, name=None):
        super().__init__(name=name)
        self.rcond = rcond

    def call(self, x):
        return backend.ops.linalg.pinv(x, rcond=self.rcond)

    def compute_output_spec(self, x):
        _assert_2d(x)
        batch_shape = x.shape[:-2]
        m, n = x.shape[-2:]
        return KerasTensor(batch_shape + (n, m), x.dtype)


@keras_export(["keras.ops.pinv", "keras.ops.linalg.pinv"])
def pinv(x, rcond=None):
    """Compute the (Moore-Penrose) pseudoinverse of a matrix.

    The pseudoinverse is computed via the singular value decomposition
    (SVD). Singular values smaller than `rcond` times the largest
    singular value (per matrix, in the batched case) are treated as zero.

    Args:
        x: Input tensor of shape `(..., M, N)`.
        rcond: Cutoff ratio for small singular values. Singular values
            less than or equal to `rcond * largest_singular_value` are
            set to zero. If `None` (default), the backend's own default
            is used.

    Returns:
        A tensor of shape `(..., N, M)` containing the pseudoinverse
        of each matrix in `x`.

    Example:

    >>> a = keras.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> keras.ops.pinv(a) @ a
    array([[1., 0.],
           [0., 1.]], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Pinv(rcond=rcond).symbolic_call(x)
    return backend.ops.linalg.pinv(x, rcond=rcond)


def _assert_1d(*arrays):
    for a in arrays:
        if a.ndim < 1:
            raise ValueError(
                f"Expected input to have rank >= 1. Received scalar input {a}."
            )


def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise ValueError(
                "Expected input to have rank >= 2. "
                f"Received input with shape {a.shape}."
            )


def _assert_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise ValueError(
                "Expected a square matrix. "
                f"Received non-square input with shape {a.shape}"
            )


def _assert_a_b_compat(a, b):
    if a.ndim == b.ndim:
        if a.shape[-2] != b.shape[-2]:
            raise ValueError(
                "Incompatible shapes between `a` and `b`. "
                "Expected `a.shape[-2] == b.shape[-2]`. "
                f"Received: a.shape={a.shape}, b.shape={b.shape}"
            )
    elif a.ndim == b.ndim - 1:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                "Incompatible shapes between `a` and `b`. "
                "Expected `a.shape[-1] == b.shape[-1]`. "
                f"Received: a.shape={a.shape}, b.shape={b.shape}"
            )


class JVP(Operation):
    def __init__(self, has_aux=False, *, name=None):
        super().__init__(name=name)
        self.has_aux = has_aux

    def call(self, fun, primals, tangents):
        """Computes the JVP of `fun` at `primals` along `tangents`.

        Args:
            fun: A callable that takes tensors (or nested structures) as input
                 and returns a tensor (or nested structure) as output.
            primals: Input tensors (or nested structures) at which the Jacobian
                     of `fun` is evaluated.
            tangents: Tensors (or nested structures) representing the direction
                      vectors for the JVP. Must have the same structure as
                      `primals`.

        Returns:
            If `has_aux` is False:
                A tuple (primals_out, tangents_out) where:
                - primals_out: Output of `fun(*primals)`
                - tangents_out: JVP of `fun` at `primals` along `tangents`
            If `has_aux` is True:
                A tuple (primals_out, tangents_out, aux) where:
                - aux: Auxiliary data returned by `fun`
        """
        return backend.ops.linalg.jvp(
            fun, primals, tangents, has_aux=self.has_aux
        )

    def compute_output_spec(self, fun, primals, tangents):
        # Infer primal output spec
        if self.has_aux:
            primals_out_spec, aux_spec = backend.compute_output_spec(
                fun, *primals
            )
        else:
            primals_out_spec = backend.compute_output_spec(fun, *primals)

        # Tangents output should match primals output in structure and shape
        tangents_out_spec = tree.map_structure(
            lambda x: KerasTensor(x.shape, x.dtype), primals_out_spec
        )

        if self.has_aux:
            return primals_out_spec, tangents_out_spec, aux_spec
        return primals_out_spec, tangents_out_spec


@keras_export(["keras.ops.jvp", "keras.ops.linalg.jvp"])
def jvp(fun, primals, tangents, has_aux=False):
    """Computes a (forward-mode) Jacobian-vector product of `fun`.
    Args:
        fun: Function to be differentiated. Its arguments should be arrays,
            scalars, or standard Python containers of arrays or scalars. It
            should return an array, scalar, or standard Python container of
            arrays or scalars.
        primals: The primal values at which the Jacobian of `fun` should be
                evaluated. Should be either a tuple or a list of arguments,
                and its length should be equal to the number of positional
                parameters of `fun`.
        tangents: The tangent vector for which the Jacobian-vector product
                should be evaluated. Should be either a tuple or a list of
                tangents, with the same tree structure and array shapes as
                `primals`.
        has_aux: Optional, bool. Indicates whether `fun` returns a pair where
                the first element is considered the output of the mathematical
                function to be differentiated and the second element is
                auxiliary data. Default is False.

    Returns:
        If `has_aux` is False, returns a (`primals_out`, `tangents_out`) pair,
        where `primals_out` is `fun(*primals)`, and `tangents_out` is the
        Jacobian-vector product of `fun` evaluated at `primals` with
        `tangents`. The `tangents_out` value has the same Python tree
        structure and shapes as `primals_out`.

        If `has_aux` is True, returns a (`primals_out`, `tangents_out`, `aux`)
        tuple where `aux` is the auxiliary data returned by `fun`.

    Example:
    >>> from keras import ops
    >>> a1, a2 = ops.convert_to_tensor(0.1), ops.convert_to_tensor(0.2)
    >>> primals, tangents = ops.jvp(ops.sin, (a1,), (a2,))
    >>> primals
    0.09983342
    >>> tangents
    0.19900084
    """
    if any_symbolic_tensors((primals, tangents)):
        return JVP(has_aux=has_aux).symbolic_call(fun, primals, tangents)
    return backend.ops.linalg.jvp(fun, primals, tangents, has_aux=has_aux)
