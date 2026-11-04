import numpy as np
import scipy.linalg as sl

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy
from keras.src.backend.numpy.ops.core import convert_to_tensor


def cholesky(a, upper=False):
    return np.linalg.cholesky(a, upper=upper)


def cholesky_inverse(a, upper=False):
    identity = np.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    if upper:
        a_inv = np.matmul(inv_chol, inv_chol.T)
    else:
        a_inv = np.matmul(inv_chol.T, inv_chol)
    return a_inv


def det(a):
    return np.linalg.det(a)


def eig(a):
    return np.linalg.eig(a)


def eigh(a):
    return np.linalg.eigh(a)


def inv(a):
    return np.linalg.inv(a)


def lu_factor(a):
    if a.ndim == 2:
        return sl.lu_factor(a)

    m, n = a.shape[-2:]
    signature = "(m,n) -> (m,n), "
    signature += "(m)" if m <= n else "(n)"
    _lu_factor_gufunc = np.vectorize(
        sl.lu_factor,
        signature=signature,
    )
    return _lu_factor_gufunc(a)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims).astype(
        dtype
    )


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return np.linalg.qr(x, mode=mode)


def solve(a, b):
    if b.ndim == a.ndim - 1:
        b = np.expand_dims(b, axis=-1)
        return np.linalg.solve(a, b).squeeze(axis=-1)
    return np.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if a.ndim == 2:
        return sl.solve_triangular(a, b, lower=lower)

    _vectorized_solve_triangular = np.vectorize(
        lambda a, b: sl.solve_triangular(a, b, lower=lower),
        signature="(n,n),(n,m)->(n,m)",
    )
    if b.ndim == a.ndim - 1:
        b = np.expand_dims(b, axis=-1)
        return _vectorized_solve_triangular(a, b).squeeze(axis=-1)
    return _vectorized_solve_triangular(a, b)


def svd(x, full_matrices=True, compute_uv=True):
    return np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    return np.linalg.matrix_rank(x, tol=tol).astype("int32")


def matrix_power(a, n):
    a = convert_to_tensor(a)
    return np.linalg.matrix_power(a, n)


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    return np.linalg.pinv(x, rcond=rcond)


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("JVP is not supported by the Numpy backend.")
