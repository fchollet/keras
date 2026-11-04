import torch

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.torch.ops.core import cast
from keras.src.backend.torch.ops.core import convert_to_tensor


def cholesky(x, upper=False):
    return torch.linalg.cholesky(x, upper=upper)


def cholesky_inverse(x, upper=False):
    return torch.cholesky_inverse(x, upper=upper)


def det(x):
    return torch.det(x)


def eig(x):
    return torch.linalg.eig(x)


def eigh(x):
    return torch.linalg.eigh(x)


def inv(x):
    return torch.linalg.inv(x)


def lu_factor(x):
    LU, pivots = torch.linalg.lu_factor(x)
    # torch returns pivots with 1-based indexing
    return LU, pivots - 1


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return torch.linalg.qr(x, mode=mode)


def solve(a, b):
    return torch.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if b.ndim == a.ndim - 1:
        b = torch.unsqueeze(b, axis=-1)
        return torch.linalg.solve_triangular(a, b, upper=not lower).squeeze(
            axis=-1
        )
    return torch.linalg.solve_triangular(a, b, upper=not lower)


def svd(x, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return torch.linalg.svdvals(x)
    return torch.linalg.svd(x, full_matrices=full_matrices)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # `torch.linalg.lstsq` defaults to the QR-based `gelsy` driver on CPU,
    # which interprets `rcond` differently from numpy and can return
    # large errors on near-rank-deficient inputs. Pin the SVD-based
    # `gelsd` driver when an explicit `rcond` is supplied so the result
    # matches numpy's `linalg.lstsq` semantics. CUDA only supports the
    # `gels` driver, which has no rcond cutoff, so fall back to SVD via
    # `pinv` to match numpy semantics there.
    if rcond is not None:
        if a.device.type == "cuda":
            return torch.linalg.pinv(a, rtol=rcond) @ b
        return torch.linalg.lstsq(a, b, rcond=rcond, driver="gelsd")[0]
    return torch.linalg.lstsq(a, b, rcond=rcond)[0]


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    # `torch.linalg.matrix_rank` uses `atol` for the absolute threshold
    # that numpy's `tol` also represents.
    return torch.linalg.matrix_rank(x, atol=tol).to(torch.int32)


def matrix_power(a, n):
    a = convert_to_tensor(a)
    return torch.linalg.matrix_power(a, n)


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    # `torch.linalg.pinv` expresses the threshold as `rtol` (relative
    # tolerance), with the same meaning as numpy's `rcond`.
    return torch.linalg.pinv(x, rtol=rcond)


def jvp(fun, primals, tangents, has_aux=False):
    return torch.func.jvp(fun, primals, tangents, has_aux=has_aux)
