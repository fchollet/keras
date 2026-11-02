import tensorflow as tf

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.tensorflow.ops.core import cast
from keras.src.backend.tensorflow.ops.core import convert_to_tensor


def cholesky(a, upper=False):
    out = tf.linalg.cholesky(a)
    # tf.linalg.cholesky simply returns NaNs for non-positive definite matrices
    out = tf.debugging.check_numerics(out, "Cholesky")
    if upper:
        return tf.linalg.adjoint(out)
    return out


def cholesky_inverse(a, upper=False):
    identity = tf.eye(num_rows=tf.shape(a)[-1], dtype=a.dtype)
    inv_chol = tf.linalg.triangular_solve(a, identity, lower=not upper)
    if upper:
        a_inv = tf.matmul(inv_chol, inv_chol, transpose_b=True)
    else:
        a_inv = tf.matmul(inv_chol, inv_chol, transpose_a=True)
    return a_inv


def det(a):
    return tf.linalg.det(a)


def eig(a):
    return tf.linalg.eig(a)


def eigh(a):
    return tf.linalg.eigh(a)


def inv(a):
    return tf.linalg.inv(a)


def lu_factor(a):
    lu, p = tf.linalg.lu(a)
    return lu, tf.math.invert_permutation(p)


def norm(x, ord=None, axis=None, keepdims=False):
    from keras.src.backend.tensorflow.ops.numpy import moveaxis

    x = convert_to_tensor(x)
    x_shape = x.shape
    ndim = x_shape.rank

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    if any(a < -ndim or a >= ndim for a in axis):
        raise ValueError(
            "All `axis` values must be in the range [-ndim, ndim). "
            f"Received inputs with ndim={ndim}, while axis={axis}"
        )
    axis = axis[0] if len(axis) == 1 else axis
    num_axes = 1 if isinstance(axis, int) else len(axis)

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)

    # Ref: jax.numpy.linalg.norm
    if num_axes == 1:
        if ord is None or ord == 2:
            return tf.sqrt(
                tf.reduce_sum(x * tf.math.conj(x), axis=axis, keepdims=keepdims)
            )
        elif ord == float("inf"):
            return tf.math.reduce_max(
                tf.math.abs(x), axis=axis, keepdims=keepdims
            )
        elif ord == float("-inf"):
            return tf.math.reduce_min(
                tf.math.abs(x), axis=axis, keepdims=keepdims
            )
        elif ord == 0:
            return tf.math.reduce_sum(
                tf.cast(tf.not_equal(x, 0), dtype=x.dtype),
                axis=axis,
                keepdims=keepdims,
            )
        elif isinstance(ord, str):
            raise ValueError(
                f"Invalid `ord` argument for vector norm. Received: ord={ord}"
            )
        else:
            ord = convert_to_tensor(ord, dtype=x.dtype)
            out = tf.math.reduce_sum(
                tf.pow(tf.math.abs(x), ord), axis=axis, keepdims=keepdims
            )
            return tf.pow(out, 1.0 / ord)
    elif num_axes == 2:
        row_axis, col_axis = axis[0], axis[1]
        row_axis = row_axis + ndim if row_axis < 0 else row_axis
        col_axis = col_axis + ndim if col_axis < 0 else col_axis
        if ord is None or ord == "fro":
            return tf.sqrt(
                tf.reduce_sum(x * tf.math.conj(x), axis=axis, keepdims=keepdims)
            )
        elif ord == 1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            x = tf.math.reduce_max(
                tf.reduce_sum(tf.math.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis,
                keepdims=keepdims,
            )
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis,
                keepdims=keepdims,
            )
        elif ord == float("inf"):
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            x = tf.math.reduce_max(
                tf.reduce_sum(tf.math.abs(x), axis=col_axis, keepdims=keepdims),
                axis=row_axis,
                keepdims=keepdims,
            )
        elif ord == float("-inf"):
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=col_axis, keepdims=keepdims),
                axis=row_axis,
                keepdims=keepdims,
            )
        elif ord in ("nuc", 2, -2):
            x = moveaxis(x, axis, (-2, -1))
            if ord == -2:
                x = tf.math.reduce_min(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            elif ord == 2:
                x = tf.math.reduce_max(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            else:
                x = tf.math.reduce_sum(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            if keepdims:
                x = tf.expand_dims(x, axis[0])
                x = tf.expand_dims(x, axis[1])
        else:
            raise ValueError(
                f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
            )
        return x
    else:
        raise ValueError(f"Invalid axis values. Received: axis={axis}")


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)


def solve(a, b):
    # tensorflow.linalg.solve only supports same rank inputs
    if b.shape.ndims == a.shape.ndims - 1:
        b = tf.expand_dims(b, axis=-1)
        return tf.squeeze(tf.linalg.solve(a, b), axis=-1)
    return tf.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if b.shape.ndims == a.shape.ndims - 1:
        b = tf.expand_dims(b, axis=-1)
        return tf.squeeze(
            tf.linalg.triangular_solve(a, b, lower=lower), axis=-1
        )
    return tf.linalg.triangular_solve(a, b, lower=lower)


def svd(x, full_matrices=True, compute_uv=True):
    if compute_uv is False:
        return tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=False)
    s, u, v = tf.linalg.svd(
        x, full_matrices=full_matrices, compute_uv=compute_uv
    )
    return u, s, tf.linalg.adjoint(v)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("Leading dimensions of input arrays must match")
    b_orig_ndim = b.ndim
    if b_orig_ndim == 1:
        b = b[:, None]
    if a.ndim != 2:
        raise TypeError(
            f"{a.ndim}-dimensional array given. Array must be two-dimensional"
        )
    if b.ndim != 2:
        raise TypeError(
            f"{b.ndim}-dimensional array given. "
            "Array must be one or two-dimensional"
        )
    m, n = a.shape
    dtype = a.dtype
    eps = tf.experimental.numpy.finfo(dtype).eps
    if a.shape == ():
        s = tf.zeros(0, dtype=a.dtype)
        x = tf.zeros((n, *b.shape[1:]), dtype=a.dtype)
    else:
        if rcond is None:
            rcond = eps * max(n, m)
        else:
            rcond = tf.where(rcond < 0, eps, rcond)
        u, s, vt = svd(a, full_matrices=False)
        mask = s >= tf.convert_to_tensor(rcond, dtype=s.dtype) * s[0]
        safe_s = tf.cast(tf.where(mask, s, 1), dtype=a.dtype)
        s_inv = tf.where(mask, 1 / safe_s, 0)[:, tf.newaxis]
        u_t_b = tf.matmul(tf.transpose(tf.math.conj(u)), b)
        x = tf.matmul(tf.transpose(tf.math.conj(vt)), s_inv * u_t_b)

    if b_orig_ndim == 1:
        x = tf.reshape(x, [-1])
    return x


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    return tf.linalg.matrix_rank(x, tol=tol)


def matrix_power(a, n):
    a = convert_to_tensor(a)
    if n == 0:
        return tf.eye(
            num_rows=tf.shape(a)[-1],
            batch_shape=tf.shape(a)[:-2],
            dtype=a.dtype,
        )

    if n < 0:
        a = inv(a)
        n = abs(n)

    if n == 1:
        return a

    result = tf.eye(
        num_rows=tf.shape(a)[-1],
        batch_shape=tf.shape(a)[:-2],
        dtype=a.dtype,
    )
    base = a
    while n > 0:
        if n % 2 == 1:
            result = tf.matmul(result, base)
        base = tf.matmul(base, base)
        n //= 2
    return result


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    return tf.linalg.pinv(x, rcond=rcond)


def jvp(fun, primals, tangents, has_aux=False):
    primal_flat = tf.nest.flatten(primals)
    tangent_flat = tf.nest.flatten(tangents)

    tangent_flat = [
        tf.cast(t, p.dtype) for t, p in zip(tangent_flat, primal_flat)
    ]

    with tf.autodiff.ForwardAccumulator(primal_flat, tangent_flat) as acc:
        if has_aux:
            primals_out, aux = fun(*primals)
        else:
            primals_out = fun(*primals)

        primals_out_flat = tf.nest.flatten(primals_out)
        tangents_out_flat = [acc.jvp(po) for po in primals_out_flat]

    tangents_out = tf.nest.pack_sequence_as(primals_out, tangents_out_flat)

    if has_aux:
        return primals_out, tangents_out, aux
    return primals_out, tangents_out
