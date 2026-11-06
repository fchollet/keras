"""Storage codecs for sub-byte codes: two 4-bit or four 2-bit codes per byte
in bit fields, and five ternary codes per byte in base 3.

`pack_*` runs in NumPy once, at quantization time, and returns the packed
tensor with its shape and the original length of the packed axis.
`unpack_*` runs in backend ops, on every forward pass, and needs that
length back to trim the padding.
"""

import numpy as np

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.quantizers.utils import device_constant


def _pack_along_axis(arr_np, axis, values_per_byte, encode):
    """Packs `values_per_byte` consecutive values along `axis` into a byte.

    Runs in NumPy: packing happens once at quantization time, and NumPy
    wraps int8 shifts the way the format expects, where some accelerators
    do not. `encode` receives the `values_per_byte` interleaved slices
    (value `i` of every byte) and returns the byte array.

    Returns `(packed_np, orig_len)`.
    """
    axis = canonicalize_axis(axis, arr_np.ndim)
    arr_np = np.moveaxis(arr_np, axis, 0)
    n = arr_np.shape[0]
    pad = (-n) % values_per_byte
    if pad:
        pad_shape = (pad,) + arr_np.shape[1:]
        arr_np = np.concatenate(
            [arr_np, np.zeros(pad_shape, dtype=arr_np.dtype)], axis=0
        )
    fields = [arr_np[i::values_per_byte] for i in range(values_per_byte)]
    packed_np = np.moveaxis(encode(fields), 0, axis)
    return packed_np, n


def _unpack_along_axis(packed, orig_len, axis, decode):
    """Inverse of `_pack_along_axis` in backend ops (runs in forward passes).

    `decode` receives the packed tensor and returns its `values_per_byte`
    fields, each already in the output dtype. Fields are interleaved along
    `axis` (byte `j` holds values `[k*j, k*j + k - 1]`), then the padding
    is trimmed. The first and last axes need no transpose; those are the
    layouts the layers store (`axis=-1`) and the reverse embedding reads
    (`axis=0`).
    """
    rank = len(packed.shape)
    axis = canonicalize_axis(axis, rank)
    if axis == rank - 1:
        stacked = ops.stack(decode(packed), axis=rank)
        unpacked = ops.reshape(stacked, tuple(ops.shape(packed)[:-1]) + (-1,))
        return unpacked[..., :orig_len]
    if axis == 0:
        transposed = packed
    else:
        perm = [axis] + [i for i in range(rank) if i != axis]
        transposed = ops.transpose(packed, perm)
    stacked = ops.stack(decode(transposed), axis=1)
    unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(transposed)[1:]))
    unpacked = unpacked[:orig_len, ...]
    if axis != 0:
        inv_perm = [perm.index(i) for i in range(rank)]
        unpacked = ops.transpose(unpacked, inv_perm)
    return unpacked


def _check_pack_dtype(dtype, arr):
    if dtype not in ("int8", "uint8"):
        raise ValueError(
            f"Expected dtype to be 'int8' or 'uint8', but got '{dtype}'."
        )
    if standardize_dtype(arr.dtype) != dtype:
        raise TypeError(
            f"Expected {dtype} tensor for packing, got "
            f"{standardize_dtype(arr.dtype)}."
        )


def _check_unpack_dtype(dtype, packed):
    if dtype not in ("int8", "uint8"):
        raise ValueError(
            f"Expected dtype to be 'int8' or 'uint8', but got '{dtype}'."
        )
    if standardize_dtype(packed.dtype) not in ("int8", "uint8"):
        raise TypeError(
            f"Expected int8 or uint8 tensor for unpacking, got {packed.dtype}"
        )


def _bitfield_encode(bits, np_dtype):
    """`8 // bits` values per byte, value `i` in bits `[i*bits, (i+1)*bits)`."""
    mask = np.array((1 << bits) - 1, dtype=np_dtype)

    def encode(fields):
        packed = None
        for i, values in enumerate(fields):
            field = np.left_shift(
                np.bitwise_and(values.astype(np_dtype), mask),
                np.array(i * bits, dtype=np_dtype),
            )
            packed = field if packed is None else np.bitwise_or(packed, field)
        return packed.astype(np_dtype)

    return encode


def _bitfield_decode(bits, dtype):
    """Inverse of `_bitfield_encode`; sign-extends when `dtype` is int8."""

    def decode(x):
        # Tensors, not operators with Python numbers: `&` and `^` are
        # logical on a `KerasTensor`.
        mask = device_constant((1 << bits) - 1, x.dtype)
        half = device_constant(1 << (bits - 1), x.dtype)
        fields = []
        for i in range(8 // bits):
            field = ops.bitwise_and(ops.right_shift(x, i * bits), mask)
            if dtype == "int8":
                # Branchless sign extension: (x ^ half) - half.
                field = ops.subtract(ops.bitwise_xor(field, half), half)
            fields.append(ops.cast(field, dtype))
        return fields

    return decode


_TRIT_PLACES = (1, 3, 9, 27, 81)


def _ternary_encode(fields):
    """Five trits per byte in base 3: `sum((t_i + 1) * 3**i)`, at most 242."""
    packed = np.zeros(fields[0].shape, dtype=np.int32)
    for place, values in zip(_TRIT_PLACES, fields):
        # Clip before shifting: round(1.6) = 2 would overflow a digit.
        digits = np.clip(np.round(values).astype(np.int32), -1, 1) + 1
        packed = packed + digits * place
    return packed.astype(np.uint8)


def _ternary_decode(x):
    codes = ops.cast(x, "int32")
    codes = ops.where(codes < 0, codes + 256, codes)
    # With operators, the Python numbers stay scalars on torch (see
    # `device_constant`).
    return [
        ops.cast((codes // place) % 3 - 1, "int8") for place in _TRIT_PLACES
    ]


@keras_export("keras.quantizers.pack_int4")
def pack_int4(arr, axis=0, dtype="int8"):
    """Pack an int4 tensor into an int8 tensor with packed nibbles.

    The input values must already be int8 in the signed range `[-8, 7]` and
    represent the desired int4 values. Packing is performed along the specified
    axis (default is 0).

    For every two consecutive rows, the **low nibble** of the output byte
    stores the value from the first row, and the **high nibble** stores
    the value from the second row.

    Args:
        arr: An `int8` or `uint8` tensor containing int4 values in the range
            `[-8, 7]`.
        axis: The axis along which to pack the tensor. Defaults to 0.
        dtype: The data type of the input and packed tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_rows)` where `packed` is
            the packed int8 tensor with int4 values stored in nibbles,
            `packed_shape` is the shape of the packed tensor, and `orig_rows`
            is the original (unpacked) row count prior to any padding that may
            have been inserted when an odd number of rows is supplied.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int4
    >>> codes = np.array([[-3, 7, 2], [-8, 1, 0]], dtype="int8")
    >>> packed, packed_shape, orig_len = pack_int4(codes, axis=1)
    >>> np.asarray(packed)  # two nibbles per byte; axis 1 padded from 3 to 4
    array([[125,   2],
           [ 24,   0]], dtype=int8)
    >>> packed_shape, orig_len
    ((2, 2), 3)
    ```
    """
    _check_pack_dtype(dtype, arr)
    packed_np, orig_len = _pack_along_axis(
        ops.convert_to_numpy(arr), axis, 2, _bitfield_encode(4, np.dtype(dtype))
    )
    return ops.convert_to_tensor(packed_np), tuple(packed_np.shape), orig_len


@keras_export("keras.quantizers.unpack_int4")
def unpack_int4(packed, orig_len, axis=0, dtype="int8"):
    """Unpack a packed int4 back to an int8 tensor in the range [-8, 7].

    This function reverses the packing performed by `pack_int4`, restoring
    the original int8 tensor (values in the range [-8, 7]) from a packed int8
    tensor where each element contains two int4 values (one in the lower nibble,
    one in the upper nibble).

    The function restores the original axis order and removes any
    padding that was added during packing.

    Args:
        packed: An int8 tensor containing packed int4 values along the
            specified axis. Each int8 value encodes two int4 values.
        orig_len: The original (unpadded) length of the axis that was
            packed. This is used to remove any padding that may have
            been added during packing to ensure an even number of rows.
        axis: The axis along which the tensor was packed. Defaults to 0.
        dtype: The data type of the input and unpacked tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        unpacked: An int8 tensor with the same shape as the original
            (unpacked) tensor, with values in the range [-8, 7].

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int4, unpack_int4
    >>> codes = np.array([[-3, 7, 2], [-8, 1, 0]], dtype="int8")
    >>> packed, _, orig_len = pack_int4(codes, axis=1)
    >>> np.asarray(unpack_int4(packed, orig_len, axis=1))  # padding removed
    array([[-3,  7,  2],
           [-8,  1,  0]], dtype=int8)
    ```
    """
    _check_unpack_dtype(dtype, packed)
    return _unpack_along_axis(
        packed, orig_len, axis, _bitfield_decode(4, dtype)
    )


@keras_export("keras.quantizers.pack_int2")
def pack_int2(arr, axis=0, dtype="int8"):
    """Pack an int2 tensor into an int8 tensor with 4 values per byte.

    The input values must already be int8/uint8 representing the desired int2
    values (signed range `[-2, 1]`, or unsigned range `[0, 3]`). Packing is
    performed along the specified axis (default is 0). Four consecutive values
    along the packing axis are stored in a single output byte, from the least
    significant 2-bit field to the most significant one.

    This mirrors the design of `pack_int4` (padding + original-length trim),
    but achieves a 4x rather than 2x storage reduction, which is what makes
    2-bit storage worthwhile relative to a plain uint8 tensor.

    Args:
        arr: An `int8` or `uint8` tensor containing int2 values in the range
            `[-2, 1]` (signed) or `[0, 3]` (unsigned).
        axis: The axis along which to pack the tensor. Defaults to 0.
        dtype: The data type of the input and packed tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_len)` where `packed` is
            the packed tensor with four int2 values per byte, `packed_shape`
            is the shape of the packed tensor, and `orig_len` is the original
            (unpacked) length along `axis` prior to any padding that was
            inserted to reach a multiple of four.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int2
    >>> codes = np.array([[-2, 1, 0, -1, 1]], dtype="int8")
    >>> packed, packed_shape, orig_len = pack_int2(codes, axis=1)
    >>> np.asarray(packed)  # four 2-bit fields per byte; axis 1 padded to 8
    array([[-58,   1]], dtype=int8)
    >>> packed_shape, orig_len
    ((1, 2), 5)
    ```
    """
    _check_pack_dtype(dtype, arr)
    packed_np, orig_len = _pack_along_axis(
        ops.convert_to_numpy(arr), axis, 4, _bitfield_encode(2, np.dtype(dtype))
    )
    return ops.convert_to_tensor(packed_np), tuple(packed_np.shape), orig_len


@keras_export("keras.quantizers.unpack_int2")
def unpack_int2(packed, orig_len, axis=0, dtype="int8"):
    """Unpack a packed int2 tensor back to an int8 tensor.

    This reverses `pack_int2`, restoring the original tensor whose values lie in
    `[-2, 1]` (signed) or `[0, 3]` (unsigned) from a packed tensor where each
    byte stores four int2 values. It restores the original axis order and
    removes any padding that was added during packing.

    Args:
        packed: An `int8` or `uint8` tensor with four int2 values per element
            along the specified axis.
        orig_len: The original (unpadded) length of the packed axis. Used to
            trim the padding inserted during packing.
        axis: The axis along which the tensor was packed. Defaults to 0.
        dtype: The data type of the input and unpacked tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        unpacked: A tensor with the same shape as the original (unpacked)
            tensor.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int2, unpack_int2
    >>> codes = np.array([[-2, 1, 0, -1, 1]], dtype="int8")
    >>> packed, _, orig_len = pack_int2(codes, axis=1)
    >>> np.asarray(unpack_int2(packed, orig_len, axis=1))
    array([[-2,  1,  0, -1,  1]], dtype=int8)
    ```
    """
    _check_unpack_dtype(dtype, packed)
    return _unpack_along_axis(
        packed, orig_len, axis, _bitfield_decode(2, dtype)
    )


@keras_export("keras.quantizers.pack_ternary")
def pack_ternary(arr, axis=0):
    """Pack a ternary tensor into a `uint8` tensor at ~1.6 bits per value.

    The input values must be in `{-1, 0, +1}`. Five ternary values (trits) are
    packed into a single `uint8` byte using base-3 encoding, which is exact
    because `3 ** 5 == 243 <= 256`. This is the information-theoretic floor for
    ternary weights (`log2(3) ~= 1.58` bits/value) and is strictly denser than
    any integer format: ~2.5x denser than int4 and ~5x denser than int8, with
    no loss (the stored values are exactly the original `{-1, 0, +1}`).

    Each trit `t` is shifted to an unsigned digit `d = t + 1` in `{0, 1, 2}`.
    Five consecutive digits `d0..d4` along `axis` are then combined into one
    byte as `d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4` (max value `242`). If the axis
    length is not a multiple of 5 it is padded with zero-trits, which are
    removed on unpacking.

    Args:
        arr: A tensor whose values are in `{-1, 0, +1}` (any numeric dtype;
            values are rounded to the nearest integer and clipped to
            `[-1, 1]` before packing).
        axis: The axis along which to pack the tensor. Defaults to 0.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_len)` where `packed` is the
            packed `uint8` tensor, `packed_shape` is its shape, and `orig_len`
            is the original (unpadded) length of `axis`, needed by
            `unpack_ternary` to strip padding.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_ternary
    >>> trits = np.array([[1, -1, 0, 1, 1, -1]], dtype="int8")
    >>> packed, packed_shape, orig_len = pack_ternary(trits, axis=1)
    >>> np.asarray(packed)  # five trits per byte in base 3; axis 1 padded to 10
    array([[227, 120]], dtype=uint8)
    >>> packed_shape, orig_len
    ((1, 2), 6)
    ```
    """
    packed_np, orig_len = _pack_along_axis(
        ops.convert_to_numpy(arr), axis, 5, _ternary_encode
    )
    return ops.convert_to_tensor(packed_np), tuple(packed_np.shape), orig_len


@keras_export("keras.quantizers.unpack_ternary")
def unpack_ternary(packed, orig_len, axis=0):
    """Unpack a base-3 packed `uint8` tensor back to ternary `{-1, 0, +1}`.

    This reverses `pack_ternary`, restoring an `int8` tensor whose values are
    in `{-1, 0, +1}`. The original axis order is preserved and any padding
    added during packing is removed.

    Args:
        packed: A `uint8` tensor produced by `pack_ternary`, with five trits
            encoded in each byte along `axis`.
        orig_len: The original (unpadded) length of the packed axis, used to
            strip padding.
        axis: The axis along which the tensor was packed. Defaults to 0.

    Returns:
        An `int8` tensor with values in `{-1, 0, +1}` and the original
        (unpacked) shape along `axis`.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_ternary, unpack_ternary
    >>> trits = np.array([[1, -1, 0, 1, 1, -1]], dtype="int8")
    >>> packed, _, orig_len = pack_ternary(trits, axis=1)
    >>> np.asarray(unpack_ternary(packed, orig_len, axis=1))
    array([[ 1, -1,  0,  1,  1, -1]], dtype=int8)
    ```
    """
    packed_dtype = standardize_dtype(packed.dtype)
    if packed_dtype not in ("uint8", "int8"):
        raise TypeError(
            "`unpack_ternary` expects a `uint8` or `int8` tensor produced by "
            f"`pack_ternary`. Received dtype: {packed_dtype}"
        )
    return _unpack_along_axis(packed, orig_len, axis, _ternary_decode)
