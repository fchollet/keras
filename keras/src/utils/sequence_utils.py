import numpy as np

from keras.src.api_export import keras_export


@keras_export(
    [
        "keras.utils.pad_sequences",
        "keras.preprocessing.sequence.pad_sequences",
    ]
)
def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0,
):
    """Pads sequences to the same length.

    This function transforms a list (of length `num_samples`)
    of sequences (lists of integers)
    into a 2D NumPy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence in the list.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` until they are `num_timesteps` long.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.

    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding or removing values from the beginning of the sequence is the
    default.

    >>> sequence = [[1], [2, 3], [4, 5, 6]]
    >>> keras.utils.pad_sequences(sequence)
    array([[0, 0, 1],
           [0, 2, 3],
           [4, 5, 6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, value=-1)
    array([[-1, -1,  1],
           [-1,  2,  3],
           [ 4,  5,  6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, padding='post')
    array([[1, 0, 0],
           [2, 3, 0],
           [4, 5, 6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, maxlen=2)
    array([[0, 1],
           [2, 3],
           [5, 6]], dtype=int32)

    Note: when called on backend tensors with dynamic shapes inside a
    `tf.function`, the result is a symbolic tensor instead of a NumPy array.

    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to `0.`)

    Returns:
        NumPy array with shape `(len(sequences), maxlen)`
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    if _needs_tensor_path(sequences):
        return _pad_sequences_tensors(
            sequences,
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
            value=value,
        )
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.bytes_
    )
    if isinstance(value, str) and dtype is not object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


def _needs_tensor_path(sequences):
    """Whether the input contains tensors with dynamic shapes.

    Sequences with fully defined shapes keep the NumPy implementation;
    only inputs that may hold dynamic shapes during `tf.function`
    tracing are routed to the tensor implementation.
    """
    if _has_dynamic_shape(sequences):
        return True
    if isinstance(sequences, (list, tuple)):
        return any(_has_dynamic_shape(x) for x in sequences)
    return False


def _has_dynamic_shape(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return False
    is_fully_defined = getattr(shape, "is_fully_defined", None)
    if is_fully_defined is not None:
        # Framework-specific shape objects, e.g. `tf.TensorShape`.
        return not shape.is_fully_defined()
    if isinstance(shape, (tuple, list)):
        return any(not isinstance(dim, int) for dim in shape)
    return False


def _as_int64_vector(values):
    from keras.src import ops

    return ops.stack(
        [ops.cast(ops.convert_to_tensor(v), "int64") for v in values], axis=0
    )


def _statically_zero_length(seq):
    """Whether `seq` is a rank-1 value of static length 0."""
    shape = getattr(seq, "shape", None)
    if shape is None:
        return False
    if isinstance(shape, (tuple, list)):
        return tuple(shape) == (0,)
    is_fully_defined = getattr(shape, "is_fully_defined", None)
    if is_fully_defined is None:
        return False
    try:
        return shape.is_fully_defined() and shape.rank == 1 and shape[0] == 0
    except (TypeError, ValueError):
        return False


def _pad_sequences_tensors(
    sequences, maxlen, dtype, padding, truncating, value
):
    """`pad_sequences` for inputs that contain dynamic-shape tensors.

    Called when the function is traced inside `tf.function` and the
    individual sequences may be symbolic tensors with unknown lengths.
    """
    from keras.src import ops

    if padding not in ("pre", "post"):
        raise ValueError(f'Padding type "{padding}" not understood')
    if truncating not in ("pre", "post"):
        raise ValueError(f'Truncating type "{truncating}" not understood')

    if not isinstance(sequences, (list, tuple)):
        # A single stacked tensor holding one sequence per leading entry:
        # `(num_samples, seq_length, *features)`.
        return _pad_stacked_tensor(
            sequences, maxlen, dtype, padding, truncating, value
        )

    for seq in sequences:
        if not hasattr(seq, "__len__"):
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(seq)}"
            )
    seqs = [ops.convert_to_tensor(seq) for seq in sequences]
    lengths = [ops.cast(ops.shape(seq)[0], "int64") for seq in seqs]
    if maxlen is None:
        maxlen = ops.max(ops.stack(lengths, axis=0), axis=0)

    sample_ndim = 1
    for seq in seqs:
        sample_ndim = max(sample_ndim, len(seq.shape))
    sample_shape = None
    for seq in seqs:
        if len(seq.shape) == sample_ndim:
            sample_shape = [ops.shape(seq)[k] for k in range(1, sample_ndim)]
            break

    filled_rows = []
    for idx, seq in enumerate(seqs):
        if _statically_zero_length(seq):
            # A sequence with a known static length of zero stays empty,
            # matching the NumPy behavior of leaving the row filled with
            # `value`.
            filled_rows.append(
                ops.full(
                    tuple([maxlen] + (sample_shape or [])),
                    value,
                    dtype=dtype,
                )
            )
            continue
        li = lengths[idx]
        m = ops.minimum(li, maxlen)
        start = 0 if truncating == "post" else li - m
        ndim = len(seq.shape)
        trailing = [ops.shape(seq)[k] for k in range(1, ndim)]
        trunc = ops.slice(
            seq,
            _as_int64_vector([start] + [0] * (ndim - 1)),
            _as_int64_vector([m] + trailing),
        )
        trunc = ops.cast(trunc, dtype)
        pad = maxlen - m
        fill = ops.full(tuple([pad] + trailing), ops.cast(value, dtype))
        if padding == "pre":
            filled_rows.append(ops.concatenate([fill, trunc], axis=0))
        else:
            filled_rows.append(ops.concatenate([trunc, fill], axis=0))
    return ops.stack(filled_rows, axis=0)


def _pad_stacked_tensor(sequences, maxlen, dtype, padding, truncating, value):
    from keras.src import ops

    rank = len(sequences.shape)
    if rank < 2:
        raise ValueError(
            "`sequences` must be a list of iterables. "
            f"Found non-iterable: {str(sequences)}"
        )
    if maxlen is None:
        return sequences
    num_samples = ops.cast(ops.shape(sequences)[0], "int64")
    length = ops.cast(ops.shape(sequences)[1], "int64")
    m = ops.minimum(length, ops.cast(maxlen, "int64"))
    start = 0 if truncating == "post" else length - m
    trailing = [ops.shape(sequences)[k] for k in range(2, rank)]
    trunc = ops.slice(
        sequences,
        _as_int64_vector([0, start] + [0] * (rank - 2)),
        _as_int64_vector([num_samples, m] + trailing),
    )
    trunc = ops.cast(trunc, dtype)
    pad = ops.cast(maxlen, "int64") - m
    fill = ops.full(
        tuple([num_samples, pad] + trailing), ops.cast(value, dtype)
    )
    if padding == "pre":
        return ops.concatenate([fill, trunc], axis=1)
    return ops.concatenate([trunc, fill], axis=1)
