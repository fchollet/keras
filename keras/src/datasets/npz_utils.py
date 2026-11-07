"""Utilities shared by the built-in datasets."""

import io
import math
import pickle
import zipfile

import numpy as np

# A dataset npz member is rejected as a decompression bomb when its declared
# size is above this floor and far larger than the bytes stored on disk. Genuine
# dataset members store real data and should not approach this expansion ratio.
_NPZ_MEMBER_BOMB_FLOOR_BYTES = 1 << 32  # 4 GiB
_NPZ_MEMBER_MAX_EXPANSION = 100

# Globals required to reconstruct the numpy object arrays stored inside the
# IMDB / Reuters `.npz` files. These datasets store ragged sequences as
# `dtype=object` arrays, which numpy can only persist as a pickle stream, so
# `np.load(allow_pickle=False)` refuses them outright ("Object arrays cannot be
# loaded when allow_pickle=False"). Rebuilding such an array uses exactly three
# globals -- `numpy.ndarray`, `numpy.dtype` and `multiarray._reconstruct` -- and
# nothing else (verified against the actual IMDB and Reuters files). Both the
# numpy < 2 (`numpy.core`) and numpy >= 2 (`numpy._core`) spellings of
# `multiarray` are listed so files written by either can be read. Anything
# outside this set (e.g. `os.system`, `builtins.eval`, `subprocess.Popen`) is
# refused, which prevents a tampered `.npz` from executing code through a pickle
# `__reduce__` gadget.
_ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)

_RECONSTRUCT_GLOBALS = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)

# Upper bound on the number of bytes an array in a dataset archive may
# describe. Every built-in dataset is far below it (the largest, IMDB, is tens
# of megabytes), while it keeps a tiny, crafted `.npy` header -- e.g. one that
# claims `shape=(2**40,)` -- from asking numpy for terabytes and raising a
# `MemoryError` that would escape `load_npz` as a denial of service.
_MAX_NPY_BYTES = 1 << 31  # 2 GiB


def _validate_shape(shape):
    """Returns `shape` as a tuple of Python ints, or refuses the stream.

    `numpy.core.multiarray._reconstruct` and `ndarray.__setstate__` both take a
    shape; validating it here keeps anything that is not a plain, non-negative
    tuple of integers out of numpy's C allocation code. Dimensions have to
    already *be* integers rather than something coercible to one: `int()` would
    silently truncate a float (`1.9` -> `1`) and raises `OverflowError` for
    `inf`, which would escape as something other than an unpickling error.
    """
    if not isinstance(shape, tuple):
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array whose shape is not a tuple "
            f"({type(shape).__name__}) while loading a Keras dataset."
        )
    for dimension in shape:
        # `bool` is a subclass of `int`, but a `True`/`False` dimension is not
        # something numpy ever writes; strings and floats are not integers at
        # all. Only the type name is reported, never `repr()`, so that a
        # crafted object cannot run code through its own `__repr__`.
        if isinstance(dimension, bool) or not isinstance(
            dimension, (int, np.integer)
        ):
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose shape contains a "
                f"non-integer dimension ({type(dimension).__name__}) while "
                "loading a Keras dataset."
            )
    shape = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in shape):
        raise pickle.UnpicklingError(
            f"Refusing to deserialize an array with a negative shape ({shape}) "
            "while loading a Keras dataset."
        )
    return shape


def _validate_dtype(dtype):
    """Returns `dtype` as a `numpy.dtype`, or refuses the stream."""
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError) as e:
        # `ValueError` covers a well-typed but malformed spec (for example the
        # dict `{'names': ['a', 'a']}`); without it that error escaped as a
        # plain `ValueError` instead of an unpickling error.
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array with an invalid dtype "
            f"({dtype!r}) while loading a Keras dataset."
        ) from e


def _nbytes(shape, dtype):
    """Number of bytes `shape` x `dtype` describes."""
    return dtype.itemsize * math.prod(shape)


class _RestrictedArray(np.ndarray):
    """An `ndarray` whose pickle state is validated before it is applied.

    The global allowlist above is not sufficient on its own: a stream that only
    ever asks for allowlisted globals can still build an array with one dtype
    and then hand numpy a *state* that carries another one. `_reconstruct()`
    allocates for the dtype it is given, while `array_setstate()` uses the
    dtype and shape of the state to decide how many bytes to copy, so a state
    describing more bytes than the array holds makes numpy memcpy past the end
    of the allocation.

    Every array produced by `RestrictedUnpickler` is therefore an instance of
    this class, and the state is checked against the array that was actually
    allocated before `ndarray.__setstate__` gets to it.
    """

    def __new__(
        cls,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        # `find_class` hands out `_RestrictedArray` for every `("numpy",
        # "ndarray")` lookup, so a stream that reaches `ndarray(...)` directly
        # through `REDUCE`/`NEWOBJ` lands here. Validate the request before
        # `np.ndarray.__new__` allocates in C: a huge shape must fail as an
        # unpickling error rather than a `MemoryError`, and external buffers or
        # custom strides are refused outright.
        shape = _validate_shape(shape)
        dtype = _validate_dtype(dtype)
        if _nbytes(shape, dtype) > _MAX_NPY_BYTES:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array declaring more than "
                f"{_MAX_NPY_BYTES} bytes while loading a Keras dataset."
            )
        if buffer is not None or offset != 0 or strides is not None:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array with an external buffer or "
                "custom strides while loading a Keras dataset."
            )
        return super().__new__(cls, shape, dtype=dtype, order=order)

    def __setstate__(self, state):
        # numpy's own `__reduce__` always writes the 5-tuple
        # (version, shape, dtype, is_fortran_order, raw_data).
        if not isinstance(state, tuple) or len(state) != 5:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state is not a "
                "5-element tuple while loading a Keras dataset."
            )
        version, shape, dtype, _is_fortran_order, raw_data = state
        if version != 1:
            raise pickle.UnpicklingError(
                f"Refusing to deserialize an array with unknown state version "
                f"{version} while loading a Keras dataset."
            )
        shape = _validate_shape(shape)
        dtype = _validate_dtype(dtype)

        # numpy serializes an object array as an empty `int8` placeholder plus
        # a state carrying the real dtype, shape and contents, and
        # `array_setstate()` allocates a fresh buffer for it. Any other array
        # was allocated with real data, and numpy reuses that buffer, so the
        # state must describe the very same array: a different dtype would
        # reinterpret the buffer, and more bytes than it holds would be copied
        # past its end.
        placeholder = self.size == 0 and self.nbytes == 0
        declared = _nbytes(shape, dtype)
        # Bound the array here as well as in `_reconstruct_array`: numpy's own
        # `__reduce__` allocates `(0,)` and defers the real shape and dtype to
        # this state, so a crafted stream can keep the constructor request tiny
        # while asking `array_setstate()` for a huge buffer.
        if declared > _MAX_NPY_BYTES:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array declaring more than "
                f"{_MAX_NPY_BYTES} bytes while loading a Keras dataset."
            )
        if not placeholder and dtype != self.dtype:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state changes "
                f"its dtype from {self.dtype!r} to {dtype!r} while loading a "
                "Keras dataset."
            )
        if not placeholder and declared > self.nbytes:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state declares "
                f"{declared} bytes over a {self.nbytes}-byte buffer; applying "
                "it would write out of bounds while loading a Keras dataset."
            )

        # `raw_data` is `None`, the list of objects of an object array, or the
        # raw bytes of a numeric one. numpy's `__reduce__` only ever writes a
        # list when `dtype.hasobject` is set, so a list on a numeric array (or
        # bytes on an object array) means a hand-crafted state and is refused
        # rather than passed on for numpy to reinterpret.
        if dtype.hasobject:
            if dtype != np.dtype("O"):
                # `dtype.hasobject` is also set for structured dtypes with
                # object fields and for subarray object dtypes, neither of
                # which the built-in datasets ever store; numpy's C
                # `array_setstate()` mis-handles them and escapes with a bare
                # `SystemError` instead of an unpickling error.
                raise pickle.UnpicklingError(
                    "Refusing to deserialize an array with a structured or "
                    f"subarray object dtype ({dtype!r}) while loading a "
                    "Keras dataset."
                )
            # numpy always writes an object array's contents as a list of
            # objects (an empty list for a zero-element array). `None` and
            # every other type mean a hand-crafted state: `None` used to slip
            # through the check and reach numpy, which raised a bare
            # `TypeError` ("object pickle not returning list") instead of an
            # unpickling error.
            if not isinstance(raw_data, list):
                raise pickle.UnpicklingError(
                    "Refusing to deserialize an object array whose pickle "
                    "state carries unexpected data "
                    f"({type(raw_data).__name__}) instead of a list of "
                    "objects while loading a Keras dataset."
                )
            # `array_setstate()` walks the array's slots and the list in
            # lockstep without bounds-checking the list, so a state holding
            # fewer objects than `shape` has elements makes numpy copy
            # objects from past the end of the list's internal array (a
            # segfault or, when the read lands on mapped memory, a dangling
            # pointer left in the array), and a longer list is silently
            # truncated. The count must match the declared shape exactly.
            num_elements = math.prod(shape)
            if len(raw_data) != num_elements:
                raise pickle.UnpicklingError(
                    "Refusing to deserialize an object array whose pickle "
                    f"state carries {len(raw_data)} objects for a {shape} "
                    f"shape of {num_elements} elements while loading a "
                    "Keras dataset."
                )
        elif not isinstance(raw_data, bytes):
            # A numeric array's contents are raw bytes; `None` or a list here
            # would either raise a bare `TypeError` in C or be reread as
            # object pointers.
            raise pickle.UnpicklingError(
                "Refusing to deserialize a numeric array whose pickle state "
                "carries unexpected data "
                f"({type(raw_data).__name__}) instead of raw bytes while "
                "loading a Keras dataset."
            )
        elif isinstance(raw_data, bytes) and len(raw_data) != declared:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state holds "
                f"{len(raw_data)} bytes for a {declared}-byte array while "
                "loading a Keras dataset."
            )
        super().__setstate__(state)


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that only allows numpy array reconstruction globals."""

    def load(self):
        # Unwrap the validating subclass from every array in the result, not
        # only the ones `load_npy_member` consumes: `cifar.py` calls `load()`
        # directly and gets back a dict of arrays, so normalization has to
        # happen here to cover both callers.
        return _normalize_array(super().load())

    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to deserialize `{module}.{name}` while loading a "
                "Keras dataset. The file may be corrupted or malicious."
            )
        obj = super().find_class(module, name)
        if (module, name) in _RECONSTRUCT_GLOBALS:
            reconstruct = obj

            def restricted_reconstruct(subtype, shape, dtype):
                return _reconstruct_array(reconstruct, subtype, shape, dtype)

            return restricted_reconstruct
        if (module, name) == ("numpy", "ndarray"):
            # `ndarray(...)` can also be reached directly through `REDUCE`, so
            # the class itself is replaced by the validating subclass.
            return _RestrictedArray
        return obj


def _reconstruct_array(reconstruct, subtype, shape, dtype):
    """`multiarray._reconstruct` returning a state-validating array."""
    if not (isinstance(subtype, type) and issubclass(subtype, np.ndarray)):
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array with a non-array subtype while "
            "loading a Keras dataset."
        )
    shape = _validate_shape(shape)
    dtype = _validate_dtype(dtype)
    # `_reconstruct` allocates for `shape` x `dtype` in C; refuse an oversized
    # request here so a crafted pickle cannot ask for terabytes before
    # `__setstate__` (or anything else) gets a chance to validate it.
    if _nbytes(shape, dtype) > _MAX_NPY_BYTES:
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array declaring more than "
            f"{_MAX_NPY_BYTES} bytes while loading a Keras dataset."
        )
    array = reconstruct(np.ndarray, shape, dtype)
    return array.view(_RestrictedArray)


def _stream_remaining(fp):
    """Bytes left in `fp` from its current position, or `None` if unknown."""
    try:
        position = fp.tell()
        fp.seek(0, io.SEEK_END)
        end = fp.tell()
        fp.seek(position)
    except (OSError, ValueError, io.UnsupportedOperation):
        return None
    return end - position


def _read_npy_header(fp):
    """Reads and validates a `.npy` header, leaving `fp` at the payload.

    Returns `(shape, dtype)`. Parsing the header up front, before numpy is
    handed the stream, lets `load_npy_member` bound the array an untrusted file
    describes and decide whether the payload is a pickle at all -- instead of
    calling `read_array` blindly and using a blanket `except ValueError` to
    guess.
    """
    fp.seek(0)
    try:
        version = np.lib.format.read_magic(fp)
    except (ValueError, EOFError, OSError) as e:
        raise ValueError(
            "Refusing to load a `.npy` member with an unreadable header "
            "while loading a Keras dataset."
        ) from e
    if version[0] == 1:
        read_header = np.lib.format.read_array_header_1_0
    elif version[0] in (2, 3):
        # numpy exposes no public `read_array_header_3_0`. The 3.0 format only
        # differs from 2.0 by encoding the header string as UTF-8 instead of
        # latin1; the 4-byte header-length layout we skip past to reach the
        # payload is identical, so 2.0's reader handles both.
        read_header = np.lib.format.read_array_header_2_0
    else:
        raise ValueError(f"Unsupported `.npy` file version: {version}.")
    try:
        shape, _fortran_order, dtype = read_header(fp)
    except (ValueError, EOFError, OSError) as e:
        raise ValueError(
            "Refusing to load a `.npy` member with a malformed header "
            "while loading a Keras dataset."
        ) from e
    # A malformed shape or dtype is a header-format error, not a pickle one:
    # keep it in the `ValueError` domain so a corrupt `.npy` header is not
    # reported as an unpickling error (a valid dtype spec only reaches here
    # through the header, never through a pickle state).
    try:
        return _validate_shape(shape), _validate_dtype(dtype)
    except pickle.UnpicklingError as e:
        raise ValueError(str(e)) from e


def _normalize_array(obj):
    """Recursively turns `_RestrictedArray` nodes back into plain `np.ndarray`.

    `_reconstruct_array` returns `_RestrictedArray` so that numpy's own
    `__setstate__` receives every array and can validate it. Object arrays nest
    arrays *inside* themselves and CIFAR stores each batch as a pickled dict of
    arrays, so unwrapping only the top-level view leaves nested arrays as
    `_RestrictedArray`. Downstream code that expects a plain array (a
    `type(sequence) is np.ndarray` check, `np.concatenate`, `np.asarray`, ...)
    would then see the validating subclass. Walk the result and unwrap every
    nested array before handing it back.
    """
    if isinstance(obj, _RestrictedArray):
        obj = obj.view(np.ndarray)
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # `ndarray.flat` writes through even for non-contiguous arrays,
            # where `reshape(-1)` would return a copy and lose the change.
            for index in range(obj.size):
                obj.flat[index] = _normalize_array(obj.flat[index])
        return obj
    if isinstance(obj, dict):
        # Only the values can hold arrays (an `ndarray` is unhashable, so it can
        # never be a key); recurse into them.
        return {key: _normalize_array(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize_array(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_array(item) for item in obj)
    return obj


def _reject_npz_member_bomb(info):
    """Raise if an npz member is a decompression bomb."""
    if (
        info.file_size > _NPZ_MEMBER_BOMB_FLOOR_BYTES
        and info.file_size > _NPZ_MEMBER_MAX_EXPANSION * info.compress_size
    ):
        from keras.src.utils.summary_utils import readable_memory_size

        raise ValueError(
            f"Refusing to load npz member '{info.filename}': it declares "
            f"{readable_memory_size(info.file_size)} but only "
            f"{readable_memory_size(info.compress_size)} are stored on disk; "
            "refusing to load a potential decompression bomb."
        )


def load_npy_member(fp):
    """Read a single `.npy` stream without allowing arbitrary unpickling.

    Numeric arrays are read with pickling disabled entirely. Object arrays
    (which genuinely require pickle) are read with `RestrictedUnpickler`, so
    only numpy array reconstruction is permitted.
    """
    shape, dtype = _read_npy_header(fp)
    declared = _nbytes(shape, dtype)
    # Refuse anything a tiny file could not possibly describe *before* numpy
    # allocates for it: a few-byte header claiming `shape=(2**40,)` used to
    # make `read_array` request terabytes and raise a `MemoryError` that
    # escaped `load_npz`.
    if declared > _MAX_NPY_BYTES:
        raise ValueError(
            f"Refusing to load a `.npy` member declaring {declared} bytes, "
            f"beyond the {_MAX_NPY_BYTES}-byte limit, while loading a Keras "
            "dataset."
        )
    if not dtype.hasobject:
        # Only a numeric array is bounded by the stream: its payload is exactly
        # `declared` raw bytes, so a short stream means the file is truncated.
        # (An object array's payload is a pickle stream, whose size is
        # unrelated to the pointer array it rebuilds.)
        remaining = _stream_remaining(fp)
        if remaining is not None and declared > remaining:
            raise ValueError(
                "Refusing to load a truncated `.npy` member: its header "
                f"declares {declared} bytes but only {remaining} remain in "
                "the stream while loading a Keras dataset."
            )
        # The header says numeric, so read it with pickle disabled. Reading it
        # up front also means a *corrupt* numeric member fails here, instead of
        # being rewound and re-fed to the unpickler by a blanket
        # `except ValueError` (which surfaced a misleading pickle-domain
        # `UnpicklingError`/`EOFError` for a merely truncated file).
        fp.seek(0)
        return np.lib.format.read_array(fp, allow_pickle=False)
    # Object array: its payload really is a pickle stream, so read it through
    # the allow-listing unpickler. `_read_npy_header` left `fp` at the payload.
    # `RestrictedUnpickler.load` already unwraps the validating subclass.
    return RestrictedUnpickler(fp).load()


def load_npz(path):
    """Safely load an `.npz` archive into a `dict` of arrays.

    This is a drop-in replacement for `np.load(path, allow_pickle=True)` for
    the built-in datasets that store ragged object arrays. Unlike
    `allow_pickle=True`, it only ever unpickles numpy arrays, so loading a
    maliciously crafted file cannot execute arbitrary code.

    Args:
        path: Path to the `.npz` file.

    Returns:
        A `dict` mapping each member name to its array.
    """
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = info.filename
            if not member.endswith(".npy"):
                continue
            _reject_npz_member_bomb(info)
            with archive.open(member) as raw:
                buffer = io.BytesIO(raw.read())
            arrays[member[: -len(".npy")]] = load_npy_member(buffer)
    return arrays
