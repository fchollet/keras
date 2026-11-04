import functools

from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common.variables import standardize_dtype

BOOL_TYPES = ("bool",)
INT_TYPES = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
)
FLOAT_TYPES = ("bfloat16", "float16", "float32", "float64")
WEAK_TYPES = ("int", "float")
COMPLEX_TYPES = ("complex64", "complex128")
# We need to separate float8 from float because there are no implicit
# conversions from float8 dtypes to other dtypes.
# Ref: https://github.com/google/jax/issues/16705
FLOAT8_TYPES = ("float8_e4m3fn", "float8_e5m2")

# All supported dtypes in Keras. Kept as an ordered tuple: several test suites
# build parametrized cases by iterating it, and pytest-xdist requires a
# deterministic collection order across workers (a set/frozenset iterates in
# hash-randomized order that differs per process).
ALLOWED_DTYPES = (
    "float16",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "bool",
    "string",
    "float8_e4m3fn",
    "float8_e5m2",
    "complex64",
    "complex128",
)
ALLOWED_DTYPES_SET = frozenset(ALLOWED_DTYPES)
PYTHON_DTYPES_MAP = {
    bool: "bool",
    int: "int64" if config.backend() == "tensorflow" else "int32",
    float: "float32",
    str: "string",
    # special case for string value
    "int": "int64" if config.backend() == "tensorflow" else "int32",
    complex: "complex128" if config.backend() == "tensorflow" else "complex64",
}

# We adapted the type promotion lattice from JAX. Ref:
# https://github.com/google/jax/blob/main/jax/_src/dtypes.py


def _type_promotion_lattice():
    """
    Return the type promotion lattice in the form of a DAG.
    This DAG maps each type to its immediately higher type on the lattice.
    """
    (b1,) = BOOL_TYPES
    (u1, u2, u4, u8, i1, i2, i4, i8) = INT_TYPES
    bf, f2, f4, f8 = FLOAT_TYPES
    i_, f_ = WEAK_TYPES
    c64, c128 = COMPLEX_TYPES
    out = {
        b1: [i_],
        u1: [i2, u2],
        u2: [i4, u4],
        u4: [i8, u8],
        u8: [f_],
        i_: [u1, i1, c64],
        i1: [i2],
        i2: [i4],
        i4: [i8],
        i8: [f_],
        f_: [bf, f2],
        bf: [f4],
        f2: [f4],
        f4: [f8, c64],
        f8: [c128],
        c64: [c128],
        c128: [],
    }
    return out


def _make_lattice_upper_bounds():
    lattice = _type_promotion_lattice()
    upper_bounds = {node: {node} for node in lattice}
    for n in lattice:
        while True:
            new_upper_bounds = set().union(
                *(lattice[b] for b in upper_bounds[n])
            )
            if n in new_upper_bounds:
                raise ValueError(
                    f"cycle detected in type promotion lattice for node {n}"
                )
            if new_upper_bounds.issubset(upper_bounds[n]):
                break
            upper_bounds[n] |= new_upper_bounds
    return upper_bounds


LATTICE_UPPER_BOUNDS = _make_lattice_upper_bounds()


@functools.lru_cache(512)
def _least_upper_bound(*nodes):
    """Compute the least upper bound of a set of nodes.

    Args:
        nodes: sequence of entries from dtypes + weak_types

    Returns:
        The type representing the least upper bound of the input nodes on the
        promotion lattice.
    """
    # This function computes the least upper bound of a set of nodes N within a
    # partially ordered set defined by the lattice generated above.
    # Given a partially ordered set S, let the set of upper bounds of n ∈ S be
    #   UB(n) ≡ {m ∈ S | n ≤ m}
    # Further, for a set of nodes N ⊆ S, let the set of common upper bounds be
    # given by
    #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
    # Then the least upper bound of N is defined as
    #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
    # The definition of an upper bound implies that
    #   c ≤ d if and only if d ∈ UB(c),
    # so the LUB can be expressed:
    #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
    # or, equivalently:
    #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
    # By definition, LUB(N) has a cardinality of 1 for a partially ordered set.
    # Note a potential algorithmic shortcut: from the definition of CUB(N),
    # we have
    #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
    # So if N ∩ CUB(N) is nonempty, if follows that LUB(N) = N ∩ CUB(N).
    N = set(nodes)
    UB = LATTICE_UPPER_BOUNDS
    try:
        bounds = [UB[n] for n in N]
    except KeyError:
        dtype = next(n for n in N if n not in UB)
        raise ValueError(
            f"{dtype=} is not a valid dtype for Keras type promotion."
        )
    CUB = set.intersection(*bounds)
    LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
    if len(LUB) == 1:
        return LUB.pop()
    elif len(LUB) == 0:
        msg = (
            f"Input dtypes {tuple(str(n) for n in nodes)} have no available "
            "implicit dtype promotion path. Try explicitly casting inputs to "
            "the desired output type."
        )
        raise ValueError(msg)
    else:
        # If we get here, it means the lattice is ill-formed.
        raise ValueError(
            f"Internal Type Promotion error: {nodes} do not have a unique "
            f"least upper bound on the specified lattice; options are {LUB}. "
            "This is an unexpected error in Keras's internal logic; "
            "please report it to the maintainers."
        )


def _dtype_and_weaktype(value):
    """Return a (dtype, weak_type) tuple for the given input."""
    is_weak_type = False
    if value is int or value is float:
        # Note that we can't use `value in [int, float]` because the dtype
        # might be equal to python scalar types.
        # e.g, tf.float32 == float is True
        is_weak_type = True
    return standardize_dtype(value), is_weak_type


@functools.lru_cache(maxsize=None)
def _respect_weak_type(dtype, weak_type):
    """Return the weak dtype of `dtype` if `weak_type==True`."""
    if weak_type:
        if dtype == "bool":
            return dtype
        elif "float" in dtype:
            return "float"
        elif "int" in dtype:
            return "int"
        elif "complex" in dtype:
            return "complex"
        else:
            raise ValueError(
                "Invalid value for argument `dtype`. Expected one of "
                f"{ALLOWED_DTYPES}. Received: dtype={dtype}"
            )
    return dtype


@functools.lru_cache(maxsize=None)
def _resolve_weak_type(dtype, precision="32"):
    """Resolve weak type by the precision of `backend.floatx()`."""
    extended_allowed_dtypes = set(ALLOWED_DTYPES).union(WEAK_TYPES)
    if dtype not in extended_allowed_dtypes:
        raise ValueError(
            "Invalid value for argument `dtype`. Expected one of "
            f"{extended_allowed_dtypes}. Received: dtype={dtype}"
        )
    if precision not in ["16", "32", "64"]:
        raise ValueError(
            f"Invalid value for argument `precision`. Expected one of "
            f"('16', '32', '64'). Received: precision={precision}"
        )
    if dtype == "bfloat16":  # special case for bfloat16
        dtype_indicator = "f"
    else:
        dtype_indicator = dtype[:1]

    if dtype_indicator == "b":
        return "bool"
    elif dtype_indicator == "i":
        return f"int{precision}"
    elif dtype_indicator == "u":
        return f"uint{precision}"
    else:
        return f"float{precision}"


BIT64_TO_BIT32_DTYPE = {
    # Since TF variables require int64 to be placed on the GPU, we exclusively
    # enable the int64 dtype for TF.
    "int64": "int64" if config.backend() == "tensorflow" else "int32",
    "uint64": "uint32",
    "float64": "float64" if config.backend() == "tensorflow" else "float32",
    "complex128": "complex64",
}


def _lattice_result_type(*args):
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
    if len(dtypes) == 1:
        out_dtype = dtypes[0]
        out_weak_type = weak_types[0]
    elif len(set(dtypes)) == 1 and not all(weak_types):
        # Trivial promotion case. This allows extended dtypes through.
        out_dtype = dtypes[0]
        out_weak_type = False
    elif all(weak_types):
        # If all inputs are weakly typed, we compute the bound of the
        # strongly-typed counterparts and apply the weak type at the end. This
        # avoids returning the incorrect result with non-canonical weak types
        # (e.g. weak int16).
        out_dtype = _least_upper_bound(
            *{_respect_weak_type(d, False) for d in dtypes}
        )
        out_weak_type = True
    else:
        out_dtype = _least_upper_bound(
            *{_respect_weak_type(d, w) for d, w in zip(dtypes, weak_types)}
        )
        out_weak_type = any(out_dtype is t for t in WEAK_TYPES)

    out_weak_type = (out_dtype != "bool") and out_weak_type
    precision = config.floatx()[-2:]
    if out_weak_type:
        out_dtype = _resolve_weak_type(out_dtype, precision=precision)
    return _cap_to_32bit(out_dtype, dtypes, weak_types)


def _cap_to_32bit(out_dtype, dtypes, weak_types):
    """Caps a 64-bit `out_dtype` to 32-bit, unless `x64_mode` keeps it.

    The cap matches JAX's default behavior, which refuses 64-bit dtypes. In
    `"explicit_x64"` a result stays 64-bit only if a strongly typed input is
    64-bit. Python scalars (weak types) and 32-bit-only inputs never promote
    to 64-bit, as with JAX's `jax_explicit_x64_dtypes="allow"`.
    """
    if config.x64_mode() == "explicit_x64" and any(
        dtype in BIT64_TO_BIT32_DTYPE and not weak_type
        for dtype, weak_type in zip(dtypes, weak_types)
    ):
        return out_dtype
    return BIT64_TO_BIT32_DTYPE.get(out_dtype, out_dtype)


@keras_export("keras.backend.result_type")
def result_type(*dtypes):
    """Returns the type from applying the Keras type promotion rules.

    In general, each argument is first parsed by `standardize_dtype` and the
    resulting dtype is determined by the least upper bound of the type promotion
    lattice.

    Note: This function attempts to match the result of `jnp.result_type`.
    64-bit dtypes are capped to 32-bit unless `keras.config.set_x64_mode()`
    says otherwise.

    Args:
        dtypes: Input dtypes.

    Returns:
        The result dtype.

    Examples:

    >>> x = keras.ops.ones((1,), dtype="bfloat16")
    >>> keras.backend.result_type(x.dtype, int)
    "bfloat16"

    >>> x = keras.ops.ones((1,), dtype="int32")
    >>> y = keras.ops.ones((1,), dtype="float32")
    >>> keras.backend.result_type(x.dtype, y.dtype)
    "float32"

    >>> z= keras.ops.ones((1,), dtype='complex64')
    >>> keras.backend.result_type(z.dtype, int)
    "float64"

    """
    if len(dtypes) == 0:
        # If no dtypes provided, default to floatx, this matches
        # `ops.convert_to_tensor([])`
        return config.floatx()
    for dtype in dtypes:
        if dtype in FLOAT8_TYPES:
            raise ValueError(
                "There is no implicit conversions from float8 dtypes to others."
                f" You must cast it internally. Received: {dtypes}"
            )
    return _lattice_result_type(
        *(config.floatx() if arg is None else arg for arg in dtypes),
    )


def promote_to_float_dtype(dtype):
    """Returns the dtype to use for an operation that needs a float input.

    Integer and bool dtypes are promoted to `config.floatx()`. Float dtypes are
    returned unchanged, including `float64`, `bfloat16` and the `float8` types.

    This only computes the dtype. Callers do the cast themselves, since `cast`
    is backend specific.

    Args:
        dtype: The dtype of the input.

    Returns:
        The float dtype to cast the input to.
    """
    dtype = standardize_dtype(dtype)
    if "int" in dtype or dtype == "bool":
        return config.floatx()
    return dtype
