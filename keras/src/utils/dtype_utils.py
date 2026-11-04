from keras.src import ops
from keras.src.backend.common import standardize_dtype

DTYPE_TO_SIZE = {
    **{f"float{i}": i for i in (16, 32, 64)},
    **{f"int{i}": i for i in (8, 16, 32, 64)},
    **{f"uint{i}": i for i in (8, 16, 32, 64)},
    "bfloat16": 16,
    "bool": 1,
}


def dtype_size(dtype):
    size = DTYPE_TO_SIZE.get(dtype, None)
    if size is None:
        raise ValueError(f"Invalid dtype: {dtype}")
    return size


def is_float(dtype):
    return "float" in dtype


def cast_to_common_dtype(tensors):
    """Cast a list of tensors to a common dtype.

    If any tensor is floating-point, all tensors are cast to a common
    floating-point dtype with sufficient precision. The promotion follows
    the highest precision floating dtype present, with special handling
    for mixed `float16` and `bfloat16`, which are promoted to `float32`.

    If no floating-point tensors are present, tensors are returned unchanged.

    Args:
        tensors: A list of tensors.

    Returns:
        List of tensors cast to a common dtype when needed.
    """
    highest_float = None
    highest_float_size = -1

    seen_float16 = False
    seen_bfloat16 = False

    for x in tensors:
        dtype = standardize_dtype(x.dtype)

        if is_float(dtype):
            if dtype == "float16":
                seen_float16 = True
            elif dtype == "bfloat16":
                seen_bfloat16 = True

            if highest_float is None or dtype_size(dtype) > highest_float_size:
                highest_float = dtype
                highest_float_size = dtype_size(dtype)

    # Promote mixed float16 + bfloat16 to float32
    # Do not downgrade if higher precision already found (e.g., float64)
    if seen_float16 and seen_bfloat16 and highest_float_size < 32:
        highest_float = "float32"

    if highest_float:
        tensors = [ops.cast(x, highest_float) for x in tensors]

    return tensors
