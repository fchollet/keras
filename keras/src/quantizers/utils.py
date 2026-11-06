import re

from keras.src import ops


def device_constant(value, dtype):
    """Returns the Python number `value` as a 0-d `dtype` tensor.

    The tensor is made on the device, from no host data. An op that gets
    a Python number, `ops.full` included, makes a tensor of it on the
    host, and torch copies that tensor to the device on each call. A CUDA
    graph capture refuses the copy. A Python number that is the operand
    of a tensor operator stays a scalar on torch. The cast keeps `dtype`
    where NumPy promotes the product: `bfloat16` times a Python float is
    `float32` there.

    Args:
        value: A Python number.
        dtype: The dtype of the tensor.

    Returns:
        The 0-d tensor.
    """
    return ops.cast(ops.ones((), dtype) * value, dtype)


def should_quantize_layer(layer, filters):
    """Determines if a layer should be quantized based on filters.

    Args:
        layer: The layer to check.
        filters: A regex string, a list of regex strings, or a callable.
            If None, returns True.

    Returns:
        True if the layer should be quantized, False otherwise.
    """
    if filters is None:
        return True
    if isinstance(filters, str):
        return bool(re.search(filters, layer.name))
    if isinstance(filters, (list, tuple)):
        return any(re.search(pat, layer.name) for pat in filters)
    if callable(filters):
        return filters(layer)
    return True
