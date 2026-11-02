import binascii
import codecs
import marshal
import types as python_types


def default(method):
    """Decorates a method to detect overrides in subclasses."""
    method._is_default = True
    return method


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)


def func_dump(func):
    """Serializes a user-defined function.

    Args:
        func: the function to serialize.

    Returns:
        A tuple `(code, defaults, closure)`.
    """
    raw_code = marshal.dumps(func.__code__)
    code = codecs.encode(raw_code, "base64").decode("ascii")
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
    """Deserializes a user defined function.

    Args:
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    Returns:
        A function object.
    """
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
        if isinstance(defaults, list):
            defaults = tuple(defaults)

    def ensure_value_to_cell(value):
        """Ensures that a value is converted to a python cell object.

        Args:
            value: Any value that needs to be casted to the cell type

        Returns:
            A value wrapped as a cell object (see function "func_load")
        """

        def dummy_fn():
            value  # just access it so it gets captured in .__closure__

        cell_value = dummy_fn.__closure__[0]
        if not isinstance(value, type(cell_value)):
            return cell_value
        return value

    if closure is not None:
        closure = tuple(ensure_value_to_cell(_) for _ in closure)
    try:
        raw_code = codecs.decode(code.encode("ascii"), "base64")
    except (UnicodeEncodeError, binascii.Error):
        raw_code = code.encode("raw_unicode_escape")
    code = marshal.loads(raw_code)
    if globs is None:
        globs = globals()
    return python_types.FunctionType(
        code, globs, name=code.co_name, argdefs=defaults, closure=closure
    )


def to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    Args:
        x: target object to be normalized.

    Returns:
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]


def remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    Args:
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    Returns:
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


def removeprefix(x, prefix):
    """Backport of `removeprefix` from PEP-616 (Python 3.9+)"""

    if len(prefix) > 0 and x.startswith(prefix):
        return x[len(prefix) :]
    else:
        return x


def removesuffix(x, suffix):
    """Backport of `removesuffix` from PEP-616 (Python 3.9+)"""

    if len(suffix) > 0 and x.endswith(suffix):
        return x[: -len(suffix)]
    else:
        return x


def remove_by_id(lst, value):
    """Remove a value from a list by id."""
    for i, v in enumerate(lst):
        if id(v) == id(value):
            del lst[i]
            return


def pythonify_logs(logs):
    """Flatten and convert log values to Python-native types.

    This function attempts to convert dict value by `float(value)` and skips
    the conversion if it fails.

    Args:
        logs: A dict containing log values.

    Returns:
        A flattened dict with values converted to Python-native types if
        possible.
    """
    from keras.src import backend

    logs = logs or {}
    result = {}
    for key, value in sorted(logs.items()):
        if isinstance(value, dict):
            result.update(pythonify_logs(value))
        else:
            try:
                # Prevent torch compiler from breaking the graph.
                if backend.ops.is_tensor(value):
                    value = backend.ops.convert_to_numpy(value)
                value = float(value)
            except:
                pass
            result[key] = value
    return result
