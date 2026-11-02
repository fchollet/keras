import inspect
import os
import traceback
import types
from functools import wraps

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype

_EXCLUDED_PATHS = (
    os.path.abspath(os.path.join(__file__, "..", "..")),
    os.path.join("tensorflow", "python"),
)


@keras_export("keras.config.enable_traceback_filtering")
def enable_traceback_filtering():
    """Turn on traceback filtering.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.disable_traceback_filtering()` and
    `keras.config.is_traceback_filtering_enabled()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.
    """
    global_state.set_global_attribute("traceback_filtering", True)


@keras_export("keras.config.disable_traceback_filtering")
def disable_traceback_filtering():
    """Turn off traceback filtering.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.enable_traceback_filtering()` and
    `keras.config.is_traceback_filtering_enabled()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.
    """
    global_state.set_global_attribute("traceback_filtering", False)


@keras_export("keras.config.is_traceback_filtering_enabled")
def is_traceback_filtering_enabled():
    """Check if traceback filtering is enabled.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.enable_traceback_filtering()` and
    `keras.config.disable_traceback_filtering()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.

    Returns:
        Boolean, `True` if traceback filtering is enabled,
        and `False` otherwise.
    """
    return global_state.get_global_attribute("traceback_filtering", True)


def include_frame(fname):
    for exclusion in _EXCLUDED_PATHS:
        if exclusion in fname:
            return False
    return True


def _process_traceback_frames(tb):
    """Iterate through traceback frames and return a new, filtered traceback."""
    last_tb = None
    tb_list = list(traceback.walk_tb(tb))
    for f, line_no in reversed(tb_list):
        if include_frame(f.f_code.co_filename):
            last_tb = types.TracebackType(last_tb, f, f.f_lasti, line_no)
    if last_tb is None and tb_list:
        # If no frames were kept during filtering, create a new traceback
        # from the outermost function.
        f, line_no = tb_list[-1]
        last_tb = types.TracebackType(last_tb, f, f.f_lasti, line_no)
    return last_tb


def filter_traceback(fn):
    """Filter out Keras-internal traceback frames in exceptions raised by fn."""

    @wraps(fn)
    def error_handler(*args, **kwargs):
        if not is_traceback_filtering_enabled():
            return fn(*args, **kwargs)

        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            # To get the full stack trace, call:
            # `keras.config.disable_traceback_filtering()`
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb

    return error_handler


def _format_raw_arguments_fallback(args, kwargs):
    """Best-effort `args=..., kwargs=...` dump for the error-message context.

    Used only when `inject_argument_info_in_error` cannot bind `args` and
    `kwargs` to the callee's signature.

    Returns:
        A single string formatted like the normal per-parameter listing, or
        `None` if there was nothing to report or the raw values could not
        be formatted either.
    """
    if not args and not kwargs:
        return None
    try:
        # `map_structure` mirrors the input structure, so a nested argument
        # comes back as a list or dict rather than a string. `str` renders
        # those without losing the per-leaf formatting.
        args_repr = ", ".join(
            str(tree.map_structure(format_argument_value, arg)) for arg in args
        )
        kwargs_repr = ", ".join(
            f"{key}={tree.map_structure(format_argument_value, value)!s}"
            for key, value in kwargs.items()
        )
    except Exception:
        return None
    return f"  • args=({args_repr})\n  • kwargs={{{kwargs_repr}}}"


def _build_augmented_exception(
    e, fn, arguments_context, tf_errors, object_name
):
    """Construct the augmented exception from a pre-formatted context string.

    Args:
        e: The original exception.
        fn: The function that was called, used as the fallback display name.
        arguments_context: Pre-formatted `  • name=value` listing (one entry
            per line) to append to the message.
        tf_errors: The `tensorflow.errors` module, or `None` if the backend
            is not TensorFlow.
        object_name: String, display name of the class/function being
            called. Defaults to `fn.__name__`.

    Returns:
        A new exception of the same type (or `RuntimeError` as a fallback)
        with argument info added to its message, or `None` if the injected
        flag could not be set on the new exception.
    """
    if tf_errors is not None and isinstance(e, tf_errors.OpError):
        message = e.message
    elif e.args:
        message = e.args[0]
    else:
        message = ""
    display_name = f"{object_name if object_name else fn.__name__}"
    message = (
        f"Exception encountered when calling {display_name}.\n\n"
        f"\x1b[1m{message}\x1b[0m\n\n"
        f"Arguments received by {display_name}:\n"
        f"{arguments_context}"
    )

    if tf_errors is not None and isinstance(e, tf_errors.OpError):
        new_e = e.__class__(e.node_def, e.op, message, e.error_code)
    else:
        try:
            new_e = e.__class__(message)
        except Exception:
            new_e = RuntimeError(message)
    try:
        new_e._keras_call_info_injected = True
    except Exception:
        return None
    return new_e


def inject_argument_info_in_error(e, fn, args, kwargs, object_name=None):
    """Add call argument info to an already-caught exception.

    Meant to be called from an `except` block, after `fn` has already raised,
    so that argument formatting only happens on the error path and adds no
    overhead to the common, no-exception case.

    Args:
        e: The exception that was raised by calling `fn(*args, **kwargs)`.
        fn: The function that was called, used to recover parameter names
            via `inspect.signature` and as the fallback display name.
        args: Positional arguments that `fn` was called with.
        kwargs: Keyword arguments that `fn` was called with.
        object_name: String, display name of the class/function being
            called, e.g. `'layer "layer_name" (LayerClass)'`. Defaults to
            `fn.__name__`.

    Returns:
        An exception with argument info added to its message, or the
        original exception unmodified if no argument info could be recovered.
        The returned exception has its traceback set to `e.__traceback__`.
    """
    if backend.backend() == "tensorflow":
        from tensorflow import errors as tf_errors
    else:
        tf_errors = None

    try:
        signature = inspect.signature(fn)
        bound_signature = signature.bind(*args, **kwargs)
    except (ValueError, TypeError):
        arguments_context = _format_raw_arguments_fallback(args, kwargs)
        if arguments_context is None:
            return e
    else:
        arguments_context = []
        for arg in signature.parameters.values():
            if arg.name in bound_signature.arguments:
                try:
                    value = tree.map_structure(
                        format_argument_value,
                        bound_signature.arguments[arg.name],
                    )
                except Exception:
                    # Formatting this specific bound value failed. Retrying
                    # via the raw args/kwargs fallback would call the same
                    # `format_argument_value` on the same value and fail the
                    # same way, so there is nothing to recover here.
                    return e
            else:
                value = arg.default
            arguments_context.append(f"  • {arg.name}={value}")

        if not arguments_context:
            return e
        arguments_context = "\n".join(arguments_context)

    augmented = _build_augmented_exception(
        e, fn, arguments_context, tf_errors, object_name
    )
    if augmented is None:
        return e
    return augmented.with_traceback(e.__traceback__)


def format_argument_value(value):
    if backend.ops.is_tensor(value):
        # Simplified representation for eager / graph tensors
        # to keep messages readable
        if backend.backend() == "tensorflow":
            tensor_cls = "tf.Tensor"
        elif backend.backend() == "jax":
            tensor_cls = "jnp.ndarray"
        elif backend.backend() == "torch":
            tensor_cls = "torch.Tensor"
        elif backend.backend() == "numpy":
            tensor_cls = "np.ndarray"
        else:
            tensor_cls = "array"

        return (
            f"{tensor_cls}(shape={value.shape}, "
            f"dtype={standardize_dtype(value.dtype)})"
        )
    return repr(value)
