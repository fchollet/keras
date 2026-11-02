import builtins
import contextlib
import functools
import os

import ml_dtypes
import numpy as np
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.backend_utils import standardize_argnums
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.config import floatx


# The promotion mode lives only on the torch dispatch stack, which is per
# thread like the Keras distribution. Keras keeps no reference to the mode, so
# no reset of Keras state can orphan a mode that is still on the stack.
def _get_dtensor_promotion_mode():
    stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
    for mode in stack:
        if isinstance(mode, KerasDTensorPromotionMode):
            return mode
    return None


def activate_dtensor_promotion():
    if _get_dtensor_promotion_mode() is None:
        KerasDTensorPromotionMode().__enter__()


def deactivate_dtensor_promotion():
    mode = _get_dtensor_promotion_mode()
    # `__exit__` pops the top of the stack, so only exit the mode when it is
    # on top. Otherwise it would pop a mode that was entered after it.
    if (
        mode is not None
        and torch.utils._python_dispatch._get_current_dispatch_mode() is mode
    ):
        mode.__exit__(None, None, None)


# Some operators such as 'aten::_foreach_mul_.Scalar'
# are not currently implemented for the MPS device.
# check https://github.com/pytorch/pytorch/issues/77764.
if "KERAS_TORCH_DEVICE" in os.environ:
    DEFAULT_DEVICE = os.environ["KERAS_TORCH_DEVICE"]
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEFAULT_DEVICE = "xpu"
else:
    from keras.src.utils.module_utils import torch_xla

    if torch_xla.available and torch_xla.core.xla_model.xla_device_count() > 0:
        DEFAULT_DEVICE = "tpu"
    else:
        DEFAULT_DEVICE = "cpu"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "uint16": torch.int32,  # TODO: Torch doesn't have `uint16` dtype.
    "uint32": torch.int64,  # TODO: Torch doesn't have `uint32` dtype.
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# TODO: Remove the MPS dtype downcasts if PyTorch adds support for these
# dtypes on MPS (Metal currently has no double-precision support, see
# https://github.com/pytorch/pytorch/issues/79019).
MPS_UNSUPPORTED_DTYPES = {
    "float64": "float32",
    "complex128": "complex64",
}


@contextlib.contextmanager
def device_scope(device_name):
    previous_device = global_state.get_global_attribute("torch_device", None)
    current_device = _parse_device_input(device_name)
    global_state.set_global_attribute("torch_device", current_device)
    try:
        yield torch.device(current_device)
    finally:
        global_state.set_global_attribute("torch_device", previous_device)


def get_device():
    device = global_state.get_global_attribute("torch_device", None)
    if device is None:
        return DEFAULT_DEVICE
    return device


def _parse_device_input(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", and need to convert
        # "gpu" to "cuda"
        device_name = device_name.lower()
        if "gpu" in device_name:
            device_name = device_name.replace("gpu", "cuda")
    else:
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or 'cpu'. "
            f"Received: device_name='{device_name}'"
        )
    # The torch.Device instance can be used directly.
    return device_name


def to_torch_dtype(dtype):
    standardized_dtype = TORCH_DTYPES.get(standardize_dtype(dtype), None)
    if standardized_dtype is None:
        raise ValueError(f"Unsupported dtype for PyTorch: {dtype}")
    return standardized_dtype


def _promote_tensor_args(args, kwargs, dtensor_arg):
    """Promotes plain tensors to DTensors using the mesh of dtensor_arg."""
    mesh = dtensor_arg.device_mesh
    placements = [Replicate()] * len(mesh.shape)

    def maybe_promote(value):
        if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
            if value.device.type != mesh.device_type:
                value = value.to(mesh.device_type)
            return DTensor.from_local(
                value, device_mesh=mesh, placements=placements
            )
        return value

    return tree.map_structure(maybe_promote, (args, kwargs))


class KerasDTensorPromotionMode(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that promotes mixed Tensor/DTensor arguments.

    This mode prevents "mixed tensor" errors by intercepting operations where
    plain tensors and DTensors collide, automatically promoting the plain
    tensor to a Replicated DTensor.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        flat_args = tree.flatten((args, kwargs))
        dtensor_arg = None
        has_plain_tensor = False

        for arg in flat_args:
            if isinstance(arg, DTensor):
                if dtensor_arg is None:
                    dtensor_arg = arg
            elif isinstance(arg, torch.Tensor):
                has_plain_tensor = True

            if dtensor_arg is not None and has_plain_tensor:
                break

        if dtensor_arg is not None and has_plain_tensor:
            args, kwargs = _promote_tensor_args(args, kwargs, dtensor_arg)

        return func(*args, **kwargs)


class Variable(KerasVariable):
    def _initialize_distributed(self, initializer):
        from keras.src.backend.torch import distribution_lib
        from keras.src.distribution.distribution_lib import TensorLayout
        from keras.src.distribution.distribution_lib import distribution

        if self._layout is None and distribution() is not None:
            self._layout = distribution().get_variable_layout(self)

        if self._layout is None:
            return None

        if isinstance(self._layout, TensorLayout):
            if all(axis is None for axis in self._layout.axes):
                return None

        if callable(initializer):
            value = initializer(self._shape, dtype=self._dtype)
        else:
            value = initializer

        if isinstance(value, torch.nn.Parameter):
            value = value.data
        if isinstance(value, torch.Tensor) and (
            value.requires_grad or value.grad_fn is not None
        ):
            value = value.detach()

        if isinstance(value, torch.Tensor):
            if isinstance(self._layout, TensorLayout):
                backend_mesh = self._layout.device_mesh.backend_mesh
            else:
                backend_mesh = self._layout.device_mesh

            if value.device.type != backend_mesh.device_type:
                value = value.to(backend_mesh.device_type)

        dtensor = distribution_lib.distribute_tensor(value, self._layout)

        return torch.nn.Parameter(dtensor, requires_grad=self.trainable)

    def _initialize_with_initializer(self, initializer):
        value = self._initialize_distributed(initializer)
        if value is not None:
            self._value = value
        else:
            super()._initialize_with_initializer(initializer)

    def _initialize(self, value):
        self._shape = self._validate_shape(value.shape)
        new_value = self._initialize_distributed(value)
        if new_value is not None:
            self._value = new_value
            return

        if isinstance(value, torch.nn.Parameter):
            # Reuse same parameter
            self._value = value
        else:
            self._value = torch.nn.Parameter(
                convert_to_tensor(value, dtype=self._dtype),
                requires_grad=self.trainable,
            ).to(get_device())

    def _direct_assign(self, value):
        # Use `self._value`, not `self.value`: inside an `AutocastScope`,
        # `self.value` is a cast copy and writes to it would be lost.
        if isinstance(self._value.data, DTensor):
            from keras.src.backend.torch import distribution_lib
            from keras.src.distribution.distribution_lib import TensorLayout

            layout = self._layout
            if isinstance(layout, TensorLayout):
                layout = layout.backend_layout
            if value.device.type != layout.device_mesh.device_type:
                value = value.to(layout.device_mesh.device_type)
            # Assignment is not differentiable, and torch only distributes
            # leaf tensors.
            value = distribution_lib.distribute_tensor(value.detach(), layout)

        with torch.no_grad():
            self._value.copy_(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        args = [arg.value if isinstance(arg, Variable) else arg for arg in args]
        if kwargs is None:
            kwargs = {}
        kwargs = {
            key: value.value if isinstance(value, Variable) else value
            for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value

    def __repr__(self):
        if isinstance(getattr(self, "_value", None), DTensor):
            # Gathering a sharded value is a collective, and a repr can run
            # on a single process (a log line, a debugger, an error message).
            return (
                f"<Variable path={self.path}, shape={self.shape}, "
                f"dtype={self.dtype}>"
            )
        return super().__repr__()

    @property
    def value(self):
        # We cannot chain super() here because it will fail TorchDynamo. The
        # reason why is unclear.
        def maybe_use_symbolic_tensor(value):
            # Create and use a symbolic tensor stub in symbolic calls.
            if str(get_device()) == "meta" and str(value.device) != "meta":
                return torch.nn.Parameter(
                    torch.empty(
                        size=self._shape,
                        dtype=to_torch_dtype(self._dtype),
                        device="meta",
                    ),
                    requires_grad=self.trainable,
                )
            return value

        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                value = self._maybe_autocast(value)
                return maybe_use_symbolic_tensor(value)
        if self._value is None:
            # Uninitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference / graph tracing
            # (anything else would be a bug, to be fixed.)
            value = self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        else:
            value = self._maybe_autocast(self._value)
        return maybe_use_symbolic_tensor(value)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._value is not None:
            self._value.requires_grad = value

    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except Exception:
            return False


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with torch backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with torch backend")
    inferred_dtype = dtype is None
    if isinstance(x, Variable) or is_tensor(x):
        if isinstance(x, Variable):
            x = x.value
        device = get_device()
        if x.is_meta:
            if dtype is None and "mps" in str(device):
                dtype = MPS_UNSUPPORTED_DTYPES.get(standardize_dtype(x.dtype))
            x = torch.empty_like(
                x,
                dtype=None if dtype is None else to_torch_dtype(dtype),
                device=device,
            )
        elif dtype is None and "mps" in str(device):
            x_dtype = standardize_dtype(x.dtype)
            fallback_dtype = MPS_UNSUPPORTED_DTYPES.get(x_dtype)
            if fallback_dtype is not None:
                x = x.to(device=device, dtype=to_torch_dtype(fallback_dtype))
            else:
                x = x.to(device)
        elif dtype is None:
            x = x.to(device)
        else:
            x = x.to(device=device, dtype=to_torch_dtype(dtype))
        return x
    if isinstance(x, (bool, int, float, complex)):
        if dtype is not None:
            dt = to_torch_dtype(dtype)
        elif isinstance(x, bool):
            dt = torch.bool
        elif isinstance(x, int):
            dt = torch.int64 if x < -(2**31) or x >= 2**31 else torch.int32
        elif isinstance(x, float):
            dt = to_torch_dtype(floatx())
        else:
            dt = torch.complex64
        return torch.as_tensor(x, dtype=dt, device=get_device())
    if isinstance(x, (torch.SymInt, torch.SymFloat)):
        # Scalar symbolic values from torch.export can't go through numpy.
        dt = to_torch_dtype(dtype) if dtype is not None else None
        return torch.as_tensor(x, dtype=dt, device=get_device())

    # Convert to np in case of any array-like that is not list or tuple.
    # Skip scalar Python values to avoid np.array(float) -> float64, which
    # causes dtype issues during torch.export (constants are lifted before
    # the cast to the requested dtype).
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and any(isinstance(x1, torch.Tensor) for x1 in x):
            # Handle list or tuple of torch tensors
            return torch.stack([convert_to_tensor(x1) for x1 in x])
        if len(x) > 0 and any(
            isinstance(x1, (torch.SymInt, torch.SymFloat))
            for x1 in tree.flatten(x)
        ):
            # Symbolic shape values from torch.export can't go through numpy
            # and don't have a .dtype attribute. Use torch.as_tensor directly.
            dt = to_torch_dtype(dtype) if dtype is not None else None
            return torch.as_tensor(x, dtype=dt, device=get_device())
    elif not isinstance(x, (bool, int, float)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint32:
            # Torch backend does not support uint32.
            x = x.astype(np.int64)
        if standardize_dtype(x.dtype) == "bfloat16":
            # Torch backend does not support converting bfloat16 ndarray.
            x = x.astype(np.float32)
            dtype = "bfloat16"
        dtype = dtype or x.dtype
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    if inferred_dtype and "mps" in str(get_device()):
        dtype = MPS_UNSUPPORTED_DTYPES.get(standardize_dtype(dtype), dtype)
    dtype = to_torch_dtype(dtype)
    return torch.as_tensor(x, dtype=dtype, device=get_device())


def convert_to_numpy(x):
    def transform(x):
        if isinstance(x, Variable):
            x = x.value
        if is_tensor(x):
            if x.requires_grad:
                x = x.detach()
            if isinstance(x, DTensor):
                # Gather the global value. This is a collective when `x` is
                # sharded across processes, so every process must convert it.
                x = x.full_tensor()
            # Tensor has to be moved to CPU before converting to numpy.
            if x.device != torch.device("cpu"):
                x = x.cpu()
            if x.dtype == torch.bfloat16:
                # Attempting to call .numpy() on a bfloat16 torch tensor leads
                # to an immediate error. Instead we upcast to float32 and then
                # convert to the numpy friendly bfloat16 type.
                # https://github.com/pytorch/pytorch/issues/90574
                return np.array(x.to(torch.float32)).astype(ml_dtypes.bfloat16)
        return np.array(x)

    if isinstance(x, (list, tuple)):
        return np.array([transform(e) for e in x])
    return transform(x)


def is_tensor(x):
    # Using the built-in `isinstance` is recommended by pytorch
    # over using torch.is_tensor
    # see: https://pytorch.org/docs/stable/generated/torch.is_tensor.html
    #
    # Also, `torch.is_tensor()` causes issues with dynamo caching when
    # a torch.Tensor and numpy.ndarray of the same size, shape, and dtype
    # is passed, if called on a Tensor first the second call with ndarray
    # will return `True` and vice-versa.
    return isinstance(x, torch.Tensor)


def shape(x):
    # Convert from `torch.Size` to plain tuple.
    return tuple(x.shape)


def cast(x, dtype):
    dtype = to_torch_dtype(dtype)
    if isinstance(x, Variable):
        x = x.value
    if is_tensor(x):
        if x.dtype == dtype:
            return x
        else:
            return x.to(dtype)
    return convert_to_tensor(x, dtype)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    def has_none_shape(x):
        """Check for if a `KerasTensor` has dynamic shape."""
        if isinstance(x, KerasTensor):
            return None in x.shape
        return False

    def convert_keras_tensor_to_torch(x, fill_value=None):
        """Convert `KerasTensor`s to `torch.Tensor`s."""
        if isinstance(x, KerasTensor):
            shape = list(x.shape)
            if fill_value:
                for i, e in enumerate(shape):
                    if e is None:
                        shape[i] = fill_value
            return torch.ones(
                size=shape,
                dtype=TORCH_DTYPES[x.dtype],
                device=get_device(),
            )
        return x

    def convert_torch_to_keras_tensor(x):
        """Convert `torch.Tensor`s to `KerasTensor`s."""
        if is_tensor(x):
            return KerasTensor(x.shape, standardize_dtype(x.dtype))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        """Call `fn` to infer output shape and dtype."""
        try:
            # First try instantiating all tensors on the `"meta"` device,
            # which  should give a "zero flop" way to trace shape, but does
            # not have universal support with torch operations.
            with device_scope("meta"):
                meta_args, meta_kwargs = tree.map_structure(
                    lambda x: convert_keras_tensor_to_torch(x, fill_value),
                    (args, kwargs),
                )
                return fn(*meta_args, **meta_kwargs)
        except:
            with device_scope(DEFAULT_DEVICE):
                # If the `"meta"` device placement fails, fall back to tracing
                # eagerly with tensors on the default device. This will be
                # more robust, but more expensive.
                eager_args, eager_kwargs = tree.map_structure(
                    lambda x: convert_keras_tensor_to_torch(x, fill_value),
                    (args, kwargs),
                )
                return fn(*eager_args, **eager_kwargs)

    with StatelessScope(), SymbolicScope(), torch.no_grad():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )
        if none_in_shape:
            outputs_1 = outputs
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(KerasTensor(shape, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        output_spec = tree.map_structure(convert_torch_to_keras_tensor, outputs)
    return output_spec


def cond(pred, true_fn, false_fn):
    # When symbolic execution, take pred as true.
    if get_device() == "meta":
        return true_fn()

    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    return torch.vmap(function)(elements)


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # Ref: jax.lax.scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [torch.zeros_like(init) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: torch.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [torch.flip(elem, (axis,)) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a_flat = [convert_to_tensor(a) for a in a_flat]
        b_flat = [convert_to_tensor(b) for b in b_flat]

        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )

    def _interleave(a, b, axis):
        """Given two Tensors of static shape, interleave them along axis."""
        if not (
            a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
        ):
            raise ValueError(
                "Shapes are incompatible for associative_scan interleaving. "
                f"a.shape[{axis}]={a.shape[axis]}, "
                f"b.shape[{axis}]={b.shape[axis]}"
            )

        # we want to get a: [a1, a2], b: [b1, b2]
        # to a: [a1, 0, a2, 0], b: [0, b1, 0, b2]
        a_shape = list(a.shape)
        a_shape[axis] = a.shape[axis] * 2 - 1

        b_shape = list(b.shape)
        b_shape[axis] = b.shape[axis] * 2 - 1

        a_dil = torch.zeros(a_shape)
        slice_along_axis(a_dil, 0, None, 2, axis).copy_(a)

        b_dil = torch.zeros(b_shape)
        slice_along_axis(b_dil, 0, None, 2, axis).copy_(b)

        a_pad = [[0, 0] for _ in range(a.dim())]
        a_pad[axis][-1] = 1 if a.shape[axis] == b.shape[axis] else 0
        a_pad = a_pad[::-1]
        a_pad = tree.flatten(a_pad)

        b_pad = [[0, 0] for _ in range(b.dim())]
        b_pad[axis] = [1, 0] if a.shape[axis] == b.shape[axis] else [1, 1]
        b_pad = b_pad[::-1]
        b_pad = tree.flatten(b_pad)

        op = torch.bitwise_or if a.dtype == torch.bool else torch.add
        return op(
            torch.nn.functional.pad(a_dil, a_pad),
            torch.nn.functional.pad(b_dil, b_pad),
        )

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [
                slice_along_axis(elem, 0, -1, step=2, axis=axis)
                for elem in elems
            ],
            [
                slice_along_axis(elem, 1, None, step=2, axis=axis)
                for elem in elems
            ],
        )

        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = _combine(
                [slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )

        even_elems = [
            torch.cat(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                dim=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            builtins.map(
                functools.partial(_interleave, axis=axis), even_elems, odd_elems
            )
        )

    scans = _scan(elems_flat)
    if reverse:
        scans = [torch.flip(scanned, (axis,)) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices, dtype="int64")
    values = convert_to_tensor(values)
    zeros = torch.zeros(shape, dtype=values.dtype, device=get_device())

    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    indices = torch.reshape(indices, [-1, index_length])
    values = torch.reshape(values, [-1] + list(value_shape))

    # Vectorized scatter-add: index_put_ with accumulate=True applies all
    # updates in one call and sums duplicate indices, instead of a Python loop
    # over every index (which is very slow for metrics like MeanIoU that
    # scatter one update per pixel).
    idx = tuple(indices.transpose(0, 1))
    zeros.index_put_(idx, values, accumulate=True)
    return zeros


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices, dtype="int64")
    updates = convert_to_tensor(updates, dtype=inputs.dtype)
    indices = torch.transpose(indices, 0, 1)
    idx = tuple(indices)

    outputs = torch.clone(inputs)
    if reduction is None:
        outputs[idx] = updates
    elif reduction == "add":
        # Use index_put_ with accumulate=True for proper accumulation
        outputs.index_put_(idx, updates, accumulate=True)
    elif reduction == "max":
        # Loop-based approach handles both scalar and slice updates.
        # Associative, so sequential application handles duplicates.
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = torch.maximum(outputs[idx], updates[i])
    elif reduction == "min":
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = torch.minimum(outputs[idx], updates[i])
    elif reduction == "mul":
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = outputs[idx] * updates[i]
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return outputs


def _to_static_index(v):
    """Returns `v` as a static slice bound, or raises `TypeError`.

    `int` and `torch.SymInt` pass through unchanged: calling `int()` on a
    `SymInt` would specialize a `torch.export` dynamic dimension to a
    constant. Anything else raises, so `slice()` falls through to
    `torch.narrow` and the bound stays traceable.
    """
    if isinstance(v, (int, torch.SymInt)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    raise TypeError(
        f"slice() index element has type {type(v).__name__!r}; "
        "using torch.narrow path"
    )


def slice(inputs, start_indices, shape):
    inputs = convert_to_tensor(inputs)

    # Fast path: plain Python slices, avoiding the tensors the slow path builds
    # from the indices, which torch.export cannot trace. An unusable bound
    # raises `TypeError`, a data-dependent symbolic one `RuntimeError`; both
    # fall through. The indexing sits in the `else` so that its own
    # `IndexError` is not mistaken for an unusable bound and retried below.
    if isinstance(start_indices, (list, tuple)) and isinstance(
        shape, (list, tuple)
    ):
        try:
            slices = []
            for start_index, length in zip(start_indices, shape):
                start = _to_static_index(start_index)
                size = _to_static_index(length)
                slices.append(builtins.slice(start, start + size))
        except (TypeError, RuntimeError):
            pass
        else:
            return inputs[tuple(slices)]

    # Slow path: tensor-based slicing via torch.narrow for truly dynamic
    # indices. torch.narrow is a proper ATen op that torch.export can
    # trace, unlike Python-level iteration over tensor elements.
    shape_dtype = to_torch_dtype("int64")
    start_indices = convert_to_tensor(start_indices).to(shape_dtype)
    shape = convert_to_tensor(shape).to(shape_dtype)
    result = inputs
    for dim in range(start_indices.shape[0]):
        result = torch.narrow(result, dim, start_indices[dim], shape[dim])
    return result


def slice_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)

    # Fast path: when start_indices is a Python int sequence, use it
    # directly. This avoids creating tensors from the indices, which would
    # introduce data-dependent expressions that torch.export cannot trace.
    if isinstance(start_indices, (list, tuple)) and all(
        isinstance(s, int) for s in start_indices
    ):
        slices = [
            builtins.slice(start_index, start_index + update_length)
            for start_index, update_length in zip(start_indices, updates.shape)
        ]
        outputs = torch.clone(inputs)
        outputs[tuple(slices)] = updates
        return outputs

    # Slow path: tensor-based start_indices.
    # We cannot use Python slice objects because they require scalar int
    # bounds, and extracting scalars from traced tensors creates unbacked
    # symbols that torch.export cannot resolve.
    # We also cannot use `torch.narrow` (as the `slice` slow-path does)
    # because `narrow` is read-only; for in-place updates we need
    # `index_put_`, which is fully traceable.
    start_indices = convert_to_tensor(start_indices, dtype="int64")
    outputs = inputs.clone()
    update_shape = list(updates.shape)
    dims = len(update_shape)
    indices_list = []
    for dim in range(dims):
        dim_indices = torch.arange(
            update_shape[dim],
            dtype=start_indices.dtype,
            device=start_indices.device,
        )
        dim_indices = dim_indices + start_indices[dim]
        indices_list.append(dim_indices)

    grids = torch.meshgrid(*indices_list, indexing="ij")
    flat_indices = [g.flatten() for g in grids]
    outputs.index_put_(tuple(flat_indices), updates.flatten())
    return outputs


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = torch.clamp(index, 0, len(branches) - 1)
    return branches[index](*operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    current_iter = 0
    iteration_check = lambda iter: (
        maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
    while cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars if is_tuple else loop_vars[0]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    # We can't use `.requires_grad_(False)` here since it only
    # works when the tensor is a leaf node in the graph.
    return variable.detach()


def unstack(x, num=None, axis=0):
    return x.unbind(axis)


def random_seed_dtype():
    # uint32 doesn't exist in torch. Seeds are conceptually uint32 values;
    # int32 is used and the bit pattern is reinterpreted as uint32 at each
    # call site (torch_seed_generator / torch.manual_seed) via & 0xFFFFFFFF.
    return "int32"


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that defines a custom gradient, which
        recomputes f on the backwards pass of a gradient call.
    """

    def wrapped(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            f, *args, use_reentrant=False, **kwargs
        )

    return wrapped


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        forward_fn: Forward pass function.
    """

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def __call__(self, *args, **kwargs):
        return CustomGradientFunction.apply(self.forward_fn, *args, **kwargs)


class CustomGradientFunction(torch.autograd.Function):
    """Enables custom forward & backward passes for gradient computation."""

    @staticmethod
    def forward(ctx, forward_fn, *args, **kwargs):
        """Forward pass computation specification.

        Args:
            ctx: Context object.
            forward_fn: Function to compute forward pass.
            *args: Arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
        """
        ctx.forward_fn = forward_fn
        ctx.save_for_backward(*args)
        result = forward_fn(*args, **kwargs)
        if isinstance(result, (list, tuple)) and len(result) == 2:
            output, ctx.grad_fn = result
        else:
            output, ctx.grad_fn = result, None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation specification.

        Args:
            ctx: Context object.
            grad_output: Gradient with respect to the output.
        """
        args = ctx.saved_tensors
        grad_fn = ctx.grad_fn
        if grad_fn is None:
            raise ValueError("grad_fn must be provided for custom gradient")
        grads = grad_fn(*args, upstream=grad_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        # Keras ops compute the gradients on `get_device()`, but autograd
        # needs each gradient on the device of its input.
        grads = list(grads)
        for i, (grad, x) in enumerate(zip(grads, args)):
            if isinstance(grad, torch.Tensor) and x is not None:
                grads[i] = grad.to(x.device)
        return (None,) + tuple(grads)


def grad(f, argnums=0):
    def grad_fn(*args, **kwargs):
        positions = standardize_argnums(argnums, len(args))
        args = list(args)

        def track(x):
            return convert_to_tensor(x).detach().requires_grad_(True)

        for i in positions:
            args[i] = tree.map_structure(track, args[i])
        inputs = [args[i] for i in positions]
        leaves = tree.flatten(inputs)
        with torch.enable_grad():
            # A gradient tape sums a non scalar output, so do the same here.
            output = torch.sum(f(*args, **kwargs))
        leaf_grads = torch.autograd.grad(output, leaves, allow_unused=True)
        leaf_grads = [
            torch.zeros_like(x) if g is None else g
            for g, x in zip(leaf_grads, leaves)
        ]
        grads = tree.pack_sequence_as(inputs, leaf_grads)
        if isinstance(argnums, int):
            return grads[0]
        return tuple(grads)

    return grad_fn
