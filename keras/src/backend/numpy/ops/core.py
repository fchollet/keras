import builtins
import contextlib
import functools
import warnings

import numpy as np

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import empty_scan_ys_spec
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = value

    def _direct_assign(self, value):
        self._value = np.array(value, dtype=self._dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __array__(self):
        return self.value


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with numpy backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with numpy backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value

    # The branch must run only for an explicit bfloat16 request:
    if not is_tensor(x) and dtype is not None and dtype == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return np.asarray(x).astype(dtype)
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    return np.array(x, dtype=dtype)


def convert_to_numpy(x):
    return np.array(x)


def is_tensor(x):
    if isinstance(x, (np.generic, np.ndarray)):
        return True
    return False


def shape(x):
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    if not isinstance(elements, (list, tuple)):
        return np.stack([function(x) for x in elements])
    else:
        if not elements:
            raise ValueError("`elements` cannot be empty.")
        batch_size = elements[0].shape[0]
        for i, x in enumerate(elements):
            if x.shape[0] != batch_size:
                raise ValueError(
                    "All elements passed to `vectorized_map` must have the "
                    f"same batch size (dimension 0). Element at index 0 has "
                    f"batch size {batch_size}, but element at index {i} has "
                    f"batch size {x.shape[0]}."
                )
        output_store = [function(list(x)) for x in zip(*elements)]
        return np.stack(output_store)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )

        def convert_keras_tensor_to_numpy(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return np.empty(
                    shape=shape,
                    dtype=x.dtype,
                )
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_numpy(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_numpy(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

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

        def convert_numpy_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = tree.map_structure(convert_numpy_to_keras_tensor, outputs)
    return output_spec


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
    dummy_y = [np.zeros_like(init) for init in init_flat]

    if n == 0:
        # Nothing to stack, so trace `f` once for the shapes of `ys`.
        packed_xs = pack_input(xs_flat) if len(xs_flat) > 0 else None
        ys_spec = empty_scan_ys_spec(compute_output_spec, f, init, packed_xs)
        if ys_spec is None:
            ys_spec = dummy_y
        stacked_y = tree.map_structure(
            lambda y: np.zeros((0, *y.shape), dtype=y.dtype),
            ys_spec,
        )
        return init, stacked_y

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: np.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [np.flip(elem, (axis,)) for elem in elems_flat]

    def _combine(a_flat, b_flat):
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

        a_dil = np.zeros(a_shape)
        np.copyto(slice_along_axis(a_dil, 0, None, 2, axis), a)
        b_dil = np.zeros(b_shape)
        np.copyto(slice_along_axis(b_dil, 0, None, 2, axis), b)

        a_pad = [[0, 0] for _ in range(a.ndim)]
        a_pad[axis][-1] = 1 if a.shape[axis] == b.shape[axis] else 0

        b_pad = [[0, 0] for _ in range(b.ndim)]
        b_pad[axis] = [1, 0] if a.shape[axis] == b.shape[axis] else [1, 1]

        op = np.bitwise_or if a.dtype == np.bool_ else np.add
        return op(
            np.pad(a_dil, a_pad),
            np.pad(b_dil, b_pad),
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
            np.concatenate(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                axis=axis,
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
        scans = [np.flip(scanned, (axis,)) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = np.zeros(shape, dtype=values.dtype)

    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    indices = np.reshape(indices, [-1, index_length])
    values = np.reshape(values, [-1] + list(value_shape))

    idx = tuple(indices.T)
    np.add.at(zeros, idx, values)
    return zeros


def scatter_update(inputs, indices, updates, reduction=None):
    indices = np.array(indices)
    indices = np.transpose(indices)
    idx = tuple(indices)
    if reduction is None:
        inputs[idx] = updates
    elif reduction == "add":
        np.add.at(inputs, idx, updates)
    elif reduction == "max":
        np.maximum.at(inputs, idx, updates)
    elif reduction == "min":
        np.minimum.at(inputs, idx, updates)
    elif reduction == "mul":
        np.multiply.at(inputs, idx, updates)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return inputs


def slice(inputs, start_indices, shape):
    # Validate inputs
    if len(start_indices) != len(shape):
        raise ValueError(
            "Length of `start_indices` must match length of `shape`. "
            f"Received: start_indices={start_indices}, shape={shape}"
        )

    # Generate list of indices arrays for each dimension
    indices = [
        np.arange(start, start + length)
        for start, length in zip(start_indices, shape)
    ]

    # Use np.ix_ to create a multidimensional index array
    mesh = np.ix_(*indices)

    return inputs[mesh]


def slice_update(inputs, start_indices, updates):
    # Generate list of indices arrays for each dimension
    indices = [
        np.arange(start, start + length)
        for start, length in zip(start_indices, updates.shape)
    ]

    # Use np.ix_ to create a multidimensional index array
    mesh = np.ix_(*indices)
    inputs[mesh] = updates
    return inputs


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = np.clip(index, 0, len(branches) - 1)
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
    return variable


def unstack(x, num=None, axis=0):
    return list(np.moveaxis(x, axis, 0))


def random_seed_dtype():
    return "uint32"


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        fun: Forward pass function.
    """

    def __init__(self, fun):
        warnings.warn(
            "`custom_gradient` for the numpy backend acts as a pass-through to "
            "support the forward pass. No gradient computation or modification "
            "takes place."
        )
        self.fun = fun

    def __call__(self, *args, **kwargs):
        outputs, _ = self.fun(*args, **kwargs)
        return outputs


@contextlib.contextmanager
def device_scope(device_name):
    yield


def remat(f):
    warnings.warn(
        "Rematerialization memory optimization is not supported by the "
        "Numpy backend. Please switch to JAX, TensorFlow, or PyTorch to "
        "utilize this feature."
    )
    return f


def grad(f, argnums=0):
    raise NotImplementedError("`grad` is not supported with the numpy backend.")
