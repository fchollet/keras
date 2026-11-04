import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.ops.operation import Operation
from keras.src.saving import serialization_lib
from keras.src.utils import traceback_utils


class Map(Operation):
    def call(self, f, xs):
        return backend.ops.core.map(f, xs)

    def compute_output_spec(self, f, xs):
        x = tree.map_structure(lambda t: t[0], xs)
        n = tree.flatten(xs)[0].shape[0]
        y = backend.compute_output_spec(f, x)

        def append_batch_axis(t):
            return KerasTensor(
                shape=(n,) + t.shape,
                dtype=t.dtype,
                sparse=t.sparse,
                ragged=t.ragged,
            )

        y = tree.map_structure(append_batch_axis, y)
        return y


@keras_export("keras.ops.map")
def map(f, xs):
    """Map a function over leading array axes.

    Like Python’s builtin map, except inputs and outputs are in the form of
    stacked arrays. Consider using the `vectorized_map()` transform instead,
    unless you need to apply a function element by element for reduced memory
    usage or heterogeneous computation with other control flow primitives.

    When `xs` is an array type, the semantics of `map()` are given by this
    Python implementation:

    ```python
    def map(f, xs):
        return np.stack([f(x) for x in xs])
    ```

    Args:
        f: Callable defines the function to apply element-wise over the first
            axis or axes of `xs`.
        xs: Values over which to map along the leading axis.

    Returns:
        Mapped values.

    Examples:

    >>> f = lambda x: x**2
    >>> xs = keras.ops.arange(10)
    >>> ys = keras.ops.map(f, xs)
    >>> ys
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    >>> f = lambda x: {"y1": x**2, "y2": x * 10}  # Can have nested outputs
    >>> ys = keras.ops.map(f, xs)
    >>> ys["y1"]
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    >>> ys["y2"]
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    """
    if any_symbolic_tensors((xs,)):
        return Map().symbolic_call(f, xs)
    return backend.ops.core.map(f, xs)


class Scan(Operation):
    def __init__(self, length=None, reverse=False, unroll=1, *, name=None):
        super().__init__(name=name)
        self.length = length
        self.reverse = reverse
        self.unroll = unroll

    def call(self, f, init, xs=None):
        return backend.ops.core.scan(
            f,
            init,
            xs,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )

    def compute_output_spec(self, f, init, xs=None):
        if xs is None:
            n = int(self.length)
            x = None
        else:
            n = (
                int(self.length)
                if self.length is not None
                else tree.flatten(xs)[0].shape[0]
            )
            x = xs[0]

        carry, y = backend.compute_output_spec(f, init, x)
        y = KerasTensor(shape=(n,) + y.shape, dtype=y.dtype, sparse=y.sparse)
        return carry, y


@keras_export("keras.ops.scan")
def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    """Scan a function over leading array axes while carrying along state.

    When the type of `xs` is an array type or `None`, and the type of `ys` is an
    array type, the semantics of `scan()` are given roughly by this Python
    implementation:

    ```python
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)
    ```

    The loop-carried value `carry` (`init`) must hold a fixed shape and dtype
    across all iterations.

    In TensorFlow, `y` must match `carry` in shape and dtype. This is not
    required in other backends.

    Args:
        f: Callable defines the logic for each loop iteration. This accepts two
            arguments where the first is a value of the loop carry and the
            second is a slice of `xs` along its leading axis.
            This callable returns a pair where the first represents a new value
            for the loop carry and the second represents a slice of the output.
        init: The initial loop carry value. This can be a scalar, tensor, or any
            nested structure. It must match the structure of the first element
            returned by `f`.
        xs: Optional value to scan along its leading axis. This can be a tensor
            or any nested structure. If `xs` is not provided, you must specify
            `length` to define the number of loop iterations.
            Defaults to `None`.
        length: Optional integer specifying the number of loop iterations.
            If `length` is not provided, it defaults to the sizes of leading
            axis of the arrays in `xs`. Defaults to `None`.
        reverse: Optional boolean specifying whether to run the scan iteration
            forward or in reverse, equivalent to reversing the leading axes of
            the arrays in both `xs` and in `ys`.
        unroll: Optional positive integer or boolean specifying how many scan
            iterations to unroll within a single iteration of a loop. If an
            integer is provided, it determines how many unrolled loop iterations
            to run within a single rolled iteration of the loop. If a boolean is
            provided, it will determine if the loop is completely unrolled
            (`unroll=True`) or left completely unrolled (`unroll=False`).
            Note that unrolling is only supported by JAX and TensorFlow
            backends.

    Returns:
        A pair where the first element represents the final loop carry value and
        the second element represents the stacked outputs of `f` when scanned
        over the leading axis of the inputs.

    Examples:

    >>> sum_fn = lambda c, x: (c + x, c + x)
    >>> init = keras.ops.array(0)
    >>> xs = keras.ops.array([1, 2, 3, 4, 5])
    >>> carry, result = keras.ops.scan(sum_fn, init, xs)
    >>> carry
    15
    >>> result
    [1, 3, 6, 10, 15]
    """
    if any_symbolic_tensors((init, xs)):
        return Scan(
            length=length, reverse=reverse, unroll=unroll
        ).symbolic_call(f, init, xs)
    return backend.ops.core.scan(
        f, init, xs, length, reverse=reverse, unroll=unroll
    )


class AssociativeScan(Operation):
    def __init__(self, reverse=False, axis=0, *, name=None):
        super().__init__(name=name)
        self.reverse = reverse
        self.axis = axis

    def call(self, f, elems):
        return backend.ops.core.associative_scan(
            f, elems, reverse=self.reverse, axis=self.axis
        )

    def compute_output_spec(self, f, elems):
        elems_flat = tree.flatten(elems)
        lens = [elem.shape[self.axis] for elem in elems_flat]
        if len(set(lens)) != 1:
            raise ValueError(
                "Array inputs to associative_scan must have the same "
                "first dimension. (saw: {})".format(
                    [elem.shape for elem in elems_flat]
                )
            )

        x = tree.pack_sequence_as(
            elems,
            [slice_along_axis(x, 0, 1, axis=self.axis) for x in elems_flat],
        )
        y_spec = backend.compute_output_spec(f, x, x)

        def _restore_shape(x):
            return KerasTensor(
                shape=elems_flat[0].shape, dtype=x.dtype, sparse=x.sparse
            )

        y_spec = tree.map_structure(_restore_shape, y_spec)
        return y_spec


@keras_export("keras.ops.associative_scan")
def associative_scan(f, elems, reverse=False, axis=0):
    """Performs a scan with an associative binary operation, in parallel.

    This operation his similar to `scan`, with the key difference that
    `associative_scan` is a parallel implementation with
    potentially significant performance benefits, especially when jit compiled.
    The catch is that it can only be used when `f` is a binary associative
    operation (i.e. it must verify `f(a, f(b, c)) == f(f(a, b), c)`).

    For an introduction to associative scans, refer to this paper:
    Blelloch, Guy E. 1990.
    [Prefix Sums and Their Applications](
        https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

    Args:
        f: A Python callable implementing an associative binary operation with
            signature `r = f(a, b)`. Function `f` must be associative, i.e.,
            it must satisfy the equation
            `f(a, f(b, c)) == f(f(a, b), c)`.
            The inputs and result are (possibly nested Python tree structures
            of) array(s) matching `elems`. Each array has a dimension in place
            of the `axis` dimension. `f` should be applied elementwise over
            the `axis` dimension.
            The result `r` has the same shape (and structure) as the
            two inputs `a` and `b`.
        elems: A (possibly nested Python tree structure of) array(s), each with
            an `axis` dimension of size `num_elems`.
        reverse: A boolean stating if the scan should be reversed with respect
            to the `axis` dimension.
        axis: an integer identifying the axis over which the scan should occur.

    Returns:
        A (possibly nested Python tree structure of) array(s) of the same shape
        and structure as `elems`, in which the `k`'th element of `axis` is
        the result of recursively applying `f` to combine the first `k`
        elements of `elems` along `axis`. For example, given
        `elems = [a, b, c, ...]`, the result would be
        `[a, f(a, b), f(f(a, b), c), ...]`.

    Examples:

    >>> sum_fn = lambda x, y: x + y
    >>> xs = keras.ops.arange(5)
    >>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)
    >>> ys
    [0, 1, 3, 6, 10]

    >>> sum_fn = lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
    >>> xs = [keras.ops.array([1, 2]) for _ in range(3)]
    >>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)
    >>> ys
    [[1, 3], [1, 3], [1, 3]]
    """
    if any_symbolic_tensors((elems,)):
        return AssociativeScan(reverse=reverse, axis=axis).symbolic_call(
            f, elems
        )
    return backend.ops.core.associative_scan(
        f, elems, reverse=reverse, axis=axis
    )


class Scatter(Operation):
    def __init__(self, shape, *, name=None):
        super().__init__(name=name)
        self.shape = shape

    def call(self, indices, values):
        return backend.ops.core.scatter(indices, values, self.shape)

    def compute_output_spec(self, indices, values):
        return KerasTensor(self.shape, dtype=values.dtype)


@keras_export("keras.ops.scatter")
def scatter(indices, values, shape):
    """Returns a tensor of shape `shape` where `indices` are set to `values`.

    At a high level, this operation does `zeros[indices] = updates` and
    returns the output. It is equivalent to:

    ```python
    zeros = keras.ops.zeros(shape)
    output = keras.ops.scatter_update(zeros, indices, values)
    ```

    Args:
        indices: A tensor or list/tuple specifying
            indices for the values in `values`.
        values: A tensor, the values to be set at `indices`.
        shape: Shape of the output tensor.

    Example:

    >>> indices = [[0, 1], [1, 1]]
    >>> values = np.array([1., 1.])
    >>> keras.ops.scatter(indices, values, shape=(2, 2))
    array([[0., 1.],
           [0., 1.]])
    """
    if any_symbolic_tensors((indices, values)):
        return Scatter(shape=shape).symbolic_call(indices, values)
    return backend.ops.core.scatter(indices, values, shape)


class ScatterUpdate(Operation):
    def __init__(self, reduction=None, *, name=None):
        super().__init__(name=name)
        self.reduction = reduction

    def call(self, inputs, indices, updates):
        return backend.ops.core.scatter_update(
            inputs, indices, updates, reduction=self.reduction
        )

    def compute_output_spec(self, inputs, indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.ops.scatter_update")
def scatter_update(inputs, indices, updates, reduction=None):
    """Update inputs via updates at scattered (sparse) indices.

    At a high level, this operation does `inputs[indices] = updates`.
    Assume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main
    usages of `scatter_update`.

    1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
        is the number of updates to perform, and `updates` is a 1D tensor of
        shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,
        and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
        we can use:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2, 3], [0, 1, 3]]
    updates = np.array([1., 1.])
    inputs = keras.ops.scatter_update(inputs, indices, updates)
    ```

    2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
        is the number of updates to perform, and `k` (`k < n`) is the size of
        each index in `indices`. `updates` is a `n - k`-D tensor of shape
        `(num_updates, inputs.shape[k:])`. For example, if
        `inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`
        and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
        `(num_updates, 2)` (`k = 2`), and `updates` would have shape
        `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2], [2, 3]]
    updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
    inputs = keras.ops.scatter_update(inputs, indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        indices: A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying
            indices to update. `N` is the number of indices to update, must be
            equal to the first dimension of `updates`.
        updates: A tensor, the new values to be put to `inputs` at `indices`.
        reduction: A string specifying the reduction operation to apply when
            multiple updates target the same index. Supported values are:
            `None` (default): Updates replace existing values (last write wins).
            `"add"`: Updates are added to existing values.
            `"max"`: The maximum of updates and existing values is kept.
            `"min"`: The minimum of updates and existing values is kept.
            `"mul"`: Updates are multiplied with existing values.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.

    Example:

    Using `reduction="add"` to accumulate values at the same index:

    >>> inputs = np.zeros((4,))
    >>> indices = [[0], [0], [1]]
    >>> updates = np.array([1., 1., 1.])
    >>> keras.ops.scatter_update(inputs, indices, updates, reduction="add")
    array([2., 1., 0., 0.])
    """
    if reduction is not None:
        reduction = reduction.lower()
        if reduction not in ("add", "max", "min", "mul"):
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                "Supported values are: None, 'add', 'max', 'min', 'mul'."
            )
    if any_symbolic_tensors((inputs, indices, updates)):
        return ScatterUpdate(reduction=reduction).symbolic_call(
            inputs, indices, updates
        )
    return backend.ops.core.scatter_update(
        inputs, indices, updates, reduction=reduction
    )


class Slice(Operation):
    def __init__(self, shape, *, name=None):
        super().__init__(name=name)
        self.shape = shape

    def call(self, inputs, start_indices):
        return backend.ops.core.slice(inputs, start_indices, self.shape)

    def compute_output_spec(self, inputs, start_indices):
        if len(self.shape) != len(inputs.shape):
            raise ValueError(
                "The number of dimensions in `inputs` must match the number of "
                f"dimensions in `shape`. Received inputs.shape={inputs.shape} "
                f"and shape={self.shape}"
            )
        if hasattr(start_indices, "__len__") and len(start_indices) != len(
            inputs.shape
        ):
            raise ValueError(
                "The number of dimensions in `start_indices` must match the "
                "number of dimensions in `inputs`. Received "
                f"start_indices={start_indices} and inputs.shape={inputs.shape}"
            )

        final_shape = []
        for i, (input_dim, slice_dim) in enumerate(
            zip(inputs.shape, self.shape)
        ):
            if slice_dim != -1:
                final_shape.append(slice_dim)
            elif isinstance(start_indices, KerasTensor) or input_dim is None:
                final_shape.append(None)
            else:
                final_shape.append(input_dim - start_indices[i])
        return KerasTensor(final_shape, dtype=inputs.dtype)


@keras_export("keras.ops.slice")
def slice(inputs, start_indices, shape):
    """Return a slice of an input tensor.

    At a high level, this operation is an explicit replacement for array slicing
    e.g. `inputs[start_indices: start_indices + shape]`.
    Unlike slicing via brackets, this operation will accept tensor start
    indices on all backends, which is useful when indices dynamically computed
    via other tensor operations.

    ```python
    inputs = np.zeros((5, 5))
    start_indices = np.array([3, 3])
    shape = np.array([2, 2])
    inputs = keras.ops.slice(inputs, start_indices, shape)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        shape: The full shape of the returned slice.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices)):
        return Slice(shape=shape).symbolic_call(inputs, start_indices)
    return backend.ops.core.slice(inputs, start_indices, shape)


class SliceUpdate(Operation):
    def call(self, inputs, start_indices, updates):
        return backend.ops.core.slice_update(inputs, start_indices, updates)

    def compute_output_spec(self, inputs, start_indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.ops.slice_update")
def slice_update(inputs, start_indices, updates):
    """Update an input by slicing in a tensor of updated values.

    At a high level, this operation does
    `inputs[start_indices: start_indices + updates.shape] = updates`.
    Assume inputs is a tensor of shape `(D0, D1, ..., Dn)`,
    `start_indices` must be a list/tuple of n integers, specifying the starting
    indices. `updates` must have the same rank as `inputs`, and the size of each
    dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
    inputs `inputs = np.zeros((5, 5))`, and we want to update the intersection
    of last 2 rows and last 2 columns as 1, i.e.,
    `inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:

    ```python
    inputs = np.zeros((5, 5))
    start_indices = [3, 3]
    updates = np.ones((2, 2))
    inputs = keras.ops.slice_update(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        updates: A tensor, the new values to be put to `inputs` at `indices`.
            `updates` must have the same rank as `inputs`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, updates)):
        return SliceUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.ops.core.slice_update(inputs, start_indices, updates)


class Switch(Operation):
    def call(self, index, branches, *operands):
        return backend.ops.core.switch(index, branches, *operands)

    def compute_output_spec(self, index, branches, *operands):
        # We use first branch for output_spec
        spec = backend.compute_output_spec(branches[0], *operands)
        return spec


@keras_export("keras.ops.switch")
def switch(index, branches, *operands):
    """Apply exactly one of the `branches` given by `index`.

    If `index` is out of bounds, it is clamped to within bounds.

    The semantics of `switch` are given roughly by this Python implementation:

    ```python
    def switch(index, branches, *operands):
        index = clamp(0, index, len(branches) - 1)
        return branches[index](*operands)
    ```

    Args:
        index: An integer scalar indicating which branch function to apply.
        branches: A sequence of functions to be applied based on `index`.
        operands: Inputs to whichever branch is applied.

    Returns:
        The outputs of `branch(*operands)` for the branch that was selected
        based on `index`.

    Examples:

    >>> add_fn = lambda x, y: x + y
    >>> subtract_fn = lambda x, y: x - y
    >>> x = keras.ops.array(2.0)
    >>> y = keras.ops.array(0.5)
    >>> branches = [add_fn, subtract_fn]
    >>> keras.ops.switch(0, branches, x, y)
    2.5

    >>> keras.ops.switch(1, branches, x, y)
    1.5
    """
    if any_symbolic_tensors(operands):
        return Switch().symbolic_call(index, branches, *operands)
    return backend.ops.core.switch(index, branches, *operands)


class WhileLoop(Operation):
    def __init__(self, cond, body, maximum_iterations=None, *, name=None):
        super().__init__(name=name)
        self.cond = cond
        self.body = body
        self.maximum_iterations = maximum_iterations

    def call(self, loop_vars):
        return backend.ops.core.while_loop(
            self.cond,
            self.body,
            loop_vars,
            maximum_iterations=self.maximum_iterations,
        )

    def compute_output_spec(self, loop_vars):
        return tree.map_structure(
            lambda v: KerasTensor(v.shape, dtype=v.dtype), loop_vars
        )


@keras_export("keras.ops.while_loop")
def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    """While loop implementation.

    Args:
        cond: A callable that represents the termination condition of the loop.
            Must accept a `loop_vars` like structure as an argument. If
            `loop_vars` is a tuple or list, each element of `loop_vars` will be
            passed positionally to the callable.
        body: A callable that represents the loop body. Must accept a
            `loop_vars` like structure as an argument, and return update value
            with the same structure. If `loop_vars` is a tuple or list, each
            element of `loop_vars` will be passed positionally to the callable.
        loop_vars: An arbitrary nested structure of tensor state to persist
            across loop iterations.
        maximum_iterations: Optional maximum number of iterations of the while
            loop to run. If provided, the `cond` output is AND-ed with an
            additional condition ensuring the number of iterations executed is
            no greater than `maximum_iterations`.

    Returns:
        A list/tuple of tensors, has the same shape and dtype as `inputs`.

    Examples:

    >>> i = 0
    >>> cond = lambda i: i < 10
    >>> body = lambda i: i + 1
    >>> keras.ops.while_loop(cond, body, i)
    10

    >>> x, y = 0, 1
    >>> cond = lambda x, y: x < 10
    >>> body = lambda x, y: (x + 1, y + 1)
    >>> keras.ops.while_loop(cond, body, (x, y))
    10, 11
    """
    if any_symbolic_tensors((loop_vars,)):
        return WhileLoop(
            cond, body, maximum_iterations=maximum_iterations
        ).symbolic_call(loop_vars)
    return backend.ops.core.while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )


class StopGradient(Operation):
    def call(self, variable):
        return backend.ops.core.stop_gradient(variable)

    def compute_output_spec(self, variable):
        return KerasTensor(variable.shape, dtype=variable.dtype)


@keras_export("keras.ops.stop_gradient")
def stop_gradient(variable):
    """Stops gradient computation.

    Args:
        variable: A tensor variable for which the gradient
            computation is to be disabled.

    Returns:
        The variable with gradient computation disabled.

    Examples:

    >>> var = keras.ops.convert_to_tensor(
    ...     [1., 2., 3.],
    ...     dtype="float32"
    ... )
    >>> var = keras.ops.stop_gradient(var)
    """
    if any_symbolic_tensors((variable,)):
        return StopGradient().symbolic_call(variable)
    return backend.ops.core.stop_gradient(variable)


class ForiLoop(Operation):
    def __init__(self, lower, upper, body_fun, *, name=None):
        super().__init__(name=name)
        self.lower = lower
        self.upper = upper
        self.body_fun = body_fun

    def call(self, init_val):
        return backend.ops.core.fori_loop(
            self.lower,
            self.upper,
            self.body_fun,
            init_val,
        )

    def compute_output_spec(self, init_val):
        return KerasTensor(init_val.shape, dtype=init_val.dtype)


@keras_export("keras.ops.fori_loop")
def fori_loop(lower, upper, body_fun, init_val):
    """For loop implementation.

    Args:
        lower: The initial value of the loop variable.
        upper: The upper bound of the loop variable.
        body_fun: A callable that represents the loop body. Must take two
            arguments: the loop variable and the loop state. The loop state
            should be updated and returned by this function.
        init_val: The initial value of the loop state.

    Returns:
        The final state after the loop.

    Example:

    >>> lower = 0
    >>> upper = 10
    >>> body_fun = lambda i, s: (i + 1, s + i)
    >>> init_val = 0
    >>> keras.ops.fori_loop(lower, upper, body_fun, init_val)
    45
    """
    if any_symbolic_tensors((lower, upper, init_val)):
        return ForiLoop(lower, upper, body_fun).symbolic_call(init_val)
    return backend.ops.core.fori_loop(lower, upper, body_fun, init_val)


class Unstack(Operation):
    def __init__(self, num=None, axis=0, *, name=None):
        super().__init__(name=name)
        self.num = num
        self.axis = axis

    def call(self, x):
        return backend.ops.core.unstack(x, self.num, self.axis)

    def compute_output_spec(self, x):
        axis = canonicalize_axis(self.axis, len(x.shape))
        output_shapes = x.shape[:axis] + x.shape[axis + 1 :]
        num = self.num
        if num is None:
            num = x.shape[axis]
        if num is None:
            raise ValueError(
                "Cannot infer argument `num` from shape "
                f"{x.shape}. Either provide a tensor with a "
                "concrete shape in the `axis` dimension or "
                "explicitly pass the `num` argument."
            )
        output = [
            KerasTensor(shape=output_shapes, dtype=x.dtype) for _ in range(num)
        ]
        return output


@keras_export("keras.ops.unstack")
def unstack(x, num=None, axis=0):
    """Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
        x: The input tensor.
        num: The length of the dimension axis. Automatically inferred
            if `None`.
        axis: The axis along which to unpack.

    Returns:
        A list of tensors unpacked along the given axis.

    Example:

    >>> x = keras.ops.array([[1, 2], [3, 4]])
    >>> keras.ops.unstack(x, axis=0)
    [array([1, 2]), array([3, 4])]
    """
    if any_symbolic_tensors((x,)):
        return Unstack(num, axis).symbolic_call(x)
    return backend.ops.core.unstack(x, num=num, axis=axis)


@keras_export("keras.ops.shape")
def shape(x):
    """Gets the shape of the tensor input.

    Note: On the TensorFlow backend, when `x` is a `tf.Tensor` with dynamic
    shape, dimensions which are dynamic in the context of a compiled function
    will have a `tf.Tensor` value instead of a static integer value.

    Args:
        x: A tensor. This function will try to access the `shape` attribute of
            the input tensor.

    Returns:
        A tuple of integers or None values, indicating the shape of the input
            tensor.

    Example:

    >>> x = keras.ops.zeros((8, 12))
    >>> keras.ops.shape(x)
    (8, 12)
    """
    if any_symbolic_tensors((x,)):
        return x.shape
    return backend.ops.core.shape(x)


@keras_export("keras.ops.dtype")
def dtype(x):
    """Return the dtype of the tensor input as a standardized string.

    Note that due to the standardization, the dtype will not compare equal
    to the backend-specific version of the dtype.

    Args:
        x: A tensor. This function will try to access the `dtype` attribute of
            the input tensor.

    Returns:
        A string indicating the dtype of the input tensor, e.g. `"float32"`.

    Example:

    >>> x = keras.ops.zeros((8, 12))
    >>> keras.ops.dtype(x)
    'float32'

    """
    return standardize_dtype(x.dtype)


class Cast(Operation):
    def __init__(self, dtype, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return backend.ops.core.cast(x, self.dtype)

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)


@keras_export("keras.ops.cast")
def cast(x, dtype):
    """Cast a tensor to the desired dtype.

    Args:
        x: A tensor or variable.
        dtype: The target type.

    Returns:
        A tensor of the specified `dtype`.

    Example:

    >>> x = keras.ops.arange(4)
    >>> x = keras.ops.cast(x, dtype="float16")
    """
    dtype = standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return Cast(dtype=dtype)(x)
    return backend.ops.core.cast(x, dtype)


class SaturateCast(Operation):
    def __init__(self, dtype, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def call(self, x):
        return _saturate_cast(x, self.dtype)

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)


@keras_export("keras.ops.saturate_cast")
def saturate_cast(x, dtype):
    """Performs a safe saturating cast to the desired dtype.

    Saturating cast prevents data type overflow when casting to `dtype` with
    smaller values range. E.g.
    `ops.cast(ops.cast([-1, 256], "float32"), "uint8")` returns `[255, 0]`,
    but `ops.saturate_cast(ops.cast([-1, 256], "float32"), "uint8")` returns
    `[0, 255]`.

    Args:
        x: A tensor or variable.
        dtype: The target type.

    Returns:
        A safely casted tensor of the specified `dtype`.

    Example:

    Image resizing with bicubic interpolation may produce values outside
    original range.
    >>> image2x2 = np.array([0, 1, 254, 255], dtype="uint8").reshape(1, 2, 2, 1)
    >>> image4x4 = tf.image.resize(image2x2, (4, 4), method="bicubic")
    >>> print(image4x4.numpy().squeeze())
    >>> # [[-22.500004 -22.204624 -21.618908 -21.32353 ]
    >>> #  [ 52.526054  52.82143   53.407146  53.70253 ]
    >>> #  [201.29752  201.59288  202.17859  202.47395 ]
    >>> #  [276.32355  276.61893  277.20465  277.50006 ]]

    Casting this resized image back to `uint8` will cause overflow.
    >>> image4x4_casted = ops.cast(image4x4, "uint8")
    >>> print(image4x4_casted.numpy().squeeze())
    >>> # [[234 234 235 235]
    >>> #  [ 52  52  53  53]
    >>> #  [201 201 202 202]
    >>> #  [ 20  20  21  21]]

    Saturate casting to `uint8` will clip values to `uint8` range before
    casting and will not cause overflow.
    >>> image4x4_saturate_casted = ops.saturate_cast(image4x4, "uint8")
    >>> print(image4x4_saturate_casted.numpy().squeeze())
    >>> # [[  0   0   0   0]
    >>> #  [ 52  52  53  53]
    >>> #  [201 201 202 202]
    >>> #  [255 255 255 255]]

    """
    dtype = standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return SaturateCast(dtype=dtype)(x)
    return _saturate_cast(x, dtype)


def _saturate_cast(x, dtype, backend_module=None):
    backend_module = backend_module or backend

    def get_dtype_min_max(dtype):
        if "bool" == dtype:
            dtype_min = 0
            dtype_max = 1
        elif "int" in dtype:
            dtype_min = ml_dtypes.iinfo(dtype).min
            dtype_max = ml_dtypes.iinfo(dtype).max
        else:
            dtype_min = ml_dtypes.finfo(dtype).min
            dtype_max = ml_dtypes.finfo(dtype).max
        return dtype_min, dtype_max

    dtype = standardize_dtype(dtype)
    in_dtype = standardize_dtype(x.dtype)
    in_min, in_max = get_dtype_min_max(in_dtype)
    out_min, out_max = get_dtype_min_max(dtype)

    # The output min/max may not actually be representable in the
    # in_dtype (e.g. casting float32 to uint32).  This can lead to undefined
    # behavior when trying to cast a value outside the valid range of the
    # target type. We work around this by nudging the min/max to fall within
    # the valid output range. The catch is that we may actually saturate
    # to a value less than the true saturation limit, but this is the best we
    # can do in order to avoid UB without backend op.
    min_limit = np.maximum(in_min, out_min).astype(in_dtype)
    if min_limit < out_min:
        min_limit = np.nextafter(min_limit, 0, dtype=in_dtype)
    max_limit = np.minimum(in_max, out_max).astype(in_dtype)
    if max_limit > out_max:
        max_limit = np.nextafter(max_limit, 0, dtype=in_dtype)

    # Unconditionally apply `clip` to fix `inf` behavior.
    x = backend_module.ops.numpy.clip(x, min_limit, max_limit)

    return backend_module.ops.cast(x, dtype)


class ConvertToTensor(Operation):
    def __init__(self, dtype=None, sparse=None, ragged=None, *, name=None):
        super().__init__(name=name)
        self.dtype = dtype
        self.sparse = sparse
        self.ragged = ragged

    def call(self, x):
        return backend.ops.core.convert_to_tensor(
            x, dtype=self.dtype, sparse=self.sparse, ragged=self.ragged
        )

    def compute_output_spec(self, x):
        dtype = standardize_dtype(x.dtype) if self.dtype is None else self.dtype
        sparse = (
            False if self.sparse is not None and not self.sparse else x.sparse
        )
        ragged = (
            False if self.ragged is not None and not self.ragged else x.ragged
        )
        return backend.KerasTensor(
            shape=x.shape, dtype=dtype, sparse=sparse, ragged=ragged
        )


@keras_export("keras.ops.convert_to_tensor")
def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    """Convert a NumPy array or Python array to a tensor.

    Native tensors for the current backend or left unchanged unless the `dtype`,
    `sparse` or `ragged` arguments are set.

    Args:
        x: A NumPy array, Python array (can be nested) or a backend tensor.
        dtype: The target type. If `None`, the type of `x` is used.
        sparse: Whether to keep sparse tensors. `False` will cause sparse
            tensors to be densified. The default value of `None` means that
            sparse tensors are kept only if the backend supports them.
        ragged: Whether to keep ragged tensors. `False` will cause ragged
            tensors to be densified. The default value of `None` means that
            ragged tensors are kept only if the backend supports them.

    Returns:
        A backend tensor of the specified `dtype` and sparseness.

    Example:

    >>> x = np.array([1, 2, 3])
    >>> y = keras.ops.convert_to_tensor(x)
    """
    dtype = None if dtype is None else standardize_dtype(dtype)
    if any_symbolic_tensors((x,)):
        return ConvertToTensor(dtype=dtype, sparse=sparse, ragged=ragged)(x)
    return backend.ops.core.convert_to_tensor(
        x, dtype=dtype, sparse=sparse, ragged=ragged
    )


@keras_export("keras.ops.convert_to_numpy")
def convert_to_numpy(x):
    """Convert a tensor to a NumPy array.

    Args:
        x: A tensor.

    Returns:
        A NumPy array.
    """
    if any_symbolic_tensors((x,)):
        # This will raise a `ValueError` defined in the `KerasTensor` class.
        # We trigger it rather than duplicate it here.
        return np.array(x)
    return backend.ops.core.convert_to_numpy(x)


class Cond(Operation):
    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        # `Operation.__call__` routes through `rematerialized_call` whenever a
        # `RematScope` is active, but `call` receives Python callables rather
        # than tensors and rematerialization cannot trace those, so `Cond`
        # dispatches on its own and stays out of that path.
        def call_fn(*args, **kwargs):
            if any_symbolic_tensors(args, kwargs):
                return self.symbolic_call(*args, **kwargs)
            else:
                return self.call(*args, **kwargs)

        if traceback_utils.is_traceback_filtering_enabled():
            try:
                return call_fn(*args, **kwargs)
            except Exception as e:
                if not getattr(e, "_keras_call_info_injected", False):
                    raise traceback_utils.inject_argument_info_in_error(
                        e,
                        self.call,
                        args,
                        kwargs,
                        object_name=(f"{self.__class__.__name__}.call()"),
                    ) from None
                raise

        # Plain flow.
        return call_fn(*args, **kwargs)

    def call(self, pred, true_fn, false_fn):
        return backend.ops.core.cond(pred, true_fn, false_fn)

    def compute_output_spec(self, pred, true_fn, false_fn):
        true_fn_spec = backend.compute_output_spec(true_fn)
        false_fn_spec = backend.compute_output_spec(false_fn)
        if not self._check_output_spec(true_fn_spec, false_fn_spec):
            raise ValueError(
                "`true_fn` and `false_fn` should return outputs "
                "of the same kind (struct, dtype and shape). "
                f"Got {true_fn_spec} and {false_fn_spec} instead."
            )
        return true_fn_spec

    def _check_output_spec(self, true_fn_spec, false_fn_spec):
        try:
            tree.assert_same_structure(true_fn_spec, false_fn_spec)
        except:
            return False

        def check_leaf(t_spec, f_spec):
            if t_spec is None or f_spec is None:
                return t_spec is None and f_spec is None
            return t_spec.shape == f_spec.shape and t_spec.dtype == f_spec.dtype

        same = tree.map_structure(check_leaf, true_fn_spec, false_fn_spec)
        return all(tree.flatten(same))


@keras_export("keras.ops.cond")
def cond(pred, true_fn, false_fn):
    """Conditionally applies `true_fn` or `false_fn`.

    Args:
        pred: Boolean scalar type
        true_fn: Callable returning the output for the `pred == True` case.
        false_fn: Callable returning the output for the `pred == False` case.

    Returns:
        The output of either `true_fn` or `false_fn` depending on pred.
    """
    return Cond()(pred, true_fn, false_fn)


class VectorizedMap(Operation):
    def __init__(self, function, *, name=None):
        super().__init__(name=name)
        self.function = function

    def call(self, elements):
        return backend.ops.core.vectorized_map(self.function, elements)

    def compute_output_spec(self, elements):
        x = tree.map_structure(lambda t: t[0], elements)
        n = tree.flatten(elements)[0].shape[0]
        y = backend.compute_output_spec(self.function, x)

        def append_batch_axis(t):
            return KerasTensor(
                shape=(n,) + t.shape,
                dtype=t.dtype,
                sparse=t.sparse,
                ragged=t.ragged,
            )

        y = tree.map_structure(append_batch_axis, y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({"function": self.function})
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["function"] = serialization_lib.deserialize_keras_object(
            config["function"]
        )
        return cls(**config)


@keras_export("keras.ops.vectorized_map")
def vectorized_map(function, elements):
    """Parallel map of `function` on axis 0 of tensor(s) `elements`.

    Schematically, `vectorized_map` implements the following,
    in the case of a single tensor input `elements`:

    ```python
    def vectorized_map(function, elements):
        outputs = []
        for e in elements:
            outputs.append(function(e))
        return np.stack(outputs)
    ```

    In the case of an iterable of tensors `elements`,
    it implements the following:

    ```python
    def vectorized_map(function, elements):
        batch_size = elements[0].shape[0]
        outputs = []
        for index in range(batch_size):
            outputs.append(function([e[index] for e in elements]))
        return np.stack(outputs)
    ```

    In this case, `function` is expected to take as input
    a single list of tensor arguments.
    """
    if any_symbolic_tensors((elements,)):
        return VectorizedMap(function)(elements)
    return backend.ops.core.vectorized_map(function, elements)


@keras_export("keras.ops.is_tensor")
def is_tensor(x):
    """Check whether the given object is a tensor.

    Note: This checks for backend specific tensors so passing a TensorFlow
    tensor would return `False` if your backend is PyTorch or JAX.

    Args:
        x: A variable.

    Returns:
        `True` if `x` is a tensor, otherwise `False`.
    """
    return backend.ops.core.is_tensor(x)


@keras_export("keras.ops.grad")
def grad(f, argnums=0):
    """Return a function that computes the gradient of `f`.

    Experimental. This works on concrete tensors only, not symbolic ones,
    and raises `NotImplementedError` on backends without automatic
    differentiation. A non-scalar output is summed before differentiating,
    the same as a gradient tape would.

    Args:
        f: A function returning a tensor. The arguments at `argnums` may be
            tensors, variables or nested structures of them.
        argnums: An integer or a tuple of integers, the positions of the
            arguments to differentiate with respect to. Defaults to `0`.

    Returns:
        A function with the signature of `f` that returns the gradient for
        each position in `argnums`, a single value for an integer and a
        tuple for a tuple. Arguments that do not affect the output get zeros.

    Example:

    ```python
    def f(x, y):
        return x * y

    x = ops.array([1.0, 2.0])
    y = ops.array([3.0, 4.0])
    dx = ops.grad(f)(x, y)
    # dx is [3.0, 4.0]
    dx, dy = ops.grad(f, argnums=(0, 1))(x, y)
    # dy is [1.0, 2.0]
    ops.grad(ops.tanh)(x)
    # the elementwise derivative of tanh at x
    ```
    """
    return backend.ops.core.grad(f, argnums=argnums)


@keras_export("keras.ops.custom_gradient")
def custom_gradient(f):
    """Decorator to define a function with a custom gradient.

    This decorator allows fine grained control over the gradients of a sequence
    for operations. This may be useful for multiple reasons, including providing
    a more efficient or numerically stable gradient for a sequence of
    operations.

    Args:
        f: Function `f(*args)` that returns a tuple
            `(output, grad_fn)`, where:
            - `args` is a sequence of (nested structures of) tensor inputs to
                the function.
            - `output` is a (nested structure of) tensor outputs of applying
                operations in `forward_fn` to `args`.
            - `grad_fn` is a function with the signature `grad_fn(*args,
                upstream)` which returns a tuple of tensors the same size as
                (flattened) `args`: the derivatives of tensors in `output` with
                respect to the tensors in `args`. `upstream` is a tensor or
                sequence of tensors holding the initial value gradients for each
                tensor in `output`.

    Returns:
        A function `h(*args)` which returns the same value as
        `f(*args)[0]` and whose gradient is determined by
        `f(*args)[1]`.


    Examples:

    1. Backend-agnostic example.

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))

        return ops.log(1 + e), grad
    ```

    Note that the grad function that returns gradient computation
    requires `args` as well as an `upstream` keyword argument, depending
    on the backend being set. With the JAX and TensorFlow backends,
    it requires only one argument, whereas it might use the `upstream`
    argument in the case of the PyTorch backend.

    When working with TensorFlow/JAX backend, `grad(upstream)`
    is sufficient. With PyTorch, the `grad` function requires
    `*args` as well as `upstream`, e.g. `def grad(*args, upstream)`.
    Follow the previous example to use `@ops.custom_gradient` in
    a way that is compatible with all backends.

    2. Here's JAX & TensorFlow-specific example:

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)
        def grad(upstream):
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
        return ops.log(1 + e), grad
    ```

    3. Lastly, here's a PyTorch-specific example,
    using `*args` & `upstream`:

    ```python
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)
        def grad(*args, upstream):
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
        return ops.log(1 + e), grad
    ```
    """
    return backend.ops.core.custom_gradient(f)
