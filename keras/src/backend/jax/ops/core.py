import contextlib
import inspect
import math as python_math

import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import export as jax_export

from keras.src import tree
from keras.src.backend import config
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import standardize_argnums
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib


class Variable(KerasVariable):
    def _initialize_layout(self):
        # We can't import the keras/distribution/distribution_lib
        # due to circular dependency.
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            self._layout = distribution.get_variable_layout(self)

    def _initialize(self, value):
        # Note that variable.shape is needed by distribution_lib
        self._shape = self._validate_shape(value.shape)
        self._initialize_layout()
        self._direct_assign(value)

    def _initialize_with_initializer(self, initializer):
        from keras.src.distribution.distribution_lib import TensorLayout

        self._initialize_layout()

        if self._layout is None:
            super()._initialize_with_initializer(initializer)
        elif "layout" in inspect.signature(initializer).parameters:
            jax_layout = (
                self._layout.backend_layout
                if isinstance(self._layout, TensorLayout)
                else self._layout
            )
            with jax.set_mesh(jax_layout.mesh):
                # Force explicit axes in this context, otherwise `out_sharding`
                # in ops ends up being ignored.

                @jax.sharding.explicit_axes
                def explicit_initializer():
                    explicit_layout = jax.sharding.NamedSharding(
                        jax.sharding.get_abstract_mesh(), jax_layout.spec
                    )
                    return initializer(
                        self._shape, dtype=self._dtype, layout=explicit_layout
                    )

                self._value = explicit_initializer(in_sharding=None)
        elif should_shard_at_init(self._layout, self._shape):
            jax_layout = (
                self._layout.backend_layout
                if isinstance(self._layout, TensorLayout)
                else self._layout
            )
            jitted_initializer = jax.jit(
                initializer.__call__,
                out_shardings=jax_layout,
                static_argnames=["shape", "dtype"],
            )
            value = jitted_initializer(shape=self._shape, dtype=self._dtype)
            self._value = value
        else:
            super()._initialize_with_initializer(initializer)

    def _direct_assign(self, value):
        if self._layout is not None:
            value = distribution_lib.distribute_tensor(value, self._layout)
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


if config.is_nnx_enabled():
    from flax import nnx

    # fallback logic for NNX Variable's raw_value attribute. This is needed for
    # compatibility with older versions of NNX that don't have get_raw_value().
    _NNX_RAW_VALUE_ATTR = (
        "_raw_value" if hasattr(nnx.Variable, "get_raw_value") else "raw_value"
    )

    def _get_raw_value(variable):
        return getattr(variable, _NNX_RAW_VALUE_ATTR, None)

    def _set_raw_value(variable, value):
        object.__setattr__(variable, _NNX_RAW_VALUE_ATTR, value)

    # NNX stamps every object with the trace that was active when it was created
    # and refuses to mutate it from any other trace. Objects created this way
    # outlive the shape-inference trace and hold concrete values, so they
    # are stamped with the enclosing trace instead: once inference
    # returns, they behave exactly like eagerly created objects.
    _enclosing_traces = []

    def stamp_with_enclosing_trace(obj):
        """Stamp an NNX object with the trace enclosing shape inference."""
        if not _enclosing_traces:
            return
        trace_state = getattr(obj, "_trace_state", None)  # nnx.Variable
        if trace_state is None:  # nnx.Module, via its state object
            for attr_value in vars(obj).values():
                trace_state = getattr(attr_value, "_trace_state", None)
                if trace_state is not None:
                    break
        if trace_state is not None:
            object.__setattr__(trace_state, "_jax_trace", _enclosing_traces[0])

    JaxVariable = Variable

    class Variable(JaxVariable, nnx.Variable):
        def __init__(
            self,
            initializer,
            shape=None,
            dtype=None,
            trainable=True,
            autocast=True,
            aggregation="none",
            synchronization="auto",
            name=None,
            layout=None,
            mutable=None,
            **nnx_metadata,
        ):
            # Ensure 'mutable' is in nnx_metadata, but explicit 'mutable'
            # param takes precedence.
            nnx_metadata["mutable"] = True if mutable is None else mutable

            # First, initialize a basic nnx.Variable with a dummy value
            # This sets up the NNX variable structure
            if shape is None:
                dummy_value = jnp.array(0.0)
            else:
                dummy_value = jnp.zeros(shape, dtype=standardize_dtype(dtype))

            # Initialize nnx.Variable first
            nnx.Variable.__init__(self, value=dummy_value, **nnx_metadata)

            # Initialize JaxVariable (which will call KerasVariable.__init__
            # and set up the real value).
            JaxVariable.__init__(
                self,
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                synchronization=synchronization,
                name=name,
                layout=layout,
            )

            # The real value is now set in self._value, sync it to raw_value
            _set_raw_value(self, self._value)

            stamp_with_enclosing_trace(self)

        def _initialize_with_initializer(self, initializer):
            value = self._convert_to_tensor(
                initializer(self._shape, dtype=self._dtype)
            )
            self._initialize(value)

        @property
        def _value(self):
            return _get_raw_value(self)

        @_value.setter
        def _value(self, new_keras_value):
            self._direct_assign(new_keras_value)

        def __getstate__(self):
            # Get the state from KerasVariable (attributes in __dict__)
            # KerasVariable does not have a custom __getstate__, so we mimic
            # default behavior.
            try:
                keras_state = KerasVariable.__getstate__(self)
            except AttributeError:
                keras_state = object.__getstate__(self)

            # Get the state from nnx.Variable
            nnx_specific_state = nnx.Variable.__getstate__(self)

            # Merge them. Keras state is primary. NNX specific state adds
            # to it.
            if _NNX_RAW_VALUE_ATTR in nnx_specific_state:
                keras_state["_value"] = nnx_specific_state[_NNX_RAW_VALUE_ATTR]

            # Add NNX attributes that are not in Keras's __dict__
            if "_trace_state" in nnx_specific_state:
                keras_state["_trace_state"] = nnx_specific_state["_trace_state"]
            if "_var_metadata" in nnx_specific_state:
                keras_state["_var_metadata"] = nnx_specific_state[
                    "_var_metadata"
                ]

            # Remove elements that might be problematic or redundant if
            # nnx.Variable's __getstate__
            keras_state.pop(_NNX_RAW_VALUE_ATTR, None)

            return keras_state

        def __setstate__(self, state):
            # Separate nnx specific keys that we added if they are not part
            # of Keras __dict__ this __getstate__ puts them into the main
            # state dictionary.
            nnx_raw_value = state["_value"]  # This was raw_value
            nnx_trace_state = state.pop("_trace_state", None)
            nnx_var_metadata = state.pop("_var_metadata", None)

            # Populate the instance's __dict__ with the Keras attributes.
            self.__dict__.update(state)

            # restore the nnx.Variable specific slotted attributes.
            _set_raw_value(self, nnx_raw_value)

            if nnx_trace_state is not None:
                object.__setattr__(self, "_trace_state", nnx_trace_state)
            else:
                pass

            if nnx_var_metadata is not None:
                object.__setattr__(self, "_var_metadata", nnx_var_metadata)
            else:
                pass

            # Ensure Keras's self._value is also consistent with the
            # restored raw_value
            self._value = nnx_raw_value

            if hasattr(self, "_shape") and self._shape is not None:
                self._ndim = len(self._shape)
            else:
                # Fallback if shape isn't immediately available.
                self._ndim = len(_get_raw_value(self).shape)

        def _direct_assign(self, value):
            # Apply JAX-specific distribution if layout is present
            if self._layout is not None:
                value = distribution_lib.distribute_tensor(value, self._layout)

            # Apply on_set_value hook if it exists
            if (
                hasattr(self, "_var_metadata")
                and "on_set_value" in self._var_metadata
            ):
                value = self._var_metadata["on_set_value"](self, value)

            # Set the value for both Keras and NNX parts
            # This ensures both systems see the same value
            _set_raw_value(self, value)

        @property
        def value(self):
            if in_stateless_scope():
                scope = get_stateless_scope()
                stateless_value = scope.get_current_value(self)
                if stateless_value is not None:
                    return self._maybe_autocast(stateless_value)
            current_value = _get_raw_value(self)
            if current_value is None:
                # Uninitialized variable (created under a stateless scope,
                # which defers initialization). Return a placeholder, like
                # `KerasVariable.value`, without mutating the variable.
                if self._initializer is not None:
                    return self._maybe_autocast(
                        self._initializer(self._shape, dtype=self._dtype)
                    )
                else:
                    raise AttributeError(
                        "Variable is not properly initialized (raw_value "
                        "missing) and has no initializer."
                    )
            if (
                hasattr(self, "_var_metadata")
                and "on_get_value" in self._var_metadata
            ):
                current_value = self._var_metadata["on_get_value"](
                    self, current_value
                )
            return self._maybe_autocast(current_value)

    def _flatten_nnx_variable(variable):
        children = (_get_raw_value(variable),)
        # We copy __dict__ to avoid side effects
        keras_state = variable.__dict__.copy()
        # Remove elements that might be problematic or redundant if
        # nnx.Variable's __getstate__
        keras_state.pop(_NNX_RAW_VALUE_ATTR, None)
        aux_data = (
            variable._var_metadata,
            getattr(variable, "_trace_state", None),
            keras_state,
        )
        return children, aux_data

    def _unflatten_nnx_variable(aux_data, children):
        var_metadata, trace_state, keras_state = aux_data
        raw_value = children[0]

        # Create uninitialized instance
        variable = Variable.__new__(Variable)

        # Restore state
        variable._var_metadata = var_metadata
        if trace_state is not None:
            variable._trace_state = trace_state
        variable.__dict__.update(keras_state)
        _set_raw_value(variable, raw_value)

        return variable

    try:
        jax.tree_util.register_pytree_node(
            Variable,
            _flatten_nnx_variable,
            _unflatten_nnx_variable,
        )
    except ValueError:
        pass

    def __setattr__(self, name, value):
        # Mirror Keras attributes to _var_metadata to ensure persistence
        # if the Pytree registration is not respected by NNX.
        if (
            name != "_var_metadata"
            and name not in ("raw_value", "_raw_value", "_trace_state")
            and hasattr(self, "_var_metadata")
        ):
            self._var_metadata[name] = value

        object.__setattr__(self, name, value)

    Variable.__setattr__ = __setattr__


def should_shard_at_init(init_layout, shape):
    from keras.src.distribution.distribution_lib import TensorLayout

    if isinstance(init_layout, TensorLayout):
        devices_count = init_layout.device_mesh.devices.size
    elif isinstance(init_layout, jax.sharding.NamedSharding):
        devices_count = init_layout.mesh.devices.size
    else:
        # Keras itself creates `NamedSharding` instances that have a `mesh`
        # attribute. However, KerasRS uses layouts and formats, which do not
        # directly have a mesh attribute. Apply the sharding in this case.
        return True

    size_threshold = 250 * 1024 * 1024
    # We multiply by the mesh size here to take into account the worst case
    # scenario of the array being first duplicated in the memory of one device
    # before being transferred to the other devices.
    size = python_math.prod(shape) * 4 * devices_count
    return size >= size_threshold


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, (jnp.ndarray, jax.Array)) and (
        dtype is None or x.dtype == dtype
    ):
        # Skip the conversion early if the instance is already a JAX array.
        # This is important in the multi-process context since jax.array(x) for
        # an existing distributed jax array will raise error.
        return x

    if isinstance(x, Variable):
        if dtype is not None and x.dtype != dtype:
            return x.value.astype(dtype)
        return x.value

    if isinstance(x, jax_sparse.JAXSparse):
        if sparse is not None and not sparse:
            x = x.todense()
        elif dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        else:
            return x

    # The branch must run only for an explicit bfloat16 request:
    if not is_tensor(x) and dtype is not None and dtype == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return jnp.asarray(x).astype(dtype)
    return jnp.asarray(x, dtype=dtype)


def convert_to_numpy(x):
    if isinstance(x, jax_sparse.JAXSparse):
        x = x.todense()
    if is_tensor(x) and x.dtype == "bfloat16":
        return np.array(x, dtype=ml_dtypes.bfloat16)
    return np.array(x)


def is_tensor(x):
    if isinstance(x, (jnp.ndarray, jax_sparse.JAXSparse)):
        return True
    return False


def shape(x):
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


@contextlib.contextmanager
def _track_enclosing_trace():
    """Record the trace enclosing symbolic shape inference, for NNX."""
    if not config.is_nnx_enabled():
        yield
        return
    from flax.nnx import tracers as nnx_tracers

    _enclosing_traces.append(nnx_tracers.current_jax_trace())
    try:
        yield
    finally:
        _enclosing_traces.pop()


# Shape / dtype / sparseness inference util
def compute_output_spec(fn, *args, **kwargs):
    with _track_enclosing_trace(), StatelessScope(), SymbolicScope():
        built_in_types = (type(None), int, float, str, bool, complex, bytes)

        # First, separate symbolic args from other args
        static_args_idx = []
        static_args = []
        maybe_symbolic_args = []
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for idx, arg in enumerate(args):
            if isinstance(arg, built_in_types):
                static_args_idx.append(idx)
                static_args.append(arg)
            else:
                maybe_symbolic_args.append(arg)
        maybe_symbolic_args = tuple(maybe_symbolic_args)
        for k, v in kwargs.items():
            if isinstance(v, built_in_types):
                static_kwargs[k] = v
            else:
                maybe_symbolic_kwargs[k] = v

        # Create a _DimExpr instance for one dimension by creating a symbolic
        # shape with one dimension and extracting it.
        #
        # We create a single dynamic dimension and reuse it instead of creating
        # N dynamic dimensions. This is for backwards compatibility. Previously
        # we would fill all dynamic dimensions with the same concrete value.
        # This can handle the case where there is an implicit assumption that
        # two dimensions are the same (e.g. square images).
        #
        # We add the constraint "dynamic_dimension>=2" to prevent JAX from
        # assuming that the dimension can be broadcastable or squeezable. It
        # removes this ambiguity.
        dynamic_dimension = jax_export.symbolic_shape(
            "(dynamic_dimension)",
            constraints=["dynamic_dimension>=2"],
        )[0]

        def convert_keras_tensor_to_jax(x):
            if isinstance(x, KerasTensor):
                shape = tuple(
                    [d if d is not None else dynamic_dimension for d in x.shape]
                )
                return jax.ShapeDtypeStruct(shape, dtype=x.dtype)
            return x

        def wrapped_fn(*args, **kwargs):
            # Turn inputs that are sparse to BCOO tensors
            def to_bcoo_if_sparse(x, maybe_symbolic_x):
                if (
                    isinstance(maybe_symbolic_x, KerasTensor)
                    and maybe_symbolic_x.sparse
                ):
                    return jax_sparse.BCOO.fromdense(x, nse=1)
                return x

            args, kwargs = tree.map_structure(
                to_bcoo_if_sparse,
                (args, kwargs),
                (maybe_symbolic_args, maybe_symbolic_kwargs),
            )

            rec_args = []
            idx_static = 0
            idx_sym = 0
            i = 0
            while idx_static < len(static_args) or idx_sym < len(args):
                if i in static_args_idx:
                    rec_args.append(static_args[idx_static])
                    idx_static += 1
                else:
                    rec_args.append(args[idx_sym])
                    idx_sym += 1

                i += 1
            with StatelessScope():
                return fn(*rec_args, **kwargs, **static_kwargs)

        maybe_symbolic_args_jax, maybe_symbolic_kwargs_jax = tree.map_structure(
            convert_keras_tensor_to_jax,
            (maybe_symbolic_args, maybe_symbolic_kwargs),
        )
        jax_out = jax.eval_shape(
            wrapped_fn, *maybe_symbolic_args_jax, **maybe_symbolic_kwargs_jax
        )

        def convert_jax_spec_to_keras_tensor(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                shape = tuple(
                    d if isinstance(d, int) else None for d in x.shape
                )
                return KerasTensor(shape, x.dtype)
            elif isinstance(x, jax_sparse.BCOO):
                shape = tuple(
                    d if isinstance(d, int) else None for d in x.shape
                )
                return KerasTensor(shape, x.dtype, sparse=True)
            return x

        return tree.map_structure(convert_jax_spec_to_keras_tensor, jax_out)


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fun=true_fn, false_fun=false_fn)


def vectorized_map(function, elements):
    return jax.vmap(function)(elements)


def map(f, xs):
    return jax.lax.map(f, xs)


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    return jax.lax.scan(
        f, init=init, xs=xs, length=length, reverse=reverse, unroll=unroll
    )


def associative_scan(f, elems, reverse=False, axis=0):
    return jax.lax.associative_scan(f, elems, reverse, axis)


def scatter(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(values)


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    indices = jnp.array(indices)
    indices = jnp.transpose(indices)
    idx = tuple(indices)
    if reduction is None:
        return inputs.at[idx].set(updates)
    elif reduction == "add":
        return inputs.at[idx].add(updates)
    elif reduction == "max":
        return inputs.at[idx].max(updates)
    elif reduction == "min":
        return inputs.at[idx].min(updates)
    elif reduction == "mul":
        return inputs.at[idx].multiply(updates)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def slice(inputs, start_indices, shape):
    # If shape[i] is -1, all remaining elements in dimension i are included in
    # the slice.
    final_shape = tuple(
        inputs.shape[i] - start_indices[i] if s == -1 else s
        for i, s in enumerate(shape)
    )
    return jax.lax.dynamic_slice(inputs, start_indices, final_shape)


def slice_update(inputs, start_indices, updates):
    return jax.lax.dynamic_update_slice(inputs, updates, start_indices)


def switch(index, branches, *operands):
    return jax.lax.switch(index, branches, *operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    if maximum_iterations is not None:
        current_iter = 0
        loop_vars = loop_vars + (current_iter,)

        # Unpack list/tuple args. The last argument is `current_iter`.
        def _cond(args):
            return cond(*args[:-1]) & (args[-1] < maximum_iterations)

        def _body(args):
            outputs = body(*args[:-1])
            outputs = tuple(outputs) if is_tuple else (outputs,)
            return outputs + (args[-1] + 1,)

    else:

        def _cond(args):
            return cond(*args)

        def _body(args):
            outputs = body(*args)
            return tuple(outputs) if is_tuple else (outputs,)

    outputs = jax.lax.while_loop(_cond, _body, loop_vars)
    if maximum_iterations is not None:
        outputs = outputs[:-1]
    return outputs if is_tuple else outputs[0]


def fori_loop(lower, upper, body_fun, init_val):
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    return jax.lax.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return list(jnp.moveaxis(x, axis, 0))


def random_seed_dtype():
    # jax random seed uses uint32.
    return "uint32"


def custom_gradient(fun):
    fun_with_custom_gradient = jax.custom_gradient(fun=fun)

    # Add a wrapper to unwrap variables, otherwise custom_gradient will fail
    def fun_with_custom_gradient_wrapper(*args, **kwargs):
        args, kwargs = tree.map_shape_structure(
            lambda x: x.value if isinstance(x, KerasVariable) else x,
            (args, kwargs),
        )
        return fun_with_custom_gradient(*args, **kwargs)

    return fun_with_custom_gradient_wrapper


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that defines a custom gradient, which
        recomputes f on the backwards pass of a gradient call.
    """
    return jax.checkpoint(f)


class name_scope(base_name_scope):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._jax_name_scope = jax.named_scope(name)

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[], set_to_default=True
        )
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        self._jax_name_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        if self._pop_on_exit:
            self._jax_name_scope.__exit__(*args, **kwargs)


def device_scope(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", etc.
        device_name = device_name.lower()
        jax_device = distribution_lib._to_backend_device(device_name)
    elif not isinstance(device_name, jax.Device):
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or a `jax.Device` instance. "
            f"Received: device_name='{device_name}'"
        )
    else:
        jax_device = device_name
    return jax.default_device(jax_device)


def grad(f, argnums=0):
    def scalar_f(*args, **kwargs):
        # A gradient tape sums a non scalar output, so do the same here.
        return jnp.sum(f(*args, **kwargs))

    def grad_fn(*args, **kwargs):
        positions = standardize_argnums(argnums, len(args))
        args = list(args)
        for i in positions:
            args[i] = tree.map_structure(convert_to_tensor, args[i])
        jax_argnums = positions[0] if isinstance(argnums, int) else positions
        return jax.grad(scalar_f, argnums=jax_argnums)(*args, **kwargs)

    return grad_fn
