import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name


class Variable:
    """Represents a backend-agnostic variable in Keras.

    A `Variable` acts as a container for state. It holds a tensor value and can
    be updated. With the JAX backend, variables are used to implement
    "functionalization", the pattern of lifting stateful operations out of
    a piece of computation to turn it into a stateless function.

    Args:
        initializer: Initial value or callable for initialization.
            If a callable is used, it should take the arguments
            `shape` and `dtype`.
        shape: Optional. Tuple for the variable's shape.
            Required if `initializer` is a callable.
        dtype: Optional. Data type of the variable. Defaults to the global float
            dtype type (`"float32"` if never configured).
        trainable: Optional. Boolean indicating if variable is trainable.
            Defaults to `True`.
        autocast: Optional. Boolean indicating whether the variable supports
            autocasting. If `True`, the layer may first convert the variable
            to the compute data type when accessed. Defaults to `True`.
        aggregation: Optional string, one of `None`, `"none"`, `"mean"`,
            `"sum"` or `"only_first_replica"` specifying how a distributed
            variable will be aggregated. This serves as a semantic annotation,
            to be taken into account by downstream backends or users. Defaults
            to `"none"`.
        name: Optional. A unique name for the variable. Automatically generated
            if not set.
        layout: Optional. Sharding layout for the variable. Either a
            `keras.distribution.TensorLayout` with its `device_mesh` set, or a
            backend-specific layout.
        kwargs: Additional backend-specific keyword arguments.

    Attributes:
        shape: The shape of the variable (tuple of integers).
        ndim: The number of dimensions of the variable (integer).
        dtype: The data type of the variable (string).
        trainable: Whether the variable is trainable (boolean).
        autocast: Whether the variable supports autocasting (boolean).
        aggregation: How a distributed variable will be aggregated (string).
        value: The current value of the variable (NumPy array or tensor).
        name: The name of the variable (string).
        path: The path of the variable within the Keras model or layer (string).

    Examples:

    **Initializing a `Variable` with a NumPy array:**

    ```python
    import numpy as np
    import keras
    initial_array = np.ones((3, 3))
    variable_from_array = keras.Variable(initializer=initial_array)
    ```

    **Using a Keras initializer to create a `Variable`:**

    ```python
    from keras.src.initializers import Ones
    variable_from_initializer = keras.Variable(
        initializer=Ones(), shape=(3, 3), dtype="float32"
    )
    ```

    **Updating the value of a `Variable`:**

    ```python
    new_value = np.zeros((3, 3), dtype="float32")
    variable_from_array.assign(new_value)
    ```

    **Marking a `Variable` as non-trainable:**

    ```python
    non_trainable_variable = keras.Variable(
        initializer=np.ones((3, 3), dtype="float32"), trainable=False
    )
    ```
    """

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
        **kwargs,
    ):
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name}"
            )
        if aggregation not in (
            None,
            "none",
            "mean",
            "sum",
            "only_first_replica",
        ):
            raise ValueError(
                "Invalid value for argument `aggregation`. Expected "
                "one of `None`, `'none'`, `'mean'`, `'sum'`, "
                "`'only_first_replica'`. "
                f"Received: aggregation={aggregation}"
            )
        if aggregation is None:
            aggregation = "none"
        if synchronization not in (
            None,
            "none",
            "on_read",
            "on_write",
            "auto",
        ):
            raise ValueError(
                "Invalid value for argument `synchronization`. Expected "
                "one of `None`, `'none'`, `'on_read'`, `'on_write'`, "
                "`'auto'`. "
                f"Received: synchronization={synchronization}"
            )
        if synchronization is None:
            synchronization = "none"
        if layout is not None:
            from keras.src.distribution.distribution_lib import TensorLayout

            if isinstance(layout, TensorLayout) and layout.device_mesh is None:
                raise ValueError(
                    "Argument `layout` must have `device_mesh` set. "
                    f"Received: layout={layout}"
                )

        self._name = name
        parent_path = current_path()
        if parent_path:
            self._path = f"{parent_path}/{name}"
        else:
            self._path = name
        self._shape = None
        self._initializer = None
        self._regularizer = None
        self._constraint = None
        self._trainable = bool(trainable)
        self._autocast = bool(autocast)
        self._aggregation = aggregation
        self._synchronization = synchronization
        self._layout = layout

        # `self._overwrite_with_gradient` is an internal property to determine
        # whether this variable should be overwritten by the computed gradient.
        # Ref: https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py
        self._overwrite_with_gradient = False
        if isinstance(initializer, str):
            from keras.src import initializers

            initializer = initializers.get(initializer)
        if callable(initializer):
            if shape is None:
                raise ValueError(
                    "When creating a Variable from an initializer, "
                    "the `shape` argument should be specified. "
                    f"Received: initializer={initializer} "
                    f"and shape={shape}"
                )
        else:
            initializer = self._convert_to_tensor(initializer, dtype=dtype)
            # If dtype is None and `initializer` is an array, use its dtype.
            if dtype is None:
                dtype = initializer.dtype
        self._dtype = standardize_dtype(dtype)

        if in_stateless_scope():
            if callable(initializer):
                self._value = None
                self._initializer = initializer
                self._shape = self._validate_shape(shape)
                register_uninitialized_variable(self)
            else:
                raise ValueError(
                    "You are attempting to create a variable "
                    "while in a stateless scope. This is disallowed. "
                    "Make sure that all variables are created "
                    "before you start using your layer/model objects.\n\n"
                    "In some cases, you might be seeing this error "
                    "because you need to "
                    "implement a `def build(self, input_shape)` method "
                    "on your layer/model, which will "
                    "create its variables.\n\n"
                    "In some other cases, you might be seeing this error "
                    "because you are instantiating a `Variable` and "
                    "assigning it to a layer without going through "
                    "self.add_variable()/self.add_weight(). Always prefer "
                    "using these methods "
                    "(with a `shape` and `initializer` argument)."
                )
        else:
            if callable(initializer):
                self._shape = self._validate_shape(shape)
                self._initialize_with_initializer(initializer)
            else:
                self._initialize(initializer)
                self._shape = self._validate_shape(self._value.shape)
        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            # If NNX is enabled, it's possible the variable was already
            # initialized by a concrete call. In this case, _deferred_initialize
            # returns early and does not raise an error.
            if config.is_nnx_enabled():
                return
            raise ValueError(f"Variable {self.path} is already initialized.")

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        self._initialize_with_initializer(self._initializer)
        self._initializer = None

    def _validate_shape(self, shape):
        shape = standardize_shape(shape)
        if None in shape:
            raise ValueError(
                "Shapes used to initialize variables must be "
                "fully-defined (no `None` dimensions). Received: "
                f"shape={shape} for variable path='{self.path}'"
            )
        return shape

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if self._autocast and autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self)

    @property
    def aggregation(self):
        """The strategy for aggregating this variable."""
        return self._aggregation

    @property
    def synchronization(self):
        """The strategy for synchronizing this variable."""
        return self._synchronization

    @property
    def value(self):
        """The current value of the variable (numpy array or backend tensor)."""
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            # Uninitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference / graph tracing
            # (anything else would be a bug, to be fixed.)
            return self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self._dtype)
        if not shape_equal(value.shape, self.shape):
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"variable.shape={self.shape}, "
                f"Received: value.shape={value.shape}. "
                f"Target variable: {self}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)
        return value

    def assign_add(self, value):
        return self.assign(self + value)

    def assign_sub(self, value):
        return self.assign(self - value)

    @property
    def dtype(self):
        """The data type of the variable."""
        autocast_scope = get_autocast_scope()
        if (
            self._autocast
            and autocast_scope is not None
            and is_float_dtype(self._dtype)
        ):
            dtype = autocast_scope.dtype
        else:
            dtype = self._dtype
        return standardize_dtype(dtype)

    @property
    def shape(self):
        """The shape of the variable."""
        return self._shape

    @property
    def ndim(self):
        """The number of dimensions of the variable."""
        return self._ndim

    @property
    def trainable(self):
        """Whether the variable is trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = bool(value)

    @property
    def name(self):
        """The name of the variable."""
        return self._name

    @property
    def path(self):
        """The path of the variable within the Keras model or layer."""
        return self._path

    @property
    def overwrite_with_gradient(self):
        """Whether this variable should be overwritten by the gradient.

        This property is designed for a special case where we want to overwrite
        the variable directly with its computed gradient. For example, in float8
        training, new `scale` and `amax_history` are computed as gradients, and
        we want to overwrite them directly instead of following the typical
        procedure such as gradient descent with a learning rate, gradient
        clipping and weight decaying.
        """
        return self._overwrite_with_gradient

    @overwrite_with_gradient.setter
    def overwrite_with_gradient(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "`overwrite_with_gradient` must be a boolean. "
                f"Received: {value}"
            )
        self._overwrite_with_gradient = value

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value):
        if value is not None and not callable(value):
            raise ValueError(
                "Invalid value for attribute `regularizer`. Expected a "
                "callable (such as a `keras.regularizers.Regularizer` "
                f"instance) or `None`. Received: regularizer={value}"
            )
        self._regularizer = value

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        if value is not None and not callable(value):
            raise ValueError(
                "Invalid value for attribute `constraint`. Expected a "
                "callable (such as a `keras.constraints.Constraint` instance) "
                f"or `None`. Received: constraint={value}"
            )
        self._constraint = value

    def __repr__(self):
        value = None
        if hasattr(self, "_value") and self._value is not None:
            try:
                value = backend.ops.convert_to_numpy(self._value)
            except:
                # In some cases the conversion to numpy can fail.
                pass
        value_str = f", value={value}" if value is not None else ""
        return (
            f"<Variable path={self.path}, shape={self.shape}, "
            f"dtype={self.dtype}{value_str}>"
        )

    def _initialize(self, value):
        raise NotImplementedError

    def _initialize_with_initializer(self, initializer):
        value = self._convert_to_tensor(
            initializer(self._shape, dtype=self._dtype)
        )
        self._initialize(value)

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __int__(self):
        if self.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={self.shape}"
            )
        return int(self.value)

    def __float__(self):
        if self.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={self.shape}"
            )
        return float(self.value)

    def __array__(self, dtype=None):
        # We can't directly use self.value.__array__ here because of scalar.
        # Numpy require this method to return as array like object. In the case
        # of scalar, it will fail the type checking from numpy. We need to
        # return a 0d array via numpy.
        return np.asarray(self.value.__array__(dtype))

    def __bool__(self):
        raise TypeError("A Keras Variable cannot be used as a boolean.")

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, other):
        return backend.ops.numpy.equal(self.value, other)

    def __ne__(self, other):
        return backend.ops.numpy.not_equal(self.value, other)

    def __lt__(self, other):
        return backend.ops.numpy.less(self.value, other)

    def __le__(self, other):
        return backend.ops.numpy.less_equal(self.value, other)

    def __gt__(self, other):
        return backend.ops.numpy.greater(self.value, other)

    def __ge__(self, other):
        return backend.ops.numpy.greater_equal(self.value, other)

    def __add__(self, other):
        return backend.ops.numpy.add(self.value, other)

    def __radd__(self, other):
        return backend.ops.numpy.add(other, self.value)

    def __sub__(self, other):
        return backend.ops.numpy.subtract(self.value, other)

    def __rsub__(self, other):
        return backend.ops.numpy.subtract(other, self.value)

    def __mul__(self, other):
        return backend.ops.numpy.multiply(self.value, other)

    def __rmul__(self, other):
        return backend.ops.numpy.multiply(other, self.value)

    def __truediv__(self, other):
        return backend.ops.numpy.true_divide(self.value, other)

    def __rtruediv__(self, other):
        return backend.ops.numpy.true_divide(other, self.value)

    def __floordiv__(self, other):
        return backend.ops.numpy.floor_divide(self.value, other)

    def __rfloordiv__(self, other):
        return backend.ops.numpy.floor_divide(other, self.value)

    def __mod__(self, other):
        return backend.ops.numpy.mod(self.value, other)

    def __rmod__(self, other):
        return backend.ops.numpy.mod(other, self.value)

    def __pow__(self, other):
        return backend.ops.numpy.power(self.value, other)

    def __rpow__(self, other):
        return backend.ops.numpy.power(other, self.value)

    def __matmul__(self, other):
        return backend.ops.numpy.matmul(self.value, other)

    def __rmatmul__(self, other):
        return backend.ops.numpy.matmul(other, self.value)

    def __and__(self, other):
        return backend.ops.numpy.logical_and(self.value, other)

    def __rand__(self, other):
        return backend.ops.numpy.logical_and(other, self.value)

    def __or__(self, other):
        return backend.ops.numpy.logical_or(self.value, other)

    def __ror__(self, other):
        return backend.ops.numpy.logical_or(other, self.value)

    def __xor__(self, other):
        return backend.ops.numpy.logical_xor(self.value, other)

    def __rxor__(self, other):
        return backend.ops.numpy.logical_xor(other, self.value)

    def __round__(self, ndigits=None):
        decimals = ndigits or 0
        return backend.ops.numpy.round(self.value, decimals=decimals)


def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute(
        "uninitialized_variables", [], set_to_default=True
    )
    uninitialized_variables.append(variable)


def initialize_all_variables():
    collection = global_state.get_global_attribute("uninitialized_variables")
    if collection:
        for v in collection:
            v._deferred_initialize()
    global_state.set_global_attribute("uninitialized_variables", [])


@keras_export(
    ["keras.utils.standardize_dtype", "keras.backend.standardize_dtype"]
)
def standardize_dtype(dtype):
    # Fast path: an already-canonical dtype string is by far the common case
    # and needs none of the normalization below.
    if isinstance(dtype, str) and dtype in dtypes.ALLOWED_DTYPES_SET:
        return dtype
    if dtype is None:
        return config.floatx()
    dtype = dtypes.PYTHON_DTYPES_MAP.get(dtype, dtype)
    if hasattr(dtype, "name"):
        dtype = dtype.name.lower()
    elif hasattr(dtype, "__name__"):
        dtype = dtype.__name__
    elif hasattr(dtype, "__str__") and (
        "torch" in str(dtype)
        or "jax.numpy" in str(dtype)
        or "mlx" in str(dtype)
    ):
        dtype = str(dtype).split(".")[-1]

    if dtype not in dtypes.ALLOWED_DTYPES_SET:
        raise ValueError(f"Invalid dtype: {dtype}")
    return dtype


def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Undefined shapes are not supported.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        if config.backend() == "tensorflow":
            if isinstance(shape, tf.TensorShape):
                if shape.rank is None:
                    raise ValueError(
                        "Tensors with unknown rank are not supported. "
                        f"Received a tensor with shape: {shape}. This "
                        "commonly happens with `tf.numpy_function` or "
                        "`tf.py_function` in a `tf.data` pipeline, which "
                        "discard static shape information. Specify the "
                        "shape explicitly, e.g. with `tensor.set_shape(...)` "
                        "or `tf.ensure_shape(tensor, shape)`."
                    )
                # `tf.TensorShape` may contain `Dimension` objects.
                # We need to convert the items in it to either int or `None`
                shape = shape.as_list()

    if config.backend() == "jax":
        # Replace `_DimExpr` (dimension expression) with None
        from jax import export as jax_export

        shape = tuple(
            None if jax_export.is_symbolic_dim(d) else d for d in shape
        )

    if config.backend() == "torch":
        # Replace symbolic dimensions with None to preserve dynamic shapes
        # during torch.export tracing
        import torch

        shape = tuple(None if isinstance(d, torch.SymInt) else d for d in shape)

    # Handle dimensions that are not ints and not None, verify they're >= 0.
    standardized_shape = []
    for d in shape:
        if d is None:
            standardized_shape.append(d)
            continue

        # Reject these even if they can be cast to int successfully.
        if isinstance(d, (str, float)):
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid dimension '{d}' of type '{type(d)}'. "
            )

        try:
            # Cast numpy scalars, tf constant tensors, etc.
            d = int(d)
        except Exception as e:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid dimension '{d}' of type '{type(d)}'. "
            ) from e
        if d < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                "Negative dimensions are not allowed."
            )
        standardized_shape.append(d)

    # This also turns subclasses of `tuple` (e.g. `torch.Size`) to plain tuple.
    return tuple(standardized_shape)


def shape_equal(a_shape, b_shape):
    """Return whether a_shape == b_shape (allows None entries)."""
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is not None and e2 is not None and e1 != e2:
            return False
    return True


@keras_export("keras.backend.is_float_dtype")
def is_float_dtype(dtype):
    dtype = standardize_dtype(dtype)
    return dtype.startswith("float") or dtype.startswith("bfloat")


@keras_export("keras.backend.is_int_dtype")
def is_int_dtype(dtype):
    dtype = standardize_dtype(dtype)
    return dtype.startswith("int") or dtype.startswith("uint")


def get_autocast_scope():
    return global_state.get_global_attribute("autocast_scope")


class AutocastScope:
    """Context manager that enables the autocasting of float variables.

    Under this context manager, float `Variables`s will be cast to `dtype`
    (note that `dtype` must also be float).
    """

    def __init__(self, dtype):
        if dtype is not None:
            dtype = standardize_dtype(dtype)
            if not is_float_dtype(dtype):
                raise ValueError(
                    "`AutocastScope` can only be used with "
                    "a floating-point target dtype, such as 'float16'. "
                    f"Received: dtype={dtype}"
                )
        self.dtype = dtype
        self.original_scope = None

    def maybe_cast(self, value):
        from keras.src import backend

        if self.dtype is not None and is_float_dtype(value.dtype):
            return backend.ops.cast(value, dtype=self.dtype)
        return value

    def __enter__(self):
        self.original_scope = get_autocast_scope()
        global_state.set_global_attribute("autocast_scope", self)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("autocast_scope", self.original_scope)
