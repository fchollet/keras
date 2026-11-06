from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name


@keras_export("keras.KerasTensor")
class KerasTensor:
    """Symbolic tensor -- encapsulates a shape and a dtype.

    You can use `KerasTensor` instances to build computation
    graphs of Keras operations, such as `keras.Function`
    objects or Functional `keras.models.Model` objects.

    Example:

    >>> x = keras.KerasTensor(shape=(3, 4), dtype="float32")
    >>> x.shape
    (3, 4)
    >>> x.dtype
    float32

    Calling a Keras operation (including a layer or a model)
    on a `KerasTensor` instance will return another `KerasTensor`
    instance with the appropriate shape and dtype. This is
    called a "symbolic call" (since there is no actual data
    involved). The computation of the correct output shape and
    dtype is called "static shape inference".
    """

    def __init__(
        self,
        shape,
        dtype="float32",
        sparse=False,
        ragged=False,
        record_history=True,
        name=None,
        **kwargs,
    ):
        from keras.src.backend.common import standardize_dtype
        from keras.src.backend.common import standardize_shape

        ragged_rank = kwargs.pop("ragged_rank", None)
        row_splits_dtype = kwargs.pop("row_splits_dtype", None)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments: {', '.join(kwargs.keys())}"
            )

        self._shape = standardize_shape(shape)
        self._dtype = standardize_dtype(dtype)
        self._sparse = bool(sparse)
        self._ragged = bool(ragged)
        if self._sparse and self._ragged:
            raise ValueError(
                "KerasTensor cannot have `sparse=True` and `ragged=True` at "
                "the same time."
            )
        self._ragged_rank = (
            int(ragged_rank) if ragged_rank is not None else None
        )
        self._row_splits_dtype = (
            standardize_dtype(row_splits_dtype)
            if row_splits_dtype is not None
            else None
        )
        self.name = name or auto_name(self.__class__.__name__)
        self.record_history = record_history

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        raise AttributeError(
            "The `shape` attribute of KerasTensor is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        raise AttributeError(
            "The `dtype` attribute of KerasTensor is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        raise AttributeError(
            "The `sparse` attribute of KerasTensor is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def ragged_rank(self):
        return self._ragged_rank

    @ragged_rank.setter
    def ragged_rank(self, value):
        raise AttributeError(
            "The `ragged_rank` attribute of KerasTensor is immutable. One "
            "should create a new instance of KerasTensor for this."
        )

    @property
    def row_splits_dtype(self):
        return self._row_splits_dtype

    @row_splits_dtype.setter
    def row_splits_dtype(self, value):
        raise AttributeError(
            "The `row_splits_dtype` attribute of KerasTensor is immutable. One "
            "should create a new instance of KerasTensor for this."
        )

    @property
    def ragged(self):
        return self._ragged

    @ragged.setter
    def ragged(self, value):
        raise AttributeError(
            "The `ragged` attribute of KerasTensor is immutable. One should "
            "create a new instance of KerasTensor for this."
        )

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, newshape):
        from keras.src import ops

        return ops.Reshape(newshape)(self)

    def squeeze(self, axis=None):
        from keras.src import ops

        return ops.Squeeze(axis)(self)

    def __int__(self):
        raise ValueError(
            "A KerasTensor is symbolic: it's a placeholder for a shape "
            "an a dtype. It doesn't have any actual numerical value. "
            "You cannot convert it to an int."
        )

    def __float__(self):
        raise ValueError(
            "A KerasTensor is symbolic: it's a placeholder for a shape "
            "an a dtype. It doesn't have any actual numerical value. "
            "You cannot convert it to a float."
        )

    def __array__(self):
        raise ValueError(
            "A KerasTensor is symbolic: it's a placeholder for a shape "
            "an a dtype. It doesn't have any actual numerical value. "
            "You cannot convert it to a NumPy array."
        )

    def __jax_array__(self):
        raise ValueError(
            "A KerasTensor cannot be used as input to a JAX function. "
            "A KerasTensor is a symbolic placeholder for a shape and dtype, "
            "used when constructing Keras Functional models "
            "or Keras Functions. You can only use it as input to a Keras layer "
            "or a Keras operation (from the namespaces `keras.layers` "
            "and `keras.ops`). "
            "You are likely doing something like:\n\n"
            "```\n"
            "x = Input(...)\n"
            "...\n"
            "jax_fn(x)  # Invalid.\n"
            "```\n\n"
            "What you should do instead is wrap `jax_fn` in a layer:\n\n"
            "```\n"
            "class MyLayer(Layer):\n"
            "    def call(self, x):\n"
            "        return jax_fn(x)\n\n"
            "x = MyLayer()(x)\n"
            "```\n"
        )

    def __tf_tensor__(self, dtype=None, name=None):
        raise ValueError(
            "A KerasTensor cannot be used as input to a TensorFlow function. "
            "A KerasTensor is a symbolic placeholder for a shape and dtype, "
            "used when constructing Keras Functional models "
            "or Keras Functions. You can only use it as input to a Keras layer "
            "or a Keras operation (from the namespaces `keras.layers` "
            "and `keras.ops`). "
            "You are likely doing something like:\n\n"
            "```\n"
            "x = Input(...)\n"
            "...\n"
            "tf_fn(x)  # Invalid.\n"
            "```\n\n"
            "What you should do instead is wrap `tf_fn` in a layer:\n\n"
            "```\n"
            "class MyLayer(Layer):\n"
            "    def call(self, x):\n"
            "        return tf_fn(x)\n\n"
            "x = MyLayer()(x)\n"
            "```\n"
        )

    def __repr__(self):
        return (
            f"<KerasTensor shape={self.shape}, dtype={self.dtype}, "
            f"sparse={self.sparse}, ragged={self.ragged}, name={self.name}>"
        )

    def __iter__(self):
        raise NotImplementedError(
            "Iterating over a symbolic KerasTensor is not supported."
        )

    def __bool__(self):
        raise TypeError("A symbolic KerasTensor cannot be used as a boolean.")

    def __add__(self, other):
        from keras.src import ops

        return ops.Add().symbolic_call(self, other)

    def __radd__(self, other):
        from keras.src import ops

        return ops.Add().symbolic_call(other, self)

    def __sub__(self, other):
        from keras.src import ops

        return ops.Subtract().symbolic_call(self, other)

    def __rsub__(self, other):
        from keras.src import ops

        return ops.Subtract().symbolic_call(other, self)

    def __mul__(self, other):
        from keras.src import ops

        return ops.Multiply().symbolic_call(self, other)

    def __rmul__(self, other):
        from keras.src import ops

        return ops.Multiply().symbolic_call(other, self)

    def __matmul__(self, other):
        from keras.src import ops

        return ops.Matmul().symbolic_call(self, other)

    def __rmatmul__(self, other):
        from keras.src import ops

        return ops.Matmul().symbolic_call(other, self)

    def __div__(self, other):
        from keras.src import ops

        return ops.Divide().symbolic_call(self, other)

    def __rdiv__(self, other):
        from keras.src import ops

        return ops.Divide().symbolic_call(other, self)

    def __truediv__(self, other):
        from keras.src import ops

        return ops.TrueDivide().symbolic_call(self, other)

    def __rtruediv__(self, other):
        from keras.src import ops

        return ops.TrueDivide().symbolic_call(other, self)

    def __neg__(self):
        from keras.src import ops

        return ops.Negative().symbolic_call(self)

    def __abs__(self):
        from keras.src import ops

        return ops.Absolute().symbolic_call(self)

    def __pow__(self, other):
        from keras.src import ops

        return ops.Power().symbolic_call(self, other)

    def __rpow__(self, other):
        from keras.src import ops

        return ops.Power().symbolic_call(other, self)

    def __floordiv__(self, other):
        from keras.src import ops

        return ops.FloorDivide().symbolic_call(self, other)

    def __rfloordiv__(self, other):
        from keras.src import ops

        return ops.FloorDivide().symbolic_call(other, self)

    def __mod__(self, other):
        from keras.src import ops

        return ops.Mod().symbolic_call(self, other)

    def __rmod__(self, other):
        from keras.src import ops

        return ops.Mod().symbolic_call(other, self)

    def __lt__(self, other):
        from keras.src import ops

        return ops.Less().symbolic_call(self, other)

    def __le__(self, other):
        from keras.src import ops

        return ops.LessEqual().symbolic_call(self, other)

    def __gt__(self, other):
        from keras.src import ops

        return ops.Greater().symbolic_call(self, other)

    def __ge__(self, other):
        from keras.src import ops

        return ops.GreaterEqual().symbolic_call(self, other)

    def __ne__(self, other):
        from keras.src import ops

        return ops.NotEqual().symbolic_call(self, other)

    def __and__(self, other):
        from keras.src import ops

        return ops.LogicalAnd().symbolic_call(self, other)

    def __rand__(self, other):
        from keras.src import ops

        return ops.LogicalAnd().symbolic_call(other, self)

    def __or__(self, other):
        from keras.src import ops

        return ops.LogicalOr().symbolic_call(self, other)

    def __ror__(self, other):
        from keras.src import ops

        return ops.LogicalOr().symbolic_call(other, self)

    def __invert__(self):
        from keras.src import ops

        return ops.LogicalNot().symbolic_call(self)

    def __xor__(self, other):
        from keras.src import ops

        return ops.LogicalXor().symbolic_call(self, other)

    def __rxor__(self, other):
        from keras.src import ops

        return ops.LogicalXor().symbolic_call(other, self)

    def __getitem__(self, key):
        from keras.src import ops

        return ops.GetItem().symbolic_call(self, key)

    def __round__(self, ndigits=None):
        from keras.src import ops

        decimals = ndigits or 0
        return ops.Round(decimals=decimals).symbolic_call(self)


def any_symbolic_tensors(args=None, kwargs=None):
    for items in (args or (), kwargs.values() if kwargs else ()):
        for x in items:
            if isinstance(x, KerasTensor):
                return True
            if tree.is_nested(x):
                for nested in tree.flatten(x):
                    if isinstance(nested, KerasTensor):
                        return True
    return False


@keras_export(["keras.utils.is_keras_tensor", "keras.backend.is_keras_tensor"])
def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a *symbolic tensor*, such as a tensor
    that was created via `Input()`. A "symbolic tensor"
    can be understood as a placeholder -- it does not
    contain any actual numerical data, only a shape and dtype.
    It can be used for building Functional models, but it
    cannot be used in actual computations.
    """
    return isinstance(x, KerasTensor)
