import warnings

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common import standardize_shape
from keras.src.layers.layer import Layer
from keras.src.ops.node import Node


@keras_export("keras.layers.InputLayer")
class InputLayer(Layer):
    def __init__(
        self,
        shape=None,
        batch_size=None,
        dtype=None,
        sparse=None,
        ragged=None,
        batch_shape=None,
        input_tensor=None,
        optional=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name)

        if "input_shape" in kwargs:
            warnings.warn(
                "Argument `input_shape` is deprecated. Use `shape` instead."
            )
            shape = kwargs.pop("input_shape")
        if "batch_input_shape" in kwargs:
            batch_shape = kwargs.pop("batch_input_shape")

        if input_tensor is not None:
            if not isinstance(input_tensor, backend.KerasTensor):
                raise ValueError(
                    "Argument `input_tensor` must be a KerasTensor. "
                    f"Received invalid type: input_tensor={input_tensor} "
                    f"(of type {type(input_tensor)})"
                )
            if batch_size is not None:
                if (
                    len(input_tensor.shape) < 1
                    or input_tensor.shape[0] != batch_size
                ):
                    raise ValueError(
                        "When providing the `input_tensor` argument, you "
                        "cannot provide an incompatible `batch_size` argument."
                    )
            if shape is not None:
                if (
                    len(shape) != len(input_tensor.shape) - 1
                    or shape != input_tensor.shape[1:]
                ):
                    raise ValueError(
                        "When providing the `input_tensor` argument, you "
                        "cannot provide an incompatible `shape` argument."
                    )
            if batch_shape is not None and batch_shape != input_tensor.shape:
                raise ValueError(
                    "When providing the `input_tensor` argument, you "
                    "cannot provide an incompatible `batch_shape` argument."
                )
            if dtype is not None and input_tensor.dtype != dtype:
                raise ValueError(
                    "When providing the `input_tensor` argument, you "
                    "cannot provide an incompatible `dtype` argument."
                )
            if sparse is not None and input_tensor.sparse != sparse:
                raise ValueError(
                    "When providing the `input_tensor` argument, you "
                    "cannot provide an incompatible `sparse` argument."
                )
            batch_shape = input_tensor.shape
            dtype = input_tensor.dtype
            sparse = input_tensor.sparse
        else:
            if shape is not None and batch_shape is not None:
                raise ValueError(
                    "You cannot pass both `shape` and `batch_shape` at the "
                    "same time."
                )
            if batch_size is not None and batch_shape is not None:
                raise ValueError(
                    "You cannot pass both `batch_size` and `batch_shape` "
                    "at the same time."
                )
            if shape is None and batch_shape is None:
                raise ValueError("You must pass a `shape` argument.")

            if shape is not None:
                shape = standardize_shape(shape)
                batch_shape = (batch_size,) + shape

        self._batch_shape = standardize_shape(batch_shape)
        self._dtype = standardize_dtype(dtype)
        self.sparse = bool(sparse)
        if self.sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            raise ValueError(
                f"`sparse=True` is not supported with the {backend.backend()} "
                "backend"
            )
        self.ragged = bool(ragged)
        if self.ragged and not backend.SUPPORTS_RAGGED_TENSORS:
            raise ValueError(
                f"`ragged=True` is not supported with the {backend.backend()} "
                "backend"
            )

        if input_tensor is None:
            input_tensor = backend.KerasTensor(
                shape=batch_shape,
                dtype=dtype,
                sparse=sparse,
                ragged=ragged,
                name=name,
            )
        self._input_tensor = input_tensor
        Node(operation=self, call_args=(), call_kwargs={}, outputs=input_tensor)
        self.built = True
        self.optional = optional

    def call(self):
        return

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def dtype(self):
        return self._dtype

    def get_config(self):
        return {
            "batch_shape": self.batch_shape,
            "dtype": self.dtype,
            "sparse": self.sparse,
            "ragged": self.ragged,
            "name": self.name,
            "optional": self.optional,
        }


@keras_export(["keras.layers.Input", "keras.Input"])
def Input(
    shape=None,
    batch_size=None,
    dtype=None,
    sparse=None,
    ragged=None,
    batch_shape=None,
    name=None,
    tensor=None,
    optional=False,
):
    """Used to instantiate a Keras tensor.

    A Keras tensor is a symbolic tensor-like object, which we augment with
    certain attributes that allow us to build a Keras model just by knowing the
    inputs and outputs of the model.

    For instance, if `a`, `b` and `c` are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    Args:
        shape: A shape tuple (tuple of integers or `None` objects),
            not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors. Elements of this tuple
            can be `None`; `None` elements represent dimensions where the shape
            is not known and may vary (e.g. sequence length).
        batch_size: Optional static batch size (integer).
        dtype: The data type expected by the input, as a string
            (e.g. `"float32"`, `"int32"`...)
        sparse: A boolean specifying whether the expected input will be sparse
            tensors. Note that, if `sparse` is `False`, sparse tensors can still
            be passed into the input - they will be densified with a default
            value of 0. This feature is only supported with the TensorFlow and
            the JAX backends. Defaults to `False`.
        ragged: A boolean specifying whether the expected input will be ragged
            tensors. Note that, if `ragged` is `False`, ragged tensors can still
            be passed into the input - they will be densified with a default
            value of 0. This feature is only supported with the TensorFlow
            backend. Defaults to `False`.
        batch_shape: Optional shape tuple (tuple of integers or `None` objects),
            including the batch size.
        name: Optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will use this tensor rather
            than creating a new placeholder tensor.
        optional: Boolean, whether the input is optional or not.
            An optional input can accept `None` values.

    Returns:
      A Keras tensor.

    Example:

    ```python
    # This is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    """
    layer = InputLayer(
        shape=shape,
        batch_size=batch_size,
        dtype=dtype,
        sparse=sparse,
        ragged=ragged,
        batch_shape=batch_shape,
        name=name,
        input_tensor=tensor,
        optional=optional,
    )
    return layer.output
