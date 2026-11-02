from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.merging.base_merge import Merge
from keras.src.utils.numerical_utils import normalize


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    Shape inference:

    Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
    If `axes` is (1, 2), to find the output shape of resultant tensor,
    loop through each dimension in `x`'s shape and `y`'s shape:

    * `x.shape[0]` : 100 : append to output shape
    * `x.shape[1]` : 20 : do not append to output shape, dimension 1 of
        `x` has been summed over. (`dot_axes[0]` = 1)
    * `y.shape[0]` : 100 : do not append to output shape, always ignore
        first dimension of `y`
    * `y.shape[1]` : 30 : append to output shape
    * `y.shape[2]` : 20 : do not append to output shape, dimension 2 of
        `y` has been summed over.
        (`dot_axes[1]` = 2) `output_shape` = `(100, 30)`

    Example:

    >>> x_batch = np.ones(shape=(32, 20, 1))
    >>> y_batch = np.ones(shape=(32, 30, 20))
    >>> xy_batch_dot = batch_dot(x_batch, y_batch, axes=(1, 2))

    Args:
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: Tuple or list of integers with target dimensions, or single
            integer. The sizes of `x.shape[axes[0]]` and `y.shape[axes[1]]`
            should be equal.
            Note that axis `0` (the batch axis) cannot be included.

    Returns:
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape (less the
        batch dimension and the dimension that was summed over). If the final
        rank is 1, we reshape it to `(batch_size, 1)`.
    """

    x_shape = x.shape
    y_shape = y.shape

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim < 2 or y_ndim < 2:
        raise ValueError(
            f"Cannot do batch_dot on inputs "
            f"with rank < 2. "
            f"Received inputs with shapes "
            f"{x_shape} and {y_shape}."
        )

    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError(
                f"Cannot do batch_dot on inputs "
                f"with different batch sizes. "
                f"Received inputs with shapes "
                f"{x_shape} and {y_shape}."
            )
    if isinstance(axes, int):
        axes = [axes, axes]

    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]

    if any(isinstance(a, (list, tuple)) for a in axes):
        raise ValueError(
            f"Multiple target dimensions are not supported. "
            f"Expected: None, int, (int, int), "
            f"Provided: {axes} "
        )

    # if tuple, convert to list.
    axes = list(axes)

    # convert negative indices.
    if axes[0] < 0:
        axes[0] += x_ndim
    if axes[1] < 0:
        axes[1] += y_ndim

    # sanity checks
    if 0 in axes:
        raise ValueError(
            "Cannot perform batch_dot over axis 0. "
            "If your inputs are not batched, "
            "add a dummy batch dimension to your "
            "inputs using keras.ops.expand_dims(x, 0)"
        )
    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]

    if d1 is not None and d2 is not None and d1 != d2:
        raise ValueError(
            f"Cannot do batch_dot on inputs with shapes "
            f"{x_shape} and {y_shape} with axes={axes}. "
            f"x.shape[{axes[0]}] != y.shape[{axes[1]}] ({d1} != {d2})."
        )

    # backup ndims. Need them later.
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    # if rank is 2, expand to 3.
    if x_ndim == 2:
        x = backend.ops.numpy.expand_dims(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = backend.ops.numpy.expand_dims(y, 2)
        y_ndim += 1

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = backend.ops.numpy.transpose(x, pattern)

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y = backend.ops.numpy.transpose(y, pattern)

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape = backend.ops.shape(x)
        x_mid_dims = x_shape[1:-1]
        x_squashed_shape = (x_shape[0], -1, x_shape[-1])
        x = backend.ops.numpy.reshape(x, x_squashed_shape)
        x_squashed = True
    else:
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape = backend.ops.shape(y)
        y_trail_dims = y_shape[2:]
        y_squashed_shape = (y_shape[0], y_shape[1], -1)
        y = backend.ops.numpy.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False

    result = backend.ops.numpy.matmul(x, y)

    # if inputs were squashed, we have to reshape the matmul output.
    output_shape = backend.ops.shape(result)
    do_reshape = False

    if x_squashed:
        output_shape = output_shape[:1] + x_mid_dims + output_shape[-1:]
        do_reshape = True

    if y_squashed:
        output_shape = output_shape[:-1] + y_trail_dims
        do_reshape = True

    if do_reshape:
        result = backend.ops.numpy.reshape(result, output_shape)

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        result = backend.ops.numpy.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = backend.ops.numpy.squeeze(result, -1)

    return result


@keras_export("keras.layers.Dot")
class Dot(Merge):
    """Computes element-wise dot product of two tensors.

    It takes a list of inputs of size 2, and the axes
    corresponding to each input along with the dot product
    is to be performed.

    Let's say `x` and `y` are the two input tensors with shapes
    `(2, 3, 5)` and `(2, 10, 3)`. The batch dimension should be
    of same size for both the inputs, and `axes` should correspond
    to the dimensions that have the same size in the corresponding
    inputs. e.g. with `axes=(1, 2)`, the dot product of `x`, and `y`
    will result in a tensor with shape `(2, 5, 10)`

    Example:

    >>> x = np.arange(10).reshape(1, 5, 2)
    >>> y = np.arange(10, 20).reshape(1, 2, 5)
    >>> keras.layers.Dot(axes=(1, 2))([x, y])

    Usage in a Keras model:

    >>> x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> y = keras.layers.Dot(axes=1)([x1, x2])

    Args:
        axes: Integer or tuple of integers, axis or axes along which to
            take the dot product. If a tuple, should be two integers
            corresponding to the desired axis from the first input and the
            desired axis from the second input, respectively. Note that the
            size of the two selected axes must match, and that
            axis `0` (the batch axis) cannot be included.
        normalize: Whether to L2-normalize samples along the dot product axis
            before taking the dot product. If set to `True`, then
            the output of the dot product is the cosine proximity
            between the two samples.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the dot product of the samples from the inputs.
    """

    def __init__(self, axes, normalize=False, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError(
                    f"Invalid type for argument `axes`: it should be "
                    f"a list or an int. Received: axes={axes}"
                )
            if len(axes) != 2:
                raise ValueError(
                    f"Invalid format for argument `axes`: it should contain "
                    f"two elements. Received: axes={axes}"
                )
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError(
                    f"Invalid format for argument `axes`: list elements should "
                    f"be integers. Received: axes={axes}"
                )
        self.axes = axes
        self.normalize = normalize
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if (
            not isinstance(input_shape[0], (tuple, list))
            or len(input_shape) != 2
        ):
            raise ValueError(
                f"A `Dot` layer should be called on a list of 2 inputs. "
                f"Received: input_shape={input_shape}"
            )
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                f"Incompatible input shapes: "
                f"axis values {shape1[axes[0]]} (at axis {axes[0]}) != "
                f"{shape2[axes[1]]} (at axis {axes[1]}). "
                f"Full input shapes: {shape1}, {shape2}"
            )

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                f"A `Dot` layer should be called on exactly 2 inputs. "
                f"Received: inputs={inputs}"
            )
        x1 = inputs[0]
        x2 = inputs[1]

        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [
                    self.axes % len(x1.shape),
                    self.axes % len(x2.shape),
                ]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % len(inputs[i].shape))
                else:
                    axes.append(self.axes[i])

        if self.normalize:
            x1 = normalize(x1, axis=axes[0])
            x2 = normalize(x2, axis=axes[1])
        output = batch_dot(x1, x2, axes)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                f"A `Dot` layer should be called on a list of 2 inputs. "
                f"Received: input_shape={input_shape}"
            )
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            "axes": self.axes,
            "normalize": self.normalize,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.layers.dot")
def dot(inputs, axes=-1, **kwargs):
    """Functional interface to the `Dot` layer.

    Args:
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
            Note that axis `0` (the batch axis) cannot be included.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to `True`, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the dot product of the samples from the inputs.
    """
    return Dot(axes=axes, **kwargs)(inputs)
