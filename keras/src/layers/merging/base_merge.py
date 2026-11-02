from keras.src import backend
from keras.src import ops
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import set_keras_mask
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.layer import Layer


class Merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    Args:
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _apply_merge_op_and_or_mask(self, op_fn, inputs):
        """Merge a set of inputs by applying `op_fn` and ORing the masks.

        We use this for `Minimum` and `Maximum` as it handles the fact that
        there is no identity element. If applicable, the mask obtained by ORing
        all masks is set on the output.

        Args:
            op_fn: binary operation to apply to tensor pair.
            inputs: array of tensors to apply operation on.
        """
        output = None
        output_mask = None

        for x in inputs:
            mask = get_keras_mask(x)
            if mask is not None:
                mask = backend.ops.numpy.broadcast_to(
                    backend.ops.numpy.expand_dims(mask, -1),
                    backend.ops.shape(x),
                )
            if output is None:
                output = x
                output_mask = mask
                continue
            if mask is not None:
                x = backend.ops.numpy.where(mask, x, output)
            if output_mask is not None:
                output = backend.ops.numpy.where(output_mask, output, x)
            if mask is not None and output_mask is not None:
                output_mask = backend.ops.numpy.logical_or(output_mask, mask)
            else:
                output_mask = None
            output = op_fn(output, x)

        if output_mask is not None:
            output_mask = backend.ops.numpy.any(
                output_mask, axis=-1, keepdims=False
            )
            set_keras_mask(output, output_mask)
        return output

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        Args:
            shape1: Tuple or None. Shape of the first tensor
            shape2: Tuple or None. Shape of the second tensor

        Returns:
            Expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        Raises:
            ValueError: If shape1 and shape2 are not compatible for
                element-wise operations.
        """

        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[: -len(shape2)])
        for i, j in zip(shape1[-len(shape2) :], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError(
                        "Inputs have incompatible shapes. "
                        f"Received shapes {shape1} and {shape2}"
                    )
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], (tuple, list)):
            raise ValueError(
                "A merge layer should be called on a list of inputs. "
                f"Received: input_shape={input_shape} (not a list of shapes)"
            )
        if len(input_shape) < 1:
            raise ValueError(
                "A merge layer should be called "
                "on a list of at least 1 input. "
                f"Received {len(input_shape)} inputs. "
                f"Full input_shape received: {input_shape}"
            )

        batch_sizes = {s[0] for s in input_shape if s} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                "Cannot merge tensors with different batch sizes. "
                f"Received tensors with shapes {input_shape}"
            )

        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]

        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape
            )

        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "A merge layer should be called on a list of inputs. "
                f"Received: inputs={inputs} (not a list of tensors)"
            )
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(backend.ops.numpy.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = backend.ops.numpy.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = backend.ops.numpy.expand_dims(x, axis=1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... ,
                # batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = backend.ops.numpy.ndim(x)

                    if x_ndim is None:
                        x_shape = backend.ops.shape(x)
                        batch_size = x_shape[0]

                        new_shape = backend.ops.numpy.concatenate(
                            [
                                x_shape[1:],
                                backend.ops.numpy.expand_dims(
                                    batch_size, axis=-1
                                ),
                            ]
                        )
                        x_transposed = backend.ops.numpy.reshape(
                            x,
                            backend.ops.numpy.stack(
                                [
                                    batch_size,
                                    backend.ops.numpy.prod(x_shape[1:]),
                                ],
                                axis=0,
                            ),
                        )
                        x_transposed = backend.ops.numpy.transpose(
                            x_transposed, perm=(1, 0)
                        )
                        x_transposed = backend.ops.numpy.reshape(
                            x_transposed, new_shape
                        )

                        reshaped_inputs.append(x_transposed)
                        transposed = True

                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(
                            backend.ops.numpy.transpose(x, perm=dims)
                        )
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or
                        # scalars.
                        reshaped_inputs.append(x)

                y = self._merge_function(reshaped_inputs)
                y_ndim = backend.ops.numpy.ndim(y)

                if transposed:
                    # If inputs have been transposed, we have to transpose the
                    # output too.
                    if y_ndim is None:
                        y_shape = backend.ops.shape(y)
                        y_ndim = backend.ops.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = backend.ops.numpy.concatenate(
                            [
                                backend.ops.numpy.expand_dims(
                                    batch_size, axis=-1
                                ),
                                y_shape[: y_ndim - 1],
                            ]
                        )
                        y = backend.ops.numpy.reshape(y, (-1, batch_size))
                        y = backend.ops.numpy.transpose(y, perm=(1, 0))
                        y = backend.ops.numpy.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = backend.ops.numpy.transpose(y, perm=dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]

        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape
            )
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape([x.shape for x in inputs])
        output_sparse = all(x.sparse for x in inputs)
        return KerasTensor(
            output_shape, dtype=self.compute_dtype, sparse=output_sparse
        )

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, (tuple, list)):
            raise ValueError(f"`mask` should be a list. Received: mask={mask}")
        if not isinstance(inputs, (tuple, list)):
            raise ValueError(
                f"`inputs` should be a list. Received: inputs={inputs}"
            )
        if len(mask) != len(inputs):
            raise ValueError(
                "The lists `inputs` and `mask` should have the same length. "
                f"Received: inputs={inputs} of length {len(inputs)}, and "
                f"mask={mask} of length {len(mask)}"
            )
        # Default implementation does an OR between the masks, which works
        # for `Add`, `Subtract`, `Average`, `Maximum`, `Minimum`, `Multiply`.
        if any(m is None for m in mask):
            return None
        output_mask = mask[0]
        for m in mask[1:]:
            output_mask = ops.logical_or(output_mask, m)
        return output_mask

    def get_config(self):
        return super().get_config()
