"""Legacy Keras 1/2 backend functions."""

import itertools
import math

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.module_utils import tensorflow as tf

py_any = any
py_all = all


@keras_export("keras._legacy.backend.abs")
def abs(x):
    """DEPRECATED."""
    return tf.abs(x)


@keras_export("keras._legacy.backend.all")
def all(x, axis=None, keepdims=False):
    """DEPRECATED."""
    x = tf.cast(x, tf.bool)
    return tf.reduce_all(x, axis, keepdims)


@keras_export("keras._legacy.backend.any")
def any(x, axis=None, keepdims=False):
    """DEPRECATED."""
    x = tf.cast(x, tf.bool)
    return tf.reduce_any(x, axis, keepdims)


@keras_export("keras._legacy.backend.argmax")
def argmax(x, axis=-1):
    """DEPRECATED."""
    return tf.argmax(x, axis)


@keras_export("keras._legacy.backend.argmin")
def argmin(x, axis=-1):
    """DEPRECATED."""
    return tf.argmin(x, axis)


@keras_export("keras._legacy.backend.arange")
def arange(start, stop=None, step=1, dtype="int32"):
    """DEPRECATED."""
    if stop is None and start < 0:
        start = 0
    result = tf.range(start, limit=stop, delta=step, name="arange")
    if dtype != "int32":
        result = tf.cast(result, dtype)
    return result


@keras_export("keras._legacy.backend.batch_dot")
def batch_dot(x, y, axes=None):
    """DEPRECATED."""
    x_shape = x.shape
    y_shape = y.shape

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim < 2 or y_ndim < 2:
        raise ValueError(
            "Cannot do batch_dot on inputs "
            "with rank < 2. "
            f"Received inputs with tf.shapes {x_shape} and {y_shape}."
        )

    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError(
                "Cannot do batch_dot on inputs "
                "with different batch sizes. "
                "Received inputs with tf.shapes "
                f"{x_shape} and {y_shape}."
            )
    if isinstance(axes, int):
        axes = [axes, axes]

    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]

    if py_any(isinstance(a, (list, tuple)) for a in axes):
        raise ValueError(
            "Multiple target dimensions are not supported. "
            "Expected: None, int, (int, int), "
            f"Provided: {axes}"
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
            "inputs using K.expand_dims(x, 0)"
        )
    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]

    if d1 is not None and d2 is not None and d1 != d2:
        raise ValueError(
            "Cannot do batch_dot on inputs with tf.shapes "
            f"{x_shape} and {y_shape} with axes={axes}. "
            "x.shape[%d] != y.shape[%d] (%d != %d)."
            % (axes[0], axes[1], d1, d2)
        )

    # backup ndims. Need them later.
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    # if rank is 2, expand to 3.
    if x_ndim == 2:
        x = tf.expand_dims(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = tf.expand_dims(y, 2)
        y_ndim += 1

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = tf.transpose(x, pattern)

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y = tf.transpose(y, pattern)

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape = tf.shape(x)
        x_mid_dims = x_shape[1:-1]
        x_squashed_shape = tf.stack([x_shape[0], -1, x_shape[-1]])
        x = tf.reshape(x, x_squashed_shape)
        x_squashed = True
    else:
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape = tf.shape(y)
        y_trail_dims = y_shape[2:]
        y_squashed_shape = tf.stack([y_shape[0], y_shape[1], -1])
        y = tf.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False

    result = tf.matmul(x, y)

    # if inputs were squashed, we have to reshape the matmul output.
    output_shape = tf.shape(result)
    do_reshape = False

    if x_squashed:
        output_shape = tf.concat(
            [output_shape[:1], x_mid_dims, output_shape[-1:]], 0
        )
        do_reshape = True

    if y_squashed:
        output_shape = tf.concat([output_shape[:-1], y_trail_dims], 0)
        do_reshape = True

    if do_reshape:
        result = tf.reshape(result, output_shape)

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        result = tf.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = tf.squeeze(result, -1)

    return result


@keras_export("keras._legacy.backend.batch_flatten")
def batch_flatten(x):
    """DEPRECATED."""
    x = tf.reshape(x, tf.stack([-1, prod(tf.shape(x)[1:])]))
    return x


@keras_export("keras._legacy.backend.batch_get_value")
def batch_get_value(tensors):
    """DEPRECATED."""
    return [x.numpy() for x in tensors]


@keras_export("keras._legacy.backend.batch_set_value")
def batch_set_value(tuples):
    """DEPRECATED."""
    if tf.executing_eagerly() or tf.inside_function():
        for x, value in tuples:
            value = np.asarray(value, dtype=x.dtype.name)
            x.assign(value)


@keras_export("keras._legacy.backend.batch_normalization")
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """DEPRECATED."""
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


@keras_export("keras._legacy.backend.bias_add")
def bias_add(x, bias, data_format=None):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    bias_shape = bias.shape
    if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
        raise ValueError(
            f"Unexpected bias dimensions {len(bias_shape)}. "
            f"Expected it to be 1 or {ndim(x) - 1} dimensions"
        )

    if len(bias_shape) == 1:
        if data_format == "channels_first":
            return tf.nn.bias_add(x, bias, data_format="NCHW")
        return tf.nn.bias_add(x, bias, data_format="NHWC")
    if ndim(x) in (3, 4, 5):
        if data_format == "channels_first":
            bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
            return x + reshape(bias, bias_reshape_axis)
        return x + reshape(bias, (1,) + bias_shape)
    return tf.nn.bias_add(x, bias)


@keras_export("keras._legacy.backend.binary_crossentropy")
def binary_crossentropy(target, output, from_logits=False):
    """DEPRECATED."""
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    # Guard unknown ranks (shape rank is None during symbolic/graph mode):
    # TensorShape raises for len()/iteration when the rank is unknown.
    target_shape = (
        tuple(target.shape.as_list()) if target.shape.rank is not None else None
    )
    output_shape = (
        tuple(output.shape.as_list()) if output.shape.rank is not None else None
    )
    if target_shape is not None and output_shape is not None:
        if len(target_shape) != len(output_shape):
            raise ValueError(
                "Arguments `target` and `output` must have the same rank "
                "(ndim). Received: "
                f"target.shape={target_shape}, output.shape={output_shape}"
            )
        for e1, e2 in zip(target_shape, output_shape):
            if e1 is not None and e2 is not None and e1 != e2:
                raise ValueError(
                    "Arguments `target` and `output` must have the same "
                    "shape. Received: "
                    f"target.shape={target_shape}, output.shape={output_shape}"
                )

    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )

    epsilon_ = tf.convert_to_tensor(backend.epsilon(), output.dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = target * tf.math.log(output + backend.epsilon())
    bce += (1 - target) * tf.math.log(1 - output + backend.epsilon())
    return -bce


@keras_export("keras._legacy.backend.binary_focal_crossentropy")
def binary_focal_crossentropy(
    target,
    output,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
):
    """DEPRECATED."""
    sigmoidal = tf.sigmoid(output) if from_logits else output

    p_t = target * sigmoidal + (1 - target) * (1 - sigmoidal)

    # Calculate focal factor
    focal_factor = tf.pow(1.0 - p_t, gamma)

    # Binary crossentropy
    bce = binary_crossentropy(
        target=target,
        output=output,
        from_logits=from_logits,
    )
    focal_bce = focal_factor * bce

    if apply_class_balancing:
        weight = target * alpha + (1 - target) * (1 - alpha)
        focal_bce = weight * focal_bce

    return focal_bce


@keras_export("keras._legacy.backend.cast")
def cast(x, dtype):
    """DEPRECATED."""
    return tf.cast(x, dtype)


@keras_export("keras._legacy.backend.cast_to_floatx")
def cast_to_floatx(x):
    """DEPRECATED."""
    if isinstance(x, (tf.Tensor, tf.Variable, tf.SparseTensor)):
        return tf.cast(x, dtype=backend.floatx())
    return np.asarray(x, dtype=backend.floatx())


@keras_export("keras._legacy.backend.categorical_crossentropy")
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """DEPRECATED."""
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)

    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, True)

    # Compute cross entropy from probabilities.
    epsilon_ = tf.convert_to_tensor(backend.epsilon(), output.dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target * tf.math.log(output), axis)


@keras_export("keras._legacy.backend.categorical_focal_crossentropy")
def categorical_focal_crossentropy(
    target,
    output,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    axis=-1,
):
    """DEPRECATED."""
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)

    if from_logits:
        output = tf.nn.softmax(output, axis=axis)

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis=axis, keepdims=True)

    epsilon_ = tf.convert_to_tensor(backend.epsilon(), output.dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Calculate cross entropy
    cce = -target * tf.math.log(output)

    # Calculate factors
    modulating_factor = tf.pow(1.0 - output, gamma)
    weighting_factor = tf.multiply(modulating_factor, alpha)

    # Apply weighting factor
    focal_cce = tf.multiply(weighting_factor, cce)
    focal_cce = tf.reduce_sum(focal_cce, axis=axis)
    return focal_cce


@keras_export("keras._legacy.backend.clip")
def clip(x, min_value, max_value):
    """DEPRECATED."""
    if isinstance(min_value, (int, float)) and isinstance(
        max_value, (int, float)
    ):
        if max_value < min_value:
            max_value = min_value
    if min_value is None:
        min_value = -np.inf
    if max_value is None:
        max_value = np.inf
    return tf.clip_by_value(x, min_value, max_value)


@keras_export("keras._legacy.backend.concatenate")
def concatenate(tensors, axis=-1):
    """DEPRECATED."""
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    if py_all(is_sparse(x) for x in tensors):
        return tf.compat.v1.sparse_concat(axis, tensors)
    elif py_all(isinstance(x, tf.RaggedTensor) for x in tensors):
        return tf.concat(tensors, axis)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)


@keras_export("keras._legacy.backend.constant")
def constant(value, dtype=None, shape=None, name=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()

    return tf.constant(value, dtype=dtype, shape=shape, name=name)


def _preprocess_conv1d_input(x, data_format):
    tf_data_format = "NWC"  # to pass TF Conv2dNative operations
    if data_format == "channels_first":
        tf_data_format = "NCW"
    return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
    tf_data_format = "NHWC"
    if data_format == "channels_first":
        if force_transpose:
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = "NCHW"
    return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
    tf_data_format = "NDHWC"
    if data_format == "channels_first":
        tf_data_format = "NCDHW"
    return x, tf_data_format


def _preprocess_padding(padding):
    if padding == "same":
        padding = "SAME"
    elif padding == "valid":
        padding = "VALID"
    else:
        raise ValueError(f"Invalid padding: {padding}")
    return padding


@keras_export("keras._legacy.backend.conv1d")
def conv1d(
    x, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    kernel_shape = kernel.shape.as_list()
    if padding == "causal":
        # causal (dilated) convolution:
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = "valid"
    padding = _preprocess_padding(padding)

    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NWC":
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW
    return x


@keras_export("keras._legacy.backend.conv2d")
def conv2d(
    x,
    kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@keras_export("keras._legacy.backend.conv2d_transpose")
def conv2d_transpose(
    x,
    kernel,
    output_shape,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    # `atrous_conv2d_transpose` only supports NHWC format, even on GPU.
    if data_format == "channels_first" and dilation_rate != (1, 1):
        force_transpose = True
    else:
        force_transpose = False

    x, tf_data_format = _preprocess_conv2d_input(
        x, data_format, force_transpose
    )

    if data_format == "channels_first" and tf_data_format == "NHWC":
        output_shape = (
            output_shape[0],
            output_shape[2],
            output_shape[3],
            output_shape[1],
        )
    if output_shape[0] is None:
        output_shape = (tf.shape(x)[0],) + tuple(output_shape[1:])

    if isinstance(output_shape, (tuple, list)):
        output_shape = tf.stack(list(output_shape))

    padding = _preprocess_padding(padding)
    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    if dilation_rate == (1, 1):
        x = tf.compat.v1.nn.conv2d_transpose(
            x,
            kernel,
            output_shape,
            strides,
            padding=padding,
            data_format=tf_data_format,
        )
    else:
        if dilation_rate[0] != dilation_rate[1]:
            raise ValueError(
                "Expected the 2 dimensions of the `dilation_rate` argument "
                "to be equal to each other. "
                f"Received: dilation_rate={dilation_rate}"
            )
        x = tf.nn.atrous_conv2d_transpose(
            x, kernel, output_shape, rate=dilation_rate[0], padding=padding
        )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@keras_export("keras._legacy.backend.conv3d")
def conv3d(
    x,
    kernel,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1, 1),
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NDHWC":
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


@keras_export("keras._legacy.backend.cos")
def cos(x):
    """DEPRECATED."""
    return tf.cos(x)


@keras_export("keras._legacy.backend.count_params")
def count_params(x):
    """DEPRECATED."""
    return math.prod(x.shape.as_list())


@keras_export("keras._legacy.backend.ctc_batch_cost")
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """DEPRECATED."""
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        ctc_label_dense_to_sparse(y_true, label_length), tf.int32
    )

    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + backend.epsilon()
    )

    return tf.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


@keras_export("keras._legacy.backend.ctc_label_dense_to_sparse")
def ctc_label_dense_to_sparse(labels, label_lengths):
    """DEPRECATED."""
    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return tf.expand_dims(tf.range(tf.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(
        tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(
        tf.reshape(
            tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
            reverse(label_shape, 0),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(
        tf.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        tf.cast(indices, tf.int64), vals_sparse, tf.cast(label_shape, tf.int64)
    )


@keras_export("keras._legacy.backend.ctc_decode")
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """DEPRECATED."""
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + backend.epsilon()
    )
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)


@keras_export("keras._legacy.backend.cumsum")
def cumsum(x, axis=0):
    """DEPRECATED."""
    return tf.cumsum(x, axis=axis)


@keras_export("keras._legacy.backend.cumprod")
def cumprod(x, axis=0):
    """DEPRECATED."""
    return tf.math.cumprod(x, axis=axis)


@keras_export("keras._legacy.backend.depthwise_conv2d")
def depthwise_conv2d(
    x,
    depthwise_kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = tf.nn.depthwise_conv2d(
        x,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        dilations=dilation_rate,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@keras_export("keras._legacy.backend.dot")
def dot(x, y):
    """DEPRECATED."""
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.shape, tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.shape, tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(
            tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:]
        )
    if is_sparse(x):
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


@keras_export("keras._legacy.backend.dropout")
def dropout(x, level, noise_shape=None, seed=None):
    """DEPRECATED."""
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.nn.dropout(x, rate=level, noise_shape=noise_shape, seed=seed)


@keras_export("keras._legacy.backend.dtype")
def dtype(x):
    """DEPRECATED."""
    return x.dtype.base_dtype.name


@keras_export("keras._legacy.backend.elu")
def elu(x, alpha=1.0):
    """DEPRECATED."""
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)


@keras_export("keras._legacy.backend.equal")
def equal(x, y):
    """DEPRECATED."""
    return tf.equal(x, y)


@keras_export("keras._legacy.backend.eval")
def eval(x):
    """DEPRECATED."""
    return get_value(to_dense(x))


@keras_export("keras._legacy.backend.exp")
def exp(x):
    """DEPRECATED."""
    return tf.exp(x)


@keras_export("keras._legacy.backend.expand_dims")
def expand_dims(x, axis=-1):
    """DEPRECATED."""
    return tf.expand_dims(x, axis)


@keras_export("keras._legacy.backend.eye")
def eye(size, dtype=None, name=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    tf_dtype = tf.as_dtype(dtype)
    return variable(tf.eye(size, dtype=tf_dtype), dtype, name)


@keras_export("keras._legacy.backend.flatten")
def flatten(x):
    """DEPRECATED."""
    return tf.reshape(x, [-1])


@keras_export("keras._legacy.backend.foldl")
def foldl(fn, elems, initializer=None, name=None):
    """DEPRECATED."""
    return tf.compat.v1.foldl(fn, elems, initializer=initializer, name=name)


@keras_export("keras._legacy.backend.foldr")
def foldr(fn, elems, initializer=None, name=None):
    """DEPRECATED."""
    return tf.compat.v1.foldr(fn, elems, initializer=initializer, name=name)


@keras_export("keras._legacy.backend.gather")
def gather(reference, indices):
    """DEPRECATED."""
    return tf.compat.v1.gather(reference, indices)


@keras_export("keras._legacy.backend.get_value")
def get_value(x):
    """DEPRECATED."""
    if not tf.is_tensor(x):
        return x
    if tf.executing_eagerly() or isinstance(x, tf.__internal__.EagerTensor):
        return x.numpy()
    if not getattr(x, "_in_graph_mode", True):
        # This is a variable which was created in an eager context, but is being
        # evaluated from a Graph.
        with tf.__internal__.eager_context.eager_mode():
            return x.numpy()
    with tf.init_scope():
        return x.numpy()


@keras_export("keras._legacy.backend.gradients")
def gradients(loss, variables):
    """DEPRECATED."""
    return tf.compat.v1.gradients(
        loss, variables, colocate_gradients_with_ops=True
    )


@keras_export("keras._legacy.backend.greater")
def greater(x, y):
    """DEPRECATED."""
    return tf.greater(x, y)


@keras_export("keras._legacy.backend.greater_equal")
def greater_equal(x, y):
    """DEPRECATED."""
    return tf.greater_equal(x, y)


@keras_export("keras._legacy.backend.hard_sigmoid")
def hard_sigmoid(x):
    """DEPRECATED."""
    point_two = tf.convert_to_tensor(0.2, dtype=x.dtype)
    point_five = tf.convert_to_tensor(0.5, dtype=x.dtype)
    x = tf.multiply(x, point_two)
    x = tf.add(x, point_five)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


@keras_export("keras._legacy.backend.in_top_k")
def in_top_k(predictions, targets, k):
    """DEPRECATED."""
    return tf.compat.v1.math.in_top_k(predictions, targets, k)


@keras_export("keras._legacy.backend.int_shape")
def int_shape(x):
    """DEPRECATED."""
    try:
        shape = x.shape
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape
    except ValueError:
        return None


@keras_export("keras._legacy.backend.is_sparse")
def is_sparse(tensor):
    """DEPRECATED."""
    spec = getattr(tensor, "_type_spec", None)
    if spec is not None:
        return isinstance(spec, tf.SparseTensorSpec)
    return isinstance(tensor, tf.SparseTensor)


@keras_export("keras._legacy.backend.l2_normalize")
def l2_normalize(x, axis=None):
    """DEPRECATED."""
    return tf.linalg.l2_normalize(x, axis=axis)


@keras_export("keras._legacy.backend.less")
def less(x, y):
    """DEPRECATED."""
    return tf.less(x, y)


@keras_export("keras._legacy.backend.less_equal")
def less_equal(x, y):
    """DEPRECATED."""
    return tf.less_equal(x, y)


@keras_export("keras._legacy.backend.log")
def log(x):
    """DEPRECATED."""
    return tf.math.log(x)


@keras_export("keras._legacy.backend.map_fn")
def map_fn(fn, elems, name=None, dtype=None):
    """DEPRECATED."""
    return tf.compat.v1.map_fn(fn, elems, name=name, dtype=dtype)


@keras_export("keras._legacy.backend.max")
def max(x, axis=None, keepdims=False):
    """DEPRECATED."""
    return tf.reduce_max(x, axis, keepdims)


@keras_export("keras._legacy.backend.maximum")
def maximum(x, y):
    """DEPRECATED."""
    return tf.maximum(x, y)


@keras_export("keras._legacy.backend.mean")
def mean(x, axis=None, keepdims=False):
    """DEPRECATED."""
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, backend.floatx())
    return tf.reduce_mean(x, axis, keepdims)


@keras_export("keras._legacy.backend.min")
def min(x, axis=None, keepdims=False):
    """DEPRECATED."""
    return tf.reduce_min(x, axis, keepdims)


@keras_export("keras._legacy.backend.minimum")
def minimum(x, y):
    """DEPRECATED."""
    return tf.minimum(x, y)


@keras_export("keras._legacy.backend.moving_average_update")
def moving_average_update(x, value, momentum):
    """DEPRECATED."""
    momentum = tf.cast(momentum, x.dtype)
    value = tf.cast(value, x.dtype)
    return x.assign_sub((x - value) * (1 - momentum))


@keras_export("keras._legacy.backend.name_scope")
def name_scope(name):
    """DEPRECATED."""
    return tf.name_scope(name)


@keras_export("keras._legacy.backend.ndim")
def ndim(x):
    """DEPRECATED."""
    return x.shape.rank


@keras_export("keras._legacy.backend.not_equal")
def not_equal(x, y):
    """DEPRECATED."""
    return tf.not_equal(x, y)


@keras_export("keras._legacy.backend.one_hot")
def one_hot(indices, num_classes):
    """DEPRECATED."""
    return tf.one_hot(indices, depth=num_classes, axis=-1)


@keras_export("keras._legacy.backend.ones")
def ones(shape, dtype=None, name=None):
    """DEPRECATED."""
    with tf.init_scope():
        if dtype is None:
            dtype = backend.floatx()
        tf_dtype = tf.as_dtype(dtype)
        v = tf.ones(shape=shape, dtype=tf_dtype, name=name)
        if py_all(v.shape.as_list()):
            return variable(v, dtype=dtype, name=name)
        return v


@keras_export("keras._legacy.backend.ones_like")
def ones_like(x, dtype=None, name=None):
    """DEPRECATED."""
    return tf.ones_like(x, dtype=dtype, name=name)


@keras_export("keras._legacy.backend.permute_dimensions")
def permute_dimensions(x, pattern):
    """DEPRECATED."""
    return tf.transpose(x, perm=pattern)


@keras_export("keras._legacy.backend.pool2d")
def pool2d(
    x,
    pool_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    pool_mode="max",
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    if len(pool_size) != 2:
        raise ValueError("`pool_size` must be a tuple of 2 integers.")
    if len(strides) != 2:
        raise ValueError("`strides` must be a tuple of 2 integers.")

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == "max":
        x = tf.compat.v1.nn.max_pool(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    elif pool_mode == "avg":
        x = tf.compat.v1.nn.avg_pool(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    else:
        raise ValueError(f"Invalid pooling mode: {str(pool_mode)}")

    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@keras_export("keras._legacy.backend.pool3d")
def pool3d(
    x,
    pool_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    pool_mode="max",
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == "NDHWC":
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == "max":
        x = tf.nn.max_pool3d(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    elif pool_mode == "avg":
        x = tf.nn.avg_pool3d(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    else:
        raise ValueError(f"Invalid pooling mode: {str(pool_mode)}")

    if data_format == "channels_first" and tf_data_format == "NDHWC":
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


@keras_export("keras._legacy.backend.pow")
def pow(x, a):
    """DEPRECATED."""
    return tf.pow(x, a)


@keras_export("keras._legacy.backend.prod")
def prod(x, axis=None, keepdims=False):
    """DEPRECATED."""
    return tf.reduce_prod(x, axis, keepdims)


@keras_export("keras._legacy.backend.random_bernoulli")
def random_bernoulli(shape, p=0.0, dtype=None, seed=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(
        tf.random.uniform(shape, dtype=dtype, seed=seed) <= p,
        tf.ones(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
    )


@keras_export("keras._legacy.backend.random_normal")
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random.normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


@keras_export("keras._legacy.backend.random_normal_variable")
def random_normal_variable(
    shape, mean, scale, dtype=None, name=None, seed=None
):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    tf_dtype = tf.as_dtype(dtype)
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.compat.v1.random_normal_initializer(
        mean, scale, dtype=tf_dtype, seed=seed
    )(shape)
    return variable(value, dtype=dtype, name=name)


@keras_export("keras._legacy.backend.random_uniform")
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )


@keras_export("keras._legacy.backend.random_uniform_variable")
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    tf_dtype = tf.as_dtype(dtype)
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.compat.v1.random_uniform_initializer(
        low, high, dtype=tf_dtype, seed=seed
    )(shape)
    return variable(value, dtype=dtype, name=name)


@keras_export("keras._legacy.backend.reshape")
def reshape(x, shape):
    """DEPRECATED."""
    return tf.reshape(x, shape)


@keras_export("keras._legacy.backend.relu")
def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    """DEPRECATED."""
    # While x can be a tensor or variable, we also see cases where
    # numpy arrays, lists, tuples are passed as well.
    # lists, tuples do not have 'dtype' attribute.
    dtype = getattr(x, "dtype", backend.floatx())
    if alpha != 0.0:
        if max_value is None and threshold == 0:
            return tf.nn.leaky_relu(x, alpha=alpha)

        if threshold != 0:
            negative_part = tf.nn.relu(-x + threshold)
        else:
            negative_part = tf.nn.relu(-x)
    else:
        negative_part = 1

    clip_max = max_value is not None

    if threshold != 0:
        # computes x for x > threshold else 0
        x = x * tf.cast(tf.greater(x, threshold), dtype=dtype)
    elif max_value == 6:
        # if no threshold, then can use nn.relu6 native TF op for performance
        x = tf.nn.relu6(x)
        clip_max = False
    else:
        x = tf.nn.relu(x)

    if clip_max:
        max_value = tf.convert_to_tensor(max_value, dtype=x.dtype)
        zero = tf.convert_to_tensor(0, dtype=x.dtype)
        x = tf.clip_by_value(x, zero, max_value)

    if alpha != 0.0:
        alpha = tf.convert_to_tensor(alpha, dtype=x.dtype)
        x -= alpha * negative_part
    return x


@keras_export("keras._legacy.backend.repeat")
def repeat(x, n):
    """DEPRECATED."""
    if ndim(x) != 2:
        raise ValueError(
            f"Expected input `x` to have rank 2. Received: rank(x)={ndim(x)}"
        )
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


@keras_export("keras._legacy.backend.repeat_elements")
def repeat_elements(x, rep, axis):
    """DEPRECATED."""
    x_shape = x.shape.as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)

    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.

    # Repeating
    auxiliary_axis = axis + 1
    x_shape = tf.shape(x)
    x_rep = tf.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.shape) + 1)
    reps[auxiliary_axis] = rep
    x_rep = tf.tile(x_rep, reps)

    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = tf.constant(reps, dtype="int32")
    x_shape *= reps
    x_rep = tf.reshape(x_rep, x_shape)

    # Fix shape representation
    x_shape = x.shape.as_list()
    x_rep.set_shape(x_shape)
    return x_rep


@keras_export("keras._legacy.backend.resize_images")
def resize_images(
    x, height_factor, width_factor, data_format, interpolation="nearest"
):
    """DEPRECATED."""
    if data_format == "channels_first":
        rows, cols = 2, 3
    elif data_format == "channels_last":
        rows, cols = 1, 2
    else:
        raise ValueError(f"Invalid `data_format` argument: {data_format}")

    new_shape = x.shape[rows : cols + 1]
    if new_shape.is_fully_defined():
        new_shape = tf.constant(new_shape.as_list(), dtype="int32")
    else:
        new_shape = tf.shape(x)[rows : cols + 1]
    new_shape *= tf.constant(
        np.array([height_factor, width_factor], dtype="int32")
    )

    if data_format == "channels_first":
        x = permute_dimensions(x, [0, 2, 3, 1])
    interpolations = {
        "area": tf.image.ResizeMethod.AREA,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "gaussian": tf.image.ResizeMethod.GAUSSIAN,
        "lanczos3": tf.image.ResizeMethod.LANCZOS3,
        "lanczos5": tf.image.ResizeMethod.LANCZOS5,
        "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    }
    interploations_list = '"' + '", "'.join(interpolations.keys()) + '"'
    if interpolation in interpolations:
        x = tf.image.resize(x, new_shape, method=interpolations[interpolation])
    else:
        raise ValueError(
            "`interpolation` argument should be one of: "
            f'{interploations_list}. Received: "{interpolation}".'
        )
    if data_format == "channels_first":
        x = permute_dimensions(x, [0, 3, 1, 2])

    return x


@keras_export("keras._legacy.backend.resize_volumes")
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    """DEPRECATED."""
    if data_format == "channels_first":
        output = repeat_elements(x, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif data_format == "channels_last":
        output = repeat_elements(x, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise ValueError(f"Invalid data_format: {data_format}")


@keras_export("keras._legacy.backend.reverse")
def reverse(x, axes):
    """DEPRECATED."""
    if isinstance(axes, int):
        axes = [axes]
    return tf.reverse(x, axes)


@keras_export("keras._legacy.backend.rnn")
def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    """DEPRECATED."""
    if not tf.__internal__.tf2.enabled():
        return_all_outputs = True  # Not supported in TF1.

    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return tf.transpose(input_t, axes)

    if not time_major:
        inputs = tf.nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = tf.nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = tf.shape(flatted_inputs[0])[0]

    for input_ in flatted_inputs:
        input_.shape.with_rank_at_least(3)

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.shape) == 2:
            mask = expand_dims(mask)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    # tf.where needs its condition tensor to be the same shape as its two
    # result tensors, but in our case the condition (mask) tensor is
    # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
    # So we need to broadcast the mask to match the shape of inputs.
    # That's what the tile call does, it just repeats the mask along its
    # second dimension n times.
    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tf.nest.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tf.nest.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = tf.expand_dims(mask_t, -1)
        multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
        return tf.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        # Process the input tensors. The input tensor need to be split on the
        # time_step dim, and reverse if go_backwards is True. In the case of
        # nested input, the input is flattened and then transformed
        # individually.  The result of this will be a tuple of lists, each of
        # the item in tuple is list of the tensor with shape (batch, feature)
        def _process_single_input_t(input_t):
            input_t = tf.unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if tf.nest.is_nested(inputs):
            processed_input = tf.nest.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tf.nest.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = tf.unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                flat_states = tf.nest.flatten(states)
                flat_new_states = tf.nest.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    tf.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tf.nest.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)

            if zero_output_for_mask:
                last_output = tf.where(
                    _expand_mask(mask_list[-1], last_output),
                    last_output,
                    zeros_like(last_output),
                )
                outputs = tf.where(
                    _expand_mask(mask, outputs, fixed_dim=2),
                    outputs,
                    zeros_like(outputs),
                )

        else:  # mask is None
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)

    else:  # Unroll == False
        states = tuple(initial_states)

        # Create input tensor array, if the inputs is nested tensors, then it
        # will be flattened first, and tensor array will be created one per
        # flattened tensor.
        input_ta = tuple(
            tf.TensorArray(
                dtype=inp.dtype,
                size=time_steps_t,
                tensor_array_name=f"input_ta_{i}",
            )
            for i, inp in enumerate(flatted_inputs)
        )
        input_ta = tuple(
            (
                ta.unstack(input_)
                if not go_backwards
                else ta.unstack(reverse(input_, 0))
            )
            for ta, input_ in zip(input_ta, flatted_inputs)
        )

        # Get the time(0) input and compute the output for that, the output will
        # be used to determine the dtype of output tensor array. Don't read from
        # input_ta due to TensorArray clear_after_read default to True.
        input_time_zero = tf.nest.pack_sequence_as(
            inputs, [inp[0] for inp in flatted_inputs]
        )
        # output_time_zero is used to determine the cell output shape and its
        # dtype.  the value is discarded.
        output_time_zero, _ = step_function(
            input_time_zero, tuple(initial_states) + tuple(constants)
        )

        output_ta_size = time_steps_t if return_all_outputs else 1
        output_ta = tuple(
            tf.TensorArray(
                dtype=out.dtype,
                size=output_ta_size,
                element_shape=out.shape,
                tensor_array_name=f"output_ta_{i}",
            )
            for i, out in enumerate(tf.nest.flatten(output_time_zero))
        )

        time = tf.constant(0, dtype="int32", name="time")

        if input_length is None:
            max_iterations = time_steps_t
        else:
            max_iterations = tf.reduce_max(input_length)

        while_loop_kwargs = {
            "cond": lambda time, *_: time < time_steps_t,
            "maximum_iterations": max_iterations,
            "parallel_iterations": 32,
            "swap_memory": True,
        }
        if mask is not None:
            if go_backwards:
                mask = reverse(mask, 0)

            mask_ta = tf.TensorArray(
                dtype=tf.bool, size=time_steps_t, tensor_array_name="mask_ta"
            )
            mask_ta = mask_ta.unstack(mask)

            def masking_fn(time):
                return mask_ta.read(time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
                    for o in flat_out
                )
                return tuple(
                    tf.where(m, o, fm)
                    for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)
                )

        elif isinstance(input_length, tf.Tensor):
            if go_backwards:
                max_len = tf.reduce_max(input_length, axis=0)
                rev_input_length = tf.subtract(max_len - 1, input_length)

                def masking_fn(time):
                    return tf.less(rev_input_length, time)

            else:

                def masking_fn(time):
                    return tf.greater(input_length, time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple(
                    tf.compat.v1.where(mask_t, o, zo)
                    for (o, zo) in zip(flat_out, flat_mask)
                )

        else:
            masking_fn = None

        if masking_fn is not None:
            # Mask for the T output will be base on the output of T - 1. In the
            # case T = 0, a zero filled tensor will be used.
            flat_zero_output = tuple(
                tf.zeros_like(o) for o in tf.nest.flatten(output_time_zero)
            )

            def _step(time, output_ta_t, prev_output, *states):
                """RNN step function.

                Args:
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    prev_output: tuple of outputs from time - 1.
                    *states: List of states.

                Returns:
                    Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
                """
                current_input = tuple(ta.read(time) for ta in input_ta)
                # maybe set shape.
                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(time)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                # mask output
                flat_output = tf.nest.flatten(output)
                flat_mask_output = (
                    flat_zero_output
                    if zero_output_for_mask
                    else tf.nest.flatten(prev_output)
                )
                flat_new_output = compute_masked_output(
                    mask_t, flat_output, flat_mask_output
                )

                # mask states
                flat_state = tf.nest.flatten(states)
                flat_new_state = tf.nest.flatten(new_states)
                for state, new_state in zip(flat_state, flat_new_state):
                    if isinstance(new_state, tf.Tensor):
                        new_state.set_shape(state.shape)
                flat_final_state = compute_masked_output(
                    mask_t, flat_new_state, flat_state
                )
                new_states = tf.nest.pack_sequence_as(
                    new_states, flat_final_state
                )

                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_new_output)
                )

                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(
                    new_states
                )

            final_outputs = tf.compat.v1.while_loop(
                body=_step,
                loop_vars=(time, output_ta, flat_zero_output) + states,
                **while_loop_kwargs,
            )
            # Skip final_outputs[2] which is the output for final timestep.
            new_states = final_outputs[3:]
        else:

            def _step(time, output_ta_t, *states):
                """RNN step function.

                Args:
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                Returns:
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = tuple(ta.read(time) for ta in input_ta)
                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_state = tf.nest.flatten(states)
                flat_new_state = tf.nest.flatten(new_states)
                for state, new_state in zip(flat_state, flat_new_state):
                    if isinstance(new_state, tf.Tensor):
                        new_state.set_shape(state.shape)

                flat_output = tf.nest.flatten(output)
                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_output)
                )

                new_states = tf.nest.pack_sequence_as(
                    initial_states, flat_new_state
                )
                return (time + 1, output_ta_t) + tuple(new_states)

            final_outputs = tf.compat.v1.while_loop(
                body=_step,
                loop_vars=(time, output_ta) + states,
                **while_loop_kwargs,
            )
            new_states = final_outputs[2:]

        output_ta = final_outputs[1]

        outputs = tuple(o.stack() for o in output_ta)
        last_output = tuple(o[-1] for o in outputs)

        outputs = tf.nest.pack_sequence_as(output_time_zero, outputs)
        last_output = tf.nest.pack_sequence_as(output_time_zero, last_output)

    # static shape inference
    def set_shape(output_):
        if isinstance(output_, tf.Tensor):
            shape = output_.shape.as_list()
            if return_all_outputs:
                shape[0] = time_steps
            else:
                shape[0] = 1
            shape[1] = batch
            output_.set_shape(shape)
        return output_

    outputs = tf.nest.map_structure(set_shape, outputs)

    if not time_major:
        outputs = tf.nest.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


@keras_export("keras._legacy.backend.round")
def round(x):
    """DEPRECATED."""
    return tf.round(x)


@keras_export("keras._legacy.backend.separable_conv2d")
def separable_conv2d(
    x,
    depthwise_kernel,
    pointwise_kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    """DEPRECATED."""
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    if len(strides) != 2:
        raise ValueError("`strides` must be a tuple of 2 integers.")

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if not isinstance(strides, tuple):
        strides = tuple(strides)
    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = tf.nn.separable_conv2d(
        x,
        depthwise_kernel,
        pointwise_kernel,
        strides=strides,
        padding=padding,
        dilations=dilation_rate,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@keras_export("keras._legacy.backend.set_value")
def set_value(x, value):
    """DEPRECATED."""
    value = np.asarray(value, dtype=x.dtype.name)
    x.assign(value)


@keras_export("keras._legacy.backend.shape")
def shape(x):
    """DEPRECATED."""
    return tf.shape(x)


@keras_export("keras._legacy.backend.sigmoid")
def sigmoid(x):
    """DEPRECATED."""
    output = tf.sigmoid(x)
    return output


@keras_export("keras._legacy.backend.sign")
def sign(x):
    """DEPRECATED."""
    return tf.sign(x)


@keras_export("keras._legacy.backend.sin")
def sin(x):
    """DEPRECATED."""
    return tf.sin(x)


@keras_export("keras._legacy.backend.softmax")
def softmax(x, axis=-1):
    """DEPRECATED."""
    if x.shape.rank <= 1:
        raise ValueError(
            f"Cannot apply softmax to a tensor that is 1D. Received input: {x}"
        )

    if isinstance(axis, int):
        output = tf.nn.softmax(x, axis=axis)
    else:
        # nn.softmax does not support tuple axis.
        numerator = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
        denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
        output = numerator / denominator

    # Cache the logits to use for crossentropy loss.
    output._keras_logits = x
    return output


@keras_export("keras._legacy.backend.softplus")
def softplus(x):
    """DEPRECATED."""
    return tf.math.softplus(x)


@keras_export("keras._legacy.backend.softsign")
def softsign(x):
    """DEPRECATED."""
    return tf.math.softsign(x)


@keras_export("keras._legacy.backend.sparse_categorical_crossentropy")
def sparse_categorical_crossentropy(
    target, output, from_logits=False, axis=-1, ignore_class=None
):
    """DEPRECATED."""
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    target = cast(target, "int64")

    if not from_logits:
        epsilon_ = tf.convert_to_tensor(backend.epsilon(), output.dtype)
        output = tf.clip_by_value(output, epsilon_, 1 - epsilon_)
        output = tf.math.log(output)

    # Permute output so that the last axis contains the logits/probabilities.
    if isinstance(output.shape, (tuple, list)):
        output_rank = len(output.shape)
    else:
        output_rank = output.shape.ndims
    if output_rank is not None:
        axis %= output_rank
        if axis != output_rank - 1:
            permutation = list(
                itertools.chain(
                    range(axis), range(axis + 1, output_rank), [axis]
                )
            )
            output = tf.transpose(output, perm=permutation)
    elif axis != -1:
        raise ValueError(
            "Cannot compute sparse categorical crossentropy with `axis={}` "
            "on an output tensor with unknown rank".format(axis)
        )

    # Try to adjust the shape so that rank of labels = rank of logits - 1.
    output_shape = tf.shape(output)
    target_rank = target.shape.ndims

    update_shape = (
        target_rank is not None
        and output_rank is not None
        and target_rank != output_rank - 1
    )
    if update_shape:
        target = flatten(target)
        output = tf.reshape(output, [-1, output_shape[-1]])

    if ignore_class is not None:
        valid_mask = tf.not_equal(target, cast(ignore_class, target.dtype))
        target = target[valid_mask]
        output = output[valid_mask]

    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=output
    )

    if ignore_class is not None:
        res_shape = cast(output_shape[:-1], "int64")
        valid_mask = tf.reshape(valid_mask, res_shape)
        res = tf.scatter_nd(tf.where(valid_mask), res, res_shape)
        res._keras_mask = valid_mask

        return res

    if update_shape and output_rank >= 3:
        # If our output includes timesteps or
        # spatial dimensions we need to reshape
        res = tf.reshape(res, output_shape[:-1])

    return res


@keras_export("keras._legacy.backend.spatial_2d_padding")
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """DEPRECATED."""
    if len(padding) != 2 or len(padding[0]) != 2 or len(padding[1]) != 2:
        raise ValueError(
            "Expected `padding` to be a tuple of 2 tuples of 2 integers. "
            f"Received: padding={padding}"
        )
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    if data_format == "channels_first":
        pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
    else:
        pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
    return tf.compat.v1.pad(x, pattern)


@keras_export("keras._legacy.backend.spatial_3d_padding")
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """DEPRECATED."""
    if (
        len(padding) != 3
        or len(padding[0]) != 2
        or len(padding[1]) != 2
        or len(padding[2]) != 2
    ):
        raise ValueError(
            "Expected `padding` to be a tuple of 3 tuples of 2 integers. "
            f"Received: padding={padding}"
        )
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    if data_format == "channels_first":
        pattern = [
            [0, 0],
            [0, 0],
            [padding[0][0], padding[0][1]],
            [padding[1][0], padding[1][1]],
            [padding[2][0], padding[2][1]],
        ]
    else:
        pattern = [
            [0, 0],
            [padding[0][0], padding[0][1]],
            [padding[1][0], padding[1][1]],
            [padding[2][0], padding[2][1]],
            [0, 0],
        ]
    return tf.compat.v1.pad(x, pattern)


@keras_export("keras._legacy.backend.sqrt")
def sqrt(x):
    """DEPRECATED."""
    zero = tf.convert_to_tensor(0.0, x.dtype)
    x = tf.maximum(x, zero)
    return tf.sqrt(x)


@keras_export("keras._legacy.backend.square")
def square(x):
    """DEPRECATED."""
    return tf.square(x)


@keras_export("keras._legacy.backend.squeeze")
def squeeze(x, axis):
    """DEPRECATED."""
    return tf.squeeze(x, [axis])


@keras_export("keras._legacy.backend.stack")
def stack(x, axis=0):
    """DEPRECATED."""
    return tf.stack(x, axis=axis)


@keras_export("keras._legacy.backend.std")
def std(x, axis=None, keepdims=False):
    """DEPRECATED."""
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, backend.floatx())
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


@keras_export("keras._legacy.backend.stop_gradient")
def stop_gradient(variables):
    """DEPRECATED."""
    if isinstance(variables, (list, tuple)):
        return map(tf.stop_gradient, variables)
    return tf.stop_gradient(variables)


@keras_export("keras._legacy.backend.sum")
def sum(x, axis=None, keepdims=False):
    """DEPRECATED."""
    return tf.reduce_sum(x, axis, keepdims)


@keras_export("keras._legacy.backend.switch")
def switch(condition, then_expression, else_expression):
    """DEPRECATED."""
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, "bool")
    cond_ndim = ndim(condition)
    if not cond_ndim:
        if not callable(then_expression):

            def then_expression_fn():
                return then_expression

        else:
            then_expression_fn = then_expression
        if not callable(else_expression):

            def else_expression_fn():
                return else_expression

        else:
            else_expression_fn = else_expression
        x = tf.compat.v1.cond(condition, then_expression_fn, else_expression_fn)
    else:
        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        expr_ndim = ndim(then_expression)
        if cond_ndim > expr_ndim:
            raise ValueError(
                "Rank of `condition` should be less than or"
                " equal to rank of `then_expression` and "
                "`else_expression`. ndim(condition)="
                f"{cond_ndim}, ndim(then_expression)={expr_ndim}"
            )
        if cond_ndim > 1:
            ndim_diff = expr_ndim - cond_ndim
            cond_shape = tf.concat(
                [tf.shape(condition), [1] * ndim_diff], axis=0
            )
            condition = tf.reshape(condition, cond_shape)
            expr_shape = tf.shape(then_expression)
            shape_diff = expr_shape - cond_shape
            tile_shape = tf.where(
                shape_diff > 0, expr_shape, tf.ones_like(expr_shape)
            )
            condition = tf.tile(condition, tile_shape)
        x = tf.where(condition, then_expression, else_expression)
    return x


@keras_export("keras._legacy.backend.tanh")
def tanh(x):
    """DEPRECATED."""
    return tf.tanh(x)


@keras_export("keras._legacy.backend.temporal_padding")
def temporal_padding(x, padding=(1, 1)):
    """DEPRECATED."""
    if len(padding) != 2:
        raise ValueError(
            "Expected `padding` to be a tuple of 2 integers. "
            f"Received: padding={padding}"
        )
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.compat.v1.pad(x, pattern)


@keras_export("keras._legacy.backend.tile")
def tile(x, n):
    """DEPRECATED."""
    if isinstance(n, int):
        n = [n]
    return tf.tile(x, n)


@keras_export("keras._legacy.backend.to_dense")
def to_dense(tensor):
    """DEPRECATED."""
    if is_sparse(tensor):
        return tf.sparse.to_dense(tensor)
    else:
        return tensor


@keras_export("keras._legacy.backend.transpose")
def transpose(x):
    """DEPRECATED."""
    return tf.transpose(x)


@keras_export("keras._legacy.backend.truncated_normal")
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random.truncated_normal(
        shape, mean, stddev, dtype=dtype, seed=seed
    )


@keras_export("keras._legacy.backend.update")
def update(x, new_x):
    """DEPRECATED."""
    return tf.compat.v1.assign(x, new_x)


@keras_export("keras._legacy.backend.update_add")
def update_add(x, increment):
    """DEPRECATED."""
    return tf.compat.v1.assign_add(x, increment)


@keras_export("keras._legacy.backend.update_sub")
def update_sub(x, decrement):
    """DEPRECATED."""
    return tf.compat.v1.assign_sub(x, decrement)


@keras_export("keras._legacy.backend.var")
def var(x, axis=None, keepdims=False):
    """DEPRECATED."""
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, backend.floatx())
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)


@keras_export("keras._legacy.backend.variable")
def variable(value, dtype=None, name=None, constraint=None):
    """DEPRECATED."""
    if dtype is None:
        dtype = backend.floatx()
    if hasattr(value, "tocoo"):
        sparse_coo = value.tocoo()
        indices = np.concatenate(
            (
                np.expand_dims(sparse_coo.row, 1),
                np.expand_dims(sparse_coo.col, 1),
            ),
            1,
        )
        v = tf.SparseTensor(
            indices=indices,
            values=sparse_coo.data,
            dense_shape=sparse_coo.shape,
        )
        v._keras_shape = sparse_coo.shape
        return v
    v = tf.Variable(
        value, dtype=tf.as_dtype(dtype), name=name, constraint=constraint
    )
    return v


@keras_export("keras._legacy.backend.zeros")
def zeros(shape, dtype=None, name=None):
    """DEPRECATED."""
    with tf.init_scope():
        if dtype is None:
            dtype = backend.floatx()
        tf_dtype = tf.as_dtype(dtype)
        v = tf.zeros(shape=shape, dtype=tf_dtype, name=name)
        if py_all(v.shape.as_list()):
            return variable(v, dtype=dtype, name=name)
        return v


@keras_export("keras._legacy.backend.zeros_like")
def zeros_like(x, dtype=None, name=None):
    """DEPRECATED."""
    return tf.zeros_like(x, dtype=dtype, name=name)
