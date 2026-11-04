import math
from itertools import combinations
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.distribution import DeviceMesh
from keras.distribution import TensorLayout
from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.torch.distributed_test_utils import (
    TorchDistributedTestMixin,
)
from keras.src.backend.torch.distribution_lib import distribute_data_input
from keras.src.backend.torch.ops.core import get_device
from keras.src.layers.convolutional.conv_test import np_conv1d
from keras.src.layers.convolutional.conv_test import np_conv2d
from keras.src.layers.convolutional.conv_test import np_conv3d
from keras.src.layers.convolutional.conv_transpose_test import (
    np_conv1d_transpose,
)
from keras.src.layers.convolutional.conv_transpose_test import (
    np_conv2d_transpose,
)
from keras.src.layers.convolutional.depthwise_conv_test import (
    np_depthwise_conv2d,
)
from keras.src.layers.pooling.average_pooling_test import np_avgpool1d
from keras.src.layers.pooling.average_pooling_test import np_avgpool2d
from keras.src.layers.pooling.max_pooling_test import np_maxpool1d
from keras.src.layers.pooling.max_pooling_test import np_maxpool2d
from keras.src.ops import nn as knn
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product


def _dot_product_attention(
    query, key, value, bias=None, mask=None, scale=None, is_causal=False
):
    # A pure and simplified numpy version of `dot_product_attention`
    # Ref: jax.nn.dot_product_attention
    # https://github.com/jax-ml/jax/blob/jax-v0.4.32/jax/_src/nn/functions.py#L828
    # Not support `query_seq_lengths` and `key_value_seq_lengths` args

    def _apply_masks(logits, mask, is_causal):
        def _get_large_negative(dtype):
            dtype = standardize_dtype(dtype)
            if dtype == "float16":
                val = 65500.0
            else:
                val = 3.38953e38
            return np.asarray(val * -0.7, dtype=dtype)

        def _get_causal_mask(query_length, key_length):
            mask = np.tril(np.ones((query_length, key_length), dtype=np.bool_))
            return mask[None, None, :, :]

        if mask is None and not is_causal:
            return logits
        combined_mask = np.ones_like(logits, dtype=np.bool_)
        if mask is not None:
            combined_mask = np.logical_and(combined_mask, mask)
        if is_causal:
            T, S = logits.shape[2], logits.shape[3]
            mask = _get_causal_mask(T, S)
            combined_mask = np.logical_and(combined_mask, mask)
        padded_logits = np.where(
            combined_mask, logits, _get_large_negative(logits.dtype)
        )
        return padded_logits

    def softmax(x, axis=None):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    _, _, _, H = key.shape
    scale = (1.0 / np.sqrt(H)) if scale is None else scale
    logits = np.einsum("BTNH,BSNH->BNTS", query, key)
    logits *= np.array(scale, dtype=logits.dtype)
    if bias is not None:
        logits = (logits + bias).astype(logits.dtype)
    padded_logits = _apply_masks(logits, mask, is_causal)
    padded_logits = padded_logits.astype(np.float32)
    probs = softmax(padded_logits, axis=-1).astype(key.dtype)
    return np.einsum("BNTS,BSNH->BTNH", probs, value)


class NNOpsDynamicShapeTest(testing.TestCase):
    def test_relu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.relu(x).shape, (None, 2, 3))

    def test_relu6(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.relu6(x).shape, (None, 2, 3))

    def test_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.sigmoid(x).shape, (None, 2, 3))

    def test_sparse_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.sparse_sigmoid(x).shape, (None, 2, 3))

    def test_softplus(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softplus(x).shape, (None, 2, 3))

    def test_softsign(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softsign(x).shape, (None, 2, 3))

    def test_silu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.silu(x).shape, (None, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (None, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (None, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (None, 2, 3))

    def test_hard_silu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_silu(x).shape, (None, 2, 3))

    def test_elu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.elu(x).shape, (None, 2, 3))

    def test_selu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.selu(x).shape, (None, 2, 3))

    def test_gelu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.gelu(x).shape, (None, 2, 3))

    def test_celu(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.celu(x).shape, (None, 2, 3))

    def test_glu(self):
        x = KerasTensor([None, 2, 4])
        self.assertEqual(knn.glu(x).shape, (None, 2, 2))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            knn.glu(x, axis=5)
        x_eager = knp.ones((2, 4))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            knn.glu(x_eager, axis=5)

    def test_tanh_shrink(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.tanh_shrink(x).shape, (None, 2, 3))

    def test_hard_tanh(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_tanh(x).shape, (None, 2, 3))

    def test_hard_shrink(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.hard_shrink(x).shape, (None, 2, 3))

    def test_threshld(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.threshold(x, 0, 0).shape, (None, 2, 3))

    def test_squareplus(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.squareplus(x).shape, (None, 2, 3))

    def test_soft_shrink(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.soft_shrink(x).shape, (None, 2, 3))

    def test_sparse_plus(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.sparse_plus(x).shape, (None, 2, 3))

    def test_softmax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.softmax(x).shape, (None, 2, 3))
        self.assertEqual(knn.softmax(x, axis=1).shape, (None, 2, 3))
        self.assertEqual(knn.softmax(x, axis=-1).shape, (None, 2, 3))

    def test_softmax_in_graph(self):
        class SoftmaxLayer(keras.Layer):
            def call(self, x):
                return ops.softmax(x, axis=-1)

        class Model(keras.Model):
            def __init__(self):
                x = keras.Input(shape=(None,))
                y = SoftmaxLayer()(x)
                super().__init__(inputs=x, outputs=y)

        # Make sure Keras is able to compile the model graph
        model = Model()
        x = ops.array([[1.0, 2.0, 3.0, 4.0]])
        model.predict(x)

    def test_log_softmax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.log_softmax(x).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=1).shape, (None, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=-1).shape, (None, 2, 3))
        with self.assertRaises(ValueError):
            knn.log_softmax(x, axis=3)

    def test_sparsemax(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.sparsemax(x).shape, (None, 2, 3))
        with self.assertRaises(ValueError):
            knn.sparsemax(x, axis=3)

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 8, 3)
        else:
            input_shape = (None, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (None, 4, 3) if data_format == "channels_last" else (None, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (None, 8, None, 3)
        else:
            input_shape = (None, 3, 8, None)
        x = KerasTensor(input_shape)
        (
            self.assertEqual(knn.max_pool(x, 2, 1).shape, (None, 7, None, 3))
            if data_format == "channels_last"
            else (None, 3, 7, None)
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )

    def test_average_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 8, 3)
        else:
            input_shape = (None, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (None, 4, 3) if data_format == "channels_last" else (None, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (None, 8, None, 3)
        else:
            input_shape = (None, 3, 8, None)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (
                (None, 7, None, 3)
                if data_format == "channels_last"
                else (None, 3, 7, None)
            ),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (
                (None, 4, None, 3)
                if data_format == "channels_last"
                else (None, 3, 4, None)
            ),
        )

    def test_adaptive_max_pool(self):
        # 1D, 2D, and 3D symbolic inputs with channels_last and channels_first
        x_1d = KerasTensor((None, 16, 3))
        self.assertEqual(
            knn.adaptive_max_pool(x_1d, 8, data_format="channels_last").shape,
            (None, 8, 3),
        )
        self.assertEqual(
            knn.adaptive_max_pool(
                x_1d, (8,), data_format="channels_last"
            ).shape,
            (None, 8, 3),
        )

        x_2d_cl = KerasTensor((2, 64, 64, 3))
        self.assertEqual(
            knn.adaptive_max_pool(
                x_2d_cl, (32, 16), data_format="channels_last"
            ).shape,
            (2, 32, 16, 3),
        )
        self.assertEqual(
            knn.adaptive_max_pool(
                x_2d_cl, [32, 16], data_format="channels_last"
            ).shape,
            (2, 32, 16, 3),
        )
        self.assertEqual(
            knn.adaptive_max_pool(
                x_2d_cl, 7, data_format="channels_last"
            ).shape,
            (2, 7, 7, 3),
        )

        x_2d_cf = KerasTensor((2, 3, 64, 64))
        self.assertEqual(
            knn.adaptive_max_pool(
                x_2d_cf, (32, 16), data_format="channels_first"
            ).shape,
            (2, 3, 32, 16),
        )
        self.assertEqual(
            knn.adaptive_max_pool(
                x_2d_cf, 7, data_format="channels_first"
            ).shape,
            (2, 3, 7, 7),
        )

        x_3d_cl = KerasTensor((2, 16, 16, 16, 3))
        self.assertEqual(
            knn.adaptive_max_pool(
                x_3d_cl, 4, data_format="channels_last"
            ).shape,
            (2, 4, 4, 4, 3),
        )

    def test_adaptive_average_pool(self):
        x_1d = KerasTensor((None, 16, 3))
        self.assertEqual(
            knn.adaptive_average_pool(
                x_1d, 8, data_format="channels_last"
            ).shape,
            (None, 8, 3),
        )

        x_2d_cl = KerasTensor((2, 64, 64, 3))
        self.assertEqual(
            knn.adaptive_average_pool(
                x_2d_cl, (32, 16), data_format="channels_last"
            ).shape,
            (2, 32, 16, 3),
        )
        self.assertEqual(
            knn.adaptive_average_pool(
                x_2d_cl, [32, 16], data_format="channels_last"
            ).shape,
            (2, 32, 16, 3),
        )
        self.assertEqual(
            knn.adaptive_average_pool(
                x_2d_cl, 7, data_format="channels_last"
            ).shape,
            (2, 7, 7, 3),
        )

        x_2d_cf = KerasTensor((2, 3, 64, 64))
        self.assertEqual(
            knn.adaptive_average_pool(
                x_2d_cf, (32, 16), data_format="channels_first"
            ).shape,
            (2, 3, 32, 16),
        )
        self.assertEqual(
            knn.adaptive_average_pool(
                x_2d_cf, 7, data_format="channels_first"
            ).shape,
            (2, 3, 7, 7),
        )

        x_3d_cl = KerasTensor((2, 16, 16, 16, 3))
        self.assertEqual(
            knn.adaptive_average_pool(
                x_3d_cl, 4, data_format="channels_last"
            ).shape,
            (2, 4, 4, 4, 3),
        )

    def test_multi_hot(self):
        x = KerasTensor([None, 3, 1])
        self.assertEqual(knn.multi_hot(x, 5).shape, (None, 1, 5))
        self.assertEqual(knn.multi_hot(x, 5, 1).shape, (None, 3, 1))
        self.assertEqual(knn.multi_hot(x, 5, 2).shape, (None, 5, 1))
        self.assertSparse(knn.multi_hot(x, 5, sparse=True))

    @parameterized.named_parameters(
        named_product(dtype=["float32", "int32", "bool"], sparse=[False, True])
    )
    def test_multi_hot_dtype(self, dtype, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")

        x = np.arange(5)
        out = knn.multi_hot(x, 5, axis=0, dtype=dtype, sparse=sparse)
        self.assertEqual(standardize_dtype(out.dtype), dtype)
        self.assertSparse(out, sparse)

    def test_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 17, 2)
                    if data_format == "channels_last"
                    else (None, 2, 17)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 20, 2)
                    if data_format == "channels_last"
                    else (None, 2, 20)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape,
            (None, 7, 2) if data_format == "channels_last" else (None, 2, 7),
        )

        # Test 2D conv.
        if data_format == "channels_last":
            input_shape = (None, 10, None, 3)
        else:
            input_shape = (None, 3, 10, None)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, None, 2)
                    if data_format == "channels_last"
                    else (None, 2, 9, None)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 10, None, 2)
                    if data_format == "channels_last"
                    else (None, 2, 10, None)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (
                (None, 4, None, 2)
                if data_format == "channels_last"
                else (None, 2, 4, None)
            ),
        )

        # Test 2D conv - H, W specified
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, 9, 2)
                    if data_format == "channels_last"
                    else (None, 2, 9, 9)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 10, 10, 2)
                    if data_format == "channels_last"
                    else (None, 2, 10, 10)
                ),
            )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (
                (None, 4, 9, 2)
                if data_format == "channels_last"
                else (None, 2, 4, 9)
            ),
        )

        # Test 3D conv.
        if data_format == "channels_last":
            input_shape = (None, 8, None, 8, 3)
        else:
            input_shape = (None, 3, 8, None, 8)
        inputs_3d = KerasTensor(input_shape)
        kernel = KerasTensor([3, 3, 3, 3, 2])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.conv(inputs_3d, kernel, 1, padding=padding).shape,
                (
                    (None, 6, None, 6, 2)
                    if data_format == "channels_last"
                    else (None, 2, 6, None, 6)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.conv(inputs_3d, kernel, (2, 1, 2), padding=padding).shape,
                (
                    (None, 4, None, 4, 2)
                    if data_format == "channels_last"
                    else (None, 2, 4, None, 4)
                ),
            )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (
                (None, 6, None, 4, 2)
                if data_format == "channels_last"
                else (None, 2, 6, None, 4)
            ),
        )

    def test_depthwise_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D depthwise conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 1])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.depthwise_conv(inputs_1d, kernel, 1, padding=padding).shape,
                (
                    (None, 17, 3)
                    if data_format == "channels_last"
                    else (None, 3, 17)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.depthwise_conv(
                    inputs_1d, kernel, (1,), padding=padding
                ).shape,
                (
                    (None, 20, 3)
                    if data_format == "channels_last"
                    else (None, 3, 20)
                ),
            )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (None, 7, 3) if data_format == "channels_last" else (None, 3, 7),
        )

        # Test 2D depthwise conv.
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 1])
        for padding in ["valid", "VALID"]:
            self.assertEqual(
                knn.depthwise_conv(inputs_2d, kernel, 1, padding=padding).shape,
                (
                    (None, 9, 9, 3)
                    if data_format == "channels_last"
                    else (None, 3, 9, 9)
                ),
            )
        for padding in ["same", "SAME"]:
            self.assertEqual(
                knn.depthwise_conv(
                    inputs_2d, kernel, (1, 2), padding=padding
                ).shape,
                (
                    (None, 10, 5, 3)
                    if data_format == "channels_last"
                    else (None, 3, 10, 5)
                ),
            )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (
                (None, 4, 4, 3)
                if data_format == "channels_last"
                else (None, 3, 4, 4)
            ),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (
                (None, 4, 5, 3)
                if data_format == "channels_last"
                else (None, 3, 4, 5)
            ),
        )

    def test_separable_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D separable conv.
        if data_format == "channels_last":
            input_shape = (None, 20, 3)
        else:
            input_shape = (None, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (None, 17, 5) if data_format == "channels_last" else (None, 5, 17),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (None, 20, 5) if data_format == "channels_last" else (None, 5, 20),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (None, 7, 5) if data_format == "channels_last" else (None, 5, 7),
        )

        # Test 2D separable conv.
        if data_format == "channels_last":
            input_shape = (None, 10, 10, 3)
        else:
            input_shape = (None, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (
                (None, 9, 9, 5)
                if data_format == "channels_last"
                else (None, 5, 9, 9)
            ),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (
                (None, 10, 5, 5)
                if data_format == "channels_last"
                else (None, 5, 10, 5)
            ),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (
                (None, 4, 5, 5)
                if data_format == "channels_last"
                else (None, 5, 4, 5)
            ),
        )

    def test_conv_transpose(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (None, 4, 3)
        else:
            input_shape = (None, 3, 4)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape,
            (None, 8, 5) if data_format == "channels_last" else (None, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (None, 8, 5) if data_format == "channels_last" else (None, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (None, 21, 5) if data_format == "channels_last" else (None, 5, 21),
        )

        if data_format == "channels_last":
            input_shape = (None, 4, 4, 3)
        else:
            input_shape = (None, 3, 4, 4)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape,
            (
                (None, 8, 8, 5)
                if data_format == "channels_last"
                else (None, 5, 8, 8)
            ),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (
                (None, 8, 8, 5)
                if data_format == "channels_last"
                else (None, 5, 8, 8)
            ),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (
                (None, 21, 21, 5)
                if data_format == "channels_last"
                else (None, 5, 21, 21)
            ),
        )

    def test_one_hot(self):
        x = KerasTensor([None, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (None, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (None, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (None, 3, 5, 1))
        self.assertEqual(knn.one_hot(x, 5, -2).shape, (None, 3, 5, 1))
        self.assertSparse(knn.one_hot(x, 5, sparse=True))

    @parameterized.named_parameters(
        named_product(dtype=["float32", "int32", "bool"], sparse=[False, True])
    )
    def test_one_hot_dtype(self, dtype, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")

        x = np.arange(5)
        out = knn.one_hot(x, 5, axis=0, dtype=dtype, sparse=sparse)
        self.assertEqual(standardize_dtype(out.dtype), dtype)
        self.assertSparse(out, sparse)

    def test_moments(self):
        x = KerasTensor([None, 3, 4])
        self.assertEqual(knn.moments(x, axes=[0])[0].shape, (3, 4))
        self.assertEqual(knn.moments(x, axes=[0, 1])[0].shape, (4,))
        self.assertEqual(
            knn.moments(x, axes=[0, 1], keepdims=True)[0].shape, (1, 1, 4)
        )

        self.assertEqual(knn.moments(x, axes=[1])[0].shape, (None, 4))
        self.assertEqual(knn.moments(x, axes=[1, 2])[0].shape, (None,))
        self.assertEqual(
            knn.moments(x, axes=[1, 2], keepdims=True)[0].shape, (None, 1, 1)
        )

    def test_batch_normalization(self):
        x = KerasTensor([None, 3, 4])
        mean = KerasTensor([4])
        variance = KerasTensor([4])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=-1).shape,
            (None, 3, 4),
        )

        x = KerasTensor([None, 3, 4, 5])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=2).shape,
            (None, 3, 4, 5),
        )

        mean = KerasTensor([3])
        variance = KerasTensor([3])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=1).shape,
            (None, 3, 4, 5),
        )

        # Test wrong offset shape
        self.assertRaisesRegex(
            ValueError,
            "`offset` must be a vector of length",
            knn.batch_normalization,
            KerasTensor([None, 3, 4, 5]),
            KerasTensor([5]),
            KerasTensor([5]),
            axis=-1,
            offset=KerasTensor([3]),
            scale=KerasTensor([5]),
        )

        # Test wrong scale shape
        self.assertRaisesRegex(
            ValueError,
            "`scale` must be a vector of length",
            knn.batch_normalization,
            KerasTensor([None, 3, 4, 5]),
            KerasTensor([5]),
            KerasTensor([5]),
            axis=-1,
            offset=KerasTensor([5]),
            scale=KerasTensor([3]),
        )

    def test_ctc_decode(self):
        # Test strategy="greedy"
        inputs = KerasTensor([None, 2, 3])
        sequence_lengths = KerasTensor([None])
        decoded, scores = knn.ctc_decode(inputs, sequence_lengths)
        self.assertEqual(decoded.shape, (1, None, 2))
        self.assertEqual(scores.shape, (None, 1))

        # Test strategy="beam_search"
        inputs = KerasTensor([None, 2, 3])
        sequence_lengths = KerasTensor([None])
        decoded, scores = knn.ctc_decode(
            inputs, sequence_lengths, strategy="beam_search", top_paths=2
        )
        self.assertEqual(decoded.shape, (2, None, 2))
        self.assertEqual(scores.shape, (None, 2))

    def test_normalize(self):
        x = KerasTensor([None, 2, 3])
        self.assertEqual(knn.normalize(x).shape, (None, 2, 3))

    def test_psnr(self):
        x1 = KerasTensor([None, 2, 3])
        x2 = KerasTensor([None, 5, 6])
        out = knn.psnr(x1, x2, max_val=224)
        self.assertEqual(out.shape, ())

    def test_dot_product_attention(self):
        query = KerasTensor([None, None, 8, 16])
        key = KerasTensor([None, None, 6, 16])
        value = KerasTensor([None, None, 6, 16])
        out = knn.dot_product_attention(query, key, value)
        self.assertEqual(out.shape, query.shape)

    def test_rms_normalization(self):
        x = KerasTensor([None, 8, 16])
        scale = KerasTensor([None, 8, 16])
        out = knn.rms_normalization(x, scale)
        self.assertEqual(out.shape, x.shape)

    def test_layer_normalization(self):
        x = KerasTensor([None, 8, 16])
        gamma = KerasTensor([None, 16])
        beta = KerasTensor([None, 16])
        out = knn.layer_normalization(x, gamma, beta)
        self.assertEqual(out.shape, x.shape)


class NNOpsStaticShapeTest(testing.TestCase):
    def test_relu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.relu(x).shape, (1, 2, 3))

    def test_relu6(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.relu6(x).shape, (1, 2, 3))

    def test_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.sigmoid(x).shape, (1, 2, 3))

    def test_sparse_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.sparse_sigmoid(x).shape, (1, 2, 3))

    def test_softplus(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softplus(x).shape, (1, 2, 3))

    def test_softsign(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softsign(x).shape, (1, 2, 3))

    def test_silu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.silu(x).shape, (1, 2, 3))

    def test_log_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.log_sigmoid(x).shape, (1, 2, 3))

    def test_leaky_relu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.leaky_relu(x).shape, (1, 2, 3))

    def test_hard_sigmoid(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_sigmoid(x).shape, (1, 2, 3))

    def test_hard_silu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_silu(x).shape, (1, 2, 3))

    def test_elu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.elu(x).shape, (1, 2, 3))

    def test_selu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.selu(x).shape, (1, 2, 3))

    def test_gelu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.gelu(x).shape, (1, 2, 3))

    def test_celu(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.celu(x).shape, (1, 2, 3))

    def test_glu(self):
        x = KerasTensor([1, 2, 4])
        self.assertEqual(knn.glu(x).shape, (1, 2, 2))

    def test_tanh_shrink(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.tanh_shrink(x).shape, (1, 2, 3))

    def test_hard_tanh(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_tanh(x).shape, (1, 2, 3))

    def test_hard_shrink(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.hard_shrink(x).shape, (1, 2, 3))

    def test_threshold(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.threshold(x, 0, 0).shape, (1, 2, 3))

    def test_squareplus(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.squareplus(x).shape, (1, 2, 3))

    def test_soft_shrink(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.soft_shrink(x).shape, (1, 2, 3))

    def test_sparse_plus(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.sparse_plus(x).shape, (1, 2, 3))

    def test_softmax(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.softmax(x).shape, (1, 2, 3))
        self.assertEqual(knn.softmax(x, axis=1).shape, (1, 2, 3))
        self.assertEqual(knn.softmax(x, axis=-1).shape, (1, 2, 3))

    def test_log_softmax(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.log_softmax(x).shape, (1, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=1).shape, (1, 2, 3))
        self.assertEqual(knn.log_softmax(x, axis=-1).shape, (1, 2, 3))

    def test_sparsemax(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.sparsemax(x).shape, (1, 2, 3))

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (1, 8, 3)
        else:
            input_shape = (1, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (1, 7, 3) if data_format == "channels_last" else (1, 3, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (1, 4, 3) if data_format == "channels_last" else (1, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (1, 8, 8, 3)
        else:
            input_shape = (1, 3, 8, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.max_pool(x, 2, 1).shape,
            (1, 7, 7, 3) if data_format == "channels_last" else (1, 3, 7, 7),
        )
        self.assertEqual(
            knn.max_pool(x, 2, 2, padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )
        self.assertEqual(
            knn.max_pool(x, (2, 2), (2, 2), padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )

    def test_average_pool(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (1, 8, 3)
        else:
            input_shape = (1, 3, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (1, 7, 3) if data_format == "channels_last" else (1, 3, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (1, 4, 3) if data_format == "channels_last" else (1, 3, 4),
        )

        if data_format == "channels_last":
            input_shape = (1, 8, 8, 3)
        else:
            input_shape = (1, 3, 8, 8)
        x = KerasTensor(input_shape)
        self.assertEqual(
            knn.average_pool(x, 2, 1).shape,
            (1, 7, 7, 3) if data_format == "channels_last" else (1, 3, 7, 7),
        )
        self.assertEqual(
            knn.average_pool(x, 2, 2, padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )
        self.assertEqual(
            knn.average_pool(x, (2, 2), (2, 2), padding="same").shape,
            (1, 4, 4, 3) if data_format == "channels_last" else (1, 3, 4, 4),
        )

    def test_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D conv.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="valid").shape,
            (2, 17, 2) if data_format == "channels_last" else (2, 2, 17),
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, 1, padding="same").shape,
            (2, 20, 2) if data_format == "channels_last" else (2, 2, 20),
        )
        self.assertEqual(
            knn.conv(inputs_1d, kernel, (2,), dilation_rate=2).shape,
            (2, 7, 2) if data_format == "channels_last" else (2, 2, 7),
        )

        # Test 2D conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="valid").shape,
            (2, 9, 9, 2) if data_format == "channels_last" else (2, 2, 9, 9),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, 1, padding="same").shape,
            (
                (2, 10, 10, 2)
                if data_format == "channels_last"
                else (2, 2, 10, 10)
            ),
        )
        self.assertEqual(
            knn.conv(inputs_2d, kernel, (2, 1), dilation_rate=(2, 1)).shape,
            (2, 4, 9, 2) if data_format == "channels_last" else (2, 2, 4, 9),
        )

        # Test 3D conv.
        if data_format == "channels_last":
            input_shape = (2, 8, 8, 8, 3)
        else:
            input_shape = (2, 3, 8, 8, 8)
        inputs_3d = KerasTensor(input_shape)
        kernel = KerasTensor([3, 3, 3, 3, 2])
        self.assertEqual(
            knn.conv(inputs_3d, kernel, 1, padding="valid").shape,
            (
                (2, 6, 6, 6, 2)
                if data_format == "channels_last"
                else (2, 2, 6, 6, 6)
            ),
        )
        self.assertEqual(
            knn.conv(inputs_3d, kernel, (2, 1, 2), padding="same").shape,
            (
                (2, 4, 8, 4, 2)
                if data_format == "channels_last"
                else (2, 2, 4, 8, 4)
            ),
        )
        self.assertEqual(
            knn.conv(
                inputs_3d, kernel, 1, padding="valid", dilation_rate=(1, 2, 2)
            ).shape,
            (
                (2, 6, 4, 4, 2)
                if data_format == "channels_last"
                else (2, 2, 6, 4, 4)
            ),
        )

    def test_depthwise_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D depthwise conv.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 1, padding="valid").shape,
            (2, 17, 3) if data_format == "channels_last" else (2, 3, 17),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, (1,), padding="same").shape,
            (2, 20, 3) if data_format == "channels_last" else (2, 3, 20),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_1d, kernel, 2, dilation_rate=2).shape,
            (2, 7, 3) if data_format == "channels_last" else (2, 3, 7),
        )

        # Test 2D depthwise conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 1])
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 1, padding="valid").shape,
            (2, 9, 9, 3) if data_format == "channels_last" else (2, 3, 9, 9),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, (1, 2), padding="same").shape,
            (2, 10, 5, 3) if data_format == "channels_last" else (2, 3, 10, 5),
        )
        self.assertEqual(
            knn.depthwise_conv(inputs_2d, kernel, 2, dilation_rate=2).shape,
            (2, 4, 4, 3) if data_format == "channels_last" else (2, 3, 4, 4),
        )
        self.assertEqual(
            knn.depthwise_conv(
                inputs_2d, kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 3) if data_format == "channels_last" else (2, 3, 4, 5),
        )

    def test_separable_conv(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([4, 3, 2])
        pointwise_kernel = KerasTensor([1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 17, 5) if data_format == "channels_last" else (2, 5, 17),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 1, padding="same"
            ).shape,
            (2, 20, 5) if data_format == "channels_last" else (2, 5, 20),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_1d, kernel, pointwise_kernel, 2, dilation_rate=2
            ).shape,
            (2, 7, 5) if data_format == "channels_last" else (2, 5, 7),
        )

        # Test 2D separable conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 3, 2])
        pointwise_kernel = KerasTensor([1, 1, 6, 5])
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 1, padding="valid"
            ).shape,
            (2, 9, 9, 5) if data_format == "channels_last" else (2, 5, 9, 9),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, (1, 2), padding="same"
            ).shape,
            (2, 10, 5, 5) if data_format == "channels_last" else (2, 5, 10, 5),
        )
        self.assertEqual(
            knn.separable_conv(
                inputs_2d, kernel, pointwise_kernel, 2, dilation_rate=(2, 1)
            ).shape,
            (2, 4, 5, 5) if data_format == "channels_last" else (2, 5, 4, 5),
        )

    def test_conv_input_channel_validation(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (2, 10, 3)
        else:
            input_shape = (2, 3, 10)
        inputs = KerasTensor(input_shape)
        # kernel input channels (5) do not match the input's channels (3)
        bad_kernel = KerasTensor([3, 5, 4])
        bad_depthwise_kernel = KerasTensor([3, 5, 1])
        pointwise_kernel = KerasTensor([1, 5, 4])

        with self.assertRaisesRegex(
            ValueError, "input channels must match the kernel"
        ):
            knn.conv(inputs, bad_kernel, padding="valid")

        with self.assertRaisesRegex(
            ValueError, "input channels must match the kernel"
        ):
            knn.depthwise_conv(inputs, bad_depthwise_kernel, padding="valid")

        with self.assertRaisesRegex(
            ValueError, "input channels must match the kernel"
        ):
            knn.separable_conv(
                inputs,
                bad_depthwise_kernel,
                pointwise_kernel,
                padding="valid",
            )

        # Dynamic channel dimension should NOT raise.
        if data_format == "channels_last":
            dyn_inputs = KerasTensor((2, 10, None))
        else:
            dyn_inputs = KerasTensor((2, None, 10))
        knn.conv(dyn_inputs, bad_kernel, padding="valid")
        knn.depthwise_conv(dyn_inputs, bad_depthwise_kernel, padding="valid")
        knn.separable_conv(
            dyn_inputs,
            bad_depthwise_kernel,
            pointwise_kernel,
            padding="valid",
        )

    def test_conv_transpose(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (2, 4, 3)
        else:
            input_shape = (2, 3, 4)
        inputs_1d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2).shape,
            (2, 8, 5) if data_format == "channels_last" else (2, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_1d, kernel, 2, padding="same").shape,
            (2, 8, 5) if data_format == "channels_last" else (2, 5, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_1d, kernel, 5, padding="valid", output_padding=4
            ).shape,
            (2, 21, 5) if data_format == "channels_last" else (2, 5, 21),
        )

        if data_format == "channels_last":
            input_shape = (2, 4, 4, 3)
        else:
            input_shape = (2, 3, 4, 4)
        inputs_2d = KerasTensor(input_shape)
        kernel = KerasTensor([2, 2, 5, 3])
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, 2).shape,
            (2, 8, 8, 5) if data_format == "channels_last" else (2, 5, 8, 8),
        )
        self.assertEqual(
            knn.conv_transpose(inputs_2d, kernel, (2, 2), padding="same").shape,
            (2, 8, 8, 5) if data_format == "channels_last" else (2, 5, 8, 8),
        )
        self.assertEqual(
            knn.conv_transpose(
                inputs_2d, kernel, (5, 5), padding="valid", output_padding=4
            ).shape,
            (
                (2, 21, 21, 5)
                if data_format == "channels_last"
                else (2, 5, 21, 21)
            ),
        )

    def test_conv_transpose_input_channel_validation(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (2, 4, 3)
        else:
            input_shape = (2, 3, 4)
        inputs = KerasTensor(input_shape)
        # conv_transpose kernel layout: (spatial..., out_channels, in_channels)
        # in_channels=5 mismatches the input's 3 channels.
        bad_kernel = KerasTensor([2, 4, 5])

        with self.assertRaisesRegex(
            ValueError, "input channels must match the kernel"
        ):
            knn.conv_transpose(inputs, bad_kernel, 2)

        # Dynamic channel dimension should NOT raise.
        if data_format == "channels_last":
            dyn_inputs = KerasTensor((2, 4, None))
        else:
            dyn_inputs = KerasTensor((2, None, 4))
        knn.conv_transpose(dyn_inputs, bad_kernel, 2)

    def test_batched_and_unbatched_inputs_multi_hot(self):
        x = KerasTensor([2, 3, 1], dtype="int32")
        unbatched_input = KerasTensor(
            [
                5,
            ],
            dtype="int32",
        )
        out = knn.multi_hot(unbatched_input, 5, -1)
        self.assertEqual(out.shape, (5,))
        self.assertEqual(out.dtype, backend.floatx())
        self.assertEqual(knn.multi_hot(unbatched_input, 5, 0).shape, (5,))
        self.assertEqual(knn.multi_hot(x, 5).shape, (2, 1, 5))
        self.assertEqual(knn.multi_hot(x, 5, 1).shape, (2, 3, 1))
        self.assertEqual(knn.multi_hot(x, 5, 2).shape, (2, 5, 1))
        self.assertEqual(knn.multi_hot(x, 5, dtype="bool").dtype, "bool")

    def test_one_hot(self):
        x = KerasTensor([2, 3, 1])
        self.assertEqual(knn.one_hot(x, 5).shape, (2, 3, 1, 5))
        self.assertEqual(knn.one_hot(x, 5, 1).shape, (2, 5, 3, 1))
        self.assertEqual(knn.one_hot(x, 5, 2).shape, (2, 3, 5, 1))
        self.assertEqual(knn.one_hot(x, 5, -2).shape, (2, 3, 5, 1))
        self.assertSparse(knn.one_hot(x, 5, sparse=True))

    def test_binary_crossentropy(self):
        x1 = KerasTensor([2, 3, 1])
        x2 = KerasTensor([2, 3, 1])
        self.assertEqual(knn.binary_crossentropy(x1, x2).shape, (2, 3, 1))

    def test_categorical_crossentropy(self):
        x1 = KerasTensor([2, 3, 4])
        x2 = KerasTensor([2, 3, 4])
        self.assertEqual(knn.categorical_crossentropy(x1, x2).shape, (2, 3))
        self.assertEqual(
            knn.categorical_crossentropy(x1, x2, axis=1).shape, (2, 4)
        )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            knn.categorical_crossentropy(x1, x2, axis=3)

    def test_sparse_categorical_crossentropy(self):
        x1 = KerasTensor([2, 3], dtype="int32")
        x2 = KerasTensor([2, 3, 4])
        self.assertEqual(
            knn.sparse_categorical_crossentropy(x1, x2).shape, (2, 3)
        )
        x1 = KerasTensor([2, 4], dtype="int32")
        x2 = KerasTensor([2, 3, 4])
        self.assertEqual(
            knn.sparse_categorical_crossentropy(x1, x2, axis=1).shape,
            (2, 4),
        )
        with self.assertRaises(ValueError):
            knn.sparse_categorical_crossentropy(x1, x2, axis=3)

    def test_moments(self):
        x = KerasTensor([2, 3, 4])
        self.assertEqual(knn.moments(x, axes=[0])[0].shape, (3, 4))
        self.assertEqual(knn.moments(x, axes=[0, 1])[0].shape, (4,))
        self.assertEqual(
            knn.moments(x, axes=[0, 1], keepdims=True)[0].shape, (1, 1, 4)
        )

    def test_batch_normalization(self):
        x = KerasTensor([10, 3, 4])
        mean = KerasTensor([4])
        variance = KerasTensor([4])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=-1).shape,
            (10, 3, 4),
        )
        self.assertEqual(
            knn.batch_normalization(
                x, mean, variance, axis=-1, offset=KerasTensor([4]), scale=None
            ).shape,
            (10, 3, 4),
        )
        self.assertEqual(
            knn.batch_normalization(
                x, mean, variance, axis=-1, offset=None, scale=KerasTensor([4])
            ).shape,
            (10, 3, 4),
        )

        x = KerasTensor([10, 3, 4, 5])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=2).shape,
            (10, 3, 4, 5),
        )

        mean = KerasTensor([3])
        variance = KerasTensor([3])
        self.assertEqual(
            knn.batch_normalization(x, mean, variance, axis=1).shape,
            (10, 3, 4, 5),
        )

    def test_ctc_loss(self):
        x = KerasTensor([10, 3, 4])
        y = KerasTensor([10, 3], dtype="int32")
        x_lengths = KerasTensor([10], dtype="int32")
        y_lengths = KerasTensor([10], dtype="int32")
        self.assertEqual(knn.ctc_loss(x, y, x_lengths, y_lengths).shape, (10,))

    def test_ctc_decode(self):
        # Test strategy="greedy"
        inputs = KerasTensor([10, 2, 3])
        sequence_lengths = KerasTensor([10])
        decoded, scores = knn.ctc_decode(inputs, sequence_lengths)
        self.assertEqual(decoded.shape, (1, 10, 2))
        self.assertEqual(scores.shape, (10, 1))

        # Test strategy="beam_search"
        inputs = KerasTensor([10, 2, 3])
        sequence_lengths = KerasTensor([10])
        decoded, scores = knn.ctc_decode(
            inputs, sequence_lengths, strategy="beam_search", top_paths=2
        )
        self.assertEqual(decoded.shape, (2, 10, 2))
        self.assertEqual(scores.shape, (10, 2))

    def test_normalize(self):
        x = KerasTensor([1, 2, 3])
        self.assertEqual(knn.normalize(x).shape, (1, 2, 3))

    def test_psnr(self):
        x1 = KerasTensor([1, 2, 3])
        x2 = KerasTensor([5, 6, 7])
        out = knn.psnr(x1, x2, max_val=224)
        self.assertEqual(out.shape, ())

    def test_dot_product_attention(self):
        query = KerasTensor([2, 3, 8, 16])
        key = KerasTensor([2, 4, 6, 16])
        value = KerasTensor([2, 4, 6, 16])
        out = knn.dot_product_attention(query, key, value)
        self.assertEqual(out.shape, query.shape)

    def test_rms_normalization(self):
        x = KerasTensor([2, 8, 16])
        scale = KerasTensor([2, 8, 16])
        self.assertEqual(knn.rms_normalization(x, scale).shape, x.shape)

    def test_layer_normalization(self):
        x = KerasTensor([2, 8, 16])
        gamma = KerasTensor([2, 16])
        beta = KerasTensor([2, 16])
        self.assertEqual(knn.layer_normalization(x, gamma, beta).shape, x.shape)

    def test_polar(self):
        abs_ = KerasTensor([3, 4])
        angle = KerasTensor([3, 4])
        out = knn.polar(abs_, angle)
        self.assertEqual(out.shape, abs_.shape)
        self.assertEqual(out.dtype, "complex64")

        # `polar` broadcasts its two inputs against each other.
        out = knn.polar(KerasTensor([2, 1]), KerasTensor([1, 3]))
        self.assertEqual(out.shape, (2, 3))


class NNOpsCorrectnessTest(testing.TestCase):
    @pytest.mark.skipif(not testing.jax_uses_tpu(), reason="JAX on TPU only")
    def test_dot_product_attention_inside_scan(self):
        import jax
        import jax.numpy as jnp

        def attention_scan_body(carry, x):
            query, key, value = x
            # dot_product_attention expects 4D inputs (B, H, S, D)
            query = jnp.expand_dims(query, axis=0)
            key = jnp.expand_dims(key, axis=0)
            value = jnp.expand_dims(value, axis=0)

            # Use a mask to trigger the issue
            mask = jnp.ones((1, 4, 8), dtype="bool")
            out = knn.dot_product_attention(query, key, value, mask=mask)

            out = jnp.squeeze(out, axis=0)
            return carry, out

        query = jnp.ones((2, 1, 4, 8))
        key = jnp.ones((2, 1, 4, 8))
        value = jnp.ones((2, 1, 4, 8))

        # Scan over the first dimension
        _, out = jax.lax.scan(attention_scan_body, None, (query, key, value))
        self.assertEqual(out.shape, (2, 1, 4, 8))

    def test_relu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(knn.relu(x), [0, 0, 1, 2, 3])

    def test_relu6(self):
        x = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        self.assertAllClose(knn.relu6(x), [0, 0, 1, 2, 3, 4, 5, 6, 6])

    def test_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.sigmoid(x), [0.26894143, 0.5, 0.7310586, 0.880797, 0.95257413]
        )

    def test_sparse_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(knn.sparse_sigmoid(x), [0.0, 0.5, 1.0, 1.0, 1.0])
        # Integer input is promoted to float. The numpy backend previously
        # truncated the `0.5` midpoint to 0 under an integer dtype.
        x_int = np.array([-1, 0, 1, 2, 3], dtype="int32")
        self.assertAllClose(
            knn.sparse_sigmoid(x_int), [0.0, 0.5, 1.0, 1.0, 1.0]
        )

    def test_softplus(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.softplus(x),
            [0.31326166, 0.6931472, 1.3132616, 2.126928, 3.0485873],
        )

    def test_softsign(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(knn.softsign(x), [-0.5, 0, 0.5, 0.6666667, 0.75])

    def test_silu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.silu(x),
            [-0.26894143, 0, 0.7310586, 1.7615942, 2.8577223],
        )

    def test_log_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.log_sigmoid(x),
            [-1.3132616, -0.6931472, -0.31326166, -0.126928, -0.04858732],
        )

    def test_leaky_relu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.leaky_relu(x),
            [-0.2, 0, 1, 2, 3],
        )
        # Integer input is promoted to float. The numpy backend previously
        # truncated `negative_slope` to 0 under an integer dtype,
        # which turned this into `relu`.
        x_int = np.array([-1, 0, 1, 2, 3], dtype="int32")
        self.assertAllClose(knn.leaky_relu(x_int), [-0.2, 0, 1, 2, 3])

    def test_hard_sigmoid(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_sigmoid(x),
            [0.33333334, 0.5, 0.6666667, 0.8333334, 1.0],
        )
        # Integer input is promoted to float. The numpy backend previously
        # truncated the `0.5` offset to 0 under an integer dtype,
        # leaving `clip(x / 6, 0, 1)` rather than `clip(x / 6 + 0.5, 0, 1)`.
        x_int = np.array([-1, 0, 1, 2, 3], dtype="int32")
        self.assertAllClose(
            knn.hard_sigmoid(x_int),
            [0.33333334, 0.5, 0.6666667, 0.8333334, 1.0],
        )

    def test_hard_silu(self):
        x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_silu(x),
            [-0.0, -0.333333, -0.333333, 0.0, 0.6666667, 1.6666667, 3.0],
        )

    def test_elu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.elu(x),
            [-0.63212055, 0, 1, 2, 3],
        )
        self.assertAllClose(
            knn.elu(x, alpha=0.5),
            [-0.31606027, 0, 1, 2, 3],
        )

    def test_selu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.selu(x),
            [-1.1113307, 0.0, 1.050701, 2.101402, 3.152103],
        )
        # Integer input is promoted to float. Previously the numpy, tensorflow
        # and openvino backends cast `scale` to the input's integer dtype where
        # it truncated to 1 (turning this into a bare `elu`), and torch raised.
        x_int = np.array([-2, -1, 0, 1, 2], dtype="int32")
        self.assertAllClose(knn.selu(x_int), knn.selu(x_int.astype("float32")))
        self.assertEqual(
            standardize_dtype(
                knn.Selu().symbolic_call(KerasTensor((5,), "int32")).dtype
            ),
            standardize_dtype(knn.selu(x_int).dtype),
        )

    def test_gelu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.gelu(x),
            [-0.15880796, 0.0, 0.841192, 1.9545977, 2.9963627],
        )

    def test_celu(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.celu(x),
            [-0.63212055, 0.0, 1.0, 2.0, 3.0],
        )

    def test_glu(self):
        x = np.array([-1, 0, 1, 2, 3, 4], dtype=np.float32)
        self.assertAllClose(
            knn.glu(x),
            [-0.8807971, 0.0, 0.98201376],
        )

    def test_tanh_shrink(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.tanh_shrink(x),
            [-0.238406, 0.0, 0.238406, 1.035972, 2.004945],
        )

    def test_hard_tanh(self):
        x = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_tanh(x),
            [-1.0, 0.0, 1.0, 1.0, 1.0],
        )

    def test_hard_shrink(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.hard_shrink(x),
            [0.0, 0.0, 1.0, 2.0, 3.0],
        )

    def test_threshold(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.threshold(x, 0, 0),
            [0.0, 0.0, 1.0, 2.0, 3.0],
        )

    def test_squareplus(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.squareplus(x),
            [0.780776, 1.0, 1.618034, 2.414214, 3.302776],
        )

    def test_soft_shrink(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.soft_shrink(x),
            [0.0, 0.0, 0.5, 1.5, 2.5],
        )
        # Integer input is promoted to float. Previously the numpy, tensorflow
        # and openvino backends cast the fractional `threshold` shift back to
        # the input's integer dtype where it truncated, and torch raised.
        x_int = np.array([-2, -1, 0, 1, 2], dtype="int32")
        self.assertAllClose(knn.soft_shrink(x_int), [-1.5, -0.5, 0.0, 0.5, 1.5])
        self.assertEqual(
            standardize_dtype(
                knn.SoftShrink().symbolic_call(KerasTensor((5,), "int32")).dtype
            ),
            standardize_dtype(knn.soft_shrink(x_int).dtype),
        )

    def test_sparse_plus(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.sparse_plus(x),
            [0.0625, 0.25, 1.0, 2.0, 3.0],
        )
        # Integer input is promoted to float. Previously the numpy, tensorflow
        # and openvino backends cast the quadratic branch back to the input's
        # integer dtype where it truncated, and torch raised.
        x_int = np.array([-2, -1, 0, 1, 2], dtype="int32")
        self.assertAllClose(knn.sparse_plus(x_int), [0.0, 0.0, 0.25, 1.0, 2.0])
        self.assertEqual(
            standardize_dtype(
                knn.SparsePlus().symbolic_call(KerasTensor((5,), "int32")).dtype
            ),
            standardize_dtype(knn.sparse_plus(x_int).dtype),
        )

    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.softmax(x, axis=None),  # Reduce on all axes.
            [[0.045015, 0.122364, 0.33262], [0.045015, 0.122364, 0.33262]],
        )
        self.assertAllClose(
            knn.softmax(x, axis=0),
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        self.assertAllClose(
            knn.softmax(x, axis=-1),
            [
                [0.09003057, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
        )
        self.assertAllClose(
            knn.softmax(x),  # Default axis should be -1.
            [
                [0.09003057, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
        )

    def test_softmax_correctness_with_axis_tuple(self):
        input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        combination = combinations(range(3), 2)
        for axis in list(combination):
            result = keras.ops.nn.softmax(input, axis=axis)
            normalized_sum_by_axis = np.sum(
                ops.convert_to_numpy(result), axis=axis
            )
            self.assertAllClose(normalized_sum_by_axis, np.ones((2,)))

    def test_log_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.log_softmax(x, axis=None),  # Reduce on all axes.
            [
                [-3.100753, -2.100753, -1.100753],
                [-3.100753, -2.100753, -1.100753],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x, axis=0),
            [
                [-0.693147, -0.693147, -0.693147],
                [-0.693147, -0.693147, -0.693147],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x, axis=-1),
            [
                [-2.407606, -1.407606, -0.407606],
                [-2.407606, -1.407606, -0.407606],
            ],
        )
        self.assertAllClose(
            knn.log_softmax(x),  # Default axis should be -1.
            [
                [-2.407606, -1.407606, -0.407606],
                [-2.407606, -1.407606, -0.407606],
            ],
        )

    def test_log_softmax_correctness_with_axis_tuple(self):
        input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        combination = combinations(range(3), 2)
        for axis in list(combination):
            result = keras.ops.nn.log_softmax(input, axis=axis)
            normalized_sum_by_axis = np.sum(
                np.exp(ops.convert_to_numpy(result)), axis=axis
            )
            self.assertAllClose(normalized_sum_by_axis, np.ones((2,)))

    @pytest.mark.skipif(
        not backend.SUPPORTS_COMPLEX_DTYPES,
        reason=f"{backend.backend()} backend doesn't support complex dtypes.",
    )
    def test_polar_corectness(self):
        abs_ = np.array([1, 2], dtype="float32")
        angle = np.array([2, 3], dtype="float32")
        out = knn.polar(abs_, angle)
        self.assertAllClose(
            out, [-0.41614684 + 0.9092974j, -1.979985 + 0.28224j], atol=1e-3
        )
        # The symbolic dtype must match the eager one, which is complex.
        symbolic_out = knn.Polar().symbolic_call(
            KerasTensor((2,), dtype="float32"),
            KerasTensor((2,), dtype="float32"),
        )
        self.assertEqual(standardize_dtype(out.dtype), symbolic_out.dtype)

    def test_sparsemax(self):
        x = np.array([-0.5, 0, 1, 2, 3], dtype=np.float32)
        self.assertAllClose(
            knn.sparsemax(x),
            [0.0, 0.0, 0.0, 0.0, 1.0],
        )

        # The case above has a support of size one, where the threshold is
        # just the largest logit. Closely spaced logits keep more than one
        # coordinate in the support and exercise the threshold itself.
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        self.assertAllClose(knn.sparsemax(x), [0.0, 0.25, 0.75])

    def test_max_pool(self):
        data_format = backend.config.image_data_format()
        # Test 1D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            np_maxpool1d(x, 2, 1, padding="valid", data_format=data_format),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, 2, padding="same"),
            np_maxpool1d(x, 2, 2, padding="same", data_format=data_format),
        )

        # Test 2D max pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.max_pool(x, 2, 1, padding="valid"),
            np_maxpool2d(x, 2, 1, padding="valid", data_format=data_format),
        )
        self.assertAllClose(
            knn.max_pool(x, 2, (2, 1), padding="same"),
            np_maxpool2d(x, 2, (2, 1), padding="same", data_format=data_format),
        )

    def test_adaptive_max_pool(self):
        x = np.ones((2, 64, 64, 3), dtype="float32")
        y = knn.adaptive_max_pool(
            x, output_size=(32, 32), data_format="channels_last"
        )
        self.assertEqual(y.shape, (2, 32, 32, 3))

        x_cf = np.ones((2, 3, 64, 64), dtype="float32")
        y_cf = knn.adaptive_max_pool(
            x_cf, output_size=(32, 32), data_format="channels_first"
        )
        self.assertEqual(y_cf.shape, (2, 3, 32, 32))

    def test_adaptive_average_pool(self):
        x = np.ones((2, 64, 64, 3), dtype="float32")
        y = knn.adaptive_average_pool(
            x, output_size=(32, 32), data_format="channels_last"
        )
        self.assertEqual(y.shape, (2, 32, 32, 3))

        x_cf = np.ones((2, 3, 64, 64), dtype="float32")
        y_cf = knn.adaptive_average_pool(
            x_cf, output_size=(32, 32), data_format="channels_first"
        )
        self.assertEqual(y_cf.shape, (2, 3, 32, 32))

    def test_average_pool_valid_padding(self):
        data_format = backend.config.image_data_format()
        # Test 1D average pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            np_avgpool1d(x, 2, 1, padding="valid", data_format=data_format),
        )

        # Test 2D average pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, 1, padding="valid"),
            np_avgpool2d(x, 2, 1, padding="valid", data_format=data_format),
        )

    def test_average_pool_same_padding(self):
        data_format = backend.config.image_data_format()
        # Test 1D average pooling.
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        x = np.arange(120, dtype=float).reshape(input_shape)

        self.assertAllClose(
            knn.average_pool(x, 2, 2, padding="same"),
            np_avgpool1d(x, 2, 2, padding="same", data_format=data_format),
        )

        # Test 2D average pooling.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, 2, (2, 1), padding="same"),
            np_avgpool2d(x, 2, (2, 1), padding="same", data_format=data_format),
        )
        # Test 2D average pooling with different pool size.
        if data_format == "channels_last":
            input_shape = (2, 10, 9, 3)
        else:
            input_shape = (2, 3, 10, 9)
        x = np.arange(540, dtype=float).reshape(input_shape)
        self.assertAllClose(
            knn.average_pool(x, (2, 3), (3, 3), padding="same"),
            np_avgpool2d(
                x, (2, 3), (3, 3), padding="same", data_format=data_format
            ),
        )

    @parameterized.named_parameters(
        ("channels_last", "channels_last"),
        ("channels_first", "channels_first"),
    )
    def test_average_pool_same_padding_asymmetric(self, data_format):
        # Test 1D asymmetric padding (pool_size=3, strides=2)
        if data_format == "channels_last":
            x = np.array([[[10.0], [20.0]]], dtype="float32")
        else:
            x = np.array([[[10.0, 20.0]]], dtype="float32")

        res = knn.average_pool(
            x, pool_size=3, strides=2, padding="same", data_format=data_format
        )
        self.assertAllClose(res, np.array([15.0]).reshape(res.shape))

        # Test 2D asymmetric padding
        if data_format == "channels_last":
            x = np.array(
                [[[[10.0], [20.0]]]], dtype="float32"
            )  # shape (1, 1, 2, 1)
        else:
            x = np.array(
                [[[[10.0, 20.0]]]], dtype="float32"
            )  # shape (1, 1, 1, 2)
        res = knn.average_pool(
            x,
            pool_size=(1, 3),
            strides=(1, 2),
            padding="same",
            data_format=data_format,
        )
        self.assertAllClose(res, np.array([15.0]).reshape(res.shape))

    @parameterized.product(
        strides=(1, 2, 3),
        padding=("valid", "same"),
        dilation_rate=(1, 2),
        data_format=("channels_first", "channels_last"),
    )
    def test_conv_1d(self, strides, padding, dilation_rate, data_format):
        if data_format == "channels_last":
            input_shape = (2, 20, 3)
        else:
            input_shape = (2, 3, 20)
        inputs_1d = np.arange(120, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([4, 3, 2])

        outputs = knn.conv(
            inputs_1d,
            kernel,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            data_format=data_format,
        )
        expected = np_conv1d(
            inputs_1d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding.lower(),
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(
        strides=(1, 2, (1, 2)),
        padding=("valid", "same"),
        data_format=("channels_last", "channels_first"),
    )
    def test_conv_2d(self, strides, padding, data_format):
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.conv(
            inputs_2d, kernel, strides, padding=padding, data_format=data_format
        )
        expected = np_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(outputs, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    @pytest.mark.skipif(backend.backend() != "torch", reason="Torch only")
    def test_torch_channels_last_pointwise_conv_direct_path(self):
        from keras.src.backend.torch.ops import nn as torch_nn

        inputs_2d = np.arange(120, dtype="float32").reshape((2, 4, 5, 3))
        kernel = np.arange(6, dtype="float32").reshape((1, 1, 3, 2))

        with mock.patch.object(
            torch_nn.tnn,
            "conv2d",
            side_effect=AssertionError("conv2d should not be called"),
        ):
            outputs = knn.conv(
                inputs_2d,
                kernel,
                strides=(2, 3),
                padding="same",
                data_format="channels_last",
            )

        expected = np_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=(2, 3),
            padding="same",
            data_format="channels_last",
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(strides=(1, 2), dilation_rate=(1, (2, 1)))
    def test_conv_2d_group_2(self, strides, dilation_rate):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 10, 4)
        else:
            input_shape = (2, 4, 10, 10)
        inputs_2d = np.ones(input_shape)
        kernel = np.ones([2, 2, 2, 6])
        outputs = knn.conv(
            inputs_2d,
            kernel,
            strides,
            padding="same",
            dilation_rate=dilation_rate,
        )
        expected = np_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((6,)),
            strides=strides,
            padding="same",
            data_format=backend.config.image_data_format(),
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(
        strides=(1, (1, 1, 1), 2),
        padding=("valid", "same"),
        data_format=("channels_first", "channels_last"),
    )
    def test_conv_3d(self, strides, padding, data_format):
        if data_format == "channels_last":
            input_shape = (2, 8, 8, 8, 3)
        else:
            input_shape = (2, 3, 8, 8, 8)
        inputs_3d = np.arange(3072, dtype=float).reshape(input_shape)
        kernel = np.arange(162, dtype=float).reshape([3, 3, 3, 3, 2])

        outputs = knn.conv(
            inputs_3d, kernel, strides, padding=padding, data_format=data_format
        )
        expected = np_conv3d(
            inputs_3d,
            kernel,
            bias_weights=np.zeros((2,)),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(
            outputs,
            expected,
            rtol=1e-5,
            atol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

        # Test for tracing error on tensorflow backend.
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            @tf.function
            def conv(x):
                return knn.conv(
                    x, kernel, strides, padding=padding, data_format=data_format
                )

            outputs = conv(inputs_3d)
            self.assertAllClose(
                outputs,
                expected,
                rtol=1e-5,
                atol=1e-5,
                tpu_atol=1e-2,
                tpu_rtol=1e-2,
            )

    @parameterized.product(
        strides=(1, (1, 1), (2, 2)),
        padding=("valid", "same"),
        dilation_rate=(1, (2, 2)),
        data_format=("channels_first", "channels_last"),
    )
    def test_depthwise_conv_2d(
        self, strides, padding, dilation_rate, data_format
    ):
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])

        outputs = knn.depthwise_conv(
            inputs_2d,
            kernel,
            strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected = np_depthwise_conv2d(
            inputs_2d,
            kernel,
            bias_weights=np.zeros((6,)),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        self.assertAllClose(outputs, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    @parameterized.product(
        strides=(1, 2),
        padding=("valid", "same"),
        dilation_rate=(1, (2, 2)),
        data_format=("channels_first", "channels_last"),
    )
    def test_separable_conv_2d(
        self, strides, padding, dilation_rate, data_format
    ):
        # Test 2D conv.
        if data_format == "channels_last":
            input_shape = (2, 10, 10, 3)
        else:
            input_shape = (2, 3, 10, 10)
        inputs_2d = np.arange(600, dtype=float).reshape(input_shape)
        depthwise_kernel = np.arange(24, dtype=float).reshape([2, 2, 3, 2])
        pointwise_kernel = np.arange(72, dtype=float).reshape([1, 1, 6, 12])

        outputs = knn.separable_conv(
            inputs_2d,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        # Depthwise followed by pointwise conv
        expected_depthwise = np_depthwise_conv2d(
            inputs_2d,
            depthwise_kernel,
            np.zeros(6),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected = np_conv2d(
            expected_depthwise,
            pointwise_kernel,
            np.zeros(6 * 12),
            strides=1,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=1,
        )
        self.assertAllClose(outputs, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    @parameterized.product(
        strides=(1, 2),
        padding=("valid", "same"),
        dilation_rate=(1, 2),
        data_format=("channels_first", "channels_last"),
    )
    def test_depthwise_conv_1d(
        self, strides, padding, dilation_rate, data_format
    ):
        if data_format == "channels_last":
            input_shape = (2, 10, 3)
        else:
            input_shape = (2, 3, 10)
        inputs_1d = np.arange(60, dtype=float).reshape(input_shape)
        kernel = np.arange(12, dtype=float).reshape([2, 3, 2])
        if data_format == "channels_last":
            x2d = inputs_1d[:, None, :, :]
            squeeze_axis = 1
        else:
            x2d = inputs_1d[:, :, None, :]
            squeeze_axis = 2

        outputs = knn.depthwise_conv(
            inputs_1d,
            kernel,
            strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected_2d = np_depthwise_conv2d(
            x2d,
            kernel[None],
            bias_weights=np.zeros((6,)),
            strides=(1, strides),
            padding=padding,
            data_format=data_format,
            dilation_rate=(1, dilation_rate),
        )
        expected = np.squeeze(expected_2d, axis=squeeze_axis)
        self.assertAllClose(outputs, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    @parameterized.product(
        strides=(1, 2),
        padding=("valid", "same"),
        dilation_rate=(1, 2),
        data_format=("channels_first", "channels_last"),
    )
    def test_separable_conv_1d(
        self, strides, padding, dilation_rate, data_format
    ):
        if data_format == "channels_last":
            input_shape = (2, 10, 3)
        else:
            input_shape = (2, 3, 10)
        inputs_1d = np.arange(60, dtype=float).reshape(input_shape)
        depthwise_kernel = np.arange(12, dtype=float).reshape([2, 3, 2])
        pointwise_kernel = np.arange(72, dtype=float).reshape([1, 6, 12])
        if data_format == "channels_last":
            x2d = inputs_1d[:, None, :, :]
            squeeze_axis = 1
        else:
            x2d = inputs_1d[:, :, None, :]
            squeeze_axis = 2

        outputs = knn.separable_conv(
            inputs_1d,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        expected_dw_2d = np_depthwise_conv2d(
            x2d,
            depthwise_kernel[None],
            np.zeros((6,)),
            strides=(1, strides),
            padding=padding,
            data_format=data_format,
            dilation_rate=(1, dilation_rate),
        )
        expected_dw = np.squeeze(expected_dw_2d, axis=squeeze_axis)
        expected = np_conv1d(
            expected_dw,
            pointwise_kernel,
            np.zeros(12),
            strides=1,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            groups=1,
        )
        self.assertAllClose(outputs, expected, tpu_atol=1e-2, tpu_rtol=1e-2)

    @parameterized.product(padding=("valid", "same"))
    def test_conv_transpose_1d(self, padding):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 4, 3)
        else:
            input_shape = (2, 3, 4)
        inputs_1d = np.arange(24, dtype=float).reshape(input_shape)
        kernel = np.arange(30, dtype=float).reshape([2, 5, 3])
        outputs = knn.conv_transpose(inputs_1d, kernel, 2, padding=padding)
        expected = np_conv1d_transpose(
            inputs_1d,
            kernel,
            bias_weights=np.zeros(5),
            strides=2,
            output_padding=None,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=1,
        )
        self.assertAllClose(outputs, expected)

    @parameterized.product(strides=(2, (2, 2)), padding=("valid", "same"))
    def test_conv_transpose_2d(self, strides, padding):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 4, 4, 3)
        else:
            input_shape = (2, 3, 4, 4)
        inputs_2d = np.arange(96, dtype=float).reshape(input_shape)
        kernel = np.arange(60, dtype=float).reshape([2, 2, 5, 3])

        outputs = knn.conv_transpose(
            inputs_2d, kernel, strides, padding=padding
        )
        expected = np_conv2d_transpose(
            inputs_2d,
            kernel,
            bias_weights=np.zeros(5),
            strides=strides,
            output_padding=None,
            padding=padding,
            data_format=backend.config.image_data_format(),
            dilation_rate=1,
        )
        self.assertAllClose(outputs, expected)

    def test_conv_transpose_functional_model(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (4, 4, 3)
            kernel_shape = (2, 2, 5, 3)
            inputs_data = np.arange(96, dtype="float32").reshape((2, 4, 4, 3))
        else:
            input_shape = (3, 4, 4)
            kernel_shape = (2, 2, 5, 3)
            inputs_data = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))
        kernel_data = np.arange(60, dtype="float32").reshape(kernel_shape)

        x = keras.layers.Input(shape=input_shape)
        kernel = keras.ops.convert_to_tensor(kernel_data)
        out = knn.conv_transpose(
            x, kernel, strides=2, padding="valid", output_padding=1
        )
        model = keras.Model(inputs=x, outputs=out)
        model_output = model(inputs_data)
        eager_output = knn.conv_transpose(
            inputs_data,
            kernel_data,
            strides=2,
            padding="valid",
            output_padding=1,
        )
        self.assertAllClose(model_output, eager_output)

    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    def test_one_hot(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")
        # Test 1D one-hot.
        indices_1d = np.array([0, 1, 2, 3])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)

        # Test 1D list one-hot.
        indices_1d = [0, 1, 2, 3]
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)

        # Test 2D one-hot.
        indices_2d = np.array([[0, 1], [2, 3]])
        output_2d = knn.one_hot(indices_2d, 4, sparse=sparse)
        self.assertAllClose(output_2d, np.eye(4)[indices_2d])
        self.assertSparse(output_2d, sparse)
        output_2d = knn.one_hot(indices_2d, 4, axis=2, sparse=sparse)
        self.assertAllClose(output_2d, np.eye(4)[indices_2d])
        self.assertSparse(output_2d, sparse)
        output_2d = knn.one_hot(indices_2d, 4, axis=1, sparse=sparse)
        self.assertAllClose(
            output_2d, np.transpose(np.eye(4)[indices_2d], (0, 2, 1))
        )
        self.assertSparse(output_2d, sparse)
        output_2d = knn.one_hot(indices_2d, 4, axis=-2, sparse=sparse)
        self.assertAllClose(
            output_2d, np.transpose(np.eye(4)[indices_2d], (0, 2, 1))
        )
        self.assertSparse(output_2d, sparse)

        # Test 1D one-hot with 1 extra dimension.
        indices_1d = np.array([[0], [1], [2], [3]])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d])
        self.assertSparse(output_1d, sparse)
        output_1d = knn.one_hot(indices_1d, 4, axis=0, sparse=sparse)
        self.assertAllClose(output_1d, np.eye(4)[indices_1d].swapaxes(1, 2))
        self.assertSparse(output_1d, sparse)

        # Test 1D one-hot with negative inputs
        indices_1d = np.array([0, -1, -1, 3])
        output_1d = knn.one_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(
            output_1d,
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
        )
        self.assertSparse(output_1d, sparse)

    def test_binary_crossentropy(self):
        # Test with from_logits=False
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.1], [0.2], [0.3], [0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=False)
        self.assertAllClose(
            result,
            np.array([[0.32508277], [1.47080801], [0.52613434], [0.91629048]]),
        )

        # Test with from_logits=True
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.1], [0.2], [0.3], [0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=True)
        self.assertAllClose(
            result,
            np.array([[0.73439666], [0.61813887], [0.79435524], [0.51301525]]),
        )

        # Test with output clipping
        target = np.array([[0.1], [0.9], [0.2], [1.0]])
        output = np.array([[0.99], [-0.2], [0.9], [-0.4]])
        result = knn.binary_crossentropy(target, output, from_logits=True)
        self.assertAllClose(
            result,
            np.array([[1.206961], [0.778139], [1.061154], [0.913015]]),
        )

    @parameterized.product(
        shape=[(4, 1), (4, 5, 1), (4, 1, 1), (4, 5, 1, 1), (2, 1, 3)],
        from_logits=[True, False],
    )
    def test_binary_crossentropy_size_one_dims(self, shape, from_logits):
        rng = np.random.default_rng(0)
        target = rng.integers(0, 2, size=shape).astype("float32")
        output = rng.uniform(0.1, 0.9, size=shape).astype("float32")
        result = knn.binary_crossentropy(
            target, output, from_logits=from_logits
        )
        probs = 1.0 / (1.0 + np.exp(-output)) if from_logits else output
        expected = -(target * np.log(probs) + (1 - target) * np.log(1 - probs))
        self.assertEqual(tuple(result.shape), shape)
        self.assertAllClose(result, expected)

    @parameterized.product(
        target_dtype=["int32", "bool", "float32"],
        output_dtype=["float32", "float16", "bfloat16"],
        from_logits=[True, False],
    )
    def test_binary_crossentropy_target_dtype(
        self, target_dtype, output_dtype, from_logits
    ):
        target = np.array([0, 1, 1, 0]).astype(target_dtype)
        output = np.array([0.1, 0.9, 0.8, 0.2]).astype(output_dtype)
        result = knn.binary_crossentropy(
            target, output, from_logits=from_logits
        )
        t = target.astype("float64")
        o = output.astype("float64")
        probs = 1.0 / (1.0 + np.exp(-o)) if from_logits else o
        expected = -(t * np.log(probs) + (1 - t) * np.log(1 - probs))
        tol = {"float32": 1e-6, "float16": 1e-3, "bfloat16": 1e-2}[output_dtype]
        self.assertAllClose(result, expected, atol=tol, rtol=tol)
        self.assertEqual(standardize_dtype(result.dtype), output_dtype)

    @parameterized.parameters(
        ("int32", True), ("int32", False), ("uint8", False), ("bool", False)
    )
    def test_binary_crossentropy_non_float_output(
        self, output_dtype, from_logits
    ):
        target = np.array([0.0, 1.0, 1.0, 0.0], "float32")
        output = np.array([0, 1, 1, 0]).astype(output_dtype)
        result = knn.binary_crossentropy(
            target, output, from_logits=from_logits
        )
        o = output.astype("float64")
        probs = 1.0 / (1.0 + np.exp(-o)) if from_logits else o
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        expected = -(target * np.log(probs) + (1 - target) * np.log(1 - probs))
        self.assertAllClose(result, expected, atol=1e-6, rtol=1e-6)
        symbolic = knn.binary_crossentropy(
            KerasTensor((4,), "float32"), KerasTensor((4,), output_dtype)
        )
        self.assertEqual(symbolic.dtype, backend.floatx())

    @parameterized.product(
        target_dtype=["int32", "bool", "float32"],
        output_dtype=["float16", "bfloat16", "float32"],
        from_logits=[True, False],
    )
    def test_binary_crossentropy_dtype(
        self, target_dtype, output_dtype, from_logits
    ):
        target = np.ones((2,), dtype=target_dtype)
        output = np.full((2,), 0.5, dtype=output_dtype)
        result = knn.binary_crossentropy(
            target, output, from_logits=from_logits
        )
        self.assertEqual(standardize_dtype(result.dtype), output_dtype)

    def test_categorical_crossentropy(self):
        target = np.array(
            [
                [0.33008796, 0.0391289, 0.9503603],
                [0.80376694, 0.92363342, 0.19147756],
            ]
        )
        output = np.array(
            [
                [0.23446431, 0.35822914, 0.06683268],
                [0.3413979, 0.05420256, 0.81619654],
            ]
        )

        # Test from_logits=False
        result = knn.categorical_crossentropy(
            target, output, from_logits=False, axis=-1
        )
        self.assertAllClose(result, np.array([2.54095299, 3.96374412]))

        # Test axis
        result = knn.categorical_crossentropy(
            target, output, from_logits=False, axis=0
        )
        self.assertAllClose(
            result, np.array([0.71683073, 1.87988172, 2.46810762])
        )

        # Test from_logits=True
        result = knn.categorical_crossentropy(
            target, output, from_logits=True, axis=-1
        )
        self.assertAllClose(result, np.array([1.59419954, 2.49880593]))

        # Test with output clipping
        output = np.array(
            [
                [1.23446431, -0.35822914, 1.06683268],
                [0.3413979, -0.05420256, 0.81619654],
            ]
        )
        result = knn.categorical_crossentropy(
            target, output, from_logits=True, axis=-1
        )
        self.assertAllClose(result, np.array([1.16825923, 2.55436813]))

    def test_sparse_categorical_crossentropy(self):
        target = np.array([0, 1, 2])
        output = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.89, 0.06], [0.05, 0.01, 0.94]]
        )
        result = knn.sparse_categorical_crossentropy(target, output)
        self.assertAllClose(result, [0.105361, 0.116534, 0.061875])

        output = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        result = knn.sparse_categorical_crossentropy(
            target, output, from_logits=True
        )
        self.assertAllClose(result, [0.001822, 0.000459, 0.169846])

    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    def test_multi_hot(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")

        # Test 1D multi-hot.
        indices_1d = np.array([0, 1, 2, 3])
        expected_output_1d = np.array([1, 1, 1, 1])
        output_1d = knn.multi_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, expected_output_1d)
        self.assertSparse(output_1d, sparse)

        # Test 2D multi-hot.
        indices_2d = np.array([[0, 1], [2, 3]])
        expected_output_2d = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        output_2d = knn.multi_hot(indices_2d, 4, sparse=sparse)
        self.assertAllClose(output_2d, expected_output_2d)
        self.assertSparse(output_2d, sparse)

        # Test 1D multi-hot with negative inputs
        indices_1d = np.array([0, -1, -1, 3])
        expected_output_1d = np.array([1, 0, 0, 1])
        output_1d = knn.multi_hot(indices_1d, 4, sparse=sparse)
        self.assertAllClose(output_1d, expected_output_1d)
        self.assertSparse(output_1d, sparse)

    def test_moments(self):
        # Test 1D moments
        x = np.array([0, 1, 2, 3, 4, 100, -200]).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0])
        self.assertAllClose(mean, np.mean(x), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x), atol=1e-5, rtol=1e-5)

        # Test batch statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0])
        self.assertAllClose(mean, np.mean(x, axis=0), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x, axis=0), atol=1e-5, rtol=1e-5)

        # Test global statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2])
        expected_mean = np.mean(x, axis=(0, 1, 2))
        expected_variance = np.var(x, axis=(0, 1, 2))
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test keepdims
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2], keepdims=True)
        expected_mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        expected_variance = np.var(x, axis=(0, 1, 2), keepdims=True)
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test float16 which causes overflow
        x = np.array(
            [-741.0, 353.2, 1099.0, -1807.0, 502.8, -83.4, 333.5, -130.9],
            dtype=np.float16,
        )
        mean, variance = knn.moments(x, axes=[0])
        expected_mean = np.mean(x.astype(np.float32)).astype(np.float16)
        # the output variance is clipped to the max value of np.float16 because
        # it is overflowed
        expected_variance = np.finfo(np.float16).max
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="synchronized=True only implemented for TF backend",
    )
    def test_moments_sync(self):
        # Test batch statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0], synchronized=True)
        self.assertAllClose(mean, np.mean(x, axis=0), atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, np.var(x, axis=0), atol=1e-5, rtol=1e-5)

        # Test global statistics for 4D moments (batch, height, width, channels)
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(x, axes=[0, 1, 2], synchronized=True)
        expected_mean = np.mean(x, axis=(0, 1, 2))
        expected_variance = np.var(x, axis=(0, 1, 2))
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

        # Test keepdims
        x = np.random.uniform(size=(2, 28, 28, 3)).astype(np.float32)
        mean, variance = knn.moments(
            x, axes=[0, 1, 2], keepdims=True, synchronized=True
        )
        expected_mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        expected_variance = np.var(x, axis=(0, 1, 2), keepdims=True)
        self.assertAllClose(mean, expected_mean, atol=1e-5, rtol=1e-5)
        self.assertAllClose(variance, expected_variance, atol=1e-5, rtol=1e-5)

    @parameterized.product(dtype=["float16", "float32"])
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="synchronized=True only implemented for TF backend",
    )
    def test_moments_sync_with_distribution_strategy(self, dtype):
        from tensorflow.python.eager import context

        from keras.src.utils.module_utils import tensorflow as tf

        context._reset_context()

        # Config 2 CPUs for testing.
        logical_cpus = tf.config.list_logical_devices("CPU")
        if len(logical_cpus) == 1:
            from tensorflow.python.eager import context

            context._reset_context()
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices("CPU")[0],
                [
                    tf.config.LogicalDeviceConfiguration(),
                    tf.config.LogicalDeviceConfiguration(),
                ],
            )

        @tf.function()
        def test_on_moments(inputs):
            return knn.moments(
                inputs, axes=-1, keepdims=True, synchronized=True
            )

        # Test output of moments.
        inputs = tf.constant([5.0, 9.0, 1.0, 3.0], dtype=dtype)
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        with strategy.scope():
            mean, variance = strategy.run(test_on_moments, args=(inputs,))
            self.assertEqual(mean.values[0], 4.5)
            self.assertEqual(variance.values[0], 8.75)
            self.assertEqual(variance.values[0], 8.75)

        context._reset_context()

    def test_batch_normalization(self):
        x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mean = np.array([0.2, 0.3, 0.4])
        variance = np.array([4.0, 16.0, 64.0])
        output = knn.batch_normalization(
            x,
            mean,
            variance,
            axis=-1,
            offset=np.array([5.0, 10.0, 15.0]),
            scale=np.array([10.0, 20.0, 30.0]),
            epsilon=1e-7,
        )
        expected_output = np.array([[4.5, 9.5, 14.625], [6.0, 11.0, 15.75]])
        self.assertAllClose(output, expected_output)

        output = knn.batch_normalization(
            x,
            mean,
            variance,
            axis=1,
            epsilon=1e-7,
        )
        expected_output = np.array(
            [[-0.05, -0.025, -0.0125], [0.1, 0.05, 0.025]]
        )
        self.assertAllClose(output, expected_output)

        output = knn.batch_normalization(
            np.random.uniform(size=[2, 3, 3, 5]),
            np.random.uniform(size=[5]),
            np.random.uniform(size=[5]),
            axis=3,
            offset=np.random.uniform(size=[5]),
            scale=np.random.uniform(size=[5]),
        )
        self.assertEqual(tuple(output.shape), (2, 3, 3, 5))

    def test_ctc_loss(self):
        labels = np.array([[1, 2, 1], [1, 2, 2]])
        outputs = np.array(
            [
                [[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]],
                [[0.4, 0.8, 0.4], [0.2, 0.3, 0.3], [0.4, 0.3, 0.2]],
            ]
        )

        label_length = np.array([3, 2])
        output_length = np.array([3, 2])

        result = knn.ctc_loss(labels, outputs, label_length, output_length)
        self.assertAllClose(
            result,
            np.array([3.4411672, 1.91680186]),
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    def test_ctc_decode(self):
        inputs = np.array(
            [
                [
                    [0.1, 0.4, 0.2, 0.4],
                    [0.3, -0.3, 0.4, 0.2],
                    [0.3, 0.2, 0.4, 0.3],
                ],
                [
                    [0.7, 0.4, 0.3, 0.2],
                    [0.3, 0.3, 0.4, 0.1],
                    [0.6, -0.1, 0.1, 0.5],
                ],
                [
                    [0.1, 0.4, 0.2, 0.7],
                    [0.3, 0.3, -0.2, 0.7],
                    [0.3, 0.2, 0.4, 0.1],
                ],
            ]
        )
        labels = np.array([[1, 2, -1], [2, -1, -1], [3, -1, -1]])
        score_labels = np.array([[-1.2], [-1.7], [-0.7]])
        repeated_labels = np.array([[1, 2, 2], [2, -1, -1], [3, -1, -1]])

        # Test strategy="greedy" and merge_repeated=True
        (decoded,), scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="greedy",
            mask_index=0,
        )
        self.assertAllClose(decoded, labels)
        self.assertAllClose(scores, score_labels)

        # Test strategy="greedy" and merge_repeated=False
        (decoded,), scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="greedy",
            merge_repeated=False,
            mask_index=0,
        )
        self.assertAllClose(decoded, repeated_labels)
        self.assertAllClose(scores, score_labels)

        labels = np.array(
            [
                [[1, 2, -1], [2, -1, -1], [3, -1, -1]],
                [[2, -1, -1], [3, -1, -1], [1, -1, -1]],
            ]
        )
        score_labels = np.array(
            [
                [-2.426537, -2.435596],
                [-2.127681, -2.182338],
                [-1.063386, -1.363386],
            ]
        )
        beam_width = 4
        top_paths = 2

        # Test strategy="beam_search"
        decoded, scores = knn.ctc_decode(
            inputs,
            sequence_lengths=[3, 3, 1],
            strategy="beam_search",
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=0,
        )
        self.assertAllClose(decoded, labels)
        self.assertAllClose(scores, score_labels)

    def test_normalize(self):
        x = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.assertAllClose(
            knn.normalize(x, axis=None),
            [
                [0.18898225, 0.3779645, 0.56694674],
                [0.18898225, 0.3779645, 0.56694674],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, axis=0),
            [
                [0.70710677, 0.70710677, 0.70710677],
                [0.70710677, 0.70710677, 0.70710677],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, axis=-1),
            [
                [0.26726124, 0.53452247, 0.8017837],
                [0.26726124, 0.53452247, 0.8017837],
            ],
        )
        self.assertAllClose(
            knn.normalize(x, order=3),
            [
                [0.30285344, 0.6057069, 0.9085603],
                [0.30285344, 0.6057069, 0.9085603],
            ],
        )

        # linalg.norm(x, ...) < epsilon
        x = np.array([[1e-6, 1e-8]], dtype=np.float32)
        self.assertAllClose(
            knn.normalize(x, axis=-1, order=2, epsilon=1e-5),
            [[1e-1, 1e-3]],
        )

    @pytest.mark.skipif(
        not backend.SUPPORTS_GRADIENT,
        reason="Backend does not support gradients.",
    )
    def test_normalize_l2_zero_vector_gradients(self):
        # The L2 (order=2) fast path must not produce NaN gradients for a zero
        # vector: rsqrt(0) is inf and its derivative is 0 * inf = NaN, which a
        # clamp on the output cannot undo (see #23075). At the data minimum the
        # clamped norm is constant, so the gradient is a finite 1 / epsilon.
        epsilon = 1e-3
        expected_grad = np.full((3,), 1.0 / epsilon, dtype="float32")

        def f(x):
            return knn.normalize(x, axis=-1, order=2, epsilon=epsilon)

        x_grad = ops.grad(f)(ops.zeros((3,)))

        x_grad = ops.convert_to_numpy(x_grad)
        self.assertFalse(np.isnan(x_grad).any())
        self.assertAllClose(x_grad, expected_grad)

    def test_psnr(self):
        x1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x2 = np.array([[0.2, 0.2, 0.3], [0.4, 0.6, 0.6]])
        max_val = 1.0
        expected_psnr_1 = 20 * np.log10(max_val) - 10 * np.log10(
            np.mean(np.square(x1 - x2))
        )
        psnr_1 = knn.psnr(x1, x2, max_val)
        self.assertAlmostEqual(psnr_1, expected_psnr_1)

        x3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x4 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        max_val = 1.0
        expected_psnr_2 = 20 * np.log10(max_val) - 10 * np.log10(
            np.mean(np.square(x3 - x4))
        )
        psnr_2 = knn.psnr(x3, x4, max_val)
        self.assertAlmostEqual(psnr_2, expected_psnr_2)

    @parameterized.named_parameters(
        named_product(
            bias=(None, True),
            scale=(None, 1.0),
            mask_and_is_causal=((None, False), (True, False), (None, True)),
            flash_attention=(None, True, False),
        )
    )
    def test_dot_product_attention(
        self, bias, scale, mask_and_is_causal, flash_attention
    ):
        mask, is_causal = mask_and_is_causal
        query_shape = (2, 3, 4, 8)
        key_shape = (2, 3, 4, 8)
        bias_shape = (2, 4, 3, 3)
        query = np.arange(math.prod(query_shape), dtype=float).reshape(
            query_shape
        )
        key = np.arange(math.prod(key_shape), dtype=float).reshape(key_shape)
        value = np.arange(math.prod(key_shape), dtype=float).reshape(key_shape)
        if mask is not None:
            mask = np.tril(np.ones((3, 3))).astype("bool")
            mask = mask[None, None, ...]
            mask = np.tile(mask, (2, 4, 1, 1))
        if bias is not None:
            if backend.backend() == "torch" and mask is not None:
                self.skipTest(
                    "torch does not support `mask` and `bias` with "
                    "`dot_product_attention`"
                )
            bias = np.arange(math.prod(bias_shape), dtype=float).reshape(
                bias_shape
            )

        if flash_attention:
            if backend.backend() == "numpy":
                self.skipTest(
                    "Flash attention is not supported in numpy backend."
                )
            elif backend.backend() == "tensorflow":
                import tensorflow as tf

                if not hasattr(tf.nn, "flash_attention"):
                    self.skipTest(
                        "This TensorFlow build doesn't have "
                        "`tf.nn.flash_attention`."
                    )
                if bias is not None or mask is not None:
                    with self.assertRaisesRegex(ValueError, "not supported"):
                        knn.dot_product_attention(
                            query,
                            key,
                            value,
                            bias=bias,
                            mask=mask,
                            flash_attention=True,
                        )
                    return
            elif backend.backend() == "torch":
                import torch

                if bias is not None:
                    self.skipTest(
                        "Flash attention doesn't support `bias` in torch "
                        "backend."
                    )
                if mask is not None:
                    self.skipTest(
                        "Flash attention doesn't support `mask=None` in torch "
                        "backend."
                    )
                if not torch.cuda.is_available():
                    self.skipTest(
                        "Flash attention must be run on CUDA in torch backend."
                    )
                cuda_compute_capability = tuple(
                    int(x) for x in torch.cuda.get_device_capability()
                )
                if cuda_compute_capability < (8, 0):
                    self.skipTest(
                        "Flash attention must be run on CUDA compute "
                        "capability >= 8.0 in torch backend."
                    )
            elif backend.backend() == "jax":
                import jax
                from jax._src import xla_bridge

                if "cuda" not in xla_bridge.get_backend().platform_version:
                    self.skipTest(
                        "Flash attention must be run on CUDA in jax backend."
                    )
                d, *_ = jax.local_devices(backend="gpu")
                cuda_compute_capability = tuple(
                    int(x) for x in d.compute_capability.split(".")
                )
                if cuda_compute_capability < (8, 0):
                    self.skipTest(
                        "Flash attention must be run on CUDA compute "
                        "capability >= 8.0 in jax backend."
                    )

            # Flash attention only supports float16 and bfloat16. We multiply
            # 0.1 to avoid overflow.
            query = (query * 0.1).astype("float16")
            key = (key * 0.1).astype("float16")
            value = (value * 0.1).astype("float16")
            if bias is not None:
                bias = (bias * 0.1).astype("float16")

        outputs = knn.dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
            is_causal=is_causal,
            flash_attention=flash_attention,
        )

        expected = _dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
            is_causal=is_causal,
        )
        self.assertAllClose(
            outputs, expected, atol=1e-3 if flash_attention else 1e-6
        )

    @parameterized.named_parameters(named_product(scale=(1.0, 10.0)))
    def test_rms_normalization(self, scale):
        x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        scale = np.array([scale] * x.shape[-1], dtype="float32")
        expected_output = (
            np.array([[0.46291, 0.92582, 1.38873], [0.78954, 0.98693, 1.18431]])
            * scale
        )

        self.assertAllClose(
            knn.rms_normalization(x, scale), expected_output, atol=1e-3
        )
        self.assertAllClose(knn.RMSNorm()(x, scale), expected_output, atol=1e-3)

    def test_layer_normalization(self):
        x = np.arange(5, dtype="float32")
        expected_output = np.array(
            [-1.4142135, -0.70710677, 0.0, 0.7071067, 1.4142135]
        )

        self.assertAllClose(
            knn.layer_normalization(x), expected_output, atol=1e-3
        )
        self.assertAllClose(knn.LayerNorm()(x), expected_output, atol=1e-3)

    @parameterized.named_parameters(
        named_product(
            axis=[[0, 1], [1, 2], [1, 2, 3], [0, 2]],
            backend_agnostic_ops=[False, True],
        )
    )
    def test_normalization_over_axes(self, axis, backend_agnostic_ops):
        # A backend kernel normalizes over trailing axes, so any other set of
        # axes exercises the backend moving them into place.
        backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
        self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
        x = np.arange(120, dtype="float32").reshape((2, 3, 4, 5)) / 120.0
        axes = tuple(axis)
        epsilon = 1e-5

        expected = x / np.sqrt(
            np.mean(np.square(x), axis=axes, keepdims=True) + epsilon
        )
        self.assertAllClose(
            knn.rms_normalization(x, axis=axis, epsilon=epsilon),
            expected,
            atol=1e-5,
        )

        mean = np.mean(x, axis=axes, keepdims=True)
        variance = np.var(x, axis=axes, keepdims=True)
        expected = (x - mean) / np.sqrt(variance + epsilon)
        self.assertAllClose(
            knn.layer_normalization(x, axis=axis, epsilon=epsilon),
            expected,
            atol=1e-5,
        )


class NNOpsDtypeTest(testing.TestCase):
    """Test the floating dtype to verify that the behavior matches JAX."""

    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    if backend.backend() == "torch":
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]
    elif backend.backend() == "tensorflow":
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint32",)]

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_elu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.elu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.elu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Elu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_gelu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # approximate = True
        expected_dtype = standardize_dtype(jnn.gelu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.gelu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Gelu().symbolic_call(x).dtype),
            expected_dtype,
        )
        # approximate = False
        expected_dtype = standardize_dtype(jnn.gelu(x_jax, False).dtype)

        self.assertEqual(
            standardize_dtype(knn.gelu(x, False).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Gelu(False).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_celu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.celu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.celu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Celu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_tanh_shrink(self, dtype):
        import torch
        import torch.nn.functional as tnn

        x = knp.ones((1), dtype=dtype)
        x_torch = torch.ones(1, dtype=getattr(torch, dtype))
        expected_dtype = standardize_dtype(tnn.tanhshrink(x_torch).dtype)

        self.assertEqual(
            standardize_dtype(knn.tanh_shrink(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.TanhShrink().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_hard_tanh(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_tanh(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_tanh(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardTanh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_hard_shrink(self, dtype):
        import torch
        import torch.nn.functional as tnn

        x = knp.ones((1), dtype=dtype)
        x_torch = torch.ones(1, dtype=getattr(torch, dtype))
        expected_dtype = standardize_dtype(tnn.hardshrink(x_torch).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_shrink(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardShrink().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_threshold(self, dtype):
        import torch
        import torch.nn.functional as tnn

        x = knp.ones((1), dtype=dtype)
        x_torch = torch.ones(1, dtype=getattr(torch, dtype))
        expected_dtype = standardize_dtype(tnn.threshold(x_torch, 0, 0).dtype)

        self.assertEqual(
            standardize_dtype(knn.threshold(x, 0, 0).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Threshold(0, 0).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_soft_shrink(self, dtype):
        import torch
        import torch.nn.functional as tnn

        x = knp.ones((1), dtype=dtype)
        x_torch = torch.ones(1, dtype=getattr(torch, dtype))
        expected_dtype = standardize_dtype(tnn.softshrink(x_torch).dtype)

        self.assertEqual(
            standardize_dtype(knn.soft_shrink(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.SoftShrink().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_sparse_plus(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.sparse_plus(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.sparse_plus(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.SparsePlus().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_glu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((2), dtype=dtype)
        x_jax = jnp.ones((2), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.glu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.glu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Glu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_squareplus(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((2), dtype=dtype)
        x_jax = jnp.ones((2), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.squareplus(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.squareplus(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Squareplus().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtype=FLOAT_DTYPES + INT_DTYPES + ["bool"])
    )
    def test_hard_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardSigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtype=FLOAT_DTYPES + INT_DTYPES + ["bool"])
    )
    def test_hard_silu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_silu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_silu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardSilu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtype=FLOAT_DTYPES + INT_DTYPES + ["bool"])
    )
    def test_leaky_relu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.leaky_relu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.leaky_relu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LeakyRelu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_log_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.log_sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.log_sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LogSigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_log_softmax(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((10,), dtype=dtype)
        x_jax = jnp.ones((10,), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.log_softmax(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.log_softmax(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.LogSoftmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_relu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.relu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.relu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Relu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_relu6(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.relu6(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.relu6(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Relu6().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_selu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.selu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.selu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Selu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Sigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtype=FLOAT_DTYPES + INT_DTYPES + ["bool"])
    )
    def test_sparse_sigmoid(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.sparse_sigmoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.sparse_sigmoid(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.SparseSigmoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_sparsemax(self, dtype):
        # `jax.nn` has no `sparsemax`, so there is no reference to compare
        # against. `sparsemax` is a projection onto the simplex and returns the
        # dtype it is given. The ranks and counts inside the implementation are
        # integers and the constants are Python floats; neither may promote the
        # result away from that dtype.
        x = knp.ones((2,), dtype=dtype)
        expected_dtype = dtype

        self.assertEqual(
            standardize_dtype(knn.sparsemax(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Sparsemax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_silu(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.silu(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.silu(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Silu().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softplus(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.softplus(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softplus(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softplus().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softmax(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((10,), dtype=dtype)
        x_jax = jnp.ones((10,), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.softmax(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softmax(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_softsign(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.soft_sign(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.softsign(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.Softsign().symbolic_call(x).dtype),
            expected_dtype,
        )

    @pytest.mark.skipif(
        not backend.SUPPORTS_COMPLEX_DTYPES,
        reason=f"{backend.backend()} backend doesn't support complex dtypes.",
    )
    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_polar(self, dtype):
        import jax.nn as jnn
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnn.hard_tanh(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knn.hard_tanh(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knn.HardTanh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_ctc_loss(self, dtype):
        labels = knp.array([[1, 2, 1]], dtype="int32")
        outputs = knp.array(
            [[[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]]], dtype=dtype
        )
        label_length = knp.array([3])
        output_length = knp.array([3])
        expected_dtype = (
            "float32" if dtype in ("float16", "bfloat16") else dtype
        )

        self.assertEqual(
            standardize_dtype(
                knn.ctc_loss(labels, outputs, label_length, output_length).dtype
            ),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knn.CTCLoss()
                .symbolic_call(labels, outputs, label_length, output_length)
                .dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_ctc_decode(self, dtype):
        inputs = knp.array(
            [[[0.4, 0.8, 0.4], [0.2, 0.8, 0.3], [0.9, 0.4, 0.5]]], dtype=dtype
        )
        sequence_length = knp.array([3])
        expected_dtype = backend.result_type(dtype, "float32")

        # Test strategy="greedy"
        decoded, scores = knn.ctc_decode(
            inputs, sequence_length, strategy="greedy"
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)
        decoded, scores = knn.CTCDecode(strategy="greedy").symbolic_call(
            inputs, sequence_length
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)

        # Test strategy="beam_search"
        decoded, scores = knn.ctc_decode(
            inputs, sequence_length, strategy="beam_search"
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)
        decoded, scores = knn.CTCDecode(strategy="beam_search").symbolic_call(
            inputs, sequence_length
        )
        self.assertEqual(standardize_dtype(decoded.dtype), "int32")
        self.assertEqual(standardize_dtype(scores.dtype), expected_dtype)

    @parameterized.named_parameters(
        named_product(
            dtypes=list(combinations(FLOAT_DTYPES, 2))
            + [(dtype, dtype) for dtype in FLOAT_DTYPES]
        )
    )
    def test_dot_product_attention(self, dtypes):
        # TODO: Get expected output from jax if `jax.nn.dot_product_attention`
        # is available.
        query_dtype, key_value_dtype = dtypes
        query = knp.ones((2, 3, 3, 8), dtype=query_dtype)
        key = knp.ones((2, 3, 3, 8), dtype=key_value_dtype)
        value = knp.ones((2, 3, 3, 8), dtype=key_value_dtype)
        expected_dtype = backend.result_type(*dtypes)

        self.assertDType(
            knn.dot_product_attention(query, key, value), expected_dtype
        )
        self.assertDType(
            knn.DotProductAttention().symbolic_call(query, key, value),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=combinations(FLOAT_DTYPES, 2))
    )
    def test_rms_normalization(self, dtypes):
        input_dtype, weight_dtype = dtypes
        inputs = knp.ones((2, 8), dtype=input_dtype)
        scale = backend.Variable(knp.ones((8,), dtype=weight_dtype))
        expected_dtype = input_dtype

        self.assertDType(knn.rms_normalization(inputs, scale), expected_dtype)
        self.assertDType(
            knn.RMSNorm().symbolic_call(inputs, scale), expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=combinations(FLOAT_DTYPES, 2))
    )
    def test_layer_normalization(self, dtypes):
        input_dtype, weight_dtype = dtypes
        inputs = knp.ones((2, 8), dtype=input_dtype)
        gamma = backend.Variable(knp.ones((8,), dtype=weight_dtype))
        beta = backend.Variable(knp.ones((8,), dtype=weight_dtype))
        expected_dtype = input_dtype

        self.assertDType(
            knn.layer_normalization(inputs, gamma, beta), expected_dtype
        )
        self.assertDType(
            knn.LayerNorm().symbolic_call(inputs, gamma, beta), expected_dtype
        )


class NNOpsBehaviorTest(testing.TestCase):
    def test_logit_recovery_binary_crossentropy(self):
        layer = layers.Dense(
            4, activation="sigmoid", use_bias=False, kernel_initializer="ones"
        )
        loss = losses.BinaryCrossentropy()
        x = np.array([[1.4, 1.6, 0.8]])
        y = np.array([[0.2, 0.6, 0.1, 0.3]])
        loss_value = loss(y, layer(x))
        self.assertAllClose(loss_value, 2.682124)

        model = models.Sequential([layer])
        model.compile(loss="binary_crossentropy", optimizer="sgd")
        out = model.evaluate(x, y)
        self.assertAllClose(out, 2.682124, atol=1e-3, rtol=1e-3)

    def test_softmax_on_axis_with_size_one_warns(self):
        x = np.array([[1.0]])
        # Applying softmax on the second axis, which has size 1
        axis = 1

        # Expected warning message
        expected_warning_regex = (
            r"You are using a softmax over axis 1 "
            r"of a tensor of shape \(1, 1\)\. This axis "
            r"has size 1\. The softmax operation will always return "
            r"the value 1, which is likely not what you intended\. "
            r"Did you mean to use a sigmoid instead\?"
        )

        with self.assertWarnsRegex(UserWarning, expected_warning_regex):
            knn.softmax(x, axis)

    def test_softmax_and_normalize_reject_out_of_range_axis(self):
        a = KerasTensor((3, 4))
        with self.assertRaisesRegex(ValueError, "axis 10 is out of bounds"):
            knn.softmax(a, axis=10)
        with self.assertRaisesRegex(ValueError, "axis 10 is out of bounds"):
            knn.normalize(a, axis=10)

    def test_normalize_order_validation(self):
        # Test with a non-integer order
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order="a")

        # Test with a negative integer
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=-1)

        # Test with zero
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=0)

        # Test with a floating-point number
        with self.assertRaisesRegex(
            ValueError, "Argument `order` must be an int >= 1"
        ):
            knn.normalize(np.array([1, 2, 3]), order=2.5)

    def test_check_shape_first_dim_mismatch(self):
        name1, shape1 = "labels", (2, 3)
        name2, shape2 = "logits", (3, 4, 5)
        ctc_loss_instance = knn.CTCLoss(mask_index=-1)
        with self.assertRaisesRegex(
            ValueError, "must have the same first dimension"
        ):
            ctc_loss_instance._check_shape_first_dim(
                name1, shape1, name2, shape2
            )

    def test_invalid_strategy_ctc_decode(self):
        inputs = np.array(
            [
                [
                    [0.1, 0.4, 0.2, 0.4],
                    [0.3, 0.3, 0.4, 0.2],
                    [0.3, 0.2, 0.4, 0.3],
                ]
            ]
        )
        beam_width = 4
        top_paths = 2
        with self.assertRaisesRegex(ValueError, "Invalid strategy"):
            knn.ctc_decode(
                inputs,
                sequence_lengths=[3, 3, 1],
                strategy="invalid",
                beam_width=beam_width,
                top_paths=top_paths,
            )

    def test_layer_normalization_rms_scaling_warning(self):
        x = np.arange(5, dtype="float32")
        with self.assertWarnsRegex(
            UserWarning, r"You passed `rms_scaling=True`, which is deprecated"
        ):
            knn.layer_normalization(x, rms_scaling=True)

    def test_unfold(self):
        # test 1 kernel_size=2
        x = ops.arange(8, dtype="float32")
        x = ops.reshape(x, [1, 1, 2, 4])
        unfold_result = knn.unfold(x, 2)
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [5.0, 6.0, 7.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 2 kernel_size=[2,4]
        x = ops.arange(16, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 4])
        unfold_result = knn.unfold(x, [2, 4])
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0, 4.0, 8.0],
                    [1.0, 5.0, 9.0],
                    [2.0, 6.0, 10.0],
                    [3.0, 7.0, 11.0],
                    [4.0, 8.0, 12.0],
                    [5.0, 9.0, 13.0],
                    [6.0, 10.0, 14.0],
                    [7.0, 11.0, 15.0],
                ]
            ],
            dtype="float32",
        )
        self.assertAllClose(unfold_result, except_result)

        # test 3 kernel_size=[3,2],stride=[3,2]
        x = ops.arange(12, dtype="float32")
        x = ops.reshape(x, [1, 1, 3, 4])
        unfold_result = knn.unfold(x, [3, 2], stride=[3, 2])
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0, 2.0],
                    [1.0, 3.0],
                    [4.0, 6.0],
                    [5.0, 7.0],
                    [8.0, 10.0],
                    [9.0, 11.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 4 kernel_size=2,dilation=2,stride=2
        x = ops.arange(16, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 4])
        unfold_result = knn.unfold(x, 2, 2, stride=2)
        except_result = ops.convert_to_tensor([0, 2, 8, 10], dtype="float32")
        except_result = ops.reshape(except_result, [1, 4, 1])
        self.assertAllClose(unfold_result, except_result)

        # test 5 kernel_size=2,padding=1
        x = ops.arange(4, dtype="float32")
        x = ops.reshape(x, [1, 1, 2, 2])
        unfold_result = knn.unfold(x, 1, padding=1)
        except_result = ops.convert_to_tensor(
            [
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        2.0,
                        3.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 6 multi channal and kernel_size=2
        x = ops.arange(8, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 2])
        unfold_result = knn.unfold(x, 2)
        except_result = ops.convert_to_tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 7 multi channal and kernel_size=[2,3]
        x = ops.arange(12, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 3])
        unfold_result = knn.unfold(x, [2, 3])
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0],
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0],
                    [8.0],
                    [9.0],
                    [10.0],
                    [11.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 8 multi channal and kernel_size=[2,3],stride=[2,3]
        x = ops.arange(12, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 3])
        unfold_result = knn.unfold(x, [2, 3], stride=[2, 3])
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0],
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0],
                    [8.0],
                    [9.0],
                    [10.0],
                    [11.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 9 multi channal and kernel_size=2,dilation=2
        x = ops.arange(32, dtype="float32")
        x = ops.reshape(x, [1, 2, 4, 4])
        unfold_result = knn.unfold(x, 2, dilation=2)
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0, 1.0, 4.0, 5.0],
                    [2.0, 3.0, 6.0, 7.0],
                    [8.0, 9.0, 12.0, 13.0],
                    [10.0, 11.0, 14.0, 15.0],
                    [16.0, 17.0, 20.0, 21.0],
                    [18.0, 19.0, 22.0, 23.0],
                    [24.0, 25.0, 28.0, 29.0],
                    [26.0, 27.0, 30.0, 31.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

        # test 10 multi channal and kernel_size=2,padding=1
        x = ops.arange(8, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 2])
        unfold_result = knn.unfold(x, 2, padding=1)
        except_result = ops.convert_to_tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 6.0, 7.0],
                    [0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 6.0, 7.0, 0.0],
                    [0.0, 4.0, 5.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        self.assertAllClose(unfold_result, except_result)

    def test_fold(self):
        # test 1: non-overlapping roundtrip (stride == kernel_size)
        x = ops.arange(16, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 4])
        patches = knn.unfold(x, kernel_size=2, stride=2)
        y = knn.fold(patches, output_size=(4, 4), kernel_size=2, stride=2)
        self.assertAllClose(y, x)

        # test 2: overlapping roundtrip — fold(unfold(x)) / fold(ones) == x
        x = ops.arange(16, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 4])
        patches = knn.unfold(x, kernel_size=3, stride=1)
        folded = knn.fold(patches, output_size=(4, 4), kernel_size=3, stride=1)
        ones = ops.ones_like(x)
        ones_patches = knn.unfold(ones, kernel_size=3, stride=1)
        divisor = knn.fold(
            ones_patches, output_size=(4, 4), kernel_size=3, stride=1
        )
        result = folded / divisor
        self.assertAllClose(result, x)

        # test 3: multi-channel non-overlapping roundtrip
        x = ops.arange(32, dtype="float32")
        x = ops.reshape(x, [1, 2, 4, 4])
        patches = knn.unfold(x, kernel_size=2, stride=2)
        y = knn.fold(patches, output_size=(4, 4), kernel_size=2, stride=2)
        self.assertAllClose(y, x)

        # test 4: dilation + padding roundtrip
        x = ops.arange(32, dtype="float32")
        x = ops.reshape(x, [1, 2, 4, 4])
        patches = knn.unfold(x, kernel_size=2, dilation=2, padding=1)
        y = knn.fold(
            patches,
            output_size=(4, 4),
            kernel_size=2,
            dilation=2,
            padding=1,
        )
        ones = ops.ones_like(x)
        ones_patches = knn.unfold(ones, kernel_size=2, dilation=2, padding=1)
        divisor = knn.fold(
            ones_patches,
            output_size=(4, 4),
            kernel_size=2,
            dilation=2,
            padding=1,
        )
        result = y / divisor
        self.assertAllClose(result, x)

        # test 5: explicit known values — single 1x1x2x2 non-overlap
        x = ops.convert_to_tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            dtype="float32",
        )
        # x shape: (1, 2, 4) — as if C=2, kernel=1x1, L=4
        y = knn.fold(x, output_size=(2, 2), kernel_size=1, stride=1)
        expected = ops.convert_to_tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
            dtype="float32",
        )
        self.assertAllClose(y, expected)

        # test 6: input validation — must be 3D
        x_bad = ops.ones((1, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, "3D"):
            knn.fold(x_bad, output_size=(4, 4), kernel_size=2)

    def test_fold_tuple_params(self):
        """Test fold with tuple parameters to cover _pair branches."""
        # Non-square kernel and stride as tuples
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 1, 6, 8])
        patches = knn.unfold(x, kernel_size=(2, 4), stride=(2, 4))
        y = knn.fold(
            patches,
            output_size=(6, 8),
            kernel_size=(2, 4),
            stride=(2, 4),
        )
        self.assertAllClose(y, x)

        # Asymmetric padding as tuple — only pad height
        x = ops.arange(16, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 4])
        patches = knn.unfold(
            x, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)
        )
        folded = knn.fold(
            patches,
            output_size=(4, 4),
            kernel_size=(3, 2),
            stride=(1, 2),
            padding=(1, 0),
        )
        ones = ops.ones_like(x)
        ones_p = knn.unfold(
            ones, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)
        )
        divisor = knn.fold(
            ones_p,
            output_size=(4, 4),
            kernel_size=(3, 2),
            stride=(1, 2),
            padding=(1, 0),
        )
        result = folded / divisor
        self.assertAllClose(result, x)

    def test_fold_no_padding(self):
        """Test fold with padding=0 to cover the skip-padding branch."""
        x = ops.arange(36, dtype="float32")
        x = ops.reshape(x, [1, 1, 6, 6])
        patches = knn.unfold(x, kernel_size=3, stride=3, padding=0)
        y = knn.fold(
            patches,
            output_size=(6, 6),
            kernel_size=3,
            stride=3,
            padding=0,
        )
        self.assertAllClose(y, x)

    def test_fold_non_square_output(self):
        """Test fold with non-square spatial dimensions."""
        x = ops.arange(24, dtype="float32")
        x = ops.reshape(x, [1, 1, 4, 6])
        patches = knn.unfold(x, kernel_size=2, stride=2)
        y = knn.fold(patches, output_size=(4, 6), kernel_size=2, stride=2)
        self.assertAllClose(y, x)

    def test_fold_batch_and_channels(self):
        """Test fold with larger batch and channel counts."""
        x = np.random.normal(size=(4, 8, 6, 6)).astype("float32")
        x = ops.convert_to_tensor(x)
        patches = knn.unfold(x, kernel_size=2, stride=2)
        y = knn.fold(patches, output_size=(6, 6), kernel_size=2, stride=2)
        self.assertAllClose(y, x, tpu_atol=1e-2, tpu_rtol=1e-2)

    def test_fold_divisibility_validation(self):
        """Test fold raises on CKK not divisible by kernel product."""
        # CKK=5, kernel=2x2 -> 5 % 4 != 0 — raises reshape error
        x_bad = ops.ones((1, 5, 4))
        with self.assertRaises((ValueError, Exception)):
            knn.fold(x_bad, output_size=(4, 4), kernel_size=2)

    def test_depth_to_space(self):
        # Test channels_last (default)
        # Input: (1, 2, 2, 12) -> Output: (1, 4, 4, 3)
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 12])
        result = knn.depth_to_space(x, block_size=2)
        self.assertEqual(result.shape, (1, 4, 4, 3))

        # Verify the transformation is correct
        # The depth channel is rearranged into spatial blocks
        # For block_size=2, channels are split into 2x2 blocks
        expected = np.array(
            [
                [
                    [[0, 1, 2], [3, 4, 5], [12, 13, 14], [15, 16, 17]],
                    [[6, 7, 8], [9, 10, 11], [18, 19, 20], [21, 22, 23]],
                    [[24, 25, 26], [27, 28, 29], [36, 37, 38], [39, 40, 41]],
                    [[30, 31, 32], [33, 34, 35], [42, 43, 44], [45, 46, 47]],
                ]
            ],
            dtype="float32",
        )
        self.assertAllClose(result, expected)

        # Test channels_first
        # Input: (1, 12, 2, 2) -> Output: (1, 3, 4, 4)
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 12, 2, 2])
        result = knn.depth_to_space(
            x, block_size=2, data_format="channels_first"
        )
        self.assertEqual(result.shape, (1, 3, 4, 4))

        # Test with different block size
        x = ops.arange(1 * 2 * 2 * 27, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 27])
        result = knn.depth_to_space(x, block_size=3)
        self.assertEqual(result.shape, (1, 6, 6, 3))

    def test_space_to_depth(self):
        # Test channels_last (default)
        # Input: (1, 4, 4, 3) -> Output: (1, 2, 2, 12)
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 4, 4, 3])
        result = knn.space_to_depth(x, block_size=2)
        self.assertEqual(result.shape, (1, 2, 2, 12))

        # Verify the transformation is correct
        expected = np.array(
            [
                [
                    [
                        [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
                        [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23],
                    ],
                    [
                        [24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41],
                        [30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47],
                    ],
                ]
            ],
            dtype="float32",
        )
        self.assertAllClose(result, expected)

        # Test channels_first
        # Input: (1, 3, 4, 4) -> Output: (1, 12, 2, 2)
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 3, 4, 4])
        result = knn.space_to_depth(
            x, block_size=2, data_format="channels_first"
        )
        self.assertEqual(result.shape, (1, 12, 2, 2))

        # Test with different block size
        x = ops.arange(1 * 6 * 6 * 3, dtype="float32")
        x = ops.reshape(x, [1, 6, 6, 3])
        result = knn.space_to_depth(x, block_size=3)
        self.assertEqual(result.shape, (1, 2, 2, 27))

    def test_depth_to_space_space_to_depth_roundtrip(self):
        # depth_to_space followed by space_to_depth should be identity
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 12])
        y = knn.depth_to_space(x, block_size=2)
        z = knn.space_to_depth(y, block_size=2)
        self.assertAllClose(x, z)

        # space_to_depth followed by depth_to_space should be identity
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 4, 4, 3])
        y = knn.space_to_depth(x, block_size=2)
        z = knn.depth_to_space(y, block_size=2)
        self.assertAllClose(x, z)

        # Test with channels_first
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 12, 2, 2])
        y = knn.depth_to_space(x, block_size=2, data_format="channels_first")
        z = knn.space_to_depth(y, block_size=2, data_format="channels_first")
        self.assertAllClose(x, z)

    def test_depth_to_space_block_size_validation(self):
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 2, 2, 12])

        # block_size must be at least 2
        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.depth_to_space(x, block_size=0)

        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.depth_to_space(x, block_size=1)

        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.depth_to_space(x, block_size=-1)

    def test_space_to_depth_block_size_validation(self):
        x = ops.arange(48, dtype="float32")
        x = ops.reshape(x, [1, 4, 4, 3])

        # block_size must be at least 2
        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.space_to_depth(x, block_size=0)

        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.space_to_depth(x, block_size=1)

        with self.assertRaisesRegex(
            ValueError, "`block_size` must be at least 2"
        ):
            knn.space_to_depth(x, block_size=-1)


@pytest.mark.skipif(
    backend.backend() == "numpy",
    reason="Key/Value broadcasting is not supported on the numpy backend.",
)
class DotProductAttentionGQATest(testing.TestCase):
    def test_gqa_broadcasting(self):
        batch_size = 2
        q_len, kv_len = 16, 16
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 32

        query = ops.ones(
            (batch_size, q_len, num_q_heads, head_dim), dtype="float32"
        )
        key = ops.ones(
            (batch_size, kv_len, num_kv_heads, head_dim), dtype="float32"
        )
        value = ops.ones(
            (batch_size, kv_len, num_kv_heads, head_dim), dtype="float32"
        )

        # Should execute successfully without raising any
        # shape mismatch RuntimeError/InvalidArgumentError
        output = knn.dot_product_attention(query, key, value)
        self.assertEqual(
            output.shape, (batch_size, q_len, num_q_heads, head_dim)
        )


@pytest.mark.no_pytest_xdist
class TorchNNDistributedTest(TorchDistributedTestMixin, testing.TestCase):
    @pytest.mark.skipif(backend.backend() != "torch", reason="Torch only")
    def test_dot_product_attention_dtensor(self):
        import torch

        device_type = get_device().split(":")[0]
        mesh = DeviceMesh(
            shape=(1,), axis_names=("data",), devices=[f"{device_type}:0"]
        )
        layout = TensorLayout(axes=(None, None, None, None), device_mesh=mesh)

        B, T, S, N, H = 2, 4, 4, 2, 8
        query_local = torch.rand((B, T, N, H), device="cpu")
        key_local = torch.rand((B, S, N, H), device="cpu")
        value_local = torch.rand((B, S, N, H), device="cpu")

        query = distribute_data_input(query_local, layout)
        key = distribute_data_input(key_local, layout)
        value = distribute_data_input(value_local, layout)

        # Test without mask
        result = knn.dot_product_attention(query, key, value)
        self.assertTrue(hasattr(result, "to_local"))
        expected = knn.dot_product_attention(
            query_local.to(get_device()),
            key_local.to(get_device()),
            value_local.to(get_device()),
        )
        self.assertAllClose(result, expected)

        # Test with mask and is_causal
        mask_local = torch.tril(
            torch.ones((B, N, T, S), dtype=torch.bool, device="cpu")
        )
        mask = distribute_data_input(mask_local, layout)

        result_mask = knn.dot_product_attention(
            query, key, value, mask=mask, is_causal=True
        )
        self.assertTrue(hasattr(result_mask, "to_local"))
        expected_mask = knn.dot_product_attention(
            query_local.to(get_device()),
            key_local.to(get_device()),
            value_local.to(get_device()),
            mask=mask_local.to(get_device()),
            is_causal=True,
        )
        self.assertAllClose(result_mask, expected_mask)
