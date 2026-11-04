import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import standardize_shape


def np_dot(a, b, axes):
    if isinstance(axes, int):
        axes = (axes, axes)
    axes = [axis if axis < 0 else axis - 1 for axis in axes]
    res = np.stack([np.tensordot(a[i], b[i], axes) for i in range(a.shape[0])])
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=1)
    return res


TEST_PARAMETERS = [
    {
        "testcase_name": "add",
        "layer_class": layers.Add,
        "np_op": np.add,
    },
    {
        "testcase_name": "subtract",
        "layer_class": layers.Subtract,
        "np_op": np.subtract,
    },
    {
        "testcase_name": "minimum",
        "layer_class": layers.Minimum,
        "np_op": np.minimum,
    },
    {
        "testcase_name": "maximum",
        "layer_class": layers.Maximum,
        "np_op": np.maximum,
    },
    {
        "testcase_name": "multiply",
        "layer_class": layers.Multiply,
        "np_op": np.multiply,
    },
    {
        "testcase_name": "average",
        "layer_class": layers.Average,
        "np_op": lambda a, b: np.multiply(np.add(a, b), 0.5),
    },
    {
        "testcase_name": "concat",
        "layer_class": layers.Concatenate,
        "np_op": lambda a, b, **kwargs: np.concatenate((a, b), **kwargs),
        "init_kwargs": {"axis": -1},
        "expected_output_shape": (2, 4, 10),
    },
    {
        "testcase_name": "dot_2d",
        "layer_class": layers.Dot,
        "np_op": np_dot,
        "init_kwargs": {"axes": -1},
        "input_shape": (2, 4),
        "expected_output_shape": (2, 1),
        "skip_mask_test": True,
    },
    {
        "testcase_name": "dot_3d",
        "layer_class": layers.Dot,
        "np_op": np_dot,
        "init_kwargs": {"axes": -1},
        "expected_output_shape": (2, 4, 4),
        "skip_mask_test": True,
    },
]


class MergingLayersTest(testing.TestCase):
    @parameterized.named_parameters(TEST_PARAMETERS)
    @pytest.mark.requires_trainable_backend
    def test_basic(
        self,
        layer_class,
        init_kwargs={},
        input_shape=(2, 4, 5),
        expected_output_shape=(2, 4, 5),
        **kwargs,
    ):
        self.run_layer_test(
            layer_class,
            init_kwargs=init_kwargs,
            input_shape=(input_shape, input_shape),
            expected_output_shape=expected_output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.named_parameters(TEST_PARAMETERS)
    def test_correctness_static(
        self,
        layer_class,
        np_op,
        init_kwargs={},
        input_shape=(2, 4, 5),
        expected_output_shape=(2, 4, 5),
        skip_mask_test=False,
    ):
        batch_size = input_shape[0]
        shape = input_shape[1:]
        x1 = np.random.rand(*input_shape)
        x2 = np.random.rand(*input_shape)
        x3 = np_op(x1, x2, **init_kwargs)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        layer = layer_class(**init_kwargs)
        out = layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, expected_output_shape)
        self.assertAllClose(res, x3, atol=1e-4, tpu_atol=1e-2, tpu_rtol=1e-2)
        self.assertIsNone(layer.compute_mask([input_1, input_2], [None, None]))
        self.assertIsNone(layer.compute_mask([x1, x2], [None, None]))
        if not skip_mask_test:
            mask1 = np.ones(input_shape[:-1], dtype=np.bool_)
            mask2 = np.ones(input_shape[:-1], dtype=np.bool_)
            self.assertTrue(
                np.all(
                    backend.ops.convert_to_numpy(
                        layer.compute_mask([x1, x2], [mask1, mask2])
                    )
                )
            )

    @parameterized.named_parameters(TEST_PARAMETERS)
    def test_correctness_dynamic(
        self,
        layer_class,
        np_op,
        init_kwargs={},
        input_shape=(2, 4, 5),
        expected_output_shape=(2, 4, 5),
        skip_mask_test=False,
    ):
        shape = input_shape[1:]
        x1 = np.random.rand(*input_shape)
        x2 = np.random.rand(*input_shape)
        x3 = np_op(x1, x2, **init_kwargs)

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)
        layer = layer_class(**init_kwargs)
        out = layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, expected_output_shape)
        self.assertAllClose(res, x3, atol=1e-4, tpu_atol=1e-2, tpu_rtol=1e-2)
        self.assertIsNone(layer.compute_mask([input_1, input_2], [None, None]))
        if not skip_mask_test:
            self.assertTrue(
                np.all(
                    backend.ops.convert_to_numpy(
                        layer.compute_mask(
                            [input_1, input_2],
                            [backend.Variable(x1), backend.Variable(x2)],
                        )
                    )
                )
            )

    @parameterized.named_parameters(TEST_PARAMETERS)
    def test_errors(
        self,
        layer_class,
        init_kwargs={},
        input_shape=(2, 4, 5),
        skip_mask_test=False,
        **kwargs,
    ):
        if skip_mask_test:
            pytest.skip("Masking not supported")

        batch_size = input_shape[0]
        shape = input_shape[1:]
        x1 = np.random.rand(*input_shape)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        layer = layer_class(**init_kwargs)

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            layer.compute_mask([input_1, input_2], [None])

    def test_subtract_layer_inputs_length_errors(self):
        shape = (4, 5)
        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)
        input_3 = layers.Input(shape=shape)

        with self.assertRaisesRegex(
            ValueError, "layer should be called on exactly 2 inputs"
        ):
            layers.Subtract()([input_1, input_2, input_3])
        with self.assertRaisesRegex(
            ValueError, "layer should be called on exactly 2 inputs"
        ):
            layers.Subtract()([input_1])

    def test_dot_higher_dim(self):
        a_shape = (1, 3, 2)
        b_shape = (1, 1, 2, 3)
        # Test symbolic call
        a = layers.Input(batch_shape=a_shape)
        b = layers.Input(batch_shape=b_shape)
        c = layers.Dot(axes=(-2, -1))([a, b])
        self.assertEqual(c.shape, (1, 2, 1, 2))
        a = np.random.random(a_shape)
        b = np.random.random(b_shape)
        c = layers.Dot(axes=(-2, -1))([a, b])
        self.assertEqual(standardize_shape(c.shape), (1, 2, 1, 2))

    def test_add_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Add()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [1, 2], [6, 8]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Add()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [1, 2], [6, 8]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_subtract_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Subtract()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [-1, -2], [0, 0]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Subtract()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [-1, -2], [0, 0]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_average_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Average()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [0.5, 1], [0.5, 1], [3, 4]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Average()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [0.5, 1], [0.5, 1], [3, 4]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_multiply_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Multiply()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [0, 0], [1, 2], [9, 16]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Multiply()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [1, 2], [9, 16]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_maximum_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor(
                [[[0, 0], [-1, -2], [0, 0], [-3, -4]]]
            )
        )
        x2 = backend.ops.convert_to_tensor(
            [[[0, 0], [0, 0], [-1, -2], [-3, -4]]]
        )

        output = layers.Maximum()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [0, 0], [-1, -2], [-3, -4]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Maximum()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [-1, -2], [-1, -2], [-3, -4]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_minimum_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Minimum()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [0, 0], [1, 2], [3, 4]]])
        self.assertIsNone(get_keras_mask(output))

        x2 = mask(x2)
        output = layers.Minimum()([x1, x2])
        self.assertAllClose(output, [[[0, 0], [1, 2], [1, 2], [3, 4]]])
        self.assertAllClose(get_keras_mask(output), [[0, 1, 1, 1]])

    def test_concatenate_with_mask(self):
        mask = layers.Masking()
        x1 = mask(
            backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        )
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])

        output = layers.Concatenate(axis=1)([x1, x2])
        self.assertAllClose(
            output,
            [[[0, 0], [1, 2], [0, 0], [3, 4], [0, 0], [0, 0], [1, 2], [3, 4]]],
        )
        self.assertAllClose(get_keras_mask(output), [[0, 1, 0, 1, 1, 1, 1, 1]])

        output = layers.Concatenate(axis=2)([x1, x2])
        self.assertAllClose(
            output,
            [[[0, 0, 0, 0], [1, 2, 0, 0], [0, 0, 1, 2], [3, 4, 3, 4]]],
        )
        self.assertAllClose(get_keras_mask(output), [[1, 1, 1, 1]])

    def test_concatenate_with_mask_symbolic(self):
        input1 = layers.Input((4, 2))
        input2 = layers.Input((4, 2))
        mask = layers.Masking()
        output = layers.Concatenate(axis=1)([mask(input1), input2])
        model = models.Model(
            inputs=[input1, input2], outputs=get_keras_mask(output)
        )
        x1 = backend.ops.convert_to_tensor([[[0, 0], [1, 2], [0, 0], [3, 4]]])
        x2 = backend.ops.convert_to_tensor([[[0, 0], [0, 0], [1, 2], [3, 4]]])
        self.assertAllClose(model([x1, x2]), [[0, 1, 0, 1, 1, 1, 1, 1]])

    def test_concatenate_errors(self):
        # This should work
        x1 = np.ones((1, 1, 1, 1, 5))
        x2 = np.ones((1, 1, 1, 1, 4))
        out = layers.Concatenate(axis=-1)([x1, x2])
        self.assertEqual(backend.ops.shape(out), (1, 1, 1, 1, 9))

        # This won't
        x1 = np.ones((1, 2, 1, 1, 5))
        x2 = np.ones((1, 1, 1, 1, 4))
        with self.assertRaisesRegex(
            ValueError,
            (
                "requires inputs with matching shapes "
                "except for the concatenation axis"
            ),
        ):
            out = layers.Concatenate(axis=-1)([x1, x2])
        x1 = np.ones((1, 2, 1, 2, 1))
        x2 = np.ones((1, 1, 1, 3, 1))
        with self.assertRaisesRegex(
            ValueError,
            (
                "requires inputs with matching shapes "
                "except for the concatenation axis"
            ),
        ):
            out = layers.Concatenate(axis=1)([x1, x2])

    @parameterized.named_parameters(TEST_PARAMETERS)
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_sparse(
        self,
        layer_class,
        np_op,
        init_kwargs={},
        input_shape=(2, 4, 5),
        expected_output_shape=(2, 4, 5),
        **kwargs,
    ):
        self.run_layer_test(
            layer_class,
            init_kwargs=init_kwargs,
            input_shape=[input_shape, input_shape],
            input_sparse=True,
            expected_output_shape=expected_output_shape,
            expected_output_sparse=True,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
            run_mixed_precision_check=False,
        )

        layer = layer_class(**init_kwargs)

        # Merging a sparse tensor with a dense tensor, or a dense tensor with a
        # sparse tensor produces a dense tensor
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x1 = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
            x3 = tf.SparseTensor([[0, 0], [1, 1]], [4.0, 5.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            # Use n_batch of 1 to be compatible with all ops.
            x1 = jax_sparse.BCOO(([[1.0, 2.0]], [[[0], [2]]]), shape=(2, 3))
            x3 = jax_sparse.BCOO(([[4.0, 5.0]], [[[0], [1]]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        x1_np = backend.ops.convert_to_numpy(x1)
        x2 = np.random.rand(2, 3)
        self.assertAllClose(layer([x1, x2]), np_op(x1_np, x2, **init_kwargs))
        self.assertAllClose(layer([x2, x1]), np_op(x2, x1_np, **init_kwargs))

        # Merging a sparse tensor with a sparse tensor produces a sparse tensor
        x3_np = backend.ops.convert_to_numpy(x3)

        self.assertSparse(layer([x1, x3]))
        self.assertAllClose(layer([x1, x3]), np_op(x1_np, x3_np, **init_kwargs))
