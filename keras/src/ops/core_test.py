import operator
from unittest.mock import Mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import random
from keras.src import testing
from keras.src import tree
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.remat import RematScope
from keras.src.layers.core import input_layer
from keras.src.ops import core
from keras.src.saving import object_registration
from keras.src.testing.test_utils import named_product
from keras.src.utils import traceback_utils


class CoreOpsDynamicShapeTest(testing.TestCase):
    def test_associative_scan(self):
        xs = (KerasTensor((5, None)), KerasTensor((5, None)))
        ys = core.associative_scan(
            f=lambda x, y: (x[0] + y[0], x[1] + y[1]), elems=xs, axis=0
        )
        self.assertEqual(ys[0].shape, (5, None))

        # sum two tuples of unknown (but same) length at axis
        def _fn(x, y):
            return tuple([x[i] + y[i] for i in range(len(x))])

        ys = core.associative_scan(f=_fn, elems=xs, axis=1)
        self.assertEqual(ys[0].shape, (5, None))

    def test_cast(self):
        x = KerasTensor((3, 5, None), dtype="float32")
        self.assertEqual(core.cast(x, "float16").shape, (3, 5, None))

    def test_convert_to_tensor(self):
        x = KerasTensor((2, None))
        self.assertEqual(core.convert_to_tensor(x).shape, (2, None))

    def test_fori_loop(self):
        def body_fun(i, x):
            return x + i

        initial_value = KerasTensor((3, 5, None))
        self.assertEqual(
            core.fori_loop(0, 10, body_fun, initial_value).shape, (3, 5, None)
        )

    def test_map(self):
        def f(x):
            return x**2

        xs = KerasTensor((None, 5))
        self.assertEqual(core.map(f, xs).shape, (None, 5))

        # Test nested output
        def f2(x):
            return {"a": x**2, "b": x * 10}

        xs = KerasTensor((None, 5))
        ys = core.map(f2, xs)
        self.assertEqual(ys["a"].shape, (None, 5))
        self.assertEqual(ys["b"].shape, (None, 5))

        # Test nested input
        def f3(x):
            return x[0] + x[1]

        xs = (KerasTensor((None, 5)), KerasTensor((None, 5)))
        self.assertEqual(core.map(f3, xs).shape, (None, 5))

    def test_saturate_cast(self):
        x = KerasTensor((3, 5, None), dtype="float32")
        self.assertEqual(core.saturate_cast(x, "float16").shape, (3, 5, None))

    def test_scan(self):
        def f(carry, xs):
            xs = xs + carry
            return carry, carry

        init = KerasTensor((None,))
        xs = KerasTensor((6, None))
        carry, result = core.scan(f, init, xs)
        self.assertEqual(carry.shape, (None,))
        self.assertEqual(result.shape, (6, None))

        def f2(carry, _):
            return carry, carry

        carry, result = core.scan(f2, init, xs=None, length=3)
        self.assertEqual(carry.shape, (None,))
        self.assertEqual(result.shape, (3, None))

    # Scatter doesn't support dynamic shape.

    def test_scatter_update(self):
        inputs = KerasTensor((4, None))
        indices = KerasTensor((5, 2))
        updates = KerasTensor((5,))
        self.assertEqual(
            core.scatter_update(inputs, indices, updates).shape, (4, None)
        )

    # Slice doesn't support dynamic shape.

    def test_slice_update(self):
        inputs = KerasTensor((4, None))
        start_indices = KerasTensor((2,))
        updates = KerasTensor((2, 2))
        self.assertEqual(
            core.slice_update(inputs, start_indices, updates).shape, (4, None)
        )

    def test_stop_gradient(self):
        variable = KerasTensor(shape=(3, None), dtype="float32")
        self.assertEqual(core.stop_gradient(variable).shape, (3, None))

    def test_switch(self):
        def fn(x, y):
            return x[:, 0], y[0, :]

        index = KerasTensor(())
        x = KerasTensor((None, 2))
        y = KerasTensor((5, None))
        result = core.switch(index, [fn], x, y)
        self.assertEqual(result[0].shape, (None,))
        self.assertEqual(result[1].shape, (None,))

    def test_vectorized_map(self):
        def f(x):
            return x**2

        xs = KerasTensor((None, 5))
        self.assertEqual(core.vectorized_map(f, xs).shape, (None, 5))

        # Test nested output
        def f2(x):
            return {"a": x**2, "b": x * 10}

        xs = KerasTensor((None, 5))
        ys = core.vectorized_map(f2, xs)
        self.assertEqual(ys["a"].shape, (None, 5))
        self.assertEqual(ys["b"].shape, (None, 5))

        # Test nested input
        def f3(x):
            return x[0] + x[1]

        xs = (KerasTensor((None, 5)), KerasTensor((None, 5)))
        self.assertEqual(core.vectorized_map(f3, xs).shape, (None, 5))

    def test_while_loop(self):
        def cond(args):
            return tree.flatten(args)[0] < 10

        def body(args):
            return tree.map_structure(lambda x: x + 1, args)

        loop_vars = KerasTensor((None,))
        self.assertEqual(core.while_loop(cond, body, loop_vars).shape, (None,))

    def test_unstack(self):
        x = KerasTensor((2, None, None))
        axis, num = 1, 3
        out = core.unstack(x, num=num, axis=axis)
        self.assertEqual(len(out), 3)
        for o in out:
            self.assertEqual(o.shape, (2, None))

    def test_associative_scan_dynamic_scan_body_unknown_length(self):
        """fully unknown leading dim forces the TF symbolic path."""
        xs = KerasTensor((None, 6))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (None, 6))

    def test_associative_scan_dynamic_scan_body_unknown_length_structured(self):
        """symbolic path with structured (tuple) input."""
        xs = (KerasTensor((None, 4)), KerasTensor((None, 4)))

        def _add(a, b):
            return (a[0] + b[0], a[1] + b[1])

        ys = core.associative_scan(f=_add, elems=xs, axis=0)
        self.assertEqual(ys[0].shape, (None, 4))
        self.assertEqual(ys[1].shape, (None, 4))

    def test_associative_scan_dynamic_scan_body_unknown_length_reverse(self):
        """symbolic path with reverse=True."""
        xs = KerasTensor((None, 3))
        ys = core.associative_scan(
            f=operator.add, elems=xs, axis=0, reverse=True
        )
        self.assertEqual(ys.shape, (None, 3))


class CoreOpsStaticShapeTest(testing.TestCase):
    def test_associative_scan(self):
        xs = (KerasTensor((5, 10)), KerasTensor((5, 10)))
        ys = core.associative_scan(
            f=lambda x, y: (x[0] + y[0], x[1] + y[1]), elems=xs, axis=0
        )
        self.assertEqual(ys[0].shape, (5, 10))

        # sum two tuples of unknown (but same) length at axis
        def _fn(x, y):
            return tuple([x[i] + y[i] for i in range(len(x))])

        ys = core.associative_scan(f=_fn, elems=xs, axis=1)
        self.assertEqual(ys[0].shape, (5, 10))

    def test_cast(self):
        x = KerasTensor((3, 5, 7), dtype="float32")
        self.assertEqual(core.cast(x, "float16").shape, (3, 5, 7))

    def test_cond(self):
        pred = KerasTensor((), dtype="bool")
        self.assertEqual(
            ops.cond(
                pred, lambda: ops.ones((1, 3)), lambda: ops.zeros((1, 3))
            ).shape,
            (1, 3),
        )

    def test_convert_to_tensor(self):
        x = KerasTensor((2, 3))
        out = core.convert_to_tensor(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(out.sparse)

        out = core.convert_to_tensor(x, sparse=True)
        self.assertFalse(out.sparse)

        x = KerasTensor((2, 3), sparse=True)
        out = core.convert_to_tensor(x)
        self.assertTrue(out.sparse)

        out = core.convert_to_tensor(x, sparse=True)
        self.assertTrue(out.sparse)

        out = core.convert_to_tensor(x, sparse=False)
        self.assertFalse(out.sparse)

    def test_fori_loop(self):
        def body_fun(i, x):
            return x + i

        initial_value = KerasTensor((3, 5, 7))
        result = core.fori_loop(0, 10, body_fun, initial_value)
        self.assertEqual(result.shape, (3, 5, 7))

    def test_map(self):
        def f(x):
            return x**2

        xs = KerasTensor((6, 5))
        ys = core.map(f, xs)
        self.assertEqual(ys.shape, (6, 5))

        # Test nested output
        def f2(x):
            return {"a": x**2, "b": x * 10}

        xs = KerasTensor((6, 5))
        ys = core.map(f2, xs)
        self.assertEqual(ys["a"].shape, (6, 5))
        self.assertEqual(ys["b"].shape, (6, 5))

        # Test nested input
        def f3(x):
            return x[0] + x[1]

        xs = (KerasTensor((6, 5)), KerasTensor((6, 5)))
        self.assertEqual(core.map(f3, xs).shape, (6, 5))

    def test_saturate_cast(self):
        x = KerasTensor((3, 5, 7), dtype="float32")
        self.assertEqual(core.saturate_cast(x, "float16").shape, (3, 5, 7))

    def test_scan(self):
        def f(carry, xs):
            xs = xs + carry
            return carry, carry

        init = KerasTensor(())
        xs = KerasTensor((6,))
        carry, result = core.scan(f, init, xs)
        self.assertEqual(carry.shape, ())
        self.assertEqual(result.shape, (6,))

        def f2(carry, _):
            return carry, carry

        carry, result = core.scan(f2, init, xs=None, length=3)
        self.assertEqual(carry.shape, ())
        self.assertEqual(result.shape, (3,))

    def test_scatter(self):
        indices = KerasTensor((5, 2))
        values = KerasTensor((5,))
        shape = (4, 4)
        self.assertEqual(core.scatter(indices, values, shape).shape, (4, 4))

    def test_scatter_update(self):
        inputs = KerasTensor((4, 4))
        indices = KerasTensor((5, 2))
        updates = KerasTensor((5,))
        self.assertEqual(
            core.scatter_update(inputs, indices, updates).shape, (4, 4)
        )

        inputs = KerasTensor((4, 4, 4))
        indices = KerasTensor((5, 2))
        updates = KerasTensor((5, 4))
        self.assertEqual(
            core.scatter_update(inputs, indices, updates).shape, (4, 4, 4)
        )

    def test_slice(self):
        inputs = KerasTensor(shape=(3, 3), dtype="float32")
        start_indices = KerasTensor(shape=(2,), dtype="int32")
        shape = (2, 2)
        self.assertEqual(core.slice(inputs, start_indices, shape).shape, (2, 2))

    def test_slice_negative_one_shape(self):
        inputs = KerasTensor(shape=(3, 3), dtype="float32")
        start_indices = (1, 1)
        shape = (-1, -1)
        self.assertEqual(core.slice(inputs, start_indices, shape).shape, (2, 2))

    def test_slice_negative_one_shape_tensor_indices(self):
        inputs = KerasTensor(shape=(3, 3), dtype="float32")
        start_indices = KerasTensor(shape=(2,), dtype="int32")
        shape = (-1, -1)
        self.assertEqual(
            core.slice(inputs, start_indices, shape).shape, (None, None)
        )

    def test_slice_negative_one_shape_dynamic_input_shape(self):
        inputs = KerasTensor(shape=(None, 3), dtype="float32")
        start_indices = (1, 1)
        shape = (-1, -1)
        self.assertEqual(
            core.slice(inputs, start_indices, shape).shape, (None, 2)
        )

    def test_slice_invalid_inputs(self):
        inputs = KerasTensor(shape=(3, 3), dtype="float32")
        start_indices = (1, 1)
        shape = (2, 2, 2)
        with self.assertRaisesRegex(
            ValueError,
            "dimensions in `inputs` must match.* dimensions in `shape`",
        ):
            core.slice(inputs, start_indices, shape)

        start_indices = (1, 1, 1)
        shape = (2, 2)
        with self.assertRaisesRegex(
            ValueError,
            "dimensions in `start_indices` must match.* dimensions in `inputs`",
        ):
            core.slice(inputs, start_indices, shape)

    def test_slice_update(self):
        inputs = KerasTensor((4, 4))
        start_indices = KerasTensor((2,))
        updates = KerasTensor((2, 2))
        self.assertEqual(
            core.slice_update(inputs, start_indices, updates).shape, (4, 4)
        )

        inputs = KerasTensor((4, 4, 4))
        start_indices = KerasTensor((3,))
        updates = KerasTensor((2, 2, 2))
        self.assertEqual(
            core.slice_update(inputs, start_indices, updates).shape, (4, 4, 4)
        )

    def test_stop_gradient(self):
        variable = KerasTensor(shape=(3, 3), dtype="float32")
        self.assertEqual(core.stop_gradient(variable).shape, (3, 3))

    def test_switch(self):
        def fn(x, y):
            return x[:, 0], y[0, :]

        index = KerasTensor(())
        x = KerasTensor((5, 2))
        y = KerasTensor((5, 2))
        self.assertEqual(core.switch(index, [fn], x, y)[0].shape, (5,))
        self.assertEqual(core.switch(index, [fn], x, y)[1].shape, (2,))

    def test_vectorized_map(self):
        def f(x):
            return x**2

        xs = KerasTensor((6, 5))
        ys = core.vectorized_map(f, xs)
        self.assertEqual(ys.shape, (6, 5))

        # Test nested output
        def f2(x):
            return {"a": x**2, "b": x * 10}

        xs = KerasTensor((6, 5))
        ys = core.vectorized_map(f2, xs)
        self.assertEqual(ys["a"].shape, (6, 5))
        self.assertEqual(ys["b"].shape, (6, 5))

        # Test nested input
        def f3(x):
            return x[0] + x[1]

        xs = (KerasTensor((6, 5)), KerasTensor((6, 5)))
        self.assertEqual(core.vectorized_map(f3, xs).shape, (6, 5))

    def test_while_loop(self):
        def cond(args):
            return tree.flatten(args)[0] < 10

        def body(args):
            return tree.map_structure(lambda x: x + 1, args)

        loop_vars = KerasTensor((10,))
        self.assertEqual(core.while_loop(cond, body, loop_vars).shape, (10,))

    def test_unstack(self):
        x = KerasTensor((2, 3, 4))
        axis = 1
        out = core.unstack(x, axis=axis)
        self.assertEqual(len(out), 3)
        for o in out:
            self.assertEqual(o.shape, (2, 4))

    def test_associative_scan_dynamic_scan_body_static_length_two(self):
        """_dynamic_scan_body: static elem_length == 2
        hits base-case-two branch."""
        xs = KerasTensor((2, 4))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (2, 4))

    def test_associative_scan_dynamic_scan_body_static_length_three(self):
        """_dynamic_scan_body: static elem_length == 3
        hits base-case-three branch."""
        xs = KerasTensor((3, 4))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (3, 4))

    def test_associative_scan_dynamic_scan_body_static_even_length(self):
        """_dynamic_scan_body: static even elem_length
        takes the recursive even path."""
        xs = KerasTensor((8, 5))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (8, 5))

    def test_associative_scan_dynamic_scan_body_static_odd_length(self):
        """_dynamic_scan_body: static odd elem_length > 3
        takes the recursive odd path."""
        xs = KerasTensor((7, 5))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (7, 5))

    def test_associative_scan_dynamic_scan_body_static_length_one(self):
        """_dynamic_scan_body: static elem_length == 1
        returns elems unchanged."""
        xs = KerasTensor((1, 4))
        ys = core.associative_scan(f=operator.add, elems=xs, axis=0)
        self.assertEqual(ys.shape, (1, 4))


class CoreOpsCorrectnessTest(testing.TestCase):
    def test_associative_scan(self):
        # Test prefix sum
        arr = np.arange(5)
        result = core.associative_scan(f=operator.add, elems=arr)
        self.assertAllClose(result, [0, 1, 3, 6, 10])
        # Test reverse
        result = core.associative_scan(f=operator.add, elems=arr, reverse=True)
        self.assertAllClose(result, [10, 10, 9, 7, 4])

        # Test multiple dimensions, across different axes
        batched_arr = np.stack([arr, arr + 1, arr + 2])
        result = core.associative_scan(
            f=operator.add, elems=batched_arr, axis=1
        )
        self.assertAllClose(result[2], [2, 5, 9, 14, 20])
        result = core.associative_scan(
            f=operator.add, elems=batched_arr, axis=0
        )
        self.assertAllClose(result[:, 0], [0, 1, 3])

        # Test structured input
        elems = {
            "a": np.array([[0, 1, 2], [3, 4, 5]]),
            "b": np.array([[6, 7, 8], [9, 10, 11]]),
        }

        def _dict_add(x, y):
            return {"a": x["a"] + y["b"], "b": x["b"] + y["b"]}

        ax0 = core.associative_scan(f=_dict_add, elems=elems, axis=0)
        self.assertAllClose(
            ax0["b"],
            [[6, 7, 8], [15, 17, 19]],
        )

        # Test parallel scan op used in mamba
        b, l, d, n = 1, 2, 3, 4
        DB = np.random.rand(b, l, d, n)
        DA = np.random.rand(b, l, d, n)

        H_seq = np.zeros((b, d, n))
        for i in range(l):
            H_seq = DA[:, i] * H_seq + DB[:, i]

        def scan_op(ci, cj):
            a = cj[0] * ci[0]
            b = cj[0] * ci[1] + cj[1]
            return (a, b)

        inputs = (DA.transpose(1, 0, 2, 3), DB.transpose(1, 0, 2, 3))
        H_par = core.associative_scan(f=scan_op, elems=inputs)[-1][-1]

        self.assertAllClose(H_seq, H_par)

        # Test Operation call.
        xs = np.arange(5, dtype="float32")
        self.assertAllClose(
            core.AssociativeScan()(operator.add, xs), ops.cumsum(xs)
        )

    def test_cast(self):
        x = ops.ones((2,), dtype="float32")
        y = ops.cast(x, "float16")
        self.assertIn("float16", str(y.dtype))

        x = ops.KerasTensor((2,), dtype="float32")
        y = ops.cast(x, "float16")
        self.assertEqual("float16", y.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(hasattr(y, "_keras_history"))

        # Test Operation call.
        x = ops.ones((2,), dtype="float32")
        self.assertDType(core.Cast("float16")(x), "float16")

    @parameterized.named_parameters(
        ("float8_e4m3fn", "float8_e4m3fn"), ("float8_e5m2", "float8_e5m2")
    )
    def test_cast_float8(self, float8_dtype):
        # Cast to float8 and cast back
        x = ops.ones((2,), dtype="float32")
        y = ops.cast(x, float8_dtype)
        self.assertIn(float8_dtype, str(y.dtype))
        x = ops.cast(y, "float32")
        self.assertIn("float32", str(x.dtype))

        x = ops.KerasTensor((2,), dtype="float32")
        y = ops.cast(x, float8_dtype)
        self.assertEqual(float8_dtype, y.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(hasattr(y, "_keras_history"))
        x = ops.cast(y, "float32")
        self.assertEqual("float32", x.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(hasattr(x, "_keras_history"))

    def test_cond(self):
        t = ops.cond(True, lambda: 0, lambda: 1)
        self.assertEqual(t, 0)
        f = ops.cond(False, lambda: 0, lambda: 1)
        self.assertEqual(f, 1)
        f = ops.cond(False, lambda: None, lambda: None)
        self.assertEqual(f, None)

        out = ops.cond(
            ops.convert_to_tensor(True),
            lambda: ops.ones((1, 3)),
            lambda: ops.zeros((1, 3)),
        )
        self.assertAllClose(out, ops.ones((1, 3)))

        out = ops.cond(
            ops.convert_to_tensor(False),
            lambda: ops.ones((3,)),
            lambda: ops.zeros((3,)),
        )
        self.assertAllClose(out, ops.zeros((3,)))

        with self.assertRaises(ValueError):
            ops.cond(
                KerasTensor((), dtype="bool"),
                lambda: ops.ones((3,)),
                lambda: ops.zeros((4,)),
            )

    def test_convert_to_tensor(self):
        x = np.ones((2,))
        x = ops.convert_to_tensor(x)
        x = ops.convert_to_numpy(x)
        self.assertAllClose(x, (1, 1))
        self.assertIsInstance(x, np.ndarray)

        # Empty lists should give an empty array.
        x = ops.convert_to_tensor([])
        np_x = ops.convert_to_numpy(x)
        self.assertTrue(ops.is_tensor(x))
        self.assertAllClose(x, [])
        self.assertIsInstance(np_x, np.ndarray)

        # Partially converted.
        x = ops.convert_to_tensor((1, ops.array(2), 3))
        self.assertAllClose(x, (1, 2, 3))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason=f"{backend.backend()} backend doesn't support sparse tensors.",
    )
    def test_convert_to_tensor_sparse(self):
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            x = jax_sparse.BCOO(([1.0, 2.0], [[0, 0], [1, 2]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        x_default = ops.convert_to_tensor(x)
        self.assertSparse(x_default)
        self.assertAllClose(x, x_default)
        x_sparse = ops.convert_to_tensor(x, sparse=True)
        self.assertSparse(x_sparse)
        self.assertAllClose(x, x_sparse)
        x_dense = ops.convert_to_tensor(x, sparse=False)
        self.assertSparse(x_dense, False)
        self.assertAllClose(x, x_dense)

        x_numpy = ops.convert_to_numpy(x)
        self.assertIsInstance(x_numpy, np.ndarray)
        self.assertAllClose(x_numpy, x_dense)

    @pytest.mark.skipif(
        not backend.SUPPORTS_RAGGED_TENSORS,
        reason=f"{backend.backend()} backend doesn't support ragged tensors.",
    )
    def test_convert_to_tensor_ragged(self):
        import tensorflow as tf

        x = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])

        x_default = ops.convert_to_tensor(x)
        self.assertIsInstance(x_default, tf.RaggedTensor)
        self.assertAllClose(x, x_default)
        x_ragged = ops.convert_to_tensor(x, ragged=True)
        self.assertIsInstance(x_ragged, tf.RaggedTensor)
        self.assertAllClose(x, x_ragged)
        x_dense = ops.convert_to_tensor(x, ragged=False)
        self.assertNotIsInstance(x_dense, tf.RaggedTensor)
        self.assertAllClose(x, x_dense)

        x_numpy = ops.convert_to_numpy(x)
        self.assertIsInstance(x_numpy, np.ndarray)
        self.assertAllClose(x_numpy, x_dense)

    @parameterized.named_parameters(named_product(use_variable=(False, True)))
    @pytest.mark.skipif(
        not backend.SUPPORTS_GRADIENT,
        reason="Backend does not support gradients.",
    )
    def test_custom_gradient(self, use_variable):
        # function to test custom_gradient on
        @ops.custom_gradient
        def log1pexp(x):
            e = ops.exp(x)

            def grad(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))

            return ops.log(1 + e), grad

        def log1pexp_nan(x):
            return ops.log(1 + ops.exp(x))

        x = ops.convert_to_tensor(100.0)
        if use_variable:

            class Log1PExpLayer(layers.Layer):
                def __init__(self):
                    super().__init__()
                    self.v = backend.Variable(5.0, trainable=False)

                def call(self, inputs):
                    # The derivative of this layer is 1 with respect to inputs.
                    # But on the side, we test passing a variable to a function
                    # using @custom_gradient
                    return log1pexp(self.v) + inputs

            to_derive = Log1PExpLayer()
        else:
            to_derive = log1pexp

        dy_dx = ops.grad(to_derive)(x)
        self.assertEqual(ops.convert_to_numpy(dy_dx), 1.0)
        if backend.backend() != "tensorflow":
            # TensorFlow's `log` and `exp` gradients are defined at 100.0.
            dz_dx = ops.grad(log1pexp_nan)(x)
            self.assertTrue(ops.isnan(dz_dx))

    def test_custom_gradient_forward_error(self):
        calls = 0

        @ops.custom_gradient
        def failing(x):
            nonlocal calls
            calls += 1
            raise RuntimeError("Forward failed.")

        with self.assertRaisesRegex(RuntimeError, "Forward failed."):
            failing(ops.ones((3,)))
        # The forward runs once and its error reaches the caller.
        self.assertEqual(calls, 1)

    def test_dynamic_slice(self):
        def cond(index, inputs, sum):
            return index < 10

        def body(index, inputs, sum):
            sum = sum + core.slice(inputs, [index], [1])
            index = index + 1
            return index, inputs, sum

        index, inputs, sum = 0, np.arange(10), np.array([0])
        index, inputs, sum = core.while_loop(cond, body, (index, inputs, sum))
        self.assertEqual(sum.shape, (1,))
        self.assertAllClose(sum, [45])

    def test_fori_loop(self):
        def body_fun(i, x):
            return x + i

        initial_value = 0
        result = core.fori_loop(0, 10, body_fun, initial_value)
        self.assertAllClose(result, 45)

        # Test Operation call.
        self.assertAllClose(core.ForiLoop(0, 10, body_fun)(initial_value), 45)

    def test_getitem(self):
        np_tensor = np.arange(24).reshape(2, 3, 4)
        tensor = ops.convert_to_tensor(np_tensor)

        t = tensor[1]
        n = np_tensor[1]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1, 2, 3]
        n = np_tensor[1, 2, 3]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2]
        n = np_tensor[1:2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2, 2:3, 3:4]
        n = np_tensor[1:2, 2:3, 3:4]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2, None]
        n = np_tensor[1:2, None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2, 2:3, ...]
        n = np_tensor[1:2, 2:3, ...]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2, ..., 3:4]
        n = np_tensor[1:2, ..., 3:4]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[None, ..., 3:4, None]
        n = np_tensor[None, ..., 3:4, None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1:2:None]
        n = np_tensor[1:2:None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[:, 2]
        n = np_tensor[:, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[None]
        n = np_tensor[None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[None, None]
        n = np_tensor[None, None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[...]
        n = np_tensor[...]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[..., 1]
        n = np_tensor[..., 1]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[..., 1, 2]
        n = np_tensor[..., 1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[..., -1, 2]
        n = np_tensor[..., -1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[..., -1:-2, 2]
        n = np_tensor[..., -1:-2, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[..., None, None]
        n = np_tensor[..., None, None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[None, ..., None]
        n = np_tensor[None, ..., None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1, 2, None, ..., None]
        n = np_tensor[1, 2, None, ..., None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[None, ..., 1, 2]
        n = np_tensor[None, ..., 1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1, None, 2]
        n = np_tensor[1, None, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        index_tensor = ops.convert_to_tensor(np.array(1, dtype=np.int32))
        t = tensor[index_tensor]
        n = np_tensor[ops.convert_to_numpy(index_tensor)]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        index_tensor = ops.convert_to_tensor(np.array(1, dtype=np.int32))
        t = tensor[index_tensor, 2, None]
        n = np_tensor[ops.convert_to_numpy(index_tensor), 2, None]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        index_tensor = ops.convert_to_tensor(np.array(-2, dtype=np.int32))
        t = tensor[index_tensor, 1]
        n = np_tensor[ops.convert_to_numpy(index_tensor), 1]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        index_tensor = ops.convert_to_tensor(np.array(-1, dtype=np.int32))
        t = tensor[-2, index_tensor]
        n = np_tensor[-2, ops.convert_to_numpy(index_tensor)]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        # Negative indexing
        t = tensor[-1]
        n = np_tensor[-1]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[1, -1, -2]
        n = np_tensor[1, -1, -2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        # Slicing with step
        t = tensor[::2]
        n = np_tensor[::2]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        # Mixed slices and integers
        t = tensor[1, :, 1:4]
        n = np_tensor[1, :, 1:4]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

        t = tensor[:, 1:2, 3]
        n = np_tensor[:, 1:2, 3]
        self.assertEqual(t.shape, n.shape)
        self.assertAllClose(t, n)

    def test_is_tensor(self):
        np_x = np.array([[1, 2, 3], [3, 2, 1]])
        x = backend.ops.convert_to_tensor(np_x)
        if backend.backend() != "numpy":
            self.assertFalse(ops.is_tensor(np_x))
        self.assertTrue(ops.is_tensor(x))
        self.assertFalse(ops.is_tensor([1, 2, 3]))

    def test_map(self):
        def f(x):
            return x**2

        xs = np.arange(10)
        self.assertAllClose(ops.map(f, xs), xs**2)

        # Test nested output
        def f2(x):
            return {"a": x**2, "b": x * 10}

        xs = np.random.rand(2, 3, 4).astype("float32")
        outputs = ops.map(f2, xs)
        self.assertAllClose(outputs["a"], xs**2)
        self.assertAllClose(outputs["b"], xs * 10)

        # Test with nested structures
        def dict_input_fn(inputs):
            x = inputs["x"][:, 0]
            y = inputs["y"] + 1
            return {"x": x, "y": y}

        def list_input_fn(inputs):
            return [x**2 for x in inputs]

        xs = {
            "x": ops.convert_to_tensor(
                np.random.rand(4, 100, 3), dtype="float32"
            ),
            "y": ops.convert_to_tensor(
                np.random.randint(0, 10, size=(4, 1)), dtype="int32"
            ),
        }
        xs1 = [
            ops.convert_to_tensor(np.random.rand(4, 100, 3), dtype="float32"),
            ops.convert_to_tensor(
                np.random.randint(0, 10, size=(4, 1)), dtype="int32"
            ),
        ]
        ys = ops.map(dict_input_fn, xs)
        self.assertEqual(ys["x"].shape, (4, 100))
        self.assertEqual(
            ops.convert_to_numpy(ys["y"]).all(),
            ops.convert_to_numpy(xs["y"] + 1).all(),
        )
        ys = ops.map(list_input_fn, xs1)
        for x, y in zip(xs1, ys):
            self.assertEqual(
                (ops.convert_to_numpy(y)).all(),
                (ops.convert_to_numpy(x) ** 2).all(),
            )

        # Test Operation call.
        xs = np.arange(10)
        self.assertAllClose(ops.Map()(f, xs), xs**2)

    def test_saturate_cast(self):
        x = ops.ones((2,), dtype="float32")
        y = ops.saturate_cast(x, "float16")
        self.assertIn("float16", str(y.dtype))

        x = ops.KerasTensor((2,), dtype="float32")
        y = ops.saturate_cast(x, "float16")
        self.assertEqual("float16", y.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(hasattr(y, "_keras_history"))

        # Test Operation call.
        x = np.array([-256, 1.0, 257.0], dtype="float32")
        y = core.SaturateCast("uint8")(x)
        self.assertDType(y, "uint8")
        # Check that the values are the same
        self.assertAllClose(y, np.clip(x, 0, 255).astype("uint8"))

    def test_scan(self):
        # Test cumsum
        def cumsum(carry, xs):
            carry = carry + xs
            return carry, carry

        init = np.array(0, dtype="float32")
        xs = np.array([1, 2, 3, 4, 10, 20], dtype="float32")
        carry, result = core.scan(cumsum, init, xs)
        self.assertAllClose(carry, 40.0)
        self.assertAllClose(result, ops.cumsum(xs))

        # Test reverse=True
        carry, result = core.scan(cumsum, init, xs, reverse=True)
        self.assertAllClose(carry, 40.0)
        self.assertAllClose(result, [40, 39, 37, 34, 30, 20])

        # Test unroll
        for unroll in (True, False, 2):
            carry, result = core.scan(cumsum, init, xs, unroll=unroll)
            self.assertAllClose(carry, 40.0)
            self.assertAllClose(result, ops.cumsum(xs))

        # Test xs is None
        def fibonaccis(carry, _):
            return (carry[1], carry[0] + carry[1]), None

        init = (np.array(0, dtype="float32"), np.array(1, dtype="float32"))
        (carry1, carry2), _ = core.scan(fibonaccis, init, length=6)
        self.assertAllClose(carry1, 8)
        self.assertAllClose(carry2, 13)

        # Test nested init
        if backend.backend() != "tensorflow":
            # tensorflow doesn't support arbitrary shape/dtype of the output of
            # `f`. It must be the same as `init`.
            def multiply_two(carry, _):
                value1 = carry["value1"]
                value2 = carry["value2"]
                return (
                    {"value1": value1 * 2, "value2": value2 * 2},
                    value1 * 2 + value2 * 2,
                )

            init = {"value1": 2.0, "value2": 3.0}
            carry, result = core.scan(multiply_two, init, length=3)
            self.assertAllClose(carry["value1"], 16)
            self.assertAllClose(carry["value2"], 24)
            self.assertAllClose(result, [10, 20, 40])

        # Test nested xs
        def reduce_add(carry, xs):
            value1 = xs["value1"]
            value2 = xs["value2"]
            return carry, value1 + value2

        init = np.array(0, dtype="float32")
        xs = {
            "value1": np.array([1, 2, 3], dtype="float32"),
            "value2": np.array([10, 20, 30], dtype="float32"),
        }
        _, result = core.scan(reduce_add, init, xs)
        self.assertAllClose(result, [11, 22, 33])

        # Test Operation call.
        init = np.array(0, dtype="float32")
        xs = np.array([1, 2, 3, 4, 10, 20], dtype="float32")
        carry, result = core.Scan()(cumsum, init, xs)
        self.assertAllClose(carry, 40.0)
        self.assertAllClose(result, ops.cumsum(xs))

    def test_scatter(self):
        # Test 1D
        indices = np.array([[1], [3], [4], [7]])
        values = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.scatter(indices, values, (8,)),
            [0, 9, 0, 10, 11, 0, 0, 12],
        )
        # Test 2D
        indices = np.array([[0, 1], [2, 0]])
        values = np.array([5, 10])
        self.assertAllClose(
            core.scatter(indices, values, (3, 2)), [[0, 5], [0, 0], [10, 0]]
        )
        # Test 3D
        indices = np.array([[1], [3]])
        values = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ]
        )
        self.assertAllClose(
            core.scatter(indices, values, (4, 4, 4)),
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ],
        )
        # Test slices
        indices = np.array([[2], [4]])
        values = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertAllClose(
            core.scatter(indices, values, (6, 3)),
            [[0, 0, 0], [0, 0, 0], [1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]],
        )
        # Duplicate indices
        indices = np.array([[0], [0]])
        values = np.array([1, 1])
        self.assertAllClose(core.scatter(indices, values, (1,)), [2])

        # Test Operation call.
        indices = np.array([[1, 0], [0, 1]])
        values = np.array([10, 20])
        shape = (2, 2)
        self.assertAllClose(
            core.Scatter(shape)(indices, values), np.array([[0, 20], [10, 0]])
        )

    def test_scatter_update(self):
        # Test 1D.
        inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        indices = [[1], [3], [4], [7]]
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.scatter_update(inputs, indices, updates),
            [0, 9, 0, 10, 11, 0, 0, 12],
        )

        # Test 2D.
        inputs = np.array([[1, 1], [1, 1], [1, 1]])
        indices = [[0, 1], [2, 0]]
        updates = np.array([5, 10])
        self.assertAllClose(
            core.scatter_update(inputs, indices, updates),
            [[1, 5], [1, 1], [10, 1]],
        )

        # Test updates has multiple dimension.
        inputs = np.ones([4, 4, 4])
        indices = [[1, 1], [2, 2]]
        updates = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype="float32")
        outputs = core.scatter_update(inputs, indices, updates)
        self.assertTrue(ops.is_tensor(outputs))
        self.assertAllClose(outputs[1, 1, :], [0, 1, 2, 3])
        self.assertAllClose(outputs[2, 2, :], [3, 2, 1, 0])

        # Test Operation call.
        inputs = np.array([[0, 0], [0, 0]])
        indices = np.array([[1, 0], [0, 1]])
        updates = np.array([10, 20])
        self.assertAllClose(
            core.ScatterUpdate()(inputs, indices, updates),
            np.array([[0, 20], [10, 0]]),
        )

    def test_scatter_update_with_reduction(self):
        # Test add reduction with duplicate indices
        inputs = np.zeros((4,))
        indices = [[0], [0], [1]]
        updates = np.array([1.0, 2.0, 3.0])
        result = core.scatter_update(inputs, indices, updates, reduction="add")
        self.assertAllClose(result, [3.0, 3.0, 0.0, 0.0])

        # Test add reduction 2D
        inputs = np.zeros((3, 3))
        indices = [[0, 0], [1, 1], [0, 0]]
        updates = np.array([1.0, 2.0, 3.0])
        result = core.scatter_update(inputs, indices, updates, reduction="add")
        self.assertAllClose(result[0, 0], 4.0)
        self.assertAllClose(result[1, 1], 2.0)

        # Test max reduction with duplicates
        inputs = np.zeros((4,))
        indices = [[0], [0], [1]]
        updates = np.array([3.0, 5.0, 2.0])
        result = core.scatter_update(inputs, indices, updates, reduction="max")
        self.assertAllClose(result, [5.0, 2.0, 0.0, 0.0])

        # Test min reduction
        inputs = np.array([10.0, 10.0, 10.0, 10.0])
        indices = [[0], [0], [1]]
        updates = np.array([3.0, 5.0, 2.0])
        result = core.scatter_update(inputs, indices, updates, reduction="min")
        self.assertAllClose(result, [3.0, 2.0, 10.0, 10.0])

        # Test mul reduction
        inputs = np.array([2.0, 3.0, 1.0, 1.0])
        indices = [[0], [0], [1]]
        updates = np.array([3.0, 5.0, 2.0])
        result = core.scatter_update(inputs, indices, updates, reduction="mul")
        self.assertAllClose(result, [30.0, 6.0, 1.0, 1.0])

        # Test Operation call with reduction
        inputs = np.zeros((4,))
        indices = [[0], [0], [1]]
        updates = np.array([1.0, 2.0, 3.0])
        result = core.ScatterUpdate(reduction="add")(inputs, indices, updates)
        self.assertAllClose(result, [3.0, 3.0, 0.0, 0.0])

    def test_shape(self):
        x = ops.ones((2, 3, 7, 1))
        self.assertEqual(core.shape(x).__class__, tuple)
        self.assertEqual(core.shape(x), (2, 3, 7, 1))

        x = KerasTensor((None, 3, None, 1))
        self.assertEqual(core.shape(x).__class__, tuple)
        self.assertEqual(core.shape(x), (None, 3, None, 1))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason=f"{backend.backend()} backend doesn't support sparse tensors.",
    )
    def test_shape_sparse(self):
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            x = jax_sparse.BCOO(([1.0, 2.0], [[0, 0], [1, 2]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        self.assertEqual(core.shape(x), (2, 3))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason=f"{backend.backend()} backend doesn't support ragged tensors.",
    )
    def test_shape_ragged(self):
        import tensorflow as tf

        x = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
        self.assertEqual(core.shape(x), (5, None))

        x = tf.RaggedTensor.from_row_lengths(tf.zeros([15, 2]), [4, 5, 6])
        self.assertEqual(core.shape(x), (3, None, 2))

    def test_slice(self):
        # Test 1D.
        inputs = np.arange(10)
        start_indices = np.array([1])
        shape = np.array([4])
        self.assertAllClose(
            core.slice(inputs, start_indices, shape),
            [1, 2, 3, 4],
        )

        # Test 2D.
        inputs = np.broadcast_to(np.arange(10), (4, 10))
        start_indices = np.array([1, 1])
        shape = np.array([2, 4])
        self.assertAllClose(
            core.slice(inputs, start_indices, shape),
            [[1, 2, 3, 4], [1, 2, 3, 4]],
        )

        # Test N-D.
        inputs = np.broadcast_to(np.arange(10), (4, 4, 4, 10))
        start_indices = np.array([1, 1, 1, 1])
        shape = np.array([1, 2, 3, 4])
        outputs = core.slice(inputs, start_indices, shape)
        expected = np.broadcast_to(np.arange(1, 5), (1, 2, 3, 4))
        self.assertAllClose(outputs, expected)

        # Test Operation call.
        inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        start_indices = np.array([1, 1])
        shape = (2, 2)
        self.assertAllClose(
            core.Slice(shape)(inputs, start_indices), np.array([[5, 6], [8, 9]])
        )

    def test_slice_update(self):
        # Test 1D.
        inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        start_indices = np.array([1])
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            core.slice_update(inputs, start_indices, updates),
            [0, 9, 10, 11, 12, 0, 0, 0],
        )

        # Test 2D.
        inputs = np.array([[1, 1], [1, 1], [1, 1]])
        start_indices = [1, 0]
        updates = np.array([[2, 2], [2, 2]])
        self.assertAllClose(
            core.slice_update(inputs, start_indices, updates),
            [[1, 1], [2, 2], [2, 2]],
        )

        # Test N-D.
        inputs = np.ones([4, 4, 4, 4])
        start_indices = [1, 1, 2, 2]
        updates = np.zeros([2, 2, 2, 2])
        outputs = core.slice_update(inputs, start_indices, updates)
        self.assertAllClose(outputs[1:3, 1:3, 2:4, 2:4], np.zeros([2, 2, 2, 2]))

        # Test Operation call.
        inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        start_indices = np.array([1, 1])
        updates = np.array([[10, 11], [12, 13]])
        self.assertAllClose(
            core.SliceUpdate()(inputs, start_indices, updates),
            np.array([[1, 2, 3], [4, 10, 11], [7, 12, 13]]),
        )

    @pytest.mark.requires_trainable_backend
    def test_stop_gradient(self):
        class ExampleLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(shape=(1,), initializer="zeros")
                self.b = self.add_weight(shape=(1,), initializer="zeros")

            def call(self, x, training=False):
                return ops.add(
                    ops.multiply(x, ops.stop_gradient(self.w)), self.b
                )

        model = models.Sequential([ExampleLayer()])
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        rng = np.random.default_rng(0)
        x = np.ones((2, 4), dtype="float32")
        y = rng.standard_normal((2, 4), dtype="float32")
        model.fit(x, y, epochs=1, batch_size=2)
        self.assertEqual(model.layers[0].w.numpy(), 0.0)
        self.assertNotEqual(model.layers[0].b.numpy(), 0.0)

    def test_stop_gradient_no_fit(self):
        x = random.uniform(shape=(2, 4), dtype="float32")
        y = ops.stop_gradient(x)
        self.assertAllClose(x, y)

        # Functional.
        a = layers.Input(shape=(2,))
        b = layers.Dense(4, kernel_initializer="ones", use_bias=False)(a)
        c = layers.Dense(4, kernel_initializer="ones", use_bias=False)(b)
        d = ops.stop_gradient(b) + c
        model = models.Model(inputs=a, outputs=d)
        output = model(ops.convert_to_tensor([[1.0, 2.0]]))
        self.assertAllClose(output, np.full((1, 4), 15.0))

        # Test Operation call.
        variable = ops.convert_to_tensor(
            np.array([1.0, 2.0, 3.0], dtype="float32")
        )
        self.assertAllClose(core.StopGradient()(variable), variable)

    def test_switch(self):
        def fn1(x, y):
            return x + y

        def fn2(x, y):
            return x - y

        x = np.random.rand(2, 3, 4).astype("float32")
        y = np.random.rand(2, 3, 4).astype("float32")
        branches = [fn1, fn2]
        self.assertAllClose(core.switch(0, branches, x, y), x + y)
        self.assertAllClose(core.switch(1, branches, x, y), x - y)

        # Test out-of-bound index
        self.assertAllClose(core.switch(-100, branches, x, y), x + y)
        self.assertAllClose(core.switch(100, branches, x, y), x - y)

        # Test Operation call.
        self.assertAllClose(core.Switch()(0, branches, x, y), x + y)
        self.assertAllClose(core.Switch()(1, branches, x, y), x - y)

    def test_vectorized_map(self):
        def fn(x):
            return x + 1

        output = ops.vectorized_map(fn, ops.zeros((2, 3), dtype="float32"))
        self.assertAllClose(
            backend.ops.convert_to_numpy(output), np.ones((2, 3))
        )

        def fn(x):
            return ops.stack([x, x])

        output = ops.vectorized_map(fn, ops.zeros((2, 3), dtype="float32"))
        self.assertAllClose(
            backend.ops.convert_to_numpy(output), np.zeros((2, 2, 3))
        )

        # Case: multiple args
        def fn(elems):
            x, y = elems
            return x + y

        output = ops.vectorized_map(fn, [ops.ones((2, 3)), ops.ones((2, 3))])
        self.assertAllClose(output, 2 * np.ones((2, 3)))

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "scalar_data_with_max",
                "loop_vars": np.array(0),
                "expected_output": np.array(5),
                "maximum_iterations": 5,
            },
            {
                "testcase_name": "scalar_data_no_max",
                "loop_vars": np.array(0),
                "expected_output": np.array(10),
                "maximum_iterations": None,
            },
            {
                "testcase_name": "nested_data_with_max",
                "loop_vars": {
                    "a": np.array(0),
                    "b": (np.array(1), np.array(2)),
                },
                "expected_output": {
                    "a": np.array(5),
                    "b": (np.array(6), np.array(7)),
                },
                "maximum_iterations": 5,
            },
            {
                "testcase_name": "nested_data_no_max",
                "loop_vars": {
                    "a": np.array(0),
                    "b": (np.array(1), np.array(2)),
                },
                "expected_output": {
                    "a": np.array(10),
                    "b": (np.array(11), np.array(12)),
                },
                "maximum_iterations": None,
            },
        ]
    )
    def test_while_loop(self, loop_vars, expected_output, maximum_iterations):
        def cond(args):
            return tree.flatten(args)[0] < 10

        def body(args):
            return tree.map_structure(lambda x: x + 1, args)

        output = core.while_loop(
            cond, body, loop_vars, maximum_iterations=maximum_iterations
        )
        tree.map_structure(self.assertAllClose, output, expected_output)

        # Test Operation call.
        output = core.WhileLoop(
            cond, body, maximum_iterations=maximum_iterations
        )(loop_vars)
        tree.map_structure(self.assertAllClose, output, expected_output)

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "with_max",
                "state": (np.array(0), np.array(1)),
                "output": (np.array(5), np.array(6)),
                "maximum_iterations": 5,
            },
            {
                "testcase_name": "no_max",
                "state": (np.array(0), np.array(1)),
                "output": (np.array(10), np.array(11)),
                "maximum_iterations": None,
            },
        ]
    )
    def test_while_loop_list_data(self, state, output, maximum_iterations):
        def cond(*args):
            return tree.flatten(args)[0] < 10

        def body(*args):
            return tree.map_structure(lambda x: x + 1, args)

        state = core.while_loop(
            cond, body, state, maximum_iterations=maximum_iterations
        )
        tree.map_structure(self.assertAllClose, state, output)

    def test_unstack(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(2, 3, 4))
        x_tensor = ops.convert_to_tensor(x)
        axis = 1
        out = ops.unstack(x_tensor, axis=axis)
        out_ex = [x[:, i, :] for i in range(x.shape[axis])]
        self.assertEqual(len(out), len(out_ex))
        for o, o_e in zip(out, out_ex):
            o = ops.convert_to_numpy(o)
            self.assertAllClose(o, o_e)

        # Test Operation call.
        out = ops.Unstack(axis=axis)(x_tensor)
        self.assertEqual(len(out), len(out_ex))
        for o, o_e in zip(out, out_ex):
            o = ops.convert_to_numpy(o)
            self.assertAllClose(o, o_e)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Dynamic scan fallback is TF-specific.",
    )
    def test_associative_scan_dynamic_scan_body_tf_tensorspec_dynamic_fallback(
        self,
    ):
        """Exercises _dynamic_scan_body base-case paths in eager mode."""
        import tensorflow as tf

        def run_scan(x):
            return core.associative_scan(f=operator.add, elems=x, axis=0)

        # length == 2
        x2 = tf.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self.assertAllClose(run_scan(x2), np.cumsum(x2.numpy(), axis=0))

        # length == 3
        x3 = tf.constant(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
        )
        self.assertAllClose(run_scan(x3), np.cumsum(x3.numpy(), axis=0))

        # length == 1  (trivial / no-op path)
        x1 = tf.constant([[10.0, 20.0, 30.0, 40.0]])
        self.assertAllClose(run_scan(x1), x1.numpy())

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Dynamic scan fallback is TF-specific.",
    )
    def test_associative_scan_dynamic_scan_body_tf_tensorspec_reverse(self):
        """Dynamic fallback with reverse=True."""
        import tensorflow as tf

        # length == 3 with reverse
        arr = tf.constant([1.0, 2.0, 3.0])
        result = core.associative_scan(
            f=operator.add, elems=arr, axis=0, reverse=True
        )
        # reverse prefix-sum: [1+2+3, 2+3, 3]
        expected = np.array([6.0, 5.0, 3.0], dtype="float32")
        self.assertAllClose(result, expected)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Dynamic scan fallback is TF-specific.",
    )
    def test_associative_scan_dynamic_scan_body_tf_tensorspec_structured_input(
        self,
    ):
        """Dynamic fallback with a tuple of tensors, lengths <= 3."""
        import tensorflow as tf

        def _add(x, y):
            return (x[0] + y[0], x[1] + y[1])

        # length == 3
        a = tf.constant(np.ones((3, 3), dtype="float32"))
        b = tf.constant(np.ones((3, 3), dtype="float32") * 2.0)
        ra, rb = core.associative_scan(f=_add, elems=(a, b), axis=0)

        self.assertAllClose(ra, np.cumsum(a.numpy(), axis=0))
        self.assertAllClose(rb, np.cumsum(b.numpy(), axis=0))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Dynamic scan fallback is TF-specific.",
    )
    def test_associative_scan_dynamic_scan_body_tf_function_dynamic_lengths(
        self,
    ):
        """Exercises the recursive unwind path for lengths > 3 under
        tf.function with fully dynamic input_signature."""
        import tensorflow as tf

        @tf.function(input_signature=[tf.TensorSpec([None, 4])])
        def run_scan(x):
            return core.associative_scan(f=operator.add, elems=x, axis=0)

        for length in [1, 2, 3, 4, 5, 7, 8, 16]:
            x = tf.constant(
                np.arange(length * 4, dtype="float32").reshape(length, 4)
            )
            result = run_scan(x)
            expected = np.cumsum(x.numpy(), axis=0)
            self.assertAllClose(
                result,
                expected,
                msg=f"Failed for dynamic length={length}",
            )


class CoreOpsDtypeTest(testing.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x
        not in (
            "string",
            "complex64",
            "complex128",
            # Remove 64-bit dtypes.
            "float64",
            "uint64",
            "int64",
        )
        + dtypes.FLOAT8_TYPES  # Remove float8 dtypes for the following tests
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]

    if backend.backend() == "torch":
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint16", "uint32")]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]

    @parameterized.named_parameters(
        named_product(
            dtype=[dtype for dtype in ALL_DTYPES if dtype is not None]
        )
    )
    def test_cast(self, dtype):
        x = np.ones((1,))

        self.assertDType(core.cast(x, dtype), dtype)
        self.assertDType(core.Cast(dtype).symbolic_call(x), dtype)

    @parameterized.parameters(
        ((), None, backend.floatx()),
        ([], None, backend.floatx()),
        (bool(0), None, "bool"),
        (int(0), None, "int32"),
        (float(0), None, backend.floatx()),
        (1, "bool", "bool"),
        (1.0, "int32", "int32"),
        (1.0, "float32", "float32"),
        ([False, True, False], None, "bool"),
        ([1, 2, 3], None, "int32"),
        ([1.0, 2.0, 3.0], None, backend.floatx()),
        ([1, 2.0, 3], None, backend.floatx()),
        ([[False], [True], [False]], None, "bool"),
        ([[1], [2], [3]], None, "int32"),
        ([[1], [2.0], [3]], None, backend.floatx()),
        *[
            (np.array(0, dtype=dtype), None, dtype)
            for dtype in ALL_DTYPES
            if dtype is not None
        ],
        *[
            ([[1, 0, 1], [1, 1, 0]], dtype, dtype)
            for dtype in ALL_DTYPES
            if dtype is not None
        ],
    )
    def test_convert_to_tensor(self, x, dtype, expected_dtype):
        self.assertDType(ops.convert_to_tensor(x, dtype=dtype), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_convert_to_numpy(self, dtype):
        expected_dtype = backend.floatx() if dtype is None else dtype

        x = ops.array([1.0, 2.0, 3.0], dtype=dtype)
        self.assertDType(ops.convert_to_numpy(x), expected_dtype)

        x = ops.array(4.0, dtype=dtype)
        self.assertDType(ops.convert_to_numpy(x), expected_dtype)

    @parameterized.named_parameters(
        named_product(
            dtype=[dtype for dtype in ALL_DTYPES if dtype is not None]
        )
    )
    def test_convert_to_tensor_with_tensor(self, dtype):
        x = ops.convert_to_tensor(np.ones((2, 3), dtype="float32"))

        self.assertDType(ops.convert_to_tensor(x, dtype=dtype), dtype)

    @parameterized.named_parameters(
        named_product(
            dtype=[dtype for dtype in ALL_DTYPES if dtype is not None]
        )
    )
    def test_convert_to_tensor_with_variable(self, dtype):
        x = backend.Variable(np.ones((2, 3), dtype="float32"))

        self.assertDType(ops.convert_to_tensor(x, dtype=dtype), dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_saturate_cast(self, dtype):
        x = np.ones((1,))

        self.assertDType(core.saturate_cast(x, dtype), dtype)
        self.assertDType(core.SaturateCast(dtype).symbolic_call(x), dtype)


class ConvertToTensorFloatxTest(testing.TestCase):
    """`convert_to_tensor` must not let `floatx` override an input's dtype.

    The jax and numpy backends gated their bfloat16 fast path on
    `standardize_dtype(dtype) == "bfloat16"`. Because
    `standardize_dtype(None)` returns `floatx()`, that branch fired for
    *every* call without an explicit `dtype` once `floatx` was
    `"bfloat16"`, and the subsequent `astype(None)` silently produced
    float32 on jax and float64 on numpy.
    """

    # 64-bit dtypes are excluded: jax truncates them to 32-bit unless
    # x64 is enabled, which is unrelated to this behavior.
    PRESERVED_DTYPES = ["bool", "uint8", "int32", "float16", "float32"]

    def setUp(self):
        super().setUp()
        self._floatx = backend.floatx()

    def tearDown(self):
        super().tearDown()
        backend.set_floatx(self._floatx)

    @parameterized.named_parameters(
        named_product(floatx=["float32", "bfloat16"])
    )
    def test_preserves_input_dtype(self, floatx):
        """An input that already carries a dtype keeps it."""
        backend.set_floatx(floatx)
        for dtype in self.PRESERVED_DTYPES:
            x = np.ones((2,), dtype=dtype)
            self.assertDType(
                ops.convert_to_tensor(x),
                dtype,
                msg=f"floatx={floatx}, input dtype={dtype}",
            )

    @parameterized.named_parameters(
        named_product(floatx=["float32", "bfloat16"])
    )
    def test_explicit_bfloat16_is_honored(self, floatx):
        """An explicit `dtype="bfloat16"` still works, for any input type."""
        backend.set_floatx(floatx)
        for x in (
            np.ones((2,), dtype="int32"),
            np.ones((2,), dtype="float32"),
            [1, 2, 3],
            (1.0, 2.0),
            1.0,
        ):
            self.assertDType(
                ops.convert_to_tensor(x, dtype="bfloat16"),
                "bfloat16",
                msg=f"floatx={floatx}, input={x}",
            )

    @parameterized.named_parameters(
        named_product(floatx=["float32", "bfloat16"])
    )
    def test_python_ints_do_not_become_floats(self, floatx):
        """Python ints and int sequences must not turn into floats.

        Unlike a numpy array, a Python list or scalar is not a backend
        tensor, so it takes the `bfloat16` branch. The numpy backend
        returned float64 for these until the same guard was applied
        there. The exact integer width is backend dependent (it follows
        `floatx` precision via `_lattice_result_type`), so only the kind
        of the dtype is asserted.
        """
        backend.set_floatx(floatx)
        for x in ([1, 2, 3], 3, (1, 2)):
            dtype = standardize_dtype(ops.convert_to_tensor(x).dtype)
            self.assertIn(
                dtype,
                dtypes.INT_TYPES,
                msg=f"floatx={floatx}, input={x}, got {dtype}",
            )
        self.assertDType(ops.convert_to_tensor(True), "bool")

    def test_integer_indices_survive_bfloat16_floatx(self):
        """Downstream ops that require integer inputs keep working."""
        backend.set_floatx("bfloat16")
        table = np.arange(12, dtype="float32").reshape(4, 3)
        indices = np.array([1, 2], dtype="int32")

        self.assertDType(ops.convert_to_tensor(indices), "int32")
        self.assertAllClose(
            ops.take(table, indices, axis=0), table[[1, 2]], atol=1e-2
        )


class CoreOpsBehaviorTests(testing.TestCase):
    def test_associative_scan_invalid_arguments(self):
        # varying dimension at scan axis
        x = (np.array([1, 2]), np.array([3, 4]), np.array([5, 6, 7]))
        with self.assertRaisesRegex(ValueError, " first dimension"):
            core.associative_scan(lambda x, y: (x[0] + y[0], x[1] + y[1]), x)

        # same error, symbolic
        x = (
            KerasTensor((None, 5)),
            KerasTensor((None, 4)),
        )
        with self.assertRaisesRegex(ValueError, " first dimension"):
            core.associative_scan(
                lambda x, y: (x[0] + y[0], x[1] + y[1]), x, axis=1
            )

    def test_cond_check_output_spec(self):
        mock_spec = Mock(dtype="float32", shape=(2, 2))
        mock_spec_different = Mock(dtype="int32", shape=(3, 3))

        # List & tuple.
        self.assertTrue(
            core.Cond()._check_output_spec(
                [mock_spec, mock_spec], [mock_spec, mock_spec]
            )
        )
        self.assertTrue(
            core.Cond()._check_output_spec([mock_spec], [mock_spec])
        )
        self.assertFalse(
            core.Cond()._check_output_spec(
                [mock_spec], [mock_spec, mock_spec_different]
            )
        )
        self.assertTrue(
            core.Cond()._check_output_spec((mock_spec,), (mock_spec,))
        )
        self.assertFalse(
            core.Cond()._check_output_spec(
                (mock_spec,), (mock_spec, mock_spec_different)
            )
        )

        # Dict.
        self.assertTrue(
            core.Cond()._check_output_spec({"a": mock_spec}, {"a": mock_spec})
        )
        self.assertFalse(
            core.Cond()._check_output_spec({"a": mock_spec}, {"b": mock_spec})
        )
        self.assertFalse(
            core.Cond()._check_output_spec(
                {"a": mock_spec}, {"a": mock_spec, "b": mock_spec}
            )
        )

        # None.
        self.assertTrue(core.Cond()._check_output_spec(None, None))
        self.assertFalse(
            core.Cond()._check_output_spec(
                None, Mock(dtype="float32", shape=(2, 2))
            )
        )
        self.assertFalse(
            core.Cond()._check_output_spec(
                Mock(dtype="float32", shape=(2, 2)), None
            )
        )

        # KerasTensor.
        mock_spec1 = KerasTensor(shape=(2, 2), dtype="float32")
        mock_spec2 = KerasTensor(shape=(2, 2), dtype="float32")
        self.assertTrue(core.Cond()._check_output_spec(mock_spec1, mock_spec2))

    def test_cond_error_message_uses_real_call_signature(self):
        """Regression test: exceptions raised inside `Cond.__call__` should
        be augmented using `Cond.call`'s real parameter names (`pred`,
        `true_fn`, `false_fn`), not the generic `args`/`kwargs` of the
        internal `call_fn` closure used to dispatch between eager and
        symbolic execution.
        """
        was_enabled = traceback_utils.is_traceback_filtering_enabled()
        traceback_utils.enable_traceback_filtering()
        try:

            def raising_true_fn():
                raise ValueError("boom")

            with self.assertRaises(ValueError) as ctx:
                core.cond(True, raising_true_fn, lambda: 0)
            msg = str(ctx.exception)
            self.assertIn("Cond.call()", msg)
            self.assertIn("pred=", msg)
            self.assertIn("true_fn=", msg)
            self.assertIn("false_fn=", msg)
            self.assertNotIn("args=", msg)
            self.assertNotIn("kwargs=", msg)
        finally:
            if not was_enabled:
                traceback_utils.disable_traceback_filtering()

    @parameterized.named_parameters(
        ("full", "full", {}),
        ("larger_than", "larger_than", {"output_size_threshold": 1}),
    )
    def test_cond_is_not_rematerialized(self, mode, kwargs):
        """`Cond.call` takes Python callables, which rematerialization cannot
        trace, so `Cond` overrides `__call__` to stay out of that path. An
        active `RematScope` must therefore leave `cond` working.
        """
        pred = ops.convert_to_tensor(True)
        x = ops.convert_to_tensor([1.0, 2.0, 3.0])
        with RematScope(mode=mode, **kwargs):
            result = ops.cond(pred, lambda: x + 1.0, lambda: x - 1.0)
        self.assertAllClose(result, [2.0, 3.0, 4.0])

    @pytest.mark.requires_trainable_backend
    def test_cond_raw_bool_compile(self):
        class ExampleLayer(layers.Layer):
            def call(self, x, training=False):
                return ops.cond(training, lambda: x, lambda: x * 2.0)

        model = models.Sequential([ExampleLayer()])
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.ones((2, 4), dtype="float32")
        y = np.zeros((2, 4), dtype="float32")
        model.evaluate(x, y, batch_size=2)

    def test_convert_to_numpy(self):
        x = ops.array([1, 2, 3], dtype="float32")
        y = ops.convert_to_numpy(x)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.dtype, "float32")
        self.assertEqual(y.shape, (3,))
        # Test assignment -- should not fail.
        y[0] = 1.0

        x = ops.array(4, dtype="float32")
        y = ops.convert_to_numpy(x)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.dtype, "float32")
        self.assertEqual(y.shape, ())
        # Test assignment -- should not fail.
        y = 1.0

        with self.assertRaises(ValueError):
            ops.convert_to_numpy(KerasTensor((2,)))

    def test_convert_to_tensor_python_float_with_dtype(self):
        # Regression: a Python float passed with an explicit dtype must
        # not go through a float64 numpy intermediate. On the torch
        # backend, torch.export would otherwise lift the float64
        # constant into the graph before the cast runs.
        t = ops.convert_to_tensor(1.5, dtype="float32")
        self.assertEqual(standardize_dtype(t.dtype), "float32")
        t = ops.convert_to_tensor(1.5, dtype="float16")
        self.assertEqual(standardize_dtype(t.dtype), "float16")

    def test_convert_to_tensor_python_int_with_dtype(self):
        t = ops.convert_to_tensor(7, dtype="int16")
        self.assertEqual(standardize_dtype(t.dtype), "int16")
        t = ops.convert_to_tensor(7, dtype="float32")
        self.assertEqual(standardize_dtype(t.dtype), "float32")

    def test_convert_to_tensor_python_bool_with_dtype(self):
        t = ops.convert_to_tensor(True, dtype="float32")
        self.assertEqual(standardize_dtype(t.dtype), "float32")
        t = ops.convert_to_tensor(True)
        self.assertEqual(standardize_dtype(t.dtype), "bool")

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="torch.export graph inspection is torch-only.",
    )
    def test_convert_to_tensor_python_scalar_no_float64_in_export(self):
        import torch

        class M(torch.nn.Module):
            def forward(self, x):
                scalar = ops.convert_to_tensor(1e-7, dtype="float32")
                return torch.maximum(x, scalar)

        # torch.export requires all tensors on the same device. Force CPU
        # so the test is independent of the runner's default device.
        with backend.device("cpu"):
            exported = torch.export.export(
                M(), (torch.ones(5, dtype=torch.float32),)
            )
        for val in exported.constants.values():
            if isinstance(val, torch.Tensor):
                self.assertNotEqual(val.dtype, torch.float64)

    def test_scan_invalid_arguments(self):
        def cumsum(carry, xs):
            carry = carry + xs
            return carry, carry

        init = np.array(0, dtype="float32")
        xs = np.array([1, 2, 3, 4, 10, 20], dtype="float32")

        # Test non-callable
        with self.assertRaisesRegex(TypeError, "should be a callable."):
            core.scan(123, init, xs)

        # Test bad unroll
        with self.assertRaisesRegex(
            ValueError, "must be an positive integer or boolean."
        ):
            core.scan(cumsum, init, xs, unroll=-1)

        # Test both xs and length are None
        with self.assertRaisesRegex(ValueError, "to scan over and"):
            core.scan(cumsum, init, xs=None, length=None)

    def test_slice_compute_output_spec(self):
        inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        start_indices = np.array([1, 1])
        shape = (2, 2)
        output_spec = core.Slice(shape).compute_output_spec(
            inputs, start_indices
        )
        self.assertEqual(output_spec.shape, shape)
        self.assertEqual(output_spec.dtype, inputs.dtype)

    def test_stop_gradient_compute_output_spec(self):
        variable = KerasTensor(shape=(3,), dtype="float32")
        stop_gradient = core.StopGradient()
        output_spec = stop_gradient.compute_output_spec(variable)
        self.assertEqual(output_spec.shape, variable.shape)
        self.assertEqual(output_spec.dtype, variable.dtype)

    def test_vectorized_map_serialization(self):
        @object_registration.register_keras_serializable()
        def f(x):
            return x + x

        inputs = input_layer.Input((10,), dtype="float32")
        outputs = core.vectorized_map(f, inputs)
        model = models.Functional(inputs, outputs)
        reloaded_model = model.from_config(model.get_config())
        x = np.random.rand(5, 10).astype("float32")
        self.assertAllClose(model(x), reloaded_model(x))

    def test_while_loop_output_spec(self):
        # Define dummy cond and body functions
        def cond(x):
            return True

        def body(x):
            return (x,)

        while_loop = core.WhileLoop(cond, body, maximum_iterations=None)
        loop_vars = (KerasTensor(shape=(10,), dtype="float32"),)
        output_spec = while_loop.compute_output_spec(loop_vars)
        self.assertEqual(output_spec[0].shape, loop_vars[0].shape)
        self.assertEqual(output_spec[0].dtype, loop_vars[0].dtype)

        # Test with KerasTensor.
        loop_vars = (np.random.rand(5, 5), np.random.randint(10, size=(3, 7)))
        keras_loop_vars = [
            KerasTensor(v.shape, dtype=v.dtype) for v in loop_vars
        ]
        while_loop = core.WhileLoop(cond, body, maximum_iterations=None)
        output_specs = while_loop.compute_output_spec(keras_loop_vars)
        self.assertEqual(output_specs[0].shape, keras_loop_vars[0].shape)
        self.assertEqual(output_specs[0].dtype, keras_loop_vars[0].dtype)
        self.assertEqual(output_specs[1].shape, keras_loop_vars[1].shape)
        self.assertEqual(output_specs[1].dtype, keras_loop_vars[1].dtype)

    def test_unstack_unknown_axis_num(self):
        x = KerasTensor((2, None, None))
        axis = 1
        with self.assertRaisesRegex(
            ValueError, r"Cannot infer argument `num` from shape"
        ):
            core.unstack(x, axis=axis)

    def test_unstack_axis_out_of_range(self):
        x = KerasTensor((3, 4))
        with self.assertRaisesRegex(ValueError, r"axis 10 is out of bounds"):
            core.unstack(x, axis=10)


class WhileLoopCaptureTest(testing.TestCase):
    def test_while_loop_body_closes_over_outer_tensor(self):
        # The body reads `captured` from the enclosing scope instead of
        # receiving it as a loop variable. Backends that lower the loop into a
        # separate subgraph still have to resolve it; it is constant across
        # iterations. Run through `predict` so the batch dim is dynamic.
        class LoopWithCapture(layers.Layer):
            def call(self, x):
                captured = x * 2.0

                def cond(i, acc):
                    return i < 3

                def body(i, acc):
                    return i + 1, acc + captured

                _, acc = core.while_loop(cond, body, (0, ops.zeros_like(x)))
                return acc

        x = np.ones((2, 4), dtype="float32")
        # three iterations, each adding 2 * ones
        expected = np.full((2, 4), 6.0, dtype="float32")

        self.assertAllClose(LoopWithCapture()(x), expected)

        inputs = input_layer.Input(shape=(4,))
        model = models.Functional(inputs, LoopWithCapture()(inputs))

        self.assertAllClose(model.predict(x), expected)


@pytest.mark.skipif(
    not backend.SUPPORTS_GRADIENT,
    reason="Backend does not support gradients.",
)
class CoreOpsGradTest(testing.TestCase):
    def test_grad_single_argument(self):
        def f(x):
            return x**2

        x = ops.array([1.0, 2.0, 3.0])
        self.assertAllClose(ops.grad(f)(x), [2.0, 4.0, 6.0])

    def test_grad_argnums(self):
        def f(x, y):
            return x * y

        x = ops.array([1.0, 2.0])
        y = ops.array([3.0, 4.0])
        self.assertAllClose(ops.grad(f, argnums=1)(x, y), [1.0, 2.0])
        dx, dy = ops.grad(f, argnums=(0, 1))(x, y)
        self.assertAllClose(dx, [3.0, 4.0])
        self.assertAllClose(dy, [1.0, 2.0])
        (dy,) = ops.grad(f, argnums=(1,))(x, y)
        self.assertAllClose(dy, [1.0, 2.0])
        self.assertAllClose(ops.grad(f, argnums=-1)(x, y), [1.0, 2.0])
        dx, dy = ops.grad(f, argnums=(-2, -1))(x, y)
        self.assertAllClose(dx, [3.0, 4.0])
        self.assertAllClose(dy, [1.0, 2.0])

    def test_grad_keyword_arguments_pass_through(self):
        def f(x, scale=1.0):
            return x * scale

        x = ops.array([1.0, 2.0])
        self.assertAllClose(ops.grad(f)(x, scale=3.0), [3.0, 3.0])

    def test_grad_nested_structure(self):
        def f(params):
            return params["a"] ** 2 + params["b"]

        params = {"a": ops.array([1.0, 2.0]), "b": ops.array([3.0])}
        grads = ops.grad(f)(params)
        self.assertEqual(set(grads.keys()), {"a", "b"})
        self.assertAllClose(grads["a"], [2.0, 4.0])
        self.assertAllClose(grads["b"], [2.0])

    def test_grad_variable_argument(self):
        def f(x):
            return x**2

        v = backend.Variable([1.0, 2.0])
        self.assertAllClose(ops.grad(f)(v), [2.0, 4.0])

    def test_grad_unused_argument_is_zeros(self):
        def f(x, y):
            return y

        x = ops.array([1.0, 2.0, 3.0])
        y = ops.array([1.0])
        dx, dy = ops.grad(f, argnums=(0, 1))(x, y)
        self.assertAllClose(dx, [0.0, 0.0, 0.0])
        self.assertAllClose(dy, [1.0])

    def test_grad_through_layer(self):
        layer = layers.Dense(2, kernel_initializer="ones", use_bias=False)
        layer.build((None, 3))

        def f(x):
            return layer(x)

        x = ops.ones((1, 3))
        self.assertAllClose(ops.grad(f)(x), [[2.0, 2.0, 2.0]])

    def test_grad_stateless_call(self):
        layer = layers.Dense(1, kernel_initializer="ones", use_bias=False)
        layer.build((None, 2))

        def f(trainable_variables, x):
            return layer.stateless_call(trainable_variables, [], x)[0]

        x = ops.array([[1.0, 2.0]])
        (dkernel,) = ops.grad(f)(
            [v.value for v in layer.trainable_variables], x
        )
        self.assertAllClose(dkernel, [[1.0], [2.0]])

    def test_grad_non_scalar_output_is_summed(self):
        x = ops.array([0.0, 1.0])
        expected = 1.0 - np.tanh(ops.convert_to_numpy(x)) ** 2
        self.assertAllClose(ops.grad(ops.tanh)(x), expected)

    def test_grad_invalid_argnums(self):
        def f(x, y):
            return ops.sum(x * y)

        x = ops.array([1.0])
        with self.assertRaisesRegex(ValueError, "positional argument 2"):
            ops.grad(f, argnums=2)(x, x)
        with self.assertRaisesRegex(ValueError, "positional argument -3"):
            ops.grad(f, argnums=-3)(x, x)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            ops.grad(f, argnums=(0, 0))(x, x)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            ops.grad(f, argnums=(1, -1))(x, x)
        with self.assertRaisesRegex(TypeError, "int or a tuple of ints"):
            ops.grad(f, argnums="0")(x, x)


@pytest.mark.skipif(
    backend.SUPPORTS_GRADIENT, reason="Backend supports gradients."
)
class CoreOpsGradUnsupportedTest(testing.TestCase):
    def test_grad_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "not supported"):
            ops.grad(lambda x: x)(ops.array([1.0]))
