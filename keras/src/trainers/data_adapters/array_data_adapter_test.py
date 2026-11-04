import jax
import jax.experimental.sparse as jax_sparse
import numpy as np
import pytest
import scipy
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src.backend.common import standardize_dtype
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters import array_data_adapter
from keras.src.trainers.data_adapters import data_adapter_test_base


class TestArrayDataAdapter(data_adapter_test_base.DataAdapterTest):
    def make_array(self, array_type, shape, dtype):
        x = np.array([[i] * shape[1] for i in range(shape[0])], dtype=dtype)
        if array_type == "np":
            return x
        elif array_type == "tf":
            return tf.constant(x)
        elif array_type == "tf_ragged":
            return tf.RaggedTensor.from_tensor(x)
        elif array_type == "tf_sparse":
            return tf.sparse.from_dense(x)
        elif array_type == "jax":
            return jax.numpy.array(x)
        elif array_type == "jax_sparse":
            return jax_sparse.BCOO.fromdense(x)
        elif array_type == "torch":
            return torch.as_tensor(x)
        elif array_type == "pandas_data_frame":
            import pandas  # Import pandas after tf, fixes deadlock on MacOS

            return pandas.DataFrame(x)
        elif array_type == "pandas_series":
            import pandas  # Import pandas after tf, fixes deadlock on MacOS

            return pandas.Series(x[:, 0])
        elif array_type == "scipy_sparse":
            return scipy.sparse.coo_matrix(x)

    @parameterized.named_parameters(
        named_product(
            array_type=[
                "np",
                "tf",
                "tf_ragged",
                "tf_sparse",
                "jax",
                "jax_sparse",
                "torch",
                "pandas_data_frame",
                "pandas_series",
                "scipy_sparse",
            ],
            array_dtype=["float32", "float64"],
            shuffle=[False, "batch", True],
        )
    )
    def test_basic_flow(self, array_type, array_dtype, shuffle):
        x = self.make_array(array_type, (34, 4), array_dtype)
        y = self.make_array(array_type, (34, 2), "int32")
        xdim1 = 1 if array_type == "pandas_series" else 4
        ydim1 = 1 if array_type == "pandas_series" else 2

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            sample_weight=None,
            batch_size=16,
            steps=None,
            shuffle=shuffle,
        )
        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            if array_type == "tf_ragged":
                expected_class = tf.RaggedTensor
                xdim1 = None
                ydim1 = None
            elif array_type in ("tf_sparse", "jax_sparse", "scipy_sparse"):
                expected_class = tf.SparseTensor
            else:
                expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            if array_type in ("tf_sparse", "jax_sparse", "scipy_sparse"):
                expected_class = jax_sparse.JAXSparse
            else:
                expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        x_order = []
        y_order = []
        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(standardize_dtype(bx.dtype), backend.floatx())
            self.assertEqual(standardize_dtype(by.dtype), "int32")
            if i < 2:
                self.assertEqual(bx.shape, (16, xdim1))
                self.assertEqual(by.shape, (16, ydim1))
            else:
                self.assertEqual(bx.shape, (2, xdim1))
                self.assertEqual(by.shape, (2, ydim1))

            if isinstance(bx, tf.SparseTensor):
                bx = tf.sparse.to_dense(bx)
                by = tf.sparse.to_dense(by)
            if isinstance(bx, jax_sparse.JAXSparse):
                bx = bx.todense()
                by = by.todense()
            x_batch_order = [float(bx[j, 0]) for j in range(bx.shape[0])]
            y_batch_order = [float(by[j, 0]) for j in range(by.shape[0])]
            x_order.extend(x_batch_order)
            y_order.extend(y_batch_order)

            if shuffle == "batch":
                self.assertAllClose(
                    sorted(x_batch_order),
                    list(range(i * 16, i * 16 + bx.shape[0])),
                )

        self.assertAllClose(x_order, y_order)
        if shuffle:
            self.assertNotAllClose(x_order, list(range(34)))
        else:
            self.assertAllClose(x_order, list(range(34)))

    @pytest.mark.skipif(backend.backend() != "tensorflow", reason="tf only")
    def test_tf_sparse_trailing_empty_rows(self):
        x = tf.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=[1.0, 2.0],
            dense_shape=(4, 3),
        )

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            batch_size=4,
            shuffle=False,
        )
        batch = next(iter(adapter.get_tf_dataset()))

        self.assertIsInstance(batch, tf.SparseTensor)
        self.assertEqual(batch.shape, (4, 3))
        self.assertAllClose(
            tf.sparse.to_dense(batch),
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

    def test_multi_inputs_and_outputs(self):
        x1 = np.random.random((34, 1))
        x2 = np.random.random((34, 2))
        y1 = np.random.random((34, 3))
        y2 = np.random.random((34, 4))
        sw = np.random.random((34,))
        adapter = array_data_adapter.ArrayDataAdapter(
            x={"x1": x1, "x2": x2},
            y=[y1, y2],
            sample_weight=sw,
            batch_size=16,
            steps=None,
            shuffle=False,
        )
        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertIsInstance(bx, dict)
            self.assertIsInstance(by, list)
            self.assertIsInstance(bw, list)

            self.assertIsInstance(bx["x1"], np.ndarray)
            self.assertIsInstance(bx["x2"], np.ndarray)
            self.assertIsInstance(by[0], np.ndarray)
            self.assertIsInstance(by[1], np.ndarray)
            self.assertIsInstance(bw[0], np.ndarray)
            self.assertIsInstance(bw[1], np.ndarray)

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(bx["x1"].shape, (16, 1))
                self.assertEqual(bx["x2"].shape, (16, 2))
                self.assertEqual(by[0].shape, (16, 3))
                self.assertEqual(by[1].shape, (16, 4))
                self.assertEqual(bw[0].shape, (16,))
                self.assertEqual(bw[1].shape, (16,))
            else:
                self.assertEqual(bx["x1"].shape, (2, 1))
                self.assertEqual(by[0].shape, (2, 3))
                self.assertEqual(bw[0].shape, (2,))
                self.assertEqual(bw[1].shape, (2,))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertIsInstance(bx, dict)
            # NOTE: the y list was converted to a tuple for tf.data
            # compatibility.
            self.assertIsInstance(by, tuple)
            self.assertIsInstance(bw, tuple)

            self.assertIsInstance(bx["x1"], tf.Tensor)
            self.assertIsInstance(bx["x2"], tf.Tensor)
            self.assertIsInstance(by[0], tf.Tensor)
            self.assertIsInstance(by[1], tf.Tensor)
            self.assertIsInstance(bw[0], tf.Tensor)
            self.assertIsInstance(bw[1], tf.Tensor)

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(tuple(bx["x1"].shape), (16, 1))
                self.assertEqual(tuple(bx["x2"].shape), (16, 2))
                self.assertEqual(tuple(by[0].shape), (16, 3))
                self.assertEqual(tuple(by[1].shape), (16, 4))
                self.assertEqual(tuple(bw[0].shape), (16,))
                self.assertEqual(tuple(bw[1].shape), (16,))
            else:
                self.assertEqual(tuple(bx["x1"].shape), (2, 1))
                self.assertEqual(tuple(by[0].shape), (2, 3))
                self.assertEqual(tuple(bw[0].shape), (2,))
                self.assertEqual(tuple(bw[1].shape), (2,))

    @parameterized.named_parameters(
        named_product(target_encoding=["int", "categorical"])
    )
    def test_class_weights(self, target_encoding):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int32")
        else:
            y = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype="float32",
            )

        class_weight = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
        }
        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            class_weight=class_weight,
            batch_size=16,
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_native_tensor_flow(self):
        x = backend.ops.convert_to_tensor(np.random.random((34, 2)))
        y = backend.ops.convert_to_tensor(np.random.random((34, 1)))
        adapter = array_data_adapter.ArrayDataAdapter(x=x, y=y, batch_size=16)
        it = adapter.get_native_iterator()
        batches = list(it)
        self.assertEqual(len(batches), 3)
        self.assertEqual(tuple(batches[0][0].shape), (16, 2))
        self.assertEqual(tuple(batches[0][1].shape), (16, 1))
        self.assertEqual(tuple(batches[-1][0].shape), (2, 2))
        self.assertEqual(tuple(batches[-1][1].shape), (2, 1))

    def test_errors(self):
        x = np.random.random((34, 1))
        y = np.random.random((34, 3))
        sw = np.random.random((34,))
        cw = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
        }

        with self.assertRaisesRegex(
            ValueError, "Expected all elements of `x` to be array-like"
        ):
            array_data_adapter.ArrayDataAdapter(x="Invalid")
        with self.assertRaisesRegex(
            ValueError, "Expected all elements of `x` to be array-like"
        ):
            array_data_adapter.ArrayDataAdapter(x=x, y="Invalid")
        with self.assertRaisesRegex(
            ValueError, "Expected all elements of `x` to be array-like"
        ):
            array_data_adapter.ArrayDataAdapter(
                x=x, y=y, sample_weight="Invalid"
            )

        with self.assertRaisesRegex(
            ValueError, "You cannot `class_weight` and `sample_weight`"
        ):
            array_data_adapter.ArrayDataAdapter(
                x=x, y=y, sample_weight=sw, class_weight=cw
            )

        nested_y = ({"x": x, "y": y},)
        with self.assertRaisesRegex(
            ValueError, "You should provide one `sample_weight` array per"
        ):
            array_data_adapter.ArrayDataAdapter(
                x=x, y=nested_y, sample_weight=[]
            )

        tensor_sw = self.make_array("tf", (34, 2), "int32")
        with self.assertRaisesRegex(
            ValueError, "For a model with multiple outputs, when providing"
        ):
            array_data_adapter.ArrayDataAdapter(
                x=x, y=nested_y, sample_weight=tensor_sw
            )

        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single",
        ):
            array_data_adapter.ArrayDataAdapter(
                x=x, y=nested_y, class_weight=cw
            )

    def test_torch_data_loader_nested_input(self):
        # This tests ArrayDataset.__len__ with nested input
        x = {"a": np.ones((10, 1)), "b": np.ones((10, 2))}
        y = np.ones((10, 3))
        adapter = array_data_adapter.ArrayDataAdapter(
            x=x, y=y, batch_size=5, shuffle=False
        )
        dl = adapter.get_torch_dataloader()
        self.assertEqual(len(dl.dataset), 10)
        batch = next(iter(dl))
        self.assertEqual(batch[0]["a"].shape, (5, 1))
        self.assertEqual(batch[0]["b"].shape, (5, 2))
        self.assertEqual(batch[1].shape, (5, 3))

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "dataparallel",
                    "dist_type": "dp",
                    "num_processes": 4,
                    "process_id": 1,
                    "mesh_shape": (4,),
                },
                {
                    "testcase_name": "modelparallel",
                    "dist_type": "mp",
                    "num_processes": 8,
                    "process_id": 5,
                    "mesh_shape": (2, 4),
                },
                {
                    "testcase_name": "modelparallel_large_mesh",
                    "dist_type": "mp",
                    "num_processes": 4,
                    "process_id": 2,
                    "mesh_shape": (8, 2),
                },
            ],
            shuffle=[False, True],
        )
    )
    def test_sharding(
        self,
        dist_type,
        num_processes,
        process_id,
        mesh_shape,
        shuffle,
    ):
        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * num_processes)
        else:
            device_mesh = dist_lib.DeviceMesh(
                shape=mesh_shape,
                axis_names=("data", "model"),
                devices=["cpu:0"] * np.prod(mesh_shape),
            )
            dist = dist_lib.ModelParallel(
                device_mesh=device_mesh,
                layout_map=dist_lib.LayoutMap(device_mesh),
                batch_dim_name="data",
            )
        dist._num_processes = num_processes
        dist._process_id = process_id
        dist.auto_shard_dataset = True

        expected_num_replicas = min(dist.num_model_replicas, num_processes)
        expected_shard_id = dist.data_shard_id

        x = self.make_array("np", (100, 10), "float32")
        y = self.make_array("np", (100, 1), "float32")
        with dist.scope():
            adapter = array_data_adapter.ArrayDataAdapter(
                x, y, batch_size=10, shuffle=shuffle
            )

            it_methods = ["get_numpy_iterator"]
            backend_it_method = {
                "tensorflow": "get_tf_dataset",
                "jax": "get_jax_iterator",
                "torch": "get_torch_dataloader",
            }.get(backend.backend())
            if backend_it_method:
                it_methods.append(backend_it_method)

        self.assertEqual(adapter._num_data_shards, expected_num_replicas)
        self.assertEqual(adapter._data_shard_id, expected_shard_id)

        def get_order(it_fn):
            order = []
            for batch in it_fn():
                bx = batch[0]
                bx = backend.ops.convert_to_numpy(bx)
                order.extend(bx[:, 0].tolist())
            return order

        for it_method in it_methods:
            it_fn = getattr(adapter, it_method)
            if not shuffle:
                batches = list(it_fn())
                expected_num_batches = (
                    10 - expected_shard_id + expected_num_replicas - 1
                ) // expected_num_replicas
                self.assertEqual(len(batches), expected_num_batches)

                for i, batch in enumerate(batches):
                    bx, by = batch
                    bx = backend.ops.convert_to_numpy(bx)
                    by = backend.ops.convert_to_numpy(by)
                    expected_batch_index = (
                        expected_shard_id + i * expected_num_replicas
                    )
                    expected_first_sample_value = expected_batch_index * 10
                    self.assertAllClose(
                        bx[:, 0],
                        np.arange(
                            expected_first_sample_value,
                            expected_first_sample_value + 10,
                        ),
                    )
                    self.assertAllClose(
                        by[:, 0],
                        np.arange(
                            expected_first_sample_value,
                            expected_first_sample_value + 10,
                        ),
                    )
            else:
                # Same epoch should have same shuffle
                adapter._epoch = 1
                order1 = get_order(it_fn)
                adapter._epoch = 1
                order2 = get_order(it_fn)
                self.assertAllClose(order1, order2)

                # Different epochs should have different shuffle
                adapter._epoch = 2
                order3 = get_order(it_fn)
                self.assertNotAllClose(order1, order3)

    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
    def test_get_jax_iterator_with_super_batch(self):
        # Even batches: 4 batches with super_batch=2 -> 2 super-batches
        x = np.ones((64, 4), dtype="float32")
        y = np.ones((64, 2), dtype="float32")
        adapter = array_data_adapter.ArrayDataAdapter(x, y, batch_size=16)
        self.verify_super_batched_iterator(
            adapter.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=False,
        )

        # Uneven batches: 5 batches with super_batch=2 -> 2 super-batches + 1
        # partial batch list
        x_uneven = np.ones((80, 4), dtype="float32")
        y_uneven = np.ones((80, 2), dtype="float32")
        adapter_uneven = array_data_adapter.ArrayDataAdapter(
            x_uneven, y_uneven, batch_size=16
        )
        self.verify_super_batched_iterator(
            adapter_uneven.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=True,
        )
