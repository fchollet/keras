from unittest import mock

import jax
import numpy as np
import pytest
import tensorflow as tf
import torch

from keras.src import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import data_adapter_test_base
from keras.src.trainers.data_adapters import tf_dataset_adapter


class TestTFDatasetAdapter(data_adapter_test_base.DataAdapterTest):
    def test_basic_flow(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

    def _test_class_weights(self, target_encoding="int"):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int64")
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
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_class_weights_int_targets(self):
        self._test_class_weights(target_encoding="int")

    def test_class_weights_categorical_targets(self):
        self._test_class_weights(target_encoding="categorical")

    def test_class_weights_missing_classes_default_to_one(self):
        x = np.random.random((4, 2))
        y = np.array([[0], [1], [2], [3]], dtype="int64")

        class_weight = {
            0: 0.1,
            2: 0.3,
        }
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 1.0, 0.3, 1.0])

    def test_empty_class_weights_ignored(self):
        x = np.random.random((4, 2))
        y = np.array([[0], [1], [2], [3]], dtype="int64")

        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds, class_weight={})
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 2)

    def test_num_batches(self):
        dataset = tf.data.Dataset.range(42)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, 42)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, 42)

        # Test for Infinite Cardinality
        dataset = tf.data.Dataset.range(42)
        dataset = dataset.repeat()
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.INFINITE_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

        # Test for Unknown Cardinality
        dataset = dataset.filter(lambda x: True)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.UNKNOWN_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

    def test_invalid_dataset_type(self):
        with self.assertRaisesRegex(
            ValueError, "Expected argument `dataset` to be a tf.data.Dataset"
        ):
            invalid_data = "This is not a tf.data.Dataset"
            tf_dataset_adapter.TFDatasetAdapter(invalid_data)

    def test_class_weight_and_sample_weight_together(self):
        x = np.random.random((4, 2))
        y = np.array([[0], [1], [2], [3]], dtype="int64")
        sw = np.array([0.5, 0.5, 0.5, 0.5])
        base_ds = tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

        with self.assertRaisesRegex(
            ValueError,
            "You cannot `class_weight` and `sample_weight` at the same time.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_different_y_shapes_with_class_weight(self):
        x = np.random.random((4, 2))
        y = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype="float32",
        )
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

        y_sparse = np.array([0, 1, 2, 3], dtype="int64")
        base_ds = tf.data.Dataset.from_tensor_slices((x, y_sparse)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_nested_y_with_class_weight(self):
        x = np.random.random((4, 2))

        # Define two target outputs, y1 and y2, for the dataset
        y1 = np.array([0, 1, 2, 3], dtype="int64")
        y2 = np.array([0, 1, 2, 3], dtype="int64")

        # Create a tf.data Dataset from the input data and two target outputs
        base_ds = tf.data.Dataset.from_tensor_slices((x, (y1, y2))).batch(16)

        # Define class weights for potential classes in the output
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single output.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_class_weights_map_fn_with_sample_weight(self):
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        class_weights_map_fn = tf_dataset_adapter.make_class_weight_map_fn(
            class_weight
        )

        x = np.array([[0.5, 0.5], [0.5, 0.5]])
        y = np.array([[1, 0], [0, 1]])
        sw = np.array([1.0, 1.0])

        with self.assertRaisesRegex(
            ValueError,
            "You cannot `class_weight` and `sample_weight` at the same time.",
        ):
            class_weights_map_fn(x, y, sw)

    def test_class_weights_map_fn_nested_y(self):
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        class_weights_map_fn = tf_dataset_adapter.make_class_weight_map_fn(
            class_weight
        )

        x = np.array([[0.5, 0.5]])
        y1 = np.array([1])
        y2 = np.array([0])

        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single output.",
        ):
            class_weights_map_fn(x, (y1, y2))

    def test_distribute_dataset(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)

        data_distribution = mock.Mock()
        data_distribution.num_processes = 2
        data_distribution.num_model_replicas = 2
        data_distribution.num_data_shards = 2
        data_distribution.data_shard_id = 0
        # Mimic that there are 2 worker, and each of the worker will get batch
        # size of 8
        data_distribution.distribute_dataset = mock.MagicMock(
            return_value=base_ds.rebatch(8).shard(2, index=0)
        )

        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, distribution=data_distribution
        )

        self.assertEqual(adapter.num_batches, None)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(bx.shape, (8, 4))
                self.assertEqual(by.shape, (8, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(tuple(bx.shape), (8, 4))
                self.assertEqual(tuple(by.shape), (8, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS and backend.backend() != "numpy",
        reason="Backend does not support sparse tensors",
    )
    def test_tf_sparse_tensors(self):
        x = tf.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=(2, 4)
        )
        y = tf.SparseTensor(
            indices=[[0, 0], [1, 1]], values=[3.0, 4.0], dense_shape=(2, 2)
        )
        base_ds = tf.data.Dataset.from_tensors((x, y))
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.SparseTensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.experimental.sparse.BCOO

        for batch in it:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.shape, (2, 4))
            self.assertEqual(by.shape, (2, 2))

    def test_distributed_datasets_from_function_adapter_properties(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0"])

        def dataset_fn(input_context):
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size=2
            )
            x = tf.random.uniform((32, 4))
            y = tf.random.uniform((32, 2))
            return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

        dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dist_dataset)
        self.assertEqual(adapter.num_batches, 16)
        self.assertIsNone(adapter.batch_size)
        self.assertIsNone(adapter.has_partial_batch)
        self.assertIsNone(adapter.partial_batch_size)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        batch_count = 0
        for batch in it:
            batch_count += 1
            self.assertEqual(len(batch), 2)
            data, labels = batch
            self.assertIsInstance(data, expected_class)
            self.assertIsInstance(labels, expected_class)
            self.assertEqual(data.shape, (2, 4))
            self.assertEqual(labels.shape, (2, 2))

        self.assertEqual(batch_count, 16)

    @pytest.mark.requires_trainable_backend
    def test_distributed_datasets_from_function_model_integration(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0"])

        def dataset_fn(input_context):
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size=2
            )
            x = tf.random.uniform((4, 1))
            y = tf.random.uniform((4, 2))
            return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

        dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

        model = Sequential([layers.Dense(2, input_shape=(1,))])
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(dist_dataset, epochs=1)
        self.assertIn("loss", history.history)

    def test_distribute_tf_dataset_multiple_replicas_per_process(self):
        dataset = tf.data.Dataset.range(16).batch(8)

        distribution = distribution_lib.DataParallel(
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        distribution._num_processes = 2
        distribution._process_id = 0
        distribution.auto_shard_dataset = True

        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()
        # Global batch size 8 is divisible by num_shards (min(4, 2)) = 2
        # per_process_batch_size = 8 // 2 = 4
        # index = data_shard_id = 0
        data = list(distributed_dataset.as_numpy_iterator())
        self.assertEqual(len(data), 2)
        self.assertAllClose(data[0], [0, 1, 2, 3])
        self.assertAllClose(data[1], [8, 9, 10, 11])

        distribution._process_id = 1
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()
        # index = data_shard_id = 1
        data = list(distributed_dataset.as_numpy_iterator())
        self.assertEqual(len(data), 2)
        self.assertAllClose(data[0], [4, 5, 6, 7])
        self.assertAllClose(data[1], [12, 13, 14, 15])

    def test_distribute_tf_dataset_error_not_divisible(self):
        distribution = distribution_lib.DataParallel(
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        distribution._num_processes = 2
        distribution._process_id = 0
        distribution.auto_shard_dataset = True

        dataset = tf.data.Dataset.range(16).batch(7)
        # Global batch size 7 is NOT divisible by num_shards (min(4, 2)) = 2
        with self.assertRaisesRegex(
            ValueError, "Global batch size must be divisible by the number"
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                dataset, distribution=distribution
            )

    def test_unknown_batch_size(self):
        distribution = distribution_lib.DataParallel(devices=["cpu:0"])
        distribution._num_processes = 2
        distribution._process_id = 0
        distribution.auto_shard_dataset = True

        dataset = tf.data.Dataset.range(16)
        with self.assertRaisesRegex(ValueError, "batch size .* is unknown"):
            tf_dataset_adapter.TFDatasetAdapter(
                dataset, distribution=distribution
            )

    def test_num_model_replicas_one(self):
        distribution = distribution_lib.DataParallel(devices=["cpu:0"])
        distribution._num_processes = 2
        distribution._process_id = 0
        distribution.auto_shard_dataset = True

        dataset = tf.data.Dataset.range(16).batch(8)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()
        # Should return the same data (with prefetch)
        data = list(distributed_dataset.as_numpy_iterator())
        self.assertEqual(len(data), 2)
        self.assertAllClose(data[0], np.arange(8))
        self.assertAllClose(data[1], np.arange(8, 16))

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="JAX only",
    )
    def test_get_jax_iterator_with_super_batch_known_batch_size(self):
        # Even batches: 4 batches of size 16 with super_batch=2 ->
        # 2 super-batches
        x = tf.ones((64, 4), dtype="float32")
        y = tf.ones((64, 2), dtype="float32")
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(
            16, drop_remainder=True
        )
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)
        self.assertEqual(adapter.batch_size, 16)
        self.verify_super_batched_iterator(
            adapter.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=False,
        )

        # Uneven batches: 5 batches of size 16 with super_batch=2 ->
        # 2 super-batches + 1 remainder super-batch of shape (1, 16, ...)
        x = tf.ones((80, 4), dtype="float32")
        y = tf.ones((80, 2), dtype="float32")
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(
            16, drop_remainder=True
        )
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)
        self.assertEqual(adapter.batch_size, 16)
        self.verify_super_batched_iterator(
            adapter.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=True,
        )

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="JAX only",
    )
    def test_get_jax_iterator_with_super_batch_unknown_batch_size(self):
        output_signature = (
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        )

        def make_gen(num_batches):
            def _gen():
                for _ in range(num_batches):
                    yield (
                        np.ones((16, 4), dtype="float32"),
                        np.ones((16, 2), dtype="float32"),
                    )

            return _gen

        # Even batches: 4 batches with super_batch=2 -> 2 super-batches
        base_ds = tf.data.Dataset.from_generator(
            make_gen(4), output_signature=output_signature
        )
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)
        self.assertIsNone(adapter.batch_size)
        self.verify_super_batched_iterator(
            adapter.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=False,
        )

        # Uneven batches: 5 batches with super_batch=2 -> 2 super-batches +
        # 1 partial batch list
        base_ds = tf.data.Dataset.from_generator(
            make_gen(5), output_signature=output_signature
        )
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)
        self.assertIsNone(adapter.batch_size)
        self.verify_super_batched_iterator(
            adapter.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=True,
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="TensorFlow only",
    )
    def test_fit_unknown_rank_dataset_steps_per_execution(self):
        def make_example(_):
            return np.ones((3,), dtype=np.float32), np.float32(1)

        def map_fn(index):
            return tf.numpy_function(
                make_example, [index], [tf.float32, tf.float32]
            )

        def map_fn_with_shape(index):
            x, y = map_fn(index)
            return tf.ensure_shape(x, (3,)), tf.ensure_shape(y, ())

        def build_model():
            inputs = layers.Input(shape=(3,))
            model = models.Model(inputs, layers.Dense(1)(inputs))
            model.compile(optimizer="sgd", loss="mse", steps_per_execution=2)
            return model

        dataset = tf.data.Dataset.range(2).map(map_fn).batch(1)
        with self.assertRaisesRegex(
            ValueError, "unknown rank.*tf.ensure_shape"
        ):
            build_model().fit(dataset, epochs=1, verbose=0)

        # Specifying the shapes makes multi-step training work.
        dataset = tf.data.Dataset.range(2).map(map_fn_with_shape).batch(1)
        build_model().fit(dataset, epochs=1, verbose=0)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX has the proper backend distribution lib",
)
@pytest.mark.multi_device
class DataShardingIntegrationTest(testing.TestCase):
    def test_distribute_dataset_sharding_behavior(self):
        num_devices = distribution_lib.get_device_count()
        self.assertGreaterEqual(
            num_devices, 4, "Number of devices must be at least 4"
        )
        self.assertEqual(num_devices % 2, 0, "Number of devices must be even")

        num_model_replicas = num_devices // 2
        global_dataset = tf.data.Dataset.range(4 * num_model_replicas).batch(
            2 * num_model_replicas
        )

        shards = []
        processes_per_replica = num_devices // num_model_replicas
        for process_id in range(num_devices):
            distribution = distribution_lib.DataParallel(
                devices=["cpu:0"] * num_model_replicas
            )
            distribution._num_processes = num_devices
            distribution._process_id = process_id
            distribution.auto_shard_dataset = True

            adapter = tf_dataset_adapter.TFDatasetAdapter(
                global_dataset, distribution=distribution
            )
            ds = adapter.get_tf_dataset()
            shards.append(list(ds.unbatch().as_numpy_iterator()))

        processes_per_replica = num_devices // num_model_replicas
        for replica_id in range(num_model_replicas):
            start = replica_id * processes_per_replica
            for process_id in range(start + 1, start + processes_per_replica):
                self.assertEqual(
                    shards[start],
                    shards[process_id],
                    f"Processes {start} and {process_id} should have same "
                    f"shard, got {shards}",
                )

        for replica_id in range(1, num_model_replicas):
            self.assertNotEqual(
                shards[0],
                shards[replica_id * processes_per_replica],
                f"Replica groups should have different shards, got {shards}",
            )
