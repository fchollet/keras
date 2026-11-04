import os
import sys

import numpy as np
import pytest
import tensorflow as tf
import torch
from absl.testing import parameterized
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import testing
from keras.src.backend import backend
from keras.src.backend.torch import distribution_lib as torch_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import tf_dataset_adapter
from keras.src.utils import rng_utils


class MultiProcessTest:
    pass


@pytest.mark.multi_device
@pytest.mark.no_pytest_xdist
@pytest.mark.skipif(backend() != "torch", reason="Torch only")
class TorchMultiProcessDistributeTest(
    MultiProcessTest, testing.TestCase, parameterized.TestCase
):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_env = {
            key: os.environ.get(key)
            for key in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
            )
        }
        if not torch.distributed.is_initialized():
            tf.config.set_visible_devices([], "GPU")
            num_processes = int(os.environ.get("WORLD_SIZE", 1))
            process_id = int(os.environ.get("RANK", 0))
            for arg in sys.argv:
                if arg.startswith("--num_processes="):
                    num_processes = int(arg.split("=", 1)[1])
                elif arg.startswith("--multiprocess_test_worker_id="):
                    worker_id = int(arg.split("=", 1)[1])
                    if worker_id >= 0:
                        process_id = worker_id
            os.environ.setdefault("LOCAL_RANK", str(process_id))
            torch_distribution_lib.initialize(
                job_addresses="127.0.0.1:29500",
                num_processes=num_processes,
                process_id=process_id,
            )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for key, old_value in cls._saved_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        # We need a consistent seed across all processes.
        rng_utils.set_random_seed(1234)

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        num_processes = torch_distribution_lib.num_processes()
        num_devices = torch_distribution_lib.get_device_count()

        if num_processes > 1:
            self.assertEqual(num_processes, 2)
            self.assertEqual(num_devices, 2)

    def _assert_weights_in_sync(self, model):
        world_size = torch.distributed.get_world_size()
        for weight in model.weights:
            value = weight.value
            if isinstance(value, DTensor):
                value = value.full_tensor()
            value = value.detach().contiguous()
            gathered = [torch.empty_like(value) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, value)
            for other in gathered[1:]:
                self.assertAllClose(gathered[0], other, msg=weight.path)

    def _model_parallel(self, mesh_shape):
        """Returns a `ModelParallel` that shards the `Dense` variables."""
        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = (None, "model")
        layout_map[".*dense.*bias"] = ("model",)
        # The tests feed NumPy arrays, which cannot be auto-sharded.
        return distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="batch",
            auto_shard_dataset=False,
        )

    def _dense_model(self, units):
        return models.Sequential(
            [layers.Input(shape=(8,)), layers.Dense(units)]
        )

    def test_list_device(self):
        devices = distribution_lib.list_devices()
        self.assertEqual(
            len(devices), torch_distribution_lib.get_device_count()
        )

        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        for d in devices:
            converted_torch_device = torch_distribution_lib._to_backend_device(
                d
            )
            self.assertIsInstance(converted_torch_device, torch.device)
            self.assertEqual(converted_torch_device.type, expected_type)

    def test_distribute_variable(self):
        num_devices = torch_distribution_lib.get_device_count()

        global_shape = (8, 4)
        kernel = np.arange(np.prod(global_shape), dtype="float32").reshape(
            global_shape
        )

        axis_names = ["batch", "model"]
        model_dim = min(2, num_devices)
        device_mesh = distribution_lib.DeviceMesh(
            shape=(num_devices // model_dim, model_dim),
            axis_names=axis_names,
            devices=distribution_lib.list_devices(),
        )
        layout = distribution_lib.TensorLayout(
            axes=("batch", "model"),
            device_mesh=device_mesh,
        )

        distributed_kernel = torch_distribution_lib.distribute_tensor(
            kernel, layout
        )
        # After distribution, the value should be a global shape DTensor.
        self.assertIsInstance(distributed_kernel, DTensor)
        self.assertEqual(tuple(distributed_kernel.shape), global_shape)
        self.assertEqual(
            distributed_kernel.placements,
            (Shard(0), Shard(1)),
        )

        # Each local process has 1 local shard with shape divided by mesh dims.
        expected_local_shape = (
            global_shape[0] // (num_devices // model_dim),
            global_shape[1] // model_dim,
        )
        self.assertEqual(
            tuple(distributed_kernel.to_local().shape), expected_local_shape
        )

        # Also make sure the gathered global value has the same value as the
        # original value.
        local_copy = distributed_kernel.full_tensor()
        self.assertAllClose(local_copy, kernel)

    def test_dataset_distribution_data_parallel(self):
        num_processes = torch_distribution_lib.num_processes()

        # Create a dataset with range, so that we can verify the numerical
        # correctness.
        global_batch_size = num_processes * 2
        num_batch = 4
        dataset = tf.data.Dataset.range(global_batch_size * num_batch).batch(
            global_batch_size
        )
        distribution = distribution_lib.DataParallel(
            devices=distribution_lib.list_devices()
        )

        # Since there are `num_processes` worker/processes, we will have
        # `num_processes` shards of the data.
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()

        process_id = torch_distribution_lib.process_id()
        per_process_batch_size = global_batch_size // num_processes
        expected_value = (
            np.arange(per_process_batch_size)
            + process_id * per_process_batch_size
        )
        for d in distributed_dataset:
            self.assertEqual(d.shape, (per_process_batch_size,))
            self.assertAllClose(d, expected_value)
            expected_value += global_batch_size

    @parameterized.named_parameters(
        [
            ("data_only", 1),
            ("data_model", 2),
            ("model_data", 4),
            ("model_only", 8),
        ]
    )
    def test_dataset_distribution_model_parallel(self, model_dim):
        num_processes = torch_distribution_lib.num_processes()
        num_devices = torch_distribution_lib.get_device_count()
        local_devices = num_devices // num_processes
        # Ensure model_dim doesn't exceed available devices.
        model_dim = min(model_dim, num_devices)
        mesh_shape = (num_devices // model_dim, model_dim)

        global_batch_size = 8
        num_batch = 4
        dataset = tf.data.Dataset.range(global_batch_size * num_batch).batch(
            global_batch_size
        )

        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        distribution = distribution_lib.ModelParallel(layout_map=layout_map)

        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()

        process_id = torch_distribution_lib.process_id()
        # Calculate how many replicas this local process is responsible for.
        num_devices_per_replica = max(1, num_devices // mesh_shape[0])
        num_local_replicas = max(1, local_devices // num_devices_per_replica)
        per_worker_batch_size = num_local_replicas * (
            global_batch_size // mesh_shape[0]
        )

        expected_value = np.arange(per_worker_batch_size)
        processes_per_replica = num_processes // mesh_shape[0]
        if processes_per_replica > 1:
            worker_factor = process_id // processes_per_replica
        else:
            worker_factor = process_id
        expected_value += worker_factor * per_worker_batch_size

        for batch_index, batch in enumerate(distributed_dataset):
            self.assertEqual(batch.shape, (per_worker_batch_size,))
            self.assertAllClose(
                batch,
                expected_value,
                msg=f"process {process_id} batch {batch_index}",
            )
            expected_value += global_batch_size

    def test_e2e_data_parallel_model(self):
        distribution = distribution_lib.DataParallel(
            devices=distribution_lib.list_devices(),
        )

        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Make sure all the weights have replicated layout.
        for weight in model.weights:
            self.assertTrue(all(axis is None for axis in weight._layout.axes))

        inputs = np.random.normal(size=(128, 28, 28, 1)).astype("float32")
        labels = np.random.normal(size=(128, 10)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(dataset, epochs=3)

            model.evaluate(dataset)
        self._assert_weights_in_sync(model)

    @parameterized.named_parameters(
        [
            ("data_only_replicated", 1, False),
            ("data_only_sharded", 1, True),
            ("model_only_replicated", 2, False),
            ("model_only_sharded", 2, True),
        ]
    )
    def test_e2e_data_model_parallel_model(self, model_dim, shard_weights):
        num_devices = torch_distribution_lib.get_device_count()
        model_dim = min(model_dim, num_devices)
        mesh_shape = (num_devices // model_dim, model_dim)

        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        if shard_weights:
            layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
                [None, "model"]
            )
            layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(
                ["model"]
            )
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )

        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        for weight in model.weights:
            if shard_weights:
                self.assertIsInstance(weight.value, DTensor)
                if "kernel" in weight.name:
                    self.assertEqual(
                        weight.value.placements, (Replicate(), Shard(1))
                    )
                elif "bias" in weight.name:
                    self.assertEqual(
                        weight.value.placements, (Replicate(), Shard(0))
                    )
            else:
                self.assertTrue(
                    all(axis is None for axis in weight._layout.axes)
                )

        inputs = np.random.normal(size=(128, 28, 28, 1)).astype("float32")
        labels = np.random.normal(size=(128, 10)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(dataset, epochs=3)

            model.evaluate(dataset)
        self._assert_weights_in_sync(model)

    @parameterized.named_parameters(
        [
            ("data_only", 1),
            ("model_only", 2),
        ]
    )
    def test_e2e_model_parallel_lamb(self, model_dim):
        # Lamb updates sharded kernels through `assign` with `Partial`
        # gradients, which must be redistributed to the kernel layout.
        num_devices = torch_distribution_lib.get_device_count()
        model_dim = min(model_dim, num_devices)
        device_mesh = distribution_lib.DeviceMesh(
            shape=(num_devices // model_dim, model_dim),
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )

        def build_model():
            inputs = layers.Input(shape=[8])
            y = layers.Dense(units=16, activation="relu")(inputs)
            y = layers.Dense(units=4)(y)
            return models.Model(inputs=inputs, outputs=y)

        inputs = np.random.normal(size=(32, 8)).astype("float32")
        labels = np.random.normal(size=(32, 4)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        # Unsharded reference, trained on the same batches.
        reference = build_model()
        initial_weights = reference.get_weights()
        reference.compile(
            optimizer=optimizers.Lamb(learning_rate=0.01), loss="mse"
        )
        reference.fit(dataset, epochs=1)

        with distribution.scope():
            model = build_model()
            model.set_weights(initial_weights)
            model.compile(
                optimizer=optimizers.Lamb(learning_rate=0.01), loss="mse"
            )
            model.fit(dataset, epochs=1)

        for weight, expected in zip(model.weights, reference.weights):
            value = weight.value
            if isinstance(value, DTensor):
                value = value.full_tensor()
            self.assertAllClose(value, expected, msg=weight.path)

    def test_convert_to_numpy_gathers_sharded_variable(self):
        num_processes = torch_distribution_lib.num_processes()
        distribution = self._model_parallel((1, num_processes))
        expected = np.arange(32, dtype="float32").reshape(8, 4)

        with distribution.scope():
            kernel = self._dense_model(4).layers[0].kernel
            kernel.assign(expected)

            # Each process holds only its own columns of the kernel.
            self.assertIsInstance(kernel.value, DTensor)
            self.assertEqual(
                tuple(kernel.value.to_local().shape), (8, 4 // num_processes)
            )
            for value in (
                ops.convert_to_numpy(kernel),
                np.asarray(kernel),
                kernel.numpy(),
            ):
                self.assertEqual(value.shape, (8, 4))
                self.assertAllClose(value, expected)

    def test_get_and_set_weights_sharded_model(self):
        num_processes = torch_distribution_lib.num_processes()
        distribution = self._model_parallel((1, num_processes))

        with distribution.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(4), layers.Dense(2)]
            )
            weights = model.get_weights()
            self.assertEqual(
                [w.shape for w in weights], [(8, 4), (4,), (4, 2), (2,)]
            )

            model.set_weights([w + 1 for w in weights])
            for old, new in zip(weights, model.get_weights()):
                self.assertAllClose(new, old + 1)

    def test_predict_gathers_sharded_output(self):
        num_processes = torch_distribution_lib.num_processes()
        distribution = self._model_parallel((1, num_processes))
        rng = np.random.default_rng(0)
        x = rng.random((16, 8), dtype="float32")
        kernel = rng.random((8, 6), dtype="float32")
        bias = rng.random((6,), dtype="float32")

        with distribution.scope():
            model = self._dense_model(6)
            model.set_weights([kernel, bias])

            outputs = model.predict(x, batch_size=8, verbose=0)
            self.assertAllClose(outputs, x @ kernel + bias, atol=1e-5)
            outputs = model.predict_on_batch(x)
            self.assertAllClose(outputs, x @ kernel + bias, atol=1e-5)

    def test_predict_keeps_process_rows(self):
        num_processes = torch_distribution_lib.num_processes()
        process_id = torch_distribution_lib.process_id()
        # Each process feeds its own rows: the batch axis spans the processes.
        distribution = self._model_parallel((num_processes, 1))
        rng = np.random.default_rng(0)
        kernel = rng.random((8, 3), dtype="float32")
        bias = rng.random((3,), dtype="float32")
        x = np.full((4, 8), process_id + 1, "float32")

        with distribution.scope():
            model = self._dense_model(3)
            model.set_weights([kernel, bias])

            outputs = model.predict(x, batch_size=4, verbose=0)
            self.assertAllClose(outputs, x @ kernel + bias, atol=1e-5)
            outputs = model.predict_on_batch(x)
            self.assertAllClose(outputs, x @ kernel + bias, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
