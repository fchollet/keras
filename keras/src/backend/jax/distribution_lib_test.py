"""Test for distribution_lib.py."""

import functools
from unittest import mock

import jax
import numpy as np
import pytest
from jax.experimental import layout as jax_layout
from jax.sharding import PartitionSpec as P

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


@pytest.mark.multi_device
class JaxDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()

        self.device_count = jax.device_count()
        self.device_backend = jax.default_backend()
        self.assertGreaterEqual(
            self.device_count, 4, "Number of devices must be at least 4"
        )
        self.assertEqual(
            self.device_count % 2, 0, "Number of devices must be even"
        )
        self.mesh_shape = (self.device_count // 2, 2)

    def _create_jax_layout(self, sharding):
        # Use jax_layout.Format or jax_layout.Layout if available.
        if hasattr(jax_layout, "Format"):
            return jax_layout.Format(sharding=sharding)
        elif hasattr(jax_layout, "Layout"):
            return jax_layout.Layout(sharding=sharding)

        return sharding

    def test_get_device_count(self):
        self.assertEqual(backend_dlib.get_device_count(), self.device_count)
        self.assertEqual(
            backend_dlib.get_device_count(self.device_backend),
            self.device_count,
        )

    def test_list_devices(self):
        self.assertEqual(
            len(distribution_lib.list_devices()), self.device_count
        )
        self.assertEqual(
            len(distribution_lib.list_devices(self.device_backend)),
            self.device_count,
        )

    def test_device_conversion(self):
        devices = distribution_lib.list_devices(self.device_backend)
        jax_devices = jax.devices(self.device_backend)

        for d, jax_d in zip(devices, jax_devices):
            converted_jax_device = backend_dlib._to_backend_device(d)
            self.assertIsInstance(converted_jax_device, jax.Device)
            self.assertEqual(jax_d, converted_jax_device)

        # jax.Device input
        device = jax.devices()[0]
        self.assertEqual(backend_dlib._to_backend_device(device), device)
        # String without ':'
        res = backend_dlib._to_backend_device("cpu")
        self.assertEqual(res.platform, "cpu")
        # Invalid device
        with self.assertRaises(RuntimeError):
            backend_dlib._to_backend_device("invalid_backend:999")

    @mock.patch.object(jax.distributed, "initialize", return_value=None)
    def test_initialize_with_all_job_addresses(self, mock_jax_initialize):
        backend_dlib.initialize("10.0.0.1:1234,10.0.0.2:2345", 2, 0)
        mock_jax_initialize.assert_called_once_with(
            coordinator_address="10.0.0.1:1234", num_processes=2, process_id=0
        )

    def test_initialize_validate_job_and_process(self):
        with self.assertRaisesRegex(
            ValueError, "has 2 jobs, but num_processes is 3"
        ):
            backend_dlib.initialize("10.0.0.1:1234,10.0.0.2:2345", 3, 0)

    @mock.patch.object(jax.distributed, "initialize", return_value=None)
    def test_initialize_with_coordinator_address(self, mock_jax_initialize):
        backend_dlib.initialize("10.0.0.1:1234", 2, 0)
        mock_jax_initialize.assert_called_once_with(
            coordinator_address="10.0.0.1:1234", num_processes=2, process_id=0
        )

    def test_distribute_tensor(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )

        inputs = jax.numpy.array(
            np.random.normal(size=(self.mesh_shape[0] * 4, 8))
        )
        target_layout = jax.sharding.NamedSharding(jax_mesh, P("batch", None))

        @functools.partial(jax.jit, static_argnames="target_layout")
        def test_function(inputs, target_layout):
            return distribution_lib.distribute_tensor(inputs, target_layout)

        result = test_function(inputs, target_layout)
        # Note that the returned tensor has a different sharding implementation
        # which is GSPMDSharding, but it should be equivalent as the target
        # layout specified.
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

        # Test without jit
        result = distribution_lib.distribute_tensor(inputs, target_layout)
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

        # Non-JAX array
        x_list = jax.numpy.array([1.0, 2.0])
        res = backend_dlib.distribute_tensor(x_list, target_layout)
        self.assertIsInstance(res, jax.Array)

        # Already distributed jax.Array (equivalent sharding)
        res = backend_dlib.distribute_tensor(result, target_layout)
        self.assertIs(res, result)

    def test_distribute_tensor_with_jax_layout(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )

        inputs = jax.numpy.array(
            np.random.normal(size=(self.mesh_shape[0] * 4, 8))
        )
        target_layout = self._create_jax_layout(
            sharding=jax.sharding.NamedSharding(jax_mesh, P("batch", None))
        )

        @functools.partial(jax.jit, static_argnames="target_layout")
        def test_function(inputs, target_layout):
            return distribution_lib.distribute_tensor(inputs, target_layout)

        result = test_function(inputs, target_layout)
        # Note that the returned tensor has a different sharding implementation
        # which is GSPMDSharding, but it should be equivalent as the target
        # layout specified.
        self.assertTrue(
            result.sharding.is_equivalent_to(target_layout.sharding, ndim=2)
        )

        # Test without jit.
        result = distribution_lib.distribute_tensor(inputs, target_layout)
        self.assertTrue(
            result.sharding.is_equivalent_to(target_layout.sharding, ndim=2)
        )

    def test_variable_with_jax_layout(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )
        target_layout = self._create_jax_layout(
            sharding=jax.sharding.NamedSharding(jax_mesh, P("batch", None))
        )
        variable = backend.Variable(
            "ones", shape=(16, 16), dtype="float32", layout=target_layout
        )

        self.assertTrue(
            variable._value.sharding.is_equivalent_to(
                target_layout.sharding, ndim=2
            )
        )

    def test_processes(self):
        self.assertEqual(backend_dlib.process_id(), 0)
        self.assertEqual(backend_dlib.num_processes(), 1)

    def test_to_backend_mesh(self):
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(self.mesh_shape, axis_names)
        jax_mesh = backend_dlib._to_backend_mesh(mesh)

        self.assertIsInstance(jax_mesh, jax.sharding.Mesh)
        self.assertEqual(jax_mesh.devices.shape, self.mesh_shape)
        self.assertEqual(jax_mesh.axis_names, ("batch", "model"))

    def test_to_backend_layout(self):
        axes = ["data", None]
        mesh = distribution_lib.DeviceMesh(self.mesh_shape, ["data", "model"])
        layout = distribution_lib.TensorLayout(axes, mesh)
        jax_sharding = backend_dlib._to_backend_layout(layout)
        jax_mesh = backend_dlib._to_backend_mesh(mesh)
        self.assertEqual(
            jax_sharding,
            jax.sharding.NamedSharding(jax_mesh, P("data", None)),
        )

    def test_validation_for_device_mesh(self):
        axes = ["data", None]
        layout = distribution_lib.TensorLayout(axes, device_mesh=None)

        with self.assertRaisesRegex(
            ValueError, "Cannot create sharding when device mesh is not set"
        ):
            backend_dlib._to_backend_layout(layout)

    def test_variable_assignment_reuse_layout(self):
        axis_names = ["batch", "model"]
        device_mesh = distribution_lib.DeviceMesh(
            self.mesh_shape, axis_names, backend_dlib.list_devices()
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )

        with distribution.scope():
            dense_layer = layers.Dense(8)
            dense_layer.build((16, 16))

        self.assertEqual(
            dense_layer.kernel._value.sharding.spec, P(None, "model")
        )
        self.assertEqual(dense_layer.bias._value.sharding.spec, P("model"))

        # Assign a numpy value to dense layer to mimic the model weight loading
        new_kernel = np.random.normal(size=(16, 8))
        new_bias = np.random.normal(size=(8))
        dense_layer.kernel.assign(new_kernel)
        dense_layer.bias.assign(new_bias)

        # Make sure the loaded value still use the layout when it is
        # initialized, even outside of the distribution scope.
        self.assertEqual(
            dense_layer.kernel._value.sharding.spec, P(None, "model")
        )
        self.assertEqual(dense_layer.bias._value.sharding.spec, P("model"))

    def test_e2e_data_parallel_model(self):
        distribution = distribution_lib.DataParallel()

        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Make sure all the weights are properly sharded.
        for weight in model.weights:
            self.assertTrue(weight._value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(self.device_count * 8, 28, 28, 1))
        labels = np.random.normal(size=(self.device_count * 8, 10))

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(inputs, labels, batch_size=self.device_count)

    def test_e2e_model_parallel_model(self):
        axis_names = ["batch", "model"]
        device_mesh = distribution_lib.DeviceMesh(
            self.mesh_shape, axis_names, backend_dlib.list_devices()
        )

        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])

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
            if "kernel" in weight.name:
                self.assertEqual(weight._value.sharding.spec, P(None, "model"))
            elif "bias" in weight.name:
                self.assertEqual(weight._value.sharding.spec, P("model"))
            else:
                self.assertTrue(weight._value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(self.device_count * 8, 28, 28, 1))
        labels = np.random.normal(size=(self.device_count * 8, 10))

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(inputs, labels, batch_size=self.device_count)

    def test_e2e_model_parallel_with_output_sharding(self):
        axis_names = ["batch", "model"]
        device_mesh = distribution_lib.DeviceMesh(
            self.mesh_shape, axis_names, backend_dlib.list_devices()
        )

        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])
        # Force the dense layer output to be batch parallel only, and not
        # sharded on model dimension.
        layout_map[".*dense.*output"] = ("batch", None)

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        sharding_capture = ShardingCaptureLayer()
        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = sharding_capture(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        for weight in model.weights:
            if "kernel" in weight.name:
                self.assertEqual(weight._value.sharding.spec, P(None, "model"))
            elif "bias" in weight.name:
                self.assertEqual(weight._value.sharding.spec, P("model"))
            else:
                self.assertTrue(weight._value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(self.device_count * 8, 28, 28, 1))
        labels = np.random.normal(size=(self.device_count * 8, 10))

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(inputs, labels, batch_size=self.device_count)

        # Note that the intermediate_tensor_layout is only captured during the
        # actual training, and not at the model building time.
        intermediate_tensor_layout = jax.sharding.NamedSharding(
            backend_dlib._to_backend_mesh(distribution.device_mesh),
            P("batch", None),
        )
        self.assertTrue(
            sharding_capture.captured_input_sharding.is_equivalent_to(
                intermediate_tensor_layout, ndim=2
            )
        )

    def test_distribute_data_input(self):
        per_process_batch = jax.numpy.arange(
            3 * self.mesh_shape[0] * 5
        ).reshape((3 * self.mesh_shape[0], 5))  # Example input array
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape),
            axis_names=["batch", "model"],
        )
        layout = jax.sharding.NamedSharding(mesh, P("batch", None))

        result = backend_dlib.distribute_data_input(
            per_process_batch, layout, "batch"
        )

        # Check the shape of the global batch array
        self.assertEqual(
            result.shape, (3 * self.mesh_shape[0], 5)
        )  # (per_replica_batch_size * num_model_replicas_total, 5)

        # Check the sharding of the global batch array
        self.assertEqual(len(result.addressable_shards), self.device_count)
        for shard in result.addressable_shards:
            self.assertEqual(shard.data.shape, (3, 5))

    def test_all_reduce_eager(self):
        # Eager mode (no distribution context -> uses local_device_count)
        axis_size = self.device_count
        x = np.ones((axis_size * 2, 4), dtype=np.float32)

        # Test sum
        res_sum = backend_dlib.all_reduce(x, op="sum", axis_name="model")
        self.assertEqual(res_sum.shape, x.shape)
        expected_sum = np.full(x.shape, axis_size, dtype=np.float32)
        np.testing.assert_allclose(res_sum, expected_sum)

        # Test mean
        res_mean = backend_dlib.all_reduce(x, op="mean", axis_name="model")
        self.assertEqual(res_mean.shape, x.shape)
        expected_mean = np.ones(x.shape, dtype=np.float32)
        np.testing.assert_allclose(res_mean, expected_mean)

        # Test errors
        with self.assertRaisesRegex(
            ValueError, "Unsupported reduction operation"
        ):
            backend_dlib.all_reduce(x, op="invalid", axis_name="model")

    def test_all_gather_eager(self):
        # Eager mode (no distribution context -> uses local_device_count)
        axis_size = self.device_count
        x = np.ones((axis_size * 2, 2), dtype=np.float32)

        # Gather along axis 1
        res = backend_dlib.all_gather(x, axis=1, axis_name="model")
        self.assertEqual(res.shape, (axis_size * 2, 2 * axis_size))

        # Gather along axis 0
        res0 = backend_dlib.all_gather(x, axis=0, axis_name="model")
        self.assertEqual(res0.shape, (axis_size * 2 * axis_size, 2))

        # Test fallback for non-sharded jax.Array
        x_jax = jax.numpy.ones((2, 2))
        res_jax = backend_dlib.all_gather(x_jax, axis=0, axis_name="model")
        axis_size_val = backend_dlib._get_axis_size("model")
        self.assertEqual(res_jax.shape, (2 * axis_size_val, 2))
        self.assertIsInstance(res_jax, jax.Array)

    def test_all_reduce_eager_with_distribution(self):
        axis_names = ["batch", "model"]
        device_mesh = distribution_lib.DeviceMesh(
            self.mesh_shape, axis_names, backend_dlib.list_devices()
        )
        # self.mesh_shape = (4, 2). 'model' axis size is 2.

        dist = distribution_lib.ModelParallel(
            device_mesh=device_mesh,
            layout_map=distribution_lib.LayoutMap(device_mesh),
        )

        with dist.scope():
            x = np.ones((2, 4), dtype=np.float32)
            # Should use axis_size=2 from the mesh axis 'model'
            res = backend_dlib.all_reduce(x, op="sum", axis_name="model")
            self.assertEqual(res.shape, x.shape)
            np.testing.assert_allclose(res, 2.0)

            # Test mean
            res_mean = backend_dlib.all_reduce(x, op="mean", axis_name="model")
            self.assertEqual(res_mean.shape, x.shape)
            np.testing.assert_allclose(res_mean, 1.0)

            # Test axis not in mesh
            res_default = backend_dlib.all_reduce(
                x, op="sum", axis_name="unknown"
            )
            # Should fallback to jax.local_device_count()
            np.testing.assert_allclose(res_default, float(self.device_count))

    def test_all_reduce_fallback_in_jit(self):
        x = np.ones((4, 4), dtype=np.float32)

        @jax.jit
        def reduce_fn(y):
            return backend_dlib.all_reduce(y, op="sum", axis_name="model")

        res = reduce_fn(x)
        # Should return x unchanged because axis is not bound
        np.testing.assert_allclose(res, x)

    def test_all_reduce_errors(self):
        x = np.ones((4, 4), dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError, "Unsupported reduction operation"
        ):
            backend_dlib.all_reduce(x, op="invalid", axis_name="model")

    def test_all_reduce_sharded(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )
        x = jax.numpy.ones((self.mesh_shape[0] * 2, self.mesh_shape[1] * 2))
        sharding = jax.sharding.NamedSharding(jax_mesh, P("batch", "model"))
        x_sharded = jax.device_put(x, sharding)

        # sum
        res_sum = backend_dlib.all_reduce(
            x_sharded, op="sum", axis_name="model"
        )
        self.assertIsInstance(res_sum, jax.Array)
        # Should be shape-preserving logically
        self.assertEqual(res_sum.shape, x.shape)
        # model axis has size self.mesh_shape[1]
        np.testing.assert_allclose(res_sum, float(self.mesh_shape[1]))
        # output should be replicated on "model" axis
        self.assertEqual(res_sum.sharding.spec, P("batch", None))

        # mean
        res_mean = backend_dlib.all_reduce(
            x_sharded, op="mean", axis_name="model"
        )
        self.assertEqual(res_mean.shape, x.shape)
        np.testing.assert_allclose(res_mean, 1.0)
        self.assertEqual(res_mean.sharding.spec, P("batch", None))

        # Test the case where axis == -1 in sharded array logic
        sharding_batch = jax.sharding.NamedSharding(jax_mesh, P("batch", None))
        x_sharded_batch = jax.device_put(
            jax.numpy.ones((self.mesh_shape[0] * 2, 4)), sharding_batch
        )
        res_not_sharded = backend_dlib.all_reduce(
            x_sharded_batch, op="sum", axis_name="model"
        )
        # model axis has size self.mesh_shape[1] = 2.
        # Since x is replicated on 'model', psum on 'model' returns x * 2.0
        np.testing.assert_allclose(
            res_not_sharded, x_sharded_batch * float(self.mesh_shape[1])
        )

    def test_all_gather_sharded(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )
        # Global shape (8, 4) if mesh is (4, 2)
        x = jax.numpy.ones((self.mesh_shape[0] * 2, self.mesh_shape[1] * 2))
        sharding = jax.sharding.NamedSharding(jax_mesh, P("batch", "model"))
        x_sharded = jax.device_put(x, sharding)

        # Gather along axis 1, mesh axis "model"
        res = backend_dlib.all_gather(x_sharded, axis=1, axis_name="model")
        self.assertIsInstance(res, jax.Array)
        self.assertEqual(res.shape, x.shape)
        # The output of all_gather is replicated along the gathered mesh axis
        self.assertEqual(res.sharding.spec, P("batch", None))

    def test_all_gather_fallback_in_jit(self):
        x = np.ones((4, 4), dtype=np.float32)

        @jax.jit
        def gather_fn(y):
            return backend_dlib.all_gather(y, axis=1, axis_name="model")

        res = gather_fn(x)
        # Should return x unchanged because axis is not bound
        np.testing.assert_allclose(res, x)

    def test_distribute_tensor_extra(self):
        # Non-JAX array
        x = jax.numpy.array([1.0, 2.0])
        layout = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        res = backend_dlib.distribute_tensor(x, layout)
        self.assertIsInstance(res, jax.Array)

        # Already distributed jax.Array (equivalent sharding)
        x_jax = jax.device_put(jax.numpy.ones((2, 2)), layout)
        res = backend_dlib.distribute_tensor(x_jax, layout)
        self.assertIs(res, x_jax)

    def test_to_backend_device_extra(self):
        device = jax.devices()[0]
        # jax.Device input
        self.assertEqual(backend_dlib._to_backend_device(device), device)
        # String without ':'
        res = backend_dlib._to_backend_device("cpu")
        self.assertEqual(res.platform, "cpu")
        with self.assertRaises(RuntimeError):
            backend_dlib._to_backend_device("invalid_backend:999")

    def test_all_reduce_axis_not_sharded(self):
        # Test the case where axis == -1 in sharded array logic
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.mesh_shape), ("batch", "model")
        )
        x = jax.numpy.ones((self.mesh_shape[0] * 2, 4))
        sharding = jax.sharding.NamedSharding(jax_mesh, P("batch", None))
        x_sharded = jax.device_put(x, sharding)

        res = backend_dlib.all_reduce(x_sharded, op="sum", axis_name="model")
        # model axis has size self.mesh_shape[1] = 2.
        # Since x is replicated on 'model', psum on 'model' returns x * 2.0
        np.testing.assert_allclose(res, x * float(self.mesh_shape[1]))

    def test_all_gather_fallback_jax_array(self):
        # Test fallback for non-sharded jax.Array
        x = jax.numpy.ones((2, 2))
        res = backend_dlib.all_gather(x, axis=0, axis_name="model")
        axis_size = backend_dlib._get_axis_size("model")
        self.assertEqual(res.shape, (2 * axis_size, 2))
        self.assertIsInstance(res, jax.Array)


class ShardingCaptureLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_input_sharding = None
        self.supports_masking = True

    def call(self, inputs):
        jax.debug.inspect_array_sharding(
            inputs, callback=lambda x: self.capture_input_sharding(x)
        )
        return inputs

    def capture_input_sharding(self, sharding):
        self.captured_input_sharding = sharding
