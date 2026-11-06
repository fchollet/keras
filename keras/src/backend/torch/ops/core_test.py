"""Tests for PyTorch backend core utilities."""

import concurrent.futures
import threading

import numpy as np
import pytest
import torch
from absl.testing import parameterized
from torch._dynamo.testing import CompileCounter
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.torch import distribution_lib
from keras.src.backend.torch.distributed_test_utils import (
    TorchDistributedTestMixin,
)
from keras.src.backend.torch.ops import core as torch_core
from keras.src.backend.torch.ops.core import KerasDTensorPromotionMode
from keras.src.backend.torch.ops.core import Variable
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.backend.torch.ops.core import slice as torch_slice
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel
from keras.src.distribution.distribution_lib import set_distribution


def _get_backed_symint(hint=2):
    """Create a backed SymInt via torch.export dynamic shapes."""

    class _M(torch.nn.Module):
        def forward(self, x):
            return x + x.shape[0]

    ep = torch.export.export(
        _M(),
        (torch.randn(hint, 3),),
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            return node.meta["val"].shape[0]
    raise RuntimeError("Could not extract SymInt from exported program")


def _get_backed_symfloat(hint=2):
    """Create a backed SymFloat from a backed SymInt."""
    return torch.sym_float(_get_backed_symint(hint))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchCoreTest(testing.TestCase):
    def _assert_sym_convert(
        self,
        value,
        expected_dtype,
        expected_item=None,
        expected_shape=None,
        expected_values=None,
        dtype=None,
    ):
        result = convert_to_tensor(value, dtype=dtype)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, expected_dtype)
        if expected_item is not None:
            self.assertEqual(result.item(), expected_item)
        if expected_shape is not None:
            self.assertEqual(tuple(result.shape), expected_shape)
        if expected_values is not None:
            self.assertListEqual(result.tolist(), expected_values)

    def test_convert_to_tensor_symint_scalar(self):
        self._assert_sym_convert(
            _get_backed_symint(5), torch.int64, expected_item=5
        )

    def test_convert_to_tensor_symfloat_scalar(self):
        self._assert_sym_convert(
            _get_backed_symfloat(5), torch.float32, expected_item=5.0
        )

    def test_convert_to_tensor_list_of_symint(self):
        self._assert_sym_convert(
            [_get_backed_symint(3), _get_backed_symint(4)],
            torch.int64,
            expected_shape=(2,),
            expected_values=[3, 4],
        )

    def test_convert_to_tensor_tuple_of_symfloat(self):
        self._assert_sym_convert(
            (_get_backed_symfloat(3), _get_backed_symfloat(4)),
            torch.float32,
            expected_shape=(2,),
            expected_values=[3.0, 4.0],
        )

    def test_convert_to_tensor_nested_list_of_symint(self):
        self._assert_sym_convert(
            [[_get_backed_symint(3), _get_backed_symint(4)]],
            torch.int64,
            expected_shape=(1, 2),
            expected_values=[[3, 4]],
        )

    def test_convert_to_tensor_explicit_dtype_for_symint(self):
        self._assert_sym_convert(
            _get_backed_symint(5),
            torch.float32,
            dtype="float32",
        )

    def _get_other_device(self):
        """Returns a device type that is not the type of `get_device()`."""
        keras_device = get_device().split(":")[0]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            if device != keras_device:
                return device
        self.skipTest(
            f"Needs a device other than `get_device()` ({keras_device!r}). "
            "This host has no CUDA or MPS device."
        )

    def _get_inputs_gradient(self, fn, x_value, device):
        x = torch.tensor(x_value, device=device, requires_grad=True)
        fn(x).sum().backward()
        return x.grad

    def test_custom_gradient_on_inputs_device(self):
        """The gradient is on the device of the input, not `get_device()`."""
        other_device = self._get_other_device()

        @ops.custom_gradient
        def scaled_square(x):
            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                return ops.multiply(upstream, ops.multiply(3.0, x))

            return ops.square(x), grad_fn

        x_value = np.arange(6, dtype="float32").reshape(2, 3)
        grad = self._get_inputs_gradient(scaled_square, x_value, other_device)
        expected = self._get_inputs_gradient(
            scaled_square, x_value, get_device()
        )
        self.assertEqual(grad.device.type, other_device)
        # The custom gradient is 3 * x, not the autodiff gradient 2 * x.
        self.assertAllClose(grad, 3.0 * x_value)
        self.assertAllClose(grad, expected)

    @parameterized.named_parameters(("int8", "int8"), ("int4", "int4"))
    def test_quantized_dense_inputs_gradient_on_inputs_device(self, mode):
        """The quantized forward uses a custom gradient for the inputs."""
        other_device = self._get_other_device()
        layer = layers.Dense(8)
        layer.build((None, 16))
        layer.quantize(mode)

        x_value = np.random.RandomState(0).standard_normal((2, 16))
        x_value = x_value.astype("float32")
        grad = self._get_inputs_gradient(layer, x_value, other_device)
        expected = self._get_inputs_gradient(layer, x_value, get_device())
        self.assertEqual(grad.device.type, other_device)
        self.assertAllClose(grad, expected)

    def test_slice_fast_path_accepts_symint(self):
        """slice fast path should accept SymInt without crashing."""
        x = torch.arange(24).reshape(2, 3, 4)
        batch = _get_backed_symint(2)
        start_indices = [0, 0, 0]
        shape = [batch, 2, 2]
        result = torch_slice(x, start_indices, shape)
        self.assertEqual(tuple(result.shape), (2, 2, 2))

    def test_slice_with_zero_d_tensor_start_indices(self):
        """slice must accept 0-d integer tensors as start indices."""
        device = get_device()
        x = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(device)
        out = torch_slice(
            x,
            [torch.tensor(0).to(device), torch.tensor(1).to(device)],
            [2, 3],
        )
        expected = x[0:2, 1:4]
        self.assertAllClose(out, expected)

    def test_to_static_index_rejects_tensor(self):
        """Tensor bounds must be rejected so `slice()` keeps them traceable."""
        from keras.src.backend.torch.ops.core import _to_static_index

        with self.assertRaises(TypeError):
            _to_static_index(torch.tensor(0))
        with self.assertRaises(TypeError):
            _to_static_index(torch.tensor(0.0))

    def test_slice_export_preserves_dynamic_dim(self):
        """A numpy-int bound alongside a symbolic dim keeps the dim dynamic."""

        class _SliceModule(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                return torch_slice(x, [np.int64(0), 0], [n, 2])

        ep = torch.export.export(
            _SliceModule(),
            (torch.arange(8).reshape(2, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2)}},
        )
        # Re-running with a different batch size must work; a specialized dim
        # would have failed export above or fixed the first output dim to 2.
        out = ep.module()(torch.arange(12).reshape(3, 4))
        self.assertEqual(tuple(out.shape), (3, 2))

    @parameterized.parameters(((3,),), ((2, 3),))
    def test_custom_gradient_single_output(self, shape):
        """A result with no `grad_fn` is the output and has no gradient."""
        calls = 0

        @torch_core.custom_gradient
        def triple(x):
            nonlocal calls
            calls += 1
            return x * 3.0

        x = torch.ones(shape, requires_grad=True)
        y = triple(x)
        self.assertEqual(calls, 1)
        self.assertEqual(tuple(y.shape), shape)
        self.assertAllClose(y, np.full(shape, 3.0))
        with self.assertRaisesRegex(ValueError, "grad_fn must be provided"):
            y.sum().backward()


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device not available"
)
class TorchCoreMpsTest(testing.TestCase):
    def test_convert_to_tensor_downcasts_ndarray_on_mps(self):
        x = np.ones(3, dtype=np.float64)
        with backend.device_scope("mps"):
            out = convert_to_tensor(x)
        self.assertEqual(standardize_dtype(out.dtype), "float32")

        x = np.ones(3, dtype=np.complex128)
        with backend.device_scope("mps"):
            out = convert_to_tensor(x)
        self.assertEqual(standardize_dtype(out.dtype), "complex64")

    def test_convert_to_tensor_downcasts_sequence_on_mps(self):
        x = [np.float64(1.0), np.float64(2.0)]
        with backend.device_scope("mps"):
            out = convert_to_tensor(x)
        self.assertEqual(standardize_dtype(out.dtype), "float32")

    def test_convert_to_tensor_downcasts_torch_tensor_on_mps(self):
        x = torch.ones(3, dtype=torch.float64)
        with backend.device_scope("mps"):
            out = convert_to_tensor(x)
        self.assertEqual(standardize_dtype(out.dtype), "float32")

    def test_convert_to_tensor_meta_tensor_on_mps(self):
        x = torch.empty(3, dtype=torch.float64, device="meta")
        with backend.device_scope("mps"):
            out = convert_to_tensor(x)
            self.assertEqual(standardize_dtype(out.dtype), "float32")
            out = convert_to_tensor(x, dtype="float32")
            self.assertEqual(standardize_dtype(out.dtype), "float32")

    def test_convert_to_tensor_explicit_float64_raises_on_mps(self):
        x = np.ones(3, dtype=np.float64)
        with backend.device_scope("mps"):
            with self.assertRaises((RuntimeError, TypeError)):
                convert_to_tensor(x, dtype="float64")

    def test_convert_to_tensor_explicit_float32_no_downcast(self):
        x = np.ones(3, dtype=np.float64)
        with backend.device_scope("mps"):
            out = convert_to_tensor(x, dtype="float32")
        self.assertEqual(standardize_dtype(out.dtype), "float32")

    def test_convert_to_tensor_no_downcast_on_cpu(self):
        x = np.ones(3, dtype=np.float64)
        with backend.device_scope("cpu"):
            out = convert_to_tensor(x)
        self.assertEqual(standardize_dtype(out.dtype), "float64")


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
@pytest.mark.no_pytest_xdist
class TorchCoreDistributedTest(TorchDistributedTestMixin, testing.TestCase):
    def tearDown(self):
        super().tearDown()

        # Reset Keras-level distribution state after each test.
        set_distribution(None)
        torch_core.deactivate_dtensor_promotion()

    def test_keras_dtensor_promotion_mode(self):
        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        local_tensor = torch.ones((2, 2), device=device_type)
        dtensor = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Replicate()]
        )

        plain_tensor = torch.zeros((2, 2), device=device_type)

        with KerasDTensorPromotionMode():
            # Test simple addition
            result = dtensor + plain_tensor
            self.assertIsInstance(result, DTensor)
            self.assertTrue(
                torch.allclose(
                    result.to_local(),
                    torch.ones((2, 2), device=device_type),
                )
            )

            # Test in-place
            dtensor += plain_tensor
            self.assertIsInstance(dtensor, DTensor)

            # Test nested structures
            result = torch.addcmul(dtensor, plain_tensor, dtensor, value=0.5)
            self.assertIsInstance(result, DTensor)

    def test_activate_and_deactivate_dtensor_promotion(self):
        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        local_tensor = torch.ones((2, 2), device=device_type)
        dtensor = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Replicate()]
        )

        torch_core.activate_dtensor_promotion()

        plain_tensor = torch.zeros((2, 2), device=device_type)
        result = dtensor + plain_tensor
        self.assertIsInstance(result, DTensor)
        self.assertTrue(
            torch.allclose(
                result.to_local(), torch.ones((2, 2), device=device_type)
            )
        )

        torch_core.deactivate_dtensor_promotion()

        with self.assertRaisesRegex(
            RuntimeError, "mixed torch.Tensor and DTensor"
        ):
            dtensor + plain_tensor

    def _sharded_distribution(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["model"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )
        layout_map = LayoutMap(mesh)
        layout_map[".*"] = (None, "model")
        return ModelParallel(layout_map=layout_map)

    def test_dtensor_promotion_is_per_thread(self):
        dist = self._sharded_distribution()
        ones = np.ones((2, 2), dtype="float32")

        def worker(v):
            # The distribution is per thread, so the worker enters its own
            # scope while the main thread is still in its scope.
            with dist.scope():
                v.assign_add(ones)

        with dist.scope():
            v = Variable(np.zeros((2, 2), dtype="float32"))
            self.assertIsInstance(v.value.data, DTensor)
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                # `result()` re-raises any error from the worker thread.
                executor.submit(worker, v).result()
            # The worker's scope exit leaves this thread's promotion active.
            v.assign_add(ones)
        self.assertAllClose(v, 2 * ones)

        # Leaving the scope ends the promotion on this thread.
        with self.assertRaisesRegex(
            RuntimeError, "mixed torch.Tensor and DTensor"
        ):
            v.assign_add(ones)

    def test_clear_session_ends_dtensor_promotion(self):
        dist = self._sharded_distribution()
        ones = np.ones((2, 2), dtype="float32")

        with dist.scope():
            v = Variable(np.zeros((2, 2), dtype="float32"))
            v.assign_add(ones)
            # `clear_session()` resets the distribution, so the promotion
            # ends with it and does not stay after the scope.
            clear_session()
            with self.assertRaisesRegex(
                RuntimeError, "mixed torch.Tensor and DTensor"
            ):
                v.assign_add(ones)
        with self.assertRaisesRegex(
            RuntimeError, "mixed torch.Tensor and DTensor"
        ):
            v.assign_add(ones)
        self.assertAllClose(v, ones)

    def test_clear_session_on_other_thread_ends_dtensor_promotion(self):
        dist = self._sharded_distribution()
        ones = np.ones((2, 2), dtype="float32")
        with dist.scope():
            v = Variable(np.zeros((2, 2), dtype="float32"))
        entered = threading.Event()
        cleared = threading.Event()

        def worker():
            with dist.scope():
                entered.set()
                cleared.wait(timeout=60)
            # `clear_session()` on the main thread does not leave the
            # worker's mode on the worker's dispatch stack.
            with self.assertRaisesRegex(
                RuntimeError, "mixed torch.Tensor and DTensor"
            ):
                v.assign_add(ones)

        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(worker)
            entered.wait(timeout=60)
            clear_session()
            cleared.set()
            future.result()

    def test_convert_to_numpy_dtensor(self):
        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        dtensor = DTensor.from_local(
            torch.ones((2, 2), device=device_type),
            device_mesh=mesh,
            placements=[Replicate()],
        )

        nv = torch_core.convert_to_numpy(dtensor)
        self.assertIsInstance(nv, np.ndarray)
        self.assertEqual(nv.shape, (2, 2))
        self.assertTrue(np.allclose(nv, 1.0))

    def test_dtensor_variable_repr_leaves_out_value(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["x"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )

        layout_map = LayoutMap(mesh)
        layout_map[".*"] = ("x", None)
        dist = ModelParallel(layout_map=layout_map)

        set_distribution(dist)

        v = Variable(torch.ones((2, 2)))

        self.assertIsInstance(v.value.data, DTensor)
        self.assertNotIn("value=", repr(v))
        self.assertIn("shape=(2, 2)", repr(v))

    @parameterized.parameters(
        ("callable",),
        ("parameter",),
        ("tensor_with_grad",),
    )
    def test_variable_initialize_distributed(self, init_type):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["x"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )

        layout_map = LayoutMap(mesh)
        layout_map[".*"] = ("x", None)
        dist = ModelParallel(layout_map=layout_map)

        set_distribution(dist)

        if init_type == "callable":

            def initializer(shape, dtype):
                return torch.ones(shape)

            v = Variable(initializer, shape=(2, 2), dtype="float32")
        elif init_type == "parameter":
            v = Variable(torch.nn.Parameter(torch.ones((2, 2))))
        elif init_type == "tensor_with_grad":
            v = Variable(torch.ones((2, 2), requires_grad=True))

        self.assertIsInstance(v.value, torch.nn.Parameter)
        self.assertIsInstance(v.value.data, DTensor)
        self.assertEqual(v.value.device.type, mesh.backend_mesh.device_type)

    def test_variable_direct_assign(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["x"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )

        layout_map = LayoutMap(mesh)
        layout_map[".*"] = ("x", None)
        dist = ModelParallel(layout_map=layout_map)

        set_distribution(dist)

        v = Variable(torch.ones((2, 2)))

        self.assertIsInstance(v.value, torch.nn.Parameter)
        self.assertIsInstance(v.value.data, DTensor)

        new_val = torch.zeros((2, 2))
        v._direct_assign(new_val)

        self.assertIsInstance(v.value.data, DTensor)
        self.assertTrue(
            torch.allclose(
                v.value.data.to_local(),
                torch.zeros((2, 2), device=v.value.data.to_local().device),
            )
        )

    def test_data_parallel_scope_keeps_torch_compile(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )
        counter = CompileCounter()

        def add_one(x):
            return x + 1

        torch._dynamo.reset()
        with DataParallel(device_mesh=mesh).scope():
            result = torch.compile(add_one, backend=counter)(torch.zeros(2))

        # TorchDynamo skips every frame while a dispatch mode is active.
        self.assertEqual(counter.frame_count, 1)
        self.assertAllClose(result, np.ones(2))
