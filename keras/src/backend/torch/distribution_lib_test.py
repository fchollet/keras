import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized
from torch.distributed import tensor as torch_tensor
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.backend.torch.distributed_test_utils import (
    TorchDistributedTestMixin,
)
from keras.src.backend.torch.ops import core
from keras.src.backend.torch.ops.core import Variable
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel
from keras.src.distribution.distribution_lib import TensorLayout


@pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
@pytest.mark.no_pytest_xdist
class TorchDistributionLibTest(TorchDistributedTestMixin, testing.TestCase):
    def set_env(self, key, value):
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(
            lambda: (
                os.environ.update({key: old})
                if old is not None
                else os.environ.pop(key, None)
            )
        )

    def _get_mesh_devices(self):
        device_type = core._parse_device_input(core.get_device()).split(":")[0]
        return np.array([f"{device_type}:0"])

    @parameterized.parameters(
        ({}, False, None),
        ({"WORLD_SIZE": "4"}, False, None),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29502",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            None,
        ),
        ({}, False, "cuda"),
        ({}, False, "gpu"),
    )
    def test_get_device_count(self, env, init, device_type):
        for k, v in env.items():
            self.set_env(k, v)
        if init and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        res = distribution_lib.get_device_count(device_type)

        if torch.distributed.is_initialized() or "WORLD_SIZE" in os.environ:
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            actual_device_type = core._parse_device_input(
                core.get_device()
            ).split(":")[0]
            if device_type in (None, "cpu", actual_device_type) or (
                device_type == "gpu" and actual_device_type == "cuda"
            ):
                self.assertEqual(res, world_size)
            else:
                self.assertEqual(res, 0)
        else:
            resolved_device_type = core._parse_device_input(
                device_type or core.get_device()
            ).split(":")[0]
            if resolved_device_type == "cuda":
                self.assertEqual(res, torch.cuda.device_count())
            elif resolved_device_type in ("cpu", "mps"):
                self.assertEqual(res, 1)
            elif resolved_device_type == "xpu":
                self.assertEqual(res, torch.xpu.device_count())
            else:
                self.assertEqual(res, 0)

    @parameterized.parameters(
        (None, {}, False, True),
        ("cpu", {}, False, False),
        (
            "gpu",
            {"WORLD_SIZE": "4", "KERAS_TORCH_DEVICE": "gpu"},
            False,
            False,
        ),
    )
    def test_list_devices(self, device_type, env, init, default):
        for k, v in env.items():
            self.set_env(k, v)
        if init and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        if default:
            devices = distribution_lib.list_devices()
            self.assertTrue(
                any(
                    devices[0].startswith(s)
                    for s in ["gpu:", "cpu:", "mps:", "xpu:"]
                )
            )
        else:
            devices = distribution_lib.list_devices(device_type)
            resolved_device_type = core._parse_device_input(
                device_type or core.get_device()
            ).split(":")[0]
            display_type = (
                "gpu"
                if resolved_device_type == "cuda"
                else resolved_device_type
            )
            count = distribution_lib.get_device_count(device_type)
            expected = [f"{display_type}:{i}" for i in range(count)]
            self.assertEqual(devices, expected)

    @parameterized.parameters(
        ({}, False, 1),
        ({"WORLD_SIZE": "4"}, False, 4),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29504",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            1,
        ),
    )
    def test_num_processes_and_id(self, env, init, expected):
        for k, v in env.items():
            self.set_env(k, v)
        if init and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        if torch.distributed.is_initialized():
            expected_num = torch.distributed.get_world_size()
        else:
            expected_num = expected
        self.assertEqual(distribution_lib.num_processes(), expected_num)
        self.assertEqual(distribution_lib.process_id(), int(env.get("RANK", 0)))

    @parameterized.parameters(
        (
            "127.0.0.1:29506",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": None},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29506"},
        ),
        (
            "127.0.0.1",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": "29507"},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29507"},
        ),
        (
            None,
            1,
            0,
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29508"},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29508"},
        ),
    )
    def test_initialize(self, addr, nproc, pid, initial, expected):
        if torch.distributed.is_initialized():
            self.skipTest("torch.distributed already initialized")

        for k, v in initial.items():
            self.set_env(k, v)

        distribution_lib.initialize(addr, nproc, pid)
        self.addCleanup(
            lambda: (
                torch.distributed.destroy_process_group()
                if torch.distributed.is_initialized()
                else None
            )
        )
        self.assertTrue(torch.distributed.is_initialized())

        for k, v in expected.items():
            self.assertEqual(os.environ.get(k), v)

    @parameterized.parameters(
        ("cpu", "cpu"),
        ("gpu", "cuda"),
        (None, None),
    )
    def test_get_device_type(self, k_dev, expected):
        if k_dev:
            self.set_env("KERAS_TORCH_DEVICE", k_dev)

        res = core._parse_device_input(core.get_device()).split(":")[0]
        self.assertEqual(res, core.get_device().split(":")[0])

    @parameterized.parameters(
        ("cpu", {}, "cpu", None),
        ("gpu", {}, "cuda", None),
        (torch.device("cuda:0"), {}, "cuda", 0),
        ("cuda", {"LOCAL_RANK": "2"}, "cuda", 2),
        ("cpu:0", {}, "cpu", None),
        (torch.device("cpu"), {}, "cpu", None),
    )
    def test_to_backend_device(self, inp, env, etype, eidx):
        for k, v in env.items():
            self.set_env(k, v)

        if etype == "cuda" and not torch.cuda.is_available():
            self.skipTest("No CUDA")

        dev = distribution_lib._to_backend_device(inp)
        self.assertEqual(dev.type, etype)
        if eidx is not None:
            self.assertEqual(dev.index, eidx)

    def test_to_backend_mesh(self):
        device_type = core._parse_device_input(core.get_device()).split(":")[0]
        devs = np.array([f"{device_type}:0"]).reshape(1)

        mesh = DeviceMesh(shape=(1,), axis_names=["x"], devices=devs)
        backend_mesh = distribution_lib._to_backend_mesh(mesh)

        self.assertIsInstance(backend_mesh, TorchDeviceMesh)
        self.assertEqual(backend_mesh.device_type, device_type)
        self.assertEqual(backend_mesh.mesh_dim_names, ("x",))

    @parameterized.parameters(
        (("x", None), torch_tensor.Shard),
        ((None, None), torch_tensor.Replicate),
        (None, None),
    )
    def test_to_backend_layout(self, axes, expected_placement_type):
        if axes is None:
            self.assertIsNone(distribution_lib._to_backend_layout(None))
            return

        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )

        layout = TensorLayout(axes=axes, device_mesh=mesh)
        backend_layout = distribution_lib._to_backend_layout(layout)
        self.assertEqual(len(backend_layout.placements), 1)
        self.assertIsInstance(
            backend_layout.placements[0], expected_placement_type
        )
        if expected_placement_type == torch_tensor.Shard:
            self.assertEqual(backend_layout.placements[0].dim, 0)

    def test_to_backend_layout_reuses_backend_mesh(self):
        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )
        sharded = distribution_lib._to_backend_layout(
            TensorLayout(axes=("x", None), device_mesh=mesh)
        )
        replicated = distribution_lib._to_backend_layout(
            TensorLayout(axes=(None, None), device_mesh=mesh)
        )
        # Building a torch mesh can create process groups, so every layout
        # of a Keras mesh shares one torch mesh.
        self.assertIs(sharded.device_mesh, mesh.backend_mesh)
        self.assertIs(replicated.device_mesh, mesh.backend_mesh)

    @parameterized.parameters(
        (
            None,
            "Cannot convert TensorLayout to PyTorch DTensor layout because "
            "the 'device_mesh' is not specified. Please ensure the layout "
            "has a valid 'device_mesh'.",
        ),
        ("invalid", "Invalid axis name 'invalid'"),
    )
    def test_to_backend_layout_errors(self, axis_name, error_msg):
        if axis_name is None:
            layout = TensorLayout(axes=("x", None), device_mesh=None)
        else:
            mesh = DeviceMesh(
                shape=(1,),
                axis_names=["data"],
                devices=self._get_mesh_devices(),
            )

            class BypassLayout(TensorLayout):
                def __init__(self, axes, mesh):
                    self._axes = axes
                    self._device_mesh = mesh

                def _validate_axes(self):
                    pass

            layout = BypassLayout(axes=(axis_name,), mesh=mesh)

        with self.assertRaisesRegex(ValueError, error_msg):
            distribution_lib._to_backend_layout(layout)

    @parameterized.parameters(
        ("tensor", False, False),
        ("variable", False, False),
        ("numpy", False, False),
        ("tensor", True, False),
        ("tensor", False, True),
    )
    def test_distribute_tensor(
        self, input_type, layout_is_none, input_is_dtensor
    ):
        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )
        layout = TensorLayout(axes=("x", None), device_mesh=mesh)

        if input_is_dtensor:
            tensor = distribution_lib.distribute_tensor(
                torch.ones((2, 2)), layout
            )
        elif input_type == "tensor":
            tensor = torch.ones((2, 2))
        elif input_type == "variable":
            tensor = Variable(torch.ones((2, 2)))
        elif input_type == "numpy":
            tensor = np.ones((2, 2), dtype="float32")

        actual_layout = None if layout_is_none else layout
        dtensor = distribution_lib.distribute_tensor(tensor, actual_layout)

        if layout_is_none or input_is_dtensor:
            self.assertIs(dtensor, tensor)
        else:
            self.assertIsInstance(dtensor, torch_tensor.DTensor)

    @parameterized.named_parameters(
        ("replicate", torch_tensor.Replicate()),
        ("partial", torch_tensor.Partial()),
    )
    def test_distribute_tensor_redistributes_dtensor(self, placement):
        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )
        layout = TensorLayout(axes=("x", None), device_mesh=mesh)
        source = torch.ones((2, 2), requires_grad=True)
        tensor = torch_tensor.DTensor.from_local(
            source, layout.backend_layout.device_mesh, [placement]
        )

        dtensor = distribution_lib.distribute_tensor(tensor, layout)

        self.assertEqual(dtensor.placements, (torch_tensor.Shard(0),))
        self.assertAllClose(dtensor.full_tensor(), np.ones((2, 2)))
        # Layer outputs are relaid out in the forward pass, so gradients
        # must flow through.
        dtensor.full_tensor().sum().backward()
        self.assertAllClose(source.grad, np.ones((2, 2)))

    @parameterized.named_parameters(
        ("replicated_dtensor", torch_tensor.Replicate(), False),
        ("partial_dtensor", torch_tensor.Partial(), False),
        ("dtensor_with_grad_fn", torch_tensor.Replicate(), True),
        ("tensor_with_grad_fn", None, True),
    )
    def test_assign_to_sharded_variable(self, placement, requires_grad):
        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )
        layout = TensorLayout(axes=("x", None), device_mesh=mesh)
        distribution = ModelParallel(layout_map=LayoutMap(mesh))
        with distribution.scope():
            variable = Variable(
                np.zeros((2, 2), dtype="float32"), layout=layout
            )
            value = torch.ones((2, 2), requires_grad=requires_grad)
            if placement is not None:
                value = torch_tensor.DTensor.from_local(
                    value, variable.value.device_mesh, [placement]
                )
            # `value * 2` has a `grad_fn` when `value` requires grad, like
            # values computed in a forward pass.
            variable.assign(value * 2)

        self.assertTrue(variable.value.is_leaf)
        self.assertEqual(variable.value.placements, (torch_tensor.Shard(0),))
        self.assertAllClose(variable.value.full_tensor(), np.full((2, 2), 2.0))

    @parameterized.parameters(
        ("tensor", False),
        ("tensor", True),
        ("dtensor", False),
        ("numpy", False),
    )
    def test_distribute_data_input(self, input_type, layout_is_none):
        mesh = DeviceMesh(
            shape=(1,), axis_names=["x"], devices=self._get_mesh_devices()
        )
        layout = TensorLayout(axes=("x", None), device_mesh=mesh)

        if input_type == "tensor":
            inp = torch.ones((2, 2))
        elif input_type == "dtensor":
            inp = distribution_lib.distribute_data_input(
                torch.ones((2, 2)), layout
            )
        elif input_type == "numpy":
            inp = np.ones((2, 2), dtype="float32")

        actual_layout = None if layout_is_none else layout
        res = distribution_lib.distribute_data_input(inp, actual_layout)

        if layout_is_none:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, torch_tensor.DTensor)
        else:
            self.assertIsInstance(res, torch_tensor.DTensor)
            self.assertEqual(res.shape, (2, 2))
            if input_type == "dtensor":
                self.assertIs(res, inp)
