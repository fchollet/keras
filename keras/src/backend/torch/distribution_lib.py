import os

import numpy as np
import torch
import torch.distributed
from torch.distributed import tensor as torch_distributed_tensor

from keras.src.backend.torch.ops import core as torch_core
from keras.src.backend.torch.ops.core import _parse_device_input
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device


def list_devices(device_type=None):
    """Return all available devices as a list of strings.

    Args:
        device_type: Optional string, one of "cpu", "gpu", "cuda", or "tpu".
            Defaults to the primary available device type.

    Returns:
        A list of strings like ["gpu:0", "gpu:1"] or ["cpu:0"].
    """
    resolved_device_type = _parse_device_input(
        device_type or get_device()
    ).split(":")[0]
    count = get_device_count(device_type)

    display_type = (
        "gpu" if resolved_device_type == "cuda" else resolved_device_type
    )

    return [f"{display_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Return the total number of devices for a given type.

    In a distributed setting, this returns the total number of processes
    managing that device type across the cluster.

    Args:
        device_type: Optional string, one of "cpu", "gpu", "cuda", or "tpu".

    Returns:
        An integer representing the device count.
    """
    device_type = device_type.lower() if device_type else None

    if torch.distributed.is_initialized() or "WORLD_SIZE" in os.environ:
        actual_device_type = _parse_device_input(get_device()).split(":")[0]

        if device_type in (None, "cpu", actual_device_type) or (
            device_type == "gpu" and actual_device_type == "cuda"
        ):
            return num_processes()

        return 0

    resolved_device_type = _parse_device_input(
        device_type or get_device()
    ).split(":")[0]

    if resolved_device_type == "cuda":
        return torch.cuda.device_count()

    if resolved_device_type == "mps":
        return 1

    if resolved_device_type == "xpu":
        return torch.xpu.device_count()

    if resolved_device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        if torch_xla.available:
            import torch_xla.core.xla_model as xm

            return xm.xla_device_count()

    return 1 if resolved_device_type == "cpu" else 0


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed process group.

    Args:
        job_addresses: Optional string, comma-separated list of host:port
            addresses. The first address is used as the MASTER_ADDR/MASTER_PORT.
        num_processes: Optional integer, the total number of processes
            (WORLD_SIZE).
        process_id: Optional integer, the rank of the current process.
    """
    if job_addresses:
        address = job_addresses.split(",")[0]

        if ":" in address:
            master_addr, master_port = address.split(":")
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", master_port)
        else:
            os.environ.setdefault("MASTER_ADDR", address)

    if num_processes is not None:
        os.environ.setdefault("WORLD_SIZE", str(num_processes))

    if process_id is not None:
        os.environ.setdefault("RANK", str(process_id))

    if not torch.distributed.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        resolved_device_type = _parse_device_input(get_device()).split(":")[0]
        if resolved_device_type == "cuda":
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        elif resolved_device_type == "xpu":
            torch.xpu.set_device(local_rank)
            backend = "ccl"
        elif resolved_device_type == "tpu":
            backend = "xla"
        else:
            backend = "gloo"

        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )


def activate_dtensor_promotion():
    torch_core.activate_dtensor_promotion()


def deactivate_dtensor_promotion():
    torch_core.deactivate_dtensor_promotion()


def num_processes():
    """Return the total number of processes in the distributed group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return int(os.environ.get("WORLD_SIZE", 1))


def process_id():
    """Return the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return int(os.environ.get("RANK", 0))


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to PyTorch backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `torch.distributed.DeviceMesh` instance.
    """
    devices = device_mesh.devices

    ranks = np.array(
        [int(d.split(":")[-1]) for d in devices.flatten()]
    ).reshape(devices.shape)

    first_device = (
        devices.flatten()[0].split(":")[0] if devices.size > 0 else "cpu"
    )

    resolved_device_type = _parse_device_input(
        first_device or get_device()
    ).split(":")[0]

    return torch.distributed.device_mesh.DeviceMesh(
        resolved_device_type,
        ranks,
        mesh_dim_names=tuple(device_mesh.axis_names),
    )


def _to_backend_device(device_name):
    """Convert a device name string to a torch.device object."""
    if isinstance(device_name, torch.device):
        return device_name

    name = str(device_name).lower()
    parts = name.split(":")

    device_type_str = parts[0]
    device_index = (
        parts[1] if len(parts) > 1 else os.environ.get("LOCAL_RANK", "0")
    )

    resolved_device_type = _parse_device_input(
        device_type_str or get_device()
    ).split(":")[0]

    if resolved_device_type == "cpu":
        return torch.device("cpu")

    return torch.device(f"{resolved_device_type}:{device_index}")


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to PyTorch DTensor placement spec."""
    if tensor_layout is None:
        return None

    from keras.src.distribution.distribution_lib import TensorLayout

    if not isinstance(tensor_layout, TensorLayout):
        # It's already a backend layout
        return tensor_layout

    keras_mesh = tensor_layout.device_mesh
    if keras_mesh is None:
        raise ValueError(
            "Cannot convert TensorLayout to PyTorch DTensor layout because "
            "the 'device_mesh' is not specified. Please ensure the layout "
            "has a valid 'device_mesh'."
        )
    torch_mesh = keras_mesh.backend_mesh

    mesh_axis_to_tensor_dim = {}
    valid_axis_names = set(keras_mesh.axis_names)
    if tensor_layout.axes is not None:
        for tensor_dim, axis_spec in enumerate(tensor_layout.axes):
            if axis_spec is None:
                continue
            axes = [axis_spec] if isinstance(axis_spec, str) else axis_spec
            for axis_name in axes:
                if axis_name is not None:
                    if axis_name not in valid_axis_names:
                        raise ValueError(
                            f"Invalid axis name '{axis_name}' in TensorLayout. "
                            f"Available mesh axes are: {keras_mesh.axis_names}"
                        )
                    mesh_axis_to_tensor_dim[axis_name] = tensor_dim

    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = mesh_axis_to_tensor_dim.get(mesh_dim_name)
        if shard_dim is not None:
            placements.append(torch_distributed_tensor.Shard(shard_dim))
        else:
            placements.append(torch_distributed_tensor.Replicate())

    return DTensorLayout(torch_mesh, tuple(placements))


def distribute_tensor(tensor, layout):
    """Scatters or replicates a tensor across devices according to the
    layout."""
    if layout is None:
        return tensor

    from keras.src.distribution.distribution_lib import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    if isinstance(tensor, torch_distributed_tensor.DTensor):
        if (
            tensor.device_mesh == layout.device_mesh
            and tensor.placements == layout.placements
        ):
            return tensor
        # `torch.distributed.tensor.distribute_tensor` rejects a DTensor with
        # a different layout, e.g. a `Partial` gradient or a value computed
        # from differently sharded operands.
        return tensor.redistribute(layout.device_mesh, layout.placements)

    if not isinstance(tensor, torch.Tensor):
        tensor = convert_to_tensor(tensor)

    return torch_distributed_tensor.distribute_tensor(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


def distribute_data_input(per_process_batch, layout):
    """Distribute a local data tensor according to a TensorLayout."""
    if layout is None:
        return per_process_batch

    from keras.src.distribution.distribution_lib import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    if isinstance(per_process_batch, torch_distributed_tensor.DTensor):
        if (
            per_process_batch.device_mesh == layout.device_mesh
            and per_process_batch.placements == layout.placements
        ):
            return per_process_batch

    if not isinstance(per_process_batch, torch.Tensor):
        per_process_batch = torch.as_tensor(per_process_batch, device="cpu")
    elif per_process_batch.device.type != "cpu":
        per_process_batch = per_process_batch.cpu()

    return torch_distributed_tensor.DTensor.from_local(
        per_process_batch,
        device_mesh=layout.device_mesh,
        placements=layout.placements,
        run_check=False,
    )


class DTensorLayout:
    """Wraps a torch DeviceMesh + placements for use as a backend layout."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements
