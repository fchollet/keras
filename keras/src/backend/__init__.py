from keras.src.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

from keras.src.api_export import keras_export
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.keras_tensor import any_symbolic_tensors
from keras.src.backend.common.keras_tensor import is_keras_tensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.config import PLUGGABLE_BACKENDS
from keras.src.backend.config import epsilon
from keras.src.backend.config import floatx
from keras.src.backend.config import image_data_format
from keras.src.backend.config import set_epsilon
from keras.src.backend.config import set_floatx
from keras.src.backend.config import set_image_data_format
from keras.src.backend.config import standardize_data_format
from keras.src.utils.module_utils import get_pluggable_backend_module

# Import backend functions.
if backend() == "tensorflow":
    from keras.src.backend.tensorflow import *  # noqa: F403
elif backend() == "jax":
    from keras.src.backend.jax import *  # noqa: F403
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
elif backend() == "numpy":
    from keras.src.backend.numpy import *  # noqa: F403

    distribution_lib = None
else:
    backend_module = get_pluggable_backend_module("src")
    if hasattr(backend_module, "__all__"):
        backend_module_names = backend_module.__all__
    else:
        backend_module_names = [
            name for name in dir(backend_module) if not name.startswith("_")
        ]
    globals().update(
        {name: getattr(backend_module, name) for name in backend_module_names}
    )


Variable.__module__ = __name__  # noqa: F405
Variable = keras_export("keras.Variable")(Variable)  # noqa: F405


name_scope.__module__ = __name__  # noqa: F405
name_scope = keras_export("keras.name_scope")(name_scope)  # noqa: F405


@keras_export("keras.device")
def device(device_name):
    """Context manager for backend-agnostic device placement.

    Use this context manager to control on which device operations are performed
    and tensors are allocated. This works across all backends (TensorFlow, JAX,
    PyTorch). This is useful for memory management, data preprocessing, and
    multi-device setups.

    Args:
        device_name: String specifying the device in format
            `"device_type:device_index"`. For example: `"cpu:0"`, `"gpu:0"`,
            `"gpu:1"`. For the PyTorch backend, `"gpu"` is automatically
            converted to `"cuda"`.

    Example:

    Basic usage with CPU and GPU:

    ```python
    # Allocate tensors on CPU
    with keras.device("cpu:0"):
        cpu_tensor = keras.ops.ones((2, 2))

    # Allocate tensors on GPU (if available)
    with keras.device("gpu:0"):
        gpu_tensor = keras.ops.ones((2, 2))
    ```

    Practical example with CPU preprocessing and GPU training:

    ```python
    # Create dummy data and model
    x_raw = np.random.rand(128, 784)
    y_train = np.random.randint(0, 10, size=(128,))
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Preprocess data on CPU
    with keras.device("cpu:0"):
        x_processed = keras.ops.cast(x_raw, "float32")

    # Train on GPU (if available)
    with keras.device("gpu:0"):
        model.fit(x_processed, y_train, epochs=2)
    ```

    Use cases:

    - **Memory management**: Keep large tensors on CPU to save GPU memory
    - **Data preprocessing**: Process data on CPU before training on GPU
    - **GPU / TPU setups**: Control what runs on GPU / TPU vs CPU
    - **Multi-device setups**: Control which device receives which tensors

    Device naming conventions:

    - `"cpu:0"` - First CPU
    - `"gpu:0"` - First GPU (works across all backends)
    - `"gpu:1"` - Second GPU

    Note: For distributed training across multiple devices, see the
    [distributed training guides](https://keras.io/guides/distributed_training/).
    """
    return device_scope(device_name)  # noqa: F405
