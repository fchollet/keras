try:
    # When using torch and tensorflow, torch needs to be imported first,
    # otherwise it will segfault upon import. This should force the torch
    # import to happen first for all tests.
    import torch  # noqa: F401
except ImportError:
    torch = None

import os  # noqa: E402

import pytest  # noqa: E402

from keras.src.backend import SUPPORTS_GRADIENT  # noqa: E402
from keras.src.backend.config import PLUGGABLE_BACKENDS
from keras.src.backend.config import backend  # noqa: E402
from keras.src.utils import module_utils
from keras.src.utils.module_utils import get_pluggable_backend_module


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )
    config.addinivalue_line(
        "markers",
        "multi_device: mark test for running with multiple devices only",
    )
    config.addinivalue_line(
        "markers",
        "no_pytest_xdist: mark test that cannot run under pytest-xdist "
        "workers (e.g. tests that bind to a fixed port)",
    )

    # Disable CUDA TF32 to get higher numerical accuracy for correctness tests.
    if backend() == "jax":
        import jax

        if jax.default_backend() == "gpu":
            jax.config.update("jax_default_matmul_precision", "float32")
    elif backend() == "tensorflow":
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            tf.config.experimental.enable_tensor_float_32_execution(False)
    elif backend() == "torch":
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False


def pytest_collection_modifyitems(config, items):
    has_multiple_devices = False

    backend_skipped_tests = set()
    if backend() in PLUGGABLE_BACKENDS:
        backend_module_file = get_pluggable_backend_module().__file__
        exclusions_path = os.path.join(
            # Remove `src/__init__.py`.
            os.path.dirname(os.path.dirname(backend_module_file)),
            "excluded_tests.txt",
        )
        # An installed backend package has no exclusion list.
        if os.path.exists(exclusions_path):
            backend_skipped_tests = read_test_list(exclusions_path)

    tf_free_skipped_tests = set()
    if not module_utils.tensorflow.available:
        tf_free_skipped_tests = read_test_list(
            os.path.join(
                os.path.dirname(__file__),
                ".github/workflows/config/tf_free_ignored_tests.txt",
            )
        )

    if backend() == "jax":
        import jax

        has_multiple_devices = jax.device_count() > 1
    elif backend() == "torch":
        from keras.src.backend.torch import distribution_lib

        has_multiple_devices = distribution_lib.get_device_count() > 1

    requires_trainable_backend = pytest.mark.skipif(
        not SUPPORTS_GRADIENT,
        reason="Trainer not implemented for this backend.",
    )
    requires_multiple_devices = (
        None
        if has_multiple_devices
        else pytest.mark.skip(reason="Requires multiple devices")
    )
    # Distributed tests (e.g. torch.distributed) cannot run under
    # pytest-xdist workers because process group init deadlocks when
    # multiple workers attempt to bind to the same port.
    is_xdist_worker = "PYTEST_XDIST_WORKER" in os.environ

    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        if requires_multiple_devices and "multi_device" in item.keywords:
            item.add_marker(requires_multiple_devices)
        if is_xdist_worker and "no_pytest_xdist" in item.keywords:
            item.add_marker(
                pytest.mark.skip(
                    reason="This test cannot run under pytest-xdist workers"
                )
            )

        # Skip concrete tests listed in the backend specific file.
        if item.nodeid in backend_skipped_tests:
            item.add_marker(
                skip_if_backend(
                    backend(),
                    f"Not supported operation by {backend()} backend",
                )
            )
        if item.nodeid in tf_free_skipped_tests:
            item.add_marker(pytest.mark.skip(reason="Requires TensorFlow"))


def read_test_list(path):
    with open(path, "r") as file:
        # Exclude empty lines and comments.
        return {
            stripped
            for line in file.readlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        }


def skip_if_backend(given_backend, reason):
    return pytest.mark.skipif(backend() == given_backend, reason=reason)
