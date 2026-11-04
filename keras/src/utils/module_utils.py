import importlib

from keras.src.backend.config import PLUGGABLE_BACKENDS
from keras.src.backend.config import backend as current_backend


def get_pluggable_backend_module(
    module_name=None, allow_missing=False, backend=None
):
    if not backend:
        backend = current_backend()
    if backend not in PLUGGABLE_BACKENDS:
        raise ValueError(f"Unsupported backend : {backend}")

    backend_module_name = (
        f"keras_{backend}.{module_name}" if module_name else f"keras_{backend}"
    )
    try:
        return importlib.import_module(backend_module_name)
    except (ImportError, AttributeError) as e:
        if allow_missing:
            return None
        raise ImportError(
            f"The {backend} backend is not installed. "
            f"You can install it via `pip install keras-{backend}`"
        ) from e


class LazyModule:
    def __init__(
        self,
        name,
        pip_name=None,
        import_error_msg=None,
        required_attr="__version__",
    ):
        self.name = name
        self.pip_name = pip_name or name
        self.import_error_msg = import_error_msg or (
            f"This requires the {self.name} module. "
            f"You can install it via `pip install {self.pip_name}`"
        )
        self.required_attr = required_attr
        self.module = None
        self._available = None

    @property
    def available(self):
        if self._available is None:
            try:
                self.initialize()
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _import(self, name):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise ImportError(self.import_error_msg)

    def _require(self, module, attr):
        if not hasattr(module, attr):
            raise ImportError(self.import_error_msg)

    def initialize(self):
        module = self._import(self.name)
        self._require(module, self.required_attr)
        self.module = module

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        return getattr(self.module, name)

    def __repr__(self):
        return f"LazyModule({self.name})"


class OrbaxLazyModule(LazyModule):
    def initialize(self):
        parent_module = self._import("orbax.checkpoint")
        self._require(parent_module, "v1")
        v1_module = parent_module.v1
        self._require(v1_module, self.required_attr)

        self.module = v1_module
        self.parent_module = parent_module

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        if name == "multihost":
            return self.parent_module.multihost
        return getattr(self.module, name)


tensorflow = LazyModule("tensorflow")
gfile = LazyModule(
    "tensorflow.io.gfile", pip_name="tensorflow", required_attr="GFile"
)
epath = LazyModule("etils.epath", pip_name="etils[epath]", required_attr="Path")
tensorflow_io = LazyModule("tensorflow_io")
scipy = LazyModule("scipy")
jax = LazyModule("jax")
h5py = LazyModule("h5py")
torch_xla = LazyModule(
    "torch_xla",
    import_error_msg=(
        "This requires the torch_xla module. You can install it via "
        "`pip install torch-xla`. Additionally, you may need to update "
        "LD_LIBRARY_PATH if necessary. Torch XLA builds a shared library, "
        "_XLAC.so, which needs to link to the version of Python it was built "
        "with. Use the following command to update LD_LIBRARY_PATH: "
        "`export LD_LIBRARY_PATH=<path to Python>/lib:$LD_LIBRARY_PATH`"
    ),
)
optree = LazyModule("optree")
dmtree = LazyModule("tree")
tf2onnx = LazyModule("tf2onnx")
jax2onnx = LazyModule("jax2onnx", required_attr="to_onnx")
grain = LazyModule("grain")
litert = LazyModule("ai_edge_litert")
ocp = OrbaxLazyModule(
    "orbax.checkpoint.v1",
    pip_name="orbax-checkpoint",
    import_error_msg=(
        "OrbaxCheckpoint requires the 'orbax-checkpoint' package. "
        "You can install it via pip install orbax-checkpoint"
    ),
    required_attr="training",
)
