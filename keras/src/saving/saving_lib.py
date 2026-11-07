"""Python-based idempotent model-saving functionality."""

import datetime
import io
import json
import math
import os
import pathlib
import shutil
import tempfile
import warnings
import zipfile

import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import dtype_utils
from keras.src.utils import file_utils
from keras.src.utils import io_utils
from keras.src.utils import naming
from keras.src.utils import plot_model
from keras.src.utils.model_visualization import check_pydot
from keras.src.utils.module_utils import h5py
from keras.src.utils.summary_utils import readable_memory_size
from keras.src.utils.summary_utils import weight_memory_size
from keras.src.version import __version__ as keras_version

try:
    import psutil
except ImportError:
    psutil = None
try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None


_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"  # Will become e.g. "model.weights.h5"
_VARS_FNAME_H5 = f"{_VARS_FNAME}.h5"
_ASSETS_DIRNAME = "assets"
_MEMORY_UPPER_BOUND = 0.5  # 50%


_MODEL_CARD_TEMPLATE = """
---
library_name: keras
---

This model has been uploaded using the Keras library and can be used with JAX,
TensorFlow, and PyTorch backends.

This model card has been generated automatically and should be completed by the
model author.
See [Model Cards documentation](https://huggingface.co/docs/hub/model-cards) for
more information.

For more details about the model architecture, check out
[config.json](./config.json)."""


def save_model(model, filepath, zipped=True):
    """Save a zip-archive representing a Keras model to the given file or path.

    The zip-based archive contains the following structure:

    - JSON-based configuration file (config.json): Records of model, layer, and
        other saveables' configuration.
    - H5-based saveable state file (model.weights.h5), with groups for layers
        and other saveables.
    - Metadata file.

    The states of Keras saveables (layers, optimizers, loss, and metrics) are
    automatically saved as long as they can be discovered through the attributes
    returned by `dir(Model)`. Typically, the state includes the variables
    associated with the saveable, but some specially purposed layers may
    contain more such as the vocabularies stored in the hashmaps. The saveables
    define how their states are saved by exposing `save_state()` and
    `load_state()` APIs.

    For the case of layer states, the variables will be visited as long as
    they are either 1) referenced via layer attributes, or 2) referenced via a
    container (list, tuple, or dict), and the container is referenced via a
    layer attribute.
    """
    if h5py is None:
        raise ImportError("h5py must be installed in order to save a model.")

    if not model.built:
        warnings.warn(
            "You are saving a model that has not yet been built. "
            "It might not contain any weights yet. "
            "Consider building the model first by calling it "
            "on some data.",
            stacklevel=2,
        )

    if isinstance(filepath, io.IOBase):
        _save_model_to_fileobj(model, filepath)
        return

    filepath = str(filepath)
    is_hf = filepath.startswith("hf://")
    if zipped and not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid `filepath` argument: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    if not zipped and filepath.endswith(".keras"):
        raise ValueError(
            "When using `zipped=False`, the `filepath` argument should not "
            f"end in `.keras`. Received: filepath={filepath}"
        )
    if zipped and is_hf:
        raise ValueError(
            "When saving to the Hugging Face Hub, you should not save the "
            f"model as zipped. Received: filepath={filepath}, zipped={zipped}"
        )
    if is_hf:
        _upload_model_to_hf(model, filepath)
    elif not zipped:
        _save_model_to_dir(model, filepath)
    else:
        if file_utils.is_remote_path(filepath):
            # Remote path. Zip to local memory byte io and copy to remote
            zip_filepath = io.BytesIO()
            _save_model_to_fileobj(model, zip_filepath)
            with file_utils.File(filepath, "wb") as f:
                f.write(zip_filepath.getvalue())
        else:
            with open(filepath, "wb") as f:
                _save_model_to_fileobj(model, f)


def _serialize_model_as_json(model):
    with ObjectSharingScope():
        serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict)
    metadata_json = json.dumps(
        {
            "keras_version": keras_version,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
    )
    return config_json, metadata_json


def _save_model_to_dir(model, dirpath):
    if not file_utils.exists(dirpath):
        file_utils.makedirs(dirpath)
    config_json, metadata_json = _serialize_model_as_json(model)
    with open(file_utils.join(dirpath, _METADATA_FILENAME), "w") as f:
        f.write(metadata_json)
    with open(file_utils.join(dirpath, _CONFIG_FILENAME), "w") as f:
        f.write(config_json)
    weights_filepath = file_utils.join(dirpath, _VARS_FNAME_H5)
    assert_dirpath = file_utils.join(dirpath, _ASSETS_DIRNAME)
    weights_store = None
    asset_store = None
    try:
        weights_store = H5IOStore(weights_filepath, mode="w")
        asset_store = DiskIOStore(assert_dirpath, mode="w")
        _save_state(
            model,
            weights_store=weights_store,
            assets_store=asset_store,
            inner_path="",
            visited_saveables=set(),
        )
    finally:
        if weights_store is not None:
            weights_store.close()
        if asset_store is not None:
            asset_store.close()


def _save_model_to_fileobj(model, fileobj):
    config_json, metadata_json = _serialize_model_as_json(model)

    with zipfile.ZipFile(fileobj, "w") as zf:
        with zf.open(_METADATA_FILENAME, "w") as f:
            f.write(metadata_json.encode())
        with zf.open(_CONFIG_FILENAME, "w") as f:
            f.write(config_json.encode())

        weights_file_path = None
        weights_store = None
        asset_store = None
        write_zf = False
        try:
            try:
                if is_memory_sufficient(model):
                    # Load the model weights into memory before writing
                    # .keras if the system memory is sufficient.
                    weights_store = H5IOStore(
                        _VARS_FNAME_H5, archive=zf, mode="w"
                    )
                else:
                    # Try opening the .h5 file, then writing it to `zf` at
                    # the end of the function call. This is more memory
                    # efficient than writing the weights into memory first.
                    working_dir = pathlib.Path(fileobj.name).parent
                    weights_file_path = tempfile.NamedTemporaryFile(
                        dir=working_dir
                    )
                    weights_store = H5IOStore(weights_file_path.name, mode="w")
                    write_zf = True
            except:
                # If we can't use the local disk for any reason, write the
                # weights into memory first, which consumes more memory.
                weights_store = H5IOStore(_VARS_FNAME_H5, archive=zf, mode="w")

            asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode="w")

            _save_state(
                model,
                weights_store=weights_store,
                assets_store=asset_store,
                inner_path="",
                visited_saveables=set(),
            )
        except:
            # Skip the final `zf.write` if any exception is raised
            write_zf = False
            if weights_store:
                weights_store.archive = None
            raise
        finally:
            if weights_store:
                weights_store.close()
            if asset_store:
                asset_store.close()
            if write_zf and weights_file_path:
                zf.write(weights_file_path.name, _VARS_FNAME_H5)
            if weights_file_path:
                weights_file_path.close()


def _upload_model_to_hf(model, hf_path):
    if huggingface_hub is None:
        raise ImportError(
            "To save models to the Hugging Face Hub, "
            "you must install the `huggingface_hub` package."
        )

    original_hf_path = hf_path
    if hf_path.startswith("hf://"):
        hf_path = hf_path[5:]
    if hf_path.count("/") > 1:
        raise ValueError(
            "Invalid `hf_path` argument: expected `namespace/model_name`"
            f" format. Received: hf_path={original_hf_path}"
        )

    api = huggingface_hub.HfApi(
        library_name="keras", library_version=keras_version
    )
    repo_url = api.create_repo(hf_path, exist_ok=True)
    repo_id = repo_url.repo_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_model_to_dir(model, tmp_dir)

        model_card = _MODEL_CARD_TEMPLATE

        if check_pydot():
            plot_path = file_utils.join(tmp_dir, "assets", "summary_plot.png")
            plot_model(
                model,
                to_file=plot_path,
                show_layer_names=True,
                show_shapes=True,
                show_dtype=True,
            )
            if len(model.layers) <= 10:
                model_card += "\n\n![](./assets/summary_plot.png)"
            else:
                model_card += (
                    "A plot of the model can be found "
                    "[here](./assets/summary_plot.png)."
                )

        with open(file_utils.join(tmp_dir, "README.md"), "w") as f:
            f.write(model_card)

        api.upload_folder(
            repo_id=repo_id,
            folder_path=tmp_dir,
            commit_message="Save model using Keras.",
        )
        io_utils.print_msg(
            f"Model saved to the Hugging Face Hub: {repo_url}\n"
            "To load back the model, use "
            f"`keras.saving.load_model('hf://{repo_id}')`"
        )


def load_model(filepath, custom_objects=None, compile=True, safe_mode=True):
    """Load a zip archive representing a Keras model."""
    if isinstance(filepath, io.IOBase):
        return _load_model_from_fileobj(
            filepath, custom_objects, compile, safe_mode
        )
    elif str(filepath).startswith("hf://"):
        if huggingface_hub is None:
            raise ImportError(
                "To load models from the Hugging Face Hub, "
                "you must install the `huggingface_hub` package."
            )

        repo_id = filepath[5:]
        folder_path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            library_name="keras",
            library_version=keras_version,
        )
        return _load_model_from_dir(
            folder_path, custom_objects, compile, safe_mode
        )
    else:
        filepath = str(filepath)
        if not filepath.endswith(".keras"):
            is_keras_dir = file_utils.isdir(filepath) and file_utils.exists(
                file_utils.join(filepath, "config.json")
            )
            if is_keras_dir:
                return _load_model_from_dir(
                    filepath, custom_objects, compile, safe_mode
                )
            raise ValueError(
                "Invalid filename: expected a `.keras` extension. "
                f"Received: filepath={filepath}"
            )
        with file_utils.File(filepath, "rb") as f:
            return _load_model_from_fileobj(
                f, custom_objects, compile, safe_mode
            )


def _load_model_from_dir(dirpath, custom_objects, compile, safe_mode):
    if not file_utils.exists(dirpath):
        raise ValueError(f"Directory doesn't exist: {dirpath}")
    if not file_utils.isdir(dirpath):
        raise ValueError(f"Path isn't a directory: {dirpath}")

    with open(file_utils.join(dirpath, _CONFIG_FILENAME), "r") as f:
        config_json = f.read()
    model = _model_from_config(config_json, custom_objects, compile, safe_mode)

    all_filenames = file_utils.listdir(dirpath)
    weights_store = None
    asset_store = None
    try:
        if _VARS_FNAME_H5 in all_filenames:
            weights_file_path = file_utils.join(dirpath, _VARS_FNAME_H5)
            weights_store = H5IOStore(weights_file_path, mode="r")
        else:
            raise ValueError(f"Expected a {_VARS_FNAME_H5} file.")
        if len(all_filenames) > 3:
            asset_store = DiskIOStore(
                file_utils.join(dirpath, _ASSETS_DIRNAME), mode="r"
            )

        else:
            asset_store = None

        failed_saveables = set()
        error_msgs = {}
        _load_state(
            model,
            weights_store=weights_store,
            assets_store=asset_store,
            inner_path="",
            visited_saveables=set(),
            failed_saveables=failed_saveables,
            error_msgs=error_msgs,
        )

    finally:
        if weights_store is not None:
            weights_store.close()
        if asset_store:
            asset_store.close()

    if failed_saveables:
        _raise_loading_failure(error_msgs)
    return model


def _model_from_config(config_json, custom_objects, compile, safe_mode):
    # Note: we should NOT use a custom JSON decoder. Anything that
    # needs custom decoding must be handled in deserialize_keras_object.
    config_dict = json.loads(config_json)
    if not compile:
        # Disable compilation
        config_dict["compile_config"] = None
    # Construct the model from the configuration file in the archive.
    with ObjectSharingScope():
        model = deserialize_keras_object(
            config_dict, custom_objects, safe_mode=safe_mode
        )
    return model


# Guard against ZIP "decompression bomb" archive members (CWE-409): a member
# can store almost nothing on disk yet declare an enormous uncompressed size,
# forcing a huge allocation when it is read into memory. Keras writes archive
# members uncompressed (stored, ratio ~1:1) and DEFLATE cannot exceed ~1032:1,
# so a member that both exceeds the floor and expands to more than
# `_ZIP_MEMBER_MAX_EXPANSION`x its on-disk size is a bomb, not a genuine
# artifact (this leaves ample headroom for an incidentally recompressed file).
# The 4 GiB floor matches the HDF5 dataset guard for CVE-2026-0897.
_ZIP_MEMBER_BOMB_FLOOR_BYTES = 1 << 32  # 4 GiB
_ZIP_MEMBER_MAX_EXPANSION = 100
# Floor for the *cumulative* (whole-archive) guard below. A much lower floor
# than the per-member one lets the cumulative guard also catch a single member
# that declares just under the per-member floor, as well as the several
# independent member reads on the load path that each stay under it.
_ZIP_CUMULATIVE_BOMB_FLOOR_BYTES = 1 << 26  # 64 MiB


def _reject_zip_bomb(archive, name):
    """Raise if a ZIP member is a decompression bomb (see CWE-409)."""
    info = archive.getinfo(name)
    if (
        info.file_size > _ZIP_MEMBER_BOMB_FLOOR_BYTES
        and info.file_size > _ZIP_MEMBER_MAX_EXPANSION * info.compress_size
    ):
        raise ValueError(
            f"Not allowed: ZIP member '{name}' declares "
            f"{readable_memory_size(info.file_size)} but only "
            f"{readable_memory_size(info.compress_size)} are stored on disk; "
            "refusing to load a potential decompression bomb."
        )


def _reject_zip_archive_bomb(archive):
    """Check ZIP directory metadata without opening or decompressing members.

    Sum only members exceeding the expansion ratio, so ordinary weights cannot
    hide highly compressed assets. The cumulative floor also bounds multiple
    small members that individually stay below it. This check covers both
    in-memory reads and asset extraction.
    """
    if not isinstance(archive, zipfile.ZipFile):
        return
    total_declared = 0
    total_stored = 0
    for info in archive.infolist():
        if info.file_size > _ZIP_MEMBER_MAX_EXPANSION * info.compress_size:
            total_declared += info.file_size
            total_stored += info.compress_size
    if total_declared > _ZIP_CUMULATIVE_BOMB_FLOOR_BYTES:
        raise ValueError(
            "Not allowed: compressed archive members declare "
            f"{readable_memory_size(total_declared)} but only "
            f"{readable_memory_size(total_stored)} are stored on disk; "
            "refusing to load a potential decompression bomb."
        )


def _safe_zip_read(archive, name):
    """Read a ZIP member into memory, rejecting bombs (see CWE-409)."""
    _reject_zip_bomb(archive, name)
    with archive.open(name, "r") as f:
        return f.read()


def _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode):
    with zipfile.ZipFile(fileobj, "r") as zf:
        _reject_zip_archive_bomb(zf)
        config_json = _safe_zip_read(zf, _CONFIG_FILENAME)

        model = _model_from_config(
            config_json, custom_objects, compile, safe_mode
        )

        all_filenames = zf.namelist()
        extract_dir = None
        weights_store = None
        asset_store = None
        try:
            if _VARS_FNAME_H5 in all_filenames:
                # Reject a decompression-bomb weights member up front, outside
                # the try/except below: the bare `except` falls back to reading
                # the weights on the fly, so a check inside it would be
                # swallowed and the bomb still read. Checking here covers the
                # in-memory, extract-to-disk, and on-the-fly paths at once.
                _reject_zip_bomb(zf, _VARS_FNAME_H5)
                try:
                    if is_memory_sufficient(model):
                        # Load the entire file into memory if the system memory
                        # is sufficient.
                        io_file = io.BytesIO(
                            zf.open(_VARS_FNAME_H5, "r").read()
                        )
                        weights_store = H5IOStore(io_file, mode="r")
                    else:
                        # Try extracting the model.weights.h5 file, and then
                        # loading it using using h5py. This is significantly
                        # faster than reading from the zip archive on the fly.
                        extract_dir = tempfile.TemporaryDirectory(
                            dir=pathlib.Path(fileobj.name).parent
                        )
                        zf.extract(_VARS_FNAME_H5, extract_dir.name)
                        weights_store = H5IOStore(
                            pathlib.Path(extract_dir.name, _VARS_FNAME_H5),
                            mode="r",
                        )
                except:
                    # If we can't use the local disk for any reason, read the
                    # weights from the zip archive on the fly, which is less
                    # efficient.
                    weights_store = H5IOStore(_VARS_FNAME_H5, zf, mode="r")
            else:
                raise ValueError(f"Expected a {_VARS_FNAME_H5} file.")

            if len(all_filenames) > 3:
                asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode="r")

            failed_saveables = set()
            error_msgs = {}
            _load_state(
                model,
                weights_store=weights_store,
                assets_store=asset_store,
                inner_path="",
                visited_saveables=set(),
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )
        finally:
            if weights_store:
                weights_store.close()
            if asset_store:
                asset_store.close()
            if extract_dir:
                extract_dir.cleanup()

        if failed_saveables:
            _raise_loading_failure(error_msgs)
    return model


def save_weights_only(
    model, filepath, max_shard_size=None, objects_to_skip=None
):
    """Save only the weights of a model to a target filepath.

    Supports both `.weights.h5` and `.keras`.
    """
    if not model.built:
        raise ValueError(
            "You are saving a model that has not yet been built. "
            "Try building the model first by calling it on some data or "
            "by using `build()`."
        )

    filepath_str = str(filepath)
    tmp_dir = None
    remote_filepath = None
    if max_shard_size is None and not filepath_str.endswith(".weights.h5"):
        raise ValueError(
            "The filename must end in `.weights.h5`. "
            f"Received: filepath={filepath_str}"
        )
    elif max_shard_size is not None and not filepath_str.endswith(
        ("weights.h5", "weights.json")
    ):
        raise ValueError(
            "The filename must end in `.weights.json` when `max_shard_size` is "
            f"specified. Received: filepath={filepath_str}"
        )
    try:
        if file_utils.is_remote_path(filepath):
            tmp_dir = get_temp_dir()
            local_filepath = os.path.join(tmp_dir, os.path.basename(filepath))
            remote_filepath = filepath
            filepath = local_filepath

        if max_shard_size is not None:
            weights_store = ShardedH5IOStore(filepath, max_shard_size, mode="w")
        else:
            weights_store = H5IOStore(filepath, mode="w")
        if objects_to_skip is not None:
            visited_saveables = set(id(o) for o in objects_to_skip)
        else:
            visited_saveables = set()
        _save_state(
            model,
            weights_store=weights_store,
            assets_store=None,
            inner_path="",
            visited_saveables=visited_saveables,
        )
        weights_store.close()
    finally:
        if tmp_dir is not None:
            file_utils.copy(filepath, remote_filepath)
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_weights_only(
    model, filepath, skip_mismatch=False, objects_to_skip=None
):
    """Load the weights of a model from a filepath (.keras or .weights.h5).

    Note: only supports h5 for now.
    """
    if not model.built:
        raise ValueError(
            "You are loading weights into a model that has not yet been built. "
            "Try building the model first by calling it on some data or "
            "by using `build()`."
        )

    archive = None
    weights_store = None
    tmp_dir = None
    filepath_str = str(filepath)

    try:
        if file_utils.is_remote_path(filepath_str):
            tmp_dir = get_temp_dir()
            local_filepath = os.path.join(
                tmp_dir, os.path.basename(filepath_str)
            )
            file_utils.copy(filepath_str, local_filepath)
            filepath_str = filepath = local_filepath

        if filepath_str.endswith("weights.h5"):
            weights_store = H5IOStore(filepath, mode="r")
        elif filepath_str.endswith("weights.json"):
            weights_store = ShardedH5IOStore(filepath, mode="r")
        elif filepath_str.endswith(".keras"):
            archive = zipfile.ZipFile(filepath, "r")
            _reject_zip_archive_bomb(archive)
            weights_store = H5IOStore(_VARS_FNAME_H5, archive=archive, mode="r")

        failed_saveables = set()
        if objects_to_skip is not None:
            visited_saveables = set(id(o) for o in objects_to_skip)
        else:
            visited_saveables = set()
        error_msgs = {}
        _load_state(
            model,
            weights_store=weights_store,
            assets_store=None,
            inner_path="",
            skip_mismatch=skip_mismatch,
            visited_saveables=visited_saveables,
            failed_saveables=failed_saveables,
            error_msgs=error_msgs,
        )
        if failed_saveables:
            _raise_loading_failure(error_msgs, warn_only=skip_mismatch)
    finally:
        if weights_store is not None:
            weights_store.close()
        if archive is not None:
            archive.close()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _raise_loading_failure(error_msgs, warn_only=False):
    first_key = list(error_msgs.keys())[0]
    ex_saveable, ex_error = error_msgs[first_key]
    msg = (
        f"A total of {len(error_msgs)} objects could not "
        "be loaded. Example error message for "
        f"object {ex_saveable}:\n\n"
        f"{ex_error}\n\n"
        "List of objects that could not be loaded:\n"
        f"{[x[0] for x in error_msgs.values()]}"
    )
    if warn_only:
        warnings.warn(msg)
    else:
        raise ValueError(msg)


def _write_to_zip_recursively(zipfile_to_save, system_path, zip_path):
    if not file_utils.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
    else:
        for file_name in file_utils.listdir(system_path):
            system_file_path = file_utils.join(system_path, file_name).replace(
                "\\", "/"
            )
            zip_file_path = file_utils.join(zip_path, file_name).replace(
                "\\", "/"
            )
            _write_to_zip_recursively(
                zipfile_to_save, system_file_path, zip_file_path
            )


def _name_key(name):
    """Make sure that private attributes are visited last."""
    if name.startswith("_"):
        return f"~{name}"
    return name


def _walk_saveable(saveable):
    from keras.src.saving.keras_saveable import KerasSaveable

    if not isinstance(saveable, KerasSaveable):
        raise ValueError(
            "Expected object to be an "
            "instance of `KerasSaveable`, but "
            f"got {saveable} of type {type(saveable)}"
        )

    obj_type = saveable._obj_type()
    attr_skipset = get_attr_skipset(obj_type)

    # Save all layers directly tracked by Sequential and Functional first.
    # This helps avoid ordering concerns for subclassed Sequential or Functional
    # models with extra attributes--the internal Keras state take precedence.
    if obj_type in ("Sequential", "Functional"):
        yield "layers", saveable.layers

    for child_attr in sorted(dir(saveable), key=lambda x: _name_key(x)):
        if child_attr.startswith("__") or child_attr in attr_skipset:
            continue
        try:
            child_obj = getattr(saveable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        yield child_attr, child_obj


def _save_state(
    saveable,
    weights_store,
    assets_store,
    inner_path,
    visited_saveables,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    if not isinstance(weights_store, (H5IOStore, ShardedH5IOStore, type(None))):
        raise ValueError(
            "Expected `weights_store` to be an instance of "
            "`H5IOStore`, `ShardedH5IOStore`, or `None`. "
            f"Received: {weights_store} of type {type(weights_store)}"
        )
    if not isinstance(assets_store, (DiskIOStore, type(None))):
        raise ValueError(
            "Expected `assets_store` to be an instance of "
            "`DiskIOStore` or `None`. "
            f"Received: {assets_store} of type {type(assets_store)}"
        )

    # If the saveable has already been saved, skip it.
    if id(saveable) in visited_saveables:
        return

    if hasattr(saveable, "save_own_variables") and weights_store:
        if hasattr(saveable, "name") and isinstance(saveable.name, str):
            metadata = {"name": saveable.name}
        else:
            metadata = None
        saveable.save_own_variables(
            weights_store.make(inner_path, metadata=metadata)
        )
    if hasattr(saveable, "save_assets") and assets_store:
        saveable.save_assets(assets_store.make(inner_path))

    visited_saveables.add(id(saveable))

    # Recursively save state of children saveables (layers, optimizers, etc.)
    for child_attr, child_obj in _walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            _save_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                visited_saveables=visited_saveables,
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _save_container_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                visited_saveables=visited_saveables,
            )


def _load_state(
    saveable,
    weights_store,
    assets_store,
    inner_path,
    skip_mismatch=False,
    visited_saveables=None,
    failed_saveables=None,
    error_msgs=None,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    if not isinstance(weights_store, (H5IOStore, ShardedH5IOStore, type(None))):
        raise ValueError(
            "Expected `weights_store` to be an instance of "
            "`H5IOStore`, `ShardedH5IOStore`, or `None`. "
            f"Received: {weights_store} of type {type(weights_store)}"
        )
    if not isinstance(assets_store, (DiskIOStore, type(None))):
        raise ValueError(
            "Expected `assets_store` to be an instance of "
            "`DiskIOStore` or `None`. "
            f"Received: {assets_store} of type {type(assets_store)}"
        )

    if visited_saveables and id(saveable) in visited_saveables:
        return

    failure = False

    if hasattr(saveable, "load_own_variables") and weights_store:
        store = weights_store.get(inner_path)
        if skip_mismatch or failed_saveables is not None:
            try:
                saveable.load_own_variables(store)
            except Exception as e:
                if failed_saveables is not None:
                    failed_saveables.add(id(saveable))
                error_msgs[id(saveable)] = saveable, e
                failure = True
        else:
            saveable.load_own_variables(store)

    if hasattr(saveable, "load_assets") and assets_store:
        if skip_mismatch or failed_saveables is not None:
            try:
                saveable.load_assets(assets_store.get(inner_path))
            except Exception as e:
                if failed_saveables is not None:
                    failed_saveables.add(id(saveable))
                error_msgs[id(saveable)] = saveable, e
                failure = True
        else:
            saveable.load_assets(assets_store.get(inner_path))

    if failed_saveables is not None:
        currently_failed = len(failed_saveables)
    else:
        currently_failed = 0

    # Recursively load states for Keras saveables such as layers/optimizers.
    for child_attr, child_obj in _walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            _load_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _load_container_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )

    if failed_saveables is not None:
        newly_failed = len(failed_saveables) - currently_failed
    else:
        newly_failed = 0

    if not failure:
        if visited_saveables is not None and newly_failed <= 0:
            visited_saveables.add(id(saveable))
        if failed_saveables is not None and id(saveable) in failed_saveables:
            failed_saveables.remove(id(saveable))
            error_msgs.pop(id(saveable))


def _get_unique_name(name, used_names):
    if name in used_names:
        used_names[name] += 1
        return f"{name}_{used_names[name]}"
    used_names[name] = 0
    return name


def _container_path_present(weights_store, assets_store, path):
    """True if either store has anything under `path`.

    Used to detect legacy files that predate the nested-container
    recursion added in PR #22362 — older files don't contain the
    `container*` groups the new loader expects.
    """
    return (weights_store is not None and weights_store.has_path(path)) or (
        assets_store is not None and assets_store.has_path(path)
    )


def _save_container_state(
    container,
    weights_store,
    assets_store,
    inner_path,
    visited_saveables,
    visited_containers=None,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    if visited_containers is None:
        visited_containers = set()
    if id(container) in visited_containers:
        return
    visited_containers.add(id(container))

    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            # Do NOT address the saveable via `saveable.name`, since
            # names are usually autogenerated and thus not reproducible
            # (i.e. they may vary across two instances of the same model).
            name = _get_unique_name(
                naming.to_snake_case(saveable.__class__.__name__),
                used_names,
            )
            _save_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, name).replace("\\", "/"),
                visited_saveables=visited_saveables,
            )
        elif isinstance(saveable, (list, dict, tuple)):
            name = _get_unique_name("container", used_names)
            _save_container_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, name).replace("\\", "/"),
                visited_saveables=visited_saveables,
                visited_containers=visited_containers,
            )


def _load_container_state(
    container,
    weights_store,
    assets_store,
    inner_path,
    skip_mismatch,
    visited_saveables,
    failed_saveables,
    error_msgs,
    visited_containers=None,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    if visited_containers is None:
        visited_containers = set()
    if id(container) in visited_containers:
        return
    visited_containers.add(id(container))

    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            name = _get_unique_name(
                naming.to_snake_case(saveable.__class__.__name__),
                used_names,
            )
            _load_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, name).replace("\\", "/"),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )
        elif isinstance(saveable, (list, dict, tuple)):
            name = _get_unique_name("container", used_names)
            nested_path = file_utils.join(inner_path, name).replace("\\", "/")
            if not _container_path_present(
                weights_store, assets_store, nested_path
            ):
                # The container's group is missing from the saved file.
                # Try to load it recursively anyway with `skip_mismatch=True`,
                # then warn only if doing so produced new failures — i.e. the
                # container genuinely held an unvisited `KerasSaveable` that
                # needed loading. This silently skips two benign cases:
                #   - Containers of pure metadata (e.g. lists of strings in
                #     `variable_serialization_spec`).
                #   - Containers that mirror already-loaded saveables (e.g.
                #     a Functional model's `_operations_by_depth` whose
                #     items are also in the `layers` collection).
                tracked_failures = set()
                tracked_errors = {}
                _load_container_state(
                    saveable,
                    weights_store,
                    assets_store,
                    inner_path=nested_path,
                    skip_mismatch=True,
                    visited_saveables=visited_saveables,
                    failed_saveables=tracked_failures,
                    error_msgs=tracked_errors,
                    visited_containers=visited_containers,
                )
                if tracked_failures:
                    # Legacy files saved before PR #22362 didn't write the
                    # `container*` groups for nested containers — silently
                    # skip so the model still loads (sublayers keep their
                    # freshly-initialized weights).
                    warnings.warn(
                        f"Skipping nested container at '{nested_path}': no "
                        "matching group found in the saved file. This "
                        "usually means the file was saved with a Keras "
                        "version that did not serialize sublayers nested "
                        "inside containers. Affected layers will retain "
                        "their freshly-initialized weights.",
                        stacklevel=2,
                    )
                continue
            _load_container_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=nested_path,
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
                visited_containers=visited_containers,
            )


class DiskIOStore:
    """Asset store backed by disk storage.

    If `archive` is specified, then `root_path` refers to the filename
    inside the archive.

    If `archive` is not specified, then `root_path` refers to the full path of
    the target directory.
    """

    def __init__(self, root_path, archive=None, mode=None):
        self.mode = mode
        self.root_path = root_path
        self.archive = archive
        self.tmp_dir = None
        if self.archive:
            if self.mode == "r":
                _reject_zip_archive_bomb(self.archive)
            self.tmp_dir = get_temp_dir()
            if self.mode == "r":
                file_utils.extract_open_archive(self.archive, self.tmp_dir)
            self.working_dir = file_utils.join(
                self.tmp_dir, self.root_path
            ).replace("\\", "/")
            if self.mode == "w":
                file_utils.makedirs(self.working_dir)
        else:
            if mode == "r":
                self.working_dir = root_path
            else:
                self.tmp_dir = get_temp_dir()
                self.working_dir = file_utils.join(
                    self.tmp_dir, self.root_path
                ).replace("\\", "/")
                file_utils.makedirs(self.working_dir)

    def _full_path(self, path):
        """Resolve `path` under the working dir, rejecting traversal."""
        # Normalize separators first so a `\\` cannot bypass the check below.
        path = path.replace("\\", "/")
        if file_utils.is_remote_path(self.working_dir):
            # `resolve_path` (realpath/abspath) would corrupt a remote prefix
            # such as `gs://bucket`, so validate remote paths by string.
            if path.startswith("/") or "://" in path or ".." in path.split("/"):
                raise ValueError(
                    f"Invalid asset path: '{path}' escapes the asset directory."
                )
            return file_utils.join(self.working_dir, path).replace("\\", "/")
        resolved = file_utils.resolve_sub_path(
            file_utils.resolve_path(self.working_dir), path
        )
        if resolved is None:
            raise ValueError(
                f"Invalid asset path: '{path}' escapes the asset directory."
            )
        return resolved.replace("\\", "/")

    def make(self, path):
        if not path:
            return self.working_dir
        path = self._full_path(path)
        if not file_utils.exists(path):
            file_utils.makedirs(path)
        return path

    def get(self, path):
        if not path:
            return self.working_dir
        path = self._full_path(path)
        if file_utils.exists(path):
            return path
        return None

    def has_path(self, path):
        """Return True if `path` exists on disk under the store's root."""
        if not path:
            return file_utils.exists(self.working_dir)
        try:
            return file_utils.exists(self._full_path(path))
        except ValueError:
            return False

    def close(self):
        if self.mode == "w" and self.archive:
            _write_to_zip_recursively(
                self.archive, self.working_dir, self.root_path
            )
        if self.tmp_dir and file_utils.exists(self.tmp_dir):
            file_utils.rmtree(self.tmp_dir)


def safe_get_h5_group(parent, name):
    """Retrieve a Group within a given File or Group.

    Args:
        parent: the parent h5py.File or h5py.Group.
        name: the name of the Group to retrieve.

    Returns:
        The child h5py.Group.
    """
    current = parent
    for name_part in name.split("/"):
        if not name_part:
            raise ValueError(f"Invalid path in H5 file: {name}")

        # Also handles the case when the group is an empty dict initially.
        if name_part not in current:
            raise KeyError(name)

        if isinstance(current, dict):
            group_type = None
        else:
            group_type = current.get(
                name_part, default=None, getclass=True, getlink=True
            )

        if group_type in (h5py.ExternalLink, h5py.SoftLink):
            raise ValueError(f"Not allowed: H5 file with {group_type.__name__}")

        current = current[name_part]
        if not isinstance(current, h5py.Group):
            raise ValueError(
                f"Invalid H5 file, expected Group but received {type(current)}"
            )

    return current


# Guard against HDF5 "shape bomb" datasets: a dataset can declare an enormous
# shape while storing almost nothing on disk (e.g. chunked + gzip-compressed
# with only a fill value), which forces a huge allocation when it is read into
# memory (CWE-789 / CWE-409). For datasets whose declared in-memory size is
# above this floor, we require it to stay within `_H5_DATASET_MAX_EXPANSION` of
# the bytes actually stored on disk. Genuine arrays (even compressed) satisfy
# this; shape/decompression bombs, which store next to nothing, do not.
_H5_DATASET_BOMB_FLOOR_BYTES = 1 << 32  # 4 GiB
_H5_DATASET_MAX_EXPANSION = 1000
# Floor for the *cumulative* (whole-file) guard below. Genuine HDF5 weight
# files store their arrays uncompressed, so their total declared size tracks
# the on-disk size (ratio ~1); only a bomb declares far more than is stored.
# A much lower floor than the per-dataset one lets the cumulative guard also
# catch a single sub-4 GiB dataset that the per-dataset floor would miss.
_H5_CUMULATIVE_BOMB_FLOOR_BYTES = 1 << 26  # 64 MiB


def safe_get_h5_dataset(group, name):
    """Retrieve a Dataset within a given Group.

    Args:
        group: the parent h5py.Group.
        name: the name of the Dataset to retrieve.

    Returns:
        The child h5py.Dataset.
    """
    if "/" in name:
        # Separate the dataset name from it's parent group.
        group_name, name = name.rsplit("/", 1)
        group = safe_get_h5_group(group, group_name)

    # Also handles the case when the group is an empty dict initially.
    if name not in group:
        raise KeyError(name)

    dataset_type = group.get(name, default=None, getclass=True, getlink=True)
    if dataset_type in (h5py.ExternalLink, h5py.SoftLink):
        raise ValueError(f"Not allowed: H5 file with {dataset_type.__name__}")

    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(
            f"Invalid H5 file, expected Dataset, received {type(dataset)}"
        )
    if dataset.external:
        raise ValueError(
            f"Not allowed: H5 file with external Dataset: {dataset.external}"
        )
    if dataset.is_virtual:
        raise ValueError("Not allowed: H5 file with virtual Dataset")
    declared_bytes = math.prod(dataset.shape) * dataset.dtype.itemsize
    stored_bytes = dataset.id.get_storage_size()
    if (
        declared_bytes > _H5_DATASET_BOMB_FLOOR_BYTES
        and declared_bytes > _H5_DATASET_MAX_EXPANSION * stored_bytes
    ):
        raise ValueError(
            f"Not allowed: H5 dataset '{name}' declares "
            f"{readable_memory_size(declared_bytes)} but only "
            f"{readable_memory_size(stored_bytes)} are stored on disk; "
            "refusing to load a potential decompression/shape bomb."
        )
    return dataset


def reject_h5_shape_bomb(h5_file):
    """Guard against an HDF5 file that is a cumulative shape bomb.

    `safe_get_h5_dataset` bounds a *single* dataset, but its per-dataset floor
    can be evaded by many datasets that each declare just under the floor while
    storing almost nothing, and that are then read into memory together (e.g.
    `load_subset_weights_from_hdf5_group` materializes every weight of a layer
    at once). This bounds the *cumulative* in-memory size of all datasets in
    the file against the bytes actually stored on disk, using the same
    expansion ratio. Because genuine weight files are uncompressed (declared
    size ~= file size), this also catches a single dataset that declares just
    under the per-dataset 4 GiB floor (CWE-789 / CWE-409).
    """
    try:
        file_size = h5_file.id.get_filesize()
    except Exception:
        # If we cannot determine the on-disk size, skip the cumulative check
        # rather than break loading; the per-dataset guard still applies.
        return
    if file_size <= 0:
        return
    total_declared = 0

    def _accumulate(_name, obj):
        nonlocal total_declared
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.is_virtual:
            # A virtual dataset maps to data outside this file, so its declared
            # size is not backed by the on-disk bytes we measure here. Reject it
            # outright, exactly as `safe_get_h5_dataset` does on access, instead
            # of letting it slip through the cumulative accounting.
            raise ValueError("Not allowed: H5 file with virtual Dataset")
        # A null-dataspace dataset has `shape is None`; skip it (`math.prod`
        # would raise on `None`).
        if obj.shape is not None:
            total_declared += math.prod(obj.shape) * obj.dtype.itemsize

    # `visititems` only walks hard-linked datasets; external/soft links are
    # rejected separately by `safe_get_h5_dataset` when they are accessed.
    h5_file.visititems(_accumulate)
    if (
        total_declared > _H5_CUMULATIVE_BOMB_FLOOR_BYTES
        and total_declared > _H5_DATASET_MAX_EXPANSION * file_size
    ):
        raise ValueError(
            "Not allowed: the HDF5 file declares "
            f"{readable_memory_size(total_declared)} of array data but only "
            f"{readable_memory_size(file_size)} are stored on disk; "
            "refusing to load a potential decompression/shape bomb."
        )


class H5IOStore:
    """Numerical variable store backed by HDF5.

    Args:
        path_or_io: `str`, `pathlib.Path` or `io.BytesIO` object. The path where
            to save the model.
        archive: Optional `zipfile.ZipFile` object. If specified, the h5 file
            will be saved inside the archive and `path_or_io` will be used as
            the filename.
        mode: `str`. One of {`"r"`, `"w"`}. The mode to open the h5 file.
            Defaults to `"r"`.
    """

    def __init__(self, path_or_io, archive=None, mode="r"):
        if mode not in ("w", "r"):
            raise ValueError(
                f"`mode` should be either 'w' or 'r'. Received: {mode}"
            )
        if isinstance(path_or_io, (str, pathlib.Path)):
            self.path_or_io = pathlib.Path(path_or_io)
        elif isinstance(path_or_io, io.BytesIO):
            if archive is not None:
                raise ValueError(
                    "When `path_or_io` is an `io.BytesIO` object, `archive` "
                    "should be `None`."
                )
            self.path_or_io = path_or_io
        else:
            raise TypeError(
                "`path_or_io` should be a `str`, `pathlib.Path` or "
                f"`io.BytesIO` object. Received: path_or_io={path_or_io} of "
                f"type {type(path_or_io)}."
            )
        self.mode = mode
        self.archive = archive
        self.io_file = None

        # Init H5 file. `_get_h5_file` applies `reject_h5_shape_bomb` on every
        # read-mode open, so it also covers `ShardedH5IOStore`'s lazily opened
        # shards (which never call this `__init__`).
        self.h5_file = self._get_h5_file(self.path_or_io)

        # Init H5 entry group.
        self._h5_entry_path = None
        self._h5_entry_group = {}
        self._h5_entry_metadata = None
        self._h5_entry_initialized = False

    def __bool__(self):
        # Delegate `__bool__` to the underlying `h5_file`. Otherwise, Python
        # will mistakenly using `__len__` to determine the value.
        return self.h5_file.__bool__()

    def _get_h5_file(self, path_or_io, mode=None):
        mode = mode or self.mode
        if mode not in ("r", "w", "a"):
            raise ValueError(
                f"`mode` should be either 'r', 'w' or 'a'. Received: {mode}"
            )
        if self.archive:
            if mode == "w":
                self.io_file = io.BytesIO()
            else:
                self.io_file = self.archive.open(str(path_or_io), "r")
            h5_file = h5py.File(self.io_file, mode=mode)
        else:
            h5_file = h5py.File(path_or_io, mode=mode)
        if mode == "r":
            # Guard every read-mode open (covers `H5IOStore`, sharded shards
            # opened lazily by `ShardedH5IOStore`, and shard switching).
            reject_h5_shape_bomb(h5_file)
        return h5_file

    def make(self, path, metadata=None):
        """Make a new H5 entry group.

        This method is only available in write mode. It defers the creation of
        the H5 entry group until `__setitem__` is called, preventing the
        creation of empty groups.

        Args:
            path: `str`. The variable path.
            metadata: Optional `dict`. The metadata to save with the H5 entry
                group. Defaults to `None`.
        """
        if self.mode != "w":
            raise ValueError("`make` is only allowed in write mode.")
        if not isinstance(metadata, (dict, type(None))):
            raise ValueError(
                f"`metadata` should be a dict or `None`. Received: {metadata}"
            )

        self._h5_entry_path = path
        if metadata:
            self._create_h5_group(path, metadata=metadata)
        else:
            # Defer to `__setitem__` for H5 group creation to prevent the
            # creation of empty groups when the store is unused.
            self._h5_entry_group = {}
            self._h5_entry_initialized = False
        return self

    def get(self, path):
        """Get the H5 entry group.

        This method is only available in read mode.

        Args:
            path: `str`. The variable path.
        """
        if self.mode != "r":
            raise ValueError("`get` is only allowed in read mode.")

        self._h5_entry_path = path
        self._h5_entry_group = {}  # Defaults to an empty dict if not found.
        self._h5_entry_initialized = True

        if not path:
            if "vars" in self.h5_file:
                self._h5_entry_group = safe_get_h5_group(self.h5_file, "vars")
            return self

        if path in self.h5_file:
            path_group = safe_get_h5_group(self.h5_file, path)
            if "vars" in path_group:
                self._h5_entry_group = safe_get_h5_group(path_group, "vars")
                return self

        # No hit. Fix for 2.13 compatibility.
        if "_layer_checkpoint_dependencies" in self.h5_file:
            path = path.replace("layers", "_layer_checkpoint_dependencies")
            if path in self.h5_file:
                path_group = safe_get_h5_group(self.h5_file, path)
                if "vars" in path_group:
                    self._h5_entry_group = safe_get_h5_group(path_group, "vars")

        return self

    def has_path(self, path):
        """Return True if a group exists at `path` in the H5 file."""
        return path in self.h5_file

    def close(self):
        self.h5_file.close()
        if self.mode == "w" and self.archive:
            self.archive.writestr(str(self.path_or_io), self.io_file.getvalue())
        if self.io_file:
            self.io_file.close()

    # H5 entry level methods.

    def _create_h5_group(self, path, metadata=None):
        if not path:
            self._h5_entry_group = self.h5_file.create_group("vars")
        else:
            self._h5_entry_group = self.h5_file.create_group(path).create_group(
                "vars"
            )
        if metadata:
            for k, v in metadata.items():
                self._h5_entry_group.attrs[k] = v

        self._h5_entry_initialized = True

    def __len__(self):
        return self._h5_entry_group.__len__()

    def keys(self):
        return self._h5_entry_group.keys()

    def __getitem__(self, key):
        value = safe_get_h5_dataset(self._h5_entry_group, key)
        if (
            hasattr(value, "attrs")
            and "dtype" in value.attrs
            and value.attrs["dtype"] == "bfloat16"
        ):
            value = np.array(value, dtype=ml_dtypes.bfloat16)
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        return value

    def __setitem__(self, key, value):
        if self.mode not in ("w", "a"):
            raise ValueError("Setting a value is only allowed in write mode.")
        if not self._h5_entry_initialized:
            self._create_h5_group(self._h5_entry_path)

        value = backend.ops.convert_to_numpy(value)
        if standardize_dtype(value.dtype) == "bfloat16":
            ds = self._h5_entry_group.create_dataset(key, data=value)
            ds.attrs["dtype"] = "bfloat16"
        else:
            self._h5_entry_group[key] = value

    def __delitem__(self, key):
        if self.mode not in ("w", "a"):
            raise ValueError("Deleting a value is only allowed in write mode.")
        del self._h5_entry_group[key]

    def __contains__(self, item):
        return item in self._h5_entry_group


class ShardedH5IOStore(H5IOStore):
    """Sharded numerical variable store backed by HDF5.

    Args:
        path_or_io: `str` or `pathlib.Path` object. The path where to save the
            model.
        max_shard_size: `int` or `float`. Maximum size in GB for each sharded
            file. If `None`, no sharding will be done. Defaults to `None`.
        archive: Optional `zipfile.ZipFile` object. If specified, the h5 file
            will be saved inside the archive and `path_or_io` will be used as
            the filename.
        mode: `str`. One of {'r', 'w'}. The mode to open the h5 file. Defaults
            to `"r"`.
    """

    def __init__(self, path_or_io, max_shard_size=5, archive=None, mode="r"):
        if mode not in ("w", "r"):
            raise ValueError(
                f"`mode` should be either 'w' or 'r'. Received: {mode}"
            )
        if not isinstance(path_or_io, (str, pathlib.Path)):
            raise TypeError(
                "`path_or_io` should be a `str`, `pathlib.Path` object. "
                f"Received: path_or_io={path_or_io} of type {type(path_or_io)}."
            )
        self.path = pathlib.Path(path_or_io)
        self.mode = mode
        self.archive = archive
        self.io_file = None

        self.max_shard_size = float(max_shard_size) * 1024**3  # To bytes.
        self.base_name = self.path.stem.replace(".weights", "")

        if self.path.suffix != ".json":
            method = "Saving" if self.mode == "w" else "Loading"
            new_path = self.path.with_suffix(".json")
            warnings.warn(
                f"{method} sharded weights requires `*.json` as the "
                f"extension. The original path: {str(self.path)} will be "
                f"renamed to {str(new_path)}."
            )
            self.path = new_path

        # Init H5 entry group.
        self._h5_entry_path = None
        self._h5_entry_group = {}
        self._h5_entry_metadata = None
        self._h5_entry_initialized = False

        # Init shard parameters.
        self.current_shard_index = 0
        self.current_shard_size = 0
        self.total_shard_size = 0  # In bytes.
        self.current_shard_path = None
        self.current_shard_filenames = []
        if self.mode == "w":
            self.sharding_config = {
                "metadata": {
                    "total_size": 0,
                },
                "weight_map": {},
            }
        else:
            if self.archive:
                self.sharding_config = json.loads(
                    _safe_zip_read(self.archive, str(self.path))
                )
            else:
                with open(self.path, "r") as map_file:
                    self.sharding_config = json.load(map_file)
        self.h5_file = self._create_new_shard_file()

    def make(self, path, metadata=None):
        """Make a new H5 entry group.

        This method is only available in write mode. It defers the creation of
        the H5 entry group until `__setitem__` is called, preventing the
        creation of empty groups.

        The information about the current shard is reset.

        Args:
            path: `str`. The variable path.
            metadata: Optional `dict`. The metadata to save with the H5 entry
                group. Defaults to `None`.
        """
        self.current_shard_filenames = []
        if self.h5_file is not None:
            self.current_shard_filenames.append(
                pathlib.Path(self.h5_file.filename).name
            )
        return super().make(path, metadata)

    def get(self, path):
        """Get the H5 entry group.

        This method is only available in read mode. If the path is not found in
        the current shard, it will switch to the correct shard.

        Args:
            path: `str`. The variable path.
        """
        if not path:
            parsed_path = "/vars"
        else:
            parsed_path = path

        # If not found, check shard map and switch files.
        weight_map = self.sharding_config["weight_map"]
        filenames = weight_map.get(parsed_path) or weight_map.get(
            f"/{parsed_path}/vars"
        )
        if filenames is not None:
            if not isinstance(filenames, list):
                filenames = [filenames]
            self.current_shard_filenames = filenames
            filename = filenames[0]
        else:
            self.current_shard_filenames = []
            filename = None

        if filename is not None and filename != self.current_shard_path.name:
            self.close()
            self.current_shard_path = self.path.with_name(filename)
            self.h5_file = self._get_h5_file(self.current_shard_path)
        return super().get(path)

    def has_path(self, path):
        """Return True if any shard has a group under `path`.

        Resolves via the shard weight-map rather than the currently open
        shard, so the answer is independent of which shard is active.
        """
        weight_map = self.sharding_config.get("weight_map", {})
        leading = f"/{path}"
        prefix = f"{leading}/"
        return any(k == leading or k.startswith(prefix) for k in weight_map)

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        if self.mode == "w":
            self.sharding_config["metadata"]["total_size"] = (
                self.total_shard_size
            )
            json_str = json.dumps(self.sharding_config, indent=4)
            if self.archive:
                self.archive.writestr(str(self.path), json_str)
                self.archive.writestr(
                    str(self.current_shard_path), self.io_file.getvalue()
                )
            else:
                with open(self.path, "w") as f:
                    f.write(json_str)
        if self.io_file:
            self.io_file.close()

    # Shard-specific methods.

    def _create_new_shard_file(self):
        """Create a new shard file and return the H5 file object."""
        new_shard_path = (
            f"{self.base_name}_{self.current_shard_index:05}.weights.h5"
        )
        self.current_shard_index += 1
        self.current_shard_path = self.path.with_name(new_shard_path)
        h5_file = self._get_h5_file(self.current_shard_path)
        self.current_shard_filenames.append(pathlib.Path(h5_file.filename).name)
        self._h5_entry_initialized = False
        return h5_file

    def _switch_h5_file(self, filename, mode):
        """Switch to a different H5 file with the specified mode.

        This is useful for retrieving information from all shards, such as the
        total length, keys, and items.
        """
        if mode not in ("r", "a"):
            raise ValueError(
                f"`mode` should be either 'r' or 'a'. Received: {mode}"
            )
        self.close()
        self.h5_file = self._get_h5_file(
            self.path.with_name(filename), mode=mode
        )
        self._get_h5_group(self._h5_entry_path)

    def _restore_h5_file(self):
        """Ensure the current shard is the last one created."""
        if (
            pathlib.Path(self.h5_file.filename).name
            != self.current_shard_path.name
        ):
            mode = "a" if self.mode == "w" else "r"
            self._switch_h5_file(self.current_shard_path.name, mode=mode)

    # H5 entry level methods.

    def _get_h5_group(self, path):
        """Get the H5 entry group. If it doesn't exist, return an empty dict."""
        try:
            if not path:
                self._h5_entry_group = safe_get_h5_group(self.h5_file, "vars")
            else:
                self._h5_entry_group = safe_get_h5_group(
                    safe_get_h5_group(self.h5_file, path), "vars"
                )
            self._h5_entry_initialized = True
        except KeyError:
            self._h5_entry_group = {}
            self._h5_entry_initialized = False

    # Dict methods.

    def __len__(self):
        total_len = self._h5_entry_group.__len__()
        for filename in self.current_shard_filenames:
            if filename == self.current_shard_path.name:
                continue
            self._switch_h5_file(filename, mode="r")
            total_len += self._h5_entry_group.__len__()
        self._restore_h5_file()
        return total_len

    def keys(self):
        keys = []
        current_shard_keys = list(self._h5_entry_group.keys())
        for filename in self.current_shard_filenames:
            if filename == self.current_shard_path.name:
                keys += current_shard_keys
            else:
                self._switch_h5_file(filename, mode="r")
                keys += list(self._h5_entry_group.keys())
        self._restore_h5_file()
        return keys

    def __getitem__(self, key):
        if key in self._h5_entry_group:
            return super().__getitem__(key)

        for filename in self.current_shard_filenames:
            if filename == self.current_shard_path.name:
                continue
            self._switch_h5_file(filename, mode="r")
            if key in self._h5_entry_group:
                item = super().__getitem__(key)
                self._restore_h5_file()
                return item
        raise KeyError(
            f"Key '{key}' not found in any of the shards: "
            f"{self.current_shard_filenames}"
        )

    def __setitem__(self, key, value):
        self._restore_h5_file()

        # Accumulate `current_shard_size`.
        value = backend.ops.convert_to_numpy(value)
        dtype = standardize_dtype(value.dtype)
        weight_counts = math.prod(value.shape)
        per_param_size = dtype_utils.dtype_size(dtype)
        value_size = weight_counts * per_param_size / 8  # In bytes.
        self.total_shard_size += value_size
        if value_size > self.max_shard_size:
            value_size_str = readable_memory_size(value_size)
            max_shard_size_str = readable_memory_size(self.max_shard_size)
            raise ValueError(
                f"The size of {key} is {value_size_str} which "
                f"exceeds the maximum shard size {max_shard_size_str}. You "
                "can increase the `max_shard_size` parameter to accommodate "
                "the size."
            )

        # Create a new shard if the current shard is full.
        self.current_shard_size += value_size
        if self.current_shard_size > self.max_shard_size:
            self.close()
            self.h5_file = self._create_new_shard_file()
            self.current_shard_size = value_size

        super().__setitem__(key, value)

        # Update the weight map.
        variable_path = self._h5_entry_group.name
        shard_filename = self.current_shard_path.name
        weight_map = self.sharding_config["weight_map"]
        if variable_path not in weight_map:
            weight_map[variable_path] = shard_filename
        else:
            if not isinstance(weight_map[variable_path], list):
                weight_map[variable_path] = [weight_map[variable_path]]
            if shard_filename not in weight_map[variable_path]:
                weight_map[variable_path].append(shard_filename)

    def __delitem__(self, key):
        if key in self._h5_entry_group:
            super().__delitem__(key)
            return

        for filename in self.current_shard_filenames:
            if filename == self.current_shard_path.name:
                continue
            self._switch_h5_file(filename, mode="a")
            if key in self._h5_entry_group:
                super().__delitem__(key)
                self._restore_h5_file()
                return
        raise KeyError(
            f"Key '{key}' not found in any of the shards: "
            f"{self.current_shard_filenames}"
        )

    def __contains__(self, item):
        if item in self._h5_entry_group:
            return True

        for filename in self.current_shard_filenames:
            if filename == self.current_shard_path.name:
                continue
            self._switch_h5_file(filename, mode="r")
            if item in self._h5_entry_group:
                self._restore_h5_file()
                return True
        self._restore_h5_file()
        return False


def get_temp_dir():
    temp_dir = tempfile.mkdtemp()
    testfile = tempfile.TemporaryFile(dir=temp_dir)
    testfile.close()
    return temp_dir


def get_attr_skipset(obj_type):
    skipset = global_state.get_global_attribute(
        f"saving_attr_skiplist_{obj_type}", None
    )
    if skipset is not None:
        return skipset

    skipset = set(
        [
            "_self_unconditional_dependency_names",
        ]
    )
    if obj_type == "Operation":
        from keras.src.ops.operation import Operation

        ref_obj = Operation()
        skipset.update(dir(ref_obj))
    elif obj_type == "Layer":
        from keras.src.layers.layer import Layer

        ref_obj = Layer()
        skipset.update(dir(ref_obj))
    elif obj_type == "Functional":
        from keras.src.layers.layer import Layer

        ref_obj = Layer()
        skipset.update(dir(ref_obj) + ["operations", "_operations"])
    elif obj_type == "Sequential":
        from keras.src.layers.layer import Layer

        ref_obj = Layer()
        skipset.update(dir(ref_obj) + ["_functional"])
    elif obj_type == "Metric":
        from keras.src.metrics.metric import Metric
        from keras.src.trainers.compile_utils import CompileMetrics

        ref_obj_a = Metric()
        ref_obj_b = CompileMetrics([], [])
        skipset.update(dir(ref_obj_a) + dir(ref_obj_b))
    elif obj_type == "Optimizer":
        from keras.src.optimizers.optimizer import Optimizer

        ref_obj = Optimizer(1.0)
        skipset.update(dir(ref_obj))
        skipset.remove("variables")
    elif obj_type == "Loss":
        from keras.src.losses.loss import Loss

        ref_obj = Loss()
        skipset.update(dir(ref_obj))
    elif obj_type == "Cross":
        from keras.src.layers.preprocessing.feature_space import Cross

        ref_obj = Cross((), 1)
        skipset.update(dir(ref_obj))
    elif obj_type == "Feature":
        from keras.src.layers.preprocessing.feature_space import Feature

        ref_obj = Feature("int32", lambda x: x, "int")
        skipset.update(dir(ref_obj))
    else:
        raise ValueError(
            f"get_attr_skipset got invalid {obj_type=}. "
            "Accepted values for `obj_type` are "
            "['Operation', 'Layer', 'Functional', 'Sequential', 'Metric', "
            "'Optimizer', 'Loss', 'Cross', 'Feature']"
        )

    global_state.set_global_attribute(
        f"saving_attr_skipset_{obj_type}", skipset
    )
    return skipset


def is_memory_sufficient(model):
    """Check if there is sufficient memory to load the model into memory.

    If psutil is installed, we can use it to determine whether the memory is
    sufficient. Otherwise, we use a predefined value of 1 GB for available
    memory.
    """
    if psutil is None:
        available_memory = 1024 * 1024 * 1024  # 1 GB in bytes
    else:
        available_memory = psutil.virtual_memory().available  # In bytes
    return (
        weight_memory_size(model.variables)
        < available_memory * _MEMORY_UPPER_BOUND
    )


def _split_path_components(path):
    """Split a relative path into a list of individual components.

    Uses ``os.path.split`` iteratively so the result is independent of
    the platform path separator.

    Example::

        _split_path_components("a/b/c.txt") -> ["a", "b", "c.txt"]
    """
    parts = []
    while True:
        head, tail = os.path.split(path)
        if tail:
            parts.append(tail)
        elif head:
            parts.append(head)
            break
        else:
            break
        path = head
    parts.reverse()
    return parts


def _write_nested_dict_to_dir(tree, base_dir):
    """Recursively write a nested dict of numpy arrays to a directory tree.

    Each dict key becomes a directory or filename. Leaf values (numpy
    arrays) are written as binary files.

    Args:
        tree: A nested dictionary of numpy arrays.
        base_dir: The base directory to write the assets to. Must be resolved
            via `file_utils.resolve_path` first.
    """
    for key, value in tree.items():
        if os.path.sep in key or (os.path.altsep and os.path.altsep in key):
            raise ValueError(f"Invalid asset path: {key}")
        child_path = file_utils.resolve_sub_path(base_dir, key)
        if child_path is None:
            raise ValueError(f"Invalid asset path: {key}")

        if isinstance(value, dict):
            os.makedirs(child_path, exist_ok=True)
            _write_nested_dict_to_dir(value, child_path)
        elif isinstance(value, np.ndarray):
            with open(child_path, "wb") as f:
                f.write(value.tobytes())


def _save_assets_to_dict(model):
    """Save model assets to a nested dictionary.

    Collects assets (e.g. vocabulary files) from the model using the
    Keras ``_save_state`` mechanism and returns them as a nested dictionary
    that mirrors the directory hierarchy. Leaf values are numpy uint8
    arrays containing file contents.

    For example, a file at ``layer/sublayer/vocab.txt`` is stored as::

        {"layer": {"sublayer": {"vocab.txt": np.array([...])}}}

    Using a nested structure instead of flat path keys avoids
    platform-specific path separator issues and zip-slip vulnerabilities.

    Args:
        model: The model whose assets should be collected.

    Returns:
        A nested dictionary of numpy uint8 arrays mirroring the asset
        directory tree, or ``None`` if the model has no assets.
    """
    assets_store = DiskIOStore("assets", mode="w")
    try:
        _save_state(
            model,
            weights_store=None,
            assets_store=assets_store,
            inner_path="",
            visited_saveables=set(),
        )

        assets_tree = {}
        working_dir = assets_store.working_dir
        for root, dirs, files in os.walk(working_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                rel = os.path.relpath(file_path, working_dir)
                parts = _split_path_components(rel)
                with open(file_path, "rb") as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                node = assets_tree
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = data

        return assets_tree if assets_tree else None
    finally:
        assets_store.close()


def _load_assets_from_dict(model, assets_dict):
    """Load assets from a nested dictionary into the model.

    Reconstructs the asset directory tree from a nested dictionary
    produced by ``_save_assets_to_dict`` and loads the assets into
    the model via the Keras ``_load_state`` mechanism.

    Args:
        model: The model to load assets into.
        assets_dict: Nested dictionary mirroring the asset directory
            tree, with numpy uint8 arrays as leaf values.
    """
    if not assets_dict:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = file_utils.resolve_path(tmp_dir)
        _write_nested_dict_to_dir(assets_dict, tmp_dir)

        assets_store = DiskIOStore(tmp_dir, mode="r")
        _load_state(
            model,
            weights_store=None,
            assets_store=assets_store,
            inner_path="",
            visited_saveables=set(),
            failed_saveables=set(),
            error_msgs={},
        )
        assets_store.close()
