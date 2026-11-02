import collections
import json
import math
import os.path
import pprint
import zipfile

import numpy as np
import rich.console

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.saving import saving_lib
from keras.src.saving.saving_lib import H5IOStore
from keras.src.utils import naming
from keras.src.utils import summary_utils
from keras.src.utils.module_utils import h5py

try:
    import IPython as ipython
except ImportError:
    ipython = None


def is_ipython_notebook():
    """Checks if the code is being executed in a notebook."""
    try:
        from IPython import get_ipython

        # Check if an active IPython shell exists.
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False


@keras_export("keras.saving.KerasFileEditor")
class KerasFileEditor:
    """Utility to inspect, edit, and resave Keras weights files.

    You will find this class useful when adapting
    an old saved weights file after having made
    architecture changes to a model.

    Args:
        filepath: The path to a local file to inspect and edit.

    Examples:

    ```python
    editor = KerasFileEditor("my_model.weights.h5")

    # Displays current contents
    editor.summary()

    # Remove the weights of an existing layer
    editor.delete_object("layers/dense_2")

    # Add the weights of a new layer
    editor.add_object("layers/einsum_dense", weights={"0": ..., "1": ...})

    # Save the weights of the edited model
    editor.resave_weights("edited_model.weights.h5")
    ```
    """

    def __init__(
        self,
        filepath,
    ):
        self.filepath = filepath
        self.metadata = None
        self.config = None
        self.model = None
        self.console = rich.console.Console(highlight=False)

        if filepath.endswith(".keras"):
            zf = zipfile.ZipFile(filepath, "r")
            try:
                saving_lib._reject_zip_archive_bomb(zf)
            except Exception:
                zf.close()
                raise
            # Reject a decompression-bomb weights member up front, mirroring
            # `saving_lib._load_model_from_fileobj`.
            saving_lib._reject_zip_bomb(zf, saving_lib._VARS_FNAME_H5)
            weights_store = H5IOStore(
                saving_lib._VARS_FNAME_H5,
                archive=zf,
                mode="r",
            )
            config_json = saving_lib._safe_zip_read(
                zf, saving_lib._CONFIG_FILENAME
            )
            metadata_json = saving_lib._safe_zip_read(
                zf, saving_lib._METADATA_FILENAME
            )
            self.config = json.loads(config_json)
            self.metadata = json.loads(metadata_json)

        elif filepath.endswith(".weights.h5"):
            weights_store = H5IOStore(filepath, mode="r")
        else:
            raise ValueError(
                "Invalid filename: "
                "expected a `.keras` `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )

        weights_dict, object_metadata = self._extract_weights_from_store(
            weights_store.h5_file
        )
        weights_store.close()
        self.weights_dict = weights_dict
        self.object_metadata = object_metadata  # {path: object_name}
        self.console.print(self._generate_filepath_info(rich_style=True))

        if self.metadata is not None:
            self.console.print(self._generate_metadata_info(rich_style=True))

    def summary(self):
        """Prints the weight structure of the opened file."""
        self._weights_summary_cli()

    def compare(self, reference_model):
        """Compares the opened file to a reference model.

        This method will list all mismatches between the
        currently opened file and the provided reference model.

        Args:
            reference_model: Model instance to compare to.

        Returns:
            Dict with the following keys:
            `'status'`, `'error_count'`, `'match_count'`.
            Status can be `'success'` or `'error'`.
            `'error_count'` is the number of mismatches found.
            `'match_count'` is the number of matching weights found.
        """
        self.console.print("Running comparison")
        ref_spec = {}
        get_weight_spec_of_saveable(reference_model, ref_spec)

        def _compare(
            target,
            ref_spec,
            inner_path,
            target_name,
            ref_name,
            error_count,
            match_count,
            checked_paths,
        ):
            base_inner_path = inner_path
            for ref_key, ref_val in ref_spec.items():
                inner_path = f"{base_inner_path}/{ref_key}"
                if inner_path in checked_paths:
                    continue

                if ref_key not in target:
                    error_count += 1
                    checked_paths.add(inner_path)
                    if isinstance(ref_val, dict):
                        self.console.print(
                            f"[color(160)]...Object [bold]{inner_path}[/] "
                            f"present in {ref_name}, "
                            f"missing from {target_name}[/]"
                        )
                        self.console.print(
                            f"    In {ref_name}, {inner_path} contains "
                            f"the following keys: {list(ref_val.keys())}"
                        )
                    else:
                        self.console.print(
                            f"[color(160)]...Weight [bold]{inner_path}[/] "
                            f"present in {ref_name}, "
                            f"missing from {target_name}[/]"
                        )
                elif isinstance(ref_val, dict):
                    _error_count, _match_count = _compare(
                        target[ref_key],
                        ref_spec[ref_key],
                        inner_path,
                        target_name,
                        ref_name,
                        error_count=error_count,
                        match_count=match_count,
                        checked_paths=checked_paths,
                    )
                    error_count += _error_count
                    match_count += _match_count
                else:
                    if target[ref_key].shape != ref_val.shape:
                        error_count += 1
                        checked_paths.add(inner_path)
                        self.console.print(
                            f"[color(160)]...Weight shape mismatch "
                            f"for [bold]{inner_path}[/][/]\n"
                            f"    In {ref_name}: "
                            f"shape={ref_val.shape}\n"
                            f"    In {target_name}: "
                            f"shape={target[ref_key].shape}"
                        )
                    else:
                        match_count += 1
            return error_count, match_count

        checked_paths = set()
        error_count, match_count = _compare(
            self.weights_dict,
            ref_spec,
            inner_path="",
            target_name="saved file",
            ref_name="reference model",
            error_count=0,
            match_count=0,
            checked_paths=checked_paths,
        )
        _error_count, _ = _compare(
            ref_spec,
            self.weights_dict,
            inner_path="",
            target_name="reference model",
            ref_name="saved file",
            error_count=0,
            match_count=0,
            checked_paths=checked_paths,
        )
        error_count += _error_count
        self.console.print("─────────────────────")
        if error_count == 0:
            status = "success"
            self.console.print(
                "[color(28)][bold]Comparison successful:[/] "
                "saved file is compatible with the reference model[/]"
            )
            if match_count == 1:
                plural = ""
            else:
                plural = "s"
            self.console.print(
                f"    Found {match_count} matching weight{plural}"
            )
        else:
            status = "error"
            if error_count == 1:
                plural = ""
            else:
                plural = "s"
            self.console.print(
                f"[color(160)][bold]Found {error_count} error{plural}:[/] "
                "saved file is not compatible with the reference model[/]"
            )
        return {
            "status": status,
            "error_count": error_count,
            "match_count": match_count,
        }

    def _edit_object(self, edit_fn, source_name, target_name=None):
        if target_name is not None and "/" in target_name:
            raise ValueError(
                "Argument `target_name` should be a leaf name, "
                "not a full path name. "
                f"Received: target_name='{target_name}'"
            )
        if "/" in source_name:
            # It's a path
            elements = source_name.split("/")
            weights_dict = self.weights_dict
            for e in elements[:-1]:
                if e not in weights_dict:
                    raise ValueError(
                        f"Path '{source_name}' not found in model."
                    )
                weights_dict = weights_dict[e]
            if elements[-1] not in weights_dict:
                raise ValueError(f"Path '{source_name}' not found in model.")
            edit_fn(
                weights_dict, source_name=elements[-1], target_name=target_name
            )
        else:
            # Ensure unicity
            def count_occurences(d, name, count=0):
                for k in d:
                    if isinstance(d[k], dict):
                        count += count_occurences(d[k], name, count)
                if name in d:
                    count += 1
                return count

            occurrences = count_occurences(self.weights_dict, source_name)
            if occurrences > 1:
                raise ValueError(
                    f"Name '{source_name}' occurs more than once in the model; "
                    "try passing a complete path"
                )
            if occurrences == 0:
                raise ValueError(
                    f"Source name '{source_name}' does not appear in the "
                    "model. Use `editor.weights_summary()` "
                    "to list all objects."
                )

            def _edit(d):
                for k in d:
                    if isinstance(d[k], dict):
                        _edit(d[k])
                if source_name in d:
                    edit_fn(d, source_name=source_name, target_name=target_name)

            _edit(self.weights_dict)

    def rename_object(self, object_name, new_name):
        """Rename an object in the file (e.g. a layer).

        Args:
            object_name: String, name or path of the
                object to rename (e.g. `"dense_2"` or
                `"layers/dense_2"`).
            new_name: String, new name of the object.
        """

        def rename_fn(weights_dict, source_name, target_name):
            weights_dict[target_name] = weights_dict[source_name]
            weights_dict.pop(source_name)

        self._edit_object(rename_fn, object_name, new_name)

    def delete_object(self, object_name):
        """Removes an object from the file (e.g. a layer).

        Args:
            object_name: String, name or path of the
                object to delete (e.g. `"dense_2"` or
                `"layers/dense_2"`).
        """

        def delete_fn(weights_dict, source_name, target_name=None):
            weights_dict.pop(source_name)

        self._edit_object(delete_fn, object_name)

    def add_object(self, object_path, weights):
        """Add a new object to the file (e.g. a layer).

        Args:
            object_path: String, full path of the
                object to add (e.g. `"layers/dense_2"`).
            weights: Dict mapping weight names to weight
                values (arrays),
                e.g. `{"0": kernel_value, "1": bias_value}`.
        """
        if not isinstance(weights, dict):
            raise ValueError(
                "Argument `weights` should be a dict "
                "where keys are weight names (usually '0', '1', etc.) "
                "and values are NumPy arrays. "
                f"Received: type(weights)={type(weights)}"
            )

        if "/" in object_path:
            # It's a path
            elements = object_path.split("/")
            partial_path = "/".join(elements[:-1])
            weights_dict = self.weights_dict
            for e in elements[:-1]:
                if e not in weights_dict:
                    raise ValueError(
                        f"Path '{partial_path}' not found in model."
                    )
                weights_dict = weights_dict[e]
            weights_dict[elements[-1]] = weights
        else:
            self.weights_dict[object_path] = weights

    def delete_weight(self, object_name, weight_name):
        """Removes a weight from an existing object.

        Args:
            object_name: String, name or path of the
                object from which to remove the weight
                (e.g. `"dense_2"` or `"layers/dense_2"`).
            weight_name: String, name of the weight to
                delete (e.g. `"0"`).
        """

        def delete_weight_fn(weights_dict, source_name, target_name=None):
            if weight_name not in weights_dict[source_name]:
                raise ValueError(
                    f"Weight {weight_name} not found "
                    f"in object {object_name}. "
                    "Weights found: "
                    f"{list(weights_dict[source_name].keys())}"
                )
            weights_dict[source_name].pop(weight_name)

        self._edit_object(delete_weight_fn, object_name)

    def add_weights(self, object_name, weights):
        """Add one or more new weights to an existing object.

        Args:
            object_name: String, name or path of the
                object to add the weights to
                (e.g. `"dense_2"` or `"layers/dense_2"`).
            weights: Dict mapping weight names to weight
                values (arrays),
                e.g. `{"0": kernel_value, "1": bias_value}`.
        """
        if not isinstance(weights, dict):
            raise ValueError(
                "Argument `weights` should be a dict "
                "where keys are weight names (usually '0', '1', etc.) "
                "and values are NumPy arrays. "
                f"Received: type(weights)={type(weights)}"
            )

        def add_weight_fn(weights_dict, source_name, target_name=None):
            weights_dict[source_name].update(weights)

        self._edit_object(add_weight_fn, object_name)

    def save(self, filepath):
        """Save the edited weights file.

        Args:
            filepath: Path to save the file to.
                Must be a `.weights.h5` file.
        """
        filepath = str(filepath)
        if not filepath.endswith(".weights.h5"):
            raise ValueError(
                "Invalid `filepath` argument: "
                "expected a `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )
        weights_store = H5IOStore(filepath, mode="w")

        def _save(weights_dict, weights_store, inner_path):
            vars_to_create = {}
            for name, value in weights_dict.items():
                if isinstance(value, dict):
                    if value:
                        _save(
                            weights_dict[name],
                            weights_store,
                            inner_path=os.path.join(inner_path, name),
                        )
                else:
                    # e.g. name="0", value=HDF5Dataset
                    vars_to_create[name] = value
            if vars_to_create:
                var_store = weights_store.make(inner_path)
                for name, value in vars_to_create.items():
                    var_store[name] = value

        _save(self.weights_dict, weights_store, inner_path="")
        weights_store.close()

    def resave_weights(self, filepath):
        self.save(filepath)

    def _extract_weights_from_store(self, data, metadata=None, inner_path=""):
        metadata = metadata or {}

        # ------------------------------------------------------
        # Collect metadata for this HDF5 group
        # ------------------------------------------------------
        object_metadata = {}
        for k, v in data.attrs.items():
            object_metadata[k] = v
        if object_metadata:
            metadata[inner_path] = object_metadata

        result = collections.OrderedDict()

        # ------------------------------------------------------
        # Iterate over all keys in this HDF5 group
        # ------------------------------------------------------
        for key in data.keys():
            # IMPORTANT:
            # Never mutate inner_path; use local variable.
            current_inner_path = f"{inner_path}/{key}"

            # Reject HDF5 `ExternalLink`/`SoftLink`.
            child_class = data.get(
                key, default=None, getclass=True, getlink=True
            )
            if child_class in (h5py.ExternalLink, h5py.SoftLink):
                raise ValueError(
                    f"Not allowed: H5 file with {child_class.__name__}"
                )

            value = data[key]

            # ------------------------------------------------------
            # CASE 1 — HDF5 GROUP → RECURSE
            # ------------------------------------------------------
            if isinstance(value, h5py.Group):
                # Skip empty groups
                if len(value) == 0:
                    continue

                # Recurse into "vars" subgroup when present
                if "vars" in value.keys():
                    vars_group = saving_lib.safe_get_h5_group(value, "vars")
                    # Skip empty "vars" groups
                    if len(vars_group) == 0:
                        continue
                    result[key], metadata = self._extract_weights_from_store(
                        vars_group,
                        metadata=metadata,
                        inner_path=current_inner_path,
                    )
                else:
                    # Recurse normally
                    result[key], metadata = self._extract_weights_from_store(
                        value,
                        metadata=metadata,
                        inner_path=current_inner_path,
                    )

                continue  # finished processing this key

            # ------------------------------------------------------
            # CASE 2 — HDF5 DATASET → SAFE LOADING
            # ------------------------------------------------------

            # Skip any objects that are not proper datasets
            if not isinstance(value, h5py.Dataset):
                continue

            if value.external:
                raise ValueError(
                    "Not allowed: H5 file Dataset with external links: "
                    f"{value.external}"
                )

            if value.is_virtual:
                raise ValueError(
                    "Not allowed: H5 file with virtual Dataset at "
                    f"{current_inner_path}"
                )

            shape = value.shape
            dtype = value.dtype

            # ------------------------------------------------------
            # Validate SHAPE (avoid malformed / malicious metadata)
            # ------------------------------------------------------

            # No negative dimensions
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    "Malformed HDF5 dataset shape encountered in .keras file; "
                    "negative dimension detected."
                )

            # Prevent absurdly high-rank tensors
            if len(shape) > 64:
                raise ValueError(
                    "Malformed HDF5 dataset shape encountered in .keras file; "
                    "tensor rank exceeds safety limit."
                )

            # Safe product computation (Python int is unbounded)
            num_elems = math.prod(shape)

            # ------------------------------------------------------
            # Validate TOTAL memory size
            # ------------------------------------------------------
            MAX_BYTES = 1 << 32  # 4 GiB

            size_bytes = num_elems * dtype.itemsize

            if size_bytes > MAX_BYTES:
                raise ValueError(
                    f"HDF5 dataset too large to load safely "
                    f"({size_bytes} bytes; limit is {MAX_BYTES})."
                )

            # ------------------------------------------------------
            # SAFE — load dataset (guaranteed ≤ 4 GiB)
            # ------------------------------------------------------
            result[key] = value[()]

        # ------------------------------------------------------
        # Return final tree and metadata
        # ------------------------------------------------------
        return result, metadata

    def _generate_filepath_info(self, rich_style=False):
        if rich_style:
            filepath = f"'{self.filepath}'"
            filepath = f"{summary_utils.highlight_symbol(filepath)}"
        else:
            filepath = f"'{self.filepath}'"
        return f"Keras model file {filepath}"

    def _generate_config_info(self, rich_style=False):
        return pprint.pformat(self.config)

    def _generate_metadata_info(self, rich_style=False):
        version = self.metadata["keras_version"]
        date = self.metadata["date_saved"]
        if rich_style:
            version = f"{summary_utils.highlight_symbol(version)}"
            date = f"{summary_utils.highlight_symbol(date)}"
        return f"Saved with Keras {version} - date: {date}"

    def _print_weights_structure(
        self, weights_dict, indent=0, is_first=True, prefix="", inner_path=""
    ):
        for idx, (key, value) in enumerate(weights_dict.items()):
            inner_path = os.path.join(inner_path, key)
            is_last = idx == len(weights_dict) - 1
            if is_first:
                is_first = False
                connector = "> "
            elif is_last:
                connector = "└─ "
            else:
                connector = "├─ "

            if isinstance(value, dict):
                bold_key = summary_utils.bold_text(key)
                object_label = f"{prefix}{connector}{bold_key}"
                if inner_path in self.object_metadata:
                    metadata = self.object_metadata[inner_path]
                    if "name" in metadata:
                        name = metadata["name"]
                        object_label += f" ('{name}')"
                self.console.print(object_label)
                if is_last:
                    appended = "    "
                else:
                    appended = "│   "
                new_prefix = prefix + appended
                self._print_weights_structure(
                    value,
                    indent + 1,
                    is_first=is_first,
                    prefix=new_prefix,
                    inner_path=inner_path,
                )
            else:
                if hasattr(value, "shape"):
                    bold_key = summary_utils.bold_text(key)
                    self.console.print(
                        f"{prefix}{connector}{bold_key}:"
                        + f" shape={value.shape}, dtype={value.dtype}"
                    )
                else:
                    self.console.print(f"{prefix}{connector}{key}: {value}")

    def _weights_summary_cli(self):
        self.console.print("Weights structure")
        self._print_weights_structure(self.weights_dict, prefix=" " * 2)

    def _weights_summary_interactive(self):
        def _generate_html_weights(dictionary, margin_left=0, font_size=1):
            html = ""
            for key, value in dictionary.items():
                if isinstance(value, dict) and value:
                    weights_html = _generate_html_weights(
                        value, margin_left + 20, font_size - 1
                    )
                    html += (
                        f'<details style="margin-left: {margin_left}px;">'
                        '<summary style="'
                        f"font-size: {font_size}em; "
                        "font-weight: bold;"
                        f'">{key}</summary>'
                        f"{weights_html}"
                        "</details>"
                    )
                else:
                    html += (
                        f'<details style="margin-left: {margin_left}px;">'
                        f'<summary style="font-size: {font_size}em;">'
                        f"{key} : shape={value.shape}"
                        f", dtype={value.dtype}</summary>"
                        f"<div style="
                        f'"margin-left: {margin_left}px;'
                        f'"margin-top: {margin_left}px;">'
                        f"{display_weight(value)}"
                        "</div>"
                        "</details>"
                    )
            return html

        output = "Weights structure"

        initialize_id_counter()
        output += _generate_html_weights(self.weights_dict)
        ipython.display.display(ipython.display.HTML(output))


def get_weight_spec_of_saveable(saveable, spec, visited_saveables=None):
    from keras.src.saving.keras_saveable import KerasSaveable

    visited_saveables = visited_saveables or set()

    # If the saveable has already been saved, skip it.
    if id(saveable) in visited_saveables:
        return

    if hasattr(saveable, "save_own_variables"):
        store = {}
        saveable.save_own_variables(store)
        if store:
            keys = sorted(store.keys())
            for k in keys:
                val = store[k]
                spec[k] = backend.KerasTensor(shape=val.shape, dtype=val.dtype)

    visited_saveables.add(id(saveable))

    for child_attr, child_obj in saving_lib._walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            sub_spec = {}
            get_weight_spec_of_saveable(
                child_obj,
                sub_spec,
                visited_saveables=visited_saveables,
            )
            if sub_spec:
                spec[child_attr] = sub_spec
        elif isinstance(child_obj, (list, dict, tuple, set)):
            sub_spec = {}
            get_weight_spec_of_container(
                child_obj,
                sub_spec,
                visited_saveables=visited_saveables,
            )
            if sub_spec:
                spec[child_attr] = sub_spec


def get_weight_spec_of_container(container, spec, visited_saveables):
    from keras.src.saving.keras_saveable import KerasSaveable

    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            name = naming.to_snake_case(saveable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            sub_spec = {}
            get_weight_spec_of_saveable(
                saveable,
                sub_spec,
                visited_saveables=visited_saveables,
            )
            if sub_spec:
                spec[name] = sub_spec


def initialize_id_counter():
    global div_id_counter
    div_id_counter = 0


def increment_id_counter():
    global div_id_counter
    div_id_counter += 1


def get_id_counter():
    return div_id_counter


def display_weight(weight, axis=-1, threshold=16):
    def _find_factors_closest_to_sqrt(num):
        sqrt_num = int(np.sqrt(num))

        for i in range(sqrt_num, 0, -1):
            if num % i == 0:
                M = i
                N = num // i

                if M > N:
                    return N, M
                return M, N

    def _color_from_rbg(value):
        return f"rgba({value[0]}, {value[1]}, {value[2]}, 1)"

    def _reduce_3d_array_by_mean(arr, n, axis):
        if axis == 2:
            trimmed_arr = arr[:, :, : arr.shape[2] - (arr.shape[2] % n)]
            reshaped = np.reshape(
                trimmed_arr, (arr.shape[0], arr.shape[1], -1, n)
            )
            mean_values = np.mean(reshaped, axis=3)

        elif axis == 1:
            trimmed_arr = arr[:, : arr.shape[1] - (arr.shape[1] % n), :]
            reshaped = np.reshape(
                trimmed_arr, (arr.shape[0], -1, n, arr.shape[2])
            )
            mean_values = np.mean(reshaped, axis=2)

        elif axis == 0:
            trimmed_arr = arr[: arr.shape[0] - (arr.shape[0] % n), :, :]
            reshaped = np.reshape(
                trimmed_arr, (-1, n, arr.shape[1], arr.shape[2])
            )
            mean_values = np.mean(reshaped, axis=1)

        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        return mean_values

    def _create_matrix_html(matrix, subplot_size=840):
        rows, cols, num_slices = matrix.shape

        M, N = _find_factors_closest_to_sqrt(num_slices)

        try:
            from matplotlib import cm
        except ImportError:
            cm = None
        if cm:
            rgb_matrix = cm.jet(matrix)
        else:
            rgb_matrix = (matrix - np.min(matrix)) / (
                np.max(matrix) - np.min(matrix)
            )
            rgb_matrix = np.stack([rgb_matrix, rgb_matrix, rgb_matrix], axis=-1)
        rgb_matrix = (rgb_matrix[..., :3] * 255).astype("uint8")

        subplot_html = ""
        for i in range(num_slices):
            cell_html = ""
            for row in rgb_matrix[..., i, :]:
                for rgb in row:
                    color = _color_from_rbg(rgb)
                    cell_html += (
                        f'<div class="cell" '
                        f'style="background-color: {color};">'
                        f"</div>"
                    )
            subplot_html += f"""
                        <div class="matrix">
                          {cell_html}
                        </div>
                        """

        cell_size = subplot_size // (N * cols)

        increment_id_counter()
        div_id = get_id_counter()

        html_code = f"""
            <div class="unique-container_{div_id}">
                  <style>
                      .unique-container_{div_id} .subplots {{
                      display: inline-grid;
                      grid-template-columns: repeat({N}, 1fr);
                      column-gap: 5px;  /* Minimal horizontal gap */
                      row-gap: 5px;     /* Small vertical gap */
                      margin: 0;
                      padding: 0;
                    }}
                    .unique-container_{div_id} .matrix {{
                      display: inline-grid;
                      grid-template-columns: repeat({cols}, {cell_size}px);
                      grid-template-rows: repeat({rows}, {cell_size}px);
                      gap: 1px;
                      margin: 0;
                      padding: 0;
                    }}
                    .unique-container_{div_id} .cell {{
                      width: {cell_size}px;
                      height: {cell_size}px;
                      display: flex;
                      justify-content: center;
                      align-items: center;
                      font-size: 5px;
                      font-weight: bold;
                      color: white;
                    }}
                     .unique-container_{div_id} {{
                      margin-top: 20px;
                      margin-bottom: 20px;
                    }}
                  </style>
                  <div class="subplots">
                    {subplot_html}
                  </div>
                  </div>
                """

        return html_code

    if weight.ndim == 1:
        weight = weight[..., np.newaxis]

    weight = np.swapaxes(weight, axis, -1)
    weight = weight.reshape(-1, weight.shape[-1])

    M, N = _find_factors_closest_to_sqrt(weight.shape[0])
    weight = weight.reshape(M, N, weight.shape[-1])

    for reduce_axis in [0, 1, 2]:
        if weight.shape[reduce_axis] > threshold:
            weight = _reduce_3d_array_by_mean(
                weight,
                weight.shape[reduce_axis] // threshold,
                axis=reduce_axis,
            )

    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-5)

    html_code = _create_matrix_html(weight)
    return html_code
