"""XLA SavedModel export archive, backed by the `xla_saved_model` package.

An XLA SavedModel is the model format consumed by the XLA Serving Engine
(https://github.com/openxla/xla-serving). The export uses `jax.export` to
capture the StableHLO of each endpoint and an Orbax checkpoint for the
weights, so it is only available with the JAX backend.
"""

import itertools
import re
import string

from keras.src import backend
from keras.src import layers
from keras.src import tree
from keras.src.export.export_utils import make_input_spec
from keras.src.utils.module_utils import LazyModule

xla_saved_model = LazyModule(
    "xla_saved_model",
    pip_name="xla-saved-model",
    import_error_msg=(
        "Exporting with `format='xla_saved_model'` requires the "
        "`xla_saved_model` package from https://github.com/openxla/xla-serving."
    ),
)

DEFAULT_MODEL_NAME = "keras_model"
DEFAULT_SERVING_SIGNATURE = "serving_default"
# Orbax checkpoint of `add_variable_collection()` variables, relative to the
# export directory. `assets.extra` is the XLA SavedModel directory for
# auxiliary files not referenced by the manifest (e.g. warmup requests).
VARIABLE_COLLECTIONS_SUBDIR = "assets.extra/variable_collections"
_BATCH_DIM_NAME = "batch"
_AUTO_TENSOR_NAME = re.compile(r"keras_tensor(_\d+)?")

_SUPPORTED_ENDPOINT_KWARGS = (
    "platforms",
    "jax_export_kwargs",
    "polymorphic_shapes",
    "polymorphic_constraints",
    "batch_options",
    "warmup_examples",
)


class XLASavedModelArchive:
    """Export archive that writes an XLA SavedModel (XLA Serving Engine).

    This is the `format="xla_saved_model"` counterpart of the TF SavedModel
    `ExportArchive`. Each endpoint is exported as a stateless JAX function via
    `jax.export` (StableHLO), while the Keras variables are saved once as an
    Orbax checkpoint. The resulting directory can be served by the XLA Serving
    Engine on CPU, GPU or TPU, or reloaded with `xla_saved_model`.

    Example:

    ```python
    export_archive = keras.export.ExportArchive(format="xla_saved_model")
    export_archive.track(model)
    export_archive.add_endpoint(
        name="serve",
        fn=model.call,
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        platforms=["cpu", "tpu"],
    )
    export_archive.write_out("path/to/location")
    ```

    Endpoints are exported as JAX functions with a single pytree argument
    whose leaves are named. The names are derived from the `input_signature`:
    a single `dict` argument is used as-is, otherwise the leaves of the
    (positional) signature are named after their `InputSpec`/`KerasTensor`
    name (e.g. `keras.Input(name="x1")`), falling back to `"inputs"` (single
    input) or `"input_{i}"`. Auto-generated names (`keras_tensor_3`) are not
    used. Outputs are named likewise: a `dict` is kept as-is, otherwise the
    output leaves are named `"output_{i}"`.

    Only the JAX backend is supported.
    """

    def __init__(self):
        if backend.backend() != "jax":
            raise NotImplementedError(
                "`ExportArchive(format='xla_saved_model')` is only compatible "
                f"with the JAX backend. Current backend: {backend.backend()}"
            )
        # Fail early with a helpful message if the package is missing.
        xla_saved_model.initialize()

        self._endpoint_names = []
        self._endpoints = {}
        self._tracked = []
        self._trainable_variables = []
        self._non_trainable_variables = []
        self._variable_collections = {}

    @property
    def variables(self):
        return self._trainable_variables + self._non_trainable_variables

    @property
    def trainable_variables(self):
        return list(self._trainable_variables)

    @property
    def non_trainable_variables(self):
        return list(self._non_trainable_variables)

    def track(self, resource):
        """Track the variables of a layer or model.

        All variables used by an endpoint function must belong to a tracked
        layer/model: the endpoint is traced statelessly and the variables are
        provided to it from the exported checkpoint.

        Args:
            resource: A Keras `Layer` or `Model`.
        """
        if not isinstance(resource, layers.Layer):
            raise ValueError(
                "Invalid resource type. Expected a Keras `Layer` or `Model`. "
                f"Received object {resource} of type '{type(resource)}'."
            )
        if not resource.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )
        self._tracked.append(resource)
        seen = set(id(v) for v in self.variables)
        for v in resource.trainable_variables:
            if id(v) not in seen:
                self._trainable_variables.append(v)
                seen.add(id(v))
        for v in resource.non_trainable_variables:
            if id(v) not in seen:
                self._non_trainable_variables.append(v)
                seen.add(id(v))

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        """Register a new serving endpoint.

        Args:
            name: `str`. The name of the endpoint. It is used as the serving
                signature name in the exported model.
            fn: A callable using only variables of tracked layers/models.
                It is called with positional arguments matching
                `input_signature`.
            input_signature: Specifies the shape and dtype of the positional
                arguments of `fn`. A list/tuple whose elements are structures
                of `keras.InputSpec`, `keras.KerasTensor`, `tf.TensorSpec` or
                backend tensors. `None` dimensions are exported as symbolic
                (polymorphic) dimensions. Required.
            **kwargs: Additional keyword arguments:
                - `platforms`: Optional sequence of lowering platforms among
                    `"cpu"`, `"cuda"` and `"tpu"`. Defaults to the platform
                    of the current JAX backend.
                - `jax_export_kwargs`: Optional `dict` of extra arguments for
                    `jax.export.export`.
                - `polymorphic_shapes`: Optional structure matching
                    `input_signature` of `jax.export` symbolic shape strings
                    (e.g. `"batch, 3"`). Computed from the `None` dimensions
                    of `input_signature` when not provided.
                - `polymorphic_constraints`: Optional sequence of constraints
                    on the symbolic dimensions (see
                    `jax.export.symbolic_args_specs`).
                - `batch_options`: Optional `BatchOptions` proto configuring
                    server-side dynamic batching for this endpoint.
                - `warmup_examples`: Optional sequence of warmup requests for
                    this endpoint. Each example is a `dict` mapping the input
                    names to NumPy arrays.

        Returns:
            A callable with the same positional arguments as `fn` that runs
            the exported (stateless) function with the current variable
            values.
        """
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")
        if input_signature is None:
            raise ValueError(
                "You must provide an `input_signature` argument to specify "
                "the shape and dtype of the function arguments when exporting "
                "an XLA SavedModel. Example:\n\n"
                "export_archive.add_endpoint(\n"
                "    name='call',\n"
                "    fn=model.call,\n"
                "    input_signature=[\n"
                "        keras.InputSpec(\n"
                "            shape=(None, 224, 224, 3),\n"
                "            dtype='float32',\n"
                "        )\n"
                "    ],\n"
                ")"
            )
        unsupported = set(kwargs) - set(_SUPPORTED_ENDPOINT_KWARGS)
        if unsupported:
            raise ValueError(
                f"Unsupported keyword arguments for XLA SavedModel export: "
                f"{sorted(unsupported)}. Supported arguments are: "
                f"{list(_SUPPORTED_ENDPOINT_KWARGS)}."
            )
        if not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]
        input_signature = list(input_signature)

        import jax

        # Build the structure of positional arguments (as `InputSpec`s) and
        # the flat, named structure that the serving layer will see.
        spec_structure = tree.map_structure(_make_input_spec, input_signature)
        for spec in tree.flatten(spec_structure):
            if spec.shape is None:
                raise ValueError(
                    "XLA SavedModel export requires inputs with a known rank "
                    "(static number of dimensions); individual dimensions may "
                    f"be `None`. Received input spec: {spec}"
                )
        names, is_single_dict = _get_input_names(spec_structure)
        polymorphic_shapes = kwargs.pop("polymorphic_shapes", None)
        if polymorphic_shapes is None:
            polymorphic_shapes = _to_polymorphic_shapes(spec_structure)
        else:
            if not isinstance(polymorphic_shapes, (list, tuple)):
                polymorphic_shapes = [polymorphic_shapes]
            tree.assert_same_structure(spec_structure, list(polymorphic_shapes))

        def to_serving_structure(structure):
            if is_single_dict:
                return structure[0]
            return dict(zip(names, tree.flatten(structure)))

        def from_serving_structure(structure):
            if is_single_dict:
                return [structure]
            return tree.pack_sequence_as(
                spec_structure, [structure[n] for n in names]
            )

        # `jax.ShapeDtypeStruct` rejects `None` dimensions, so the serving
        # signature uses the symbolic dimension names instead.
        serving_input_signature = to_serving_structure(
            tree.map_structure(
                lambda spec, poly: jax.ShapeDtypeStruct(
                    shape=_symbolic_shape(spec.shape, poly), dtype=spec.dtype
                ),
                spec_structure,
                list(polymorphic_shapes),
            )
        )
        serving_polymorphic_shapes = to_serving_structure(polymorphic_shapes)

        def stateless_fn(params, inputs):
            # `params` is the flat list of variable values, in the order of
            # `self.variables` at `write_out()` time.
            state_mapping = zip(self.variables, params)
            with backend.StatelessScope(state_mapping=state_mapping):
                return fn(*from_serving_structure(inputs))

        def apply_fn(params, inputs):
            # The serving layer addresses outputs by name, so outputs that are
            # not already a dict are flattened into `output_{i}` entries.
            return _to_serving_outputs(stateless_fn(params, inputs))

        jax_export_kwargs = dict(kwargs.pop("jax_export_kwargs", None) or {})
        platforms = kwargs.pop("platforms", None)
        if platforms is not None:
            jax_export_kwargs["platforms"] = list(platforms)
        elif "platforms" not in jax_export_kwargs:
            jax_export_kwargs["platforms"] = [_default_platform()]

        apply_fn_bundle = xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,
            input_polymorphic_shapes=serving_polymorphic_shapes,
            polymorphic_constraints=tuple(
                kwargs.pop("polymorphic_constraints", None) or ()
            ),
            jax_export_kwargs=jax_export_kwargs,
        )
        self._endpoints[name] = {
            "apply_fn_bundle": apply_fn_bundle,
            "input_signature": serving_input_signature,
            "batch_options": kwargs.pop("batch_options", None),
            "warmup_examples": kwargs.pop("warmup_examples", None),
            "positional_signature": spec_structure,
            # Layer/model `fn` is a bound method of, if any (see
            # `_check_endpoint_owners_are_tracked`).
            "owner": getattr(fn, "__self__", None),
        }
        self._endpoint_names.append(name)

        def endpoint_fn(*args):
            return stateless_fn(self._get_params(), to_serving_structure(args))

        return endpoint_fn

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        """Track the variables and register a new serving endpoint.

        This function combines the functionality of `track` and `add_endpoint`.
        It tracks the variables of the `resource` (either a layer or a model)
        and registers a serving endpoint using `resource.__call__`.

        Args:
            name: `str`. The name of the endpoint.
            resource: A trackable Keras resource, such as a layer or model.
            input_signature: Specifies the shape and dtype of the inputs. See
                `add_endpoint()`.
            **kwargs: Additional keyword arguments. See `add_endpoint()`.
        """
        self.track(resource)
        return self.add_endpoint(
            name, resource.__call__, input_signature, **kwargs
        )

    def add_variable_collection(self, name, variables):
        """Register a set of variables to be retrieved after reloading.

        The XLA SavedModel manifest only references the weights used by the
        exported endpoints, which the server loads for every request. Variable
        collections are therefore written to a separate Orbax checkpoint at
        `<filepath>/assets.extra/variable_collections`, as a `dict` mapping the
        collection name to the list of variable values. It is ignored by the
        XLA Serving Engine and can be reloaded with Orbax:

        ```python
        import orbax.checkpoint as ocp

        export_archive = ExportArchive(format="xla_saved_model")
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        )
        export_archive.add_variable_collection(
            name="optimizer_variables", variables=model.optimizer.variables)
        export_archive.write_out("path/to/location")

        # Retrieve the variables
        collections = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
            "path/to/location/assets.extra/variable_collections"
        )
        optimizer_variables = collections["optimizer_variables"]
        ```

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `keras.Variable` instances.
        """
        if not isinstance(variables, (list, tuple, set)):
            raise ValueError(
                "Expected `variables` to be a list/tuple/set. "
                f"Received instead object of type '{type(variables)}'."
            )
        if not all(isinstance(v, backend.Variable) for v in variables):
            raise ValueError(
                "Expected all elements in `variables` to be "
                "`keras.Variable` instances. Found instead the following "
                f"types: {list(set(type(v) for v in variables))}"
            )
        if name in self._variable_collections:
            raise ValueError(f"Variable collection '{name}' is already taken.")
        self._variable_collections[name] = list(variables)

    def write_out(self, filepath, options=None, verbose=True):
        """Write the XLA SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Directory where to save the artifact.
            options: Optional `dict` of extra keyword arguments forwarded to
                `xla_saved_model.save()` (e.g. `{"save_shlo_to_file": True}`).
            verbose: whether to print the exported endpoints.

        **Note on serving**: all endpoints registered via `add_endpoint()`
        are exported as serving signatures. In addition, the first endpoint
        registered is also exported under the alias `"serving_default"`
        (unless an endpoint with that name was registered manually), which is
        the signature the XLA Serving Engine uses by default.
        """
        from keras.src.utils import io_utils

        if not self._endpoint_names:
            raise ValueError(
                "No endpoints have been set yet. Call add_endpoint()."
            )
        if options is not None and not isinstance(options, dict):
            raise ValueError(
                "`options` must be a dict of keyword arguments for "
                f"`xla_saved_model.save()`. Received: {options}"
            )
        filepath = str(filepath)

        signature_to_endpoint = {name: name for name in self._endpoint_names}
        if DEFAULT_SERVING_SIGNATURE not in signature_to_endpoint:
            signature_to_endpoint[DEFAULT_SERVING_SIGNATURE] = (
                self._endpoint_names[0]
            )

        model = xla_saved_model.JaxModel(
            apply_fn_bundle={
                name: self._endpoints[name]["apply_fn_bundle"]
                for name in self._endpoint_names
            },
            params=self._get_params(),
            name=DEFAULT_MODEL_NAME,
        )

        serving_configs = []
        warmup_examples = {}
        multiple_methods = len(self._endpoint_names) > 1
        for signature, endpoint_name in signature_to_endpoint.items():
            endpoint = self._endpoints[endpoint_name]
            orchestration = None
            if multiple_methods:
                orchestration = xla_saved_model.serving_config.Orchestration(
                    model_calls=[
                        xla_saved_model.serving_config.ModelCall(
                            model_name=DEFAULT_MODEL_NAME,
                            method_key=endpoint_name,
                            use_batching=endpoint["batch_options"] is not None,
                        )
                    ]
                )
            serving_configs.append(
                xla_saved_model.ServingConfig(
                    serving_signature=signature,
                    input_signature=endpoint["input_signature"],
                    batch_options=endpoint["batch_options"],
                    orchestration=orchestration,
                )
            )
            if endpoint["warmup_examples"] is not None:
                warmup_examples[signature] = list(endpoint["warmup_examples"])

        xla_saved_model.save(
            model,
            export_path=filepath,
            serving_configs=serving_configs,
            warmup_examples=warmup_examples or None,
            **(options or {}),
        )
        if self._variable_collections:
            self._write_variable_collections(filepath)

        if verbose:
            endpoints = "\n\n".join(
                _print_signature(name, self._endpoints[name])
                for name in self._endpoint_names
            )
            io_utils.print_msg(
                f"Saved artifact at '{filepath}'. "
                "The following endpoints are available:\n\n"
                f"{endpoints}"
            )

    def _get_params(self):
        """Returns the flat list of variable values, matching `self.variables`.

        Unlike the TF SavedModel archive, variables are not inferred from the
        traced functions, so they must be tracked explicitly: under
        `StatelessScope`, an untracked variable silently resolves to its
        concrete value and would be baked into the StableHLO as a constant.
        """
        self._check_endpoint_owners_are_tracked()
        if not self.variables:
            # `xla_saved_model` requires a weights checkpoint: `params=None`
            # produces an artifact the XLA Serving Engine refuses to load
            # ("Object key for weight names ... not found in manifest") and
            # an empty pytree cannot be saved by Orbax ("Found empty item").
            raise ValueError(
                "No variables have been tracked. Call `track(model)` (or "
                "`track_and_add_endpoint()`) before `write_out()`. The XLA "
                "SavedModel format requires a weights checkpoint, so purely "
                "stateless endpoints cannot be exported on their own."
            )
        return [v.value for v in self.variables]

    def _check_endpoint_owners_are_tracked(self):
        """Raises if an endpoint is a method of a layer that is not tracked.

        This catches the common mistake of registering `model.call` /
        `model.__call__` without `track(model)`. Variables used by other
        callables cannot be detected and would be embedded as constants.
        """
        tracked_ids = set(id(v) for v in self.variables)
        for name, endpoint in self._endpoints.items():
            owner = endpoint["owner"]
            if not isinstance(owner, layers.Layer):
                continue
            untracked = [
                v.path for v in owner.variables if id(v) not in tracked_ids
            ]
            if untracked:
                raise ValueError(
                    f"Endpoint '{name}' is a method of {owner.name!r}, but "
                    "the following variables of that layer/model have not "
                    f"been tracked: {untracked}. Call `track({owner.name})` "
                    "before `write_out()` so that the variables are exported "
                    "to the checkpoint instead of being embedded as constants."
                )

    def _write_variable_collections(self, filepath):
        """Saves the variable collections to a dedicated Orbax checkpoint.

        The collections are deliberately kept out of the model `params`: every
        leaf of `params` is registered in the manifest as a weight the XLA
        Serving Engine loads at start-up, and these variables are not used by
        any endpoint.
        """
        import os

        import orbax.checkpoint as ocp

        collections = {
            name: [v.value for v in variables]
            for name, variables in self._variable_collections.items()
        }
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        checkpointer.save(
            os.path.abspath(
                os.path.join(filepath, VARIABLE_COLLECTIONS_SUBDIR)
            ),
            collections,
            force=True,
        )


def _make_input_spec(x):
    """`make_input_spec` that also accepts `tf.TensorSpec`s."""
    if hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "name"):
        # Covers `tf.TensorSpec` without importing TensorFlow.
        if type(x).__name__ == "TensorSpec":
            if x.shape.rank is None:
                # Unknown rank: rejected with a clear error in
                # `add_endpoint()`, since symbolic shapes need a rank.
                shape = None
            else:
                shape = tuple(x.shape.as_list())
            return layers.InputSpec(
                shape=shape, dtype=x.dtype.name, name=x.name
            )
    return make_input_spec(x)


def _get_input_names(spec_structure):
    """Computes the serving input names for a positional signature.

    Returns:
        A tuple `(names, is_single_dict)`. `is_single_dict` is `True` when
        the signature is a single `dict` argument, which is then used as the
        serving structure as-is (and `names` is `None`).
    """
    if len(spec_structure) == 1 and isinstance(spec_structure[0], dict):
        return None, True
    specs = tree.flatten(spec_structure)
    names = [spec.name for spec in specs]
    # Auto-generated `KerasTensor` names (`keras_tensor`, `keras_tensor_12`)
    # are not stable across sessions, so they are not used as serving names.
    names = [
        n
        for n in names
        if isinstance(n, str) and n and not _AUTO_TENSOR_NAME.fullmatch(n)
    ]
    if len(names) == len(specs) and len(set(names)) == len(names):
        return names, False
    if len(specs) == 1:
        return ["inputs"], False
    return [f"input_{i}" for i in range(len(specs))], False


def _to_polymorphic_shapes(spec_structure):
    """Returns `jax.export` symbolic shape strings for a signature.

    The leading `None` dimension is shared across all inputs as `"batch"`;
    other `None` dimensions get a unique symbol each.
    """
    # Generates unique names: a, b, ... z, aa, ab, ... az, ba, ... zz
    dim_names = itertools.chain(
        string.ascii_lowercase,
        itertools.starmap(
            lambda a, b: a + b,
            itertools.product(string.ascii_lowercase, repeat=2),
        ),
    )

    def convert_shape(spec):
        poly_shape = []
        for index, dim in enumerate(spec.shape):
            if dim is not None:
                poly_shape.append(str(dim))
            elif index == 0:
                poly_shape.append(_BATCH_DIM_NAME)
            else:
                poly_shape.append(next(dim_names))
        return ", ".join(poly_shape)

    return tree.map_structure(convert_shape, spec_structure)


def _to_serving_outputs(outputs):
    """Names the outputs of an endpoint for the serving layer.

    A `dict` output is used as-is (its keys are the output names). Any other
    structure is flattened and its leaves are named `output_0`, `output_1`,
    etc., in `keras.tree.flatten` order.
    """
    if isinstance(outputs, dict):
        return outputs
    return {f"output_{i}": leaf for i, leaf in enumerate(tree.flatten(outputs))}


def _symbolic_shape(shape, polymorphic_shape):
    """Replaces `None` dims of `shape` with the symbol in `polymorphic_shape`.

    `jax.ShapeDtypeStruct` accepts symbolic dimension names as strings, which
    is how `xla_saved_model` expects dynamic dimensions in the serving
    signature. Known dims are kept as ints.
    """
    symbols = [
        s.strip() for s in polymorphic_shape.strip().strip("()").split(",")
    ]
    symbols = [s for s in symbols if s]
    if len(symbols) != len(shape):
        raise ValueError(
            f"The polymorphic shape '{polymorphic_shape}' does not match the "
            f"rank of the input shape {tuple(shape)}."
        )
    result = []
    for dim, symbol in zip(shape, symbols):
        if dim is not None:
            result.append(int(dim))
        elif symbol.isdigit():
            result.append(int(symbol))
        else:
            result.append(symbol)
    return tuple(result)


def _default_platform():
    import jax

    platform = jax.default_backend()
    return {"gpu": "cuda"}.get(platform, platform)


def _print_signature(name, endpoint):
    lines = [f"* Endpoint '{name}'"]
    for input_name, spec in sorted(
        tree.flatten_with_path(endpoint["input_signature"])
    ):
        path = "/".join(str(p) for p in input_name)
        lines.append(
            f"    {path}: shape={tuple(spec.shape)}, dtype={spec.dtype}"
        )
    platforms = endpoint["apply_fn_bundle"].jax_export_kwargs.get("platforms")
    lines.append(f"  platforms: {platforms}")
    return "\n".join(lines)
