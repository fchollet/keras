"""Library for exporting SavedModel for Keras models/layers."""

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.export.export_utils import get_input_signature

# Re-export for backward compatibility (used by tfsm_layer.py)
from keras.src.export.saved_model_export_archive import (  # noqa: F401
    _list_variables_used_by_fns,
)
from keras.src.export.xla_saved_model_archive import XLASavedModelArchive
from keras.src.utils.module_utils import get_pluggable_backend_module

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.export import SavedModelExportArchive
elif backend.backend() == "jax":
    from keras.src.backend.jax.export import SavedModelExportArchive
elif backend.backend() == "torch":
    from keras.src.backend.torch.export import SavedModelExportArchive
elif backend.backend() == "numpy":
    from keras.src.export.saved_model_export_archive import (
        BaseSavedModelExportArchive as SavedModelExportArchive,
    )
else:
    backend_export_module = get_pluggable_backend_module(
        "src.export", allow_missing=True
    )
    if backend_export_module is not None:
        SavedModelExportArchive = backend_export_module.SavedModelExportArchive
    else:
        from keras.src.export.saved_model_export_archive import (
            BaseSavedModelExportArchive as SavedModelExportArchive,
        )


DEFAULT_ENDPOINT_NAME = "serve"


def export_saved_model(
    model, filepath, verbose=None, input_signature=None, **kwargs
):
    """Export the model as a TensorFlow SavedModel artifact for inference.

    This method lets you export a model to a lightweight SavedModel artifact
    that contains the model's forward pass only (its `call()` method)
    and can be served via e.g. TensorFlow Serving. The forward pass is
    registered under the name `serve()` (see example below).

    The original code of the model (including any custom layers you may
    have used) is *no longer* necessary to reload the artifact -- it is
    entirely standalone.

    Args:
        filepath: `str` or `pathlib.Path` object. The path to save the artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which uses the default value set by different backends and
            formats.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
            `backend.KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
        **kwargs: Additional keyword arguments:
            - Specific to the JAX backend:
                - `is_static`: Optional `bool`. Indicates whether `fn` is
                    static. Set to `False` if `fn` involves state updates
                    (e.g., RNG seeds).
                - `jax2tf_kwargs`: Optional `dict`. Arguments for
                    `jax2tf.convert`. See [`jax2tf.convert`](
                        https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                    If `native_serialization` and `polymorphic_shapes` are not
                    provided, they are automatically computed.

    **Note:** This feature is currently supported only with TensorFlow, JAX and
    Torch backends. Support for the Torch backend is experimental.

    **Note:** The dynamic shape feature is not yet supported with Torch
    backend. As a result, you must fully define the shapes of the inputs using
    `input_signature`. If `input_signature` is not provided, all instances of
    `None` (such as the batch size) will be replaced with `1`.

    Example:

    ```python
    # Export the model as a TensorFlow SavedModel artifact
    model.export("path/to/location", format="tf_saved_model")

    # Load the artifact in a different process/environment
    reloaded_artifact = tf.saved_model.load("path/to/location")
    predictions = reloaded_artifact.serve(input_data)
    ```

    If you would like to customize your serving endpoints, you can
    use the lower-level `keras.export.ExportArchive` class. The
    `export()` method relies on `ExportArchive` internally.
    """
    if verbose is None:
        verbose = True  # Defaults to `True` for all backends.
    export_archive = ExportArchive()
    if input_signature is None:
        input_signature = get_input_signature(model)

    export_archive.track_and_add_endpoint(
        DEFAULT_ENDPOINT_NAME, model, input_signature, **kwargs
    )
    export_archive.write_out(filepath, verbose=verbose)


@keras_export("keras.export.ExportArchive")
class ExportArchive:
    """ExportArchive is used to write SavedModel artifacts for inference.

    If you have a Keras model or layer that you want to export as SavedModel for
    serving (e.g. via TensorFlow-Serving), you can use `ExportArchive`
    to configure the different serving endpoints you need to make available,
    as well as their signatures. Simply instantiate an `ExportArchive`,
    use `track()` to register the layer(s) or model(s) to be used,
    then use the `add_endpoint()` method to register a new serving endpoint.
    When done, use the `write_out()` method to save the artifact.

    The resulting artifact is a SavedModel and can be reloaded via
    `tf.saved_model.load`.

    Examples:

    Here's how to export a model for inference.

    ```python
    export_archive = ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        name="serve",
        fn=model.call,
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.write_out("path/to/location")

    # Elsewhere, we can reload the artifact and serve it.
    # The endpoint we added is available as a method:
    serving_model = tf.saved_model.load("path/to/location")
    outputs = serving_model.serve(inputs)
    ```

    Here's how to export a model with one endpoint for inference and one
    endpoint for a training-mode forward pass (e.g. with dropout on).

    ```python
    export_archive = ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        name="call_inference",
        fn=lambda x: model.call(x, training=False),
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.add_endpoint(
        name="call_training",
        fn=lambda x: model.call(x, training=True),
        input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
    )
    export_archive.write_out("path/to/location")
    ```

    **Note on resource tracking:**

    `ExportArchive` is able to automatically track all `keras.Variables` used
    by its endpoints, so most of the time calling `.track(model)`
    is not strictly required. However, if your model uses lookup layers such
    as `IntegerLookup`, `StringLookup`, or `TextVectorization`,
    it will need to be tracked explicitly via `.track(model)`.

    Explicit tracking is also required if you need to be able to access
    the properties `variables`, `trainable_variables`, or
    `non_trainable_variables` on the revived archive.
    """

    def __new__(cls, format="saved_model", **kwargs):
        if format == "saved_model":
            return SavedModelExportArchive()
        elif format == "xla_saved_model":
            return XLASavedModelArchive()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def track(self, resource):
        """Track the variables (of a layer or model) and other assets.

        By default, all variables used by an endpoint function are automatically
        tracked when you call `add_endpoint()`. However, non-variables assets
        such as lookup tables need to be tracked manually. Note that lookup
        tables used by built-in Keras layers (`TextVectorization`,
        `IntegerLookup`, `StringLookup`) are automatically tracked by
        `add_endpoint()`.

        Args:
            resource: A layer, model or a TensorFlow trackable resource.
        """
        raise NotImplementedError(
            "track() is not implemented for this backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        """Register a new serving endpoint.

        Args:
            name: `str`. The name of the endpoint.
            fn: A callable. It should only leverage resources
                (e.g. `keras.Variable` objects or `tf.lookup.StaticHashTable`
                objects) that are available on the models/layers tracked by the
                `ExportArchive` (you can call `.track(model)` to track a new
                model).
                The shape and dtype of the inputs to the function must be
                known. For that purpose, you can either 1) make sure that `fn`
                is a `tf.function` that has been called at least once, or 2)
                provide an `input_signature` argument that specifies the shape
                and dtype of the inputs (see below).
            input_signature: Optional. Specifies the shape and dtype of `fn`.
                Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
                `backend.KerasTensor`, or backend tensor (see below for an
                example showing a `Functional` model with 2 input arguments). If
                not provided, `fn` must be a `tf.function` that has been called
                at least once. Defaults to `None`.
            **kwargs: Additional keyword arguments:
                - Specific to the JAX backend:
                    - `is_static`: Optional `bool`. Indicates whether `fn` is
                        static. Set to `False` if `fn` involves state updates
                        (e.g., RNG seeds).
                    - `jax2tf_kwargs`: Optional `dict`. Arguments for
                        `jax2tf.convert`. See [`jax2tf.convert`](
                            https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                        If `native_serialization` and `polymorphic_shapes` are
                        not provided, they are automatically computed.

        Returns:
            The `tf.function` wrapping `fn` that was added to the archive.

        Example:

        Adding an endpoint using the `input_signature` argument when the
        model has a single input argument:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        )
        ```

        Adding an endpoint using the `input_signature` argument when the
        model has two positional input arguments:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                keras.InputSpec(shape=(None, 3), dtype="float32"),
                keras.InputSpec(shape=(None, 4), dtype="float32"),
            ],
        )
        ```

        Adding an endpoint using the `input_signature` argument when the
        model has one input argument that is a list of 2 tensors (e.g.
        a Functional model with 2 inputs):

        ```python
        model = keras.Model(inputs=[x1, x2], outputs=outputs)

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                [
                    keras.InputSpec(shape=(None, 3), dtype="float32"),
                    keras.InputSpec(shape=(None, 4), dtype="float32"),
                ],
            ],
        )
        ```

        This also works with dictionary inputs:

        ```python
        model = keras.Model(inputs={"x1": x1, "x2": x2}, outputs=outputs)

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[
                {
                    "x1": keras.InputSpec(shape=(None, 3), dtype="float32"),
                    "x2": keras.InputSpec(shape=(None, 4), dtype="float32"),
                },
            ],
        )
        ```

        Adding an endpoint that is a `tf.function`:

        ```python
        @tf.function()
        def serving_fn(x):
            return model(x)

        # The function must be traced, i.e. it must be called at least once.
        serving_fn(tf.random.normal(shape=(2, 3)))

        export_archive = ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(name="serve", fn=serving_fn)
        ```

        Combining a model with some TensorFlow preprocessing, which can use
        TensorFlow resources:

        ```python
        lookup_table = tf.lookup.StaticHashTable(initializer, default_value=0.0)

        export_archive = ExportArchive()
        model_fn = export_archive.track_and_add_endpoint(
            "model_fn",
            model,
            input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)],
        )
        export_archive.track(lookup_table)

        @tf.function()
        def serving_fn(x):
            x = lookup_table.lookup(x)
            return model_fn(x)

        export_archive.add_endpoint(name="serve", fn=serving_fn)
        ```
        """
        raise NotImplementedError(
            "add_endpoint() is not implemented for this backend."
        )

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        """Track the variables and register a new serving endpoint.

        This function combines the functionality of `track` and `add_endpoint`.
        It tracks the variables of the `resource` (either a layer or a model)
        and registers a serving endpoint using `resource.__call__`.

        Args:
            name: `str`. The name of the endpoint.
            resource: A trackable Keras resource, such as a layer or model.
            input_signature: Optional. Specifies the shape and dtype of `fn`.
                Can be a structure of `keras.InputSpec`, `tf.TensorSpec`,
                `backend.KerasTensor`, or backend tensor (see below for an
                example showing a `Functional` model with 2 input arguments). If
                not provided, `fn` must be a `tf.function` that has been called
                at least once. Defaults to `None`.
            **kwargs: Additional keyword arguments:
                - Specific to the JAX backend:
                    - `is_static`: Optional `bool`. Indicates whether `fn` is
                        static. Set to `False` if `fn` involves state updates
                        (e.g., RNG seeds).
                    - `jax2tf_kwargs`: Optional `dict`. Arguments for
                        `jax2tf.convert`. See [`jax2tf.convert`](
                            https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                        If `native_serialization` and `polymorphic_shapes` are
                        not provided, they are automatically computed.

        """
        raise NotImplementedError(
            "track_and_add_endpoint() is not implemented for this backend."
        )

    def add_variable_collection(self, name, variables):
        """Register a set of variables to be retrieved after reloading.

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `keras.Variable` instances.

        Example:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        # Register an endpoint
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[keras.InputSpec(shape=(None, 3), dtype="float32")],
        )
        # Save a variable collection
        export_archive.add_variable_collection(
            name="optimizer_variables", variables=model.optimizer.variables)
        export_archive.write_out("path/to/location")

        # Reload the object
        revived_object = tf.saved_model.load("path/to/location")
        # Retrieve the variables
        optimizer_variables = revived_object.optimizer_variables
        ```
        """
        raise NotImplementedError(
            "add_variable_collection() is not implemented for this backend."
        )

    def write_out(self, filepath, options=None, verbose=True):
        """Write the corresponding SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the artifact.
            options: `tf.saved_model.SaveOptions` object that specifies
                SavedModel saving options.
            verbose: whether to print all the variables of an
                exported SavedModel.

        **Note on TF-Serving**: all endpoints registered via `add_endpoint()`
        are made visible for TF-Serving in the SavedModel artifact. In addition,
        the first endpoint registered is made visible under the alias
        `"serving_default"` (unless an endpoint with the name
        `"serving_default"` was already registered manually),
        since TF-Serving requires this endpoint to be set.
        """
        raise NotImplementedError(
            "write_out() is not implemented for this backend."
        )
