"""Tests for the XLA SavedModel export archive."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.export import saved_model
from keras.src.export import xla_saved_model_archive

xla_saved_model = xla_saved_model_archive.xla_saved_model


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        model = models.Sequential(layer_list)
        model.build((None,) + input_shape)
        return model
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        model = CustomModel(layer_list)
        model(np.zeros((1,) + input_shape, dtype="float32"))
        return model


def load_endpoint(export_dir, signature):
    """Reloads an exported endpoint as `fn(params, inputs)` plus its params.

    This mirrors how the XLA Serving Engine consumes the artifact: the
    StableHLO function is read from the manifest and the weights from the
    Orbax checkpoint it references.
    """
    import orbax.checkpoint as ocp
    from xla_saved_model._src import core as xsm
    from xla_saved_model._src.xsm2jax import main_lib as xsm2jax
    from xla_saved_model.protos import jax_supplemental_pb2

    manifest = xsm.load(export_dir)
    # Without orchestration (single endpoint) the function is named after the
    # signature; with orchestration (multiple endpoints) it is named
    # `{signature}_{model_name}_{method_key}`.
    prefix = f"{signature}_{xla_saved_model_archive.DEFAULT_MODEL_NAME}_"
    function_names = [
        name
        for name, obj in manifest.objects.items()
        if obj.HasField("function")
        and (name == signature or name.startswith(prefix))
    ]
    if len(function_names) != 1:
        raise ValueError(
            f"Expected exactly one function for signature '{signature}', "
            f"found: {function_names}"
        )
    xsm_function = manifest.objects[function_names[0]].function
    supplemental_path = xsm_function.body.stable_hlo_body.supplemental_info[
        xsm.JAX_SPECIFIC_INFO
    ].file_system_location.string_path
    supplemental = jax_supplemental_pb2.Function()
    with open(os.path.join(export_dir, supplemental_path), "rb") as f:
        supplemental.ParseFromString(f.read())
    apply_fn = xsm2jax.xsm_fn_to_apply_fn(xsm_function, supplemental)

    checkpoint_path = manifest.objects[
        "weights"
    ].value.external.data.file_system_location.string_path
    params = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        os.path.join(export_dir, checkpoint_path)
    )
    return apply_fn, params


def model_runner(export_dir):
    """Loads an exported model with the XLA Serving Engine C++ runtime.

    Returns `None` when `xla_saved_model_runner` is not installed: it is an
    optional binary package, so the runtime round-trip tests are skipped
    rather than failed.
    """
    try:
        from xla_saved_model_runner import model_runner as runner_lib
    except ImportError:
        return None
    return runner_lib.ModelRunner(model_path=export_dir)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="XLA SavedModel export only supports the JAX backend.",
)
@pytest.mark.skipif(
    not xla_saved_model.available,
    reason="Requires the `xla_saved_model` package.",
)
class XLASavedModelArchiveTest(testing.TestCase):
    def _export_dir(self, name="exported_model"):
        return os.path.join(self.get_temp_dir(), name)

    def test_export_archive_factory(self):
        archive = saved_model.ExportArchive(format="xla_saved_model")
        self.assertIsInstance(
            archive, xla_saved_model_archive.XLASavedModelArchive
        )

    @parameterized.named_parameters(
        ("sequential", "sequential"),
        ("functional", "functional"),
        ("subclass", "subclass"),
    )
    def test_standard_model_export(self, model_type):
        export_dir = self._export_dir()
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        endpoint_fn = archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        # The returned callable runs the stateless function with the
        # current variable values.
        self.assertAllClose(endpoint_fn(ref_input), ref_output)

        archive.write_out(export_dir)

        self.assertTrue(os.path.exists(os.path.join(export_dir, "manifest.pb")))
        self.assertTrue(os.path.isdir(os.path.join(export_dir, "checkpoint")))

        for signature in ("serve", "serving_default"):
            apply_fn, params = load_endpoint(export_dir, signature)
            self.assertAllClose(
                apply_fn(params, {"inputs": ref_input})["output_0"], ref_output
            )
        # Symbolic batch dimension: a different batch size also works.
        apply_fn, params = load_endpoint(export_dir, "serve")
        other_input = np.random.normal(size=(5, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": other_input})["output_0"],
            model(other_input),
        )

    def test_track_and_add_endpoint(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        ref_output = model(ref_input)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track_and_add_endpoint(
            "call",
            model,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        self.assertLen(archive.variables, len(model.variables))
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "call")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"], ref_output
        )

    def test_multi_input_model_export(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs=[x1, x2], outputs=output)

        ref_inputs = [
            np.random.normal(size=(3, 3)).astype("float32"),
            np.random.normal(size=(3, 4)).astype("float32"),
        ]
        ref_output = model(ref_inputs)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            # One positional argument that is a list of two tensors.
            input_signature=[
                [
                    layers.InputSpec(
                        shape=(None, 3), dtype="float32", name="x1"
                    ),
                    layers.InputSpec(
                        shape=(None, 4), dtype="float32", name="x2"
                    ),
                ]
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "serve")
        self.assertAllClose(
            apply_fn(params, {"x1": ref_inputs[0], "x2": ref_inputs[1]})[
                "output_0"
            ],
            ref_output,
        )

    def test_dict_input_model_export(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs={"x1": x1, "x2": x2}, outputs=output)

        ref_inputs = {
            "x1": np.random.normal(size=(3, 3)).astype("float32"),
            "x2": np.random.normal(size=(3, 4)).astype("float32"),
        }
        ref_output = model(ref_inputs)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                {
                    "x1": layers.InputSpec(shape=(None, 3), dtype="float32"),
                    "x2": layers.InputSpec(shape=(None, 4), dtype="float32"),
                }
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "serve")
        self.assertAllClose(
            apply_fn(params, ref_inputs)["output_0"], ref_output
        )

    def test_multiple_endpoints(self):
        export_dir = self._export_dir()
        model = get_model("functional")
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        # Compute the references first: the stateful `training=True` call
        # updates the BatchNormalization moving statistics.
        ref_inference = model(ref_input, training=False)
        ref_training = model(ref_input, training=True)

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]
        archive.add_endpoint(
            "call_inference",
            lambda x: model(x, training=False),
            input_signature=input_signature,
            platforms=["cpu"],
        )
        archive.add_endpoint(
            "call_training",
            lambda x: model(x, training=True),
            input_signature=input_signature,
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)

        apply_fn, params = load_endpoint(export_dir, "call_inference")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input, training=False),
        )
        apply_fn, params = load_endpoint(export_dir, "call_training")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"], ref_training
        )
        # The first endpoint is aliased as `serving_default`.
        apply_fn, params = load_endpoint(export_dir, "serving_default")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input, training=False),
        )
        del ref_inference

    def test_warmup_examples_and_options(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
            warmup_examples=[{"inputs": np.zeros((1, 10), dtype="float32")}],
        )
        archive.write_out(
            export_dir, options={"save_shlo_to_file": True}, verbose=False
        )
        self.assertTrue(os.path.exists(os.path.join(export_dir, "manifest.pb")))
        # The warmup requests are written under `assets.extra`.
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    export_dir, "assets.extra", "xla_serving_warmup_requests"
                )
            )
        )

    def test_polymorphic_shapes_override(self):
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            polymorphic_shapes=["b, 10"],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)
        ref_input = np.random.normal(size=(4, 10)).astype("float32")
        apply_fn, params = load_endpoint(export_dir, "serve")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input),
        )

    def test_errors(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        with self.assertRaisesRegex(ValueError, "has not yet been built"):
            archive.track(models.Sequential([layers.Dense(2)]))
        with self.assertRaisesRegex(ValueError, "Invalid resource type"):
            archive.track("not a layer")
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            archive.write_out(self._export_dir())
        with self.assertRaisesRegex(ValueError, "must provide an `input_sig"):
            archive.add_endpoint("serve", model.__call__)
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=input_signature,
                jax2tf_kwargs={},
            )
        archive.track(model)
        archive.add_endpoint(
            "serve", model.__call__, input_signature=input_signature
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            archive.add_endpoint(
                "serve", model.__call__, input_signature=input_signature
            )
        with self.assertRaisesRegex(ValueError, "to be a list/tuple/set"):
            archive.add_variable_collection("vars", model.variables[0])
        with self.assertRaisesRegex(ValueError, "`keras.Variable` instances"):
            archive.add_variable_collection("vars", [np.zeros((2,))])
        archive.add_variable_collection("vars", model.variables)
        with self.assertRaisesRegex(ValueError, "already taken"):
            archive.add_variable_collection("vars", model.variables)
        with self.assertRaisesRegex(ValueError, "`options` must be a dict"):
            archive.write_out(self._export_dir(), options="bad")

    def test_variable_collections(self):
        import orbax.checkpoint as ocp

        export_dir = self._export_dir()
        model = get_model("sequential")
        extra = backend.Variable(np.arange(4, dtype="float32"), name="extra")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        archive.add_variable_collection("my_vars", model.trainable_variables)
        archive.add_variable_collection("extra", [extra])
        archive.write_out(export_dir, verbose=False)

        collections = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
            os.path.join(
                export_dir, xla_saved_model_archive.VARIABLE_COLLECTIONS_SUBDIR
            )
        )
        self.assertEqual(set(collections.keys()), {"my_vars", "extra"})
        self.assertLen(collections["my_vars"], len(model.trainable_variables))
        for restored, variable in zip(
            collections["my_vars"], model.trainable_variables
        ):
            self.assertAllClose(restored, variable.value)
        self.assertAllClose(collections["extra"][0], np.arange(4))

        # The collections are not part of the model weights loaded by the
        # server: the model checkpoint only contains the tracked variables.
        apply_fn, params = load_endpoint(export_dir, "serve")
        self.assertLen(params, len(model.variables))
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input),
        )

    def test_model_export_api(self):
        export_dir = self._export_dir()
        model = get_model("functional")
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        model.export(export_dir, format="xla_saved_model", platforms=["cpu"])

        for signature in ("serve", "serving_default"):
            apply_fn, params = load_endpoint(export_dir, signature)
            self.assertAllClose(
                apply_fn(params, {"inputs": ref_input})["output_0"],
                ref_output,
            )

    def test_untracked_variables_error(self):
        model = get_model("sequential")
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        # A bound method of an untracked model is detected explicitly.
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.add_endpoint("serve", model.__call__, input_signature)
        with self.assertRaisesRegex(ValueError, "have not been tracked"):
            archive.write_out(self._export_dir())

        # Otherwise, exporting without any tracked variables is rejected: the
        # format requires a weights checkpoint.
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.add_endpoint("serve", lambda x: model(x), input_signature)
        with self.assertRaisesRegex(ValueError, "No variables have been"):
            archive.write_out(self._export_dir())

    def test_track_deduplicates_variables(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.track(model)  # Tracking twice must not duplicate variables.
        archive.track(model.layers[0])
        self.assertLen(archive.variables, len(model.variables))
        self.assertLen(
            archive.trainable_variables, len(model.trainable_variables)
        )
        self.assertLen(
            archive.non_trainable_variables,
            len(model.non_trainable_variables),
        )
        self.assertEqual(
            archive.variables,
            archive.trainable_variables + archive.non_trainable_variables,
        )

    def test_single_spec_and_dynamic_non_batch_dims(self):
        # A 3D input with a dynamic sequence length: the non-batch `None` dim
        # gets its own symbol, and the exported function accepts any length.
        export_dir = self._export_dir()
        model = models.Sequential([layers.Dense(4)])
        model.build((None, None, 8))
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        # A single spec (not wrapped in a list) is accepted.
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=layers.InputSpec(
                shape=(None, None, 8), dtype="float32"
            ),
            jax_export_kwargs={"platforms": ["cpu"]},
        )
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "serve")
        for seq_len in (3, 7):
            ref_input = np.random.normal(size=(2, seq_len, 8)).astype("float32")
            self.assertAllClose(
                apply_fn(params, {"inputs": ref_input})["output_0"],
                model(ref_input),
            )

    def test_polymorphic_shapes_single_string_and_concrete_dim(self):
        export_dir = self._export_dir()
        model = models.Sequential([layers.Dense(4)])
        model.build((None, None, 8))
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        # A single polymorphic shape string (not wrapped in a list) is
        # accepted, and can pin a `None` dimension to a concrete size.
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=layers.InputSpec(
                shape=(None, None, 8), dtype="float32"
            ),
            polymorphic_shapes="batch, 5, 8",
            platforms=["cpu"],
        )
        self.assertEqual(
            archive._endpoints["serve"]["input_signature"]["inputs"].shape,
            ("batch", 5, 8),
        )
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "serve")
        ref_input = np.random.normal(size=(2, 5, 8)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input),
        )

    def test_polymorphic_shapes_rank_mismatch_error(self):
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        with self.assertRaisesRegex(ValueError, "does not match the rank"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[
                    layers.InputSpec(shape=(None, 10), dtype="float32")
                ],
                polymorphic_shapes=["batch"],
            )

    def test_backend_tensor_input_signature(self):
        # Backend tensors / KerasTensors can be used as the input signature;
        # their batch dimension is made dynamic.
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[ops.ones((2, 10), dtype="float32")],
            platforms=["cpu"],
        )
        self.assertEqual(
            archive._endpoints["serve"]["input_signature"]["inputs"].shape,
            ("batch", 10),
        )
        archive.write_out(export_dir, verbose=False)
        apply_fn, params = load_endpoint(export_dir, "serve")
        ref_input = np.random.normal(size=(4, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"],
            model(ref_input),
        )

    def test_explicit_serving_default_endpoint(self):
        # No alias is added when `serving_default` is registered manually.
        export_dir = self._export_dir()
        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        archive.add_endpoint(
            "serving_default",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
            platforms=["cpu"],
        )
        archive.write_out(export_dir, verbose=False)

        from xla_saved_model._src import core as xsm

        manifest = xsm.load(export_dir)
        functions = [
            name
            for name, obj in manifest.objects.items()
            if obj.HasField("function")
        ]
        self.assertEqual(functions, ["serving_default"])

    def test_untracked_owner_error(self):
        # Tracking a different model is not enough: the endpoint's own model
        # must be tracked, otherwise its weights would be baked in as
        # constants.
        model = get_model("sequential")
        other = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(other)
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32")
            ],
        )
        with self.assertRaisesRegex(ValueError, "have not been tracked"):
            archive.write_out(self._export_dir())

    def test_unknown_rank_error(self):
        tf = pytest.importorskip("tensorflow")

        model = get_model("sequential")
        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        with self.assertRaisesRegex(ValueError, "known rank"):
            archive.add_endpoint(
                "serve",
                model.__call__,
                input_signature=[tf.TensorSpec(shape=None, dtype="float32")],
            )
        # Scalar and partially-known `tf.TensorSpec`s are accepted.
        archive.add_endpoint(
            "serve",
            lambda x, s: model(x) * s,
            input_signature=[
                tf.TensorSpec(shape=(None, 10), dtype="float32"),
                tf.TensorSpec(shape=(), dtype="float32"),
            ],
        )

    def test_load_and_run_with_model_runner(self):
        """Loads the artifact with the XLA Serving Engine runtime and runs it.

        This exercises the actual serving path (manifest, orchestration,
        checkpoint loading, named inputs/outputs, dynamic batch dimension)
        rather than the Python reload used by the other tests.
        """
        export_dir = self._export_dir()
        model = get_model("functional")
        model.export(export_dir, format="xla_saved_model", platforms=["cpu"])

        runner = model_runner(export_dir)
        if runner is None:
            self.skipTest(
                "`xla_saved_model_runner` is not installed; skipping the "
                "runtime round-trip."
            )
        self.assertEqual(runner.ifrt_platform_name(), "cpu")
        self.assertEqual(
            sorted(runner.signature_names()), ["serve", "serving_default"]
        )

        for batch_size in (3, 5):
            ref_input = np.random.normal(size=(batch_size, 10)).astype(
                "float32"
            )
            ref_output = np.asarray(model(ref_input))
            for signature in ("serve", "serving_default"):
                out = runner.signature(signature).run_with_dict_return(
                    inputs=ref_input
                )
                self.assertEqual(list(out.keys()), ["output_0"])
                self.assertEqual(out["output_0"].dtype, np.float32)
                self.assertEqual(out["output_0"].shape, (batch_size, 1))
                self.assertAllClose(out["output_0"], ref_output)

    def test_load_and_run_multiple_endpoints_with_model_runner(self):
        export_dir = self._export_dir()
        x1 = layers.Input(shape=(3,), name="x1")
        x2 = layers.Input(shape=(4,), name="x2")
        output = layers.Dense(2)(layers.Concatenate()([x1, x2]))
        model = models.Model(inputs={"x1": x1, "x2": x2}, outputs=output)
        ref_inputs = {
            "x1": np.random.normal(size=(3, 3)).astype("float32"),
            "x2": np.random.normal(size=(3, 4)).astype("float32"),
        }
        ref_output = np.asarray(model(ref_inputs))

        archive = saved_model.ExportArchive(format="xla_saved_model")
        archive.track(model)
        input_signature = [
            {
                "x1": layers.InputSpec(shape=(None, 3), dtype="float32"),
                "x2": layers.InputSpec(shape=(None, 4), dtype="float32"),
            }
        ]
        archive.add_endpoint(
            "serve",
            model.__call__,
            input_signature=input_signature,
            platforms=["cpu"],
        )
        # Dict outputs keep their keys as output names.
        archive.add_endpoint(
            "serve_named",
            lambda inputs: {"logits": model(inputs)},
            input_signature=input_signature,
            platforms=["cpu"],
            warmup_examples=[
                {
                    "x1": np.zeros((1, 3), dtype="float32"),
                    "x2": np.zeros((1, 4), dtype="float32"),
                }
            ],
        )
        archive.write_out(export_dir, verbose=False)

        runner = model_runner(export_dir)
        if runner is None:
            self.skipTest(
                "`xla_saved_model_runner` is not installed; skipping the "
                "runtime round-trip."
            )
        self.assertEqual(
            sorted(runner.signature_names()),
            ["serve", "serve_named", "serving_default"],
        )
        out = runner.signature("serve").run_with_dict_return(**ref_inputs)
        self.assertEqual(list(out.keys()), ["output_0"])
        self.assertAllClose(out["output_0"], ref_output)

        out = runner.signature("serve_named").run_with_dict_return(**ref_inputs)
        self.assertEqual(list(out.keys()), ["logits"])
        self.assertAllClose(out["logits"], ref_output)

        # `run()` returns the outputs as a list, in output order.
        (out,) = runner.signature("serving_default").run(**ref_inputs)
        self.assertAllClose(out, ref_output)
