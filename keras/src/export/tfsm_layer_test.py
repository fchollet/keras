import copy
import os
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import activations
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import utils
from keras.src.export import saved_model
from keras.src.export import tfsm_layer
from keras.src.export.saved_model_test import get_model
from keras.src.saving import object_registration
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib


@object_registration.register_keras_serializable(package="TFSMLayerTest")
class TFSMWrapperLayer(layers.Layer):
    """A custom layer that forwards a config value to `TFSMLayer()`.

    Reconstructing this layer never goes through `TFSMLayer.from_config()`,
    so the constructor guard is the only check on the SavedModel it loads.
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.reloaded = tfsm_layer.TFSMLayer(filepath)

    def call(self, inputs):
        return self.reloaded(inputs)

    def get_config(self):
        return {**super().get_config(), "filepath": self.filepath}


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="TFSM Layer reloading is only for the TF backend.",
)
class TestTFSMLayer(testing.TestCase):
    def test_reloading_export_archive(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        self.assertAllClose(reloaded_layer(ref_input), ref_output, atol=1e-7)
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

    def test_reloading_default_saved_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        tf.saved_model.save(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath, call_endpoint="serving_default"
        )
        # The output is a dict, due to the nature of SavedModel saving.
        new_output = reloaded_layer(ref_input)
        self.assertAllClose(
            new_output[list(new_output.keys())[0]],
            ref_output,
            atol=1e-7,
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )
        for keras_var in reloaded_layer.weights:
            self.assertIsInstance(keras_var, backend.Variable)

    def test_call_training(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        utils.set_random_seed(1337)
        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(10),
                layers.Dropout(0.99999),
            ]
        )
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="call_inference",
            fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.add_endpoint(
            name="call_training",
            fn=lambda x: model(x, training=True),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath,
            call_endpoint="call_inference",
            call_training_endpoint="call_training",
        )
        inference_output = reloaded_layer(
            tf.random.normal((1, 10)), training=False
        )
        training_output = reloaded_layer(
            tf.random.normal((1, 10)), training=True
        )
        self.assertAllClose(np.mean(training_output), 0.0, atol=1e-7)
        self.assertNotAllClose(np.mean(inference_output), 0.0, atol=1e-7)

    def test_serialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        # Test reinstantiation from config
        config = reloaded_layer.get_config()
        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(
            config, safe_mode=False
        )
        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

        # Test whole model saving with reloaded layer inside
        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(self.get_temp_dir(), "m.keras")
        model.save(temp_model_filepath, save_format="keras_v3")
        reloaded_model = saving_lib.load_model(
            temp_model_filepath, safe_mode=False
        )
        self.assertAllClose(reloaded_model(ref_input), ref_output, atol=1e-7)

    def test_safe_mode_blocks_model_loading(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Create and export a model
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        # Wrap SavedModel in TFSMLayer and save as .keras
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        wrapper_model = models.Sequential([reloaded_layer])

        model_path = os.path.join(self.get_temp_dir(), "tfsm_model.keras")
        wrapper_model.save(model_path)

        # Default safe_mode=True should block loading
        with self.assertRaisesRegex(
            ValueError,
            "arbitrary code execution",
        ):
            saving_lib.load_model(model_path)

        # Explicit opt-out should allow loading
        loaded_model = saving_lib.load_model(model_path, safe_mode=False)

        x = tf.random.normal((2, 10))
        self.assertAllClose(loaded_model(x), wrapper_model(x))

    def test_safe_mode_blocks_constructor(self):
        with mock.patch.object(tf.saved_model, "load") as load:
            with serialization_lib.SafeModeScope(True):
                with self.assertRaisesRegex(
                    ValueError, "arbitrary code execution"
                ):
                    tfsm_layer.TFSMLayer("unused_saved_model")
            load.assert_not_called()

    @parameterized.parameters(
        (None, None, True),
        (None, False, False),
        (None, True, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    )
    def test_from_config_safe_mode(self, outer_mode, safe_mode, blocked):
        if blocked:
            # The constructor raises before `tf.saved_model.load()`, so the
            # blocked rows need no exported SavedModel.
            temp_filepath = "unused_saved_model"
        else:
            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            model = get_model()
            ref_input = tf.random.normal((3, 10))
            ref_output = model(ref_input)
            saved_model.export_saved_model(model, temp_filepath)

        with serialization_lib.SafeModeScope(outer_mode):
            with mock.patch.object(
                tf.saved_model, "load", wraps=tf.saved_model.load
            ) as load:
                if blocked:
                    with self.assertRaisesRegex(
                        ValueError, "arbitrary code execution"
                    ):
                        tfsm_layer.TFSMLayer.from_config(
                            {"filepath": temp_filepath}, safe_mode=safe_mode
                        )
                    load.assert_not_called()
                else:
                    layer = tfsm_layer.TFSMLayer.from_config(
                        {"filepath": temp_filepath}, safe_mode=safe_mode
                    )
                    load.assert_called_once_with(temp_filepath)
                    self.assertAllClose(layer(ref_input), ref_output)
            self.assertIs(serialization_lib.in_safe_mode(), outer_mode)

    def _make_constructor_archive(self, temp_dir, saved_model_filepath):
        """Crafts a `.keras` file whose `Lambda` layer calls `TFSMLayer()`.

        The `Lambda` function resolves to the `TFSMLayer` constructor itself,
        so reloading the archive reaches the constructor without ever going
        through `TFSMLayer.from_config()`.

        Returns the archive path, its model config, and a reference
        input/output pair for the one legitimate (`Dense`) output.
        """
        inputs = layers.Input((10,))
        outputs = layers.Dense(1, name="dense")(inputs)
        lambda_outputs = layers.Lambda(activations.relu, name="lambda")(inputs)
        model = models.Model(inputs, [outputs, lambda_outputs])
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)[0]

        config = model.get_config()
        lambda_config = next(
            layer for layer in config["layers"] if layer["name"] == "lambda"
        )
        # Resolve the constructor as a callable, bypassing from_config().
        lambda_config["config"]["function"] = {
            "module": "keras.layers",
            "class_name": "function",
            "config": "TFSMLayer",
            "registered_name": "TFSMLayer",
        }
        lambda_config["inbound_nodes"][0] = {
            "args": [],
            "kwargs": {"inputs": saved_model_filepath},
        }
        config["output_layers"] = config["output_layers"][0]

        model_path = os.path.join(temp_dir, "constructor.keras")
        # `save()` reads the config once, so patching it is enough to write
        # the crafted archive.
        with mock.patch.object(model, "get_config", return_value=config):
            model.save(model_path)
        return model_path, config, ref_input, ref_output

    def test_safe_mode_blocks_constructor_during_model_loading(self):
        temp_dir = self.get_temp_dir()
        temp_filepath = os.path.join(temp_dir, "exported_model")
        exported_model = get_model()
        exported_model(tf.zeros((1, 10)))
        saved_model.export_saved_model(exported_model, temp_filepath)

        model_path, _, ref_input, ref_output = self._make_constructor_archive(
            temp_dir, temp_filepath
        )

        with mock.patch.object(
            tf.saved_model, "load", wraps=tf.saved_model.load
        ) as load:
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                saving_api.load_model(model_path)
            load.assert_not_called()

            # `safe_mode=None` reaches the scope unchanged; only an explicit
            # `False` may opt out.
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                saving_api.load_model(model_path, safe_mode=None)
            load.assert_not_called()

            loaded_model = saving_api.load_model(model_path, safe_mode=False)
            load.assert_called_once_with(temp_filepath)
            self.assertAllClose(loaded_model(ref_input), ref_output)

    def test_safe_mode_blocks_constructor_in_model_from_config(self):
        temp_dir = self.get_temp_dir()
        temp_filepath = os.path.join(temp_dir, "exported_model")
        exported_model = get_model()
        exported_model(tf.zeros((1, 10)))
        saved_model.export_saved_model(exported_model, temp_filepath)

        _, config, ref_input, ref_output = self._make_constructor_archive(
            temp_dir, temp_filepath
        )

        with mock.patch.object(
            tf.saved_model, "load", wraps=tf.saved_model.load
        ) as load:
            # `Model.from_config()` opens no deserialization scope of its own,
            # and the layers are called after the scope opened for each of
            # them has closed.
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                models.Model.from_config(copy.deepcopy(config))
            load.assert_not_called()

            # The opt-out still loads. `from_config()` restores the
            # architecture only, so the reference output cannot be compared
            # here; the recorded load call is what identifies the SavedModel.
            with serialization_lib.SafeModeScope(False):
                loaded_model = models.Model.from_config(copy.deepcopy(config))
            load.assert_called_once_with(temp_filepath)
            self.assertEqual(
                loaded_model(ref_input).shape, tuple(ref_output.shape)
            )

    def test_safe_mode_blocks_wrapper_layer_during_model_loading(self):
        temp_dir = self.get_temp_dir()
        original_filepath = os.path.join(temp_dir, "original_export")
        tampered_filepath = os.path.join(temp_dir, "tampered_export")
        for filepath in (original_filepath, tampered_filepath):
            exported_model = get_model()
            exported_model(tf.zeros((1, 10)))
            saved_model.export_saved_model(exported_model, filepath)

        inputs = layers.Input((10,))
        outputs = TFSMWrapperLayer(original_filepath, name="wrapper")(inputs)
        model = models.Model(inputs, outputs)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        # Point the saved config at a second SavedModel, the way an attacker
        # would. `TFSMLayer.from_config()` is never reached on this route.
        config = model.get_config()
        wrapper_config = next(
            layer for layer in config["layers"] if layer["name"] == "wrapper"
        )
        wrapper_config["config"]["filepath"] = tampered_filepath
        model_path = os.path.join(temp_dir, "wrapper.keras")
        with mock.patch.object(model, "get_config", return_value=config):
            model.save(model_path)

        with mock.patch.object(
            tf.saved_model, "load", wraps=tf.saved_model.load
        ) as load:
            # `Operation.from_config()` and `deserialize_keras_object()` each
            # wrap the `ValueError`, so the raised type is a `TypeError` here.
            with self.assertRaisesRegex(
                (TypeError, ValueError), "arbitrary code execution"
            ):
                saving_api.load_model(model_path)
            load.assert_not_called()

            # The opt-out loads the SavedModel named by the config, which is
            # the tampered one: the recorded call is the evidence, since the
            # weights stored in the archive are restored on top of whichever
            # SavedModel was loaded.
            loaded_model = saving_api.load_model(model_path, safe_mode=False)
            load.assert_called_once_with(tampered_filepath)
            self.assertAllClose(loaded_model(ref_input), ref_output)

    def test_errors(self):
        # Test missing call endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Input((2,)), layers.Dense(3)])
        saved_model.export_saved_model(model, temp_filepath)
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(temp_filepath, call_endpoint="wrong")

        # Test missing call training endpoint
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(
                temp_filepath,
                call_endpoint="serve",
                call_training_endpoint="wrong",
            )
