"""Tests for Keras python-based idempotent saving functions."""

import json
import os
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.saving import saving_lib
from keras.src.utils import file_utils
from keras.src.utils.module_utils import LazyModule


@keras.saving.register_keras_serializable(package="my_custom_package")
class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.nested_layer = keras.layers.Dense(self.units, name="dense")

    def build(self, input_shape):
        self.additional_weights = [
            self.add_weight(
                shape=(),
                name="my_additional_weight",
                initializer="ones",
                trainable=True,
            ),
            self.add_weight(
                shape=(),
                name="my_additional_weight_2",
                initializer="ones",
                trainable=True,
            ),
        ]
        self.weights_in_dict = {
            "my_weight": self.add_weight(
                shape=(),
                name="my_dict_weight",
                initializer="ones",
                trainable=True,
            ),
        }
        self.nested_layer.build(input_shape)

    def call(self, inputs):
        return self.nested_layer(inputs)

    def two(self):
        return 2


ASSETS_DATA = "These are my assets"
VARIABLES_DATA = np.random.random((10,))


@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomSaving(MyDense):
    def build(self, input_shape):
        self.assets = ASSETS_DATA
        self.stored_variables = VARIABLES_DATA
        return super().build(input_shape)

    def save_assets(self, inner_path):
        with open(os.path.join(inner_path, "assets.txt"), "w") as f:
            f.write(self.assets)

    def save_own_variables(self, store):
        store["variables"] = self.stored_variables

    def load_assets(self, inner_path):
        with open(os.path.join(inner_path, "assets.txt"), "r") as f:
            text = f.read()
        self.assets = text

    def load_own_variables(self, store):
        self.stored_variables = np.array(store["variables"])


@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithNestedLayers(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1, name="dense")

    def build(self, input_shape):
        self.dense.build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithNestedLayers(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = LayerWithNestedLayers(name="dense1")
        self.dense2 = LayerWithNestedLayers(name="dense2")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


@keras.saving.register_keras_serializable(package="my_custom_package")
class CustomModelX(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1, name="my_dense_1")
        self.dense2 = MyDense(1, name="my_dense_2")

    def call(self, inputs):
        out = self.dense1(inputs)
        return self.dense2(out)

    def one(self):
        return 1


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithKwargsCall(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_a = keras.layers.Dense(1, name="dense_a")
        self.dense_b = keras.layers.Dense(1, name="dense_b")

    def build(self, a_shape, b_shape):
        self.dense_a.build(a_shape)
        self.dense_b.build(b_shape)
        super().build(a_shape)

    def call(self, a, b):
        return self.dense_a(a) + self.dense_b(b)


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomSaving(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_dense = LayerWithCustomSaving(1)

    def call(self, inputs):
        return self.custom_dense(inputs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

    def call(self, inputs):
        return self.dense1(inputs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingSequential(keras.Sequential):
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class SubclassFunctional(keras.Model):
    """Subclassed functional identical to `_get_basic_functional_model`."""

    def __init__(self, **kwargs):
        inputs = keras.Input(shape=(4,), batch_size=2, name="input")
        dense = keras.layers.Dense(1, name="first_dense")
        x = dense(inputs)
        outputs = keras.layers.Dense(1, name="second_dense")(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        # Attrs for layers in the functional graph should not affect saving
        self.layer_attr = dense

    @property
    def layer_property(self):
        # Properties for layers in the functional graph should not affect saving
        return self.layer_attr


@keras.saving.register_keras_serializable(package="my_custom_package")
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, but as a custom fn."""
    return ops.mean(ops.square(y_pred - y_true), axis=-1)


def _get_subclassed_model(compile=True):
    subclassed_model = CustomModelX(name="custom_model_x")
    if compile:
        subclassed_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return subclassed_model


def _get_subclassed_model_with_kwargs(compile=True):
    subclassed_model = ModelWithKwargsCall(name="model_with_kwargs_call")
    if compile:
        subclassed_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return subclassed_model


def _get_subclassed_model_with_nested(compile=True):
    model = ModelWithNestedLayers(name="model_with_nested_layers")
    if compile:
        model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return model


def _get_custom_sequential_model(compile=True):
    sequential_model = keras.Sequential(
        [MyDense(1), MyDense(1)], name="sequential"
    )
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_basic_sequential_model(compile=True):
    sequential_model = keras.Sequential(
        [
            keras.layers.Dense(1, name="dense_1"),
            keras.layers.Dense(1, name="dense_2"),
        ],
        name="sequential",
    )
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_custom_functional_model(compile=True):
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = MyDense(1, name="first_dense")(inputs)
    outputs = MyDense(1, name="second_dense")(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_basic_functional_model(compile=True):
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = keras.layers.Dense(1, name="first_dense")(inputs)
    outputs = keras.layers.Dense(1, name="second_dense")(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_nested_functional_model(compile=True):
    inputs1 = keras.Input(shape=(4,), batch_size=2)
    outputs1 = keras.layers.Dense(1, name="dense")(inputs1)
    functional_model_1 = keras.Model(inputs1, outputs1)

    inputs2 = keras.Input(shape=(1,), batch_size=2)
    outputs2 = keras.layers.Dense(1, name="dense")(inputs2)
    functional_model_2 = keras.Model(inputs2, outputs2)

    inputs = keras.Input(shape=(4,), batch_size=2)
    x = functional_model_1(inputs)
    x = functional_model_2(x)
    functional_model = keras.Model(inputs, x)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_subclassed_functional_model(compile=True):
    functional_model = SubclassFunctional()
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


# We need a global function for `Pool.apply_async`
def _load_model_fn(filepath):
    saving_lib.load_model(filepath)


class SavingTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Set `_MEMORY_UPPER_BOUND` to zero for testing purpose.
        self.original_value = saving_lib._MEMORY_UPPER_BOUND
        saving_lib._MEMORY_UPPER_BOUND = 0

    def tearDown(self):
        super().tearDown()
        saving_lib._MEMORY_UPPER_BOUND = self.original_value

    def _test_inference_after_instantiation(self, model):
        if isinstance(model, ModelWithKwargsCall):
            args = ()
            kwargs = {
                "a": np.random.random((2, 4)),
                "b": np.random.random((2, 4)),
            }
        else:
            args = (np.random.random((2, 4)),)
            kwargs = {}
        y_ref = model(*args, **kwargs)
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertFalse(model.compiled)

        self.assertLen(loaded_model.variables, len(model.variables))
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertEqual(w.name, w_ref.name)
            self.assertEqual(w.path, w_ref.path)
            self.assertAllClose(w, w_ref)

        layers = model._flatten_layers(include_self=True)
        loaded_layers = loaded_model._flatten_layers(include_self=True)
        self.assertLen(loaded_layers, len(layers))
        for l, loaded_l in zip(layers, loaded_layers):
            self.assertIs(loaded_l.__class__, l.__class__)
            self.assertEqual(loaded_l.name, l.name)
            self.assertEqual(loaded_l.path, l.path)

        self.assertAllClose(loaded_model(*args, **kwargs), y_ref)

    @parameterized.named_parameters(
        ("subclassed", _get_subclassed_model),
        ("subclassed_with_nested", _get_subclassed_model_with_nested),
        ("subclassed_with_kwargs", _get_subclassed_model_with_kwargs),
        ("basic_sequential", _get_basic_sequential_model),
        ("basic_functional", _get_basic_functional_model),
        ("custom_sequential", _get_custom_sequential_model),
        ("custom_functional", _get_custom_functional_model),
        ("nested_functional", _get_nested_functional_model),
        ("subclassed_functional", _get_subclassed_functional_model),
    )
    def test_inference_after_instantiation(self, model_fn):
        model = model_fn(compile=False)
        self._test_inference_after_instantiation(model)

        # Test small model path
        saving_lib._MEMORY_UPPER_BOUND = 1.0
        self._test_inference_after_instantiation(model)

    def _test_compile_preserved(self, model):
        x_ref = np.random.random((2, 4))
        y_ref = np.random.random((2, 1))

        model.fit(x_ref, y_ref)
        out_ref = model(x_ref)
        ref_metrics = model.evaluate(x_ref, y_ref)
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertTrue(model.compiled)
        self.assertTrue(loaded_model.built)
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertAllClose(w, w_ref)
        self.assertAllClose(loaded_model(x_ref), out_ref)

        self.assertEqual(
            model.optimizer.__class__, loaded_model.optimizer.__class__
        )
        self.assertEqual(
            model.optimizer.get_config(), loaded_model.optimizer.get_config()
        )
        for w_ref, w in zip(
            model.optimizer.variables, loaded_model.optimizer.variables
        ):
            self.assertAllClose(w, w_ref)

        new_metrics = loaded_model.evaluate(x_ref, y_ref)
        for ref_m, m in zip(ref_metrics, new_metrics):
            self.assertAllClose(m, ref_m)

    @parameterized.named_parameters(
        ("subclassed", _get_subclassed_model),
        ("basic_sequential", _get_basic_sequential_model),
        ("basic_functional", _get_basic_functional_model),
        ("custom_sequential", _get_custom_sequential_model),
        ("custom_functional", _get_custom_functional_model),
        ("subclassed_functional", _get_subclassed_functional_model),
    )
    @pytest.mark.requires_trainable_backend
    def test_compile_preserved(self, model_fn):
        model = model_fn(compile=True)
        self._test_compile_preserved(model)

        # Test small model path
        saving_lib._MEMORY_UPPER_BOUND = 1.0
        self._test_compile_preserved(model)

    def test_saving_preserve_unbuilt_state(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = CustomModelX()
        subclassed_model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(subclassed_model.compiled, loaded_model.compiled)
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    @pytest.mark.requires_trainable_backend
    def test_saved_module_paths_and_class_names(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = _get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model.save(temp_filepath)

        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._CONFIG_FILENAME, "r") as c:
                config_json = c.read()
        config_dict = json.loads(config_json)
        self.assertEqual(
            config_dict["registered_name"], "my_custom_package>CustomModelX"
        )
        # exclude optimizer name
        del config_dict["compile_config"]["optimizer"]["config"]["name"]
        expected_config = keras.src.saving.serialize_keras_object(
            keras.src.optimizers.get("adam")
        )
        del expected_config["config"]["name"]
        self.assertEqual(
            config_dict["compile_config"]["optimizer"],
            expected_config,
        )
        self.assertEqual(
            config_dict["compile_config"]["loss"]["config"],
            "my_custom_package>my_mean_squared_error",
        )

    @pytest.mark.requires_trainable_backend
    def test_saving_custom_assets_and_variables(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = ModelWithCustomSaving()
        model.compile(
            optimizer="adam",
            loss="mse",
        )
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)

        # Assert that the archive has not been saved.
        self.assertFalse(os.path.exists(temp_filepath))

        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(loaded_model.custom_dense.assets, ASSETS_DATA)
        self.assertEqual(
            loaded_model.custom_dense.stored_variables.tolist(),
            VARIABLES_DATA.tolist(),
        )

    def _test_compile_overridden_warnings(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = (
            CompileOverridingModel()
            if model_type == "subclassed"
            else CompileOverridingSequential(
                [keras.layers.Embedding(4, 1), MyDense(1), MyDense(1)]
            )
        )
        model.compile("sgd", "mse")
        model.save(temp_filepath)

        with mock.patch.object(warnings, "warn") as mock_warn:
            saving_lib.load_model(temp_filepath)
        if not mock_warn.call_args_list:
            raise AssertionError("Did not warn.")
        self.assertIn(
            "`compile()` was not called as part of model loading "
            "because the model's `compile()` method is custom. ",
            mock_warn.call_args_list[0][0][0],
        )

    def test_compile_overridden_warnings_sequential(self):
        self._test_compile_overridden_warnings("sequential")

    def test_compile_overridden_warnings_subclassed(self):
        self._test_compile_overridden_warnings("subclassed")

    def test_metadata(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "my_model.keras")
        )
        model = CompileOverridingModel()
        model.save(temp_filepath)
        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._METADATA_FILENAME, "r") as c:
                metadata_json = c.read()
        metadata = json.loads(metadata_json)
        self.assertIn("keras_version", metadata)
        self.assertIn("date_saved", metadata)

    # def test_gfile_copy_local_called(self):
    #     temp_filepath = Path(
    #         os.path.join(self.get_temp_dir(), "my_model.keras")
    #     )
    #     model = CompileOverridingModel()
    #     with mock.patch(
    #         "re.match", autospec=True
    #     ) as mock_re_match, mock.patch(
    #         "tensorflow.compat.v2.io.file_utils.copy", autospec=True
    #     ) as mock_copy:
    #         # Mock Remote Path check to true to test gfile copy logic
    #         mock_re_match.return_value = True
    #         model.save(temp_filepath)
    #         mock_re_match.assert_called()
    #         mock_copy.assert_called()
    #         self.assertIn(str(temp_filepath), mock_re_match.call_args.args)
    #         self.assertIn(str(temp_filepath), mock_copy.call_args.args)

    def test_save_load_weights_only(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_save_weights_only_with_unbuilt_model(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_subclassed_model()
        with self.assertRaisesRegex(
            ValueError, "You are saving a model that has not yet been built."
        ):
            saving_lib.save_weights_only(model, temp_filepath)

    def test_load_weights_only_with_unbuilt_model(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_subclassed_model()
        x = np.random.random((100, 32))
        _ = model.predict(x)  # Build the model by calling it on some data
        saving_lib.save_weights_only(model, temp_filepath)
        saving_lib.load_weights_only(model, temp_filepath)

        new_model = _get_subclassed_model()
        with self.assertRaisesRegex(
            ValueError,
            "You are loading weights into a model that has not yet been built.",
        ):
            saving_lib.load_weights_only(new_model, temp_filepath)

    def test_load_weights_only_with_keras_file(self):
        # Test loading weights from whole saved model
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_model(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_save_weights_subclassed_functional(self):
        # The subclassed and basic functional model should have the same
        # weights structure.
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        # Test saving basic, loading subclassed.
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_subclassed_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test saving subclassed, loading basic.
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    @pytest.mark.requires_trainable_backend
    def test_compile_arg(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = _get_basic_functional_model()
        model.compile("sgd", "mse")
        model.fit(np.random.random((2, 4)), np.random.random((2, 1)))
        saving_lib.save_model(model, temp_filepath)

        model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model.compiled, True)
        model = saving_lib.load_model(temp_filepath, compile=False)
        self.assertEqual(model.compiled, False)

    # def test_overwrite(self):
    #     temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
    #     model = _get_basic_functional_model()
    #     model.save(temp_filepath)
    #     model.save(temp_filepath, overwrite=True)
    #     with self.assertRaises(EOFError):
    #         model.save(temp_filepath, overwrite=False)

    #     temp_filepath = os.path.join(
    #         self.get_temp_dir(), "mymodel.weights.h5"
    #     )
    #     model = _get_basic_functional_model()
    #     model.save_weights(temp_filepath)
    #     model.save_weights(temp_filepath, overwrite=True)
    #     with self.assertRaises(EOFError):
    #         model.save_weights(temp_filepath, overwrite=False)

    def test_partial_load(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        original_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
            ]
        )
        original_model.save(temp_filepath)

        # Test with a model that has a differently shaped layer
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(6),
            ]
        )
        new_layer_kernel_value = np.array(new_model.layers[1].kernel)
        with self.assertRaisesRegex(ValueError, "must match"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        ref_weights = original_model.layers[0].get_weights()
        new_weights = new_model.layers[0].get_weights()
        self.assertEqual(len(ref_weights), len(new_weights))
        for ref_w, w in zip(ref_weights, new_weights):
            self.assertAllClose(w, ref_w)
        self.assertAllClose(
            np.array(new_model.layers[1].kernel), new_layer_kernel_value
        )

        # Test with a model that has a new layer at the end
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
                keras.layers.Dense(5),
            ]
        )
        new_layer_kernel_value = np.array(new_model.layers[2].kernel)
        with self.assertRaisesRegex(ValueError, "received 0 variables"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        for layer_index in [0, 1]:
            ref_weights = original_model.layers[layer_index].get_weights()
            new_weights = new_model.layers[layer_index].get_weights()
            self.assertEqual(len(ref_weights), len(new_weights))
            for ref_w, w in zip(ref_weights, new_weights):
                self.assertAllClose(w, ref_w)
        self.assertAllClose(
            np.array(new_model.layers[2].kernel), new_layer_kernel_value
        )

    @pytest.mark.requires_trainable_backend
    def test_save_to_fileobj(self):
        model = keras.Sequential(
            [keras.layers.Dense(1, input_shape=(1,)), keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")

        out = BytesIO()
        saving_lib.save_model(model, out)
        out.seek(0)
        model = saving_lib.load_model(out)

        model.fit(np.array([1, 2]), np.array([1, 2]))
        pred1 = model.predict(np.array([1, 2]))

        out = BytesIO()
        saving_lib.save_model(model, out)
        out.seek(0)
        new_model = saving_lib.load_model(out)

        pred2 = new_model.predict(np.array([1, 2]))

        self.assertAllClose(pred1, pred2, atol=1e-5)

    @parameterized.named_parameters(
        ("high_memory_config", True),
        ("low_memory_config", False),
    )
    def test_save_model_exception_raised(self, is_memory_sufficient):
        if is_memory_sufficient:
            saving_lib._MEMORY_UPPER_BOUND = 0.5  # 50%

        # Assume we have an error in `save_own_variables`.
        class RaiseErrorLayer(keras.layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(units)

            def call(self, inputs):
                return self.dense(inputs)

            def save_own_variables(self, store):
                raise ValueError

        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer(1)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        with self.assertRaises(ValueError):
            saving_lib.save_model(model, filepath)

        # Ensure we don't have a bad "model.weights.h5" inside the zip file.
        self.assertTrue(Path(filepath).exists())
        with zipfile.ZipFile(filepath) as zf:
            all_filenames = zf.namelist()
            self.assertNotIn("model.weights.h5", all_filenames)

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    @parameterized.named_parameters(
        # A directory where the weights file must go.
        ("weights_store", "model.weights.h5", True, IsADirectoryError),
        # A regular file where the assets directory must go.
        ("assets_store", "assets", False, FileExistsError),
    )
    def test_save_model_to_dir_store_error_raised(
        self, blocked_name, blocked_by_dir, error_type
    ):
        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(1)])
        dirpath = os.path.join(self.get_temp_dir(), "my_model")
        os.makedirs(dirpath)
        blocked_path = os.path.join(dirpath, blocked_name)
        if blocked_by_dir:
            os.makedirs(blocked_path)
        else:
            Path(blocked_path).touch()

        # The store fails to open. Its error must reach the caller.
        with self.assertRaises(error_type):
            saving_lib.save_model(model, dirpath, zipped=False)

    @parameterized.named_parameters(
        ("high_memory_config", True),
        ("low_memory_config", False),
    )
    def test_load_model_exception_raised(self, is_memory_sufficient):
        if is_memory_sufficient:
            saving_lib._MEMORY_UPPER_BOUND = 0.5  # 50%

        # Assume we have an error in `load_own_variables`.
        class RaiseErrorLayer(keras.layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(units)

            def call(self, inputs):
                return self.dense(inputs)

            def load_own_variables(self, store):
                raise ValueError

        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer(1)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)
        with self.assertRaises(ValueError):
            saving_lib.load_model(
                filepath, custom_objects={"RaiseErrorLayer": RaiseErrorLayer}
            )

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    def test_load_model_read_only_system(self):
        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(32)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)

        # Load the model correctly, regardless of whether an OSError occurs.
        original_mode = os.stat(Path(filepath).parent).st_mode
        os.chmod(Path(filepath).parent, mode=0o555)
        model = saving_lib.load_model(filepath)
        os.chmod(Path(filepath).parent, mode=original_mode)

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX backend doesn't support Python's multiprocessing",
    )
    @pytest.mark.skipif(
        testing.uses_gpu(),
        reason="This test doesn't support GPU",
    )
    def test_load_model_concurrently(self):
        import multiprocessing as mp

        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(2)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)

        # Load the model concurrently.
        results = []
        with mp.Pool(4) as pool:
            for i in range(4):
                results.append(pool.apply_async(_load_model_fn, (filepath,)))
            pool.close()
            pool.join()
        [r.get() for r in results]  # No error occurs here

    def test_load_model_containing_reused_layer(self):
        # https://github.com/keras-team/keras/issues/20307
        inputs = keras.Input((4,))
        reused_layer = keras.layers.Dense(4)
        x = reused_layer(inputs)
        x = keras.layers.Dense(4)(x)
        outputs = reused_layer(x)
        model = keras.Model(inputs, outputs)

        self.assertLen(model.layers, 3)  # Input + 2 Dense layers
        self._test_inference_after_instantiation(model)

    @parameterized.named_parameters(
        ("efficientnet_b0_512", "efficientnet_b0", 1),  # Only 1 sharded file.
        ("efficientnet_b0_10", "efficientnet_b0", 0.01),
    )
    def test_weights_sharding(self, model_name, max_shard_size):
        from keras.src.applications import efficientnet

        if backend.image_data_format() == "channels_last":
            shape = (224, 224, 3)
        else:
            shape = (3, 224, 224)

        if model_name == "efficientnet_b0":
            model_fn = efficientnet.EfficientNetB0

        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.json")
        )
        model = model_fn(weights=None, input_shape=shape)
        ref_input = np.random.random((1, *shape)).astype("float32")
        ref_output = model.predict(ref_input)

        # Save the sharded files.
        saving_lib.save_weights_only(
            model, temp_filepath, max_shard_size=max_shard_size
        )
        self.assertIn("mymodel.weights.json", os.listdir(temp_filepath.parent))
        if max_shard_size == 1:
            # 1 sharded file + 1 config file = 2.
            self.assertLen(os.listdir(temp_filepath.parent), 2)
        elif max_shard_size == 0.01:
            # 3 sharded file + 1 config file = 4.
            self.assertLen(os.listdir(temp_filepath.parent), 4)

        with open(temp_filepath, "r") as f:
            sharding_config = json.load(f)
        self.assertIn("metadata", sharding_config)
        self.assertIn("weight_map", sharding_config)

        # Instantiate new model and load the sharded files.
        model = model_fn(weights=None, input_shape=shape)
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)


class SavingAPITest(testing.TestCase):
    def test_saving_api_errors(self):
        from keras.src.saving import saving_api

        model = _get_basic_functional_model()

        # Saving API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel")
        with self.assertRaisesRegex(ValueError, "argument is deprecated"):
            saving_api.save_model(model, temp_filepath, save_format="keras")

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.notkeras")
        with self.assertRaisesRegex(ValueError, "Invalid filepath extension"):
            saving_api.save_model(model, temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        with self.assertRaisesRegex(ValueError, "are not supported"):
            saving_api.save_model(model, temp_filepath, invalid_arg="hello")

        # Loading API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "non_existent.keras")
        with self.assertRaisesRegex(
            ValueError, "Please ensure the file is an accessible"
        ):
            _ = saving_api.load_model(temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "my_saved_model")
        with self.assertRaisesRegex(ValueError, "File format not supported"):
            _ = saving_api.load_model(temp_filepath)

    def test_model_api_endpoint(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_model_api_endpoint_h5(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.h5"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_model_api_errors(self):
        model = _get_basic_functional_model()

        # Saving API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel")
        with self.assertRaisesRegex(ValueError, "argument is deprecated"):
            model.save(temp_filepath, save_format="keras")

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.notkeras")
        with self.assertRaisesRegex(ValueError, "Invalid filepath extension"):
            model.save(temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        with self.assertRaisesRegex(ValueError, "are not supported"):
            model.save(temp_filepath, invalid_arg="hello")

    def test_safe_mode(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "unsafe_model.keras")
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Lambda(lambda x: x * 2),
            ]
        )
        model.save(temp_filepath)
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            model = saving_lib.load_model(temp_filepath)
        model = saving_lib.load_model(temp_filepath, safe_mode=False)

    def test_normalization_kpl(self):
        # With adapt
        temp_filepath = os.path.join(self.get_temp_dir(), "norm_model.keras")
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Normalization(),
            ]
        )
        data = np.random.random((3, 3))
        model.layers[0].adapt(data)
        ref_out = model(data)
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(out, ref_out, atol=1e-6)

        # Without adapt
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Normalization(
                    mean=np.random.random((3,)),
                    variance=np.random.random((3,)),
                ),
            ]
        )
        ref_out = model(data)
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(out, ref_out, atol=1e-6)


# This class is properly registered with a `get_config()` method.
# However, since it does not subclass keras.layers.Layer, it lacks
# `from_config()` for deserialization.
@keras.saving.register_keras_serializable()
class GrowthFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, inputs):
        return inputs * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.saving.register_keras_serializable(package="Complex")
class FactorLayer(keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


# This custom model does not explicitly deserialize the layers it includes
# in its `get_config`. Explicit deserialization in a `from_config` override
# or `__init__` is needed here, or an error will be thrown at loading time.
@keras.saving.register_keras_serializable(package="Complex")
class ComplexModel(keras.layers.Layer):
    def __init__(self, first_layer, second_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.first_layer = first_layer
        if second_layer is not None:
            self.second_layer = second_layer
        else:
            self.second_layer = keras.layers.Dense(8)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "first_layer": self.first_layer,
                "second_layer": self.second_layer,
            }
        )
        return config

    def call(self, inputs):
        return self.first_layer(self.second_layer(inputs))


class SavingBattleTest(testing.TestCase):
    def test_custom_object_without_from_config(self):
        temp_filepath = os.path.join(
            self.get_temp_dir(), "custom_fn_model.keras"
        )

        inputs = keras.Input(shape=(4, 4))
        outputs = keras.layers.Dense(1, activation=GrowthFactor(0.5))(inputs)
        model = keras.Model(inputs, outputs)

        model.save(temp_filepath)

        with self.assertRaisesRegex(
            TypeError, "Unable to reconstruct an instance"
        ):
            _ = saving_lib.load_model(temp_filepath)

    def test_complex_model_without_explicit_deserialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "complex_model.keras")

        inputs = keras.Input((32,))
        outputs = ComplexModel(first_layer=FactorLayer(0.5))(inputs)
        model = keras.Model(inputs, outputs)

        model.save(temp_filepath)

        with self.assertRaisesRegex(TypeError, "are explicitly deserialized"):
            _ = saving_lib.load_model(temp_filepath)

    def test_redefinition_of_trackable(self):
        """Test that a trackable can be aliased under a new name."""

        class NormalModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(3)

            def call(self, x):
                return self.dense(x)

        class WeirdModel(keras.Model):
            def __init__(self):
                super().__init__()
                # This property will be traversed first,
                # but "_dense" isn't in the saved file
                # generated by NormalModel.
                self.a_dense = keras.layers.Dense(3)

            @property
            def dense(self):
                return self.a_dense

            def call(self, x):
                return self.dense(x)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "normal_model.weights.h5"
        )
        model_a = NormalModel()
        model_a(np.random.random((2, 2)))
        model_a.save_weights(temp_filepath)
        model_b = WeirdModel()
        model_b(np.random.random((2, 2)))
        model_b.load_weights(temp_filepath)
        self.assertAllClose(
            model_a.dense.kernel.numpy(), model_b.dense.kernel.numpy()
        )

    def test_normalization_legacy_h5_format(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "custom_model.h5")

        inputs = keras.Input((32,))
        normalization = keras.layers.Normalization()
        outputs = normalization(inputs)

        model = keras.Model(inputs, outputs)

        x = np.random.random((1, 32))
        normalization.adapt(x)
        ref_out = model(x)

        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(out, ref_out, atol=1e-6)

    def test_legacy_h5_format(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "custom_model.h5")

        inputs = keras.Input((32,))
        x = MyDense(2)(inputs)
        outputs = CustomModelX()(x)
        model = keras.Model(inputs, outputs)

        x = np.random.random((1, 32))
        ref_out = model(x)

        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(out, ref_out, atol=1e-6)

    def test_nested_functional_model_saving(self):
        def func(in_size=4, out_size=2, name=None):
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)((inputs))
            return keras.Model(inputs, outputs=outputs, name=name)

        input_a, input_b = keras.Input((4,)), keras.Input((4,))
        out_a = func(out_size=2, name="func_a")(input_a)
        out_b = func(out_size=3, name="func_b")(input_b)
        model = keras.Model([input_a, input_b], outputs=[out_a, out_b])

        temp_filepath = os.path.join(self.get_temp_dir(), "nested_func.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = [np.random.random((2, 4))], np.random.random((2, 4))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_nested_shared_functional_model_saving(self):
        def func(in_size=4, out_size=2, name=None):
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)((inputs))
            return keras.Model(inputs, outputs=outputs, name=name)

        inputs = [keras.Input((4,)), keras.Input((4,))]
        func_shared = func(out_size=4, name="func_shared")
        shared_a = func_shared(inputs[0])
        shared_b = func_shared(inputs[1])
        out_a = keras.layers.Dense(2)(shared_a)
        out_b = keras.layers.Dense(2)(shared_b)
        model = keras.Model(inputs, outputs=[out_a, out_b])

        temp_filepath = os.path.join(
            self.get_temp_dir(), "nested_shared_func.keras"
        )
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = [np.random.random((2, 4))], np.random.random((2, 4))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_bidirectional_lstm_saving(self):
        inputs = keras.Input((3, 2))
        outputs = keras.layers.Bidirectional(keras.layers.LSTM(64))(inputs)
        model = keras.Model(inputs, outputs)
        temp_filepath = os.path.join(self.get_temp_dir(), "bidir_lstm.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = np.random.random((1, 3, 2))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(out, ref_out)

    @parameterized.named_parameters(
        ("list", list, lambda b: b),
        ("tuple", tuple, lambda b: b),
        ("dict", dict, lambda b: b.values()),
    )
    def test_nested_container_layer_saving(self, container_type, get_values):
        """Test that layers stored in nested containers are saved/loaded."""

        package = f"test_nested_{container_type.__name__}"

        @keras.saving.register_keras_serializable(package=package)
        class NestedContainerModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                inner1 = [keras.layers.Dense(8), keras.layers.Dense(8)]
                inner2 = [keras.layers.Dense(8), keras.layers.Dense(8)]
                if container_type is dict:
                    self.blocks = {
                        "block_a": inner1,
                        "block_b": inner2,
                    }
                elif container_type is list:
                    self.blocks = [inner1, inner2]
                else:
                    self.blocks = (tuple(inner1), tuple(inner2))
                self.out_layer = keras.layers.Dense(2)

            def call(self, x):
                for block in get_values(self.blocks):
                    for layer in block:
                        x = layer(x)
                return self.out_layer(x)

            def get_config(self):
                return super().get_config()

        model = NestedContainerModel()
        x = np.random.random((2, 4))
        model(x)  # build weights

        # Assign distinct constant kernels so that any layer-swapping
        # on reload would produce a detectable mismatch.
        for i, block in enumerate(get_values(model.blocks)):
            for j, layer in enumerate(block):
                val = float(i * 2 + j + 1)  # 1.0, 2.0, 3.0, 4.0
                layer.kernel.assign(np.full_like(layer.kernel, val))
                layer.bias.assign(np.zeros_like(layer.bias))

        ref_out = model(x)
        suffix = container_type.__name__
        temp_filepath = os.path.join(
            self.get_temp_dir(), f"nested_{suffix}.keras"
        )
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)

        # Verify each nested layer's weights were individually restored.
        for i, (orig_block, new_block) in enumerate(
            zip(get_values(model.blocks), get_values(new_model.blocks))
        ):
            for j, (orig_layer, new_layer) in enumerate(
                zip(orig_block, new_block)
            ):
                self.assertAllClose(
                    orig_layer.kernel,
                    new_layer.kernel,
                    msg=f"Kernel mismatch at blocks[{i}][{j}]",
                )

        out = new_model(x)
        self.assertAllClose(ref_out, out)

    def test_deeply_nested_container_layer_saving(self):
        """Test 3+ levels of nested containers."""

        @keras.saving.register_keras_serializable(package="test_deeply_nested")
        class DeeplyNestedModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.blocks = [[[keras.layers.Dense(4), keras.layers.Dense(4)]]]
                self.out_layer = keras.layers.Dense(2)

            def call(self, x):
                for block_list in self.blocks:
                    for block in block_list:
                        for layer in block:
                            x = layer(x)
                return self.out_layer(x)

            def get_config(self):
                return super().get_config()

        model = DeeplyNestedModel()
        x = np.random.random((2, 4))
        model(x)
        ref_out = model(x)

        temp_filepath = os.path.join(self.get_temp_dir(), "deeply_nested.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out)

    def test_mixed_nested_container_layer_saving(self):
        """Test mixed container types (list + tuple)."""

        @keras.saving.register_keras_serializable(package="test_mixed_nested")
        class MixedNestedModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.blocks = [(keras.layers.Dense(4), keras.layers.Dense(4))]
                self.out_layer = keras.layers.Dense(2)

            def call(self, x):
                for block in self.blocks:
                    for layer in block:
                        x = layer(x)
                return self.out_layer(x)

            def get_config(self):
                return super().get_config()

        model = MixedNestedModel()
        x = np.random.random((2, 4))
        model(x)
        ref_out = model(x)

        temp_filepath = os.path.join(self.get_temp_dir(), "mixed_nested.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out)

    def test_cyclic_container_saving(self):
        """Test that cyclic container references do not cause RecursionError."""

        @keras.saving.register_keras_serializable(
            package="test_cyclic_container"
        )
        class CyclicContainerModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.blocks = []
                self.blocks.append(self.blocks)
                self.out = keras.layers.Dense(2)

            def call(self, x):
                return self.out(x)

            def get_config(self):
                return super().get_config()

        model = CyclicContainerModel()
        x = np.random.random((2, 4))
        model(x)
        ref_out = model(x)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "cyclic_container.keras"
        )
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out)

    def test_legacy_load_skips_missing_nested_containers(self):
        """Regression test for PR #22362 backward-compat.

        Loads a real pre-PR-22362 .keras fixture whose sublayers live in
        a nested container (`self.blocks = [(Dense,), ...]`). The pre-PR
        saver silently skipped the inner tuples, so the file has all
        Dense weights stored under the model's flat `.layers/` walk path
        — exactly the on-disk shape users have. Post-PR load must not
        raise on it; it should warn that the inner `container*` groups
        are missing and recover the weights from `/layers/dense_X`.

        Fixture was generated from the parent of PR #22362
        (commit d5a88bdb1). See the PR description for the regen script
        and pinned SHA-256.
        """

        # Same class definition used to generate the fixture. Must match
        # the get_config / call signature so deserialization rebuilds the
        # same Python container topology.
        @keras.saving.register_keras_serializable(
            package="legacy_nested_container"
        )
        class NestedDenseModel(keras.Model):
            def __init__(self, num=3, **kwargs):
                super().__init__(**kwargs)
                self.num = num
                self.blocks = [
                    (keras.layers.Dense(4, use_bias=False, name=f"d_{i}"),)
                    for i in range(num)
                ]
                self.out = keras.layers.Dense(2)

            def call(self, x):
                for (d,) in self.blocks:
                    x = d(x)
                return self.out(x)

            def get_config(self):
                cfg = super().get_config()
                cfg["num"] = self.num
                return cfg

        fixture_path = file_utils.join(
            os.path.dirname(__file__),
            "test_data",
            "pre_pr22362_nested_container.keras",
        )
        self.assertTrue(
            file_utils.exists(fixture_path),
            f"Backward-compat fixture missing at {fixture_path}",
        )

        # Loading must succeed and emit the legacy-skip warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = keras.saving.load_model(fixture_path)
        self.assertTrue(
            any("Skipping nested container" in str(w.message) for w in caught),
            msg="Expected a 'Skipping nested container' UserWarning.",
        )

        # The fixture was saved with deterministic kernels:
        #   blocks[i][0].kernel = 0.1 * (i + 1)
        #   out.kernel = 0.5, out.bias = 0.0
        for i, (d,) in enumerate(loaded.blocks):
            self.assertAllClose(
                ops.convert_to_numpy(d.kernel),
                np.full((4, 4), 0.1 * (i + 1), dtype="float32"),
                atol=1e-6,
                msg=f"blocks[{i}][0].kernel mismatch",
            )
        self.assertAllClose(
            ops.convert_to_numpy(loaded.out.kernel),
            np.full((4, 2), 0.5, dtype="float32"),
            atol=1e-6,
        )
        self.assertAllClose(
            ops.convert_to_numpy(loaded.out.bias),
            np.zeros((2,), dtype="float32"),
            atol=1e-6,
        )

    def test_fresh_save_load_emits_no_container_skip_warning(self):
        # The legacy-skip warning should fire only when an in-memory
        # container would have contributed unvisited `KerasSaveable` state
        # that isn't present in the file. Containers of pure metadata
        # (e.g. `variable_serialization_spec`) and containers that mirror
        # already-loaded saveables (e.g. Functional's `_operations_by_depth`)
        # have nothing left to load, so a missing group there is expected.
        path = os.path.join(self.get_temp_dir(), "model.keras")
        inp = keras.layers.Input(shape=(5,))
        model = keras.Model(
            inp, keras.layers.EinsumDense("ab,bc->ac", output_shape=(7,))(inp)
        )
        model.save(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            keras.saving.load_model(path)
        spurious = [
            str(w.message)
            for w in caught
            if "Skipping nested container" in str(w.message)
        ]
        self.assertEqual(
            spurious,
            [],
            msg=(
                "Loading a freshly-saved model should not emit the "
                f"'Skipping nested container' warning. Got: {spurious}"
            ),
        )

    def test_remove_weights_only_saving_and_loading(self):
        def is_remote_path(path):
            return True

        temp_filepath = os.path.join(self.get_temp_dir(), "model.weights.h5")

        with mock.patch(
            "keras.src.utils.file_utils.is_remote_path", is_remote_path
        ):
            model = _get_basic_functional_model()
            model.save_weights(temp_filepath)
            model.load_weights(temp_filepath)

    def test_remote_path_saving_and_loading_with_epath(self):
        # A local directory stands in for the remote location, and TensorFlow
        # is hidden so that etils.epath is used.
        remote_dir = os.path.join(self.get_temp_dir(), "remote")
        os.makedirs(remote_dir)
        filepath = os.path.join(remote_dir, "model.keras")
        model = _get_basic_functional_model()
        x = np.random.random((2, 4))

        with (
            mock.patch.object(
                file_utils,
                "is_remote_path",
                lambda path: str(path).startswith(remote_dir),
            ),
            mock.patch.object(
                file_utils, "gfile", LazyModule("keras_test_unavailable_module")
            ),
        ):
            saving_lib.save_model(model, filepath)
            new_model = saving_lib.load_model(filepath)

        self.assertAllClose(model.predict(x), new_model.predict(x))


class SavingH5IOStoreTest(testing.TestCase):
    def test_h5_io_store_basics(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "store.h5"))

        # Pre-defined data.
        a = np.random.random((2, 4)).astype("float32")
        b = np.random.random((4, 8)).astype("int32")

        # Set.
        store = saving_lib.H5IOStore(temp_filepath, mode="w")
        vars_store = store.make("vars")
        vars_store["a"] = a
        vars_store["b"] = b
        vars_store["c"] = 42
        self.assertAllClose(vars_store["a"], a)
        self.assertAllClose(vars_store["b"], b)
        self.assertEqual(int(vars_store["c"][()]), 42)

        # Delete.
        del vars_store["c"]

        # Contain.
        self.assertNotIn("c", vars_store)

        store.close()
        self.assertTrue(os.path.exists(temp_filepath))

        # Get.
        store = saving_lib.H5IOStore(temp_filepath, mode="r")
        vars_store = store.get("vars")
        self.assertAllClose(vars_store["a"], a)
        self.assertAllClose(vars_store["b"], b)
        self.assertNotIn("c", vars_store)

    def test_h5_io_store_lora(self):
        # For `keras_hub.models.backbone.save_lora_weights` and
        # `keras_hub.models.backbone.load_lora_weights`
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "layer.lora.h5"))
        layer = keras.layers.Dense(units=16)
        layer.build((None, 8))
        layer.enable_lora(4)

        ref_input = np.random.random((1, 8)).astype("float32")
        ref_output = layer(ref_input)

        # Save the LoRA weights.
        store = saving_lib.H5IOStore(temp_filepath, mode="w")
        lora_store = store.make("lora")
        lora_store["rank"] = layer.lora_rank
        inner_store = store.make("lora/0")
        inner_store["lora_kernel_a"] = layer.lora_kernel_a
        inner_store["lora_kernel_b"] = layer.lora_kernel_b
        store.close()

        # Load the LoRA weights.
        revived_layer = keras.layers.Dense(units=16)
        revived_layer.build((None, 8))
        store = saving_lib.H5IOStore(temp_filepath, mode="r")
        lora_store = store.get("lora")
        revived_layer.enable_lora(int(lora_store["rank"][()]))
        lora_kernel_a = store.get("lora/0")["lora_kernel_a"]
        lora_kernel_b = store.get("lora/0")["lora_kernel_b"]
        revived_layer._kernel.assign(layer._kernel)
        revived_layer.bias.assign(layer.bias)
        revived_layer.lora_kernel_a.assign(lora_kernel_a)
        revived_layer.lora_kernel_b.assign(lora_kernel_b)
        self.assertAllClose(revived_layer(ref_input), ref_output, atol=1e-6)

    def test_h5_io_store_exception_raised(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "store.h5"))

        # Bad `path_or_io`.
        with self.assertRaisesRegex(
            TypeError,
            (
                r"`path_or_io` should be a `str`, `pathlib.Path` or "
                r"`io.BytesIO` object."
            ),
        ):
            saving_lib.H5IOStore(None, mode="w")

        # Bad `mode`.
        with self.assertRaisesRegex(
            ValueError, r"`mode` should be either 'w' or 'r'."
        ):
            saving_lib.H5IOStore(temp_filepath, mode="x")

        # No archive when using `io.BytesIO` as `path_or_io`.
        with self.assertRaisesRegex(
            ValueError,
            (
                r"When `path_or_io` is an `io.BytesIO` object, `archive` "
                r"should be `None`."
            ),
        ):
            saving_lib.H5IOStore(BytesIO(), archive="archive", mode="w")

        store = saving_lib.H5IOStore(temp_filepath, mode="w")

        # Bad `metadata`.
        with self.assertRaisesRegex(
            ValueError, r"`metadata` should be a dict or `None`."
        ):
            store.make("vars", metadata="metadata")

        store.close()

        store = saving_lib.H5IOStore(temp_filepath, mode="r")
        vars_store = store.get("vars")

        # Set in read mode.
        with self.assertRaisesRegex(
            ValueError, r"Setting a value is only allowed in write mode."
        ):
            vars_store["weights"] = np.random.random((2, 4)).astype("float32")

        # Delete in read mode.
        with self.assertRaisesRegex(
            ValueError, r"Deleting a value is only allowed in write mode."
        ):
            del vars_store["weights"]

    def test_sharded_h5_io_store_basics(self):
        name = "sharded_store"
        temp_filepath = Path(os.path.join(self.get_temp_dir(), f"{name}.json"))

        # Pre-defined data. Each has about 0.0037GB.
        a = np.random.random((1000, 1000)).astype("float32")
        b = np.random.random((1000, 1000)).astype("int32")

        # Set.
        store = saving_lib.ShardedH5IOStore(
            temp_filepath, max_shard_size=0.005, mode="w"
        )
        vars_store = store.make("vars")
        vars_store["a"] = a
        vars_store["b"] = b
        vars_store["c"] = 42
        self.assertLen(store.sharding_config["weight_map"]["/vars/vars"], 2)
        self.assertLen(vars_store, 3)
        self.assertAllClose(vars_store["a"], a)
        self.assertAllClose(vars_store["b"], b)
        self.assertEqual(int(vars_store["c"][()]), 42)

        # Delete.
        del vars_store["c"]
        self.assertLen(vars_store, 2)
        del vars_store["a"]  # Delete from an older shard.
        self.assertLen(vars_store, 1)
        vars_store["a"] = a

        # Contain.
        self.assertIn("a", vars_store)
        self.assertNotIn("c", vars_store)

        store.close()
        self.assertTrue(os.path.exists(temp_filepath))
        self.assertTrue(
            os.path.exists(temp_filepath.with_name(f"{name}_00000.weights.h5"))
        )

        # Get.
        store = saving_lib.ShardedH5IOStore(temp_filepath, mode="r")
        vars_store = store.get("vars")
        self.assertLen(vars_store, 2)
        self.assertAllClose(vars_store["a"], a)
        self.assertAllClose(vars_store["b"], b)
        self.assertNotIn("c", vars_store)

        # Keys.
        for key in ["a", "b"]:
            self.assertIn(key, vars_store.keys())

    def test_sharded_h5_io_store_cross_shard_read(self):
        name = "cross_shard_read"
        temp_filepath = Path(os.path.join(self.get_temp_dir(), f"{name}.json"))
        a = np.random.random((1000, 1000)).astype("float32")
        b = np.random.random((1000, 1000)).astype("float32")

        store = saving_lib.ShardedH5IOStore(
            temp_filepath, max_shard_size=0.005, mode="w"
        )
        vars_store = store.make("layer_a")
        vars_store["0"] = a
        vars_store = store.make("layer_b")
        vars_store["0"] = b
        store.close()

        self.assertTrue(
            os.path.exists(temp_filepath.with_name(f"{name}_00001.weights.h5"))
        )

        store = saving_lib.ShardedH5IOStore(temp_filepath, mode="r")
        store.get("layer_b")
        vars_store = store.get("layer_a")
        self.assertLen(vars_store.keys(), 1)
        self.assertAllClose(vars_store["0"], a)
        store.close()

    def test_sharded_h5_io_store_exception_raised(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "store.h5"))

        # Bad `path_or_io`.
        with self.assertRaisesRegex(
            TypeError,
            r"`path_or_io` should be a `str`, `pathlib.Path` object. ",
        ):
            saving_lib.ShardedH5IOStore(None, mode="w")

        # Bad `mode`.
        with self.assertRaisesRegex(
            ValueError, r"`mode` should be either 'w' or 'r'."
        ):
            saving_lib.ShardedH5IOStore(temp_filepath, mode="x")

        store = saving_lib.ShardedH5IOStore(
            temp_filepath, max_shard_size=0.00001, mode="w"
        )
        vars_store = store.make("vars")

        # Too large data.
        with self.assertRaisesRegex(
            ValueError, r"exceeds the maximum shard size"
        ):
            vars_store["weights"] = np.random.random((100, 100)).astype(
                "float32"
            )

        # Bad `get`.
        with self.assertRaisesRegex(
            KeyError, r"Key 'abc' not found in any of the shards:"
        ):
            vars_store["abc"]

        # Bad `del`.
        with self.assertRaisesRegex(
            KeyError, r"Key 'abc' not found in any of the shards:"
        ):
            del vars_store["abc"]

        store.close()


class SafeZipReadTest(testing.TestCase):
    def _zip_with_member(self, name, data, compression=zipfile.ZIP_DEFLATED):
        path = os.path.join(self.get_temp_dir(), "a.zip")
        with zipfile.ZipFile(path, "w", compression=compression) as zf:
            zf.writestr(name, data)
        return path

    def test_rejects_decompression_bomb(self):
        # Highly compressible member: large declared size, ~nothing on disk.
        path = self._zip_with_member("config.json", b"A" * 100_000)
        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with zipfile.ZipFile(path, "r") as zf:
                with self.assertRaisesRegex(ValueError, "decompression bomb"):
                    saving_lib._safe_zip_read(zf, "config.json")

    def test_allows_incompressible_member(self):
        # Stored (uncompressed) member: declared size == stored size.
        data = os.urandom(100_000)
        path = self._zip_with_member("w", data, compression=zipfile.ZIP_STORED)
        with mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64):
            with zipfile.ZipFile(path, "r") as zf:
                self.assertEqual(saving_lib._safe_zip_read(zf, "w"), data)

    def test_load_model_rejects_bomb_config(self):
        # End-to-end: a tiny `.keras` whose config.json decompresses huge is
        # rejected before the read allocates, with safe_mode=True.
        path = os.path.join(self.get_temp_dir(), "bomb.keras")
        payload = b'{"x":"' + b" " * 200_000 + b'"}'
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", b'{"keras_version":"3"}')
            zf.writestr("config.json", payload)
        self.assertLess(os.path.getsize(path), 1 << 16)  # tiny file on disk
        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib.load_model(path)

    def test_load_model_rejects_bomb_weights(self):
        # A bomb `model.weights.h5` member must be rejected up front, before the
        # in-memory / extract-to-disk / on-the-fly read paths (and not swallowed
        # by the on-the-fly fallback's bare `except`).
        import keras

        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        good = os.path.join(self.get_temp_dir(), "good.keras")
        model.save(good)
        with zipfile.ZipFile(good) as zf:
            cfg = zf.read("config.json")
            meta = zf.read("metadata.json")

        evil = os.path.join(self.get_temp_dir(), "evil.keras")
        with zipfile.ZipFile(
            evil, "w"
        ) as zf:  # config/metadata stored (ratio 1)
            zf.writestr("metadata.json", meta)
            zf.writestr("config.json", cfg)
            info = zipfile.ZipInfo("model.weights.h5")
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, b"\x00" * 200_000)  # deflated bomb member

        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib.load_model(evil)

    def test_rejects_cumulative_archive_bomb_under_floor(self):
        # Several members that each stay under the (real, 4 GiB) per-member
        # floor but jointly declare far more than is stored on disk: the
        # per-member guard passes each one, the cumulative guard rejects them.
        path = os.path.join(self.get_temp_dir(), "a.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(4):
                zf.writestr(f"m{i}.json", b"A" * 100_000)
        self.assertLess(os.path.getsize(path), 1 << 16)  # tiny file on disk
        with mock.patch.object(
            saving_lib, "_ZIP_CUMULATIVE_BOMB_FLOOR_BYTES", 250_000
        ):
            with zipfile.ZipFile(path, "r") as zf:
                # The per-member guard (real 4 GiB floor) passes each member.
                for i in range(4):
                    saving_lib._reject_zip_bomb(zf, f"m{i}.json")
                # The cumulative guard rejects the archive as a whole.
                with self.assertRaisesRegex(ValueError, "decompression bomb"):
                    saving_lib._reject_zip_archive_bomb(zf)

    @parameterized.parameters("model", "weights", "editor")
    def test_load_model_rejects_cumulative_bomb(self, consumer):
        # End-to-end: a tiny `.keras` whose config.json declares well under the
        # 4 GiB per-member floor (so `_reject_zip_bomb` passes) is still
        # rejected by the cumulative guard before any member is read.
        path = os.path.join(self.get_temp_dir(), "bomb.keras")
        payload = b'{"x":"' + b" " * 400_000 + b'"}'
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", b'{"keras_version":"3"}')
            zf.writestr("config.json", payload)
        self.assertLess(os.path.getsize(path), 1 << 16)  # tiny file on disk
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        with (
            mock.patch.object(
                saving_lib, "_ZIP_CUMULATIVE_BOMB_FLOOR_BYTES", 64
            ),
            mock.patch.object(
                zipfile.ZipFile,
                "open",
                side_effect=AssertionError("Member read"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                if consumer == "model":
                    saving_lib.load_model(path)
                elif consumer == "weights":
                    saving_lib.load_weights_only(model, path)
                else:
                    keras.saving.KerasFileEditor(path)

    def test_archive_check_only_reads_directory_metadata(self):
        path = self._zip_with_member("weights", os.urandom(100_000))
        with zipfile.ZipFile(path) as archive:
            with mock.patch.object(
                archive, "open", side_effect=AssertionError("Member read")
            ):
                saving_lib._reject_zip_archive_bomb(archive)

    def test_stored_weights_do_not_hide_compressed_assets(self):
        path = os.path.join(self.get_temp_dir(), "padded.keras")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("model.weights.h5", b"A" * 100_000)
            archive.writestr(
                "assets/large.txt",
                b"A" * 100_000,
                compress_type=zipfile.ZIP_DEFLATED,
            )
        with (
            zipfile.ZipFile(path) as archive,
            mock.patch.object(
                saving_lib, "_ZIP_CUMULATIVE_BOMB_FLOOR_BYTES", 64
            ),
        ):
            infos = archive.infolist()
            self.assertLess(
                sum(info.file_size for info in infos),
                100 * sum(info.compress_size for info in infos),
            )
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib._reject_zip_archive_bomb(archive)

    @parameterized.parameters(True, False)
    def test_load_model_requires_h5_weights(self, zipped):
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        config, metadata = saving_lib._serialize_model_as_json(model)
        path = os.path.join(self.get_temp_dir(), "unsupported")
        if zipped:
            path += ".keras"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("config.json", config)
                archive.writestr("metadata.json", metadata)
                archive.writestr("model.weights.npz", b"")
        else:
            os.makedirs(path)
            with open(os.path.join(path, "config.json"), "w") as f:
                f.write(config)
            with open(os.path.join(path, "model.weights.npz"), "wb"):
                pass
        with self.assertRaisesRegex(ValueError, "Expected a model.weights.h5"):
            saving_lib.load_model(path)

    def test_does_not_reject_genuine_keras(self):
        # A genuine `.keras` (members stored ~uncompressed, ratio ~1) must pass
        # both the direct cumulative guard and an end-to-end load.
        model = keras.Sequential(
            [keras.Input((16,)), keras.layers.Dense(64), keras.layers.Dense(8)]
        )
        path = os.path.join(self.get_temp_dir(), "good.keras")
        model.save(path)
        with zipfile.ZipFile(path, "r") as zf:
            saving_lib._reject_zip_archive_bomb(zf)  # must not raise
        keras.saving.load_model(path)  # must not raise

    def test_load_model_rejects_extraction_bomb(self):
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        good = os.path.join(self.get_temp_dir(), "good.keras")
        model.save(good)
        with zipfile.ZipFile(good) as zf:
            base = {n: zf.read(n) for n in zf.namelist()}

        evil = os.path.join(self.get_temp_dir(), "evil.keras")
        with zipfile.ZipFile(evil, "w") as zf:
            for name, data in base.items():
                zf.writestr(name, data)  # genuine members stored (ratio ~1)
            info = zipfile.ZipInfo("assets/bomb.bin")
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, b"\x00" * 200_000)  # 4th member, deflated bomb

        with mock.patch.object(
            saving_lib, "_ZIP_CUMULATIVE_BOMB_FLOOR_BYTES", 64
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib.load_model(evil)


class SafeGetH5DatasetTest(testing.TestCase):
    def _shape_bomb_file(self):
        """An HDF5 file with a dataset declaring ~8 PiB but storing ~nothing."""
        path = os.path.join(self.get_temp_dir(), "bomb.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "d",
                shape=(2**50,),
                dtype="float64",
                chunks=(1024,),
                compression="gzip",
                fillvalue=0.0,
            )
        return path

    def test_rejects_shape_bomb(self):
        path = self._shape_bomb_file()
        self.assertLess(os.path.getsize(path), 1 << 20)  # tiny file on disk
        with h5py.File(path, "r") as f:
            with self.assertRaisesRegex(ValueError, "shape bomb"):
                saving_lib.safe_get_h5_dataset(f, "d")

    def test_load_weights_rejects_shape_bomb(self):
        model = keras.Sequential(
            [keras.Input((4,)), keras.layers.Dense(3, name="d")]
        )
        good_path = os.path.join(self.get_temp_dir(), "good.weights.h5")
        model.save_weights(good_path)

        # Replace a real weight dataset with a shape bomb at the same path.
        datasets = []
        with h5py.File(good_path, "r") as f:

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and "/vars/" in "/" + name:
                    datasets.append(name)

            f.visititems(collect)
        with h5py.File(good_path, "r+") as f:
            del f[datasets[0]]
            f.create_dataset(
                datasets[0],
                shape=(2**50,),
                dtype="float64",
                chunks=(1024,),
                compression="gzip",
                fillvalue=0.0,
            )

        reloaded = keras.Sequential(
            [keras.Input((4,)), keras.layers.Dense(3, name="d")]
        )
        with self.assertRaises(ValueError):
            reloaded.load_weights(good_path)


class SavingDiskIOStoreTest(testing.TestCase):
    def test_disk_io_store_rejects_path_traversal(self):
        store = saving_lib.DiskIOStore("assets", archive=None, mode="w")
        working_dir = os.path.realpath(store.working_dir)
        for bad in ["../escape", os.path.join("a", "..", "..", "escape"), ".."]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.make(bad)
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.get(bad)
            self.assertFalse(store.has_path(bad))
        # Nothing was created outside the working directory.
        self.assertFalse(
            os.path.exists(os.path.join(os.path.dirname(working_dir), "escape"))
        )
        # Normal nested asset paths still work.
        made = store.make(os.path.join("layers", "dense"))
        self.assertTrue(os.path.isdir(made))
        self.assertTrue(os.path.realpath(made).startswith(working_dir + os.sep))
        self.assertIsNotNone(store.get(os.path.join("layers", "dense")))
        store.close()

    def test_disk_io_store_rejects_backslash_traversal(self):
        store = saving_lib.DiskIOStore("assets", archive=None, mode="w")
        for bad in ["..\\escape", "a\\..\\..\\escape"]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.make(bad)
        store.close()

    def test_disk_io_store_remote_working_dir_is_preserved(self):
        store = saving_lib.DiskIOStore(
            "gs://bucket/model", archive=None, mode="r"
        )
        resolved = store._full_path(os.path.join("layers", "dense"))
        self.assertTrue(resolved.startswith("gs://bucket/model/"))
        self.assertTrue(resolved.endswith("layers/dense"))
        for bad in ["../escape", "/abs", os.path.join("x", "..", "..", "y")]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store._full_path(bad)
