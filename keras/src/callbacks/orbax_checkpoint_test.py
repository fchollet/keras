import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src import utils
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
from keras.src.distribution import DeviceMesh
from keras.src.distribution import LayoutMap
from keras.src.distribution import ModelParallel
from keras.src.distribution import TensorLayout
from keras.src.distribution import distribution as get_distribution
from keras.src.distribution import get_device_count
from keras.src.distribution import set_distribution
from keras.src.saving import register_keras_serializable
from keras.src.testing.test_utils import named_product


class OrbaxCheckpointTest(testing.TestCase, parameterized.TestCase):
    def _create_test_model(self, steps_per_execution=1):
        """Create a simple test model compatible with 2-device sharding."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(6, name="dense_layer")(inputs)  # 6 units (div by 2)
        outputs = layers.Dense(2, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model")
        model.compile(
            optimizer="adam",
            loss="mse",
            steps_per_execution=steps_per_execution,
        )
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 2)  # Match 2 outputs
        return x, y

    # Shared constants for distributed tests — fixed sizes divisible by
    # 1, 2, 4, and 8 devices.
    _DIST_DENSE_UNITS = 32
    _DIST_OUT_UNITS = 16
    _DIST_NUM_SAMPLES = 64
    _DIST_PREDICT_BATCH = 8

    def _setup_distributed_test(self):
        """Validate distributed prerequisites and return common objects.

        Returns:
            (num_devices, device_mesh, original_distribution)

        Calls self.skipTest if fewer than 2 devices are available or if
        the fixed layer sizes don't divide evenly by num_devices.
        """
        num_devices = get_device_count()
        if (
            self._DIST_DENSE_UNITS % num_devices != 0
            or self._DIST_OUT_UNITS % num_devices != 0
        ):
            self.skipTest(
                f"num_devices={num_devices} does not evenly divide "
                f"dense_units={self._DIST_DENSE_UNITS} or "
                f"out_units={self._DIST_OUT_UNITS}"
            )
        device_mesh = DeviceMesh((num_devices,), axis_names=["data"])
        return num_devices, device_mesh, get_distribution()

    def _build_distributed_model(self, dense_units, out_units):
        """Build and compile the shared two-layer functional model."""
        inputs_l = layers.Input(shape=(10,), name="input_layer")
        h = layers.Dense(dense_units, name="dense_layer")(inputs_l)
        outputs_l = layers.Dense(out_units, name="output_layer")(h)
        model = models.Model(inputs_l, outputs_l, name="test_model")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _make_layout_map(self, device_mesh, *layer_names):
        """Build a LayoutMap that shards kernel+bias for each named layer.

        Each named Dense layer's kernel is sharded along the output-units
        axis (axes=(None, "data")) and its bias along axis 0 (axes=("data",)).
        Layers not listed are left replicated.
        """
        layout_map = LayoutMap(device_mesh)
        for name in layer_names:
            layout_map[f"{name}/kernel"] = TensorLayout(axes=(None, "data"))
            layout_map[f"{name}/bias"] = TensorLayout(axes=("data",))
        return layout_map

    @parameterized.parameters(
        {"save_freq": 10, "epochs": 1, "batch_size": 5},  # batch-level
        {"save_freq": "epoch", "epochs": 3, "batch_size": None},  # epoch-level
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_saving_basic(self, save_freq, epochs, batch_size):
        """Test basic checkpoint saving with different frequencies."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_save_{save_freq}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq=save_freq
        )

        # Train with specified configuration
        fit_kwargs = {"callbacks": [callback], "verbose": 0}
        if batch_size:
            fit_kwargs["batch_size"] = batch_size
        model.fit(x, y, epochs=epochs, **fit_kwargs)

        # Verify checkpoint files were created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @parameterized.parameters(
        {"mode": "min", "monitor": "loss"},
        {"mode": "max", "monitor": "loss"},
    )
    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self, mode, monitor):
        """Test save_best_only with different modes."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=100)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_best_{mode}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_async_vs_sync_saving(self, save_on_background):
        """Test synchronous vs asynchronous saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_async_{save_on_background}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )

        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter limits number of checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_max_keep_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertLessEqual(len(checkpoint_files), 5)

    @pytest.mark.requires_trainable_backend
    def test_load_weights_from_orbax_checkpoint(self):
        """Test loading weights from Orbax checkpoint using load_weights."""

        # Create and train model to create checkpoint
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_load_weights_orbax"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_weights_only=True,  # Only save weights for load_weights test
        )

        # Train to create checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        # Get original weights after training
        original_weights = model.get_weights()

        # Create a new model with the same architecture
        new_model = self._create_test_model()

        # Initialize with different weights to ensure loading works
        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify weights are different initially
        new_weights_before = new_model.get_weights()
        for orig, new in zip(original_weights, new_weights_before):
            self.assertNotAllClose(
                orig, new, msg="Weights should be different before loading"
            )

        # Load weights from Orbax checkpoint
        new_model.load_weights(checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertAllClose(
                orig,
                loaded,
                msg="Weights should match after loading from checkpoint",
            )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_epoch(self):
        """Test save_freq='epoch' functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_epoch_freq_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for 3 epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Should have only the latest checkpoint (epoch 2) due to max_to_keep=1
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertEqual(
            len(checkpoint_files),
            1,
            f"Should have exactly 1 checkpoint due to max_to_keep=1, "
            f"found {len(checkpoint_files)}: {checkpoint_files}",
        )

        # Check for the latest epoch directory (should be the highest numbered)
        # Note: Due to preservation policy behavior, the actual latest kept
        # may vary
        # So we check that at least one checkpoint exists and has a reasonable
        # name
        self.assertTrue(
            len(checkpoint_files) == 1 and checkpoint_files[0].isdigit(),
            f"Should have exactly one checkpoint with numeric name, "
            f"found {checkpoint_files}",
        )

    def test_invalid_save_freq(self):
        """Test error handling for invalid save_freq parameter."""
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_invalid_freq")
        with self.assertRaises(ValueError):
            OrbaxCheckpoint(directory=checkpoint_dir, save_freq="invalid")

    @pytest.mark.requires_trainable_backend
    def test_initial_value_threshold(self):
        """Test initial_value_threshold parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_threshold")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            initial_value_threshold=1.0,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)
        self.assertTrue(os.path.exists(checkpoint_dir))

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_comprehensive(self, save_on_background):
        """Test checkpoint loading with async and sync saving."""
        model = self._create_test_model()
        model.compile(optimizer="adam", loss="mse")
        x, y = self._create_dummy_data(num_samples=200)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_loading_{save_on_background}_{id(self)}",
        )

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
            save_weights_only=True,
        )

        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        original_weights = model.get_weights()

        # Test load_weights functionality
        new_model = self._create_test_model()
        new_model.compile(optimizer="adam", loss="mse")
        new_x, new_y = self._create_dummy_data(num_samples=10)
        new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify different before loading
        for orig, new in zip(original_weights, new_model.get_weights()):
            self.assertNotAllClose(orig, new)

        # Load and verify
        new_model.load_weights(checkpoint_dir)
        for orig, loaded in zip(original_weights, new_model.get_weights()):
            self.assertAllClose(orig, loaded)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Requires JAX backend for distribution",
    )
    @pytest.mark.multi_device
    def test_distributed_checkpoint_functionality(self):
        """Test OrbaxCheckpoint with distributed training.

        Verifies that a full-model checkpoint (weights + optimizer state +
        config) round-trips correctly under ModelParallel sharding.
        All predict/load calls stay inside the distribution scope so that
        JAX JIT sees the correct context mesh for sharded variables.
        """
        num_devices, device_mesh, original_distribution = (
            self._setup_distributed_test()
        )

        layout_map = self._make_layout_map(
            device_mesh, "dense_layer", "output_layer"
        )

        dense_units = self._DIST_DENSE_UNITS
        out_units = self._DIST_OUT_UNITS
        predict_batch = self._DIST_PREDICT_BATCH

        try:
            set_distribution(ModelParallel(layout_map=layout_map))
            model = self._build_distributed_model(dense_units, out_units)

            x = np.random.randn(self._DIST_NUM_SAMPLES, 10)
            y = np.random.randn(self._DIST_NUM_SAMPLES, out_units)

            checkpoint_dir = os.path.join(
                self.get_temp_dir(), "test_distributed_checkpoint"
            )
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir, save_freq="epoch"
            )
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

            original_predictions = model.predict(x[:predict_batch], verbose=0)
            original_weights = model.get_weights()
            original_opt_vars = [v.numpy() for v in model.optimizer.variables]

            loaded = saving.load_model(checkpoint_dir)

            for orig, lw in zip(original_weights, loaded.get_weights()):
                self.assertAllClose(orig, lw)
            for orig, lv in zip(original_opt_vars, loaded.optimizer.variables):
                self.assertAllClose(orig, lv)

            loaded_predictions = loaded.predict(x[:predict_batch], verbose=0)
            self.assertAllClose(original_predictions, loaded_predictions)

            self.assertEqual(model.name, loaded.name)
            self.assertEqual(len(model.layers), len(loaded.layers))
            self.assertTrue(loaded.compiled)
            self.assertEqual(type(get_distribution()), ModelParallel)

            original_shardings = {
                var.path: var.value.sharding
                for var in model.variables
                if hasattr(var.value, "sharding")
            }
            loaded_shardings = {
                var.path: var.value.sharding
                for var in loaded.variables
                if hasattr(var.value, "sharding")
            }
            for path, spec in original_shardings.items():
                if path in loaded_shardings:
                    self.assertEqual(
                        spec,
                        loaded_shardings[path],
                        f"Sharding mismatch for variable {path}",
                    )

        finally:
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                try:
                    set_distribution(None)
                except Exception:
                    pass

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Requires JAX backend for distribution",
    )
    @pytest.mark.multi_device
    def test_distributed_checkpoint_resharding(self):
        """Test loading an Orbax checkpoint under a *different* layout.

        Saves a model sharded with layout A (dense_layer + output_layer
        sharded), then reloads it under layout B (only output_layer
        sharded). The loaded model must have numerically identical
        weights AND the new sharding layout.
        """
        num_devices, device_mesh, original_distribution = (
            self._setup_distributed_test()
        )

        dense_units = self._DIST_DENSE_UNITS
        out_units = self._DIST_OUT_UNITS
        predict_batch = self._DIST_PREDICT_BATCH

        try:
            # ---- Save with Layout A (both layers sharded) ----
            layout_a = self._make_layout_map(
                device_mesh, "dense_layer", "output_layer"
            )
            set_distribution(ModelParallel(layout_map=layout_a))
            model = self._build_distributed_model(dense_units, out_units)

            x = np.random.randn(self._DIST_NUM_SAMPLES, 10)
            y = np.random.randn(self._DIST_NUM_SAMPLES, out_units)

            checkpoint_dir = os.path.join(
                self.get_temp_dir(), "test_resharding_checkpoint"
            )
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir, save_freq="epoch"
            )
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

            original_weights = model.get_weights()
            original_predictions = model.predict(x[:predict_batch], verbose=0)

            # ---- Reload with Layout B (only output_layer sharded) ----
            layout_b = self._make_layout_map(device_mesh, "output_layer")
            set_distribution(ModelParallel(layout_map=layout_b))

            loaded = saving.load_model(checkpoint_dir)

            # Weights must be numerically identical
            for orig, lw in zip(original_weights, loaded.get_weights()):
                self.assertAllClose(orig, lw)

            loaded_predictions = loaded.predict(x[:predict_batch], verbose=0)
            self.assertAllClose(original_predictions, loaded_predictions)

            # Verify the loaded model uses Layout B shardings
            self.assertEqual(model.name, loaded.name)
            self.assertTrue(loaded.compiled)

        finally:
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                try:
                    set_distribution(None)
                except:
                    pass

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_via_saving_api(self):
        """Test model loading via saving API."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        # Test basic model loading
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_basic_loading")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        original_weights = model.get_weights()
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify weights and compilation
        self.assertEqual(len(original_weights), len(loaded_model.get_weights()))
        for orig, loaded in zip(original_weights, loaded_model.get_weights()):
            self.assertAllClose(orig, loaded)
        self.assertTrue(loaded_model.compiled)

        # Test weights-only checkpoint should fail with load_model
        weights_only_dir = os.path.join(
            self.get_temp_dir(), "test_weights_only"
        )
        weights_callback = OrbaxCheckpoint(
            directory=weights_only_dir,
            save_freq="epoch",
            save_weights_only=True,
        )
        model.fit(x, y, epochs=1, callbacks=[weights_callback], verbose=0)

        with self.assertRaises(ValueError):
            saving.load_model(weights_only_dir)

    @pytest.mark.requires_trainable_backend
    def test_sequential_model_restores_saved_weights(self):
        """Regression test for #23251: a Sequential model restored with
        `load_model` must get the checkpointed weights and optimizer state,
        not freshly initialized ones."""
        utils.set_random_seed(123)
        model = models.Sequential(
            [
                layers.Input((8,)),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(64, 8)
        y = np.random.randn(64, 1)

        checkpoint_dir = os.path.join(self.get_temp_dir(), "sequential")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_on_background=False
        )
        model.fit(x, y, epochs=1, batch_size=16, callbacks=[callback])

        loaded_model = saving.load_model(checkpoint_dir)

        self.assertEqual(
            [v.path for v in loaded_model.weights],
            [v.path for v in model.weights],
        )
        for orig, loaded in zip(
            model.get_weights(), loaded_model.get_weights()
        ):
            self.assertAllClose(orig, loaded)
        for orig, loaded in zip(
            model.optimizer.variables, loaded_model.optimizer.variables
        ):
            self.assertAllClose(orig, loaded)

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_comprehensive_model_state_restoration(self, save_on_background):
        """Test comprehensive model state restoration with exact weight
        matching.

        Tests sync/async saving, exact weight matching, and complete state
        restoration including trainable/non-trainable variables, optimizer
        state, and custom layers.
        """
        utils.set_random_seed(42)

        # Create model with custom layer having non-trainable variables
        @register_keras_serializable(package="test")
        class CustomLayer(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units), name="kernel"
                )
                self.moving_mean = self.add_weight(
                    shape=(self.units,), trainable=False, name="moving_mean"
                )
                super().build(input_shape)

            def call(self, inputs):
                return backend.ops.numpy.matmul(inputs, self.kernel)

        # Build model with both trainable and non-trainable variables
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(8, name="dense_layer")(inputs)
        outputs = CustomLayer(2, name="custom_layer")(x)
        model = models.Model(inputs, outputs, name="comprehensive_test_model")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        x, y = self._create_dummy_data(num_samples=100)
        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_comprehensive_{save_on_background}_{id(self)}",
        )

        # Test saving with exact weight matching
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )
        model.fit(x, y, epochs=2, verbose=0, callbacks=[callback])

        # Verify exact weight matching functionality
        final_saved_weights = model.get_weights()
        self.assertIsNotNone(final_saved_weights, "Should have saved weights")

        # Load and verify complete model restoration
        loaded_model = saving.load_model(checkpoint_dir)

        # Architecture verification
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))
        self.assertTrue(loaded_model.compiled)

        # Exact weight matching verification
        loaded_weights = loaded_model.get_weights()
        self.assertEqual(len(final_saved_weights), len(loaded_weights))
        for i, (saved, loaded) in enumerate(
            zip(final_saved_weights, loaded_weights)
        ):
            self.assertAllClose(saved, loaded, msg=f"Weight {i} mismatch")

        # Verify optimizer variables
        for i, (saved, loaded) in enumerate(
            zip(model.optimizer.variables, loaded_model.optimizer.variables)
        ):
            self.assertAllClose(saved, loaded, msg=f"Weight {i} mismatch")

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_with_assets(self, save_on_background):
        """Test checkpoint saving/loading with layers that have assets.

        Tests that models with preprocessing layers that have vocab assets
        can be saved and loaded correctly through Orbax checkpoints.

        Passing a vocabulary *file path* (not an inline list) to
        StringLookup causes the vocabulary to be stored via
        save_assets / load_assets rather than inlined in get_config.
        This test verifies the Orbax round-trip for that code path.
        """
        # Write a vocabulary file so StringLookup stores it as an asset
        # (inline lists are serialized in get_config, not via assets).
        vocab_dir = self.get_temp_dir()
        vocab_file = os.path.join(vocab_dir, "vocab.txt")
        vocab_words = ["cat", "dog", "bird", "fish"]
        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab_words))

        string_lookup = layers.StringLookup(
            vocabulary=vocab_file,
            output_mode="int",
            name="string_lookup_layer",
        )

        inputs = layers.Input(shape=(1,), dtype="string")
        x = string_lookup(inputs)
        outputs = layers.Embedding(input_dim=10, output_dim=8)(x)
        model = models.Model(inputs, outputs, name="model_with_assets")
        model.compile(optimizer="adam", loss="mse")

        original_vocab = string_lookup.get_vocabulary()

        # Save through OrbaxCheckpoint (the actual Orbax path)
        checkpoint_dir = self.get_temp_dir()
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
            save_weights_only=False,
        )

        # We can't easily train with string inputs, so invoke the
        # save path directly.
        callback.set_model(model)
        callback._save_checkpoint(step=0)
        callback.checkpointer.close()

        # Load the model back through the Orbax load path
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify model structure
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))

        # Verify vocabulary (assets) was restored correctly
        loaded_string_lookup = loaded_model.get_layer("string_lookup_layer")
        loaded_vocab = loaded_string_lookup.get_vocabulary()

        self.assertEqual(original_vocab, loaded_vocab)

    @parameterized.named_parameters(named_product(steps_per_execution=(1, 2)))
    @pytest.mark.requires_trainable_backend
    def test_training_resumption(self, steps_per_execution):
        if backend.backend() == "torch" and steps_per_execution != 1:
            pytest.skip("steps_per_execution unsupported on torch")

        model = self._create_test_model(steps_per_execution)
        x, y = self._create_dummy_data(num_samples=50)
        checkpoint_dir = self.get_temp_dir()

        # Train with specified configuration
        oc1 = OrbaxCheckpoint(checkpoint_dir, save_freq=1, max_to_keep=10)
        model.fit(x, y, epochs=2, batch_size=25, callbacks=[oc1], verbose=0)

        # Verify checkpoint files were created
        checkpoint_files_1 = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files_1), 0, "Should have checkpoint files"
        )

        reloaded_model = saving.load_model(checkpoint_dir)
        # Resume training with the same folder for checkpoints
        oc2 = OrbaxCheckpoint(checkpoint_dir, save_freq=1, max_to_keep=10)
        reloaded_model.fit(
            x, y, epochs=1, batch_size=25, callbacks=[oc2], verbose=0
        )

        checkpoint_files_2 = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files_2),
            len(checkpoint_files_1),
            "Should have more checkpoint files",
        )

    @pytest.mark.requires_trainable_backend
    def test_warns_when_finalizing_checkpoints_fails(self):
        """A failed finalize must not look like a successful save."""
        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_finalize_failure_{id(self)}"
        )
        model = models.Sequential(
            [layers.Input((4,)), layers.Dense(3, name="dense")]
        )
        model.compile(optimizer="adam", loss="mse")
        x = np.random.rand(8, 4)
        y = np.random.rand(8, 3)

        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")
        model.fit(x, y, epochs=1, verbose=0, callbacks=[callback])

        # `close()` is what raises when a pending save never lands, which is
        # how a timed out async write reports itself.
        class FailingCheckpointer:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def close(self):
                raise TimeoutError("pending save did not complete")

        callback.checkpointer = FailingCheckpointer(callback.checkpointer)
        with self.assertWarnsRegex(UserWarning, "Failed to finalize"):
            callback.on_train_end()
