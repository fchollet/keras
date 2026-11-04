import os
import warnings
from unittest import mock

import pytest

from keras.src import callbacks
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.models import Sequential
from keras.src.testing import test_utils
from keras.src.utils import file_utils
from keras.src.utils import numerical_utils
from keras.src.utils.module_utils import LazyModule

try:
    import h5py
except ImportError:
    h5py = None

TRAIN_SAMPLES = 30
TEST_SAMPLES = 30
NUM_CLASSES = 3
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class ModelCheckpointTest(testing.TestCase):
    @pytest.mark.skipif(
        h5py is None,
        reason="`h5py` is a required dependency for `ModelCheckpoint` tests.",
    )
    @pytest.mark.skipif(
        testing.jax_uses_gpu(),
        reason="Mysterious core dump on CI after upgrading JAX",
    )
    @pytest.mark.requires_trainable_backend
    def test_model_checkpoint_options(self):
        def get_model():
            model = Sequential(
                [
                    layers.Dense(NUM_HIDDEN, activation="relu"),
                    layers.Dense(NUM_CLASSES, activation="softmax"),
                ]
            )
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=[metrics.Accuracy("acc")],
            )
            return model

        model = get_model()
        temp_dir = self.get_temp_dir()

        # Save model to a subdir inside the temp_dir so we can test
        # automatic directory creation.
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.keras")
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            random_seed=42,
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = numerical_utils.to_categorical(y_test, num_classes=NUM_CLASSES)
        y_train = numerical_utils.to_categorical(
            y_train, num_classes=NUM_CLASSES
        )

        # Case 1
        monitor = "val_loss"
        save_best_only = False
        mode = "auto"

        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 2
        mode = "min"
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 3
        mode = "max"
        monitor = "val_acc"
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 4
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 5: metric not available.
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor="unknown", save_best_only=True, mode="min"
            )
        ]
        with pytest.warns(UserWarning):
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=1,
                verbose=0,
            )
        self.assertTrue(os.path.exists(filepath))

        # Case 6
        with warnings.catch_warnings(record=True) as warning_logs:
            warnings.simplefilter("always")
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode="unknown",
            )
            self.assertIn(
                "ModelCheckpoint mode 'unknown' is unknown",
                str(warning_logs[-1].message),
            )

        # Case 8a: `ModelCheckpoint` with an integer `save_freq`
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.keras")
        save_best_only = False
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq=15,
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        model.fit(
            x_train,
            y_train,
            batch_size=6,  # 5 batches / epoch, so should backup every 3 epochs
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5)))
        self.assertTrue(os.path.exists(filepath.format(epoch=6)))
        self.assertFalse(os.path.exists(filepath.format(epoch=7)))
        self.assertFalse(os.path.exists(filepath.format(epoch=8)))
        self.assertTrue(os.path.exists(filepath.format(epoch=9)))
        os.remove(filepath.format(epoch=3))
        os.remove(filepath.format(epoch=6))
        os.remove(filepath.format(epoch=9))

        # Case 8b: `ModelCheckpoint` with int `save_freq` & `save_weights_only`
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}.weights.h5"
        )
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=15, save_weights_only=True
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        model.fit(
            x_train,
            y_train,
            batch_size=6,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5)))
        self.assertTrue(os.path.exists(filepath.format(epoch=6)))
        self.assertFalse(os.path.exists(filepath.format(epoch=7)))
        self.assertFalse(os.path.exists(filepath.format(epoch=8)))
        self.assertTrue(os.path.exists(filepath.format(epoch=9)))

        # Case 9: `ModelCheckpoint` with valid and invalid save_freq argument.
        with self.assertRaisesRegex(ValueError, "Unrecognized save_freq"):
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=True,
                mode=mode,
                save_freq="invalid_save_freq",
            )
        # The following should not raise ValueError.
        callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=True,
            mode=mode,
            save_freq="epoch",
        )
        callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=True,
            mode=mode,
            save_freq=3,
        )

        # Case 10a: `ModelCheckpoint` save with batch in filename.
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.keras"
        )
        cbks = [
            callbacks.ModelCheckpoint(filepath, monitor=monitor, save_freq=1)
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=15,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=1,
        )
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=2)))

        # Case 10b: `ModelCheckpoint` save weights with batch in filename.
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.weights.h5"
        )
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=1, save_weights_only=True
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=15,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=1,
        )

        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=2)))

        # Case 11: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "max"
        monitor = "val_acc"
        initial_value_threshold = -0.01
        save_best_only = True
        filepath = os.path.join(temp_dir, "checkpoint.keras")
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 12: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = None
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 13: ModelCheckpoint doesn't save model if loss was minimum
        # earlier
        mode = "min"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath))

        # Case 14: ModelCheckpoint doesn't save model if loss was min earlier in
        # auto mode
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath))

        # Case 15: ModelCheckpoint doesn't save model if auc was max earlier in
        # auto mode
        mode = "auto"
        monitor = "val_auc"
        initial_value_threshold = 1
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=[metrics.AUC()],
        )
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath))

    @pytest.mark.skipif(
        h5py is None,
        reason="`h5py` is a required dependency for `ModelCheckpoint` tests.",
    )
    @pytest.mark.requires_trainable_backend
    def test_model_checkpoint_loading(self):
        def get_model():
            inputs = layers.Input(shape=(INPUT_DIM,), batch_size=5)
            x = layers.Dense(NUM_HIDDEN, activation="relu")(inputs)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            functional_model = models.Model(inputs, outputs)
            functional_model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=[metrics.Accuracy("acc")],
            )
            return functional_model

        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            random_seed=42,
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = numerical_utils.to_categorical(y_test, num_classes=NUM_CLASSES)
        y_train = numerical_utils.to_categorical(
            y_train, num_classes=NUM_CLASSES
        )

        # Model Checkpoint load model (default)
        model = get_model()
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.model.keras")
        mode = "auto"
        monitor = "val_loss"
        save_best_only = True

        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        ref_weights = model.get_weights()
        self.assertTrue(os.path.exists(filepath))
        new_model = saving.load_model(filepath)
        new_weights = new_model.get_weights()
        self.assertEqual(len(ref_weights), len(new_weights))
        for ref_w, w in zip(ref_weights, new_weights):
            self.assertAllClose(w, ref_w)

        # Model Checkpoint load model weights
        model = get_model()
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.weights.h5")
        mode = "auto"
        monitor = "val_loss"
        save_best_only = True

        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=True,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        ref_weights = model.get_weights()
        self.assertTrue(os.path.exists(filepath))
        new_model = get_model()
        new_model.load_weights(filepath)
        new_weights = new_model.get_weights()
        self.assertEqual(len(ref_weights), len(new_weights))
        for ref_w, w in zip(ref_weights, new_weights):
            self.assertAllClose(w, ref_w)

    def test_get_most_recently_modified_file_matching_pattern_remote(self):
        test_dir = self.get_temp_dir()
        old_path = os.path.join(test_dir, "epoch01.keras")
        new_path = os.path.join(test_dir, "epoch02.keras")
        for mtime, path in enumerate([old_path, new_path], start=1):
            open(path, "w").close()
            os.utime(path, (mtime, mtime))

        cbk = callbacks.ModelCheckpoint(os.path.join(test_dir, "x.keras"))
        with (
            mock.patch.object(file_utils, "is_remote_path", lambda path: True),
            mock.patch.object(
                file_utils, "gfile", LazyModule("keras_test_unavailable_module")
            ),
        ):
            self.assertEqual(
                cbk._get_most_recently_modified_file_matching_pattern(
                    os.path.join(test_dir, "epoch{epoch:02d}.keras")
                ),
                new_path,
            )
