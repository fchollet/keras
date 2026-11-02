import os
import zipfile
from unittest import mock

import h5py
import numpy as np
import pytest

import keras
from keras.src import testing
from keras.src.saving import saving_lib
from keras.src.saving.file_editor import KerasFileEditor


def get_source_model():
    inputs = keras.Input((2,))
    x = keras.layers.Dense(3, name="mydense")(inputs)
    outputs = keras.layers.Dense(3, name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    return model


def get_target_model():
    inputs = keras.Input((2,))
    x = keras.layers.Dense(3, name="mydense")(inputs)
    x = keras.layers.Dense(3, name="myotherdense")(x)
    outputs = keras.layers.Dense(3, name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    return model


class SavingTest(testing.TestCase):
    def test_basics(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")

        model = get_source_model()
        model.save(temp_filepath)

        editor = KerasFileEditor(temp_filepath)
        editor.summary()

        target_model = get_target_model()

        out = editor.compare(model)  # Succeeds
        self.assertEqual(out["status"], "success")
        out = editor.compare(target_model)  # Fails

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 2)

        editor.rename_object("dense_3", "dense_4")
        editor.rename_object("layers/dense_4", "dense_2")
        editor.add_weights("dense_2", weights={"1": np.random.random((3,))})
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.delete_object("layers/dense_3")
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")
        editor.summary()

        temp_filepath = os.path.join(self.get_temp_dir(), "resaved.weights.h5")
        editor.save(temp_filepath)
        target_model.load_weights(temp_filepath)

        editor = KerasFileEditor(temp_filepath)
        editor.summary()
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

        editor.delete_weight("dense_2", "1")
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.add_weights("dense_2", {"1": np.zeros((7,))})
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.delete_weight("dense_2", "1")
        editor.add_weights("dense_2", {"1": np.zeros((3,))})
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

    @pytest.mark.requires_trainable_backend
    def test_scalar_weight(self):
        model = keras.Sequential(name="my_sequential")
        model.add(keras.Input(shape=(1,), name="my_input"))
        model.add(keras.layers.Dense(1, activation="sigmoid", name="my_dense"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(np.array([[1]]), np.array([[1]]), verbose=0)
        model_fpath = os.path.join(self.get_temp_dir(), "model.keras")
        weights_fpath = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save(model_fpath)
        model.save_weights(weights_fpath)

        model_editor = KerasFileEditor(model_fpath)
        self.assertEqual(
            len(keras.src.tree.flatten(model_editor.weights_dict)), 8
        )
        model_weights_editor = KerasFileEditor(weights_fpath)
        self.assertEqual(
            len(keras.src.tree.flatten(model_weights_editor.weights_dict)), 8
        )

    def test_rejects_external_link_at_group_level(self):
        target_fpath = os.path.join(
            self.get_temp_dir(), "victim_private.weights.h5"
        )
        model = keras.Sequential(
            [keras.Input(shape=(3,)), keras.layers.Dense(5)]
        )
        model.save_weights(target_fpath)

        attacker_fpath = os.path.join(
            self.get_temp_dir(), "attacker_payload.weights.h5"
        )
        with h5py.File(attacker_fpath, "w") as f:
            f["layers"] = h5py.ExternalLink(target_fpath, "/layers")

        with self.assertRaisesRegex(ValueError, "ExternalLink"):
            KerasFileEditor(attacker_fpath)

    def test_rejects_soft_link_at_group_level(self):
        attacker_fpath = os.path.join(
            self.get_temp_dir(), "softlink_payload.weights.h5"
        )
        with h5py.File(attacker_fpath, "w") as f:
            real = f.create_group("real_layers")
            real.create_dataset("inner", data=np.zeros((1,)))
            f["layers"] = h5py.SoftLink("/real_layers")

        with self.assertRaisesRegex(ValueError, "SoftLink"):
            KerasFileEditor(attacker_fpath)

    def test_rejects_virtual_dataset(self):
        temp_dir = self.get_temp_dir()
        other_fpath = os.path.join(temp_dir, "other.weights.h5")
        with h5py.File(other_fpath, "w") as f:
            f.create_dataset("s", data=np.arange(5, dtype="float32"))

        virtual_fpath = os.path.join(temp_dir, "virtual.weights.h5")
        with h5py.File(virtual_fpath, "w") as f:
            vars_group = (
                f.create_group("layers")
                .create_group("dense")
                .create_group("vars")
            )
            layout = h5py.VirtualLayout(shape=(5,), dtype="float32")
            layout[:] = h5py.VirtualSource(other_fpath, "s", shape=(5,))
            vars_group.create_virtual_dataset("0", layout)

        with self.assertRaisesRegex(ValueError, "virtual"):
            KerasFileEditor(virtual_fpath)

    def test_rejects_weights_decompression_bomb(self):
        good_fpath = os.path.join(self.get_temp_dir(), "good.keras")
        model = keras.Sequential(
            [keras.Input(shape=(3,)), keras.layers.Dense(5)]
        )
        model.save(good_fpath)

        with zipfile.ZipFile(good_fpath) as zf:
            config = zf.read(saving_lib._CONFIG_FILENAME)
            metadata = zf.read(saving_lib._METADATA_FILENAME)
            weights = zf.read(saving_lib._VARS_FNAME_H5)

        evil_fpath = os.path.join(self.get_temp_dir(), "evil.keras")
        with zipfile.ZipFile(evil_fpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(saving_lib._CONFIG_FILENAME, config)
            zf.writestr(saving_lib._METADATA_FILENAME, metadata)
            zf.writestr(saving_lib._VARS_FNAME_H5, weights + b"\0" * 200_000)

        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                KerasFileEditor(evil_fpath)

    def test_rejects_decompression_bomb_config(self):
        # A `.keras` whose config.json decompresses to far more than it stores
        # is a decompression bomb: opening the file reads that member fully
        # into memory. Keep the weights/metadata members valid so only the
        # config read is exercised.
        model = keras.Sequential(
            [keras.Input(shape=(3,)), keras.layers.Dense(5)]
        )
        good_fpath = os.path.join(self.get_temp_dir(), "good.keras")
        model.save(good_fpath)
        with zipfile.ZipFile(good_fpath) as zf:
            metadata = zf.read(saving_lib._METADATA_FILENAME)
            weights = zf.read(saving_lib._VARS_FNAME_H5)

        bomb_fpath = os.path.join(self.get_temp_dir(), "bomb.keras")
        with zipfile.ZipFile(
            bomb_fpath, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.writestr(saving_lib._METADATA_FILENAME, metadata)
            zf.writestr(
                saving_lib._CONFIG_FILENAME, b'{"x":"' + b" " * 200_000 + b'"}'
            )
            info = zipfile.ZipInfo(saving_lib._VARS_FNAME_H5)
            zf.writestr(info, weights)

        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                KerasFileEditor(bomb_fpath)
