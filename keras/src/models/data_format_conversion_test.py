import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.models.data_format_conversion import convert_data_format
from keras.src.models.functional import Functional
from keras.src.models.sequential import Sequential


def _build_sequential_cnn(data_format):
    return Sequential(
        [
            layers.Input(
                shape=(8, 8, 3) if data_format == "channels_last" else (3, 8, 8)
            ),
            layers.Conv2D(4, 3, padding="same", data_format=data_format),
            layers.BatchNormalization(
                axis=-1 if data_format == "channels_last" else 1
            ),
            layers.GlobalAveragePooling2D(data_format=data_format),
            layers.Dense(2),
        ]
    )


def _build_functional_cnn(data_format):
    inputs = layers.Input(
        shape=(8, 8, 3) if data_format == "channels_last" else (3, 8, 8)
    )
    x = layers.Conv2D(4, 3, padding="same", data_format=data_format)(inputs)
    x = layers.BatchNormalization(
        axis=-1 if data_format == "channels_last" else 1
    )(x)
    x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
    outputs = layers.Dense(2)(x)
    return Functional(inputs, outputs)


class ConvertDataFormatTest(testing.TestCase):
    def test_rejects_bad_target(self):
        m = _build_sequential_cnn("channels_last")
        with self.assertRaisesRegex(ValueError, "target_data_format"):
            convert_data_format(m, "NCHW")

    def test_rejects_subclassed_model(self):
        from keras.src.models.model import Model

        class Sub(Model):
            def call(self, x):
                return x

        with self.assertRaisesRegex(TypeError, "Sequential or Functional"):
            convert_data_format(Sub(), "channels_first")

    def test_noop_when_already_target_format(self):
        m = _build_sequential_cnn("channels_last")
        out = convert_data_format(m, "channels_last")
        self.assertIs(out, m)

    def _skip_if_tf_cpu_nchw_unsupported(self):
        # The TF backend's Conv2D kernel only supports NHWC on CPU, so we
        # can't *execute* a freshly converted channels_first model under
        # tensorflow. The conversion itself is still verified.
        if backend.backend() == "tensorflow":
            self.skipTest(
                "TF Conv2D does not support NCHW on CPU; conversion "
                "produces a structurally correct model but cannot be run."
            )

    def test_sequential_channels_last_to_channels_first(self):
        self._skip_if_tf_cpu_nchw_unsupported()
        m = _build_sequential_cnn("channels_last")
        x_cl = np.random.uniform(size=(2, 8, 8, 3)).astype("float32")
        y_cl = backend.convert_to_numpy(m(x_cl, training=False))

        m_cf = convert_data_format(m, "channels_first")
        x_cf = np.transpose(x_cl, (0, 3, 1, 2))
        y_cf = backend.convert_to_numpy(m_cf(x_cf, training=False))

        # Outputs are Dense(2), already channel-collapsed by global pool,
        # so they should match directly.
        self.assertAllClose(y_cl, y_cf, atol=1e-5)

    def test_functional_channels_last_to_channels_first(self):
        self._skip_if_tf_cpu_nchw_unsupported()
        m = _build_functional_cnn("channels_last")
        x_cl = np.random.uniform(size=(2, 8, 8, 3)).astype("float32")
        y_cl = backend.convert_to_numpy(m(x_cl, training=False))

        m_cf = convert_data_format(m, "channels_first")
        x_cf = np.transpose(x_cl, (0, 3, 1, 2))
        y_cf = backend.convert_to_numpy(m_cf(x_cf, training=False))

        self.assertAllClose(y_cl, y_cf, atol=1e-5)

    def test_sequential_structural_conversion_on_any_backend(self):
        # On any backend, verify the converted model's input/output shapes
        # match the channel-moved expectation, even where we can't run it.
        m = _build_sequential_cnn("channels_last")
        m_cf = convert_data_format(m, "channels_first")
        # Sequential's input shape comes from its first layer (InputLayer).
        self.assertEqual(tuple(m_cf.inputs[0].shape), (None, 3, 8, 8))

    def test_channels_first_to_channels_last_roundtrip(self):
        m = _build_functional_cnn("channels_last")
        m_cf = convert_data_format(m, "channels_first")
        m_back = convert_data_format(m_cf, "channels_last")

        x = np.random.uniform(size=(2, 8, 8, 3)).astype("float32")
        y_orig = backend.convert_to_numpy(m(x, training=False))
        y_back = backend.convert_to_numpy(m_back(x, training=False))
        self.assertAllClose(y_orig, y_back, atol=1e-5)

    def test_weights_are_preserved(self):
        m = _build_sequential_cnn("channels_last")
        m_cf = convert_data_format(m, "channels_first")
        for old, new in zip(m.weights, m_cf.weights):
            self.assertAllClose(
                backend.convert_to_numpy(old.value),
                backend.convert_to_numpy(new.value),
            )

    def test_concatenate_does_not_block_conversion(self):
        # `Concatenate` is in the "ambiguous" set: its `axis` depends on
        # data_format and cannot be rewritten reliably from config alone.
        # The conversion must still succeed (and emit a warning, per the
        # docstring) — we just check that the structural conversion still
        # produces a valid channels_first model.
        inputs = layers.Input(shape=(8, 8, 3))
        a = layers.Conv2D(4, 3, padding="same", data_format="channels_last")(
            inputs
        )
        b = layers.Conv2D(4, 3, padding="same", data_format="channels_last")(
            inputs
        )
        merged = layers.Concatenate(axis=-1)([a, b])
        outputs = layers.GlobalAveragePooling2D(data_format="channels_last")(
            merged
        )
        m = Functional(inputs, outputs)

        m_cf = convert_data_format(m, "channels_first")
        self.assertEqual(tuple(m_cf.inputs[0].shape), (None, 3, 8, 8))
