import numpy as np

from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import regularizers
from keras.src import testing
from keras.src.backend.common import AutocastScope


class LayerNormalizationTest(testing.TestCase):
    def test_ln_basics(self):
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={
                "gamma_regularizer": regularizers.L2(0.01),
                "beta_regularizer": regularizers.L2(0.01),
            },
            input_shape=(3, 4, 2),
            expected_output_shape=(3, 4, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={
                "gamma_initializer": "ones",
                "beta_initializer": "ones",
            },
            input_shape=(3, 4, 2),
            expected_output_shape=(3, 4, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"scale": False, "center": False},
            input_shape=(3, 3),
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"rms_scaling": True},
            input_shape=(3, 3),
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"axis": (-3, -2, -1)},
            input_shape=(2, 8, 8, 3),
            expected_output_shape=(2, 8, 8, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={},
            input_shape=(1, 0, 10),
            expected_output_shape=(1, 0, 10),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_invalid_axis(self):
        with self.assertRaisesRegex(
            TypeError,
            ("Expected an int or a list/tuple of ints for the argument 'axis'"),
        ):
            layers.LayerNormalization(axis={"axis": -1})

    def test_out_of_bounds_axis(self):
        # `build()` used to index `input_shape` directly, so an out-of-bounds
        # axis surfaced as a bare `IndexError: tuple index out of range`
        # instead of the `ValueError` already raised by
        # `compute_output_shape()`.
        x = np.zeros((2, 4), dtype="float32")
        for axis in (2, 99, -3, -99):
            with self.assertRaisesRegex(ValueError, "is out of bounds"):
                layers.LayerNormalization(axis=axis)(x)
        # A list of axes is validated element-wise.
        with self.assertRaisesRegex(ValueError, "is out of bounds"):
            layers.LayerNormalization(axis=[0, 99])(x)
        # `build()` and `compute_output_shape()` agree.
        with self.assertRaisesRegex(ValueError, "is out of bounds"):
            layers.LayerNormalization(axis=99).build((2, 4))
        with self.assertRaisesRegex(ValueError, "is out of bounds"):
            layers.LayerNormalization(axis=99).compute_output_shape((2, 4))

    def test_duplicate_axis(self):
        # `[-1, 1]` refers to the same axis twice for a rank 2 input, which
        # used to build mismatched weights and fail in the backend.
        x = np.zeros((2, 4), dtype="float32")
        with self.assertRaisesRegex(ValueError, "Duplicate axes"):
            layers.LayerNormalization(axis=[-1, 1])(x)
        with self.assertRaisesRegex(ValueError, "Duplicate axes"):
            layers.LayerNormalization(axis=[1, 1])(x)

    def test_axis_is_canonicalized(self):
        layer = layers.LayerNormalization(axis=-1)
        layer.build((2, 4))
        self.assertEqual(layer.axis, [1])

    def test_correctness(self):
        layer = layers.LayerNormalization(dtype="float32")
        layer.build(input_shape=(2, 2, 2))
        inputs = np.random.normal(
            loc=5.0, scale=10.0, size=(1000, 2, 2, 2)
        ).astype("float32")

        out = layer(inputs)
        out = ops.subtract(out, layer.beta)
        out = ops.divide(out, layer.gamma)

        self.assertAllClose(ops.mean(out), 0.0, atol=1e-1)
        self.assertAllClose(ops.std(out), 1.0, atol=1e-1)

    def test_output(self):
        layer = layers.LayerNormalization(
            dtype="float32",
            beta_initializer="ones",
            gamma_initializer="ones",
        )
        inputs = np.arange(5).astype("float32")[None, :]
        out = layer(inputs)
        self.assertAllClose(out, [[-0.41386, 0.29307, 1.0, 1.70693, 2.41386]])

    def test_output_with_rms_scaling(self):
        layer = layers.LayerNormalization(
            dtype="float32",
            rms_scaling=True,
            gamma_initializer="ones",
        )
        inputs = np.arange(5).astype("float32")[None, :]
        out = layer(inputs)
        self.assertAllClose(out, [[0.0, 0.70693, 1.41386, 2.12079, 2.82772]])

    def test_large_value_within_autocast_scope(self):
        layer = layers.LayerNormalization()
        layer.build((1, 4, 4, 3))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.gamma.shape, 70000)
        with AutocastScope("float16"):
            layer.gamma.assign(large_value)
            self.assertAllClose(layer.gamma.value, large_value)

    def test_beta_preserved_for_large_constant_inputs(self):
        x = ops.full((2, 1), 1e10)
        layer = layers.LayerNormalization(
            axis=-1,
            epsilon=1e-3,
            gamma_initializer="ones",
            beta_initializer=initializers.Constant(7.0),
        )
        self.assertAllClose(layer(x), ops.full((2, 1), 7.0))

        layer = layers.LayerNormalization(
            axis=-1,
            epsilon=1e-3,
            gamma_initializer=initializers.Constant(3.0),
            beta_initializer=initializers.Constant(7.0),
        )
        self.assertAllClose(layer(x), ops.full((2, 1), 7.0))

    def test_unsorted_axis(self):
        x = np.random.randn(2, 3, 4).astype("float32")
        layer_sorted = layers.LayerNormalization(axis=[-2, -1])
        layer_unsorted = layers.LayerNormalization(axis=[-1, -2])
        out_sorted = layer_sorted(x)
        out_unsorted = layer_unsorted(x)
        self.assertEqual(out_sorted.shape, (2, 3, 4))
        self.assertEqual(out_unsorted.shape, (2, 3, 4))
        self.assertAllClose(out_sorted, out_unsorted)
