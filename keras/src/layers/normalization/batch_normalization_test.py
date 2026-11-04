import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src import utils
from keras.src.backend.common import AutocastScope
from keras.src.losses import MeanSquaredError
from keras.src.models import Model


class BatchNormalizationTest(testing.TestCase):
    def test_bn_basics(self):
        # vector case
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": False,
                "scale": False,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # image case, with regularizers
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "beta_regularizer": "l2",
                "gamma_regularizer": "l2",
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    @parameterized.product(
        axis=(-1, 1),
        input_shape=((5, 2, 3), (5, 3, 3, 2)),
        moving_mean_initializer=("zeros", "ones"),
        moving_variance_initializer=("zeros", "ones"),
    )
    def test_correctness(
        self,
        axis,
        input_shape,
        moving_mean_initializer,
        moving_variance_initializer,
    ):
        # Training
        layer = layers.BatchNormalization(
            axis=axis,
            momentum=0,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
        )
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=input_shape)
        out = x
        for _ in range(3):
            out = layer(out, training=True)

        # Assert the normalization is correct.
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
        out = backend.ops.convert_to_numpy(out)
        out = out - np.reshape(
            backend.ops.convert_to_numpy(layer.beta), broadcast_shape
        )
        out = out / np.reshape(
            backend.ops.convert_to_numpy(layer.gamma), broadcast_shape
        )

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        reduction_axes = tuple(reduction_axes)
        self.assertAllClose(
            np.mean(out, axis=reduction_axes),
            np.zeros((input_shape[axis],)),
            atol=1e-3,
        )
        self.assertAllClose(
            np.std(out, axis=reduction_axes),
            np.ones((input_shape[axis],)),
            atol=1e-3,
        )
        self.assertAllClose(
            layer.moving_mean, np.zeros((input_shape[axis],)), atol=1e-3
        )
        self.assertAllClose(
            layer.moving_variance, np.ones((input_shape[axis],)), atol=1e-3
        )

        # Inference done before training shouldn't match.
        inference_out = layer(x, training=False)
        training_out = layer(x, training=True)
        self.assertNotAllClose(inference_out, training_out)

        # Since momentum is zero, inference after training should match.
        training_out = layer(x, training=True)
        inference_out = layer(x, training=False)
        self.assertAllClose(inference_out, training_out)

        # Masked result with no training should not differ
        x[:, 1, :] = 0.0
        unmasked_out = layer(x, training=False)
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=False)
        self.assertAllClose(unmasked_out, masked_out)

        # Masked result should differ from unmasked result
        unmasked_out = layer(x, training=False)
        x[:, 1, :] = 0.0
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=True)
        self.assertNotAllClose(unmasked_out, masked_out)

    @parameterized.product(
        synchronized=(
            (False, True) if backend.backend == "tensorflow" else (False,)
        ),
    )
    def test_input_fully_masked(self, synchronized):
        norm = layers.BatchNormalization(
            scale=False,
            center=False,
            synchronized=synchronized,
        )
        x = np.zeros((4, 5))
        mask = np.zeros((4,), dtype=np.float32)
        y = norm(x, mask=mask, training=True)
        self.assertAllClose(y, np.zeros_like(x, dtype=np.float32))

    @parameterized.product(run_eagerly=(True, False), mask_value=(0.0, 0.1, 1))
    @pytest.mark.requires_trainable_backend
    def test_bachnorm_ignore_masked_values(self, run_eagerly, mask_value):
        padded_data = np.array(
            [
                [
                    [1, 5],
                    [2, 5],
                    [mask_value, mask_value],
                    [mask_value, mask_value],
                ]
                for _ in range(10)
            ],
            dtype="float32",
        )

        inputs = layers.Input((None, 2))
        masked = layers.Masking(mask_value=mask_value)(inputs)
        normed = layers.BatchNormalization(momentum=0.0)(masked)
        model = Model(inputs, normed)
        loss = MeanSquaredError()
        model.compile(
            "rmsprop",
            loss=loss,
            run_eagerly=run_eagerly,
        )
        model.fit(x=padded_data, y=padded_data, batch_size=10, epochs=5)
        self.assertAllClose(model.layers[2].moving_mean, [1.5, 5.0])
        self.assertAllClose(model.layers[2].moving_variance, [0.25, 0.0])

    def test_trainable_behavior(self):
        layer = layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=1e-7)
        layer.build((1, 4, 4, 3))
        layer.trainable = False
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)

        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(200, 4, 4, 3))

        out = layer(x, training=True)
        self.assertAllClose(out, x)

        layer.trainable = True
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)

        for _ in range(10):
            out = layer(x, training=True)

        out = backend.ops.convert_to_numpy(out)
        out = out - np.reshape(
            backend.ops.convert_to_numpy(layer.beta), (1, 1, 1, 3)
        )
        out = out / np.reshape(
            backend.ops.convert_to_numpy(layer.gamma), (1, 1, 1, 3)
        )

        self.assertAllClose(
            np.mean(out, axis=(0, 1, 2)), np.zeros((3,)), atol=1e-3
        )
        self.assertAllClose(
            np.std(out, axis=(0, 1, 2)), np.ones((3,)), atol=1e-3
        )

    def test_large_value_within_autocast_scope(self):
        layer = layers.BatchNormalization()
        layer.build((1, 4, 4, 3))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.moving_variance.shape, 70000)
        with AutocastScope("float16"):
            layer.moving_variance.assign(large_value)
            self.assertAllClose(layer.moving_variance.value, large_value)

    def test_masked_broadcast_normalization(self):
        input_shape = (1, 2, 3, 4)
        mask_shape = (1, 2, 1)
        x = ops.ones(input_shape)
        mask = ops.ones(mask_shape)

        layer = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)

        y = layer(x, training=True, mask=mask)

        mean_y = ops.mean(y, axis=[0, 1, 2])

        self.assertAllClose(mean_y, ops.zeros((4,)), atol=1e-6)
        self.assertAllClose(y, ops.zeros_like(y), atol=1e-6)

        self.assertAllClose(layer.moving_mean, ops.ones((4,)), atol=1e-6)
        self.assertAllClose(layer.moving_variance, ops.zeros((4,)), atol=1e-6)

    def test_renorm_basics(self):
        # Test basic renorm functionality
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "renorm": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            # moving_mean, moving_variance, moving_stddev, renorm_mean,
            # renorm_stddev
            expected_num_non_trainable_weights=5,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # Test renorm with clipping
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "renorm": True,
                "renorm_clipping": {"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=5,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_renorm_invalid_clipping_keys(self):
        with self.assertRaisesRegex(ValueError, "Received invalid keys"):
            layers.BatchNormalization(
                renorm=True, renorm_clipping={"random_key": 1.0}
            )
        with self.assertRaisesRegex(ValueError, "rmax should be"):
            layers.BatchNormalization(
                renorm=True, renorm_clipping={"rmax": 0.0, "rmin": 1.0}
            )
        with self.assertRaisesRegex(ValueError, "dmax should be non-negative"):
            layers.BatchNormalization(
                renorm=True, renorm_clipping={"rmax": 1.0, "dmax": -1.0}
            )

    def test_invalid_momentum(self):
        for momentum in (-0.1, 1.5):
            with self.assertRaisesRegex(ValueError, "`momentum` must be"):
                layers.BatchNormalization(momentum=momentum)
        # Boundary values are accepted.
        layers.BatchNormalization(momentum=0.0)
        layers.BatchNormalization(momentum=1.0)

    def test_renorm_stddev_initializer(self):
        # `moving_stddev` and `renorm_stddev` should be initialized as
        # `sqrt` of `moving_variance_initializer`.
        layer = layers.BatchNormalization(
            renorm=True,
            moving_variance_initializer=initializers.Constant(4.0),
        )
        layer.build((None, 5))

        self.assertAllClose(layer.moving_stddev, np.full((5,), 2.0), atol=1e-6)
        self.assertAllClose(layer.renorm_stddev, np.full((5,), 2.0), atol=1e-6)

    def test_renorm_inference(self):
        # At inference time, the behaviour of both with and without renorm
        # should be the same.
        bn = layers.BatchNormalization(renorm=False)
        bn_renorm = layers.BatchNormalization(renorm=True)

        bn.build((None, 10))
        bn_renorm.build((None, 10))

        # Copy the vars to renorm layer.
        for attr in ["gamma", "beta", "moving_mean", "moving_variance"]:
            getattr(bn, attr).assign(random.normal(shape=(10,)))
            getattr(bn_renorm, attr).assign(getattr(bn, attr))

        x = np.random.normal(size=(4, 10))
        out = bn(x, training=False)
        out_renorm = bn_renorm(x, training=False)

        self.assertAllClose(out, out_renorm, atol=1e-5, rtol=1e-5)

    def test_renorm_correctness(self):
        utils.set_random_seed(1337)

        epsilon = 1e-3
        momentum = 0.9
        renorm_momentum = 0.8

        # Create layer
        layer = layers.BatchNormalization(
            axis=-1,
            epsilon=epsilon,
            momentum=momentum,
            renorm=True,
            renorm_momentum=renorm_momentum,
        )
        layer.build((None, 3))

        # Assign initial values.
        size = (3,)
        init_moving_mean = np.random.normal(0.0, 1.0, size=size)
        init_moving_var = np.abs(np.random.normal(1.0, 0.5, size=size))
        init_moving_stddev = np.sqrt(init_moving_var)
        init_renorm_mean = np.random.normal(0.0, 1.0, size=size)
        init_renorm_stddev = np.abs(np.random.normal(1.0, 0.5, size=size))
        init_gamma = np.random.normal(1.0, 0.1, size=size)
        init_beta = np.random.normal(0.0, 0.1, size=size)

        layer.moving_mean.assign(init_moving_mean)
        layer.moving_variance.assign(init_moving_var)
        layer.moving_stddev.assign(init_moving_stddev)
        layer.renorm_mean.assign(init_renorm_mean)
        layer.renorm_stddev.assign(init_renorm_stddev)
        layer.gamma.assign(init_gamma)
        layer.beta.assign(init_beta)

        # Input data
        x = np.array(
            [[4.0, 6.0, 2.0], [8.0, -2.0, 5.0], [6.0, 4.0, 3.0]],
            dtype="float32",
        )

        # Manually compute expected output.
        # Normalise input.
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_stddev = np.sqrt(batch_var + epsilon)

        # Compute r, d, and then expected output.
        r = batch_stddev / init_renorm_stddev
        d = (batch_mean - init_renorm_mean) / init_renorm_stddev

        expected_output = (
            ((x - batch_mean) / batch_stddev) * r + d
        ) * init_gamma + init_beta
        actual_output = layer(x, training=True)
        self.assertAllClose(actual_output, expected_output, atol=1e-5)

        # Verify moving statistics.
        expected_renorm_mean = (
            init_renorm_mean * renorm_momentum
            + batch_mean * (1 - renorm_momentum)
        )
        self.assertAllClose(
            layer.renorm_mean,
            expected_renorm_mean,
            atol=1e-5,
        )
        expected_renorm_stddev = (
            init_renorm_stddev * renorm_momentum
            + batch_stddev * (1 - renorm_momentum)
        )
        self.assertAllClose(
            layer.renorm_stddev,
            expected_renorm_stddev,
            atol=1e-5,
        )
        expected_moving_mean = init_moving_mean * momentum + batch_mean * (
            1 - momentum
        )
        self.assertAllClose(
            layer.moving_mean,
            expected_moving_mean,
            atol=1e-5,
        )
        expected_moving_stddev = (
            init_moving_stddev * momentum + batch_stddev * (1 - momentum)
        )
        self.assertAllClose(
            layer.moving_stddev,
            expected_moving_stddev,
            atol=1e-5,
        )
        expected_moving_var = expected_moving_stddev**2 - epsilon
        self.assertAllClose(
            layer.moving_variance,
            expected_moving_var,
            atol=1e-5,
        )

    def test_serialization(self):
        layer = layers.BatchNormalization(
            renorm=True,
            renorm_clipping={"rmax": 3.0, "rmin": 0.3, "dmax": 5.0},
            renorm_momentum=0.95,
        )

        config = layer.get_config()
        new_layer = layers.BatchNormalization.from_config(config)
        self.assertEqual(new_layer.get_config(), config)
