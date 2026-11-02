import os.path
import tempfile

import pytest
import tensorflow as tf
from tensorflow.python.eager import context

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import losses
from keras.src import metrics
from keras.src import optimizers
from keras.src import saving
from keras.src import testing
from keras.src.models import Sequential
from keras.src.testing import test_utils
from keras.src.utils import numerical_utils


class SwapEMAWeightsTest(testing.TestCase):
    def setUp(self):
        super().setUp()

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(3,),
            num_classes=2,
            random_seed=2023,
        )
        y_train = numerical_utils.to_categorical(y_train)

        self.x_train = x_train
        self.y_train = y_train

    def _get_compiled_model(
        self, use_ema=True, jit_compile=True, loss_scale=False
    ):
        optimizer = optimizers.SGD(use_ema=use_ema, ema_momentum=0.9)
        if loss_scale:
            optimizer = optimizers.LossScaleOptimizer(optimizer)
        model = Sequential(
            [layers.Dense(2, kernel_initializer="ones", use_bias=False)]
        )
        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            jit_compile=jit_compile,
        )
        return model

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_with_invalid_optimizer(self):
        model = self._get_compiled_model(use_ema=False)
        with self.assertRaisesRegex(
            ValueError,
            ("SwapEMAWeights must be used when `use_ema=True` is set"),
        ):
            model.fit(
                self.x_train,
                self.y_train,
                epochs=2,
                callbacks=[callbacks.SwapEMAWeights()],
                validation_data=(self.x_train, self.y_train),
            )

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights(self):
        # not using SwapEMAWeights
        model = self._get_compiled_model()
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            validation_data=(self.x_train, self.y_train),
        )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        # final metric during fitting is different from the evaluation
        self.assertNotEqual(
            history.history["val_mean_squared_error"][-1],
            logs["mean_squared_error"],
        )

        # using SwapEMAWeights
        model = self._get_compiled_model()
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            callbacks=[callbacks.SwapEMAWeights()],
            validation_data=(self.x_train, self.y_train),
        )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        # final metric during fitting is same as the evaluation
        self.assertEqual(
            history.history["val_mean_squared_error"][-1],
            logs["mean_squared_error"],
        )

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_on_predict(self):
        model = self._get_compiled_model()
        # the first step sets the EMA weights to the trained weights, so a
        # second step is needed to make them differ
        model.train_on_batch(self.x_train, self.y_train)
        model.train_on_batch(self.x_train, self.y_train)
        weights = model.get_weights()
        y_pred = model.predict(self.x_train)

        y_pred_ema = model.predict(
            self.x_train, callbacks=[callbacks.SwapEMAWeights()]
        )
        # prediction uses the EMA weights
        self.assertNotAllClose(y_pred_ema, y_pred)
        # the model weights are restored after prediction
        for weight, new_weight in zip(weights, model.get_weights()):
            self.assertAllClose(new_weight, weight)

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_on_epoch(self):
        # using SwapEMAWeights together with ModelCheckpoint
        model = self._get_compiled_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model.fit(
                self.x_train,
                self.y_train,
                epochs=2,
                callbacks=[
                    callbacks.SwapEMAWeights(swap_on_epoch=True),
                    callbacks.ModelCheckpoint(
                        os.path.join(temp_dir, "{epoch:1d}.keras")
                    ),
                ],
                validation_data=(self.x_train, self.y_train),
            )
            model2 = saving.load_model(os.path.join(temp_dir, "2.keras"))

        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        logs2 = model2.evaluate(self.x_train, self.y_train, return_dict=True)
        # saved checkpoint will be applied by EMA weights
        self.assertEqual(
            logs["mean_squared_error"],
            logs2["mean_squared_error"],
        )

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_with_loss_scale_optimizer(self):
        model = self._get_compiled_model(loss_scale=True)
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            callbacks=[callbacks.SwapEMAWeights()],
            validation_data=(self.x_train, self.y_train),
        )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        # final metric during fitting is same as the evaluation
        self.assertEqual(
            history.history["val_mean_squared_error"][-1],
            logs["mean_squared_error"],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="The distribute test can only run with TF backend.",
    )
    def test_swap_ema_weights_with_tf_distribute(self):
        # Need at least 2 devices for distribution related tests.
        cpus = tf.config.list_physical_devices("CPU")
        context._reset_context()
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ],
        )
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        with strategy.scope():
            # TODO: set jit_compile=True once the issue is resolved in
            # integration_tests/tf_distribute_training_test.py#L52
            model = self._get_compiled_model(jit_compile=False)
            with tempfile.TemporaryDirectory() as temp_dir:
                model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=2,
                    callbacks=[
                        callbacks.SwapEMAWeights(swap_on_epoch=True),
                        callbacks.ModelCheckpoint(
                            os.path.join(
                                temp_dir, "distributed_{epoch:1d}.keras"
                            )
                        ),
                    ],
                    validation_data=(self.x_train, self.y_train),
                )
                model2 = saving.load_model(
                    os.path.join(temp_dir, "distributed_2.keras")
                )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        logs2 = model2.evaluate(self.x_train, self.y_train, return_dict=True)
        # saved checkpoint will be applied by EMA weights
        self.assertEqual(
            logs["mean_squared_error"],
            logs2["mean_squared_error"],
        )
