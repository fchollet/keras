# flake8: noqa

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.eager import context

from keras.src import backend
from keras.src import testing
from keras.src.optimizers.sgd import SGD


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The distribute test can only run with TF backend.",
)
class OptimizerDistributeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
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
        self.strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

    def test_config(self):
        with self.strategy.scope():
            optimizer = SGD(
                learning_rate=0.5,
                momentum=0.06,
                nesterov=True,
                weight_decay=0.004,
            )
        self.run_class_serialization_test(optimizer)

    @parameterized.parameters([("keras_sgd",), ("tf_keras_sgd",)])
    def test_single_step(self, optimizer_type):
        if optimizer_type == "tf_keras_sgd":
            try:
                import tf_keras

                optimizer_fn = tf_keras.optimizers.SGD
            except (ImportError, AttributeError):
                self.skipTest("tf_keras not installed")
        else:
            optimizer_fn = SGD
        with self.strategy.scope():
            optimizer = optimizer_fn(
                learning_rate=0.5,
                momentum=0.06,
            )
            # use tf variable to work both in k2 & k3.
            vars = tf.Variable([1.0, 2.0, 3.0, 4.0])

            def update():
                grads = tf.constant([1.0, 6.0, 7.0, 2.0])
                optimizer.apply_gradients(zip([grads], [vars]))

            self.strategy.run(update)
            self.assertAllClose(
                vars, [0.0, -4.0, -4.0, 2.0], rtol=1e-4, atol=1e-4
            )

    def test_weight_decay(self):
        with self.strategy.scope():
            grads, var1, var2, var3 = (
                tf.zeros(()),
                backend.Variable(2.0),
                backend.Variable(3.0, name="exclude"),
                backend.Variable(4.0),
            )
            optimizer_1 = SGD(learning_rate=1.0, weight_decay=0.004)
            self.strategy.run(
                lambda: optimizer_1.apply_gradients(zip([grads], [var1]))
            )

            # `exclude_from_weight_decay` must run before the optimizer is
            # built, so call it once, outside of `strategy.run`.
            optimizer_2 = SGD(learning_rate=1.0, weight_decay=0.004)
            optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
            self.strategy.run(
                lambda: optimizer_2.apply_gradients(
                    zip([grads, grads], [var1, var2])
                )
            )

            optimizer_3 = SGD(learning_rate=1.0, weight_decay=0.004)
            optimizer_3.exclude_from_weight_decay(var_list=[var3])
            self.strategy.run(
                lambda: optimizer_3.apply_gradients(
                    zip([grads, grads], [var1, var3])
                )
            )

        self.assertAlmostEqual(var1.numpy(), 1.9760959)
        self.assertAlmostEqual(var2.numpy(), 3.0)
        self.assertAlmostEqual(var3.numpy(), 4.0)

    def test_correctness_with_golden(self):
        with self.strategy.scope():
            optimizer = SGD(nesterov=True)
            x = backend.Variable(np.ones([10]))

            def update_grads():
                grads = backend.ops.convert_to_tensor(np.arange(0.1, 1.1, 0.1))
                optimizer.apply_gradients(zip([grads], [x]))

            def update_first_grads():
                first_grads = backend.ops.convert_to_tensor(
                    np.full((10,), 0.01)
                )
                optimizer.apply_gradients(zip([first_grads], [x]))

        # fmt: off
        golden = np.array(
            [
                [0.9980, 0.9960, 0.9940, 0.9920, 0.9900, 0.9880, 0.9860, 0.9840, 0.9820, 0.9800],
                [0.9978, 0.9958, 0.9938, 0.9918, 0.9898, 0.9878, 0.9858, 0.9838, 0.9818, 0.9798],
                [0.9976, 0.9956, 0.9936, 0.9916, 0.9896, 0.9876, 0.9856, 0.9836, 0.9816, 0.9796],
                [0.9974, 0.9954, 0.9934, 0.9914, 0.9894, 0.9874, 0.9854, 0.9834, 0.9814, 0.9794],
                [0.9972, 0.9952, 0.9932, 0.9912, 0.9892, 0.9872, 0.9852, 0.9832, 0.9812, 0.9792],
            ]
        )
        # fmt: on

        self.strategy.run(update_grads)
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            self.strategy.run(update_first_grads)

    def test_clip_norm(self):
        with self.strategy.scope():
            optimizer = SGD(clipnorm=1)
            grad = [np.array([100.0, 100.0])]
            clipped_grad = optimizer._clip_gradients(grad)
            self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        with self.strategy.scope():
            optimizer = SGD(clipvalue=1)
            grad = [np.array([100.0, 100.0])]
            clipped_grad = optimizer._clip_gradients(grad)
            self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    def test_stateless_not_supported(self):
        optimizer = SGD(learning_rate=0.5)
        grads = [np.array([1.0, 6.0, 7.0, 2.0])]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        with self.assertRaisesRegex(ValueError, "not supported"):
            optimizer.stateless_apply(optimizer.variables, grads, vars)

    def test_ema(self):
        with self.strategy.scope():
            v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
            grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
            optimizer = SGD(
                learning_rate=1.0,
                use_ema=True,
                ema_momentum=0.9,
                ema_overwrite_frequency=3,
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[2.0, 3.0], [4.0, 5.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[2.0, 3.0], [4.0, 5.0]],  # initialized after first step
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.9, 2.9], [3.9, 4.9]],
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            # Variables were overwritten with EMA
            self.assertAllClose(v, [[1.71, 2.71], [3.71, 4.71]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.71, 2.71], [3.71, 4.71]],
            )

    def test_ema_in_tf_function(self):
        with self.strategy.scope():
            v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
            grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
            optimizer = SGD(
                learning_rate=1.0,
                use_ema=True,
                ema_momentum=0.9,
                ema_overwrite_frequency=3,
            )

            @tf.function
            def distributed_update():
                self.strategy.run(
                    lambda: optimizer.apply_gradients([(grads, v)])
                )

            distributed_update()
            self.assertAllClose(v, [[2.0, 3.0], [4.0, 5.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[2.0, 3.0], [4.0, 5.0]],  # initialized after first step
            )
            distributed_update()
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.9, 2.9], [3.9, 4.9]],
            )
            distributed_update()
            # Variables were overwritten with EMA
            self.assertAllClose(v, [[1.71, 2.71], [3.71, 4.71]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.71, 2.71], [3.71, 4.71]],
            )

    def test_gradient_accumulation(self):
        with self.strategy.scope():
            v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
            grads = backend.ops.convert_to_tensor([[1.0, 1.0], [2.0, 2.0]])
            optimizer = SGD(learning_rate=1.0, gradient_accumulation_steps=3)
            self.assertEqual(optimizer.gradient_accumulation_steps, 3)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[1.0, 1.0], [2.0, 2.0]]
            )
            self.assertAllClose(optimizer._iterations, 1)
            self.assertAllClose(optimizer.iterations, 0)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[2.0, 2.0], [4.0, 4.0]]
            )
            self.assertAllClose(optimizer._iterations, 2)
            self.assertAllClose(optimizer.iterations, 0)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[-1.0, 0.0], [-1.0, 0.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[0.0, 0.0], [0.0, 0.0]]
            )
            self.assertAllClose(optimizer._iterations, 3)
            self.assertAllClose(optimizer.iterations, 1)
