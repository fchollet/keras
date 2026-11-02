import inspect
import os
import pickle

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.api_export import REGISTERED_NAMES_TO_OBJS


class OptimizerTest(testing.TestCase):
    def test_iterations_counter(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.Adam(learning_rate=1.0)
        self.assertAllClose(optimizer.iterations, 0)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(optimizer.iterations, 1)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(optimizer.iterations, 2)

    def test_empty_gradients(self):
        # Test no valid gradient
        v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
        grads = None
        optimizer = optimizers.SGD(learning_rate=1.0)
        with self.assertRaisesRegex(
            ValueError, "No gradients provided for any variable."
        ):
            optimizer.apply_gradients([(grads, v)])

        # Test filtering of empty gradients
        v2 = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
        grads2 = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(learning_rate=1.0)
        with self.assertWarns(Warning):
            optimizer.apply_gradients([(grads, v), (grads2, v2)])
        self.assertAllClose(v, [[3.0, 4.0], [5.0, 6.0]])
        self.assertAllClose(v2, [[2.0, 3.0], [4.0, 5.0]])

    def test_clip_args(self):
        optimizer = optimizers.SGD(learning_rate=1.0, clipnorm=0.1)
        self.assertEqual(optimizer.clipnorm, 0.1)
        optimizer = optimizers.SGD(learning_rate=1.0, clipvalue=0.1)
        self.assertEqual(optimizer.clipvalue, 0.1)
        optimizer = optimizers.SGD(learning_rate=1.0, global_clipnorm=0.1)
        self.assertEqual(optimizer.global_clipnorm, 0.1)

        # Test invalid arguments
        with self.assertRaisesRegex(
            ValueError,
            "Only one of `clipnorm`, `clipvalue` and `global_clipnorm` can",
        ):
            optimizers.SGD(
                learning_rate=1.0,
                clipnorm=0.1,
                clipvalue=0.1,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Only one of `clipnorm`, `clipvalue` and `global_clipnorm` can",
        ):
            optimizers.SGD(
                learning_rate=1.0,
                clipnorm=0.1,
                global_clipnorm=0.1,
            )

    def test_clip_norm(self):
        optimizer = optimizers.SGD(clipnorm=1)
        grad = backend.ops.convert_to_tensor([100.0, 100.0])
        clipped_grad = optimizer._clip_gradients([grad])
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = optimizers.SGD(clipvalue=1)
        grad = backend.ops.convert_to_tensor([100.0, 100.0])
        clipped_grad = optimizer._clip_gradients([grad])
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    def test_global_clip_norm(self):
        optimizer = optimizers.SGD(global_clipnorm=1)
        grad = np.array([50.0, 100.0], dtype="float32")
        global_norm = np.linalg.norm(grad)
        clipped_grad = optimizer._clip_gradients(
            [backend.ops.convert_to_tensor(grad)]
        )
        self.assertAllClose(clipped_grad[0], grad / global_norm)

    def test_ema(self):
        v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(
            learning_rate=1.0,
            use_ema=True,
            ema_momentum=0.9,
            ema_overwrite_frequency=3,
        )
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[2.0, 3.0], [4.0, 5.0]])
        self.assertAllClose(
            optimizer._model_variables_moving_average[0],
            [[2.0, 3.0], [4.0, 5.0]],  # initialized after first step
        )
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._model_variables_moving_average[0],
            [[1.9, 2.9], [3.9, 4.9]],
        )
        optimizer.apply_gradients([(grads, v)])
        # Variables were overwritten with EMA
        self.assertAllClose(v, [[1.71, 2.71], [3.71, 4.71]])
        self.assertAllClose(
            optimizer._model_variables_moving_average[0],
            [[1.71, 2.71], [3.71, 4.71]],
        )

    @pytest.mark.requires_trainable_backend
    def test_ema_with_model_fit(self):
        x_train = np.ones((1, 1)).astype("float32")
        y_train = np.zeros((1, 1)).astype("float32")
        optimizer = optimizers.SGD(
            learning_rate=0.1, use_ema=True, ema_momentum=0.9
        )
        model = models.Sequential(
            [layers.Dense(2, kernel_initializer="ones", use_bias=False)]
        )
        model.compile(loss="mse", optimizer=optimizer, run_eagerly=True)
        model.fit(x_train, y_train, batch_size=1, epochs=2)
        self.assertAllClose(
            optimizer._model_variables_moving_average[0].numpy(),
            [[0.891, 0.891]],
            atol=1e-5,
        )
        self.assertAllClose(
            model.trainable_variables[0].numpy(),
            [[0.891, 0.891]],
            atol=1e-5,
        )

    def test_constraints_are_applied(self):
        v = backend.Variable(np.random.random((2, 2)) - 1.0)
        v.constraint = constraints.NonNeg()
        optimizer = optimizers.SGD(learning_rate=0.0001)
        grad = backend.ops.numpy.zeros((2, 2))
        optimizer.apply_gradients([(grad, v)])
        self.assertAlmostEqual(np.min(v), 0.0)

    def test_get_method(self):
        obj = optimizers.get("sgd")
        self.assertIsInstance(obj, optimizers.SGD)
        obj = optimizers.get("adamw")
        self.assertIsInstance(obj, optimizers.AdamW)

        obj = optimizers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            optimizers.get("typo")

    def test_get_all_public_optimizers(self):
        not_objects = {"get", "serialize", "deserialize"}
        # Wrappers need an inner optimizer, so `get` can't build them by name.
        wrappers = {"LossScaleOptimizer", "MultiOptimizer", "OptimizerMap"}
        for name, obj in REGISTERED_NAMES_TO_OBJS.items():
            if (
                name.startswith("keras.optimizers.")
                and name.count(".") == 2
                and name.rsplit(".", 1)[1] not in not_objects | wrappers
            ):
                get_result = optimizers.get(name.rsplit(".", 1)[1])
                if inspect.isclass(obj):
                    self.assertIsInstance(get_result, obj)
                else:
                    self.assertIs(get_result, obj)

    def test_static_loss_scaling(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) * 1024.0
        optimizer = optimizers.SGD(learning_rate=1.0, loss_scale_factor=1024.0)
        optimizer.apply_gradients([(grads, v)])
        self.assertEqual(optimizer.scale_loss(1.0), 1024.0)
        self.assertAllClose(v, [[0.0, 0.0], [0.0, 0.0]])

    def test_set_weights(self):
        x = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer_1 = optimizers.Adam()
        grads = backend.ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        optimizer_1.apply_gradients(zip([grads], [x]))
        optimizer_2 = optimizers.Adam()
        with self.assertRaisesRegex(ValueError, "You are calling*"):
            optimizer_2.set_weights(optimizer_1.variables)
        optimizer_2.build([x])
        optimizer_2.set_weights(optimizer_1.variables)
        for i in range(len(optimizer_1.variables)):
            self.assertAllClose(
                optimizer_1.variables[i],
                optimizer_2.variables[i],
            )

    def test_gradient_accumulation(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(
            learning_rate=1.0, gradient_accumulation_steps=3
        )
        self.assertEqual(optimizer.gradient_accumulation_steps, 3)

        # Iteration 1
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(optimizer._iterations, 1)
        self.assertAllClose(optimizer.iterations, 0)

        # Iteration 2
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[2.0, 2.0], [2.0, 2.0]]
        )
        self.assertAllClose(optimizer._iterations, 2)
        self.assertAllClose(optimizer.iterations, 0)

        # Iteration 3
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[0.0, 1.0], [2.0, 3.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[0.0, 0.0], [0.0, 0.0]]
        )
        self.assertAllClose(optimizer._iterations, 3)
        self.assertAllClose(optimizer.iterations, 1)

        # Iteration 4
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[0.0, 1.0], [2.0, 3.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(optimizer._iterations, 4)
        self.assertAllClose(optimizer.iterations, 1)

    @pytest.mark.skipif(backend.backend() != "tensorflow", reason="Requires TF")
    def test_tf_checkpointing(self):
        import tensorflow as tf

        model = models.Sequential([layers.Dense(2)])
        optimizer = optimizers.Adam()
        x, y = np.random.random((1, 2)), np.random.random((1, 2))
        model.compile(optimizer, "mse")
        model.train_on_batch(x, y)
        ref_pred = model.predict(x)

        # Both model and optimizer are Trackables
        checkpoint = tf.train.Checkpoint(model, optimizer=optimizer)
        temp_filepath = os.path.join(self.get_temp_dir(), "tf_ckpt")
        save_path = checkpoint.save(temp_filepath)

        # Keep training the model (predictions now differ)
        model.train_on_batch(x, y)
        pred = model.predict(x)
        self.assertNotAllClose(pred, ref_pred, atol=1e-3)

        # Restore the model and check prediction correctness
        checkpoint.restore(save_path)
        pred = model.predict(x)
        self.assertAllClose(pred, ref_pred, atol=1e-5)

    def test_callable_learning_rate(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(learning_rate=lambda: 0.1)
        self.assertAllClose(optimizer.iterations, 0)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[0.9, 1.9], [2.9, 3.9]])
        self.assertAllClose(optimizer.iterations, 1)

    def test_overwrite_with_gradient(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        v.overwrite_with_gradient = True
        v2 = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        grads2 = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])

        optimizer = optimizers.SGD(learning_rate=1.0)
        optimizer.apply_gradients([(grads, v), (grads2, v2)])

        # `v` is overwritten by its gradient but `v2` is updated normally
        self.assertAllClose(v, [[1.0, 1.0], [1.0, 1.0]])
        self.assertAllClose(v2, [[0.0, 1.0], [2.0, 3.0]])

    def test_overwrite_with_gradient_with_gradient_accumulation(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        v.overwrite_with_gradient = True
        v2 = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grad_ones = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_twos = backend.ops.convert_to_tensor([[2.0, 2.0], [2.0, 2.0]])
        optimizer = optimizers.SGD(
            learning_rate=1.0, gradient_accumulation_steps=2
        )

        # Iteration 1
        optimizer.apply_gradients([(grad_ones, v), (grad_ones, v2)])
        self.assertAllClose(optimizer._iterations, 1)
        self.assertAllClose(optimizer.iterations, 0)
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(v2, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(
            optimizer._accumulated_gradients[1], [[1.0, 1.0], [1.0, 1.0]]
        )

        # Iteration 2
        optimizer.apply_gradients([(grad_twos, v), (grad_twos, v2)])
        self.assertAllClose(optimizer._iterations, 2)
        self.assertAllClose(optimizer.iterations, 1)
        self.assertAllClose(v, [[2.0, 2.0], [2.0, 2.0]])
        self.assertAllClose(v2, [[-0.5, 0.5], [1.5, 2.5]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[0.0, 0.0], [0.0, 0.0]]
        )
        self.assertAllClose(
            optimizer._accumulated_gradients[1], [[0.0, 0.0], [0.0, 0.0]]
        )

        # Iteration 3
        optimizer.apply_gradients([(grad_ones, v), (grad_ones, v2)])
        self.assertAllClose(optimizer._iterations, 3)
        self.assertAllClose(optimizer.iterations, 1)
        self.assertAllClose(v, [[2.0, 2.0], [2.0, 2.0]])
        self.assertAllClose(v2, [[-0.5, 0.5], [1.5, 2.5]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(
            optimizer._accumulated_gradients[1], [[1.0, 1.0], [1.0, 1.0]]
        )

    @parameterized.parameters(
        [
            ("adam",),
            ("sgd",),
            ("adamw",),
            ("adagrad",),
            ("rmsprop",),
            ("adadelta",),
            ("adamax",),
            ("lion",),
            ("nadam",),
            ("ftrl",),
            ("adafactor",),
        ]
    )
    def test_gradient_accumulation_with_weigth_decay(self, optimizer):
        optimizer1 = optimizers.get(
            {"class_name": optimizer, "config": {"weight_decay": 0.05}}
        )
        optimizer3 = optimizers.get(
            {
                "class_name": optimizer,
                "config": {
                    "weight_decay": 0.05,
                    "gradient_accumulation_steps": 3,
                },
            }
        )
        variable1 = backend.Variable([[0.9], [0.5]])
        variable3 = backend.Variable([[0.9], [0.5]])

        for epoch in range(8):
            grads3 = np.random.random([3, 2, 1]).astype("float32")

            grads1 = backend.ops.convert_to_tensor(grads3.mean(axis=0))
            optimizer1.apply_gradients([(grads1, variable1)])

            for batch in range(3):
                grads3_ = backend.ops.convert_to_tensor(grads3[batch])
                optimizer3.apply_gradients([(grads3_, variable3)])

        self.assertAllClose(variable1, variable3)

    def test_exclude_from_weight_decay_after_build_raises(self):
        optimizer = optimizers.AdamW(weight_decay=0.05)
        optimizer.build([backend.Variable([0.9, 0.5])])
        with self.assertRaisesRegex(ValueError, "before the optimizer is"):
            optimizer.exclude_from_weight_decay(var_names=["bias"])

    def test_setting_lr_to_callable_untracks_lr_var(self):
        adam = optimizers.Adam(learning_rate=0.001)
        self.assertLen(adam.variables, 2)
        adam.learning_rate = optimizers.schedules.PolynomialDecay(
            adam.learning_rate, 4
        )
        self.assertLen(adam.variables, 1)

    @parameterized.parameters(
        [
            ("adam",),
            ("sgd",),
            ("adamw",),
            ("adagrad",),
            ("rmsprop",),
            ("adadelta",),
            ("adamax",),
            ("lion",),
            ("nadam",),
            ("ftrl",),
            ("adafactor",),
        ]
    )
    def test_pickleable_optimizers(self, optimizer):
        optimizer = optimizers.get(optimizer)
        reloaded = pickle.loads(pickle.dumps(optimizer))

        self.assertEqual(optimizer.get_config(), reloaded.get_config())

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="The tf.Variable test can only run with TensorFlow backend.",
    )
    def test_mixed_with_tf_variables(self):
        import tensorflow as tf

        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        tf_v = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
        tf_grads = backend.ops.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.Adam(learning_rate=1.0)
        optimizer.apply_gradients([(grads, v), (tf_grads, tf_v)])
        self.assertAllClose(optimizer.iterations, 1)

        # Test with no grads
        with self.assertWarnsRegex(
            UserWarning, "Gradients do not exist for variables"
        ):
            optimizer.apply_gradients([(grads, v), (None, tf_v)])
            self.assertAllClose(optimizer.iterations, 2)

    def test_optimizer_auto_naming(self):
        adam_1 = optimizers.Adam()
        adam_2 = optimizers.Adam()
        sgd_1 = optimizers.SGD()
        sgd_2 = optimizers.SGD()
        adamw_1 = optimizers.AdamW()
        adamw_2 = optimizers.AdamW()

        self.assertEqual(adam_1.name, "adam")
        self.assertEqual(adam_2.name, "adam_1")
        self.assertEqual(sgd_1.name, "sgd")
        self.assertEqual(sgd_2.name, "sgd_1")
        self.assertEqual(adamw_1.name, "adam_w")
        self.assertEqual(adamw_2.name, "adam_w_1")

        # Test explicit naming
        custom_adam = optimizers.Adam(name="my_adam")
        self.assertEqual(custom_adam.name, "my_adam")

        # Test serialization and deserialization preserves name
        config_adam_2 = optimizers.serialize(adam_2)
        reloaded_adam_2 = optimizers.deserialize(config_adam_2)
        self.assertEqual(reloaded_adam_2.name, "adam_1")

        config_custom = optimizers.serialize(custom_adam)
        reloaded_custom = optimizers.deserialize(config_custom)
        self.assertEqual(reloaded_custom.name, "my_adam")
