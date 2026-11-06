import numpy as np
import pytest

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.layers import Dense
from keras.src.layers import Embedding
from keras.src.layers import Input
from keras.src.models import Sequential
from keras.src.optimizers.muon import Muon


class MuonTest(testing.TestCase):
    def test_config(self):
        optimizer = Muon(
            learning_rate=0.5,
            epsilon=1e-5,
        )
        self.run_class_serialization_test(optimizer)

    def test_Newton_Schulz(self):
        optimizer = Muon()
        tensor_input = ops.array([[0.2499, 0.9105], [0.2655, 0.8824]])
        except_output = ops.array([[-0.4422, 0.6457], [0.7285, 0.2968]])
        output = optimizer.zeropower_via_newtonschulz5(tensor_input, 5)
        self.assertAllClose(
            output,
            except_output,
            rtol=1e-3,
            atol=1e-3,
            tpu_atol=1e-1,
            tpu_rtol=1e-1,
        )

    def test_adamw_single_step(self):
        optimizer = Muon()
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0], name="test_vars")
        optimizer.build([vars])
        optimizer.update_step(grads, vars, 0.5)
        self.assertAllClose(vars, [0.5, 1.5, 2.5, 3.5], rtol=1e-4, atol=1e-4)

    def test_should_use_adamw(self):
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer = Muon(exclude_layers=["var"])
        self.assertTrue(optimizer._should_use_adamw(vars))
        embedding = Embedding(2, 2)
        embedding.build()
        self.assertTrue(optimizer._should_use_adamw(embedding.weights[0]))
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer = Muon()
        self.assertFalse(optimizer._should_use_adamw(vars))
        dense = Dense(2)
        dense.build([None, 2])
        self.assertFalse(optimizer._should_use_adamw(dense.weights[0]))

    def test_muon_single_step(self):
        optimizer = Muon(
            learning_rate=0.5,
            weight_decay=0,
        )
        grads = ops.array([[1.0, 6.0], [7.0, 2.0]])
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer.build([vars])
        optimizer.update_step(grads, vars, 0.5)
        self.assertAllClose(
            vars,
            [[0.988775, 1.887053], [2.873428, 3.97035]],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_clip_norm(self):
        optimizer = Muon(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Muon(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    def test_muon_weight_decay(self):
        variable = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        weight_decay = 0.01
        expected_variable = variable - variable * weight_decay
        optimizer = Muon(learning_rate=1.0, weight_decay=weight_decay)
        optimizer.build([variable])
        optimizer._apply_weight_decay([variable])
        self.assertAllClose(variable, expected_variable, rtol=1e-4, atol=1e-4)

    def test_adamw_weight_decay(self):
        variable = backend.Variable(2.0)
        weight_decay = 0.01
        expected_variable = variable - variable * weight_decay
        optimizer = Muon(learning_rate=1.0, adam_weight_decay=weight_decay)
        optimizer.build([variable])
        optimizer._apply_weight_decay([variable])

        self.assertAllClose(variable, expected_variable, rtol=1e-4, atol=1e-4)

    def test_lr_adjust_none(self):
        opt = Muon(rms_rate=None)
        x = ops.ones((4, 4))
        want = x
        self.assertAllClose(opt.lr_adjust(x), want)

    def test_lr_adjust_2d(self):
        opt = Muon(rms_rate=0.2)
        x = ops.ones((4, 2))
        want = x * 0.2 * 2
        self.assertAllClose(opt.lr_adjust(x), want)

    @pytest.mark.requires_trainable_backend
    def test_model_fit(self):
        model = Sequential([Input((10,)), Dense(5), Dense(1, name="last")])
        x = ops.ones((1, 10))
        y = ops.ones((1, 1))
        optimizer = Muon(learning_rate=1e-3, exclude_layers=["last"])
        model.compile(optimizer=optimizer, loss="mse")
        model.fit(x, y)
