import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.utils import summary_utils


class SummaryUtilsTest(testing.TestCase):
    @parameterized.parameters([("adam",), (None,)])
    @pytest.mark.requires_trainable_backend
    def test_print_model_summary(self, optimizer):
        inputs = layers.Input((2,))
        outputs = layers.Dense(3)(inputs)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
        if optimizer:
            # Trigger the optimizer weights creation
            model.fit(x=np.zeros([4, 2]), y=np.zeros([4, 3]))

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        try:
            summary_utils.print_summary(model, print_fn=print_to_variable)
            summary_content = "\n".join(summary_content)
            if optimizer:
                self.assertIn("Total params: 29", summary_content)
                self.assertIn("Trainable params: 9", summary_content)
                self.assertIn("Non-trainable params: 0", summary_content)
                self.assertIn("Optimizer params: 20", summary_content)
            else:
                self.assertIn("Total params: 9", summary_content)
                self.assertIn("Trainable params: 9", summary_content)
                self.assertIn("Non-trainable params: 0", summary_content)
                self.assertNotIn("Optimizer params", summary_content)
        except ImportError:
            pass

    def test_print_model_summary_custom_build(self):
        class MyModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(4, activation="relu")
                self.dense2 = layers.Dense(2, activation="softmax")
                self.unbuilt_dense = layers.Dense(1)

            def build(self, input_shape):
                self.dense1.build(input_shape)
                input_shape = self.dense1.compute_output_shape(input_shape)
                self.dense2.build(input_shape)

            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)

        model = MyModel()
        model.build((None, 2))

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        summary_utils.print_summary(model, print_fn=print_to_variable)
        summary_content = "\n".join(summary_content)
        self.assertIn("(None, 4)", summary_content)  # dense1
        self.assertIn("(None, 2)", summary_content)  # dense2
        self.assertIn("?", summary_content)  # unbuilt_dense
        self.assertIn("Total params: 22", summary_content)
        self.assertIn("Trainable params: 22", summary_content)
        self.assertIn("Non-trainable params: 0", summary_content)

    def test_print_model_summary_op_as_layer(self):
        inputs = layers.Input((2,))
        x = layers.Dense(4)(inputs)
        outputs = ops.mean(x)
        model = models.Model(inputs, outputs)

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        summary_utils.print_summary(
            model, print_fn=print_to_variable, show_trainable=True
        )
        summary_content = "\n".join(summary_content)
        self.assertIn("(None, 4)", summary_content)  # dense
        self.assertIn("Y", summary_content)  # dense
        self.assertIn("()", summary_content)  # mean
        self.assertIn("-", summary_content)  # mean
        self.assertIn("Total params: 12", summary_content)
        self.assertIn("Trainable params: 12", summary_content)
        self.assertIn("Non-trainable params: 0", summary_content)

    def test_print_model_summary_with_mha(self):
        # In Keras <= 3.6, MHA exposes `output_shape` property which breaks this
        # test.
        class MyModel(models.Model):
            def __init__(self):
                super().__init__()
                self.mha = layers.MultiHeadAttention(2, 2, output_shape=(4,))

            def call(self, inputs):
                return self.mha(inputs, inputs, inputs)

        model = MyModel()
        model(np.ones((1, 2, 2)))

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        summary_utils.print_summary(model, print_fn=print_to_variable)
        summary_content = "\n".join(summary_content)
        self.assertIn("(1, 2, 4)", summary_content)  # mha
        self.assertIn("Total params: 56", summary_content)
        self.assertIn("Trainable params: 56", summary_content)
        self.assertIn("Non-trainable params: 0", summary_content)

    def test_print_model_summary_dict_output(self):
        """model.summary() should not crash when a layer returns a dict."""

        class DictHead(layers.Layer):
            def __init__(self):
                super().__init__()
                self.head1 = layers.Dense(10)

            def call(self, x):
                return {"head1": self.head1(x)}

            def compute_output_shape(self, input_shape):
                return {"head1": self.head1.compute_output_shape(input_shape)}

        class MyModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(32)
                self.dict_head = DictHead()

            def call(self, x):
                return self.dict_head(self.dense(x))

        model = MyModel()
        model(np.zeros((1, 16)))

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        summary_utils.print_summary(model, print_fn=print_to_variable)
        summary_content = "\n".join(summary_content)
        self.assertIn("(1, 10)", summary_content)  # dict_head

    def test_print_model_summary_escapes_connected_layer_names(self):
        # Layer names can come from a loaded file; the "Connected to" column
        # must print them literally rather than parse them as Rich markup.
        inputs = layers.Input((2,), name="inputs")
        x = layers.Dense(3, name="[red]dense")(inputs)
        y = layers.Dense(3, name="other")(inputs)
        outputs = layers.Add()([x, y])
        model = models.Model(inputs, outputs)

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        summary_utils.print_summary(model, print_fn=print_to_variable)
        summary_content = "\n".join(summary_content)
        self.assertIn("[red]dense[0][0]", summary_content)
