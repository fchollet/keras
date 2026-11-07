import os
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import applications
from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.dtype_policies import dtype_policy
from keras.src.layers.core.input_layer import Input
from keras.src.layers.input_spec import InputSpec
from keras.src.models import Functional
from keras.src.models import Model
from keras.src.models import Sequential
from keras.src.models.model import model_from_json


class FunctionalTest(testing.TestCase):
    def test_basic_flow_multi_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs, name="basic")
        model.summary()

        self.assertEqual(model.name, "basic")
        self.assertIsInstance(model, Functional)
        self.assertIsInstance(model, Model)

        # Eager call
        in_val = [np.random.random((2, 3)), np.random.random((2, 3))]
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2, name="input_a_2")
        input_b_2 = Input(shape=(3,), batch_size=2, name="input_b_2")
        in_val = [input_a_2, input_b_2]
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    def test_scalar_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(), batch_size=2, name="input_b")
        outputs = input_a + input_b[:, None]
        model = Functional([input_a, input_b], outputs)
        model.summary()

        in_val = [np.zeros((2, 3)), np.ones((2,))]
        out_val = model(in_val)
        self.assertAllClose(out_val, np.ones((2, 3)))

    def test_mutable_state(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        outputs = layers.Dense(5)(x)
        model = Functional(inputs, outputs)
        # Allow attaching state to a model that isn't directly part of the DAG.
        # Most useful for functional subclasses.
        model.extra_layer = layers.Dense(5)

    def test_basic_flow_multi_output(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        output_a = layers.Dense(4)(x)
        output_b = layers.Dense(5)(x)
        model = Functional(inputs, [output_a, output_b])

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = model(in_val)
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

        # Symbolic call
        out_val = model(Input(shape=(3,), batch_size=2))
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

    def test_basic_flow_dict_io(self):
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)

        with self.assertRaisesRegex(
            ValueError, "All `inputs` values must be KerasTensors"
        ):
            model = Functional({"a": "input_a", "b": input_b}, outputs)

        with self.assertRaisesRegex(
            ValueError, "All `outputs` values must be KerasTensors"
        ):
            model = Functional({"a": input_a, "b": input_b}, "outputs")

        model = Functional({"a": input_a, "b": input_b}, outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3)), "b": np.random.random((2, 3))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        input_b_2 = Input(shape=(3,), batch_size=2)
        in_val = {"a": input_a_2, "b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    def test_basic_flow_as_a_submodel(self):
        # Build submodel
        submodel_inputs = Input([4])
        submodel_outputs = layers.Flatten()(submodel_inputs)
        submodel = Model(submodel_inputs, submodel_outputs)

        inputs = Input((None, 4))
        outputs = layers.TimeDistributed(submodel)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        x = np.random.random((2, 3, 4))
        y = model(x)
        self.assertEqual(y.shape, (2, 3, 4))

    def test_nested_model_propagates_symbolic_output_mask(self):
        class MaskCombiningLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, inputs, mask=None):
                x, y = inputs
                return x + y

            def compute_mask(self, inputs, previous_mask):
                x_mask, y_mask = previous_mask
                return ops.logical_and(x_mask, y_mask)

        embedding = layers.Embedding(3, 5, mask_zero=True)
        inputs = Input((3,))
        direct_output = embedding(inputs)
        embedding_model = Model(inputs, direct_output)

        nested_output = embedding_model(inputs)
        direct_mask = backend.get_keras_mask(direct_output)
        nested_mask = backend.get_keras_mask(nested_output)
        self.assertIsNotNone(nested_mask)

        combined = MaskCombiningLayer()((direct_output, nested_output))
        combined_mask = backend.get_keras_mask(combined)
        self.assertIsNotNone(combined_mask)

        mask_model = Model(inputs, [direct_mask, nested_mask, combined_mask])
        mask_values = mask_model(np.array([[0, 1, 2], [1, 0, 2]]))
        self.assertAllClose(mask_values[0], mask_values[1])
        self.assertAllClose(
            mask_values[2],
            np.array([[False, True, True], [True, False, True]]),
        )

    def test_compute_output_spec_does_not_create_duplicate_nodes(self):
        inputs = Input((3,), dtype="int32")
        embedding = layers.Embedding(3, 5, mask_zero=True)
        outputs = embedding(inputs)
        model = Model(inputs, outputs)

        initial_node_count = len(embedding._inbound_nodes)
        _ = model.compute_output_spec(Input((3,), dtype="int32"))

        self.assertEqual(len(embedding._inbound_nodes), initial_node_count)

    def test_compute_output_spec_propagates_caller_mask(self):
        inputs = Input((3, 5))
        outputs = layers.ReLU()(inputs)
        model = Model(inputs, outputs)
        self.assertIsNone(backend.get_keras_mask(outputs))

        input_mask = KerasTensor((None, 3), dtype="bool")
        test_input = Input((3, 5))
        backend.set_keras_mask(test_input, input_mask)

        output_spec = model.compute_output_spec(test_input)

        self.assertIs(backend.get_keras_mask(output_spec), input_mask)

    def test_named_input_dict_io(self):
        # Single input
        input_a = Input(shape=(3,), batch_size=2, name="a")
        x = layers.Dense(5)(input_a)
        outputs = layers.Dense(4)(x)
        model = Functional(input_a, outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        in_val = {"a": input_a_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # ----
        # Two inputs, input is list
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(4,), batch_size=2, name="b")
        a = layers.Dense(5)(input_a)
        b = layers.Dense(5)(input_b)
        x = layers.Concatenate()([a, b])
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3)), "b": np.random.random((2, 4))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        input_b_2 = Input(shape=(4,), batch_size=2)
        in_val = {"a": input_a_2, "b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # ----
        # Two inputs, input is dict
        model = Functional({"a": input_a, "b": input_b}, outputs)

        # Eager call
        in_val = {"a": np.random.random((2, 3)), "b": np.random.random((2, 4))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        input_b_2 = Input(shape=(4,), batch_size=2)
        in_val = {"a": input_a_2, "b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # ----
        # Two inputs, input is dict with incorrect names
        model = Functional({"c": input_a, "d": input_b}, outputs)

        # Eager call
        in_val = {"c": np.random.random((2, 3)), "d": np.random.random((2, 4))}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2)
        input_b_2 = Input(shape=(4,), batch_size=2)
        in_val = {"c": input_a_2, "d": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Now we can't use the input names:
        with self.assertRaises(ValueError):
            in_val = {
                "a": np.random.random((2, 3)),
                "b": np.random.random((2, 4)),
            }
            out_val = model(in_val)

    def test_input_dict_with_extra_field(self):
        input_a = Input(shape=(3,), batch_size=2, name="a")
        x = input_a * 5
        outputs = x + 2

        model = Functional({"a": input_a}, outputs)

        with pytest.warns() as record:
            # Eager call
            in_val = {
                "a": np.random.random((2, 3)),
                "b": np.random.random((2, 1)),
            }
            out_val = model(in_val)
            self.assertEqual(out_val.shape, (2, 3))

            # Symbolic call
            input_a_2 = Input(shape=(3,), batch_size=2)
            input_b_2 = Input(shape=(1,), batch_size=2)
            in_val = {"a": input_a_2, "b": input_b_2}
            out_val = model(in_val)
            self.assertEqual(out_val.shape, (2, 3))
        self.assertLen(record, 1)
        self.assertStartsWith(
            str(record[0].message),
            r"The structure of `inputs` doesn't match the expected structure",
        )

    @parameterized.named_parameters(
        ("list", list),
        ("tuple", tuple),
        ("dict", dict),
    )
    def test_restored_multi_output_type(self, out_type):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        x = layers.Dense(5)(inputs)
        output_a = layers.Dense(4)(x)
        output_b = layers.Dense(5)(x)
        if out_type is dict:
            outputs = {"a": output_a, "b": output_b}
        else:
            outputs = out_type([output_a, output_b])
        model = Functional(inputs, outputs)
        model_restored = Functional.from_config(model.get_config())

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = model_restored(in_val)
        self.assertIsInstance(out_val, out_type)

        # Symbolic call
        out_val = model_restored(Input(shape=(3,), batch_size=2))
        self.assertIsInstance(out_val, out_type)

    def test_restored_nested_input(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        x = layers.Dense(5)(input_a)
        outputs = layers.Dense(4)(x)
        model = Functional([[input_a]], outputs)

        # Serialize and deserialize the model
        json_config = model.to_json()
        restored_json_config = model_from_json(json_config).to_json()

        # Check that the serialized model is the same as the original
        self.assertEqual(json_config, restored_json_config)

    def test_functional_input_shape_and_type(self):
        input = layers.Input((1024, 4))
        conv = layers.Conv1D(32, 3)(input)
        model = Functional(input, conv)

        self.assertIsInstance(model.input, KerasTensor)
        self.assertEqual(model.input_shape, (None, 1024, 4))

    def test_layer_getters(self):
        # Test mixing ops and layers
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5, name="dense_1")(x)
        outputs = layers.Dense(4, name="dense_2")(x)
        model = Functional([input_a, input_b], outputs)

        self.assertEqual(len(model.layers), 4)
        self.assertEqual(len(model._operations), 5)
        self.assertEqual(model.get_layer(index=0).name, "input_a")
        self.assertEqual(model.get_layer(index=1).name, "input_b")
        self.assertEqual(model.get_layer(index=2).name, "dense_1")
        self.assertEqual(model.get_layer(index=3).name, "dense_2")
        self.assertEqual(model.get_layer(name="dense_1").name, "dense_1")

    def test_training_arg(self):
        class CustomLayer(layers.Layer):
            def call(self, inputs, training=False):
                return inputs * 0.0 if training else inputs

        inputs = layers.Input((20,))
        outputs = CustomLayer()(inputs)
        model = Functional(inputs=inputs, outputs=outputs)
        self.assertAllClose(model(np.ones((4, 20))), np.ones((4, 20)))
        self.assertAllClose(
            model(np.ones((4, 20)), training=False), np.ones((4, 20))
        )
        self.assertAllClose(
            model(np.ones((4, 20)), training=True), np.zeros((4, 20))
        )

        inputs = layers.Input((20,))
        # This should hardcode `training=True` even if we pass
        # `training=False` to the model.
        outputs = CustomLayer()(inputs, training=True)
        model = Functional(inputs=inputs, outputs=outputs)
        self.assertAllClose(model(np.ones((4, 20))), np.zeros((4, 20)))
        self.assertAllClose(
            model(np.ones((4, 20)), training=False), np.zeros((4, 20))
        )
        self.assertAllClose(
            model(np.ones((4, 20)), training=True), np.zeros((4, 20))
        )

    def test_mask_arg(self):
        # TODO
        pass

    def test_passing_inputs_by_name(self):
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional([input_a, input_b], outputs)

        # Eager call
        in_val = {
            "input_a": np.random.random((2, 3)),
            "input_b": np.random.random((2, 3)),
        }
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        input_a_2 = Input(shape=(3,), batch_size=2, name="input_a_2")
        input_b_2 = Input(shape=(3,), batch_size=2, name="input_b_2")
        in_val = {"input_a": input_a_2, "input_b": input_b_2}
        out_val = model(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    def test_rank_standardization(self):
        # Downranking
        inputs = Input(shape=(3,), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs)
        out_val = model(np.random.random((2, 3, 1)))
        self.assertEqual(out_val.shape, (2, 3))

        # Upranking
        inputs = Input(shape=(3, 1), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs)
        out_val = model(np.random.random((2, 3)))
        self.assertEqual(out_val.shape, (2, 3, 3))

    def test_rank_standardization_failure(self):
        # Simple input and rank too high
        inputs = Input(shape=(3,), name="foo")
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs)
        with self.assertRaisesRegex(ValueError, "name 'foo' .* path ''"):
            model(np.random.random((2, 3, 4)))

        # Deeply nested input and rank too low
        inputs = [{"foo": Input(shape=(3,), name="my_input")}]
        outputs = layers.Dense(3)(inputs[0]["foo"])
        model = Functional(inputs, outputs)
        with self.assertRaisesRegex(
            ValueError, "name 'my_input' .* path '0.foo'"
        ):
            model(np.random.random(()))

    def test_last_axis_must_match_at_same_rank(self):
        # A last axis of size 1 only bridges a rank difference of one. At the
        # same rank it is compared like any other axis.
        inputs = Input(shape=(28, 28, 1))
        model = Functional(inputs, layers.Identity()(inputs))
        with self.assertRaisesRegex(
            ValueError,
            r"expected shape=\(None, 28, 28, 1\), "
            r"found shape=\(4, 28, 28, 3\)",
        ):
            model(np.zeros((4, 28, 28, 3)))

        inputs = Input(shape=(3,))
        model = Functional(inputs, layers.Identity()(inputs))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 3\), found shape=\(2, 1\)"
        ):
            model(np.zeros((2, 1)))

    def test_rank_standardization_checks_the_other_axes(self):
        # A last axis of size 1 bridges a rank difference of one only when
        # the other axes match.
        inputs = Input(shape=(3, 1))
        model = Functional(inputs, layers.Identity()(inputs))
        # Downranking
        with self.assertRaisesRegex(
            ValueError,
            r"expected shape=\(None, 3, 1\), found shape=\(2, 3, 5, 1\)",
        ):
            model(np.zeros((2, 3, 5, 1)))
        # Upranking
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 3, 1\), found shape=\(2, 1\)"
        ):
            model(np.zeros((2, 1)))

    def test_dtype_standardization(self):
        float_input = Input(shape=(2,), dtype="float16")
        int_input = Input(shape=(2,), dtype="int32")
        float_output = float_input + 2
        int_output = int_input + 2
        model = Functional((float_input, int_input), (float_output, int_output))
        float_data, int_data = model((np.ones((2, 2)), np.ones((2, 2))))

        self.assertEqual(standardize_dtype(float_data.dtype), "float16")
        self.assertEqual(standardize_dtype(int_data.dtype), "int32")

    def test_serialization(self):
        # Test basic model
        inputs = Input(shape=(3,), batch_size=2)
        outputs = layers.Dense(3)(inputs)
        model = Functional(inputs, outputs, trainable=False)
        self.run_class_serialization_test(model)

        # Test multi-io model
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        xa = layers.Dense(5, name="middle_a")(input_a)
        xb = layers.Dense(5, name="middle_b")(input_b)
        output_a = layers.Dense(4, name="output_a")(xa)
        output_b = layers.Dense(4, name="output_b")(xb)
        model = Functional(
            [input_a, input_b], [output_a, output_b], name="func"
        )
        self.run_class_serialization_test(model)

        # Test model that includes floating ops
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        x = input_a + input_b
        x = layers.Dense(5, name="middle")(x)
        output_a = layers.Dense(4, name="output_a")(x)
        output_b = layers.Dense(4, name="output_b")(x)
        model = Functional(
            [input_a, input_b], [output_a, output_b], name="func"
        )
        self.run_class_serialization_test(model)

        # Test model with dict i/o
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        x = input_a + input_b
        x = layers.Dense(5)(x)
        outputs = layers.Dense(4)(x)
        model = Functional({"a": input_a, "b": input_b}, outputs)
        self.run_class_serialization_test(model)

        # Test model with unmodified input as output
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        output_a = input_a * 2
        output_b = input_b
        model = Functional(
            {"a": input_a, "b": input_b}, {"a": output_a, "b": output_b}
        )
        self.run_class_serialization_test(model)

        # Test model with unused input
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        output_a = input_a * 2
        model = Functional({"a": input_a, "b": input_b}, output_a)
        self.run_class_serialization_test(model)

    def test_bad_input_spec(self):
        # Single input
        inputs = Input(shape=(4,))
        outputs = layers.Dense(2)(inputs)
        model = Functional(inputs, outputs)
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, r"expects 1 .*input"):
            model([np.zeros((2, 4)), np.zeros((2, 4))])

        # List input
        input_a = Input(shape=(4,), name="a")
        input_b = Input(shape=(4,), name="b")
        x = input_a + input_b
        outputs = layers.Dense(2)(x)
        model = Functional([input_a, input_b], outputs)
        with self.assertRaisesRegex(ValueError, r"expects 2 .*input"):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model([np.zeros((2, 3)), np.zeros((2, 4))])

        # Dict input
        model = Functional({"a": input_a, "b": input_b}, outputs)
        with self.assertRaisesRegex(ValueError, r"named input\(s\) with keys"):
            model(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            model({"a": np.zeros((2, 3)), "b": np.zeros((2, 4))})

    def test_manual_input_spec(self):
        inputs = Input(shape=(None, 3))
        outputs = layers.Dense(2)(inputs)
        model = Functional(inputs, outputs)
        model.input_spec = InputSpec(shape=(None, 4, 3))
        with self.assertRaisesRegex(
            ValueError,
            r"expected shape=\(None, 4, 3\), found shape=\(2, 3, 3\)",
        ):
            model(np.zeros((2, 3, 3)))
        model(np.zeros((2, 4, 3)))

    def test_input_spec_skipped_for_nested_inputs(self):
        a = Input(shape=(3,), name="a")
        b = Input(shape=(3,), name="b")
        c = Input(shape=(3,), name="c")
        outputs = layers.Add()([layers.Add()([a, b]), c])

        # Flat specs describe a flat sequence of inputs, so they are built
        # for a flat list and skipped for anything nested inside it.
        self.assertLen(Functional([a, b, c], outputs).input_spec, 3)
        self.assertIsNone(Functional([[a, b], c], outputs).input_spec)
        self.assertIsNone(Functional([{"a": a, "b": b}, c], outputs).input_spec)

        # A single input is not a sequence but is still describable.
        inputs = Input(shape=(3,), name="solo")
        model = Functional(inputs, layers.Dense(2)(inputs))
        self.assertLen(model.input_spec, 1)

    def test_functional_slicing(self):
        inputs = Input(shape=(None, 2), name="input")
        x1 = layers.Dense(3, name="dense1")(inputs)
        x2 = layers.Dense(4, name="dense2")(x1)
        outputs = layers.Dense(5, name="dense3")(x2)

        full_model = Functional(inputs, outputs, name="full_model")
        self.assertLen(full_model.layers, 4)

        partial_model_1 = Functional(x2, outputs, name="partial1")
        self.assertLen(partial_model_1.layers, 2)  # input_layer, dense3
        self.assertIsInstance(partial_model_1.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_1.layers[1].name, "dense3")

        partial_model_2 = Functional(x1, x2, name="partial2")
        self.assertLen(partial_model_2.layers, 2)  # input_layer, dense2
        self.assertIsInstance(partial_model_2.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_2.layers[1].name, "dense2")

        partial_model_3 = Functional(
            full_model.get_layer("dense2").input, outputs, name="partial3"
        )
        self.assertLen(partial_model_3.layers, 3)  # input_layer, dense2, dense3
        self.assertIsInstance(partial_model_3.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_3.layers[1].name, "dense2")
        self.assertEqual(partial_model_3.layers[2].name, "dense3")

        partial_model_4 = Functional(
            full_model.get_layer("dense1").input,
            full_model.get_layer("dense2").output,
            name="partial4",
        )
        self.assertLen(partial_model_4.layers, 3)  # input_layer, dense1, dense2
        self.assertIsInstance(partial_model_4.layers[0], layers.InputLayer)
        self.assertEqual(partial_model_4.layers[1].name, "dense1")
        self.assertEqual(partial_model_4.layers[2].name, "dense2")

    def test_deeply_nested_model(self):
        i1, i2, i3 = Input((1,)), Input((2,)), Input((3,))
        o1, o2, o3 = (
            layers.Dense(1)(i1),
            layers.Dense(2)(i2),
            layers.Dense(3)(i3),
        )
        model = Model(
            {"1": i1, "others": {"2": i2, "3": i3}},
            {"1": o1, "others": {"2": o2, "3": o3}},
        )
        out_eager = model(
            {
                "1": np.ones((2, 1)),
                "others": {"2": np.ones((2, 2)), "3": np.ones((2, 3))},
            }
        )
        out_symbolic = model(
            {
                "1": Input((1,), batch_size=2),
                "others": {
                    "2": Input((2,), batch_size=2),
                    "3": Input((3,), batch_size=2),
                },
            }
        )
        for out in [out_eager, out_symbolic]:
            self.assertIsInstance(out, dict)
            self.assertEqual(set(out.keys()), {"1", "others"})
            self.assertEqual(out["1"].shape, (2, 1))
            self.assertIsInstance(out["others"], dict)
            self.assertEqual(set(out["others"].keys()), {"2", "3"})
            self.assertEqual(out["others"]["2"].shape, (2, 2))
            self.assertEqual(out["others"]["3"].shape, (2, 3))

        # Test serialization boundaries
        temp_filepath = os.path.join(self.get_temp_dir(), "deeply_nested.keras")
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)
        new_out_eager = loaded_model(
            {
                "1": np.ones((2, 1)),
                "others": {"2": np.ones((2, 2)), "3": np.ones((2, 3))},
            }
        )
        self.assertAllClose(out_eager["1"], new_out_eager["1"])
        self.assertAllClose(
            out_eager["others"]["2"], new_out_eager["others"]["2"]
        )
        self.assertAllClose(
            out_eager["others"]["3"], new_out_eager["others"]["3"]
        )

    def test_optional_inputs(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        i1 = Input((2,))
        i2 = Input((2,), optional=True)
        outputs = OptionalInputLayer()(i1, i2)
        model = Model([i1, i2], outputs)

        # Eager test
        out = model([np.ones((2, 2)), None])
        self.assertAllClose(out, np.ones((2, 2)))
        # Note: it's not intended to work in symbolic mode (yet).

    def test_optional_dict_inputs(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2", optional=True)
        outputs = OptionalInputLayer()(i1, i2)
        model = Model({"input1": i1, "input2": i2}, outputs)

        # Eager test
        out = model({"input1": np.ones((2, 2)), "input2": None})
        self.assertAllClose(out, np.ones((2, 2)))
        # Note: it's not intended to work in symbolic mode (yet).

    def test_unmodified_inputs(self):
        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2")
        o1 = i1 * 2
        o2 = i2
        model = Model(
            {"input1": i1, "input2": i2}, {"output1": o1, "output2": o2}
        )

        # Eager call
        out = model({"input1": np.ones((2, 2)), "input2": np.zeros((2, 2))})
        self.assertAllClose(out["output1"], np.ones((2, 2)) * 2)
        self.assertAllClose(out["output2"], np.zeros((2, 2)))

        # Symbolic call
        i1_symbolic = Input((2,))
        i2_symbolic = Input((2,))
        out_symbolic = model({"input1": i1_symbolic, "input2": i2_symbolic})
        self.assertIsInstance(out_symbolic, dict)
        self.assertEqual(out_symbolic["output1"].shape, (None, 2))
        self.assertEqual(out_symbolic["output2"].shape, (None, 2))

    def test_unused_inputs(self):
        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2")
        o1 = i1 * 2
        model = Model({"input1": i1, "input2": i2}, o1)

        # Eager call
        out = model({"input1": np.ones((2, 2)), "input2": np.zeros((2, 2))})
        self.assertAllClose(out, np.ones((2, 2)) * 2)

        # Symbolic call
        i1_symbolic = Input((2,))
        i2_symbolic = Input((2,))
        out_symbolic = model({"input1": i1_symbolic, "input2": i2_symbolic})
        self.assertEqual(out_symbolic.shape, (None, 2))

    def test_disconnected_output(self):
        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2")
        o1 = i1 * 2
        o2 = i2 * 3
        with self.assertRaisesRegex(
            ValueError, "Output with path `output2` is not connected"
        ):
            Model(i1, {"output1": o1, "output2": o2})

    def test_required_input_none_raises(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2", optional=False)
        outputs = OptionalInputLayer()(i1, i2)
        model = Model({"input1": i1, "input2": i2}, outputs)

        # Passing None for a required input must raise
        with self.assertRaisesRegex(ValueError, "not optional"):
            model({"input1": np.ones((2, 2)), "input2": None})

    def test_optional_and_required_mixed_structures(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        i1 = Input((2,), name="input1")
        i2 = Input((2,), name="input2", optional=True)
        outputs = OptionalInputLayer()(i1, i2)
        model = Model({"input1": i1, "input2": i2}, outputs)

        # List structure
        out_list = model([np.ones((2, 2)), None])
        self.assertAllClose(out_list, np.ones((2, 2)))

        # Dict structure
        out_dict = model({"input1": np.ones((2, 2)), "input2": None})
        self.assertAllClose(out_dict, np.ones((2, 2)))

    @pytest.mark.requires_trainable_backend
    def test_required_none_raises_in_fit(self):
        # Small trainable model where input is required. fit should error
        x = layers.Input((3,), name="x")
        y = layers.Dense(1)(x)
        model = Model(x, y)
        model.compile(optimizer="sgd", loss="mse")

        y_data = np.ones((4, 1))

        # Passing None for required input should raise during fit
        with self.assertRaisesRegex(ValueError, "not optional"):
            model.fit([None], y_data, epochs=1)

    def test_warning_for_mismatched_inputs_structure(self):
        def is_input_warning(w):
            return str(w.message).startswith(
                "The structure of `inputs` doesn't match the expected structure"
            )

        i1 = Input((2,))
        i2 = Input((2,))
        outputs = layers.Add()([i1, i2])

        model = Model({"i1": i1, "i2": i2}, outputs)
        with pytest.warns() as warning_logs:
            model.predict([np.ones((2, 2)), np.zeros((2, 2))], verbose=0)
            self.assertLen(list(filter(is_input_warning, warning_logs)), 1)
        # No warning for mismatched tuples and lists.
        model = Model([i1, i2], outputs)
        with warnings.catch_warnings(record=True) as warning_logs:
            model.predict((np.ones((2, 2)), np.zeros((2, 2))), verbose=0)
            self.assertLen(list(filter(is_input_warning, warning_logs)), 0)

    def test_for_functional_in_sequential(self):
        # Test for a v3.4.1 regression.
        if backend.image_data_format() == "channels_first":
            image_size = (3, 100, 100)
        else:
            image_size = (100, 100, 3)
        base_model = applications.mobilenet.MobileNet(
            include_top=False, weights=None
        )
        model = Sequential()
        model.add(layers.Input(shape=image_size))
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(7, activation="softmax"))
        config = model.get_config()
        model = Sequential.from_config(config)

    def test_add_loss(self):
        # TODO
        pass

    def test_layers_setter(self):
        inputs = Input(shape=(3,), batch_size=2, name="input")
        outputs = layers.Dense(5)(inputs)
        model = Functional(inputs, outputs)
        with self.assertRaisesRegex(
            AttributeError, "`Model.layers` attribute is reserved"
        ):
            model.layers = [layers.Dense(4)]

    @pytest.mark.requires_trainable_backend
    def test_dict_input_to_list_model(self):
        vocabulary_size = 100
        num_tags = 10
        num_departments = 3
        num_samples = 128

        title = layers.Input(shape=(vocabulary_size,), name="title")
        text_body = layers.Input(shape=(vocabulary_size,), name="text_body")
        tags = layers.Input(shape=(num_tags,), name="tags")
        features = layers.Concatenate()([title, text_body, tags])
        features = layers.Dense(64, activation="relu")(features)
        priority = layers.Dense(1, activation="sigmoid", name="priority")(
            features
        )
        department = layers.Dense(
            num_departments, activation="softmax", name="department"
        )(features)
        model = Functional(
            inputs=[title, text_body, tags], outputs=[priority, department]
        )

        title_data = np.random.randint(
            0, 2, size=(num_samples, vocabulary_size)
        )
        text_body_data = np.random.randint(
            0, 2, size=(num_samples, vocabulary_size)
        )
        tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
        priority_data = np.random.random(size=(num_samples, 1))
        department_data = np.random.randint(
            0, 2, size=(num_samples, num_departments)
        )

        # List style fit
        model.compile(
            optimizer="adam",
            loss=["mean_squared_error", "categorical_crossentropy"],
            metrics=[["mean_absolute_error"], ["accuracy"]],
        )
        model.fit(
            [title_data, text_body_data, tags_data],
            [priority_data, department_data],
            epochs=1,
        )
        model.evaluate(
            [title_data, text_body_data, tags_data],
            [priority_data, department_data],
        )
        priority_preds, department_preds = model.predict(
            [title_data, text_body_data, tags_data]
        )

        # Dict style fit
        model.compile(
            optimizer="adam",
            loss={
                "priority": "mean_squared_error",
                "department": "categorical_crossentropy",
            },
            metrics={
                "priority": ["mean_absolute_error"],
                "department": ["accuracy"],
            },
        )
        model.fit(
            {
                "title": title_data,
                "text_body": text_body_data,
                "tags": tags_data,
            },
            {"priority": priority_data, "department": department_data},
            epochs=1,
        )
        model.evaluate(
            {
                "title": title_data,
                "text_body": text_body_data,
                "tags": tags_data,
            },
            {"priority": priority_data, "department": department_data},
        )
        priority_preds, department_preds = model.predict(
            {
                "title": title_data,
                "text_body": text_body_data,
                "tags": tags_data,
            }
        )

    def test_list_input_with_dict_build(self):
        x1 = Input((10,), name="IT")
        x2 = Input((10,), name="IS")
        y = layers.subtract([x1, x2])
        model = Model(inputs={"IT": x1, "IS": x2}, outputs=y)
        x1 = ops.ones((1, 10))
        x2 = ops.zeros((1, 10))
        # Works
        _ = model({"IT": x1, "IS": x2})
        with self.assertRaisesRegex(
            ValueError,
            "The structure of `inputs` doesn't match the expected structure",
        ):
            model([x1, x2])

    def test_functional_with_dtype_policy(self):
        original_dtype_policy = dtype_policy.dtype_policy()
        try:
            dtype_policy.set_dtype_policy("mixed_float16")

            inputs = Input((10,), name="input")
            outputs = layers.Dense(5)(inputs)
            model = Model(inputs=inputs, outputs=outputs)

            # Verify that no cast node appears in the graph.
            self.assertLen(model.operations, 2)
            self.assertIsInstance(model.operations[0], layers.InputLayer)
            self.assertIsInstance(model.operations[1], layers.Dense)
        finally:
            dtype_policy.set_dtype_policy(original_dtype_policy)

    def test_invalid_config_deserialization_loop(self):
        config = {
            "name": "infinite_loop_model",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "name": "input_1",
                    "config": {
                        "batch_input_shape": [None, 10],
                        "dtype": "float32",
                        "name": "input_1",
                    },
                    "inbound_nodes": [],
                },
                {
                    "class_name": "Dense",
                    "name": "node_a",
                    "config": {"units": 10, "name": "node_a"},
                    "inbound_nodes": [[["node_b", 0, 0, {}]]],
                },
                {
                    "class_name": "Dense",
                    "name": "node_b",
                    "config": {"units": 10, "name": "node_b"},
                    "inbound_nodes": [[["node_a", 0, 0, {}]]],
                },
            ],
            "input_layers": [["input_1", 0, 0]],
            "output_layers": [["node_b", 0, 0]],
        }
        with self.assertRaisesRegex(ValueError, "loop"):
            Functional.from_config(config)

    def test_invalid_config_deserialization_missing_node(self):
        config = {
            "name": "infinite_loop_model",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "name": "input_1",
                    "config": {
                        "batch_input_shape": [None, 10],
                        "dtype": "float32",
                        "name": "input_1",
                    },
                    "inbound_nodes": [],
                },
                {
                    "class_name": "Dense",
                    "name": "node_a",
                    "config": {"units": 10, "name": "node_a"},
                    "inbound_nodes": [[["node_b", 0, 0, {}]]],
                },
            ],
            "input_layers": [["input_1", 0, 0]],
            "output_layers": [["node_a", 0, 0]],
        }
        with self.assertRaisesRegex(ValueError, "Missing node: node_b"):
            Functional.from_config(config)
