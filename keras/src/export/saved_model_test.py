"""Tests for SavedModel exporting utilities."""

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import random
from keras.src import testing
from keras.src import tree
from keras.src.export import saved_model
from keras.src.export import xla_saved_model_archive
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        return models.Sequential(layer_list)
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        return CustomModel(layer_list)


@pytest.mark.skipif(
    backend.backend() not in ("tensorflow", "jax", "torch"),
    reason=(
        "`export_saved_model` only currently supports the tensorflow, jax and "
        "torch backends."
    ),
)
@pytest.mark.skipif(
    testing.torch_uses_gpu(), reason="Leads to core dumps on CI"
)
@pytest.mark.skipif(
    backend.backend() == "torch" and np.version.version.startswith("2."),
    reason="Torch backend export (via torch_xla) is incompatible with np 2.0",
)
class ExportSavedModelTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.export_kwargs = {}
        if testing.jax_uses_gpu():
            self.export_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "cuda")
                }
            }
        elif testing.jax_uses_tpu():
            self.export_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "tpu")
                }
            }

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input)

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason=(
            "RuntimeError: mutating a non-functional tensor with a "
            "functional tensor is not allowed in the torch backend."
        ),
    )
    def test_model_with_rng_export(self, model_type):
        class RandomLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_generator = backend.random.SeedGenerator()

            def call(self, inputs):
                return inputs + random.uniform(
                    ops.shape(inputs), seed=self.seed_generator
                )

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[RandomLayer()])
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertEqual(ref_output.shape, revived_model.serve(ref_input).shape)
        # Test with a different batch size
        input = tf.random.normal((6, 10))
        output1 = revived_model.serve(input)
        output2 = revived_model.serve(input)
        # Verify RNG seeding works and produces random outputs
        self.assertNotAllClose(output1, output2)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason=(
            "RuntimeError: mutating a non-functional tensor with a "
            "functional tensor is not allowed in the torch backend."
        ),
    )
    def test_model_with_non_trainable_state_export(self, model_type):
        class StateLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.counter = self.add_variable(
                    (), "zeros", "int32", trainable=False
                )

            def call(self, inputs):
                self.counter.assign_add(1)
                return ops.array(inputs), ops.array(self.counter.value)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[StateLayer()])
        model(tf.random.normal((3, 10)))

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)

        # The non-trainable counter is expected to increment
        input = tf.random.normal((6, 10))
        output1, counter1 = revived_model.serve(input)
        self.assertAllClose(output1, input, tpu_atol=0.01, tpu_rtol=0.01)
        self.assertAllClose(counter1, 2, tpu_atol=0.01, tpu_rtol=0.01)
        output2, counter2 = revived_model.serve(input)
        self.assertAllClose(output2, input, tpu_atol=0.01, tpu_rtol=0.01)
        self.assertAllClose(counter2, 3, tpu_atol=0.01, tpu_rtol=0.01)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_with_tf_data_layer(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type, layer_list=[layers.Rescaling(scale=2.0)])
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input)

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        class TupleModel(models.Model):
            def call(self, inputs):
                x, y = inputs
                return ops.add(x, y)

        class ArrayModel(models.Model):
            def call(self, inputs):
                x = inputs[0]
                y = inputs[1]
                return ops.add(x, y)

        class DictModel(models.Model):
            def call(self, inputs):
                x = inputs["x"]
                y = inputs["y"]
                return ops.add(x, y)

        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        if struct_type == "tuple":
            model = TupleModel()
            ref_input = (ref_input, ref_input * 2)
        elif struct_type == "array":
            model = ArrayModel()
            ref_input = [ref_input, ref_input * 2]
        elif struct_type == "dict":
            model = DictModel()
            ref_input = {"x": ref_input, "y": ref_input * 2}

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        ref_output = model(tree.map_structure(ops.convert_to_tensor, ref_input))

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )

        # Test with keras.saving_lib
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.keras"
        )
        saving_lib.save_model(model, temp_filepath)
        revived_model = saving_lib.load_model(
            temp_filepath,
            {
                "TupleModel": TupleModel,
                "ArrayModel": ArrayModel,
                "DictModel": DictModel,
            },
        )
        self.assertAllClose(
            revived_model(ref_input), ref_output, tpu_atol=0.01, tpu_rtol=0.01
        )
        saved_model.export_saved_model(
            revived_model, self.get_temp_dir(), **self.export_kwargs
        )

        # Test with a different batch size
        bigger_input = tree.map_structure(
            lambda x: tf.concat([x, x], axis=0), ref_input
        )
        revived_model(bigger_input)

    def test_model_with_multiple_inputs(self):
        class TwoInputsModel(models.Model):
            def call(self, x, y):
                return x + y

            def build(self, y_shape, x_shape):
                self.built = True

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TwoInputsModel()
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input_x, ref_input_y)

        saved_model.export_saved_model(
            model, temp_filepath, **self.export_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input_x, ref_input_y),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(
            tf.random.normal((6, 10)), tf.random.normal((6, 10))
        )

    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass"],
            input_signature=[
                layers.InputSpec(
                    dtype="float32", shape=(None, 10), name="inputs"
                ),
                tf.TensorSpec((None, 10), dtype="float32", name="inputs"),
                backend.KerasTensor((None, 10), dtype="float32", name="inputs"),
                "backend_tensor",
            ],
        )
    )
    def test_input_signature(self, model_type, input_signature):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = random.normal((batch_size, 10))
        ref_output = model(ref_input)

        if input_signature == "backend_tensor":
            input_signature = (ref_input,)
        else:
            input_signature = (input_signature,)
        saved_model.export_saved_model(
            model,
            temp_filepath,
            input_signature=input_signature,
            **self.export_kwargs,
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ops.convert_to_numpy(ref_input)),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )

    def test_input_signature_error(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("functional")
        with self.assertRaisesRegex(TypeError, "Unsupported x="):
            input_signature = (123,)
            saved_model.export_saved_model(
                model, temp_filepath, input_signature=input_signature
            )

    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass"],
            is_static=(True, False),
            jax2tf_kwargs=(
                None,
                {"enable_xla": True, "native_serialization": True},
            ),
        )
    )
    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is only for the jax backend.",
    )
    def test_jax_specific_kwargs(self, model_type, is_static, jax2tf_kwargs):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = random.uniform((3, 10))
        ref_output = model(ref_input)

        export_kwargs = self.export_kwargs.copy()
        if jax2tf_kwargs is not None:
            export_kwargs.setdefault("jax2tf_kwargs", {}).update(jax2tf_kwargs)

        saved_model.export_saved_model(
            model,
            temp_filepath,
            is_static=is_static,
            **export_kwargs,
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )


@pytest.mark.skipif(
    backend.backend()
    not in (
        "tensorflow",
        "jax",
        # "torch",  # TODO: Support low-level operations in the torch backend.
    ),
    reason="Export only currently supports the TF and JAX backends.",
)
@pytest.mark.skipif(testing.jax_uses_gpu(), reason="Leads to core dumps on CI")
@pytest.mark.skipif(
    testing.torch_uses_gpu(), reason="Leads to core dumps on CI"
)
class ExportArchiveTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.add_endpoint_kwargs = {}
        if testing.jax_uses_gpu():
            self.add_endpoint_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "cuda")
                }
            }
        elif testing.jax_uses_tpu():
            self.add_endpoint_kwargs = {
                "jax2tf_kwargs": {
                    "native_serialization_platforms": ("cpu", "tpu")
                }
            }

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_low_level_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model(model_type)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        # Test variable tracking
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        self.assertLen(export_archive.variables, 8)
        self.assertLen(export_archive.trainable_variables, 6)
        self.assertLen(export_archive.non_trainable_variables, 2)

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.call(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.call(tf.random.normal((6, 10)))

    def test_low_level_model_export_with_alias(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        fn = export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(
            temp_filepath,
            tf.saved_model.SaveOptions(function_aliases={"call_alias": fn}),
        )
        revived_model = tf.saved_model.load(
            temp_filepath,
            options=tf.saved_model.LoadOptions(
                experimental_load_function_aliases=True
            ),
        )
        self.assertAllClose(
            revived_model.function_aliases["call_alias"](ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.function_aliases["call_alias"](tf.random.normal((6, 10)))

    def test_export_does_not_duplicate_variable_storage(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        model(ref_input)  # Build the model.

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)

        reader = tf.train.load_checkpoint(
            os.path.join(temp_filepath, "variables", "variables")
        )
        shapes = reader.get_variable_to_shape_map()
        total = sum(
            int(np.prod(s))
            for n, s in shapes.items()
            if "CHECKPOINTABLE" not in n
        )
        self.assertEqual(total, model.count_params())

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_low_level_model_export_with_dynamic_dims(self, model_type):
        class ReductionLayer(layers.Layer):
            def call(self, inputs):
                return ops.max(inputs, axis=1)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model(
            model_type,
            input_shape=[(None,), (None,)],
            layer_list=[layers.Concatenate(), ReductionLayer()],
        )
        ref_input = [tf.random.normal((3, 8)), tf.random.normal((3, 6))]
        ref_output = model(ref_input)

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[
                [
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                ]
            ],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.call(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.call([tf.random.normal((6, 8)), tf.random.normal((6, 6))])
        # Test with a different batch size and different dynamic sizes
        revived_model.call([tf.random.normal((6, 3)), tf.random.normal((6, 5))])

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is only for the JAX backend.",
    )
    def test_low_level_model_export_with_jax2tf_kwargs(self):
        export_archive = saved_model.ExportArchive()
        with self.assertRaisesRegex(
            ValueError, "native_serialization_platforms.*bogus"
        ):
            export_archive.add_endpoint(
                "call",
                lambda x: x,
                input_signature=[
                    tf.TensorSpec(shape=(None, 10), dtype=tf.float32)
                ],
                jax2tf_kwargs={
                    "native_serialization": True,
                    "native_serialization_platforms": ("cpu", "bogus"),
                },
            )

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is only for the JAX backend.",
    )
    def test_low_level_model_export_with_jax2tf_polymorphic_shapes(self):
        class SquareLayer(layers.Layer):
            def call(self, inputs):
                return ops.matmul(inputs, inputs)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = CustomModel([SquareLayer()])
        ref_input = tf.random.normal((3, 10, 10))
        ref_output = model(ref_input)
        signature = [tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)]

        with self.assertRaises(TypeError):
            # This will fail because the polymorphic_shapes that is
            # automatically generated will not account for the fact that
            # dynamic dimensions 1 and 2 must have the same value.
            export_archive = saved_model.ExportArchive()
            export_archive.track(model)
            export_archive.add_endpoint(
                "call",
                model.__call__,
                input_signature=signature,
                **self.add_endpoint_kwargs,
            )
            export_archive.write_out(temp_filepath)

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        add_endpoint_kwargs = self.add_endpoint_kwargs.copy()
        add_endpoint_kwargs.setdefault("jax2tf_kwargs", {}).update(
            {"polymorphic_shapes": ["(batch, a, a)"]}
        )
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=signature,
            **add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.call(ref_input),
            ref_output,
            tpu_atol=0.05,
            tpu_rtol=1.0,
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="This test is native to the TF backend.",
    )
    def test_endpoint_registration_tf_function(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        # Test variable tracking
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        self.assertLen(export_archive.variables, 8)
        self.assertLen(export_archive.trainable_variables, 6)
        self.assertLen(export_archive.non_trainable_variables, 2)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        # Test registering an endpoint that is a tf.function (called)
        my_endpoint(ref_input)  # Trace fn

        export_archive.add_endpoint(
            "call",
            my_endpoint,
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)

        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(revived_model.call(ref_input), ref_output)
        self.assertLen(revived_model.variables, 8)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 2)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is native to the JAX backend.",
    )
    def test_jax_endpoint_registration_tf_function(self):
        model = get_model()
        ref_input = np.random.normal(size=(3, 10))
        model(ref_input)

        # build a JAX function
        def model_call(x):
            return model(x)

        from jax.experimental import jax2tf

        add_endpoint_kwargs = self.add_endpoint_kwargs.copy()
        add_endpoint_kwargs.setdefault("jax2tf_kwargs", {})

        # now, convert JAX function
        converted_model_call = jax2tf.convert(
            model_call,
            polymorphic_shapes=["(b, 10)"],
            **add_endpoint_kwargs["jax2tf_kwargs"],
        )

        # you can now build a TF inference function
        @tf.function(
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            autograph=False,
        )
        def infer_fn(x):
            return converted_model_call(x)

        ref_output = infer_fn(ref_input)

        # Export with TF inference function as endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model")
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve", infer_fn, **self.add_endpoint_kwargs
        )
        export_archive.write_out(temp_filepath)

        # Reload and verify outputs
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        self.assertLen(revived_model.variables, 8)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 2)

        # Assert all variables wrapped as `tf.Variable`
        self.assertIsInstance(export_archive.variables[0], tf.Variable)
        self.assertIsInstance(
            export_archive.trainable_variables[0], tf.Variable
        )
        self.assertIsInstance(
            export_archive.non_trainable_variables[0], tf.Variable
        )

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="This test is native to the JAX backend.",
    )
    def test_jax_multi_unknown_endpoint_registration(self):
        window_size = 100

        X = np.random.random((1024, window_size, 1))
        Y = np.random.random((1024, window_size, 1))

        model = models.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="relu"),
            ]
        )

        model.compile(optimizer="adam", loss="mse")

        model.fit(X, Y, batch_size=32)

        # build a JAX function
        def model_call(x):
            return model(x)

        from jax.experimental import jax2tf

        add_endpoint_kwargs = self.add_endpoint_kwargs.copy()
        add_endpoint_kwargs.setdefault("jax2tf_kwargs", {})

        # now, convert JAX function
        converted_model_call = jax2tf.convert(
            model_call,
            polymorphic_shapes=["(b, t, 1)"],
            **add_endpoint_kwargs["jax2tf_kwargs"],
        )

        # you can now build a TF inference function
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
            ],
            autograph=False,
        )
        def infer_fn(x):
            return converted_model_call(x)

        ref_input = np.random.random((1024, window_size, 1))
        ref_output = infer_fn(ref_input)

        # Export with TF inference function as endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model")
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve", infer_fn, **self.add_endpoint_kwargs
        )
        export_archive.write_out(temp_filepath)

        # Reload and verify outputs
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        self.assertLen(revived_model.variables, 6)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 0)

        # Assert all variables wrapped as `tf.Variable`
        self.assertIsInstance(export_archive.variables[0], tf.Variable)
        self.assertIsInstance(
            export_archive.trainable_variables[0], tf.Variable
        )

    def test_layer_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.BatchNormalization()
        ref_input = tf.random.normal((3, 10))
        ref_output = layer(ref_input)  # Build layer (important)

        export_archive = saved_model.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_layer.call(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )

    def test_multi_input_output_functional_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        x1 = layers.Input((2,))
        x2 = layers.Input((2,))
        y1 = layers.Dense(3)(x1)
        y2 = layers.Dense(3)(x2)
        model = models.Model([x1, x2], [y1, y2])

        ref_inputs = [tf.random.normal((3, 2)), tf.random.normal((3, 2))]
        ref_outputs = model(ref_inputs)

        model.export(temp_filepath, **self.add_endpoint_kwargs)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_outputs[0],
            revived_model.serve(ref_inputs)[0],
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        self.assertAllClose(
            ref_outputs[1],
            revived_model.serve(ref_inputs)[1],
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(
            [tf.random.normal((6, 2)), tf.random.normal((6, 2))]
        )

        # Now test dict inputs
        model = models.Model({"x1": x1, "x2": x2}, [y1, y2])

        ref_inputs = {
            "x1": tf.random.normal((3, 2)),
            "x2": tf.random.normal((3, 2)),
        }
        ref_outputs = model(ref_inputs)

        model.export(temp_filepath, **self.add_endpoint_kwargs)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_outputs[0],
            revived_model.serve(ref_inputs)[0],
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        self.assertAllClose(
            ref_outputs[1],
            revived_model.serve(ref_inputs)[1],
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(
            {
                "x1": tf.random.normal((6, 2)),
                "x2": tf.random.normal((6, 2)),
            }
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="String lookup requires TensorFlow backend",
    )
    def test_model_with_lookup_table(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        text_vectorization = layers.TextVectorization()
        text_vectorization.adapt(["one two", "three four", "five six"])
        model = models.Sequential(
            [
                layers.Input(shape=(), dtype="string"),
                text_vectorization,
                layers.Embedding(10, 32),
                layers.Dense(1),
            ]
        )
        ref_input = tf.convert_to_tensor(["one two three four"])
        ref_output = model(ref_input)

        saved_model.export_saved_model(
            model, temp_filepath, **self.add_endpoint_kwargs
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(revived_model.serve(ref_input), ref_output)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="String lookup requires TensorFlow backend",
    )
    def test_model_with_tracked_collection(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        text_vectorization = layers.TextVectorization()
        text_vectorization.adapt(["one two", "three four", "five six"])

        # CustomModel has a list of layers. The `TrackedList` that Keras uses is
        # not a TensorFlow Trackable, but `Layer._trackable_children` makes it
        # work.
        model = CustomModel(
            [
                text_vectorization,
                layers.Embedding(10, 32),
                layers.Dense(1),
            ]
        )
        ref_input = tf.convert_to_tensor(["one two three four"])
        ref_output = model(ref_input)

        saved_model.export_saved_model(
            model,
            temp_filepath,
            input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)],
            **self.add_endpoint_kwargs,
        )
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(revived_model.serve(ref_input), ref_output)

    def test_track_multiple_layers(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        layer_1 = layers.Dense(2)
        ref_input_1 = tf.random.normal((3, 4))
        ref_output_1 = layer_1(ref_input_1)
        layer_2 = layers.Dense(3)
        ref_input_2 = tf.random.normal((3, 5))
        ref_output_2 = layer_2(ref_input_2)

        export_archive = saved_model.ExportArchive()
        export_archive.add_endpoint(
            "call_1",
            layer_1.call,
            input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.add_endpoint(
            "call_2",
            layer_2.call,
            input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_layer.call_1(ref_input_1),
            ref_output_1,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        self.assertAllClose(
            revived_layer.call_2(ref_input_2),
            ref_output_2,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )

    def test_non_standard_layer_signature(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2)  # Build layer (important)
        export_archive = saved_model.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
            ],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_layer.call(x1, x2), ref_output, tpu_atol=0.05, tpu_rtol=0.1
        )

    def test_non_standard_layer_signature_with_kwargs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2)  # Build layer (important)
        export_archive = saved_model.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2, 2), dtype=tf.float32),
            ],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_layer.call(query=x1, value=x2),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_layer.call(
            query=tf.random.normal((6, 2, 2)), value=tf.random.normal((6, 2, 2))
        )

    def test_variable_collection(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(2),
                layers.Dense(2),
            ]
        )

        # Test variable tracking
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.add_variable_collection(
            "my_vars", model.layers[1].weights
        )

        self.assertLen(export_archive._tf_trackable.my_vars, 2)
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertLen(revived_model.my_vars, 2)

    def test_export_saved_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = models.Sequential([layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            saved_model.export_saved_model(model, temp_filepath)

        # Subclassed model has not been called
        model = get_model("subclass")
        model.build((2, 10))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            saved_model.export_saved_model(model, temp_filepath)

    def test_export_archive_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Dense(2)])
        model(tf.random.normal((2, 3)))

        # Endpoint name reuse
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            export_archive.add_endpoint(
                "call",
                model.__call__,
                input_signature=[
                    tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
                ],
            )

        # Write out with no endpoints
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            export_archive.write_out(temp_filepath)

        # Invalid object type
        with self.assertRaisesRegex(ValueError, "Invalid resource type"):
            export_archive = saved_model.ExportArchive()
            export_archive.track("model")

        # Set endpoint with no input signature
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must provide an `input_signature`"
        ):
            export_archive.add_endpoint("call", model.__call__)

        # Set endpoint that has never been called
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must either provide a function"
        ):
            export_archive.add_endpoint("call", my_endpoint)

    def test_export_no_assets(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Case where there are legitimately no assets.
        model = models.Sequential([layers.Flatten()])
        model(tf.random.normal((2, 3)))
        export_archive = saved_model.ExportArchive()
        export_archive.add_endpoint(
            "call",
            model.__call__,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.write_out(temp_filepath)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_export_method(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        model.export(temp_filepath, **self.add_endpoint_kwargs)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.serve(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )
        # Test with a different batch size
        revived_model.serve(tf.random.normal((6, 10)))

    def test_model_combined_with_tf_preprocessing(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["a", "b", "c"]), tf.constant([1.0, 2.0, 3.0])
            ),
            default_value=-1.0,
        )
        ref_input = tf.constant([["c", "b", "c", "a", "d"]])
        ref_intermediate = lookup_table.lookup(ref_input)

        model = models.Sequential([layers.Dense(1)])
        ref_output = model(ref_intermediate)

        export_archive = saved_model.ExportArchive()
        model_fn = export_archive.track_and_add_endpoint(
            "model",
            model,
            input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)],
            **self.add_endpoint_kwargs,
        )
        export_archive.track(lookup_table)

        @tf.function()
        def combined_fn(x):
            x = lookup_table.lookup(x)
            x = model_fn(x)
            return x

        self.assertAllClose(
            combined_fn(ref_input), ref_output, tpu_atol=0.01, tpu_rtol=0.01
        )

        export_archive.add_endpoint(
            "combined_fn", combined_fn, **self.add_endpoint_kwargs
        )
        export_archive.write_out(temp_filepath)

        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            revived_model.combined_fn(ref_input),
            ref_output,
            tpu_atol=0.01,
            tpu_rtol=0.01,
        )


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="`export_xla_saved_model` only supports the JAX backend.",
)
@pytest.mark.skipif(
    not xla_saved_model_archive.xla_saved_model.available,
    reason="Requires the `xla_saved_model` package.",
)
class ExportXLASavedModelTest(testing.TestCase):
    """Tests for `export_xla_saved_model` / `model.export("xla_saved_model")`.

    The `XLASavedModelArchive` itself is covered in
    `xla_saved_model_archive_test.py`; these tests cover the high-level
    export entry points in `saved_model.py` and `Model.export()`.
    """

    def _load_serve(self, filepath, signature="serve"):
        # Reloads the exported function as `fn(params, inputs)` and the
        # Orbax checkpoint it references, as the XLA Serving Engine does.
        from keras.src.export import xla_saved_model_archive_test

        return xla_saved_model_archive_test.load_endpoint(filepath, signature)

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        # The input signature is inferred from the model.
        saved_model.export_xla_saved_model(
            model, temp_filepath, platforms=["cpu"]
        )
        self.assertTrue(
            os.path.isfile(os.path.join(temp_filepath, "manifest.pb"))
        )
        self.assertTrue(
            os.path.isdir(os.path.join(temp_filepath, "checkpoint"))
        )
        # Registered as "serve" and aliased as "serving_default".
        for signature in ("serve", "serving_default"):
            apply_fn, params = self._load_serve(temp_filepath, signature)
            self.assertAllClose(
                apply_fn(params, {"inputs": ref_input})["output_0"],
                ref_output,
            )
        # Test with a different batch size
        apply_fn, params = self._load_serve(temp_filepath)
        other_input = np.random.normal(size=(6, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": other_input})["output_0"],
            model(other_input),
        )

    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass"],
            input_signature=[
                layers.InputSpec(
                    dtype="float32", shape=(None, 10), name="inputs"
                ),
                tf.TensorSpec((None, 10), dtype="float32", name="inputs"),
                backend.KerasTensor((None, 10), dtype="float32", name="inputs"),
                "backend_tensor",
            ],
        )
    )
    def test_input_signature(self, model_type, input_signature):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = random.normal((3, 10))
        ref_output = model(ref_input)

        if input_signature == "backend_tensor":
            input_signature = (ref_input,)
        else:
            input_signature = (input_signature,)
        saved_model.export_xla_saved_model(
            model,
            temp_filepath,
            input_signature=input_signature,
            platforms=["cpu"],
        )
        apply_fn, params = self._load_serve(temp_filepath)
        self.assertAllClose(
            apply_fn(params, {"inputs": ops.convert_to_numpy(ref_input)})[
                "output_0"
            ],
            ref_output,
        )

    def test_model_with_multiple_inputs(self):
        class TupleModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(4)

            def call(self, inputs):
                x, y = inputs
                return self.dense(ops.add(x, y))

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TupleModel()
        ref_inputs = (
            np.random.normal(size=(3, 10)).astype("float32"),
            np.random.normal(size=(3, 10)).astype("float32"),
        )
        ref_output = model(ref_inputs)

        saved_model.export_xla_saved_model(
            model,
            temp_filepath,
            input_signature=[
                (
                    layers.InputSpec(
                        shape=(None, 10), dtype="float32", name="x"
                    ),
                    layers.InputSpec(
                        shape=(None, 10), dtype="float32", name="y"
                    ),
                )
            ],
            platforms=["cpu"],
        )
        apply_fn, params = self._load_serve(temp_filepath)
        self.assertAllClose(
            apply_fn(params, {"x": ref_inputs[0], "y": ref_inputs[1]})[
                "output_0"
            ],
            ref_output,
        )

    def test_verbose(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        model.build((None, 10))
        # `verbose=None` defaults to True and prints the endpoints; both
        # settings must export successfully.
        saved_model.export_xla_saved_model(
            model, temp_filepath, verbose=None, platforms=["cpu"]
        )
        saved_model.export_xla_saved_model(
            model, temp_filepath, verbose=False, platforms=["cpu"]
        )
        self.assertTrue(
            os.path.isfile(os.path.join(temp_filepath, "manifest.pb"))
        )

    def test_export_xla_saved_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = models.Sequential([layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            saved_model.export_xla_saved_model(model, temp_filepath)

        # Subclassed model has not been called
        model = get_model("subclass")
        model.build((2, 10))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            saved_model.export_xla_saved_model(model, temp_filepath)

        # Unsupported input signature element
        model = get_model("functional")
        with self.assertRaisesRegex(TypeError, "Unsupported x="):
            saved_model.export_xla_saved_model(
                model, temp_filepath, input_signature=(123,)
            )

        # TF SavedModel-only kwargs are rejected
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            saved_model.export_xla_saved_model(
                model, temp_filepath, jax2tf_kwargs={}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported keyword arg"):
            saved_model.export_xla_saved_model(
                model, temp_filepath, is_static=True
            )

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional", "subclass"])
    )
    def test_model_export_method(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        ref_output = model(ref_input)

        model.export(temp_filepath, format="xla_saved_model", platforms=["cpu"])
        apply_fn, params = self._load_serve(temp_filepath)
        self.assertAllClose(
            apply_fn(params, {"inputs": ref_input})["output_0"], ref_output
        )
        # Test with a different batch size
        other_input = np.random.normal(size=(6, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"inputs": other_input})["output_0"],
            model(other_input),
        )

    def test_model_export_method_forwards_kwargs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        model.build((None, 10))
        model.export(
            temp_filepath,
            format="xla_saved_model",
            verbose=False,
            input_signature=[
                layers.InputSpec(shape=(None, 10), dtype="float32", name="x")
            ],
            platforms=["cpu"],
            warmup_examples=[{"x": np.zeros((1, 10), dtype="float32")}],
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    temp_filepath, "assets.extra", "xla_serving_warmup_requests"
                )
            )
        )
        apply_fn, params = self._load_serve(temp_filepath)
        ref_input = np.random.normal(size=(2, 10)).astype("float32")
        self.assertAllClose(
            apply_fn(params, {"x": ref_input})["output_0"], model(ref_input)
        )
