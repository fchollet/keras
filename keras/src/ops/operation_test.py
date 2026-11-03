import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.ops import numpy as knp
from keras.src.ops import operation
from keras.src.utils import traceback_utils


class OpWithMultipleInputs(operation.Operation):
    def call(self, x, y, z=None):
        # `z` has to be put first due to the order of operations issue with
        # torch backend.
        return 3 * z + x + 2 * y

    def compute_output_spec(self, x, y, z=None):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithMultipleOutputs(operation.Operation):
    def call(self, x):
        return (x, x + 1)

    def compute_output_spec(self, x):
        return (
            keras_tensor.KerasTensor(x.shape, x.dtype),
            keras_tensor.KerasTensor(x.shape, x.dtype),
        )


class OpWithCustomConstructor(operation.Operation):
    def __init__(self, alpha, *, beta=1.0, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return self.alpha * x + self.beta

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithCustomConstructorNoName(operation.Operation):
    def __init__(self, alpha, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return self.alpha * x + self.beta

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithKwargsInConstructor(operation.Operation):
    def __init__(self, alpha, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return self.alpha * x + self.beta

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithRenamedKwargsInConstructor(operation.Operation):
    # Same as `OpWithKwargsInConstructor`, but the variadic keyword argument
    # is not literally named `kwargs`.
    def __init__(self, alpha, beta=1.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return self.alpha * x + self.beta

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithArgsInConstructor(operation.Operation):
    def __init__(self, alpha, *args, name=None):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, x):
        return self.alpha * x + self.beta

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OpWithCustomConstructorGetConfig(operation.Operation):
    def __init__(self, alpha, *, name=None):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, x):
        return self.alpha * x

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha}


class OpWithKwargsInConstructorGetConfig(operation.Operation):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        return self.alpha * x

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha}


class OpWithQuantizedCall(operation.Operation):
    def call(self, x):
        return x

    def quantized_call(self, x):
        return x + 1

    def compute_output_spec(self, x):
        return keras_tensor.KerasTensor(x.shape, x.dtype)


class OperationTest(testing.TestCase):
    def test_symbolic_call(self):
        x = keras_tensor.KerasTensor(shape=(2, 3), name="x")
        y = keras_tensor.KerasTensor(shape=(2, 3), name="y")
        z = keras_tensor.KerasTensor(shape=(2, 3), name="z")

        # Positional arguments
        op = OpWithMultipleInputs(name="test_op")
        self.assertEqual(op.name, "test_op")
        out = op(x, y, z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Keyword arguments
        op = OpWithMultipleInputs(name="test_op")
        out = op(x=x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Mix
        op = OpWithMultipleInputs(name="test_op")
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)

        # Test op reuse
        prev_out = out
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 2)
        self.assertEqual(op.output, prev_out)

        # Test multiple outputs
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], keras_tensor.KerasTensor)
        self.assertIsInstance(out[1], keras_tensor.KerasTensor)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.output, list(out))

    def test_eager_call(self):
        x = knp.ones((2, 3))
        y = knp.ones((2, 3))
        z = knp.ones((2, 3))
        op = OpWithMultipleInputs(name="test_op")
        self.assertEqual(op.name, "test_op")

        # Positional arguments
        out = op(x, y, z)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Keyword arguments
        out = op(x=x, y=y, z=z)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Mixed arguments
        out = op(x, y=y, z=z)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Test multiple outputs
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertEqual(len(out), 2)
        self.assertTrue(backend.ops.is_tensor(out[0]))
        self.assertTrue(backend.ops.is_tensor(out[1]))
        self.assertAllClose(out[0], np.ones((2, 3)))
        self.assertAllClose(out[1], np.ones((2, 3)) + 1)

    def test_rematerialized_call(self):
        from keras.src.backend.common.remat import RematScope

        x = knp.ones((2, 3))

        # Test standard rematerialized call using RematScope
        with RematScope(mode="full"):
            op = OpWithMultipleInputs(name="test_op")
        out = op(x, x, x)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))

        # Test quantized rematerialized call
        with RematScope(mode="full"):
            op_q = OpWithQuantizedCall(name="test_op_q")
        op_q.quantization_mode = object()
        out = op_q(x)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 2 * np.ones((2, 3)))

        # Test traceback filtering branch
        was_enabled = traceback_utils.is_traceback_filtering_enabled()
        traceback_utils.enable_traceback_filtering()
        try:
            # Standard rematerialized
            with RematScope(mode="full"):
                op = OpWithMultipleInputs(name="test_op_traceback")
            out = op(x, x, x)
            self.assertTrue(backend.ops.is_tensor(out))
            self.assertAllClose(out, 6 * np.ones((2, 3)))

            # Quantized rematerialized
            with RematScope(mode="full"):
                op_q = OpWithQuantizedCall(name="test_op_q_traceback")
            op_q.quantization_mode = object()
            out = op_q(x)
            self.assertTrue(backend.ops.is_tensor(out))
            self.assertAllClose(out, 2 * np.ones((2, 3)))
        finally:
            if not was_enabled:
                traceback_utils.disable_traceback_filtering()

    def test_serialization_with_default_init_and_get_config(self):
        # Explicit name passed in constructor is serialized and deserialized.
        op = OpWithMultipleInputs(name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"name": "test_op"})
        revived = OpWithMultipleInputs.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = OpWithMultipleInputs()
        config = op.get_config()
        self.assertEqual(config, {"name": op.name})
        revived = OpWithMultipleInputs.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

    def test_serialization_custom_constructor_with_name_auto_config(self):
        # Explicit name passed in constructor is serialized and deserialized.
        op = OpWithCustomConstructor(alpha=0.2, name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 1.0, "name": "test_op"})
        revived = OpWithCustomConstructor.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = OpWithCustomConstructor(alpha=0.2, beta=0.0)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 0.0, "name": op.name})
        revived = OpWithCustomConstructor.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

    def test_serialization_custom_constructor_with_no_name_auto_config(self):
        # Auto generated name is not serialized.
        op = OpWithCustomConstructorNoName(alpha=0.2)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 1.0})
        revived = OpWithCustomConstructorNoName.from_config(config)
        self.assertEqual(revived.get_config(), config)

    @parameterized.named_parameters(
        ("kwargs", OpWithKwargsInConstructor),
        ("renamed_kwargs", OpWithRenamedKwargsInConstructor),
    )
    def test_serialization_custom_constructor_with_kwargs_auto_config(
        self, op_class
    ):
        # Explicit name passed in constructor is serialized and deserialized.
        op = op_class(alpha=0.2, name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 1.0, "name": "test_op"})
        revived = op_class.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = op_class(alpha=0.2, beta=0.0)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 0.0, "name": op.name})
        revived = op_class.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

    def test_failing_serialization_non_serializable_auto_config(
        self,
    ):
        class NonSerializable:
            pass

        # Custom class cannot be automatically serialized.
        op = OpWithCustomConstructor(alpha=NonSerializable(), name="test_op")
        with self.assertRaises(NotImplementedError):
            _ = op.get_config()

    def test_failing_serialization_custom_constructor_with_args_auto_config(
        self,
    ):
        # Custom constructor with variadic args cannot be automatically
        # serialized.
        op = OpWithArgsInConstructor(0.2, "a", "b", "c", name="test_op")
        with self.assertRaises(NotImplementedError):
            _ = op.get_config()

    def test_serialization_custom_constructor_custom_get_config(self):
        # Explicit name passed in constructor is serialized and deserialized.
        op = OpWithCustomConstructorGetConfig(alpha=0.2, name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "name": "test_op"})
        revived = OpWithCustomConstructorGetConfig.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = OpWithCustomConstructorGetConfig(alpha=0.2)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "name": op.name})
        revived = OpWithCustomConstructorGetConfig.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

    def test_serialization_custom_constructor_with_kwargs_custom_get_config(
        self,
    ):
        # Explicit name passed in constructor is serialized and deserialized.
        op = OpWithKwargsInConstructorGetConfig(alpha=0.2, name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "name": "test_op"})
        revived = OpWithKwargsInConstructorGetConfig.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = OpWithKwargsInConstructorGetConfig(alpha=0.2)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "name": op.name})
        revived = OpWithKwargsInConstructorGetConfig.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

    def test_input_conversion(self):
        x = np.ones((2,))
        y = np.ones((2,))
        z = knp.ones((2,))  # mix
        if backend.backend() == "torch":
            z = z.cpu()
        op = OpWithMultipleInputs()
        out = op(x, y, z)
        self.assertTrue(backend.ops.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2,)))

    def test_valid_naming(self):
        OpWithMultipleOutputs(name="test_op")

        with self.assertRaisesRegex(
            ValueError, "must be a string and cannot contain character `/`."
        ):
            OpWithMultipleOutputs(name="test/op")
