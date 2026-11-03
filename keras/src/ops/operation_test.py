import numpy as np
import pytest

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.backend.common import standardize_dtype
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

    def test_serialization_custom_constructor_with_kwargs_auto_config(self):
        # Explicit name passed in constructor is serialized and deserialized.
        op = OpWithKwargsInConstructor(alpha=0.2, name="test_op")
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 1.0, "name": "test_op"})
        revived = OpWithKwargsInConstructor.from_config(config)
        self.assertEqual(revived.get_config(), config)
        self.assertEqual(revived.name, op.name)

        # Auto generated name is serialized and deserialized.
        op = OpWithKwargsInConstructor(alpha=0.2, beta=0.0)
        config = op.get_config()
        self.assertEqual(config, {"alpha": 0.2, "beta": 0.0, "name": op.name})
        revived = OpWithKwargsInConstructor.from_config(config)
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


class ElementwiseUnaryOperationTest(testing.TestCase):
    def test_elementwise_unary_operation_basic(self):
        class DummyRelu(operation.ElementwiseUnaryOperation):
            call_fn = backend.ops.nn.relu
            preserves_sparse = True

        op = DummyRelu()
        x = np.array([-2.0, 0.0, 3.0], dtype="float32")
        out = op(x)
        self.assertAllClose(out, [0.0, 0.0, 3.0])

        sym_x = keras_tensor.KerasTensor((2, 3), dtype="float32")
        sym_out = op(sym_x)
        self.assertEqual(sym_out.shape, (2, 3))
        self.assertEqual(sym_out.dtype, "float32")
        self.assertFalse(sym_out.sparse)
        self.assertFalse(sym_out.ragged)

        sym_sparse_x = keras_tensor.KerasTensor(
            (2, 3), dtype="float32", sparse=True
        )
        sym_sparse_out = op(sym_sparse_x)
        self.assertTrue(sym_sparse_out.sparse)
        self.assertFalse(sym_sparse_out.ragged)

        # Non-sparse preserving op should not propagate sparse=True
        class DummySigmoid(operation.ElementwiseUnaryOperation):
            call_fn = backend.ops.nn.sigmoid

        dense_op = DummySigmoid()
        self.assertFalse(dense_op(sym_sparse_x).sparse)

        sym_ragged_x = keras_tensor.KerasTensor(
            (2, None), dtype="float32", ragged=True
        )
        sym_ragged_out = op(sym_ragged_x)
        self.assertFalse(sym_ragged_out.sparse)
        self.assertTrue(sym_ragged_out.ragged)

    def test_elementwise_unary_operation_missing_call_fn(self):
        with self.assertRaisesRegex(ValueError, "must define `call_fn`"):

            class MissingFnOp(operation.ElementwiseUnaryOperation):
                pass

    def test_elementwise_unary_operation_output_dtype(self):
        class DummyIsNan(operation.ElementwiseUnaryOperation):
            call_fn = backend.ops.numpy.isnan
            output_dtype = "bool"

        op = DummyIsNan()
        sym_x = keras_tensor.KerasTensor((2, 3), dtype="float32")
        sym_out = op(sym_x)
        self.assertEqual(sym_out.shape, (2, 3))
        self.assertEqual(sym_out.dtype, "bool")

    def test_elementwise_unary_operation_promote_to_float(self):
        class DummyPromoteOp(operation.ElementwiseUnaryOperation):
            call_fn = backend.ops.nn.relu
            promote_to_float = True

        op = DummyPromoteOp()
        sym_int = keras_tensor.KerasTensor((2, 3), dtype="int32")
        sym_out = op(sym_int)
        self.assertEqual(sym_out.shape, (2, 3))
        self.assertEqual(sym_out.dtype, backend.floatx())

        sym_bool = keras_tensor.KerasTensor((2, 3), dtype="bool")
        sym_out_bool = op(sym_bool)
        self.assertEqual(sym_out_bool.dtype, backend.floatx())

        sym_float16 = keras_tensor.KerasTensor((2, 3), dtype="float16")
        sym_out_f16 = op(sym_float16)
        self.assertEqual(sym_out_f16.dtype, "float16")

    def test_subclasses_match_eager(self):
        for cls in operation.ElementwiseUnaryOperation.__subclasses__():
            if not cls.__module__.startswith(
                "keras.src.ops."
            ) or cls.__module__.endswith("_test"):
                continue
            if cls.__name__ == "Imag" and backend.backend() == "torch":
                continue
            op = cls(0.5, 0.0) if cls.__name__ == "Threshold" else cls()
            dt = "int32" if "Bitwise" in cls.__name__ else "float32"
            for shape in ((), (3,), (2, 3)):
                out = op(ops.ones(shape, dtype=dt))
                sym = op(keras_tensor.KerasTensor(shape, dtype=dt))
                self.assertEqual(
                    tuple(sym.shape), tuple(out.shape), cls.__name__
                )
                self.assertEqual(
                    sym.dtype, standardize_dtype(out.dtype), cls.__name__
                )

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_preserves_sparse_flags_match_backend(self):
        for cls in operation.ElementwiseUnaryOperation.__subclasses__():
            if (
                cls.preserves_sparse
                and cls.__module__.startswith("keras.src.ops.")
                and not cls.__module__.endswith("_test")
            ):
                dt = "int32" if "Bitwise" in cls.__name__ else "float32"
                sparse_2x2 = ops.one_hot([0, 1], 2, dtype=dt, sparse=True)
                out = cls()(sparse_2x2)
                self.assertSparse(out)
