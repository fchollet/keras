import itertools
from unittest.mock import create_autospec

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common import standardize_shape
from keras.src.backend.common.variables import AutocastScope
from keras.src.backend.common.variables import shape_equal
from keras.src.distribution.distribution_lib import TensorLayout
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product


class VariableInitializationTest(test_case.TestCase):
    """Tests for Variable.__init__()"""

    def test_deferred_initialization(self):
        """Tests deferred initialization of variables."""
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            # Variables can nevertheless be accessed
            _ = v + 1
        self.assertEqual(v._value.shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "while in a stateless scope"):
            with backend.StatelessScope():
                v = backend.Variable(initializer=0)

    def test_variable_initialization_with_numpy_array(self):
        """Test variable init with numpy array initializer."""
        v = backend.Variable(
            initializer=np.ones((2, 2), dtype=np.int32), trainable=False
        )
        self.assertAllClose(v.value, np.ones((2, 2)))
        self.assertEqual(v.dtype, "int32")

    def test_variable_initialization_with_native_array(self):
        """Test variable init with native array initializer."""
        v = backend.Variable(
            initializer=ops.ones((2, 2), dtype="int32"), trainable=False
        )
        self.assertAllClose(v.value, np.ones((2, 2)))
        self.assertEqual(v.dtype, "int32")

    def test_variable_initialization_with_python_array(self):
        """Test variable init with python array initializer."""
        v = backend.Variable(initializer=[[1, 1], [1, 1]], trainable=False)
        self.assertAllClose(v.value, np.ones((2, 2)))
        self.assertEqual(v.dtype, "int32")
        v = backend.Variable(
            initializer=[[1.0, 1.0], [1.0, 1.0]], trainable=False
        )
        self.assertAllClose(v.value, np.ones((2, 2)))
        self.assertEqual(v.dtype, "float32")

    def test_variable_initialization_with_lambda_expression(self):
        # Test Python number
        v = backend.Variable(
            initializer=lambda *a, **kw: 1.0,
            shape=(),
            dtype="float32",
        )
        self.assertAllClose(v.value, 1.0)
        self.assertEqual(v.dtype, "float32")

        # Test Python array
        v = backend.Variable(
            initializer=lambda *a, **kw: [1.0],
            shape=(1,),
            dtype="float32",
        )
        self.assertAllClose(v.value, np.ones((1,)))
        self.assertEqual(v.dtype, "float32")

        # Test numpy array
        v = backend.Variable(
            initializer=lambda *a, **kw: np.ones((1,)),
            shape=(1,),
            dtype="float32",
        )
        self.assertAllClose(v.value, np.ones((1,)))
        self.assertEqual(v.dtype, "float32")

        # Test backend array
        v = backend.Variable(
            initializer=lambda *a, **kw: ops.ones((1,)),
            shape=(1,),
            dtype="float32",
        )
        self.assertAllClose(v.value, np.ones((1,)))
        self.assertEqual(v.dtype, "float32")

    def test_variable_initialization_with_strings(self):
        """Test variable init with non-callable initializer."""
        v = backend.Variable(initializer="ones", shape=(2, 2))
        self.assertAllClose(v.value, np.ones((2, 2)))

    def test_variable_initialization_with_non_trainable(self):
        """Test variable initialization with non-trainable flag."""
        v = backend.Variable(initializer=np.ones((2, 2)), trainable=False)
        self.assertFalse(v.trainable)

    def test_variable_initialization_without_shape(self):
        """Test variable init without a shape."""
        with self.assertRaisesRegex(
            ValueError,
            "When creating a Variable from an initializer, the `shape` ",
        ):
            backend.Variable(initializer=initializers.RandomNormal())

    def test_deferred_initialize_already_initialized(self):
        """Test deferred init on an already initialized variable."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            ValueError, f"Variable {v.path} is already initialized."
        ):
            v._deferred_initialize()

    def test_variable_initialize(self):
        """Test initializing a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        init_value = np.array([4.0, 5.0, 6.0])
        v._initialize(value=init_value)
        self.assertAllClose(v.value, init_value)

    def test_variable_without_shape_from_callable_initializer(self):
        """Test that Variable raises error
        if shape is not provided for callable initializer."""
        with self.assertRaisesRegex(
            ValueError, "When creating a Variable from an initializer"
        ):
            backend.Variable(initializer=lambda: np.ones((2, 2)))

    def test_variable_with_layout_with_no_device_mesh(self):
        """Test that Variable raises error if layout has no mesh."""
        layout = TensorLayout(axes=(None, "model"), device_mesh=None)
        with self.assertRaisesRegex(
            ValueError, "Argument `layout` must have `device_mesh` set."
        ):
            backend.Variable(initializer=np.ones((2, 2)), layout=layout)


class VariablePropertiesTest(test_case.TestCase):
    """Tests for Variable._deferred_initialize Variable._maybe_autocast"""

    def test_deferred_assignment(self):
        """Tests deferred assignment to variables."""
        with backend.StatelessScope() as scope:
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            v.assign(np.zeros((2, 2)))
            v.assign_add(2 * np.ones((2, 2)))
            v.assign_sub(np.ones((2, 2)))
        out = scope.get_current_value(v)
        self.assertAllClose(out, np.ones((2, 2)))

    def test_trainable_setter(self):
        """Tests the trainable setter."""
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
        )
        self.assertTrue(v.trainable)
        v.trainable = False
        self.assertFalse(v.trainable)

        if backend.backend() == "torch":
            v.trainable = True
            self.assertTrue(v._value.requires_grad)
            v.trainable = False
            self.assertFalse(v._value.requires_grad)

    def test_constraint_setter_accepts_plain_callable(self):
        from keras.src.constraints import NonNeg

        v = backend.Variable(initializer="ones", shape=(2,))

        def clip_fn(w):
            return w * 0.5

        v.constraint = clip_fn
        self.assertIs(v.constraint, clip_fn)

        v.constraint = NonNeg()
        self.assertIsInstance(v.constraint, NonNeg)

        v.constraint = None
        self.assertIsNone(v.constraint)

        with self.assertRaisesRegex(
            ValueError, "Invalid value for attribute `constraint`"
        ):
            v.constraint = "not callable"

    def test_registered_callable_constraint_survives_serialization(self):
        from keras.src.layers import Dense
        from keras.src.saving import register_keras_serializable

        @register_keras_serializable(package="test_22221")
        def half_constraint(w):
            return w * 0.5

        layer = Dense(4, kernel_constraint=half_constraint)
        layer.build((None, 3))
        config = layer.get_config()
        restored = Dense.from_config(config)
        restored.build((None, 3))
        self.assertIs(restored.kernel_constraint, half_constraint)

    def test_regularizer_setter_accepts_plain_callable(self):
        from keras.src.regularizers import L1

        v = backend.Variable(initializer="ones", shape=(2,))

        def reg_fn(w):
            return w * w

        v.regularizer = reg_fn
        self.assertIs(v.regularizer, reg_fn)

        v.regularizer = L1()
        self.assertIsInstance(v.regularizer, L1)

        v.regularizer = None
        self.assertIsNone(v.regularizer)

        with self.assertRaisesRegex(
            ValueError, "Invalid value for attribute `regularizer`"
        ):
            v.regularizer = "not callable"

    def test_registered_callable_regularizer_survives_serialization(self):
        from keras.src.layers import Dense
        from keras.src.saving import register_keras_serializable

        @register_keras_serializable(package="test")
        def reg_fn(w):
            return w * w

        layer = Dense(4, kernel_regularizer=reg_fn)
        layer.build((None, 3))
        config = layer.get_config()
        restored = Dense.from_config(config)
        restored.build((None, 3))
        self.assertIs(restored.kernel_regularizer, reg_fn)

    def test_autocasting_float(self):
        # Tests autocasting of float variables
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")
        with AutocastScope("float16"):
            self.assertEqual(standardize_dtype(v.value.dtype), "float16")
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

    def test_autocasting_float_assign(self):
        # Tests assigning value to variable within an autocast scope
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

        # Assign float16 value within float16 scope
        with AutocastScope("float16"):
            self.assertEqual(standardize_dtype(v.value.dtype), "float16")
            v.assign(ops.ones((2, 2), "float16"))
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

        # Assign float32 value within float16 scope
        with AutocastScope("float16"):
            self.assertEqual(standardize_dtype(v.value.dtype), "float16")
            v.assign(ops.zeros((2, 2), "float32"))
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

    def test_autocasting_int(self):
        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(),
            shape=(2, 2),
            dtype="int32",
            trainable=False,
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(standardize_dtype(v.value.dtype), "int32")

        with AutocastScope("float16"):
            self.assertEqual(standardize_dtype(v.value.dtype), "int32")

    def test_autocasting_float_with_autocast_off(self):
        # Test autocast argument
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
            autocast=False,
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")
        with AutocastScope("float16"):
            self.assertEqual(
                standardize_dtype(v.value.dtype),
                "float32",  # ignore AutocastScope
            )
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

    @parameterized.named_parameters(
        ("autocast_on", True),
        ("autocast_off", False),
    )
    def test_assign_in_autocast_scope(self, autocast):
        # Assignments in an `AutocastScope` of another dtype must update the
        # variable itself, not the cast copy returned by `value`.
        v = backend.Variable(
            initializer=initializers.Zeros(),
            shape=(2, 2),
            dtype="float32",
            autocast=autocast,
        )
        with AutocastScope("bfloat16"):
            v.assign(np.ones((2, 2), "float32"))
            v.assign_add(np.full((2, 2), 2.0, "float32"))
        self.assertAllClose(v.value, np.full((2, 2), 3.0))
        self.assertEqual(standardize_dtype(v.value.dtype), "float32")

    @parameterized.named_parameters(
        named_product(
            dtype=(
                dtype for dtype in dtypes.ALLOWED_DTYPES if dtype != "string"
            )
        )
    )
    def test_standardize_dtype(self, dtype):
        """Tests standardize_dtype for all ALLOWED_DTYPES except string."""
        if backend.backend() == "torch" and dtype in (
            "uint16",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        ):
            self.skipTest(f"torch backend does not support dtype {dtype}")

        if backend.backend() == "jax":
            if dtype in ("complex128",):
                self.skipTest(f"jax backend does not support dtype {dtype}")
            import jax

            if not jax.config.x64_enabled and "64" in dtype:
                self.skipTest(
                    f"jax backend does not support {dtype} without x64 enabled"
                )

        if not backend.SUPPORTS_COMPLEX_DTYPES and dtype in (
            "complex64",
            "complex128",
        ):
            self.skipTest(f"backend does not support dtype {dtype}")

        x = backend.ops.convert_to_tensor(np.zeros(()), dtype)
        actual = standardize_dtype(x.dtype)
        self.assertEqual(actual, dtype)

    def test_standardize_dtype_with_torch_dtype(self):
        """Tests dtype standardization with PyTorch dtypes."""
        import torch

        x = torch.randn(4, 4)
        standardize_dtype(x.dtype)

    def test_name_validation(self):
        """Tests validation of variable names."""
        with self.assertRaisesRegex(
            ValueError, "Argument `name` must be a string"
        ):
            backend.Variable(
                initializer=initializers.RandomNormal(), name=12345
            )

        with self.assertRaisesRegex(ValueError, "cannot contain character `/`"):
            backend.Variable(
                initializer=initializers.RandomNormal(), name="invalid/name"
            )

    def test_standardize_shape_with_none(self):
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        shape = (3, 4, 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_valid_input_with_none(self):
        shape = (3, None, 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, None, 5))

    def test_standardize_shape_with_valid_not_tuple_input(self):
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_numpy(self):
        shape = [3, np.int32(4), np.int64(5)]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_string(self):
        shape_with_string = (3, 4, "5")
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension '5'.",
        ):
            standardize_shape(shape_with_string)

    def test_standardize_shape_with_float(self):
        shape_with_float = (3, 4, 5.0)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension '5.0'.",
        ):
            standardize_shape(shape_with_float)

    def test_standardize_shape_with_object(self):
        shape_with_object = (3, 4, object())
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid dimension .*object",
        ):
            standardize_shape(shape_with_object)

    def test_standardize_shape_with_negative_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Negative dimensions",
        ):
            standardize_shape((3, 4, -5))

    @parameterized.named_parameters(
        ("all_dynamic", (None, None, None, 64), (None, None, None, 64)),
        ("mixed", (None, 224, 224, 3), (None, 224, 224, 3)),
        ("all_static", (1, 224, 224, 3), (1, 224, 224, 3)),
    )
    def test_standardize_shape_preserves_none(self, input_shape, expected):
        """Test that None dimensions are preserved correctly."""
        result = standardize_shape(input_shape)
        self.assertEqual(result, expected)

    def test_shape_equal_length_mismatch(self):
        """Test mismatch in lengths of shapes."""
        self.assertFalse(shape_equal((3, 2), (3, 2, 4)))
        self.assertFalse(shape_equal((), (3,)))
        self.assertFalse(shape_equal((3, 2, 4, 5), (3, 2, 4)))

    def test_autocast_scope_with_non_float_dtype(self):
        """Tests autocast scope with non-float dtype."""
        with self.assertRaisesRegex(
            ValueError,
            "`AutocastScope` can only be used with a floating-point",
        ):
            _ = AutocastScope("int32")

    def test_variable_path_creation(self):
        """Test path creation for a variable."""
        v = backend.Variable(initializer=np.ones((2, 2)), name="test_var")
        self.assertEqual(v.path, "test_var")

        with backend.name_scope("test_scope"):
            v = backend.Variable(initializer=np.ones((2, 2)), name="test_var")
            self.assertEqual(v.path, "test_scope/test_var")

    def test_overwrite_with_gradient_setter(self):
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
        )
        self.assertFalse(v.overwrite_with_gradient)
        v.overwrite_with_gradient = True
        self.assertTrue(v.overwrite_with_gradient)

        with self.assertRaisesRegex(TypeError, "must be a boolean."):
            v.overwrite_with_gradient = "true"


class VariableNumpyValueAndAssignmentTest(test_case.TestCase):
    """tests for Variable.numpy(), Variable.value() and Variable.assign()"""

    def test_variable_numpy(self):
        """Test retrieving the value of a variable as a numpy array."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertIsInstance(v.numpy(), np.ndarray)
        self.assertAllClose(v.numpy(), np.array([1.0, 2.0, 3.0]))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Tests for MirroredVariable under tf backend",
    )
    def test_variable_numpy_scalar(self):
        from keras.src.utils.module_utils import tensorflow as tf

        strategy = tf.distribute.MirroredStrategy(["cpu:0", "cpu:1"])
        with strategy.scope():
            v = backend.Variable(initializer=0.0)

        np_value = backend.ops.convert_to_numpy(v)
        self.assertIsInstance(np_value, np.ndarray)
        self.assertAllClose(np_value, 0.0)

    def test_variable_value(self):
        """Test retrieving the value of a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v.value, np.array([1.0, 2.0, 3.0]))

    def test_variable_assign(self):
        """Test assigning a new value to a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v.assign(np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v.value, np.array([4.0, 5.0, 6.0]))

    def test_variable_assign_return(self):
        """Test assigning a new value and returning."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        r = v.assign(np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(r, np.array([4.0, 5.0, 6.0]))

    def test_variable_assign_add(self):
        """Test the assign_add method on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v.assign_add(np.array([1.0, 1.0, 1.0]))
        self.assertAllClose(v.value, np.array([2.0, 3.0, 4.0]))

    def test_variable_assign_add_return(self):
        """Test assign_add a new value and returning."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        r = v.assign_add(np.array([1.0, 1.0, 1.0]))
        self.assertAllClose(r, np.array([2.0, 3.0, 4.0]))

    def test_variable_assign_sub(self):
        """Test the assign_sub method on a variable."""
        v = backend.Variable(initializer=np.array([2.0, 3.0, 4.0]))
        v.assign_sub(np.array([1.0, 1.0, 1.0]))
        self.assertAllClose(v.value, np.array([1.0, 2.0, 3.0]))

    def test_variable_assign_sub_return(self):
        """Test assign_sub a new value and returning."""
        v = backend.Variable(initializer=np.array([2.0, 3.0, 4.0]))
        r = v.assign_sub(np.array([1.0, 1.0, 1.0]))
        self.assertAllClose(r, np.array([1.0, 2.0, 3.0]))

    def test_deferred_initialize_within_stateless_scope(self):
        """Test deferred init within a stateless scope."""
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            with self.assertRaisesRegex(
                ValueError,
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed.",
            ):
                v._deferred_initialize()


class VariableDtypeShapeNdimRepr(test_case.TestCase):
    """tests for dtype, shape, ndim, __repr__"""

    def test_variable_dtype(self):
        """Test retrieving the dtype of a variable."""
        v = backend.Variable(
            initializer=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        self.assertEqual(v.dtype, "float32")

    def test_variable_shape(self):
        """Test retrieving the shape of a variable."""
        v = backend.Variable(initializer=np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(v.shape, (2, 2))

    def test_variable_ndim(self):
        """Test retrieving the number of dimensions of a variable."""
        v = backend.Variable(initializer=np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(v.ndim, 2)

    def test_variable_repr(self):
        """Test the string representation of a variable."""
        v = backend.Variable(
            initializer=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            name="test_var",
        )
        expected_repr = (
            "<Variable path=test_var, shape=(3,), dtype=float32, "
            "value=[1. 2. 3.]>"
        )
        self.assertEqual(repr(v), expected_repr)

        # Test with `backend.StatelessScope()`
        with backend.StatelessScope():
            v = backend.Variable(
                initializer="zeros", shape=(3,), name="test_var"
            )
            expected_repr = (
                "<Variable path=test_var, shape=(3,), dtype=float32>"
            )
            self.assertEqual(repr(v), expected_repr)

    def test_variable_getitem(self):
        """Test getting an item from a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertEqual(v[0], 1)

    def test_variable_initialize(self):
        """Test initializing a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        init_value = np.array([4.0, 5.0, 6.0])
        v._initialize(value=init_value)
        self.assertAllClose(v.value, init_value)

    def test_variable_convert_to_tensor(self):
        """Test converting a variable to a tensor."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(
            v._convert_to_tensor(v.value), np.array([1.0, 2.0, 3.0])
        )

    def test_variable_convert_to_tensor_with_dtype(self):
        """Test converting a variable to a tensor with a dtype."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(
            v._convert_to_tensor(v.value, dtype="float32"),
            np.array([1.0, 2.0, 3.0]),
        )

    def test_variable_array(self):
        """Test converting a variable to an array."""
        v = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v.__array__(), np.array([1.0, 2.0, 3.0]))


class VariableOpsCorrectnessTest(test_case.TestCase):
    """Tests for operations on Variable."""

    def test_int(self):
        v = backend.Variable(initializer=np.array(-1.1))
        self.assertAllClose(int(v), np.array(-1))

    def test_float(self):
        v = backend.Variable(initializer=np.array(-1.1))
        self.assertAllClose(float(v), np.array(-1.1))

    def test__neg__(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1.0, 2.0]), trainable=False)
        self.assertAllClose(v.__neg__(), np.array([1.0, -2.0]))

    def test__abs__(self):
        """Test absolute value on a variable."""
        v = backend.Variable(initializer=np.array([-1.0, 2.0]), trainable=False)
        self.assertAllClose(v.__abs__(), np.array([1.0, 2.0]))

    def test__invert__(self):
        """Test bitwise not on a variable."""
        v = backend.Variable(
            initializer=np.array([True, False]), trainable=False, dtype="bool"
        )
        self.assertAllClose(v.__invert__(), np.array([False, True]))

    def test__eq__(self):
        """Test equality comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__eq__(np.array([1.0, 2.0])), np.array([True, True])
        )

    def test__ne__(self):
        """Test inequality comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__ne__(np.array([1.0, 2.0])), np.array([False, False])
        )

    def test__lt__(self):
        """Test less than comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__lt__(np.array([1.0, 2.0])), np.array([False, False])
        )

    def test__le__(self):
        """Test less than or equal to comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__le__(np.array([1.0, 2.0])), np.array([True, True])
        )

    def test__gt__(self):
        """Test greater than comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__gt__(np.array([1.0, 2.0])), np.array([False, False])
        )

    def test__ge__(self):
        """Test greater than or equal to comparison on a variable."""
        v = backend.Variable(initializer=np.array([1.0, 2.0]), trainable=False)
        self.assertAllClose(
            v.__ge__(np.array([1.0, 2.0])), np.array([True, True])
        )

    def test__add__(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__add__(v2), np.array([5.0, 7.0, 9.0]))

    def test__radd__(self):
        """Test reverse addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__radd__(v2), np.array([5.0, 7.0, 9.0]))

    def test__sub__(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__sub__(v2), np.array([-3.0, -3.0, -3.0]))

    def test__rsub__(self):
        """Test reverse subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        v2 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v1.__rsub__(v2), np.array([-3.0, -3.0, -3.0]))

    def test__mul__(self):
        """Test multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__mul__(v2), np.array([4.0, 10.0, 18.0]))

    def test__rmul__(self):
        """Test reverse multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__rmul__(v2), np.array([4.0, 10.0, 18.0]))

    def test__truediv__(self):
        """Test true division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__truediv__(v2), np.array([0.25, 0.4, 0.5]))

    def test__rtruediv__(self):
        """Test reverse true division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        v2 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v1.__rtruediv__(v2), np.array([0.25, 0.4, 0.5]))

    def test__floordiv__(self):
        """Test floordiv operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([-4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__floordiv__(v2), np.array([-1.0, 0.0, 0.0]))

    def test__rfloordiv__(self):
        """Test reverse floordiv operation on a variable."""
        v1 = backend.Variable(initializer=np.array([-4.0, 5.0, 6.0]))
        v2 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v1.__rfloordiv__(v2), np.array([-1.0, 0.0, 0.0]))

    def test__mod__(self):
        """Test mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([-4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__mod__(v2), np.array([-3.0, 2.0, 3.0]))

    def test__rmod__(self):
        """Test reverse mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v1.__rmod__(v2), np.array([0.0, 0.0, 0.0]))

    def test__pow__(self):
        """Test pow operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([-4.0, 5.0, 6.0]))
        self.assertAllClose(v1.__pow__(v2), np.array([1.0, 32.0, 729.0]))

    def test__rpow__(self):
        """Test reverse power operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(v1.__rpow__(v2), np.array([1.0, 4.0, 27.0]))

    def test__matmul__(self):
        """Test matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1.0, 2.0], [3.0, 4.0]]))
        v2 = backend.Variable(initializer=np.array([[5.0, 6.0], [7.0, 8.0]]))
        self.assertAllClose(
            v1.__matmul__(v2), np.array([[19.0, 22.0], [43.0, 50.0]])
        )

    def test__rmatmul__(self):
        """Test reverse matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1.0, 2.0], [3.0, 4.0]]))
        v2 = backend.Variable(initializer=np.array([[5.0, 6.0], [7.0, 8.0]]))
        self.assertAllClose(
            v1.__rmatmul__(v2), np.array([[23.0, 34.0], [31.0, 46.0]])
        )

    def test__and__(self):
        """Test bitwise and operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__and__(v2), np.array([True, False]))

    def test__rand__(self):
        """Test reverse bitwise and operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__rand__(v2), np.array([True, False]))

    def test__or__(self):
        """Test bitwise or operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__or__(v2), np.array([True, True]))

    def test__ror__(self):
        """Test reverse bitwise or operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__ror__(v2), np.array([True, True]))

    def test__xor__(self):
        """Test bitwise xor operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__xor__(v2), np.array([False, True]))

    def test__rxor__(self):
        """Test reverse bitwise xor operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__rxor__(v2), np.array([False, True]))

    def test__pos__(self):
        """Test unary plus on a variable."""
        v = backend.Variable(initializer=np.array([-1.0, 2.0]), trainable=False)
        self.assertAllClose(v.__pos__(), np.array([-1.0, 2.0]))

    def test_variable_pow(self):
        """Test pow operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        result = v1**v2
        self.assertAllClose(result, np.array([1.0, 32.0, 729.0]))

    def test_variable_rpow(self):
        """Test reverse power operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1.0, 2.0, 3.0]))
        v2 = backend.Variable(initializer=np.array([4.0, 5.0, 6.0]))
        result = v2**v1
        self.assertAllClose(result, np.array([4.0, 25.0, 216.0]))

    def test_round(self):
        v = backend.Variable(initializer=np.array([1.1, 2.2, 3.3]))
        self.assertAllClose(round(v), np.array([1.0, 2.0, 3.0]))


class VariableOpsBehaviorTest(test_case.TestCase):
    def test_invalid_bool(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            TypeError, "A Keras Variable cannot be used as a boolean."
        ):
            bool(v)

    def test_invalid_int(self):
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            TypeError, "Only scalar arrays can be converted to Python scalars."
        ):
            int(v)

    def test_invalid_float(self):
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            TypeError, "Only scalar arrays can be converted to Python scalars."
        ):
            float(v)


class VariableOpsDTypeTest(test_case.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x
        not in (
            "string",
            "complex128",
            # Remove 64-bit dtypes.
            "float64",
            "uint64",
            "int64",
        )
        + dtypes.FLOAT8_TYPES  # Remove float8 dtypes for the following tests
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]
    COMPLEX_DTYPES = ["complex32", "complex64"]
    if backend.backend() == "torch":
        ALL_DTYPES = [
            x for x in ALL_DTYPES if x not in ("uint16", "uint32", "complex64")
        ]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]
    elif backend.backend() == "tensorflow":
        # TODO(hongyu): Re-enable uint32 tests once we determine how to handle
        # dtypes.result_type(uint32, int*) -> int64 promotion.
        # Since TF variables require int64 to be placed on the GPU, we
        # exclusively enable the int64 dtype for TF. However, JAX does not
        # natively support int64, which prevents us from comparing the dtypes.
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint32",)]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint32",)]
    elif not backend.SUPPORTS_COMPLEX_DTYPES:
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("complex64",)]
    NON_COMPLEX_DTYPES = [
        x for x in ALL_DTYPES if x and x not in ["complex32", "complex64"]
    ]

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_eq(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.equal(x1_jax, x2_jax).dtype)

        self.assertDType(x1 == x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_ne(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.not_equal(x1_jax, x2_jax).dtype)

        self.assertDType(x1 != x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_lt(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.less(x1_jax, x2_jax).dtype)

        self.assertDType(x1 < x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_le(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.less_equal(x1_jax, x2_jax).dtype)

        self.assertDType(x1 <= x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_gt(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.greater(x1_jax, x2_jax).dtype)

        self.assertDType(x1 > x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_ge(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.greater_equal(x1_jax, x2_jax).dtype
        )

        self.assertDType(x1 >= x2, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_add(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.add(x1_jax, x2_jax).dtype)

        self.assertDType(x1 + x2, expected_dtype)
        self.assertDType(x1.__radd__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_sub(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.add(x1_jax, x2_jax).dtype)

        self.assertDType(x1 - x2, expected_dtype)
        self.assertDType(x1.__rsub__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_mul(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.add(x1_jax, x2_jax).dtype)

        self.assertDType(x1 * x2, expected_dtype)
        self.assertDType(x1.__rmul__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_truediv(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.true_divide(x1_jax, x2_jax).dtype
        )

        self.assertDType(x1 / x2, expected_dtype)
        self.assertDType(x1.__rtruediv__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_floordiv(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.floor_divide(x1_jax, x2_jax).dtype
        )

        self.assertDType(x1 // x2, expected_dtype)
        self.assertDType(x1.__rfloordiv__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(NON_COMPLEX_DTYPES, 2))
    )
    def test_mod(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.mod(x1_jax, x2_jax).dtype)

        self.assertDType(x1 % x2, expected_dtype)
        self.assertDType(x1.__rmod__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_pow(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.power(x1_jax, x2_jax).dtype)

        self.assertDType(x1**x2, expected_dtype)
        self.assertDType(x1.__rpow__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_matmul(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.matmul(x1_jax, x2_jax).dtype)

        self.assertDType(x1 @ x2, expected_dtype)
        self.assertDType(x1.__rmatmul__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_and(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.logical_and(x1_jax, x2_jax).dtype
        )

        self.assertDType(x1 & x2, expected_dtype)
        self.assertDType(x1.__rand__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_or(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.logical_or(x1_jax, x2_jax).dtype)

        self.assertDType(x1 | x2, expected_dtype)
        self.assertDType(x1.__ror__(x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_xor(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.logical_xor(x1_jax, x2_jax).dtype
        )

        self.assertDType(x1 ^ x2, expected_dtype)
        self.assertDType(x1.__rxor__(x2), expected_dtype)


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for standardize_shape with Torch backend",
)
class TestStandardizeShapeWithTorch(test_case.TestCase):
    def test_standardize_shape_with_torch_size(self):
        import torch

        tensor = torch.randn(3, 4, 5)
        shape = tensor.size()
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        self.assertIs(type(standardized_shape), tuple)
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_torch_symint(self):
        """Test that torch.SymInt dimensions are converted to None.

        This validates the fix for GitHub issue #22102 where torch.SymInt
        objects from torch.export were causing "Constraints violated" errors.
        """
        import torch

        # Create a mock SymInt object
        sym_int = create_autospec(torch.SymInt, instance=True)
        shape_with_sym_int = (sym_int, 224, 224, 64)

        # SymInt should be converted to None
        result = standardize_shape(shape_with_sym_int)
        self.assertEqual(result, (None, 224, 224, 64))

        # Test with multiple SymInts
        sym_int2 = create_autospec(torch.SymInt, instance=True)
        shape_with_multiple_sym_ints = (sym_int, sym_int2, 64)
        result = standardize_shape(shape_with_multiple_sym_ints)
        self.assertEqual(result, (None, None, 64))


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Tests for standardize_shape with TensorFlow backend",
)
class TestStandardizeShapeWithTensorflow(test_case.TestCase):
    def test_standardize_shape_with_tensor_size(self):
        import tensorflow as tf

        shape = (3, tf.constant(4, dtype=tf.int64), 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))
        self.assertIs(type(standardized_shape), tuple)
        for d in standardized_shape:
            self.assertIsInstance(d, int)

    def test_standardize_shape_with_unknown_rank(self):
        import tensorflow as tf

        with self.assertRaisesRegex(
            ValueError, "unknown rank.*tf.ensure_shape"
        ):
            standardize_shape(tf.TensorShape(None))

    def test_standardize_shape_with_partially_known_shape(self):
        import tensorflow as tf

        self.assertEqual(
            standardize_shape(tf.TensorShape([None, 3])), (None, 3)
        )
