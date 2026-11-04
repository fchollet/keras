from unittest.mock import patch

import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product
from keras.src.testing.test_utils import x64_mode_scope

X64_SUPPORTED = backend.backend() in config._X64_BACKENDS


class DtypesTest(test_case.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x
        not in (
            "string",
            "complex128",
            "float64",
            "uint64",
            "int64",
        )
        + dtypes.FLOAT8_TYPES  # Remove float8 dtypes for the following tests
    ] + [None]
    if backend.backend() == "torch":
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint16", "uint32")]
    elif backend.backend() == "tensorflow":
        # TODO(hongyu): Re-enable uint32 tests once we determine how to handle
        # dtypes.result_type(uint32, int*) -> int64 promotion.
        # Since TF variables require int64 to be placed on the GPU, we
        # exclusively enable the int64 dtype for TF. However, JAX does not
        # natively support int64, which prevents us from comparing the dtypes.
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint32",)]
    if not backend.SUPPORTS_COMPLEX_DTYPES:
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("complex64",)]

    # The 64-bit dtypes, kept under `x64_mode="explicit_x64"`.
    X64_DTYPES = ["int64", "float64"]
    if backend.backend() != "torch":  # No uint64 kernels in torch.
        X64_DTYPES += ["uint64"]
    if backend.SUPPORTS_COMPLEX_DTYPES:
        X64_DTYPES += ["complex128"]
    EXPLICIT_X64_DTYPES = ALL_DTYPES + X64_DTYPES

    def assert_result_type_matches_jax(self, *dtypes):
        """Checks `result_type` against `jnp.result_type` for tensor inputs."""
        import jax.numpy as jnp

        tensors = [ops.ones((1,), dtype=dtype) for dtype in dtypes]
        jax_tensors = [jnp.ones((1,), dtype=dtype) for dtype in dtypes]
        out = backend.result_type(*(x.dtype for x in tensors))
        self.assertEqual(out, jnp.result_type(*jax_tensors).name)

    def assert_scalar_result_type_matches_jax(self, dtype, scalar_type):
        """Checks `result_type` against `jnp.result_type` for a scalar type."""
        import jax.numpy as jnp

        out = backend.result_type(dtype, scalar_type)
        self.assertEqual(out, jnp.result_type(dtype, scalar_type).name)

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=[bool, int, float])
    )
    def test_result_type_with_python_scalar_types(self, dtype1, dtype2):
        self.assert_scalar_result_type_matches_jax(dtype1, dtype2)

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    )
    def test_result_type_with_tensor(self, dtype1, dtype2):
        self.assert_result_type_matches_jax(dtype1, dtype2)

    @parameterized.named_parameters(
        named_product(dtype1=EXPLICIT_X64_DTYPES, dtype2=[bool, int, float])
    )
    @pytest.mark.skipif(not X64_SUPPORTED, reason="Unsupported backend.")
    def test_result_type_with_python_scalar_types_explicit_x64(
        self, dtype1, dtype2
    ):
        with x64_mode_scope("explicit_x64"):
            self.assert_scalar_result_type_matches_jax(dtype1, dtype2)

    @parameterized.named_parameters(
        named_product(dtype1=EXPLICIT_X64_DTYPES, dtype2=EXPLICIT_X64_DTYPES)
    )
    @pytest.mark.skipif(not X64_SUPPORTED, reason="Unsupported backend.")
    def test_result_type_with_tensor_explicit_x64(self, dtype1, dtype2):
        with x64_mode_scope("explicit_x64"):
            self.assert_result_type_matches_jax(dtype1, dtype2)

    @parameterized.named_parameters(
        named_product(
            dtype=[
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
            ]
        )
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TensorFlow only"
    )
    def test_result_type_with_int64(self, dtype):
        # https://github.com/keras-team/keras/issues/21677
        x1 = ops.ones((1,), dtype="int64")
        x2 = ops.ones((1,), dtype=dtype)
        out = backend.result_type(x1.dtype, x2.dtype)
        self.assertEqual(out, "int64")

    @parameterized.named_parameters(
        named_product(
            dtype=[
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
            ]
        )
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TensorFlow only"
    )
    def test_result_type_with_float64(self, dtype):
        # Float types have a similar issue as int64 in TF.:
        # https://github.com/keras-team/keras/issues/21677
        x1 = ops.ones((1,), dtype="float64")
        x2 = ops.ones((1,), dtype=dtype)
        out = backend.result_type(x1.dtype, x2.dtype)
        self.assertEqual(out, "float64")

    def test_result_type_with_none(self):
        import jax.numpy as jnp

        self.assertEqual(backend.result_type(None), jnp.result_type(None).name)

    def test_result_type_empty_list(self):
        self.assertEqual(backend.result_type(), "float32")

    def test_respect_weak_type_for_bool(self):
        self.assertEqual(dtypes._respect_weak_type("bool", True), "bool")

    def test_respect_weak_type_for_int(self):
        self.assertEqual(dtypes._respect_weak_type("int32", True), "int")

    def test_respect_weak_type_for_float(self):
        self.assertEqual(dtypes._respect_weak_type("float32", True), "float")

    def test_resolve_weak_type_for_bfloat16(self):
        self.assertEqual(dtypes._resolve_weak_type("bfloat16"), "float32")

    def test_resolve_weak_type_for_bfloat16_with_precision(self):
        self.assertEqual(
            dtypes._resolve_weak_type("bfloat16", precision="64"), "float64"
        )

    def test_respect_weak_type_for_complex64(self):
        self.assertEqual(
            dtypes._respect_weak_type("complex64", True), "complex"
        )

    def test_respect_weak_type_for_complex128(self):
        self.assertEqual(
            dtypes._respect_weak_type("complex128", True), "complex"
        )

    def test_invalid_dtype_for_keras_promotion(self):
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion."
        ):
            dtypes._least_upper_bound("invalid_dtype")

    def test_resolve_weak_type_for_invalid_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `dtype`. Expected one of"
        ):
            dtypes._resolve_weak_type("invalid_dtype")

    def test_resolve_weak_type_for_invalid_precision(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument `precision`. Expected one of",
        ):
            dtypes._resolve_weak_type("int32", precision="invalid_precision")

    def test_cycle_detection_in_make_lattice_upper_bounds(self):
        original_lattice_function = dtypes._type_promotion_lattice

        def mock_lattice():
            lattice = original_lattice_function()
            lattice["int32"].append("float32")
            lattice["float32"].append("int32")
            return lattice

        dtypes._type_promotion_lattice = mock_lattice

        with self.assertRaisesRegex(
            ValueError, "cycle detected in type promotion lattice for node"
        ):
            dtypes._make_lattice_upper_bounds()

        dtypes._type_promotion_lattice = original_lattice_function

    def test_respect_weak_type_for_invalid_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `dtype`. Expected one of"
        ):
            dtypes._respect_weak_type("invalid_dtype", True)

    def test_invalid_dtype_in_least_upper_bound(self):
        invalid_dtype = "non_existent_dtype"
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion"
        ):
            dtypes._least_upper_bound(invalid_dtype)

    def test_empty_lub_in_least_upper_bound(self):
        dtype1 = "float32"
        dtype2 = "int32"
        dtypes._least_upper_bound.cache_clear()
        with patch.dict(
            dtypes.LATTICE_UPPER_BOUNDS,
            {"float32": set(), "int32": set()},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, "no available implicit dtype promotion path"
            ):
                dtypes._least_upper_bound(dtype1, dtype2)
        dtypes._least_upper_bound.cache_clear()

    def test_valid_dtype_leading_to_single_lub_element(self):
        self.assertEqual(
            dtypes._least_upper_bound("float32", "int32"), "float32"
        )

    def test_valid_dtype_leading_to_keyerror_and_valueerror(self):
        invalid_dtype = "non_existent_dtype"
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion"
        ):
            dtypes._least_upper_bound(invalid_dtype)

    def test_resolve_weak_type_bool(self):
        self.assertEqual(dtypes._resolve_weak_type("bool"), "bool")

    def test_resolve_weak_type_int(self):
        self.assertEqual(
            dtypes._resolve_weak_type("int32", precision="32"), "int32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("int64", precision="64"), "int64"
        )

    def test_resolve_weak_type_uint(self):
        self.assertEqual(
            dtypes._resolve_weak_type("uint32", precision="32"), "uint32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("uint64", precision="64"), "uint64"
        )

    def test_resolve_weak_type_float(self):
        self.assertEqual(
            dtypes._resolve_weak_type("float32", precision="32"), "float32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("float64", precision="64"), "float64"
        )

    def test_least_upper_bound_ensure_order_independence(self):
        # Test to ensure _least_upper_bound is order-independent.
        result1 = dtypes._least_upper_bound("float32", "int32")
        result2 = dtypes._least_upper_bound("int32", "float32")
        self.assertEqual(result1, result2)

    def test_least_upper_bound_single_element(self):
        dtypes.LATTICE_UPPER_BOUNDS["test_dtype"] = {"test_dtype"}
        self.assertEqual(dtypes._least_upper_bound("test_dtype"), "test_dtype")

    def test_least_upper_bound_no_element(self):
        dtypes.LATTICE_UPPER_BOUNDS["test_dtype"] = set()
        with self.assertRaisesRegex(
            ValueError, "no available implicit dtype promotion path"
        ):
            dtypes._least_upper_bound("test_dtype")

    def test_least_upper_bound_with_no_common_upper_bound(self):
        with patch.dict(
            dtypes.LATTICE_UPPER_BOUNDS,
            {"test_dtype1": set(), "test_dtype2": set()},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, "no available implicit dtype promotion path"
            ):
                dtypes._least_upper_bound("test_dtype1", "test_dtype2")

    def test_invalid_float8_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "There is no implicit conversions from float8 dtypes"
        ):
            dtypes.result_type("float8_e4m3fn", "bfloat16")
        with self.assertRaisesRegex(
            ValueError, "There is no implicit conversions from float8 dtypes"
        ):
            dtypes.result_type("float8_e5m2", "bfloat16")
