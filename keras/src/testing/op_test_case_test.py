import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops.operation import Operation
from keras.src.testing import op_test_case
from keras.src.testing.op_test_case import X
from keras.src.testing.op_test_case import named_dtype_combinations


class Square(Operation):
    """Squares `x`. The subclasses below break one of its properties."""

    def call(self, x):
        return ops.square(x)


class WrongRank(Square):
    def compute_output_spec(self, x):
        return KerasTensor(x.shape + (1,), dtype=x.dtype)


class WrongDtype(Square):
    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="float32")


class NeedsStaticShape(Square):
    def compute_output_spec(self, x):
        if None in x.shape:
            raise ValueError("Needs a static shape")
        return KerasTensor(x.shape, dtype=x.dtype)


class TupleOfOne(Square):
    """Returns a tuple of one tensor eagerly, but one tensor symbolically."""

    def call(self, x):
        return (ops.square(x),)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def function_form(op):
    """Wraps an `Operation` instance as a function, like `knp.square`."""
    return lambda *args, **kwargs: op(*args, **kwargs)


def methods_of(cls):
    return sorted(n for n in dir(cls) if n.startswith("test"))


def outcomes(cls):
    """Runs the tests of `cls`: `{name: passed / skipped: … / failed: …}`."""
    tests = list(unittest.TestLoader().loadTestsFromTestCase(cls))
    result = unittest.TestResult()
    unittest.TestSuite(tests).run(result)
    out = {test._testMethodName: "passed" for test in tests}
    for test, traceback in result.failures + result.errors:
        error = traceback.strip().rsplit("\n", 1)[-1]
        out[test._testMethodName] = f"failed: {error}"
    for test, reason in result.skipped:
        out[test._testMethodName] = f"skipped: {reason}"
    return out


class SquareTest(testing.UnaryOpTest):
    op, op_class = ops.square, Square
    reference_fn, jax_fn = np.square, jnp.square


class OpTestTest(testing.TestCase):
    def test_inherited_tests(self):
        names = methods_of(SquareTest)
        self.assertEqual(
            [n for n in names if "input_dtypes" not in n],
            [
                "test_eager",
                "test_eager_class",
                "test_symbolic_dynamic",
                "test_symbolic_static",
            ],
        )
        self.assertLen([n for n in names if "input_dtypes" in n], 10)
        failed = {
            k: v for k, v in outcomes(SquareTest).items() if "failed" in v
        }
        self.assertEqual(failed, {})

        class Add(testing.BinaryOpTest):
            op, reference_fn, jax_fn = ops.add, np.add, jnp.add

        names = methods_of(Add)
        self.assertLen(
            [n for n in names if "input_dtypes" in n],
            len(named_dtype_combinations(2)),
        )
        self.assertIn("test_python_types_int8", names)

        class Sum(testing.ReductionOpTest):
            op, reference_fn, jax_fn = ops.sum, np.sum, jnp.sum

        self.assertLen(Sum.cases, 10)
        self.assertEqual(outcomes(Sum)["test_eager"], "passed")

    def test_abstract_classes_are_not_collected(self):
        for cls in (
            testing.OpTest,
            testing.UnaryOpTest,
            testing.BinaryOpTest,
            testing.ReductionOpTest,
        ):
            self.assertFalse(cls.__test__)
        self.assertTrue(SquareTest.__test__)

    def test_override_by_name(self):
        class Case(SquareTest):
            def test_input_dtypes_int8(self):
                self.skipTest("overridden")

        out = outcomes(Case)
        self.assertEqual(out["test_input_dtypes_int8"], "skipped: overridden")
        self.assertEqual(out["test_input_dtypes_uint8"], "passed")

    def test_functions_are_unbound(self):
        case = SquareTest("test_eager")
        self.assertAllClose(case.op(X), case.reference_fn(X))
        self.assertAllClose(case.jax_fn(X), np.square(X))

    def test_no_op_class_and_no_arrays(self):
        class Ones(testing.OpTest):
            op, reference_fn = ops.ones, np.ones
            inputs = ((2, 3),)

        self.assertEqual(
            outcomes(Ones),
            {
                "test_eager": "passed",
                "test_eager_class": "skipped: No `op_class`",
                "test_symbolic_static": (
                    "skipped: No array inputs, so no symbolic path"
                ),
                "test_symbolic_dynamic": (
                    "skipped: No array inputs, so no symbolic path"
                ),
            },
        )

    def test_disabled_dynamic_checks_are_skipped(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(NeedsStaticShape()), np.square
            inputs = (X,)
            run_dynamic_shape_checks = False

        out = outcomes(Case)
        self.assertEqual(out["test_symbolic_static"], "passed")
        self.assertEqual(
            out["test_symbolic_dynamic"], "skipped: Disabled by the test"
        )

    def test_wrong_symbolic_rank_fails(self):
        class Case(testing.OpTest):
            op, op_class = function_form(WrongRank()), WrongRank
            reference_fn = np.square
            inputs = (X,)

        out = outcomes(Case)
        self.assertEqual(out["test_eager"], "passed")
        self.assertEqual(out["test_eager_class"], "passed")
        self.assertRegex(
            out["test_symbolic_static"],
            r"^failed: AssertionError: Output differs: "
            r"symbolic=int\d\d\[2, 3, 1\], eager=int\d\d\[2, 3\] "
            r"for inputs int32\[2, 3\], kwargs \{\}$",
        )
        # The failing symbolic shapes are added to the error.
        self.assertEqual(
            out["test_symbolic_dynamic"],
            "failed: Symbolic input shapes: [(None, None)]",
        )

    def test_unknown_dims_failure_names_the_shapes(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(NeedsStaticShape()), np.square
            cases = {"a": ((X,), {}), "b": ((X[0],), {})}
            run_dynamic_shape_checks = True

        out = outcomes(Case)
        self.assertEqual(out["test_symbolic_static"], "passed")
        self.assertEqual(
            out["test_symbolic_dynamic"],
            "failed: Symbolic input shapes: [(None, None)]",
        )

    def test_unknown_dim_shapes(self):
        shapes = [(2, 3), (2,)]
        self.assertEqual(
            op_test_case._all_dims_unknown(shapes), [(None, None), (None,)]
        )
        self.assertEqual(
            op_test_case._one_dim_unknown(shapes),
            [
                [(None, 3), (2,)],
                [(2, None), (2,)],
                [(2, 3), (None,)],
            ],
        )

    def test_structure_mismatch_fails(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(TupleOfOne()), np.square
            inputs = (X,)

        out = outcomes(Case)
        self.assertRegex(out["test_eager"], "^failed: ")  # Tuple vs array.
        self.assertRegex(out["test_symbolic_static"], "^failed: ")

    def test_symbolic_dtype_mismatch_fails(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(WrongDtype()), np.square
            inputs = (X,)

        self.assertRegex(
            outcomes(Case)["test_symbolic_static"],
            "^failed: AssertionError: Output differs: symbolic=float32",
        )

    def test_unknown_output_dims_are_accepted(self):
        case = SquareTest("test_eager")
        case.check_symbolic((X,), {}, [(None, 3)])


class CheckInputDtypesTest(testing.TestCase):
    def test_dtype_mismatch_vs_jax_fails(self):
        class Case(testing.UnaryOpTest):
            op = lambda x: ops.cast(ops.square(x), "int32")  # noqa: E731
            reference_fn, jax_fn = np.square, jnp.square

        self.assertRegex(
            outcomes(Case)["test_input_dtypes_uint8"],
            r"Output dtype differs from JAX for inputs uint8\[2, 3\]",
        )

    def test_skipped_dtype_has_reason(self):
        skipped = {backend.backend(): {"int8": "no int8 here"}}
        with mock.patch.dict(op_test_case._SKIPPED_DTYPES, skipped):
            self.assertEqual(
                outcomes(SquareTest)["test_input_dtypes_int8"],
                "skipped: int8: no int8 here",
            )

    def test_dtype_count_mismatch_raises(self):
        case = SquareTest("test_eager")
        with self.assertRaises(ValueError):
            case.check_input_dtypes(("int8",), (X, X), {})

    def test_python_scalar_input(self):
        class Add(testing.BinaryOpTest):
            op, reference_fn, jax_fn = ops.add, np.add, jnp.add

        self.assertEqual(outcomes(Add)["test_python_types_int8"], "passed")


class NamedDtypeCombinationsTest(testing.TestCase):
    POOL = op_test_case.DEFAULT_INPUT_DTYPES

    def test_shared_pool_counts(self):
        self.assertLen(named_dtype_combinations(1, dtypes=self.POOL), 10)
        self.assertLen(named_dtype_combinations(2, dtypes=self.POOL), 55)
        self.assertLen(
            named_dtype_combinations(2, ordered=True, dtypes=self.POOL), 100
        )

    def test_names_and_values(self):
        cases = named_dtype_combinations(2, dtypes=self.POOL)
        names = [name for name, _ in cases]
        self.assertLen(set(names), len(names))
        self.assertIn("int8_uint32", names)
        self.assertNotIn("uint32_int8", names)
        self.assertIn(("int8_int8", ("int8", "int8")), cases)

    def test_per_input_pools(self):
        pools = [self.POOL, self.POOL[:3]]
        cases = named_dtype_combinations(2, dtypes=pools)
        self.assertLen(cases, 30)
        self.assertEqual(cases, named_dtype_combinations(2, True, pools))
        for _, dtypes in cases:
            self.assertIn(dtypes[1], self.POOL[:3])

    def test_per_input_pools_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Expected 2 dtype pools"):
            named_dtype_combinations(2, dtypes=[self.POOL] * 3)

    def test_binary_pool_default(self):
        is_torch = backend.backend() == "torch"
        self.assertEqual(
            "float64" in op_test_case.BINARY_INPUT_DTYPES, is_torch
        )
        self.assertLen(named_dtype_combinations(1), 10)
        self.assertLen(named_dtype_combinations(2), 66 if is_torch else 55)
