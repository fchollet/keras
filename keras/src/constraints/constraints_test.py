import numpy as np

from keras.src import backend
from keras.src import constraints
from keras.src import testing


def get_example_array():
    np.random.seed(3537)
    example_array = np.random.random((100, 100)) * 100.0 - 50.0
    example_array[0, 0] = 0.0
    return example_array


class ConstraintsTest(testing.TestCase):
    def test_max_norm(self):
        constraint_fn = constraints.MaxNorm(2.0)
        x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
        target = np.array(
            [
                [0, 0, 0],
                [1.0, 0, 0],
                [2.0, 0, 0],
                [2.0 / np.sqrt(3), 2.0 / np.sqrt(3), 2.0 / np.sqrt(3)],
            ]
        ).T
        output = constraint_fn(x)
        self.assertAllClose(output, target)

    def test_non_neg(self):
        constraint_fn = constraints.NonNeg()
        output = constraint_fn(get_example_array())
        output = backend.ops.convert_to_numpy(output)
        self.assertTrue((np.min(output, axis=1) >= 0.0).all())

    def test_unit_norm(self):
        constraint_fn = constraints.UnitNorm()
        output = constraint_fn(get_example_array())
        output = backend.ops.convert_to_numpy(output)
        l2 = np.sqrt(np.sum(np.square(output), axis=0))
        self.assertAllClose(l2, np.ones((100,)))

    def test_min_max_norm(self):
        constraint_fn = constraints.MinMaxNorm(min_value=0.2, max_value=0.5)
        output = constraint_fn(get_example_array())
        output = backend.ops.convert_to_numpy(output)
        l2 = np.sqrt(np.sum(np.square(output), axis=0))
        self.assertTrue(np.all(l2 >= 0.2))
        self.assertTrue(np.all(l2 <= 0.5 + 1e-6))

    def test_get_method(self):
        obj = constraints.get("unit_norm")
        self.assertIsInstance(obj, constraints.UnitNorm)

        obj = constraints.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            constraints.get("typo")

    def test_default_constraint_call(self):
        constraint_fn = constraints.Constraint()
        x = np.array([1.0, 2.0, 3.0])
        output = constraint_fn(x)
        self.assertAllClose(x, output)

    def test_constraint_get_config(self):
        constraint_fn = constraints.Constraint()
        config = constraint_fn.get_config()
        self.assertEqual(config, {})

    def test_constraint_from_config(self):
        constraint_fn = constraints.Constraint()
        config = constraint_fn.get_config()
        recreated_constraint_fn = constraints.Constraint.from_config(config)
        self.assertIsInstance(recreated_constraint_fn, constraints.Constraint)

    def test_max_norm_get_config(self):
        constraint_fn = constraints.MaxNorm(max_value=3.0, axis=1)
        config = constraint_fn.get_config()
        expected_config = {"max_value": 3.0, "axis": 1}
        self.assertEqual(config, expected_config)
        restored = constraints.MaxNorm.from_config(config)
        self.assertEqual(restored.max_value, constraint_fn.max_value)
        self.assertEqual(restored.axis, constraint_fn.axis)
        x = get_example_array()
        self.assertAllClose(
            backend.ops.convert_to_numpy(constraint_fn(x)),
            backend.ops.convert_to_numpy(restored(x)),
        )

    def test_unit_norm_get_config(self):
        constraint_fn = constraints.UnitNorm(axis=1)
        config = constraint_fn.get_config()
        expected_config = {"axis": 1}
        self.assertEqual(config, expected_config)
        restored = constraints.UnitNorm.from_config(config)
        self.assertEqual(restored.axis, constraint_fn.axis)
        x = get_example_array()
        self.assertAllClose(
            backend.ops.convert_to_numpy(constraint_fn(x)),
            backend.ops.convert_to_numpy(restored(x)),
        )

    def test_min_max_norm_get_config(self):
        constraint_fn = constraints.MinMaxNorm(
            min_value=0.5, max_value=2.0, rate=0.7, axis=1
        )
        config = constraint_fn.get_config()
        expected_config = {
            "min_value": 0.5,
            "max_value": 2.0,
            "rate": 0.7,
            "axis": 1,
        }
        self.assertEqual(config, expected_config)
        restored = constraints.MinMaxNorm.from_config(config)
        self.assertEqual(restored.min_value, constraint_fn.min_value)
        self.assertEqual(restored.max_value, constraint_fn.max_value)
        self.assertEqual(restored.rate, constraint_fn.rate)
        self.assertEqual(restored.axis, constraint_fn.axis)
        x = get_example_array()
        self.assertAllClose(
            backend.ops.convert_to_numpy(constraint_fn(x)),
            backend.ops.convert_to_numpy(restored(x)),
        )

    def test_non_neg_get_config(self):
        constraint_fn = constraints.NonNeg()
        config = constraint_fn.get_config()
        self.assertEqual(config, {})

    def test_non_neg_from_config(self):
        constraint_fn = constraints.NonNeg()
        config = constraint_fn.get_config()
        recreated = constraints.NonNeg.from_config(config)
        self.assertIsInstance(recreated, constraints.NonNeg)

    def test_non_neg_serialization_roundtrip(self):
        original = constraints.NonNeg()
        config = original.get_config()
        restored = constraints.NonNeg.from_config(config)
        x = get_example_array()
        out_original = backend.ops.convert_to_numpy(original(x))
        out_restored = backend.ops.convert_to_numpy(restored(x))
        self.assertAllClose(out_original, out_restored)

    def test_non_neg_zeroes_negatives(self):
        constraint_fn = constraints.NonNeg()
        x = np.full((5, 5), -3.0)
        output = backend.ops.convert_to_numpy(constraint_fn(x))
        self.assertAllClose(output, np.zeros((5, 5)))

    def test_non_neg_preserves_positives(self):
        constraint_fn = constraints.NonNeg()
        x = np.full((5, 5), 3.0)
        output = backend.ops.convert_to_numpy(constraint_fn(x))
        self.assertAllClose(output, x)

    def test_non_neg_all_zeros_input(self):
        constraint_fn = constraints.NonNeg()
        x = np.zeros((5, 5))
        output = backend.ops.convert_to_numpy(constraint_fn(x))
        self.assertAllClose(output, np.zeros((5, 5)))
