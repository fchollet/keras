import inspect
import re
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import losses as losses_module
from keras.src import ops
from keras.src import testing
from keras.src.api_export import REGISTERED_NAMES_TO_OBJS
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import set_keras_mask
from keras.src.layers import Input
from keras.src.losses import losses
from keras.src.models import Functional


class MeanSquaredErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.MeanSquaredError(name="mymse"))

    def test_base_function_reduction(self):
        mse_fn = losses.mean_squared_error
        y_true = np.array([4, 8, 12])
        y_pred = np.array([[3], [0], [1]])
        loss = mse_fn(y_true, y_pred)
        self.assertEqual(backend.ops.shape(loss), (3,))

    def test_all_correct_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mse_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 49.5)

    def test_scalar_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 113.85)

    def test_sample_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 767.8 / 6)

    def test_timestep_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mse_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 97.833336)

    def test_zero_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_no_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [84.3333, 143.3666])

    def test_sum_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 227.69998)

    def test_mean_with_sample_weight_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction="mean_with_sample_weight")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(
            loss, (110 / 3 * 1.2 + 187 / 3 * 3.4) / (1.2 + 3.4)
        )

    def test_dtype_arg(self):
        mse_obj = losses.MeanSquaredError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsoluteError(name="myname")
        )

    def test_all_correct_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mae_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 5.5)

    def test_scalar_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 12.65)

    def test_sample_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 81.4 / 6)

    def test_timestep_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mae_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 13.833333)

    def test_zero_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_no_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [10.7333, 14.5666])

    def test_sum_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 25.29999)

    def test_mean_with_sample_weight_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction="mean_with_sample_weight")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(
            loss, (14 / 3 * 1.2 + 19 / 3 * 3.4) / (1.2 + 3.4)
        )

    def test_dtype_arg(self):
        mae_obj = losses.MeanAbsoluteError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanAbsolutePercentageErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsolutePercentageError(name="mymape")
        )

    def test_all_correct_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mape_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 211.8518, 3)

    def test_scalar_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 487.259, 3)

    def test_sample_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 422.8888, 3)

    def test_timestep_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mape_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 694.4444)

    def test_zero_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_no_reduction(self):
        mape_obj = losses.MeanAbsolutePercentageError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [621.8518, 352.6666])

    def test_mean_with_sample_weight_reduction(self):
        mape_obj = losses.MeanAbsolutePercentageError(
            reduction="mean_with_sample_weight"
        )
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 183.865)

    def test_dtype_arg(self):
        mape_obj = losses.MeanAbsolutePercentageError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanSquaredLogarithmicErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanSquaredLogarithmicError(name="mysloge")
        )

    def test_unweighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4370, 3)

    def test_scalar_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 3.3051, 3)

    def test_sample_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 3.7856, 3)

    def test_timestep_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = msle_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 2.647374)

    def test_zero_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_mean_with_sample_weight_reduction(self):
        msle_obj = losses.MeanSquaredLogarithmicError(
            reduction="mean_with_sample_weight"
        )
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.646)

    def test_dtype_arg(self):
        msle_obj = losses.MeanSquaredLogarithmicError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        self.assertDType(loss, "bfloat16")


class HingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.3, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.6, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.1, 1.5])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.Hinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.55, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.1, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.1, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.Hinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.Hinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class SquaredHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.86, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.72, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.46, 2.26])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.SquaredHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.73, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.46, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.46, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.SquaredHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.SquaredHinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.8, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.2, 1.6])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.CategoricalHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.6, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.2, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.2, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.CategoricalHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.CategoricalHinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CosineSimilarityTest(testing.TestCase):
    def l2_norm(self, x, axis):
        epsilon = 1e-12
        square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
        x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
        return np.multiply(x, x_inv_norm)

    def setup(self, axis=1):
        self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        y_true = self.l2_norm(self.np_y_true, axis)
        y_pred = self.l2_norm(self.np_y_pred, axis)
        self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

        self.y_true = self.np_y_true
        self.y_pred = self.np_y_pred

    def test_config(self):
        cosine_obj = losses.CosineSimilarity(
            axis=2, reduction="sum", name="cosine_loss"
        )
        self.assertEqual(cosine_obj.name, "cosine_loss")
        self.assertEqual(cosine_obj.reduction, "sum")
        config = cosine_obj.get_config()
        self.assertEqual(
            config, {"name": "cosine_loss", "reduction": "sum", "axis": 2}
        )
        self.run_class_serialization_test(cosine_obj)

    def test_unweighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = 2.3
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_sample_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = np.asarray([1.2, 3.4])
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        np_y_true = self.np_y_true.reshape((2, 3, 1))
        np_y_pred = self.np_y_pred.reshape((2, 3, 1))
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape((2, 3))

        y_true = self.l2_norm(np_y_true, 2)
        y_pred = self.l2_norm(np_y_pred, 2)
        expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(2,))

        y_true = np_y_true
        y_pred = np_y_pred
        loss = cosine_obj(y_true, y_pred, sample_weight=sample_weight)

        expected_loss = -np.mean(expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_axis(self):
        self.setup(axis=1)
        cosine_obj = losses.CosineSimilarity(axis=1)
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_dtype_arg(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity(dtype="bfloat16")
        loss = cosine_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class HuberLossTest(testing.TestCase):
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_pred - y_true
        abs_error = np.abs(error)

        quadratic = np.minimum(abs_error, delta)
        linear = np.subtract(abs_error, quadratic)
        return np.add(
            np.multiply(0.5, np.multiply(quadratic, quadratic)),
            np.multiply(delta, linear),
        )

    def setup(self, delta=1.0):
        self.np_y_pred = np.array([[0.9, 0.2, 0.2], [0.8, 0.4, 0.6]])
        self.np_y_true = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        self.batch_size = 6
        self.expected_losses = self.huber_loss(
            self.np_y_true, self.np_y_pred, delta
        )

        self.y_pred = self.np_y_pred
        self.y_true = self.np_y_true

    def test_config(self):
        h_obj = losses.Huber(delta=2.0, reduction="sum", name="huber")
        self.assertEqual(h_obj.name, "huber")
        self.assertEqual(h_obj.reduction, "sum")
        config = h_obj.get_config()
        self.assertEqual(
            config, {"name": "huber", "reduction": "sum", "delta": 2.0}
        )
        self.run_class_serialization_test(h_obj)

    def test_all_correct(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_pred)
        actual_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = np.array([[1.2], [3.4]])

        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        y_pred = self.np_y_pred.reshape((2, 3, 1))
        y_true = self.np_y_true.reshape((2, 3, 1))
        expected_losses = self.huber_loss(y_true, y_pred)

        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))
        loss = h_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        actual_loss = np.multiply(expected_losses, sample_weight)
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 0
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_non_default_delta(self):
        self.setup(delta=0.8)
        h_obj = losses.Huber(delta=0.8)
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_invalid_delta(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            losses.Huber(delta=0)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            losses.Huber(delta=-1.0)
        with self.assertRaisesRegex(ValueError, "Expected a float"):
            losses.Huber(delta=1)
        with self.assertRaisesRegex(ValueError, "Expected a float"):
            losses.Huber(delta=np.float32(1.0))

    def test_dtype_arg(self):
        self.setup()
        h_obj = losses.Huber(dtype="bfloat16")
        loss = h_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class LogCoshTest(testing.TestCase):
    def setup(self):
        y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        self.batch_size = 6
        error = y_pred - y_true
        self.expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)

        self.y_true = y_true
        self.y_pred = y_pred

    def test_config(self):
        logcosh_obj = losses.LogCosh(reduction="sum", name="logcosh_loss")
        self.assertEqual(logcosh_obj.name, "logcosh_loss")
        self.assertEqual(logcosh_obj.reduction, "sum")
        config = logcosh_obj.get_config()
        self.assertEqual(config, {"name": "logcosh_loss", "reduction": "sum"})

    def test_unweighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        loss = logcosh_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 2.3

        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        sample_weight = np.asarray([1.2, 3.4])
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        error = y_pred - y_true
        expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))

        loss = logcosh_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 0
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        logcosh_obj = losses.LogCosh(dtype="bfloat16")
        loss = logcosh_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class KLDivergenceTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray(
            [0.4, 0.9, 0.12, 0.36, 0.3, 0.4], dtype=np.float32
        ).reshape((2, 3))
        self.y_true = np.asarray(
            [0.5, 0.8, 0.12, 0.7, 0.43, 0.8], dtype=np.float32
        ).reshape((2, 3))

        self.batch_size = 2
        self.expected_losses = np.multiply(
            self.y_true, np.log(self.y_true / self.y_pred)
        )

    def test_config(self):
        k_obj = losses.KLDivergence(reduction="sum", name="kld")
        self.assertEqual(k_obj.name, "kld")
        self.assertEqual(k_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        k_obj = losses.KLDivergence()

        loss = k_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = 2.3

        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = np.asarray([1.2, 3.4], dtype=np.float32).reshape((2, 1))
        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray(
                [1.2, 1.2, 1.2, 3.4, 3.4, 3.4], dtype=np.float32
            ).reshape(2, 3),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3)
        expected_losses = np.sum(
            np.multiply(y_true, np.log(y_true / y_pred)), axis=-1
        )
        loss = k_obj(y_true, y_pred, sample_weight=sample_weight)

        num_timesteps = 3
        expected_loss = np.sum(expected_losses * sample_weight) / (
            self.batch_size * num_timesteps
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        loss = k_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        k_obj = losses.KLDivergence(dtype="bfloat16")
        loss = k_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class PoissonTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray([1, 9, 2, 5, 2, 6], dtype=np.float32).reshape(
            (2, 3)
        )
        self.y_true = np.asarray([4, 8, 12, 8, 1, 3], dtype=np.float32).reshape(
            (2, 3)
        )

        self.batch_size = 6
        self.expected_losses = self.y_pred - np.multiply(
            self.y_true, np.log(self.y_pred)
        )

    def test_config(self):
        poisson_obj = losses.Poisson(reduction="sum", name="poisson")
        self.assertEqual(poisson_obj.name, "poisson")
        self.assertEqual(poisson_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        loss = poisson_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        sample_weight = 2.3
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        sample_weight = np.asarray([1.2, 3.4]).reshape((2, 1))
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3, 1)
        expected_losses = y_pred - np.multiply(y_true, np.log(y_pred))

        loss = poisson_obj(
            y_true,
            y_pred,
            sample_weight=np.asarray(sample_weight).reshape((2, 3)),
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        loss = poisson_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        poisson_obj = losses.Poisson(dtype="bfloat16")
        loss = poisson_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class BinaryCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryCrossentropy(name="bce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
            ]
        )
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        y_pred = np.array(
            [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype="float32"
        )
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)
        self.assertAllClose(loss, 0.20046903)

        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.98559)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 3.3333)

    def test_scalar_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        loss = bce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 9.1668)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 7.666)

    def test_sample_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 4.7827)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        weights = np.array([4, 3])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=weights)
        self.assertAlmostEqual(loss, 10.0)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True, reduction=None)
        loss = bce_obj(y_true, logits)
        self.assertAllClose(loss, [0.0, 6.666], atol=1e-3)

    def test_invalid_label_smoothing(self):
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.binary_crossentropy(
                [[0.0, 1.0]], [[0.6, 0.4]], label_smoothing=-0.1
            )
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.BinaryCrossentropy(label_smoothing=1.1)(
                [[0.0, 1.0]], [[0.6, 0.4]]
            )
        with self.assertRaisesRegex(ValueError, "must be a float or int"):
            losses.binary_crossentropy(
                [[0.0, 1.0]], [[0.6, 0.4]], label_smoothing=True
            )
        with self.assertRaisesRegex(ValueError, "must be a float or int"):
            losses.binary_crossentropy(
                [[0.0, 1.0]], [[0.6, 0.4]], label_smoothing=np.bool_(True)
            )
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.binary_crossentropy(
                [[0.0, 1.0]], [[0.6, 0.4]], label_smoothing=np.float32(-0.1)
            )

    def test_label_smoothing(self):
        logits = np.array([[10.0, -10.0, -10.0]])
        y_true = np.array([[1, 0, 1]])
        label_smoothing = 0.1
        bce_obj = losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = bce_obj(y_true, logits)
        expected_value = (10.0 + 5.0 * label_smoothing) / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )
        cce_obj = losses.BinaryCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch doesn't support bfloat16 for BinaryCrossentropy",
    )
    def test_dtype_arg(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        y_pred = np.array(
            [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype="float32"
        )
        bce_obj = losses.BinaryCrossentropy(dtype="bfloat16")
        loss = bce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalCrossentropy(name="cce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317)

    def test_warning_for_invalid_shape(self):
        # 2D case: should warn
        y_true = np.array([[1], [1]], dtype="float32")
        y_pred = np.array([[0.5], [0.5]], dtype="float32")
        with pytest.warns(
            SyntaxWarning,
            match=r"expected y_pred.shape to be \(batch_size, num_classes\)",
        ):
            losses.categorical_crossentropy(y_true, y_pred)

        # 3D case: axis=1, classes=1, should warn
        y_true = np.ones((2, 1, 4), dtype="float32")
        y_pred = np.ones((2, 1, 4), dtype="float32")
        with pytest.warns(
            SyntaxWarning,
            match=r"expected y_pred.shape to be \(batch_size, num_classes\)",
        ):
            losses.categorical_crossentropy(y_true, y_pred, axis=1)

        # 3D case: axis=1, classes=4, should NOT warn
        y_true = np.ones((2, 4, 1), dtype="float32")
        y_pred = np.ones((2, 4, 1), dtype="float32")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            losses.categorical_crossentropy(y_true, y_pred, axis=1)
            # Check if any SyntaxWarning was issued
            syntax_warnings = [
                warn for warn in w if issubclass(warn.category, SyntaxWarning)
            ]
            self.assertEqual(len(syntax_warnings), 0)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, (0.001822, 0.000459, 0.169846))

    def test_invalid_label_smoothing(self):
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.categorical_crossentropy(
                [[1.0, 0.0, 0.0]],
                [[0.8, 0.1, 0.1]],
                label_smoothing=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.CategoricalCrossentropy(label_smoothing=1.1)(
                [[1.0, 0.0, 0.0]], [[0.8, 0.1, 0.1]]
            )
        with self.assertRaisesRegex(ValueError, "must be a float or int"):
            losses.categorical_crossentropy(
                [[1.0, 0.0, 0.0]], [[0.8, 0.1, 0.1]], label_smoothing=True
            )
        with self.assertRaisesRegex(ValueError, "must be a float or int"):
            losses.categorical_crossentropy(
                [[1.0, 0.0, 0.0]],
                [[0.8, 0.1, 0.1]],
                label_smoothing=np.bool_(True),
            )
        with self.assertRaisesRegex(ValueError, "label_smoothing.*range"):
            losses.categorical_crossentropy(
                [[1.0, 0.0, 0.0]],
                [[0.8, 0.1, 0.1]],
                label_smoothing=np.float32(-0.1),
            )

    def test_label_smoothing(self):
        logits = np.array([[100.0, -100.0, -100.0]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_label_smoothing_ndarray(self):
        logits = np.asarray([[100.0, -100.0, -100.0]])
        y_true = np.asarray([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_label_smoothing_with_axis(self):
        # (batch, classes, spatial)
        num_classes = 3
        y_true = np.array(
            [[[1, 0], [0, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]],
            dtype="float32",
        )
        y_pred = np.array(
            [
                [[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]],
                [[0.1, 0.8], [0.8, 0.1], [0.1, 0.1]],
            ],
            dtype="float32",
        )
        label_smoothing = 0.1
        axis = 1
        cce_obj = losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing, axis=axis
        )
        loss = cce_obj(y_true, y_pred)

        y_true_smoothed = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )
        expected_loss = -np.sum(y_true_smoothed * np.log(y_pred), axis=axis)
        expected_loss = np.mean(expected_loss)

        self.assertAlmostEqual(loss, expected_loss, decimal=5)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )

        cce_obj = losses.CategoricalCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)

    def test_dtype_arg(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalCrossentropy(dtype="bfloat16")
        loss = cce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class SparseCategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.SparseCategoricalCrossentropy(name="scce")
        )
        scce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=3, axis=0, name="scce"
        )
        config = scce_obj.get_config()
        self.assertEqual(config["from_logits"], True)
        self.assertEqual(config["ignore_class"], 3)
        self.assertEqual(config["axis"], 0)
        self.run_class_serialization_test(scce_obj)

    def test_all_correct_unweighted(self):
        y_true = np.array([[0], [1], [2]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.SparseCategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([0, 1, 2])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317, 3)

    def test_sample_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829, 3)

    def test_no_reduction(self):
        y_true = np.array([[0], [1], [2]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, (0.001822, 0.000459, 0.169846))

    def test_ignore_class(self):
        y_true = np.array([[-1, 2]])
        logits = np.array([[[0.854, 0.698, 0.598], [0.088, 0.86, 0.018]]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=-1, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, [[0.0, 1.480129]])

        y_true = np.array([[[-1], [2]]])
        logits = np.array([[[0.854, 0.698, 0.598], [0.088, 0.86, 0.018]]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=-1, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, [[0.0, 1.480129]])

    def test_binary_segmentation(self):
        y_true = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        )
        output = losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        self.assertAllClose(output, 0.0)

        y_true = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.4]],
            ]
        )
        expected = np.array([-np.log(0.2), -np.log(0.4)])
        output = losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        self.assertAllClose(output, expected.sum() / 16.0)  # 16 pixels

    def test_binary_segmentation_different_axis(self):
        y_true = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        )
        y_pred_reshaped = np.moveaxis(y_pred, source=2, destination=0)
        if backend.backend() == "tensorflow":
            expected_message = (
                "Only axis=-1 is currently supported. Received: axis=0"
            )
            escaped_message = re.escape(expected_message)

            with pytest.raises(ValueError, match=escaped_message):
                losses.SparseCategoricalCrossentropy(axis=0)(
                    y_true, y_pred_reshaped
                )
        elif backend.backend() == "jax":
            expected_message = (
                "Arguments `target` and `output` "
                "must have the same shape up until"
                " the last dimension: target.shape=(4, 4),"
                " output.shape=(2, 4, 4)"
            )
            escaped_message = re.escape(expected_message)

            with pytest.raises(ValueError, match=escaped_message):
                losses.SparseCategoricalCrossentropy(axis=0)(
                    y_true, y_pred_reshaped
                )
        elif backend.backend() == "torch":
            output = losses.SparseCategoricalCrossentropy(axis=0)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, 0.0)

        if backend.backend() == "torch":
            y_true = np.array(
                [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
            )
            y_pred = np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
                    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.4]],
                ]
            )
            y_pred_reshaped = np.moveaxis(y_pred, source=2, destination=0)
            expected = np.array([-np.log(0.2), -np.log(0.4)])
            output = losses.SparseCategoricalCrossentropy(axis=0)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, expected.sum() / 16.0)

            y_true = np.array([y_true, y_true, y_true])
            y_pred_reshaped = np.array(
                [y_pred_reshaped, y_pred_reshaped, y_pred_reshaped]
            )
            output = losses.SparseCategoricalCrossentropy(axis=1)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, expected.sum() / 16.0)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Channels-first axis only supported on Torch.",
    )
    def test_squeezes_singleton_class_dim_on_user_axis(self):
        y_true = np.random.randint(0, 2, size=(2, 1, 4, 4)).astype("float32")
        y_pred = np.random.random((2, 2, 4, 4)).astype("float32")
        loss = losses.sparse_categorical_crossentropy(y_true, y_pred, axis=1)
        self.assertEqual(loss.shape, (2, 4, 4))

    def test_multi_class_segmentation(self):
        y_true = np.array(
            [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )
        output = losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        self.assertAllClose(output, 0.0)

        y_true = np.array(
            [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.0, 0.8],
                ],
                [
                    [0.7, 0.3, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )
        expected = np.array(
            [
                -np.log(0.2),
                -np.log(0.3),
                -np.log(0.5),
            ]
        )
        output = losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        self.assertAllClose(output, expected.sum() / 16.0)  # 16 pixels

    def test_multi_class_segmentation_different_axis(self):
        y_true = np.array(
            [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )
        y_pred_reshaped = np.moveaxis(y_pred, source=2, destination=0)
        if backend.backend() == "tensorflow":
            expected_message = (
                "Only axis=-1 is currently supported. Received: axis=0"
            )
            escaped_message = re.escape(expected_message)

            with pytest.raises(ValueError, match=escaped_message):
                losses.SparseCategoricalCrossentropy(axis=0)(
                    y_true, y_pred_reshaped
                )
        elif backend.backend() == "jax":
            expected_message = (
                "Arguments `target` and `output` "
                "must have the same shape up until"
                " the last dimension: target.shape=(4, 4),"
                " output.shape=(3, 4, 4)"
            )
            escaped_message = re.escape(expected_message)

            with pytest.raises(ValueError, match=escaped_message):
                losses.SparseCategoricalCrossentropy(axis=0)(
                    y_true, y_pred_reshaped
                )
        elif backend.backend() == "torch":
            output = losses.SparseCategoricalCrossentropy(axis=0)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, 0.0)

        if backend.backend() == "torch":
            y_true = np.array(
                [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
            )
            y_pred = np.array(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.2, 0.0, 0.8],
                    ],
                    [
                        [0.7, 0.3, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                ]
            )
            expected = np.array(
                [
                    -np.log(0.2),
                    -np.log(0.3),
                    -np.log(0.5),
                ]
            )
            y_pred_reshaped = np.moveaxis(y_pred, source=2, destination=0)
            output = losses.SparseCategoricalCrossentropy(axis=0)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, expected.sum() / 16.0)
            y_true = np.array([y_true, y_true, y_true])
            y_pred_reshaped = np.array(
                [y_pred_reshaped, y_pred_reshaped, y_pred_reshaped]
            )
            output = losses.SparseCategoricalCrossentropy(axis=1)(
                y_true, y_pred_reshaped
            )
            self.assertAllClose(output, expected.sum() / 16.0)

    def test_dtype_arg(self):
        y_true = np.array([[0], [1], [2]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.SparseCategoricalCrossentropy(dtype="bfloat16")
        loss = cce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")

    def test_label_smoothing_config(self):
        scce_obj = losses.SparseCategoricalCrossentropy(
            label_smoothing=0.1, name="scce"
        )
        self.assertEqual(scce_obj.get_config()["label_smoothing"], 0.1)
        self.run_class_serialization_test(scce_obj)

    def test_label_smoothing_matches_one_hot(self):
        # Smoothing an integer label must match `categorical_crossentropy`
        # on the equivalent one-hot label.
        y_true = np.array([1, 2, 0], dtype="int32")
        logits = np.array(
            [[0.5, 2.0, -1.0], [1.5, 0.0, 0.5], [-0.5, 1.0, 2.0]],
            dtype="float32",
        )
        probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        one_hot = np.eye(3, dtype="float32")[y_true]
        for from_logits, y_pred in ((True, logits), (False, probs)):
            for label_smoothing in (0.0, 0.1, 0.5, 1.0):
                expected = losses.categorical_crossentropy(
                    one_hot,
                    y_pred,
                    from_logits=from_logits,
                    label_smoothing=label_smoothing,
                )
                output = losses.sparse_categorical_crossentropy(
                    y_true,
                    y_pred,
                    from_logits=from_logits,
                    label_smoothing=label_smoothing,
                )
                self.assertAllClose(output, expected)

    def test_label_smoothing_matches_one_hot_3d(self):
        # Same equivalence on a segmentation-shaped input.
        y_true = np.array([[0, 2], [1, 1]], dtype="int32")
        logits = np.array(
            [
                [[0.5, 2.0, -1.0], [1.5, 0.0, 0.5]],
                [[-0.5, 1.0, 2.0], [0.2, 0.3, 0.4]],
            ],
            dtype="float32",
        )
        one_hot = np.eye(3, dtype="float32")[y_true]
        expected = losses.categorical_crossentropy(
            one_hot, logits, from_logits=True, label_smoothing=0.1
        )
        output = losses.sparse_categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=0.1
        )
        self.assertAllClose(output, expected)

    def test_label_smoothing_explicit_value(self):
        # For label_smoothing=0.3 over 3 classes, the target is 0.8 for the
        # true class and 0.1 for each of the other two.
        y_true = np.array([0], dtype="int32")
        y_pred = np.array([[0.6, 0.3, 0.1]], dtype="float32")
        label_smoothing = 0.3
        smooth = label_smoothing / 3.0
        expected = -(
            (1.0 - label_smoothing + smooth) * np.log(0.6)
            + smooth * np.log(0.3)
            + smooth * np.log(0.1)
        )
        output = losses.sparse_categorical_crossentropy(
            y_true, y_pred, label_smoothing=label_smoothing
        )
        self.assertAllClose(output, [expected])

    def test_label_smoothing_default_is_a_no_op(self):
        y_true = np.array([0, 1], dtype="int32")
        y_pred = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], dtype="float32")
        self.assertAllClose(
            losses.sparse_categorical_crossentropy(y_true, y_pred),
            losses.sparse_categorical_crossentropy(
                y_true, y_pred, label_smoothing=0.0
            ),
        )

    def test_label_smoothing_with_ignore_class(self):
        y_true = np.array([0, 255, 2], dtype="int32")
        y_pred = np.array(
            [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]],
            dtype="float32",
        )
        output = losses.sparse_categorical_crossentropy(
            y_true, y_pred, ignore_class=255, label_smoothing=0.1
        )
        # The reference uses an in-range label in the ignored position, since
        # the entries are independent.
        reference = losses.sparse_categorical_crossentropy(
            np.array([0, 0, 2], dtype="int32"), y_pred, label_smoothing=0.1
        )
        self.assertAllClose(output[0], reference[0])
        self.assertAllClose(output[2], reference[2])
        self.assertAllClose(output[1], 0.0)
        self.assertAllClose(get_keras_mask(output), [True, False, True])


class BinaryFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryFocalCrossentropy(name="bfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype="float32",
        )
        obj = losses.BinaryFocalCrossentropy(gamma=1.5)
        loss = obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [
                [100.0, -100.0, -100.0],
                [-100.0, 100.0, -100.0],
                [-100.0, -100.0, 100.0],
            ]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.268, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.799, 3)

    def test_scalar_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=1.23)
        self.assertAlmostEqual(loss, 0.3296, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=3.21)
        self.assertAlmostEqual(loss, 2.565, 3)

    def test_sample_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.34415, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.95977, 3)

    def test_no_reduction(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(
            gamma=2.0,
            reduction=None,
        )
        loss = obj(y_true, y_pred)
        self.assertAllClose(loss, (0.515547, 0.020513))

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch doesn't support bfloat16 for BinaryFocalCrossentropy",
    )
    def test_dtype_arg(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(dtype="bfloat16")
        loss = obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalFocalCrossentropy(name="cfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.02059, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.000345, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.047368, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.000794, 4)

    def test_warning_for_invalid_shape(self):
        # 2D case: should warn
        y_true = np.array([[1], [1]], dtype="float32")
        y_pred = np.array([[0.5], [0.5]], dtype="float32")
        with pytest.warns(
            SyntaxWarning,
            match=r"expected y_pred.shape to be \(batch_size, num_classes\)",
        ):
            losses.categorical_focal_crossentropy(y_true, y_pred)

        # 3D case: axis=1, classes=1, should warn
        y_true = np.ones((2, 1, 4), dtype="float32")
        y_pred = np.ones((2, 1, 4), dtype="float32")
        with pytest.warns(
            SyntaxWarning,
            match=r"expected y_pred.shape to be \(batch_size, num_classes\)",
        ):
            losses.categorical_focal_crossentropy(y_true, y_pred, axis=1)

        # 3D case: axis=1, classes=4, should NOT warn
        y_true = np.ones((2, 4, 1), dtype="float32")
        y_pred = np.ones((2, 4, 1), dtype="float32")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            losses.categorical_focal_crossentropy(y_true, y_pred, axis=1)
            # Check if any SyntaxWarning was issued
            syntax_warnings = [
                warn for warn in w if issubclass(warn.category, SyntaxWarning)
            ]
            self.assertEqual(len(syntax_warnings), 0)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.06987, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.001933, 3)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, (1.5096224e-09, 2.4136547e-11, 1.0360638e-03))

    def test_label_smoothing(self):
        logits = np.array([[4.9, -0.5, 2.05]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1

        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)

        expected_value = 0.06685
        self.assertAlmostEqual(loss, expected_value, 3)

    def test_label_smoothing_with_axis(self):
        # (batch, classes, spatial)
        num_classes = 3
        y_true = np.array(
            [[[1, 0], [0, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]],
            dtype="float32",
        )
        y_pred = np.array(
            [
                [[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]],
                [[0.1, 0.8], [0.8, 0.1], [0.1, 0.1]],
            ],
            dtype="float32",
        )
        label_smoothing = 0.1
        axis = 1
        focal_obj = losses.CategoricalFocalCrossentropy(
            label_smoothing=label_smoothing, axis=axis
        )
        loss = focal_obj(y_true, y_pred)

        # Manually compute smoothed focal loss for verification
        y_true_smoothed = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )
        # alpha=0.25, gamma=2.0 (defaults)
        alpha = 0.25
        gamma = 2.0

        cross_entropy = -y_true_smoothed * np.log(y_pred)
        p_t = y_pred
        focal_loss = alpha * np.power(1 - p_t, gamma) * cross_entropy
        expected_loss = np.sum(focal_loss, axis=axis)
        expected_loss = np.mean(expected_loss)

        self.assertAlmostEqual(loss, expected_loss, decimal=5)

    def test_dtype_arg(self):
        logits = np.array([[4.9, -0.5, 2.05]])
        y_true = np.array([[1, 0, 0]])
        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, dtype="bfloat16"
        )
        loss = cce_obj(y_true, logits)
        self.assertDType(loss, "bfloat16")


class SparseCategoricalFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.SparseCategoricalFocalCrossentropy(name="scfce")
        )
        loss_obj = losses.SparseCategoricalFocalCrossentropy(
            alpha=[0.1, 0.2, 0.7],
            gamma=1.5,
            from_logits=True,
            ignore_class=255,
            axis=1,
            reduction="sum",
            name="scfce",
        )
        config = loss_obj.get_config()
        self.assertEqual(config["alpha"], [0.1, 0.2, 0.7])
        self.assertEqual(config["gamma"], 1.5)
        self.assertEqual(config["from_logits"], True)
        self.assertEqual(config["ignore_class"], 255)
        self.assertEqual(config["axis"], 1)
        self.run_class_serialization_test(loss_obj)

    @parameterized.named_parameters(
        ("probabilities_scalar_alpha", False, 0.25),
        ("probabilities_per_class_alpha", False, [0.1, 0.3, 0.6]),
        ("logits_scalar_alpha", True, 0.25),
        ("logits_per_class_alpha", True, [0.1, 0.3, 0.6]),
    )
    def test_matches_categorical_focal_crossentropy(self, from_logits, alpha):
        y_true = np.array([0, 1, 2])
        y_true_one_hot = np.eye(3)[y_true]
        if from_logits:
            y_pred = np.array(
                [
                    [8.0, 1.0, 1.0],
                    [0.0, 9.0, 1.0],
                    [2.0, 3.0, 5.0],
                ]
            )
        else:
            y_pred = np.array(
                [
                    [0.9, 0.05, 0.05],
                    [0.5, 0.89, 0.6],
                    [0.05, 0.01, 0.94],
                ],
                dtype="float32",
            )

        categorical_alpha = (
            alpha
            if np.ndim(alpha) == 0
            else ops.convert_to_tensor(alpha, dtype="float32")
        )
        expected = losses.categorical_focal_crossentropy(
            y_true_one_hot,
            y_pred,
            alpha=categorical_alpha,
            gamma=1.5,
            from_logits=from_logits,
        )
        result = losses.sparse_categorical_focal_crossentropy(
            y_true,
            y_pred,
            alpha=alpha,
            gamma=1.5,
            from_logits=from_logits,
        )
        self.assertAllClose(result, expected)

    def test_gamma_zero_matches_weighted_sparse_crossentropy(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
            dtype="float32",
        )
        alpha = np.array([0.2, 0.3, 0.5], dtype="float32")
        expected = losses.sparse_categorical_crossentropy(
            y_true, y_pred
        ) * ops.convert_to_tensor(alpha[y_true])
        result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, alpha=alpha, gamma=0.0
        )
        self.assertAllClose(result, expected)

    def test_ignore_class(self):
        y_true = np.array([0, 255, 2])
        y_pred = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
            dtype="float32",
        )
        loss_obj = losses.SparseCategoricalFocalCrossentropy(
            ignore_class=255, reduction=None
        )
        function_result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, ignore_class=255
        )
        self.assertAllEqual(
            get_keras_mask(function_result), [True, False, True]
        )
        result = loss_obj(y_true, y_pred)
        expected_valid = losses.sparse_categorical_focal_crossentropy(
            y_true[[0, 2]], y_pred[[0, 2]]
        )
        self.assertAllClose(
            result,
            ops.convert_to_tensor([expected_valid[0], 0.0, expected_valid[1]]),
        )

        weighted_result = loss_obj(
            y_true, y_pred, sample_weight=np.array([2.0, 100.0, 3.0])
        )
        self.assertAllClose(
            weighted_result,
            ops.convert_to_tensor(
                [expected_valid[0] * 2.0, 0.0, expected_valid[1] * 3.0]
            ),
        )

        reduced_result = losses.SparseCategoricalFocalCrossentropy(
            ignore_class=255
        )(y_true, y_pred)
        self.assertAllClose(
            reduced_result, np.mean(ops.convert_to_numpy(expected_valid))
        )

        all_ignored_result = losses.SparseCategoricalFocalCrossentropy(
            ignore_class=255
        )(np.array([255, 255]), y_pred[:2])
        self.assertAllClose(all_ignored_result, 0.0)

    def test_ignore_class_combines_with_y_pred_mask(self):
        y_true = np.array([0, 255, 2])
        y_pred_values = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
            dtype="float32",
        )
        y_pred = ops.convert_to_tensor(y_pred_values)
        set_keras_mask(y_pred, ops.convert_to_tensor([False, True, True]))

        result = losses.SparseCategoricalFocalCrossentropy(
            ignore_class=255, reduction=None
        )(y_true, y_pred)
        expected = losses.sparse_categorical_focal_crossentropy(
            np.array([2]), y_pred_values[2:]
        )
        self.assertAllClose(
            result, ops.convert_to_tensor([0.0, 0.0, expected[0]])
        )

    def test_custom_axis(self):
        y_true = np.array([[0, 1], [2, 0]])
        y_true_one_hot = np.moveaxis(np.eye(3)[y_true], -1, 1)
        y_pred = np.array(
            [
                [[0.8, 0.1], [0.1, 0.7], [0.1, 0.2]],
                [[0.1, 0.6], [0.2, 0.2], [0.7, 0.2]],
            ],
            dtype="float32",
        )
        expected = losses.categorical_focal_crossentropy(
            y_true_one_hot, y_pred, gamma=1.5, axis=1
        )
        result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, gamma=1.5, axis=1
        )
        self.assertAllClose(result, expected)

    def test_class_custom_axis_with_trailing_singleton_dimension(self):
        y_true = np.array([[[0], [1]], [[2], [0]]], dtype="int32")
        y_pred = np.array(
            [
                [
                    [[0.8], [0.1]],
                    [[0.1], [0.7]],
                    [[0.1], [0.2]],
                ],
                [
                    [[0.1], [0.6]],
                    [[0.2], [0.2]],
                    [[0.7], [0.2]],
                ],
            ],
            dtype="float32",
        )
        expected = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, axis=1
        )
        result = losses.SparseCategoricalFocalCrossentropy(
            axis=1, reduction=None
        )(y_true, y_pred)
        self.assertEqual(result.shape, (2, 2, 1))
        self.assertAllClose(result, expected)

    @parameterized.named_parameters(
        ("negative", -1),
        ("equal_to_num_classes", 3),
        ("int64_max", np.iinfo("int64").max),
    )
    def test_out_of_range_labels_produce_nan(self, invalid_label):
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype="float32")
        y_true = np.array([0, invalid_label], dtype="int64")
        result = losses.sparse_categorical_focal_crossentropy(y_true, y_pred)
        result = ops.convert_to_numpy(result)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isnan(result[1]))

    # JAX and NumPy standardize int64 label arrays to int32 before the loss
    # sees them, so int64.max cannot remain a distinct sentinel.
    @parameterized.named_parameters(
        ("negative", -1),
        ("equal_to_num_classes", 3),
        ("int32_max", np.iinfo("int32").max),
        *(
            ()
            if backend.backend() in ("jax", "numpy")
            else (("int64_max", np.iinfo("int64").max),)
        ),
    )
    def test_out_of_range_ignore_class(self, ignored_label):
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype="float32")
        y_true = np.array([0, ignored_label], dtype="int64")
        result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, ignore_class=ignored_label
        )
        self.assertAllClose(result[1], 0.0)

    @parameterized.named_parameters(
        ("too_short", [0.1, 0.2]),
        ("too_long", [0.1, 0.2, 0.3, 0.4]),
        ("rank_two", np.ones((3, 1))),
    )
    def test_alpha_shape_validation(self, alpha):
        y_true = np.array([0, 1])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype="float32")
        with self.assertRaisesRegex(ValueError, "`alpha` must|one value"):
            losses.sparse_categorical_focal_crossentropy(
                y_true, y_pred, alpha=alpha
            )

    @parameterized.named_parameters(
        ("uint8", "uint8", 256, 255),
        ("int8", "int8", 128, 127),
    )
    def test_narrow_integer_labels(self, dtype, num_classes, label):
        y_true = np.array([label], dtype=dtype)
        y_pred = np.full(
            (1, num_classes),
            0.1 / (num_classes - 1),
            dtype="float32",
        )
        y_pred[0, label] = 0.9
        y_true_one_hot = np.eye(num_classes)[y_true.astype("int32")]
        expected = losses.categorical_focal_crossentropy(y_true_one_hot, y_pred)
        result = losses.sparse_categorical_focal_crossentropy(
            y_true,
            y_pred,
            # This value is intentionally outside the label dtype.
            ignore_class=num_classes,
        )
        self.assertAllClose(result, expected)

        loss_obj = losses.SparseCategoricalFocalCrossentropy(
            dtype="float16", ignore_class=num_classes, reduction=None
        )
        mixed_precision_result = loss_obj(y_true, y_pred)
        self.assertDType(mixed_precision_result, "float16")
        self.assertTrue(
            np.all(np.isfinite(ops.convert_to_numpy(mixed_precision_result)))
        )

    # JAX and NumPy standardize int64 label arrays to int32 before the loss
    # sees them, so int64.max cannot remain a distinct sentinel.
    @parameterized.named_parameters(
        ("int32_max", np.iinfo("int32").max),
        *(
            ()
            if backend.backend() in ("jax", "numpy")
            else (("int64_max", np.iinfo("int64").max),)
        ),
    )
    def test_class_preserves_labels_with_low_precision_dtype(
        self, ignored_label
    ):
        num_classes = 300
        label = 257
        y_true = np.array([label], dtype="int64")
        y_pred = np.full(
            (1, num_classes), 0.1 / (num_classes - 1), dtype="float32"
        )
        y_pred[0, label] = 0.9
        low_precision_y_pred = ops.convert_to_tensor(y_pred, dtype="bfloat16")
        expected = losses.sparse_categorical_focal_crossentropy(
            y_true, low_precision_y_pred
        )
        result = losses.SparseCategoricalFocalCrossentropy(
            dtype="bfloat16", reduction=None
        )(y_true, y_pred)
        self.assertDType(result, "bfloat16")
        self.assertAllClose(result, expected)

        ignored_result = losses.SparseCategoricalFocalCrossentropy(
            dtype="bfloat16",
            ignore_class=ignored_label,
            reduction=None,
        )(
            np.array([ignored_label], dtype="int64"),
            np.full((1, 3), 1 / 3, dtype="float32"),
        )
        self.assertAllClose(ignored_result, [0.0])

    def test_symbolic_inputs(self):
        y_true = Input(shape=(), dtype="int32", name="y_true")
        y_pred = Input(shape=(3,), name="y_pred")
        result = losses.sparse_categorical_focal_crossentropy(y_true, y_pred)
        self.assertEqual(result.shape, (None,))

        class_result = losses.SparseCategoricalFocalCrossentropy(
            reduction=None
        )(y_true, y_pred)
        self.assertEqual(class_result.shape, (None,))

        model = Functional([y_true, y_pred], result)
        class_model = Functional([y_true, y_pred], class_result)
        input_values = [
            np.array([0, 2], dtype="int32"),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
        ]
        actual = model(input_values)
        expected = losses.sparse_categorical_focal_crossentropy(
            input_values[0], input_values[1]
        )
        self.assertAllClose(actual, expected)
        self.assertAllClose(class_model(input_values), expected)

        invalid = model(
            [
                np.array([0, 3], dtype="int32"),
                np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
            ]
        )
        invalid = ops.convert_to_numpy(invalid)
        self.assertTrue(np.isfinite(invalid[0]))
        self.assertTrue(np.isnan(invalid[1]))

    def test_symbolic_dynamic_class_dimension(self):
        y_true = Input(shape=(), dtype="int32")
        y_pred = Input(shape=(None,))
        result = losses.sparse_categorical_focal_crossentropy(y_true, y_pred)
        self.assertEqual(result.shape, (None,))
        model = Functional([y_true, y_pred], result)
        actual = model(
            [
                np.array([0, 2], dtype="int32"),
                np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
            ]
        )
        expected = losses.sparse_categorical_focal_crossentropy(
            np.array([0, 2], dtype="int32"),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
        )
        self.assertAllClose(actual, expected)

        weighted_result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, alpha=[0.1, 0.2, 0.7]
        )
        weighted_model = Functional([y_true, y_pred], weighted_result)
        actual = weighted_model(
            [
                np.array([0, 2], dtype="int32"),
                np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
            ]
        )
        expected = losses.sparse_categorical_focal_crossentropy(
            np.array([0, 2], dtype="int32"),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
            alpha=[0.1, 0.2, 0.7],
        )
        self.assertAllClose(actual, expected)

        y_true = Input(shape=(2,), dtype="int32")
        y_pred = Input(shape=(None, 2))
        result = losses.sparse_categorical_focal_crossentropy(
            y_true, y_pred, axis=1
        )
        self.assertEqual(result.shape, (None, 2))
        model = Functional([y_true, y_pred], result)
        y_true_value = np.array([[0, 1], [2, 0]], dtype="int32")
        y_pred_value = np.array(
            [
                [[0.8, 0.1], [0.1, 0.7], [0.1, 0.2]],
                [[0.1, 0.6], [0.2, 0.2], [0.7, 0.2]],
            ],
            dtype="float32",
        )
        actual = model([y_true_value, y_pred_value])
        expected = losses.sparse_categorical_focal_crossentropy(
            y_true_value, y_pred_value, axis=1
        )
        self.assertAllClose(actual, expected)

    def test_integer_predictions(self):
        y_true = np.array([0, 1])
        y_true_one_hot = np.eye(3, dtype="int32")[y_true]
        y_pred = np.array([[2, 1, 1], [1, 3, 1]], dtype="int32")
        expected = losses.categorical_focal_crossentropy(
            y_true_one_hot, y_pred.astype("float32")
        )
        result = losses.sparse_categorical_focal_crossentropy(y_true, y_pred)
        self.assertAllClose(result, expected)

    def test_singleton_class_dimension_in_targets(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
            dtype="float32",
        )
        result = losses.sparse_categorical_focal_crossentropy(y_true, y_pred)
        expected = losses.sparse_categorical_focal_crossentropy(
            np.squeeze(y_true, axis=-1), y_pred
        )
        self.assertAllClose(result, expected)

    def test_dtype_arg(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
            dtype="float32",
        )
        loss_obj = losses.SparseCategoricalFocalCrossentropy(dtype="bfloat16")
        result = loss_obj(y_true, y_pred)
        self.assertDType(result, "bfloat16")


class CTCTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.CTC(name="myctc"))

    def test_correctness(self):
        logits = (np.arange(24).reshape((2, 4, 3)).astype("float32") - 12) / 100
        y_true = np.array(([[1, 2, 1, 0], [1, 2, 0, 2]]))
        output = losses.CTC()(y_true, logits)
        self.assertAllClose(output, 2.448645, tpu_atol=1e-3, tpu_rtol=1e-3)

    def test_dtype_arg(self):
        logits = (np.arange(24).reshape((2, 4, 3)).astype("float32") - 12) / 100
        y_true = np.array(([[1, 2, 1, 0], [1, 2, 0, 2]]))
        output = losses.CTC(dtype="bfloat16")(y_true, logits)
        self.assertDType(output, "bfloat16")


class DiceTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.Dice(name="mydice"))

    def test_correctness(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Dice()(y_true, y_pred)
        self.assertAllClose(output, -0.55555546)

    def test_binary_segmentation(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Dice()(y_true, y_pred)
        self.assertAllClose(output, 0.77777773)

    def test_binary_segmentation_with_axis(self):
        y_true = np.array(
            [[[[1.0], [1.0]], [[0.0], [0.0]]], [[[1.0], [1.0]], [[0.0], [0.0]]]]
        )
        y_pred = np.array(
            [[[[0.0], [1.0]], [[0.0], [1.0]]], [[[0.4], [0.0]], [[0.0], [0.9]]]]
        )
        output = losses.Dice(axis=(1, 2, 3), reduction=None)(y_true, y_pred)
        self.assertAllClose(output, [0.5, 0.75757575])

    def test_dtype_arg(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Dice(dtype="bfloat16")(y_true, y_pred)
        self.assertDType(output, "bfloat16")


class TverskyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.Tversky(name="mytversky"))

    def test_correctness(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky()(y_true, y_pred)
        self.assertAllClose(output, -0.55555558)

    def test_correctness_custom_coefficients(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky(alpha=0.2, beta=0.8)(y_true, y_pred)
        self.assertAllClose(output, -0.94444442)

    def test_binary_segmentation(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Tversky()(y_true, y_pred)
        self.assertAllClose(output, 0.77777779)

    def test_binary_segmentation_with_axis(self):
        y_true = np.array(
            [[[[1.0], [1.0]], [[0.0], [0.0]]], [[[1.0], [1.0]], [[0.0], [0.0]]]]
        )
        y_pred = np.array(
            [[[[0.0], [1.0]], [[0.0], [1.0]]], [[[0.4], [0.0]], [[0.0], [0.9]]]]
        )
        output = losses.Tversky(axis=(1, 2, 3), reduction=None)(y_true, y_pred)
        self.assertAllClose(output, [0.5, 0.75757575])

    def test_binary_segmentation_custom_coefficients(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Tversky(alpha=0.2, beta=0.8)(y_true, y_pred)
        self.assertAllClose(output, 0.76190472)

    def test_binary_segmentation_custom_coefficients_with_axis(self):
        y_true = np.array(
            [[[[1.0], [1.0]], [[0.0], [0.0]]], [[[1.0], [1.0]], [[0.0], [0.0]]]]
        )
        y_pred = np.array(
            [[[[0.0], [1.0]], [[0.0], [1.0]]], [[[0.4], [0.0]], [[0.0], [0.9]]]]
        )
        output = losses.Tversky(
            alpha=0.2, beta=0.8, axis=(1, 2, 3), reduction=None
        )(y_true, y_pred)
        self.assertAllClose(output, [0.5, 0.78494626])

    def test_dtype_arg(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky(dtype="bfloat16")(y_true, y_pred)
        self.assertDType(output, "bfloat16")


class CircleTest(testing.TestCase):
    def setup(self):
        super().setUp()
        self.y_true = np.array([1, 1, 2, 2, 3])
        self.y_pred = np.array(
            [
                [0.70014004, -0.42008403, 0.14002801, 0.56011203],
                [0.17609018, 0.70436073, -0.52827054, 0.44022545],
                [-0.34050261, 0.25537696, -0.68100522, 0.59587957],
                [0.32163376, -0.75047877, 0.53605627, -0.21442251],
                [0.51261459, -0.34174306, 0.17087153, 0.76892189],
            ]
        )
        self.ref_labels = np.array([1, 1, 2, 2, 3, 4])
        self.ref_embeddings = np.array(
            [
                [0.40824829, -0.54433105, 0.27216553, 0.68041382],
                [0.76376261, 0.10910895, -0.54554473, 0.32732684],
                [-0.74420841, 0.24806947, 0.49613894, -0.3721042],
                [0.52981294, -0.13245324, 0.79471941, -0.26490647],
                [0.54554473, -0.32732684, 0.10910895, 0.76376261],
                [-0.27216553, 0.68041382, 0.40824829, -0.54433105],
            ]
        )

    def test_config(self):
        self.run_class_serialization_test(
            losses.Circle(name="mycircle", gamma=80.0, margin=0.4)
        )

    def test_correctness(self):
        self.setup()
        circle_loss = losses.Circle(gamma=80.0, margin=0.4)
        loss = circle_loss(self.y_true, self.y_pred)
        self.assertAlmostEqual(loss, 188.3883, tpu_decimal=0)

        circle_loss = losses.Circle(gamma=256, margin=0.25)
        loss = circle_loss(self.y_true, self.y_pred)
        self.assertAlmostEqual(loss, 652.7617, tpu_decimal=0)

        loss = losses.circle(
            self.y_true,
            self.y_pred,
            ref_labels=self.ref_labels,
            ref_embeddings=self.ref_embeddings,
            gamma=80.0,
            margin=0.4,
            remove_diagonal=False,
        )

        self.assertAllClose(
            loss,
            (61.5844, 94.3465, 276.9344, 90.9873, 48.8963),
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    def test_correctness_weighted(self):
        self.setup()
        sample_weight = np.array([2.0, 2.0, 1.0, 1.0, 0.5])
        circle_loss = losses.Circle(gamma=80.0, margin=0.4)
        loss = circle_loss(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, 244.91918, tpu_decimal=0)

    def test_no_reduction(self):
        self.setup()
        circle_loss = losses.Circle(gamma=80.0, margin=0.4, reduction=None)
        loss = circle_loss(self.ref_labels, self.ref_embeddings)

        self.assertAllClose(
            loss,
            [82.9116, 36.7942, 92.4590, 52.6798, 0.0, 0.0],
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    def test_sum_reduction(self):
        self.setup()
        circle_loss = losses.Circle(gamma=80.0, margin=0.4, reduction="sum")
        loss = circle_loss(self.ref_labels, self.ref_embeddings)

        self.assertAlmostEqual(loss, 264.845, tpu_decimal=0)

    def test_mean_with_sample_weight_reduction(self):
        self.setup()
        sample_weight = np.array([2.0, 2.0, 1.0, 1.0, 0.5])
        circle_loss = losses.Circle(
            gamma=80.0, margin=0.4, reduction="mean_with_sample_weight"
        )
        loss = circle_loss(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, 163.27948, tpu_decimal=0)

    def test_dtype_arg(self):
        self.setup()
        circle_loss = losses.Circle(dtype="bfloat16")
        loss = circle_loss(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalGeneralizedCrossEntropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalGeneralizedCrossEntropy(name="gce")
        )
        self.run_class_serialization_test(
            losses.CategoricalGeneralizedCrossEntropy(q=0.1, name="gce")
        )

    def test_basic_correctness_for_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
        # Calculate expected GCE loss manually
        # For q=0.5:
        # First sample (class 0): gce = (1 - 0.7^0.5) / 0.5
        # Second sample (class 1): gce = (1 - 0.8^0.5) / 0.5
        # Third sample (class 0): gce = (1 - 0.6^0.5) / 0.5
        # Fourth sample (class 1): gce = (1 - 0.6^0.5) / 0.5
        expected = np.array(
            [
                (1 - np.power(0.7, 0.5)) / 0.5,
                (1 - np.power(0.8, 0.5)) / 0.5,
                (1 - np.power(0.6, 0.5)) / 0.5,
                (1 - np.power(0.6, 0.5)) / 0.5,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy()(y_true, y_pred)
        self.assertAllClose(output, expected.sum() / len(expected))

        expected_q_08 = np.array(
            [
                (1 - np.power(0.7, 0.8)) / 0.8,
                (1 - np.power(0.8, 0.8)) / 0.8,
                (1 - np.power(0.6, 0.8)) / 0.8,
                (1 - np.power(0.6, 0.8)) / 0.8,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.8)(
            y_true, y_pred
        )
        self.assertAllClose(output, expected_q_08.sum() / len(expected_q_08))

    def test_basic_correctness_for_multi_class(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array(
            [[0.7, 0.3, 0.0], [0.2, 0.2, 0.6], [0.6, 0.4, 0.0], [0.2, 0.2, 0.6]]
        )
        # Calculate expected GCE loss manually
        # For q=0.5:
        # First sample (class 0): gce = (1 - 0.7^0.5) / 0.5
        # Second sample (class 1): gce = (1 - 0^0.5) / 0.5
        # Third sample (class 0): gce = (1 - 0.6^0.5) / 0.5
        # Fourth sample (class 1): gce = (1 - 0.0^0.5) / 0.5
        expected = np.array(
            [
                (1 - np.power(0.7, 0.5)) / 0.5,
                (1 - np.power(0.2, 0.5)) / 0.5,
                (1 - np.power(0.6, 0.5)) / 0.5,
                (1 - np.power(0.2, 0.5)) / 0.5,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy()(y_true, y_pred)
        self.assertAllClose(output, expected.sum() / len(expected))

        expected_q_08 = np.array(
            [
                (1 - np.power(0.7, 0.8)) / 0.8,
                (1 - np.power(0.2, 0.8)) / 0.8,
                (1 - np.power(0.6, 0.8)) / 0.8,
                (1 - np.power(0.2, 0.8)) / 0.8,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.8)(
            y_true, y_pred
        )
        self.assertAllClose(output, expected_q_08.sum() / len(expected_q_08))

    def test_binary_segmentation(self):
        y_true = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.5)(
            y_true, y_pred
        )
        self.assertAllClose(output, 0.0)

        y_true = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.4]],
            ]
        )
        expected = np.array(
            [
                (1 - np.power(0.2, 0.5)) / 0.5,
                (1 - np.power(0.4, 0.5)) / 0.5,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.5)(
            y_true, y_pred
        )
        self.assertAllClose(output, expected.sum() / 16.0)  # 16 pixels

    def test_multi_class_segmentation(self):
        y_true = np.array(
            [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.5)(
            y_true, y_pred
        )
        self.assertAllClose(output, 0.0)

        y_true = np.array(
            [[0, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
        )
        y_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.0, 0.8],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )
        expected = np.array(
            [
                (1 - np.power(0.2, 0.5)) / 0.5,
                (1 - np.power(0.0, 0.5)) / 0.5,
                (1 - np.power(0.5, 0.5)) / 0.5,
            ]
        )
        output = losses.CategoricalGeneralizedCrossEntropy(q=0.5)(
            y_true, y_pred
        )
        self.assertAllClose(output, expected.sum() / 16.0)  # 16 pixels

    def test_dtype_arg(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
        output = losses.CategoricalGeneralizedCrossEntropy(dtype="bfloat16")(
            y_true, y_pred
        )
        self.assertDType(output, "bfloat16")


class LossRegistryTest(testing.TestCase):
    def test_get_all_public_losses(self):
        not_objects = {"get", "serialize", "deserialize"}
        for name, obj in REGISTERED_NAMES_TO_OBJS.items():
            if (
                name.startswith("keras.losses.")
                and name.count(".") == 2
                and name.rsplit(".", 1)[1] not in not_objects
            ):
                get_result = losses_module.get(name.rsplit(".", 1)[1])
                if inspect.isclass(obj):
                    self.assertIsInstance(get_result, obj)
                else:
                    self.assertIs(get_result, obj)
