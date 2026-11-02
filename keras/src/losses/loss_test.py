import pickle

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import losses as losses_module
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import set_keras_mask
from keras.src.backend.common import standardize_dtype
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank


class ExampleLoss(Loss):
    def call(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class LossTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._global_dtype_policy = dtype_policies.dtype_policy.dtype_policy()
        self._floatx = backend.floatx()

    def tearDown(self):
        super().tearDown()
        dtype_policies.dtype_policy.set_dtype_policy(self._global_dtype_policy)
        backend.set_floatx(self._floatx)

    def test_squeeze_or_expand(self):
        x1 = ops.ones((3,))
        x2 = ops.ones((3, 1))
        x1, x2 = squeeze_or_expand_to_same_rank(x1, x2)
        self.assertEqual(ops.shape(x1), (3, 1))
        self.assertEqual(ops.shape(x2), (3, 1))

        x1 = ops.ones((3, 2))
        x2 = ops.ones((3, 2, 1))
        x1, x2 = squeeze_or_expand_to_same_rank(x1, x2)
        self.assertEqual(ops.shape(x1), (3, 2))
        self.assertEqual(ops.shape(x2), (3, 2))

        x1 = ops.ones((3,))
        x2 = ops.ones((3, 1))
        x2, x1 = squeeze_or_expand_to_same_rank(x2, x1)
        self.assertEqual(ops.shape(x1), (3, 1))
        self.assertEqual(ops.shape(x2), (3, 1))

        x1 = ops.ones((3, 2))
        x2 = ops.ones((3, 2, 1))
        x2, x1 = squeeze_or_expand_to_same_rank(x2, x1)
        self.assertEqual(ops.shape(x1), (3, 2))
        self.assertEqual(ops.shape(x2), (3, 2))

    def test_reduction(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # No reduction
        loss_fn = ExampleLoss(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose((y_true - y_pred) ** 2, loss)

        # sum
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2), loss)

        # sum_over_batch_size or mean
        loss_fn = ExampleLoss(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2) / 4, loss)

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            ExampleLoss(reduction="abc")

    def test_mask(self):
        mask = np.array([True, False, True, True])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum((masked_y_true - masked_y_pred) ** 2) / 3, loss
        )

        # Test edge case where everything is masked.
        mask = np.array([False, False, False, False])
        set_keras_mask(y_pred, mask)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    def test_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4, loss
        )

        # Test edge case where every weight is 0.
        sample_weight = np.array([0.0, 0.0, 0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    def test_mask_and_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        masked_sample_weight = np.array([0.4, 0.2, 0.1])
        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
            / 3,
            loss,
        )

    def test_mask_and_sample_weight_rank2(self):
        # check loss of inputs with duplicate rows doesn't change
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        rank1_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)

        # duplicate rows
        mask = ops.tile(ops.expand_dims(mask, axis=0), (2, 1))
        y_true = ops.tile(ops.expand_dims(y_true, axis=0), (2, 1))
        y_pred = ops.tile(ops.expand_dims(y_pred, axis=0), (2, 1))
        sample_weight = ops.tile(ops.expand_dims(sample_weight, axis=0), (2, 1))
        set_keras_mask(y_pred, mask)
        rank2_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(rank1_loss, rank2_loss)

    @parameterized.named_parameters(
        ("mask", "mask"),
        ("sample_weight", "sample_weight"),
        ("ys", "ys"),
    )
    def test_rank_adjustment(self, uprank):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        if uprank == "mask":
            mask = np.expand_dims(mask, -1)
        elif uprank == "sample_weight":
            sample_weight = np.expand_dims(sample_weight, -1)
        elif uprank == "ys":
            y_true = np.expand_dims(y_true, -1)
            y_pred = np.expand_dims(y_pred, -1)

        masked_sample_weight = np.array([0.4, 0.2, 0.1])
        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
            / 3,
            loss,
        )

    def test_mixed_dtypes(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1], dtype="float64")
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="int32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4,
            loss,
        )

    def test_pickle(self):
        loss = losses_module.get("mse")
        loss = pickle.loads(pickle.dumps(loss))
        self.assertEqual(loss, losses_module.mean_squared_error)

    def test_get_method(self):
        loss = losses_module.get("mse")
        self.assertEqual(loss, losses_module.mean_squared_error)

        loss = losses_module.get(None)
        self.assertEqual(loss, None)

        with self.assertRaises(ValueError):
            losses_module.get("typo")

    def test_dtype_arg(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Note: we use float16 and not float64 to test this because
        # JAX will map float64 to float32.
        loss_fn = ExampleLoss(dtype="float16")
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float16")

        # Test DTypePolicy for `dtype` argument
        loss_fn = ExampleLoss(dtype=dtype_policies.DTypePolicy("mixed_float16"))
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float16")

        # `dtype` setter should raise AttributeError
        with self.assertRaises(AttributeError):
            loss_fn.dtype = "bfloat16"

    def test_default_dtype(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Defaults to `keras.config.floatx()` not global `dtype_policy`
        dtype_policies.dtype_policy.set_dtype_policy("mixed_float16")
        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float32")

        backend.set_floatx("float16")
        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, backend.floatx())
