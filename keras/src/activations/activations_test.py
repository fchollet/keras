import numpy as np

from keras.src import activations
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.saving import object_registration


def _ref_softmax(values):
    m = np.max(values)
    e = np.exp(values - m)
    return e / np.sum(e)


def _ref_softplus(x):
    return np.log(np.ones_like(x) + np.exp(x))


def _ref_log_softmax(values):
    max_val = np.max(values)  # for numerical stability
    stabilized_values = values - max_val
    log_sum_exp = np.log(np.sum(np.exp(stabilized_values)))
    return stabilized_values - log_sum_exp


def _ref_leaky_relu(x, alpha=0.2):
    return x if x > 0 else alpha * x


def _ref_relu6(x):
    return min(max(0, x), 6)


def _ref_silu(x):
    return x / (1 + np.exp(-x))


def _ref_hard_sigmoid(x):
    x = (x / 6.0) + 0.5
    z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
    return z


def _ref_sparse_sigmoid(x):
    return np.where(x <= -1, 0, np.where(x >= 1, 1, 0.5 * (x + 1)))


def _ref_log_sigmoid(x):
    return -1 * _ref_softplus(-x)


def _ref_hard_silu(x):
    return x * np.minimum(np.maximum(0.0, x + 3.0), 6.0) * (1.0 / 6.0)


def _ref_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)


def _ref_softsign(x):
    return np.divide(x, np.ones_like(x) + np.absolute(x))


class ActivationsTest(testing.TestCase):
    def test_softmax(self):
        x = np.random.random((2, 5))

        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0])
        self.assertAllClose(result[0], expected, rtol=1e-5)

    def test_softmax_2d_axis_0(self):
        x = np.random.random((2, 5))
        result = activations.softmax(x[np.newaxis, :], axis=1)[0]
        expected = np.zeros((2, 5))
        for i in range(5):
            expected[:, i] = _ref_softmax(x[:, i])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_softmax_3d_axis_tuple(self):
        x = np.random.random((2, 3, 5))
        result = activations.softmax(x, axis=(1, 2))
        expected = np.zeros((2, 3, 5))
        for i in range(2):
            expected[i, :, :] = _ref_softmax(x[i, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_softmax_1d(self):
        x = np.random.random(5)
        result = activations.softmax(x)
        expected = _ref_softmax(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_softmax_higher_dim(self):
        x = np.random.random((2, 3, 4, 5))
        result = activations.softmax(x, axis=(2, 3))
        expected = np.zeros((2, 3, 4, 5))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :] = _ref_softmax(x[i, j, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_softmax_higher_dim_multiple_axes(self):
        x = np.random.random((2, 3, 4, 5, 6))
        result = activations.softmax(x, axis=(2, 3, 4))
        expected = np.zeros((2, 3, 4, 5, 6))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :, :] = _ref_softmax(x[i, j, :, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_softmax_negative_axis(self):
        x = np.random.random((2, 5))
        result = activations.softmax(x, axis=-1)
        expected = np.zeros((2, 5))
        for i in range(2):
            expected[i, :] = _ref_softmax(x[i, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_temporal_softmax(self):
        x = np.random.random((2, 2, 3)) * 10
        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0, 0])
        self.assertAllClose(result[0, 0], expected, rtol=1e-5)

    def test_log_softmax_2d_axis_0(self):
        x = np.random.random((2, 5))
        result = activations.log_softmax(x[np.newaxis, :], axis=1)[0]
        expected = np.zeros((2, 5))
        for i in range(5):
            expected[:, i] = _ref_log_softmax(x[:, i])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_log_softmax_3d_axis_tuple(self):
        x = np.random.random((2, 3, 5))
        result = activations.log_softmax(x, axis=(1, 2))
        expected = np.zeros((2, 3, 5))
        for i in range(2):
            expected[i, :, :] = _ref_log_softmax(x[i, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_log_softmax_1d(self):
        x = np.random.random(5)
        result = activations.log_softmax(x)
        expected = _ref_log_softmax(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_log_softmax_higher_dim(self):
        x = np.random.random((2, 3, 4, 5))
        result = activations.log_softmax(x, axis=(2, 3))
        expected = np.zeros((2, 3, 4, 5))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :] = _ref_log_softmax(x[i, j, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_log_softmax_higher_dim_multiple_axes(self):
        x = np.random.random((2, 3, 4, 5, 6))
        result = activations.log_softmax(x, axis=(2, 3, 4))
        expected = np.zeros((2, 3, 4, 5, 6))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :, :] = _ref_log_softmax(x[i, j, :, :, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_log_softmax_negative_axis(self):
        x = np.random.random((2, 5))
        result = activations.log_softmax(x, axis=-1)
        expected = np.zeros((2, 5))
        for i in range(2):
            expected[i, :] = _ref_log_softmax(x[i, :])
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_temporal_log_softmax(self):
        x = np.random.random((2, 2, 3)) * 10
        result = activations.log_softmax(x[np.newaxis, :])[0]
        expected = _ref_log_softmax(x[0, 0])
        self.assertAllClose(result[0, 0], expected, rtol=1e-5)

    def test_selu(self):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        positive_values = np.array([[1, 2]], dtype=backend.floatx())
        result = activations.selu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values * scale, rtol=1e-5)

        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.selu(negative_values[np.newaxis, :])[0]
        true_result = (np.exp(negative_values) - 1) * scale * alpha
        self.assertAllClose(result, true_result)

    def test_softplus(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.softplus(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_softplus)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.softplus(x_1d)
        expected_1d = np.vectorize(_ref_softplus)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.softplus(x_3d)
        expected_3d = np.vectorize(_ref_softplus)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.softplus(x_zero)
        expected_zero = np.vectorize(_ref_softplus)(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = activations.softplus(x_large_positive)
        expected_large_positive = np.vectorize(_ref_softplus)(x_large_positive)
        self.assertAllClose(
            result_large_positive, expected_large_positive, rtol=1e-5
        )

        # Test large negative values
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = activations.softplus(x_large_negative)
        expected_large_negative = np.vectorize(_ref_softplus)(x_large_negative)
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_softsign(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.softsign(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_softsign)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.softsign(x_1d)
        expected_1d = np.vectorize(_ref_softsign)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.softsign(x_3d)
        expected_3d = np.vectorize(_ref_softsign)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.softsign(x_zero)
        expected_zero = np.vectorize(_ref_softsign)(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = activations.softsign(x_large_positive)
        expected_large_positive = np.vectorize(_ref_softsign)(x_large_positive)
        self.assertAllClose(
            result_large_positive, expected_large_positive, rtol=1e-5
        )

        # Test large negative values
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = activations.softsign(x_large_negative)
        expected_large_negative = np.vectorize(_ref_softsign)(x_large_negative)
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_sigmoid(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.sigmoid(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_sigmoid)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.sigmoid(x_1d)
        expected_1d = np.vectorize(_ref_sigmoid)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.sigmoid(x_3d)
        expected_3d = np.vectorize(_ref_sigmoid)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.sigmoid(x_zero)
        expected_zero = np.vectorize(_ref_sigmoid)(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = activations.sigmoid(x_large_positive)
        expected_large_positive = np.vectorize(_ref_sigmoid)(x_large_positive)
        self.assertAllClose(
            result_large_positive, expected_large_positive, rtol=1e-5
        )

        # Test large negative values
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = activations.sigmoid(x_large_negative)
        expected_large_negative = np.vectorize(_ref_sigmoid)(x_large_negative)
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_hard_sigmoid(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.hard_sigmoid(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_hard_sigmoid)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.hard_sigmoid(x_1d)
        expected_1d = np.vectorize(_ref_hard_sigmoid)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.hard_sigmoid(x_3d)
        expected_3d = np.vectorize(_ref_hard_sigmoid)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test with strictly positive values much larger than 1
        x_positive_above_1 = np.random.uniform(
            5, 10, (2, 5)
        )  # Adjusted this range
        result_positive_above_1 = activations.hard_sigmoid(x_positive_above_1)
        expected_positive_above_1 = np.ones((2, 5))
        self.assertAllClose(
            result_positive_above_1, expected_positive_above_1, rtol=1e-5
        )

    def test_sparse_sigmoid(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.sparse_sigmoid(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_sparse_sigmoid)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.sparse_sigmoid(x_1d)
        expected_1d = np.vectorize(_ref_sparse_sigmoid)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.sparse_sigmoid(x_3d)
        expected_3d = np.vectorize(_ref_sparse_sigmoid)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = activations.sparse_sigmoid(x_large_positive)
        expected_large_positive = np.vectorize(_ref_sparse_sigmoid)(
            x_large_positive
        )
        self.assertAllClose(
            result_large_positive, expected_large_positive, rtol=1e-5
        )

        # Test large negative values
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = activations.sparse_sigmoid(x_large_negative)
        expected_large_negative = np.vectorize(_ref_sparse_sigmoid)(
            x_large_negative
        )
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_log_sigmoid(self):
        # Basic test for random values between 0 and 1
        x = np.random.uniform(0, 1, (2, 5))
        result = activations.log_sigmoid(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_log_sigmoid)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.log_sigmoid(x_1d)
        expected_1d = np.vectorize(_ref_log_sigmoid)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.log_sigmoid(x_3d)
        expected_3d = np.vectorize(_ref_log_sigmoid)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = activations.log_sigmoid(x_large_positive)
        expected_large_positive = np.vectorize(_ref_log_sigmoid)(
            x_large_positive
        )
        self.assertAllClose(
            result_large_positive, expected_large_positive, rtol=1e-5
        )

        # Test large negative values
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = activations.log_sigmoid(x_large_negative)
        expected_large_negative = np.vectorize(_ref_log_sigmoid)(
            x_large_negative
        )
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_hard_silu(self):
        # Basic test for random values between -3 and 3
        x = np.random.uniform(-3, 3, (2, 5)).astype("float32")
        result = activations.hard_silu(x[np.newaxis, :])[0]
        expected = np.vectorize(_ref_hard_silu)(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5).astype("float32")
        result_1d = activations.hard_silu(x_1d)
        expected_1d = np.vectorize(_ref_hard_silu)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3)).astype("float32")
        result_3d = activations.hard_silu(x_3d)
        expected_3d = np.vectorize(_ref_hard_silu)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test with strictly positive values much larger than 3
        x_positive_above_3 = np.random.uniform(5, 10, (2, 5)).astype("float32")
        result_positive_above_3 = activations.hard_silu(x_positive_above_3)
        expected_positive_above_3 = x_positive_above_3
        self.assertAllClose(
            result_positive_above_3, expected_positive_above_3, rtol=1e-5
        )

        # Test with strictly negative values much smaller than -3
        x_negatives = np.random.uniform(-10, -5, (2, 5)).astype("float32")
        result = activations.hard_silu(x_negatives)
        expected_zeros = np.zeros_like(x_negatives)
        self.assertAllClose(result, expected_zeros, rtol=1e-5)

    def test_relu_negative_slope(self):
        # Define the input tensor
        x = np.array([-10, -5, 0.0, 5, 10])

        # Test with only negative_slope
        result_negative_slope = activations.relu(x, negative_slope=0.5)
        expected_negative_slope = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        self.assertAllClose(
            result_negative_slope, expected_negative_slope, rtol=1e-5
        )

    def test_relu_max_value(self):
        # Define the input tensor
        x = np.array([-10, -5, 0.0, 5, 10])

        # Test with only max_value
        result_max_value = activations.relu(x, max_value=5.0)
        expected_max_value = np.array([0.0, 0.0, 0.0, 5.0, 5.0])
        self.assertAllClose(result_max_value, expected_max_value, rtol=1e-5)

    def test_relu_threshold(self):
        # Define the input tensor
        x = np.array([-10, -5, 0.0, 5, 10])

        # Test with only threshold
        result_threshold = activations.relu(x, threshold=5.0)
        expected_threshold = np.array([-0.0, -0.0, 0.0, 0.0, 10.0])
        self.assertAllClose(result_threshold, expected_threshold, rtol=1e-5)

    def test_relu_combined_threshold_and_max_value(self):
        # Define the input tensor
        x = np.array([-10, -5, 0.0, 5, 10])

        # Test with threshold and max_value
        result_combined = activations.relu(x, threshold=5.0, max_value=5.0)
        expected_combined = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
        self.assertAllClose(result_combined, expected_combined, rtol=1e-5)

    def test_relu_combined_all_parameters(self):
        # Define the input tensor
        x = np.array([-10, -5, 0.0, 5, 10])

        # Test with negative_slope, max_value, and threshold
        result_combined = activations.relu(
            x, negative_slope=0.5, max_value=5.0, threshold=5.0
        )
        expected_combined = np.array([-7.5, -5.0, -2.5, 0.0, 5.0])
        self.assertAllClose(result_combined, expected_combined, rtol=1e-5)

    def test_relu_to_trigger_relu6(self):
        x = np.array([-10, -5, 0.0, 5, 10, 12])
        result_relu6 = activations.relu(x, max_value=6.0)
        expected_relu6 = np.array([0.0, 0.0, 0.0, 5.0, 6.0, 6.0])
        self.assertAllClose(result_relu6, expected_relu6, rtol=1e-5)

    def test_relu_to_trigger_leaky(self):
        x = np.array([-10, -5, 0.0, 5, 10])
        result_leaky = activations.relu(x, negative_slope=0.5)
        expected_leaky = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        self.assertAllClose(result_leaky, expected_leaky, rtol=1e-5)

    def test_relu(self):
        # Basic test for positive values
        positive_values = np.random.uniform(0.1, 10, (2, 5))
        result = activations.relu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values, rtol=1e-5)

        # Basic test for negative values
        negative_values = np.random.uniform(-10, -0.1, (2, 5))
        result = activations.relu(negative_values[np.newaxis, :])[0]
        expected = np.zeros((2, 5))
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.relu(x_1d)
        expected_1d = np.maximum(0, x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.relu(x_3d)
        expected_3d = np.maximum(0, x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.relu(x_zero)
        expected_zero = np.maximum(0, x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large positive values
        x_large_positive = np.random.uniform(1e4, 1e5, (2, 5))
        result_large_positive = activations.relu(x_large_positive)
        self.assertAllClose(result_large_positive, x_large_positive, rtol=1e-5)

        # Test large negative values
        x_large_negative = np.random.uniform(-1e5, -1e4, (2, 5))
        result_large_negative = activations.relu(x_large_negative)
        expected_large_negative = np.zeros((2, 5))
        self.assertAllClose(
            result_large_negative, expected_large_negative, rtol=1e-5
        )

    def test_leaky_relu(self):
        leaky_relu_vectorized = np.vectorize(_ref_leaky_relu)

        # Test for negative_slope = 0.01
        # Test positive values
        positive_values = np.random.random((2, 5))
        result = activations.leaky_relu(
            positive_values[np.newaxis, :], negative_slope=0.01
        )[0]
        expected = leaky_relu_vectorized(positive_values, alpha=0.01)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.leaky_relu(
            negative_values[np.newaxis, :], negative_slope=0.01
        )[0]
        expected = leaky_relu_vectorized(negative_values, alpha=0.01)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test for negative_slope = 0.3
        # Test positive values
        positive_values = np.random.random((2, 5))
        result = activations.leaky_relu(
            positive_values[np.newaxis, :], negative_slope=0.3
        )[0]
        expected = leaky_relu_vectorized(positive_values, alpha=0.3)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.leaky_relu(
            negative_values[np.newaxis, :], negative_slope=0.3
        )[0]
        expected = leaky_relu_vectorized(negative_values, alpha=0.3)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_relu6(self):
        relu6_vectorized = np.vectorize(_ref_relu6)

        # Test positive values less than 6
        positive_values = np.random.uniform(0, 5.9, (2, 5))
        result = activations.relu6(positive_values[np.newaxis, :])[0]
        expected = relu6_vectorized(positive_values)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test positive values greater than 6
        positive_values_above_6 = np.random.uniform(6.1, 10, (2, 5))
        result = activations.relu6(positive_values_above_6[np.newaxis, :])[0]
        expected = relu6_vectorized(positive_values_above_6)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.relu6(negative_values[np.newaxis, :])[0]
        expected = relu6_vectorized(negative_values)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_silu(self):
        silu_vectorized = np.vectorize(_ref_silu)

        # Test positive values
        positive_values = np.random.uniform(0, 5.9, (2, 5))
        result = activations.silu(positive_values[np.newaxis, :])[0]
        expected = silu_vectorized(positive_values)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test values around zero (to ensure sigmoid behaves correctly)
        around_zero_values = np.random.uniform(-1, 1, (2, 5))
        result = activations.silu(around_zero_values[np.newaxis, :])[0]
        expected = silu_vectorized(around_zero_values)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test negative values
        negative_values = np.random.uniform(-5.9, 0, (2, 5))
        result = activations.silu(negative_values[np.newaxis, :])[0]
        expected = silu_vectorized(negative_values)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_gelu(self):
        def gelu(x, approximate=False):
            if approximate:
                return (
                    0.5
                    * x
                    * (
                        1.0
                        + np.tanh(
                            np.sqrt(2.0 / np.pi)
                            * (x + 0.044715 * np.power(x, 3))
                        )
                    )
                )
            else:
                from scipy.stats import norm

                return x * norm.cdf(x)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :])[0]
        expected = gelu(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :], approximate=True)[0]
        expected = gelu(x, True)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_celu(self):
        def celu(x, alpha=1.0):
            return np.maximum(x, 0.0) + alpha * np.expm1(
                np.minimum(x, 0.0) / alpha
            )

        x = np.random.random((2, 5))
        result = activations.celu(x[np.newaxis, :])[0]
        expected = celu(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        x = np.random.random((2, 5))
        result = activations.celu(x[np.newaxis, :], alpha=0.5)[0]
        expected = celu(x, alpha=0.5)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_glu(self):
        def glu(x, axis=-1):
            x1, x2 = np.split(x, 2, axis)
            return x1 * (1 / (1 + np.exp(-x2)))

        x = np.random.random((2, 4))
        result = activations.glu(x[np.newaxis, :])[0]
        expected = glu(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        x = np.random.random((2, 4))
        result = activations.glu(x[np.newaxis, :], axis=-2)[0]
        expected = glu(x, axis=-2)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_tanh_shrink(self):
        def tanh_shrink(x):
            return x - np.tanh(x)

        x = np.random.random((2, 5))
        result = activations.tanh_shrink(x[np.newaxis, :])[0]
        expected = tanh_shrink(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_hard_tanh(self):
        def hard_tanh(x):
            return np.clip(x, -1.0, 1.0)

        x = np.random.random((2, 5))
        result = activations.hard_tanh(x[np.newaxis, :])[0]
        expected = hard_tanh(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_hard_shrink(self):
        def hard_shrink(x):
            return np.where(np.abs(x) > 0.5, x, 0.0)

        x = np.random.random((2, 5))
        result = activations.hard_shrink(x[np.newaxis, :])[0]
        expected = hard_shrink(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_threshold(self):
        def threshold(x, threshold_value, value):
            return np.where(
                x > threshold_value, x, np.array(value, dtype=x.dtype)
            )

        x = np.random.random((2, 5))
        result = activations.threshold(x[np.newaxis, :], 0, 0)[0]
        expected = threshold(x, 0, 0)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_squareplus(self):
        def squareplus(x, b=4):
            y = x + np.sqrt(x**2 + b)
            return y / 2

        x = np.random.random((2, 5))
        result = activations.squareplus(x[np.newaxis, :])[0]
        expected = squareplus(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_soft_shrink(self):
        def soft_shrink(x, threshold=0.5):
            return np.where(
                x > threshold,
                x - threshold,
                np.where(x < -threshold, x + threshold, 0.0),
            )

        x = np.random.random((2, 5))
        result = activations.soft_shrink(x[np.newaxis, :])[0]
        expected = soft_shrink(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_sparse_plus(self):
        def sparse_plus(x):
            return np.where(
                x <= -1,
                np.zeros_like(x),
                np.where(x < 1, (1 / 4) * (x + 1) ** 2, x),
            )

        x = np.random.random((2, 5))
        result = activations.sparse_plus(x[np.newaxis, :])[0]
        expected = sparse_plus(x)
        self.assertAllClose(result, expected, rtol=1e-5)

    def test_elu(self):
        x = np.random.random((2, 5))
        result = activations.elu(x[np.newaxis, :])[0]
        self.assertAllClose(result, x, rtol=1e-5)
        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.elu(negative_values[np.newaxis, :])[0]
        true_result = np.exp(negative_values) - 1
        self.assertAllClose(result, true_result)

    def test_tanh(self):
        # Basic test for the tanh activation function
        x = np.random.random((2, 5))
        result = activations.tanh(x[np.newaxis, :])[0]
        expected = np.tanh(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Basic test for the tanh activation function
        x = np.random.uniform(-10, 10, (2, 5))
        result = activations.tanh(x[np.newaxis, :])[0]
        expected = np.tanh(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.tanh(x_1d)
        expected_1d = np.tanh(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.tanh(x_3d)
        expected_3d = np.tanh(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test with strictly positive values
        x_positive = np.random.uniform(0, 10, (2, 5))
        result_positive = activations.tanh(x_positive)
        expected_positive = np.tanh(x_positive)
        self.assertAllClose(result_positive, expected_positive, rtol=1e-5)

        # Test with strictly negative values
        x_negative = np.random.uniform(-10, 0, (2, 5))
        result_negative = activations.tanh(x_negative)
        expected_negative = np.tanh(x_negative)
        self.assertAllClose(result_negative, expected_negative, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.tanh(x_zero)
        expected_zero = np.tanh(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large values to check stability
        x_large = np.random.uniform(1e4, 1e5, (2, 5))
        result_large = activations.tanh(x_large)
        expected_large = np.tanh(x_large)
        self.assertAllClose(result_large, expected_large, rtol=1e-5)

    def test_exponential(self):
        # Basic test for the exponential activation function
        x = np.random.random((2, 5))
        result = activations.exponential(x[np.newaxis, :])[0]
        expected = np.exp(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        x = np.random.uniform(-10, 10, (2, 5))
        result = activations.exponential(x[np.newaxis, :])[0]
        expected = np.exp(x)
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.exponential(x_1d)
        expected_1d = np.exp(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.exponential(x_3d)
        expected_3d = np.exp(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test with strictly positive values
        x_positive = np.random.uniform(0, 10, (2, 5))
        result_positive = activations.exponential(x_positive)
        expected_positive = np.exp(x_positive)
        self.assertAllClose(result_positive, expected_positive, rtol=1e-5)

        # Test with strictly negative values
        x_negative = np.random.uniform(-10, 0, (2, 5))
        result_negative = activations.exponential(x_negative)
        expected_negative = np.exp(x_negative)
        self.assertAllClose(result_negative, expected_negative, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.exponential(x_zero)
        expected_zero = np.exp(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large values to check stability
        x_large = np.random.uniform(1e4, 1e5, (2, 5))
        result_large = activations.exponential(x_large)
        expected_large = np.exp(x_large)
        self.assertAllClose(result_large, expected_large, rtol=1e-5)

    def test_mish(self):
        # Basic test for the mish activation function
        x = np.random.random((2, 5))
        result = activations.mish(x[np.newaxis, :])[0]
        expected = x * np.tanh(_ref_softplus(x))
        self.assertAllClose(result, expected, rtol=1e-5)

        x = np.random.uniform(-10, 10, (2, 5))
        result = activations.mish(x[np.newaxis, :])[0]
        expected = x * np.tanh(_ref_softplus(x))
        self.assertAllClose(result, expected, rtol=1e-5)

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = activations.mish(x_1d)
        expected_1d = x_1d * np.tanh(_ref_softplus(x_1d))
        self.assertAllClose(result_1d, expected_1d, rtol=1e-5)

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = activations.mish(x_3d)
        expected_3d = x_3d * np.tanh(_ref_softplus(x_3d))
        self.assertAllClose(result_3d, expected_3d, rtol=1e-5)

        # Test with strictly positive values
        x_positive = np.random.uniform(0, 10, (2, 5))
        result_positive = activations.mish(x_positive)
        expected_positive = x_positive * np.tanh(_ref_softplus(x_positive))
        self.assertAllClose(result_positive, expected_positive, rtol=1e-5)

        # Test with strictly negative values
        x_negative = np.random.uniform(-10, 0, (2, 5))
        result_negative = activations.mish(x_negative)
        expected_negative = x_negative * np.tanh(_ref_softplus(x_negative))
        self.assertAllClose(result_negative, expected_negative, rtol=1e-5)

        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = activations.mish(x_zero)
        expected_zero = x_zero * np.tanh(_ref_softplus(x_zero))
        self.assertAllClose(result_zero, expected_zero, rtol=1e-5)

        # Test large values to check stability
        x_large = np.random.uniform(1e4, 1e5, (2, 5))
        result_large = activations.mish(x_large)
        expected_large = x_large * np.tanh(_ref_softplus(x_large))
        self.assertAllClose(result_large, expected_large, rtol=1e-5)

    def test_linear(self):
        x = np.random.random((10, 5))
        self.assertAllClose(x, activations.linear(x))

        # Test with 1D array
        x_1d = np.random.uniform(-10, 10, 5)
        self.assertAllClose(x_1d, activations.linear(x_1d))

        # Test with 2D array
        x = np.random.uniform(-10, 10, (10, 5))
        self.assertAllClose(x, activations.linear(x))

        # Test with 3D array
        x_3d = np.random.uniform(-10, 10, (5, 5, 5))
        self.assertAllClose(x_3d, activations.linear(x_3d))

        # Test with float32 data type
        x_float32 = np.random.uniform(-10, 10, (10, 5)).astype(np.float32)
        self.assertAllClose(x_float32, activations.linear(x_float32))
        # Test with int32 data type
        x_int32 = np.random.randint(-10, 10, (10, 5)).astype(np.int32)
        self.assertAllClose(x_int32, activations.linear(x_int32))

    def test_sparsemax(self):
        # result check with 1d
        x_1d = np.linspace(1, 12, num=12)
        expected_result = np.zeros_like(x_1d)
        expected_result[-1] = 1.0
        self.assertAllClose(activations.sparsemax(x_1d), expected_result)

        # result check with 2d
        x_2d = np.linspace(1, 12, num=12).reshape(-1, 2)
        expected_result = np.zeros_like(x_2d)
        expected_result[:, -1] = 1.0
        self.assertAllClose(activations.sparsemax(x_2d), expected_result)

        # result check with 3d
        x_3d = np.linspace(1, 12, num=12).reshape(-1, 1, 3)
        expected_result = np.zeros_like(x_3d)
        expected_result[:, :, -1] = 1.0
        self.assertAllClose(activations.sparsemax(x_3d), expected_result)

        # result check with axis=-2 with 2d input
        x_2d = np.linspace(1, 12, num=12).reshape(-1, 2)
        expected_result = np.zeros_like(x_2d)
        expected_result[-1, :] = 1.0
        self.assertAllClose(
            activations.sparsemax(x_2d, axis=-2), expected_result
        )

        # result check with axis=-2 with 3d input
        x_3d = np.linspace(1, 12, num=12).reshape(-1, 1, 3)
        expected_result = np.ones_like(x_3d)
        self.assertAllClose(
            activations.sparsemax(x_3d, axis=-2), expected_result
        )

        # result check with axis=-3 with 3d input
        x_3d = np.linspace(1, 12, num=12).reshape(-1, 1, 3)
        expected_result = np.zeros_like(x_3d)
        expected_result[-1, :, :] = 1.0
        self.assertAllClose(
            activations.sparsemax(x_3d, axis=-3), expected_result
        )

        # Closely spaced logits keep more than one coordinate in the
        # support. Every case above expects a one hot result, which only
        # exercises a support of size one.
        x_dense = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected_result = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertAllClose(activations.sparsemax(x_dense), expected_result)

        # sparsemax projects onto the probability simplex, so whatever the
        # support size, each slice along the axis sums to one.
        x_2d = np.array([[0.1, 0.2, 0.3], [-1.0, 0.5, 0.4]])
        self.assertAllClose(
            np.sum(
                backend.ops.convert_to_numpy(activations.sparsemax(x_2d)),
                axis=-1,
            ),
            np.ones(2),
        )

        # result check with axis=-3 with 4d input
        x_4d = np.linspace(1, 12, num=12).reshape(-1, 1, 1, 2)
        expected_result = np.ones_like(x_4d)
        self.assertAllClose(
            activations.sparsemax(x_4d, axis=-3), expected_result
        )

    def test_get_method(self):
        obj = activations.get("relu")
        self.assertEqual(obj, activations.relu)

        obj = activations.get(None)
        self.assertEqual(obj, activations.linear)

        with self.assertRaises(ValueError):
            activations.get("typo")

    def test_serialize_deserialize_built_in_activation(self):
        serialized = activations.serialize(activations.gelu)
        self.assertEqual(serialized, "gelu")

        deserialized = activations.deserialize(serialized)
        self.assertIs(deserialized, activations.gelu)

    def test_serialize_deserialize_op_activation(self):
        serialized = activations.serialize(ops.gelu)
        self.assertIsInstance(serialized, dict)

        deserialized = activations.deserialize(serialized)
        self.assertIs(deserialized, ops.gelu)
        self.assertIsNot(deserialized, activations.gelu)

    def test_serialize_deserialize_registered_function(self):
        @object_registration.register_keras_serializable(package="serial_test")
        def gelu(x):
            return x

        serialized = activations.serialize(gelu)
        self.assertIsInstance(serialized, dict)

        deserialized = activations.deserialize(serialized)
        self.assertIs(deserialized, gelu)

    def test_serialize_deserialize_unregistered_function(self):
        def gelu(x):
            return x

        serialized = activations.serialize(gelu)
        self.assertIsInstance(serialized, dict)

        deserialized = activations.deserialize(
            serialized, custom_objects={"gelu": gelu}
        )
        self.assertIs(deserialized, gelu)
