import numpy as np
from scipy.stats import pearsonr

from keras.src import testing
from keras.src.metrics import ConcordanceCorrelation
from keras.src.metrics import PearsonCorrelation
from keras.src.metrics import correlation_metrics


class CorrelationsTest(testing.TestCase):
    def _get_data(self):
        # Sample data for testing
        y_true = np.array(
            [[0, 1, 0.5], [1, 1, 0.2], [1, 1, 0.1], [0.1, 0.7, 0.0]],
            dtype="float32",
        )
        y_pred = np.array(
            [[0.1, 0.9, 0.5], [1, 0.9, 0.2], [0.2, 0.8, 0], [0.3, 0.3, 0.9]],
            dtype="float32",
        )

        ccc_expected = np.array(
            [0.97560976, 0.98765432, 0.46511628, -0.46376812]
        )
        # pcc_expected = np.array([1, 0.99339927, 0.69337525, -0.60999428])
        pcc_expected = np.array(
            [pearsonr(yt, yp).statistic for yt, yp in zip(y_true, y_pred)]
        )
        return y_true, y_pred, ccc_expected, pcc_expected

    def test_pearson_function(self):
        """Test the functional API for Pearson Correlation Coefficient."""
        y_true, y_pred, _, pcc_expected = self._get_data()
        result = correlation_metrics.pearson_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, pcc_expected)

    def test_concordance_function(self):
        """Test the functional API for Concordance Correlation Coefficient."""
        y_true, y_pred, ccc_expected, _ = self._get_data()
        result = correlation_metrics.concordance_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, ccc_expected)

    def test_pearson_class(self):
        """Test the PearsonCorrelation metric class."""
        y_true, y_pred, _, pcc_expected = self._get_data()
        m = PearsonCorrelation(axis=-1, dtype="float32")
        m.update_state(y_true[:2], y_pred[:2])
        self.assertAllClose(m.result(), np.mean(pcc_expected[:2]))
        m.update_state(y_true[2:], y_pred[2:])
        self.assertAllClose(m.result(), np.mean(pcc_expected))

    def test_concordance_class(self):
        """Test the ConcordanceCorrelation metric class."""
        y_true, y_pred, ccc_expected, _ = self._get_data()
        m = ConcordanceCorrelation(axis=-1, dtype="float32")
        m.update_state(y_true[:2], y_pred[:2])
        self.assertAllClose(m.result(), np.mean(ccc_expected[:2]))
        m.update_state(y_true[2:], y_pred[2:])
        self.assertAllClose(m.result(), np.mean(ccc_expected))

    def test_pearson_config(self):
        """Test the get_config method for PearsonCorrelation."""
        m = PearsonCorrelation(axis=-1, dtype="float16")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float16")
        self.assertEqual(config["name"], "pearson_correlation")

    def test_concordance_config(self):
        """Test the get_config method for ConcordanceCorrelation."""
        m = ConcordanceCorrelation(axis=-1, dtype="float32")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float32")
        self.assertEqual(config["name"], "concordance_correlation")

    def test_pearson_zero_variance_whole_batch_is_forced_to_zero(self):
        """A single, wholly-constant row: OPTION B forces the undefined
        correlation to 0.0 (matching sklearn's r_regression(force_finite=
        True) default and this file's own concordance_correlation), rather
        than propagating NaN."""
        y_true = np.array([[1.0, 1.0, 1.0]], dtype="float32")
        y_pred = np.array([[0.1, 0.5, 0.9]], dtype="float32")
        result = correlation_metrics.pearson_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, [0.0])
        self.assertFalse(np.isnan(np.array(result)).any())

    def test_pearson_one_degenerate_row_diluted_not_poisoned(self):
        """One constant row among otherwise-normal rows: OPTION B forces
        that row's contribution to 0.0, so the aggregated batch mean is
        the (real_row + 0.0) / 2 average -- diluted toward zero, but a
        finite, computable number, and a later normal batch is NOT
        poisoned (unlike Option A's NaN, which persists across
        update_state calls once seen)."""
        y_true_normal = np.array([[0.1, 0.5, 0.9]], dtype="float32")
        y_pred_normal = np.array([[0.2, 0.4, 0.8]], dtype="float32")
        y_true_degenerate = np.array([[2.0, 2.0, 2.0]], dtype="float32")
        y_pred_degenerate = np.array([[1.9, 2.1, 2.0]], dtype="float32")

        real_row_correlation = correlation_metrics.pearson_correlation(
            y_true_normal, y_pred_normal, axis=-1
        )

        m = PearsonCorrelation(axis=-1, dtype="float32")
        m.update_state(y_true_normal, y_pred_normal)
        m.update_state(y_true_degenerate, y_pred_degenerate)
        expected_diluted = np.array(real_row_correlation).mean() / 2
        self.assertAllClose(m.result(), expected_diluted)
        self.assertFalse(np.isnan(np.array(m.result())).any())

        # A later, perfectly normal batch keeps moving the mean normally --
        # no permanent poisoning.
        m.update_state(y_true_normal, y_pred_normal)
        self.assertFalse(np.isnan(np.array(m.result())).any())

    def test_pearson_normal_variance_control_unchanged(self):
        """Control: no degenerate rows at all -- OPTION B must not change
        the ordinary-case numeric result versus the original
        implementation (regression check against scipy.stats.pearsonr,
        same fixture as test_pearson_function)."""
        y_true, y_pred, _, pcc_expected = self._get_data()
        result = correlation_metrics.pearson_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, pcc_expected)
