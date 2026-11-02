from keras.src import testing
from keras.src.api_export import REGISTERED_NAMES_TO_OBJS


class ApiExportTest(testing.TestCase):
    def test_exported_api_matches_symbol_name(self):
        """Verify export names and real names match.

        The canonical exported name and the name of the actual function or class
        should match. Any discrepancy would be unintentional and confusing.
        """
        for full_name, symbol in REGISTERED_NAMES_TO_OBJS.items():
            # `LegacyOptimizerWarning` is a dummy umbrella class for unsupported
            # legacy optimizers.
            if symbol.__name__ == "LegacyOptimizerWarning":
                continue
            name = full_name.rsplit(".", 1)[1]
            self.assertEqual(
                name,
                symbol.__name__,
                f"Mismatched name for symbol '{full_name}'",
            )
