import warnings

import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend import config
from keras.src.testing.test_utils import x64_mode_scope

X64_SUPPORTED = backend.backend() in config._X64_BACKENDS


class X64ModeTest(testing.TestCase):
    def test_default_is_off(self):
        self.assertEqual(config.x64_mode(), "off")

    @pytest.mark.skipif(not X64_SUPPORTED, reason="Unsupported backend.")
    def test_set_and_restore(self):
        with x64_mode_scope("explicit_x64"):
            self.assertEqual(config.x64_mode(), "explicit_x64")
        self.assertEqual(config.x64_mode(), "off")

    def test_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown `x64_mode`"):
            config.set_x64_mode("bogus")

    def test_enable_x64_is_not_supported_yet(self):
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            config.set_x64_mode("enable_x64")

    @pytest.mark.skipif(X64_SUPPORTED, reason="Supported backend.")
    def test_unsupported_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "not supported on the"):
            config.set_x64_mode("explicit_x64")
        self.assertEqual(config.x64_mode(), "off")

    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only.")
    def test_explicit_x64_sets_jax_flag(self):
        import jax

        previous = jax.config.jax_explicit_x64_dtypes
        try:
            config.set_x64_mode("explicit_x64")
            self.assertEqual(jax.config.jax_explicit_x64_dtypes.name, "ALLOW")
        finally:
            jax.config.update("jax_explicit_x64_dtypes", previous)
            config.set_x64_mode("off")

    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only.")
    def test_matching_jax_flag_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*does not match")
            # The scope sets JAX's flag first, like a user who sets it
            # before calling `set_x64_mode`.
            with x64_mode_scope("explicit_x64"):
                config.set_x64_mode("explicit_x64")

    @parameterized.named_parameters(
        ("allow_off", "jax_explicit_x64_dtypes", "allow", "off", "int32"),
        ("enable_off", "jax_enable_x64", True, "off", "int32"),
        ("enable_explicit", "jax_enable_x64", True, "explicit_x64", "int64"),
    )
    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only.")
    def test_mismatched_jax_flag_warns(self, flag, value, mode, expected):
        import jax

        previous = getattr(jax.config, flag)
        jax.config.update(flag, value)
        try:
            with self.assertWarnsRegex(UserWarning, f"`jax.config.{flag}`"):
                config.set_x64_mode(mode)
            # Keras keeps its own mode and does not follow JAX's flags.
            self.assertEqual(config.x64_mode(), mode)
            self.assertEqual(backend.result_type("int64", "int64"), expected)
        finally:
            jax.config.update(flag, previous)
            config.set_x64_mode("off")
