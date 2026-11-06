import json
import os

import numpy as np
from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantization_config import validate_and_resolve_config
from keras.src.quantizers.quantizers import AbsMaxQuantizer


class QuantizationConfigTest(testing.TestCase):
    def test_base_quantization_config(self):
        config = QuantizationConfig()
        with self.assertRaises(NotImplementedError):
            _ = config.mode

    def test_int8_quantization_config_valid(self):
        config = Int8QuantizationConfig()
        self.assertEqual(config.mode, "int8")
        self.assertIsNone(config.weight_quantizer)

        # Valid weight quantizer
        q = AbsMaxQuantizer(axis=0, value_range=(-127, 127))
        config = Int8QuantizationConfig(weight_quantizer=q)
        self.assertEqual(config.weight_quantizer, q)

    def test_int8_quantization_config_invalid(self):
        # Invalid value_range
        with self.assertRaisesRegex(ValueError, "value_range"):
            AbsMaxQuantizer(axis=0, value_range=(-256, 256))

    def test_int4_quantization_config_valid(self):
        config = Int4QuantizationConfig()
        self.assertEqual(config.mode, "int4")
        self.assertIsNone(config.weight_quantizer)

        # Valid weight quantizer with per-channel mode
        # (custom quantizers require block_size=None or -1)
        q = AbsMaxQuantizer(axis=0, value_range=(-8, 7))
        config = Int4QuantizationConfig(weight_quantizer=q, block_size=None)
        self.assertEqual(config.weight_quantizer, q)

    def test_int4_quantization_config_invalid(self):
        # Invalid value_range
        q = AbsMaxQuantizer(axis=0, value_range=(-127, 127))
        with self.assertRaisesRegex(ValueError, "value_range"):
            Int4QuantizationConfig(weight_quantizer=q, block_size=None)

    def test_int4_quantization_config_subchannel_rejects_custom_quantizer(self):
        # Sub-channel quantization does not support custom quantizers
        weight_q = AbsMaxQuantizer(axis=0, value_range=(-8, 7))
        activation_q = AbsMaxQuantizer(axis=-1)

        # Default block_size=128 is sub-channel, should reject custom quantizer
        with self.assertRaisesRegex(
            ValueError, "sub-channel quantization.*does not support"
        ):
            Int4QuantizationConfig(weight_quantizer=weight_q)

        # Explicit positive block_size should also reject weight quantizer
        with self.assertRaisesRegex(
            ValueError, "sub-channel quantization.*does not support"
        ):
            Int4QuantizationConfig(weight_quantizer=weight_q, block_size=64)

        # Sub-channel should also reject activation quantizer
        with self.assertRaisesRegex(
            ValueError, "sub-channel quantization.*does not support"
        ):
            Int4QuantizationConfig(activation_quantizer=activation_q)

        with self.assertRaisesRegex(
            ValueError, "sub-channel quantization.*does not support"
        ):
            Int4QuantizationConfig(
                activation_quantizer=activation_q, block_size=64
            )

        # Per-channel (block_size=None or -1) should accept custom quantizers
        config = Int4QuantizationConfig(
            weight_quantizer=weight_q, block_size=None
        )
        self.assertEqual(config.weight_quantizer, weight_q)

        config = Int4QuantizationConfig(
            weight_quantizer=weight_q, block_size=-1
        )
        self.assertEqual(config.weight_quantizer, weight_q)

        config = Int4QuantizationConfig(
            activation_quantizer=activation_q, block_size=None
        )
        self.assertEqual(config.activation_quantizer, activation_q)

        config = Int4QuantizationConfig(
            activation_quantizer=activation_q, block_size=-1
        )
        self.assertEqual(config.activation_quantizer, activation_q)

    def test_quantization_config_serialization(self):
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0),
            activation_quantizer=AbsMaxQuantizer(axis=-1),
        )
        serialized = config.get_config()
        deserialized = Int8QuantizationConfig.from_config(serialized)
        self.assertIsInstance(deserialized, Int8QuantizationConfig)
        self.assertIsInstance(deserialized.weight_quantizer, AbsMaxQuantizer)
        self.assertIsInstance(
            deserialized.activation_quantizer, AbsMaxQuantizer
        )
        self.assertEqual(deserialized.weight_quantizer.axis, (0,))
        self.assertEqual(deserialized.activation_quantizer.axis, (-1,))

    def test_validate_and_resolve_config(self):
        # 1. String mode
        config = validate_and_resolve_config("int8", None)
        self.assertIsInstance(config, Int8QuantizationConfig)
        self.assertEqual(config.mode, "int8")

        config = validate_and_resolve_config("int4", None)
        self.assertIsInstance(config, Int4QuantizationConfig)
        self.assertEqual(config.mode, "int4")

        # 2. Config object
        config_in = Int8QuantizationConfig()
        config_out = validate_and_resolve_config(None, config_in)
        self.assertIs(config_out, config_in)

        # 3. Mode + Config (matching)
        config_in = Int8QuantizationConfig()
        config_out = validate_and_resolve_config("int8", config_in)
        self.assertIs(config_out, config_in)

        # 4. Mode + Config (mismatch)
        config_in = Int8QuantizationConfig()
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("int4", config_in)

        # 5. Invalid mode
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode"):
            validate_and_resolve_config("invalid_mode", None)

        # 6. GPTQ without config
        with self.assertRaisesRegex(ValueError, "must be of type `GPTQConfig`"):
            validate_and_resolve_config("gptq", None)

        # 7. Contradictory config
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("gptq", Int8QuantizationConfig())

        # 8. GPTQ with invalid config type (but correct mode)
        class FakeGPTQConfig(QuantizationConfig):
            @property
            def mode(self):
                return "gptq"

        with self.assertRaisesRegex(ValueError, "requires a valid `config`"):
            validate_and_resolve_config("gptq", FakeGPTQConfig())

    def test_int8_quantization_config_output_dtype_mismatch(self):
        # Invalid output_dtype
        q = AbsMaxQuantizer(
            axis=0, value_range=(-127, 127), output_dtype="int16"
        )
        with self.assertRaisesRegex(ValueError, "output_dtype='int8'"):
            Int8QuantizationConfig(weight_quantizer=q)

    def test_int4_quantization_config_output_dtype_mismatch(self):
        # Invalid output_dtype (using per-channel mode to test output_dtype)
        q = AbsMaxQuantizer(axis=0, value_range=(-8, 7), output_dtype="int16")
        with self.assertRaisesRegex(ValueError, "output_dtype='int8'"):
            Int4QuantizationConfig(weight_quantizer=q, block_size=None)

    def test_model_save_and_load(self):
        """
        Test custom quantizer serialization for model save and load.
        """
        # Setup
        weight_range = [-100, 100]
        custom_quantizer = AbsMaxQuantizer(axis=0, value_range=weight_range)
        config = Int8QuantizationConfig(
            weight_quantizer=custom_quantizer,
            activation_quantizer=None,
        )

        layer = layers.Dense(10)
        layer.build((None, 5))
        layer.quantize("int8", config=config)

        model = models.Sequential([layer])
        model.build((None, 5))

        # Save to temp file
        filepath = os.path.join(self.get_temp_dir(), "quantized_model.keras")
        model.save(filepath)

        # Load back
        loaded_model = saving.load_model(filepath)

        # Verify
        loaded_layer = loaded_model.layers[0]
        self.assertIsInstance(
            loaded_layer.quantization_config, Int8QuantizationConfig
        )

        quantizer = loaded_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertEqual(quantizer.axis, (0,))
        self.assertAllEqual(quantizer.value_range, weight_range)
        self.assertIsNone(loaded_layer.quantization_config.activation_quantizer)
        self.assertTrue(loaded_layer._is_quantized)

    def test_awq_requires_config(self):
        """Test that AWQ mode requires a config."""
        with self.assertRaisesRegex(ValueError, "AWQConfig"):
            validate_and_resolve_config("awq", None)

    def test_awq_requires_correct_config_type(self):
        """Test that AWQ requires AWQConfig type."""
        # Int8QuantizationConfig has mode='int8', so passing mode='awq' raises
        # a contradictory arguments error
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("awq", Int8QuantizationConfig())


class Int4ConfigListValueRangeTest(testing.TestCase):
    """JSON stores the `value_range` tuple of a weight quantizer as a list."""

    def test_list_value_range(self):
        q = AbsMaxQuantizer(axis=0, value_range=[-8, 7])
        config = Int4QuantizationConfig(weight_quantizer=q, block_size=None)
        self.assertIs(config.weight_quantizer, q)

    def test_config_json_round_trip(self):
        config = Int4QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0, value_range=(-8, 7)),
            block_size=None,
        )
        restored = Int4QuantizationConfig.from_config(
            json.loads(json.dumps(config.get_config()))
        )
        self.assertIsNone(restored.block_size)
        self.assertAllEqual(restored.weight_quantizer.value_range, (-8, 7))

    def test_per_channel_custom_quantizer_save_and_load(self):
        config = Int4QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0, value_range=(-8, 7)),
            block_size=-1,
        )
        inputs = layers.Input((8,))
        model = models.Model(inputs, layers.Dense(6)(inputs))
        model.quantize(config=config)
        x = np.random.default_rng(0).normal(size=(2, 8)).astype("float32")
        y = model(x)

        filepath = os.path.join(self.get_temp_dir(), "int4_model.keras")
        model.save(filepath)
        loaded_model = saving.load_model(filepath)

        loaded_config = loaded_model.layers[1].quantization_config
        self.assertIsInstance(loaded_config, Int4QuantizationConfig)
        self.assertEqual(loaded_config.block_size, -1)
        self.assertAllEqual(loaded_config.weight_quantizer.value_range, (-8, 7))
        self.assertAllEqual(loaded_model(x), y)


_DATASET = [np.ones((1, 4), "float32")]


class ConfigMatchesPolicyTest(testing.TestCase):
    @parameterized.named_parameters(
        (
            "other_mode",
            "int8_from_float32",
            lambda: Int4QuantizationConfig(block_size=8),
            "quantization_config.mode='int4'",
        ),
        (
            # float8 keeps the float kernel, which `Dense.build` adds
            # before the mode's variables.
            "float8_other_mode",
            "float8_from_float32",
            lambda: Int8QuantizationConfig(),
            "quantization_config.mode='int8'",
        ),
        (
            "other_block_size",
            "int4/4_from_float32",
            lambda: Int4QuantizationConfig(block_size=8),
            "quantization_config.block_size=8",
        ),
        (
            "grouped_config_per_channel_policy",
            "int4/-1_from_float32",
            lambda: Int4QuantizationConfig(block_size=8),
            "quantization_config.block_size=8",
        ),
        (
            "other_weight_bits",
            "gptq/4/8_from_float32",
            lambda: GPTQConfig(
                dataset=_DATASET, tokenizer=None, weight_bits=2, group_size=8
            ),
            "quantization_config.weight_bits=2",
        ),
        (
            "other_group_size",
            "awq/4/8_from_float32",
            lambda: AWQConfig(dataset=_DATASET, tokenizer=None, group_size=4),
            "quantization_config.group_size=4",
        ),
    )
    def test_build_refuses_a_contradictory_config(
        self, policy, make_config, message
    ):
        layer = layers.Dense(8, dtype=policy, quantization_config=make_config())
        with self.assertRaisesRegex(
            ValueError, f"Contradictory arguments: dtype_policy='{policy}'"
        ) as cm:
            layer.build((None, 16))
        self.assertIn(message, str(cm.exception))
        # The refusal comes before the mode creates a variable.
        self.assertFalse(layer.built)
        self.assertEqual(layer.variables, [])

    def test_map_entry_is_checked_at_build(self):
        # The layer's own entry of a `DTypePolicyMap` is the policy that
        # the config must agree with.
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["target"] = "int4/4_from_float32"
        layer = layers.Dense(
            8,
            name="target",
            dtype=policy_map,
            quantization_config=Int4QuantizationConfig(block_size=8),
        )
        with self.assertRaisesRegex(
            ValueError,
            "Contradictory arguments: dtype_policy='int4/4_from_float32' but "
            "quantization_config.block_size=8",
        ):
            layer.build((None, 16))
        self.assertEqual(layer.variables, [])

    def test_policy_set_after_the_config_is_checked_at_build(self):
        layer = layers.Dense(
            8, quantization_config=Int4QuantizationConfig(block_size=8)
        )
        layer.dtype_policy = "int4/4_from_float32"
        with self.assertRaisesRegex(
            ValueError,
            "Contradictory arguments: dtype_policy='int4/4_from_float32' but "
            "quantization_config.block_size=8",
        ):
            layer.build((None, 16))
        self.assertEqual(layer.variables, [])

    @parameterized.named_parameters(
        ("grouped", "int4/8_from_float32", 8),
        ("per_channel_none", "int4/-1_from_float32", None),
        ("per_channel_legacy_none", "int4/None_from_float32", -1),
        # A legacy bare "int4" name states no block size, so the config
        # gives it, as in a file that the policy setter of Keras 3.14 and
        # 3.15 wrote.
        ("legacy_bare", "int4_from_float32", 8),
    )
    def test_agreeing_config_builds_and_reloads(self, policy, block_size):
        config = Int4QuantizationConfig(block_size=block_size)
        inputs = layers.Input((16,))
        layer = layers.Dense(8, dtype=policy, quantization_config=config)
        model = models.Model(inputs, layer(inputs))
        x = np.random.default_rng(0).standard_normal((2, 16))
        groups = 1 if block_size in (None, -1) else 16 // block_size
        expected = (8,) if groups == 1 else (groups, 8)
        self.assertEqual(tuple(layer.kernel_scale.shape), expected)

        filepath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(filepath)
        restored = saving.load_model(filepath)
        self.assertAllEqual(restored(x), model(x))

        if policy != "int4_from_float32":
            # The name alone builds the same variables.
            policy_map = dtype_policies.DTypePolicyMap()
            policy_map[layer.path] = policy
            twin = layers.Dense(8, dtype=policy_map, name=layer.name)
            twin.build((None, 16))
            store = {}
            layer.save_own_variables(store)
            twin.load_own_variables(store)
            self.assertAllEqual(twin(x), layer(x))
