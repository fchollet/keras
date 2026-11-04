import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import set_keras_mask
from keras.src.backend.config import disable_flash_attention
from keras.src.backend.config import enable_flash_attention
from keras.src.backend.config import is_flash_attention_enabled


class GroupedQueryAttentionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Flash attention is a newly introduced feature. We need to disable it
        # for testing purposes.
        disable_flash_attention()

    def tearDown(self):
        super().tearDown()
        enable_flash_attention()

    def test_basics(self):
        self.assertFalse(is_flash_attention_enabled())
        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "use_bias": False,
                "dropout": 0.5,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "use_gate": True,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=10,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

        self.run_layer_test(
            layers.GroupedQueryAttention,
            init_kwargs={
                "num_query_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "use_bias": False,
                "dropout": 0.5,
                "use_gate": True,
            },
            input_shape={"query_shape": (2, 8, 16), "value_shape": (2, 4, 16)},
            expected_output_shape=(2, 8, 16),
            expected_num_trainable_weights=5,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    @pytest.mark.skipif(
        backend.backend() not in ("jax", "torch"),
        reason="Flash attention only supported on JAX and Torch",
    )
    def test_basics_with_flash_attention(self):
        enable_flash_attention()
        init_kwargs = {
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "dtype": "float16",
        }
        input_shape = {
            "query_shape": (2, 8, 16),
            "value_shape": (2, 4, 16),
        }
        expected_output_shape = (2, 8, 16)
        if backend.backend() == "torch":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "PyTorch version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if (
                    "Flash attention is not supported with the provided inputs"
                    in str(e.args[0])
                ):
                    self.assertTrue(
                        (
                            "Flash attention is not supported with the "
                            "provided inputs"
                        )
                        in str(e.args[0])
                    )
        elif backend.backend() == "jax":
            try:
                self.run_layer_test(
                    layers.GroupedQueryAttention,
                    init_kwargs=init_kwargs,
                    input_shape=input_shape,
                    expected_output_shape=expected_output_shape,
                    expected_num_trainable_weights=8,
                    expected_num_non_trainable_weights=0,
                    expected_num_seed_generators=0,
                    expected_num_losses=0,
                    supports_masking=True,
                    run_training_check=False,
                )
            except ImportError as e:
                if "Flash attention is not supported" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
                    )
            except RuntimeError as e:
                if "cuDNN" in str(e.args[0]):
                    self.assertTrue("cuDNN is not detected." in str(e.args[0]))
                elif "Require at least" in str(e.args[0]):
                    self.assertTrue(
                        "Require at least Ampere arch to run" in str(e.args[0])
                    )
                elif "Flash attention" in str(e.args[0]):
                    self.assertTrue(
                        (
                            "Flash attention is not supported in your current "
                            "JAX version."
                        )
                        in str(e.args[0])
                    )

    @parameterized.named_parameters(
        ("without_key_proj_mha", (4, 8), (2, 8), None, 2, 2),
        ("with_key_proj_mha", (4, 8), (2, 8), (2, 3), 2, 2),
        ("without_key_proj_gqa", (4, 8), (2, 8), None, 4, 2),
        ("with_key_proj_gqa", (4, 8), (2, 8), (2, 3), 4, 2),
        ("without_key_value_proj_mqa", (4, 8), (2, 8), None, 4, 1),
        ("with_key_value_proj_mqa", (4, 8), (2, 8), (2, 3), 4, 1),
    )
    def test_compute_output_shape(
        self,
        query_dims,
        value_dims,
        key_dims,
        num_query_heads,
        num_key_value_heads,
    ):
        """Test computed shape is equal to the layer output's shape."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=2,
        )
        batch_size = 7
        query_shape = (batch_size,) + query_dims
        value_shape = (batch_size,) + value_dims
        key_shape = (batch_size,) + key_dims if key_dims else None

        query = np.ones(query_shape)
        value = np.ones(value_shape)
        key = np.ones(key_shape) if key_shape else None
        output = layer(query=query, value=value, key=key)
        comp_output_shape = layer.compute_output_shape(
            query_shape, value_shape, key_shape
        )
        self.assertEqual(output.shape, comp_output_shape)

    @parameterized.named_parameters(
        ("query_value_dim_mismatch", (2, 4, 8), (2, 2, 7), 2),
        ("key_value_dim_mismatch", (2, 4, 8), (2, 2, 8), (2, 1, 7)),
    )
    def test_shape_mismatch_error(self, query_shape, value_shape, key_shape):
        """Test dimension mismatches"""
        layer = layers.GroupedQueryAttention(
            num_query_heads=4,
            num_key_value_heads=4,
            head_dim=2,
        )
        with self.assertRaisesRegex(ValueError, r"must be equal"):
            layer.compute_output_shape(query_shape, value_shape, key_shape)

    def test_symbolic_shape_with_return_attention_scores(self):
        layer = layers.GroupedQueryAttention(
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        query = layers.Input(shape=(10, 32))
        value = layers.Input(shape=(15, 32))

        output, scores = layer(query, value, return_attention_scores=True)
        self.assertEqual(output.shape, (None, 10, 32))
        self.assertEqual(scores.shape, (None, 4, 10, 15))

    def test_symbolic_shape_without_return_attention_scores(self):
        layer = layers.GroupedQueryAttention(
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        query = layers.Input(shape=(10, 32))
        value = layers.Input(shape=(15, 32))

        output = layer(query, value)
        self.assertEqual(output.shape, (None, 10, 32))

    def test_initializer(self):
        # Test with a specified initializer.
        layer = layers.GroupedQueryAttention(
            num_query_heads=16,
            num_key_value_heads=16,
            head_dim=64,
            use_gate=True,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        layer.build((2, 4, 8), (2, 4, 8))

        # Make sure the sub layers have different kernel init value.
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._key_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._value_dense.kernel,
        )
        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._output_dense.kernel,
        )

        self.assertNotAllClose(
            layer._query_dense.kernel,
            layer._gate_dense.kernel,
        )

    def test_query_mask_propagation(self):
        """Test automatic propagation of the query's mask."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2
        )
        self.assertTrue(layer.supports_masking)
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.random.normal(size=(3, 3, 8))
        output = layer(query=masked_query, value=value)
        self.assertAllClose(
            get_keras_mask(masked_query), get_keras_mask(output)
        )

        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2, use_gate=True
        )
        self.assertTrue(layer.supports_masking)
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.random.normal(size=(3, 3, 8))
        output = layer(query=masked_query, value=value)
        self.assertAllClose(
            get_keras_mask(masked_query), get_keras_mask(output)
        )

    @parameterized.named_parameters(("causal", True), ("not_causal", 0))
    def test_masking(self, use_causal_mask):
        """Test that the value and causal masks are taken into account."""
        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2
        )
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
        masked_value = layers.Embedding(6, 8, mask_zero=True)(value)
        output = layer(
            query=masked_query,
            value=masked_value,
            use_causal_mask=use_causal_mask,
        )
        mask = np.array(
            [[[1, 1, 0]] * 3 + [[0, 0, 0]] * 2]
            + [[[1, 0, 0]] * 5]
            + [[[1, 1, 1]] + [[0, 0, 0]] * 4]
        ).astype(bool)
        if use_causal_mask:
            mask = mask & np.array(
                [[[1, 0, 0], [1, 1, 0]] + [[1, 1, 1]] * 3]
            ).astype(bool)
        set_keras_mask(masked_query, None)
        set_keras_mask(masked_value, None)
        output_with_manual_mask = layer(
            query=masked_query, value=masked_value, attention_mask=mask
        )
        self.assertAllClose(output, output_with_manual_mask)

        layer = layers.GroupedQueryAttention(
            num_query_heads=2, num_key_value_heads=2, head_dim=2, use_gate=True
        )
        query = np.array([[1, 2, 3, 0, 0], [3, 3, 1, 1, 2], [1, 0, 0, 0, 0]])
        masked_query = layers.Embedding(4, 8, mask_zero=True)(query)
        value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
        masked_value = layers.Embedding(6, 8, mask_zero=True)(value)
        output = layer(
            query=masked_query,
            value=masked_value,
            use_causal_mask=use_causal_mask,
        )
        mask = np.array(
            [[[1, 1, 0]] * 3 + [[0, 0, 0]] * 2]
            + [[[1, 0, 0]] * 5]
            + [[[1, 1, 1]] + [[0, 0, 0]] * 4]
        ).astype(bool)
        if use_causal_mask:
            mask = mask & np.array(
                [[[1, 0, 0], [1, 1, 0]] + [[1, 1, 1]] * 3]
            ).astype(bool)
        set_keras_mask(masked_query, None)
        set_keras_mask(masked_value, None)
        output_with_manual_mask = layer(
            query=masked_query, value=masked_value, attention_mask=mask
        )
        self.assertAllClose(output, output_with_manual_mask)

    def test_sliding_window(self):
        layer = layers.GroupedQueryAttention(
            head_dim=4,
            num_query_heads=4,
            num_key_value_heads=2,
            sliding_window=3,
        )
        x = np.random.RandomState(0).randn(1, 6, 8).astype("float32")
        _, scores = layer(
            x, x, use_causal_mask=True, return_attention_scores=True
        )
        scores = ops.convert_to_numpy(scores)[0, 0]
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                in_window = (j <= i) and (i - j < 3)
                if in_window:
                    self.assertGreater(scores[i, j], 0.0)
                else:
                    self.assertEqual(scores[i, j], 0.0)

    def test_causal_only_fast_path(self):
        # When `use_causal_mask=True` is the only mask source, the layer
        # should route through `dot_product_attention(is_causal=True)` and
        # not materialize a [T, S] mask. The output must match the explicit
        # mask path.
        layer = layers.GroupedQueryAttention(
            head_dim=4, num_query_heads=4, num_key_value_heads=2
        )
        x = np.random.RandomState(0).randn(2, 5, 16).astype("float32")
        _ = layer(x, x, use_causal_mask=True)
        fast = layer(x, x, use_causal_mask=True)

        t = x.shape[1]
        causal = np.tril(np.ones((1, t, t), dtype="bool"))
        slow = layer(x, x, attention_mask=causal)

        self.assertAllClose(fast, slow, atol=1e-5)

    def test_causal_only_skipped_for_subclass(self):
        # A subclass with the previous _compute_attention signature must
        # still receive the explicit causal mask, not None.
        seen = {}

        class Sub(layers.GroupedQueryAttention):
            def _compute_attention(
                self, query, key, value, attention_mask=None, training=None
            ):
                seen["mask_is_none"] = attention_mask is None
                return super()._compute_attention(
                    query, key, value, attention_mask, training
                )

        layer = Sub(head_dim=4, num_query_heads=4, num_key_value_heads=2)
        x = np.random.RandomState(0).randn(2, 5, 16).astype("float32")
        _ = layer(x, x, use_causal_mask=True)
        self.assertFalse(seen["mask_is_none"])

    def test_causal_only_with_other_mask_falls_back(self):
        # If any other mask is also present, the is_causal=True shortcut
        # must not be taken since SDPA rejects mask + is_causal together.
        # A regression that silently kept the shortcut would also drop the
        # extra mask entirely, so compare against the explicit merged-mask
        # path to prove `extra_mask` is honored.
        layer = layers.GroupedQueryAttention(
            head_dim=4, num_query_heads=4, num_key_value_heads=2
        )
        x = np.random.RandomState(0).randn(2, 5, 16).astype("float32")
        # Build once so both calls share weights.
        layer.build(query_shape=x.shape, value_shape=x.shape)

        # Non-trivial extra mask: query position 2 must not attend to key 3.
        extra_mask = np.ones((2, 5, 5), dtype="bool")
        extra_mask[:, 2, 3] = False
        causal = np.tril(np.ones((5, 5), dtype="bool"))
        merged = extra_mask & causal

        out1 = layer(x, x, attention_mask=extra_mask, use_causal_mask=True)
        out2 = layer(x, x, attention_mask=merged, use_causal_mask=False)
        self.assertEqual(tuple(out1.shape), (2, 5, 16))
        self.assertAllClose(out1, out2, atol=1e-5)

    @parameterized.named_parameters(
        ("scores_only", 0.0, True),
        ("dropout_only", 0.1, False),
        ("dropout_and_scores", 0.1, True),
    )
    def test_causal_only_masks_when_fused_path_disabled(
        self, dropout, return_attention_scores
    ):
        # dropout>0 and return_attention_scores=True both disable the fused
        # is_causal kernel and send us down the manual softmax path. That path
        # has to apply the causal mask too, or future keys leak in. See #22910.
        t = 6
        future = np.triu(np.ones((t, t)), k=1).astype(bool)
        x = np.random.RandomState(0).randn(1, t, 16).astype("float32")
        layer = layers.GroupedQueryAttention(
            head_dim=4,
            num_query_heads=4,
            num_key_value_heads=2,
            dropout=dropout,
        )

        if return_attention_scores:
            # No weight should land on a strictly-future key.
            _, scores = layer(
                x,
                x,
                use_causal_mask=True,
                return_attention_scores=True,
                training=False,
            )
            leak = backend.ops.convert_to_numpy(scores)[..., future].sum()
            self.assertLess(leak, 1e-6)
        else:
            # use_causal_mask should match passing the same mask explicitly.
            out_causal = layer(x, x, use_causal_mask=True, training=False)
            explicit = (~future)[None]
            out_explicit = layer(x, x, attention_mask=explicit, training=False)
            self.assertAllClose(out_causal, out_explicit, atol=1e-5)

    @parameterized.named_parameters(
        ("fused_fallback", False),
        ("manual_path", True),
    )
    def test_compute_attention_combines_causal_and_explicit_mask(
        self, return_attention_scores
    ):
        # `_compute_attention` is a documented override point. Called directly
        # with both use_causal_mask=True and an explicit attention_mask, it must
        # apply both. `call` folds the causal mask in upstream, but a subclass
        # reusing this method does not, so the method must honor its own flag.
        t = 5
        layer = layers.GroupedQueryAttention(
            head_dim=4, num_query_heads=4, num_key_value_heads=2
        )
        layer.build(query_shape=(1, t, 16), value_shape=(1, t, 16))
        layer._return_attention_scores = return_attention_scores

        rng = np.random.RandomState(0)
        # After projection and head repeat, q/k/v are [B, T, num_query_heads,
        # head_dim].
        query = rng.randn(1, t, 4, 4).astype("float32")
        key = rng.randn(1, t, 4, 4).astype("float32")
        value = rng.randn(1, t, 4, 4).astype("float32")

        # A non-causal extra constraint so the causal mask is not redundant.
        extra = np.ones((1, t, t), dtype="bool")
        extra[:, 3, 1] = False
        causal = np.tril(np.ones((t, t), dtype="bool"))[None]
        merged = extra & causal

        out_both, _ = layer._compute_attention(
            query, key, value, attention_mask=extra, use_causal_mask=True
        )
        out_merged, _ = layer._compute_attention(
            query, key, value, attention_mask=merged, use_causal_mask=False
        )
        self.assertAllClose(out_both, out_merged, atol=1e-5)

    def test_sliding_window_validation(self):
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            layers.GroupedQueryAttention(
                head_dim=2,
                num_query_heads=2,
                num_key_value_heads=2,
                sliding_window=0,
            )

    def test_sliding_window_serialization(self):
        layer = layers.GroupedQueryAttention(
            head_dim=4,
            num_query_heads=2,
            num_key_value_heads=2,
            sliding_window=5,
        )
        config = layer.get_config()
        self.assertEqual(config["sliding_window"], 5)
        restored = layers.GroupedQueryAttention.from_config(config)
        self.assertEqual(restored.sliding_window, 5)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only the TensorFlow backend has dropout in flash attention.",
    )
    @parameterized.named_parameters(("causal", True), ("not_causal", False))
    def test_flash_attention_dropout_tensorflow(self, use_causal_mask):
        import tensorflow as tf

        if not hasattr(tf.nn, "flash_attention"):
            self.skipTest("This TensorFlow build has no tf.nn.flash_attention.")
        query = np.random.normal(size=(2, 16, 32)).astype("float32")
        value = np.random.normal(size=(2, 16, 32)).astype("float32")

        def make_layer(dropout, seed=None):
            return layers.GroupedQueryAttention(
                head_dim=16,
                num_query_heads=4,
                num_key_value_heads=2,
                flash_attention=True,
                dropout=dropout,
                seed=seed,
            )

        layer = make_layer(0.3, seed=1)
        no_dropout = make_layer(0.0)
        layer(query, value)
        no_dropout(query, value)
        no_dropout.set_weights(layer.get_weights())

        def run(lyr, training):
            return ops.convert_to_numpy(
                lyr(
                    query,
                    value,
                    use_causal_mask=use_causal_mask,
                    training=training,
                )
            )

        # Inference: no dropout.
        self.assertAllClose(run(layer, False), run(no_dropout, False))
        # Training: dropout inside the kernel, with a new mask every call.
        first = run(layer, True)
        second = run(layer, True)
        self.assertNotAllClose(first, run(no_dropout, False))
        self.assertNotAllClose(first, second)
        # The same `seed` reproduces the same sequence of masks.
        same_seed = make_layer(0.3, seed=1)
        same_seed(query, value)
        same_seed.set_weights(layer.get_weights())
        self.assertAllClose(run(same_seed, True), first)
        # Gradients flow through the dropout path.
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                layer(
                    query, value, use_causal_mask=use_causal_mask, training=True
                )
                ** 2
            )
        grads = tape.gradient(loss, layer.trainable_variables)
        self.assertTrue(
            all(np.isfinite(ops.convert_to_numpy(g)).all() for g in grads)
        )

    @parameterized.named_parameters(
        ("true", True), ("false", False), ("none", None)
    )
    def test_flash_attention_serialization(self, flash_attention):
        # The argument round-trips as given, so a saved
        # `flash_attention=True` still uses flash attention after loading,
        # and `None` still follows the global setting.
        layer = layers.GroupedQueryAttention(
            head_dim=8,
            num_query_heads=4,
            num_key_value_heads=2,
            flash_attention=flash_attention,
        )
        config = layer.get_config()
        self.assertIs(config["flash_attention"], flash_attention)
        restored = layers.GroupedQueryAttention.from_config(config)
        self.assertEqual(restored._flash_attention, layer._flash_attention)
        self.assertIs(restored.get_config()["flash_attention"], flash_attention)
        restored = saving.deserialize_keras_object(
            saving.serialize_keras_object(layer)
        )
        self.assertEqual(restored._flash_attention, layer._flash_attention)

    def test_flash_attention_config_missing(self):
        # Configs saved before `flash_attention` was serialized still load,
        # following the global setting.
        config = layers.GroupedQueryAttention(
            head_dim=8, num_query_heads=4, num_key_value_heads=2
        ).get_config()
        del config["flash_attention"]
        restored = layers.GroupedQueryAttention.from_config(config)
        self.assertFalse(restored._flash_attention)

    @parameterized.named_parameters(
        ("disable_flash_attention", False), ("enable_flash_attention", True)
    )
    def test_correctness(self, flash_attention):
        if flash_attention:
            # Let the backend decide whether to use flash attention
            enable_flash_attention()
        dtype = "float16"  # Flash attention only accepts float16/bfloat16
        head_dim = 8  # key_dim % 8 == 0 to enable flash attention
        num_query_heads = num_key_value_heads = 8

        query = np.identity(head_dim)[np.newaxis, ...]
        key = np.identity(head_dim)[np.newaxis, ...]
        value = (
            np.reshape(np.arange(head_dim * head_dim), (1, head_dim, head_dim))
            / 100.0  # Prevent overflow/underflow
        )

        # Setup layer.
        layer = layers.GroupedQueryAttention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            dtype=dtype,
        )
        layer.build(query.shape, key.shape, value.shape)

        # Set layer weights.
        kernel = np.identity(head_dim)
        # To get an identity kernel we need to add a head dim and repeat on it.
        kernel = np.repeat(kernel[:, np.newaxis, :], num_query_heads, axis=1)
        # Zeros for all biases.
        bias = np.zeros((num_query_heads, head_dim))
        output_bias = np.zeros((head_dim,))
        layer.set_weights([kernel, bias] * 3 + [kernel, output_bias])

        # Call layer and assert output.
        expected_output = np.array(
            [2.406, 2.440, 2.473, 2.504, 2.535, 2.568, 2.602, 2.633]
        )
        expected_output = np.tile(
            expected_output[np.newaxis, :, np.newaxis], (1, 1, head_dim)
        )
        expected_score = np.array(
            [
                [0.1187] * 0 + [0.1691] + [0.1187] * 7,
                [0.1187] * 1 + [0.1691] + [0.1187] * 6,
                [0.1187] * 2 + [0.1691] + [0.1187] * 5,
                [0.1187] * 3 + [0.1691] + [0.1187] * 4,
                [0.1187] * 4 + [0.1691] + [0.1187] * 3,
                [0.1187] * 5 + [0.1691] + [0.1187] * 2,
                [0.1187] * 6 + [0.1691] + [0.1187] * 1,
                [0.1187] * 7 + [0.1691] + [0.1187] * 0,
            ]
        )
        expected_score = np.tile(
            expected_score[np.newaxis, np.newaxis, ...], (1, head_dim, 1, 1)
        )
        if flash_attention:
            output = layer(query=query, value=value, key=key)
            self.assertAllClose(output, expected_output, atol=1e-2)
        else:
            output, scores = layer(
                query=query,
                value=value,
                key=key,
                return_attention_scores=True,
            )
            self.assertAllClose(output, expected_output, atol=1e-2)
            self.assertAllClose(scores, expected_score, atol=1e-2)

    def test_flash_attention_with_errors(self):
        if backend.backend() in ("numpy", "tensorflow"):
            pytest.skip(
                reason=(
                    "Flash attention is not supported on tensorflow and numpy."
                )
            )
        # Check `flash_attention=True` and `dropout=0.1`
        with self.assertRaisesRegex(
            ValueError,
            "Dropout is not supported when flash attention is enabled.",
        ):
            layer = layers.GroupedQueryAttention(
                head_dim=2,
                num_query_heads=2,
                num_key_value_heads=2,
                flash_attention=True,
                dropout=0.1,
            )

        # Check `flash_attention=True` and `return_attention_scores=True`
        layer = layers.GroupedQueryAttention(
            head_dim=2,
            num_query_heads=2,
            num_key_value_heads=2,
            flash_attention=True,
        )
        self.assertTrue(layer._flash_attention)
        query = np.random.random((2, 4, 8))
        value = np.random.random((2, 4, 8))
        with self.assertRaisesRegex(
            ValueError,
            "Returning attention scores is not supported when flash "
            "attention is enabled. Please disable flash attention to access"
            " attention scores.",
        ):
            layer(query=query, value=value, return_attention_scores=True)
