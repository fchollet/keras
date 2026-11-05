from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import Variable
from keras.src.distribution.tensor_parallel import autoconfig


class AutoconfigTest(testing.TestCase):
    def test_analyze_dense_layer(self):
        # Case 1: Dense Up Projection (Expansion)
        layer_up = layers.Dense(64)
        layer_up.build((None, 16))  # Input 16, Output 64
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer_up), "up_projection"
        )

        # Case 2: Dense Down Projection (Contraction)
        layer_down = layers.Dense(16)
        layer_down.build((None, 64))  # Input 64, Output 16
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer_down), "down_projection"
        )

        # Case 3: Standard Dense
        layer_dense = layers.Dense(20)
        layer_dense.build((None, 16))  # Input 16, Output 20
        self.assertEqual(autoconfig.analyze_dense_layer(layer_dense), "dense")

        # Case 4: Custom threshold
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer_up, expansion_threshold=5.0),
            "dense",
        )

    def test_analyze_dense_layer_multi_input(self):
        # Test robust handling of list/tuple input_shape
        class MultiInputLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = layers.Dense(64)

            def build(self, input_shape):
                # input_shape is a list of shapes
                self.dense.build(input_shape[0])
                self.built = True

        layer = MultiInputLayer()
        layer.build([(None, 16), (None, 16)])
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer.dense), "up_projection"
        )

    def test_analyze_dense_layer_einsum(self):
        # EinsumDense with 3D kernel (up projection heuristic)
        # Equation: "ab,bcd->acd" -> kernel shape (b, c, d)
        # Input dim is 'b', output dims are 'c' and 'd'
        layer = layers.EinsumDense("ab,bcd->acd", output_shape=(4, 8))
        layer.build((None, 2))  # Input dim 2
        # Kernel shape: (2, 4, 8) -> input_dim=2, output_dim=32
        self.assertEqual(autoconfig.analyze_dense_layer(layer), "up_projection")

        # EinsumDense with 3D kernel (down projection heuristic)
        layer_down = layers.EinsumDense("ab,bcd->acd", output_shape=(1, 1))
        layer_down.build((None, 16))  # Input dim 16
        # Kernel shape: (16, 1, 1) -> input_dim=16, output_dim=1
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer_down), "down_projection"
        )

    def test_get_variable_key(self):
        # Variable with path
        var_with_path = Variable(0.0, name="my_var")
        var_with_path._path = "layer_1/kernel"
        self.assertEqual(
            autoconfig._get_variable_key(var_with_path), "layer_1/kernel"
        )

        # Variable without explicit path (auto-generated)
        var_without_path = Variable(0.0)
        # Keras auto-generates a path/name like 'variable' or 'variable_1'
        expected_key = (
            var_without_path.path
            if var_without_path.path
            else id(var_without_path)
        )
        self.assertEqual(
            autoconfig._get_variable_key(var_without_path), expected_key
        )

    def test_get_default_config(self):
        # Create a dummy model to test config generation traversal
        model = models.Sequential(
            [
                layers.Embedding(1000, 64, name="embeddings"),
                layers.Dense(256, name="dense_up"),
                layers.Dropout(0.5, name="dropout"),
                layers.Dense(64, name="dense_down"),
            ]
        )
        model.build((None, 10))

        device_ids = ["gpu:0", "gpu:1"]

        config = autoconfig.get_default_config(model, device_ids)
        self.assertIsNotNone(config)

        # Verify that the state rules are populated correctly
        # state_rules maps variable keys (paths) to sharding functions
        state_keys = list(config.state_rules.keys())
        self.assertTrue(any("embeddings" in k for k in state_keys))
        self.assertTrue(any("dense_up" in k for k in state_keys))
        self.assertTrue(any("dense_down" in k for k in state_keys))

        # Verify specific sharding logic
        # Embedding should be column-parallel (dim=1)
        emb_key = next(k for k in state_keys if "embeddings" in k)
        emb_rule = config.state_rules[emb_key]
        self.assertEqual(emb_rule.keywords["dim"], 1)

        # dense_up should be up_projection -> column-parallel (dim=1)
        up_key = next(k for k in state_keys if "dense_up/kernel" in k)
        up_rule = config.state_rules[up_key]
        self.assertEqual(up_rule.keywords["dim"], 1)

        # dense_down should be down_projection -> row-parallel (dim=0)
        down_key = next(k for k in state_keys if "dense_down/kernel" in k)
        down_rule = config.state_rules[down_key]
        self.assertEqual(down_rule.keywords["dim"], 0)

        # Verify output rules
        output_paths = [str(p) for p in config.output_rules.keys()]
        self.assertTrue(any("embeddings" in p for p in output_paths))
        self.assertTrue(any("dense_down" in p for p in output_paths))
        # Use dropout layer ID or path to verify
        dropout_layer = model.get_layer("dropout")
        self.assertTrue(
            any(
                (dropout_layer.path and dropout_layer.path in p)
                or str(id(dropout_layer)) in p
                for p in output_paths
            )
        )
