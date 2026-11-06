import numpy as np

from keras.src import testing
from keras.src.backend import Variable
from keras.src.distribution.tensor_parallel import tensor_layout


class TensorLayoutTest(testing.TestCase):
    def test_split_tensor_for_parallelism_1d(self):
        tensor = np.arange(8)
        # Split into 2 sections
        shard0 = tensor_layout.split_tensor_for_parallelism(
            tensor, index=0, device_count=2, dim=0
        )
        shard1 = tensor_layout.split_tensor_for_parallelism(
            tensor, index=1, device_count=2, dim=0
        )

        self.assertAllClose(shard0, [0, 1, 2, 3])
        self.assertAllClose(shard1, [4, 5, 6, 7])

    def test_split_tensor_for_parallelism_2d(self):
        tensor = np.arange(16).reshape((4, 4))

        # Split along row (dim=0)
        shard0_row = tensor_layout.split_tensor_for_parallelism(
            tensor, index=0, device_count=2, dim=0
        )
        shard1_row = tensor_layout.split_tensor_for_parallelism(
            tensor, index=1, device_count=2, dim=0
        )
        self.assertAllClose(shard0_row, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertAllClose(shard1_row, [[8, 9, 10, 11], [12, 13, 14, 15]])

        # Split along col (dim=1)
        shard0_col = tensor_layout.split_tensor_for_parallelism(
            tensor, index=0, device_count=2, dim=1
        )
        shard1_col = tensor_layout.split_tensor_for_parallelism(
            tensor, index=1, device_count=2, dim=1
        )
        self.assertAllClose(shard0_col, [[0, 1], [4, 5], [8, 9], [12, 13]])
        self.assertAllClose(shard1_col, [[2, 3], [6, 7], [10, 11], [14, 15]])

    def test_split_tensor_for_parallelism_negative_dim(self):
        tensor = np.arange(16).reshape((4, 4))
        # dim=-1 should be canonicalized to dim=1
        shard0_col = tensor_layout.split_tensor_for_parallelism(
            tensor, index=0, device_count=2, dim=-1
        )
        self.assertAllClose(shard0_col, [[0, 1], [4, 5], [8, 9], [12, 13]])

    def test_split_tensor_for_parallelism_variable_input(self):
        tensor_np = np.arange(8).reshape((2, 4)).astype("float32")
        var = Variable(tensor_np)

        # Verify it extracts the value and splits correctly
        shard0 = tensor_layout.split_tensor_for_parallelism(
            var, index=0, device_count=2, dim=1
        )
        self.assertAllClose(shard0, [[0, 1], [4, 5]])

    def test_uneven_splits(self):
        tensor = np.arange(5)
        # np.array_split handles uneven splits by making the first shards larger
        shard0 = tensor_layout.split_tensor_for_parallelism(
            tensor, index=0, device_count=2, dim=0
        )
        shard1 = tensor_layout.split_tensor_for_parallelism(
            tensor, index=1, device_count=2, dim=0
        )

        self.assertAllClose(shard0, [0, 1, 2])
        self.assertAllClose(shard1, [3, 4])

    def test_layout_map_namedtuple(self):
        state_rules = {"kernel": "split"}
        output_rules = {"output": "all_reduce"}
        layout = tensor_layout.ParallelLayoutMap(
            state_rules=state_rules, output_rules=output_rules
        )
        self.assertEqual(layout.state_rules, state_rules)
        self.assertEqual(layout.output_rules, output_rules)

    def test_invalid_inputs(self):
        tensor = np.arange(8)
        with self.assertRaisesRegex(
            ValueError, "Expected device_count to be a positive integer"
        ):
            tensor_layout.split_tensor_for_parallelism(
                tensor, index=0, device_count=0, dim=0
            )
        with self.assertRaisesRegex(
            ValueError, "Expected index to be an integer in the range"
        ):
            tensor_layout.split_tensor_for_parallelism(
                tensor, index=2, device_count=2, dim=0
            )
