"""Tests for PyTorch backend nn utilities."""

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class DotProductAttentionCompileTest(testing.TestCase):
    def _qkv(self, batch=2, seq=16, heads=4, head_dim=8):
        rng = np.random.default_rng(0)
        shape = (batch, seq, heads, head_dim)
        return tuple(
            ops.convert_to_tensor(rng.standard_normal(shape).astype("float32"))
            for _ in range(3)
        )

    @parameterized.named_parameters(
        ("no_options", False, False, False, None),
        ("is_causal", True, False, False, None),
        ("mask", False, True, False, None),
        ("mask_and_is_causal", True, True, False, None),
        ("bias", False, False, True, None),
        ("scale", False, False, False, 0.125),
    )
    def test_compiled_matches_eager(self, is_causal, use_mask, use_bias, scale):
        query, key, value = self._qkv()
        kwargs = {"is_causal": is_causal}
        if scale is not None:
            kwargs["scale"] = scale
        rng = np.random.default_rng(1)
        # `mask` and `bias` must broadcast to (batch, heads, q_len, kv_len).
        if use_mask:
            kwargs["mask"] = ops.convert_to_tensor(
                rng.integers(0, 2, (2, 1, 16, 16)).astype("bool")
            )
        if use_bias:
            kwargs["bias"] = ops.convert_to_tensor(
                rng.standard_normal((2, 1, 16, 16)).astype("float32")
            )

        # `fullgraph=True` turns a graph break into an error, so this also
        # covers the untraceable flash attention probe regressing back in.
        compiled = torch.compile(ops.dot_product_attention, fullgraph=True)
        self.assertAllClose(
            compiled(query, key, value, **kwargs),
            ops.dot_product_attention(query, key, value, **kwargs),
        )


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class PoolingChannelsLastContiguityTest(testing.TestCase):
    def test_pooling_2d_channels_last_output_is_contiguous(self):
        x = torch.randn((2, 8, 8, 4))
        # 7x7 makes `same` padding uneven, so the input goes through `F.pad`.
        # `max_pool` is left out there: it pads in `replicate` mode, which
        # returns the default memory format on CUDA.
        x_odd = torch.randn((2, 7, 7, 4))
        for out in (
            ops.max_pool(x, (2, 2), data_format="channels_last"),
            ops.average_pool(x, (2, 2), data_format="channels_last"),
            ops.average_pool(
                x_odd, (2, 2), padding="same", data_format="channels_last"
            ),
            ops.adaptive_average_pool(x, (4, 4), data_format="channels_last"),
        ):
            self.assertTrue(out.is_contiguous())

    def test_max_pool_3d_channels_last_output_is_contiguous(self):
        # Of the 3D pooling kernels, only `max_pool3d` keeps `channels_last_3d`.
        x = torch.randn((2, 6, 6, 6, 4))
        out = ops.max_pool(x, (2, 2, 2), data_format="channels_last")
        self.assertTrue(out.is_contiguous())


class ConvVectorizedMapTest(testing.TestCase):
    def test_vectorized_map_conv3d_channels_last(self):
        # Regression test for #23587: channels_last conv used to query
        # `is_contiguous(memory_format=channels_last_3d)` inside torch.vmap.
        def conv_one(sample):
            sample = ops.expand_dims(sample, axis=0)
            kernel = ops.ones((3, 3, 3, 1, 1), dtype="float32")
            return ops.conv(
                sample, kernel, padding="same", data_format="channels_last"
            )[0]

        batch = ops.ones((2, 8, 8, 8, 1), dtype="float32")
        result = ops.vectorized_map(conv_one, batch)
        expected = ops.conv(
            batch,
            ops.ones((3, 3, 3, 1, 1), dtype="float32"),
            padding="same",
            data_format="channels_last",
        )
        self.assertEqual(tuple(result.shape), (2, 8, 8, 8, 1))
        self.assertAllClose(result, expected)

    def test_vectorized_map_conv2d_channels_last(self):
        def conv_one(sample):
            sample = ops.expand_dims(sample, axis=0)
            kernel = ops.ones((3, 3, 1, 1), dtype="float32")
            return ops.conv(
                sample, kernel, padding="same", data_format="channels_last"
            )[0]

        batch = ops.ones((2, 8, 8, 1), dtype="float32")
        result = ops.vectorized_map(conv_one, batch)
        expected = ops.conv(
            batch,
            ops.ones((3, 3, 1, 1), dtype="float32"),
            padding="same",
            data_format="channels_last",
        )
        self.assertEqual(tuple(result.shape), (2, 8, 8, 1))
        self.assertAllClose(result, expected)
