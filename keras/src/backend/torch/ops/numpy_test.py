"""Tests for PyTorch backend numpy ops."""

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.torch.ops import numpy as torch_numpy
from keras.src.backend.torch.ops.core import get_device


class ContractionRecorder(torch.overrides.TorchFunctionMode):
    """Records the left operand of each `torch.matmul` and `torch.einsum`."""

    def __init__(self):
        super().__init__()
        self.left_operands = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.matmul:
            self.left_operands.append((args[0].dtype, tuple(args[0].shape)))
        elif func is torch.einsum:
            self.left_operands.append((args[1].dtype, tuple(args[1].shape)))
        return func(*args, **(kwargs or {}))


def int8_operands(x1_shape, x2_shape, values):
    rng = np.random.default_rng(0)
    if values == "uniform":
        x1 = rng.integers(-128, 128, x1_shape)
        x2 = rng.integers(-128, 128, x2_shape)
    elif values == "large":
        # Products of one sign near the int8 limits, so that the sums grow
        # past 2^24, where float32 no longer holds every integer.
        x1 = rng.integers(120, 128, x1_shape)
        x2 = rng.integers(-128, -119, x2_shape)
    else:
        # Each product is 2^14, so a full chunk sums to exactly 2^24.
        x1 = np.full(x1_shape, -128)
        x2 = np.full(x2_shape, -128)
    return x1.astype("int8"), x2.astype("int8")


def chunked(x1_shape, axis, sizes):
    """The float32 left operands of a contraction chunked along `axis`."""
    return [
        (torch.float32, x1_shape[:axis] + (size,) + x1_shape[axis + 1 :])
        for size in sizes
    ]


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class Int8ContractionTest(testing.TestCase):
    def assertExactInt8Contraction(self, equation, x1, x2, left_operands):
        """Checks the op bit for bit against NumPy in int64.

        Also checks the dtype and shape of the left operand of each torch
        contraction that runs. `equation` is `None` for `matmul`.
        """
        with ContractionRecorder() as recorder:
            x1_tensor = ops.convert_to_tensor(x1)
            x2_tensor = ops.convert_to_tensor(x2)
            if equation is None:
                result = ops.matmul(x1_tensor, x2_tensor)
            else:
                result = ops.einsum(equation, x1_tensor, x2_tensor)
        x1 = x1.astype("int64")
        x2 = x2.astype("int64")
        if equation is None:
            expected = np.matmul(x1, x2)
        else:
            expected = np.einsum(equation, x1, x2)
        self.assertDType(result, "int32")
        self.assertAllEqual(result, expected)
        self.assertEqual(recorder.left_operands, left_operands)

    # The cases of the tests below have at least 32 rows, the largest value
    # of `_INT8_CHUNKED_MIN_ROWS`, so they chunk on every device. No 2D case
    # meets the shape conditions of `torch._int_mm`, which CUDA would take.
    @parameterized.named_parameters(
        ("one_chunk", (32, 1024), (1024, 12), [1024]),
        ("one_over", (32, 1025), (1025, 16), [1024, 1]),
        ("ragged", (33, 3000), (3000, 33), [1024, 1024, 952]),
        ("k_16384", (32, 16384), (16384, 12), [1024] * 16),
        ("batched", (2, 16, 2048), (2048, 8), [1024, 1024]),
        ("broadcast", (2, 1, 16, 1500), (3, 1500, 5), [1024, 476]),
        ("vector_kernel", (32, 2049), (2049,), [1024, 1024, 1]),
        ("empty_contraction", (32, 0), (0, 4), [0]),
    )
    def test_matmul_is_exact(self, x1_shape, x2_shape, sizes):
        left_operands = chunked(x1_shape, len(x1_shape) - 1, sizes)
        for values in ("uniform", "large", "min"):
            x1, x2 = int8_operands(x1_shape, x2_shape, values)
            self.assertExactInt8Contraction(None, x1, x2, left_operands)

    @parameterized.named_parameters(
        # The query, key, value and output projections of keras-hub models,
        # and a feedforward kernel. Each one chunks along axis 2 of `x1`.
        ("btd_dnh", "btd,dnh->btnh", (2, 16, 1500), (1500, 3, 5), [1024, 476]),
        ("btd_ndh", "btd,ndh->btnh", (2, 16, 1500), (3, 1500, 5), [1024, 476]),
        (
            "bqm_muh",
            "bqm,muh->bquh",
            (1, 32, 2049),
            (2049, 2, 3),
            [1024, 1024, 1],
        ),
        ("abc_cde", "abc,cde->abde", (2, 16, 1025), (1025, 2, 3), [1024, 1]),
        ("btnh_nhd", "btnh,nhd->btd", (2, 16, 6, 300), (6, 300, 7), [3, 3]),
        ("bquh_uhm", "bquh,uhm->bqm", (1, 32, 20, 64), (20, 64, 5), [16, 4]),
        (
            "abcd_cde",
            "abcd,cde->abe",
            (2, 16, 33, 64),
            (33, 64, 5),
            [16, 16, 1],
        ),
        ("btd_df", "btd,df->btf", (1, 32, 16384), (16384, 4), [1024] * 16),
    )
    def test_einsum_is_exact(self, equation, x1_shape, x2_shape, sizes):
        left_operands = chunked(x1_shape, 2, sizes)
        for values in ("uniform", "large", "min"):
            x1, x2 = int8_operands(x1_shape, x2_shape, values)
            self.assertExactInt8Contraction(equation, x1, x2, left_operands)

    @parameterized.named_parameters(
        ("bfloat16", torch.bfloat16), ("float16", torch.float16)
    )
    def test_exact_under_autocast(self, dtype):
        # Autocast would run the float32 chunks in `dtype`.
        with torch.autocast(torch.device(get_device()).type, dtype=dtype):
            x1, x2 = int8_operands((32, 1500), (1500, 12), "large")
            self.assertExactInt8Contraction(
                None, x1, x2, chunked((32, 1500), 1, [1024, 476])
            )
            x1, x2 = int8_operands((2, 16, 1500), (3, 1500, 5), "large")
            self.assertExactInt8Contraction(
                "btd,ndh->btnh",
                x1,
                x2,
                chunked((2, 16, 1500), 2, [1024, 476]),
            )

    @parameterized.named_parameters(
        ("short", (32, 1024), (2048, 12), "32x1024 and 2048x12"),
        ("long", (32, 2048), (1024, 12), "32x2048 and 1024x12"),
        ("batched", (2, 16, 3072), (1024, 8), "32x3072 and 1024x8"),
        ("ragged", (32, 1500), (1000, 12), "32x1500 and 1000x12"),
        ("scalar", (32, 1024), (), "at least 1D"),
    )
    def test_matmul_bad_shapes_raise(self, x1_shape, x2_shape, message):
        x1, x2 = int8_operands(x1_shape, x2_shape, "uniform")
        with self.assertRaisesRegex(RuntimeError, message):
            ops.matmul(x1, x2)

    def test_int32_path(self):
        device = torch.device(get_device()).type
        min_rows = torch_numpy._INT8_CHUNKED_MIN_ROWS.get(device)
        if min_rows is None:
            self.skipTest(f"Int8 contractions have no threshold on {device}.")
        for rows in (1, min_rows - 1, min_rows):
            for equation, x1_shape in (
                (None, (rows, 1500)),
                ("btd,df->btf", (1, rows, 1500)),
            ):
                if rows < min_rows:
                    left_operands = [(torch.int32, x1_shape)]
                else:
                    left_operands = chunked(
                        x1_shape, len(x1_shape) - 1, [1024, 476]
                    )
                x1, x2 = int8_operands(x1_shape, (1500, 8), "large")
                self.assertExactInt8Contraction(equation, x1, x2, left_operands)

        # A vector has one row.
        x1, x2 = int8_operands((1500,), (1500, 8), "large")
        self.assertExactInt8Contraction(None, x1, x2, [(torch.int32, (1500,))])

        for equation, x1_shape, x2_shape in (
            # Above 1024 products per step of the chunked label, a chunk of
            # one step would not be exact in float32.
            ("btnh,nhd->btd", (1, min_rows, 2, 1100), (2, 1100, 3)),
            # The label to chunk broadcasts from size 1.
            ("td,dn->tn", (min_rows, 1), (1500, 4)),
            # The chunking does not read an ellipsis.
            ("...d,dn->...n", (min_rows, 1500), (1500, 4)),
        ):
            x1, x2 = int8_operands(x1_shape, x2_shape, "large")
            self.assertExactInt8Contraction(
                equation, x1, x2, [(torch.int32, x1_shape)]
            )
