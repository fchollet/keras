"""Tests for JAX backend numpy ops."""

import functools
import math
import re

import jax
import numpy as np
import pytest
from absl.testing import parameterized
from jax import export as jax_export

from keras.src import backend
from keras.src import testing
from keras.src.ops import numpy as knp


@pytest.mark.skipif(
    backend.backend() != "jax" or testing.uses_gpu() or testing.uses_tpu(),
    reason="XLA CPU specific test",
)
class Int8MatmulTest(testing.TestCase):
    """XLA CPU runs a one-row int8 dot on an int32 copy of the right operand.

    The tests compile `matmul` and check that the compiled program holds no
    such copy. `keras/src/ops/numpy_test.py` checks the values on every
    backend.
    """

    @parameterized.named_parameters(
        ("rank1", (64,)),
        ("one_row", (1, 64)),
        ("two_rows", (2, 64)),
        ("one_row_rank3", (1, 1, 64)),
        ("three_rows_rank3", (3, 1, 64)),
        ("one_row_k16384", (1, 16384)),
    )
    def test_matmul_int8(self, x_shape):
        rng = np.random.default_rng(0)
        x = rng.integers(-128, 128, size=x_shape, dtype="int8")
        y = rng.integers(-128, 128, size=(x_shape[-1], 256), dtype="int8")
        expected = np.matmul(x.astype("int64"), y.astype("int64"))

        compiled = jax.jit(knp.matmul).lower(x, y).compile()
        self.assertNotIn(f"s32[{x_shape[-1]},256]", compiled.as_text())
        self.assertAllEqual(compiled(x, y), expected)

    @parameterized.named_parameters(
        ("rank1", (64,)),
        ("rank2", (1, 64)),
        ("rank3", (1, 1, 64)),
    )
    def test_matmul_int8_one_row_transposed(self, x_shape):
        # A tied `ReversibleEmbedding` contracts against the transpose of
        # its table.
        rng = np.random.default_rng(0)
        x = rng.integers(-128, 128, size=x_shape, dtype="int8")
        table = rng.integers(-128, 128, size=(256, 64), dtype="int8")
        expected = np.matmul(x.astype("int64"), table.T.astype("int64"))

        def matmul_transposed(x, table):
            return knp.matmul(x, knp.transpose(table))

        compiled = jax.jit(matmul_transposed).lower(x, table).compile()
        self.assertNotIn("s32[256,64]", compiled.as_text())
        self.assertNotIn("s32[64,256]", compiled.as_text())
        self.assertAllEqual(compiled(x, table), expected)


# The int8 `einsum` equations of `keras/src/ops/numpy_test.py`. Each kernel
# has at least 32768 elements: with `--xla_cpu_multi_thread_eigen=false`, a
# smaller kernel can get an int32 copy even at two rows.
INT8_EINSUM_ONE_ROW = [
    ("btd_ndh", "btd,ndh->btnh", (1, 1, 256), (4, 256, 32)),
    ("btnh_nhd", "btnh,nhd->btd", (1, 1, 4, 32), (4, 32, 256)),
    ("btd_df", "btd,df->btf", (1, 1, 256), (256, 256)),
    ("bqm_muh", "bqm,muh->bquh", (1, 1, 256), (256, 4, 32)),
    ("bquh_uhm", "bquh,uhm->bqm", (1, 1, 4, 32), (4, 32, 256)),
    ("abc_cde", "abc,cde->abde", (1, 1, 256), (256, 4, 32)),
    ("abcd_cde", "abcd,cde->abe", (1, 1, 4, 32), (4, 32, 256)),
    ("ab_bc", "ab,bc->ac", (1, 256), (256, 256)),
    ("d_df", "d,df->f", (256,), (256, 256)),
]


def int8_operands(x_shape, kernel_shape):
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=x_shape, dtype="int8")
    kernel = rng.integers(-128, 128, size=kernel_shape, dtype="int8")
    return x, kernel


def int8_einsum_reference(equation, x, kernel):
    return np.einsum(equation, x.astype("int64"), kernel.astype("int64"))


@pytest.mark.skipif(
    backend.backend() != "jax" or testing.uses_gpu() or testing.uses_tpu(),
    reason="XLA CPU specific test",
)
class Int8EinsumTest(testing.TestCase):
    """XLA CPU runs a one-row int8 dot on an int32 copy of the kernel.

    The tests compile `einsum` and check that a one-row `einsum` is padded to
    two rows and that the compiled program holds no such copy.
    `keras/src/ops/numpy_test.py` checks the values on every backend.
    """

    def assertNoInt32KernelCopy(self, compiled, kernel_shape):
        shapes = re.findall(r"\bs32\[([0-9,]*)\]", compiled.as_text())
        sizes = [math.prod(int(d) for d in s.split(",") if d) for s in shapes]
        self.assertNotIn(math.prod(kernel_shape), sizes)

    def assertPadded(self, einsum, x, kernel, padded):
        jaxpr = str(jax.make_jaxpr(einsum)(x, kernel))
        self.assertEqual("pad[" in jaxpr, padded)

    @parameterized.named_parameters(INT8_EINSUM_ONE_ROW)
    def test_einsum_int8_one_row(self, equation, x_shape, kernel_shape):
        x, kernel = int8_operands(x_shape, kernel_shape)
        einsum = functools.partial(knp.einsum, equation)
        self.assertPadded(einsum, x, kernel, True)
        compiled = jax.jit(einsum).lower(x, kernel).compile()
        self.assertNoInt32KernelCopy(compiled, kernel_shape)
        expected = int8_einsum_reference(equation, x, kernel)
        self.assertAllEqual(compiled(x, kernel), expected)

    @parameterized.named_parameters(INT8_EINSUM_ONE_ROW[:-1])
    def test_einsum_int8_two_rows(self, equation, x_shape, kernel_shape):
        x, kernel = int8_operands((2,) + x_shape[1:], kernel_shape)
        einsum = functools.partial(knp.einsum, equation)
        self.assertPadded(einsum, x, kernel, False)
        compiled = jax.jit(einsum).lower(x, kernel).compile()
        self.assertNoInt32KernelCopy(compiled, kernel_shape)

    @parameterized.named_parameters(INT8_EINSUM_ONE_ROW)
    def test_einsum_int8_one_row_transposed_kernel(
        self, equation, x_shape, kernel_shape
    ):
        # The kernel arrives with its axes reversed and is transposed inside
        # the function.
        def einsum_transposed(x, kernel_t):
            return knp.einsum(equation, x, knp.transpose(kernel_t))

        x, kernel = int8_operands(x_shape, kernel_shape)
        compiled = jax.jit(einsum_transposed).lower(x, kernel.T).compile()
        self.assertNoInt32KernelCopy(compiled, kernel_shape)
        expected = int8_einsum_reference(equation, x, kernel)
        self.assertAllEqual(compiled(x, kernel.T), expected)

    @parameterized.named_parameters(
        ("btnh_nhd", "btnh,nhd->btd", (1, 1, 4, 32), (4, 32, 256)),
        ("bquh_uhm", "bquh,uhm->bqm", (1, 1, 4, 32), (4, 32, 256)),
    )
    def test_einsum_int8_one_row_no_kernel_transpose(
        self, equation, x_shape, kernel_shape
    ):
        # Sorted contracted labels make XLA write a transposed int8 copy of
        # the kernel.
        x, kernel = int8_operands(x_shape, kernel_shape)
        einsum = functools.partial(knp.einsum, equation)
        compiled = jax.jit(einsum).lower(x, kernel).compile()
        temp_bytes = compiled.memory_analysis().temp_size_in_bytes
        self.assertLess(temp_bytes, math.prod(kernel_shape))

    @parameterized.named_parameters(
        ("batch", "bij,bjk->bik", (1, 1, 256), (1, 256, 64)),
        ("transposed_output", "ab,bc->ca", (1, 256), (256, 256)),
        ("summed_by_x1", "abc,cd->ad", (1, 1, 256), (256, 256)),
        ("ellipsis", "...d,df->...f", (1, 1, 256), (256, 256)),
        ("broadcast", "btd,df->btf", (1, 1, 1), (256, 256)),
    )
    def test_einsum_int8_one_row_unpadded(
        self, equation, x_shape, kernel_shape
    ):
        x, kernel = int8_operands(x_shape, kernel_shape)
        einsum = functools.partial(knp.einsum, equation)
        self.assertPadded(einsum, x, kernel, False)
        expected = int8_einsum_reference(equation, x, kernel)
        self.assertAllEqual(einsum(x, kernel), expected)

    def test_einsum_int8_one_row_contracted_sizes_differ(self):
        # A contracted size that differs keeps the unpadded `einsum`, so the
        # error names the subscripts of the caller.
        x, kernel = int8_operands((1, 1, 3), (4, 256))
        with self.assertRaisesRegex(ValueError, "label 'd'"):
            knp.einsum("btd,df->btf", x, kernel)

    def test_einsum_int8_symbolic_batch(self):
        # A symbolic batch size keeps the unpadded `einsum`. The exported
        # function runs here with a batch of one.
        equation = "btd,df->btf"
        exported = jax_export.export(
            jax.jit(functools.partial(knp.einsum, equation))
        )(
            jax.ShapeDtypeStruct(jax_export.symbolic_shape("b,1,256"), "int8"),
            jax.ShapeDtypeStruct((256, 256), "int8"),
        )
        self.assertNotIn("stablehlo.pad", exported.mlir_module())
        x, kernel = int8_operands((1, 1, 256), (256, 256))
        expected = int8_einsum_reference(equation, x, kernel)
        self.assertAllEqual(exported.call(x, kernel), expected)
