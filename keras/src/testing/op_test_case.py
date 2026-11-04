import functools
import itertools

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.testing.test_case import TestCase

# Input dtypes of the dtype tests. No 64-bit dtypes yet; torch also tests
# float64 for multi-input ops, as the old `NumpyDtypeTest` did.
DEFAULT_INPUT_DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "bfloat16",
    "float16",
    "float32",
)
BINARY_INPUT_DTYPES = DEFAULT_INPUT_DTYPES + (
    ("float64",) if backend.backend() == "torch" else ()
)

# Input dtypes the dtype tests skip per backend, with the reason.
_SKIPPED_DTYPES = {
    "torch": {
        "uint16": "Not supported by the torch backend",
        "uint32": "Not supported by the torch backend",
    },
    "tensorflow": {"uint32": "uint32 + int promotes to int64, unlike JAX"},
}

# Standard op test inputs: small non-negative ints fit every dtype. `Z`
# broadcasts against `X`. 32-bit, as 64-bit dtypes are not covered yet.
X = np.array([[1, 2, 3], [3, 2, 1]], dtype="int32")
Y = np.array([[4, 5, 6], [3, 2, 1]], dtype="int32")
Z = np.array([[[1, 2, 3], [3, 2, 1]]], dtype="int32")


def named_dtype_combinations(num_inputs, ordered=False, dtypes=None):
    """Returns `(name, dtypes)` pairs of input dtype combinations.

    `dtypes` is one pool shared by all inputs (default: the pools above),
    giving unordered combinations with same-dtype pairs, or every ordered
    combination if `ordered`; or a list of one pool per input, giving their
    product. Names are the joined dtypes, e.g. `"int8_uint32"`. The pairs
    are valid `named_parameters` entries.
    """
    if dtypes is None:
        dtypes = (
            DEFAULT_INPUT_DTYPES if num_inputs == 1 else BINARY_INPUT_DTYPES
        )
    if isinstance(dtypes[0], str):
        if ordered:
            combos = itertools.product(dtypes, repeat=num_inputs)
        else:
            combos = itertools.combinations_with_replacement(dtypes, num_inputs)
    elif len(dtypes) != num_inputs:
        raise ValueError(
            f"Expected {num_inputs} dtype pools, one per input. "
            f"Received: {len(dtypes)}."
        )
    else:
        combos = itertools.product(*dtypes)
    return [("_".join(combo), combo) for combo in combos]


class OpTest(TestCase):
    """Standard tests of an op, driven by class attributes.

    A subclass sets the attributes below and inherits the tests. The
    abstract subclasses `UnaryOpTest`, `BinaryOpTest` and `ReductionOpTest`
    add the standard inputs of their category and the dtype tests.

    - `test_eager`: for every case, the eager outputs equal `reference_fn`.
    - `test_eager_class`: the same, through `op_class(**kwargs)`.
    - `test_symbolic_static`: for every case, the symbolic outputs have the
      structure, dtypes and shapes of the eager outputs.
    - `test_symbolic_dynamic`: the same with every input dimension unknown,
      then with one unknown dimension at a time.
    - `test_input_dtypes_<dtypes>` (abstract subclasses): with the arrays
      of the first case cast to `dtypes`, the eager output dtypes equal
      `jax_fn`'s and the symbolic output dtypes equal the eager ones.

    Tensors under test go in `inputs`. `kwargs` are static arguments: they
    are passed unchanged to the op and to both references. A test defined
    on a subclass replaces the inherited one, e.g. to skip one dtype
    combination with a reason.

    Attributes:
        op: Function form of the op, e.g. `knp.add`.
        op_class: `Operation` class, e.g. `knp.Add`. `None` skips its test.
        reference_fn: NumPy reference, e.g. `np.add`.
        jax_fn: JAX reference for the output dtypes.
        inputs: Tuple of NumPy arrays and Python scalars passed to the op.
        kwargs: Static keyword arguments of the op.
        cases: `{name: (inputs, kwargs)}` to test several inputs instead.
        atol: Absolute tolerance of the eager tests.
        rtol: Relative tolerance of the eager tests.
        run_dynamic_shape_checks: Disable for ops that need static sizes.
    """

    __test__ = False  # Abstract. Subclasses that set `op` are collected.
    op = None
    op_class = None
    reference_fn = None
    jax_fn = None
    inputs = ()
    kwargs = {}
    cases = None
    atol = 1e-6
    rtol = 1e-6
    run_dynamic_shape_checks = True

    def __init_subclass__(cls, **class_kwargs):
        super().__init_subclass__(**class_kwargs)
        cls.__test__ = cls.op is not None
        for name in ("op", "reference_fn", "jax_fn"):
            fn = cls.__dict__.get(name)
            if callable(fn) and not isinstance(fn, staticmethod):
                # A function stored on a class binds `self` when read from
                # an instance; `staticmethod` keeps `self.op` a plain call.
                setattr(cls, name, staticmethod(fn))

    def _cases(self):
        return self.cases or {"": (self.inputs, self.kwargs)}

    def _first_case(self):
        return next(iter(self._cases().values()))

    def test_eager(self):
        for inputs, kwargs in self._cases().values():
            self.check_eager(self.op, self.reference_fn, inputs, kwargs)

    def test_eager_class(self):
        if self.op_class is None:
            self.skipTest("No `op_class`")
        for inputs, kwargs in self._cases().values():
            op = self.op_class(**kwargs)
            reference_fn = functools.partial(self.reference_fn, **kwargs)
            self.check_eager(op, reference_fn, inputs, {})

    def test_symbolic_static(self):
        for inputs, kwargs in self._cases().values():
            self.check_symbolic(inputs, kwargs)

    def test_symbolic_dynamic(self):
        if not self.run_dynamic_shape_checks:
            self.skipTest("Disabled by the test")
        for inputs, kwargs in self._cases().values():
            shapes = [x.shape for x in _arrays(inputs)]
            variants = [_all_dims_unknown(shapes)] + _one_dim_unknown(shapes)
            for symbolic_shapes in variants:
                try:
                    self.check_symbolic(inputs, kwargs, symbolic_shapes)
                except Exception as e:
                    e.add_note(f"Symbolic input shapes: {symbolic_shapes}")
                    raise

    def check_eager(self, op, reference_fn, inputs, kwargs):
        """Checks the structure and values of the eager outputs."""
        outputs = op(*_map_arrays(ops.convert_to_tensor, inputs), **kwargs)
        expected = reference_fn(*inputs, **kwargs)
        msg = f"Unexpected eager output for {_describe(inputs, kwargs)}"
        tree.assert_same_structure(outputs, expected)
        for output, expected_output in zip(
            tree.flatten(outputs), tree.flatten(expected)
        ):
            self.assertAllClose(
                output, expected_output, atol=self.atol, rtol=self.rtol, msg=msg
            )

    def check_symbolic(self, inputs, kwargs, symbolic_shapes=None):
        """Checks the symbolic outputs against the eager ones.

        `symbolic_shapes` has one shape per array input, with `None` for the
        unknown dimensions; by default the arrays' shapes. The symbolic
        outputs must have the structure and dtypes of the eager outputs, and
        their known dimensions.
        """
        arrays = _arrays(inputs)
        if not arrays:
            self.skipTest("No array inputs, so no symbolic path")
        if symbolic_shapes is None:
            symbolic_shapes = [x.shape for x in arrays]
        tensors = [ops.convert_to_tensor(x) for x in arrays]
        keras_tensors = [  # With the dtypes Keras gave the arrays.
            KerasTensor(shape, dtype=standardize_dtype(t.dtype))
            for shape, t in zip(symbolic_shapes, tensors, strict=True)
        ]
        symbolic_inputs = _with_arrays(inputs, keras_tensors)
        symbolic = self.op(*symbolic_inputs, **kwargs)
        eager = self.op(*_with_arrays(inputs, tensors), **kwargs)
        msg = (
            f"Output differs: symbolic={_signature(symbolic)}, "
            f"eager={_signature(eager)} "
            f"for {_describe(symbolic_inputs, kwargs)}"
        )
        tree.assert_same_structure(symbolic, eager)
        for symbolic_output, eager_output in zip(
            tree.flatten(symbolic), tree.flatten(eager)
        ):
            if not _symbolic_matches(symbolic_output, eager_output):
                self.fail(msg)

    def check_input_dtypes(self, dtypes, inputs=None, kwargs=None):
        """Checks the output dtypes with the array inputs cast to `dtypes`.

        `inputs` and `kwargs` default to the first case. The eager output
        dtypes must equal `jax_fn`'s, and the symbolic output dtypes must
        equal the eager ones.
        """
        if inputs is None:
            inputs, kwargs = self._first_case()
        skipped = _SKIPPED_DTYPES.get(backend.backend(), {})
        for dtype in dtypes:
            if dtype in skipped:
                self.skipTest(f"{dtype}: {skipped[dtype]}")
        arrays = [
            x.astype(d) for x, d in zip(_arrays(inputs), dtypes, strict=True)
        ]
        inputs = _with_arrays(inputs, arrays)
        import jax.numpy as jnp

        outputs = self.op(*_map_arrays(ops.convert_to_tensor, inputs), **kwargs)
        jax_outputs = self.jax_fn(*_map_arrays(jnp.asarray, inputs), **kwargs)
        self.assertEqual(
            _dtypes_of(outputs),
            _dtypes_of(jax_outputs),
            f"Output dtype differs from JAX for {_describe(inputs, kwargs)}",
        )
        self.check_symbolic(inputs, kwargs)


class UnaryOpTest(OpTest):
    """`OpTest` of an elementwise unary op, on the standard input `X`."""

    inputs = (X,)

    @parameterized.named_parameters(named_dtype_combinations(1))
    def test_input_dtypes(self, dtypes):
        self.check_input_dtypes(dtypes)


class BinaryOpTest(OpTest):
    """`OpTest` of an elementwise binary op: broadcasting, Python scalars."""

    cases = {"same_shape": ((X, Y), {}), "broadcast": ((X, Z), {})}

    @parameterized.named_parameters(named_dtype_combinations(2))
    def test_input_dtypes(self, dtypes):
        self.check_input_dtypes(dtypes)

    @parameterized.named_parameters(named_dtype_combinations(1))
    def test_python_types(self, dtypes):
        (x, _), kwargs = self._first_case()
        for scalar in (1, 1.0):
            self.check_input_dtypes(dtypes, (x, scalar), kwargs)


class ReductionOpTest(UnaryOpTest):
    """`OpTest` of a reduction over `axis`, with and without `keepdims`."""

    cases = {
        f"{name}_keepdims" if keepdims else name: (
            (X,),
            {"axis": axis, "keepdims": keepdims},
        )
        for name, axis in [
            ("all", None),
            ("axis0", 0),
            ("axis1", 1),
            ("last", -1),
            ("axes01", (0, 1)),
        ]
        for keepdims in (False, True)
    }


def _arrays(inputs):
    """The NumPy arrays among `inputs`; the other inputs are Python scalars."""
    return [x for x in inputs if isinstance(x, np.ndarray)]


def _map_arrays(fn, inputs):
    """Applies `fn` to the arrays among `inputs`; scalars pass through."""
    return tuple(fn(x) if isinstance(x, np.ndarray) else x for x in inputs)


def _with_arrays(inputs, arrays):
    """`inputs` with its arrays replaced by `arrays`, in order."""
    arrays = iter(arrays)
    return tuple(
        next(arrays) if isinstance(x, np.ndarray) else x for x in inputs
    )


def _dtypes_of(outputs):
    return [standardize_dtype(x.dtype) for x in tree.flatten(outputs)]


def _symbolic_matches(symbolic, eager):
    """Whether a `KerasTensor` has the dtype and known dims of a tensor."""
    return (
        isinstance(symbolic, KerasTensor)
        and standardize_dtype(symbolic.dtype) == standardize_dtype(eager.dtype)
        and len(symbolic.shape) == len(eager.shape)
        and all(s in (None, e) for s, e in zip(symbolic.shape, eager.shape))
    )


def _all_dims_unknown(shapes):
    return [(None,) * len(shape) for shape in shapes]


def _one_dim_unknown(shapes):
    """Every way of making one dimension of one of `shapes` unknown."""
    variants = []
    for i, shape in enumerate(shapes):
        for d in range(len(shape)):
            unknown = shape[:d] + (None,) + shape[d + 1 :]
            variants.append(shapes[:i] + [unknown] + shapes[i + 1 :])
    return variants


def _signature(values):
    """Describes inputs or outputs for messages, e.g. `int32[2, 3], 1.0`."""
    return ", ".join(
        f"{standardize_dtype(x.dtype)}{list(x.shape)}"
        if hasattr(x, "shape")
        else repr(x)
        for x in tree.flatten(values)
    )


def _describe(inputs, kwargs):
    return f"inputs {_signature(inputs)}, kwargs {kwargs}"
