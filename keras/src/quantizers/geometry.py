"""Quantization geometry: the layer-side protocol behind the strategy registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The strategies in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

Two geometry families exist today:

- Projection: a float kernel contracted against the inputs. A strategy writes
  one projection implementation and the geometry supplies what differs per
  layer: how to contract (a plain matmul for `Dense` and `TernaryDense`,
  `ProjectionGeometry`; an einsum for `EinsumDense`,
  `EinsumProjectionGeometry`, whose axis analysis lives on the layer
  itself and is reached through the geometry's hooks), which axes the
  quantizers reduce over, how a scale lines up with the kernel and with
  the outputs, and the 2D `(rows, columns)` view of an N-D kernel.
- Lookup: a float embeddings table indexed by the inputs. `Embedding` is the
  plain case (`LookupGeometry`); `ReversibleEmbedding` adds a reverse
  projection (`ReversibleLookupGeometry`).

Making a layer quantizable
--------------------------

Return a geometry, and list the modes the layer supports:

```python
class MyProjection(Layer):
    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        # Doubles as the capability declaration: a mode absent from this
        # mapping is rejected for this layer.
        return {
            None: ["kernel", "bias"],
            "int8": ["kernel", "bias", "kernel_scale"],
        }
```

The geometry is a thin adapter, so the strategies still read
state directly off the layer. Beyond what `Layer` already provides, a
quantizable layer must define:

- Projections: `_kernel` (the float kernel variable), `kernel_shape` (its
  shape, recorded in `build()`), `units`, `bias` and `activation` (either
  may be `None`). `EinsumProjectionGeometry` additionally relies on the
  `einsum_axes` record `EinsumDense` derives from its equation in
  `build()`, and on the equation analysis `EinsumDense` prepares in
  `_set_quantization_info()` (called through the geometry's `prepare()`).
- Lookups: `_embeddings`, `input_dim` and `output_dim`. A reversible
  lookup adds `tie_weights`, `logit_soft_cap`, and, when untied, the
  `reverse_embeddings` variables.

LoRA is optional: `Layer` defines `lora_enabled = False`. A layer that
supports it sets `lora_enabled` in `enable_lora()` and defines
`lora_kernel_a`, `lora_kernel_b`, `lora_alpha` and `lora_rank` (a lookup
defines `lora_embeddings_a` and `lora_embeddings_b` in place of the kernel
factors).

The rest comes from `Layer` itself: strategies read `compute_dtype`,
`dtype_policy` and `path`, create their quantized variables through
`add_weight`, and re-enter through `Layer.quantized_build`, which routes
straight back to the strategy. A layer never needs to know which mode is
running, and implements none of these itself.

Defining `_quantization_geometry()` on a subclass also makes that subclass
the owner of its quantization support: `Layer.quantize`'s type check
accepts instances of the exact class that defines the method. A `Dense`
subclass therefore opts in by defining it; without it the subclass is
skipped by `Model.quantize` and remains reachable through
`quantize(..., type_check=False)`.

Customizing what a strategy does to a layer
-------------------------------------------

Override a geometry hook rather than a strategy method: the hooks on the
classes below are the only points at which strategies vary per
layer. `TernaryDense` is the in-tree example: its geometry supplies its
own straight-through ternarization values, and the ternary strategy needs no
knowledge of the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one strategy's math for itself alone, because that surface lives
on the strategy; a layer that contracts its kernel differently overrides
the geometry hooks, and anything beyond that means replacing the strategy (by
subclassing it, overriding the one handler, and registering it under a
new mode name). A new geometry family, on the other hand, needs no dispatcher
change at all: declare its `family`, list it in the mode's
`geometry_families` and implement the strategy's handlers for it, which
`GeometryDispatchStrategy` lists (`keras.src.quantizers.modes.common`).
"""

import string

from keras.src import ops
from keras.src.quantizers.quantizers import ternarize


class QuantizationGeometry:
    """Base class for a layer's quantization geometry.

    A geometry names the *family* it belongs to. A strategy built on
    `GeometryDispatchStrategy` implements one handler per verb for each
    family it lists in `geometry_families`, so introducing a family is a
    declaration plus those handlers, with no dispatch chain to edit
    anywhere.
    """

    # Dispatch key: each `GeometryDispatchStrategy` verb resolves to the
    # mode's handler for this family.
    family = None
    # Whether the layer also projects back through its weight. Not a
    # dispatch key: strategies branch on it where the reverse table
    # matters, and the forward handler takes the layer's `reverse` argument.
    reversible = False
    # Attributes the layer's `build()` records and the strategies read.
    # `Layer.quantize` checks them before it changes the layer.
    build_attributes = ()

    def __init__(self, layer):
        self.layer = layer

    @property
    def weight_shape(self):
        """Shape of the float weight that quantization replaces."""
        raise NotImplementedError(
            f"{type(self).__name__} must define `weight_shape`."
        )


class ProjectionGeometry(QuantizationGeometry):
    """Geometry of a 2D kernel `(input_dim, units)` contracted by matmul."""

    family = "projection"
    build_attributes = ("kernel_shape",)

    @property
    def weight_shape(self):
        """Shape of the float kernel, as the layer recorded it in `build()`.

        Quantized storage may be packed or flattened, so this is the
        logical shape every strategy reads rather than a variable's shape.
        """
        return tuple(self.layer.kernel_shape)

    def prepare(self):
        """Computes any layout analysis the geometry needs (idempotent)."""

    def calibration_rows_columns(self, kernel_shape):
        """2D `(rows, columns)` view used by the calibration strategies.

        Kept apart from `rows_columns`: the calibration strategies may split
        a kernel by a different rule than the weight-only ones.
        """
        return kernel_shape[0], kernel_shape[1]

    def contract(self, inputs, kernel):
        """Contracts `inputs` against a kernel in the contraction shape."""
        return ops.matmul(inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        """Gradient of `contract` with respect to its inputs."""
        return ops.matmul(upstream, ops.transpose(float_kernel))

    def rows_columns(self, kernel_shape):
        """2D `(rows, columns)` shape a plain reshape of the kernel takes.

        `rows` is the product of the contracted axes and `columns` that of
        the rest, so the reshape is `(contracted, rest)` only when the
        contracted axes lead the kernel.
        """
        return kernel_shape[0], kernel_shape[1]

    @property
    def kernel_reduced_axes(self):
        """Kernel axes a weight quantizer reduces over."""
        return 0

    @property
    def inputs_quantization_axis(self):
        """Input axes an activation quantizer reduces over."""
        return -1

    def align_inputs_scale(self, scale):
        """Aligns an activation scale with the contraction's outputs."""
        return scale

    def kernel_scale_shape(self, kernel_shape):
        """Shape of a per-channel scale stored alongside the kernel."""
        return (kernel_shape[1],)

    @property
    def kernel_scale_axis(self):
        """Kernel axis a per-channel scale is shared along.

        `None` when the stored scale is laid out for the outputs and
        `kernel_scale_for_dequant` lays it out against the kernel instead.
        """
        return 0

    def kernel_scale_for_storage(self, scale):
        """Aligns a freshly computed kernel scale with its stored layout."""
        return ops.squeeze(scale, axis=0)

    def kernel_scale_for_dequant(self, scale):
        """Aligns the stored kernel scale with the kernel for dequantization."""
        return scale

    def add_lora_delta(self, inputs, x):
        """Adds the LoRA update to the contraction's output, when enabled."""
        layer = self.layer
        if layer.lora_enabled:
            lora_x = ops.matmul(inputs, layer.lora_kernel_a)
            lora_x = ops.matmul(lora_x, layer.lora_kernel_b)
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, layer.compute_dtype)
        return x

    def ternary_values(self):
        """Returns `(ternary_kernel, scale)` for ternary quantization.

        The default applies the BitNet b1.58 rule to the float kernel
        (`quantizers.ternarize`): `threshold = 0.5 * mean(|W|)` and
        `scale = mean(|W|)`. A layer that owns its own ternarization rule
        (`TernaryDense` and its straight-through estimator) overrides this
        in its geometry.
        """
        return ternarize(self.layer._kernel)


def _lora_equations(equation):
    """The two einsums that apply a LoRA update to `equation` in low-rank
    form, contracting the rank axis by name.

    `lora_kernel_a` carries the rank on the kernel's last axis and
    `lora_kernel_b` maps it to that axis's size. Contracting the rank with
    a matmul would only work when the kernel's last axis is also the last
    axis of the output; naming it works for every equation (an ellipsis in
    the output, a permuted output, or a kernel whose last axis is
    contracted away).

    Returns:
        `(first, second, a_first)`: `first` contracts the inputs against
        the factor that shares their subscripts, `second` contracts the
        rank axis against the other factor; `a_first` is whether that
        order is `(lora_kernel_a, lora_kernel_b)`, which holds when the
        kernel's last axis survives in the output.
    """
    inputs_spec, rest = equation.split(",")
    kernel_spec, output_spec = rest.split("->")
    last = kernel_spec[-1]
    rank = next(c for c in string.ascii_letters if c not in equation)
    if last in output_spec:
        mid = output_spec.replace(last, rank)
        return (
            f"{inputs_spec},{kernel_spec[:-1]}{rank}->{mid}",
            f"{mid},{rank}{last}->{output_spec}",
            True,
        )
    mid = inputs_spec.replace(last, rank)
    return (
        f"{inputs_spec},{rank}{last}->{mid}",
        f"{mid},{kernel_spec[:-1]}{rank}->{output_spec}",
        False,
    )


class EinsumProjectionGeometry(ProjectionGeometry):
    """Geometry of an N-D einsum kernel (`EinsumDense`).

    The equation-derived axis analysis (reduced/transpose/expand/squeeze
    axes, the custom-gradient equation) is the layer's own geometry
    implementation; this class routes the strategies to it.
    """

    build_attributes = ("kernel_shape", "einsum_axes")

    def prepare(self):
        self.layer._set_quantization_info()

    def calibration_rows_columns(self, kernel_shape):
        if len(kernel_shape) not in (2, 3):
            raise ValueError(
                "Calibration only supports 2D or 3D kernels. Received: "
                f"kernel_shape={tuple(kernel_shape)}"
            )
        if len(kernel_shape) == 2:
            return kernel_shape[0], kernel_shape[1]
        # 3D kernels are split by locating the model dimension (the largest
        # one): [d_model, heads, head_dim] is a QKV projection, while
        # [heads, head_dim, d_model] is an attention output projection.
        shape = list(kernel_shape)
        d_model_dim_index = shape.index(max(shape))
        if d_model_dim_index == 0:  # QKV projection case
            in_features, heads, head_dim = shape
            return in_features, heads * head_dim
        elif d_model_dim_index in [1, 2]:  # Attention Output case
            heads, head_dim, out_features = shape
            return heads * head_dim, out_features
        raise ValueError("Could not determine row/column split.")

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        # From https://stackoverflow.com/a/47609896
        return ops.einsum(
            self.layer.einsum_axes.gradient_equation, upstream, float_kernel
        )

    def rows_columns(self, kernel_shape):
        rows = 1
        columns = 1
        for i, dim in enumerate(kernel_shape):
            if i in self.layer._kernel_reduced_axes:
                rows *= dim
            else:
                columns *= dim
        return rows, columns

    @property
    def kernel_reduced_axes(self):
        return self.layer._kernel_reduced_axes

    @property
    def inputs_quantization_axis(self):
        return self.layer.einsum_axes.input_reduced_axes

    def align_inputs_scale(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "input")

    def kernel_scale_shape(self, kernel_shape):
        return self.layer._get_kernel_scale_shape(kernel_shape)

    @property
    def kernel_scale_axis(self):
        # The equation analysis may transpose or expand the stored scale
        # even for a 2-D kernel; `kernel_scale_for_dequant` lays it out.
        return None

    def kernel_scale_for_storage(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "kernel")

    def kernel_scale_for_dequant(self, scale):
        return self.layer._adjust_scale_for_dequant(scale)

    def add_lora_delta(self, inputs, x):
        layer = self.layer
        if layer.lora_enabled:
            first, second, a_first = _lora_equations(layer.equation)
            factors = (layer.lora_kernel_a, layer.lora_kernel_b)
            if not a_first:
                factors = factors[::-1]
            lora_x = ops.einsum(first, inputs, factors[0])
            lora_x = ops.einsum(second, lora_x, factors[1])
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, dtype=layer.compute_dtype)
        return x


class LookupGeometry(QuantizationGeometry):
    """Geometry of an embeddings table indexed by integer inputs."""

    family = "lookup"

    @property
    def weight_shape(self):
        return (self.layer.input_dim, self.layer.output_dim)


class ReversibleLookupGeometry(LookupGeometry):
    """Lookup geometry with a reverse projection (`ReversibleEmbedding`)."""

    reversible = True
