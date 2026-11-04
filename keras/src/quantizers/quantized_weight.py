"""A read-only view over the stored variables of one quantized weight.

A quantization mode stores a weight as integer codes (often packed several
to a byte), a scale, and for a grouped scheme a zero point and a group
index. `QuantizedWeight` gathers those variables with a `PackLayout`, which
says how the codes are packed, and a `WeightScheme`, which says what they
mean.
"""

import copy
import dataclasses
import math

from keras.src import backend
from keras.src import ops
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.packing import pack_int2
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.packing import pack_ternary
from keras.src.quantizers.packing import unpack_int2
from keras.src.quantizers.packing import unpack_int4
from keras.src.quantizers.packing import unpack_ternary
from keras.src.quantizers.quantizers import dequantize_with_sz_map

# The stored tensors of a view, in the order `read_tensors` returns them.
_TENSOR_FIELDS = ("codes", "scale", "zero_point", "g_idx", "input_scales")


@dataclasses.dataclass(frozen=True, kw_only=True)
class WeightScheme:
    """How the integer codes of one quantized weight map to real values.

    Args:
        code_range: `(min, max)` of the unpacked codes, e.g. `(-127, 127)`
            for int8 or `(0, 15)` for unsigned 4-bit codes.
        scale_form: `"divisor"` when the real value is
            `(code - zero_point) / scale` (int8, per-channel int4), or
            `"multiplier"` when it is `(code - zero_point) * scale`
            (grouped int4, GPTQ, AWQ, ternary). A grouped scale is a
            multiplier.
        has_zero_point: Whether a zero point is stored.
        group_size: Length of a group along the view's `axis` for a grouped
            scale, or `None` for a per-channel or per-tensor scale. `g_idx`
            is always authoritative: a group may be shorter, and under
            GPTQ's activation order the groups are not contiguous.
    """

    code_range: tuple
    scale_form: str
    has_zero_point: bool = False
    group_size: int = None

    def __post_init__(self):
        if self.scale_form not in ("divisor", "multiplier"):
            raise ValueError(
                "`scale_form` must be 'divisor' or 'multiplier'. "
                f"Received: scale_form={self.scale_form!r}"
            )
        if self.group_size is not None and (
            self.scale_form != "multiplier" or not self.has_zero_point
        ):
            raise ValueError(
                "A grouped scheme stores a multiplier scale and a zero point. "
                f"Received: scale_form={self.scale_form!r}, "
                f"has_zero_point={self.has_zero_point}, "
                f"group_size={self.group_size}"
            )


class PackLayout:
    """How the codes of one weight are packed in their storage variable.

    One subclass per storage format: `values_per_byte` codes share a
    stored element along the layout's axis, `packed_length` gives the
    stored length of that axis, `unpack` restores one code per element
    and `pack` stores them.
    """

    values_per_byte = 1

    def unpack(self, codes):
        """Returns the unpacked codes, one per element."""
        raise NotImplementedError

    def pack(self, codes):
        """Returns `codes`, one per element, in their stored form."""
        raise NotImplementedError

    @classmethod
    def packed_length(cls, length):
        """Stored length of an axis holding `length` codes."""
        return length

    def __repr__(self):
        return f"{type(self).__name__}()"


class NoPack(PackLayout):
    """One code per stored element."""

    def unpack(self, codes):
        return codes

    def pack(self, codes):
        return codes


class _AxisPack(PackLayout):
    """A layout that packs several codes per byte along one axis."""

    def __init__(self, axis, orig_len):
        self.axis = axis
        self.orig_len = orig_len

    @classmethod
    def packed_length(cls, length):
        return math.ceil(length / cls.values_per_byte)

    def __repr__(self):
        return (
            f"{type(self).__name__}(axis={self.axis}, orig_len={self.orig_len})"
        )


class Int4Pairs(_AxisPack):
    """Two 4-bit codes per byte along `axis`, as `pack_int4` writes them.

    The codes' own dtype (`int8` or `uint8`) says whether a nibble is
    sign-extended on unpack.
    """

    values_per_byte = 2

    def unpack(self, codes):
        dtype = standardize_dtype(codes.dtype)
        return unpack_int4(codes, self.orig_len, axis=self.axis, dtype=dtype)

    def pack(self, codes):
        dtype = standardize_dtype(codes.dtype)
        packed, _, _ = pack_int4(codes, axis=self.axis, dtype=dtype)
        return packed


class Int2Quads(_AxisPack):
    """Four 2-bit codes per byte along `axis`, as `pack_int2` writes them."""

    values_per_byte = 4

    def unpack(self, codes):
        dtype = standardize_dtype(codes.dtype)
        return unpack_int2(codes, self.orig_len, axis=self.axis, dtype=dtype)

    def pack(self, codes):
        dtype = standardize_dtype(codes.dtype)
        packed, _, _ = pack_int2(codes, axis=self.axis, dtype=dtype)
        return packed


class TernaryTrits(_AxisPack):
    """Five ternary codes per byte along `axis`, in base 3.

    The unpack is arithmetic, not a mask and shift, so a bit-width cannot
    describe it.
    """

    values_per_byte = 5

    def unpack(self, codes):
        return unpack_ternary(codes, self.orig_len, axis=self.axis)

    def pack(self, codes):
        packed, _, _ = pack_ternary(codes, axis=self.axis)
        return packed


class QuantizedWeight:
    """A read-only view over the stored variables of one quantized weight.

    The view holds references to the tensors a mode stores for the weight
    (the layer's variables, or traced tensors inside a forward pass) and
    does not cache, so it reads their current values.
    `QuantizationStrategy.quantized_weight(layer)` builds it on demand. A
    forward pass that passes the tensors to `ops.custom_gradient` reads
    them with `read_tensors` and gets the same view over them with
    `with_tensors`.

    The codes are stored as the weight reshaped: `layout.unpack(codes)` is
    `reshape(W, s)`, where `W` is the weight in `shape` and `s` is the
    unpacked shape of `codes` (an int4, GPTQ or AWQ einsum kernel is stored
    as 2-D). `axis` and the stored scale refer to these stored coordinates;
    `unpack()` and `dequantize()` return `shape`.

    Args:
        codes: The stored codes, packed as `layout` describes.
        scale: The stored scale, applied as `scheme.scale_form` says.
        layout: The `PackLayout` of `codes`.
        scheme: The `WeightScheme` of the weight.
        shape: Shape of the weight the codes stand for.
        axis: Axis of the unpacked codes that a scale entry is shared
            along. A per-channel scale has the codes' shape without this
            axis. A grouped scale and zero point have one entry per group
            along it, and `g_idx` maps each position on it to its group.
            `None` when the scale broadcasts against the codes as it is (a
            per-tensor scalar) or `align_scale` lays it out.
        zero_point: The stored zero point, given exactly when
            `scheme.has_zero_point`.
        g_idx: The stored group index, one entry per position along
            `axis`, given exactly when `scheme.group_size` is set.
        input_scales: Optional scales with one entry per position along
            `axis`, divided out of the weight (AWQ's `awq_scales`). The
            codes, scale and zero point describe the weight multiplied by
            them.
        align_scale: Optional callable that lays the stored scale out
            against the codes, for a scale stored in another layout (an
            int8 einsum kernel's scale is stored for the outputs). Needs
            `axis=None` and an ungrouped scheme; `scale` stays the stored
            variable.
    """

    def __init__(
        self,
        *,
        codes,
        scale,
        layout,
        scheme,
        shape,
        axis=None,
        zero_point=None,
        g_idx=None,
        input_scales=None,
        align_scale=None,
    ):
        if scheme.has_zero_point != (zero_point is not None):
            raise ValueError(
                "`zero_point` must be given exactly when the scheme has a "
                f"zero point. Received: scheme={scheme!r}, "
                f"zero_point={'given' if zero_point is not None else None}"
            )
        if (scheme.group_size is not None) != (g_idx is not None):
            raise ValueError(
                "`g_idx` must be given exactly when the scheme is grouped. "
                f"Received: scheme={scheme!r}, "
                f"g_idx={'given' if g_idx is not None else None}"
            )
        if (g_idx is not None or input_scales is not None) and not isinstance(
            axis, int
        ):
            raise ValueError(
                "`g_idx` and `input_scales` have one entry per position "
                "along `axis`, so they need one integer `axis`. Received: "
                f"axis={axis}, "
                f"g_idx={'given' if g_idx is not None else None}, "
                f"input_scales={'given' if input_scales is not None else None}"
            )
        if align_scale is not None and (
            axis is not None or scheme.group_size is not None
        ):
            raise ValueError(
                "`align_scale` lays the stored scale out itself, so it "
                "needs `axis=None` and an ungrouped scheme. Received: "
                f"axis={axis}, scheme={scheme!r}"
            )
        self.codes = codes
        self.scale = scale
        self.layout = layout
        self.scheme = scheme
        self.shape = tuple(int(d) for d in shape)
        self.axis = axis
        self.zero_point = zero_point
        self.g_idx = g_idx
        self.input_scales = input_scales
        self.align_scale = align_scale

    def read_tensors(self):
        """Returns the stored tensors, each variable through its `value`.

        The order is `codes` and `scale`, then `zero_point`, `g_idx` and
        `input_scales` when given. A float variable that autocasts reads in
        the dtype of the current autocast scope.
        """
        tensors = [getattr(self, name) for name in self._tensor_names()]
        return tuple(
            ops.convert_to_tensor(
                tensor.value if isinstance(tensor, backend.Variable) else tensor
            )
            for tensor in tensors
        )

    def with_tensors(self, tensors):
        """Returns this view over `tensors`, in the order of `read_tensors`."""
        view = copy.copy(self)
        for name, tensor in zip(self._tensor_names(), tensors, strict=True):
            setattr(view, name, tensor)
        return view

    def _tensor_names(self):
        """Names of the stored tensors that are given, in order."""
        return [
            name for name in _TENSOR_FIELDS if getattr(self, name) is not None
        ]

    def unpack(self):
        """Returns the integer codes in `shape`."""
        return self._restore_shape(self.layout.unpack(self.codes))

    def dequantize(self, dtype):
        """Returns the real-valued weight in `shape`, as `dtype`.

        The codes are cast to `dtype` before the arithmetic, and the result
        is cast to `dtype` after it.
        """
        codes = ops.cast(self.layout.unpack(self.codes), dtype)
        if self.g_idx is not None:
            weight = dequantize_with_sz_map(
                codes,
                self.scale,
                self.zero_point,
                self.g_idx,
                group_axis=self.axis,
            )
        else:
            if self.zero_point is not None:
                zero_point = ops.cast(self._align(self.zero_point), dtype)
                codes = ops.subtract(codes, zero_point)
            if self.scheme.scale_form == "multiplier":
                weight = ops.multiply(codes, self._align(self.scale))
            else:
                weight = ops.divide(codes, self._align(self.scale))
        if self.input_scales is not None:
            shape = [1] * len(weight.shape)
            shape[self.axis] = -1
            weight = ops.divide(weight, ops.reshape(self.input_scales, shape))
        return self._restore_shape(ops.cast(weight, dtype))

    def _align(self, tensor):
        """Lays a per-channel scale or zero point out against the codes."""
        if self.align_scale is not None:
            return self.align_scale(tensor)
        if self.axis is None:
            return tensor
        return ops.expand_dims(tensor, self.axis)

    def _restore_shape(self, tensor):
        """Reshapes stored coordinates to `shape`."""
        if tuple(tensor.shape) != self.shape:
            tensor = ops.reshape(tensor, self.shape)
        return tensor

    def __repr__(self):
        return (
            f"{type(self).__name__}(shape={self.shape}, axis={self.axis}, "
            f"layout={self.layout!r}, scheme={self.scheme!r})"
        )
