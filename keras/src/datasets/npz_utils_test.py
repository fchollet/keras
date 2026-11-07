import io
import os
import pickle
import zipfile
from unittest import mock

import numpy as np

from keras.src import testing
from keras.src.datasets import npz_utils


class LoadNpzTest(testing.TestCase):
    def test_loads_ragged_object_arrays(self):
        # IMDB/Reuters store ragged sequences as `dtype=object` arrays.
        xs = np.array([[1, 14, 22], [1, 194], [1, 14, 47, 8]], dtype=object)
        ys = np.array([1, 0, 1])
        path = os.path.join(self.get_temp_dir(), "ragged.npz")
        np.savez(path, x=xs, y=ys)

        loaded = npz_utils.load_npz(path)

        self.assertEqual(
            [list(seq) for seq in loaded["x"]],
            [[1, 14, 22], [1, 194], [1, 14, 47, 8]],
        )
        self.assertAllEqual(loaded["y"], ys)

    def test_loads_numeric_arrays(self):
        path = os.path.join(self.get_temp_dir(), "numeric.npz")
        np.savez(path, a=np.arange(10), b=np.ones((3, 4), dtype="float32"))

        loaded = npz_utils.load_npz(path)

        self.assertAllEqual(loaded["a"], np.arange(10))
        self.assertEqual(loaded["b"].shape, (3, 4))
        self.assertEqual(loaded["b"].dtype, np.float32)

    def test_rejects_pickle_gadget(self):
        # A crafted member whose unpickling would run code must be refused,
        # without executing the payload.
        marker = os.path.join(self.get_temp_dir(), "marker")

        class Exploit:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        payload = np.empty(1, dtype=object)
        payload[0] = Exploit()
        path = os.path.join(self.get_temp_dir(), "evil.npz")
        np.savez(path, x=payload)

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)
        self.assertFalse(os.path.exists(marker))

    def test_rejects_decompression_bomb_member(self):
        path = os.path.join(self.get_temp_dir(), "bomb.npz")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("x.npy", b"\0" * 200_000)

        with (
            mock.patch.object(npz_utils, "_NPZ_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(npz_utils, "_NPZ_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                npz_utils.load_npz(path)


_RECONSTRUCT = (getattr(np, "_core", None) or np.core).multiarray._reconstruct


def _crafted_npz(path, construct_args, state):
    """Writes a `.npz` whose `.npy` member holds a hand-built pickle stream.

    `construct_args` are the arguments handed to `multiarray._reconstruct` and
    `state` the 5-tuple numpy's `__reduce__` writes, so the tests below can
    describe an array that is deliberately inconsistent with what the
    constructor allocated.
    """

    class Crafted:
        def __reduce__(self):
            return (_RECONSTRUCT, construct_args, state())

    header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    header += b" " * ((64 - (len(header) + 10) % 64) % 64)
    npy = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    npy += pickle.dumps(Crafted(), protocol=3)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.npy", npy)
    return path


def _raw_npy_member(path, header, payload=b""):
    """Writes a `.npz` whose `.npy` member has a raw header and payload.

    Used to craft members whose *header* (rather than pickle state) describes a
    huge or truncated numeric array, which never reaches the unpickler.
    """
    header += b" " * ((64 - (len(header) + 10) % 64) % 64)
    npy = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    npy += payload
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.npy", npy)
    return path


class LoadNpzStateHardeningTest(testing.TestCase):
    def test_object_array_elements_are_plain_ndarrays(self):
        # The validating subclass is an implementation detail: it must not leak
        # into the arrays nested inside an object array. IMDB/Reuters hand back
        # `x_train[i]` as a plain `np.ndarray`, so `type(seq) is np.ndarray`
        # has to hold for every element, not only for the outer array.
        xs = np.empty(3, dtype=object)
        sequences = ([1, 14, 22], [1, 194], [1, 14, 47, 8])
        for index, sequence in enumerate(sequences):
            xs[index] = np.array(sequence)
        path = os.path.join(self.get_temp_dir(), "objects.npz")
        np.savez(path, x=xs)

        loaded = npz_utils.load_npz(path)

        self.assertIs(type(loaded["x"]), np.ndarray)
        for element in loaded["x"]:
            self.assertIs(type(element), np.ndarray)
            self.assertEqual(element.dtype, np.dtype("int64"))

    def test_loads_nested_object_array_with_plain_elements(self):
        # An object array can nest another object array; every level has to be
        # unwrapped, not just the outermost one.
        inner = np.empty(2, dtype=object)
        inner[0] = np.arange(3)
        inner[1] = np.arange(5)
        outer = np.empty(1, dtype=object)
        outer[0] = inner
        path = os.path.join(self.get_temp_dir(), "nested.npz")
        np.savez(path, x=outer)

        loaded = npz_utils.load_npz(path)

        self.assertIs(type(loaded["x"]), np.ndarray)
        self.assertIs(type(loaded["x"][0]), np.ndarray)
        for element in loaded["x"][0]:
            self.assertIs(type(element), np.ndarray)

    def test_rejects_array_state_dtype_confusion(self):
        # Only allow-listed globals are used, but the state declares a larger
        # dtype than `_reconstruct` allocated for the same shape: applying it
        # used to write past the end of the buffer and kill the interpreter.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "confusion.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (
                1,
                (1,),
                np.dtype([("a", "i8"), ("b", "O")]),
                False,
                b"\x00" * 16,
            ),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_array_state_byte_overflow(self):
        # Same idea without changing the dtype: the state describes more bytes
        # than the array holds.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "overflow.npz"),
            (np.ndarray, (1,), np.dtype("i8")),
            lambda: (1, (8,), np.dtype("i8"), False, b"\x00" * 64),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_carrying_raw_bytes(self):
        # An object array must carry its contents as a list of objects; raw
        # bytes would be reread as object pointers.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "rawbytes.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O"), False, b"\x00" * 8),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_fewer_objects_than_shape(self):
        # A state that carries fewer objects than `shape` has elements used to
        # make numpy index past the end of the list without any bounds check,
        # which segfaulted the interpreter.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_too_few.npz"),
            (np.ndarray, (2, 3), np.dtype("O")),
            lambda: (1, (2, 3), np.dtype("O"), False, [None] * 5),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_empty_object_list(self):
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_empty.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O"), False, []),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_more_objects_than_shape(self):
        # A list longer than the array is silently truncated by numpy, and the
        # surplus objects are dropped without being released.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_too_many.npz"),
            (np.ndarray, (2,), np.dtype("O")),
            lambda: (1, (2,), np.dtype("O"), False, [None] * 3),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_numeric_state_carrying_a_list(self):
        # Raw bytes are the only thing numpy writes for a numeric array; a list
        # used to reach numpy and raise a bare `TypeError`.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "numeric_list.npz"),
            (np.ndarray, (2,), np.dtype("i8")),
            lambda: (1, (2,), np.dtype("i8"), False, [1, 2]),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_shape_with_non_integer_dimension(self):
        # `int(dimension)` truncated a float shape silently, and `inf` raised an
        # `OverflowError` that escaped as something other than an unpickling
        # error, so both are now refused before any conversion happens.
        for name, dimension in (
            ("float", 1.5),
            ("integral_float", 1.0),
            ("infinity", float("inf")),
            ("nan", float("nan")),
            ("bool", True),
            ("string", "1"),
        ):
            path = _crafted_npz(
                os.path.join(self.get_temp_dir(), f"shape_{name}.npz"),
                (np.ndarray, (1,), np.dtype("i8")),
                lambda dimension=dimension: (
                    1,
                    (dimension,),
                    np.dtype("i8"),
                    False,
                    b"\x00" * 8,
                ),
            )

            with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
                npz_utils.load_npz(path)

    def test_rejects_non_tuple_shape(self):
        # A shape that is not a tuple (here a list) must be refused rather than
        # handed to numpy.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "shape_list.npz"),
            (np.ndarray, [1], np.dtype("i8")),
            lambda: (1, [1], np.dtype("i8"), False, b"\x00" * 8),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_negative_shape(self):
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "shape_negative.npz"),
            (np.ndarray, (-1,), np.dtype("i8")),
            lambda: (1, (-1,), np.dtype("i8"), False, b""),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_carrying_no_data(self):
        # An object array must carry a list of objects; `None` used to slip
        # past the validator and reach numpy, which raised a bare `TypeError`.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_none.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O"), False, None),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_dtype_that_raises_value_error(self):
        # A crafted state whose dtype spec is well-typed but malformed (here a
        # duplicated field name) made `numpy.dtype` raise `ValueError`, which
        # escaped as a plain `ValueError` instead of an unpickling error.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "dtype_value_error.npz"),
            (np.ndarray, (1,), np.dtype("i8")),
            lambda: (1, (1,), {"names": ["a", "a"]}, False, b"\x00" * 8),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_huge_numeric_shape_without_allocating(self):
        # A tiny member whose header claims `shape=(2**40,)` of `int8` used to
        # make numpy request 1 TiB and raise a `MemoryError` that escaped
        # `load_npz`. The header is now validated before anything is allocated.
        header = (
            b"{'descr': '|i1', 'fortran_order': False, "
            b"'shape': (1099511627776,), }"
        )
        path = _raw_npy_member(
            os.path.join(self.get_temp_dir(), "huge_numeric.npz"), header
        )

        with self.assertRaisesRegex(ValueError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_huge_object_shape_without_allocating(self):
        # Same for the pickle path: the `_reconstruct` arguments are bounded
        # before numpy allocates the object-pointer buffer.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "huge_object.npz"),
            (np.ndarray, (2**40,), np.dtype("O")),
            lambda: (1, (2**40,), np.dtype("O"), False, []),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_corrupt_numeric_member_is_not_reinterpreted_as_objects(self):
        # A truncated *numeric* member must fail as a format error. It used to
        # be rewound and re-fed to the pickle unpickler by a blanket
        # `except ValueError`, surfacing a misleading pickle-domain error.
        header = b"{'descr': '|i8', 'fortran_order': False, 'shape': (4,), }"
        path = _raw_npy_member(
            os.path.join(self.get_temp_dir(), "truncated.npz"),
            header,
            b"\x00" * 8,
        )

        with self.assertRaises(ValueError) as ctx:
            npz_utils.load_npz(path)
        self.assertNotIsInstance(ctx.exception, pickle.UnpicklingError)

    def test_loads_empty_object_array(self):
        # A zero-element object array legitimately carries an empty list, which
        # the object-state length check has to keep accepting.
        path = os.path.join(self.get_temp_dir(), "empty_object.npz")
        np.savez(path, x=np.empty((0,), dtype=object))

        loaded = npz_utils.load_npz(path)

        self.assertEqual(loaded["x"].shape, (0,))
        self.assertEqual(loaded["x"].dtype, np.dtype(object))

    def test_load_unwraps_arrays_inside_a_dict_payload(self):
        # `cifar.py` calls `RestrictedUnpickler(...).load()` directly on a
        # pickle stream whose top-level object is a dict of arrays, so every
        # array in the result has to come back as a plain `np.ndarray`,
        # including an object array nested inside the dict.
        ragged = np.empty(2, dtype=object)
        ragged[0] = np.arange(3)
        ragged[1] = np.arange(4)
        payload = {
            b"data": np.arange(6).reshape(2, 3),
            b"labels": np.array([0, 1]),
            b"ragged": ragged,
        }
        stream = io.BytesIO(pickle.dumps(payload, protocol=3))

        loaded = npz_utils.RestrictedUnpickler(stream, encoding="bytes").load()

        self.assertIs(type(loaded), dict)
        self.assertIs(type(loaded[b"data"]), np.ndarray)
        self.assertIs(type(loaded[b"labels"]), np.ndarray)
        self.assertIs(type(loaded[b"ragged"]), np.ndarray)
        self.assertEqual(loaded[b"data"].tolist(), [[0, 1, 2], [3, 4, 5]])
        for element in loaded[b"ragged"]:
            self.assertIs(type(element), np.ndarray)
            self.assertEqual(element.dtype, np.dtype("int64"))

    def test_rejects_huge_shape_deferred_to_the_pickle_state(self):
        # numpy's own `__reduce__` allocates `(0,)` and puts the real shape and
        # dtype in the state, so a crafted stream can keep the constructor
        # request tiny while asking `__setstate__` for a huge buffer. The byte
        # bound has to hold there too, not only in `_reconstruct_array`.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "huge_state.npz"),
            (np.ndarray, (0,), np.dtype("O")),
            lambda: (1, (2**40,), np.dtype("O"), False, []),
        )

        with self.assertRaisesRegex(
            pickle.UnpicklingError, "declaring more than"
        ):
            npz_utils.load_npz(path)

    def test_rejects_malformed_header_shape_as_value_error(self):
        # A malformed shape in the `.npy` *header* is a format error, so it has
        # to fail as `ValueError`; the same validation reports an
        # `UnpicklingError` only when the malformed shape arrives through a
        # pickle state.
        for name, shape in (("negative", "(-1,)"), ("float", "(1.5,)")):
            header = (
                b"{'descr': '|i8', 'fortran_order': False, "
                b"'shape': " + shape.encode() + b", }"
            )
            path = _raw_npy_member(
                os.path.join(self.get_temp_dir(), f"bad_header_{name}.npz"),
                header,
            )

            with self.assertRaises(ValueError) as ctx:
                npz_utils.load_npz(path)
            self.assertNotIsInstance(ctx.exception, pickle.UnpicklingError)

    def test_rejects_state_that_is_not_a_five_tuple(self):
        # numpy always writes a 5-tuple state; anything else is hand-crafted.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "state_short.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O")),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "5-element tuple"):
            npz_utils.load_npz(path)

    def test_rejects_unknown_state_version(self):
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "state_version.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (2, (1,), np.dtype("O"), False, []),
        )

        with self.assertRaisesRegex(
            pickle.UnpicklingError, "unknown state version"
        ):
            npz_utils.load_npz(path)

    def test_rejects_numeric_state_with_the_wrong_byte_count(self):
        # A numeric array's state must carry exactly `declared` raw bytes.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "numeric_short.npz"),
            (np.ndarray, (1,), np.dtype("i8")),
            lambda: (1, (1,), np.dtype("i8"), False, b""),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "bytes for a"):
            npz_utils.load_npz(path)

    def test_rejects_non_array_subtype(self):
        # `_reconstruct` is only meaningful for an `ndarray` subtype.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "bad_subtype.npz"),
            (np.dtype("i8"), (1,), np.dtype("i8")),
            lambda: (1, (1,), np.dtype("i8"), False, b"\x00" * 8),
        )

        with self.assertRaisesRegex(
            pickle.UnpicklingError, "non-array subtype"
        ):
            npz_utils.load_npz(path)

    def test_loads_version_2_and_3_npy_members(self):
        # `_read_npy_header` accepts the 1.0, 2.0 and 3.0 formats; 2.0's reader
        # also covers 3.0, which only differs by header encoding.
        for version in ((2, 0), (3, 0)):
            array = np.arange(4, dtype="int64")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, version=version)
            path = os.path.join(
                self.get_temp_dir(), f"version_{version[0]}.npz"
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("x.npy", buffer.getvalue())

            loaded = npz_utils.load_npz(path)

            self.assertAllEqual(loaded["x"], array)

    def test_rejects_unsupported_npy_version(self):
        header = b"{'descr': '|i8', 'fortran_order': False, 'shape': (1,), }"
        header += b" " * ((64 - (len(header) + 10) % 64) % 64)
        member = b"\x93NUMPY\x04\x00" + len(header).to_bytes(2, "little")
        member += header
        path = os.path.join(self.get_temp_dir(), "bad_version.npz")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.npy", member)

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            npz_utils.load_npz(path)

    def test_rejects_member_without_the_npy_magic(self):
        path = os.path.join(self.get_temp_dir(), "bad_magic.npz")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.npy", b"NOT-A-NPY-FILE")

        with self.assertRaisesRegex(ValueError, "unreadable header"):
            npz_utils.load_npz(path)

    def test_skips_members_that_are_not_npy(self):
        path = os.path.join(self.get_temp_dir(), "extra_member.npz")
        np.savez(path, a=np.arange(3))
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("readme.txt", b"not an array")

        loaded = npz_utils.load_npz(path)

        self.assertEqual(set(loaded), {"a"})

    def test_load_unwraps_arrays_inside_containers(self):
        # A pickle stream can build a tuple/list alongside dicts and arrays.
        payload = (np.arange(2), [np.arange(3)], {b"a": np.arange(4)})
        stream = io.BytesIO(pickle.dumps(payload, protocol=3))

        loaded = npz_utils.RestrictedUnpickler(stream).load()

        self.assertIs(type(loaded), tuple)
        self.assertIs(type(loaded[0]), np.ndarray)
        self.assertIs(type(loaded[1]), list)
        self.assertIs(type(loaded[1][0]), np.ndarray)
        self.assertIs(type(loaded[2]), dict)

    def test_stream_remaining_returns_none_when_unseekable(self):
        # A stream without `tell`/`seek` cannot be bounded: the size check is
        # skipped rather than raising.
        class Unseekable:
            def tell(self):
                raise OSError("cannot tell")

        self.assertIsNone(npz_utils._stream_remaining(Unseekable()))

    def test_rejects_direct_ndarray_constructor_that_would_allocate_huge(
        self,
    ):
        # A REDUCE stream that calls `numpy.ndarray(shape, dtype)` directly
        # bypasses `_reconstruct_array`/`__setstate__`; `__new__` must catch
        # the huge allocation before numpy's C allocator raises MemoryError.
        path = os.path.join(self.get_temp_dir(), "direct_new.npz")

        class DirectNew:
            def __reduce__(self):
                return (np.ndarray, ((2**40,), "i8"))

        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
        header += b" " * ((64 - (len(header) + 10) % 64) % 64)
        npy = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
        npy += pickle.dumps(DirectNew(), protocol=3)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("x.npy", npy)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_structured_object_dtype(self):
        path = os.path.join(self.get_temp_dir(), "structured.npz")

        def state():
            return (1, (1,), np.dtype([("a", "i8"), ("b", "O")]), False, [None])

        _crafted_npz(path, (np.ndarray, (0,), "O"), state)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_numeric_state_with_none_raw_data(self):
        path = os.path.join(self.get_temp_dir(), "noneraw.npz")

        def state():
            return (1, (1,), np.dtype("<i8"), False, None)

        _crafted_npz(path, (np.ndarray, (0,), "<i8"), state)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)
