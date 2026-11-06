import pytest

from keras.src import backend
from keras.src import testing
from keras.src.utils import sequence_utils


class PadSequencesTest(testing.TestCase):
    def test_pad_sequences(self):
        a = [[1], [1, 2], [1, 2, 3]]

        # test padding
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="pre")
        self.assertAllClose(b, [[0, 0, 1], [0, 1, 2], [1, 2, 3]])
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="post")
        self.assertAllClose(b, [[1, 0, 0], [1, 2, 0], [1, 2, 3]])

        # test truncating
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="pre")
        self.assertAllClose(b, [[0, 1], [1, 2], [2, 3]])
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="post")
        self.assertAllClose(b, [[0, 1], [1, 2], [1, 2]])

        # test value
        b = sequence_utils.pad_sequences(a, maxlen=3, value=1)
        self.assertAllClose(b, [[1, 1, 1], [1, 1, 2], [1, 2, 3]])

    def test_pad_sequences_float(self):
        a = [[1.2], [1.2, 2.3], [1.2, 2.3, 3.4]]

        # test padding
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="pre", dtype="float32"
        )
        self.assertAllClose(b, [[0, 0, 1.2], [0, 1.2, 2.3], [1.2, 2.3, 3.4]])
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="post", dtype="float32"
        )
        self.assertAllClose(b, [[1.2, 0, 0], [1.2, 2.3, 0], [1.2, 2.3, 3.4]])

        # test truncating
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="pre", dtype="float32"
        )
        self.assertAllClose(b, [[0, 1.2], [1.2, 2.3], [2.3, 3.4]])
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="post", dtype="float32"
        )
        self.assertAllClose(b, [[0, 1.2], [1.2, 2.3], [1.2, 2.3]])

        # test value
        b = sequence_utils.pad_sequences(a, maxlen=3, value=1, dtype="float32")
        self.assertAllClose(b, [[1, 1, 1.2], [1, 1.2, 2.3], [1.2, 2.3, 3.4]])

    def test_pad_sequences_str(self):
        a = [["1"], ["1", "2"], ["1", "2", "3"]]

        # test padding
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="pre", value="pad", dtype=object
        )
        self.assertAllEqual(
            b, [["pad", "pad", "1"], ["pad", "1", "2"], ["1", "2", "3"]]
        )
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="post", value="pad", dtype="<U3"
        )
        self.assertAllEqual(
            b, [["1", "pad", "pad"], ["1", "2", "pad"], ["1", "2", "3"]]
        )

        # test truncating
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="pre", value="pad", dtype=object
        )
        self.assertAllEqual(b, [["pad", "1"], ["1", "2"], ["2", "3"]])
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="post", value="pad", dtype="<U3"
        )
        self.assertAllEqual(b, [["pad", "1"], ["1", "2"], ["1", "2"]])

        with self.assertRaisesRegex(
            ValueError, "`dtype` int32 is not compatible with "
        ):
            sequence_utils.pad_sequences(
                a, maxlen=2, truncating="post", value="pad"
            )

    def test_pad_sequences_vector(self):
        a = [[[1, 1]], [[2, 1], [2, 2]], [[3, 1], [3, 2], [3, 3]]]

        # test padding
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="pre")
        self.assertAllClose(
            b,
            [
                [[0, 0], [0, 0], [1, 1]],
                [[0, 0], [2, 1], [2, 2]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="post")
        self.assertAllClose(
            b,
            [
                [[1, 1], [0, 0], [0, 0]],
                [[2, 1], [2, 2], [0, 0]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )

        # test truncating
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="pre")
        self.assertAllClose(
            b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 2], [3, 3]]]
        )

        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="post")
        self.assertAllClose(
            b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 1], [3, 2]]]
        )

        # test value
        b = sequence_utils.pad_sequences(a, maxlen=3, value=1)
        self.assertAllClose(
            b,
            [
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [2, 1], [2, 2]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires the TensorFlow backend.",
    )
    def test_pad_sequences_dynamic_list_in_tf_function(self):
        import tensorflow as tf

        @tf.function
        def symbolic_fn(k, padding, maxlen):
            base = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
            seq1 = base[:k]
            seq2 = base[2 : 2 + k]
            seq3 = tf.constant([[9.0, 9.0]])
            return sequence_utils.pad_sequences(
                [seq1, seq2, seq3],
                maxlen=maxlen,
                padding=padding,
                truncating="post",
                value=-1.0,
                dtype="float32",
            )

        self.assertAllClose(
            symbolic_fn(tf.constant(2), "post", None),
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[9.0, 9.0], [-1.0, -1.0]],
            ],
        )
        self.assertAllClose(
            symbolic_fn(tf.constant(3), "pre", 3),
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[-1.0, -1.0], [3.0, 3.0], [4.0, 4.0]],
                [[-1.0, -1.0], [-1.0, -1.0], [9.0, 9.0]],
            ],
        )
        self.assertAllClose(
            symbolic_fn(tf.constant(4), "post", 2),
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[9.0, 9.0], [-1.0, -1.0]],
            ],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires the TensorFlow backend.",
    )
    def test_pad_sequences_dynamic_list_with_jit_compile(self):
        import tensorflow as tf

        @tf.function(jit_compile=True)
        def symbolic_fn(k):
            seq1 = tf.ones([2, 2])[:k]
            seq2 = tf.ones([2, 2])
            return sequence_utils.pad_sequences(
                [seq1, seq2], padding="post", value=-1.0, dtype="float32"
            )

        self.assertAllClose(
            symbolic_fn(tf.constant(1)),
            [
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires the TensorFlow backend.",
    )
    def test_pad_sequences_dynamic_tensor_input(self):
        import tensorflow as tf

        @tf.function
        def symbolic_fn(k):
            base = tf.constant([[1.0, 1.0], [2.0, 2.0]])
            seq1 = base[:k]
            seq2 = base[:k]
            return sequence_utils.pad_sequences(
                tf.stack([seq1, seq2]),
                maxlen=3,
                padding="post",
                value=-1.0,
                dtype="float32",
            )

        self.assertAllClose(
            symbolic_fn(tf.constant(2)),
            [
                [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]],
                [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]],
            ],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires the TensorFlow backend.",
    )
    def test_pad_sequences_static_tensor_list(self):
        import tensorflow as tf

        out = sequence_utils.pad_sequences(
            [tf.constant([1]), tf.constant([2, 3])], maxlen=2
        )
        self.assertAllClose(out, [[0, 1], [2, 3]])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Requires the TensorFlow backend.",
    )
    def test_pad_sequences_empty_tensor_input(self):
        import tensorflow as tf

        out = sequence_utils.pad_sequences(
            [tf.constant([1.0, 2.0]), tf.constant([], shape=(0,))],
            maxlen=2,
            padding="post",
            value=-1.0,
            dtype="float32",
        )
        self.assertAllClose(out, [[1.0, 2.0], [-1.0, -1.0]])
