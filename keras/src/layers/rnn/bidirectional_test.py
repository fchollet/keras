import numpy as np

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import testing


class SimpleRNNTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.Bidirectional,
            init_kwargs={"layer": layers.SimpleRNN(4)},
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 8),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Bidirectional,
            init_kwargs={
                "layer": layers.SimpleRNN(4),
                "backward_layer": layers.SimpleRNN(4, go_backwards=True),
                "merge_mode": "sum",
            },
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 4),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
        )

    def test_correctness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        forward_layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(
            layer=forward_layer,
        )
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.39687276, 0.39687276, 0.10004295, 0.10004295],
                    [0.7237238, 0.7237238, 0.53391594, 0.53391594],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode="ave")
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array([[0.24845785, 0.24845785], [0.6288199, 0.6288199]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.Bidirectional(layer=forward_layer, merge_mode=None)
        output1, output2 = layer(sequence)
        self.assertAllClose(
            output1,
            np.array([[0.39687276, 0.39687276], [0.7237238, 0.7237238]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            output2,
            np.array([[0.10004295, 0.10004295], [0.53391594, 0.53391594]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        backward_layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.03),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.01),
            go_backwards=True,
        )
        layer = layers.Bidirectional(
            layer=forward_layer, backward_layer=backward_layer, merge_mode="mul"
        )
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array([[0.08374989, 0.08374989], [0.6740834, 0.6740834]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        forward_layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
        )
        layer = layers.Bidirectional(layer=forward_layer, merge_mode="sum")
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array(
                [
                    [
                        [0.20937867, 0.20937867],
                        [0.34462988, 0.34462988],
                        [0.40290534, 0.40290534],
                    ],
                    [
                        [0.59829646, 0.59829646],
                        [0.6734641, 0.6734641],
                        [0.6479671, 0.6479671],
                    ],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            stateful=True,
        )
        layer = layers.Bidirectional(layer=forward_layer)
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.26234663, 0.26234663, 0.16959146, 0.16959146],
                    [0.6137073, 0.6137073, 0.5381646, 0.5381646],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        layer.reset_state()
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.26234663, 0.26234663, 0.16959146, 0.16959146],
                    [0.6137073, 0.6137073, 0.5381646, 0.5381646],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = [
            np.arange(4).reshape((2, 2)).astype("float32") * 1,
            np.arange(4).reshape((2, 2)).astype("float32") * 2,
            np.arange(4).reshape((2, 2)).astype("float32") * 3,
            np.arange(4).reshape((2, 2)).astype("float32") * 4,
        ]
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(
            layer=forward_layer,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.20794602, 0.4577124, 0.14046375, 0.48191673],
                    [0.6682636, 0.6711909, 0.60943645, 0.60950446],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer = layers.Bidirectional(layer=forward_layer)
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.19393763, 0.19393763, 0.11669192, 0.11669192],
                    [0.30818558, 0.30818558, 0.28380975, 0.28380975],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_return_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        forward_layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_state=True,
        )
        layer = layers.Bidirectional(layer=forward_layer)
        output, h1, c1, h2, c2 = layer(sequence)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.1990008, 0.1990008, 0.12659755, 0.12659755],
                    [0.52335435, 0.52335435, 0.44717982, 0.44717982],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            h1,
            np.array([[0.1990008, 0.1990008], [0.52335435, 0.52335435]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            c1,
            np.array([[0.35567185, 0.35567185], [1.0492687, 1.0492687]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            h2,
            np.array([[0.12659755, 0.12659755], [0.44717982, 0.44717982]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            c2,
            np.array([[0.2501858, 0.2501858], [0.941473, 0.941473]]),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_output_shape(self):
        x = np.array([[[101, 202], [303, 404]]])
        for merge_mode in ["ave", "concat", "mul", "sum", None]:
            sub_layer = layers.LSTM(2, return_state=True)
            layer = layers.Bidirectional(sub_layer, merge_mode=merge_mode)
            output = layer(x)
            output_shape = layer.compute_output_shape(x.shape)
            for out, shape in zip(output, output_shape):
                self.assertEqual(out.shape, shape)

        for merge_mode in ["concat", "ave", "mul", "sum"]:
            sub_layer = layers.LSTM(2, return_state=False)
            layer = layers.Bidirectional(sub_layer, merge_mode=merge_mode)
            output = layer(x)
            output_shape = layer.compute_output_shape(x.shape)
            self.assertEqual(output.shape, output_shape)

        # return_state=False & merge_mode=None
        sub_layer = layers.LSTM(2, return_state=False)
        layer = layers.Bidirectional(sub_layer, merge_mode=None)
        output = layer(x)
        output_shape = layer.compute_output_shape(x.shape)
        for out, shape in zip(output, output_shape):
            self.assertEqual(out.shape, shape)

    def test_output_shape_with_asymmetric_backward_layer(self):
        # An explicit `backward_layer` may have a different number of units
        # from the forward layer.
        x = np.array([[[101, 202], [303, 404]]])
        forward_units, backward_units = 3, 5

        # merge_mode="concat" concatenates along the last axis, so the width
        # is the sum of the two, not twice the forward one.
        layer = layers.Bidirectional(
            layers.LSTM(forward_units, return_sequences=True),
            backward_layer=layers.LSTM(
                backward_units, return_sequences=True, go_backwards=True
            ),
            merge_mode="concat",
        )
        output = layer(x)
        self.assertEqual(
            layer.compute_output_shape(x.shape), tuple(output.shape)
        )
        self.assertEqual(output.shape[-1], forward_units + backward_units)

        # merge_mode=None returns the two sequences separately, so each keeps
        # its own width.
        layer = layers.Bidirectional(
            layers.LSTM(forward_units, return_sequences=True),
            backward_layer=layers.LSTM(
                backward_units, return_sequences=True, go_backwards=True
            ),
            merge_mode=None,
        )
        output = layer(x)
        output_shape = layer.compute_output_shape(x.shape)
        for out, shape in zip(output, output_shape):
            self.assertEqual(tuple(out.shape), tuple(shape))

        # return_state appends the forward states then the backward states,
        # which are sized by their own layers.
        for merge_mode in ["concat", None]:
            layer = layers.Bidirectional(
                layers.LSTM(forward_units, return_state=True),
                backward_layer=layers.LSTM(
                    backward_units, return_state=True, go_backwards=True
                ),
                merge_mode=merge_mode,
            )
            output = layer(x)
            output_shape = layer.compute_output_shape(x.shape)
            self.assertEqual(len(output), len(output_shape))
            for out, shape in zip(output, output_shape):
                self.assertEqual(tuple(out.shape), tuple(shape))

    def test_asymmetric_backward_layer_rejected_for_elementwise_merge(self):
        # "sum", "mul" and "ave" combine the two directions elementwise, so
        # differing widths cannot work.
        x = np.array([[[101, 202], [303, 404]]])
        for merge_mode in ["sum", "mul", "ave"]:
            layer = layers.Bidirectional(
                layers.LSTM(3, return_sequences=True),
                backward_layer=layers.LSTM(
                    5, return_sequences=True, go_backwards=True
                ),
                merge_mode=merge_mode,
            )
            with self.assertRaisesRegex(
                ValueError, "must produce the same number of units"
            ):
                layer.compute_output_shape(x.shape)
            with self.assertRaisesRegex(
                ValueError, "must produce the same number of units"
            ):
                layer(x)

        # Matching widths remain valid for the same merge modes.
        for merge_mode in ["sum", "mul", "ave"]:
            layer = layers.Bidirectional(
                layers.LSTM(3, return_sequences=True),
                backward_layer=layers.LSTM(
                    3, return_sequences=True, go_backwards=True
                ),
                merge_mode=merge_mode,
            )
            output = layer(x)
            self.assertEqual(
                layer.compute_output_shape(x.shape), tuple(output.shape)
            )

    def test_keeps_use_cudnn(self):
        # keep use_cudnn if the layer has it
        for rnn_class in [layers.GRU, layers.LSTM]:
            for use_cudnn in [True, False, "auto"]:
                rnn = rnn_class(1, use_cudnn=use_cudnn)
                bidi = layers.Bidirectional(rnn)
                self.assertEqual(bidi.forward_layer.use_cudnn, use_cudnn)
                self.assertEqual(bidi.backward_layer.use_cudnn, use_cudnn)

        # otherwise ignore it
        rnn = layers.SimpleRNN(1)
        bidi = layers.Bidirectional(rnn)
        self.assertFalse(hasattr(bidi.forward_layer, "use_cudnn"))
        self.assertFalse(hasattr(bidi.backward_layer, "use_cudnn"))

    def test_fused_lstm_eligibility(self):
        # LSTM with use_cudnn != False passes the layer-level precondition.
        bidi = layers.Bidirectional(layers.LSTM(4, use_cudnn="auto"))
        bidi.build((None, 5, 3))
        self.assertTrue(bidi._can_attempt_fused_lstm(mask=None))

        # GRU and SimpleRNN are not eligible.
        gru = layers.Bidirectional(layers.GRU(4, use_cudnn="auto"))
        gru.build((None, 5, 3))
        self.assertFalse(gru._can_attempt_fused_lstm(mask=None))

        simple = layers.Bidirectional(layers.SimpleRNN(4))
        simple.build((None, 5, 3))
        self.assertFalse(simple._can_attempt_fused_lstm(mask=None))

        # use_cudnn=False disables the fast path.
        off = layers.Bidirectional(layers.LSTM(4, use_cudnn=False))
        off.build((None, 5, 3))
        self.assertFalse(off._can_attempt_fused_lstm(mask=None))

        # Dropout disables the fast path.
        dp = layers.Bidirectional(layers.LSTM(4, use_cudnn="auto", dropout=0.1))
        dp.build((None, 5, 3))
        self.assertFalse(dp._can_attempt_fused_lstm(mask=None))

        # A mask disables the fast path.
        eligible = layers.Bidirectional(layers.LSTM(4, use_cudnn="auto"))
        eligible.build((None, 5, 3))
        mask = np.ones((2, 5), dtype="bool")
        self.assertFalse(eligible._can_attempt_fused_lstm(mask=mask))

        # Wrong initial_state arity (should be 4: fwd_h, fwd_c, bwd_h, bwd_c).
        wrong_state = [np.zeros((2, 4), dtype="float32")] * 2
        self.assertFalse(
            eligible._can_attempt_fused_lstm(
                mask=None, initial_state=wrong_state
            )
        )

    def test_fused_lstm_matches_unfused(self):
        # The fused path requires backend support (JAX with cuDNN today).
        # On other backends and on CPU runners, `backend.rnn.bidirectional_lstm`
        # raises `NotImplementedError` and the layer falls back to the
        # two-call path, so this test trivially passes; on GPU it
        # exercises the fused dispatch and asserts numerical equivalence
        # with the two-call reference.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 6, 4)).astype("float32")

        def _build(use_cudnn):
            layer = layers.Bidirectional(
                layers.LSTM(
                    5,
                    use_cudnn=use_cudnn,
                    return_sequences=True,
                    kernel_initializer=initializers.GlorotUniform(seed=1),
                    recurrent_initializer=initializers.Orthogonal(seed=2),
                )
            )
            layer.build(x.shape)
            return layer

        ref = _build(use_cudnn=False)
        fused = _build(use_cudnn="auto")
        for rv, fv in zip(ref.weights, fused.weights):
            fv.assign(rv.value)

        self.assertAllClose(ref(x), fused(x), atol=1e-5)

    def test_fused_gru_eligibility(self):
        # Shared gating (use_cudnn, dropout, mask, initial_state arity)
        # is already exercised by test_fused_lstm_eligibility. Cover only
        # the GRU-specific gates here.

        # GRU with use_cudnn != False passes.
        bidi = layers.Bidirectional(layers.GRU(4, use_cudnn="auto"))
        bidi.build((None, 5, 3))
        self.assertTrue(bidi._can_attempt_fused_gru(mask=None))

        # LSTM is not eligible.
        lstm = layers.Bidirectional(layers.LSTM(4, use_cudnn="auto"))
        lstm.build((None, 5, 3))
        self.assertFalse(lstm._can_attempt_fused_gru(mask=None))

        # reset_after=False is incompatible with the cuDNN GRU formulation.
        legacy = layers.Bidirectional(
            layers.GRU(4, use_cudnn="auto", reset_after=False)
        )
        legacy.build((None, 5, 3))
        self.assertFalse(legacy._can_attempt_fused_gru(mask=None))

        # GRU has one state per direction (length 2), not LSTM's four.
        wrong_state = [np.zeros((2, 4), dtype="float32")] * 4
        self.assertFalse(
            bidi._can_attempt_fused_gru(mask=None, initial_state=wrong_state)
        )

    def test_fused_gru_matches_unfused(self):
        # The fused path requires backend support (torch with cuDNN today).
        # On other backends and on CPU runners, `backend.rnn.bidirectional_gru`
        # raises `NotImplementedError` and the layer falls back to the
        # two-call path, so this test trivially passes. On GPU it exercises
        # the fused dispatch and asserts numerical equivalence with the
        # two-call reference.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 6, 4)).astype("float32")

        def _build(use_cudnn):
            layer = layers.Bidirectional(
                layers.GRU(
                    5,
                    use_cudnn=use_cudnn,
                    return_sequences=True,
                    kernel_initializer=initializers.GlorotUniform(seed=1),
                    recurrent_initializer=initializers.Orthogonal(seed=2),
                )
            )
            layer.build(x.shape)
            return layer

        ref = _build(use_cudnn=False)
        fused = _build(use_cudnn="auto")
        for rv, fv in zip(ref.weights, fused.weights):
            fv.assign(rv.value)

        self.assertAllClose(ref(x), fused(x), atol=1e-5)

    def test_torch_cudnn_bidirectional_gru_dispatch_fires(self):
        # `backend.rnn.bidirectional_gru` is wrapped by a try/except in the
        # Bidirectional layer, so a regression in the fused cuDNN call
        # silently routes every Bidirectional(GRU) call through the
        # two-pass fallback while every existing test still passes.
        # Assert that `torch._VF.gru` actually fires when the layer runs
        # against cuDNN-eligible inputs on CUDA.
        if backend.backend() != "torch":
            self.skipTest("Guards the torch-backend cuDNN dispatch path.")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("Requires a CUDA device.")

        from unittest import mock

        x = torch.randn(4, 6, 5, device="cuda")
        layer = layers.Bidirectional(
            layers.GRU(8, use_cudnn="auto", return_sequences=True)
        )
        layer(x)  # build on cuda

        real_vf_gru = torch._VF.gru
        calls = []

        def spy(*args, **kwargs):
            calls.append(True)
            return real_vf_gru(*args, **kwargs)

        with mock.patch.object(torch._VF, "gru", side_effect=spy):
            _ = layer(x)

        self.assertGreaterEqual(
            len(calls),
            1,
            msg=(
                "torch._VF.gru was never invoked from Bidirectional(GRU); "
                "the fused cuDNN dispatch is silently inactive and every "
                "call is routing through the two-pass fallback."
            ),
        )
