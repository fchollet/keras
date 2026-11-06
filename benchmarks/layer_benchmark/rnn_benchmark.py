"""Benchmark rnn layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.rnn_benchmark \
    --benchmark_name=benchmark_lstm \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

import keras
from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_conv_lstm1d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "ConvLSTM1D"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 256, 3],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_conv_lstm2d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "ConvLSTM2D"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 32, 3],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_conv_lstm3d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "ConvLSTM3D"
    init_args = {
        "filters": 8,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[8, 16, 16, 16, 3],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_gru(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GRU"
    init_args = {
        "units": 32,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_lstm(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "LSTM"
    init_args = {
        "units": 32,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_simple_rnn(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "SimpleRNN"
    init_args = {
        "units": 32,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_bidirectional(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Bidirectional"
    init_args = {}
    keras_layer = keras.layers.Bidirectional(keras.layers.LSTM(32))
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
        keras_layer=keras_layer,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_time_distributed(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "TimeDistributed"
    init_args = {}
    keras_layer = keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3)))
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[10, 32, 32, 3],
        jit_compile=jit_compile,
        keras_layer=keras_layer,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


BENCHMARK_NAMES = {
    "benchmark_conv_lstm1d": benchmark_conv_lstm1d,
    "benchmark_conv_lstm2d": benchmark_conv_lstm2d,
    "benchmark_conv_lstm3d": benchmark_conv_lstm3d,
    "benchmark_gru": benchmark_gru,
    "benchmark_lstm": benchmark_lstm,
    "benchmark_simple_rnn": benchmark_simple_rnn,
    "benchmark_bidirectional": benchmark_bidirectional,
    "benchmark_time_distributed": benchmark_time_distributed,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name is None:
        for name, benchmark_fn in BENCHMARK_NAMES.items():
            benchmark_fn(num_samples, batch_size, jit_compile)
        return

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
            f"be one of {BENCHMARK_NAMES.keys()}"
        )
    benchmark_fn = BENCHMARK_NAMES[benchmark_name]
    benchmark_fn(num_samples, batch_size, jit_compile)


if __name__ == "__main__":
    app.run(main)
