# Benchmark the layer performance

This directory contains benchmarks of `keras.layers.XXX` performance, for both
the forward pass and the train step (forward & backward pass).

To run the benchmark, use the command below and change the flags according to
your target:

```shell
python3 -m benchmarks.layer_benchmark.conv_benchmark \
    --benchmark_name=benchmark_conv2D \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```

To compare backends, `run_multi_backend.py` runs a benchmark module once per
backend, each in its own process, and prints a throughput table:

```shell
python3 -m benchmarks.layer_benchmark.run_multi_backend \
    --backends=jax,torch \
    benchmarks.layer_benchmark.conv_benchmark \
    --benchmark_name=benchmark_conv2D
```
