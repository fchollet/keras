"""Run a layer benchmark once per backend and compare the throughputs."""

import argparse
import json
import os
import subprocess
import sys

RESULT_PREFIX = "BENCHMARK_RESULT:"


def run_backend(backend, module, flags):
    env = dict(os.environ, KERAS_BACKEND=backend, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-m", module, *flags, "--print_results_json"]
    results = []
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, text=True
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            if RESULT_PREFIX in line:
                results.append(json.loads(line.split(RESULT_PREFIX, 1)[1]))
    return proc.returncode == 0, results


def print_table(backends, results):
    rows = {}
    for backend in backends:
        for r in results[backend]:
            key = f"{r['layer']} ({r['mode']})"
            rows.setdefault(key, {})[backend] = r["throughput"]
    if not rows:
        print("No benchmark results collected.")
        return
    print("\nThroughput (samples/sec)\n")
    print(f"{'Benchmark':<30}" + "".join(f"{b:>15}" for b in backends))
    for key, values in rows.items():
        cells = "".join(
            f"{values[b]:>15.2f}" if b in values else f"{'n/a':>15}"
            for b in backends
        )
        print(f"{key:<30}{cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="jax,torch,tensorflow")
    parser.add_argument("module")
    args, flags = parser.parse_known_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    results, failed = {}, []
    for backend in backends:
        print(f"=== {backend} ===")
        ok, results[backend] = run_backend(backend, args.module, flags)
        if not ok:
            failed.append(backend)

    print_table(backends, results)
    if failed:
        sys.exit(f"Failed backends: {', '.join(failed)}")


if __name__ == "__main__":
    main()
