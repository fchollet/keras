"""
Refresh the per-test durations that balance the GPU and TPU test shards.

`gpu_tests.yml` and `tpu_tests.yml` split each backend's tests into four shards
using `.github/test_durations/<device>/<backend>.json`. Every master run
uploads the fresh timings as artifacts. This script downloads those artifacts,
unions the four shard files per backend and rewrites the committed files.

Requires the GitHub CLI (`gh auth login`).

Usage:
python .github/workflows/scripts/merge_test_durations.py
python .github/workflows/scripts/merge_test_durations.py \
    --gpu-run-id 123 --tpu-run-id 456
"""

import argparse
import json
import pathlib
import subprocess
import tempfile

SHARDS = 4
DEVICES = {
    "gpu": ("jax", "tensorflow", "torch"),
    "tpu": ("jax",),
}
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "test_durations"


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def latest_master_run(repo, workflow):
    runs = json.loads(
        gh(
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--branch",
            "master",
            "--event",
            "push",
            "--status",
            "completed",
            "--limit",
            "1",
            "--json",
            "databaseId,createdAt,conclusion,headSha",
        )
    )
    if not runs:
        raise SystemExit(f"No completed {workflow} run on master.")
    run = runs[0]
    print(
        f"Using {workflow} run {run['databaseId']} ({run['createdAt']}, "
        f"{run['conclusion']}, {run['headSha'][:10]})"
    )
    return str(run["databaseId"])


def process_device(repo, device, backends, run_id):
    merged = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gh(
                "run",
                "download",
                run_id,
                "--repo",
                repo,
                "--pattern",
                f"durations-{device}-*",
                "--dir",
                tmp,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"{e.stderr.strip()}\nRun {run_id} has no durations-{device}-* "
                "artifacts. Pick a newer run with run_id args."
            )
        tmp = pathlib.Path(tmp)

        for backend in backends:
            files = sorted(
                tmp.glob(f"durations-{device}-{backend}-*/{backend}.json")
            )
            if len(files) != SHARDS:
                raise SystemExit(
                    f"{device}/{backend}: expected {SHARDS} shard files, "
                    f"found {len(files)}."
                )
            merged[backend] = {}
            for path in files:
                merged[backend].update(json.loads(path.read_text()))

    device_out = OUT_DIR / device
    device_out.mkdir(parents=True, exist_ok=True)

    for backend, durations in merged.items():
        target = device_out / f"{backend}.json"
        target.write_text(
            json.dumps(durations, indent=4, sort_keys=True) + "\n"
        )
        total = sum(durations.values()) / 60
        print(
            f"{device}/{backend}: {len(durations)} tests, "
            f"{total:.1f} min -> {target}"
        )
        for nodeid, seconds in sorted(durations.items(), key=lambda kv: -kv[1])[
            :5
        ]:
            print(f"  {seconds:7.1f}s  {nodeid}")


def main():
    parser = argparse.ArgumentParser(description="Merge test durations.")
    parser.add_argument(
        "--repo", default="keras-team/keras", help="GitHub repo"
    )
    parser.add_argument("--gpu-run-id", help="GPU workflow run ID")
    parser.add_argument("--tpu-run-id", help="TPU workflow run ID")
    args = parser.parse_args()

    gpu_run_id = args.gpu_run_id or latest_master_run(
        args.repo, "gpu_tests.yml"
    )
    process_device(args.repo, "gpu", DEVICES["gpu"], gpu_run_id)

    tpu_run_id = args.tpu_run_id or latest_master_run(
        args.repo, "tpu_tests.yml"
    )
    process_device(args.repo, "tpu", DEVICES["tpu"], tpu_run_id)


if __name__ == "__main__":
    main()
