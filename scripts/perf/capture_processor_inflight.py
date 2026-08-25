#!/usr/bin/env python3
"""Capture process-local processor phases during a drain tail."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .regional_capacity_suite import (
        artifact_dir,
        move_to_aborted,
        release_identity,
        write_status,
    )
else:
    from regional_capacity_suite import (
        artifact_dir,
        move_to_aborted,
        release_identity,
        write_status,
    )


NAMESPACE = "gpu-fault-system"
KUBECONFIG = os.getenv(
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "/tmp/gpu-fault-control-plane.kubeconfig",
)


def run(argv: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{' '.join(argv)}\n{result.stderr.decode()}"
        )
    return result.stdout.decode()


def kubectl(*args: str, check: bool = True) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            KUBECONFIG,
            "-n",
            NAMESPACE,
            *args,
        ],
        check=check,
    )


def worker_pods() -> list[str]:
    document = json.loads(
        kubectl(
            "get",
            "pods",
            "-l",
            "app=gpu-fault-control-worker",
            "-o",
            "json",
        )
    )
    return sorted(
        item["metadata"]["name"]
        for item in document["items"]
        if item.get("status", {}).get("phase") == "Running"
    )


def sample_pod(pod: str, samples: int) -> list[dict]:
    script = r"""
import json
import os
import sys
import urllib.request

token = os.environ["GPU_FAULT_EXECUTION_TOKEN"]
documents = []
for _ in range(int(sys.argv[1])):
    request = urllib.request.Request(
        "http://127.0.0.1:8081/v1/processor/status",
        headers={
            "Connection": "close",
            "X-GPU-Fault-Execution-Token": token,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            documents.append(json.loads(response.read()))
    except Exception as exc:
        documents.append({"error": repr(exc)})
print(json.dumps(documents, separators=(",", ":")))
"""
    output = kubectl(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        script,
        str(samples),
        check=False,
    )
    for line in reversed(output.splitlines()):
        if line.startswith("["):
            return json.loads(line)
    return [{"error": output[-2000:]}]


def in_flight(document: dict) -> list[dict]:
    return [
        *document.get("in_flight_requests", []),
        *document.get("inbound_replay_requests", []),
    ]


def process_documents(documents: list[dict]) -> list[dict]:
    for document in reversed(documents):
        published = document.get("pod_processes", [])
        if published:
            return published
    by_pid = {}
    for document in documents:
        process = document.get("process")
        if process is not None:
            by_pid[int(process["pid"])] = document
    return list(by_pid.values())


def dump_stack(pod: str, pid: int) -> None:
    kubectl(
        "exec",
        pod,
        "--",
        "/bin/sh",
        "-c",
        f"kill -USR2 {pid}",
    )


def collect_logs(pods: set[str], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for pod in sorted(pods):
        output = kubectl(
            "logs",
            pod,
            "--since=30m",
            "--tail=10000",
            check=False,
        )
        (target / f"{pod}.log").write_text(
            output,
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=900)
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--samples-per-pod", type=int, default=12)
    parser.add_argument("--dump-stacks-after", type=float, default=30)
    parser.add_argument("--stop-after-idle-seconds", type=float, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="existing capacity run directory; writes inflight.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if options.output and options.run_dir:
        raise SystemExit("--output and --run-dir are mutually exclusive")
    standalone = options.output is None and options.run_dir is None
    identity: dict[str, str] = {}
    if options.output:
        output = options.output
        run_dir = output.parent
    elif options.run_dir:
        run_dir = options.run_dir
        output = run_dir / "inflight.jsonl"
    else:
        identity = release_identity()
        run_dir = artifact_dir(
            Path("artifacts/perf"),
            "processor-inflight",
            identity["release_id"] or "unknown-release",
        )
        output = run_dir / "inflight.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if standalone:
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "case": "processor-inflight",
                    "suite_id": f"processor-inflight-{stamp}",
                    **identity,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_status(run_dir, status="running")
    dumped: set[tuple[str, int]] = set()
    dumped_pods: set[str] = set()
    saw_in_flight = False
    idle_since: float | None = None
    started = time.monotonic()
    try:
        with output.open("a", encoding="utf-8") as target:
            while time.monotonic() - started < options.duration_seconds:
                observed = datetime.now(timezone.utc).isoformat()
                record = {
                    "observed_at": observed,
                    "pods": {},
                }
                active = 0
                for pod in worker_pods():
                    documents = sample_pod(pod, options.samples_per_pod)
                    processes = process_documents(documents)
                    record["pods"][pod] = processes
                    for process in processes:
                        requests = in_flight(process)
                        active += len(requests)
                        pid = int(process["process"]["pid"])
                        longest = max(
                            (
                                float(
                                    item.get(
                                        "phase_elapsed_seconds",
                                        item.get(
                                            "elapsed_seconds",
                                            0,
                                        ),
                                    )
                                )
                                for item in requests
                            ),
                            default=0.0,
                        )
                        key = (pod, pid)
                        if longest >= options.dump_stacks_after and key not in dumped:
                            dump_stack(pod, pid)
                            dumped.add(key)
                            dumped_pods.add(pod)
                record["active_requests"] = active
                target.write(json.dumps(record) + "\n")
                target.flush()
                print(
                    f"{observed} active={active} stack_dumps={len(dumped)}",
                    flush=True,
                )
                if active:
                    saw_in_flight = True
                    idle_since = None
                elif saw_in_flight:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= options.stop_after_idle_seconds:
                        break
                time.sleep(options.interval_seconds)
        if dumped_pods:
            collect_logs(dumped_pods, run_dir / "executors")
        if standalone:
            write_status(run_dir, status="ok")
    except BaseException as exc:
        if standalone:
            write_status(
                run_dir,
                status="aborted",
                reason=f"{type(exc).__name__}: {exc}",
            )
            moved = move_to_aborted(
                Path("artifacts/perf"),
                run_dir,
            )
            print(moved)
        raise
    print(output)


if __name__ == "__main__":
    main()
