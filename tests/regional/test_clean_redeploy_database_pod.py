"""The cleanup picks a Ready, non-terminating database client Pod.

Live 2026-09-13: join-cluster stamped the worker's failure-domain map last and
returned while the worker was still rolling; the uninstall that followed chose a
Running worker Pod that was already being deleted and its first exec failed
with "cannot exec into a container in a completed pod".
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "control-plane" / "regional" / "prepare-clean-redeploy.sh"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
case "$*" in
  *"get pod -l app=gpu-fault-control-worker"*)
    cat "${FAKE_WORKER_PODS}"; exit "${FAKE_WORKER_EXIT:-0}" ;;
  *"get pod -l app=gpu-fault-api-ha"*) cat "${FAKE_INGRESS_PODS}" ;;
  *) echo "unexpected kubectl call: $*" >&2; exit 9 ;;
esac
"""


def _pod(name: str, *, ready: bool, terminating: bool) -> dict:
    metadata: dict = {"name": name}
    if terminating:
        metadata["deletionTimestamp"] = "2026-09-13T13:51:50Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "control-worker", "ready": ready}],
        },
    }


def _extract_function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index(f"\n{name}() {{") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _find(
    tmp_path: Path,
    workers: list[dict],
    ingress: list[dict],
    *,
    worker_exit: int = 0,
    worker_payload: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "workers.json").write_text(
        json.dumps({"items": workers}) if worker_payload is None else worker_payload,
        encoding="utf-8",
    )
    (tmp_path / "ingress.json").write_text(
        json.dumps({"items": ingress}), encoding="utf-8"
    )
    log = tmp_path / "kubectl.log"
    harness = "\n".join(
        [
            "set -uo pipefail",
            "NAMESPACE=gpu-fault-system",
            'CPU_DATABASE_POD_PREFERENCE=("gpu-fault-control-worker" "gpu-fault-api-ha")',
            'cpu_kubectl() { kubectl "$@"; }',
            _extract_function("find_database_pod"),
            "find_database_pod",
        ]
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
        "FAKE_WORKER_PODS": str(tmp_path / "workers.json"),
        "FAKE_WORKER_EXIT": str(worker_exit),
        "FAKE_INGRESS_PODS": str(tmp_path / "ingress.json"),
    }
    result = subprocess.run(
        ["bash", "-c", harness], check=False, text=True, capture_output=True, env=env
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_database_pod_skips_terminating_and_unready_pods(tmp_path: Path) -> None:
    result, _calls = _find(
        tmp_path,
        workers=[
            _pod("worker-old", ready=True, terminating=True),
            _pod("worker-starting", ready=False, terminating=False),
            _pod("worker-new", ready=True, terminating=False),
        ],
        ingress=[],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "worker-new", (
        "a terminating or unready Pod must not be chosen"
    )


def test_database_pod_falls_through_to_the_next_preference(tmp_path: Path) -> None:
    result, calls = _find(
        tmp_path,
        workers=[_pod("worker-old", ready=True, terminating=True)],
        ingress=[_pod("ingress-a", ready=True, terminating=False)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ingress-a", (
        "only terminating workers left; the ingress is next"
    )
    assert any("app=gpu-fault-api-ha" in call for call in calls), calls


def test_database_pod_is_empty_when_nothing_is_ready(tmp_path: Path) -> None:
    result, _calls = _find(
        tmp_path,
        workers=[_pod("worker-old", ready=True, terminating=True)],
        ingress=[_pod("ingress-a", ready=False, terminating=False)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "no Ready, non-terminating Pod exists"


def test_database_pod_can_use_ready_ingress_when_no_worker_exists(
    tmp_path: Path,
) -> None:
    result, calls = _find(
        tmp_path, workers=[], ingress=[_pod("ingress-a", ready=True, terminating=False)]
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ingress-a"
    assert [call for call in calls if "get pod" in call] == [
        "-n gpu-fault-system get pod -l app=gpu-fault-control-worker "
        "--field-selector=status.phase=Running -o json",
        "-n gpu-fault-system get pod -l app=gpu-fault-api-ha "
        "--field-selector=status.phase=Running -o json",
    ], "database Pod selection skipped a preference or hid a read failure"


@pytest.mark.parametrize(
    "statuses",
    [None, [], [{"ready": False}], [{"ready": True}, {"ready": False}], [{"ready": 1}]],
)
def test_database_pod_requires_nonempty_explicit_ready_container_statuses(
    tmp_path: Path, statuses: object
) -> None:
    candidate = _pod("worker-incomplete", ready=True, terminating=False)
    candidate["status"]["containerStatuses"] = statuses

    result, _calls = _find(tmp_path, workers=[candidate], ingress=[])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "unknown container readiness selected a Pod"


@pytest.mark.parametrize("phase", ["Pending", "Succeeded", "Failed", "Unknown"])
def test_database_pod_checks_running_phase_even_after_server_filtering(
    tmp_path: Path, phase: str
) -> None:
    candidate = _pod("worker-not-running", ready=True, terminating=False)
    candidate["status"]["phase"] = phase

    result, _calls = _find(tmp_path, workers=[candidate], ingress=[])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "a non-running Pod was selected for exec"


@pytest.mark.parametrize(
    ("worker_exit", "worker_payload"),
    [
        (1, None),
        (0, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, '{"items":{}}'),
        (0, '{"items":[{"metadata":{},"status":{}}]}'),
    ],
)
def test_database_pod_read_failures_do_not_fall_back_to_healthy_ingress(
    tmp_path: Path, worker_exit: int, worker_payload: str | None
) -> None:
    result, calls = _find(
        tmp_path,
        workers=[_pod("worker-ready", ready=True, terminating=False)],
        ingress=[_pod("ingress-ready", ready=True, terminating=False)],
        worker_exit=worker_exit,
        worker_payload=worker_payload,
    )

    assert result.returncode != 0, "a failed or malformed Pod read was swallowed"
    assert result.stdout.strip() == "", "a failed Pod read produced a selection"
    assert len(calls) == 1, "a failed worker read was mistaken for an empty list"


def test_cpu_rollout_wait_preserves_the_captured_name_and_namespace(
    tmp_path: Path,
) -> None:
    log = tmp_path / "rollout.jsonl"
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "with Path(sys.argv[1]).open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[2:]) + '\\n')\n",
        encoding="utf-8",
    )
    harness = "\n".join(
        [
            "set -euo pipefail",
            "NAMESPACE=gpu-fault-system",
            "TIMEOUT_SECONDS=17",
            "CPU_INGRESS_DEPLOYMENTS=($'gpu-fault-api-ha\\tingress-ns\\tcpu')",
            "CPU_CONSUMER_DEPLOYMENTS=($'gpu-fault-control-worker\\tworker-ns\\tcpu')",
            'deployment_replicas_cpu() { parse_target "$1"; printf "1"; }',
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            f'cpu_kubectl() {{ python3 "{recorder}" "{log}" "$@"; }}',
            _extract_function("parse_target"),
            _extract_function("wait_for_cpu_rollouts"),
            "wait_for_cpu_rollouts",
        ]
    )

    result = subprocess.run(
        ["bash", "-c", harness], check=False, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        [
            "-n",
            "ingress-ns",
            "rollout",
            "status",
            "deployment/gpu-fault-api-ha",
            "--timeout=17s",
        ],
        [
            "-n",
            "worker-ns",
            "rollout",
            "status",
            "deployment/gpu-fault-control-worker",
            "--timeout=17s",
        ],
    ], "rollout lost the captured deployment identity or namespace"
