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

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "control-plane" / "regional" / "prepare-clean-redeploy.sh"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
case "$*" in
  *"get pod -l app=gpu-fault-control-worker"*) cat "${FAKE_WORKER_PODS}" ;;
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
    start = text.index(f"\n{name}() {{\n") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _find(
    tmp_path: Path, workers: list[dict], ingress: list[dict]
) -> tuple[str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "workers.json").write_text(
        json.dumps({"items": workers}), encoding="utf-8"
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
        "FAKE_INGRESS_PODS": str(tmp_path / "ingress.json"),
    }
    result = subprocess.run(
        ["bash", "-c", harness], check=False, text=True, capture_output=True, env=env
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result.stdout.strip(), calls


def test_database_pod_skips_terminating_and_unready_pods(tmp_path: Path) -> None:
    chosen, _calls = _find(
        tmp_path,
        workers=[
            _pod("worker-old", ready=True, terminating=True),
            _pod("worker-starting", ready=False, terminating=False),
            _pod("worker-new", ready=True, terminating=False),
        ],
        ingress=[],
    )

    assert chosen == "worker-new", "a terminating or unready Pod must not be chosen"


def test_database_pod_falls_through_to_the_next_preference(tmp_path: Path) -> None:
    chosen, calls = _find(
        tmp_path,
        workers=[_pod("worker-old", ready=True, terminating=True)],
        ingress=[_pod("ingress-a", ready=True, terminating=False)],
    )

    assert chosen == "ingress-a", "only terminating workers left; the ingress is next"
    assert any("app=gpu-fault-api-ha" in call for call in calls), calls


def test_database_pod_is_empty_when_nothing_is_ready(tmp_path: Path) -> None:
    chosen, _calls = _find(
        tmp_path,
        workers=[_pod("worker-old", ready=True, terminating=True)],
        ingress=[_pod("ingress-a", ready=False, terminating=False)],
    )

    assert chosen == "", "no Ready, non-terminating Pod exists"


def test_database_pod_moves_to_the_ingress_once_the_worker_is_gone(
    tmp_path: Path,
) -> None:
    """The abandonment step re-picks the Pod after CONTROL_CONSUMERS_STOPPED
    scaled the worker to zero: with no worker Pod at all, the ingress Pod (still
    running at that point of the sequence) carries the Aurora writes."""

    chosen, calls = _find(
        tmp_path, workers=[], ingress=[_pod("ingress-a", ready=True, terminating=False)]
    )

    assert chosen == "ingress-a", "no worker Pod exists; the ingress is next"
    assert [call for call in calls if "get pod" in call] == [
        (
            "-n gpu-fault-system get pod -l app=gpu-fault-control-worker "
            "--field-selector=status.phase=Running -o json"
        ),
        (
            "-n gpu-fault-system get pod -l app=gpu-fault-api-ha "
            "--field-selector=status.phase=Running -o json"
        ),
    ], calls
