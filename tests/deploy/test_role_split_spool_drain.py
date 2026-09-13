"""The role-split apply's spool-drain gate reads the spool-worker's own port.

The first live enabled->disabled transition (2026-09-13) exec'd
``http://127.0.0.1:8080/metrics`` in a spool-worker Pod that serves on 8082,
and ``set -e`` turned the refused connection into a failed release. The gate
now reads the ``http`` container port from the Pod and treats a Pod that is
Running but not answering yet as a poll to repeat, not a failed drain.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "deploy" / "control-plane" / "tools" / "apply-control-plane-role-split.sh"
)

FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Records every invocation; answers the calls wait_for_spool_drain makes.
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
case "$*" in
  *"get pod -l app=gpu-fault-telemetry-spool-worker"*)
    printf '%s' "spool-0"
    ;;
  *"get pod spool-0 -o jsonpath"*)
    printf '%s' "${FAKE_SPOOL_PORT}"
    ;;
  *"exec spool-0 --"*)
    count_file="${FAKE_KUBECTL_LOG}.execs"
    count=$(( $(cat "${count_file}" 2>/dev/null || echo 0) + 1 ))
    printf '%s' "${count}" >"${count_file}"
    if (( count <= FAKE_REFUSE_FIRST )); then
      echo "urllib.error.URLError: <urlopen error [Errno 111] Connection refused>" >&2
      exit 1
    fi
    printf 'gpu_fault_telemetry_spool_depth 0\ngpu_fault_telemetry_spool_leased 0\n'
    ;;
  *)
    echo "unexpected kubectl call: $*" >&2
    exit 9
    ;;
esac
"""


def _extract_function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index(f"\n{name}() {{\n") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _run_drain(
    tmp_path: Path, *, port: str, refuse_first: int
) -> tuple[int, list[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "kubectl.log"
    harness = "\n".join(
        [
            "set -uo pipefail",
            "kubectl_args=()",
            "NAMESPACE=gpu-fault-system",
            "SPOOL_WORKER_METRICS_PORT=8082",
            "sleep() { :; }",
            _extract_function("wait_for_spool_drain"),
            "wait_for_spool_drain",
        ]
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
        "FAKE_SPOOL_PORT": port,
        "FAKE_REFUSE_FIRST": str(refuse_first),
        "GPU_FAULT_TELEMETRY_SPOOL_DRAIN_TIMEOUT_SECONDS": "30",
    }
    result = subprocess.run(
        ["bash", "-c", harness], check=False, text=True, capture_output=True, env=env
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result.returncode, calls, result.stderr


def test_spool_drain_reads_the_spool_worker_http_port_from_the_pod(
    tmp_path: Path,
) -> None:
    code, calls, stderr = _run_drain(tmp_path, port="8082", refuse_first=0)

    assert code == 0, stderr
    execs = [call for call in calls if "exec spool-0" in call]
    assert len(execs) == 1, calls
    assert "http://127.0.0.1:8082/metrics" in execs[0], execs[0]
    assert "8080" not in execs[0], (
        "the api-ha port must not be assumed for the spool-worker"
    )


def test_spool_drain_falls_back_to_the_manifest_port_without_a_named_port(
    tmp_path: Path,
) -> None:
    code, calls, stderr = _run_drain(tmp_path, port="", refuse_first=0)

    assert code == 0, stderr
    execs = [call for call in calls if "exec spool-0" in call]
    assert execs and "http://127.0.0.1:8082/metrics" in execs[0], calls


def test_spool_drain_polls_again_when_the_pod_is_not_answering_yet(
    tmp_path: Path,
) -> None:
    """A refused connection inside the drain window is a retry, not a failure."""

    code, calls, stderr = _run_drain(tmp_path, port="8082", refuse_first=2)

    assert code == 0, stderr
    execs = [call for call in calls if "exec spool-0" in call]
    assert len(execs) == 3, calls
    assert stderr.count("not answering on port 8082 yet") == 2, stderr
