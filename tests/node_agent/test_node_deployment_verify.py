"""``verify-gpu-fault-collector.sh`` run for real against PATH stubs.

GPU health is not software integrity: the verifier warns on persistence
mode, DCGM metrics and a partial GPU enumeration but keeps every service
and delivery check fatal. The contract is read from the script's exit code
and SUMMARY line, plus the static checks that it never sources the secret
environment file.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from tests.node_agent._deployment_support import NODE_SCRIPTS, _write_stub


def test_verifier_does_not_source_secret_environment_file() -> None:
    verifier = NODE_SCRIPTS[1].read_text()

    assert 'source "${ENV_FILE}"' not in verifier
    assert "shlex.split" in verifier
    assert "VERIFY_STARTED_EPOCH" in verifier
    assert '"SSL_CERT_FILE": ""' in verifier
    assert 'CURL_TLS=(--cacert "${SSL_CERT_FILE}")' in verifier
    assert "verify-certificate-bundle" in verifier
    assert "check-control-plane-certificate" in verifier
    assert "for _ in {1..60}; do" in verifier


# GPU health is not software integrity: a node with one GPU off the bus is the
# node the Agent exists to remediate, so `verify` must report it and still let
# the install stand. These two lists are the contract owner decision 6 fixes;
# a name may appear in both when the severity depends on the observation
# (zero GPUs enumerated is fatal, a partial enumeration is a warning).
VERIFY_WARN_NAMES = (
    "GPU persistence mode",
    "DCGM exporter supported metrics",
    "NVIDIA GPU enumeration",
)

VERIFY_FATAL_NAMES = (
    "NVIDIA GPU enumeration",
    "control plane health",
    "metrics collector service",
    "host telemetry collector service",
    "Fabric Manager SXID collector service",
    "GPU persistence service",
    "signed node action agent service",
    "signed node action agent health",
    "metrics delivered to control plane",
)


def _run_verify(
    workspace: Path,
    *,
    persistence_modes: tuple[str, ...] = ("Enabled",),
    persistence_exit: int = 0,
    enumerated_gpus: int = 8,
    enumeration_exit: int = 0,
    dcgm_metrics: bool = True,
    inactive_units: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the real verifier against PATH stubs for nvidia-smi/systemctl/curl.

    Substring assertions cannot tell a WARN that is counted and non-fatal from
    a WARN that still leaves ``failed=1`` behind, so the contract is checked by
    executing the script and reading its exit code.
    """
    binaries = workspace / "bin"
    binaries.mkdir(parents=True)
    enumeration = "".join(
        f"GPU {index}: NVIDIA H100 80GB HBM3 (UUID: GPU-{index:012d})\n"
        for index in range(enumerated_gpus)
    )
    modes = "".join(f"{mode}\n" for mode in persistence_modes)
    detailed = "".join(
        f"{index}, GPU-{index:012d}, {mode}\n"
        for index, mode in enumerate(persistence_modes)
    )
    _write_stub(
        binaries,
        "nvidia-smi",
        'case "$*" in\n'
        f"*index,uuid,persistence_mode*) printf '%s' {shlex.quote(detailed)};;\n"
        f"*persistence_mode*) printf '%s' {shlex.quote(modes)};"
        f" exit {persistence_exit};;\n"
        f"-L) printf '%s' {shlex.quote(enumeration)}; exit {enumeration_exit};;\n"
        "*) printf '0, GPU-000000000000, 41, 300\\n';;\n"
        "esac\n"
        "exit 0\n",
    )
    down = " ".join(("gpu-fault-log-collector.service", *inactive_units))
    _write_stub(
        binaries,
        "systemctl",
        'verb="${1:-}"\n'
        "shift || true\n"
        'while [[ "${1:-}" == -* ]]; do shift; done\n'
        'unit="${1:-}"\n'
        f'down="{down}"\n'
        "for name in ${down}; do\n"
        '    if [[ "${unit}" == "${name}" ]]; then\n'
        '        case "${verb}" in\n'
        "        is-active|is-enabled) exit 3 ;;\n"
        "        esac\n"
        "    fi\n"
        "done\n"
        "exit 0\n",
    )
    dcgm_body = (
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 41\n'
        if dcgm_metrics
        else "# the exporter answered without a single supported metric\n"
    )
    _write_stub(
        binaries,
        "curl",
        'output=""\n'
        'url=""\n'
        "while (( $# )); do\n"
        '    case "$1" in\n'
        '    --output) output="$2"; shift 2 ;;\n'
        "    -H|--cacert) shift 2 ;;\n"
        "    -*) shift ;;\n"
        '    *) url="$1"; shift ;;\n'
        "    esac\n"
        "done\n"
        "emit() {\n"
        '    if [[ -n "${output}" ]]; then printf \'%s\' "$1" > "${output}"\n'
        "    else printf '%s' \"$1\"\n"
        "    fi\n"
        "}\n"
        'case "${url}" in\n'
        "*/healthz) exit 0 ;;\n"
        f"*9400/metrics) emit {shlex.quote(dcgm_body)}; exit 0 ;;\n"
        '*/latest) emit "[{\\"observed_at\\": \\"$(date -u +%FT%TZ)\\"}]"; exit 0 ;;\n'
        "esac\n"
        "exit 22\n",
    )
    env_file = workspace / "collector.env"
    env_file.write_text(
        "GPU_FAULT_CONTROL_PLANE_URL=https://control-plane.invalid\n"
        "GPU_FAULT_CONTROL_PLANE_TOKEN=cluster-token\n"
        "GPU_FAULT_CLUSTER_ID=cluster-a\n"
        "NODE_NAME=node-a\n"
        "GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9400/metrics\n"
        "GPU_FAULT_METRICS_MODE=dcgm\n"
        "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR=false\n"
        "GPU_FAULT_NODE_AGENT_PORT=9099\n"
        "GPU_FAULT_NODE_ADVERTISE_URL=http://127.0.0.1:9099\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(NODE_SCRIPTS[1])],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": ":".join([str(binaries), "/usr/bin", "/bin"]),
            "GPU_FAULT_ENV_FILE": str(env_file),
        },
    )


def _summary(stdout: str) -> dict[str, int]:
    line = [row for row in stdout.splitlines() if row.startswith("SUMMARY ")]
    assert len(line) == 1, f"expected exactly one SUMMARY line, got {line}"
    return {
        key: int(value)
        for key, value in (field.split("=", 1) for field in line[0].split()[1:])
    }


def test_verify_warns_on_gpu_health_and_still_fails_on_software_integrity(
    tmp_path: Path,
) -> None:
    degraded = _run_verify(
        tmp_path / "degraded",
        persistence_modes=("Enabled", "Disabled", "Enabled"),
        dcgm_metrics=False,
    )

    assert degraded.returncode == 0, (
        "a GPU that lost persistence mode must not fail the install: "
        f"{degraded.stdout}{degraded.stderr}"
    )
    assert "WARN  GPU persistence mode" in degraded.stdout, degraded.stdout
    assert "WARN  DCGM exporter supported metrics" in degraded.stdout, degraded.stdout
    assert "FAIL" not in degraded.stdout, (
        f"no software-integrity gate was broken in this scenario: {degraded.stdout}"
    )
    assert _summary(degraded.stdout)["warn"] == 2, degraded.stdout
    assert _summary(degraded.stdout)["fail"] == 0, degraded.stdout
    assert _summary(degraded.stdout)["pass"] > 0, degraded.stdout

    stopped = _run_verify(
        tmp_path / "stopped",
        persistence_modes=("Disabled",),
        inactive_units=("gpu-fault-metrics-collector.service",),
    )

    assert stopped.returncode == 1, (
        f"an inactive collector unit must stay fatal: {stopped.stdout}{stopped.stderr}"
    )
    assert "FAIL  metrics collector service" in stopped.stdout, stopped.stdout
    assert "WARN  GPU persistence mode" in stopped.stdout, stopped.stdout
    assert _summary(stopped.stdout)["fail"] == 1, stopped.stdout
    assert _summary(stopped.stdout)["warn"] == 1, stopped.stdout


def test_verify_fails_when_no_gpu_is_enumerated_at_all(tmp_path: Path) -> None:
    empty = _run_verify(tmp_path / "empty", enumerated_gpus=0, enumeration_exit=1)

    assert empty.returncode == 1, (
        f"zero enumerated GPUs must stay fatal: {empty.stdout}{empty.stderr}"
    )
    assert "FAIL  NVIDIA GPU enumeration" in empty.stdout, empty.stdout

    partial = _run_verify(tmp_path / "partial", enumerated_gpus=7, enumeration_exit=255)

    assert partial.returncode == 0, (
        "a partial enumeration is the degraded case the Agent must reach: "
        f"{partial.stdout}{partial.stderr}"
    )
    assert "WARN  NVIDIA GPU enumeration" in partial.stdout, partial.stdout


def test_verify_keeps_every_software_integrity_check_fatal() -> None:
    verifier = NODE_SCRIPTS[1].read_text()

    warned = set(re.findall(r'^\s*warn "([^"]+)"', verifier, re.MULTILINE))
    fatal = set(re.findall(r'^\s*(?:check|fail) "([^"]+)"', verifier, re.MULTILINE))

    assert warned == set(VERIFY_WARN_NAMES), (
        f"only the GPU-health gates may warn, found {sorted(warned)}"
    )
    for name in VERIFY_FATAL_NAMES:
        assert name in fatal, f"{name} must stay a fatal verifier check"
    assert 'exit "${failed}"' in verifier, (
        "the verifier exit code must still be driven by FAILs alone"
    )


def test_verify_warns_when_the_persistence_query_answers_nothing(
    tmp_path: Path,
) -> None:
    result = _run_verify(tmp_path, persistence_modes=(), persistence_exit=255)

    assert "WARN  GPU persistence mode (nvidia-smi reported no GPU)" in result.stdout, (
        f"an empty persistence query is not a pass: {result.stdout}"
    )
    assert "PASS  GPU persistence mode" not in result.stdout, result.stdout
    assert result.returncode == 0, (
        f"a GPU-health warning must not fail the verifier: {result.stdout}"
    )
    assert _summary(result.stdout)["warn"] >= 1, result.stdout
