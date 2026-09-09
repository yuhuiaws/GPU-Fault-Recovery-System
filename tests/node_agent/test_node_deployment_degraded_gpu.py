"""A degraded GPU is reported, not fatal, on the install and preflight paths.

Runs the installer's own marker writer and persistence verdict and the
preflight's enumeration gate as extracted bash probes: a GPU off the bus or
a dead driver records ``installer-degraded-gpu.json`` instead of aborting,
a healthy node clears a stale marker, rollback removes it before the state
root, and only a node with zero enumerable GPUs is kept off the rollout.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT, _write_stub


DEGRADED_MARKER_START = (
    'DEGRADED_GPU_MARKER="/var/lib/gpu-fault/installer-degraded-gpu.json"'
)

DEGRADED_MARKER_END = (
    'clear_degraded_gpu_marker() {\n    rm -f "${DEGRADED_GPU_MARKER}"\n}'
)


def _degraded_gpu_probe(target: Path) -> Path:
    """Run the installer's own marker writer, not a paraphrase of it."""
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(DEGRADED_MARKER_START) == 1, (
        "the installer must declare exactly one degraded-GPU marker path"
    )
    assert installer.count(DEGRADED_MARKER_END) == 1, (
        "the installer must clear the degraded-GPU marker from one place"
    )
    start = installer.index(DEGRADED_MARKER_START)
    end = installer.index(DEGRADED_MARKER_END) + len(DEGRADED_MARKER_END)
    probe = target / "degraded-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "PYTHON_COMMAND=python3\n"
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'DEGRADED_GPU_MARKER="$1"\n'
        'record_degraded_gpu_marker "$2"\n',
        encoding="utf-8",
    )
    return probe


def test_installer_records_the_degraded_gpu_marker_instead_of_dying(
    tmp_path: Path,
) -> None:
    probe = _degraded_gpu_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(
        binaries,
        "nvidia-smi",
        "printf '%s\\n' "
        "'0, GPU-aaaaaaaaaaaa, Enabled' "
        "'1, [N/A], [N/A]' "
        "'2, GPU-cccccccccccc, Disabled'\n"
        "exit 255\n",
    )
    marker = tmp_path / "state" / "installer-degraded-gpu.json"
    reason = "persistence mode was not Enabled after the persistence unit restart"

    result = subprocess.run(
        ["bash", str(probe), str(marker), reason],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": f"{binaries}:/usr/bin:/bin"},
    )

    assert result.returncode == 0, (
        f"a degraded GPU must not abort the install: {result.stdout}{result.stderr}"
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["observed_at"].endswith("Z"), payload
    assert payload["gpus"] == [
        {
            "index": "0",
            "uuid": "GPU-aaaaaaaaaaaa",
            "persistence_mode": "Enabled",
            "error": "",
        },
        {"index": "1", "uuid": "[N/A]", "persistence_mode": "[N/A]", "error": reason},
        {
            "index": "2",
            "uuid": "GPU-cccccccccccc",
            "persistence_mode": "Disabled",
            "error": reason,
        },
    ], payload
    assert marker.stat().st_mode & 0o777 == 0o644, oct(marker.stat().st_mode)
    assert not list(marker.parent.glob("*.tmp")), (
        "the marker must be published by tmp+mv without leaving the temporary file"
    )


PERSISTENCE_BLOCK_START = (
    '    persistence_modes="$(nvidia-smi --query-gpu=persistence_mode \\\n'
    '        --format=csv,noheader 2>/dev/null || true)"'
)

PERSISTENCE_BLOCK_END = (
    '    elif [[ "${DEGRADED_GPU_RECORDED}" == "false" ]]; then\n'
    "        clear_degraded_gpu_marker\n"
    "    fi"
)


def _persistence_probe(target: Path) -> Path:
    """Run the installer's own persistence verdict with the marker calls stubbed."""
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(PERSISTENCE_BLOCK_START) == 1, (
        "the installer must capture the persistence query in exactly one place"
    )
    assert installer.count(PERSISTENCE_BLOCK_END) == 1, (
        "the installer must decide about the degraded marker in one place"
    )
    start = installer.index(PERSISTENCE_BLOCK_START)
    end = installer.index(PERSISTENCE_BLOCK_END) + len(PERSISTENCE_BLOCK_END)
    probe = target / "persistence-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'DEGRADED_GPU_RECORDED="false"\n'
        "record_degraded_gpu_marker() { printf 'RECORDED %s\\n' \"$1\"; }\n"
        "clear_degraded_gpu_marker() { printf 'CLEARED\\n'; }\n"
        f"{installer[start:end]}\n",
        encoding="utf-8",
    )
    return probe


def test_installer_does_not_read_a_dead_driver_as_healthy_persistence(
    tmp_path: Path,
) -> None:
    """An empty persistence query has no line that differs from "Enabled".

    Piping a failed ``nvidia-smi`` into ``grep -Fvx Enabled`` therefore scored a
    driver that answered nothing as "every GPU is in persistence mode", and the
    new marker logic went one step further and deleted the marker an earlier
    install had written.
    """

    probe = _persistence_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{binaries}:/usr/bin:/bin", "NO_START": "false"},
        )

    _write_stub(binaries, "nvidia-smi", "exit 255\n")
    dead = run()
    assert dead.returncode == 0, (
        f"a dead persistence query must not abort the install: {dead.stderr}"
    )
    assert "RECORDED" in dead.stdout, (
        f"a driver that reports no GPU is a degraded GPU observation: {dead.stdout}"
    )
    assert "CLEARED" not in dead.stdout, (
        "an unreadable driver must never clear a marker an earlier install wrote"
    )

    _write_stub(binaries, "nvidia-smi", "printf 'Enabled\\nEnabled\\n'\n")
    healthy = run()
    assert healthy.returncode == 0, healthy.stderr
    assert "CLEARED" in healthy.stdout, (
        f"a healthy node must drop a stale marker: {healthy.stdout}"
    )
    assert "RECORDED" not in healthy.stdout, healthy.stdout


def test_installer_rollback_removes_the_degraded_gpu_marker() -> None:
    """A first install that warned and then died must leave no state root.

    ``rmdir`` refuses a directory that still holds the marker, so the state root
    the installer created would survive its own rollback.
    """

    installer = NODE_SCRIPTS[0].read_text()
    branch = installer.index(
        'if [[ "${status}" -ne 0 && "${STATE_ROOT_CREATED}" == "true" ]]; then'
    )
    removal = installer.index(
        "rm -f /var/lib/gpu-fault/installer-degraded-gpu.json", branch
    )
    rmdir = installer.index("rmdir /var/lib/gpu-fault", branch)
    assert removal < rmdir, (
        "the marker must be removed before the state root is reclaimed"
    )


PREFLIGHT_ENUMERATION_START = (
    'gpu_enumeration="$(\n'
    "    host_shell \"nvidia-smi -L 2>/dev/null || printf 'ENUMERATION_FAILED\\n'\"\n"
    ')"'
)

PREFLIGHT_ENUMERATION_END = (
    'if [[ "${gpu_enumeration}" == *ENUMERATION_FAILED* ]]; then\n'
    "    printf 'WARN  NVIDIA GPU enumeration (only %s GPU(s) are enumerable)\\n' \\\n"
    '        "${enumerated_gpus}"\n'
    "fi"
)


def _preflight_enumeration_probe(target: Path) -> Path:
    """Run the preflight's own enumeration gate.

    ``host_shell`` chroots into the host, which a unit test cannot do, so the
    probe replaces it with the same ``bash -ceu`` the chroot would run -- that
    is what makes the quoting of the fallback part of the contract.
    """
    preflight = (ROOT / "deploy/node/preflight-gpu-fault-node.sh").read_text()
    assert preflight.count(PREFLIGHT_ENUMERATION_START) == 1, (
        "the preflight must enumerate GPUs in exactly one place"
    )
    assert preflight.count(PREFLIGHT_ENUMERATION_END) == 1, (
        "the preflight must warn about a partial enumeration in one place"
    )
    start = preflight.index(PREFLIGHT_ENUMERATION_START)
    end = preflight.index(PREFLIGHT_ENUMERATION_END) + len(PREFLIGHT_ENUMERATION_END)
    probe = target / "preflight-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        'host_shell() { /bin/bash -ceu "$1"; }\n'
        f"{preflight[start:end]}\n"
        "printf 'PREFLIGHT_CONTINUED\\n'\n",
        encoding="utf-8",
    )
    return probe


def test_preflight_blocks_only_a_node_with_no_enumerable_gpu(tmp_path: Path) -> None:
    """One unreadable GPU must not disqualify the node from getting the Agent.

    ``nvidia-smi -L`` exits non-zero as soon as a single GPU is unreadable while
    it still lists the healthy ones, so the old ``die`` kept the release
    candidate off exactly the nodes that needed remediation.
    """

    probe = _preflight_enumeration_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{binaries}:/usr/bin:/bin"},
        )

    healthy = "".join(f"GPU {index}: NVIDIA H100 80GB HBM3\n" for index in range(8))
    _write_stub(binaries, "nvidia-smi", f"printf '%s' {shlex.quote(healthy)}\n")
    intact = run()
    assert intact.returncode == 0, intact.stdout + intact.stderr
    assert "WARN" not in intact.stdout, intact.stdout
    assert "PREFLIGHT_CONTINUED" in intact.stdout, intact.stdout

    partial = "".join(f"GPU {index}: NVIDIA H100 80GB HBM3\n" for index in range(7))
    _write_stub(
        binaries,
        "nvidia-smi",
        f"printf '%s' {shlex.quote(partial)}\n"
        "printf 'Unable to determine the device handle for GPU 7\\n' >&2\n"
        "exit 255\n",
    )
    degraded = run()
    assert degraded.returncode == 0, (
        f"a partial enumeration must not block the rollout: {degraded.stdout}"
    )
    assert (
        "WARN  NVIDIA GPU enumeration (only 7 GPU(s) are enumerable)" in degraded.stdout
    ), degraded.stdout
    assert "PREFLIGHT_CONTINUED" in degraded.stdout, degraded.stdout

    _write_stub(binaries, "nvidia-smi", "printf 'No devices were found\\n'\nexit 9\n")
    empty = run()
    assert empty.returncode != 0, (
        f"a node with no enumerable GPU is still fatal: {empty.stdout}"
    )
    assert "DIE: NVIDIA driver enumerated zero GPUs" in empty.stdout, empty.stdout
    assert "PREFLIGHT_CONTINUED" not in empty.stdout, empty.stdout


def test_preflight_no_longer_dies_on_a_partial_enumeration() -> None:
    preflight = (ROOT / "deploy/node/preflight-gpu-fault-node.sh").read_text()

    assert "NVIDIA driver cannot enumerate GPUs" not in preflight, (
        "a non-zero nvidia-smi -L is no longer by itself a preflight failure"
    )
