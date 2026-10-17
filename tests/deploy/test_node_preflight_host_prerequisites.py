"""The host preflight against a fake node root, through a stubbed ``chroot``.

Trial 11/attempt 7 (2026-10-04): two ``gpu-fault-preflight-*`` Jobs ran on
gpu-eks-cluster's p5en nodes (AL2023_x86_64_NVIDIA AMI) and the reconciler
exited 1 with its stderr redacted. The AMI ships neither ``/usr/bin/python3.12``
nor ``strace``; the preflight reported the first missing command only, and
nothing of it survived the sensitive-command redaction.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "deploy/node/preflight-gpu-fault-node.sh"
REQUIRED_MEMBERS = (
    "deploy/node/install-gpu-fault-collector.sh",
    "deploy/node/runtime-slot.sh",
    "deploy/node/runtime_integrity.py",
    "deploy/node/preflight-gpu-fault-node.sh",
    "deploy/systemd/gpu-fault-node-agent.service",
    "deploy/systemd/gpu-fault-host-collector.service",
    "deploy/systemd/gpu-fault-kernel-collector.service",
    "deploy/systemd/gpu-fault-metrics-collector.service",
    "requirements/node-runtime.lock",
)
HYPERPOD_TOOLS = ("curl", "nvidia-smi", "ps", "ss", "strace", "systemctl", "timeout")
#: What the EKS AL2023 NVIDIA AMI has: everything but python3.12 and strace.
AL2023_TOOLS = ("curl", "nvidia-smi", "ps", "ss", "systemctl", "timeout")


def _bundle(path: Path) -> tuple[str, str]:
    wheel = b"fixture-node-runtime-wheel"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for member in (
            *REQUIRED_MEMBERS,
            "dist/gpu_fault_node_runtime-0.0.0-py3-none-any.whl",
        ):
            data = wheel if member.endswith(".whl") else b"#\n"
            info = tarfile.TarInfo(f"gpu-fault-node-installer-0/{member}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    path.write_bytes(buffer.getvalue())
    return (
        hashlib.sha256(buffer.getvalue()).hexdigest(),
        hashlib.sha256(wheel).hexdigest(),
    )


def _fake_node(
    tmp_path: Path,
    tools: tuple[str, ...],
    *,
    python312: bool,
    bodies: dict[str, str] | None = None,
) -> Path:
    """A host root whose ``bin`` holds stubs for ``tools`` (and python3.12 when
    the AMI has it); ``chroot`` is stubbed to run the command against it.
    ``bodies`` replaces the stub body of individual tools."""

    root = tmp_path / "host"
    (root / "opt").mkdir(parents=True)
    (root / "dev").mkdir()
    (root / "dev/kmsg").write_text("")
    (root / "sys/class/infiniband").mkdir(parents=True)  # EFA devices present
    (root / "usr/bin").mkdir(parents=True)
    host_bin = root / "bin"
    host_bin.mkdir()
    bodies = {
        "nvidia-smi": "printf 'GPU 0: NVIDIA H200 (UUID: GPU-0)\\n'\n",
        "systemctl": (
            'case "$1" in\n'
            "  is-active) printf 'inactive\\n'; exit 3 ;;\n"
            "  is-enabled) printf 'disabled\\n'; exit 1 ;;\n"
            "esac\n"
            "printf 'running\\n'\n"
        ),
        "ethtool": "exit 0\n",
        "rdma": "exit 0\n",
        **(bodies or {}),
    }
    for tool in (*tools, "ethtool", "rdma"):
        stub = host_bin / tool
        # /bin/bash by path: inside the stubbed chroot PATH is the fake bin only.
        stub.write_text("#!/bin/bash\n" + bodies.get(tool, "exit 0\n"))
        stub.chmod(0o755)
    if python312:
        python = root / "usr/bin/python3.12"
        python.write_text("#!/bin/bash\nexit 0\n")
        python.chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    chroot = stubs / "chroot"
    chroot.write_text(
        "#!/usr/bin/env bash\n"
        "# chroot ROOT /bin/bash -ceu CMD -> run CMD against the fake root\n"
        'root="$1"; cmd="$4"\n'
        'cmd="${cmd//\\/usr\\/bin\\/python3.12/${root}/usr/bin/python3.12}"\n'
        'PATH="${root}/bin" exec /bin/bash -ceu "${cmd}"\n'
    )
    chroot.chmod(0o755)
    return root


def _run(
    tmp_path: Path, root: Path, *, provider: str, dcgm_hostengine: str | None = None
) -> subprocess.CompletedProcess[str]:
    bundle = tmp_path / "bundle.tar.gz"
    bundle_sha, wheel_sha = _bundle(bundle)
    return subprocess.run(
        ["bash", str(PREFLIGHT)],
        env={
            "PATH": os.pathsep.join((str(tmp_path / "stubs"), os.environ["PATH"])),
            "GPU_FAULT_PREFLIGHT_HOST_ROOT": str(root),
            "GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE": str(bundle),
            "GPU_FAULT_PREFLIGHT_BUNDLE_SHA256": bundle_sha,
            "GPU_FAULT_PREFLIGHT_ARTIFACT_SHA256": wheel_sha,
            "GPU_FAULT_INSTALLER_MIN_FREE_BYTES": "1",
            "GPU_FAULT_CLUSTER_PROVIDER": provider,
            "TARGET_NODE_NAME": "ip-10-0-33-205.us-west-2.compute.internal",
            "TARGET_NODE_UID": "node-uid-1",
            **(
                {"GPU_FAULT_DCGM_HOSTENGINE": dcgm_hostengine}
                if dcgm_hostengine is not None
                else {}
            ),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_al2023_node_without_python312_and_strace_fails_with_one_full_verdict(
    tmp_path: Path,
) -> None:
    root = _fake_node(tmp_path, AL2023_TOOLS, python312=False)
    result = _run(tmp_path, root, provider="eks-managed")
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "ERROR: host prerequisites unavailable: /usr/bin/python3.12, strace; the EKS "
        "AL2023 NVIDIA AMI does not ship them; install them in the managed node group "
        "launch template user data (dnf install -y python3.12 strace; see "
        "管理员日常运维 / administrator-operations §5.0.1) and roll the node group, "
        "then rerun the join"
    ) in result.stderr, result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict == {
        "node_id": "ip-10-0-33-205.us-west-2.compute.internal",
        "node_uid": "node-uid-1",
        "status": "FAILED",
        "reasons": ["host prerequisites unavailable: /usr/bin/python3.12, strace"],
        "dcgm": {
            "mode": "embedded",
            "state": "not-installed",
            "service": {"active": "inactive", "enabled": "disabled"},
            "dcgmi": False,
            "nv_hostengine": False,
            "cuda_packages": [],
            "driver_cuda_major": "",
            "processes": [],
            "discovery": "skipped",
            "conflict": False,
        },
    }, "the machine-readable verdict names every missing prerequisite at once"


def test_hyperpod_node_remedy_does_not_mention_the_eks_ami(tmp_path: Path) -> None:
    root = _dcgm_host(
        tmp_path,
        STATES["host-active"],
        runtime=("curl", "nvidia-smi", "ps", "ss", "systemctl", "timeout"),
    )
    result = _run(tmp_path, root, provider="hyperpod")
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "ERROR: host prerequisites unavailable: strace; install them on the node image"
        in (result.stderr)
    ), result.stderr
    assert "AL2023" not in result.stderr, "HyperPod nodes get the generic remedy"


@pytest.mark.parametrize("provider", ["hyperpod", "eks-managed"])
def test_a_complete_node_passes_with_the_unchanged_verdict(
    tmp_path: Path, provider: str
) -> None:
    # A HyperPod node is complete with its host DCGM running; a plain EKS node
    # (embedded mode) needs none.
    root = (
        _dcgm_host(tmp_path, STATES["host-active"])
        if provider == "hyperpod"
        else _fake_node(tmp_path, HYPERPOD_TOOLS, python312=True)
    )
    result = _run(tmp_path, root, provider=provider)
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict["status"] == "PASSED" and verdict["node_uid"] == "node-uid-1", (
        verdict
    )
    assert result.stderr == "", "a passing preflight is silent on stderr"


# --- the DCGM host engine of a GPU node -------------------------------------------
#
# Every verdict reports the node's DCGM host engine ("dcgm"). Host mode (an
# eks-managed node group labelled gpu-fault.io/dcgm-hostengine=host) refuses
# everything but a running nvidia-dcgm that sees the GPUs and owns every
# nv-hostengine, listed with the other missing prerequisites; the embedded mode
# refuses any host engine next to the exporter's embedded one. A HyperPod node
# is always host mode: the four engine checks refuse, a CUDA flavor mismatch or
# an unowned nv-hostengine is reported (WARN) but not refused.
UNIT_CGROUP = "0::/system.slice/nvidia-dcgm.service\n"
SESSION_CGROUP = "0::/user.slice/user-1000.slice/session-3.scope\n"
CORE = "datacenter-gpu-manager-4-core"
DCGM_REMEDY = (
    "the managed node group declares gpu-fault.io/dcgm-hostengine=host: install "
    "the pinned datacenter-gpu-manager-4 packages and systemctl enable --now "
    "nvidia-dcgm in the launch template user data (see 管理员日常运维 / "
    "administrator-operations §5.0.1), or remove the label and deploy to run the "
    "exporter's embedded hostengine"
)
CUDA_REMEDY = (
    "the DCGM CUDA flavor must match the driver's CUDA major: install "
    "datacenter-gpu-manager-4-cuda13 (nvidia-smi reports CUDA Version 13)"
)
UNMANAGED_REMEDY = (
    "stop the hand-started nv-hostengine; nvidia-dcgm.service must own the host engine"
)
CONFLICT_REMEDY = (
    "a host nv-hostengine runs or is enabled next to the exporter's embedded "
    "hostengine: label the node group gpu-fault.io/dcgm-hostengine=host and "
    "deploy, or disable nvidia-dcgm (systemctl disable --now nvidia-dcgm; stop a "
    "hand-started nv-hostengine)"
)
AL2023_REMEDY = (
    "the EKS AL2023 NVIDIA AMI does not ship them; install them in the managed "
    "node group launch template user data (dnf install -y python3.12 strace; see "
    "管理员日常运维 / administrator-operations §5.0.1) and roll the node group, "
    "then rerun the join"
)


@dataclass(frozen=True)
class DcgmNode:
    """One node's DCGM installation, as the host stubs present it."""

    dcgmi: bool = True
    hostengine: bool = True
    packages: tuple[str, ...] = (CORE, "datacenter-gpu-manager-4-cuda13")
    driver_major: str = "13"
    active: str = "inactive"
    enabled: str = "disabled"
    processes: tuple[tuple[int, str], ...] = ()
    discovery_status: int = 0


#: The six states of an eks-managed GPU node, one installation each.
STATES = {
    # (1) the AL2023 NVIDIA AMI as shipped
    "not-installed": DcgmNode(dcgmi=False, hostengine=False, packages=(), enabled=""),
    # (2) packages from the launch template, nvidia-dcgm disabled
    "installed-inactive": DcgmNode(),
    # (3) the host-mode target
    "host-active": DcgmNode(
        active="active", enabled="enabled", processes=((4242, UNIT_CGROUP),)
    ),
    # (4) the cuda12 build on a CUDA 13 driver
    "cuda-mismatch": DcgmNode(packages=(CORE, "datacenter-gpu-manager-4-cuda12")),
    # (5) enabled and crash-looping (auto-restart reads activating)
    "host-failed": DcgmNode(active="activating", enabled="enabled"),
    # (6) an nv-hostengine somebody started by hand
    "unmanaged-hostengine": DcgmNode(processes=((777, SESSION_CGROUP),)),
}


def _host_bodies(node: DcgmNode) -> dict[str, str]:
    enabled_exit = 0 if node.enabled == "enabled" else 1
    smi_header = (
        f"printf '| NVIDIA-SMI 580.95.05  Driver Version: 580.95.05  CUDA "
        f"Version: {node.driver_major}.0 |\\n'"
        if node.driver_major
        else "printf '| NVIDIA-SMI 580.95.05  Driver Version: 580.95.05 |\\n'"
    )
    cuda = " ".join(item for item in node.packages if "-cuda" in item)
    return {
        "systemctl": (
            'case "$1" in\n'
            f"  is-active) printf '%s\\n' '{node.active}'; "
            f"[[ '{node.active}' == active ]]; exit ;;\n"
            + (
                f"  is-enabled) printf '%s\\n' '{node.enabled}'; exit {enabled_exit} ;;\n"
                if node.enabled
                else "  is-enabled) exit 1 ;;\n"
            )
            + "esac\nprintf 'running\\n'\n"
        ),
        "nvidia-smi": (
            'if [[ "${1:-}" == -L ]]; then '
            "printf 'GPU 0: NVIDIA H200 (UUID: GPU-0)\\n'; else "
            + smi_header
            + "; fi\n"
        ),
        "rpm": (f"printf '%s\\n' {cuda}\n" if cuda else "exit 0\n"),
        "pgrep": "".join(f"printf '%s\\n' {pid}\n" for pid, _ in node.processes)
        or "exit 1\n",
        "dcgmi": (
            f'[[ "$1 ${{2:-}}" == "discovery -l" ]] && exit {node.discovery_status}\n'
            "exit 0\n"
        ),
        "timeout": 'shift\nexec "$@"\n',
    }


def _dcgm_host(
    tmp_path: Path,
    node: DcgmNode,
    *,
    runtime: tuple[str, ...] = HYPERPOD_TOOLS,
    python312: bool = True,
) -> Path:
    tools = (
        *runtime,
        "rpm",
        "pgrep",
        *(("dcgmi",) if node.dcgmi else ()),
        *(("nv-hostengine",) if node.hostengine else ()),
    )
    root = _fake_node(tmp_path, tools, python312=python312, bodies=_host_bodies(node))
    for pid, cgroup in node.processes:
        (root / "proc" / str(pid)).mkdir(parents=True)
        (root / "proc" / str(pid) / "cgroup").write_text(cgroup)
    return root


def _verdict(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


#: (state, mode) -> the reason the verdict fails with, or None for PASSED.
MATRIX = {
    ("not-installed", "host"): (
        "host prerequisites unavailable: dcgmi, nv-hostengine, "
        "nvidia-dcgm.service (inactive)"
    ),
    ("installed-inactive", "host"): (
        "host prerequisites unavailable: nvidia-dcgm.service (inactive)"
    ),
    ("host-active", "host"): None,
    ("cuda-mismatch", "host"): (
        "host prerequisites unavailable: nvidia-dcgm.service (inactive), "
        "datacenter-gpu-manager-4-cuda13 (installed: cuda12)"
    ),
    ("host-failed", "host"): (
        "host prerequisites unavailable: nvidia-dcgm.service (activating)"
    ),
    ("unmanaged-hostengine", "host"): (
        "host prerequisites unavailable: nvidia-dcgm.service (inactive), "
        "nv-hostengine pid 777 outside nvidia-dcgm.service"
    ),
    ("not-installed", "embedded"): None,
    ("installed-inactive", "embedded"): None,
    ("host-active", "embedded"): (
        "DCGM host engine conflict: nvidia-dcgm.service (active, enabled), "
        "nv-hostengine pid 4242"
    ),
    ("cuda-mismatch", "embedded"): None,
    ("host-failed", "embedded"): (
        "DCGM host engine conflict: nvidia-dcgm.service (activating, enabled)"
    ),
    ("unmanaged-hostengine", "embedded"): (
        "DCGM host engine conflict: nv-hostengine pid 777 outside nvidia-dcgm.service"
    ),
}
REMEDIES = {
    ("not-installed", "host"): DCGM_REMEDY,
    ("installed-inactive", "host"): DCGM_REMEDY,
    ("cuda-mismatch", "host"): f"{DCGM_REMEDY}; {CUDA_REMEDY}",
    ("host-failed", "host"): DCGM_REMEDY,
    ("unmanaged-hostengine", "host"): f"{DCGM_REMEDY}; {UNMANAGED_REMEDY}",
    ("host-active", "embedded"): CONFLICT_REMEDY,
    ("host-failed", "embedded"): CONFLICT_REMEDY,
    ("unmanaged-hostengine", "embedded"): CONFLICT_REMEDY,
}


@pytest.mark.parametrize(("state", "mode"), sorted(MATRIX))
def test_each_dcgm_state_in_each_mode(tmp_path: Path, state: str, mode: str) -> None:
    root = _dcgm_host(tmp_path, STATES[state])
    result = _run(tmp_path, root, provider="eks-managed", dcgm_hostengine=mode)
    verdict = _verdict(result)
    dcgm = verdict["dcgm"]
    assert (dcgm["mode"], dcgm["state"]) == (mode, state), dcgm
    reason = MATRIX[(state, mode)]
    if reason is None:
        assert result.returncode == 0, result.stdout + result.stderr
        assert verdict["status"] == "PASSED" and dcgm["conflict"] is False, verdict
        assert result.stderr == "", "a passing node is silent on stderr"
        return
    assert result.returncode == 1, result.stdout + result.stderr
    assert verdict["status"] == "FAILED" and verdict["reasons"] == [reason], verdict
    assert result.stderr.strip().splitlines()[-1] == (
        f"ERROR: {reason}; {REMEDIES[(state, mode)]}"
    ), result.stderr
    assert dcgm["conflict"] is (
        mode == "embedded" or state == "unmanaged-hostengine"
    ), "a conflict is a second (or an unowned) host engine"


def test_the_dcgm_member_describes_the_node(tmp_path: Path) -> None:
    root = _dcgm_host(tmp_path, STATES["host-active"])
    result = _run(tmp_path, root, provider="eks-managed", dcgm_hostengine="host")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _verdict(result)["dcgm"] == {
        "mode": "host",
        "state": "host-active",
        "service": {"active": "active", "enabled": "enabled"},
        "dcgmi": True,
        "nv_hostengine": True,
        "cuda_packages": ["13"],
        "driver_cuda_major": "13",
        "processes": [{"pid": 4242, "unit": "nvidia-dcgm.service"}],
        "discovery": "ok",
        "conflict": False,
    }, "the deploy sees which engine the node runs and who owns it"
    unmanaged = _dcgm_host(tmp_path / "unmanaged", STATES["unmanaged-hostengine"])
    dcgm = _verdict(
        _run(
            tmp_path / "unmanaged",
            unmanaged,
            provider="eks-managed",
            dcgm_hostengine="embedded",
        )
    )["dcgm"]
    assert dcgm["processes"] == [{"pid": 777, "unit": ""}], dcgm
    assert dcgm["discovery"] == "ok", "a running engine is asked for its GPUs"
    assert (
        _verdict(_run(tmp_path, root, provider="eks-managed", dcgm_hostengine=None))[
            "dcgm"
        ]["mode"]
        == "embedded"
    ), "an eks node without the mode is embedded"


def test_an_engine_that_runs_but_sees_no_gpu_is_host_failed(tmp_path: Path) -> None:
    node = DcgmNode(
        active="active",
        enabled="enabled",
        processes=((4242, UNIT_CGROUP),),
        discovery_status=226,
    )
    root = _dcgm_host(tmp_path, node)
    result = _run(tmp_path, root, provider="eks-managed", dcgm_hostengine="host")
    verdict = _verdict(result)
    assert verdict["dcgm"]["state"] == "host-failed", verdict
    assert verdict["dcgm"]["discovery"] == "exit 226", verdict
    assert verdict["reasons"] == [
        "host prerequisites unavailable: dcgmi discovery -l (exit 226)"
    ], verdict


def test_host_mode_lists_every_item_with_the_missing_runtime_tools(
    tmp_path: Path,
) -> None:
    root = _dcgm_host(
        tmp_path, STATES["cuda-mismatch"], runtime=AL2023_TOOLS, python312=False
    )
    result = _run(tmp_path, root, provider="eks-managed", dcgm_hostengine="host")
    missing = (
        "/usr/bin/python3.12, strace, nvidia-dcgm.service (inactive), "
        "datacenter-gpu-manager-4-cuda13 (installed: cuda12)"
    )
    assert _verdict(result)["reasons"] == [
        f"host prerequisites unavailable: {missing}"
    ], result.stdout
    assert result.stderr.strip().splitlines()[-1] == (
        f"ERROR: host prerequisites unavailable: {missing}; {AL2023_REMEDY}; "
        f"{DCGM_REMEDY}; {CUDA_REMEDY}"
    ), result.stderr


def test_missing_tools_and_a_conflict_fail_in_one_verdict(tmp_path: Path) -> None:
    root = _dcgm_host(
        tmp_path, STATES["host-active"], runtime=AL2023_TOOLS, python312=False
    )
    result = _run(tmp_path, root, provider="eks-managed", dcgm_hostengine="embedded")
    assert result.returncode == 1, result.stdout + result.stderr
    assert _verdict(result)["reasons"] == [
        "host prerequisites unavailable: /usr/bin/python3.12, strace",
        "DCGM host engine conflict: nvidia-dcgm.service (active, enabled), "
        "nv-hostengine pid 4242",
    ], result.stdout
    assert result.stderr.strip().splitlines() == [
        "ERROR: host prerequisites unavailable: /usr/bin/python3.12, strace; "
        + AL2023_REMEDY,
        "ERROR: DCGM host engine conflict: nvidia-dcgm.service (active, enabled), "
        "nv-hostengine pid 4242; " + CONFLICT_REMEDY,
    ], result.stderr


HYPERPOD_DCGM_REMEDY = (
    "a HyperPod node runs DCGM as a host service the exporter dials at "
    "127.0.0.1:5555: the instance group's AMI must ship datacenter-gpu-manager-4 "
    "and run nvidia-dcgm (systemctl enable --now nvidia-dcgm; dcgmi discovery -l "
    "lists the GPUs; see 管理员日常运维 / administrator-operations §5.0.1)"
)
#: HyperPod (always host mode): the four engine checks refuse, nothing else.
HYPERPOD_MATRIX = {
    "not-installed": (
        "host prerequisites unavailable: dcgmi, nv-hostengine, "
        "nvidia-dcgm.service (inactive)"
    ),
    "installed-inactive": (
        "host prerequisites unavailable: nvidia-dcgm.service (inactive)"
    ),
    "host-active": None,
    "cuda-mismatch": "host prerequisites unavailable: nvidia-dcgm.service (inactive)",
    "host-failed": "host prerequisites unavailable: nvidia-dcgm.service (activating)",
    "unmanaged-hostengine": (
        "host prerequisites unavailable: nvidia-dcgm.service (inactive)"
    ),
}


@pytest.mark.parametrize("state", sorted(HYPERPOD_MATRIX))
def test_a_hyperpod_node_is_held_to_the_host_engine_the_exporter_dials(
    tmp_path: Path, state: str
) -> None:
    """The HyperPod AMI is expected to run nvidia-dcgm (the checked-in exporter
    dials it); an AMI release without it must not pass silently."""

    root = _dcgm_host(tmp_path, STATES[state])
    result = _run(tmp_path, root, provider="hyperpod")
    verdict = _verdict(result)
    assert (verdict["dcgm"]["mode"], verdict["dcgm"]["state"]) == ("host", state), (
        verdict
    )
    reason = HYPERPOD_MATRIX[state]
    if reason is None:
        assert result.returncode == 0, result.stdout + result.stderr
        assert verdict["status"] == "PASSED", verdict
        return
    assert result.returncode == 1, result.stdout + result.stderr
    assert verdict["reasons"] == [reason], verdict
    assert result.stderr.strip().splitlines()[-1] == (
        f"ERROR: {reason}; {HYPERPOD_DCGM_REMEDY}"
    ), result.stderr
    assert "AL2023" not in result.stderr and "dcgm-hostengine=host" not in (
        result.stderr
    ), "a HyperPod node gets the HyperPod remedy"


def test_a_hyperpod_node_reports_but_keeps_a_flavor_mismatch_and_a_foreign_engine(
    tmp_path: Path,
) -> None:
    node = DcgmNode(
        packages=(CORE, "datacenter-gpu-manager-4-cuda12"),
        active="active",
        enabled="enabled",
        processes=((4242, UNIT_CGROUP), (777, SESSION_CGROUP)),
    )
    root = _dcgm_host(tmp_path, node)
    result = _run(tmp_path, root, provider="hyperpod")
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = _verdict(result)
    assert verdict["status"] == "PASSED", verdict
    assert verdict["dcgm"]["state"] == "unmanaged-hostengine", verdict
    assert verdict["dcgm"]["conflict"] is True, "status flags the unowned engine"
    assert (
        "WARN  DCGM host engine: datacenter-gpu-manager-4-cuda13 (installed: cuda12)"
        in result.stdout
    ), result.stdout
    assert (
        "WARN  DCGM host engine: nv-hostengine pid 777 outside nvidia-dcgm.service"
        in result.stdout
    ), result.stdout
    assert result.stderr == "", "a warning is no refusal"


def test_a_hyperpod_refusal_lists_the_runtime_tools_with_a_working_engine(
    tmp_path: Path,
) -> None:
    root = _dcgm_host(
        tmp_path, STATES["host-active"], runtime=AL2023_TOOLS, python312=False
    )
    result = _run(tmp_path, root, provider="hyperpod")
    verdict = _verdict(result)
    assert verdict["reasons"] == [
        "host prerequisites unavailable: /usr/bin/python3.12, strace"
    ], verdict
    assert verdict["dcgm"]["state"] == "host-active", verdict
    assert result.stderr.strip().splitlines()[-1] == (
        "ERROR: host prerequisites unavailable: /usr/bin/python3.12, strace; "
        "install them on the node image before joining"
    ), result.stderr
