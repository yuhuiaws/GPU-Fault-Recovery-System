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


def _fake_node(tmp_path: Path, tools: tuple[str, ...], *, python312: bool) -> Path:
    """A host root whose ``bin`` holds stubs for ``tools`` (and python3.12 when
    the AMI has it); ``chroot`` is stubbed to run the command against it."""

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
        "systemctl": "printf 'running\\n'\n",
        "ethtool": "exit 0\n",
        "rdma": "exit 0\n",
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
    tmp_path: Path, root: Path, *, provider: str
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
        "AL2023 NVIDIA AMI does not ship them; install them on the node image or in "
        "the managed node group launch template user data "
        "(dnf install -y python3.12 strace), then rerun the join"
    ) in result.stderr, result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict == {
        "node_id": "ip-10-0-33-205.us-west-2.compute.internal",
        "node_uid": "node-uid-1",
        "status": "FAILED",
        "reasons": ["host prerequisites unavailable: /usr/bin/python3.12, strace"],
    }, "the machine-readable verdict names every missing prerequisite at once"


def test_hyperpod_node_remedy_does_not_mention_the_eks_ami(tmp_path: Path) -> None:
    root = _fake_node(
        tmp_path,
        ("curl", "nvidia-smi", "ps", "ss", "systemctl", "timeout"),
        python312=True,
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
    root = _fake_node(tmp_path, HYPERPOD_TOOLS, python312=True)
    result = _run(tmp_path, root, provider=provider)
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict["status"] == "PASSED" and verdict["node_uid"] == "node-uid-1", (
        verdict
    )
    assert result.stderr == "", "a passing preflight is silent on stderr"
