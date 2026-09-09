"""The installer's wheel discovery and the xz-compressed wheel ConfigMaps.

Runs the installer's own discovery block against both release layouts (a
repo checkout's ``dist/<release_id>/`` and the flattened node bundle) and
pins the release-manifest digest gate; the deploy scripts must accept the
``<wheel>.xz`` ConfigMap key.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT


WHEEL_BLOCK_START = 'MANIFEST_WHEEL_SHA256=""'

WHEEL_BLOCK_END = (
    'sha256 ${WHEEL_SHA256} does not match expected ${EXPECTED_WHEEL_SHA256}"'
)


def _wheel_discovery_probe(target: Path) -> Path:
    """Run the installer's own wheel discovery, not a paraphrase of it.

    ``bash -n`` and substring assertions both pass on a ``find`` that matches
    nothing, which is exactly how ``-maxdepth 1 -type f`` survived: the node
    bundle flattens the wheel into ``dist/``, so the rule looked right, while
    a repo checkout keeps it in the content-addressed ``dist/<release_id>/``
    and every checkout install died with "collector wheel not found".

    The probe takes the expected wheel digest as its second argument, the
    way the Job passes ``--wheel-sha256``; the digest is a hard gate.
    """
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(WHEEL_BLOCK_START) == 1
    assert installer.count(WHEEL_BLOCK_END) == 1
    start = installer.index(WHEEL_BLOCK_START)
    end = installer.index(WHEEL_BLOCK_END) + len(WHEEL_BLOCK_END)
    probe = target / "probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'REPO_DIR="$1"\n'
        'EXPECTED_WHEEL_SHA256="${2:-}"\n'
        "PYTHON_COMMAND=python3\n"
        'WHEEL=""\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'printf \'WHEEL=%s\\nSHA=%s\\n\' "${WHEEL}" "${WHEEL_SHA256}"\n',
        encoding="utf-8",
    )
    return probe


def _run_probe(
    probe: Path, repo_dir: Path, expected: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(repo_dir), expected],
        capture_output=True,
        text=True,
        check=False,
    )


def test_installer_finds_the_wheel_in_both_release_layouts(tmp_path: Path) -> None:
    probe = _wheel_discovery_probe(tmp_path)

    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_control_plane-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"checkout-wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest = checkout / "dist/current-release.json"
    manifest.write_text(
        json.dumps(
            {"wheel": wheel.relative_to(checkout).as_posix(), "wheel_sha256": digest}
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, checkout, digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={wheel}" in result.stdout
    assert f"SHA={digest}" in result.stdout

    bundle = tmp_path / "bundle"
    flat = bundle / "dist/gpu_fault_node_runtime-0.10.0.whl"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"bundle-wheel")

    result = _run_probe(probe, bundle, hashlib.sha256(b"bundle-wheel").hexdigest())
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={flat}" in result.stdout


def test_installer_refuses_a_wheel_the_release_manifest_disowns(tmp_path: Path) -> None:
    probe = _wheel_discovery_probe(tmp_path)
    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_control_plane-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"tampered")
    original = hashlib.sha256(b"original").hexdigest()
    (checkout / "dist/current-release.json").write_text(
        json.dumps(
            {"wheel": wheel.relative_to(checkout).as_posix(), "wheel_sha256": original}
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, checkout, original)
    assert result.returncode != 0
    assert "does not match" in result.stdout

    empty = tmp_path / "empty"
    (empty / "dist").mkdir(parents=True)
    result = _run_probe(probe, empty, original)
    assert result.returncode != 0
    assert "collector wheel not found" in result.stdout


def test_deploy_scripts_accept_xz_compressed_wheel_configmaps() -> None:
    """Wheel ConfigMaps hold ``<wheel>.xz`` since the control-plane wheel
    outgrew the 1 MiB ceiling; the GPU-stage reconciler deploy refused the
    executor wheel ConfigMap live because it only looked for the raw key."""
    reconciler = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text()
    assert (
        'if sys.argv[1] not in keys and sys.argv[1] + ".xz" not in keys:' in reconciler
    ), "reconciler deploy must accept the xz-compressed executor wheel key"
    role_split = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text()
    assert 'for key in (name + ".xz", name):' in role_split, (
        "role-split apply must derive the wheel digest from either form"
    )
    assert "lzma.decompress(data)" in role_split, role_split[:0]
