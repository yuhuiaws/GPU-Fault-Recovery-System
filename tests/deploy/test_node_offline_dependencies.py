from __future__ import annotations

import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

import pytest
import yaml

from gpu_fault.node_installer_rendering import configure_node_dependencies
from tests.deploy.test_node_preflight_inputs import inputs, render

ROOT = Path(__file__).resolve().parents[2]
INTEGRITY = runpy.run_path(str(ROOT / "deploy/node/runtime_integrity.py"))
VALIDATE = INTEGRITY["validate_wheelhouse"]


@pytest.fixture
def wheelhouse(tmp_path):
    directory = tmp_path / "wheelhouse"
    directory.mkdir()
    files = {}
    locks = {}
    for name, content in (
        ("node-runtime.lock", b"runtime lock"),
        ("node-tools.lock", b"tools lock"),
        ("dependency.whl", b"opaque pinned wheel bytes"),
    ):
        (directory / name).write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        files[name] = {"sha256": sha, "size": len(content)}
        if name.endswith(".lock"):
            locks[name] = sha
    from scripts.node_wheelhouse import PLATFORM

    inventory = directory / "inventory.json"
    inventory.write_text(
        json.dumps(
            {"schema_version": 1, "platform": PLATFORM, "files": files, "locks": locks}
        )
    )
    expected = hashlib.sha256(inventory.read_bytes()).hexdigest()
    return directory, expected


def test_offline_inventory_is_checked_without_installing_or_extracting(
    wheelhouse, tmp_path
):
    directory, sha = wheelhouse
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in ("node-runtime.lock", "node-tools.lock"):
            raw = (directory / name).read_bytes()
            item = tarfile.TarInfo("bundle/requirements/" + name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    before = set(tmp_path.rglob("*"))
    VALIDATE(directory, sha, lock=None, bundle=bundle)
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    "damage", ["inventory", "wheel", "extra", "symlink", "lock", "platform"]
)
def test_offline_inventory_refuses_damaged_or_mismatched_payload(
    wheelhouse, tmp_path, damage
):
    directory, sha = wheelhouse
    lock = directory / "node-runtime.lock"
    if damage == "inventory":
        sha = "0" * 64
    elif damage == "wheel":
        (directory / "dependency.whl").write_bytes(b"tampered")
    elif damage == "extra":
        (directory / "extra.whl").write_bytes(b"extra")
    elif damage == "symlink":
        source = directory / "dependency.whl"
        target = tmp_path / "elsewhere.whl"
        source.rename(target)
        source.symlink_to(target)
    elif damage == "lock":
        lock = tmp_path / "candidate.lock"
        lock.write_text("different lock")
    else:
        path = directory / "inventory.json"
        data = json.loads(path.read_bytes())
        data["platform"]["machine"] = "aarch64"
        path.write_text(json.dumps(data))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        VALIDATE(directory, sha, lock=lock, bundle=None)


def test_dependency_init_mounts_no_host_or_secret_and_installer_cannot_write(tmp_path):
    rendered = render(tmp_path, inputs())
    assert rendered.returncode == 0, rendered.stderr
    job = yaml.safe_load(rendered.stdout)
    configure_node_dependencies(
        job, "registry.example/node@sha256:" + "a" * 64, "b" * 64
    )
    pod = job["spec"]["template"]["spec"]
    init = pod["initContainers"][0]
    assert pod["automountServiceAccountToken"] is False
    assert init["volumeMounts"] == [
        {"name": "node-wheelhouse", "mountPath": "/wheelhouse"}
    ]
    assert init["securityContext"]["allowPrivilegeEscalation"] is False
    installer = pod["containers"][0]
    mount = next(
        item for item in installer["volumeMounts"] if item["name"] == "node-wheelhouse"
    )
    assert mount["readOnly"] is True
    assert mount["mountPath"] == "/host/run/gpu-fault-node-wheelhouse"
    assert "inventory.json" not in str(
        next(item for item in pod["volumes"] if item["name"] == "installer")
    )
