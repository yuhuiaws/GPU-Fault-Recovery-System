"""Batch 245: the registry Secret reaches running CPU Pods as a file too.

``GPU_FAULT_REGIONAL_CLUSTERS_JSON`` is an env from a ``secretKeyRef``, frozen
at container start, so after a join or remove every replica that predated it
compared a stale copy with the durable head and reported ``secret_drift``.
The three regional role Deployments now also project the Secret's
``clusters.json`` read-only under one directory, without ``subPath`` (kubelet
never refreshes a subPath mount), and the application factory hands that file
to the registry runtime. The env stays: settings validation and the first
bootstrap seed read it. Nothing here talks to kubectl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import gpu_fault.app.factory as factory_module
from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.regional_registry import (
    REGIONAL_CLUSTERS_SECRET_DIR,
    REGIONAL_CLUSTERS_SECRET_FILE,
)
from gpu_fault.regional_registry_runtime import SECRET_SOURCE_MOUNTED
from gpu_fault.store import InMemoryStore
from tests.regional._regional_support import TOKEN_A, registration

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "deploy" / "control-plane" / "regional"
PATCH = REGIONAL / "regional-control-plane-patch.yaml"
GENERATED = REGIONAL / "generated"
SECRET_NAME = "gpu-fault-regional-clusters"
SECRET_KEY = "clusters.json"
VOLUME_NAME = "regional-clusters"
ENV_NAME = "GPU_FAULT_REGIONAL_CLUSTERS_JSON"
ROLE_MANIFESTS = (
    "gpu-fault-api-ha-ingress.yaml",
    "gpu-fault-control-worker.yaml",
    "gpu-fault-telemetry-spool-worker.yaml",
)


def _pod_spec(path: Path) -> dict[str, Any]:
    documents = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    (deployment,) = [d for d in documents if d.get("kind") == "Deployment"]
    return deployment["spec"]["template"]["spec"]


def _assert_secret_file(spec: dict[str, Any], label: str) -> None:
    volumes = [v for v in spec.get("volumes", []) if v.get("name") == VOLUME_NAME]
    assert len(volumes) == 1, f"{label}: one {VOLUME_NAME} volume"
    secret = volumes[0].get("secret") or {}
    assert secret.get("secretName") == SECRET_NAME, f"{label}: projects {SECRET_NAME}"
    # Only the key the env reads: the release's previous-clusters.json backup
    # (written while a release stages the registry) never enters a Pod.
    assert secret.get("items") == [{"key": SECRET_KEY, "path": SECRET_KEY}], label
    # The env already requires the Secret and the key; so does the mount.
    assert secret.get("optional") is not True, f"{label}: the mount is required"

    container = spec["containers"][0]
    mounts = [m for m in container.get("volumeMounts", []) if m["name"] == VOLUME_NAME]
    assert len(mounts) == 1, f"{label}: the container mounts {VOLUME_NAME}"
    mount = mounts[0]
    assert mount.get("mountPath") == REGIONAL_CLUSTERS_SECRET_DIR, label
    assert mount.get("readOnly") is True, f"{label}: the Secret mount is read-only"
    assert "subPath" not in mount and "subPathExpr" not in mount, (
        f"{label}: kubelet never refreshes a subPath mount"
    )
    env = {item["name"]: item for item in container.get("env", [])}
    assert env[ENV_NAME]["valueFrom"]["secretKeyRef"] == {
        "name": SECRET_NAME,
        "key": SECRET_KEY,
    }, f"{label}: the start-up copy stays for settings and the bootstrap seed"


def test_the_runtime_reads_the_file_the_volume_projects() -> None:
    assert (
        REGIONAL_CLUSTERS_SECRET_FILE == f"{REGIONAL_CLUSTERS_SECRET_DIR}/{SECRET_KEY}"
    )


def test_the_regional_patch_mounts_the_registry_secret() -> None:
    (patch,) = [d for d in yaml.safe_load_all(PATCH.read_text(encoding="utf-8")) if d]
    _assert_secret_file(patch["spec"]["template"]["spec"], PATCH.name)


@pytest.mark.parametrize("name", ROLE_MANIFESTS)
def test_every_rendered_role_mounts_the_registry_secret(name: str) -> None:
    """The regional deploy applies regional/generated/; re-render after editing
    the patch (deploy/control-plane/tools/render-control-plane-role-split.sh)."""

    _assert_secret_file(_pod_spec(GENERATED / name), name)


def test_the_factory_hands_the_registry_runtime_the_mounted_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    item = registration("cluster-a", TOKEN_A).model_dump(mode="json")
    for field in ("token_sha256", "created_at", "updated_at", "lifecycle_state"):
        item.pop(field)
    secret = tmp_path / SECRET_KEY
    secret.write_text(json.dumps([{**item, "token": TOKEN_A}]), encoding="utf-8")
    monkeypatch.setattr(factory_module, "REGIONAL_CLUSTERS_SECRET_FILE", str(secret))
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context = ApplicationContext(store=store, execution_token="t" * 32)
    context.regional_mode = True

    runtime = create_app(context).state.regional_registry_runtime

    status = runtime.status()
    assert status["secret_source"] == SECRET_SOURCE_MOUNTED, status
    assert status["secret_drift"] is False, status
    assert status["secret_config_sha256"] == status["durable_config_sha256"]
