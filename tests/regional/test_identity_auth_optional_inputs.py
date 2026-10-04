"""Identity handlers with their optional inputs left out or refused.

AUTH-013 without ``--node``/``--host-probe-image`` must mark the expiry alert
unevaluated (and so failed), skip the probe cleanup check and write nothing
without a case directory; AUTH-015 preparation must refuse an unsafe
master-reference scan, a node-action Secret that does not cover the GPU node
set, and must otherwise return the context the custody phase consumes; the
node-key restore must rewrite a Secret only when the drift is its own.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import identity_acceptance_common as common

TLS_RESULT: dict[str, Any] = {
    "host": "control.example",
    "sans": ["control.example"],
    "ca_exists": True,
    "ssl_cert_file_set": False,
    "requests_ca_bundle_set": False,
    "hostname_in_san": True,
    "default_handshake_ok": True,
    "empty_ca_rejected": True,
    "wrong_hostname_rejected": True,
    "remaining_days": 60,
    "not_after": "2026-12-12T00:00:00Z",
}


def _tls_site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        namespace="gpu-system",
        gpu_kubeconfig=tmp_path / "kubeconfig",
        config={"health": {"certificate_min_validity_days": 30}},
        any_executor_pod=lambda _target: "executor",
        pod_json=lambda *_args: dict(TLS_RESULT),
    )


def test_auth013_without_a_node_cannot_claim_the_expiry_alert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_probe(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no host probe may be created without a node")

    monkeypatch.setattr(auth, "HostProbeFixture", no_probe)

    result = auth.run_auth013(
        _tls_site(tmp_path), SimpleNamespace(context="gpu-context"), case_dir=tmp_path
    )

    assert result["verdict"] == "FAIL"
    assert result["checks"]["expiry_threshold_configured"] is False
    assert "alert_probe_removed" not in result["checks"]
    assert set(result["not_evaluated"]) == {"expiry_threshold_configured"}
    assert (
        "no --node/--host-probe-image"
        in result["not_evaluated"]["expiry_threshold_configured"]
    )
    assert result["expiry_alert"] is None
    assert result["alert_probe_residuals"] == {}
    details = json.loads((tmp_path / "auth013-details.json").read_text())
    assert details["verdict"] == "FAIL"


def test_auth013_without_a_case_dir_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(auth, "HostProbeFixture", lambda _settings: None)

    result = auth.run_auth013(
        _tls_site(tmp_path),
        SimpleNamespace(context="gpu-context"),
        case_dir=None,  # type: ignore[arg-type]
        node="node-a",
    )

    assert result["checks"]["expiry_threshold_configured"] is False
    assert list(tmp_path.iterdir()) == []


def _node_keys(*names: str) -> dict[str, str]:
    return {name: base64.b64encode((name * 8).encode()).decode() for name in names}


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    scan: dict[str, Any],
    gpu_nodes: tuple[str, ...],
) -> tuple[Any, list[tuple[str, ...]]]:
    master = tmp_path / "unit-master"
    master.write_text("m" * 64, encoding="ascii")
    master.chmod(0o600)
    data = _node_keys("node-a", "node-b")
    monkeypatch.setattr(
        auth, "secret_document", lambda *_args: {"data": copy.deepcopy(data)}
    )
    monkeypatch.setattr(auth, "gpu_master_reference_scan", lambda *_args: scan)
    probes: list[Any] = []
    monkeypatch.setattr(auth, "HostProbeSettings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        auth, "HostProbeFixture", lambda settings: probes.append(settings) or settings
    )
    python_calls: list[tuple[str, ...]] = []

    def cpu_python(*arguments: str) -> dict[str, Any]:
        python_calls.append(arguments)
        return {"agents": list(arguments[2:])}

    region = SimpleNamespace(
        gpu_nodes=lambda: [{"name": name} for name in gpu_nodes], cpu_python=cpu_python
    )
    site = SimpleNamespace(
        regional=lambda *_args: region,
        gpu_kubeconfig=tmp_path / "gpu",
        namespace="gpu-system",
    )
    context = auth.prepare_auth015(
        site,
        SimpleNamespace(context="gpu-context", cluster_id="cluster-a"),
        nodes=("node-a", "node-b"),
        fleet_master_file=master,
        host_probe_image="unit@sha256:" + "a" * 64,
        case_dir=tmp_path,
    )
    return context, python_calls


@pytest.mark.parametrize(
    "scan",
    [
        {"installer_resources": [], "hits": []},
        {"installer_resources": ["Job/installer"], "hits": ["Pod/leak"]},
    ],
)
def test_auth015_prepare_refuses_an_unsafe_master_reference_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scan: dict[str, Any]
) -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="missing or unsafe"):
        _prepare(monkeypatch, tmp_path, scan=scan, gpu_nodes=("node-a", "node-b"))


def test_auth015_prepare_refuses_a_secret_that_does_not_cover_the_node_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="exactly the GPU node"):
        _prepare(
            monkeypatch,
            tmp_path,
            scan={"installer_resources": ["Job/installer"], "hits": []},
            gpu_nodes=("node-a", "node-b", "node-c"),
        )


def test_auth015_prepare_returns_the_context_the_custody_phase_consumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, python_calls = _prepare(
        monkeypatch,
        tmp_path,
        scan={"installer_resources": ["Job/installer"], "hits": []},
        gpu_nodes=("node-b", "node-a"),
    )

    assert context.gpu_node_names == ["node-a", "node-b"]
    assert sorted(context.before_keys) == ["node-a", "node-b"]
    assert context.before_keys == context.before_cpu_keys
    assert context.before_agents == {"agents": ["node-a", "node-b"]}
    assert python_calls[0][1:] == ("cluster-a", "node-a", "node-b")
    assert [probe["node"] for probe in context.probes] == ["node-a", "node-b"]
    assert all(
        probe["case_id"] == "GF-REGIONAL-AUTH-015" for probe in context.probes
    ), context.probes
    assert context.master_sha256 == common.secret_digest("m" * 64)


def test_node_key_restore_rewrites_only_its_own_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        "metadata": {
            "name": "keys",
            "uid": "uid",
            "namespace": "training",
            "resourceVersion": "7",
        },
        "data": {"node": "old"},
    }
    rotated = copy.deepcopy(original)
    rotated["data"] = {"node": "rotated-by-the-case"}
    monkeypatch.setattr(auth, "secret_document", lambda *_args: copy.deepcopy(rotated))
    patches: list[tuple[str, tuple[str, ...], list[dict[str, Any]]]] = []

    def kubectl(plane: str, *arguments: str, input_text: str) -> str:
        patches.append((plane, arguments, json.loads(input_text)))
        return ""

    site = SimpleNamespace(regional=lambda _target: SimpleNamespace(kubectl=kubectl))

    auth.restore_secret(site, "gpu", None, original, expected=rotated)

    assert len(patches) == 1
    plane, arguments, operations = patches[0]
    assert plane == "gpu"
    assert arguments[:3] == ("patch", "secret", "keys")
    assert operations[-1] == {
        "op": "replace",
        "path": "/data",
        "value": {"node": "old"},
    }
    assert {
        "op": "test",
        "path": "/metadata/resourceVersion",
        "value": "7",
    } in operations
    assert {
        "op": "test",
        "path": "/data",
        "value": {"node": "rotated-by-the-case"},
    } in operations
