from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import identity_acceptance_common as common
from tests.regional._regional_support import TOKEN_A, registration


@pytest.mark.parametrize("drift", ["entries", "generation", "none"])
def test_registry_write_cas_binds_the_reviewed_snapshot(
    drift: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = common.IdentitySite.__new__(common.IdentitySite)
    original = [registration("cluster-a", TOKEN_A).model_dump(mode="json")]
    current = copy.deepcopy(original)
    if drift == "entries":
        current[0]["enabled"] = False
    posted = []
    monkeypatch.setattr(site, "registry_generation", lambda: 7)
    monkeypatch.setattr(
        site,
        "api_pod_json",
        lambda *args: {
            "generation": 8 if drift == "generation" else 7,
            "registrations": current,
        },
    )
    monkeypatch.setattr(
        site, "registry_api", lambda *args: posted.append(args) or {"status": 200}
    )
    monkeypatch.setattr(site, "wait_registry_ready", lambda *args: {})
    if drift != "none":
        with pytest.raises(common.IdentityAcceptanceError, match="changed outside"):
            site.write_registry(original, expected_entries=original)
        assert posted == []
    else:
        site.write_registry(original, expected_entries=original)
        assert posted[0][2]["expected_generation"] == 7


def test_registry_restore_refuses_concurrent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = common.IdentitySite.__new__(common.IdentitySite)
    original = [registration("cluster-a", TOKEN_A).model_dump(mode="json")]
    site.last_registry_payload = copy.deepcopy(original)
    site.last_registry_payload[0]["enabled"] = False
    current = copy.deepcopy(site.last_registry_payload)
    current[0]["allowed_namespaces"] = ["peer-change"]
    monkeypatch.setattr(site, "registry", lambda: current)
    writes = []
    monkeypatch.setattr(
        site, "write_registry", lambda *args, **kwargs: writes.append(args)
    )
    with pytest.raises(common.IdentityAcceptanceError, match="concurrent drift"):
        site.restore_registry(original)
    assert writes == []


@pytest.mark.parametrize("drift", ["uid", "token", "none"])
def test_cluster_token_write_is_uid_and_content_fenced(drift: str) -> None:
    encoded = base64.b64encode(b"old-test-credential").decode()
    patches = []

    def gpu(target: Any, *args: str, **kwargs: Any) -> str:
        if args[0] == "get":
            return json.dumps(
                {
                    "metadata": {
                        "uid": "foreign" if drift == "uid" else "original",
                        "resourceVersion": "9",
                    },
                    "data": {
                        "cluster-token": base64.b64encode(b"changed").decode()
                        if drift == "token"
                        else encoded
                    },
                }
            )
        patches.append(json.loads(kwargs["input_text"]))
        return ""

    site = SimpleNamespace(gpu=gpu)
    if drift != "none":
        with pytest.raises(common.IdentityAcceptanceError):
            common.write_cluster_token(
                site,
                None,
                "new-test-credential",
                expected_token="old-test-credential",
                expected_uid="original",
            )
        assert patches == []
    else:
        common.write_cluster_token(
            site,
            None,
            "new-test-credential",
            expected_token="old-test-credential",
            expected_uid="original",
        )
        assert patches[0][:2] == [
            {"op": "test", "path": "/metadata/uid", "value": "original"},
            {"op": "test", "path": "/metadata/resourceVersion", "value": "9"},
        ]
        assert patches[0][2] == {
            "op": "test",
            "path": "/data/cluster-token",
            "value": encoded,
        }


def test_rotation_entry_delegates_all_proofs_to_the_production_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.e2e.regional import auth016_lifecycle as lifecycle

    primary = object()
    site = SimpleNamespace(regional=lambda _target: primary)
    target = SimpleNamespace(cluster_id="cluster-a")
    identity = {"artifact": "a" * 64}
    calls = []
    expected = {"verdict": "FAIL", "checks": {"consumer_missing": True}}
    monkeypatch.setattr(auth, "executor_claim_identity", lambda *_args: identity)
    monkeypatch.setattr(
        auth,
        "direct_claim",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 403,
    )

    def production(selected_site, selected_target, *, case_dir, retired_probe):
        assert selected_site is site and selected_target is target
        assert case_dir == tmp_path
        assert retired_probe(TOKEN_A) == 403
        return expected

    monkeypatch.setattr(lifecycle, "run_rotation_acceptance", production)
    assert auth.run_auth016(site, target, case_dir=tmp_path) is expected
    assert calls == [((primary, target), {"token": TOKEN_A, "identity": identity})]


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "installer",
        "master-hit",
        "missing-scan",
        "tests",
        "rotation-ack",
        "cleanup",
    ],
)
def test_master_isolation_handler_stops_before_unsafe_rotation_and_fences_restore(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    master_file = tmp_path / "synthetic-master"
    master_file.write_text("synthetic-unit-value-" + "x" * 32)
    master_file.chmod(0o600)
    original = {
        plane: {
            "metadata": {
                "name": auth.NODE_ACTION_KEYS_SECRET,
                "namespace": "gpu-system",
                "uid": plane + "-uid",
                "resourceVersion": "1",
            },
            "data": {
                node: base64.b64encode((node * 8).encode()).decode()
                for node in ("node-a", "node-b")
            },
        }
        for plane in ("cpu", "gpu")
    }
    current = copy.deepcopy(original)
    events: list[str] = []
    ticks = [0]

    def cpu_python(*args: Any) -> dict[str, Any]:
        ticks[0] += 1
        return {
            "agents": {
                node: {
                    "generation": 1,
                    "lifecycle_state": "ACTIVE",
                    "node_action_key_version": 2,
                    "last_heartbeat_at": f"2026-09-12T00:00:0{ticks[0]}Z",
                }
                for node in ("node-a", "node-b")
            }
        }

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        assert args[:2] == ("patch", "secret")
        operations = json.loads(kwargs["input_text"])
        assert operations[0]["value"] == original[plane]["metadata"]["uid"]
        assert operations[1]["path"] == "/metadata/resourceVersion"
        assert operations[2]["value"] == current[plane]["data"]
        current[plane]["data"] = operations[3]["value"]
        events.append("restore-" + plane)
        return ""

    regional = SimpleNamespace(
        gpu_nodes=lambda: [{"name": "node-a"}, {"name": "node-b"}],
        cpu_python=cpu_python,
        kubectl=kubectl,
    )
    site = SimpleNamespace(
        regional=lambda target: regional,
        namespace="gpu-system",
        gpu_kubeconfig=tmp_path / "gpu",
        cpu_kubeconfig=tmp_path / "cpu",
    )
    target = SimpleNamespace(
        context="context-a", cluster_id="cluster-a", hyperpod_cluster_name="hyperpod-a"
    )

    class Probe:
        def __init__(self, settings: Any) -> None:
            assert settings.state_directory == tmp_path / "host-probes"
            self.settings = settings

        def create(self) -> None:
            events.append("create-" + self.settings.node)

        def execute(self, *args: str) -> dict[str, Any]:
            result = {
                "master_matches": ["synthetic-hit"] if defect == "master-hit" else [],
                "values_scanned": 10,
                "host_tmp_exists": True,
                "systemd_environment_exists": True,
            }
            if defect == "missing-scan":
                del result["master_matches"]
            return result

        def cleanup(self) -> dict[str, bool]:
            events.append("cleanup-" + self.settings.node)
            return {"pod": defect == "cleanup", "configmap": False}

    def provision(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        events.append("rotate")
        for document in current.values():
            document["data"]["node-a"] = base64.b64encode(
                b"new-synthetic-node-key"
            ).decode()
            document["metadata"]["resourceVersion"] = "2"
        return subprocess.CompletedProcess(
            command, 1 if defect == "rotation-ack" else 0, "", ""
        )

    monkeypatch.setattr(
        auth, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(auth, "HostProbeFixture", Probe)
    monkeypatch.setattr(
        auth,
        "secret_document",
        lambda _site, plane, *_args: copy.deepcopy(current[plane]),
    )
    monkeypatch.setattr(
        auth,
        "gpu_master_reference_scan",
        lambda *args: {
            "installer_resources": []
            if defect == "installer"
            else ["Job/test-installer"],
            "hits": [],
        },
    )
    monkeypatch.setattr(auth, "run", provision)
    monkeypatch.setattr(auth.time, "sleep", lambda seconds: None)
    kwargs = {
        "nodes": ("node-a", "node-b"),
        "fleet_master_file": master_file,
        "host_probe_image": "test@sha256:" + "a" * 64,
        "case_dir": tmp_path,
        "focused_tests": {"passed": defect != "tests"},
    }
    result = auth.run_auth015(site, target, **kwargs)
    assert result["verdict"] == "FAIL"
    assert result["requires_new_authorized_evidence"] is True
    assert set(result["not_evaluated"]) == {
        "installation_time_master_custody",
        "deployed_node_a_key_activation",
        "deployed_cross_node_command_and_result_signatures",
    }
    assert events == [], "snapshots must not authorize any probe, rotation or restore"
    assert all(
        current[plane]["data"] == original[plane]["data"] for plane in current
    ), "unproven installations must remain untouched"
