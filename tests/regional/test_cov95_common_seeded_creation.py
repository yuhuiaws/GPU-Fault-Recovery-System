from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import seeded_command_fixture as seeded
from tests.regional._cov95_cap_cases import Clock
from tests.regional._cov95_common_seeded import (
    IMAGE,
    RUN_ID,
    ResourceAPI,
    metadata,
    probe,
)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ResourceAPI:
    value = ResourceAPI()
    monkeypatch.setattr(seeded, "dataplane", value)
    monkeypatch.setattr(seeded, "NAMESPACE", "unit-namespace")
    monkeypatch.setattr(seeded, "CONTROL_NAMESPACE", "unit-control")
    monkeypatch.setattr(seeded, "DATAPLANE_CONTEXT", "unit-context")

    def identity(**kwargs: Any) -> dict[str, str]:
        assert kwargs == {"require_dataplane_deployment": True}
        return {
            "executor_artifact_sha256": "a" * 64,
            "executor_compatibility_digest": "b" * 64,
        }

    monkeypatch.setattr(seeded, "executor_identity", identity)
    monkeypatch.setattr(seeded, "time", Clock())
    return value


@pytest.mark.parametrize(
    "updates,problem",
    [
        ({"run_prefix": ""}, "plain identifier"),
        ({"run_prefix": "bad%"}, "plain identifier"),
        ({"pod_deadline_seconds": 59}, "outside"),
        ({"pod_deadline_seconds": 7201}, "outside"),
    ],
)
def test_probe_requires_bounded_owned_inputs(
    tmp_path: Path, updates: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        probe(tmp_path, **updates)


def test_missing_script_or_environment_stops_before_any_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: ResourceAPI
) -> None:
    with pytest.raises(ValueError, match="script does not exist"):
        probe(tmp_path, script=tmp_path / "absent.py")
    monkeypatch.setattr(seeded, "DATAPLANE_CONTEXT", "")
    with pytest.raises(seeded.SeededCommandError, match="CONTEXT"):
        seeded.require_environment()
    assert api.calls == []


def test_probe_creation_publishes_scoped_inputs_and_records_uid_before_readiness(
    tmp_path: Path, api: ResourceAPI
) -> None:
    selected = probe(tmp_path)
    api.file_responses = ["", "present"]
    assert seeded.create_probe_pod(selected, tmp_path, run_id=RUN_ID) == {"ready": True}
    assert [item["kind"] for item in api.created] == ["ConfigMap", "Pod"]
    configmap, pod = api.created
    assert set(configmap["data"]) == {selected.script.name}
    assert configmap["metadata"]["namespace"] == "unit-namespace"
    assert pod["spec"]["activeDeadlineSeconds"] == 900
    assert pod["spec"]["restartPolicy"] == "Never"
    container = pod["spec"]["containers"][0]
    assert container["image"] == IMAGE
    assert container["command"] == [
        "/opt/gpu-fault/executor/bin/python",
        f"/scripts/{selected.script.name}",
    ]
    environment = {item["name"]: item for item in container["env"]}
    assert environment["EXECUTOR_ARTIFACT_SHA256"]["value"] == "a" * 64
    assert environment["EXECUTOR_COMPATIBILITY_DIGEST"]["value"] == "b" * 64
    assert environment["CONTROL_PLANE_URL"]["valueFrom"] == {
        "secretKeyRef": {"name": seeded.CONNECTION_SECRET, "key": "control-plane-url"}
    }
    assert "GPU_FAULT_EXECUTION_TOKEN" not in environment
    assert seeded.probe_resource_uid(tmp_path, "pod", selected.pod, RUN_ID) == "uid-pod"
    assert (
        seeded.probe_resource_uid(tmp_path, "configmap", selected.configmap, RUN_ID)
        == "uid-configmap"
    )
    assert json.loads((tmp_path / "probe-ready.json").read_text()) == {"ready": True}


@pytest.mark.parametrize("kind", ["configmap", "pod"])
def test_create_ack_loss_is_recorded_without_replaying_the_create(
    tmp_path: Path, api: ResourceAPI, kind: str
) -> None:
    selected = probe(tmp_path)
    api.create_error_kind = kind
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        seeded.create_probe_pod(selected, tmp_path, run_id=RUN_ID)
    assert [item["kind"].lower() for item in api.created] == (
        ["configmap"] if kind == "configmap" else ["configmap", "pod"]
    )
    name = selected.configmap if kind == "configmap" else selected.pod
    assert seeded.probe_resource_uid(tmp_path, kind, name, RUN_ID) == f"uid-{kind}"
    assert not (tmp_path / "probe-ready.json").exists(), (
        "failed create claimed readiness"
    )


@pytest.mark.parametrize("kind", ["configmap", "pod"])
def test_invalid_create_receipt_does_not_authorize_readiness(
    tmp_path: Path, api: ResourceAPI, kind: str
) -> None:
    selected = probe(tmp_path)
    api.invalid_receipt_kind = kind
    with pytest.raises(seeded.SeededCommandError, match="identity was not confirmed"):
        seeded.create_probe_pod(selected, tmp_path, run_id=RUN_ID)
    name = selected.configmap if kind == "configmap" else selected.pod
    assert seeded.probe_resource_uid(tmp_path, kind, name, RUN_ID) is None
    assert not any(args[0] == "wait" for args, _kwargs in api.calls), (
        "a malformed create receipt reached the readiness barrier"
    )


def test_ready_receipt_cannot_substitute_for_the_created_pod_uid(
    tmp_path: Path, api: ResourceAPI
) -> None:
    selected = probe(tmp_path)
    api.replace_at_ready = True
    with pytest.raises(seeded.SeededCommandError, match="replaced before readiness"):
        seeded.create_probe_pod(selected, tmp_path, run_id=RUN_ID)
    assert seeded.probe_resource_uid(tmp_path, "pod", selected.pod, RUN_ID) == "uid-pod"
    assert not (tmp_path / "probe-ready.json").exists(), "replacement Pod was accepted"


@pytest.mark.parametrize("fault", ["", "missing", "foreign", "replaced"])
def test_registry_creation_requires_the_owned_token_secret_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: ResourceAPI, fault: str
) -> None:
    calls = []
    item = metadata("secret")
    if fault == "missing":
        item.pop("resourceVersion")
    elif fault == "foreign":
        item["labels"][seeded.RUN_LABEL] = "another-run"
    elif fault == "replaced":
        item["uid"] = "replacement"
    api.resources[("secret", seeded.TOKEN_SECRET)] = item

    def register(count: int, directory: Path, **kwargs: Any) -> None:
        calls.append((count, directory, kwargs))
        seeded.write_json(
            directory / "registry-token-proof.json",
            {"run_id": RUN_ID, "uid": "uid-secret"},
        )

    monkeypatch.setattr(seeded, "register", register)
    if fault:
        with pytest.raises(
            seeded.SeededCommandError, match="identity|another run|replaced"
        ):
            seeded.register_synthetic_cluster(tmp_path, RUN_ID)
    else:
        seeded.register_synthetic_cluster(tmp_path, RUN_ID)
    assert len(calls) == 1
    assert calls[0][:2] == (1, tmp_path)
    options = calls[0][2]
    assert options["run_id"] == RUN_ID
    assert options["allow_live_registry"] is True
    assert options["live_registry_confirmation"] == seeded.LIVE_REGISTRY_CONFIRMATION
    assert (
        1700
        < (options["expires_at"] - datetime.now(timezone.utc)).total_seconds()
        <= 1800
    )
