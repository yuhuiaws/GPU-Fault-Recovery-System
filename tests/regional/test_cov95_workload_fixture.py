from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import managed_workload_fixture as managed
from tests.regional._cov95_identity_support import Clock
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_managed_workload_fixture import harness


@pytest.mark.parametrize(
    "field", ["kind", "name", "uid", "namespace", "resourceVersion"]
)
def test_resource_identity_rejects_unbound_incarnations(field: str) -> None:
    document = {
        "kind": "Pod",
        "metadata": {
            "name": "unit",
            "uid": "uid",
            "namespace": "training",
            "resourceVersion": "1",
        },
    }
    if field == "kind":
        document["kind"] = "Node"
    else:
        document["metadata"][field] = ""
    with pytest.raises(managed.RegionalFixtureError, match="incomplete or foreign"):
        managed.resource_identity(
            SimpleNamespace(settings=SimpleNamespace(namespace="training")), document
        )


@pytest.mark.parametrize("defect", ["list", "name", "kind", "absent"])
def test_resource_read_refuses_nonobjects_and_foreign_targets(defect: str) -> None:
    document: Any = {
        "kind": "Pod",
        "metadata": {
            "name": "unit",
            "uid": "uid",
            "namespace": "training",
            "resourceVersion": "1",
        },
    }
    if defect == "list":
        document = []
    elif defect == "name":
        document["metadata"]["name"] = "foreign"
    elif defect == "kind":
        document["kind"] = "Job"
    regional = SimpleNamespace(
        settings=SimpleNamespace(namespace="training"),
        kubectl=lambda *args, **kwargs: ""
        if defect == "absent"
        else json.dumps(document),
    )
    if defect == "absent":
        assert managed.read_resource(regional, "pod", "unit") is None
    else:
        with pytest.raises(
            managed.RegionalFixtureError, match="not an object|another identity"
        ):
            managed.read_resource(regional, "pod", "unit")


@pytest.mark.parametrize(
    "defect", ["none", "delete-ack", "failed-delete", "replacement", "timeout"]
)
def test_uid_delete_requires_a_confirmed_absent_original(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    document = {
        "kind": "Job",
        "metadata": {
            "name": "unit",
            "uid": "owned-uid",
            "namespace": "training",
            "resourceVersion": "7",
        },
    }
    clock = Clock(step=50)
    monkeypatch.setattr(managed, "time", clock)
    reads = []
    deletes = []

    def kubectl(*args: str, **kwargs: Any) -> str:
        if args[1] == "delete":
            deletes.append((args, json.loads(kwargs["input_text"])))
            if defect in {"delete-ack", "failed-delete"}:
                raise managed.RegionalFixtureError("synthetic DELETE failed")
            return ""
        assert args[1:4] == ("get", "job", "unit")
        reads.append(None)
        if defect == "delete-ack" or defect == "none" and len(reads) > 1:
            return ""
        current = copy.deepcopy(document)
        if defect == "replacement":
            current["metadata"]["uid"] = "new-uid"
        return json.dumps(current)

    regional = SimpleNamespace(
        settings=SimpleNamespace(namespace="training"), kubectl=kubectl
    )
    if defect in {"none", "delete-ack"}:
        managed.delete_resource(regional, document)
    else:
        with pytest.raises(
            managed.RegionalFixtureError, match="failed|replaced|not confirmed"
        ):
            managed.delete_resource(regional, document)
    assert len(deletes) == 1
    assert deletes[0][0][3] == "/apis/batch/v1/namespaces/training/jobs/unit"
    assert deletes[0][1] == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": "owned-uid", "resourceVersion": "7"},
        "propagationPolicy": "Foreground",
    }
    assert reads, "DELETE acknowledgement cannot replace an absence readback"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"manifest": Path("/unit-nonexistent-manifest")}, "manifest"),
        ({"site_file": Path("/unit-nonexistent-site")}, "site file"),
        ({"job_id": ""}, "identity"),
        ({"attempt_id": ""}, "identity"),
        ({"restart_budget": -1}, "negative"),
        ({"expected_pods": 0}, "positive"),
        ({"expected_gpu_count": 0}, "positive"),
    ],
)
def test_workload_settings_reject_unsafe_expectations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: dict[str, Any],
    message: str,
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        replace(fixture.settings, **change)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"kind": "Job", "metadata": {}},
        {"kind": "Pod", "metadata": {"name": "unit"}},
        {"kind": "Job", "metadata": {"name": "unit", "namespace": "foreign"}},
    ],
)
def test_managed_fixture_constructor_refuses_wrong_kind_or_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, document: Any
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(document), encoding="ascii")
    with pytest.raises(ValueError, match="mapping|kind/name|namespace"):
        managed.ManagedWorkloadFixture(
            fixture.regional, replace(fixture.settings, manifest=manifest)
        )


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"kind": "Job"},
        {"kind": "Deployment", "spec": {}},
        {"kind": "Job", "spec": {}},
        {"kind": "Job", "spec": {"template": {}}},
        {"kind": "PyTorchJob", "spec": {"pytorchReplicaSpecs": {"Master": None}}},
    ],
)
def test_node_pinning_refuses_malformed_templates(
    tmp_path: Path, document: Any
) -> None:
    source, destination = tmp_path / "source.yaml", tmp_path / "rendered.yaml"
    source.write_text(yaml.safe_dump(document), encoding="ascii")
    with pytest.raises(ValueError, match="manifest|supports|templates|template"):
        managed.render_node_pinned_manifest(source, destination, node="node-a")
    assert not destination.exists(), "invalid templates must not produce a manifest"


@pytest.mark.parametrize("kind", ["Job", "PyTorchJob"])
def test_node_pinning_uses_structured_templates_and_keeps_source_unchanged(
    tmp_path: Path, kind: str
) -> None:
    template = {
        "metadata": {"labels": {"owned": "test"}},
        "spec": {"containers": [{"name": "training", "image": "example-image"}]},
    }
    document = {
        "kind": kind,
        "spec": {"template": template}
        if kind == "Job"
        else {
            "pytorchReplicaSpecs": {
                "Master": {"template": template},
                "Worker": {"template": copy.deepcopy(template)},
                "Ignored": None,
            }
        },
    }
    source, destination = (
        tmp_path / "source.yaml",
        tmp_path / "nested" / "rendered.yaml",
    )
    source.write_text(yaml.safe_dump(document), encoding="ascii")
    assert (
        managed.render_node_pinned_manifest(source, destination, node="node-a")
        == destination
    )
    rendered = yaml.safe_load(destination.read_text())
    templates = (
        [rendered["spec"]["template"]]
        if kind == "Job"
        else [
            value["template"]
            for value in rendered["spec"]["pytorchReplicaSpecs"].values()
            if isinstance(value, dict)
        ]
    )
    assert all(value["spec"]["nodeName"] == "node-a" for value in templates), (
        "every workload Pod template must bind the selected node"
    )
    assert yaml.safe_load(source.read_text()) == document
    assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("text", "healthy"),
    [
        ("", False),
        ("HEARTBEAT all_reduce=1e+", False),
        ("SUCCESS rank=0/2", False),
        ("SUCCESS rank=0/1", True),
        ("HEARTBEAT all_reduce=1", True),
    ],
)
def test_heartbeat_requires_the_expected_collective(text: str, healthy: bool) -> None:
    assert managed.heartbeat_healthy(text, world_size=1) is healthy
    with pytest.raises(ValueError, match="positive"):
        managed.expected_all_reduce(0)


@pytest.mark.parametrize("items", [None, "unknown", [None]])
def test_workload_inventory_never_treats_unknown_as_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, items: Any
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture.regional,
        "kubectl",
        lambda *args, **kwargs: json.dumps({"items": items}),
    )
    with pytest.raises(managed.RegionalFixtureError, match="inventory"):
        fixture.inventory()


@pytest.mark.parametrize("rendered", ["[]", "kind: Other\nmetadata: {}\n"])
def test_submit_refuses_an_unbound_render_before_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rendered: str
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    with pytest.raises(
        managed.RegionalFixtureError, match="rendered workload identity"
    ):
        fixture.submit_rendered(rendered)
    assert api.created == [] and fixture.submission_started is False


def test_managed_submission_is_create_only_and_cannot_be_repeated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    value = yaml.safe_load(rendered)
    value["spec"]["testMetadata"] = [{"metadata": {"labels": {"retained": "yes"}}}]
    fixture.submit_rendered(yaml.safe_dump(value))
    assert api.created[0]["spec"]["testMetadata"][0]["metadata"]["labels"] == {
        "retained": "yes",
        managed.OWNER_LABEL: fixture.owner,
    }
    with pytest.raises(managed.RegionalFixtureError, match="already started"):
        fixture.submit_rendered(rendered)
    assert len(api.created) == 1


def test_create_without_an_object_receipt_keeps_cleanup_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    original = api.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = original(command, **kwargs)
        if "get" in command and "-l" not in command:
            result.stdout = ""
        return result

    monkeypatch.setattr(fixture.regional, "run", run)
    with pytest.raises(
        managed.RegionalFixtureError, match="created managed workload is missing"
    ):
        fixture.submit_rendered(rendered)
    assert fixture.submission_started is True
    assert len(api.created) == 1 and api.deletes == []
    monkeypatch.setattr(fixture.regional, "run", original)
    fixture.delete()
    assert api.objects == {}


@pytest.mark.parametrize("annotation", [None, "false"])
def test_auto_resume_annotation_changes_are_uid_fenced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, annotation: str | None
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    source = api.objects[(fixture.resource, fixture.name)]
    source["metadata"]["annotations"] = {
        "retained": "yes",
        "sagemaker.amazonaws.com/enable-job-auto-resume": "false",
    }
    original = fixture.regional.kubectl
    patches = []

    def kubectl(*args: str, **kwargs: Any) -> str:
        if args[1] == "patch":
            patches.append(json.loads(kwargs["input_text"]))
            return ""
        return original(*args, **kwargs)

    monkeypatch.setattr(fixture.regional, "kubectl", kubectl)
    fixture.annotate_auto_resume(annotation)
    assert patches[0][:2] == [
        {"op": "test", "path": "/metadata/uid", "value": source["metadata"]["uid"]},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
    ]
    expected = {"retained": "yes"}
    if annotation is not None:
        expected["sagemaker.amazonaws.com/enable-job-auto-resume"] = annotation
    assert patches[0][2] == {
        "op": "add",
        "path": "/metadata/annotations",
        "value": expected,
    }


def test_workload_object_protocol_requires_an_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    monkeypatch.setattr(fixture.regional, "kubectl", lambda *args, **kwargs: "[]")
    with pytest.raises(managed.RegionalFixtureError, match="JSON object"):
        fixture.workload()


def ready_snapshot(uid: str = "new") -> dict[str, Any]:
    return {
        "pods": [
            {
                "name": "training",
                "uid": uid,
                "node": "node-a",
                "phase": "Running",
                "ready": True,
            }
        ],
        "heartbeat_logs": {"training": "HEARTBEAT all_reduce=1.0"},
    }


@pytest.mark.parametrize("healthy", [False, True])
def test_wait_running_retries_unknown_state_and_requires_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, healthy: bool
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    clock = Clock()
    monkeypatch.setattr(managed, "time", clock)
    calls = []

    def snapshot() -> dict[str, Any]:
        calls.append(None)
        if len(calls) == 1:
            raise managed.RegionalFixtureError("temporary read failure")
        result = ready_snapshot()
        if len(calls) == 2 or not healthy:
            result["heartbeat_logs"] = {}
        return result

    monkeypatch.setattr(fixture, "snapshot", snapshot)
    if healthy:
        assert fixture.wait_running(timeout_seconds=50) == ready_snapshot()
        assert len(calls) == 3
    else:
        with pytest.raises(
            managed.RegionalFixtureError, match="did not become healthy"
        ):
            fixture.wait_running(timeout_seconds=30)


@pytest.mark.parametrize(
    "defect", ["none", "not-replaced", "no-heartbeat", "old-reappears"]
)
def test_restart_waiter_checks_replacement_before_logs_and_rejects_old_uid_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    clock = Clock()
    monkeypatch.setattr(managed, "time", clock)
    events = []
    reads, snapshots = [], []

    def pods() -> list[dict[str, Any]]:
        events.append("pods")
        reads.append(None)
        if len(reads) == 1:
            raise managed.RegionalFixtureError("temporary Pod read failure")
        return ready_snapshot(
            "old" if len(reads) == 2 or defect == "not-replaced" else "new"
        )["pods"]

    def snapshot() -> dict[str, Any]:
        events.append("snapshot")
        snapshots.append(None)
        if len(snapshots) == 1:
            raise managed.RegionalFixtureError("temporary snapshot failure")
        result = ready_snapshot("old" if defect == "old-reappears" else "new")
        if len(snapshots) == 2 or defect == "no-heartbeat":
            result["heartbeat_logs"] = {}
        return result

    monkeypatch.setattr(fixture, "pods", pods)
    monkeypatch.setattr(fixture, "snapshot", snapshot)
    if defect == "none":
        assert (
            fixture.wait_restarted({"old"}, timeout_seconds=50, poll_seconds=1)
            == ready_snapshot()
        )
        assert events == ["pods"] * 3 + ["snapshot"] * 3
    else:
        message = {
            "not-replaced": "were not replaced",
            "no-heartbeat": "did not heartbeat",
            "old-reappears": "reappeared",
        }[defect]
        with pytest.raises(managed.RegionalFixtureError, match=message):
            fixture.wait_restarted({"old"}, timeout_seconds=20, poll_seconds=1)
        if defect == "not-replaced":
            assert snapshots == [], "logs must not be read before replacement is proven"


@pytest.mark.parametrize("changed", [False, True])
def test_unchanged_uid_window_checks_every_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: bool
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    clock = Clock()
    monkeypatch.setattr(managed, "time", clock)
    calls = []

    def snapshot() -> dict[str, Any]:
        calls.append(None)
        return ready_snapshot("changed" if changed and len(calls) > 1 else "new")

    monkeypatch.setattr(fixture, "snapshot", snapshot)
    if changed:
        with pytest.raises(managed.RegionalFixtureError, match="UIDs changed"):
            fixture.wait_pod_uids_unchanged({"new"}, timeout_seconds=15)
    else:
        assert (
            fixture.wait_pod_uids_unchanged({"new"}, timeout_seconds=15)
            == ready_snapshot()
        )
    assert len(calls) > 1


def test_snapshot_preserves_owned_uid_and_bounded_training_log_tails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    api.add(
        {
            "kind": "Pod",
            "metadata": {
                "name": "training",
                "labels": api.created[0]["metadata"]["labels"],
            },
            "spec": {"nodeName": "node-a", "containers": [{"name": "training"}]},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"name": "training", "ready": True}],
            },
        }
    )
    original = api.run
    lines = [
        "noise",
        *[f"HEARTBEAT step={index} all_reduce=1" for index in range(25)],
        *[f"rank=0 step={index} loss=1" for index in range(25)],
    ]

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, "\n".join(lines), "")
        return original(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)
    result = fixture.snapshot()
    assert (
        result["workload"]["uid"]
        == api.objects[(fixture.resource, fixture.name)]["metadata"]["uid"]
    )
    assert result["pods"][0]["ready"] is True
    logs = result["heartbeat_logs"]["training"].splitlines()
    assert logs == lines[6:26] + lines[31:]
    assert fixture.logs_healthy(result["pods"], result["heartbeat_logs"]) is True


@pytest.mark.parametrize("nodes", [[], ["node-a", "node-a"], [""]])
def test_prewarm_refuses_an_unbounded_or_duplicate_target_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nodes: list[str]
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="unit", run_id="unit"
    )
    with pytest.raises(managed.RegionalFixtureError, match="nonempty unique"):
        prewarm.create(nodes)
    assert api.created == []


@pytest.mark.parametrize("cached", [False, True])
def test_prewarm_reuses_cached_digests_and_never_recreates_owned_pods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="unit", run_id="unit"
    )
    original = fixture.regional.kubectl

    def kubectl(*args: str, **kwargs: Any) -> str:
        if args[:3] == ("gpu", "get", "node"):
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "node-a"},
                            "status": {
                                "images": [
                                    {
                                        "names": [
                                            prewarm.image
                                            if cached
                                            else "other@sha256:abc"
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(fixture.regional, "kubectl", kubectl)
    assert prewarm.create(["node-a"]) == {
        "skipped": ["node-a"] if cached else [],
        "created": [] if cached else ["node-a"],
    }
    if not cached:
        with pytest.raises(managed.RegionalFixtureError, match="already created"):
            prewarm.create(["node-a"])
    assert len(api.created) == int(not cached)
    prewarm.cleanup()
    assert api.objects == {}


def test_prewarm_refuses_an_incomplete_node_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="unit", run_id="unit"
    )
    monkeypatch.setattr(fixture.regional, "kubectl", lambda *args, **kwargs: "{}")
    with pytest.raises(managed.RegionalFixtureError, match="inventory is incomplete"):
        prewarm.cached_nodes()
