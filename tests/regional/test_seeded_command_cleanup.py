"""The synthetic-command fixture must clean only mutations it attempted."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import net_command_fixture as net
from scripts.e2e.regional import run_cmd018_open_sibling_hold as cmd018
from scripts.e2e.regional import run_net003_result_retry as net003
from scripts.e2e.regional import seeded_command_fixture as seeded


@pytest.mark.parametrize("fixture", [seeded, net])
def test_cleanup_before_mutation_performs_no_cluster_calls(
    fixture: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(seeded, "dataplane", lambda *a, **k: calls.append("gpu") or "")
    monkeypatch.setattr(seeded, "teardown", lambda **k: calls.append("broad teardown"))
    monkeypatch.setattr(seeded, "database_residuals", lambda p: {"total": 0})
    monkeypatch.setattr(seeded, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda p: {"count": 0})
    if fixture is net:
        monkeypatch.setattr(net, "dataplane", lambda *a, **k: calls.append("gpu") or "")
        monkeypatch.setattr(net, "database_residuals", lambda p: {"total": 0})
        monkeypatch.setattr(net, "registry_residuals", lambda: {"count": 0})
        monkeypatch.setattr(net, "kubernetes_residuals", lambda p: {"count": 0})
    result = {"verdict": "FAIL"}
    fixture.cleanup(
        SimpleNamespace(pod="pod", configmap="cm", run_prefix="review-"),
        tmp_path,
        "review-run",
        result,
        {},
        state={},
    )
    assert calls == []
    assert result["verdict"] == "FAIL"


def test_cleanup_stops_claimant_before_purge_and_never_calls_perf_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        seeded, "delete_owned_resource", lambda kind, *a, **k: calls.append(kind)
    )
    monkeypatch.setattr(
        seeded, "deregister_synthetic_cluster", lambda *a, **k: calls.append("registry")
    )
    monkeypatch.setattr(
        seeded,
        "teardown",
        lambda **k: pytest.fail("broad performance cleanup is never authorized here"),
    )
    monkeypatch.setattr(seeded, "database_residuals", lambda p: {"total": 0})
    monkeypatch.setattr(seeded, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda p: {"count": 0})

    def purge(seed: dict) -> dict:
        calls.append("purge")
        return {"remaining": [], "remaining_links": 0}

    result = {"verdict": "PASS"}
    seeded.cleanup(
        SimpleNamespace(pod="pod", configmap="cm", run_prefix="review-"),
        tmp_path,
        "review-run",
        result,
        {"command_id": "remote-review"},
        state={"registry_started": True, "probe_started": True},
        purge=purge,
    )
    assert calls[0:2] == ["pod", "purge"]
    assert "configmap" in calls and "secret" in calls and "registry" in calls
    assert result["verdict"] == "PASS"


def test_unconfirmed_claimant_stop_withholds_purge_but_cleans_independent_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def delete(kind: str, *args: Any, **kwargs: Any) -> None:
        calls.append(kind)
        if kind == "pod":
            raise RuntimeError("pod is still running")

    monkeypatch.setattr(seeded, "delete_owned_resource", delete)
    monkeypatch.setattr(
        seeded, "deregister_synthetic_cluster", lambda *a, **k: calls.append("registry")
    )
    monkeypatch.setattr(seeded, "database_residuals", lambda p: {"total": 1})
    monkeypatch.setattr(seeded, "registry_residuals", lambda: {"count": 1})
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda p: {"count": 1})
    result = {"verdict": "PASS"}
    seeded.cleanup(
        SimpleNamespace(pod="pod", configmap="cm", run_prefix="review-"),
        tmp_path,
        "review-run",
        result,
        {"command_id": "remote-review"},
        state={"registry_started": True, "probe_started": True},
        purge=lambda seed: calls.append("purge") or {},
    )
    assert "purge" not in calls
    assert "configmap" in calls
    assert result["verdict"] == "FAIL"
    assert "pod is still running" in result["cleanup_error"]


def test_resource_from_another_run_is_never_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    def dataplane(*args: str, **kwargs: Any) -> str:
        calls.append(args)
        return '{"uid":"uid-1","labels":{"gpu-fault.io/acceptance-run":"other"}}'

    monkeypatch.setattr(seeded, "dataplane", dataplane)
    with pytest.raises(RuntimeError, match="ownership|another run"):
        seeded.delete_owned_resource("pod", "pod", "review-run")
    assert all(args[0] == "get" for args in calls), (
        f"foreign resource was mutated: {calls}"
    )


@pytest.mark.parametrize(
    "response", [{}, {"remaining": []}, {"remaining": [], "remaining_links": False}]
)
@pytest.mark.parametrize("module", [seeded, net003])
def test_seed_purge_requires_explicit_empty_record_and_link_proof(
    response: dict[str, Any], module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = seeded if module is seeded else net
    monkeypatch.setattr(transport, "cpu_python", lambda *a, **k: response)
    seed = {
        **seeded.seed_identity("review-run"),
        "notification_id": "notification-review",
        "deduplication_key": "dedup-review",
    }
    with pytest.raises(RuntimeError, match="residual"):
        module.purge_seed(seed)


def test_cmd018_tracks_first_command_before_starting_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    first_id = "remote-review-first"
    dispatch = {
        "workflow_id": "wf-review",
        "incident_id": "inc-review",
        "event_id": "event-review",
        "first": {"remote_command_id": first_id},
        "all_command_ids": [first_id],
    }
    monkeypatch.setattr(seeded, "require_environment", lambda: None)
    monkeypatch.setattr(seeded, "preflight_residuals", lambda *a, **k: {})
    monkeypatch.setattr(seeded, "register_synthetic_cluster", lambda *a, **k: None)
    monkeypatch.setattr(seeded, "cpu_python", lambda *a, **k: dispatch)
    monkeypatch.setattr(cmd018.verdicts, "dispatch_errors", lambda d: [])

    def fail_probe(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("probe creation failed")

    monkeypatch.setattr(seeded, "create_probe_pod", fail_probe)
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        seeded, "cleanup", lambda *a, state, **k: observed.append(state)
    )
    assert (
        cmd018.run_case(tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1))
        == 1
    )
    assert len(observed) == 1
    assert observed[0]["command_ids"] == [first_id]


@pytest.mark.parametrize(
    "outcome", ["removed", "removing", "unchanged", "replaced", "read-failed"]
)
def test_delete_ack_loss_requires_checked_absence_of_the_bound_uid(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    metadata = {
        "uid": "original",
        "resourceVersion": "1",
        "labels": {seeded.RUN_LABEL: "review-run"},
    }
    deletions = []
    reads_after_delete = []

    def client(*args: str, **kwargs: Any) -> str:
        assert kwargs.get("check", True), "cleanup requires checked API responses"
        if args[0] == "delete":
            options = json.loads(kwargs["stdin"])
            assert options["preconditions"] == {
                "uid": "original",
                "resourceVersion": "1",
            }
            deletions.append(args)
            raise TimeoutError("delete acknowledgement lost")
        assert args[0] == "get"
        if not deletions:
            return json.dumps(metadata)
        reads_after_delete.append(args)
        if outcome == "read-failed":
            raise RuntimeError("read unavailable")
        if outcome == "removed" or (
            outcome == "removing" and len(reads_after_delete) > 1
        ):
            return ""
        if outcome == "removing":
            return json.dumps({**metadata, "deletionTimestamp": "2026-09-12T00:00:00Z"})
        if outcome == "replaced":
            return json.dumps({**metadata, "uid": "replacement"})
        return json.dumps(metadata)

    monkeypatch.setattr(seeded.time, "sleep", lambda seconds: None)
    if outcome in {"removed", "removing"}:
        seeded.delete_owned_resource(
            "pod",
            "probe",
            "review-run",
            client=client,
            expected_uid="original",
            require_uid=True,
        )
    else:
        with pytest.raises(
            (RuntimeError, TimeoutError), match="acknowledgement|replaced|unavailable"
        ):
            seeded.delete_owned_resource(
                "pod",
                "probe",
                "review-run",
                client=client,
                expected_uid="original",
                require_uid=True,
            )
    assert len(deletions) == 1
    assert reads_after_delete, f"lost delete ACK was not reconciled for {outcome}"


@pytest.mark.parametrize("receipt", [None, "old-uid"])
def test_cleanup_refuses_a_missing_or_replaced_pod_uid_before_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt: str | None
) -> None:
    seeded.write_json(
        tmp_path / "probe-resource-identities.json",
        {"run_id": "review-run", "resources": {"pod/probe": receipt}},
    )

    def dataplane(*args: str, **kwargs: Any) -> str:
        assert args[0] == "get", "an unproven Pod must not be deleted"
        if args[1:3] == ("pod", "probe"):
            return json.dumps(
                {
                    "uid": "replacement",
                    "resourceVersion": "2",
                    "labels": {seeded.RUN_LABEL: "review-run"},
                }
            )
        return ""

    monkeypatch.setattr(seeded, "dataplane", dataplane)
    monkeypatch.setattr(seeded, "database_residuals", lambda prefix: {"total": 1})
    monkeypatch.setattr(seeded, "registry_residuals", lambda: {"count": 1})
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda probe: {"count": 1})
    monkeypatch.setattr(
        seeded,
        "deregister_synthetic_cluster",
        lambda *a: pytest.fail("a possibly running claimant prevents deregistration"),
    )
    result = {"verdict": "PASS"}
    seeded.cleanup(
        SimpleNamespace(pod="probe", configmap="cm", run_prefix="review"),
        tmp_path,
        "review-run",
        result,
        seeded.seed_identity("review-run"),
        state={"probe_started": True, "registry_started": True},
        purge=lambda seed: pytest.fail("a possibly running claimant prevents purge"),
    )
    assert result["verdict"] == "FAIL"
    assert "probe stop" in result["cleanup_error"]


@pytest.mark.parametrize("lose_ack", [False, True])
def test_probe_creation_records_uid_before_any_readiness_or_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lose_ack: bool
) -> None:
    probe = seeded.SeededCommandProbe(
        case_id="GF-REGIONAL-NET-003",
        run_prefix="review",
        pod="probe",
        configmap="cm",
        owner="review-owner",
        script=Path(net003.__file__).with_name("probes") / "net003_executor.py",
    )
    resources: dict[str, dict[str, Any]] = {}
    creations = []

    def dataplane(*args: str, **kwargs: Any) -> str:
        if args[0] == "create":
            manifest = json.loads(kwargs["stdin"])
            kind = manifest["kind"].lower()
            manifest["metadata"].update(uid=f"uid-{kind}", resourceVersion="1")
            resources[kind] = manifest["metadata"]
            creations.append(kind)
            if kind == "pod" and lose_ack:
                raise TimeoutError("create acknowledgement lost")
            return json.dumps(manifest["metadata"])
        if args[0] == "get":
            return json.dumps(resources[args[1]])
        if args[0] == "wait":
            assert (
                seeded.probe_resource_uid(tmp_path, "pod", "probe", "review-run")
                == "uid-pod"
            )
            return "Ready"
        raise AssertionError(args)

    monkeypatch.setattr(seeded, "dataplane", dataplane)
    monkeypatch.setattr(
        seeded,
        "executor_identity",
        lambda **k: {
            "executor_artifact_sha256": "a" * 64,
            "executor_compatibility_digest": "b" * 64,
        },
    )
    monkeypatch.setattr(seeded, "executor_image", lambda: "image@sha256:" + "a" * 64)
    monkeypatch.setattr(seeded, "wait_file", lambda *a: None)
    monkeypatch.setattr(seeded, "read_state", lambda *a: {"ready": True})
    if lose_ack:
        with pytest.raises(TimeoutError, match="acknowledgement"):
            seeded.create_probe_pod(probe, tmp_path, run_id="review-run")
    else:
        assert seeded.create_probe_pod(probe, tmp_path, run_id="review-run") == {
            "ready": True
        }
    assert creations == ["configmap", "pod"]
    assert (
        seeded.probe_resource_uid(tmp_path, "pod", "probe", "review-run") == "uid-pod"
    )
    assert (
        seeded.probe_resource_uid(tmp_path, "configmap", "cm", "review-run")
        == "uid-configmap"
    )
