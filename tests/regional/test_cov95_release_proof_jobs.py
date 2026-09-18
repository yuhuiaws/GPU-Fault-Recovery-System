from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault_release import regional_release_probe_job as jobs
from gpu_fault_release import regional_release_store_proof as store_proof
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._prerequisite_repair_support import repair_release


def proof(instance: Any, records: list[Any]) -> dict[str, Any]:
    return store_proof.bootstrap_store_proof(
        instance,
        lambda record: records.append(copy.deepcopy(record)),
        require_empty=False,
    )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("owner", "no live registered refresher owner"),
        ("created-absent", "created proof Job is absent"),
        ("incomplete", "no UID-bound completion"),
        ("malformed-log", "invalid evidence"),
        ("nonobject-log", "non-object evidence"),
        ("incomplete-proof", "safety proof failed or is incomplete"),
    ],
)
def test_proof_job_requires_owned_creation_completion_and_typed_evidence(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    instance = repair_release(monkeypatch)
    records: list[Any] = []
    if fault == "owner":
        instance.runner.live.pop("cronjob.batch")
    original = instance.runner.run

    def run(arguments: list[str], **kwargs: Any) -> str:
        output = original(arguments, **kwargs)
        if fault == "created-absent" and "create" in arguments and "-f" in arguments:
            instance.runner.jobs.clear()
        if fault == "incomplete" and arguments[0] == "bash":
            for job in instance.runner.jobs.values():
                job["status"]["conditions"] = []
        if "logs" in arguments:
            if fault == "malformed-log":
                return "{"
            if fault == "nonobject-log":
                return "[]"
            if fault == "incomplete-proof":
                return "{}"
        return output

    monkeypatch.setattr(instance.runner, "run", run)
    with pytest.raises(ReleaseError, match=problem):
        proof(instance, records)
    assert instance.runner.jobs == {}
    if fault == "owner":
        assert records == []
    else:
        assert records[-1]["status"] == "REMOVED"


@pytest.mark.parametrize(
    "field", ["namespace", "name", "run_id", "spec_sha256", "admitted_spec_sha256"]
)
def test_cleanup_refuses_unbound_job_record_before_any_read(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    instance = repair_release(monkeypatch)
    proof(instance, [])
    record = jobs.probe_job_record(instance.runner.created_jobs[0])
    record[field] = "invalid"
    before = list(instance.runner.events)
    with pytest.raises(ReleaseError, match="namespace or ownership record differs"):
        jobs.cleanup_probe_job(instance, record, lambda _record: None)
    assert instance.runner.events == before


def test_job_record_normalizes_only_explicit_controller_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    proof(instance, [])
    original = copy.deepcopy(instance.runner.created_jobs[0])
    original["spec"]["template"].pop("metadata", None)
    original["spec"]["template"]["spec"]["initContainers"] = [
        {"name": "setup", "image": "example/image:1"}
    ]
    expected = jobs.probe_job_record(original)
    admitted = copy.deepcopy(original)
    uid = admitted["metadata"]["uid"]
    name = admitted["metadata"]["name"]
    admitted["spec"]["selector"] = {"matchLabels": {"controller-uid": uid}}
    admitted["spec"]["template"]["metadata"] = {
        "creationTimestamp": None,
        "labels": {
            "batch.kubernetes.io/controller-uid": uid,
            "controller-uid": uid,
            "batch.kubernetes.io/job-name": name,
            "job-name": name,
        },
    }
    pod = admitted["spec"]["template"]["spec"]
    pod["serviceAccount"] = "default"
    pod["serviceAccountName"] = "default"
    pod["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    for volume in pod["volumes"]:
        if "configMap" in volume:
            volume["configMap"]["defaultMode"] = 420
    actual = jobs.probe_job_record(admitted)
    assert actual["admitted_spec_sha256"] == expected["admitted_spec_sha256"]
    assert actual["spec_sha256"] != expected["spec_sha256"]


def test_cleanup_acknowledgement_is_not_proof_of_resource_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    records: list[Any] = []
    original = instance.runner.run

    def run(arguments: list[str], **kwargs: Any) -> str:
        if "delete" in arguments or "--for=delete" in arguments:
            return ""
        return original(arguments, **kwargs)

    monkeypatch.setattr(instance.runner, "run", run)
    with pytest.raises(ReleaseError, match="cleanup did not converge"):
        proof(instance, records)
    assert records[-1]["status"] == "RUNNING"
    assert len(instance.runner.jobs) == 1


def test_proof_job_records_uid_before_successful_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    records: list[Any] = []
    result = proof(instance, records)
    assert result["safe"] is True
    assert result["job_uid"] == records[-1]["uid"]
    assert [record["status"] for record in records] == ["PLANNED", "RUNNING", "REMOVED"]
    assert instance.runner.jobs == {}


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("unconfigured", "requires an Aurora cluster identity"),
        ("missing", "identity is unavailable"),
        ("binding", "binding differs or is incomplete"),
    ],
)
def test_database_proof_requires_complete_database_identity(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    instance = repair_release(monkeypatch)
    if fault == "unconfigured":
        instance.config.health.aurora_cluster_id = None
    original = instance.runner.run

    def read(arguments: list[str], **kwargs: Any) -> str:
        result = original(arguments, **kwargs)
        if arguments[:3] == ["aws", "rds", "describe-db-clusters"]:
            value = json.loads(result)
            if fault == "missing":
                value["DBClusters"] = []
            elif fault == "binding":
                value["DBClusters"][0]["Port"] = 5433
            return json.dumps(value)
        return result

    monkeypatch.setattr(instance.runner, "run", read)
    with pytest.raises(ReleaseError, match=problem):
        store_proof.database_identity(instance)
    assert instance.runner.created_jobs == []


def test_database_identity_drift_during_proof_is_rejected_after_job_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    original = instance.runner.run
    reads = 0

    def read(arguments: list[str], **kwargs: Any) -> str:
        nonlocal reads
        result = original(arguments, **kwargs)
        if arguments[:3] == ["aws", "rds", "describe-db-clusters"]:
            reads += 1
            if reads > 1:
                value = json.loads(result)
                value["DBClusters"][0]["DbClusterResourceId"] = "replacement"
                return json.dumps(value)
        return result

    monkeypatch.setattr(instance.runner, "run", read)
    records: list[Any] = []
    with pytest.raises(ReleaseError, match="identity changed during the read"):
        proof(instance, records)
    assert records[-1]["status"] == "REMOVED"
    assert instance.runner.jobs == {}
