from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from tests.regional._cov95_ha011_support import IMAGE_ID
from tests.regional.test_cov95_ha011_resources import prepare


@pytest.mark.parametrize(
    "report",
    [
        {
            "case_id": contracts.CASE_ID,
            "verdict": "FAIL",
            "stage": "identity",
            "error_type": "ProofError",
        },
        {
            "case_id": contracts.CASE_ID,
            "verdict": "FAIL",
            "stage": ["unsafe"],
            "error_type": "ProofError",
        },
        {
            "case_id": contracts.CASE_ID,
            "verdict": "FAIL",
            "stage": "identity",
            "error_type": "unapproved-sensitive-input",
        },
        {
            "case_id": contracts.CASE_ID,
            "verdict": "FAIL",
            "stage": "identity",
            "error_type": "ProofError",
            "raw_message": "unapproved-sensitive-input",
        },
        "unapproved-sensitive-input",
    ],
)
def test_early_container_exit_keeps_only_bounded_failure_fields_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, report: object
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    original = fake.dispatch

    def dispatch(args, namespace, text):
        if args[0] == "logs":
            assert args[-1] == "--limit-bytes=65536"
            return json.dumps(report)
        output = original(args, namespace, text)
        if args[0] == "patch":
            pod = fake.object("Pod", contracts.POD_NAME)
            runtime = pod["status"]["containerStatuses"][0]
            runtime["state"] = {"terminated": {"exitCode": 1}}
        return output

    fake.dispatch = dispatch
    with pytest.raises(contracts.ProofError, match="terminated before arm"):
        lifetime.arm(IMAGE_ID)
    assert not fake.armed, (
        "test_early_container_exit_keeps_only_bounded_failure_fields_before_cleanup: expected no fake.armed"
    )
    evidence = lifetime.failure_details
    assert evidence is not None
    assert evidence["stage"] == "before-arm"
    assert evidence["container_exits"] == {"runtime": 1}
    if (
        isinstance(report, dict)
        and len(report) == 4
        and report.get("stage") == "identity"
        and (report.get("error_type") == "ProofError")
    ):
        assert evidence["probe_stage"] == "identity"
        assert evidence["probe_error_type"] == "ProofError"
    else:
        assert evidence["probe_report"] == "unrecognized"
    assert "unapproved-sensitive-input" not in json.dumps(evidence)
    assert lifetime.cleanup()["priority_class_absent"] is True
    assert ("Namespace", None, settings.isolated_namespace) not in fake.objects


def test_database_exit_fails_immediately_without_waiting_for_runtime_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    lifetime.arm(IMAGE_ID)
    pod = fake.object("Pod", contracts.POD_NAME)
    for status in pod["status"]["containerStatuses"]:
        status["state"] = (
            {"terminated": {"exitCode": 2}}
            if status["name"] == "postgres"
            else {"running": {"startedAt": "observed"}}
        )
    log_reads = sum(args[0] == "logs" for _namespace, args in fake.calls)
    with pytest.raises(contracts.ProofError, match="database container terminated"):
        lifetime.collect(IMAGE_ID)
    assert lifetime.failure_details == {
        "stage": "collect",
        "container_exits": {"postgres": 2},
    }
    assert sum(args[0] == "logs" for _namespace, args in fake.calls) == log_reads
    assert lifetime.cleanup()["priority_class_absent"] is True
