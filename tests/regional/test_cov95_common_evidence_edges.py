from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from scripts.e2e.regional import boot023_verdicts as history
from scripts.e2e.regional import regional_live_fixture as regional
from scripts.e2e.regional import run_boot023_release_history as history_runner
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from tests.regional._cov95_boot_history import HistoryModel
from tests.regional._cov95_common_live import LiveModel


@pytest.mark.parametrize(
    "value", [None, datetime(2026, 9, 12), datetime(2026, 9, 12, tzinfo=timezone.utc)]
)
def test_history_time_parser_keeps_missing_time_distinct_from_a_valid_utc_time(
    value,
) -> None:
    assert history.parse_time(value) == (
        None if value is None else datetime(2026, 9, 12, tzinfo=timezone.utc)
    ), "only valid timestamps may enter history ordering comparisons"


def test_unparsed_history_and_missing_mirror_never_count_as_valid_evidence(
    tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    preflight = model.plan(tmp_path)
    errors = history.preflight_errors(
        classification=preflight["classification"],
        release_id=preflight["release_id"],
        state_phase="complete",
        probes=preflight["registry_probes"],
        rollback_flag_message=preflight["rollback_flag_rejection"],
        predecessor_valid=True,
        tests_passed=True,
        history_before=[{"_unparsed": "fixture-invalid"}],
        snapshot_groups_before=preflight["snapshot_groups"],
    )
    assert len(errors) == 1 and "unparsable lines" in errors[0], (
        "malformed historical lines must not disappear from preflight validation"
    )
    assert history.history_entry_errors(
        {"_unparsed": "fixture-invalid"}, release_id="unit-release", label="entry"
    ) == ["entry: history line is not a JSON object"]
    assert history.mirror_errors([], []) == ["no appended entries to mirror"], (
        "two empty histories cannot prove an append or a mirrored release"
    )


def test_registry_probe_without_pod_identity_is_not_complete(
    tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    probes = history_runner.registry_probes(model.regional, ("api",))
    probes[0].pop("pod")
    assert (
        "registry probe Pod identity is missing or duplicated"
        in history.registry_probe_errors(probes)
    ), "correct digests without a known Pod do not prove replica coverage"


@pytest.mark.parametrize("failure", ["cpu-file", "gpu-file", "context"])
def test_regional_settings_refuse_missing_connections_or_identity(
    failure, tmp_path
) -> None:
    model = LiveModel(tmp_path)
    changes = (
        {"gpu_context": ""}
        if failure == "context"
        else {
            ("cpu_kubeconfig" if failure == "cpu-file" else "gpu_kubeconfig"): tmp_path
            / "missing"
        }
    )
    with pytest.raises(ValueError, match="does not exist|empty identity"):
        replace(model.settings, **changes)
    assert model.calls == [], "invalid settings must not execute a transport"


def test_interpreter_and_ready_pod_selection_refuse_unknown_targets(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown Kubernetes plane"):
        regional.component_python("unbound")
    model = LiveModel(tmp_path)
    model.documents[("gpu", "pod", "")] = {"items": []}
    with pytest.raises(regional.RegionalFixtureError, match="no Ready gpu Pod"):
        model.ready_pod("gpu", "fixture")


def test_missing_runtime_deployment_cannot_pass_identity_validation(tmp_path) -> None:
    model = LiveModel(tmp_path)
    identity = model.runtime_identity()
    missing = next(iter(identity["deployments"]["cpu"]))
    identity["deployments"]["cpu"].pop(missing)
    assert (
        f"cpu/{missing} runtime deployment is unavailable"
        in regional.runtime_identity_errors(identity)
    ), "a partial runtime snapshot cannot establish complete release identity"


@pytest.mark.parametrize("body", ["not-json", "[]", "null"])
def test_unreadable_predecessor_is_invalid_not_absent(body, tmp_path) -> None:
    path = tmp_path / "predecessor.json"
    path.write_text(body)
    result = regional.predecessor_evidence(path, "fixture-case")
    assert result["verdict"] == "INVALID", (
        "read failure must not become MISSING or PASS"
    )
    assert result["execution_allowed"] is False and result["evidence_valid"] is False, (
        "invalid predecessor evidence cannot authorize a successor"
    )


@pytest.mark.parametrize("failure", ["case", "inputs"])
def test_evidence_resume_refuses_changed_case_or_target_without_rewriting(
    failure, tmp_path
) -> None:
    path = tmp_path / "evidence.json"
    recorder = EvidenceRecorder(path, case_id="fixture-case", inputs={"target": "a"})
    recorder.stage("observed", lambda: {"count": 1})
    before = json.loads(path.read_text())
    with pytest.raises(RuntimeError, match="another case|inputs differ"):
        EvidenceRecorder(
            path,
            case_id="other-case" if failure == "case" else "fixture-case",
            inputs={"target": "b" if failure == "inputs" else "a"},
        )
    assert json.loads(path.read_text()) == before, (
        "failed resume must preserve prior evidence"
    )


def test_failed_evidence_resume_never_promotes_its_old_verdict(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    recorder = EvidenceRecorder(path, case_id="fixture-case", inputs={})
    recorder.note("verdict", "FAIL")
    resumed = EvidenceRecorder(path, case_id="fixture-case", inputs={})
    assert resumed.document["verdict"] == "FAIL", (
        "resuming does not manufacture successful evidence"
    )
