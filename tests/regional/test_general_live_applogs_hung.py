"""GF-GENERAL-LIVE-017 ``missing_flight_recorder``: the hung chain's process capture.

The variant's context (``test_general_live_applogs.mfr_ctx``) runs 011's fakes
without a flight recorder and scripts the hung records: the finding's
parameters on the COLLECT_HUNG_TRIAGE steps, the bundle step as
``bind_hung_bundle`` binds it for its decision. These tests judge
``general_live_applog_hung`` -- the triage's request, the sampled rows' process
state and the bundle's binding against its decision: the live HyperPod shape
(2026-10-26) passes, a bundle an eligible decision targets passes, and each
contradiction is named.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_applog_hung as hung
from scripts.e2e.regional import general_live_applogs as mod
from tests.regional.test_general_live_applogs import (
    CONSERVATIVE_BUNDLE,
    HUNG_PARAMETERS,
    NODE,
    UNDETERMINED_DECISION,
    fake_jobs,
    mfr_ctx,
)
from tests.regional.test_general_live_nccl import FakeNcclJob, hung_records


@pytest.fixture(autouse=True)
def _patch_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jobs(monkeypatch)


ELIGIBLE_DECISION = {
    **UNDETERMINED_DECISION,
    "classification": "PLAUSIBLE",
    "confidence": "PLAUSIBLE",
    "culprit_ranks": [0],
    "control_ranks": [1],
    "candidate_count": 1,
    "targeted_capture_eligible": True,
    "temporal_verified": True,
    "membership_verified": True,
}
TARGETED_BUNDLE = {
    **CONSERVATIVE_BUNDLE,
    "capture_process_state": True,
    "hung_triage_decision": ELIGIBLE_DECISION,
    "target_pids_by_node": {NODE: [4242, 4243]},
    "max_processes": 2,
    "operator_escalation_required": False,
    "bundle_scope": "identity_bound_candidates",
}
#: A sampled rank without a flight recorder: its py-spy sample was collected.
FR_OFF_RANK = {
    "pid": 4242,
    "rank": 0,
    "flight_recorder": {"status": "not_configured"},
    "python_stack": {"status": "COLLECTED"},
}


def test_missing_flight_recorder_passes_on_the_independent_trigger(
    tmp_path: Path,
) -> None:
    """The live shape: triage requested and took the capture; the bundle, bound
    to an UNDETERMINED decision, is conservative."""

    ctx = mfr_ctx(tmp_path, ranks=[FR_OFF_RANK], parameters=HUNG_PARAMETERS)
    result = mod.missing_flight_recorder(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeNcclJob.instances[-1]
    assert job.flight_recorder is False and job.released == 1
    assert (
        result["proof_scope"] == "native" and result["application_log_incidents"] == []
    )


def test_hung_oracle_module_reads_the_live_shape_directly() -> None:
    records = hung_records("2026-10-24T00:00:00+00:00", ranks=[FR_OFF_RANK])
    for step in records["workflows"]["wf"]["official_steps"]:
        if step["operation"] == "COLLECT_HUNG_TRIAGE":
            step["parameters"] = dict(HUNG_PARAMETERS)
        elif step["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE":
            step["parameters"] = dict(CONSERVATIVE_BUNDLE)
    assert hung.independent_trigger_errors(records) == []
    assert hung.bundle_binding_errors(records) == []
    assert hung.sampled_process_state_errors(records) == []
    assert mod.independent_trigger_errors is hung.independent_trigger_errors


def test_missing_flight_recorder_accepts_a_bundle_an_eligible_decision_targets(
    tmp_path: Path,
) -> None:
    ctx = mfr_ctx(
        tmp_path,
        ranks=[FR_OFF_RANK],
        parameters=HUNG_PARAMETERS,
        bundle=TARGETED_BUNDLE,
    )
    assert mod.missing_flight_recorder(ctx)["stages"]["independent_trigger"] == []


@pytest.mark.parametrize(
    "ranks, parameters, bundle, needle",
    [
        (
            [FR_OFF_RANK],
            {**HUNG_PARAMETERS, "capture_process_state": False},
            None,
            "the hung triage did not request process state capture",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {**CONSERVATIVE_BUNDLE, "capture_process_state": True},
            "captures process state without an eligible decision",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {**TARGETED_BUNDLE, "operator_escalation_required": True},
            "operator_escalation_required is True",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {**TARGETED_BUNDLE, "target_pids_by_node": {}},
            "a targeted bundle names no PIDs",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {**CONSERVATIVE_BUNDLE, "target_pids_by_node": {NODE: [4242]}},
            "a conservative bundle still targets PIDs",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {**CONSERVATIVE_BUNDLE, "bundle_scope": "identity_bound_candidates"},
            "contradicts capture_process_state=False",
        ),
        (
            [FR_OFF_RANK],
            HUNG_PARAMETERS,
            {
                k: v
                for k, v in CONSERVATIVE_BUNDLE.items()
                if k != "hung_triage_decision"
            },
            "never bound to a hung_triage_decision",
        ),
        (
            [{"pid": 4242, "rank": 0, "flight_recorder": {"status": "not_configured"}}],
            HUNG_PARAMETERS,
            None,
            "none of the 1 sampled rank(s) carries process state",
        ),
        ([], HUNG_PARAMETERS, None, "COLLECT_HUNG_TRIAGE sampled no rank process"),
    ],
)
def test_missing_flight_recorder_judges_the_triage_request_and_the_bundle_binding(
    tmp_path: Path,
    ranks: list[dict[str, Any]],
    parameters: dict[str, Any],
    bundle: dict[str, Any] | None,
    needle: str,
) -> None:
    ctx = mfr_ctx(tmp_path, ranks=ranks, parameters=parameters, bundle=bundle)
    result = mod.missing_flight_recorder(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(needle in error for error in result["stages"]["independent_trigger"]), (
        result["errors"]
    )


def test_missing_flight_recorder_fails_on_a_fabricated_dump_or_a_log_trigger(
    tmp_path: Path,
) -> None:
    paired = {"pid": 4242, "rank": 0, "flight_recorder": {"status": "PAIRED"}}
    result = mod.missing_flight_recorder(
        mfr_ctx(tmp_path / "dump", ranks=[paired], parameters=HUNG_PARAMETERS)
    )
    assert result["verdict"] == "FAIL" and any(
        "FR was not configured" in e for e in result["errors"]
    )
    logged = {
        **HUNG_PARAMETERS,
        "detection_basis": "APPLICATION_LOG",
        "diagnostic_reason": "APPLICATION_ERROR",
    }
    result = mod.missing_flight_recorder(
        mfr_ctx(tmp_path / "log", ranks=[FR_OFF_RANK], parameters=logged)
    )
    assert result["verdict"] == "FAIL"
    assert any("detection basis" in error for error in result["errors"]), result[
        "errors"
    ]
    result = mod.missing_flight_recorder(
        mfr_ctx(tmp_path / "none", ranks=[FR_OFF_RANK], parameters=None)
    )
    assert any("no diagnostic parameters" in error for error in result["errors"]), (
        result["errors"]
    )
