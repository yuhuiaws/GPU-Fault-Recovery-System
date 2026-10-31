"""GF-GENERAL-LIVE-025/031 injector/verdict contract over scripted records."""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.gpu_collection_errors import collection_error_finding
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSource, GpuMetricsThresholds
from scripts.e2e.regional import general_live_collectors as col
from scripts.e2e.regional.general_live_fixture import PAUSE_MODULE, VariantNotRun
from scripts.e2e.regional.general_live_verdicts import finding_incidents
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    workflow,
)

SPECS = {spec.variant: spec for spec in col.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey(
    *, exporters: int = 1, units: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"systemctl": True, "systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "exporter_processes": [
            {"pid": 100 + i, "start_ticks": "5"} for i in range(exporters)
        ],
        "units": units
        or {
            "gpu-fault-host-collector.service": {
                "UnitFileState": "enabled",
                "ActiveState": "active",
            },
            "gpu-fault-log-collector.service": {
                "UnitFileState": "disabled",
                "ActiveState": "inactive",
            },
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def node_probe(survey_value: dict[str, Any], **extra: Any) -> FakeProbe:
    script = {
        "survey": survey_value,
        "exporter-pause": {"pid": 100, "stopped": True, "deadman": "d"},
        "exporter-resume": {"processes": [{"pid": 100, "resumed": True}]},
        "unit-pause": {"unit": "u", "stopped": True, "deadman": "d"},
        "unit-resume": {"unit": "u"},
        "log-collector-enable": {"after": {"ActiveState": "active"}},
        "log-collector-disable": {"restored": True},
    }
    script.update(extra)
    return FakeProbe(script)


def observation(channel: str, state: str, reason: str | None = None) -> dict[str, Any]:
    """The silence scan's ``collector_observation_valid`` sample
    (``observation_health.record_observation``): a HEALTHY sample reads
    FRESH_SUCCESS, an UNKNOWN one the reason -- a paused producer's silence is
    STALE_OBSERVATION, a failing poll COLLECTION_ERROR."""

    if reason is None:
        reason = "FRESH_SUCCESS" if state == "HEALTHY" else "STALE_OBSERVATION"
    return {
        "name": "collector_observation_valid",
        "device": channel,
        "value": 1 if state == "HEALTHY" else 0,
        "labels": {"state": state, "reason": reason},
        "observed_at": "t",
    }


def completeness_event(batch: str = f"dcgm-{NODE}-1761363470000000") -> str:
    """The event id the product gives a failed DCGM scrape's finding.

    Minted by the product's own path -- ``collection_error_finding`` books
    the streak, ``gpu_node_health_findings`` hands it to the node-health
    ingest (its context is only read for a finding without an action).
    """

    errors = ["DCGM exporter scrape failed: timed out"]
    batch_record = GpuMetricBatch(
        batch_id=batch,
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        observed_at=datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[],
        collection_errors=errors,
    )
    booked = collection_error_finding(batch_record, None, GpuMetricsThresholds())
    assert booked is not None, "a scrape with collection errors books a finding"
    return gpu_node_health_findings(None, [booked[1]])[0].event_id


def test_completeness_matcher_takes_the_incident_the_product_mints() -> None:
    event = completeness_event()
    product = {"event_id": event, "event_type": "NODE_HEALTH"}
    host_shaped = {
        "event_id": f"b-dcgm_field_completeness-{NODE}",
        "event_type": "NODE_HEALTH",
    }
    document = {
        "incidents": [
            {**product, "incident_id": f"inc-{event}"},
            {**host_shaped, "incident_id": "inc-host-shaped"},
        ]
    }
    selected = finding_incidents(document, "dcgm_field_completeness")
    assert [item["incident_id"] for item in selected] == [f"inc-{event}"], event


#: What the product's VALIDATE_HOST records while the exporter is paused: the
#: OBSERVABILITY contract's wait (``health_evidence_validation.pending``) plus
#: the executor's wait accounting (``execution/step_bounds``) past 300 s.
WAITING_DETAILS = {
    "host_diagnostic_failure": True,
    "health_evidence_pending": "collector state is UNKNOWN",
    "missing_observations": {f"{NODE}/GPU_METRICS": "COLLECTION_ERROR"},
    "step_waiting_seconds": 412,
    "step_waiting_slow": True,
}
#: The fake clock of ``make_context`` starts 2026-10-24T00:00Z and moves one
#: second per reading: a validation that finished at 00:30Z finished after the
#: resume, one that finished the day before did not.
VALIDATION_STARTED = "2026-10-24T00:24:00+00:00"
VALIDATION_FINISHED = "2026-10-24T00:30:00+00:00"
VALIDATION_STALE = "2026-10-23T23:59:00+00:00"


def validation_execution(
    status: str,
    *,
    details: dict[str, Any] | None = None,
    error: str | None = None,
    started_at: str = VALIDATION_STARTED,
    updated_at: str = VALIDATION_FINISHED,
) -> dict[str, Any]:
    return {
        "step_index": 2,
        "operation": "VALIDATE_HOST",
        "status": status,
        "details": dict(WAITING_DETAILS if details is None else details),
        "error": error,
        "started_at": started_at,
        "updated_at": updated_at,
    }


WAITING_EXECUTION = validation_execution("WAITING", updated_at=VALIDATION_STARTED)
SUCCEEDED_EXECUTION = validation_execution(
    "SUCCEEDED", details={"validation": "VALIDATE_HOST", "observation_state": "HEALTHY"}
)
#: The executor's per-step cap (``bounded_waiting_outcome``): FAILED with the
#: cap in the details and the diagnostic-only incident closed RECOVERED.
CAP_EXECUTION = validation_execution(
    "FAILED",
    details={**WAITING_DETAILS, "step_waiting_timeout_seconds": 600},
    error="step 2/VALIDATE_HOST stayed non-terminal for 600s, past the 600s per-step cap",
)


def dcgm_error_doc(
    *,
    state: str = "UNKNOWN",
    with_workflow: bool = True,
    execution: dict[str, Any] | None = WAITING_EXECUTION,
    workflow_status: str = "RUNNING",
    incident_state: str = "ACTION_PENDING",
    collector_statuses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The node's activity while the exporter is paused: the completeness
    incident, its chain at VALIDATE_HOST (``execution``; None = not yet run)."""

    event = completeness_event()
    inc = {
        **incident(
            "inc-dcgm",
            "dcgm_field_completeness",
            workflow="wf-dcgm",
            state=incident_state,
        ),
        "event_id": event,
        "incident_id": "inc-dcgm",
    }
    workflows = []
    if with_workflow:
        workflows = [
            workflow(
                "wf-dcgm",
                "inc-dcgm",
                ["FREEZE_EVIDENCE", "TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"],
                status=workflow_status,
                executions=[execution] if execution is not None else [],
            )
        ]
    return {
        "incidents": [inc],
        "workflows": workflows,
        "batches": [],
        "metric_latest": [observation("GPU_METRICS", state)],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": collector_statuses
        or [
            {
                "collector": "GPU_METRICS",
                "last_error_at": "2026-10-24T00:05:00+00:00",
                "last_success_at": "2026-10-24T00:00:00+00:00",
                "errors": ["scrape failed"],
            }
        ],
    }


def recovered_dcgm_doc(
    execution: dict[str, Any] = SUCCEEDED_EXECUTION,
    *,
    workflow_status: str = "SUCCEEDED",
    incident_state: str = "RECOVERED",
) -> dict[str, Any]:
    """The node's activity after the resume: the chain terminal on fresh
    GPU_METRICS evidence, the channel HEALTHY with a new success."""

    fresh = fresh_success_doc("GPU_METRICS")
    return {
        **dcgm_error_doc(
            state="HEALTHY",
            execution=execution,
            workflow_status=workflow_status,
            incident_state=incident_state,
            collector_statuses=fresh["collector_statuses"],
        ),
        "agent": fresh["agent"],
    }


def fresh_success_doc(channel: str) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "metric_latest": [observation(channel, "HEALTHY")],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [
            {
                "collector": channel,
                "last_success_at": "2026-10-24T01:00:00+00:00",
                "last_error_at": None,
                "errors": [],
            }
        ],
        "agent": {"lifecycle_state": "ACTIVE"},
    }


def silence_doc(channel: str) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [
            workflow(
                "wf-s",
                "collector-silent",
                ["TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"],
                status="SUCCEEDED",
            )
        ],
        "batches": [],
        "metric_latest": [observation(channel, "UNKNOWN")],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [
            {
                "notification_id": "n1",
                "incident_id": "collector-silent",
                "category": "COLLECTOR",
                "deduplication_key": f"collector-silent/c/{NODE}/{channel}/1",
                "created_at": "t",
                "subject": "silent",
                "status": "SENT",
            }
        ],
        "collector_statuses": [
            {
                "collector": channel,
                "last_success_at": "2026-10-24T00:00:00+00:00",
                "last_error_at": None,
                "errors": [],
            }
        ],
        "agent": {"lifecycle_state": "ACTIVE"},
    }


def settings(spec: Any, tmp_path: Path, *, approvals: tuple[str, ...] | None = None):
    s = make_settings(spec)
    updates: dict[str, Any] = {"site_file": tmp_path / "site.yaml"}
    (tmp_path / "site.yaml").write_text("site: test\n")
    if approvals is not None:
        updates["approvals"] = approvals
    return type(s)(**{**s.__dict__, **updates})


RESUME_CALLS = {"exporter-resume", "unit-resume", "log-collector-disable"}


def context(
    spec: Any,
    tmp_path: Path,
    activity_docs: Any,
    *,
    survey_value: dict[str, Any] | None = None,
    approvals=None,
) -> Any:
    """Activity returns the injected document until the producer is resumed on
    the node probe, then the recovered document; this mirrors the real timeline
    without depending on how many times the module polls the store."""

    node = node_probe(survey_value or survey())
    injected = activity_docs if isinstance(activity_docs, dict) else activity_docs[0]
    recovered = activity_docs[-1] if isinstance(activity_docs, list) else activity_docs

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        resumed = any(call and call[0] in RESUME_CALLS for call in node.calls)
        return dict(recovered if resumed else injected)

    regional = FakeRegional(cpu_python=responder)
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path, approvals=approvals),
        probes={f"node@{NODE}": node},
    )


def exporter_timeout_context(
    tmp_path: Path, paused: dict[str, Any], recovered: dict[str, Any]
) -> tuple[Any, list[bool]]:
    """``exporter_timeout`` over two activity shapes -- ``paused`` until the node
    probe resumed the exporter, ``recovered`` after -- recording for every
    store read whether the resume had already happened."""

    spec = SPECS["exporter_timeout"]
    node = node_probe(survey())
    reads: list[bool] = []

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        resumed = any(call and call[0] == "exporter-resume" for call in node.calls)
        reads.append(resumed)
        return dict(recovered if resumed else paused)

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        settings=settings(spec, tmp_path),
        probes={f"node@{NODE}": node},
    )
    return ctx, reads


#: The node's producer-side steps, in the order the silence variants take them;
#: a phased responder answers every store read with the document of the last
#: step the node probe has taken.
PHASE_CALLS = (
    "log-collector-enable",
    "unit-pause",
    "unit-resume",
    "log-collector-disable",
    "exporter-pause",
    "exporter-resume",
)
#: What the R13g node log collector's status carried live (2026-10-26, plain
#: EKS ip-10-0-32-162, GF-GENERAL-LIVE-031.expired_agent plan at 11:04:01Z):
#: b190's refusal, the cursor drop and the cumulative counters.
LIVE_REFUSAL_ERRORS = [
    "journalctl exited 1: Please specify only one of --since=, --cursor=, and --after-cursor=.",
    "journal cursor unavailable; next poll replays timestamp window",
    "cumulative loss since collector start: journal-window-capped-seconds=16779, journalctl-failures=72",
]
BASELINE_SUCCESS = "2026-10-24T00:00:00+00:00"
FIRST_SUCCESS = "2026-10-24T00:10:00+00:00"
RECOVERED_SUCCESS = "2026-10-24T01:00:00+00:00"


def status(
    channel: str,
    *,
    success: str | None,
    error: str | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "collector": channel,
        "last_success_at": success,
        "last_error_at": error,
        "errors": list(errors or []),
    }


def channel_doc(
    channel: str,
    *,
    state: str,
    success: str | None,
    reason: str | None = None,
    error: str | None = None,
    errors: list[str] | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    """The node's activity as one read sees it: the channel's status and its
    observation; ``silent`` adds the hour's advisory and the refresh workflow."""

    document = {
        **fresh_success_doc(channel),
        "metric_latest": [observation(channel, state, reason)],
        "collector_statuses": [
            status(channel, success=success, error=error, errors=errors)
        ],
    }
    if silent:
        document["workflows"] = silence_doc(channel)["workflows"]
        document["notifications"] = silence_doc(channel)["notifications"]
    return document


def logs_silent_phases(
    *, enabled: dict[str, Any] | None = None, resumed: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """The live timeline of a clean ``logs_silent``: a success hours old before
    the enable, the enabled collector's first success, the silence under the
    pause (the success time it had at the pause), a new success after the
    resume. ``enabled``/``resumed`` replace the two reads the product may fail."""

    channel = "NODE_LOGS"
    return {
        "baseline": channel_doc(channel, state="UNKNOWN", success=BASELINE_SUCCESS),
        "log-collector-enable": enabled
        or channel_doc(channel, state="HEALTHY", success=FIRST_SUCCESS),
        "unit-pause": channel_doc(
            channel, state="UNKNOWN", success=FIRST_SUCCESS, silent=True
        ),
        "unit-resume": resumed
        or channel_doc(channel, state="HEALTHY", success=RECOVERED_SUCCESS),
    }


def phased_context(
    spec: Any,
    tmp_path: Path,
    phases: dict[str, Any],
    *,
    approvals=None,
    survey_value: dict[str, Any] | None = None,
) -> Any:
    """Activity answers with ``phases[<last producer step>]`` (``"baseline"``
    before any); a phase value may be a callable taking the read's ordinal."""

    node = node_probe(survey_value or survey())
    reads = {"count": 0}

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        reads["count"] += 1
        phase = "baseline"
        for call in node.calls:
            if call and call[0] in PHASE_CALLS and call[0] in phases:
                phase = call[0]
        value = phases[phase]
        return dict(value(reads["count"]) if callable(value) else value)

    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        settings=settings(spec, tmp_path, approvals=approvals),
        probes={f"node@{NODE}": node},
    )


def probe_calls(ctx: Any) -> list[str]:
    return [call[0] for call in ctx.probes[f"node@{NODE}"].calls]


def test_exporter_timeout_passes(tmp_path: Path) -> None:
    """Live 2026-10-26 (eksb): the variant demanded a finished VALIDATE_HOST
    while the exporter was still paused, which the product refuses by design
    (it waits for a GPU_METRICS success newer than the finding). The proof is
    now ordered: paused -> incident + planned bounded chain + a validation that
    waits; resume; then the validation SUCCEEDED on the evidence it waited for.
    """

    spec = SPECS["exporter_timeout"]
    ctx, reads = exporter_timeout_context(
        tmp_path, dcgm_error_doc(), recovered_dcgm_doc()
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) == {
        "paused",
        "unknown",
        "chain",
        "validation",
        "hardware",
        "recovery",
    }, result["stages"]
    node = ctx.probes[f"node@{NODE}"]
    names = [call[0] for call in node.calls]
    assert names.count("exporter-resume") == 1, names
    assert names.index("exporter-pause") < names.index("exporter-resume"), names
    # The chain was read while paused and judged terminal only after the resume.
    assert reads[0] is False and reads[-1] is True, reads
    validation = result["validation"]
    assert validation["status"] == "SUCCEEDED", validation
    assert validation["finished_after_resume"] is True, validation
    assert validation["waited_seconds"] == 360, validation
    paused = validation["while_paused"]
    assert paused["status"] == "WAITING", paused
    assert paused["health_evidence_pending"] == "collector state is UNKNOWN", paused
    assert f"{NODE}/GPU_METRICS" in paused["missing_observations"], paused
    assert paused["step_waiting_slow"] is True, paused
    assert paused["step_waiting_seconds"] == 412, paused
    assert "GPU_METRICS" in paused["pending"], paused


def test_exporter_timeout_waits_for_the_validation_to_be_reached(
    tmp_path: Path,
) -> None:
    """The paused read is taken once VALIDATE_HOST has a record: a chain that
    only reached its snapshot is polled again, with the exporter still paused."""

    spec = SPECS["exporter_timeout"]
    node = node_probe(survey())
    before = [dcgm_error_doc(execution=None)] * 3 + [dcgm_error_doc()]
    served = iter(before)
    polls: list[str] = []

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if any(call and call[0] == "exporter-resume" for call in node.calls):
            polls.append("after")
            return recovered_dcgm_doc()
        polls.append("before")
        return dict(next(served, before[-1]))

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        settings=settings(spec, tmp_path),
        probes={f"node@{NODE}": node},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert polls.count("before") == 4, polls
    assert polls.index("after") == 4, polls


def test_exporter_timeout_fails_a_validation_passed_while_paused(
    tmp_path: Path,
) -> None:
    """A SUCCEEDED VALIDATE_HOST while the only GPU_METRICS source is paused is
    a stale success passing the observation validation (spec 025)."""

    spec = SPECS["exporter_timeout"]
    stale = dcgm_error_doc(
        execution=validation_execution("SUCCEEDED", updated_at=VALIDATION_STALE),
        workflow_status="SUCCEEDED",
        incident_state="RECOVERED",
    )
    ctx, _reads = exporter_timeout_context(tmp_path, stale, recovered_dcgm_doc())
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["paused"] == [
        "VALIDATE_HOST SUCCEEDED while the exporter was still paused: a stale "
        "success passed the observation validation"
    ], result["stages"]


def test_exporter_timeout_fails_a_validation_that_predates_the_resume(
    tmp_path: Path,
) -> None:
    """The terminal record is dated: a success stamped before the resume did
    not come from the evidence the resume produced."""

    spec = SPECS["exporter_timeout"]
    early = recovered_dcgm_doc(
        validation_execution("SUCCEEDED", updated_at=VALIDATION_STALE)
    )
    ctx, _reads = exporter_timeout_context(tmp_path, dcgm_error_doc(), early)
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["chain"] == [], result["stages"]
    assert len(result["stages"]["validation"]) == 1, result["stages"]
    assert "before the exporter was resumed" in result["stages"]["validation"][0]
    assert result["validation"]["finished_after_resume"] is False


def test_exporter_timeout_names_the_product_cap(tmp_path: Path) -> None:
    """A VALIDATE_HOST that ran into the executor's 600 s per-step cap before
    fresh evidence arrived is named as such, next to the chain's verdict."""

    spec = SPECS["exporter_timeout"]
    capped = recovered_dcgm_doc(CAP_EXECUTION, workflow_status="FAILED")
    ctx, _reads = exporter_timeout_context(tmp_path, dcgm_error_doc(), capped)
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["validation"] == [
        "VALIDATE_HOST failed on the product's 600 s per-step cap before fresh "
        "GPU_METRICS evidence arrived: step 2/VALIDATE_HOST stayed non-terminal "
        "for 600s, past the 600s per-step cap"
    ], result["stages"]
    assert any("ended FAILED" in error for error in result["stages"]["chain"]), result[
        "stages"
    ]
    assert result["validation"]["step_waiting_timeout_seconds"] == 600


def test_exporter_timeout_reports_a_validation_still_waiting(tmp_path: Path) -> None:
    """A validation that never finishes inside VALIDATION_BUDGET after the
    resume is reported with what it waits for -- the product's words."""

    spec = SPECS["exporter_timeout"]
    stuck = recovered_dcgm_doc(WAITING_EXECUTION, workflow_status="RUNNING")
    ctx, reads = exporter_timeout_context(tmp_path, dcgm_error_doc(), stuck)
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["chain"] == [
        "VALIDATE_HOST did not finish: still WAITING (collector state is UNKNOWN: "
        f"{NODE}/GPU_METRICS=COLLECTION_ERROR)"
    ], result["stages"]
    assert result["stages"]["validation"] == [], result["stages"]
    # The post-resume wait ran its 300 s budget on the fixture clock.
    assert reads.count(True) >= col.VALIDATION_BUDGET // 2, len(reads)


def test_exporter_timeout_without_a_chain_reads_once_after_the_resume(
    tmp_path: Path,
) -> None:
    """No incident inside DCGM_BUDGET: the resume is not followed by a
    validation wait -- one read, and both stages say the chain never came."""

    spec = SPECS["exporter_timeout"]
    empty = {**dcgm_error_doc(), "incidents": [], "workflows": []}
    ctx, reads = exporter_timeout_context(
        tmp_path, empty, {**recovered_dcgm_doc(), "incidents": [], "workflows": []}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["paused"] == [
        "no dcgm_field_completeness finding reached an incident"
    ], result["stages"]
    assert result["stages"]["chain"] == [
        "no dcgm_field_completeness finding reached an incident"
    ], result["stages"]
    # one activity read for the chain after the resume, then the fresh-success
    # read pair -- never the 300 s validation wait.
    assert reads.count(True) <= 4, reads


def test_exporter_timeout_budgets_follow_the_product(tmp_path: Path) -> None:
    """The deadman outlives the detection budget (720 s) by three minutes as
    before; the post-resume validation budget is half the product's 600 s
    per-step cap; the window covers detection + validation + a fresh success."""

    assert col.EXPORTER_DEADMAN == 900, col.EXPORTER_DEADMAN
    assert col.VALIDATION_BUDGET == 300, col.VALIDATION_BUDGET
    assert SPECS["exporter_timeout"].max_minutes == 20, SPECS["exporter_timeout"]
    ctx, _reads = exporter_timeout_context(
        tmp_path, dcgm_error_doc(), recovered_dcgm_doc()
    )
    SPECS["exporter_timeout"].execute(ctx)
    node = ctx.probes[f"node@{NODE}"]
    pause = next(call for call in node.calls if call[0] == "exporter-pause")
    assert pause[pause.index("--seconds") + 1] == "900", pause


def test_exporter_timeout_inconclusive_without_exporter(tmp_path: Path) -> None:
    spec = SPECS["exporter_timeout"]
    ctx = context(spec, tmp_path, dcgm_error_doc(), survey_value=survey(exporters=0))
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_exporter_timeout_fails_on_hardware_step(tmp_path: Path) -> None:
    spec = SPECS["exporter_timeout"]
    bad = dcgm_error_doc()
    bad["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    recovered = recovered_dcgm_doc()
    recovered["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    ctx, _reads = exporter_timeout_context(tmp_path, bad, recovered)
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("RESET_GPU" in error for error in result["stages"]["paused"]), result[
        "stages"
    ]
    assert any("RESET_GPU" in error for error in result["stages"]["hardware"]), result[
        "stages"
    ]


def test_sustained_unknown_and_post_refresh_pass(tmp_path: Path) -> None:
    for variant, docs in (
        ("sustained_unknown", dcgm_error_doc()),
        ("post_refresh_success", [dcgm_error_doc(), fresh_success_doc("GPU_METRICS")]),
    ):
        spec = SPECS[variant]
        # sustained_unknown ends by reading a fresh success too
        if variant == "sustained_unknown":
            docs = [
                dcgm_error_doc(),
                dcgm_error_doc(),
                fresh_success_doc("GPU_METRICS"),
            ]
        ctx = context(spec, tmp_path, docs)
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])


@pytest.mark.parametrize(
    "variant,channel",
    [
        ("inventory_silent", "GPU_INVENTORY"),
        ("metrics_silent", "GPU_METRICS"),
        ("host_silent", "HOST_TELEMETRY"),
    ],
)
def test_silence_variants_pass(tmp_path: Path, variant: str, channel: str) -> None:
    spec = SPECS[variant]
    ctx = context(
        spec,
        tmp_path,
        [silence_doc(channel), silence_doc(channel), fresh_success_doc(channel)],
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_logs_silent_requires_approval(tmp_path: Path) -> None:
    spec = SPECS["logs_silent"]
    ctx = context(spec, tmp_path, silence_doc("NODE_LOGS"), approvals=())
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_logs_silent_passes_with_approval(tmp_path: Path) -> None:
    """The live shape of a clean run: the enabled collector's first success is
    awaited (not held for), the pause's UNKNOWN is the silence's own
    STALE_OBSERVATION, the resume advances the success time again."""

    spec = SPECS["logs_silent"]
    ctx = phased_context(
        spec, tmp_path, logs_silent_phases(), approvals=("ENABLE_NODE_LOG_COLLECTOR",)
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["success_before_enable"] == BASELINE_SUCCESS, result
    assert result["first_success"]["last_success_at"] == FIRST_SUCCESS, result
    assert result["first_success"]["budget_seconds"] == 812, result["first_success"]
    assert result["success_before_resume"] == FIRST_SUCCESS, result
    assert result["recovery"]["last_success_at"] == RECOVERED_SUCCESS, result
    calls = probe_calls(ctx)
    assert calls.index("log-collector-enable") < calls.index("unit-pause"), calls
    assert calls.index("unit-resume") < calls.index("log-collector-disable"), calls
    timeline = ctx.case_dir / "log-first-success-timeline.json"
    assert timeline.is_file(), "the first-success wait leaves its timeline"


def test_logs_silent_awaits_the_first_success_instead_of_holding(
    tmp_path: Path,
) -> None:
    """Live 2026-10-26 (R13g): the enabled collector never succeeded in the 900 s
    the variant held blind, so the UNKNOWN it read came from a success five
    hours old. A channel that never succeeds after the enable is not paused --
    its silence would not be the pause's -- and the stage says what the
    product reported: b190's refusal with the failures counting up."""

    spec = SPECS["logs_silent"]

    def refusing(read: int) -> dict[str, Any]:
        errors = list(LIVE_REFUSAL_ERRORS)
        if read < 20:
            errors[-1] = errors[-1].replace(
                "journalctl-failures=72", "journalctl-failures=22"
            )
        return channel_doc(
            "NODE_LOGS",
            state="UNKNOWN",
            reason="STALE_OBSERVATION",
            success=BASELINE_SUCCESS,
            error="2026-10-24T00:05:00+00:00",
            errors=errors,
        )

    ctx = phased_context(
        spec,
        tmp_path,
        logs_silent_phases(enabled=refusing),
        approvals=("ENABLE_NODE_LOG_COLLECTOR",),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert list(result["stages"]) == ["first_success"], result["stages"]
    [error] = result["stages"]["first_success"]
    assert error.startswith(
        "NODE_LOGS did not report a new success after the enable within 812 s; "
        "collector alternates refusals: known product gap b190 "
        "(journalctl-failures 22->72: 'Please specify only one of --since=, "
        "--cursor=, and --after-cursor=')"
    ), error
    calls = probe_calls(ctx)
    assert "unit-pause" not in calls, calls
    assert calls[-1] == "log-collector-disable", calls
    assert any("was not paused" in note for note in result["notes"]), result["notes"]
    assert result["first_success"]["journalctl_failures"] == [22, 72], result
    assert result["first_success"]["latest_status"]["errors"] == LIVE_REFUSAL_ERRORS


def test_logs_silent_recovery_names_the_refusal_pattern(tmp_path: Path) -> None:
    """A resume after which no summary slot lands on a clean poll: still a FAIL,
    but the stage names the product gap instead of a bare "no new success"."""

    spec = SPECS["logs_silent"]
    refusing = channel_doc(
        "NODE_LOGS",
        state="UNKNOWN",
        reason="COLLECTION_ERROR",
        success=FIRST_SUCCESS,
        error="2026-10-24T00:40:00+00:00",
        errors=LIVE_REFUSAL_ERRORS,
    )
    ctx = phased_context(
        spec,
        tmp_path,
        logs_silent_phases(resumed=refusing),
        approvals=("ENABLE_NODE_LOG_COLLECTOR",),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["recovery"] == [
        "NODE_LOGS did not report a new success after recovery within 812 s; "
        "collector alternates refusals: known product gap b190 "
        "(journalctl-failures=72: 'Please specify only one of --since=, "
        "--cursor=, and --after-cursor=')"
    ], result["stages"]
    # The pause itself was judged: UNKNOWN for the silence, the advisory, the cap.
    assert result["stages"]["unknown"] == [], result["stages"]
    assert result["stages"]["advisory"] == [], result["stages"]
    assert result["recovery"]["budget_seconds"] == 812, result["recovery"]
    assert probe_calls(ctx)[-1] == "log-collector-disable", probe_calls(ctx)


def test_recovery_distinguishes_a_success_without_a_healthy_observation(
    tmp_path: Path,
) -> None:
    """The success time advanced but the silence scan never read the channel
    HEALTHY: a different failure from "no new success", said as such."""

    spec = SPECS["host_silent"]
    silent = silence_doc("HOST_TELEMETRY")
    advanced = channel_doc(
        "HOST_TELEMETRY",
        state="UNKNOWN",
        reason="COLLECTION_ERROR",
        success=RECOVERED_SUCCESS,
        error="2026-10-24T01:00:30+00:00",
        errors=["diskstats unreadable"],
    )
    ctx = context(spec, tmp_path, [silent, silent, advanced])
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["recovery"] == [
        f"HOST_TELEMETRY reported a new success at {RECOVERED_SUCCESS} after "
        "recovery but its observation never read HEALTHY within 817 s (latest "
        "{'state': 'UNKNOWN', 'reason': 'COLLECTION_ERROR'}); the channel's "
        "latest errors: ['diskstats unreadable']"
    ], result["stages"]


def test_silence_unknown_must_be_the_pauses_stale_observation(tmp_path: Path) -> None:
    """An UNKNOWN for COLLECTION_ERROR is a failing poll, not the pause: with
    b190 every other node-log poll fails, so a refused summary slot before the
    pause would otherwise pass the "unknown" stage 20 s into the SIGSTOP."""

    spec = SPECS["host_silent"]
    failing = {
        **silence_doc("HOST_TELEMETRY"),
        "metric_latest": [observation("HOST_TELEMETRY", "UNKNOWN", "COLLECTION_ERROR")],
    }
    ctx = context(
        spec, tmp_path, [failing, failing, fresh_success_doc("HOST_TELEMETRY")]
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["unknown"] == [
        "HOST_TELEMETRY observation is UNKNOWN for COLLECTION_ERROR, not the "
        "pause's STALE_OBSERVATION"
    ], result["stages"]
    assert result["stages"]["recovery"] == [], result["stages"]


def test_recovery_budget_follows_the_channels_cadence(tmp_path: Path) -> None:
    """Two posts at their longest (+-5 % phase move), the start-up spread and its
    random part, one poll, the silence scan and the retry ladder; read from the
    node's collector environment; never below the 600 s the wait had before."""

    spec = SPECS["host_silent"]
    ctx = context(spec, tmp_path, silence_doc("HOST_TELEMETRY"))
    ctx.survey = survey()
    assert col.recovery_budget(ctx, "NODE_LOGS") == 2 * 330 + 15 + 20 + 10 + 60 + 47
    assert col.recovery_budget(ctx, "NODE_LOGS") == 812
    assert (
        col.recovery_budget(ctx, "HOST_TELEMETRY") == 2 * 330 + 15 + 20 + 15 + 60 + 47
    )
    assert col.recovery_budget(ctx, "GPU_METRICS") == 600
    assert col.recovery_budget(ctx, "GPU_INVENTORY") == 600
    assert col.recovery_budget(ctx, "NVIDIA_KERNEL") == 600
    ctx.survey = {
        **survey(),
        "collector_env": {
            **survey()["collector_env"],
            "GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS": "600",
            "GPU_FAULT_LOG_INTERVAL_SECONDS": "30",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "120",
        },
    }
    assert col.recovery_budget(ctx, "NODE_LOGS") == 2 * 660 + 15 + 20 + 30 + 60 + 47
    assert col.recovery_budget(ctx, "HOST_TELEMETRY") == 600, "the floor holds"
    assert col.STARTUP_JITTER == 20 and col.DELIVERY_LADDER == 47
    assert col.SILENCE_SCAN == 60 and col.STARTUP_SPREAD == 15
    assert SPECS["logs_silent"].max_minutes == 50, SPECS["logs_silent"]


def test_journal_refusals_read_the_live_status() -> None:
    refused, failures = col.journal_refusals({"errors": LIVE_REFUSAL_ERRORS})
    assert (refused, failures) == (True, 72)
    assert col.journal_refusals({"errors": []}) == (False, None)
    assert col.journal_refusals(None) == (False, None)
    assert col.journal_refusals(
        {"errors": ["cumulative loss since collector start: journalctl-failures=3"]}
    ) == (False, 3)


def test_disabled_unit_and_expired_agent_and_new_success(tmp_path: Path) -> None:
    # These scripted records passed all three before (S59b): an empty window,
    # an Agent record that merely existed, a recovery without the silence. The
    # product-backed oracles are in test_general_live_collector_scope.py.
    spec = SPECS["disabled_unit"]
    doc = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "metric_latest": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    with pytest.raises(col.VariantInconclusive, match="does not report"):
        spec.execute(context(spec, tmp_path, doc))

    spec = SPECS["expired_agent"]
    ctx = context(spec, tmp_path, {**doc, "agent": {"lifecycle_state": "EXPIRED"}})
    with pytest.raises(VariantNotRun) as raised:
        spec.execute(ctx)
    assert raised.value.reason == "FIXTURE_NOT_READY", raised.value

    spec = SPECS["new_success"]
    docs = [
        {
            **silence_doc("HOST_TELEMETRY"),
            "collector_statuses": [
                {
                    "collector": "HOST_TELEMETRY",
                    "last_success_at": "2026-10-24T00:00:00+00:00",
                    "last_error_at": None,
                    "errors": [],
                }
            ],
        },
        silence_doc("HOST_TELEMETRY"),
        fresh_success_doc("HOST_TELEMETRY"),
    ]
    ctx = context(spec, tmp_path, docs)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_new_success_fails_when_the_paused_channel_never_read_unknown(
    tmp_path: Path,
) -> None:
    """The old variant discarded its UNKNOWN wait and passed on the recovery
    alone: a channel that stayed HEALTHY under the pause passed "leaves UNKNOWN
    only with a new success" without ever being UNKNOWN."""

    spec = SPECS["new_success"]
    still_healthy = {
        **fresh_success_doc("HOST_TELEMETRY"),
        "collector_statuses": [
            {
                "collector": "HOST_TELEMETRY",
                "last_success_at": "2026-10-24T00:00:00+00:00",
                "last_error_at": None,
                "errors": [],
            }
        ],
    }
    ctx = context(spec, tmp_path, [still_healthy, fresh_success_doc("HOST_TELEMETRY")])
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["recovery"] == [], result["stages"]
    (error,) = result["stages"]["unknown"]
    assert "expected UNKNOWN" in error, error


def test_capability_and_replay_variants_are_deferred() -> None:
    assert SPECS["inactive_cluster"].deferred.startswith("CAPABILITY_ABSENT"), (
        "SPECS['inactive_cluster'].deferred.startswith('CAPABILITY_ABSENT')"
    )
    assert SPECS["notification_failure"].deferred.startswith("CAPABILITY_ABSENT"), (
        "SPECS['notification_failure'].deferred.startswith('CAPABILITY_ABSENT')"
    )
    # The five DCGM field variants moved to the reviewed field replay
    # (general_live_gpu_fields); this module no longer carries them.
    for variant in (
        "missing_field",
        "one_gpu_missing",
        "conflicting_alias",
        "nan_negative",
        "duplicate_slot",
    ):
        assert variant not in SPECS, variant


def test_silence_advisory_must_name_the_silent_channel(tmp_path: Path) -> None:
    """The metrics unit also carries GPU_INVENTORY (180 s): its advisory comes
    first and must not stand in for the GPU_METRICS one (live 2026-10-25, the
    advisory stage passed on the inventory key while GPU_METRICS stayed fresh)."""

    spec = SPECS["metrics_silent"]
    doc = silence_doc("GPU_METRICS")
    doc["notifications"][0]["deduplication_key"] = (
        f"collector-silent/c/{NODE}/GPU_INVENTORY/1"
    )
    ctx = context(spec, tmp_path, [doc, doc, fresh_success_doc("GPU_METRICS")])
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert result["stages"]["advisory"] == [
        "no collector-silent advisory for GPU_METRICS"
    ], result["stages"]


def test_silence_recovery_needs_a_success_after_the_resume(tmp_path: Path) -> None:
    """Resuming the producer must advance a new success time; the success the
    channel already had before the resume proves nothing (spec 031)."""

    spec = SPECS["host_silent"]
    silent = silence_doc("HOST_TELEMETRY")
    unchanged = fresh_success_doc("HOST_TELEMETRY")
    unchanged["collector_statuses"][0]["last_success_at"] = silent[
        "collector_statuses"
    ][0]["last_success_at"]
    ctx = context(spec, tmp_path, [silent, silent, unchanged])
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    # Two 300 s host summaries at their longest + spread + poll + scan + ladder.
    assert result["stages"]["recovery"] == [
        "HOST_TELEMETRY did not report a new success after recovery within 817 s"
    ], result["stages"]
    assert result["recovery"]["budget_seconds"] == 817, result["recovery"]


def test_silence_accepts_the_hours_deduplicated_advisory(tmp_path: Path) -> None:
    """One advisory per collector per alert hour is the product's dedup: a
    retry inside the hour finds no new notification, only the hour's key."""

    spec = SPECS["host_silent"]
    silent = silence_doc("HOST_TELEMETRY")
    silent["notifications"] = []
    now = col.utc_now()
    bucket = int(now.timestamp() // col.SILENT_ALERT_INTERVAL)
    earlier = silence_doc("HOST_TELEMETRY")
    earlier["notifications"][0]["deduplication_key"] = (
        f"collector-silent/c/{NODE}/HOST_TELEMETRY/{bucket}"
    )
    earlier["notifications"][0]["created_at"] = (
        datetime.fromtimestamp(bucket * col.SILENT_ALERT_INTERVAL, tz=timezone.utc)
    ).isoformat()
    bucket_starts = {
        datetime.fromtimestamp(
            (bucket - shift) * col.SILENT_ALERT_INTERVAL, tz=timezone.utc
        ).isoformat()
        for shift in (0, 1)
    }
    node = node_probe(survey())
    recovered = fresh_success_doc("HOST_TELEMETRY")

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if any(call and call[0] in RESUME_CALLS for call in node.calls):
            return dict(recovered)
        return dict(earlier if arguments[2] in bucket_starts else silent)

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        settings=settings(spec, tmp_path),
        probes={f"node@{NODE}": node},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert any("deduplicated" in note for note in result["notes"]), result["notes"]


def test_pause_commands_ship_the_pause_module_once_before_the_first_pause(
    tmp_path: Path,
) -> None:
    """A probe ships as one file: the pause commands' module is installed
    sha256-checked once per run before the first pause (logs_silent: the log
    collector enable), never again for the unit pause that follows."""

    spec = SPECS["logs_silent"]
    ctx = phased_context(
        spec, tmp_path, logs_silent_phases(), approvals=("ENABLE_NODE_LOG_COLLECTOR",)
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    node = ctx.probes[f"node@{NODE}"]
    names = [call[0] for call in node.calls]
    assert names.count("install-source") == 1, names
    assert names.index("install-source") < names.index("log-collector-enable"), names
    assert names.index("log-collector-enable") < names.index("unit-pause"), names
    shipped = next(call for call in node.calls if call[0] == "install-source")
    content = PAUSE_MODULE.read_bytes()
    assert shipped[shipped.index("--name") + 1] == PAUSE_MODULE.name, shipped
    assert (
        shipped[shipped.index("--sha256") + 1] == hashlib.sha256(content).hexdigest()
    ), "the module is sha256-checked like the source replay's builders"
    assert base64.b64decode(shipped[shipped.index("--b64") + 1]) == content, (
        "the checkout's module is what ships"
    )
    assert "--run-id" in shipped, shipped
