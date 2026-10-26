"""GF-GENERAL-LIVE-019 injector/verdict contract over scripted records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in mem.VARIANTS}
GPU = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any, settings: Any = None):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = settings or make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def test_registry_covers_every_019_variant() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    rows = reg.coverage_rows()["GF-GENERAL-LIVE-019"]
    assert set(rows) == set(SPECS), set(rows) ^ set(SPECS)
    assert all(spec.proof_mode == "live-replay" for spec in mem.VARIANTS), (
        "every 019 variant is a live-replay"
    )


def test_corrected_ecc_is_a_clean_diagnostic_chain(tmp_path: Path) -> None:
    incident = gpu_incident(
        "inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", state="RECOVERED"
    )
    workflow = gpu_workflow(
        "wf-1",
        "inc-ecc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    activity = {**_activity([incident], [workflow]), "batches": [{"observed_at": "t"}]}
    probe = dcgm_probe(audit=audit_records({f"{GPU}/ecc_sbe_volatile_total": 1.0}))
    ctx = _ctx(
        tmp_path,
        "corrected_ecc",
        probe=probe,
        cpu=gpu_cpu_python(
            activity=activity,
            gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
        ),
    )
    out = mem.corrected_ecc(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["proof_scope"] == "live-replay"
    # the replay was deployed, armed and closed by the probe's one idempotent
    # close-run (restore before stop happens inside it), once.
    calls = [c[0] for c in probe.calls]
    assert calls.index("arm") < calls.index("close-run")
    assert calls.count("close-run") == 1
    assert "restore-collector" not in calls and "stop-proxy" not in calls


def test_diagnostic_fails_when_a_reset_is_planned(tmp_path: Path) -> None:
    incident = gpu_incident(
        "inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", action="RESET_GPU"
    )
    workflow = gpu_workflow(
        "wf-1", "inc-ecc", ["FREEZE_EVIDENCE", "RESET_GPU", "VALIDATE_GPU"]
    )
    activity = _activity([incident], [workflow])
    ctx = _ctx(
        tmp_path,
        "aggregate_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_aggregate_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity, gpu_state={"active_findings": []}),
    )
    out = mem.aggregate_dbe(ctx)
    assert out["verdict"] == "FAIL"
    assert any("RESET_GPU" in e for e in out["errors"]), out["errors"]


def test_inconclusive_when_the_replay_never_served(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    ctx = _ctx(
        tmp_path,
        "retired_delta",
        probe=dcgm_probe(audit=audit_records()),
        cpu=gpu_cpu_python(),
    )
    with pytest.raises(VariantInconclusive, match="never served"):
        mem.retired_delta(ctx)


def test_inconclusive_when_the_replay_aborted(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    records = audit_records(
        {f"{GPU}/ecc_sbe_volatile_total": 1.0}, aborted="real upstream change"
    )
    ctx = _ctx(
        tmp_path, "corrected_ecc", probe=dcgm_probe(audit=records), cpu=gpu_cpu_python()
    )
    with pytest.raises(VariantInconclusive, match="abort"):
        mem.corrected_ecc(ctx)


def test_volatile_dbe_quarantines_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = gpu_incident(
        "inc-dbe",
        f"b-{GPU}-ecc_dbe_volatile_total",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf-q",
    )
    workflow = gpu_workflow(
        "wf-q",
        "inc-dbe",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="BLOCKED",
    )
    activity = _activity([incident], [workflow])
    restored = {}
    monkeypatch.setattr(
        mem,
        "_validated_restore",
        lambda ctx, incidents: (restored.update(called=incidents), [])[1],
    )
    ctx = _ctx(
        tmp_path,
        "volatile_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_volatile_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
    )
    out = mem.volatile_dbe(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert restored["called"][0]["incident_id"] == "inc-dbe"


def test_volatile_dbe_fails_if_a_reset_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = gpu_incident(
        "inc-dbe",
        f"b-{GPU}-ecc_dbe_volatile_total",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf-q",
    )
    workflow = gpu_workflow(
        "wf-q", "inc-dbe", ["MARK_UNSCHEDULABLE", "QUARANTINE", "RESET_GPU"]
    )
    activity = _activity([incident], [workflow])
    monkeypatch.setattr(mem, "_validated_restore", lambda ctx, incidents: [])
    ctx = _ctx(
        tmp_path,
        "volatile_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_volatile_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
    )
    out = mem.volatile_dbe(ctx)
    assert out["verdict"] == "FAIL" and any("RESET_GPU" in e for e in out["errors"])


def test_pending_supported_requires_a_reset_chain(tmp_path: Path) -> None:
    spec = SPECS["pending_supported"]
    assert spec.approvals == ("GPU_RESET",) and spec.flags == "X" and spec.risk == "H"
    incident = gpu_incident(
        "inc-pend", f"b-{GPU}-row_remap_pending", action="RESET_GPU", workflow="wf-r"
    )
    workflow = gpu_workflow(
        "wf-r",
        "inc-pend",
        [
            "QUIESCE_GPU_SERVICES",
            "VERIFY_NO_GPU_CLIENTS",
            "RESET_GPU",
            "RESTORE_GPU_SERVICES",
            "VALIDATE_GPU",
        ],
        status="RUNNING",
    )
    activity = _activity([incident], [workflow])
    settings = make_settings(spec, node="ip-10-0-43-5.us-west-2.compute.internal")
    object.__setattr__(settings, "approvals", ("GPU_RESET",))
    ctx = _ctx(
        tmp_path,
        "pending_supported",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/row_remap_pending": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
        settings=settings,
    )
    out = mem.pending_supported(ctx)
    assert out["verdict"] == "PASS", out["errors"]


def test_pending_rejects_a_non_modern_gpu(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    probe = dcgm_probe(survey(model="NVIDIA L40S"))
    ctx = _ctx(tmp_path, "pending_supported", probe=probe, cpu=gpu_cpu_python())
    with pytest.raises(VariantNotRun, match="modern"):
        mem.pending_supported(ctx)


def test_deferred_variants_record_their_reason() -> None:
    for variant in ("pending_unsupported", "observe_profile"):
        spec = SPECS[variant]
        assert spec.deferred.startswith("CAPABILITY_ABSENT"), spec.deferred
        assert spec.execute is None


def _activity(incidents: list, workflows: list) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


# --------------------------------------------------------------------------- #
# remap_failure: the operator exit (live 2026-10-26, gpu-eks-cluster)
# --------------------------------------------------------------------------- #
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
REMAP_OPERATIONS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "RUN_FIELD_DIAGNOSTIC",
    "VALIDATE_GPU",
    "ESCALATE_SUPPORT",
]


def _isolated_node(incident_id: str) -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "node-uid",
        "boot_id": "boot-1",
        "ready": "True",
        "unschedulable": True,
        "taints": [
            {
                "key": "gpu-fault.io/quarantined",
                "value": "q-digest",
                "effect": "NoSchedule",
            }
        ],
        "ownership_annotations": {
            "gpu-fault.io/incident-id": incident_id,
            "gpu-fault.io/fencing-token": "1",
            "gpu-fault.io/previous-unschedulable": "false",
        },
        "spare_pool_annotations": {},
        "gpu_allocatable": "8",
    }


def _released_node() -> dict[str, Any]:
    return {
        **_isolated_node(""),
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }


class OperatorExitProduct:
    """A recording double of the CPU-Pod reads and the one write the exit makes.

    Scripts are matched by the fixture's own constants; the activity and GPU
    state probes fall through to ``gpu_cpu_python``. ``close`` scripts what the
    closure service answers; ``release_node`` makes the close release the
    node (what the live site showed); ``after_close`` overrides the incident
    and workflow documents the fresh reads return.
    """

    def __init__(
        self,
        regional: FakeRegional,
        incident: dict[str, Any],
        workflow: dict[str, Any],
        *,
        settled_state: str = "ESCALATED",
        close: dict[str, Any] | None = None,
        release_node: bool = True,
        after_close: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.regional = regional
        self.incident = {
            **incident,
            "cluster_id": "gpu-eks-cluster",
            "state": settled_state,
        }
        self.workflow = {**workflow, "blocked_kind": "NEEDS_OPERATOR"}
        self.close = close or {"closed": True, "refusal": None, "state": "RECOVERED"}
        self.release_node = release_node
        self.after_close = after_close or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.inner = gpu_cpu_python(activity=_activity([incident], [workflow]))

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        from scripts.e2e.regional import warm_spare_fixture as warm

        names = {
            warm.INCIDENT_CLEANUP_STATE: "settle",
            warm.CLOSE_INCIDENT_WITH_EVIDENCE: "close",
            warm.CLOSE_INCIDENT_POST: "close-post",
            warm.CREATE_RESTORE_WORKFLOW: "restore",
            warm.INCIDENT_BY_ID: "incident",
            warm.WORKFLOW_BY_ID: "workflow",
        }
        name = names.get(script)
        if name is None:
            return self.inner(script, *arguments)
        self.calls.append((name, arguments))
        if name == "settle":
            return {
                "incident": dict(self.incident),
                "workflow_status": self.workflow["status"],
                "active_workflows": [],
                "open_commands": [],
            }
        if name == "close":
            if self.close.get("closed"):
                self.incident = {
                    **self.incident,
                    **self.after_close.get("incident", {"state": "RECOVERED"}),
                }
                self.workflow = {
                    **self.workflow,
                    **self.after_close.get("workflow", {"status": "SUPERSEDED"}),
                }
                if self.release_node:
                    self.regional.node_snapshot_value = _released_node()
            return dict(self.close)
        if name == "incident":
            return dict(self.incident)
        if name == "workflow":
            return dict(self.workflow)
        raise AssertionError(f"the operator exit must never run {name}")


def _remap_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **product: Any):
    from scripts.e2e.regional import general_live_common as common
    from tests.regional._general_live_fakes import store_poll_clock

    incident = gpu_incident(
        "inc-remap",
        f"b-{GPU}-row_remap_failure",
        state="ESCALATED",
        action="DRAIN",
        workflow="wf-drain",
    )
    workflow = gpu_workflow("wf-drain", "inc-remap", REMAP_OPERATIONS, status="BLOCKED")
    regional = FakeRegional(node_snapshot=_isolated_node("inc-remap"))
    cpu = OperatorExitProduct(regional, incident, workflow, **product)
    regional.cpu_python_responder = cpu
    spec = SPECS["remap_failure"]
    settings = make_settings(spec, node=NODE)
    object.__setattr__(settings, "approvals", ("GPU_FIELD_DIAGNOSTIC",))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{NODE}"] = dcgm_probe(
        audit=audit_records({f"{GPU}/row_remap_failure": 1.0})
    )
    # The settle read is one quiet read in tests; live it is 15 s apart twice.
    monkeypatch.setattr(common, "OPERATOR_EXIT_QUIET_SECONDS", 0)
    store_poll_clock(ctx)
    return ctx, cpu


def test_remap_failure_closes_the_escalation_through_the_operator_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-10-26: the DRAIN chain ended ESCALATE_SUPPORT with the record
    BLOCKED/NEEDS_OPERATOR, the incident ESCALATED and the node cordoned; the
    validated restore was refused. The variant performs the operator close the
    runner used to make afterwards, and proves the release itself."""

    ctx, product = _remap_ctx(tmp_path, monkeypatch)
    out = mem.remap_failure(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    names = [name for name, _ in product.calls]
    assert "restore" not in names, "a validated restore is the wrong exit here"
    assert names.index("settle") < names.index("close") < names.index("incident"), (
        "settle, then close, then the fresh proof reads"
    )
    assert names.count("close") == 1, "a close is never repeated"
    (close,) = [args for name, args in product.calls if name == "close"]
    incident_id, reason, operator, reference, evidence_json = close
    assert incident_id == "inc-remap", close
    assert "GF-GENERAL-LIVE-019.remap_failure" in reason and "by design" in reason, (
        reason
    )
    assert operator == "acceptance-fixture", operator
    assert reference == ctx.run_id, reference
    evidence = json.loads(evidence_json)
    assert evidence[0]["node_id"] == NODE and evidence[0]["unschedulable"] is True, (
        "the evidence is the node as read before the close, cordoned by the drain"
    )
    assert evidence[0]["isolation_annotations"]["gpu-fault.io/incident-id"] == (
        "inc-remap"
    ), evidence
    (record,) = out["operator_exit"]
    assert record["state_before"] == "ESCALATED", record
    assert record["state_after"] == "RECOVERED", record
    assert record["workflow_status_after"] == "SUPERSEDED", record
    assert record["node_before"]["unschedulable"] is True, record
    assert record["node_after"] == {
        "ownership_annotations": {},
        "taints": [],
        "unschedulable": False,
    }, record
    assert any("operator exit" in note for note in out["notes"]), out["notes"]
    assert (ctx.case_dir / "operator-exit.json").is_file(), "the exit is evidence"


def test_remap_failure_refused_close_is_a_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal = (
        "incident inc-remap still has an open workflow wf-drain "
        "(BLOCKED/NEEDS_OPERATOR); wait for it to end or reconcile it"
    )
    ctx, product = _remap_ctx(
        tmp_path,
        monkeypatch,
        close={"closed": False, "refusal": refusal, "state": None},
    )
    out = mem.remap_failure(ctx)
    assert out["verdict"] == "FAIL", out
    assert out["stages"]["chain"] == [] and out["stages"]["hardware"] == [], out
    (error,) = out["stages"]["recovery"]
    assert error.startswith("operator close failed: RegionalFixtureError:"), error
    assert "refused" in error and "BLOCKED/NEEDS_OPERATOR" in error, error
    names = [name for name, _ in product.calls]
    assert names.count("close") == 1, "a refusal is never retried blindly"
    assert "incident" not in names and "workflow" not in names, (
        "nothing is proven after a refusal"
    )
    assert ctx.regional.node_snapshot(NODE)["unschedulable"] is True, (
        "the harness never releases the node by hand"
    )


def test_remap_failure_leftover_isolation_is_a_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import general_live_common as common

    ctx, product = _remap_ctx(tmp_path, monkeypatch, release_node=False)
    out = mem.remap_failure(ctx)
    assert out["verdict"] == "FAIL", out
    (error,) = out["stages"]["recovery"]
    assert "is not proven after" in error, error
    assert "gpu-fault.io/incident-id" in error, "the leftover annotation is named"
    assert "gpu-fault.io/quarantined" in error, "the leftover taint is named"
    assert "cordoned" in error, error
    assert "not RECOVERED" not in error, "the incident itself did close"
    reads = [name for name, _ in product.calls if name == "incident"]
    budget = common.OPERATOR_EXIT_PROOF_SECONDS // common.STORE_POLL_SECONDS
    assert budget <= len(reads) <= budget + 2, (
        f"the proof polls to its bound and stops: {len(reads)} reads"
    )
    timeline = json.loads((ctx.case_dir / "operator-exit-0-timeline.json").read_text())
    assert all(entry["accepted"] is False for entry in timeline["entries"]), timeline


def test_remap_failure_workflow_left_open_is_a_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, _product = _remap_ctx(
        tmp_path,
        monkeypatch,
        after_close={
            "workflow": {"status": "BLOCKED", "blocked_kind": "NEEDS_OPERATOR"}
        },
    )
    out = mem.remap_failure(ctx)
    assert out["verdict"] == "FAIL", out
    (error,) = out["stages"]["recovery"]
    assert "workflow wf-drain of inc-remap is still open (BLOCKED/NEEDS_OPERATOR)" in (
        error
    ), error


def test_remap_failure_incident_not_left_escalated_is_a_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, product = _remap_ctx(tmp_path, monkeypatch, settled_state="RECOVERED")
    out = mem.remap_failure(ctx)
    assert out["verdict"] == "FAIL", out
    (error,) = out["stages"]["recovery"]
    assert "settled RECOVERED, not ESCALATED" in error, error
    assert "nothing was left to close" in error, error
    assert [name for name, _ in product.calls if name == "close"] == [], (
        "an incident that is not ESCALATED is never closed by the variant"
    )


def test_operator_exit_problems_follow_the_products_open_workflow_rule() -> None:
    from scripts.e2e.regional.general_live_common import (
        operator_exit_problems,
        workflow_open,
    )

    recovered = {"incident_id": "inc-remap", "state": "RECOVERED"}
    released = _released_node()
    assert workflow_open({"status": "BLOCKED", "blocked_kind": "NEEDS_OPERATOR"}) is (
        True
    ), "an occupying BLOCKED record still gates the node"
    assert workflow_open({"status": "BLOCKED", "blocked_kind": "SAFETY_SETTLED"}) is (
        False
    ), "a settled safety plan holds nothing"
    assert workflow_open(
        {"status": "SUPERSEDED", "blocked_kind": "NEEDS_OPERATOR"}
    ) is (False), (
        "the close leaves the kind on the SUPERSEDED record; the status decides"
    )
    assert workflow_open({"status": "RUNNING"}) is True, "executable rows are open"
    assert workflow_open({"status": "???"}) is True, "an unknown status proves nothing"
    assert (
        operator_exit_problems(recovered, {"status": "SUPERSEDED"}, released) == []
    ), "RECOVERED, settled and released is the proven exit"
    assert operator_exit_problems(recovered, None, released) == [], (
        "an incident without a workflow has no record to settle"
    )
    problems = operator_exit_problems(
        {"incident_id": "inc-remap", "state": "ESCALATED"},
        {
            "request_id": "wf-drain",
            "status": "BLOCKED",
            "blocked_kind": "NEEDS_OPERATOR",
        },
        _isolated_node("inc-remap"),
    )
    assert problems[0] == "incident inc-remap is ESCALATED, not RECOVERED", problems
    assert "workflow wf-drain of inc-remap is still open (BLOCKED/NEEDS_OPERATOR)" in (
        problems
    ), problems
    assert any(
        "gpu-fault.io/fencing-token, gpu-fault.io/incident-id" in p for p in problems
    ), problems
    assert any("taints gpu-fault.io/quarantined" in p for p in problems), problems
    assert any("still cordoned" in p for p in problems), problems
