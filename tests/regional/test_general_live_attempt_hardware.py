"""GF-GENERAL-LIVE-029 ``hardware_chain_continues`` over fakes (no cluster, no node).

Pins the XID 46 catalog entry and its citation, runs the reviewed line through
the product's kernel-log normalizer, drives the seed probe over the product's
in-memory ledger, and runs the whole variant over scripted product records:
PASS on the spec'd chain, FAIL on each defect (hardware chain skipped --
ISO-007's shape --, restart run, wrong reason, quarantined node, contradicted
attempt records, missing notification, changed ledger, stale telemetry),
INCONCLUSIVE when the XID never reached the product, NOT_RUN before any
mutation for a code that is not reset grade, and the LIFO cleanup proven: the
kmsg window, the job and the seeded ledger are undone in that order even when
the variant dies midway, and a ledger row that is not provably gone raises.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from gpu_fault.store import NotFoundError
from scripts.e2e.regional import general_live_attempt_hardware as hw
from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
from scripts.e2e.regional.general_live_kernel_xid import XID_DECISION_PROBE, xid_message
from scripts.e2e.regional.general_live_restart_budget import BUDGET_PROBE
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE, ATTEMPT_PROBE
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    budget_ledger_store,
    budget_probe_run,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import dcgm_probe, survey

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "src" / "gpu_fault" / "data" / "nvidia-xid-catalog-610.generated.yaml"
SPEC = next(item for item in hw.VARIANTS if item.variant == hw.VARIANT)
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
WF = "wf-xid46"
INC = "inc-xid46"
EVENT = "kernel-log-kmsg-boot-900000064-xid-46-cluster-abc"
NOTIFICATION = "notification-budget"
RESET_AT = "2026-10-24T00:10:00+00:00"
AFTER_RESET = "2026-10-24T00:12:00+00:00"
BEFORE_RESET = "2026-10-24T00:05:00+00:00"


# --------------------------------------------------------------------------- #
# probes in-process over the product's in-memory store
# --------------------------------------------------------------------------- #
def probe_run(code: str, store: Any, arguments: Sequence[str]) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["probe", *arguments]),
        contextlib.redirect_stdout(output),
    ):
        exec(compile(code, "<general-live-probe>", "exec"), {"__name__": "__main__"})
    return json.loads(output.getvalue().splitlines()[-1])


# --------------------------------------------------------------------------- #
# the managed job stand-in
# --------------------------------------------------------------------------- #
class FakeJob:
    created: list[FakeJob] = []
    log: list[str] = []

    def __init__(
        self, ctx: Any, *, name: str, restart_budget: int = 1, gpu_count: int = 8
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.restart_budget = restart_budget
        self.gpu_count = gpu_count
        self.node = ctx.settings.node
        self.started = False
        self.cleaned = False
        self.refuse_cleanup: str | None = None
        FakeJob.created.append(self)

    def start(self) -> dict[str, Any]:
        self.started = True
        FakeJob.log.append("job-start")
        return {"pods": [{"name": "p", "uid": "uid-1", "phase": "Running"}]}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "restart_budget": self.restart_budget,
            "pods": [{"name": "p", "uid": "uid-1"}],
            "pod_uids": ["uid-1"],
        }

    def cleanup(self) -> dict[str, Any]:
        FakeJob.log.append("job-cleanup")
        if self.refuse_cleanup:
            raise RuntimeError(self.refuse_cleanup)
        self.cleaned = True
        return {"workload": {"deleted": {"pytorchjob": self.name}}, "prewarm": {}}


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJob.created = []
    FakeJob.log = []
    monkeypatch.setattr(hw, "ManagedTrainingJob", FakeJob)


# --------------------------------------------------------------------------- #
# the product's records, scripted
# --------------------------------------------------------------------------- #
def decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": EVENT,
        "xid": hw.HARDWARE_XID,
        "gpu_uuid": G1,
        "pci_bdf": "0000:02:00",
        "event_source": "KERNEL_LOG",
        "product": "NVIDIA H200",
        "observed_at": "2026-10-24T00:01:00+00:00",
        "source_boot_id": "boot-1",
        "decision": True,
        "disposition": "EXECUTABLE",
        "official_action": "RESET_GPU",
        "investigatory_action": "CONTACT_SUPPORT",
        "action": "RESET_GPU",
        "safety_action": None,
        "severity": "CRITICAL",
        "requires_operator": False,
        "incident_id": INC,
        "incident_state": "ACTION_PENDING",
        "workflow_request_id": WF,
        "investigatory_notification_id": None,
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": ["exact NVIDIA Catalog 610 Immediate Action for XID 46"],
    }
    base.update(overrides)
    return base


def chain_workflow(
    *,
    ops: Sequence[str] = hw.HARDWARE_CHAIN,
    status: str = "FAILED",
    reason: str | None = "restart budget exhausted for gpu-eks-cluster/job: 1/1",
    restart_status: str = "FAILED",
    restart_reason: str = hw.WITHHELD_REASON,
    restart_count: int = 1,
    restart_budget: int = 1,
    notification_id: str | None = NOTIFICATION,
    skip: Sequence[str] = (),
    display_restored: bool | None = None,
    superseded: Sequence[int] | None = None,
) -> dict[str, Any]:
    ops = list(ops)
    steps = [{"operation": op, "parameters": {}} for op in ops]
    executions: list[dict[str, Any]] = []
    for index, op in enumerate(ops):
        if op in skip:
            continue
        if op == "RESTART_WORKLOAD":
            details: dict[str, Any] = {
                "reason": restart_reason,
                "restart_count": restart_count,
                "restart_budget": restart_budget,
            }
            if notification_id:
                details["notification_id"] = notification_id
            if restart_status == "SUCCEEDED":
                details = {"restart_submitted": True}
            executions.append(
                {
                    "step_index": index,
                    "operation": op,
                    "status": restart_status,
                    "error": None if restart_status == "SUCCEEDED" else reason,
                    "details": details,
                    "started_at": AFTER_RESET,
                    "updated_at": AFTER_RESET,
                }
            )
            continue
        details = {}
        if op == "RESET_GPU":
            details["gpu_reset_busy_attempts"] = 1
            if display_restored is not None:
                details.update(
                    {
                        "display_stack_restored": display_restored,
                        "display_stack_remedy": "modprobe nvidia_drm",
                        "display_modules_missing": ["nvidia_drm"],
                        "display_drm_devices_missing": ["card1"],
                    }
                )
        executions.append(
            {
                "step_index": index,
                "operation": op,
                "status": "SUCCEEDED",
                "error": None,
                "details": details,
                "started_at": RESET_AT,
                "updated_at": RESET_AT,
            }
        )
    if superseded is None:
        superseded = (
            [ops.index("RESTART_WORKLOAD")]
            if "RESTART_WORKLOAD" in ops and restart_status == "FAILED"
            else []
        )
    return {
        "request_id": WF,
        "incident_id": INC,
        "status": status,
        "created_at": "2026-10-24T00:01:30+00:00",
        "official_steps": steps,
        "step_executions": executions,
        "terminal_failure_reason": reason,
        "superseded_step_indexes": list(superseded),
    }


def xid_incident(
    incident_id: str = INC, *, state: str = "ESCALATED", event_id: str = EVENT
) -> dict[str, Any]:
    return {
        "incident_id": incident_id,
        "event_id": event_id,
        "event_type": "XID",
        "state": state,
        "created_at": "2026-10-24T00:01:20+00:00",
        "workflow_request_id": WF,
        "node_ids": [NODE],
    }


def budget_notification(job_id: str) -> dict[str, Any]:
    return {
        "notification_id": NOTIFICATION,
        "incident_id": INC,
        "category": "RESTART_GUARD",
        "deduplication_key": f"{CLUSTER}/{job_id}/restart-budget-exhausted/1",
        "subject": "[ACTION REQUIRED] automatic restarts stopped",
        "created_at": AFTER_RESET,
        "status": "SENT",
    }


#: ``Scenario(workflow=DEFAULT)`` is the spec'd chain; ``workflow=None`` is a
#: product that minted no workflow at all.
DEFAULT: Any = object()


class Scenario:
    """A ``cpu_python`` for every probe the variant runs, over a real ledger."""

    def __init__(
        self,
        *,
        decisions: list[dict[str, Any]] | None = None,
        workflow: dict[str, Any] | None = DEFAULT,
        incidents: list[dict[str, Any]] | None = None,
        terminal: str | None = "STOPPED",
        completion: str | None = "NO_ACTION",
        containment: dict[str, Any] | None = None,
        notifications: list[dict[str, Any]] | None = None,
        batches: Sequence[str] = (AFTER_RESET,),
        active_findings: Sequence[str] = (),
        ledger_override: dict[str, Any] | None = None,
        decision_error: Exception | None = None,
        mutate_cleanup: Any = None,
    ) -> None:
        self.store = budget_ledger_store()
        self.decisions = [decision()] if decisions is None else decisions
        self.workflow = chain_workflow() if workflow is DEFAULT else workflow
        self.incidents = [xid_incident()] if incidents is None else incidents
        self.terminal = terminal
        self.completion = completion
        self.containment = containment
        self.notifications = notifications
        self.batches = list(batches)
        self.active_findings = list(active_findings)
        self.ledger_override = ledger_override
        self.decision_error = decision_error
        self.mutate_cleanup = mutate_cleanup
        self.calls: list[str] = []
        self.job_id: str | None = None

    def activity(self) -> dict[str, Any]:
        return {
            "incidents": list(self.incidents),
            "workflows": [self.workflow] if self.workflow else [],
            "batches": [],
            "markers": [],
            "commands": [],
            "notifications": [],
            "metric_latest": [],
            "collector_statuses": [],
            "agent": None,
        }

    def ledger(self, job_id: str) -> dict[str, Any] | None:
        if self.ledger_override is not None:
            return self.ledger_override
        try:
            return self.store.get_restart_budget(CLUSTER, job_id).model_dump(
                mode="json"
            )
        except NotFoundError:
            return None

    def attempt(self, arguments: Sequence[str]) -> dict[str, Any]:
        _cluster, job_id, attempt_id = arguments[:3]
        self.job_id = job_id
        entry: dict[str, Any] = {
            "event": {"terminal_status": self.terminal} if self.terminal else None,
            "decision": {"status": self.completion, "reason": "stopped by incident"}
            if self.completion
            else None,
            "containment_incident": self.containment,
            "containment_workflows": [],
            "recovery": [
                {
                    "incident": {
                        "incident_id": INC,
                        "job_id": job_id,
                        "state": "ESCALATED",
                    },
                    "workflow": self.workflow,
                }
            ]
            if self.workflow
            else [],
        }
        return {"attempts": {attempt_id: entry}, "restart_budget": self.ledger(job_id)}

    def gpu_state(self) -> dict[str, Any]:
        return {
            "latest": [],
            "active_findings": [
                {"canonical_name": name} for name in self.active_findings
            ],
            "findings": [],
            "xid_events": [],
            "batches": [{"observed_at": moment} for moment in self.batches],
            "inventory_records": [],
            "inventory_snapshot": None,
            "collector_statuses": [],
            "agent": None,
            "capabilities": {},
        }

    def cpu_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        if script == hw.SEED_PROBE:
            self.calls.append("seed")
            FakeJob.log.append("seed")
            return probe_run(hw.SEED_PROBE, self.store, arguments)
        if script == BUDGET_PROBE:
            self.calls.append(f"budget-{arguments[0]}")
            FakeJob.log.append(f"budget-{arguments[0]}")
            document = budget_probe_run(self.store, arguments)
            if self.mutate_cleanup is not None and arguments[0] == "cleanup":
                self.mutate_cleanup(document)
            return document
        if script == XID_DECISION_PROBE:
            self.calls.append("decisions")
            if self.decision_error is not None:
                raise self.decision_error
            return {"decisions": list(self.decisions)}
        if script == ACTIVITY_PROBE:
            self.calls.append("activity")
            return self.activity()
        if script == ATTEMPT_PROBE:
            self.calls.append("attempt")
            return self.attempt(arguments)
        if script == GPU_STATE_PROBE:
            self.calls.append("gpu-state")
            return self.gpu_state()
        if script == hw.INCIDENT_NOTIFICATIONS_PROBE:
            self.calls.append("notifications")
            if self.notifications is not None:
                return {"notifications": list(self.notifications)}
            return {"notifications": [budget_notification(str(self.job_id))]}
        raise AssertionError(f"unexpected probe: {script[:60]!r}")


def kernel_script(lines: int = 6) -> dict[str, Any]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}, "mode": "kmsg"},
        "arm": {"armed": True, "lines_written": lines},
        "status": {"injected": {"mode": "kmsg", "fixture_lines": lines}},
        "close-window": {"clean": True},
    }


def context(
    tmp_path: Path,
    scenario: Scenario,
    *,
    lines: int = 6,
    node_snapshot: dict[str, Any] | None = None,
) -> Any:
    regional = FakeRegional(cpu_python=scenario.cpu_python, node_snapshot=node_snapshot)
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(SPEC, node=NODE),
    )
    ctx.probes[f"dcgm@{NODE}"] = dcgm_probe()
    ctx.probes[f"kernel-replay-kmsg@{NODE}"] = FakeProbe(kernel_script(lines))
    return ctx


def kernel_calls(ctx: Any) -> list[str]:
    return [call[0] for call in ctx.probes[f"kernel-replay-kmsg@{NODE}"].calls]


def seeded_job_id(ctx: Any) -> str:
    return hw.ExhaustedRestartBudget(ctx).job_id


# --------------------------------------------------------------------------- #
# registry and catalog pins
# --------------------------------------------------------------------------- #
def test_registry_binds_the_variant_to_the_hardware_module() -> None:
    spec = reg.load_registry()["GF-GENERAL-LIVE-029.hardware_chain_continues"]
    assert spec.implemented, "the variant is no longer deferred"
    assert spec.execute is hw.hardware_chain_continues, "spec.execute is the module's"
    assert spec.proof_mode == reg.LIVE_REPLAY
    assert spec.lanes == ("eks", "hp")
    assert (spec.flags, spec.risk, spec.max_minutes) == ("X", "H", 60)
    assert spec.approvals == ("GPU_RESET", "KERNEL_SOURCE_REPLAY")
    assert spec.confirmation == "GENERALLIVE029_EXECUTE"
    assert hw.__name__ in reg.CATEGORY_MODULES, "the module is a category module"
    assert "hardware_chain_continues" not in {item.variant for item in att.VARIANTS}, (
        "the attempts module no longer carries the deferred entry"
    )
    assert "hardware_chain_continues" in reg.APPROVALS["GPU_RESET"], (
        "the GPU_RESET token names this branch"
    )


def test_xid_46_is_reset_grade_in_the_pinned_catalog_and_the_citation_holds() -> None:
    entry = hw.require_reset_grade(hw.HARDWARE_XID)
    assert entry.xid == 46
    assert entry.mnemonic == "ROBUST_CHANNEL_GPU_TIMEOUT_ERROR"
    assert entry.immediate_action == "RESET_GPU"
    assert entry.investigatory_action == "CONTACT_SUPPORT"
    assert "H100" in entry.products, "the site's H200 is the H100 catalog family"
    assert hw.reset_grade_errors(entry, xid=46) == []
    cited = CATALOG.read_text(encoding="utf-8").splitlines()[461:473]
    assert cited[0].strip() == "- xid: 46", cited[0]
    assert "mnemonic: ROBUST_CHANNEL_GPU_TIMEOUT_ERROR" in cited[1], cited[1]
    assert any("immediateAction: RESET_GPU" in line for line in cited), cited
    assert any("- H100" in line for line in cited), cited
    assert "lines 462-473" in hw.CATALOG_CITATION, hw.CATALOG_CITATION


@pytest.mark.parametrize(
    ("xid", "fragment"),
    [
        (137, "Immediate Action is IGNORE"),  # monitor grade: lane K's, not a reset
        (95, "pre-actions"),  # RESET_GPU with ALL_APPLICATIONS pre-actions
        (79, "PCIe"),  # joins the PCIe replay composite
        (48, "resolver"),  # its own workflow
        (155, "does not apply"),  # B100/GB200 only
        (987, "absent"),  # unknown code -> SITE_SAFETY quarantine
    ],
)
def test_other_codes_are_refused(xid: int, fragment: str) -> None:
    from scripts.e2e.regional.general_live_kernel_xid import catalog_entry

    errors = hw.reset_grade_errors(catalog_entry(xid), xid=xid)
    assert any(fragment in error for error in errors), errors
    with pytest.raises(VariantNotRun) as info:
        hw.require_reset_grade(xid)
    assert info.value.reason == "FIXTURE_NOT_READY"


def test_reviewed_line_parses_through_the_product_normalizer() -> None:
    message = xid_message(
        46, "0000:02:00", run_id="gl029-hc-a1-deadbeef", label=hw.LABEL
    )
    batch = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id=CLUSTER,
            node_id=NODE,
            record_id="kmsg-boot-900000064",
            observed_at=datetime.now(timezone.utc),
            message=message,
        )
    )
    assert [event.xid for event in batch.xid_events] == [46]
    assert batch.xid_events[0].pci_bdf == "0000:02:00"
    assert batch.sxid_events == []


def test_target_gpu_picks_the_highest_index_gpu_the_line_can_name() -> None:
    entry = hw.require_reset_grade(46)
    assert hw.target_gpu(GpuSurvey(survey()), entry) == (G1, "0000:02:00")
    with pytest.raises(VariantNotRun) as info:
        hw.target_gpu(GpuSurvey(survey(model="Tesla V100-SXM2-32GB")), entry)
    assert info.value.reason == "CAPABILITY_ABSENT"
    raw = survey()
    for gpu in raw["exporter"]["gpus"].values():
        gpu["pci_bdf"] = ""
    with pytest.raises(VariantNotRun, match="PCI BDF"):
        hw.target_gpu(GpuSurvey(raw), entry)


# --------------------------------------------------------------------------- #
# the seed probe over the product's ledger API
# --------------------------------------------------------------------------- #
def test_seed_probe_exhausts_the_ledger_through_the_product_api() -> None:
    store = budget_ledger_store()
    arguments = [
        "seed",
        CLUSTER,
        "gl29hc-deadbeef-a1",
        "1",
        "gl29hc-deadbeef-a1-workflow",
    ]
    document = probe_run(hw.SEED_PROBE, store, arguments)
    expected = ["gl29hc-deadbeef-a1-workflow/0/RESTART_WORKLOAD"]
    assert hw.seed_errors(document, budget=1, reservation_ids=expected) == []
    assert document["ledger"]["before"] is None
    assert document["ledger"]["after"]["restart_count"] == 1
    assert document["exhausted_check"]["reserved"] is False, (
        "the product refuses a restart past the seeded budget"
    )
    state = store.get_restart_budget(CLUSTER, "gl29hc-deadbeef-a1")
    assert (state.budget, state.restart_count, list(state.reservation_ids)) == (
        1,
        1,
        expected,
    )
    again = probe_run(hw.SEED_PROBE, store, arguments)
    assert "preexisting ledger" in again["refused"], again
    assert "refused" in hw.seed_errors(again, budget=1, reservation_ids=expected)[0]
    with pytest.raises(ValueError, match="test prefix"):
        probe_run(hw.SEED_PROBE, store, ["seed", CLUSTER, "business-job", "1", "wf"])
    with pytest.raises(ValueError, match="at least 1"):
        probe_run(hw.SEED_PROBE, store, ["seed", CLUSTER, "gl29hc-x-a1", "0", "wf"])
    with pytest.raises(ValueError, match="unknown seed probe mode"):
        probe_run(hw.SEED_PROBE, store, ["drain", CLUSTER, "gl29hc-x-a1", "1", "wf"])


def test_seed_probe_cleanup_is_the_proven_budget_probe_cleanup() -> None:
    store = budget_ledger_store()
    probe_run(hw.SEED_PROBE, store, ["seed", CLUSTER, "gl29hc-aa-a1", "2", "wf"])
    cleaned = budget_probe_run(store, ["cleanup", CLUSTER, "gl29hc-aa-a1", "wf"])
    assert cleaned["released"] == ["wf/0/RESTART_WORKLOAD", "wf/1/RESTART_WORKLOAD"]
    assert cleaned["clean"] is True, cleaned
    with pytest.raises(NotFoundError):
        store.get_restart_budget(CLUSTER, "gl29hc-aa-a1")


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d["reservations"][0].update(reserved=False), "was refused"),
        (lambda d: d["ledger"]["after"].update(restart_count=2), "expected 1/1"),
        (lambda d: d["ledger"]["after"].update(reservation_ids=["other"]), "seed ids"),
        (
            lambda d: d["exhausted_check"].update(reserved=True),
            "past the seeded budget",
        ),
        (lambda d: d["ledger"].update(after=None), "no ledger row"),
        (lambda d: d.update(reservations=[]), "0 seed reservations"),
    ],
)
def test_seed_errors_name_each_defect(mutate: Any, fragment: str) -> None:
    store = budget_ledger_store()
    document = probe_run(
        hw.SEED_PROBE, store, ["seed", CLUSTER, "gl29hc-bb-a1", "1", "wf"]
    )
    mutate(document)
    errors = hw.seed_errors(
        document, budget=1, reservation_ids=["wf/0/RESTART_WORKLOAD"]
    )
    assert any(fragment in error for error in errors), errors


def test_notifications_probe_lists_the_incident_mails() -> None:
    from gpu_fault.models import (
        AdvisoryNotification,
        NotificationResult,
        NotificationStatus,
    )
    from gpu_fault.store import InMemoryStore

    store = InMemoryStore()
    saved = store.save_notification_if_absent(
        AdvisoryNotification(
            deduplication_key=f"{CLUSTER}/gl29hc-cc-a1/restart-budget-exhausted/1",
            cluster_name=CLUSTER,
            incident_id=INC,
            subject="[ACTION REQUIRED] restarts stopped",
            body_text="budget",
            support_case_draft="resubmit",
        )
    )
    store.save_notification_if_absent(
        AdvisoryNotification(
            deduplication_key="other/key",
            cluster_name=CLUSTER,
            incident_id="inc-other",
            subject="other",
            body_text="other",
            support_case_draft="other",
        )
    )
    store.save_notification_result(
        NotificationResult(
            notification_id=saved.notification_id, status=NotificationStatus.SENT
        )
    )
    out = probe_run(hw.INCIDENT_NOTIFICATIONS_PROBE, store, [CLUSTER, INC])
    assert [item["notification_id"] for item in out["notifications"]] == [
        saved.notification_id
    ]
    assert out["notifications"][0]["status"] == "SENT"
    assert (
        hw.notification_errors(
            out["notifications"], notification_id=saved.notification_id, incident_id=INC
        )
        == []
    )


# --------------------------------------------------------------------------- #
# the whole variant over scripted product records
# --------------------------------------------------------------------------- #
def test_hardware_chain_passes_and_cleans_up_lifo(tmp_path: Path) -> None:
    scenario = Scenario()
    ctx = context(tmp_path, scenario)
    out = hw.hardware_chain_continues(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {
        "decision",
        "chain",
        "withheld",
        "iso007_contrast",
        "incident",
        "notification",
        "attempt",
        "gpu_health",
        "display_stack",
    }
    assert (out["xid"], out["gpu_uuid"], out["pci_bdf"]) == (46, G1, "0000:02:00")
    assert out["proof_scope"] == "live-replay" and out["hardware_actions_real"] is True
    assert (
        out["workflow_status"] == "FAILED"
        and "budget" in out["terminal_failure_reason"]
    )
    assert out["gpu_reset_busy_attempts"] == 1
    assert out["job_id"] == seeded_job_id(ctx) and out["job_id"].startswith("gl29hc-")
    assert out["seed_reservation_ids"] == [
        f"{out['job_id']}-workflow/0/RESTART_WORKLOAD"
    ]
    assert out["kernel_replay"]["mode"] == "kmsg" and out["kernel_replay"]["xid"] == 46
    # The seed came first, then the job, then the fault; the window closed as
    # soon as the decision existed, before the chain was read.
    assert FakeJob.log[:2] == ["seed", "job-start"], FakeJob.log
    assert kernel_calls(ctx) == [
        "install-records",
        "open-window",
        "arm",
        "status",
        "close-window",
    ]
    opened = next(
        c
        for c in ctx.probes[f"kernel-replay-kmsg@{NODE}"].calls
        if c[0] == "open-window"
    )
    assert "--approved" in opened and "kmsg" in opened
    assert scenario.calls.index("decisions") < scenario.calls.index("activity")
    (job,) = FakeJob.created
    assert job.restart_budget == 1 and job.started, "the job ran with budget 1"
    # LIFO: the window, the job, then the seeded ledger.
    job_name = f"job-{out['job_id']}"
    budget_name = f"budget-{out['job_id']}"
    assert ctx.cleanup.pending() == [budget_name, job_name, "kernel-replay-kmsg"]
    results = ctx.cleanup.run()
    assert [item["name"] for item in results] == [
        "kernel-replay-kmsg",
        job_name,
        budget_name,
    ]
    assert all(item["ok"] for item in results), results
    assert job.cleaned, "the job was deleted"
    assert kernel_calls(ctx).count("close-window") == 1, (
        "the early close is not repeated by the stack"
    )
    with pytest.raises(NotFoundError):
        scenario.store.get_restart_budget(CLUSTER, out["job_id"])
    assert (ctx.case_dir / "budget-seed.json").is_file(), "seed receipt in evidence"
    assert (ctx.case_dir / "hardware-chain.json").is_file(), "chain evidence"
    assert (ctx.case_dir / "attempt-records.json").is_file(), "attempt evidence"
    assert (ctx.case_dir / "incident-notifications.json").is_file(), "mail evidence"


def test_fails_when_the_hardware_chain_is_skipped_iso007_shape(tmp_path: Path) -> None:
    # ISO-007's shape: the workflow FAILED on the budget with the single FAILED
    # restart step, but no reset ever ran (a RESTART_APP-style stop/restart).
    scenario = Scenario(
        workflow=chain_workflow(
            ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD")
        )
    )
    out = hw.hardware_chain_continues(context(tmp_path, scenario))
    assert out["verdict"] == "FAIL"
    assert any(
        "ISO-007" in error and "RESET_GPU" in error for error in out["errors"]
    ), out["errors"]
    assert any("RESET_GPU has no receipt" in error for error in out["errors"]), out[
        "errors"
    ]


def test_fails_when_restart_workload_runs(tmp_path: Path) -> None:
    scenario = Scenario(
        workflow=chain_workflow(
            status="SUCCEEDED", reason=None, restart_status="SUCCEEDED"
        )
    )
    out = hw.hardware_chain_continues(context(tmp_path, scenario))
    assert out["verdict"] == "FAIL"
    assert any("succeeded despite the exhausted budget" in e for e in out["errors"]), (
        out["errors"]
    )
    assert any("ended SUCCEEDED" in error for error in out["errors"]), out["errors"]


@pytest.mark.parametrize(
    ("workflow", "fragment"),
    [
        (
            chain_workflow(restart_reason="ADAPTER_REFUSED"),
            "expected RESTART_BUDGET_EXHAUSTED",
        ),
        (chain_workflow(restart_count=0, restart_budget=0), "expected 1/1"),
        (
            chain_workflow(notification_id=None),
            "names no BUDGET_EXHAUSTED notification",
        ),
        (chain_workflow(superseded=[]), "not superseded"),
        (chain_workflow(skip=("STOP_WORKLOADS",)), "STOP_WORKLOADS has no receipt"),
        (chain_workflow(skip=("VALIDATE_GPU",)), "VALIDATE_GPU has no receipt"),
        (
            chain_workflow(skip=("RESTORE_SCHEDULING",)),
            "RESTORE_SCHEDULING was planned",
        ),
        (chain_workflow(status="RUNNING"), "still RUNNING"),
        (chain_workflow(reason="stop rewrite"), "does not name the restart budget"),
        (
            chain_workflow(ops=(*hw.HARDWARE_CHAIN[:-1],)),
            "compiled no RESTART_WORKLOAD",
        ),
        (chain_workflow(display_restored=False), "display stack"),
    ],
)
def test_fails_on_each_chain_defect(
    tmp_path: Path, workflow: dict[str, Any], fragment: str
) -> None:
    out = hw.hardware_chain_continues(context(tmp_path, Scenario(workflow=workflow)))
    assert out["verdict"] == "FAIL"
    assert any(fragment in error for error in out["errors"]), out["errors"]


def test_forbidden_operations_and_remote_restart_commands_fail() -> None:
    workflow = chain_workflow(ops=(*hw.HARDWARE_CHAIN, "RESTART_NODE"))
    errors = hw.chain_receipt_errors(workflow)
    assert any("RESTART_NODE" in error for error in errors), errors
    commands = [{"command_id": "c-1", "step": {"operation": "RESTART_WORKLOAD"}}]
    errors = hw.withheld_errors(chain_workflow(), commands, budget=1)
    assert any("remote command c-1" in error for error in errors), errors
    batched = [
        {
            "command_id": "c-2",
            "step": {"operation": "RESET_GPU"},
            "batched_steps": [{"step": {"operation": "RESTART_WORKLOAD"}}],
        }
    ]
    assert hw.withheld_errors(chain_workflow(), batched, budget=1), "batched restart"
    assert hw.withheld_errors(chain_workflow(), [], budget=1) == []


def test_fails_when_the_incident_is_quarantined_and_restores_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored: list[list[dict[str, Any]]] = []

    def restore(_ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        restored.append(list(incidents))
        return []

    monkeypatch.setattr(hw, "restore_incidents", restore)
    scenario = Scenario(incidents=[xid_incident(state="QUARANTINED")])
    isolated = {
        "name": NODE,
        "uid": "node-uid",
        "boot_id": "boot-1",
        "ready": "True",
        "unschedulable": True,
        "taints": [{"key": "gpu-fault.io/quarantine", "effect": "NoSchedule"}],
        "ownership_annotations": {"gpu-fault.io/incident-id": INC},
        "spare_pool_annotations": {},
        "gpu_allocatable": "8",
    }
    out = hw.hardware_chain_continues(
        context(tmp_path, scenario, node_snapshot=isolated)
    )
    assert out["verdict"] == "FAIL"
    assert any("QUARANTINED" in error for error in out["errors"]), out["errors"]
    assert out["stages"]["restore"] == [], "the validated restore was handed the node"
    assert restored == [[{"incident_id": INC}]]


def test_escalated_incident_with_an_isolated_node_is_a_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hw, "restore_incidents", lambda *_a, **_k: [])
    isolated = {**FakeRegional().node_snapshot(""), "unschedulable": True}
    out = hw.hardware_chain_continues(
        context(tmp_path, Scenario(), node_snapshot=isolated)
    )
    assert out["verdict"] == "FAIL"
    assert any("still isolated" in error for error in out["errors"]), out["errors"]
    assert "restore" in out["stages"], "an isolated node is always handed back"


def test_fails_when_a_second_incident_or_decision_defect_appears(
    tmp_path: Path,
) -> None:
    out = hw.hardware_chain_continues(
        context(
            tmp_path,
            Scenario(
                incidents=[
                    xid_incident(),
                    xid_incident("inc-dup", event_id=EVENT + "-2"),
                ]
            ),
        )
    )
    assert out["verdict"] == "FAIL"
    assert any("further incident" in error for error in out["errors"]), out["errors"]
    for overrides, fragment in [
        (
            {
                "disposition": "SITE_SAFETY",
                "safety_action": "QUARANTINE",
                "action": None,
            },
            "safety action",
        ),
        ({"official_action": "RESTART_APP"}, "official action"),
        ({"gpu_uuid": G0}, "resolved to GPU"),
        ({"event_source": "DCGM_XID_GAUGE"}, "KERNEL_LOG"),
        ({"correlated_event_id": "e-2"}, "correlated"),
        ({"duplicate": True}, "duplicate"),
        ({"decision": False}, "no FaultPolicyDecision"),
    ]:
        out = hw.hardware_chain_continues(
            context(tmp_path, Scenario(decisions=[decision(**overrides)]))
        )
        assert out["verdict"] == "FAIL", (overrides, out["errors"])
        assert any(fragment in error for error in out["errors"]), (
            fragment,
            out["errors"],
        )


def test_fails_when_the_decision_minted_no_workflow(tmp_path: Path) -> None:
    scenario = Scenario(
        decisions=[decision(workflow_request_id=None, incident_state="RECOVERED")],
        workflow=None,
        incidents=[xid_incident(state="RECOVERED")],
    )
    out = hw.hardware_chain_continues(context(tmp_path, scenario))
    assert out["verdict"] == "FAIL"
    assert any("hardware chain was skipped" in error for error in out["errors"]), out[
        "errors"
    ]
    assert any("not in the product's records" in error for error in out["errors"]), out[
        "errors"
    ]


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        (
            {"terminal": "FAILED", "containment": {"incident_id": "inc-p"}},
            "application failure",
        ),
        ({"terminal": "FAILED"}, "expected STOPPED"),
        ({"terminal": None}, "expected STOPPED"),
        ({"completion": "PLAN_CREATED"}, "NO_ACTION"),
        ({"notifications": []}, "0 restart-budget-exhausted notifications"),
        (
            {"notifications": [{**budget_notification("j"), "notification_id": "n-x"}]},
            "is not saved for incident",
        ),
        (
            {
                "ledger_override": {
                    "restart_count": 0,
                    "reservation_ids": [],
                    "budget": 1,
                }
            },
            "reserves nothing",
        ),
        ({"ledger_override": None, "batches": (BEFORE_RESET,)}, "did not come back"),
        ({"active_findings": ("ecc_dbe_volatile_total",)}, "active GPU findings"),
    ],
)
def test_fails_on_attempt_notification_ledger_and_telemetry_defects(
    tmp_path: Path, options: dict[str, Any], fragment: str
) -> None:
    out = hw.hardware_chain_continues(context(tmp_path, Scenario(**options)))
    assert out["verdict"] == "FAIL", out["errors"]
    assert any(fragment in error for error in out["errors"]), (fragment, out["errors"])


def test_attempt_errors_catch_a_restart_of_another_job() -> None:
    records = {
        "attempts": {
            "j-a001": {
                "event": {"terminal_status": "STOPPED"},
                "decision": {"status": "NO_ACTION"},
                "containment_incident": None,
                "recovery": [
                    {
                        "incident": {"incident_id": "inc-x", "job_id": "other-job"},
                        "workflow": chain_workflow(
                            status="SUCCEEDED", reason=None, restart_status="SUCCEEDED"
                        ),
                    }
                ],
            }
        },
        "restart_budget": {"restart_count": 1, "reservation_ids": ["r"], "budget": 1},
    }
    errors = hw.attempt_errors(
        records, job_id="j", attempt_id="j-a001", budget=1, reservation_ids=["r"]
    )
    assert any("another job" in error for error in errors), errors
    assert any("succeeded in workflow" in error for error in errors), errors
    records["restart_budget"] = None
    errors = hw.attempt_errors(
        records, job_id="j", attempt_id="j-a001", budget=1, reservation_ids=["r"]
    )
    assert any("gone before cleanup" in error for error in errors), errors


def test_inconclusive_when_the_xid_never_reached_the_product(tmp_path: Path) -> None:
    with pytest.raises(VariantInconclusive, match="no reviewed XID 46"):
        hw.hardware_chain_continues(context(tmp_path, Scenario(), lines=0))
    assert FakeJob.created and FakeJob.created[0].started, (
        "the job ran before the fault"
    )
    scenario = Scenario(decisions=[])
    ctx = context(tmp_path / "b", scenario)
    with pytest.raises(VariantInconclusive, match="never reached the product"):
        hw.hardware_chain_continues(ctx)
    assert kernel_calls(ctx)[-1] == "close-window", "the window closed before raising"
    assert "activity" not in scenario.calls, "no chain was graded"


def test_inconclusive_when_the_seed_did_not_exhaust(tmp_path: Path) -> None:
    scenario = Scenario()
    ctx = context(tmp_path, scenario)
    real = scenario.cpu_python

    def granting(script: str, *arguments: str, **options: Any) -> dict[str, Any]:
        document = real(script, *arguments, **options)
        if script == hw.SEED_PROBE:
            document["exhausted_check"]["reserved"] = True
        return document

    ctx.regional.cpu_python_responder = granting
    with pytest.raises(
        VariantInconclusive, match="exhausted ledger was not established"
    ):
        hw.hardware_chain_continues(ctx)
    assert FakeJob.created == [], "no job was started on an unproven seed"
    assert ctx.cleanup.pending() == [f"budget-{seeded_job_id(ctx)}"]


def test_not_run_before_any_mutation_for_a_code_that_is_not_reset_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hw, "HARDWARE_XID", 137)
    scenario = Scenario()
    ctx = context(tmp_path, scenario)
    with pytest.raises(VariantNotRun) as info:
        hw.hardware_chain_continues(ctx)
    assert info.value.reason == "FIXTURE_NOT_READY"
    assert scenario.calls == [] and FakeJob.created == [], "nothing was touched"
    assert kernel_calls(ctx) == [] and ctx.cleanup.pending() == []


def test_seed_refuses_a_preexisting_row_as_not_run_and_cleanup_removes_it(
    tmp_path: Path,
) -> None:
    scenario = Scenario()
    ctx = context(tmp_path, scenario)
    job_id = seeded_job_id(ctx)
    scenario.store.reserve_job_restart(
        CLUSTER, job_id, 1, f"{job_id}-workflow/0/RESTART_WORKLOAD"
    )
    with pytest.raises(VariantNotRun) as info:
        hw.hardware_chain_continues(ctx)
    assert info.value.reason == "FIXTURE_NOT_READY"
    assert "preexisting ledger" in info.value.detail
    assert FakeJob.created == [], "no job on a refused seed"
    assert ctx.cleanup.pending() == [f"budget-{job_id}"], "the residue cleanup is armed"
    results = ctx.cleanup.run()
    assert results[0]["ok"], results
    with pytest.raises(NotFoundError):
        scenario.store.get_restart_budget(CLUSTER, job_id)


def test_cleanup_runs_lifo_even_when_the_variant_dies_midway(tmp_path: Path) -> None:
    scenario = Scenario(decision_error=RuntimeError("API Pod gone"))
    ctx = context(tmp_path, scenario)
    with pytest.raises(RuntimeError, match="API Pod gone"):
        hw.hardware_chain_continues(ctx)
    job_id = seeded_job_id(ctx)
    assert ctx.cleanup.pending() == [
        f"budget-{job_id}",
        f"job-{job_id}",
        "kernel-replay-kmsg",
    ]
    results = ctx.cleanup.run()
    assert [item["name"] for item in results] == [
        "kernel-replay-kmsg",
        f"job-{job_id}",
        f"budget-{job_id}",
    ]
    assert all(item["ok"] for item in results), results
    assert kernel_calls(ctx)[-1] == "close-window", "the open window was closed"
    assert FakeJob.created[0].cleaned, "the job was deleted"
    with pytest.raises(NotFoundError):
        scenario.store.get_restart_budget(CLUSTER, job_id)
    assert FakeJob.log[-2:] == ["job-cleanup", "budget-cleanup"], FakeJob.log


def test_every_undo_is_attempted_when_one_fails(tmp_path: Path) -> None:
    scenario = Scenario()
    ctx = context(tmp_path, scenario)
    out = hw.hardware_chain_continues(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    job = FakeJob.created[0]
    job.refuse_cleanup = "workload occupied by a stranger"
    results = ctx.cleanup.run()
    assert [item["ok"] for item in results] == [True, False, True], results
    assert "stranger" in results[1]["error"]
    with pytest.raises(NotFoundError):
        scenario.store.get_restart_budget(CLUSTER, out["job_id"])


def test_budget_cleanup_raises_when_the_row_is_not_provably_gone(
    tmp_path: Path,
) -> None:
    def keep_row(document: dict[str, Any]) -> None:
        document["clean"] = False
        document["remaining"] = {"restart_count": 0, "reservation_ids": []}

    scenario = Scenario(mutate_cleanup=keep_row)
    ctx = context(tmp_path, scenario)
    budget = hw.ExhaustedRestartBudget(ctx)
    budget.exhaust()
    with pytest.raises(RuntimeError, match="ledger remains"):
        budget.cleanup()
    # A reservation this run did not make keeps the row and fails the undo
    # (the budget is immutable per job, so a stranger takes the freed slot).
    scenario = Scenario()
    ctx = context(tmp_path / "foreign", scenario)
    budget = hw.ExhaustedRestartBudget(ctx)
    budget.exhaust()
    scenario.store.release_job_restart(
        CLUSTER, budget.job_id, budget.seed_reservation_ids[0]
    )
    scenario.store.reserve_job_restart(
        CLUSTER, budget.job_id, 1, "someone-else/0/RESTART_WORKLOAD"
    )
    with pytest.raises(RuntimeError, match="ledger remains"):
        budget.cleanup()
    state = scenario.store.get_restart_budget(CLUSTER, budget.job_id)
    assert list(state.reservation_ids) == ["someone-else/0/RESTART_WORKLOAD"], (
        "only this run's reservations were released"
    )


def test_exhausted_budget_identity_and_guards(tmp_path: Path) -> None:
    ctx = context(tmp_path, Scenario())
    budget = hw.ExhaustedRestartBudget(ctx)
    assert budget.job_id.startswith("gl29hc-") and budget.job_id.endswith("-a1")
    assert budget.workflow_id == f"{budget.job_id}-workflow"
    assert budget.seed_reservation_ids == [f"{budget.workflow_id}/0/RESTART_WORKLOAD"]
    with pytest.raises(ValueError, match="at least 1"):
        hw.ExhaustedRestartBudget(ctx, budget=0)
    budget.exhaust()
    # A second seed meets its own row: refused by the probe (NOT_RUN), the
    # cleanup registered once.
    with pytest.raises(VariantNotRun, match="preexisting ledger"):
        budget.exhaust()
    assert ctx.cleanup.pending().count(budget.cleanup_name) == 1


def test_contrast_and_display_helpers_read_the_records() -> None:
    assert hw.contrast_errors({}) == [] and hw.withheld_errors({}, [], budget=1) == []
    two_failed = chain_workflow(skip=("VALIDATE_GPU",))
    two_failed["step_executions"].append(
        {
            "step_index": 7,
            "operation": "VALIDATE_GPU",
            "status": "FAILED",
            "details": {},
        }
    )
    errors = hw.contrast_errors(two_failed)
    assert any("FAILED steps are" in error for error in errors), errors
    assert hw.display_stack_errors(None) == []
    assert hw.display_stack_errors({"details": {"display_stack_restored": True}}) == []
    assert (
        hw.gpu_health_errors(
            {"active_findings": [], "batches": []}, reset_execution=None
        )
        == []
    )
    assert hw.result({"a": []})["verdict"] == "PASS"
    assert hw.result({"a": ["x"]})["errors"] == ["a: x"]


# --------------------------------------------------------------------------- #
# the plan path: both approvals, bound on plan and execute alike
# --------------------------------------------------------------------------- #
def _preflight_regional(gpu_pods: Sequence[dict[str, Any]] = ()) -> FakeRegional:
    from tests.regional._general_live_fakes import store_script

    fresh = datetime.now(timezone.utc).isoformat()
    return FakeRegional(
        node_metadata={"name": NODE, "labels": {"eks.amazonaws.com/nodegroup": "g"}},
        gpu_pods=gpu_pods,
        cpu_python=store_script(
            {
                "preflight": {
                    "agent": {"lifecycle_state": "ACTIVE"},
                    "open_incidents": [],
                    "active_workflows": [],
                    "collector_statuses": [
                        {
                            "collector": c,
                            "last_success_at": fresh,
                            "last_error_at": None,
                        }
                        for c in ("HOST_TELEMETRY", "GPU_INVENTORY", "GPU_METRICS")
                    ],
                }
            }
        ),
    )


@pytest.mark.parametrize("approvals", [(), ("GPU_RESET",), ("KERNEL_SOURCE_REPLAY",)])
def test_plan_is_not_authorized_without_both_approvals(
    approvals: tuple[str, ...],
) -> None:
    from scripts.e2e.regional.general_live_fixture import read_only_preflight

    settings = make_settings(SPEC, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "approvals": approvals})
    result = read_only_preflight(settings, SPEC, _preflight_regional())
    assert result["planned_verdict"] == "NOT_RUN(NOT_AUTHORIZED)"
    assert sorted(result["missing_approvals"]) == sorted(
        set(SPEC.approvals) - set(approvals)
    )


def test_plan_passes_on_an_idle_node_with_both_approvals() -> None:
    from scripts.e2e.regional.general_live_fixture import read_only_preflight

    settings = make_settings(SPEC, node=NODE)
    result = read_only_preflight(settings, SPEC, _preflight_regional())
    assert result["planned_verdict"] is None, result
    assert result["errors"] == [], result["errors"]
    assert result["missing_approvals"] == []
    # The variant creates its own GPU job: a node already running one is refused.
    busy = _preflight_regional(
        gpu_pods=[{"node": NODE, "name": "someone-elses-trainer"}]
    )
    result = read_only_preflight(settings, SPEC, busy)
    assert any("GPU-requesting Pods" in error for error in result["errors"]), result[
        "errors"
    ]
