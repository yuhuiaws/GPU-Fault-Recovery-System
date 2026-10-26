"""GF-GENERAL-LIVE-032 unknown error: category-module verdict tests.

Driven over fakes: a ``FakeProbe`` stands in for the kmsg-mode replay host and a
``FakeRegional`` answers both the XID policy-decision read and the NVIDIA_KERNEL
activity read, so the unknown-code -> SITE_SAFETY grading and the dedup of a
repeated event are checked on the product's own ``FaultPolicyDecision`` shape.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_unknown as mod
from scripts.e2e.regional.general_live_registry import coverage_rows
from scripts.e2e.regional.general_live_source_replay import KERNEL_UNIT
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import survey as gpu_survey

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def _spec(variant: str) -> object:
    return next(item for item in mod.VARIANTS if item.variant == variant)


def _host_script(*, consumed: bool = True) -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True, "fixture": "/run/.../kmsg.fixture"},
        "status": {
            "injected": {
                "mode": "kmsg",
                "armed": True,
                "fixture_lines": 5,
                "fixture_bytes": 440,
                "consumed": consumed,
                "followed": consumed,
                "stream_reopens": 0 if consumed else 541,
                "collector": {
                    "main_pid": "2073979",
                    "fixture_open": consumed,
                    "offset": 440 if consumed else None,
                },
            }
        },
        "close-window": {"clean": True},
    }


def _cpu_python(
    decisions: list[dict[str, object]],
    *,
    sxid_decisions: list[dict[str, object]] | None = None,
    workflows: list[dict[str, object]] | None = None,
):
    def responder(script: str, *arguments: str) -> dict[str, object]:
        if "list_xid_events" in script:
            return {"decisions": decisions}
        if "get_xid_policy_decision" in script:
            return {"decisions": sxid_decisions or []}
        return {
            "incidents": [],
            "workflows": workflows or [],
            "markers": [],
            "notifications": [],
            "batches": [],
            "metric_latest": [],
            "collector_statuses": [],
        }

    return responder


def _decision(**overrides: object) -> dict[str, object]:
    base = {
        "event_id": "xid-987-1",
        "xid": mod.UNKNOWN_XID,
        "disposition": "SITE_SAFETY",
        "action": None,
        "requires_operator": True,
        "incident_id": "inc-xid-987",
        "workflow_request_id": None,
        "duplicate": False,
        "reasons": ["XID is absent from the pinned NVIDIA Catalog"],
    }
    base.update(overrides)
    return base


def _ctx(variant: str, *, decisions: list[dict[str, object]], tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    regional = FakeRegional(cpu_python=_cpu_python(decisions))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx


def test_unknown_xid_is_site_safety_without_action(tmp_path: Path) -> None:
    ctx = _ctx("unknown_xid", decisions=[_decision()], tmp_path=tmp_path)
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_unknown_xid_fails_if_it_earns_an_nvidia_action(tmp_path: Path) -> None:
    ctx = _ctx(
        "unknown_xid",
        decisions=[_decision(disposition="EXECUTABLE", action="RESET_GPU")],
        tmp_path=tmp_path,
    )
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "recovery action" in error or "SITE_SAFETY" in error
        for error in result["errors"]
    ), result["errors"]


def test_unknown_xid_inconclusive_without_injection(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    spec = _spec("unknown_xid")
    fake = FakeProbe({**_host_script(), "status": {"injected": {"fixture_lines": 0}}})
    regional = FakeRegional(cpu_python=_cpu_python([]))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.unknown_xid(ctx)
    except VariantInconclusive as exc:
        assert "no reviewed records" in str(exc), str(exc)
    else:
        raise AssertionError("an empty kmsg fixture must be INCONCLUSIVE")


def test_unknown_xid_inconclusive_when_the_collector_never_read_the_fixture(
    tmp_path: Path,
) -> None:
    """Live R13g (ip-10-0-32-162): five copies of the XID 987 line in the fixture,
    the collector reopening it at EOF every second and reading none, no XID
    event in the store -- the grading claim is unproven, not failed."""

    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    spec = _spec("unknown_xid")
    fake = FakeProbe(_host_script(consumed=False))
    regional = FakeRegional(cpu_python=_cpu_python([]))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.unknown_xid(ctx)
    except VariantInconclusive as exc:
        assert "did not follow the reviewed kmsg fixture" in str(exc), str(exc)
        assert "541 time(s)" in str(exc), str(exc)
    else:
        raise AssertionError("a fixture the collector never read must be INCONCLUSIVE")
    assert [call[0] for call in fake.calls] == [
        "install-records",
        "open-window",
        "arm",
        "status",
    ], "the decision wait is skipped; the cleanup stack still closes the window"
    assert "kernel-replay-kmsg" in ctx.cleanup.pending()


def test_duplicate_event_isolates_once(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1"),
        _decision(event_id="xid-987-2", duplicate=True, incident_id="inc-xid-987"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_duplicate_fails_on_second_isolation(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1", incident_id="inc-a"),
        _decision(event_id="xid-987-2", incident_id="inc-b"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("isolated" in error for error in result["errors"]), result["errors"]


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-032"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "032 registers every coverage variant"
    )
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {
        "unknown_xid",
        "duplicate",
        "missing_topology",
        "missing_version",
    }, "032 implements the kmsg variants and the kernel-collector version replay"
    deferred = [item for item in mod.VARIANTS if item.deferred]
    assert len(deferred) == 4, (
        "the other four 032 variants stay in the denominator as NOT_RUN"
    )
    version = _spec("missing_version")
    assert version.implemented and version.approvals == (
        "KERNEL_COLLECTOR_SOURCE_REPLAY",
        mod.KERNEL_SOURCE_REPLAY,
    ), "missing_version needs the kernel collector window and the kmsg replay"
    assert version.lanes == ("eks", "hp") and (version.flags, version.risk) == (
        "S",
        "M",
    )
    assert (
        "CAPABILITY_ABSENT" in version.expected and "product gate" in version.expected
    )
    assert version.expected_outcome == "PASS"
    topology = _spec("missing_topology")
    assert topology.lanes == ("eks", "hp") and topology.approvals == (
        mod.KERNEL_SOURCE_REPLAY,
    )


# --------------------------------------------------------------------------- #
# missing_topology: a fatal SXID with no trusted port-scope evidence
# --------------------------------------------------------------------------- #
def _sxid_decision(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "event_id": f"kernel-log-r1-sxid-{mod.MISSING_TOPOLOGY_SXID}-abc-cluster-deadbeef",
        "incident_id": "inc-sxid",
        "incident_state": "QUARANTINED",
        "disposition": "BLOCKED_MISSING_EVIDENCE",
        "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
        "investigatory_action": None,
        "action": None,
        "safety_action": "QUARANTINE",
        "pre_actions": ["MARK_UNSCHEDULABLE", "STOP_WORKLOAD"],
        "requires_operator": True,
        "workflow_request_id": "wf-sxid",
        "duplicate": False,
        "reasons": ["fatal SXID requires a trusted access/trunk port-scope mapping"],
    }
    base.update(overrides)
    return base


def _quarantine_workflow(*operations: str) -> dict[str, object]:
    return {
        "request_id": "wf-sxid",
        "incident_id": "inc-sxid",
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:00:30+00:00",
        "official_steps": [
            {"operation": item, "parameters": {}} for item in operations
        ],
        "step_executions": [],
    }


def _topology_ctx(
    decisions: list[dict[str, object]],
    workflows: list[dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, list[list[dict[str, object]]]]:
    spec = _spec("missing_topology")
    fake = FakeProbe(_host_script())
    regional = FakeRegional(
        cpu_python=_cpu_python([], sxid_decisions=decisions, workflows=workflows)
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    restored: list[list[dict[str, object]]] = []

    def restore(
        context: object, incidents: list[dict[str, object]], *, reason: str
    ) -> list:
        restored.append(list(incidents))
        return [
            {"incident_id": item["incident_id"], "status": "SUCCEEDED"}
            for item in incidents
        ]

    monkeypatch.setattr(mod, "restore_incidents", restore)
    return ctx, restored


def test_missing_topology_is_blocked_with_the_gap_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, restored = _topology_ctx(
        [_sxid_decision()],
        [
            _quarantine_workflow(
                "FREEZE_EVIDENCE",
                "COLLECT_DIAGNOSTIC_BUNDLE",
                "MARK_UNSCHEDULABLE",
                "QUARANTINE",
            )
        ],
        tmp_path,
        monkeypatch,
    )
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay" and result["incidents"] == [
        "inc-sxid"
    ]
    assert restored == [[{"incident_id": "inc-sxid"}]], "the quarantine is handed back"
    assert "Link" not in mod.MISSING_TOPOLOGY_LINE.split(",", 2)[-1].replace("link", "")
    assert str(mod.MISSING_TOPOLOGY_SXID) in mod.MISSING_TOPOLOGY_LINE


def test_missing_topology_fails_when_the_reset_was_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    granted = _sxid_decision(
        disposition="EXECUTABLE",
        action=None,
        requires_operator=False,
        reasons=[
            "fatal trunk SXID requires stopping Fabric Manager and coordinated reset"
        ],
    )
    ctx, _restored = _topology_ctx(
        [granted],
        [
            _quarantine_workflow(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "RESET_ALL_GPUS_NVSWITCHES"
            )
        ],
        tmp_path,
        monkeypatch,
    )
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert (
        "expected BLOCKED_MISSING_EVIDENCE" in joined and "port-scope mapping" in joined
    ), joined
    assert (
        "RESET_ALL_GPUS_NVSWITCHES" in joined
        and "did not require an operator" in joined
    ), joined


def test_missing_topology_fails_without_a_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, restored = _topology_ctx([], [], tmp_path, monkeypatch)
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("did not reach a FaultPolicyDecision" in e for e in result["errors"]), (
        result
    )
    assert restored == [], "nothing to restore without an incident"


def test_the_reviewed_sxid_line_is_blocked_by_the_real_parser_and_engine() -> None:
    """kmsg filter -> nvidia_logs parser -> topology service -> policy engine."""

    from datetime import datetime, timezone

    from gpu_fault.collectors.logs.fabric_manager import SXID_SUMMARY_PATTERN
    from gpu_fault.models import RecoveryAction
    from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
    from gpu_fault.policy.catalog import load_sxid_policy
    from gpu_fault.policy.engine import GpuFaultPolicyEngine
    from gpu_fault.policy.models import (
        NVIDIA_ALWAYS_FATAL_SXIDS,
        ActionDisposition,
        SxidClassification,
        SxidLinkScope,
    )
    from gpu_fault.telemetry import NvSwitchPortTopologyService
    from tests._builders import build_store

    assert SXID_SUMMARY_PATTERN.search(mod.MISSING_TOPOLOGY_LINE), (
        "the kernel collector forwards Fabric Manager summary lines only"
    )
    rule = next(
        item
        for item in load_sxid_policy().rules
        if item.sxid == mod.MISSING_TOPOLOGY_SXID
    )
    assert rule.classification is SxidClassification.FATAL
    assert rule.official_action == "SCOPE_DEPENDENT_RESET"
    assert mod.MISSING_TOPOLOGY_SXID not in NVIDIA_ALWAYS_FATAL_SXIDS
    source = NvidiaKernelLogEvent(
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        record_id="r1",
        observed_at=datetime(2026, 10, 24, tzinfo=timezone.utc),
        message=mod.MISSING_TOPOLOGY_LINE,
        product="H200",
        driver_branch=580,
        runtime_profile_version="simulated-v1",
    )
    (event,) = NvidiaLogNormalizer().normalize_kernel(source).sxid_events
    assert event.sxid == mod.MISSING_TOPOLOGY_SXID
    assert event.classification is SxidClassification.FATAL
    assert event.port is None and event.switch_id == "0"
    assert event.link_scope is SxidLinkScope.UNKNOWN
    store = build_store()
    resolved = NvSwitchPortTopologyService(store).resolve(event)
    assert resolved.link_scope is SxidLinkScope.UNKNOWN, (
        "no trusted nvswitch_port_topology and no port: the H200 invariant cannot apply"
    )
    decision = GpuFaultPolicyEngine().evaluate_sxid(resolved)
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert decision.action is None and decision.requires_operator
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert any(mod.TOPOLOGY_REASON in reason for reason in decision.reasons), (
        decision.reasons
    )
    view = {
        "event_id": decision.event_id,
        "incident_id": None,
        "disposition": decision.disposition.value,
        "official_action": decision.official_action,
        "action": None,
        "requires_operator": decision.requires_operator,
        "reasons": list(decision.reasons),
    }
    assert mod.missing_topology_errors([view]) == []
    linked = source.model_copy(
        update={"message": mod.MISSING_TOPOLOGY_LINE + " Link 3"}
    )
    (with_link,) = NvidiaLogNormalizer().normalize_kernel(linked).sxid_events
    assert (
        NvSwitchPortTopologyService(store).resolve(with_link).link_scope
        is SxidLinkScope.ACCESS
    ), (
        "a line naming a link would get ACCESS from the H200 invariant: the replay must not name one"
    )


# --------------------------------------------------------------------------- #
# missing_version: the kernel collector's version withheld, XID 149 replayed
# --------------------------------------------------------------------------- #
#: The GPU fake's first GPU: exporter BDF ``00000000:01:00.0`` -> kernel form.
VERSION_PCI = "0000:01:00"
VERSION_REASON_TEXT = "NVLink5 decode requires driver branch, IntrInfo and Error Status"


def _version_decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "kernel-log-r1-xid-149",
        "xid": mod.MISSING_VERSION_XID,
        "product": "NVIDIA B200",
        "driver_branch": None,
        "intr_info": mod.MISSING_VERSION_INTR_INFO,
        "error_status": mod.MISSING_VERSION_ERROR_STATUS,
        "gpu_uuid": "GPU-00000000-0000-4000-8000-000000000000",
        "pci_bdf": VERSION_PCI,
        "event_source": "KERNEL_LOG",
        "observed_at": "2026-10-24T00:00:10+00:00",
        "decision": True,
        "disposition": "BLOCKED_MISSING_EVIDENCE",
        "official_action": "WORKFLOW_NVLINK5_ERR",
        "investigatory_action": "WORKFLOW_NVLINK5_ERR",
        "action": None,
        "safety_action": "QUARANTINE",
        "pre_actions": ["MARK_UNSCHEDULABLE", "STOP_WORKLOAD"],
        "requires_operator": True,
        "incident_id": "inc-149",
        "incident_state": "QUARANTINED",
        "workflow_request_id": "wf-149",
        "duplicate": False,
        "decoded_subcode": None,
        "matched_decode_rules": [],
        "reasons": [VERSION_REASON_TEXT],
    }
    base.update(overrides)
    return base


def _version_workflow(*operations: str) -> dict[str, Any]:
    return {
        "request_id": "wf-149",
        "incident_id": "inc-149",
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:00:30+00:00",
        "official_steps": [
            {"operation": item, "parameters": {}} for item in operations
        ],
        "step_executions": [],
    }


QUARANTINE_STEPS = (
    "FREEZE_EVIDENCE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
)
UNAVAILABLE_LINE = (
    "nvidia-smi software version discovery unavailable; using configured "
    "driver/CUDA versions"
)


def _kernel_window_script(
    *, served: int = 1, journal_lines: list[str] | None = None
) -> dict[str, Any]:
    return {
        "install-source": {"bytes": 1},
        "install-wrapper": {
            "shim": "/run/gfrs-live/source-x/bin/nvidia-smi",
            "unit": KERNEL_UNIT,
        },
        "open-window": {
            "after": {"ActiveState": "active"},
            "unit": KERNEL_UNIT,
            "proof_scope": "live-replay",
        },
        "audit-read": {
            "entries": [
                {
                    "query": "driver_version",
                    "applied_rules": [0],
                    "served_returncode": 1,
                }
            ]
            * served,
            "calls": {"nvidia-smi/0": served},
        },
        "journal-read": {
            "lines": [UNAVAILABLE_LINE] if journal_lines is None else journal_lines
        },
        "close-window": {"clean": True},
    }


def _version_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: list[dict[str, Any]],
    workflows: list[dict[str, Any]] | None = None,
    model: str = "NVIDIA B200",
    served: int = 1,
    journal_lines: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    spec = _spec("missing_version")
    fakes = {
        "node": FakeProbe({"survey": {"residue": {}}}),
        "dcgm": FakeProbe({"survey": gpu_survey(model=model)}),
        "kernel_window": FakeProbe(
            _kernel_window_script(served=served, journal_lines=journal_lines)
        ),
        "kmsg": FakeProbe(_host_script()),
    }

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if script == mod.VERSION_DECISION_PROBE:
            assert arguments[3] == str(mod.MISSING_VERSION_XID)
            return {"decisions": decisions}
        if script == ACTIVITY_PROBE:
            return {
                "incidents": [],
                "workflows": list(workflows or []),
                "markers": [],
                "notifications": [],
                "batches": [],
                "metric_latest": [],
                "collector_statuses": [],
            }
        raise AssertionError(f"unexpected probe script: {script[:80]}")

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        probes={
            f"node@{NODE}": fakes["node"],
            f"dcgm@{NODE}": fakes["dcgm"],
            f"kernel-collector-replay@{NODE}": fakes["kernel_window"],
            f"kernel-replay-kmsg@{NODE}": fakes["kmsg"],
        },
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    restored: list[list[dict[str, Any]]] = []

    def restore(
        context: Any, incidents: list[dict[str, Any]], *, reason: str
    ) -> list[dict[str, Any]]:
        restored.append(list(incidents))
        return [
            {"incident_id": item["incident_id"], "status": "SUCCEEDED"}
            for item in incidents
        ]

    monkeypatch.setattr(mod, "restore_incidents", restore)
    fakes["restored"] = restored
    return ctx, fakes


def _b64_payload(call: tuple[str, ...]) -> Any:
    return json.loads(base64.b64decode(call[call.index("--b64") + 1]))


def test_missing_version_is_capability_absent_on_an_h200_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This site: the product gate answers before the decoder asks for the version."""

    ctx, fakes = _version_ctx(tmp_path, monkeypatch, decisions=[], model="NVIDIA H200")
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    with pytest.raises(VariantNotRun) as info:
        mod.missing_version(ctx)
    assert info.value.reason == "CAPABILITY_ABSENT"
    assert "product gate" in info.value.detail and "NVIDIA H200" in info.value.detail
    assert "B100/GB200" in info.value.detail
    assert fakes["kernel_window"].calls == [] and fakes["kmsg"].calls == [], (
        "nothing was installed, opened or injected"
    )
    assert fakes["restored"] == []
    assert mod.version_gate_errors("NVIDIA B200") == []
    assert mod.version_gate_errors("NVIDIA GB200") == []
    assert mod.version_gate_errors(None) and mod.version_gate_errors("NVIDIA A100")
    assert mod.version_gate_errors("NVIDIA B200", xid=79), (
        "an XID without the NVLink5 workflow has no version gate"
    )


def test_missing_version_blocks_for_the_driver_branch_on_a_blackwell_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, fakes = _version_ctx(
        tmp_path,
        monkeypatch,
        decisions=[_version_decision()],
        workflows=[_version_workflow(*QUARANTINE_STEPS)],
    )
    result = mod.missing_version(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay" and result["product"] == "NVIDIA B200"
    assert result["incidents"] == ["inc-149"] and fakes["restored"] == [
        [{"incident_id": "inc-149"}]
    ], "the quarantine is handed back"
    assert result["window"]["driver_version_calls_served"] == 1
    assert result["window"]["journal_unavailable_lines"] == 1
    assert result["xid_line"].startswith(
        f"NVRM: Xid (PCI:{VERSION_PCI}): 149, pid=0, name=gfrs-live, NETIR_LINK_EVT "
        "Fatal XC0 i0 Link 08 (0x004505c6 0x00000000)"
    ), result["xid_line"]
    window_calls = fakes["kernel_window"].calls
    # The wrapper and its plan, the window, then -- once -- the builders file
    # that carries the read-only ``audit-read``/``journal-read`` since
    # 2026-10-26, before the first of them.
    assert [call[0] for call in window_calls] == [
        "install-source",
        "install-source",
        "install-wrapper",
        "open-window",
        "install-source",
        "audit-read",
        "journal-read",
    ], window_calls
    assert "general_live_source_builders.py" in window_calls[4], window_calls[4]
    plan = _b64_payload(
        next(
            call
            for call in window_calls
            if call[0] == "install-source" and "plan-nvidia-smi.json" in call
        )
    )
    assert plan["rules"] == [
        {
            "query": "driver_version",
            "action": "exit",
            "exit_code": mod.VERSION_WITHHELD_EXIT,
            "stderr": mod.VERSION_WITHHELD_STDERR,
        }
    ], "the one reviewed rule: the driver-version query fails"
    installed = next(call for call in window_calls if call[0] == "install-wrapper")
    assert ("--unit", KERNEL_UNIT) == installed[installed.index("--unit") :][:2]
    opened = next(call for call in window_calls if call[0] == "open-window")
    assert "--mode" in opened and opened[opened.index("--mode") + 1] == "command"
    assert opened[opened.index("--unit") + 1] == KERNEL_UNIT
    assert "--bind" not in opened and "--env" not in opened
    kmsg_calls = fakes["kmsg"].calls
    assert [call[0] for call in kmsg_calls] == [
        "install-records",
        "open-window",
        "arm",
        "status",
    ], "the window is proven before the line is injected"
    records = _b64_payload(kmsg_calls[0])
    assert records[0]["message"] == result["xid_line"]
    assert records[0]["transport"] == "kernel"
    assert "kernel-collector-replay" in ctx.cleanup.pending()
    assert "kernel-replay-kmsg" in ctx.cleanup.pending()


def test_missing_version_fails_when_the_decoder_acted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acted = _version_decision(
        disposition="EXECUTABLE",
        action="RESET_GPU",
        safety_action=None,
        pre_actions=[],
        requires_operator=False,
        reasons=[
            "matched the pinned official XID 144-150 decode table: "
            "NETIR_LINK_EVT/NETIR_LINK_DOWN"
        ],
    )
    ctx, fakes = _version_ctx(
        tmp_path,
        monkeypatch,
        decisions=[acted],
        workflows=[_version_workflow("FREEZE_EVIDENCE", "RESET_GPU")],
    )
    result = mod.missing_version(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert "expected BLOCKED_MISSING_EVIDENCE" in joined
    assert (
        "earned action RESET_GPU" in joined and "did not require an operator" in joined
    )
    assert "does not name the missing driver branch" in joined
    assert "plans ['RESET_GPU']" in joined
    assert fakes["restored"] == [[{"incident_id": "inc-149"}]]


def test_missing_version_errors_pin_the_registers_and_the_reason() -> None:
    assert mod.missing_version_errors([]) == [
        "the reviewed XID 149 line did not reach a FaultPolicyDecision"
    ]
    assert mod.missing_version_errors([_version_decision()]) == []
    errors = mod.missing_version_errors(
        [_version_decision(intr_info=None, error_status=None)]
    )
    assert errors and "not for the version" in errors[0], errors
    duplicate_only = _version_decision(duplicate=True, disposition="SITE_SAFETY")
    assert mod.missing_version_errors([duplicate_only]), (
        "a duplicate-only read is still graded rather than passed"
    )
    assert mod.version_stamped(
        [_version_decision(), _version_decision(driver_branch=580)]
    ) == [_version_decision(driver_branch=580)]


def test_missing_version_is_inconclusive_when_the_collector_never_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    ctx, fakes = _version_ctx(tmp_path, monkeypatch, decisions=[], served=0)
    with pytest.raises(VariantInconclusive, match="never took effect"):
        mod.missing_version(ctx)
    assert fakes["kmsg"].calls == [], "no XID line is injected without the proof"
    assert [call[0] for call in fakes["kernel_window"].calls][-2:] == [
        "audit-read",
        "journal-read",
    ]


def test_missing_version_is_inconclusive_when_the_journal_found_a_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    ctx, fakes = _version_ctx(
        tmp_path,
        monkeypatch,
        decisions=[],
        journal_lines=["discovered NVIDIA driver branch 580 and CUDA version 12.8"],
    )
    with pytest.raises(VariantInconclusive, match="discovered driver branch"):
        mod.missing_version(ctx)
    assert fakes["kmsg"].calls == []


def test_missing_version_is_inconclusive_when_a_branch_was_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    stamped = _version_decision(
        driver_branch=580,
        reasons=[
            "IntrInfo/Error Status has no match in the pinned official XID 144-150 "
            "decode table"
        ],
    )
    ctx, fakes = _version_ctx(tmp_path, monkeypatch, decisions=[stamped])
    with pytest.raises(VariantInconclusive, match="driver_branch=580"):
        mod.missing_version(ctx)
    assert fakes["restored"] == [[{"incident_id": "inc-149"}]], (
        "an ineffective window still hands the quarantine back"
    )


def test_the_reviewed_xid149_line_through_the_real_parser_and_engine() -> None:
    """kmsg line -> nvidia_logs parser -> policy engine, per product and branch.

    Pins the site finding (H200: NOT_APPLICABLE before the decoder) and the
    Blackwell outcomes the variant grades and guards against.
    """

    from datetime import datetime, timezone

    from gpu_fault.models import RecoveryAction
    from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
    from gpu_fault.policy.engine import GpuFaultPolicyEngine
    from gpu_fault.policy.models import ActionDisposition

    line = mod.missing_version_line(VERSION_PCI, "gl032-missing-version-a1-deadbeef")

    def decide(product: str, branch: int | None) -> tuple[Any, Any]:
        source = NvidiaKernelLogEvent(
            cluster_id="gpu-eks-cluster",
            node_id=NODE,
            record_id="r1",
            observed_at=datetime(2026, 10, 24, tzinfo=timezone.utc),
            message=line,
            product=product,
            driver_branch=branch,
            runtime_profile_version="simulated-v1",
        )
        (event,) = NvidiaLogNormalizer().normalize_kernel(source).xid_events
        assert event.xid == mod.MISSING_VERSION_XID
        assert event.intr_info == mod.MISSING_VERSION_INTR_INFO
        assert event.error_status == mod.MISSING_VERSION_ERROR_STATUS
        assert event.pci_bdf == VERSION_PCI
        return event, GpuFaultPolicyEngine().evaluate_xid(event)

    # This site: every WORKFLOW_NVLINK5_ERR XID is NOT_APPLICABLE on an H200,
    # with or without a branch -- the version gate is never reached.
    for branch in (None, 580):
        _event, decision = decide("NVIDIA H200", branch)
        assert decision.disposition is ActionDisposition.NOT_APPLICABLE
        assert "not applicable to product NVIDIA H200" in decision.reasons[0]
    assert mod.version_gate_errors("NVIDIA H200"), (
        "the variant records CAPABILITY_ABSENT"
    )
    # A Blackwell node with no branch stamped: the decoder blocks for the version.
    event, decision = decide("NVIDIA B200", None)
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.reasons == [VERSION_REASON_TEXT]
    assert decision.official_action == "WORKFLOW_NVLINK5_ERR"
    assert decision.action is None and decision.requires_operator
    assert decision.safety_action is RecoveryAction.QUARANTINE
    view = {
        **_version_decision(),
        "driver_branch": event.driver_branch,
        "intr_info": event.intr_info,
        "error_status": event.error_status,
        "disposition": decision.disposition.value,
        "official_action": decision.official_action,
        "action": None,
        "requires_operator": decision.requires_operator,
        "reasons": list(decision.reasons),
    }
    assert (
        mod.missing_version_errors([view]) == [] and mod.version_stamped([view]) == []
    )
    # The same line with the branch stamped: this site's R580 (V2) matches no
    # rule and is still blocked; a V1 driver would decode it to RESET_GPU --
    # which is why the window is proven effective before the line is injected.
    _event, v2 = decide("NVIDIA B200", 580)
    assert v2.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert "no match" in v2.reasons[0]
    _event, v1 = decide("NVIDIA B200", 570)
    assert v1.disposition is ActionDisposition.EXECUTABLE
    assert v1.action is RecoveryAction.RESET_GPU
    assert "NETIR_LINK_EVT/NETIR_LINK_DOWN" in v1.reasons[0]
