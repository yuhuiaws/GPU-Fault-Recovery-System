"""The monitor-grade trusted kernel XID helper, over fakes (no node, no cluster).

Pins the catalog entries the 020/024 variants replay (IGNORE-class, H100
family, no workflow resolver, not a PCIe link XID), runs the reviewed line
through the product's own kernel-log normalizer, drives ``inject_kernel_xid``
over a ``FakeProbe`` standing in for the kmsg-mode replay host, and checks the
independence grading on the product's ``FaultPolicyDecision`` shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from scripts.e2e.regional import general_live_common
from scripts.e2e.regional import general_live_kernel_xid as kx
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_gpu_composite import VARIANTS
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
RUN = "gl020-independent-fatal-event-a1-deadbeef"


# --------------------------------------------------------------------------- #
# the pinned catalog
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("xid", "mnemonic"), [(137, "NVLINK_PRIV_ERR"), (92, "EXCESSIVE_SBE_INTERRUPTS")]
)
def test_chosen_entries_are_monitor_grade_in_the_pinned_catalog(
    xid: int, mnemonic: str
) -> None:
    entry = kx.catalog_entry(xid)
    assert entry is not None, f"XID {xid} must be in the pinned catalog"
    assert entry.mnemonic == mnemonic
    assert entry.immediate_action == "IGNORE"
    assert "H100" in entry.products
    assert kx.monitor_grade_errors(entry, xid=xid) == []
    assert kx.require_monitor_grade(xid) == entry


@pytest.mark.parametrize(
    ("xid", "fragment"),
    [
        (46, "RESET_GPU"),  # catalog RESET_GPU: a reset live
        (31, "RESTART_APP"),  # NO_ACTION only when the node is known IDLE
        (32, "PCIe"),  # joins the PCIe replay composite
        (45, "resolver"),  # its own companion workflow
        (48, "resolver"),
        (987, "absent"),  # unknown code -> SITE_SAFETY quarantine
    ],
)
def test_non_monitor_grade_entries_are_refused_before_any_mutation(
    xid: int, fragment: str
) -> None:
    errors = kx.monitor_grade_errors(kx.catalog_entry(xid), xid=xid)
    assert any(fragment in error for error in errors), errors
    with pytest.raises(VariantNotRun) as info:
        kx.require_monitor_grade(xid)
    assert info.value.reason == "FIXTURE_NOT_READY"


def test_h200_product_is_the_h100_catalog_family() -> None:
    assert kx.product_family("NVIDIA H200") == kx.SITE_FAMILY


# --------------------------------------------------------------------------- #
# the reviewed line
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00000000:1B:00.0", "0000:1b:00"),
        ("0000:53:00", "0000:53:00"),
        ("1b:00.0", "0000:1b:00"),
        ("", None),
        (None, None),
        ("not-a-bdf", None),
    ],
)
def test_kmsg_pci_bdf_prints_the_kernel_form(
    value: str | None, expected: str | None
) -> None:
    assert kx.kmsg_pci_bdf(value) == expected


def test_reviewed_line_parses_through_the_product_normalizer() -> None:
    message = kx.xid_message(137, "0000:1b:00", run_id=RUN, label="020-independent")
    batch = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="gpu-eks-cluster",
            node_id=NODE,
            record_id="kmsg-boot-900000064",
            observed_at=datetime.now(timezone.utc),
            message=message,
        )
    )
    assert [event.xid for event in batch.xid_events] == [137]
    assert batch.xid_events[0].pci_bdf == "0000:1b:00"
    assert batch.sxid_events == []
    assert RUN in message, "the line names the run so evidence tells runs apart"
    with pytest.raises(ValueError):
        kx.xid_message(137, "nobdf", run_id=RUN, label="x")


# --------------------------------------------------------------------------- #
# independence grading
# --------------------------------------------------------------------------- #
def _decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "kernel-log-kmsg-boot-900000064-xid-137-cluster-abc",
        "xid": 137,
        "gpu_uuid": G1,
        "pci_bdf": "0000:1b:00",
        "event_source": "KERNEL_LOG",
        "product": "NVIDIA H200",
        "observed_at": "2026-10-25T00:00:10+00:00",
        "decision": True,
        "disposition": "MONITOR_ONLY",
        "official_action": "IGNORE",
        "investigatory_action": "XID_137_FLOW",
        "action": "NO_ACTION",
        "safety_action": None,
        "severity": "INFO",
        "requires_operator": False,
        "incident_id": "inc-kernel-log-kmsg-boot-900000064-xid-137-cluster-abc",
        "incident_state": "RECOVERED",
        "workflow_request_id": None,
        "investigatory_notification_id": "n-1",
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": ["exact NVIDIA Catalog 610 Immediate Action for XID 137"],
    }
    base.update(overrides)
    return base


ENTRY = kx.catalog_entry(137)
assert ENTRY is not None, "XID 137 is pinned in the catalog"


def test_independence_holds_for_a_monitor_only_own_incident() -> None:
    errors = kx.independence_errors(
        [_decision()], entry=ENTRY, gpu_uuid=G1, foreign_incident_ids={"inc-counter"}
    )
    assert errors == []


@pytest.mark.parametrize(
    ("decisions", "fragment"),
    [
        ([], "reached no decision"),
        ([_decision(decision=False)], "no FaultPolicyDecision"),
        ([_decision(duplicate=True)], "duplicate"),
        ([_decision(incident_id="inc-counter")], "absorbed"),
        ([_decision(disposition="EXECUTABLE", action="RESET_GPU")], "recovery action"),
        ([_decision(disposition="SITE_SAFETY", safety_action="QUARANTINE")], "safety"),
        ([_decision(official_action="RESET_GPU")], "official action"),
        ([_decision(requires_operator=True)], "operator"),
        ([_decision(correlated_event_id="e-2")], "correlated"),
        ([_decision(workflow_request_id="wf-9")], "workflow"),
        ([_decision(gpu_uuid=G0)], "resolved to GPU"),
        ([_decision(incident_state="ACTION_PENDING")], "RECOVERED"),
        ([_decision(incident_state=None)], "not in the store"),
        ([_decision(incident_id="inc-other")], "not its own"),
        ([_decision(event_source="DCGM_XID_GAUGE")], "KERNEL_LOG"),
        ([_decision(incident_id=None)], "no incident"),
        ([_decision(), _decision(event_id="e-2")], "did not dedup"),
        # A companion merge, or a flagged decision on another incident, is
        # named as such, not hidden behind its duplicate flag.
        (
            [
                _decision(
                    duplicate=True, correlated_event_id="e-2", incident_id="inc-counter"
                )
            ],
            "correlated",
        ),
        ([_decision(duplicate=True, incident_id="inc-counter")], "absorbed"),
        ([_decision(duplicate=True, incident_id="inc-other")], "not its own"),
    ],
)
def test_independence_errors_name_each_defect(
    decisions: list[dict[str, Any]], fragment: str
) -> None:
    errors = kx.independence_errors(
        decisions, entry=ENTRY, gpu_uuid=G1, foreign_incident_ids={"inc-counter"}
    )
    assert any(fragment in error for error in errors), errors


# --------------------------------------------------------------------------- #
# a marker association is bookkeeping, not absorption (batch 236)
# --------------------------------------------------------------------------- #
def _association_reasons(associated_with: str) -> list[str]:
    """The reasons the product stores for this XID once provider-event
    correlation associated its marker with ``associated_with``: the engine's
    own decision, flagged duplicate the way ``correlate_provider_event`` does,
    then bound to the event's own incident by ``execution_decision`` -- the
    product's code writes the wording, so the oracle's reading of it is pinned
    to the release under test."""

    from gpu_fault.models import WorkloadState
    from gpu_fault.orchestration.provider_correlation import execution_decision
    from gpu_fault.policy import XidEvent
    from tests._builders import build_context

    event = XidEvent(
        event_id=_decision()["event_id"],
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        observed_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
        event_source="KERNEL_LOG",
        xid=137,
        gpu_uuid=G1,
        pci_bdf="0000:1b:00",
        product="NVIDIA H200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    decision = build_context().policy.evaluate_xid(event)
    associated = decision.model_copy(
        update={
            "marker": decision.marker.model_copy(
                update={"incident_id": associated_with}
            ),
            "duplicate": True,
        }
    )
    bound = execution_decision(event, associated, None)
    assert bound.marker.incident_id == _decision()["incident_id"], (
        "the association leaves the event its own incident"
    )
    return list(bound.reasons)


def _associated(**overrides: Any) -> dict[str, Any]:
    """The live R16 shape on 024: duplicate, the association named, nothing else."""

    return _decision(
        duplicate=True, reasons=_association_reasons("inc-counter"), **overrides
    )


def test_the_association_is_read_from_the_products_own_reason() -> None:
    assert kx.marker_associations(_associated()) == ["inc-counter"]
    assert kx.marker_associations(_decision()) == [], (
        "the catalog's reason is no association"
    )
    assert kx.own_incident_id(_decision()) == _decision()["incident_id"]


def test_a_marker_association_alone_is_bookkeeping() -> None:
    associated = _associated()
    assert not kx.repeated_event(associated), "an association is no repeat"
    errors = kx.independence_errors(
        [associated], entry=ENTRY, gpu_uuid=G1, foreign_incident_ids={"inc-counter"}
    )
    assert errors == [], errors


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"incident_id": "inc-counter"}, "absorbed"),
        ({"incident_id": "inc-other"}, "not its own"),
        ({"workflow_request_id": "wf-9"}, "workflow"),
        ({"correlated_event_id": "e-2"}, "correlated"),
        ({"incident_state": "ACTION_PENDING"}, "RECOVERED"),
        ({"incident_state": None}, "not in the store"),
        ({"disposition": "EXECUTABLE", "action": "RESET_GPU"}, "recovery action"),
        ({"official_action": "RESET_GPU"}, "official action"),
        ({"requires_operator": True}, "operator"),
        ({"gpu_uuid": G0}, "resolved to GPU"),
    ],
)
def test_an_associated_decision_still_fails_when_its_handling_was_taken(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = kx.independence_errors(
        [_associated(**overrides)],
        entry=ENTRY,
        gpu_uuid=G1,
        foreign_incident_ids={"inc-counter"},
    )
    assert any(fragment in error for error in errors), errors


def test_a_duplicate_the_product_does_not_explain_is_a_repeat() -> None:
    repeat = _decision(duplicate=True)
    assert kx.repeated_event(repeat), "no association, no companion: a repeat"
    assert not kx.repeated_event(_decision()), "a fresh decision is no repeat"
    assert not kx.repeated_event(
        _decision(duplicate=True, correlated_event_id="e-2")
    ), "a companion merge is judged as a correlation, not as a repeat"
    errors = kx.independence_errors(
        [repeat], entry=ENTRY, gpu_uuid=G1, foreign_incident_ids={"inc-counter"}
    )
    assert len(errors) == 1 and "neither a marker association" in errors[0], errors
    assert (
        kx.independence_errors(
            [repeat, _associated(event_id="e-2", incident_id="inc-e-2")],
            entry=ENTRY,
            gpu_uuid=G1,
            foreign_incident_ids={"inc-counter"},
        )
        == []
    ), "the association-only decision is the event's own; the repeat is ignored"


def test_isolating_decisions_pick_actions_safety_and_workflows() -> None:
    quiet = _decision()
    loud = [
        _decision(event_id="a", safety_action="QUARANTINE"),
        _decision(event_id="b", action="RESET_GPU"),
        _decision(event_id="c", workflow_request_id="wf"),
    ]
    assert kx.isolating_decisions([quiet, *loud]) == loud
    assert (
        kx.isolating_decisions([quiet, _decision(decision=False, action="RESET_GPU")])
        == []
    )


# --------------------------------------------------------------------------- #
# inject / wait over fakes
# --------------------------------------------------------------------------- #
def _host_script(lines: int = 6, *, consumed: bool | None = True) -> dict[str, Any]:
    injected: dict[str, Any] = {"mode": "kmsg", "fixture_lines": lines}
    if consumed is not None:
        injected.update(
            {
                "fixture_bytes": 110 * lines,
                "consumed": consumed,
                "followed": consumed,
                "stream_reopens": 0 if consumed else 91,
                "collector": {
                    "main_pid": "2073979",
                    "fixture_open": consumed,
                    "offset": 110 * lines if consumed else None,
                },
            }
        )
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}, "mode": "kmsg"},
        "arm": {"armed": True, "lines_written": lines},
        "status": {"injected": injected},
        "close-window": {"clean": True},
    }


def _ctx(tmp_path: Path, cpu: Any, fake: FakeProbe) -> Any:
    spec = next(item for item in VARIANTS if item.variant == "independent_xid")
    regional = FakeRegional(cpu_python=cpu)
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
    )


def test_inject_installs_opens_arms_reads_back_and_closes(tmp_path: Path) -> None:
    fake = FakeProbe(_host_script())
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    injection = kx.inject_kernel_xid(ctx, xid=137, pci_bdf="0000:1b:00", label="t")
    verbs = [call[0] for call in fake.calls]
    assert verbs == ["install-records", "open-window", "arm", "status"]
    opened = next(call for call in fake.calls if call[0] == "open-window")
    assert "--approved" in opened and "kmsg" in opened
    assert injection.message.startswith("NVRM: Xid (PCI:0000:1b:00): 137,"), (
        injection.message
    )
    assert injection.xid == 137 and injection.pci_bdf == "0000:1b:00"
    assert (ctx.case_dir / "kernel-xid-137-open.json").is_file(), "open receipt"
    assert (ctx.case_dir / "kernel-xid-137-arm.json").is_file(), "arm receipt"
    assert "kernel-replay-kmsg" in ctx.cleanup.pending(), (
        "the window's cleanup is on the variant's LIFO stack"
    )
    assert injection.close() == {"clean": True}
    assert injection.close() == {}, "closing twice is a no-op"
    assert fake.calls[-1][0] == "close-window"
    evidence = injection.evidence()
    assert evidence["proof_scope"] == "live-replay"
    assert evidence["mode"] == "kmsg" and evidence["xid"] == 137
    assert evidence["records_sha256"] == injection.records_sha256


def test_inject_is_inconclusive_when_the_fixture_got_no_line(tmp_path: Path) -> None:
    fake = FakeProbe(_host_script(lines=0))
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    with pytest.raises(VariantInconclusive, match="no reviewed XID 137"):
        kx.inject_kernel_xid(ctx, xid=137, pci_bdf="0000:1b:00", label="t")


def test_inject_is_inconclusive_when_the_collector_never_read_the_fixture(
    tmp_path: Path,
) -> None:
    """Live R13g: the fixture held every copy of the line and the collector,
    reopening it at EOF once a second, read none -- not a grading verdict."""

    fake = FakeProbe(_host_script(consumed=False))
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    with pytest.raises(VariantInconclusive) as caught:
        kx.inject_kernel_xid(ctx, xid=137, pci_bdf="0000:1b:00", label="t")
    message = str(caught.value)
    assert message.startswith("XID 137: the kernel collector did not follow"), message
    assert "91 time(s)" in message, message
    assert "fixture open in the collector=False" in message, message
    assert "kernel-replay-kmsg" in ctx.cleanup.pending(), (
        "the window is still closed by the cleanup stack"
    )


def test_inject_accepts_a_probe_without_the_consumption_oracle(tmp_path: Path) -> None:
    fake = FakeProbe(_host_script(consumed=None))
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    injection = kx.inject_kernel_xid(ctx, xid=137, pci_bdf="0000:1b:00", label="t")
    assert injection.xid == 137


def test_inject_refuses_a_non_monitor_grade_code_before_touching_the_node(
    tmp_path: Path,
) -> None:
    fake = FakeProbe(_host_script())
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    with pytest.raises(VariantNotRun):
        kx.inject_kernel_xid(ctx, xid=46, pci_bdf="0000:1b:00", label="t")
    assert fake.calls == [], "nothing reached the node"


def test_inject_reviewed_kernel_line_sends_the_message_as_given(tmp_path: Path) -> None:
    """The caller owns the live-safety argument; the line travels unchanged."""

    fake = FakeProbe(_host_script())
    ctx = _ctx(tmp_path, lambda *_: {"decisions": []}, fake)
    message = (
        "NVRM: Xid (PCI:0000:1b:00): 149, pid=0, name=gfrs-live, "
        "NETIR_LINK_EVT Fatal XC0 i0 Link 08 (0x004505c6 0x00000000)"
    )
    injection = kx.inject_reviewed_kernel_line(
        ctx, xid=149, pci_bdf="0000:1b:00", message=message, label="032"
    )
    assert injection.message == message and injection.xid == 149
    assert [call[0] for call in fake.calls] == [
        "install-records",
        "open-window",
        "arm",
        "status",
    ]
    assert injection.records_sha256 == kx.records_sha256(
        [
            kx.reviewed_record(
                message, transport="kernel", emissions=kx.EMISSIONS, label="032"
            )
        ]
    ), "the installed record is the given line, kernel transport"
    with pytest.raises(ValueError, match="does not carry XID 150"):
        kx.inject_reviewed_kernel_line(
            ctx, xid=150, pci_bdf="0000:1b:00", message=message, label="032"
        )
    with pytest.raises(ValueError, match="PCI BDF"):
        kx.inject_reviewed_kernel_line(
            ctx, xid=149, pci_bdf="nope", message=message, label="032"
        )


def test_wait_xid_decisions_filters_the_code_and_records_evidence(
    tmp_path: Path,
) -> None:
    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        assert script == kx.XID_DECISION_PROBE, "the decision read uses the probe"
        assert arguments[1] == NODE
        return {"decisions": [_decision(), _decision(xid=92, event_id="other")]}

    ctx = _ctx(tmp_path, cpu, FakeProbe(_host_script()))
    out = kx.wait_xid_decisions(ctx, datetime.now(timezone.utc), xid=137, name="t")
    assert [item["xid"] for item in out] == [137]
    assert (ctx.case_dir / "kernel-xid-137-decisions.json").is_file(), (
        "the decisions read is written to evidence"
    )


def test_wait_xid_decisions_returns_the_last_read_when_no_decision_arrives(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        lambda *_: {"decisions": [_decision(decision=False)]},
        FakeProbe(_host_script()),
    )
    out = kx.wait_xid_decisions(
        ctx, datetime.now(timezone.utc), xid=137, name="t", timeout_seconds=5
    )
    assert len(out) == 1 and out[0]["decision"] is False
    errors = kx.independence_errors(
        out, entry=ENTRY, gpu_uuid=G1, foreign_incident_ids=set()
    )
    assert any("no FaultPolicyDecision" in error for error in errors), errors


def test_restore_if_isolated_only_restores_isolating_open_incidents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, lambda *_: {}, FakeProbe(_host_script()))
    assert kx.restore_if_isolated(ctx, [_decision()]) == []
    seen: list[list[dict[str, Any]]] = []

    def restore(_ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        seen.append(list(incidents))
        return []

    monkeypatch.setattr(general_live_common, "restore_incidents", restore)
    quarantined = _decision(
        disposition="SITE_SAFETY",
        safety_action="QUARANTINE",
        incident_id="inc-q",
        incident_state="QUARANTINED",
    )
    assert kx.restore_if_isolated(ctx, [quarantined]) == []
    assert seen == [[{"incident_id": "inc-q"}]]

    def failing(_ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        raise RuntimeError("restore refused")

    monkeypatch.setattr(general_live_common, "restore_incidents", failing)
    errors = kx.restore_if_isolated(ctx, [quarantined])
    assert errors and "restore refused" in errors[0], errors
