"""GF-GENERAL-LIVE-016 severe kernel log: category-module verdict tests.

Driven over fakes (no systemd, journalctl or control plane): a ``FakeProbe``
stands in for the kernel-log replay host probe and a scripted ``FakeRegional``
answers the node-log activity reads, so each variant's finding -> incident ->
cordon/quarantine -> notification chain is graded on realistic documents. The
negative variants prove the trusted-source and current-boot gates.
"""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import general_live_kernel as mod
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_kernel_replay import replay_entry_prefix
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests._builders import build_store
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
#: ``rdma-devices`` on a live node: rdma-core's udev rule named every EFA
#: device after its PCI location (HyperPod GL-013 survey, plain EKS 11-04).
RDMA_DEVICES = {
    "devices": [
        {"name": "rdmap86s0", "pci_bdf": "0000:56:00.0", "driver": "efa"},
        {"name": "rdmap85s0", "pci_bdf": "0000:55:00.0", "driver": "efa"},
        {"name": "mlx5_0", "pci_bdf": "0000:3b:00.0", "driver": "mlx5_core"},
    ]
}
#: The reviewed ``rdma_fatal`` line those devices make (first EFA by name).
LIVE_RDMA_FATAL = (
    "efa 0000:55:00.0 rdmap85s0: Failed to process command GET_STATS (opcode 13) "
    "err -62"
)


def _spec(variant: str) -> object:
    return next(item for item in mod.VARIANTS if item.variant == variant)


def _host_script() -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
        },
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
        "rdma-devices": RDMA_DEVICES,
    }


def _generation_answer(message: str) -> dict[str, Any]:
    """``EFA_LINE_GENERATION_PROBE`` run in-process over this product."""

    import contextlib
    import io

    out = io.StringIO()
    argv = sys.argv
    sys.argv = ["efa-line-generation-probe", message]
    try:
        with contextlib.redirect_stdout(out):
            exec(  # noqa: S102 - the reviewed probe text, as the API Pod runs it
                compile(mod.EFA_LINE_GENERATION_PROBE, "<efa-generation>", "exec"),
                {"__name__": "__main__"},
            )
    finally:
        sys.argv = argv
    return json.loads(out.getvalue().strip().splitlines()[-1])


def _efa_store(
    activity: dict[str, object],
    *,
    generation: dict[str, Any] | None = None,
    received: list[dict[str, Any]] | Exception | None = None,
) -> Callable[..., dict[str, Any]]:
    """``store_script`` plus the two reads an EFA variant adds: the product's
    rule table (the real probe over this product unless ``generation``) and
    the control plane's copies of the reviewed record."""

    base = store_script({"activity": activity})

    def dispatch(script: str, *arguments: str) -> dict[str, Any]:
        if script == mod.EFA_LINE_GENERATION_PROBE:
            if generation is not None:
                return dict(generation)
            return _generation_answer(arguments[0])
        if script == mod.NODE_LOG_ENTRY_PROBE:
            if isinstance(received, Exception):
                raise received
            return {"entries": list(received or [])}
        return base(script, *arguments)

    return dispatch


def _log_incident(
    *,
    incident_id: str = "inc-log-1",
    event_id: str = "log-gpu-eks-cluster-node-abc",
    policy_source: str = "SITE_KERNEL_LOG",
    action: str = "QUARANTINE",
    state: str = "QUARANTINED",
    workflow: str = "wf-log-1",
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "event_id": event_id,
        "event_type": "NODE_HEALTH",
        "policy_source": policy_source,
        "effective_action": action,
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": [NODE],
    }


def _log_workflow(
    operations: list[str],
    *,
    incident_id: str = "inc-log-1",
    request_id: str = "wf-log-1",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in operations],
        "step_executions": [],
    }


def _notification(incident_id: str = "inc-log-1") -> dict[str, object]:
    return {
        "notification_id": "ntf-1",
        "incident_id": incident_id,
        "deduplication_key": f"kernel-log/{incident_id}",
        "subject": "severe kernel log",
        "status": "SENT",
    }


def _activity(
    *,
    incidents: list[dict[str, object]],
    workflows: list[dict[str, object]],
    notifications: list[dict[str, object]] | None = None,
    markers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "markers": markers or [],
        "notifications": notifications or [],
        "batches": [],
        "metric_latest": [],
        "collector_statuses": [],
    }


def _ctx(variant: str, *, regional: FakeRegional, tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx, fake


def _isolated_node(isolated: bool) -> dict[str, object]:
    return {
        "name": NODE,
        "ready": "True",
        "unschedulable": isolated,
        "taints": [{"key": "gpu-fault.io/quarantine"}] if isolated else [],
        "ownership_annotations": {"gpu-fault.io/incident-id": "inc-log-1"}
        if isolated
        else {},
    }


def test_rdma_fatal_quarantines_and_notifies(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[
            _log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"])
        ],
        notifications=[_notification()],
    )
    regional = FakeRegional(
        cpu_python=_efa_store(activity), node_snapshot=_isolated_node(True)
    )
    ctx, fake = _ctx("rdma_fatal", regional=regional, tmp_path=tmp_path)
    result = mod.rdma_fatal(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result
    assert result["product_generation"]["graded"] is True, result["product_generation"]
    assert "product_gap" not in result, "this product grades the line: no gap"
    install = next(call for call in fake.calls if call[0] == "install-records")
    [record] = json.loads(base64.b64decode(install[install.index("--b64") + 1]))
    assert record["message"] == LIVE_RDMA_FATAL, record
    assert record["transport"] == "kernel", record
    order = [call[0] for call in fake.calls]
    assert order.index("rdma-devices") < order.index("install-records"), (
        "the device is read before anything is installed"
    )


def test_rdma_fatal_before_b251_is_the_expected_fail_with_the_gap_named(
    tmp_path: Path,
) -> None:
    # A product whose prefix takes only efa_N and whose timeout errno is -110
    # matches no rule on the node's real line: the node never ships it, nothing
    # is minted and the node stays schedulable. That FAIL is the gap, named.
    regional = FakeRegional(
        cpu_python=_efa_store(
            _activity(incidents=[], workflows=[]),
            generation={"present": True, "shipped": [], "trusted": []},
        ),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("rdma_fatal", regional=regional, tmp_path=tmp_path)
    result = mod.rdma_fatal(ctx)
    assert result["verdict"] == "FAIL", "the runner never flips a verdict"
    assert result["expected_outcome"] == "FAIL", result
    assert result["product_gap"] == mod.EFA_RENAMED_DEVICE_GAP, result
    assert result["expected_gap"] == mod.EFA_RENAMED_DEVICE_GAP, result
    assert result["product_generation"]["graded"] is False, result
    assert any(note.startswith(mod.PRE_B251_GENERATION) for note in result["notes"]), (
        result["notes"]
    )
    assert result["received"] == [], "the node never shipped the line"
    assert "batch 251" in mod.EFA_RENAMED_DEVICE_GAP, "the gap names its fix"


def test_a_failed_read_of_the_received_copies_leaves_the_fail_graded(
    tmp_path: Path,
) -> None:
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    regional = FakeRegional(
        cpu_python=_efa_store(
            _activity(incidents=[], workflows=[]),
            generation={"present": True, "shipped": [], "trusted": []},
            received=RegionalFixtureError("the API Pod was replaced"),
        ),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("rdma_fatal", regional=regional, tmp_path=tmp_path)
    result = mod.rdma_fatal(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["product_gap"] == mod.EFA_RENAMED_DEVICE_GAP, result
    assert "the API Pod was replaced" in result["received_error"], result


def test_a_graded_line_that_still_fails_is_a_plain_fail(tmp_path: Path) -> None:
    # The deployed product grades the line, yet nothing came of it: a product
    # (or delivery) failure of its own, never relabelled as the b251 gap.
    regional = FakeRegional(
        cpu_python=_efa_store(_activity(incidents=[], workflows=[])),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("rdma_fatal", regional=regional, tmp_path=tmp_path)
    result = mod.rdma_fatal(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["product_generation"]["graded"] is True, result
    assert "product_gap" not in result and "expected_gap" not in result, result
    assert result.get("expected_outcome", "PASS") == "PASS", result


def test_rdma_fatal_without_an_efa_device_is_not_run(tmp_path: Path) -> None:
    regional = FakeRegional(
        cpu_python=_efa_store(_activity(incidents=[], workflows=[])),
        node_snapshot=_isolated_node(False),
    )
    spec = _spec("rdma_fatal")
    fake = FakeProbe(
        {
            **_host_script(),
            "rdma-devices": {
                "devices": [
                    {"name": "mlx5_0", "pci_bdf": "0000:3b:00.0", "driver": "mlx5_core"}
                ]
            },
        }
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    with pytest.raises(VariantNotRun) as raised:
        mod.rdma_fatal(ctx)
    assert raised.value.reason == "CAPABILITY_ABSENT", raised.value
    assert [call[0] for call in fake.calls] == ["rdma-devices"], (
        "nothing is installed on a node without an EFA device"
    )


@pytest.mark.parametrize(
    "devices, expected",
    [
        (RDMA_DEVICES["devices"], ("rdmap85s0", "0000:55:00.0")),
        # The registered name, before rdma-core renamed it.
        (
            [{"name": "efa_0", "pci_bdf": "0000:00:1d.0", "driver": "efa"}],
            ("efa_0", "0000:00:1d.0"),
        ),
        # A device whose PCI function or driver did not read is skipped.
        (
            [
                {"name": "rdmap100s0", "pci_bdf": None, "driver": "efa"},
                {"name": "rdmap101s0", "pci_bdf": "0000:65:00.0", "driver": None},
                {"name": "rdmap110s0", "pci_bdf": "0000:6e:00.0", "driver": "efa"},
            ],
            ("rdmap110s0", "0000:6e:00.0"),
        ),
    ],
)
def test_efa_device_takes_the_first_efa_function_by_name(
    devices: list[dict[str, Any]], expected: tuple[str, str]
) -> None:
    device = mod.efa_device({"devices": devices})
    assert (device.name, device.pci_bdf) == expected, device
    assert device.prefix == f"efa {expected[1]} {expected[0]}: ", device


@pytest.mark.parametrize(
    "device",
    [
        mod.EfaDevice("rdmap85s0", "0000:55:00.0"),
        mod.EfaDevice("rdmap110s0", "0000:6e:00.0"),
        mod.EfaDevice("efa_0", "0000:00:1d.0"),
    ],
)
def test_the_reviewed_efa_line_is_graded_critical_quarantine_by_the_product(
    device: mod.EfaDevice,
) -> None:
    message = mod.rdma_fatal_message(device)
    assert message == (
        f"efa {device.pci_bdf} {device.name}: "
        "Failed to process command GET_STATS (opcode 13) err -62"
    ), "__ibdev_printk()'s driver, PCI name and ib name before the 3.3.1g text"
    assert _product_decision(message) == (
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "SITE_KERNEL_LOG",
    ), message
    answer = _generation_answer(message)
    assert mod.EfaLineGeneration(message, answer).graded is True, answer


def test_quarantine_fails_when_workflow_plans_hardware(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[
            _log_workflow(
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE", "RESET_GPU"]
            )
        ],
        notifications=[_notification()],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("uncorrectable_mce", regional=regional, tmp_path=tmp_path)
    result = mod.uncorrectable_mce(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("RESET_GPU" in error for error in result["errors"]), result["errors"]


def test_inconclusive_when_nothing_was_ingested(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    regional = FakeRegional(
        cpu_python=store_script({"activity": _activity(incidents=[], workflows=[])}),
        node_snapshot=_isolated_node(False),
    )
    spec = _spec("rdma_fatal")
    fake = FakeProbe(
        {**_host_script(), "status": {"injected": {"emission_counts": {}}}}
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.rdma_fatal(ctx)
    except VariantInconclusive as exc:
        assert "no reviewed records" in str(exc), str(exc)
    else:
        raise AssertionError("an empty injection oracle must be INCONCLUSIVE")


def _product_decision(message: str) -> tuple[Severity, RecoveryAction, str]:
    """What the product's own rule chain makes of a trusted kernel line."""

    [finding] = NodeHealthPolicy(build_store()).evaluate_logs(
        NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            entries=[
                NodeLogEntry(
                    entry_id="gfrs-replay-run-0-boot",
                    source="dmesg",
                    observed_at=NOW,
                    message=message,
                    fields={"_TRANSPORT": "kernel"},
                )
            ],
        )
    )
    return finding.severity, finding.recommended_action, finding.policy_source


@pytest.mark.parametrize(
    "variant", ["uncorrectable_mce", "readonly_storage", "kernel_lockup"]
)
def test_severe_reviewed_lines_are_graded_critical_quarantine_by_the_product(
    variant: str,
) -> None:
    # The harness and the product must agree on the reviewed text before a
    # live run spends a node on it: a severe line reaches the kernel-log
    # quarantine path, not a diagnostic one.
    assert _product_decision(mod.SEVERE_MESSAGES[variant]) == (
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "SITE_KERNEL_LOG",
    ), variant


def test_corrected_reviewed_line_is_graded_diagnostic_by_the_product() -> None:
    assert _product_decision(mod.CORRECTED_MCE_MESSAGE) == (
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "SITE_APPLICATION_LOG",
    ), "the __print_mce header with UC clear is a corrected MCE"


def test_corrected_mce_installs_the_kernel_header_line(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[
            _log_incident(
                policy_source="SITE_APPLICATION_LOG",
                action="RUN_DIAGNOSTICS",
                state="ACTION_PENDING",
            )
        ],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(False),
    )
    ctx, fake = _ctx("corrected_mce", regional=regional, tmp_path=tmp_path)
    mod.corrected_mce(ctx)
    install = next(call for call in fake.calls if call[0] == "install-records")
    encoded = install[install.index("--b64") + 1]
    [record] = json.loads(base64.b64decode(encoded))
    assert record["message"] == mod.CORRECTED_MCE_MESSAGE, record
    assert record["message"].startswith(
        "mce: [Hardware Error]: CPU 1: Machine Check: 0 Bank 7: "
    ), "the reviewed record is the kernel's __print_mce header, not a paraphrase"
    assert record["transport"] == "kernel", record


def test_corrected_mce_is_diagnostic_not_isolation(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[
            _log_incident(
                policy_source="SITE_APPLICATION_LOG",
                action="RUN_DIAGNOSTICS",
                state="ACTION_PENDING",
            )
        ],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("corrected_mce", regional=regional, tmp_path=tmp_path)
    result = mod.corrected_mce(ctx)
    assert result["verdict"] == "PASS", result["errors"]


BOOT = "7679ce8a-3ab8-484a-8d34-4ca9eab39fe5"


def _received(variant: str, **entry: object) -> list[dict[str, object]]:
    """``NODE_LOG_ENTRY_PROBE``'s answer: the control plane stored the reviewed
    record under this run's replay cursor (``_ctx`` names the run)."""

    run = f"gl-{variant.replace('_', '-')}-a1-deadbeef"
    return [
        {
            "record_id": "node-logs/logs-1",
            "batch_id": "logs-1",
            "collected_at": "2026-10-24T00:01:00+00:00",
            "entry_id": replay_entry_prefix(run, 0) + BOOT,
            "source": "journal",
            "transport": "syslog",
            "boot_id": BOOT,
            "collector_boot_id": BOOT,
            "message": mod.SEVERE_MESSAGES["uncorrectable_mce"],
            **entry,
        }
    ]


def _negative_store(
    activity: dict[str, object], received: list[dict[str, object]]
) -> object:
    store = store_script({"activity": activity})

    def dispatch(script: str, *arguments: str) -> dict[str, object]:
        if script == mod.NODE_LOG_ENTRY_PROBE:
            return {"entries": received}
        return store(script, *arguments)

    return dispatch


def test_user_text_produces_no_finding(tmp_path: Path) -> None:
    regional = FakeRegional(
        cpu_python=_negative_store(
            _activity(incidents=[], workflows=[]), _received("user_text")
        ),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("user_text", regional=regional, tmp_path=tmp_path)
    result = mod.user_text(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["received"][0]["source"] == "journal", result["received"]


def test_user_text_is_inconclusive_when_the_product_never_received_it(
    tmp_path: Path,
) -> None:
    """The old pass: the wrapper's emission count and an empty activity read.
    Nothing says the control plane ever graded the line."""

    regional = FakeRegional(
        cpu_python=_negative_store(_activity(incidents=[], workflows=[]), []),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("user_text", regional=regional, tmp_path=tmp_path)
    result = mod.user_text(ctx)
    assert result["verdict"] == "INCONCLUSIVE", result
    assert "stored no NODE_LOGS batch carrying the reviewed record" in result["detail"]


@pytest.mark.parametrize(
    "variant,entry,wanted",
    [
        ("user_text", {"source": "dmesg", "transport": "kernel"}, "user-space"),
        ("old_boot", {"source": "dmesg", "transport": "kernel"}, "prior boot"),
    ],
)
def test_a_record_received_without_the_refused_trait_is_inconclusive(
    tmp_path: Path, variant: str, entry: dict[str, object], wanted: str
) -> None:
    """The gate's refusal is only proved on the trait it refuses: a user_text
    line that arrived as a kernel line, an old_boot line stamped with the
    current boot."""

    regional = FakeRegional(
        cpu_python=_negative_store(
            _activity(incidents=[], workflows=[]), _received(variant, **entry)
        ),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx(variant, regional=regional, tmp_path=tmp_path)
    result = getattr(mod, variant)(ctx)
    assert result["verdict"] == "INCONCLUSIVE", result
    assert wanted in result["detail"], result["detail"]


def test_old_boot_passes_on_a_prior_boot_kernel_line(tmp_path: Path) -> None:
    received = _received(
        "old_boot", source="dmesg", transport="kernel", boot_id=mod.OLD_BOOT_ID
    )
    regional = FakeRegional(
        cpu_python=_negative_store(_activity(incidents=[], workflows=[]), received),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("old_boot", regional=regional, tmp_path=tmp_path)
    result = mod.old_boot(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_user_text_fails_if_it_isolates(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "QUARANTINE"])],
    )
    regional = FakeRegional(
        cpu_python=_negative_store(activity, _received("user_text")),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("user_text", regional=regional, tmp_path=tmp_path)
    result = mod.user_text(ctx)
    assert result["verdict"] == "FAIL", result


def test_any_finding_of_the_reviewed_record_fails_a_negative(tmp_path: Path) -> None:
    """Not only a QUARANTINE: a diagnostic finding the record produced is a
    finding the gate should never have let through (the old stage counted
    QUARANTINE incidents only). Its marker names the replay cursor."""

    [entry] = _received("user_text")
    incident = _log_incident(
        policy_source="SITE_APPLICATION_LOG",
        action="RUN_DIAGNOSTICS",
        state="ACTION_PENDING",
    )
    marker = {
        "marker_id": "marker-log-1",
        "incident_id": incident["incident_id"],
        "raw_evidence_ref": f"journal://{NODE}/{entry['entry_id']}",
    }
    activity = _activity(
        incidents=[incident],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
        markers=[marker],
    )
    for received in ([entry], []):
        regional = FakeRegional(
            cpu_python=_negative_store(activity, received),
            node_snapshot=_isolated_node(False),
        )
        ctx, _ = _ctx(
            "user_text", regional=regional, tmp_path=tmp_path / str(len(received))
        )
        result = mod.user_text(ctx)
        assert result["verdict"] == "FAIL", (received, result)
        assert result["stages"]["no_finding"] == [
            f"the reviewed record {entry['entry_id']} produced a finding: marker "
            f"marker-log-1 on incident {incident['incident_id']}"
        ], result["stages"]


def test_unknown_attempt_isolates_without_stop(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[
            _log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"])
        ],
    )
    regional = FakeRegional(
        cpu_python=_efa_store(activity), node_snapshot=_isolated_node(True)
    )
    ctx, fake = _ctx("unknown_attempt", regional=regional, tmp_path=tmp_path)
    result = mod.unknown_attempt(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    install = next(call for call in fake.calls if call[0] == "install-records")
    [record] = json.loads(base64.b64decode(install[install.index("--b64") + 1]))
    assert record["message"] == LIVE_RDMA_FATAL, "the node's own EFA device"


def test_unknown_attempt_fails_if_it_stops_a_workload(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[
            _log_workflow(["MARK_UNSCHEDULABLE", "STOP_WORKLOADS", "QUARANTINE"])
        ],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("unknown_attempt", regional=regional, tmp_path=tmp_path)
    result = mod.unknown_attempt(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("stopped a workload" in error for error in result["errors"]), result[
        "errors"
    ]


#: The boot the reviewed records' synthetic cursors name (``open-window``'s
#: ``boot_id``); the run id is ``make_context``'s.
BOOT = "7679ce8a-3ab8-484a-8d34-4ca9eab39fe5"
PAGINATION_RUN = "gl-cursor-pagination-a1-deadbeef"
PAGINATION_LABELS = ("mce", "storage", "lockup")


def _record_marker(index: int, incident_id: str) -> dict[str, object]:
    """The marker the product writes for reviewed record ``index``'s finding.

    ``NodeHealthFinding.marker()``: ``marker-<event id>``, the entry it was read
    from in ``raw_evidence_ref`` (``dmesg://<node>/<entry id>``, the entry id
    being the wrapper's cursor) and, once ingestion picked the incident,
    that incident's id -- the incident the finding was *routed* to, which for
    the second and third finding of a burst is the first one's.
    """

    return {
        "marker_id": f"marker-log-gpu-eks-cluster-{NODE}-{index:032x}",
        "incident_id": incident_id,
        "raw_evidence_ref": (
            f"dmesg://{NODE}/gfrs-replay-{PAGINATION_RUN}-{index}-{BOOT}"
        ),
        "raw_reason": [
            "CPU or memory hardware error",
            "persistent storage I/O failure",
            "kernel liveness failure",
        ][index],
        "severity": "critical",
        "trusted": True,
    }


def _group_incident() -> dict[str, object]:
    """The live record of 2026-10-27 18:46:32Z, R13h: one incident for the burst.

    The first finding (record 0, the MCE) minted ``inc-<its event id>``; the
    storage and lockup findings joined that in-flight isolation
    (``families/drain.py`` ``ingest_terminal_node_quarantine``), which
    re-labelled the record ``NODE_HEALTH_GROUP`` and added one reason each.
    """

    return {
        **_log_incident(
            incident_id="inc-log-burst",
            event_id="log-gpu-eks-cluster-burst-0",
            state="ACTION_PENDING",
            workflow="wf-burst",
        ),
        "event_type": "NODE_HEALTH_GROUP",
        "reasons": [
            "CPU or memory hardware error",
            f"{NODE}: STORAGE: persistent storage I/O failure",
            f"{NODE}: SYSTEM_LOG: kernel liveness failure",
        ],
    }


def _pagination_ctx(tmp_path: Path, activity: dict[str, object]) -> object:
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    spec = _spec("cursor_pagination")
    script = {
        **_host_script(),
        "open-window": {
            "after": {"ActiveState": "active"},
            "boot_id": BOOT,
            "proof_scope": "live-replay",
        },
        "status": {"injected": {"emission_counts": {"0": 4, "1": 4, "2": 4}}},
    }
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": FakeProbe(script)},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    assert ctx.run_id == PAGINATION_RUN, ctx.run_id
    return ctx


def test_cursor_pagination_passes_on_the_products_one_isolation_for_the_burst(
    tmp_path: Path,
) -> None:
    # Live 2026-10-27 18:45:47-18:55:56Z (R13h, plain EKS ip-10-0-43-5) and
    # 2026-10-26 (R13g, HyperPod and EKS): FAIL "only 0/3 severe kernel records
    # reached an incident; the cursor starved the rest" -- while the first
    # armed poll had delivered all three records in one batch and the product
    # had folded them into one NODE_HEALTH_GROUP incident (three reasons, three
    # markers) and quarantined the node. The grade took only NODE_HEALTH and
    # wanted one incident per record.
    activity = _activity(
        incidents=[_group_incident()],
        workflows=[
            _log_workflow(
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
                incident_id="inc-log-burst",
                request_id="wf-burst",
            )
        ],
        markers=[_record_marker(index, "inc-log-burst") for index in range(3)],
    )
    result = mod.cursor_pagination(_pagination_ctx(tmp_path, activity))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["incidents"] == ["inc-log-burst"], result
    assert {
        label: (link or {}).get("incident_id")
        for label, link in result["records"].items()
    } == dict.fromkeys(PAGINATION_LABELS, "inc-log-burst"), result["records"]


def test_cursor_pagination_does_not_depend_on_the_grouping(tmp_path: Path) -> None:
    # Records that reach the control plane after the first isolation settled
    # mint incidents of their own; each record's marker still names its own.
    incidents = [
        _log_incident(
            incident_id=f"inc-log-{i}", event_id=f"log-c-{i}", workflow=f"wf-{i}"
        )
        for i in range(3)
    ]
    workflows = [
        _log_workflow(
            ["MARK_UNSCHEDULABLE", "QUARANTINE"],
            incident_id=f"inc-log-{i}",
            request_id=f"wf-{i}",
        )
        for i in range(3)
    ]
    activity = _activity(
        incidents=incidents,
        workflows=workflows,
        markers=[_record_marker(i, f"inc-log-{i}") for i in range(3)],
    )
    result = mod.cursor_pagination(_pagination_ctx(tmp_path, activity))
    assert result["verdict"] == "PASS", result["errors"]
    assert len(result["incidents"]) == 3, result


def test_cursor_pagination_names_the_record_that_reached_no_incident(
    tmp_path: Path,
) -> None:
    activity = _activity(
        incidents=[_group_incident()],
        workflows=[
            _log_workflow(
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
                incident_id="inc-log-burst",
                request_id="wf-burst",
            )
        ],
        # The lockup record's finding is missing: it was starved.
        markers=[_record_marker(index, "inc-log-burst") for index in range(2)],
    )
    result = mod.cursor_pagination(_pagination_ctx(tmp_path, activity))
    assert result["verdict"] == "FAIL", result
    [error] = result["stages"]["all_processed"]
    assert error.startswith("only 2/3 severe kernel records"), error
    assert "lockup" in error and "storage" not in error, error
    assert result["records"]["lockup"] is None, result["records"]


def test_cursor_pagination_counts_no_incident_older_than_the_injection(
    tmp_path: Path,
) -> None:
    # The activity read also lists the incident of any workflow created since
    # the injection (a validated restore of an older incident): a marker on
    # such an incident is not this run's finding reaching an incident.
    old = {**_group_incident(), "created_at": "2026-10-23T23:00:00+00:00"}
    activity = {
        **_activity(
            incidents=[old],
            workflows=[],
            markers=[_record_marker(index, "inc-log-burst") for index in range(3)],
        ),
        "since": "2026-10-24T00:00:00+00:00",
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
    }
    result = mod.cursor_pagination(_pagination_ctx(tmp_path, activity))
    assert result["verdict"] == "FAIL", result
    assert result["incidents"] == [], result


@pytest.mark.parametrize(
    "label",
    ["NODE_HEALTH", "NODE_HEALTH_GROUP", "NODE_RESOURCE_GROUP", "GPU_FAULT_GROUP"],
)
def test_a_severe_record_counts_under_every_label_the_product_gives_its_incident(
    tmp_path: Path, label: str
) -> None:
    activity = _activity(
        incidents=[{**_log_incident(), "event_type": label}],
        workflows=[
            _log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"])
        ],
        notifications=[_notification()],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("kernel_lockup", regional=regional, tmp_path=tmp_path)
    result = mod.kernel_lockup(ctx)
    assert result["verdict"] == "PASS", (label, result["errors"])


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-016"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "016 registers every coverage variant"
    )
    assert all(item.proof_mode == "live-replay" for item in mod.VARIANTS), (
        "016 is live-replay"
    )
