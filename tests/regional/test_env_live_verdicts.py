"""GF-ENV-LIVE verdicts are judged on the product's own workflow records.

Each ``test_case_0NN`` exercises the pass/fail boundary for one case and is the
``related_pytest`` the catalog points that case at (supporting logic coverage,
never a substitute for a live run, which stays NOT_RUN).
"""

from __future__ import annotations

from scripts.e2e.regional import env_live_verdicts as verdicts

REVIEW_WORKFLOW = {
    "official_steps": [
        {"operation": "FREEZE_EVIDENCE"},
        {"operation": "MARK_UNSCHEDULABLE"},
        {"operation": "STOP_WORKLOADS"},
        {"operation": "QUARANTINE"},
        {"operation": "COLLECT_DIAGNOSTIC_BUNDLE"},
        {"operation": "TRIGGER_HEALTH_SNAPSHOT"},
        {"operation": "ESCALATE_SUPPORT"},
    ]
}
DIAGNOSTIC_WORKFLOW = {
    "official_steps": [
        {"operation": "FREEZE_EVIDENCE"},
        {"operation": "RUN_DCGM_DIAGNOSTIC"},
        {"operation": "VALIDATE_GPU"},
    ]
}


def _severe(**overrides):
    base = {
        "severe": True,
        "source": "DCGM_EXPORTER",
        "source_boot_id": "boot-1",
        "pci_bdf": "59:00",
        "observed_at": "2030-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
def test_case_001_baseline_requires_readings_samples_and_no_action() -> None:
    clean = verdicts.baseline_errors(
        per_gpu_readings={
            "GPU-a": {
                "gpu_temperature_c",
                "power_usage_w",
                "power_limit_w",
                "clock_throttle_reasons",
                "gpu_slowdown_tlimit_c",
                "gpu_shutdown_tlimit_c",
            }
        },
        unsupported={},
        valid_sample_count=3,
        source_boot_id="boot-1",
        collector_mode="DCGM",
        new_environment_actions=[],
        task_progressed=True,
    )
    assert clean == []
    missing = verdicts.baseline_errors(
        per_gpu_readings={"GPU-a": {"gpu_temperature_c"}},
        unsupported={},
        valid_sample_count=1,
        source_boot_id=None,
        collector_mode=None,
        new_environment_actions=["DRAIN"],
        task_progressed=False,
    )
    joined = "\n".join(missing)
    assert "power_usage_w" in joined and "neither" in joined
    assert "only 1 valid" in joined and "no source_boot_id" in joined
    assert "collector mode" in joined and "created environment actions" in joined
    assert "did not advance" in joined


def test_case_001_unsupported_record_satisfies_a_missing_scale() -> None:
    assert (
        verdicts.baseline_errors(
            per_gpu_readings={
                "GPU-a": {"power_usage_w", "power_limit_w", "clock_throttle_reasons"}
            },
            unsupported={
                "GPU-a": {
                    "gpu_temperature_c",
                    "gpu_slowdown_tlimit_c",
                    "gpu_shutdown_tlimit_c",
                }
            },
            valid_sample_count=5,
            source_boot_id="boot-1",
            collector_mode="DCGM",
            new_environment_actions=[],
            task_progressed=True,
        )
        == []
    )


def test_case_002_forbidden_operations_and_diagnostic_only() -> None:
    # Power cap forms a read-only diagnostic candidate; no isolating step.
    assert (
        verdicts.read_only_diagnostic_errors(
            workflow=DIAGNOSTIC_WORKFLOW,
            incident_state="RECOVERED",
            node_isolated=False,
            task_progressed=True,
            label="002",
        )
        == []
    )
    assert verdicts.forbidden_operation_errors(
        {"official_steps": [{"operation": "RESET_GPU"}]}, label="002"
    ) == ["002: forbidden operation RESET_GPU appeared"]


def test_case_003_warning_stays_read_only() -> None:
    assert (
        verdicts.read_only_diagnostic_errors(
            workflow=DIAGNOSTIC_WORKFLOW,
            incident_state="RECOVERED",
            node_isolated=False,
            task_progressed=True,
            label="003",
        )
        == []
    )
    bad = verdicts.read_only_diagnostic_errors(
        workflow=REVIEW_WORKFLOW,
        incident_state="ESCALATED",
        node_isolated=True,
        task_progressed=False,
        label="003",
    )
    text = "\n".join(bad)
    assert "isolating steps" in text and "node was isolated" in text
    assert "task stopped" in text and "escalated to ESCALATED" in text


def test_case_004_severe_review_full_chain() -> None:
    assert (
        verdicts.severe_review_errors(
            confirmed_sample_run=3,
            required_samples=3,
            pre_confirmation_stopped=False,
            workflow=REVIEW_WORKFLOW,
            incident_state="ESCALATED",
            node_cordoned=True,
            quarantine_taint=True,
            stopped_owned_task=True,
            diagnostic_bundle=True,
            health_snapshot=True,
            notification_sent=True,
            reset_or_reboot=False,
            task_restarted=False,
            held_isolated_after_cooldown=True,
            label="004",
        )
        == []
    )


def test_case_004_missing_receipts_and_premature_stop_fail() -> None:
    bad = verdicts.severe_review_errors(
        confirmed_sample_run=2,
        required_samples=3,
        pre_confirmation_stopped=True,
        workflow={"official_steps": [{"operation": "RESET_GPU"}]},
        incident_state="RECOVERED",
        node_cordoned=False,
        quarantine_taint=False,
        stopped_owned_task=False,
        diagnostic_bundle=False,
        health_snapshot=False,
        notification_sent=False,
        reset_or_reboot=True,
        task_restarted=True,
        held_isolated_after_cooldown=False,
        label="004",
    )
    text = "\n".join(bad)
    assert "pre-confirmation severe sample" in text
    assert "confirmation reached 2 of 3" in text
    assert "forbidden operation RESET_GPU" in text
    assert "reset/reboot/RMA" in text and "was restarted" in text
    assert "not cordoned" in text and "quarantine taint is absent" in text
    assert "ESCALATED" in text and "300s cooldown" in text


def test_case_005_transient_and_repeat_do_not_confirm() -> None:
    assert (
        verdicts.transient_errors(
            cleared_through_edge_filter=True,
            confirmation_after_clear=1,
            duplicate_batch_advanced=False,
            stopped=False,
            retriggered_run=3,
            required_samples=3,
            label="005",
        )
        == []
    )
    bad = verdicts.transient_errors(
        cleared_through_edge_filter=False,
        confirmation_after_clear=3,
        duplicate_batch_advanced=True,
        stopped=True,
        retriggered_run=1,
        required_samples=3,
        label="005",
    )
    text = "\n".join(bad)
    assert "did not pass the edge filter" in text
    assert "reached 3 after a clear" in text
    assert "replayed accepted batch advanced" in text
    assert "stopped work" in text and "did not re-trigger" in text


def test_case_006_duration_stays_read_only() -> None:
    assert (
        verdicts.read_only_diagnostic_errors(
            workflow=None,
            incident_state=None,
            node_isolated=False,
            task_progressed=True,
            label="006",
        )
        == []
    )


def test_case_007_power_brake_discrimination() -> None:
    assert (
        verdicts.power_brake_errors(
            transient_isolated=False,
            sustained_review=[],
            non_brake_actions={"0x4": "RUN_DIAGNOSTICS", "0x8": None},
            mixed_recognized_as_brake={"0x84": True, "0xc0": True},
            thermal_warning_downgraded_brake=False,
            receipt_is_power_check=True,
            rebooted_or_auto_recovered=False,
            label="007",
        )
        == []
    )
    bad = verdicts.power_brake_errors(
        transient_isolated=True,
        sustained_review=["007: node was not cordoned"],
        non_brake_actions={"0x4": "DRAIN"},
        mixed_recognized_as_brake={"0x84": False},
        thermal_warning_downgraded_brake=True,
        receipt_is_power_check=False,
        rebooted_or_auto_recovered=True,
        label="007",
    )
    text = "\n".join(bad)
    assert "single brake sample isolated" in text
    assert "ordinary bit 0x4 drove DRAIN" in text
    assert "mixed mask 0x84 was not recognized" in text
    assert "downgraded a confirmed brake" in text
    assert "not a power-delivery check" in text
    assert "reboot or automatic recovery" in text
    assert "node was not cordoned" in text


def test_case_008_unknown_not_fabricated_brake() -> None:
    assert (
        verdicts.unknown_field_errors(
            reported_unknown=True,
            fabricated_brake=False,
            stopped_or_isolated=False,
            missing_field_gpu_judged_alone=True,
            label="008",
        )
        == []
    )
    bad = verdicts.unknown_field_errors(
        reported_unknown=False,
        fabricated_brake=True,
        stopped_or_isolated=True,
        missing_field_gpu_judged_alone=False,
        label="008",
    )
    text = "\n".join(bad)
    assert "not reported UNKNOWN" in text and "read as a Power Brake" in text
    assert "STOP or isolation" in text and "judged by the batch's other GPUs" in text


def test_case_009_interruption_recounts() -> None:
    assert (
        verdicts.interruption_errors(
            recount_from_one=True,
            single_producer=True,
            gap_not_treated_healthy=True,
            source_change_inherited_confirmation=False,
            cross_boot_verified=True,
            label="009",
        )
        == []
    )
    bad = verdicts.interruption_errors(
        recount_from_one=False,
        single_producer=False,
        gap_not_treated_healthy=False,
        source_change_inherited_confirmation=True,
        cross_boot_verified=False,
        label="009",
    )
    text = "\n".join(bad)
    assert "did not restart after the >60s gap" in text
    assert "more than one metrics producer" in text
    assert "treated as healthy" in text and "inherited the old confirmation" in text
    assert "cross-boot path was not verified" in text


def test_case_010_scope_and_dedup() -> None:
    assert (
        verdicts.scope_errors(
            control_metrics_unchanged=True,
            control_task_unaffected=True,
            finding_gpu_scope_correct=True,
            stop_covers_owned_job=True,
            stop_targeted_single_rank=False,
            workflows_merged=True,
            notifications_deduplicated=True,
            duplicate_stop=False,
            label="010",
        )
        == []
    )
    bad = verdicts.scope_errors(
        control_metrics_unchanged=False,
        control_task_unaffected=False,
        finding_gpu_scope_correct=False,
        stop_covers_owned_job=False,
        stop_targeted_single_rank=True,
        workflows_merged=False,
        notifications_deduplicated=False,
        duplicate_stop=True,
        label="010",
    )
    text = "\n".join(bad)
    assert "control GPU's metrics changed" in text
    assert "independent control task was affected" in text
    assert "finding/diagnostic GPU scope is wrong" in text
    assert "did not cover the owned distributed job" in text
    assert "wrongly targeted a single rank" in text
    assert "did not merge" in text and "not deduplicated" in text
    assert "duplicate STOP" in text


def test_case_011_shutdown_one_sample_review() -> None:
    assert (
        verdicts.severe_review_errors(
            confirmed_sample_run=1,
            required_samples=1,
            pre_confirmation_stopped=False,
            workflow=REVIEW_WORKFLOW,
            incident_state="ESCALATED",
            node_cordoned=True,
            quarantine_taint=True,
            stopped_owned_task=True,
            diagnostic_bundle=True,
            health_snapshot=True,
            notification_sent=True,
            reset_or_reboot=False,
            task_restarted=False,
            held_isolated_after_cooldown=True,
            label="011",
        )
        == []
    )


def test_case_012_is_not_run_physical() -> None:
    # 012 is a physical / vendor-lab fault; there is no telemetry-replay verdict
    # to compute. The runner marks it NOT_RUN; this asserts the verdict helper
    # treats an empty error map as PASS and a populated one as FAIL so a real
    # physical run that did observe records is still judged.
    assert verdicts.case_verdict({"a": [], "b": []}) == "PASS"
    assert verdicts.case_verdict({"a": ["x"]}) == "FAIL"


def test_valid_severe_samples_counts_run_and_breaks() -> None:
    run, notes = verdicts.valid_severe_samples(
        [
            _severe(),
            _severe(observed_at="2030-01-01T00:00:10+00:00"),
            _severe(observed_at="2030-01-01T00:00:20+00:00"),
        ]
    )
    assert run == 3 and notes == []
    # A >60s gap restarts the count.
    run, notes = verdicts.valid_severe_samples(
        [_severe(), _severe(observed_at="2030-01-01T00:02:00+00:00")]
    )
    assert run == 1 and any("gap exceeds" in note for note in notes)
    # A source/boot/BDF change restarts the count.
    run, notes = verdicts.valid_severe_samples(
        [
            _severe(),
            _severe(observed_at="2030-01-01T00:00:05+00:00", source_boot_id="boot-2"),
        ]
    )
    assert run == 1 and any("changed" in note for note in notes)
    # A clearing (non-severe) sample and a collection error reset.
    run, _ = verdicts.valid_severe_samples(
        [_severe(), {"severe": False}, _severe(observed_at="2030-01-01T00:00:30+00:00")]
    )
    assert run == 1
    run, notes = verdicts.valid_severe_samples(
        [_severe(), {"collection_error": True, "observed_at": "x"}]
    )
    assert run == 0 and any("collection error" in note for note in notes)


def test_cleanup_errors_and_case_verdict() -> None:
    assert (
        verdicts.cleanup_errors(
            collector_endpoint_restored=True,
            single_producer_after=True,
            fresh_batches_after=3,
            proxy_closed_after_endpoint=True,
            probe_residuals_absent=True,
            label="cleanup",
        )
        == []
    )
    bad = verdicts.cleanup_errors(
        collector_endpoint_restored=False,
        single_producer_after=False,
        fresh_batches_after=0,
        proxy_closed_after_endpoint=False,
        probe_residuals_absent=False,
        label="cleanup",
    )
    assert len(bad) == 5
    assert verdicts.case_verdict({"x": bad}) == "FAIL"
