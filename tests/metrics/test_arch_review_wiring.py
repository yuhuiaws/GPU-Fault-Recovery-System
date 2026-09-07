"""Counters the 2026-09-07 architecture review left in process memory reach
/metrics (ARCH-D6, ARCH-A3, ARCH-G1, ARCH-G2, ARCH-H2).

Each group grew an in-memory counter or snapshot on its own component; none
of them can be alerted on until the metrics family renders it. Every
optional source -- a controller the deployment did not build, a registry
runtime the role does not run, an ingestion service not yet bound to the
context -- renders nothing rather than zero, so a missing series says
"absent", not "quiet".
"""

from __future__ import annotations

from types import SimpleNamespace

from gpu_fault.app.builtin_metric_contributors import (
    control_loop_metric_lines,
    regional_registry_metric_lines,
    remote_command_metric_lines,
    spare_reservation_metric_lines,
)
from gpu_fault.app.metrics import METRIC_CONTRIBUTORS
from gpu_fault.app.metrics_sections import render_processor_metrics_2
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from tests._builders import build_store

OPEN_SIBLING = "gpu_fault_remote_command_open_sibling_holds_total"
SPARE_ACTIVE = "gpu_fault_spare_reservations_active"
SPARE_RECLAIMED = "gpu_fault_spare_reservations_reclaimed_total"
FAULT_REJECTIONS = "gpu_fault_processor_fault_rejections_total"
COMPLETIONS_BY_PATH = "gpu_fault_processor_completions_by_path_status_total"
UNRESOLVED = "gpu_fault_ingest_unresolved_fault_signals_total"
SECRET_DRIFT = "gpu_fault_regional_registry_secret_drift"


def _samples(lines: list[str], name: str) -> list[str]:
    return [
        line
        for line in lines
        if line.startswith(f"{name} ") or line.startswith(f"{name}{{")
    ]


def _remote_stats() -> dict[str, object]:
    return {
        "by_status": {},
        "oldest_unclaimed_age_seconds_by_cluster": {},
        "executor_internal_error_total": 0,
        "executor_internal_error_last_seen_timestamp_seconds": 0.0,
        "unclaimed_expired_total": 0,
    }


def _remote_runtime(adapters: list[object] | None) -> SimpleNamespace:
    context = SimpleNamespace(
        regional_mode=True, store=SimpleNamespace(remote_command_stats=_remote_stats)
    )
    if adapters is not None:
        context.workflow_executor = SimpleNamespace(adapters=adapters)
    return SimpleNamespace(context=context)


def test_open_sibling_holds_are_summed_over_the_regional_adapters() -> None:
    """The adapter is one of the executor's adapters, not a context attribute;
    a plain adapter without the counter is skipped rather than read as 0."""
    adapters = [
        SimpleNamespace(open_sibling_holds_total=3),
        SimpleNamespace(),
        SimpleNamespace(open_sibling_holds_total=4),
    ]

    lines = remote_command_metric_lines(_remote_runtime(adapters))

    assert _samples(lines, OPEN_SIBLING) == [f"{OPEN_SIBLING} 7"], lines
    assert any(line.startswith(f"# TYPE {OPEN_SIBLING} counter") for line in lines), (
        "the hold count only ever grows, so it is a counter"
    )


def test_open_sibling_holds_render_zero_before_any_hold() -> None:
    adapter = RegionalRemoteWorkflowAdapter(build_store(), owners={"regional"})
    assert adapter.open_sibling_holds_total == 0, "a fresh adapter has held nothing"

    lines = remote_command_metric_lines(_remote_runtime([adapter]))

    assert _samples(lines, OPEN_SIBLING) == [f"{OPEN_SIBLING} 0"], lines


def test_open_sibling_holds_render_zero_when_no_executor_is_bound() -> None:
    """A context built without a production executor still publishes the
    series at zero, so a rate() over it never starts from a gap."""
    lines = remote_command_metric_lines(_remote_runtime(None))

    assert _samples(lines, OPEN_SIBLING) == [f"{OPEN_SIBLING} 0"], lines


def test_open_sibling_holds_stay_out_of_a_non_regional_control_plane() -> None:
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            regional_mode=False,
            workflow_executor=SimpleNamespace(
                adapters=[SimpleNamespace(open_sibling_holds_total=9)]
            ),
        )
    )

    assert remote_command_metric_lines(runtime) == [], (
        "the remote-command family is regional-only"
    )


def test_spare_reservations_render_from_the_controller_snapshot() -> None:
    controller = SimpleNamespace(
        metrics_snapshot=lambda: {
            "spare_reservations_active": 2,
            "spare_reservations_reclaimed_total": 5,
            "spare_reservations_observed_at": "2026-09-07T12:00:00+00:00",
        }
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(spare_health_controller=controller)
    )

    lines = spare_reservation_metric_lines(runtime)

    assert _samples(lines, SPARE_ACTIVE) == [f"{SPARE_ACTIVE} 2"], lines
    assert _samples(lines, SPARE_RECLAIMED) == [f"{SPARE_RECLAIMED} 5"], lines
    assert any(line.startswith(f"# TYPE {SPARE_ACTIVE} gauge") for line in lines), (
        "active reservations rise and fall, so the series is a gauge"
    )
    assert any(
        line.startswith(f"# TYPE {SPARE_RECLAIMED} counter") for line in lines
    ), "reclaimed reservations only ever grow, so the series is a counter"
    assert not any("observed" in line for line in lines), (
        "the observed-at stamp is an ISO string, not an epoch; it is not rendered"
    )


def test_spare_reservations_render_nothing_without_a_controller() -> None:
    absent = spare_reservation_metric_lines(
        SimpleNamespace(context=SimpleNamespace(spare_health_controller=None))
    )
    unbound = spare_reservation_metric_lines(SimpleNamespace(context=SimpleNamespace()))

    assert absent == [], "a deployment without spare health publishes no series"
    assert unbound == [], "a context predating the attribute publishes no series"


def _processor_runtime(**overrides: object) -> dict[str, object]:
    runtime: dict[str, object] = {
        "deadline_exceeded_total": 0,
        "completion_retries_total": 0,
        "completion_failures_total": 0,
        "stale_superseded_total": 0,
        "notification_listener_connected": 0,
        "healthy": 1,
    }
    runtime.update(overrides)
    return runtime


def _render_processor_2(runtime: dict[str, object]) -> list[str]:
    io = SimpleNamespace(
        in_flight=0,
        max_in_flight=1,
        rejected_total=0,
        admission_wait_count=0,
        admission_wait_sum_seconds=0.0,
        admission_wait_max_seconds=0.0,
    )
    batcher = SimpleNamespace(pending_depth=0, pending_depth_max=0, submitted_total=0)
    lines: list[str] = []
    render_processor_metrics_2(lines, runtime, {}, 0, batcher, 0, 0, io, io)
    return lines


def test_processor_fault_rejections_and_completions_by_path_render() -> None:
    lines = _render_processor_2(
        _processor_runtime(
            fault_rejections_total=3,
            completions_by_path_status={
                "/v1/nvidia-kernel-events": {"4xx": 3, "2xx": 10},
                "/v1/gpu-metrics": {"2xx": 7},
            },
        )
    )

    assert _samples(lines, FAULT_REJECTIONS) == [f"{FAULT_REJECTIONS} 3"], lines
    assert _samples(lines, COMPLETIONS_BY_PATH) == [
        f'{COMPLETIONS_BY_PATH}{{path="/v1/gpu-metrics",status_class="2xx"}} 7',
        f'{COMPLETIONS_BY_PATH}{{path="/v1/nvidia-kernel-events",status_class="2xx"}} 10',
        f'{COMPLETIONS_BY_PATH}{{path="/v1/nvidia-kernel-events",status_class="4xx"}} 3',
    ], "paths and status classes are sorted so the exposition is stable"
    index = lines.index(f"{FAULT_REJECTIONS} 3")
    assert "gpu_fault_processor_retry_horizon_failures_total" in lines[index - 3], (
        "the G1 counters follow the retry-horizon counter in the processor section"
    )


def test_processor_g1_counters_default_to_zero_and_no_series() -> None:
    """A processor snapshot from before G1 -- or the offline default -- has
    neither key; the scalar counter still reads 0 and the labelled family
    has its header but no samples."""
    lines = _render_processor_2(_processor_runtime())

    assert _samples(lines, FAULT_REJECTIONS) == [f"{FAULT_REJECTIONS} 0"], lines
    assert _samples(lines, COMPLETIONS_BY_PATH) == [], lines
    assert any(
        line.startswith(f"# TYPE {COMPLETIONS_BY_PATH} counter") for line in lines
    ), "the family header is published even when no completion has happened"


def test_unresolved_fault_signals_render_by_kind_from_fault_ingestion() -> None:
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            fault_ingestion=SimpleNamespace(
                unresolved_signal_totals={
                    "unparsed_xid_line": 2,
                    "unclassified_sxid": 1,
                }
            )
        )
    )

    lines = control_loop_metric_lines(runtime)

    assert _samples(lines, UNRESOLVED) == [
        f'{UNRESOLVED}{{kind="unclassified_sxid"}} 1',
        f'{UNRESOLVED}{{kind="unparsed_xid_line"}} 2',
    ], lines
    assert any(line.startswith(f"# TYPE {UNRESOLVED} counter") for line in lines), (
        "unresolved signal totals only grow, so the family is a counter"
    )


def test_unresolved_fault_signals_render_nothing_until_the_service_is_bound() -> None:
    """factory.py binds ``ctx.fault_ingestion`` in another change; until it
    lands, or when it is None, the family is absent -- not a row of zeros."""
    unbound = control_loop_metric_lines(SimpleNamespace(context=SimpleNamespace()))
    none = control_loop_metric_lines(
        SimpleNamespace(context=SimpleNamespace(fault_ingestion=None))
    )

    assert not any(UNRESOLVED in line for line in unbound), unbound
    assert not any(UNRESOLVED in line for line in none), none


def test_registry_secret_drift_renders_as_a_flag_per_service_role() -> None:
    drifted = SimpleNamespace(service_role="api-ha", secret_drift=lambda: True)
    aligned = SimpleNamespace(service_role="control-worker", secret_drift=lambda: False)

    drifted_lines = regional_registry_metric_lines(
        SimpleNamespace(context=SimpleNamespace(regional_registry_runtime=drifted))
    )
    aligned_lines = regional_registry_metric_lines(
        SimpleNamespace(context=SimpleNamespace(regional_registry_runtime=aligned))
    )

    assert _samples(drifted_lines, SECRET_DRIFT) == [
        f'{SECRET_DRIFT}{{service_role="api-ha"}} 1'
    ], drifted_lines
    assert _samples(aligned_lines, SECRET_DRIFT) == [
        f'{SECRET_DRIFT}{{service_role="control-worker"}} 0'
    ], aligned_lines
    assert any(
        line.startswith(f"# TYPE {SECRET_DRIFT} gauge") for line in drifted_lines
    ), "drift is a current condition, so the series is a gauge"


def test_registry_secret_drift_renders_nothing_without_a_runtime() -> None:
    absent = regional_registry_metric_lines(
        SimpleNamespace(context=SimpleNamespace(regional_registry_runtime=None))
    )
    unbound = regional_registry_metric_lines(SimpleNamespace(context=SimpleNamespace()))

    assert absent == [], "a role that runs no registry runtime publishes no series"
    assert unbound == [], "a context without the attribute publishes no series"


def test_the_new_families_are_registered_contributors() -> None:
    names = METRIC_CONTRIBUTORS.names

    assert "spare-reservations" in names, names
    assert "regional-registry" in names, names
