from __future__ import annotations

import json

import pytest

from scripts.e2e.regional import notify007_verdicts as notify
from scripts.e2e.regional.capacity_acceptance_base import CapError
from scripts.e2e.regional.capacity_acceptance_traffic import sparse_cluster_queue_depth
from scripts.e2e.regional.ha009_observation import parse_pool_metrics
from scripts.e2e.regional.notification_evidence import validate_duplicate_evidence


def test_labelled_pool_counters_are_aggregated_across_every_process() -> None:
    name = "gpu_fault_postgres_pool_connections_errors_total"
    text = f'{name}{{process="0"}} 2\n{name}{{process="3"}} 4\n'
    assert parse_pool_metrics(text) == {name: 6}


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1"])
def test_invalid_pool_samples_fail_instead_of_disappearing(value: str) -> None:
    with pytest.raises(ValueError, match="pool metric"):
        parse_pool_metrics(f'gpu_fault_postgres_pool_size{{process="0"}} {value}')


def test_duplicate_pool_series_are_not_double_counted() -> None:
    sample = 'gpu_fault_postgres_pool_size{process="0"} 1\n'
    with pytest.raises(ValueError, match="duplicate"):
        parse_pool_metrics(sample * 2)


def notification_metrics(*, worker: bool) -> str:
    lines = [f"{name} 0" for name in notify.REQUIRED_FAMILIES]
    if worker:
        lines.extend(
            f'{notify.DELIVERY_METRIC}{{status="{status}"}} 0'
            for status in ("PENDING", "LEASED", "RETRY", "SENT", "DEAD")
        )
        lines.extend(
            (
                f'{notify.RESULT_METRIC}{{status="FAILED"}} 0',
                f"{notify.OLDEST_PENDING_METRIC} 0",
            )
        )
    return "\n".join(lines)


def test_each_cpu_role_proves_its_own_metric_contract() -> None:
    roles = {"w1": "worker", "w2": "worker", "api": "ingress", "spool": "spool-worker"}
    metrics = {
        pod: notification_metrics(worker=role == "worker")
        for pod, role in roles.items()
    }
    assert notify.exported_family_errors(metrics, roles=roles) == []
    assert notify.production_untouched_errors(metrics, metrics, roles=roles) == []
    metrics["w2"] = notification_metrics(worker=False)
    assert any(
        "w2" in error and notify.DELIVERY_METRIC in error
        for error in notify.exported_family_errors(metrics, roles=roles)
    ), "a missing worker census must be attributed to that replica"


@pytest.mark.parametrize("role", ["worker", "ingress", "spool-worker"])
def test_terminal_failure_family_is_required_and_monitored_on_all_roles(
    role: str,
) -> None:
    good = notification_metrics(worker=role == "worker")
    missing = good.replace(f"{notify.TERMINAL_FAILURE_METRIC} 0", "")
    roles = {"pod": role}
    assert any(
        notify.TERMINAL_FAILURE_METRIC in error
        for error in notify.exported_family_errors({"pod": missing}, roles=roles)
    ), "every role must expose terminal notification failure time"
    changed = good.replace(
        f"{notify.TERMINAL_FAILURE_METRIC} 0", f"{notify.TERMINAL_FAILURE_METRIC} 5"
    )
    assert any(
        notify.TERMINAL_FAILURE_METRIC in error
        for error in notify.production_untouched_errors(
            {"pod": good}, {"pod": changed}, roles=roles
        )
    ), "a new terminal failure must invalidate the untouched-production proof"


def test_unknown_or_missing_role_cannot_suppress_fleet_checks() -> None:
    metrics = {"worker": notification_metrics(worker=False)}
    for roles in ({}, {"worker": "unknown"}, {"different": "worker"}):
        assert notify.exported_family_errors(metrics, roles=roles), (
            "unknown roles cannot omit required metric families"
        )
        assert notify.production_untouched_errors(metrics, metrics, roles=roles), (
            "unknown roles cannot prove no production changes"
        )


@pytest.mark.parametrize("total,sibling", [(0.0, 0.0), (5.0, 5.0)])
def test_empty_cluster_is_zero_only_with_a_complete_queue_census(
    total, sibling
) -> None:
    values = [("gpu_fault_processor_queue_depth", {}, total)]
    if sibling:
        values.append(
            (
                "gpu_fault_processor_cluster_queue_depth",
                {"cluster_id": "other"},
                sibling,
            )
        )
    assert sparse_cluster_queue_depth(values, "empty") == 0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [("gpu_fault_processor_queue_depth", {}, 1)],
        [
            ("gpu_fault_processor_queue_depth", {}, 1),
            ("gpu_fault_processor_cluster_queue_depth", {"cluster_id": "other"}, -1),
        ],
        [
            ("gpu_fault_processor_queue_depth", {}, 0),
            ("gpu_fault_processor_cluster_queue_depth", {}, 0),
        ],
    ],
)
def test_missing_or_inconsistent_queue_census_is_not_zero(values) -> None:
    with pytest.raises(CapError):
        sparse_cluster_queue_depth(values, "empty")


def test_duplicate_only_ses_window_excludes_the_initial_send(tmp_path) -> None:
    drill = {
        "initial_completed_at": "2026-09-15T10:00:01+00:00",
        "duplicate_window_start": "2026-09-15T10:00:02+00:00",
        "duplicate_window_end": "2026-09-15T10:00:03+00:00",
    }
    evidence = {
        "method": "provider-receipt",
        "reference": "isolated-fixture",
        "window_start": drill["initial_completed_at"],
        "window_end": drill["duplicate_window_end"],
        "send_count_delta": 0,
        "duplicate_inbox_count": 0,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(evidence))
    assert validate_duplicate_evidence(path, drill)["valid"] is True
    evidence["window_start"] = "2026-09-15T10:00:00+00:00"
    path.write_text(json.dumps(evidence))
    assert validate_duplicate_evidence(path, drill)["valid"] is False
    assert validate_duplicate_evidence(path, {})["valid"] is False
