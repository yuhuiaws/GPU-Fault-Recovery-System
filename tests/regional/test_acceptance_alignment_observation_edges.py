from __future__ import annotations

import copy
import runpy
import sys
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from scripts.e2e.regional import ha009_observation as ha009
from scripts.e2e.regional import notify007_verdicts as notify
from scripts.e2e.regional.capacity_acceptance_retry import run_claim_retry_proof
from scripts.e2e.regional.capacity_wire import CapacityWireClient
from scripts.e2e.regional.ha_telemetry_evidence import (
    admission_errors,
    telemetry_replay_errors,
)
from scripts.e2e.regional.probes import notification_drill, notify003_requeue_drill
from tests.regional._cov95_notify_harness import Notifier
from tests.regional.test_acceptance_alignment_ha_telemetry import telemetry_proof
from tests.regional.test_acceptance_alignment_metrics import notification_metrics


@pytest.mark.parametrize("payload", ["", "not-json", "[]"])
def test_pool_probe_requires_an_object_receipt(payload) -> None:
    with pytest.raises(ValueError, match="Pod health probe"):
        ha009.pod_observation(lambda *a, **kw: payload, "p", 8080, python="isolated")


@pytest.mark.parametrize(
    "defect", ["no-pods", "missing-digest", "wrong-digest", "no-samples", "log-error"]
)
def test_rotation_cannot_pass_incomplete_replica_observations(defect) -> None:
    pods = {"p"}
    propagation = {"digest": "new", "pods": {"p": "new"}}
    observation = {
        "samples": {
            "p": [
                {
                    "healthz_status": 200,
                    "metrics_status": 200,
                    "fresh_connection": True,
                    "metrics": {"gpu_fault_postgres_pool_connections_errors_total": 0},
                }
            ]
        },
        "auth_failures_in_logs": {"p": 0},
    }
    if defect == "no-pods":
        pods = set()
    elif defect == "missing-digest":
        propagation["pods"] = {}
    elif defect == "wrong-digest":
        propagation["pods"]["p"] = "old"
    elif defect == "no-samples":
        observation["samples"]["p"] = []
    else:
        observation["auth_failures_in_logs"]["p"] = 1
    assert ha009.observation_errors(pods, propagation, observation), (
        "an incomplete replica observation must fail"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing-before",
        "missing-after",
        "uid",
        "generation",
        "bad-count",
        "changed-count",
        "zero-count",
        "old-pods",
        "new-pods",
        "pod-uid",
        "restart",
        "not-ready",
        "available",
    ],
)
def test_rotation_rejects_every_unsteady_deployment_fact(defect) -> None:
    before = {
        "role": {
            "uid": "deployment-uid",
            "generation": 1,
            "replicas": 1,
            "ready": 1,
            "updated": 1,
            "available": 1,
            "pods": [["pod", {"uid": "pod-uid", "restarts": 0, "ready": True}]],
        }
    }
    after = copy.deepcopy(before)
    if defect == "missing-before":
        before.clear()
    elif defect == "missing-after":
        after.clear()
    elif defect in {"uid", "generation", "changed-count", "available"}:
        key = {"changed-count": "replicas"}.get(defect, defect)
        after["role"][key] = "other" if key == "uid" else 2
    elif defect == "bad-count":
        before["role"]["replicas"] = True
    elif defect == "zero-count":
        before["role"]["replicas"] = after["role"]["replicas"] = 0
    elif defect == "old-pods":
        before["role"]["pods"] = []
    elif defect == "new-pods":
        after["role"]["pods"] = []
    else:
        pod = after["role"]["pods"][0][1]
        pod.update(
            {
                "pod-uid": {"uid": "replacement"},
                "restart": {"restarts": 1},
                "not-ready": {"ready": False},
            }[defect]
        )
    assert ha009.steady_deployments(before, after, ["role"]), defect


def test_notification_metrics_reject_absence_and_negative_counters() -> None:
    assert notify.exported_family_errors([]) == [
        "no control-plane metrics were observed"
    ]
    good = notification_metrics(worker=True)
    bad = good.replace(
        f"{notify.DEAD_LETTERED_METRIC} 0", f"{notify.DEAD_LETTERED_METRIC} -1"
    )
    assert any("invalid" in error for error in notify.exported_family_errors([bad])), (
        "negative dispatch counters must fail"
    )
    bad = good.replace(
        f"{notify.TERMINAL_FAILURE_METRIC} 0", f"{notify.TERMINAL_FAILURE_METRIC} -1"
    )
    assert any(
        "invalid" in error for error in notify.production_untouched_errors([bad], [bad])
    ), "negative event time cannot prove production was untouched"


def test_telemetry_requires_valid_queue_ids_and_final_observation() -> None:
    probe, receipts = telemetry_proof(spooled=False)
    assert admission_errors(probe, ["processor-unit"]) == []
    probe["admissions"][0]["request_id"] = ""
    assert admission_errors(probe, []), "queue admission requires a usable receipt ID"
    assert telemetry_replay_errors({"attempted_events": []}, receipts["telemetry"]), (
        "no final event means no replay proof"
    )


@pytest.mark.parametrize("nonempty", [False, True])
def test_executor_capacity_retry_cannot_succeed_from_a_permanent_outage_or_work(
    monkeypatch, tmp_path, nonempty
) -> None:
    calls, releases = [], []

    def claim(*args, **kwargs):
        calls.append(True)
        if nonempty:
            return [object()]
        raise ClusterExecutorError("isolated unavailable", status_code=503)

    monkeypatch.setattr(CapacityWireClient, "claim", claim)
    result = run_claim_retry_proof(
        "http://127.0.0.1:18333",
        "fixture",
        tmp_path,
        lambda: releases.append(True),
        poll_seconds=0.001,
    )
    assert result["passed"] is False
    assert len(calls) == (1 if nonempty else 5)
    assert len(releases) == int(not nonempty)


def test_notifier_itself_checks_the_window_and_probe_rejects_naive_deadlines(
    monkeypatch,
) -> None:
    notifier = Notifier()
    wrapped = notification_drill.DrillNotifier(
        notifier, "unit", datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    with pytest.raises(RuntimeError, match="window ended"):
        wrapped.send(notification_drill.build_notification("gpu-reset", "unit", "c"))
    assert notifier.sent == []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--kind",
            "gpu-reset",
            "--drill-id",
            "unit",
            "--cluster-id",
            "c",
            "--maintenance-window-end",
            "2020-01-01T00:00:00",
        ],
    )
    with pytest.raises(ValueError, match="timezone"):
        notification_drill.main()


@pytest.mark.parametrize("during_sleep", [False, True])
def test_deadline_during_completion_or_duplicate_wait_prevents_further_sends(
    monkeypatch, during_sleep
) -> None:
    now = [datetime.now(timezone.utc)]
    deadline = now[0] + timedelta(seconds=70)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    class Sender(Notifier):
        def send(self, item):
            response = super().send(item)
            if not during_sleep:
                now[0] = deadline
            return response

    monkeypatch.setattr(notification_drill, "datetime", Clock)
    monkeypatch.setattr(
        notification_drill.time, "sleep", lambda _: now.__setitem__(0, deadline)
    )
    notifier = Sender()
    with pytest.raises(RuntimeError, match="window ended"):
        notification_drill.replay_completion(
            "gpu-reset",
            "unit",
            "c",
            notifier,
            deadline=deadline,
            duplicate_delay_seconds=65 if during_sleep else 0,
        )
    assert len(notifier.sent) == 1


def test_private_notification_entrypoints_use_only_the_injected_notifier(
    monkeypatch,
) -> None:
    import gpu_fault.notifications

    notifier = Notifier()
    monkeypatch.setattr(
        gpu_fault.notifications,
        "notification_notifier_from_environment",
        lambda: notifier,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--kind", "gpu-reset", "--drill-id", "unit", "--cluster-id", "c"],
    )
    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(notification_drill.__file__, run_name="__main__")
    assert exit_code.value.code == 0 and len(notifier.sent) == 1
    runpy.run_path(notify003_requeue_drill.__file__, run_name="__main__")
