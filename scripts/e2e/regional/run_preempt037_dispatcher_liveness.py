#!/usr/bin/env python3
"""GF-REGIONAL-PREEMPT-037: a stopped dispatcher is visible within five minutes.

Every progress gauge the workflow dispatcher exports is written by the loop it
describes, so a thread that died left each one frozen at its last healthy
value. ARCH-E3 stamps ``gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds``
at the top of every cycle and ``GpuFaultWorkflowDispatcherStalled`` pages when
no replica has ticked for five minutes. This case switches the dispatcher off
on every control-worker replica for one bounded window, watches the stamp age
and the alert expression hold for its ``for`` duration, and restores.

The window is a ``kubectl set env`` of one compiled-in variable on one
Deployment, recorded to a baseline file first and restored exactly (absent
goes back to absent). Nothing is dispatched while it is open, so the runner
refuses to open it while any workflow is PENDING, RUNNING or WAITING, and it
is a maintenance-window action. Plan-only by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import preempt037_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.preempt037_watchdog import DispatcherWatchdog  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    CONTROL_WORKER_APP,
    METRICS_PROBE,
    WORKER_METRICS_PORT,
    metric_max,
)
from scripts.e2e.regional.control_plane_env_window import replica_env  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    component_python,
    RegionalFixtureError,
    RegionalLiveFixture,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
RULES = ROOT / "deploy" / "observability" / "amp-rules.yaml"
RUNBOOK = ROOT / "docs" / "管理员日常运维.md"
ROLLOUT_TIMEOUT_SECONDS = 600

WORKFLOW_STATUSES_PROBE = r"""
import json

from gpu_fault.app import ApplicationContext

store = ApplicationContext.from_environment().store
with store._db.cursor() as cursor:
    cursor.execute(
        "SELECT status,count(*) FROM gpu_fault_workflow_records "
        "WHERE kind='workflow' GROUP BY status"
    )
    counts = {str(status): count for status, count in cursor.fetchall()}
print(json.dumps({"status_counts": counts}))
"""


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "any workflow is PENDING, RUNNING or WAITING",
        "the control-worker Deployment already carries a dispatcher setting this "
        "runner did not record",
        "the stall expression never becomes true, flickers, or the periodic runner "
        "stamp ages too",
        "the env cannot be restored to the recorded baseline",
    ]


def worker_metrics(regional: RegionalLiveFixture) -> list[str]:
    texts: list[str] = []
    for pod in regional.ready_pods("cpu", CONTROL_WORKER_APP):
        output = regional.kubectl(
            "cpu",
            "exec",
            "-i",
            str(pod["name"]),
            "--",
            component_python("cpu"),
            "-",
            str(WORKER_METRICS_PORT),
            input_text=METRICS_PROBE,
            timeout=60,
        )
        texts.append(str(json.loads(output.splitlines()[-1])["metrics"]))
    return texts


def deployment_variable(regional: RegionalLiveFixture) -> dict[str, Any]:
    value = json.loads(
        regional.kubectl("cpu", "get", "deployment", verdicts.DEPLOYMENT, "-o", "json")
    )
    containers = value["spec"]["template"]["spec"]["containers"]
    selected = [
        (index, item)
        for index, item in enumerate(containers)
        if item.get("name") == verdicts.CONTAINER
    ]
    if len(selected) != 1:
        raise RegionalFixtureError(f"{verdicts.DEPLOYMENT} has no {verdicts.CONTAINER}")
    container_index, container = selected[0]
    present = [
        (index, item)
        for index, item in enumerate(container.get("env") or [])
        if item.get("name") == verdicts.VARIABLE
    ]
    if len(present) > 1 or any("valueFrom" in item for _, item in present):
        raise RegionalFixtureError(f"{verdicts.VARIABLE} is set from a reference")
    metadata = value.get("metadata") or {}
    if not metadata.get("uid") or not metadata.get("resourceVersion"):
        raise RegionalFixtureError("dispatcher Deployment identity is unknown")
    return {
        "present": bool(present),
        "value": present[0][1].get("value") if present else None,
        "generation": metadata["generation"],
        "uid": metadata["uid"],
        "resource_version": metadata["resourceVersion"],
        "image": container["image"],
        "container_index": container_index,
        "variable_index": present[0][0] if present else None,
        "env_present": "env" in container,
    }


def replicas(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    return replica_env(
        regional,
        plane="cpu",
        deployment=verdicts.DEPLOYMENT,
        names=(verdicts.VARIABLE,),
    )


def set_variable(
    regional: RegionalLiveFixture, assignment: str, *, baseline: dict[str, Any]
) -> str:
    live = deployment_variable(regional)
    if live["uid"] != baseline["uid"] or live["image"] != baseline["image"]:
        raise RegionalFixtureError(
            "dispatcher Deployment identity changed during the window"
        )
    if assignment not in {
        f"{verdicts.VARIABLE}=false",
        f"{verdicts.VARIABLE}=true",
        f"{verdicts.VARIABLE}-",
    }:
        raise RegionalFixtureError("unsupported dispatcher window assignment")
    current = {key: live[key] for key in ("present", "value")}
    original = {key: baseline[key] for key in ("present", "value")}
    owned_window = {"present": True, "value": verdicts.WINDOW_VALUE}
    if current != original and (
        assignment.endswith("=false") or current != owned_window
    ):
        raise RegionalFixtureError(
            "dispatcher setting is no longer owned by this window"
        )
    path = f"/spec/template/spec/containers/{live['container_index']}/env"
    changes: list[dict[str, Any]] = []
    if assignment.endswith("-"):
        if live["variable_index"] is not None:
            changes.append({"op": "remove", "path": f"{path}/{live['variable_index']}"})
    else:
        value = assignment.rsplit("=", 1)[1]
        if live["variable_index"] is not None:
            changes.append(
                {
                    "op": "replace",
                    "path": f"{path}/{live['variable_index']}/value",
                    "value": value,
                }
            )
        elif live["env_present"]:
            changes.append(
                {
                    "op": "add",
                    "path": path + "/-",
                    "value": {"name": verdicts.VARIABLE, "value": value},
                }
            )
        else:
            changes.append(
                {
                    "op": "add",
                    "path": path,
                    "value": [{"name": verdicts.VARIABLE, "value": value}],
                }
            )
    if changes:
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": baseline["uid"]},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": live["resource_version"],
            },
            *changes,
        ]
        regional.kubectl(
            "cpu",
            "patch",
            "deployment",
            verdicts.DEPLOYMENT,
            "--type=json",
            "--patch-file=/dev/stdin",
            input_text=json.dumps(patch),
        )
    return regional.kubectl(
        "cpu",
        "rollout",
        "status",
        f"deployment/{verdicts.DEPLOYMENT}",
        f"--timeout={ROLLOUT_TIMEOUT_SECONDS}s",
        timeout=ROLLOUT_TIMEOUT_SECONDS + 60,
    ).strip()


def wait_replicas(
    regional: RegionalLiveFixture, expected: str | None
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + ROLLOUT_TIMEOUT_SECONDS
    while True:
        current = replicas(regional)
        if current and all(
            (item.get("values") or {}).get(verdicts.VARIABLE) == expected
            for item in current
        ):
            return current
        if time.monotonic() >= deadline:
            raise RegionalFixtureError(
                f"control-worker replicas did not all read {verdicts.VARIABLE}={expected!r}: "
                f"{current}"
            )
        time.sleep(5)


def execute(
    regional: RegionalLiveFixture, case_dir: Path, deadline: datetime
) -> dict[str, Any]:
    parameters = verdicts.stall_rule_parameters(RULES.read_text(encoding="utf-8"))
    stages: dict[str, list[str]] = {
        "rule": verdicts.rule_errors(parameters, RUNBOOK.read_text(encoding="utf-8"))
    }
    stages["quiescence"] = verdicts.workflow_census_errors(
        regional.cpu_python(WORKFLOW_STATUSES_PROBE)
    )
    if any(stages.values()):
        raise RegionalFixtureError(
            "; ".join(error for errors in stages.values() for error in errors)
        )
    baseline = deployment_variable(regional)
    if baseline["present"] and baseline["value"] != "true":
        raise RegionalFixtureError(
            f"{verdicts.VARIABLE} is already {baseline['value']!r} on the Deployment"
        )
    needed = (
        parameters["threshold_seconds"]
        + parameters["for_seconds"]
        + 4 * verdicts.POLL_SECONDS
    )
    if (deadline - utc_now_dt()).total_seconds() < needed + ROLLOUT_TIMEOUT_SECONDS * 2:
        raise RegionalFixtureError(
            f"the maintenance window cannot hold a {needed}s stall plus two rollouts"
        )
    record: dict[str, Any] = {
        "variable": verdicts.VARIABLE,
        "value": verdicts.WINDOW_VALUE,
        "baseline": baseline,
        "opened_at": utc_now(),
    }
    # What the processes read before the window: the restore must bring every
    # replica back to this, which is not "absent" when an envFrom ConfigMap
    # supplies the variable.
    record["effective_before"] = verdicts.effective_variable_value(replicas(regional))
    if record["effective_before"] != "true":
        raise RegionalFixtureError(
            "every control-worker must have dispatch enabled before the window"
        )
    baseline_path = case_dir / "env-window-baseline.json"
    write_json_atomic(baseline_path, record)
    before = worker_metrics(regional)
    if verdicts.stalled(
        before, now=time.time(), threshold_seconds=parameters["threshold_seconds"]
    ):
        raise RegionalFixtureError("dispatcher was already stalled before the window")
    watchdog = DispatcherWatchdog(
        regional,
        baseline=baseline,
        restore_at=min(
            deadline.timestamp() - ROLLOUT_TIMEOUT_SECONDS,
            time.time() + needed + ROLLOUT_TIMEOUT_SECONDS,
        ),
    )
    record["watchdog"] = {
        "name": watchdog.name,
        "run_id": watchdog.run_id,
        "restore_at": watchdog.restore_at,
    }
    write_json_atomic(baseline_path, record)
    timeline: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    window_attempted = False
    restored = False
    try:
        watchdog.arm()
        fresh_quiescence = verdicts.workflow_census_errors(
            regional.cpu_python(WORKFLOW_STATUSES_PROBE)
        )
        if fresh_quiescence:
            raise RegionalFixtureError("; ".join(fresh_quiescence))
        window_attempted = True
        record["rollout"] = set_variable(
            regional,
            f"{verdicts.VARIABLE}={verdicts.WINDOW_VALUE}",
            baseline=baseline,
        )
        record["replicas"] = wait_replicas(regional, verdicts.WINDOW_VALUE)
        write_json_atomic(baseline_path, record)
        stages["window"] = verdicts.window_errors(record)
        first_true: float | None = None
        end_at = time.monotonic() + needed + parameters["threshold_seconds"]
        while time.monotonic() < end_at:
            in_flight = verdicts.workflow_census_errors(
                regional.cpu_python(WORKFLOW_STATUSES_PROBE)
            )
            if in_flight:
                raise RegionalFixtureError("; ".join(in_flight))
            texts = worker_metrics(regional)
            now = time.time()
            is_stalled = verdicts.stalled(
                texts, now=now, threshold_seconds=parameters["threshold_seconds"]
            )
            sample = {
                "observed_at": utc_now(),
                "observed_epoch": now,
                "stalled": is_stalled,
                "periodic_alive": verdicts.periodic_alive(
                    texts, now=now, threshold_seconds=parameters["threshold_seconds"]
                ),
                "dispatch_stamp_max": metric_max(texts, verdicts.DISPATCH_METRIC),
            }
            timeline.append(sample)
            write_json_atomic(case_dir / "stall-timeline.json", {"entries": timeline})
            if is_stalled and first_true is None:
                first_true = now
            if first_true is not None and now - first_true >= parameters["for_seconds"]:
                break
            time.sleep(verdicts.POLL_SECONDS)
        stages["stall"] = verdicts.stall_timeline_errors(
            timeline, for_seconds=parameters["for_seconds"]
        )
    finally:
        try:
            if window_attempted:
                restore = (
                    f"{verdicts.VARIABLE}={baseline['value']}"
                    if baseline["present"]
                    else f"{verdicts.VARIABLE}-"
                )
                record["restore_started_at"] = utc_now()
                record["rollout_after_close"] = set_variable(
                    regional, restore, baseline=baseline
                )
                fields = ("present", "value", "uid", "image")
                current = deployment_variable(regional)
                record["restored_state"] = {key: current[key] for key in fields}
                record["baseline"] = {key: baseline[key] for key in fields}
                record["replicas_after_close"] = wait_replicas(
                    regional, record["effective_before"]
                )
                stages["restore"] = verdicts.restore_errors(record)
                restored = not stages["restore"]
                if restored:
                    record["closed_at"] = utc_now()
        finally:
            record["watchdog_retained"] = window_attempted and not restored
            try:
                write_json_atomic(baseline_path, record)
            finally:
                if not window_attempted or restored:
                    watchdog.cleanup()
    end_at = time.monotonic() + parameters["threshold_seconds"]
    while time.monotonic() < end_at:
        texts = worker_metrics(regional)
        now = time.time()
        recovery.append(
            {
                "observed_at": utc_now(),
                "stalled": verdicts.stalled(
                    texts, now=now, threshold_seconds=parameters["threshold_seconds"]
                ),
            }
        )
        if not recovery[-1]["stalled"]:
            break
        time.sleep(verdicts.POLL_SECONDS)
    stages["recovery"] = verdicts.recovery_errors(
        recovery, threshold_seconds=parameters["threshold_seconds"]
    )
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "rule": parameters,
        "env_window": record,
        "metrics_before_window_families": sorted(
            {
                line.split("{", 1)[0].split(" ", 1)[0]
                for text in before
                for line in text.splitlines()
                if line.startswith("gpu_fault_workflow_dispatch_")
            }
        ),
        "stall_timeline": timeline,
        "recovery_samples": recovery,
        "limitations": [
            "The alert is evaluated by the runner on the replicas' /metrics with the "
            "rule's threshold and for; AMP's own evaluation is not read.",
            "Switching the dispatcher off stops all workflow dispatch for the window; "
            "the runner refuses while any workflow is in flight.",
        ],
    }


def utc_now_dt() -> datetime:
    return datetime.fromisoformat(utc_now())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded PREEMPT-037 acceptance: a stopped workflow dispatcher "
            "ages its last-cycle stamp and the stall alert expression holds."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = settings_from_arguments(arguments)
    predecessor_id, path = predecessor_path(
        arguments.run_dir, CASE_ID, arguments.predecessor_evidence
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    environment = {**settings.environment(), "GPU_FAULT_LIVENESS_CASE": CASE_ID}
    parameters = verdicts.stall_rule_parameters(RULES.read_text(encoding="utf-8"))
    if not arguments.execute:
        plan = build_plan(
            arguments=arguments,
            preflight_passed=predecessor.get("valid") is True,
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details={
                "risk": "live-service-action",
                "predecessor": predecessor,
                "deployment": verdicts.DEPLOYMENT,
                "variable": verdicts.VARIABLE,
                "rule": parameters,
                "mutation": (
                    f"set {verdicts.VARIABLE}={verdicts.WINDOW_VALUE} on the "
                    f"{verdicts.CONTAINER} container of {verdicts.DEPLOYMENT} (rolls every "
                    "replica), hold until the stall expression has been true for the "
                    "rule's for duration, then restore the recorded baseline exactly"
                ),
                "stop_conditions": stop_conditions(),
                "rollback": {
                    "baseline_recorded_before_mutation": True,
                    "runner_finally_restores_the_variable": True,
                    "absent_goes_back_to_absent": True,
                    "independent_cpu_watchdog": True,
                    "watchdog_permissions": "get/patch only the control-worker Deployment",
                },
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if arguments.confirm != CONFIRMATION:
        raise RegionalFixtureError(f"confirmation must be exactly {CONFIRMATION}")
    deadline = authorize_execution(
        arguments, case_id=CASE_ID, confirmation=CONFIRMATION, environment=environment
    )
    if not predecessor.get("valid", False):
        raise RegionalFixtureError("formal predecessor evidence is not PASS")
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    regional = RegionalLiveFixture(settings)
    identity = regional.evidence_identity()
    try:
        outcome = execute(regional, case_dir, deadline)
        if regional.evidence_identity() != identity:
            raise RegionalFixtureError(
                "release identity changed during dispatcher window"
            )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        outcome = {"verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **identity,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, CASE_ID), result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
