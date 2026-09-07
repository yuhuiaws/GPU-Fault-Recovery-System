#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-022: the executor reclaims a stale warm-spare reservation.

One declared, cordoned, idle warm spare (the node DESTR-003/008 ran on). The
runner writes a reservation for a synthetic incident that exists nowhere --
``gpu-fault.io/spare-reservation``, a ``spare-reserved-at`` back-dated two days
past the one-day TTL, ``spare-pool-state=ALLOCATED`` -- and leaves the spare
cordoned. The regional cluster executor is storeless, so its
``SpareReservationSweep`` can only judge that reservation by its timestamp
(ARCH-A4b); within one sweep interval it must release it, log the reclaim
naming node and incident, and move ``spare_reservations_reclaimed_total`` in
the claim-state breadcrumb (ARCH-A4c), the executor's only counter surface.

Nothing physical happens and nothing can: the spare is never uncordoned, the
synthetic incident has no workflow, no other node is touched, and no provider
API is involved. Cleanup restores exactly the tracked annotations and cordon
the declaration recorded, whether or not the sweep already did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import destr022_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    NodeMutationFixture,
    NodePatch,
    WarmSpareLiveFixture,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
EXECUTOR_APP = "gpu-fault-cluster-executor"
EXECUTOR_PROBE_SCRIPT = (
    Path(__file__).with_name("probes") / "destr022_executor_probe.py"
)
POLL_SECONDS = 10

INCIDENT_LOOKUP = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

incident_id = sys.argv[1]
store = ApplicationContext.from_environment().store
try:
    incident = store.get_incident(incident_id)
except NotFoundError:
    print(json.dumps({"incident_id": incident_id, "found": False, "workflows": []}))
else:
    workflows = (
        [incident.workflow_request_id] if incident.workflow_request_id else []
    )
    print(json.dumps({
        "incident_id": incident_id,
        "found": True,
        "state": str(incident.state),
        "workflows": workflows,
    }, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    hyperpod_cluster: str
    spare_node: str
    predecessor_path: Path
    reclaim_timeout_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_SPARE_NODE": self.spare_node,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    timeout = int(arguments.reclaim_timeout_seconds)
    if not (
        verdicts.MIN_RECLAIM_TIMEOUT_SECONDS
        <= timeout
        <= verdicts.MAX_RECLAIM_TIMEOUT_SECONDS
    ):
        raise RegionalFixtureError(
            "reclaim timeout is outside "
            f"{verdicts.MIN_RECLAIM_TIMEOUT_SECONDS}.."
            f"{verdicts.MAX_RECLAIM_TIMEOUT_SECONDS} seconds"
        )
    return Settings(
        regional=settings_from_arguments(arguments),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        spare_node=required(
            arguments.spare_node or os.getenv("GPU_FAULT_SPARE_NODE", ""),
            "spare node",
        ),
        predecessor_path=predecessor,
        reclaim_timeout_seconds=timeout,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_spare_reservations.py",
        "tests/regional/test_destr022_spare_reservation_reclaim.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def executor_probes(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    """The claim-state breadcrumb and sweep env of every Ready executor Pod.

    Each replica runs its own sweep and its own counters, so the probe is run
    per Pod rather than through ``executor_python`` (which picks one).
    """

    script = EXECUTOR_PROBE_SCRIPT.read_text(encoding="utf-8")
    result = []
    for pod in regional.ready_pods("gpu", EXECUTOR_APP):
        output = regional.kubectl(
            "gpu",
            "exec",
            "-i",
            str(pod["name"]),
            "--",
            "python3",
            "-",
            input_text=script,
            timeout=120,
        )
        value = json.loads(output.splitlines()[-1])
        if not isinstance(value, dict):
            raise RegionalFixtureError("executor probe did not return a JSON object")
        result.append({"pod": str(pod["name"]), "uid": str(pod["uid"]), **value})
    return sorted(result, key=lambda item: str(item["pod"]))


def incident_lookup(regional: RegionalLiveFixture, incident_id: str) -> dict[str, Any]:
    return regional.cpu_python(INCIDENT_LOOKUP, incident_id)


def synthetic_incident(run_dir: Path, attempt: int) -> str:
    return verdicts.synthetic_incident_id(
        f"{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    )


def read_only_preflight(
    settings: Settings, case_dir: Path, incident_id: str
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    spare = warm.node_snapshot(settings.spare_node)
    declared = warm.spare_nodes()
    workloads = regional.business_workloads(settings.spare_node)
    executor_env = warm.executor_environment()
    probes = executor_probes(regional)
    state = regional.store_snapshot(node=settings.spare_node)
    lookup = incident_lookup(regional, incident_id)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    tests = focused_tests(case_dir)
    runtime_identity = regional.runtime_identity()
    errors = verdicts.preflight_errors(
        spare=spare,
        spare_node=settings.spare_node,
        declared_spares=declared,
        workloads=workloads,
        executor_env=executor_env,
        executor_probes=probes,
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        incident_lookup=lookup,
        predecessor_valid=bool(predecessor["valid"]),
        tests_passed=tests["passed"],
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    result = {
        "release_id": state.get("release_id"),
        "synthetic_incident_id": incident_id,
        "spare": spare,
        "declared_spares": declared,
        "business_workloads": workloads,
        "executor_environment": executor_env,
        "executor_probes": probes,
        "incident_lookup": lookup,
        "queue": state.get("queue"),
        "remote_commands": state.get("remote_commands"),
        "gpu_nodes": regional.gpu_nodes(),
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "runtime_identity": runtime_identity,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    spare = preflight["spare"]
    return {
        "release_id": preflight["release_id"],
        "spare_uid": spare.get("uid"),
        "spare_unschedulable": bool(spare.get("unschedulable")),
        "declared_spares": list(preflight["declared_spares"]),
        "executor_pod_uids": sorted(
            str(item.get("uid")) for item in preflight["executor_probes"]
        ),
        "synthetic_incident_id": preflight["synthetic_incident_id"],
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-node-mutation",
        "predecessor": preflight["predecessor"],
        "spare_node": settings.spare_node,
        "reclaim_timeout_seconds": settings.reclaim_timeout_seconds,
        "mutation": (
            "write a spare reservation for a synthetic incident that exists "
            "nowhere, back-dated two days past the reservation TTL, onto the one "
            "declared and cordoned warm spare; keep the spare cordoned; wait for "
            "the deployed cluster executor's reservation sweep to release it. "
            "No workflow, no incident, no uncordon, no other node, no provider "
            "call."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "preflight or focused regression failure",
            "the declared spare set is not exactly the requested node",
            "the spare is not Ready, not cordoned, already reserved, "
            "quarantined or carries a workload",
            "any executor Pod runs without GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"
            "=true (the sweep is not constructed) or its breadcrumb lacks "
            f"counters.{verdicts.RECLAIM_COUNTER} (executor predates ARCH-A4b)",
            "any remote command or processor work is open",
            "the synthetic incident id already exists in the store",
            "the injection uncordoned the spare or did not land as written",
            "the spare becomes schedulable at any sample",
            "provider mutation appears",
            "cleanup cannot restore the declared baseline",
        ],
        "rollback": {
            "spare_is_never_uncordoned": True,
            "synthetic_incident_has_no_workflow_and_never_will": True,
            "node_mutation_restores_recorded_baseline_in_finally": True,
            "restore_is_idempotent_after_the_executor_reclaim": True,
            "no_other_node_is_patched": True,
            "no_provider_action_is_authorized": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    warm: WarmSpareLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    incident_id: str
    mutation: NodeMutationFixture
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    injected_at: datetime | None = None
    reclaimed_at: datetime | None = None
    reserved_at: str = ""
    baseline_spare: dict[str, Any] = field(default_factory=dict)
    executor_before: list[dict[str, Any]] = field(default_factory=list)


def _prepare_live_run(settings: Settings, run_dir: Path, attempt: int) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    incident_id = synthetic_incident(run_dir, attempt)
    preflight = read_only_preflight(settings, case_dir, incident_id)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    mutation = NodeMutationFixture(
        warm,
        settings.spare_node,
        annotation_keys=verdicts.TRACKED_ANNOTATIONS,
        track_unschedulable=True,
    )
    run = _LiveRun(
        settings=settings,
        regional=regional,
        warm=warm,
        case_dir=case_dir,
        preflight=preflight,
        incident_id=incident_id,
        mutation=mutation,
    )
    run.baseline_spare = dict(mutation.baseline)
    write_json_atomic(case_dir / "spare-baseline.json", run.baseline_spare)
    run.executor_before = list(preflight["executor_probes"])
    write_json_atomic(case_dir / "executor-before.json", {"pods": run.executor_before})
    return run


def _inject(run: _LiveRun) -> list[str]:
    """Write the stale reservation; the spare stays cordoned."""

    refusals = verdicts.spare_refusals(run.warm.node_snapshot(run.settings.spare_node))
    if refusals:
        raise RegionalFixtureError(
            "the spare changed since preflight: " + "; ".join(refusals)
        )
    run.injected_at = datetime.now(timezone.utc)
    run.reserved_at = verdicts.stale_reserved_at(run.injected_at)
    run.mutation.apply(
        NodePatch(
            labels={},
            annotations={
                verdicts.SPARE_RESERVATION_ANNOTATION: run.incident_id,
                verdicts.SPARE_RESERVED_AT_ANNOTATION: run.reserved_at,
                verdicts.SPARE_POOL_STATE_ANNOTATION: verdicts.POOL_STATE_ALLOCATED,
            },
            # Deliberately not False: a reservation the coordinator writes
            # uncordons the spare for the failover; this one must never let
            # anything schedule onto the node.
            unschedulable=None,
        )
    )
    snapshot = run.warm.node_snapshot(run.settings.spare_node)
    write_json_atomic(
        run.case_dir / "injection.json",
        {
            "injected_at": run.injected_at.isoformat(),
            "incident_id": run.incident_id,
            "reserved_at": run.reserved_at,
            "snapshot": snapshot,
        },
    )
    return verdicts.injection_errors(
        snapshot, incident_id=run.incident_id, reserved_at=run.reserved_at
    )


def _wait_reclaim(run: _LiveRun) -> list[str]:
    deadline = time.monotonic() + run.settings.reclaim_timeout_seconds
    timeline: list[dict[str, Any]] = []
    while True:
        snapshot = run.warm.node_snapshot(run.settings.spare_node)
        observed_at = datetime.now(timezone.utc)
        timeline.append({"observed_at": observed_at.isoformat(), "snapshot": snapshot})
        write_json_atomic(run.case_dir / "reclaim-timeline.json", {"entries": timeline})
        if not snapshot.get("unschedulable"):
            raise RegionalFixtureError(
                "the spare became schedulable during the sweep window; aborting"
            )
        if verdicts.reclaimed(snapshot):
            run.reclaimed_at = observed_at
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(POLL_SECONDS)
    return verdicts.timeline_errors(timeline, incident_id=run.incident_id)


def _executor_evidence(run: _LiveRun) -> list[str]:
    if run.injected_at is None:
        raise RegionalFixtureError("executor evidence follows the injection")
    # The breadcrumb is rewritten on the next successful claim round-trip
    # (every poll interval, ~2 s, even on an empty queue); give it one.
    time.sleep(5)
    after = executor_probes(run.regional)
    write_json_atomic(run.case_dir / "executor-after.json", {"pods": after})
    lines: list[str] = []
    logs: list[dict[str, Any]] = []
    for pod in run.regional.ready_pods("gpu", EXECUTOR_APP):
        output = run.regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            "--since-time",
            run.injected_at.isoformat(),
            check=False,
            timeout=120,
        )
        matching = [
            line[:500]
            for line in output.splitlines()
            if verdicts.RECLAIM_LOG_PREFIX in line or "sweep failed" in line
        ]
        lines.extend(matching)
        logs.append(
            {
                "pod": pod["name"],
                "sha256": hashlib.sha256(output.encode()).hexdigest(),
                "matching_lines": matching,
            }
        )
    write_json_atomic(run.case_dir / "executor-logs.json", {"entries": logs})
    errors = verdicts.log_errors(
        lines, node=run.settings.spare_node, incident_id=run.incident_id
    )
    errors.extend(
        verdicts.counter_errors(
            run.executor_before,
            after,
            reclaimed_at=run.reclaimed_at or datetime.now(timezone.utc),
        )
    )
    return errors


def _store_and_provider_errors(run: _LiveRun) -> list[str]:
    after = incident_lookup(run.regional, run.incident_id)
    write_json_atomic(run.case_dir / "incident-lookup-after.json", after)
    errors = verdicts.store_errors(run.preflight["incident_lookup"], after)
    events = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": events})
    if events:
        errors.append("provider mutation appeared during DESTR-022")
    return errors


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "spare_node": settings.spare_node,
        "synthetic_incident_id": run.incident_id,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        errors = _inject(run)
        if errors:
            raise RegionalFixtureError("injection did not land: " + "; ".join(errors))
        errors.extend(_wait_reclaim(run))
        errors.extend(_executor_evidence(run))
        errors.extend(_store_and_provider_errors(run))
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "injected_at": run.injected_at.isoformat() if run.injected_at else None,
                "reclaimed_at": (
                    run.reclaimed_at.isoformat() if run.reclaimed_at else None
                ),
                "reserved_at": run.reserved_at,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = _cleanup(run)
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    guard("restore_spare", lambda: _restore_spare(run))
    guard("other_nodes", lambda: _other_nodes(run))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage=f"after {CASE_ID} cleanup",
        ),
    )
    return result


def _restore_spare(run: _LiveRun) -> dict[str, Any]:
    # Idempotent: after a reclaim the annotations are already absent and the
    # patch re-asserts the recorded baseline (absent stays absent, the cordon
    # stays as declared); after a timeout it removes what this case wrote.
    final = run.mutation.restore()
    write_json_atomic(run.case_dir / "spare-final.json", final)
    errors = verdicts.final_errors(run.baseline_spare, final)
    if run.regional.business_workloads(run.settings.spare_node):
        errors.append("spare node acquired a non-system workload")
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return final


def _other_nodes(run: _LiveRun) -> dict[str, Any]:
    nodes = run.regional.gpu_nodes()
    write_json_atomic(run.case_dir / "gpu-nodes-final.json", {"nodes": nodes})
    errors = verdicts.other_nodes_errors(
        run.preflight["gpu_nodes"], nodes, spare_node=run.settings.spare_node
    )
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return {"count": len(nodes)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-022 acceptance: a stale synthetic warm-spare "
            "reservation on the declared spare, reclaimed by the deployed "
            "cluster executor's reservation sweep."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--spare-node", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--reclaim-timeout-seconds",
        type=int,
        default=verdicts.RECLAIM_TIMEOUT_SECONDS,
        help=(
            "how long to wait for the executor sweep (one 300 s interval plus "
            f"margin; {verdicts.MIN_RECLAIM_TIMEOUT_SECONDS}.."
            f"{verdicts.MAX_RECLAIM_TIMEOUT_SECONDS})"
        ),
    )
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(
            settings,
            case_dir,
            synthetic_incident(arguments.run_dir, arguments.attempt),
        )
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
