#!/usr/bin/env python3
"""The remediation storm breaker step of the perf suite (perf plan §8.5).

Every other step keeps its synthetic clusters below the breaker; this one
makes one cluster cross it on purpose, at the product's constants, and judges
the breaker's whole cycle on the product's own state
(``regional_storm_breaker``):

1. ``perf-cap-000`` registers ``--fleet-nodes-per-cluster`` GPU nodes (the
   step's workflow nodes and a healthy fleet), so its threshold is 52 by
   default; ``admitted + held`` job-less workflows of
   ``--nodes-per-workflow`` nodes each are seeded at once (one incident each:
   one workload domain each), exactly enough that the ``admitted + 1``-th
   first admission trips the breaker;
2. ``trip``: the breaker trips at that admission and counted the cluster's
   registered fleet (its ``trip_cluster_nodes`` and ``trip_threshold``);
3. ``hold``: the rest are held -- PENDING on ``storm:perf-cap-000``, listed by
   the product, no remote command on their nodes -- while the admitted ones run
   to SUCCEEDED; the operator notice is recorded on the breaker, the gauges
   read the trip and AMP reports ``GpuFaultRemediationStormBreakerOpen``
   firing for the cluster;
4. ``release``: the operator exit's own Pod program releases it;
5. ``drain``: every held workflow runs to SUCCEEDED without counting again;
   ``release_to_held_completion_seconds`` is the step's headline number;
6. ``resolve``: the alert leaves AMP and the gauge reads 0.

Every phase is bounded and a phase that times out ends the step (the release
still runs when the trip happened, so a missing alert does not strand the held
work). Whatever ends it, the step's cleanup reads the breaker and, when it is
still open, hands the held work over (``abort``) and releases it, then reads
it CLOSED; the fleet's own exit does the same, and the teardown purges the
cluster's data, the fleet and the breaker record and proves none is left.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_storm_breaker as storm
    from .regional_action_capacity_suite import (
        JOB_NAME,
        LOAD_SERVICE_ACCOUNT,
        SCRIPT_CONFIGMAP,
        aggregate_executor_documents,
        collect_executor_logs,
        executor_identity,
        executor_job,
        release_agent_identity,
        run_agent_heartbeat_refresher,
        seed,
    )
    from .regional_capacity_recovery import (
        abort_on_signals,
        cleanup_signal_guard,
        resume_identity,
        write_aborted_status,
    )
    from .regional_capacity_registry import (
        NAMESPACE,
        dataplane_context_problem,
        dataplane,
        log,
        register,
        validate_registry_target,
    )
    from .regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )
    from .regional_capacity_results import artifact_dir, write_status
    from .regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        release_identity,
        step_status,
        teardown,
        upsert_configmap,
        validate_registered_resources,
    )
    from .regional_executor_support import executor_job_failure, executor_scripts
    from .regional_synthetic_fleet import (
        add_fleet_argument,
        fleet_plan,
        synthetic_fleet,
        verify_fleet_cleanup,
    )
else:
    import regional_storm_breaker as storm
    from regional_action_capacity_suite import (
        JOB_NAME,
        LOAD_SERVICE_ACCOUNT,
        SCRIPT_CONFIGMAP,
        aggregate_executor_documents,
        collect_executor_logs,
        executor_identity,
        executor_job,
        release_agent_identity,
        run_agent_heartbeat_refresher,
        seed,
    )
    from regional_capacity_recovery import (
        abort_on_signals,
        cleanup_signal_guard,
        resume_identity,
        write_aborted_status,
    )
    from regional_capacity_registry import (
        NAMESPACE,
        dataplane_context_problem,
        dataplane,
        log,
        register,
        validate_registry_target,
    )
    from regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )
    from regional_capacity_results import artifact_dir, write_status
    from regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        release_identity,
        step_status,
        teardown,
        upsert_configmap,
        validate_registered_resources,
    )
    from regional_executor_support import executor_job_failure, executor_scripts
    from regional_synthetic_fleet import (
        add_fleet_argument,
        fleet_plan,
        synthetic_fleet,
        verify_fleet_cleanup,
    )

CLUSTER = storm.STORM_CLUSTER
#: The breaker counts a cluster's nodes at most once a minute per process
#: (``storm_admission.NODE_COUNT_TTL_SECONDS``); the step seeds its workflows
#: no sooner than this after its fleet was complete, so no count a process
#: cached for the cluster's id before the fleet existed can judge them (unless
#: a re-read fails: that process then judges with the older count, never above
#: the fleet's, for up to ``storm_admission.NODE_COUNT_LAST_GOOD_SECONDS``).
COUNT_SETTLE_SECONDS = 65.0
VERDICT_FAILED = "storm breaker step verdict failed"
#: The longest the held workflows may wait for the release. Two warnings with
#: no ``cluster_id`` label -- so the drill sink does not keep them --
#: count a storm hold: GpuFaultRemediationBudgetStarved (a wait of 30 min) and
#: GpuFaultWorkflowPendingAge (a PENDING workflow older than 30 min, for 10 min).
HOLD_LIMIT_SECONDS = 1500
#: Status reads in a row a phase survives (a dropped exec stream).
MAX_FAILED_READS = 3


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def phase_budget_seconds(args: argparse.Namespace) -> int:
    """Every phase's bound and 600 s of setup: the executors' wall clock."""

    return int(
        args.trip_timeout_seconds
        + args.hold_timeout_seconds
        + args.drain_timeout_seconds
        + args.alert_timeout_seconds
        + 600
    )


class StormBreakerStep:
    """One run of the step: its phases, their history and the evidence."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        plan: dict[str, Any],
        run_id: str,
        artifacts: Path,
        resources: RunResources,
        registry_scope: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.args, self.plan, self.run_id = args, plan, run_id
        self.artifacts, self.resources = artifacts, resources
        self.registry_scope = registry_scope
        self.monotonic, self.sleep = monotonic, sleep
        self.started = monotonic()
        self.history: list[dict[str, Any]] = []
        self.phases: dict[str, dict[str, Any]] = {}
        self.violations: list[str] = []
        self.status: dict[str, Any] = {}
        self.held_ids: list[str] = []
        self.admitted_succeeded_before_release: list[str] = []
        self.seeded: dict[str, Any] | None = None
        self.seeded_at: datetime | None = None
        self.released_at: datetime | None = None
        self.release: dict[str, Any] | None = None
        self.trip: dict[str, Any] | None = None
        self.alert: dict[str, Any] = {"firing_at": None, "pending_at": None}
        self.gauges: dict[str, Any] = {"open_matched_at": None, "closed_at": None}
        self.refresh = run_agent_heartbeat_refresher(
            run_id=run_id, clusters=1, lease_seconds=args.lease_seconds
        )

    # -- reads ---------------------------------------------------------------

    def read(self, phase: str) -> dict[str, Any]:
        self.refresh()
        self.status = storm.breaker_status(
            run_id=self.run_id, workflows=self.plan["workflows"]
        )
        snapshot = storm.classify(self.status)
        if self.released_at is None and snapshot["breaker_open"]:
            self.violations.extend(
                item
                for item in storm.held_violations(snapshot, snapshot["product_held"])
                if item not in self.violations
            )
        if self.released_at is not None and snapshot["breaker_state"] != "CLOSED":
            reopened = (
                f"the breaker is {snapshot['breaker_state']} again after the release"
            )
            if reopened not in self.violations:
                self.violations.append(reopened)
        self.history.append(
            {
                "phase": phase,
                "at": utc_now().isoformat(),
                "elapsed_seconds": round(self.monotonic() - self.started, 3),
                **{
                    key: snapshot[key]
                    for key in ("breaker_state", "product_held", "succeeded")
                },
                "with_commands": len(snapshot["with_commands"]),
            }
        )
        return snapshot

    def read_alert(self) -> list[str]:
        try:
            states = storm.alert_states(CLUSTER)
        except Exception as exc:  # noqa: BLE001 - an unread alert is no alert
            self.alert["last_error"] = f"{type(exc).__name__}: {exc}"
            return []
        self.alert.pop("last_error", None)
        now = utc_now().isoformat()
        if "pending" in states and self.alert["pending_at"] is None:
            self.alert["pending_at"] = now
        if "firing" in states and self.alert["firing_at"] is None:
            self.alert["firing_at"] = now
        self.alert["last_states"] = states
        return states

    def read_gauges(self) -> dict[str, float]:
        try:
            values = storm.storm_gauges(CLUSTER)
        except Exception as exc:  # noqa: BLE001 - recorded; the criterion fails
            self.gauges["last_error"] = f"{type(exc).__name__}: {exc}"
            return {}
        self.gauges.pop("last_error", None)
        self.gauges["last"] = values
        return values

    # -- phases --------------------------------------------------------------

    def phase(
        self,
        name: str,
        timeout_seconds: float,
        done: Callable[[dict[str, Any]], bool],
    ) -> bool:
        started = self.monotonic()
        outcome = "timeout"
        record = self.phases.setdefault(name, {"read_errors": 0})
        failed_reads = 0
        while True:
            try:
                snapshot = self.read(name)
            except Exception as exc:  # noqa: BLE001 - a lost exec; bounded below
                failed_reads += 1
                record["read_errors"] += 1
                record["last_read_error"] = f"{type(exc).__name__}: {exc}"
                if failed_reads >= MAX_FAILED_READS:
                    raise
            else:
                failed_reads = 0
                if done(snapshot):
                    outcome = "done"
                    break
            failure = executor_job_failure(JOB_NAME, dataplane)
            if failure is not None:
                outcome = "executor-failed"
                record["executor_failure"] = failure
                break
            if self.monotonic() - started >= timeout_seconds:
                break
            self.sleep(self.args.poll_seconds)
        record.update(outcome=outcome, seconds=round(self.monotonic() - started, 3))
        log(f"storm breaker phase {name}: {outcome}")
        return outcome == "done"

    def tripped(self, snapshot: dict[str, Any]) -> bool:
        if snapshot["breaker_state"] != "TRIPPED":
            return False
        self.trip = dict(self.status["breaker"])
        return True

    def held_and_alerted(self, snapshot: dict[str, Any]) -> bool:
        plan = self.plan
        breaker = self.status.get("breaker") or {}
        held = snapshot["product_held"]
        others = sorted(set(self.status["workflows"]) - set(held))
        settled = (
            len(held) == plan["held_workflows"]
            and held == snapshot["held"]
            and set(others) <= set(snapshot["succeeded"])
            and breaker.get("notified_trip_id") == breaker.get("trip_id")
        )
        gauges = self.read_gauges()
        expected = {
            "open": 1.0,
            "threshold_nodes": float(plan["threshold_nodes"]),
            "window_nodes": float(plan["trip_nodes"]),
            "held_workflows": float(plan["held_workflows"]),
        }
        if self.gauges["open_matched_at"] is None and all(
            gauges.get(key) == value for key, value in expected.items()
        ):
            self.gauges["open_matched_at"] = utc_now().isoformat()
        firing = "firing" in self.read_alert()
        return settled and firing and self.gauges["open_matched_at"] is not None

    def completed(self, snapshot: dict[str, Any]) -> bool:
        return (
            len(snapshot["succeeded"]) == self.plan["workflows"]
            and snapshot["breaker_state"] == "CLOSED"
        )

    def resolved(self, _snapshot: dict[str, Any]) -> bool:
        closed = self.read_gauges().get("open") == 0.0
        if closed and self.gauges["closed_at"] is None:
            self.gauges["closed_at"] = utc_now().isoformat()
        states = self.read_alert()
        return closed and not states and "last_error" not in self.alert

    def do_release(self) -> None:
        self.release = storm.storm_operator_action(
            CLUSTER,
            "release",
            reason=f"perf storm breaker step {self.run_id}: the held work resumes",
            reference=f"PERF-STORM-{self.run_id}",
        )
        self.released_at = storm.parse_time(self.release["recorded_at"])
        log(f"storm breaker released: {self.release.get('status')}")

    def start_executors(self) -> None:
        upsert_configmap(
            SCRIPT_CONFIGMAP, resources=self.resources, text=executor_scripts()
        )
        manifest = executor_job(
            clusters=1,
            expected_commands=self.plan["commands"],
            workers=self.args.executor_workers,
            min_concurrent_commands=1,
            max_seconds=phase_budget_seconds(self.args),
            delay_scale=self.args.delay_scale,
            lease_seconds=self.args.lease_seconds,
            inject_renew_failure_once=False,
            **executor_identity(
                require_dataplane_deployment=self.registry_scope == "live"
            ),
        )
        ensure_load_service_account(
            self.resources, name=LOAD_SERVICE_ACCOUNT, namespace=NAMESPACE
        )
        manifest = run_manifest(manifest, self.run_id)
        (self.artifacts / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.resources.create(manifest)
        deadline = self.monotonic() + 300
        while self.monotonic() < deadline:
            running = dataplane(
                "get",
                "pod",
                "-l",
                f"batch.kubernetes.io/job-name={JOB_NAME}",
                "--field-selector=status.phase=Running",
                "-o",
                "name",
            ).splitlines()
            if running:
                return
            self.sleep(5)
        raise RuntimeError("storm breaker step executor did not start")

    def run(self, *, fleet_ready_at: float) -> dict[str, Any]:
        """The phases (module docstring); returns the summary."""

        self.start_executors()
        wait = COUNT_SETTLE_SECONDS - (self.monotonic() - fleet_ready_at)
        if wait > 0:
            self.sleep(wait)
        self.seeded = seed(
            run_id=self.run_id,
            clusters=1,
            workflows_per_cluster=self.plan["workflows"],
            nodes_per_workflow=self.plan["nodes_per_workflow"],
            agent_identity=(
                release_agent_identity() if self.registry_scope == "isolated" else None
            ),
        )
        self.seeded_at = utc_now()
        args = self.args
        if self.phase("trip", args.trip_timeout_seconds, self.tripped):
            self.phase("hold", args.hold_timeout_seconds, self.held_and_alerted)
            # What the release finds: the product's held list, and the
            # admitted workflows that already ran to SUCCEEDED while it was open.
            before = storm.classify(self.status)
            self.held_ids = before["product_held"]
            self.admitted_succeeded_before_release = sorted(
                set(before["succeeded"]) - set(self.held_ids)
            )
            self.do_release()
            if self.phase("drain", args.drain_timeout_seconds, self.completed):
                self.phase("resolve", args.alert_timeout_seconds, self.resolved)
        return self.summary()

    # -- evidence ------------------------------------------------------------

    def executor_evidence(self) -> dict[str, Any]:
        state = dataplane(
            "get",
            "job",
            JOB_NAME,
            "-o",
            "jsonpath={.status.succeeded}|{.status.failed}",
            check=False,
        )
        documents = collect_executor_logs(self.artifacts / "executors")
        return {"job_state": state.strip(), **aggregate_executor_documents(documents)}

    def summary(self) -> dict[str, Any]:
        plan, trip = self.plan, self.trip or {}
        workflow_ids = sorted(self.status.get("workflows") or {})
        admitted = sorted(set(workflow_ids) - set(self.held_ids))
        measurements = (
            storm.drain_measurements(
                self.status,
                released_at=self.released_at,
                held_ids=self.held_ids,
                admitted_ids=admitted,
            )
            if self.released_at is not None
            else {}
        )
        tripped_at = storm.parse_time(trip.get("tripped_at"))
        if tripped_at and self.seeded_at:
            measurements["seed_to_trip_seconds"] = round(
                (tripped_at - self.seeded_at).total_seconds(), 3
            )
        if tripped_at and self.alert["firing_at"]:
            measurements["trip_to_alert_firing_seconds"] = round(
                (
                    storm.parse_time(self.alert["firing_at"]) - tripped_at
                ).total_seconds(),
                3,
            )
        executor = self.executor_evidence()
        return {
            "run_id": self.run_id,
            "plan": plan,
            "seeded": self.seeded,
            "phases": self.phases,
            "trip": trip,
            "held_workflows": self.held_ids,
            "admitted_workflows": admitted,
            "admitted_succeeded_before_release": self.admitted_succeeded_before_release,
            "release": self.release,
            "alert": self.alert,
            "gauges": self.gauges,
            "final_breaker": self.status.get("breaker"),
            "violations": self.violations,
            "measurements": measurements,
            "executor": executor,
        }

    def close_breaker(self) -> dict[str, Any]:
        """Whatever ended the step: leave the breaker CLOSED, read it so."""

        record: dict[str, Any] = {}
        try:
            before = storm.classify(
                storm.breaker_status(
                    run_id=self.run_id, workflows=self.plan["workflows"]
                )
            )
            record["state_before"] = before["breaker_state"]
            options = {
                "reason": f"perf storm breaker step {self.run_id} ends",
                "reference": f"PERF-STORM-{self.run_id}",
            }
            if before["breaker_state"] == "TRIPPED" and before["product_held"]:
                record["abort"] = storm.storm_operator_action(
                    CLUSTER, "abort", **options
                )
            if before["breaker_open"]:
                record["release"] = storm.storm_operator_action(
                    CLUSTER, "release", **options
                )
            after = storm.breaker_status(
                run_id=self.run_id, workflows=self.plan["workflows"]
            )
            record["state_after"] = storm.classify(after)["breaker_state"]
        except Exception as exc:  # noqa: BLE001 - the fleet's exit tries again
            record["error"] = f"{type(exc).__name__}: {exc}"
        (self.artifacts / "storm-breaker-exit.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        return record


def verdict(summary: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    """PASS when every part of the breaker's cycle held (module docstring)."""

    plan, trip = summary["plan"], summary["trip"] or {}
    phases = summary["phases"]
    final = summary["final_breaker"] or {}
    executor = summary["executor"]
    held = summary["held_workflows"]
    checks = {
        "fleet_counted": trip.get("trip_cluster_nodes") == plan["nodes_per_cluster"]
        and trip.get("trip_threshold") == plan["threshold_nodes"],
        "tripped_at_planned_admission": len(trip.get("trip_node_ids") or [])
        == plan["trip_nodes"]
        and trip.get("trip_domains") == plan["trip_domains"]
        and trip.get("trip_workflow_id") in held,
        "held_while_open": len(held) == plan["held_workflows"]
        and not summary["violations"],
        "in_flight_completed_while_open": len(
            summary["admitted_succeeded_before_release"]
        )
        == plan["admitted_workflows"],
        "operator_notice": bool(trip.get("trip_id"))
        and final.get("notified_trip_id") == trip.get("trip_id"),
        "gauges_read_the_trip": summary["gauges"].get("open_matched_at") is not None,
        "alert_firing": summary["alert"].get("firing_at") is not None,
        "released": (summary["release"] or {}).get("status") == "released",
        "held_completed_after_release": (phases.get("drain") or {}).get("outcome")
        == "done",
        "no_recount_after_release": final.get("state") == "CLOSED"
        and final.get("window_nodes") == 0
        and final.get("trip_id") == trip.get("trip_id"),
        "alert_resolved": (phases.get("resolve") or {}).get("outcome") == "done",
        "executor_clean": executor.get("completed_commands") == plan["commands"]
        and not executor.get("duplicate_claims")
        and not executor.get("claim_errors")
        and not executor.get("result_errors"),
    }
    return ("PASS" if all(checks.values()) else "FAIL"), checks


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_fleet_argument(value)
    value.add_argument("--nodes-per-workflow", type=int, default=4)
    value.add_argument("--held-workflows", type=int, default=4)
    value.add_argument("--executor-workers", type=int, default=16)
    value.add_argument("--delay-scale", type=float, default=1.0)
    value.add_argument("--lease-seconds", type=int, default=120)
    value.add_argument("--poll-seconds", type=float, default=10.0)
    value.add_argument("--trip-timeout-seconds", type=int, default=300)
    value.add_argument("--hold-timeout-seconds", type=int, default=900)
    value.add_argument("--drain-timeout-seconds", type=int, default=900)
    value.add_argument("--alert-timeout-seconds", type=int, default=600)
    value.add_argument("--synthetic-ttl-seconds", type=int, default=5400)
    value.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    value.add_argument("--label")
    value.add_argument("--suite-id")
    value.add_argument("--allow-live-registry", action="store_true")
    value.add_argument("--confirm-live-registry")
    return value


def write_run_record(
    artifacts: Path,
    args: argparse.Namespace,
    *,
    run_id: str,
    plan: dict[str, Any],
    fleet: dict[str, Any],
    resume: dict[str, str],
) -> None:
    record = {
        "run_id": run_id,
        "storm_breaker_plan": plan,
        "synthetic_fleet": fleet,
        "timeouts_seconds": {
            name: getattr(args, f"{name}_timeout_seconds")
            for name in ("trip", "hold", "drain", "alert")
        },
        "executor_workers": args.executor_workers,
        "delay_scale": args.delay_scale,
        "lease_seconds": args.lease_seconds,
        "started_at": utc_now().isoformat(),
        **resume,
    }
    (artifacts / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


@abort_on_signals
def main(argv: list[str] | None = None) -> int:
    options = parser()
    args = options.parse_args(argv)
    if problem := dataplane_context_problem():
        options.error(problem)
    try:
        registry_scope = validate_registry_target(
            allow_live_registry=args.allow_live_registry,
            confirmation=args.confirm_live_registry,
        )
        plan = storm.breaker_plan(
            nodes_per_cluster=args.fleet_nodes_per_cluster,
            nodes_per_workflow=args.nodes_per_workflow,
            held_workflows=args.held_workflows,
        )
        fleet = fleet_plan(
            clusters=1,
            model_nodes_per_cluster=plan["model_nodes"],
            nodes_per_cluster=args.fleet_nodes_per_cluster,
            storm=True,
        )
    except (RuntimeError, ValueError) as exc:
        options.error(str(exc))
    if args.synthetic_ttl_seconds < phase_budget_seconds(args) + 600:
        options.error("--synthetic-ttl-seconds must outlast every phase by 600 s")
    if args.trip_timeout_seconds + args.hold_timeout_seconds > HOLD_LIMIT_SECONDS:
        options.error(
            "--trip-timeout-seconds + --hold-timeout-seconds must stay within "
            f"{HOLD_LIMIT_SECONDS} s: a longer hold fires the site-wide "
            "GpuFaultRemediationBudgetStarved and GpuFaultWorkflowPendingAge"
        )
    run_id = args.suite_id or utc_now().strftime("storm%Y%m%dT%H%M%S")
    identity = release_identity()
    artifacts = artifact_dir(
        args.artifact_root,
        args.label or "storm-breaker",
        identity["release_id"] or "unknown-release",
    )
    if any(artifacts.iterdir()):
        raise RuntimeError("storm breaker artifacts already exist")
    write_run_record(
        artifacts,
        args,
        run_id=run_id,
        plan=plan,
        fleet=fleet,
        resume=resume_identity(
            suite_id=run_id, scope=registry_scope, identity=identity
        ),
    )
    write_status(artifacts, status="running")
    log(storm.plan_line(plan))
    failure: BaseException | None = None
    cleanup_error: Exception | None = None
    result = 1
    try:
        if not storm.alert_rule_loaded():
            raise RuntimeError(
                f"the site's AMP rules carry no {storm.ALERT_NAME}: deploy the "
                "release with the remediation storm breaker first"
            )
        register(
            1,
            artifacts,
            run_id=run_id,
            expires_at=utc_now() + timedelta(seconds=args.synthetic_ttl_seconds),
            allow_live_registry=args.allow_live_registry,
            live_registry_confirmation=args.confirm_live_registry,
        )
        validate_registered_resources(artifacts, run_id, 1)
        resources = RunResources(artifacts, run_id, NAMESPACE, dataplane)
        with synthetic_fleet(
            artifacts,
            run_id=run_id,
            plan=fleet,
            agent_identity=(
                release_agent_identity() if registry_scope == "isolated" else None
            ),
            deadline_seconds=args.synthetic_ttl_seconds,
            expected_trips=frozenset({CLUSTER}),
        ):
            step = StormBreakerStep(
                args,
                plan=plan,
                run_id=run_id,
                artifacts=artifacts,
                resources=resources,
                registry_scope=registry_scope,
            )
            try:
                summary = step.run(fleet_ready_at=time.monotonic())
            finally:
                with cleanup_signal_guard():  # the cleanup starts here
                    exit_record = step.close_breaker()
                # Every poll, also when a phase raised: the evidence of what it saw.
                (artifacts / "storm-breaker-history.json").write_text(
                    json.dumps(step.history, indent=2, sort_keys=True) + "\n"
                )
            summary["breaker_exit"] = exit_record
            status, checks = verdict(summary)
            summary.update(status=status, checks=checks)
            (artifacts / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            result = int(status != "PASS")
    except BaseException as exc:
        failure = exc
    finally:
        with cleanup_signal_guard() as late:
            try:
                if (artifacts / "registry-registration-intent.json").exists():
                    teardown(
                        purge=True,
                        deregister_clusters=True,
                        allow_live_registry=args.allow_live_registry,
                        live_registry_confirmation=args.confirm_live_registry,
                        artifacts=artifacts,
                        run_id=run_id,
                    )
                    verify_fleet_cleanup(artifacts, run_id=run_id, clusters=1)
            except Exception as exc:
                cleanup_error = exc
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"storm breaker teardown also failed: {exc}")
        failure = late.abort(failure)
    if failure is not None:
        write_aborted_status(
            artifacts,
            failure,
            cleanup_error=cleanup_error,
            run_id=run_id,
            scope=registry_scope,
        )
        raise failure.with_traceback(failure.__traceback__)
    write_status(artifacts, **step_status(result, VERDICT_FAILED))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
