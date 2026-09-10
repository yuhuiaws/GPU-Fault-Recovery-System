#!/usr/bin/env python3
"""GF-REGIONAL-HA-010: an Aurora writer failover must not restart CPU Pods.

No workflow is in flight. The runner asks RDS to fail the writer over --
HA-003's mechanism, reused -- while an in-Pod sampler on every CPU
control-plane Pod records ``/livez`` and ``/healthz`` every two seconds.
Liveness is process-local (ARCH-H1) and must answer 200 throughout;
readiness may refuse with 503 while the registry refresh is stale but never
crash into another 5xx, and it must be 200 again within the registry stale
window plus one refresh after RDS reports ``available``. No pre-existing
container may restart. Immediately after the failover request one
``gpu-fault-api-ha`` replica is deleted (H-2): its replacement has to come up
through the bounded start-up retry, never CrashLoopBackOff. ``/healthz``
reports ``regional_registry.secret_drift=false`` before and after (ARCH-H3).

The case is refused while any remote command is open or the processor queue
holds work: a failover under a live destructive step is HA-003's job, and
mixing the two would make a restart here unattributable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import ha010_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
PROBE_SCRIPT = Path(__file__).with_name("probes") / "ha010_probe.py"
HEALTHZ_READ = r"""
import json
import sys
from urllib.error import HTTPError
from urllib.request import urlopen

port = sys.argv[1]
try:
    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
        body = response.read()
        status = response.status
except HTTPError as exc:
    body = exc.read()
    status = exc.code
try:
    payload = json.loads(body) if body else {}
except ValueError:
    payload = {"unparsable": body.decode("utf-8", "replace")[:500]}
print(json.dumps({"http_status": status, "payload": payload}, default=str))
"""
ENV_READ = r"""
import json
import os
import sys

print(json.dumps({name: os.getenv(name) for name in sys.argv[1:]}))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    rds_cluster_id: str
    predecessor_path: Path
    observe_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_AURORA_CLUSTER_ID": self.rds_cluster_id,
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
    low, high = verdicts.OBSERVE_SECONDS_BOUNDS
    if not low <= int(arguments.observe_seconds) <= high:
        raise RegionalFixtureError(f"observe seconds is outside {low}..{high}")
    return Settings(
        regional=settings_from_arguments(arguments),
        rds_cluster_id=required(
            arguments.rds_cluster_id or os.getenv("GPU_FAULT_AURORA_CLUSTER_ID", ""),
            "Aurora cluster ID",
        ),
        predecessor_path=predecessor,
        observe_seconds=int(arguments.observe_seconds),
    )


def _rds(settings: Settings) -> Any:
    """HA-003's RDS helpers read ``regional.region`` and ``rds_cluster_id``."""

    return cast(Any, settings)


def rds_snapshot(settings: Settings) -> dict[str, Any]:
    return ha003.rds_snapshot(_rds(settings))


# --------------------------------------------------------------------------- #
# Pods
# --------------------------------------------------------------------------- #
def _http_port(pod: dict[str, Any]) -> int | None:
    for container in (pod.get("spec") or {}).get("containers") or []:
        for port in container.get("ports") or []:
            if port.get("name") == "http" and isinstance(
                port.get("containerPort"), int
            ):
                return int(port["containerPort"])
    return None


def pod_record(pod: dict[str, Any]) -> dict[str, Any]:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    reasons: list[str] = []
    for status in statuses:
        for state_key in ("state", "lastState"):
            waiting = (status.get(state_key) or {}).get("waiting") or {}
            if waiting.get("reason"):
                reasons.append(str(waiting["reason"]))
    ready_at = None
    for condition in (pod.get("status") or {}).get("conditions") or []:
        if condition.get("type") == "Ready" and condition.get("status") == "True":
            ready_at = condition.get("lastTransitionTime")
    return {
        "name": pod["metadata"]["name"],
        "uid": pod["metadata"]["uid"],
        "node": (pod.get("spec") or {}).get("nodeName"),
        "phase": (pod.get("status") or {}).get("phase"),
        "ready": bool(statuses)
        and all(bool(status.get("ready")) for status in statuses),
        "restarts": sum(int(status.get("restartCount") or 0) for status in statuses),
        "waiting_reasons": sorted(set(reasons)),
        "ready_at": ready_at,
        "port": _http_port(pod),
        "created_at": pod["metadata"].get("creationTimestamp"),
    }


def pods_by_deployment(
    regional: RegionalLiveFixture,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for deployment in verdicts.CPU_DEPLOYMENTS:
        value = json.loads(
            regional.kubectl(
                "cpu", "get", "pod", "-l", f"app={deployment}", "-o", "json"
            )
        )
        result[deployment] = sorted(
            (pod_record(item) for item in value.get("items", [])),
            key=lambda item: str(item["name"]),
        )
    return result


def deployments_scaled_to_zero(regional: RegionalLiveFixture) -> frozenset[str]:
    """The CPU Deployments the site runs at 0 replicas (spool admission off)."""

    zero = set()
    for deployment in verdicts.CPU_DEPLOYMENTS:
        replicas = regional.kubectl(
            "cpu", "get", "deployment", deployment, "-o", "jsonpath={.spec.replicas}"
        ).strip()
        if replicas in {"", "0"}:
            zero.add(deployment)
    return frozenset(zero)


def healthz_by_pod(
    regional: RegionalLiveFixture,
    pods: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for records in pods.values():
        for record in records:
            if not record.get("ready") or record.get("port") is None:
                continue
            output = regional.kubectl(
                "cpu",
                "exec",
                "-i",
                str(record["name"]),
                "--",
                "python3",
                "-",
                str(record["port"]),
                input_text=HEALTHZ_READ,
                timeout=60,
            )
            result[str(record["name"])] = json.loads(output.splitlines()[-1])
    return result


def live_env(
    regional: RegionalLiveFixture,
    pod: str,
    names: tuple[str, ...],
) -> dict[str, str | None]:
    output = regional.kubectl(
        "cpu",
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-",
        *names,
        input_text=ENV_READ,
        timeout=60,
    )
    return cast(dict[str, str | None], json.loads(output.splitlines()[-1]))


def _first_ready_pod(pods: dict[str, list[dict[str, Any]]], deployment: str) -> str:
    for record in pods.get(deployment) or []:
        if record.get("ready"):
            return str(record["name"])
    raise RegionalFixtureError(f"no Ready Pod for {deployment}")


# --------------------------------------------------------------------------- #
# Preflight and plan
# --------------------------------------------------------------------------- #
def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    pods = pods_by_deployment(regional)
    state = regional.store_snapshot()
    rds = rds_snapshot(settings)
    healthz = healthz_by_pod(regional, pods)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    runtime_identity = regional.runtime_identity()
    api_pod = _first_ready_pod(pods, verdicts.ROLLED_DEPLOYMENT)
    env = live_env(
        regional,
        api_pod,
        (verdicts.STALE_SECONDS_VARIABLE, verdicts.STARTUP_RETRY_VARIABLE),
    )
    scaled_to_zero = deployments_scaled_to_zero(regional)
    errors = verdicts.preflight_errors(
        pods=pods,
        rds=rds,
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        healthz=healthz,
        predecessor_valid=bool(predecessor["valid"]),
        scaled_to_zero=scaled_to_zero,
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    budgets: dict[str, float] = {}
    try:
        budgets["stale_seconds"] = verdicts.positive_seconds(
            env.get(verdicts.STALE_SECONDS_VARIABLE),
            default=verdicts.DEFAULT_STALE_SECONDS,
            label=verdicts.STALE_SECONDS_VARIABLE,
        )
        budgets["startup_retry_seconds"] = verdicts.positive_seconds(
            env.get(verdicts.STARTUP_RETRY_VARIABLE),
            default=verdicts.DEFAULT_STARTUP_RETRY_SECONDS,
            label=verdicts.STARTUP_RETRY_VARIABLE,
        )
    except ValueError as exc:
        errors.append(str(exc))
    result = {
        "release_id": state.get("release_id"),
        "pods": pods,
        "healthz": healthz,
        "rds": rds,
        "store": {
            "queue": state.get("queue"),
            "remote_commands": state.get("remote_commands"),
        },
        "live_env": env,
        "budgets": budgets,
        "predecessor": predecessor,
        "runtime_identity": runtime_identity,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_id": preflight["release_id"],
        "rds_writer": (preflight.get("rds") or {}).get("writer"),
        "rds_members": [
            item.get("identifier")
            for item in (preflight.get("rds") or {}).get("members") or []
        ],
        "pod_uids": {
            deployment: sorted(str(item["uid"]) for item in records)
            for deployment, records in (preflight.get("pods") or {}).items()
        },
        "budgets": preflight.get("budgets"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-service-action",
        "predecessor": preflight["predecessor"],
        "rds_cluster_id": settings.rds_cluster_id,
        "observe_seconds": settings.observe_seconds,
        "mutation": (
            "call RDS failover-db-cluster once with no workflow in flight, and "
            "delete exactly one Ready gpu-fault-api-ha Pod right after the "
            "request; sample /livez and /healthz inside every CPU Pod. No GPU "
            "node, Node Agent, provider or executor action is involved."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "any remote command is open or the processor queue holds work",
            "Aurora is not available or has no failover-capable reader",
            f"fewer than {verdicts.MINIMUM_ROLLED_READY_REPLICAS} Ready "
            f"{verdicts.ROLLED_DEPLOYMENT} replicas",
            "any CPU control-plane Pod is not Ready or reports secret_drift",
            "the live stale/start-up budgets are not positive numbers",
            "runtime identity is unsafe or drifts during the case",
        ],
        "rollback": {
            "aurora_failover_is_allowed_to_complete_forward": True,
            "no_destructive_step_can_be_in_flight": True,
            "samplers_are_bounded_by_observe_seconds": True,
            "deleted_replica_is_recreated_by_its_deployment": True,
            "runner_finally_waits_for_every_deployment_to_be_ready": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def feed_script(process: subprocess.Popen[str], text: str) -> None:
    """Hand the probe its script on stdin and detach the pipe.

    ``Popen.communicate`` flushes ``stdin`` whenever the attribute is set, and
    a pipe that was closed by hand is a closed file: the first live run that
    reached the collection step died with ``I/O operation on closed file``.
    Dropping the attribute after the close tells ``communicate`` there is no
    stdin left to flush.
    """

    if process.stdin is None:
        raise RegionalFixtureError("sampler stdin pipe was not opened")
    process.stdin.write(text)
    process.stdin.close()
    process.stdin = None


@dataclass
class Sampler:
    pod: str
    process: subprocess.Popen[str]
    started_at: datetime

    def collect(self, timeout: float) -> dict[str, Any]:
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            stdout, stderr = self.process.communicate()
        lines = stdout.strip().splitlines()
        payload: dict[str, Any] = {}
        if lines:
            try:
                payload = json.loads(lines[-1])
            except ValueError:
                payload = {}
        return {
            "pod": self.pod,
            "returncode": self.process.returncode,
            "stderr": stderr[-2000:],
            "started_at": self.started_at.isoformat(),
            **payload,
        }

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()


def start_sampler(
    regional: RegionalLiveFixture,
    record: dict[str, Any],
    *,
    duration_seconds: int,
) -> Sampler:
    command = [
        "kubectl",
        "--kubeconfig",
        str(regional.settings.cpu_kubeconfig),
        "-n",
        regional.settings.namespace,
        "exec",
        "-i",
        str(record["name"]),
        "--",
        "python3",
        "-",
        str(record["port"]),
        str(duration_seconds),
        str(verdicts.SAMPLE_INTERVAL_SECONDS),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    feed_script(process, PROBE_SCRIPT.read_text(encoding="utf-8"))
    return Sampler(
        pod=str(record["name"]),
        process=process,
        started_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    run_id: str
    samplers: list[Sampler] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failover_requested_at: datetime | None = None
    rds_available_at: datetime | None = None
    deleted_pod: str = ""
    deleted_at: datetime | None = None


def _prepare_live_run(settings: Settings, run_dir: Path, attempt: int) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    return _LiveRun(
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        preflight=preflight,
        run_id=f"ha010-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}",
    )


def _quiet_control_plane(run: _LiveRun) -> None:
    """Refuse the failover while anything is open for this control plane."""

    state = run.regional.store_snapshot(queue_attempts=1)
    write_json_atomic(run.case_dir / "store-before-failover.json", state)
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        raise RegionalFixtureError(
            "remote commands are open; a failover now is HA-003, not HA-010"
        )
    if processor_queue_backlog(state.get("queue") or {}):
        raise RegionalFixtureError(
            "the processor queue is not empty before the failover"
        )


def _start_samplers(run: _LiveRun) -> None:
    for records in run.preflight["pods"].values():
        for record in records:
            run.samplers.append(
                start_sampler(
                    run.regional,
                    record,
                    duration_seconds=run.settings.observe_seconds,
                )
            )
    # Let every exec channel open and take its first sample before the
    # failover, so each timeline provably starts ahead of the request.
    time.sleep(2 * verdicts.SAMPLE_INTERVAL_SECONDS + 3)


def request_failover(run: _LiveRun) -> dict[str, Any]:
    _quiet_control_plane(run)
    run.failover_requested_at = datetime.now(timezone.utc)
    failover = ha003.aws_rds(
        _rds(run.settings),
        "failover-db-cluster",
        "--db-cluster-identifier",
        run.settings.rds_cluster_id,
    )
    write_json_atomic(run.case_dir / "failover-request.json", failover)
    run.deleted_pod = _first_ready_pod(
        run.preflight["pods"], verdicts.ROLLED_DEPLOYMENT
    )
    run.deleted_at = datetime.now(timezone.utc)
    run.regional.kubectl(
        "cpu",
        "delete",
        "pod",
        run.deleted_pod,
        "--wait=false",
        timeout=120,
    )
    write_json_atomic(
        run.case_dir / "deleted-pod.json",
        {"pod": run.deleted_pod, "deleted_at": run.deleted_at.isoformat()},
    )
    # HA-003's wait returns the final RDS document *and* the samples it took
    # on the way; the first live run unpacked neither and died on the merge
    # below with the failover already requested and the Pod already deleted.
    rds_after, failover_samples = ha003.wait_rds_failover(
        _rds(run.settings),
        previous_writer=str(run.preflight["rds"]["writer"]),
        timeout_seconds=verdicts.FAILOVER_TIMEOUT_SECONDS,
    )
    run.rds_available_at = datetime.now(timezone.utc)
    write_json_atomic(
        run.case_dir / "rds-after.json",
        {**rds_after, "available_at": run.rds_available_at.isoformat()},
    )
    write_json_atomic(
        run.case_dir / "rds-failover-samples.json", {"samples": failover_samples}
    )
    return rds_after


def _wait_replacement(run: _LiveRun) -> dict[str, Any]:
    if run.deleted_at is None:
        raise RegionalFixtureError("the replacement wait follows the deletion")
    budget = float(run.preflight["budgets"]["startup_retry_seconds"]) + (
        verdicts.REPLACEMENT_READY_MARGIN_SECONDS
    )
    known = {
        str(item["uid"]) for item in run.preflight["pods"][verdicts.ROLLED_DEPLOYMENT]
    }
    deadline = time.monotonic() + budget + 60
    seen_reasons: set[str] = set()
    replacement: dict[str, Any] = {}
    while time.monotonic() < deadline:
        records = pods_by_deployment(run.regional)[verdicts.ROLLED_DEPLOYMENT]
        fresh = [item for item in records if str(item["uid"]) not in known]
        if fresh:
            replacement = fresh[0]
            seen_reasons.update(str(item) for item in replacement["waiting_reasons"])
            if replacement.get("ready"):
                break
        time.sleep(5)
    replacement = {**replacement, "waiting_reasons": sorted(seen_reasons)}
    logs = ""
    if replacement.get("name"):
        logs = run.regional.kubectl(
            "cpu",
            "logs",
            str(replacement["name"]),
            check=False,
            timeout=120,
        )
    relevant = verdicts.relevant_log_lines(logs)
    replacement["logs_sha256"] = hashlib.sha256(logs.encode()).hexdigest()
    replacement["relevant_log_lines"] = relevant
    replacement["startup_retry_observed"] = verdicts.startup_retry_observed(
        logs.splitlines()
    )
    replacement["budget_seconds"] = budget
    write_json_atomic(run.case_dir / "replacement-pod.json", replacement)
    return replacement


def _wait_deployments_ready(run: _LiveRun, timeout_seconds: int = 600) -> None:
    for deployment in verdicts.CPU_DEPLOYMENTS:
        run.regional.kubectl(
            "cpu",
            "rollout",
            "status",
            f"deployment/{deployment}",
            f"--timeout={timeout_seconds}s",
            timeout=timeout_seconds + 60,
        )


def _collect_samplers(run: _LiveRun) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    remaining = float(run.settings.observe_seconds) + 60
    for sampler in run.samplers:
        elapsed = (datetime.now(timezone.utc) - sampler.started_at).total_seconds()
        report = sampler.collect(max(5.0, remaining - elapsed))
        write_json_atomic(run.case_dir / "samplers" / f"{sampler.pod}.json", report)
        result[sampler.pod] = report
    return result


def _timeline_errors(
    run: _LiveRun,
    reports: dict[str, dict[str, Any]],
) -> list[str]:
    if run.failover_requested_at is None or run.rds_available_at is None:
        raise RegionalFixtureError("timelines are judged after the failover")
    stale = float(run.preflight["budgets"]["stale_seconds"])
    errors: list[str] = []
    for pod, report in sorted(reports.items()):
        if pod == run.deleted_pod:
            # Its exec channel died with the Pod; the replacement is judged
            # separately and the deletion itself is the case's own act.
            continue
        errors.extend(
            verdicts.sampler_errors(
                list(report.get("samples") or []),
                pod=pod,
                stale_seconds=stale,
                rds_available_at=run.rds_available_at,
                failover_requested_at=run.failover_requested_at,
            )
        )
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
        "rds_cluster_id": settings.rds_cluster_id,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        write_json_atomic(run.case_dir / "pods-before.json", run.preflight["pods"])
        _start_samplers(run)
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before the failover"
            )
        rds_after = request_failover(run)
        replacement = _wait_replacement(run)
        _wait_deployments_ready(run)
        reports = _collect_samplers(run)
        pods_after = pods_by_deployment(run.regional)
        write_json_atomic(run.case_dir / "pods-after.json", pods_after)
        healthz_after = healthz_by_pod(run.regional, pods_after)
        write_json_atomic(run.case_dir / "healthz-after.json", healthz_after)
        if run.deleted_at is None:
            raise RegionalFixtureError("the deletion moment was not recorded")
        errors = _timeline_errors(run, reports)
        errors.extend(
            verdicts.restart_errors(
                run.preflight["pods"], pods_after, deleted_pod=run.deleted_pod
            )
        )
        errors.extend(
            verdicts.replacement_errors(
                replacement,
                deleted_pod=run.deleted_pod,
                deleted_at=run.deleted_at,
                budget_seconds=float(replacement.get("budget_seconds") or 0),
            )
        )
        errors.extend(verdicts.registry_errors(healthz_after, label="after failover"))
        outages = {
            pod: verdicts.readiness_outage_seconds(list(report.get("samples") or []))
            for pod, report in reports.items()
            if pod != run.deleted_pod
        }
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "failover_requested_at": run.failover_requested_at.isoformat()
                if run.failover_requested_at
                else None,
                "rds_available_at": run.rds_available_at.isoformat()
                if run.rds_available_at
                else None,
                "rds_before": run.preflight["rds"],
                "rds_after": rds_after,
                "deleted_pod": run.deleted_pod,
                "replacement_pod": replacement.get("name"),
                "startup_retry_observed": replacement.get("startup_retry_observed"),
                "readiness_outage_seconds_by_pod": outages,
                "budgets": run.preflight["budgets"],
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
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

    def stop_samplers() -> dict[str, bool]:
        stopped = {}
        for sampler in run.samplers:
            sampler.stop()
            stopped[sampler.pod] = sampler.process.poll() is not None
        return stopped

    guard("samplers_stopped", stop_samplers)

    def deployments_ready() -> bool:
        _wait_deployments_ready(run)
        return True

    guard("deployments_ready", deployments_ready)
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage=f"after {CASE_ID} cleanup",
        ),
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded HA-010 acceptance: one Aurora writer failover with "
            "no workflow in flight, in-Pod /livez and /healthz sampling, and one "
            "api-ha replica deleted into the outage."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--rds-cluster-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--observe-seconds",
        type=int,
        default=verdicts.OBSERVE_SECONDS_DEFAULT,
        help="how long each in-Pod sampler runs (failover + stale window + margin)",
    )
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
