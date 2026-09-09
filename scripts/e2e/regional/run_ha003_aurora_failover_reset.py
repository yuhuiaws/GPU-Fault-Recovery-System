#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    PROVIDER_MUTATIONS,
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    WaitingEvidence,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

CASE_ID = "GF-REGIONAL-HA-003"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-008"
CONFIRMATION = "HA003_FAILOVER_AURORA_DURING_GPU_RESET"
NON_TERMINAL_COMMAND_STATUSES = {"PENDING", "LEASED", "WAITING"}
# The executor logs a rejected result as "could not report result" followed by
# the control plane's "rejected request (NNN)"; an accepted result logs nothing.
RESULT_REJECTED_LINE = "could not report result"
REJECTED_STATUS_PATTERN = re.compile(r"rejected request \((\d{3})\)")
ALLOWED_RESULT_SUBMISSION_CODES = {200, 409}


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    rds_cluster_id: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
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
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        rds_cluster_id=required(
            arguments.rds_cluster_id or os.getenv("GPU_FAULT_AURORA_CLUSTER_ID", ""),
            "Aurora cluster ID",
        ),
        predecessor_path=predecessor,
    )


def aws_rds(settings: Settings, *arguments: str) -> dict[str, Any]:
    value = json.loads(
        RegionalLiveFixture.run(
            [
                "aws",
                "rds",
                *arguments,
                "--region",
                settings.regional.region,
                "--output",
                "json",
            ],
            timeout=180,
        ).stdout
    )
    if not isinstance(value, dict):
        raise RegionalFixtureError("RDS response is not a JSON object")
    return value


def rds_snapshot(settings: Settings) -> dict[str, Any]:
    value = aws_rds(
        settings,
        "describe-db-clusters",
        "--db-cluster-identifier",
        settings.rds_cluster_id,
    )
    clusters = value.get("DBClusters", [])
    if len(clusters) != 1:
        raise RegionalFixtureError("Aurora lookup did not return one cluster")
    cluster = clusters[0]
    members = sorted(
        [
            {
                "identifier": item.get("DBInstanceIdentifier"),
                "writer": bool(item.get("IsClusterWriter")),
                "promotion_tier": item.get("PromotionTier"),
            }
            for item in cluster.get("DBClusterMembers", [])
        ],
        key=lambda item: str(item["identifier"]),
    )
    writers = [str(item["identifier"]) for item in members if item["writer"]]
    return {
        "status": cluster.get("Status"),
        "endpoint": cluster.get("Endpoint"),
        "reader_endpoint": cluster.get("ReaderEndpoint"),
        "members": members,
        "writer": writers[0] if len(writers) == 1 else None,
    }


def reset_command_status(state: dict[str, Any]) -> str | None:
    commands = [
        item
        for item in state.get("commands") or []
        if item.get("step", {}).get("operation") == "RESET_GPU"
    ]
    if len(commands) != 1:
        return None
    return cast(str | None, commands[0].get("status"))


def wait_rds_failover(
    settings: Settings,
    *,
    previous_writer: str,
    timeout_seconds: int = 900,
    observe: Callable[[], dict[str, Any] | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll Aurora until the writer moved, pairing each poll with a store sample.

    ``observe`` returns the store snapshot taken alongside the RDS read (or
    None); each pairing is kept as a sample so ``failover_overlap_observed``
    can decide afterwards whether the writer change happened while the reset
    command was still open. The workflow keeps moving during the failover, so
    the WAITING records of the steps finishing in this window are read here too.
    """

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = rds_snapshot(settings)
        state = observe() if observe is not None else None
        samples.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "rds_status": last.get("status"),
                "writer": last.get("writer"),
                "reset_command_status": (
                    reset_command_status(state) if state is not None else None
                ),
            }
        )
        if (
            last.get("status") == "available"
            and last.get("writer")
            and last.get("writer") != previous_writer
        ):
            return last, samples
        sleep(5)
    raise RegionalFixtureError(f"Aurora failover did not converge: {last}")


def failover_overlap_observed(
    samples: list[dict[str, Any]],
    *,
    previous_writer: str,
) -> bool:
    """Whether Aurora was seen failing over while RESET_GPU was still open.

    Requesting the failover when the command is LEASED/WAITING proves nothing
    about the window itself: the writer may only switch after the command has
    already reached SUCCEEDED, in which case the case observed a reset and a
    failover, not a reset *during* a failover. At least one sample must show
    the cluster mid-failover (writer changed or status not ``available``) with
    the command still non-terminal.
    """

    for sample in samples:
        mid_failover = sample.get("rds_status") != "available" or (
            sample.get("writer") and sample.get("writer") != previous_writer
        )
        if mid_failover and sample.get("reset_command_status") in (
            NON_TERMINAL_COMMAND_STATUSES
        ):
            return True
    return False


def result_submission_codes(logs: str) -> list[int]:
    """HTTP codes of rejected result submissions in executor logs.

    A rejected ``complete`` logs "could not report result" followed by the
    traceback whose last line carries "rejected request (NNN)"; an accepted
    result is silent. So the codes returned here are the non-200 submissions,
    and the catalog allows only 409 among them.
    """

    codes: list[int] = []
    pending = False
    for line in logs.splitlines():
        if RESULT_REJECTED_LINE in line:
            pending = True
            continue
        if pending:
            match = REJECTED_STATUS_PATTERN.search(line)
            if match:
                codes.append(int(match.group(1)))
                pending = False
    return codes


def executor_logs(
    regional: RegionalLiveFixture,
    started_at: datetime,
) -> dict[str, Any]:
    entries = []
    codes: list[int] = []
    for pod in regional.ready_pods("gpu", "gpu-fault-cluster-executor"):
        output = regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            "--since-time",
            started_at.isoformat(),
            check=False,
            timeout=120,
        )
        pod_codes = result_submission_codes(output)
        codes.extend(pod_codes)
        entries.append(
            {
                "pod": pod["name"],
                "sha256": hashlib.sha256(output.encode()).hexdigest(),
                "rejected_result_codes": pod_codes,
                "relevant_lines": [
                    line[:500]
                    for line in output.splitlines()
                    if RESULT_REJECTED_LINE in line or "rejected request" in line
                ][-50:],
            }
        )
    return {
        "entries": entries,
        "rejected_result_codes": codes,
        "forbidden_result_codes": sorted(
            {code for code in codes if code not in ALLOWED_RESULT_SUBMISSION_CODES}
        ),
    }


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/execution/test_executor.py::"
        "test_transient_store_error_does_not_block_active_workflow",
        "tests/execution/test_executor.py::"
        "test_preemption_survives_store_failover_and_lease_takeover",
        "tests/processor/test_processor_leadership.py::"
        "test_processor_retries_fresh_5xx_but_bounds_old_requests",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_is_idempotent_and_rechecks_clients",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    workloads = regional.business_workloads(settings.node)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **regional.evidence_identity(),
    )
    rds = rds_snapshot(settings)
    # The plan phase already paid for the focused pytest run; --execute reuses
    # its recorded result when the source digest still matches.
    reused = reusable_focused_tests(case_dir / "plan.json")
    tests = (
        {**reused, "reused_from_plan": True}
        if reused is not None
        else focused_tests(case_dir)
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("DESTR-008 predecessor evidence is not PASS")
    if node["ready"] != "True" or not node["unschedulable"]:
        errors.append("target node is not Ready and pre-cordoned")
    if node["ownership_annotations"]:
        errors.append("target node has pre-existing gpu-fault ownership")
    if workloads:
        errors.append("target node has non-system running Pods")
    agent = state.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    reset = reset_case.capability(state.get("profile"), "gpuReset")
    if reset is None or (
        reset.get("mode") != "OWN"
        or reset.get("owner") != "gpu-fault-node-agent"
        or reset.get("adapter") != "node-action"
    ):
        errors.append("gpuReset is not OWN by the Node Agent")
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if rds.get("status") != "available":
        errors.append("Aurora cluster is not available")
    if not rds.get("writer") or len(rds.get("members") or []) < 2:
        errors.append("Aurora cluster has no failover-capable reader")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "store": state,
        "business_workloads": workloads,
        "rds": rds,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def wait_reset_claim(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    marker: str,
    observed_after: datetime,
    timeout_seconds: int = 120,
    evidence: WaitingEvidence | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = regional.store_snapshot(
            node=settings.node,
            marker=marker,
            observed_after=observed_after,
            queue_attempts=1,
        )
        if evidence is not None:
            evidence.observe(last)
        commands = [
            item
            for item in last.get("commands") or []
            if item.get("step", {}).get("operation") == "RESET_GPU"
        ]
        # The reset is in flight on the node from the first claim until the
        # Node Agent's result lands. The remote command is LEASED only for the
        # claim round trip: the executor hands the action to the Node Agent,
        # which answers "pending", and the executor reports WAITING and lets
        # the command be re-claimed on its next cycle. On 2026-09-05 a 45s
        # RESET_GPU never showed LEASED to a sampler that needs ~6s per store
        # read, so the window has to be recognised by either non-terminal
        # state. What must still be true afterwards -- one physical reset,
        # one SUCCEEDED execution, one terminal status, the same command
        # identity -- is judged unchanged below.
        if len(commands) == 1 and commands[0].get("status") in {
            "LEASED",
            "WAITING",
        }:
            return last, cast(dict[str, Any], commands[0])
        if commands and commands[0].get("status") in {
            "SUCCEEDED",
            "FAILED",
        }:
            raise RegionalFixtureError(
                "RESET_GPU completed before the Aurora failover window was captured"
            )
        time.sleep(0.25)
    raise RegionalFixtureError(f"RESET_GPU was not observed in flight: {last}")


def control_logs(
    regional: RegionalLiveFixture,
    started_at: datetime,
) -> dict[str, Any]:
    entries = []
    for app in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-processor",
    ):
        for pod in regional.ready_pods("cpu", app):
            output = regional.kubectl(
                "cpu",
                "logs",
                str(pod["name"]),
                "--since-time",
                started_at.isoformat(),
                check=False,
                timeout=120,
            )
            relevant = [
                line[:500]
                for line in output.splitlines()
                if any(
                    token in line.lower()
                    for token in (
                        "operationalerror",
                        "read-only transaction",
                        "unexpected eof",
                        "connection",
                    )
                )
            ]
            entries.append(
                {
                    "app": app,
                    "pod": pod["name"],
                    "sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "relevant_lines": relevant[-50:],
                }
            )
    return {"entries": entries}


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    details: dict[str, Any] = {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "rds_cluster_id": settings.rds_cluster_id,
        "mutation": (
            "execute one real GPU reset and call RDS failover-db-cluster while "
            "the RESET_GPU remote command is observed in flight (LEASED, or "
            "WAITING on the Node Agent) and before it reaches a terminal state"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "agent_generation": (state.get("agent") or {}).get("generation"),
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "rds_writer": preflight["rds"]["writer"],
        },
        "stop_conditions": [
            "DESTR-008 predecessor evidence is not PASS",
            "target node is not Ready, cordoned and idle",
            "Aurora has no available reader or writer identity drifts",
            "RESET_GPU is not observed in flight (LEASED or WAITING) before it "
            "reaches a terminal state",
            "Aurora failover does not change writer and return available",
            "workflow becomes BLOCKED or reset ledger/journal count differs from one",
            "cleanup cannot restore quiesce state or node ownership",
        ],
        "inconclusive_conditions": [
            "no store sample shows the writer changing (or the cluster leaving "
            "'available') while RESET_GPU is still LEASED/WAITING/PENDING",
        ],
        "rollback": {
            "Aurora failover is allowed to complete forward": True,
            "quiesce_fail_safe_timer_is_independent": True,
            "runner_finally_restores_GPU_services": True,
            "runner_finally_uses_validation_first_node_restore_if_needed": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": (preflight["store"].get("agent") or {}).get("generation"),
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
        "rds_writer": preflight["rds"]["writer"],
    }
    if current != planned:
        raise RegionalFixtureError(f"HA-003 plan drifted: {planned} != {current}")


def evaluate_reset(
    *,
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    preflight: dict[str, Any],
    state: dict[str, Any],
    command: dict[str, Any],
    baseline_host: dict[str, Any],
    target_bdf: str,
    run_id: str,
    injected_at: datetime,
    failover_requested_at: datetime,
    case_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors = reset_case.workflow_errors(state)
    workflow = state.get("workflow") or {}
    reset_successes = [
        item
        for item in workflow.get("step_executions", [])
        if item.get("operation") == "RESET_GPU" and item.get("status") == "SUCCEEDED"
    ]
    if len(reset_successes) != 1:
        errors.append("RESET_GPU does not have exactly one SUCCEEDED execution")
    reset_commands = [
        item
        for item in state.get("commands") or []
        if item.get("step", {}).get("operation") == "RESET_GPU"
    ]
    if len(reset_commands) != 1 or reset_commands[0].get("status") != "SUCCEEDED":
        errors.append("RESET_GPU remote command is not uniquely terminal")
    if reset_commands and reset_commands[0].get("command_id") != command.get(
        "command_id"
    ):
        errors.append("RESET_GPU remote command identity changed")
    after = host.execute(
        "snapshot",
        "--since-epoch",
        str(injected_at.timestamp()),
        "--pci-bdf",
        target_bdf,
        "--run-id",
        run_id,
        timeout=180,
    )
    write_json_atomic(case_dir / "host-after.json", after)
    errors.extend(
        reset_case.host_errors(
            baseline_host,
            after,
            expected_gpu_count=len(baseline_host["gpu_inventory"]),
            target_bdf=target_bdf,
        )
    )
    logs = control_logs(regional, failover_requested_at)
    write_json_atomic(case_dir / "control-logs.json", logs)
    executor = executor_logs(regional, failover_requested_at)
    write_json_atomic(case_dir / "executor-logs.json", executor)
    if executor["forbidden_result_codes"]:
        errors.append(
            "executor result submissions were not only 409/200: "
            f"{executor['forbidden_result_codes']}"
        )
    if workflow.get("status") == "BLOCKED" or workflow.get("blocked_reasons"):
        errors.append("Aurora failover left the workflow BLOCKED")
    final_node = regional.node_snapshot(settings.node)
    if final_node["ready"] != "True":
        errors.append("target node is not Ready after reset")
    if final_node["ownership_annotations"]:
        errors.append("target node retained workflow ownership")
    if final_node["unschedulable"] != preflight["node"]["unschedulable"]:
        errors.append("target node cordon state differs from baseline")
    # A negative CloudTrail claim cannot be proven inside the 15-minute
    # delivery window; poll for what is visible and label the rest provisional.
    window_end = datetime.now(timezone.utc)
    provider = regional.wait_provider_events(
        injected_at,
        event_names=set(PROVIDER_MUTATIONS),
        expected_count=0,
        ended_at=window_end,
        timeout_seconds=0,
    )
    if provider:
        errors.append("provider node mutation appeared during HA-003")
    if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    return errors, {
        "workflow": workflow,
        "provider_events": provider,
        "provider_events_provisional": regional.provider_events_provisional(window_end),
        "control_logs": logs,
        "executor_logs": executor,
        "host_after": after,
        "final_node": final_node,
    }


def verdict_for(errors: list[str], *, overlap_observed: bool) -> str:
    """FAIL on any error; PASS only if the failover provably overlapped the reset.

    A clean run whose failover landed after the command was already terminal
    is INCONCLUSIVE: nothing went wrong, but the property the case exists to
    test was not exercised.
    """

    if errors:
        return "FAIL"
    return "PASS" if overlap_observed else "INCONCLUSIVE"


def cleanup_case(
    *,
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    preflight: dict[str, Any],
    incident_id: str,
    run_id: str,
    sampler_started: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    if incident_id:
        try:
            result["quiesce_recovery"] = host.execute(
                "restore-quiesce",
                "--incident-id",
                incident_id,
                timeout=300,
            )
        except Exception as exc:
            result["errors"].append(f"quiesce recovery: {type(exc).__name__}: {exc}")
    if sampler_started:
        try:
            result["sampler_final"] = host.execute(
                "stop-reset-sampler",
                "--run-id",
                run_id,
                timeout=60,
            )
        except Exception as exc:
            result["errors"].append(f"sampler cleanup: {type(exc).__name__}: {exc}")
    try:
        residuals = host.cleanup()
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["errors"].append("host probe resources remain")
    except Exception as exc:
        result["errors"].append(f"probe cleanup: {type(exc).__name__}: {exc}")
    if incident_id:
        try:
            node = regional.node_snapshot(settings.node)
            if node["ownership_annotations"]:
                restore = WarmSpareLiveFixture(regional, "")
                created = restore.create_restore_workflow(
                    incident_id=incident_id,
                    node=settings.node,
                    profile_version=str(
                        (preflight["store"].get("profile") or {}).get("profile_version")
                        or ""
                    ),
                    reason="HA-003 validated cleanup",
                )
                result["node_restore"] = restore.wait_workflow_id(
                    str(created["workflow_request_id"])
                )
                if result["node_restore"].get("status") != "SUCCEEDED":
                    result["errors"].append("node restore workflow failed")
        except Exception as exc:
            result["errors"].append(f"node restore: {type(exc).__name__}: {exc}")
    return result


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    regional = RegionalLiveFixture(settings.regional)
    run_id = f"ha003-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"ha003-{int(time.time())}-a{attempt}"
    host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=reset_case.PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    incident_id = ""
    sampler_started = False
    try:
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA clients")
        target_bdf = str(baseline_host["gpu_inventory"][0]["pci_bdf"])
        host.execute(
            "start-reset-sampler",
            "--run-id",
            run_id,
            "--probe-script",
            host.host_script,
            timeout=60,
        )
        sampler_started = True
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before injection")
        injected_at = datetime.now(timezone.utc)
        injection = host.execute(
            "write-xid46",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            target_bdf,
        )
        write_json_atomic(case_dir / "injection.json", injection)
        # Every store read from injection onwards feeds the WAITING evidence:
        # the steps before and during the failover finish long before
        # `wait_for_workflow` starts polling, and their WAITING records are
        # gone by then.
        waiting_evidence = WaitingEvidence()
        claimed_state, command = wait_reset_claim(
            regional,
            settings,
            marker=marker,
            observed_after=injected_at,
            evidence=waiting_evidence,
        )
        write_json_atomic(case_dir / "reset-claimed.json", claimed_state)
        failover_requested_at = datetime.now(timezone.utc)
        failover = aws_rds(
            settings,
            "failover-db-cluster",
            "--db-cluster-identifier",
            settings.rds_cluster_id,
        )
        write_json_atomic(case_dir / "failover-request.json", failover)

        def observe() -> dict[str, Any]:
            sample = regional.store_snapshot(
                node=settings.node,
                marker=marker,
                observed_after=injected_at,
                queue_attempts=1,
            )
            waiting_evidence.observe(sample)
            return sample

        previous_writer = str(preflight["rds"]["writer"])
        rds_after, failover_samples = wait_rds_failover(
            settings,
            previous_writer=previous_writer,
            observe=observe,
        )
        write_json_atomic(case_dir / "rds-after.json", rds_after)
        overlap = failover_overlap_observed(
            failover_samples, previous_writer=previous_writer
        )
        write_json_atomic(
            case_dir / "failover-window.json",
            {"samples": failover_samples, "overlap_observed": overlap},
        )
        state = waiting_evidence.merged_into(
            regional.wait_for_workflow(
                node=settings.node,
                marker=marker,
                observed_after=injected_at,
                case_dir=case_dir,
                timeout_seconds=1800,
            )
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        errors, evidence = evaluate_reset(
            settings=settings,
            regional=regional,
            host=host,
            preflight=preflight,
            state=state,
            command=command,
            baseline_host=baseline_host,
            target_bdf=target_bdf,
            run_id=run_id,
            injected_at=injected_at,
            failover_requested_at=failover_requested_at,
            case_dir=case_dir,
        )
        result.update(
            {
                "verdict": verdict_for(errors, overlap_observed=overlap),
                "errors": errors,
                "failover_overlap_observed": overlap,
                "failover_samples": failover_samples,
                "marker": marker,
                "incident_id": incident_id,
                "workflow_request_id": evidence["workflow"].get("request_id"),
                "reset_command_id": command.get("command_id"),
                "rds_before": preflight["rds"],
                "rds_after": rds_after,
                "provider_events": evidence["provider_events"],
                "provider_events_provisional": evidence["provider_events_provisional"],
                "executor_result_codes": evidence["executor_logs"][
                    "rejected_result_codes"
                ],
            }
        )
        if not overlap:
            result["inconclusive_reason"] = (
                "no store sample showed Aurora mid-failover while RESET_GPU was "
                "still open; the reset and the failover did not provably overlap"
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = cleanup_case(
            settings=settings,
            regional=regional,
            host=host,
            preflight=preflight,
            incident_id=incident_id,
            run_id=run_id,
            sampler_started=sampler_started,
        )
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    result.update(regional.evidence_identity())
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run HA-003 Aurora failover during one real GPU reset."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--rds-cluster-id", default="")
    value.add_argument("--predecessor-evidence", default="")
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
