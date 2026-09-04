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
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    TRAINING_IMAGE,
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
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

DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-009"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR009_RESTART_24GPU_WORKLOAD"
INSTANCE_TYPE_GPU_PRODUCTS = {
    "p5.4xlarge": "H100",
    "p5.48xlarge": "H100",
    "p5e.48xlarge": "H200",
    "p5en.48xlarge": "H200",
    "p6-b200.48xlarge": "B200",
    "p6-b300.48xlarge": "B300",
}
CORE_OPERATIONS = {
    "FREEZE_EVIDENCE",
    "STOP_WORKLOADS",
    "RESTART_WORKLOAD",
}
FORBIDDEN_OPERATIONS = {
    "QUIESCE_GPU_SERVICES",
    "RESET_GPU",
    "RESTART_NODE",
    "REPLACE_NODE",
}
CLEANUP_TERMINAL_WORKFLOW_STATUSES = {"SUCCEEDED", "FAILED"}
CLEANUP_TERMINAL_COMMAND_STATUSES = {"SUCCEEDED", "FAILED"}
CLEANUP_QUIET_SECONDS = 15
CLEANUP_TIMEOUT_SECONDS = 300
CLEANUP_POLL_SECONDS = 5


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    job_id: str
    attempt_id: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr009-{suffix}"
    return job_id, f"{job_id}-a001"


def configure(arguments: argparse.Namespace) -> Settings:
    job_id, attempt_id = derived_identity(arguments.run_dir, arguments.attempt)
    configured_job = arguments.job_id.strip() or job_id
    configured_attempt = arguments.attempt_id.strip() or (f"{configured_job}-a001")
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
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        manifest=Path(arguments.manifest).expanduser().resolve(),
        job_id=configured_job,
        attempt_id=configured_attempt,
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/regional/test_regional_control_plane.py::"
        "test_remote_command_lease_and_result_advance_adapter",
        "tests/regional/test_regional_control_plane.py::"
        "test_regional_api_sends_executor_workload_restart_notification",
        "tests/execution/test_restart_safety.py::"
        "test_restart_budget_blocks_second_restart_for_same_job",
        "tests/execution/_node_action_cases_1.py::"
        "test_restart_workload_remote_waiting_is_not_preempted",
        "tests/regional/test_production_safety_config.py::"
        "test_regional_restart_covers_every_running_role_in_order",
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


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def managed_owner_errors(profile: dict[str, Any] | None) -> list[str]:
    errors = []
    for name in ("workloadStop", "workloadRestart"):
        item = capability(profile, name)
        if item is None:
            errors.append(f"runtime profile has no {name} capability")
        elif (
            item.get("mode") != "OWN"
            or item.get("owner") != "gpu-fault-kubernetes-adapter"
            or item.get("adapter") != "regional-cluster-executor"
        ):
            errors.append(f"{name} is not OWN by the Kubernetes adapter")
    return errors


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    if not settings.site_file.is_file():
        raise RegionalFixtureError("regional site file does not exist")
    if not settings.manifest.is_file():
        raise RegionalFixtureError("training manifest does not exist")
    fixture = RegionalLiveFixture(settings.regional)
    gpu_nodes = fixture.gpu_nodes()
    candidates = [
        item
        for item in gpu_nodes
        if item["ready"] == "True" and not item["unschedulable"] and not item["taints"]
    ]
    state = (
        fixture.store_snapshot(node=str(candidates[0]["name"])) if candidates else {}
    )
    runtime_identity = fixture.runtime_identity()
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    errors = []
    errors.extend(runtime_identity_errors(runtime_identity))
    if not predecessor["valid"]:
        errors.append("DESTR-002 predecessor evidence is not PASS")
    if len(candidates) < 3:
        errors.append("fewer than three Ready schedulable untainted GPU nodes")
    gpu_workloads = fixture.gpu_workloads()
    if gpu_workloads:
        errors.append("GPU cluster already has active or pending GPU workloads")
    errors.extend(managed_owner_errors(state.get("profile")))
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if int((state.get("queue") or {}).get("depth") or 0):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    manifest_text = settings.manifest.read_text(encoding="utf-8")
    if TRAINING_IMAGE not in manifest_text:
        errors.append("training manifest image digest differs from the contract")
    result = {
        "release_id": state.get("release_id"),
        "gpu_nodes": gpu_nodes,
        "candidate_nodes": candidates,
        "gpu_workloads": gpu_workloads,
        "store": state,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def observation_gpu_uuids(observation: dict[str, Any]) -> set[str]:
    return {
        str(gpu_uuid)
        for container in observation.get("containers", [])
        for gpu_uuid in container.get("gpu_uuids", [])
    }


def observation_gpu_count(observation: dict[str, Any]) -> int:
    declared = sum(
        int(container.get("gpu_count") or 0)
        for container in observation.get("containers", [])
    )
    return declared or len(observation_gpu_uuids(observation))


def wait_observation(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    node: str,
    expected_gpu_count: int = 24,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        state = regional.store_snapshot(
            node=node,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
        )
        last = state.get("observations") or []
        if len(last) == 1:
            observation = last[0]
            if (
                observation.get("workload_phase") == "RUNNING"
                and observation_gpu_count(observation) == expected_gpu_count
                and len(observation_gpu_uuids(observation)) == expected_gpu_count
            ):
                return observation
        time.sleep(5)
    raise RegionalFixtureError(
        f"{expected_gpu_count}-GPU ACTIVE attempt observation did not appear: {last}"
    )


def normalize_product(value: str | None) -> str:
    match = re.search(
        r"(?:NVIDIA[-_ ]*)?(H200|H100|B300|B200|B100|A100|A800|H800)",
        str(value or ""),
        re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).upper()
    instance_type = str(value or "").strip().lower().removeprefix("ml.")
    product = INSTANCE_TYPE_GPU_PRODUCTS.get(instance_type)
    if product is not None:
        return product
    raise RegionalFixtureError(f"cannot normalize GPU product: {value!r}")


def xid11_payload(
    settings: Settings,
    *,
    case_id: str = CASE_ID,
    marker: str,
    node: str,
    product: str,
    observation: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "cluster_id": settings.regional.cluster_id,
        "node_id": node,
        "record_id": marker,
        "observed_at": observed_at.isoformat(),
        "message": (f"NVRM: Xid (PCI:0000:b9:00): 11, Ch 00000001, marker={marker}"),
        "runtime_profile_version": observation["runtime_profile_version"],
        "product": product,
        "workload_state": "ACTIVE",
        "affected_workload_ids": observation.get("workload_ids") or [],
        "evidence_ref": f"api-replay://{case_id}/{marker}",
    }


def control_plane_log_snapshot(
    regional: RegionalLiveFixture,
    since: datetime,
    workload_name: str,
) -> dict[str, Any]:
    entries = []
    suspicious = []
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
                since.isoformat(),
                check=False,
                timeout=120,
            )
            path_key = f"{app}/{pod['name']}"
            entries.append(
                {
                    "pod": path_key,
                    "line_count": len(output.splitlines()),
                    "sha256": hashlib.sha256(output.encode()).hexdigest(),
                }
            )
            for line in output.splitlines():
                lowered = line.lower()
                if workload_name.lower() in lowered and any(
                    token in lowered
                    for token in (
                        " patch ",
                        " delete ",
                        " create ",
                        " suspend",
                        "kubernetes write",
                    )
                ):
                    suspicious.append({"pod": path_key, "line": line[:500]})
    return {"entries": entries, "suspicious": suspicious}


def remote_waiting_evidence(state: dict[str, Any], operation: str) -> bool:
    workflow = state.get("workflow") or {}
    executions = [
        *(workflow.get("step_executions") or []),
        *(state.get("observed_waiting_step_executions") or []),
    ]
    return any(
        item.get("operation") == operation
        and item.get("status") == "WAITING"
        and (item.get("details") or {}).get("mutation_submitted_by_control_plane")
        is False
        for item in executions
    )


def remote_execution_evidence(state: dict[str, Any], operation: str) -> bool:
    if remote_waiting_evidence(state, operation):
        return True
    workflow = state.get("workflow") or {}
    terminal = terminal_step(workflow, operation)
    command_succeeded = any(
        (item.get("step") or {}).get("operation") == operation
        and item.get("status") == "SUCCEEDED"
        for item in state.get("commands") or []
    )
    return bool(
        terminal
        and terminal.get("status") == "SUCCEEDED"
        and str(terminal.get("adapter_operation_id") or "").startswith("remote/")
        and command_succeeded
    )


def terminal_step(
    workflow: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in workflow.get("step_executions", [])
        if item.get("operation") == operation
        and item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    return matches[-1] if matches else None


def workflow_errors(
    state: dict[str, Any],
    *,
    expected_gpu_count: int = 24,
) -> list[str]:
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    if event.get("xid") != 11:
        errors.append("matched event is not XID 11")
    if decision.get("official_action") != "RESTART_APP":
        errors.append("policy did not resolve XID 11 to RESTART_APP")
    if workflow.get("status") != "SUCCEEDED":
        errors.append("workload restart workflow is not SUCCEEDED")
    operations = [item.get("operation") for item in workflow.get("official_steps", [])]
    for operation in CORE_OPERATIONS:
        if operations.count(operation) != 1:
            errors.append(f"workflow does not contain exactly one {operation}")
    if FORBIDDEN_OPERATIONS.intersection(operations):
        errors.append("workload restart workflow contains a node mutation")
    for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD"):
        if not remote_execution_evidence(state, operation):
            errors.append(f"{operation} lacks remote execution evidence")
        terminal = terminal_step(workflow, operation)
        if terminal is None or terminal.get("status") != "SUCCEEDED":
            errors.append(f"{operation} did not reach SUCCEEDED")
        elif not str(terminal.get("adapter_operation_id") or "").startswith("remote/"):
            errors.append(f"{operation} adapter_operation_id is not remote/")
    restart = terminal_step(workflow, "RESTART_WORKLOAD") or {}
    details = restart.get("details") or {}
    notification_context = details.get("notification_context") or {}
    source_gpu_count = details.get(
        "source_gpu_count",
        notification_context.get("source_gpu_count"),
    )
    target_gpu_count = details.get(
        "target_gpu_count",
        notification_context.get("target_gpu_count"),
    )
    restart_count = (state.get("restart_budget") or {}).get(
        "restart_count",
        details.get("restart_count", notification_context.get("restart_count")),
    )
    if source_gpu_count != expected_gpu_count:
        errors.append(f"RESTART_WORKLOAD source GPU count is not {expected_gpu_count}")
    if target_gpu_count != expected_gpu_count:
        errors.append(f"RESTART_WORKLOAD target GPU count is not {expected_gpu_count}")
    if restart_count != 1:
        errors.append("RESTART_WORKLOAD restart count is not one")
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
        "risk": "live-workload-restart",
        "predecessor": preflight["predecessor"],
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "site_file": str(settings.site_file),
        "candidate_nodes": [
            {"name": item["name"], "uid": item["uid"]}
            for item in preflight["candidate_nodes"]
        ],
        "mutation": (
            "prewarm the pinned training image, submit one managed three-node "
            "24-GPU PyTorchJob, replay XID 11 through the collector API, and "
            "allow the GPU-side executor to stop and restart it once"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "candidate_node_uids": sorted(
                str(item["uid"]) for item in preflight["candidate_nodes"]
            ),
            "runtime_identity": preflight["runtime_identity"],
        },
        "stop_conditions": [
            "DESTR-002 has not passed in formal sequence",
            "preflight or focused regression failure",
            "fewer than three idle Ready GPU nodes",
            "Profile, node inventory or release drift",
            "image prewarm, PyTorchJob scheduling or NCCL heartbeat failure",
            "attempt observation lacks 24 unique GPU UUIDs",
            "processor receipt is not HTTP 200",
            "workflow differs from the three-step workload contract",
            "restart budget, Pod replacement or target GPU count is incorrect",
            "cleanup cannot prove workflow and remote-command quiescence",
            "any node/provider/control-plane Kubernetes mutation appears",
        ],
        "rollback": {
            "runner_waits_for_async_quiescence_before_workload_delete": True,
            "runner_deletes_test_PyTorchJob_only_after_quiescence": True,
            "runner_finally_deletes_image_prewarm_Pods": True,
            "no_node_reboot_reset_or_replacement_is_authorized": True,
            "failed_workload_cleanup_is_reported_as_a_blocker": True,
        },
        "preflight": preflight,
    }


def cleanup_quiescence_summary(state: dict[str, Any]) -> dict[str, Any]:
    workflow = state.get("workflow") or {}
    commands = state.get("commands") or []
    observations = state.get("observations") or []
    observation_phases = sorted(
        {
            str(item.get("workload_phase") or "")
            for item in observations
            if isinstance(item, dict)
        }
    )
    return {
        "event_observed": bool(state.get("event")),
        "workflow_request_id": workflow.get("request_id"),
        "workflow_status": workflow.get("status"),
        "observation_phases": observation_phases,
        "observation_terminal": bool(
            len(observations) == 1
            and observation_phases
            and observation_phases[0] in {"SUCCEEDED", "FAILED", "STOPPED"}
        ),
        "commands": [
            {
                "operation": (item.get("step") or {}).get("operation"),
                "status": item.get("status"),
                "status_source": item.get("status_source"),
                "updated_at": item.get("updated_at"),
            }
            for item in commands
        ],
    }


def wait_for_cleanup_quiescence(
    *,
    regional: RegionalLiveFixture,
    node: str,
    marker: str,
    observed_after: datetime,
    job_id: str,
    attempt_id: str,
    case_dir: Path,
    timeout_seconds: int = CLEANUP_TIMEOUT_SECONDS,
    quiet_seconds: int = CLEANUP_QUIET_SECONDS,
    poll_seconds: int = CLEANUP_POLL_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    stable_since: float | None = None
    stable_signature = ""
    entries: list[dict[str, Any]] = []
    last_summary: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = regional.store_snapshot(
            node=node,
            marker=marker,
            observed_after=observed_after,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        summary = cleanup_quiescence_summary(state)
        last_summary = summary
        workflow_terminal = (
            summary["workflow_status"] in CLEANUP_TERMINAL_WORKFLOW_STATUSES
        )
        commands = summary["commands"]
        commands_terminal = all(
            item.get("status") in CLEANUP_TERMINAL_COMMAND_STATUSES for item in commands
        )
        quiescent = bool(
            summary["event_observed"]
            and workflow_terminal
            and commands_terminal
            and summary["observation_terminal"]
        )
        signature = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        now = time.monotonic()
        if quiescent and signature == stable_signature:
            stable_since = stable_since if stable_since is not None else now
        elif quiescent:
            stable_signature = signature
            stable_since = now
        else:
            stable_signature = ""
            stable_since = None
        quiet_elapsed = (
            max(0.0, now - stable_since) if stable_since is not None else 0.0
        )
        entries.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "quiescent": quiescent,
                "quiet_elapsed_seconds": round(quiet_elapsed, 3),
                **summary,
            }
        )
        report = {
            "safe_to_delete": bool(quiescent and quiet_elapsed >= quiet_seconds),
            "quiet_seconds": quiet_seconds,
            "entries": entries,
        }
        write_json_atomic(case_dir / "cleanup-quiescence.json", report)
        if report["safe_to_delete"]:
            return report
        time.sleep(poll_seconds)
    raise RegionalFixtureError(
        "cleanup could not prove workflow and remote-command quiescence: "
        + json.dumps(last_summary, sort_keys=True)
    )


def cleanup_case(
    *,
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    prewarm: ImagePrewarmFixture,
    case_dir: Path,
    runtime_identity: dict[str, Any],
    result: dict[str, Any],
    node: str | None,
    marker: str | None,
    observed_after: datetime | None,
    job_id: str,
    attempt_id: str,
) -> None:
    cleanup_deferred = False
    if node and marker and observed_after is not None:
        try:
            result["cleanup_quiescence"] = wait_for_cleanup_quiescence(
                regional=regional,
                node=node,
                marker=marker,
                observed_after=observed_after,
                job_id=job_id,
                attempt_id=attempt_id,
                case_dir=case_dir,
            )
        except Exception as exc:
            cleanup_deferred = True
            result["cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["workload_cleanup_deferred"] = True
            result["verdict"] = "FAIL"
    try:
        if cleanup_deferred:
            workload_residual = True
        else:
            workload.delete()
            workload_residual = bool(
                regional.kubectl(
                    "gpu",
                    "get",
                    workload.resource,
                    workload.name,
                    "--ignore-not-found",
                    "-o",
                    "name",
                    check=False,
                ).strip()
            )
    except Exception as exc:
        workload_residual = True
        result["workload_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    result["workload_residual"] = workload_residual
    if workload_residual:
        result["verdict"] = "FAIL"
    try:
        prewarm_residuals = prewarm.cleanup()
    except Exception as exc:
        prewarm_residuals = {"cleanup_error": True}
        result["prewarm_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    result["prewarm_residuals"] = prewarm_residuals
    if any(prewarm_residuals.values()):
        result["verdict"] = "FAIL"
    try:
        regional.verify_runtime_identity(
            runtime_identity,
            evidence_path=case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-009 cleanup",
        )
    except Exception as exc:
        errors = result.setdefault("errors", [])
        if isinstance(errors, list):
            errors.append(f"{type(exc).__name__}: {exc}")
        result["verdict"] = "FAIL"


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
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
        "candidate_node_uids": sorted(
            str(item["uid"]) for item in preflight["candidate_nodes"]
        ),
        "runtime_identity": preflight["runtime_identity"],
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-009 plan drifted: {planned} != {current}")

    regional = RegionalLiveFixture(settings.regional)
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=settings.manifest,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )
    run_id = f"destr009-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=run_id,
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    injection_started: datetime | None = None
    marker: str | None = None
    target_node: str | None = None
    try:
        candidate_names = [str(item["name"]) for item in preflight["candidate_nodes"]]
        prewarm.create(candidate_names)
        cached = prewarm.cached_nodes()
        write_json_atomic(case_dir / "image-cache.json", {"cached_nodes": cached})
        if not set(candidate_names) <= set(cached):
            raise RegionalFixtureError(
                "training image is not cached on every candidate"
            )
        submission = workload.submit()
        write_json_atomic(case_dir / "submission.json", submission)
        source = workload.wait_running(timeout_seconds=900)
        write_json_atomic(case_dir / "workload-source.json", source)
        source_uids = {str(item["uid"]) for item in source["pods"]}
        target_node = str(source["pods"][0]["node"])
        observation = wait_observation(regional, settings, node=target_node)
        write_json_atomic(case_dir / "source-observation.json", observation)
        node_metadata = regional.node_metadata(target_node)
        product = normalize_product(node_metadata.get("product"))
        regional.verify_runtime_identity(
            planned["runtime_identity"],
            evidence_path=case_dir / "runtime-identity-before-xid.json",
            stage="before DESTR-009 XID replay",
        )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before XID replay"
            )
        marker = f"destr009-{int(time.time())}-a{attempt}"
        injection_started = datetime.now(timezone.utc)
        payload = xid11_payload(
            settings,
            marker=marker,
            node=target_node,
            product=product,
            observation=observation,
            observed_at=injection_started,
        )
        injection = regional.post_xid_event(payload)
        write_json_atomic(case_dir / "injection.json", injection)
        receipt = injection.get("receipt") or {}
        if receipt.get("status") != 200:
            raise RegionalFixtureError(f"processor receipt is not HTTP 200: {receipt}")
        state = regional.wait_for_workflow(
            node=target_node,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=1200,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        errors = workflow_errors(state)
        target = workload.wait_restarted(source_uids, timeout_seconds=900)
        write_json_atomic(case_dir / "workload-target.json", target)
        regional.verify_runtime_identity(
            planned["runtime_identity"],
            evidence_path=case_dir / "runtime-identity-after-restart.json",
            stage="after DESTR-009 workload restart",
        )
        target_uids = {str(item["uid"]) for item in target["pods"]}
        if not target_uids.isdisjoint(source_uids):
            errors.append("old and new PyTorchJob Pod UIDs overlap")
        if len({item["node"] for item in target["pods"]}) != 3:
            errors.append("restarted PyTorchJob is not spread across three nodes")
        budget = state.get("restart_budget") or {}
        if budget.get("restart_count") != 1 or budget.get("budget") != 1:
            errors.append("restart budget did not advance exactly once")
        provider = regional.provider_events(
            injection_started,
            datetime.now(timezone.utc),
        )
        write_json_atomic(case_dir / "provider-events.json", {"events": provider})
        if provider:
            errors.append("provider mutation appeared during workload restart")
        node_states = [
            regional.node_snapshot(str(item["name"]))
            for item in preflight["candidate_nodes"]
        ]
        write_json_atomic(case_dir / "gpu-node-postflight.json", {"nodes": node_states})
        if any(
            item["ownership_annotations"]
            or any(
                taint.get("key") == "gpu-fault.io/quarantined"
                for taint in item["taints"]
            )
            for item in node_states
        ):
            errors.append("GPU node quarantine ownership appeared")
        logs = control_plane_log_snapshot(
            regional,
            injection_started,
            workload.name,
        )
        write_json_atomic(case_dir / "control-plane-logs.json", logs)
        if logs["suspicious"]:
            errors.append("control-plane logs show a Kubernetes workload write")
        cpu_after = regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "target_node": target_node,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(target_uids),
                "restart_budget": budget,
                "provider_events": provider,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_case(
            regional=regional,
            workload=workload,
            prewarm=prewarm,
            case_dir=case_dir,
            runtime_identity=planned["runtime_identity"],
            result=result,
            node=target_node,
            marker=marker,
            observed_after=injection_started,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
        )
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-009 workload restart acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
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
    return execute_case(
        settings,
        arguments.run_dir,
        arguments.attempt,
        deadline,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
