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
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
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
CLEANUP_POLL_SECONDS = 5
# The completion watcher keeps republishing a vanished attempt as RUNNING for
# GPU_FAULT_COMPLETION_ATTEMPT_MISSING_GRACE_SECONDS (default 300, not
# overridden by the release) before it tombstones the attempt STOPPED, so the
# quiescence gate's observation cannot turn terminal earlier than that after
# STOP_WORKLOADS. DESTR-012 group D (2026-09-14) reached the gate 20 s after
# the stop with a 300 s budget and missed the tombstone by seconds, deferring
# the delete and leaving the restarted Job running. The budget must outlive
# the grace plus the quiet window and the polling slack; the loop returns as
# soon as the gate holds, so a large budget costs nothing when it does.
ATTEMPT_MISSING_GRACE_SECONDS = 300
CLEANUP_TIMEOUT_SECONDS = ATTEMPT_MISSING_GRACE_SECONDS + 300
CONTROL_PLANE_LOG_APPS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-processor",
)
# A Kubernetes write against the workload, in the plain-text form the control
# plane logs (`LOG_FORMAT` in gpu_fault.logging_setup is `%(asctime)s
# %(levelname)s %(name)s %(message)s`, not JSON) and in the structured form a
# client library or a future JSON handler would emit.
WORKLOAD_WRITE_LOG_TOKENS = (
    " patch ",
    " delete ",
    " create ",
    " suspend",
    "kubernetes write",
    '"verb":"patch"',
    '"verb": "patch"',
    '"verb":"delete"',
    '"verb": "delete"',
    '"verb":"create"',
    '"verb": "create"',
    '"verb":"update"',
    '"verb": "update"',
)


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


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``.

    ``reuse`` consults ``reusable_focused_tests`` on the plan this case wrote:
    a passing result taken against the same source digest speaks for the tree
    now, and the ~minute the suite costs is not paid twice per run.
    """

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
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
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    if not settings.site_file.is_file():
        raise RegionalFixtureError("regional site file does not exist")
    if not settings.manifest.is_file():
        raise RegionalFixtureError("training manifest does not exist")
    fixture = RegionalLiveFixture(settings.regional)
    identity = fixture.evidence_identity()
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
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
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
    if processor_queue_backlog(state.get("queue") or {}):
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
        "evidence_identity": identity,
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
            queue_attempts=1,
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


def workload_write_lines(output: str, workload_name: str) -> list[str]:
    """Log lines that name ``workload_name`` next to a Kubernetes write verb."""

    needle = workload_name.lower()
    return [
        line[:500]
        for line in output.splitlines()
        if needle in line.lower()
        and any(token in line.lower() for token in WORKLOAD_WRITE_LOG_TOKENS)
    ]


def silence_evidence(
    regional: RegionalLiveFixture,
    *,
    plane: str,
    pod: str,
    since: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    """Prove that a Pod's empty window read means "nothing logged", or say why not.

    The ingress tier runs with ``--no-access-log`` and logs only on events, so
    an idle window legitimately yields zero lines (DESTR-009 attempt 1,
    2026-09-14: all three api-ha Pods, INCONCLUSIVE). Zero lines is trusted only
    when the container has been running since before the window (no restart
    could have dropped lines) and its log stream has history (the read path
    works); otherwise the Pod stays inconclusive with the reason recorded.
    """

    try:
        described = json.loads(
            regional.kubectl(plane, "get", "pod", pod, "-o", "json", timeout=60)
        )
    except (RegionalFixtureError, ValueError) as exc:
        return None, f"pod could not be described: {exc}"[:300]
    statuses = (described.get("status") or {}).get("containerStatuses") or []
    if not statuses:
        return None, "pod reports no container status"
    running = ((statuses[0].get("state") or {}).get("running") or {}).get("startedAt")
    if not running:
        return None, "container is not running"
    started_at = datetime.fromisoformat(str(running).replace("Z", "+00:00"))
    if started_at > since:
        return None, (
            f"container started at {started_at.isoformat()} inside the window; "
            "lines before the restart are gone"
        )
    try:
        history = regional.kubectl(plane, "logs", pod, "--tail=1", timeout=60)
    except RegionalFixtureError as exc:
        return None, f"log stream could not be read: {exc}"[:300]
    if not history.strip():
        return None, "log stream has no history at all"
    return {
        "container_started_at": started_at.isoformat(),
        "restart_count": statuses[0].get("restartCount"),
        "history_tail_lines": len(history.splitlines()),
    }, None


def log_write_snapshot(
    regional: RegionalLiveFixture,
    *,
    plane: str,
    apps: tuple[str, ...],
    since: datetime,
    workload_name: str,
) -> dict[str, Any]:
    """Grep the Pods of ``apps`` for a write against the workload.

    A Pod whose window read failed, or returned no lines without the proof in
    ``silence_evidence``, has not been checked: it is listed under
    ``inconclusive`` and the verdict is INCONCLUSIVE, not CLEAN. A Pod that
    returned no lines *with* that proof is ``silent`` -- checked, and clean.
    The old shape counted every empty read as "no suspicious lines".
    """

    entries: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    inconclusive: list[str] = []
    silent: list[str] = []
    for app in apps:
        for pod in regional.ready_pods(plane, app):
            name = str(pod["name"])
            path_key = f"{app}/{name}"
            try:
                output = regional.kubectl(
                    plane,
                    "logs",
                    name,
                    "--since-time",
                    since.isoformat(),
                    timeout=120,
                )
            except RegionalFixtureError as exc:
                entries.append(
                    {
                        "pod": path_key,
                        "line_count": 0,
                        "sha256": None,
                        "classification": "inconclusive",
                        "reason": f"window read failed: {exc}"[:300],
                    }
                )
                inconclusive.append(path_key)
                continue
            line_count = len(output.splitlines())
            entry: dict[str, Any] = {
                "pod": path_key,
                "line_count": line_count,
                "sha256": hashlib.sha256(output.encode()).hexdigest(),
                "classification": "checked",
            }
            if line_count == 0:
                evidence, reason = silence_evidence(
                    regional, plane=plane, pod=name, since=since
                )
                if evidence is None:
                    entry["classification"] = "inconclusive"
                    entry["reason"] = reason
                    inconclusive.append(path_key)
                else:
                    entry["classification"] = "silent"
                    entry["silence_evidence"] = evidence
                    silent.append(path_key)
            entries.append(entry)
            suspicious.extend(
                {"pod": path_key, "line": line}
                for line in workload_write_lines(output, workload_name)
            )
    if suspicious:
        verdict = "SUSPICIOUS"
    elif inconclusive or not entries:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "CLEAN"
    return {
        "entries": entries,
        "suspicious": suspicious,
        "inconclusive": inconclusive,
        "silent": silent,
        "verdict": verdict,
    }


def log_write_errors(logs: dict[str, Any], label: str) -> list[str]:
    errors = []
    if logs.get("suspicious"):
        errors.append(f"{label} logs show a Kubernetes workload write")
    if logs.get("verdict") == "INCONCLUSIVE":
        errors.append(
            f"{label} logs are INCONCLUSIVE: no lines from "
            + (", ".join(logs.get("inconclusive") or []) or "any Pod")
        )
    return errors


def control_plane_log_snapshot(
    regional: RegionalLiveFixture,
    since: datetime,
    workload_name: str,
) -> dict[str, Any]:
    return log_write_snapshot(
        regional,
        plane="cpu",
        apps=CONTROL_PLANE_LOG_APPS,
        since=since,
        workload_name=workload_name,
    )


def workflow_official_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The compiled steps with their owners, as the case evidence records them.

    DESTR-012 group A reads this from the DESTR-009 evidence instead of
    restarting a second 24-GPU job to look at the same two owner fields.
    """

    return [
        {
            "operation": item.get("operation"),
            "execution_owner": item.get("execution_owner"),
        }
        for item in (state.get("workflow") or {}).get("official_steps") or []
        if isinstance(item, dict)
    ]


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
    xid: int = 11,
) -> list[str]:
    """DESTR-009 replays XID 11; COLLECT-016 A reaches the same RESTART_APP
    contract from a real kmsg XID 31, so the event it must name is a
    parameter."""

    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    if event.get("xid") != xid:
        errors.append(f"matched event is not XID {xid}")
    if decision.get("official_action") != "RESTART_APP":
        errors.append(f"policy did not resolve XID {xid} to RESTART_APP")
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
    details = {
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
    record_focused_tests(details, preflight["focused_tests"])
    return details


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
    workflow_request_ids: list[str] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    stable_since: float | None = None
    stable_signature = ""
    entries: list[dict[str, Any]] = []
    last_summary: dict[str, Any] = {}
    snapshot_arguments: dict[str, Any] = {}
    if workflow_request_ids:
        snapshot_arguments["workflow_request_ids"] = list(workflow_request_ids)
    while time.monotonic() < deadline:
        # A wait loop wants the cheap queue read; the drained-backlog gate is
        # for preflights (see store_snapshot).
        state = regional.store_snapshot(
            node=node,
            marker=marker,
            observed_after=observed_after,
            job_id=job_id,
            attempt_id=attempt_id,
            queue_attempts=1,
            **snapshot_arguments,
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
    workflow_request_ids: list[str] | None = None,
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
                workflow_request_ids=workflow_request_ids,
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
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
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
        **preflight["evidence_identity"],
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "focused_tests_reused": bool(
            preflight["focused_tests"].get("focused_tests_reused")
        ),
    }
    injection_started: datetime | None = None
    marker: str | None = None
    target_node: str | None = None
    workflow_request_ids: list[str] = []
    try:
        candidate_names = [str(item["name"]) for item in preflight["candidate_nodes"]]
        prewarmed = prewarm.create(candidate_names)
        cached = prewarm.cached_nodes()
        write_json_atomic(
            case_dir / "image-cache.json",
            {"cached_nodes": cached, **prewarmed},
        )
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
        workflow_request_id = (state.get("workflow") or {}).get("request_id")
        if workflow_request_id:
            workflow_request_ids.append(str(workflow_request_id))
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
        provider_window_end = datetime.now(timezone.utc)
        provider = regional.provider_events(injection_started, provider_window_end)
        # "No provider mutation" cannot be proven inside CloudTrail's delivery
        # window; an empty read is recorded as provisional, not as proof.
        provider_provisional = not provider and regional.provider_events_provisional(
            provider_window_end
        )
        write_json_atomic(
            case_dir / "provider-events.json",
            {"events": provider, "provider_events_provisional": provider_provisional},
        )
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
        errors.extend(log_write_errors(logs, "control-plane"))
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
                "workflow_request_id": workflow_request_id,
                "workflow_official_steps": workflow_official_steps(state),
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(target_uids),
                "restart_budget": budget,
                "provider_events": provider,
                "provider_events_provisional": provider_provisional,
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
            workflow_request_ids=workflow_request_ids or None,
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
