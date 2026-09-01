#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    provider_event_actor_matches_role,
    required,
    settings_from_arguments,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-002"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-001"
CONFIRMATION = "DESTR002_REBOOT_ISOLATED_NODE"
EXPECTED_ACTION = "REBOOT_NODE"
REBOOT_EVENTS = {"BatchRebootClusterNodes", "RebootClusterNodes"}
FORBIDDEN_EVENTS = {
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
}


PREFLIGHT_PROBE = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodLifecycleAdapter,
)

node = sys.argv[1]
executor = executor_from_environment()
step_adapter = next(
    item
    for item in executor.adapters
    if getattr(item, "owner", "") == "gpu-fault-hyperpod-adapter"
)
lifecycle = step_adapter.dispatcher.adapter
positive = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[node],
)
missing = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[],
)
wrong = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=["audit-other-node"],
)
disabled = HyperPodLifecycleAdapter(
    lifecycle.config.model_copy(
        update={"execution_enabled": False, "reboot_enabled": False}
    ),
    client=lifecycle.client,
)
disabled_result = disabled.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[node],
)
print(json.dumps({
    "configured_cluster_name": lifecycle.config.cluster_name,
    "positive": positive.model_dump(mode="json"),
    "missing_isolation": missing.model_dump(mode="json"),
    "wrong_isolation": wrong.model_dump(mode="json"),
    "disabled": disabled_result.model_dump(mode="json"),
}, sort_keys=True, default=str))
"""


DIRECT_DUPLICATE_REPLAY = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment
from gpu_fault.hyperpod import (
    HyperPodAction,
    hyperpod_submission_idempotency_key,
)
from gpu_fault.regional import RemoteActionCommand

command = RemoteActionCommand.model_validate(json.loads(sys.argv[1]))
executor = executor_from_environment()
step_adapter = next(
    item
    for item in executor.adapters
    if getattr(item, "owner", "") == "gpu-fault-hyperpod-adapter"
)
submission_key = (
    command.result_details.get("submission_idempotency_key")
    or hyperpod_submission_idempotency_key(
        command.workflow.request_id,
        command.step_index,
        command.step.operation,
    )
)
lifecycle = step_adapter.dispatcher.adapter
if lifecycle.store is None:
    raise RuntimeError("HyperPod durable submission store is unavailable")
record = lifecycle.store.get_hyperpod_submission(
    lifecycle.config.cluster_name,
    submission_key,
)
expected_identity = (
    HyperPodAction.REBOOT,
    tuple(sorted(command.step.node_ids)),
)
if (
    record.idempotency_key != submission_key
    or record.request_identity != expected_identity
    or record.state != "SUBMITTED"
    or record.result is None
):
    raise RuntimeError(
        "exact HyperPod submission is not safely replayable"
    )
result = step_adapter.dispatcher.adapter.submit(
    HyperPodAction.REBOOT,
    command.step.node_ids,
    isolation_verified_nodes=command.step.node_ids,
    confirm_cluster_name=executor.confirm_cluster_name,
    workflow_fencing_token=command.workflow.fencing_token,
    expected_fencing_token=command.fencing_token,
    idempotency_key=submission_key,
)
print(json.dumps(result.model_dump(mode="json"), sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    hyperpod_cluster: str
    executor_role_arn: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_EXECUTOR_ROLE_ARN": self.executor_role_arn,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }

    @property
    def executor_role_name(self) -> str:
        return self.executor_role_arn.rsplit("/", 1)[-1]


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
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        executor_role_arn=required(
            arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", ""),
            "executor role ARN",
        ),
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_hyperpod.py::"
        "test_read_only_preflight_reports_all_mutation_gates",
        "tests/hyperpod/test_hyperpod.py::"
        "test_provider_replace_can_be_disabled_without_disabling_reboot",
        "tests/hyperpod/test_hyperpod.py::test_reboot_submission_is_idempotent",
        "tests/execution/_node_action_cases_2.py::"
        "test_hyperpod_reboot_auto_confirms_new_ready_agent_incarnation",
        "tests/execution/_node_action_cases_2.py::"
        "test_hyperpod_reboot_waits_for_post_reboot_stabilization",
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


def preflight_probe_errors(
    probe: dict[str, Any],
    hyperpod_cluster: str,
) -> list[str]:
    errors = []
    positive = probe.get("positive") or {}
    missing = probe.get("missing_isolation") or {}
    wrong = probe.get("wrong_isolation") or {}
    disabled = probe.get("disabled") or {}
    if probe.get("configured_cluster_name") != hyperpod_cluster:
        errors.append("executor HyperPod cluster name differs from the plan")
    if not positive.get("safe_to_submit"):
        errors.append("positive reboot preflight is not safe_to_submit")
    if positive.get("node_recovery") != "None":
        errors.append("HyperPod NodeRecovery is not None")
    for name, result in (("missing", missing), ("wrong", wrong)):
        failures = " ".join(result.get("gate_failures") or [])
        if result.get("safe_to_submit") or (
            "trusted scheduler isolation evidence is missing" not in failures
        ):
            errors.append(f"{name} isolation preflight did not fail closed")
    disabled_failures = " ".join(disabled.get("gate_failures") or [])
    if disabled.get("safe_to_submit") or (
        "HyperPod REBOOT mutation is disabled by configuration" not in disabled_failures
    ):
        errors.append("disabled reboot preflight did not fail closed")
    return errors


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    fixture = RegionalLiveFixture(settings.regional)
    state = fixture.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.node)
    workloads = fixture.business_workloads(settings.node)
    tests = focused_tests(case_dir)
    probe = fixture.executor_python(PREFLIGHT_PROBE, settings.node)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    errors = preflight_probe_errors(probe, settings.hyperpod_cluster)
    if not predecessor["valid"]:
        errors.append("DESTR-001 predecessor evidence is not PASS")
    if node["ready"] != "True":
        errors.append("target node is not Ready")
    if node["unschedulable"]:
        errors.append("target node is already unschedulable")
    if node["taints"]:
        errors.append("target node has pre-existing taints")
    if workloads:
        errors.append("target node has non-system running Pods")
    if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    reboot = capability(state.get("profile"), "nodeReboot")
    if reboot is None:
        errors.append("runtime profile has no nodeReboot capability")
    elif (
        reboot.get("mode") != "OWN"
        or reboot.get("owner") != "gpu-fault-hyperpod-adapter"
        or reboot.get("adapter") != "regional-cluster-executor"
    ):
        errors.append("nodeReboot is not OWN by the HyperPod adapter")
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if int((state.get("queue") or {}).get("depth") or 0):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if state.get("event") is not None:
        errors.append("target node has a recent XID event")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "provider_preflight": probe,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def wait_for_submission(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    marker: str,
    observed_after: datetime,
    case_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = regional.store_snapshot(
            node=settings.node,
            marker=marker,
            observed_after=observed_after,
            hyperpod_cluster=settings.hyperpod_cluster,
        )
        workflow = last.get("workflow") or {}
        submission = last.get("submission") or {}
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "workflow_status": workflow.get("status"),
                "completed_operations": workflow.get("completed_operations") or [],
                "submission_state": submission.get("state"),
            }
        )
        write_json_atomic(
            case_dir / "submission-timeline.json",
            {"entries": timeline},
        )
        if submission.get("state") == "SUBMITTED":
            return last
        if workflow.get("status") in {
            "BLOCKED",
            "FAILED",
            "SUCCEEDED",
            "SUPERSEDED",
        }:
            return last
        time.sleep(5)
    raise RegionalFixtureError(
        f"HyperPod reboot submission did not reach SUBMITTED: {last}"
    )


def restart_executor(
    regional: RegionalLiveFixture,
    command: dict[str, Any],
) -> dict[str, Any]:
    pods_before = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
    if len(pods_before) < 2:
        raise RegionalFixtureError("expected at least two Ready executor Pods")
    details = command.get("result_details") or {}
    executor_id = str(
        details.get("executor_id")
        or command.get("last_lease_owner")
        or command.get("lease_owner")
        or ""
    )
    target = next(
        (item for item in pods_before if str(item["name"]) in executor_id),
        pods_before[0],
    )
    regional.kubectl(
        "gpu",
        "delete",
        "pod",
        str(target["name"]),
        "--wait=false",
        timeout=30,
    )
    deadline = time.monotonic() + 300
    pods_after: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        pods_after = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
        if len(pods_after) >= len(pods_before) and all(
            item["uid"] != target["uid"] for item in pods_after
        ):
            return {
                "deleted": target,
                "before": pods_before,
                "after": pods_after,
            }
        time.sleep(5)
    raise RegionalFixtureError(
        f"executor Deployment did not replace deleted Pod: {pods_after}"
    )


def workflow_errors(
    state: dict[str, Any],
    *,
    expected_artifact: str | None,
    expected_boot_id: str | None,
) -> list[str]:
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    submission = state.get("submission") or {}
    agent = state.get("agent") or {}
    if event.get("xid") != 79:
        errors.append("matched event is not XID 79")
    if decision.get("action") != EXPECTED_ACTION:
        errors.append("policy did not resolve XID 79 to REBOOT_NODE")
    if workflow.get("status") != "SUCCEEDED":
        errors.append("reboot workflow is not SUCCEEDED")
    operations = [item.get("operation") for item in workflow.get("official_steps", [])]
    for operation in (
        "MARK_UNSCHEDULABLE",
        "RESTART_NODE",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ):
        if operation not in operations:
            errors.append(f"reboot workflow is missing {operation}")
    if submission.get("state") != "SUBMITTED":
        errors.append("HyperPod submission record is not SUBMITTED")
    if submission.get("action") != "REBOOT":
        errors.append("HyperPod submission action is not REBOOT")
    if expected_artifact and agent.get("artifact_sha256") != expected_artifact:
        errors.append("Node Agent artifact changed across reboot")
    if expected_boot_id and agent.get("boot_id") == expected_boot_id:
        errors.append("fleet Agent boot ID did not change")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("Node Agent did not return ACTIVE")
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    targets = (preflight["provider_preflight"].get("positive") or {}).get(
        "targets"
    ) or []
    return {
        "risk": "destructive-provider-reboot",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "hyperpod_cluster": settings.hyperpod_cluster,
        "provider_targets": targets,
        "mutation": (
            "write one synthetic XID 79 to /dev/kmsg, allow the executor role "
            "to submit one HyperPod reboot, then restart the submitting "
            "executor Pod and verify durable duplicate replay"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "node_boot_id": preflight["node"]["boot_id"],
            "agent_generation": (state.get("agent") or {}).get("generation"),
            "agent_artifact_sha256": (state.get("agent") or {}).get("artifact_sha256"),
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
        },
        "stop_conditions": [
            "DESTR-001 has not passed in formal sequence",
            "preflight or focused regression failure",
            "NodeRecovery is not None",
            "target node is not Ready/schedulable/idle",
            "node, Agent, Profile, cluster name or release drift",
            "submission record does not reach SUBMITTED",
            "CloudTrail shows more than one reboot or any replace/delete",
            "node does not return before the timeout",
            "GPU/host/fabric validation does not pass",
        ],
        "rollback": {
            "provider_reboot_does_not_destroy_the_instance": True,
            "runner_waits_for_the_original_node_to_return": True,
            "failed_return_keeps_the_node_out_of_service_for_operator_action": True,
            "provider_replace_is_never_attempted": True,
            "runner_finally_deletes_probe_resources": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "node_boot_id": preflight["node"]["boot_id"],
        "agent_generation": (preflight["store"].get("agent") or {}).get("generation"),
        "agent_artifact_sha256": (preflight["store"].get("agent") or {}).get(
            "artifact_sha256"
        ),
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-002 plan drifted: {planned} != {current}")


def host_probe(settings: Settings, run_id: str) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )


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
    run_id = f"destr002-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr002-{int(time.time())}-a{attempt}"
    host = host_probe(settings, run_id)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    injection_started: datetime | None = None
    try:
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA compute clients")
        if not baseline_host["kmsg_writable"]:
            raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
        target_bdf = str(baseline_host["gpu_inventory"][0]["pci_bdf"])
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        injection_started = datetime.now(timezone.utc)
        injection = host.execute(
            "write-xid79",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            target_bdf,
        )
        write_json_atomic(case_dir / "injection.json", injection)
        submitted = wait_for_submission(
            regional,
            settings,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=600,
        )
        write_json_atomic(case_dir / "submitted-state.json", submitted)
        if (submitted.get("submission") or {}).get("state") != "SUBMITTED":
            raise RegionalFixtureError("reboot submission did not reach SUBMITTED")
        reboot_commands = [
            item
            for item in submitted.get("commands") or []
            if item.get("step", {}).get("operation") == "RESTART_NODE"
        ]
        if len(reboot_commands) != 1:
            raise RegionalFixtureError("reboot remote command is not unique")
        executor_restart = restart_executor(regional, reboot_commands[0])
        write_json_atomic(case_dir / "executor-restart.json", executor_restart)
        duplicate = regional.executor_python(
            DIRECT_DUPLICATE_REPLAY,
            json.dumps(reboot_commands[0], sort_keys=True),
            timeout=180,
        )
        write_json_atomic(case_dir / "duplicate-replay.json", duplicate)
        if duplicate.get("duplicate") is not True:
            raise RegionalFixtureError("durable HyperPod replay was not duplicate")

        node_after_boot = regional.wait_node_ready(
            settings.node,
            timeout_seconds=1800,
            expected_boot_id=preflight["node"]["boot_id"],
        )
        write_json_atomic(case_dir / "node-after-boot.json", node_after_boot)
        state = regional.wait_for_workflow(
            node=settings.node,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=1800,
            hyperpod_cluster=settings.hyperpod_cluster,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        errors = workflow_errors(
            state,
            expected_artifact=(preflight["store"].get("agent") or {}).get(
                "artifact_sha256"
            ),
            expected_boot_id=(preflight["store"].get("agent") or {}).get("boot_id"),
        )
        if node_after_boot["boot_id"] == preflight["node"]["boot_id"]:
            errors.append("Kubernetes Node boot ID did not change")
        if (
            node_after_boot["uid"] != preflight["node"]["uid"]
            or node_after_boot["gpu_allocatable"]
            != preflight["node"]["gpu_allocatable"]
        ):
            errors.append("Kubernetes Node identity or GPU capacity changed")
        provider = regional.provider_events(
            injection_started,
            datetime.now(timezone.utc),
        )
        write_json_atomic(case_dir / "provider-events.json", {"events": provider})
        reboot_events = [
            item for item in provider if item["event_name"] in REBOOT_EVENTS
        ]
        forbidden = [
            item for item in provider if item["event_name"] in FORBIDDEN_EVENTS
        ]
        if len(reboot_events) != 1:
            errors.append("CloudTrail does not contain exactly one reboot event")
        elif not provider_event_actor_matches_role(
            reboot_events[0],
            settings.executor_role_arn,
        ):
            errors.append("CloudTrail reboot actor is not the executor role")
        if forbidden:
            errors.append("CloudTrail contains replace/delete mutation")
        submission = state.get("submission") or {}
        requested = set(submission.get("requested_node_identifiers") or [])
        target_aliases = {
            str(item)
            for target in (preflight["provider_preflight"].get("positive") or {}).get(
                "targets", []
            )
            for item in (
                target.get("node_logical_id"),
                target.get("instance_id"),
                target.get("private_dns_hostname"),
            )
            if item
        }
        if not requested or requested.isdisjoint(target_aliases | {settings.node}):
            errors.append("submission record does not identify the target node")
        final_node = regional.node_snapshot(settings.node)
        write_json_atomic(case_dir / "node-final.json", final_node)
        if final_node["ready"] != "True":
            errors.append("target node is not Ready after reboot")
        if final_node["unschedulable"] or final_node["ownership_annotations"]:
            errors.append("target node scheduling ownership was not restored")
        cpu_after = regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "submission": submission,
                "duplicate_replay": duplicate,
                "provider_events": provider,
                "executor_restart": executor_restart,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            residuals = host.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
        try:
            final_node = regional.node_snapshot(settings.node)
            result["final_node"] = final_node
            if final_node["ready"] != "True":
                result["verdict"] = "FAIL"
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureError(f"received signal {signum}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-002 HyperPod reboot acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", default="")
    value.add_argument("--region", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)
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
    raise SystemExit(main())
