#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_restart,
)
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

DEFAULT_A_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)
DEFAULT_D_MANIFEST = (
    Path(__file__).with_name("manifests") / "training" / "xid11-single-node-job.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-012"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-009"
CONFIRMATION = "DESTR012_VERIFY_EXCLUSIVE_WORKLOAD_RECOVERY"
AUTO_RESUME_ANNOTATION = "sagemaker.amazonaws.com/enable-job-auto-resume"
MANAGED_WORKLOAD_OWNER = "gpu-fault-kubernetes-adapter"
GROUP_A_OWNER_OPERATIONS = ("STOP_WORKLOADS", "RESTART_WORKLOAD")
GROUP_A_EVIDENCE_SOURCE = f"{PREDECESSOR_CASE_ID} evidence"
GROUP_A_WORKLOAD_SOURCE = "rerun 24-GPU workload restart"


PROFILE_REPLICA_PROBE = r"""
import hashlib
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, profile_version = sys.argv[1:]
store = ApplicationContext.from_environment().store
registration = store.get_regional_cluster(cluster_id)
profile = store.get_profile(profile_version)
payload = profile.model_dump(mode="json")
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
print(json.dumps({
    "cluster_id": registration.cluster_id,
    "allowed_namespaces": registration.allowed_namespaces,
    "profile": payload,
    "profile_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
}, sort_keys=True))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    a_manifest: Path
    d_manifest: Path
    a_job_id: str
    a_attempt_id: str
    d_job_id: str
    d_attempt_id: str
    predecessor_path: Path
    # Group A's default reads the owners from the DESTR-009 evidence; the flag
    # restores the old behaviour of restarting a second 24-GPU job for them.
    rerun_group_a_workload: bool = False

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_DESTR012_A_MANIFEST": str(self.a_manifest),
            "GPU_FAULT_DESTR012_D_MANIFEST": str(self.d_manifest),
            "GPU_FAULT_DESTR012_A_JOB_ID": self.a_job_id,
            "GPU_FAULT_DESTR012_A_ATTEMPT_ID": self.a_attempt_id,
            "GPU_FAULT_DESTR012_D_JOB_ID": self.d_job_id,
            "GPU_FAULT_DESTR012_D_ATTEMPT_ID": self.d_attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
            "GPU_FAULT_DESTR012_RERUN_GROUP_A": str(self.rerun_group_a_workload),
        }


def derived_identities(
    run_dir: Path,
    attempt: int,
) -> tuple[str, str, str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:10]
    a_job = f"destr012-a-{suffix}"
    d_job = f"destr012-d-{suffix}"
    return (
        a_job,
        f"{a_job}-a001",
        d_job,
        f"{d_job}-a001",
    )


def configure(arguments: argparse.Namespace) -> Settings:
    defaults = derived_identities(arguments.run_dir, arguments.attempt)
    a_job = arguments.a_job_id.strip() or defaults[0]
    d_job = arguments.d_job_id.strip() or defaults[2]
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
        a_manifest=Path(arguments.a_manifest).expanduser().resolve(),
        d_manifest=Path(arguments.d_manifest).expanduser().resolve(),
        a_job_id=a_job,
        a_attempt_id=arguments.a_attempt_id.strip() or f"{a_job}-a001",
        d_job_id=d_job,
        d_attempt_id=arguments.d_attempt_id.strip() or f"{d_job}-a001",
        predecessor_path=predecessor,
        rerun_group_a_workload=bool(arguments.rerun_group_a_workload),
    )


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``.

    See ``run_destr009_workload_restart.focused_tests``: a passing result
    recorded by ``--plan`` against the same source digest is not re-run.
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
        "tests/execution/test_kubernetes.py::"
        "test_kubernetes_refuses_workload_with_hyperpod_auto_resume",
        "tests/execution/test_kubernetes.py::"
        "test_kubernetes_stops_workload_without_hyperpod_auto_resume",
        "tests/execution/_misc_cases_1.py::"
        "test_managed_recovery_observer_never_submits_mutation",
        "tests/hyperpod/test_e2e_xid94.py::"
        "test_active_executor_xid94_waits_for_managed_recovery",
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


def pod_python(
    regional: RegionalLiveFixture,
    plane: str,
    pod: str,
    script: str,
    *arguments: str,
) -> dict[str, Any]:
    output = regional.kubectl(
        plane,
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-",
        *arguments,
        input_text=script,
        timeout=120,
    )
    value = json.loads(output.splitlines()[-1])
    if not isinstance(value, dict):
        raise RegionalFixtureError("profile probe did not return a JSON object")
    return cast(dict[str, Any], value)


def profile_replica_audit(
    regional: RegionalLiveFixture,
    *,
    profile_version: str,
) -> dict[str, Any]:
    replicas = []
    for app in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
        pods = regional.ready_pods("cpu", app)
        if not pods:
            raise RegionalFixtureError(f"no Ready CPU replicas for app={app}")
        for pod in pods:
            value = pod_python(
                regional,
                "cpu",
                str(pod["name"]),
                PROFILE_REPLICA_PROBE,
                regional.settings.cluster_id,
                profile_version,
            )
            replicas.append(
                {
                    "app": app,
                    "pod": pod["name"],
                    **value,
                }
            )
    hashes = {item["profile_sha256"] for item in replicas}
    namespaces = {tuple(sorted(item["allowed_namespaces"])) for item in replicas}
    errors = []
    if len(hashes) != 1:
        errors.append("CPU replicas disagree on the Runtime Profile")
    if len(namespaces) != 1:
        errors.append("CPU replicas disagree on allowed namespaces")
    for replica in replicas:
        errors.extend(
            f"{replica['pod']}: {error}"
            for error in workload_restart.managed_owner_errors(replica["profile"])
        )
    return {
        "replicas": replicas,
        "profile_sha256s": sorted(hashes),
        "allowed_namespaces": (
            list(next(iter(namespaces))) if len(namespaces) == 1 else []
        ),
        "errors": errors,
    }


def namespace_auto_resume_audit(
    regional: RegionalLiveFixture,
    namespaces: list[str],
) -> dict[str, Any]:
    objects = []
    errors = []
    for namespace in namespaces:
        try:
            value = json.loads(
                regional.kubectl(
                    "gpu",
                    "get",
                    "pytorchjobs.kubeflow.org,jobs.batch,pods",
                    "-o",
                    "json",
                    namespace=namespace,
                )
            )
        except Exception as exc:
            errors.append(
                f"{namespace}: cannot enumerate managed workload resources: {exc}"
            )
            continue
        for item in value.get("items", []):
            annotations = item["metadata"].get("annotations", {})
            raw = annotations.get(AUTO_RESUME_ANNOTATION)
            objects.append(
                {
                    "namespace": namespace,
                    "kind": item.get("kind"),
                    "name": item["metadata"].get("name"),
                    "managed": (
                        item["metadata"].get("labels", {}).get("gpu-fault.io/managed")
                        == "true"
                    ),
                    "auto_resume": raw,
                }
            )
    enabled = [
        item
        for item in objects
        if str(item.get("auto_resume") or "").strip().lower() == "true"
    ]
    if enabled:
        errors.append("managed namespaces contain auto-resume=true resources")
    return {
        "namespaces": namespaces,
        "objects": objects,
        "enabled": enabled,
        "errors": errors,
    }


def group_b_audit(
    regional: RegionalLiveFixture,
    *,
    profile_version: str,
) -> dict[str, Any]:
    profile = profile_replica_audit(
        regional,
        profile_version=profile_version,
    )
    namespaces = profile["allowed_namespaces"]
    workloads = namespace_auto_resume_audit(regional, namespaces)
    return {
        "profile": profile,
        "workloads": workloads,
        "errors": [*profile["errors"], *workloads["errors"]],
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    for path in (settings.site_file, settings.a_manifest, settings.d_manifest):
        if not path.is_file():
            raise RegionalFixtureError(f"required fixture file does not exist: {path}")
    regional = RegionalLiveFixture(settings.regional)
    identity = regional.evidence_identity()
    gpu_nodes = regional.gpu_nodes()
    candidates = [
        item
        for item in gpu_nodes
        if item["ready"] == "True" and not item["unschedulable"] and not item["taints"]
    ]
    state = (
        regional.store_snapshot(node=str(candidates[0]["name"])) if candidates else {}
    )
    runtime_identity = regional.runtime_identity()
    profile_version = str((state.get("profile") or {}).get("profile_version") or "")
    group_b = (
        group_b_audit(regional, profile_version=profile_version)
        if profile_version
        else {"errors": ["runtime profile version is unavailable"]}
    )
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    errors = list(group_b["errors"])
    errors.extend(runtime_identity_errors(runtime_identity))
    if not predecessor["valid"]:
        errors.append("DESTR-009 predecessor evidence is not PASS")
    if len(candidates) < 3:
        errors.append("fewer than three Ready schedulable untainted GPU nodes")
    gpu_workloads = regional.gpu_workloads()
    if gpu_workloads:
        errors.append("GPU cluster already has active or pending GPU workloads")
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    for manifest in (settings.a_manifest, settings.d_manifest):
        if TRAINING_IMAGE not in manifest.read_text(encoding="utf-8"):
            errors.append(f"training image digest differs in {manifest.name}")
    result = {
        "release_id": state.get("release_id"),
        "evidence_identity": identity,
        "gpu_nodes": gpu_nodes,
        "candidate_nodes": candidates,
        "gpu_workloads": gpu_workloads,
        "store": state,
        "runtime_identity": runtime_identity,
        "group_b": group_b,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workload_settings(
    settings: Settings,
    *,
    group: str,
) -> workload_restart.Settings:
    if group == "A":
        return workload_restart.Settings(
            regional=settings.regional,
            site_file=settings.site_file,
            manifest=settings.a_manifest,
            job_id=settings.a_job_id,
            attempt_id=settings.a_attempt_id,
            predecessor_path=settings.predecessor_path,
        )
    if group == "D":
        return workload_restart.Settings(
            regional=settings.regional,
            site_file=settings.site_file,
            manifest=settings.d_manifest,
            job_id=settings.d_job_id,
            attempt_id=settings.d_attempt_id,
            predecessor_path=settings.predecessor_path,
        )
    raise ValueError(f"unknown DESTR-012 group: {group}")


def executor_log_snapshot(
    regional: RegionalLiveFixture,
    since: datetime,
    workload_name: str,
) -> dict[str, Any]:
    return workload_restart.log_write_snapshot(
        regional,
        plane="gpu",
        apps=("gpu-fault-cluster-executor",),
        since=since,
        workload_name=workload_name,
    )


def delete_after_quiescence(
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    result: dict[str, Any],
    *,
    case_settings: workload_restart.Settings,
    case_dir: Path,
    node: str | None,
    marker: str | None,
    observed_after: datetime | None,
    workflow_request_ids: list[str],
) -> None:
    """Delete a group's workload only once its workflow has gone quiet.

    The same rule DESTR-009 applies to its own job: a delete issued while
    RESTART_WORKLOAD is still PENDING on the executor races the adapter that is
    about to act on the object. Without quiescence the delete is deferred, the
    residual is reported, and the group fails.
    """

    if node and marker and observed_after is not None:
        try:
            result["cleanup_quiescence"] = workload_restart.wait_for_cleanup_quiescence(
                regional=regional,
                node=node,
                marker=marker,
                observed_after=observed_after,
                job_id=case_settings.job_id,
                attempt_id=case_settings.attempt_id,
                case_dir=case_dir,
                workflow_request_ids=workflow_request_ids or None,
            )
        except Exception as exc:
            result["cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["workload_cleanup_deferred"] = True
            result["verdict"] = "FAIL"
            return
    try:
        workload.delete()
    except Exception as exc:
        result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def group_a_owner_errors(steps: list[dict[str, Any]]) -> list[str]:
    owners = {
        item.get("operation"): item.get("execution_owner")
        for item in steps
        if isinstance(item, dict)
    }
    return [
        f"group A {operation} owner is not Kubernetes adapter"
        for operation in GROUP_A_OWNER_OPERATIONS
        if owners.get(operation) != MANAGED_WORKLOAD_OWNER
    ]


def group_a_from_evidence(
    predecessor_path: Path,
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    """Group A read from the DESTR-009 evidence this run already validated.

    The old group A restarted a second 24-GPU PyTorchJob to look at two owner
    fields DESTR-009 had just produced. The predecessor evidence carries the
    compiled steps (``workflow_official_steps``), so the check is a read; an
    evidence file from a runner that did not record them fails with the
    remedy named, and ``--rerun-group-a-workload`` keeps the old path.
    """

    result: dict[str, Any] = {
        "group": "A",
        "verdict": "FAIL",
        "group_a_source": GROUP_A_EVIDENCE_SOURCE,
        "predecessor_path": str(predecessor_path),
    }
    remedy = (
        "rerun DESTR-009 on this release with the current runner, or pass "
        "--rerun-group-a-workload"
    )
    if not predecessor.get("evidence_valid"):
        result["error"] = (
            "DESTR-009 evidence is not a PASS bound to this release/cluster "
            "("
            + str(
                predecessor.get("evidence_error")
                or predecessor.get("error")
                or predecessor.get("verdict")
            )
            + f"); {remedy}"
        )
        return result
    try:
        value = json.loads(predecessor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"cannot read DESTR-009 evidence: {exc}; {remedy}"
        return result
    steps = value.get("workflow_official_steps") if isinstance(value, dict) else None
    if not isinstance(steps, list) or not steps:
        result["error"] = (
            "DESTR-009 evidence lacks workflow_official_steps with execution "
            f"owners; {remedy}"
        )
        return result
    errors = group_a_owner_errors(steps)
    result.update(
        {
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "workflow_request_id": value.get("workflow_request_id"),
            "workflow_official_steps": steps,
            "evidence_release_id": value.get("release_id"),
            "evidence_cluster_id": value.get("cluster_id"),
        }
    )
    return result


def run_group_a(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    maintenance_window_end: datetime,
) -> dict[str, Any]:
    case_settings = workload_settings(settings, group="A")
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=case_settings.manifest,
            site_file=case_settings.site_file,
            job_id=case_settings.job_id,
            attempt_id=case_settings.attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )
    result: dict[str, Any] = {
        "group": "A",
        "verdict": "FAIL",
        "group_a_source": GROUP_A_WORKLOAD_SOURCE,
    }
    target_node: str | None = None
    marker: str | None = None
    injected_at: datetime | None = None
    workflow_request_ids: list[str] = []
    try:
        submission = workload.submit()
        write_json_atomic(case_dir / "group-a-submission.json", submission)
        source = workload.wait_running(timeout_seconds=900)
        write_json_atomic(case_dir / "group-a-source.json", source)
        annotation = source["workload"]["annotations"].get(AUTO_RESUME_ANNOTATION)
        if str(annotation or "").strip().lower() == "true":
            raise RegionalFixtureError("group A workload enables auto-resume")
        source_uids = {str(item["uid"]) for item in source["pods"]}
        target_node = str(source["pods"][0]["node"])
        observation = workload_restart.wait_observation(
            regional,
            case_settings,
            node=target_node,
            expected_gpu_count=24,
        )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before group A")
        marker = f"destr012-a-{int(time.time())}"
        injected_at = datetime.now(timezone.utc)
        payload = workload_restart.xid11_payload(
            case_settings,
            case_id=CASE_ID,
            marker=marker,
            node=target_node,
            product=workload_restart.normalize_product(
                regional.node_metadata(target_node).get("product")
            ),
            observation=observation,
            observed_at=injected_at,
        )
        injection = regional.post_xid_event(payload)
        write_json_atomic(case_dir / "group-a-injection.json", injection)
        if (injection.get("receipt") or {}).get("status") != 200:
            raise RegionalFixtureError("group A processor receipt is not HTTP 200")
        state = regional.wait_for_workflow(
            node=target_node,
            marker=marker,
            observed_after=injected_at,
            case_dir=case_dir / "group-a",
            timeout_seconds=1200,
            job_id=case_settings.job_id,
            attempt_id=case_settings.attempt_id,
        )
        write_json_atomic(case_dir / "group-a-workflow.json", state)
        workflow_request_id = (state.get("workflow") or {}).get("request_id")
        if workflow_request_id:
            workflow_request_ids.append(str(workflow_request_id))
        errors = workload_restart.workflow_errors(
            state,
            expected_gpu_count=24,
        )
        official_steps = workload_restart.workflow_official_steps(state)
        errors.extend(group_a_owner_errors(official_steps))
        target = workload.wait_restarted(source_uids, timeout_seconds=900)
        write_json_atomic(case_dir / "group-a-target.json", target)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "target_node": target_node,
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
                "workflow_request_id": workflow_request_id,
                "workflow_official_steps": official_steps,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        delete_after_quiescence(
            regional,
            workload,
            result,
            case_settings=case_settings,
            case_dir=case_dir / "group-a",
            node=target_node,
            marker=marker,
            observed_after=injected_at,
            workflow_request_ids=workflow_request_ids,
        )
    return result


def group_d_failure_errors(
    state: dict[str, Any],
    *,
    expected_workload_id: str,
) -> list[str]:
    errors = []
    workflow = state.get("workflow") or {}
    if workflow.get("status") != "FAILED":
        errors.append("group D violating workflow is not FAILED")
    stop = workload_restart.terminal_step(workflow, "STOP_WORKLOADS")
    if stop is None or stop.get("status") != "FAILED":
        errors.append("group D STOP_WORKLOADS did not fail")
        return errors
    if "enable-job-auto-resume is enabled" not in str(stop.get("error") or ""):
        errors.append("group D failure does not identify auto-resume")
    details = stop.get("details") or {}
    if details.get("managed_job_recovery_workloads") != [expected_workload_id]:
        errors.append("group D details do not list only the violating workload")
    if details.get("required_annotation") != AUTO_RESUME_ANNOTATION:
        errors.append("group D details have the wrong required annotation")
    if details.get("required_annotation_value") != "absent or false":
        errors.append("group D details have the wrong required annotation value")
    commands = details.get("remediation_commands") or []
    if len(commands) != 1 or f"{AUTO_RESUME_ANNOTATION}-" not in commands[0]:
        errors.append("group D remediation command is missing")
    return errors


def run_group_d(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    maintenance_window_end: datetime,
) -> dict[str, Any]:
    case_settings = workload_settings(settings, group="D")
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=case_settings.manifest,
            site_file=case_settings.site_file,
            job_id=case_settings.job_id,
            attempt_id=case_settings.attempt_id,
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=1,
        ),
    )
    result: dict[str, Any] = {"group": "D", "verdict": "FAIL"}
    target_node: str | None = None
    # The quiescence gate before the delete follows the most recent injection:
    # the remediated retry when it happened, the blocked one otherwise.
    quiescence_marker: str | None = None
    quiescence_after: datetime | None = None
    workflow_request_ids: list[str] = []
    try:
        submission = workload.submit()
        write_json_atomic(case_dir / "group-d-submission.json", submission)
        source = workload.wait_running(timeout_seconds=900)
        target_node = str(source["pods"][0]["node"])
        observation = workload_restart.wait_observation(
            regional,
            case_settings,
            node=target_node,
            expected_gpu_count=1,
        )
        workload.annotate_auto_resume("true")
        violating = workload.snapshot()
        write_json_atomic(case_dir / "group-d-violating-baseline.json", violating)
        source_uids = {str(item["uid"]) for item in violating["pods"]}
        expected_workload_id = str(observation["workload_ids"][0])
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before group D")
        first_marker = f"destr012-d-block-{int(time.time())}"
        first_at = datetime.now(timezone.utc)
        quiescence_marker, quiescence_after = first_marker, first_at
        first_payload = workload_restart.xid11_payload(
            case_settings,
            case_id=CASE_ID,
            marker=first_marker,
            node=target_node,
            product=workload_restart.normalize_product(
                regional.node_metadata(target_node).get("product")
            ),
            observation=observation,
            observed_at=first_at,
        )
        first_injection = regional.post_xid_event(first_payload)
        write_json_atomic(
            case_dir / "group-d-blocked-injection.json",
            first_injection,
        )
        if (first_injection.get("receipt") or {}).get("status") != 200:
            raise RegionalFixtureError("group D blocked receipt is not HTTP 200")
        blocked = regional.wait_for_workflow(
            node=target_node,
            marker=first_marker,
            observed_after=first_at,
            case_dir=case_dir / "group-d-blocked",
            timeout_seconds=600,
            job_id=case_settings.job_id,
            attempt_id=case_settings.attempt_id,
        )
        write_json_atomic(case_dir / "group-d-blocked-workflow.json", blocked)
        blocked_request_id = (blocked.get("workflow") or {}).get("request_id")
        if blocked_request_id:
            workflow_request_ids.append(str(blocked_request_id))
        errors = group_d_failure_errors(
            blocked,
            expected_workload_id=expected_workload_id,
        )
        unchanged = workload.snapshot()
        write_json_atomic(case_dir / "group-d-after-block.json", unchanged)
        if unchanged["workload"]["suspend"] != violating["workload"]["suspend"]:
            errors.append("group D changed Job spec.suspend before failing")
        if {str(item["uid"]) for item in unchanged["pods"]} != source_uids:
            errors.append("group D changed Pod UIDs before failing")
        logs = executor_log_snapshot(regional, first_at, workload.name)
        write_json_atomic(case_dir / "group-d-executor-logs.json", logs)
        errors.extend(workload_restart.log_write_errors(logs, "group D executor"))
        workload.annotate_auto_resume(None)
        remediated = workload.snapshot()
        if (
            str(
                remediated["workload"]["annotations"].get(AUTO_RESUME_ANNOTATION) or ""
            ).lower()
            == "true"
        ):
            errors.append("group D auto-resume annotation was not removed")
        second_marker = f"destr012-d-retry-{int(time.time())}"
        second_at = datetime.now(timezone.utc)
        quiescence_marker, quiescence_after = second_marker, second_at
        second_payload = workload_restart.xid11_payload(
            case_settings,
            case_id=CASE_ID,
            marker=second_marker,
            node=target_node,
            product=workload_restart.normalize_product(
                regional.node_metadata(target_node).get("product")
            ),
            observation=observation,
            observed_at=second_at,
        )
        second_injection = regional.post_xid_event(second_payload)
        write_json_atomic(
            case_dir / "group-d-remediated-injection.json",
            second_injection,
        )
        if (second_injection.get("receipt") or {}).get("status") != 200:
            errors.append("group D remediated receipt is not HTTP 200")
        remediated_state = regional.wait_for_workflow(
            node=target_node,
            marker=second_marker,
            observed_after=second_at,
            case_dir=case_dir / "group-d-remediated",
            timeout_seconds=1200,
            job_id=case_settings.job_id,
            attempt_id=case_settings.attempt_id,
        )
        write_json_atomic(
            case_dir / "group-d-remediated-workflow.json",
            remediated_state,
        )
        remediated_request_id = (remediated_state.get("workflow") or {}).get(
            "request_id"
        )
        if remediated_request_id:
            workflow_request_ids.append(str(remediated_request_id))
        errors.extend(
            workload_restart.workflow_errors(
                remediated_state,
                expected_gpu_count=1,
            )
        )
        target = workload.wait_restarted(source_uids, timeout_seconds=900)
        write_json_atomic(case_dir / "group-d-target.json", target)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "blocked_marker": first_marker,
                "remediated_marker": second_marker,
                "blocked_workflow_request_id": (
                    (blocked.get("workflow") or {}).get("request_id")
                ),
                "remediated_workflow_request_id": (
                    (remediated_state.get("workflow") or {}).get("request_id")
                ),
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            workload.annotate_auto_resume(None)
        except Exception:
            pass
        delete_after_quiescence(
            regional,
            workload,
            result,
            case_settings=case_settings,
            case_dir=case_dir / "group-d",
            node=target_node,
            marker=quiescence_marker,
            observed_after=quiescence_after,
            workflow_request_ids=workflow_request_ids,
        )
    return result


def groups_not_run(groups: dict[str, dict[str, Any]]) -> list[str]:
    """The group letters whose verdict is NOT_RUN, in plan order."""

    return [
        name
        for name, group in groups.items()
        if isinstance(group, dict) and group.get("verdict") == "NOT_RUN"
    ]


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    details = {
        "risk": "live-workload-restart",
        "predecessor": preflight["predecessor"],
        "groups": ["B", "A", "D", "C"],
        "group_c": {
            "optional": True,
            "planned_status": "NOT_RUN",
            "reason": (
                "requires an independently provisioned control plane and database; "
                "the production site Profile is never modified"
            ),
        },
        "group_a": {
            "source": (
                GROUP_A_WORKLOAD_SOURCE
                if settings.rerun_group_a_workload
                else GROUP_A_EVIDENCE_SOURCE
            ),
            "job_id": settings.a_job_id,
            "attempt_id": settings.a_attempt_id,
            "manifest": str(settings.a_manifest),
        },
        "group_d": {
            "job_id": settings.d_job_id,
            "attempt_id": settings.d_attempt_id,
            "manifest": str(settings.d_manifest),
        },
        "candidate_nodes": [
            {"name": item["name"], "uid": item["uid"]}
            for item in preflight["candidate_nodes"]
        ],
        "mutation": (
            "B reads all managed namespaces and Profile replicas; A "
            + (
                "restarts one compliant 24-GPU PyTorchJob"
                if settings.rerun_group_a_workload
                else "reads the STOP/RESTART owners from the DESTR-009 evidence"
            )
            + "; D proves auto-resume=true is rejected before any workload "
            "write, removes it, then proves a new record can restart a one-GPU Job"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "profile_sha256s": (preflight["group_b"].get("profile") or {}).get(
                "profile_sha256s", []
            ),
            "candidate_node_uids": sorted(
                str(item["uid"]) for item in preflight["candidate_nodes"]
            ),
            "runtime_identity": preflight["runtime_identity"],
        },
        "stop_conditions": [
            "DESTR-009 has not passed in formal sequence",
            "B finds auto-resume=true or replica/Profile disagreement",
            "preflight or focused regression failure",
            "release rollback is active or a required deployment is not fully rolled out",
            "fewer than three idle Ready GPU nodes",
            "A differs from the DESTR-009 restart contract",
            "D mutates spec.suspend or Pod UIDs before failing",
            "D details omit the violating workload or remediation command",
            "D cannot restart after the annotation is removed",
            "any provider or control-plane Kubernetes mutation appears",
        ],
        "rollback": {
            "runner_finally_removes_the_D_group_annotation": True,
            "runner_finally_deletes_both_test_workloads": "only_after_quiescence",
            "group_workloads_are_deleted_only_after_quiescence": True,
            "runner_finally_verifies_no_test_workload_residual": True,
            "runner_finally_deletes_image_prewarm_Pods": True,
            "production_Runtime_Profile_is_never_modified": True,
            "group_C_requires_a_separate_isolated_database_fixture": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


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
        "profile_sha256s": (preflight["group_b"].get("profile") or {}).get(
            "profile_sha256s", []
        ),
        "candidate_node_uids": sorted(
            str(item["uid"]) for item in preflight["candidate_nodes"]
        ),
        "runtime_identity": preflight["runtime_identity"],
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-012 plan drifted: {planned} != {current}")

    regional = RegionalLiveFixture(settings.regional)
    profile_version = str(current["runtime_profile_version"])
    run_id = f"destr012-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=run_id,
    )
    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        **preflight["evidence_identity"],
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "focused_tests_reused": bool(
            preflight["focused_tests"].get("focused_tests_reused")
        ),
    }
    try:
        # The execute preflight ran the group B audit moments ago and gated on
        # it; a third audit (after --plan's and the preflight's) would read the
        # same replicas again for the same answer.
        group_b_before = {
            **preflight["group_b"],
            "reused_from": "execute preflight",
        }
        write_json_atomic(case_dir / "group-b-before.json", group_b_before)
        if group_b_before["errors"]:
            raise RegionalFixtureError(
                "group B failed: " + "; ".join(group_b_before["errors"])
            )
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
        regional.verify_runtime_identity(
            planned["runtime_identity"],
            evidence_path=case_dir / "runtime-identity-before-group-a.json",
            stage="before DESTR-012 group A",
        )

        if settings.rerun_group_a_workload:
            group_a = run_group_a(
                settings,
                regional,
                case_dir,
                maintenance_window_end,
            )
        else:
            group_a = group_a_from_evidence(
                settings.predecessor_path,
                preflight["predecessor"],
            )
        write_json_atomic(case_dir / "group-a.json", group_a)
        if group_a["verdict"] != "PASS":
            raise RegionalFixtureError(
                "group A failed: "
                + str(group_a.get("error") or "; ".join(group_a.get("errors") or []))
            )
        regional.verify_runtime_identity(
            planned["runtime_identity"],
            evidence_path=case_dir / "runtime-identity-after-group-a.json",
            stage="after DESTR-012 group A",
        )

        group_d = run_group_d(
            settings,
            regional,
            case_dir,
            maintenance_window_end,
        )
        write_json_atomic(case_dir / "group-d.json", group_d)
        if group_d["verdict"] != "PASS":
            raise RegionalFixtureError("group D failed")
        regional.verify_runtime_identity(
            planned["runtime_identity"],
            evidence_path=case_dir / "runtime-identity-after-group-d.json",
            stage="after DESTR-012 group D",
        )

        group_c = {
            "group": "C",
            "verdict": "NOT_RUN",
            "optional": True,
            "reason": (
                "an isolated control plane and independent database were not "
                "provisioned; the production Profile was not modified"
            ),
        }
        write_json_atomic(case_dir / "group-c.json", group_c)

        group_b_after = group_b_audit(
            regional,
            profile_version=profile_version,
        )
        write_json_atomic(case_dir / "group-b-after.json", group_b_after)
        errors = list(group_b_after["errors"])
        provider_window_end = datetime.now(timezone.utc)
        provider = regional.provider_events(started_at, provider_window_end)
        # A negative CloudTrail read inside the delivery window is provisional.
        provider_provisional = not provider and regional.provider_events_provisional(
            provider_window_end
        )
        write_json_atomic(
            case_dir / "provider-events.json",
            {"events": provider, "provider_events_provisional": provider_provisional},
        )
        if provider:
            errors.append("provider mutation appeared during DESTR-012")
        cpu_after = regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        groups = {
            "B_before": group_b_before,
            "A": group_a,
            "D": group_d,
            "C": group_c,
            "B_after": group_b_after,
        }
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "groups": groups,
                "groups_not_run": groups_not_run(groups),
                "group_a_source": group_a.get("group_a_source"),
                "provider_events": provider,
                "provider_events_provisional": provider_provisional,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Each group deletes its own workload behind the quiescence gate; a
        # second unconditional delete here would race that gate. What remains
        # for the case is to prove nothing was left behind.
        workload_residuals: dict[str, bool] = {}
        for group in ("A", "D"):
            if group == "A" and not settings.rerun_group_a_workload:
                continue
            case_settings = workload_settings(settings, group=group)
            fixture = ManagedWorkloadFixture(
                regional,
                ManagedWorkloadSettings(
                    manifest=case_settings.manifest,
                    site_file=case_settings.site_file,
                    job_id=case_settings.job_id,
                    attempt_id=case_settings.attempt_id,
                    restart_budget=1,
                    expected_pods=3 if group == "A" else 1,
                    expected_gpu_count=24 if group == "A" else 1,
                ),
            )
            try:
                workload_residuals[group] = bool(
                    regional.kubectl(
                        "gpu",
                        "get",
                        fixture.resource,
                        fixture.name,
                        "--ignore-not-found",
                        "-o",
                        "name",
                        check=False,
                    ).strip()
                )
            except Exception as exc:
                workload_residuals[group] = True
                result[f"group_{group.lower()}_residual_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        result["workload_residuals"] = workload_residuals
        if any(workload_residuals.values()):
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
                planned["runtime_identity"],
                evidence_path=case_dir / "runtime-identity-after-cleanup.json",
                stage="after DESTR-012 cleanup",
            )
        except Exception as exc:
            errors = result.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append(f"{type(exc).__name__}: {exc}")
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-012 managed recovery acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--a-manifest", type=Path, default=DEFAULT_A_MANIFEST)
    value.add_argument("--d-manifest", type=Path, default=DEFAULT_D_MANIFEST)
    value.add_argument("--a-job-id", default="")
    value.add_argument("--a-attempt-id", default="")
    value.add_argument("--d-job-id", default="")
    value.add_argument("--d-attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--rerun-group-a-workload",
        action="store_true",
        help=(
            "restart a second 24-GPU PyTorchJob for group A instead of reading "
            "the STOP/RESTART owners from the DESTR-009 evidence"
        ),
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
