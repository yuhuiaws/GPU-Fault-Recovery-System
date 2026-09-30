from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_kernel_evidence import (  # noqa: E402
    kmsg_identity_errors,  # noqa: E402
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget as MultiClusterTarget,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    MultiClusterSettings,
    registration_snapshot,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_commands import (  # noqa: E402
    RegionalFixtureError,
    run_fixture_command,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
    component_python,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
)
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402
from scripts.e2e.regional.readme_workload_fixture import (  # noqa: E402
    ReadmeWorkloadFixture,
    training_cli,
)
from scripts.e2e.regional.workload_acceptance_checks import (  # noqa: E402
    LOSS_LINE_PATTERN as LOSS_LINE_PATTERN,
    MAX_RECORDED_SUSPICIOUS_LINES as MAX_RECORDED_SUSPICIOUS_LINES,
    NOTIFICATION_DELIVERY_POLL_SECONDS as NOTIFICATION_DELIVERY_POLL_SECONDS,
    SUCCESS_LINE_PATTERN as SUCCESS_LINE_PATTERN,
    SUSPICIOUS_LOG_PATTERN as SUSPICIOUS_LOG_PATTERN,
    collective_errors as collective_errors,
    command_ids_in_logs as command_ids_in_logs,
    control_plane_blast_errors as control_plane_blast_errors,
    gpu_nodes_clean as gpu_nodes_clean,
    identity_baseline_errors as identity_baseline_errors,
    loss_errors as loss_errors,
    metadata_errors as metadata_errors,
    notification_errors as notification_errors,
    observation_pod_names as observation_pod_names,
    observation_pod_uids as observation_pod_uids,
    observation_rank_errors as observation_rank_errors,
    recovery_identity_errors as recovery_identity_errors,
    source_observation_errors as source_observation_errors,
    suspicious_log_lines as suspicious_log_lines,
    virtual_isolation_errors as virtual_isolation_errors,
    wait_for_notification_results as wait_for_notification_results,
)
from scripts.e2e.regional.workload_acceptance_probes import (  # noqa: E402
    E2E_PREFLIGHT_PROBE as E2E_PREFLIGHT_PROBE,
)
from scripts.e2e.regional.workload_acceptance_probes import (  # noqa: E402
    KMSG_EVIDENCE_PROBE as KMSG_EVIDENCE_PROBE,
)
from scripts.e2e.regional.workload_acceptance_probes import (  # noqa: E402
    OBSERVATION_POST_PROBE as OBSERVATION_POST_PROBE,
)
from scripts.e2e.regional.workload_acceptance_probes import (  # noqa: E402
    WORKLOAD_STORE_PROBE as WORKLOAD_STORE_PROBE,
)

CASE_IDS = (
    "GF-REGIONAL-WORKLOAD-001",
    "GF-REGIONAL-WORKLOAD-002",
    "GF-REGIONAL-ISO-001",
    "GF-REGIONAL-E2E-001",
)
BASELINE_MANIFEST = ROOT / "examples/hyperpod/three-node-pytorchjob.yaml"
LONG_RUNNING_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)
E2E001_PROBE = Path(__file__).with_name("probes") / "e2e001_node_probe.py"
# The executor-log markers E2E-001 step 3 greps for. Matched on word boundaries
# so `error_count=0`, a `401`-suffixed request ID or a `403`-byte payload size
# do not fail the preflight; the case-sensitive spelling is the spec's own.
# `gpu-fault-admin status` costs up to thirty minutes; the group runs it once,
# on the case that closes it, unless the operator asks otherwise.
ADMIN_STATUS_CASE = "GF-REGIONAL-WORKLOAD-002"
EXECUTOR_LOG_MAX_WINDOW_SECONDS = 4 * 3600


class WorkloadAcceptanceError(RuntimeError):
    pass


class WorkloadCaseError(WorkloadAcceptanceError):
    """A case body failed after producing a partial outcome.

    ``outcome`` carries what the case had recorded up to the failure --
    cleanup_errors, prewarm residuals, a deferred workload -- so the evidence
    file written by ``main`` shows the state the cluster was left in rather
    than only the first exception's text.
    """

    def __init__(self, cause: BaseException, outcome: dict[str, Any]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.outcome = outcome


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def raise_with_outcome(failure: BaseException | None, result: dict[str, Any]) -> None:
    """Re-raise a case body's failure with the cleanup-annotated ``result``."""

    if failure is None:
        return
    if isinstance(failure, WorkloadCaseError):
        failure.outcome.update(result)
        raise failure
    raise WorkloadCaseError(failure, result) from failure


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_fixture_command(
        command,
        input_text=input_text,
        check=check,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )


@dataclass(frozen=True)
class SiteTarget:
    cluster_id: str
    context: str
    region: str


class WorkloadSite:
    def __init__(self, site_file: Path) -> None:
        self.site_file = site_file.resolve()
        self.site = load_site(self.site_file, repository_root=ROOT)
        self.config = self.site.release_config
        self.namespace = str(self.config["namespace"])
        self.region = str(self.config["aws_region"])
        self.cpu_kubeconfig = Path(str(self.config["cpu_kubeconfig"])).resolve()
        gpu_value = str(
            self.config.get("gpu_kubeconfig")
            or self.site.environment.get("KUBECONFIG")
            or ""
        )
        if not gpu_value:
            raise WorkloadAcceptanceError("site does not resolve a GPU kubeconfig")
        self.gpu_kubeconfig = Path(gpu_value).resolve()
        self.targets = {
            str(item["cluster_id"]): SiteTarget(
                cluster_id=str(item["cluster_id"]),
                context=str(item["context"]),
                region=str(item["region"]),
            )
            for item in self.config["clusters"]
        }

    def target(self, cluster_id: str) -> SiteTarget:
        if not cluster_id:
            if len(self.targets) != 1:
                raise WorkloadAcceptanceError(
                    "--cluster-id is required for a multi-cluster site"
                )
            return next(iter(self.targets.values()))
        try:
            return self.targets[cluster_id]
        except KeyError as exc:
            raise WorkloadAcceptanceError(
                f"cluster is absent from site: {cluster_id}"
            ) from exc

    def regional(self, target: SiteTarget) -> RegionalLiveFixture:
        return RegionalLiveFixture(
            RegionalLiveSettings(
                cpu_kubeconfig=self.cpu_kubeconfig,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=target.context,
                namespace=self.namespace,
                cluster_id=target.cluster_id,
                region=self.region,
            )
        )

    def multi(
        self,
        primary: SiteTarget,
        secondary: SiteTarget,
    ) -> MultiClusterSettings:
        return MultiClusterSettings(
            cpu_kubeconfig=self.cpu_kubeconfig,
            namespace=self.namespace,
            region=self.region,
            cluster_a=MultiClusterTarget(
                cluster_id=primary.cluster_id,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=primary.context,
            ),
            cluster_b=MultiClusterTarget(
                cluster_id=secondary.cluster_id,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=secondary.context,
            ),
        )


def derived_identity(
    run_dir: Path,
    attempt: int,
    case_id: str,
) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{case_id}".encode()
    ).hexdigest()[:12]
    job_id = f"{case_id.lower().replace('gf-regional-', '')}-{suffix}"
    return job_id, f"{job_id}-a001"


def managed_fixture(
    regional: RegionalLiveFixture,
    *,
    manifest: Path,
    site_file: Path,
    job_id: str,
    attempt_id: str,
) -> ManagedWorkloadFixture:
    return ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )


def readme_fixture(
    regional: RegionalLiveFixture,
    *,
    manifest: Path,
    site_file: Path,
    job_id: str,
    attempt_id: str,
    case_dir: Path,
) -> ReadmeWorkloadFixture:
    return ReadmeWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
        case_dir=case_dir,
        training_container="pytorch",
    )


def workload_store(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
    workflow_request_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    arguments = [regional.settings.cluster_id, job_id, attempt_id]
    if workflow_request_ids:
        for request_id in workflow_request_ids:
            if not request_id or "," in request_id:
                raise ValueError("workflow request IDs must be non-empty, no comma")
        arguments.append(",".join(workflow_request_ids))
    return regional.cpu_python(WORKLOAD_STORE_PROBE, *arguments)


def assert_clean_identity_baseline(
    states: Sequence[dict[str, Any]],
    *,
    job_id: str,
    attempt_id: str,
) -> None:
    findings = []
    for state in states:
        errors = identity_baseline_errors(state)
        if errors:
            findings.append(f"{state.get('cluster_id')}: {'; '.join(errors)}")
    if findings:
        raise RegionalFixtureError(
            f"job {job_id}/{attempt_id} already has control-plane state "
            f"({' | '.join(findings)}); change --attempt/--job-id and rerun"
        )


def wait_terminal_observation(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = workload_store(
            regional,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        observations = last["observations"]
        if len(observations) == 1:
            observation = observations[0]
            if (
                observation.get("workload_phase") == "SUCCEEDED"
                and len(observation.get("containers") or []) == 3
                and all(
                    item.get("terminated") and item.get("exit_code") == 0
                    for item in observation.get("containers") or []
                )
            ):
                return last
        time.sleep(5)
    raise RegionalFixtureError(f"terminal observation did not converge: {last}")


def wait_finite_workload(
    fixture: ManagedWorkloadFixture,
    *,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = fixture.snapshot()
        except Exception:
            time.sleep(5)
            continue
        pods = last["pods"]
        logs = last["heartbeat_logs"]
        if (
            len(pods) == 3
            and len({item["node"] for item in pods}) == 3
            and all(item["phase"] == "Succeeded" for item in pods)
            and all(
                "SUCCESS" in logs.get(str(item["name"]), "")
                and "all_reduce=300.0" in logs.get(str(item["name"]), "")
                for item in pods
            )
        ):
            return last
        time.sleep(5)
    raise RegionalFixtureError(f"finite workload did not succeed: {last}")


def watcher_interval_seconds(regional: RegionalLiveFixture) -> int:
    deployment = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "deployment",
            "gpu-fault-completion-watcher",
            "-o",
            "json",
        )
    )
    environment = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0].get(
            "env", []
        )
        if "value" in item
    }
    return int(environment.get("GPU_FAULT_COMPLETION_WATCH_INTERVAL_SECONDS") or 30)


def admin_status(state_dir: Path) -> dict[str, Any]:
    completed = run(
        [
            sys.executable,
            "-m",
            "gpu_fault.admin.cli",
            "status",
            "--state-dir",
            str(state_dir.resolve()),
        ],
        check=False,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    return {
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }


def prewarm_nodes(regional: RegionalLiveFixture) -> list[str]:
    nodes = [
        str(item["name"])
        for item in regional.gpu_nodes()
        if item["ready"] == "True" and not item["unschedulable"]
    ]
    if len(nodes) < 3:
        raise RegionalFixtureError("at least three schedulable GPU nodes are required")
    return nodes


def admin_status_policy(case_id: str, *, skip: bool) -> tuple[bool, str]:
    """Whether this case runs ``gpu-fault-admin status``, and why not if not."""

    if skip:
        return False, "skipped by --skip-admin-status"
    if case_id != ADMIN_STATUS_CASE:
        return False, (
            f"admin status runs once per group, on {ADMIN_STATUS_CASE}; "
            f"{case_id} records the workload checks only"
        )
    return True, f"{case_id} closes the WORKLOAD group"


def executor_logs_since(
    regional: RegionalLiveFixture,
    *,
    since: datetime,
) -> dict[str, str]:
    """Each Ready executor Pod's log since ``since`` (bounded, never negative)."""

    elapsed = int((datetime.now(timezone.utc) - since).total_seconds())
    window = max(60, min(elapsed + 60, EXECUTOR_LOG_MAX_WINDOW_SECONDS))
    logs = {}
    pods = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
    if not pods:
        raise RegionalFixtureError("no Ready executor for the log window")
    for pod in pods:
        logs[str(pod["name"])] = regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            f"--since={window}s",
            timeout=120,
        )
        if not logs[str(pod["name"])].strip():
            raise RegionalFixtureError("executor log window is empty")
    return logs


def wait_pod_uids_unchanged(
    workload: ManagedWorkloadFixture,
    expected_uids: set[str],
    *,
    timeout_seconds: int = 60,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    """Hold that the workload's Pod UIDs stay ``expected_uids`` for the window.

    Reads ``pods()`` -- one kubectl get per poll -- rather than the fixture's
    ``snapshot()``, which also pulls three Pods' logs every poll for a check
    that only looks at UIDs. The final snapshot confirms the Pods still
    heartbeat.
    """

    deadline = time.monotonic() + timeout_seconds
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = workload.pods()
        if {str(item["uid"]) for item in last} != expected_uids:
            raise RegionalFixtureError(f"managed workload Pod UIDs changed: {last}")
        time.sleep(poll_seconds)
    snapshot = workload.snapshot()
    if {str(item["uid"]) for item in snapshot["pods"]} != expected_uids:
        raise RegionalFixtureError("managed workload Pod UIDs changed")
    return snapshot


def workload_residuals(
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    job_id: str,
) -> list[dict[str, Any]]:
    document = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            f"pod,{workload.resource}",
            "-l",
            f"gpu-fault.io/job-id={job_id}",
            "-o",
            "json",
        )
    )
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise RegionalFixtureError("workload cleanup query is not a resource list")
    return [
        {
            "kind": item.get("kind"),
            "name": (item.get("metadata") or {}).get("name"),
            "uid": (item.get("metadata") or {}).get("uid"),
        }
        for item in document["items"]
    ]


def cleanup_workload(
    *,
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    injection: dict[str, Any] | None,
    result: dict[str, Any],
    label: str,
) -> list[str]:
    """Delete the case's workload only once its workflow is provably quiet.

    A delete while STOP_WORKLOADS/RESTART_WORKLOAD is still RUNNING races the
    executor and leaves a workflow acting on a workload that no longer exists.
    When the fault was injected (``injection`` names node/marker/time) the
    DESTR-009 quiescence wait runs first; if it cannot prove quiescence the
    workload is left in place and ``workload_cleanup_deferred`` says so.
    """

    errors: list[str] = []
    if injection:
        try:
            result[f"{label}_cleanup_quiescence"] = (
                workload_case.wait_for_cleanup_quiescence(
                    regional=regional,
                    node=str(injection["node"]),
                    marker=str(injection["marker"]),
                    observed_after=injection["observed_after"],
                    job_id=job_id,
                    attempt_id=attempt_id,
                    case_dir=case_dir,
                    workflow_request_ids=injection.get("workflow_request_ids"),
                )
            )
            if result[f"{label}_cleanup_quiescence"].get("safe_to_delete") is not True:
                raise RegionalFixtureError("workload quiescence was not proven")
        except Exception as exc:
            result[f"{label}_cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["workload_cleanup_deferred"] = True
            errors.append(f"{label}: workflow not quiescent; workload left in place")
            return errors
    else:
        try:
            state = workload_store(regional, job_id=job_id, attempt_id=attempt_id)
            commands = state.get("commands")
            workflows = state.get("workflows")
            if (
                not isinstance(commands, list)
                or not isinstance(workflows, list)
                or any(
                    item.get("status")
                    not in workload_case.CLEANUP_TERMINAL_COMMAND_STATUSES
                    for item in commands
                )
                or any(
                    item.get("status")
                    not in workload_case.CLEANUP_TERMINAL_WORKFLOW_STATUSES
                    for item in workflows
                )
            ):
                raise RegionalFixtureError(
                    "workload has unknown or active control records"
                )
        except Exception as exc:
            result[f"{label}_cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["workload_cleanup_deferred"] = True
            return [f"{label}: control records not quiescent; workload left in place"]
    try:
        workload.delete()
        residuals = workload_residuals(regional, workload, job_id)
        result[f"{label}_workload_residuals"] = residuals
        if residuals:
            errors.append(f"{label}: workload resources remain after deletion")
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return errors


def run_workload_baseline(
    *,
    case_id: str,
    site: WorkloadSite,
    target: SiteTarget,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    state_dir: Path | None,
    attempt: int,
    skip_admin_status: bool = False,
) -> dict[str, Any]:
    regional = site.regional(target)
    if regional.gpu_workloads():
        raise RegionalFixtureError("target cluster already has a GPU workload")
    run_status, status_reason = admin_status_policy(case_id, skip=skip_admin_status)
    if run_status and state_dir is None:
        raise WorkloadAcceptanceError(
            f"{case_id} requires --state-dir for admin status"
        )
    baseline = workload_store(regional, job_id=job_id, attempt_id=attempt_id)
    assert_clean_identity_baseline([baseline], job_id=job_id, attempt_id=attempt_id)
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=case_id,
        run_id=f"{case_id.lower()}-{attempt}",
    )
    source_manifest = BASELINE_MANIFEST
    rendered_manifest = case_dir / "managed-workload.yaml"
    fixture: ReadmeWorkloadFixture | None = None
    result: dict[str, Any] = {"verdict": "FAIL"}
    failure: BaseException | None = None
    deletion_verified = False
    try:
        prewarm.create(prewarm_nodes(regional))
        not_applicable: dict[str, str] = {}
        submission: dict[str, Any]
        render_equivalence: bool | None = None
        fixture = readme_fixture(
            regional,
            manifest=source_manifest,
            site_file=site.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            case_dir=case_dir,
        )
        if case_id == "GF-REGIONAL-WORKLOAD-001":
            # `submit` raises on a non-zero exit, so reaching this line is
            # the returncode-0 fact the check below records.
            submission = {**fixture.submit(), "returncode": 0}
            not_applicable["render_contract_equivalent"] = (
                "WORKLOAD-001 submits through gpu-training-submit directly; the "
                "annotate/apply render comparison is WORKLOAD-002's check"
            )
        else:
            submission_arguments = fixture.submission_arguments()
            annotate = run(
                [
                    training_cli("gpu-fault-workload-annotate"),
                    *submission_arguments,
                    "--output",
                    str(rendered_manifest),
                ],
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=False,
            )
            dry_run = run(
                [
                    training_cli("gpu-training-submit"),
                    *submission_arguments,
                    "--dry-run",
                ],
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=False,
            )
            if annotate.returncode or dry_run.returncode:
                raise RegionalFixtureError("annotate or submit dry-run failed")
            render_equivalence = (
                rendered_manifest.read_text(encoding="utf-8") == dry_run.stdout
            )
            if not render_equivalence:
                raise RegionalFixtureError("annotate and submit dry-run differ")
            created = fixture.apply_rendered(rendered_manifest)
            submission = {
                **created,
                "preexisting_resource_refused": True,
                "annotate_returncode": annotate.returncode,
                "dry_run_returncode": dry_run.returncode,
            }
        finite = wait_finite_workload(fixture)
        training_logs = {
            str(item["name"]): regional.kubectl(
                "gpu",
                "logs",
                str(item["name"]),
                "--tail=400",
                timeout=60,
            )
            for item in finite["pods"]
        }
        losses = loss_errors(training_logs)
        collectives = collective_errors(training_logs)
        workload = fixture.workload()
        profile_version = str(site.config["runtime_profile"]["version"])
        metadata = metadata_errors(
            workload,
            job_id=job_id,
            attempt_id=attempt_id,
            profile_version=profile_version,
        )
        terminal = wait_terminal_observation(
            regional,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        terminal_observation = terminal["observations"][0]
        rank_errors = observation_rank_errors(
            terminal_observation, expected_ranks={0, 1, 2}
        )
        decision = terminal.get("decision") or {}
        terminal_event_key = f"{target.cluster_id}/{attempt_id}/TrainingAttemptTerminal"
        fixture.delete()
        if workload_residuals(regional, fixture, job_id):
            raise RegionalFixtureError("workload resources remain after deletion")
        deletion_verified = True
        time.sleep(watcher_interval_seconds(regional) * 3)
        after_delete = workload_store(
            regional,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        observations = after_delete["observations"]
        checks: dict[str, bool] = {
            "submission_succeeded": int(submission["returncode"]) == 0,
            "managed_metadata_complete": not metadata,
            "three_pods_on_three_nodes": len(finite["pods"]) == 3
            and len({item["node"] for item in finite["pods"]}) == 3,
            "nccl_all_reduce_succeeded": not collectives,
            "per_rank_loss_non_increasing": not losses,
            "terminal_observation_succeeded": (
                len(terminal["observations"]) == 1
                and terminal_observation["workload_phase"] == "SUCCEEDED"
                and terminal_observation.get("expected_critical_ranks") == 3
                and observation_pod_uids(terminal_observation)
                == {str(item["uid"]) for item in finite["pods"]}
            ),
            "observation_ranks_are_0_1_2": not rank_errors,
            "terminal_event_key_deterministic": (
                decision.get("event_key") == terminal_event_key
            ),
            "deletion_does_not_regress_observation": (
                len(observations) == 1
                and observations[0]["workload_phase"] == "SUCCEEDED"
                and observations[0].get("containers")
                == terminal_observation.get("containers")
                and observations[0].get("expected_critical_ranks") == 3
            ),
            "no_remote_mutation_commands": not terminal["commands"],
        }
        if render_equivalence is not None:
            checks["render_contract_equivalent"] = render_equivalence
        status: dict[str, Any]
        if run_status:
            status = admin_status(cast(Path, state_dir))
            checks["admin_status_passed"] = status["returncode"] == 0
        else:
            status = {"skipped": True, "reason": status_reason}
            not_applicable["admin_status_passed"] = status_reason
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "not_applicable": not_applicable,
            "submission": {
                key: value
                for key, value in submission.items()
                if key not in {"stdout", "stderr"}
            },
            "metadata_errors": metadata,
            "loss_errors": losses,
            "collective_errors": collectives,
            "observation_rank_errors": rank_errors,
            "workload": finite,
            "terminal": terminal,
            "after_delete": after_delete,
            "admin_status": status,
        }
    except Exception as exc:
        failure = exc
    finally:
        # Each step in its own guard: a TimeoutExpired from the workload delete
        # used to skip the prewarm cleanup and replace the original error.
        cleanup_errors: list[str] = []
        if fixture is not None and not deletion_verified:
            cleanup_errors.extend(
                cleanup_workload(
                    regional=regional,
                    workload=fixture,
                    case_dir=case_dir,
                    job_id=job_id,
                    attempt_id=attempt_id,
                    injection=None,
                    result=result,
                    label="workload",
                )
            )
        try:
            residuals = prewarm.cleanup()
            result["prewarm_residuals"] = residuals
            if any(residuals.values()):
                cleanup_errors.append("prewarm resources remain")
        except Exception as exc:
            cleanup_errors.append(f"prewarm: {type(exc).__name__}: {exc}")
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["verdict"] = "FAIL"
    raise_with_outcome(failure, result)
    result["limitations"] = [
        "The baseline proves the managed metadata and Completion Watcher path "
        "for the supplied three-node PyTorchJob; it does not inject a fault."
    ]
    return result


def post_observation(
    regional: RegionalLiveFixture,
    payload: dict[str, Any],
) -> dict[str, Any]:
    # One attempt: the sink is max_attempts=1 and a kubectl retry after a lost
    # receipt would post the observation twice.
    return regional.executor_python(
        OBSERVATION_POST_PROBE,
        json.dumps(payload, sort_keys=True),
        attempts=1,
    )


def workload_case_settings(
    *,
    regional: RegionalLiveFixture,
    site_file: Path,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> workload_case.Settings:
    return workload_case.Settings(
        regional=regional.settings,
        site_file=site_file,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
        predecessor_path=Path("/dev/null"),
    )


def virtualize_observation(
    observation: dict[str, Any],
    *,
    node_id: str,
) -> dict[str, Any]:
    value = json.loads(json.dumps(observation))
    value["observed_at"] = utc_now()
    value["workload_phase"] = "RUNNING"
    for container in value.get("containers") or []:
        container["node_id"] = node_id
    return cast(dict[str, Any], value)


def refresh_observation(observation: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(observation))
    value["observed_at"] = utc_now()
    return cast(dict[str, Any], value)


def iso001_recovery_outcome(
    *,
    cluster_ids: list[str],
    virtual_states: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    shared_node: str,
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    regional_b: RegionalLiveFixture,
    injected_at: datetime,
    target_a: dict[str, Any],
    target_b: dict[str, Any],
    b_uids: set[str],
    errors: list[str],
    baseline_states: list[dict[str, Any]],
    virtual_results: list[dict[str, Any]],
    injection: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    workflow_a = state_a.get("workflow") or {}
    virtual_errors = virtual_isolation_errors(
        virtual_states,
        sources=sources,
        cluster_ids=cluster_ids,
        shared_node=shared_node,
    )
    incident_a = state_a.get("incident") or {}
    # Every command A's workflow issued; none may show up in B's executor
    # logs, which are read over the window that starts at the injection.
    command_ids_a = sorted(
        str(item.get("command_id") or "")
        for item in state_a.get("commands") or []
        if item.get("command_id")
    )
    b_executor_logs = executor_logs_since(regional_b, since=injected_at)
    leaked_command_ids = command_ids_in_logs(command_ids_a, b_executor_logs)
    checks = {
        "two_physical_registrations": True,
        "same_virtual_node_is_cluster_scoped": not virtual_errors,
        "pre_fault_decision_and_budget_absent_on_both": all(
            item.get("decision") is None and item.get("restart_budget") is None
            for item in virtual_states
        ),
        "primary_workflow_succeeded": not errors,
        "primary_incident_and_workflow_cluster_scoped": (
            incident_a.get("cluster_id") == cluster_ids[0]
            and incident_a.get("job_id") == job_id
            and bool(workflow_a.get("request_id"))
        ),
        "primary_restart_budget_one": (
            (state_a.get("restart_budget") or {}).get("restart_count") == 1
        ),
        "secondary_has_no_restart_budget": state_b["restart_budget"] is None,
        "secondary_has_no_decision": state_b["decision"] is None,
        "secondary_has_no_incident_or_workflow": (
            not state_b.get("incidents")
            and not state_b.get("workflows")
            and state_b.get("commands") == []
        ),
        "secondary_pod_uids_unchanged": {str(item["uid"]) for item in target_b["pods"]}
        == b_uids,
        "secondary_executor_logs_free_of_primary_command_ids": (
            bool(command_ids_a) and not leaked_command_ids
        ),
    }
    result = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "virtual_isolation_errors": virtual_errors,
        "baseline_states": baseline_states,
        "virtual_posts": virtual_results,
        "virtual_states": virtual_states,
        "injection": injection,
        "primary_state": state_a,
        "primary_target": target_a,
        "primary_command_ids": command_ids_a,
        "secondary_state": state_b,
        "secondary_executor_logs": {
            pod: {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "lines": len(text.splitlines()),
            }
            for pod, text in b_executor_logs.items()
        },
        "secondary_leaked_command_ids": leaked_command_ids,
        "incident_counts": {
            cluster_ids[0]: len(state_a.get("incidents") or [])
            or int(bool(incident_a)),
            cluster_ids[1]: len(state_b.get("incidents") or []),
        },
        "workflow_counts": {
            cluster_ids[0]: len(state_a.get("workflows") or [])
            or int(bool(workflow_a)),
            cluster_ids[1]: len(state_b.get("workflows") or []),
        },
    }
    return result


def prepare_iso001_workload(
    item: tuple[RegionalLiveFixture, ManagedWorkloadFixture, ImagePrewarmFixture],
    *,
    fixtures: tuple[RegionalLiveFixture, ...],
    submission_started: list[bool],
    site_file: Path,
    job_id: str,
    attempt_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    regional, workload, prewarm = item
    prewarm.create(prewarm_nodes(regional))
    submission_started[fixtures.index(regional)] = True
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    observation = workload_case.wait_observation(
        regional,
        workload_case_settings(
            regional=regional,
            site_file=site_file,
            manifest=LONG_RUNNING_MANIFEST,
            job_id=job_id,
            attempt_id=attempt_id,
        ),
        node=str(source["pods"][0]["node"]),
        expected_gpu_count=24,
    )
    if source_observation_errors(
        observation,
        source,
        cluster_id=regional.settings.cluster_id,
        job_id=job_id,
        attempt_id=attempt_id,
    ):
        raise RegionalFixtureError("source Observation is not fresh and workload-bound")
    return source, observation


def run_iso001(
    *,
    site: WorkloadSite,
    primary: SiteTarget,
    secondary: SiteTarget,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    attempt: int,
    maintenance_window_end: datetime,
) -> dict[str, Any]:
    multi = site.multi(primary, secondary)
    regional_a = multi.regional(multi.cluster_a)
    regional_b = multi.regional(multi.cluster_b)
    registrations = registration_snapshot(regional_a, multi)
    if not registrations_are_distinct_physical_clusters(registrations):
        raise RegionalFixtureError("ISO-001 requires two physical clusters")
    fixtures = (regional_a, regional_b)
    workloads = tuple(
        managed_fixture(
            regional,
            manifest=LONG_RUNNING_MANIFEST,
            site_file=site.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        for regional in fixtures
    )
    prewarms = tuple(
        ImagePrewarmFixture(
            regional,
            case_id="GF-REGIONAL-ISO-001",
            run_id=f"iso001-{index}-{attempt}",
        )
        for index, regional in enumerate(fixtures)
    )
    cluster_ids = [regional.settings.cluster_id for regional in fixtures]
    result: dict[str, Any] = {"verdict": "FAIL", "errors": []}
    submission_started = [False, False]
    failure: BaseException | None = None
    injection_context: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []
    virtual_pending = [False, False]
    try:
        for regional in fixtures:
            if regional.gpu_workloads():
                raise RegionalFixtureError("target cluster already has GPU workloads")
        # Spec step 5's empty baseline, taken before anything is submitted:
        # a budget, decision or observation left by an earlier run under this
        # identity would be read as this run's isolation result.
        baseline_states = [
            workload_store(regional, job_id=job_id, attempt_id=attempt_id)
            for regional in fixtures
        ]
        assert_clean_identity_baseline(
            baseline_states, job_id=job_id, attempt_id=attempt_id
        )
        result["baseline_states"] = baseline_states

        prepare = partial(
            prepare_iso001_workload,
            fixtures=fixtures,
            submission_started=submission_started,
            site_file=site.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
        )

        # The two clusters' prewarm/submit/wait chains share nothing, and each
        # is the longest stretch of the case; run them together.
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(
                pool.map(prepare, zip(fixtures, workloads, prewarms, strict=True))
            )
        sources = [source for source, _observation in prepared]
        observations = [observation for _source, observation in prepared]
        shared_node = f"iso-collision-node-{attempt}"
        virtual_results = []
        for index, (regional, observation) in enumerate(
            zip(fixtures, observations, strict=True)
        ):
            virtual_pending[index] = True
            virtual_results.append(
                post_observation(
                    regional,
                    virtualize_observation(observation, node_id=shared_node),
                )
            )
        virtual_states = [
            workload_store(
                regional,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            for regional in fixtures
        ]
        virtual_errors = virtual_isolation_errors(
            virtual_states,
            sources=sources,
            cluster_ids=cluster_ids,
            shared_node=shared_node,
        )
        if virtual_errors:
            result["virtual_isolation_errors"] = virtual_errors
            raise RegionalFixtureError("virtual observation isolation failed")
        for index, (regional, observation) in enumerate(
            zip(fixtures, observations, strict=True)
        ):
            post_observation(regional, refresh_observation(observation))
            virtual_pending[index] = False
        refreshed = [
            workload_case.wait_observation(
                regional,
                workload_case_settings(
                    regional=regional,
                    site_file=site.site_file,
                    manifest=LONG_RUNNING_MANIFEST,
                    job_id=job_id,
                    attempt_id=attempt_id,
                ),
                node=str(source["pods"][0]["node"]),
                expected_gpu_count=24,
            )
            for regional, source in zip(fixtures, sources, strict=True)
        ]
        b_uids = {str(item["uid"]) for item in sources[1]["pods"]}
        if any(
            source_observation_errors(
                observation,
                source,
                cluster_id=cluster_id,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            for observation, source, cluster_id in zip(
                refreshed, sources, cluster_ids, strict=True
            )
        ):
            raise RegionalFixtureError(
                "physical Observation was not restored before injection"
            )
        injected_at = datetime.now(timezone.utc)
        if injected_at >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        node_a = str(sources[0]["pods"][0]["node"])
        marker = f"iso001-{attempt}-{int(time.time())}"
        payload = workload_case.xid11_payload(
            workload_case_settings(
                regional=regional_a,
                site_file=site.site_file,
                manifest=LONG_RUNNING_MANIFEST,
                job_id=job_id,
                attempt_id=attempt_id,
            ),
            case_id="GF-REGIONAL-ISO-001",
            marker=marker,
            node=node_a,
            product=workload_case.normalize_product(
                regional_a.node_metadata(node_a).get("product")
            ),
            observation=refreshed[0],
            observed_at=injected_at,
        )
        injection_context = {
            "node": node_a,
            "marker": marker,
            "observed_after": injected_at,
        }
        injection = regional_a.post_xid_event(payload)
        state_a = regional_a.wait_for_workflow(
            node=node_a,
            marker=marker,
            observed_after=injected_at,
            case_dir=case_dir / "cluster-a",
            timeout_seconds=1200,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        workflow_a = state_a.get("workflow") or {}
        if workflow_a.get("request_id"):
            injection_context["workflow_request_ids"] = [str(workflow_a["request_id"])]
        workloads[0].authorize_restart(state_a)
        errors = workload_case.workflow_errors(state_a, expected_gpu_count=24)
        errors.extend(
            recovery_identity_errors(
                state_a,
                cluster_id=cluster_ids[0],
                job_id=job_id,
                attempt_id=attempt_id,
                node=node_a,
                marker=marker,
                observed_after=injected_at,
            )
        )
        target_a = workloads[0].wait_restarted(
            {str(item["uid"]) for item in sources[0]["pods"]},
            timeout_seconds=900,
        )
        target_b = wait_pod_uids_unchanged(
            workloads[1],
            b_uids,
            timeout_seconds=120,
        )
        state_b = workload_store(
            regional_b,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        result = iso001_recovery_outcome(
            cluster_ids=cluster_ids,
            virtual_states=virtual_states,
            sources=sources,
            shared_node=shared_node,
            state_a=state_a,
            state_b=state_b,
            regional_b=regional_b,
            injected_at=injected_at,
            target_a=target_a,
            target_b=target_b,
            b_uids=b_uids,
            errors=errors,
            baseline_states=baseline_states,
            virtual_results=virtual_results,
            injection=injection,
            job_id=job_id,
        )
    except Exception as exc:
        failure = exc
    finally:
        cleanup_errors: list[str] = []
        if injection_context is None:
            for index, pending in enumerate(virtual_pending):
                if not pending:
                    continue
                try:
                    post_observation(
                        fixtures[index], refresh_observation(observations[index])
                    )
                except Exception as exc:
                    cleanup_errors.append(
                        f"physical Observation restore: {type(exc).__name__}"
                    )
        # Cluster A's workflow may still be RUNNING when the body fails; its
        # workload is deleted only once quiescence is proven. Cluster B never
        # had a workflow, so its workload goes straight away.
        for index, (regional, workload) in enumerate(zip(fixtures, workloads)):
            if not submission_started[index]:
                continue
            cleanup_errors.extend(
                cleanup_workload(
                    regional=regional,
                    workload=workload,
                    case_dir=case_dir / ("cluster-a" if index == 0 else "cluster-b"),
                    job_id=job_id,
                    attempt_id=attempt_id,
                    injection=injection_context if index == 0 else None,
                    result=result,
                    label=f"cluster_{'ab'[index]}",
                )
            )
        prewarm_residuals = []
        for prewarm in prewarms:
            try:
                prewarm_residuals.append(prewarm.cleanup())
            except Exception as exc:
                cleanup_errors.append(f"prewarm: {type(exc).__name__}: {exc}")
        result["cleanup_errors"] = cleanup_errors
        result["prewarm_residuals"] = prewarm_residuals
        if cleanup_errors or any(any(item.values()) for item in prewarm_residuals):
            result["verdict"] = "FAIL"
    raise_with_outcome(failure, result)
    result["limitations"] = [
        "The fault is an authenticated software replay into cluster A; it proves "
        "cluster-scoped state and restart behavior, not a hardware-originated XID.",
        "GPU health findings are not asserted: the XID path may leave both "
        "clusters without one, which the case allows.",
    ]
    return result


def e2e_executor_pods(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    inventory = json.loads(
        regional.kubectl(
            "gpu", "get", "pod", "-l", "app=gpu-fault-cluster-executor", "-o", "json"
        )
    )
    pods = ready_pod_records(inventory)
    if len(pods) != len(inventory["items"]):
        raise RegionalFixtureError("not every executor Pod is fully Ready")
    if len({pod["name"] for pod in pods}) != len(pods) or len(
        {pod["uid"] for pod in pods}
    ) != len(pods):
        raise RegionalFixtureError("executor Pod identities are not unique")
    raw_by_uid = {item["metadata"]["uid"]: item for item in inventory["items"]}
    for pod in pods:
        raw = raw_by_uid[pod["uid"]]
        if (
            raw["metadata"].get("namespace") != regional.settings.namespace
            or not isinstance(pod["node"], str)
            or not pod["node"].strip()
        ):
            raise RegionalFixtureError("executor Pod namespace or node is missing")
        containers = []
        for status in raw["status"]["containerStatuses"]:
            running = (status.get("state") or {}).get("running") or {}
            fields = {
                "name": status["name"],
                "container_id": status.get("containerID"),
                "image_id": status.get("imageID"),
                "started_at": running.get("startedAt"),
            }
            restarts = status.get("restartCount")
            if (
                any(
                    not isinstance(value, str) or not value.strip()
                    for value in fields.values()
                )
                or type(restarts) is not int
                or restarts < 0
            ):
                raise RegionalFixtureError("executor container identity is incomplete")
            containers.append({**fields, "restart_count": restarts})
        if not any(item["name"] == "executor" for item in containers):
            raise RegionalFixtureError("executor container is missing")
        pod["containers"] = sorted(containers, key=lambda item: item["name"])
    return pods


def e2e_preflight(
    regional: RegionalLiveFixture,
) -> dict[str, Any]:
    state = regional.cpu_python(
        E2E_PREFLIGHT_PROBE,
        regional.settings.cluster_id,
    )
    metadata = json.loads(
        regional.kubectl(
            "cpu",
            "get",
            "configmap",
            "gpu-fault-release-metadata",
            "-o",
            "json",
        )
    )["data"]
    nodes = regional.gpu_nodes()
    status_by_node: dict[str, set[str]] = {}
    for item in state["collector_statuses"]:
        # CollectorStatus names its source ``collector``; older snapshots said
        # ``kind`` (live 2026-09-07: E2E-001 died here before any check ran).
        status_by_node.setdefault(str(item["node_id"]), set()).add(
            str(item.get("collector") or item.get("kind"))
        )
    required_kinds = {"NVIDIA_KERNEL", "GPU_METRICS", "HOST_TELEMETRY"}
    agents = state["agents"]
    active_agents = [
        item
        for item in agents
        if item.get("lifecycle_state") == "ACTIVE"
        and item.get("lease_expires_at")
        and datetime.fromisoformat(str(item["lease_expires_at"]).replace("Z", "+00:00"))
        > datetime.now(timezone.utc)
    ]
    required_artifact = metadata.get("required-agent-artifact-sha256")
    executor_logs = []
    suspicious: list[dict[str, Any]] = []
    executor_pods = e2e_executor_pods(regional)
    claim_probe = (
        Path(__file__).with_name("probes") / "e2e_executor_readiness.py"
    ).read_text(encoding="utf-8")
    for pod in executor_pods:
        text = regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            "-c",
            "executor",
            "--since=10m",
            check=True,
            timeout=120,
        )
        container = next(
            item for item in pod["containers"] if item["name"] == "executor"
        )
        readiness = json.loads(
            regional.kubectl(
                "gpu",
                "exec",
                "-i",
                str(pod["name"]),
                "-c",
                "executor",
                "--",
                component_python("gpu"),
                "-",
                regional.settings.cluster_id,
                str(pod["name"]),
                container["started_at"],
                input_text=claim_probe,
                check=True,
                timeout=60,
            )
        )
        if (
            not isinstance(readiness, dict)
            or readiness.get("authenticated_ready") is not True
            or readiness.get("cluster_id") != regional.settings.cluster_id
            or readiness.get("pod") != pod["name"]
            or not isinstance(readiness.get("claim_before"), dict)
            or not isinstance(readiness.get("claim_after"), dict)
            or not readiness["claim_before"]
            or not readiness["claim_after"]
        ):
            raise RegionalFixtureError("executor readiness evidence is incomplete")
        executor_logs.append(
            {
                "pod": pod["name"],
                "uid": pod["uid"],
                "read_succeeded": True,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "lines": len(text.splitlines()),
                "readiness": readiness,
            }
        )
        # Word-boundary matches with the offending lines recorded: the old
        # substring scan flagged `error_count=0` and any 401/403 digit run,
        # and told the operator nothing about what it had seen.
        suspicious.extend(
            {"pod": pod["name"], **hit} for hit in suspicious_log_lines(text)
        )
    if e2e_executor_pods(regional) != executor_pods:
        raise RegionalFixtureError("executor Pod or container identity changed")
    errors = []
    if not nodes:
        errors.append("GPU node inventory is empty")
    if not executor_logs:
        errors.append("no Ready executor was observed")
    for node in nodes:
        # A spare-pool node is cordoned by design while it waits in the pool
        # (hyperpod_spares keeps spec.unschedulable=True until it is handed
        # over), so only its readiness is a preflight condition.
        spare = (node.get("labels") or {}).get("gpu-fault.io/spare") == "true"
        if node["ready"] != "True" or (node["unschedulable"] and not spare):
            errors.append(f"{node['name']} is not Ready and schedulable")
        if not required_kinds <= status_by_node.get(str(node["name"]), set()):
            errors.append(f"{node['name']} lacks required Collector status")
    expected_nodes = {str(item["name"]) for item in nodes}
    if (
        len(active_agents) != len(expected_nodes)
        or {str(item.get("node_id") or "") for item in active_agents} != expected_nodes
    ):
        errors.append("not every GPU node has a live ACTIVE Agent")
    if not required_artifact:
        errors.append("required Agent artifact is missing from release metadata")
    if any(item.get("artifact_sha256") != required_artifact for item in active_agents):
        errors.append("an ACTIVE Agent artifact differs from release metadata")
    if suspicious:
        errors.append("executor logs contain an authentication/runtime error")
    return {
        "nodes": nodes,
        "collector_kinds_by_node": {
            key: sorted(value) for key, value in status_by_node.items()
        },
        "active_agents": active_agents,
        "required_agent_artifact_sha256": required_artifact,
        "executor_pods": executor_pods,
        "executor_logs": executor_logs,
        "suspicious_executor_logs": suspicious,
        "errors": errors,
    }


def e2e001_recovery_outcome(
    *,
    regional: RegionalLiveFixture,
    target: SiteTarget,
    preflight: dict[str, Any],
    blast_before: dict[str, Any],
    source: dict[str, Any],
    workload: ManagedWorkloadFixture,
    observation: dict[str, Any],
    host: dict[str, Any],
    injection: dict[str, Any],
    state: dict[str, Any],
    injection_context: dict[str, Any],
    job_id: str,
    attempt_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    node = str(injection_context["node"])
    marker = str(injection_context["marker"])
    injected_at = injection_context["observed_after"]
    workflow = state.get("workflow") or {}
    errors = workload_case.workflow_errors(state, expected_gpu_count=24)
    errors.extend(
        recovery_identity_errors(
            state,
            cluster_id=target.cluster_id,
            job_id=job_id,
            attempt_id=attempt_id,
            node=node,
            marker=marker,
            observed_after=injected_at,
        )
    )
    errors.extend(
        wait_for_notification_results(
            state,
            snapshot=lambda: regional.store_snapshot(
                node=node,
                marker=marker,
                observed_after=injected_at,
                job_id=job_id,
                attempt_id=attempt_id,
                queue_attempts=1,
            ),
        )
    )
    event = state.get("event") or {}
    incident = state.get("incident") or {}
    if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
        errors.append("event evidence reference is not a real kmsg reference")
    if not event.get("source_boot_id") or event.get("source_monotonic_us") is None:
        errors.append("event lacks real boot ID or monotonic source timestamp")
    elif event["source_boot_id"] != host.get("boot_id"):
        errors.append("event boot ID differs from the probed host")
    if (
        type(event.get("source_monotonic_us")) is not int
        or event["source_monotonic_us"] < 0
        or not event.get("pci_bdf")
        or str(event["pci_bdf"]).lower().removesuffix(".0")
        != str(host.get("gpu_bdf") or "").lower().removesuffix(".0")
        or event.get("gpu_uuid") not in workload_case.observation_gpu_uuids(observation)
    ):
        errors.append(
            "kernel event source clock or GPU identity differs from injection"
        )
    evidence = regional.cpu_python(
        KMSG_EVIDENCE_PROBE, target.cluster_id, str(event.get("evidence_ref") or "")
    )
    records = evidence.get("evidence") or []
    if len(records) != 1:
        errors.append("injection does not bind exactly one raw kernel evidence record")
    errors.extend(
        kmsg_identity_errors(
            records,
            boot_id=str(host.get("boot_id") or ""),
            node=node,
            cluster_id=target.cluster_id,
        )
    )
    # "Catalog 返回 RESTART_APP 且 incident 与 recovery plan 的 cluster_id
    # 正确": the plan is the workflow's incident scope.
    scope_errors = []
    if incident.get("cluster_id") != target.cluster_id:
        errors.append("incident cluster_id is not the target cluster")
        scope_errors.append(f"incident: {incident.get('cluster_id')!r}")
    if workflow.get("incident_id") != incident.get("incident_id"):
        errors.append("recovery plan does not belong to the incident")
        scope_errors.append("workflow.incident_id differs")
    target_workload = workload.wait_restarted(
        {str(item["uid"]) for item in source["pods"]},
        timeout_seconds=900,
    )
    source_attempts = {str(item.get("attempt_id") or "") for item in source["pods"]}
    target_attempts = {
        str(item.get("attempt_id") or "") for item in target_workload["pods"]
    }
    blast_after = regional.cpu_blast_snapshot()
    blast_errors = control_plane_blast_errors(blast_before, blast_after)
    nodes_after = regional.gpu_nodes()
    checks = {
        "collector_agent_executor_preflight": not preflight["errors"],
        "real_kmsg_injection": injection["bytes_written"] > 0,
        "workflow_contract": not errors,
        "incident_and_plan_cluster_scoped": not scope_errors,
        "workload_pod_uids_changed": {
            str(item["uid"]) for item in target_workload["pods"]
        }.isdisjoint({str(item["uid"]) for item in source["pods"]}),
        # The new attempt is a different, single, non-empty ID on every
        # replaced Pod; UID disjointness alone also holds for a plain Pod
        # restart of the same attempt.
        "workload_attempt_id_changed": (
            source_attempts == {attempt_id}
            and len(target_attempts) == 1
            and bool(next(iter(target_attempts)))
            and target_attempts.isdisjoint(source_attempts)
        ),
        "control_plane_eks_identical": not blast_errors,
        "gpu_nodes_restored": gpu_nodes_clean(nodes_after),
    }
    result = {
        **result,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "scope_errors": scope_errors,
        "preflight": preflight,
        "source_workload": source,
        "target_workload": target_workload,
        "source_attempt_ids": sorted(source_attempts),
        "target_attempt_ids": sorted(target_attempts),
        "injection": injection,
        "state": state,
        "kernel_evidence": evidence,
        "control_plane_eks_diff": "IDENTICAL" if not blast_errors else "CHANGED",
        "control_plane_eks_errors": blast_errors,
        "gpu_nodes_after_restart": nodes_after,
    }
    # Consumed by BLAST-001 (blast_acceptance_cases_1): maintenance_window
    # .start/.end and control-plane-current.json's workflows[] with their
    # official_steps/safety_steps/step_executions. Keep these keys.
    return result


def run_e2e001(
    *,
    site: WorkloadSite,
    target: SiteTarget,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    host_probe_image: str,
    attempt: int,
    maintenance_window_end: datetime,
) -> dict[str, Any]:
    regional = site.regional(target)
    identity = regional.evidence_identity()
    preflight = e2e_preflight(regional)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "E2E-001 preflight failed: " + "; ".join(preflight["errors"])
        )
    if regional.gpu_workloads():
        raise RegionalFixtureError("target cluster already has a GPU workload")
    baseline_state = workload_store(regional, job_id=job_id, attempt_id=attempt_id)
    assert_clean_identity_baseline(
        [baseline_state], job_id=job_id, attempt_id=attempt_id
    )
    cpu_nodes_before = json.loads(regional.kubectl("cpu", "get", "node", "-o", "json"))
    write_json_atomic(case_dir / "cpu-nodes-before.json", cpu_nodes_before)
    blast_before = regional.cpu_blast_snapshot()
    prewarm = ImagePrewarmFixture(
        regional,
        case_id="GF-REGIONAL-E2E-001",
        run_id=f"e2e001-{attempt}",
    )
    workload = managed_fixture(
        regional,
        manifest=LONG_RUNNING_MANIFEST,
        site_file=site.site_file,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    probe: HostProbeFixture | None = None
    result: dict[str, Any] = {
        **identity,
        "verdict": "FAIL",
        "errors": [],
        "baseline_state": baseline_state,
    }
    failure: BaseException | None = None
    injection_context: dict[str, Any] | None = None
    submission_started = False
    try:
        prewarm.create(prewarm_nodes(regional))
        submission_started = True
        workload.submit()
        source = workload.wait_running(timeout_seconds=900)
        node = str(source["pods"][0]["node"])
        observation = workload_case.wait_observation(
            regional,
            workload_case_settings(
                regional=regional,
                site_file=site.site_file,
                manifest=LONG_RUNNING_MANIFEST,
                job_id=job_id,
                attempt_id=attempt_id,
            ),
            node=node,
            expected_gpu_count=24,
        )
        if source_observation_errors(
            observation,
            source,
            cluster_id=target.cluster_id,
            job_id=job_id,
            attempt_id=attempt_id,
        ):
            raise RegionalFixtureError(
                "source Observation is not fresh and workload-bound"
            )
        expected_nodes = {(item["name"], item["uid"]) for item in preflight["nodes"]}
        if (
            node not in {name for name, _uid in expected_nodes}
            or {(item["name"], item["uid"]) for item in regional.gpu_nodes()}
            != expected_nodes
        ):
            raise RegionalFixtureError("GPU node identity changed before injection")
        probe = HostProbeFixture(
            HostProbeSettings(
                state_directory=case_dir / "host-probes",
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=host_probe_image,
                case_id="GF-REGIONAL-E2E-001",
                run_id=f"e2e001-{attempt}-{int(time.time())}",
                probe_script=E2E001_PROBE,
                active_deadline_seconds=1800,
            )
        )
        probe.create()
        host = probe.execute("snapshot")
        if (
            not host["kmsg_writable"]
            or host["kernel_collector"].get("ActiveState") != "active"
        ):
            raise RegionalFixtureError("kernel Collector host preflight failed")
        marker = f"e2e001-{attempt}-{int(time.time())}"
        injected_at = datetime.now(timezone.utc)
        if injected_at >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        injection_context = {
            "node": node,
            "marker": marker,
            "observed_after": injected_at,
        }
        injection = probe.execute(
            "write-xid11",
            "--marker",
            marker,
            "--pci-bdf",
            str(host["gpu_bdf"]),
        )
        state = regional.wait_for_workflow(
            node=node,
            marker=marker,
            observed_after=injected_at,
            case_dir=case_dir,
            timeout_seconds=1200,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        workflow = state.get("workflow") or {}
        result["state"] = state
        if workflow.get("request_id"):
            injection_context["workflow_request_ids"] = [str(workflow["request_id"])]
        workload.authorize_restart(state)
        result = e2e001_recovery_outcome(
            regional=regional,
            target=target,
            preflight=preflight,
            blast_before=blast_before,
            source=source,
            workload=workload,
            observation=observation,
            host=host,
            injection=injection,
            state=state,
            injection_context=injection_context,
            job_id=job_id,
            attempt_id=attempt_id,
            result=result,
        )
    except Exception as exc:
        failure = exc
    finally:
        cleanup_errors: list[str] = []
        if probe is not None:
            try:
                probe_residuals = probe.cleanup()
                result["probe_residuals"] = probe_residuals
                if any(probe_residuals.values()):
                    cleanup_errors.append("host probe resources remain")
            except Exception as exc:
                cleanup_errors.append(f"probe: {type(exc).__name__}: {exc}")
        if submission_started:
            cleanup_errors.extend(
                cleanup_workload(
                    regional=regional,
                    workload=workload,
                    case_dir=case_dir,
                    job_id=job_id,
                    attempt_id=attempt_id,
                    injection=injection_context,
                    result=result,
                    label="workload",
                )
            )
        try:
            prewarm_residuals = prewarm.cleanup()
            result["prewarm_residuals"] = prewarm_residuals
            if any(prewarm_residuals.values()):
                cleanup_errors.append("prewarm resources remain")
        except Exception as exc:
            cleanup_errors.append(f"prewarm: {type(exc).__name__}: {exc}")
        # "清理后节点恢复 Ready 与可调度" is judged after the cleanup, not on the
        # snapshot taken while the workload was still being replaced.
        try:
            nodes_after_cleanup = regional.gpu_nodes()
            result["gpu_nodes_after_cleanup"] = nodes_after_cleanup
            restored = gpu_nodes_clean(nodes_after_cleanup) and {
                (item.get("name"), item.get("uid")) for item in nodes_after_cleanup
            } == {(item.get("name"), item.get("uid")) for item in preflight["nodes"]}
        except Exception as exc:
            cleanup_errors.append(f"node recheck: {type(exc).__name__}: {exc}")
            restored = False
        checks_after = result.setdefault("checks", {})
        checks_after["gpu_nodes_restored_after_cleanup"] = restored
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors or not restored:
            result["verdict"] = "FAIL"
        if injection_context is not None:
            state_path = case_dir / "control-plane-current.json"
            write_json_atomic(
                state_path,
                {"workflows": [(result.get("state") or {}).get("workflow") or {}]},
            )
            write_json_atomic(
                case_dir / "execution-card.json",
                {
                    "schema_version": 1,
                    "case_id": "GF-REGIONAL-E2E-001",
                    **identity,
                    "verdict": result["verdict"] if failure is None else "FAIL",
                    "cleanup_complete": not cleanup_errors and restored,
                    "maintenance_window": {
                        "start": injection_context["observed_after"].isoformat(),
                        "end": datetime.now(timezone.utc).isoformat(),
                    },
                    "approved_window_end": maintenance_window_end.isoformat(),
                    "node": injection_context["node"],
                    "job_id": job_id,
                    "attempt_id": attempt_id,
                    "baseline_sha256": hashlib.sha256(
                        (case_dir / "cpu-nodes-before.json").read_bytes()
                    ).hexdigest(),
                    "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                },
            )
    raise_with_outcome(failure, result)
    result["limitations"] = [
        "The XID line is written by an approved user-space probe into real "
        "/dev/kmsg; it validates the software chain but is not hardware damage.",
        "control_plane_eks_identical compares CPU node metadata and gpu-fault "
        "Jobs exactly and eviction events as 'none new in the window'; the raw "
        "event set is unbounded and churns on its own.",
    ]
    return result


def case_plan(
    case_id: str,
    *,
    primary: SiteTarget,
    secondary: SiteTarget | None,
    job_id: str,
    attempt_id: str,
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    mutations = {
        "GF-REGIONAL-WORKLOAD-001": (
            "submit one managed three-node PyTorchJob through gpu-training-submit"
        ),
        "GF-REGIONAL-WORKLOAD-002": (
            "render with gpu-fault-workload-annotate, server-dry-run and apply"
        ),
        "GF-REGIONAL-ISO-001": (
            "submit the same job/attempt identity to two physical clusters and "
            "restart only cluster A via authenticated XID11 replay"
        ),
        "GF-REGIONAL-E2E-001": (
            "submit a managed 24-GPU workload and write one XID11 line to real "
            "/dev/kmsg on its target node"
        ),
    }
    return {
        "risk": "case-defined",
        "predecessor": predecessor,
        "primary": {
            "cluster_id": primary.cluster_id,
            "context": primary.context,
        },
        "secondary": (
            {
                "cluster_id": secondary.cluster_id,
                "context": secondary.context,
            }
            if secondary is not None
            else None
        ),
        "job_id": job_id,
        "attempt_id": attempt_id,
        "mutation": mutations[case_id],
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "fewer than three Ready schedulable GPU nodes",
            "a target cluster already has a GPU workload",
            "attempt observation is missing, stale or has the wrong GPU count",
            "a workflow or restart budget crosses cluster scope",
            "workload, prewarm, host probe, taint or cordon cleanup is incomplete",
        ],
        "rollback": {
            "delete_test_workloads": True,
            "delete_image_prewarm_pods": True,
            "host_probe_has_active_deadline": True,
            "no_provider_node_replacement": True,
        },
    }


def failure_outcome(exc: BaseException) -> dict[str, Any]:
    """The evidence body for a case whose handler raised.

    A ``WorkloadCaseError`` carries the partial result its ``finally`` block
    annotated (cleanup_errors, residuals, a deferred workload), so the file
    shows the state the cluster was left in and not only the exception text.
    """

    partial = dict(exc.outcome) if isinstance(exc, WorkloadCaseError) else {}
    partial.pop("verdict", None)
    return {
        **partial,
        "verdict": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "limitations": [
            "The case stopped at the first failed assertion; later checks "
            "were not treated as executed.",
            *partial.get("limitations", []),
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run one guarded WORKLOAD/ISO-001/E2E-001 regional acceptance case."
        )
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--site", type=Path, required=True)
    value.add_argument("--state-dir", type=Path)
    value.add_argument("--cluster-id", default="")
    value.add_argument("--secondary-cluster-id", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--skip-admin-status",
        action="store_true",
        help=(
            "do not run gpu-fault-admin status at the end; by default only "
            f"{ADMIN_STATUS_CASE} runs it, once for the WORKLOAD group"
        ),
    )
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    site = WorkloadSite(arguments.site)
    primary = site.target(arguments.cluster_id)
    secondary = None
    if arguments.case == "GF-REGIONAL-ISO-001":
        if not arguments.secondary_cluster_id:
            raise WorkloadAcceptanceError("ISO-001 requires --secondary-cluster-id")
        secondary = site.target(arguments.secondary_cluster_id)
        if secondary.cluster_id == primary.cluster_id:
            raise WorkloadAcceptanceError(
                "ISO-001 primary and secondary clusters must differ"
            )
    runs_admin_status, _status_reason = admin_status_policy(
        arguments.case, skip=arguments.skip_admin_status
    )
    if runs_admin_status and arguments.state_dir is None:
        raise WorkloadAcceptanceError(
            f"{arguments.case} requires --state-dir for final admin status "
            "(or --skip-admin-status)"
        )
    if arguments.case == "GF-REGIONAL-E2E-001" and (
        not arguments.host_probe_image or "@sha256:" not in arguments.host_probe_image
    ):
        raise WorkloadAcceptanceError(
            "E2E-001 requires an immutable --host-probe-image"
        )
    default_job, default_attempt = derived_identity(
        arguments.run_dir,
        arguments.attempt,
        arguments.case,
    )
    job_id = arguments.job_id.strip() or default_job
    attempt_id = arguments.attempt_id.strip() or (
        f"{job_id}-a001" if arguments.job_id else default_attempt
    )
    # The release and cluster this evidence is bound to: the predecessor must
    # have earned its PASS against the same pair, and the result carries it so
    # the next case can demand the same.
    identity = site.regional(primary).evidence_identity()
    if not identity["release_id"].strip():
        raise WorkloadAcceptanceError("the deployed release identity is missing")
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id, **identity)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    confirmation = (
        arguments.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    )
    environment = {
        "GPU_FAULT_WORKLOAD_CASE": arguments.case,
        "GPU_FAULT_SITE_FILE": str(arguments.site.resolve()),
        "GPU_FAULT_CONTROL_KUBECONFIG": str(site.cpu_kubeconfig),
        "KUBECONFIG": str(site.gpu_kubeconfig),
        "GPU_FAULT_PRIMARY_CLUSTER_ID": primary.cluster_id,
        "GPU_FAULT_SECONDARY_CLUSTER_ID": (
            secondary.cluster_id if secondary is not None else ""
        ),
        "GPU_FAULT_TEST_JOB_ID": job_id,
        "GPU_FAULT_TEST_ATTEMPT_ID": attempt_id,
    }
    details = {
        **case_plan(
            arguments.case,
            primary=primary,
            secondary=secondary,
            job_id=job_id,
            attempt_id=attempt_id,
            predecessor=predecessor,
        ),
        "site_sha256": hashlib.sha256(arguments.site.read_bytes()).hexdigest(),
        "release_id": identity["release_id"],
    }
    if not arguments.execute:
        plan = build_plan(
            arguments=arguments,
            preflight_passed=predecessor.get("valid") is True,
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid") is True else 1
    if arguments.confirm != confirmation:
        raise WorkloadAcceptanceError(f"confirmation must be exactly {confirmation}")
    deadline = authorize_execution(
        arguments,
        case_id=arguments.case,
        confirmation=confirmation,
        environment=environment,
        details=details,
    )
    if predecessor.get("valid") is not True:
        raise WorkloadAcceptanceError("formal predecessor evidence is not PASS")
    case_dir = arguments.run_dir / "cases" / arguments.case
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "GF-REGIONAL-WORKLOAD-001": lambda: run_workload_baseline(
                case_id=arguments.case,
                site=site,
                target=primary,
                case_dir=case_dir,
                job_id=job_id,
                attempt_id=attempt_id,
                state_dir=arguments.state_dir,
                attempt=arguments.attempt,
                skip_admin_status=arguments.skip_admin_status,
            ),
            "GF-REGIONAL-WORKLOAD-002": lambda: run_workload_baseline(
                case_id=arguments.case,
                site=site,
                target=primary,
                case_dir=case_dir,
                job_id=job_id,
                attempt_id=attempt_id,
                state_dir=arguments.state_dir,
                attempt=arguments.attempt,
                skip_admin_status=arguments.skip_admin_status,
            ),
            "GF-REGIONAL-ISO-001": lambda: run_iso001(
                site=site,
                primary=primary,
                secondary=cast(SiteTarget, secondary),
                case_dir=case_dir,
                job_id=job_id,
                attempt_id=attempt_id,
                attempt=arguments.attempt,
                maintenance_window_end=deadline,
            ),
            "GF-REGIONAL-E2E-001": lambda: run_e2e001(
                site=site,
                target=primary,
                case_dir=case_dir,
                job_id=job_id,
                attempt_id=attempt_id,
                host_probe_image=arguments.host_probe_image,
                attempt=arguments.attempt,
                maintenance_window_end=deadline,
            ),
        }
        outcome = handlers[arguments.case]()
    except Exception as exc:
        outcome = failure_outcome(exc)
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": arguments.case,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **identity,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
