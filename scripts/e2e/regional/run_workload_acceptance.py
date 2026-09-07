from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
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
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
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
SUSPICIOUS_LOG_PATTERN = re.compile(
    r"\b(?:ERROR|Traceback|401|403|CERTIFICATE_VERIFY_FAILED)\b"
)
MAX_RECORDED_SUSPICIOUS_LINES = 20
LOSS_LINE_PATTERN = re.compile(
    r"\brank=(\d+)\b.*?\bstep=(\d+)\b.*?\bloss=([-+0-9.eE]+)"
)
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
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    if check and completed.returncode:
        raise WorkloadAcceptanceError(
            f"command failed ({completed.returncode}): {' '.join(command[:6])}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    return completed


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


WORKLOAD_STORE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

# The fourth argument arrived after the third; a caller that still passes
# three gets the job-scoped workflow read below.
argv = list(sys.argv[1:])
if len(argv) == 3:
    argv.append("")
cluster_id, job_id, attempt_id, workflow_request_ids_text = argv
store = ApplicationContext.from_environment().store
observations = [
    item.model_dump(mode="json")
    for item in store.list_attempt_observations(cluster_id)
    if item.job_id == job_id and item.attempt_id == attempt_id
]
decision = store.get_decision_by_attempt(cluster_id, attempt_id)
try:
    budget = store.get_restart_budget(cluster_id, job_id).model_dump(mode="json")
except NotFoundError:
    budget = None
# Incidents and workflows the store already scopes to this cluster and job:
# the ones still acting on the attempt and the ones that restarted into it.
# Every backend filters both in the store, so this probe never pages the
# whole workflow table -- or, as it used to, the whole remote-command table
# every five seconds -- through the API Pod.
pairs = {}
for incident, workflow in store.list_active_workflow_incidents(
    cluster_id, job_id=job_id
):
    pairs[workflow.request_id] = (incident, workflow)
for incident, workflow in store.list_job_recovery_workflow_incidents(
    cluster_id, job_id, attempt_id
):
    pairs[workflow.request_id] = (incident, workflow)
explicit_ids = [item for item in workflow_request_ids_text.split(",") if item]
request_ids = sorted(set(pairs) | set(explicit_ids))
commands = [
    item.model_dump(mode="json")
    for item in (
        store.list_remote_commands(workflow_request_ids=request_ids)
        if request_ids
        else []
    )
    if any(
        job_id in workload_id
        for workload_id in item.step.workload_ids
    )
    or item.workflow_request_id in explicit_ids
]
print(json.dumps({
    "cluster_id": cluster_id,
    "observations": observations,
    "decision": decision.model_dump(mode="json") if decision else None,
    "restart_budget": budget,
    "commands": commands,
    "incidents": [
        {
            "incident_id": incident.incident_id,
            "cluster_id": incident.cluster_id,
            "job_id": incident.job_id,
            "attempt_id": incident.attempt_id,
            "workflow_request_id": incident.workflow_request_id,
        }
        for incident, _workflow in pairs.values()
    ],
    "workflows": [
        {
            "request_id": workflow.request_id,
            "incident_id": workflow.incident_id,
            "status": workflow.status.value,
        }
        for _incident, workflow in pairs.values()
    ],
}, sort_keys=True, default=str))
"""


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


def identity_baseline_errors(state: dict[str, Any]) -> list[str]:
    """Why ``state`` is not the empty pre-fault baseline for a job/attempt.

    ISO-001 step 5 and E2E-001 both require that nothing has been recorded for
    the identity yet: a restart budget, a decision or an observation left by an
    earlier run would be read as this run's isolation result.
    """

    errors = []
    if state.get("restart_budget") is not None:
        errors.append("restart budget already exists")
    if state.get("decision") is not None:
        errors.append("attempt decision already exists")
    if state.get("observations"):
        errors.append(f"{len(state['observations'])} attempt observations exist")
    if state.get("incidents") or state.get("workflows"):
        errors.append("an incident or workflow already references the job")
    return errors


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


def metadata_errors(
    workload: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    profile_version: str,
) -> list[str]:
    errors = []
    labels = workload["metadata"].get("labels", {})
    expected_labels = {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": job_id,
        "gpu-fault.io/attempt-id": attempt_id,
    }
    for key, value in expected_labels.items():
        if labels.get(key) != value:
            errors.append(f"workload label {key} differs")
    replicas = workload["spec"]["pytorchReplicaSpecs"]
    for role, expected_offset in (("Master", "0"), ("Worker", "1")):
        template = replicas[role]["template"]["metadata"]
        template_labels = template.get("labels", {})
        annotations = template.get("annotations", {})
        for key, value in {
            **expected_labels,
            "gpu-fault.io/critical": "true",
        }.items():
            if template_labels.get(key) != value:
                errors.append(f"{role} template label {key} differs")
        if annotations.get("gpu-fault.io/expected-critical-ranks") != "3":
            errors.append(f"{role} expected critical ranks differs")
        if annotations.get("gpu-fault.io/rank-offset") != expected_offset:
            errors.append(f"{role} rank offset differs")
        if annotations.get("gpu-fault.io/runtime-profile-version") != profile_version:
            errors.append(f"{role} runtime profile differs")
        if annotations.get("gpu-fault.io/restart-budget") != "1":
            errors.append(f"{role} restart budget differs")
        if annotations.get("gpu-fault.io/training-container") != "pytorch":
            errors.append(f"{role} training container differs")
    return errors


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


def suspicious_log_lines(
    text: str, *, limit: int = MAX_RECORDED_SUSPICIOUS_LINES
) -> list[dict[str, Any]]:
    """Executor log lines that hit a spec marker on a word boundary.

    Returns the marker and the line (truncated) so the evidence shows what
    matched rather than only that something did.
    """

    hits: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = SUSPICIOUS_LOG_PATTERN.search(line)
        if match is None:
            continue
        hits.append({"line": number, "marker": match.group(0), "text": line[:300]})
        if len(hits) >= limit:
            break
    return hits


def loss_errors(logs: dict[str, str]) -> list[str]:
    """Why the fixture's per-rank ``loss=`` lines do not show a converging model.

    Every rank trains the same Linear(16, 1) on constant data with SGD, so the
    loss it prints must not increase from one step to the next; a rank whose
    loss climbs, or a Pod that printed none, is a training defect the SUCCESS
    line alone would hide.
    """

    errors = []
    for pod, text in sorted(logs.items()):
        series: dict[int, list[tuple[int, float]]] = {}
        for line in text.splitlines():
            match = LOSS_LINE_PATTERN.search(line)
            if match is None:
                continue
            try:
                value = float(match.group(3))
            except ValueError:
                errors.append(f"{pod}: unparseable loss line {line[:80]!r}")
                continue
            series.setdefault(int(match.group(1)), []).append(
                (int(match.group(2)), value)
            )
        if not series:
            errors.append(f"{pod}: no loss lines")
            continue
        for rank, points in sorted(series.items()):
            ordered = [value for _step, value in sorted(points)]
            if any(later > earlier for earlier, later in zip(ordered, ordered[1:])):
                errors.append(f"{pod}: rank {rank} loss increased: {ordered}")
    return errors


def observation_rank_errors(
    observation: dict[str, Any],
    *,
    expected_ranks: set[int],
) -> list[str]:
    """The observation's containers must cover exactly the expected ranks."""

    ranks = [container.get("rank") for container in observation.get("containers") or []]
    if any(rank is None for rank in ranks):
        return ["an observed container has no rank"]
    actual = {int(rank) for rank in ranks if rank is not None}
    if len(actual) != len(ranks):
        return [f"duplicate container ranks: {sorted(ranks)}"]
    if actual != expected_ranks:
        return [f"container ranks {sorted(actual)} != {sorted(expected_ranks)}"]
    return []


def observation_pod_names(observation: dict[str, Any]) -> set[str]:
    return {
        str(container.get("pod_name") or "")
        for container in observation.get("containers") or []
    }


def observation_pod_uids(observation: dict[str, Any]) -> set[str]:
    return {
        str(container.get("pod_uid") or "")
        for container in observation.get("containers") or []
    }


def virtual_isolation_errors(
    virtual_states: Sequence[dict[str, Any]],
    *,
    sources: Sequence[dict[str, Any]],
    cluster_ids: Sequence[str],
    shared_node: str,
) -> list[str]:
    """Why the same virtual node name did not stay cluster-scoped.

    Both clusters posted an observation naming ``shared_node``; each store
    scope must hold exactly one, stamped with its own cluster_id and carrying
    its own Pods (names, UIDs) and GPUs -- and none of the other side's. A
    check that only compared node_id sets passed when B's observation had been
    overwritten by A's, because both said ``shared_node``.
    """

    errors = []
    if not (len(virtual_states) == len(sources) == len(cluster_ids) == 2):
        return ["virtual isolation needs exactly two clusters"]
    own_uids = [{str(item["uid"]) for item in source["pods"]} for source in sources]
    own_names = [{str(item["name"]) for item in source["pods"]} for source in sources]
    observations: list[dict[str, Any]] = []
    for index, (state, cluster_id) in enumerate(zip(virtual_states, cluster_ids)):
        items = state.get("observations") or []
        if len(items) != 1:
            errors.append(f"{cluster_id}: {len(items)} observations, expected 1")
            observations.append({})
            continue
        observation = items[0]
        observations.append(observation)
        if state.get("cluster_id") not in (None, cluster_id):
            errors.append(f"{cluster_id}: store probe answered for another cluster")
        if observation.get("cluster_id") != cluster_id:
            errors.append(
                f"{cluster_id}: observation cluster_id is {observation.get('cluster_id')!r}"
            )
        nodes = {
            container.get("node_id")
            for container in observation.get("containers") or []
        }
        if nodes != {shared_node}:
            errors.append(
                f"{cluster_id}: container node IDs are {sorted(map(str, nodes))}"
            )
        if observation_pod_uids(observation) != own_uids[index]:
            errors.append(f"{cluster_id}: observation Pod UIDs are not its own")
        if observation_pod_names(observation) != own_names[index]:
            errors.append(f"{cluster_id}: observation Pod names are not its own")
        if state.get("decision") is not None:
            errors.append(f"{cluster_id}: pre-fault decision is not 404")
        if state.get("restart_budget") is not None:
            errors.append(f"{cluster_id}: pre-fault restart budget is not 404")
    if all(observations):
        a_gpus = workload_case.observation_gpu_uuids(observations[0])
        b_gpus = workload_case.observation_gpu_uuids(observations[1])
        if a_gpus & b_gpus:
            errors.append("GPU UUIDs are shared between the two cluster scopes")
        if observation_pod_uids(observations[0]) & observation_pod_uids(
            observations[1]
        ):
            errors.append("Pod UIDs are shared between the two cluster scopes")
    return errors


def command_ids_in_logs(command_ids: Sequence[str], logs: dict[str, str]) -> list[str]:
    """Which of ``command_ids`` appear in any of the executor ``logs``."""

    found = set()
    for text in logs.values():
        for command_id in command_ids:
            if command_id and command_id in text:
                found.add(command_id)
    return sorted(found)


def executor_logs_since(
    regional: RegionalLiveFixture,
    *,
    since: datetime,
) -> dict[str, str]:
    """Each Ready executor Pod's log since ``since`` (bounded, never negative)."""

    elapsed = int((datetime.now(timezone.utc) - since).total_seconds())
    window = max(60, min(elapsed + 60, EXECUTOR_LOG_MAX_WINDOW_SECONDS))
    logs = {}
    for pod in regional.ready_pods("gpu", "gpu-fault-cluster-executor"):
        logs[str(pod["name"])] = regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            f"--since={window}s",
            check=False,
            timeout=120,
        )
    return logs


def control_plane_blast_errors(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Why the control-plane EKS is not identical across the fault window.

    Nodes and gpu-fault Jobs must match exactly. Eviction events are judged
    as "none new": the whole event set was compared before, and it grew
    whenever an unrelated event aged out of the API server between the two
    snapshots -- a FAIL with nothing to do with the workflow.
    """

    errors = []
    if before.get("nodes") != after.get("nodes"):
        errors.append("CPU node taints/cordon/gpu-fault metadata changed")
    if before.get("gpu_fault_jobs") != after.get("gpu_fault_jobs"):
        errors.append("gpu-fault labelled Jobs on the control plane changed")
    new_events = [
        item
        for item in after.get("eviction_events") or []
        if item not in (before.get("eviction_events") or [])
    ]
    if new_events:
        errors.append(f"{len(new_events)} new eviction events in the window")
    return errors


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
        except Exception as exc:
            result[f"{label}_cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["workload_cleanup_deferred"] = True
            errors.append(f"{label}: workflow not quiescent; workload left in place")
            return errors
    try:
        workload.delete()
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
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=case_id,
        run_id=f"{case_id.lower()}-{attempt}",
    )
    source_manifest = BASELINE_MANIFEST
    rendered_manifest = case_dir / "managed-workload.yaml"
    fixture: ManagedWorkloadFixture | None = None
    result: dict[str, Any] = {"verdict": "FAIL"}
    failure: BaseException | None = None
    try:
        prewarm.create(prewarm_nodes(regional))
        not_applicable: dict[str, str] = {}
        submission: dict[str, Any]
        render_equivalence: bool | None = None
        if case_id == "GF-REGIONAL-WORKLOAD-001":
            fixture = managed_fixture(
                regional,
                manifest=source_manifest,
                site_file=site.site_file,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            # `submit` raises on a non-zero exit, so reaching this line is
            # the returncode-0 fact the check below records.
            submission = {**fixture.submit(), "returncode": 0}
            not_applicable["render_contract_equivalent"] = (
                "WORKLOAD-001 submits through gpu-training-submit directly; the "
                "annotate/apply render comparison is WORKLOAD-002's check"
            )
        else:
            annotate = run(
                [
                    sys.executable,
                    "-m",
                    "gpu_fault.workload_annotate_cli",
                    str(source_manifest),
                    "--site",
                    str(site.site_file),
                    "--job-id",
                    job_id,
                    "--attempt-id",
                    attempt_id,
                    "--restart-budget",
                    "1",
                    "--namespace",
                    site.namespace,
                    "--output",
                    str(rendered_manifest),
                ],
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=False,
            )
            dry_run = run(
                [
                    sys.executable,
                    "-m",
                    "gpu_fault.training_submit_cli",
                    str(source_manifest),
                    "--site",
                    str(site.site_file),
                    "--job-id",
                    job_id,
                    "--attempt-id",
                    attempt_id,
                    "--restart-budget",
                    "1",
                    "--namespace",
                    site.namespace,
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
            server_dry_run = regional.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(regional.settings.gpu_kubeconfig),
                    "--context",
                    regional.settings.gpu_context,
                    "-n",
                    regional.settings.namespace,
                    "apply",
                    "--dry-run=server",
                    "-f",
                    str(rendered_manifest),
                ],
                cwd=ROOT,
                check=False,
            )
            if server_dry_run.returncode:
                raise RegionalFixtureError("server-side dry-run rejected manifest")
            fixture = managed_fixture(
                regional,
                manifest=rendered_manifest,
                site_file=site.site_file,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            # Spec step 1: a same-named PyTorchJob from an earlier run would
            # make `apply` an update of the old object rather than a fresh
            # submission, and its Pods would carry the old attempt.
            fixture.delete()
            apply = regional.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(regional.settings.gpu_kubeconfig),
                    "--context",
                    regional.settings.gpu_context,
                    "-n",
                    regional.settings.namespace,
                    "apply",
                    "-f",
                    str(rendered_manifest),
                ],
                cwd=ROOT,
            )
            submission = {
                "pre_apply_delete": True,
                "annotate_returncode": annotate.returncode,
                "dry_run_returncode": dry_run.returncode,
                "server_dry_run_returncode": server_dry_run.returncode,
                "apply_returncode": apply.returncode,
                "returncode": max(
                    annotate.returncode,
                    dry_run.returncode,
                    server_dry_run.returncode,
                    apply.returncode,
                ),
            }
        finite = wait_finite_workload(fixture)
        training_logs = {
            str(item["name"]): regional.kubectl(
                "gpu",
                "logs",
                str(item["name"]),
                "--tail=400",
                check=False,
                timeout=60,
            )
            for item in finite["pods"]
        }
        losses = loss_errors(training_logs)
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
            "nccl_all_reduce_succeeded": all(
                "all_reduce=300.0"
                in finite["heartbeat_logs"].get(str(item["name"]), "")
                for item in finite["pods"]
            ),
            "per_rank_loss_non_increasing": not losses,
            "terminal_observation_succeeded": (
                len(terminal["observations"]) == 1
                and terminal_observation["workload_phase"] == "SUCCEEDED"
            ),
            "observation_ranks_are_0_1_2": not rank_errors,
            "terminal_event_key_deterministic": (
                decision.get("event_key") == terminal_event_key
            ),
            "deletion_does_not_regress_observation": (
                len(observations) == 1
                and observations[0]["workload_phase"] == "SUCCEEDED"
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
        if fixture is not None:
            try:
                fixture.delete()
            except Exception as exc:
                cleanup_errors.append(f"workload: {type(exc).__name__}: {exc}")
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


OBSERVATION_POST_PROBE = r"""
import json
import os
import sys
from gpu_fault.collectors.sinks import HttpEventSink

payload = json.loads(sys.argv[1])
os.environ["SSL_CERT_FILE"] = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
    timeout_seconds=15,
    max_attempts=1,
)
result = sink.post("/v1/workload-observations", payload)
print(json.dumps(result, sort_keys=True))
"""


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


def run_iso001(
    *,
    site: WorkloadSite,
    primary: SiteTarget,
    secondary: SiteTarget,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    attempt: int,
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
    failure: BaseException | None = None
    injection_context: dict[str, Any] | None = None
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

        def prepare(
            item: tuple[
                RegionalLiveFixture, ManagedWorkloadFixture, ImagePrewarmFixture
            ],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            regional, workload, prewarm = item
            prewarm.create(prewarm_nodes(regional))
            workload.submit()
            source = workload.wait_running(timeout_seconds=900)
            observation = workload_case.wait_observation(
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
            return source, observation

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
        for regional, observation in zip(fixtures, observations, strict=True):
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
        for regional, observation in zip(fixtures, observations, strict=True):
            post_observation(regional, refresh_observation(observation))
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
        injected_at = datetime.now(timezone.utc)
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
        errors = workload_case.workflow_errors(state_a, expected_gpu_count=24)
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
                not state_b.get("incidents") and not state_b.get("workflows")
            ),
            "secondary_pod_uids_unchanged": {
                str(item["uid"]) for item in target_b["pods"]
            }
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
    except Exception as exc:
        failure = exc
    finally:
        cleanup_errors: list[str] = []
        # Cluster A's workflow may still be RUNNING when the body fails; its
        # workload is deleted only once quiescence is proven. Cluster B never
        # had a workflow, so its workload goes straight away.
        for index, (regional, workload) in enumerate(zip(fixtures, workloads)):
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


E2E_PREFLIGHT_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext

cluster_id = sys.argv[1]
store = ApplicationContext.from_environment().store
statuses = [
    item.model_dump(mode="json")
    for item in store.list_collector_statuses(cluster_id)
]
agents = [
    item.model_dump(mode="json")
    for item in store.list_agents(cluster_id)
]
print(json.dumps({
    "collector_statuses": statuses,
    "agents": agents,
    "observed_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True, default=str))
"""


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
    for pod in regional.ready_pods("gpu", "gpu-fault-cluster-executor"):
        text = regional.kubectl(
            "gpu",
            "logs",
            str(pod["name"]),
            "--since=10m",
            check=False,
        )
        executor_logs.append(
            {
                "pod": pod["name"],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "lines": len(text.splitlines()),
            }
        )
        # Word-boundary matches with the offending lines recorded: the old
        # substring scan flagged `error_count=0` and any 401/403 digit run,
        # and told the operator nothing about what it had seen.
        suspicious.extend(
            {"pod": pod["name"], **hit} for hit in suspicious_log_lines(text)
        )
    errors = []
    for node in nodes:
        # A spare-pool node is cordoned by design while it waits in the pool
        # (hyperpod_spares keeps spec.unschedulable=True until it is handed
        # over), so only its readiness is a preflight condition.
        spare = (node.get("labels") or {}).get("gpu-fault.io/spare") == "true"
        if node["ready"] != "True" or (node["unschedulable"] and not spare):
            errors.append(f"{node['name']} is not Ready and schedulable")
        if not required_kinds <= status_by_node.get(str(node["name"]), set()):
            errors.append(f"{node['name']} lacks required Collector status")
    if len(active_agents) < len(nodes):
        errors.append("not every GPU node has a live ACTIVE Agent")
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
        "executor_logs": executor_logs,
        "suspicious_executor_logs": suspicious,
        "errors": errors,
    }


def notification_errors(state: dict[str, Any]) -> list[str]:
    notifications = state.get("notifications") or []
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in notifications:
        notification = item.get("notification") or {}
        result = item.get("result") or {}
        categories.setdefault(str(notification.get("category")), []).append(result)
    errors = []
    for category in ("FAULT_DETECTED", "ACTION_COMPLETED"):
        values = categories.get(category) or []
        if len(values) != 1:
            errors.append(f"{category} notification count is not one")
            continue
        if values[0].get("status") != "SENT":
            errors.append(f"{category} notification is not SENT")
        if not values[0].get("provider_message_id"):
            errors.append(f"{category} notification has no provider message ID")
    return errors


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
        "verdict": "FAIL",
        "errors": [],
        "baseline_state": baseline_state,
    }
    failure: BaseException | None = None
    injection_context: dict[str, Any] | None = None
    try:
        prewarm.create(prewarm_nodes(regional))
        workload.submit()
        source = workload.wait_running(timeout_seconds=900)
        node = str(source["pods"][0]["node"])
        workload_case.wait_observation(
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
        probe = HostProbeFixture(
            HostProbeSettings(
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
        if workflow.get("request_id"):
            injection_context["workflow_request_ids"] = [str(workflow["request_id"])]
        errors = workload_case.workflow_errors(state, expected_gpu_count=24)
        errors.extend(notification_errors(state))
        event = state.get("event") or {}
        incident = state.get("incident") or {}
        if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
            errors.append("event evidence reference is not a real kmsg reference")
        if not event.get("source_boot_id") or event.get("source_monotonic_us") is None:
            errors.append("event lacks real boot ID or monotonic source timestamp")
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
            "control_plane_eks_diff": "IDENTICAL" if not blast_errors else "CHANGED",
            "control_plane_eks_errors": blast_errors,
            "gpu_nodes_after_restart": nodes_after,
        }
        # Consumed by BLAST-001 (blast_acceptance_cases_1): maintenance_window
        # .start/.end and control-plane-current.json's workflows[] with their
        # official_steps/safety_steps/step_executions. Keep these keys.
        write_json_atomic(
            case_dir / "execution-card.json",
            {
                "case_id": "GF-REGIONAL-E2E-001",
                "maintenance_window": {
                    "start": injected_at.isoformat(),
                    "end": maintenance_window_end.isoformat(),
                },
                "cluster_id": target.cluster_id,
                "node": node,
                "job_id": job_id,
                "attempt_id": attempt_id,
            },
        )
        write_json_atomic(
            case_dir / "control-plane-current.json",
            {"workflows": [workflow]},
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
            restored = gpu_nodes_clean(nodes_after_cleanup)
        except Exception as exc:
            cleanup_errors.append(f"node recheck: {type(exc).__name__}: {exc}")
            restored = False
        checks_after = result.setdefault("checks", {})
        checks_after["gpu_nodes_restored_after_cleanup"] = restored
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors or not restored:
            result["verdict"] = "FAIL"
    raise_with_outcome(failure, result)
    result["limitations"] = [
        "The XID line is written by an approved user-space probe into real "
        "/dev/kmsg; it validates the software chain but is not hardware damage.",
        "control_plane_eks_identical compares CPU node metadata and gpu-fault "
        "Jobs exactly and eviction events as 'none new in the window'; the raw "
        "event set is unbounded and churns on its own.",
    ]
    return result


def gpu_nodes_clean(nodes: Sequence[dict[str, Any]]) -> bool:
    """Every GPU node Ready, schedulable (spares excepted) and free of our taints."""

    return all(
        item["ready"] == "True"
        and (
            not item["unschedulable"]
            # spare-pool nodes stay cordoned by design (see e2e_preflight)
            or (item.get("labels") or {}).get("gpu-fault.io/spare") == "true"
        )
        and not any(
            str(taint.get("key", "")).startswith("gpu-fault.io/")
            for taint in item.get("taints") or []
        )
        for item in nodes
    )


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
        "GPU_FAULT_PRIMARY_CLUSTER_ID": primary.cluster_id,
        "GPU_FAULT_SECONDARY_CLUSTER_ID": (
            secondary.cluster_id if secondary is not None else ""
        ),
        "GPU_FAULT_TEST_JOB_ID": job_id,
        "GPU_FAULT_TEST_ATTEMPT_ID": attempt_id,
    }
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details=case_plan(
                arguments.case,
                primary=primary,
                secondary=secondary,
                job_id=job_id,
                attempt_id=attempt_id,
                predecessor=predecessor,
            ),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if arguments.confirm != confirmation:
        raise WorkloadAcceptanceError(f"confirmation must be exactly {confirmation}")
    deadline = authorize_execution(
        arguments,
        case_id=arguments.case,
        confirmation=confirmation,
        environment=environment,
    )
    if not predecessor.get("valid", False):
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
