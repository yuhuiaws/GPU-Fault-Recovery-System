from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, cast


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin_site import load_site  # noqa: E402

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
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget as MultiClusterTarget,
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
    predecessor_evidence,
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


class WorkloadAcceptanceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

cluster_id, job_id, attempt_id = sys.argv[1:]
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
commands = [
    item.model_dump(mode="json")
    for item in store.list_remote_commands()
    if any(
        job_id in workload_id
        for workload_id in item.step.workload_ids
    )
]
print(json.dumps({
    "observations": observations,
    "decision": decision.model_dump(mode="json") if decision else None,
    "restart_budget": budget,
    "commands": commands,
}, sort_keys=True, default=str))
"""


def workload_store(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        regional.cpu_python(
            WORKLOAD_STORE_PROBE,
            regional.settings.cluster_id,
            job_id,
            attempt_id,
        ),
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
            "gpu_fault.admin_cli",
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


def run_workload_baseline(
    *,
    case_id: str,
    site: WorkloadSite,
    target: SiteTarget,
    case_dir: Path,
    job_id: str,
    attempt_id: str,
    state_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    regional = site.regional(target)
    if regional.gpu_workloads():
        raise RegionalFixtureError("target cluster already has a GPU workload")
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=case_id,
        run_id=f"{case_id.lower()}-{attempt}",
    )
    source_manifest = BASELINE_MANIFEST
    rendered_manifest = case_dir / "managed-workload.yaml"
    fixture: ManagedWorkloadFixture | None = None
    result: dict[str, Any] = {"verdict": "FAIL"}
    try:
        prewarm.create(prewarm_nodes(regional))
        if case_id == "GF-REGIONAL-WORKLOAD-001":
            fixture = managed_fixture(
                regional,
                manifest=source_manifest,
                site_file=site.site_file,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            submission = fixture.submit()
            render_equivalence = True
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
            regional.kubectl("gpu", "apply", "-f", str(rendered_manifest))
            submission = {
                "annotate_returncode": annotate.returncode,
                "server_dry_run_returncode": server_dry_run.returncode,
            }
            fixture = managed_fixture(
                regional,
                manifest=rendered_manifest,
                site_file=site.site_file,
                job_id=job_id,
                attempt_id=attempt_id,
            )
        finite = wait_finite_workload(fixture)
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
        checks = {
            "submission_succeeded": bool(submission),
            "render_contract_equivalent": render_equivalence,
            "managed_metadata_complete": not metadata,
            "three_pods_on_three_nodes": len(finite["pods"]) == 3
            and len({item["node"] for item in finite["pods"]}) == 3,
            "nccl_all_reduce_succeeded": all(
                "all_reduce=300.0"
                in finite["heartbeat_logs"].get(str(item["name"]), "")
                for item in finite["pods"]
            ),
            "terminal_observation_succeeded": (
                len(terminal["observations"]) == 1
                and terminal["observations"][0]["workload_phase"] == "SUCCEEDED"
            ),
            "terminal_event_key_deterministic": (
                decision.get("event_key") == terminal_event_key
            ),
            "deletion_does_not_regress_observation": (
                len(observations) == 1
                and observations[0]["workload_phase"] == "SUCCEEDED"
            ),
            "no_remote_mutation_commands": not terminal["commands"],
        }
        status = admin_status(state_dir)
        checks["admin_status_passed"] = status["returncode"] == 0
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "metadata_errors": metadata,
            "workload": finite,
            "terminal": terminal,
            "after_delete": after_delete,
            "admin_status": status,
        }
    finally:
        if fixture is not None:
            fixture.delete()
        residuals = prewarm.cleanup()
        result["prewarm_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
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
    return cast(
        dict[str, Any],
        regional.executor_python(
            OBSERVATION_POST_PROBE,
            json.dumps(payload, sort_keys=True),
        ),
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
    result: dict[str, Any] = {"verdict": "FAIL", "errors": []}
    try:
        sources = []
        observations = []
        for regional, workload, prewarm in zip(
            fixtures,
            workloads,
            prewarms,
            strict=True,
        ):
            if regional.gpu_workloads():
                raise RegionalFixtureError("target cluster already has GPU workloads")
            prewarm.create(prewarm_nodes(regional))
            workload.submit()
            source = workload.wait_running(timeout_seconds=900)
            sources.append(source)
            settings = workload_case_settings(
                regional=regional,
                site_file=site.site_file,
                manifest=LONG_RUNNING_MANIFEST,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            observations.append(
                workload_case.wait_observation(
                    regional,
                    settings,
                    node=str(source["pods"][0]["node"]),
                    expected_gpu_count=24,
                )
            )
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
        errors = workload_case.workflow_errors(state_a, expected_gpu_count=24)
        target_a = workloads[0].wait_restarted(
            {str(item["uid"]) for item in sources[0]["pods"]},
            timeout_seconds=900,
        )
        target_b = workloads[1].wait_pod_uids_unchanged(
            b_uids,
            timeout_seconds=120,
        )
        state_b = workload_store(
            regional_b,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        virtual_isolated = all(
            len(item["observations"]) == 1
            and {
                container.get("node_id")
                for container in item["observations"][0]["containers"]
            }
            == {shared_node}
            for item in virtual_states
        )
        checks = {
            "two_physical_registrations": True,
            "same_virtual_node_is_cluster_scoped": virtual_isolated,
            "primary_workflow_succeeded": not errors,
            "primary_restart_budget_one": (
                (state_a.get("restart_budget") or {}).get("restart_count") == 1
            ),
            "secondary_has_no_restart_budget": state_b["restart_budget"] is None,
            "secondary_has_no_decision": state_b["decision"] is None,
            "secondary_pod_uids_unchanged": {
                str(item["uid"]) for item in target_b["pods"]
            }
            == b_uids,
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "errors": errors,
            "virtual_posts": virtual_results,
            "injection": injection,
            "primary_state": state_a,
            "primary_target": target_a,
            "secondary_state": state_b,
        }
    finally:
        cleanup_errors = []
        for workload in workloads:
            try:
                workload.delete()
            except Exception as exc:
                cleanup_errors.append(f"workload: {type(exc).__name__}: {exc}")
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
    result["limitations"] = [
        "The fault is an authenticated software replay into cluster A; it proves "
        "cluster-scoped state and restart behavior, not a hardware-originated XID."
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
        status_by_node.setdefault(str(item["node_id"]), set()).add(str(item["kind"]))
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
    suspicious = []
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
            }
        )
        for marker in ("ERROR", "Traceback", "401", "403", "CERTIFICATE_VERIFY_FAILED"):
            if marker.lower() in text.lower():
                suspicious.append({"pod": pod["name"], "marker": marker})
    errors = []
    for node in nodes:
        if node["ready"] != "True" or node["unschedulable"]:
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
    result: dict[str, Any] = {"verdict": "FAIL", "errors": []}
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
        errors = workload_case.workflow_errors(state, expected_gpu_count=24)
        errors.extend(notification_errors(state))
        event = state.get("event") or {}
        if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
            errors.append("event evidence reference is not a real kmsg reference")
        if not event.get("source_boot_id") or event.get("source_monotonic_us") is None:
            errors.append("event lacks real boot ID or monotonic source timestamp")
        target_workload = workload.wait_restarted(
            {str(item["uid"]) for item in source["pods"]},
            timeout_seconds=900,
        )
        blast_after = regional.cpu_blast_snapshot()
        nodes_after = regional.gpu_nodes()
        clean_nodes = all(
            item["ready"] == "True"
            and not item["unschedulable"]
            and not any(
                str(taint.get("key", "")).startswith("gpu-fault.io/")
                for taint in item["taints"]
            )
            for item in nodes_after
        )
        checks = {
            "collector_agent_executor_preflight": not preflight["errors"],
            "real_kmsg_injection": injection["bytes_written"] > 0,
            "workflow_contract": not errors,
            "workload_pod_uids_changed": {
                str(item["uid"]) for item in target_workload["pods"]
            }.isdisjoint({str(item["uid"]) for item in source["pods"]}),
            "control_plane_eks_identical": blast_before == blast_after,
            "gpu_nodes_restored": clean_nodes,
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "errors": errors,
            "preflight": preflight,
            "source_workload": source,
            "target_workload": target_workload,
            "injection": injection,
            "state": state,
            "control_plane_eks_diff": (
                "IDENTICAL" if blast_before == blast_after else "CHANGED"
            ),
        }
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
            },
        )
        write_json_atomic(
            case_dir / "control-plane-current.json",
            {"workflows": [state.get("workflow") or {}]},
        )
    finally:
        cleanup_errors = []
        if probe is not None:
            try:
                probe_residuals = probe.cleanup()
                result["probe_residuals"] = probe_residuals
                if any(probe_residuals.values()):
                    cleanup_errors.append("host probe resources remain")
            except Exception as exc:
                cleanup_errors.append(f"probe: {type(exc).__name__}: {exc}")
        try:
            workload.delete()
        except Exception as exc:
            cleanup_errors.append(f"workload: {type(exc).__name__}: {exc}")
        try:
            prewarm_residuals = prewarm.cleanup()
            result["prewarm_residuals"] = prewarm_residuals
            if any(prewarm_residuals.values()):
                cleanup_errors.append("prewarm resources remain")
        except Exception as exc:
            cleanup_errors.append(f"prewarm: {type(exc).__name__}: {exc}")
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["verdict"] = "FAIL"
    result["limitations"] = [
        "The XID line is written by an approved user-space probe into real "
        "/dev/kmsg; it validates the software chain but is not hardware damage."
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
    return value


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureError(f"received signal {signum}")


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)
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
    if (
        arguments.case
        in {
            "GF-REGIONAL-WORKLOAD-001",
            "GF-REGIONAL-WORKLOAD-002",
        }
        and arguments.state_dir is None
    ):
        raise WorkloadAcceptanceError(
            f"{arguments.case} requires --state-dir for final admin status"
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
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
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
                state_dir=cast(Path, arguments.state_dir),
                attempt=arguments.attempt,
            ),
            "GF-REGIONAL-WORKLOAD-002": lambda: run_workload_baseline(
                case_id=arguments.case,
                site=site,
                target=primary,
                case_dir=case_dir,
                job_id=job_id,
                attempt_id=attempt_id,
                state_dir=cast(Path, arguments.state_dir),
                attempt=arguments.attempt,
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
        outcome = {
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "limitations": [
                "The case stopped at the first failed assertion; later checks "
                "were not treated as executed."
            ],
        }
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": arguments.case,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
