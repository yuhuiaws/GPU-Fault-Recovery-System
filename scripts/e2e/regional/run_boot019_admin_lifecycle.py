from __future__ import annotations

import argparse
import base64
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin import cluster_join_commit  # noqa: E402
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError  # noqa: E402
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster  # noqa: E402
from gpu_fault.admin.cluster_removal import (  # noqa: E402
    RemoveClusterRequest,
    remove_cluster,
)
from gpu_fault.admin.resource_registry import (  # noqa: E402
    fetch_installation_resource_registry,
)
from gpu_fault.admin.site import RenderedSite, load_site  # noqa: E402
from gpu_fault.admin.uninstall import UninstallRequest, uninstall  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder  # noqa: E402
from scripts.e2e.regional.boot019_revocation import RevocationCredentials  # noqa: E402
from scripts.e2e.regional.boot_membership_observation import (  # noqa: E402
    membership_observation,
    membership_transition_errors,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    predecessor_evidence,
)

CASE_ID = "GF-REGIONAL-BOOT-019"
CONFIRMATION = "RUN_BOOT019_ADMIN_LIFECYCLE"
REMOVE_CONFIRMATION = "REMOVE_GPU_CLUSTER"
UNINSTALL_CONFIRMATION = "UNINSTALL_GPU_FAULT"
# Uninstall with ``--cpu-cluster keep`` preserves at least the CPU EKS cluster
# and the GPU cluster records; fewer preserved entries means it deleted
# something the request told it to keep.
MIN_PRESERVED_REGISTRY_ENTRIES = 2
AURORA_CLUSTER_RESOURCE_KEY = "aws/aurora/cluster"


class InjectedAcceptanceFailure(RuntimeError):
    pass


class AcceptanceCheckError(RuntimeError):
    """A live observation contradicted the BOOT-019 contract.

    Raised explicitly rather than through ``assert``: ``python -O`` strips
    ``assert`` statements, and a runner whose checks vanish under an
    optimisation flag would report a constant PASS.
    """


def _require(condition: bool, message: str, context: Any = None) -> None:
    if not condition:
        detail = f"{message}: {context!r}" if context is not None else message
        raise AcceptanceCheckError(detail)


class AdminLifecycleBackend(Protocol):
    def snapshot(self) -> dict[str, Any]: ...

    def join(self, fault: str | None = None) -> dict[str, Any]: ...

    def capture_joined_token(self, cluster_id: str) -> dict[str, Any]: ...

    def remove(self, cluster_id: str) -> dict[str, Any]: ...

    def probe_revoked_token(self, capture: dict[str, Any]) -> dict[str, Any]: ...

    def uninstall(self) -> dict[str, Any]: ...

    def cleanup_sensitive_files(self, *, completed: bool = False) -> None: ...


def _assert_consistent_snapshot(
    snapshot: dict[str, Any],
    expected: set[str],
) -> None:
    for field in (
        "site_cluster_ids",
        "registry_secret_cluster_ids",
        "release_state_cluster_ids",
        "installation_registry_cluster_ids",
    ):
        _require(set(snapshot[field]) == expected, f"{field} differs", snapshot)
    _require(
        snapshot["cpu_control_plane_ready"] is True,
        "CPU control plane is not ready",
        snapshot,
    )


def assert_uninstall_result(result: dict[str, Any]) -> None:
    """What a keep-CPU uninstall must have left behind.

    ``cpu_cluster == "keep"`` and ``delete_policy_residuals == 0`` are
    literals the uninstall code writes unconditionally, so asserting them
    proved nothing. What varies with the live run is how many registry
    entries were preserved and whether any resource is still waiting to be
    deleted in the final registry snapshot.
    """

    preserved = result.get("registry_entries_preserved")
    _require(
        isinstance(preserved, int) and preserved >= MIN_PRESERVED_REGISTRY_ENTRIES,
        f"uninstall preserved fewer than {MIN_PRESERVED_REGISTRY_ENTRIES} "
        "registry entries",
        result,
    )
    statuses = result.get("final_registry_statuses")
    _require(
        isinstance(statuses, dict) and bool(statuses),
        "uninstall result carries no final registry statuses",
        result,
    )
    pending = sorted(
        key
        for key, status in dict(statuses or {}).items()
        if status == "DELETE_PENDING"
    )
    _require(not pending, "final registry still has DELETE_PENDING entries", pending)
    _require(
        all(
            status in {"DELETED", "DETACHED", "PRESERVED"}
            for status in dict(statuses or {}).values()
        ),
        "final registry contains a nonterminal or unknown status",
    )
    # The reinstall contract: the Aurora cluster (and so the site's records)
    # survives a keep-CPU uninstall that did not ask for ``--reset-database``.
    _require(
        dict(statuses or {}).get(AURORA_CLUSTER_RESOURCE_KEY) == "PRESERVED",
        "keep-CPU uninstall did not preserve the Aurora cluster",
        statuses,
    )


def run_admin_lifecycle(
    backend: AdminLifecycleBackend,
    recorder: EvidenceRecorder,
) -> dict[str, Any]:
    recorder.document["status"] = "RUNNING"
    recorder.note("verdict", "FAIL")
    try:
        _require(
            not {
                "join_failure_before_site_commit",
                "join_failure_after_site_commit",
            }.intersection(recorder.document["stages"]),
            "legacy site-commit evidence cannot authorize the activation contract",
        )
        baseline = recorder.stage("baseline", backend.snapshot)
        baseline_ids = set(baseline["site_cluster_ids"])
        release_id = baseline.get("release_id")
        _require(
            isinstance(release_id, str) and bool(release_id),
            "baseline release identity is missing",
        )
        _require(
            len(baseline_ids) == 1,
            "BOOT-019 requires an isolated site with exactly one managed GPU cluster",
            sorted(baseline_ids),
        )
        _assert_consistent_snapshot(baseline, baseline_ids)

        pre_commit = recorder.stage(
            "join_failure_before_activation",
            lambda: backend.join("before-activation"),
        )
        _require(pre_commit["phase"] == "ROLLED_BACK", "pre-commit phase", pre_commit)
        if "join_failure_after_activation" not in recorder.document["stages"]:
            _assert_consistent_snapshot(backend.snapshot(), baseline_ids)

        post_commit = recorder.stage(
            "join_failure_after_activation",
            lambda: backend.join("after-activation"),
        )
        _require(
            post_commit["phase"] == "FAILED_AFTER_ACTIVATION",
            "post-commit phase",
            post_commit,
        )
        joined_id = str(post_commit["cluster_id"])
        _require(joined_id not in baseline_ids, "joined cluster was already managed")

        joined = recorder.stage("join_resumed", backend.join)
        _require(joined["phase"] == "COMPLETED", "resumed join phase", joined)
        _require(joined["cluster_id"] == joined_id, "resumed join cluster", joined)
        joined_ids = baseline_ids | {joined_id}
        after_join = recorder.stage("joined_snapshot", backend.snapshot)
        _assert_consistent_snapshot(after_join, joined_ids)
        _require(
            not membership_transition_errors(
                baseline["membership_cpu"], after_join["membership_cpu"]
            ),
            "join changed CPU roles outside the failure-domain publication",
        )

        token_capture = recorder.stage(
            "joined_token_captured",
            lambda: backend.capture_joined_token(joined_id),
        )
        removed = recorder.stage(
            "joined_cluster_removed",
            lambda: backend.remove(joined_id),
        )
        _require(removed["phase"] == "COMPLETED", "remove phase", removed)
        _require(
            set(removed["remaining_cluster_ids"]) == baseline_ids,
            "remaining clusters after remove",
            removed,
        )
        revoked = recorder.stage(
            "removed_token_rejected",
            lambda: backend.probe_revoked_token(token_capture),
        )
        _require(revoked["status"] == 403, "revoked token status", revoked)
        _require(
            revoked["detail"] == "regional cluster authentication failed",
            "revoked token detail",
            revoked,
        )
        after_remove = recorder.stage("post_remove_snapshot", backend.snapshot)
        _assert_consistent_snapshot(after_remove, baseline_ids)
        _require(
            not membership_transition_errors(
                after_join["membership_cpu"], after_remove["membership_cpu"]
            ),
            "remove changed CPU roles outside the failure-domain publication",
        )

        last_cluster_id = next(iter(baseline_ids))
        last_removed = recorder.stage(
            "last_cluster_removed",
            lambda: backend.remove(last_cluster_id),
        )
        _require(
            last_removed["phase"] == "COMPLETED", "last remove phase", last_removed
        )
        _require(
            last_removed["remaining_cluster_ids"] == [],
            "clusters remain after the last remove",
            last_removed,
        )
        empty = recorder.stage("empty_registry_snapshot", backend.snapshot)
        _assert_consistent_snapshot(empty, set())
        _require(
            not membership_transition_errors(
                after_remove["membership_cpu"], empty["membership_cpu"]
            ),
            "last remove changed CPU roles outside the failure-domain publication",
        )

        uninstall_result = recorder.stage("uninstall_keep_cpu", backend.uninstall)
        assert_uninstall_result(uninstall_result)
    except BaseException as exc:
        recorder.fail(exc)
        raise
    finally:
        # Failed runs retain only run-bound 0600 credentials needed to prove
        # revocation after a new process resumes; ordinary evidence has digests.
        try:
            backend.cleanup_sensitive_files()
        except BaseException as exc:
            recorder.fail(exc)
            raise
    try:
        backend.cleanup_sensitive_files(completed=True)
    except BaseException as exc:
        recorder.fail(exc)
        raise
    recorder.document.update(
        {
            "schema_version": 1,
            "report_type": "fault-acceptance",
            "release_id": release_id,
            "cluster_id": next(iter(baseline_ids)),
            "verdict": "PASS",
        }
    )
    return recorder.complete()


class LiveAdminLifecycleBackend:
    def __init__(
        self,
        *,
        site_path: Path,
        gpu_cluster_arn: str,
        cluster_id: str | None,
        allowed_namespaces: tuple[str, ...],
        join_state_dir: Path,
        run_dir: Path,
    ) -> None:
        self.site_path = site_path
        self.gpu_cluster_arn = gpu_cluster_arn
        self.cluster_id = cluster_id
        self.allowed_namespaces = allowed_namespaces
        self.join_state_dir = join_state_dir
        self.run_dir = run_dir
        self._captured_tokens: dict[str, bytes] = {}
        self._captures: dict[str, dict[str, Any]] = {}
        self._credential_custody = RevocationCredentials(
            run_dir, site_path=site_path, join_arn=gpu_cluster_arn
        )

    def _site(self) -> RenderedSite:
        return load_site(self.site_path, repository_root=ROOT)

    def _join_state(self) -> dict[str, Any]:
        path = self.join_state_dir / "state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _join_request(self) -> JoinClusterRequest:
        return JoinClusterRequest(
            site=self._site(),
            gpu_cluster_arn=self.gpu_cluster_arn,
            cluster_id=self.cluster_id,
            allowed_namespaces=self.allowed_namespaces,
            state_dir=self.join_state_dir,
        )

    def join(self, fault: str | None = None) -> dict[str, Any]:
        if fault not in {None, "before-activation", "after-activation"}:
            raise AcceptanceCheckError("unknown join fault boundary")
        state = self._join_state()
        if fault == "before-activation" and state.get("phase") == "ROLLED_BACK":
            return {
                **state,
                "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
            }
        if (
            fault == "after-activation"
            and state.get("phase") == "FAILED_AFTER_ACTIVATION"
        ):
            return {
                **state,
                "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
            }

        original_complete = cluster_join_commit.complete_step
        try:
            if fault is not None:
                boundary = (
                    "REGISTRY_UPDATED"
                    if fault == "before-activation"
                    else "ACTIVATION_STARTED"
                )

                def complete_then_fail(
                    path: Path,
                    state: dict[str, Any],
                    step: str,
                    evidence: dict[str, Any] | None = None,
                ) -> None:
                    original_complete(path, state, step, evidence)
                    if step == boundary:
                        raise InjectedAcceptanceFailure(
                            f"BOOT-019 injected failure after {boundary}"
                        )

                cluster_join_commit.complete_step = complete_then_fail
            try:
                return dict(join_cluster(self._join_request()))
            except InjectedAcceptanceFailure:
                state = self._join_state()
                return {
                    **state,
                    "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
                }
        finally:
            cluster_join_commit.complete_step = original_complete

    def _kubectl_json(self, *arguments: str) -> dict[str, Any]:
        site = self._site()
        command = [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            *arguments,
            "-o",
            "json",
        ]
        output = RegionalLiveFixture.run(
            command,
            cwd=ROOT,
            check=True,
            timeout=300,
        ).stdout
        value = json.loads(output)
        if not isinstance(value, dict):
            raise AcceptanceCheckError("kubectl did not return a JSON object")
        return value

    def snapshot(self) -> dict[str, Any]:
        site = self._site()
        namespace = str(site.release_config["namespace"])
        site_ids = sorted(
            item["cluster_id"] for item in site.release_config["clusters"]
        )
        registry = self._kubectl_json(
            "-n",
            namespace,
            "get",
            "secret",
            "gpu-fault-regional-clusters",
        )
        raw_registry = base64.b64decode(registry["data"]["clusters.json"]).decode()
        secret_ids = sorted(item["cluster_id"] for item in json.loads(raw_registry))
        release_state = self._kubectl_json(
            "-n",
            namespace,
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
        )
        state = json.loads(release_state["data"]["state.json"])
        snapshot = fetch_installation_resource_registry(site)
        active_registry_ids = sorted(
            {
                item.resource_key.split("/")[1]
                for item in snapshot.resources
                if item.resource_key.startswith("cluster/")
                and item.resource_key.endswith("/eks")
                and item.status.value == "ACTIVE"
            }
        )
        deployment = self._kubectl_json(
            "-n",
            namespace,
            "get",
            "deployment",
            "gpu-fault-api-ha",
        )
        desired = int(deployment["spec"].get("replicas") or 0)
        ready = int(deployment.get("status", {}).get("readyReplicas") or 0)
        return {
            "release_id": state.get("release_id"),
            "site_cluster_ids": site_ids,
            "registry_secret_cluster_ids": secret_ids,
            "release_state_cluster_ids": sorted(state.get("cluster_ids") or []),
            "installation_registry_cluster_ids": active_registry_ids,
            "cpu_control_plane_ready": desired > 0 and ready == desired,
            "membership_cpu": membership_observation(site),
        }

    def capture_joined_token(self, cluster_id: str) -> dict[str, Any]:
        site = self._site()
        record = next(
            item
            for item in site.release_config["clusters"]
            if item["cluster_id"] == cluster_id
        )
        token = Path(record["token_file"]).read_bytes()
        self._captured_tokens[cluster_id] = token
        capture = {
            **self._credential_custody.capture(cluster_id, token),
            "control_plane_url": record["control_plane_url"],
            "ca_file": record["ca_file"],
            "ca_sha256": hashlib.sha256(
                Path(record["ca_file"]).read_bytes()
            ).hexdigest(),
        }
        self._captures[cluster_id] = capture
        return capture

    def remove(self, cluster_id: str) -> dict[str, Any]:
        return dict(
            remove_cluster(
                RemoveClusterRequest(
                    site=self._site(),
                    cluster_id=cluster_id,
                    confirmation=REMOVE_CONFIRMATION,
                )
            )
        )

    def probe_revoked_token(self, capture: dict[str, Any]) -> dict[str, Any]:
        cluster_id = str(capture["cluster_id"])
        stored = self._credential_custody.load(capture)
        if hashlib.sha256(
            Path(capture["ca_file"]).read_bytes()
        ).hexdigest() != capture.get("ca_sha256"):
            raise AcceptanceCheckError("revocation TLS trust identity changed")
        self._captures[cluster_id] = capture
        token = stored.decode("utf-8").strip()
        request = urllib.request.Request(
            str(capture["control_plane_url"]).rstrip("/")
            + "/v1/regional/executors/readiness",
            data=b"{}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-GPU-Fault-Cluster-ID": cluster_id,
            },
            method="POST",
        )
        context = ssl.create_default_context(cafile=str(capture["ca_file"]))
        try:
            with urllib.request.urlopen(
                request,
                context=context,
                timeout=20,
            ) as response:
                return {"status": response.status, "detail": "unexpected success"}
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read() or b"{}")
            return {"status": exc.code, "detail": payload.get("detail")}

    def uninstall(self) -> dict[str, Any]:
        # A keep-CPU uninstall is a reinstall: the Aurora cluster and its records
        # stay for the next deploy, so the database is never reset here and the
        # final-snapshot policy (a delete-mode choice) is left at its default.
        result = uninstall(
            UninstallRequest(
                site=self._site(),
                cpu_disposition="keep",
                confirmation=UNINSTALL_CONFIRMATION,
                reset_database=False,
            )
        )
        return {**result, "final_registry_statuses": final_registry_statuses(result)}

    def cleanup_sensitive_files(self, *, completed: bool = False) -> None:
        self._captured_tokens.clear()
        if completed:
            for capture in self._captures.values():
                self._credential_custody.remove(capture)
            self._captures.clear()

    def resume_revocation_capture(self, capture: dict[str, Any]) -> None:
        self._captures[str(capture["cluster_id"])] = capture


def final_registry_statuses(result: dict[str, Any]) -> dict[str, str]:
    """``resource_key -> status`` of the final registry snapshot uninstall wrote."""

    path = result.get("final_registry")
    if not path:
        raise AcceptanceCheckError("uninstall result names no final registry")
    document = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    resources = document.get("resources")
    if not isinstance(resources, list):
        raise AcceptanceCheckError("final registry has no resources list")
    return {
        str(item["resource_key"]): str(item["status"])
        for item in resources
        if isinstance(item, dict)
    }


def epoch_targets(
    site_path: Path,
    protected_path: Path,
    join_arn: str,
    *,
    resume: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if site_path.resolve() == protected_path.resolve():
        raise AcceptanceCheckError("BOOT-019 cannot use the protected site")
    lifecycle = load_site(site_path, repository_root=ROOT).release_config
    protected = load_site(protected_path, repository_root=ROOT).release_config

    def eks_arn(value: Any) -> str:
        try:
            parsed = Arn.parse(value) if isinstance(value, str) else None
            valid = (
                parsed is not None
                and parsed.service == "eks"
                and bool(parsed.resource_name)
            )
        except BootstrapError:
            valid = False
        if not valid:
            raise AcceptanceCheckError(
                "BOOT-019 isolation requires explicit physical EKS ARNs"
            )
        return str(value).strip()

    cpu = eks_arn(lifecycle.get("cpu_eks_arn"))
    protected_cpu = eks_arn(protected.get("cpu_eks_arn"))
    gpu = [eks_arn(item.get("eks_cluster_arn")) for item in lifecycle["clusters"]]
    protected_gpu = [
        eks_arn(item.get("eks_cluster_arn")) for item in protected["clusters"]
    ]
    join = eks_arn(join_arn)
    original = None
    if resume is not None:
        original = (resume.get("inputs") or {}).get("epoch_targets")
        if (
            resume.get("case_id") != CASE_ID
            or not isinstance(original, dict)
            or original.get("lifecycle_cpu_eks_arn") != cpu
            or original.get("join_gpu_eks_arn") != join
            or original.get("protected_cpu_eks_arn") != protected_cpu
            or original.get("protected_gpu_eks_arns") != sorted(protected_gpu)
        ):
            raise AcceptanceCheckError("BOOT-019 resume target identity changed")
        baseline = original.get("lifecycle_gpu_eks_arns")
        if not isinstance(baseline, list) or len(baseline) != 1:
            raise AcceptanceCheckError("BOOT-019 resume lacks its baseline")
        stages = resume.get("stages") or {}
        permitted = [set(baseline)]
        if "join_failure_before_activation" in stages:
            permitted.append({*baseline, join})
        if "post_remove_snapshot" in stages:
            permitted.append(set())
        if set(gpu) not in permitted or len(gpu) != len(set(gpu)):
            raise AcceptanceCheckError(
                "BOOT-019 membership does not match its recorded phase"
            )
        gpu = [eks_arn(item) for item in baseline]
    if len(gpu) != 1 or not protected_gpu:
        raise AcceptanceCheckError(
            "BOOT-019 requires one disposable baseline GPU and a protected fleet"
        )
    if (
        cpu == protected_cpu
        or set([cpu, *gpu, join]).intersection({protected_cpu, *protected_gpu})
        or len({cpu, *gpu, join}) != 3
    ):
        raise AcceptanceCheckError(
            "BOOT-019 targets overlap the protected or baseline clusters"
        )
    return {
        "lifecycle_cpu_eks_arn": cpu,
        "lifecycle_gpu_eks_arns": gpu,
        "join_gpu_eks_arn": join,
        "protected_cpu_eks_arn": protected_cpu,
        "protected_gpu_eks_arns": sorted(protected_gpu),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--site", required=True, type=Path)
    value.add_argument("--protected-site", required=True, type=Path)
    value.add_argument("--gpu-cluster-arn", required=True)
    value.add_argument("--cluster-id")
    value.add_argument(
        "--allowed-namespace",
        action="append",
        default=["gpu-fault-system", "training"],
    )
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    case_path = case_evidence_path(arguments.run_dir, CASE_ID)
    resume = (
        json.loads(case_path.read_text(encoding="utf-8"))
        if case_path.exists()
        else None
    )
    targets = epoch_targets(
        arguments.site,
        arguments.protected_site,
        arguments.gpu_cluster_arn,
        **({"resume": resume} if resume is not None else {}),
    )
    previous_id, previous_path = predecessor_path(
        arguments.run_dir, CASE_ID, arguments.predecessor_evidence
    )
    predecessor = (
        predecessor_evidence(previous_path, previous_id)
        if previous_id is not None and previous_path is not None
        else {"valid": True, "verdict": "NOT_REQUIRED"}
    )
    environment = {
        "GPU_FAULT_SITE_FILE": str(arguments.site.resolve()),
        "GPU_FAULT_PROTECTED_SITE_FILE": str(arguments.protected_site.resolve()),
    }
    plan = {
        "case_id": CASE_ID,
        "site": str(arguments.site),
        "gpu_cluster_arn": arguments.gpu_cluster_arn,
        "run_dir": str(arguments.run_dir),
        "site_sha256": hashlib.sha256(arguments.site.read_bytes()).hexdigest(),
        "protected_site": str(arguments.protected_site.resolve()),
        "protected_site_sha256": hashlib.sha256(
            arguments.protected_site.read_bytes()
        ).hexdigest(),
        "epoch_targets": targets,
        "predecessor": predecessor,
        "stages": [
            "inject pre-activation join failure and verify rollback",
            "inject post-activation-intent failure and verify fail-forward resume",
            "remove the joined cluster and reject its old token",
            "remove the final GPU cluster and verify explicit empty registry",
            "uninstall solution resources while preserving CPU/GPU clusters",
        ],
    }
    if not arguments.execute:
        document = build_plan(
            arguments=arguments,
            preflight_passed=predecessor.get("valid") is True,
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details=plan,
        )
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid") is True else 1
    authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment,
        details=plan,
    )
    if predecessor.get("valid") is not True:
        raise AcceptanceCheckError("formal predecessor evidence is not PASS")
    case_path = case_evidence_path(arguments.run_dir, CASE_ID)
    case_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    inputs = {
        "acceptance_contract": 4,
        "site": str(arguments.site.resolve()),
        "protected_site": str(arguments.protected_site.resolve()),
        "epoch_targets": targets,
        "gpu_cluster_arn": arguments.gpu_cluster_arn,
        "cluster_id": arguments.cluster_id,
        "allowed_namespaces": sorted(set(arguments.allowed_namespace)),
    }
    recorder = EvidenceRecorder(
        case_path,
        case_id=CASE_ID,
        inputs=inputs,
    )
    backend = LiveAdminLifecycleBackend(
        site_path=arguments.site.resolve(),
        gpu_cluster_arn=arguments.gpu_cluster_arn,
        cluster_id=arguments.cluster_id,
        allowed_namespaces=tuple(sorted(set(arguments.allowed_namespace))),
        join_state_dir=case_path.parent / "join-state",
        run_dir=case_path.parent,
    )
    capture = recorder.document["stages"].get("joined_token_captured")
    if isinstance(capture, dict):
        backend.resume_revocation_capture(capture)
    result = run_admin_lifecycle(backend, recorder)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
