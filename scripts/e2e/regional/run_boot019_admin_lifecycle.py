from __future__ import annotations

import argparse
import base64
import hashlib
import json
import ssl
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault import admin_cluster_join  # noqa: E402
from gpu_fault.admin_cluster_join import JoinClusterRequest, join_cluster  # noqa: E402
from gpu_fault.admin_cluster_removal import (  # noqa: E402
    RemoveClusterRequest,
    remove_cluster,
)
from gpu_fault.admin_resource_registry import (  # noqa: E402
    fetch_installation_resource_registry,
)
from gpu_fault.admin_site import RenderedSite, load_site  # noqa: E402
from gpu_fault.admin_uninstall import UninstallRequest, uninstall  # noqa: E402

from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder  # noqa: E402

CASE_ID = "GF-REGIONAL-BOOT-019"
CONFIRMATION = "RUN_BOOT019_ADMIN_LIFECYCLE"
REMOVE_CONFIRMATION = "REMOVE_GPU_CLUSTER"
UNINSTALL_CONFIRMATION = "UNINSTALL_GPU_FAULT"


class InjectedAcceptanceFailure(RuntimeError):
    pass


class AdminLifecycleBackend(Protocol):
    def snapshot(self) -> dict[str, Any]: ...

    def join(self, fault: str | None = None) -> dict[str, Any]: ...

    def capture_joined_token(self, cluster_id: str) -> dict[str, Any]: ...

    def remove(self, cluster_id: str) -> dict[str, Any]: ...

    def probe_revoked_token(self, capture: dict[str, Any]) -> dict[str, Any]: ...

    def uninstall(self) -> dict[str, Any]: ...

    def cleanup_sensitive_files(self) -> None: ...


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
        assert set(snapshot[field]) == expected, (field, snapshot)
    assert snapshot["cpu_control_plane_ready"] is True, snapshot


def run_admin_lifecycle(
    backend: AdminLifecycleBackend,
    recorder: EvidenceRecorder,
) -> dict[str, Any]:
    try:
        baseline = recorder.stage("baseline", backend.snapshot)
        baseline_ids = set(baseline["site_cluster_ids"])
        assert len(baseline_ids) == 1, (
            "BOOT-019 requires an isolated site with exactly one managed GPU cluster"
        )
        _assert_consistent_snapshot(baseline, baseline_ids)

        pre_commit = recorder.stage(
            "join_failure_before_site_commit",
            lambda: backend.join("before-site-commit"),
        )
        assert pre_commit["phase"] == "ROLLED_BACK", pre_commit
        _assert_consistent_snapshot(backend.snapshot(), baseline_ids)

        post_commit = recorder.stage(
            "join_failure_after_site_commit",
            lambda: backend.join("after-site-commit"),
        )
        assert post_commit["phase"] == "FAILED_AFTER_COMMIT", post_commit
        joined_id = str(post_commit["cluster_id"])
        assert joined_id not in baseline_ids

        joined = recorder.stage("join_resumed", backend.join)
        assert joined["phase"] == "COMPLETED", joined
        assert joined["cluster_id"] == joined_id, joined
        joined_ids = baseline_ids | {joined_id}
        after_join = recorder.stage("joined_snapshot", backend.snapshot)
        _assert_consistent_snapshot(after_join, joined_ids)

        token_capture = recorder.stage(
            "joined_token_captured",
            lambda: backend.capture_joined_token(joined_id),
        )
        removed = recorder.stage(
            "joined_cluster_removed",
            lambda: backend.remove(joined_id),
        )
        assert removed["phase"] == "COMPLETED", removed
        assert set(removed["remaining_cluster_ids"]) == baseline_ids, removed
        revoked = recorder.stage(
            "removed_token_rejected",
            lambda: backend.probe_revoked_token(token_capture),
        )
        assert revoked["status"] == 403, revoked
        assert revoked["detail"] == "regional cluster authentication failed", revoked
        after_remove = recorder.stage("post_remove_snapshot", backend.snapshot)
        _assert_consistent_snapshot(after_remove, baseline_ids)

        last_cluster_id = next(iter(baseline_ids))
        last_removed = recorder.stage(
            "last_cluster_removed",
            lambda: backend.remove(last_cluster_id),
        )
        assert last_removed["phase"] == "COMPLETED", last_removed
        assert last_removed["remaining_cluster_ids"] == [], last_removed
        empty = recorder.stage("empty_registry_snapshot", backend.snapshot)
        _assert_consistent_snapshot(empty, set())

        uninstall_result = recorder.stage("uninstall_keep_cpu", backend.uninstall)
        assert uninstall_result["cpu_cluster"] == "keep", uninstall_result
        assert uninstall_result["gpu_clusters"] == "preserved", uninstall_result
        assert uninstall_result["delete_policy_residuals"] == 0, uninstall_result
        backend.cleanup_sensitive_files()
        return recorder.complete()
    except BaseException as exc:
        recorder.fail(exc)
        raise


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
        final_snapshot_policy: str,
    ) -> None:
        self.site_path = site_path
        self.gpu_cluster_arn = gpu_cluster_arn
        self.cluster_id = cluster_id
        self.allowed_namespaces = allowed_namespaces
        self.join_state_dir = join_state_dir
        self.secure_dir = run_dir / "secure"
        self.secure_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.final_snapshot_policy = final_snapshot_policy

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
        state = self._join_state()
        if fault == "before-site-commit" and state.get("phase") == "ROLLED_BACK":
            return {
                **state,
                "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
            }
        if fault == "after-site-commit" and state.get("phase") == "FAILED_AFTER_COMMIT":
            return {
                **state,
                "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
            }

        original_deploy = admin_cluster_join._deploy_and_commit
        original_commit = admin_cluster_join._commit_site
        try:
            if fault == "before-site-commit":

                def fail_before_commit(*_args, **_kwargs):
                    raise InjectedAcceptanceFailure(
                        "BOOT-019 injected failure before site commit"
                    )

                admin_cluster_join._deploy_and_commit = fail_before_commit
            elif fault == "after-site-commit":

                def commit_then_fail(*args, **kwargs):
                    original_commit(*args, **kwargs)
                    raise InjectedAcceptanceFailure(
                        "BOOT-019 injected failure after site commit"
                    )

                admin_cluster_join._commit_site = commit_then_fail
            try:
                return join_cluster(self._join_request())
            except InjectedAcceptanceFailure:
                state = self._join_state()
                return {
                    **state,
                    "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
                }
        finally:
            admin_cluster_join._deploy_and_commit = original_deploy
            admin_cluster_join._commit_site = original_commit

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
        output = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return json.loads(output)

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
            "site_cluster_ids": site_ids,
            "registry_secret_cluster_ids": secret_ids,
            "release_state_cluster_ids": sorted(state.get("cluster_ids") or []),
            "installation_registry_cluster_ids": active_registry_ids,
            "cpu_control_plane_ready": desired > 0 and ready == desired,
        }

    def capture_joined_token(self, cluster_id: str) -> dict[str, Any]:
        site = self._site()
        record = next(
            item
            for item in site.release_config["clusters"]
            if item["cluster_id"] == cluster_id
        )
        source = Path(record["token_file"])
        token = source.read_bytes()
        destination = self.secure_dir / f"{cluster_id}.revoked-token"
        destination.write_bytes(token)
        destination.chmod(0o600)
        return {
            "cluster_id": cluster_id,
            "token_path": str(destination),
            "token_sha256": hashlib.sha256(token).hexdigest(),
            "control_plane_url": record["control_plane_url"],
            "ca_file": record["ca_file"],
        }

    def remove(self, cluster_id: str) -> dict[str, Any]:
        return remove_cluster(
            RemoveClusterRequest(
                site=self._site(),
                cluster_id=cluster_id,
                confirmation=REMOVE_CONFIRMATION,
            )
        )

    def probe_revoked_token(self, capture: dict[str, Any]) -> dict[str, Any]:
        token = Path(capture["token_path"]).read_text(encoding="utf-8").strip()
        request = urllib.request.Request(
            str(capture["control_plane_url"]).rstrip("/")
            + "/v1/regional/executors/readiness",
            data=b"{}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-GPU-Fault-Cluster-ID": str(capture["cluster_id"]),
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
        return uninstall(
            UninstallRequest(
                site=self._site(),
                cpu_disposition="keep",
                confirmation=UNINSTALL_CONFIRMATION,
                final_snapshot_policy=self.final_snapshot_policy,
            )
        )

    def cleanup_sensitive_files(self) -> None:
        for path in self.secure_dir.glob("*.revoked-token"):
            path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--site", required=True, type=Path)
    value.add_argument("--gpu-cluster-arn", required=True)
    value.add_argument("--cluster-id")
    value.add_argument(
        "--allowed-namespace",
        action="append",
        default=["gpu-fault-system", "training"],
    )
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument(
        "--aurora-final-snapshot",
        choices=("retain", "skip"),
        default="retain",
    )
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm")
    return value


def main() -> int:
    arguments = parser().parse_args()
    plan = {
        "case_id": CASE_ID,
        "site": str(arguments.site),
        "gpu_cluster_arn": arguments.gpu_cluster_arn,
        "run_dir": str(arguments.run_dir),
        "stages": [
            "inject pre-commit join failure and verify rollback",
            "inject post-commit join failure and verify fail-forward resume",
            "remove the joined cluster and reject its old token",
            "remove the final GPU cluster and verify explicit empty registry",
            "uninstall solution resources while preserving CPU/GPU clusters",
        ],
    }
    if not arguments.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if arguments.confirm != CONFIRMATION:
        raise SystemExit(f"--execute requires --confirm {CONFIRMATION}")
    arguments.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    inputs = {
        "site": str(arguments.site.resolve()),
        "gpu_cluster_arn": arguments.gpu_cluster_arn,
        "cluster_id": arguments.cluster_id,
        "allowed_namespaces": sorted(set(arguments.allowed_namespace)),
        "aurora_final_snapshot": arguments.aurora_final_snapshot,
    }
    recorder = EvidenceRecorder(
        arguments.run_dir / f"{CASE_ID}.json",
        case_id=CASE_ID,
        inputs=inputs,
    )
    backend = LiveAdminLifecycleBackend(
        site_path=arguments.site.resolve(),
        gpu_cluster_arn=arguments.gpu_cluster_arn,
        cluster_id=arguments.cluster_id,
        allowed_namespaces=tuple(sorted(set(arguments.allowed_namespace))),
        join_state_dir=arguments.run_dir / "join-state",
        run_dir=arguments.run_dir,
        final_snapshot_policy=arguments.aurora_final_snapshot,
    )
    result = run_admin_lifecycle(backend, recorder)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
