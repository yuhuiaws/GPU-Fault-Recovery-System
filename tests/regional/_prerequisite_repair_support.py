"""In-memory Kubernetes/RDS model for prerequisite transactions, never a CLI."""

from __future__ import annotations

import base64
import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import yaml

from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from gpu_fault_release import regional_release_aurora_refresh as REFRESH
from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._release_orchestrator_support import phase_release
from tests.regional.test_release_aurora_refresh_transaction import (
    CANDIDATE,
    MASTER_ARN,
    NAMESPACE,
    ObjectRunner,
    objects,
)


class RepairRunner(ObjectRunner):
    def __init__(self, *, bootstrap: bool = False) -> None:
        super().__init__([] if bootstrap else objects())
        self.cloud_state: dict[str, Any] = (
            {}
            if bootstrap
            else {
                "phase": "complete",
                "release_id": "old",
                "transaction_committed": True,
                "updated_at_epoch": 1,
            }
        )
        self.jobs: dict[str, dict[str, Any]] = {}
        self.created_jobs: list[dict[str, Any]] = []
        self.outputs: dict[str, str] = {}
        self.refresh_status: dict[str, Any] | None = None
        self.create_ack_lost = False
        self.fail_job = ""
        self.fail_delete = False
        self.foreign_job = False
        self.replace_on_completion = False
        self.unsafe_store = False
        self.database_state = "uninitialized_empty" if bootstrap else "initialized"
        self.snapshots: list[dict[str, Any]] = []
        self.namespace_uid = "namespace-uid"
        self.aurora_cluster_id = "aurora"
        self.database_resource_id = "cluster-resource"
        self.proof_schema_version = LATEST_POSTGRES_SCHEMA_VERSION
        self.store_blockers = {"workflow": 0, "remote_command": 0, "observation": 0}

    def run(self, args: list[str], **kwargs: Any) -> str:
        if args[:2] == ["aws", "rds"]:
            return json.dumps(
                {
                    "DBClusters": [
                        {
                            "DBClusterIdentifier": self.aurora_cluster_id,
                            "DBClusterArn": (
                                "arn:aws:rds:us-east-1:123456789012:cluster:"
                                + self.aurora_cluster_id
                            ),
                            "DbClusterResourceId": self.database_resource_id,
                            "Engine": "aurora-postgresql",
                            "Status": "available",
                            "Endpoint": "aurora.example.rds.amazonaws.com",
                            "Port": 5432,
                            "DatabaseName": "gpu_fault",
                            "MasterUsername": "administrator",
                            "MasterUserSecret": {"SecretArn": MASTER_ARN},
                        }
                    ]
                }
            )
        if "get" in args:
            kind = args[args.index("get") + 1]
            if kind == "namespace":
                return json.dumps(
                    {
                        "kind": "Namespace",
                        "metadata": {"name": NAMESPACE, "uid": self.namespace_uid},
                    }
                )
            if kind == "configmap":
                if not self.cloud_state:
                    return ""
                return json.dumps(
                    {
                        "kind": "ConfigMap",
                        "metadata": {
                            "name": "gpu-fault-regional-release-state",
                            "namespace": NAMESPACE,
                            "uid": "state-uid",
                        },
                        "data": {"state.json": json.dumps(self.cloud_state)},
                    }
                )
            if kind == "deployment":
                return json.dumps(
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "gpu-fault-api-ha",
                            "namespace": NAMESPACE,
                        },
                    }
                )
            if kind == "job":
                value = copy.deepcopy(self.jobs.get(args[args.index("get") + 2]))
                if value is not None and self.foreign_job:
                    value["metadata"]["uid"] = "foreign-uid"
                return json.dumps(value) if value else ""
            if r"jsonpath={.data.last-refresh-status\.json}" in args:
                return (
                    base64.b64encode(json.dumps(self.refresh_status).encode()).decode()
                    if self.refresh_status
                    else ""
                )
        if ("apply" in args or "create" in args) and "-f" in args:
            text = kwargs.get("input_text")
            documents = list(yaml.safe_load_all(text)) if text else []
            if len(documents) == 1 and documents[0].get("kind") == "Job":
                document = copy.deepcopy(documents[0])
                if "--dry-run=server" in args:
                    self.events.append("job-admission")
                    return json.dumps(document)
                self.events.append("job-create")
                name = document["metadata"]["name"]
                document["metadata"]["uid"] = f"job-{len(self.created_jobs)}"
                self.jobs[name] = document
                self.created_jobs.append(copy.deepcopy(document))
                if self.create_ack_lost:
                    self.create_ack_lost = False
                    raise ReleaseError("create ACK lost")
                return json.dumps(document)
        if any(value.endswith("/wait-for-kubernetes-job.sh") for value in args):
            name = args[3]
            job = self.jobs[name]
            container = job["spec"]["template"]["spec"]["containers"][0]
            mode = "store" if container["name"] == "proof" else "credential"
            self.events.append(f"{mode}-job")
            if self.fail_job == mode:
                raise ReleaseError(f"{mode} Job failed")
            job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
            if self.replace_on_completion:
                job["metadata"]["uid"] = "replacement-uid"
            if mode == "credential":
                self.refresh_status = {
                    "status": "ok",
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            else:
                environment = {
                    item["name"]: item.get("value") for item in container["env"]
                }
                self.outputs[name] = json.dumps(
                    {
                        "safe": not self.unsafe_store,
                        "run_id": environment["GPU_FAULT_PROOF_RUN_ID"],
                        "identity_sha256": environment[
                            "GPU_FAULT_PROOF_IDENTITY_SHA256"
                        ],
                        "database_state": self.database_state,
                        "schema_version": (
                            0
                            if self.database_state == "uninitialized_empty"
                            else self.proof_schema_version
                        ),
                        "finished_at": datetime.now(UTC).isoformat(),
                        "schema_ensure_required": (
                            self.database_state == "initialized"
                            and self.proof_schema_version
                            < LATEST_POSTGRES_SCHEMA_VERSION
                        ),
                        "blockers": {
                            **self.store_blockers,
                            "workflow": int(self.unsafe_store)
                            + self.store_blockers["workflow"],
                        },
                    }
                )
            return ""
        if "logs" in args:
            name = args[args.index("logs") + 1].removeprefix("job/")
            return self.outputs[name]
        if "delete" in args and "--raw" in args:
            if self.fail_delete:
                raise ReleaseError("cleanup unavailable")
            name = args[args.index("--raw") + 1].rsplit("/", 1)[-1]
            options = json.loads(kwargs["input_text"])
            assert options["propagationPolicy"] == "Foreground", (
                "proof cleanup must wait for its Pods"
            )
            assert (
                options["preconditions"]["uid"] == self.jobs[name]["metadata"]["uid"]
            ), "cleanup attempted to delete a foreign Job UID"
            self.jobs.pop(name)
            self.events.append("job-delete")
            return ""
        if "wait" in args and "--for=delete" in args:
            name = next(
                value for value in args if value.startswith("job/")
            ).removeprefix("job/")
            assert name not in self.jobs, "proof cleanup returned before deletion"
            return ""
        return str(super().run(args, **kwargs))


def repair_release(monkeypatch: Any, *, bootstrap: bool = False) -> SimpleNamespace:
    runner = RepairRunner(bootstrap=bootstrap)
    calls = runner.events
    config = SimpleNamespace(
        namespace=NAMESPACE,
        aws_region="us-east-1",
        site_name="proof-site",
        cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        database_schema_version=LATEST_POSTGRES_SCHEMA_VERSION,
        health=SimpleNamespace(aurora_cluster_id="aurora"),
        delivery_component_digests={"aurora_refresh": "manifest"},
        clusters=(),
        auto_rollback=False,
        upgrade_max_parallel_clusters=1,
        installation_id=None,
        retained_database_handoff=None,
    )

    def save(phase: str, **updates: Any) -> None:
        instance.state.update(phase=phase, **updates)
        instance.state["updated_at_epoch"] = (
            int(instance.state.get("updated_at_epoch", 0)) + 1
        )
        runner.cloud_state = copy.deepcopy(instance.state)
        runner.snapshots.append(copy.deepcopy(instance.state))
        calls.append("persist")

    def load() -> dict[str, Any]:
        instance.state = copy.deepcopy(runner.cloud_state)
        return instance.state

    def gate() -> bool:
        assert runner.refresh_status is not None, (
            "Store gate ran before AWSCURRENT repair"
        )
        calls.append("store-gate")
        return True

    def capture(**_kwargs: Any) -> dict[str, Any]:
        calls.append("full-snapshot")
        return {
            "release_id": "old",
            "aurora_refresh": REFRESH.capture_aurora_refresh_snapshot(instance),
        }

    instance = phase_release(
        calls,
        config=config,
        runner=runner,
        release_id="candidate",
        runtime_image=CANDIDATE,
        wheel_cm="candidate-wheel",
        rendered_manifest_digest="a" * 64,
        approved_manifest_digest=None,
        state=copy.deepcopy(runner.cloud_state),
        _load_state=load,
        _save_state=lambda phase, **updates: save(
            phase, release_id="candidate", **updates
        ),
        _cpu=lambda *args: ["kubectl", "--context", "cpu", *args],
        _get_json=lambda args: json.loads(runner.run(list(args))),
        _ensure_contexts=lambda: calls.append("contexts"),
        _scale_if_present=lambda *_args, **_kwargs: calls.append("scale-if-present"),
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=gate,
        _require_no_inflight_installs=lambda **_kwargs: {"checked": gate()},
        _refresh_aurora_credentials=lambda **_kwargs: calls.append("credential-check"),
        _backup_release_secrets=lambda: {},
        _capture_previous=capture,
        _validate_resume_checkpoint=lambda **_kwargs: calls.append("resume-checkpoint"),
        _aurora_refresh_drift=lambda: REFRESH.aurora_refresh_drift(instance),
        enforce_manifest_plan_pin=lambda: calls.append("approval"),
        pin_approved_manifest_plan=lambda digest: setattr(
            instance, "approved_manifest_digest", digest
        ),
    )
    monkeypatch.setattr(
        REPAIR,
        "save_recorded_state",
        lambda _release, phase, **updates: save(phase, **updates),
    )
    return instance
