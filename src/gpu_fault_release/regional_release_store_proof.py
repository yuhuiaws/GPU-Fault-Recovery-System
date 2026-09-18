"""Fresh database identity and bootstrap safety proof without a business Pod."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.installation_lifecycle import aurora_binding
from gpu_fault_release import regional_release_store_probe, repository_root
from gpu_fault_release.regional_release_aurora_refresh import aurora_master_secret_arn
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_probe_job import (
    RUN_ANNOTATION,
    Checkpoint,
    run_probe_job,
)
from gpu_fault_release.regional_release_runtime_identity import CONTROL_PLANE_PYTHON
from gpu_fault_release.regional_release_state import aws_json
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease


def database_identity(release: RegionalRelease) -> dict[str, Any]:
    namespace = cast(
        dict[str, Any] | None,
        probe_resource(
            release.runner,
            release._cpu(),
            ResourceRef("namespace", "Namespace", release.config.namespace, None),
        ).require_readable(),
    )
    if namespace is None or (namespace.get("metadata") or {}).get("deletionTimestamp"):
        raise ReleaseError("database proof has no live namespace identity")
    cluster_id = release.config.health.aurora_cluster_id
    if not cluster_id:
        raise ReleaseError("database proof requires an Aurora cluster identity")
    values = aws_json(
        release,
        ["rds", "describe-db-clusters", "--db-cluster-identifier", cluster_id],
        cached=False,
    ).get("DBClusters")
    if not isinstance(values, list) or len(values) != 1:
        raise ReleaseError("database proof Aurora identity is unavailable")
    try:
        binding = aurora_binding(
            values[0],
            cpu_eks_arn=release.config.cpu_eks_arn,
            aws_region=release.config.aws_region,
            cluster_id=cluster_id,
            master_secret_arn=aurora_master_secret_arn(release),
        )
    except BootstrapError:
        raise ReleaseError(
            "database proof Aurora binding differs or is incomplete"
        ) from None
    return {
        **binding,
        "cpu_eks_arn": release.config.cpu_eks_arn,
        "namespace": release.config.namespace,
        "namespace_uid": namespace["metadata"]["uid"],
        "schema_version": release.config.database_schema_version,
    }


def bootstrap_store_proof(
    release: RegionalRelease,
    checkpoint: Checkpoint,
    *,
    require_empty: bool = True,
    allow_retained_schema_upgrade: bool = False,
) -> dict[str, Any]:
    if require_empty and allow_retained_schema_upgrade:
        raise ReleaseError("empty bootstrap cannot authorize retained schema adoption")
    identity = database_identity(release)
    run_id = uuid.uuid4().hex
    started = datetime.now(UTC)
    document = yaml.safe_load(
        (
            repository_root() / "deploy/migrations/postgres-schema-preflight-job.yaml"
        ).read_text()
    )
    document["metadata"] = {
        "name": f"gpu-fault-store-proof-{run_id[:16]}",
        "namespace": release.config.namespace,
        "annotations": {
            RUN_ANNOTATION: run_id,
            "gpu-fault.io/cleanup-phase": "auxiliary",
            "gpu-fault.io/cleanup-order": "30",
        },
    }
    document["spec"].update(
        backoffLimit=0, activeDeadlineSeconds=120, ttlSecondsAfterFinished=300
    )
    pod = document["spec"]["template"]["spec"]
    pod.update(serviceAccountName="default", automountServiceAccountToken=False)
    container = pod["containers"][0]
    container.update(
        name="proof",
        image=release.runtime_image,
        command=[
            CONTROL_PLANE_PYTHON,
            "-I",
            "-c",
            Path(regional_release_store_probe.__file__).read_text(),
        ],
        args=[],
        securityContext={
            "runAsNonRoot": True,
            "runAsUser": 65534,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
    )
    container["env"] = [
        item
        for item in container["env"]
        if item["name"] in {"PATH", "GPU_FAULT_STORE_URL", "GPU_FAULT_RDS_CA_BUNDLE"}
    ] + [
        {"name": "GPU_FAULT_PROOF_DATABASE", "value": json.dumps(identity)},
        {"name": "GPU_FAULT_PROOF_RUN_ID", "value": run_id},
        {
            "name": "GPU_FAULT_PROOF_IDENTITY_SHA256",
            "value": canonical_sha256(identity),
        },
        {
            "name": "GPU_FAULT_PROOF_RETAINED_SCHEMA_UPGRADE",
            "value": "true" if allow_retained_schema_upgrade else "false",
        },
    ]
    pod["volumes"] = [
        item for item in pod["volumes"] if item["name"] == "rds-ca-bundle"
    ]
    container["volumeMounts"] = [
        item for item in container["volumeMounts"] if item["name"] == "rds-ca-bundle"
    ]
    result = run_probe_job(release, document, checkpoint)
    if database_identity(release) != identity:
        raise ReleaseError("database proof identity changed during the read")
    checks: dict[str, bool] = {}

    def check(name: str, value: bool) -> bool:
        # Record only evaluated predicates; unevaluated checks are not failures.
        checks[name] = value
        return value

    reason = "predicate_failed"
    try:
        finished = datetime.fromisoformat(result["finished_at"])
        valid = (
            check("run_id_match", result["run_id"] == run_id)
            and check(
                "identity_match",
                result["identity_sha256"] == canonical_sha256(identity),
            )
            and check("timestamp_aware", finished.tzinfo is not None)
            and check("timestamp_after_start", started <= finished)
            and check("timestamp_not_future", finished <= datetime.now(UTC))
            and check("safe", result["safe"] is True)
            and check(
                "blockers_clear",
                result["blockers"]
                == {"workflow": 0, "remote_command": 0, "observation": 0},
            )
            and check(
                "database_schema",
                (
                    result["database_state"] == "uninitialized_empty"
                    and result["schema_version"] == 0
                )
                or (
                    result["database_state"] == "initialized"
                    and type(result["schema_version"]) is int
                    and (
                        result["schema_version"] == identity["schema_version"]
                        or allow_retained_schema_upgrade
                        and regional_release_store_probe.compatible_retained_schema(
                            result["schema_version"], identity["schema_version"]
                        )
                        and result.get("schema_ensure_required") is True
                    )
                ),
            )
            and check(
                "empty_requirement",
                not require_empty or result["database_state"] == "uninitialized_empty",
            )
        )
    except KeyError:
        valid = False
        reason = "missing_field"
    except TypeError:
        valid = False
        reason = "invalid_type"
    except ValueError:
        valid = False
        reason = "invalid_timestamp"
    if not valid:
        stage = result.get("probe_stage")
        error_type = result.get("error_type")
        diagnostics = {
            "validation_reason": next(
                (name for name, passed in checks.items() if not passed), reason
            ),
            "probe_stage": stage
            if isinstance(stage, str)
            and stage
            in {item.value for item in regional_release_store_probe.ProbeStage}
            else "unknown",
            "error_type": error_type
            if isinstance(error_type, str)
            and error_type in regional_release_store_probe.PROBE_ERROR_TYPES
            else "unknown",
            **{
                f"has_{name}": name in result
                for name in (
                    "run_id",
                    "identity_sha256",
                    "finished_at",
                    "safe",
                    "blockers",
                    "database_state",
                    "schema_version",
                )
            },
            **checks,
        }
        raise ReleaseError(
            "bootstrap database/workflow safety proof failed or is incomplete; "
            f"diagnostics={json.dumps(diagnostics, sort_keys=True)}"
        ) from None
    return result
