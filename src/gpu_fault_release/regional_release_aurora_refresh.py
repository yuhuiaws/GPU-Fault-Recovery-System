"""Versioned Aurora refresher objects belong to the release transaction.

Snapshots contain the CronJob and its namespaced RBAC, never the database
Secret. Rollback restores the previous program before asking it to synchronize
AWSCURRENT; it must not depend on a broken candidate program running first.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import repository_root
from gpu_fault_release.regional_aurora_credentials import CRONJOB_NAME
from gpu_fault_release.regional_manifest_snapshot import (
    apply_snapshot_objects,
    capture_declared_objects,
    declared_manifest_objects,
    delete_absent_objects,
    resource_argument,
    snapshot_parts,
)
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_preflight import require_cpu_namespace_anchor

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

MANIFEST_PATH = Path("deploy/control-plane/regional/aurora-credential-refresh.yaml")
MASTER_ARN_PLACEHOLDER = "REPLACE_WITH_AURORA_MASTER_SECRET_ARN"
SNAPSHOT_LABEL = "previous Aurora credential refresher snapshot"
OBJECT_RESOURCES = frozenset(
    {
        "serviceaccount",
        "role.rbac.authorization.k8s.io",
        "rolebinding.rbac.authorization.k8s.io",
        "cronjob.batch",
    }
)
SENSITIVE_ENVIRONMENT = re.compile(
    r"TOKEN|PASSWORD|PRIVATE_KEY|CREDENTIAL|(?:STORE|DATABASE)_URL|DSN"
)


def render_aurora_refresh(
    *,
    namespace: str,
    runtime_image: str,
    wheel_config_map: str,
    master_secret_arn: str = MASTER_ARN_PLACEHOLDER,
    source: str | None = None,
) -> str:
    replacements = {
        "gpu-fault-system": namespace,
        "gpu-fault-control-plane-wheel-0100": wheel_config_map,
        "public.ecr.aws/docker/library/python:3.12-slim": runtime_image,
        MASTER_ARN_PLACEHOLDER: master_secret_arn,
    }

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return replacements.get(value, value) if isinstance(value, str) else value

    documents = yaml.safe_load_all(
        source
        if source is not None
        else (repository_root() / MANIFEST_PATH).read_text()
    )
    rendered = [replace(item) for item in documents if item]
    for item in rendered:
        if item.get("kind") == "CronJob":
            item["spec"]["suspend"] = False
            container = item["spec"]["jobTemplate"]["spec"]["template"]["spec"][
                "containers"
            ][0]
            container["args"] = []
            container["env"] = [
                value
                for value in container.get("env", [])
                if value["name"] != "GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS"
            ] + [
                {
                    "name": "GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS",
                    "value": "false",
                }
            ]
    return str(yaml.safe_dump_all(rendered, sort_keys=False))


def aurora_master_secret_arn(release: RegionalRelease) -> str:
    # Read the reference only. Reading the whole Secret would unnecessarily
    # bring the database password into the deploy process and its diagnostics.
    encoded = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            "gpu-fault-aurora",
            "-o",
            "jsonpath={.data.master-secret-arn}",
            "--request-timeout=15s",
        ),
        capture=True,
        timeout_seconds=20,
    ).strip()
    try:
        arn = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ReleaseError("Aurora master Secret reference is invalid") from exc
    if not re.fullmatch(
        rf"arn:aws(?:-[a-z]+)*:secretsmanager:{re.escape(release.config.aws_region)}:"
        r"\d{12}:secret:[A-Za-z0-9/_+=.!@-]+",
        arn,
    ):
        raise ReleaseError("Aurora master Secret reference has an invalid identity")
    if arn.split(":")[4] != release.config.cpu_eks_arn.split(":")[4]:
        raise ReleaseError("Aurora master Secret reference belongs to another account")
    return arn


def _candidate_manifest(release: RegionalRelease, *, suspended: bool = False) -> str:
    manifest = render_aurora_refresh(
        namespace=release.config.namespace,
        runtime_image=release.runtime_image,
        wheel_config_map=release.wheel_cm,
        master_secret_arn=aurora_master_secret_arn(release),
    )
    if not suspended:
        return manifest
    documents = list(yaml.safe_load_all(manifest))
    for document in documents:
        if document["kind"] == "CronJob":
            document["spec"]["suspend"] = True
    return str(yaml.safe_dump_all(documents, sort_keys=False))


def validate_aurora_refresh_snapshot(
    release: RegionalRelease, snapshot: object
) -> tuple[str, list[Any], list[Any]]:
    namespace, objects, absent = snapshot_parts(snapshot, label=SNAPSHOT_LABEL)
    if namespace != release.config.namespace:
        raise ReleaseError(f"{SNAPSHOT_LABEL} namespace differs")
    identities: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise ReleaseError(f"{SNAPSHOT_LABEL} object is invalid")
        metadata = item["metadata"]
        resource = resource_argument(
            str(item.get("apiVersion") or ""), str(item.get("kind") or "")
        )
        if (
            metadata.get("namespace") != namespace
            or metadata.get("name") != CRONJOB_NAME
            or resource not in OBJECT_RESOURCES
            or resource in identities
        ):
            raise ReleaseError(f"{SNAPSHOT_LABEL} object identity differs")
        if resource == "cronjob.batch":
            _validate_refresher_references(item, namespace)
        if resource == "rolebinding.rbac.authorization.k8s.io":
            if item.get("roleRef") != {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": CRONJOB_NAME,
            } or item.get("subjects") != [
                {"kind": "ServiceAccount", "name": CRONJOB_NAME, "namespace": namespace}
            ]:
                raise ReleaseError(f"{SNAPSHOT_LABEL} RBAC identity differs")
        identities.add(resource)
    for item in absent:
        resource = item["resource"]
        if (
            item["name"] != CRONJOB_NAME
            or resource not in OBJECT_RESOURCES
            or resource in identities
        ):
            raise ReleaseError(f"{SNAPSHOT_LABEL} absence identity differs")
        identities.add(resource)
    if identities != OBJECT_RESOURCES:
        raise ReleaseError(f"{SNAPSHOT_LABEL} is incomplete")
    if any(item["resource"] == "cronjob.batch" for item in absent) and (
        not isinstance(snapshot, dict) or snapshot.get("absence_verified") is not True
    ):
        raise ReleaseError(f"{SNAPSHOT_LABEL} absence is unverified")
    return namespace, objects, absent


def _validate_refresher_references(document: dict[str, Any], namespace: str) -> None:
    try:
        pod = document["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        containers = pod["containers"]
        if (
            not isinstance(containers, list)
            or not containers
            or pod.get("serviceAccountName") != CRONJOB_NAME
        ):
            raise ValueError("invalid Pod identity")
        for container in [*containers, *pod.get("initContainers", [])]:
            for item in container.get("env", []):
                name = str(item.get("name") or "")
                if "value" in item and SENSITIVE_ENVIRONMENT.search(name):
                    raise ReleaseError(
                        f"{SNAPSHOT_LABEL} contains a sensitive literal environment field"
                    )
                expected = {
                    "GPU_FAULT_NAMESPACE": namespace,
                    "GPU_FAULT_AURORA_SECRET": "gpu-fault-aurora",
                }.get(name)
                if expected is not None and item.get("value") != expected:
                    raise ValueError("invalid Secret or namespace reference")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ReleaseError(
            f"{SNAPSHOT_LABEL} program identity is incomplete or differs"
        ) from exc


def capture_aurora_refresh_snapshot(
    release: RegionalRelease, *, bootstrap: bool = False
) -> dict[str, Any]:
    declared = declared_manifest_objects(
        (repository_root() / MANIFEST_PATH).read_text(),
        label="Aurora refresher manifest",
    )
    if (
        len(declared) != len(OBJECT_RESOURCES)
        or {item["resource"] for item in declared if item["name"] == CRONJOB_NAME}
        != OBJECT_RESOURCES
    ):
        raise ReleaseError("Aurora refresher manifest declares an unexpected resource")
    captured = capture_declared_objects(
        release, declared, label="live Aurora refresher"
    )
    if captured["absent"]:
        require_cpu_namespace_anchor(release, bootstrap=bootstrap)
        captured["absence_verified"] = True
    validate_aurora_refresh_snapshot(release, captured)
    return captured


def aurora_refresh_preflight(
    release: RegionalRelease, *, bootstrap: bool = False
) -> dict[str, Any]:
    snapshot = capture_aurora_refresh_snapshot(release, bootstrap=bootstrap)
    return {"absent": snapshot["absent"], "objects": len(snapshot["objects"])}


def require_aurora_refresh_snapshot(
    release: RegionalRelease, previous: dict[str, Any], plan: ReleaseExecutionPlan
) -> None:
    if plan.has(ReleaseComponent.AURORA_REFRESH) or (
        release.config.auto_rollback
        and (
            "aurora_refresh" not in previous
            and plan.has(
                ReleaseComponent.CPU_STAGE,
                ReleaseComponent.CPU_FINALIZE,
            )
        )
    ):
        validate_aurora_refresh_snapshot(release, previous.get("aurora_refresh"))


def _manifest_diff(release: RegionalRelease, manifest: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-aurora-probe-") as directory:
        path = Path(directory) / "refresher.yaml"
        path.write_text(manifest)
        path.chmod(0o600)
        code, _output, _error = release.runner.probe_output(
            release._cpu("diff", "-f", str(path)), timeout_seconds=60
        )
    if code not in {0, 1}:
        raise ReleaseError("cannot read Aurora refresher drift; state is unknown")
    return code == 1


def aurora_refresh_drift(release: RegionalRelease, *, suspended: bool = False) -> bool:
    if release.runner.dry_run or not release.config.delivery_component_digests.get(
        "aurora_refresh"
    ):
        return False
    return _manifest_diff(release, _candidate_manifest(release, suspended=suspended))


def apply_aurora_refresh(
    release: RegionalRelease, *, refresh: bool = True, suspended: bool = False
) -> None:
    release.enforce_manifest_plan_pin()
    manifest = _candidate_manifest(release, suspended=suspended)
    release.runner.run(
        release._cpu("apply", "--dry-run=server", "-f", "-"), input_text=manifest
    )
    release.runner.run(release._cpu("apply", "-f", "-"), input_text=manifest)
    if not release.runner.dry_run and _manifest_diff(release, manifest):
        raise ReleaseError("candidate Aurora refresher did not converge before refresh")
    if refresh:
        release._refresh_aurora_credentials(required=True)


def verify_aurora_refresh_snapshot(release: RegionalRelease, snapshot: object) -> None:
    namespace, objects, absent = validate_aurora_refresh_snapshot(release, snapshot)
    if objects and _manifest_diff(
        release, json.dumps({"apiVersion": "v1", "kind": "List", "items": objects})
    ):
        raise ReleaseError("rollback Aurora credential refresher did not converge")
    for item in absent:
        output = release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "get",
                item["resource"],
                item["name"],
                "--ignore-not-found",
                "-o",
                "name",
                "--request-timeout=15s",
            ),
            capture=True,
            timeout_seconds=20,
        )
        if output.strip():
            raise ReleaseError(
                "rollback Aurora credential refresher absence did not converge"
            )


def restore_aurora_refresh_snapshot(release: RegionalRelease, snapshot: object) -> None:
    namespace, objects, absent = validate_aurora_refresh_snapshot(release, snapshot)
    apply_snapshot_objects(release, objects)
    delete_absent_objects(release, namespace, absent)
    verify_aurora_refresh_snapshot(release, snapshot)
