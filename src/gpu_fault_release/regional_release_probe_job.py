"""Bounded CPU proof Jobs with persisted ownership and UID-scoped cleanup.

The admitted Job template is verified. This does not attest to later Pod
admission mutations; the CPU namespace's Pod admission policy remains trusted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import (
    DeploymentDeadlineExceeded,
    cleanup_deadline,
    deadline_scope,
    remaining_timeout,
)
from gpu_fault.admin.process_supervisor import (
    ensure_supervision_safe,
    write_diagnostic,
)
from gpu_fault_release import repository_root
from gpu_fault_release.regional_aurora_credentials import (
    CRONJOB_NAME,
    read_aurora_refresh_cronjob,
)
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_probe_job_diagnostics import (
    collect_job_diagnostics,
)
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

RUN_ANNOTATION = "gpu-fault.io/release-proof-run"
SPEC_ANNOTATION = "gpu-fault.io/release-proof-spec-sha256"
Checkpoint = Callable[[dict[str, Any]], None]


def probe_job_record(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document["metadata"]
    return {
        "name": metadata["name"],
        "namespace": metadata["namespace"],
        "run_id": metadata["annotations"][RUN_ANNOTATION],
        "spec_sha256": canonical_sha256(document["spec"]),
        "admitted_spec_sha256": _job_spec_sha256(document),
        "owner_uid": metadata["ownerReferences"][0]["uid"],
        "uid": None,
        "status": "PLANNED",
    }


def _read_owned_job(
    release: RegionalRelease, record: dict[str, Any]
) -> dict[str, Any] | None:
    run_id = str(record.get("run_id") or "")
    if (
        record.get("namespace") != release.config.namespace
        or re.fullmatch(r"[a-f0-9]{32}", run_id) is None
        or record.get("name")
        not in {f"gpu-fault-store-proof-{run_id[:16]}", f"{CRONJOB_NAME}-{run_id[:8]}"}
        or any(
            re.fullmatch(r"[a-f0-9]{64}", str(record.get(key) or "")) is None
            for key in ("spec_sha256", "admitted_spec_sha256")
        )
    ):
        raise ReleaseError("proof Job namespace or ownership record differs")
    value = cast(
        dict[str, Any] | None,
        probe_resource(
            release.runner,
            release._cpu(),
            ResourceRef("job", "Job", str(record["name"]), release.config.namespace),
        ).require_readable(),
    )
    if value is None:
        return None
    metadata = value.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    if (
        annotations.get(RUN_ANNOTATION) != record["run_id"]
        or annotations.get(SPEC_ANNOTATION) != record["spec_sha256"]
        or (record.get("uid") and metadata.get("uid") != record["uid"])
        or _job_spec_sha256(value) != record["admitted_spec_sha256"]
        or (
            record.get("owner_uid")
            and metadata.get("ownerReferences")
            != [_proof_owner(str(record["owner_uid"]))]
        )
    ):
        raise ReleaseError("proof Job ownership or UID differs")
    return value


def _proof_owner(uid: str) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "name": CRONJOB_NAME,
        "uid": uid,
        "controller": False,
        "blockOwnerDeletion": False,
    }


def _drop_defaults(value: dict[str, Any], defaults: dict[str, Any]) -> None:
    for name, default in defaults.items():
        if value.get(name) == default:
            value.pop(name, None)


def _job_spec_sha256(document: dict[str, Any]) -> str:
    """Compare the entire spec, allowing only explicit API/controller defaults."""
    spec = json.loads(json.dumps(document["spec"]))
    metadata = document.get("metadata") or {}
    uid, name = metadata.get("uid"), metadata.get("name")
    if uid and not spec.get("manualSelector"):
        for key in ("batch.kubernetes.io/controller-uid", "controller-uid"):
            if spec.get("selector") == {"matchLabels": {key: uid}}:
                spec.pop("selector")
                break
    _drop_defaults(
        spec,
        {
            "parallelism": 1,
            "completions": 1,
            "completionMode": "NonIndexed",
            "manualSelector": False,
            "suspend": False,
            "podReplacementPolicy": "TerminatingOrFailed",
            "managedBy": "kubernetes.io/job-controller",
        },
    )
    template = spec["template"]
    template_metadata = template.get("metadata") or {}
    labels = template_metadata.get("labels") or {}
    for key, expected in (
        ("batch.kubernetes.io/controller-uid", uid),
        ("controller-uid", uid),
        ("batch.kubernetes.io/job-name", name),
        ("job-name", name),
    ):
        if expected and labels.get(key) == expected:
            labels.pop(key)
    if not labels:
        template_metadata.pop("labels", None)
    if template_metadata.get("creationTimestamp") is None:
        template_metadata.pop("creationTimestamp", None)
    if not template_metadata:
        template.pop("metadata", None)
    pod = template["spec"]
    if pod.get("serviceAccount") == pod.get("serviceAccountName", "default"):
        pod.pop("serviceAccount", None)
    _drop_defaults(
        pod,
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
            "enableServiceLinks": True,
            "serviceAccountName": "default",
            "securityContext": {},
            "hostNetwork": False,
            "hostPID": False,
            "hostIPC": False,
        },
    )
    for container in [*pod.get("containers", []), *pod.get("initContainers", [])]:
        _drop_defaults(
            container,
            {
                "args": [],
                "terminationMessagePath": "/dev/termination-log",
                "terminationMessagePolicy": "File",
                "resources": {},
            },
        )
        if "@sha256:" in str(container.get("image") or ""):
            _drop_defaults(container, {"imagePullPolicy": "IfNotPresent"})
    for volume in pod.get("volumes", []):
        for key in ("secret", "configMap", "projected", "downwardAPI"):
            if isinstance(volume.get(key), dict):
                _drop_defaults(volume[key], {"defaultMode": 420})
    return canonical_sha256(spec)


def cleanup_probe_job(
    release: RegionalRelease, record: dict[str, Any], checkpoint: Checkpoint
) -> None:
    ensure_supervision_safe(allow_interrupted=True)
    with cleanup_deadline("release proof Job cleanup", 90):
        value = _read_owned_job(release, record)
        if value is not None:
            uid = value["metadata"]["uid"]
            if not record.get("uid"):
                record.update(uid=uid)
                checkpoint(record)
            uri = (
                f"/apis/batch/v1/namespaces/{release.config.namespace}"
                f"/jobs/{record['name']}"
            )
            release.runner.run(
                release._cpu("delete", "--raw", uri, "-f", "-"),
                input_text=json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "preconditions": {"uid": uid},
                        "propagationPolicy": "Foreground",
                    }
                ),
                capture=True,
                timeout_seconds=20,
            )
            release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "wait",
                    "--for=delete",
                    f"job/{record['name']}",
                    "--timeout=45s",
                ),
                capture=True,
                timeout_seconds=50,
            )
            if _read_owned_job(release, record) is not None:
                raise ReleaseError("proof Job cleanup did not converge")
        record.update(status="REMOVED")
        checkpoint(record)


def _diagnose_failure(
    release: RegionalRelease, record: dict[str, Any], failure: BaseException
) -> None:
    try:
        with deadline_scope("release proof Job diagnostics", 20):
            ensure_supervision_safe()
            job = _read_owned_job(release, record)
            if job is None:
                return
            summary = collect_job_diagnostics(release, job)
            current = _read_owned_job(release, record)
            if current is None or current["metadata"]["uid"] != job["metadata"]["uid"]:
                raise ReleaseError("proof Job changed during diagnostics")
            remaining_timeout(1)
            detail = diagnostic_text(json.dumps(summary), limit=3072)
    except DeploymentDeadlineExceeded:
        raise
    except Exception as diagnostic:
        ensure_supervision_safe()
        remaining_timeout(1)
        detail = f"unavailable ({type(diagnostic).__name__})"
    message = f"proof Job {record['name']} diagnostics: {detail}"
    failure.add_note(message)
    write_diagnostic(message + "\n", final=True)


def run_probe_job(
    release: RegionalRelease,
    document: dict[str, Any],
    checkpoint: Checkpoint,
    *,
    verify: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist before create; a lost create ACK is recovered by its run binding."""
    owner = read_aurora_refresh_cronjob(release)
    metadata = (owner or {}).get("metadata") or {}
    if (
        not isinstance(metadata.get("uid"), str)
        or not metadata["uid"]
        or metadata.get("deletionTimestamp")
    ):
        raise ReleaseError("proof Job has no live registered refresher owner")
    document["metadata"]["ownerReferences"] = [_proof_owner(metadata["uid"])]
    record = probe_job_record(document)
    document["metadata"]["annotations"][SPEC_ANNOTATION] = record["spec_sha256"]
    checkpoint(record)
    failure: BaseException | None = None
    try:
        rendered = json.dumps(document)
        preview = release.runner.run(
            release._cpu("apply", "--dry-run=server", "-f", "-", "-o", "json"),
            input_text=rendered,
            capture=True,
            timeout_seconds=30,
        )
        try:
            admitted = json.loads(preview)
            admitted_metadata = admitted["metadata"]
            if (
                _job_spec_sha256(admitted) != record["admitted_spec_sha256"]
                or admitted_metadata.get("name") != record["name"]
                or admitted_metadata.get("namespace") != record["namespace"]
                or admitted_metadata.get("ownerReferences")
                != document["metadata"]["ownerReferences"]
                or any(
                    (admitted_metadata.get("annotations") or {}).get(key) != expected
                    for key, expected in (
                        (RUN_ANNOTATION, record["run_id"]),
                        (SPEC_ANNOTATION, record["spec_sha256"]),
                    )
                )
            ):
                raise ValueError("admission changed the proof program")
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseError("proof Job admission identity differs") from exc
        release.runner.run(
            release._cpu("create", "-f", "-"),
            input_text=rendered,
            capture=True,
            timeout_seconds=30,
        )
        value = _read_owned_job(release, record)
        if value is None:
            raise ReleaseError("created proof Job is absent")
        record.update(uid=value["metadata"]["uid"], status="RUNNING")
        checkpoint(record)
        release.runner.run(
            [
                "bash",
                str(
                    repository_root()
                    / "deploy/control-plane/tools/wait-for-kubernetes-job.sh"
                ),
                "300",
                record["name"],
                *release._cpu("-n", release.config.namespace),
            ],
            capture=True,
            timeout_seconds=330,
        )
        value = _read_owned_job(release, record)
        if value is None or not any(
            item.get("type") == "Complete" and item.get("status") == "True"
            for item in (value.get("status") or {}).get("conditions", [])
        ):
            raise ReleaseError("proof Job has no UID-bound completion")
        if verify is not None:
            return verify()
        raw = release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "logs",
                f"job/{record['name']}",
                "--all-containers=true",
                "--tail=-1",
                "--limit-bytes=65536",
            ),
            capture=True,
            sensitive=True,
            timeout_seconds=20,
        )
        try:
            result = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ReleaseError("database proof Job returned invalid evidence") from exc
        if not isinstance(result, dict):
            raise ReleaseError("database proof Job returned non-object evidence")
        result["job_uid"] = record["uid"]
        return result
    except BaseException as exc:
        failure = exc
        try:
            if isinstance(exc, Exception) and not isinstance(
                exc, DeploymentDeadlineExceeded
            ):
                _diagnose_failure(release, record, exc)
        except BaseException as diagnostic:
            failure = diagnostic
            raise
        raise
    finally:
        try:
            cleanup_probe_job(release, record, checkpoint)
        except Exception as cleanup:
            if failure is None:
                raise
            failure.add_note(f"proof Job cleanup also failed: {type(cleanup).__name__}")
