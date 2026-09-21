"""Creation-ACK custody and actual admitted Job/Pod verification."""

from __future__ import annotations

import urllib.parse
from typing import Any

from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.destr008_watchdog_journal import (
    ARM_SECONDS,
    KINDS,
    LifecycleOwner,
    ManagedPod,
    ManagedResource,
    ObserverJob,
    object_document as _object,
    parse_document as _document,
    resource_fingerprint as _fingerprint,
)
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


class WatchdogAdmission:
    def __init__(self, owner: LifecycleOwner) -> None:
        self.owner = owner

    def identity(self, actual: dict[str, Any], item: ManagedResource) -> dict[str, Any]:
        meta = _object(actual.get("metadata"))
        version, kind, _, _ = KINDS[item.kind]
        if (
            actual.get("apiVersion") != version
            or actual.get("kind") != kind
            or meta.get("namespace") != self.owner.runtime.namespace
            or meta.get("name") != item.name
            or not isinstance(meta.get("uid"), str)
            or not meta["uid"]
            or not isinstance(meta.get("resourceVersion"), str)
            or not meta["resourceVersion"]
            or (item.uid is not None and meta["uid"] != item.uid)
        ):
            raise RegionalFixtureError("watchdog resource UID or namespace differs")
        return meta

    def unchanged(self, actual: dict[str, Any], item: ManagedResource) -> None:
        self.identity(actual, item)
        if item.uid is None:
            raise RegionalFixtureError(
                "watchdog creation outcome is unknown; adoption is forbidden"
            )
        mutable_control = (
            item.approved and item.kind == "configmap" and item.name == self.owner.name
        )
        expected = item.shape_sha256 if item.approved else item.ack_sha256
        if _fingerprint(actual, control=mutable_control) != expected:
            raise RegionalFixtureError(
                "watchdog resource changed after its creation acknowledgement"
            )

    def create(self, expected: dict[str, Any]) -> None:
        kind = expected["kind"].lower()
        name = expected["metadata"]["name"]
        key = kind + "/" + name
        existing = self.owner.record.support.get(key)
        if existing is not None:
            if existing.uid is None:
                raise RegionalFixtureError(
                    "watchdog create ACK is unknown; adoption is forbidden"
                )
            if not existing.approved or existing.removing:
                raise RegionalFixtureError(
                    "watchdog resource has deletion-only authority"
                )
            self.owned_resource(existing, expected)
            return
        if self.owner._read(kind, name) is not None:
            raise RegionalFixtureError("watchdog resource name is already occupied")
        item = ManagedResource.model_validate({"kind": kind, "name": name})
        self.owner._put_resource(item)
        self.create_ack(expected, item, index=None)

    def create_ack(
        self, expected: dict[str, Any], item: ManagedResource, *, index: int | None
    ) -> None:
        raw = self.owner.cpu(
            "create",
            "-f",
            "-",
            "-o",
            "json",
            stdin=wire.encode(expected).encode(),
            timeout=60,
        )
        actual = _document(raw)
        metadata = self.identity(actual, item)
        # Capture direct ACK custody before any readback or approval check.
        item = item.model_copy(
            update={
                "uid": metadata["uid"],
                "ack_sha256": _fingerprint(actual),
            }
        )
        self.owner._put_resource(item, index)
        self.owner._namespace()
        if item.kind == "job":
            resources.validate_job(actual, expected, uid=metadata["uid"])
        else:
            resources.validate_supporting_resource(
                actual, expected, uid=metadata["uid"]
            )
        item = item.model_copy(
            update={
                "approved": True,
                "shape_sha256": _fingerprint(
                    actual,
                    control=item.kind == "configmap" and item.name == self.owner.name,
                ),
            }
        )
        self.owner._put_resource(item, index)
        self.owned_resource(item, expected)

    def owned_resource(
        self, item: ManagedResource, expected: dict[str, Any]
    ) -> dict[str, Any]:
        actual = self.owner._read(item.kind, item.name)
        if actual is None:
            raise RegionalFixtureError("watchdog owned resource disappeared")
        self.unchanged(actual, item)
        if not item.approved or item.uid is None or item.removing:
            raise RegionalFixtureError(
                "watchdog resource is not approved for execution"
            )
        if item.kind == "job":
            resources.validate_job(actual, expected, uid=item.uid)
        elif item.kind == "configmap" and item.name == self.owner.name:
            self.owner._control_snapshot()
        else:
            resources.validate_supporting_resource(actual, expected, uid=item.uid)
        return actual

    def manifests(self) -> list[dict[str, Any]]:
        return resources.supporting_manifests(
            self.owner.plan,
            self.owner.runtime,
            name=self.owner.name,
            control_uid=self.owner._control().uid,
            source_data=self.owner.record.source_data,
        )

    def support(self) -> None:
        control = self.owner.record.support.get("configmap/" + self.owner.name)
        if control is None:
            raise RegionalFixtureError(
                "watchdog control creation has not been acknowledged"
            )
        self.owner._control_snapshot()
        for expected in self.manifests()[:-1]:
            key = expected["kind"].lower() + "/" + expected["metadata"]["name"]
            item = self.owner.record.support.get(key)
            if item is None:
                raise RegionalFixtureError("watchdog supporting resource is missing")
            self.owned_resource(item, expected)

    def job_manifest(
        self, *, name: str, cleanup_id: str | None, seconds: int
    ) -> dict[str, Any]:
        expected = self.manifests()[-1]
        expected["metadata"]["name"] = name
        if cleanup_id is not None:
            expected["spec"]["activeDeadlineSeconds"] = ARM_SECONDS + seconds + 60
            expected["spec"]["template"]["spec"]["containers"][0]["command"].extend(
                [
                    "--cleanup-only",
                    "--cleanup-seconds",
                    str(seconds),
                    "--cleanup-attempt-id",
                    cleanup_id,
                ]
            )
        return expected

    def pods(self, job: ObserverJob) -> list[dict[str, Any]]:
        # kubectl v1.35 prints a client-side v1/List with no resourceVersion, and
        # the namespace-wide read outgrew the bounded document on the live site
        # (17 Pods = 462 KiB, DESTR-008 attempt 11): read the server's own typed
        # list, narrowed to the Pods the Job controller labels with this Job.
        path = resources.raw_list_path("pod", "v1", self.owner.runtime.namespace)
        selector = "batch.kubernetes.io/controller-uid=" + self.owner._uid(job.resource)
        value = _document(
            self.owner.cpu(
                "get",
                "--raw",
                path + "?labelSelector=" + urllib.parse.quote(selector, safe=""),
                timeout=30,
            )
        )
        meta = _object(value.get("metadata"))
        items = value.get("items")
        if (
            value.get("apiVersion") != "v1"
            or value.get("kind") != "PodList"
            or not isinstance(items, list)
            or meta.get("continue", "") != ""
            or not isinstance(meta.get("resourceVersion"), str)
            or not meta["resourceVersion"]
        ):
            raise RegionalFixtureError("watchdog Pod discovery is incomplete")
        selected = []
        for raw in items:
            item = _object(raw)
            # The typed list carries bare items; the envelope proved their kind.
            item.setdefault("apiVersion", "v1")
            item.setdefault("kind", "Pod")
            metadata = _object(item.get("metadata"))
            owners = metadata.get("ownerReferences", [])
            labels = _object(metadata.get("labels", {}))
            if (
                item.get("kind") != "Pod"
                or item.get("apiVersion") != "v1"
                or metadata.get("namespace") != self.owner.runtime.namespace
                or not isinstance(owners, list)
                or any(not isinstance(owner, dict) for owner in owners)
            ):
                raise RegionalFixtureError(
                    "watchdog Pod discovery identity is ambiguous"
                )
            candidate = (
                any(
                    owner.get("uid") == job.resource.uid
                    or (
                        owner.get("kind") == "Job"
                        and owner.get("name") == job.resource.name
                    )
                    for owner in owners
                )
                or any(
                    labels.get(key) == job.resource.uid
                    for key in ("controller-uid", "batch.kubernetes.io/controller-uid")
                )
                or any(
                    labels.get(key) == job.resource.name
                    for key in ("job-name", "batch.kubernetes.io/job-name")
                )
                or (
                    job.pod is not None
                    and (
                        metadata.get("uid") == job.pod.uid
                        or metadata.get("name") == job.pod.name
                    )
                )
            )
            if candidate:
                selected.append(item)
        if len(selected) > 1:
            raise RegionalFixtureError("watchdog requires exactly one owned Pod")
        return selected

    def pod_owner(self, actual: dict[str, Any], job: ObserverJob) -> dict[str, Any]:
        meta = _object(actual.get("metadata"))
        expected = [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": job.resource.name,
                "uid": job.resource.uid,
                "controller": True,
                "blockOwnerDeletion": True,
            }
        ]
        if (
            actual.get("apiVersion") != "v1"
            or actual.get("kind") != "Pod"
            or meta.get("namespace") != self.owner.runtime.namespace
            or not isinstance(meta.get("name"), str)
            or not meta["name"]
            or not isinstance(meta.get("uid"), str)
            or not meta["uid"]
            or not isinstance(meta.get("resourceVersion"), str)
            or not meta["resourceVersion"]
            or wire.digest(meta.get("ownerReferences")) != wire.digest(expected)
            or (
                job.pod is not None
                and (
                    meta["name"] != job.pod.name
                    or meta["uid"] != job.pod.uid
                    or job.pod.gone
                )
            )
        ):
            raise RegionalFixtureError("watchdog Pod owner or recorded UID differs")
        return meta

    def discover_pod(self, index: int, *, execution: bool) -> dict[str, Any] | None:
        job = self.owner.record.jobs[index]
        found = self.pods(job)
        if not found:
            if execution and job.pod is not None:
                raise RegionalFixtureError("watchdog recorded Pod disappeared")
            return None
        actual = found[0]
        meta = self.pod_owner(actual, job)
        if job.pod is None:
            captured = ManagedPod(
                name=meta["name"], uid=meta["uid"], ack_sha256=_fingerprint(actual)
            )
            job = self.owner._put_job(index, pod=captured)
        pod = job.pod
        if pod is None:
            raise RegionalFixtureError("watchdog Pod identity was not recorded")
        if _fingerprint(actual, pod=pod.shape_sha256 is not None) != (
            pod.shape_sha256 or pod.ack_sha256
        ):
            raise RegionalFixtureError(
                "watchdog Pod changed after its recorded admission"
            )
        if (
            pod.node_name is not None
            and _object(actual.get("spec")).get("nodeName") != pod.node_name
        ):
            raise RegionalFixtureError("watchdog Pod node binding changed")
        if execution and pod.shape_sha256 is None:
            resources.validate_pod(
                actual,
                job.expected,
                job_uid=self.owner._uid(job.resource),
                pod_name=pod.name,
                pod_uid=pod.uid,
                phase="gated",
            )
            self.owner._put_job(
                index,
                pod=pod.model_copy(
                    update={"shape_sha256": _fingerprint(actual, pod=True)}
                ),
            )
        return actual

    def job_read(self, index: int) -> dict[str, Any]:
        job = self.owner.record.jobs[index]
        if (
            not job.resource.approved
            or job.resource.removing
            or job.sources != self.owner.sources
            or job.expected
            != self.job_manifest(
                name=job.resource.name,
                cleanup_id=job.cleanup_id,
                seconds=job.cleanup_seconds,
            )
        ):
            raise RegionalFixtureError(
                "watchdog Job is not an approved current observer"
            )
        actual = self.owner._read("job", job.resource.name)
        if (
            actual is None
            or _object(actual.get("metadata")).get("deletionTimestamp") is not None
        ):
            raise RegionalFixtureError(
                "watchdog current Job disappeared or is deleting"
            )
        self.unchanged(actual, job.resource)
        resources.validate_job_for_cleanup(
            actual, job.expected, uid=self.owner._uid(job.resource)
        )
        self.job_progress(actual, job)
        return actual

    def job_progress(self, actual: dict[str, Any], job: ObserverJob) -> None:
        status = _object(actual.get("status", {}))
        counts = {
            key: status.get(key, 0)
            for key in ("active", "ready", "succeeded", "failed", "terminating")
        }
        raw_conditions = status.get("conditions", [])
        if not isinstance(raw_conditions, list):
            raise RegionalFixtureError("watchdog Job progress is malformed")
        conditions: dict[str, str] = {}
        for item in raw_conditions:
            item = _object(item)
            kind, value = item.get("type"), item.get("status")
            if (
                not isinstance(kind, str)
                or kind
                not in {
                    "Complete",
                    "SuccessCriteriaMet",
                    "Failed",
                    "FailureTarget",
                    "Suspended",
                }
                or not isinstance(value, str)
                or value not in {"True", "False"}
                or kind in conditions
            ):
                raise RegionalFixtureError(
                    "watchdog Job progress conditions are unknown"
                )
            conditions[kind] = value
        uncounted = _object(status.get("uncountedTerminatedPods", {}))
        pending = uncounted.get("succeeded", [])
        if (
            any(
                type(value) is not int or not 0 <= value <= 1
                for value in counts.values()
            )
            or counts["failed"]
            or counts["terminating"]
            or counts["ready"] > counts["active"]
            or counts["active"] + counts["succeeded"] > 1
            or any(
                conditions.get(key) == "True"
                for key in ("Failed", "FailureTarget", "Suspended")
            )
            or set(uncounted) - {"succeeded", "failed"}
            or uncounted.get("failed", []) != []
            or not isinstance(pending, list)
            or (pending and (job.pod is None or pending != [job.pod.uid]))
            or (pending and (counts["active"] or counts["succeeded"]))
            or (
                conditions.get("Complete") == "True"
                and (counts["succeeded"] != 1 or pending)
            )
        ):
            raise RegionalFixtureError(
                "watchdog Job failed or has inconsistent progress counts"
            )

    def release_gate(self, index: int) -> None:
        deadline = self.owner.monotonic() + ARM_SECONDS
        while self.owner.monotonic() < deadline:
            job_value = self.job_read(index)
            pod_value = self.discover_pod(index, execution=True)
            if (
                pod_value is None
                or _object(job_value.get("status", {})).get("active", 0) == 0
            ):
                self.owner.sleep(1)
                continue
            job = self.owner.record.jobs[index]
            pod = job.pod
            if pod is None:
                raise RegionalFixtureError("watchdog gate release has no Pod identity")
            self.support()
            self.owner._live_runtime(execution=job.cleanup_id is None)
            if not job.release_requested:
                resources.validate_job(
                    job_value,
                    job.expected,
                    uid=self.owner._uid(job.resource),
                    phase="gated",
                )
                resources.validate_pod(
                    pod_value,
                    job.expected,
                    job_uid=self.owner._uid(job.resource),
                    pod_name=pod.name,
                    pod_uid=pod.uid,
                    phase="gated",
                )
                job = self.owner._put_job(index, release_requested=True)
            observed_spec = _object(pod_value.get("spec"))
            if observed_spec.get("schedulingGates"):
                metadata = pod_value["metadata"]
                patch = [
                    {"op": "test", "path": "/metadata/uid", "value": pod.uid},
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": metadata["resourceVersion"],
                    },
                    {"op": "test", "path": "/spec", "value": observed_spec},
                    {"op": "remove", "path": "/spec/schedulingGates"},
                ]
                try:
                    self.owner.cpu(
                        "patch",
                        "pod",
                        pod.name,
                        "--type=json",
                        "--patch-file=/dev/stdin",
                        "-o",
                        "json",
                        stdin=wire.encode(patch).encode(),
                        timeout=30,
                    )
                except RegionalFixtureError:
                    # The UID and complete gated shape predate this known-identity CAS.
                    pass
            current = self.discover_pod(index, execution=True)
            if current is None or _object(current.get("spec")).get("schedulingGates"):
                raise RegionalFixtureError(
                    "watchdog scheduling gate release is unconfirmed"
                )
            resources.validate_pod_for_cleanup(
                current,
                job.expected,
                job_uid=self.owner._uid(job.resource),
                pod_name=pod.name,
                pod_uid=pod.uid,
                release_requested=True,
            )
            self.owner._put_job(index, release_confirmed=True)
            return
        raise RegionalFixtureError("watchdog gated Pod deadline expired")

    def running(self, index: int) -> None:
        job_value = self.job_read(index)
        pod_value = self.discover_pod(index, execution=True)
        job = self.owner.record.jobs[index]
        pod = job.pod
        if not job.release_confirmed or pod_value is None or pod is None:
            raise RegionalFixtureError("watchdog has no verified released Pod")
        resources.validate_job(
            job_value, job.expected, uid=self.owner._uid(job.resource), phase="running"
        )
        resources.validate_pod(
            pod_value,
            job.expected,
            job_uid=self.owner._uid(job.resource),
            pod_name=pod.name,
            pod_uid=pod.uid,
            phase="running",
        )
        node = pod_value["spec"]["nodeName"]
        if pod.node_name is None:
            self.owner._put_job(index, pod=pod.model_copy(update={"node_name": node}))

    def preflight_names(self) -> None:
        planned = [
            ("configmap", self.owner.name),
            ("configmap", self.owner.name + "-code"),
            ("serviceaccount", self.owner.name),
            ("role", self.owner.name),
            ("rolebinding", self.owner.name),
            ("job", self.owner.name),
        ]
        known = set(self.owner.record.support)
        known.update("job/" + job.resource.name for job in self.owner.record.jobs)
        for kind, name in planned:
            if (
                kind + "/" + name not in known
                and self.owner._read(kind, name) is not None
            ):
                raise RegionalFixtureError("watchdog resource name is already occupied")
