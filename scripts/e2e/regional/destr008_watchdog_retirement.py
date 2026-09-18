"""UID/RV retirement and confirmed cessation of owned CPU observers."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.destr008_watchdog_admission import WatchdogAdmission
from scripts.e2e.regional.destr008_watchdog_journal import (
    KINDS,
    STOP_SECONDS,
    LifecycleOwner,
    ManagedResource,
    object_document as _object,
)
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.seeded_command_fixture import RUN_LABEL, delete_owned_resource


class WatchdogRetirement:
    def __init__(self, owner: LifecycleOwner, admission: WatchdogAdmission) -> None:
        self.owner = owner
        self.admission = admission

    def delete_only(self, item: ManagedResource, actual: dict[str, Any]) -> None:
        _, _, api, plural = KINDS[item.kind]
        meta = self.admission.identity(actual, item)
        path = f"/{api}/namespaces/{self.owner.runtime.namespace}/{plural}/{item.name}"
        options = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {
                "uid": item.uid,
                "resourceVersion": meta["resourceVersion"],
            },
            "propagationPolicy": "Foreground",
        }
        try:
            self.owner.cpu(
                "delete",
                "--raw",
                path,
                "-f",
                "-",
                stdin=wire.encode(options).encode(),
                timeout=120,
            )
        except RegionalFixtureError:
            pass
        deadline = self.owner.monotonic() + STOP_SECONDS
        while self.owner.monotonic() < deadline:
            remaining = self.owner._read(item.kind, item.name)
            if remaining is None:
                return
            self.admission.unchanged(remaining, item)
            if not _object(remaining.get("metadata")).get("deletionTimestamp"):
                raise RegionalFixtureError(
                    "watchdog deletion-only request was not acknowledged"
                )
            self.owner.sleep(1)
        raise RegionalFixtureError("watchdog deletion-only resource did not disappear")

    def remove_resource(
        self, item: ManagedResource, *, index: int | None = None
    ) -> None:
        self.owner._uid(item)
        actual = self.owner._read(item.kind, item.name)
        if item.removed:
            if actual is not None:
                raise RegionalFixtureError(
                    "a retired watchdog resource name was recreated"
                )
            return
        if actual is not None:
            self.admission.unchanged(actual, item)
            if index is not None and item.approved:
                resources.validate_job_for_cleanup(
                    actual,
                    self.owner.record.jobs[index].expected,
                    uid=self.owner._uid(item),
                )
        item = item.model_copy(update={"removing": True})
        self.owner._put_resource(item, index)
        if actual is not None:
            labels = _object(_object(actual.get("metadata")).get("labels", {}))
            if labels.get(RUN_LABEL) == self.owner.plan.run_id:
                delete_owned_resource(
                    item.kind,
                    item.name,
                    self.owner.plan.run_id,
                    client=self.owner.cpu,
                    namespace=self.owner.runtime.namespace,
                    expected_uid=self.owner._uid(item),
                    expected_resource_version=actual["metadata"]["resourceVersion"],
                    require_uid=True,
                )
            else:
                # A direct ACK can authorize removing an unapproved object without its labels.
                self.delete_only(item, actual)
            if self.owner._read(item.kind, item.name) is not None:
                raise RegionalFixtureError(
                    "watchdog resource deletion was not confirmed"
                )
        self.owner._put_resource(item.model_copy(update={"removed": True}), index)

    def stop_job(self, index: int) -> None:
        job = self.owner.record.jobs[index]
        self.owner._uid(job.resource)
        pod_value = self.admission.discover_pod(index, execution=False)
        job = self.owner.record.jobs[index]
        pod = job.pod
        if pod_value is not None and pod is not None and pod.shape_sha256 is not None:
            resources.validate_pod_for_cleanup(
                pod_value,
                job.expected,
                job_uid=self.owner._uid(job.resource),
                pod_name=pod.name,
                pod_uid=pod.uid,
                release_requested=job.release_requested,
            )
        self.remove_resource(job.resource, index=index)
        deadline = self.owner.monotonic() + STOP_SECONDS
        while self.owner.monotonic() < deadline:
            current = self.admission.discover_pod(index, execution=False)
            job = self.owner.record.jobs[index]
            pod = job.pod
            if current is None:
                if pod is not None and not pod.gone:
                    job = self.owner._put_job(
                        index, pod=pod.model_copy(update={"gone": True})
                    )
                self.owner._put_job(index, stopped=True)
                return
            if pod is None:
                raise RegionalFixtureError(
                    "watchdog orphan Pod has no recorded identity"
                )
            delete_owned_resource(
                "pod",
                pod.name,
                self.owner.plan.run_id,
                client=self.owner.cpu,
                namespace=self.owner.runtime.namespace,
                expected_uid=pod.uid,
                expected_resource_version=current["metadata"]["resourceVersion"],
                require_uid=True,
            )
            self.owner.sleep(1)
        raise RegionalFixtureError("watchdog Pod cessation was not confirmed")

    def stop_all_jobs(self) -> None:
        for index in range(len(self.owner.record.jobs)):
            self.stop_job(index)
        if any(not job.stopped for job in self.owner.record.jobs):
            raise RegionalFixtureError("an earlier watchdog observer is still present")
