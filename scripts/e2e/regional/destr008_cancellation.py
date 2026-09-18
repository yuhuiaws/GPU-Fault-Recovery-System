"""UID-bound CPU watchdog lifecycle; never submits a fault or removes a GPU fence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_watchdog_admission as admission
from scripts.e2e.regional import destr008_watchdog_control as control_api
from scripts.e2e.regional import destr008_watchdog_journal as journal
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional import destr008_watchdog_retirement as retirement
from scripts.e2e.regional.destr008_watchdog_journal import (
    ARM_SECONDS as ARM_SECONDS,
    COMPATIBILITY as COMPATIBILITY,
    KINDS as KINDS,
    MAX_DOCUMENT_BYTES as MAX_DOCUMENT_BYTES,
    MAX_JOBS as MAX_JOBS,
    STOP_SECONDS as STOP_SECONDS,
    Count as Count,
    JobChanges,
    Journal as Journal,
    Kind as Kind,
    ManagedPod as ManagedPod,
    ManagedResource as ManagedResource,
    ObserverJob as ObserverJob,
    has_saved_plan as has_saved_plan,
    journal_path as _path,
    load_saved_plan as load_saved_plan,
    object_document as _object,
    parse_document as _document,
    read_journal as _load,
    watchdog_name as _name,
)
from scripts.e2e.regional.live_driver_guard import connection_identity
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)

CancellationControl = control_api.CancellationControl
CpuRuntime = resources.CpuRuntime
Plan = wire.Plan
Receipt = wire.Receipt


def _configuration(regional: RegionalLiveFixture) -> dict[str, Any]:
    settings = regional.settings
    connections = connection_identity(argparse.Namespace(**asdict(settings)), {})
    if len(connections) != 2 or any(
        set(item) != {"path", "sha256"} for item in connections.values()
    ):
        raise RegionalFixtureError("watchdog connection identity is incomplete")
    return {
        "connections": connections,
        "gpu_context": settings.gpu_context,
        "namespace": settings.namespace,
        "region": settings.region,
        "cluster_id": settings.cluster_id,
    }


def _sources() -> dict[str, str]:
    result = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in {
            "resources": resources.__file__,
            "control": control_api.__file__,
            "locking": locking.__file__,
            "protocol": wire.__file__,
        }.items()
    }
    # Preserve the persisted key set while binding every composed implementation byte.
    result["controller"] = wire.digest(
        {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {
                "entry": __file__,
                "journal": journal.__file__,
                "admission": admission.__file__,
                "retirement": retirement.__file__,
            }.items()
        }
    )
    result["probe"] = wire.source_sha256(resources.CODE_SOURCE)
    return result


class CancellationWatchdog:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        plan: Plan,
        runtime: CpuRuntime,
        directory: Path,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.regional, self.plan, self.runtime = regional, plan, runtime
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.name = _name(plan.run_id)
        self.path = _path(directory, plan.run_id)
        resources.control_manifest(plan, runtime, self.name)
        if (
            plan.cluster_id != regional.settings.cluster_id
            or runtime.namespace != regional.settings.namespace
        ):
            raise RegionalFixtureError("watchdog namespace or cluster binding differs")
        self.configuration = _configuration(regional)
        self.host_sha256 = locking.host_identity()
        self.sources = _sources()
        owner = journal.LifecycleOwner(
            plan=self.plan,
            runtime=self.runtime,
            name=self.name,
            sources=self.sources,
            snapshot=lambda: self.record,
            monotonic=lambda: self.monotonic(),
            sleep=lambda seconds: self.sleep(seconds),
            request=lambda args, stdin, timeout: self.cpu(
                *args, stdin=stdin, timeout=timeout
            ),
            read=self._read,
            namespace=self._namespace,
            runtime_check=lambda execution: self._live_runtime(execution=execution),
            control=self._control,
            control_snapshot=self._control_snapshot,
            update_resource=self._put_resource,
            update_job=self._put_job,
            uid=self._uid,
        )
        self._admission = admission.WatchdogAdmission(owner)
        self._retirement = retirement.WatchdogRetirement(owner, self._admission)
        with locking.controller_ownership(self.path):
            if self.path.exists() or self.path.is_symlink():
                self.record = _load(self.path)
                self._bind()
            else:
                self.record = Journal(
                    schema_version=1,
                    compatibility=COMPATIBILITY,
                    plan=plan,
                    runtime=runtime.identity(),
                    configuration=self.configuration,
                    host_sha256=self.host_sha256,
                    sources=self.sources,
                    source_data=resources.watchdog_source(plan),
                    cleanup_only=False,
                    ever_armed=False,
                    closed=False,
                    support={},
                    jobs=[],
                    last_control=None,
                    last_receipt=None,
                    quiescence=None,
                )
                self._save()

    def _bind(self) -> None:
        allowed = {
            "configmap/" + self.name,
            "configmap/" + self.name + "-code",
            "serviceaccount/" + self.name,
            "role/" + self.name,
            "rolebinding/" + self.name,
        }
        if (
            self.record.plan != self.plan
            or self.record.runtime != self.runtime.identity()
            or self.record.configuration != self.configuration
            or self.record.host_sha256 != self.host_sha256
            or self.record.compatibility != COMPATIBILITY
            or set(self.record.sources) != set(self.sources)
            or set(self.record.support) - allowed
            or any(
                job.resource.name
                != (
                    self.name
                    if index == 0 and job.cleanup_id is None
                    else self.name + "-c" + str(index)
                )
                for index, job in enumerate(self.record.jobs)
            )
        ):
            raise RegionalFixtureError("watchdog journal identity differs")
        resources.watchdog_source(self.plan, source_data=self.record.source_data)

    def _save(self) -> None:
        if not locking.ownership_held(self.path):
            raise RegionalFixtureError(
                "watchdog journal write requires controller ownership"
            )
        data = self.record.model_dump(mode="json")
        Journal.model_validate(data)
        if (
            len(json.dumps(data, indent=2, sort_keys=True).encode()) + 1
            > MAX_DOCUMENT_BYTES
        ):
            raise RegionalFixtureError("watchdog journal exceeds its bounded size")
        write_json_atomic(self.path, data)

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with locking.controller_ownership(self.path):
            self.record = _load(self.path)
            self._bind()
            self._scope()
            self._namespace()
            yield

    def _scope(self) -> None:
        if (
            _configuration(self.regional) != self.configuration
            or locking.host_identity() != self.host_sha256
        ):
            raise RegionalFixtureError("watchdog controller connection or host changed")
        if _sources() != self.sources:
            raise RegionalFixtureError(
                "watchdog controller source changed during operation"
            )

    def _namespace(self) -> None:
        namespace = resources.read_object(
            self.regional, "namespace", self.runtime.namespace
        )
        if namespace["metadata"]["uid"] != self.runtime.namespace_uid:
            raise RegionalFixtureError("watchdog CPU namespace was replaced")

    def _live_runtime(self, *, execution: bool) -> None:
        self._namespace()
        if execution:
            if (
                resources.read_runtime(self.regional).identity()
                != self.runtime.identity()
            ):
                raise RegionalFixtureError("watchdog CPU runtime changed")
        else:
            deployment = resources.read_object(
                self.regional, "deployment", resources.DEPLOYMENT
            )
            meta = deployment["metadata"]
            containers = _object(
                _object(_object(deployment.get("spec")).get("template")).get("spec")
            ).get("containers")
            if (
                meta["uid"] != self.runtime.deployment_uid
                or wire.digest(meta.get("generation"))
                != wire.digest(self.runtime.generation)
                or not isinstance(containers, list)
                or len(containers) != 1
                or _object(containers[0]).get("name") != "control-worker"
                or containers[0].get("image") != self.runtime.image
            ):
                raise RegionalFixtureError("watchdog cleanup runtime identity changed")
            for name, identity in self.runtime.config_identities.items():
                meta = resources.read_object(self.regional, "configmap", name)[
                    "metadata"
                ]
                if (
                    meta["uid"] != identity["uid"]
                    or meta["resourceVersion"] != identity["resource_version"]
                ):
                    raise RegionalFixtureError("watchdog cleanup configuration changed")
        scope = self.regional.evidence_identity()
        if (
            scope.get("release_id") != self.plan.release_id
            or scope.get("cluster_id") != self.plan.cluster_id
        ):
            raise RegionalFixtureError("watchdog release or cluster identity changed")

    def cpu(self, *args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        with locking.controller_ownership(self.path):
            self._scope()
            current = _load(self.path)
            if args[0] in {"create", "patch", "delete"}:
                self._namespace()
            if (current.cleanup_only or self.sources != current.sources) and args[
                :3
            ] == ("patch", "configmap", self.name):
                self._cleanup_patch(stdin)
            try:
                result = self.regional.kubectl(
                    "cpu",
                    *args,
                    input_text=None if stdin is None else stdin.decode("utf-8"),
                    **kwargs,
                )
            except Exception:
                raise RegionalFixtureError(
                    "watchdog CPU Kubernetes request failed"
                ) from None
            return result

    def _cleanup_patch(self, stdin: bytes | None) -> None:
        try:
            patch = json.loads(stdin or b"", object_pairs_hook=wire.unique_object)
            changed = [
                operation["value"]
                for operation in patch
                if operation.get("op") == "replace"
                and operation.get("path") == "/data/control.json"
            ]
            tested = [
                operation["value"]
                for operation in patch
                if operation.get("op") == "test"
                and operation.get("path") == "/data/control.json"
            ]
            if len(changed) != 1 or len(tested) != 1:
                raise ValueError("control CAS")
            before = wire.decode(wire.Control, tested[0])
            after = wire.decode(wire.Control, changed[0])
            late_ack = (
                before.producer.state == "SUBMITTING"
                and after.producer.state == "ACKNOWLEDGED"
                and before.producer.claim_id == after.producer.claim_id
                and before.producer.claimed_at == after.producer.claimed_at
            )
            if (
                (after.producer != before.producer and not late_ack)
                or (
                    before.revocation is not None
                    and before.revocation != after.revocation
                )
                or (
                    before.close_request is not None
                    and before.close_request != after.close_request
                )
                or (
                    after.revocation is None
                    and after.close_request is None
                    and not late_ack
                )
            ):
                raise ValueError("submission authority")
        except (ValueError, TypeError, KeyError, AttributeError, wire.ProbeError):
            raise RegionalFixtureError(
                "cleanup cannot restore watchdog producer authority"
            ) from None

    def _read(self, kind: str, name: str) -> dict[str, Any] | None:
        raw = self.cpu(
            "get", kind, name, "--ignore-not-found", "-o", "json", timeout=30
        )
        return _document(raw) if raw.strip() else None

    def _put_resource(self, item: ManagedResource, index: int | None = None) -> None:
        if index is None:
            self.record.support[item.kind + "/" + item.name] = item
        else:
            self.record.jobs[index] = self.record.jobs[index].model_copy(
                update={"resource": item}
            )
        self._save()

    def _put_job(self, index: int, changes: JobChanges) -> ObserverJob:
        job = self.record.jobs[index].model_copy(update=changes)
        self.record.jobs[index] = job
        self._save()
        return job

    def _control(self) -> CancellationControl:
        item = self.record.support.get("configmap/" + self.name)
        if item is None or item.uid is None or not item.approved or item.removed:
            raise RegionalFixtureError("watchdog has no approved control identity")
        return CancellationControl(
            plan=self.plan,
            namespace=self.runtime.namespace,
            name=self.name,
            uid=item.uid,
            cpu=self.cpu,
            clock=self.clock,
            monotonic=self.monotonic,
        )

    def _control_snapshot(self) -> control_api.ControlSnapshot:
        item = self.record.support.get("configmap/" + self.name)
        control = self._control()
        actual = self._read("configmap", self.name)
        if actual is None or item is None:
            raise RegionalFixtureError("watchdog control resource disappeared")
        self._admission.unchanged(actual, item)
        try:
            snapshot = control.parse(wire.encode(actual))
        except (ValueError, TypeError, wire.ProbeError):
            raise RegionalFixtureError("watchdog control protocol is invalid") from None
        expected = resources.control_manifest(self.plan, self.runtime, self.name)
        expected["data"] = snapshot.data
        resources.validate_supporting_resource(actual, expected, uid=control.uid)
        previous, receipt = self.record.last_control, snapshot.receipt
        last = self.record.last_receipt
        if previous is not None and (
            (
                previous.revocation is not None
                and previous.revocation != snapshot.control.revocation
            )
            or (
                previous.close_request is not None
                and previous.close_request != snapshot.control.close_request
            )
            or (
                previous.producer.state != "NOT_STARTED"
                and (
                    snapshot.control.producer.state == "NOT_STARTED"
                    or previous.producer.claim_id != snapshot.control.producer.claim_id
                    or previous.producer.claimed_at
                    != snapshot.control.producer.claimed_at
                    or (
                        previous.producer.ack is not None
                        and previous.producer != snapshot.control.producer
                    )
                )
            )
        ):
            raise RegionalFixtureError("watchdog durable producer history regressed")
        if last is not None and (
            receipt is None
            or receipt.sequence < last.sequence
            or receipt.observed_at < last.observed_at
            or (receipt.sequence == last.sequence and receipt != last)
            or (last.failure is not None and receipt.failure != last.failure)
            or (last.root is not None and receipt.root != last.root)
            or not set(last.workflow_ids).issubset(receipt.workflow_ids)
            or not set(last.command_ids).issubset(receipt.command_ids)
        ):
            raise RegionalFixtureError("watchdog durable receipt history regressed")
        self.record = self.record.model_copy(
            update={
                "last_control": snapshot.control,
                "last_receipt": receipt,
            }
        )
        self._save()
        return snapshot

    def _new_job(self, *, cleanup_id: str | None, seconds: int, sequence: int) -> int:
        index = len(self.record.jobs)
        if index >= MAX_JOBS:
            raise RegionalFixtureError("watchdog cleanup attempt budget is exhausted")
        name = (
            self.name
            if index == 0 and cleanup_id is None
            else self.name + "-c" + str(index)
        )
        expected = self._admission.job_manifest(
            name=name, cleanup_id=cleanup_id, seconds=seconds
        )
        if self._read("job", name) is not None:
            raise RegionalFixtureError("watchdog observer Job name is already occupied")
        item = ManagedResource(kind="job", name=name)
        job = ObserverJob(
            resource=item,
            expected=expected,
            created_at=int(self.clock()),
            sources=self.sources,
            cleanup_id=cleanup_id,
            cleanup_seconds=seconds,
            sequence_floor=sequence,
        )
        self.record.jobs.append(job)
        self._save()
        self._admission.create_ack(expected, item, index=index)
        return index

    def _uid(self, item: ManagedResource) -> str:
        if item.uid is None:
            raise RegionalFixtureError(
                "watchdog create ACK is unknown; adoption is forbidden"
            )
        return item.uid

    def _execution(self) -> None:
        if (
            self.record.cleanup_only
            or self.record.closed
            or self.sources != self.record.sources
            or not self.plan.created_at <= int(self.clock()) < self.plan.deadline_at
        ):
            raise RegionalFixtureError(
                "watchdog execution authority is closed or source changed"
            )
        self._live_runtime(execution=True)
        resources.require_cpu_history_capability(self.regional, self.runtime)

    def _active_job(self) -> int:
        active = [
            index for index, job in enumerate(self.record.jobs) if not job.stopped
        ]
        if len(active) != 1:
            raise RegionalFixtureError("watchdog has no unique current observer Job")
        return active[0]

    def arm(self) -> CancellationControl:
        with self._operation():
            self._execution()
            if self.record.quiescence is not None:
                raise RegionalFixtureError("a quiescent watchdog cannot rearm")
            self._admission.preflight_names()
            self._admission.create(
                resources.control_manifest(self.plan, self.runtime, self.name)
            )
            for expected in self._admission.manifests()[:-1]:
                self._admission.create(expected)
            if not self.record.jobs:
                index = self._new_job(cleanup_id=None, seconds=0, sequence=0)
            else:
                index = self._active_job()
                if self.record.jobs[index].cleanup_id is not None:
                    raise RegionalFixtureError(
                        "a cleanup observer cannot authorize execution"
                    )
            if not self.record.jobs[index].release_confirmed:
                self._admission.release_gate(index)
            deadline = min(
                self.monotonic() + ARM_SECONDS,
                self.monotonic() + self.plan.deadline_at - int(self.clock()),
            )
            while self.monotonic() < deadline:
                job_value = self._admission.job_read(index)
                pod_value = self._admission.discover_pod(index, execution=True)
                if pod_value is not None:
                    phase = _object(pod_value.get("status")).get("phase")
                    if phase not in {"Pending", "Running"}:
                        raise RegionalFixtureError(
                            "watchdog observer ended before arming"
                        )
                    if (
                        phase == "Running"
                        and _object(job_value.get("status")).get("ready") == 1
                    ):
                        self._admission.running(index)
                        snapshot = self._control_snapshot()
                        if snapshot.receipt is not None:
                            self._control().assert_armed()
                            self.record = self.record.model_copy(
                                update={"ever_armed": True}
                            )
                            self._save()
                            return self._control()
                self.sleep(1)
            raise RegionalFixtureError("watchdog running/ARMED deadline expired")

    def validate_running(self) -> None:
        with self._operation():
            self._execution()
            if not self.record.ever_armed:
                raise RegionalFixtureError("watchdog has never completed arming")
            self._admission.support()
            self._admission.running(self._active_job())
            self._control_snapshot()
            self._control().assert_armed()

    def _receipt_job(self, receipt: Receipt, job: ObserverJob) -> None:
        cleanup = receipt.cleanup
        if (
            receipt.sequence <= job.sequence_floor
            or receipt.observed_at < job.created_at
            or (job.cleanup_id is None and cleanup is not None)
            or (
                job.cleanup_id is not None
                and (
                    cleanup is None
                    or cleanup.attempt_id != job.cleanup_id
                    or cleanup.deadline_at - cleanup.started_at != job.cleanup_seconds
                    or not receipt.case_failed
                    or not receipt.producer_revoked
                )
            )
        ):
            raise RegionalFixtureError(
                "watchdog receipt belongs to an earlier observer"
            )

    def _wait_quiescence(self, index: int, *, seconds: int) -> Receipt:
        deadline = self.monotonic() + seconds
        while self.monotonic() < deadline:
            self._admission.support()
            snapshot = self._control_snapshot()
            current = snapshot.receipt
            job_value = self._admission.job_read(index)
            pod_value = self._admission.discover_pod(index, execution=True)
            job = self.record.jobs[index]
            if current is not None and current.sequence > job.sequence_floor:
                self._receipt_job(current, job)
                if current.state == "FAILED" and not current.monitoring:
                    raise RegionalFixtureError(
                        "watchdog failed to prove terminal quiescence"
                    )
                if current.state == "QUIESCENT" and pod_value is not None:
                    phase = _object(pod_value.get("status")).get("phase")
                    if (
                        phase == "Succeeded"
                        and _object(job_value.get("status")).get("succeeded") == 1
                        and any(
                            condition.get("type") == "Complete"
                            and condition.get("status") == "True"
                            for condition in job_value["status"].get("conditions", [])
                        )
                    ):
                        pod = job.pod
                        if pod is None:
                            raise RegionalFixtureError(
                                "watchdog completion has no recorded Pod"
                            )
                        resources.validate_job(
                            job_value,
                            job.expected,
                            uid=self._uid(job.resource),
                            phase="succeeded",
                        )
                        resources.validate_pod(
                            pod_value,
                            job.expected,
                            job_uid=self._uid(job.resource),
                            pod_name=pod.name,
                            pod_uid=pod.uid,
                            phase="succeeded",
                        )
                        verified = self._control().quiescence()
                        self._receipt_job(verified, job)
                        if verified.monitoring:
                            raise RegionalFixtureError(
                                "watchdog observer has not terminated its monitoring"
                            )
                        self.record = self.record.model_copy(
                            update={"quiescence": verified, "last_receipt": verified}
                        )
                        self._save()
                        return verified
            if pod_value is not None and _object(pod_value.get("status")).get(
                "phase"
            ) not in {"Pending", "Running", "Succeeded"}:
                raise RegionalFixtureError(
                    "watchdog Pod did not terminate successfully"
                )
            self.sleep(1)
        raise RegionalFixtureError("watchdog terminal quiescence deadline expired")

    def wait_quiescence(self) -> Receipt:
        with self._operation():
            if self.record.closed:
                raise RegionalFixtureError(
                    "a retired watchdog cannot certify quiescence"
                )
            self._live_runtime(execution=False)
            snapshot = self._control_snapshot()
            if (
                snapshot.control.close_request is None
                and snapshot.control.revocation is None
            ):
                raise RegionalFixtureError(
                    "watchdog close or revocation must precede quiescence"
                )
            return self._wait_quiescence(
                self._active_job(), seconds=wire.DRAIN_SECONDS + 30
            )

    def _mark_cleanup(self) -> None:
        self.record = self.record.model_copy(update={"cleanup_only": True})
        self._save()

    def _close_unclaimed(self) -> None:
        before = self._control_snapshot()
        if (
            before.control.revocation is not None
            or before.control.close_request is not None
            or before.control.producer.state != "NOT_STARTED"
        ):
            return
        self._control().request_close()
        after = self._control_snapshot()
        if (
            after.control.close_request is None
            or after.control.producer != before.control.producer
        ):
            raise RegionalFixtureError(
                "watchdog unclaimed producer closure is unconfirmed"
            )

    def resume_cleanup(self, seconds: int = wire.MAX_CLEANUP_SECONDS) -> Receipt:
        if (
            type(seconds) is not int
            or not wire.QUIET_SECONDS <= seconds <= wire.MAX_CLEANUP_SECONDS
        ):
            raise RegionalFixtureError("watchdog cleanup window is outside its bound")
        with self._operation():
            if self.record.closed:
                raise RegionalFixtureError(
                    "a retired watchdog cannot start an observer"
                )
            self._mark_cleanup()
            self.record = self.record.model_copy(update={"quiescence": None})
            self._save()
            self._live_runtime(execution=False)
            self._retirement.stop_all_jobs()
            self._close_unclaimed()
            self._admission.support()
            snapshot = self._control_snapshot()
            sequence = 0 if snapshot.receipt is None else snapshot.receipt.sequence
            index = self._new_job(
                cleanup_id="cleanup-" + uuid4().hex,
                seconds=seconds,
                sequence=sequence,
            )
            self._admission.release_gate(index)
            return self._wait_quiescence(index, seconds=ARM_SECONDS + seconds + 30)

    def control_client(self) -> CancellationControl:
        """Validate and return the existing bound control without arming or claiming."""
        with self._operation():
            self._control_snapshot()
            return self._control()

    def cleanup(self) -> None:
        with self._operation():
            self._mark_cleanup()
            self._live_runtime(execution=False)
            # Unknown create ACKs cannot be converted into absence/adoption by a later GET.
            for item in self.record.support.values():
                self._uid(item)
            self._retirement.stop_all_jobs()
            control_item = self.record.support.get("configmap/" + self.name)
            if (
                control_item is not None
                and control_item.approved
                and not control_item.removed
            ):
                self._close_unclaimed()
                snapshot = self._control_snapshot()
                if self.record.ever_armed:
                    if (
                        self.record.quiescence is None
                        or snapshot.receipt != self.record.quiescence
                    ):
                        raise RegionalFixtureError(
                            "watchdog retirement requires verified terminal quiescence"
                        )
                elif snapshot.control.producer.state != "NOT_STARTED":
                    raise RegionalFixtureError(
                        "watchdog has unresolved producer authority"
                    )
            for key in reversed(list(self.record.support)):
                item = self.record.support[key]
                if item.kind == "configmap" and item.name == self.name:
                    continue
                self._retirement.remove_resource(item)
            if control_item is not None:
                self._retirement.remove_resource(control_item)
            self.record = self.record.model_copy(update={"closed": True})
            self._save()
