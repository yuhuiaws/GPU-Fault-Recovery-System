"""Parent-side CAS and fresh, complete receipts for the CPU cancellation watchdog."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from gpu_fault.host_health import NodeHealthIngestionResult
from scripts.e2e.regional.probes.destr008_cancellation_protocol import (
    MAX_WINDOW_SECONDS,
    Acknowledgement,
    Control,
    Plan,
    Receipt,
    acknowledge_submission,
    claim_submission,
    decode,
    digest,
    encode,
    request_close,
    source_sha256,
    unique_object,
    validate_control,
    validate_receipt,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

MAX_RECEIPT_AGE = 30
MAX_CONTROL_BYTES = 262144


@dataclass(frozen=True)
class ControlSnapshot:
    uid: str
    version: str
    data: dict[str, str]
    control: Control
    receipt: Receipt | None


class CancellationControl:
    def __init__(
        self,
        *,
        plan: Plan,
        namespace: str,
        name: str,
        uid: str,
        cpu: Callable[..., str],
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if any(
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", value) is None
            for value in (namespace, name, uid)
        ):
            raise RegionalFixtureError("watchdog control identity is invalid")
        try:
            self.plan = decode(Plan, encode(plan))
        except Exception:
            raise RegionalFixtureError("watchdog control plan is invalid") from None
        if self.plan.probe_sha256 != source_sha256():
            raise RegionalFixtureError("watchdog control source identity differs")
        self.namespace = namespace
        self.name = name
        self.uid = uid
        self.cpu = cpu
        self.clock = clock
        self.monotonic = monotonic
        self._binding = (namespace, name, uid, encode(self.plan))
        self._last: ControlSnapshot | None = None

    def _bound(self) -> None:
        try:
            same = (
                self.namespace,
                self.name,
                self.uid,
                encode(self.plan),
            ) == self._binding
        except Exception:
            same = False
        if not same:
            raise RegionalFixtureError("watchdog control binding changed")

    def _now(self) -> int:
        value = self.clock()
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not 0 < value <= 253_402_300_799
        ):
            raise RegionalFixtureError("watchdog control clock is invalid")
        return int(value)

    def identity(self) -> dict[str, Any]:
        self._bound()
        return {
            "namespace": self.namespace,
            "name": self.name,
            "uid": self.uid,
            "plan_sha256": digest(self.plan),
            "probe_sha256": self.plan.probe_sha256,
            "plan": self.plan.model_dump(mode="json"),
        }

    def parse(self, raw: str) -> ControlSnapshot:
        self._bound()
        try:
            return self._parse(raw)
        except RegionalFixtureError:
            raise
        except Exception:
            raise RegionalFixtureError(
                "watchdog control schema validation failed"
            ) from None

    def _parse(self, raw: str) -> ControlSnapshot:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_CONTROL_BYTES:
            raise RegionalFixtureError(
                "watchdog control response exceeds its size limit"
            )
        value = json.loads(raw, object_pairs_hook=unique_object)
        encode(value)  # Reject non-finite JSON even in Kubernetes metadata.
        if not isinstance(value, dict):
            raise RegionalFixtureError("watchdog control response is not an object")
        meta = value.get("metadata")
        data = value.get("data")
        if (
            value.get("apiVersion") != "v1"
            or value.get("kind") != "ConfigMap"
            or not isinstance(meta, dict)
            or meta.get("name") != self.name
            or meta.get("namespace") != self.namespace
            or meta.get("uid") != self.uid
            or not isinstance(meta.get("resourceVersion"), str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", meta["resourceVersion"]) is None
            or meta.get("deletionTimestamp") is not None
            or value.get("binaryData") not in (None, {})
            or (value.get("immutable") is not None and value["immutable"] is not False)
            or not isinstance(data, dict)
            or set(data) != {"plan.json", "control.json", "status.json"}
        ):
            raise RegionalFixtureError("watchdog control identity or shape changed")
        observed = decode(Plan, data["plan.json"])
        if observed != self.plan:
            raise RegionalFixtureError("watchdog plan differs from the authorized plan")
        control = decode(Control, data["control.json"])
        now = self._now()
        validate_control(self.plan, control, now)
        receipt = (
            None
            if data["status.json"] == "null"
            else decode(Receipt, data["status.json"])
        )
        if receipt is not None:
            # A late POST acknowledgement can resolve a stopped observer, not restart it.
            validate_receipt(
                self.plan,
                control,
                receipt,
                uid=self.uid,
                now=now,
                cleanup_only=True,
            )
        return ControlSnapshot(
            self.uid, meta["resourceVersion"], dict(data), control, receipt
        )

    def _accept(self, after: ControlSnapshot) -> ControlSnapshot:
        before = self._last
        if before is not None:
            old, new = before.control, after.control
            if (
                (before.version == after.version and before.data != after.data)
                or before.data["plan.json"] != after.data["plan.json"]
                or (old.revocation is not None and new.revocation != old.revocation)
                or (
                    old.close_request is not None
                    and new.close_request != old.close_request
                )
                or (
                    old.producer.state != "NOT_STARTED"
                    and (
                        new.producer.state == "NOT_STARTED"
                        or old.producer.claim_id != new.producer.claim_id
                        or old.producer.claimed_at != new.producer.claimed_at
                        or (
                            old.producer.ack is not None
                            and old.producer != new.producer
                        )
                    )
                )
            ):
                raise RegionalFixtureError(
                    "watchdog control state regressed or changed"
                )
            previous, current = before.receipt, after.receipt
            if previous is not None and (
                current is None
                or current.sequence < previous.sequence
                or (current.sequence == previous.sequence and current != previous)
                or (
                    previous.failure is not None and current.failure != previous.failure
                )
                or (previous.root is not None and current.root != previous.root)
                or not set(previous.workflow_ids).issubset(current.workflow_ids)
                or not set(previous.command_ids).issubset(current.command_ids)
            ):
                raise RegionalFixtureError(
                    "watchdog receipt state regressed or changed"
                )
        self._last = after
        return after

    def read(self, *, timeout: float = 30) -> ControlSnapshot:
        self._bound()
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(timeout)
            or not 0 < timeout <= 30
        ):
            raise RegionalFixtureError("watchdog control timeout is invalid")
        try:
            raw = self.cpu(
                "get",
                "configmap",
                self.name,
                "--namespace",
                self.namespace,
                "-o",
                "json",
                timeout=timeout,
            )
        except Exception:
            raise RegionalFixtureError("watchdog control read failed") from None
        return self._accept(self.parse(raw))

    def _parent_change(self, before: ControlSnapshot, changed: Control) -> None:
        old = before.control
        if changed == old:
            return
        if changed.revocation != old.revocation:
            raise RegionalFixtureError("parent cannot modify watchdog revocation")
        if old.producer != changed.producer:
            if (
                old.close_request != changed.close_request
                or old.close_request is not None
                or (old.producer.state, changed.producer.state)
                not in {("NOT_STARTED", "SUBMITTING"), ("SUBMITTING", "ACKNOWLEDGED")}
                or (
                    old.producer.state == "SUBMITTING"
                    and (
                        old.producer.claim_id != changed.producer.claim_id
                        or old.producer.claimed_at != changed.producer.claimed_at
                    )
                )
            ):
                raise RegionalFixtureError("watchdog producer transition is invalid")
            if old.producer.state == "NOT_STARTED":
                self._armed(before)
        elif old.close_request is not None or changed.close_request is None:
            raise RegionalFixtureError("watchdog close request is immutable")

    def change(self, transform: Callable[[Control, int], Control]) -> ControlSnapshot:
        for attempt in range(4):
            before = self.read()
            try:
                candidate = transform(before.control, self._now())
                if (
                    not isinstance(candidate, Control)
                    or candidate.model_fields_set - Control.model_fields.keys()
                ):
                    raise ValueError("unsupported control fields")
                changed = decode(Control, encode(candidate))
                validate_control(self.plan, changed, self._now())
            except Exception:
                raise RegionalFixtureError(
                    "watchdog control transition was refused"
                ) from None
            self._parent_change(before, changed)
            if changed == before.control:
                return before
            # Pin only what the parent owns and reads: our UID, the immutable
            # plan, and the exact control.json we are replacing. The daemon
            # rewrites status.json every POLL_SECONDS and bumps the ConfigMap
            # resourceVersion, so pinning the whole version or every /data key
            # (status.json included) could never converge against a live
            # heartbeat. control.json stays parent-owned except for the daemon's
            # own ACK, which this test guards against a lost update.
            patch = [
                {"op": "test", "path": "/metadata/uid", "value": self.uid},
                {
                    "op": "test",
                    "path": "/data/plan.json",
                    "value": before.data["plan.json"],
                },
                {
                    "op": "test",
                    "path": "/data/control.json",
                    "value": before.data["control.json"],
                },
                {
                    "op": "replace",
                    "path": "/data/control.json",
                    "value": encode(changed),
                },
            ]
            try:
                response = self.cpu(
                    "patch",
                    "configmap",
                    self.name,
                    "--type=json",
                    "--namespace",
                    self.namespace,
                    "--patch-file=/dev/stdin",
                    "-o",
                    "json",
                    stdin=encode(patch).encode(),
                    timeout=30,
                )
            except Exception:
                after = self.read()
                if after.control != changed:
                    if attempt < 3:
                        continue
                    raise RegionalFixtureError(
                        "watchdog control retries were exhausted"
                    ) from None
            else:
                after = self.parse(response)
            if (
                after.version != before.version
                and after.control == changed
                and after.data["plan.json"] == before.data["plan.json"]
            ):
                return self._accept(after)
            raise RegionalFixtureError(
                "watchdog control transition was not acknowledged"
            )
        raise RegionalFixtureError("watchdog control retries were exhausted")

    def _armed(
        self, snapshot: ControlSnapshot, *, claim_id: str | None = None
    ) -> Receipt:
        current = snapshot.receipt
        now = self._now()
        if (
            current is None
            or current.state != "ARMED"
            or not current.monitoring
            or current.case_failed
            or current.cleanup is not None
            or snapshot.control.revocation is not None
            or snapshot.control.close_request is not None
            or now >= self.plan.deadline_at
            or now - current.observed_at > MAX_RECEIPT_AGE
            or snapshot.control.producer.state
            != ("NOT_STARTED" if claim_id is None else "SUBMITTING")
            or snapshot.control.producer.claim_id != claim_id
        ):
            raise RegionalFixtureError(
                "independent cancellation watchdog is not freshly armed"
            )
        return current

    def assert_armed(self) -> Receipt:
        return self._armed(self.read())

    def claim(self) -> str:
        self.assert_armed()
        claim_id = "claim-" + uuid4().hex
        self.change(
            lambda control, now: claim_submission(
                self.plan, control, claim_id=claim_id, now=now
            )
        )
        snapshot = self.read()
        self._armed(snapshot, claim_id=claim_id)
        return claim_id

    def acknowledge(self, claim_id: str, response: dict[str, Any]) -> None:
        body = response.get("body") if isinstance(response, dict) else None
        if (
            not isinstance(response, dict)
            or type(response.get("status")) is not int
            or response["status"] != 200
            or not isinstance(body, dict)
            or body.get("batch_id") != self.plan.event_id
            or body.get("duplicate") is not False
            or not isinstance(body.get("incident_ids"), list)
            or len(body["incident_ids"]) != 1
            or not isinstance(body.get("workflow_request_ids"), list)
            or len(body["workflow_request_ids"]) != 1
        ):
            raise RegionalFixtureError(
                "replacement producer has no complete source acknowledgement"
            )
        try:
            if len(encode(body)) > MAX_CONTROL_BYTES:
                raise ValueError("oversize acknowledgement")
            result = NodeHealthIngestionResult.model_validate_json(
                encode(body), strict=True
            )
            if any(
                finding.event_id != self.plan.event_id
                or finding.cluster_id != self.plan.cluster_id
                or finding.job_id != self.plan.job_id
                or finding.attempt_id != self.plan.attempt_id
                or finding.node_id != self.plan.fault_node
                or finding.runtime_profile_version != self.plan.runtime_profile_version
                or set(finding.affected_workload_ids) != set(self.plan.workload_ids)
                for finding in result.findings
            ):
                raise ValueError("source finding binding changed")
            acknowledgement = Acknowledgement(
                claim_id=claim_id,
                event_id=self.plan.event_id,
                incident_id=body["incident_ids"][0],
                workflow_request_id=body["workflow_request_ids"][0],
                completed_at=self._now(),
            )
        except Exception:
            raise RegionalFixtureError(
                "replacement producer acknowledgement is invalid"
            ) from None
        previous = self.read().control.producer.ack
        if previous is not None:
            if (
                previous.model_copy(
                    update={"completed_at": acknowledgement.completed_at}
                )
                != acknowledgement
            ):
                raise RegionalFixtureError(
                    "replacement producer acknowledgement changed"
                )
            return
        self.change(
            lambda control, now: acknowledge_submission(
                self.plan, control, acknowledgement=acknowledgement, now=now
            )
        )

    def request_close(self) -> None:
        snapshot = self.read()
        if snapshot.control.close_request is not None:
            return
        self.change(lambda control, now: request_close(self.plan, control, now=now))

    def quiescence(self, *, timeout: float = 30) -> Receipt:
        try:
            snapshot = self.read(timeout=timeout)
            current = snapshot.receipt
            if (
                current is None
                or current.state != "QUIESCENT"
                or snapshot.control.revocation is None
                or current.producer != snapshot.control.producer
                or self._now() - current.observed_at > MAX_RECEIPT_AGE
            ):
                raise ValueError("quiescence not proven")
            return current
        except Exception:
            raise RegionalFixtureError(
                "fresh complete watchdog quiescence is unavailable"
            ) from None

    def wait_quiescence(
        self, *, seconds: int = 180, sleep: Callable[[float], None] = time.sleep
    ) -> Receipt:
        if type(seconds) is not int or not 1 <= seconds <= MAX_WINDOW_SECONDS:
            raise RegionalFixtureError("watchdog wait duration is invalid")
        started = self.monotonic()
        if not math.isfinite(started):
            raise RegionalFixtureError("watchdog wait clock is invalid")
        deadline = started + seconds
        while True:
            now = self.monotonic()
            if not math.isfinite(now) or now < started:
                raise RegionalFixtureError("watchdog wait clock is invalid")
            if now >= deadline:
                break
            snapshot = self.read(timeout=min(30, deadline - now))
            if self.monotonic() >= deadline:
                break
            if snapshot.receipt is not None:
                if snapshot.receipt.state == "QUIESCENT":
                    remaining = deadline - self.monotonic()
                    if remaining <= 0:
                        break
                    result = self.quiescence(timeout=min(30, remaining))
                    if self.monotonic() >= deadline:
                        break
                    return result
                if (
                    snapshot.receipt.state == "FAILED"
                    and not snapshot.receipt.monitoring
                ):
                    raise RegionalFixtureError(
                        "watchdog could not prove terminal quiescence"
                    )
            sleep(min(1, max(0, deadline - self.monotonic())))
        raise RegionalFixtureError("watchdog quiescence deadline expired")
