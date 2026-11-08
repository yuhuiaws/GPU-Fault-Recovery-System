"""A fake Kubernetes Node for the S74 cordon-lift tests: one Node with a
resourceVersion counter, JSON-patch semantics (``test`` / ``replace`` / ``add``),
and a watch that delivers queued events, then runs the next scripted product
write (each with the time it takes), then closes the chunk. The product writes
follow the code the lift is built against (``general_live_cordon_lift``):
MARK_UNSCHEDULABLE stamps the successor's incident on the cordoned node; its
QUARANTINE re-writes the cordon, visible only when the cordon was lifted; the
support-after hand-off takes the node over after a refused reboot; a submitted
reboot's ``_reassert_isolation`` re-cordons >= 10 s later.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

NODE = "hyperpod-test-node-a"
SOURCE_WORKFLOW = "workflow-aaf326cf4be4f315223ef25b"
SOURCE_INCIDENT = "inc-host-node-a-efa_inventory_mismatch-node"
SUCCESSOR = "inc-reboot-after-" + SOURCE_WORKFLOW
HANDOFF = "inc-support-after-workflow-reboot-after-" + SOURCE_WORKFLOW
BOOT = "boot-before"
INCIDENT_KEY = "gpu-fault.io/incident-id"
FENCING_KEY = "gpu-fault.io/fencing-token"
#: Idle watch chunks after the script ran out before the fake refuses: a hold
#: that never stops fails the test instead of running out its window.
MAX_IDLE_CHUNKS = 200


class FakeApiError(Exception):
    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"({status}) {reason}")
        self.status = status
        self.reason = reason


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.base = datetime(2026, 11, 8, 3, 0, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.base + timedelta(seconds=self.now - 1000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _pointer_get(document: dict[str, Any], path: str) -> Any:
    target: Any = document
    for raw in path.strip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or key not in target:
            raise KeyError(path)
        target = target[key]
    return target


class FakeNodeServer:
    """One Node: reads, a scripted watch, JSON patches, product writes."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        incident: str | None = SOURCE_INCIDENT,
        fencing: str | None = "1",
        unschedulable: bool = True,
        boot: str = BOOT,
        successor: str = SUCCESSOR,
        handoff: str = HANDOFF,
    ) -> None:
        self.clock = clock
        self.successor = successor
        self.handoff = handoff
        self.idle_chunks = 0
        self.version = 100
        annotations = {}
        if incident:
            annotations[INCIDENT_KEY] = incident
        if fencing:
            annotations[FENCING_KEY] = fencing
        self.document: dict[str, Any] = {
            "metadata": {
                "name": NODE,
                "resourceVersion": str(self.version),
                "annotations": annotations,
                "managedFields": [{"manager": "x", "time": "2026-11-08T03:00:00Z"}],
            },
            "spec": {"taints": [{"key": "gpu-fault.io/quarantined"}]},
            "status": {"nodeInfo": {"bootID": boot}},
        }
        if unschedulable:
            self.document["spec"]["unschedulable"] = True
        self.queue: list[dict[str, Any]] = []
        self.script: list[Callable[[FakeNodeServer], Any]] = []
        self.patches: list[list[dict[str, Any]]] = []
        self.patch_errors: list[Exception | None] = []
        self.before_patch: Callable[[FakeNodeServer], None] | None = None
        self.watch_errors: list[Exception] = []
        self.read_errors: list[Exception] = []
        self.reads = 0
        self.watches: list[tuple[str, int]] = []
        self.closed = 0
        self.patch_seconds = 0.02

    # ------------------------------------------------------------ the store
    def commit(self) -> None:
        self.version += 1
        self.document["metadata"]["resourceVersion"] = str(self.version)
        self.queue.append({"type": "MODIFIED", "object": copy.deepcopy(self.document)})

    def product_write(
        self,
        *,
        incident: str | None = None,
        fencing: str | None = None,
        unschedulable: bool | None = None,
        boot: str | None = None,
        drop_incident: bool = False,
    ) -> None:
        annotations = self.document["metadata"]["annotations"]
        if incident is not None:
            annotations[INCIDENT_KEY] = incident
        if drop_incident:
            annotations.pop(INCIDENT_KEY, None)
        if fencing is not None:
            annotations[FENCING_KEY] = fencing
        if unschedulable is not None:
            if unschedulable:
                self.document["spec"]["unschedulable"] = True
            else:
                self.document["spec"].pop("unschedulable", None)
        if boot is not None:
            self.document["status"]["nodeInfo"]["bootID"] = boot
        self.commit()

    # --------------------------------------------------- the client surface
    def read(self) -> dict[str, Any]:
        self.reads += 1
        if self.read_errors:
            raise self.read_errors.pop(0)
        return copy.deepcopy(self.document)

    def watch(self, resource_version: str, seconds: int) -> Iterator[dict[str, Any]]:
        self.watches.append((resource_version, seconds))
        if self.watch_errors:
            raise self.watch_errors.pop(0)
        while True:
            if self.queue:
                yield self.queue.pop(0)
                continue
            if not self.script:
                self.idle_chunks += 1
                if self.idle_chunks > MAX_IDLE_CHUNKS:
                    raise AssertionError("the hold never stopped")
                self.clock.advance(seconds)
                return
            outcome = self.script.pop(0)(self)
            if outcome == "close":
                return

    def patch(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        self.patches.append(copy.deepcopy(ops))
        self.clock.advance(self.patch_seconds)
        if self.before_patch is not None:
            hook, self.before_patch = self.before_patch, None
            hook(self)
        if self.patch_errors:
            error = self.patch_errors.pop(0)
            if error is not None:
                raise error
        trial = copy.deepcopy(self.document)
        for op in ops:
            if op["op"] == "test":
                try:
                    value = _pointer_get(trial, op["path"])
                except KeyError:
                    raise FakeApiError(422, "test path missing") from None
                if value != op["value"]:
                    raise FakeApiError(422, "test failed")
            elif op["op"] in ("replace", "add"):
                parent, key = op["path"].rsplit("/", 1)
                container = _pointer_get(trial, parent)
                if op["op"] == "replace" and key not in container:
                    raise FakeApiError(422, "replace path missing")
                container[key] = op["value"]
            else:
                raise AssertionError(f"unexpected op {op}")
        if trial["spec"].get("unschedulable") is False:
            trial["spec"].pop("unschedulable")
        self.document = trial
        self.commit()
        return copy.deepcopy(self.document)

    def close(self) -> None:
        self.closed += 1

    # --------------------------------------------------------- inspection
    @property
    def cordoned(self) -> bool:
        return bool(self.document["spec"].get("unschedulable"))


def mark(server: FakeNodeServer) -> None:
    """The reboot-after successor's MARK_UNSCHEDULABLE: takes the cordoned
    node over (incident + fencing), the cordon stays."""

    server.clock.advance(0.3)
    server.product_write(incident=server.successor, fencing="1", unschedulable=True)


def quarantine(server: FakeNodeServer) -> None:
    """QUARANTINE: re-writes the cordon; visible only if it was lifted."""

    server.clock.advance(0.2)
    if not server.cordoned:
        server.product_write(unschedulable=True)


def handoff(server: FakeNodeServer) -> None:
    """The refused reboot's support-after takes the node over and cordons."""

    server.clock.advance(3.0)
    server.product_write(incident=server.handoff, fencing="1", unschedulable=True)


def reassert(server: FakeNodeServer) -> None:
    """A submitted reboot's ``_reassert_isolation`` re-cordons, ~12 s on."""

    server.clock.advance(12.0)
    server.product_write(unschedulable=True)
