"""Controller lifetime of the two independent read-only reset exec witnesses."""

from __future__ import annotations

import time
from typing import Any

from scripts.e2e.regional.destr015_physical_evidence import ResetIntervalScope
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_probe_bundle import probe_program
from scripts.e2e.regional.late_ownership_stream import ProbeStream, open_executor_stream


class ResetIntervalWitness:
    def __init__(self, probe: HostProbeFixture, scope: ResetIntervalScope) -> None:
        self.probe = probe
        self.scope = scope
        self.stream: ProbeStream | None = None
        self.capture: dict[str, Any] = {"clock_exchanges": []}
        self.finished = False

    def _exchange(self, kind: str, expected: str) -> dict[str, Any]:
        if self.stream is None:
            raise BoundaryDenied("physical witness stream is missing")
        sent = time.monotonic_ns()
        self.stream.send(
            kind,
            self.scope.model_dump(mode="json") if kind == "arm" else {},
        )
        result = self.stream.receive(expected)
        received = time.monotonic_ns()
        if type(result.get("monotonic_ns")) is not int:
            raise BoundaryDenied("physical witness returned no clock binding")
        self.capture["clock_exchanges"].append(
            {
                "sent_ns": sent,
                "received_ns": received,
                "monotonic_ns": result["monotonic_ns"],
            }
        )
        return result

    def start(self) -> None:
        if self.stream is not None or self.finished:
            raise BoundaryDenied("physical witness cannot be reused")
        self.probe._check_target()
        program, _ = probe_program("reset-interval")
        self.stream = ProbeStream(
            open_executor_stream(
                kubeconfig=str(self.probe.settings.kubeconfig),
                context=self.probe.settings.context,
                namespace=self.probe.settings.namespace,
                pod=self.probe.pod,
                container="probe",
                chroot="/host",
                python="/opt/gpu-fault/current/venv/bin/python",
                program=program,
            ),
            scope_sha256=self.scope.digest(),
            check_identity=self.probe._check_target,
            deadline=time.monotonic() + 3600,
        )
        self.capture["start"] = self._exchange("arm", "armed")["start"]
        self.poll()

    def poll(self) -> None:
        self._exchange("clock", "clock")

    def finish(self) -> dict[str, Any]:
        self.capture["end"] = self._exchange("finish", "finished")["end"]
        if self.stream is None:
            raise BoundaryDenied("physical witness stream disappeared")
        self.stream.finish()
        self.finished = True
        return self.capture

    def close(self) -> dict[str, Any]:
        if self.finished:
            return {"closed": True, "proof_complete": True}
        if self.stream is None:
            return {"closed": True, "not_started": True}
        try:
            self.stream.send("abort", {})
            result = self.stream.receive("closed")
            self.stream.finish()
            return result
        finally:
            # Closing the channel is not a successful cleanup receipt; callers
            # retain any error. The node witness independently expires/detaches.
            self.stream.close()
