"""Real DESTR014 recovery protocol over hermetic host and transport doubles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.probes import destr014_recovery_probe as host_probe
from tests.regional._cov95_destr_warm import NOW, Clock
from tests.regional.test_destr014_recovery_controller import ProbeTransport
from tests.regional.test_destr014_recovery_probe import HostHarness

if TYPE_CHECKING:
    from tests.regional._cov95_destr_exhaustion import ExhaustionHarness


class RecoveryClock(Clock):
    def __init__(self, host: HostHarness) -> None:
        super().__init__()
        self.host = host
        self.host.now = NOW.timestamp()

    def monotonic(self) -> float:
        return self.host.now - NOW.timestamp()

    def time(self) -> float:
        return self.host.now

    def sleep(self, seconds: float) -> None:
        self.host.sleep(seconds)

    def now(self, tz: Any = None) -> datetime:
        return datetime.fromtimestamp(self.host.now, tz=tz or timezone.utc)


class RecoveryTransport(ProbeTransport):
    def __init__(self, harness: ExhaustionHarness, host: HostHarness) -> None:
        super().__init__(host)
        self.harness = harness

    def create(self) -> None:
        self.harness.call("recovery.probe.create")
        super().create()

    def execute(
        self, command: str, flag: str, raw: str, *, timeout: int
    ) -> dict[str, Any]:
        self.harness.call("recovery." + command)
        self.host.scope = json.loads(raw)
        self.host.recovery = host_probe.Recovery(self.host.scope)
        saved = json.loads(self.harness.journal_path.read_text())
        assert saved["host_binding"] == self.host.scope, (
            "request must use the durable host binding"
        )
        assert saved["host_request"] == command, (
            "request intent must precede transport I/O"
        )
        if command == "disable":
            assert saved["run"]["agent_disabled"] is True
            assert saved["host_ack"]["phase"] == "ARMED"
            assert saved["host_ack"]["ack"]["invocation_id"] == self.host.invocation
        return super().execute(command, flag, raw, timeout=timeout)

    def cleanup(self) -> dict[str, bool]:
        self.harness.call("recovery.probe.cleanup")
        return dict(super().cleanup())
