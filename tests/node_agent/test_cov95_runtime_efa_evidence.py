from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._cov95_runtime_hung import BundleRunner, capture_bundle
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    "failure", ["missing-root", "read", "ethtool-io", "ethtool-timeout", "none"]
)
def test_efa_bundle_preserves_partial_counters_and_explicit_capture_failures(
    node_factory: Callable[..., NodeActionExecutor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / "infiniband"
    port = root / "efa0" / "ports" / "1"
    state, counter = port / "state", port / "hw_counters" / "rx_bytes"
    if failure != "missing-root":
        (root / "efa0" / "device" / "net" / "ens5").mkdir(parents=True)
        (port / "hw_counters" / "not-a-counter").mkdir(parents=True)
        state.write_text("4: ACTIVE")
        counter.write_text("100")
        (root / "not-a-device").write_text("ignored")
    original_read = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if failure == "read" and path in {state, counter}:
            raise PermissionError("synthetic counter read denied")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    class Runner(BundleRunner):
        def __call__(
            self, argv: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess:
            if argv[0] != "ethtool":
                return super().__call__(argv, **kwargs)
            self.calls.append(argv)
            if failure == "ethtool-io":
                raise OSError("synthetic ethtool unavailable")
            if failure == "ethtool-timeout":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            return subprocess.CompletedProcess(argv, 0, "packets: 10", "unit advisory")

    runner = Runner("")
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE}, runner=runner
    )
    manifest, contents = capture_bundle(
        agent, diagnostic_reason="EFA_TRAFFIC_HUNG_SUSPECTED"
    )
    snapshot = json.loads(contents["infiniband-counters.json"])
    captures = {item["file"]: item for item in manifest["captures"]}
    if failure == "missing-root":
        assert snapshot["devices"] == {}
        assert "FileNotFoundError" in snapshot["error"]
        assert captures["infiniband-counters.json"]["returncode"] is None
        assert not any(argv[0] == "ethtool" for argv in runner.calls), (
            "missing interface inventory must not trigger an ethtool query"
        )
        return
    data = snapshot["devices"]["efa0"]["ports"]["1"]
    assert set(snapshot["devices"]) == {"efa0"}
    assert "not-a-counter" not in data["hw_counters"]
    assert data["counters"] == {}
    assert "phys_state" not in data
    if failure == "read":
        assert "PermissionError" in data["state"]["error"]
        assert "PermissionError" in data["hw_counters"]["rx_bytes"]["error"]
    else:
        assert data["state"] == "4: ACTIVE"
        assert data["hw_counters"]["rx_bytes"] == "100"
    ethtool = captures["ethtool-ens5-statistics.txt"]
    assert ethtool["command"] == ["ethtool", "-S", "ens5"]
    if failure.startswith("ethtool"):
        assert ethtool["returncode"] is None
        assert "error" in ethtool
    else:
        assert ethtool["returncode"] == 0
        assert b"[stderr]\nunit advisory" in contents["ethtool-ens5-statistics.txt"]


def test_triage_counter_delta_ignores_nontraffic_or_unreadable_samples(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path
) -> None:
    counters = tmp_path / "infiniband" / "efa0" / "ports" / "1" / "hw_counters"
    counters.mkdir(parents=True)
    (counters / "rx_bytes").write_text("10")
    (counters / "tx_bytes").write_text("invalid")
    (counters / "send_packets").mkdir()
    (counters / "unrelated").write_text("1000")

    def sleep(seconds: float) -> None:
        (counters / "rx_bytes").write_text("30")
        (counters / "unrelated").write_text("9000")

    runner = BundleRunner("")
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="",
        sleep=sleep,
    )
    result = agent.execute(envelope(command(WorkflowOperation.COLLECT_HUNG_TRIAGE)))
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["efa"] == {
        "counter_deltas": {str(counters / "rx_bytes"): 20},
        "total_delta": 20,
    }
    assert result.details["ranks"] == []
    assert result.details["read_only"] is True
