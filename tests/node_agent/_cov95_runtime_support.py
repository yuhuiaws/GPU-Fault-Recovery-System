from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.node_agent import NodeActionExecutor, NodeActionLedger
from tests.node_agent._support import NOW, SECRET, FakeRunner


@pytest.fixture(name="node_factory")
def node_factory_fixture(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Callable[..., NodeActionExecutor]:
    ledgers = []

    def create(**changes: Any) -> NodeActionExecutor:
        ledger = NodeActionLedger(str(tmp_path / f"runtime-{len(ledgers)}.db"))
        ledgers.append(ledger)
        request.addfinalizer(ledger.close)
        options = {
            "secret": SECRET,
            "node_ids": {"node-a"},
            "allowed_operations": set(),
            "reset_enabled": False,
            "ledger": ledger,
            "runner": FakeRunner(),
            "device_client_finder": lambda targets: [],
            "gpu_device_path_finder": lambda: {
                "GPU-a": "/unit/nvidia0",
                "GPU-b": "/unit/nvidia1",
            },
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "infiniband"),
            "diagnostic_output_dir": str(tmp_path / "diagnostics"),
            "health_snapshot_request_dir": str(tmp_path / "health"),
            "now": lambda: NOW,
            "sleep": lambda seconds: None,
            **changes,
        }
        return NodeActionExecutor(**options)

    return create
