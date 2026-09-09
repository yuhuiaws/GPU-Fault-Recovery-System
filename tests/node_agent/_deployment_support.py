"""Shared fixtures for the ``test_node_deployment_*`` modules.

The node deployment scripts under ``deploy/node`` are exercised by six
thematic test modules (rollout, units, wheel, verify, degraded GPU,
ledger drain). They share the script table, the PATH-stub writer and the
ledger-drain probe, which ``test_node_installer_drain_budget.py`` also
borrows; those live here so no test module imports another test module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from gpu_fault.node_agent.ledger import NodeActionLedger


ROOT = Path(__file__).parents[2]

NODE_SCRIPTS = (
    ROOT / "deploy/node/install-gpu-fault-collector.sh",
    ROOT / "deploy/node/verify-gpu-fault-collector.sh",
    ROOT / "deploy/node/uninstall-gpu-fault-collector.sh",
    ROOT / "deploy/node/build-node-installer-bundle.sh",
    ROOT / "deploy/node/run-hyperpod-installer-job.sh",
    ROOT / "deploy/node/preflight-gpu-fault-node.sh",
    ROOT / "deploy/node/provision-node-action-keys.sh",
    ROOT / "deploy/node/verify-certificate-bundle.sh",
    ROOT / "deploy/node/check-control-plane-certificate.sh",
)


def _write_stub(directory: Path, name: str, body: str) -> None:
    stub = directory / name
    stub.write_text(f"#!/usr/bin/env bash\n{body}", encoding="utf-8")
    stub.chmod(0o755)


LEDGER_DRAIN_START = "in_progress_node_action_ids() {"

LEDGER_DRAIN_END = (
    "drain_node_agent_before_restart() {\n"
    "    systemctl is-active --quiet gpu-fault-node-agent.service || return 0\n"
    "    wait_for_node_action_ledger_idle\n"
    "}"
)


def _ledger_drain_probe(target: Path) -> Path:
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(LEDGER_DRAIN_START) == 1, (
        "the installer must read the IN_PROGRESS ledger rows from one place"
    )
    assert installer.count(LEDGER_DRAIN_END) == 1, (
        "the drain must be guarded by the Agent unit being active"
    )
    start = installer.index(LEDGER_DRAIN_START)
    end = installer.index(LEDGER_DRAIN_END) + len(LEDGER_DRAIN_END)
    probe = target / "drain-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "PYTHON_COMMAND=python3\n"
        'NODE_ACTION_DB="$1"\n'
        'SQLITE_COMMAND="$2"\n'
        'NODE_AGENT_STOP_TIMEOUT_SECONDS="${3:-0}"\n'
        'NODE_AGENT_LEDGER_POLL_SECONDS="${4:-1}"\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        "wait_for_node_action_ledger_idle\n"
        "printf 'DRAINED\\n'\n",
        encoding="utf-8",
    )
    return probe


def _ledger_with_state(path: Path, state: str) -> None:
    NodeActionLedger(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO results (command_id, attempt, payload, state)"
            " VALUES (?, ?, ?, ?)",
            ("cmd-in-flight", 1, "{}", state),
        )
