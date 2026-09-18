from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional.probes import destr019_node_probe as agent
from scripts.e2e.regional.probes import destructive_node_probe as probe
from tests.regional._cov95_destr_node_io import NodeIO
from tests.regional._cov95_destr_warm import NOW
from tests.regional.test_destr_barrier_authorization import install_ledger, proof


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("naive-deadline", "no valid deadline"),
        ("incomplete-identity", "identity is incomplete"),
        ("missing-completion", "timestamp is unknown"),
        ("stale-refusal", "fresh GPU-client refusal"),
        ("wrong-refusal", "fresh GPU-client refusal"),
        ("advanced-workflow", "advanced beyond client verification"),
        ("ambiguous-quiesce", "missing or ambiguous"),
    ],
)
def test_node_guard_rechecks_authorization_before_any_fake_kmsg_write(
    defect: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    state_path = install_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr(authorization, "datetime", h.clock)
    permit = proof()
    rows = probe.ledger_rows()
    if defect == "naive-deadline":
        permit["observed_at"] = "2026-09-12T12:00:00"
    elif defect == "incomplete-identity":
        permit["fencing_token"] = True
    elif defect == "ambiguous-quiesce":
        (tmp_path / "quiesce-duplicate.json").write_text(
            state_path.read_text(), encoding="utf-8"
        )
    elif defect == "advanced-workflow":
        rows.append(
            {
                "workflow_request_id": permit["workflow_request_id"],
                "operation": "RESTORE_GPU_SERVICES",
            }
        )
    else:
        verify = next(
            row for row in rows if row["operation"] == "VERIFY_NO_GPU_CLIENTS"
        )
        if defect == "missing-completion":
            verify.pop("completed_at")
        elif defect == "stale-refusal":
            verify["completed_at"] = (NOW - timedelta(seconds=31)).isoformat()
        else:
            verify["error"] = "unknown verifier failure"
    monkeypatch.setattr(probe, "ledger_rows", lambda: rows)
    code, result = h.main(
        monkeypatch,
        "write-xid46",
        "--marker",
        "owned",
        "--drill-id",
        "review-r",
        "--pci-bdf",
        "0000:0a:00",
        "--barrier-authorization",
        json.dumps(permit),
    )
    assert code == 1 and expected in result["error"], result
    assert h.writes == h.opened == [], h.writes
    assert h.calls == [], h.calls


@pytest.mark.parametrize("exists", [False, True])
def test_agent_addressing_environment_never_returns_credential_or_unknown_keys(
    exists: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "node-agent.env"
    monkeypatch.setattr(agent, "AGENT_ENV", path)
    expected: dict[str, Any] = {}
    if exists:
        path.write_text(
            "\n# ignored comment\nnot an assignment\n"
            "GPU_FAULT_NODE_AGENT_HOST='127.0.0.1'\n"
            'GPU_FAULT_NODE_AGENT_PORT="8443"\n'
            "GPU_FAULT_NODE_ADVERTISE_URL=https://node.invalid:8443/path?q=1\n"
            "GPU_FAULT_CLUSTER_TOKEN=fake-unit-value\n"
            "UNKNOWN_KEY=ignored\n",
            encoding="utf-8",
        )
        expected = {
            "GPU_FAULT_NODE_AGENT_HOST": "127.0.0.1",
            "GPU_FAULT_NODE_AGENT_PORT": "8443",
            "GPU_FAULT_NODE_ADVERTISE_URL": "https://node.invalid:8443/path?q=1",
        }
    observed = agent.agent_env()
    assert observed == expected, {"keys": sorted(observed)}
    assert set(observed) <= set(agent.AGENT_ENV_KEYS), sorted(observed)
