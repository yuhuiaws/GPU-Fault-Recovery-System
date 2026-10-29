"""GF-REGIONAL-DESTR-014's remediation budget probe, run on a store.

The probe runs in a control-worker Pod on whatever release that Pod carries:
the node scopes of the case's two branches, the limits that release has, and
the cluster's remediation storm breaker -- open, it would hold the case's
first destructive admission, so the preflight refuses.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.remediation_storm import StormBreakerState, closed_breaker
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014


def test_the_budget_probe_runs_on_the_store_and_reads_the_storm_breaker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InMemoryStore()
    tripped = closed_breaker("cluster-a", datetime.now(timezone.utc)).model_copy(
        update={"state": StormBreakerState.TRIPPED, "trip_id": "cluster-a@t"}
    )
    store.replace_remediation_storm_breaker(tripped, expected=None)
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=store)),
    )
    monkeypatch.setattr(sys, "argv", ["probe", "cluster-a", "node-a", "node-b"])

    exec(compile(destr014.BUDGET_HEADROOM, "<budget-headroom>", "exec"), {})

    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(result["scopes"]) == {"node:cluster-a:node-a", "node:cluster-a:node-b"}
    assert result["policy"] == {"node_limit": 1, "failure_domain_limit": 1}
    assert result["storm_breaker"] == {
        "open": True,
        "state": "TRIPPED",
        "trip_id": "cluster-a@t",
    }
    errors = destr014.budget_headroom_errors(result)
    assert any("storm breaker" in item for item in errors), errors
