"""Capacity acceptance base: the CAP-002 AMP deadline, the exit cleanup, case stubs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.credentials import Credentials

from scripts.e2e.regional import capacity_acceptance_base as base
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture


@pytest.fixture
def signed(capacity: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The capacity harness with local SigV4 signing over fixed fake credentials
    (installed after the harness fixture, which refuses ambient AWS sessions)."""

    session = SimpleNamespace(
        get_credentials=lambda: Credentials("unit-access", "unit-signing-key")
    )
    monkeypatch.setattr(base.boto3, "Session", lambda: session)
    return capacity


@pytest.mark.parametrize(
    ("deadline", "message"),
    [
        (
            datetime(2026, 11, 5, 12, 0),
            "CAP-002 requires an aware maintenance deadline",
        ),
        (
            datetime.now(timezone.utc) - timedelta(minutes=1),
            "CAP-002 maintenance deadline has expired",
        ),
    ],
)
def test_cap002_amp_reads_need_an_open_aware_maintenance_deadline(
    signed: Any, deadline: datetime, message: str
) -> None:
    harness, _api = signed
    harness.case_id = "GF-REGIONAL-CAP-002"
    harness.maintenance_deadline = deadline

    # The fixture fails any real URL open: the refusal comes before the request.
    with pytest.raises(base.CapError, match=message):
        harness.amp_request("GET", "/api/v1/query", {"query": "up"})


def test_the_exit_cleanup_records_nothing_for_a_clean_probe(
    capacity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, _api = capacity
    probe = SimpleNamespace(case="CAP-001", deployment="probe-cap001", database="db")
    harness.active_probe = probe
    calls: list[str] = []

    def cleanup_probe(value: Any) -> dict[str, Any]:
        calls.append(f"probe {value.case}")
        return {"database_dropped": True, "residual_probe_pods": []}

    monkeypatch.setattr(harness, "cleanup_probe", cleanup_probe)
    monkeypatch.setattr(harness, "cleanup_common", lambda: calls.append("common"))

    assert harness.cleanup_all() == []
    assert calls == ["probe CAP-001", "common"]
    assert (harness.run_dir / "cap-001-cleanup-on-exit.json").is_file(), (
        "the probe's cleanup receipt is kept"
    )


@pytest.mark.parametrize("case", ["case_001", "case_002_v2", "case_003", "case_004"])
def test_the_base_harness_implements_no_case(capacity: Any, case: str) -> None:
    harness, _api = capacity

    with pytest.raises(NotImplementedError):
        getattr(harness, case)()
