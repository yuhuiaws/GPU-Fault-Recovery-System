from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.store.shared.errors import WorkflowLeaseError
from scripts.e2e.regional.probes import cap004_commands as probe

RUN_ID = "cap004cleanup"


class LeaseStore:
    def __init__(self) -> None:
        self.rows = [
            item.model_copy(
                update={
                    "status": RemoteCommandStatus.LEASED,
                    "lease_owner": "stopped-fixture",
                    "lease_token": f"original-{index}",
                    "lease_expires_at": datetime.now(timezone.utc)
                    + timedelta(seconds=10),
                }
            )
            for index, item in enumerate(probe.commands_for_run(RUN_ID)[:2])
        ]
        self.claims = 0
        self.allow_reclaim = True
        self.completions: list[str] = []

    def list_agents(self):
        return []

    def list_remote_commands(self):
        return [item.model_copy(deep=True) for item in self.rows]

    def claim_remote_commands(self, cluster_id, owner, **kwargs):
        assert cluster_id == probe.CLUSTER_ID
        assert owner == f"{RUN_ID}-cleanup"
        assert kwargs["execution_owners"] == {probe.OWNER}
        self.claims += 1
        if self.claims < 3 or not self.allow_reclaim:
            return []
        self.rows[0] = self.rows[0].model_copy(
            update={"lease_owner": owner, "lease_token": f"cleanup-{self.claims}"}
        )
        return [self.rows[0].model_copy(deep=True)]

    def complete_remote_command(self, cluster_id, command_id, result):
        assert cluster_id == probe.CLUSTER_ID
        self.completions.append(result.lease_token)
        index = next(
            index
            for index, item in enumerate(self.rows)
            if item.command_id == command_id
        )
        if result.lease_token == "original-0":
            self.rows[index] = self.rows[index].model_copy(
                update={
                    "lease_token": "another-holder-token",
                    "lease_owner": "another-holder",
                }
            )
            raise WorkflowLeaseError("the original fixture lease expired")
        assert result.lease_token == self.rows[index].lease_token, (
            "cleanup attempted completion without the current lease"
        )
        self.rows[index] = self.rows[index].model_copy(
            update={
                "status": result.status,
                "lease_owner": None,
                "lease_token": None,
                "lease_expires_at": None,
                "result_details": result.details,
            }
        )
        return self.rows[index].model_copy(deep=True)


@pytest.fixture
def clock(monkeypatch):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep)
    )
    monkeypatch.setattr(probe, "CLEANUP_TIMEOUT_SECONDS", 0.4)
    return elapsed


def test_cleanup_reclaims_a_changed_lease_without_borrowing_its_token(clock):
    store = LeaseStore()
    snapshot = probe.command_snapshot(store, RUN_ID, "cleanup")
    assert snapshot["status_counts"] == {"FAILED": 2}, (
        "cleanup must preserve failure instead of fabricating successful execution"
    )
    assert store.completions == ["original-0", "original-1", "cleanup-3"]
    assert all(not row["lease_present"] for row in snapshot["commands"]), (
        "successful cleanup must prove every lease is released"
    )
    assert clock[0] < probe.CLEANUP_TIMEOUT_SECONDS


def test_cleanup_stops_at_its_deadline_when_a_new_holder_never_releases(clock):
    store = LeaseStore()
    store.allow_reclaim = False
    with pytest.raises(TimeoutError, match="CAP004 cleanup"):
        probe.command_snapshot(store, RUN_ID, "cleanup")
    assert store.completions == ["original-0", "original-1"], (
        "a new holder's lease must not become cleanup authority"
    )
    assert store.rows[0].status is RemoteCommandStatus.LEASED
    assert clock[0] == probe.CLEANUP_TIMEOUT_SECONDS


def test_cleanup_refuses_missing_initial_authority_without_waiting(clock):
    store = LeaseStore()
    store.allow_reclaim = False
    store.rows[0] = store.rows[0].model_copy(
        update={
            "status": RemoteCommandStatus.PENDING,
            "lease_owner": None,
            "lease_token": None,
            "lease_expires_at": None,
        }
    )
    with pytest.raises(ValueError, match="cannot establish command ownership"):
        probe.command_snapshot(store, RUN_ID, "cleanup")
    assert store.claims == 1
    assert store.completions == [], (
        "missing initial claim authority must prevent completion attempts"
    )
    assert clock[0] == 0, "missing authority is not an expired-lease waiting state"


def test_cleanup_revalidates_identity_before_completing_a_reclaimed_row(
    clock, monkeypatch
):
    store = LeaseStore()
    claim = store.claim_remote_commands

    def drift(*args, **kwargs):
        rows = claim(*args, **kwargs)
        if rows:
            rows[0] = rows[0].model_copy(update={"fencing_token": 2})
        return rows

    monkeypatch.setattr(store, "claim_remote_commands", drift)
    with pytest.raises(ValueError, match="identity differs"):
        probe.command_snapshot(store, RUN_ID, "cleanup")
    assert store.completions == ["original-0", "original-1"], (
        "identity drift must stop cleanup before completing the changed row"
    )


def test_cleanup_does_not_retry_unknown_store_errors(clock, monkeypatch):
    store = LeaseStore()

    def unavailable(*_args, **_kwargs):
        raise OSError("Store outcome unavailable")

    monkeypatch.setattr(store, "complete_remote_command", unavailable)
    with pytest.raises(OSError, match="outcome unavailable"):
        probe.command_snapshot(store, RUN_ID, "cleanup")
    assert store.claims == 1, "only lease rejection may enter the reclamation loop"
    assert clock[0] == 0
