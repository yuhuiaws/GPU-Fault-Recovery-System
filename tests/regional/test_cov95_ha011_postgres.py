"""Real local PG/worker-process takeover, not deployed-Pod or saturation evidence."""

from __future__ import annotations

import os
from contextlib import closing
from uuid import uuid4

import pytest

from scripts.e2e.regional.ha011_contracts import BACKLOG, MIN_CPU_SECONDS, ROLES
from scripts.e2e.regional.probes.ha011_probe import exercise_role
from tests.regional._cov95_ha011_postgres import GrantedWorkers, isolated_database

pytestmark = pytest.mark.skipif(
    not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="requires the separately granted serial local PostgreSQL slot",
)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.allows_cluster_binaries("docker")
def test_postgres_real_worker_crash_preserves_live_fence_and_drains_backlog(
    monkeypatch, role: str
) -> None:
    # The shared grant guard only inspects the allocated container's owner/port.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("HA011 PostgreSQL process tests require explicit serial -n0")
    for name in tuple(os.environ):
        if name.startswith("GPU_FAULT_") and name != "GPU_FAULT_TEST_POSTGRES_URL":
            monkeypatch.delenv(name)
    identity = uuid4().hex
    pod_identity = f"local-pg-owned-process-{identity}"
    monkeypatch.setenv("POD_UID", pod_identity)
    with isolated_database() as store_factory:
        with closing(store_factory()) as store:
            proof = exercise_role(
                store, role, identity, factory=GrantedWorkers(store_factory)
            )
    assert proof["old_exitcode"] == -9, (
        "the old owner must really exit from the owned SIGKILL"
    )
    assert proof["old_cpu_seconds"] >= MIN_CPU_SECONDS, (
        "the crashed production worker must really be busy"
    )
    assert proof["replacement_cpu_seconds"] >= MIN_CPU_SECONDS, (
        "the replacement must still be busy during late completion"
    )
    assert proof["same_durable_work"] and proof["fence_changed"], (
        "takeover must preserve work and change its lease fence"
    )
    assert (
        proof["replacement_live_before_late"] and proof["replacement_live_after_late"]
    ), (
        "late completion must be rejected while the replacement still owns a live PostgreSQL row"
    )
    assert proof["late_completion_refused"] and proof["early_claim_count"] == 0, (
        "neither an unexpired takeover nor an old-owner completion may succeed"
    )
    assert proof["completed_count"] == BACKLOG + 1 and proof["final_depth"] == 0, (
        "real worker processes must complete the exact original PostgreSQL backlog"
    )
    assert proof["owned_processes_stopped"], (
        "all owned processes must be drained before database cleanup"
    )
    assert len(set(proof["owners"])) == 2 and all(
        owner.startswith(pod_identity + ":") for owner in proof["owners"]
    ), "takeover must be between two actually spawned local worker identities"
    assert "validation_scope" not in proof, (
        "this local process/PG contract must not manufacture deployed acceptance evidence"
    )
