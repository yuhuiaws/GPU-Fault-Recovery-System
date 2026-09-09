"""The release gates run without the deploy's own consent variables.

Deploy #18 and #22 (2026-09-09): the gate's pytest inherited
GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE / GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION
from the deploy command and judged the tree as if every release had consented.
"""

from __future__ import annotations

from gpu_fault.admin import release_artifacts
from gpu_fault.admin.release_consent import (
    ACCEPT_SCHEMA_CHANGE_ENV,
    ALLOW_INFLIGHT_INSTALLS_ENV,
    SUPERSEDE_FAILED_TRANSACTION_ENV,
)


def test_consent_variables_stay_out_of_the_gate_environment() -> None:
    parent = {
        "PATH": "/usr/bin",
        "KUBECONFIG": "/secure/cpu.kubeconfig",
        ACCEPT_SCHEMA_CHANGE_ENV: "snapshot",
        SUPERSEDE_FAILED_TRANSACTION_ENV: "1",
        ALLOW_INFLIGHT_INSTALLS_ENV: "1",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://stale",
    }

    environment = release_artifacts.release_gate_environment(
        parent,
        postgres_url="postgresql://gate:pw@127.0.0.1:5432/postgres",
        cosign_password="secret",
    )

    assert environment["PATH"] == "/usr/bin" and environment["KUBECONFIG"], environment
    for name in (
        ACCEPT_SCHEMA_CHANGE_ENV,
        SUPERSEDE_FAILED_TRANSACTION_ENV,
        ALLOW_INFLIGHT_INSTALLS_ENV,
    ):
        assert name not in environment, name
    assert environment["GPU_FAULT_TEST_POSTGRES_URL"].startswith(
        "postgresql://gate:"
    ), "the gate's own PostgreSQL wins over an inherited URL"
    assert environment["COSIGN_PASSWORD"] == "secret", (
        "the signing password is passed through"
    )
    assert "COSIGN_PASSWORD" not in release_artifacts.release_gate_environment(
        {}, postgres_url="", cosign_password=None
    ), "no password means no variable"
    assert parent[ACCEPT_SCHEMA_CHANGE_ENV] == "snapshot", (
        "the parent mapping is not mutated"
    )
