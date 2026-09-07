"""Every retention sweep says what it deleted: kind, count and the first keys.

Architecture review 2026-09-07, item D7. Six cleanup paths deleted rows and
returned only a count, so a sweep that removed the wrong rows -- or a record an
operator was about to look at -- left nothing to correlate against. Each path
now logs one INFO line per kind through ``store.shared.cleanup_log`` with the
count and a bounded prefix of the deleted keys.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.fleet import (
    DeploymentNode,
    DeploymentNodeStatus,
    DeploymentStatus,
    FleetDeployment,
)
from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.cleanup_log import CLEANUP_LOG_KEY_LIMIT, log_cleanup
from gpu_fault.telemetry import EvidenceKind, EvidenceService
from tests._builders import build_store, copy_model, processor_request
from tests.store._postgres_processor_claim_support import _command

LOGGER_NAME = "gpu_fault.store.cleanup"
NOW = datetime.now(timezone.utc)
LATER = NOW + timedelta(days=2)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    sqlite = SqliteStore(str(tmp_path / "cleanup-log.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def _messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.INFO
    ]


def test_the_helper_bounds_the_keys_it_prints(caplog) -> None:
    keys = [f"key-{index:02d}" for index in range(CLEANUP_LOG_KEY_LIMIT + 5)]

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        assert log_cleanup("example", keys) == len(keys)
        assert log_cleanup("example", []) == 0, "an empty sweep returns zero"

    (message,) = _messages(caplog)
    assert "example" in message
    assert str(len(keys)) in message
    assert "key-19" in message
    assert "key-20" not in message, "only the first 20 keys are printed"
    assert "+5 more" in message


def test_raw_evidence_sweep_logs_what_it_deleted(store, caplog) -> None:
    service = EvidenceService(store, retention=timedelta(hours=1))
    for index in range(2):
        service.capture(
            record_id=f"evidence-{index}",
            cluster_id="cluster-a",
            node_id="node-a",
            kind=EvidenceKind.GPU_METRICS,
            observed_at=NOW,
            attempt_ids=[],
            payload={},
        )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        deleted = store.cleanup_expired_raw_evidence(now=LATER)

    assert deleted == 2
    (message,) = _messages(caplog)
    assert "raw_evidence" in message
    assert "2" in message
    assert "evidence-0" in message and "evidence-1" in message


def test_terminal_remote_command_sweep_logs_what_it_deleted(store, caplog) -> None:
    command = _command(store, "remote-done", request_id="wf-done")
    claimed = store.claim_remote_commands(
        command.cluster_id, "executor-a", limit=1, lease_seconds=60
    )
    store.complete_remote_command(
        command.cluster_id,
        "remote-done",
        RemoteCommandResult(
            lease_token=claimed[0].lease_token, status=RemoteCommandStatus.SUCCEEDED
        ),
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        deleted = store.cleanup_terminal_remote_commands(older_than=LATER, limit=10)

    assert deleted == 1
    (message,) = _messages(caplog)
    assert "remote_command" in message and "remote-done" in message


def test_terminal_fleet_deployment_sweep_logs_what_it_deleted(store, caplog) -> None:
    deployment = FleetDeployment(
        cluster_id="cluster-a",
        desired_agent_version="0.9.0",
        desired_artifact_sha256="a" * 64,
        desired_policy_version="catalog-a",
        desired_runtime_profile_version="profile-a",
        desired_config_digest="c" * 64,
        max_unavailable=1,
        waves=[["node-a"]],
        nodes=[
            DeploymentNode(
                node_id="node-a", status=DeploymentNodeStatus.READY, updated_at=NOW
            )
        ],
        status=DeploymentStatus.SUCCEEDED,
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_fleet_deployment(deployment)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        deleted = store.cleanup_terminal_fleet_deployments(older_than=LATER, limit=10)

    assert deleted == 1
    (message,) = _messages(caplog)
    assert "fleet_deployment" in message and deployment.deployment_id in message


def test_completed_processor_request_sweep_logs_what_it_deleted(store, caplog) -> None:
    request = copy_model(
        processor_request("/v1/collector-events/host-telemetry"),
        status=ProcessorRequestStatus.COMPLETED,
        updated_at=NOW,
    )
    store.enqueue_processor_request(request)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        deleted = store.cleanup_completed_processor_requests(older_than=LATER, limit=10)

    assert deleted == 1
    (message,) = _messages(caplog)
    assert "processor_request" in message and request.request_id in message


def test_an_empty_sweep_logs_nothing(store, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        store.cleanup_expired_raw_evidence(now=LATER)
        store.cleanup_terminal_remote_commands(older_than=LATER, limit=10)
        store.cleanup_terminal_fleet_deployments(older_than=LATER, limit=10)
        store.cleanup_completed_processor_requests(older_than=LATER, limit=10)
        store.cleanup_processor_lanes(older_than=LATER, limit=10)
        store.cleanup_hot_state(now=LATER)

    assert _messages(caplog) == []
