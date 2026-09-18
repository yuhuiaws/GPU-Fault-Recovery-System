from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_state import load_join_state
from tests.admin._cov95_join_support import JoinScenario, target


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return JoinScenario(tmp_path, monkeypatch)


def expire_during_runtime_read(scenario, monkeypatch, limit):
    clock = SimpleNamespace(value=datetime.now(timezone.utc), expired=0)
    timestamps = set()
    proxy = SimpleNamespace(
        now=lambda _zone: clock.value, fromisoformat=datetime.fromisoformat
    )
    monkeypatch.setattr(evidence, "datetime", proxy)
    monkeypatch.setattr(batch, "datetime", proxy)
    command = scenario.command

    def read(arguments, **options):
        result = command(arguments, **options)
        if arguments[-1] == evidence.REGISTRY_STATUS_CLIENT and clock.expired < limit:
            for suffix in ("b", "c"):
                if not scenario.state_path(suffix).is_file():
                    continue
                state = scenario.state(suffix)
                proof = state["evidence"].get("VERIFIED") or {}
                timestamp = proof.get("verified_at")
                if (
                    timestamp
                    and timestamp not in timestamps
                    and "ACTIVATION_STARTED" not in state["completed_steps"]
                ):
                    timestamps.add(timestamp)
                    clock.value += timedelta(seconds=901)
                    clock.expired += 1
                    break
        return result

    monkeypatch.setattr(evidence, "run_command", read)
    return clock


@pytest.mark.parametrize("batch_mode", [False, True])
@pytest.mark.parametrize("expirations", [1, 2])
def test_join_reverification_is_bounded_and_never_repeats_gpu_rollout(
    scenario, monkeypatch, batch_mode, expirations
):
    clock = expire_during_runtime_read(scenario, monkeypatch, expirations)
    run = scenario.batch if batch_mode else scenario.join
    if expirations == 1:
        assert run()["phase"] == "COMPLETED"
    else:
        with pytest.raises(BootstrapError, match="expired again|failed clusters"):
            run()
        assert not any(
            mode == "activate-cluster" for mode, _cluster in scenario.driver_calls
        ), "repeatedly expired evidence authorized cluster activation"
    assert clock.expired == expirations
    assert scenario.driver_calls.count(("verify", None)) == 2
    assert scenario.driver_calls.count(("join-cluster", "hp-gpu-b")) == 1
    if batch_mode:
        assert scenario.driver_calls.count(("join-cluster", "hp-gpu-c")) == 1


def test_saved_expired_join_proof_is_reverified_before_resuming_commit(scenario):
    scenario.failure = "registry-sync"
    with pytest.raises(BootstrapError, match="modeled join boundary"):
        scenario.join()
    state = scenario.state()
    state["evidence"]["VERIFIED"]["verified_at"] = "2020-01-01T00:00:00+00:00"
    scenario.state_path().write_text(json.dumps(state))
    scenario.failure = None
    assert scenario.join()["phase"] == "COMPLETED"
    assert scenario.driver_calls.count(("verify", None)) == 2
    assert scenario.driver_calls.count(("join-cluster", "hp-gpu-b")) == 1


def test_batch_discovery_failure_does_not_block_independent_target(scenario):
    scenario.targets[target().eks_arn] = replace(target(), node_recovery="Automatic")
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert scenario.state()["phase"] == "FAILED"
    assert scenario.state("c")["phase"] == "COMPLETED"
    assert not any(
        event.endswith("gpu-b") and event.startswith(("executor-role:", "node-keys:"))
        for event in scenario.events
    ), "invalid discovered target reached role or key preparation"


def test_batch_can_report_existing_members_alongside_new_cluster(scenario):
    scenario.targets[target().eks_arn] = replace(
        target("a"), input_arn=target().eks_arn
    )
    result = batch.join_clusters(
        (scenario.request(cluster_id="gpu-a"), scenario.request("c")),
        runner_factory=lambda: scenario,
    )
    assert result["already_managed"] == ["gpu-a"]
    assert result["joined"] == ["hp-gpu-c"]
    assert scenario.state()["evidence"]["DISCOVERED"]["already_managed"] is True


def test_batch_stops_before_preflight_when_all_local_preparations_fail(scenario):
    scenario.failure = "kubeconfig"
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert scenario.state()["phase"] == scenario.state("c")["phase"] == "FAILED"
    assert scenario.driver_calls == []


def test_batch_local_ownership_conflict_is_explicitly_blocked(scenario):
    token = scenario.path.parent / "secure/hp-gpu-b.token"
    token.write_text("example-owned-test-token")
    token.chmod(0o600)
    before = token.read_bytes()
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert scenario.state()["phase"] == "BLOCKED_IDENTITY"
    assert scenario.state("c")["phase"] == "COMPLETED"
    assert token.read_bytes() == before


@pytest.mark.parametrize("boundary", ["node-keys:gpu-b", "activate-cluster:hp-gpu-b"])
def test_batch_interruption_in_prepare_or_commit_preserves_every_unfinished_record(
    scenario, boundary
):
    scenario.failure = boundary
    scenario.failure_error = KeyboardInterrupt("example interrupted batch boundary")
    with pytest.raises(KeyboardInterrupt, match="interrupted batch boundary"):
        scenario.batch()
    assert scenario.state()["phase"] == (
        "FAILED_AFTER_ACTIVATION" if boundary.startswith("activate") else "FAILED"
    )
    assert scenario.state("c")["phase"] == "FAILED"
    scenario.failure = None
    assert scenario.batch()["phase"] == "COMPLETED"
    assert scenario.cluster_states["hp-gpu-b"] == "ACTIVE"
    assert scenario.cluster_states["hp-gpu-c"] == "ACTIVE"


def test_batch_node_inventory_interruption_precedes_key_preparation(
    scenario, monkeypatch
):
    run = scenario.run

    def interrupted(arguments, **options):
        if "nodes" in arguments and "gpu-a" in arguments:
            raise KeyboardInterrupt("example node inventory cancellation")
        return run(arguments, **options)

    monkeypatch.setattr(scenario, "run", interrupted)
    with pytest.raises(KeyboardInterrupt, match="node inventory cancellation"):
        scenario.batch()
    assert scenario.driver_calls == []
    assert not any(event.startswith("node-keys:") for event in scenario.events), (
        "cancelled cross-cluster node inventory reached key provisioning"
    )


@pytest.mark.parametrize("phase", ["ROLLBACK_STARTED", "ROLLED_BACK"])
def test_batch_restarts_clean_early_rollback_with_new_attempt(scenario, phase):
    for suffix in ("b", "c"):
        _directory, path, state = load_join_state(scenario.request(suffix))
        state["phase"] = phase
        path.write_text(json.dumps(state))
    assert scenario.batch()["phase"] == "COMPLETED"
    for suffix in ("b", "c"):
        assert scenario.state(suffix)["attempt"] == 2
        assert (
            scenario.state_path(suffix).with_name("state.attempt-001.json").is_file()
        ), "resumed early rollback lost its first-attempt audit"


def test_batch_refuses_rollback_journal_that_already_crossed_activation(scenario):
    for suffix in ("b", "c"):
        _directory, path, state = load_join_state(scenario.request(suffix))
        state.update(phase="ROLLBACK_FAILED", completed_steps=["ACTIVATION_STARTED"])
        path.write_text(json.dumps(state))
    with pytest.raises(BootstrapError, match="rollback is forbidden"):
        scenario.batch()
    assert scenario.events == []
    assert (
        scenario.state()["phase"] == scenario.state("c")["phase"] == "ROLLBACK_FAILED"
    )


def test_stale_completed_batch_record_cannot_reuse_unowned_cluster_token(scenario):
    scenario.batch()
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["clusters"] = [
        item for item in document["spec"]["clusters"] if item["clusterId"] != "hp-gpu-b"
    ]
    scenario.path.write_text(yaml.safe_dump(document))
    before = list(scenario.driver_calls)
    with pytest.raises(BootstrapError, match="no matching join ownership"):
        scenario.batch()
    assert scenario.state()["attempt"] == 2
    assert scenario.state()["phase"] == "BLOCKED_IDENTITY"
    assert scenario.driver_calls == before
