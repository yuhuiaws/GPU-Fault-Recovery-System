from __future__ import annotations

import copy

import pytest

from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.regional._security_token_rotation_support import RotationWorld


@pytest.mark.parametrize(
    "defect", ["deployments", "nodes", "retirement", "quiet", "loss", "timestamp"]
)
def test_production_journal_requires_each_claimed_postcondition(
    monkeypatch, tmp_path, defect
):
    world = RotationWorld(tmp_path, monkeypatch)
    auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    state = copy.deepcopy(world.state)
    steps = state["steps"]
    if defect == "deployments":
        steps[lifecycle.STEP_DATA_PLANE_ROLLED]["evidence"]["deployments"] = [
            "gpu-fault-cluster-executor"
        ]
    elif defect == "nodes":
        steps[lifecycle.STEP_NODES_ROLLED]["evidence"]["reinstalled_nodes"] = ["node-a"]
    elif defect == "retirement":
        steps[lifecycle.STEP_RETIRING_DROPPED]["evidence"]["retiring_token_dropped"] = (
            False
        )
    elif defect == "quiet":
        steps[lifecycle.STEP_ACCEPTED]["evidence"]["waited_seconds"] = 0
    elif defect == "loss":
        steps[lifecycle.STEP_ACCEPTED]["evidence"]["remote_command_losses"] = {
            "FAILED": 1
        }
    else:
        steps[lifecycle.STEP_ACCEPTED]["completed_at"] = "not-a-time"
    assert lifecycle.rotation_journal_errors(
        state,
        reference=state["reference"],
        cluster_id="cluster-a",
        old_digest=state["old_token_sha256"],
        node_names={"node-a", "node-b"},
    ), "completed rotation stages must prove their claimed postconditions"


def test_production_failure_is_never_replaced_by_a_runner_secret_restore(
    monkeypatch, tmp_path
):
    world = RotationWorld(tmp_path, monkeypatch)
    original = world.invoke

    def failed(*args):
        original(*args)
        raise RuntimeError("private diagnostic sentinel")

    monkeypatch.setattr(lifecycle, "invoke_rotation", failed)
    with pytest.raises(lifecycle.IdentityCaseFailure) as raised:
        auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    result = raised.value.details
    assert result["verdict"] == "FAIL" and result["cleanup_complete"] is False
    assert result["failure"] == "RuntimeError"
    assert "private diagnostic sentinel" not in str(result)
    assert world.token_file.read_text() == world.new
    assert world.events[-1][0] == "sampler-joined"


def test_sampler_exceptions_are_explicit_missing_evidence(monkeypatch, tmp_path):
    world = RotationWorld(tmp_path, monkeypatch)
    original = world.claim

    def sample(*args):
        if world.phase == "overlap":
            raise OSError("private diagnostic sentinel")
        return original(*args)

    monkeypatch.setattr(lifecycle, "claim", sample)
    result = auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["sampler_errors"] == ["OSError"]
    assert "private diagnostic sentinel" not in str(result)


def test_reusing_a_case_intent_does_not_start_another_rotation(monkeypatch, tmp_path):
    world = RotationWorld(tmp_path, monkeypatch)
    auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    before = list(world.events)
    with pytest.raises(
        lifecycle.IdentityAcceptanceError, match="already has an intent"
    ):
        auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    assert world.events == before
