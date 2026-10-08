from __future__ import annotations

import copy

import pytest

from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.regional._security_token_rotation_support import RotationWorld


@pytest.mark.parametrize(
    "defect",
    [
        "deployments",
        "control-plane",
        "control-plane-secret",
        "nodes",
        "retirement",
        "quiet",
        "loss",
        "timestamp",
    ],
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
    elif defect == "control-plane":
        steps[lifecycle.STEP_CONTROL_PLANE_ROLLED]["evidence"]["deployments"] = [
            "gpu-fault-api-ha"
        ]
    elif defect == "control-plane-secret":
        steps[lifecycle.STEP_CONTROL_PLANE_ROLLED]["evidence"][
            "registry_secret_rewritten"
        ] = False
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


def test_claim_transport_failures_are_named_errors_with_the_active_step(
    monkeypatch, tmp_path
):
    """Live 2026-10-08: one claim came back with status None during the
    control-plane roll and the sampler kept only the None. The probe's cause
    (transport error, kind, latency) and the rotation step active at that
    instant are now the evidence, the failure is a named sampler error, and
    the verdict stays fail-closed."""
    world = RotationWorld(tmp_path, monkeypatch)
    original_claim = world.claim
    original_load = lifecycle.load_rotation_state

    def failing_claim(*args):
        if world.phase == "new-consumers":
            return {
                "status": None,
                "transport_error": "URLError",
                "transport_failure_kind": "network",
                "latency_seconds": 3.25,
            }
        return original_claim(*args)

    rolling_reads: list[str] = []

    def rolling_state(path):
        # Only the sampler's read inside the new-consumers tick sees the
        # control-plane roll in progress; the runner's own journal read after
        # the rotation sees the completed journal.
        state = original_load(path)
        if state and world.phase == "new-consumers" and not rolling_reads:
            rolling_reads.append("sampler")
            state = copy.deepcopy(state)
            rolled = state["steps"].pop(lifecycle.STEP_CONTROL_PLANE_ROLLED)
            state["started_steps"] = {
                lifecycle.STEP_CONTROL_PLANE_ROLLED: rolled["completed_at"]
            }
        return state

    monkeypatch.setattr(lifecycle, "claim", failing_claim)
    monkeypatch.setattr(lifecycle, "load_rotation_state", rolling_state)

    result = auth.run_auth016(world.site, world.target, case_dir=tmp_path)

    assert result["verdict"] == "FAIL", result["checks"]
    assert result["checks"]["consumer_claim_continuity"] is False
    assert result["sampler_errors"] == [
        "claim transport error URLError (network, 3.2s) during CONTROL_PLANE_ROLLED"
    ], result["sampler_errors"]
    (failed,) = result["claim_failures"]
    assert failed["phase"] == "new-consumers" and failed["status"] is None, failed
    assert failed["active_step"] == lifecycle.STEP_CONTROL_PLANE_ROLLED, failed
    assert failed["transport_error"] == "URLError", failed
    assert failed["transport_failure_kind"] == "network", failed
    assert failed["latency_seconds"] == 3.25, failed
    good = [item for item in result["samples"] if item["status"] == 200]
    assert good and all("transport_error" not in item for item in good), (
        "healthy samples carry no transport fields"
    )
    assert all("active_step" in item for item in result["samples"]), (
        "every sample names the rotation step it met (None outside the rotation)"
    )
    assert set(result["step_timeline"]) == set(lifecycle.ROTATION_STEPS), (
        "the details carry the rotation's step timeline for correlation"
    )


def test_a_non_200_claim_is_a_named_failure_but_not_a_transport_error(
    monkeypatch, tmp_path
):
    world = RotationWorld(tmp_path, monkeypatch, defect="overlap")

    result = auth.run_auth016(world.site, world.target, case_dir=tmp_path)

    assert result["verdict"] == "FAIL"
    assert result["sampler_errors"] == [], (
        "an HTTP status is a measured answer, not a sampler transport error"
    )
    (failed,) = result["claim_failures"]
    assert failed["status"] == 403 and failed["phase"] == "overlap", failed
    assert lifecycle.claim_failure_error(failed) == (
        "claim returned 403 with command_count 0 during overlap"
    )
