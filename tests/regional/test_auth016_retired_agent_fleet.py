"""AUTH-016 binds the live Agent fleet, not the retired records of departed nodes.

Live 2026-10-16 (round 9 AUTH-016 a1, gpu-eks-cluster): a spot reclaim two days
earlier had removed ip-10-0-38-180 and ``workflow-reconcile
--retire-departed-agents`` had retired its Agent (REVOKED). The store keeps that
record, so the runner's baseline listed three Agents while the cluster had two
GPU Nodes. ``rotate-token`` reinstalled both Nodes and COMPLETED; the runner then
judged "rotation did not reinstall the complete bound node fleet" and the
evidence carried only ``IdentityAcceptanceError``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from tests.regional._security_token_rotation_support import (
    RotationWorld,
    consumer_snapshot,
)

DEPARTED = "node-gone"


def with_departed_agent(
    snapshot: dict[str, Any], *, lifecycle_state: str = "REVOKED"
) -> dict[str, Any]:
    """The snapshot plus the retired record a spot-reclaimed node leaves behind."""

    captured = datetime.fromisoformat(snapshot["captured_at"])
    snapshot["agents"][DEPARTED] = {
        "incarnation": "boot-" + DEPARTED,
        "generation": 10,
        "lifecycle": lifecycle_state,
        "last_seen_at": (captured - timedelta(days=2)).isoformat(),
        "lease_expires_at": (captured - timedelta(days=2)).isoformat(),
        "required_collectors": ["nvidia-kernel"],
    }
    snapshot["collectors"][DEPARTED + "/nvidia-kernel"] = {
        "ingested_at": (captured - timedelta(days=2)).isoformat(),
        "last_success_at": (captured - timedelta(days=2)).isoformat(),
        "errors": False,
    }
    return snapshot


def departed_world(tmp_path, monkeypatch) -> RotationWorld:
    world = RotationWorld(tmp_path, monkeypatch)
    snapshot = world.cpu_python
    monkeypatch.setattr(
        world, "cpu_python", lambda *args: with_departed_agent(snapshot(*args))
    )
    return world


def test_retired_departed_agent_does_not_fail_a_completed_rotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    world = departed_world(tmp_path, monkeypatch)
    result = lifecycle.run_rotation_acceptance(
        world.site, world.target, case_dir=tmp_path, retired_probe=lambda _token: 403
    )
    assert result["verdict"] == "PASS", (
        "a rotation that reinstalled every live node is complete"
    )
    assert all(result["checks"].values()), result["checks"]
    assert result["agent_fleet"] == {
        "bound": ["node-a", "node-b"],
        "retired": [DEPARTED],
    }, "the evidence names the bound consumers and the retired records apart"
    saved = json.loads((tmp_path / "auth016-details.json").read_text())
    assert saved["agent_fleet"] == result["agent_fleet"], (
        "the persisted details carry the fleet the verdict was judged on"
    )


def test_retired_agent_stays_retired_and_is_not_judged_as_a_consumer() -> None:
    stamp = datetime.fromisoformat("2026-10-16T06:22:56+00:00")
    before = with_departed_agent(consumer_snapshot(activated=False, stamp=stamp))
    after = with_departed_agent(consumer_snapshot(activated=True, stamp=stamp))
    assert (
        lifecycle.consumer_errors(
            before, after, cluster_id="cluster-a", after_withdrawal=stamp
        )
        == []
    ), "a retired record needs no post-withdrawal heartbeat or delivery"
    revived = with_departed_agent(
        consumer_snapshot(activated=True, stamp=stamp), lifecycle_state="ACTIVE"
    )
    assert lifecycle.consumer_errors(
        before, revived, cluster_id="cluster-a", after_withdrawal=stamp
    ) == [f"{DEPARTED}: retired Agent is ACTIVE after the rotation"], (
        "a rotation must never bring a retired identity back"
    )
    gone = consumer_snapshot(activated=True, stamp=stamp)
    assert lifecycle.consumer_errors(
        before, gone, cluster_id="cluster-a", after_withdrawal=stamp
    ) == ["credential consumer node inventory is incomplete or changed"], (
        "the retired record is still part of the unchanged inventory"
    )


@pytest.mark.parametrize("defect", ["draining", "expired", "only-retired"])
def test_agent_baseline_that_cannot_bind_refuses_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path, defect: str
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    snapshot = world.cpu_python

    def baseline(*args: Any) -> dict[str, Any]:
        value = snapshot(*args)
        agent = value["agents"]["node-b"]
        if defect == "draining":
            agent["lifecycle"] = "DRAINING"
        elif defect == "expired":
            agent["lease_expires_at"] = agent["last_seen_at"]
        else:
            for item in value["agents"].values():
                item["lifecycle"] = "REVOKED"
        return value

    monkeypatch.setattr(world, "cpu_python", baseline)
    with pytest.raises(
        lifecycle.IdentityAcceptanceError, match="complete Agent baseline"
    ) as raised:
        lifecycle.run_rotation_acceptance(
            world.site,
            world.target,
            case_dir=tmp_path,
            retired_probe=lambda _token: 403,
        )
    if defect != "only-retired":
        assert "node-b" in str(raised.value), "the refusal names the Agent"
    assert world.events == [] and world.thread is None, (
        "an Agent baseline the rotation cannot satisfy must not start it"
    )
    assert not (tmp_path / "auth016-production-intent.json").exists(), (
        "a refused baseline must not reserve a production rotation intent"
    )


def test_failed_postcondition_is_named_in_the_case_error_and_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    invoke = world.invoke

    def partial_wave(*args: Any) -> None:
        invoke(*args)
        world.state["steps"][lifecycle.STEP_NODES_ROLLED]["evidence"] = {
            "reinstalled_nodes": ["node-a"]
        }
        lifecycle.write_json_atomic(world.state_path, world.state)

    monkeypatch.setattr(lifecycle, "invoke_rotation", partial_wave)
    with pytest.raises(lifecycle.IdentityCaseFailure) as raised:
        lifecycle.run_rotation_acceptance(
            world.site,
            world.target,
            case_dir=tmp_path,
            retired_probe=lambda _token: 403,
        )
    reason = (
        "rotation did not reinstall the complete bound node fleet; "
        "bound, not reinstalled: ['node-b']"
    )
    assert str(raised.value) == (
        "production rotation acceptance did not complete: " + reason
    ), "the case error must say which postcondition failed"
    saved = json.loads((tmp_path / "auth016-details.json").read_text())
    assert saved["failure"] == "IdentityAcceptanceError", saved["failure"]
    assert saved["failure_reason"] == reason, saved.get("failure_reason")


def test_node_fleet_error_names_the_nodes_on_each_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    lifecycle.run_rotation_acceptance(
        world.site, world.target, case_dir=tmp_path, retired_probe=lambda _token: 403
    )
    state = world.state
    assert lifecycle.rotation_journal_errors(
        state,
        reference=state["reference"],
        cluster_id="cluster-a",
        old_digest=state["old_token_sha256"],
        node_names={"node-a", DEPARTED},
    ) == [
        "rotation did not reinstall the complete bound node fleet; "
        f"bound, not reinstalled: [{DEPARTED!r}]; reinstalled, not bound: ['node-b']"
    ], "both directions of a node-set mismatch are named"


@pytest.mark.parametrize("kind", ["command", "foreign"])
def test_failure_reason_never_carries_command_output_or_foreign_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path, kind: str
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)

    def failing(*_args: Any) -> None:
        if kind == "command":
            raise RegionalCommandFailed(
                4, "Traceback\nRuntimeError: " + world.new, ["gpu-fault-admin"]
            )
        raise RuntimeError("payload " + world.new)

    monkeypatch.setattr(lifecycle, "invoke_rotation", failing)
    with pytest.raises(lifecycle.IdentityCaseFailure) as raised:
        lifecycle.run_rotation_acceptance(
            world.site,
            world.target,
            case_dir=tmp_path,
            retired_probe=lambda _token: 403,
        )
    saved = (tmp_path / "auth016-details.json").read_text()
    reason = json.loads(saved)["failure_reason"]
    if kind == "command":
        assert reason.startswith("command failed (4): "), reason
        assert "RuntimeError" in reason, "the redacted summary keeps the class"
    else:
        assert reason == "RuntimeError", reason
    assert world.new not in saved and world.new not in str(raised.value), (
        "neither the evidence nor the case error may quote command output"
    )
