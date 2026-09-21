from __future__ import annotations

import copy
import sys
from datetime import datetime, timezone

import pytest

from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.regional._security_token_rotation_support import (
    RotationWorld,
    consumer_snapshot,
)


@pytest.mark.parametrize(
    "defect", ["none", "baseline", "overlap", "completed", "backlog", "sampler"]
)
def test_production_rotation_and_all_consumers_are_required(
    monkeypatch, tmp_path, defect
):
    world = RotationWorld(tmp_path, monkeypatch, defect=defect)
    if defect in {"baseline", "backlog"}:
        with pytest.raises(lifecycle.IdentityAcceptanceError):
            auth.run_auth016(world.site, world.target, case_dir=tmp_path)
        assert not any(event[0] == "production-rotation" for event in world.events), (
            "an invalid baseline or active backlog must stop before token rotation"
        )
        return
    result = auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    if defect == "sampler":
        assert result["cleanup_complete"] is False
    assert len(result["pod_consumers"]) == len(lifecycle.DEPLOYMENTS)
    assert world.token_file.read_text() == world.new, (
        "a committed credential must not be restored by a runner"
    )
    assert [event[0] for event in world.events] == [
        "production-rotation",
        "sampler-joined",
    ]
    assert world.new not in str(result) and world.old not in str(result)


@pytest.mark.parametrize("missing", list(lifecycle.ROTATION_STEPS))
def test_every_production_rotation_stage_is_part_of_the_evidence(
    monkeypatch, tmp_path, missing
):
    world = RotationWorld(tmp_path, monkeypatch)
    auth.run_auth016(world.site, world.target, case_dir=tmp_path)
    state = copy.deepcopy(world.state)
    state["steps"].pop(missing)
    assert lifecycle.rotation_journal_errors(
        state,
        reference=state["reference"],
        cluster_id="cluster-a",
        old_digest=state["old_token_sha256"],
        node_names={"node-a", "node-b"},
    ), "rotation evidence must include every required lifecycle stage"


@pytest.mark.parametrize(
    "defect",
    [
        "missing-agent",
        "old-agent",
        "missing-collector",
        "disabled-collector",
        "old-collector",
        "old-watcher",
    ],
)
def test_consumer_freshness_is_not_inferred_from_secret_update_or_executor_ready(
    defect,
):
    now = datetime.now(timezone.utc)
    before = consumer_snapshot(activated=False, stamp=now)
    after = consumer_snapshot(activated=True, stamp=now)
    if defect == "missing-agent":
        after["agents"].pop("node-b")
    elif defect == "old-agent":
        # Never re-registered after the wave: the generation did not advance.
        after["agents"]["node-b"]["generation"] = before["agents"]["node-b"][
            "generation"
        ]
    elif defect == "missing-collector":
        after["collectors"].pop("node-b/nvidia-kernel")
    elif defect == "disabled-collector":
        after["agents"]["node-b"]["required_collectors"] = []
    elif defect == "old-collector":
        after["collectors"]["node-b/nvidia-kernel"]["ingested_at"] = now.isoformat()
    else:
        after["watcher"]["observed_at"] = now.isoformat()
    assert lifecycle.consumer_errors(
        before, after, cluster_id="cluster-a", after_withdrawal=now
    ), "stale or missing consumers must not be accepted as rotated"


def test_runner_invokes_the_public_production_lifecycle_not_secret_mutations(
    monkeypatch, tmp_path
):
    invoke = lifecycle.invoke_rotation
    world = RotationWorld(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        lifecycle, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    monkeypatch.setattr(
        lifecycle, "effective_environment", lambda _site: {"PATH": "/unit"}
    )
    command = [
        sys.executable,
        "-m",
        "gpu_fault.admin.cli",
        "rotate-token",
        "--state-dir",
        str(world.site_file.parent),
        "--gpu-cluster-arn",
        world.target.eks_cluster_arn,
        "--reference",
        "unit-change",
    ]
    invoke(world.site, world.target, "unit-change")
    assert calls[0][0] == command
    assert calls[0][1]["timeout"] == lifecycle.ROTATION_TIMEOUT_SECONDS


def test_runner_prefers_the_state_dirs_own_deploy_host_cli(monkeypatch, tmp_path):
    # Live 2026-09-20 (AUTH-016 a1): the checkout's module CLI refused with
    # "<state-dir> has its own deploy-host CLI; run <state-dir>/deployer-venv/bin/
    # gpu-fault-admin rotate-token ... instead" (rc 2) before any journal or log
    # existed. A managed state dir that carries its own deployer venv must be
    # driven through that binary; the module form stays for sites without one.
    invoke = lifecycle.invoke_rotation  # RotationWorld replaces it with a fake
    world = RotationWorld(tmp_path, monkeypatch)
    admin = world.site_file.parent / "deployer-venv" / "bin" / "gpu-fault-admin"
    admin.parent.mkdir(parents=True)
    admin.write_text("#!/bin/sh\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        lifecycle, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    monkeypatch.setattr(
        lifecycle, "effective_environment", lambda _site: {"PATH": "/unit"}
    )
    invoke(world.site, world.target, "unit-change")
    assert calls[0][0] == [
        str(admin),
        "rotate-token",
        "--state-dir",
        str(world.site_file.parent),
        "--gpu-cluster-arn",
        world.target.eks_cluster_arn,
        "--reference",
        "unit-change",
    ]
    assert calls[0][1]["timeout"] == lifecycle.ROTATION_TIMEOUT_SECONDS


def test_agent_activation_is_a_generation_advance_not_a_new_incarnation():
    # Live 2026-09-20 (AUTH-016 a2): the production rotation re-installed every
    # node (agent stopped 03:41:10, started 03:41:20, new env file) and the fleet
    # re-registered each agent, yet every incarnation stayed identical because it
    # hashes cluster/node/instance/boot_id. Requiring a new incarnation demanded a
    # reboot the rotation never performs; the proof is the generation advance plus
    # heartbeats after the retiring token was dropped (an old-token heartbeat is 403).
    now = datetime.now(timezone.utc)
    before = consumer_snapshot(activated=False, stamp=now)
    after = consumer_snapshot(activated=True, stamp=now)
    assert all(
        after["agents"][node]["incarnation"] == before["agents"][node]["incarnation"]
        for node in before["agents"]
    ), "a token rotation must not change any consumer incarnation"
    assert (
        lifecycle.consumer_errors(
            before, after, cluster_id="cluster-a", after_withdrawal=now
        )
        == []
    )
    rebooted = copy.deepcopy(after)
    rebooted["agents"]["node-a"]["incarnation"] = "boot-node-a-after-reboot"
    assert (
        lifecycle.consumer_errors(
            before, rebooted, cluster_id="cluster-a", after_withdrawal=now
        )
        == []
    ), "a reboot during the window changes the incarnation and is not a defect"
    missing = copy.deepcopy(after)
    missing["agents"]["node-a"]["incarnation"] = ""
    assert lifecycle.consumer_errors(
        before, missing, cluster_id="cluster-a", after_withdrawal=now
    ), "an agent without an incarnation is not a proven consumer"


def test_registry_commit_check_reads_the_models_lifecycle_field():
    # Live 2026-09-20 (AUTH-016 a2): the durable registry head carried
    # token_sha256 == new digest, no retiring token, enabled=True and
    # lifecycle_state=ACTIVE, yet target_registry_committed was False because the
    # runner read ``membership_state`` -- a name that existed only in test fakes
    # (the NET-007 lesson: pin verdict fields against the model).
    from gpu_fault.regional import RegionalClusterRegistration

    assert "lifecycle_state" in RegionalClusterRegistration.model_fields
    assert "membership_state" not in RegionalClusterRegistration.model_fields
    entry = RegionalClusterRegistration(
        cluster_id="cluster-a",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/a",
        token_sha256="b" * 64,
        allowed_namespaces=["gpu-fault-system"],
        agent_endpoint_allowed_cidrs=["10.90.0.0/16"],
    ).model_dump(mode="json")
    assert lifecycle.registry_entry_committed(entry, "b" * 64) is True
    assert lifecycle.registry_entry_committed(entry, "c" * 64) is False
    retiring = {**entry, "retiring_token_sha256": "a" * 64}
    assert lifecycle.registry_entry_committed(retiring, "b" * 64) is False
    disabled = {**entry, "enabled": False}
    assert lifecycle.registry_entry_committed(disabled, "b" * 64) is False
    draining = {**entry, "lifecycle_state": "DRAINING"}
    assert lifecycle.registry_entry_committed(draining, "b" * 64) is False
