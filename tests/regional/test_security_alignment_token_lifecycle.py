from __future__ import annotations

import copy
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
        after["agents"]["node-b"]["incarnation"] = before["agents"]["node-b"][
            "incarnation"
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
        lifecycle.sys.executable,
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
