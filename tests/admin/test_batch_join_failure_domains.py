"""Real batch/commit paths against an offline, stateful publication transport."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.admin import cluster_join_failure_domains as barrier
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.failure_domain_map import FAILURE_DOMAIN_MAP_ANNOTATION
from tests.admin._cov95_join_support import JoinScenario


def test_batch_publishes_all_verified_pending_members_once_before_activation(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    result = scenario.batch()
    assert result["joined"] == ["hp-gpu-b", "hp-gpu-c"]
    assert scenario.events.count("map-nodes") == 3
    assert scenario.events.count("map-apply") == 1
    assert scenario.events.count("map-patch") == 1
    assert scenario.events.count("map-rollout") == 1
    activation = scenario.events.index("activate-cluster:hp-gpu-b")
    assert scenario.events.index("verify") < scenario.events.index("map-apply")
    assert scenario.events.index("map-rollout") < activation
    assert (
        "map-readback"
        in scenario.events[scenario.events.index("map-rollout") + 1 : activation]
    )
    manifest = scenario.map_kubectl.manifest
    assert manifest is not None
    assert json.loads(manifest["data"]["failure-domains.json"]) == {
        "gpu-a": {"node-a": "group-a"},
        "hp-gpu-b": {"node-b": "group-b"},
        "hp-gpu-c": {"node-c": "group-c"},
    }
    records = [
        scenario.state(suffix)["evidence"][barrier.PUBLICATION_READY]
        for suffix in ("b", "c")
    ]
    assert records[0]["publication_id"] == records[1]["publication_id"]
    assert records[0]["node_uids"]["hp-gpu-c"] == {"node-c": "uid-node-c"}
    assert all(record["worker_uid"] == "worker-uid" for record in records), (
        "every member receipt must retain the worker UID proved by the rollout"
    )


def test_failed_convergence_keeps_every_member_pending_and_can_resume(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    scenario.failure = "map-rollout"
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert scenario.cluster_states == {
        "gpu-a": "ACTIVE",
        "hp-gpu-b": "PENDING",
        "hp-gpu-c": "PENDING",
    }
    for suffix in ("b", "c"):
        state = scenario.state(suffix)
        assert barrier.PUBLICATION_STARTED in state["evidence"]
        assert barrier.PUBLICATION_READY not in state["evidence"]
        assert "ACTIVATION_STARTED" not in state["completed_steps"]
    scenario.failure = None
    result = scenario.batch()
    assert result["joined"] == ["hp-gpu-b", "hp-gpu-c"]
    assert scenario.events.count("map-nodes") == 6
    assert scenario.events.count("map-rollout") == 2


def test_individual_fail_forward_retry_rechecks_publication_without_republishing(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    scenario.failure = "activate-cluster:hp-gpu-c"
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert scenario.state("b")["phase"] == "COMPLETED"
    assert scenario.state("c")["phase"] == "FAILED_AFTER_ACTIVATION"
    old = scenario.state("c")
    old["evidence"]["VERIFIED"]["verified_at"] = (
        datetime.now(UTC) - timedelta(hours=1)
    ).isoformat()
    scenario.state_path("c").write_text(json.dumps(old))
    before = list(scenario.events)
    scenario.failure = None
    assert scenario.join("c")["phase"] == "COMPLETED"
    assert scenario.events.count("map-nodes") == 3
    assert scenario.events.count("map-rollout") == 1
    assert "map-readback" in scenario.events[len(before) :]
    assert scenario.cluster_states["hp-gpu-c"] == "ACTIVE"


@pytest.mark.parametrize("drift", ["worker", "configmap", "replicas", "digest"])
def test_individual_retry_never_uses_a_stale_convergence_receipt(
    tmp_path, monkeypatch, drift
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    scenario.failure = "activate-cluster:hp-gpu-c"
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    scenario.failure = None
    if drift == "worker":
        scenario.map_kubectl.worker_uid = "replacement-worker"
    elif drift == "configmap":
        scenario.map_kubectl.configmap_uid = "replacement-map"
    elif drift == "replicas":
        scenario.map_kubectl.worker_ready = False
    else:
        scenario.map_kubectl.worker_annotations[FAILURE_DOMAIN_MAP_ANNOTATION] = (
            "0" * 64
        )
    with pytest.raises(BootstrapError, match="failure-domain"):
        scenario.join("c")
    assert scenario.cluster_states["hp-gpu-c"] == "PENDING"
    assert scenario.events.count("map-rollout") == 1


def test_registry_generation_change_during_publication_blocks_activation(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    event = scenario.event

    def changed(name):
        event(name)
        if name == "map-rollout":
            scenario.generation += 1

    monkeypatch.setattr(scenario, "event", changed)
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert not any(name.startswith("activate-cluster:") for name in scenario.events), (
        "registry generation drift must prevent every pending activation"
    )
    for suffix in ("b", "c"):
        assert barrier.PUBLICATION_READY not in scenario.state(suffix)["evidence"]


def test_node_uid_change_before_shared_publication_never_applies_the_map(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    run = scenario.map_command

    def changed(arguments, **options):
        result = run(arguments, **options)
        if "nodes" in arguments:
            document = json.loads(result.stdout)
            document["items"][0]["metadata"]["uid"] = "replaced-node"
            result.stdout = json.dumps(document)
        return result

    from gpu_fault.admin import failure_domain_map

    monkeypatch.setattr(failure_domain_map, "run_command", changed)
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert "map-apply" not in scenario.events
    assert not any(name.startswith("activate-cluster:") for name in scenario.events), (
        "a replaced node must not reach cluster activation"
    )


def test_publication_ready_checkpoint_loss_requires_convergence_again(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    complete = barrier.complete_step
    failed = False

    def interrupted(path, state, step, evidence=None):
        nonlocal failed
        if step == barrier.PUBLICATION_READY and not failed:
            failed = True
            raise OSError("modeled publication checkpoint failure")
        complete(path, state, step, evidence)

    monkeypatch.setattr(barrier, "complete_step", interrupted)
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    assert not any(name.startswith("activate-cluster:") for name in scenario.events), (
        "a missing publication checkpoint cannot authorize activation"
    )
    assert scenario.events.count("map-rollout") == 1
    monkeypatch.setattr(barrier, "complete_step", complete)
    assert scenario.batch()["phase"] == "COMPLETED"
    assert scenario.events.count("map-rollout") == 2
