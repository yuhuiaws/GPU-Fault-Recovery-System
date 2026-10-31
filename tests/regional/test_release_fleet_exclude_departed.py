"""The fleet probe's ``exclude-departed``: a departed node leaves the rollout record.

The record is the control plane's (``FleetRegistry``) and an older control
plane -- the one a rollback restores -- must still parse it, so the operation
adds no status and no field: the node leaves the record's ``nodes`` and
``waves`` and the status is recomputed, which ends the record ``SUCCEEDED``
over the nodes that remain instead of ``FAILED`` for a node that left. The
shipped probe program runs against a real registry on the in-memory store.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.fleet import FleetDeployment, FleetDeploymentRequest
from tests.regional._node_churn_site import OLD_TEMPLATE, ChurnSite, node_name
from tests.regional.test_release_node_config_waves import AGENT_VERSION, POLICY, PROFILE
from tests.regional.test_release_node_image_targets import ARTIFACT, BUNDLE, CONFIG

NODES = [node_name(index) for index in range(1, 5)]
TARGET = SimpleNamespace(cluster_id="gpu-a")
NEW_TEMPLATE = "4" * 64


def site_with_rollout(*, first_wave: int = 1) -> tuple[ChurnSite, str]:
    """Four nodes on the old template, one rollout of the new one, not started."""

    site = ChurnSite(TARGET, NODES, template=OLD_TEMPLATE)
    deployment = site.registry.create_deployment(
        FleetDeploymentRequest(
            deployment_id="release-upgrade-candidate-0001",
            cluster_id="gpu-a",
            node_ids=NODES,
            desired_agent_version=AGENT_VERSION,
            desired_artifact_sha256=ARTIFACT,
            desired_compatibility_digest=ARTIFACT,
            desired_bundle_sha256=BUNDLE,
            desired_template_sha256=NEW_TEMPLATE,
            desired_policy_version=POLICY,
            desired_runtime_profile_version=PROFILE,
            desired_config_digest=CONFIG,
            max_unavailable=4,
            first_wave_max_unavailable=first_wave,
            max_unavailable_per_failure_domain=4,
        )
    )
    return site, deployment.deployment_id


def converge(site: ChurnSite, name: str) -> None:
    site.agents[name] = NEW_TEMPLATE
    site.beat(name)


def go_silent(site: ChurnSite, *names: str) -> None:
    for name in names:
        site.reclaim(name)
    site.tick(120)


def exclude(site: ChurnSite, deployment_id: str, *nodes: str) -> dict[str, Any]:
    return site.run_fleet_probe(
        {
            "operation": "exclude-departed",
            "deployment_id": deployment_id,
            "node_ids": list(nodes),
        }
    )


def record(site: ChurnSite, deployment_id: str) -> FleetDeployment:
    return site.registry.store.get_fleet_deployment(deployment_id)


def test_a_departed_node_in_the_active_wave_leaves_the_record() -> None:
    site, deployment_id = site_with_rollout()
    site.registry.start_next_wave(deployment_id)  # canary NODES[0]: INSTALLING
    converge(site, NODES[0])
    site.registry.start_next_wave(deployment_id)  # NODES[1:]: INSTALLING
    for name in (NODES[1], NODES[3]):
        converge(site, name)
    go_silent(site, NODES[2])

    result = exclude(site, deployment_id, NODES[2])

    assert (result["excluded"], result["absent"], result["kept"]) == (
        [NODES[2]],
        [],
        [],
    )
    stored = record(site, deployment_id)
    assert stored.status.value == "SUCCEEDED", stored
    assert sorted(item.node_id for item in stored.nodes) == sorted(
        set(NODES) - {NODES[2]}
    )
    assert all(NODES[2] not in wave for wave in stored.waves), stored.waves
    assert result["deployment"]["status"] == "SUCCEEDED"
    # The rewritten record keeps the shape every control plane parses.
    assert FleetDeployment.model_validate_json(stored.model_dump_json()) == stored


def test_a_ready_node_stays_and_a_repeat_is_idempotent() -> None:
    site, deployment_id = site_with_rollout()
    site.registry.start_next_wave(deployment_id)
    converge(site, NODES[0])
    go_silent(site, NODES[0], NODES[3])

    first = exclude(site, deployment_id, NODES[0], NODES[3])
    again = exclude(site, deployment_id, NODES[3])

    assert (first["excluded"], first["kept"]) == ([NODES[3]], [NODES[0]])
    assert (again["excluded"], again["absent"]) == ([], [NODES[3]])
    statuses = {
        item.node_id: item.status.value for item in record(site, deployment_id).nodes
    }
    assert statuses[NODES[0]] == "READY", "a node that converged before it left"


def test_a_departed_canary_leaves_a_one_node_canary_behind() -> None:
    site, deployment_id = site_with_rollout()
    site.registry.start_next_wave(deployment_id)  # canary INSTALLING
    go_silent(site, NODES[0])

    exclude(site, deployment_id, NODES[0])

    stored = record(site, deployment_id)
    assert stored.waves == [[NODES[1]], sorted(NODES[2:])], stored.waves
    assert stored.status.value == "PLANNED"
    lease = site.registry.start_next_wave(deployment_id)
    assert lease.node_ids == [NODES[1]], "the next node goes alone"


def test_a_started_wave_is_not_split() -> None:
    site, deployment_id = site_with_rollout(first_wave=4)
    site.registry.start_next_wave(deployment_id)  # all four INSTALLING
    go_silent(site, NODES[0])

    exclude(site, deployment_id, NODES[0])

    assert record(site, deployment_id).waves == [sorted(NODES[1:])]


def test_a_node_whose_agent_still_holds_a_live_lease_is_refused() -> None:
    site, deployment_id = site_with_rollout()
    site.registry.start_next_wave(deployment_id)

    with pytest.raises(ValueError, match="live Agent lease and are not departed"):
        exclude(site, deployment_id, NODES[0])

    statuses = {
        item.node_id: item.status.value for item in record(site, deployment_id).nodes
    }
    assert statuses[NODES[0]] == "INSTALLING", "nothing was written"


def test_a_record_every_node_of_which_departed_is_refused() -> None:
    site, deployment_id = site_with_rollout()
    go_silent(site, *NODES)

    with pytest.raises(ValueError, match="every node of deployment"):
        exclude(site, deployment_id, *NODES)


def test_a_terminal_record_is_refused() -> None:
    site, deployment_id = site_with_rollout()
    site.registry.cancel_deployment(deployment_id, reason="superseded")
    go_silent(site, NODES[0])

    with pytest.raises(ValueError, match="already terminal"):
        exclude(site, deployment_id, NODES[0])


def test_the_lease_is_judged_on_the_registry_clock() -> None:
    """A lease that lapsed on the Pod clock, whatever the deploy host thinks."""

    site, deployment_id = site_with_rollout()
    site.registry.start_next_wave(deployment_id)
    site.reclaim(NODES[0])
    site.clock += timedelta(seconds=91)

    result = exclude(site, deployment_id, NODES[0])

    assert result["excluded"] == [NODES[0]]
