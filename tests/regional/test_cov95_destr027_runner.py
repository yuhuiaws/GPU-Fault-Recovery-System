"""GF-REGIONAL-DESTR-027 driven end to end against a fake cluster.

The plan and the execution run through ``read_only_preflight`` and
``execute_case`` with fakes at the transport boundary only; the shared oracle
judges records built by its own test's builders and DESTR-001's contract reads
the fake probes' host snapshots. Each test names the refusal, the timeout or
the cleanup path it walks and what the evidence must then say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr027_two_idle_node_parallel_reset as destr027
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_parallel_domains import (
    CLUSTER,
    RESET_OWNER,
    World,
    profile,
    regional_settings,
    reset_hosts,
    reset_node_state,
)
from tests.regional.test_workload_domain_oracle import part, premise_snapshot

NODES = ("node-a", "node-b", "node-c")
TARGETS = ("node-a", "node-b")
BEFORE_INJECTION = "before DESTR-027 injection"
AFTER_CLEANUP = "after DESTR-027 cleanup"


def idle_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """Three idle GPU nodes of one group; the first two in name order fault."""

    settings = destr027.Settings(
        regional=regional_settings(tmp_path),
        host_probe_image="example.test/probe@sha256:" + "b" * 64,
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )
    world = World(
        destr027,
        tmp_path,
        monkeypatch,
        settings=settings,
        nodes=NODES,
        runtime_profile=profile(RESET_OWNER),
    )
    for node in NODES:
        world.states[node] = reset_node_state(node)
        world.baselines[node], world.afters[node] = reset_hosts(node)
    world.records_of(
        part("node-a", "node-a", start=0), part("node-b", "node-b", start=4)
    )
    return world


def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    world = idle_world(tmp_path, monkeypatch)
    preflight = world.plan()
    assert preflight["errors"] == [], preflight["errors"]
    assert preflight["targets"] == list(TARGETS), preflight["targets"]
    return world


# --------------------------------------------------------------------------- #
# The pass: two idle nodes reset in parallel
# --------------------------------------------------------------------------- #
def test_two_idle_nodes_reset_in_parallel_is_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.PASS, result
    assert result["errors"] == [], result["errors"]
    assert result["oracle"]["outcome"] == oracle.PARALLEL, result["oracle"]
    assert result["oracle"]["facts"]["breaker"]["counted_total"] == 2, result["oracle"]
    assert result["workflow_request_ids"] == {
        "node-a": "workflow-node-a",
        "node-b": "workflow-node-b",
    }
    assert result["incident_ids"] == {
        "node-a": "incident-node-a",
        "node-b": "incident-node-b",
    }
    assert set(result["markers"]) == set(TARGETS), result["markers"]
    assert result["provider_events"] == [], result
    assert result["provider_events_provisional"] is True, result
    for node in TARGETS:
        report = result["nodes"][node]
        assert report["quiesce_recovery"]["product_restoration_observed"], report
        assert report["probe_residuals"] == {"pod": False}, report
        assert report["final_node"]["uid"] == f"uid-{node}", report
    names = world.names()
    first_sampler = names.index("probe.start-reset-sampler:node-a")
    assert names.index("probe.snapshot:node-b") < first_sampler, names
    assert names.index(f"runtime.verify:{BEFORE_INJECTION}") < first_sampler, names
    assert max(
        names.index(f"probe.start-reset-sampler:{node}") for node in TARGETS
    ) < min(names.index(f"probe.write-xid46:{node}") for node in TARGETS), (
        "both samplers are armed before either XID is written"
    )
    assert names.index("oracle.read") > names.index("workflow.wait:node-b"), names
    for node in TARGETS:
        # The sampler is stopped by the post-reset read, never again in cleanup.
        assert names.count(f"probe.stop-reset-sampler:{node}") == 1, names
        assert names.index(f"probe.stop-reset-sampler:{node}") > names.index(
            f"probe.snapshot:{node}", names.index("oracle.read")
        ), names
    assert names[-1] == f"runtime.verify:{AFTER_CLEANUP}", names
    assert "probe.write-xid46:node-c" not in names, "the third node is never faulted"
    injection = world.evidence("injection.json")
    assert set(injection["markers"]) == {"a", "b"}, injection
    assert world.evidence("oracle-records.json")["cluster_id"] == CLUSTER
    host_after = world.evidence("node-b/host-after.json")
    assert host_after["ledger"][-1]["workflow_request_id"] == "workflow-node-b"


# --------------------------------------------------------------------------- #
# Preflight refusals
# --------------------------------------------------------------------------- #
def test_a_group_without_two_eligible_idle_nodes_names_every_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = idle_world(tmp_path, monkeypatch)
    world.nodes["node-a"]["unschedulable"] = True
    world.nodes["node-b"]["taints"] = [{"key": "gpu-fault.io/quarantined"}]
    world.nodes["node-c"]["gpu_allocatable"] = "0"
    world.stores["node-a"]["queue"] = {"depth": 3, "fault_backlog_depth": 2}
    world.tests_pass = False
    world.predecessor_valid = False

    preflight = world.plan()

    errors = preflight["errors"]
    assert preflight["targets"] == [], preflight
    assert errors[0].startswith(
        "node group gpu-group has no two eligible idle nodes now: "
    ), errors
    assert "node-a is cordoned" in errors[0], errors
    assert "node-b carries isolating taints" in errors[0], errors
    assert "node-c allocates no GPU" in errors[0], errors
    assert "processor queue is not empty" in errors, errors
    assert "focused regression tests failed" in errors, errors
    assert "DESTR-001 predecessor evidence is not PASS" in errors, errors
    assert not any("storm breaker" in item for item in errors), (
        "without targets the breaker premise is not judged"
    )
    details = world.evidence("plan.json")["details"]
    assert details["targets"] == [] and details["preflight_identity"]["targets"] == {}


def test_a_tripped_breaker_open_records_and_a_warned_profile_refuse_the_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = idle_world(tmp_path, monkeypatch)
    world.premise = premise_snapshot(state="TRIPPED")
    world.premise["open_incidents"] = [
        {"node_ids": ["node-b"], "incident_id": "incident-old", "state": "ESCALATED"}
    ]
    world.stores["node-b"]["profile"]["warnings"] = ["profile drift"]

    preflight = world.plan()

    errors = preflight["errors"]
    assert preflight["targets"] == list(TARGETS), preflight
    assert "node-b: runtime profile has warnings" in errors, errors
    assert any(
        "the storm breaker of gpu-cluster is TRIPPED" in item for item in errors
    ), errors
    assert "['node-b'] already have incident incident-old in ESCALATED" in errors, (
        errors
    )
    assert world.evidence("plan.json")["details"]["storm_premise"]["breaker_state"] == (
        "TRIPPED"
    )


def test_a_group_that_shrank_after_the_plan_records_not_run_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.gpu_nodes = world.gpu_nodes[:1]

    code, result = world.execute()

    assert code == 0, result
    assert (result["verdict"], result["reason"]) == (
        oracle.NOT_RUN,
        "CAPABILITY_ABSENT",
    )
    assert "has 1" in result["detail"], result["detail"]
    assert result["mutation_performed"] is False, result
    assert result["node_group_gate"]["minimum_gpu_nodes_in_one_node_group"] == 2
    assert not any(name.startswith("probe.") for name in world.names()), world.names()


def test_a_failed_preflight_at_execute_raises_before_any_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.predecessor_valid = False

    with pytest.raises(RegionalFixtureError, match="preflight failed: DESTR-001"):
        world.execute()

    assert not any(name.startswith("probe.") for name in world.names()), world.names()
    assert not (world.case_dir / f"{destr027.CASE_ID}.json").exists(), (
        "a refused execution writes no case result"
    )


def test_a_target_that_rebooted_since_the_plan_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.nodes["node-b"]["boot_id"] = "boot-node-b-2"
    world.stores["node-b"]["gpu_identity"]["source_boot_id"] = "boot-node-b-2"

    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        world.execute()

    assert not any(name.startswith("probe.") for name in world.names()), world.names()


# --------------------------------------------------------------------------- #
# Stops inside the execution
# --------------------------------------------------------------------------- #
def test_a_window_that_ends_after_the_baselines_writes_no_xid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.advance_at["probe.snapshot:node-b"] = 7200

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["error"] == (
        "RegionalFixtureError: maintenance window ended before the sampler start "
        "on node-a"
    ), result
    names = world.names()
    assert not any("write-xid" in name or "sampler" in name for name in names), names
    for node in TARGETS:
        recovery = result["nodes"][node]["quiesce_recovery"]
        assert recovery["operator_review_required"] is False, recovery
        assert f"probe.cleanup:{node}" in names, names


def test_a_workflow_wait_that_times_out_leaves_the_samplers_to_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures["workflow.wait:node-b"] = RegionalFixtureError(
        "workflow of node-b did not finish in 1500 s"
    )
    world.failures["probe.stop-reset-sampler:node-a"] = HostProbeTransportError(
        "node-a probe Pod is gone"
    )

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert "did not finish in 1500 s" in result["error"], result
    node_a, node_b = result["nodes"]["node-a"], result["nodes"]["node-b"]
    assert node_a["sampler_cleanup_error"] == (
        "HostProbeTransportError: node-a probe Pod is gone"
    ), node_a
    assert "sampler_cleanup_error" not in node_b, node_b
    assert world.names().count("probe.stop-reset-sampler:node-b") == 1
    assert node_b["quiesce_recovery"]["operator_review_required"] is True, (
        "an injected node whose workflow was never read needs the operator"
    )
    assert node_a["quiesce_recovery"] == {
        **node_b["quiesce_recovery"],
        "product_restoration_observed": False,
    }, "a SUCCEEDED workflow whose host was never re-read is not a restoration"
    assert (
        "probe.snapshot:node-a"
        not in world.names()[world.names().index("workflow.wait:node-a") :]
    ), "the post-reset host read never ran"


def test_a_workflow_the_store_cannot_name_is_inconclusive_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.states["node-b"]["workflow"]["request_id"] = ""

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.INCONCLUSIVE, result
    assert result["oracle"]["inconclusive"] == [
        "a target's XID never produced a workflow the store could name"
    ], result["oracle"]
    assert "oracle.read" not in world.names(), "no store read for an unnamed workflow"


def test_an_unreadable_store_makes_the_oracle_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures["oracle.read"] = RegionalFixtureError("API Pod exec timed out")

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.INCONCLUSIVE, result
    assert result["oracle"]["inconclusive"] == [
        "the store could not be read: API Pod exec timed out"
    ], result["oracle"]
    assert not (world.case_dir / "oracle-records.json").exists(), "nothing was read"


def test_a_reboot_that_ended_one_reset_proof_fails_that_node_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.on_injection["node-b"] = {"boot_id": "boot-node-b-rebooted"}

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    reboot = [item for item in result["errors"] if item.startswith("node-b: ")]
    assert len(reboot) == 1 and "boot id changed" in reboot[0], result["errors"]
    assert not any(item.startswith("node-a: ") for item in result["errors"]), result
    names = world.names()
    assert "probe.stop-reset-sampler:node-b" not in names, (
        "the reboot ended node-b's sampler; nobody asks the dead Pod"
    )
    assert result["nodes"]["node-b"]["host_after_skipped"].startswith("boot id"), (
        result["nodes"]["node-b"]
    )


def test_provider_mutation_and_control_plane_drift_fail_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.events = [{"event_name": "RebootInstances", "instance": "i-0"}]
    world.cpu_after = {"nodes": ["cpu-a", "cpu-b"]}

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["errors"] == [
        "provider mutation appeared during the two GPU resets",
        "control-plane EKS state differs from baseline",
    ], result["errors"]
    assert result["provider_events_provisional"] is False, result
    assert world.evidence("cpu-blast-after.json") == world.cpu_after


def failing_cleanup(world: World) -> dict[str, Any]:
    world.failures["probe.cleanup:node-b"] = HostProbeTransportError("exec refused")
    world.probe_residuals["node-a"] = {"pod": True}
    world.on_cleanup["node-a"] = {
        "unschedulable": True,
        "taints": [{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
        "ownership_annotations": {"gpu-fault.io/incident-id": "incident-other"},
    }
    world.failures[f"runtime.verify:{AFTER_CLEANUP}"] = RegionalFixtureError(
        "control-worker rolled during the case"
    )
    _code, result = world.execute()
    return result


def test_every_cleanup_failure_fails_an_otherwise_passing_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)

    result = failing_cleanup(world)

    assert result["verdict"] == oracle.FAIL, result
    node_a, node_b = result["nodes"]["node-a"], result["nodes"]["node-b"]
    assert node_a["probe_residuals"] == {"pod": True}, node_a
    assert node_b["probe_residuals"] == {"cleanup_error": True}, node_b
    assert node_b["probe_cleanup_error"] == "HostProbeTransportError: exec refused"
    assert node_a["verdict"] == "FAIL", node_a
    assert node_a["post_case_isolation"]["other_incident_owner"] == "incident-other"
    assert result["errors"] == [
        "node-a: target node is still isolated at the end of the case: owned by "
        "another incident incident-other; taint "
        "gpu-fault.io/quarantined:NoSchedule; unschedulable"
    ], result["errors"]
    assert result["runtime_identity_error"] == (
        "RegionalFixtureError: control-worker rolled during the case"
    ), result
