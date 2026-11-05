"""GF-REGIONAL-DESTR-029 driven end to end against a fake cluster.

A two-node job (XID 79 on its master) and an idle node (XID 46) of one node
group recover at once: the plan and the execution run through
``read_only_preflight`` and ``execute_case`` with fakes at the transport
boundary only. The shared oracle judges records built by its own test's
builders; the job reboot contract and DESTR-001's reset contract judge the two
faulted nodes.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from gpu_fault.remediation_storm import StormEntry
from scripts.e2e.regional import run_destr029_job_and_idle_parallel as destr029
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_parallel_domains import (
    REBOOT_OWNER,
    RESET_OWNER,
    WORKLOAD_OWNERS,
    World,
    bind_job,
    job_host,
    job_node_state,
    profile,
    regional_settings,
    reset_hosts,
    reset_node_state,
)
from tests.regional.test_workload_domain_oracle import part, premise_snapshot

AFTER_CLEANUP = "after DESTR-029 cleanup"
MASTER, WORKER, IDLE = "node-a", "node-b", "node-c"


def mixed_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    settings = destr029.Settings(
        regional=regional_settings(tmp_path),
        host_probe_image="example.test/probe@sha256:" + "d" * 64,
        site_file=site,
        manifest=destr029.TRAINING_MANIFEST,
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )
    world = World(
        destr029,
        tmp_path,
        monkeypatch,
        settings=settings,
        nodes=(MASTER, WORKER, IDLE),
        runtime_profile=profile(*WORKLOAD_OWNERS, REBOOT_OWNER, RESET_OWNER),
    )
    plan = destr029.layout((MASTER, WORKER, IDLE), world.run_id())
    world.states[MASTER] = job_node_state("job")
    world.baselines[MASTER] = job_host()
    world.on_injection[MASTER] = {"boot_id": f"boot-{MASTER}-2"}
    world.states[IDLE] = reset_node_state("idle")
    world.baselines[IDLE], world.afters[IDLE] = reset_hosts("idle")
    # The job's RESTART_NODE and the idle RESET_GPU run in the same seconds.
    job = part("job", MASTER, start=20, job=plan.job_id, peers=(WORKER,))
    world.records_of(bind_job(job, plan.workload_id), part("idle", IDLE, start=6))
    return world


def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    world = mixed_world(tmp_path, monkeypatch)
    preflight = world.plan()
    assert preflight["errors"] == [], preflight["errors"]
    return world


def job_id(world: World) -> str:
    return f"{world.run_id()}-job"


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
def test_a_job_reboot_and_an_idle_reset_recover_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.PASS, result
    assert result["errors"] == [], result["errors"]
    assert result["layout"]["master"] == MASTER and result["layout"]["idle"] == IDLE
    assert result["oracle"]["facts"]["breaker"]["counted_domains"] == [
        "incident:incident-idle",
        f"job:{job_id(world)}",
    ], result["oracle"]["facts"]["breaker"]
    assert result["workflow_request_ids"] == {
        "job": "workflow-job",
        "idle": "workflow-idle",
    }
    assert result["incident_ids"] == {"job": "incident-job", "idle": "incident-idle"}
    names = world.names()
    assert "probe.write-xid79:node-a" in names and "probe.write-xid46:node-c" in names
    assert not any(name.endswith(WORKER) and "write-xid" in name for name in names), (
        "the job's healthy worker is never faulted"
    )
    assert names.index(f"job.running:{job_id(world)}") < names.index(
        "probe.write-xid79:node-a"
    ), "the job runs before its master is faulted"
    assert "probe.stop-reset-sampler:node-a" not in names, "the reboot ended it"
    assert names.count("probe.stop-reset-sampler:node-c") == 1, names
    assert names.count(f"job.delete:{job_id(world)}") == 1, names
    assert world.evidence("prewarm.json") == {"cached_nodes": [MASTER, WORKER]}
    settings = world.workloads[job_id(world)].settings
    assert (settings.expected_pods, settings.expected_gpu_count) == (2, 16)
    rendered = yaml.safe_load(settings.manifest.read_text(encoding="utf-8"))
    replicas = rendered["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["template"]["spec"]["nodeName"] == MASTER, replicas
    assert replicas["Worker"]["template"]["spec"]["nodeName"] == WORKER, replicas
    assert result["target_reports"] == {
        "job": {"probe_residuals": {"pod": False}},
        "idle": {"probe_residuals": {"pod": False}},
    }, result["target_reports"]
    assert world.evidence("idle/host-after.json")["ledger"][-1][
        "workflow_request_id"
    ] == ("workflow-idle")


# --------------------------------------------------------------------------- #
# Preflight refusals
# --------------------------------------------------------------------------- #
def test_a_group_without_three_eligible_idle_nodes_names_every_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = mixed_world(tmp_path, monkeypatch)
    world.stores[IDLE]["event"] = {"xid": 46}
    world.stores[MASTER]["queue"] = {"depth": 9}
    world.tests_pass = False
    world.predecessor_valid = False

    preflight = world.plan()

    errors = preflight["errors"]
    assert (preflight["targets"], preflight["layout"]) == ([], None), preflight
    assert errors[0].startswith("node group gpu-group has no 3 eligible idle nodes"), (
        errors
    )
    assert "node-c had an XID in the last 600s" in errors[0], errors
    assert "processor queue is not empty" in errors, errors
    assert "focused regression tests failed" in errors, errors
    assert "DESTR-002 predecessor evidence is not PASS" in errors, errors


def test_an_unowned_reset_a_counting_breaker_and_open_records_refuse_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = mixed_world(tmp_path, monkeypatch)
    world.stores[IDLE]["profile"] = profile(*WORKLOAD_OWNERS, REBOOT_OWNER)
    world.premise = premise_snapshot(
        StormEntry(
            node_id="node-z",
            domain="incident:other-case",
            workflow_id="w-other",
            admitted_at=world.clock.now() - timedelta(seconds=30),
        )
    )
    world.premise["open_incidents"] = [
        {"node_ids": [WORKER], "incident_id": "incident-old", "state": "QUARANTINED"}
    ]

    preflight = world.plan()

    errors = preflight["errors"]
    assert preflight["layout"]["idle"] == IDLE, preflight
    assert "node-c: runtime profile has no gpuReset capability" in errors, errors
    assert any(
        "with the case's ['node-a', 'node-c'] that is 3 nodes" in item
        for item in errors
    ), "the job's master and the idle node are added to what the breaker counts"
    assert "['node-b'] already have incident incident-old in QUARANTINED" in errors


def test_a_group_that_shrank_after_the_plan_records_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.gpu_nodes = world.gpu_nodes[1:]

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.NOT_RUN, result
    assert result["mutation_performed"] is False, result
    assert world.workloads == {}, "a NOT_RUN creates no job fixture"


def test_a_failed_preflight_at_execute_raises_before_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.predecessor_valid = False

    with pytest.raises(RegionalFixtureError, match="DESTR-002 predecessor"):
        world.execute()

    assert world.workloads == {}, "no job before the preflight passes again"


def test_a_target_whose_agent_changed_since_the_plan_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.stores[WORKER]["agent"]["generation"] = 9

    with pytest.raises(RegionalFixtureError, match="DESTR-029 plan identity drifted"):
        world.execute()


def test_a_rendered_job_named_otherwise_is_refused_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    monkeypatch.setattr(
        destr029.job_reboot_contract,
        "name_rendered_job",
        lambda path, value: "gpu-fault-two-node-replace",
    )

    with pytest.raises(RegionalFixtureError, match="the plan expects"):
        world.execute()

    assert world.workloads == {}, "the misnamed job is never handed to the fixture"


# --------------------------------------------------------------------------- #
# Stops inside the execution and the cleanup
# --------------------------------------------------------------------------- #
def test_a_window_that_ends_during_the_prewarm_submits_no_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.advance_at["prewarm.create"] = 7200

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["error"] == (
        f"RegionalFixtureError: maintenance window ended before submit {job_id(world)}"
    ), result
    names = world.names()
    assert not any(name.startswith("job.") for name in names), names
    assert "job_delete" not in result, "a job never submitted is not deleted"
    assert {"probe.cleanup:node-a", "probe.cleanup:node-c", "prewarm.cleanup"} <= set(
        names
    ), names


def test_a_wait_that_times_out_leaves_samplers_job_and_prewarm_to_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures[f"workflow.wait:{IDLE}"] = RegionalFixtureError("1800 s elapsed")
    world.failures[f"probe.stop-reset-sampler:{MASTER}"] = HostProbeTransportError(
        "node-a is rebooting"
    )
    world.failures[f"job.delete:{job_id(world)}"] = RegionalFixtureError("finalizer")
    world.failures["prewarm.cleanup"] = RegionalFixtureError("DaemonSet stuck")
    world.failures[f"runtime.verify:{AFTER_CLEANUP}"] = RegionalFixtureError(
        "api-ha rolled"
    )

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["error"] == "RegionalFixtureError: 1800 s elapsed", result
    reports = result["target_reports"]
    assert reports["job"]["sampler_cleanup_error"] == (
        "HostProbeTransportError: node-a is rebooting"
    ), reports
    assert "sampler_cleanup_error" not in reports["idle"], reports
    assert world.names().count(f"probe.stop-reset-sampler:{IDLE}") == 1
    assert result["job_delete_error"] == "RegionalFixtureError: finalizer", result
    assert result["prewarm_residuals"] == {"cleanup_error": True}, result
    assert result["prewarm_cleanup_error"] == "RegionalFixtureError: DaemonSet stuck"
    assert result["runtime_identity_error"] == "RegionalFixtureError: api-ha rolled"


def test_residual_probe_and_prewarm_resources_fail_a_recovered_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures[f"probe.cleanup:{IDLE}"] = HostProbeTransportError("exec refused")
    world.probe_residuals[MASTER] = {"pod": True}
    world.prewarm_residuals = {"pods": True}

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["errors"] == [], "the recovery itself passed"
    reports = result["target_reports"]
    assert reports["job"]["probe_residuals"] == {"pod": True}, reports
    assert reports["idle"]["probe_residuals"] == {"cleanup_error": True}, reports
    assert reports["idle"]["probe_cleanup_error"] == (
        "HostProbeTransportError: exec refused"
    ), reports
    assert result["job_delete"] == {"deleted": job_id(world)}, result


def test_an_unnamed_workflow_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.states[IDLE]["workflow"]["request_id"] = ""

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.INCONCLUSIVE, result
    assert result["oracle"]["inconclusive"] == [
        "a fault never produced a workflow the store could name"
    ], result["oracle"]


def test_an_unreadable_store_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures["oracle.read"] = RegionalFixtureError("exec failed")

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.INCONCLUSIVE, result
    assert result["oracle"]["inconclusive"] == [
        "the store could not be read: exec failed"
    ], result["oracle"]


def test_an_idle_reset_that_escalated_to_a_reboot_loses_its_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.on_injection[IDLE] = {"boot_id": f"boot-{IDLE}-2"}
    world.cpu_after = {"nodes": ["cpu-a", "cpu-z"]}

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    idle = [item for item in result["errors"] if item.startswith(f"{IDLE}: ")]
    assert len(idle) == 1 and "boot id changed" in idle[0], result["errors"]
    assert result["errors"][-1] == "control-plane EKS state differs from baseline"
    assert f"probe.stop-reset-sampler:{IDLE}" not in world.names(), (
        "the reboot ended the idle node's sampler"
    )


def test_a_master_whose_reboot_failed_is_judged_and_its_sampler_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.on_injection.pop(MASTER)
    state = world.states[MASTER]
    state["workflow"]["status"] = "FAILED"
    state["workflow"]["step_executions"] = [
        {**item, "status": "FAILED"} if item["operation"] == "RESTART_NODE" else item
        for item in state["workflow"]["step_executions"]
    ]

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert "node-a: Kubernetes Node boot ID did not change" in result["errors"]
    assert world.names().count(f"probe.stop-reset-sampler:{MASTER}") == 1, (
        "the master never rebooted, so the cleanup stops its sampler"
    )


# --------------------------------------------------------------------------- #
# Pure target selection around kube-dns
# --------------------------------------------------------------------------- #
def test_a_dns_holding_last_target_is_swapped_for_the_next_eligible_node() -> None:
    candidates = {f"node-{name}": [] for name in "abcd"}

    assert destr029.choose_targets(candidates, count=3, dns_nodes=["node-c"]) == (
        "node-a",
        "node-b",
        "node-d",
    )


def test_dns_held_by_the_first_targets_leaves_no_target_set() -> None:
    candidates = {f"node-{name}": [] for name in "abcde"}

    assert destr029.choose_targets(candidates, count=3, dns_nodes=["node-a"]) is None, (
        "swapping the last target cannot free DNS held by the first"
    )
