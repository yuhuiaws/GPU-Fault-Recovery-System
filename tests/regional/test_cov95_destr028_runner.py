"""GF-REGIONAL-DESTR-028 driven end to end against a fake cluster.

Two training jobs of one node group, a reboot-class XID on each job's master:
the plan and the execution run through ``read_only_preflight`` and
``execute_case`` with fakes at the transport boundary only (store, Kubernetes,
host probes, managed jobs, image prewarm). The shared oracle judges records
built by its own test's builders, the job reboot contract judges the nodes.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from gpu_fault.remediation_storm import StormEntry
from scripts.e2e.regional import run_destr028_two_job_parallel_reboot as destr028
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_parallel_domains import (
    REBOOT_OWNER,
    WORKLOAD_OWNERS,
    World,
    bind_job,
    job_host,
    job_node_state,
    profile,
    regional_settings,
    workload_id,
)
from tests.regional.test_workload_domain_oracle import part, premise_snapshot

AFTER_CLEANUP = "after DESTR-028 cleanup"


def job_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, nodes: int = 4
) -> World:
    """``nodes`` idle GPU nodes of one group: four carry 2+2, three 2+1."""

    names = tuple(f"node-{letter}" for letter in "abcd"[:nodes])
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    settings = destr028.Settings(
        regional=regional_settings(tmp_path),
        host_probe_image="example.test/probe@sha256:" + "c" * 64,
        site_file=site,
        manifest=destr028.TRAINING_MANIFEST,
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )
    world = World(
        destr028,
        tmp_path,
        monkeypatch,
        settings=settings,
        nodes=names,
        runtime_profile=profile(*WORKLOAD_OWNERS, REBOOT_OWNER),
    )
    jobs = destr028.job_layout(
        names, world.run_id(), workload_bases=settings.workload_bases()
    )
    parts = []
    for spec, start in zip(jobs, (0, 5), strict=True):
        label = f"job-{spec.label}"
        world.states[spec.master] = job_node_state(label)
        world.baselines[spec.master] = job_host()
        world.on_injection[spec.master] = {"boot_id": f"boot-{spec.master}-2"}
        item = part(
            label, spec.master, start=start, job=spec.job_id, peers=spec.workers
        )
        parts.append(bind_job(item, spec.workload_id))
    world.records_of(*parts)
    return world


def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: int) -> World:
    world = job_world(tmp_path, monkeypatch, **kwargs)
    preflight = world.plan()
    assert preflight["errors"] == [], preflight["errors"]
    return world


def job_ids(world: World) -> list[str]:
    return sorted(world.workloads)


# --------------------------------------------------------------------------- #
# The pass shapes
# --------------------------------------------------------------------------- #
def test_two_two_node_jobs_reboot_their_masters_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.PASS, result
    assert (result["layout"], result["errors"]) == ("2+2", []), result
    assert result["oracle"]["facts"]["breaker"]["counted_nodes"] == {
        "job-a": ["node-a"],
        "job-b": ["node-c"],
    }, result["oracle"]["facts"]["breaker"]
    assert result["workflow_request_ids"] == {
        "a": "workflow-job-a",
        "b": "workflow-job-b",
    }
    assert set(result["markers"]) == {"node-a", "node-c"}, result["markers"]
    names = world.names()
    assert names.index("prewarm.create") < min(
        names.index(f"job.submit:{job}") for job in job_ids(world)
    ), names
    assert max(names.index(f"job.running:{job}") for job in job_ids(world)) < min(
        names.index("probe.write-xid79:node-a"), names.index("probe.write-xid79:node-c")
    ), "both jobs run before either master is faulted"
    assert not any(
        "write-xid" in name and name.endswith(("node-b", "node-d")) for name in names
    ), "the workers are never faulted"
    assert not any("stop-reset-sampler" in name for name in names), (
        "the reboot ended both samplers' transient units"
    )
    for job in job_ids(world):
        assert names.count(f"job.delete:{job}") == 1, names
        settings = world.workloads[job].settings
        assert (settings.expected_pods, settings.expected_gpu_count) == (2, 16)
        rendered = yaml.safe_load(settings.manifest.read_text(encoding="utf-8"))
        assert (
            rendered["metadata"]["name"]
            == (workload_id("gpu-fault-two-node-replace", job).rsplit("/", 1)[-1])
        ), rendered["metadata"]
    assert result["prewarm_residuals"] == {"pods": False}, result
    assert set(result["job_reports"]) == {"a", "b"}, result["job_reports"]
    assert world.evidence("prewarm.json") == {
        "cached_nodes": ["node-a", "node-b", "node-c", "node-d"]
    }
    assert world.evidence("job-a/node-after.json")["boot_id"] == "boot-node-a-2"


def test_three_nodes_carry_a_two_node_and_a_single_node_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch, nodes=3)

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.PASS, result
    assert result["layout"] == "2+1", result
    single = next(item for item in result["jobs"] if item["label"] == "b")
    assert (single["nodes"], single["workers"], single["gpu_count"]) == (
        ["node-c"],
        [],
        8,
    ), single
    settings = world.workloads[single["job_id"]].settings
    assert (settings.expected_pods, settings.expected_gpu_count) == (1, 8)
    rendered = yaml.safe_load(settings.manifest.read_text(encoding="utf-8"))
    assert set(rendered["spec"]["pytorchReplicaSpecs"]) == {"Master"}, rendered
    assert world.evidence("prewarm.json") == {
        "cached_nodes": ["node-a", "node-b", "node-c"]
    }


# --------------------------------------------------------------------------- #
# Preflight refusals
# --------------------------------------------------------------------------- #
def test_a_group_without_three_eligible_nodes_names_every_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = job_world(tmp_path, monkeypatch, nodes=3)
    world.views["node-b"] = "ACTIVE"
    world.stores["node-a"]["queue"] = {"depth": 0, "fault_backlog_depth": 4}
    world.tests_pass = False
    world.predecessor_valid = False

    preflight = world.plan()

    errors = preflight["errors"]
    assert (preflight["targets"], preflight["layout"], preflight["jobs"]) == (
        [],
        None,
        [],
    ), preflight
    assert errors[0].startswith(
        "node group gpu-group has no 3 eligible idle nodes now: "
    ), errors
    assert "node-b workload state is ACTIVE, not IDLE" in errors[0], errors
    assert "processor queue is not empty" in errors, errors
    assert "focused regression tests failed" in errors, errors
    assert "DESTR-002 predecessor evidence is not PASS" in errors, errors
    details = world.evidence("plan.json")["details"]
    assert (details["layout"], details["jobs"]) == (None, []), details


def test_unowned_recovery_a_counting_breaker_and_open_records_refuse_the_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = job_world(tmp_path, monkeypatch)
    world.stores["node-c"]["profile"] = {
        "warnings": ["drift"],
        "capabilities": list(WORKLOAD_OWNERS),
    }
    world.premise = premise_snapshot(
        StormEntry(
            node_id="node-x",
            domain="incident:other-case",
            workflow_id="w-other",
            admitted_at=world.clock.now() - timedelta(seconds=60),
        )
    )
    world.premise["active_workflow_incidents"] = [
        {"node_ids": ["node-d"], "workflow_request_id": "w-old", "status": "RUNNING"}
    ]

    preflight = world.plan()

    errors = preflight["errors"]
    assert preflight["layout"] == "2+2", preflight
    assert "node-c: runtime profile has no nodeReboot capability" in errors, errors
    assert "node-c: runtime profile has warnings" in errors, errors
    assert any(
        "still counts ['node-x'] from the last 600s; with the case's "
        "['node-a', 'node-c'] that is 3 nodes" in item
        for item in errors
    ), "only the two masters are added to what the breaker still counts"
    assert "['node-d'] already have an open workflow w-old (RUNNING)" in errors, errors


def test_a_group_that_shrank_after_the_plan_records_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.gpu_nodes = world.gpu_nodes[:2]

    code, result = world.execute()

    assert code == 0 and result["verdict"] == oracle.NOT_RUN, result
    assert "has 2" in result["detail"], result
    assert world.workloads == {} and "prewarm.create" not in world.names(), (
        "a NOT_RUN submits no job and prewarms nothing"
    )


def test_a_failed_preflight_or_a_drifted_plan_stops_before_any_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.tests_pass = False
    with pytest.raises(RegionalFixtureError, match="focused regression tests failed"):
        world.execute()
    world.tests_pass = True
    world.stores["node-d"]["agent"]["generation"] = 8

    with pytest.raises(RegionalFixtureError, match="DESTR-028 plan identity drifted"):
        world.execute()

    assert world.workloads == {}, "no job is created before the plan is re-proven"


# --------------------------------------------------------------------------- #
# Stops inside the execution and the cleanup
# --------------------------------------------------------------------------- #
def test_a_window_that_ends_between_the_submissions_deletes_only_the_first_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    run_id = world.run_id()
    world.advance_at[f"job.submit:{run_id}-a"] = 7200

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["error"] == (
        f"RegionalFixtureError: maintenance window ended before submit {run_id}-b"
    ), result
    names = world.names()
    assert names.count(f"job.delete:{run_id}-a") == 1, names
    assert f"job.submit:{run_id}-b" not in names, names
    assert f"job.delete:{run_id}-b" not in names, "a job never submitted is not deleted"
    assert "job_delete" not in result["job_reports"]["b"], result["job_reports"]
    assert not any("sampler" in name for name in names), names
    assert "prewarm.cleanup" in names, names


def test_a_master_whose_reboot_failed_keeps_its_sampler_for_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.on_injection.pop("node-c")
    state = world.states["node-c"]
    state["workflow"]["status"] = "FAILED"
    for item in state["workflow"]["step_executions"]:
        if item["operation"] == "RESTART_NODE":
            item["status"] = "FAILED"

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    job_b = f"{world.run_id()}-b"
    assert (
        f"{job_b}: node-c: Kubernetes Node boot ID did not change" in result["errors"]
    )
    assert f"{job_b}: node-c: job reboot workflow is not SUCCEEDED" in result["errors"]
    names = world.names()
    assert names.count("probe.stop-reset-sampler:node-c") == 1, (
        "a node that never rebooted still runs its sampler: the cleanup stops it"
    )
    assert "probe.stop-reset-sampler:node-a" not in names, "node-a rebooted"
    assert world.evidence("job-b/sampler-final.json")["action"] == "stop-reset-sampler"


def test_a_job_wait_that_times_out_leaves_both_samplers_and_jobs_to_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    run_id = world.run_id()
    world.failures["workflow.wait:node-c"] = RegionalFixtureError("1800 s elapsed")
    world.failures["probe.stop-reset-sampler:node-a"] = HostProbeTransportError(
        "node-a is NotReady"
    )
    world.failures[f"job.delete:{run_id}-b"] = RegionalFixtureError("delete hung")

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["error"] == "RegionalFixtureError: 1800 s elapsed", result
    reports = result["job_reports"]
    assert reports["a"]["sampler_cleanup_error"] == (
        "HostProbeTransportError: node-a is NotReady"
    ), reports
    assert reports["a"]["job_delete"] == {"deleted": f"{run_id}-a"}, reports
    assert reports["b"]["job_delete_error"] == "RegionalFixtureError: delete hung"
    assert world.names().count("probe.stop-reset-sampler:node-c") == 1


def test_residual_probe_and_prewarm_resources_fail_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures["probe.cleanup:node-a"] = HostProbeTransportError("exec refused")
    world.probe_residuals["node-c"] = {"pod": True}
    world.prewarm_residuals = {"pods": True}
    world.failures[f"runtime.verify:{AFTER_CLEANUP}"] = RegionalFixtureError(
        "executor rolled"
    )

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["errors"] == [], "the recovery itself passed"
    reports = result["job_reports"]
    assert reports["a"]["probe_residuals"] == {"cleanup_error": True}, reports
    assert (
        reports["a"]["probe_cleanup_error"] == "HostProbeTransportError: exec refused"
    )
    assert reports["b"]["probe_residuals"] == {"pod": True}, reports
    assert result["prewarm_residuals"] == {"pods": True}, result
    assert result["runtime_identity_error"] == "RegionalFixtureError: executor rolled"


def test_a_prewarm_cleanup_that_raises_is_a_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch, nodes=3)
    world.failures["prewarm.cleanup"] = RegionalFixtureError("DaemonSet stuck")

    code, result = world.execute()

    assert code == 1 and result["verdict"] == oracle.FAIL, result
    assert result["prewarm_residuals"] == {"cleanup_error": True}, result
    assert result["prewarm_cleanup_error"] == "RegionalFixtureError: DaemonSet stuck"


def test_a_job_workflow_the_store_cannot_name_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.states["node-a"]["workflow"]["request_id"] = ""

    code, unnamed = world.execute()

    assert code == 1 and unnamed["verdict"] == oracle.INCONCLUSIVE, unnamed
    assert unnamed["oracle"]["inconclusive"] == [
        "a job's XID never produced a workflow the store could name"
    ], unnamed["oracle"]
    assert "oracle.read" not in world.names(), "no store read for an unnamed workflow"


def test_an_unreadable_store_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch)
    world.failures["oracle.read"] = RegionalFixtureError("store unreachable")

    code, unreadable = world.execute()

    assert code == 1 and unreadable["verdict"] == oracle.INCONCLUSIVE, unreadable
    assert unreadable["oracle"]["inconclusive"] == [
        "the store could not be read: store unreachable"
    ], unreadable["oracle"]
    read = next(detail for name, detail in world.calls if name == "oracle.read")
    assert read["nodes"] == "node-a,node-b,node-c,node-d", (
        "the residue read covers every node of both jobs, workers included"
    )


def test_control_plane_drift_fails_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = planned(tmp_path, monkeypatch, nodes=3)
    world.cpu_after = {"nodes": []}

    code, result = world.execute()

    assert code == 1 and result["errors"] == [
        "control-plane EKS state differs from baseline"
    ], result


# --------------------------------------------------------------------------- #
# Pure edges of target selection and rendering
# --------------------------------------------------------------------------- #
def test_targets_holding_dns_in_their_first_nodes_cannot_be_swapped_free() -> None:
    candidates = {f"node-{name}": [] for name in "abcdef"}

    assert (
        destr028.choose_targets(candidates, count=4, dns_nodes=["node-a", "node-b"])
        is None
    ), "swapping the last target never frees DNS held by the first ones"


def test_a_job_of_three_nodes_has_no_manifest(tmp_path: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    settings = destr028.Settings(
        regional=regional_settings(tmp_path),
        host_probe_image="example.test/probe",
        site_file=site,
        manifest=destr028.TRAINING_MANIFEST,
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )
    spec = destr028.JobSpec(
        label="a", job_id="job-x", nodes=("n0", "n1", "n2"), workload_name="job-x"
    )

    with pytest.raises(ValueError, match="job-x: no manifest for 3 nodes"):
        destr028.render_job_manifest(settings, spec, tmp_path)

    assert not (tmp_path / "job.yaml").exists(), "nothing is rendered"
