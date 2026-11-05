"""GF-REGIONAL-DESTR-026 ``--execute`` against fakes at the transport boundary.

A clean run passes through the real verdicts; every refusal before the
injection stops before the next mutation; a fault that did not set up the case
stops before the forty-minute observation; a workflow that never ends, or ends
FAILED, owes no restart; a fault node that never answers again is never exec'd.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr026_same_node_two_gpu_reset as destr026
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr026_harness import (
    BDF_A,
    BDF_B,
    JOB_ID,
    RESTARTED_ATTEMPT,
    Destr026Harness,
    isolate,
)
from tests.regional.test_destr026_same_node_two_gpu_reset import (
    GPU_A,
    GPU_B,
    INCIDENT,
    NODE,
    PARTNER,
    REQUEST,
)


def harness(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Destr026Harness:
    h = Destr026Harness(tmp_path, monkeypatch)
    preflight = h.plan()
    assert preflight["errors"] == [], preflight["errors"]
    return h


def probe_args(h: Destr026Harness, action: str, probe: str) -> list[list[str]]:
    return [
        detail["args"]
        for detail in h.details(f"probe.{action}")
        if detail["probe"] == probe
    ]


# --------------------------------------------------------------------------- #
# The pass shape
# --------------------------------------------------------------------------- #
def test_a_clean_run_resets_both_gpus_in_one_window_and_passes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.pending_reads = 1  # the first terminal read still shows the job running
    h.attempts_missing = 1  # the watcher reports the restart on its next pass
    h.outage_seconds = 120  # the quiesce stops kubelet on the fault node

    code, report = h.execute()

    assert code == 0, report
    assert report["verdict"] == "PASS", report
    assert report["errors"] == [] and report["cleanup"]["errors"] == [], report
    assert report["targets"] == {"a": GPU_A, "b": GPU_B}, report["targets"]
    assert report["pci_bdfs"] == {"a": BDF_A, "b": BDF_B}, report["pci_bdfs"]
    assert (report["incident_id"], report["workflow_request_id"]) == (
        INCIDENT,
        REQUEST,
    ), report
    assert set(report["components"]) == {
        "preflight_identity",
        "workflow",
        "incident",
        "hosts",
    }, report["components"]
    assert report["summary"]["boot_ids"] == {
        "fault": ["boot-a", "boot-a"],
        "node_b": ["boot-b", "boot-b"],
    }, report["summary"]
    assert report["summary"]["target_pods"] == ["new-node-a", "new-node-b"], report
    writes = {
        probe: args[0][args[0].index("--pci-bdf") + 1]
        for probe in ("a", "b")
        for args in [probe_args(h, "write-xid46", probe)]
    }
    assert writes == {"a": BDF_A, "b": BDF_B}, "one XID 46 per GPU, one probe each"
    names = h.names()
    assert names.count("workload.authorize_restart") == 1, "the job restarts once"
    assert names.count("store.attempts") == 2, "the attempts read waits for the restart"
    observe = [d for d in h.details("store.workflow") if "workflow_request_ids" in d]
    assert len(observe) == 2, "a non-terminal read is followed by one more"
    assert observe[0]["workflow_request_ids"] == [REQUEST], observe[0]
    # The fault node is read again only after it answers: the wait comes
    # before the post-reset snapshot, and the sampler stops right after it.
    after = names.index("node.wait_ready")
    snapshots = [
        index
        for index, (name, detail) in enumerate(h.calls)
        if name == "probe.snapshot" and "--since-epoch" in detail["args"]
    ]
    assert snapshots and min(snapshots) > after, "no exec before the node answers"
    assert names[snapshots[0] + 1] == "probe.stop-reset-sampler", names
    assert names.count("probe.stop-reset-sampler") == 1, (
        "cleanup does not stop it again"
    )
    assert names.index("workload.delete") > max(
        index for index, name in enumerate(names) if name == "store.workflow"
    ), "the workload is deleted only after quiescence"
    assert h.evidence("fault-node-return.json")["exec_allowed"] is True
    assert h.evidence("step-timeline.json")["transitions"], "the timeline is recorded"
    attempts = h.evidence("job-attempts.json")["observations"]
    assert {item["attempt_id"] for item in attempts} >= {RESTARTED_ATTEMPT}, attempts
    assert h.evidence("workload-target.json")["pods_after_quiet_period"], "quiet read"
    assert h.evidence("targets.json")["targets"] == {"a": GPU_A, "b": GPU_B}
    assert set(h.evidence("injection.json")["injected_at"]) == {"a", "b"}


# --------------------------------------------------------------------------- #
# Refusals before anything is created
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["site_file", "manifest"])
def test_the_preflight_refuses_a_missing_site_file_or_manifest(
    missing: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Destr026Harness(tmp_path, monkeypatch)
    settings = replace(h.settings, **{missing: tmp_path / "absent.yaml"})

    with pytest.raises(RegionalFixtureError, match="does not exist"):
        destr026.read_only_preflight(settings, tmp_path)

    assert h.calls == [], "nothing is read before the inputs exist"


def test_execute_refuses_a_preflight_that_no_longer_passes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.predecessor_valid = False

    with pytest.raises(RegionalFixtureError, match="preflight failed: DESTR-015"):
        h.execute()

    assert "workload.fixture" not in h.names(), "no workload before a passing preflight"
    assert "probe.create" not in h.names(), "no probe before a passing preflight"


def test_execute_refuses_a_plan_whose_node_identity_drifted(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.nodes[PARTNER]["boot_id"] = "boot-b-rebooted"

    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        h.execute()

    assert "prewarm.create" not in h.names(), "a drifted plan prewarms nothing"


@pytest.mark.parametrize(
    ("stage", "advancing_call"),
    [
        ("case setup", None),
        ("workload submission", "prewarm.create"),
        ("sampler start", "runtime.verify"),
        ("injection", "probe.start-reset-sampler"),
    ],
)
def test_a_closed_maintenance_window_stops_before_the_next_mutation(
    stage: str,
    advancing_call: str | None,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness(tmp_path, monkeypatch)
    if advancing_call is not None:
        h.advance_at[advancing_call] = 7200

    code, report = h.execute(seconds=-1 if advancing_call is None else 600)

    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["error"] == (
        f"RegionalFixtureError: maintenance window ended before {stage}"
    ), report
    names = h.names()
    assert "probe.write-xid46" not in names, "no fault is written after the window"
    cleanup = report["cleanup"]
    assert "quiescence" not in cleanup, "nothing was injected, nothing to wait for"
    assert "restore_isolated_nodes" in cleanup, "the isolation check still runs"
    submitted = stage in {"sampler start", "injection"}
    assert ("workload.submit" in names) is submitted, names
    assert ("workload.delete" in names) is submitted, "a submitted job is deleted"
    sampler = stage == "injection"
    assert ("sampler_stop" in cleanup) is sampler, cleanup
    assert ("fault_node_return" in cleanup) is sampler, cleanup
    assert cleanup["errors"] == [], cleanup["errors"]
    assert names[-2:] == ["prewarm.cleanup", "runtime.verify"], names


# --------------------------------------------------------------------------- #
# Host baselines, image cache and target selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("probe", "change", "message"),
    [
        ("a", {"gpu_inventory": "short"}, f"{NODE} host GPU inventory is not 8"),
        (
            "partner",
            {"gpu_inventory": "short"},
            f"{PARTNER} host GPU inventory is not 8",
        ),
        (
            "a",
            {"quiesce_states": [{"phase": "QUIESCED"}]},
            f"{NODE} has a pre-existing",
        ),
        (
            "partner",
            {"quiesce_states": [{"phase": "QUIESCED"}]},
            f"{PARTNER} has a pre-existing",
        ),
        (
            "a",
            {"kmsg_writable": False},
            "/dev/kmsg is not writable from the host probe",
        ),
    ],
    ids=[
        "fault-inventory",
        "partner-inventory",
        "fault-quiesce",
        "partner-quiesce",
        "kmsg",
    ],
)
def test_a_host_baseline_that_cannot_carry_the_case_refuses_the_injection(
    probe: str,
    change: dict[str, Any],
    message: str,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness(tmp_path, monkeypatch)
    baseline = h.baselines[probe]
    if change.get("gpu_inventory") == "short":
        baseline["gpu_inventory"] = baseline["gpu_inventory"][:7]
    else:
        baseline.update(change)

    code, report = h.execute()

    assert code == 1 and report["error"].startswith("RegionalFixtureError: "), report
    assert message in report["error"], report["error"]
    names = h.names()
    assert names.count("probe.create") == 3, "every probe was created before the read"
    assert "probe.write-xid46" not in names, "no XID on a node that failed its baseline"
    assert "workload.delete" in names, "the submitted job is deleted"
    assert sorted(h.details("probe.cleanup")) == ["a", "b", "partner"], h.calls


def test_an_image_not_cached_on_both_nodes_refuses_the_submission(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.cached = [NODE]

    code, report = h.execute()

    assert code == 1, report
    assert report["error"] == (
        "RegionalFixtureError: training image is not cached on both nodes"
    ), report
    assert h.evidence("image-cache.json") == {"cached_nodes": [NODE]}
    assert "workload.submit" not in h.names(), "no job without the image on both nodes"


@pytest.mark.parametrize(
    ("bdf_a", "allocation", "message"),
    [
        (
            "",
            [GPU_A, *(f"GPU-a{index}" for index in range(2, 8)), "GPU-elsewhere"],
            "the job does not hold the target GPUs ['GPU-a1']",
        ),
        ("0000:99:00.0", None, "no GPU of the node answers at ['0000:99:00']"),
    ],
    ids=["not-allocated", "unknown-bdf"],
)
def test_target_gpus_the_job_does_not_hold_refuse_the_injection(
    bdf_a: str,
    allocation: list[str] | None,
    message: str,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Destr026Harness(tmp_path, monkeypatch)
    h.settings = replace(h.settings, pci_bdf_a=bdf_a)
    h.plan()
    if allocation is not None:
        h.allocation = allocation

    code, report = h.execute()

    assert code == 1, report
    assert report["error"] == f"RegionalFixtureError: target GPUs: {message}", report
    assert "probe.write-xid46" not in h.names(), "no XID on a GPU the job does not hold"
    assert not (h.case_dir / "targets.json").exists(), "no targets are recorded"


# --------------------------------------------------------------------------- #
# The injection and the observation
# --------------------------------------------------------------------------- #
def test_two_faults_in_two_workflows_stop_before_the_observation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.split = True

    code, report = h.execute()

    assert code == 1, report
    assert report["error"].startswith(
        "RegionalFixtureError: stopping before observation"
    ), report
    assert any("landed in 2 workflows" in item for item in report["errors"]), report
    assert any("opened 2 incidents" in item for item in report["errors"]), report
    names = h.names()
    assert "workload.authorize_restart" not in names, "a split case restarts nothing"
    assert not (h.case_dir / "workflow-state.json").exists(), "no forty-minute watch"
    # Both XIDs are on the node: cleanup waits for quiescence, then stops the
    # sampler once the fault node answers.
    assert "quiescence" in report["cleanup"], report["cleanup"]
    assert report["cleanup"]["fault_node_return"]["exec_allowed"] is True
    assert "sampler_stop" in report["cleanup"], report["cleanup"]
    assert report["cleanup"]["errors"] == [], report["cleanup"]["errors"]


def test_an_observation_budget_that_expires_owes_no_restart_and_defers_cleanup(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.pending_reads = 10**6  # the workflow never leaves RUNNING
    h.advance_at["store.workflow"] = 600  # every store read takes ten minutes

    code, report = h.execute()

    assert code == 1 and report["verdict"] == "FAIL", report
    assert "error" not in report, "an unfinished workflow is graded, not an exception"
    assert "reset workflow is not SUCCEEDED" in report["errors"], report["errors"]
    assert any(
        "workflow status is not SUCCEEDED: RUNNING" in item for item in report["errors"]
    ), report["errors"]
    assert h.evidence("workflow-state.json")["workflow"]["status"] == "RUNNING"
    names = h.names()
    assert "workload.authorize_restart" not in names, (
        "a running workflow owes no restart"
    )
    assert "workload.pods" in names, "the job's Pods are read as they are"
    cleanup = report["cleanup"]
    assert any(
        item.startswith("quiescence: RegionalFixtureError: cleanup could not prove")
        for item in cleanup["errors"]
    ), cleanup["errors"]
    assert cleanup["workload_cleanup_deferred"] is True, cleanup
    assert cleanup["isolation_restore_deferred"] is True, cleanup
    assert "workload.delete" not in names, "a job of an unquiet workflow stays"


def test_a_failed_workflow_owes_no_restart_and_cleanup_restores_the_node(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.workflow["status"] = "FAILED"
    h.hooks["workflow.wait"] = lambda: isolate(h.nodes[NODE], INCIDENT)

    code, report = h.execute()

    assert code == 1, report
    assert any(
        "workflow status is not SUCCEEDED: FAILED" in item for item in report["errors"]
    ), report["errors"]
    assert f"{NODE} is not schedulable after RESTORE_SCHEDULING" in report["errors"]
    assert "workload.authorize_restart" not in h.names(), (
        "no restart after a FAILED run"
    )
    restore = report["cleanup"]["restore_isolated_nodes"]
    assert restore == {
        NODE: {"isolated": True, "restore": "SUCCEEDED"},
        PARTNER: {"isolated": False},
    }, restore
    (created,) = h.details("restore.create")
    assert created == {
        "incident_id": INCIDENT,
        "node": NODE,
        "profile_version": "profile-v1",
        "reason": "DESTR-026 validated cleanup",
    }, created
    assert h.details("restore.idle") == [INCIDENT], "the incident settles first"
    assert h.nodes[NODE]["unschedulable"] is False, "released by the validated restore"
    assert report["cleanup"]["errors"] == [], report["cleanup"]["errors"]


def test_a_fault_node_that_never_answers_again_is_never_exec_d(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.outage_seconds = float("inf")

    code, report = h.execute()

    assert code == 1, report
    assert report["error"] == (
        f"RegionalFixtureError: {NODE} did not answer again (the node is not Ready "
        "and the holder may still be alive); no host read is attempted on a "
        "NotReady node"
    ), report["error"]
    cleanup = report["cleanup"]
    assert cleanup["fault_node_return"]["exec_allowed"] is False, cleanup
    assert any(
        item.startswith(f"{NODE} never answered again")
        and "left for manual reconciliation" in item
        for item in cleanup["errors"]
    ), cleanup["errors"]
    assert h.details("probe.cleanup") == ["partner"], "only node B's probe is removed"
    assert "probe.stop-reset-sampler" not in h.names(), "no exec into a NotReady node"
    assert probe_args(h, "snapshot", "a") == [["--run-id", h.probes["a"].run_id]], (
        "the fault node's only host read is the baseline before the injection"
    )


# --------------------------------------------------------------------------- #
# What the hosts, the cluster and the provider say afterwards
# --------------------------------------------------------------------------- #
def test_a_kernel_journal_with_a_second_reset_fails_the_case(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    h.hosts_after["b"]["kernel_reset_journal"]["target_reset_count"] = 2

    code, report = h.execute()

    assert code == 1, report
    assert report["errors"] == [
        f"kernel journal shows more than one reset for {BDF_B}"
    ], report["errors"]
    assert h.evidence("host-after-fault_second_bdf.json")["kernel_reset_journal"] == {
        "target_reset_count": 2
    }


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("cpu", "control-plane EKS state differs from baseline"),
        ("node-c", "node-c scheduling state changed: []"),
        (
            "provider",
            "provider mutation appeared during a reset-only case: RebootInstances",
        ),
    ],
)
def test_anything_else_the_case_moved_fails_it(
    change: str, expected: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(tmp_path, monkeypatch)
    if change == "cpu":
        h.cpu_after = {"nodes": ["cpu-a", "cpu-b"]}
    elif change == "node-c":
        h.gpu_nodes_after[2]["unschedulable"] = True
    else:
        h.events = [{"event_name": "RebootInstances"}]

    code, report = h.execute()

    assert code == 1, report
    assert report["errors"] == [expected], report["errors"]
    assert report["cleanup"]["errors"] == [], "a clean cleanup does not hide the FAIL"
    provider = h.evidence("provider-events.json")
    assert provider["provisional"] is True, provider
    assert report["job_id"] == JOB_ID, report
