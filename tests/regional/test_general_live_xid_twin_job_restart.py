"""GF-GENERAL-LIVE-024 ``xid_twin_unbound_job``: the product's restart of the job,
adopted by the variant's own journaled fixture.

Live 2026-11-02 (plain EKS, R16): the occurrence's one workflow stopped the
managed job, reset the XID's GPU (one the job does not hold) and restarted the
job -- in place, as RESTART_WORKLOAD resumes a job that is not terminal: the
same PyTorchJob UID, the derived attempt id ``<source[:52]>-r-<hash>`` on it
and its Pod template, a new Pod. The variant then waited for the restarted Pod
through ``ManagedWorkloadFixture.wait_restarted`` without binding the restart,
and the fixture's strict rule (the source attempt id only) refused it:
"managed workload ownership or UID changed". A journaled fixture could bind no
restart at all, and the XID's incident names no job (its GPU held no managed
attempt), which the restart custody required.

Here the variant's own ``start_job`` / ``judge_job`` run over the in-memory
kubectl of ``test_general_live_managed_job_cleanup`` (its training operator
makes the replica Pods); the product's STOP_WORKLOADS and in-place
RESTART_WORKLOAD are applied to the cluster the way
``adapters/kubernetes/workload_operations`` patches them, and its records are
the live shape (``_workload_restart_support.restart_state(unbound=True)``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_xid_twins as twins
from scripts.e2e.regional.managed_workload_fixture import (
    ManagedWorkloadFixture,
    WorkloadOwnershipError,
)
from tests.regional._workload_restart_support import (
    apply_restart_metadata,
    restart_state,
)
from tests.regional.test_general_live_managed_job_cleanup import (
    ATTEMPT_LABEL,
    Cluster,
    journal,
    private_umask,  # noqa: F401 - autouse: the journals need the runner's umask
    variant_context,
)

KEY = "GF-GENERAL-LIVE-024.xid_twin_unbound_job"
#: The node's other four GPUs; the XID is on a fifth.
JOB_GPUS = [f"GPU-0000000{index}-0000-4000-8000-00000000000{index}" for index in "0123"]
XID_GPU = "GPU-00000007-0000-4000-8000-000000000007"


def stop(cluster: Cluster, name: str, state: dict[str, Any]) -> None:
    """The occurrence's STOP_WORKLOADS: the step annotations and the suspend in
    one patch, and the operator removes the attempt's Pod."""

    job = cluster.objects[("pytorchjob", name)]
    workflow = state["workflow"]["request_id"]
    incident = state["incident"]["incident_id"]
    job["metadata"].setdefault("annotations", {}).update(
        {
            "gpu-fault.io/incident-id": incident,
            "gpu-fault.io/workflow-id": workflow,
            "gpu-fault.io/operation-id": f"{workflow}/2/STOP_WORKLOADS",
            "gpu-fault.io/termination-initiator-incident-id": incident,
        }
    )
    job["spec"].setdefault("runPolicy", {})["suspend"] = True
    cluster.bump(job)
    cluster.reconcile(job)


def restart(cluster: Cluster, name: str, state: dict[str, Any]) -> dict[str, Any]:
    """The occurrence's RESTART_WORKLOAD of the suspended job: the same object,
    the derived attempt id on it and its template, the restart annotations,
    unsuspended -- the operator makes a new Pod."""

    job = cluster.objects[("pytorchjob", name)]
    apply_restart_metadata(job, state)
    target = job["metadata"]["labels"][ATTEMPT_LABEL]
    for replica in job["spec"]["pytorchReplicaSpecs"].values():
        replica["template"]["metadata"]["labels"][ATTEMPT_LABEL] = target
    job["spec"]["runPolicy"]["suspend"] = False
    cluster.bump(job)
    cluster.reconcile(job)
    return job


def activity(state: dict[str, Any]) -> dict[str, Any]:
    """The node's activity read as the variant has it after the settle."""

    return {
        "workflows": [state["workflow"]],
        "incidents": [state["incident"]],
        "commands": state["commands"],
    }


def attempt_records(
    _regional: Any, _cluster: str, _job_id: str, attempts: list[str]
) -> dict[str, Any]:
    """The live attempt records: STOPPED by the incident, NO_ACTION, ledger 1."""

    return {
        "attempts": {
            attempts[0]: {
                "event": {"terminal_status": "STOPPED"},
                "decision": {"status": "NO_ACTION"},
                "containment_incident": None,
            }
        },
        "restart_budget": {"restart_count": 1, "budget": 2},
    }


def started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, Cluster, dict[str, Any], dict[str, Any]]:
    """The variant's job running, stopped and restarted in place by the product."""

    cluster = Cluster()
    ctx = variant_context(KEY, tmp_path, monkeypatch, cluster)
    job = twins.start_job(ctx)
    held = {
        "container_gpus": JOB_GPUS,
        "watcher_gpus": JOB_GPUS,
        "pod_uids": job.identity()["pod_uids"],
    }
    state = restart_state(job.fixture, unbound=True)
    stop(cluster, job.fixture.name, state)
    restarted = copy.deepcopy(restart(cluster, job.fixture.name, state))
    monkeypatch.setattr(twins, "attempt_records", attempt_records)
    return ctx, job, cluster, state, {"held": held, "job": restarted}


def test_the_unbound_restart_is_the_live_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The variant's wait as it ran live: unbound, the fixture refuses the
    product's restarted Pod by its derived attempt id."""

    _ctx, job, _cluster, _state, run = started(tmp_path, monkeypatch)
    old = set(run["held"]["pod_uids"])

    with pytest.raises(WorkloadOwnershipError, match="ownership or UID changed"):
        job.fixture.wait_restarted(old, timeout_seconds=30, poll_seconds=0)


def test_the_job_restarted_in_place_is_adopted_and_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, job, cluster, state, run = started(tmp_path, monkeypatch)
    restarted = run["job"]
    facts = {"workflow_request_id": state["workflow"]["request_id"]}

    errors = twins.judge_job(ctx, job, activity(state), facts, run["held"], gpu=XID_GPU)

    assert errors == [], errors
    target = state["commands"][0]["result_details"]["restart_attempt_id"]
    assert facts["restart_attempt_id"] == target, facts
    assert restarted["metadata"]["labels"][ATTEMPT_LABEL] == target
    evidence = journal(ctx.case_dir / "job-restarted.json")
    assert evidence["workload"]["uid"] == restarted["metadata"]["uid"], (
        "the same PyTorchJob, restarted in place"
    )
    assert evidence["workload"]["labels"][ATTEMPT_LABEL] == target
    [pod] = evidence["pods"]
    assert pod["attempt_id"] == target and pod["uid"] not in run["held"]["pod_uids"]
    record = journal(job.fixture.ownership.path)
    assert record["observed"][f"pod/{pod['name']}"] == pod["uid"], (
        "the journal holds the restarted Pod for the cleanup"
    )

    cleanup = job.cleanup()

    workload = cleanup["workload"]
    assert workload["product_restarts"] == [], (
        "adopted under the restart custody, not as an unbound residue"
    )
    assert (
        workload["deleted"][f"pytorchjob/{job.fixture.name}"]
        == (restarted["metadata"]["uid"])
    )
    assert workload["deleted"][f"pod/{pod['name']}"] == pod["uid"]
    assert cluster.kinds() == {"deployment"}, cluster.objects
    assert journal(job.fixture.ownership.path)["completed"] is True


def test_a_restart_the_records_do_not_bind_is_never_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workflow's STOP receipt names another controller: the job stage
    fails by name, nothing is adopted, and the cleanup still removes the
    product's restart under the lineage rule."""

    ctx, job, cluster, state, run = started(tmp_path, monkeypatch)
    receipt = state["workflow"]["step_executions"][1]["details"][
        "stop_ownership_receipt_v1"
    ]
    receipt["workloads"][0]["workload_id"] = "gpu-fault-system/pytorchjob/another"
    before = dict(job.fixture.owned_uids)
    facts = {"workflow_request_id": state["workflow"]["request_id"]}

    errors = twins.judge_job(ctx, job, activity(state), facts, run["held"], gpu=XID_GPU)

    assert errors == [
        "the product's restart is not bound to the job: WorkloadOwnershipError: "
        "managed workload restart custody: no contained STOP receipt of this "
        "workflow names the unbound job's source controller"
    ], errors
    assert job.fixture.restart_custody is None
    assert job.fixture.owned_uids == before
    cleanup = job.cleanup()
    assert cleanup["workload"]["product_restarts"] == sorted(
        [f"pod/{job.fixture.name}-master-0", f"pytorchjob/{job.fixture.name}"]
    ), cleanup
    assert cluster.kinds() == {"deployment"}, cluster.objects


def test_a_reopened_journal_binds_no_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart custody is the submitting process's: a fixture over the same
    journal reopened by a later process stays cleanup-only."""

    _ctx, job, _cluster, state, _run = started(tmp_path, monkeypatch)
    reopened = ManagedWorkloadFixture(
        job.fixture.regional,
        job.fixture.settings,
        state_path=job.fixture.ownership.path,
    )

    with pytest.raises(WorkloadOwnershipError, match="non-resuming"):
        reopened.authorize_restart(state, unbound=True)
    assert reopened.restart_custody is None
