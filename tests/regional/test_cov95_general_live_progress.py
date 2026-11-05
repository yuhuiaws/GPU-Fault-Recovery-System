"""GF-GENERAL-LIVE-027 progress job: diagnostics, rank lookups and manifest edits.

The job is the runner's own ``ManagedProgressJob`` over the fake cluster of
``test_general_live_progress_job``; nothing here reaches a cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.workload_restart_custody import WorkloadOwnershipError
from tests.regional.test_general_live_progress_job import (
    NODE,
    progress_job,
    status,
)
from tests.regional.test_general_live_progress_job import (
    private_umask as private_umask,
)

pytestmark = pytest.mark.usefixtures("private_umask")
MANIFESTS = Path(progress.__file__).with_name("manifests") / "training"


def test_identity_names_the_job_attempt_ranks_and_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)

    assert job.identity() == {
        "job_id": job.name,
        "attempt_id": f"{job.name}-a001",
        "ranks": [0],
        "nodes": [NODE],
    }


def test_pod_for_refuses_a_foreign_rank_and_a_node_without_one_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)

    with pytest.raises(
        RegionalFixtureError, match=f"rank 7 is not a rank of {job.name}"
    ):
        job.pod_for(7)
    # Nothing submitted yet: no Pod runs rank 0.
    with pytest.raises(
        RegionalFixtureError, match=f"has 0 Pods on {NODE}, rank 0 needs one"
    ):
        job.pod_for(0)


def test_reporter_logs_name_a_reporter_that_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)

    logs = job.reporter_logs()

    assert list(logs) == ["progress-reporter-0"]
    assert logs["progress-reporter-0"].startswith(
        "unreadable: RegionalFixtureError: "
    ), logs
    assert "rank 0 needs one" in logs["progress-reporter-0"]


def test_unready_containers_reports_each_container_that_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)
    reads: list[tuple[str, ...]] = []
    listing = {
        "items": [
            "not-a-pod",
            {
                "metadata": {"name": "pod-0"},
                "status": {
                    "containerStatuses": [
                        status("pytorch", ready=True),
                        status(
                            "progress-reporter-0",
                            waiting="CrashLoopBackOff",
                            message="x" * 400,
                        ),
                        status("sidecar", terminated=(137, "OOMKilled")),
                        "junk",
                    ]
                },
            },
        ]
    }

    def kubectl(*arguments: str, **_options: Any) -> str:
        reads.append(arguments)
        return json.dumps(listing)

    monkeypatch.setattr(job.ctx.regional, "kubectl", kubectl)

    assert job.unready_containers() == {
        "pod-0/progress-reporter-0": {
            "state": "waiting",
            "reason": "CrashLoopBackOff",
            "exit_code": None,
            "message": "x" * 300,
        },
        "pod-0/sidecar": {
            "state": "terminated",
            "reason": "OOMKilled",
            "exit_code": 137,
            "message": "",
        },
    }
    assert reads == [
        ("gpu", "get", "pod", "-l", f"gpu-fault.io/job-id={job.name}", "-o", "json")
    ]


def test_unready_containers_is_a_diagnostic_that_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)

    def kubectl(*_arguments: str, **_options: Any) -> str:
        raise RegionalFixtureError("the GPU API server timed out")

    monkeypatch.setattr(job.ctx.regional, "kubectl", kubectl)

    assert job.unready_containers() == {
        "unreadable": "RegionalFixtureError: the GPU API server timed out"
    }


def test_start_lets_an_ownership_refusal_through_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = progress_job(tmp_path, monkeypatch)
    calls: list[str] = []
    refusal = WorkloadOwnershipError("the job is owned by another run")

    def wait_running(**_options: Any) -> dict[str, Any]:
        calls.append("wait")
        raise refusal

    monkeypatch.setattr(job.prewarm, "create", lambda nodes: calls.append("prewarm"))
    monkeypatch.setattr(job.fixture, "submit", lambda: calls.append("submit"))
    monkeypatch.setattr(job.fixture, "wait_running", wait_running)
    monkeypatch.setattr(
        job, "unready_containers", lambda: pytest.fail("no diagnostic for a refusal")
    )

    with pytest.raises(WorkloadOwnershipError) as caught:
        job.start()

    assert caught.value is refusal
    assert calls == ["prewarm", "submit", "wait"]


def two_node_document() -> dict[str, Any]:
    return yaml.safe_load(
        (MANIFESTS / progress.TWO_NODE_MANIFEST).read_text(encoding="utf-8")
    )


def test_pin_replicas_refuses_a_replica_without_a_pod_spec() -> None:
    document = two_node_document()
    document["spec"]["pytorchReplicaSpecs"]["Worker"]["template"] = {"metadata": {}}

    with pytest.raises(ValueError, match="the Worker replica has no Pod spec"):
        progress.pin_replicas(document, {"Master": "node-a", "Worker": "node-b"})


def test_install_reporter_skips_specless_replicas_and_non_container_entries() -> None:
    document = two_node_document()
    replicas = document["spec"]["pytorchReplicaSpecs"]
    master = replicas["Master"]["template"]["spec"]
    master["containers"].append("not-a-container")
    replicas["Launcher"] = {"replicas": 1, "template": {}}

    progress.install_reporter(
        document,
        "gl27-job",
        {"Master": (0,), "Worker": (1,)},
        "registry.example/reporter@sha256:" + "2" * 64,
    )

    assert replicas["Launcher"] == {"replicas": 1, "template": {}}
    assert "not-a-container" in master["containers"]
    mounted = [
        container["name"]
        for container in master["containers"]
        if isinstance(container, dict)
        and {"name": "gfrs-progress", "mountPath": progress.PROGRESS_DIR}
        in container.get("volumeMounts", [])
    ]
    assert "progress-reporter-0" in mounted
    assert len(mounted) == len(
        [item for item in master["containers"] if isinstance(item, dict)]
    )
