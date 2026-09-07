from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_cleaned_forces_full_cpu_and_gpu_recovery() -> None:
    completed, cpu_checkpoint, check_live_cpu = MODULE.bootstrap_resume_context(
        {
            "phase": "bootstrap-cleaned",
            "resume_phase": "bootstrap-cpu-ready",
            "release_id": "release-a",
            "completed_cluster_ids": ["gpu-a"],
        },
        "release-a",
    )

    assert completed == set()
    assert cpu_checkpoint is False
    assert check_live_cpu is False


@pytest.mark.parametrize(
    "phase",
    (
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
    ),
)
def test_bootstrap_cleanup_phase_never_reuses_old_checkpoint(phase: str) -> None:
    completed, cpu_checkpoint, check_live_cpu = MODULE.bootstrap_resume_context(
        {
            "phase": phase,
            "resume_phase": "bootstrap-endpoint-ready",
            "release_id": "release-a",
            "completed_cluster_ids": ["gpu-a"],
        },
        "release-a",
    )

    assert completed == set()
    assert cpu_checkpoint is False
    assert check_live_cpu is False


def test_bootstrap_cleanup_resets_stale_resume_progress() -> None:
    saved: list[tuple[str, dict[str, object]]] = []
    release = SimpleNamespace(
        state={},
        config=SimpleNamespace(clusters=[]),
        _cpu=lambda: ["kubectl"],
        _scale_if_present=lambda *_args, **_kwargs: None,
        _save_state=lambda phase, **updates: saved.append((phase, updates)),
    )

    MODULE.RegionalRelease._cleanup_bootstrap(release)

    assert [phase for phase, _updates in saved] == [
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-progress",
        "bootstrap-cleaned",
    ]
    assert saved[-1][1]["resume_phase"] == "bootstrap-started"
    assert saved[-1][1]["completed_cluster_ids"] == []
    assert saved[-1][1]["bootstrap_cleanup_completed_steps"] == [
        "cpu-scaled-down",
        "gpu-scaled-down",
        "installer-jobs-cancelled",
    ]


def test_bootstrap_cleanup_cancels_installer_jobs_before_gpu_scale_down() -> None:
    calls: list[tuple[str, str]] = []
    target = SimpleNamespace(cluster_id="gpu-a")

    def scale(_kubectl, deployment, _replicas, *, wait=False):
        calls.append(("scale", deployment))
        assert wait is True

    release = SimpleNamespace(
        state={},
        config=SimpleNamespace(clusters=[target]),
        _cpu=lambda: ["cpu"],
        _gpu=lambda _target: ["gpu"],
        _scale_if_present=scale,
        _cancel_active_installer_jobs=lambda _target: calls.append(
            ("cancel", "installer-jobs")
        ),
        _save_state=lambda *_args, **_kwargs: None,
    )

    MODULE.RegionalRelease._cleanup_bootstrap(release)

    reconciler = MODULE.inventory.GPU_RECONCILER_DEPLOYMENT
    first_gpu_runtime = MODULE.inventory.DEPLOYMENTS[0]
    assert calls.index(("scale", reconciler)) < calls.index(
        ("cancel", "installer-jobs")
    )
    assert calls.index(("cancel", "installer-jobs")) < calls.index(
        ("scale", first_gpu_runtime)
    )


def test_bootstrap_cleanup_resumes_only_missing_steps() -> None:
    calls: list[str] = []
    release = SimpleNamespace(
        state={
            "bootstrap_cleanup_completed_steps": [
                "installer-jobs-cancelled",
                "gpu-scaled-down",
            ]
        },
        config=SimpleNamespace(clusters=[]),
        _cpu=lambda: ["cpu"],
        _scale_if_present=lambda *_args, **_kwargs: calls.append("scale"),
        _save_state=lambda *_args, **_kwargs: None,
    )

    MODULE.RegionalRelease._cleanup_bootstrap(release)

    assert calls == ["scale"] * len(MODULE.inventory.CPU_DEPLOYMENTS)
