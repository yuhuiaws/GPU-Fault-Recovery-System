from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    "rollout_regional_release_resume",
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py",
)


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


def test_bootstrap_cleanup_resets_stale_resume_progress() -> None:
    saved = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=[]),
        _cpu=lambda: ["kubectl"],
        _scale_if_present=lambda *_args: None,
        _save_state=lambda phase, **updates: saved.append((phase, updates)),
    )

    MODULE.RegionalRelease._cleanup_bootstrap(release)

    assert saved == [
        (
            "bootstrap-cleaned",
            {
                "previous": None,
                "resume_phase": "bootstrap-started",
                "completed_cluster_ids": [],
            },
        )
    ]
