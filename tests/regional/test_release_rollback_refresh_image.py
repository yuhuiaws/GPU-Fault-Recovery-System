"""The rollback verify restores the Aurora refresh CronJob image before judging it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release.regional_release_config import ReleaseError

OLD = "registry/gpu-fault/runtime-old@sha256:" + "1" * 64
CANDIDATE = "registry/gpu-fault/runtime-old@sha256:" + "2" * 64


def _release(cronjob_images: list[str], runs: list[list[str]]) -> SimpleNamespace:
    images = iter(cronjob_images)

    def get_json(arguments: list[str]) -> dict:
        if "deployment" in arguments:
            return {"spec": {"template": {"spec": {"containers": [{"image": OLD}]}}}}
        return {
            "spec": {
                "jobTemplate": {
                    "spec": {
                        "template": {"spec": {"containers": [{"image": next(images)}]}}
                    }
                }
            }
        }

    return SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        runner=SimpleNamespace(
            probe=lambda _arguments: True,
            run=lambda arguments: runs.append(list(arguments)),
        ),
        _cpu=lambda *arguments: ["kubectl", *arguments],
        _deployment_wheel=lambda _kubectl, _name: "wheel-cm",
        _get_json=get_json,
    )


def test_verify_restores_a_cronjob_repointed_by_the_bootstrap_task() -> None:
    """Live 2026-09-08: the deploy re-installed the CronJob with the candidate
    image before resuming the rollback at verify, whose cpu-restore had
    already run; the verify must restore, not just complain."""
    runs: list[list[str]] = []
    release = _release([CANDIDATE, OLD], runs)

    VALIDATION.validate_cpu_rollback(release, {"cpu_wheel": "wheel-cm"}, OLD)

    assert runs == [
        [
            "kubectl",
            "-n",
            "gpu-fault-system",
            "set",
            "image",
            "cronjob/gpu-fault-aurora-credential-refresh",
            f"refresh={OLD}",
        ]
    ], runs


def test_verify_still_fails_when_the_restore_does_not_take() -> None:
    runs: list[list[str]] = []
    release = _release([CANDIDATE, CANDIDATE], runs)

    with pytest.raises(ReleaseError, match="Aurora refresh image did not converge"):
        VALIDATION.validate_cpu_rollback(release, {"cpu_wheel": "wheel-cm"}, OLD)
    assert len(runs) == 1, runs


def test_verify_leaves_a_converged_cronjob_alone() -> None:
    runs: list[list[str]] = []
    VALIDATION.validate_cpu_rollback(
        _release([OLD], runs), {"cpu_wheel": "wheel-cm"}, OLD
    )
    assert runs == [], runs
