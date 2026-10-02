"""``deploy`` (and every other preflight consumer) refuses while a rotation is open.

``gpu-fault-admin deploy`` re-renders the registry Secret from the site's
token files and republishes it; run between two steps of a ``rotate-token``
it wiped the retiring digest the rotation still had to drop and rewrote
``site.yaml``, and the journal could neither resume nor roll back. The
mirror of ``rotate-token``'s own "a release transaction is open" guard: a
journal that is ``IN_PROGRESS`` or still owes its pending-token cleanup
blocks the release preflight and names the command that finishes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import rotate_token_journal as journal
from gpu_fault.admin.site import ADMIN_STATE_DIR_SIDECAR
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_token_rotation_safety as SAFETY
from gpu_fault_release.regional_release_config import ReleaseError

GPU_A_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
GPU_B_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-b"


def _release(state_dir: Path | None) -> SimpleNamespace:
    clusters = tuple(
        SimpleNamespace(cluster_id=cluster_id, eks_cluster_arn=arn)
        for cluster_id, arn in (("gpu-a", GPU_A_ARN), ("gpu-b", GPU_B_ARN))
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            site_name="staging", admin_state_dir=state_dir, clusters=clusters
        )
    )


def _write_journal(state_dir: Path, cluster_id: str, state: dict) -> Path:
    path = journal.rotation_journal_path(state_dir, cluster_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def _journal(status: str, *steps: str, **extra: object) -> dict:
    return {
        "schema_version": journal.STATE_SCHEMA_VERSION,
        "cluster_id": "gpu-a",
        "status": status,
        "steps": {step: {"completed_at": "x", "evidence": {}} for step in steps},
        "started_steps": {step: "x" for step in steps},
        "new_token_sha256": "1" * 64,
        **extra,
    }


def test_an_in_progress_rotation_blocks_the_release_and_names_the_remedy(
    tmp_path: Path,
) -> None:
    path = _write_journal(
        tmp_path,
        "gpu-a",
        _journal(
            journal.STATUS_IN_PROGRESS,
            *journal.ROTATION_STEPS[
                : journal.ROTATION_STEPS.index(journal.STEP_RETIRING_DROPPED)
            ],
        ),
    )

    with pytest.raises(ReleaseError) as failure:
        SAFETY.token_rotation_snapshot(_release(tmp_path))

    message = str(failure.value)
    assert "gpu-a" in message
    assert "IN_PROGRESS" in message
    assert journal.STEP_CONTROL_PLANE_ROLLED in message, "the last completed step"
    assert (
        f"gpu-fault-admin rotate-token --state-dir {tmp_path} "
        f"--gpu-cluster-arn {GPU_A_ARN}" in message
    )
    assert "--rollback" not in message, (
        "past TOKEN_FILE_WRITTEN the only way out is forward"
    )
    assert str(path) in message
    assert "1" * 64 not in message, "the refusal carries no digest"


def test_a_rotation_before_the_token_file_write_offers_the_rollback_too(
    tmp_path: Path,
) -> None:
    _write_journal(
        tmp_path,
        "gpu-a",
        _journal(
            journal.STATUS_IN_PROGRESS,
            journal.STEP_PREPARED,
            journal.STEP_OVERLAP_PUBLISHED,
        ),
    )

    with pytest.raises(ReleaseError, match="--rollback"):
        SAFETY.token_rotation_snapshot(_release(tmp_path))


def test_a_started_rollback_is_finished_with_rollback_only(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        "gpu-a",
        _journal(
            journal.STATUS_IN_PROGRESS,
            journal.STEP_PREPARED,
            journal.STEP_OVERLAP_PUBLISHED,
            rollback_started_at="x",
        ),
    )

    with pytest.raises(ReleaseError) as failure:
        SAFETY.token_rotation_snapshot(_release(tmp_path))

    message = str(failure.value)
    assert f"--gpu-cluster-arn {GPU_A_ARN} --rollback" in message
    assert " or " not in message.split("--rollback")[0].split("rotate-token")[-1]


@pytest.mark.parametrize(
    ("status", "flag"),
    [(journal.STATUS_COMPLETED, ""), (journal.STATUS_ROLLED_BACK, " --rollback")],
)
def test_pending_token_cleanup_blocks_the_release(
    tmp_path: Path, status: str, flag: str
) -> None:
    _write_journal(
        tmp_path,
        "gpu-a",
        _journal(
            status, *journal.ROTATION_STEPS, pending_token_cleanup_completed=False
        ),
    )

    with pytest.raises(ReleaseError) as failure:
        SAFETY.token_rotation_snapshot(_release(tmp_path))

    message = str(failure.value)
    assert "pending token cleanup" in message
    assert status in message
    assert f"--gpu-cluster-arn {GPU_A_ARN}{flag}" in message


def test_finished_or_absent_journals_pass(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        "gpu-a",
        _journal(
            journal.STATUS_COMPLETED,
            *journal.ROTATION_STEPS,
            pending_token_cleanup_completed=True,
        ),
    )

    snapshot = SAFETY.token_rotation_snapshot(_release(tmp_path))

    assert snapshot == {
        "state_dir": str(tmp_path),
        "clusters": {"gpu-a": journal.STATUS_COMPLETED, "gpu-b": "absent"},
    }


def test_every_blocked_cluster_is_named(tmp_path: Path) -> None:
    _write_journal(tmp_path, "gpu-a", _journal(journal.STATUS_IN_PROGRESS))
    _write_journal(
        tmp_path,
        "gpu-b",
        {**_journal(journal.STATUS_IN_PROGRESS), "cluster_id": "gpu-b"},
    )

    with pytest.raises(ReleaseError) as failure:
        SAFETY.token_rotation_snapshot(_release(tmp_path))

    message = str(failure.value)
    assert GPU_A_ARN in message
    assert GPU_B_ARN in message


@pytest.mark.parametrize(
    "content", ["not json", json.dumps([1, 2]), json.dumps({"status": "SOMETHING_NEW"})]
)
def test_an_unreadable_or_unknown_journal_fails_closed(
    tmp_path: Path, content: str
) -> None:
    path = journal.rotation_journal_path(tmp_path, "gpu-a")
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(Exception, match="rotate-token|SOMETHING_NEW|Expecting"):
        SAFETY.token_rotation_snapshot(_release(tmp_path))


def test_the_state_dir_reaches_the_engine_through_a_sidecar_not_the_config(
    tmp_path: Path,
) -> None:
    """The materialized config bytes feed identity digests, so the managed state
    directory travels in a 0600 sidecar beside them and nowhere else."""

    from gpu_fault_release import regional_release_config as config_module

    config = tmp_path / "work" / "regional-release.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    assert config_module.admin_state_dir_sidecar(config) is None

    sidecar = config.with_name(ADMIN_STATE_DIR_SIDECAR)
    state_dir = tmp_path / "state"
    sidecar.write_text(f"{state_dir}\n", encoding="utf-8")
    assert config_module.admin_state_dir_sidecar(config) == state_dir, (
        "an in-process engine may be built before the directory exists"
    )

    sidecar.write_text("relative/state", encoding="utf-8")
    with pytest.raises(ReleaseError, match="absolute"):
        config_module.admin_state_dir_sidecar(config)


def test_the_preflight_check_passes_warns_and_fails_through_the_report(
    tmp_path: Path,
) -> None:
    summary, details, status = SAFETY.token_rotation_check(_release(tmp_path))
    assert status == "PASS"
    assert details["clusters"] == {"gpu-a": "absent", "gpu-b": "absent"}
    assert "no unfinished token rotation" in summary

    summary, details, status = SAFETY.token_rotation_check(_release(None))
    assert status == "WARN", "an engine run without a managed state dir cannot look"
    assert details is None
    assert "state directory" in summary

    _write_journal(tmp_path, "gpu-a", _journal(journal.STATUS_IN_PROGRESS))
    with pytest.raises(ReleaseError, match="gpu-a"):
        SAFETY.token_rotation_check(_release(tmp_path))


def test_build_preflight_report_fails_closed_on_an_unfinished_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "_check_tools",
        "_check_local_inputs",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "check_aurora_refresh",
        "check_control_record_archive_bucket",
        "check_email_notifications",
        "_check_monitoring",
        "workflow_safety_snapshot",
    ):
        monkeypatch.setattr(
            CHECKS, name, lambda *args, name=name: CHECKS.CheckValue(name)
        )
    _write_journal(
        tmp_path,
        "gpu-a",
        _journal(journal.STATUS_IN_PROGRESS, *journal.ROTATION_STEPS[:-1]),
    )

    report = CHECKS.build_preflight_report(_release(tmp_path))

    (check,) = [item for item in report["checks"] if item["name"] == "token_rotation"]
    assert report["healthy"] is False
    assert check["status"] == "FAIL"
    assert "gpu-fault-admin rotate-token" in check["summary"]

    (tmp_path / journal.STATE_ROOT / "gpu-a" / "state.json").unlink()
    report = CHECKS.build_preflight_report(_release(tmp_path))
    (check,) = [item for item in report["checks"] if item["name"] == "token_rotation"]
    assert report["healthy"] is True
    assert check["status"] == "PASS"
