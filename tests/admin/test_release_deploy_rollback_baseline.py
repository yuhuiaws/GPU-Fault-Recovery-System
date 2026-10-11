"""The rollback baseline is per attempt and follows the committed site.

Live incident: a cluster joined after a release was first deployed vanished
from the management site when a later ``config`` re-roll of that release failed
and its automatic rollback rebuilt ``site.yaml`` from the release directory's
write-once ``site.before.yaml`` (one cluster). The baseline is now recorded at
every attempt start from the committed site, and kept only while a transaction
is being resumed or after a rollback whose alignment did not run through.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.rollback_baseline import (
    BASELINE_FILE,
    baseline_refresh_decision,
    record_rollback_baseline,
)
from scripts import release_deploy
from tests.admin.test_release_deploy import _site

COMMITTED = {
    "release_id": "release-a",
    "phase": "complete",
    "transaction_committed": True,
    "release_lifecycle": "COMMITTED",
}
ROLLED_BACK = {
    "release_id": "release-a",
    "phase": "rolled-back",
    "rollback_result": {"status": "PASSED"},
    "rollback_cleanup_completed": True,
}
MID_TRANSACTION = {"release_id": "release-a", "phase": "data-plane-progress"}


def _plan(site: Path, monkeypatch: pytest.MonkeyPatch):
    """The Runtime Profile plan of an unchanged profile: the live digest is the
    digest of whatever source the site declares (the repository template on the
    first attempt, the immutable snapshot afterwards)."""

    del monkeypatch
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    source = Path(str(document["spec"]["runtimeProfile"]["source"]))
    if not source.is_absolute():
        source = site.parent / "repo" / source
    return release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )


def _join_cluster(site: Path, cluster_id: str) -> None:
    """What join-cluster commits: one more entry under spec.clusters."""

    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["clusters"].append(
        {
            **document["spec"]["clusters"][0],
            "clusterId": cluster_id,
            "context": cluster_id,
            "hyperpodClusterName": f"hp-{cluster_id}",
            "eksClusterArn": f"arn:aws:eks:us-east-1:123456789012:cluster/{cluster_id}",
        }
    )
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _cluster_ids(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [item["clusterId"] for item in document["spec"]["clusters"]]


@pytest.mark.parametrize(
    ("live_state", "previous_record", "expected"),
    [
        (COMMITTED, None, True),
        (ROLLED_BACK, None, True),
        ({}, None, True),
        (MID_TRANSACTION, None, False),
        ({**COMMITTED, "transaction_committed": False}, None, False),
        ({**COMMITTED, "commit_cleanup_completed": False}, None, False),
        ({**ROLLED_BACK, "rollback_cleanup_completed": False}, None, False),
        ({**ROLLED_BACK, "rollback_result": {"status": "FAILED"}}, None, False),
        ({"phase": "rollback-data-restoring"}, None, False),
        (None, None, False),
        (ROLLED_BACK, {"rollback": {"status": "FAILED", "error": "alignment"}}, False),
        (
            ROLLED_BACK,
            {
                "rollback": {
                    "status": "ALIGNMENT_FAILED",
                    "management_state_synced": False,
                }
            },
            False,
        ),
        (ROLLED_BACK, {"rollback": {"status": "CLEANUP_PENDING"}}, False),
        (
            ROLLED_BACK,
            {"rollback": {"status": "PASSED", "management_state_synced": True}},
            True,
        ),
    ],
)
def test_baseline_refresh_follows_the_committed_or_rolled_back_live_state(
    live_state, previous_record, expected
) -> None:
    refresh, reason = baseline_refresh_decision(
        live_state, previous_record=previous_record
    )
    assert refresh is expected, reason
    assert len(reason) > 20, "the decision explains itself for the plan record"


def test_first_attempt_records_the_committed_site(tmp_path: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("spec:\n  clusters: []\n", encoding="utf-8")
    state_dir = tmp_path / "release-deploy" / "release-a"
    state_dir.mkdir(parents=True)

    evidence = record_rollback_baseline(state_dir, site, live_state=None)

    baseline = state_dir / BASELINE_FILE
    assert evidence["action"] == "written"
    assert baseline.read_bytes() == site.read_bytes()
    assert baseline.stat().st_mode & 0o777 == 0o600
    assert evidence["sha256"] == hashlib.sha256(site.read_bytes()).hexdigest()


def test_a_committed_live_release_refreshes_the_baseline_and_keeps_the_old_copy(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("spec:\n  clusters:\n  - clusterId: gpu-a\n", encoding="utf-8")
    state_dir = tmp_path / "release-deploy" / "release-a"
    state_dir.mkdir(parents=True)
    first = record_rollback_baseline(state_dir, site, live_state=COMMITTED)
    stale = (state_dir / BASELINE_FILE).read_bytes()
    _join_cluster_minimal(site, "hp-b")

    second = record_rollback_baseline(state_dir, site, live_state=COMMITTED)

    assert second["action"] == "refreshed", second
    assert (state_dir / BASELINE_FILE).read_bytes() == site.read_bytes()
    assert second["superseded_sha256"] == first["sha256"]
    superseded = Path(second["superseded_copy"])
    assert superseded.parent == state_dir
    assert superseded.read_bytes() == stale, "the stale baseline is kept for audit"
    assert superseded.stat().st_mode & 0o777 == 0o600

    third = record_rollback_baseline(state_dir, site, live_state=COMMITTED)
    assert third["action"] == "kept"
    assert "already equals" in third["reason"]
    assert len(list(state_dir.glob("site.before.superseded-*.yaml"))) == 1, (
        "an unchanged site archives nothing"
    )


def _join_cluster_minimal(site: Path, cluster_id: str) -> None:
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["clusters"].append({"clusterId": cluster_id})
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("live_state", "record"),
    [
        (MID_TRANSACTION, None),
        (None, None),
        (ROLLED_BACK, {"phase": "FAILED", "rollback": {"status": "FAILED"}}),
        (
            COMMITTED,
            {
                "phase": "ROLLED_BACK",
                "rollback": {
                    "status": "ALIGNMENT_FAILED",
                    "management_state_synced": False,
                },
            },
        ),
    ],
)
def test_the_baseline_is_kept_while_resuming_or_after_an_unaligned_rollback(
    tmp_path: Path, live_state, record
) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("spec:\n  clusters:\n  - clusterId: gpu-a\n", encoding="utf-8")
    state_dir = tmp_path / "release-deploy" / "release-a"
    state_dir.mkdir(parents=True)
    record_rollback_baseline(state_dir, site, live_state=COMMITTED)
    stale = (state_dir / BASELINE_FILE).read_bytes()
    if record is not None:
        (state_dir / "state.json").write_text(json.dumps(record), encoding="utf-8")
    _join_cluster_minimal(site, "hp-b")

    evidence = record_rollback_baseline(state_dir, site, live_state=live_state)

    assert evidence["action"] == "kept", evidence
    assert (state_dir / BASELINE_FILE).read_bytes() == stale
    assert list(state_dir.glob("site.before.superseded-*.yaml")) == []


def test_prepare_site_release_refreshes_the_baseline_after_a_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident, replayed: first deploy with one cluster, join a second,
    then re-roll the same release. The attempt's baseline must list both."""

    site = _site(tmp_path, monkeypatch)
    first = release_deploy.prepare_site_release(
        site, profile_plan=_plan(site, monkeypatch), live_state={}
    )
    assert _cluster_ids(first.state_dir / BASELINE_FILE) == ["gpu-a"]
    _join_cluster(site, "hp-cluster-b")

    second = release_deploy.prepare_site_release(
        site, profile_plan=_plan(site, monkeypatch), live_state=COMMITTED
    )

    assert second.state_dir == first.state_dir, "the live release directory is reused"
    assert _cluster_ids(second.state_dir / BASELINE_FILE) == ["gpu-a", "hp-cluster-b"]
    plan = json.loads((second.state_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["rollback_baseline"]["action"] == "refreshed"
    assert plan["rollback_baseline"]["path"] == str(second.state_dir / BASELINE_FILE)
    assert "committed" in plan["rollback_baseline"]["reason"]
    state = json.loads((second.state_dir / "state.json").read_text(encoding="utf-8"))
    assert state["rollback_baseline"] == plan["rollback_baseline"]
    superseded = Path(plan["rollback_baseline"]["superseded_copy"])
    assert _cluster_ids(superseded) == ["gpu-a"]


def test_prepare_site_release_keeps_the_baseline_of_a_resumed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    first = release_deploy.prepare_site_release(
        site, profile_plan=_plan(site, monkeypatch), live_state={}
    )
    _join_cluster(site, "hp-cluster-b")

    resumed = release_deploy.prepare_site_release(
        site, profile_plan=_plan(site, monkeypatch), live_state=MID_TRANSACTION
    )

    assert _cluster_ids(resumed.state_dir / BASELINE_FILE) == ["gpu-a"]
    plan = json.loads((first.state_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["rollback_baseline"]["action"] == "kept"
    assert "mid-transaction" in plan["rollback_baseline"]["reason"]


def test_prepare_site_release_without_a_live_state_keeps_the_recorded_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers that did not read the engine state (the historical signature)
    never replace a baseline they cannot judge."""

    site = _site(tmp_path, monkeypatch)
    release_deploy.prepare_site_release(site, profile_plan=_plan(site, monkeypatch))
    _join_cluster(site, "hp-cluster-b")

    repeated = release_deploy.prepare_site_release(
        site, profile_plan=_plan(site, monkeypatch)
    )

    assert _cluster_ids(repeated.state_dir / BASELINE_FILE) == ["gpu-a"]
