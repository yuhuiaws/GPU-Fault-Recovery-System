"""An abandoned upgrade repair over a committed release is retired by the next candidate.

Live on 2026-09-18 a candidate's Aurora repair reached READY and the deploy then
died at a workflow gate. Its release id never returned (a release id is a digest
of the tree), the repair journal stayed bound to it, and both the next deploy
and ``deploy --rollback`` refused with ``candidate binding differs``: the site
could not move. These tests pin the way out and its limits.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_prerequisite_baseline import (
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_prerequisite_handoff import (
    validate_upgrade_repair_handoff,
)
from tests.regional._prerequisite_repair_support import repair_release
from tests.regional.test_release_aurora_refresh_transaction import CANDIDATE, OLD

DIFF = diff_from_changed({"aurora_refresh_drift"})
PLAN = build_execution_plan(DIFF)
MUTATIONS = {"persist", "apply", "delete", "job-create", "job-delete"}
SUCCESSOR = "registry/runtime@sha256:" + "3" * 64


def prepare(instance: Any, **options: Any) -> dict[str, Any] | None:
    return REPAIR.prepare_prerequisite_repair(instance, diff=DIFF, plan=PLAN, **options)


def cronjob_image(live: dict[str, Any]) -> str | None:
    cronjob = live.get("cronjob.batch")
    if cronjob is None:
        return None
    pod = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return str(pod["containers"][0]["image"])


def committed(instance: Any) -> None:
    """The harness state as a committed release, with the fields a commit writes."""
    instance.state.update(release_lifecycle="COMMITTED", commit_cleanup_completed=True)
    instance.runner.cloud_state = copy.deepcopy(instance.state)


def abandon(instance: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Candidate A repairs to READY and dies; candidate B loads what A left.

    Returns the orphan journal and the state B loaded (the digest the deploy
    driver pins before preflight).
    """
    committed(instance)
    prepare(instance)
    orphan = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    assert orphan["status"] == "READY" and cronjob_image(instance.runner.live) == (
        CANDIDATE
    ), "the abandoned candidate did not leave its own refresher live"
    # B starts from the state A left behind, exactly as the deploy loads it.
    instance.state = copy.deepcopy(instance.runner.cloud_state)
    loaded = copy.deepcopy(instance.state)
    instance.release_id = "candidate-b"
    instance.runtime_image = SUCCESSOR
    instance.wheel_cm = "candidate-b-wheel"
    instance.rendered_manifest_digest = "b" * 64
    instance.approved_manifest_digest = None
    instance.runner.events.clear()
    return orphan, loaded


def record_images(monkeypatch: pytest.MonkeyPatch, instance: Any) -> list[str | None]:
    runner = instance.runner
    images: list[str | None] = []
    original_run = runner.run

    def run(args: list[str], **kwargs: Any) -> str:
        output = original_run(args, **kwargs)
        if "apply" in args and "--dry-run=server" not in args:
            images.append(cronjob_image(runner.live))
        return output

    monkeypatch.setattr(runner, "run", run)
    return images


def test_next_candidate_restores_original_refresher_then_takes_the_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    orphan, loaded = abandon(instance)
    images = record_images(monkeypatch, instance)
    successor = prepare(instance)
    events = instance.runner.events
    assert isinstance(successor, dict) and successor["status"] == "READY", (
        "the successor did not complete its own repair"
    )
    assert images[0] == OLD, (
        "the abandoned candidate's refresher was not restored before any candidate mutation"
    )
    assert events.index("apply") < events.index("rds-ca-bundle"), (
        "candidate resources moved before the original refresher was restored"
    )
    assert images[-1] == SUCCESSOR and instance.runner.jobs == {}, (
        "the successor's refresher is not live or its proof Job survived"
    )
    assert successor["binding"]["release_id"] == "candidate-b", (
        "the journal is not bound to the candidate that now owns it"
    )
    assert successor["binding"]["runtime_image"] == SUCCESSOR, (
        "the successor kept the predecessor's image binding"
    )
    assert successor["previous_refresher"] == orphan["previous_refresher"], (
        "the successor recaptured the abandoned candidate as its rollback snapshot"
    )
    assert (
        cronjob_image(
            {
                "cronjob.batch": next(
                    item
                    for item in successor["previous_refresher"]["objects"]
                    if item["kind"] == "CronJob"
                )
            }
        )
        == OLD
    ), "the carried rollback snapshot is not the committed release's refresher"
    assert successor["attempt_id"] != orphan["attempt_id"], (
        "the handoff rebound the abandoned attempt instead of opening a new journal"
    )
    assert successor["supersedes"]["attempt_id"] == orphan["attempt_id"], (
        "the successor lost its predecessor audit link"
    )
    assert successor["supersedes"]["release_id"] == "candidate", (
        "the audit does not name the abandoned candidate"
    )
    assert successor["supersedes"]["state_sha256"] == canonical_sha256(loaded), (
        "the audit does not carry the state digest the deploy diff was pinned to"
    )
    assert successor["supersedes"]["record_sha256"] == canonical_sha256(orphan), (
        "the predecessor audit digest is not of the cleaned original journal"
    )
    assert instance.state["release_id"] == "old", (
        "the handoff changed the live application identity"
    )
    assert validated_prerequisite_baseline(instance.state, successor)["release_id"] == (
        "old"
    ), "the successor's baseline is not the committed release"
    assert events.count("credential-job") == 1, (
        "the successor reused the abandoned candidate's credential proof"
    )
    assert not ({"schema", "cpu-finalize", "registry", "endpoint"} & set(events)), (
        "the prerequisite handoff performed an application mutation"
    )


def test_deploy_pin_accepts_the_state_the_diff_saw_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    _orphan, loaded = abandon(instance)
    successor = prepare(instance)
    assert isinstance(successor, dict), "an enabled repair did not return its journal"
    assert REPAIR.matches_pre_repair_state(
        instance, instance.state, canonical_sha256(loaded)
    ), "the apply step would refuse the deploy that performed the handoff"
    assert REPAIR.matches_pre_repair_state(
        instance, instance.state, successor["baseline_sha256"]
    ), "the successor's own baseline transition stopped matching"
    assert not REPAIR.matches_pre_repair_state(instance, instance.state, "0" * 64), (
        "an unrelated digest matched the recorded transition"
    )
    tampered = copy.deepcopy(instance.state)
    tampered[REPAIR.REPAIR_KEY]["supersedes"]["state_sha256"] = "0" * 64
    assert not REPAIR.matches_pre_repair_state(
        instance, tampered, canonical_sha256(loaded)
    ), "the pin trusted a digest the journal does not record"


@pytest.mark.parametrize("option", ["resume", "commit_live", "previous_override"])
def test_handoff_refuses_resume_supersede_and_commit_live(
    monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    instance = repair_release(monkeypatch)
    orphan, _loaded = abandon(instance)
    with pytest.raises(ReleaseError, match="candidate binding differs"):
        prepare(instance, **{option: {} if option == "previous_override" else True})
    assert not (MUTATIONS & set(instance.runner.events)), (
        "a refused handoff mutated the site"
    )
    assert instance.state[REPAIR.REPAIR_KEY] == orphan, (
        "a refused handoff changed the abandoned journal"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "application-in-flight",
        "orphan-committed",
        "baseline-drift",
        "database-drift",
        "site-drift",
        "bootstrap-binding",
        "snapshot-tampered",
        "unknown-status",
        "foreign-job",
    ],
)
def test_handoff_requires_the_exact_committed_baseline_site_and_journal(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    instance = repair_release(monkeypatch)
    _orphan, _loaded = abandon(instance)
    record = instance.state[REPAIR.REPAIR_KEY]
    if fault == "application-in-flight":
        instance.state.update(
            transaction_committed=False, previous={"release_id": "old"}
        )
    elif fault == "orphan-committed":
        instance.state["release_id"] = "candidate"
    elif fault == "baseline-drift":
        instance.state["unowned_field"] = "do-not-ignore"
    elif fault == "database-drift":
        instance.runner.database_resource_id = "another-cluster"
    elif fault == "site-drift":
        instance.config.namespace = "other-namespace"
    elif fault == "bootstrap-binding":
        record["binding"]["bootstrap"] = True
        record["binding_sha256"] = canonical_sha256(record["binding"])
    elif fault == "snapshot-tampered":
        record["previous_refresher"]["objects"][0]["metadata"]["labels"] = {"x": "y"}
    elif fault == "unknown-status":
        record["status"] = "UNKNOWN"
    else:
        record["jobs"]["credential"]["owner_uid"] = None
    instance.runner.cloud_state = copy.deepcopy(instance.state)
    before = copy.deepcopy(instance.state)
    with pytest.raises(ReleaseError):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        f"{fault}: a refused handoff mutated the site"
    )
    assert instance.state == before, f"{fault}: a refused handoff changed the state"
    if fault == "application-in-flight":
        # An application transaction owns the journal now; the standalone
        # restore defers to the application rollback instead of refusing.
        REPAIR.restore_standalone_prerequisite(instance)
    else:
        with pytest.raises(ReleaseError):
            REPAIR.restore_standalone_prerequisite(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        f"{fault}: a refused standalone restore mutated the site"
    )
    assert instance.state[REPAIR.REPAIR_KEY] == before[REPAIR.REPAIR_KEY], (
        f"{fault}: the standalone restore changed the abandoned journal"
    )


def test_restore_failure_keeps_the_abandoned_journal_for_the_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    orphan, _loaded = abandon(instance)
    runner = instance.runner
    original_run = runner.run
    failures = {"remaining": 1}

    def run(args: list[str], **kwargs: Any) -> str:
        if "apply" in args and "--dry-run=server" not in args and failures["remaining"]:
            failures["remaining"] -= 1
            raise ReleaseError("apiserver unavailable")
        return str(original_run(args, **kwargs))

    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(ReleaseError, match="apiserver unavailable"):
        prepare(instance)
    kept = instance.state[REPAIR.REPAIR_KEY]
    assert (
        kept["attempt_id"] == orphan["attempt_id"]
        and kept["binding"] == (orphan["binding"])
    ), "a failed restoration replaced or rebound the abandoned journal"
    assert kept["previous_refresher"] == orphan["previous_refresher"], (
        "a failed restoration lost the original rollback snapshot"
    )
    successor = prepare(instance)
    assert isinstance(successor, dict) and successor["status"] == "READY", (
        "the retry did not complete the handoff"
    )
    assert successor["supersedes"]["attempt_id"] == orphan["attempt_id"], (
        "the retry lost the predecessor audit link"
    )
    assert cronjob_image(runner.live) == SUCCESSOR, (
        "the retry did not leave the successor's refresher live"
    )


def test_standalone_rollback_restores_another_candidates_abandoned_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    committed(instance)
    original = copy.deepcopy(instance.runner.live)
    _orphan, _loaded = abandon(instance)
    status = copy.deepcopy(instance.runner.refresh_status)
    with pytest.raises(ReleaseError, match="rerun rollback"):
        REPAIR.restore_standalone_prerequisite(instance)
    assert instance.runner.live == original, (
        "the standalone rollback did not restore every original refresher object"
    )
    assert instance.runner.refresh_status == status, (
        "the standalone rollback touched database credentials"
    )
    assert REPAIR.REPAIR_KEY not in instance.runner.cloud_state, (
        "the abandoned journal survived its restoration"
    )
    assert instance.state["release_id"] == "old" and instance.state["phase"] == (
        "complete"
    ), "the standalone rollback changed the committed release identity"
    assert not (
        {"schema", "cpu-finalize", "registry", "endpoint"} & set(instance.runner.events)
    ), "the standalone prerequisite rollback rolled back the application"


def test_same_candidate_journal_keeps_its_own_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    committed(instance)
    prepare(instance)
    record = instance.state[REPAIR.REPAIR_KEY]
    identity = record["binding"]["database"]
    with pytest.raises(ReleaseError, match="handoff binding is invalid"):
        validate_upgrade_repair_handoff(instance, instance.state, identity=identity)
    record["binding_sha256"] = "wrong"
    with pytest.raises(ReleaseError, match="candidate or snapshot binding differs"):
        REPAIR.restore_prerequisite_repair(instance)
    assert "apply" not in instance.runner.events[-1:], (
        "a tampered same-candidate journal authorized a restoration"
    )
