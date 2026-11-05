"""GF-REGIONAL-BOOT-038 live driver against a fake site.

``LiveMembershipDriver`` spawns the admin verb (a fake process replaying the
real narration), follows its output, opens and restores the kubelet window on
the narration's triggers and reads the four finding surfaces. These tests walk
the paths the full remove/re-join never takes: plan B, a window that never
takes the node down, an overdue restore, the join's profile-plan stop, refused
ownership labels, unreadable registry, status and notification reads, the
after-run diff, a failed window close and the NotReady poller itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot038_membership as membership
from scripts.e2e.regional import run_boot038_cross_cluster_membership as runner
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_boot038_harness import (
    OTHER_NODE,
    TARGET_ID,
    World,
    threads_end_inside,
    verb_result,
)

REMOVE, JOIN = membership.REMOVE_VERB, membership.JOIN_VERB
DEPARTED = "[GPU node departed: operator action needed] hp: hp-node-1"


@pytest.fixture(autouse=True)
def threads_end_with_the_test() -> Iterator[None]:
    with threads_end_inside():
        yield


def world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str = membership.WINDOW_KUBELET,
) -> World:
    return World(tmp_path, monkeypatch, window=window)


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
def test_plan_b_opens_no_window_and_records_what_the_node_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch, window=membership.WINDOW_NONE)
    driver = site.driver()

    removal = driver.run_verb(REMOVE)
    site.failures["warm.snapshot"] = RegionalFixtureError("node read timed out")
    join = driver.run_verb(JOIN)

    assert removal.window_opened is False, removal
    assert removal.other_node_not_ready_before_verify is None, removal
    assert removal.window_observation == f"{OTHER_NODE} ready=True", removal
    assert set(removal.window_timeline) == {"verify_begin"}, removal.window_timeline
    assert join.window_observation == "unreadable (RegionalFixtureError)", join
    assert not any(name.startswith("window.") for name in site.names()), site.names()
    assert driver.end_windows() == {
        "ended": [],
        "resume_cleanup": [],
        "open_journals": [],
    }, "plan B leaves no window to end"


def test_a_window_that_never_takes_the_node_down_is_a_missed_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.stop_takes_node_down = False

    evidence = site.driver().run_verb(REMOVE)

    assert evidence.window_opened is True, evidence
    assert evidence.other_node_not_ready_before_verify is False, evidence
    assert "not_ready_seen" not in evidence.window_timeline, evidence.window_timeline
    assert membership.judge_verb(evidence)[0] == membership.VERDICT_INCONCLUSIVE
    assert site.names().count("window.cleanup") == 1, "the window is still restored"


def test_an_overdue_restore_resumes_the_window_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.node_stays_down = True

    evidence = site.driver().run_verb(REMOVE)

    assert (evidence.restore_ok, evidence.resume_cleanup_called) == (False, True)
    assert "restore" in evidence.window_timeline, evidence.window_timeline
    names = site.names()
    assert names.index("window.cleanup") < names.index("window.resume"), names
    verdict, reasons = membership.judge_verb(evidence)
    assert verdict == membership.VERDICT_FAIL, reasons
    waited = site.clock.elapsed
    assert waited >= runner.RESTORE_READY_TIMEOUT_SECONDS, (
        "the restore waits its whole bound on the virtual clock",
        waited,
    )


# --------------------------------------------------------------------------- #
# The join's prerequisites and its profile-plan stop
# --------------------------------------------------------------------------- #
def test_a_join_that_stopped_for_a_profile_plan_is_rerun_with_its_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.exit_codes[JOIN] = 2
    site.streams[JOIN] = site.streams[JOIN][: -len(verb_result())]
    site.pending_plan = {"plan_sha256": "f" * 64}
    site.rerun_stream = verb_result(phase="JOINED")

    evidence = site.driver().run_verb(JOIN)

    (rerun,) = site.details("verb.rerun")
    assert rerun[rerun.index("--approve-profile-plan") + 1] == "f" * 64, rerun
    assert rerun[rerun.index("--reference") + 1] == f"{runner.CASE_ID}/deadbeef"
    assert rerun[: rerun.index("--approve-profile-plan")] == runner.verb_command(
        site.settings, JOIN
    ), "the approval reruns the same join"
    assert evidence.exit_code == 0, "the rerun's exit code is the verb's"
    assert evidence.finding_surfaces["result"] is True, evidence.finding_surfaces
    record = site.evidence("join-ownership-reapply.json")
    assert record["reapply"] == {"reapplied": [site.settings.target_cluster_arn]}


def test_a_failed_join_without_a_profile_plan_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.exit_codes[JOIN] = 3

    evidence = site.driver().run_verb(JOIN)

    assert evidence.exit_code == 1, evidence
    assert site.details("profile-plan") == [str(site.state_dir)], site.calls
    assert "verb.rerun" not in site.names(), "nothing to approve, nothing rerun"


def test_a_join_whose_node_groups_lack_the_ownership_label_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.node_scope_status = "refused"

    with pytest.raises(RegionalFixtureError, match="do not carry the ownership label"):
        site.driver().run_verb(JOIN)

    assert "verb.spawn:join-cluster" not in site.names(), "the join is never spawned"
    scope = site.evidence("join-ownership-reapply.json")["node_scope_precondition"]
    assert scope["clusters"][0]["status"] == "refused", scope


def test_a_label_reapply_that_fails_is_a_closed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.failures["ownership.reapply"] = RegionalFixtureError("tombstone missing")

    with pytest.raises(
        RegionalFixtureError,
        match="re-applying the ownership label before the join failed: "
        "RegionalFixtureError: tombstone missing",
    ):
        site.driver().run_verb(JOIN)

    assert "node_scope.precondition" not in site.names(), site.names()


# --------------------------------------------------------------------------- #
# The surfaces and the registry convergence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("failure", "verdict", "converged"),
    [
        ("registry.inventory", "PASS", False),
        ("registry.check", "PASS", False),
        ("", "NOT_APPLICABLE", True),
        ("", "FAIL", False),
    ],
    ids=["baseline-unread", "check-unread", "not-applicable", "failed"],
)
def test_the_registry_convergence_fails_closed(
    failure: str,
    verdict: str,
    converged: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = world(tmp_path, monkeypatch)
    if failure:
        site.failures[failure] = RegionalFixtureError("CPU plane unreachable")
    site.registry_verdict = verdict

    evidence = site.driver().run_verb(REMOVE)

    assert evidence.registry_convergence_pass is converged, evidence
    baseline = site.case_dir / f"{REMOVE}.registry-baseline.json"
    assert baseline.exists() is (failure != "registry.inventory"), baseline
    check = site.case_dir / f"{REMOVE}.registry-check.json"
    assert check.exists() is (not failure), check


@pytest.mark.parametrize(
    ("failure", "returncode", "error"),
    [
        (OSError("admin CLI missing"), 0, "status --full could not run: OSError"),
        (None, 2, "status --full exited 2"),
    ],
    ids=["cannot-run", "nonzero"],
)
def test_a_status_read_that_did_not_succeed_is_an_evidence_error(
    failure: BaseException | None,
    returncode: int,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = world(tmp_path, monkeypatch)
    if failure is not None:
        site.failures["status"] = failure
    site.status_rc = returncode

    evidence = site.driver().run_verb(REMOVE)

    assert evidence.status_error == error, evidence
    assert evidence.finding_surfaces["status_lines"] is False, evidence


# --------------------------------------------------------------------------- #
# After the run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("baseline_read", [True, False], ids=["read", "unread"])
def test_the_after_run_reports_only_departed_mail_the_removal_sent(
    baseline_read: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.baseline_notifications = [{"subject": DEPARTED, "incident_id": "old"}]
    if not baseline_read:
        site.failures["store.notifications"] = RegionalFixtureError("store busy")
    site.after["notifications"] = [
        {"subject": DEPARTED, "incident_id": "departed-node-hp-node-1"},
        {"subject": "[NOTICE] unrelated", "incident_id": "x"},
    ]
    site.after["departed_ledger"] = [{"node_id": OTHER_NODE, "status": "PENDING"}]
    driver = site.driver()
    driver.run_verb(REMOVE)

    after = driver.read_after()

    expected = [] if baseline_read else [f"{DEPARTED} (hp-node-1)"]
    assert list(after.target_departed_mail) == expected, after
    assert after.other_node_ready is True and after.window_residue is False, after
    assert after.other_agent_retired is False, after
    assert list(after.other_node_ledger_mail) == [], "PENDING is not a mail"
    assert after.target_agents_all_active is True, after
    assert site.details("store.after") == [
        [site.settings.other_cluster_id, OTHER_NODE, TARGET_ID]
    ], site.calls


def test_ending_the_windows_resumes_a_failed_close_and_records_a_failed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    driver = site.driver()
    driver.run_verb(REMOVE)
    driver.run_verb(JOIN)
    site.failures[f"window.cleanup:{REMOVE}"] = RegionalFixtureError("exec refused")
    site.failures[f"window.cleanup:{JOIN}"] = RegionalFixtureError("exec refused")
    site.failures[f"window.resume:{JOIN}"] = RegionalFixtureError("journal locked")

    residue = driver.end_windows()

    assert residue["ended"] == [], residue
    assert residue["resume_cleanup"] == [
        {"resumed": f"boot038-deadbeef-{REMOVE}"},
        {"error": "RegionalFixtureError: journal locked"},
    ], residue
    assert residue["open_journals"] == [], residue
    assert site.names()[-2:] == ["window.resume", "window.close"], site.names()
    assert site.names().count("window.close") == 2, "every window is closed"


# --------------------------------------------------------------------------- #
# The NotReady poller
# --------------------------------------------------------------------------- #
class ScriptedWarm:
    """A Node read that answers ``ready`` or fails."""

    def __init__(self, ready: str | None) -> None:
        self.ready = ready
        self.reads = 0
        self.lock = threading.Lock()

    def node_snapshot(self, node: str) -> dict[str, Any]:
        with self.lock:
            self.reads += 1
        if self.ready is None:
            raise RegionalFixtureError(f"{node} unreadable")
        return {"name": node, "ready": self.ready}


def test_a_poller_whose_reads_fail_never_claims_not_ready() -> None:
    poller = runner.NotReadyPoller(ScriptedWarm(None), OTHER_NODE)

    assert poller.not_ready_now() is False, "an unreadable node is not NotReady"
    assert poller.not_ready_seen_at is None, poller


def test_a_started_poller_records_not_ready_until_it_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(membership, "NOT_READY_POLL_SECONDS", 0.01)
    warm = ScriptedWarm("Unknown")
    poller = runner.NotReadyPoller(warm, OTHER_NODE)
    seen = threading.Event()

    poller.start()
    for _attempt in range(200):
        if poller.not_ready_seen_at is not None and warm.reads >= 2:
            seen.set()
            break
        threading.Event().wait(0.01)
    poller.stop()

    assert seen.is_set(), "the poller kept reading and saw the node NotReady"
    warm.ready = "True"
    assert poller.not_ready_now() is False, "a node Ready again is no longer NotReady"
