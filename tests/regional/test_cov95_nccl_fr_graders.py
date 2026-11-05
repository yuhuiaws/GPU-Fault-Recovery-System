"""GF-GENERAL-LIVE-011 flight-recorder verdict stages over hand-built readings.

``test_general_live_nccl_fr`` runs every stage of ``general_live_nccl_fr`` over
the real product pipeline and the real test responder: that proves the PASS
path and a few perturbations, at the price of a product run per shape. Here a
``Reading`` is built straight from a replay shape and a faithful decision (the
fields the stages read, as ``hung_classification`` and ``bind_hung_bundle``
name them), the faithful reading is shown to pass, and then exactly one product
fact is broken per case, so every remaining FAIL and short-circuit path of the
stages names the defect it exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import general_live_fr_replay as replay
from scripts.e2e.regional import general_live_nccl_fr as nf

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
NODES = (NODE, PEER)
ALL_RANKS = tuple(range(replay.WORLD_SIZE))
#: The first node's ranks: what wave 1 samples when it already suffices.
NODE_RANKS = tuple(range(replay.PROCS_PER_NODE))
T0 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
NO_DECISION = "no hung_triage_decision on the bundle step or a triage execution"
NO_BRANCH = "the decision has no fr_temporal branch (no PAIRED series)"
FR_LEAD = {"rank": 0, "reasons": ["FR_PROGRESS_OUTLIER", "CPU_IDLE"]}

Change = Callable[[nf.Reading], None]


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def stamp(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def paired_series() -> dict[str, Any]:
    """Two dumps 2.5 s apart, the second requested after the first was read."""

    return {
        "status": "PAIRED",
        "first": {"status": "dumped"},
        "second": {"status": "dumped"},
        "interval_seconds": 2.5,
        "first_observed_at": stamp(0),
        "second_requested_at": stamp(2.5),
        "second_observed_at": stamp(2.6),
    }


def withheld_series() -> dict[str, Any]:
    """The first dump read; nothing answered the second request."""

    return {
        "status": "SECOND_UNAVAILABLE",
        "first": {"status": "dumped"},
        "second": {"status": "dump_missing"},
    }


def rank_row(rank: int, series: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "node_id": NODES[rank // replay.PROCS_PER_NODE],
        "pid": 100 + rank,
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": series,
    }


def membership(members: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()


def finding(group: dict[str, Any], sampled: Sequence[int]) -> dict[str, Any]:
    """The FR temporal finding the product forms for one expanded shape group."""

    kind = group["progress"]
    members = sorted(set(group["members"]) & set(sampled))
    enqueued = kind in (replay.ADVANCE, replay.ENQUEUE_ONLY)
    pending = kind in (replay.PENDING, replay.ENQUEUE_ONLY)
    return {
        "pg_name": group["name"],
        "expected_members": list(group["members"]),
        "observed_members": members,
        "membership_sha256": membership(group["members"]),
        "group_stall_observed": kind == replay.PENDING,
        "watermark_deltas_by_rank": {
            str(rank): dict(replay.DELTAS[kind]) for rank in members
        },
        "completion_advanced_ranks": members if kind == replay.ADVANCE else [],
        "enqueue_advanced_ranks": members if enqueued else [],
        "persistent_pending_ranks": members if pending else [],
    }


def conservative_bundle() -> dict[str, Any]:
    return {
        "operation": "COLLECT_DIAGNOSTIC_BUNDLE",
        "node_ids": list(NODES),
        "parameters": {
            "bundle_scope": nf.CONSERVATIVE_SCOPE,
            "capture_process_state": False,
            "operator_escalation_required": True,
            "target_pids_by_node": {},
            "max_processes": 0,
        },
    }


def temporal_reading(
    shape: replay.FlightRecorderReplay, ranks: Sequence[int] = ALL_RANKS
) -> nf.Reading:
    """A PAIRED reading of ``shape`` whose decision states exactly its progress."""

    view = nf.Reading(
        shape=shape,
        nodes=list(NODES),
        rows=[rank_row(rank, paired_series()) for rank in ranks],
        unreadable=[],
        decision=None,
        bundle=conservative_bundle(),
    )
    groups = view.groups()
    view.decision = {
        "classification": "INCONCLUSIVE",
        "culprit_ranks": [],
        "control_ranks": [],
        "candidate_count": 0,
        "candidate_processes": [],
        "targeted_capture_eligible": False,
        "stalled_process_groups": [
            group["name"]
            for group in groups
            if group["progress"] == replay.PENDING
            and len(group["members"]) < replay.WORLD_SIZE
        ],
        "evidence_branches": {
            "fr_temporal": {
                "process_groups": [finding(group, view.ranks) for group in groups]
            },
            "snapshot_authoritative": False,
        },
    }
    return view


def findings(view: nf.Reading) -> list[dict[str, Any]]:
    assert view.decision is not None, "the reading carries a decision"
    return view.decision["evidence_branches"]["fr_temporal"]["process_groups"]


def finding_named(view: nf.Reading, name: str) -> dict[str, Any]:
    return next(item for item in findings(view) if item["pg_name"] == name)


def without_finding(name: str) -> Change:
    def change(view: nf.Reading) -> None:
        findings(view)[:] = [item for item in findings(view) if item["pg_name"] != name]

    return change


def with_finding(name: str, **values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        finding_named(view, name).update(values)

    return change


def with_decision(**values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        assert view.decision is not None, "the reading carries a decision"
        view.decision.update(values)

    return change


def with_branch(**values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        assert view.decision is not None, "the reading carries a decision"
        view.decision["evidence_branches"].update(values)

    return change


def without_decision(view: nf.Reading) -> None:
    view.decision = None


def candidate_reading() -> nf.Reading:
    """few_candidates as wave 1 reads it: the first node sampled, rank 0 one
    collective behind its tp group, rank 1 the control, a targeted bundle."""

    view = temporal_reading(replay.few_candidates_shape(), NODE_RANKS)
    with_decision(
        classification="PLAUSIBLE",
        culprit_ranks=[0],
        control_ranks=[1],
        candidate_count=1,
        candidate_processes=[FR_LEAD],
        temporal_verified=True,
        membership_verified=True,
        targeted_capture_eligible=True,
    )(view)
    view.bundle = {
        "operation": "COLLECT_DIAGNOSTIC_BUNDLE",
        "node_ids": [NODE],
        "parameters": {
            "bundle_scope": nf.TARGETED_SCOPE,
            "capture_process_state": True,
            "operator_escalation_required": False,
            "target_pids_by_node": {NODE: [100, 101]},
            "max_processes": 2,
        },
    }
    return view


def withheld_reading() -> nf.Reading:
    """fr_second_missing: every second dump withheld, a snapshot-only decision."""

    return nf.Reading(
        shape=replay.second_missing_shape(),
        nodes=list(NODES),
        rows=[rank_row(rank, withheld_series()) for rank in ALL_RANKS],
        unreadable=[],
        decision={
            "classification": "INCONCLUSIVE",
            "culprit_ranks": [],
            "stalled_process_groups": [],
            "targeted_capture_eligible": False,
            "candidate_processes": [],
            "evidence_branches": {"fr_temporal": None, "snapshot_authoritative": True},
        },
        bundle=conservative_bundle(),
    )


def process_item(rank: int) -> dict[str, Any]:
    """A held rank rewriting checkpoints while its FR completion advances."""

    writes = {"code": nf.HOST_WRITES}
    return {
        "node_id": NODES[rank // replay.PROCS_PER_NODE],
        "pid": 100 + rank,
        "fr_completion_advanced_groups": ["0:default_pg", "1:tp"],
        "fr_persistent_pending_groups": [],
        "behavior": {
            "io_windows": [{"write_bytes_per_second": 4 * 1024 * 1024}],
            "observations": [writes],
        },
        "observations": [writes, {"code": "CPU_ACTIVITY"}],
    }


def process_reading() -> nf.Reading:
    view = temporal_reading(replay.plus_process_shape(), NODE_RANKS)
    with_branch(
        process_observations={"processes": [process_item(rank) for rank in (0, 1)]}
    )(view)
    return view


def processes(view: nf.Reading) -> list[dict[str, Any]]:
    assert view.decision is not None, "the reading carries a decision"
    return view.decision["evidence_branches"]["process_observations"]["processes"]


def with_process(index: int, **values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        processes(view)[index].update(values)

    return change


def with_behavior(index: int, **values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        processes(view)[index]["behavior"].update(values)

    return change


def with_bundle(**values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        view.bundle["parameters"].update(values)

    return change


# --------------------------------------------------------------------------- #
# lookups
# --------------------------------------------------------------------------- #
def test_finding_and_pid_lookups_answer_none_when_nothing_matches() -> None:
    view = temporal_reading(replay.whole_group_shape())
    assert view.decision is not None, "the builder states a decision"
    node_group = list(NODE_RANKS)
    assert nf.finding_for(view.decision, "1:tp", node_group) == finding_named(
        view, "1:tp"
    ), "the finding of a group is found by its name and full membership"
    assert nf.finding_for(view.decision, "1:tp", node_group[:-1]) is None, (
        "a finding with another membership is a different group"
    )
    assert nf.finding_for(view.decision, "9:tp", node_group) is None, (
        "a group the product never reported has no finding"
    )
    assert (
        nf.finding_for({"evidence_branches": {"fr_temporal": "?"}}, "1:tp", node_group)
        is None
    ), "a malformed temporal branch holds no finding"
    assert view.pid_of(3) == (NODE, 103), "a sampled rank is located by its row"
    assert view.pid_of(99) is None, "an unsampled rank has no PID"
    view.rows[3]["pid"] = "103"
    assert view.pid_of(3) is None, (
        "a PID the triage did not record as an integer is no PID"
    )


# --------------------------------------------------------------------------- #
# the two-dump series
# --------------------------------------------------------------------------- #
def series_change(**values: Any) -> Change:
    def change(view: nf.Reading) -> None:
        view.rows[2]["flight_recorder_series"].update(values)

    return change


def dump_status(point: str, status: str) -> Change:
    def change(view: nf.Reading) -> None:
        view.rows[2]["flight_recorder_series"][point]["status"] = status

    return change


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (dump_status("second", "dump_missing"), "the second dump is not 'dumped'"),
        (dump_status("first", "timeout"), "the first dump is not 'dumped'"),
        (series_change(second_observed_at=None), "the series lacks a timestamp"),
        (series_change(first_observed_at=""), "the series lacks a timestamp"),
        (
            series_change(second_requested_at=stamp(-1)),
            "series timestamps are out of order",
        ),
        (
            series_change(second_observed_at=stamp(2.4)),
            "series timestamps are out of order",
        ),
    ],
    ids=[
        "second-not-dumped",
        "first-not-dumped",
        "no-second-observation",
        "no-first-observation",
        "requested-before-the-first",
        "observed-before-requested",
    ],
)
def test_paired_errors_name_an_undumped_point_and_a_broken_timeline(
    change: Change, expected: str
) -> None:
    view = temporal_reading(replay.whole_group_shape())
    assert nf.paired_errors(view) == [], "a faithful PAIRED reading passes"
    change(view)
    assert nf.paired_errors(view) == [f"rank 2: {expected}"], expected


def test_withheld_errors_read_the_first_dump_and_the_missing_second_one() -> None:
    view = withheld_reading()
    assert nf.withheld_errors(view) == [], "a faithful withheld reading passes"
    view.rows[3]["flight_recorder"]["status"] = "dump_missing"
    view.rows[4]["flight_recorder_series"]["second"]["status"] = "dumped"
    assert nf.withheld_errors(view) == [
        "rank 3: the first dump was not read as dumped",
        "rank 4: the withheld second dump is not dump_missing",
    ], "a lost first dump and an answered second request are both named"
    empty = withheld_reading()
    empty.rows = []
    empty.unreadable = [f"{PEER}: the triage result does not decode"]
    assert nf.withheld_errors(empty) == [
        f"{PEER}: the triage result does not decode",
        "COLLECT_HUNG_TRIAGE sampled no ranked process",
    ], "nothing sampled is a FAIL that keeps the unreadable results"


# --------------------------------------------------------------------------- #
# leads and the bundle scope
# --------------------------------------------------------------------------- #
TARGETED_PIDS = f"{{'{NODE}': {{100, 101}}}}"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_decision, [NO_DECISION]),
        (
            with_decision(
                classification="INCONCLUSIVE",
                temporal_verified=False,
                membership_verified=None,
            ),
            [
                "classification 'INCONCLUSIVE', expected PLAUSIBLE",
                "temporal_verified is False",
                "membership_verified is None",
            ],
        ),
        (
            with_decision(control_ranks=[9]),
            [
                "control rank 9 was never sampled",
                "rank 9 has no PID in the triage rows",
                f"bundle targets {TARGETED_PIDS}, expected the candidate PIDs "
                f"{{'{NODE}': {{100}}}}",
                "bundle max_processes is 2",
            ],
        ),
        (
            with_bundle(operator_escalation_required=True),
            ["bundle operator_escalation_required is True"],
        ),
    ],
    ids=["no-decision", "not-plausible", "unsampled-control", "escalated-targeted"],
)
def test_candidate_errors_name_the_decision_and_scope_defects(
    change: Change, expected: list[str]
) -> None:
    view = candidate_reading()
    assert nf.candidate_errors(view, targeted=True) == [], (
        "a faithful targeted decision passes"
    )
    change(view)
    assert nf.candidate_errors(view, targeted=True) == expected, expected


def test_a_lead_without_a_recorded_pid_cannot_be_targeted() -> None:
    view = candidate_reading()
    view.rows[0]["pid"] = None
    assert nf.candidate_errors(view, targeted=True) == [
        "rank 0 has no PID in the triage rows",
        f"bundle targets {TARGETED_PIDS}, expected the candidate PIDs "
        f"{{'{NODE}': {{101}}}}",
        "bundle max_processes is 2",
    ], "the culprit's PID is owed by its triage row, not taken from the bundle"


def test_the_bundle_scope_check_needs_the_step_and_its_candidate_nodes() -> None:
    view = candidate_reading()
    view.bundle["node_ids"] = [NODE, PEER]
    assert nf.candidate_errors(view, targeted=True) == [
        f"bundle nodes {[NODE, PEER]} are not the candidates' {[NODE]}"
    ], "a targeted bundle runs on the candidates' nodes only"
    view.bundle = {}
    assert nf.candidate_errors(view, targeted=True) == [
        "no COLLECT_DIAGNOSTIC_BUNDLE step in the hung workflow"
    ], "a decision without its bundle step is a FAIL, not a pass"


# --------------------------------------------------------------------------- #
# watermarks, whole-group and separate-group stalls
# --------------------------------------------------------------------------- #
ENQUEUE_GROUP = "1:gfrs_enqueue_only"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_decision, [NO_DECISION]),
        (with_branch(fr_temporal=None), [NO_BRANCH]),
        (
            without_finding(ENQUEUE_GROUP),
            [f"no FR temporal finding for group {ENQUEUE_GROUP} {list(ALL_RANKS)}"],
        ),
        (with_decision(culprit_ranks=[3]), ["uniform progress yielded culprits [3]"]),
    ],
    ids=["no-decision", "no-temporal-branch", "unreported-group", "invented-culprit"],
)
def test_watermark_errors_need_a_finding_per_group_and_no_culprit(
    change: Change, expected: list[str]
) -> None:
    view = temporal_reading(replay.temporal_completion_shape())
    assert nf.watermark_errors(view) == [], "a faithful temporal decision passes"
    change(view)
    assert nf.watermark_errors(view) == expected, expected


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_finding("2:tp"), ["no FR temporal finding for group 2:tp"]),
        (
            with_finding("1:tp", persistent_pending_ranks=list(range(7))),
            ["1:tp: persistent_pending_ranks [0, 1, 2, 3, 4, 5, 6]"],
        ),
        (
            with_finding("0:default_pg", persistent_pending_ranks=[]),
            ["0:default_pg: persistent_pending_ranks [] != every sampled rank"],
        ),
        (
            without_finding("0:default_pg"),
            ["0:default_pg: persistent_pending_ranks [] != every sampled rank"],
        ),
        (
            with_decision(candidate_processes=[FR_LEAD]),
            ["a rank was marked FR_PROGRESS_OUTLIER in a whole-group stall"],
        ),
        (
            with_decision(targeted_capture_eligible=None),
            ["targeted_capture_eligible is not False"],
        ),
    ],
    ids=[
        "unreported-node-group",
        "node-group-not-all-pending",
        "job-group-not-pending",
        "unreported-job-group",
        "outlier-lead",
        "eligibility-unset",
    ],
)
def test_whole_group_errors_hold_every_sampled_rank_pending_without_a_lead(
    change: Change, expected: list[str]
) -> None:
    view = temporal_reading(replay.whole_group_shape())
    assert nf.whole_group_errors(view) == [], "a faithful whole-group decision passes"
    change(view)
    assert nf.whole_group_errors(view) == expected, expected


def test_a_shape_without_a_node_local_pending_group_has_nothing_to_observe() -> None:
    view = temporal_reading(replay.second_missing_shape())
    assert nf.whole_group_errors(view) == [
        "the shape has no node-local pending group to observe"
    ], "only the whole job pends: the node-group stall is not observable"


def spanning_group(view: nf.Reading) -> None:
    members = [6, 7, 8, 9]
    findings(view).append(
        {
            "pg_name": "9:mixed",
            "expected_members": members,
            "observed_members": members,
            "membership_sha256": membership(members),
        }
    )


def shared_digest(view: nf.Reading) -> None:
    finding_named(view, "2:tp")["membership_sha256"] = finding_named(view, "1:tp")[
        "membership_sha256"
    ]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_decision, [NO_DECISION]),
        (spanning_group, [f"9:mixed: a node group spans {[NODE, PEER]}"]),
        (without_finding("2:tp"), ["no FR temporal finding for group 2:tp"]),
        (
            with_finding("1:tp", group_stall_observed=False),
            ["1:tp: its own stall was not observed"],
        ),
        (shared_digest, ["two different node groups share one membership digest"]),
    ],
    ids=[
        "no-decision",
        "group-across-nodes",
        "unreported-group",
        "stall-unobserved",
        "shared-membership",
    ],
)
def test_separate_groups_errors_keep_each_node_group_apart(
    change: Change, expected: list[str]
) -> None:
    view = temporal_reading(replay.different_process_groups_shape())
    assert nf.separate_groups_errors(view) == [], "a faithful per-group decision passes"
    change(view)
    assert nf.separate_groups_errors(view) == expected, expected


# --------------------------------------------------------------------------- #
# a withheld second dump, FR progress beside process evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_decision, [NO_DECISION]),
        (
            with_branch(snapshot_authoritative=False),
            ["the single snapshot is not marked authoritative"],
        ),
        (with_decision(culprit_ranks=[4]), ["culprit ranks [4] from one dump"]),
        (
            with_decision(targeted_capture_eligible=None),
            ["targeted_capture_eligible is not False"],
        ),
        (
            with_decision(candidate_processes=[FR_LEAD]),
            ["FR candidate codes ['FR_PROGRESS_OUTLIER'] without a second dump"],
        ),
    ],
    ids=[
        "no-decision",
        "snapshot-not-authoritative",
        "culprit-from-one-dump",
        "eligibility-unset",
        "fr-lead-from-one-dump",
    ],
)
def test_second_missing_errors_invent_nothing_from_one_dump(
    change: Change, expected: list[str]
) -> None:
    view = withheld_reading()
    assert nf.second_missing_errors(view) == [], (
        "a faithful snapshot-only decision passes"
    )
    change(view)
    assert nf.second_missing_errors(view) == expected, expected


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (without_decision, [NO_DECISION]),
        (
            with_branch(process_observations={"processes": []}),
            ["the decision carries no process observations"],
        ),
        (
            with_process(0, fr_persistent_pending_groups=["1:tp"]),
            [f"{NODE}/100: a pending group ['1:tp']"],
        ),
        (
            with_behavior(1, io_windows=[]),
            [f"{NODE}/101: no process I/O windows kept beside the FR branch"],
        ),
        (
            with_behavior(1, io_windows={"write_bytes_per_second": 1}),
            [f"{NODE}/101: no process I/O windows kept beside the FR branch"],
        ),
        (
            with_decision(culprit_ranks=[2]),
            ["advancing FR progress yielded culprits [2]"],
        ),
        (
            with_decision(candidate_processes=[FR_LEAD]),
            ["FR candidate codes ['FR_PROGRESS_OUTLIER'] on advancing progress"],
        ),
    ],
    ids=[
        "no-decision",
        "no-processes",
        "pending-group",
        "empty-io-windows",
        "io-windows-not-a-list",
        "invented-culprit",
        "fr-lead",
    ],
)
def test_process_evidence_errors_keep_the_io_evidence_beside_fr_progress(
    change: Change, expected: list[str]
) -> None:
    view = process_reading()
    assert nf.process_evidence_errors(view) == [], (
        "a faithful decision with both evidence branches passes"
    )
    change(view)
    assert nf.process_evidence_errors(view) == expected, expected
