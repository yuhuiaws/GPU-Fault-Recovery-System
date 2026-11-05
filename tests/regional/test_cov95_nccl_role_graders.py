"""GF-GENERAL-LIVE-011 role graders over the triage the product got wrong.

``test_general_live_nccl_roles`` proves each grader accepts a faithful triage
and names the headline defect of its variant. These cases take the same rows
(built the way the Node Agent emits them, classified by the product's own
``classify_hung_signals``) and break one more field at a time -- a thread that
maps to no host tid, a window measured against the wrong clock, a cohort that
holds a process of another role, a holder py-spy could sample -- and check that
the grader names exactly that defect, and that a rank the trainer never
declared, a node the product never sampled or a process without a report is
skipped instead of judged.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_nccl_role_trainer as plan
from scripts.e2e.regional import general_live_nccl_roles as roles
from scripts.e2e.regional import general_live_nccl_triage_view as view
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional.test_general_live_nccl import NODE, PEER, FakeNcclJob, Traffic
from tests.regional.test_general_live_nccl_roles import (
    ATTEMPT,
    JOB,
    AttemptProduct,
    FakeSecondAttempt,
    account_of,
    host_pid,
    rank_row,
    run_attempt_change,
    scenario,
)

NODES = (NODE, PEER)
Rows = list[dict[str, Any]]


def label(row: dict[str, Any]) -> str:
    return f"{row['node_id']} rank {row['rank']} pid {row['pid']}"


def by_rank(rows: Rows, rank: int) -> dict[str, Any]:
    return next(row for row in rows if row["rank"] == rank)


def report_of(reports: Rows, rank: int, node: str = NODE) -> dict[str, Any]:
    """The decision's own report object (``report_for`` hands out a copy)."""

    return next(
        item for item in reports if item["rank"] == rank and item["node_id"] == node
    )


def graded(mode: str) -> tuple[Rows, dict[int, view.Declaration], Rows, Rows]:
    """A faithful triage of ``mode``: rows, declarations, reports, cohorts."""

    rows, decision = scenario(mode)
    decls = view.declarations(account_of(mode))
    return rows, decls, view.process_reports(decision), view.cohorts(decision)


def stack_thread(row: dict[str, Any], sample: int, name: str) -> dict[str, Any]:
    threads = row["stack_series"]["samples"][sample]["threads"]
    return next(item for item in threads if item["name"] == name)


# --------------------------------------------------------------------------- #
# coverage and context
# --------------------------------------------------------------------------- #
def test_a_node_the_product_never_sampled_owes_no_ranks() -> None:
    rows, _decision = scenario(plan.ROLES)
    master = [row for row in rows if row["node_id"] == NODE]
    assert roles.coverage_errors(master, NODES) == [], (
        "the product chose to sample only the master; the worker owes nothing"
    )
    assert roles.coverage_errors(
        [row for row in master if row["rank"] != 3], NODES
    ) == [f"{NODE}: ranks [3] were not sampled"], (
        "a sampled node still owes every one of its ranks"
    )


def test_context_errors_skip_an_unread_rank_and_name_a_trainer_that_left_the_plan() -> (
    None
):
    rows, decls, _reports, _groups = graded(plan.ROLES)
    by_rank(rows, 2)["comparison_context"]["pipeline_stage"] = "pp9"
    unread = {rank: decl for rank, decl in decls.items() if rank != 2}
    assert (
        roles.context_errors(
            rows, unread, mode=plan.ROLES, job_id=JOB, attempt_id=ATTEMPT
        )
        == []
    ), "a rank whose ROLES line never reached the log is not judged"
    expected = plan.expected_context(plan.ROLES, 0, world_size=roles.WORLD_SIZE)
    assert expected is not None, "local rank 0 declares a pipeline stage"
    drifted = {
        **unread,
        0: dataclasses.replace(decls[0], group="pp9-expert-node0"),
        1: dataclasses.replace(decls[1], members=(0, 1, 2)),
    }
    assert roles.context_errors(
        rows, drifted, mode=plan.ROLES, job_id=JOB, attempt_id=ATTEMPT
    ) == [
        f"{label(by_rank(rows, 0))}: the trainer declared pp9-expert-node0 "
        f"(0, 1, 2, 3), the plan says {expected['group_name']} {expected['members']}",
        f"{label(by_rank(rows, 1))}: the trainer declared {expected['group_name']} "
        f"(0, 1, 2), the plan says {expected['group_name']} {expected['members']}",
    ], "a trainer that declared another group than the plan's is named"


# --------------------------------------------------------------------------- #
# thread identity
# --------------------------------------------------------------------------- #
def test_host_thread_errors_skip_undeclared_ranks() -> None:
    rows, decls, _reports, _groups = graded(plan.ROLES)
    assert not decls[7].declared and not decls[15].declared, (
        "local rank 7 declares no role in the roles mode"
    )
    by_rank(rows, 7)["comparison_context"]["host_thread_roles"] = {"1": "TRAINING"}
    assert roles.host_thread_errors(rows, decls) == [], (
        "an undeclared rank's host_thread_roles are not judged against a declaration"
    )


def test_host_thread_errors_name_unmapped_and_undeclared_host_tids() -> None:
    decls = view.declarations(account_of(plan.THREADS))
    row = rank_row(plan.THREADS, 0, flat_namespace=True)
    errors = roles.host_thread_errors([row], decls)
    natives = sorted(decls[0].roles.items())
    assert errors[: len(natives)] == [
        f"{label(row)}: native tid {native} ({name}) maps to host tids []"
        for native, (_role, name) in natives
    ], "without an NSpid boundary no native tid maps to a host thread"
    hosts = sorted(row["comparison_context"]["host_thread_roles"])
    assert errors[len(natives) :] == [
        f"{label(row)}: host_thread_roles names undeclared tids {hosts}"
    ], "every role the product kept names a host tid no declaration maps to"


def test_host_thread_errors_name_a_training_thread_that_is_not_the_pid() -> None:
    rows, decls, _reports, _groups = graded(plan.THREADS)
    row = by_rank(rows, 1)
    main, comm = str(decls[1].pid), decls[1].tids("COMMUNICATION")[0]
    swapped = dict(decls[1].roles)
    swapped[main], swapped[comm] = swapped[comm], swapped[main]
    decls = {**decls, 1: dataclasses.replace(decls[1], roles=swapped)}
    pid = host_pid(1)
    assert roles.host_thread_errors(rows, decls) == [
        f"{label(row)}: host tid {pid} (gfrs-comm) reads TRAINING, declared COMMUNICATION",
        f"{label(row)}: host tid {pid + 1} (MainThread) reads COMMUNICATION, "
        "declared TRAINING",
        f"{label(row)}: the training thread is host tid {pid + 1}, not the pid",
    ], "a training thread declared on a helper's tid is not the process's own pid"


def test_stack_identity_errors_name_missing_and_misattributed_stack_threads() -> None:
    rows, decls, _reports, _groups = graded(plan.ROLES)
    assert roles.stack_identity_errors(rows, decls) == [], (
        "the faithful roles triage passes, its undeclared rank 7 skipped"
    )
    for sample in by_rank(rows, 1)["stack_series"]["samples"]:
        sample["status"], sample["returncode"] = "FAILED", 1
    blind = by_rank(rows, 2)["stack_series"]["samples"][0]
    blind["threads"] = [item for item in blind["threads"] if item["role"] != "TRAINING"]
    keeper = stack_thread(by_rank(rows, 3), 1, "gfrs-keeper")
    keeper["role"] = "COMMUNICATION"
    errors = roles.stack_identity_errors(rows, decls)
    assert errors == [
        f"{label(by_rank(rows, 1))}: no collected stack sample",
        f"{label(by_rank(rows, 2))}: stack 0 has 0 training threads",
        f"{label(by_rank(rows, 3))}: stack 1 thread {keeper['os_thread_id']} reads "
        "COMMUNICATION from the application, declared ('BACKGROUND', 'gfrs-keeper')",
    ], errors


def test_stability_errors_compare_stable_threads_with_the_kernel_samples() -> None:
    rows, decls, reports, _groups = graded(plan.ROLES)
    assert roles.stability_errors(rows, decls, reports) == [], (
        "the faithful roles triage passes, its undeclared rank 7 skipped"
    )
    rows, decls, reports, _groups = graded(plan.THREADS)
    assert roles.stability_errors(rows, decls, []) == [], (
        "a process without a report in the decision has no stable threads to judge"
    )
    shifted = report_of(reports, 1)["stack"]["stable_threads"][0]
    shifted["thread_start_ticks"] += 7
    relabelled = next(
        item
        for item in report_of(reports, 2)["stack"]["stable_threads"]
        if item["name"] == "gfrs-comm"
    )
    relabelled["role"] = "BACKGROUND"
    errors = roles.stability_errors(rows, decls, reports)
    assert errors == [
        f"{label(by_rank(rows, 1))}: stable thread {shifted['host_thread_id']} start "
        f"{shifted['thread_start_ticks']} against sampled starts "
        f"['{shifted['thread_start_ticks'] - 7}']",
        f"{label(by_rank(rows, 2))}: stable thread {relabelled['host_thread_id']} "
        "reads BACKGROUND, declared COMMUNICATION",
    ], errors


# --------------------------------------------------------------------------- #
# CPU windows (the shared normalization check)
# --------------------------------------------------------------------------- #
def drop_window(row: dict[str, Any]) -> None:
    del row["process_series"]["windows"][1]


def partial_window(row: dict[str, Any]) -> None:
    row["process_series"]["windows"][0]["status"] = "PARTIAL"


def foreign_clock(row: dict[str, Any]) -> None:
    row["process_series"]["windows"][1]["clock_ticks_per_second"] = 250


def short_window(row: dict[str, Any]) -> None:
    row["process_series"]["windows"][0]["elapsed_seconds"] = 0.1


def no_delta(row: dict[str, Any]) -> None:
    row["process_series"]["windows"][1]["cpu_ticks_delta"] = None


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (drop_window, "1 process windows, expected 2"),
        (partial_window, "window 0 is PARTIAL"),
        (foreign_clock, "window 1 uses 250 ticks/s, the node has 100"),
        (short_window, "window 0 elapsed 0.1 s is not a measured interval"),
        (no_delta, "window 1 has no CPU delta (None, 0.9)"),
    ],
)
def test_cpu_windows_must_be_measured_intervals_at_the_node_clock(
    defect: Callable[[dict[str, Any]], None], message: str
) -> None:
    rows, decls, reports, _groups = graded(plan.CPU)
    sharers = view.children(account_of(plan.CPU), plan.SHARER_MARKER)
    row = by_rank(rows, 3)
    defect(row)
    errors = roles.cpu_errors(rows, decls, reports, sharers)
    assert f"{label(row)}: {message}" in errors, errors
    assert all(error.startswith(label(row)) for error in errors), (
        "only the broken rank is named"
    )


# --------------------------------------------------------------------------- #
# wchan_zero
# --------------------------------------------------------------------------- #
def test_wchan_grader_needs_one_spinner_and_one_sleeper_mapped_to_host_threads() -> (
    None
):
    rows, decls, reports, _groups = graded(plan.ROLES)
    assert roles.wchan_errors(rows, decls, reports) == [
        f"{label(row)}: declared 0 spinner(s), 0 sleeper(s)"
        for row in sorted(rows, key=lambda item: (item["node_id"], item["rank"]))
        if decls[row["rank"]].declared
    ], "a mode without the helpers has nothing to grade; undeclared ranks skipped"
    flat, decision = scenario(plan.WCHAN, flat_namespace=True)
    errors = roles.wchan_errors(
        flat, view.declarations(account_of(plan.WCHAN)), view.process_reports(decision)
    )
    assert len(errors) == roles.WORLD_SIZE and all(
        error.endswith(": spinner maps to [], sleeper to []") for error in errors
    ), errors


def test_wchan_grader_reads_the_kernel_samples_of_both_helpers() -> None:
    rows, decls, reports, _groups = graded(plan.WCHAN)
    unread = {rank: decl for rank, decl in decls.items() if rank != 5}
    by_rank(rows, 5)["process_series"]["samples"][0]["threads"].clear()
    assert roles.wchan_errors(rows, unread, reports) == [], (
        "a rank without a ROLES line is not graded, whatever its samples read"
    )
    rows, decls, reports, _groups = graded(plan.WCHAN)
    row = by_rank(rows, 0)
    spin = view.host_tids(row, decls[0].tids("COMMUNICATION", "gfrs-spinner")[0])[0]
    sleep = view.host_tids(row, decls[0].tids("BACKGROUND", "gfrs-sleeper")[0])[0]
    samples = row["process_series"]["samples"]
    del samples[1]["threads"][spin], samples[1]["threads"][sleep]
    samples[0]["threads"][spin]["state"] = "S"
    samples[2]["threads"][spin]["state"] = "D"
    samples[2]["threads"][sleep]["wchan"] = "do_nanosleep"
    errors = roles.wchan_errors(rows, decls, reports)
    assert errors == [
        f"{label(row)}: the spinner was never sampled running (states ['D', 'S'])",
        f"{label(row)}: the sleeper's wait channel reads "
        "['do_nanosleep', 'futex_wait_queue']",
    ], errors


def test_wchan_grader_reads_the_windows_and_the_decisions_waits() -> None:
    rows, decls, reports, _groups = graded(plan.WCHAN)
    hosts = {}
    for rank in (1, 2, 3, 4):
        row = by_rank(rows, rank)
        hosts[rank] = (
            view.host_tids(row, decls[rank].tids("COMMUNICATION", "gfrs-spinner")[0])[
                0
            ],
            view.host_tids(row, decls[rank].tids("BACKGROUND", "gfrs-sleeper")[0])[0],
        )
    spin, sleep = hosts[1]
    windows = by_rank(rows, 1)["process_series"]["windows"]
    windows[0]["wait_threads"].append({"tid": int(spin), "same_wait": True})
    windows[1]["wait_threads"] = [
        item for item in windows[1]["wait_threads"] if str(item["tid"]) != sleep
    ]
    observations = report_of(reports, 2)["behavior"]["observations"]
    observations.append(
        {"code": roles.THREAD_WAIT, "thread_id": int(hosts[2][0]), "role": "X"}
    )
    report_of(reports, 3)["behavior"]["observations"] = [
        item
        for item in report_of(reports, 3)["behavior"]["observations"]
        if str(item.get("thread_id")) != hosts[3][1]
    ]
    for item in report_of(reports, 4)["behavior"]["observations"]:
        if str(item.get("thread_id")) == hosts[4][1]:
            item["candidate"] = True
    errors = roles.wchan_errors(rows, decls, reports)
    assert errors[:2] == [
        f"{label(by_rank(rows, 1))}: window 0 lists the spinner as a wait thread",
        f"{label(by_rank(rows, 1))}: window 1 misses the sleeper's repeated wait",
    ], errors
    assert errors[2:4] == [
        f"{label(by_rank(rows, 2))}: the decision reports a repeated wait for the spinner",
        f"{label(by_rank(rows, 3))}: the decision has no repeated wait for the sleeper",
    ], errors
    assert len(errors) == 5 and errors[4].startswith(
        f"{label(by_rank(rows, 4))}: the sleeper's wait is {{"
    ), errors
    assert errors[4].endswith("not a BACKGROUND non-candidate") and (
        "'candidate': True" in errors[4]
    ), errors


# --------------------------------------------------------------------------- #
# cpu_interval
# --------------------------------------------------------------------------- #
def test_cpu_grader_needs_both_sample_stamps_to_check_an_interval() -> None:
    rows, decls, reports, _groups = graded(plan.CPU)
    sharers = view.children(account_of(plan.CPU), plan.SHARER_MARKER)
    row = by_rank(rows, 2)
    row["process_series"]["windows"][1]["elapsed_seconds"] = 3.0
    interval = f"{label(row)}: window 1 elapsed 3.0 s against samples 1.00 s apart"
    assert interval in roles.cpu_errors(rows, decls, reports, sharers), (
        "a window measured over three seconds between samples one second apart"
    )
    row["process_series"]["samples"][2]["observed_at"] = None
    errors = roles.cpu_errors(rows, decls, reports, sharers)
    assert interval not in errors and errors, (
        "without the second sample's stamp the interval is not compared, the "
        f"normalization still is: {errors}"
    )
    unread = {rank: decl for rank, decl in decls.items() if rank != 2}
    assert roles.cpu_errors(rows, unread, reports, sharers) == [], (
        "a rank without a ROLES line is not graded"
    )


def scope_lost(rows: Rows, reports: Rows) -> dict[str, Any]:
    by_rank(rows, 1)["gpu_series"]["scope"] = "PID"
    return by_rank(rows, 1)


def mapping_unverified(rows: Rows, reports: Rows) -> dict[str, Any]:
    by_rank(rows, 2)["gpu_series"]["mapping_verified"] = None
    return by_rank(rows, 2)


def exclusive_shared(rows: Rows, reports: Rows) -> dict[str, Any]:
    by_rank(rows, 3)["gpu_series"]["shared_with_other_processes"] = True
    return by_rank(rows, 3)


def shared_in_cohort(rows: Rows, reports: Rows) -> dict[str, Any]:
    report_of(reports, 6)["peer_comparison"] = roles.ROLE_COHORT
    return by_rank(rows, 6)


def exclusive_alone(rows: Rows, reports: Rows) -> dict[str, Any]:
    report_of(reports, 4)["peer_comparison"] = roles.FEW_PEERS
    return by_rank(rows, 4)


def cores_missing(rows: Rows, reports: Rows) -> dict[str, Any]:
    report_of(reports, 5)["behavior"]["cpu_cores"] = None
    return by_rank(rows, 5)


def cores_drifted(rows: Rows, reports: Rows) -> dict[str, Any]:
    report_of(reports, 0)["behavior"]["cpu_cores"] = 2.5
    return by_rank(rows, 0)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (scope_lost, "GPU utilization is not scoped DEVICE_NOT_PID"),
        (mapping_unverified, "the PID's GPU mapping is not verified"),
        (exclusive_shared, "an exclusive GPU reads shared=True"),
        (shared_in_cohort, "a rank on a shared GPU joined a cohort (ROLE_COHORT)"),
        (
            exclusive_alone,
            "an exclusive declared rank is INSUFFICIENT_OR_NONOVERLAPPING_PEERS, "
            "not ROLE_COHORT",
        ),
        (cores_missing, "the decision has no CPU cores"),
        (
            cores_drifted,
            "the decision's 2.5 cores is not the windows' weighted mean 0.9000",
        ),
    ],
)
def test_cpu_grader_reads_the_gpu_scope_and_the_decisions_cores(
    defect: Callable[[Rows, Rows], dict[str, Any]], message: str
) -> None:
    rows, decls, reports, _groups = graded(plan.CPU)
    sharers = view.children(account_of(plan.CPU), plan.SHARER_MARKER)
    row = defect(rows, reports)
    assert roles.cpu_errors(rows, decls, reports, sharers) == [
        f"{label(row)}: {message}"
    ], "exactly the broken field of the broken rank is named"


def test_cpu_grader_without_a_decision_and_with_an_unshared_sharer() -> None:
    rows, decls, reports, _groups = graded(plan.CPU)
    sharers = view.children(account_of(plan.CPU), plan.SHARER_MARKER)
    errors = roles.cpu_errors(rows, decls, [], sharers)
    assert not any("has no CPU cores" in error for error in errors), (
        "with no report at all nothing claims the decision's cores are missing"
    )
    assert sum("is not scoped DEVICE_NOT_PID" in error for error in errors) == len(
        [row for row in rows if row["rank"] is not None]
    ), errors
    sharer = next(row for row in rows if row["rank"] is None and row["node_id"] == PEER)
    sharer["gpu_series"]["shared_with_other_processes"] = False
    assert roles.cpu_errors(rows, decls, reports, sharers) == [
        f"{PEER} rank None pid {sharer['pid']}: the sharer's GPU does not read shared"
    ], "the sharer process itself must read its GPU as shared"


# --------------------------------------------------------------------------- #
# pyspy_nonzero
# --------------------------------------------------------------------------- #
def holder_row(rows: Rows, node: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["rank"] is None
        and row["node_id"] == node
        and plan.HOLDER_EXE in row["process_name"]
    )


def holder_report(reports: Rows, row: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in reports if item["pid"] == row["pid"])


def unverified_holder(rows: Rows, reports: Rows) -> tuple[dict[str, Any], str]:
    row = holder_row(rows, NODE)
    row["process_identity"]["verified"] = False
    return row, "the holder's identity is not verified"


def short_holder_series(rows: Rows, reports: Rows) -> tuple[dict[str, Any], str]:
    row = holder_row(rows, PEER)
    del row["stack_series"]["samples"][2]
    return row, "2 stack samples of the holder"


def holder_windows_lost(rows: Rows, reports: Rows) -> tuple[dict[str, Any], str]:
    row = holder_row(rows, NODE)
    row["process_series"]["windows"][1]["status"] = "PARTIAL"
    return row, "the holder's /proc windows were not kept"


def holder_windows_dropped(rows: Rows, reports: Rows) -> tuple[dict[str, Any], str]:
    row = holder_row(rows, PEER)
    holder_report(reports, row)["behavior"]["valid_windows"] = 1
    return row, "the decision dropped the holder's process windows"


def holder_unreported(rows: Rows, reports: Rows) -> tuple[dict[str, Any], str]:
    row = holder_row(rows, PEER)
    reports.remove(holder_report(reports, row))
    return row, "the holder has no process report in the decision"


@pytest.mark.parametrize(
    "defect",
    [
        unverified_holder,
        short_holder_series,
        holder_windows_lost,
        holder_windows_dropped,
        holder_unreported,
    ],
)
def test_pyspy_grader_names_each_holder_defect(
    defect: Callable[[Rows, Rows], tuple[dict[str, Any], str]],
) -> None:
    rows, _decls, reports, _groups = graded(plan.PYSPY)
    holders = view.children(account_of(plan.PYSPY), plan.HOLDER_MARKER)
    row, message = defect(rows, reports)
    assert roles.pyspy_errors(rows, reports, holders) == [
        f"{row['node_id']} rank None pid {row['pid']}: {message}"
    ], "exactly the holder's broken field is named"


def test_pyspy_grader_reads_the_python_ranks_thread_json() -> None:
    rows, _decls, reports, _groups = graded(plan.PYSPY)
    holders = view.children(account_of(plan.PYSPY), plan.HOLDER_MARKER)
    row = by_rank(rows, 9)
    thread = row["stack_series"]["samples"][-1]["threads"][0]
    thread["thread_key"] = "not-a-key"
    assert roles.pyspy_errors(rows, reports, holders) == [
        f"{label(row)}: py-spy thread JSON structure {thread}"
    ], "a thread key that is not <thread id>/<native tid> is not py-spy's JSON"


# --------------------------------------------------------------------------- #
# role_mismatch cohorts
# --------------------------------------------------------------------------- #
def test_cohort_grader_names_a_member_of_another_role() -> None:
    rows, decls, reports, groups = graded(plan.ROLES)
    keyless = {"node_id": NODE, "pid": 4242, "rank": None, "process_identity": {}}
    assert roles.cohort_errors([*rows, keyless], decls, reports, groups) == [], (
        "a row without a process identity is no cohort member"
    )
    pp1 = next(
        cohort
        for cohort in groups
        if cohort["definition"]["pipeline_stage"] == "pp1"
        and cohort["processes"][0][0] == NODE
    )
    intruder = by_rank(rows, 0)
    pp1["processes"].append(list(view.process_key(intruder) or ()))
    tag = str(pp1["cohort"])[:12]
    errors = roles.cohort_errors(rows, decls, reports, groups)
    assert errors == [
        f"cohort {tag} (pp1/dense) holds {label(intruder)} declaring pp0/expert",
        f"cohort {tag} mixes roles [('pp0-expert-node0', 'pp0', 'expert'), "
        "('pp1-dense-node0', 'pp1', 'dense')]",
    ], errors


def test_cohort_grader_skips_unreported_ranks_and_expects_few_peers() -> None:
    rows, decls, reports, groups = graded(plan.ROLES)
    report_of(reports, 2)["peer_comparison"] = roles.NO_COHORT
    assert roles.cohort_errors(rows, decls, reports, groups) == [
        f"{label(by_rank(rows, 2))}: peer comparison {roles.NO_COHORT}, "
        f"expected {roles.ROLE_COHORT}"
    ], "a declared rank among four peers of its group is compared"
    unreported = [item for item in reports if item["rank"] != 2]
    assert roles.cohort_errors(rows, decls, unreported, groups) == [], (
        "a rank without a process report has no comparison to judge"
    )
    pair = [copy.deepcopy(by_rank(rows, rank)) for rank in (0, 1)]
    assert roles.cohort_errors(pair, decls, reports, []) == [
        f"{label(row)}: peer comparison {roles.ROLE_COHORT}, expected {roles.FEW_PEERS}"
        for row in pair
    ], "two sampled peers of a group are too few to compare"
    rescored = [
        {**item, "peer_comparison": roles.FEW_PEERS} if item["rank"] in (0, 1) else item
        for item in reports
    ]
    assert roles.cohort_errors(pair, decls, rescored, []) == [], (
        "the product's INSUFFICIENT_OR_NONOVERLAPPING_PEERS verdict is the expected one"
    )


# --------------------------------------------------------------------------- #
# attempt_change
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeNcclJob, "instances", [])
    monkeypatch.setattr(FakeNcclJob, "traffic", Traffic())
    monkeypatch.setattr(FakeNcclJob, "alive_value", True)
    monkeypatch.setattr(FakeNcclJob, "events", None)
    monkeypatch.setattr(roles, "ManagedNcclJob", FakeNcclJob)
    monkeypatch.setattr(roles, "SecondAttemptNcclJob", FakeSecondAttempt)


@pytest.mark.usefixtures("fake_jobs")
def test_attempt_change_is_inconclusive_when_attempt_a_dies_in_its_stall(
    tmp_path: Path,
) -> None:
    FakeNcclJob.alive_value = False
    product = AttemptProduct()
    with pytest.raises(
        VariantInconclusive, match="attempt A did not stay alive through its stall"
    ):
        run_attempt_change(tmp_path, product)
    assert product.calls == [], "a dead attempt A is never compared with the store"
    assert len(FakeNcclJob.instances) == 1 and FakeNcclJob.instances[0].stalled, (
        "attempt B is never submitted after A died in its stall"
    )
