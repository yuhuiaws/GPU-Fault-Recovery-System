"""GF-REGIONAL-DESTR-013: the EKS CloudTrail pages and the run-dir evidence edges.

``test_destr013_audit_integration`` drives ``main`` over a whole fake site and
``test_audit_destr013_replacement_invariant`` the run-dir readers on the shapes
the runners write. These cases pin what is left: the EC2 lookup an
eks-managed registration is audited through, page by page -- a page without
an inventory, a row that is not an object, a continuation token that is
empty, repeated or never ends, an event from another source or without one, an
event outside the target window, a detail the normaliser could not read --
and the evidence directory a run leaves behind when it cannot be read: a
plan that is not JSON, a listing the kernel refuses, a subdirectory os.walk
cannot enter, an evidence file that is a symlink, no ``cases`` directory at
all. ``main`` itself is pinned on a window that misses the run's own evidence
and on an operator interrupt; the script's import bootstrap and its
``__main__`` entry are run the way a shell runs them.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as audit
from scripts.e2e.regional import regional_commands
from tests.regional._destr013_not_run_records import write_not_run_case
from tests.regional.test_destr013_audit_integration import NOW, LocalAudit, write_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/e2e/regional/audit_destr013_replacement_invariant.py"
REGION = "us-west-2"
ACCOUNT = "123456789012"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/regional/executor"
OTHER_ROLE = f"arn:aws:iam::{ACCOUNT}:role/karpenter-controller"
STARTED = datetime(2026, 9, 7, 9, tzinfo=timezone.utc)
ENDED = STARTED + timedelta(hours=3)
REBOOT = audit.EKS_POSITIVE_CONTROL_EVENT
needs_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads a directory whatever its mode"
)


# --------------------------------------------------------------------------- #
# the script as a shell runs it
# --------------------------------------------------------------------------- #
def test_run_as_a_file_the_audit_puts_the_repository_on_the_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "path", [item for item in sys.path if Path(item or ".").resolve() != ROOT]
    )

    namespace = runpy.run_path(str(SCRIPT), run_name="destr013_audit_as_a_file")

    assert sys.path[0] == str(ROOT)
    assert namespace["ROOT"] == ROOT
    assert namespace["CASE_ID"] == audit.CASE_ID


def test_the_script_entry_point_prints_its_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert "provider-replacement invariants" in usage
    assert "--window-start" in usage and "--expect-no-reboots" in usage, usage


# --------------------------------------------------------------------------- #
# the query window
# --------------------------------------------------------------------------- #
def test_a_window_that_ends_before_it_starts_is_refused_alone() -> None:
    errors, provisional = audit.window_errors(
        started_at=ENDED,
        ended_at=STARTED,
        now=ENDED + timedelta(hours=1),
        accept_recent_window=False,
        reboot_events=0,
        expect_no_reboots=False,
    )

    assert (errors, provisional) == (["window end must be after window start"], False)


# --------------------------------------------------------------------------- #
# the EC2 lookup of an eks-managed registration
# --------------------------------------------------------------------------- #
def ec2_event(
    name: str = REBOOT,
    *,
    event_id: str = "event-1",
    instance: str = "i-00000000000000001",
    issuer: str = ROLE,
    at: datetime = STARTED + timedelta(hours=1),
    lookup: dict[str, Any] | None = None,
    **detail: Any,
) -> dict[str, Any]:
    """One ``lookup-events`` row in the live EC2 shape."""

    value: dict[str, Any] = {
        "eventID": event_id,
        "requestID": "request-" + event_id,
        "eventSource": "ec2.amazonaws.com",
        "eventName": name,
        "eventTime": at.isoformat(),
        "awsRegion": REGION,
        "recipientAccountId": ACCOUNT,
        "requestParameters": {"instancesSet": {"items": [{"instanceId": instance}]}},
        "responseElements": {"_return": True},
        "userIdentity": {
            "type": "AssumedRole",
            "arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/executor/session",
            "accountId": ACCOUNT,
            "sessionContext": {
                "sessionIssuer": {"type": "Role", "arn": issuer, "accountId": ACCOUNT}
            },
        },
    }
    for key, item in detail.items():
        if item is None:
            value.pop(key, None)
        else:
            value[key] = item
    row = {
        "EventId": event_id,
        "EventName": name,
        "EventTime": at.isoformat(),
        "CloudTrailEvent": json.dumps(value),
    }
    row.update(lookup or {})
    return row


class Trail:
    """``aws cloudtrail lookup-events`` answered from scripted pages.

    ``pages[(event name, token)]`` is the page served for that lookup; an
    event name with no page of its own reads an empty, final page.
    """

    def __init__(self, pages: dict[tuple[str, str], Any]) -> None:
        self.pages = pages
        self.lookups: list[tuple[str, str]] = []

    def __call__(
        self, command: list[str], **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["cloudtrail", "lookup-events"], (
            f"the EC2 lookup asks CloudTrail only, not {command[1:3]}"
        )
        name = command[command.index("--lookup-attributes") + 1].split(
            "AttributeValue="
        )[1]
        token = (
            command[command.index("--next-token") + 1]
            if "--next-token" in command
            else ""
        )
        self.lookups.append((name, token))
        page = self.pages.get((name, token), {"Events": []})
        if callable(page):
            page = page(len(self.lookups))
        return subprocess.CompletedProcess(command, 0, json.dumps(page), "")


@pytest.fixture
def trail(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    def install(pages: dict[tuple[str, str], Any]) -> Trail:
        fake = Trail(pages)
        monkeypatch.setattr(regional_commands, "run_command", fake)
        return fake

    yield install


def read_trail() -> dict[str, Any]:
    return audit.eks_cloudtrail_events(
        REGION, STARTED, ENDED, account=ACCOUNT, role_arn=ROLE
    )


def test_the_lookup_follows_every_page_and_sorts_each_call(trail: Any) -> None:
    fake = trail(
        {
            (REBOOT, ""): {
                "Events": [
                    ec2_event(event_id="event-1"),
                    ec2_event(event_id="event-2", issuer=OTHER_ROLE),
                    ec2_event(
                        event_id="event-3",
                        errorCode="Client.UnauthorizedOperation",
                        responseElements=None,
                    ),
                    ec2_event(event_id="event-4", eventSource="sts.amazonaws.com"),
                ],
                "NextToken": "page-2",
            },
            (REBOOT, "page-2"): {
                "Events": [
                    ec2_event(event_id="event-5", instance="i-00000000000000002")
                ]
            },
        }
    )

    value = read_trail()

    at = (STARTED + timedelta(hours=1)).isoformat()
    assert value["events"][REBOOT] == [
        {
            "event_time": at,
            "event_name": REBOOT,
            "event_id": "event-1",
            "instance_ids": ["i-00000000000000001"],
        },
        {
            "event_time": at,
            "event_name": REBOOT,
            "event_id": "event-5",
            "instance_ids": ["i-00000000000000002"],
        },
    ]
    assert value["foreign_calls"][REBOOT] == 1
    assert value["refused_calls"][REBOOT] == 1
    assert [item for item in fake.lookups if item[0] == REBOOT] == [
        (REBOOT, ""),
        (REBOOT, "page-2"),
    ], "the continuation token is handed back on the next lookup"
    assert {name for name, _token in fake.lookups} == set(audit.EKS_EVENT_NAMES)


@pytest.mark.parametrize(
    "page, message",
    [
        ({"NextToken": None}, "CloudTrail page has no complete Events inventory"),
        ({"Events": "none"}, "CloudTrail page has no complete Events inventory"),
        ({"Events": ["not-an-event"]}, "CloudTrail event is malformed"),
        ({"Events": [], "NextToken": ""}, "pagination is malformed or repeated"),
        ({"Events": [], "NextToken": 7}, "pagination is malformed or repeated"),
    ],
    ids=["no-events", "events-not-a-list", "row-not-an-object", "empty", "number"],
)
def test_an_incomplete_page_or_token_fails_the_lookup(
    trail: Any, page: dict[str, Any], message: str
) -> None:
    trail({(REBOOT, ""): page})

    with pytest.raises(audit.AuditError, match=message):
        read_trail()


def test_a_repeated_continuation_token_fails_the_lookup(trail: Any) -> None:
    fake = trail(
        {
            (REBOOT, ""): {"Events": [], "NextToken": "page-2"},
            (REBOOT, "page-2"): {"Events": [], "NextToken": "page-2"},
        }
    )

    with pytest.raises(audit.AuditError, match="malformed or repeated"):
        read_trail()
    assert fake.lookups[-2:] == [(REBOOT, ""), (REBOOT, "page-2")]


def test_a_lookup_that_never_ends_is_bounded(trail: Any) -> None:
    fake = trail(
        {
            (name, token): (lambda count: {"Events": [], "NextToken": f"t{count}"})
            for name in (REBOOT,)
            for token in ("", *(f"t{count}" for count in range(1, 1200)))
        }
    )

    with pytest.raises(audit.AuditError, match="pagination exceeds the audit bound"):
        read_trail()
    assert sum(1 for name, _token in fake.lookups if name == REBOOT) == 1000


@pytest.mark.parametrize(
    "row, message",
    [
        (ec2_event(eventSource=None), "CloudTrail event source is missing"),
        (
            ec2_event(awsRegion="eu-west-1"),
            "identity/time differs from the target window",
        ),
        (
            ec2_event(at=ENDED + timedelta(minutes=5)),
            "identity/time differs from the target window",
        ),
        (
            ec2_event(lookup={"EventId": "event-other"}),
            "CloudTrail event detail is malformed",
        ),
    ],
    ids=["no-source", "other-region", "after-the-window", "lookup-detail-drift"],
)
def test_an_event_the_audit_cannot_attribute_fails_the_lookup(
    trail: Any, row: dict[str, Any], message: str
) -> None:
    trail({(REBOOT, ""): {"Events": [row]}})

    with pytest.raises(audit.AuditError, match=message):
        read_trail()


# --------------------------------------------------------------------------- #
# the run directory's evidence
# --------------------------------------------------------------------------- #
def plan_only(case: Path, *, plan: str, preflight: Any) -> Path:
    case.mkdir(parents=True)
    (case / "plan.json").write_text(plan, encoding="utf-8")
    write_json(case / "preflight.json", preflight)
    return case


def test_a_plan_that_is_not_json_is_unreadable_evidence(tmp_path: Path) -> None:
    case = plan_only(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-016",
        plan="{truncated",
        preflight={"errors": []},
    )

    with pytest.raises(
        audit.AuditError,
        match="cannot be read as JSON: GF-REGIONAL-DESTR-016/plan.json",
    ):
        audit.not_applicable_reason(case)


def test_a_preflight_without_an_error_list_is_not_a_provider_refusal(
    tmp_path: Path,
) -> None:
    case = plan_only(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-016",
        plan=json.dumps({"preflight_passed": False, "mutation_performed": False}),
        preflight={"errors": "not applicable to this provider"},
    )

    assert audit.not_applicable_reason(case) is None
    assert audit.not_applicable_cases(tmp_path) == []


def test_a_case_path_that_is_not_a_real_directory_records_no_not_run(
    tmp_path: Path,
) -> None:
    real = write_not_run_case(tmp_path / "elsewhere", "GF-REGIONAL-DESTR-028")
    cases = tmp_path / "run" / "cases"
    cases.mkdir(parents=True)
    linked = cases / "GF-REGIONAL-DESTR-028"
    linked.symlink_to(real, target_is_directory=True)

    assert audit.not_run_record(real) is not None, "the real directory is a record"
    assert audit.not_run_record(linked) is None, "a redirected directory is not"
    assert audit.not_run_record(cases / "GF-REGIONAL-DESTR-029") is None


@needs_permissions
def test_a_not_run_directory_the_kernel_will_not_list_is_unreadable(
    tmp_path: Path,
) -> None:
    case = write_not_run_case(tmp_path, "GF-REGIONAL-DESTR-028")
    case.chmod(0o100)
    try:
        with pytest.raises(
            audit.AuditError, match="destructive evidence directory cannot be read"
        ):
            audit.not_run_record(case)
    finally:
        case.chmod(0o700)
    assert audit.not_run_record(case) is not None, "readable again, it is a record"


def test_a_run_dir_without_cases_has_no_readable_inventory(tmp_path: Path) -> None:
    for reader in (audit.not_applicable_cases, audit.not_run_cases):
        with pytest.raises(
            audit.AuditError,
            match="--run-dir destructive evidence inventory cannot be read",
        ):
            reader(tmp_path)


@needs_permissions
def test_a_subdirectory_os_walk_cannot_enter_fails_the_inventory(
    tmp_path: Path,
) -> None:
    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-002"
    write_json(case / "result.json", {"started_at": STARTED.isoformat()})
    locked = case / "steps"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(
            audit.AuditError, match="destructive evidence directory cannot be read"
        ):
            audit.run_timestamps(tmp_path)
    finally:
        locked.chmod(0o700)
    assert [item["key"] for item in audit.run_timestamps(tmp_path)] == ["started_at"]


def test_an_evidence_file_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    write_json(outside, {"started_at": STARTED.isoformat()})
    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-002"
    case.mkdir(parents=True)
    (case / "result.json").symlink_to(outside)

    with pytest.raises(
        audit.AuditError,
        match="must not be a symlink: cases/GF-REGIONAL-DESTR-002/result.json",
    ):
        audit.run_timestamps(tmp_path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def test_main_refuses_a_window_that_misses_the_runs_own_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    h.argv[h.argv.index("--window-start") + 1] = (NOW - timedelta(hours=1)).isoformat()

    code, report = h.run()

    assert (code, report["verdict"], report["status"]) == (1, "FAIL", "FAILED")
    assert report["error"].startswith(
        "the window does not cover the run's evidence (1 timestamp(s) outside)"
    ), report["error"]
    assert h.calls == [], "a window that misses the evidence reads nothing remote"


def test_main_reports_an_operator_interrupt_as_exit_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)

    def interrupt(_command: list[str]) -> None:
        raise KeyboardInterrupt

    h.before_command = interrupt

    code, report = h.run()

    assert code == 130
    assert (report["verdict"], report["status"]) == ("FAIL", "FAILED")
    assert report["error"].startswith("KeyboardInterrupt"), report["error"]
    assert report["formal_sequence_satisfied"] is False
    assert len(h.calls) == 1, "nothing is read after the interrupt"
