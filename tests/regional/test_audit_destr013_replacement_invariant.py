"""DESTR-013's final invariant: what it asserts beyond "no replace in the window".

The 2026-09-07 review found the audit collecting evidence it never judged
(``DeleteCluster``/``UpdateCluster``), a focused-test result that was a
constant, no check that the synthetic replacement route DESTR-003/008 open had
been closed again, and a CloudTrail window taken at face value -- an empty
query in the wrong region reads exactly like a clean one.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as destr013

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def _replica(pod: str, **overrides: str | None) -> dict[str, str | None]:
    value: dict[str, str | None] = {
        "pod": pod,
        "allow_replace": "false",
        "allow_reboot": "true",
        "allow_automatic": None,
        "legacy_mutation": None,
    }
    value.update(overrides)
    return value


def test_every_executor_replica_must_independently_enable_reboot() -> None:
    # DESTR-011's unique check, absorbed here now that it is superseded: a
    # fleet that disabled every HyperPod verb satisfies the replace invariant
    # while making the reboot path DESTR-002 depends on impossible.
    assert destr013.environment_errors([_replica("a"), _replica("b")]) == []
    errors = destr013.environment_errors(
        [_replica("a"), _replica("b", allow_reboot=None)]
    )
    assert errors == ["an executor replica does not independently enable reboot"]
    errors = destr013.environment_errors([_replica("a", allow_replace="true")])
    assert "an executor replica violates the mutation invariants" in errors
    assert destr013.environment_errors([]) == [
        "no running executor replicas were inspected"
    ]


def test_the_synthetic_replacement_route_must_be_closed_on_every_api_replica() -> None:
    closed = [{"pod": "api-a", "synthetic_route": None}]
    assert destr013.synthetic_route_errors(closed) == []
    left_open = [
        {"pod": "api-a", "synthetic_route": None},
        {"pod": "api-b", "synthetic_route": "true"},
        {"pod": "api-c", "synthetic_route": "false"},
    ]
    errors = destr013.synthetic_route_errors(left_open)
    # Even a literal "false" is a variable the shipped manifest does not have;
    # the window helper restores absence, so presence means an unclosed window.
    assert errors == [
        f"{destr013.SYNTHETIC_ROUTE_ENV} is still set on API replicas: api-b, api-c"
    ]
    assert destr013.synthetic_route_errors([]) == [
        "no running API replicas were inspected for the synthetic route"
    ]


def test_delete_and_update_cluster_are_judged_not_just_collected() -> None:
    clean: dict[str, list[dict[str, str]]] = {name: [] for name in destr013.EVENT_NAMES}
    assert destr013.cloudtrail_invariant_errors(clean) == []
    flipped = {**clean, "UpdateCluster": [{"event_name": "UpdateCluster"}]}
    assert destr013.cloudtrail_invariant_errors(flipped) == [
        "CloudTrail contains UpdateCluster"
    ]
    deleted = {**clean, "DeleteCluster": [{"event_name": "DeleteCluster"}]}
    assert destr013.cloudtrail_invariant_errors(deleted) == [
        "CloudTrail contains DeleteCluster"
    ]
    replaced = {**clean, "ReplaceClusterNodes": [{"event_name": "ReplaceClusterNodes"}]}
    assert destr013.cloudtrail_invariant_errors(replaced) == [
        "CloudTrail contains provider replacement"
    ]
    # The positive control is not a violation.
    rebooted = {**clean, "BatchRebootClusterNodes": [{"event_name": "x"}]}
    assert destr013.cloudtrail_invariant_errors(rebooted) == []


def test_a_window_ending_inside_cloudtrail_lag_is_refused_unless_accepted() -> None:
    recent_end = NOW - timedelta(minutes=5)
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=recent_end,
        now=NOW,
        accept_recent_window=False,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert len(errors) == 1 and "--accept-recent-window" in errors[0], errors
    assert provisional is False

    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=recent_end,
        now=NOW,
        accept_recent_window=True,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert errors == []
    assert provisional is True

    settled_end = NOW - timedelta(minutes=20)
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=settled_end,
        now=NOW,
        accept_recent_window=False,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert (errors, provisional) == ([], False)


def test_an_empty_window_needs_a_positive_control_or_an_explicit_waiver() -> None:
    common: dict[str, Any] = {
        "started_at": NOW - timedelta(hours=3),
        "ended_at": NOW - timedelta(minutes=20),
        "now": NOW,
        "accept_recent_window": False,
    }
    errors, _ = destr013.window_errors(
        **common, reboot_events=0, expect_no_reboots=False
    )
    assert len(errors) == 1 and "wrong region or hours" in errors[0], errors
    errors, _ = destr013.window_errors(
        **common, reboot_events=0, expect_no_reboots=True
    )
    assert errors == []
    # The waiver is itself checked: reboots in a window the operator swore was
    # reboot-free mean the operator described a different run.
    errors, _ = destr013.window_errors(
        **common, reboot_events=1, expect_no_reboots=True
    )
    assert len(errors) == 1 and "--expect-no-reboots was given" in errors[0], errors


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_timestamps_read_top_level_keys_and_timeline_entries_only(
    tmp_path: Path,
) -> None:
    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-003"
    _write(
        case / "GF-REGIONAL-DESTR-003.json",
        {
            "case_id": "GF-REGIONAL-DESTR-003",
            "started_at": "2026-09-07T09:00:00+00:00",
            "maintenance_window_end": "2026-09-08T00:00:00+00:00",
            "store": {"agents": [{"created_at": "2026-01-01T00:00:00+00:00"}]},
        },
    )
    _write(
        case / "timeline.json", {"entries": [{"observed_at": "2026-09-07T09:05:00Z"}]}
    )
    # Other case families and non-object documents are not the run's window.
    _write(
        tmp_path / "cases" / "GF-REGIONAL-HA-004" / "x.json",
        {"started_at": "2020-01-01T00:00:00Z"},
    )
    _write(case / "list.json", ["2026-09-07T09:00:00+00:00"])

    found = destr013.run_timestamps(tmp_path)

    assert sorted(item["at"] for item in found) == [
        "2026-09-07T09:00:00+00:00",
        "2026-09-07T09:05:00+00:00",
    ], found
    assert {item["key"] for item in found} == {"started_at", "entries[].observed_at"}


def test_the_window_must_cover_the_runs_recorded_timestamps() -> None:
    timestamps = [
        {"path": "a", "key": "started_at", "at": "2026-09-07T09:00:00+00:00"},
        {"path": "b", "key": "observed_at", "at": "2026-09-07T10:30:00+00:00"},
    ]
    start = datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc)
    assert (
        destr013.coverage_errors(
            started_at=start,
            ended_at=datetime(2026, 9, 7, 11, 0, tzinfo=timezone.utc),
            timestamps=timestamps,
        )
        == []
    )
    errors = destr013.coverage_errors(
        started_at=start,
        ended_at=datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
        timestamps=timestamps,
    )
    assert len(errors) == 1 and "1 timestamp(s) outside" in errors[0], errors
    assert destr013.coverage_errors(
        started_at=start, ended_at=start, timestamps=[]
    ) == ["--run-dir carries no destructive-case timestamps to check against"]


def test_focused_tests_report_a_failing_run_instead_of_a_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="1 failed\n", stderr="")

    monkeypatch.setattr(destr013, "run", fake_run)

    result = destr013.focused_tests(tmp_path)

    assert result["passed"] is False, "a failed subprocess must fail the audit"
    assert result["returncode"] == 1, "preserve the focused subprocess exit code"
    log = (tmp_path / "focused-tests.log").read_text(encoding="utf-8")
    assert log == "returncode=1\n<sensitive output redacted>\n", (
        "subprocess logs must not expose raw environment or traceback data"
    )


@pytest.mark.parametrize("value", [None, "", "not-a-time", 123, "2026-09-07T09:00:00"])
@pytest.mark.parametrize("shape", ["top-level", "entries", "transitions"])
def test_invalid_timestamp_cannot_disappear_beside_valid_evidence(
    tmp_path: Path, value: Any, shape: str
) -> None:
    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-002"
    _write(case / "valid.json", {"started_at": "2026-09-07T09:00:00Z"})
    document = (
        {"observed_at": value}
        if shape == "top-level"
        else {shape: [{"observed_at": value}]}
    )
    _write(case / "invalid.json", document)
    with pytest.raises(destr013.AuditError, match="invalid.json.*(ISO-8601|timezone)"):
        destr013.run_timestamps(tmp_path)


OBSERVED = {"observed_at": "2026-09-07T09:05:00Z"}


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param(None, id="null"),
        pytest.param({}, id="object"),
        pytest.param("invalid", id="string"),
        pytest.param([OBSERVED, {"pod": "gpu-fault-api-ha-0"}], id="mixed-dict"),
        pytest.param([OBSERVED, None], id="mixed-null"),
        pytest.param([OBSERVED, "2026-09-07T09:06:00Z"], id="mixed-string"),
    ],
)
def test_malformed_timeline_cannot_disappear(tmp_path: Path, entries: Any) -> None:
    """A non-list, or a timeline with holes, is an integrity failure.

    Once any element carries ``observed_at`` the list is a timeline, and every
    other element must carry one too: an entry that lost its timestamp beside
    kept ones is how a window's evidence would silently shrink.
    """

    _write(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-002" / "timeline.json",
        {"started_at": "2026-09-07T09:00:00Z", "entries": entries},
    )
    with pytest.raises(destr013.AuditError, match="timeline is malformed"):
        destr013.run_timestamps(tmp_path)


def test_a_collectors_entries_list_is_not_a_timeline(tmp_path: Path) -> None:
    """The control-plane log collector reuses ``entries`` for per-Pod verdicts.

    Its document (absent/entries/inconclusive/silent/suspicious/verdict) lists
    log classifications, none carrying ``observed_at``; the audit must read the
    case's run timestamps from its other files and neither fail on this list
    nor descend into the nested container timestamps.
    """

    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-009"
    _write(
        case / "GF-REGIONAL-DESTR-009.json",
        {"case_id": "GF-REGIONAL-DESTR-009", "started_at": "2026-09-07T09:00:00Z"},
    )
    _write(
        case / "control-plane-logs.json",
        {
            "absent": ["gpu-fault-control-worker-0"],
            "entries": [
                {
                    "classification": "silent",
                    "line_count": 0,
                    "pod": "gpu-fault-api-ha-0",
                    "sha256": "0" * 64,
                    "silence_evidence": {
                        "container_name": "api",
                        "container_started_at": "2020-01-01T00:00:00+00:00",
                        "restart_count": 0,
                    },
                },
                {
                    "classification": "absent",
                    "line_count": 0,
                    "pod": "gpu-fault-control-worker-0",
                    "sha256": "0" * 64,
                },
            ],
            "inconclusive": [],
            "silent": ["gpu-fault-api-ha-0"],
            "suspicious": [],
            "verdict": "CLEAN",
        },
    )

    found = destr013.run_timestamps(tmp_path)

    assert found == [
        {
            "path": "cases/GF-REGIONAL-DESTR-009/GF-REGIONAL-DESTR-009.json",
            "key": "started_at",
            "at": "2026-09-07T09:00:00+00:00",
        }
    ], found


@pytest.mark.parametrize("entries_key", ["entries", "transitions"])
@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([], id="empty"),
        pytest.param([{}], id="bare-object"),
        pytest.param([{"probed_at": "2026-09-07T09:05:00Z"}], id="coverage-probe"),
        pytest.param(["a log line"], id="strings"),
    ],
)
def test_a_list_without_observed_at_is_not_a_timeline(
    tmp_path: Path, entries_key: str, entries: Any
) -> None:
    """A list in which no element carries ``observed_at`` is skipped, not judged."""

    _write(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-024" / "coverage-idle-timeline.json",
        {"started_at": "2026-09-07T09:00:00Z", entries_key: entries},
    )

    found = destr013.run_timestamps(tmp_path)

    assert [(item["key"], item["at"]) for item in found] == [
        ("started_at", "2026-09-07T09:00:00+00:00")
    ], found


@pytest.mark.parametrize("defect", ["unreadable", "invalid-json", "missing-case-time"])
def test_one_valid_case_does_not_cover_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    _write(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-002" / "valid.json",
        {"started_at": "2026-09-07T09:00:00Z"},
    )
    other = tmp_path / "cases" / "GF-REGIONAL-DESTR-003" / "data.json"
    _write(other, {"nodes": []})
    if defect == "invalid-json":
        other.write_text("{", encoding="utf-8")
    elif defect == "unreadable":
        original = Path.read_text

        def read(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == other:
                raise OSError("unreadable")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(destr013.AuditError, match="cannot be read|no run timestamp"):
        destr013.run_timestamps(tmp_path)


NOT_APPLICABLE_REASON = (
    "executor has no HyperPod reboot adapter (owners: gpu-fault-eks-adapter, "
    "gpu-fault-kubernetes-adapter, gpu-fault-node-agent); DESTR-002 audits the "
    "HyperPod reboot path and does not apply to an eks-managed cluster"
)


def _plan_only_case(
    root: Path, case_id: str, errors: list[str], *, mutated: bool = False
) -> Path:
    """What a destructive case's directory holds after its preflight refused:
    the plan and the preflight, no execute evidence, no run timestamp."""

    case = root / "cases" / case_id
    _write(
        case / "plan.json",
        {
            "case_id": case_id,
            "attempt": 5,
            "preflight_passed": False,
            "mutation_performed": mutated,
            "execution_scope": "selective",
        },
    )
    _write(case / "preflight.json", {"errors": errors, "node": {}})
    return case


def test_a_case_its_preflight_refused_as_not_applicable_is_not_missing_evidence(
    tmp_path: Path,
) -> None:
    """Live (plain EKS run dir): the HyperPod-only DESTR-002/016/025 were
    planned, refused by their own preflight as not applicable to an eks-managed
    cluster and recorded SKIPPED; the audit read their plan-only directories as
    "no run timestamp evidence" and failed."""

    _write(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-003" / "GF-REGIONAL-DESTR-003.json",
        {"started_at": "2026-09-07T09:00:00Z"},
    )
    _plan_only_case(
        tmp_path,
        "GF-REGIONAL-DESTR-002",
        [NOT_APPLICABLE_REASON, "target node is already unschedulable"],
    )
    _plan_only_case(
        tmp_path,
        "GF-REGIONAL-DESTR-016",
        [
            "nodeReboot is OWN by gpu-fault-eks-adapter (EC2 RebootInstances), not "
            "the HyperPod adapter; GF-REGIONAL-DESTR-016 audits the HyperPod reboot "
            "path (BatchRebootClusterNodes) and does not apply to an eks-managed "
            "cluster"
        ],
    )

    found = destr013.run_timestamps(tmp_path)

    assert [item["path"] for item in found] == [
        "cases/GF-REGIONAL-DESTR-003/GF-REGIONAL-DESTR-003.json"
    ], "the executed case is the window's evidence; the refused ones add none"
    assert destr013.not_applicable_cases(tmp_path) == [
        {"case_id": "GF-REGIONAL-DESTR-002", "reason": NOT_APPLICABLE_REASON},
        {
            "case_id": "GF-REGIONAL-DESTR-016",
            "reason": (
                "nodeReboot is OWN by gpu-fault-eks-adapter (EC2 RebootInstances), "
                "not the HyperPod adapter; GF-REGIONAL-DESTR-016 audits the HyperPod "
                "reboot path (BatchRebootClusterNodes) and does not apply to an "
                "eks-managed cluster"
            ),
        },
    ], "each refusal is recorded with the preflight's own words"


@pytest.mark.parametrize(
    ("errors", "mutated"),
    [
        (["target node is already unschedulable"], False),
        (["nodeReboot is not OWN by the HyperPod adapter"], False),
        ([NOT_APPLICABLE_REASON], True),
        ([], False),
    ],
    ids=["other-refusal", "generic-owner-refusal", "mutation-performed", "no-errors"],
)
def test_a_plan_only_case_refused_for_any_other_reason_is_still_missing_evidence(
    tmp_path: Path, errors: list[str], mutated: bool
) -> None:
    """Only the explicit not-applicable words settle a plan-only directory; a
    preflight refused for the node's state, the older generic owner wording,
    or a plan that says a mutation happened keeps the fail-closed rule."""

    _write(
        tmp_path / "cases" / "GF-REGIONAL-DESTR-003" / "GF-REGIONAL-DESTR-003.json",
        {"started_at": "2026-09-07T09:00:00Z"},
    )
    case = _plan_only_case(tmp_path, "GF-REGIONAL-DESTR-002", errors, mutated=mutated)

    assert destr013.not_applicable_reason(case) is None
    with pytest.raises(destr013.AuditError, match="no run timestamp evidence"):
        destr013.run_timestamps(tmp_path)
    assert destr013.not_applicable_cases(tmp_path) == []


def test_the_audit_cannot_supply_its_own_window_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path / "cases" / destr013.CASE_ID / f"{destr013.CASE_ID}.json",
        {"window_start": "2026-09-07T09:00:00Z"},
    )
    assert destr013.run_timestamps(tmp_path) == [], (
        "initial and previous DESTR013 output cannot prove earlier destructive coverage"
    )


def test_future_window_is_not_accepted_as_provisional() -> None:
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=1),
        ended_at=NOW + timedelta(minutes=5),
        now=NOW,
        accept_recent_window=True,
        reboot_events=1,
        expect_no_reboots=False,
    )
    assert errors == ["window must not extend into the future"], (
        "an explicitly accepted recent window must still end in the past"
    )
    assert provisional is False, "future windows are invalid, not provisional"


def test_parser_accepts_the_window_edge_flags_and_the_control_plane(
    tmp_path: Path,
) -> None:
    arguments = destr013.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--window-start",
            "2026-08-30T00:00:00Z",
            "--window-end",
            "2026-08-31T00:00:00Z",
            "--accept-recent-window",
            "--expect-no-reboots",
            "--cpu-kubeconfig",
            "/tmp/cpu",
        ]
    )
    assert arguments.accept_recent_window is True
    assert arguments.expect_no_reboots is True
    assert arguments.cpu_kubeconfig == "/tmp/cpu"
    defaults = destr013.parser().parse_args(
        ["--run-dir", str(tmp_path), "--window-start", "x", "--window-end", "y"]
    )
    assert defaults.accept_recent_window is False
    assert defaults.expect_no_reboots is False


def test_destr011_is_recorded_as_superseded() -> None:
    assert destr013.SUPERSEDED_CASE_IDS == ("GF-REGIONAL-DESTR-011",)
    assert destr013.POSITIVE_CONTROL_EVENT in destr013.EVENT_NAMES
    assert "UpdateCluster" in destr013.FORBIDDEN_EVENT_NAMES


# --------------------------------------------------------------------------- #
# eks-managed branch (live mixed site: the registry's second cluster)
# --------------------------------------------------------------------------- #
EKS_NAME = "gpu-eks"
INSTANCE_A = "i-00000000000000001"
INSTANCE_B = "i-00000000000000002"


def _trail(**events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {name: [] for name in destr013.EKS_EVENT_NAMES}
    rows.update(events)
    return {"events": rows, "refused_calls": {}, "foreign_calls": {}}


def _reboot(*instance_ids: str) -> dict[str, Any]:
    return {
        "event_time": NOW.isoformat(),
        "event_name": "RebootInstances",
        "event_id": "event-" + "-".join(instance_ids),
        "instance_ids": list(instance_ids),
    }


def test_window_errors_name_the_providers_positive_control_event() -> None:
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=NOW - timedelta(minutes=20),
        now=NOW,
        accept_recent_window=False,
        reboot_events=0,
        expect_no_reboots=False,
        control_event=destr013.EKS_POSITIVE_CONTROL_EVENT,
    )
    assert provisional is False
    assert errors and errors[0].startswith("no RebootInstances in the window"), errors
    errors, _provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=NOW - timedelta(minutes=20),
        now=NOW,
        accept_recent_window=False,
        reboot_events=2,
        expect_no_reboots=True,
        control_event=destr013.EKS_POSITIVE_CONTROL_EVENT,
    )
    assert errors == [
        "--expect-no-reboots was given but 2 RebootInstances event(s) are in the window"
    ]


def test_eks_cloudtrail_invariant_judges_the_executor_roles_ec2_calls_only() -> None:
    tags = {INSTANCE_A: EKS_NAME, INSTANCE_B: EKS_NAME}
    clean = _trail(RebootInstances=[_reboot(INSTANCE_A), _reboot(INSTANCE_B)])
    assert destr013.eks_cloudtrail_invariant_errors(
        clean, instance_cluster_tags=tags, eks_name=EKS_NAME
    ) == ([], {"bound": [INSTANCE_A, INSTANCE_B], "unbound": [], "outside_cluster": []})

    replaced = _trail(
        RebootInstances=[_reboot(INSTANCE_A)],
        TerminateInstances=[
            {
                "event_time": "t",
                "event_name": "TerminateInstances",
                "event_id": "x",
                "instance_ids": [INSTANCE_B],
            }
        ],
        RunInstances=[
            {
                "event_time": "t",
                "event_name": "RunInstances",
                "event_id": "y",
                "instance_ids": [],
            }
        ],
    )
    errors, _bindings = destr013.eks_cloudtrail_invariant_errors(
        replaced, instance_cluster_tags=tags, eks_name=EKS_NAME
    )
    assert errors == [
        "CloudTrail contains an accepted EC2 RunInstances by the executor role",
        "CloudTrail contains an accepted EC2 TerminateInstances by the executor role",
    ]

    foreign = _trail(
        RebootInstances=[_reboot(INSTANCE_A), _reboot("i-0000000000000000")]
    )
    errors, bindings = destr013.eks_cloudtrail_invariant_errors(
        foreign,
        instance_cluster_tags={**tags, "i-0000000000000000": "other-cluster"},
        eks_name=EKS_NAME,
    )
    assert errors == [
        "CloudTrail contains 1 RebootInstances target(s) by the executor role "
        "outside the registered EKS cluster"
    ]
    assert bindings["outside_cluster"] == ["i-0000000000000000"]
    errors, bindings = destr013.eks_cloudtrail_invariant_errors(
        foreign,
        instance_cluster_tags={**tags, "i-0000000000000000": ""},
        eks_name=EKS_NAME,
    )
    assert errors, "an instance EC2 knows but does not tag to the cluster is outside it"

    gone = _trail(RebootInstances=[_reboot(INSTANCE_A), _reboot("i-0000000000000000")])
    errors, bindings = destr013.eks_cloudtrail_invariant_errors(
        gone, instance_cluster_tags=tags, eks_name=EKS_NAME
    )
    assert errors == [], "an instance EC2 no longer describes is recorded, not judged"
    assert bindings == {
        "bound": [INSTANCE_A],
        "unbound": ["i-0000000000000000"],
        "outside_cluster": [],
    }

    nameless = _trail(RebootInstances=[_reboot()])
    errors, _bindings = destr013.eks_cloudtrail_invariant_errors(
        nameless, instance_cluster_tags=tags, eks_name=EKS_NAME
    )
    assert errors == [
        "CloudTrail RebootInstances by the executor role names no instance"
    ]


def test_eks_iam_errors_allow_reboot_only() -> None:
    clean = {
        "ec2:RebootInstances": "allowed",
        "ec2:RunInstances": "implicitDeny",
        "ec2:StartInstances": "implicitDeny",
        "ec2:StopInstances": "explicitDeny",
        "ec2:TerminateInstances": "implicitDeny",
    }
    assert destr013.eks_iam_errors(clean) == []
    assert destr013.eks_iam_errors({**clean, "ec2:TerminateInstances": "allowed"}) == [
        "executor IAM permits ec2:TerminateInstances"
    ]
    assert destr013.eks_iam_errors({**clean, "ec2:RebootInstances": "mixed"}) == [
        "executor IAM does not allow ec2:RebootInstances"
    ]
    assert destr013.eks_iam_errors({}) == [
        f"executor IAM permits {action}" for action in destr013.EKS_IAM_ACTIONS[1:]
    ] + ["executor IAM does not allow ec2:RebootInstances"]


def _eks_inventory(**overrides: Any) -> dict[str, Any]:
    value = {
        "count": 2,
        "sha256": "f" * 64,
        "cluster_name": EKS_NAME,
        "nodes": [
            {
                "node_logical_id": "node-a",
                "instance_id": INSTANCE_A,
                "instance_group": "gpu",
            },
            {
                "node_logical_id": "node-b",
                "instance_id": INSTANCE_B,
                "instance_group": "gpu",
            },
        ],
        "nodegroups": {
            "gpu": {
                "cluster_name": EKS_NAME,
                "status": "ACTIVE",
                "health_issue_codes": [],
            }
        },
        "instance_cluster_tags": {INSTANCE_A: EKS_NAME, INSTANCE_B: EKS_NAME},
    }
    value.update(overrides)
    return value


def test_eks_inventory_errors_bind_nodes_to_the_registered_cluster() -> None:
    assert destr013.eks_inventory_errors(_eks_inventory()) == []
    assert destr013.eks_inventory_errors(
        _eks_inventory(nodes=[], instance_cluster_tags={})
    ) == ["EKS GPU node inventory is empty"]
    assert destr013.eks_inventory_errors(
        _eks_inventory(instance_cluster_tags={INSTANCE_A: EKS_NAME, INSTANCE_B: ""})
    ) == [f"1 GPU instance(s) are not tagged eks:cluster-name={EKS_NAME}"]
    degraded = _eks_inventory(
        nodegroups={
            "gpu": {
                "cluster_name": "other",
                "status": "DEGRADED",
                "health_issue_codes": ["AsgInstanceLaunchFailures"],
            }
        }
    )
    assert destr013.eks_inventory_errors(degraded) == [
        "managed node group gpu is not ACTIVE on the registered EKS cluster",
        "managed node group gpu reports health issues: ['AsgInstanceLaunchFailures']",
    ]
    assert destr013.provider_inventory_errors(
        "eks-managed", _eks_inventory(), _eks_inventory(count=1)
    ) == ["EKS GPU node inventory changed during the read-only audit"]


def test_provider_dispatch_keeps_hyperpod_messages_and_vocabulary() -> None:
    assert destr013.provider_control_event("hyperpod") == "BatchRebootClusterNodes"
    assert destr013.provider_control_event("eks-managed") == "RebootInstances"
    assert destr013.provider_cluster_errors(
        "hyperpod", {"cluster_status": "Updating", "node_recovery": "Automatic"}
    ) == ["HyperPod cluster is not InService", "HyperPod NodeRecovery is not None"]
    assert destr013.provider_cluster_errors(
        "eks-managed", {"cluster_status": "UPDATING"}
    ) == ["EKS cluster is not ACTIVE"]
    assert destr013.provider_iam_errors(
        "hyperpod", {"sagemaker:BatchReplaceClusterNodes": "allowed"}
    ) == [
        "executor IAM permits BatchReplaceClusterNodes",
        "executor IAM does not allow BatchRebootClusterNodes",
    ]
    assert set(destr013.EKS_EVENT_NAMES) == {
        "RebootInstances",
        "RunInstances",
        "StartInstances",
        "StopInstances",
        "TerminateInstances",
    }
