"""The harnesses' read of the deletion verbs' registry sync gate.

``uninstall`` and ``remove-cluster`` refuse a new attempt while the site's last
installation registry sync is failed, missing or unreadable
(``resource_registry_outcome.deletion_gate``) and resume a started attempt on
the verdict it started with. ``registry_sync_preflight`` judges the same record
read-only for BOOT-032, the BOOT isolated-site cleanup and the BOOT-029 driver,
names the product's remedy and never passes the override flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin.resource_registry_outcome import ACCEPT_FLAG, record_path
from scripts.e2e.regional import registry_sync_preflight as preflight
from tests.admin._registry_sync_support import record_registry_sync

EKS_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
OTHER_EKS_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-b"
HYPERPOD_ARN = "arn:aws:sagemaker:us-west-2:123456789012:cluster/abc123def456"
FAILURE = "fixture registry sync failed"


def unsynchronize(state: Path, record: str) -> None:
    if record == "failed":
        record_registry_sync(state, error=FAILURE)
    elif record == "unreadable":
        record_path(state).parent.mkdir(parents=True, exist_ok=True)
        record_path(state).write_text("{", encoding="utf-8")


def removal_journal(
    state: Path,
    name: str,
    *,
    phase: str = "COMPLETED",
    eks_arn: str = EKS_ARN,
    discovered: dict[str, str] | None = None,
) -> Path:
    path = state / "remove-cluster" / name / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = (
        {} if discovered is None else {"DISCOVERED": {"provider_identity": discovered}}
    )
    path.write_text(
        json.dumps(
            {
                "phase": phase,
                "target": {"eks_cluster_arn": eks_arn},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("record", "status", "reason"),
    [
        ("missing", "MISSING", "no installation registry sync outcome is recorded"),
        ("failed", "UNAVAILABLE", FAILURE),
        ("unreadable", "UNREADABLE", "is not valid JSON"),
    ],
    ids=["missing", "failed", "unreadable"],
)
def test_a_new_attempt_is_refused_with_the_recorded_outcome_and_the_remedy(
    tmp_path: Path, record: str, status: str, reason: str
) -> None:
    unsynchronize(tmp_path, record)

    refusal = preflight.deletion_refusal(
        tmp_path, verb="uninstall", harness="CASE-X", then="fix it, then rerun"
    )

    assert refusal is not None, "an unsynchronized registry refuses a new attempt"
    assert refusal.startswith(
        f"uninstall would refuse to start (installation registry sync {status}): "
    ), refusal
    assert reason in refusal, f"the refusal names the recorded outcome: {refusal}"
    assert f"gpu-fault-admin deploy --state-dir {tmp_path}" in refusal, (
        f"the refusal names the product's remedy: {refusal}"
    )
    assert refusal.endswith(f"CASE-X never passes {ACCEPT_FLAG}: fix it, then rerun"), (
        f"the harness says it never overrides the gate: {refusal}"
    )


def test_a_recorded_success_refuses_nothing(tmp_path: Path) -> None:
    record_registry_sync(tmp_path)

    assert (
        preflight.deletion_refusal(tmp_path, verb="uninstall", harness="h", then="t")
        is None
    ), "a recorded success lets a new attempt start"
    assert (
        preflight.removal_refusal(tmp_path, EKS_ARN, harness="h", then="t") is None
    ), "a recorded success lets a new removal start"


def test_a_started_uninstall_is_left_to_the_verdict_it_started_with(
    tmp_path: Path,
) -> None:
    assert preflight.uninstall_refusal(tmp_path, harness="h", then="t") is not None, (
        "without a journal the missing record refuses"
    )
    journal = tmp_path / "uninstall" / "state.json"
    journal.parent.mkdir()
    journal.write_text("{}", encoding="utf-8")

    assert preflight.uninstall_refusal(tmp_path, harness="h", then="t") is None, (
        "a started uninstall resumes on its own verdict"
    )


def test_a_removal_journal_of_this_cluster_is_left_to_remove_cluster(
    tmp_path: Path,
) -> None:
    removal_journal(tmp_path, "cluster-a", phase="STARTED")
    assert preflight.removal_started(tmp_path, EKS_ARN), (
        "an unfinished removal of the cluster resumes"
    )
    assert preflight.removal_refusal(tmp_path, EKS_ARN, harness="h", then="t") is None
    removal_journal(tmp_path, "cluster-a")
    assert preflight.removal_started(tmp_path, EKS_ARN), (
        "a completed removal of the cluster is only reported again"
    )
    removal_journal(
        tmp_path,
        "cluster-a",
        discovered={"eks_arn": EKS_ARN, "hyperpod_arn": HYPERPOD_ARN},
    )
    assert preflight.removal_started(tmp_path, HYPERPOD_ARN), (
        "a HyperPod cluster named by its SageMaker ARN matches its discovered identity"
    )


def test_journals_of_other_clusters_do_not_spare_a_new_removal(tmp_path: Path) -> None:
    removal_journal(tmp_path, "cluster-b", eks_arn=OTHER_EKS_ARN)
    removal_journal(
        tmp_path,
        "cluster-c",
        phase="STARTED",
        eks_arn=OTHER_EKS_ARN,
        discovered={"eks_arn": OTHER_EKS_ARN, "hyperpod_arn": ""},
    )

    refusal = preflight.removal_refusal(tmp_path, EKS_ARN, harness="h", then="t")

    assert refusal is not None and refusal.startswith(
        "remove-cluster would refuse to start (installation registry sync MISSING)"
    ), f"another cluster's journals leave this removal a new attempt: {refusal}"


@pytest.mark.parametrize(
    "journal", ["undiscovered-hyperpod", "unreadable", "invalid-arn"]
)
def test_a_journal_that_cannot_be_attributed_leaves_the_verdict_to_the_product(
    tmp_path: Path, journal: str
) -> None:
    arn = HYPERPOD_ARN
    if journal == "undiscovered-hyperpod":
        # A SageMaker ARN maps to its EKS ARN only through discovery, which
        # this attempt never recorded: it may be this cluster's.
        removal_journal(tmp_path, "cluster-a", phase="STARTED")
    elif journal == "unreadable":
        path = removal_journal(tmp_path, "cluster-a")
        path.write_text("[", encoding="utf-8")
    else:
        removal_journal(tmp_path, "cluster-a")
        arn = "not-an-arn"

    assert preflight.removal_started(tmp_path, arn), (
        "a journal that may be this cluster's is never refused by the harness"
    )


@pytest.mark.parametrize("verb", ["uninstall", "remove-cluster"])
def test_the_command_line_prints_one_verdict_and_exits_by_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    arguments = [verb, "--state-dir", str(tmp_path)]
    if verb == "remove-cluster":
        arguments += ["--gpu-arn", EKS_ARN]

    assert preflight.main(arguments) == 1, "a missing record refuses"
    refused = capsys.readouterr().out.splitlines()
    record_registry_sync(tmp_path)
    assert preflight.main(arguments) == 0, "a recorded success passes"
    passed = capsys.readouterr().out.splitlines()

    assert len(refused) == 1 and refused[0].startswith(
        f"{verb} would refuse to start (installation registry sync MISSING)"
    ), refused
    assert refused[0].endswith(
        f"BOOT-029 never passes {ACCEPT_FLAG}: {preflight.DRIVER_REMEDY}"
    ), refused
    assert passed == [
        "installation registry sync SYNCED (release release-a, synced at "
        f"2026-01-01T00:00:00Z): {verb} may start a new attempt"
    ], passed


def test_the_command_line_names_a_resume_without_judging_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    removal_journal(tmp_path, "cluster-a", phase="STARTED")

    code = preflight.main(
        ["remove-cluster", "--state-dir", str(tmp_path), "--gpu-arn", EKS_ARN]
    )

    (line,) = capsys.readouterr().out.splitlines()
    assert code == 0, "a started removal is left to remove-cluster"
    assert line.startswith(
        "remove-cluster may continue an attempt it already started"
    ), line


@pytest.mark.parametrize(
    "arguments",
    [
        ["remove-cluster", "--state-dir", "s"],
        ["uninstall", "--state-dir", "s", "--gpu-arn", EKS_ARN],
    ],
    ids=["removal-without-arn", "uninstall-with-arn"],
)
def test_the_command_line_requires_the_arn_exactly_for_a_removal(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        preflight.main(arguments)

    assert caught.value.code == 2, "a usage error exits 2"
