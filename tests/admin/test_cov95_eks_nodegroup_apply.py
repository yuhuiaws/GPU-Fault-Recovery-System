"""``prepare-eks-nodegroup`` apply at the edges the recorded world never reaches.

The per-group journal refuses a file it cannot read, one of another shape and
one that journals another cluster, and keeps attempts in memory when there is
no state directory. ``failure_text`` names unknown errors without a hint and
skips entries that are not mappings. ``settled_nodegroup`` waits a bounded
time for a group still ``UPDATING``. ``apply_group`` stops without writing on a
plan EKS already refused or a group that is no longer ``ACTIVE``, refuses a
launch template version or an update it cannot identify, and resumes a
journaled update: a failed one is the stop, one still running keeps its
journal entry, and an unexpected EKS error is raised rather than read as gone.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import eks_nodegroup_apply as apply
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.eks_nodegroup_plan import (
    CHANGES,
    IN_PROGRESS,
    REFUSED,
    STEP_LAUNCH_TEMPLATE,
    STEP_ROLL,
    STEP_UPDATE_CONFIG,
    ClusterFacts,
    GroupPlan,
    Step,
)

REGION = "us-west-2"
CLUSTER = ClusterFacts(
    arn=f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-eks",
    region=REGION,
    name="gpu-eks",
    nodegroups={},
)
GROUP = "gpu-group"
TEMPLATE = {"id": "lt-0gpu", "name": "gpu-template"}
NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
ACTIVE = {
    "status": "ACTIVE",
    "launchTemplate": {"id": "lt-0gpu", "version": "3"},
    "capacityType": "SPOT",
}
UPDATE_CONFIG = Step(
    STEP_UPDATE_CONFIG,
    "label the group",
    ("eks", "update-nodegroup-config", "--cluster-name", "gpu-eks"),
)
LAUNCH_TEMPLATE = Step(
    STEP_LAUNCH_TEMPLATE,
    "create the prerequisites version",
    ("ec2", "create-launch-template-version", "--launch-template-id", "lt-0gpu"),
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Eks(CommandRunner):
    """Scripted ``aws`` answers per operation; the last answer repeats."""

    def __init__(self, **answers: list[Any]) -> None:
        super().__init__()
        self.answers = {
            name.replace("_", "-"): list(value) for name, value in answers.items()
        }
        self.calls: list[tuple[str, bool]] = []

    def aws_json(self, region, *arguments, mutate=False, **_options) -> dict:
        assert region == REGION, "apply left the cluster's Region"
        operation = arguments[1]
        self.calls.append((operation, mutate))
        queue = self.answers.get(operation)
        assert queue, f"unexpected AWS call: {operation}"
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def writes(self) -> list[str]:
        return [operation for operation, mutate in self.calls if mutate]


def context(runner: Eks, root: Path | None, clock: Clock, **overrides: Any):
    values: dict[str, Any] = {
        "runner": runner,
        "cluster": CLUSTER,
        "root": root,
        "reference": "CHG-1234",
        "operator": "operator@example.com",
        "site_id": "site-a",
        "roll_timeout_seconds": 120.0,
        "sleep": clock.sleep,
        "clock": clock.clock,
        "now": lambda: NOW,
        **overrides,
    }
    return apply.ApplyContext(**values)


def plan(**overrides: Any) -> GroupPlan:
    values: dict[str, Any] = {
        "name": GROUP,
        "document": dict(ACTIVE),
        "verdict": CHANGES,
        "template": TEMPLATE,
        "current_version": 3,
        "steps": (UPDATE_CONFIG,),
        **overrides,
    }
    return GroupPlan(**values)


def journal_document(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(result["journal"]).read_text(encoding="utf-8"))


def never_replanned() -> GroupPlan:
    raise AssertionError("the group was planned again after a stopping resume")


# ------------------------------------------------------------------- journal


def test_a_journal_that_names_another_cluster_is_refused(tmp_path: Path) -> None:
    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    journal.begin({"reference": "CHG-1"})
    assert journal.path is not None and journal.path.is_file()
    document = json.loads(journal.path.read_text())
    document["cluster_arn"] = CLUSTER.arn.replace("gpu-eks", "other-eks")
    journal.path.write_text(json.dumps(document))

    with pytest.raises(BootstrapError, match="journals another cluster"):
        apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("{not json", "cannot read the journal"),
        ("[]", "has an unexpected shape"),
        ('{"attempts": {"1": {}}}', "has an unexpected shape"),
    ],
    ids=["not-json", "not-a-mapping", "attempts-not-a-list"],
)
def test_an_unreadable_journal_is_refused_by_path(
    tmp_path: Path, content: str, reason: str
) -> None:
    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    assert journal.path is not None
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text(content)

    with pytest.raises(BootstrapError, match=reason) as raised:
        apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    assert str(journal.path) in str(raised.value)


def test_a_journal_without_a_state_directory_stays_in_memory(tmp_path: Path) -> None:
    journal = apply.Journal(None, cluster=CLUSTER, nodegroup=GROUP)

    journal.begin({"reference": "CHG-1"})
    journal.step({"step": STEP_UPDATE_CONFIG})
    journal.pending({"update_id": "u-1", "step": STEP_ROLL})
    journal.finish("COMPLETED", at="2030-01-02T03:04:05+00:00")

    assert journal.path is None
    assert journal.pending_update() == {"update_id": "u-1", "step": STEP_ROLL}
    assert journal.document["attempts"] == [
        {
            "attempt": 1,
            "reference": "CHG-1",
            "steps": [{"step": STEP_UPDATE_CONFIG}],
            "outcome": "COMPLETED",
            "finished_at": "2030-01-02T03:04:05+00:00",
        }
    ]
    assert list(tmp_path.iterdir()) == [], "a journal without a root wrote a file"


# ------------------------------------------------------------- update text


def test_failure_text_names_unknown_errors_without_a_hint() -> None:
    text = apply.failure_text(
        {
            "errors": [
                "not-a-mapping",
                {"errorCode": "SomethingNew", "errorMessage": "it broke"},
                {"errorMessage": "no code given"},
                {"errorCode": "PodEvictionFailure"},
            ]
        }
    )

    parts = text.split("; ")
    assert parts[:2] == ["SomethingNew: it broke", "unknown: no code given"]
    assert parts[2].startswith("PodEvictionFailure: -- a PodDisruptionBudget"), parts
    assert apply.failure_text({"errors": ["only-text"]}) == "no error detail"


# ------------------------------------------------------------------- settle


def test_a_finishing_update_is_waited_out_before_the_group_is_judged() -> None:
    clock = Clock()
    runner = Eks(
        describe_nodegroup=[
            {"nodegroup": {"status": "UPDATING"}},
            {"nodegroup": {"status": "UPDATING"}},
            {"nodegroup": dict(ACTIVE)},
        ]
    )

    document = apply.settled_nodegroup(context(runner, None, clock), GROUP)

    assert document["status"] == "ACTIVE"
    assert clock.sleeps == [apply.SETTLE_INTERVAL_SECONDS] * 2


def test_the_settle_wait_is_bounded_for_a_group_stuck_updating() -> None:
    clock = Clock()
    runner = Eks(describe_nodegroup=[{"nodegroup": {"status": "UPDATING"}}])

    document = apply.settled_nodegroup(context(runner, None, clock), GROUP)

    assert document["status"] == "UPDATING"
    assert sum(clock.sleeps) == apply.SETTLE_SECONDS
    assert runner.writes() == []


# -------------------------------------------------------------- apply stops


def test_a_refused_plan_stops_without_any_aws_call(tmp_path: Path) -> None:
    runner = Eks()

    result = apply.apply_group(
        context(runner, tmp_path, Clock()),
        plan(verdict=REFUSED, refusal="the group runs AL2 AMIs", steps=()),
        replan=never_replanned,
    )

    assert result["outcome"] == REFUSED
    assert result["reason"] == "the group runs AL2 AMIs"
    assert runner.calls == []
    assert journal_document(result)["attempts"][-1]["outcome"] == REFUSED


@pytest.mark.parametrize(
    ("nodegroup", "named"), [({"status": "DEGRADED"}, "DEGRADED"), ({}, "unknown")]
)
def test_a_group_that_left_active_is_not_written(
    tmp_path: Path, nodegroup: dict[str, Any], named: str
) -> None:
    runner = Eks(describe_nodegroup=[{"nodegroup": nodegroup}])

    result = apply.apply_group(
        context(runner, tmp_path, Clock()), plan(), replan=never_replanned
    )

    assert result["outcome"] == apply.CHANGED
    assert result["reason"] == (
        f"node group {GROUP} is {named}: nothing more is written; rerun to plan again"
    )
    assert runner.writes() == []


@pytest.mark.parametrize(
    "created",
    [{}, {"LaunchTemplateVersion": {"VersionNumber": "4"}}],
    ids=["no-version", "version-not-a-number"],
)
def test_an_unidentified_launch_template_version_fails_the_attempt(
    tmp_path: Path, created: dict[str, Any]
) -> None:
    runner = Eks(create_launch_template_version=[created])
    group = plan(steps=(LAUNCH_TEMPLATE,))

    with pytest.raises(BootstrapError, match="returned an unexpected version"):
        apply.apply_group(
            context(runner, tmp_path, Clock()), group, replan=never_replanned
        )

    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    attempt = journal.document["attempts"][-1]
    assert attempt["outcome"] == apply.FAILED
    assert attempt["steps"] == [], "an unidentified version was journaled as created"


def test_an_update_without_an_id_is_never_waited_for(tmp_path: Path) -> None:
    runner = Eks(
        describe_nodegroup=[{"nodegroup": dict(ACTIVE)}],
        update_nodegroup_config=[{"update": {"status": "InProgress"}}],
    )

    with pytest.raises(BootstrapError, match="update-config of gpu-group returned no"):
        apply.apply_group(
            context(runner, tmp_path, Clock()), plan(), replan=never_replanned
        )

    assert ("describe-update", False) not in runner.calls
    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    assert journal.pending_update() is None
    assert journal.document["attempts"][-1]["outcome"] == apply.FAILED


# ------------------------------------------------------------------ resume


def journaled(tmp_path: Path, update_id: str = "u-roll-1") -> GroupPlan:
    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    journal.begin({"reference": "CHG-0"})
    pending = {"update_id": update_id, "step": STEP_ROLL, "started_at": "earlier"}
    journal.pending(pending)
    return plan(
        verdict=IN_PROGRESS,
        steps=(),
        pending_update=pending,
        document={"status": "UPDATING"},
    )


def test_a_failed_journaled_roll_is_the_stop_and_is_not_planned_again(
    tmp_path: Path,
) -> None:
    group = journaled(tmp_path)
    runner = Eks(
        describe_update=[
            {
                "update": {
                    "status": "Failed",
                    "errors": [
                        {"errorCode": "NodeCreationFailure", "errorMessage": "no Spot"}
                    ],
                }
            }
        ]
    )

    result = apply.apply_group(
        context(runner, tmp_path, Clock()), group, replan=never_replanned
    )

    assert result["outcome"] == apply.ROLL_FAILED
    reason = result["reason"]
    assert "u-roll-1 (roll) ended Failed: NodeCreationFailure: no Spot" in reason
    assert reason.endswith(
        "rerun the same command later (the prepared launch template version is reused)"
    ), "a retryable roll failure must tell the operator to rerun later"
    assert [step["step"] for step in result["steps"]] == ["resume-roll"]
    assert journal_document(result)["pending_update"] is None
    assert runner.writes() == []


def test_a_journaled_roll_still_running_keeps_its_journal_entry(tmp_path: Path) -> None:
    group = journaled(tmp_path)
    clock = Clock()
    runner = Eks(describe_update=[{"update": {"status": "InProgress"}}])

    result = apply.apply_group(
        context(runner, tmp_path, clock, roll_timeout_seconds=60.0),
        group,
        replan=never_replanned,
    )

    assert result["outcome"] == apply.ROLL_IN_PROGRESS
    assert "still in progress after 60.0 s" in result["reason"]
    assert journal_document(result)["pending_update"]["update_id"] == "u-roll-1", (
        "the next run could not resume an update still in flight"
    )
    assert sum(clock.sleeps) == 60.0


def test_an_unexpected_eks_error_on_resume_is_raised_not_read_as_gone(
    tmp_path: Path,
) -> None:
    group = journaled(tmp_path)
    runner = Eks(
        describe_update=[BootstrapError("An error occurred (AccessDeniedException)")]
    )

    with pytest.raises(BootstrapError, match="AccessDeniedException"):
        apply.apply_group(
            context(runner, tmp_path, Clock()), group, replan=never_replanned
        )

    journal = apply.Journal(tmp_path, cluster=CLUSTER, nodegroup=GROUP)
    assert journal.pending_update() is not None, "an unknown error cleared the update"
    assert journal.document["attempts"][-1]["outcome"] == apply.FAILED
