"""Unit tests for the DESTR-018 control-worker env window helper.

The helper is the only thing in the case that writes to the control plane, so
its refusals are the case's blast-radius boundary: an allow-list of the six
interlocking workflow-timing knobs, a bounded value range, the full set of
cross-variable rules that keep a compressed window *bootable* (the control
plane's start-up guard CrashLoopBackOffs a lifetime that sits below the step
ceilings, the managed-recovery window or the lease) and keep the failure the
case measures a *lifetime* miss rather than an execution-timeout miss, and a
restore that puts an absent variable back to absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import control_plane_env_window as env_window

LIFETIME = env_window.LIFETIME_VARIABLE
TIMEOUT = env_window.EXECUTION_TIMEOUT_VARIABLE
STEP = env_window.STEP_TIMEOUT_VARIABLE
MANAGED = env_window.MANAGED_RECOVERY_VARIABLE
WARNING = env_window.STEP_WARNING_VARIABLE
LEASE = env_window.LEASE_DURATION_VARIABLE

# The full internally consistent compressed set the runner opens the window
# with: lifetime == execution == step timeout == managed recovery, the step
# warning below the step timeout, the lease below the execution timeout. Every
# other test that needs a *valid* window starts from this and perturbs one knob.
FULL_SET = {
    LIFETIME: "180",
    TIMEOUT: "180",
    STEP: "180",
    MANAGED: "180",
    WARNING: "150",
    LEASE: "120",
}


class FakeRegional:
    """Enough of ``RegionalLiveFixture`` for the read paths under test."""

    def __init__(self, deployment: dict[str, Any]) -> None:
        self.deployment = deployment
        self.calls: list[tuple[str, ...]] = []

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        self.calls.append((plane, *arguments))
        return json.dumps(self.deployment)


def _deployment(
    env: list[dict[str, Any]], *, container: str = "control-worker"
) -> dict:
    return {
        "metadata": {"generation": 7, "resourceVersion": "1234"},
        "spec": {
            "replicas": 2,
            "template": {"spec": {"containers": [{"name": container, "env": env}]}},
        },
    }


def test_allowlist_accepts_the_six_workflow_timing_knobs() -> None:
    assignments = env_window.parse_assignments(
        [f"{name}={value}" for name, value in FULL_SET.items()]
    )

    assert assignments == FULL_SET
    assert env_window.ALLOWED_VARIABLES == (
        LIFETIME,
        TIMEOUT,
        STEP,
        MANAGED,
        WARNING,
        LEASE,
    )
    assert env_window.PLANE == "cpu"
    assert env_window.DEPLOYMENT == "gpu-fault-control-worker"
    assert env_window.CONTAINER == "control-worker"
    assert env_window.OPEN_CONFIRMATION == "OPEN_CONTROL_PLANE_ENV_WINDOW"
    assert env_window.CLOSE_CONFIRMATION == "CLOSE_CONTROL_PLANE_ENV_WINDOW"


@pytest.mark.parametrize(
    "pairs",
    [
        # Not on the allow-list: knobs the drill must not touch even though they
        # would also make the workflow end sooner. The verify attempt budget is
        # deliberately never lowered (it would hide the deadline behind the
        # attempt budget), and the job lifetime is a different bound.
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=60"],
        ["GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS=300"],
        # Malformed or out of range.
        [f"{LIFETIME}"],
        [f"{LIFETIME}="],
        [f"{LIFETIME}=five"],
        [f"{LIFETIME}=0"],
        [f"{LIFETIME}=-300"],
        [f"{LIFETIME}=59"],
        [f"{LIFETIME}=3601"],
        [f"{LIFETIME}=300.0"],
        # The same name twice cannot be resolved by order.
        [f"{LIFETIME}=300", f"{LIFETIME}=600"],
    ],
)
def test_assignments_outside_the_contract_are_refused(pairs: list[str]) -> None:
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(pairs)


def test_the_full_consistent_compressed_set_is_accepted() -> None:
    assert env_window.assignment_errors(FULL_SET) == []
    # No window at all: the shipped defaults are internally consistent, so an
    # empty set carries no error and the read-only survey path stays clean.
    assert env_window.assignment_errors({}) == []


def test_compressing_the_lifetime_alone_is_refused() -> None:
    """The live FAIL this fix exists for: lowering only the lifetime.

    Setting the lifetime (and execution timeout) to 180 while the step timeout
    and managed-recovery window stayed at their 600s/1800s defaults is exactly
    what CrashLoopBackOff'd every control-worker replica on the live run --
    ``validate_timing_relationships`` refuses a step waiting ceiling above the
    node lifetime. The helper now rejects it on the command line instead.
    """

    errors = env_window.assignment_errors({LIFETIME: "180", TIMEOUT: "180"})

    assert errors, "compressing the lifetime alone must be refused"
    assert any(STEP in message and "boot" in message for message in errors), errors
    assert any(MANAGED in message for message in errors), errors
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments([f"{LIFETIME}=180", f"{TIMEOUT}=180"])


def test_execution_timeout_below_the_lifetime_is_refused() -> None:
    """The deadline that fires first decides which failure the case records.

    ``claim_deadlines`` stamps ``min(execution, lifetime)``, and
    ``workflow_deadline_failure`` only marks ``workflow_lifetime_exceeded`` when
    the lifetime is the bound that passed. A window that lowered the lifetime to
    180 while the execution timeout stayed at 150 would produce a workflow that
    FAILED on time but with the wrong reason -- a green-looking run that proves
    nothing.
    """

    # A set consistent in every other rule (lease 120 < execution 150), so the
    # only violation is the execution timeout sitting below the lifetime.
    errors = env_window.assignment_errors(
        {**FULL_SET, TIMEOUT: "150", LEASE: "120"}
    )

    assert len(errors) == 1, errors
    assert "workflow_lifetime_exceeded" in errors[0]
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(
            [f"{name}={value}" for name, value in {**FULL_SET, TIMEOUT: "150"}.items()]
        )


def test_execution_timeout_above_the_lifetime_is_refused() -> None:
    """Above the lifetime, claim_deadlines silently truncates the value, so an
    execution timeout larger than the lifetime is a no-op dressed as a change."""

    errors = env_window.assignment_errors({**FULL_SET, TIMEOUT: "300"})

    assert any("truncated" in message for message in errors), errors


@pytest.mark.parametrize(
    "override, needle",
    [
        # A step waiting ceiling above the node lifetime: control plane refuses.
        ({STEP: "300", MANAGED: "300"}, "refuse to boot"),
        # Managed recovery above the node lifetime.
        ({MANAGED: "300"}, MANAGED),
        # Managed recovery below the default step timeout (from_mapping).
        ({STEP: "120", WARNING: "120", MANAGED: "100"}, "below"),
        # Step warning above the step timeout (from_mapping).
        ({WARNING: "200"}, WARNING),
        # Lease at or above the execution timeout.
        ({LEASE: "180"}, "not below"),
    ],
)
def test_each_ordering_rule_refuses_its_violation(
    override: dict[str, str], needle: str
) -> None:
    errors = env_window.assignment_errors({**FULL_SET, **override})

    assert any(needle in message for message in errors), (override, errors)
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(
            [f"{name}={value}" for name, value in {**FULL_SET, **override}.items()]
        )


def test_a_lease_over_the_job_lifetime_bound_is_refused() -> None:
    """Twice the managed-recovery ceiling must fit inside the job lifetime.

    A managed-recovery window past half the 3600s job lifetime would let a
    reboot-then-delegate branch be failed by the job bound; the rule fires only
    when the window is not compressing the lifetime below it, so it is exercised
    on a set that lifts managed recovery without compressing the lifetime.
    """

    errors = env_window.assignment_errors({MANAGED: "1900"})

    assert any(str(env_window.JOB_LIFETIME_SECONDS) in message for message in errors), (
        errors
    )


def test_restore_puts_an_absent_variable_back_to_absent() -> None:
    baseline = {
        "variables": {
            LIFETIME: {"present": False, "value": None},
            TIMEOUT: {"present": True, "value": "1800"},
        }
    }

    assert env_window.restore_arguments(baseline) == [f"{LIFETIME}-", f"{TIMEOUT}=1800"]
    assert env_window.open_arguments({TIMEOUT: "300", LIFETIME: "300"}) == [
        f"{LIFETIME}=300",
        f"{TIMEOUT}=300",
    ]
    assert env_window.restore_arguments({}) == []


def test_convergence_requires_every_ready_replica() -> None:
    expected = {LIFETIME: "300", TIMEOUT: "300"}
    settled = [
        {"pod": "a", "values": {LIFETIME: "300", TIMEOUT: "300"}},
        {"pod": "b", "values": {LIFETIME: "300", TIMEOUT: "300"}},
    ]

    assert env_window.converged(settled, expected) is True
    lagging = [settled[0], {"pod": "b", "values": {LIFETIME: "3600", TIMEOUT: "300"}}]
    assert env_window.converged(lagging, expected) is False
    assert env_window.converged([], expected) is False, (
        "zero ready replicas is not convergence; the workers are mid rollout"
    )
    restored = {LIFETIME: None, TIMEOUT: None}
    assert env_window.converged([{"pod": "a", "values": {}}], restored) is True, (
        "an unset variable reads back as None on every replica"
    )


def test_observed_value_refuses_to_average_a_disagreement() -> None:
    poll = "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS"
    agreed = [
        {"pod": "a", "values": {poll: "5.0"}},
        {"pod": "b", "values": {poll: "5.0"}},
    ]

    assert env_window.observed_value(agreed, poll) == "5.0"
    split = [{"pod": "a", "values": {poll: "5.0"}}, {"pod": "b", "values": {poll: "1"}}]
    assert env_window.observed_value(split, poll) is None, (
        "a mid-rollout split must not yield a cadence a case computes a margin from"
    )
    assert env_window.observed_value([], poll) is None
    assert env_window.observed_value([{"pod": "a", "values": {}}], poll) is None
    assert env_window.SURVEYED_VARIABLES[:2] == (LIFETIME, TIMEOUT)
    assert poll in env_window.OBSERVED_VARIABLES
    for name in env_window.OBSERVED_VARIABLES:
        assert name not in env_window.ALLOWED_VARIABLES, (
            f"{name} is only observed and must never be writable"
        )


def test_open_refuses_unrecorded_live_values_and_resumes_its_own() -> None:
    baseline = {"variables": {LIFETIME: {"present": False, "value": None}}}
    live_changed = {"variables": {LIFETIME: {"present": True, "value": "300"}}}
    assignments = {LIFETIME: "300"}

    assert env_window.open_decision(None, live_changed, assignments) == "refuse", (
        "a live window nobody recorded cannot be adopted; its baseline is lost"
    )
    assert env_window.open_decision(None, baseline, assignments) == "open"
    record = {"opened_at": "x", "baseline": baseline, "assignments": assignments}
    assert env_window.open_decision(record, live_changed, assignments) == "resume"
    assert env_window.open_decision(record, baseline, assignments) == "refuse"
    closed = {**record, "closed_at": "y"}
    assert env_window.open_decision(closed, baseline, assignments) == "open"
    assert env_window.open_decision(record, live_changed, {LIFETIME: "600"}) == "refuse"


def test_deployment_env_reads_the_worker_container() -> None:
    regional = FakeRegional(_deployment([{"name": TIMEOUT, "value": "1800"}]))

    value = env_window.deployment_env(regional)

    assert value["deployment"] == "gpu-fault-control-worker"
    assert value["container"] == "control-worker"
    assert value["plane"] == "cpu"
    assert value["generation"] == 7
    assert value["resource_version"] == "1234"
    assert value["replicas"] == 2
    assert value["variables"] == {
        LIFETIME: {"present": False, "value": None},
        TIMEOUT: {"present": True, "value": "1800"},
        STEP: {"present": False, "value": None},
        MANAGED: {"present": False, "value": None},
        WARNING: {"present": False, "value": None},
        LEASE: {"present": False, "value": None},
    }
    assert regional.calls == [
        ("cpu", "get", "deployment", "gpu-fault-control-worker", "-o", "json")
    ]


def test_deployment_env_refuses_a_reference_value() -> None:
    regional = FakeRegional(
        _deployment(
            [{"name": LIFETIME, "valueFrom": {"configMapKeyRef": {"name": "c"}}}]
        )
    )

    with pytest.raises(env_window.RegionalFixtureError, match="reference"):
        env_window.deployment_env(regional)


def test_deployment_env_refuses_an_unknown_container() -> None:
    regional = FakeRegional(_deployment([], container="api"))

    with pytest.raises(env_window.RegionalFixtureError, match="control-worker"):
        env_window.deployment_env(regional)


def test_read_baseline_refuses_a_missing_record(tmp_path: Path) -> None:
    with pytest.raises(env_window.RegionalFixtureError, match="baseline"):
        env_window.read_baseline(tmp_path / "missing.json")
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"opened_at": "x"}), encoding="utf-8")
    assert env_window.read_baseline(path) == {"opened_at": "x"}


def test_reports_drop_the_survey_bodies() -> None:
    record = {
        "opened_at": "x",
        "pre_window_survey": {"replicas": []},
        "close_survey": {"replicas": []},
        "assignments": {LIFETIME: "300"},
    }

    assert env_window.without_survey(record) == {
        "opened_at": "x",
        "assignments": {LIFETIME: "300"},
    }


def test_parser_is_read_only_by_default() -> None:
    parser = env_window.parser()

    arguments = parser.parse_args(["--baseline", "/tmp/b.json"])
    assert arguments.open is False
    assert arguments.close is False
    assert arguments.rollout_timeout_seconds == 600
    opened = parser.parse_args(
        [
            "--baseline",
            "/tmp/b.json",
            "--open",
            "--confirm",
            env_window.OPEN_CONFIRMATION,
            "--set",
            f"{LIFETIME}=300",
            "--set",
            f"{TIMEOUT}=300",
        ]
    )
    assert opened.open is True
    assert len(opened.set) == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline", "/tmp/b.json", "--open", "--close"])


def test_configure_requires_a_baseline_path(tmp_path: Path) -> None:
    parser = env_window.parser()

    settings = env_window.configure(
        parser.parse_args(
            ["--baseline", str(tmp_path / "b.json"), "--rollout-timeout-seconds", "300"]
        )
    )

    assert settings.baseline == tmp_path / "b.json"
    assert settings.rollout_timeout_seconds == 300
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.configure(parser.parse_args([]))


class _RollingRegional:
    """A control-worker mid-rollout: ``ready_pods`` lists a replica that is
    then terminated, so its exec 404s while a surviving replica answers."""

    def __init__(
        self,
        vanished: str,
        survivor_values: dict[str, str],
        *,
        stderr: str = 'Error from server (NotFound): pods "gone" not found',
    ) -> None:
        self._vanished = vanished
        self._survivor_values = survivor_values
        self._stderr = stderr
        self.exec_targets: list[str] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": self._vanished}, {"name": "survivor"}]

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        assert arguments[0] == "exec"
        target = arguments[1]
        self.exec_targets.append(target)
        if target == self._vanished:
            raise env_window.RegionalFixtureError(
                f"command failed (1): kubectl ... exec {target} ...; stderr={self._stderr}"
            )
        return json.dumps(self._survivor_values)


@pytest.mark.parametrize(
    "stderr",
    [
        'Error from server (NotFound): pods "gone-9b7cl" not found',
        "error: cannot exec into a container in a completed pod; "
        "current phase is Succeeded",
        'unable to upgrade connection: container not found ("control-worker")',
        "error: Internal error occurred: error executing command in container: "
        "container is not running",
    ],
)
def test_replica_env_skips_a_replica_that_rolled_away_mid_survey(stderr: str) -> None:
    values = {LIFETIME: "180", TIMEOUT: "180"}
    regional = _RollingRegional("gone-9b7cl", values, stderr=stderr)

    replicas = env_window.replica_env(
        regional,
        plane=env_window.PLANE,
        deployment=env_window.DEPLOYMENT,
        names=(LIFETIME, TIMEOUT),
    )

    assert regional.exec_targets == ["gone-9b7cl", "survivor"], (
        "the vanished replica must be attempted before being dropped"
    )
    assert replicas == [{"pod": "survivor", "values": values}], (
        "a NotFound on one replica during a rollout must not fail the survey"
    )


def test_replica_env_still_raises_a_real_exec_failure() -> None:
    class _Broken(_RollingRegional):
        def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
            self.exec_targets.append(arguments[1])
            raise env_window.RegionalFixtureError(
                "command failed (1): kubectl ... exec ...; stderr=OCI runtime exec "
                "failed: exec failed: unable to start container process"
            )

    regional = _Broken("a", {LIFETIME: "180"})
    with pytest.raises(env_window.RegionalFixtureError, match="OCI runtime"):
        env_window.replica_env(
            regional,
            plane=env_window.PLANE,
            deployment=env_window.DEPLOYMENT,
            names=(LIFETIME,),
        )
