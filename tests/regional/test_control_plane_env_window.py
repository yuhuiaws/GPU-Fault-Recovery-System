"""Unit tests for the DESTR-018 control-worker env window helper.

The helper is the only thing in the case that writes to the control plane, so
its refusals are the case's blast-radius boundary: an allow-list that is exactly
the timing set the control plane's boot validator binds together, a bounded
value range that never exceeds a shipped default, a cross-variable rule that
keeps the failure the case measures a *lifetime* miss rather than an
execution-timeout miss, the validator itself run on the completed set, and a
restore that puts an absent variable back to absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.execution.config import validate_timing_from_environment
from scripts.e2e.regional import control_plane_env_window as env_window

LIFETIME = env_window.LIFETIME_VARIABLE
TIMEOUT = env_window.EXECUTION_TIMEOUT_VARIABLE


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


def test_allowlist_is_the_timing_set_the_boot_validator_binds_together() -> None:
    assignments = env_window.parse_assignments([f"{LIFETIME}=300", f"{TIMEOUT}=300"])

    assert assignments == {LIFETIME: "300", TIMEOUT: "300"}, (
        "parse keeps what the operator chose; open_window completes the set"
    )
    assert env_window.CHOSEN_VARIABLES == (LIFETIME, TIMEOUT)
    assert env_window.ALLOWED_VARIABLES == (
        LIFETIME,
        TIMEOUT,
        "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS",
        "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS",
        "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS",
        "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS",
        "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS",
    )
    assert env_window.PLANE == "cpu"
    assert env_window.DEPLOYMENT == "gpu-fault-control-worker"
    assert env_window.CONTAINER == "control-worker"
    assert env_window.OPEN_CONFIRMATION == "OPEN_CONTROL_PLANE_ENV_WINDOW"
    assert env_window.CLOSE_CONFIRMATION == "CLOSE_CONTROL_PLANE_ENV_WINDOW"


@pytest.mark.parametrize(
    "pairs",
    [
        # Not on the allow-list: the variables the drill must not touch even
        # though they would also make the workflow end sooner.
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=6"],
        ["GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS=1"],
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
        # A derived variable is never raised above what the release ships.
        [f"{LIFETIME}=300", "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS=601"],
        [f"{LIFETIME}=300", "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS=1901"],
        # The same name twice cannot be resolved by order.
        [f"{LIFETIME}=300", f"{LIFETIME}=600"],
    ],
)
def test_assignments_outside_the_contract_are_refused(pairs: list[str]) -> None:
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(pairs)


def test_execution_timeout_below_the_lifetime_is_refused() -> None:
    """The deadline that fires first decides which failure the case records.

    ``claim_deadlines`` stamps ``min(execution, lifetime)``, and
    ``workflow_deadline_failure`` only marks ``workflow_lifetime_exceeded`` when
    the lifetime is the bound that passed. A window that lowered the lifetime to
    300 while the execution timeout stayed at 240 would produce a workflow that
    FAILED on time but with the wrong reason -- a green-looking run that proves
    nothing.
    """

    errors = env_window.assignment_errors({LIFETIME: "300", TIMEOUT: "240"})

    assert len(errors) == 1, errors
    assert "workflow_lifetime_exceeded" in errors[0]
    assert env_window.assignment_errors({LIFETIME: "300", TIMEOUT: "300"}) == []
    # An execution timeout above the lifetime is what the control plane itself
    # refuses at boot (claim_deadlines would silently truncate it).
    above = env_window.assignment_errors({LIFETIME: "300", TIMEOUT: "600"})
    assert len(above) == 1 and "exceeds" in above[0], above
    # The lifetime alone derives the execution timeout and the rest of the set.
    assert env_window.assignment_errors({LIFETIME: "300"}) == []
    assert env_window.assignment_errors({}) == []
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments([f"{LIFETIME}=300", f"{TIMEOUT}=240"])


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
        **{
            name: {"present": False, "value": None}
            for name in env_window.ALLOWED_VARIABLES
        },
        TIMEOUT: {"present": True, "value": "1800"},
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


# --- the lockstep set (F4) ------------------------------------------------------------
#
# ``validate_timing_relationships`` fails closed at boot when the node lifetime
# is lowered under any per-step ceiling (managed recovery 1800 s, the three
# installs 1900 s), when an override sits below the default step cap, when the
# lease is not below the execution timeout, and (F2) when the lifetime holds the
# install ceiling but not the install floor. A window that moved only the
# lifetime and the execution timeout therefore CrashLooped the control plane.


def test_a_compressed_window_rendered_by_the_runner_passes_the_boot_validator() -> None:
    assignments = env_window.lockstep_assignments(180, 180)

    assert validate_timing_from_environment(assignments) == [], assignments
    assert assignments[LIFETIME] == "180" and assignments[TIMEOUT] == "180"
    assert set(assignments) == set(env_window.ALLOWED_VARIABLES), (
        "every variable the window may write is derived, so the set moves as one"
    )
    for lifetime in (60, 120, 600, 1900, 2400, 3600):
        rendered = env_window.lockstep_assignments(lifetime, lifetime)
        assert validate_timing_from_environment(rendered) == [], rendered


def test_the_lockstep_set_carries_the_install_knobs() -> None:
    install = "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS"
    containment = "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS"
    rendered = env_window.lockstep_assignments(180, 180)

    assert install in env_window.ALLOWED_VARIABLES
    assert containment in env_window.ALLOWED_VARIABLES
    # Every ceiling equals the lifetime: above it the validator refuses to
    # boot, below it the step's own cap -- not the lifetime the drill measures
    # -- would end the wait first.
    assert rendered[install] == "180"
    assert rendered["GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS"] == "180"
    assert rendered["GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS"] == "180"
    assert rendered["GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS"] == "90", (
        "the warning keeps the shipped half-of-the-cap lead"
    )
    assert rendered[containment] == "0", (
        "a 180 s lifetime leaves nothing above the install ceiling"
    )
    assert rendered["GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS"] == "90", (
        "the lease is lowered under the execution timeout"
    )


def test_the_lockstep_set_never_raises_a_knob_above_its_shipped_default() -> None:
    rendered = env_window.lockstep_assignments(3600)

    assert rendered == {
        LIFETIME: "3600",
        TIMEOUT: "1800",
        "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS": "600",
        "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS": "300",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS": "1800",
        "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS": "1900",
        "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS": "600",
        "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS": "180",
    }, "an uncompressed window is exactly the shipped defaults"
    for name, value in env_window.lockstep_assignments(2400, 1800).items():
        assert int(value) <= int(env_window.SHIPPED_DEFAULTS[name]), (name, value)
    for name in env_window.ALLOWED_VARIABLES:
        assert env_window.value_bounds(name)[1] >= int(
            env_window.SHIPPED_DEFAULTS[name]
        ), name


def test_completing_a_two_variable_request_fills_the_rest_of_the_set() -> None:
    """The DESTR-018 driver still hands over only the lifetime and the
    execution timeout; the runner derives the others so the control plane
    comes back up. An explicit value wins over the derived one."""

    completed = env_window.complete_assignments({LIFETIME: "180", TIMEOUT: "180"})

    assert completed == env_window.lockstep_assignments(180, 180)
    explicit = env_window.complete_assignments(
        {
            LIFETIME: "180",
            TIMEOUT: "180",
            "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS": "60",
        }
    )
    assert explicit["GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS"] == "60"
    assert env_window.complete_assignments({}) == {}, (
        "without a lifetime there is nothing to derive from"
    )


def test_assignment_errors_report_what_the_control_plane_would_refuse_at_boot() -> None:
    errors = env_window.assignment_errors(
        {
            LIFETIME: "180",
            TIMEOUT: "180",
            "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS": "180",
        }
    )

    assert len(errors) == 1, errors
    assert "workflow lease duration" in errors[0], errors[0]
    assert env_window.assignment_errors({LIFETIME: "180", TIMEOUT: "180"}) == []
    with pytest.raises(env_window.RegionalFixtureError, match="lease duration"):
        env_window.parse_assignments(
            [
                f"{LIFETIME}=180",
                f"{TIMEOUT}=180",
                "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS=180",
            ]
        )


def test_open_records_and_converges_on_the_completed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fixture(FakeRegional):
        def __init__(self) -> None:
            super().__init__(_deployment([]))
            self.env_arguments: list[str] = []

        def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
            self.calls.append((plane, *arguments))
            if arguments[:2] == ("set", "env"):
                self.env_arguments = list(arguments[4:])
                return ""
            if arguments[:2] == ("rollout", "status"):
                return "deployment rolled out"
            return json.dumps(self.deployment)

        def ready_pods(self, plane: str, deployment: str) -> list[dict[str, Any]]:
            return []

    regional = Fixture()
    expected = env_window.lockstep_assignments(180, 180)

    def _converged(settings: Any, fixture: Any, wanted: dict, *, sleep: Any) -> list:
        assert wanted == expected, wanted
        return [{"pod": "a", "values": dict(wanted)}]

    monkeypatch.setattr(env_window, "converge", _converged)
    record = env_window.open_window(
        env_window.Settings(baseline=tmp_path / "b.json", rollout_timeout_seconds=1),
        regional,
        {"deployment": env_window.deployment_env(regional)},
        {LIFETIME: "180", TIMEOUT: "180"},
    )

    assert record["assignments"] == expected, record["assignments"]
    assert sorted(regional.env_arguments) == sorted(
        f"{name}={value}" for name, value in expected.items()
    )
    baseline = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert baseline["assignments"] == expected
    assert set(baseline["baseline"]["variables"]) == set(
        env_window.ALLOWED_VARIABLES
    ), (
        "the baseline records every variable the window may write, so --close "
        "deletes each one it added"
    )


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
