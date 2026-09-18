"""NET command runner lifecycle with real evidence predicates and fake transport."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_net002_command_recovery as net002
from scripts.e2e.regional import run_net003_result_retry as net003
from scripts.e2e.regional import run_net006_lease_loss_withheld_result as net006
from tests.regional import _cov95_net_commands as command_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional._cov95_net_commands import run_command

command_runner = command_support.command_runner


def result(harness: Any) -> Any:
    return json.loads(
        (
            harness.root
            / "cases"
            / harness.module.CASE_ID
            / f"{harness.module.CASE_ID}.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "preflight",
        "registry",
        "probe",
        "seed",
        "wait-file",
        "read-state",
        "wait-command",
        "cleanup",
        "pod-phase",
    ],
)
def test_run_case_cleans_every_started_stage_and_records_outcome(
    command_runner: Any, failure: str | None
) -> None:
    harness = command_runner
    if failure == "cleanup":
        harness.cleanup_failure = True
    elif failure == "pod-phase":
        harness.phase = "Failed"
    else:
        harness.fail_at = failure
    assert run_command(harness) == (0 if failure is None else 1)
    document = result(harness)
    assert document["verdict"] == ("PASS" if failure is None else "FAIL")
    names = [call[0] for call in harness.calls]
    assert "cleanup" in names, (
        "every started synthetic command case needs owned cleanup"
    )
    if failure in {
        "registry",
        "probe",
        "seed",
        "wait-file",
        "read-state",
        "wait-command",
        "cleanup",
        "pod-phase",
        None,
    }:
        assert harness.cleanup_state["registry_started"] is True
    if failure in {
        "probe",
        "seed",
        "wait-file",
        "read-state",
        "wait-command",
        "cleanup",
        "pod-phase",
        None,
    }:
        assert harness.cleanup_state["probe_started"] is True
    if failure in {
        "seed",
        "wait-file",
        "read-state",
        "wait-command",
        "cleanup",
        "pod-phase",
        None,
    }:
        assert harness.cleanup_state["seed"]["command_id"], (
            "seed identity must survive a lost write ACK"
        )
    if failure is None:
        assert document["ledger"] == {"physical_count": 1, "keys": ["workflow/0"]}
        if harness.module is not net003:
            assert (
                names.index("block") < names.index("unblock") < names.index("cleanup")
            )
        else:
            assert names.index("command-snapshot") < names.index("block")
            assert "unblock" not in names
            observed = next(
                index
                for index, (name, args) in enumerate(harness.calls)
                if name == "read-state"
                and args == ("/state/action-gate-observed.json",)
            )
            submitted = next(
                index
                for index, (name, args) in enumerate(harness.calls)
                if name == "read-state"
                and args == ("/state/result-submit-started.json",)
            )
            assert names.index("block") < observed < submitted
        if harness.module is not net006:
            assert document["cluster_id"] == "cluster-a"
            assert document["release_id"] == "release-a"


@pytest.mark.parametrize(
    "failure", ["registered-agent", "lease", "ready-or-notification"]
)
def test_runner_refuses_invalid_execution_premises_before_injection(
    command_runner: Any, failure: str
) -> None:
    harness = command_runner
    if failure == "registered-agent":
        harness.seed["registered_agents"] = ["not-synthetic"]
    elif failure == "lease":
        harness.leased["status"] = "PENDING"
    elif harness.module is net006:
        harness.ready["owner"] = "foreign-owner"
    elif harness.module is net003:
        harness.seed["notification_id"] = "foreign-notification"
    else:
        harness.leased["lease_expires_at"] = None
    assert run_command(harness) == 1
    assert not any(call[0] == "block" for call in harness.calls), (
        "bad premise must stop before network injection"
    )
    assert result(harness)["verdict"] == "FAIL"


def test_preflight_checks_deadline_and_timing_before_resources(
    command_runner: Any, monkeypatch: Any
) -> None:
    harness = command_runner
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(
        harness.transport.SeededCommandError
        if harness.module is net006
        else harness.module.CaseError,
        match="window has ended",
    ):
        harness.module.preflight_metadata(1, expired)
    timing_owner = (
        harness.module.verdicts if harness.module is net006 else harness.module
    )
    monkeypatch.setattr(timing_owner, "timing_errors", lambda: ["invalid timing"])
    with pytest.raises(RuntimeError, match="invalid timing"):
        harness.module.preflight_metadata(1, harness.deadline)
    assert not any(call[0] in {"registry", "probe"} for call in harness.calls), (
        "invalid timing cannot create resources"
    )


@pytest.mark.parametrize("command_runner", [net003], indirect=True)
@pytest.mark.parametrize(
    "failure", ["expired", "foreign-lease", "foreign-receipt", "stale-receipt"]
)
def test_net003_gate_refuses_stale_or_foreign_evidence(
    command_runner: Any, failure: str
) -> None:
    harness = command_runner
    if failure == "expired":
        harness.leased["lease_expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
    elif failure == "foreign-lease":
        harness.leased["idempotency_key"] = "foreign"
    elif failure == "foreign-receipt":
        harness.states["/state/action-gate-observed.json"]["idempotency_key"] = (
            "foreign"
        )
    else:
        harness.states["/state/action-gate-observed.json"]["observed_at_epoch"] = 1000.0
    assert run_command(harness) == 1
    assert result(harness)["verdict"] == "FAIL"
    assert harness.calls[-1][0] == "identity"
    assert any(name == "cleanup" for name, _args in harness.calls), (
        "a rejected action gate must still clean its owned resources"
    )
    assert not any(
        name == "read-state" and args == ("/state/result-submit-started.json",)
        for name, args in harness.calls
    ), "an invalid gate cannot advance to result-submission observation"
    if failure in {"expired", "foreign-lease"}:
        assert not any(name == "block" for name, _args in harness.calls), (
            "an expired or foreign lease cannot authorize the action gate"
        )


@pytest.mark.parametrize("module", [net002, net003])
def test_entrypoint_delegates_same_case_and_callable_contract(
    module: Any, monkeypatch: Any
) -> None:
    calls = []
    monkeypatch.setattr(
        module.fixture, "run_main", lambda **kwargs: calls.append(kwargs) or 7
    )
    assert module.main() == 7
    assert calls[0]["case_id"] == module.CASE_ID
    assert calls[0]["run_case"] is module.run_case
    assert (
        calls[0]["parser"]().parse_args(["--run-dir", "/tmp/fixture"]).execute is False
    )
    assert calls[0]["plan_details"]({"valid": True})["predecessor"] == {"valid": True}


def test_net006_entrypoint_retains_plain_identity_preflight(
    monkeypatch: Any, tmp_path: Path
) -> None:
    seen = []
    monkeypatch.setattr(net006, "run_plain_case", lambda case: seen.append(case) or 8)
    assert net006.main() == 8
    assert seen == [net006.CASE]
    monkeypatch.setattr(
        net006, "plain_case_preflight", lambda *a, **k: seen.append((a, k)) or {}
    )
    assert net006.CASE.read_only_preflight("settings", tmp_path) == {}
    assert seen[-1][1]["read_environment"] is net006.environment_snapshot


@pytest.mark.parametrize("module", [net002, net003])
def test_predecessor_and_final_identity_failure_cannot_be_passed_as_bound_evidence(
    monkeypatch: Any, tmp_path: Path, module: Any
) -> None:
    calls = []
    monkeypatch.setattr(module.fixture, "run_identity", lambda *a: "fixture-run")
    monkeypatch.setattr(
        module.fixture, "cleanup", lambda *a, **k: calls.append(k["state"])
    )

    def identity(_cluster: str) -> Any:
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(module.fixture, "evidence_identity", identity)
    assert (
        module.run_case(
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
            predecessor={"valid": False},
            cluster_id="cluster-a",
        )
        == 1
    )
    document = json.loads(
        (tmp_path / "cases" / module.CASE_ID / f"{module.CASE_ID}.json").read_text()
    )
    assert "predecessor" in document["error"]
    assert "identity unavailable" in document["identity_error"]
    assert calls == [{"seed": {}}]


@pytest.mark.parametrize(
    "remaining",
    [
        {"remaining": [{}], "remaining_links": 0},
        {"remaining": [], "remaining_links": None},
        {"remaining": [], "remaining_links": 1},
    ],
)
def test_net003_seed_cleanup_requires_exact_zero_residuals(
    monkeypatch: Any, remaining: Any
) -> None:
    calls = []
    monkeypatch.setattr(
        net003.fixture,
        "cpu_python",
        lambda script, *args: calls.append(args) or remaining,
    )
    with pytest.raises(net003.CaseError, match="residual"):
        net003.purge_seed(
            {
                "command_id": "command-a",
                "workflow_id": "workflow-a",
                "incident_id": "incident-a",
                "notification_id": "notification-a",
                "deduplication_key": "dedup-a",
                "event_id": "event-a",
            }
        )
    assert calls == [
        (
            "command-a",
            "workflow-a",
            "incident-a",
            "notification-a",
            "dedup-a",
            "event-a",
        )
    ]
