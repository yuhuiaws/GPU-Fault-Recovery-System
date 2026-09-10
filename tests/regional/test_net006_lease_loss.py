"""Contract tests for GF-REGIONAL-NET-006 and the seeded-command fixture.

The verdicts are judged against the evidence documents the runner writes --
the probe's ready/state files, store snapshots of the seeded command, the Pod
log -- once on the intended run and once per way the run can be wrong. The
fixture tests pin what makes the seeded command harmless: a synthetic cluster,
synthetic node ids, and a purge that reports what it removed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import net006_verdicts as verdicts
from scripts.e2e.regional import run_net006_lease_loss_withheld_result as net006
from scripts.e2e.regional import seeded_command_fixture as seeded

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
UNBLOCKED_AT = T0 + timedelta(seconds=120)
EXECUTOR_ID = "net006-test-executor"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Timing arithmetic
# --------------------------------------------------------------------------- #
def test_the_shipped_timing_declares_the_loss_by_failures_inside_the_hold() -> None:
    assert verdicts.timing_errors() == [], "the shipped constants must be coherent"
    interval = verdicts.renewal_interval_seconds(verdicts.LEASE_SECONDS)
    assert interval == 20.0, interval
    assert interval * verdicts.LEASE_FAILURE_LIMIT < verdicts.ACTION_HOLD_SECONDS, (
        "three renewal failures must land inside the action hold"
    )
    assert (
        verdicts.LEASE_SECONDS < verdicts.ACTION_HOLD_SECONDS < verdicts.BLOCK_SECONDS
    )


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"hold_seconds": 50}, "does not outlast the lease"),
        ({"hold_seconds": 61, "failure_limit": 4}, "local clock alone"),
        ({"block_seconds": 90}, "must outlast the hold"),
        ({"rollback_seconds": 100}, "automatic rollback"),
        ({"http_timeout_seconds": 60}, "HTTP timeout"),
    ],
)
def test_a_timing_that_could_pass_for_the_wrong_reason_is_refused(
    overrides: dict[str, Any], fragment: str
) -> None:
    assert fragment in _text(verdicts.timing_errors(**overrides)), overrides


def test_the_probe_ready_document_is_checked_against_the_same_arithmetic() -> None:
    ready = {
        "proxy_mode": "refuse-while-blocked",
        "action_requires_network_block": True,
        "owner": verdicts.OWNER,
        "block_rollback_seconds": verdicts.BLOCK_ROLLBACK_SECONDS,
        "action_hold_seconds": verdicts.ACTION_HOLD_SECONDS,
        "lease_seconds": verdicts.LEASE_SECONDS,
        "lease_failure_limit": verdicts.LEASE_FAILURE_LIMIT,
        "http_timeout_seconds": verdicts.HTTP_TIMEOUT_SECONDS,
    }
    assert verdicts.ready_errors(ready) == [], "the probe's own contract passes"
    holding = {**ready, "proxy_mode": "hold-while-blocked"}
    assert "does not refuse" in _text(verdicts.ready_errors(holding))
    short = {**ready, "action_hold_seconds": 10}
    assert "does not outlast" in _text(verdicts.ready_errors(short))


# --------------------------------------------------------------------------- #
# Executor counters (C7/C8)
# --------------------------------------------------------------------------- #
def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "claimed_total": 2,
        "reported_failures": 0,
        "unexpected_failures": 0,
        "lease_renewal_failures": 4,
        "lease_lost_total": 1,
        "results_withheld_total": 1,
        "cancellations_observed_total": 0,
        "barrier_unavailable_holds_total": 0,
        "last_successful_claim_at": T0.isoformat(),
        "executor_id": EXECUTOR_ID,
    }
    state.update(overrides)
    return state


def test_the_counter_contract_passes_on_one_withheld_result_and_one_reclaim() -> None:
    assert verdicts.executor_state_errors(_state()) == [], "the intended counters pass"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"claimed_total": 1}, "claimed_total is 1"),
        ({"results_withheld_total": 0}, "results_withheld_total is 0"),
        ({"lease_lost_total": 0}, "never moved"),
        ({"lease_renewal_failures": 1}, "below the failure limit"),
        ({"reported_failures": 1}, "posted under the lost lease"),
        ({"unexpected_failures": 1}, "unexpected failure"),
        ({"cancellations_observed_total": 1}, "nothing cancelled"),
        ({"last_successful_claim_at": None}, "last_successful_claim_at"),
    ],
)
def test_each_counter_deviation_fails_the_case(
    overrides: dict[str, Any], fragment: str
) -> None:
    assert fragment in _text(verdicts.executor_state_errors(_state(**overrides)))


def test_the_breadcrumb_must_carry_the_same_counters() -> None:
    state = _state()
    crumb = {"executor_id": EXECUTOR_ID, "counters": verdicts.counters(state)}
    assert verdicts.breadcrumb_errors(crumb, state) == [], "matching counters pass"
    stale = {
        "executor_id": EXECUTOR_ID,
        "counters": {**verdicts.counters(state), "results_withheld_total": 0},
    }
    assert "differ from the executor" in _text(verdicts.breadcrumb_errors(stale, state))
    assert "no counters" in _text(
        verdicts.breadcrumb_errors({"executor_id": EXECUTOR_ID}, state)
    )


def test_the_lease_guard_must_have_seen_the_loss_on_the_action_thread() -> None:
    assert (
        verdicts.lease_guard_errors(
            {"reason": "lease renewal failed 3 time(s) in a row: URLError: x"}
        )
        == []
    )
    assert (
        verdicts.lease_guard_errors(
            {"reason": "lease expired locally after 60s without a successful renewal"}
        )
        == []
    )
    assert "no hold reason" in _text(verdicts.lease_guard_errors({"reason": None}))
    assert "not a lost-lease reason" in _text(
        verdicts.lease_guard_errors(
            {"reason": "cancellation requested by the control plane"}
        )
    )


def test_the_log_must_show_the_withhold_and_no_stale_lease_409() -> None:
    good = f"... {verdicts.LOST_LOG}: command=x ...\n... {verdicts.WITHHELD_LOG}: command=x ..."
    assert verdicts.log_errors(good) == [], "both C7 lines present, no 409"
    posted = good + f"\n... {verdicts.STALE_LEASE_409} ..."
    assert "posted under the lost lease" in _text(verdicts.log_errors(posted))
    assert "no 'withheld" in _text(verdicts.log_errors(f"... {verdicts.LOST_LOG} ..."))


# --------------------------------------------------------------------------- #
# Command journey, ledger, interruption
# --------------------------------------------------------------------------- #
def _command(status: str, **overrides: Any) -> dict[str, Any]:
    command = {
        "status": status,
        "status_source": None,
        "lease_expires_at": (T0 + timedelta(seconds=60)).isoformat(),
        "last_lease_owner": EXECUTOR_ID,
        "result_details": {},
    }
    command.update(overrides)
    return command


def _command_errors(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "leased": _command("LEASED"),
        "withheld": _command("LEASED"),
        "final": _command(
            "SUCCEEDED", result_details={"cached": True, "simulated": True}
        ),
        "unblocked_at": UNBLOCKED_AT,
        "executor_id": EXECUTOR_ID,
    }
    arguments.update(overrides)
    return verdicts.command_errors(**arguments)


def test_the_command_journey_passes_when_the_reclaim_finishes_from_the_ledger() -> None:
    assert _command_errors() == [], "leased -> still leased -> cached SUCCEEDED passes"


def test_the_command_journey_rejects_an_early_finish_or_a_fresh_execution() -> None:
    early = _command_errors(withheld=_command("SUCCEEDED", status_source="x"))
    assert "left LEASED/PENDING" in _text(early), early
    fresh = _command_errors(
        final=_command("SUCCEEDED", result_details={"cached": False})
    )
    assert "idempotency ledger" in _text(fresh), fresh
    other = _command_errors(
        final=_command(
            "SUCCEEDED", result_details={"cached": True}, last_lease_owner="someone"
        )
    )
    assert "final lease owner" in _text(other), other
    long_lease = _command_errors(
        leased=_command(
            "LEASED", lease_expires_at=(UNBLOCKED_AT + timedelta(seconds=1)).isoformat()
        )
    )
    assert "had not expired" in _text(long_lease), long_lease


def test_the_ledger_and_interruption_contracts() -> None:
    assert verdicts.ledger_errors({"physical_count": 1, "keys": ["k"]}) == [], (
        "one action"
    )
    assert "not one" in _text(
        verdicts.ledger_errors({"physical_count": 2, "keys": ["k", "j"]})
    )
    gate = {"observed_at_epoch": 1000.0}
    returned = {"observed_at_epoch": 1100.0, "cached": False}
    assert (
        verdicts.interruption_errors(
            blocked_seconds=121.0,
            action_gate=gate,
            action_returned=returned,
            rollback=None,
        )
        == []
    ), "a 100s hold inside a 121s block with no automatic rollback passes"
    short = verdicts.interruption_errors(
        blocked_seconds=121.0,
        action_gate=gate,
        action_returned={"observed_at_epoch": 1030.0, "cached": False},
        rollback=None,
    )
    assert "not past the 60s lease" in _text(short), short
    fired = verdicts.interruption_errors(
        blocked_seconds=121.0,
        action_gate=gate,
        action_returned=returned,
        rollback={"automatic": True},
    )
    assert "automatic rollback fired" in _text(fired), fired
    brief = verdicts.interruption_errors(
        blocked_seconds=30.0, action_gate=gate, action_returned=returned, rollback=None
    )
    assert "ended before 120" in _text(brief), brief


# --------------------------------------------------------------------------- #
# Seeded-command fixture
# --------------------------------------------------------------------------- #
def test_the_probe_definition_is_the_shipped_script_with_bounded_deadline() -> None:
    probe = net006.probe_definition()
    assert probe.script.is_file(), probe.script
    assert probe.pod == verdicts.POD and probe.owner == verdicts.OWNER, probe
    assert probe.environment["ACTION_HOLD_SECONDS"] == str(verdicts.ACTION_HOLD_SECONDS)
    with pytest.raises(ValueError, match="deadline"):
        seeded.SeededCommandProbe(
            case_id="x",
            run_prefix="x-",
            pod="p",
            configmap="c",
            owner="o",
            script=probe.script,
            pod_deadline_seconds=10,
        )
    with pytest.raises(ValueError, match="plain identifier"):
        seeded.SeededCommandProbe(
            case_id="x",
            run_prefix="x%",
            pod="p",
            configmap="c",
            owner="o",
            script=probe.script,
        )


def test_the_pod_manifest_pins_the_executor_identity_and_the_case_env() -> None:
    probe = net006.probe_definition()
    manifest = seeded.pod_manifest(
        probe,
        "registry.example/executor@sha256:" + "0" * 64,
        {
            "executor_artifact_sha256": "a" * 64,
            "executor_compatibility_digest": "b" * 64,
        },
    )
    container = manifest["spec"]["containers"][0]
    env = {item["name"]: item.get("value") for item in container["env"]}
    assert env["EXECUTOR_ARTIFACT_SHA256"] == "a" * 64, env
    assert env["EXECUTOR_OWNER"] == verdicts.OWNER, env
    assert env["LEASE_SECONDS"] == str(verdicts.LEASE_SECONDS), env
    assert container["command"][-1] == "/scripts/net006_executor.py", container[
        "command"
    ]
    assert manifest["spec"]["activeDeadlineSeconds"] == verdicts.POD_DEADLINE_SECONDS
    assert manifest["spec"]["restartPolicy"] == "Never", manifest["spec"][
        "restartPolicy"
    ]
    assert (
        manifest["metadata"]["labels"]["gpu-fault.io/acceptance-case"]
        == verdicts.CASE_ID
    )


def test_the_seed_refuses_node_ids_that_cannot_round_trip() -> None:
    with pytest.raises(seeded.SeededCommandError, match="comma-free"):
        seeded.seed_command(
            "run", owner="o", operation="FREEZE_EVIDENCE", node_ids=["a,b"]
        )
    with pytest.raises(seeded.SeededCommandError, match="comma-free"):
        seeded.seed_command("run", owner="o", operation="FREEZE_EVIDENCE", node_ids=[])


def test_the_seed_leases_the_workflow_to_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unleased seeded workflow is claimed by the deployed dispatcher, which
    fails it and has the seeded command cancelled under the probe (attempt 1,
    2026-09-10); the seed script receives the lease length as its last argument."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        seeded, "cpu_python", lambda script, *args: calls.append(tuple(args)) or {}
    )
    seeded.seed_command("run", owner="o", operation="FREEZE_EVIDENCE", node_ids=["n-a"])
    assert calls[-1][-1] == str(seeded.SEED_LEASE_SECONDS)
    assert seeded.SEED_LEASE_SECONDS >= 30 * 60, (
        "must outlive the synthetic registry entry"
    )
    with pytest.raises(seeded.SeededCommandError, match="at least 60"):
        seeded.seed_command(
            "run",
            owner="o",
            operation="FREEZE_EVIDENCE",
            node_ids=["n-a"],
            lease_seconds=5,
        )


def test_a_finished_document_is_residual_free_only_when_every_postflight_is_zero() -> (
    None
):
    clean = {
        "database_postflight": {"total": 0},
        "registry_postflight": {"count": 0},
        "kubernetes_postflight": {"count": 0},
    }
    assert seeded.residual_free(clean) is True, clean
    assert seeded.residual_free({**clean, "registry_postflight": {"count": 1}}) is False
    assert seeded.residual_free({**clean, "cleanup_error": "x"}) is False
    assert seeded.run_identity(
        Path("/runs/acceptance-20260906T1000-ab12"), 2, "net006"
    ) == ("net006-ab12-a2")


def test_the_runner_is_plan_by_default_with_the_documented_flags() -> None:
    parser = net006.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "plan by default"
    help_text = parser.format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag
    details = net006.plan_details()
    assert details["risk"] == "live-non-destructive", details
    assert "no Node Agents" in details["hard_stop"], details["hard_stop"]
    assert details["synthetic_cluster_id"] == seeded.SYNTHETIC_CLUSTER_ID, details
    for path in (
        ROOT / "scripts/e2e/regional/run_net006_lease_loss_withheld_result.py",
        ROOT / "scripts/e2e/regional/probes/net006_executor.py",
    ):
        assert path.stat().st_mode & 0o777 == 0o775, path
        assert (
            path.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env python3"
        )
