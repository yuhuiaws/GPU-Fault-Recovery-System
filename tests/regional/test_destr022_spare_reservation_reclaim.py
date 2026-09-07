"""Contract tests for GF-REGIONAL-DESTR-022.

Every verdict is judged against synthetic evidence documents -- spare node
snapshots, executor claim-state breadcrumbs, executor log lines, store
lookups -- once on the intended run and once per way the run can be wrong.
Nothing here touches a cluster; the runner's live phases only call these
functions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr022_verdicts as verdicts
from scripts.e2e.regional import run_destr022_spare_reservation_reclaim as destr022
from scripts.e2e.regional.probes import destr022_executor_probe as probe
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
SPARE = "spare-a"
INCIDENT = verdicts.synthetic_incident_id("abc123-a1")
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
RECLAIMED_AT = T0 + timedelta(minutes=4)
RESERVED_AT = verdicts.stale_reserved_at(T0)
TOPOLOGY_STRINGS = ("/secure/gpu-fault-bootstrap", "514385905925", "gpu-fault-gpu-1-")


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_destr008() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr022.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-node-mutation",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-022",
        predecessor=destr022.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "DESTR022_", prefix
    assert destr022.CONFIRMATION.startswith(prefix), destr022.CONFIRMATION
    assert destr022.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-008", (
        "the spare declared for DESTR-003/008 must still be declared and proven"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-DESTR-022", verdicts.CASE_ID


def test_the_stale_timestamp_is_well_past_the_one_day_ttl() -> None:
    reserved = verdicts.parse_time(RESERVED_AT)
    assert reserved is not None, RESERVED_AT
    assert (T0 - reserved).total_seconds() >= 2 * 86400, "two full days of margin"
    with pytest.raises(ValueError, match="two days"):
        verdicts.stale_reserved_at(T0, days=1)


def test_the_synthetic_incident_id_is_recognisable_as_acceptance_fabrication() -> None:
    assert INCIDENT.startswith("acceptance-stale-"), INCIDENT


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _spare(**overrides: Any) -> dict[str, Any]:
    spare: dict[str, Any] = {
        "name": SPARE,
        "uid": "uid-spare",
        "ready": "True",
        "unschedulable": True,
        "taints": [],
        "labels": {
            verdicts.SPARE_LABEL: "true",
            verdicts.HYPERPOD_HEALTH_LABEL: "Schedulable",
        },
        "annotations": {
            verdicts.SPARE_RESERVATION_ANNOTATION: None,
            verdicts.SPARE_RESERVED_AT_ANNOTATION: None,
            verdicts.SPARE_POOL_STATE_ANNOTATION: "AVAILABLE",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(spare.get(key), dict):
            spare[key] = {**spare[key], **value}
        else:
            spare[key] = value
    return spare


def _breadcrumb(counter: int | None = 3, claimed_at: datetime | None = None) -> dict:
    counters: dict[str, Any] = {"claimed_total": 10}
    if counter is not None:
        counters[verdicts.RECLAIM_COUNTER] = counter
    return {
        "executor_id": "cluster-a/executor",
        "execution_owners": ["gpu-fault-kubernetes-adapter"],
        "last_successful_claim_at": (claimed_at or T0).isoformat(),
        "counters": counters,
    }


def _probe(pod: str = "executor-0", **overrides: Any) -> dict[str, Any]:
    value = {
        "pod": pod,
        "uid": f"uid-{pod}",
        "claim_state": _breadcrumb(),
        "claim_state_path": "/tmp/executor-claim-state.json",
        "claim_state_error": None,
        "env": {"GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER": "true"},
    }
    value.update(overrides)
    return value


def _executor_env(**overrides: Any) -> dict[str, Any]:
    value = {
        "pod": "executor-0",
        "spare_failover": "true",
        "remote_state": "true",
        "allow_replace": "false",
        "spare_label": None,
    }
    value.update(overrides)
    return value


def _preflight(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "spare": _spare(),
        "spare_node": SPARE,
        "declared_spares": [SPARE],
        "workloads": [],
        "executor_env": [_executor_env()],
        "executor_probes": [_probe()],
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "incident_lookup": {"found": False, "workflows": []},
        "predecessor_valid": True,
        "tests_passed": True,
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def test_the_intended_preflight_is_clean() -> None:
    assert _preflight() == [], _text(_preflight())


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"spare": _spare(ready="False")}, "not Ready"),
        ({"spare": _spare(unschedulable=False)}, "not cordoned"),
        ({"spare": _spare(labels={verdicts.SPARE_LABEL: None})}, "spare label"),
        (
            {"spare": _spare(annotations={verdicts.SPARE_RESERVATION_ANNOTATION: "x"})},
            "already carries a reservation",
        ),
        (
            {
                "spare": _spare(
                    annotations={verdicts.SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"}
                )
            },
            "not AVAILABLE",
        ),
        (
            {"spare": _spare(taints=[{"key": verdicts.QUARANTINE_TAINT}])},
            "quarantine taint",
        ),
        (
            {"spare": _spare(annotations={"gpu-fault.io/incident-id": "inc"})},
            "ownership",
        ),
        ({"declared_spares": [SPARE, "spare-b"]}, "not exactly"),
        ({"declared_spares": []}, "not exactly"),
        ({"workloads": [{"pod": "train"}]}, "non-system workload"),
        ({"executor_env": []}, "no Ready cluster executor"),
        ({"executor_env": [_executor_env(spare_failover="false")]}, "deployment fact"),
        (
            {"executor_env": [_executor_env(allow_replace="true")]},
            "ALLOW_HYPERPOD_REPLACE",
        ),
        ({"executor_probes": []}, "no executor claim-state"),
        (
            {"executor_probes": [_probe(claim_state=_breadcrumb(counter=None))]},
            "predates ARCH-A4b",
        ),
        ({"executor_probes": [_probe(claim_state=None)]}, "predates ARCH-A4b"),
        ({"queue": {"depth": 2}}, "queue is not empty"),
        ({"remote_commands": {"open_by_cluster": {"c": 1}}}, "remote command queue"),
        ({"incident_lookup": {"found": True}}, "already exists"),
        ({"incident_lookup": {}}, "already exists"),
        ({"predecessor_valid": False}, "DESTR-008"),
        ({"tests_passed": False}, "focused regression"),
    ],
)
def test_the_preflight_refuses_each_unsafe_precondition(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _preflight(**overrides)
    assert any(fragment in item for item in errors), (fragment, errors)


# --------------------------------------------------------------------------- #
# Injection and reclaim
# --------------------------------------------------------------------------- #
def _reserved(**overrides: Any) -> dict[str, Any]:
    return _spare(
        annotations={
            verdicts.SPARE_RESERVATION_ANNOTATION: INCIDENT,
            verdicts.SPARE_RESERVED_AT_ANNOTATION: RESERVED_AT,
            verdicts.SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
        },
        **overrides,
    )


def test_the_injection_is_accepted_only_when_written_verbatim_and_cordoned() -> None:
    clean = verdicts.injection_errors(
        _reserved(), incident_id=INCIDENT, reserved_at=RESERVED_AT
    )
    assert clean == [], _text(clean)
    wrong = verdicts.injection_errors(
        _reserved(unschedulable=False), incident_id=INCIDENT, reserved_at=RESERVED_AT
    )
    assert any("uncordoned" in item for item in wrong), wrong
    missing = verdicts.injection_errors(
        _spare(), incident_id=INCIDENT, reserved_at=RESERVED_AT
    )
    assert len(missing) == 3, missing


def test_the_reclaim_verdict_passes_on_a_released_cordoned_spare() -> None:
    errors = verdicts.reclaim_errors(_spare(), incident_id=INCIDENT)
    assert errors == [], _text(errors)
    assert verdicts.reclaimed(_spare()) is True, "released spare must read as reclaimed"
    assert verdicts.reclaimed(_reserved()) is False, "reserved spare is not reclaimed"


@pytest.mark.parametrize(
    ("snapshot", "fragment"),
    [
        (_reserved(), "the synthetic reservation is still"),
        (
            _spare(annotations={verdicts.SPARE_RESERVATION_ANNOTATION: "other"}),
            "another (other) reservation",
        ),
        (
            _spare(annotations={verdicts.SPARE_RESERVED_AT_ANNOTATION: RESERVED_AT}),
            "reserved-at annotation was not cleared",
        ),
        (
            _spare(annotations={verdicts.SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"}),
            "not AVAILABLE",
        ),
        (_spare(unschedulable=False), "must stay cordoned"),
    ],
)
def test_the_reclaim_verdict_fails_each_residue(
    snapshot: dict[str, Any], fragment: str
) -> None:
    errors = verdicts.reclaim_errors(snapshot, incident_id=INCIDENT)
    assert any(fragment in item for item in errors), (fragment, errors)


def test_the_timeline_requires_a_cordoned_spare_at_every_sample() -> None:
    good = [
        {"snapshot": _reserved()},
        {"snapshot": _reserved()},
        {"snapshot": _spare()},
    ]
    assert verdicts.timeline_errors(good, incident_id=INCIDENT) == [], "intended"
    uncordoned = [{"snapshot": _reserved(unschedulable=False)}, {"snapshot": _spare()}]
    errors = verdicts.timeline_errors(uncordoned, incident_id=INCIDENT)
    assert any("sample 0 shows the spare schedulable" in item for item in errors), (
        errors
    )
    timed_out = [{"snapshot": _reserved()}]
    errors = verdicts.timeline_errors(timed_out, incident_id=INCIDENT)
    assert any("still on the spare" in item for item in errors), errors
    assert verdicts.timeline_errors([], incident_id=INCIDENT) == [
        "no reclaim timeline was recorded"
    ], "an empty timeline is a FAIL, not a PASS"


# --------------------------------------------------------------------------- #
# Executor log and counter
# --------------------------------------------------------------------------- #
def _reclaim_line(node: str = SPARE, incident: str = INCIDENT) -> str:
    return (
        "WARNING gpu_fault.cluster_executor reclaimed stale spare reservation: "
        f"node={node} incident={incident} reason=reservation by {incident} "
        "exceeded 86400s TTL"
    )


def test_the_log_verdict_needs_exactly_our_reclaim_line() -> None:
    assert (
        verdicts.log_errors([_reclaim_line()], node=SPARE, incident_id=INCIDENT) == []
    ), "the intended line"
    assert any(
        "carry no" in item
        for item in verdicts.log_errors([], node=SPARE, incident_id=INCIDENT)
    ), "missing line"
    assert any(
        "carry no" in item
        for item in verdicts.log_errors(
            [_reclaim_line(incident="other")], node=SPARE, incident_id=INCIDENT
        )
    ), "wrong incident is not ours"
    foreign = verdicts.log_errors(
        [_reclaim_line(), _reclaim_line(node="spare-b", incident="inc-real")],
        node=SPARE,
        incident_id=INCIDENT,
    )
    assert any("did not write" in item for item in foreign), foreign
    twice = verdicts.log_errors(
        [_reclaim_line(), _reclaim_line()], node=SPARE, incident_id=INCIDENT
    )
    assert any("2 times" in item for item in twice), twice
    failed = verdicts.log_errors(
        [_reclaim_line(), "WARNING spare reservation sweep failed"],
        node=SPARE,
        incident_id=INCIDENT,
    )
    assert any("sweep reported a failure" in item for item in failed), failed


def test_the_counter_verdict_requires_the_increment_only_on_a_rewritten_breadcrumb() -> (
    None
):
    before = [_probe(), _probe("executor-1")]
    later = RECLAIMED_AT + timedelta(seconds=3)
    moved = [
        _probe(claim_state=_breadcrumb(counter=4, claimed_at=later)),
        _probe("executor-1", claim_state=_breadcrumb(counter=3, claimed_at=later)),
    ]
    assert verdicts.counter_errors(before, moved, reclaimed_at=RECLAIMED_AT) == [], (
        "one replica reclaimed, the other did not"
    )
    unmoved = [
        _probe(claim_state=_breadcrumb(counter=3, claimed_at=later)),
        _probe("executor-1", claim_state=_breadcrumb(counter=3, claimed_at=later)),
    ]
    errors = verdicts.counter_errors(before, unmoved, reclaimed_at=RECLAIMED_AT)
    assert any("no executor replica moved" in item for item in errors), errors
    stale = [
        _probe(claim_state=_breadcrumb(counter=3, claimed_at=T0)),
        _probe("executor-1", claim_state=_breadcrumb(counter=3, claimed_at=T0)),
    ]
    assert verdicts.counter_errors(before, stale, reclaimed_at=RECLAIMED_AT) == [], (
        "a breadcrumb that predates the reclaim cannot be judged for the increment"
    )
    backwards = [
        _probe(claim_state=_breadcrumb(counter=1, claimed_at=later)),
        _probe("executor-1", claim_state=_breadcrumb(counter=3, claimed_at=later)),
    ]
    errors = verdicts.counter_errors(before, backwards, reclaimed_at=RECLAIMED_AT)
    assert any("went backwards" in item for item in errors), errors
    lost = [
        _probe(claim_state=_breadcrumb(counter=None, claimed_at=later)),
        _probe("executor-1", claim_state=_breadcrumb(counter=3, claimed_at=later)),
    ]
    errors = verdicts.counter_errors(before, lost, reclaimed_at=RECLAIMED_AT)
    assert any("lost counters" in item for item in errors), errors
    double = [
        _probe(claim_state=_breadcrumb(counter=4, claimed_at=later)),
        _probe("executor-1", claim_state=_breadcrumb(counter=4, claimed_at=later)),
    ]
    errors = verdicts.counter_errors(before, double, reclaimed_at=RECLAIMED_AT)
    assert any("moved by 2" in item for item in errors), errors
    replaced = [_probe(claim_state=_breadcrumb(counter=4, claimed_at=later))]
    errors = verdicts.counter_errors(before, replaced, reclaimed_at=RECLAIMED_AT)
    assert any("replica set changed" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Store, final node, other nodes
# --------------------------------------------------------------------------- #
def test_the_store_verdict_requires_the_incident_to_never_exist() -> None:
    absent = {"found": False, "workflows": []}
    assert verdicts.store_errors(absent, absent) == [], "intended"
    appeared = verdicts.store_errors(absent, {"found": True, "workflows": ["wf-1"]})
    assert len(appeared) == 2, appeared
    assert any(
        "existed before" in item for item in verdicts.store_errors({}, absent)
    ), "an unanswered lookup is not proof of absence"


def test_the_final_verdict_compares_tracked_annotations_and_cordon_to_baseline() -> (
    None
):
    baseline = _spare()
    assert verdicts.final_errors(baseline, _spare()) == [], "intended"
    left = verdicts.final_errors(baseline, _reserved())
    assert len(left) == 3, left
    errors = verdicts.final_errors(baseline, _spare(unschedulable=False))
    assert any("cordon state" in item for item in errors), errors
    errors = verdicts.final_errors(
        baseline, _spare(labels={verdicts.SPARE_LABEL: None})
    )
    assert any("spare label changed" in item for item in errors), errors
    errors = verdicts.final_errors(baseline, _spare(taints=[{"key": "x"}]))
    assert any("taints changed" in item for item in errors), errors


def test_other_gpu_nodes_must_not_move() -> None:
    before = [
        {"name": SPARE, "unschedulable": True, "taints": []},
        {"name": "node-b", "unschedulable": False, "taints": []},
    ]
    same_spare_moved = [
        {"name": SPARE, "unschedulable": False, "taints": []},
        {"name": "node-b", "unschedulable": False, "taints": []},
    ]
    assert (
        verdicts.other_nodes_errors(before, same_spare_moved, spare_node=SPARE) == []
    ), "the spare itself is judged elsewhere"
    other_moved = [
        {"name": SPARE, "unschedulable": True, "taints": []},
        {"name": "node-b", "unschedulable": True, "taints": []},
    ]
    assert verdicts.other_nodes_errors(before, other_moved, spare_node=SPARE) == [
        "another GPU node's scheduling state changed during the case"
    ], "node-b was cordoned"


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def test_the_executor_probe_reports_the_breadcrumb_and_sweep_env(
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim-state.json"
    path.write_text(json.dumps(_breadcrumb()), encoding="utf-8")
    environ = {
        probe.CLAIM_STATE_PATH_ENV: str(path),
        "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER": "true",
    }
    value = probe.report(environ, "executor-host")
    assert value["claim_state_path"] == str(path), value
    assert verdicts.counter_of(value["claim_state"]) == 3, value
    assert value["claim_state_error"] is None, value
    assert value["env"]["GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"] == "true", value
    assert value["env"]["GPU_FAULT_HYPERPOD_SPARE_LABEL"] is None, value
    assert value["hostname"] == "executor-host", value


def test_the_executor_probe_tolerates_a_missing_or_broken_breadcrumb(
    tmp_path: Path,
) -> None:
    missing = probe.report({probe.CLAIM_STATE_PATH_ENV: str(tmp_path / "nope")}, "h")
    assert missing["claim_state"] is None, missing
    assert "does not exist" in str(missing["claim_state_error"]), missing
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2]", encoding="utf-8")
    value = probe.report({probe.CLAIM_STATE_PATH_ENV: str(broken)}, "h")
    assert value["claim_state"] is None, value
    assert "not a JSON object" in str(value["claim_state_error"]), value
    default = probe.report({}, "h")
    assert default["claim_state_path"] == probe.DEFAULT_CLAIM_STATE_PATH, default
    assert verdicts.counter_of(None) is None, "no breadcrumb, no counter"
    assert verdicts.counter_of({"counters": {verdicts.RECLAIM_COUNTER: "x"}}) is None, (
        "a non-integer counter is unreadable, not zero"
    )


# --------------------------------------------------------------------------- #
# Runner guard
# --------------------------------------------------------------------------- #
def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / destr022.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    preflight = {
        "release_id": "rel-1",
        "spare": _spare(),
        "declared_spares": [SPARE],
        "executor_probes": [_probe()],
        "synthetic_incident_id": INCIDENT,
    }
    with pytest.raises(Exception, match="plan drifted"):
        destr022.verify_plan_identity(case_dir, preflight)


def test_the_plan_names_the_stop_conditions_that_make_the_case_safe() -> None:
    preflight = {
        "release_id": "rel-1",
        "spare": _spare(),
        "declared_spares": [SPARE],
        "executor_probes": [_probe()],
        "synthetic_incident_id": INCIDENT,
        "predecessor": {"valid": True},
    }
    settings = destr022.Settings(
        regional=None,  # type: ignore[arg-type]
        hyperpod_cluster="hp",
        spare_node=SPARE,
        predecessor_path=Path("/tmp/p.json"),
        reclaim_timeout_seconds=480,
    )
    details = destr022.plan_details(settings, preflight)
    assert details["risk"] == "live-node-mutation", details["risk"]
    conditions = "\n".join(details["stop_conditions"])
    assert "SPARE_FAILOVER" in conditions, conditions
    assert "predates ARCH-A4b" in conditions, conditions
    assert "schedulable" in conditions, conditions
    assert details["rollback"]["spare_is_never_uncordoned"] is True, details["rollback"]
    assert "no uncordon" in details["mutation"], details["mutation"]


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr022.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    assert plan.reclaim_timeout_seconds == verdicts.RECLAIM_TIMEOUT_SECONDS, (
        plan.reclaim_timeout_seconds
    )
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr022.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--spare-node",
            SPARE,
        ]
    )
    assert execute.execute is True and execute.confirm == destr022.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def _configure_arguments(tmp_path: Path, timeout: str) -> argparse.Namespace:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return destr022.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "ctx",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--hyperpod-cluster",
            "hp",
            "--spare-node",
            SPARE,
            "--reclaim-timeout-seconds",
            timeout,
        ]
    )


def test_configure_bounds_the_reclaim_wait(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="reclaim timeout"):
        destr022.configure(_configure_arguments(tmp_path, "30"))
    with pytest.raises(Exception, match="reclaim timeout"):
        destr022.configure(_configure_arguments(tmp_path, "3600"))
    settings = destr022.configure(_configure_arguments(tmp_path, "480"))
    assert settings.spare_node == SPARE, settings
    assert settings.environment()["GPU_FAULT_SPARE_NODE"] == SPARE, (
        settings.environment()
    )
    assert settings.predecessor_path.name == f"{destr022.PREDECESSOR_CASE_ID}.json", (
        settings.predecessor_path
    )


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr022.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_the_runner_and_probe_are_executable_with_a_shebang_and_no_topology() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_destr022_spare_reservation_reclaim.py",
        ROOT / "scripts/e2e/regional/probes/destr022_executor_probe.py",
        ROOT / "scripts/e2e/regional/destr022_verdicts.py",
    ):
        source = path.read_text(encoding="utf-8")
        for value in TOPOLOGY_STRINGS:
            assert value not in source, (path.name, value)
        if path.name.startswith("destr022_verdicts"):
            continue
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        assert source.splitlines()[0] == "#!/usr/bin/env python3", path.name
