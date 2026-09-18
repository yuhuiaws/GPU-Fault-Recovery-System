"""NET command fixture read protocol, bounded polling and cleanup delegation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import net_command_fixture as fixture
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401


def test_cached_cpu_pod_refresh_and_read_failure_never_replay_script(
    monkeypatch: Any,
) -> None:
    fixture.reset_cpu_pod_cache()
    calls = []
    output = {"value": "notice\n{}"}

    def control(*args: str, **kwargs: Any) -> str:
        calls.append(args)
        return "pod-a\n" if args[0] == "get" else output["value"]

    monkeypatch.setattr(fixture, "control", control)
    assert fixture.cpu_pod() == "pod-a"
    assert fixture.cpu_pod() == "pod-a"
    assert len(calls) == 1
    assert fixture.cpu_pod(refresh=True) == "pod-a"
    assert len(calls) == 2
    assert fixture.cpu_python("opaque query", "arg-a") == {}
    assert calls[-1][-1] == "arg-a"
    output["value"] = ""
    with pytest.raises(fixture.NetCommandError, match="no output"):
        fixture.cpu_python("opaque write", "arg-b")
    assert len([call for call in calls if call[0] == "exec"]) == 2, (
        "a write with a missing ACK must never be implicitly replayed"
    )
    output["value"] = "[]"
    with pytest.raises(fixture.NetCommandError, match="JSON object"):
        fixture.cpu_python("opaque query")
    monkeypatch.setattr(fixture, "control", lambda *a, **k: "")
    with pytest.raises(fixture.NetCommandError, match="no Running"):
        fixture.cpu_pod(refresh=True)
    fixture.reset_cpu_pod_cache()


@pytest.mark.parametrize("failure", [None, "database", "registry", "kubernetes"])
def test_preflight_stops_on_first_nonempty_owned_resource_population(
    monkeypatch: Any, tmp_path: Path, failure: str | None
) -> None:
    calls = []
    probe = SimpleNamespace(run_prefix="net-fixture")
    monkeypatch.setattr(
        fixture,
        "cpu_python",
        lambda script, prefix: calls.append("database")
        or {"total": int(failure == "database")},
    )
    monkeypatch.setattr(
        fixture,
        "registry_residuals",
        lambda: calls.append("registry") or {"count": int(failure == "registry")},
    )
    monkeypatch.setattr(
        fixture,
        "kubernetes_residuals",
        lambda current: calls.append("kubernetes")
        or {"count": int(failure == "kubernetes")},
    )
    if failure:
        with pytest.raises(fixture.NetCommandError, match="residuals"):
            fixture.preflight_residuals(probe, tmp_path)
        assert calls[-1] == failure
    else:
        assert fixture.preflight_residuals(probe, tmp_path) == {
            "database": {"total": 0},
            "registry": {"count": 0},
            "kubernetes": {"count": 0},
        }
        assert calls == ["database", "registry", "kubernetes"]


@pytest.mark.parametrize("found", [False, True])
def test_wait_command_reads_current_status_until_success_or_deadline(
    monkeypatch: Any, found: bool
) -> None:
    clock = Clock()
    calls = []
    monkeypatch.setattr(
        fixture,
        "cpu_python",
        lambda script, command: calls.append(command)
        or {"status": "SUCCEEDED" if found and clock.now >= 1002 else "LEASED"},
    )
    if found:
        assert fixture.wait_command(
            "command-a", "SUCCEEDED", 5, clock=clock.time, sleep=clock.sleep
        ) == {"status": "SUCCEEDED"}
    else:
        with pytest.raises(fixture.NetCommandError, match="did not reach"):
            fixture.wait_command(
                "command-a", "SUCCEEDED", 5, clock=clock.time, sleep=clock.sleep
            )
    assert calls[0] == "command-a"
    assert len(calls) == (2 if found else 3)


def test_seed_cleanup_and_identity_calls_forward_exact_scope(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        fixture.seeded,
        "seed_command",
        lambda run_id, **kw: calls.append(("seed", run_id, kw))
        or {"command_id": "command-a"},
    )
    monkeypatch.setattr(
        fixture.seeded, "purge_seed", lambda seed: calls.append(("purge", seed)) or {}
    )
    seed = fixture.seed_command(
        "run-a",
        owner="owner-a",
        operation="FREEZE_EVIDENCE",
        node_ids=["synthetic-node"],
    )
    assert seed == {"command_id": "command-a"}
    assert fixture.purge_seed(seed) == {}
    monkeypatch.setattr(
        fixture.seeded, "cleanup", lambda *a, **kw: calls.append(("cleanup", a, kw))
    )
    state = {"seed": seed}
    fixture.cleanup("probe", tmp_path, "run-a", {}, seed, state=state)
    assert calls[-1][2]["state"] is state
    assert calls[-1][2]["database_probe"] is fixture.database_residuals
    monkeypatch.setattr(
        fixture,
        "control",
        lambda *a: json.dumps(
            {"data": {"state.json": json.dumps({"release_id": "release-a"})}}
        ),
    )
    assert fixture.evidence_identity("cluster-a") == {
        "release_id": "release-a",
        "cluster_id": "cluster-a",
    }


def test_predecessor_gate_and_run_identity_do_not_guess_missing_targets(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(fixture, "predecessor_path", lambda *a: (None, None))
    assert fixture.predecessor_gate(tmp_path, "case", None) == {
        "valid": True,
        "case_id": None,
        "verdict": "NOT_REQUIRED",
    }
    path = tmp_path / "previous.json"
    seen = []
    monkeypatch.setattr(fixture, "predecessor_path", lambda *a: ("previous", path))
    monkeypatch.setattr(
        fixture, "predecessor_evidence", lambda *a: seen.append(a) or {"valid": True}
    )
    assert fixture.predecessor_gate(tmp_path, "case", path) == {"valid": True}
    assert seen == [(path, "previous")]
    with pytest.raises(ValueError, match="prefix"):
        fixture.run_identity(tmp_path, 1, "../unsafe")
    assert fixture.run_identity(tmp_path, 1, "net") != fixture.run_identity(
        tmp_path / "other", 1, "net"
    )


@pytest.mark.parametrize("execute,valid", [(False, False), (False, True), (True, True)])
def test_guarded_fixture_entry_passes_actual_identity_and_deadline(
    monkeypatch: Any, tmp_path: Path, execute: bool, valid: bool
) -> None:
    calls = []
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    arguments = argparse.Namespace(
        run_dir=tmp_path,
        predecessor_evidence="",
        execute=execute,
        attempt=2,
        cluster_id="cluster-a",
    )
    monkeypatch.setattr(
        fixture, "install_site_profile", lambda: calls.append(("site",))
    )
    monkeypatch.setattr(
        fixture, "os", SimpleNamespace(**{**vars(os), "umask": lambda mode: None})
    )
    monkeypatch.setattr(fixture, "predecessor_gate", lambda *a: {"valid": valid})
    monkeypatch.setattr(
        fixture, "build_plan", lambda **kw: calls.append(("plan", kw)) or {}
    )
    monkeypatch.setattr(
        fixture,
        "authorize_execution",
        lambda *a, **kw: calls.append(("authorize", kw)) or deadline,
    )
    result = fixture.run_main(
        case_id="GF-REGIONAL-NET-002",
        confirmation="CONFIRM",
        parser=lambda: SimpleNamespace(parse_args=lambda: arguments),
        plan_details=lambda predecessor: {"predecessor": predecessor},
        run_case=lambda *a, **kw: calls.append(("case", a, kw)) or 7,
    )
    assert result == (7 if execute else 0 if valid else 1)
    if execute:
        assert calls[-1] == (
            "case",
            (tmp_path, 2, deadline),
            {"predecessor": {"valid": True}, "cluster_id": "cluster-a"},
        )
    else:
        assert calls[-1][1]["preflight_passed"] is valid
        assert not any(call[0] == "case" for call in calls), (
            "plan cannot execute the runner"
        )
