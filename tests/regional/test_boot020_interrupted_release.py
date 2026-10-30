"""BOOT-020 against a release transaction an earlier run left open.

R13j attempt 23 was killed mid ``full`` stage while the engine rolled drill
candidate D forward (``data-plane-progress``, uncommitted). Attempt 24 started
fresh and died at the public AdminConfig apply, and ``--resume`` would have
died too: its convergence finished only pending rollbacks and then opened a new
transaction over the in-flight forward one.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release.regional_release_config import ReleaseError
from scripts.e2e.regional import boot020_interrupted_release as interrupted
from scripts.e2e.regional import run_boot020_release_rolling as rolling
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from tests.regional._cov95_boot_extra_rolling import BackendModel, RollingEntry
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional.test_boot020_runner import FakeReleaseRollingBackend
from tests.regional.test_boot_lifecycle_entrypoints import entrypoint, execution_flags

CANDIDATE_D = "release-full"
INTERRUPTED_D = {
    "phase": "data-plane-progress",
    "release_id": CANDIDATE_D,
    "transaction_committed": False,
    "release_lifecycle": "ROLLING_CLUSTERS",
    "admin_config_sha256": "a" * 64,
}


def _interrupted_in_full(tmp_path: Path) -> tuple[Path, FakeReleaseRollingBackend]:
    """Attempt 23: noop..agent passed, killed inside the FULL injected deploy.

    The evidence keeps ``full_classification``/``full_before`` and is still
    RUNNING (a kill records no failure); the backend is at the agent stage's
    end with candidate D's forward roll open.
    """

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(evidence, case_id=rolling.CASE_ID, inputs={})
    rolling.run_release_rolling(FakeReleaseRollingBackend(), recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in list(document["stages"]):
        if name.startswith(("full", "restore")) and name not in {
            "full_classification",
            "full_before",
        }:
            document["stages"].pop(name)
    document["status"] = "RUNNING"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor", "agent"})
    second.live["cpu_wheel"] = "cpu-v2"
    second.live["clusters"]["cluster-a"].update(
        wheel="executor-v2", reconciler_wheel="node-v2", bundle="bundle-v2"
    )
    second.open_transaction = dict(INTERRUPTED_D)
    return evidence, second


def _resume(evidence: Path, backend: FakeReleaseRollingBackend) -> dict[str, Any]:
    recorder = EvidenceRecorder(evidence, case_id=rolling.CASE_ID, inputs={})
    return rolling.resume_release_rolling(backend, recorder)


def test_resume_first_finishes_the_interrupted_forward_roll_of_candidate_d(
    tmp_path: Path,
) -> None:
    evidence, backend = _interrupted_in_full(tmp_path)

    result = _resume(evidence, backend)

    assert result["status"] == "COMPLETED", result.get("error")
    assert backend.calls == [
        # Finish D the way gpu-fault-admin deploy would, then the old steps:
        # no rollback pending, the agent precondition deployed (a NOOP here).
        ("resume_upgrade", "full"),
        ("resume_rollback", "agent"),
        ("agent", None, False, None, "NOOP"),
        # The full stage reruns from its classification, then the restore.
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
        ("noop", None, False, None, "FULL"),
    ], backend.calls
    finished, converged = result["convergence"]
    assert finished["finished_interrupted_release"] == "full", finished
    assert (finished["phase_before"], finished["phase"]) == (
        "data-plane-progress",
        "complete",
    ), finished
    assert converged["converged_to"] == "agent", converged
    assert converged["forward_resume"]["resumed"] is True, converged
    assert result["stages"]["resumed_at_full"]["skipped_stages"] == [
        "noop",
        "control_plane",
        "executor",
        "agent",
    ]


@pytest.mark.parametrize(
    ("release_id", "admin", "expected"),
    [
        ("release-noop", "a" * 64, "noop"),
        ("release-noop", "c" * 64, "control_plane"),
        (CANDIDATE_D, "a" * 64, "full"),
        ("release-executor", None, "executor"),
        # The site's own release with another AdminConfig: an operator's
        # gpu-fault-admin config apply, not the drill's transaction.
        ("release-noop", "b" * 64, None),
        ("release-from-a-new-deploy", "a" * 64, None),
        ("", "a" * 64, None),
    ],
)
def test_the_open_transaction_is_matched_by_release_and_admin_config(
    release_id: str, admin: str | None, expected: str | None
) -> None:
    candidates = FakeReleaseRollingBackend().candidate_releases()
    state = {"phase": "cpu-staged", "release_id": release_id}
    if admin is not None:
        state["admin_config_sha256"] = admin
    if expected is None:
        with pytest.raises(RuntimeError, match="not one of this drill's releases"):
            interrupted.forward_roll_scenario(state, candidates)
    else:
        assert interrupted.forward_roll_scenario(state, candidates) == expected


def test_resume_never_resumes_a_release_the_drill_did_not_derive(
    tmp_path: Path,
) -> None:
    evidence, backend = _interrupted_in_full(tmp_path)
    backend.open_transaction = {**INTERRUPTED_D, "release_id": "foreign-release"}

    with pytest.raises(
        RuntimeError, match="will not resume .*foreign-release"
    ) as error:
        _resume(evidence, backend)

    assert "gpu-fault-admin deploy resumes its own release" in str(error.value)
    assert backend.calls == [], "the site was touched for a foreign transaction"
    saved = json.loads(evidence.read_text(encoding="utf-8"))
    assert "convergence" not in saved and saved["verdict"] == "FAIL", saved


@pytest.mark.parametrize("ends_in", ["partial-convergence", "data-plane-progress"])
def test_a_resume_that_leaves_the_roll_open_stops_the_convergence(
    tmp_path: Path, ends_in: str
) -> None:
    evidence, backend = _interrupted_in_full(tmp_path)
    backend.resume_ends_in = ends_in

    with pytest.raises(RuntimeError, match="could not finish the interrupted full"):
        _resume(evidence, backend)

    assert backend.calls == [("resume_upgrade", "full")], backend.calls
    (recorded,) = json.loads(evidence.read_text(encoding="utf-8"))["convergence"]
    assert recorded["finished_interrupted_release"] == "full", recorded
    assert (recorded["phase"], recorded["error"]) == (ends_in, "unit failure")


def test_a_resume_the_engine_compensated_continues_the_convergence(
    tmp_path: Path,
) -> None:
    """Automatic rollback on: a failed resume restores the previous release and
    the convergence goes on from there, with the failure on record."""

    evidence, backend = _interrupted_in_full(tmp_path)
    backend.resume_ends_in = "rolled-back"

    result = _resume(evidence, backend)

    assert result["status"] == "COMPLETED", result.get("error")
    assert backend.calls[:3] == [
        ("resume_upgrade", "full"),
        ("resume_rollback", "agent"),
        ("agent", None, False, None, "NOOP"),
    ], backend.calls
    assert result["convergence"][0]["error"] == "unit failure"


def test_a_complete_but_uncommitted_candidate_is_not_resumed(tmp_path: Path) -> None:
    """Every drill stage leaves its candidate ``complete`` and uncommitted, and
    a new transaction over that is what the chain does at every stage."""

    evidence, backend = _interrupted_in_full(tmp_path)
    backend.open_transaction = {
        **INTERRUPTED_D,
        "phase": "complete",
        "release_id": "release-agent",
    }

    result = _resume(evidence, backend)

    assert result["status"] == "COMPLETED", result.get("error")
    assert ("resume_upgrade", "agent") not in backend.calls, backend.calls
    assert backend.calls[0] == ("resume_rollback", "agent"), backend.calls
    assert result["convergence"][-1]["forward_resume"] == {
        "resumed": False,
        "phase": "complete",
        "release_id": "release-agent",
    }


def test_live_backend_resumes_with_run_resume_and_automatic_rollback_on() -> None:
    model = BackendModel()
    model.config = replace(model.config, auto_rollback=False)
    backend = model.backend()
    resumed = []

    def run_resume(release: Any) -> None:
        resumed.append(release)
        release.state.update(phase="complete", transaction_committed=False)

    backend.commands = SimpleNamespace(run_resume=run_resume)

    result = backend.resume_upgrade("full")

    (engine,) = model.engines
    assert resumed == [engine], "run_resume did not get the full config's release"
    assert model.loaded == [backend.configs["full"]]
    assert engine.config.auto_rollback is True, "the resume must compensate"
    assert model.config.auto_rollback is False, "override mutated the loaded config"
    assert (result["phase_before"], result["phase"]) == ("complete", "complete")
    assert result["error"] is None and result["release_id"] == "unit-release"

    def refuse(release: Any) -> None:
        raise ReleaseError("resume release diff does not match the candidate")

    backend.commands = SimpleNamespace(run_resume=refuse)
    failed = backend.resume_upgrade("full")
    assert failed["error"] == (
        "ReleaseError: resume release diff does not match the candidate"
    ), failed
    assert failed["phase"] == "complete", "the state the refusal left is recorded"


def test_live_backend_reads_the_state_and_names_each_config_release() -> None:
    model = BackendModel()
    backend = model.backend()

    state = backend.release_state()

    assert state == model.engines[0].state and model.loaded == [backend.configs["noop"]]
    backend.config_module = SimpleNamespace(
        ReleaseConfig=SimpleNamespace(
            load=lambda path: SimpleNamespace(
                release_id=f"release-{Path(path).stem}",
                admin_config=SimpleNamespace(sha256=lambda: "d" * 64),
            )
        )
    )
    assert backend.candidate_releases() == {
        name: {
            "release_id": f"release-{Path(backend.configs[name]).stem}",
            "admin_config_sha256": "d" * 64,
        }
        for name in interrupted.DRILL_SCENARIOS
    }


# --- the fresh-run precondition -----------------------------------------------


@pytest.mark.parametrize(
    ("state", "site", "refused"),
    [
        (
            {
                "phase": "complete",
                "release_id": "rel-live",
                "transaction_committed": True,
            },
            "rel-live",
            False,
        ),
        (
            {
                "phase": "complete",
                "release_id": "rel-live",
                "transaction_committed": True,
            },
            None,
            False,
        ),
        (
            {
                "phase": "complete",
                "release_id": "rel-live",
                "transaction_committed": False,
            },
            "rel-live",
            True,
        ),
        (
            {
                "phase": "complete",
                "release_id": "rel-other",
                "transaction_committed": True,
            },
            "rel-live",
            True,
        ),
        (
            {
                "phase": "rolled-back",
                "release_id": "rel-live",
                "transaction_committed": True,
            },
            "rel-live",
            True,
        ),
        (
            {"phase": "complete", "release_id": "", "transaction_committed": True},
            None,
            True,
        ),
    ],
)
def test_only_the_sites_own_committed_release_admits_a_fresh_run(
    state: dict[str, Any], site: str | None, refused: bool
) -> None:
    refusal = interrupted.release_refusal(
        state, site_release_id=site, drill_release_ids={}
    )
    assert (refusal is not None) is refused, refusal


def test_the_refusal_names_phase_release_commit_and_the_way_out() -> None:
    refusal = interrupted.release_refusal(
        INTERRUPTED_D,
        site_release_id="rel-live",
        drill_release_ids={"noop": "rel-live", "full": CANDIDATE_D},
    )
    assert refusal is not None
    for text in (
        "phase 'data-plane-progress'",
        f"release_id {CANDIDATE_D!r}",
        "transaction_committed=False",
        "(this drill's full release)",
        "the site's own committed release 'rel-live'",
        "--resume on its run directory",
    ):
        assert text in refusal, (text, refusal)


@pytest.mark.parametrize(
    ("resume", "start_stage", "roundtrip_recorded", "expected"),
    [
        (False, "noop", False, True),
        (False, "noop", True, True),
        (True, "noop", False, True),
        (True, "noop", True, False),
        (False, "agent", True, False),
    ],
)
def test_which_runs_must_find_the_sites_own_release(
    tmp_path: Path,
    resume: bool,
    start_stage: str,
    roundtrip_recorded: bool,
    expected: bool,
) -> None:
    case_path = tmp_path / "case.json"
    if roundtrip_recorded:
        recorder = EvidenceRecorder(case_path, case_id=rolling.CASE_ID, inputs={})
        recorder.stage(interrupted.ROUNDTRIP_STAGE, lambda: {"restored": True})
    arguments = SimpleNamespace(resume=resume, start_stage=start_stage)
    assert interrupted.must_start_on_site_release(arguments, case_path) is expected


def test_the_sites_release_comes_from_the_snapshot_or_the_deploy_record(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "dist").mkdir(parents=True)
    (snapshot / "dist/current-release.json").write_text('{"release_id": "rel-snap"}')
    state = tmp_path / "state"
    state.mkdir()
    candidates = {"action": "reused", "snapshot_repo": str(snapshot)}
    assert interrupted.site_release_id(state, candidates) == "rel-snap"
    explicit = {"action": "explicit"}
    assert interrupted.site_release_id(state, explicit) is None
    (state / "source-deploy-success.json").write_text(
        json.dumps({"live": {"release_id": "rel-deployed"}})
    )
    assert interrupted.site_release_id(state, explicit) == "rel-deployed"


def test_an_unreadable_live_release_refuses_the_fresh_run(tmp_path: Path) -> None:
    def unreadable(_state_dir: Path) -> dict[str, Any]:
        raise OSError("cannot read live ConfigMap gpu-fault-regional-release-state")

    arguments = SimpleNamespace(resume=False, start_stage="noop")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(interrupted, "read_live_release_state", unreadable)
        refusal = interrupted.fresh_run_refusal(
            arguments,
            state_dir=tmp_path,
            candidates={"action": "explicit"},
            identity=None,
            case_path=tmp_path / "case.json",
        )
    assert refusal is not None and "could not be checked" in refusal, refusal
    assert "cannot read live ConfigMap" in refusal, refusal


def test_a_plan_over_an_interrupted_drill_does_not_pass_its_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, arguments, _site, calls = entrypoint(
        "020", tmp_path, monkeypatch, predecessor_valid=True
    )
    monkeypatch.setattr(
        interrupted, "read_live_release_state", lambda _state_dir: dict(INTERRUPTED_D)
    )
    monkeypatch.setattr(sys, "argv", arguments)

    assert runner.main() == 1, "a refused precondition must fail the plan"

    plan = json.loads(
        (tmp_path / "run" / "cases" / runner.CASE_ID / "plan.json").read_text()
    )
    assert plan["preflight_passed"] is False, plan
    (error,) = plan["details"]["preflight_errors"]
    assert "phase 'data-plane-progress'" in error and "--resume" in error, error
    monkeypatch.setattr(sys, "argv", [*arguments, *execution_flags(runner)])
    with pytest.raises(RuntimeError, match="preflight"):
        runner.main()
    assert calls == [], "a refused plan reached a live backend"


def test_an_execute_rechecks_the_live_release_before_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan saw the site's release; by the execute a transaction is open."""

    runner, arguments, _site, calls = entrypoint(
        "020", tmp_path, monkeypatch, predecessor_valid=True
    )
    roundtrips = []
    monkeypatch.setattr(
        runner, "public_config_roundtrip", lambda *_a, **_k: roundtrips.append(1)
    )
    monkeypatch.setattr(sys, "argv", arguments)
    assert runner.main() == 0, "the committed site release passes the plan"
    monkeypatch.setattr(
        interrupted, "read_live_release_state", lambda _state_dir: dict(INTERRUPTED_D)
    )
    monkeypatch.setattr(sys, "argv", [*arguments, *execution_flags(runner)])

    with pytest.raises(rolling.AcceptanceCheckError, match="cannot start a fresh run"):
        runner.main()

    assert calls == [] and roundtrips == [], (calls, roundtrips)
    evidence = tmp_path / "run" / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json"
    assert not evidence.exists(), "a refused execute published evidence"


@pytest.mark.parametrize("recorded", [False, True])
def test_a_resume_reads_the_live_release_only_when_it_starts_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: bool
) -> None:
    entry = RollingEntry(tmp_path, monkeypatch)
    backend = FakeReleaseRollingBackend()
    monkeypatch.setattr(rolling, "LiveReleaseRollingBackend", lambda _configs: backend)
    if recorded:
        # An interrupted run's evidence: it already went through the
        # roundtrip and owns whatever the site is on now.
        entry.live_release = dict(INTERRUPTED_D)
        entry.path.parent.mkdir(parents=True)
        first = EvidenceRecorder(
            entry.path, case_id=rolling.CASE_ID, inputs={"unused": True}
        )
        first.stage(interrupted.ROUNDTRIP_STAGE, lambda: {"restored": True})
        document = json.loads(entry.path.read_text(encoding="utf-8"))
        assert interrupted.ROUNDTRIP_STAGE in document["stages"]
    monkeypatch.setattr(sys, "argv", [*entry.argv, "--resume"])

    reads_before = len(entry.live_reads)
    if recorded:
        # The inputs differ from this unit evidence, which is enough to show
        # the precondition let it through to the evidence check.
        with pytest.raises(rolling.AcceptanceCheckError, match="inputs differ"):
            rolling.main()
    else:
        assert rolling.main() == 0
    assert len(entry.live_reads) - reads_before == (0 if recorded else 1)
