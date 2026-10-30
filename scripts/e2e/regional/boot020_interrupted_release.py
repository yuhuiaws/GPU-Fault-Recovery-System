"""BOOT-020 against a release transaction an earlier run left open.

R13j attempt 23 (live 2026-10-29) was killed about two hours in, mid ``full``
stage, while the release engine rolled drill candidate D forward (phase
``data-plane-progress``, lifecycle ``ROLLING_CLUSTERS``, uncommitted); no
compensation ran. Attempt 24 started a fresh run and died 29 s later inside the
public AdminConfig apply -- the site was not on its own release -- and
``--resume`` could not have helped: its convergence finished only a pending
*rollback*, then deployed the previous stage's config as a new transaction,
which the engine refuses over an in-flight forward one. Two rules close that:

* ``fresh_run_refusal``: a run that starts at its beginning needs the site on
  its own committed release (``complete``, committed, the release the
  candidates derive from). The plan reports anything else as a preflight error
  (``preflight_passed`` false); the execute checks again before it builds or
  touches anything. The message names the way out: ``--resume`` on the
  interrupted run's directory.
* ``converge_to_precondition`` (``--resume``): an open forward transaction of
  one of the drill's own configs is finished first, the way
  ``gpu-fault-admin deploy`` finishes one -- ``regional_admin_commands.run_resume``
  on that config's release, automatic rollback on -- then a pending rollback,
  then the previous stage's config is deployed as before. A transaction of any
  other release is never resumed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_admin_commands
from scripts.e2e.regional import boot020_release_prerequisites as prerequisites
from scripts.e2e.regional.acceptance_runner_common import utc_now

# The drill's five release configs, in chain order; ``restore`` re-applies
# ``noop``. ``noop`` and ``control_plane`` name the site's own release and
# differ only in their AdminConfig.
DRILL_SCENARIOS = ("noop", "control_plane", "executor", "agent", "full")
ROUNDTRIP_STAGE = "public_admin_config_roundtrip"
COMMITTED_PHASE = "complete"
# An upgrade checkpoint the engine resumes from. ``complete`` with an open
# commit is ``_upgrade_resume_required`` too, but that is where every drill
# stage leaves its candidate (the drill never commits one), and a new
# transaction over it is what the chain does at every stage; resuming it would
# only re-enter the engine for nothing.
OPEN_FORWARD_PHASES = regional_admin_commands.RESUMABLE_PHASES


# --- the fresh-run precondition ---------------------------------------------


def recorded_stages(case_path: Path) -> dict[str, Any]:
    """The stages this run's evidence already recorded; read-only, ``{}`` if none."""

    try:
        document = json.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    stages = document.get("stages") if isinstance(document, dict) else None
    return stages if isinstance(stages, dict) else {}


def must_start_on_site_release(arguments: Any, case_path: Path) -> bool:
    """Whether this run needs the site on its own committed release first.

    Every run from ``noop`` does, except a ``--resume`` whose evidence already
    holds the public AdminConfig roundtrip: that one converges whatever the
    interrupted attempt left. A non-resume rerun over such evidence would replay
    stale records, which is what ``--resume`` is for, so it is held to the
    precondition too. ``--start-stage`` reruns find the site on a drill
    candidate by design.
    """

    if arguments.start_stage != "noop":
        return False
    return not arguments.resume or ROUNDTRIP_STAGE not in recorded_stages(case_path)


def site_release_id(state_dir: Path, candidates: Mapping[str, Any]) -> str | None:
    """The site's own committed release: the deployed snapshot's release id.

    Derived candidates name the snapshot they are built from (``snapshot_repo``,
    also before a build-at-execute); its ``dist/current-release.json`` is what
    the noop config deploys and what the public AdminConfig apply demands live
    (the candidates identity's ``snapshot_release_id``). Explicit configs fall
    back to the deploy record's live release, and to none without a record --
    phase and commit are still checked.
    """

    snapshot = candidates.get("snapshot_repo")
    if snapshot:
        return prerequisites.snapshot_release_id(Path(str(snapshot)))
    if (state_dir / "source-deploy-success.json").is_file():
        return prerequisites.live_release_id(state_dir)
    return None


def read_live_release_state(state_dir: Path) -> dict[str, Any]:
    """The live release state, read as the public AdminConfig apply reads it.

    The engine's state ConfigMap through ``gpu_fault.admin.release_state`` --
    the read whose refusal stopped attempt 24 -- from the site alone: a
    build-at-execute plan has no release config yet for the engine backend.
    """

    from gpu_fault.admin.release_state import live_release_state as admin_read
    from gpu_fault.admin.site import load_site

    return admin_read(load_site(state_dir / "site.yaml"))


def release_refusal(
    state: Mapping[str, Any],
    *,
    site_release_id: str | None,
    drill_release_ids: Mapping[str, Any],
) -> str | None:
    """Why ``state`` is no place to start a fresh run, or ``None``."""

    phase = str(state.get("phase") or "")
    release_id = str(state.get("release_id") or "")
    committed = state.get("transaction_committed")
    if (
        phase == COMMITTED_PHASE
        and committed is True
        and release_id
        and (site_release_id is None or release_id == site_release_id)
    ):
        return None
    owners = [
        name for name in DRILL_SCENARIOS if drill_release_ids.get(name) == release_id
    ]
    expected = f" {site_release_id!r}" if site_release_id else ""
    return (
        f"BOOT-020 cannot start a fresh run: the live regional release is in phase "
        f"{phase or 'none'!r}, release_id {release_id or 'none'!r}, "
        f"transaction_committed={committed!r}"
        + (f" (this drill's {'/'.join(owners)} release)" if owners else "")
        + f"; a fresh run needs the site's own committed release{expected} "
        "(phase 'complete', transaction_committed=True), and the public "
        "AdminConfig apply refuses anything else. If a BOOT-020 run was "
        "interrupted, continue THAT run: --resume on its run directory (plan and "
        "execute the same command line with --attempt <n> --resume) finishes the "
        "open transaction and converges the site; any other open transaction "
        "belongs to whoever started it (gpu-fault-admin deploy resumes its own "
        "release)"
    )


def fresh_run_refusal(
    arguments: Any,
    *,
    state_dir: Path,
    candidates: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    case_path: Path,
) -> str | None:
    """The plan's preflight error and the execute's refusal; ``None`` = may start.

    Read-only, and fail closed: a state that cannot be read proves nothing.
    ``identity`` (``candidate_identity``, absent before a build-at-execute) only
    lets the message name the drill config that owns the open release.
    """

    if not must_start_on_site_release(arguments, case_path):
        return None
    try:
        state = read_live_release_state(state_dir)
        expected = site_release_id(state_dir, candidates)
    except Exception as exc:  # noqa: BLE001 - reported as the refusal
        return (
            "BOOT-020 cannot start a fresh run: the live regional release could "
            f"not be checked against the site's own release ({type(exc).__name__}: "
            f"{exc})"
        )
    return release_refusal(
        state,
        site_release_id=expected,
        drill_release_ids=dict((identity or {}).get("release_ids") or {}),
    )


# --- --resume after an interrupted forward roll -------------------------------


def forward_roll_scenario(
    state: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]
) -> str:
    """The drill config whose release the open transaction rolls; else refuse.

    The release id alone is ambiguous -- ``noop`` and ``control_plane`` both
    name the site's own release -- so the state's ``admin_config_sha256`` (the
    engine stamps it on every checkpoint) must match as well. Nothing matching
    both is a release the drill did not derive, and it is never resumed here.
    """

    release_id = str(state.get("release_id") or "")
    admin = state.get("admin_config_sha256")
    for scenario in DRILL_SCENARIOS:
        candidate = candidates.get(scenario) or {}
        if candidate.get("release_id") == release_id and (
            admin is None or candidate.get("admin_config_sha256") == admin
        ):
            return scenario
    known = ", ".join(
        f"{name}={(candidates.get(name) or {}).get('release_id')}"
        for name in DRILL_SCENARIOS
    )
    raise RuntimeError(
        "BOOT-020 will not resume the open release transaction of "
        f"{release_id or 'an unnamed release'} (phase {state.get('phase')!r}, "
        f"admin_config_sha256 {admin!r}): it is not one of this drill's releases "
        f"({known}). Finish it with whatever started it (gpu-fault-admin deploy "
        "resumes its own release), then rerun --resume"
    )


def finish_interrupted_forward_roll(backend: Any, recorder: Any) -> dict[str, Any]:
    """Finish an open forward transaction of a drill config before converging.

    Recorded in the convergence history as soon as it ran -- a resumed roll
    moved the site even when a later convergence step fails. A resume whose
    end state still has an open forward transaction refuses the convergence
    (a new transaction over it is what the engine refuses); one that ended
    ``rolled-back`` or in a rollback phase leaves that to the pending-rollback
    branch and the deploy that follow.
    """

    state = backend.release_state()
    phase = str(state.get("phase") or "")
    if phase not in OPEN_FORWARD_PHASES:
        return {
            "resumed": False,
            "phase": phase,
            "release_id": state.get("release_id"),
        }
    scenario = forward_roll_scenario(state, backend.candidate_releases())
    result = {"resumed": True, **backend.resume_upgrade(scenario)}
    recorder.note(
        "convergence",
        [
            *recorder.document.get("convergence", []),
            {"finished_interrupted_release": scenario, **result, "at": utc_now()},
        ],
    )
    if str(result.get("phase") or "") in OPEN_FORWARD_PHASES:
        raise RuntimeError(
            f"BOOT-020 could not finish the interrupted {scenario} release "
            f"{state.get('release_id')} before resuming: {result}"
        )
    return result


def converge_to_precondition(
    backend: Any, recorder: Any, scenario: str
) -> dict[str, Any]:
    """Bring the live release back to the precondition ``scenario`` leaves.

    ``scenario`` is the stage before the one ``--resume`` restarts, and that
    stage begins by classifying its config against the state ``scenario``
    left, and asserting the classification. A failed stage's auto-rollback left
    the live state at that precondition already, but pinned at
    ``rolled-back``; classifying and deploying ``scenario``'s config is a NOOP
    that clears the phase to ``complete`` without touching artifacts. In the
    rarer case where the stage failed after applying its own config, the same
    deploy really rolls the artifacts back to the precondition. Either way the
    resume then runs exactly as a first pass would. Before that deploy, an
    interrupted forward roll of a drill config is finished
    (``finish_interrupted_forward_roll``) and a pending rollback too: the
    engine refuses a new transaction over either.
    """

    forward = finish_interrupted_forward_roll(backend, recorder)
    # A failed stage may have left the engine's transaction mid-rollback
    # (live 2026-09-20 08:02Z: a HyperPod software update rebooted every CPU
    # node during the control_plane stage; the injected upgrade failed for real
    # and the automatic rollback died on an unavailable webhook, phase
    # rollback-failed). Finish the rollback first; a clean state is a no-op.
    rollback_resume = backend.resume_rollback(scenario)
    if rollback_resume.get("resumed") and rollback_resume.get("phase") not in {
        "rolled-back",
        "complete",
    }:
        raise RuntimeError(
            "BOOT-020 could not finish the pending rollback before resuming: "
            f"{rollback_resume}"
        )
    diff = backend.classify(scenario)
    applied = backend.deploy(scenario, diff=diff)
    if applied.get("phase") != "complete":
        raise RuntimeError(
            f"BOOT-020 could not converge the live state to the {scenario} "
            f"precondition before resuming: {applied}"
        )
    history = list(recorder.document.get("convergence", []))
    history.append(
        {
            "converged_to": scenario,
            "forward_resume": forward,
            "rollback_resume": rollback_resume,
            "classification": diff,
            "phase": applied.get("phase"),
            "at": utc_now(),
        }
    )
    recorder.note("convergence", history)
    return applied


# --- LiveReleaseRollingBackend methods (bound in run_boot020_release_rolling) --


def live_release_state(backend: Any) -> dict[str, Any]:
    """The engine's release state, read once (any drill config's release reads
    the same state ConfigMap; ``noop`` is the site's own)."""

    state: dict[str, Any] = backend._release("noop")._load_state()
    return state


def live_candidate_releases(backend: Any) -> dict[str, dict[str, str]]:
    """Each drill config's release id and AdminConfig digest, from its file."""

    releases: dict[str, dict[str, str]] = {}
    for scenario in DRILL_SCENARIOS:
        config = backend.config_module.ReleaseConfig.load(backend.configs[scenario])
        releases[scenario] = {
            "release_id": str(config.release_id),
            "admin_config_sha256": str(config.admin_config.sha256()),
        }
    return releases


def live_resume_upgrade(backend: Any, scenario: str) -> dict[str, Any]:
    """``run_resume`` on ``scenario``'s release with automatic rollback on.

    What ``gpu-fault-admin deploy`` does for an interrupted transaction of its
    own candidate: pin the approved manifest digest the transaction was planned
    against (M-23) and resume the upgrade from its checkpoints. With automatic
    rollback on, a resume that fails compensates instead of leaving the site
    mid-roll. The engine's error is returned with the state it left, not
    raised, so the convergence records both and decides.
    """

    release = backend._release(scenario, auto_rollback=True)
    before = release._load_state()
    started = time.monotonic()
    error = None
    try:
        backend.commands.run_resume(release)
    except Exception as exc:  # noqa: BLE001 - recorded with the state it left
        error = f"{type(exc).__name__}: {exc}"
    after = release._load_state()
    return {
        "scenario": scenario,
        "release_id": before.get("release_id"),
        "phase_before": before.get("phase"),
        "phase": after.get("phase"),
        "transaction_committed": after.get("transaction_committed") is True,
        "error": error,
        "operation_duration_seconds": time.monotonic() - started,
    }
