"""Run the independent preflight checks of a transaction at the same time.

Every kubectl call costs about a second from the deploy host, and the upgrade
preflight used to spend them one after another: the Aurora credential refresh
Job (create, ``wait``, delete: 8-10 s), the remote-command idle probe (an exec
in the ingress Pod, 2-3 s), the in-flight install gate (another exec, 2-3 s)
and the previous-state capture (about thirty reads). Live, the whole
``preflight`` phase of a control-plane-only config release took 79 s.

The steps are independent of each other, so they run as *lanes* on a small
thread pool:

* ``aurora-refresh`` rewrites the ``gpu-fault-aurora`` Secret and nothing
  else. It exists because RDS rotates the master password every 7 days and a
  release is a burst of new Pods (2026-09-07: an automatic rollback restarted
  control-worker into a rotated-out password and the transaction landed in
  ``rollback-failed``). It fails closed. It stays behind ``_apply_rds_ca_bundle``
  because the refresh Job mounts the bundle.
* ``store-probes`` reads the store through the memoised ingress Pod: the idle
  probe first, then the in-flight install gate, in their original order and on
  one thread, so the two never race each other for the memoised Pod name.
* ``previous-capture`` validates the transaction and then snapshots the live
  release (ConfigMaps, Deployments, the release-metadata and state ConfigMaps,
  the AMP/Route53 objects). It reads no Secret the refresh writes -- the only
  Secrets it touches are the ones it backs up, and that backup is a mutation
  that still happens *after* every lane has passed (``_upgrade_context``).

Failure semantics are the serial ones: every lane runs to its own end (a
half-finished ``kubectl create job`` cannot be taken back by killing it), then
the first failure in declared order is raised unchanged and the others are
printed. Nothing here saves state; the caller's first ``_save_state`` still
comes after everything has passed. A dry run keeps the serial order so its
trace stays readable, as the GPU stage does (``regional_release_gpu_stage``).

The rollback's pair -- the same refresh, then the in-flight install gate before
any restore is planned (the previous release's control plane would re-submit a
PENDING/WAITING install) -- uses the same runner.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step

PREFLIGHT_NARRATION = "preflight-concurrent"


@dataclass(frozen=True)
class PreflightLane:
    """One independent preflight step, named for the failure report."""

    name: str
    action: Callable[[], Any]


@dataclass(frozen=True)
class UpgradePreflight:
    """What the upgrade preflight lanes produced for the transaction."""

    verdict: Any
    acceptance: dict[str, Any] | None
    previous: dict[str, Any] | None


def _dry_run(release: Any) -> bool:
    # A test double models the release as a bare namespace without a runner; it
    # takes the concurrent path, which is the one worth exercising.
    return bool(getattr(getattr(release, "runner", None), "dry_run", False))


def _timed(lane: PreflightLane, durations: dict[str, float]) -> Any:
    started = time.monotonic()
    try:
        return lane.action()
    finally:
        durations[lane.name] = time.monotonic() - started


def run_preflight_lanes(
    release: Any,
    lanes: Sequence[PreflightLane],
    *,
    phase: str,
) -> dict[str, Any]:
    """Run every lane at once; return ``{name: result}``.

    Raises the first failure in declared order after every lane has finished.
    The narrated line says what each lane cost, the wall time, and how much the
    overlap saved against running them one after another.
    """

    durations: dict[str, float] = {}
    results: dict[str, Any] = {}
    failures: list[tuple[PreflightLane, Exception]] = []
    wall_started = time.monotonic()
    if len(lanes) == 1 or _dry_run(release):
        for lane in lanes:
            results[lane.name] = _timed(lane, durations)
    else:
        with ThreadPoolExecutor(
            max_workers=len(lanes),
            thread_name_prefix=f"{phase}-lane",
        ) as pool:
            futures = [(lane, pool.submit(_timed, lane, durations)) for lane in lanes]
            for lane, future in futures:
                try:
                    results[lane.name] = future.result()
                except Exception as exc:
                    failures.append((lane, exc))
    wall = time.monotonic() - wall_started
    serial = sum(durations.values())
    narrate_step(
        PREFLIGHT_NARRATION,
        phase=phase,
        **{name: f"{seconds:.1f}s" for name, seconds in durations.items()},
        wall=f"{wall:.1f}s",
        serial=f"{serial:.1f}s",
        saved=f"{max(0.0, serial - wall):.1f}s",
    )
    if not failures:
        return results
    for lane, error in failures[1:]:
        print(
            f"{phase}: {lane.name} also failed while the lanes ran: {error}",
            file=sys.stderr,
            flush=True,
        )
    raise failures[0][1]


def store_preflight_probes(release: Any, *, action: str) -> Any:
    """The two store probes in their original order, on the calling thread.

    Both exec into the memoised CPU ingress Pod; keeping them on one thread means
    the memo is only ever read and refreshed by one of them at a time. Returns
    the in-flight install verdict (``regional_release_store_preflight``).
    """

    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    # One store read; refuses by name while a driver/firmware/EFA install is
    # PENDING or WAITING.
    return release._require_no_inflight_installs(action=action)


def run_upgrade_preflight(
    release: Any,
    *,
    validate: Callable[[], dict[str, Any] | None],
    capture: Callable[[], dict[str, Any]] | None,
) -> UpgradePreflight:
    """The upgrade's preflight: refresh, store probes, validate-then-capture.

    ``validate`` is the transaction validation that used to run after the
    probes; it stays ahead of ``capture`` on the same lane so a refused
    transaction never pays for the snapshot, exactly as before. ``capture`` is
    ``None`` for a resume or a superseding transaction, which take their
    previous state from the persisted one.
    """

    def validate_then_capture() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        acceptance = validate()
        return acceptance, (capture() if capture is not None else None)

    results = run_preflight_lanes(
        release,
        (
            PreflightLane("aurora-refresh", release._refresh_aurora_credentials),
            PreflightLane(
                "store-probes",
                lambda: store_preflight_probes(release, action="upgrade"),
            ),
            PreflightLane("previous-capture", validate_then_capture),
        ),
        phase="upgrade-preflight",
    )
    acceptance, previous = results["previous-capture"]
    return UpgradePreflight(
        verdict=results["store-probes"],
        acceptance=acceptance,
        previous=previous,
    )


def run_rollback_preflight(
    release: Any,
    *,
    check_installs: bool,
    automatic: bool,
) -> Any:
    """The rollback's preflight: the refresh beside the in-flight install gate.

    The gate is skipped once the control plane is already restored (a cleanup
    re-entry). An automatic rollback proceeds only when no Running control-plane
    Pod with a ready container could run the probe (``StoreUnreachable``); a
    kubectl-level failure on a ready Pod and a manual rollback refuse. Returns
    the gate's verdict (``None`` when it did not run) for the caller to persist.
    """

    lanes = [PreflightLane("aurora-refresh", release._refresh_aurora_credentials)]
    if check_installs:
        lanes.append(
            PreflightLane(
                "inflight-installs",
                lambda: release._require_no_inflight_installs(
                    action="rollback",
                    unreadable="proceed" if automatic else "refuse",
                ),
            )
        )
    results = run_preflight_lanes(release, lanes, phase="rollback-preflight")
    return results.get("inflight-installs")
