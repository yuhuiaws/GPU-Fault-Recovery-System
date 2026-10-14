"""A run's data settles before its purge; what lands after it is purged too.

Live 2026-10-14 (formal suite): action-raw-32x4 purged 10,112 records and
2,816 links and deregistered its 32 synthetic clusters; 57 s later the
correlated suite's registration preflight found 117 records and 39 links on
perf-cap-000..031 and refused ("synthetic cluster data already exists"). A
teardown with the action run's id then removed exactly those as that run's
drill residue. 117 records and 39 links are 39 notifications, each with its
delivery and result, and their deduplication links: the signature of
``save_notification_if_absent`` plus a delivered send. The control plane's
collector-silence scan (``gpu_fault.app.collector_silence``, every 60 s, one
advisory per silent collector and hour bucket; synthetic nodes never report)
reads a cluster's Agents and then writes their advisories, so a scan that
listed the run's Agents before the purge committed wrote them after it -- and
the purge had just removed the deduplication links that would have absorbed
them.

So the shared purge (:func:`purge_run_data`, the body of every suite's
``purge_audit_rows``):

1. waits until the run's data settles -- no nonterminal workflow or command,
   no open processor row, the same counts in two probes ``QUIET_SECONDS``
   apart -- for at most ``SETTLE_SECONDS``, and no longer than
   ``STUCK_WINDOWS`` unchanged windows while open work remains (the run's
   executors are gone by then; nothing finishes it);
2. purges, exactly as before (one transaction, every ownership check);
3. probes again ``QUIET_SECONDS`` later and, while run-owned rows keep
   appearing, settles and purges again -- at most ``LATE_PURGES`` times, each
   one logged and recorded by kind, notification source and table; rows still
   appearing after that fail the teardown attempt, which the retried teardown
   repeats.

A registration whose data preflight finds residue (:func:`adopt_earlier_run_residue`)
adopts it only when every row is provably an earlier run's: the run the
registry history last registered these clusters for, every row that run's
drill residue -- what that run's own teardown accepts once its clusters are
deregistered. It then waits up to ``ADOPT_SETTLE_SECONDS`` for the residue to
settle, runs that run's purge and logs it. Anything else refuses as before.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from . import regional_capacity_data as capacity_data
else:
    import regional_capacity_data as capacity_data

QUIET_SECONDS = 30.0
SETTLE_SECONDS = 300.0
STUCK_WINDOWS = 2
LATE_PURGES = 3
ADOPT_SETTLE_SECONDS = 120.0
# Probe fields that describe ownership, not the data a quiet window compares.
_OWNERSHIP_FIELDS = frozenset({"residue_owner", "residue_owner_proven"})
REFUSAL = "synthetic cluster data already exists; registration cannot adopt it"

Probe = Callable[[], dict[str, Any]]


def describe(document: dict[str, Any], *, deleted: bool = False) -> str:
    """Counts of a probe (or, ``deleted``, of a purge result) on one line."""

    kinds = document.get("deleted_by_kind" if deleted else "by_kind") or {}
    sources = document.get("notification_sources") or {}
    tables = (
        document.get("deleted_table_rows") or {}
        if deleted
        else {
            key: value
            for key, value in document.items()
            if key.startswith("gpu_fault_") and type(value) is int and value
        }
    )
    links = document.get("deleted_links" if deleted else "links") or 0
    parts = []
    for kind, count in sorted(kinds.items()):
        note = ""
        if kind == "notification" and sources:
            note = (
                " ["
                + ", ".join(f"{name} {n}" for name, n in sorted(sources.items()))
                + "]"
            )
        parts.append(f"{kind}={count}{note}")
    summary = ", ".join(parts) or "no records"
    if links:
        summary += f"; links {links}"
    if tables:
        summary += "; table rows " + ", ".join(
            f"{table}={count}" for table, count in sorted(tables.items())
        )
    return summary


def _fingerprint(probe: dict[str, Any]) -> str:
    return json.dumps(
        {key: value for key, value in probe.items() if key not in _OWNERSHIP_FIELDS},
        sort_keys=True,
    )


def _open_work(probe: dict[str, Any]) -> bool:
    return bool(probe.get("nonterminal") or probe.get("open_queue_rows"))


def wait_until_settled(
    probe: Probe,
    *,
    quiet_seconds: float = QUIET_SECONDS,
    bound_seconds: float = SETTLE_SECONDS,
    stuck_windows: int = STUCK_WINDOWS,
    sleep: Callable[[float], None] | None = None,
    first: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe until two probes ``quiet_seconds`` apart agree with no open work.

    ``first`` is a probe just taken, if any. An empty first probe settles at
    once: nothing is there to land late. The waits are counted, not clocked,
    so the bound holds whatever the clock does. ``outcome`` is ``settled``,
    ``empty``, ``stuck`` (open work that did not move for ``stuck_windows``
    windows) or ``bound``.
    """

    pause = time.sleep if sleep is None else sleep
    probes = [probe() if first is None else first]
    if probes[0].get("total") == 0:
        return {"outcome": "empty", "probes": 1, "waited_seconds": 0.0}
    waited = 0.0
    unchanged = 0
    outcome = "bound"
    while waited + quiet_seconds <= bound_seconds:
        pause(quiet_seconds)
        waited += quiet_seconds
        probes.append(probe())
        same = _fingerprint(probes[-1]) == _fingerprint(probes[-2])
        unchanged = unchanged + 1 if same else 0
        if same and not _open_work(probes[-1]):
            outcome = "settled"
            break
        if unchanged >= stuck_windows:
            outcome = "stuck"
            break
    return {
        "outcome": outcome,
        "probes": len(probes),
        "waited_seconds": waited,
        "last": describe(probes[-1]),
        "open_work": _open_work(probes[-1]),
    }


def _intent(artifacts: Path | None, run_id: str | None) -> list[str]:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("capacity cleanup requires an explicit run identity")
    if artifacts is None:
        raise RuntimeError("capacity cleanup requires its registration receipt")
    intent = json.loads((artifacts / "registry-registration-intent.json").read_text())
    if (
        intent.get("run_id") != run_id
        or intent.get("data_empty_before_registration") is not True
    ):
        raise RuntimeError("capacity cleanup registration scope is unproven")
    return list(intent["cluster_ids"])


def purge_run_data(
    control: Callable[..., str],
    *,
    run_id: str | None,
    artifacts: Path | None,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Settle, purge, and purge again what lands late (module docstring)."""

    cluster_ids = _intent(artifacts, run_id)
    assert artifacts is not None and run_id is not None

    def probe() -> dict[str, Any]:
        return capacity_data.invoke(
            control, run_id=run_id, cluster_ids=cluster_ids, cleanup=False
        )

    def purge() -> dict[str, Any]:
        result = capacity_data.invoke(
            control,
            run_id=run_id,
            cluster_ids=cluster_ids,
            cleanup=True,
            force_nonterminal=True,  # executor Jobs are gone; open commands never finish
        )
        if type(result.get("total")) is not int or result["total"] != 0:
            raise RuntimeError("capacity exact run-owned cleanup left data")
        return result

    settle = wait_until_settled(probe, sleep=sleep)
    if settle["outcome"] not in {"settled", "empty"}:
        log(
            f"run {run_id} data did not settle ({settle['outcome']} after "
            f"{settle['waited_seconds']:.0f} s: {settle['last']}); purging anyway"
        )
    result = purge()
    late: list[dict[str, Any]] = []
    # Nothing at all before the purge: nothing can be in flight to land after it.
    rounds = 0 if settle["outcome"] == "empty" else LATE_PURGES + 1
    for round_number in range(1, rounds + 1):
        (time.sleep if sleep is None else sleep)(QUIET_SECONDS)
        after = probe()
        if after.get("total") == 0:
            break
        if round_number == rounds:
            raise RuntimeError(
                f"capacity run-owned data kept appearing after {len(late) + 1} "
                f"purges: {describe(after)}"
            )
        wait_until_settled(probe, sleep=sleep, first=after)
        removed = purge()
        late.append(removed)
        log(
            f"run {run_id} data appeared after its purge; purge {len(late) + 1} "
            f"removed {describe(removed, deleted=True)}"
        )
    document = {
        **result,
        "settle": settle,
        "late_purges": [
            {**item, "summary": describe(item, deleted=True)} for item in late
        ],
    }
    (artifacts / "capacity-data-cleanup.json").write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def adopt_earlier_run_residue(
    control: Callable[..., str],
    *,
    run_id: str,
    cluster_ids: list[str],
    found: dict[str, Any],
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Purge an earlier run's late residue before registering, or refuse.

    ``found`` is the registration's data preflight. Its ``residue_owner`` is
    the run the registry history last registered these clusters for, and
    ``residue_owner_proven`` that every row is that run's drill residue
    (``regional_capacity_data``); the purge re-proves both in its transaction.
    """

    owner = found.get("residue_owner")
    if (
        not isinstance(owner, str)
        or not owner
        or owner == run_id
        or found.get("residue_owner_proven") is not True
    ):
        why = (
            f"not provably run {owner}'s drill residue"
            if isinstance(owner, str) and owner
            else "no earlier run in the registry history owns it"
        )
        raise RuntimeError(f"{REFUSAL} ({describe(found)}; {why})")
    log(
        f"registration of run {run_id} found {describe(found)} on "
        f"{len(cluster_ids)} synthetic clusters, the drill residue of earlier run "
        f"{owner}; waiting for it to settle, then purging it as {owner}'s teardown "
        "would"
    )

    def probe() -> dict[str, Any]:
        return capacity_data.invoke(
            control, run_id=owner, cluster_ids=cluster_ids, cleanup=False
        )

    settle = wait_until_settled(
        probe, bound_seconds=ADOPT_SETTLE_SECONDS, sleep=sleep, first=found
    )
    try:
        # The cleanup proves the ownership again: no current registration, so
        # only that run's drill residue may be deleted (one transaction).
        removed = capacity_data.invoke(
            control, run_id=owner, cluster_ids=cluster_ids, cleanup=True
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{REFUSAL} (purging run {owner}'s residue: {exc})") from exc
    if type(removed.get("total")) is not int or removed["total"] != 0:
        raise RuntimeError(f"{REFUSAL} (the purge of run {owner}'s residue left data)")
    after = capacity_data.invoke(
        control, run_id=run_id, cluster_ids=cluster_ids, cleanup=False
    )
    if after.get("total") != 0:
        raise RuntimeError(f"{REFUSAL} ({describe(after)} after purging run {owner}'s)")
    log(
        f"purged earlier run {owner}'s residue: {describe(removed, deleted=True)}; "
        f"registering run {run_id}"
    )
    return {
        **after,
        "adopted_residue": {
            "owner_run_id": owner,
            "found": describe(found),
            "settle": settle,
            "purged": removed,
        },
    }
