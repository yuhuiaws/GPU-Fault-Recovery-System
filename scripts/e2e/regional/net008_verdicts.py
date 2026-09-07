"""Pure verdict functions and constants of GF-REGIONAL-NET-008.

The case proves ARCH-G2/G3/G4 on a real regional deployment, in three phases
on one idle node's kernel collector:

* **A -- 403 is transient.** With a deliberately wrong cluster token the
  collector's posts are refused with 403; the records go to the outbox as
  ``replayable=true`` (never dead-lettered), ``gpu-fault-collector outbox
  stats`` counts them, the service does not crash-loop, and once the real
  token is back they are delivered exactly once.
* **B -- a replay 4xx dead-letters.** A record naming a channel the control
  plane does not serve is 404 at replay: it stays in the file as
  ``replayable=false``, ``stats`` shows it under ``dead``, ``list`` shows only
  metadata, and ``requeue-dead --yes`` (refused without ``--yes``) flips it
  back to replayable.
* **C -- the kernel stream is kept.** During a NET-001-style blackout the
  collector's ``/dev/kmsg`` fd, PID and invocation stay the same and its
  position only grows; the line written during the blackout arrives after it.

Every function here judges documents the runner wrote and touches no cluster.
"""

from __future__ import annotations

from typing import Any

CASE_ID = "GF-REGIONAL-NET-008"
CONFIRMATION = "NET008_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-NET-007"
UNIT = "gpu-fault-kernel-collector.service"
COLLECTOR = "kernel"
RETIRED_CHANNEL_PATH = "/v1/collector-events/retired-acceptance-channel"
TRANSIENT_EVENTS = 2
WINDOW_RESTORE_SECONDS = 600
BLOCK_TTL_SECONDS = 300
BLOCK_SECONDS = 90
REPLAY_TIMEOUT_SECONDS = 420


def window_errors(opened: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if opened.get("unit") != UNIT:
        errors.append(f"window opened on {opened.get('unit')!r}, not {UNIT!r}")
    if "GPU_FAULT_CONTROL_PLANE_TOKEN" not in (opened.get("overrides") or []):
        errors.append("the window did not override the cluster token")
    if (opened.get("after") or {}).get("ActiveState") != "active":
        errors.append("the kernel collector is not active after the window opened")
    return errors


def transient_outbox_errors(
    records: list[dict[str, Any]], stats: dict[str, Any], *, expected: int
) -> list[str]:
    """The refused events wait as replayable records, none of them dead."""

    marked = [item for item in records if item.get("marker_present")]
    errors: list[str] = []
    if len(marked) != expected:
        errors.append(f"{len(marked)} marked outbox records, expected {expected}")
    for item in marked:
        if item.get("replayable") is not True:
            errors.append(f"a 403-refused record is not replayable: {item}")
        if "403" not in str(item.get("error") or ""):
            errors.append(f"a refused record does not name the 403: {item}")
    if int(stats.get("dead") or 0):
        errors.append(
            f"outbox stats count {stats.get('dead')} dead record(s) after 403s"
        )
    if int(stats.get("replayable") or 0) < expected:
        errors.append(
            f"outbox stats count {stats.get('replayable')} replayable, expected "
            f"at least {expected}"
        )
    return errors


def service_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Deliberate restarts (open/close) are ours; automatic ones are a crash loop."""

    errors: list[str] = []
    if after.get("ActiveState") != "active":
        errors.append(f"{UNIT} is not active: {after}")
    if after.get("NRestarts") != before.get("NRestarts"):
        errors.append(
            f"{UNIT} restarted on its own: NRestarts {before.get('NRestarts')} -> "
            f"{after.get('NRestarts')}"
        )
    return errors


def delivered_errors(
    evidence: list[dict[str, Any]],
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    markers: list[str],
) -> list[str]:
    """Every buffered event arrives exactly once and leaves the outbox."""

    errors: list[str] = []
    for marker in markers:
        hits = [item for item in evidence if marker in str(item.get("payload"))]
        if len(hits) != 1:
            errors.append(
                f"marker {marker} has {len(hits)} evidence records, expected 1"
            )
    if any(item.get("marker_present") for item in records):
        errors.append("delivered records are still in the outbox")
    if int(stats.get("replayable") or 0):
        errors.append(f"{stats.get('replayable')} replayable record(s) remain")
    return errors


def dead_letter_errors(
    records: list[dict[str, Any]], stats: dict[str, Any], *, marker: str
) -> list[str]:
    """The retired-channel record is dead, kept, and counted."""

    seeded = [
        item
        for item in records
        if item.get("path") == RETIRED_CHANNEL_PATH and item.get("marker_present")
    ]
    if len(seeded) != 1:
        return [f"{len(seeded)} seeded records in the outbox, expected 1 ({marker})"]
    errors: list[str] = []
    record = seeded[0]
    if record.get("replayable") is not False:
        errors.append(f"the retired-channel record was not dead-lettered: {record}")
    if "404" not in str(record.get("error") or ""):
        errors.append(f"the dead letter does not name the 404: {record.get('error')!r}")
    if int(stats.get("dead") or 0) < 1:
        errors.append(f"outbox stats count no dead record: {stats}")
    return errors


def listing_errors(lines: list[str]) -> list[str]:
    """``outbox list`` prints metadata only -- never a payload body."""

    errors: list[str] = []
    if not any("dead" in line and RETIRED_CHANNEL_PATH in line for line in lines):
        errors.append("outbox list does not show the dead retired-channel record")
    if any("acceptance-dead-letter-" in line for line in lines):
        errors.append("outbox list leaks the record payload (record_id)")
    return errors


def requeue_errors(requeued: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if requeued.get("refused_without_yes") is not True:
        errors.append("requeue-dead ran without --yes")
    if "requeued 1 dead record" not in str(requeued.get("output") or ""):
        errors.append(
            f"requeue-dead did not report one record: {requeued.get('output')!r}"
        )
    stats = requeued.get("stats") or {}
    if int(stats.get("replayable") or 0) < 1:
        errors.append(f"the requeued record is not replayable in stats: {stats}")
    return errors


def stream_identity_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Same process, same invocation, same fd, position never went backwards."""

    errors: list[str] = []
    if not before.get("pid") or before.get("pid") != after.get("pid"):
        errors.append(
            f"kernel collector PID changed: {before.get('pid')} -> {after.get('pid')}"
        )
    if before.get("invocation_id") != after.get("invocation_id"):
        errors.append("kernel collector InvocationID changed during the blackout")
    fds_before = {
        int(item["fd"]): item.get("pos") for item in before.get("kmsg_streams") or []
    }
    fds_after = {
        int(item["fd"]): item.get("pos") for item in after.get("kmsg_streams") or []
    }
    if not fds_before:
        errors.append("the kernel collector held no /dev/kmsg stream at baseline")
    if set(fds_before) != set(fds_after):
        errors.append(
            f"the /dev/kmsg fd set changed: {sorted(fds_before)} -> {sorted(fds_after)}; "
            "the stream was reopened"
        )
    for fd, position in fds_before.items():
        later = fds_after.get(fd)
        if position is not None and later is not None and later < position:
            errors.append(f"fd {fd} position went backwards: {position} -> {later}")
    return errors


def blackout_errors(blocked: dict[str, Any], unblocked: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if any((blocked.get("connectivity") or {}).values()):
        errors.append("the control plane stayed reachable while blocked")
    if (blocked.get("timer") or {}).get("ActiveState") != "active":
        errors.append("the rollback timer was not active while blocked")
    if unblocked.get("rules"):
        errors.append(f"firewall rules remain after unblock: {unblocked.get('rules')}")
    if not all((unblocked.get("connectivity") or {}).values()):
        errors.append("the control plane is not reachable after unblock")
    return errors


def purge_errors(purged: dict[str, Any]) -> list[str]:
    if int(purged.get("removed") or 0) != 1:
        return [f"purge removed {purged.get('removed')} seeded record(s), expected 1"]
    return []


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
