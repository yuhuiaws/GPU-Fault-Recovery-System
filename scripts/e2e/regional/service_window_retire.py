"""Retire a stale DESTR-008 service window through the harness's own host probe.

A service window whose run never closed it keeps its controller journal open in
the case directory and its drop-in, units, claim and directory on the node
(live plain EKS 2026-10-27: the pre-``restore_through_node`` undo got no response
while kubelet was down; the window's own restore unit brought kubelet back at
``restore_at``, but nothing ever retired the window, and the R13h Agent wave
then changed the baseline fingerprints its cleanup compares). The journal can
be resumed only by the execution that wrote it, so a rerun -- and every later
window on that node, refused the claim -- was blocked for good (030 BLOCKED).

``retire_journal`` is the reusable, idempotent way out, and acts only on proof:

* local -- the journal is CLEANUP_REQUIRED or CREATING (a finished run's open
  custody), supervision was never lost, it is the node's and its binding is
  valid and its own; no process holds its lock; the Node still has the UID the
  journal recorded; the earlier host probe's Pod and ConfigMap are proven that
  record's by labels, owner annotation, script digest, UID and node
  (``host_probe_sweep.foreign_reason``);
* host -- a dedicated retire probe (``retire_probe_source``: the warm-spare
  probe's own source plus ``probes/warm_spare_retire.py``) re-proves the window
  on the node and removes exactly its artifacts (see that module).

Then the earlier probe's Pod and ConfigMap are deleted under UID preconditions,
its ownership record is closed, the retire probe is cleaned, and the journal is
marked CLOSED with a retirement record: who (the operator identity), when, by
what (``via``) and the host's proofs. Any failed proof leaves everything as it
was and is returned as a named refusal; the variant then stays
BLOCKED(FIXTURE_NOT_READY) with those reasons. A rerun of a partly finished
retirement resumes it: the host side is idempotent, a closed record or an
absent object is skipped, and the journal closes last.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.host_probe_sweep import (
    ProbeKubectl,
    foreign_reason,
    wait_gone,
)
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.probes import warm_spare_retire
from scripts.e2e.regional.regional_commands import (
    RegionalFixtureError,
    run_fixture_command,
)
from scripts.e2e.regional.service_window_node_restore import stale_journals

#: The fixture identity every service window is bound under (the on-node probe
#: honours a window only for it).
SERVICE_WINDOW_CASE = "GF-REGIONAL-DESTR-008"
#: A finished run's open custody: its cleanup ran and failed, or its create died.
RETIRABLE_PHASES = frozenset({"CLEANUP_REQUIRED", "CREATING"})
#: Under the journal's directory: the retire probe's script and ownership record.
RETIRE_STATE = "retire"
RETIRE_SCRIPT = "service-window-retire-probe.py"
HOST_PART_MARKER = "# --- retire probe host part ---\n"
BASE_TRAILER = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
RECORD_NAME = re.compile(r"gpu-fault-host-probe-[0-9a-f]{10}\.json")
RETIRE_PROBE_DEADLINE_SECONDS = 900


@dataclass(frozen=True)
class RetireSettings:
    """Where and through what a retirement runs.

    ``node_uid`` reads the Node's current UID through the API (``""`` when the
    Node is gone); ``host_probe`` builds the retire probe's transport
    (``HostProbeFixture``); ``run`` is the kubectl runner for the earlier probe's
    objects (``run_fixture_command`` when ``None``). Tests hand in fakes.
    """

    kubeconfig: Path
    context: str
    namespace: str
    image: str
    node_uid: Callable[[str], str]
    host_probe: Callable[[HostProbeSettings], Any] = HostProbeFixture
    run: Callable[..., Any] | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic


def retire_probe_source() -> str:
    """The retire probe: the warm-spare probe's source, then the host part.

    One file, stdlib only, as every host probe ships: the base's entry point is
    replaced by ``retire_main`` and ``probe`` is bound to the combined module, so
    the host part runs on the base's own names, receipts and systemd reads.
    """

    base = Path(probe.__file__).read_text(encoding="utf-8")
    extension = Path(warm_spare_retire.__file__).read_text(encoding="utf-8")
    _head, marker, host_part = extension.partition(HOST_PART_MARKER)
    if not base.endswith(BASE_TRAILER) or not marker:
        raise RegionalFixtureError("the retire probe's sources lost their seams")
    return (
        base[: -len(BASE_TRAILER)]
        + "\n# The retire probe: warm_spare_node_probe.py above, "
        "warm_spare_retire.py below.\n"
        + "import sys as _retire_sys  # noqa: E402\n\n"
        + "probe = _retire_sys.modules[__name__]\n\n"
        + host_part
        + '\n\nif __name__ == "__main__":\n    raise SystemExit(retire_main())\n'
    )


def write_retire_probe(directory: Path) -> Path:
    """The retire probe script in ``directory`` (private), rewritten if stale."""

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / RETIRE_SCRIPT
    source = retire_probe_source()
    try:
        if path.read_text(encoding="utf-8") == source:
            return path
    except FileNotFoundError:
        pass
    temporary = directory / f".{RETIRE_SCRIPT}.tmp"
    temporary.write_text(source, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return path


def retire_run_id(run_id: str) -> str:
    """The retire probe's own identity, distinct from the window's run."""

    value = f"{run_id}-retire"
    return (
        value
        if len(value) <= 63
        else f"retire-{sha256(run_id.encode()).hexdigest()[:16]}"
    )


def journal_refusals(data: Any, journal: Path, node: str) -> list[str]:
    """The local proofs: why this journal is not a finished run's own custody."""

    if not isinstance(data, dict):
        return ["the journal is not a JSON object"]
    refusals: list[str] = []
    phase = data.get("phase")
    if data.get("schema_version") != 1:
        refusals.append("the journal's schema is unknown")
    if phase not in RETIRABLE_PHASES:
        refusals.append(
            f"the journal is {phase!r}, not a finished run's open custody "
            "(CLEANUP_REQUIRED or CREATING)"
        )
    if data.get("supervision_lost") is not False:
        refusals.append("the journal records lost supervision: a command may still run")
    scope = data.get("scope")
    if not isinstance(scope, dict):
        return [*refusals, "the journal has no scope"]
    if scope.get("node") != node:
        refusals.append(f"the journal is for {scope.get('node')!r}, not {node}")
    if scope.get("case_id") != SERVICE_WINDOW_CASE:
        refusals.append("the journal is not a DESTR-008 service window's")
    for key in ("run_id", "node_uid", "context", "namespace"):
        if not isinstance(scope.get(key), str) or not scope[key]:
            refusals.append(f"the journal's scope has no {key}")
    host = scope.get("host_state_path")
    if (
        not isinstance(host, str)
        or not RECORD_NAME.fullmatch(Path(host).name)
        or Path(host).parent.resolve() != journal.parent.resolve()
    ):
        refusals.append("the journal's host-probe record is not beside it")
    binding = data.get("binding")
    if binding is not None:
        try:
            probe.binding_key(binding)
        except (probe.ProbeError, KeyError, TypeError, ValueError) as exc:
            refusals.append(f"the journal's binding is invalid ({exc})")
        else:
            if (
                binding["node"] != node
                or binding["owner"] != data.get("owner")
                or binding["run_id"] != scope.get("run_id")
            ):
                refusals.append("the binding is not this journal's own")
        if phase == "CREATING":
            refusals.append("a CREATING journal cannot carry a binding")
    return refusals


def inspect_journal(
    journal: Path, node: str, *, settings: RetireSettings
) -> dict[str, Any]:
    """The read-only half of ``retire_journal``: the local proofs, the Node,
    the earlier probe's objects and what the window would leave on the node.
    Nothing is locked, created or deleted."""

    entry: dict[str, Any] = {"journal": journal.name, "node": node, "refusals": []}
    refusals: list[str] = entry["refusals"]
    try:
        data = probe.private_read(journal)
    except (OSError, ValueError, probe.ProbeError) as exc:
        refusals.append(f"the journal cannot be read ({type(exc).__name__})")
        return entry
    refusals.extend(journal_refusals(data, journal, node))
    recorded = data.get("scope")
    scope: dict[str, Any] = recorded if isinstance(recorded, dict) else {}
    entry.update(
        run_id=scope.get("run_id"),
        release_id=scope.get("release_id"),
        phase=data.get("phase"),
        last_failure=data.get("last_failure"),
    )
    if refusals:
        return entry
    if (scope["context"], scope["namespace"]) != (settings.context, settings.namespace):
        refusals.append("the journal belongs to another cluster context or namespace")
        return entry
    current = settings.node_uid(node)
    if current != scope["node_uid"]:
        refusals.append(
            f"{node} is gone or was replaced (Node UID {current or 'absent'})"
        )
        return entry
    record_path = Path(scope["host_state_path"])
    record = _read_record(record_path, refusals)
    objects = _probe_objects(record_path, record, settings, refusals)
    entry["host_probe"] = {
        "record": record_path.name,
        "record_closed": None if record is None else record["closed"],
        "objects": [f"{kind}/{name}" for kind, name, _uid in objects],
    }
    binding = data.get("binding")
    if isinstance(binding, dict):
        window = probe.ServiceWindow(binding)
        entry["window"] = {
            "key": window.key[:32],
            "service": window.service,
            "drop_in": str(window.dropin),
            "units": sorted(window.units.values()),
            "claim": str(window.targets["claim"]),
            "directory": str(window.directory),
        }
    return entry


def retire_stale_windows(
    directory: Path, node: str, *, settings: RetireSettings, by: str, via: str
) -> list[dict[str, Any]]:
    """``retire_journal`` for every journal ``directory`` holds open on ``node``."""

    return [
        retire_journal(path, node, settings=settings, by=by, via=via)
        for path in stale_journals(directory, node)
    ]


def retire_journal(
    journal: Path, node: str, *, settings: RetireSettings, by: str, via: str
) -> dict[str, Any]:
    """Retire one stale window, or name every proof that failed (nothing changed)."""

    entry: dict[str, Any] = {
        "journal": journal.name,
        "node": node,
        "retired": False,
        "refusals": [],
    }
    try:
        lock = _lock(journal.with_suffix(".lock"))
    except BlockingIOError:
        entry["refusals"].append("another process holds the journal")
        return entry
    try:
        _retire(journal, node, settings=settings, by=by, via=via, entry=entry)
    except (HostProbeError, RegionalFixtureError, OSError, ValueError) as exc:
        # An API, file or settings failure is no proof either way: nothing more
        # is done, the journal stays open and the next call resumes from what
        # holds.
        entry["refusals"].append(
            f"the retirement could not complete ({type(exc).__name__}: {exc})"
        )
    finally:
        os.close(lock)
    return entry


def _retire(
    journal: Path,
    node: str,
    *,
    settings: RetireSettings,
    by: str,
    via: str,
    entry: dict[str, Any],
) -> None:
    refusals: list[str] = entry["refusals"]
    try:
        data = probe.private_read(journal)
    except (OSError, ValueError, probe.ProbeError) as exc:
        refusals.append(f"the journal cannot be read ({type(exc).__name__})")
        return
    refusals.extend(journal_refusals(data, journal, node))
    if refusals:
        return
    scope, binding = data["scope"], data.get("binding")
    entry.update(
        run_id=scope["run_id"], release_id=scope.get("release_id"), phase=data["phase"]
    )
    if (scope["context"], scope["namespace"]) != (settings.context, settings.namespace):
        refusals.append("the journal belongs to another cluster context or namespace")
        return
    current = settings.node_uid(node)
    if current != scope["node_uid"]:
        refusals.append(
            f"{node} is gone or was replaced (Node UID {current or 'absent'}, the "
            f"journal's {scope['node_uid']}): nothing of the window can be proven"
        )
        return
    record_path = Path(scope["host_state_path"])
    try:
        record_lock = _lock(record_path.with_suffix(".lock"))
    except BlockingIOError:
        refusals.append("another process holds the earlier host probe")
        return
    try:
        record = _read_record(record_path, refusals)
        objects = _probe_objects(record_path, record, settings, refusals)
        if refusals:
            return
        stale = bool(
            record and record.get("closed") is False and record["script_may_exist"]
        )
        if binding is not None or stale:
            _host_retire(journal, data, record_path, record, stale, settings, entry)
            if refusals:
                return
        entry["host_probe_deleted"] = _delete(objects, settings, refusals)
        if refusals:
            return
        if record is not None and record.get("closed") is False:
            record.update(closed=True, script_may_exist=False)
            record["retired"] = {
                "at": _now(),
                "by": by,
                "via": via,
                "deleted": entry["host_probe_deleted"],
            }
            write_json_atomic(record_path, record)
    finally:
        os.close(record_lock)
    data["phase"] = "CLOSED"
    data["retirement"] = {
        "retired_at": _now(),
        "retired_by": by,
        "via": via,
        "phase_before": entry["phase"],
        "proofs": (entry.get("host") or {}).get("proofs"),
        "host_removed": (entry.get("host") or {}).get("removed", []),
        "host_probe_deleted": entry["host_probe_deleted"],
        "retire_probe_residuals": entry.get("retire_probe"),
    }
    write_json_atomic(journal, data)
    entry["retired"] = True


def _host_retire(
    journal: Path,
    data: dict[str, Any],
    record_path: Path,
    record: dict[str, Any] | None,
    stale: bool,
    settings: RetireSettings,
    entry: dict[str, Any],
) -> None:
    refusals: list[str] = entry["refusals"]
    binding = data.get("binding")
    directory = journal.parent / RETIRE_STATE
    fixture = settings.host_probe(
        HostProbeSettings(
            kubeconfig=settings.kubeconfig,
            context=settings.context,
            namespace=settings.namespace,
            node=data["scope"]["node"],
            image=settings.image,
            case_id=SERVICE_WINDOW_CASE,
            run_id=retire_run_id(data["scope"]["run_id"]),
            probe_script=write_retire_probe(directory),
            state_directory=directory,
            active_deadline_seconds=RETIRE_PROBE_DEADLINE_SECONDS,
        )
    )
    arguments = (
        ["retire", "--binding", json.dumps(binding, sort_keys=True)]
        if binding is not None
        else ["retire-script"]
    )
    if stale and record is not None:
        arguments += [
            "--stale-script",
            f"/run/{record_path.stem}.py",
            "--stale-script-sha256",
            str(record["scope"].get("script_sha256")),
        ]
    host: dict[str, Any] | None = None
    try:
        fixture.create()
        host = fixture.execute(*arguments, timeout=300)
    except (HostProbeError, RegionalFixtureError) as exc:
        refusals.append(
            f"the host retirement did not finish ({type(exc).__name__}: {exc})"
        )
    finally:
        try:
            residuals = fixture.cleanup()
        except (HostProbeError, RegionalFixtureError) as exc:
            residuals = None
            refusals.append(f"the retire probe was not cleaned up ({exc})")
    entry["host"] = host
    entry["retire_probe"] = residuals
    if isinstance(residuals, dict) and any(residuals.values()):
        refusals.append(
            f"the retire probe left residuals {sorted(k for k, v in residuals.items() if v)}"
        )
    if host is not None and host.get("retired") is not True:
        refusals.extend(
            host.get("refusals") or ["the host refused without naming a reason"]
        )


def _read_record(path: Path, refusals: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        refusals.append(
            f"the earlier host-probe record cannot be read ({type(exc).__name__})"
        )
        return None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("scope"), dict)
        or not isinstance(value.get("resources"), dict)
        or type(value.get("closed")) is not bool
        or type(value.get("script_may_exist")) is not bool
        or not RECORD_NAME.fullmatch(path.name)
    ):
        refusals.append("the earlier host-probe record is malformed")
        return None
    return value


def _probe_objects(
    record_path: Path,
    record: dict[str, Any] | None,
    settings: RetireSettings,
    refusals: list[str],
) -> list[tuple[str, str, str]]:
    """The earlier probe's Pod and ConfigMap still present, proven its own."""

    kubectl = _kubectl(settings)
    found: list[tuple[str, str, str]] = []
    for kind in ("pod", "configmap"):
        name = record_path.stem
        if record is not None:
            name = str((record["resources"].get(kind) or {}).get("name") or name)
        value = kubectl.read(kind, name)
        if value is None:
            continue
        if record is None:
            refusals.append(f"{kind}/{name} exists but its ownership record does not")
            continue
        problem = foreign_reason(value, kind, record)
        if problem:
            refusals.append(f"{kind}/{name} is not the earlier probe's own: {problem}")
            continue
        found.append((kind, name, str(value["metadata"]["uid"])))
    return found


def _delete(
    objects: list[tuple[str, str, str]], settings: RetireSettings, refusals: list[str]
) -> list[str]:
    kubectl = _kubectl(settings)
    deleted: list[str] = []
    for kind, name, uid in objects:
        kubectl.delete(kind, name, uid=uid)
        if not wait_gone(
            kubectl, kind, name, sleep=settings.sleep, clock=settings.clock
        ):
            refusals.append(f"{kind}/{name} deletion was not acknowledged")
            break
        deleted.append(f"{kind}/{name}")
    return deleted


def _kubectl(settings: RetireSettings) -> ProbeKubectl:
    run = settings.run if settings.run is not None else run_fixture_command
    return ProbeKubectl(settings.kubeconfig, settings.context, settings.namespace, run)


def _lock(path: Path) -> int:
    descriptor = os.open(
        path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "RETIRABLE_PHASES",
    "RetireSettings",
    "inspect_journal",
    "journal_refusals",
    "retire_journal",
    "retire_probe_source",
    "retire_stale_windows",
    "write_retire_probe",
]
