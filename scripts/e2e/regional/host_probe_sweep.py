"""Close the ownership records of dead host probes before a variant starts.

A host probe (``host_probe_fixture``) keeps one ownership record per Pod under
``<case dir>/host-probes/<pod>.json`` and closes it in ``cleanup``. A record
still open when the next run of the case starts belongs to a run that never
reached that cleanup. Live 2026-10-26, both clusters: the GF-GENERAL-LIVE 024
composites closed their environment replay and discarded its undo without the
replay's probe ever being the context's to tear down, and 022
``recovery_evidence`` died of ENOSPC on the runner host in the middle of its
teardown. Those Pods ran to ``activeDeadlineSeconds`` (exit 143) and stayed
Failed with their ConfigMaps; a rerun of the case in the same run directory
derives the same probe name (``run_identity`` is stable), met the open record
-- carrying the earlier probe script's digest -- and was refused with
``host probe ownership record does not match this run``.

``sweep_dead_host_probe_records`` runs at variant start over that directory.
A record whose Pod is gone or terminal (Failed/Succeeded) is a dead probe's:
the Pod and ConfigMap it names are deleted under the same proof the fixture
itself applies (case/run labels, owner annotation, script digest, node, UID
precondition) and the record is closed with a ``swept`` note. A record whose
Pod is still running is left exactly as it is and reported -- that is a live
probe (another process, or this run's earlier attempt), and the fixture's own
refusal stands. The script copy a dead probe installed under ``/run`` cannot
be removed without a Pod on the node; the record keeps its digest as
``stale_host_script_sha256`` so the next probe of the same identity overwrites
that one copy instead of refusing it (``HostProbeFixture.execute``).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.host_probe_fixture import (
    STALE_HOST_SCRIPT_KEY,
    SWEPT_KEY,
    HostProbeError,
)
from scripts.e2e.regional.regional_commands import run_fixture_command

#: Pod phases of a probe that will never answer another ``kubectl exec``.
TERMINAL_PHASES = frozenset({"Failed", "Succeeded"})
#: How long a dead Pod's delete may take to be observed (it has no work left).
POD_GONE_WAIT_SECONDS = 60
#: The evidence file ``run_general_live`` writes the report under.
SWEEP_EVIDENCE = "host-probe-sweep"


def sweep_dead_host_probe_records(
    state_directory: Path,
    *,
    kubeconfig: Path,
    context: str,
    namespace: str,
    run: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Close every open record of ``state_directory`` whose Pod is dead.

    Returns the report: ``swept`` (records closed, what was deleted), ``live``
    (open records whose Pod still runs; untouched), ``skipped`` (records this
    sweep would not touch, each with its reason) and the counts. ``run`` is
    the command runner (``run_fixture_command`` unless a test hands in its API).
    """

    runner = run if run is not None else run_fixture_command
    report: dict[str, Any] = {
        "directory": str(state_directory),
        "records": 0,
        "open": 0,
        "swept": [],
        "live": [],
        "skipped": [],
    }
    if not state_directory.is_dir():
        return report
    for path in sorted(state_directory.glob("*.json")):
        report["records"] += 1
        try:
            record = _load(path)
        except (OSError, ValueError) as exc:
            report["skipped"].append(
                {"record": path.stem, "reason": f"unreadable: {type(exc).__name__}"}
            )
            continue
        if not isinstance(record, dict) or record.get("closed") is not False:
            continue
        report["open"] += 1
        entry = _sweep_record(
            path,
            kubeconfig=kubeconfig,
            context=context,
            namespace=namespace,
            run=runner,
            sleep=sleep,
            clock=clock,
        )
        report[entry.pop("outcome")].append(entry)
    return report


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {"outcome": "skipped", "record": name, "reason": reason}


def _sweep_record(
    path: Path,
    *,
    kubeconfig: Path,
    context: str,
    namespace: str,
    run: Callable[..., Any],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    name = path.stem
    descriptor = os.open(
        path.with_suffix(".lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _skipped(name, "another process owns this host probe")
        # Re-read under the lock: the owner may have closed it meanwhile.
        record = _load(path)
        if not isinstance(record, dict) or record.get("closed") is not False:
            return _skipped(name, "closed by its owner meanwhile")
        scope = record.get("scope")
        resources = record.get("resources")
        if not (
            isinstance(scope, dict)
            and isinstance(resources, dict)
            and isinstance(record.get("owner"), str)
            and record["owner"]
        ):
            return _skipped(name, "malformed record")
        if scope.get("context") != context or scope.get("namespace") != namespace:
            return _skipped(
                name, "record belongs to another cluster context or namespace"
            )
        pod_name = (resources.get("pod") or {}).get("name")
        configmap_name = (resources.get("configmap") or {}).get("name")
        if not (
            isinstance(pod_name, str)
            and pod_name
            and isinstance(configmap_name, str)
            and configmap_name
        ):
            return _skipped(name, "malformed record")
        kubectl = _Kubectl(kubeconfig, context, namespace, run)
        pod = kubectl.read("pod", pod_name)
        phase = (pod.get("status") or {}).get("phase") if pod is not None else None
        if pod is not None and phase not in TERMINAL_PHASES:
            return {
                "outcome": "live",
                "record": name,
                "pod": pod_name,
                "phase": phase,
                "node": scope.get("node"),
                "started_at": (pod.get("status") or {}).get("startTime"),
            }
        deleted: list[str] = []
        if pod is not None:
            problem = foreign_reason(pod, "pod", record)
            if problem:
                return _skipped(name, f"terminal Pod is not this record's: {problem}")
            kubectl.delete("pod", pod_name, uid=pod["metadata"]["uid"])
            if not _wait_gone(kubectl, "pod", pod_name, sleep=sleep, clock=clock):
                return _skipped(name, "Pod deletion was not acknowledged")
            deleted.append(f"pod/{pod_name}")
        configmap = kubectl.read("configmap", configmap_name)
        if configmap is not None:
            problem = foreign_reason(configmap, "configmap", record)
            if problem:
                return _skipped(name, f"ConfigMap is not this record's: {problem}")
            kubectl.delete(
                "configmap", configmap_name, uid=configmap["metadata"]["uid"]
            )
            if kubectl.read("configmap", configmap_name) is not None:
                return _skipped(name, "ConfigMap deletion was not acknowledged")
            deleted.append(f"configmap/{configmap_name}")
        stale = scope.get("script_sha256") if record.get("script_may_exist") else None
        record["closed"] = True
        record["script_may_exist"] = False
        if isinstance(stale, str) and stale:
            record[STALE_HOST_SCRIPT_KEY] = stale
        record[SWEPT_KEY] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pod_phase": phase or "absent",
            "deleted": deleted,
            "host_script_left_sha256": stale if isinstance(stale, str) else None,
        }
        write_json_atomic(path, record)
        return {
            "outcome": "swept",
            "record": name,
            "pod": pod_name,
            "pod_phase": phase or "absent",
            "node": scope.get("node"),
            "deleted": deleted,
            "host_script_left": bool(stale),
        }
    finally:
        os.close(descriptor)


def foreign_reason(value: dict[str, Any], kind: str, record: dict[str, Any]) -> str:
    """Why ``value`` is not the resource ``record`` owns; ``""`` when it is.

    The same facts ``HostProbeFixture._owned`` checks, read from the record
    instead of a live fixture: the case/run labels, the owner annotation, the
    script digest, the recorded UID and -- for the Pod -- the node and its UID.
    """

    scope = record["scope"]
    metadata = value.get("metadata") or {}
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    uid = metadata.get("uid")
    if not isinstance(uid, str) or not uid:
        return "no UID"
    if labels.get("gpu-fault.io/acceptance-case") != scope.get("case_id") or labels.get(
        "gpu-fault.io/acceptance-run"
    ) != scope.get("run_id"):
        return "case/run labels differ"
    if annotations.get("gpu-fault.io/probe-owner") != record.get("owner"):
        return "owner annotation differs"
    if annotations.get("gpu-fault.io/probe-script-sha256") != scope.get(
        "script_sha256"
    ):
        return "script digest differs"
    recorded = (record["resources"].get(kind) or {}).get("uid")
    if recorded is not None and recorded != uid:
        return "UID differs from the record"
    if kind == "pod":
        if (value.get("spec") or {}).get("nodeName") != scope.get("node"):
            return "node differs"
        if annotations.get("gpu-fault.io/probe-node-uid") != record.get("node_uid"):
            return "node UID differs"
    return ""


def _wait_gone(
    kubectl: _Kubectl,
    kind: str,
    name: str,
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> bool:
    deadline = clock() + POD_GONE_WAIT_SECONDS
    while kubectl.read(kind, name) is not None:
        if clock() >= deadline:
            return False
        sleep(2)
    return True


class _Kubectl:
    """The two API operations the sweep needs, in the fixture's own shape."""

    def __init__(
        self, kubeconfig: Path, context: str, namespace: str, run: Callable[..., Any]
    ) -> None:
        self._prefix = [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "--context",
            context,
            "-n",
            namespace,
        ]
        self._namespace = namespace
        self._run = run

    def read(self, kind: str, name: str) -> dict[str, Any] | None:
        result = self._run(
            [*self._prefix, "get", kind, name, "--ignore-not-found", "-o", "json"],
            timeout=60,
        )
        if not result.stdout.strip():
            return None
        value = json.loads(result.stdout)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("metadata"), dict)
            or value["metadata"].get("name") != name
        ):
            raise HostProbeError("host probe resource read is malformed")
        return value

    def delete(self, kind: str, name: str, *, uid: str) -> None:
        plural = {"pod": "pods", "configmap": "configmaps"}[kind]
        path = (
            f"/api/v1/namespaces/{quote(self._namespace, safe='')}/"
            f"{plural}/{quote(name, safe='')}"
        )
        options = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": uid},
            "propagationPolicy": "Foreground",
            "gracePeriodSeconds": 0,
        }
        self._run(
            [*self._prefix, "delete", "--raw", path, "-f", "-"],
            input_text=json.dumps(options),
            check=False,
            timeout=60,
        )


__all__ = [
    "POD_GONE_WAIT_SECONDS",
    "SWEEP_EVIDENCE",
    "TERMINAL_PHASES",
    "foreign_reason",
    "sweep_dead_host_probe_records",
]
