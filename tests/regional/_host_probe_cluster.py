"""An in-memory cluster for many host probes at once.

``_host_probe_support.ProbeApi`` keeps one Pod and one ConfigMap; a variant
that opens two probes on one node, or a sweep over several dead records, needs
objects by ``(kind, name)``. The transport answers the ``kubectl`` shapes
``host_probe_fixture`` and ``host_probe_sweep`` issue (``get``, ``create``,
``delete --raw`` with the UID precondition, ``wait``, ``exec``) and emulates
the probe's install step: a copy already at the probe's host path is accepted
only with the digest the call names or the stale digest it tolerates.
"""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from typing import Any
from urllib.parse import unquote

from scripts.e2e.regional.regional_commands import RegionalCommandFailed

TERMINAL = frozenset({"Failed", "Succeeded"})


class ProbeCluster:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.node_uid = "node-uid"
        self.serial = 0
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.probe_stdout = '{"ok": true}\n'
        self.probe_returncode = 0
        #: Host path -> digest of the script copy the install step left there.
        self.host_scripts: dict[str, str] = {}
        #: ``(kind, name)`` whose deletes are swallowed (a Pod stuck Terminating).
        self.stuck: set[tuple[str, str]] = set()
        #: The host-script removal fails (the cleanup aborts before any delete).
        self.cleanup_error = False

    # -- what a test reads back ------------------------------------------ #
    def pods(self) -> list[str]:
        return sorted(name for kind, name in self.objects if kind == "pod")

    def configmaps(self) -> list[str]:
        return sorted(name for kind, name in self.objects if kind == "configmap")

    def set_phase(self, pod: str, phase: str) -> None:
        self.objects[("pod", pod)]["status"]["phase"] = phase

    def exec_calls(self) -> list[list[str]]:
        return [arguments for arguments, _ in self.calls if arguments[0] == "exec"]

    # -- the transport --------------------------------------------------- #
    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(command[7:])
        self.calls.append((arguments, kwargs))
        verb = arguments[0]
        output, code = "", 0
        if verb == "get":
            kind, name = arguments[1], arguments[2]
            if kind == "node":
                value: dict[str, Any] | None = {
                    "metadata": {"name": name, "uid": self.node_uid}
                }
            else:
                value = self.objects.get((kind, name))
            output = json.dumps(value) if value is not None else ""
        elif verb == "create":
            value = json.loads(kwargs["input_text"])
            kind = value["kind"].lower()
            name = value["metadata"]["name"]
            if (kind, name) in self.objects:
                raise RegionalCommandFailed(1, "AlreadyExists")
            self.serial += 1
            value["metadata"]["uid"] = f"{kind}-{self.serial}"
            if kind == "pod":
                value["status"] = {
                    "phase": "Running",
                    "startTime": "2026-10-26T21:26:36Z",
                }
            self.objects[(kind, name)] = deepcopy(value)
            output = json.dumps(value)
        elif verb == "delete":
            options = json.loads(kwargs["input_text"])
            path = arguments[2]
            kind = "pod" if "/pods/" in path else "configmap"
            name = unquote(path.rsplit("/", 1)[1])
            current = self.objects.get((kind, name))
            if (
                current
                and options["preconditions"]["uid"] != current["metadata"]["uid"]
            ):
                raise RegionalCommandFailed(1, "UID conflict")
            if (kind, name) not in self.stuck:
                self.objects.pop((kind, name), None)
        elif verb == "exec":
            output, code = self._exec(arguments)
        elif verb != "wait":
            raise AssertionError(f"unexpected fake API operation: {verb}")
        return subprocess.CompletedProcess(command, code, output, "")

    def _exec(self, arguments: list[str]) -> tuple[str, int]:
        pod = self.objects.get(("pod", arguments[1]))
        if pod is None:
            raise RegionalCommandFailed(1, "NotFound")
        if pod["status"]["phase"] in TERMINAL:
            raise RegionalCommandFailed(1, "cannot exec into a terminated container")
        if "host-probe-cleanup" in arguments:
            if self.cleanup_error:
                raise RegionalCommandFailed(1, "host cleanup failed")
            host_path = arguments[arguments.index("host-probe-cleanup") + 1]
            self.host_scripts.pop(host_path, None)
            return "", 0
        index = arguments.index("host-probe")
        host_path, digest, stale = arguments[index + 2 : index + 5]
        existing = self.host_scripts.get(host_path)
        if existing is not None and existing not in (digest, stale):
            return '{"error": "host probe setup rejected"}\n', 1
        self.host_scripts[host_path] = digest
        return self.probe_stdout, self.probe_returncode
