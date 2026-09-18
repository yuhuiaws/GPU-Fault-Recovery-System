from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
from scripts.e2e.regional.regional_commands import RegionalCommandFailed


def host_probe(tmp_path: Path, **overrides: Any) -> HostProbeFixture:
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
    settings = {
        "kubeconfig": kubeconfig,
        "context": "gpu-context",
        "namespace": "gpu-fault-system",
        "node": "node-a",
        "image": "registry.example/probe@sha256:" + "a" * 64,
        "case_id": "GF-REGIONAL-DESTR-001",
        "run_id": "run-a",
        "probe_script": probe,
        "state_directory": tmp_path / "host-probes",
        **overrides,
    }
    return HostProbeFixture(HostProbeSettings(**settings))


class ProbeApi:
    """In-memory API transport with UID-enforced deletion and lost-ACK injection."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.node_uid = "node-uid"
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.serial = 0
        self.read_error = False
        self.cleanup_error = False
        self.stuck_pod = False
        self.lost_ack: str | None = None
        self.reject_create = False
        self.probe_stdout = '{"ok": true}\n'
        self.probe_stderr = ""
        self.probe_returncode = 0
        self.after_execute: Any = None
        self.admit_sidecar = False
        self.exec_containers: list[str] = []
        self.host_script_exists = False

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[7:]
        self.calls.append((arguments, kwargs))
        verb = arguments[0]
        output, code = "", 0
        if verb == "get":
            if self.read_error:
                raise RegionalCommandFailed(1, "Forbidden")
            kind, name = arguments[1:3]
            value = (
                {"metadata": {"name": name, "uid": self.node_uid}}
                if kind == "node"
                else self.objects.get(kind)
            )
            output = json.dumps(value) if value is not None else ""
        elif verb == "create":
            value = json.loads(kwargs["input_text"])
            kind = value["kind"].lower()
            if kind in self.objects:
                raise RegionalCommandFailed(1, "AlreadyExists")
            if self.reject_create:
                raise RegionalCommandFailed(1, "create failed before acknowledgement")
            self.serial += 1
            value["metadata"]["uid"] = f"{kind}-{self.serial}"
            if kind == "pod":
                value["status"] = {"phase": "Running"}
                if self.admit_sidecar:
                    value["spec"]["containers"].append(
                        {"name": "sidecar", "image": "registry.example/sidecar:unit"}
                    )
            self.objects[kind] = deepcopy(value)
            if self.lost_ack == kind:
                self.lost_ack = None
                raise RegionalCommandFailed(1, "response lost")
            output = json.dumps(value)
        elif verb == "delete":
            options = json.loads(kwargs["input_text"])
            kind = "pod" if "/pods/" in arguments[2] else "configmap"
            current = self.objects.get(kind)
            if (
                current
                and options["preconditions"]["uid"] != current["metadata"]["uid"]
            ):
                raise RegionalCommandFailed(1, "UID conflict")
            if not (
                self.stuck_pod
                and kind == "pod"
                and options.get("gracePeriodSeconds") != 0
            ):
                self.objects.pop(kind, None)
        elif verb == "exec":
            pod = self.objects["pod"]
            options = arguments[: arguments.index("--")]
            container = (
                options[options.index("-c") + 1]
                if "-c" in options
                else pod["metadata"]["annotations"].get(
                    "kubectl.kubernetes.io/default-container",
                    pod["spec"]["containers"][0]["name"],
                )
            )
            self.exec_containers.append(container)
            if "host-probe-cleanup" in arguments:
                if self.cleanup_error:
                    raise RegionalCommandFailed(1, "host cleanup failed")
                if container == "probe":
                    self.host_script_exists = False
            else:
                if container != "probe":
                    raise RegionalCommandFailed(1, "probe mount is unavailable")
                self.host_script_exists = True
                output, code = self.probe_stdout, self.probe_returncode
                if self.after_execute is not None:
                    self.after_execute()
        elif verb != "wait":
            raise AssertionError(f"unexpected fake API operation: {verb}")
        return subprocess.CompletedProcess(command, code, output, self.probe_stderr)
