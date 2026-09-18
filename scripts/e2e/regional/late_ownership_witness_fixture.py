"""An owned witness Pod whose main process survives kubelet quiesce."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    NodeIdentity,
    ProcessIdentity,
)
from scripts.e2e.regional.late_ownership_probe_bundle import probe_program, stdin_loader
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from scripts.e2e.regional.probes.late_ownership_node_probe import mailbox_path


class WitnessPod(HostProbeFixture):
    def __init__(self, settings: HostProbeSettings, *, control_path: Path) -> None:
        super().__init__(settings)
        self.control_path = control_path

    def manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        configmap, pod = super().manifests()
        pod["spec"]["automountServiceAccountToken"] = False
        container = pod["spec"]["containers"][0]
        container["args"] = [
            'exec chroot /host /opt/gpu-fault/current/venv/bin/python -I -u -c "$(cat /probe/probe.py)"'
        ]
        container["env"] = [
            {
                "name": "LATE_OWNERSHIP_WITNESS_POD_UID",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
            }
        ]
        container["readinessProbe"] = {
            "exec": {"command": ["test", "-S", "/host" + str(self.control_path)]},
            "periodSeconds": 1,
            "failureThreshold": 60,
        }
        return configmap, pod


class NodeWitness:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        original: HostProbeSettings,
        scope: AcceptanceScope,
        node: NodeIdentity,
        directory: Path,
    ) -> None:
        self.regional = regional
        self.scope = scope
        self.node = node
        program, self.bundle_sha256 = probe_program("node")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        source = directory / f"late-ownership-witness-{node.name}.py"
        initial = json.dumps(
            {"scope": scope.model_dump(mode="json"), "node": node.name, "daemon": True}
        )
        descriptor = os.open(
            source, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(
                "import io, sys\nsys.stdin = io.StringIO("
                + repr(initial + "\n")
                + ")\n"
                + program
            )
        self.probe = WitnessPod(
            replace(
                original,
                probe_script=source,
                state_directory=directory / "ownership",
                case_id=scope.case_id,
            ),
            control_path=mailbox_path(scope, node) / "control.sock",
        )
        self.peer: dict[str, Any] | None = None

    def create(self) -> None:
        self.probe.create()

    def request(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.probe._check_target()
        program, digest = probe_program("node")
        if digest != self.bundle_sha256:
            raise BoundaryDenied("witness source changed during the live experiment")
        output = self.regional.kubectl(
            "gpu",
            "exec",
            "-i",
            self.probe.pod,
            "-c",
            "probe",
            "--",
            "chroot",
            "/host",
            "/opt/gpu-fault/current/venv/bin/python",
            "-I",
            "-u",
            "-c",
            stdin_loader(program),
            input_text=program
            + json.dumps(
                {
                    "scope": self.scope.model_dump(mode="json"),
                    "node": self.node.name,
                    "action": action,
                    "payload": payload,
                }
            )
            + "\n",
            timeout=45,
        )
        self.probe._check_target()
        value = json.loads(output.splitlines()[-1])
        if (
            not isinstance(value, dict)
            or value.get("scope_sha256") != self.scope.digest()
        ):
            raise BoundaryDenied("node witness response is outside the owned scope")
        peer = value.get("peer")
        if not isinstance(peer, dict) or (self.peer is not None and peer != self.peer):
            raise BoundaryDenied("node witness daemon was replaced")
        identity = ProcessIdentity.model_validate(peer)
        if identity.boot_id != self.node.boot_id or (
            value.get("producer") is not None and value["producer"] != peer
        ):
            raise BoundaryDenied("node witness response came from a different process")
        for key in ("receipt", "observation"):
            if key in value and value[key].get("producer") != peer:
                raise BoundaryDenied("node witness receipt producer is unbound")
        self.peer = peer
        return value

    def cleanup(self, transport: HostProbeFixture) -> None:
        if not self.probe.state_path.exists():
            if any(self.probe.cleanup().values()):
                raise BoundaryDenied("unacknowledged witness resources remain")
            return
        record = json.loads(self.probe.state_path.read_text())
        pod_uid = record["resources"]["pod"]["uid"]
        if any(self.probe.cleanup().values()):
            raise BoundaryDenied("node witness Pod cleanup left owned resources")
        if pod_uid is None:
            if record["resources"]["pod"]["create_started"]:
                raise BoundaryDenied("witness Pod creation is unresolved")
            return
        transport._check_target()
        program, _ = probe_program("node")
        output = self.regional.kubectl(
            "gpu",
            "exec",
            "-i",
            transport.pod,
            "-c",
            "probe",
            "--",
            "chroot",
            "/host",
            "/opt/gpu-fault/current/venv/bin/python",
            "-I",
            "-u",
            "-c",
            stdin_loader(program),
            input_text=program
            + json.dumps(
                {
                    "scope": self.scope.model_dump(mode="json"),
                    "node": self.node.name,
                    "action": "cleanup-mailbox",
                    "payload": {"pod_uid": pod_uid},
                }
            )
            + "\n",
            timeout=45,
        )
        transport._check_target()
        value = json.loads(output.splitlines()[-1])
        if (
            value.get("scope_sha256") != self.scope.digest()
            or value.get("mailbox_absent") is not True
        ):
            raise BoundaryDenied("node witness mailbox cleanup is unconfirmed")
