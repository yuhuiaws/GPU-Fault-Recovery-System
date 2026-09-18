"""Private controller intent and cleanup-only resumption for service windows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError

T = TypeVar("T")
PHASES = {
    "CREATING",
    "CREATED",
    "ARMING",
    "STOP_REQUESTED",
    "CLEANUP_REQUIRED",
    "CLOSED",
}


def require_report(
    value: dict[str, Any], binding: dict[str, Any], phases: set[str]
) -> None:
    window = probe.ServiceWindow(binding)
    if (
        not isinstance(value, dict)
        or value.get("binding_sha256") != window.key
        or value.get("phase") not in phases
        or value.get("service") != binding["service"]
        or value.get("restore_at") != binding["restore_at"]
        or value.get("expires_at") != binding["expires_at"]
        or value.get("restore_unit") != window.units["restore-service"]
        or value.get("stop_unit") != window.units["stop-timer"]
        or value.get("stop_service") != window.units["stop-service"]
        or type(value.get("stop_requested")) is not bool
        or type(value.get("stop_acknowledged")) is not bool
        or type(value.get("start_intent")) is not bool
        or value.get("record_kind")
        != ("TOMBSTONE" if value.get("phase") == "CLOSED" else "RECOVERY_REQUIRED")
    ):
        raise RegionalFixtureError("service window ACK is unbound or incomplete")
    if value["phase"] in {"RESTORED", "CLOSED"}:
        after = value.get("after")
        quiet = value.get("stop_quiescence")
        if (
            not isinstance(after, dict)
            or (value["stop_requested"] and not value["stop_acknowledged"])
            or not probe.STATE_FIELDS <= after.keys()
            or after["Id"] != binding["service"]
            or not probe.running(after)
            or not isinstance(quiet, dict)
            or set(quiet) != {"timer", "service"}
        ):
            raise RegionalFixtureError("service restoration proof is incomplete")
        required = [
            (quiet["timer"], window.units["stop-timer"]),
            (quiet["service"], window.units["stop-service"]),
        ]
        if value["phase"] == "CLOSED":
            required.append(
                (value.get("restore_quiescence"), window.units["restore-service"])
            )
        for observed, name in required:
            if (
                not isinstance(observed, dict)
                or observed.get("unit") != name
                or observed.get("load_state") not in {"loaded", "not-found"}
                or observed.get("active_state") not in {"inactive", "failed"}
                or any(
                    type(observed.get(k)) is not int or observed[k] != 0
                    for k in ("job_id", "main_pid", "control_pid")
                )
                or observed.get("cgroup_empty") is not True
            ):
                raise RegionalFixtureError(
                    "service stop/process quiescence is unproven"
                )


class ServiceWindowController:
    def __init__(
        self,
        host: HostProbeFixture,
        *,
        cluster_id: str,
        node_uid: str,
        plan_sha256: str,
        release_id: str,
        maintenance_expires_at: datetime,
        read_node_uid: Callable[[], str],
    ) -> None:
        self.host = host
        self.read_node_uid = read_node_uid
        if (
            maintenance_expires_at.tzinfo is None
            or maintenance_expires_at.utcoffset() is None
            or not math.isfinite(maintenance_expires_at.timestamp())
        ):
            raise RegionalFixtureError(
                "service maintenance expiry must be timezone-aware"
            )
        for key, value in (
            ("cluster ID", cluster_id),
            ("Node UID", node_uid),
            ("release ID", release_id),
        ):
            probe.safe_id(value, key)
        if not probe.HEX.fullmatch(plan_sha256):
            raise RegionalFixtureError(
                "service window requires an approved plan digest"
            )
        settings = host.settings
        self.host_scope = {
            "kubeconfig": str(settings.kubeconfig.resolve()),
            "context": settings.context,
            "namespace": settings.namespace,
            "node": settings.node,
            "image": settings.image,
            "case_id": settings.case_id,
            "run_id": settings.run_id,
            "script_sha256": hashlib.sha256(
                settings.probe_script.read_bytes()
            ).hexdigest(),
        }
        self.scope: dict[str, Any] = {
            **self.host_scope,
            "cluster_id": cluster_id,
            "node_uid": node_uid,
            "plan_sha256": plan_sha256,
            "release_id": release_id,
            "maintenance_expires_at": int(maintenance_expires_at.timestamp()),
            "kubeconfig_sha256": hashlib.sha256(
                settings.kubeconfig.read_bytes()
            ).hexdigest(),
            "controller_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "host_state_path": str(host.state_path.resolve()),
        }
        # Mutable source/config/deadline inputs must not select a fresh journal.
        key = probe.digest([settings.case_id, settings.run_id, settings.node])[:24]
        self.path = settings.state_directory / f"destr008-service-window-{key}.json"
        self.data: dict[str, Any] = {}
        self.fd: int | None = None
        self.resumed = False

    def acquire(self, *, existing: bool = True) -> None:
        if self.fd is not None:
            if self.data.get("supervision_lost"):
                raise RegionalFixtureError(
                    "service window supervision was lost; review required"
                )
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe.require_directory(self.path.parent, private=True)
        fd = os.open(
            self.path.with_suffix(".lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise RegionalFixtureError("service controller lock is not private")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists() or self.path.is_symlink():
                data = probe.private_read(self.path)
                if (
                    data.get("schema_version") != 1
                    or data.get("scope") != self.scope
                    or not isinstance(data.get("phase"), str)
                    or data.get("phase") not in PHASES
                    or not isinstance(data.get("owner"), str)
                    or not re.fullmatch(r"[0-9a-f]{32}", data["owner"])
                    or type(data.get("supervision_lost")) is not bool
                ):
                    raise RegionalFixtureError(
                        "service controller identity or state changed"
                    )
                if data["supervision_lost"]:
                    raise RegionalFixtureError(
                        "service window supervision was lost; review required"
                    )
                self.data = data
                self.resumed = True
            elif existing:
                raise RegionalFixtureError(
                    "service cleanup requires its controller journal"
                )
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def save(self) -> None:
        if self.fd is None:
            raise RegionalFixtureError("service controller journal is not locked")
        write_json_atomic(self.path, self.data)
        probe.sync_directory(self.path.parent)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def invoke(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except ProcessSupervisionLost:
            self.data["supervision_lost"] = True
            self.save()
            raise
        except Exception as exc:
            self.data["last_failure"] = type(exc).__name__
            self.save()
            raise

    def host_proof(self, *, active: bool, closed: bool = False) -> dict[str, Any]:
        value = probe.private_read(self.host.state_path)
        if (
            value.get("schema_version") != 1
            or value.get("scope") != self.host_scope
            or value.get("node_uid") != self.scope["node_uid"]
            or not isinstance(value.get("owner"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", value["owner"])
            or type(value.get("closed")) is not bool
            or type(value.get("script_may_exist")) is not bool
            or (active and value["closed"])
            or (closed and not value["closed"])
            or not isinstance(value.get("resources"), dict)
            or set(value["resources"]) != {"pod", "configmap"}
        ):
            raise RegionalFixtureError("service host transport ownership is unproven")
        for kind, name in (("pod", self.host.pod), ("configmap", self.host.configmap)):
            item = value["resources"][kind]
            if (
                not isinstance(item, dict)
                or item.get("name") != name
                or type(item.get("create_started")) is not bool
                or (
                    active
                    and (
                        item["create_started"] is not True
                        or not isinstance(item.get("uid"), str)
                        or not item["uid"]
                    )
                )
            ):
                raise RegionalFixtureError(
                    "service host resource receipt is incomplete"
                )
        proof = {
            "owner": value["owner"],
            "scope": value["scope"],
            "node_uid": value["node_uid"],
        }
        previous = self.data.get("host_proof")
        if previous is not None and proof != previous:
            raise RegionalFixtureError("service host transport owner changed")
        return proof

    @property
    def binding(self) -> dict[str, Any]:
        value = self.data.get("binding")
        if not isinstance(value, dict):
            raise RegionalFixtureError("service window has no durable host binding")
        probe.binding_key(value)
        for key in (
            "owner",
            "case_id",
            "run_id",
            "cluster_id",
            "node",
            "node_uid",
            "release_id",
            "plan_sha256",
        ):
            expected = self.data["owner"] if key == "owner" else self.scope[key]
            if value[key] != expected:
                raise RegionalFixtureError("service host/controller binding changed")
        if (
            value["helper_sha256"] != self.scope["script_sha256"]
            or value["expires_at"] > self.scope["maintenance_expires_at"]
        ):
            raise RegionalFixtureError("service source or absolute expiry changed")
        return value

    @property
    def service(self) -> str:
        return str(self.binding["service"]) if "binding" in self.data else ""

    @property
    def failsafe_at(self) -> datetime | None:
        if "binding" not in self.data:
            return None
        return datetime.fromtimestamp(self.binding["restore_at"], timezone.utc)

    def create(self) -> None:
        self.acquire(existing=False)
        if self.data:
            raise RegionalFixtureError(
                "service fixture already exists; resume cleanup only"
            )
        if self.read_node_uid() != self.scope["node_uid"]:
            raise RegionalFixtureError("approved service Node UID changed")
        if self.scope["maintenance_expires_at"] <= time.time():
            raise RegionalFixtureError("service maintenance window has expired")
        if self.host.state_path.exists() or self.host.state_path.is_symlink():
            raise RegionalFixtureError(
                "pre-existing host transport has no controller proof"
            )
        self.data = {
            "schema_version": 1,
            "scope": self.scope,
            "owner": uuid4().hex,
            "phase": "CREATING",
            "supervision_lost": False,
        }
        self.save()
        self.invoke(self.host.create)
        self.data["host_proof"] = self.host_proof(active=True)
        self.data["phase"] = "CREATED"
        self.save()

    def request(self, command: str, phases: set[str]) -> dict[str, Any]:
        binding = self.binding
        if "host_proof" not in self.data:
            raise RegionalFixtureError(
                "service request has no durable host transport proof"
            )
        self.host_proof(active=True)
        self.data["last_request"] = command
        self.save()
        result = self.invoke(
            lambda: self.host.execute(
                command, "--binding", json.dumps(binding, sort_keys=True), timeout=180
            )
        )
        require_report(result, binding, phases)
        self.data["host_ack"] = result
        self.save()
        return result

    def stop(
        self, service: str, *, restore_seconds: int = 180, delay_seconds: int = 0
    ) -> dict[str, Any]:
        self.acquire()
        if self.resumed or self.data["phase"] != "CREATED" or "binding" in self.data:
            raise RegionalFixtureError(
                "service stop is create-once; resume cleanup only"
            )
        probe.service_name(service)
        if type(restore_seconds) is not int or not 60 <= restore_seconds <= 600:
            raise RegionalFixtureError("service restore seconds is outside 60..600")
        if type(delay_seconds) is not int or not 0 <= delay_seconds <= 120:
            raise RegionalFixtureError("service stop delay is outside 0..120")
        self.host_proof(active=True)
        snapshot = self.invoke(
            lambda: self.host.execute("snapshot", "--service", service)
        )
        pins = snapshot.get("pins")
        if (
            snapshot.get("service") != service
            or not isinstance(pins, dict)
            or any(
                pins.get(k) != self.scope[k] for k in ("cluster_id", "node", "node_uid")
            )
            or not isinstance(snapshot.get("state"), dict)
            or not probe.STATE_FIELDS <= snapshot["state"].keys()
            or not probe.running(snapshot["state"])
        ):
            raise RegionalFixtureError(
                "service baseline does not match the approved target"
            )
        now = int(time.time())
        binding = {
            **{
                k: self.scope[k]
                for k in (
                    "case_id",
                    "run_id",
                    "cluster_id",
                    "node",
                    "node_uid",
                    "release_id",
                    "plan_sha256",
                )
            },
            "owner": self.data["owner"],
            "helper_sha256": self.scope["script_sha256"],
            **{
                k: snapshot.get(k)
                for k in ("baseline_sha256", "host_sha256", "boot_id")
            },
            "service": service,
            "restore_at": now + delay_seconds + restore_seconds,
            "expires_at": now
            + delay_seconds
            + restore_seconds
            + probe.RECOVERY_SECONDS,
            "stop_delay_seconds": delay_seconds,
        }
        probe.binding_key(binding)
        if binding["expires_at"] > self.scope["maintenance_expires_at"]:
            raise RegionalFixtureError(
                "service recovery exceeds the approved maintenance window"
            )
        self.data.update(binding=binding, phase="ARMING")
        self.save()  # Prepare may be accepted even if its transport reply is lost.
        self.request("prepare", {"INSTALLED"})
        deadline = time.monotonic() + probe.ARM_SECONDS
        while True:
            report = self.request("status", {"INSTALLED", "ARMED"})
            if report["phase"] == "ARMED":
                ack = report.get("ack")
                if (
                    not isinstance(ack, dict)
                    or ack.get("boot_id") != binding["boot_id"]
                    or not isinstance(ack.get("invocation_id"), str)
                    or not re.fullmatch(r"[0-9a-f]{32}", ack["invocation_id"])
                    or type(ack.get("pid")) is not int
                    or ack["pid"] <= 0
                    or type(ack.get("at")) not in {int, float}
                    or not math.isfinite(ack["at"])
                    or not 0 <= time.time() - ack["at"] <= probe.ACK_SECONDS
                ):
                    raise RegionalFixtureError(
                        "service independent ACK is stale or malformed"
                    )
                break
            if time.monotonic() >= deadline:
                raise RegionalFixtureError(
                    "service independent arm ACK was not observed"
                )
            time.sleep(1)
        self.data["phase"] = "STOP_REQUESTED"
        self.save()
        return self.request("stop-with-failsafe", {"SCHEDULED"})

    def restore(self) -> dict[str, Any]:
        self.acquire()
        if self.data["phase"] == "CLOSED":
            raise RegionalFixtureError("service window is closed; resume cleanup only")
        if "binding" not in self.data:
            return {"phase": "NOT_STOPPED", "reason": "no service window was requested"}
        self.data["phase"] = "CLEANUP_REQUIRED"
        self.save()
        return self.request("restore-service", {"RESTORED", "CLOSED"})

    def verify_residuals(self, residuals: dict[str, bool]) -> None:
        required = {f"pod/{self.host.pod}", f"configmap/{self.host.configmap}"}
        if (
            not isinstance(residuals, dict)
            or not required <= residuals.keys()
            or any(type(v) is not bool or v for v in residuals.values())
        ):
            raise RegionalFixtureError(
                "service probe cleanup has unknown or remaining resources"
            )

    def cleanup(self) -> dict[str, bool]:
        self.acquire()
        if self.data["phase"] == "CLOSED":
            if "binding" in self.data:
                require_report(
                    self.data.get("host_cleanup", {}), self.binding, {"CLOSED"}
                )
            if "host_proof" in self.data:
                self.host_proof(active=False, closed=True)
            elif self.host.state_path.exists() or self.host.state_path.is_symlink():
                raise RegionalFixtureError(
                    "closed service fixture found an unrecorded host owner"
                )
            residuals = self.invoke(self.host.residuals)
            self.verify_residuals(residuals)
            self.close()
            return residuals
        self.data["phase"] = "CLEANUP_REQUIRED"
        self.save()
        if "binding" in self.data:
            host_record = probe.private_read(self.host.state_path)
            if "host_cleanup" not in self.data or host_record.get("closed") is not True:
                self.data["host_cleanup"] = self.request("cleanup", {"CLOSED"})
                self.save()
            else:
                require_report(self.data["host_cleanup"], self.binding, {"CLOSED"})
        # With no binding, no service mutation could have been authorized. A
        # partial create is cleaned by HostProbe's own recorded resource receipts.
        if self.host.state_path.exists() or self.host.state_path.is_symlink():
            self.data["host_proof"] = self.host_proof(active=False)
            self.save()
        elif "host_proof" in self.data or "binding" in self.data:
            raise RegionalFixtureError("service host ownership journal disappeared")
        residuals = self.invoke(self.host.cleanup)
        self.verify_residuals(residuals)
        if self.host.state_path.exists():
            self.host_proof(active=False, closed=True)
        self.data.update(phase="CLOSED", probe_residuals=residuals)
        self.save()
        self.close()
        return residuals

    def resume_cleanup(self) -> dict[str, Any]:
        self.acquire()
        self.resumed = True
        residuals = self.cleanup()
        return {
            "phase": "CLOSED",
            "resumed": True,
            "host": self.data.get("host_cleanup", {"phase": "NOT_REQUESTED"}),
            "probe_residuals": residuals,
        }
