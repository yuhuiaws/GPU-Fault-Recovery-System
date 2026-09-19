"""Controller intent and ACK handling for the DESTR-014 host recovery window."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.probes import destr014_recovery_probe as host
from scripts.e2e.regional.regional_commands import RegionalFixtureError

PROBE = Path(host.__file__)
RECOVERY_SECONDS = host.RECOVERY_SECONDS
# Controller journal phases of the host recovery: the probe republished its
# saved link (RESTORED) or, on a new boot, the node-installer had already
# re-enabled the Agent and the probe only retired itself (RESTORED_BY_REBOOT).
RESTORED = "RESTORED"
RESTORED_BY_REBOOT = "RESTORED_BY_REBOOT"
RESTORERS = frozenset({"probe", "reboot-installer"})
# Journal fields that exist only once the host was touched. The recovery window
# writes the ``host_*`` records from ``arm`` onwards (the last three never
# appear without ``host_binding``; they are listed so the rule reads as "no host
# record at all"). The runner checkpoints ``agent_disabled`` before disabling
# the sibling's Agent, ``holder_armed`` before arming the GPU device holder,
# ``injection_started``/``physical_outcome_unknown``/``marker`` before writing
# the XIDs, and ``incident_id``/``follow_up_incident_id`` once the control plane
# opened the resulting workflows. The env-window flags are deliberately absent:
# the windows keep their own records and guards and are closed through them.
HOST_RECORD_KEYS = (
    "host_binding",
    "host_ack",
    "host_request",
    "host_recovery_phase",
    "host_boot_id_observed",
    "host_cleanup",
)
HOST_MUTATION_FLAGS = (
    "holder_armed",
    "agent_disabled",
    "injection_started",
    "physical_outcome_unknown",
    "incident_id",
    "marker",
    "follow_up_incident_id",
)
NEVER_ARMED = "never armed on the host"


def recovery_phase(report: dict[str, Any]) -> str:
    return (
        RESTORED_BY_REBOOT
        if report.get("restored_by") == "reboot-installer"
        else RESTORED
    )


def reboot_restore_proof(
    report: Any,
    binding: dict[str, Any],
    *,
    agent_unit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the host recovery record proves about the sibling's reboot.

    The unknown-outcome hold asks one physical question -- did the node reboot
    and come back with its Agent? -- and the record answers it without any
    exec into a NotReady node: ``boot_id_observed`` is the boot the record was
    sealed on, ``restored_by`` who put the Agent back (the probe's saved inode
    or the installer's fresh link). Nothing is inferred from a Ready condition
    or a CloudTrail receipt. ``agent_unit`` is the runner's own later snapshot.
    """
    gaps: list[str] = []
    record = report if isinstance(report, dict) else {}
    observed = record.get("boot_id_observed")
    if record.get("phase") not in {"RESTORED", "CLOSED"}:
        gaps.append("recovery record is not RESTORED or CLOSED")
    if record.get("disable_started") is not True:
        gaps.append("this recovery never disabled the Agent")
    if not isinstance(observed, str) or not observed:
        gaps.append("no boot id observed on the host")
    elif observed == binding["boot_id"]:
        gaps.append("boot id unchanged: the reboot never landed or was not observed")
    if record.get("restored_by") not in RESTORERS:
        gaps.append("no Agent restoration receipt")
    if agent_unit is not None and (
        not str(agent_unit.get("UnitFileState") or "").startswith("enabled")
        or agent_unit.get("ActiveState") != "active"
    ):
        gaps.append("sibling Node Agent snapshot is not enabled and active")
    return {
        "proven": not gaps,
        "gaps": gaps,
        "boot_id_before": binding["boot_id"],
        "boot_id_observed": observed,
        "restored_by": record.get("restored_by"),
        "phase": record.get("phase"),
    }


def require_report(
    value: dict[str, Any], binding: dict[str, Any], phases: set[str]
) -> None:
    if (
        value.get("binding_sha256") != host.binding_key(binding)
        or value.get("phase") not in phases
        or type(value.get("disable_started")) is not bool
        or type(value.get("start_requested")) is not bool
        or value.get("record_kind")
        != (
            "FORENSIC_TOMBSTONE"
            if value.get("phase") == "CLOSED"
            else "UNFINISHED_RECOVERY"
        )
    ):
        raise RegionalFixtureError("DESTR-014 recovery ACK is unbound or incomplete")


class RunJournal:
    """One private, locked attempt; an existing record authorizes cleanup only."""

    def __init__(self, path: Path, scope: dict[str, Any]) -> None:
        self.path = path
        self.scope = scope
        self.fd: int | None = None
        self.data: dict[str, Any] = {}
        self.resumed = False

    def acquire(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(
            self.path.with_suffix(".lock"),
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
            ):
                raise RegionalFixtureError("DESTR-014 recovery journal lock is invalid")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data: Any = None
            retired_journal: dict[str, Any] | None = None
            if self.path.exists() or self.path.is_symlink():
                info = self.path.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                    or info.st_nlink != 1
                ):
                    raise RegionalFixtureError(
                        "DESTR-014 recovery journal is not private"
                    )
                data = json.loads(self.path.read_text())
                if self._archive_closed_foreign(data):
                    data = None
                else:
                    retired_journal = self._archive_rebooted_foreign(data)
                    if retired_journal is None:
                        retired_journal = self._archive_unarmed_foreign(data)
                    if retired_journal is not None:
                        data = None
            if data is not None:
                if (
                    not isinstance(data, dict)
                    or data.get("schema_version") != 1
                    or data.get("scope") != self.scope
                    or data.get("phase") not in {"OPEN", "RECOVERY_REQUIRED", "CLOSED"}
                    or not isinstance(data.get("run"), dict)
                ):
                    raise RegionalFixtureError(
                        "DESTR-014 recovery journal identity changed"
                    )
                if data.get("supervision_lost"):
                    raise RegionalFixtureError(
                        "DESTR-014 lost supervision; independent review required"
                    )
                self.data = data
                self.resumed = True
            if data is None:
                self.data = {
                    "schema_version": 1,
                    "scope": self.scope,
                    "phase": "OPEN",
                    "run": {},
                }
                if retired_journal is not None:
                    # Keep the lineage of a retired journal (rebooted away or
                    # never armed) so cleanup and verdict evidence can trace it
                    # back to its archive.
                    self.data["run"]["retired_journals"] = [retired_journal]
                self.save()
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def _archive_closed_foreign(self, data: Any) -> bool:
        """Move aside a CLOSED journal that another attempt of this case wrote.

        The journal lives at one path per case directory, so a later attempt
        finds the previous attempt's forensic tombstone under a different
        scope (run id, plan digest, release) and used to be refused as
        "identity changed" -- attempt 3 of 2026-09-18 died there without
        touching the node. A CLOSED journal has finished its recovery and owns
        nothing; it is kept next to the live one under its own run id. An
        unfinished journal that lost supervision, or one that may own host
        state, still refuses here; an unfinished journal armed on a boot the
        node has since replaced is retired instead by
        ``_archive_rebooted_foreign``, and one that provably never touched the
        host by ``_archive_unarmed_foreign``.
        """

        if (
            not isinstance(data, dict)
            or data.get("phase") != "CLOSED"
            or data.get("supervision_lost")
            or data.get("scope") == self.scope
        ):
            return False
        self.path.replace(self._archive_path("closed", data.get("scope")))
        return True

    def _archive_path(self, kind: str, old_scope: Any) -> Path:
        """``<stem>.<kind>-<old run id><suffix>`` next to the live journal; a
        second archive of the same run id gets this process id appended."""

        run_id = "unknown"
        if isinstance(old_scope, dict):
            run_id = str(old_scope.get("run_id") or "unknown")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)
        archive = self.path.with_name(
            f"{self.path.stem}.{kind}-{safe}{self.path.suffix}"
        )
        if archive.exists():
            archive = self.path.with_name(
                f"{self.path.stem}.{kind}-{safe}-{os.getpid()}{self.path.suffix}"
            )
        return archive

    def _archive_rebooted_foreign(self, data: Any) -> dict[str, Any] | None:
        """Retire an unfinished journal armed on a boot the node has replaced.

        The journal lives at one path per case directory and guards host state
        on the boot it was armed on (``scope.boot_id``). When a later attempt
        finds an unfinished journal (OPEN or RECOVERY_REQUIRED) under another
        scope whose boot id is a non-empty string differing from this attempt's,
        the node has rebooted since: the node-installer recreated the Node Agent
        and the recovery probe's new-boot decision table already declared that
        boot's host state retired, so the old journal owns nothing on the
        current boot (the live wedge of 2026-09-18 attempt 4, where a
        RECOVERY_REQUIRED journal from a since-rebooted sibling refused every
        later attempt as "identity changed"). It is moved aside to
        ``<stem>.rebooted-<run id>`` (collision-safe, like the closed archive)
        and its lineage is returned for the new journal's
        ``run["retired_journals"]``. A journal whose boot id still matches (the
        node did not reboot), one that lost supervision, a CLOSED tombstone, or
        one where either scope is missing a string boot id is never retired
        here -- such a journal may own host state and keeps refusing, unless
        ``_archive_unarmed_foreign`` can prove it never touched the host.
        """

        if (
            not isinstance(data, dict)
            or data.get("phase") not in {"OPEN", "RECOVERY_REQUIRED"}
            or data.get("supervision_lost")
            or data.get("scope") == self.scope
        ):
            return None
        old_scope = data.get("scope")
        if not isinstance(old_scope, dict):
            return None
        old_boot = old_scope.get("boot_id")
        new_boot = self.scope.get("boot_id")
        if (
            not isinstance(old_boot, str)
            or not old_boot
            or not isinstance(new_boot, str)
            or not new_boot
            or old_boot == new_boot
        ):
            return None
        archive = self._archive_path("rebooted", old_scope)
        self.path.replace(archive)
        return {
            "archive": str(archive),
            "old_run_id": old_scope.get("run_id"),
            "old_phase": data.get("phase"),
            "old_boot_id": old_boot,
            "old_host_request": data.get("host_request"),
            "retired_at": datetime.now(timezone.utc).isoformat(),
        }

    def _archive_unarmed_foreign(self, data: Any) -> dict[str, Any] | None:
        """Retire an unfinished journal that provably never touched the host.

        Runs after ``_archive_rebooted_foreign`` declined, so the boot is
        unchanged (or unprovable) and the only question left is whether the
        old attempt armed anything. An attempt that died before its first host
        mutation -- at the env-window step, say -- leaves an OPEN or
        RECOVERY_REQUIRED journal whose ``run`` carries none of the checkpoints
        the runner writes ahead of a host mutation (``HOST_MUTATION_FLAGS``) and
        none of the recovery window's ``host_*`` records (``HOST_RECORD_KEYS``);
        the live wedge of 2026-09-19 attempt 7 met such a journal from attempt
        6 on the same boot and, since every later attempt has a new run id, it
        would have refused the case forever as "identity changed". Such a
        journal owns no host state: it is moved aside to
        ``<stem>.abandoned-<run id>`` (collision-safe, like the other archives)
        and its lineage returned for the new journal's
        ``run["retired_journals"]`` with ``reason`` ``NEVER_ARMED``. The
        env-window flags do not block retirement -- the windows keep their own
        records and guards and are closed through them. Any present host
        record, any truthy mutation flag, lost supervision, a CLOSED tombstone
        or a malformed ``scope``/``run`` leaves the journal to refuse exactly as
        before.
        """

        if (
            not isinstance(data, dict)
            or data.get("phase") not in {"OPEN", "RECOVERY_REQUIRED"}
            or data.get("supervision_lost")
            or data.get("scope") == self.scope
        ):
            return None
        old_scope = data.get("scope")
        run = data.get("run")
        if not isinstance(old_scope, dict) or not isinstance(run, dict):
            return None
        if any(data.get(key) is not None for key in HOST_RECORD_KEYS) or any(
            run.get(flag) for flag in HOST_MUTATION_FLAGS
        ):
            return None
        archive = self._archive_path("abandoned", old_scope)
        self.path.replace(archive)
        return {
            "archive": str(archive),
            "old_run_id": old_scope.get("run_id"),
            "old_phase": data.get("phase"),
            "old_boot_id": old_scope.get("boot_id"),
            "old_host_request": data.get("host_request"),
            "retired_at": datetime.now(timezone.utc).isoformat(),
            "reason": NEVER_ARMED,
        }

    def save(self) -> None:
        write_json_atomic(self.path, self.data)
        host.sync_directory(self.path.parent)

    def checkpoint(self, **fields: Any) -> None:
        self.data["run"].update(fields)
        self.save()

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def make_binding(
    *,
    scope: dict[str, Any],
    owner: str,
    agent: dict[str, Any],
    restore_at: int,
    expires_at: int,
) -> dict[str, Any]:
    binding = {
        "case_id": host.CASE,
        "run_id": scope["run_id"],
        "owner": owner,
        "release_id": scope["release_id"],
        "plan_sha256": scope["plan_sha256"],
        "helper_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        "cluster_id": scope["cluster_id"],
        "node": scope["node"],
        "node_uid": scope["node_uid"],
        "boot_id": scope["boot_id"],
        "artifact_sha256": agent.get("artifact_sha256"),
        "bundle_sha256": agent.get("installer_bundle_sha256"),
        "profile_version": agent.get("runtime_profile_version"),
        "restore_at": restore_at,
        "expires_at": expires_at,
    }
    host.binding_key(binding)
    if agent.get("node_instance_id") != binding["node_uid"]:
        raise RegionalFixtureError("DESTR-014 Agent and Node UID differ")
    return binding


class AgentRecoveryWindow:
    def __init__(self, journal: RunJournal, probe: HostProbeFixture) -> None:
        self.journal = journal
        self.probe = probe

    @property
    def binding(self) -> dict[str, Any]:
        value = self.journal.data.get("host_binding")
        if not isinstance(value, dict):
            raise RegionalFixtureError("DESTR-014 has no durable host recovery binding")
        host.binding_key(value)
        return value

    def request(self, command: str, phases: set[str]) -> dict[str, Any]:
        binding = self.binding
        self.journal.data["host_request"] = command
        self.journal.save()
        self.probe.create()
        result = self.probe.execute(
            command,
            "--binding",
            json.dumps(binding, sort_keys=True),
            timeout=120,
        )
        require_report(result, binding, phases)
        self.journal.data["host_ack"] = result
        self.journal.save()
        return result

    def arm(self, binding: dict[str, Any]) -> dict[str, Any]:
        if "host_binding" in self.journal.data:
            raise RegionalFixtureError(
                "DESTR-014 recovery already started; cleanup only"
            )
        host.binding_key(binding)
        self.journal.data["host_binding"] = binding
        self.journal.save()
        self.request("prepare", {"INSTALLED"})
        deadline = time.monotonic() + 60
        while True:
            result = self.request("status", {"INSTALLED", "ARMED"})
            if result["phase"] == "ARMED":
                ack = result.get("ack") or {}
                at = ack.get("at")
                if (
                    ack.get("boot_id") != binding["boot_id"]
                    or not isinstance(ack.get("invocation_id"), str)
                    or not ack["invocation_id"]
                    or not isinstance(at, (int, float))
                    or isinstance(at, bool)
                    or not 0 <= time.time() - at <= host.ACK_SECONDS
                ):
                    raise RegionalFixtureError(
                        "DESTR-014 independent arm proof is stale"
                    )
                return result
            if time.monotonic() >= deadline:
                raise RegionalFixtureError(
                    "DESTR-014 independent arm ACK was not observed"
                )
            time.sleep(1)

    def disable(self) -> dict[str, Any]:
        require_report(self.journal.data.get("host_ack", {}), self.binding, {"ARMED"})
        result = self.request("disable", {"DISABLED"})
        if result["disable_started"] is not True:
            raise RegionalFixtureError("DESTR-014 Agent disable was not acknowledged")
        return result

    def restore(self) -> dict[str, Any]:
        result = self.request("restore", {"RESTORED"})
        # The deadline safeguard firing first invalidates the scenario; the
        # installer re-enabling the Agent on the rebooted node is the product
        # acting, judged by the workflow verdicts, not a safeguard.
        if (
            result.get("restored_by") != "reboot-installer"
            and result.get("restore_reason") != "controller"
        ):
            raise RegionalFixtureError(
                "DESTR-014 automatic safeguard fired before scenario completion"
            )
        self.record_recovery_phase(result)
        return result

    def record_recovery_phase(self, result: dict[str, Any]) -> None:
        self.journal.data["host_recovery_phase"] = recovery_phase(result)
        self.journal.data["host_boot_id_observed"] = result.get("boot_id_observed")
        self.journal.save()

    def cleanup(self) -> dict[str, Any]:
        if "host_binding" not in self.journal.data:
            return {"phase": "NOT_CREATED"}
        result = self.request("cleanup", {"CLOSED"})
        residuals = self.probe.cleanup()
        if not isinstance(residuals, dict) or any(residuals.values()):
            raise RegionalFixtureError("DESTR-014 recovery probe cleanup is incomplete")
        self.journal.data["host_cleanup"] = result
        if result.get("restored_by") in RESTORERS:
            self.record_recovery_phase(result)
        self.journal.save()
        return result
