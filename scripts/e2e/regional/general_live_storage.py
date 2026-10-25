"""GF-GENERAL-LIVE-004 local disk full and 006 disk I/O saturation.

Both use a *test* file system or device of ``general_live_storage_probe.py``,
mounted under ``/mnt/gfrs-live/<digest>/<name>`` and added to the host
collector's ``GPU_FAULT_FILESYSTEMS`` list through a collector-window drop-in
(004) or measured as its own ``dm-N`` device in ``/proc/diskstats`` (006). The
system disk, ``/var`` and the log disk are never touched.

004 ``filesystem_used_percent`` >= 98 is CRITICAL QUARANTINE (immediate):
MARK_UNSCHEDULABLE then QUARANTINE, no STOP (no ``stop_on_isolation``), the
incident ends QUARANTINED and the node is returned through the product's
validated restore. ``filesystem_inodes_used_percent`` (``f_files/f_ffree``) is
graded the same way, so ``inode_full`` runs that chain on the inode signal. A
configured mount that is read-only, gone, or whose ``statvfs`` raises is
``filesystem_unavailable`` (READ_ONLY / MOUNT_MISSING / PROBE_FAILED): a
VALIDATE_HOST diagnostic with no capacity finding and no reset, reboot or
quarantine (batch 163), the node held while it stays unusable and readmitted
once the mount is restored. ``probe_error``'s faulting mount is the storage
probe's loop + device-mapper device carrying FAT16 (``dm-create --fstype
vfat``): ``dm-retarget --target error`` swaps its table to the ``error``
target built into ``dm_mod`` under a fresh mount, so the mount stays, reads
read-write in the host table, and every ``statvfs`` fails EIO (FAT counts its
free clusters from the device at the first ``statfs`` after a mount; ext4
answers from memory). ``--target data`` restores it. The node log collector
must be inactive (its storage rule grades a ``Buffer I/O error`` line
QUARANTINE; the swap is made with nothing dirty so no such line is written,
but an active collector would make the chain ambiguous). A node without
``mkfs.vfat`` (the AL2023 EKS GPU AMI) keeps ``probe_error``
NOT_RUN(CAPABILITY_ABSENT) unless the optional ``TEMP_TEST_PACKAGE_INSTALL`` is
given; then the shared temporary-package step (``general_live_deps``) installs
exactly ``dosfstools`` for this run before the device is built and removes it
as the last LIFO cleanup, after the device is destroyed. 006
``disk_io_util_percent``/``disk_io_await_ms`` are WARNING
RUN_DIAGNOSTICS with a 120 s sustain; the oracle's own ``/proc/diskstats`` read
computes the standard await (fields 4+8 over 1+5) to compare with the product's,
which has used the same definition since batch 172 (``reference_iostat`` expects
PASS).

The 006 test device is built by a bounded transient setup unit on the node:
``dm-create`` only starts it and ``_await_dm_setup`` polls ``dm-status`` on the
context clock. Every 006 probe call gets the probe's own per-subcommand budget
plus ``TRANSPORT_MARGIN_SECONDS`` as its ``kubectl exec`` timeout, and every
undo is on the LIFO stack before the call it undoes (see the storage probe's
docstring for why a ``dmsetup`` in the probe Pod never returned).
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    await_metric_recovery,
    Oracle,
    activity,
    close_window,
    open_window,
    restore_incidents,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_deps import (
    PACKAGE_APPROVAL,
    dependency_summary,
    deps_deadman_seconds,
    ensure_dependencies,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    backing_root,
    require_module,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    oracle_values,
    samples,
    sustained,
    verdict,
)
from scripts.e2e.regional.probes import general_live_storage_probe as storage_probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CASE_004 = "GF-GENERAL-LIVE-004"
CASE_006 = "GF-GENERAL-LIVE-006"
QUARANTINE_BUDGET = 420
IO_SUSTAIN = 120
#: The sample window of ``reference_iostat`` and the settle budget: the
#: detection budget of the sustained variants is ``io_detection_hold``.
IO_BUDGET = IO_SUSTAIN + 300 + 90
#: How long the writer may take to bring the oracle to the threshold: the
#: dm-delay device reads its 200 ms from the first request, a linear device
#: needs util to climb over the first collector intervals.
IO_RAMP_BUDGET = 180
#: Ingest, dispatch and the +-5 % summary jitter on top of the cadence terms.
IO_CHAIN_MARGIN = 180
SHORT_FLUSH = 45
TEST_FS = "fs"
TEST_DM = "io"
#: The 004 ``probe_error`` faulting device (FAT16 over loop + dm).
TEST_ERR = "err"
ERR_SIZE_MIB = 256
LOG_COLLECTOR_UNIT = "gpu-fault-log-collector.service"
#: What a 006 storage call may take on top of the probe's own budget for it
#: (``storage_probe.CALL_BUDGET_SECONDS``): the ``kubectl exec`` round trip,
#: the chroot and the interpreter start.
TRANSPORT_MARGIN_SECONDS = 60
#: How often the runner reads the setup unit's progress, and for how long: the
#: unit's own ``RuntimeMaxSec`` and one more status call, so the unit has
#: always ended -- ``ready``, or ``failed`` at a recorded step -- by then.
SETUP_POLL_SECONDS = 5
SETUP_WAIT_SECONDS = (
    storage_probe.SETUP_MAX_SECONDS
    + storage_probe.CALL_BUDGET_SECONDS["dm-status"]
    + TRANSPORT_MARGIN_SECONDS
)


def _fs_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_FS}"


def _dm_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_DM}"


def _err_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_ERR}"


def _digest(ctx: VariantContext) -> str:
    import hashlib

    return hashlib.sha256(ctx.run_id.encode()).hexdigest()[:12]


def _create_fs(
    ctx: VariantContext, *, inodes: int = 0, size_mib: int = 2048
) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "losetup", "mkfs.ext4", "mount", "fallocate")
    root = backing_root(survey)
    arguments = [
        "fs-create",
        "--name",
        TEST_FS,
        "--size-mib",
        str(size_mib),
        "--backing",
        root,
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    ]
    if inodes:
        arguments += ["--inodes", str(inodes)]
    created = ctx.storage(*arguments)
    ctx.cleanup.push("fs", lambda: ctx.storage("fs-destroy", "--name", TEST_FS))
    ctx.evidence.write("fs-create", created)
    return created


def _declare_fs(ctx: VariantContext, mount: str | None = None) -> None:
    """Add the test mount to GPU_FAULT_FILESYSTEMS through a window drop-in."""

    open_window(
        ctx,
        f"GPU_FAULT_FILESYSTEMS={mount or _fs_mount(ctx)}",
        restore_seconds=QUARANTINE_BUDGET + 600,
    )


def capacity_full(ctx: VariantContext) -> dict[str, Any]:
    _create_fs(ctx)
    _declare_fs(ctx)
    mount = _fs_mount(ctx)
    started = utc_now()
    filled = ctx.storage("fs-fill", "--name", TEST_FS, "--percent", "99")
    ctx.evidence.write("fs-fill", filled)
    used = (filled.get("statvfs") or {}).get("used_percent")
    if not isinstance(used, (int, float)) or used < 98:
        raise VariantInconclusive(f"the test file system only reached {used}% used")
    detection = wait_for_incident(
        ctx,
        started,
        "filesystem_used_percent",
        device=mount,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"filesystem_used_percent"},
        name="fs-quarantine",
    )
    incidents = finding_incidents(detection, "filesystem_used_percent", device=mount)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="filesystem_used_percent",
                device=mount,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=frozenset({"STOP_WORKLOADS"}) | hardware_only(),
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
    }
    settled = wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"filesystem_used_percent"},
        name="fs-settled",
    )
    # Release capacity, close the window, then readmit through validated restore.
    ctx.storage("fs-unfill", "--name", TEST_FS)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restored = restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-004 capacity_full restore"
    )
    stages["restore"] = (
        [] if restored or not incidents else ["validated restore ran no workflow"]
    )
    return _result(stages, extra={"used_percent": used, "settled": _statuses(settled)})


def below_threshold(ctx: VariantContext) -> dict[str, Any]:
    _create_fs(ctx)
    _declare_fs(ctx)
    mount = _fs_mount(ctx)
    started = utc_now()
    filled = ctx.storage("fs-fill", "--name", TEST_FS, "--percent", "97")
    used = (filled.get("statvfs") or {}).get("used_percent")
    if not isinstance(used, (int, float)) or not 90 <= used < 98:
        raise VariantInconclusive(
            f"the test file system reached {used}% (wanted 90..98)"
        )
    observed = _watch_no_finding(
        ctx, started, "filesystem_used_percent", QUARANTINE_BUDGET, device=mount
    )
    stages = {
        "no_quarantine": chain_errors(
            observed,
            ChainExpectation(
                metric="filesystem_used_percent", device=mount, present=False
            ),
        ),
    }
    ctx.storage("fs-unfill", "--name", TEST_FS)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(stages, extra={"used_percent": used})


def duplicate(ctx: VariantContext) -> dict[str, Any]:
    _create_fs(ctx)
    _declare_fs(ctx)
    mount = _fs_mount(ctx)
    started = utc_now()
    ctx.storage("fs-fill", "--name", TEST_FS, "--percent", "99")
    detection = wait_for_incident(
        ctx,
        started,
        "filesystem_used_percent",
        device=mount,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"filesystem_used_percent"},
        name="dup-quarantine",
    )
    incidents = finding_incidents(detection, "filesystem_used_percent", device=mount)
    # Keep it full across a collector restart (the window close/open cycle) and a
    # second health summary; the latch must not build a second incident/workflow.
    ctx.window(
        "start-unit",
        "--unit",
        "gpu-fault-host-collector.service",
        "--run-id",
        ctx.run_id,
    )
    ctx.hold(ctx.host_summary + ctx.host_interval * 3, name="duplicate-hold")
    final = activity(ctx, started, metrics={"filesystem_used_percent"})
    stages = {
        "single": chain_errors(
            final,
            ChainExpectation(
                metric="filesystem_used_percent",
                device=mount,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
    }
    ctx.storage("fs-unfill", "--name", TEST_FS)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restore_incidents(ctx, incidents, reason="GF-GENERAL-LIVE-004 duplicate restore")
    return _result(stages)


UNAVAILABLE_METRIC = "filesystem_unavailable"
#: The ``failure_mode`` label the product puts on each unusable reading.
UNAVAILABLE_MODES = {
    "read_only": "READ_ONLY",
    "required_mount_missing": "MOUNT_MISSING",
    "probe_error": "PROBE_FAILED",
}


def _refuse_active_log_collector(survey: dict[str, Any]) -> None:
    """``probe_error`` needs the node log collector off.

    Its storage rule (``log_rules.py``) grades ``Buffer I/O error`` /
    ``blk_update_request ... error`` lines CRITICAL QUARANTINE. The table swap
    is made with nothing dirty so the error target is never written, but with
    the collector active any such line would turn the diagnostic-only chain
    under test into a quarantine the verdict could not attribute.
    """

    state = (survey.get("units") or {}).get(LOG_COLLECTOR_UNIT) or {}
    residue = survey.get("residue") or {}
    if state.get("ActiveState") == "active" or residue.get(
        "log_collector_enabled_by_run"
    ):
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            "the node log collector is active; its storage rule would grade a "
            "block I/O error line QUARANTINE and mask the probe_error chain",
        )


def _ensure_mkfs_vfat(ctx: VariantContext, survey: dict[str, Any]) -> dict[str, Any]:
    """``mkfs.vfat`` as the node has it, or ``dosfstools`` for this run only.

    The shared temporary-package step: NOT_RUN(CAPABILITY_ABSENT) naming the
    token without ``TEMP_TEST_PACKAGE_INSTALL``; with it the probe installs
    exactly ``dosfstools``, the survey then says the node has ``mkfs.vfat`` for
    ``_create_dm``'s own tool check, and the removal is the last LIFO cleanup,
    after the test device is destroyed. The package deadman is timed after the
    device's (``max_minutes * 60 + 600``).
    """

    return ensure_dependencies(
        ctx,
        survey,
        tools=("mkfs.vfat",),
        deadman_seconds=deps_deadman_seconds(ctx.spec.max_minutes * 60 + 600),
        evidence="probe-error-dependencies",
    )


def _failure_mode_errors(detection: dict[str, Any], mount: str, mode: str) -> list[str]:
    """The unusable readings carry the variant's ``failure_mode`` label.

    Only readings that carry a label are judged: a product build whose
    accepted batches keep no labels is noted, not failed.
    """

    seen = {
        str(sample["labels"]["failure_mode"])
        for sample in samples(detection, UNAVAILABLE_METRIC, device=mount)
        if sample.get("value") == 1
        and isinstance(sample.get("labels"), dict)
        and sample["labels"].get("failure_mode")
    }
    if seen and mode not in seen:
        return [
            f"filesystem_unavailable on {mount} is labelled {sorted(seen)}, not {mode}"
        ]
    return []


def _unusable_filesystem_variant(
    ctx: VariantContext, *, variant: str
) -> dict[str, Any]:
    """A configured mount made unusable without touching its capacity.

    Read-only, vanished, or an unreadable ``statvfs``: the product reads each as
    ``filesystem_unavailable`` and runs the VALIDATE_HOST diagnostic -- never a
    capacity finding (the free bytes are untouched), never a reset, reboot or
    quarantine. The node is held while the mount stays unusable, and readmitted
    once the injection is undone and the signal clears to 0 (batch 163 closed
    the capacity-only gap these variants used to record as expected FAILs).
    ``probe_error`` uses the FAT16 dm device and its ``error`` table swap; the
    others the loop ext4 file system.
    """

    survey = run_survey(ctx)
    dependencies: dict[str, Any] = {"changed": False, "installed": []}
    if variant == "probe_error":
        _refuse_active_log_collector(survey)
        dependencies = _ensure_mkfs_vfat(ctx, survey)
        created = _create_dm(ctx, name=TEST_ERR, fstype="vfat", size_mib=ERR_SIZE_MIB)
        mount = str(created.get("mountpoint") or _err_mount(ctx))
    else:
        require_tools(survey, "losetup", "mkfs.ext4", "mount", "fallocate")
        mount = _fs_mount(ctx)
        _create_fs(ctx)
    _declare_fs(ctx, mount)
    started = utc_now()
    injected = _inject_unusable(ctx, variant)
    ctx.evidence.write("injection", injected)
    detection = wait_for_incident(
        ctx,
        started,
        UNAVAILABLE_METRIC,
        device=mount,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={UNAVAILABLE_METRIC, "filesystem_used_percent"},
        validation="VALIDATE_HOST",
        name=f"{variant}-detection",
    )
    incidents = finding_incidents(detection, UNAVAILABLE_METRIC, device=mount)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric=UNAVAILABLE_METRIC,
                device=mount,
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric=UNAVAILABLE_METRIC,
                forbidden=frozenset(
                    {"STOP_WORKLOADS", "MARK_UNSCHEDULABLE", "QUARANTINE"}
                )
                | hardware_only(),
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        # The free bytes are untouched: no capacity finding may stand in for the
        # unusable-mount one (the spec's "cannot judge health by free bytes").
        "bytes_not_graded": chain_errors(
            detection,
            ChainExpectation(
                metric="filesystem_used_percent", device=mount, present=False
            ),
        ),
        "hardware": hardware_errors(detection),
        "failure_mode": _failure_mode_errors(
            detection, mount, UNAVAILABLE_MODES[variant]
        ),
    }
    # Undo the injection, then readmit once the availability signal clears.
    _restore_unusable(ctx, variant)
    recovered = await_metric_recovery(ctx, UNAVAILABLE_METRIC, 1.0, device=mount)
    stages["recovery"] = recovered["errors"]
    if not incidents:
        stages["recovery"].append("no filesystem_unavailable finding to recover from")
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={UNAVAILABLE_METRIC},
        name=f"{variant}-settled",
    )
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(
        stages,
        extra={
            "injection": injected,
            "recovery": recovered,
            "dependencies": dependency_summary(dependencies),
        },
    )


def _inject_unusable(ctx: VariantContext, variant: str) -> dict[str, Any]:
    """Make the test mount unusable and prove the injection with its statvfs."""

    if variant == "read_only":
        injected = ctx.storage("fs-remount", "--name", TEST_FS, "--mode", "ro")
        if not (injected.get("statvfs") or {}).get("read_only"):
            raise VariantInconclusive("the remount did not report read-only")
        ctx.cleanup.push(
            "fs-rw",
            lambda: ctx.storage("fs-remount", "--name", TEST_FS, "--mode", "rw"),
        )
        return injected
    if variant == "required_mount_missing":
        injected = ctx.storage("fs-umount", "--name", TEST_FS, "--rmdir")
        if (injected.get("statvfs") or {}).get(
            "used_percent"
        ) is not None and injected.get("directory_present"):
            raise VariantInconclusive("the mount did not disappear")
        ctx.cleanup.push("fs-mount", lambda: ctx.storage("fs-mount", "--name", TEST_FS))
        return injected
    if variant == "probe_error":
        # Registered before the swap: a lost answer still puts the data table
        # back before the device is destroyed (LIFO).
        ctx.cleanup.push("dm-data", lambda: _retarget(ctx, "data"))
        injected = _retarget(ctx, "error")
        if not injected.get("faulted") or not injected.get("mounted_on_host", True):
            raise VariantInconclusive(
                "the error-table swap left the mount's statvfs answering "
                f"(attempts {injected.get('attempts')}); no EIO to inject"
            )
        return injected
    raise VariantInconclusive(f"unknown unusable-filesystem variant {variant!r}")


def _retarget(ctx: VariantContext, target: str) -> dict[str, Any]:
    return _device_call(ctx, "dm-retarget", "--name", TEST_ERR, "--target", target)


def _restore_unusable(ctx: VariantContext, variant: str) -> None:
    if variant == "read_only":
        ctx.storage("fs-remount", "--name", TEST_FS, "--mode", "rw")
        ctx.cleanup.discard("fs-rw")
    elif variant == "required_mount_missing":
        ctx.storage("fs-mount", "--name", TEST_FS)
        ctx.cleanup.discard("fs-mount")
    elif variant == "probe_error":
        restored = _retarget(ctx, "data")
        ctx.cleanup.discard("dm-data")
        if "error" in (restored.get("statvfs") or {}):
            raise RegionalFixtureError("the data table did not restore the mount")


INODE_METRIC = "filesystem_inodes_used_percent"


def inode_full(ctx: VariantContext) -> dict[str, Any]:
    """Inodes exhausted while bytes stay free: the inode signal quarantines.

    ``filesystem_inodes_used_percent`` >= 98 is graded like capacity: CRITICAL
    QUARANTINE, MARK_UNSCHEDULABLE then QUARANTINE, no STOP, the incident
    QUARANTINED. The byte signal must not be what fired -- the bytes stay
    below 5 % -- and the node is readmitted through the validated restore
    once ``fs-unfill`` has deleted the files, so no quarantine outlives the
    variant.
    """

    _create_fs(ctx, inodes=256)
    _declare_fs(ctx)
    mount = _fs_mount(ctx)
    started = utc_now()
    injected = ctx.storage("fs-inodes", "--name", TEST_FS, timeout=300)
    ctx.evidence.write("injection", injected)
    if not injected.get("enospc"):
        raise VariantInconclusive("inode exhaustion did not reach ENOSPC")
    detection = wait_for_incident(
        ctx,
        started,
        INODE_METRIC,
        device=mount,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={INODE_METRIC, "filesystem_used_percent"},
        name="inode-quarantine",
    )
    incidents = finding_incidents(detection, INODE_METRIC, device=mount)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric=INODE_METRIC,
                device=mount,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=frozenset({"STOP_WORKLOADS"}) | hardware_only(),
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
        "bytes_not_graded": chain_errors(
            detection,
            ChainExpectation(
                metric="filesystem_used_percent", device=mount, present=False
            ),
        ),
    }
    settled = wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={INODE_METRIC},
        name="inode-settled",
    )
    # Delete the files, close the window, then readmit through validated restore.
    ctx.storage("fs-unfill", "--name", TEST_FS)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restored = restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-004 inode_full restore"
    )
    stages["restore"] = (
        [] if restored or not incidents else ["validated restore ran no workflow"]
    )
    return _result(stages, extra={"injection": injected, "settled": _statuses(settled)})


def read_only(ctx: VariantContext) -> dict[str, Any]:
    return _unusable_filesystem_variant(ctx, variant="read_only")


def required_mount_missing(ctx: VariantContext) -> dict[str, Any]:
    return _unusable_filesystem_variant(ctx, variant="required_mount_missing")


def probe_error(ctx: VariantContext) -> dict[str, Any]:
    return _unusable_filesystem_variant(ctx, variant="probe_error")


# --------------------------------------------------------------------------- #
# 006 disk I/O
# --------------------------------------------------------------------------- #
def _device_call(ctx: VariantContext, subcommand: str, *arguments: str) -> Any:
    """One 006 storage probe call, timed out only after the probe's budget."""

    timeout = storage_probe.CALL_BUDGET_SECONDS[subcommand] + TRANSPORT_MARGIN_SECONDS
    return ctx.storage(subcommand, *arguments, timeout=timeout)


def _checked_dm_cleanup(ctx: VariantContext, name: str = TEST_DM) -> dict[str, Any]:
    """Destroy the test device with its units, deadmen and a module this run
    loaded; fail the cleanup stack unless the probe proves the node clean."""

    value = _device_call(ctx, "cleanup", "--kind", f"dm-{name}")
    if value.get("clean") is not True:
        raise RegionalFixtureError(
            f"the test device cleanup left residue: {value.get('residue')}"
        )
    return value


def _create_dm(
    ctx: VariantContext,
    *,
    delay_ms: int = 0,
    size_mib: int = 2048,
    name: str = TEST_DM,
    fstype: str = "ext4",
) -> dict[str, Any]:
    survey = ctx.survey or run_survey(ctx)
    require_tools(
        survey,
        "losetup",
        "dmsetup",
        storage_probe.MKFS[fstype][0],
        "mount",
        "blockdev",
        "fallocate",
        "systemd-run",
        "nsenter",
    )
    require_module(survey, "dm_mod", ctx.settings.approvals)
    load = bool(delay_ms) and require_module(
        survey, storage_probe.DELAY_MODULE, ctx.settings.approvals
    )
    if fstype == "vfat" and require_module(
        survey, storage_probe.VFAT_MODULE, ctx.settings.approvals
    ):
        load = True
    if load:
        require_tools(survey, "modprobe")
    root = backing_root(survey)
    arguments = [
        "--name",
        name,
        "--size-mib",
        str(size_mib),
        "--delay-ms",
        str(delay_ms),
        "--fstype",
        fstype,
        "--backing",
        root,
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    ]
    if load:
        arguments.append("--load-module")
    # Registered before the create: a failed or half-built device is destroyed
    # by this runner too, not only by the probe's deadman minutes later.
    ctx.cleanup.push("dm", lambda: _checked_dm_cleanup(ctx, name))
    started = _device_call(ctx, "dm-create", *arguments)
    ctx.evidence.write("dm-create", started)
    ready = _await_dm_setup(ctx, name)
    ctx.evidence.write("dm-ready", ready)
    return ready


def _await_dm_setup(ctx: VariantContext, name: str = TEST_DM) -> dict[str, Any]:
    """Poll ``dm-status`` until the setup unit has built the device or failed."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _device_call(ctx, "dm-status", "--name", name)
        return last if last.get("phase") in {"ready", "failed"} else None

    status = ctx.wait_until(
        accept,
        timeout_seconds=SETUP_WAIT_SECONDS,
        poll_seconds=SETUP_POLL_SECONDS,
        name="dm-setup",
    )
    if status is None:
        raise RegionalFixtureError(
            f"the test device setup did not finish within {SETUP_WAIT_SECONDS} s "
            f"(last step {last.get('step')!r})"
        )
    if status.get("phase") != "ready":
        # Fixed tokens only (step, class, digest, site): the message stays on
        # the node with the rest of the probe's output.
        trail = " ".join(
            str(status[key])
            for key in ("setup_error_class", "setup_error_code")
            if status.get(key)
        )
        site = status.get("setup_error_site")
        raise RegionalFixtureError(
            f"the test device setup failed at {status.get('failed_step')!r}: "
            f"{trail}" + (f" at {site}" if site else "")
        )
    return status


def _device_name(created: dict[str, Any]) -> str:
    device = created.get("device")
    if not device:
        raise VariantInconclusive("the test device has no dm-N name")
    return str(device)


def _run_io(
    ctx: VariantContext,
    seconds: int,
    *,
    direct: bool = True,
    threads: int = 16,
    pace_ms: int = 0,
) -> dict[str, Any]:
    arguments = [
        "io-start",
        "--name",
        TEST_DM,
        "--threads",
        str(threads),
        "--seconds",
        str(seconds),
    ]
    if direct:
        arguments.append("--direct")
    if pace_ms:
        arguments += ["--pace-ms", str(pace_ms)]
    # Registered before the start: a writer whose answer is lost is stopped too.
    ctx.cleanup.push("io", lambda: _device_call(ctx, "io-stop", "--name", TEST_DM))
    return _device_call(ctx, *arguments)


def _stop_io(ctx: VariantContext) -> dict[str, Any]:
    stopped = _device_call(ctx, "io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
    return stopped


def io_detection_hold(ctx: VariantContext) -> int:
    """The detection budget from the oracle's crossing, on the product's cadence.

    The host collector ships a threshold breach once, as its edge
    (``collectors/host/collector.py`` ``_edge_reasons``: ``threshold:<metric>``
    stays in ``_active_edge_reasons`` while any device breaches), then only on
    its health summaries (``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS``, +-5 %
    jitter). The control plane mints the finding on the first *later* active
    sample at least the rule's sustain after the first
    (``store/shared/xid.py`` ``_next_health_signal_state``): up to the sustain
    plus one summary after the edge; VALIDATE_HOST is then judged on evidence
    newer than itself, one more summary at worst. Live 2026-10-25 17:04Z (eksb,
    R13e): await edge 17:04:25, util edge 17:05:11, the summary 17:08:30 minted
    util 199 s after its edge; the 510-s budget from the load start was never
    the blocker there (it ran to 17:13:01) but covers the worst case by seconds
    only. Mirrors 003's ``page_cache_hold``.
    """

    return IO_SUSTAIN + 2 * ctx.host_summary + ctx.host_interval + IO_CHAIN_MARGIN


def _await_crossing(
    ctx: VariantContext,
    oracle: Oracle,
    *,
    metric: str,
    device: str,
    key: str,
    threshold: float,
) -> dict[str, Any]:
    """The oracle's first reading at or above ``threshold``: when the condition
    began on the device, from which the detection budget is counted."""

    crossing: dict[str, Any] = {}

    def crossed() -> dict[str, Any] | None:
        oracle.read()
        for record in oracle.records:
            value = record.get("diskstats", {}).get(device, {}).get(key)
            if isinstance(value, (int, float)) and value >= threshold:
                crossing.update({"at": record.get("at"), "value": float(value)})
                return crossing
        return None

    if not ctx.wait_until(
        crossed, timeout_seconds=IO_RAMP_BUDGET, poll_seconds=15, name=f"{metric}-ramp"
    ):
        values = oracle_values(
            [record.get("diskstats", {}).get(device, {}) for record in oracle.records],
            lambda record: record.get(key),
        )
        raise VariantInconclusive(
            f"condition not established: the writer did not bring {key} on "
            f"{device} to {threshold:g} within {IO_RAMP_BUDGET} s "
            f"(last {[round(value, 1) for value in values[-6:]]})"
        )
    return crossing


def _io_metric_variant(
    ctx: VariantContext, *, metric: str, delay_ms: int
) -> dict[str, Any]:
    created = _create_dm(ctx, delay_ms=delay_ms)
    device = _device_name(created)
    started = utc_now()
    key = "util_percent" if metric == "disk_io_util_percent" else "standard_await_ms"
    threshold = 98.0 if metric == "disk_io_util_percent" else 100.0
    ramp = IO_RAMP_BUDGET
    hold = io_detection_hold(ctx)
    oracle = Oracle(ctx, "disk")
    oracle.start(ramp + hold + 240, interval=2, diskstats=[device])
    load = _run_io(ctx, ramp + hold + 240)
    ctx.evidence.write(
        "io-load",
        {
            **load,
            "ramp_budget_seconds": ramp,
            "detection_budget_seconds": hold,
            "host_summary_seconds": ctx.host_summary,
            "host_interval_seconds": ctx.host_interval,
        },
    )
    # Detection is timed from the moment the runner's own oracle reads the
    # threshold on the device, not from the load start, over the product's
    # cadence (``io_detection_hold``).
    crossing = _await_crossing(
        ctx, oracle, metric=metric, device=device, key=key, threshold=threshold
    )
    detection = wait_for_incident(
        ctx,
        started,
        metric,
        device=device,
        timeout_seconds=hold,
        metrics={metric},
        validation="VALIDATE_HOST",
        name=f"{metric}-detection",
    )
    oracle.read()
    values = oracle_values(
        [record.get("diskstats", {}).get(device, {}) for record in oracle.records],
        lambda record: record.get(key),
    )
    held = sustained(values, lambda value: value >= threshold, IO_SUSTAIN // 2)
    incidents = finding_incidents(detection, metric, device=device)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric=metric,
                device=device,
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric=metric,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
    }
    # The recovery proof runs whatever the verdict so far: a device destroyed
    # with its signal still active -- and notified once a finding was minted --
    # keeps that state in the store, where the next run's active samples read
    # as the same notified episode and mint nothing (live 2026-10-25 17:04Z on
    # eksb: the 15:03Z reference_iostat await incident, no clearing sample,
    # "no disk_io_await_ms on dm-0 finding reached an incident").
    _stop_io(ctx)
    recovery_load = _recovery_load(ctx, metric, delay_ms)
    recovered = _await_io_recovery(ctx, metric, device, threshold)
    if recovery_load:
        _stop_io(ctx)
    if not held:
        raise VariantInconclusive(
            f"the {metric} oracle never sustained >= {threshold}: {values[-8:]}"
        )
    stages["recovery"] = recovered["errors"]
    if incidents:
        # No incident is nothing to settle: a wait could never accept.
        wait_settled(
            ctx,
            started,
            [item["incident_id"] for item in incidents],
            timeout_seconds=IO_BUDGET,
            metrics={metric},
            name=f"{metric}-settled",
        )
    return _result(
        stages,
        extra={
            "device": device,
            "crossing": crossing,
            "detection_budget_seconds": hold,
            "recovery": recovered,
            "recovery_load": recovery_load,
        },
    )


#: The recovery writer's pace: ten 64 KiB O_DIRECT writes a second keep
#: completed requests in every collector interval at well under 1 % util.
RECOVERY_LOAD_PACE_MS = 100
RECOVERY_LOAD_MARGIN_SECONDS = 60


def _recovery_load(ctx: VariantContext, metric: str, delay_ms: int) -> dict[str, Any]:
    """What the await recovery proof needs on the device, started before the
    wait: the injected delay out of the data table, and completed I/O.

    ``disk_io_await_ms`` exists only for intervals with completed requests
    (the zero-completed-I/O rule), so an idle device never yields the fresh
    below-threshold sample that clears the signal -- live 2026-10-25 14:47Z
    (ip-10-0-43-5, R13e) the 690 s wait saw none, the oracle read no
    completion after ``io-stop`` -- and with the 200 ms table still loaded
    any I/O reads above the threshold. The table goes back to linear in
    place (same dm-N, mount and files) and a paced single O_DIRECT writer
    keeps completed requests flowing for the length of the wait; the caller
    stops it afterwards. Util needs neither: field 10 over elapsed time reads
    0 % on an idle device.
    """

    if metric != "disk_io_await_ms":
        return {}
    removed = None
    if delay_ms:
        removed = _device_call(
            ctx, "dm-retarget", "--name", TEST_DM, "--target", "data", "--delay-ms", "0"
        )
        ctx.evidence.write("dm-delay-removed", removed)
    seconds = (
        ctx.host_summary * 2 + ctx.host_interval * 6 + RECOVERY_LOAD_MARGIN_SECONDS
    )
    writer = _run_io(ctx, seconds, threads=1, pace_ms=RECOVERY_LOAD_PACE_MS)
    return {"delay_removed": removed, "writer": writer}


def sustained_util(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    # dm-delay 20 ms keeps util high at tiny throughput so the backing disk is
    # barely touched; without a loaded dm_delay the plain linear device (this
    # variant loads no module; it may read INCONCLUSIVE on fast NVMe, a
    # distinct, honest outcome). The 2026-10-24/25 TimeoutExpired was the
    # Pod-side dmsetup udev wait (see the storage probe), not the delay target.
    delay = 20 if _dm_delay_loaded(ctx) else 0
    return _io_metric_variant(ctx, metric="disk_io_util_percent", delay_ms=delay)


def sustained_latency(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    module = (survey.get("modules") or {}).get("dm_delay") or {}
    if not module.get("loaded") and "KERNEL_MODULE_LOAD" not in ctx.settings.approvals:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "dm_delay is not loaded; load it with KERNEL_MODULE_LOAD or use the replay",
        )
    # A missing dm_delay is loaded for this run (``dm-create --load-module``)
    # and unloaded by the device cleanup. Before, nothing loaded it and the
    # probe refused the delay table ("dm_delay is not loaded") -- consistent
    # with the earlier live "dm-delay injector ProbeError".
    return _io_metric_variant(ctx, metric="disk_io_await_ms", delay_ms=200)


def short_flush(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device = _device_name(created)
    started = utc_now()
    _run_io(ctx, SHORT_FLUSH)
    ctx.hold(SHORT_FLUSH + 10, name="flush")
    _stop_io(ctx)
    observed = _watch_no_finding(
        ctx, started, "disk_io_util_percent", IO_SUSTAIN + 420, device=device
    )
    stages = {
        "no_finding": chain_errors(
            observed,
            ChainExpectation(
                metric="disk_io_util_percent", device=device, present=False
            ),
        )
    }
    return _result(stages, extra={"device": device})


def zero_completed_io(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device = _device_name(created)
    started = utc_now()
    oracle = Oracle(ctx, "disk")
    oracle.start(180, interval=2, diskstats=[device])
    _run_io(ctx, 180)
    # Registered before the suspend: a lost answer still resumes the device
    # before the writer is stopped (LIFO), as the probe's deadman would.
    ctx.cleanup.push(
        "dm-resume", lambda: _device_call(ctx, "dm-resume", "--name", TEST_DM)
    )
    suspended = _device_call(ctx, "dm-suspend", "--name", TEST_DM, "--seconds", "90")
    ctx.evidence.write("dm-suspend", suspended)
    ctx.hold(90, name="suspend")
    _device_call(ctx, "dm-resume", "--name", TEST_DM)
    ctx.cleanup.discard("dm-resume")
    _stop_io(ctx)
    oracle.read()
    # No await sample may be emitted with zero completions (Delta ios == 0); the
    # product must not fabricate a rate.
    observed = activity(
        ctx, started, metrics={"disk_io_await_ms", "disk_io_util_percent"}
    )
    await_samples = [
        sample
        for sample in samples(observed, "disk_io_await_ms", device=device)
        if during_suspend(sample, suspended)
    ]
    stages = {
        "no_pseudo_rate": [
            f"an await sample was emitted with zero completed I/O: {await_samples}"
        ]
        if await_samples
        else [],
    }
    return _result(
        stages,
        notes=["writers in D state raise loadavg while suspended"],
        extra={"device": device},
    )


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device_before = _device_name(created)
    started = utc_now()
    _run_io(ctx, 60)
    ctx.hold(30, name="pre-reset")
    # The writer's open files keep the test mount busy: stop it, remove and
    # recreate the mapping (its counters restart), then write again so the
    # collector's next delta spans the reset.
    _stop_io(ctx)
    recreated = _device_call(ctx, "dm-recreate", "--name", TEST_DM)
    ctx.evidence.write("dm-recreate", recreated)
    _run_io(ctx, 60)
    ctx.hold(30, name="post-reset")
    _stop_io(ctx)
    observed = activity(
        ctx, started, metrics={"disk_io_util_percent", "disk_io_await_ms"}
    )
    # The interval spanning the restart is re-baselined (no sample, batch 172);
    # no pseudo-rate and no hardware recovery.
    stages = {"hardware": hardware_errors(observed)}
    return _result(
        stages,
        notes=[
            "the interval spanning the counter restart produces no sample (batch 172 "
            "re-baselines); recorded as an observation"
        ],
        extra={"device_before": device_before, "recreated": recreated},
    )


#: Below this standard await the weighted/standard comparison cannot
#: discriminate: ``_await_divergence`` tolerates 5 ms, so sub-5 ms awaits read
#: "no divergence" whatever formula the product uses.
AWAIT_TOLERANCE_FLOOR_MS = 5.0


def reference_iostat(ctx: VariantContext) -> dict[str, Any]:
    """Compare the product's await with the standard await of the same interval.

    The oracle reads ``/proc/diskstats`` itself and computes iostat's await
    (fields 4+8 over 1+5); a divergence beyond 10 % or 5 ms is the spec FAIL.
    Expected PASS since batch 172, when the product stopped dividing the
    weighted field 11 by the completions.

    The comparison needs awaits above the 5 ms tolerance floor, so the test
    device gets the 200 ms dm-delay table -- with a loaded ``dm_delay``, or one
    loaded for the run under the optional ``KERNEL_MODULE_LOAD``. A plain
    linear device on NVMe reads sub-millisecond awaits that no formula
    difference can lift past the floor: 2026-10-25 on ip-10-0-43-5 the
    product's 0.30-0.44 ms against the oracle's 0.33-0.76 ms read PASS on this
    expected-FAIL variant, proving nothing; that reading is INCONCLUSIVE now.
    """

    run_survey(ctx)  # whether dm_delay is loaded is the survey's answer
    delayed = _dm_delay_loaded(ctx) or "KERNEL_MODULE_LOAD" in ctx.settings.approvals
    created = _create_dm(ctx, delay_ms=200 if delayed else 0)
    device = _device_name(created)
    started = utc_now()
    oracle = Oracle(ctx, "disk")
    oracle.start(IO_BUDGET, interval=1, diskstats=[device])
    _run_io(ctx, IO_BUDGET)
    detection = wait_for_incident(
        ctx,
        started,
        "disk_io_await_ms",
        device=device,
        timeout_seconds=IO_BUDGET,
        metrics={"disk_io_await_ms"},
        name="iostat-detection",
    )
    oracle.read()
    product = [
        sample["value"]
        for sample in samples(detection, "disk_io_await_ms", device=device)
        if isinstance(sample.get("value"), (int, float))
    ]
    standard = oracle_values(
        [record.get("diskstats", {}).get(device, {}) for record in oracle.records],
        lambda record: record.get("standard_await_ms"),
    )
    _stop_io(ctx)
    # Before any verdict: the device's await signal is cleared with the delay
    # out and a paced writer (``_recovery_load``). Live 2026-10-25 eksb this
    # variant (14:56-15:05Z) minted an await incident at 15:03:10Z and tore
    # the device down with no below-threshold sample ever delivered; the
    # signal stayed active and notified until 17:14:22Z and the next
    # sustained_latency on dm-0 had its finding withheld as the same notified
    # episode (store/shared/xid.py _next_health_signal_state).
    recovery_load = _recovery_load(ctx, "disk_io_await_ms", 200 if delayed else 0)
    recovered = _await_io_recovery(ctx, "disk_io_await_ms", device, 100.0)
    _stop_io(ctx)
    if not product or not standard:
        # Nothing to compare is not "no divergence": the oracle or the product
        # produced no await sample for the device.
        raise VariantInconclusive(
            f"no await samples to compare (product {len(product)}, "
            f"oracle {len(standard)})"
        )
    median = _median(standard) or 0.0
    if median < AWAIT_TOLERANCE_FLOOR_MS:
        raise VariantInconclusive(
            f"the standard await stayed under the {AWAIT_TOLERANCE_FLOOR_MS:g} ms "
            f"tolerance floor (median {median:.2f} ms): the weighted/standard "
            "comparison cannot discriminate; run with --approve KERNEL_MODULE_LOAD "
            "so the dm-delay table lifts the awaits"
        )
    diverged = _await_divergence(product, standard)
    stages = {
        "await_definition": [
            "the product's await diverges from the standard await of the same "
            "device and interval beyond tolerance ((ms_read+ms_write)/ios)"
        ]
        if diverged
        else [],
        # A signal the product does not clear on fresh below-threshold samples
        # is its defect, and would mask the next run's finding on this node.
        "signal_cleared": recovered["errors"],
    }
    return _result(
        stages,
        extra={
            "delay_ms": 200 if delayed else 0,
            "product_await_ms": product[-5:],
            "standard_await_ms": standard[-5:],
            "diverged": diverged,
            "recovery": recovered,
            "recovery_load": recovery_load,
        },
    )


def io_recovery(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device = _device_name(created)
    started = utc_now()
    hold = io_detection_hold(ctx)
    oracle = Oracle(ctx, "disk")
    oracle.start(IO_RAMP_BUDGET + hold + 240, interval=2, diskstats=[device])
    load = _run_io(ctx, IO_RAMP_BUDGET + hold + 240)
    ctx.evidence.write(
        "io-load",
        {
            **load,
            "ramp_budget_seconds": IO_RAMP_BUDGET,
            "detection_budget_seconds": hold,
            "host_summary_seconds": ctx.host_summary,
            "host_interval_seconds": ctx.host_interval,
        },
    )
    crossing = _await_crossing(
        ctx,
        oracle,
        metric="disk_io_util_percent",
        device=device,
        key="util_percent",
        threshold=98.0,
    )
    detection = wait_for_incident(
        ctx,
        started,
        "disk_io_util_percent",
        device=device,
        timeout_seconds=hold,
        metrics={"disk_io_util_percent"},
        name="io-recovery-detection",
    )
    incidents = finding_incidents(detection, "disk_io_util_percent", device=device)
    _stop_io(ctx)
    recovered = _await_io_recovery(ctx, "disk_io_util_percent", device, 98.0)
    if incidents:
        wait_settled(
            ctx,
            started,
            [item["incident_id"] for item in incidents],
            timeout_seconds=IO_BUDGET,
            metrics={"disk_io_util_percent"},
            name="io-recovery-settled",
        )
    stages = {"recovery": recovered["errors"], "hardware": hardware_errors(detection)}
    if not incidents:
        stages["recovery"].append("no disk I/O finding to recover from")
    return _result(
        stages,
        extra={
            "device": device,
            "crossing": crossing,
            "detection_budget_seconds": hold,
            "recovery": recovered,
        },
    )


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def hardware_only() -> frozenset[str]:
    from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

    return HARDWARE_OPERATIONS


def during_suspend(sample: dict[str, Any], suspended: dict[str, Any]) -> bool:
    return True  # the activity window begins at the suspend; every sample is in it


def _dm_delay_loaded(ctx: VariantContext) -> bool:
    return bool((ctx.survey.get("modules") or {}).get("dm_delay", {}).get("loaded"))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    import statistics

    return float(statistics.median(values))


def _await_divergence(product: list[float], standard: list[float]) -> bool:
    if not product or not standard:
        return False
    product_median = _median(product) or 0.0
    standard_median = _median(standard) or 0.0
    return abs(product_median - standard_median) > max(
        AWAIT_TOLERANCE_FLOOR_MS, 0.10 * standard_median
    )


def _await_io_recovery(
    ctx: VariantContext, metric: str, device: str, threshold: float
) -> dict[str, Any]:
    return await_metric_recovery(ctx, metric, threshold, device=device)


def _watch_no_finding(
    ctx: VariantContext,
    since: Any,
    metric: str,
    window_seconds: int,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    last = activity(ctx, since, metrics={metric})

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={metric})
        return last if finding_incidents(last, metric, device=device) else None

    hit = ctx.wait_until(
        accept,
        timeout_seconds=window_seconds,
        poll_seconds=ctx.host_interval * 2,
        name="negative-window",
    )
    return hit if hit is not None else last


def _statuses(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"request_id": item.get("request_id"), "status": item.get("status")}
        for item in activity_document.get("workflows") or []
    ]


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def _fs_spec(
    variant: str,
    injection: str,
    expected: str,
    cleanup: str,
    execute,
    *,
    outcome: str = "PASS",
    gap: str = "",
    approvals=(),
    optional_approvals=(),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_004,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=18,
        execute=execute,
        expected_outcome=outcome,
        product_gap=gap,
        approvals=tuple(approvals),
        optional_approvals=tuple(optional_approvals),
    )


def _io_spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    max_minutes: int = 18,
    outcome: str = "PASS",
    gap: str = "",
    lanes=("eks", "hp"),
    approvals=(),
    optional_approvals=(),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_006,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk="M",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="stop the writer; remove the test device",
        max_minutes=max_minutes,
        execute=execute,
        expected_outcome=outcome,
        product_gap=gap,
        approvals=tuple(approvals),
        optional_approvals=tuple(optional_approvals),
    )


VARIANTS = [
    _fs_spec(
        "capacity_full",
        "fallocate a fill to 99 % used on a test ext4 loop fs declared in GPU_FAULT_FILESYSTEMS",
        "filesystem_used_percent >= 98 -> MARK_UNSCHEDULABLE, QUARANTINE; no STOP; incident QUARANTINED; validated restore",
        "remove the fill, close the window, validated restore, residual check",
        capacity_full,
    ),
    _fs_spec(
        "below_threshold",
        "fill the test fs to 97 %",
        "no 98 % rule, no QUARANTINE (only the 90 % warning)",
        "remove the fill",
        below_threshold,
    ),
    _fs_spec(
        "inode_full",
        "exhaust inodes with -N 256 while bytes stay < 5 %",
        "filesystem_inodes_used_percent >= 98 -> MARK_UNSCHEDULABLE, QUARANTINE; "
        "no STOP; no filesystem_used_percent finding; incident QUARANTINED; "
        "validated restore",
        "remove the files, close the window, validated restore",
        inode_full,
    ),
    _fs_spec(
        "read_only",
        "remount the test fs read-only",
        "filesystem_unavailable (READ_ONLY) -> FREEZE_EVIDENCE, VALIDATE_HOST; no "
        "capacity finding, no reset/reboot/quarantine (batch 163: ST_RDONLY is read "
        "and graded); readmit on remount rw",
        "remount rw",
        read_only,
    ),
    _fs_spec(
        "required_mount_missing",
        "umount the declared test fs (and rmdir)",
        "statvfs ENOENT -> filesystem_unavailable (MOUNT_MISSING) -> VALIDATE_HOST; "
        "no reset/reboot/quarantine (batch 163: a vanished configured mount is "
        "reported, not skipped); readmit on remount",
        "remount",
        required_mount_missing,
    ),
    _fs_spec(
        "probe_error",
        "a declared FAT16 test mount over loop + dm whose table is swapped to the "
        "dm_mod error target under a fresh mount: statvfs EIO while mounted rw "
        "(KERNEL_MODULE_LOAD loads a missing vfat; the optional "
        "TEMP_TEST_PACKAGE_INSTALL installs a missing mkfs.vfat as dosfstools for "
        "the run; the node log collector must be off)",
        "filesystem_unavailable (PROBE_FAILED) -> FREEZE_EVIDENCE, VALIDATE_HOST; "
        "no capacity finding, no reset/reboot/quarantine (batch 163: an "
        "unreadable status is unusable, never healthy); readmit once the data "
        "table is back",
        "data table back, close the window, destroy the device, unload vfat if loaded "
        "here, remove dosfstools if installed here",
        probe_error,
        approvals=("KERNEL_MODULE_LOAD",),
        optional_approvals=(PACKAGE_APPROVAL,),
    ),
    _fs_spec(
        "duplicate",
        "hold 99 % across >= 2 health summaries plus one collector restart",
        "a single incident and a single workflow; no second QUARANTINE",
        "remove the fill, validated restore",
        duplicate,
    ),
    _io_spec(
        "sustained_util",
        "dm-linear/dm-delay over a loop file, 16 O_DIRECT writers; detection is "
        "timed from the oracle's crossing over two host health summaries",
        "disk_io_util_percent >= 98 for 120 s -> VALIDATE_HOST on dm-N; no reset; "
        "fresh samples below threshold once the writer stops",
        sustained_util,
        max_minutes=45,
    ),
    _io_spec(
        "sustained_latency",
        "dm-delay 200 ms with queued writers; detection is timed from the oracle's "
        "crossing over two host health summaries",
        "disk_io_await_ms >= 100 -> VALIDATE_HOST; a missing dm_delay is loaded "
        "for the run (KERNEL_MODULE_LOAD) and unloaded in cleanup; the delay is "
        "taken out and a paced writer proves the await recovered on the device",
        sustained_latency,
        max_minutes=45,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _io_spec(
        "short_flush",
        "45 s burst (< the 120 s sustain)",
        "no finding across a >= 540 s window",
        short_flush,
        max_minutes=14,
    ),
    _io_spec(
        "zero_completed_io",
        "dmsetup suspend with outstanding writes; resume by deadman <= 120 s",
        "util may read 100 but no await sample (Delta ios == 0); no pseudo-rate",
        zero_completed_io,
        max_minutes=10,
    ),
    _io_spec(
        "counter_reset",
        "dmsetup remove and recreate under the same name/minor",
        "the interval spanning the restart produces no sample (re-baselined); no "
        "pseudo-rate, no hardware recovery",
        counter_reset,
        max_minutes=10,
    ),
    _io_spec(
        "reference_iostat",
        "dm-delay 200 ms (dm_delay loaded, or loaded for the run under the optional "
        "KERNEL_MODULE_LOAD); sample /proc/diskstats at 1 s and compute the standard "
        "await vs the product's await",
        "the product's await is iostat's await (fields 4+8 over 1+5, batch 172) and "
        "agrees with the oracle within 10 % / 5 ms; a divergence beyond that is the spec "
        "FAIL; awaits under the 5 ms tolerance floor (no dm-delay) are INCONCLUSIVE",
        reference_iostat,
        max_minutes=30,
        optional_approvals=("KERNEL_MODULE_LOAD",),
    ),
    _io_spec(
        "recovery",
        "stop the writer (detection timed from the oracle's crossing over two "
        "host health summaries)",
        "fresh samples below threshold; validation passes; data intact",
        io_recovery,
        max_minutes=45,
    ),
]
