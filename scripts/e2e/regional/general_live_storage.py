"""GF-GENERAL-LIVE-004 local disk full.

The 004 variants use a *test* file system of ``general_live_storage_probe.py``,
mounted under ``/mnt/gfrs-live/<digest>/<name>`` and added to the host
collector's ``GPU_FAULT_FILESYSTEMS`` list through a collector-window drop-in.
The system disk, ``/var`` and the log disk are never touched.

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
as the last LIFO cleanup, after the device is destroyed.

006 disk I/O saturation lives in the sibling module
``general_live_storage_io`` together with the device-mapper toolkit
(``dm-create`` / ``_create_dm``, ``_device_call``) that ``probe_error`` borrows
for its FAT16 carrier and the verdict helpers both categories share; its
``IO_VARIANTS`` are registered through this module's ``VARIANTS``, so the
registry sees one storage category with the same keys as before the split.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    await_metric_recovery,
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
    samples,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.general_live_storage_io import (
    IO_VARIANTS,
    TEST_DM,
    _create_dm,
    _device_call,
    _digest,
    _result,
    _statuses,
    _watch_no_finding,
    hardware_only,
)


CASE_004 = "GF-GENERAL-LIVE-004"
QUARANTINE_BUDGET = 420
TEST_FS = "fs"
#: The 004 ``probe_error`` faulting device (FAT16 over loop + dm).
TEST_ERR = "err"
ERR_SIZE_MIB = 256
LOG_COLLECTOR_UNIT = "gpu-fault-log-collector.service"


def _fs_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_FS}"


def _dm_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_DM}"


def _err_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_ERR}"


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
    *IO_VARIANTS,
]
