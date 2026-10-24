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
once the mount is restored; ``probe_error`` stays live-deferred for want of a
faulting-mount fixture. 006
``disk_io_util_percent``/``disk_io_await_ms`` are WARNING
RUN_DIAGNOSTICS with a 120 s sustain; the oracle's own ``/proc/diskstats`` read
computes the standard await to compare with the product's weighted-time one.
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
    deferred_variant,
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

CASE_004 = "GF-GENERAL-LIVE-004"
CASE_006 = "GF-GENERAL-LIVE-006"
QUARANTINE_BUDGET = 420
IO_SUSTAIN = 120
IO_BUDGET = IO_SUSTAIN + 300 + 90
SHORT_FLUSH = 45
TEST_FS = "fs"
TEST_DM = "io"
#: Why the dm-delay-backed 006 I/O-load signal is NOT_RUN(FIXTURE_NOT_READY).
#: The Bidirectional host-mount fix (host_probe_fixture) lets the ``systemd-run``
#: writer reach the test dm device instead of the root fs, but the live run still
#: showed the native dm-I/O load path failing independently of mount visibility
#: (sustained_util: host probe TimeoutExpired; sustained_latency: the 200 ms
#: dm-delay injector raised ProbeError). Making it reliable needs a real
#: dm/dm-delay device this harness cannot exercise (the unit tests fake the
#: probe), so the dm-delay-backed signal is deferred rather than blind-fixed.
DM_IO_LOAD_DEFERRED = (
    "the native dm-I/O load path live-failed (sustained_util host probe "
    "TimeoutExpired, sustained_latency dm-delay injector ProbeError); the "
    "Bidirectional host mount lets the writer reach the dm device, but the "
    "long-running O_DIRECT writer unit and the dm-delay injector still need a "
    "real dm/dm-delay device this harness cannot exercise, so the dm-delay-backed "
    "signal is deferred rather than given a blind, unverifiable fix"
)


def _fs_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_FS}"


def _dm_mount(ctx: VariantContext) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{TEST_DM}"


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


def _declare_fs(ctx: VariantContext) -> None:
    """Add the test mount to GPU_FAULT_FILESYSTEMS through a window drop-in."""

    open_window(
        ctx,
        f"GPU_FAULT_FILESYSTEMS={_fs_mount(ctx)}",
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
    """

    survey = run_survey(ctx)
    require_tools(survey, "losetup", "mkfs.ext4", "mount", "fallocate")
    mount = _fs_mount(ctx)
    _create_fs(ctx)
    _declare_fs(ctx)
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
    return _result(stages, extra={"injection": injected, "recovery": recovered})


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
    raise VariantInconclusive(f"unknown unusable-filesystem variant {variant!r}")


def _restore_unusable(ctx: VariantContext, variant: str) -> None:
    if variant == "read_only":
        ctx.storage("fs-remount", "--name", TEST_FS, "--mode", "rw")
        ctx.cleanup.discard("fs-rw")
    elif variant == "required_mount_missing":
        ctx.storage("fs-mount", "--name", TEST_FS)
        ctx.cleanup.discard("fs-mount")


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


# --------------------------------------------------------------------------- #
# 006 disk I/O
# --------------------------------------------------------------------------- #
def _create_dm(
    ctx: VariantContext, *, delay_ms: int = 0, size_mib: int = 2048
) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(
        survey, "losetup", "dmsetup", "mkfs.ext4", "mount", "blockdev", "fallocate"
    )
    require_module(survey, "dm_mod", ctx.settings.approvals)
    if delay_ms:
        require_module(survey, "dm_delay", ctx.settings.approvals)
    root = backing_root(survey)
    created = ctx.storage(
        "dm-create",
        "--name",
        TEST_DM,
        "--size-mib",
        str(size_mib),
        "--delay-ms",
        str(delay_ms),
        "--backing",
        root,
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
        timeout=600,
    )
    ctx.cleanup.push("dm", lambda: ctx.storage("dm-destroy", "--name", TEST_DM))
    ctx.evidence.write("dm-create", created)
    return created


def _device_name(created: dict[str, Any]) -> str:
    device = created.get("device")
    if not device:
        raise VariantInconclusive("the test device has no dm-N name")
    return str(device)


def _run_io(
    ctx: VariantContext, seconds: int, *, direct: bool = True, threads: int = 16
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
    started = ctx.storage(*arguments)
    ctx.cleanup.push("io", lambda: ctx.storage("io-stop", "--name", TEST_DM))
    return started


def _io_metric_variant(
    ctx: VariantContext, *, metric: str, delay_ms: int
) -> dict[str, Any]:
    created = _create_dm(ctx, delay_ms=delay_ms)
    device = _device_name(created)
    started = utc_now()
    oracle = Oracle(ctx, "disk")
    oracle.start(IO_BUDGET + 120, interval=2, diskstats=[device])
    hold = IO_BUDGET + 180
    _run_io(ctx, hold)
    detection = wait_for_incident(
        ctx,
        started,
        metric,
        device=device,
        timeout_seconds=IO_BUDGET,
        metrics={metric},
        name=f"{metric}-detection",
    )
    oracle.read()
    key = "util_percent" if metric == "disk_io_util_percent" else "weighted_await_ms"
    threshold = 98.0 if metric == "disk_io_util_percent" else 100.0
    values = oracle_values(
        [record.get("diskstats", {}).get(device, {}) for record in oracle.records],
        lambda record: record.get(key),
    )
    if not sustained(values, lambda value: value >= threshold, IO_SUSTAIN // 2):
        raise VariantInconclusive(
            f"the {metric} oracle never sustained >= {threshold}: {values[-8:]}"
        )
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
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
    recovered = _await_io_recovery(ctx, metric, device, threshold)
    stages["recovery"] = recovered["errors"]
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=IO_BUDGET,
        metrics={metric},
        name=f"{metric}-settled",
    )
    return _result(stages, extra={"device": device, "recovery": recovered})


def sustained_util(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    # The reliable util signal uses a dm-delay device (20 ms keeps util high at
    # tiny throughput so the backing disk is barely touched). That native
    # dm-I/O load path live-failed and cannot be exercised here, so when dm_delay
    # is in play the variant is NOT_RUN(FIXTURE_NOT_READY); without dm_delay the
    # plain linear device is still attempted (it may read INCONCLUSIVE on fast
    # NVMe, which is a distinct, honest outcome).
    if (survey.get("modules") or {}).get("dm_delay", {}).get("loaded"):
        raise VariantNotRun("FIXTURE_NOT_READY", DM_IO_LOAD_DEFERRED)
    return _io_metric_variant(ctx, metric="disk_io_util_percent", delay_ms=0)


def sustained_latency(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    module = (survey.get("modules") or {}).get("dm_delay") or {}
    if not module.get("loaded") and "KERNEL_MODULE_LOAD" not in ctx.settings.approvals:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "dm_delay is not loaded; load it with KERNEL_MODULE_LOAD or use the replay",
        )
    # dm_delay is available, but the 200 ms dm-delay injector live-failed on the
    # native dm-I/O load path; deferred rather than given a blind, unverifiable
    # fix (see DM_IO_LOAD_DEFERRED).
    raise VariantNotRun("FIXTURE_NOT_READY", DM_IO_LOAD_DEFERRED)


def short_flush(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device = _device_name(created)
    started = utc_now()
    _run_io(ctx, SHORT_FLUSH)
    ctx.hold(SHORT_FLUSH + 10, name="flush")
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
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
    suspended = ctx.storage("dm-suspend", "--name", TEST_DM, "--seconds", "90")
    ctx.cleanup.push("dm-resume", lambda: ctx.storage("dm-resume", "--name", TEST_DM))
    ctx.evidence.write("dm-suspend", suspended)
    ctx.hold(90, name="suspend")
    ctx.storage("dm-resume", "--name", TEST_DM)
    ctx.cleanup.discard("dm-resume")
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
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
    recreated = ctx.storage("dm-recreate", "--name", TEST_DM)
    ctx.evidence.write("dm-recreate", recreated)
    ctx.hold(30, name="post-reset")
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
    observed = activity(
        ctx, started, metrics={"disk_io_util_percent", "disk_io_await_ms"}
    )
    # The negative delta is clamped to 0; no pseudo-rate and no hardware recovery.
    stages = {"hardware": hardware_errors(observed)}
    return _result(
        stages,
        notes=[
            "counter reset is invisible after the clamp (X5); recorded as an observation"
        ],
        extra={"device_before": device_before, "recreated": recreated},
    )


def reference_iostat(ctx: VariantContext) -> dict[str, Any]:
    """Compare the product's weighted await with the standard await (spec FAIL)."""

    created = _create_dm(ctx, delay_ms=200 if _dm_delay_loaded(ctx) else 0)
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
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
    diverged = _await_divergence(product, standard)
    stages = {
        "await_definition": [
            "the product's weighted await diverges from the standard await beyond "
            "tolerance (weighted in-flight time / ios, not (ms_read+ms_write)/ios)"
        ]
        if diverged
        else []
    }
    return _result(
        stages,
        extra={
            "product_await_ms": product[-5:],
            "standard_await_ms": standard[-5:],
            "diverged": diverged,
        },
    )


def io_recovery(ctx: VariantContext) -> dict[str, Any]:
    created = _create_dm(ctx)
    device = _device_name(created)
    started = utc_now()
    oracle = Oracle(ctx, "disk")
    oracle.start(IO_BUDGET, interval=2, diskstats=[device])
    _run_io(ctx, IO_BUDGET)
    detection = wait_for_incident(
        ctx,
        started,
        "disk_io_util_percent",
        device=device,
        timeout_seconds=IO_BUDGET,
        metrics={"disk_io_util_percent"},
        name="io-recovery-detection",
    )
    incidents = finding_incidents(detection, "disk_io_util_percent", device=device)
    ctx.storage("io-stop", "--name", TEST_DM)
    ctx.cleanup.discard("io")
    recovered = _await_io_recovery(ctx, "disk_io_util_percent", device, 98.0)
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
    return _result(stages, extra={"device": device, "recovery": recovered})


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


def _await_divergence(product: list[float], standard: list[float]) -> bool:
    if not product or not standard:
        return False
    import statistics

    product_median = statistics.median(product)
    standard_median = statistics.median(standard)
    return abs(product_median - standard_median) > max(5.0, 0.10 * standard_median)


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
    deferred_variant(
        CASE_004,
        "probe_error",
        proof_mode="live-replay",
        reason=(
            "FIXTURE_NOT_READY: the product reports a statvfs EIO/EACCES as "
            "filesystem_unavailable (PROBE_FAILED) since batch 163, but injecting a "
            "real one live needs a faulting test mount (FUSE/dm-error) not built this "
            "phase; a read-only file bind cannot (statvfs resolves the mount, not the "
            "bound file)"
        ),
        lanes=("eks",),
        expected=(
            "a configured mount whose statvfs raises is unusable -> "
            "filesystem_unavailable (PROBE_FAILED) -> VALIDATE_HOST (batch 163)"
        ),
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
        "dm-linear/dm-delay over a loop file, 16-32 O_DIRECT writers for >= 8 min",
        "disk_io_util_percent >= 98 for 120 s -> VALIDATE_HOST on dm-N; no reset",
        sustained_util,
        max_minutes=20,
    ),
    _io_spec(
        "sustained_latency",
        "dm-delay 200 ms with queued writers",
        "disk_io_await_ms >= 100 -> VALIDATE_HOST; without dm_delay NOT_RUN(CAPABILITY_ABSENT)",
        sustained_latency,
        max_minutes=20,
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
        "the negative delta is clamped to 0; no pseudo-rate, no hardware recovery",
        counter_reset,
        max_minutes=10,
    ),
    _io_spec(
        "reference_iostat",
        "sample /proc/diskstats at 1 s and compute the standard await vs the product's weighted await",
        "the product uses weighted time / ios; a divergence > 10 % or 5 ms is the spec FAIL",
        reference_iostat,
        max_minutes=20,
        outcome="FAIL",
        gap="disk_io_await_ms uses the weighted in-flight time (field 11) / ios, not the standard await",
    ),
    _io_spec(
        "recovery",
        "stop the writer",
        "fresh samples below threshold; validation passes; data intact",
        io_recovery,
        max_minutes=18,
    ),
]
