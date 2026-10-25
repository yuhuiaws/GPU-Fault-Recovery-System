"""GF-GENERAL-LIVE-005 shared storage unavailable / full.

The host collector grades shared storage in ``collectors/host/system_metrics.py``
``_shared_filesystems``: every ``nfs``/``nfs4``/``lustre`` mount it finds in its
own ``/proc/self/mountinfo`` -- there is no shared-filesystem setting; a host
mount propagates into the unit's namespace -- gets a bounded ``statvfs`` (5 s per
mount, 15 s per tick) and is reported as ``shared_filesystem_unavailable`` 1
(CRITICAL RUN_DIAGNOSTICS: FREEZE_EVIDENCE, VALIDATE_HOST) on a timeout or
error, else 0 plus ``shared_filesystem_used_percent`` (98 % WARNING
RUN_DIAGNOSTICS). ``GPU_FAULT_FILESYSTEMS`` is deliberately *not* extended with
the test export: a configured mount runs the local-disk rules
(``filesystem_used_percent`` 98 % QUARANTINE, MOUNT_MISSING), which would turn a
shared-storage condition into a bad-node verdict and mask the gaps 005 exists
to show.

The fixture (``shared_export_fixture.TestExportServer`` + the node probe
``probes/general_live_shared_probe.py``) stands up a disposable NFSv4 export --
a userspace server Pod over a size-limited tmpfs -- and mounts it ``soft`` under
``/mnt/gfrs-live/<digest>/<name>`` in the client node's host mount namespace;
``SIGSTOP`` on the server hangs every call, ``SIGCONT`` restores the same file
handles. Verdicts read only the product's records; the probe's bounded
``statvfs``/I/O readings are the oracle.

Expected FAILs (product gaps, registered as such): ``mount_disappears`` -- the
collector keeps no memory of shared mounts it discovered, so one that vanishes
simply stops being sampled and is never reported unavailable;
``io_only_failure`` -- the mount is judged by ``statvfs`` alone, no bounded
read/open/fsync probe; ``shared_failure_domain`` -- two nodes on one export mint
two independent node incidents and nothing records the common failure domain.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_common import run_survey
from scripts.e2e.regional.general_live_fixture import (
    GeneralLiveSettings,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    node_provider,
    require_module,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_store import node_activity, signal_key
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    parse_time,
    samples,
    settled,
    verdict,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.shared_export_fixture import (
    ExportServerSettings,
    TestExportServer,
)

CASE = "GF-GENERAL-LIVE-005"
UNAVAILABLE = "shared_filesystem_unavailable"
USED = "shared_filesystem_used_percent"
#: A host metric every tick carries: proof the collector kept delivering while
#: one mount hung.
COMPANION_METRIC = "memory_used_percent"
ACTIVITY_METRICS = frozenset({UNAVAILABLE, USED, COMPANION_METRIC})
PRIMARY = "shared"
OTHER = "other"
DETECTION_BUDGET = 420
EXPORT_READY_SECONDS = 600
STORE_POLL_SECONDS = 20
FILL_PERCENT = 99.0
#: Isolation is never right for a shared-storage condition.
ISOLATION = frozenset({"STOP_WORKLOADS", "MARK_UNSCHEDULABLE", "QUARANTINE"})
FORBIDDEN = ISOLATION | HARDWARE_OPERATIONS
GAP_MOUNT_DISAPPEARS = (
    "collectors/host/system_metrics.py HostSystemMetricsMixin._shared_filesystems "
    "rediscovers shared mounts from /proc/self/mountinfo every tick and keeps no "
    "memory of them: a shared mount that vanishes stops being sampled, "
    "shared_filesystem_unavailable is never raised for it and its health signal "
    "keeps the last value 0"
)
GAP_IO_ONLY = (
    "collectors/host/system_metrics.py HostSystemMetricsMixin._shared_filesystems "
    "judges a shared mount by statvfs alone; there is no bounded read/open/fsync "
    "probe of the export, so an export whose data path fails while FSSTAT answers "
    "is reported healthy (shared_filesystem_unavailable 0)"
)
GAP_FAILURE_DOMAIN = (
    "host_health.py grades each node's shared_filesystem_unavailable reading into "
    "its own NODE_HEALTH incident; no component correlates the same export path "
    "across nodes into a common failure-domain record (one incident naming both "
    "nodes, or a marker/notification of the shared export), so two task nodes on "
    "one hung export are two unrelated node diagnoses"
)


# --------------------------------------------------------------------------- #
# fixture glue
# --------------------------------------------------------------------------- #
def _digest(ctx: VariantContext) -> str:
    return hashlib.sha256(ctx.run_id.encode()).hexdigest()[:12]


def mount_path(ctx: VariantContext, name: str) -> str:
    return f"/mnt/gfrs-live/{_digest(ctx)}/{name}"


def _survey(ctx: VariantContext, node: str) -> dict[str, Any]:
    """The capability survey of ``node`` (the target's is the context's)."""

    if node == ctx.settings.node:
        return ctx.survey or run_survey(ctx)
    value = ctx.probe("node", node=node).execute("survey", "--run-id", ctx.run_id)
    ctx.evidence.write(f"survey-{node}", value)
    return value


def _require_client(ctx: VariantContext, node: str) -> bool:
    """The node can mount NFSv4; whether ``nfsv4`` must be loaded for it."""

    survey = _survey(ctx, node)
    require_tools(survey, "mount.nfs4", "mount", "umount", "nsenter", "systemd-run")
    return require_module(survey, "nfsv4", ctx.settings.approvals)


def _require_peer(ctx: VariantContext) -> str:
    peer = ctx.settings.peer_node
    if not peer or peer == ctx.settings.node:
        raise VariantNotRun(
            "FIXTURE_NOT_READY", "shared_failure_domain needs a distinct --node-b"
        )
    metadata = ctx.regional.node_metadata(peer)
    provider = node_provider(metadata.get("labels") or {})
    if provider not in ctx.spec.lanes:
        raise VariantNotRun(
            "FIXTURE_NOT_READY", f"{peer} is a {provider or 'unknown'} node"
        )
    snapshot = ctx.regional.node_snapshot(peer)
    if snapshot.get("ready") != "True" or snapshot.get("unschedulable"):
        raise VariantNotRun("FIXTURE_NOT_READY", f"{peer} is not Ready and schedulable")
    return peer


def _export_image(ctx: VariantContext) -> str:
    image = ctx.settings.export_image
    if not image or "@sha256:" not in image:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            "no immutable test export server image (--export-image or "
            "GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE)",
        )
    return image


def start_export(
    ctx: VariantContext, suffix: str, clients: Iterable[str]
) -> TestExportServer:
    """Create one test export Pod away from ``clients`` and wait until Ready.

    The delete is on the LIFO stack before the create; the Pod's own
    ``activeDeadlineSeconds`` is the cluster-side deadman.
    """

    settings = ExportServerSettings(
        image=_export_image(ctx),
        case_id=ctx.spec.case_id,
        run_id=ctx.run_id,
        suffix=suffix,
        avoid_nodes=tuple(dict.fromkeys(clients)),
        node=ctx.settings.export_node,
        active_deadline_seconds=min(4 * 3600, ctx.spec.max_minutes * 60 + 1200),
    )
    server = TestExportServer(
        ctx.regional,
        settings,
        journal=ctx.case_dir / f"export-{suffix}.json",
        clock=lambda: ctx.clock().timestamp(),
        sleep=ctx.sleep,
    )
    ctx.cleanup.push(f"export-{suffix}", server.delete)
    server.create()
    ready = server.wait_ready(timeout_seconds=EXPORT_READY_SECONDS)
    ctx.evidence.write(f"export-{suffix}", server.describe())
    if ready["node"] in set(clients):
        raise RegionalFixtureError(
            f"the test export {suffix} landed on client node {ready['node']}"
        )
    return server


def mount_export(
    ctx: VariantContext, server: TestExportServer, node: str, name: str
) -> dict[str, Any]:
    """Mount ``server`` on ``node`` under the run's mount root; verify capacity.

    The probe's unmount is on the LIFO stack before the mount, so a lost answer
    is still undone (the probe's deadman unmounts too).
    """

    load = _require_client(ctx, node)
    arguments = [
        "nfs-mount",
        "--name",
        name,
        "--server",
        server.ready["ip"],
        "--export",
        server.ready["export"],
        "--deadman-seconds",
        str(min(5400, ctx.spec.max_minutes * 60 + 600)),
    ]
    if load:
        arguments.append("--load-module")
    ctx.cleanup.push(
        f"mount-{name}@{node}",
        lambda: _checked_unmount(ctx, node, name),
    )
    mounted = ctx.shared(*arguments, node=node, timeout=240)
    ctx.evidence.write(f"mount-{name}-{node}", mounted)
    reading = mounted.get("statvfs") or {}
    total = reading.get("total_bytes")
    expected = server.settings.size_mib * 1024 * 1024
    if (
        not isinstance(total, (int, float))
        or not 0.5 * expected <= total <= 1.5 * expected
    ):
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"the test export's capacity is not bounded ({total} bytes for a "
            f"{server.settings.size_mib} MiB tmpfs: SizeMemoryBackedVolumes off?)",
        )
    return mounted


def _checked_unmount(ctx: VariantContext, node: str, name: str) -> dict[str, Any]:
    value = ctx.shared("cleanup", "--kind", f"nfs-{name}", node=node, timeout=300)
    if value.get("clean") is not True:
        raise RegionalFixtureError(
            f"the shared mount cleanup on {node} left residue: {value.get('residue')}"
        )
    return value


def _stat(
    ctx: VariantContext, node: str, name: str, timeout: float = 5.0
) -> dict[str, Any]:
    value = ctx.shared(
        "nfs-stat", "--name", name, "--timeout", str(timeout), node=node, timeout=120
    )
    ctx.evidence.append_oracle([{"node": node, **value}], source="shared-stat")
    reading: dict[str, Any] = dict(value.get("statvfs") or {})
    return reading


def _pause(
    ctx: VariantContext, server: TestExportServer, suffix: str
) -> dict[str, Any]:
    """Hang the export; the resume is registered first and proven by the oracle."""

    ctx.cleanup.push(f"export-{suffix}-resume", server.resume)
    paused = server.pause()
    ctx.evidence.write(f"export-{suffix}-pause", paused)
    return paused


def _resume(
    ctx: VariantContext, server: TestExportServer, suffix: str
) -> dict[str, Any]:
    resumed = server.resume()
    ctx.cleanup.discard(f"export-{suffix}-resume")
    ctx.evidence.write(f"export-{suffix}-resume", resumed)
    return resumed


def _hung(reading: dict[str, Any]) -> bool:
    return bool(reading.get("timed_out")) or "error" in reading


# --------------------------------------------------------------------------- #
# product records, per node
# --------------------------------------------------------------------------- #
def _activity(
    ctx: VariantContext,
    node: str,
    since: datetime,
    *,
    mount: str | None = None,
    metrics: Iterable[str] = ACTIVITY_METRICS,
) -> dict[str, Any]:
    keys = (
        [signal_key(ctx.settings.regional.cluster_id, node, UNAVAILABLE, mount)]
        if mount
        else []
    )
    return node_activity(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        node,
        since,
        metrics=metrics,
        signal_keys=keys,
    )


def await_baseline(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    """The product discovered the test export: a 0 reading and a used % sample.

    The collector delivers an unchanged normal reading once per health summary,
    so one delivered sample (the mount's first appearance is an edge) is the
    baseline; the spec's three normal samples would cost three summaries.
    """

    def accept() -> dict[str, Any] | None:
        current = _activity(ctx, node, since, mount=mount)
        zero = [
            item
            for item in samples(current, UNAVAILABLE, device=mount)
            if item.get("value") == 0
        ]
        used = samples(current, USED, device=mount)
        return current if zero and used else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=f"baseline-{node}",
    )
    if value is None:
        raise VariantInconclusive(
            f"the product never delivered a healthy {UNAVAILABLE}/{USED} reading "
            f"for {mount} on {node}: the test export was not discovered as a shared mount"
        )
    return value


def wait_finding(
    ctx: VariantContext,
    node: str,
    since: datetime,
    metric: str,
    mount: str,
    *,
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _activity(ctx, node, since, mount=mount)
        return last if finding_incidents(last, metric, device=mount) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    return value if value is not None else last


def wait_quiet(
    ctx: VariantContext, node: str, since: datetime, mount: str, *, name: str
) -> dict[str, Any]:
    """A health summary plus margin without the finding: the gap window."""

    return wait_finding(
        ctx,
        node,
        since,
        UNAVAILABLE,
        mount,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 4,
        name=name,
    )


def await_clear(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    """Fresh product evidence that the mount reads available again."""

    key = signal_key(ctx.settings.regional.cluster_id, node, UNAVAILABLE, mount)

    def accept() -> dict[str, Any] | None:
        current = _activity(ctx, node, since, mount=mount)
        fresh = [
            item
            for item in samples(current, UNAVAILABLE, device=mount)
            if item.get("value") == 0
        ]
        signal = (current.get("health_signals") or {}).get(key) or {}
        observed = parse_time(signal.get("observed_at"))
        cleared = signal.get("active") is False and (
            observed is not None and observed >= since
        )
        if len(fresh) >= 2 or (fresh and cleared):
            return {"samples": fresh, "health_signal": signal}
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=f"recovery-{node}",
    )
    if value is None:
        return {
            "errors": [
                f"{UNAVAILABLE} on {mount} ({node}) did not recover: no fresh 0 "
                "reading with a cleared health signal, nor two fresh readings"
            ]
        }
    return {"errors": [], **value}


def wait_all_settled(
    ctx: VariantContext, node: str, since: datetime, incident_ids: list[str]
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _activity(ctx, node, since)
        return last if settled(last, incident_ids) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=DETECTION_BUDGET,
        poll_seconds=STORE_POLL_SECONDS,
        name=f"settled-{node}",
    )
    return value if value is not None else last


# --------------------------------------------------------------------------- #
# verdict stages
# --------------------------------------------------------------------------- #
def diagnostic_chain(activity: dict[str, Any], metric: str, mount: str) -> list[str]:
    """FREEZE_EVIDENCE + VALIDATE_HOST that fails on the held fault; no isolation."""

    return chain_errors(
        activity,
        ChainExpectation(
            metric=metric,
            device=mount,
            operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
            validation="VALIDATE_HOST",
            validation_must_fail=True,
            probe_metric=metric,
            forbidden=FORBIDDEN,
            incident_states=DIAGNOSTIC_ONLY_STATES,
        ),
    )


def delivery_errors(activity: dict[str, Any], since: datetime) -> list[str]:
    """The collector kept delivering bounded batches while a mount hung."""

    batches = [
        item
        for item in activity.get("batches") or []
        if (moment := parse_time(item.get("observed_at"))) is not None
        and moment >= since
    ]
    with_companion = [
        item
        for item in batches
        if any(
            sample.get("name") == COMPANION_METRIC
            for sample in item.get("samples") or []
        )
    ]
    errors = []
    if len(batches) < 2:
        errors.append(
            f"only {len(batches)} host batch(es) were accepted after the injection"
        )
    if not with_companion:
        errors.append(
            f"no host batch after the injection carried {COMPANION_METRIC}: the hung "
            "mount stalled the collector's other metrics"
        )
    return errors


def other_mount_errors(
    activity: dict[str, Any], other: str, since: datetime
) -> list[str]:
    """The healthy second export stayed reported as available."""

    fresh = [
        item
        for item in samples(activity, UNAVAILABLE, device=other)
        if item.get("value") == 0
        and (moment := parse_time(item.get("observed_at"))) is not None
        and moment >= since
    ]
    return (
        []
        if fresh
        else [
            f"no {UNAVAILABLE}=0 reading for the healthy mount {other} after the hang"
        ]
    )


def common_domain_errors(
    activities: dict[str, dict[str, Any]], nodes: tuple[str, str], mount: str
) -> list[str]:
    """Some product record ties both nodes' export failures together."""

    wanted = set(nodes)
    for activity in activities.values():
        for incident in activity.get("incidents") or []:
            if wanted <= set(incident.get("node_ids") or []):
                return []
        for record in [
            *(activity.get("markers") or []),
            *(activity.get("notifications") or []),
        ]:
            text = str(record)
            if all(node in text for node in nodes) and mount in text:
                return []
    return [
        f"no incident, marker or notification names both {nodes[0]} and {nodes[1]} "
        f"for the shared export {mount}: two unrelated node diagnoses"
    ]


def _incident_ids(activity: dict[str, Any], metric: str, mount: str) -> list[str]:
    return [
        str(item["incident_id"])
        for item in finding_incidents(activity, metric, device=mount)
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


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def _prepare(
    ctx: VariantContext, *, suffix: str = "a", name: str = PRIMARY
) -> tuple[TestExportServer, str, datetime]:
    """Export up, mounted on the target node, discovered by the product."""

    node = ctx.settings.node
    _require_client(ctx, node)
    server = start_export(ctx, suffix, [node])
    since = utc_now()
    mount_export(ctx, server, node, name)
    mount = mount_path(ctx, name)
    baseline = await_baseline(ctx, node, mount, since)
    ctx.evidence.write(f"baseline-{name}", baseline)
    return server, mount, since


def _hang_and_detect(
    ctx: VariantContext, server: TestExportServer, mount: str, *, suffix: str = "a"
) -> tuple[datetime, dict[str, Any]]:
    """Pause the export, prove the hang with the oracle, wait for the finding."""

    started = utc_now()
    _pause(ctx, server, suffix)
    reading = _stat(ctx, ctx.settings.node, PRIMARY)
    if not _hung(reading):
        _resume(ctx, server, suffix)
        raise VariantInconclusive(
            f"the paused export still answered statvfs ({reading}); no hang to inject"
        )
    detection = wait_finding(
        ctx,
        ctx.settings.node,
        started,
        UNAVAILABLE,
        mount,
        timeout_seconds=DETECTION_BUDGET,
        name="hang-detection",
    )
    return started, detection


def stat_timeout(ctx: VariantContext) -> dict[str, Any]:
    server, mount, _since = _prepare(ctx)
    started, detection = _hang_and_detect(ctx, server, mount)
    stages = {
        "chain": diagnostic_chain(detection, UNAVAILABLE, mount),
        "bounded_delivery": delivery_errors(detection, started),
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, UNAVAILABLE, mount)
    _resume(ctx, server, "a")
    recovered = await_clear(ctx, ctx.settings.node, mount, utc_now())
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, ctx.settings.node, started, incidents)
    return _result(stages, extra={"mount": mount, "recovery": recovered})


def capacity_full(ctx: VariantContext) -> dict[str, Any]:
    server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started = utc_now()
    ctx.cleanup.push(
        "unfill", lambda: ctx.shared("nfs-unfill", "--name", PRIMARY, timeout=240)
    )
    filled = ctx.shared(
        "nfs-fill", "--name", PRIMARY, "--percent", str(FILL_PERCENT), timeout=300
    )
    ctx.evidence.write("fill", filled)
    used = (filled.get("statvfs") or {}).get("used_percent")
    if not isinstance(used, (int, float)) or used < 98:
        raise VariantInconclusive(f"the test export only reached {used}% used")
    detection = wait_finding(
        ctx,
        node,
        started,
        USED,
        mount,
        timeout_seconds=DETECTION_BUDGET,
        name="full-detection",
    )
    stages = {
        "chain": diagnostic_chain(detection, USED, mount),
        # Capacity on a shared export is a diagnosis, never an isolation.
        "no_isolation": chain_errors(
            detection,
            ChainExpectation(metric=USED, device=mount, forbidden=FORBIDDEN),
        )
        if finding_incidents(detection, USED, device=mount)
        else [],
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, USED, mount)
    unfilled = ctx.shared("nfs-unfill", "--name", PRIMARY, timeout=240)
    ctx.cleanup.discard("unfill")
    ctx.evidence.write("unfill", unfilled)
    recovered = _await_used_recovery(ctx, node, mount, utc_now())
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
    return _result(
        stages, extra={"mount": mount, "used_percent": used, "recovery": recovered}
    )


def _await_used_recovery(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    def accept() -> dict[str, Any] | None:
        current = _activity(ctx, node, since, mount=mount)
        fresh = [
            item
            for item in samples(current, USED, device=mount)
            if isinstance(item.get("value"), (int, float)) and item["value"] < 98
        ]
        return {"samples": fresh} if fresh else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="used-recovery",
    )
    if value is None:
        return {"errors": [f"{USED} on {mount} never read below 98 % after the unfill"]}
    return {"errors": [], **value}


def mount_disappears(ctx: VariantContext) -> dict[str, Any]:
    _server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started = utc_now()
    gone = ctx.shared("nfs-umount", "--name", PRIMARY, node=node, timeout=240)
    ctx.evidence.write("umount", gone)
    if gone.get("host_mount") is not None:
        raise VariantInconclusive("the test export is still in the host mount table")
    observed = wait_quiet(ctx, node, started, mount, name="vanished-window")
    stages = {
        "vanished_reported": []
        if finding_incidents(observed, UNAVAILABLE, device=mount)
        else [
            f"the vanished shared mount {mount} was never reported {UNAVAILABLE}=1 "
            "(the collector rediscovers shared mounts each tick and forgets one that is gone)"
        ],
        "hardware": hardware_errors(observed),
    }
    signal = (observed.get("health_signals") or {}).get(
        signal_key(ctx.settings.regional.cluster_id, node, UNAVAILABLE, mount)
    )
    return _result(stages, extra={"mount": mount, "last_health_signal": signal})


def io_only_failure(ctx: VariantContext) -> dict[str, Any]:
    server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started = utc_now()
    ctx.cleanup.push("export-a-mode", lambda: server.set_export_mode("1777"))
    denied = server.set_export_mode("000")
    ctx.evidence.write("export-a-mode", denied)
    reading = _stat(ctx, node, PRIMARY)
    io = ctx.shared("nfs-io", "--name", PRIMARY, "--mib", "1", node=node, timeout=240)
    ctx.evidence.append_oracle([{"node": node, **io}], source="shared-io")
    if _hung(reading) or (io.get("io") or {}).get("ok"):
        server.set_export_mode("1777")
        ctx.cleanup.discard("export-a-mode")
        raise VariantInconclusive(
            f"statvfs {reading} / I/O {io.get('io')}: not a statvfs-OK, I/O-failing export"
        )
    observed = wait_quiet(ctx, node, started, mount, name="io-failure-window")
    stages = {
        "io_failure_reported": []
        if finding_incidents(observed, UNAVAILABLE, device=mount)
        else [
            f"no {UNAVAILABLE} finding on {mount} while every read/open failed "
            f"(errno {(io.get('io') or {}).get('errno')}) and statvfs answered: "
            "the collector probes shared mounts with statvfs only"
        ],
        "hardware": hardware_errors(observed),
    }
    restored = server.set_export_mode("1777")
    ctx.cleanup.discard("export-a-mode")
    proof = ctx.shared(
        "nfs-io", "--name", PRIMARY, "--mib", "1", node=node, timeout=240
    )
    stages["io_restored"] = (
        []
        if (proof.get("io") or {}).get("ok")
        else ["I/O did not recover after chmod 1777"]
    )
    return _result(
        stages, extra={"mount": mount, "restored": restored, "io": io.get("io")}
    )


def healthy_other_mount(ctx: VariantContext) -> dict[str, Any]:
    node = ctx.settings.node
    _require_client(ctx, node)
    server_a = start_export(ctx, "a", [node])
    server_b = start_export(ctx, "b", [node])
    since = utc_now()
    mount_export(ctx, server_a, node, PRIMARY)
    mount_export(ctx, server_b, node, OTHER)
    mount, other = mount_path(ctx, PRIMARY), mount_path(ctx, OTHER)
    ctx.evidence.write("baseline-shared", await_baseline(ctx, node, mount, since))
    ctx.evidence.write("baseline-other", await_baseline(ctx, node, other, since))
    started, detection = _hang_and_detect(ctx, server_a, mount)
    healthy = _stat(ctx, node, OTHER)
    if _hung(healthy):
        _resume(ctx, server_a, "a")
        raise VariantInconclusive(
            f"the second export hung too ({healthy}); no healthy control"
        )
    stages = {
        "chain": diagnostic_chain(detection, UNAVAILABLE, mount),
        "other_not_graded": chain_errors(
            detection, ChainExpectation(metric=UNAVAILABLE, device=other, present=False)
        ),
        "other_still_reported": other_mount_errors(detection, other, started),
        "bounded_delivery": delivery_errors(detection, started),
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, UNAVAILABLE, mount)
    _resume(ctx, server_a, "a")
    recovered = await_clear(ctx, node, mount, utc_now())
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
    return _result(
        stages, extra={"mount": mount, "other": other, "recovery": recovered}
    )


def recovery(ctx: VariantContext) -> dict[str, Any]:
    server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started, detection = _hang_and_detect(ctx, server, mount)
    stages = {
        "chain": diagnostic_chain(detection, UNAVAILABLE, mount),
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, UNAVAILABLE, mount)
    resumed_at = utc_now()
    _resume(ctx, server, "a")
    reading = _stat(ctx, node, PRIMARY)
    io = ctx.shared("nfs-io", "--name", PRIMARY, "--mib", "4", node=node, timeout=240)
    ctx.evidence.append_oracle([{"node": node, **io}], source="shared-io")
    stages["fresh_io"] = (
        []
        if not _hung(reading) and (io.get("io") or {}).get("ok")
        else [
            f"no fresh real I/O on the restored export: statvfs {reading}, io {io.get('io')}"
        ]
    )
    recovered = await_clear(ctx, node, mount, resumed_at)
    stages["recovery"] = list(recovered["errors"])
    final = wait_all_settled(ctx, node, started, incidents)
    fresh_used = [
        item
        for item in samples(final, USED, device=mount)
        if (moment := parse_time(item.get("observed_at"))) is not None
        and moment >= resumed_at
    ]
    stages["fresh_samples"] = (
        []
        if fresh_used or recovered.get("samples")
        else [f"no fresh {USED} sample after the resume"]
    )
    return _result(
        stages,
        notes=[
            "no managed training job: task progress is out of this host category's scope"
        ],
        extra={"mount": mount, "recovery": recovered, "io": io.get("io")},
    )


def shared_failure_domain(ctx: VariantContext) -> dict[str, Any]:
    node = ctx.settings.node
    peer = _require_peer(ctx)
    _require_client(ctx, node)
    _require_client(ctx, peer)
    server = start_export(ctx, "a", [node, peer])
    since = utc_now()
    mount_export(ctx, server, node, PRIMARY)
    mount_export(ctx, server, peer, PRIMARY)
    mount = mount_path(ctx, PRIMARY)
    for client in (node, peer):
        ctx.evidence.write(
            f"baseline-{client}", await_baseline(ctx, client, mount, since)
        )
    started = utc_now()
    _pause(ctx, server, "a")
    readings = {client: _stat(ctx, client, PRIMARY) for client in (node, peer)}
    if not all(_hung(item) for item in readings.values()):
        _resume(ctx, server, "a")
        raise VariantInconclusive(
            f"the paused export still answered on a client: {readings}"
        )
    activities = {
        client: wait_finding(
            ctx,
            client,
            started,
            UNAVAILABLE,
            mount,
            timeout_seconds=DETECTION_BUDGET,
            name=f"domain-detection-{client}",
        )
        for client in (node, peer)
    }
    stages: dict[str, list[str]] = {}
    for client, activity in activities.items():
        stages[f"chain@{client}"] = diagnostic_chain(activity, UNAVAILABLE, mount)
        stages[f"hardware@{client}"] = hardware_errors(activity)
    stages["common_failure_domain"] = common_domain_errors(
        activities, (node, peer), mount
    )
    _resume(ctx, server, "a")
    resumed_at = utc_now()
    for client, activity in activities.items():
        recovered = await_clear(ctx, client, mount, resumed_at)
        stages[f"recovery@{client}"] = list(recovered["errors"])
        wait_all_settled(
            ctx, client, started, _incident_ids(activity, UNAVAILABLE, mount)
        )
    return _result(stages, extra={"mount": mount, "nodes": [node, peer]})


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _preflight(settings: GeneralLiveSettings) -> list[str]:
    image = settings.export_image
    if not image or "@sha256:" not in image:
        return [
            "the 005 test export server image is missing (--export-image or "
            "GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE, an immutable digest)"
        ]
    return []


def _pair_preflight(settings: GeneralLiveSettings) -> list[str]:
    errors = _preflight(settings)
    if not settings.peer_node:
        errors.append("shared_failure_domain needs --node-b (a second client node)")
    return errors


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    cleanup: str = "resume the export, unmount the client, delete the export Pod and ConfigMap",
    max_minutes: int = 35,
    flags: str = "S",
    outcome: str = "PASS",
    gap: str = "",
    preflight: Callable[[GeneralLiveSettings], list[str]] = _preflight,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="M",
        flags=flags,
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        approvals=("KERNEL_MODULE_LOAD",),
        execute=execute,
        expected_outcome=outcome,
        product_gap=gap,
        preflight=preflight,
    )


VARIANTS = [
    _spec(
        "stat_timeout",
        "a test NFSv4 export (userspace server Pod over a 256 MiB tmpfs) mounted soft "
        "on the node, discovered by fstype; SIGSTOP the server so every statvfs hangs",
        f"{UNAVAILABLE}=1 -> FREEZE_EVIDENCE, VALIDATE_HOST failing on it; other host "
        "metrics keep arriving (bounded statvfs); no isolation/hardware; 0 again on SIGCONT",
        stat_timeout,
    ),
    _spec(
        "capacity_full",
        "fill the test export to 99 % from the client (fsync)",
        f"{USED} >= 98 -> FREEZE_EVIDENCE, VALIDATE_HOST; no QUARANTINE for shared "
        "capacity; below 98 after the unfill",
        capacity_full,
        cleanup="remove the fill, unmount, delete the export",
    ),
    _spec(
        "mount_disappears",
        "unmount the discovered test export on the node while the server lives",
        f"the vanished shared mount is reported {UNAVAILABLE}=1, not silently dropped",
        mount_disappears,
        outcome="FAIL",
        gap=GAP_MOUNT_DISAPPEARS,
        cleanup="forget the mount, delete the export",
    ),
    _spec(
        "io_only_failure",
        "chmod 000 the export root on the server (root-squashed client): statvfs answers, "
        "every open/read/fsync fails EACCES",
        f"a bounded I/O probe reports the export unusable ({UNAVAILABLE}=1)",
        io_only_failure,
        outcome="FAIL",
        gap=GAP_IO_ONLY,
        cleanup="chmod 1777 back, unmount, delete the export",
    ),
    _spec(
        "healthy_other_mount",
        "two test exports mounted; SIGSTOP only the first",
        f"{UNAVAILABLE}=1 on the hung mount -> VALIDATE_HOST fails on it despite the "
        "healthy second mount, which stays reported 0 and mints nothing",
        healthy_other_mount,
        max_minutes=45,
        cleanup="resume, unmount both, delete both exports",
    ),
    _spec(
        "recovery",
        "SIGSTOP then SIGCONT the export server",
        f"after the resume: {UNAVAILABLE} back to 0 with a cleared signal, a fresh "
        f"{USED} sample, and a real write/fsync/read-back on the export",
        recovery,
    ),
    _spec(
        "shared_failure_domain",
        "two task nodes (--fault-node, --node-b) mount one test export; SIGSTOP it",
        "both nodes mint a diagnostic-only chain and a product record names the common "
        "failure domain (one incident with both nodes, or a marker/notification of the "
        "export); neither node is isolated or treated as bad hardware",
        shared_failure_domain,
        max_minutes=45,
        flags="X",
        outcome="FAIL",
        gap=GAP_FAILURE_DOMAIN,
        preflight=_pair_preflight,
        cleanup="resume, unmount both nodes, delete the export",
    ),
]
