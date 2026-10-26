"""GF-GENERAL-LIVE-005 shared storage unavailable / full.

The host collector grades shared storage in
``collectors/host/shared_filesystems.py`` ``_shared_filesystems``: every
``nfs``/``nfs4``/``lustre`` mount of the host's mount table (``/proc/1/mountinfo``;
there is no shared-filesystem setting -- a host mount propagates into the
unit's namespace) gets a bounded ``statvfs`` and, when that answers, one bounded
read-only directory-entry read (5 s per call, 15 s per tick), and is reported as
``shared_filesystem_unavailable`` 1 (CRITICAL RUN_DIAGNOSTICS: FREEZE_EVIDENCE,
VALIDATE_HOST) with a ``failure_mode`` label -- ``PROBE_TIMEOUT``/``PROBE_FAILED``
for statvfs, ``IO_TIMEOUT``/``IO_ERROR`` (+ ``errno``) for the data path -- else 0
plus ``shared_filesystem_used_percent`` (98 % WARNING RUN_DIAGNOSTICS). A
node-scoped shared mount the collector has seen that vanishes from the host
table while its mount point directory remains is ``MOUNT_MISSING`` from a memory
kept across collector restarts (``collectors/host/shared_mounts.py``); the
directory removed too is a teardown and forgets the mount. Every shared sample
names its export (``source`` = the mount table's source field, ``fstype``).
``GPU_FAULT_FILESYSTEMS`` is deliberately *not* extended with the test export: a
configured mount runs the local-disk rules (``filesystem_used_percent`` 98 %
QUARANTINE, MOUNT_MISSING), which would turn a shared-storage condition into a
bad-node verdict.

The fixture (``shared_export_fixture.TestExportServer`` + the node probe
``probes/general_live_shared_probe.py``) stands up a disposable NFSv4 export --
a userspace server Pod over a size-limited tmpfs -- and mounts it ``soft`` under
``/mnt/gfrs-live/<digest>/<name>`` in the client node's host mount namespace;
``SIGSTOP`` on the server hangs every call, ``SIGCONT`` restores the same file
handles. Verdicts read only the product's records; the probe's deadline-bounded
``statvfs``/I/O readings are the oracle (the soft mount option bounds nothing
against a paused export -- live, no EIO within 185 s; the reader's thread
deadline does). The mount stays node-scoped under ``/mnt/gfrs-live/<digest>/``
and its cleanup unmounts *and* removes the directory, which is what makes the
product forget the test mount (batch 176 remembers a vanished shared mount
whose directory remains).

``mount_disappears`` unmounts the export with ``nfs-umount`` *without*
``--forget``: the mount point directory stays, which is exactly the condition
the product must report; ``nfs-remount`` is the recovery, and the LIFO
cleanup's ``cleanup --kind nfs-<name>`` removes the directory, so the product
forgets the mount once the variant is over.

``shared_failure_domain`` reads the record batch 177 added
(``shared_storage_failure_domain.SharedStorageFailureDomainCorrelator``): when
two nodes report the same export (the ``source`` label batch 176 put on every
shared sample; the mount point is deliberately no identity) unavailable within
600 s, the control plane records one cluster-scoped incident
``event_type=SHARED_STORAGE_FAILURE_DOMAIN`` with
``policy_reference=shared-filesystem:<export>``, both nodes in ``node_ids``,
reasons naming the export, the nodes and every member's per-node incident, no
workflow and no action; each per-node incident gains one ``common failure
domain <id>`` reason, exactly one ``SHARED_STORAGE_FAILURE_DOMAIN`` notification
goes out, both per-node chains stay diagnostic-only, and the record clears
RECOVERED once both nodes report the export available again. Both clients must
see the same export ``source`` before the hang, else the variant is NOT_READY.

The delivery hold and the recovery reads -- how the host collector ships a held
breach (its edge once, then the health summary) and its recovery (one
``recovered`` edge, then summaries moving the latest record), what that means
for every ``since`` stamp, and the two live verdicts of 2026-10-26 that were
wrong about a right product -- live in ``general_live_shared_recovery.py``.
"""

from __future__ import annotations

import errno
import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_common import (
    run_survey,
    sustained_detection_budget,
)
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
from scripts.e2e.regional.general_live_shared_recovery import (
    UNAVAILABLE,
    USED,
    await_clear,
    await_used_recovery,
    baseline_readings,
    delivery_errors,
    hold_hang_until_delivered,
    read_activity,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    chain_judgeable,
    collector_delivery,
    finding_incidents,
    hardware_errors,
    parse_time,
    samples,
    settled,
    verdict,
    workflows_of,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.shared_export_fixture import (
    ExportServerSettings,
    TestExportServer,
)

CASE = "GF-GENERAL-LIVE-005"
PRIMARY = "shared"
OTHER = "other"
DETECTION_BUDGET = 420
EXPORT_READY_SECONDS = 600
STORE_POLL_SECONDS = 20
FILL_PERCENT = 99.0
#: Isolation is never right for a shared-storage condition.
ISOLATION = frozenset({"STOP_WORKLOADS", "MARK_UNSCHEDULABLE", "QUARANTINE"})
FORBIDDEN = ISOLATION | HARDWARE_OPERATIONS
#: The ``failure_mode`` label the product puts on each unavailable reading.
MODE_MOUNT_MISSING = "MOUNT_MISSING"
IO_MODES = frozenset({"IO_ERROR", "IO_TIMEOUT"})
#: The cluster-scoped failure-domain record and its notification
#: (``shared_storage_failure_domain.py``, batch 177).
DOMAIN_EVENT_TYPE = "SHARED_STORAGE_FAILURE_DOMAIN"
DOMAIN_NOTIFICATION_CATEGORY = "SHARED_STORAGE_FAILURE_DOMAIN"
DOMAIN_REFERENCE_PREFIX = "shared-filesystem:"
DOMAIN_BACK_REFERENCE = "common failure domain "


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
    try:
        mounted = ctx.shared(*arguments, node=node, timeout=240)
    except HostProbeError as exc:
        # The export Pod is deleted by the LIFO cleanup right after this; its
        # log is the only record of why the server refused the mount (live
        # attempt 5: six variants, "output withheld", log gone with the Pod).
        suffix = server.settings.suffix
        evidence = f"export-{suffix}-log-at-mount-failure"
        ctx.evidence.write(
            evidence,
            {
                "node": node,
                "mount": name,
                "server": dict(server.ready),
                "error": str(exc),
                "log": server.logs(200),
            },
        )
        raise RegionalFixtureError(
            f"mounting the test export {server.name} "
            f"({server.ready.get('ip')}:{server.ready.get('export')}) on {node} "
            f"failed: {exc}; the export Pod log is saved as {evidence}.json"
        ) from exc
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
    """``cleanup --kind nfs-<name>`` on ``node``; the receipt must be clean for its scope.

    The probe answers for the named mount only -- its mount, its deadman timer,
    its mount point -- and leaves the module and the run directory to the run's
    last cleanup (the receipt's ``scope.run_level``), so in a two-mount run the
    still-mounted sibling is never this entry's residue (live 2026-10-26,
    ``healthy_other_mount``: the LIFO stack unmounted ``other`` first and a
    run-scoped check blamed that entry for ``shared``, its timer and the module).
    """

    value = ctx.shared("cleanup", "--kind", f"nfs-{name}", node=node, timeout=300)
    if value.get("clean") is not True:
        scope = value.get("scope") or {}
        level = "run-level" if scope.get("run_level") else "mount-scoped"
        raise RegionalFixtureError(
            f"the shared mount cleanup of {name} on {node} ({level}) left residue: "
            f"{value.get('residue')}"
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


def _baseline_view(activity: dict[str, Any], mount: str) -> dict[str, Any]:
    """What the store showed while the baseline was awaited: the host
    collector's delivery inside the read, the mounts its shared latest records
    name, and the readings of ``mount`` itself."""

    return {
        "since": activity.get("since"),
        "host_delivery": collector_delivery(
            activity, "HOST_TELEMETRY", batch_prefix="host-"
        ),
        "evidence_batches": len(activity.get("batches") or []),
        "shared_latest": [
            {key: item.get(key) for key in ("name", "device", "value", "observed_at")}
            for item in activity.get("metric_latest") or []
            if item.get("name") in {UNAVAILABLE, USED}
        ],
        "mount_readings": {
            metric: [
                item.get("value") for item in baseline_readings(activity, metric, mount)
            ]
            for metric in (UNAVAILABLE, USED)
        },
    }


def _baseline_shortfall(seen: dict[str, Any], mount: str) -> str:
    """One clause on why the baseline read came up empty: a silent collector,
    a mount read but never healthy, or a mount the collector does not list."""

    delivery = seen.get("host_delivery")
    if delivery is None:
        return (
            "the host collector delivered no batch since the mount (no HOST_TELEMETRY "
            "host- status inside the read): the collector, not the mount, is silent"
        )
    readings = seen.get("mount_readings") or {}
    if readings.get(UNAVAILABLE) or readings.get(USED):
        return (
            f"{mount} was read but never healthy: {UNAVAILABLE}="
            f"{readings.get(UNAVAILABLE)} {USED}={readings.get(USED)}"
        )
    devices = sorted(
        {str(item.get("device")) for item in seen.get("shared_latest") or []}
    )
    return (
        "the test export was not discovered as a shared mount (the host collector "
        f"delivered until {delivery.get('last_success_at')}; its shared latest "
        f"records name {devices if devices else 'no mount'})"
    )


def await_baseline(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    """The product discovered the test export: a 0 reading and a used % sample.

    One accepted reading is the baseline (the spec's three normal samples would
    cost three health summaries). It is read from evidence batches *and* the
    latest-metric records (``baseline_readings``): a healthy new mount reaches
    the store only inside a health summary, which is never raw evidence, so the
    budget is one summary period plus margin. When it runs out the error says
    what the store did show -- whether the host collector delivered at all since
    ``since``, which mounts its shared latest records name -- so a mount the
    collector truly does not list (a sandbox view, a filtered root) is told
    apart from a quiet window; the last read is saved as
    ``baseline-<node>-last-read.json``.
    """

    seen: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        current = read_activity(ctx, node, since, mount=mount)
        seen.clear()
        seen.update(_baseline_view(current, mount))
        zero = [
            item
            for item in baseline_readings(current, UNAVAILABLE, mount)
            if item.get("value") == 0
        ]
        used = baseline_readings(current, USED, mount)
        return current if zero and used else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=f"baseline-{node}",
    )
    if value is None:
        ctx.evidence.write(f"baseline-{node}-last-read", seen)
        raise VariantInconclusive(
            f"the product never delivered a healthy {UNAVAILABLE}/{USED} reading "
            f"for {mount} on {node}: {_baseline_shortfall(seen, mount)}"
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
    validation: str | None = None,
) -> dict[str, Any]:
    """The incident ``metric`` minted on ``mount``; with ``validation``, until
    that step of its chain can be judged (terminal, not WAITING for the next
    health summary's evidence)."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read_activity(ctx, node, since, mount=mount)
        if validation is not None:
            ready = chain_judgeable(last, metric, device=mount, validation=validation)
        else:
            ready = bool(finding_incidents(last, metric, device=mount))
        return last if ready else None

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


def wait_all_settled(
    ctx: VariantContext, node: str, since: datetime, incident_ids: list[str]
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read_activity(ctx, node, since)
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


def other_mount_errors(
    activity: dict[str, Any], other: str, since: datetime
) -> list[str]:
    """The healthy second export stayed reported as available.

    An evidence batch's 0 reading of ``other`` or its latest-metric record moved
    to a stamp at or after ``since``: the summary under the hang carries the
    healthy mount but is no raw evidence (``baseline_readings``).
    """

    fresh = [
        item
        for item in baseline_readings(activity, UNAVAILABLE, other)
        if item.get("value") == 0
        and (moment := parse_time(item.get("observed_at"))) is not None
        and moment >= since
    ]
    return (
        []
        if fresh
        else [
            f"no {UNAVAILABLE}=0 reading for the healthy mount {other} after the hang "
            "(no evidence batch sample, no moved latest-metric record)"
        ]
    )


def _labelled_unavailable(activity: dict[str, Any], mount: str) -> list[dict[str, Any]]:
    return [
        sample
        for sample in samples(activity, UNAVAILABLE, device=mount)
        if sample.get("value") == 1 and isinstance(sample.get("labels"), dict)
    ]


def failure_mode_errors(
    activity: dict[str, Any], mount: str, modes: Iterable[str]
) -> list[str]:
    """The unavailable readings carry one of the variant's ``failure_mode`` labels.

    The product labels every unavailable reading, so an unlabelled one is a
    regression, not an older build to be lenient with.
    """

    wanted = set(modes)
    seen = {
        str(sample["labels"]["failure_mode"])
        for sample in _labelled_unavailable(activity, mount)
        if sample["labels"].get("failure_mode")
    }
    if not seen:
        return [f"no {UNAVAILABLE}=1 reading for {mount} carries a failure_mode label"]
    if not seen & wanted:
        return [
            f"{UNAVAILABLE} on {mount} is labelled {sorted(seen)}, not {sorted(wanted)}"
        ]
    return []


def export_source_errors(
    activity: dict[str, Any], mount: str, server: TestExportServer
) -> list[str]:
    """The unavailable readings name the export (``source``), not just the path.

    ``source`` is the mount table's source field, ``<server>:<export>``; the
    control plane's failure-domain correlation keys on it, so it must name the
    test export's server and export path.
    """

    ready = server.ready or {}
    ip, export = str(ready.get("ip") or ""), str(ready.get("export") or "")
    seen = {
        str(sample["labels"]["source"])
        for sample in _labelled_unavailable(activity, mount)
        if sample["labels"].get("source")
    }
    if not seen:
        return [
            f"no {UNAVAILABLE}=1 reading for {mount} names its export (source label)"
        ]
    if not any(ip and ip in source and export in source for source in seen):
        return [
            f"{UNAVAILABLE} on {mount} names export {sorted(seen)}, not {ip}:{export}"
        ]
    return []


def export_sources(activity: dict[str, Any], mount: str) -> set[str]:
    """The ``source`` labels the product's shared samples of ``mount`` carry."""

    return {
        str(sample["labels"]["source"])
        for sample in baseline_readings(activity, UNAVAILABLE, mount)
        if isinstance(sample.get("labels"), dict) and sample["labels"].get("source")
    }


def require_same_export(baselines: dict[str, dict[str, Any]], mount: str) -> str:
    """Both clients report one and the same export ``source``, or NOT_READY.

    The failure-domain correlation keys on the collector's ``source`` label
    (batch 176); a node whose samples carry none, or two nodes naming different
    sources, cannot form a domain and the injection would prove nothing.
    """

    seen = {
        node: export_sources(activity, mount) for node, activity in baselines.items()
    }
    missing = sorted(node for node, sources in seen.items() if not sources)
    if missing:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"the shared samples of {mount} on {', '.join(missing)} carry no source "
            "label; the control plane cannot correlate a failure domain without it",
        )
    common = set.intersection(*seen.values())
    if len(common) != 1 or any(len(sources) != 1 for sources in seen.values()):
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"the clients do not see one and the same export source for {mount}: "
            + "; ".join(
                f"{node} {sorted(sources)}" for node, sources in sorted(seen.items())
            ),
        )
    return common.pop()


def domain_records(
    activities: dict[str, dict[str, Any]], export: str
) -> list[dict[str, Any]]:
    """The failure-domain incidents of ``export`` across the nodes' reads, once each."""

    found: dict[str, dict[str, Any]] = {}
    for activity in activities.values():
        for incident in activity.get("incidents") or []:
            if (
                incident.get("event_type") == DOMAIN_EVENT_TYPE
                and incident.get("policy_reference")
                == f"{DOMAIN_REFERENCE_PREFIX}{export}"
            ):
                found[str(incident.get("incident_id"))] = incident
    return [found[key] for key in sorted(found)]


def failure_domain_errors(
    activities: dict[str, dict[str, Any]],
    nodes: tuple[str, str],
    export: str,
    node_incidents: dict[str, list[str]],
) -> list[str]:
    """One failure-domain record ties both nodes' export failures together.

    ``node_incidents`` maps each node to the ids of its own
    ``shared_filesystem_unavailable`` incidents. The record must name the
    export (``policy_reference``), both nodes, every member's incident, carry
    no workflow and no action; each per-node incident must point back at it;
    and exactly one SHARED_STORAGE_FAILURE_DOMAIN notification must exist.
    """

    records = domain_records(activities, export)
    if not records:
        return [
            f"no {DOMAIN_EVENT_TYPE} incident with policy_reference "
            f"{DOMAIN_REFERENCE_PREFIX}{export} names the export: two unrelated node "
            "diagnoses"
        ]
    errors: list[str] = []
    if len(records) > 1:
        errors.append(
            f"{len(records)} {DOMAIN_EVENT_TYPE} records for {export}: "
            + ", ".join(str(item.get("incident_id")) for item in records)
        )
    domain = records[0]
    domain_id = str(domain.get("incident_id"))
    members = set(domain.get("node_ids") or [])
    absent = sorted(set(nodes) - members)
    if absent:
        errors.append(f"the failure-domain record {domain_id} does not name {absent}")
    if domain.get("effective_action") or domain.get("workflow_request_id"):
        errors.append(
            f"the failure-domain record {domain_id} carries an action/workflow "
            f"({domain.get('effective_action')} / {domain.get('workflow_request_id')}): "
            "it must be a record, never a chain"
        )
    for activity in activities.values():
        for workflow in workflows_of(activity, domain_id):
            errors.append(
                f"workflow {workflow.get('request_id')} planned on the failure-domain record"
            )
    reasons = [str(item) for item in domain.get("reasons") or []]
    first = reasons[0] if reasons else ""
    if export not in first or not all(node in first for node in nodes):
        errors.append(
            f"the first reason of {domain_id} does not name the export and both nodes: "
            f"{first[:200]!r}"
        )
    joined = "\n".join(reasons)
    for node in nodes:
        ids = node_incidents.get(node) or []
        if not ids:
            errors.append(f"{node} has no {UNAVAILABLE} incident to be a member with")
        elif not any(
            f"member {node}: incident {incident_id}" in joined for incident_id in ids
        ):
            errors.append(
                f"the record {domain_id} names no member line for {node} with its "
                f"incident ({ids})"
            )
    for node in nodes:
        activity = activities.get(node) or {}
        for incident in activity.get("incidents") or []:
            if str(incident.get("incident_id")) not in (node_incidents.get(node) or []):
                continue
            prefix = f"{DOMAIN_BACK_REFERENCE}{domain_id}:"
            if not any(
                str(item).startswith(prefix) for item in incident.get("reasons") or []
            ):
                errors.append(
                    f"{node}'s incident {incident.get('incident_id')} carries no "
                    f"'{prefix}' reason back to the record"
                )
    mails: dict[str, dict[str, Any]] = {}
    for activity in activities.values():
        for item in activity.get("notifications") or []:
            if (
                item.get("category") == DOMAIN_NOTIFICATION_CATEGORY
                and item.get("incident_id") == domain_id
            ):
                mails[str(item.get("notification_id"))] = item
    if len(mails) != 1:
        errors.append(
            f"{len(mails)} {DOMAIN_NOTIFICATION_CATEGORY} notification(s) for the record "
            f"{domain_id}, expected exactly one"
        )
    return errors


def domain_state(activity: dict[str, Any], domain_id: str) -> str | None:
    for incident in activity.get("incidents") or []:
        if str(incident.get("incident_id")) == domain_id:
            return str(incident.get("state"))
    return None


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
    """The export hangs: the finding, the delivery under the held hang, the recovery.

    The hang is held past the finding until the collector's next scheduled
    delivery is in the store (``hold_hang_until_delivered``), the stages are
    graded on that last read under the hang -- no post-resume batch can be in
    it -- and the recovery's ``since`` is stamped before the resume.
    """

    server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started, _detection = _hang_and_detect(ctx, server, mount)
    held = hold_hang_until_delivered(ctx, node, mount, started, name="hang-delivery")
    undo_since = utc_now()
    _resume(ctx, server, "a")
    stages = {
        "chain": diagnostic_chain(held, UNAVAILABLE, mount),
        "bounded_delivery": delivery_errors(held, started, mount=mount),
        "hardware": hardware_errors(held),
    }
    incidents = _incident_ids(held, UNAVAILABLE, mount)
    recovered = await_clear(ctx, node, mount, undo_since)
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
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
        raise VariantInconclusive(_short_fill(filled, used))
    # The 98 % rule has no sustain: the fill's edge batch mints the finding, and
    # VALIDATE_HOST is judged on the health summary after it -- wait for that
    # step to be terminal before grading the chain (one summary, not the first
    # snapshot that shows the incident).
    detection = wait_finding(
        ctx,
        node,
        started,
        USED,
        mount,
        timeout_seconds=sustained_detection_budget(ctx, 0, summaries=1),
        name="full-detection",
        validation="VALIDATE_HOST",
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
    undo_since = utc_now()
    unfilled = ctx.shared("nfs-unfill", "--name", PRIMARY, timeout=240)
    ctx.cleanup.discard("unfill")
    ctx.evidence.write("unfill", unfilled)
    recovered = await_used_recovery(ctx, node, mount, undo_since)
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
    return _result(
        stages, extra={"mount": mount, "used_percent": used, "recovery": recovered}
    )


def _short_fill(filled: dict[str, Any], used: Any) -> str:
    """Why the fill left the export below 98 %: the percentage it reached (or
    that the fill read no statvfs), what was written against what was wanted,
    and the fill's own first failure -- class, errno and its name (ENOSPC before
    the target, a quota). The probe hands a short fill over as a reading; the
    verdict is INCONCLUSIVE, never a failed call."""

    fill = filled.get("fill") if isinstance(filled.get("fill"), dict) else {}
    text = (
        f"the test export only reached {used:.1f} % used"
        if isinstance(used, (int, float))
        else "the fill left no statvfs reading of the test export"
    )
    notes = []
    written, wanted = fill.get("fill_bytes"), fill.get("wanted_bytes")
    if isinstance(written, int) and isinstance(wanted, int):
        notes.append(f"fill wrote {written >> 20} of {wanted >> 20} MiB")
    if isinstance(fill.get("error"), str):
        code = fill.get("errno")
        name = errno.errorcode.get(code, "") if isinstance(code, int) else ""
        notes.append(f"{fill['error']} errno {code} {name}".rstrip())
    return f"{text} ({'; '.join(notes)})" if notes else text


def mount_disappears(ctx: VariantContext) -> dict[str, Any]:
    """The export is unmounted; its mount point directory stays (a failure, not a teardown)."""

    server, mount, _since = _prepare(ctx)
    node = ctx.settings.node
    started = utc_now()
    gone = ctx.shared("nfs-umount", "--name", PRIMARY, node=node, timeout=240)
    ctx.evidence.write("umount", gone)
    if gone.get("host_mount") is not None:
        raise VariantInconclusive("the test export is still in the host mount table")
    if gone.get("directory_present") is False:
        raise VariantInconclusive(
            f"the mount point {mount} went with the mount: the product reads that as a "
            "teardown and forgets the mount; the injection must leave the directory"
        )
    detection = wait_finding(
        ctx,
        node,
        started,
        UNAVAILABLE,
        mount,
        timeout_seconds=DETECTION_BUDGET,
        name="vanished-detection",
    )
    stages = {
        "vanished_reported": []
        if finding_incidents(detection, UNAVAILABLE, device=mount)
        else [
            f"the vanished shared mount {mount} was never reported {UNAVAILABLE}=1 "
            "while its mount point directory remained"
        ],
        "failure_mode": failure_mode_errors(detection, mount, {MODE_MOUNT_MISSING}),
        "export_source": export_source_errors(detection, mount, server),
        "chain": diagnostic_chain(detection, UNAVAILABLE, mount),
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, UNAVAILABLE, mount)
    # Stamped before the remount: the recovered edge can land between the
    # mount syscall and the probe's answer (live 2026-10-26 14:33Z; see
    # ``general_live_shared_recovery.await_clear``).
    undo_since = utc_now()
    remounted = ctx.shared("nfs-remount", "--name", PRIMARY, node=node, timeout=240)
    ctx.evidence.write("remount", remounted)
    if remounted.get("host_mount") is None:
        raise VariantInconclusive(
            f"the test export could not be remounted at {mount}: no recovery to prove"
        )
    recovered = await_clear(ctx, node, mount, undo_since)
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
    return _result(
        stages, extra={"mount": mount, "umount": gone, "recovery": recovered}
    )


def io_only_failure(ctx: VariantContext) -> dict[str, Any]:
    """statvfs keeps answering while every open/read on the export is denied."""

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
    detection = wait_finding(
        ctx,
        node,
        started,
        UNAVAILABLE,
        mount,
        timeout_seconds=DETECTION_BUDGET,
        name="io-failure-detection",
    )
    stages = {
        "io_failure_reported": []
        if finding_incidents(detection, UNAVAILABLE, device=mount)
        else [
            f"no {UNAVAILABLE} finding on {mount} while every read/open failed "
            f"(errno {(io.get('io') or {}).get('errno')}) and statvfs answered"
        ],
        "failure_mode": failure_mode_errors(detection, mount, IO_MODES),
        "export_source": export_source_errors(detection, mount, server),
        "chain": diagnostic_chain(detection, UNAVAILABLE, mount),
        "hardware": hardware_errors(detection),
    }
    incidents = _incident_ids(detection, UNAVAILABLE, mount)
    undo_since = utc_now()
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
    recovered = await_clear(ctx, node, mount, undo_since)
    stages["recovery"] = list(recovered["errors"])
    wait_all_settled(ctx, node, started, incidents)
    return _result(
        stages,
        extra={
            "mount": mount,
            "restored": restored,
            "io": io.get("io"),
            "recovery": recovered,
        },
    )


def healthy_other_mount(ctx: VariantContext) -> dict[str, Any]:
    """One export hangs while a second stays healthy on the same node.

    The hang is held past the finding until the collector's next scheduled
    delivery is in the store (``hold_hang_until_delivered``); the stages are
    graded on that last read under the hang, which the resume closes, and the
    recovery's ``since`` is stamped before the resume.
    """

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
    started, _detection = _hang_and_detect(ctx, server_a, mount)
    healthy = _stat(ctx, node, OTHER)
    if _hung(healthy):
        _resume(ctx, server_a, "a")
        raise VariantInconclusive(
            f"the second export hung too ({healthy}); no healthy control"
        )
    held = hold_hang_until_delivered(ctx, node, mount, started, name="hang-delivery")
    undo_since = utc_now()
    _resume(ctx, server_a, "a")
    stages = {
        "chain": diagnostic_chain(held, UNAVAILABLE, mount),
        "other_not_graded": chain_errors(
            held, ChainExpectation(metric=UNAVAILABLE, device=other, present=False)
        ),
        "other_still_reported": other_mount_errors(held, other, started),
        "bounded_delivery": delivery_errors(held, started, mount=mount),
        "hardware": hardware_errors(held),
    }
    incidents = _incident_ids(held, UNAVAILABLE, mount)
    recovered = await_clear(ctx, node, mount, undo_since)
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


def _await_domain_recovered(
    ctx: VariantContext, node: str, since: datetime, domain_id: str
) -> list[str]:
    """The failure-domain record cleared RECOVERED once both nodes reported 0."""

    def accept() -> str | None:
        state = domain_state(read_activity(ctx, node, since), domain_id)
        return state if state == "RECOVERED" else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="domain-recovered",
    )
    if value is None:
        return [
            f"the failure-domain record {domain_id} did not clear RECOVERED after both "
            "nodes reported the export available again"
        ]
    return []


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
    baselines = {}
    for client in (node, peer):
        baselines[client] = await_baseline(ctx, client, mount, since)
        ctx.evidence.write(f"baseline-{client}", baselines[client])
    export = require_same_export(baselines, mount)
    ctx.evidence.write("export-identity", {"source": export, "mount": mount})
    started = utc_now()
    _pause(ctx, server, "a")
    readings = {client: _stat(ctx, client, PRIMARY) for client in (node, peer)}
    if not all(_hung(item) for item in readings.values()):
        _resume(ctx, server, "a")
        raise VariantInconclusive(
            f"the paused export still answered on a client: {readings}"
        )
    for client in (node, peer):
        wait_finding(
            ctx,
            client,
            started,
            UNAVAILABLE,
            mount,
            timeout_seconds=DETECTION_BUDGET,
            name=f"domain-detection-{client}",
        )
    # Both findings exist (or the budget ran out): read both nodes again so the
    # record the second activation created is in both views.
    activities = {
        client: read_activity(ctx, client, started, mount=mount)
        for client in (node, peer)
    }
    node_incidents = {
        client: _incident_ids(activity, UNAVAILABLE, mount)
        for client, activity in activities.items()
    }
    stages: dict[str, list[str]] = {}
    for client, activity in activities.items():
        stages[f"chain@{client}"] = diagnostic_chain(activity, UNAVAILABLE, mount)
        stages[f"hardware@{client}"] = hardware_errors(activity)
    stages["common_failure_domain"] = failure_domain_errors(
        activities, (node, peer), export, node_incidents
    )
    records = domain_records(activities, export)
    resumed_at = utc_now()
    _resume(ctx, server, "a")
    for client, activity in activities.items():
        recovered = await_clear(ctx, client, mount, resumed_at)
        stages[f"recovery@{client}"] = list(recovered["errors"])
        wait_all_settled(ctx, client, started, node_incidents[client])
    stages["domain_recovered"] = (
        _await_domain_recovered(ctx, node, started, str(records[0]["incident_id"]))
        if records
        else []
    )
    return _result(
        stages,
        notes=[
            "the per-node chains stay diagnostic-only; the cluster-scoped record names "
            "the export, both nodes and their incidents, carries no action and clears "
            "once both nodes report the export available",
        ],
        extra={
            "mount": mount,
            "export": export,
            "nodes": [node, peer],
            "domain_records": [str(item.get("incident_id")) for item in records],
            "node_incidents": node_incidents,
        },
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _preflight(settings: GeneralLiveSettings, _regional: Any = None) -> list[str]:
    image = settings.export_image
    if not image or "@sha256:" not in image:
        return [
            "the 005 test export server image is missing (--export-image or "
            "GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE, an immutable digest)"
        ]
    return []


def _pair_preflight(settings: GeneralLiveSettings, regional: Any = None) -> list[str]:
    errors = _preflight(settings, regional)
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
    preflight: Callable[[GeneralLiveSettings, Any], list[str]] = _preflight,
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
        "on the node, discovered by fstype; SIGSTOP the server so every statvfs hangs, "
        "held across the collector's next health summary",
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
        "unmount the discovered test export on the node while the server lives, "
        "leaving the mount point directory in place",
        f"the vanished shared mount is reported {UNAVAILABLE}=1 failure_mode=MOUNT_MISSING "
        "naming its export -> FREEZE_EVIDENCE, VALIDATE_HOST failing on it; 0 again once "
        "remounted",
        mount_disappears,
        cleanup="remount for the recovery; unmount and remove the mount point, delete the export",
    ),
    _spec(
        "io_only_failure",
        "chmod 000 the export root on the server (root-squashed client): statvfs answers, "
        "every open/read/fsync fails EACCES",
        f"the bounded directory read reports the export unusable ({UNAVAILABLE}=1 "
        "failure_mode=IO_ERROR with its errno, or IO_TIMEOUT) -> FREEZE_EVIDENCE, "
        "VALIDATE_HOST failing on it; 0 again after chmod 1777",
        io_only_failure,
        cleanup="chmod 1777 back, unmount, delete the export",
    ),
    _spec(
        "healthy_other_mount",
        "two test exports mounted; SIGSTOP only the first, held across the collector's "
        "next health summary",
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
        "two task nodes (--fault-node, --node-b) mount one test export whose source "
        "both report alike; SIGSTOP it",
        "both nodes mint a diagnostic-only chain; one cluster-scoped "
        f"{DOMAIN_EVENT_TYPE} incident (policy_reference shared-filesystem:<export>, "
        "both nodes, reasons naming the export, the nodes and each node's incident, no "
        "workflow or action) with each per-node incident pointing back at it and exactly "
        f"one {DOMAIN_NOTIFICATION_CATEGORY} notification; the record clears RECOVERED "
        "once both nodes report 0; neither node is isolated or treated as bad hardware",
        shared_failure_domain,
        max_minutes=45,
        flags="X",
        preflight=_pair_preflight,
        cleanup="resume, unmount both nodes, delete the export",
    ),
]
