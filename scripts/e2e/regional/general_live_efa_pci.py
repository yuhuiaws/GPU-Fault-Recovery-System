"""GF-GENERAL-LIVE-013 ``pci_missing``: one EFA PCI function absent -> reboot chain.

The host collector counts EFA devices three ways (``collectors/host/
observation.efa_inventory_counts``): PCI functions under
``GPU_FAULT_PCI_DEVICES_ROOT``, driver-bound RDMA devices and ACTIVE ports
under ``/sys/class/infiniband``. ``pci_discovered < expected`` resolves to
``failure_mode=PCI_DEVICE_MISSING`` (``collectors/host/inventory.py``), the
``efa_inventory_mismatch`` gauge confirms after
``GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES`` (2) consecutive ticks and
``host_health.efa_inventory_rule`` makes it CRITICAL ``REBOOT_NODE``:
MARK_UNSCHEDULABLE, QUARANTINE, RESTART_NODE through the provider, VALIDATE_GPU
/ VALIDATE_HOST / VALIDATE_FABRIC on evidence newer than the reboot
(``adapters/gpu_validation.EFA_INVENTORY_RECOVERY_OPERATIONS`` lists
RESTART_NODE, so the post-reboot validation re-reads the inventory), then
RESTORE_SCHEDULING. This site has no vendor lab and never removes a real
function, so the variant is ``live-replay`` on the shared host source replay
(``general_live_source_replay.py``, ``HOST_SOURCE_REPLAY``): the node's real
PCI root is copied with ONE real EFA function absent (``build_pci(source=...,
drop=[bdf])``) and the real RDMA tree with that function's device absent
(``build_infiniband(drop=[device])`` -- the way a function that fell off the
bus takes its RDMA device with it), both bound into the host collector unit's
private environment for the ticks the rule needs. Parsing, confirmation,
grading, the database, scheduling and the approved node reboot stay real.

Negative stage first (the spec's "unreadable must not authorize a reboot" and
the known risk that two failed reads are counted as zero cards): the PCI root
is made *unreadable* -- the fixture path is a regular file, so every listing
fails with ENOTDIR whatever the unit's privileges -- and held for twice the
confirmation window. The product must record ``EFA inventory observation
unavailable`` (UNKNOWN), report no inventory count or mismatch sample, mint no
finding, plan no hardware step and isolate nothing; ``_efa_inventory`` also
resets the confirmation streak on that path, which the module tests pin
against the real collector. Only the inventory's own samples are graded
(``EFA_INVENTORY_METRICS``): the RDMA tree is real in this window, so
``_rdma``'s per-port ``rdma_link_down`` keeps flowing and is recorded, not
counted -- grading it as an inventory count failed a correct product (live hp
2026-10-27). A finding from the unreadable root is FAIL, its chain is waited
out (it may reboot) and the missing-function stage is not run.

Positive stage: the window is closed the moment the finding exists -- the
drop-in, the deadman and the fixture are gone before RESTART_NODE executes, so
every later read of the product is real. Should the product reboot faster
than the close (the chain needs only two quick cordon/taint steps before the
provider call), the close fails harmlessly: the drop-in, the deadman's
transient units and the fixture all live under ``/run`` and do not survive
the reboot, a deadman that fires during shutdown can at most fail a
``systemctl restart`` the shutdown kills anyway, and the runner proves the
clean state on the NEW boot (``snapshot``: no drop-in, no state dir, no
deadman, the unit's ``GPU_FAULT_PCI_DEVICES_ROOT`` not a fixture path;
``read_back`` of the RDMA tree reads the real sysfs). The reboot itself is
waited for the way COLLECT-004 waits (``regional.node_snapshot`` boot id and
Ready, same Node UID; ``collector_inventory_reboot.wait_node_settled_after_
reboot`` before the probe Pod -- a plain Pod that died with the node -- is
re-created). Grading reads the product's records only: the confirmed sample's
``failure_mode``/counts, exactly one SUCCEEDED RESTART_NODE execution and
remote command scoped to the node, no REPLACE_NODE, no second reboot, the
incident RECOVERED, the node Ready and schedulable with the EFA and GPU
allocatable at baseline, fresh post-reboot ``efa_inventory_active_count ==
expected`` / ``efa_inventory_mismatch == 0`` -- newer than the product's own
RESTART_NODE completion, the anchor its inventory gate reads from
(``adapters/gpu_validation.inventory_observation_window``), not the runner's
later bookkeeping --, the provider's own receipt in CloudTrail (EC2
``RebootInstances`` on plain EKS, SageMaker ``BatchRebootClusterNodes`` on
HyperPod) naming this node and no other -- EC2 by instance id, SageMaker by
the ``NodeLogicalId`` the product submits (``hyperpod.py`` ``NodeLogicalIds``),
bound to the instance through ``list-cluster-nodes --include-node-logical-ids``
before the window opens --, and -- when ``--node-b`` is given -- the pair node
untouched (same boot id, Ready, not isolated, no incident). The runner's
oracle is the fixture receipt that names the dropped BDF and RDMA device; the
product's own records carry the counts (it names no BDF). A node that vanishes
during the reboot wait (Spot reclaim) is INCONCLUSIVE, not a product verdict.
If the chain ends with the node still isolated, cleanup hands it back through
the product's validated restore.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.collector_inventory_reboot import (
    wait_node_settled_after_reboot,
)
from scripts.e2e.regional.collector_reboot_evidence import (
    provider_event_source,
    reboot_event_names,
)
from scripts.e2e.regional.general_live_common import (
    STORE_POLL_SECONDS,
    activity,
    restore_incidents,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    node_provider,
    utc_now,
)
from scripts.e2e.regional.general_live_inventory import (
    CASE_013,
    DETECTION_BUDGET,
    EFA_INVENTORY_METRICS,
    EFA_METRICS,
    INFINIBAND_ROOT,
    NODE_REBOOT,
    STEP_TIMEOUT_SECONDS,
    efa_collection_errors,
    efa_failure_mode_errors,
    port_active,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    VariantSpec,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    INFINIBAND_TREE,
    HostSourceReplay,
    host_source_replay,
)
from scripts.e2e.regional.general_live_store import node_activity
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    executions,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    operations,
    parse_time,
    samples,
    verdict,
    workflows_of,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeError

PCI_ROOT = "/sys/bus/pci/devices"
EFA_RESOURCE = "vpc.amazonaws.com/efa"
MISMATCH_METRIC = "efa_inventory_mismatch"
PCI_DEVICE_MISSING = "PCI_DEVICE_MISSING"
#: The collection error ``_efa_inventory`` records for a read it could not make.
UNKNOWN_MARKER = "EFA inventory observation unavailable"
#: Fixture paths live here; a unit still pointed at one has not been restored.
FIXTURE_PREFIX = "/run/gfrs-live/"
INSTANCE_ID = re.compile(r"i-[0-9a-f]{17}")
#: Host ticks the unreadable root is held: twice the 2-sample confirmation window.
NEGATIVE_TICKS = 4
#: From the confirmed finding to the node Ready again on a new boot id.
REBOOT_BUDGET = 1500
#: Post-reboot VALIDATE_GPU / VALIDATE_HOST / VALIDATE_FABRIC (each bounded by
#: the workflow step timeout) and RESTORE_SCHEDULING.
POST_REBOOT_SETTLE_BUDGET = 3 * STEP_TIMEOUT_SECONDS + 300
#: CloudTrail is eventually consistent: typically 5 minutes, guaranteed 15.
PROVIDER_RECEIPT_BUDGET = 900
#: ``sagemaker list-cluster-nodes``: the instance <-> NodeLogicalId binding.
HYPERPOD_INVENTORY_TIMEOUT = 180
#: The maintenance window the positive stage needs ahead of it.
POSITIVE_WINDOW_SECONDS = DETECTION_BUDGET + REBOOT_BUDGET + POST_REBOOT_SETTLE_BUDGET
#: The product's REBOOT_NODE plan (``families/health.py`` ``_reboot_operations``;
#: STOP_WORKLOADS only on an active node, VALIDATE_GPU/HOST between).
REBOOT_CHAIN = (
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "RESTART_NODE",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
#: Hardware steps the chain may not contain besides its one reboot.
FORBIDDEN_OPERATIONS = HARDWARE_OPERATIONS - {"RESTART_NODE"}
#: ``general_live_fixture.node_provider`` lane -> ``collector_reboot_evidence`` provider.
PROVIDER_KEYS = {"eks": "eks-managed", "hp": "hyperpod"}
REBOOT_POLL_SECONDS = 15


def _numeric(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item.get("value"), (int, float))]


def expected_efa_count(ctx: VariantContext, fallback: int) -> int:
    """The count the host collector grades against, from the product's record."""

    latest = activity(ctx, utc_now(), metrics={"efa_inventory_expected_count"}).get(
        "metric_latest"
    )
    for item in latest or []:
        if item.get("name") == "efa_inventory_expected_count" and isinstance(
            item.get("value"), (int, float)
        ):
            return int(item["value"])
    return fallback


def efa_functions(inventory: dict[str, Any]) -> list[tuple[str, str]]:
    """``(RDMA device, BDF)`` of every EFA device with an ACTIVE/LinkUp port."""

    found = []
    for name, summary in sorted((inventory.get("devices") or {}).items()):
        if not summary.get("efa") or not isinstance(summary.get("pci"), str):
            continue
        if any(port_active(states) for states in (summary.get("ports") or {}).values()):
            found.append((str(name), str(summary["pci"])))
    return found


def node_identity(ctx: VariantContext, node: str | None = None) -> dict[str, Any]:
    """The runner's node snapshot plus the EC2 instance and the EFA allocatable."""

    name = node or ctx.settings.node
    snapshot = ctx.regional.node_snapshot(name)
    document = json.loads(
        ctx.regional.kubectl("gpu", "get", "node", name, "-o", "json", timeout=60)
    )
    provider_id = str((document.get("spec") or {}).get("providerID") or "")
    match = INSTANCE_ID.search(provider_id)
    allocatable = (document.get("status") or {}).get("allocatable") or {}
    return {
        **snapshot,
        "provider_id": provider_id,
        "instance_id": match.group(0) if match else None,
        "efa_allocatable": allocatable.get(EFA_RESOURCE),
    }


class _ContextClock:
    """``time``-shaped clock over the variant context (stubbed in tests)."""

    def __init__(self, ctx: VariantContext) -> None:
        self.ctx = ctx
        self.origin = ctx.clock()

    def monotonic(self) -> float:
        return (self.ctx.clock() - self.origin).total_seconds()

    def sleep(self, seconds: float) -> None:
        self.ctx.sleep(seconds)


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    isolation = isolation_operations(observed)
    errors = [f"isolation operations planned: {isolation}"] if isolation else []
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("the node is isolated")
    return errors


# --------------------------------------------------------------------------- #
# negative stage: an unreadable PCI root is UNKNOWN, never zero functions
# --------------------------------------------------------------------------- #
def unreadable_root(ctx: VariantContext, replay: HostSourceReplay) -> dict[str, Any]:
    """Hold an unreadable PCI root for ``NEGATIVE_TICKS`` ticks; grade UNKNOWN only.

    Two consecutive failed reads are the confirmation window of the mismatch
    rule; the hold spans twice that, so a product that counted a failed read
    as zero functions would have confirmed and planned REBOOT_NODE inside it.
    """

    built = replay.build_pci(unreadable=True)
    ctx.evidence.write("unreadable-root-fixture", built)
    hold_seconds = ctx.host_interval * NEGATIVE_TICKS + 30
    opened = replay.open_pci_root(restore_seconds=max(600, hold_seconds + 300))
    ctx.evidence.write("unreadable-root-open", opened)
    started = utc_now()
    ctx.hold(hold_seconds, name="unreadable-root")
    observed = activity(ctx, started, metrics=EFA_METRICS)
    ctx.evidence.write("unreadable-root-close", replay.close())
    errors = efa_collection_errors(observed)
    # Only the inventory's own samples are counts here: the RDMA tree is real
    # in this window, so ``_rdma``'s per-port ``rdma_link_down`` keeps flowing
    # (live hp 2026-10-27: 16 per delivered tick) and is recorded, not graded.
    counts = [
        (name, item)
        for name in sorted(EFA_INVENTORY_METRICS)
        for item in _numeric(samples(observed, name))
    ]
    rdma_samples = [
        item
        for name in sorted(EFA_METRICS - EFA_INVENTORY_METRICS)
        for item in _numeric(samples(observed, name))
    ]
    incidents = finding_incidents(observed, MISMATCH_METRIC)
    ctx.evidence.append_oracle(
        [
            {
                "efa_collection_errors": errors,
                "ticks_held": NEGATIVE_TICKS,
                "hold_seconds": hold_seconds,
                "counts_reported": len(counts),
                "rdma_samples_recorded": len(rdma_samples),
                **replay.evidence(fixture=built.get("path"), unreadable=True),
            }
        ],
        source="pci-root",
    )
    stages = {
        "negative_unknown": []
        if any(UNKNOWN_MARKER in error for error in errors)
        else [
            f"no {UNKNOWN_MARKER!r} collection error was recorded while the PCI "
            f"root was unreadable for {NEGATIVE_TICKS} ticks"
        ],
        "negative_no_count": [
            f"{name}={item['value']:g} was reported from an unreadable PCI root at "
            f"{item['observed_at']}"
            for name, item in counts
        ],
        "negative_no_finding": [
            f"an unreadable PCI root minted an {MISMATCH_METRIC} incident: "
            f"{item['incident_id']}"
            for item in incidents
        ],
        "negative_no_reboot": hardware_errors(observed),
        "negative_not_isolated": _isolation_errors(ctx, observed),
    }
    return {
        "stages": stages,
        "started": started,
        "incidents": incidents,
        "collection_errors": errors,
        "observed": observed,
    }


# --------------------------------------------------------------------------- #
# positive stage helpers
# --------------------------------------------------------------------------- #
def confirmed_sample_errors(detection: dict[str, Any], *, expected: int) -> list[str]:
    """The confirmed mismatch sample carries the counts of one absent function."""

    confirmed = [
        item
        for item in _numeric(samples(detection, MISMATCH_METRIC))
        if item["value"] >= 1
    ]
    if not confirmed:
        return [f"no {MISMATCH_METRIC}=1 sample reached the product"]
    labels = confirmed[0].get("labels") or {}
    wanted = {
        "expected_count": str(expected),
        "observed_count": str(expected - 1),
        "missing_count": "1",
        "discovered_count": str(expected - 1),
        "driver_bound_count": str(expected - 1),
    }
    errors = [
        f"confirmed sample label {key}={labels.get(key)!r}, expected {value!r}"
        for key, value in wanted.items()
        if labels.get(key) != value
    ]
    streak = labels.get("consecutive_mismatch_samples")
    required = labels.get("required_consecutive_samples")
    if not (str(streak).isdigit() and str(required).isdigit()) or int(streak) < int(
        required
    ):
        errors.append(
            f"confirmed sample streak {streak!r} is below the required {required!r}"
        )
    return errors


def wait_reboot_planned(
    ctx: VariantContext, since: datetime, incident_ids: list[str]
) -> dict[str, Any] | None:
    """The incident's workflow once it plans RESTART_NODE (or None)."""

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics=EFA_METRICS)
        for incident_id in incident_ids:
            for workflow in workflows_of(current, incident_id):
                if "RESTART_NODE" in operations(workflow):
                    return workflow
        return None

    return ctx.wait_until(
        accept,
        timeout_seconds=STEP_TIMEOUT_SECONDS,
        poll_seconds=STORE_POLL_SECONDS,
        name="reboot-planned",
    )


def wait_reboot(ctx: VariantContext, before: dict[str, Any]) -> dict[str, Any] | None:
    """The node Ready on a NEW boot id with the same Node UID (or None).

    A Node whose UID changed was replaced, not rebooted (a Spot reclaim during
    the window): the product's chain has no node left to validate and the
    variant cannot be graded -- INCONCLUSIVE, not a product verdict.
    """

    node = ctx.settings.node

    def accept() -> dict[str, Any] | None:
        current = ctx.regional.node_snapshot(node)
        if current.get("uid") != before.get("uid"):
            raise VariantInconclusive(
                f"{node} was replaced during the reboot wait (Node UID "
                f"{before.get('uid')} -> {current.get('uid')}; Spot reclaim?)"
            )
        boot_id = current.get("boot_id")
        if (
            current.get("ready") == "True"
            and boot_id
            and boot_id != before.get("boot_id")
        ):
            return current
        return None

    return ctx.wait_until(
        accept,
        timeout_seconds=REBOOT_BUDGET,
        poll_seconds=REBOOT_POLL_SECONDS,
        name="reboot",
    )


def residue_errors(snapshot: dict[str, Any]) -> list[str]:
    """Nothing of the window survived the reboot; the unit reads a real root."""

    errors = [
        f"{key} is still present on the new boot"
        for key in ("window_open", "dropin", "state_dir", "deadman")
        if snapshot.get(key)
    ]
    root = snapshot.get("unit_pci_root")
    if isinstance(root, str) and root.startswith(FIXTURE_PREFIX):
        errors.append(f"the host collector still reads the fixture PCI root {root}")
    if snapshot.get("unit_active_state") not in (None, "active"):
        errors.append(
            f"the host collector is {snapshot.get('unit_active_state')} on the new boot"
        )
    return errors


def _node_workflows(observed: dict[str, Any]) -> list[dict[str, Any]]:
    return list(observed.get("workflows") or [])


def reboot_receipt_errors(
    observed: dict[str, Any], incident_ids: list[str], node: str
) -> list[str]:
    """Exactly one SUCCEEDED RESTART_NODE execution and remote command, node-scoped."""

    errors = []
    workflows = [
        workflow
        for incident_id in incident_ids
        for workflow in workflows_of(observed, incident_id)
    ]
    request_ids = {str(item.get("request_id")) for item in workflows}
    succeeded = [
        item
        for workflow in workflows
        for item in executions(workflow, "RESTART_NODE")
        if item.get("status") == "SUCCEEDED"
    ]
    if len(succeeded) != 1:
        errors.append(
            f"{len(succeeded)} SUCCEEDED RESTART_NODE executions (expected exactly one)"
        )
    elif not succeeded[0].get("adapter_operation_id"):
        errors.append("the RESTART_NODE execution names no adapter operation")
    commands = [
        item
        for item in observed.get("commands") or []
        if (item.get("step") or {}).get("operation") == "RESTART_NODE"
        and str(item.get("workflow_request_id")) in request_ids
    ]
    if not commands:
        errors.append("no RESTART_NODE remote command was recorded for the chain")
    for command in commands:
        if command.get("status") != "SUCCEEDED" or command.get("error"):
            errors.append(
                f"RESTART_NODE command {command.get('command_id')} ended "
                f"{command.get('status')}: {str(command.get('error') or '')[:200]}"
            )
        if (command.get("step") or {}).get("node_ids") != [node]:
            errors.append(
                f"RESTART_NODE command scope {(command.get('step') or {}).get('node_ids')} "
                f"is not [{node!r}]"
            )
    return errors


def single_reboot_errors(observed: dict[str, Any]) -> list[str]:
    """One reboot planned and executed on the node in the whole read; no replacement."""

    errors = []
    planned = sum(
        operations(workflow).count("RESTART_NODE")
        for workflow in _node_workflows(observed)
    )
    if planned != 1:
        errors.append(
            f"{planned} RESTART_NODE steps planned on the node (expected one)"
        )
    executed = sum(
        1
        for workflow in _node_workflows(observed)
        for item in executions(workflow, "RESTART_NODE")
        if item.get("status") == "SUCCEEDED"
    )
    if executed > 1:
        errors.append(f"{executed} RESTART_NODE executions SUCCEEDED (expected one)")
    replacing = [
        str(workflow.get("request_id"))
        for workflow in _node_workflows(observed)
        if "REPLACE_NODE" in operations(workflow)
    ] + [
        str(command.get("command_id"))
        for command in observed.get("commands") or []
        if (command.get("step") or {}).get("operation") == "REPLACE_NODE"
    ]
    if replacing:
        errors.append(f"REPLACE_NODE planned or commanded: {replacing}")
    return errors


def fresh_inventory_errors(
    observed: dict[str, Any], *, expected: int, after: datetime
) -> list[str]:
    """Post-reboot product samples: the full count, no mismatch, newer than ``after``."""

    def fresh(name: str) -> list[dict[str, Any]]:
        found = [
            item
            for item in _numeric(samples(observed, name))
            if (moment := parse_time(item.get("observed_at"))) is not None
            and moment > after
        ]
        for item in observed.get("metric_latest") or []:
            moment = parse_time(item.get("observed_at"))
            if (
                item.get("name") == name
                and isinstance(item.get("value"), (int, float))
                and moment is not None
                and moment > after
            ):
                found.append(dict(item))
        return sorted(found, key=lambda item: str(item.get("observed_at")))

    errors = []
    active = fresh("efa_inventory_active_count")
    mismatch = fresh(MISMATCH_METRIC)
    anchor = f"after the reboot completed at {after.isoformat()}"
    if not active or active[-1]["value"] != expected:
        errors.append(
            f"no fresh efa_inventory_active_count={expected} {anchor} "
            f"(latest {active[-1]['value'] if active else None})"
        )
    if not mismatch or mismatch[-1]["value"] != 0:
        errors.append(
            f"no fresh {MISMATCH_METRIC}=0 {anchor} "
            f"(latest {mismatch[-1]['value'] if mismatch else None})"
        )
    return errors


def reboot_completed_at(
    observed: dict[str, Any], incident_ids: Iterable[str]
) -> datetime | None:
    """When the chain's RESTART_NODE SUCCEEDED, by the product's own record.

    The product's inventory gate judges only samples newer than its completed
    recovery step (``adapters/gpu_validation.inventory_observation_window``:
    the latest ``updated_at`` of the SUCCEEDED recovery executions), so the
    runner's freshness anchor is the same moment. Anchoring on the runner's
    clock after its post-reboot bookkeeping (node settle, probe re-create,
    snapshot, read-back) rejected the new boot's first batch -- the only one
    the edge filter delivers until the next health summary -- and failed a
    SUCCEEDED chain (live hp 2026-10-27: "no fresh efa_inventory_active_count
    =16 after the reboot (latest None)").
    """

    moments = [
        moment
        for incident_id in incident_ids
        for workflow in workflows_of(observed, incident_id)
        for item in executions(workflow, "RESTART_NODE")
        if item.get("status") == "SUCCEEDED"
        and (moment := parse_time(item.get("updated_at"))) is not None
    ]
    return max(moments) if moments else None


def node_restored_errors(
    before: dict[str, Any], after: dict[str, Any], rebooted: dict[str, Any] | None
) -> list[str]:
    errors = []
    if after.get("ready") != "True":
        errors.append(f"the node is Ready={after.get('ready')}")
    if node_isolated(after):
        errors.append("the node is still isolated (cordoned, tainted or owned)")
    if after.get("uid") != before.get("uid"):
        errors.append("the Node UID changed")
    if rebooted is not None and after.get("boot_id") != rebooted.get("boot_id"):
        errors.append(
            f"the boot id changed again after the reboot ({rebooted.get('boot_id')} "
            f"-> {after.get('boot_id')})"
        )
    for key in ("efa_allocatable", "gpu_allocatable"):
        if after.get(key) != before.get(key):
            errors.append(
                f"{key} {after.get(key)!r} differs from baseline {before.get(key)!r}"
            )
    return errors


def hyperpod_logical_id(ctx: VariantContext, instance_id: str | None) -> dict[str, Any]:
    """The SageMaker ``NodeLogicalId`` bound to ``instance_id`` on the lane's cluster.

    The product reboots a HyperPod node by logical id (``hyperpod.py``:
    ``BatchRebootClusterNodes(NodeLogicalIds=...)``, ``IncludeNodeLogicalIds``
    required), so its CloudTrail receipt carries ``request_node_logical_ids``
    and a null ``request_node_ids``: matching the instance id there never
    succeeds (live hp 2026-10-27: "no accepted [...] event naming i-...; 1
    accepted reboot events name another target: BatchRebootClusterNodes:None"
    on the receipt that named this very node). The binding is the provider's
    own -- ``list-cluster-nodes --include-node-logical-ids``, the COLLECT-004
    scope's read (``collector_reboot_scope``) -- taken before anything is
    mutated; a read that fails or binds no single node is returned as its
    error and fails the receipt stage by name, never silently.
    """

    cluster = ctx.settings.hyperpod_cluster
    if not instance_id or not cluster:
        return {
            "node_logical_id": None,
            "error": "no EC2 instance id or no --hyperpod-cluster to resolve it on",
            "cluster": cluster or None,
        }
    try:
        page = json.loads(
            ctx.regional.run(
                [
                    "aws",
                    "sagemaker",
                    "list-cluster-nodes",
                    "--cluster-name",
                    cluster,
                    "--include-node-logical-ids",
                    "--region",
                    ctx.regional.settings.region,
                    "--output",
                    "json",
                ],
                timeout=HYPERPOD_INVENTORY_TIMEOUT,
            ).stdout
            or "null"
        )
    except Exception as error:  # recorded and graded by the receipt stage
        return {
            "node_logical_id": None,
            "error": f"{type(error).__name__}: {str(error)[:200]}",
            "cluster": cluster,
        }
    page = page if isinstance(page, dict) else {}
    bound = sorted(
        {
            str(row["NodeLogicalId"])
            for row in page.get("ClusterNodeSummaries") or []
            if isinstance(row, dict)
            and row.get("InstanceId") == instance_id
            and row.get("NodeLogicalId")
        }
    )
    if len(bound) != 1 or page.get("NextToken"):
        return {
            "node_logical_id": None,
            "error": f"{len(bound)} HyperPod nodes of {cluster} bind {instance_id}"
            + (" (inventory incomplete)" if page.get("NextToken") else ""),
            "cluster": cluster,
        }
    return {"node_logical_id": bound[0], "error": None, "cluster": cluster}


def _targets_text(item: dict[str, Any]) -> str:
    return (
        f"{item.get('event_name')}:{item.get('request_node_ids')}/"
        f"{item.get('request_node_logical_ids')}"
    )


def provider_receipt(
    ctx: VariantContext,
    *,
    provider: str | None,
    instance_id: str | None,
    since: datetime,
    node_logical_id: str | None = None,
    binding_error: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The provider's own reboot record (CloudTrail) names this node, no other.

    EC2 ``RebootInstances`` names the instance; SageMaker ``BatchRebootCluster
    Nodes`` names the ``NodeLogicalId`` the product submitted, which
    :func:`hyperpod_logical_id` bound to the instance (``node_logical_id``).
    """

    key = PROVIDER_KEYS.get(provider or "")
    if key is None:
        return [f"no provider reboot API is known for lane {provider!r}"], []
    if not instance_id:
        return ["the node's providerID names no EC2 instance"], []
    names = set(reboot_event_names(key))
    events = ctx.regional.wait_provider_events(
        since,
        event_names=names,
        expected_count=1,
        timeout_seconds=PROVIDER_RECEIPT_BUDGET,
        event_source=provider_event_source(key),
    )
    accepted = [item for item in events if not item.get("api_error_code")]

    def ours(target: Any) -> bool:
        text = str(target)
        return instance_id in text or (
            node_logical_id is not None and text == node_logical_id
        )

    judged = [
        (
            item,
            [
                ours(target)
                for target in (
                    *(item.get("request_node_ids") or []),
                    *(item.get("request_node_logical_ids") or []),
                )
            ],
        )
        for item in accepted
    ]
    mine = [item for item, flags in judged if any(flags)]
    foreign = [item for item, flags in judged if not any(flags)]
    shared = [item for item, flags in judged if any(flags) and not all(flags)]
    errors = []
    if key == "hyperpod" and node_logical_id is None:
        errors.append(
            f"the node's HyperPod NodeLogicalId is unresolved ({binding_error}); "
            "BatchRebootClusterNodes names logical ids only"
        )
    if not mine:
        errors.append(
            f"no accepted {sorted(names)} event naming {instance_id}"
            + (f" or its NodeLogicalId {node_logical_id}" if node_logical_id else "")
            + f" within {PROVIDER_RECEIPT_BUDGET}s"
            + (" (CloudTrail may still be delivering)" if not events else "")
        )
    if len(mine) > 1:
        errors.append(f"{len(mine)} reboot events name {instance_id} (expected one)")
    if shared:
        errors.append(
            f"{len(shared)} reboot events name this node together with another "
            "target: " + ", ".join(_targets_text(item) for item in shared)[:400]
        )
    if foreign:
        errors.append(
            f"{len(foreign)} accepted reboot events name another target: "
            + ", ".join(_targets_text(item) for item in foreign)[:400]
        )
    return errors, accepted


def peer_errors(
    ctx: VariantContext, before: dict[str, Any] | None, since: datetime
) -> list[str]:
    """``--node-b`` untouched: same boot, Ready, not isolated, no incident of its own."""

    peer = ctx.settings.peer_node
    if not peer or before is None:
        return []
    after = ctx.regional.node_snapshot(peer)
    errors = []
    if after.get("boot_id") != before.get("boot_id"):
        errors.append(
            f"{peer} rebooted ({before.get('boot_id')} -> {after.get('boot_id')})"
        )
    if after.get("ready") != "True":
        errors.append(f"{peer} is Ready={after.get('ready')}")
    if node_isolated(after):
        errors.append(f"{peer} is isolated")
    peer_activity = node_activity(
        ctx.regional, ctx.settings.regional.cluster_id, peer, since
    )
    incidents = [
        str(item.get("incident_id")) for item in peer_activity.get("incidents") or []
    ]
    if incidents:
        errors.append(f"{peer} has incidents since the window opened: {incidents}")
    return errors


def hand_back(ctx: VariantContext, incidents: list[dict[str, Any]]) -> dict[str, Any]:
    """Cleanup: the product's validated restore if the node is still isolated."""

    node = ctx.regional.node_snapshot(ctx.settings.node)
    if not node_isolated(node):
        return {"skipped": True, "reason": "the node is not isolated"}
    return {
        "restored": restore_incidents(
            ctx,
            incidents,
            reason=(
                "GF-GENERAL-LIVE-013 pci_missing: the replayed PCI root is closed; "
                "hand the node back on fresh evidence"
            ),
        )
    }


def _close_window(replay: HostSourceReplay) -> dict[str, Any]:
    """Close the window; a node already rebooting makes that fail harmlessly."""

    try:
        return replay.close()
    except HostProbeError as error:
        return {
            "clean": False,
            "error": f"{type(error).__name__}: {error}",
            "note": (
                "the probe could not be reached while closing: the window's /run "
                "drop-in, deadman and fixture do not survive the reboot; verified "
                "on the new boot (post_reboot_clean)"
            ),
        }


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes,
        **extra,
    }


# --------------------------------------------------------------------------- #
# the variant
# --------------------------------------------------------------------------- #
def _negative_failed(
    ctx: VariantContext,
    negative: dict[str, Any],
    stages: dict[str, list[str]],
    notes: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """The product minted a finding from a failed read: FAIL, chain waited out.

    Its chain (REBOOT_NODE, if zero functions were graded as missing) is never
    cancelled; the missing-function stage is not run on top of it.
    """

    incident_ids = [str(item["incident_id"]) for item in negative["incidents"]]
    ctx.cleanup.push("validated-restore", lambda: hand_back(ctx, negative["incidents"]))
    settled = wait_settled(
        ctx,
        negative["started"],
        incident_ids,
        timeout_seconds=STEP_TIMEOUT_SECONDS
        + REBOOT_BUDGET
        + POST_REBOOT_SETTLE_BUDGET,
        metrics=EFA_METRICS,
        name="negative-settled",
    )
    stages["positive_skipped"] = [
        "the unreadable-root stage minted an efa_inventory_mismatch incident; "
        "the missing-function stage was not run"
    ]
    stages["negative_no_reboot"] = hardware_errors(settled)
    return _result(stages, notes=notes, extra={**extra, "negative_settled": True})


def _inject_missing_function(
    ctx: VariantContext,
    replay: HostSourceReplay,
    *,
    device: str,
    bdf: str,
    expected: int,
) -> dict[str, Any]:
    """Open the missing-function window, wait for the finding, close at once."""

    require_action_time(POSITIVE_WINDOW_SECONDS)
    pci_built = replay.build_pci(source=PCI_ROOT, drop=[bdf])
    ctx.evidence.write("pci-fixture", pci_built)
    if pci_built.get("efa_devices") != expected - 1:
        raise VariantBlocked(
            "FIXTURE_NOT_READY",
            f"the node's PCI root lists {pci_built.get('efa_devices')} EFA functions "
            f"without {bdf}; the product expects {expected}",
        )
    tree = replay.build_infiniband(source=INFINIBAND_ROOT, drop=[device])
    ctx.evidence.write("infiniband-fixture", tree)
    started = utc_now()
    opened = replay.open_window(
        trees=[(INFINIBAND_TREE, INFINIBAND_ROOT)],
        pci_root=True,
        restore_seconds=DETECTION_BUDGET + 300,
    )
    ctx.evidence.write("source-window-open", opened)
    detection = wait_for_incident(
        ctx,
        started,
        MISMATCH_METRIC,
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="pci-missing-detection",
    )
    incidents = finding_incidents(detection, MISMATCH_METRIC)
    closed = _close_window(replay)
    ctx.evidence.write("source-window-close", closed)
    missing = [
        item
        for item in _numeric(samples(detection, "efa_inventory_missing_count"))
        if item["value"] >= 1
    ]
    if not missing:
        raise VariantInconclusive(
            "the product never reported efa_inventory_missing_count >= 1 from the "
            f"replayed PCI root without {bdf}"
        )
    ctx.evidence.append_oracle(
        [
            {
                "dropped_bdf": bdf,
                "dropped_rdma_device": device,
                "pci_fixture": pci_built,
                "window_closed_at": utc_now().isoformat(),
                **replay.evidence(device=device),
            }
        ],
        source="pci-root",
    )
    return {
        "started": started,
        "detection": detection,
        "incidents": incidents,
        "closed": closed,
    }


def _follow_reboot(
    ctx: VariantContext,
    replay: HostSourceReplay,
    *,
    before: dict[str, Any],
    started: datetime,
    incident_ids: list[str],
    stages: dict[str, list[str]],
) -> dict[str, Any] | None:
    """Wait for the planned reboot; on the new boot re-create the probe and
    prove nothing of the window survived. Returns the rebooted snapshot or None."""

    planned = wait_reboot_planned(ctx, started, incident_ids)
    stages["reboot_planned"] = (
        []
        if planned is not None
        else [
            f"no workflow of {incident_ids} planned RESTART_NODE within "
            f"{STEP_TIMEOUT_SECONDS}s"
        ]
    )
    if planned is None:
        stages["rebooted"] = [
            "no RESTART_NODE was planned, so no reboot was waited for"
        ]
        return None
    rebooted = wait_reboot(ctx, before)
    if rebooted is None:
        stages["rebooted"] = [
            f"the node kept boot id {before.get('boot_id')}; no reboot within "
            f"{REBOOT_BUDGET}s of the plan"
        ]
        return None
    # When the new boot was first seen: the freshness anchor's fallback when
    # the product's RESTART_NODE record carries no completion time.
    rebooted = {**rebooted, "observed_at": utc_now().isoformat()}
    stages["rebooted"] = []
    settled_node = wait_node_settled_after_reboot(
        ctx.regional,
        node=ctx.settings.node,
        boot_id=str(rebooted["boot_id"]),
        clock=_ContextClock(ctx),
    )
    ctx.evidence.write("node-settled", settled_node)
    ctx.evidence.write("probe-recreate", replay.recreate())
    residue = replay.snapshot()
    ctx.evidence.write("post-reboot-snapshot", residue)
    stages["post_reboot_clean"] = residue_errors(residue)
    readback = replay.read_back(INFINIBAND_ROOT)
    ctx.evidence.write("post-reboot-read-back", readback)
    stages["real_reads"] = (
        []
        if readback.get("same") is True
        else ["the host collector unit does not read the real RDMA tree again"]
    )
    return rebooted


def _grade_settled(
    ctx: VariantContext,
    *,
    before: dict[str, Any],
    peer_before: dict[str, Any] | None,
    provider: str | None,
    started: datetime,
    incident_ids: list[str],
    rebooted: dict[str, Any] | None,
    rebooted_at: datetime,
    expected: int,
    binding: dict[str, Any],
    stages: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grade the settled chain on the product's records; returns (observed, events).

    ``rebooted_at`` is when the runner first saw the new boot: the fresh-inventory
    anchor only when the product's record names no RESTART_NODE completion.
    """

    settled = wait_settled(
        ctx,
        started,
        incident_ids,
        timeout_seconds=POST_REBOOT_SETTLE_BUDGET,
        metrics=EFA_METRICS,
        name="pci-missing-settled",
    )
    observed = settled or activity(ctx, started, metrics=EFA_METRICS)
    stages["chain"] = chain_errors(
        observed,
        ChainExpectation(
            metric=MISMATCH_METRIC,
            operations=REBOOT_CHAIN,
            validation="VALIDATE_FABRIC",
            incident_states=frozenset({"RECOVERED"}),
            forbidden=FORBIDDEN_OPERATIONS,
        ),
    )
    stages["reboot_receipts"] = reboot_receipt_errors(
        observed, incident_ids, ctx.settings.node
    )
    stages["single_reboot"] = single_reboot_errors(observed)
    after = node_identity(ctx)
    ctx.evidence.write("node-after", after)
    stages["node_restored"] = node_restored_errors(before, after, rebooted)
    completed_at = reboot_completed_at(observed, incident_ids)
    ctx.evidence.write(
        "fresh-inventory-anchor",
        {
            "restart_node_completed_at": (
                completed_at.isoformat() if completed_at else None
            ),
            "new_boot_observed_at": rebooted_at.isoformat(),
        },
    )
    stages["fresh_inventory"] = fresh_inventory_errors(
        observed, expected=expected, after=completed_at or rebooted_at
    )
    events: list[dict[str, Any]] = []
    if rebooted is None:
        stages["provider_receipt"] = [
            "no reboot happened, so no provider record exists"
        ]
    else:
        stages["provider_receipt"], events = provider_receipt(
            ctx,
            provider=provider,
            instance_id=before.get("instance_id"),
            since=started,
            node_logical_id=binding.get("node_logical_id"),
            binding_error=binding.get("error"),
        )
    ctx.evidence.write("provider-events", {"events": events})
    stages["peer_untouched"] = peer_errors(ctx, peer_before, started)
    return observed, events


def pci_missing(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: one EFA function absent -> PCI_DEVICE_MISSING -> the reboot chain."""

    replay = host_source_replay(ctx)
    replay.create()
    before = node_identity(ctx)
    peer_before = (
        ctx.regional.node_snapshot(ctx.settings.peer_node)
        if ctx.settings.peer_node
        else None
    )
    provider = node_provider(
        (ctx.regional.node_metadata(ctx.settings.node) or {}).get("labels") or {}
    )
    # HyperPod reboots by NodeLogicalId: bind it to the instance before anything
    # is mutated, so the receipt is read against the provider's own mapping.
    binding = (
        hyperpod_logical_id(ctx, before.get("instance_id"))
        if provider == "hp"
        else {"node_logical_id": None, "error": None}
    )
    before["node_logical_id"] = binding.get("node_logical_id")
    ctx.evidence.write(
        "node-before",
        {
            "node": before,
            "peer": peer_before,
            "provider": provider,
            "hyperpod_binding": binding,
        },
    )
    inventory = replay.inventory(source=INFINIBAND_ROOT)
    ctx.evidence.write("infiniband-inventory", inventory)
    functions = efa_functions(inventory)
    if not functions:
        raise VariantBlocked(
            "CAPABILITY_ABSENT",
            "the node lists no EFA RDMA device with an ACTIVE/LinkUp port and a "
            "resolvable PCI function",
        )
    expected = expected_efa_count(ctx, len(functions))
    device, bdf = functions[-1]
    negative = unreadable_root(ctx, replay)
    stages: dict[str, list[str]] = dict(negative["stages"])
    extra: dict[str, Any] = {
        "device": device,
        "bdf": bdf,
        "expected_count": expected,
        "provider": provider,
        "instance_id": before.get("instance_id"),
        "boot_id_before": before.get("boot_id"),
        "negative_collection_errors": negative["collection_errors"],
        **replay.evidence(),
    }
    notes = [
        f"negative stage: the PCI root was unreadable (ENOTDIR) for {NEGATIVE_TICKS} "
        "host ticks -- twice the confirmation window -- and the product's reading "
        "must be UNKNOWN with no count, finding, reboot or isolation",
    ]
    if negative["incidents"]:
        return _negative_failed(ctx, negative, stages, notes, extra)
    injected = _inject_missing_function(
        ctx, replay, device=device, bdf=bdf, expected=expected
    )
    started, detection = injected["started"], injected["detection"]
    incidents = injected["incidents"]
    extra["window_close"] = injected["closed"]
    stages["failure_mode"] = efa_failure_mode_errors(detection, PCI_DEVICE_MISSING)
    stages["confirmed_sample"] = confirmed_sample_errors(detection, expected=expected)
    stages["finding"] = (
        []
        if incidents
        else [f"the confirmed {PCI_DEVICE_MISSING} mismatch minted no incident"]
    )
    if not incidents:
        return _result(stages, notes=notes, extra=extra)
    incident_ids = [str(item["incident_id"]) for item in incidents]
    ctx.cleanup.push("validated-restore", lambda: hand_back(ctx, incidents))
    rebooted = _follow_reboot(
        ctx,
        replay,
        before=before,
        started=started,
        incident_ids=incident_ids,
        stages=stages,
    )
    rebooted_at = parse_time((rebooted or {}).get("observed_at")) or utc_now()
    extra["boot_id_after"] = (rebooted or {}).get("boot_id")
    if provider == "hp" and binding.get("node_logical_id") is None:
        # The pre-window read failed: one more try before the receipt is graded.
        binding = hyperpod_logical_id(ctx, before.get("instance_id"))
        before["node_logical_id"] = binding.get("node_logical_id")
    extra["node_logical_id"] = binding.get("node_logical_id")
    observed, events = _grade_settled(
        ctx,
        before=before,
        peer_before=peer_before,
        provider=provider,
        started=started,
        incident_ids=incident_ids,
        rebooted=rebooted,
        rebooted_at=rebooted_at,
        expected=expected,
        binding=binding,
        stages=stages,
    )
    notes.append(
        f"{bdf} ({device}) was absent from the replayed PCI root and RDMA tree only; "
        "the window was closed the moment the finding existed, so every read behind "
        "the reboot is real; nothing under /run survives the reboot"
    )
    return _result(
        stages,
        notes=notes,
        extra={
            **extra,
            "provider_events": [
                {
                    "event_name": item.get("event_name"),
                    "event_time": item.get("event_time"),
                    "request_node_ids": item.get("request_node_ids"),
                    "request_node_logical_ids": item.get("request_node_logical_ids"),
                }
                for item in events
            ],
            "settled": [
                {
                    "request_id": item.get("request_id"),
                    "status": item.get("status"),
                    "operations": operations(item),
                }
                for item in observed.get("workflows") or []
            ],
        },
    )


VARIANTS = [
    VariantSpec(
        case_id=CASE_013,
        variant="pci_missing",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="H",
        flags="X",
        injection=(
            "first an unreadable PCI root (a regular file at GPU_FAULT_PCI_DEVICES_ROOT) "
            f"inside the host collector unit for {NEGATIVE_TICKS} ticks; then a copy of "
            "the node's real PCI root with one real EFA function absent plus the real "
            "/sys/class/infiniband without that function's RDMA device, bound into the "
            "unit until the finding exists"
        ),
        expected=(
            "unreadable root: 'EFA inventory observation unavailable' only, no "
            "inventory count (the real RDMA tree's rdma_link_down keeps flowing), "
            "no finding, no reboot, no isolation; missing function: "
            "efa_inventory_mismatch failure_mode=PCI_DEVICE_MISSING confirmed after the "
            "consecutive samples -> REBOOT_NODE (MARK_UNSCHEDULABLE, QUARANTINE, exactly "
            "one RESTART_NODE through the provider, VALIDATE_GPU/HOST/FABRIC on "
            "post-reboot evidence, RESTORE_SCHEDULING); new boot id, node Ready and "
            "schedulable, EFA/GPU allocatable at baseline, incident RECOVERED, no "
            "REPLACE_NODE, CloudTrail names this node only (EC2 by instance id, "
            "SageMaker by its NodeLogicalId), --node-b untouched"
        ),
        cleanup=(
            "close the window the moment the finding exists (deadman-backed); the "
            "product's reboot chain is waited to settle, never cancelled; the probe "
            "Pod is re-created on the new boot and proves no drop-in, deadman, state "
            "or fixture PCI root survived; a node left isolated is handed back "
            "through the validated restore"
        ),
        max_minutes=90,
        approvals=(HOST_SOURCE_REPLAY, NODE_REBOOT),
        execute=pci_missing,
    ),
]

__all__ = [
    "CASE_013",
    "FORBIDDEN_OPERATIONS",
    "NEGATIVE_TICKS",
    "PCI_ROOT",
    "POST_REBOOT_SETTLE_BUDGET",
    "PROVIDER_KEYS",
    "REBOOT_BUDGET",
    "REBOOT_CHAIN",
    "UNKNOWN_MARKER",
    "VARIANTS",
    "confirmed_sample_errors",
    "efa_functions",
    "expected_efa_count",
    "fresh_inventory_errors",
    "hand_back",
    "hyperpod_logical_id",
    "node_identity",
    "node_restored_errors",
    "pci_missing",
    "peer_errors",
    "provider_receipt",
    "reboot_completed_at",
    "reboot_receipt_errors",
    "residue_errors",
    "single_reboot_errors",
    "unreadable_root",
    "wait_reboot",
    "wait_reboot_planned",
]
