"""Provider-side proof that a node Kubernetes no longer has is really gone.

On HyperPod the node name carries the EC2 instance id and ``aws sagemaker
list-cluster-nodes`` is the membership (``hyperpod_departed_node_evidence``).
A plain EKS cluster has neither: node names are private DNS names and the only membership
is the instance itself. The proof here is therefore the Node absent from
Kubernetes *and* ``aws ec2 describe-instances`` answering ``terminated`` -- or
nothing at all -- for the instance id the fleet recorded for that node. That
id is the Agent record's ``node_instance_id`` (the installer passes the
providerID's instance id on eks-managed nodes; a HyperPod-era record carries
a Node UID, which is no EC2 id and therefore no proof). A still-running,
stopped or shutting-down instance, a missing or unusable recorded id, a
cluster entry without a region and a failed lookup all keep the node blocked
and say why; nothing here ever reads as "absent" by default.

The callers pass ``run_command`` / ``run_script`` in so the test doubles that
patch ``workflow_reconcile`` keep working and this module imports nothing from
it (no import cycle).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text

# The verdicts are the shared rule (``gpu_fault.departed_nodes``), which the
# control plane's departed-node cleanup judges with too; the names stay
# importable from here for the reconcile paths and the release engine.
from gpu_fault.departed_nodes import (
    EC2_SOURCE,
    EKS_PROVIDER,
    HYPERPOD_NODE_NAME,
    HYPERPOD_PROVIDER,
    INSTANCE_ID,
    NO_HYPERPOD_INSTANCE_BLOCKER,
    NO_RECORDED_INSTANCE_BLOCKER,
    NOT_FOUND,
    TERMINATED,
    cluster_provider_of,
    ec2_state_blocker,
    hyperpod_instance_id,
    hyperpod_listing_blocker,
    provider_lookup_blocker,
    recorded_instance_id,
)

__all__ = [
    "EC2_SOURCE",
    "EKS_PROVIDER",
    "HYPERPOD_NODE_NAME",
    "HYPERPOD_PROVIDER",
    "INSTANCE_ID",
    "NOT_FOUND",
    "TERMINATED",
    "RecordedInstanceIds",
    "cluster_provider_of",
    "departed_node_evidence",
    "ec2_instance_states",
    "hyperpod_departed_node_evidence",
    "hyperpod_instance_ids",
    "recorded_instance_ids",
]

# Pages of ``list-cluster-nodes`` read before the lookup is declared broken.
HYPERPOD_INVENTORY_PAGE_LIMIT = 100
#: The cache slot a failed lookup occupies for its cluster.
LOOKUP_ERROR = "__error__"

# Reads one field of the Agent records the Pod holds; ``store.get_agent`` and
# ``node_instance_id`` predate every release this can be asked to repair.
AGENT_INSTANCE_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext

payload = json.load(sys.stdin)
store = ApplicationContext.from_environment().store
instance_ids = {}
for node_id in payload["node_ids"]:
    try:
        record = store.get_agent(payload["cluster_id"], node_id)
    except KeyError:
        continue
    instance_ids[node_id] = record.node_instance_id
print(json.dumps({"mode": "agent-instance-ids", "instance_ids": instance_ids}))
"""


def recorded_instance_ids(
    site: Any,
    cluster_id: str,
    node_ids: Sequence[str],
    *,
    run_script: Callable[..., dict[str, Any]],
) -> dict[str, str | None]:
    """``node_id -> node_instance_id`` from the fleet's Agent records.

    A node with no Agent record is absent from the answer; a record whose
    instance id is not EC2-shaped maps to ``None``.
    """

    result = run_script(
        site,
        {
            "mode": "agent-instance-ids",
            "cluster_id": cluster_id,
            "node_ids": sorted({str(item) for item in node_ids}),
        },
        script=AGENT_INSTANCE_SCRIPT,
    )
    values = result.get("instance_ids")
    if not isinstance(values, dict):
        raise BootstrapError("agent instance ids from the control plane are invalid")
    return {
        str(node_id): recorded_instance_id(value) for node_id, value in values.items()
    }


def ec2_instance_states(
    region: str,
    instance_ids: Sequence[str],
    *,
    run_command: Callable[..., Any],
) -> dict[str, str]:
    """``instance_id -> EC2 state name``; an id EC2 no longer knows is absent.

    Read by filter, not ``--instance-ids``: the latter fails the whole call on
    one unknown id, and "unknown to EC2" is exactly the answer wanted here.
    """

    unique = sorted({str(item) for item in instance_ids if str(item).strip()})
    if not unique:
        return {}
    completed = run_command(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--filters",
            "Name=instance-id,Values=" + ",".join(unique),
            "--query",
            "Reservations[].Instances[].[InstanceId,State.Name]",
            "--output",
            "json",
        ],
        timeout_seconds=120,
    )
    if completed.returncode:
        raise BootstrapError(
            f"cannot describe EC2 instances in {region}: "
            f"{diagnostic_text(completed.stderr, sensitive=True)}"
        )
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError("EC2 instance description is invalid JSON") from exc
    if not isinstance(rows, list):
        raise BootstrapError("EC2 instance description is invalid")
    states: dict[str, str] = {}
    for row in rows:
        if (
            isinstance(row, list)
            and len(row) == 2
            and isinstance(row[0], str)
            and isinstance(row[1], str)
        ):
            states[row[0]] = row[1].strip().lower()
        else:
            raise BootstrapError("EC2 instance description is invalid")
    return states


def _state_of(
    target: Mapping[str, Any],
    cluster_id: str,
    instance_id: str,
    *,
    aws_region: object,
    states: dict[str, dict[str, str]],
    read_at: dict[str, str] | None,
    run_command: Callable[..., Any],
) -> str:
    """The cached EC2 state of ``instance_id``; raises on an unreadable lookup."""

    cache = states.setdefault(cluster_id, {})
    if LOOKUP_ERROR in cache:
        raise BootstrapError(cache[LOOKUP_ERROR])
    if instance_id not in cache:
        region = str(target.get("region") or aws_region or "").strip()
        if not region:
            raise BootstrapError(
                f"GPU cluster {cluster_id} has no AWS region in the site"
            )
        if read_at is not None:
            read_at[EC2_SOURCE] = datetime.now(timezone.utc).isoformat()
        try:
            found = ec2_instance_states(region, [instance_id], run_command=run_command)
        except BootstrapError as exc:
            cache[LOOKUP_ERROR] = str(exc)
            raise
        cache[instance_id] = found.get(instance_id, NOT_FOUND)
    return cache[instance_id]


def departed_node_evidence(
    target: Mapping[str, Any],
    cluster_id: str,
    node_id: str,
    *,
    instance_id: str | None,
    aws_region: object,
    missing: Mapping[str, Any],
    states: dict[str, dict[str, str]],
    read_at: dict[str, str] | None,
    run_command: Callable[..., Any],
) -> dict[str, Any]:
    """Evidence for a node Kubernetes no longer has on an eks-managed cluster.

    ``missing`` is the Kubernetes-only verdict (``exists: False`` with its
    blocker); ``instance_id`` the fleet's recorded EC2 id for the node, or
    None. ``states`` caches one cluster's lookups (and its failure) for the
    plan being built; ``read_at`` records when EC2 was read (``ec2``).
    """

    base = {**missing, "provider": EKS_PROVIDER, "instance_id": instance_id}
    if not instance_id or recorded_instance_id(instance_id) is None:
        return {
            **base,
            "blockers": [*missing["blockers"], NO_RECORDED_INSTANCE_BLOCKER],
        }
    try:
        state = _state_of(
            target,
            cluster_id,
            instance_id,
            aws_region=aws_region,
            states=states,
            read_at=read_at,
            run_command=run_command,
        )
    except BootstrapError as exc:
        return {
            **base,
            "blockers": [*missing["blockers"], provider_lookup_blocker(exc)],
        }
    blocker = ec2_state_blocker(instance_id, state)
    if blocker is None:
        return {
            "node_id": node_id,
            "exists": False,
            "provider": EKS_PROVIDER,
            "instance_id": instance_id,
            "ec2_state": state,
            "absent_from_provider": True,
            "restored": True,
            "blockers": [],
        }
    return {
        "node_id": node_id,
        "exists": False,
        "provider": EKS_PROVIDER,
        "instance_id": instance_id,
        "ec2_state": state,
        "absent_from_provider": False,
        "restored": False,
        "blockers": [blocker],
    }


def hyperpod_instance_ids(
    site: Any,
    target: Mapping[str, Any],
    cluster_id: str,
    *,
    run_command: Callable[..., Any],
) -> frozenset[str]:
    """The EC2 instance ids HyperPod currently lists for ``cluster_id``.

    Read with ``aws sagemaker list-cluster-nodes`` in the cluster's region,
    every page. A cluster the site does not describe as HyperPod-managed, a
    failed call and an unreadable answer all raise ``BootstrapError``: the
    caller turns that into a blocker, never into "absent".
    """

    name = str(target.get("hyperpod_cluster_name") or "").strip()
    if not name:
        raise BootstrapError(
            f"GPU cluster {cluster_id} is not HyperPod-managed in the site"
        )
    region = str(
        target.get("region") or site.release_config.get("aws_region") or ""
    ).strip()
    if not region:
        raise BootstrapError(f"GPU cluster {cluster_id} has no AWS region in the site")
    instance_ids: set[str] = set()
    token: str | None = None
    for _page in range(HYPERPOD_INVENTORY_PAGE_LIMIT):
        command = [
            "aws",
            "sagemaker",
            "list-cluster-nodes",
            "--cluster-name",
            name,
            "--region",
            region,
            "--output",
            "json",
        ]
        if token:
            command.extend(["--next-token", token])
        completed = run_command(command, timeout_seconds=120)
        if completed.returncode:
            raise BootstrapError(
                f"cannot list HyperPod nodes of {cluster_id}: "
                f"{diagnostic_text(completed.stderr, sensitive=True)}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BootstrapError(
                f"HyperPod node inventory of {cluster_id} is invalid JSON"
            ) from exc
        summaries = (
            value.get("ClusterNodeSummaries") if isinstance(value, dict) else None
        )
        if not isinstance(summaries, list):
            raise BootstrapError(f"HyperPod node inventory of {cluster_id} is invalid")
        for row in summaries:
            if isinstance(row, dict):
                instance_id = row.get("InstanceId")
                if isinstance(instance_id, str) and instance_id.strip():
                    instance_ids.add(instance_id.strip())
        next_token = value.get("NextToken")
        if not next_token:
            return frozenset(instance_ids)
        if not isinstance(next_token, str) or next_token == token:
            raise BootstrapError(
                f"HyperPod node inventory of {cluster_id} repeats its page token"
            )
        token = next_token
    raise BootstrapError(
        f"HyperPod node inventory of {cluster_id} exceeds "
        f"{HYPERPOD_INVENTORY_PAGE_LIMIT} pages"
    )


def hyperpod_departed_node_evidence(
    site: Any,
    target: Mapping[str, Any],
    cluster_id: str,
    node_id: str,
    *,
    missing: Mapping[str, Any],
    provider_inventories: dict[str, Any],
    read_at: dict[str, str] | None,
    run_command: Callable[..., Any],
) -> dict[str, Any]:
    """Evidence for a HyperPod node Kubernetes no longer has.

    Restored only when the node name maps to an instance HyperPod no longer
    lists; ``provider_inventories`` caches one ``list-cluster-nodes`` read (or
    its failure) per cluster, ``read_at`` records it under ``hyperpod``.
    """

    instance_id = hyperpod_instance_id(node_id)
    if instance_id is None:
        return {
            **missing,
            "blockers": [*missing["blockers"], NO_HYPERPOD_INSTANCE_BLOCKER],
        }
    inventory = provider_inventories.get(cluster_id)
    if inventory is None:
        if read_at is not None:
            read_at["hyperpod"] = datetime.now(timezone.utc).isoformat()
        try:
            inventory = hyperpod_instance_ids(
                site, target, cluster_id, run_command=run_command
            )
        except BootstrapError as exc:
            inventory = str(exc)
        provider_inventories[cluster_id] = inventory
    if isinstance(inventory, str):
        return {
            **missing,
            "instance_id": instance_id,
            "blockers": [*missing["blockers"], provider_lookup_blocker(inventory)],
        }
    listed = hyperpod_listing_blocker(instance_id, inventory)
    if listed is not None:
        return {
            "node_id": node_id,
            "exists": False,
            "provider": HYPERPOD_PROVIDER,
            "instance_id": instance_id,
            "absent_from_provider": False,
            "restored": False,
            "blockers": [listed],
        }
    return {
        "node_id": node_id,
        "exists": False,
        "provider": HYPERPOD_PROVIDER,
        "instance_id": instance_id,
        "absent_from_provider": True,
        "restored": True,
        "blockers": [],
    }


class RecordedInstanceIds:
    """The fleet's recorded EC2 instance id per node, read once per eks cluster.

    Only an ``eks-managed`` cluster is asked, and only for the node ids the
    plan's items name; a read that fails leaves every id unknown, which the
    evidence reports as "provider membership unknown" rather than guessing.
    """

    def __init__(
        self,
        site: Any,
        items: Sequence[dict[str, Any]],
        *,
        cluster_target: Callable[[Any, str], dict[str, Any]],
        run_script: Callable[..., dict[str, Any]],
    ) -> None:
        self.site = site
        self.cluster_target = cluster_target
        self.run_script = run_script
        self.node_ids: dict[str, set[str]] = {}
        for item in items:
            cluster_id = str(item.get("cluster_id") or "")
            self.node_ids.setdefault(cluster_id, set()).update(
                str(value) for value in item.get("node_ids") or []
            )
        self.cache: dict[str, dict[str, str | None]] = {}

    def get(self, cluster_id: str, node_id: str) -> str | None:
        target = self.cluster_target(self.site, cluster_id)
        if cluster_provider_of(target) != EKS_PROVIDER:
            return None
        if cluster_id not in self.cache:
            try:
                self.cache[cluster_id] = recorded_instance_ids(
                    self.site,
                    cluster_id,
                    sorted(self.node_ids.get(cluster_id, set())),
                    run_script=self.run_script,
                )
            except BootstrapError:
                self.cache[cluster_id] = {}
        return self.cache[cluster_id].get(node_id)
