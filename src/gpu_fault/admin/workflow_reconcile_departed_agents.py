"""``workflow-reconcile --retire-departed-agents``: the Agent record of a node that left.

When HyperPod reclaims a spot GPU node the Agent process dies with the
instance. Nothing revokes its fleet record: the Agent's own lifecycle ends
with its last heartbeat, the HyperPod lifecycle adapter revokes agents only
for nodes *it* replaces, and the record stays ``ACTIVE`` with an expired lease
for good. The release verification then reads one more Agent than there are
nodes (``Agent coverage drift``) and rolls the deploy back, every time.

This module is the administrator's answer. It plans with the same evidence as
the never-dispatched close (``workflow_reconcile_never_dispatched``): the node
must be absent from the cluster's Kubernetes Nodes *and* its instance absent
from ``aws sagemaker list-cluster-nodes``, both read on the deploy host. A node
that is NotReady, cordoned, still in Kubernetes, still listed by HyperPod, on a
cluster the site does not describe as HyperPod-managed, or behind a failed
lookup keeps the Agent ineligible, naming why; so does a heartbeat younger
than ``DEPARTED_AGENT_GUARD_AGE`` -- an Agent that spoke ten minutes ago is
mid-restart, not gone. ``DRAINING`` qualifies beside ``ACTIVE``: a drain whose
revoke never followed is the same residue.

The write goes through the registry's own lifecycle path, in the Pod:
``FleetRegistry.drain_agent`` then ``revoke_agent`` with one
``AgentTransitionRequest`` whose ``transition_id`` names this reconcile and
whose ``reason`` is the audit line carrying the departed-node evidence (the
sources, their verdicts and read times, the operator and the plan digest).
The record keeps both; the archive under ``--state-dir`` keeps the full plan.
Generation, lifecycle state and ``last_seen_at`` are compare-and-set in the
Pod; evidence that moves between plan and apply refuses the apply by field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from gpu_fault.admin import operator_identity
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_never_dispatched as never_dispatched
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite

RETIRE_DEPARTED_AGENTS = "retire-departed-agents"
DEPARTED_AGENT_GUARD_AGE = timedelta(minutes=10)
HISTORY_PATH = Path("workflow-reconcile/departed-agents")
# Read times are reported beside the evidence and left out of the digest: the
# verdicts (present/listed) and the record's keys bind the approval.
DIGEST_EXCLUDED_ITEM_FIELDS = frozenset({"evidence_read_at"})
RETIREABLE_LIFECYCLES = frozenset({"ACTIVE", "DRAINING"})
GUARD_REASON = "agent heartbeat is younger than the 10-minute departure guard"
NO_RECORD_REASON = "no Agent record in the fleet"
KUBERNETES_PRESENT_REASON = (
    "node is still in Kubernetes; an Agent is retired only for a node that "
    "left Kubernetes and HyperPod"
)
# The fields of an Agent record the plan carries and the apply re-checks.
RECORD_FIELDS = (
    "cluster_id",
    "node_id",
    "lifecycle_state",
    "generation",
    "node_instance_id",
    "agent_incarnation_id",
    "transition_id",
    "first_seen_at",
    "last_seen_at",
    "lease_expires_at",
    "agent_version",
    "runtime_profile_version",
)

# Runs against whatever ``gpu_fault`` is deployed in the CPU ingress Pod; the
# registry primitives it uses (``drain_agent``/``revoke_agent``) predate every
# release this can be asked to repair.
AGENT_SCRIPT = """
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentLifecycleState, AgentTransitionRequest

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
store = context.store
now = datetime.now(timezone.utc)
GUARD_AGE = timedelta(minutes=10)
GUARD_REASON = "agent heartbeat is younger than the 10-minute departure guard"
FIELDS = {
    "cluster_id",
    "node_id",
    "lifecycle_state",
    "generation",
    "node_instance_id",
    "agent_incarnation_id",
    "transition_id",
    "first_seen_at",
    "last_seen_at",
    "lease_expires_at",
    "agent_version",
    "runtime_profile_version",
}

if payload["mode"] == "list-agents":
    agents = []
    for cluster_id in payload["cluster_ids"]:
        agents.extend(
            record.model_dump(mode="json", include=FIELDS)
            for record in store.list_agents(cluster_id)
        )
    result = {
        "mode": "agent-inventory",
        "evaluated_at": now.isoformat(),
        "agents": sorted(agents, key=lambda item: (item["cluster_id"], item["node_id"])),
    }
elif payload["mode"] == "retire-departed-agents":
    registry = context.fleet_registry
    if registry is None:
        raise ValueError("the agent registry is not enabled in this Pod")
    retired = []
    failures = {}
    for item in payload["items"]:
        key = f"{item['cluster_id']}/{item['node_id']}"
        try:
            current = store.get_agent(item["cluster_id"], item["node_id"])
            if current.generation != item["generation"]:
                raise ValueError(
                    f"agent generation changed: expected {item['generation']}, "
                    f"found {current.generation}"
                )
            if current.lifecycle_state.value != item["lifecycle_state"]:
                raise ValueError(
                    "agent lifecycle state changed: expected "
                    f"{item['lifecycle_state']}, found {current.lifecycle_state.value}"
                )
            # The inventory serialized the heartbeat as JSON (``Z`` suffix); the
            # record yields ``+00:00``. Compare instants, not spellings (live
            # 2026-10-01: every apply failed on the suffix alone).
            expected_seen = datetime.fromisoformat(
                str(item["last_seen_at"]).replace("Z", "+00:00")
            )
            if expected_seen.tzinfo is None or current.last_seen_at != expected_seen:
                raise ValueError(
                    "agent heartbeat changed: expected "
                    f"{expected_seen.isoformat()}, found {current.last_seen_at.isoformat()}"
                )
            if current.last_seen_at > now - GUARD_AGE:
                raise ValueError(GUARD_REASON)
            evidence = item["departed_node_evidence"]
            if (
                evidence.get("node_id") != item["node_id"]
                or evidence.get("absent_from_kubernetes") is not True
                or evidence.get("absent_from_provider") is not True
            ):
                raise ValueError("departed-node evidence does not prove the node gone")
            if (
                current.node_instance_id
                and evidence.get("instance_id")
                and current.node_instance_id != evidence["instance_id"]
            ):
                raise ValueError(
                    "agent instance changed: the record names "
                    f"{current.node_instance_id}, the evidence {evidence['instance_id']}"
                )
            hyperpod = (evidence.get("sources") or {}).get("hyperpod") or {}
            read_at = evidence.get("read_at") or {}
            audit = (
                f"operator reconciliation {payload['reference']}: retired Agent "
                f"{item['node_id']} after its node left Kubernetes and HyperPod "
                f"(instance {evidence.get('instance_id')}, HyperPod cluster "
                f"{hyperpod.get('cluster_name')} in {hyperpod.get('region')}; "
                f"kubernetes read {read_at.get('kubernetes')}, hyperpod read "
                f"{read_at.get('hyperpod')}; actor {payload.get('actor')}; "
                f"admin plan {payload.get('admin_plan_sha256')})"
            )
            if current.lifecycle_state is AgentLifecycleState.ACTIVE:
                request = AgentTransitionRequest(
                    expected_generation=current.generation,
                    transition_id=payload["transition_id"],
                    reason=audit,
                )
                registry.drain_agent(item["cluster_id"], item["node_id"], request)
            elif current.lifecycle_state is AgentLifecycleState.DRAINING:
                # The drain happened under another transition and its revoke
                # never followed; the registry lets only that transition finish.
                request = AgentTransitionRequest(
                    expected_generation=current.generation,
                    transition_id=current.transition_id,
                    reason=audit,
                )
            else:
                raise ValueError(
                    f"agent lifecycle state is {current.lifecycle_state.value}"
                )
            revoked = registry.revoke_agent(item["cluster_id"], item["node_id"], request)
            retired.append(
                {
                    "agent": key,
                    "generation": revoked.generation,
                    "transition_id": revoked.transition_id,
                    "lifecycle_state": revoked.lifecycle_state.value,
                }
            )
        except Exception as exc:  # per-item isolation, reported like the apply
            failures[key] = f"{type(exc).__name__}: {exc}"
    result = {
        "mode": "retire-departed-agents-apply",
        "retired_agents": sorted(retired, key=lambda item: item["agent"]),
        "failed_agents": sorted(failures),
        "failures": dict(sorted(failures.items())),
    }
else:
    raise ValueError("unsupported departed-agent mode")
print(json.dumps(result, sort_keys=True))
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``--retire-departed-agents [--node ID ...]`` on ``workflow-reconcile``.

    Rides the same verb because it is the same kind of operator reconciliation
    with the same evidence, reference, dry run and archive.
    """

    parser.add_argument(
        "--retire-departed-agents",
        action="store_true",
        help=(
            "retire (REVOKE) the fleet Agent record of every node absent from "
            "Kubernetes and from the HyperPod node list for over 10 minutes; "
            "--dry-run prints the plan with its evidence, --node narrows it"
        ),
    )
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        metavar="ID",
        help="retire only this node's Agent record (repeatable)",
    )


def conflicting_flags(arguments: argparse.Namespace) -> list[str]:
    """The ``workflow-reconcile`` selectors that do not combine with this one."""

    chosen = {
        "--workflow-id": bool(getattr(arguments, "workflow_id", None)),
        "--incident-id": bool(getattr(arguments, "incident_id", None)),
        "--max-items": getattr(arguments, "max_items", None) is not None,
        "--close-incident": bool(getattr(arguments, "close_incident", None)),
        "--close-escalated": bool(getattr(arguments, "close_escalated", False)),
        "--close-quarantined": bool(getattr(arguments, "close_quarantined", False)),
    }
    return [flag for flag, given in chosen.items() if given]


def run_pod_script(site: RenderedSite, payload: dict[str, Any]) -> dict[str, Any]:
    return reconcile.run_control_plane_script(site, payload, script=AGENT_SCRIPT)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan_digest_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in item.items()
            if key not in DIGEST_EXCLUDED_ITEM_FIELDS
        }
        for item in items
    ]


def _agent_key(cluster_id: object, node_id: object) -> str:
    return f"{cluster_id}/{node_id}"


def _inventory(site: RenderedSite) -> dict[str, Any]:
    result = run_pod_script(
        site,
        {
            "mode": "list-agents",
            "cluster_ids": sorted(
                str(item["cluster_id"]) for item in site.release_config["clusters"]
            ),
        },
    )
    agents = result.get("agents")
    if not isinstance(agents, list) or not all(
        isinstance(item, dict) for item in agents
    ):
        raise BootstrapError("agent inventory from the control plane is invalid")
    evaluated_at = result.get("evaluated_at")
    if not isinstance(evaluated_at, str):
        raise BootstrapError("agent inventory from the control plane has no clock")
    return {"agents": agents, "evaluated_at": evaluated_at}


def agent_node_evidence(
    site: RenderedSite,
    cluster_id: str,
    node_id: str,
    *,
    kubernetes_nodes: dict[str, dict[str, Any]],
    provider_inventories: dict[str, frozenset[str] | str],
    read_at: dict[str, str],
) -> dict[str, Any]:
    """The departed-node evidence of one Agent, in the never-dispatched shape.

    A node Kubernetes still has is the blocker whatever its condition; a node
    it no longer has is explained by HyperPod (``absent_from_provider``) or
    stays blocked with the reason the provider could not clear it.
    """

    target = reconcile.cluster_target(site, cluster_id)
    departed = (
        None
        if kubernetes_nodes.get(node_id) is not None
        else reconcile._departed_node_evidence(
            site, cluster_id, node_id, provider_inventories, read_at
        )
    )
    evidence = never_dispatched.node_evidence(
        node_id,
        departed=departed,
        target=target,
        aws_region=site.release_config.get("aws_region"),
    )
    if departed is None:
        evidence["blockers"] = [KUBERNETES_PRESENT_REASON]
    return evidence


def _plan_item(
    site: RenderedSite,
    agent: dict[str, Any],
    *,
    evaluated_at: datetime,
    kubernetes_inventories: dict[str, dict[str, dict[str, Any]]],
    provider_inventories: dict[str, frozenset[str] | str],
    read_at: dict[str, dict[str, str]],
) -> dict[str, Any]:
    cluster_id = str(agent.get("cluster_id") or "")
    node_id = str(agent.get("node_id") or "")
    item: dict[str, Any] = {
        "agent": _agent_key(cluster_id, node_id),
        **{field: agent.get(field) for field in RECORD_FIELDS},
        "terminalization": RETIRE_DEPARTED_AGENTS,
    }
    reasons: list[str] = []
    lifecycle = str(agent.get("lifecycle_state") or "")
    if lifecycle not in RETIREABLE_LIFECYCLES:
        reasons.append(f"agent lifecycle state is {lifecycle or 'unknown'}")
    last_seen = agent.get("last_seen_at")
    try:
        seen_at = datetime.fromisoformat(str(last_seen))
    except ValueError:
        seen_at = None
    if seen_at is None or seen_at.tzinfo is None:
        reasons.append("agent last_seen_at is missing or not timezone-aware")
    elif seen_at > evaluated_at - DEPARTED_AGENT_GUARD_AGE:
        reasons.append(GUARD_REASON)
    if cluster_id not in {
        str(entry["cluster_id"]) for entry in site.release_config["clusters"]
    }:
        # A record of a cluster the site no longer manages: no kubeconfig to
        # read its nodes with, so no evidence; reported, never retired here.
        reasons.append(f"cluster {cluster_id or 'unknown'} is not in the managed site")
        item.update({"node_evidence": None, "eligible": False, "reasons": reasons})
        return item
    inventory = kubernetes_inventories.get(cluster_id)
    if inventory is None:
        read_at[cluster_id] = {"kubernetes": datetime.now(timezone.utc).isoformat()}
        inventory = reconcile.cluster_nodes(site, cluster_id)
        kubernetes_inventories[cluster_id] = inventory
    evidence = agent_node_evidence(
        site,
        cluster_id,
        node_id,
        kubernetes_nodes=inventory,
        provider_inventories=provider_inventories,
        read_at=read_at[cluster_id],
    )
    reasons.extend(f"{node_id}: {reason}" for reason in evidence["blockers"])
    item["node_evidence"] = evidence
    item["evidence_read_at"] = dict(read_at[cluster_id])
    item["eligible"] = not reasons
    item["reasons"] = reasons
    return item


def _plan(site: RenderedSite, *, node_ids: Sequence[str]) -> dict[str, Any]:
    inventory = _inventory(site)
    evaluated_at = datetime.fromisoformat(inventory["evaluated_at"])
    if evaluated_at.tzinfo is None:
        raise BootstrapError("agent inventory clock is not timezone-aware")
    agents = list(inventory["agents"])
    wanted = {str(item).strip() for item in node_ids}
    revoked = [item for item in agents if item.get("lifecycle_state") == "REVOKED"]
    candidates = [
        item
        for item in agents
        if item.get("lifecycle_state") != "REVOKED"
        and (not wanted or str(item.get("node_id")) in wanted)
    ]
    kubernetes_inventories: dict[str, dict[str, dict[str, Any]]] = {}
    provider_inventories: dict[str, frozenset[str] | str] = {}
    read_at: dict[str, dict[str, str]] = {}
    items = [
        _plan_item(
            site,
            agent,
            evaluated_at=evaluated_at,
            kubernetes_inventories=kubernetes_inventories,
            provider_inventories=provider_inventories,
            read_at=read_at,
        )
        for agent in candidates
    ]
    known = {str(item.get("node_id")) for item in candidates}
    for node_id in sorted(wanted - known):
        # Named but not in the fleet (or already REVOKED): reported, never
        # widened into a discovery.
        items.append(
            {
                "agent": node_id,
                "node_id": node_id,
                "terminalization": RETIRE_DEPARTED_AGENTS,
                "eligible": False,
                "reasons": [NO_RECORD_REASON],
            }
        )
    items.sort(key=lambda item: str(item["agent"]))
    plan = {
        "schema_version": 1,
        "mode": "retire-departed-agents-plan",
        "evaluated_at": inventory["evaluated_at"],
        "site_identity": reconcile._site_identity(site),
        "discovery": {
            "agents": len(agents),
            "revoked": len(revoked),
            "candidates": len(candidates),
            "eligible": sum(1 for item in items if item["eligible"]),
        },
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "site_identity": plan["site_identity"],
            "items": plan_digest_items(items),
        }
    )
    return plan


def _plan_drift(
    saved_items: list[dict[str, Any]], current_items: list[dict[str, Any]]
) -> list[str]:
    """Which digest fields moved between the plan and the re-plan before apply."""

    saved = {str(item["agent"]): item for item in plan_digest_items(saved_items)}
    current = {str(item["agent"]): item for item in plan_digest_items(current_items)}
    drift: list[str] = []
    for key in sorted(set(saved) - set(current)):
        drift.append(f"{key}: no longer in the plan")
    for key in sorted(set(current) - set(saved)):
        drift.append(f"{key}: newly in the plan")
    for key in sorted(set(saved) & set(current)):
        before, after = saved[key], current[key]
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field):
                drift.append(
                    f"{key}: {field} {before.get(field)!r} -> {after.get(field)!r}"
                )
    return drift


def _ineligible(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(item["agent"]): [str(reason) for reason in item.get("reasons") or []]
        for item in items
        if not item.get("eligible")
    }


def _refuse(prefix: str, ineligible: dict[str, list[str]]) -> BootstrapError:
    return BootstrapError(
        prefix
        + " | ".join(
            f"{key}: " + "; ".join(reasons)
            for key, reasons in sorted(ineligible.items())
        )
    )


def apply_entry(item: dict[str, Any]) -> dict[str, Any]:
    """The Pod's apply entry: the compare-and-set keys and the evidence."""

    return {
        "cluster_id": str(item["cluster_id"]),
        "node_id": str(item["node_id"]),
        "generation": int(item["generation"]),
        "lifecycle_state": str(item["lifecycle_state"]),
        "last_seen_at": str(item["last_seen_at"]),
        "departed_node_evidence": {
            **{
                key: item["node_evidence"].get(key)
                for key in (
                    "node_id",
                    "instance_id",
                    "provider",
                    "absent_from_kubernetes",
                    "absent_from_provider",
                    "sources",
                )
            },
            "read_at": dict(item.get("evidence_read_at") or {}),
        },
    }


def run_retire_departed_agents(
    site: RenderedSite,
    state_dir: Path,
    *,
    node_ids: Sequence[str] = (),
    reference: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Plan, and unless ``dry_run``, retire, in one invocation.

    Discovery (no ``--node``) may list Agents that are not eligible; those are
    reported under ``ineligible`` and skipped. A node the operator named that
    is not eligible refuses the whole apply before any write, with its
    reasons. Eligible Agents are re-planned immediately before the apply and
    any field that moved refuses the apply by name. Each record is written on
    its own inside the Pod; ``failed_agents`` and ``failures`` say which did
    not land, and the CLI exits 1 on any.
    """

    if any(not str(item).strip() for item in node_ids):
        raise BootstrapError("--node must not be blank")
    requested = [str(item).strip() for item in node_ids]
    normalized_reference = reconcile._validate_request(
        workflow_ids=(),
        incident_ids=(),
        max_items=None,
        reference=reference,
        dry_run=dry_run,
    )
    plan = _plan(site, node_ids=requested)
    if dry_run:
        return {**plan, "dry_run": True}
    assert normalized_reference is not None
    ineligible = _ineligible(plan["items"])
    if requested and ineligible:
        raise _refuse("retire-departed-agents refuses ineligible Agents: ", ineligible)
    eligible = [item for item in plan["items"] if item.get("eligible")]
    result: dict[str, Any]
    if not eligible:
        result = {
            "mode": "retire-departed-agents-apply",
            "retired_agents": [],
            "failed_agents": [],
            "failures": {},
        }
    else:
        # Re-planned for exactly the Agents about to be written and compared
        # field by field with the plan just made.
        current = _plan(site, node_ids=[str(item["node_id"]) for item in eligible])
        drift = _plan_drift(eligible, list(current["items"]))
        if drift:
            raise BootstrapError(
                "retire-departed-agents plan changed before apply: " + " | ".join(drift)
            )
        still_ineligible = _ineligible(current["items"])
        if still_ineligible:
            raise _refuse(
                "retire-departed-agents plan contains ineligible Agents: ",
                still_ineligible,
            )
        actor = operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
        result = run_pod_script(
            site,
            {
                "mode": RETIRE_DEPARTED_AGENTS,
                "items": [apply_entry(item) for item in current["items"]],
                "reference": normalized_reference,
                "actor": actor,
                "admin_plan_sha256": current["plan_sha256"],
                "transition_id": (
                    f"workflow-reconcile/{normalized_reference}/"
                    f"{current['plan_sha256'][:16]}"
                ),
            },
        )
        result["actor"] = actor
        result["admin_plan_sha256"] = current["plan_sha256"]
        plan = current
    result["reference"] = normalized_reference
    result["plan_sha256"] = plan["plan_sha256"]
    result["ineligible"] = ineligible
    result["dry_run"] = False
    archive = state_dir / HISTORY_PATH / str(plan["plan_sha256"])
    write_json_atomic(archive / "plan.json", plan)
    write_json_atomic(archive / "applied.json", result)
    return result


def retire_departed_agents_command(
    arguments: argparse.Namespace, *, site: RenderedSite, state_dir: Path
) -> dict[str, Any]:
    """The CLI entry: refuse selectors of the other shapes, then run."""

    conflicts = conflicting_flags(arguments)
    if conflicts:
        raise BootstrapError(
            "--retire-departed-agents takes --node, --reference and --dry-run only; "
            "not " + ", ".join(conflicts)
        )
    return run_retire_departed_agents(
        site,
        state_dir,
        node_ids=tuple(arguments.node),
        reference=arguments.reference,
        dry_run=bool(getattr(arguments, "dry_run", False)),
    )
