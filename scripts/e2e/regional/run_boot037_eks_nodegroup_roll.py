#!/usr/bin/env python3
"""GF-REGIONAL-BOOT-037: the prepare apply rolls a plain EKS GPU node group.

The destructive counterpart of BOOT-036. ``gpu-fault-admin
prepare-eks-nodegroup --apply`` (administrator operations §5.0.1a) publishes a
launch template version whose only change is the managed user data part, sets
``updateStrategy`` MINIMAL, rolls the group onto it and verifies the new nodes
over SSM. A roll replaces **every** GPU node of the group: EKS terminates each
node before it launches its replacement, every workload on the group stops,
and the replacements come back with new names and instance ids. The case
proves the whole path on a site-managed group:

* the read-only plan of the target configuration rolls the group with MINIMAL
  (a first preparation, a new managed part, another DCGM pin or an adopted
  hand-written part; a group that already runs the target is refused here);
* nothing else holds the group: every node Ready and schedulable without an
  isolating taint, no business workload and no attempt the control plane binds
  to a node, no active incident or executable workflow on the cluster, every
  node's Agent ACTIVE, and a maintenance window covering the bounded waits;
* the apply ends COMPLETED with the roll update Successful and the tool's own
  SSM verification passed;
* every departed Node is gone and as many replacement Nodes are Ready, labelled
  for the site (and ``gpu-fault.io/dcgm-hostengine=host`` in host mode);
* the read-only plan afterwards is COMPLIANT with every node check ok on
  exactly the replacement instances, the DCGM pin paired with the exporter;
* the node installer reconciler made every replacement's Agent ACTIVE and the
  departed nodes' Agents retired by themselves, no incident opened since the
  roll began, and the replacements stay Ready, schedulable and untainted for
  ten minutes;
* the result names the renames (``node_renames``) and the site profile
  arguments that name a departed node (``profile_updates``): the campaign
  inputs follow them before any later case runs on the site -- the case never
  edits them itself.

A roll cannot be undone. A failed roll is completed by rerunning the same apply
(the tool's journal resumes its update and reuses the prepared launch template
version); a departed Agent still ACTIVE is retired with ``gpu-fault-admin
workflow-reconcile --retire-departed-agents``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import eks_nodegroup_case as case  # noqa: E402
from scripts.e2e.regional import eks_nodegroup_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.admin_cli import admin_command  # noqa: E402
from scripts.e2e.regional.eks_nodegroup_case import admin_tool  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = "GF-REGIONAL-BOOT-037"
CONFIRMATION = "BOOT037_REPLACE_EVERY_GPU_NODE_OF_THE_GROUP"
CLOCK = case.Clock()
NODEGROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
DEFAULT_ROLL_TIMEOUT_MINUTES = 150
MAX_ROLL_TIMEOUT_MINUTES = 360
#: The apply beside its roll: two configuration updates (15 min each), the
#: tool's SSM verification of the new nodes (25 min) and a margin.
APPLY_OVERHEAD_SECONDS = 2 * 900 + 1500 + 600
REPLACEMENT_WAIT_SECONDS = 1800.0
AGENT_WAIT_SECONDS = 1800.0
RETIRE_WAIT_SECONDS = 1200.0
STABILITY_WAIT_SECONDS = 1200.0
STABLE_READY_SECONDS = 600.0
POLL_SECONDS = 30.0
POST_ROLL_SECONDS = (
    REPLACEMENT_WAIT_SECONDS
    + case.PLAN_TIMEOUT_SECONDS
    + AGENT_WAIT_SECONDS
    + RETIRE_WAIT_SECONDS
    + STABILITY_WAIT_SECONDS
)
LIMITATIONS = [
    "A roll replaces instances: the departed nodes' identities are gone for good "
    "and the replacement names cannot be predicted; the renames are paired in name "
    "order by convention.",
    "MINIMAL takes maxUnavailable nodes out at a time; on Spot the replacement may "
    "not launch for want of capacity, which ends the apply ROLL_FAILED or "
    "ROLL_IN_PROGRESS and is completed by rerunning the same apply.",
    "The case judges the replacements the roll produced; it does not prove a later "
    "AMI release or an EKS-initiated replacement prepares nodes the same way.",
]


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    state_dir: Path
    nodegroup: str
    dcgm_hostengine: str | None
    dcgm_version: str | None
    adopt_parts: tuple[str, ...]
    roll_timeout_minutes: int
    reference: str
    #: The approved window end as typed (or from the profile); "" when unset.
    window_end: str
    predecessor_id: str | None
    predecessor_path: Path | None

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_STATE_DIR": str(self.state_dir),
        }

    def target_flags(self, *, adopt: bool = True) -> list[str]:
        """The target configuration; the adoption is consumed by the roll."""

        flags = ["--nodegroup", self.nodegroup]
        if self.dcgm_hostengine:
            flags += ["--dcgm-hostengine", self.dcgm_hostengine]
        if self.dcgm_version:
            flags += ["--dcgm-version", self.dcgm_version]
        if adopt:
            for part in self.adopt_parts:
                flags += ["--adopt-user-data-part", part]
        return flags

    def apply_flags(self) -> list[str]:
        return [
            *self.target_flags(),
            "--roll-timeout-minutes",
            str(self.roll_timeout_minutes),
            "--apply",
            "--confirm",
            case.TOOL_CONFIRMATION,
            "--reference",
            self.reference,
        ]

    def required_window_seconds(self) -> float:
        return (
            self.roll_timeout_minutes * 60.0
            + APPLY_OVERHEAD_SECONDS
            + POST_ROLL_SECONDS
        )


def default_reference(run_dir: Path, attempt: int) -> str:
    token = hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]
    return f"{CASE_ID}:a{attempt}:{token}"


def configure(arguments: argparse.Namespace) -> Settings:
    if not str(arguments.state_dir or "").strip():
        raise RegionalFixtureError(
            "--state-dir (the site's state directory) is required"
        )
    if len(arguments.nodegroup or ()) != 1 or not NODEGROUP_NAME.fullmatch(
        arguments.nodegroup[0]
    ):
        raise RegionalFixtureError("--nodegroup names exactly one node group to roll")
    if arguments.dcgm_version and arguments.dcgm_hostengine != "host":
        raise RegionalFixtureError(
            "--dcgm-version is used only with --dcgm-hostengine host"
        )
    minutes = int(arguments.roll_timeout_minutes)
    if not 1 <= minutes <= MAX_ROLL_TIMEOUT_MINUTES:
        raise RegionalFixtureError(
            f"--roll-timeout-minutes must be 1..{MAX_ROLL_TIMEOUT_MINUTES}"
        )
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    reference = arguments.reference or default_reference(
        run_dir, int(arguments.attempt)
    )
    if not REFERENCE.fullmatch(reference):
        raise RegionalFixtureError(
            f"--reference {reference!r} is not a change reference"
        )
    previous, path = predecessor_path(run_dir, CASE_ID, arguments.predecessor_evidence)
    return Settings(
        regional=settings_from_arguments(arguments),
        state_dir=Path(arguments.state_dir).expanduser().resolve(),
        nodegroup=arguments.nodegroup[0],
        dcgm_hostengine=arguments.dcgm_hostengine,
        dcgm_version=arguments.dcgm_version,
        adopt_parts=tuple(dict.fromkeys(arguments.adopt_user_data_part or ())),
        roll_timeout_minutes=minutes,
        reference=reference,
        window_end=str(arguments.maintenance_window_end or "").strip(),
        predecessor_id=previous,
        predecessor_path=path,
    )


# --- preflight ------------------------------------------------------------------


def expected_mode(settings: Settings, target: case.SiteTarget) -> str:
    return settings.dcgm_hostengine or target.dcgm_mode


def plan_summary(run: case.ToolRun, nodegroup: str) -> dict[str, Any]:
    """The facts of the tool's plan the case approves, without the user data."""

    group = verdicts.groups_by_name(run.document).get(nodegroup) or {}
    dcgm = group.get("dcgm_hostengine") or {}
    inventory = group.get("current_nodes") or {}
    return {
        "status": (run.document or {}).get("status"),
        "verdict": group.get("verdict"),
        "steps": [
            {key: step.get(key) for key in ("step", "summary")}
            for step in group.get("steps") or ()
        ],
        "launch_template": group.get("launch_template"),
        "dcgm": {key: dcgm.get(key) for key in ("mode", "pin", "exporter_pairing")},
        "update_strategy": group.get("update_strategy"),
        "roll": group.get("roll"),
        "inventory": {
            "outcome": inventory.get("outcome"),
            "instances": sorted(
                str(node.get("instance_id")) for node in inventory.get("nodes") or ()
            ),
            "failed": sorted(
                str(node.get("instance_id"))
                for node in inventory.get("nodes") or ()
                if node.get("passed") is not True
            ),
        },
    }


def node_errors(
    nodegroup: str,
    nodes: list[dict[str, Any]],
    instances: list[str],
    *,
    desired: object,
    inventoried: list[str],
) -> list[str]:
    errors = []
    if isinstance(desired, int) and len(nodes) != desired:
        errors.append(f"{nodegroup} has {len(nodes)} Nodes, desired {desired}")
    for node in nodes:
        if node["ready"] != "True" or node["unschedulable"] or node["blocking_taints"]:
            errors.append(
                f"{node['name']} is held (Ready {node['ready']}, cordoned "
                f"{node['unschedulable']}, taints {node['blocking_taints']})"
            )
    if sorted(node["instance_id"] for node in nodes) != sorted(instances):
        errors.append(
            f"{nodegroup}: Nodes {sorted(node['instance_id'] for node in nodes)}, "
            f"running instances {sorted(instances)}"
        )
    if inventoried != sorted(instances):
        errors.append(
            f"{nodegroup}: the plan inventoried {inventoried}, not {instances}"
        )
    return errors


def workload_errors(
    fixture: Any, nodes: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """No business Pod on a node and no attempt the control plane binds to it."""

    seen: dict[str, Any] = {}
    errors = []
    for node in nodes:
        name = str(node["name"])
        business = fixture.business_workloads(name)
        view = fixture.node_workload_view(name)
        seen[name] = {"business_workloads": business, "control_plane_view": view}
        if business:
            errors.append(f"{name} runs business workloads {business}")
        if view.get("attempt_ids"):
            errors.append(f"{name} is bound to attempts {view['attempt_ids']}")
    return seen, errors


def window_errors(settings: Settings) -> list[str]:
    if not settings.window_end:
        return []
    end = datetime.fromisoformat(settings.window_end.replace("Z", "+00:00"))
    remaining = (end - CLOCK.now()).total_seconds()
    required = settings.required_window_seconds()
    if remaining >= required:
        return []
    return [
        f"the maintenance window leaves {remaining:.0f} s, the roll and its waits need "
        f"up to {required:.0f} s (--roll-timeout-minutes {settings.roll_timeout_minutes})"
    ]


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    """The plan of the target configuration and everything that must not hold
    the group; reads only, the tool's read-only plan included."""

    fixture = RegionalLiveFixture(settings.regional)
    errors: list[str] = []
    result: dict[str, Any] = {"errors": errors, "identity": {}, "predecessor": {}}
    try:
        result["identity"] = dict(fixture.evidence_identity())
        result["predecessor"] = case.predecessor_gate(
            settings.predecessor_id, settings.predecessor_path, result["identity"]
        )
        if result["predecessor"].get("valid") is not True:
            errors.append(f"formal predecessor {settings.predecessor_id} is not PASS")
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"release identity unavailable: {type(exc).__name__}: {exc}")
    try:
        target = case.site_target(settings.state_dir, settings.regional.cluster_id)
        result["target"] = target.describe()
        tool = admin_tool(fixture, settings.state_dir)
        result["cli"], problems = case.cli_errors(tool)
        errors += problems
        document = case.nodegroup_document(fixture, target, settings.nodegroup)
        desired = (document.get("scalingConfig") or {}).get("desiredSize")
        plan = tool(
            target.tool_arguments(*settings.target_flags()),
            timeout=case.PLAN_TIMEOUT_SECONDS,
        )
        case.record_run(case_dir, "tool-plan-before", plan)
        summary = plan_summary(plan, settings.nodegroup)
        result["tool_plan"] = summary
        errors += verdicts.roll_plan_errors(
            plan.document,
            target=target,
            nodegroup=settings.nodegroup,
            mode=expected_mode(settings, target),
            desired=desired,
        )
        instances = case.running_instances(fixture, target, settings.nodegroup)
        nodes = case.group_nodes(fixture, settings.nodegroup)
        result["nodegroup"] = {
            "name": settings.nodegroup,
            "status": document.get("status"),
            "capacity_type": document.get("capacityType"),
            "desired": desired,
            "update_config": document.get("updateConfig"),
            "instances": instances,
            "nodes": nodes,
        }
        errors += node_errors(
            settings.nodegroup,
            nodes,
            instances,
            desired=desired,
            inventoried=summary["inventory"]["instances"],
        )
        result["workloads"], problems = workload_errors(fixture, nodes)
        errors += problems
        store = case.store_state(fixture, settings.regional.cluster_id)
        result["store"] = store
        names = [str(node["name"]) for node in nodes]
        errors += verdicts.incident_errors(store, nodes=names)
        errors += verdicts.agent_errors(store, active=names)
        errors += window_errors(settings)
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"read-only preflight failed: {type(exc).__name__}: {exc}")
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    identity = preflight.get("identity") or {}
    summary = preflight.get("tool_plan") or {}
    group = preflight.get("nodegroup") or {}
    return {
        "release_id": identity.get("release_id"),
        "cluster_id": identity.get("cluster_id"),
        "site_id": (preflight.get("target") or {}).get("site_id"),
        "nodegroup": group.get("name"),
        "launch_template": summary.get("launch_template"),
        "steps": [step.get("step") for step in summary.get("steps") or ()],
        "dcgm": summary.get("dcgm"),
        "instances": group.get("instances"),
        "nodes": [node.get("name") for node in group.get("nodes") or ()],
    }


# --- the roll -------------------------------------------------------------------


def wait_replacement(
    fixture: Any, nodegroup: str, departed: list[dict[str, Any]], desired: object
) -> tuple[list[dict[str, Any]], list[str], float]:
    """Until no departed Node is left and every replacement is Ready.

    EKS reports the update done while the last departed instance still shuts
    down: its Node lingers NotReady until the node lifecycle controller deletes
    it, so the replacement set is compared only once that one is gone too.
    """

    def probe() -> tuple[bool, Any]:
        nodes = case.group_nodes(fixture, nodegroup)
        errors = verdicts.replacement_errors(departed, nodes, desired=desired)
        return not errors, (nodes, errors)

    _reached, (nodes, errors), waited = case.wait_until(
        probe,
        timeout_seconds=REPLACEMENT_WAIT_SECONDS,
        interval_seconds=POLL_SECONDS,
        clock=CLOCK,
    )
    return nodes, errors, waited


def wait_agents(
    fixture: Any, cluster_id: str, *, active: list[str], departed: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """The replacements' Agents ACTIVE (node installer reconciler), then the
    departed nodes' Agents retired (departed-node cleanup)."""

    def stage(departing: list[str]) -> tuple[bool, Any]:
        store = case.store_state(fixture, cluster_id)
        errors = verdicts.agent_errors(store, active=active, departed=departing)
        return not errors, (store, errors)

    reached, (store, errors), waited = case.wait_until(
        lambda: stage([]),
        timeout_seconds=AGENT_WAIT_SECONDS,
        interval_seconds=POLL_SECONDS,
        clock=CLOCK,
    )
    timing = {"agents_active_after_seconds": waited}
    if not reached:
        return errors, {**timing, "store": store}
    _done, (store, errors), retired = case.wait_until(
        lambda: stage(departed),
        timeout_seconds=RETIRE_WAIT_SECONDS,
        interval_seconds=POLL_SECONDS,
        clock=CLOCK,
    )
    return errors, {**timing, "departed_retired_after_seconds": retired, "store": store}


def wait_stable(
    fixture: Any, nodegroup: str, departed: list[dict[str, Any]], desired: object
) -> tuple[list[dict[str, Any]], list[str]]:
    def probe() -> tuple[bool, Any]:
        nodes = case.group_nodes(fixture, nodegroup)
        errors = verdicts.stability_errors(
            nodes, now=CLOCK.now(), ready_seconds=STABLE_READY_SECONDS
        ) + verdicts.replacement_errors(departed, nodes, desired=desired)
        return not errors, (nodes, errors)

    _reached, (nodes, errors), _waited = case.wait_until(
        probe,
        timeout_seconds=STABILITY_WAIT_SECONDS,
        interval_seconds=POLL_SECONDS,
        clock=CLOCK,
    )
    return nodes, errors


def replan_errors(
    settings: Settings,
    case_dir: Path,
    fixture: Any,
    target: case.SiteTarget,
    *,
    nodes: list[dict[str, Any]],
    desired: object,
    pin: str | None,
) -> list[str]:
    """The read-only plan of the same target after the roll: COMPLIANT."""

    run = admin_tool(fixture, settings.state_dir)(
        target.tool_arguments(*settings.target_flags(adopt=False)),
        timeout=case.PLAN_TIMEOUT_SECONDS,
    )
    case.record_run(case_dir, "tool-plan-after", run)
    facts = {
        settings.nodegroup: verdicts.GroupFacts(
            instances=tuple(
                case.running_instances(fixture, target, settings.nodegroup)
            ),
            node_instances=tuple(str(node["instance_id"]) for node in nodes),
            desired=desired if isinstance(desired, int) else None,
        )
    }
    return verdicts.compliant_plan_errors(
        run.document,
        target=target,
        facts=facts,
        mode=expected_mode(settings, target),
        pin=pin,
    )


def apply_command(settings: Settings, target: case.SiteTarget) -> str:
    """The apply as an operator reruns it from the deploy host."""

    return shlex.join(
        [
            *admin_command(settings.state_dir),
            case.TOOL_VERB,
            *target.tool_arguments(*settings.apply_flags()),
        ]
    )


def exits(
    settings: Settings, target: case.SiteTarget, stages: dict[str, list[str]]
) -> list[str]:
    """What the operator runs to finish what a failed stage left."""

    apply = apply_command(settings, target)
    result = []
    if stages.get("apply") or stages.get("replacement"):
        result.append(
            "a roll short of capacity finishes on a rerun of the same apply (the "
            f"journal resumes it): {apply}"
        )
    if any("departed node's Agent" in item for item in stages.get("agents", ())):
        result.append(
            "retire each departed Agent: gpu-fault-admin workflow-reconcile "
            f"--state-dir {settings.state_dir} --retire-departed-agents --node <node> "
            f"--reference {settings.reference}"
        )
    return result


def roll(
    settings: Settings, case_dir: Path, result: dict[str, Any], *, deadline: datetime
) -> None:
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    case.verify_plan_identity(case_dir, CASE_ID, plan_identity(preflight))
    result.update(preflight["identity"])
    result["predecessor"] = preflight["predecessor"]
    remaining = (deadline - CLOCK.now()).total_seconds()
    if remaining < settings.required_window_seconds():
        raise RegionalFixtureError(
            f"the maintenance window leaves {remaining:.0f} s, the roll needs up to "
            f"{settings.required_window_seconds():.0f} s"
        )
    fixture = RegionalLiveFixture(settings.regional)
    target = case.site_target(settings.state_dir, settings.regional.cluster_id)
    group = preflight["nodegroup"]
    departed = [
        {key: node[key] for key in ("name", "instance_id")} for node in group["nodes"]
    ]
    desired = group["desired"]
    stages: dict[str, list[str]] = {}
    result.update(nodegroup=settings.nodegroup, departed_nodes=departed, stages=stages)
    started = CLOCK.now()
    result["roll_started_at"] = started.isoformat()
    result["mutation_performed"] = True
    applied = admin_tool(fixture, settings.state_dir)(
        target.tool_arguments(*settings.apply_flags()),
        timeout=settings.roll_timeout_minutes * 60.0 + APPLY_OVERHEAD_SECONDS,
    )
    result["apply"] = case.record_run(case_dir, "tool-apply", applied)
    result["roll_ended_at"] = CLOCK.now().isoformat()
    stages["apply"] = verdicts.apply_errors(applied, nodegroup=settings.nodegroup)
    nodes: list[dict[str, Any]] = []
    if not stages["apply"]:
        nodes, stages["replacement"], waited = wait_replacement(
            fixture, settings.nodegroup, departed, desired
        )
        result["replacement_wait_seconds"] = waited
    if stages["apply"] or stages.get("replacement"):
        result["state_after_roll"] = {
            "nodegroup": case.nodegroup_document(fixture, target, settings.nodegroup),
            "nodes": case.group_nodes(fixture, settings.nodegroup),
        }
    else:
        stages["replan"] = replan_errors(
            settings,
            case_dir,
            fixture,
            target,
            nodes=nodes,
            desired=desired,
            pin=settings.dcgm_version or preflight["tool_plan"]["dcgm"].get("pin"),
        )
        stages["labels"] = verdicts.label_errors(
            nodes, site_id=target.site_id, mode=expected_mode(settings, target)
        )
        stages["agents"], result["agents"] = wait_agents(
            fixture,
            settings.regional.cluster_id,
            active=[str(node["name"]) for node in nodes],
            departed=[str(node["name"]) for node in departed],
        )
        nodes, stages["stability"] = wait_stable(
            fixture, settings.nodegroup, departed, desired
        )
    store = case.store_state(fixture, settings.regional.cluster_id, since=started)
    write_json_atomic(case_dir / "store-after.json", store)
    stages["incidents"] = verdicts.incident_errors(
        store,
        since=started,
        nodes=[str(node["name"]) for node in [*departed, *nodes]],
    )
    renames = verdicts.node_renames(departed, nodes)
    result["replacement_nodes"] = [
        {key: node[key] for key in ("name", "instance_id")} for node in nodes
    ]
    result["node_renames"] = renames
    result["profile_updates"] = verdicts.profile_updates(
        case.profile_arguments(), renames
    )
    identity = dict(fixture.evidence_identity())
    stages["identity"] = (
        []
        if identity == preflight["identity"]
        else [f"the release changed during the case: {identity}"]
    )
    result["errors"] = [
        f"{stage}: {item}" for stage, items in stages.items() for item in items
    ]
    result["exits"] = exits(settings, target, stages)
    result["verdict"] = verdicts.case_verdict(stages)


def execute_case(
    settings: Settings, run_dir: Path, attempt: int, deadline: datetime
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    evidence = case_dir / f"{CASE_ID}.json"
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "started_at": CLOCK.now().isoformat(),
        "maintenance_window_end": deadline.isoformat(),
        "reference": settings.reference,
        "mutation_performed": False,
        "limitations": LIMITATIONS,
    }
    write_json_atomic(evidence, {**result, "stage": "preflight"})
    interrupted: BaseException | None = None
    try:
        roll(settings, case_dir, result, deadline=deadline)
    except BaseException as exc:
        result["verdict"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            interrupted = exc
    result["executed_at"] = CLOCK.now().isoformat()
    write_json_atomic(evidence, result)
    print(json.dumps(result, sort_keys=True))
    if interrupted is not None:
        raise interrupted
    return 0 if result["verdict"] == "PASS" else 1


# --- plan -----------------------------------------------------------------------


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    target = preflight.get("target") or {}
    apply = shlex.join(
        [
            *admin_command(settings.state_dir),
            case.TOOL_VERB,
            "--cpu-cluster-arn",
            str(target.get("cpu_cluster_arn")),
            "--gpu-cluster-arn",
            str(target.get("gpu_cluster_arn")),
            "--state-dir",
            str(settings.state_dir),
            *settings.apply_flags(),
        ]
    )
    # The same words apply_command() renders from the site record once the
    # target is read; the plan shows them before any run of the tool.
    return {
        "risk": "destructive",
        "predecessor": preflight.get("predecessor"),
        "target": target,
        "nodegroup": settings.nodegroup,
        "mutation": (
            "gpu-fault-admin prepare-eks-nodegroup --apply: create a launch template "
            "version changing only UserData, set updateStrategy MINIMAL where it is "
            "not, and roll the node group onto it -- EKS terminates every GPU node "
            "of the group, maxUnavailable at a time, before launching its "
            "replacement; every workload on the group stops and the replacements "
            "get new names and instance ids. The case edits no campaign input."
        ),
        "apply_command": apply,
        "nodes_to_replace": [
            {key: node.get(key) for key in ("name", "instance_id")}
            for node in (preflight.get("nodegroup") or {}).get("nodes") or ()
        ],
        "required_window_seconds": settings.required_window_seconds(),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            "formal predecessor evidence (BOOT-036) is not PASS",
            "the plan of the target configuration does not roll the group with "
            "MINIMAL (already compliant, refused, an earlier update in progress)",
            "a node is held, runs a business workload or is bound to an attempt",
            "an active incident or executable workflow on the cluster, or a "
            "node's Agent not ACTIVE",
            "the maintenance window cannot cover the roll and its bounded waits",
            "the release, the plan, the instances or the Nodes drifted between "
            "plan and execute",
        ],
        "rollback": {
            "a_roll_cannot_be_undone": True,
            "a_failed_roll_completes_on_a_rerun_of_the_same_apply": True,
            "departed_agents_retire_with_workflow_reconcile": True,
            "campaign_inputs_follow_node_renames_and_profile_updates": True,
        },
        "preflight": preflight,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded BOOT-037 acceptance: prepare-eks-nodegroup --apply rolls "
            "one plain EKS GPU node group with updateStrategy MINIMAL, the "
            "replacement nodes come back prepared and adopted by the fleet."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    for name in (
        "cpu-kubeconfig",
        "gpu-kubeconfig",
        "gpu-context",
        "cluster-id",
        "region",
        "predecessor-evidence",
        "state-dir",
        "reference",
    ):
        value.add_argument(f"--{name}", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument(
        "--nodegroup",
        action="append",
        default=[],
        help="the one node group to roll (required)",
    )
    value.add_argument("--dcgm-hostengine", choices=("embedded", "host"), default=None)
    value.add_argument("--dcgm-version", default=None)
    value.add_argument("--adopt-user-data-part", action="append", default=[])
    value.add_argument(
        "--roll-timeout-minutes", type=int, default=DEFAULT_ROLL_TIMEOUT_MINUTES
    )
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
