#!/usr/bin/env python3
"""GF-REGIONAL-BOOT-036: a prepared plain EKS GPU node group plans nothing.

``gpu-fault-admin prepare-eks-nodegroup`` (administrator operations §5.0.1a)
prepares a plain EKS (``eks-managed``) GPU managed node group: the managed user
data part (python3.12, strace, in host mode the paired DCGM pin and the
nvidia-dcgm restart drop-in), node auto repair off, the ownership label and
``updateStrategy`` MINIMAL, then a roll and an SSM check of the new nodes. This
case proves on the live site, read-only, that the site's groups stay prepared:

* the read-only plan of every group the site labels is COMPLIANT -- no step --
  and its SSM inventory passes every node check on exactly the group's running
  instances, with the DCGM pin paired with the product's exporter and no
  finding (``eks_nodegroup_verdicts.compliant_plan_errors``) -- except that a
  single-node group's MINIMAL roll takes its only GPU node out whatever
  ``maxUnavailable`` says: that finding joins the result's ``limitations``;
* the refusals a plan must show stop before anything is read or changed: the
  CPU control plane, a HyperPod cluster and a HyperPod-orchestrated EKS cluster
  as the GPU cluster, an unknown node group, an ``--adopt-user-data-part``
  naming no part, and a node group of the cluster the verb cannot prepare;
* every group's configuration, launch template head, instances and Nodes and
  the tool's journals are exactly as they were (``snapshot_drift``), and no
  incident opened while the inventory ran ``nvidia-smi -q`` and
  ``dcgmi discovery -l`` on the nodes.

Nothing is applied: every run of the tool is its default read-only plan, which
the CLI executes through a probe runner that refuses any mutating call. Plan
mode (``--plan``) reads the site, the provider, Kubernetes and the store only;
``--execute`` runs the tool. BOOT-037 is the destructive counterpart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

CASE_ID = "GF-REGIONAL-BOOT-036"
CONFIRMATION = "BOOT036_READ_ONLY_EKS_NODEGROUP_PLAN"
CLOCK = case.Clock()
NODEGROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
LIMITATIONS = [
    "The node checks are the tool's own SSM inventory judged per DCGM mode; the "
    "case re-reads the instances and Nodes the inventory must cover, not what "
    "each check reads on the node.",
    "A group whose nodes the product holds (not Ready, cordoned, an isolating "
    "taint, an open incident or workflow) is never inventoried: the preflight "
    "refuses instead.",
    "The unsupported-group refusal runs only where the cluster has such a group, "
    "the HyperPod-orchestrated EKS refusal only on a site with a HyperPod member.",
    "A single-node group's whole-group roll is a limitation, not a failure: no "
    "maxUnavailable keeps its only GPU node up, so each such group gets an entry "
    "of its own in this list; for two or more nodes the finding fails the case.",
]


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    state_dir: Path
    nodegroups: tuple[str, ...]
    dcgm_hostengine: str | None
    dcgm_version: str | None
    predecessor_id: str | None
    predecessor_path: Path | None
    #: Names this run's probe targets; stable across plan and execute.
    run_token: str

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_STATE_DIR": str(self.state_dir),
        }

    def mode_flags(self) -> list[str]:
        flags: list[str] = []
        if self.dcgm_hostengine:
            flags += ["--dcgm-hostengine", self.dcgm_hostengine]
        if self.dcgm_version:
            flags += ["--dcgm-version", self.dcgm_version]
        return flags


def configure(arguments: argparse.Namespace) -> Settings:
    if not str(arguments.state_dir or "").strip():
        raise RegionalFixtureError(
            "--state-dir (the site's state directory) is required"
        )
    if arguments.dcgm_version and arguments.dcgm_hostengine != "host":
        raise RegionalFixtureError(
            "--dcgm-version is used only with --dcgm-hostengine host"
        )
    names = tuple(dict.fromkeys(arguments.nodegroup or ()))
    invalid = [name for name in names if not NODEGROUP_NAME.fullmatch(name)]
    if invalid:
        raise RegionalFixtureError(f"invalid --nodegroup: {', '.join(invalid)}")
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    previous, path = predecessor_path(run_dir, CASE_ID, arguments.predecessor_evidence)
    return Settings(
        regional=settings_from_arguments(arguments),
        state_dir=Path(arguments.state_dir).expanduser().resolve(),
        nodegroups=names,
        dcgm_hostengine=arguments.dcgm_hostengine,
        dcgm_version=arguments.dcgm_version,
        predecessor_id=previous,
        predecessor_path=path,
        run_token=hashlib.sha256(str(run_dir).encode()).hexdigest()[:8],
    )


# --- preflight ------------------------------------------------------------------


def selected_nodegroups(
    settings: Settings,
    target: case.SiteTarget,
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """The named groups, else every group labelled for the site; with errors."""

    if settings.nodegroups:
        missing = sorted(set(settings.nodegroups) - set(documents))
        return sorted(set(settings.nodegroups) & set(documents)), (
            [f"{target.eks_name} has no node group {', '.join(missing)}"]
            if missing
            else []
        )
    labelled = sorted(
        name
        for name, document in documents.items()
        if (document.get("labels") or {}).get(case.OWNERSHIP_LABEL) == target.site_id
    )
    return labelled, (
        []
        if labelled
        else [
            f"no node group of {target.eks_name} carries "
            f"{case.OWNERSHIP_LABEL}={target.site_id}; name one with --nodegroup"
        ]
    )


def group_facts(
    fixture: Any, target: case.SiteTarget, name: str, document: dict[str, Any]
) -> dict[str, Any]:
    template = document.get("launchTemplate") or {}
    return {
        "status": document.get("status"),
        "launch_template": {
            "id": template.get("id"),
            "version": template.get("version"),
        },
        "desired": (document.get("scalingConfig") or {}).get("desiredSize"),
        "instances": case.running_instances(fixture, target, name),
        "nodes": case.group_nodes(fixture, name),
    }


def hold_errors(name: str, facts: dict[str, Any]) -> list[str]:
    """Why the inventory must not read the group's nodes now: it runs
    ``nvidia-smi -q`` and ``dcgmi discovery`` on every one of them, and a node
    the product holds (quiesced, cordoned, rebooting) is not read."""

    errors = []
    if facts["status"] != "ACTIVE":
        errors.append(f"{name} is {facts['status']}, not ACTIVE")
    if not facts["instances"]:
        errors.append(f"{name} runs no instance: there is nothing to inventory")
    for node in facts["nodes"]:
        if node["ready"] != "True" or node["unschedulable"] or node["blocking_taints"]:
            errors.append(
                f"{name}: {node['name']} is held (Ready {node['ready']}, cordoned "
                f"{node['unschedulable']}, taints {node['blocking_taints']})"
            )
    return errors


def refusal_targets(
    fixture: Any,
    settings: Settings,
    target: case.SiteTarget,
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The probe targets each refusal points the tool at."""

    absent = f"gf-acceptance-absent-{settings.run_token}"
    member = target.hyperpod_members[0] if target.hyperpod_members else None
    hyperpod = member.hyperpod_name if member else target.cpu_hyperpod_name
    unsupported = case.unsupported_nodegroup(documents)
    return {
        "absent_nodegroup": absent if absent not in documents else None,
        "cpu_cluster_arn": target.cpu_cluster_arn,
        "hyperpod_member": (
            {"hyperpod_name": member.hyperpod_name, "eks_arn": member.eks_arn}
            if member
            else None
        ),
        "hyperpod_cluster_arn": (
            case.hyperpod_cluster_arn(fixture, target.region, hyperpod)
            if hyperpod
            else None
        ),
        "absent_user_data_part": f"gf-acceptance-absent-part-{settings.run_token}.sh",
        "unsupported_nodegroup": (
            {"name": unsupported[0], "text": unsupported[1]} if unsupported else None
        ),
    }


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    """Identity, predecessor, target, the groups and their nodes, the probe
    targets; reads only, and no run of the tool but ``--help``."""

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
        result["cli"], problems = case.cli_errors(
            admin_tool(fixture, settings.state_dir)
        )
        errors += problems
        documents = case.cluster_nodegroups(fixture, target)
        names, problems = selected_nodegroups(settings, target, documents)
        errors += problems
        groups = {
            name: group_facts(fixture, target, name, documents[name]) for name in names
        }
        result["nodegroups"] = groups
        for name, facts in groups.items():
            errors += hold_errors(name, facts)
        result["refusal_targets"] = refusal_targets(
            fixture, settings, target, documents
        )
        store = case.store_state(fixture, settings.regional.cluster_id)
        result["store"] = store
        errors += verdicts.incident_errors(
            store,
            nodes=[
                node["name"] for facts in groups.values() for node in facts["nodes"]
            ],
            cluster_wide=False,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"read-only preflight failed: {type(exc).__name__}: {exc}")
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    identity = preflight.get("identity") or {}
    return {
        "release_id": identity.get("release_id"),
        "cluster_id": identity.get("cluster_id"),
        "site_id": (preflight.get("target") or {}).get("site_id"),
        "nodegroups": {
            name: {
                "launch_template": facts["launch_template"],
                "instances": facts["instances"],
                "nodes": [node["name"] for node in facts["nodes"]],
            }
            for name, facts in sorted((preflight.get("nodegroups") or {}).items())
        },
        "refusal_targets": preflight.get("refusal_targets"),
    }


# --- the tool runs --------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    refusal: verdicts.Refusal
    arguments: tuple[str, ...]


def main_arguments(
    settings: Settings, target: case.SiteTarget, groups: list[str]
) -> list[str]:
    flags = [word for name in groups for word in ("--nodegroup", name)]
    return target.tool_arguments(*flags, *settings.mode_flags())


def managed_part(document: dict[str, Any] | None, group: str) -> bool:
    entry = verdicts.groups_by_name(document).get(group) or {}
    parts = (entry.get("user_data") or {}).get("parts") or ()
    return any(part.get("managed") is True for part in parts)


def adoption_probe(
    settings: Settings,
    target: case.SiteTarget,
    part: str,
    group: str,
    main_document: dict[str, Any] | None,
) -> Probe:
    """``--adopt-user-data-part`` naming no part: once the managed part is in
    place a no-op note (the plan is unchanged), before it a refusal."""

    arguments = target.tool_arguments(
        "--nodegroup", group, *settings.mode_flags(), "--adopt-user-data-part", part
    )
    if main_document is not None and managed_part(main_document, group):
        refusal = verdicts.Refusal(
            "adopt-absent-part",
            f"--adopt-user-data-part {part}: nothing to adopt",
            document=main_document.get("status"),
            exit_code=0,
        )
    else:
        refusal = verdicts.Refusal(
            "adopt-absent-part",
            f"--adopt-user-data-part {part} names no part of the user data",
            document=verdicts.REFUSED,
            forbidden_calls=verdicts.NODE_READS,
        )
    return Probe(refusal, tuple(arguments))


def refusal_probes(
    settings: Settings,
    target: case.SiteTarget,
    targets: dict[str, Any],
    groups: list[str],
    main_document: dict[str, Any] | None,
) -> list[Probe]:
    """The refusal runs, each with what it must show."""

    cluster_level = (*verdicts.NODEGROUP_READS, *verdicts.NODE_READS)
    probes = [
        Probe(
            verdicts.Refusal(
                "cpu-control-plane",
                "is the CPU control plane, not a GPU cluster",
                forbidden_calls=cluster_level,
            ),
            tuple(target.tool_arguments(gpu_cluster_arn=target.cpu_cluster_arn)),
        )
    ]
    if targets.get("hyperpod_cluster_arn"):
        probes.append(
            Probe(
                verdicts.Refusal(
                    "hyperpod-cluster",
                    "is a HyperPod cluster; prepare-eks-nodegroup never touches",
                    forbidden_calls=cluster_level,
                ),
                tuple(
                    target.tool_arguments(
                        gpu_cluster_arn=str(targets["hyperpod_cluster_arn"])
                    )
                ),
            )
        )
    member = targets.get("hyperpod_member")
    if member:
        probes.append(
            Probe(
                verdicts.Refusal(
                    "hyperpod-orchestrated-eks",
                    f"is orchestrated by HyperPod cluster {member['hyperpod_name']}",
                    forbidden_calls=cluster_level,
                ),
                tuple(target.tool_arguments(gpu_cluster_arn=str(member["eks_arn"]))),
            )
        )
    absent = targets.get("absent_nodegroup")
    if absent:
        probes.append(
            Probe(
                verdicts.Refusal(
                    "unknown-nodegroup",
                    f"has no managed node group {absent}",
                    forbidden_calls=verdicts.NODE_READS,
                ),
                tuple(target.tool_arguments("--nodegroup", str(absent))),
            )
        )
    probes.append(
        adoption_probe(
            settings,
            target,
            str(targets["absent_user_data_part"]),
            groups[0],
            main_document,
        )
    )
    unsupported = targets.get("unsupported_nodegroup")
    if unsupported:
        probes.append(
            Probe(
                verdicts.Refusal(
                    "unsupported-nodegroup",
                    str(unsupported["text"]),
                    document=verdicts.REFUSED,
                    forbidden_calls=verdicts.NODE_READS,
                ),
                tuple(target.tool_arguments("--nodegroup", str(unsupported["name"]))),
            )
        )
    return probes


def group_inputs(
    snapshot: dict[str, Any], groups: list[str]
) -> dict[str, verdicts.GroupFacts]:
    facts = {}
    for name in groups:
        entry = snapshot["nodegroups"][name]
        facts[name] = verdicts.GroupFacts(
            instances=tuple(entry["instances"]),
            node_instances=tuple(node["instance_id"] for node in entry["nodes"]),
            desired=(entry.get("scaling") or {}).get("desiredSize"),
        )
    return facts


def audit(
    settings: Settings,
    case_dir: Path,
    result: dict[str, Any],
    *,
    started: datetime,
    deadline: datetime,
) -> None:
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    case.verify_plan_identity(case_dir, CASE_ID, plan_identity(preflight))
    result.update(preflight["identity"])
    result["predecessor"] = preflight["predecessor"]
    if CLOCK.now() >= deadline:
        raise RegionalFixtureError("approved maintenance window has ended")
    fixture = RegionalLiveFixture(settings.regional)
    target = case.site_target(settings.state_dir, settings.regional.cluster_id)
    tool = admin_tool(fixture, settings.state_dir)
    groups = sorted(preflight["nodegroups"])
    targets = preflight["refusal_targets"]
    unsupported = targets.get("unsupported_nodegroup")
    probed = [str(unsupported["name"])] if unsupported else []
    before = case.snapshot(fixture, target, groups, configuration_only=probed)
    write_json_atomic(case_dir / "snapshot-before.json", before)
    main = tool(
        main_arguments(settings, target, groups), timeout=case.PLAN_TIMEOUT_SECONDS
    )
    result["main_plan"] = case.record_run(case_dir, "tool-plan", main)
    result["limitations"] = [
        *LIMITATIONS,
        *verdicts.strategy_limitations(main.document),
    ]
    stages: dict[str, list[str]] = {
        "main_plan": verdicts.compliant_plan_errors(
            main.document,
            target=target,
            facts=group_inputs(before, groups),
            mode=settings.dcgm_hostengine or target.dcgm_mode,
            pin=settings.dcgm_version,
        )
    }
    refusals = {}
    for probe in refusal_probes(settings, target, targets, groups, main.document):
        run = tool(list(probe.arguments), timeout=case.PLAN_TIMEOUT_SECONDS)
        refusals[probe.refusal.name] = {
            "expected": probe.refusal.describe(),
            "run": case.record_run(case_dir, f"tool-refusal-{probe.refusal.name}", run),
            "errors": verdicts.refusal_errors(run, probe.refusal),
        }
    result["refusals"] = refusals
    stages["refusals"] = [
        error for item in refusals.values() for error in item["errors"]
    ]
    after = case.snapshot(fixture, target, groups, configuration_only=probed)
    write_json_atomic(case_dir / "snapshot-after.json", after)
    stages["no_mutation"] = verdicts.snapshot_drift(before, after)
    store = case.store_state(fixture, settings.regional.cluster_id, since=started)
    write_json_atomic(case_dir / "store-after.json", store)
    stages["incidents"] = verdicts.incident_errors(
        store,
        since=started,
        nodes=[
            node["name"]
            for name in groups
            for node in before["nodegroups"][name]["nodes"]
        ],
        cluster_wide=False,
    )
    identity = dict(fixture.evidence_identity())
    stages["identity"] = (
        []
        if identity == preflight["identity"]
        else [f"the release changed during the case: {identity}"]
    )
    result.update(
        nodegroups=groups,
        stages=stages,
        errors=[
            f"{stage}: {item}" for stage, items in stages.items() for item in items
        ],
        verdict=verdicts.case_verdict(stages),
    )


def execute_case(
    settings: Settings, run_dir: Path, attempt: int, deadline: datetime
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    evidence = case_dir / f"{CASE_ID}.json"
    started = CLOCK.now()
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "started_at": started.isoformat(),
        "maintenance_window_end": deadline.isoformat(),
        "mutation_performed": False,
        "limitations": LIMITATIONS,
    }
    write_json_atomic(evidence, {**result, "stage": "preflight"})
    interrupted: BaseException | None = None
    try:
        audit(settings, case_dir, result, started=started, deadline=deadline)
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
    groups = sorted(preflight.get("nodegroups") or {})
    runs: list[dict[str, Any]] = []
    targets = preflight.get("refusal_targets")
    if groups and targets:
        target = case.site_target(settings.state_dir, settings.regional.cluster_id)
        runs.append(
            {"name": "tool-plan", "arguments": main_arguments(settings, target, groups)}
        )
        runs += [
            {
                "name": f"tool-refusal-{probe.refusal.name}",
                "arguments": list(probe.arguments),
            }
            for probe in refusal_probes(settings, target, targets, groups, None)
        ]
    return {
        "risk": "read-only-signal-replay",
        "predecessor": preflight.get("predecessor"),
        "target": preflight.get("target"),
        "nodegroups": groups,
        "mutation": (
            "none. Every run is gpu-fault-admin prepare-eks-nodegroup's default "
            "read-only plan through the site's deploy-host CLI: AWS describe/list "
            "reads, a temporary kubeconfig for kubectl get, and one SSM "
            "AWS-RunShellScript read-only inventory per planned group (python3.12 "
            "venv import, strace, nvidia-smi -q, rpm -qa, systemctl is-active/show, "
            "pgrep nv-hostengine, dcgmi discovery -l) on its running instances; the "
            "CLI keeps its read-only log under <state-dir>/logs/readonly/"
        ),
        "tool_runs": runs,
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "the target member is not a site-managed eks-managed cluster",
            "a group's node is held by the product (not Ready, cordoned, isolating "
            "taint) or an active incident or workflow names it",
            "the release, the groups, their launch template, instances or Nodes "
            "drifted between plan and execute",
        ],
        "rollback": {
            "nothing_to_roll_back": True,
            "every_tool_run_is_a_read_only_plan": True,
            "no_snapshot_drift_is_part_of_the_verdict": True,
        },
        "preflight": preflight,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded BOOT-036 acceptance: the read-only plan of the site's "
            "plain EKS GPU node groups is COMPLIANT with every node check ok, and "
            "the plan's refusals stop before any change."
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
    ):
        value.add_argument(f"--{name}", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument(
        "--nodegroup",
        action="append",
        default=[],
        help="node group to audit; repeatable. Default: every group labelled for the site",
    )
    value.add_argument(
        "--dcgm-hostengine",
        choices=("embedded", "host"),
        default=None,
        help="passed to the tool; default: the mode each group's label declares",
    )
    value.add_argument(
        "--dcgm-version",
        default=None,
        help="the pin the groups must carry (host mode); default: any paired pin",
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
