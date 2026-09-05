#!/usr/bin/env python3
"""Declare or release one operator-provided warm spare GPU node.

`GF-REGIONAL-DESTR-003` refuses to mutate anything until a spare is labeled,
cordoned, monitored and unreserved, and its `read_only_preflight` already names
every condition that fails. What had no supported command was the *declaration*
itself, so that precondition was satisfied by hand `kubectl label` plus
`kubectl cordon` -- the manual step the acceptance standard forbids, and one
that leaves nothing to restore from if the operator forgets what the node
looked like beforehand.

This is a helper, not a case entry point, and it deliberately performs the
smallest mutation that a declaration can be: the spare label and the cordon.

* It does not write `gpu-fault.io/spare-pool-state`. That annotation belongs to
  the control plane, and DESTR-003 accepts it absent. Writing `AVAILABLE` by
  hand would fabricate the very state the case exists to observe.
* It does not make the case self-provisioning. DESTR-003 grades the site;
  `preflight_errors` asserts the declared spare set is *exactly* the requested
  node in order to catch a site with a stray label elsewhere, and a case that
  created its own spare would be grading its own setup -- the assertion would
  hold by construction even if the coordinator's filter had drifted away from
  it.

So the checks below are refusals to declare something that cannot honestly be a
spare. They are not a substitute for the case's own preflight, which still runs
against the site afterwards and is still the authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    install_abort_signals,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.site_profile import (  # noqa: E402
    bind_site_profile,
    install_site_profile,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    OWNERSHIP_ANNOTATIONS,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    NodeMutationFixture,
    NodePatch,
    WarmSpareLiveFixture,
    agent_by_node,
    instance_type,
)

DECLARE_CONFIRMATION = "DECLARE_WARM_SPARE_CORDON"
RELEASE_CONFIRMATION = "RELEASE_WARM_SPARE_UNCORDON"


@dataclass(frozen=True)
class Settings:
    node: str
    fault_node: str
    hyperpod_cluster: str
    baseline: Path


def configure(arguments: argparse.Namespace) -> Settings:
    return Settings(
        node=required(
            arguments.spare_node or os.getenv("GPU_FAULT_SPARE_NODE", ""),
            "spare node",
        ),
        fault_node=arguments.fault_node.strip()
        or os.getenv("GPU_FAULT_FAULT_NODE", "").strip(),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        baseline=Path(
            required(arguments.baseline, "baseline record path")
        ).expanduser(),
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def topology(snapshot: dict[str, Any]) -> dict[str, str | None]:
    return {
        "instance_group": snapshot["labels"].get(INSTANCE_GROUP_LABEL),
        "instance_type": instance_type(snapshot),
    }


def declare_refusals(
    settings: Settings,
    *,
    spare: dict[str, Any],
    fault: dict[str, Any] | None,
    declared: list[str],
    workloads: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    refusals = []
    if spare["ready"] != "True":
        refusals.append("node is not Ready")
    if spare["labels"].get(SPARE_LABEL) == "true":
        refusals.append("node is already declared as a spare")
    if spare["labels"].get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
        refusals.append("HyperPod health label is not Schedulable")
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        refusals.append("node already carries a spare reservation")
    if spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION) not in {
        None,
        "AVAILABLE",
    }:
        refusals.append("spare pool state is not AVAILABLE")
    if any(item.get("key") == QUARANTINE_TAINT for item in spare["taints"]) or any(
        spare["annotations"].get(key) for key in OWNERSHIP_ANNOTATIONS
    ):
        refusals.append("node carries quarantine ownership from an earlier incident")
    if [item for item in workloads if item.get("node") == settings.node]:
        # Cordoning does not evict, so a spare declared under a live job stays
        # busy and the case would allocate a node that is already working.
        refusals.append("node still has an active GPU workload")
    agent = agent_by_node(state, settings.node)
    if agent is None or agent.get("lifecycle_state") != "ACTIVE":
        refusals.append("node does not have exactly one ACTIVE Agent")
    if settings.node in declared:
        refusals.append("node is already in the declared spare set")
    if fault is not None:
        if settings.node == settings.fault_node:
            refusals.append("spare and fault node are identical")
        if topology(fault) != topology(spare):
            refusals.append(
                "topology does not match the fault node: "
                f"{topology(fault)} vs {topology(spare)}"
            )
    return refusals


def release_refusals(spare: dict[str, Any]) -> list[str]:
    refusals = []
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        # The pool allocated this node to an incident. Uncordoning it now would
        # hand a reserved spare back to the scheduler while a workflow is still
        # counting on it.
        refusals.append("node is reserved by an incident; release it through the case")
    if any(item.get("key") == QUARANTINE_TAINT for item in spare["taints"]):
        refusals.append(
            "node is quarantined; use restore_validated_quarantine.py instead"
        )
    return refusals


def survey(settings: Settings, warm: WarmSpareLiveFixture, regional) -> dict[str, Any]:
    spare = warm.node_snapshot(settings.node)
    fault = warm.node_snapshot(settings.fault_node) if settings.fault_node else None
    declared = warm.spare_nodes()
    result: dict[str, Any] = {
        "observed_at": now(),
        "cluster_id": regional.settings.cluster_id,
        "node": spare,
        "topology": topology(spare),
        "declared_spares": declared,
        "gpu_workloads": [
            item for item in regional.gpu_workloads() if item.get("node")
        ],
        "agents": warm.store_snapshot().get("agents"),
    }
    if fault is not None:
        result["fault_node"] = fault
        result["fault_topology"] = topology(fault)
    return result


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegionalFixtureError(f"no warm-spare baseline record at {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def declare(settings: Settings, warm: WarmSpareLiveFixture, report: dict[str, Any]):
    if settings.baseline.is_file() and not read_baseline(settings.baseline).get(
        "released_at"
    ):
        raise RegionalFixtureError(
            f"{settings.baseline} still records an unreleased declaration; "
            "release it before declaring again"
        )
    mutation = NodeMutationFixture(
        warm,
        settings.node,
        label_keys=(SPARE_LABEL,),
        track_unschedulable=True,
    )
    # Written before the mutation, so an interrupted declaration still leaves an
    # exact record of what to put back.
    record = {
        "node": settings.node,
        "declared_at": now(),
        "confirmation": DECLARE_CONFIRMATION,
        "baseline": mutation.baseline,
        "pre_declaration_survey": report,
    }
    write_json_atomic(settings.baseline, record)
    mutation.apply(
        NodePatch(labels={SPARE_LABEL: "true"}, annotations={}, unschedulable=True)
    )
    after = warm.node_snapshot(settings.node)
    if after["labels"].get(SPARE_LABEL) != "true" or not after["unschedulable"]:
        raise RegionalFixtureError(
            "declaration did not take effect: the node is not labeled and cordoned"
        )
    record["declared_state"] = after
    record["declared_spares"] = warm.spare_nodes()
    write_json_atomic(settings.baseline, record)
    return record


def release(settings: Settings, warm: WarmSpareLiveFixture, report: dict[str, Any]):
    record = read_baseline(settings.baseline)
    if record.get("released_at"):
        raise RegionalFixtureError(
            f"{settings.baseline} was already released at {record['released_at']}"
        )
    if record.get("node") != settings.node:
        raise RegionalFixtureError(
            f"{settings.baseline} records node {record.get('node')}, not {settings.node}"
        )
    mutation = NodeMutationFixture(
        warm,
        settings.node,
        label_keys=(SPARE_LABEL,),
        track_unschedulable=True,
    )
    # Restore against the recorded baseline rather than the node as it is now,
    # so a node that was already cordoned before the declaration stays cordoned.
    mutation.baseline = record["baseline"]
    restored = mutation.restore()
    record["released_at"] = now()
    record["release_survey"] = report
    record["restored_state"] = restored
    write_json_atomic(settings.baseline, record)
    return record


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Declare or release one warm spare GPU node for "
            "GF-REGIONAL-DESTR-003/008. Read-only unless --declare or "
            "--release is given."
        )
    )
    value.add_argument("--site-profile", default="")
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--hyperpod-cluster", default="")
    # --spare-node, not --node: this shares a site profile with the collector
    # cases, whose --node is an entirely different machine. A helper that cordons
    # whichever node happened to be in the profile under the shorter name is a
    # way to take the wrong node out of schedulable capacity.
    value.add_argument(
        "--spare-node", default="", help="the node to declare as a spare"
    )
    value.add_argument(
        "--fault-node",
        default="",
        help="optional: verify the spare matches this node's topology",
    )
    value.add_argument(
        "--baseline",
        default="",
        help="file recording the node's pre-declaration labels and cordon state",
    )
    value.add_argument("--report", type=Path, default=None)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--declare", action="store_true")
    mode.add_argument("--release", action="store_true")
    value.add_argument(
        "--confirm",
        default="",
        help=(
            f"--declare requires exactly {DECLARE_CONFIRMATION}; "
            f"--release requires exactly {RELEASE_CONFIRMATION}"
        ),
    )
    # Not a live_driver_guard runner, so it binds the profile itself.
    return bind_site_profile(value)


def main() -> int:
    install_site_profile()
    install_abort_signals()
    arguments = parser().parse_args()
    settings = configure(arguments)
    regional = RegionalLiveFixture(settings_from_arguments(arguments))
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    report = survey(settings, warm, regional)

    if arguments.declare:
        if arguments.confirm != DECLARE_CONFIRMATION:
            raise RegionalFixtureError(
                f"--declare requires --confirm {DECLARE_CONFIRMATION}"
            )
        refusals = declare_refusals(
            settings,
            spare=report["node"],
            fault=report.get("fault_node"),
            declared=report["declared_spares"],
            workloads=report["gpu_workloads"],
            state={"agents": report["agents"]},
        )
        report["refusals"] = refusals
        if refusals:
            raise RegionalFixtureError(
                "node cannot be declared a warm spare: " + "; ".join(refusals)
            )
        report["declaration"] = declare(settings, warm, report)
    elif arguments.release:
        if arguments.confirm != RELEASE_CONFIRMATION:
            raise RegionalFixtureError(
                f"--release requires --confirm {RELEASE_CONFIRMATION}"
            )
        refusals = release_refusals(report["node"])
        report["refusals"] = refusals
        if refusals:
            raise RegionalFixtureError(
                "node cannot be released: " + "; ".join(refusals)
            )
        report["release"] = release(settings, warm, report)
    else:
        report["refusals"] = declare_refusals(
            settings,
            spare=report["node"],
            fault=report.get("fault_node"),
            declared=report["declared_spares"],
            workloads=report["gpu_workloads"],
            state={"agents": report["agents"]},
        )
        report["mode"] = "read-only"

    if arguments.report is not None:
        write_json_atomic(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
