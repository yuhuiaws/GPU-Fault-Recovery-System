#!/usr/bin/env python3
"""Acceptance wrapper over ``gpu-fault-admin config spare``.

The supported operator lever for declaring or releasing a warm spare is the
admin CLI (``gpu_fault.admin.warm_spare``). This script keeps the acceptance
site-profile interface the DESTR-003/008 runbooks were written against --
``--site-profile``, ``--spare-node``, ``--baseline`` -- and binds that profile
to the same refusals, the same two-field mutation (spare label and cordon) and
the same baseline record. It holds no logic of its own: every check runs in the
admin module, so the two entry points cannot drift.

Why a wrapper still exists: the acceptance runners read the profile, not a
managed state directory, and DESTR-003 grades the site with a spare that was
declared *outside* the case (its ``preflight_errors`` asserts the declared
spare set is exactly the requested node, which a self-provisioning case would
satisfy by construction). The wrapper keeps that separation without keeping a
second implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin import warm_spare  # noqa: E402
from gpu_fault.admin.atomic_json import write_json_atomic  # noqa: E402
from gpu_fault.admin.warm_spare import (  # noqa: E402
    DECLARE_CONFIRMATION,
    RELEASE_CONFIRMATION,
    AgentState,
    KubectlNodeApi,
    WarmSpareRequest,
    perform_warm_spare,
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
    WarmSpareLiveFixture,
    agent_by_node,
)

# The logic lives in the admin module; these names stay bound here so a reader
# of the runbooks finds the checks where the runbooks say they are.
declare_refusals = warm_spare.declare_refusals
release_refusals = warm_spare.release_refusals
declare = warm_spare.declare
release = warm_spare.release
without_survey = warm_spare.without_survey


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


def agent_state(state: dict[str, Any], node: str) -> AgentState:
    """The profile-side Agent check: exactly one row for the node, or absent."""

    agent = agent_by_node(state, node)
    if agent is None:
        return AgentState(lifecycle_state=None)
    return AgentState(lifecycle_state=str(agent.get("lifecycle_state")))


def store_agent_lookup(warm: WarmSpareLiveFixture) -> warm_spare.AgentLookup:
    def lookup(_cluster_id: str, node: str) -> AgentState:
        try:
            state = warm.store_snapshot()
        except RegionalFixtureError as exc:
            # Unreadable is a refusal, never a pass.
            return AgentState(lifecycle_state=None, error=str(exc))
        return agent_state(state, node)

    return lookup


def node_api(regional: RegionalLiveFixture) -> KubectlNodeApi:
    return KubectlNodeApi(
        [
            "kubectl",
            "--kubeconfig",
            str(regional.settings.gpu_kubeconfig),
            "--context",
            regional.settings.gpu_context,
        ]
    )


def mode(arguments: argparse.Namespace) -> str:
    if arguments.declare:
        return "declare"
    if arguments.release:
        return "release"
    return "check"


def request(settings: Settings, arguments: argparse.Namespace) -> WarmSpareRequest:
    return WarmSpareRequest(
        state_dir=settings.baseline.parent,
        node=settings.node,
        fault_node=settings.fault_node,
        cluster_id=None,
        # The profile has no change reference; the wrapper records the case it
        # is running for so the baseline still says who declared and why.
        reference=arguments.reference or "acceptance-warm-spare",
        mode=mode(arguments),
        confirmation=arguments.confirm,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Declare or release one warm spare GPU node for "
            "GF-REGIONAL-DESTR-003/008 through the admin module. Read-only "
            "unless --declare or --release is given. Operators should use "
            "`gpu-fault-admin config spare` instead."
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
    value.add_argument(
        "--reference",
        default="",
        help="change or case reference recorded on the baseline",
    )
    value.add_argument("--report", type=Path, default=None)
    mode_group = value.add_mutually_exclusive_group()
    mode_group.add_argument("--declare", action="store_true")
    mode_group.add_argument("--release", action="store_true")
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
    try:
        report = perform_warm_spare(
            node_api(regional),
            request(settings, arguments),
            cluster_id=regional.settings.cluster_id,
            record=settings.baseline,
            agent_lookup=store_agent_lookup(warm),
        )
    except warm_spare.WarmSpareError as exc:
        raise RegionalFixtureError(str(exc)) from exc
    if arguments.report is not None:
        write_json_atomic(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
