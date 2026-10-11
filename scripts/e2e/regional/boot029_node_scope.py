"""The ownership-label precondition of a BOOT-029 stage, checked before it acts.

``deploy`` and ``join-cluster`` select a GPU cluster's nodes through the node
groups labelled ``gpu-fault.io/control-plane=<site id>`` and refuse a cluster
without one -- but only after the state directory exists and, for a first
deploy, after minutes of cold image build. The driver therefore asks the
product's own reads first: the site id the CPU cluster derives
(``bootstrap_site.site_identifier`` over the HyperPod identity ``deploy`` would
discover) and, per GPU cluster, the node groups and the ownership labels the
provider API reports (``bootstrap_ownership.resolve_node_scope``). A cluster
that would be refused stops the stage here, with the product's refusal and
the describe evidence in the stage log, before anything is created.

Read-only: ``sagemaker list-clusters`` / ``describe-cluster``, ``eks
list-nodegroups`` / ``describe-nodegroup``. The record (stdout JSON) carries
digests and counts only -- ARNs, the site id and node group names stay out of
it, as every BOOT-029 receipt does; the readable evidence and the product's
refusal go to stderr, which the driver appends to the stage log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Sequence

from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_discovery import find_hyperpod_for_eks
from gpu_fault.admin.bootstrap_ownership import (
    NodeScope,
    eks_nodegroup_facts,
    hyperpod_instance_group_facts,
    resolve_node_scope,
)
from gpu_fault.admin.bootstrap_site import site_identifier
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, ClusterProvider
from scripts.e2e.regional.boot029_providers import gpu_cluster_provider

__all__ = ["cpu_site_id", "gpu_node_scope", "main", "precondition_report"]


def _hyperpod_description(runner: CommandRunner, arn: str) -> dict[str, Any]:
    parsed = Arn.parse(arn)
    if parsed.service == "sagemaker":
        return dict(
            runner.aws_json(
                parsed.region, "sagemaker", "describe-cluster", "--cluster-name", arn
            )
        )
    if parsed.service != "eks":
        raise BootstrapError(
            f"a cluster is identified by its EKS or SageMaker ARN: {arn}"
        )
    hyperpod = find_hyperpod_for_eks(runner, eks_arn=arn, region=parsed.region)
    if hyperpod is None:
        raise BootstrapError(f"no HyperPod cluster orchestrates {arn}")
    return dict(hyperpod)


def cpu_site_id(cpu_arn: str, *, runner: CommandRunner) -> str:
    """The site id ``deploy`` derives for ``cpu_arn`` (a HyperPod control plane)."""

    hyperpod = _hyperpod_description(runner, cpu_arn)
    parsed = Arn.parse(cpu_arn)
    identity = ClusterIdentity(
        input_arn=cpu_arn,
        role="cpu",
        region=parsed.region,
        account_id=parsed.account,
        hyperpod_arn=str(hyperpod.get("ClusterArn") or ""),
        hyperpod_name=str(hyperpod.get("ClusterName") or ""),
        eks_arn=str(
            ((hyperpod.get("Orchestrator") or {}).get("Eks") or {}).get(
                "ClusterArn", ""
            )
        ),
        eks_name="",
        vpc_id="",
        subnet_ids=(),
        node_recovery="",
        context="",
    )
    if not identity.hyperpod_arn or not identity.hyperpod_name:
        raise BootstrapError(f"the CPU cluster {cpu_arn} exposes no HyperPod identity")
    return site_identifier(identity, [])


def gpu_node_scope(
    arn: str, *, provider: str, site_id: str, runner: CommandRunner
) -> NodeScope:
    """The node scope ``deploy``/``join-cluster`` would resolve for ``arn``."""

    parsed = Arn.parse(arn)
    if provider == ClusterProvider.EKS_MANAGED.value:
        eks_name = parsed.resource_name
        names = sorted(
            str(item)
            for item in runner.aws_json(
                parsed.region, "eks", "list-nodegroups", "--cluster-name", eks_name
            ).get("nodegroups", [])
            if item
        )
        facts = [
            eks_nodegroup_facts(
                runner.aws_json(
                    parsed.region,
                    "eks",
                    "describe-nodegroup",
                    "--cluster-name",
                    eks_name,
                    "--nodegroup-name",
                    name,
                ).get("nodegroup")
                or {}
            )
            for name in names
        ]
        return resolve_node_scope(
            facts, provider=provider, cluster=eks_name, site_id=site_id
        )
    hyperpod = _hyperpod_description(runner, arn)
    name = str(hyperpod.get("ClusterName") or "")
    facts = [
        hyperpod_instance_group_facts(group, cluster_name=name)
        for group in hyperpod.get("InstanceGroups") or ()
        if isinstance(group, dict)
    ]
    return resolve_node_scope(
        facts,
        provider=provider,
        cluster=name,
        site_id=site_id,
        hyperpod_cluster_name=name,
    )


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def precondition_report(
    cpu_arn: str, gpu_arns: Sequence[str], *, runner: CommandRunner | None = None
) -> tuple[dict[str, Any], list[str]]:
    """``(record, log lines)``: per GPU cluster, the ownership label state the
    stage is about to depend on. The record holds digests and counts; the log
    lines hold the names, the evidence and the product's refusals."""

    active = runner or CommandRunner()
    site_id = cpu_site_id(cpu_arn, runner=active)
    clusters: list[dict[str, Any]] = []
    lines = [
        f"node-scope precondition: site id {site_id}, label {OWNERSHIP_LABEL}={site_id}"
    ]
    for arn in gpu_arns:
        provider = gpu_cluster_provider(arn, runner=active)
        entry: dict[str, Any] = {"arn_sha256": _digest(arn), "provider": provider}
        try:
            scope = gpu_node_scope(
                arn, provider=provider, site_id=site_id, runner=active
            )
        except BootstrapError as exc:
            entry.update(status="refused")
            lines.append(f"node-scope precondition: {exc}")
        else:
            entry.update(
                status="labelled",
                owned_groups_sha256=[_digest(name) for name in scope.owned_groups],
                node_groups_seen=len(scope.evidence),
            )
            lines.append(
                f"node-scope precondition: {provider} cluster "
                f"{entry['arn_sha256'][:19]} owns {', '.join(scope.owned_groups)}; "
                "node groups seen: "
                + "; ".join(item.describe() for item in scope.evidence)
            )
        clusters.append(entry)
    record = {
        "site_id_sha256": _digest(site_id),
        "ownership_label": OWNERSHIP_LABEL,
        "clusters": clusters,
    }
    return record, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-arn", required=True)
    parser.add_argument("--gpu-arn", action="append", required=True)
    args = parser.parse_args()
    try:
        record, lines = precondition_report(args.cpu_arn, list(args.gpu_arn))
    except BootstrapError as exc:
        print(f"node-scope precondition: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    for line in lines:
        print(line, file=sys.stderr)
    print(json.dumps({"node_scope_precondition": record}, sort_keys=True))
    if any(item["status"] == "refused" for item in record["clusters"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
