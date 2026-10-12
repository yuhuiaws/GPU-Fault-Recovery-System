"""The ownership label between BOOT-029 stages: gone after a de-registration,
back before a re-join.

``remove-cluster`` (stage 2) and ``uninstall`` (stage 4) end with the product's
OWNERSHIP RELEASE step: ``gpu-fault.io/control-plane=<site id>`` leaves the
node groups the site selected (``gpu_fault.admin.ownership_release``). The
driver therefore has two more things to prove and one to do:

* ``released``: after each stage-2 removal and after stage 4, no node group of
  the de-registered cluster carries the label for this site any more (the
  product's own ``describe`` reads; a group still claiming the site fails the
  stage with the evidence in the log).
* ``reapply``: before stage 3 re-joins and before stage 5 re-deploys, the label
  goes back on exactly the groups the previous stage's journal recorded as
  ``released`` -- with the operator's credentials and the exact commands
  ``gpu-fault-admin site-id`` prints (``bootstrap_ownership.label_remedy``:
  ``aws eks update-nodegroup-config ... addOrUpdateLabels={...}``; ``aws
  sagemaker update-cluster`` with the group's full current definition plus the
  label) -- and the driver waits until the provider shows it applied, the way
  an administrator would before a first ``deploy``. A group the journal says
  was retained or left to a manual edit still carries the label and needs
  nothing.

The records (stdout JSON) carry digests and counts only, as every BOOT-029
receipt does; names, evidence and the product's refusals go to stderr, which
the driver appends to the stage log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_discovery import find_hyperpod_for_eks
from gpu_fault.admin.ownership_release import (
    RELEASED,
    ClusterGroups,
    describe_cluster_groups,
    hyperpod_instance_group_specification,
    require_hyperpod_writable,
    wait_for_label,
)
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, ClusterProvider
from scripts.e2e.regional.boot029_node_scope import cpu_site_id
from scripts.e2e.regional.boot029_providers import gpu_cluster_provider

__all__ = [
    "cluster_target",
    "main",
    "reapply_report",
    "released_groups_from_journals",
    "released_report",
]

REMOVAL_STEP = "OWNERSHIP_RELEASED"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def cluster_target(arn: str, *, runner: CommandRunner) -> dict[str, Any]:
    """The site-entry shape of one GPU cluster ARN, as the release step reads it.

    A HyperPod cluster (a SageMaker ARN, or an EKS ARN exactly one HyperPod
    cluster orchestrates) is named by its HyperPod name and its orchestrator
    EKS ARN; a plain EKS cluster by its EKS ARN alone.
    """

    parsed = Arn.parse(arn)
    provider = gpu_cluster_provider(arn, runner=runner)
    if provider == ClusterProvider.EKS_MANAGED.value:
        return {
            "cluster_id": parsed.resource_name,
            "provider": provider,
            "eks_cluster_arn": arn,
            "hyperpod_cluster_name": "",
            "region": parsed.region,
        }
    if parsed.service == "sagemaker":
        hyperpod: dict[str, Any] | None = dict(
            runner.aws_json(
                parsed.region, "sagemaker", "describe-cluster", "--cluster-name", arn
            )
        )
    else:
        hyperpod = find_hyperpod_for_eks(runner, eks_arn=arn, region=parsed.region)
    if hyperpod is None:
        raise BootstrapError(f"no HyperPod cluster orchestrates {arn}")
    name = str(hyperpod.get("ClusterName") or "")
    eks_arn = str(
        ((hyperpod.get("Orchestrator") or {}).get("Eks") or {}).get("ClusterArn", "")
    )
    if not name or not eks_arn:
        raise BootstrapError(f"the HyperPod cluster behind {arn} exposes no identity")
    return {
        "cluster_id": name,
        "provider": provider,
        "eks_cluster_arn": eks_arn,
        "hyperpod_cluster_name": name,
        "region": parsed.region,
    }


def released_report(
    cpu_arn: str, gpu_arns: Sequence[str], *, runner: CommandRunner | None = None
) -> tuple[dict[str, Any], list[str], bool]:
    """``(record, log lines, released)``: per GPU cluster, whether any node group
    still carries this site's label after the de-registration."""

    active = runner or CommandRunner()
    site_id = cpu_site_id(cpu_arn, runner=active)
    clusters: list[dict[str, Any]] = []
    lines = [f"ownership release: site id {site_id}, label {OWNERSHIP_LABEL}={site_id}"]
    released = True
    for arn in gpu_arns:
        target = cluster_target(arn, runner=active)
        groups = describe_cluster_groups(
            active, region=str(target["region"]), target=target
        )
        claiming = [item.name for item in groups.owned_by(site_id)]
        released = released and not claiming
        clusters.append(
            {
                "arn_sha256": _digest(arn),
                "provider": groups.provider.value,
                "node_groups_seen": len(groups.facts),
                "labelled_groups": len(claiming),
                "status": "claimed" if claiming else "released",
            }
        )
        lines.append(
            f"ownership release: {groups.display} "
            + (
                f"still carries the label on {groups.group_kind}(s) "
                + ", ".join(claiming)
                + " -- the de-registration did not release it"
                if claiming
                else "carries the label on no node group"
            )
            + "; node groups seen: "
            + ("; ".join(item.describe() for item in groups.facts) or "none")
        )
    record = {
        "ownership_label_released": {
            "site_id_sha256": _digest(site_id),
            "ownership_label": OWNERSHIP_LABEL,
            "clusters": clusters,
        }
    }
    return record, lines, released


def _journal(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _released_names(evidence: object) -> list[str]:
    if not isinstance(evidence, dict):
        return []
    return [
        str(item["name"])
        for item in evidence.get("groups") or []
        if isinstance(item, dict)
        and item.get("outcome") == RELEASED
        and item.get("name")
    ]


def _same_cluster(entry: dict[str, Any], target: dict[str, Any]) -> bool:
    if str(entry.get("eks_cluster_arn") or "") != str(target["eks_cluster_arn"]):
        return False
    recorded = str(entry.get("hyperpod_cluster_name") or "")
    expected = str(target["hyperpod_cluster_name"] or "")
    return not (recorded and expected) or recorded == expected


def released_groups_from_journals(
    state: Path, *, stage: int, target: dict[str, Any]
) -> tuple[str, list[str]]:
    """``(journal, group names)`` the previous stage recorded as released for ``target``.

    Stage 3 reads the COMPLETED ``remove-cluster`` tombstone of the cluster
    (stage 2's), stage 5 the COMPLETED uninstall journal (stage 4's). A cluster
    without release evidence is refused: the driver never guesses which groups
    to label.
    """

    if stage == 3:
        for path in sorted((state / "remove-cluster").glob("*/state.json")):
            value = _journal(path)
            if value is None or value.get("phase") != "COMPLETED":
                continue
            journal_target = value.get("target") or {}
            if not isinstance(journal_target, dict) or not _same_cluster(
                journal_target, target
            ):
                continue
            evidence = (value.get("evidence") or {}).get(REMOVAL_STEP)
            if evidence is None:
                raise BootstrapError(
                    f"{path.relative_to(state)} has no {REMOVAL_STEP} evidence: the "
                    "removal did not run the release step"
                )
            return str(path.relative_to(state)), _released_names(evidence)
    elif stage == 5:
        value = _journal(state / "uninstall" / "state.json")
        if value is not None and value.get("phase") == "COMPLETED":
            release = value.get("ownership_release")
            if not isinstance(release, dict):
                raise BootstrapError(
                    "uninstall/state.json has no ownership_release evidence: the "
                    "uninstall did not run the release phase"
                )
            for entry in release.get("clusters") or []:
                if isinstance(entry, dict) and _same_cluster(entry, target):
                    return "uninstall/state.json", _released_names(entry)
    else:
        raise BootstrapError("the label is re-applied before stage 3 and stage 5 only")
    raise BootstrapError(
        f"no COMPLETED journal of {state} records a release for {target['cluster_id']}; "
        "label the GPU node groups by hand with the commands gpu-fault-admin "
        "site-id prints, then resume"
    )


def _apply(
    runner: CommandRunner,
    groups: ClusterGroups,
    names: Sequence[str],
    *,
    region: str,
    site_id: str,
) -> None:
    """The exact ``site-id`` commands, for the groups still missing the label."""

    if groups.provider is ClusterProvider.EKS_MANAGED:
        for name in names:
            runner.aws_json(
                region,
                "eks",
                "update-nodegroup-config",
                "--cluster-name",
                groups.cluster,
                "--nodegroup-name",
                name,
                "--labels",
                f"addOrUpdateLabels={{{OWNERSHIP_LABEL}={site_id}}}",
                mutate=True,
            )
        return
    require_hyperpod_writable(groups, names)
    specifications = []
    for name in names:
        details = groups.documents.get(name) or {}
        kubernetes = details.get("KubernetesConfig") or {}
        current = {
            str(key): str(value)
            for key, value in (kubernetes.get("CurrentLabels") or {}).items()
        }
        specifications.append(
            hyperpod_instance_group_specification(
                details, labels={**current, OWNERSHIP_LABEL: site_id}
            )
        )
    runner.aws_json(
        region,
        "sagemaker",
        "update-cluster",
        "--cluster-name",
        groups.cluster,
        "--instance-groups",
        json.dumps(specifications, separators=(",", ":")),
        mutate=True,
    )


def _in_flight(groups: ClusterGroups, name: str, site_id: str) -> bool:
    """A HyperPod group whose ``DesiredLabels`` already carry the id: only wait."""

    if groups.provider is not ClusterProvider.HYPERPOD:
        return False
    kubernetes = (groups.documents.get(name) or {}).get("KubernetesConfig") or {}
    desired = kubernetes.get("DesiredLabels") or {}
    return str(desired.get(OWNERSHIP_LABEL) or "") == site_id


def reapply_report(
    cpu_arn: str,
    gpu_arns: Sequence[str],
    *,
    state: Path,
    stage: int,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    timeout_seconds: float = 600.0,
    interval_seconds: float = 10.0,
) -> tuple[dict[str, Any], list[str]]:
    """Re-label the released groups of every GPU cluster; ``(record, log lines)``."""

    active = runner or CommandRunner()
    site_id = cpu_site_id(cpu_arn, runner=active)
    clusters: list[dict[str, Any]] = []
    lines = [
        f"ownership re-apply: site id {site_id}, label {OWNERSHIP_LABEL}={site_id}"
    ]
    for arn in gpu_arns:
        target = cluster_target(arn, runner=active)
        region = str(target["region"])
        journal, names = released_groups_from_journals(
            state, stage=stage, target=target
        )
        groups = describe_cluster_groups(active, region=region, target=target)
        missing = [name for name in names if name not in groups.documents]
        if missing:
            raise BootstrapError(
                f"{groups.display} no longer has {groups.group_kind}(s) "
                + ", ".join(missing)
                + f" that {journal} recorded as released"
            )
        labelled = {item.name for item in groups.owned_by(site_id)}
        to_apply = [
            name
            for name in names
            if name not in labelled and not _in_flight(groups, name, site_id)
        ]
        waited = 0.0
        if to_apply:
            _apply(active, groups, to_apply, region=region, site_id=site_id)
        if any(name not in labelled for name in names):
            waited, _after = wait_for_label(
                active,
                region=region,
                target=target,
                names=names,
                value=site_id,
                description=f"the re-applied {OWNERSHIP_LABEL}",
                sleep=sleep,
                clock=clock,
                timeout_seconds=timeout_seconds,
                interval_seconds=interval_seconds,
            )
        clusters.append(
            {
                "arn_sha256": _digest(arn),
                "provider": groups.provider.value,
                "journal_path_sha256": _digest(journal),
                "groups_sha256": [_digest(name) for name in names],
                "already_labelled": len([name for name in names if name in labelled]),
                "applied": len(to_apply),
                "converged_after_seconds": round(waited, 3),
            }
        )
        lines.append(
            f"ownership re-apply: {groups.display} {groups.group_kind}(s) "
            + (", ".join(names) or "none")
            + f" ({journal}): applied to "
            + (", ".join(to_apply) or "none")
            + f", converged after {waited:.1f}s"
        )
    record = {
        "ownership_label_reapplied": {
            "site_id_sha256": _digest(site_id),
            "ownership_label": OWNERSHIP_LABEL,
            "stage": stage,
            "clusters": clusters,
        }
    }
    return record, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("released", "reapply"))
    parser.add_argument("--cpu-arn", required=True)
    parser.add_argument("--gpu-arn", action="append", required=True)
    parser.add_argument(
        "--state", type=Path, help="reapply: the journals' state directory"
    )
    parser.add_argument(
        "--stage", type=int, choices=(3, 5), help="reapply: the stage ahead"
    )
    args = parser.parse_args()
    try:
        if args.action == "released":
            record, lines, released = released_report(args.cpu_arn, list(args.gpu_arn))
        else:
            if args.state is None or args.stage is None:
                parser.error("reapply requires --state and --stage")
            record, lines = reapply_report(
                args.cpu_arn, list(args.gpu_arn), state=args.state, stage=args.stage
            )
            released = True
    except BootstrapError as exc:
        print(f"ownership {args.action}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    for line in lines:
        print(line, file=sys.stderr)
    print(json.dumps(record, sort_keys=True))
    if not released:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
