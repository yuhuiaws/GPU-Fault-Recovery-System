"""A GPU cluster without nodes through status, verify, deploy and rollback.

Live, the only GPU node group of an eks-managed cluster was scaled to zero and
the departed-node cleanup retired its two Agents. ``status`` then failed three
checks (``regional_contexts``: no Kubernetes nodes; ``gpu_cluster``: no Ready
"HyperPod" nodes; ``control_api``: no registered Node Agents) and every deploy
was refused at preflight. A cluster proven parked is now a named state that
does not fail, its own Deployments and executor still judged, and a release
of any kind proceeds -- including its failure path: the automatic rollback's
unscoped ``sync-state`` accepts the cluster too.

A cluster without nodes that is not parked -- its group still wants nodes:
an outage -- is *unavailable*: ``status`` keeps failing on it, with the
reason, while a deploy, its preflight and verify file its zero-node findings
as ``WARN`` and proceed; every other finding of the cluster still gates. Only
a release that could not finish is refused up front; zero nodes on every
cluster refuse nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_checks_readiness as READINESS
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_deployment_inventory as INVENTORY
from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_release_agent_set as AGENT_SET
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_preflight as PREFLIGHT
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_admin_checks_coverage import (
    annotate_kubernetes_nodes,
    check_agent_coverage,
    judge_cluster_fleet,
)
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from scripts import release_deploy_evidence, release_failure_recovery
from tests.regional._cov95_release_support import NEW_IMAGE, OLD_IMAGE, deployment
from tests.regional._parked_cluster_support import (
    EKS,
    GROUP,
    HYPERPOD,
    OWNERSHIP,
    SITE,
    ParkedSite,
    agent,
    node,
    nodegroup,
)

NOT_PARKED = f"not parked: node group {GROUP} wants 2 node(s)"
WANTS = f"node group {GROUP} wants 2 node(s) (desiredSize 2): its nodes are lost or still joining"


def _revoked(node_id: str) -> dict[str, Any]:
    return {"node_id": node_id, "lifecycle_state": "REVOKED"}


def _cluster_section(site: ParkedSite) -> dict[str, Any]:
    """``control_api``'s section of the parked cluster, annotated as the
    report annotates it."""

    report = {
        "clusters": {
            EKS: {
                "expected_node_ids": [],
                "agents": [_revoked("ip-10-0-1-10"), _revoked("ip-10-0-1-11")],
                "fleet_readiness": None,
                "collector_readiness": {
                    "cluster_id": EKS,
                    "ready": False,
                    "nodes": [],
                    "retired_nodes": [{"node_id": "ip-10-0-1-10"}],
                },
            }
        }
    }
    annotate_kubernetes_nodes(report, {EKS: []}, release=site)
    return report


# --- control_api: coverage, fleet and collector readiness ------------------------


def test_a_parked_cluster_is_covered_by_zero_agents() -> None:
    site = ParkedSite()
    report = _cluster_section(site)
    section = report["clusters"][EKS]

    assert section["parked"]["parked"] is True, section["parked"]
    assert check_agent_coverage(EKS, section) == [], "zero Agents for zero nodes"
    release = SimpleNamespace(config=SimpleNamespace(clusters=[]))
    judge_cluster_fleet(release, EKS, section)  # no Runtime Profile, no fleet question
    evidence = READINESS.judge_collector_readiness(site, report, grace_seconds=0)
    assert evidence["ready"] is True, evidence
    assert evidence["zero_node_not_awaited"] == [EKS]
    with PARKED.gating_findings(site):
        assert check_agent_coverage(EKS, section, release=site) == [], (
            "parked is a state, never a finding: status passes too"
        )


def test_status_fails_an_unavailable_clusters_coverage_and_says_why() -> None:
    """An outage, not a park: in ``status`` the refusal keeps its words, says
    why and carries the proof for the report's ``zero_node_clusters``."""

    site = ParkedSite(desired=2)
    report = _cluster_section(site)
    section = report["clusters"][EKS]

    with PARKED.gating_findings(site):
        with pytest.raises(CheckFailure) as refused:
            judge_cluster_fleet(site, EKS, section)
        with pytest.raises(CheckFailure, match="no ACTIVE Agent has reported"):
            READINESS.judge_collector_readiness(site, report, grace_seconds=0)
    assert str(
        refused.value
    ) == f"{EKS} has no registered Node Agents; not parked: " + (WANTS)
    assert refused.value.details["clusters"][EKS]["parked"]["parked"] is False
    with pytest.raises(ReleaseError, match=re.escape(NOT_PARKED)):
        check_agent_coverage(EKS, section)  # a caller without a release: as before


def test_a_verify_files_an_unavailable_clusters_coverage_and_waits_for_nothing() -> (
    None
):
    """Outside ``status`` the same zero-node findings do not gate: filed under
    the cluster's section, narrated, and the check reads ``WARN``."""

    site = ParkedSite(desired=2)
    report = _cluster_section(site)
    section = report["clusters"][EKS]

    judge_cluster_fleet(site, EKS, section)
    evidence = READINESS.judge_collector_readiness(site, report, grace_seconds=0)

    finding = f"{EKS} has no registered Node Agents; not parked: {WANTS}"
    assert section["zero_node_findings"] == [finding]
    assert evidence["ready"] is True and evidence["zero_node_not_awaited"] == [EKS]
    verify = PARKED.report_fields(
        [{"name": "control_api", "status": "PASS", "summary": "ok", "details": report}]
    )
    (check,) = verify["checks"]
    assert check["status"] == "WARN", check
    assert check["summary"] == (
        "ok; not gating (GPU cluster without nodes, unavailable): " + finding
    )
    entry = verify["zero_node_clusters"][EKS]
    assert entry["state"] == "unavailable" and entry["reason"] == WANTS
    assert entry["not_gating_findings"] == [
        {"check": "control_api", "finding": finding}
    ]


def test_a_cluster_with_nodes_is_never_asked_about_its_provider() -> None:
    site = ParkedSite()
    report = {"clusters": {HYPERPOD: {"expected_node_ids": ["node-a"], "agents": []}}}

    annotate_kubernetes_nodes(report, {HYPERPOD: ["node-a"]}, release=site)

    assert "parked" not in report["clusters"][HYPERPOD]
    assert site.provider.calls == [], "no provider read for a cluster with nodes"
    with pytest.raises(
        ReleaseError, match=f"^{HYPERPOD} has no registered Node Agents$"
    ):
        check_agent_coverage(HYPERPOD, report["clusters"][HYPERPOD])


# --- gpu_cluster:<id> -----------------------------------------------------------


class GpuClusterRelease:
    """The reads of one ``gpu_cluster:<id>`` check of a verify."""

    def __init__(self, site: ParkedSite, target: Any, items: list[dict[str, Any]]):
        self.site = site
        self.target = target
        self.items = items
        self.ready_replicas = 1
        self.tls = {"status": "ok"}
        self.execs: list[list[str]] = []
        self.node_wheel_sha = "n" * 64
        self.runner = self
        self.config = SimpleNamespace(
            site_name="test-site", namespace="gpu-fault-system", clusters=[target]
        )

    # runner
    dry_run = False

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        if arguments[:1] == ["aws"]:
            return self.site.provider.run(arguments, **kwargs)
        self.execs.append(list(arguments))
        return json.dumps(self.tls) if "-c" in arguments else "ready"

    def probe_output(self, arguments: list[str], **kwargs: Any) -> tuple[int, str, str]:
        return self.site.provider.probe_output(arguments, **kwargs)

    # release
    def _gpu(self, _target: Any, *arguments: str) -> list[str]:
        return list(arguments)

    def _get_json(self, arguments: list[str]) -> dict[str, Any]:
        if "nodes" in arguments:
            return {"items": self.items}
        return {
            "spec": {"replicas": 1},
            "status": {"readyReplicas": self.ready_replicas},
        }


def _verify_gpu_cluster(
    monkeypatch: pytest.MonkeyPatch, release: GpuClusterRelease
) -> tuple[dict[str, Any], dict[str, Any]]:
    for name in (
        "_check_contexts",
        "check_cpu_secrets",
        "_check_cpu_workloads",
        "check_email_notifications",
        "check_aurora_refresh",
        "check_adot_self_metrics",
        "_verify_profile",
        "run_read_only_verifiers",
        "_check_runtime_component_identity",
        "_check_control_api",
        "_check_nlb_runtime",
        "_check_aurora",
        "_check_monitoring",
    ):
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CheckValue(name)
        )
    monkeypatch.setattr(
        CHECKS, "node_scope_report", lambda *_args: {"selector": OWNERSHIP}
    )
    report = CHECKS.build_health_report(release, mode="verify")
    check = next(
        item
        for item in report["checks"]
        if item["name"] == f"gpu_cluster:{release.target.cluster_id}"
    )
    return report, check


def test_a_parked_cluster_verifies_its_deployments_and_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ParkedSite()
    release = GpuClusterRelease(site, site.eks, [])

    report, check = _verify_gpu_cluster(monkeypatch, release)

    assert check["status"] == "PASS", check
    assert check["summary"] == f"{EKS} data plane is healthy; parked at zero GPU nodes"
    assert check["details"]["parked"]["groups"][0]["name"] == GROUP
    assert check["details"]["nodes"] == []
    assert check["details"]["tls"] == {"status": "ok"}
    assert any(
        "deployment/gpu-fault-cluster-executor" in call for call in release.execs
    ), "the executor's readiness and TLS path are still probed"
    (entry,) = report["zero_node_clusters"].values()
    assert list(report["zero_node_clusters"]) == [EKS], "named at the report's top"
    assert entry["state"] == "parked" and entry["groups"][0]["name"] == GROUP
    assert report["healthy"] is True


def test_a_parked_cluster_still_fails_on_its_own_deployments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ParkedSite()
    release = GpuClusterRelease(site, site.eks, [])
    release.ready_replicas = 0

    _report, check = _verify_gpu_cluster(monkeypatch, release)

    assert check["status"] == "FAIL", check
    assert "parked" not in check["summary"]


def test_an_unavailable_cluster_verifies_its_deployments_and_files_its_nodes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site = ParkedSite(desired=2)
    release = GpuClusterRelease(site, site.eks, [])

    report, check = _verify_gpu_cluster(monkeypatch, release)

    finding = f"{EKS} has no Ready GPU nodes; not parked: {WANTS}"
    assert check["status"] == "WARN", check
    assert check["summary"] == (
        f"{EKS} data plane Deployments and executor are healthy; no GPU node "
        "(unavailable); not gating (GPU cluster without nodes, unavailable): " + finding
    )
    assert any(
        "deployment/gpu-fault-cluster-executor" in call for call in release.execs
    ), "the executor's readiness and TLS path still gate"
    assert report["healthy"] is True
    assert report["zero_node_clusters"][EKS]["not_gating_findings"] == [
        {"check": f"gpu_cluster:{EKS}", "finding": finding}
    ]
    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("zero-node-gpu-cluster ")
    )
    for field in (
        f"cluster={EKS}",
        "state=unavailable",
        f"check=gpu_cluster:{EKS}",
        "gating=no",
        "finding=" + json.dumps(finding),
    ):
        assert field in line, line


def test_an_unavailable_cluster_still_fails_on_its_own_deployments_and_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ParkedSite(desired=2)
    release = GpuClusterRelease(site, site.eks, [])
    release.tls = {"status": "refused"}

    report, check = _verify_gpu_cluster(monkeypatch, release)

    assert check["status"] == "FAIL", check
    assert check["summary"] == f"{EKS} TLS probe did not return ok"
    assert report["healthy"] is False


def test_status_fails_an_unavailable_cluster_without_ready_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ParkedSite(desired=2)
    release = GpuClusterRelease(site, site.eks, [])

    with PARKED.gating_findings(release):
        _report, check = _verify_gpu_cluster(monkeypatch, release)

    assert check["status"] == "FAIL", check
    assert check["summary"] == f"{EKS} has no Ready GPU nodes; not parked: {WANTS}"
    assert check["details"]["parked"]["parked"] is False


@pytest.mark.parametrize(
    "provider,items,message",
    [
        (
            "eks",
            [node("ip-10-0-1-10", {"gpu-fault.io/control-plane": SITE}, ready=False)],
            f"{EKS} has no Ready GPU nodes",
        ),
        (
            "hyperpod",
            [
                node(
                    "node-a",
                    {"sagemaker.amazonaws.com/cluster-name": f"hp-{HYPERPOD}"},
                    ready=False,
                )
            ],
            f"{HYPERPOD} has no Ready HyperPod nodes",
        ),
    ],
    ids=["eks-not-ready", "hyperpod-not-ready"],
)
def test_a_cluster_without_ready_nodes_is_named_by_its_own_kind_of_node(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    items: list[dict[str, Any]],
    message: str,
) -> None:
    site = ParkedSite(desired=2)
    target = site.eks if provider == "eks" else site.config.clusters[0]
    release = GpuClusterRelease(site, target, items)

    _report, check = _verify_gpu_cluster(monkeypatch, release)

    assert check["status"] == "FAIL", check
    assert check["summary"] == message


# --- regional_contexts (deploy preflight, status --full, verify) ------------------


class ContextSite(ParkedSite):
    """What ``ensure_region_contexts`` reads of the CPU and the parked cluster."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config.cpu_hyperpod_cluster_name = "hp-cpu"
        self.config.clusters = (
            replace(self.eks, agent_endpoint_allowed_cidrs=("10.0.0.0/16",)),
        )
        self.iam: list[str] = []
        provider_run = self.provider.run

        def run(arguments: list[str], **kwargs: Any) -> str:
            if "config" in arguments and "view" in arguments:
                return (
                    self.config.clusters[0].eks_cluster_arn
                    if "--context" in arguments
                    else self.config.cpu_eks_arn
                )
            if "--raw=/readyz" in arguments:
                return "ok"
            if arguments[:3] == ["aws", "sagemaker", "describe-cluster"]:
                return json.dumps(
                    {
                        "EksClusterArn": self.config.cpu_eks_arn,
                        "NodeRecovery": "None",
                        "InstanceGroups": [],
                    }
                )
            if "nodes" in arguments and "-l" in arguments:
                return json.dumps({"items": []})
            return provider_run(arguments, **kwargs)

        self.provider.run = run  # type: ignore[method-assign]

    def _validate_executor_iam_role(self, target: Any) -> None:
        self.iam.append(target.cluster_id)


def _deployments_listing(site: ContextSite) -> None:
    site.documents[(EKS, "deployments", "")] = {"items": []}


def test_the_context_preflight_records_a_parked_cluster() -> None:
    site = ContextSite()
    _deployments_listing(site)

    facts = PREFLIGHT.ensure_region_contexts(site)

    proof = facts["providers"][EKS]["parked"]
    assert proof["parked"] is True, proof
    assert [(item["name"], item["desired"]) for item in proof["groups"]] == [(GROUP, 0)]
    assert site.iam == [EKS], "the executor IAM is judged as for every cluster"
    describes = [
        call
        for call in site.provider.calls
        if "describe-nodegroup" in call and EKS in call
    ]
    assert len(describes) == 2, "the proof reuses the documents the gate read"


def test_the_context_preflight_files_an_outage_and_still_judges_the_cluster() -> None:
    """A deploy's preflight over an unavailable cluster: the zero-node finding
    is filed, every other gate of the cluster still runs."""

    site = ContextSite(desired=2)
    _deployments_listing(site)

    facts = PREFLIGHT.ensure_region_contexts(site)

    entry = facts["providers"][EKS]
    assert entry["parked"]["parked"] is False, entry
    assert entry["zero_node_findings"] == [
        f"{EKS} has no Kubernetes nodes; not parked: {WANTS}"
    ]
    assert site.iam == [EKS], "the executor IAM still gates"


def test_status_refuses_the_contexts_of_an_unavailable_cluster() -> None:
    site = ContextSite(desired=2)
    _deployments_listing(site)

    with PARKED.gating_findings(site), pytest.raises(CheckFailure) as refused:
        PREFLIGHT.ensure_region_contexts(site)

    assert str(refused.value) == f"{EKS} has no Kubernetes nodes; not parked: {WANTS}"
    assert refused.value.details["providers"][EKS]["parked"]["parked"] is False
    assert site.iam == [], "the refusal comes before IAM acceptance, as before"


# --- status ------------------------------------------------------------------------


def test_status_names_the_parked_cluster_in_its_header_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ParkedSite()
    section = _cluster_section(site)
    monkeypatch.setattr(
        CHECKS, "_check_cpu_workloads", lambda _release: CheckValue("CPU workloads")
    )
    monkeypatch.setattr(
        CHECKS,
        "_check_control_api",
        lambda _release, **_kwargs: CheckValue("control API", section),
    )
    monkeypatch.setattr(
        ADMIN,
        "build_release_summary",
        lambda _release, **_kwargs: {"live_release": {}, "next_deploy": {}},
    )

    status = ADMIN.build_quick_status(site, state={})
    lines = ADMIN.status_header_lines(status, json_destination="status.json")

    assert status["healthy"] is True
    assert list(status["zero_node_clusters"]) == [EKS]
    assert status["zero_node_clusters"][EKS]["state"] == "parked"
    assert "failing checks: none" in lines
    parked = [line for line in lines if line.startswith("parked GPU cluster")]
    assert parked and parked[0].startswith(
        f"parked GPU cluster {EKS} (eks-managed): no node selected; "
    ), lines


class StatusSite(ParkedSite):
    def _apply_health_baseline(self, _state: Any) -> None:
        return None


def test_status_keeps_an_unavailable_cluster_red_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator's view: healthy NO, the failing check, and one header line
    with the reason; the gate is ``status``'s alone."""

    site = StatusSite(desired=2)
    report = _cluster_section(site)

    def control_api(release: Any, **_kwargs: Any) -> CheckValue:
        assert PARKED.gating(release), "status gates the zero-node findings"
        judge_cluster_fleet(release, EKS, report["clusters"][EKS])
        return CheckValue("control API", report)

    monkeypatch.setattr(
        CHECKS, "_check_cpu_workloads", lambda _release: CheckValue("CPU workloads")
    )
    monkeypatch.setattr(CHECKS, "_check_control_api", control_api)
    monkeypatch.setattr(
        ADMIN,
        "build_release_summary",
        lambda _release, **_kwargs: {"live_release": {}, "next_deploy": {}},
    )

    status = ADMIN.build_status(site, full=False)
    lines = ADMIN.status_header_lines(status, json_destination="status.json")

    assert status["healthy"] is False
    assert "failing checks: control_api" in lines
    entry = status["zero_node_clusters"][EKS]
    assert entry["state"] == "unavailable" and entry["reason"] == WANTS
    assert (
        f"GPU cluster {EKS} unavailable: no node selected; {WANTS}; the deploy "
        "skips its node work"
    ) in lines
    assert PARKED.gating(site) is False, "only status gates"


# --- deploy: preflight gate, Agent inventory, snapshots ---------------------------


class PreflightSite(ParkedSite):
    def _load_state(self) -> dict[str, Any]:
        return {"release_id": "live", "phase": "complete"}


def _deploy_preflight(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    changed: list[str],
    *,
    only_the_parked_cluster: bool = False,
    desired: int = 0,
    instance_types: tuple[str, ...] = ("p5en.48xlarge",),
) -> dict[str, Any]:
    site = PreflightSite(desired=desired)
    site.provider.nodegroups[GROUP]["instanceTypes"] = list(instance_types)
    if only_the_parked_cluster:
        site.config.clusters = (site.eks,)
    proof = _cluster_section(site)["clusters"][EKS]["parked"]
    section: dict[str, Any] = {"parked": proof}
    if not proof["parked"]:
        section["zero_node_findings"] = [f"{EKS} has no Kubernetes nodes"]
    contexts = {
        "name": "regional_contexts",
        "status": "PASS",
        "summary": "",
        "details": {"providers": {EKS: section}},
    }
    monkeypatch.setattr(
        ADMIN, "probe_resource", lambda *_a, **_k: SimpleNamespace(exists=lambda: True)
    )
    for name in ("_require_expected_state", "_refuse_foreign_candidate_resume"):
        monkeypatch.setattr(ADMIN, name, lambda *_a, **_k: None)
    for name in ("_bootstrap_required", "_rollback_pending", "supersede_requested"):
        monkeypatch.setattr(ADMIN, name, lambda *_a, **_k: False)
    monkeypatch.setattr(
        ADMIN,
        "next_deploy",
        lambda *_a: {"action": "upgrade", "kind": kind, "changed": changed},
    )
    monkeypatch.setattr(
        ADMIN,
        "build_preflight_report",
        lambda _release, **_k: {
            "mode": "preflight",
            "healthy": True,
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [contexts],
        },
    )
    monkeypatch.setattr(
        ADMIN, "cpu_update_window", lambda _release: CheckValue("no CPU update window")
    )
    return ADMIN.build_deploy_preflight_report(site)


@pytest.mark.parametrize(
    "kind,changed",
    [
        ("CONTROL_PLANE_ONLY", ["cpu_ingress_manifests"]),
        ("DATA_PLANE_COMPATIBLE", ["executor_wheel"]),
        ("DATA_PLANE_COMPATIBLE", ["node_runtime_wheel", "node_bundle"]),
    ],
    ids=["control-plane-only", "executor", "agent-waves"],
)
@pytest.mark.parametrize("desired", [0, 2], ids=["parked", "unavailable"])
def test_every_release_kind_passes_preflight_beside_a_cluster_without_nodes(
    monkeypatch: pytest.MonkeyPatch, kind: str, changed: list[str], desired: int
) -> None:
    """Zero GPU nodes affect only their own cluster, parked or not."""

    report = _deploy_preflight(monkeypatch, kind, changed, desired=desired)

    assert report["healthy"] is True, report
    assert list(report["zero_node_clusters"]) == [EKS]
    assert report["zero_node_clusters"][EKS]["state"] == (
        "parked" if desired == 0 else "unavailable"
    )


@pytest.mark.parametrize(
    "kind,changed,refused",
    [
        ("DATA_PLANE_COMPATIBLE", ["node_runtime_wheel", "node_bundle"], True),
        # The Reconciler runs the executor wheel: an executor release rolls it.
        ("DATA_PLANE_COMPATIBLE", ["executor_wheel"], True),
        ("CONTROL_PLANE_ONLY", ["cpu_ingress_manifests"], False),
    ],
    ids=["agent-waves", "executor", "control-plane-only"],
)
def test_a_reconciler_roll_is_refused_over_a_cluster_it_cannot_render(
    monkeypatch: pytest.MonkeyPatch, kind: str, changed: list[str], refused: bool
) -> None:
    """No declared GPU instance type the installer knows: no node to render
    the template from. Refused before anything changes, and only when the
    release rolls the Reconciler or Agents."""

    report = _deploy_preflight(
        monkeypatch, kind, changed, desired=2, instance_types=("x9.huge",)
    )

    assert report["healthy"] is (not refused), report
    if refused:
        check = next(
            item
            for item in report["checks"]
            if item["name"] == "zero_node_gpu_clusters"
        )
        assert check["summary"].startswith(
            f"GPU cluster(s) {EKS} (declared: x9.huge) have zero nodes"
        ), check["summary"]
        assert check["summary"].endswith("Nothing was changed."), check["summary"]


def test_agent_waves_pass_the_preflight_when_no_cluster_has_a_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero GPU nodes anywhere block no deploy: an Agent-wave release over a
    site whose only GPU cluster is parked is not refused."""

    report = _deploy_preflight(
        monkeypatch,
        "DATA_PLANE_COMPATIBLE",
        ["node_runtime_wheel", "node_bundle"],
        only_the_parked_cluster=True,
    )

    assert report["healthy"] is True, report
    assert all(item["name"] != "zero_node_gpu_clusters" for item in report["checks"]), (
        "no zero-node refusal"
    )
    assert list(report["zero_node_clusters"]) == [EKS]


def test_the_cpu_roll_inventory_holds_a_parked_cluster_with_no_agents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CONTROL_PLANE_ONLY release that rolls the ingress waits for every
    cluster's live Agents; a parked cluster has none to wait for."""

    monkeypatch.setattr(
        FLEET, "_read_active_agent_node_sets", lambda _release: {HYPERPOD: ["node-a"]}
    )

    sets = FLEET.capture_active_agent_node_sets(ParkedSite())

    assert sets == {HYPERPOD: {"node_ids": ["node-a"]}, EKS: {"node_ids": []}}
    err = capsys.readouterr().err
    assert "zero-node-gpu-cluster" in err and "step=active_agent_node_sets" in err, err
    assert 'recorded="no live Agent to wait for"' in err, "a quoted field"
    unavailable = FLEET.capture_active_agent_node_sets(ParkedSite(desired=2))
    assert unavailable[EKS] == {"node_ids": []}, "no node: nothing to wait for"
    assert "state=unavailable" in capsys.readouterr().err
    populated = ParkedSite()
    populated.documents[(EKS, "nodes", "-l " + OWNERSHIP)] = {
        "items": [node("ip-10-0-1-10", {"gpu-fault.io/control-plane": SITE})]
    }
    with pytest.raises(ReleaseError, match=f"missing clusters: {EKS}$"):
        FLEET.capture_active_agent_node_sets(populated)


@pytest.fixture
def identity_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The identity probe answers the site's ``records``; no lease re-read."""

    monkeypatch.setattr(AGENT_SET, "SLEEP", lambda _seconds: None)
    monkeypatch.setattr(AGENT_SET, "_lease_facts", lambda *_a: {})
    monkeypatch.setattr(
        STATE,
        "exec_cpu_ingress_probe",
        lambda release, **_k: json.dumps(release.records),
    )


def test_the_agent_snapshot_records_a_parked_cluster_as_an_explicit_entry(
    identity_probe: None, capsys: pytest.CaptureFixture[str]
) -> None:
    identities = AGENT_SET.capture_agent_identities_matching_nodes(ParkedSite())

    assert identities[HYPERPOD]["node_ids"] == ["node-a"]
    entry = identities[EKS]
    assert PARKED.is_zero_node_identity(entry), entry
    assert entry["zero_nodes"]["state"] == "parked"
    assert entry["zero_nodes"]["groups"][0]["name"] == GROUP
    err = capsys.readouterr().err
    assert "zero-node-gpu-cluster" in err and f"cluster={EKS}" in err, err


def test_the_agent_snapshot_records_an_unavailable_cluster_as_an_explicit_entry(
    identity_probe: None,
) -> None:
    identities = AGENT_SET.capture_agent_identities_matching_nodes(
        ParkedSite(desired=2)
    )

    entry = identities[EKS]
    assert PARKED.is_zero_node_identity(entry), entry
    assert entry["zero_nodes"]["state"] == "unavailable"
    assert entry["zero_nodes"]["reason"] == WANTS


def test_a_cluster_with_a_node_but_no_agent_still_refuses_the_snapshot(
    identity_probe: None,
) -> None:
    site = ParkedSite(desired=2)
    site.documents[(EKS, "nodes", "-l " + OWNERSHIP)] = {
        "items": [node("ip-10-0-1-10", {"gpu-fault.io/control-plane": SITE})]
    }

    with pytest.raises(
        ReleaseError, match=f"{EKS} has no active Agent identity to capture"
    ):
        AGENT_SET.capture_agent_identities_matching_nodes(site)


@pytest.mark.parametrize("labelled", [True, False], ids=["parked", "unavailable"])
def test_live_agents_on_nodes_the_selector_dropped_still_gate(
    identity_probe: None, monkeypatch: pytest.MonkeyPatch, labelled: bool
) -> None:
    """A misconfiguration: the selector selects nothing while the Agents of
    the nodes it no longer selects are live. Whatever the provider says, the
    snapshot refuses and the verify's coverage drift gates."""

    site = ParkedSite(records=[agent(HYPERPOD, "node-a"), agent(EKS, "ip-10-0-1-10")])
    site.provider.nodegroups[GROUP] = nodegroup(labelled=labelled)
    with pytest.raises(ReleaseError):
        AGENT_SET.capture_agent_identities_matching_nodes(site)
    report = _cluster_section(site)
    section = report["clusters"][EKS]
    assert section["parked"]["parked"] is labelled, section["parked"]
    section["agents"] = [{"node_id": "ip-10-0-1-10", "lifecycle_state": "ACTIVE"}]
    with pytest.raises(ReleaseError, match=f"{EKS} Agent coverage drift"):
        judge_cluster_fleet(site, EKS, section)  # a verify: findings never gate
    assert "zero_node_findings" not in section


def test_a_release_snapshot_carries_the_parked_entry(
    identity_probe: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = ParkedSite()
    site.state = {
        "release_id": "live",
        "runtime_image": NEW_IMAGE,
        "admin_config": default_admin_config().as_dict(),
    }
    site.node_installer_image = OLD_IMAGE
    site.adot_image = OLD_IMAGE
    for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS:
        site.documents[("cpu", "deployment", name)] = deployment(name)
    site.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {"data": {}}
    monkeypatch.setattr(STATE, "remote_command_stats", lambda _release: {})
    monkeypatch.setattr(
        STATE,
        "capture_gpu_cluster_snapshot",
        lambda _release, target, **_k: ({}, {target.cluster_id: NEW_IMAGE}, OLD_IMAGE),
    )
    monkeypatch.setattr(
        STATE, "capture_previous_monitoring", lambda *_a, **_k: ({}, OLD_IMAGE)
    )

    previous = STATE.capture_previous(
        site, ReleaseExecutionPlan(nodes=(ReleaseComponent.AGENT,))
    )

    assert PARKED.is_zero_node_identity(previous["agent_identities"][EKS]), previous
    assert previous["agent_identities"][HYPERPOD]["node_ids"] == ["node-a"]


# --- the failure path of a CONTROL_PLANE_ONLY release ----------------------------


@pytest.mark.parametrize(
    "desired,with_node,status",
    [(0, False, "PASSED"), (2, False, "PASSED"), (2, True, "FAILED")],
    ids=["parked", "unavailable", "node-without-agent"],
)
def test_a_rollback_over_a_cluster_without_nodes_completes_its_sync_state(
    identity_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    desired: int,
    with_node: bool,
    status: str,
) -> None:
    """A CONTROL_PLANE_ONLY release whose verify failed rolls back, then the
    driver aligns the engine state with an unscoped ``sync-state``. Before,
    that refused (``has no active Agent identity to capture``) and the
    rollback record ended FAILED; a cluster without nodes now passes, parked
    or not, and a node without an Agent still fails it."""

    site = ParkedSite(desired=desired)
    if with_node:
        site.documents[(EKS, "nodes", "-l " + OWNERSHIP)] = {
            "items": [node("ip-10-0-1-10", {"gpu-fault.io/control-plane": SITE})]
        }
    monkeypatch.setattr(
        ROLLOUT, "sync_state_dcgm", lambda release: (dict(release.state), {})
    )
    monkeypatch.setattr(
        release_failure_recovery,
        "reconcile_rollback_management",
        lambda *_args, **_kwargs: tmp_path,
    )
    modes: list[str] = []

    def run_release_mode(_site: Any, *, mode: str, **_kwargs: Any) -> None:
        modes.append(mode)
        if mode == "sync-state":
            ROLLOUT.sync_release_state(site)

    result = release_failure_recovery.rollback_after_failure(
        SimpleNamespace(release_id="release-b", state_dir=tmp_path),
        site_file=tmp_path / "site.yaml",
        root=tmp_path,
        environment={},
        failure_error="release verification failed",
        failed_at="2026-01-01T00:00:00+00:00",
        run_release_mode=run_release_mode,
        read_live_state=lambda _site: {"phase": "rolled-back"},
        update_phase=lambda *_args, **_kwargs: None,
    )

    assert modes == ["rollback", "sync-state"]
    assert result["status"] == status, result
    if status == "PASSED":
        assert [phase for phase, _values in site.saved] == ["complete"]
    else:
        assert "has no active Agent identity to capture" in result["error"]


def test_the_deploy_record_names_the_parked_clusters(tmp_path: Any) -> None:
    report_path = tmp_path / "verify.json"
    report_path.write_text("{}", encoding="utf-8")

    entry = {"state": "parked", "reason": "r", "groups": [], "provider": "eks-managed"}
    metadata = release_deploy_evidence.verification_metadata(
        report_path, {"summary": {"FAIL": 0}, "zero_node_clusters": {EKS: entry}}
    )
    stability = release_deploy_evidence.verification_metadata(
        report_path, {"summary": {"FAIL": 0}}
    )

    assert metadata["zero_node_clusters"] == {EKS: entry}
    assert "zero_node_clusters" not in stability, "only a report that says so"


def test_the_source_deploy_success_record_reads_the_zero_node_clusters(
    tmp_path: Any,
) -> None:
    """``source-deploy-success.json`` carries what the release record's verify
    named, outside its ``live`` evidence (which the next run compares)."""

    from scripts import staging_deploy

    entry = {"state": "parked", "reason": "r", "groups": [], "provider": "eks-managed"}
    record = tmp_path / "release-deploy" / "release-b" / "state.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps({"verification": {"zero_node_clusters": {EKS: entry}}}),
        encoding="utf-8",
    )

    assert staging_deploy.release_record_zero_node_clusters(tmp_path) == {EKS: entry}
    assert staging_deploy.release_record_zero_node_clusters(tmp_path / "none") is None
