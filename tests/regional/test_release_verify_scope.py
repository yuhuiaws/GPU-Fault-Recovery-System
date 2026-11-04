"""A verify scoped to a membership verb's target, and the verify's departure wait.

Live 2026-11-04: while a HyperPod cluster joined, an EC2 Spot interruption took
a node of the plain EKS GPU cluster. Five minutes later the join's ``verify``
still saw that Node listed (NotReady) and its Agent ACTIVE, ``control_api``
failed on ``gpu-eks-cluster Agent coverage drift`` and the HyperPod join was
rolled back. A verify scoped to the joining cluster reports the other
cluster's own findings and gates on the shared control plane and the target
only; wherever coverage still gates (the target, every cluster in ``deploy``),
an Agent whose Node is going is re-judged for the departure guard and excused
once proven gone.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_checks_readiness as READINESS
from gpu_fault_release import regional_release_preflight as PREFLIGHT
from gpu_fault_release import regional_release_runtime_identity as IDENTITY
from gpu_fault_release import regional_release_verify_churn as CHURN
from gpu_fault_release import regional_validation_evidence as EVIDENCE
from gpu_fault_release import regional_verify_scope as SCOPE
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_regional_release_automatic_updates import EKS as EKS_TARGET
from tests.regional.test_regional_release_automatic_updates import HYPERPOD as HP_TARGET
from tests.regional.test_regional_release_automatic_updates import (
    ContextsRelease,
    ContextsRunner,
)
from tests.regional.test_release_churn_verify import OTHER_CHECKS
from tests.regional.test_release_verify_departure_wait import (
    EKS,
    EKS_GONE,
    EKS_NODE,
    HP,
    HP_GOING,
    HP_NODE,
    ControlPlane,
    by_name,
    departure_clock,
    health,
    no_wait,
    node,
    proves_gone,
    two_cluster_release,
    unprovable,
    verified,
)
from tests.regional.test_runtime_component_identity import Release as IdentityRelease
from tests.regional.test_runtime_component_identity import Runner as IdentityRunner


def scoped(release: Any, verb: str = SCOPE.JOIN_VERB, *targets: str) -> Any:
    release.verify_scope = SCOPE.VerifyScope(verb, targets or (HP,))
    return release


# -- the live shape: another cluster loses a node while one joins --------------------


@pytest.mark.parametrize(
    "eks_nodes",
    [[node(EKS_NODE), node(EKS_GONE, ready=False)], [node(EKS_NODE)]],
    ids=["still-listed-notready", "no-longer-listed"],
)
def test_a_join_verify_reports_another_clusters_lost_node_without_failing(
    monkeypatch: pytest.MonkeyPatch, eks_nodes: list[dict[str, Any]]
) -> None:
    """Before: ``gpu-eks-cluster Agent coverage drift`` FAILED the join's verify
    and the HyperPod join was rolled back."""

    asked = unprovable(monkeypatch)
    no_wait(monkeypatch)
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})
    nodes = {EKS: eks_nodes, HP: [node(HP_NODE)]}

    report = health(
        monkeypatch, scoped(two_cluster_release()), nodes=nodes, plane=plane
    )

    record = by_name(report)["control_api"]
    assert report["healthy"] is True, record
    assert record["status"] == "WARN", record
    assert "not gating this join-cluster hp-cluster-join" in record["summary"], record
    assert f"{EKS}: {EKS} Agent coverage drift" in record["summary"], record
    assert list(record["details"]["not_gating"]) == [EKS]
    assert report["cluster_scope"]["target_cluster_ids"] == [HP]
    findings = report["cluster_scope"]["not_gating_findings"]
    assert [(item["check"], item["cluster_id"]) for item in findings] == [
        ("control_api", EKS)
    ]
    assert findings[0]["finding"].startswith(f"{EKS} Agent coverage drift"), findings
    assert all(item == [EKS_GONE] for item in asked), "only the Agent without a Node"


def test_the_same_verify_without_a_scope_still_fails_after_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ad048f18's verify and a deploy's: every cluster gates, and an Agent whose
    Node stays listed is refused once the departure guard has passed."""

    unprovable(monkeypatch)
    still = [node(EKS_NODE), node(EKS_GONE, ready=False)]
    clock = departure_clock(monkeypatch, [still])
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})

    report = health(
        monkeypatch,
        two_cluster_release(),
        nodes={EKS: still, HP: [node(HP_NODE)]},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert report["healthy"] is False
    assert record["status"] == "FAIL"
    assert record["summary"].startswith(f"{EKS} Agent coverage drift"), record
    assert (
        f"departure not proven: {EKS_GONE}: the Node is still in Kubernetes"
        in (record["summary"])
    ), record
    assert clock.now >= CHURN.DEPARTURE_DECISION_SECONDS
    assert "cluster_scope" not in report


def test_the_target_clusters_own_drift_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Ready Node of the joining cluster without an Agent is never excused."""

    no_wait(monkeypatch)
    extra = "hyperpod-i-0000000000b9"
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE]})

    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE), node(extra)]},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert record["status"] == "FAIL", record
    assert record["summary"].startswith(f"{HP} Agent coverage drift"), record
    assert report["healthy"] is False


@pytest.mark.parametrize("breakage", ["registry", "healthz", "remote-commands"])
def test_shared_control_plane_findings_still_gate_a_scoped_verify(
    monkeypatch: pytest.MonkeyPatch, breakage: str
) -> None:
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE]})
    original = plane.__call__

    def broken(release: Any, *, arguments: tuple[str, ...], **kw: Any) -> str:
        report = json.loads(original(release, arguments=arguments, **kw))
        if breakage == "registry":
            report["registry"] = [{"cluster_id": EKS}]
        elif breakage == "healthz":
            report["healthz"] = {"status": "degraded"}
        else:
            report["remote_commands"]["oldest_unclaimed_age_seconds"] = 3600
        return json.dumps(report)

    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=broken,  # type: ignore[arg-type]
    )

    assert by_name(report)["control_api"]["status"] == "FAIL"
    assert report["healthy"] is False


def test_a_removal_reports_every_remaining_clusters_own_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remove-cluster targets the cluster it removed: the remaining clusters'
    own findings are reported, the shared control plane gates."""

    unprovable(monkeypatch)
    no_wait(monkeypatch)
    release = scoped(two_cluster_release(), SCOPE.REMOVE_VERB, "gpu-removed")
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE, HP_GOING]})

    report = health(
        monkeypatch,
        release,
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=plane,
    )

    assert report["healthy"] is True
    findings = report["cluster_scope"]["not_gating_findings"]
    assert sorted(item["cluster_id"] for item in findings) == [EKS, HP]
    assert report["cluster_scope"]["verb"] == "remove-cluster"


# -- each cluster's own checks beside control_api -------------------------------------


def test_another_clusters_gpu_cluster_check_warns_and_the_targets_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gpu_cluster(_release: Any, target: Any, **_kw: Any) -> CheckValue:
        raise ReleaseError(f"{target.cluster_id} has no Ready HyperPod nodes")

    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE]})
    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=plane,
        gpu_cluster=gpu_cluster,
    )

    checks = by_name(report)
    other, target = checks[f"gpu_cluster:{EKS}"], checks[f"gpu_cluster:{HP}"]
    assert other["status"] == "WARN", other
    assert other["details"]["not_gating"] == {
        EKS: [f"{EKS} has no Ready HyperPod nodes"]
    }
    assert (target["status"], target["summary"]) == (
        "FAIL",
        f"{HP} has no Ready HyperPod nodes",
    )


def test_another_clusters_collectors_are_neither_awaited_nor_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_seconds: float) -> None:
        raise AssertionError("the verify waited for another cluster's collectors")

    monkeypatch.setattr(READINESS.time, "sleep", refuse)
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE]}, stale=frozenset({EKS}))

    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert record["status"] == "WARN", record
    assert record["details"]["not_gating"][EKS] == [
        f"collector readiness: {EKS}: not ready"
    ]
    assert record["details"]["collector_readiness"]["not_gating"] == {
        EKS: [f"{EKS}: not ready"]
    }


def test_the_targets_collectors_still_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(READINESS, "_capped_grace", lambda _grace: 0.0)
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE]}, stale=frozenset({HP}))

    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert record["status"] == "FAIL", record
    assert record["summary"].startswith(f"{HP} collector readiness failed"), record


def test_another_clusters_executor_verifier_is_a_finding() -> None:
    class Runner:
        def run(self, _arguments: Any, *, env: Any = None, **_kw: Any) -> str:
            context = (env or {}).get("GPU_FAULT_KUBE_CONTEXT")
            if context == "ctx-eks":
                raise ReleaseError("executor check failed: no Running pod")
            return "executor check passed" if context else "role split passed"

    release = SimpleNamespace(
        runner=Runner(),
        runtime_image="runtime:candidate",
        executor_wheel_cm="gpu-fault-executor-wheel-abc",
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            cpu_kubeconfig="/tmp/cpu.kubeconfig",
            clusters=[
                SimpleNamespace(cluster_id=EKS, context="ctx-eks"),
                SimpleNamespace(cluster_id=HP, context="ctx-hp"),
            ],
        ),
    )

    scoped(release)
    details = EVIDENCE.read_only_verifier_details(release)
    assert details["not_gating"] == {
        EKS: ["executor verifier: executor check failed: no Running pod"]
    }
    assert details["clusters"] == {HP: "executor check passed"}

    scoped(release, SCOPE.JOIN_VERB, EKS)
    with pytest.raises(ReleaseError, match="no Running pod"):
        EVIDENCE.read_only_verifier_details(release)


def test_another_clusters_runtime_identity_is_a_finding_the_cpus_never() -> None:
    class Drifted(IdentityRunner):
        def run(self, arguments: list[str], **kwargs: Any) -> str:
            if "ctx-eks" in arguments and f"PATH={IDENTITY.EXECUTOR_PATH}" in arguments:
                return "f" * 64
            return super().run(arguments, **kwargs)

    release = IdentityRelease(runner=Drifted())
    release.config.clusters = (
        SimpleNamespace(cluster_id=EKS, context="ctx-eks"),
        SimpleNamespace(cluster_id=HP, context="ctx-hp"),
    )

    scoped(release)
    result = IDENTITY.validate_runtime_component_identity(release)
    assert list(result["not_gating"]) == [EKS]
    assert len(result["not_gating"][EKS]) == 4, "one per Deployment of the cluster"
    assert all("module digest" in item for item in result["not_gating"][EKS]), result

    scoped(release, SCOPE.JOIN_VERB, EKS)
    with pytest.raises(ReleaseError, match="runtime component identity"):
        IDENTITY.validate_runtime_component_identity(release)


def test_another_clusters_context_is_a_finding() -> None:
    class Release(ContextsRelease):
        def _validate_executor_iam_role(self, target: Any) -> None:
            super()._validate_executor_iam_role(target)
            if target.cluster_id == EKS_TARGET.cluster_id:
                raise ReleaseError(f"{target.cluster_id} executor role is missing")

    release = Release(ContextsRunner(), (HP_TARGET, EKS_TARGET))
    release.verify_scope = SCOPE.VerifyScope(SCOPE.JOIN_VERB, (HP_TARGET.cluster_id,))

    facts = PREFLIGHT.ensure_region_contexts(release)  # type: ignore[arg-type]
    assert facts["not_gating"] == {
        EKS_TARGET.cluster_id: [f"{EKS_TARGET.cluster_id} executor role is missing"]
    }

    release.verify_scope = None
    with pytest.raises(ReleaseError, match="executor role is missing"):
        PREFLIGHT.ensure_region_contexts(release)  # type: ignore[arg-type]


# -- the departure wait (test_release_verify_departure_wait) ----------------------------


def test_a_scoped_verify_does_not_wait_for_another_clusters_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    proves_gone(monkeypatch)
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})

    report = health(
        monkeypatch,
        scoped(two_cluster_release()),
        nodes={EKS: [node(EKS_NODE), node(EKS_GONE, ready=False)], HP: [node(HP_NODE)]},
        plane=plane,
    )

    assert report["healthy"] is True
    churn = by_name(report)["control_api"]["details"]["clusters"][EKS]["release_churn"]
    assert churn["departure_unproven"] == {
        EKS_GONE: CHURN.PRESENT_REASON.format(state="not Ready")
    }


def test_status_never_waits_and_still_reports_the_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})
    for name in OTHER_CHECKS:
        monkeypatch.setattr(CHECKS, name, lambda *_a, n=name, **_k: CheckValue(n))
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, target: (
            "selector",
            {
                "items": [node(EKS_NODE), node(EKS_GONE, ready=False)]
                if target.cluster_id == EKS
                else [node(HP_NODE)]
            },
        ),
    )
    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", plane)

    report = CHECKS.build_quick_health_report(two_cluster_release())

    record = by_name(report)["control_api"]

    assert record["status"] == "FAIL"
    assert record["summary"].startswith(f"{EKS} Agent coverage drift"), record


# -- the engine command line ---------------------------------------------------------------


def arguments(*words: str) -> Any:
    parsed = ROLLOUT.parse_arguments(["verify", "--config", "site.json", *words])
    ROLLOUT.validate_cluster_arguments(parsed)
    return parsed


def test_the_scope_flags_name_the_verb_and_its_targets() -> None:
    release = two_cluster_release()

    joined = SCOPE.scope_of(
        arguments("--joining-cluster-id", HP, "--joining-cluster-id", EKS), release
    )
    removed = SCOPE.scope_of(arguments("--removed-cluster-id", "gpu-gone"), release)

    assert joined == SCOPE.VerifyScope("join-cluster", (EKS, HP))
    assert removed == SCOPE.VerifyScope("remove-cluster", ("gpu-gone",))
    assert SCOPE.scope_of(arguments(), SimpleNamespace()) is None, "reads nothing"


@pytest.mark.parametrize(
    "words,problem",
    [
        (("--joining-cluster-id", "gpu-unknown"), "names clusters the site does not"),
        (("--removed-cluster-id", EKS), "is still in the site"),
        (
            ("--joining-cluster-id", HP, "--removed-cluster-id", "gpu-gone"),
            "exclude each other",
        ),
        (("--joining-cluster-id", " "), "must be non-empty"),
        (("--removed-cluster-id", ""), "must be non-empty"),
    ],
)
def test_a_wrong_scope_is_refused_not_verified_leniently(
    words: tuple[str, ...], problem: str
) -> None:
    with pytest.raises(ReleaseError, match=problem):
        SCOPE.scope_of(arguments(*words), two_cluster_release())


@pytest.mark.parametrize(
    "words",
    [("status",), ("join-cluster", "--cluster-id", HP), ("deploy",)],
    ids=["status", "join-cluster-mode", "deploy"],
)
def test_the_scope_flags_scope_verify_preflight_and_sync_state_only(
    words: tuple[str, ...],
) -> None:
    parsed = ROLLOUT.parse_arguments(
        [words[0], "--config", "site.json", *words[1:], "--joining-cluster-id", HP]
    )
    with pytest.raises(ReleaseError, match="only scope verify, preflight, sync-state"):
        ROLLOUT.validate_cluster_arguments(parsed)


def test_the_engine_verify_with_a_scope_prints_a_healthy_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unprovable(monkeypatch)
    no_wait(monkeypatch)
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=plane,
    )

    assert code == 0, report
    assert SCOPE.not_gating_findings(report)[0]["cluster_id"] == EKS
    assert report["cluster_scope"]["verb"] == "join-cluster"


def test_a_check_names_three_findings_then_counts_the_rest() -> None:
    release = scoped(two_cluster_release())
    details: dict[str, Any] = {}
    for index in range(5):
        SCOPE.record_not_gating(details, EKS, f"finding {index}")

    (check,) = SCOPE.mark_not_gating(
        release,
        [{"name": "x", "status": "PASS", "summary": "fine", "details": details}],
    )

    assert check["status"] == "WARN"
    assert check["summary"].endswith(
        f"{EKS}: finding 0; {EKS}: finding 1; {EKS}: finding 2; "
        "+2 more in details.not_gating"
    ), check["summary"]
    assert SCOPE.mark_not_gating(SimpleNamespace(), [check]) == [check]
