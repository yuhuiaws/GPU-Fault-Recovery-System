"""A node lost on one GPU cluster blocks no step of another cluster's join.

b250 scoped the join's final verify. Before it, the join ran region-wide gates
on every cluster: the candidate ``preflight`` (``regional_contexts`` validated
every cluster's context -- an EKS node group DEGRADED because Spot capacity
could not replace the lost node, a cluster left without a Kubernetes node --
and ``workflow_safety`` counted every cluster's destructive workflows,
including the record the product opens for a node that left), and the engine's
``join-cluster`` mode (``join_target`` refused while any remote command was
open anywhere -- the departed node's Agent never claims its commands --, then
validated every context, then every fleet wave gate refused on any cluster's
open commands and destructive workflows). A refusal there failed the join
before it started or, once it had started, rolled it back.

Now the same scope covers those calls: the joining cluster's own findings and
the shared control plane still gate, another site cluster's are findings
(``WARN`` in a report, a ``not-gating`` line in a mode without one); counts
that cannot be attributed, and every unscoped call, still gate on the fleet.
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_gpu_rollout as GPU_ROLLOUT
from gpu_fault_release import regional_release_preflight as PREFLIGHT
from gpu_fault_release import regional_release_store_preflight as STORE
from gpu_fault_release import regional_release_workflow_safety as WORKFLOWS
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_regional_release_automatic_updates import EKS as EKS_TARGET
from tests.regional.test_regional_release_automatic_updates import HYPERPOD as HP_TARGET
from tests.regional.test_regional_release_automatic_updates import (
    ContextsRelease,
    ContextsRunner,
)

JOINING = HP_TARGET.cluster_id  # gpu-hp
OTHER = EKS_TARGET.cluster_id  # gpu-eks
DEGRADED = f"EKS node group gpu-ng of {EKS_TARGET.eks_cluster_arn.rsplit('/', 1)[-1]}"
OPEN = ("PENDING", "LEASED", "WAITING")
#: Every preflight check but the two a node loss elsewhere could trip.
PASSING = (
    "_check_tools",
    "_check_local_inputs",
    "check_artifact_config_maps",
    "_check_aws_identity",
    "_check_cpu_capacity",
    "check_cpu_secrets",
    "_check_load_balancer_controller",
    "_check_nlb_inputs",
    "_check_aurora",
    "check_aurora_refresh",
    "check_control_record_archive_bucket",
    "check_diagnostic_evidence_destination",
    "check_email_notifications",
    "_check_monitoring",
)


class SpotLossRunner(ContextsRunner):
    """The contexts after a Spot reclaim on the EKS GPU cluster: its GPU node
    group DEGRADED (no capacity to replace the node) or no node left at all;
    ``joining_broken`` breaks the joining HyperPod cluster's own context."""

    def __init__(self, *, lost: str = "degraded", joining_broken: bool = False):
        super().__init__()
        self.lost = lost
        self.joining_broken = joining_broken

    def run(self, args: list[str], **options: Any) -> str:
        if (
            self.lost == "no-nodes"
            and args[0] == "kubectl"
            and "nodes" in args
            and EKS_TARGET.context in args
        ):
            return json.dumps({"items": []})
        output = super().run(args, **options)
        if args[:3] == ["aws", "eks", "describe-nodegroup"] and "gpu-ng" in args:
            if self.lost == "degraded":
                value = json.loads(output)
                value["status"] = "DEGRADED"
                return json.dumps(value)
        if (
            self.joining_broken
            and args[:3] == ["aws", "sagemaker", "describe-cluster"]
            and "hp-gpu" in args
        ):
            value = json.loads(output)
            value["NodeRecovery"] = "Automatic"
            return json.dumps(value)
        return output


class Site(ContextsRelease):
    """A two-cluster site the engine's real context validation reads."""

    _ensure_contexts = PREFLIGHT.ensure_region_contexts

    def __init__(self, runner: Any) -> None:
        super().__init__(runner, (HP_TARGET, EKS_TARGET))

    def _target(self, cluster_id: str) -> Any:
        return next(
            target for target in self.config.clusters if target.cluster_id == cluster_id
        )


def blockers(**by_cluster: list[str]) -> dict[str, Any]:
    """The ``workflow_safety`` probe's answer with these blockers per cluster."""

    ids = [item for names in by_cluster.values() for item in names]
    return {
        "blockers": ids,
        "blocker_count": len(ids),
        "blocker_count_by_cluster": {key: len(v) for key, v in by_cluster.items()},
        "blockers_by_cluster": by_cluster,
    }


def preflight(
    monkeypatch: pytest.MonkeyPatch,
    release: Any,
    *scope: str,
    workflows: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """The engine's ``preflight`` mode as the join's driver runs it."""

    for name in PASSING:
        monkeypatch.setattr(CHECKS, name, lambda *_a, name=name, **_k: CheckValue(name))
    monkeypatch.setattr(
        CHECKS.rotation_safety, "token_rotation_check", lambda _r: ("no rotation", {})
    )
    monkeypatch.setattr(
        WORKFLOWS,
        "exec_cpu_ingress_probe",
        lambda *_a, **_k: json.dumps(workflows or blockers()),
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = ROLLOUT._dispatch_mode(
            release,
            ROLLOUT.parse_arguments(
                ["preflight", "--config", "site.json", "--full", *scope]
            ),
        )
    return code, json.loads(output.getvalue())


def by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in report["checks"]}


# -- the join's candidate preflight ----------------------------------------------------


@pytest.mark.parametrize(
    "lost,finding",
    [
        ("degraded", f"{DEGRADED} is not ACTIVE (DEGRADED)"),
        (
            "no-nodes",
            f"{OTHER} has no Kubernetes nodes; not parked: node group gpu-ng "
            "reports no scalingConfig.desiredSize",
        ),
    ],
)
def test_another_clusters_lost_node_is_a_finding_of_the_join_preflight(
    monkeypatch: pytest.MonkeyPatch, lost: str, finding: str
) -> None:
    """Before: ``regional_contexts`` FAILED the HyperPod join's preflight (and
    ``--joining-cluster-id`` was refused outside a verify)."""

    code, report = preflight(
        monkeypatch,
        Site(SpotLossRunner(lost=lost)),
        "--joining-cluster-id",
        JOINING,
        workflows=blockers(**{OTHER: ["workflow-departed-node"]}),
    )

    checks = by_name(report)
    assert code == 0, report
    assert report["healthy"] is True
    assert checks["regional_contexts"]["status"] == "WARN", checks["regional_contexts"]
    assert checks["regional_contexts"]["details"]["not_gating"] == {OTHER: [finding]}
    assert checks["workflow_safety"]["status"] == "WARN", checks["workflow_safety"]
    assert checks["workflow_safety"]["details"]["not_gating"] == {
        OTHER: ["1 active destructive workflow(s): workflow-departed-node"]
    }
    assert sorted(
        (item["check"], item["cluster_id"])
        for item in report["cluster_scope"]["not_gating_findings"]
    ) == [("regional_contexts", OTHER), ("workflow_safety", OTHER)]
    assert report["cluster_scope"]["target_cluster_ids"] == [JOINING]


def test_the_joining_clusters_own_context_still_fails_the_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, report = preflight(
        monkeypatch,
        Site(SpotLossRunner(joining_broken=True)),
        "--joining-cluster-id",
        JOINING,
    )

    record = by_name(report)["regional_contexts"]
    assert code == 1
    assert record["status"] == "FAIL"
    assert record["summary"] == "HyperPod cluster hp-gpu must set NodeRecovery=None"


@pytest.mark.parametrize(
    "workflows,problem",
    [
        (blockers(**{JOINING: ["workflow-rejoin"]}), "the joining cluster's"),
        (blockers(**{"": ["workflow-unreadable"]}), "unattributed"),
        (blockers(**{"gpu-removed": ["workflow-orphan"]}), "an unmanaged cluster's"),
        (
            {"blockers": ["workflow-x"], "blocker_count": 1},
            "an older probe without the breakdown",
        ),
        (
            {**blockers(**{OTHER: ["workflow-x"]}), "blocker_count": 2},
            "a breakdown short of the count",
        ),
    ],
    ids=["target", "unattributed", "unmanaged", "no-breakdown", "short"],
)
def test_a_blocker_the_scope_cannot_set_aside_still_fails_the_preflight(
    monkeypatch: pytest.MonkeyPatch, workflows: dict[str, Any], problem: str
) -> None:
    code, report = preflight(
        monkeypatch,
        Site(SpotLossRunner(lost="none")),
        "--joining-cluster-id",
        JOINING,
        workflows=workflows,
    )

    record = by_name(report)["workflow_safety"]
    assert code == 1, problem
    assert record["status"] == "FAIL", (problem, record)
    assert record["summary"].startswith("active destructive workflows block release"), (
        record
    )


def test_an_unscoped_preflight_still_judges_every_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deploy's (and any operator's) preflight: nothing is set aside."""

    code, report = preflight(
        monkeypatch,
        Site(SpotLossRunner()),
        workflows=blockers(**{OTHER: ["workflow-departed-node"]}),
    )

    checks = by_name(report)
    assert code == 1
    assert checks["regional_contexts"]["status"] == "FAIL"
    assert checks["regional_contexts"]["summary"].startswith(DEGRADED), checks
    assert checks["workflow_safety"]["status"] == "FAIL"
    assert "cluster_scope" not in report


@pytest.mark.parametrize(
    "words,problem",
    [
        (("preflight", "--for-deploy"), "never scope a deploy"),
        (("join-cluster", "--cluster-id", JOINING), "only scope verify"),
    ],
)
def test_a_scope_flag_is_refused_where_it_does_not_belong(
    words: tuple[str, ...], problem: str
) -> None:
    arguments = ROLLOUT.parse_arguments(
        [words[0], "--config", "site.json", *words[1:], "--joining-cluster-id", JOINING]
    )
    with pytest.raises(ReleaseError, match=problem):
        ROLLOUT.validate_cluster_arguments(arguments)


# -- the engine's join-cluster mode ----------------------------------------------------


class JoiningSite(Site):
    """The steps of ``RegionalRelease.join_cluster`` a node loss elsewhere can
    trip, run for real: ``join_target`` (the idle check), the contexts and the
    pre-mutation wave gate on the joining cluster."""

    _remote_commands_are_idle = STORE.remote_commands_are_idle

    def join_cluster(self, cluster_id: str) -> None:
        target = GPU_ROLLOUT.join_target(self, cluster_id)
        self.contexts = self._ensure_contexts()
        self.wave = FLEET.ensure_rollout_wave_safe(
            self, target, wave=("hp-node",), node_names=("hp-node",)
        )


def stats(**open_by_cluster: dict[str, int]) -> dict[str, Any]:
    """``remote_command_stats()`` with these open commands per cluster."""

    by_status = {status: 0 for status in OPEN}
    for cells in open_by_cluster.values():
        for status, count in cells.items():
            by_status[status] += count
    return {"by_status": by_status, "by_cluster_status": open_by_cluster}


def wave_snapshot(
    *, open_commands: dict[str, dict[str, int]], destructive: dict[str, list[str]]
) -> dict[str, Any]:
    """The fleet wave probe's answer: no Agent blockers, these per cluster."""

    totals = stats(**open_commands)["by_status"]
    ids = [item for names in destructive.values() for item in names]
    return {
        "open_remote": totals,
        "open_remote_by_cluster": open_commands,
        "destructive_workflow_count": len(ids),
        "destructive_workflows": ids,
        "destructive_workflow_count_by_cluster": {
            key: len(names) for key, names in destructive.items()
        },
        "destructive_workflows_by_cluster": destructive,
        "never_dispatched_destructive_workflows": [],
        "never_dispatched_young_destructive_workflows": [
            {
                "request_id": item,
                "cluster_id": key,
                "node_ids": ["ip-10-0-38-176"],
                "created_at": "2026-11-04T12:16:00+00:00",
            }
            for key, names in destructive.items()
            for item in names
        ],
        "agent_blocker_count": 0,
        "agent_blockers": [],
        "minimum_lease_remaining_seconds": 90.0,
    }


def join_mode(
    monkeypatch: pytest.MonkeyPatch,
    release: JoiningSite,
    *,
    idle: dict[str, Any],
    wave: dict[str, Any],
) -> int:
    monkeypatch.setattr(STORE, "cpu_ingress_pod_if_running", lambda _r: "api-ha-0")
    monkeypatch.setattr(
        STORE, "exec_cpu_ingress_probe", lambda *_a, **_k: json.dumps(idle)
    )
    monkeypatch.setattr(FLEET, "rollout_wave_safety_snapshot", lambda *_a, **_k: wave)
    return ROLLOUT._dispatch_mode(
        release,
        ROLLOUT.parse_arguments(
            ["join-cluster", "--config", "site.json", "--cluster-id", JOINING]
        ),
    )


def spot_loss_elsewhere() -> dict[str, Any]:
    return {
        "idle": stats(**{OTHER: {"PENDING": 2}}),
        "wave": wave_snapshot(
            open_commands={OTHER: {"PENDING": 2}},
            destructive={OTHER: ["workflow-departed-node"]},
        ),
    }


def test_another_clusters_spot_loss_does_not_stop_the_join_cluster_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: ``remote commands are PENDING/LEASED/WAITING`` -- the departed
    node's Agent never claims its commands -- refused the mode at once (then
    the contexts, then ``fleet wave safety blocked by ...``), and the join was
    rolled back."""

    release = JoiningSite(SpotLossRunner())

    code = join_mode(monkeypatch, release, **spot_loss_elsewhere())

    assert code == 0
    assert release.contexts["not_gating"] == {
        OTHER: [f"{DEGRADED} is not ACTIVE (DEGRADED)"]
    }
    assert release.wave["open_remote"] == {"PENDING": 0, "LEASED": 0, "WAITING": 0}
    assert release.wave["destructive_workflow_count"] == 0
    assert release.wave["not_gating"] == {
        OTHER: [
            "open remote commands: PENDING=2",
            "1 active destructive workflow(s): workflow-departed-node",
        ]
    }
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("not-gating ")
    ]
    assert [line.split(" check=")[1].split(" ")[0] for line in lines] == [
        "remote_commands_idle",
        "regional_contexts",
        "fleet_wave_safety",
        "fleet_wave_safety",
    ], lines
    assert all(f"cluster={OTHER} " in line for line in lines), lines
    assert all(f"verb=join-cluster targets={JOINING} " in line for line in lines), lines


@pytest.mark.parametrize(
    "idle,wave,problem",
    [
        (
            stats(**{JOINING: {"LEASED": 1}}),
            spot_loss_elsewhere()["wave"],
            "remote commands are PENDING/LEASED/WAITING",
        ),
        (
            {"by_status": {"PENDING": 2, "LEASED": 0, "WAITING": 0}},
            spot_loss_elsewhere()["wave"],
            "remote commands are PENDING/LEASED/WAITING",
        ),
        (
            stats(),
            wave_snapshot(open_commands={JOINING: {"WAITING": 1}}, destructive={}),
            "fleet wave safety blocked by remote commands",
        ),
        (
            stats(),
            wave_snapshot(
                open_commands={}, destructive={JOINING: ["workflow-on-the-joiner"]}
            ),
            "fleet wave safety blocked by active destructive workflows",
        ),
        (
            stats(),
            {
                **wave_snapshot(
                    open_commands={}, destructive={OTHER: ["workflow-departed-node"]}
                ),
                "destructive_workflow_count_by_cluster": {},
            },
            "fleet wave safety blocked by active destructive workflows",
        ),
    ],
    ids=[
        "joining-clusters-own-command",
        "older-control-plane",
        "joining-clusters-wave-command",
        "joining-clusters-workflow",
        "unattributable-workflows",
    ],
)
def test_what_the_scope_cannot_set_aside_still_stops_the_join_cluster_mode(
    monkeypatch: pytest.MonkeyPatch,
    idle: dict[str, Any],
    wave: dict[str, Any],
    problem: str,
) -> None:
    with pytest.raises(ReleaseError, match=problem):
        join_mode(
            monkeypatch, JoiningSite(SpotLossRunner(lost="none")), idle=idle, wave=wave
        )


def test_the_joining_clusters_own_context_still_stops_the_join_cluster_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ReleaseError, match="must set NodeRecovery=None"):
        join_mode(
            monkeypatch,
            JoiningSite(SpotLossRunner(joining_broken=True)),
            **spot_loss_elsewhere(),
        )


def test_an_upgrades_wave_gate_still_judges_every_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unscoped -- an upgrade's or a rollback's waves restart every cluster's
    Agents: another cluster's destructive workflow still blocks."""

    release = JoiningSite(SpotLossRunner(lost="none"))
    monkeypatch.setattr(
        FLEET,
        "rollout_wave_safety_snapshot",
        lambda *_a, **_k: wave_snapshot(
            open_commands={}, destructive={OTHER: ["workflow-departed-node"]}
        ),
    )
    monkeypatch.setattr(FLEET, "departed_workflow_exclusions", lambda *_a: ((), ()))

    with pytest.raises(FLEET.WaveWorkflowSafetyError, match="workflow-departed-node"):
        FLEET.ensure_rollout_wave_safe(
            release, HP_TARGET, wave=("hp-node",), node_names=("hp-node",)
        )


def test_the_join_cluster_mode_scope_is_its_cluster_id() -> None:
    release = SimpleNamespace(config=SimpleNamespace(clusters=[HP_TARGET, EKS_TARGET]))
    arguments = ROLLOUT.parse_arguments(
        ["join-cluster", "--config", "site.json", "--cluster-id", JOINING]
    )
    ROLLOUT.validate_cluster_arguments(arguments)

    ROLLOUT.bind_verify_scope(release, arguments)

    assert (release.verify_scope.verb, release.verify_scope.targets) == (
        "join-cluster",
        (JOINING,),
    )
    for mode in ("plan", "status", "activate-cluster", "remove-cluster"):
        words = [mode, "--config", "site.json"]
        if mode.endswith("-cluster"):
            words += ["--cluster-id", JOINING]
        arguments = ROLLOUT.parse_arguments(words)
        ROLLOUT.validate_cluster_arguments(arguments)
        ROLLOUT.bind_verify_scope(release, arguments)
        assert release.verify_scope is None, mode
