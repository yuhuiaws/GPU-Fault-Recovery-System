"""The verify, ``status`` and stability window over a release's own node churn.

The driver verifies a release right after the deploy and rolls it back on any
FAIL, so the capacity a release saw arrive or leave read as a broken release:
a newcomer the steady-state Reconciler was still installing failed the
``gpu_cluster`` check (``installer is not Succeeded``) and the Agent coverage
(``Agent coverage drift``), a departed node's Agent record -- ACTIVE with a
lapsed lease until the departed-node cleanup retires it -- failed the coverage
too, and the newcomer's first DaemonSet Pods failed the stability window. A
``verify`` now leaves exactly those out and names them in a WARN; ``status``
judges the fleet as it is, as before; a newcomer the installer finished on is
judged like every node.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_verify_churn as CHURN
from gpu_fault_release import regional_admin_commands as COMMANDS
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release.regional_release_verify_churn import RELEASE_CHURN_KEY
from gpu_fault_release.regional_release_departed_nodes import DepartureProof
from tests.regional.test_regional_admin_checks import (
    control_api_release,
    healthy_control_api_report,
)

ARTIFACT = "a" * 64
OLD_ARTIFACT = "9" * 64
PLANNED = "hyperpod-i-000000000001"
NEWCOMER = "hyperpod-i-000000000005"
GONE = "hyperpod-i-000000000003"
OTHER_CHECKS = (
    "_check_contexts",
    "check_cpu_secrets",
    "_check_cpu_workloads",
    "check_email_notifications",
    "check_aurora_refresh",
    "check_adot_self_metrics",
    "_verify_profile",
    "run_read_only_verifiers",
    "_check_runtime_component_identity",
    "_check_nlb_runtime",
    "_check_aurora",
    "_check_monitoring",
)


def node(name: str, state: str = "Succeeded", artifact: str = ARTIFACT) -> dict:
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-a"},
            "annotations": {
                "gpu-fault.io/installer-state": state,
                "gpu-fault.io/installer-artifact-sha256": artifact,
            },
        },
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }


def churn_state(release: Any, *, captured: list[str]) -> dict[str, Any]:
    return {
        "release_id": "candidate",
        "previous": {"agent_identities": {"gpu-a": {"node_ids": captured}}},
        "departed_nodes": {
            "gpu-a": {
                "release_id": "candidate",
                "phase": "fleet-wave",
                "nodes": [{"node_id": GONE}],
            }
        },
    }


def gpu_release(items: list[dict], *, captured: list[str] | None) -> SimpleNamespace:
    class Runner:
        def run(self, arguments: list[str], **_kwargs: Any) -> str:
            return '{"status": "ok"}' if "-c" in arguments else "ready"

    def get_json(arguments: list[str]) -> dict:
        if "nodes" in arguments:
            return {"items": items}
        return {"spec": {"replicas": 1}, "status": {"readyReplicas": 1}}

    release = SimpleNamespace(
        runner=Runner(),
        node_wheel_sha=ARTIFACT,
        config=SimpleNamespace(
            site_name="test-site",
            namespace="gpu-fault-system",
            clusters=[
                SimpleNamespace(
                    cluster_id="gpu-a", context="gpu-a", hyperpod_cluster_name="hp-a"
                )
            ],
        ),
        _get_json=get_json,
        _gpu=lambda _target, *arguments: list(arguments),
    )
    if captured is not None:
        release.state = churn_state(release, captured=captured)
    return release


def checks(
    monkeypatch: pytest.MonkeyPatch, release: Any, *, mode: str, keep: str
) -> dict[str, dict[str, Any]]:
    """Every check but ``keep`` (and the GPU cluster one) passes by fiat."""

    names = [*OTHER_CHECKS, "_check_control_api", "_check_gpu_cluster"]
    for name in names:
        if name == keep:
            continue
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CHECKS.CheckValue(name)
        )
    report = CHECKS.build_health_report(release, mode=mode)
    return {item["name"]: item for item in report["checks"]}


# -- the gpu_cluster check --------------------------------------------------


def test_a_verify_names_a_newcomer_still_installing_in_a_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: ``gpu-a/<newcomer> installer is not Succeeded`` -- a FAIL, and
    the driver's automatic rollback."""

    release = gpu_release(
        [node(PLANNED), node(NEWCOMER, "Installing", "")], captured=[PLANNED]
    )

    record = checks(monkeypatch, release, mode="verify", keep="_check_gpu_cluster")[
        "gpu_cluster:gpu-a"
    ]

    assert record["status"] == "WARN", record
    assert record["summary"] == (
        "gpu-a data plane is healthy; 1 node(s) that joined during the release "
        "are still being installed by the steady-state Reconciler: "
        f"{NEWCOMER} (Installing)"
    )
    assert record["details"]["joined_nodes_installing"] == {NEWCOMER: "Installing"}
    assert record["details"]["nodes"] == sorted([PLANNED, NEWCOMER])


@pytest.mark.parametrize(
    "items,problem",
    [
        (
            [node(PLANNED, "Installing", ""), node(NEWCOMER)],
            f"gpu-a/{PLANNED} installer is not Succeeded",
        ),
        (
            [node(PLANNED), node(NEWCOMER, "Succeeded", OLD_ARTIFACT)],
            f"gpu-a/{NEWCOMER} artifact drift",
        ),
    ],
    ids=["planned-node-not-installed", "newcomer-on-the-old-artifact"],
)
def test_a_verify_still_fails_a_planned_node_and_a_drifted_newcomer(
    monkeypatch: pytest.MonkeyPatch, items: list[dict], problem: str
) -> None:
    release = gpu_release(items, captured=[PLANNED])

    record = checks(monkeypatch, release, mode="verify", keep="_check_gpu_cluster")[
        "gpu_cluster:gpu-a"
    ]

    assert (record["status"], record["summary"]) == ("FAIL", problem)


@pytest.mark.parametrize(
    "mode,captured",
    [("status", [PLANNED]), ("verify", None)],
    ids=["status", "no-capture"],
)
def test_status_and_a_release_without_a_capture_stay_strict(
    monkeypatch: pytest.MonkeyPatch, mode: str, captured: list[str] | None
) -> None:
    release = gpu_release(
        [node(PLANNED), node(NEWCOMER, "Installing", "")], captured=captured
    )

    record = checks(monkeypatch, release, mode=mode, keep="_check_gpu_cluster")[
        "gpu_cluster:gpu-a"
    ]

    assert (record["status"], record["summary"]) == (
        "FAIL",
        f"gpu-a/{NEWCOMER} installer is not Succeeded",
    )


# -- the control_api check ------------------------------------------------------


class ControlPlane:
    """The probe exec: what it was asked, and a healthy answer over that."""

    def __init__(self, agents: list[str]) -> None:
        self.agents = agents
        self.asked: dict[str, list[str]] = {}

    def __call__(self, _release: Any, *, arguments: tuple[str, ...], **_: Any) -> str:
        expected = next(
            json.loads(item.split("=", 1)[1])
            for item in arguments
            if item.startswith("ADMIN_EXPECTED_NODES_JSON=")
        )
        self.asked = expected
        report = healthy_control_api_report()
        report["clusters"]["gpu-a"].update(
            expected_node_ids=expected["gpu-a"],
            agents=[
                {
                    "node_id": name,
                    "lifecycle_state": "ACTIVE",
                    "runtime_profile_version": (
                        "previous-profile" if name == GONE else "hyperpod-v1"
                    ),
                }
                for name in self.agents
            ],
            collector_readiness={
                "ready": GONE not in self.agents,
                "nodes": [
                    {"node_id": name, "ready": name != GONE} for name in self.agents
                ],
            },
        )
        return json.dumps(report)


def control_release(captured: list[str] | None) -> SimpleNamespace:
    release = control_api_release()
    release.config.site_name = "test-site"
    release.config.clusters = [SimpleNamespace(cluster_id="gpu-a")]
    if captured is not None:
        release.state = churn_state(release, captured=captured)
    return release


def control_api(
    monkeypatch: pytest.MonkeyPatch,
    release: Any,
    plane: ControlPlane,
    *,
    mode: str,
    items: list[dict],
) -> dict[str, Any]:
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, _target: ("selector", {"items": items}),
    )
    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", plane)
    return checks(monkeypatch, release, mode=mode, keep="_check_control_api")[
        "control_api"
    ]


def test_a_verify_does_not_ask_about_a_newcomer_nor_count_a_departed_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: ``Agent coverage drift`` for both -- the newcomer has no Agent
    yet, the departed node's record is still ACTIVE."""

    plane = ControlPlane(agents=[PLANNED, GONE])
    items = [node(PLANNED), node(NEWCOMER, "Installing", "")]

    record = control_api(
        monkeypatch, control_release([PLANNED, GONE]), plane, mode="verify", items=items
    )

    assert plane.asked == {"gpu-a": [PLANNED]}, "readiness is not asked for it"
    assert record["status"] == "WARN", record
    assert record["summary"] == (
        "control-plane API, fleet and collectors are healthy; not counted: "
        f"gpu-a departed (proven by this release): {GONE}; "
        f"gpu-a joined, still installing: {NEWCOMER}"
    )
    churn = record["details"]["clusters"]["gpu-a"][RELEASE_CHURN_KEY]
    assert churn == {
        "departed": [GONE],
        "proven_by_verify": [],
        "joined_installing": {NEWCOMER: "Installing"},
    }
    readiness = record["details"]["collector_readiness"]
    assert readiness["departed_not_awaited"] == {"gpu-a": [GONE]}


def test_status_still_reports_the_coverage_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane = ControlPlane(agents=[PLANNED, GONE])
    items = [node(PLANNED), node(NEWCOMER, "Installing", "")]

    record = control_api(
        monkeypatch, control_release([PLANNED, GONE]), plane, mode="status", items=items
    )

    assert plane.asked == {"gpu-a": sorted([PLANNED, NEWCOMER])}
    assert record["status"] == "FAIL"
    assert record["summary"].startswith("gpu-a Agent coverage drift"), record


def proof_of(monkeypatch: pytest.MonkeyPatch, *, departed: bool) -> list[list[str]]:
    """The verify's own departure proof, answered; returns what it was asked."""

    asked: list[list[str]] = []

    def proof(_release: Any, _target: Any, nodes: list[str], **_kw: Any) -> Any:
        asked.append(list(nodes))
        if departed:
            return DepartureProof(({"node_id": GONE},), ())
        return DepartureProof((), (f"{GONE}: HyperPod cluster hp-a still lists it",))

    monkeypatch.setattr(CHURN, "departure_proof", proof)
    return asked


def test_a_departed_agent_no_proof_covers_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = proof_of(monkeypatch, departed=False)
    plane = ControlPlane(agents=[PLANNED, GONE])
    release = control_release([PLANNED, GONE])
    release.state["departed_nodes"] = {}

    record = control_api(
        monkeypatch, release, plane, mode="verify", items=[node(PLANNED)]
    )

    assert asked == [[GONE]], "the Agent without a Node is put to the proof"
    assert record["status"] == "FAIL"
    assert "Agent coverage drift" in record["summary"], record


def test_a_departure_the_verify_proves_itself_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node left after its cluster's last wave (or before it rolled): no
    gate of the release judged it, the same read-only proof does now."""

    proof_of(monkeypatch, departed=True)
    plane = ControlPlane(agents=[PLANNED, GONE])
    release = control_release([PLANNED, GONE])
    release.state["departed_nodes"] = {}

    record = control_api(
        monkeypatch, release, plane, mode="verify", items=[node(PLANNED)]
    )

    assert record["status"] == "WARN", record
    churn = record["details"]["clusters"]["gpu-a"][RELEASE_CHURN_KEY]
    assert (churn["departed"], churn["proven_by_verify"]) == ([GONE], [GONE])


# -- status and the stability window --------------------------------------------


def test_status_shows_the_newcomers_of_this_release() -> None:
    state = {
        "release_id": "candidate",
        "joined_nodes": {
            "gpu-a": {
                "release_id": "candidate",
                "phase": "finalize",
                "nodes": [{"node_id": NEWCOMER, "outcome": "pending"}],
            },
            "gpu-b": {"release_id": "older", "nodes": []},
        },
    }

    summary = COMMANDS.joined_nodes_summary(state)

    assert summary == {"joined_nodes": {"gpu-a": state["joined_nodes"]["gpu-a"]}}
    assert COMMANDS.joined_nodes_summary({**state, "release_id": "next"}) == {}


def pod(name: str, node_name: str, *, owner: str, ready: bool) -> dict:
    return {
        "metadata": {"name": name, "ownerReferences": [{"kind": owner}]},
        "spec": {"nodeName": node_name},
        "status": {
            "phase": "Running" if ready else "Pending",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"name": "c", "restartCount": 0 if ready else 2}],
        },
    }


def test_the_stability_window_reports_a_newcomers_daemonset_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: ``release stability window has non-Ready Pods`` for the DCGM
    exporter Pod of a node that had just joined."""

    gpu_pods = [
        pod("dcgm-exporter-a", PLANNED, owner="DaemonSet", ready=True),
        pod("dcgm-exporter-n", NEWCOMER, owner="DaemonSet", ready=False),
        pod("executor-x", "cpu-node-1", owner="ReplicaSet", ready=False),
    ]

    def get_json(arguments: list[str]) -> dict:
        return {"items": gpu_pods if "--context" in arguments else []}

    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            clusters=[SimpleNamespace(cluster_id="gpu-a", context="gpu-a")],
        ),
        state=churn_state(None, captured=[PLANNED]),
        _cpu=lambda *arguments: ["kubectl", *arguments],
        _gpu=lambda target, *arguments: [
            "kubectl",
            "--context",
            target.context,
            *arguments,
        ],
        _get_json=get_json,
        _critical_amp_alerts=lambda: {"count": 0, "alerts": []},
    )
    monkeypatch.setattr(
        VALIDATION,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: json.dumps({"queue": {}, "remote_commands": {}}),
    )

    sample = VALIDATION.stability_snapshot(release)

    assert sample["joined_node_pods"] == [f"gpu-a/dcgm-exporter-n@{NEWCOMER}"]
    assert sample["not_ready"] == ["gpu-a/executor-x"], (
        "a Deployment Pod is judged wherever it runs"
    )
    assert "gpu-a/dcgm-exporter-n/c" not in sample["restarts"]
    assert sample["restarts"]["gpu-a/dcgm-exporter-a/c"] == 0
