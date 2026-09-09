"""The acceptance wrapper over ``gpu-fault-admin config spare``.

The refusal matrix, the baseline-before-patch ordering and the exact-baseline
release are tested where the logic lives, in
``tests/admin/test_admin_warm_spare.py``. What is tested here is the wrapper's
own surface: it binds the same functions rather than copies of them, it maps a
site profile onto the admin request, and it still refuses the shared ``--node``
flag that would cordon the wrong machine.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import warm_spare
from scripts.e2e.regional import declare_warm_spare as helper
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


def test_the_wrapper_binds_the_admin_module_not_a_copy() -> None:
    assert helper.declare_refusals is warm_spare.declare_refusals, (
        "the wrapper must not re-implement the declare refusals"
    )
    assert helper.release_refusals is warm_spare.release_refusals, (
        "the wrapper must not re-implement the release refusals"
    )
    assert helper.declare is warm_spare.declare, "declare must be the admin one"
    assert helper.release is warm_spare.release, "release must be the admin one"
    assert helper.DECLARE_CONFIRMATION == "DECLARE_WARM_SPARE_CORDON"
    assert helper.RELEASE_CONFIRMATION == "RELEASE_WARM_SPARE_UNCORDON"
    assert helper.DECLARE_CONFIRMATION == warm_spare.DECLARE_CONFIRMATION
    assert helper.RELEASE_CONFIRMATION == warm_spare.RELEASE_CONFIRMATION


def test_helper_is_read_only_by_default() -> None:
    arguments = helper.parser().parse_args(["--spare-node", "node-spare"])

    assert arguments.declare is False
    assert arguments.release is False
    assert arguments.confirm == ""
    assert helper.mode(arguments) == "check"


def test_the_spare_is_never_named_by_the_shared_node_flag() -> None:
    # A site profile's `node` is the collector cases' target, a different machine
    # from the spare. If this helper accepted --node it would inherit that value
    # from the profile and cordon the wrong node.
    with pytest.raises(SystemExit):
        helper.parser().parse_args(["--node", "node-spare"])


def test_helper_carries_no_site_topology() -> None:
    source = Path(helper.__file__).read_text(encoding="utf-8")

    assert "/secure/gpu-fault-bootstrap" not in source
    assert "514385905925" not in source
    assert "hyperpod-i-" not in source


def test_configure_reads_the_profile_flags_and_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_SPARE_NODE", raising=False)
    monkeypatch.setenv("GPU_FAULT_FAULT_NODE", "node-fault")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", "hp-a")
    arguments = helper.parser().parse_args(
        ["--spare-node", "node-spare", "--baseline", str(tmp_path / "b.json")]
    )

    settings = helper.configure(arguments)

    assert settings == helper.Settings(
        node="node-spare",
        fault_node="node-fault",
        hyperpod_cluster="hp-a",
        baseline=tmp_path / "b.json",
    )
    with pytest.raises(RegionalFixtureError, match="baseline record path"):
        helper.configure(helper.parser().parse_args(["--spare-node", "node-spare"]))


def test_the_request_maps_the_mode_and_records_the_baseline_directory(
    tmp_path: Path,
) -> None:
    settings = helper.Settings(
        node="node-spare",
        fault_node="node-fault",
        hyperpod_cluster="hp-a",
        baseline=tmp_path / "records" / "spare.json",
    )
    arguments = helper.parser().parse_args(
        [
            "--spare-node",
            "node-spare",
            "--declare",
            "--confirm",
            helper.DECLARE_CONFIRMATION,
        ]
    )

    request = helper.request(settings, arguments)

    assert request.mode == "declare"
    assert request.node == "node-spare"
    assert request.fault_node == "node-fault"
    assert request.confirmation == helper.DECLARE_CONFIRMATION
    assert request.reference == "acceptance-warm-spare"
    assert request.state_dir == tmp_path / "records"
    released = helper.request(
        settings,
        helper.parser().parse_args(
            ["--spare-node", "node-spare", "--release", "--reference", "DESTR-003"]
        ),
    )
    assert released.mode == "release"
    assert released.reference == "DESTR-003"


def test_the_agent_check_needs_exactly_one_active_row() -> None:
    # "纳入监控" is a hard precondition of the warm-spare path: an unmonitored
    # node has no Agent to fence and no health signal to trust.
    node = "node-spare"

    assert helper.agent_state({"agents": []}, node).lifecycle_state is None
    assert (
        helper.agent_state(
            {"agents": [{"node_id": node, "lifecycle_state": "REVOKED"}]}, node
        ).lifecycle_state
        == "REVOKED"
    )
    # Two Agents claiming one node is ambiguous, not "the first one wins".
    assert (
        helper.agent_state(
            {
                "agents": [
                    {"node_id": node, "lifecycle_state": "ACTIVE"},
                    {"node_id": node, "lifecycle_state": "ACTIVE"},
                ]
            },
            node,
        ).lifecycle_state
        is None
    )
    active = helper.agent_state(
        {"agents": [{"node_id": node, "lifecycle_state": "ACTIVE"}]}, node
    )
    assert active == warm_spare.AgentState(lifecycle_state="ACTIVE")
    assert warm_spare.declare_refusals(
        node,
        spare=_ready_snapshot(node),
        fault=None,
        fault_node="",
        declared=[],
        workloads=[],
        agent=helper.agent_state({"agents": []}, node),
    ) == ["node does not have exactly one ACTIVE Agent"]


def test_an_unreadable_store_is_a_refusal_not_a_pass() -> None:
    class Warm:
        def store_snapshot(self) -> dict[str, Any]:
            raise RegionalFixtureError("cpu python probe failed")

    state = helper.store_agent_lookup(Warm())("cluster-a", "node-spare")  # type: ignore[arg-type]

    assert state.lifecycle_state is None
    assert state.error == "cpu python probe failed"


def test_the_node_api_is_kubectl_bound_to_the_profile_gpu_cluster() -> None:
    regional = SimpleNamespace(
        settings=SimpleNamespace(
            gpu_kubeconfig=Path("/p/gpu.kubeconfig"), gpu_context="ctx"
        )
    )

    api = helper.node_api(regional)  # type: ignore[arg-type]

    assert isinstance(api, warm_spare.KubectlNodeApi), type(api)
    assert api.kubectl == [
        "kubectl",
        "--kubeconfig",
        "/p/gpu.kubeconfig",
        "--context",
        "ctx",
    ]


def _ready_snapshot(node: str) -> dict[str, Any]:
    return warm_spare.node_snapshot(
        {
            "metadata": {
                "name": node,
                "labels": {
                    warm_spare.HYPERPOD_HEALTH_LABEL: "Schedulable",
                    warm_spare.INSTANCE_GROUP_LABEL: "group-a",
                    "node.kubernetes.io/instance-type": "ml.p5en.48xlarge",
                },
            },
            "spec": {},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
    )
