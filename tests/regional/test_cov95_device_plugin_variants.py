"""GF-GENERAL-LIVE-014 fixture and verdict branches the variant tests leave out.

``test_general_live_device_plugin.py`` drives every variant over its
``FakeCluster`` and scripted activity; this module reuses those fakes (with its
own ``tmp_path`` context) for what a live node or product can also do: list
plugin Pods of another node, DaemonSet or a terminating/companion Pod, list two
plugins, advertise a non-integer count, freeze twice and restore; and a product
whose restart deletes another Pod or leaves no delete receipt, ends with two or
an unready plugin Pod, fails its restart step, leaves allocatable short or the
mismatch up, labels its liveness sample off-shape or late, or isolates the node.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_device_plugin as mod
from scripts.e2e.regional.general_live_fixture import VariantBlocked
from tests.regional._general_live_fakes import FakeProbe, make_context, make_settings
from tests.regional.test_general_live_device_plugin import (
    NODE,
    SPECS,
    U0,
    U1,
    U2,
    FakeCluster,
    _cluster,
    _count_batch,
    _doc,
    _incident,
    _plugin_probe,
    _pod,
    _restart_scenario,
    _workflow,
)

GPU = mod.GPU_TARGET
PREFIX = GPU.kind + "_kubernetes"


def _context(variant: str, cluster: FakeCluster, plugin: FakeProbe, tmp_path: Path):
    spec = SPECS[variant]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=cluster,  # type: ignore[arg-type]
        probes={
            f"node@{NODE}": FakeProbe({"survey": {"residue": {}, "clean": True}}),
            f"plugin@{NODE}": plugin,
        },
        settings=make_settings(spec, node=NODE),
    )
    return spec, ctx


def _settled(
    responder: Callable[[tuple[str, ...]], dict[str, Any]],
    cluster: FakeCluster,
    change: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[tuple[str, ...]], dict[str, Any]]:
    """``responder`` with every settled read (a SUCCEEDED workflow) passed
    through ``change`` -- which may also move the cluster (pods, allocatable)."""

    def respond(arguments: tuple[str, ...]) -> dict[str, Any]:
        value = responder(arguments)
        if any(item.get("status") == "SUCCEEDED" for item in value["workflows"]):
            return change(dict(value))
        return value

    return respond


def _run(
    tmp_path: Path,
    *,
    variant: str = "gpu_plugin",
    change: Callable[[FakeCluster, dict[str, Any]], dict[str, Any]] | None = None,
    **scenario: Any,
) -> dict[str, Any]:
    cluster = _cluster(GPU)
    responder = _restart_scenario(GPU, cluster, **scenario)
    if change is not None:
        responder = _settled(responder, cluster, lambda doc: change(cluster, doc))
    cluster.activity_responder = responder
    spec, ctx = _context(variant, cluster, _plugin_probe(cluster), tmp_path)
    return spec.execute(ctx)


# --------------------------------------------------------------------------- #
# The fixture: discovery, freeze, restore
# --------------------------------------------------------------------------- #
def _discovering(
    tmp_path: Path, cluster: FakeCluster
) -> mod.DevicePluginRegistrationFixture:
    _, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster), tmp_path)
    return mod.DevicePluginRegistrationFixture(ctx, GPU)


def test_only_the_daemonsets_live_plugin_pod_on_the_node_is_the_plugin(
    tmp_path: Path,
) -> None:
    cluster = _cluster(GPU)
    elsewhere = _pod(U1, ds_uid="ds-uid")
    elsewhere["spec"]["nodeName"] = "ip-10-0-99-9.us-west-2.compute.internal"
    foreign = _pod(U2, ds_uid="another-daemonset")
    leaving = _pod("00000000-0000-0000-0000-00000000dddd", ds_uid="ds-uid")
    leaving["metadata"]["deletionTimestamp"] = "2026-10-24T00:00:05Z"
    companion = _pod("00000000-0000-0000-0000-00000000eeee", ds_uid="ds-uid")
    companion["spec"]["volumes"] = [{"hostPath": {"path": "/var/run"}}]
    cluster.pods = [*cluster.pods, elsewhere, foreign, leaving, companion]
    fixture = _discovering(tmp_path, cluster)
    evidence = fixture.discover()
    assert evidence["original_pod_uid"] == U0, evidence
    assert [pod["metadata"]["uid"] for pod in fixture.plugin_pods_on_node()] == [U0]

    cluster.pods.append(_pod(U1, ds_uid="ds-uid"))
    with pytest.raises(
        VariantBlocked, match="expected exactly one .* found 2"
    ) as refused:
        _discovering(tmp_path, cluster).discover()
    assert refused.value.reason == "FIXTURE_NOT_READY", refused.value.detail


def test_a_node_whose_count_is_no_integer_advertises_nothing_to_lose(
    tmp_path: Path,
) -> None:
    cluster = _cluster(GPU)
    cluster.allocatable_value = "8Gi"  # type: ignore[assignment]
    fixture = _discovering(tmp_path, cluster)
    assert fixture.allocatable() == 0, "an unreadable count is no count"
    with pytest.raises(VariantBlocked, match=f"advertises no {GPU.resource}"):
        fixture.discover()


def test_a_second_freeze_keeps_one_restore_and_the_restore_proves_the_node(
    tmp_path: Path,
) -> None:
    cluster = _cluster(GPU)
    cluster.pods[0]["spec"]["containers"] = [
        {"name": "plugin", "command": ["/usr/bin/nvidia-device-plugin"], "image": "x"}
    ]
    plugin = _plugin_probe(cluster)
    _, ctx = _context("gpu_plugin", cluster, plugin, tmp_path)
    fixture = mod.DevicePluginRegistrationFixture(ctx, GPU)
    assert fixture.restore() == {"skipped": "no freeze happened"}
    fixture.discover()
    fixture.freeze_pod(U0, seconds=600)
    fixture.freeze_pod(U0, seconds=600)
    assert ctx.cleanup.pending() == ["device-plugin-restore"], "one restore, once"
    freeze = next(call for call in plugin.calls if call[0] == "freeze")
    assert freeze[freeze.index("--pod-command-binary") + 1] == "nvidia-device-plugin"
    restored = fixture.restore()
    assert restored == {"resumed": [], "allocatable": 8, "plugin_pod_uid": U0}
    assert (ctx.case_dir / "plugin-restore.json").is_file(), "the restore is evidence"


# --------------------------------------------------------------------------- #
# The restart's receipts and the Pod census
# --------------------------------------------------------------------------- #
def test_a_restart_that_deleted_another_pod_is_named(tmp_path: Path) -> None:
    other = {"deleted_pod_uid": U2, "allocatable": 8, "expected": 8}
    result = _run(
        tmp_path,
        delete_receipt={**other, "restart_reason": mod.UNRESPONSIVE_FAILURE_MODE},
        final_receipt={**other, "already_healthy": False, "replacement_pod_uids": [U1]},
    )
    assert result["verdict"] == "FAIL", result["stages"]
    assert result["stages"]["deleted_original"] == [
        f"the restart deleted pod {U2}, not the frozen plugin pod {U0}"
    ]


def test_without_a_delete_receipt_the_census_proves_the_delete(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        delete_receipt={"allocatable": 8, "expected": 8},
        final_receipt={
            "already_healthy": False,
            "replacement_pod_uids": [U1],
            "allocatable": 8,
        },
    )
    assert result["stages"]["deleted_original"] == [], result["stages"]
    assert result["verdict"] == "PASS", result["errors"]
    assert any(
        "replaced before a poll caught it" in note for note in result["notes"]
    ), result["notes"]


@pytest.mark.parametrize(
    ("pods", "expected"),
    [
        (
            lambda: [_pod(U1, ds_uid="ds-uid"), _pod(U2, ds_uid="ds-uid")],
            "2 plugin Pods on the node after the restart, expected one",
        ),
        (
            lambda: [_pod(U1, ds_uid="ds-uid", ready=False)],
            f"the replacement plugin Pod {U1} is not Ready",
        ),
    ],
)
def test_the_replacement_census_needs_one_ready_pod(
    tmp_path: Path, pods: Callable[[], list[dict[str, Any]]], expected: str
) -> None:
    def census(cluster: FakeCluster, doc: dict[str, Any]) -> dict[str, Any]:
        cluster.pods = pods()
        return doc

    result = _run(tmp_path, change=census)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["replacement_ready"] == [expected]


def test_a_failed_restart_step_is_no_confirmed_replacement(tmp_path: Path) -> None:
    def failed(cluster: FakeCluster, doc: dict[str, Any]) -> dict[str, Any]:
        workflow = dict(doc["workflows"][0], status="FAILED")
        executions = [dict(item) for item in workflow["step_executions"]]
        executions[0].update(status="FAILED", error="replacement never became Ready")
        workflow["step_executions"] = executions
        return {**doc, "workflows": [workflow]}

    result = _run(
        tmp_path,
        variant="replacement_uid",
        change=failed,
        final_receipt={"already_healthy": False, "replacement_pod_uids": [U1]},
    )
    receipt = result["restart_receipt"]
    assert receipt["restart_failed"] is True, receipt
    assert receipt["restart_error"] == "replacement never became Ready"
    assert receipt["final_allocatable"] == 8, (
        "a pass that names no count keeps the count an earlier pass recorded"
    )
    assert result["stages"]["no_redelete"] == [
        "the restart step failed rather than confirming the replacement"
    ]


@pytest.mark.parametrize(
    ("count", "mismatch", "allocatable", "expected"),
    [
        (
            None,
            None,
            7,
            f"no fresh {GPU.allocatable_metric} sample and allocatable still short",
        ),
        (
            6,
            0,
            8,
            f"the last {GPU.allocatable_metric} sample is 6, below the baseline 8",
        ),
        (8, 2, 8, f"{GPU.mismatch_metric} is still 2"),
    ],
)
def test_the_restored_allocatable_is_read_from_the_products_last_count(
    tmp_path: Path,
    count: int | None,
    mismatch: int | None,
    allocatable: int,
    expected: str,
) -> None:
    def counted(cluster: FakeCluster, doc: dict[str, Any]) -> dict[str, Any]:
        cluster.allocatable_value = allocatable
        batches = [item for item in doc["batches"] if item.get("batch_id") != "k8s-1"]
        if count is not None:
            batches.append(_count_batch(PREFIX, count, mismatch=mismatch or 0))
        return {**doc, "batches": batches}

    result = _run(tmp_path, change=counted)
    assert result["stages"]["allocatable_restored"] == [expected], result["stages"]


# --------------------------------------------------------------------------- #
# The liveness sample and the node
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"failure_mode": "socket_absent"},
            [
                f"failure_mode is 'socket_absent', expected {mod.UNRESPONSIVE_FAILURE_MODE}"
            ],
        ),
        (
            {"consecutive_timeout_samples": "1"},
            [
                "the sample is 1 at consecutive_timeout_samples=1 below "
                "required_consecutive_samples=2"
            ],
        ),
        (
            {"consecutive_timeout_samples": "two"},
            ["unreadable consecutive/required labels two/2"],
        ),
        ({"consecutive_timeout_samples": None}, []),
    ],
)
def test_the_liveness_sample_must_carry_the_hang_and_its_confirmation(
    tmp_path: Path, overrides: dict[str, Any], expected: list[str]
) -> None:
    result = _run(tmp_path, sample_overrides=overrides)
    assert result["stages"]["detected"] == expected, result["stages"]


def test_a_sample_after_the_detection_budget_is_named_late_not_graded(
    tmp_path: Path,
) -> None:
    cluster = _cluster(GPU)
    inner = _restart_scenario(GPU, cluster)
    phase = {"name": ""}

    def respond(arguments: tuple[str, ...]) -> dict[str, Any]:
        value = inner(arguments)
        if phase["name"] == "liveness-detection":
            return {**value, "batches": []}  # the edge is not stored yet
        return value

    cluster.activity_responder = respond
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster), tmp_path)
    wait = ctx.wait_until

    def tracked(accept: Callable[[], Any], **options: Any) -> Any:
        phase["name"] = options["name"]
        return wait(accept, **options)

    ctx.wait_until = tracked  # type: ignore[method-assign]
    result = spec.execute(ctx)
    (error,) = result["stages"]["detected"]
    assert error.startswith(f"no {GPU.liveness_metric}=1 sample within"), error
    assert error.endswith("; a sample arrived only at 2026-10-24T00:01:40+00:00"), error


class IsolatingCluster(FakeCluster):
    """A node the product cordoned once the restart settled."""

    cordoned = False

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {**super().node_snapshot(node), "unschedulable": self.cordoned}


def test_a_node_left_cordoned_after_the_fault_fails_the_isolation_stage(
    tmp_path: Path,
) -> None:
    template = _cluster(GPU)
    cluster = IsolatingCluster(
        resource=GPU.resource,
        baseline=8,
        daemonset=template.daemonset,
        pods=template.pods,
    )

    def cordon(cluster_: FakeCluster, doc: dict[str, Any]) -> dict[str, Any]:
        cluster.cordoned = True
        return doc

    cluster.activity_responder = _settled(
        _restart_scenario(GPU, cluster), cluster, lambda doc: cordon(cluster, doc)
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster), tmp_path)
    result = spec.execute(ctx)
    assert result["stages"]["not_isolated"] == [
        "the node is isolated after a device-plugin fault"
    ]


def test_one_sample_ignores_a_workflow_that_plans_no_plugin_restart(
    tmp_path: Path,
) -> None:
    cluster = _cluster(GPU)
    unrelated = _workflow(
        "wf-gpu", "inc-gpu", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], "SUCCEEDED", []
    )
    cluster.activity_responder = lambda _arguments: _doc(
        incidents=[_incident("inc-gpu", "b0-gpu_xid-node", "RECOVERED", "wf-gpu")],
        workflows=[unrelated],
    )
    spec, ctx = _context("one_sample", cluster, _plugin_probe(cluster), tmp_path)
    result = spec.execute(ctx)
    assert result["stages"]["no_action"] == [], result["stages"]
    assert result["verdict"] == "PASS", result["errors"]
