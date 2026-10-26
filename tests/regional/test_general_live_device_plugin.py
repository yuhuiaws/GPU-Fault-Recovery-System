"""GF-GENERAL-LIVE-014 native device-plugin registration faults.

No cluster and no node: a ``FakeProbe`` answers the device-plugin probe's
``freeze``/``resume``/``cleanup`` subcommands (and models kubelet de-registering
the resource when the plugin is frozen), a ``FakeCluster`` answers the read-only
kubectl/store reads the fixture and verdict make, and a scripted ``activity``
document carries the product's records -- the finding, the
``RESTART_*_DEVICE_PLUGIN`` workflow with its step receipts, the escalation. The
tests drive every variant's real ``execute`` and assert the verdict and the
stage errors the product's records produce (PASS / FAIL / INCONCLUSIVE), plus
the cleanup refusals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.e2e.regional import general_live_device_plugin as mod
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import (
    APPROVALS,
    coverage_rows,
    load_registry,
)
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE, PREFLIGHT_PROBE
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from tests.regional._general_live_fakes import FakeProbe, make_context, make_settings

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CASE = mod.CASE_014
SINCE = "2026-10-24T00:00:00+00:00"
U0 = "00000000-0000-0000-0000-00000000aaaa"
U1 = "00000000-0000-0000-0000-00000000bbbb"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}


# --------------------------------------------------------------------------- #
# cluster + probe fakes
# --------------------------------------------------------------------------- #
def _kubeconfig(name: str) -> Path:
    path = Path("/tmp/pt-d3") / f"fake-{name}.kubeconfig"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    return path


def _daemonset(name: str, token: str, uid: str) -> dict[str, Any]:
    return {
        "metadata": {"namespace": "kube-system", "name": name, "uid": uid},
        "spec": {
            "template": {
                "spec": {
                    "volumes": [
                        {"hostPath": {"path": "/var/lib/kubelet/device-plugins"}}
                    ]
                }
            }
        },
        "status": {"desiredNumberScheduled": 1},
        "_token": token,
    }


def _pod(
    uid: str, *, ds_uid: str, ready: bool = True, phase: str = "Running"
) -> dict[str, Any]:
    return {
        "metadata": {
            "name": f"plugin-{uid[:4]}",
            "uid": uid,
            "ownerReferences": [{"kind": "DaemonSet", "uid": ds_uid}],
        },
        "spec": {
            "nodeName": NODE,
            "volumes": [{"hostPath": {"path": "/var/lib/kubelet/device-plugins"}}],
        },
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


class FakeCluster:
    """Read-only kubectl/store surface, with mutable allocatable and pods."""

    def __init__(
        self,
        *,
        resource: str,
        baseline: int,
        daemonset: dict[str, Any],
        pods: list[dict[str, Any]],
        activity: Callable[[tuple[str, ...]], dict[str, Any]],
    ) -> None:
        self.settings = RegionalLiveSettings(
            cpu_kubeconfig=_kubeconfig("cpu"),
            gpu_kubeconfig=_kubeconfig("gpu"),
            gpu_context="ctx",
            namespace="gpu-fault-system",
            cluster_id="gpu-eks-cluster",
            region="us-west-2",
        )
        self.resource = resource
        self.allocatable_value = baseline
        self.daemonset = daemonset
        self.pods = pods
        self.activity_responder = activity

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {
            "name": NODE,
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }

    def node_metadata(self, node: str) -> dict[str, Any]:
        return {"name": NODE, "labels": {"eks.amazonaws.com/nodegroup": "g"}}

    def evidence_identity(self, **_: Any) -> dict[str, str]:
        return {"release_id": "rel", "cluster_id": "gpu-eks-cluster"}

    def cpu_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        if script == ACTIVITY_PROBE:
            return self.activity_responder(arguments)
        if script == PREFLIGHT_PROBE:
            return {"agent": {"lifecycle_state": "ACTIVE"}}
        raise AssertionError("unexpected cpu_python script")

    def kubectl(
        self,
        binary: str,
        *arguments: str,
        namespace: str | None = None,
        all_namespaces: bool = False,
        input_text: str | None = None,
        **_: Any,
    ) -> str:
        args = list(arguments)
        if args[:2] == ["get", "daemonset"]:
            return json.dumps({"items": [self.daemonset]})
        if args[:2] == ["get", "pod"]:
            return json.dumps({"items": self.pods})
        if args[:2] == ["get", "node"]:
            return json.dumps(
                {
                    "status": {
                        "allocatable": {self.resource: str(self.allocatable_value)}
                    }
                }
            )
        raise AssertionError(f"unexpected kubectl: {args}")


def _node_probe() -> FakeProbe:
    return FakeProbe({"survey": {"residue": {}, "clean": True}})


def _plugin_probe(
    cluster: FakeCluster, *, on_freeze: Callable[[], None] | None = None
) -> FakeProbe:
    def freeze(_arguments: tuple[str, ...]) -> dict[str, Any]:
        cluster.allocatable_value = 0
        if on_freeze is not None:
            on_freeze()
        return {
            "frozen": [{"pid": 4242, "binary": "nvidia-device-plugin", "stopped": True}]
        }

    return FakeProbe(
        {
            "freeze": freeze,
            "resume": {"processes": [{"pid": 4242, "resumed": True}]},
            "cleanup": {
                "processes": [],
                "residue": {"paused_processes": {}},
                "clean": True,
            },
        }
    )


def _context(spec_variant: str, cluster: FakeCluster, plugin: FakeProbe):
    spec = SPECS[spec_variant]
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=Path("/tmp/pt-d3") / f"ctx-{spec_variant}",
        regional=cluster,  # type: ignore[arg-type]
        probes={f"node@{NODE}": _node_probe(), f"plugin@{NODE}": plugin},
        settings=settings,
    )
    return spec, ctx


# --------------------------------------------------------------------------- #
# scripted activity documents
# --------------------------------------------------------------------------- #
def _doc(
    *,
    incidents: list[dict[str, Any]],
    workflows: list[dict[str, Any]],
    batches: list[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "since": SINCE,
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "incidents": incidents,
        "workflows": workflows,
        "batches": list(batches),
        "markers": [],
        "commands": [],
        "health_signals": {},
        "metric_latest": [],
        "collector_statuses": [],
        "notifications": [],
        "agent": {},
    }


def _incident(iid: str, event_id: str, state: str, wf: str) -> dict[str, Any]:
    return {
        "incident_id": iid,
        "event_id": event_id,
        "event_type": "NODE_HEALTH",
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": wf,
        "cluster_id": "gpu-eks-cluster",
        "node_ids": [NODE],
    }


def _exec(
    op: str, status: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "operation": op,
        "status": status,
        "details": details or {},
        "step_index": 0,
        "updated_at": "2026-10-24T00:05:00+00:00",
    }


def _workflow(
    wf: str, iid: str, ops: list[str], status: str, execs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "request_id": wf,
        "incident_id": iid,
        "status": status,
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in ops],
        "step_executions": execs,
    }


def _count_batch(metric_prefix: str, count: int, *, mismatch: int) -> dict[str, Any]:
    labels = {"consecutive_mismatch_samples": str(0 if mismatch == 0 else 2)}
    return {
        "observed_at": "2026-10-24T00:05:00+00:00",
        "batch_id": "k8s-1",
        "edge_filter_reasons": [],
        "collection_errors": [],
        "samples": [
            {
                "name": f"{metric_prefix}_allocatable_count",
                "device": None,
                "value": count,
                "labels": labels,
            },
            {
                "name": f"{metric_prefix}_allocatable_mismatch",
                "device": None,
                "value": mismatch,
                "labels": labels,
            },
        ],
    }


def _restart_plugin_scenario(
    target: mod.PluginTarget,
    cluster: FakeCluster,
    *,
    extra_ops: list[str] = (),
    restart_status: str = "SUCCEEDED",
    wf_status: str = "SUCCEEDED",
    incident_state: str = "RECOVERED",
) -> Callable[[tuple[str, ...]], dict[str, Any]]:
    """An activity responder: ACTION_PENDING with the delete receipt first, then
    the settled RECOVERED document (restoring the cluster's allocatable+Pod)."""

    prefix = target.kind + "_kubernetes"
    restart = target.restart_operation
    ops = [restart, "TRIGGER_HEALTH_SNAPSHOT", target.validate_operation, *extra_ops]
    pending = _doc(
        incidents=[
            _incident(
                "inc-1", f"b0-{target.mismatch_metric}-node", "ACTION_PENDING", "wf-1"
            )
        ],
        workflows=[
            _workflow(
                "wf-1",
                "inc-1",
                ops,
                "RUNNING",
                [
                    _exec(
                        restart,
                        "WAITING",
                        {
                            "node_results": {
                                NODE: {
                                    "deleted_pod_uid": U0,
                                    "allocatable": 0,
                                    "expected": cluster.allocatable_value or 8,
                                }
                            }
                        },
                    )
                ],
            )
        ],
    )
    settled_execs = [
        _exec(
            restart,
            restart_status,
            {
                "node_results": {
                    NODE: {
                        "already_healthy": False,
                        "replacement_pod_uids": [U1],
                        "allocatable": 8,
                        "expected": 8,
                    }
                }
            },
        ),
        _exec("TRIGGER_HEALTH_SNAPSHOT", "SUCCEEDED"),
        _exec(target.validate_operation, "SUCCEEDED"),
    ]
    settled = _doc(
        incidents=[
            _incident(
                "inc-1", f"b0-{target.mismatch_metric}-node", incident_state, "wf-1"
            )
        ],
        workflows=[_workflow("wf-1", "inc-1", ops, wf_status, settled_execs)],
        batches=[_count_batch(prefix, 8, mismatch=0)],
    )

    calls = {"n": 0}

    def responder(_arguments: tuple[str, ...]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return pending
        cluster.allocatable_value = 8
        cluster.pods = [_pod(U1, ds_uid=cluster.daemonset["metadata"]["uid"])]
        return settled

    return responder


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_native_014_variants() -> None:
    registry = load_registry()
    native = {
        key.split(".", 1)[1]
        for key in registry
        if key.startswith(CASE + ".") and registry[key].implemented
    }
    assert native == {
        "gpu_plugin",
        "efa_plugin",
        "one_sample",
        "replacement_uid",
        "validation_failure",
    }
    coverage = set(coverage_rows()[CASE])
    assert {spec.variant for spec in mod.VARIANTS} <= coverage


def test_registry_shapes() -> None:
    for spec in mod.VARIANTS:
        assert spec.proof_mode == "native"
        assert spec.expected_outcome == "PASS"
        assert mod.DEVICE_PLUGIN_RESTART in spec.approvals
        assert mod.DEVICE_PLUGIN_RESTART in APPROVALS
    assert SPECS["validation_failure"].approvals == (
        mod.DEVICE_PLUGIN_RESTART,
        mod.NODE_REBOOT,
    )
    assert SPECS["validation_failure"].flags == "X"
    for name in ("gpu_plugin", "efa_plugin", "one_sample", "replacement_uid"):
        assert mod.NODE_REBOOT not in SPECS[name].approvals
    assert all(spec.lanes == ("eks", "hp") for spec in mod.VARIANTS), (
        "all five native 014 variants run on both clusters"
    )


# --------------------------------------------------------------------------- #
# gpu_plugin / efa_plugin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "variant,target", [("gpu_plugin", mod.GPU_TARGET), ("efa_plugin", mod.efa_target())]
)
def test_plugin_restart_pass(variant: str, target: mod.PluginTarget) -> None:
    ds = _daemonset(f"{target.daemonset_token}-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: None,  # replaced below
    )
    cluster.activity_responder = _restart_plugin_scenario(target, cluster)
    plugin = _plugin_probe(cluster)
    spec, ctx = _context(variant, cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["restart_receipt"]["deleted_pod_uid"] == U0
    assert result["restart_receipt"]["replacement_pod_uids"] == [U1]
    assert result["proof_scope"] == "native"


def test_gpu_plugin_fail_on_reboot() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: None,
    )
    # The product restarted the plugin AND planned a RESTART_NODE: no_reboot fails.
    cluster.activity_responder = _restart_plugin_scenario(
        target, cluster, extra_ops=["RESTART_NODE"]
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("no_reboot" in error for error in result["errors"]), result["errors"]


def test_gpu_plugin_inconclusive_when_allocatable_never_drops() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    # A freeze that does not reduce allocatable (kubelet kept the devices).
    plugin = FakeProbe(
        {
            "freeze": {"frozen": [{"pid": 4242, "stopped": True}]},
            "cleanup": {
                "processes": [],
                "residue": {"paused_processes": {}},
                "clean": True,
            },
        }
    )
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_freeze_inconclusive_when_no_process() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    plugin = FakeProbe({"freeze": {"frozen": []}})
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# replacement_uid
# --------------------------------------------------------------------------- #
def test_replacement_uid_pass() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: None,
    )
    cluster.activity_responder = _restart_plugin_scenario(target, cluster)
    spec, ctx = _context("replacement_uid", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert "no_redelete" in result["stages"]


def test_replacement_uid_fail_on_redelete() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: None,
    )
    responder = _restart_plugin_scenario(target, cluster)

    def redelete(arguments: tuple[str, ...]) -> dict[str, Any]:
        doc = responder(arguments)
        # A third, different Pod is on the node: the product re-deleted the
        # recorded replacement.
        cluster.pods = [_pod("00000000-0000-0000-0000-00000000cccc", ds_uid="ds-uid")]
        return doc

    cluster.activity_responder = redelete
    spec, ctx = _context("replacement_uid", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "no_redelete" in error or "replacement_ready" in error
        for error in result["errors"]
    ), result["errors"]


# --------------------------------------------------------------------------- #
# one_sample
# --------------------------------------------------------------------------- #
def _one_sample_activity(
    consecutive_seq: list[int],
) -> Callable[[tuple[str, ...]], dict[str, Any]]:
    prefix = "gpu_kubernetes"
    calls = {"n": 0}

    def responder(_arguments: tuple[str, ...]) -> dict[str, Any]:
        index = min(calls["n"], len(consecutive_seq) - 1)
        consecutive = consecutive_seq[index]
        calls["n"] += 1
        labels = {"consecutive_mismatch_samples": str(consecutive)}
        batch = {
            "observed_at": f"2026-10-24T00:0{index}:30+00:00",
            "batch_id": f"k8s-{index}",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                {
                    "name": f"{prefix}_allocatable_mismatch",
                    "device": None,
                    "value": 1 if consecutive >= 2 else 0,
                    "labels": labels,
                },
                {
                    "name": f"{prefix}_allocatable_count",
                    "device": None,
                    "value": 0 if consecutive else 8,
                    "labels": labels,
                },
            ],
        }
        return _doc(incidents=[], workflows=[], batches=[batch])

    return responder


def test_one_sample_pass() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=_one_sample_activity([1, 0]),
    )
    plugin = _plugin_probe(cluster, on_freeze=lambda: None)
    spec, ctx = _context("one_sample", cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    # the plugin was resumed at once (single-sample window)
    assert any(call[0] == "resume" for call in plugin.calls), plugin.calls


def test_one_sample_inconclusive_when_confirmed() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=_one_sample_activity([2, 2]),
    )
    spec, ctx = _context("one_sample", cluster, _plugin_probe(cluster))
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# validation_failure
# --------------------------------------------------------------------------- #
def _validation_failure_activity(
    cluster: "FakeCluster",
) -> Callable[[tuple[str, ...]], dict[str, Any]]:
    restart = mod.GPU_TARGET.restart_operation
    incidents = [
        _incident(
            "inc-1", f"b0-{mod.GPU_TARGET.mismatch_metric}-node", "ESCALATED", "wf-1"
        ),
        {
            "incident_id": "inc-reboot",
            "event_id": "reboot-after-wf-1",
            "event_type": "NODE_HEALTH",
            "state": "QUARANTINED",
            "created_at": "2026-10-24T00:10:00+00:00",
            "workflow_request_id": "wf-reboot",
            "cluster_id": "gpu-eks-cluster",
            "node_ids": [NODE],
        },
    ]
    workflows = [
        _workflow(
            "wf-1",
            "inc-1",
            [restart, "TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_GPU"],
            "FAILED",
            [
                _exec(
                    restart,
                    "FAILED",
                    {
                        "node_results": {NODE: {"allocatable": 0, "expected": 8}},
                        "error": "device plugin did not restore nvidia.com/gpu",
                    },
                )
            ],
        ),
        _workflow(
            "wf-reboot",
            "inc-reboot",
            ["MARK_UNSCHEDULABLE", "QUARANTINE", "RESTART_NODE", "VALIDATE_GPU"],
            "RUNNING",
            [],
        ),
    ]

    calls = {"n": 0}

    def responder(_arguments: tuple[str, ...]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            # The product deleted the frozen original; a replacement appears
            # (the next poll's re-freeze loop will re-break it). No escalation
            # yet.
            cluster.pods = [_pod(U1, ds_uid="ds-uid")]
            return _doc(incidents=[incidents[0]], workflows=[workflows[0]])
        return _doc(incidents=incidents, workflows=workflows)

    return responder


def test_validation_failure_pass() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: None,
    )
    cluster.activity_responder = _validation_failure_activity(cluster)
    # A replacement Pod appears on the first poll and is re-frozen; allocatable
    # stays short across the restart's confirmation window.
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("validation_failure", cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["reboot_escalations"] == ["inc-reboot"]
    # the replacement Pod was re-frozen
    assert sum(1 for call in plugin.calls if call[0] == "freeze") >= 2


def test_validation_failure_fail_without_escalation() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    restart = target.restart_operation
    doc = _doc(
        incidents=[
            _incident("inc-1", f"b0-{target.mismatch_metric}-node", "ESCALATED", "wf-1")
        ],
        workflows=[
            _workflow(
                "wf-1",
                "inc-1",
                [restart],
                "FAILED",
                [_exec(restart, "FAILED", {"error": "device plugin did not restore"})],
            )
        ],
    )
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U1, ds_uid="ds-uid")],
        activity=lambda a: doc,
    )
    spec, ctx = _context("validation_failure", cluster, _plugin_probe(cluster))
    cluster.pods = [_pod(U1, ds_uid="ds-uid")]
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("reboot_escalation" in error for error in result["errors"]), result[
        "errors"
    ]


# --------------------------------------------------------------------------- #
# cleanup refusals
# --------------------------------------------------------------------------- #
def test_cleanup_refuses_when_allocatable_not_restored() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    fixture.freeze_pod(U0, seconds=120)  # drops allocatable to 0; never restored
    with pytest.raises(RegionalFixtureError, match="allocatable"):
        fixture.restore()


def test_cleanup_refuses_on_extra_pod() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    fixture.freeze_pod(U0, seconds=120)
    cluster.allocatable_value = 8
    cluster.pods = [_pod(U0, ds_uid="ds-uid"), _pod(U1, ds_uid="ds-uid")]
    with pytest.raises(RegionalFixtureError, match="exactly one"):
        fixture.restore()


def test_cleanup_refuses_when_probe_not_clean() -> None:
    target = mod.GPU_TARGET
    ds = _daemonset("nvidia-device-plugin-ds", target.daemonset_token, "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    plugin = FakeProbe(
        {
            "freeze": (
                lambda a: (
                    setattr(cluster, "allocatable_value", 0),
                    {"frozen": [{"pid": 1}]},
                )[1]
            ),
            "cleanup": {"residue": {"paused_processes": {"1": {}}}, "clean": False},
        }
    )
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    fixture.freeze_pod(U0, seconds=120)
    with pytest.raises(RegionalFixtureError, match="not clean"):
        fixture.restore()


def test_discover_blocks_on_missing_daemonset() -> None:
    from scripts.e2e.regional.general_live_fixture import VariantBlocked

    target = mod.GPU_TARGET
    ds = _daemonset("other", "something-else", "ds-uid")
    cluster = FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[],
        activity=lambda a: _doc(incidents=[], workflows=[]),
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    with pytest.raises(VariantBlocked):
        fixture.discover()
