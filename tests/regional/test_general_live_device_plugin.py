"""GF-GENERAL-LIVE-014 native device-plugin faults, against both products.

No cluster and no node: a ``FakeProbe`` answers the device-plugin probe's
``feature`` / ``freeze`` / ``resume`` / ``cleanup`` / ``probe-log`` subcommands
(modelling kubelet keeping allocatable full while the plugin is frozen, and the
host collector's journal of timed-out probe ticks), a ``FakeCluster`` answers
the read-only kubectl/store reads the fixture and verdict make (including the
control-plane feature probe), and a scripted ``activity`` document carries the
product's records -- the ``*_device_plugin_unresponsive`` sample with its
labels, the finding, the ``RESTART_*_DEVICE_PLUGIN`` workflow with its delete
and confirm receipts. The tests drive every variant's real ``execute`` and
assert the verdict and the stage errors (PASS / FAIL / INCONCLUSIVE) for the
liveness-probe product and the earlier allocatable-only product, plus the
cleanup refusals.
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
U2 = "00000000-0000-0000-0000-00000000cccc"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}
FROZEN_EPOCH = 1_761_559_200.0
FIRST_TIMEOUT = {
    "kind": "timeout",
    "realtime_epoch": FROZEN_EPOCH + 12.4,
    "sockets": ["nvidia-gpu.sock"],
    "timeout_seconds": 5.0,
    "consecutive": 1,
    "required": 2,
    "message": "device plugin nvidia-gpu.sock did not answer ... (1 consecutive; "
    "finding after 2)",
}
PRESENT_MODULES = {
    "device_plugin": True,
    "device_plugin_probe": True,
    "device_plugin_liveness": True,
}
ABSENT_MODULES = {key: False for key in PRESENT_MODULES}


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
            "name": f"plugin-{uid[-4:]}",
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
        activity: Callable[[tuple[str, ...]], dict[str, Any]] | None = None,
        control_plane: dict[str, Any] | None = None,
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
        self.activity_responder = activity or (
            lambda a: _doc(incidents=[], workflows=[])
        )
        #: The API Pod's answer to the control-plane feature probe.
        self.control_plane = (
            control_plane
            if control_plane is not None
            else {"present": True, "rule": True, "severity": "critical"}
        )
        self.feature_probes: list[tuple[str, ...]] = []

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
        if script == mod.LIVENESS_FEATURE_PROBE:
            self.feature_probes.append(arguments)
            return dict(self.control_plane)
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


def _cluster(target: mod.PluginTarget, **options: Any) -> FakeCluster:
    ds = _daemonset(f"{target.daemonset_token}-ds", target.daemonset_token, "ds-uid")
    return FakeCluster(
        resource=target.resource,
        baseline=8,
        daemonset=ds,
        pods=[_pod(U0, ds_uid="ds-uid")],
        **options,
    )


def _node_probe() -> FakeProbe:
    return FakeProbe({"survey": {"residue": {}, "clean": True}})


def _plugin_probe(
    cluster: FakeCluster,
    *,
    modules: dict[str, bool] | None = None,
    device_plugin_env: dict[str, str] | None = None,
    drop_allocatable: bool = False,
    first_timeout: dict[str, Any] | None = FIRST_TIMEOUT,
    second_timeout_seen: bool = False,
    journal_timeouts: list[dict[str, Any]] | None = None,
    on_freeze: Callable[[], None] | None = None,
) -> FakeProbe:
    """The device-plugin probe: ``feature`` names the node generation, ``freeze``
    models kubelet (allocatable stays full unless ``drop_allocatable``) and the
    journal-driven first-timeout resume, ``probe-log`` the host collector's
    timed-out ticks."""

    def freeze(arguments: tuple[str, ...]) -> dict[str, Any]:
        if drop_allocatable:
            cluster.allocatable_value = 0
        if on_freeze is not None:
            on_freeze()
        receipt: dict[str, Any] = {
            "frozen": [
                {"pid": 4242, "binary": "nvidia-device-plugin", "stopped": True}
            ],
            "frozen_since_epoch": FROZEN_EPOCH,
        }
        if "--resume-after-first-timeout" in arguments:
            receipt.update(
                {
                    "window_seconds": int(
                        arguments[arguments.index("--window-seconds") + 1]
                    ),
                    "first_timeout": first_timeout,
                    "second_timeout_seen": second_timeout_seen,
                    "timeouts": [first_timeout] if first_timeout else [],
                    "resumed": [{"pid": 4242, "resumed": True}],
                    "resumed_at_epoch": FROZEN_EPOCH + 12.9,
                    "resume_latency_seconds": 0.5 if first_timeout else None,
                }
            )
        return receipt

    timeouts = (
        journal_timeouts
        if journal_timeouts is not None
        else ([first_timeout] if first_timeout else [])
    )
    return FakeProbe(
        {
            "feature": {
                "unit": "gpu-fault-host-collector.service",
                "venv": "/opt/gpu-fault/current/venv",
                "modules": PRESENT_MODULES if modules is None else modules,
                "device_plugin_env": device_plugin_env or {},
            },
            "freeze": freeze,
            "resume": {"processes": [{"pid": 4242, "resumed": True}]},
            "probe-log": {"timeouts": timeouts, "entries": timeouts},
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


def _calls(plugin: FakeProbe, subcommand: str) -> list[tuple[str, ...]]:
    return [call for call in plugin.calls if call[0] == subcommand]


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


def liveness_labels(target: mod.PluginTarget, **overrides: str) -> dict[str, str]:
    """The host collector's sample labels for a hung plugin (b204
    ``collectors/host/device_plugin.py``)."""

    labels = {
        "resource": target.kind.upper(),
        "resource_name": target.resource,
        "node_instance_type": "p5en.48xlarge",
        "sockets": target.socket_name,
        "probe_state": "TIMEOUT",
        "probe_states": f"{target.socket_name}=TIMEOUT",
        "probe_detail": f"{target.socket_name}: no bytes from the plugin within 5s",
        "transport_answered": "false",
        "latency_ms": "5000.4",
        "timeout_seconds": "5",
        "consecutive_timeout_samples": "2",
        "required_consecutive_samples": "2",
        "failure_mode": mod.UNRESPONSIVE_FAILURE_MODE,
        "expected_count": "8",
    }
    labels.update(overrides)
    return labels


def _liveness_batch(
    target: mod.PluginTarget,
    value: int,
    *,
    observed_at: str = "2026-10-24T00:01:40+00:00",
    **overrides: str,
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at[-14:-6]}",
        "edge_filter_reasons": (
            [f"threshold:{target.liveness_metric}"] if value else ["recovered"]
        ),
        "collection_errors": [],
        "samples": [
            {
                "name": target.liveness_metric,
                "device": None,
                "value": value,
                "labels": liveness_labels(target, **overrides),
            }
        ],
    }


def _restart_scenario(
    target: mod.PluginTarget,
    cluster: FakeCluster,
    *,
    liveness: bool = True,
    extra_ops: list[str] = (),
    delete_receipt: dict[str, Any] | None = None,
    final_receipt: dict[str, Any] | None = None,
    incident_state: str = "RECOVERED",
    sample_overrides: dict[str, str] | None = None,
) -> Callable[[tuple[str, ...]], dict[str, Any]]:
    """An activity responder for the restart variants.

    Liveness product: the detection read returns the ``*_device_plugin_
    unresponsive`` = 1 edge batch, the first settle poll the ACTION_PENDING
    document with the delete-pass receipt, every later poll the settled
    RECOVERED document (restoring the cluster's Pod). Earlier product: the
    allocatable-mismatch chain, no liveness batch, no ``restart_reason``.
    """

    metric = target.liveness_metric if liveness else target.mismatch_metric
    prefix = target.kind + "_kubernetes"
    restart = target.restart_operation
    ops = [restart, "TRIGGER_HEALTH_SNAPSHOT", target.validate_operation, *extra_ops]
    edge = [_liveness_batch(target, 1, **(sample_overrides or {}))] if liveness else []
    if delete_receipt is None:
        delete_receipt = {
            "deleted_pod": f"kube-system/plugin-{U0[-4:]}",
            "deleted_pod_uid": U0,
            "allocatable": 8 if liveness else 0,
            "expected": 8,
        }
        if liveness:
            delete_receipt["restart_reason"] = mod.UNRESPONSIVE_FAILURE_MODE
    if final_receipt is None:
        final_receipt = {
            "already_healthy": False,
            "deleted_pod_uid": U0,
            "deleted_pod": f"kube-system/plugin-{U0[-4:]}",
            "replacement_pod_uids": [U1],
            "allocatable": 8,
            "expected": 8,
        }
    pending = _doc(
        incidents=[_incident("inc-1", f"b0-{metric}-node", "ACTION_PENDING", "wf-1")],
        workflows=[
            _workflow(
                "wf-1",
                "inc-1",
                ops,
                "RUNNING",
                [_exec(restart, "WAITING", {"node_results": {NODE: delete_receipt}})],
            )
        ],
        batches=edge,
    )
    settled_execs = [
        _exec(restart, "SUCCEEDED", {"node_results": {NODE: final_receipt}}),
        _exec("TRIGGER_HEALTH_SNAPSHOT", "SUCCEEDED"),
        _exec(target.validate_operation, "SUCCEEDED"),
    ]
    settled = _doc(
        incidents=[_incident("inc-1", f"b0-{metric}-node", incident_state, "wf-1")],
        workflows=[_workflow("wf-1", "inc-1", ops, "SUCCEEDED", settled_execs)],
        batches=[
            *edge,
            _count_batch(prefix, 8, mismatch=0),
            *(
                [_liveness_batch(target, 0, observed_at="2026-10-24T00:06:10+00:00")]
                if liveness
                else []
            ),
        ],
    )
    calls = {"n": 0}

    def responder(_arguments: tuple[str, ...]) -> dict[str, Any]:
        calls["n"] += 1
        # Liveness product: read 1 is the detection poll (edge only), read 2
        # the first settle poll; earlier product: read 1 is the settle poll.
        if liveness and calls["n"] == 1:
            return _doc(incidents=[], workflows=[], batches=edge)
        if calls["n"] == (2 if liveness else 1):
            return pending
        cluster.allocatable_value = 8
        cluster.pods = [_pod(U1, ds_uid=cluster.daemonset["metadata"]["uid"])]
        return settled

    return responder


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_014_variant_with_the_reboot_path_deferred() -> None:
    registry = load_registry()
    keys = {key.split(".", 1)[1] for key in registry if key.startswith(CASE + ".")}
    assert keys == set(coverage_rows()[CASE])
    implemented = {
        spec.variant for spec in mod.VARIANTS if registry[spec.key].implemented
    }
    assert implemented == {"gpu_plugin", "efa_plugin", "one_sample", "replacement_uid"}
    deferred = SPECS["validation_failure"]
    assert deferred.deferred.startswith("CAPABILITY_ABSENT:"), deferred.deferred
    assert "Allocate" in deferred.deferred and "kubelet" in deferred.deferred, (
        "the NOT_RUN reason cites the kubelet mechanism the harness cannot produce"
    )
    assert deferred.execute is None
    assert "re-frozen replacement" in deferred.deferred


def test_registry_shapes() -> None:
    for spec in mod.VARIANTS:
        assert spec.proof_mode == "native"
        assert spec.expected_outcome == "PASS"
        assert spec.lanes == ("eks", "hp"), "every 014 variant runs on both clusters"
    for name in ("gpu_plugin", "efa_plugin", "one_sample", "replacement_uid"):
        assert SPECS[name].approvals == (mod.DEVICE_PLUGIN_RESTART,)
        assert mod.NODE_REBOOT not in SPECS[name].approvals
        assert "INCONCLUSIVE" in SPECS[name].expected or name in {
            "one_sample",
            "replacement_uid",
        }
    assert mod.DEVICE_PLUGIN_RESTART in APPROVALS
    assert "nvidia-gpu.sock" in SPECS["gpu_plugin"].expected
    assert "aws-efa-device-plugin.sock" in SPECS["efa_plugin"].expected
    assert mod.GPU_TARGET.liveness_metric in mod.GPU_TARGET.metrics
    assert mod.efa_target().socket_name == "aws-efa-device-plugin.sock"


# --------------------------------------------------------------------------- #
# gpu_plugin / efa_plugin against the liveness-probe product
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "variant,target", [("gpu_plugin", mod.GPU_TARGET), ("efa_plugin", mod.efa_target())]
)
def test_plugin_restart_pass_on_the_liveness_product(
    variant: str, target: mod.PluginTarget
) -> None:
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(target, cluster)
    plugin = _plugin_probe(cluster)
    spec, ctx = _context(variant, cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) >= {
        "detected",
        "restart_chain",
        "deleted_original",
        "replacement_ready",
        "allocatable_full_while_frozen",
        "allocatable_restored",
        "no_reboot",
        "not_isolated",
    }
    receipt = result["restart_receipt"]
    assert receipt["deleted_pod_uid"] == U0
    assert receipt["restart_reason"] == mod.UNRESPONSIVE_FAILURE_MODE
    assert receipt["replacement_pod_uids"] == [U1]
    assert receipt["already_healthy"] is False
    assert receipt["delete_pass_seen"] is True
    assert result["liveness_sample"]["labels"]["sockets"] == target.socket_name
    assert result["generation"]["liveness"] is True
    assert all(
        item["allocatable"] == 8 for item in result["allocatable_while_frozen"]
    ), result["allocatable_while_frozen"]
    assert result["probe_journal_timeouts"] == [FIRST_TIMEOUT]
    # The generation was read before the freeze: feature probe (node) and the
    # control-plane rule probe for this family's metric.
    assert [call[0] for call in plugin.calls][:2] == ["feature", "freeze"]
    assert cluster.feature_probes == [(target.liveness_metric,)]
    assert result["proof_scope"] == "native"


def test_detection_fails_when_the_probe_sample_never_comes() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    # The node carries the probe, the plugin is frozen, yet the product
    # records neither a sample nor a finding.
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        error.startswith("detected:") and "no gpu_device_plugin_unresponsive=1" in error
        for error in result["errors"]
    ), result["errors"]
    assert result["liveness_sample"] is None


def test_detection_fails_on_wrong_sample_labels() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(
        target,
        cluster,
        sample_overrides={
            "transport_answered": "true",
            "sockets": "nvidia-mig-1g.10gb.sock",
            "probe_state": "REFUSED",
        },
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    detected = result["stages"]["detected"]
    assert any("transport_answered" in error for error in detected), detected
    assert any("nvidia-gpu.sock" in error for error in detected), detected
    assert any("probe_state" in error for error in detected), detected


def test_restart_that_reads_the_hung_plugin_as_already_healthy_fails() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    # The restart step confirmed the node on its full allocatable and deleted
    # nothing: the frozen Pod stays, the receipt says already_healthy.
    cluster.activity_responder = _restart_scenario(
        target,
        cluster,
        delete_receipt={"allocatable": 8, "expected": 8, "already_healthy": True},
        final_receipt={
            "allocatable": 8,
            "expected": 8,
            "already_healthy": True,
            "replacement_pod_uids": [U0],
        },
    )
    responder = cluster.activity_responder

    def keep_frozen_pod(arguments: tuple[str, ...]) -> dict[str, Any]:
        doc = responder(arguments)
        cluster.pods = [_pod(U0, ds_uid="ds-uid")]
        return doc

    cluster.activity_responder = keep_frozen_pod
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "already_healthy" in error for error in result["stages"]["replacement_ready"]
    ), result["errors"]
    assert any(
        "still on the node" in error for error in result["stages"]["deleted_original"]
    ), result["errors"]


def test_delete_receipt_without_restart_reason_fails() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(
        target,
        cluster,
        delete_receipt={
            "deleted_pod": "kube-system/plugin-aaaa",
            "deleted_pod_uid": U0,
            "allocatable": 8,
            "expected": 8,
        },
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "restart_reason=None" in error for error in result["stages"]["deleted_original"]
    ), result["errors"]


def test_missed_delete_pass_is_a_note_not_a_failure() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    responder = _restart_scenario(target, cluster)

    def skip_delete_pass(arguments: tuple[str, ...]) -> dict[str, Any]:
        doc = responder(arguments)
        for workflow in doc["workflows"]:
            for execution in workflow["step_executions"]:
                if execution["status"] == "WAITING":
                    # The poll landed on a later waiting pass: no restart_reason.
                    execution["details"] = {
                        "node_results": {
                            NODE: {
                                "allocatable": 8,
                                "expected": 8,
                                "waiting_for_replacement_pod": True,
                                "deleted_pod_uid": U0,
                            }
                        }
                    }
        return doc

    cluster.activity_responder = skip_delete_pass
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert "delete_pass_seen" not in result["restart_receipt"]
    assert any("restart_reason" in note for note in result["notes"]), result["notes"]


def test_allocatable_drop_while_frozen_is_reported() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(target, cluster)
    # kubelet did de-register (not the premise): the stage names it.
    spec, ctx = _context(
        "gpu_plugin", cluster, _plugin_probe(cluster, drop_allocatable=True)
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["allocatable_full_while_frozen"], result["stages"]
    assert "premise" in result["stages"]["allocatable_full_while_frozen"][0]


def test_gpu_plugin_fail_on_reboot() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(
        target, cluster, extra_ops=["RESTART_NODE"]
    )
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("no_reboot" in error for error in result["errors"]), result["errors"]


# --------------------------------------------------------------------------- #
# the earlier product (allocatable-only detection) and mixed generations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["gpu_plugin", "one_sample", "replacement_uid"])
def test_earlier_product_is_inconclusive_when_allocatable_never_drops(
    variant: str,
) -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target, control_plane={"present": False, "rule": False})
    plugin = _plugin_probe(cluster, modules=ABSENT_MODULES)
    spec, ctx = _context(variant, cluster, plugin)
    with pytest.raises(VariantInconclusive) as exc:
        spec.execute(ctx)
    text = str(exc.value)
    assert (
        "ListAndWatch" in text and "predates the device-plugin liveness probe" in text
    )
    assert "R13h" in text
    # The plugin was frozen (and the freeze confirmed) before the verdict.
    assert _calls(plugin, "freeze"), plugin.calls
    assert not any(
        "--resume-after-first-timeout" in call for call in _calls(plugin, "freeze")
    ), "the earlier product has no journal line to resume on"


def test_earlier_product_allocatable_chain_still_passes_when_it_does_drop() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target, control_plane={"present": False, "rule": False})
    cluster.activity_responder = _restart_scenario(target, cluster, liveness=False)
    plugin = _plugin_probe(cluster, modules=ABSENT_MODULES, drop_allocatable=True)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert "detected" not in result["stages"]
    assert result["generation"]["liveness"] is False
    assert result["restart_receipt"]["deleted_pod_uid"] == U0
    assert result["probe_journal_timeouts"] is None


def test_earlier_product_one_sample_after_a_drop_is_inconclusive() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target, control_plane={"present": False, "rule": False})
    plugin = _plugin_probe(cluster, modules=ABSENT_MODULES, drop_allocatable=True)
    spec, ctx = _context("one_sample", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="single-sample window"):
        spec.execute(ctx)
    assert _calls(plugin, "resume"), "the plugin is resumed before the verdict"


def test_mixed_generations_refuse_before_the_freeze() -> None:
    target = mod.GPU_TARGET
    # The node probes but this control plane has no rule for the metric.
    cluster = _cluster(target, control_plane={"present": False, "rule": False})
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="mixed product generations"):
        spec.execute(ctx)
    assert not _calls(plugin, "freeze"), "nothing is frozen on a mixed site"


def test_probe_disabled_by_configuration_refuses_before_the_freeze() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(
        cluster, device_plugin_env={"GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR": ""}
    )
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="disabled by configuration"):
        spec.execute(ctx)
    assert not _calls(plugin, "freeze"), "a disabled probe freezes nothing"


def test_freeze_inconclusive_when_no_process() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster)
    plugin.script["freeze"] = {"frozen": []}
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="froze no plugin process"):
        spec.execute(ctx)


@pytest.mark.parametrize(
    "variant,target", [("gpu_plugin", mod.GPU_TARGET), ("efa_plugin", mod.efa_target())]
)
def test_freeze_inconclusive_immediately_when_not_stopped(
    variant: str, target: mod.PluginTarget
) -> None:
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster)
    # The probe SIGSTOPped the process but it never reached the stopped state:
    # a running plugin keeps answering the product's probe, so the fixture
    # refuses now rather than waiting out the detection window.
    plugin.script["freeze"] = {
        "frozen": [
            {"pid": 4242, "binary": target.binaries[0], "stopped": False, "state": "S"}
        ]
    }
    spec, ctx = _context(variant, cluster, plugin)
    with pytest.raises(VariantInconclusive, match="did not enter the stopped state"):
        spec.execute(ctx)
    assert [call[0] for call in plugin.calls] == ["feature", "freeze"], plugin.calls


def test_plugin_binary_candidates_prefers_real_command_and_image_basenames() -> None:
    pod = {
        "spec": {
            "containers": [
                {
                    # A shell shim entrypoint must not widen the signal set.
                    "command": ["/bin/sh", "-c", "exec efa-k8s-device-plugin-hpd"],
                    "image": "123.dkr.ecr.aws/x/efa-device-plugin@sha256:" + "a" * 64,
                }
            ]
        }
    }
    got = mod.plugin_binary_candidates(pod)
    assert "sh" not in got
    assert "efa-device-plugin" in got


def test_plugin_binary_candidates_from_explicit_command() -> None:
    pod = {
        "spec": {
            "containers": [{"command": ["/opt/aws/aws-efa-k8s-device-plugin-hyperpod"]}]
        }
    }
    assert mod.plugin_binary_candidates(pod) == ("aws-efa-k8s-device-plugin-hyperpod",)


# --------------------------------------------------------------------------- #
# replacement_uid
# --------------------------------------------------------------------------- #
def test_replacement_uid_pass() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    cluster.activity_responder = _restart_scenario(target, cluster)
    spec, ctx = _context("replacement_uid", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert "no_redelete" in result["stages"]
    assert result["final_plugin_pod_uids"] == [U1]


def test_replacement_uid_fail_on_redelete() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    responder = _restart_scenario(target, cluster)

    def redelete(arguments: tuple[str, ...]) -> dict[str, Any]:
        doc = responder(arguments)
        # A third, different Pod is on the node: the product re-deleted the
        # recorded replacement.
        cluster.pods = [_pod(U2, ds_uid="ds-uid")]
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
def test_one_sample_pass() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("one_sample", cluster, plugin)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) == {
        "no_confirmed_sample",
        "no_finding",
        "no_action",
        "not_isolated",
    }
    freeze = _calls(plugin, "freeze")[0]
    assert "--resume-after-first-timeout" in freeze
    # one tick (15 s default) + the 5 s probe timeout + the margin
    assert freeze[freeze.index("--window-seconds") + 1] == str(
        15 + mod.LIVENESS_PROBE_TIMEOUT_SECONDS + mod.FIRST_TIMEOUT_MARGIN_SECONDS
    )
    assert result["first_timeout"]["consecutive"] == 1
    assert result["resume_latency_seconds"] == 0.5
    assert result["probe_journal_timeouts"] == [FIRST_TIMEOUT]
    # The resume happened on the node inside the freeze call; no control-plane
    # round trip was needed.
    assert not _calls(plugin, "resume"), plugin.calls


def test_one_sample_inconclusive_without_a_logged_tick() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster, first_timeout=None)
    spec, ctx = _context("one_sample", cluster, plugin)
    with pytest.raises(
        VariantInconclusive, match="no timed-out GetDevicePluginOptions"
    ):
        spec.execute(ctx)


def test_one_sample_inconclusive_when_the_second_tick_timed_out_first() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster, second_timeout_seen=True)
    spec, ctx = _context("one_sample", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="second consecutive"):
        spec.execute(ctx)


def test_one_sample_inconclusive_when_the_journal_shows_two_ticks() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    second = {**FIRST_TIMEOUT, "consecutive": 2, "realtime_epoch": FROZEN_EPOCH + 32.4}
    plugin = _plugin_probe(cluster, journal_timeouts=[FIRST_TIMEOUT, second])
    spec, ctx = _context("one_sample", cluster, plugin)
    with pytest.raises(VariantInconclusive, match="2 timed-out ticks"):
        spec.execute(ctx)


def test_one_sample_fails_when_the_product_confirms_anyway() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    restart = target.restart_operation
    doc = _doc(
        incidents=[
            _incident(
                "inc-1", f"b0-{target.liveness_metric}-node", "ACTION_PENDING", "wf-1"
            )
        ],
        workflows=[
            _workflow(
                "wf-1", "inc-1", [restart, "TRIGGER_HEALTH_SNAPSHOT"], "RUNNING", []
            )
        ],
        batches=[_liveness_batch(target, 1)],
    )
    cluster.activity_responder = lambda a: doc
    spec, ctx = _context("one_sample", cluster, _plugin_probe(cluster))
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["no_confirmed_sample"], result["stages"]
    assert result["stages"]["no_finding"], result["stages"]
    assert result["stages"]["no_action"], result["stages"]


# --------------------------------------------------------------------------- #
# cleanup refusals
# --------------------------------------------------------------------------- #
def test_cleanup_refuses_when_allocatable_not_restored() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster, drop_allocatable=True)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    fixture.freeze_pod(U0, seconds=120)  # drops allocatable to 0; never restored
    with pytest.raises(RegionalFixtureError, match="allocatable"):
        fixture.restore()


def test_cleanup_refuses_on_extra_pod() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster)
    spec, ctx = _context("gpu_plugin", cluster, plugin)
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    fixture.freeze_pod(U0, seconds=120)
    cluster.pods = [_pod(U0, ds_uid="ds-uid"), _pod(U1, ds_uid="ds-uid")]
    with pytest.raises(RegionalFixtureError, match="exactly one"):
        fixture.restore()


def test_cleanup_refuses_when_probe_not_clean() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    plugin = _plugin_probe(cluster)
    plugin.script["cleanup"] = {
        "residue": {"paused_processes": {"1": {}}},
        "clean": False,
    }
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
    cluster = FakeCluster(resource=target.resource, baseline=8, daemonset=ds, pods=[])
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    with pytest.raises(VariantBlocked):
        fixture.discover()


def test_generation_record_describes_both_sides() -> None:
    target = mod.GPU_TARGET
    cluster = _cluster(target)
    spec, ctx = _context("gpu_plugin", cluster, _plugin_probe(cluster))
    fixture = mod.DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    generation = fixture.generation()
    assert generation.liveness is True
    record = generation.record()
    assert record["node_modules"] == PRESENT_MODULES
    assert record["control_plane"]["rule"] is True
    assert "carries the liveness probe" in record["description"]
    assert "grades gpu_device_plugin_unresponsive" in record["description"]
