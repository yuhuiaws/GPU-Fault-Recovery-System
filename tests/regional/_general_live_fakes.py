"""Fakes the GF-GENERAL-LIVE category tests drive their ``execute`` through.

No cluster, no node, no systemd: a ``FakeProbe`` answers the host-probe
subcommands from a scripted table and records every call, a ``FakeRegional``
answers the read-only store/kubectl reads the runner makes, and
``make_context`` wires them into a real ``VariantContext`` with a real
``Evidence`` under ``tmp_path`` and a no-op sleep. A test scripts the product's
records (accepted batches, incidents, workflows with step executions, health
signals) so the verdict helpers run over realistic documents; ``activity_read``
runs the real node-activity probe over an in-memory store when a test must
know what that probe lists rather than script it.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
import types
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.e2e.regional.general_live_fixture import (
    Evidence,
    GeneralLiveSettings,
    VariantContext,
)
from scripts.e2e.regional.general_live_registry import VariantSpec
from scripts.e2e.regional.general_live_store import (
    ACTIVITY_PROBE,
    ATTEMPT_PROBE,
    PREFLIGHT_PROBE,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings


class FakeProbe:
    """One scripted host probe. ``script[subcommand]`` is a value or a callable.

    A callable receives the full argument tuple and returns the JSON object the
    probe would print. A list value is consumed one call at a time; the last
    entry repeats. A missing subcommand returns ``{}``.
    """

    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, ...]] = []
        self.created = True
        self._iterators: dict[str, Any] = {}

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments)
        key = arguments[0]
        value = self.script.get(key, {})
        if callable(value):
            return value(arguments)
        if isinstance(value, list):
            iterator = self._iterators.setdefault(key, iter(value))
            try:
                return dict(next(iterator))
            except StopIteration:
                return dict(value[-1]) if value else {}
        return dict(value)

    def cleanup(self) -> dict[str, bool]:
        return {"pod": False, "configmap": False}


class FakeRegional:
    """The read-only RegionalLiveFixture surface the runner uses in tests."""

    def __init__(
        self,
        *,
        cluster_id: str = "gpu-eks-cluster",
        node_snapshot: dict[str, Any] | None = None,
        node_metadata: dict[str, Any] | None = None,
        business: Sequence[dict[str, Any]] = (),
        gpu_pods: Sequence[dict[str, Any]] = (),
        release_id: str = "rel-test",
        cpu_python: Callable[..., dict[str, Any]] | None = None,
        run_output: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.settings = RegionalLiveSettings(
            cpu_kubeconfig=_touch_kubeconfig("cpu"),
            gpu_kubeconfig=_touch_kubeconfig("gpu"),
            gpu_context="ctx",
            namespace="gpu-fault-system",
            cluster_id=cluster_id,
            region="us-west-2",
        )
        self.node_snapshot_value = node_snapshot or _default_node()
        self._node_metadata = node_metadata or {
            "name": self.node_snapshot_value["name"],
            "labels": {"eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group"},
        }
        self._business = list(business)
        self._gpu_pods = list(gpu_pods)
        self._release_id = release_id
        self.cpu_python_responder = cpu_python
        self._run_output = run_output

    def evidence_identity(self, **_: Any) -> dict[str, str]:
        return {"release_id": self._release_id, "cluster_id": self.settings.cluster_id}

    def release_id(self, **_: Any) -> str:
        return self._release_id

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return dict(self.node_snapshot_value)

    def node_metadata(self, node: str) -> dict[str, Any]:
        return dict(self._node_metadata)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return list(self._business)

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return list(self._gpu_pods)

    def cpu_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        if self.cpu_python_responder is None:
            raise AssertionError("cpu_python was called without a script")
        return self.cpu_python_responder(script, *arguments)

    def run(self, command: list[str], **_: Any) -> Any:
        output = self._run_output(command) if self._run_output else "null"

        class Completed:
            stdout = output

        return Completed()


def _touch_kubeconfig(name: str) -> Path:
    path = Path("/tmp/pt-gl1") / f"fake-{name}.kubeconfig"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    return path


def _default_node() -> dict[str, Any]:
    return {
        "name": "ip-10-0-43-5.us-west-2.compute.internal",
        "uid": "node-uid",
        "boot_id": "boot-1",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "spare_pool_annotations": {},
        "gpu_allocatable": "8",
    }


def make_settings(
    spec: VariantSpec, *, node: str = "", peer: str = ""
) -> GeneralLiveSettings:
    node = node or _default_node()["name"]
    return GeneralLiveSettings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=_touch_kubeconfig("cpu"),
            gpu_kubeconfig=_touch_kubeconfig("gpu"),
            gpu_context="ctx",
            namespace="gpu-fault-system",
            cluster_id="gpu-eks-cluster",
            region="us-west-2",
        ),
        case_id=spec.case_id,
        variant=spec.variant,
        node=node,
        peer_node=peer,
        host_probe_image="public.ecr.aws/x@sha256:" + "0" * 64,
        hyperpod_cluster="hp-cluster",
        site_file=None,
        approvals=spec.approvals,
    )


def make_context(
    spec: VariantSpec,
    *,
    tmp_path: Path,
    regional: FakeRegional,
    probes: dict[str, FakeProbe],
    settings: GeneralLiveSettings | None = None,
    window_minutes: int = 120,
) -> VariantContext:
    settings = settings or make_settings(spec, node=regional.node_snapshot("")["name"])
    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    counter = itertools.count()
    case_dir = tmp_path / "cases" / spec.key
    ctx = VariantContext(
        spec=spec,
        settings=settings,
        regional=regional,  # type: ignore[arg-type]
        case_dir=case_dir,
        run_id=f"gl-{spec.variant.replace('_', '-')}-a1-deadbeef",
        attempt=1,
        deadline=now + timedelta(minutes=window_minutes),
        evidence=Evidence(case_dir),
        clock=lambda: now + timedelta(seconds=next(counter)),
        sleep=lambda _seconds: None,
    )
    for key, probe in probes.items():
        ctx.probes[key] = probe  # type: ignore[assignment]
    return ctx


def node_probe(ctx: VariantContext, probe: FakeProbe) -> None:
    ctx.probes[f"node@{ctx.settings.node}"] = probe  # type: ignore[assignment]


def store_script(responses: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """A ``cpu_python`` that answers each probe script by its constant name.

    ``responses`` maps ``"activity"``, ``"preflight"``, ``"attempt"`` to a value
    or a callable taking the probe arguments; a list is consumed per call.
    """

    iterators: dict[str, Any] = {}

    def dispatch(script: str, *arguments: str) -> dict[str, Any]:
        name = {
            ACTIVITY_PROBE: "activity",
            PREFLIGHT_PROBE: "preflight",
            ATTEMPT_PROBE: "attempt",
        }.get(script, "unknown")
        value = responses.get(name, {})
        if callable(value):
            return value(arguments)
        if isinstance(value, list):
            iterator = iterators.setdefault(name, iter(value))
            try:
                return dict(next(iterator))
            except StopIteration:
                return dict(value[-1]) if value else {}
        return dict(value)

    return dispatch


ACTIVITY_CODE = compile(ACTIVITY_PROBE, "<general-live-activity-probe>", "exec")


def activity_read(store: Any, arguments: Sequence[str]) -> dict[str, Any]:
    """Run the real ``ACTIVITY_PROBE`` in-process over ``store``.

    ``arguments`` are what ``general_live_store.node_activity`` passes the
    probe in the API Pod; the probe reads ``store`` through the product's
    own store API and prints the document a variant receives. A test that
    needs to know what the probe lists (every incident state, the older
    incident a later workflow pulls in) runs it instead of scripting it.
    """

    from gpu_fault.app import ApplicationContext

    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["activity-probe", *arguments]),
        contextlib.redirect_stdout(output),
    ):
        exec(ACTIVITY_CODE, {"__name__": "__main__"})
    return json.loads(output.getvalue())


def save_diagnostic_incident(
    store: Any,
    event_id: str,
    metric: str,
    *,
    cluster_id: str,
    node: str,
    created: datetime,
    reason: str,
) -> str:
    """Store a diagnostic-only NODE_HEALTH incident as the product leaves it.

    ``inc-<event id>`` closed RECOVERED "diagnostic inconclusive" behind a
    FAILED FREEZE_EVIDENCE + VALIDATE_HOST workflow whose validation failed
    on ``metric``'s host diagnostic probe, all written at ``created``
    (detailed design 6.3; live 2026-10-25: LOW_MEMORY_AVAILABLE and the
    co-firing memory_used_percent, created and RECOVERED within one second).
    Returns the incident id.
    """

    from gpu_fault.models import (
        IncidentState,
        RecoveryAction,
        WorkflowOperation,
        WorkflowStatus,
        WorkflowStepStatus,
    )
    from tests._builders import (
        fault_incident,
        workflow_request,
        workflow_step,
        workflow_step_execution,
    )

    incident_id = f"inc-{event_id}"
    request_id = f"wf-{event_id}"
    probe = {"metric": metric, "node_id": node, "device": None}
    store.save_incident_and_workflow(
        fault_incident(
            incident_id,
            event_id,
            "NODE_HEALTH",
            cluster_id,
            node_ids=[node],
            policy_source="SITE_HOST_RESOURCE_HEALTH",
            effective_action=RecoveryAction.RUN_DIAGNOSTICS,
            state=IncidentState.RECOVERED,
            workflow_request_id=request_id,
            reasons=[reason, "diagnostic inconclusive: validation failed"],
            created_at=created,
            updated_at=created,
        ),
        workflow_request(
            request_id,
            incident_id,
            WorkflowStatus.FAILED,
            official_steps=[
                workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=[node]),
                workflow_step(
                    WorkflowOperation.VALIDATE_HOST,
                    node_ids=[node],
                    parameters={"host_diagnostic_probe": probe},
                ),
            ],
            step_executions=[
                workflow_step_execution(
                    1,
                    WorkflowOperation.VALIDATE_HOST,
                    WorkflowStepStatus.FAILED,
                    details={
                        "host_diagnostic_failure": True,
                        "node_failures": {node: [f"host_diagnostic:{metric}"]},
                    },
                )
            ],
            created_at=created,
            updated_at=created,
        ),
    )
    return incident_id


def host_batch(
    observed_at: str, metric: str, value: float, *, device: str | None = None
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": ["cpu_usage_percent"],
        "collection_errors": [],
        "samples": [{"name": metric, "device": device, "value": value, "labels": {}}],
    }


def incident(
    incident_id: str,
    metric: str,
    *,
    device: str = "node",
    rule_id: str | None = None,
    state: str = "ACTION_PENDING",
    workflow: str = "wf-1",
) -> dict[str, Any]:
    from scripts.e2e.regional.general_live_verdicts import safe_device

    event = f"batch0-{metric}-{safe_device(device)}" + (
        f"-{rule_id.lower()}" if rule_id else ""
    )
    return {
        "incident_id": incident_id,
        "event_id": event,
        "event_type": "NODE_HEALTH",
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": ["ip-10-0-43-5.us-west-2.compute.internal"],
    }


def workflow(
    request_id: str,
    incident_id: str,
    operations: Sequence[str],
    *,
    status: str = "RUNNING",
    executions: Sequence[dict[str, Any]] = (),
    probe_metric: str | None = None,
) -> dict[str, Any]:
    steps = []
    for operation in operations:
        parameters: dict[str, Any] = {}
        if probe_metric and operation.startswith("VALIDATE_"):
            parameters["host_diagnostic_probe"] = {"metric": probe_metric}
        steps.append({"operation": operation, "parameters": parameters})
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": status,
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": steps,
        "step_executions": list(executions),
    }
