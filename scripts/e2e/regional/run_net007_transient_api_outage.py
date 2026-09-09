#!/usr/bin/env python3
"""GF-REGIONAL-NET-007: a transient Kubernetes API failure makes an isolation
step WAIT, not FAIL, and the workflow closes once it clears.

Order matters:

* The deadman Job is created *before* the webhook, so a runner that dies the
  moment after the webhook exists still has something that removes it.
* The webhook is created *before* the fault is injected: the isolation step
  must meet the 500 on its first attempt, not on a retry the runner happened
  to catch.
* The outage is held for ``OUTAGE_SECONDS`` after the first observed
  retryable WAITING (the EFA finding reaches the control plane on the host
  collector's summary cadence, so a window fixed to the injection time could
  close before the step ever ran). If no isolation step reaches the webhook
  within ``EVIDENCE_TIMEOUT_SECONDS`` the webhook is removed and the case
  fails with that reason.
* The webhook is removed first in cleanup -- before anything else waits on
  the control plane -- and the EFA function is restored in its own ``finally``
  exactly as COLLECT-017 A does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import net007_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import run_collect017_efa_plugin as c017  # noqa: E402
from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    required,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
EXECUTOR_DEPLOYMENT = "gpu-fault-cluster-executor"
MIN_SERVER_MINOR_FOR_MATCH_CONDITIONS = 30
SAMPLE_INTERVAL_SECONDS = 5.0

# Runs inside the deadman Job with the executor image, which carries the
# kubernetes client. It deletes the webhook when its deadline passes and
# treats an already-deleted webhook (the runner did its job) as success.
DEADMAN_SCRIPT = r"""
import os
import time

from kubernetes import client, config

deadline = time.monotonic() + float(os.environ["NET007_DEADMAN_SECONDS"])
name = os.environ["NET007_WEBHOOK_NAME"]
while time.monotonic() < deadline:
    time.sleep(5)
config.load_incluster_config()
api = client.AdmissionregistrationV1Api()
try:
    api.delete_validating_webhook_configuration(name)
    print("deadman deleted", name, flush=True)
except client.exceptions.ApiException as exc:
    if exc.status != 404:
        raise
    print("deadman found nothing to delete", name, flush=True)
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    site_file: Path
    host_probe_image: str
    outage_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_NET007_OUTAGE_SECONDS": str(self.outage_seconds),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    outage = int(arguments.outage_seconds)
    if not verdicts.MIN_OUTAGE_SECONDS <= outage <= verdicts.MAX_OUTAGE_SECONDS:
        raise RegionalFixtureError(
            f"outage seconds is outside {verdicts.MIN_OUTAGE_SECONDS}.."
            f"{verdicts.MAX_OUTAGE_SECONDS}"
        )
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""), "target node"
        ),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        outage_seconds=outage,
    )


def run_identity(run_dir: Path, attempt: int) -> str:
    return f"net007-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"


def resource_names(run_id: str) -> dict[str, str]:
    stem = f"gpu-fault-acceptance-{run_id}"
    return {
        "webhook": stem,
        "service_account": f"{stem}-deadman",
        "cluster_role": f"{stem}-deadman",
        "cluster_role_binding": f"{stem}-deadman",
        "job": f"{stem}-deadman",
    }


# --------------------------------------------------------------------------- #
# Read-only preflight
# --------------------------------------------------------------------------- #
def executor_identity(regional: RegionalLiveFixture) -> dict[str, str]:
    document = json.loads(
        regional.kubectl("gpu", "get", "deployment", EXECUTOR_DEPLOYMENT, "-o", "json")
    )
    spec = document["spec"]["template"]["spec"]
    containers = spec.get("containers") or []
    return {
        "service_account": str(spec.get("serviceAccountName") or "default"),
        "image": str(containers[0]["image"]) if containers else "",
    }


def server_minor(regional: RegionalLiveFixture) -> int:
    document = json.loads(regional.kubectl("gpu", "version", "-o", "json"))
    minor = str((document.get("serverVersion") or {}).get("minor") or "0")
    digits = "".join(ch for ch in minor if ch.isdigit())
    return int(digits or 0)


def executor_has_classifier(regional: RegionalLiveFixture) -> bool:
    """Whether the deployed GPU-plane executor routes adapter exceptions
    through ``retryable_adapter_error`` (the spec's stated precondition)."""

    output = regional.kubectl(
        "gpu",
        "exec",
        f"deployment/{EXECUTOR_DEPLOYMENT}",
        "--",
        "python",
        "-c",
        (
            "import json, gpu_fault.cluster_executor as m; "
            "print(json.dumps({'classifier': 'retryable_adapter_error' in "
            "open(m.__file__, encoding='utf-8').read()}))"
        ),
        timeout=60,
    )
    return bool(json.loads(output.strip().splitlines()[-1]).get("classifier"))


def can_i(regional: RegionalLiveFixture, verb: str, resource: str) -> bool:
    output = regional.kubectl(
        "gpu", "auth", "can-i", verb, resource, check=False, timeout=60
    )
    return output.strip().lower().startswith("yes")


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/execution/test_retryable_adapter_errors.py::"
        "test_transient_adapter_error_becomes_a_waiting_step_not_a_failure",
        "tests/regional/test_net007_transient_api_outage.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=300)
    (case_dir / "focused-tests.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    return {"passed": completed.returncode == 0, "command": command}


def preflight_errors(
    settings: Settings,
    *,
    node: dict[str, Any],
    workloads: list[dict[str, Any]],
    state: dict[str, Any],
    executor: dict[str, str],
    minor: int,
    classifier: bool,
    existing_webhook: str,
    permissions: dict[str, bool],
    tests_passed: bool,
) -> list[str]:
    errors: list[str] = []
    if node.get("ready") != "True" or node.get("unschedulable"):
        errors.append("target node is not Ready and schedulable")
    if node.get("ownership_annotations"):
        errors.append("target node has pre-existing workflow ownership")
    if (node.get("labels") or {}).get("kubernetes.io/hostname") != settings.node:
        errors.append(
            "target node's kubernetes.io/hostname label differs from its name; "
            "the objectSelector could not pin it"
        )
    if workloads:
        errors.append("target node has a non-system workload")
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not executor.get("service_account"):
        errors.append("cluster executor ServiceAccount is unknown")
    if not executor.get("image"):
        errors.append("cluster executor image is unknown")
    if minor < MIN_SERVER_MINOR_FOR_MATCH_CONDITIONS:
        errors.append(
            f"server minor {minor} < {MIN_SERVER_MINOR_FOR_MATCH_CONDITIONS}: "
            "matchConditions are not guaranteed"
        )
    if not classifier:
        errors.append(
            "deployed cluster executor lacks the retryable-adapter-error "
            "classifier; record NOT_RUN instead of executing"
        )
    if existing_webhook:
        errors.append(
            f"a webhook of this run's name already exists: {existing_webhook}"
        )
    denied = sorted(name for name, allowed in permissions.items() if not allowed)
    if denied:
        errors.append(f"kubeconfig may not: {denied}")
    if not settings.site_file.is_file():
        errors.append("regional site file does not exist")
    if not tests_passed:
        errors.append("focused regression tests failed")
    return errors


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    workloads = regional.business_workloads(settings.node)
    state = regional.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc),
    )
    executor = executor_identity(regional)
    minor = server_minor(regional)
    classifier = executor_has_classifier(regional)
    names = resource_names(run_identity(case_dir.parent.parent, 1))
    existing = regional.kubectl(
        "gpu",
        "get",
        "validatingwebhookconfiguration",
        names["webhook"],
        "--ignore-not-found",
        "-o",
        "name",
        check=False,
    ).strip()
    permissions = {
        "create validatingwebhookconfigurations": can_i(
            regional, "create", "validatingwebhookconfigurations"
        ),
        "delete validatingwebhookconfigurations": can_i(
            regional, "delete", "validatingwebhookconfigurations"
        ),
        "create clusterroles": can_i(regional, "create", "clusterroles"),
        "create clusterrolebindings": can_i(regional, "create", "clusterrolebindings"),
        "create jobs": can_i(regional, "create", "jobs"),
        "create serviceaccounts": can_i(regional, "create", "serviceaccounts"),
    }
    tests = focused_tests(case_dir)
    errors = preflight_errors(
        settings,
        node=node,
        workloads=workloads,
        state=state,
        executor=executor,
        minor=minor,
        classifier=classifier,
        existing_webhook=existing,
        permissions=permissions,
        tests_passed=tests["passed"],
    )
    result = {
        "release_id": state.get("release_id") or regional.release_id(),
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "executor": executor,
        "server_minor": minor,
        "executor_has_classifier": classifier,
        "permissions": permissions,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-node-mutation",
        "target_node": settings.node,
        "mutation": (
            "one ValidatingWebhookConfiguration scoped to the target node, the "
            "executor ServiceAccount and node UPDATE (failurePolicy Fail, 1 s "
            "timeout, absent Service) held for the outage; one EFA function "
            "unbound with a 900 s bind fail-safe; a deadman Job that removes the "
            "webhook if the runner dies; no reset, no reboot"
        ),
        "outage_seconds": settings.outage_seconds,
        "evidence_timeout_seconds": verdicts.EVIDENCE_TIMEOUT_SECONDS,
        "executor": preflight.get("executor"),
        "preflight_identity": {
            "release_id": preflight.get("release_id"),
            "node_uid": (preflight.get("node") or {}).get("uid"),
            "executor_image": (preflight.get("executor") or {}).get("image"),
        },
        "stop_conditions": verdicts.stop_conditions(),
        "rollback": {
            "webhook_deleted_first_in_cleanup": True,
            "deadman_job_deletes_webhook_if_runner_dies": True,
            "EFA_has_host_side_auto_bind": True,
        },
        "preflight": preflight,
    }


# --------------------------------------------------------------------------- #
# Outage resources
# --------------------------------------------------------------------------- #
def rbac_manifests(names: dict[str, str], *, namespace: str, run_id: str) -> list[dict]:
    labels = {
        "gpu-fault.io/acceptance-case": CASE_ID,
        "gpu-fault.io/acceptance-run": run_id,
    }
    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": names["service_account"],
                "namespace": namespace,
                "labels": labels,
            },
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": names["cluster_role"], "labels": labels},
            "rules": [
                {
                    "apiGroups": ["admissionregistration.k8s.io"],
                    "resources": ["validatingwebhookconfigurations"],
                    "resourceNames": [names["webhook"]],
                    "verbs": ["get", "delete"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": names["cluster_role_binding"], "labels": labels},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": names["cluster_role"],
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": names["service_account"],
                    "namespace": namespace,
                }
            ],
        },
    ]


def deadman_manifest(
    names: dict[str, str],
    *,
    namespace: str,
    image: str,
    deadman_seconds: int,
    run_id: str,
) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": names["job"],
            "namespace": namespace,
            "labels": {
                "gpu-fault.io/acceptance-case": CASE_ID,
                "gpu-fault.io/acceptance-run": run_id,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": deadman_seconds + 120,
            "template": {
                "metadata": {"labels": {"gpu-fault.io/acceptance-case": CASE_ID}},
                "spec": {
                    "serviceAccountName": names["service_account"],
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "deadman",
                            "image": image,
                            "command": ["python", "-c", DEADMAN_SCRIPT],
                            "env": [
                                {
                                    "name": "NET007_DEADMAN_SECONDS",
                                    "value": str(deadman_seconds),
                                },
                                {
                                    "name": "NET007_WEBHOOK_NAME",
                                    "value": names["webhook"],
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "128Mi"},
                                "limits": {"cpu": "200m", "memory": "256Mi"},
                            },
                        }
                    ],
                },
            },
        },
    }


class OutageFixture:
    """Creates, verifies and removes the webhook and its deadman."""

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        names: dict[str, str],
        node: str,
        username: str,
        image: str,
        run_id: str,
        deadman_seconds: int,
    ) -> None:
        self.regional = regional
        self.names = names
        self.node = node
        self.username = username
        self.image = image
        self.run_id = run_id
        self.deadman_seconds = deadman_seconds
        self.webhook_created = False

    @property
    def namespace(self) -> str:
        return self.regional.settings.namespace

    def _apply(self, manifest: dict[str, Any]) -> None:
        self.regional.kubectl(
            "gpu", "apply", "-f", "-", input_text=json.dumps(manifest)
        )

    def arm_deadman(self) -> None:
        for manifest in rbac_manifests(
            self.names, namespace=self.namespace, run_id=self.run_id
        ):
            self._apply(manifest)
        self._apply(
            deadman_manifest(
                self.names,
                namespace=self.namespace,
                image=self.image,
                deadman_seconds=self.deadman_seconds,
                run_id=self.run_id,
            )
        )
        # The deadman must be running before the webhook exists.
        self.regional.kubectl(
            "gpu",
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            f"job-name={self.names['job']}",
            "--timeout=180s",
            timeout=210,
        )

    def open(self) -> dict[str, Any]:
        self._apply(
            verdicts.webhook_manifest(
                name=self.names["webhook"],
                node=self.node,
                namespace=self.namespace,
                username=self.username,
                run_id=self.run_id,
            )
        )
        self.webhook_created = True
        applied = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "validatingwebhookconfiguration",
                self.names["webhook"],
                "-o",
                "json",
            )
        )
        errors = verdicts.webhook_errors(
            applied, node=self.node, username=self.username
        )
        if errors:
            self.close()
            raise RegionalFixtureError(
                "the server did not keep the webhook's scoping; refusing to run "
                "an unscoped outage: " + "; ".join(errors)
            )
        return cast(dict[str, Any], applied)

    def close(self) -> str:
        output = self.regional.kubectl(
            "gpu",
            "delete",
            "validatingwebhookconfiguration",
            self.names["webhook"],
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=120,
        )
        self.webhook_created = False
        return output.strip()

    def deadman_log(self) -> str:
        return self.regional.kubectl(
            "gpu",
            "logs",
            "-l",
            f"job-name={self.names['job']}",
            "--tail=20",
            check=False,
            timeout=60,
        ).strip()

    def cleanup(self) -> dict[str, bool]:
        self.close()
        for kind, name in (
            ("job", self.names["job"]),
            ("clusterrolebinding", self.names["cluster_role_binding"]),
            ("clusterrole", self.names["cluster_role"]),
            ("serviceaccount", self.names["service_account"]),
        ):
            self.regional.kubectl(
                "gpu",
                "delete",
                kind,
                name,
                "--ignore-not-found",
                "--wait=true",
                check=False,
                timeout=120,
            )
        deadline = time.monotonic() + 60
        residuals = self.residuals()
        while any(residuals.values()) and time.monotonic() < deadline:
            time.sleep(2)
            residuals = self.residuals()
        return residuals

    def residuals(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for kind, name in (
            ("validatingwebhookconfiguration", self.names["webhook"]),
            ("job", self.names["job"]),
            ("clusterrolebinding", self.names["cluster_role_binding"]),
            ("clusterrole", self.names["cluster_role"]),
            ("serviceaccount", self.names["service_account"]),
        ):
            present = self.regional.kubectl(
                "gpu",
                "get",
                kind,
                name,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            ).strip()
            result[f"{kind}/{name}"] = bool(present)
        return result


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
@dataclass
class _Run:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    attempt: int
    run_id: str
    preflight: dict[str, Any]
    collector: CollectorAcceptanceFixture
    outage: OutageFixture
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    base_settings: Any = None
    bdf: str = ""
    baseline_inventory: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    webhook_closed_at: datetime | None = None
    injected_at: datetime | None = None


def _base_settings(settings: Settings) -> Any:
    return base.Settings(
        regional=settings.regional,
        case_id=CASE_ID,
        node=settings.node,
        second_node=None,
        host_probe_image=settings.host_probe_image,
        hyperpod_cluster="",
        executor_role_arn="",
        site_file=settings.site_file,
        predecessor_path=settings.site_file,
    )


def _bound_efa_devices(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in inventory.get("devices") or []
        if item.get("pci_bdf") and item.get("driver") == "efa"
    ]


def _wait_efa_inventory(
    collector: CollectorAcceptanceFixture,
    *,
    discovered_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = collector.snapshot()["efa_inventory"]
        if int(last.get("discovered_count") or -1) == discovered_count:
            return cast(dict[str, Any], last)
        time.sleep(5)
    raise RegionalFixtureError(
        f"EFA inventory did not reach {discovered_count} functions: {last}"
    )


def _sample(run: _Run) -> dict[str, Any]:
    """One instant of the control plane: the node's newest workflow since the
    injection and that workflow's remote commands."""

    state = run.regional.cpu_python(
        base.LATEST_NODE_WORKFLOW,
        run.settings.regional.cluster_id,
        run.settings.node,
        run.injected_at.isoformat() if run.injected_at else run.started_at.isoformat(),
    )
    matches = state.get("matches") or []
    sample: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "workflow_request_id": None,
        "workflow_status": None,
        "step_executions": [],
        "remote_commands": [],
    }
    if not matches:
        return sample
    workflow = matches[0].get("workflow") or {}
    sample["workflow_request_id"] = workflow.get("request_id")
    sample["workflow_status"] = workflow.get("status")
    sample["step_executions"] = [
        {
            "step_index": item.get("step_index"),
            "operation": item.get("operation"),
            "status": item.get("status"),
            "details": {
                key: value
                for key, value in (item.get("details") or {}).items()
                if key
                in {
                    "retryable_adapter_error",
                    "retryable_transport_error",
                    "reason",
                    "attempt",
                    "remote_command_id",
                    "remote_status",
                }
            },
        }
        for item in workflow.get("step_executions") or []
    ]
    if workflow.get("request_id"):
        commands = run.regional.cpu_python(
            c017.REMOTE_COMMANDS, str(workflow["request_id"])
        )
        sample["remote_commands"] = commands.get("remote_commands") or []
    return sample


def _hold_outage(run: _Run) -> None:
    """Sample until an isolation step is seen riding out the 500, then keep the
    webhook for the configured outage, then remove it."""

    evidence_deadline = time.monotonic() + verdicts.EVIDENCE_TIMEOUT_SECONDS
    hold_until: float | None = None
    while True:
        sample = _sample(run)
        run.samples.append(sample)
        write_json_atomic(
            run.case_dir / "outage-samples.json", {"samples": run.samples}
        )
        if run.evidence is None:
            run.evidence = verdicts.outage_evidence(run.samples)
            if run.evidence is not None:
                hold_until = time.monotonic() + run.settings.outage_seconds
        if hold_until is not None and time.monotonic() >= hold_until:
            break
        if run.evidence is None and time.monotonic() >= evidence_deadline:
            break
        if sample.get("workflow_status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            # The workflow ended before any isolation step met the webhook (or
            # after); holding the outage longer cannot produce evidence.
            break
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    run.webhook_closed_at = datetime.now(timezone.utc)
    run.outage.close()


def _inject_and_observe(run: _Run) -> dict[str, Any]:
    baseline = run.collector.snapshot()["efa_inventory"]
    bound = _bound_efa_devices(baseline)
    if not bound:
        raise RegionalFixtureError("no bound EFA BDF was discovered")
    run.bdf = str(bound[0]["pci_bdf"])
    run.baseline_inventory = baseline
    run.injected_at = datetime.now(timezone.utc)
    injection = run.collector.execute(
        "unbind-efa",
        "--run-id",
        run.run_id,
        "--pci-bdf",
        run.bdf,
        "--restore-seconds",
        str(verdicts.EFA_RESTORE_SECONDS),
    )
    write_json_atomic(run.case_dir / "injection.json", injection)
    unbound: dict[str, Any] = {}
    recovered: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    try:
        unbound = _wait_efa_inventory(
            run.collector,
            discovered_count=int(baseline["discovered_count"]) - 1,
            timeout_seconds=60,
        )
        _hold_outage(run)
        bundle = base.latest_node_workflow(
            run.regional,
            run.base_settings,
            observed_after=run.injected_at,
            timeout_seconds=verdicts.WORKFLOW_TIMEOUT_SECONDS,
        )
        deadline = time.monotonic() + verdicts.INCIDENT_TIMEOUT_SECONDS
        while (bundle.get("incident") or {}).get(
            "state"
        ) != "RECOVERED" and time.monotonic() < deadline:
            time.sleep(5)
            bundle = base.latest_node_workflow(
                run.regional,
                run.base_settings,
                observed_after=run.injected_at,
                timeout_seconds=30,
            )
        recovered = _wait_efa_inventory(
            run.collector,
            discovered_count=int(baseline["discovered_count"]),
            timeout_seconds=30,
        )
        request_id = str((bundle.get("workflow") or {}).get("request_id") or "")
        if request_id:
            bundle = {
                **bundle,
                **run.regional.cpu_python(c017.REMOTE_COMMANDS, request_id),
            }
    finally:
        if run.outage.webhook_created:
            run.webhook_closed_at = run.webhook_closed_at or datetime.now(timezone.utc)
            run.outage.close()
        restore = run.collector.execute(
            "restore-efa", "--run-id", run.run_id, "--pci-bdf", run.bdf
        )
    write_json_atomic(run.case_dir / "workflow-state.json", bundle)
    errors = list(verdicts.outage_errors(run.samples))
    errors.extend(
        verdicts.recovery_errors(
            bundle,
            waited_operation=(run.evidence or {}).get("operation"),
        )
    )
    errors.extend(
        c017.efa_unbind_errors(
            bundle,
            node=run.settings.node,
            bdf=run.bdf,
            baseline=baseline,
            unbound=unbound,
            recovered=recovered,
            restore=restore,
        )
    )
    return {
        "errors": errors,
        "efa_bdf": run.bdf,
        "evidence": run.evidence,
        "samples": len(run.samples),
        "webhook_closed_at": (
            run.webhook_closed_at.isoformat() if run.webhook_closed_at else None
        ),
        "baseline_inventory": baseline,
        "unbound_inventory": unbound,
        "recovered_inventory": recovered,
        "restore": restore,
        "workflow": bundle,
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    run_id = run_identity(run_dir, attempt)
    names = resource_names(run_id)
    executor = preflight["executor"]
    run = _Run(
        settings=settings,
        regional=regional,
        case_dir=case_dir,
        attempt=attempt,
        run_id=run_id,
        preflight=preflight,
        collector=CollectorAcceptanceFixture(
            regional,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
        ),
        outage=OutageFixture(
            regional,
            names=names,
            node=settings.node,
            username=verdicts.executor_username(
                settings.regional.namespace, executor["service_account"]
            ),
            image=executor["image"],
            run_id=run_id,
            deadman_seconds=(
                verdicts.EVIDENCE_TIMEOUT_SECONDS
                + settings.outage_seconds
                + verdicts.DEADMAN_GRACE_SECONDS
            ),
        ),
    )
    run.base_settings = _base_settings(settings)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
        "resources": names,
        "executor": executor,
    }
    try:
        run.collector.create()
        run.outage.arm_deadman()
        result["webhook"] = run.outage.open()
        write_json_atomic(case_dir / "webhook.json", result["webhook"])
        observed = _inject_and_observe(run)
        result["errors"].extend(observed["errors"])
        result["observed"] = {k: v for k, v in observed.items() if k != "workflow"}
        result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup: dict[str, Any] = {"errors": []}
        try:
            cleanup["deadman_log"] = run.outage.deadman_log()
            residuals = run.outage.cleanup()
            cleanup["outage_residuals"] = residuals
            cleanup["errors"].extend(verdicts.residual_errors(residuals))
        except Exception as exc:
            cleanup["errors"].append(f"outage cleanup: {type(exc).__name__}: {exc}")
        try:
            probe = run.collector.cleanup()
            cleanup["probe_residuals"] = probe
            if any(probe.values()):
                cleanup["errors"].append("collector probe resources remain")
        except Exception as exc:
            cleanup["errors"].append(f"probe cleanup: {type(exc).__name__}: {exc}")
        try:
            final = regional.node_snapshot(settings.node)
            write_json_atomic(case_dir / "node-final.json", final)
            cleanup["errors"].extend(verdicts.node_final_errors(final))
        except Exception as exc:
            cleanup["errors"].append(f"final node: {type(exc).__name__}: {exc}")
        try:
            events = regional.provider_events(
                run.started_at, datetime.now(timezone.utc)
            )
            write_json_atomic(case_dir / "provider-events.json", {"events": events})
            cleanup["errors"].extend(verdicts.provider_errors(events))
        except Exception as exc:
            cleanup["errors"].append(f"provider events: {type(exc).__name__}: {exc}")
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["errors"].extend(cleanup["errors"])
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run GF-REGIONAL-NET-007: a transient Kubernetes API failure makes "
            "an isolation step wait, not fail."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument(
        "--outage-seconds",
        type=int,
        default=verdicts.OUTAGE_SECONDS,
        help=(
            "how long the webhook stays after the first retryable WAITING is "
            f"observed ({verdicts.MIN_OUTAGE_SECONDS}..{verdicts.MAX_OUTAGE_SECONDS})"
        ),
    )
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
