"""Shared mechanism for the GF-ENV-LIVE acceptance cases.

One privileged host probe on the approved test node runs the audited loopback
replay proxy (``tools/gpu_environment_replay.py``) and repoints the node's
single metrics collector at it through a bounded single-node maintenance change
with an independent rollback (``probes/env_replay_proxy.py``). The reviewed
plans come from the live device's own limits (``env_live_plans``); the verdicts
come from the product's workflow records (``env_live_verdicts``). This module
owns the read-only preflight, the device-limit read, the proxy lifecycle and the
control-plane reads the cases share; each case's orchestration lives in the
runner, the bounded wait in ``env_live_watch`` and the per-case judgement in
``env_live_judges``, and no function here decides a verdict.

Nothing here declares a LIVE PASS. The replay tool only serves telemetry; the
incident, workflow, notification and cleanup receipts the control plane writes
are the evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.gpu_metrics import GpuMetricsThresholds  # noqa: E402

from scripts.e2e.regional import env_live_plans as plans  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_action_guard import (  # noqa: E402
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.general_live_gpu_store import (  # noqa: E402
    GPU_STATE_PROBE,
)
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE  # noqa: E402
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import add_live_arguments  # noqa: E402
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    project_node_snapshot,
    required,
    settings_from_arguments,
)

#: The audited replay tool and the controller probe this fixture ships.
REPLAY_TOOL = ROOT / "tools" / "gpu_environment_replay.py"
PROBE_SCRIPT = Path(__file__).with_name("probes") / "env_replay_proxy.py"
#: The file names the node side writes the plan and the tool under.
PLAN_FILE = "plan.json"
TOOL_FILE = "gpu_environment_replay.py"
EXPORTER_POD_PREFIX = "gpu-fault-dcgm-exporter"
#: The live T.Limit reading must be at least this far above zero before the run
#: arms, so the device is plainly in a safe state (the tool aborts at <= 0).
SAFE_TLIMIT_MARGIN_C = 10.0
#: One read of the observation loop. A slow API Pod costs one poll, never the
#: replay window: ``pod_python`` never retries a timeout, and the second attempt
#: only covers an exec kubectl refused before the script ran.
READ_TIMEOUT_SECONDS = 45
READ_ATTEMPTS = 2
NODE_READ_TIMEOUT_SECONDS = 30


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EnvLiveSettings:
    regional: RegionalLiveSettings
    case_id: str
    confirmation: str
    node: str
    host_probe_image: str
    gpu_uuid: str
    predecessor_path: Path | None
    predecessor_id: str | None
    site_file: Path | None
    #: The per-branch confirmation tokens the operator typed, carried through so
    #: ``execute_case`` (which the spine calls with settings only) can require
    #: them before an isolating / stopping branch runs.
    isolate_confirm: str = ""
    stop_confirm: str = ""

    def environment(self) -> dict[str, str]:
        value = {
            **self.regional.environment(),
            "GPU_FAULT_ENV_LIVE_CASE": self.case_id,
            "GPU_FAULT_ENV_LIVE_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }
        if self.gpu_uuid:
            value["GPU_FAULT_ENV_LIVE_GPU_UUID"] = self.gpu_uuid
        return value


def add_env_live_arguments(parser: argparse.ArgumentParser, confirmation: str) -> None:
    add_live_arguments(parser, confirmation=confirmation)
    parser.add_argument("--case", required=True)
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--node", default="")
    parser.add_argument("--host-probe-image", default="")
    parser.add_argument("--gpu-uuid", default="")
    parser.add_argument("--site-file", default="")
    parser.add_argument("--predecessor-evidence", default="")
    # Per-branch confirmation tokens the isolate/stop/restart paths demand on
    # top of the per-case ENVLIVE<NNN>_EXECUTE; recorded, checked by the runner.
    parser.add_argument("--confirm-isolate", default="")
    parser.add_argument("--confirm-stop", default="")
    parser.add_argument("--confirm-restart", default="")


def configure(arguments: argparse.Namespace, confirmation: str) -> EnvLiveSettings:
    # GF-ENV-LIVE cases have no formal regional predecessor (they are not in
    # the regional execution order). A predecessor is validated only when the
    # operator passes one explicitly; otherwise it is NOT_REQUIRED.
    explicit = str(getattr(arguments, "predecessor_evidence", "") or "")
    predecessor_id: str | None = None
    path: Path | None = None
    if explicit:
        predecessor_id = "operator-supplied"
        path = Path(explicit).expanduser().resolve()
    image = required(arguments.host_probe_image, "host probe image")
    if "@sha256:" not in image:
        raise RegionalFixtureError("host probe image must use an immutable digest")
    return EnvLiveSettings(
        regional=settings_from_arguments(arguments),
        case_id=arguments.case,
        confirmation=confirmation,
        node=required(arguments.node, "target node"),
        host_probe_image=image,
        gpu_uuid=str(arguments.gpu_uuid or ""),
        predecessor_path=path,
        predecessor_id=predecessor_id,
        site_file=Path(arguments.site_file).expanduser().resolve()
        if arguments.site_file
        else None,
        isolate_confirm=str(getattr(arguments, "confirm_isolate", "") or ""),
        stop_confirm=str(getattr(arguments, "confirm_stop", "") or ""),
    )


# --------------------------------------------------------------------------- #
# Read-only device-limit read (an exec into the node's existing exporter Pod)
# --------------------------------------------------------------------------- #
def _exporter_pod_on_node(regional: RegionalLiveFixture, node: str) -> str:
    value = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "pod",
            "-l",
            "app.kubernetes.io/name=gpu-fault-dcgm-exporter",
            "--field-selector",
            f"spec.nodeName={node},status.phase=Running",
            "-o",
            "json",
        )
    )
    pods = [
        item["metadata"]["name"]
        for item in value.get("items", [])
        if str(item["metadata"].get("name", "")).startswith(EXPORTER_POD_PREFIX)
    ]
    if not pods:
        # Fall back to a name scan: the DaemonSet's label key has varied.
        value = json.loads(
            regional.kubectl(
                "gpu",
                "get",
                "pod",
                "--field-selector",
                f"spec.nodeName={node},status.phase=Running",
                "-o",
                "json",
            )
        )
        pods = [
            item["metadata"]["name"]
            for item in value.get("items", [])
            if str(item["metadata"].get("name", "")).startswith(EXPORTER_POD_PREFIX)
        ]
    if not pods:
        raise RegionalFixtureError(
            f"no running {EXPORTER_POD_PREFIX} Pod on {node} to read device limits"
        )
    return sorted(pods)[0]


def read_device_limits(
    regional: RegionalLiveFixture, node: str
) -> list[dict[str, Any]]:
    """Per-GPU temperature limits parsed with the product's own XML parser.

    Read-only: ``nvidia-smi -q -x`` runs in the node's existing exporter Pod,
    and the XML is parsed by ``query_nvidia_temperature_limits`` through a
    runner that just returns that text, so the scale decision is identical to
    the collector's.
    """

    from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits

    pod = _exporter_pod_on_node(regional, node)
    xml = regional.kubectl(
        "gpu", "exec", pod, "--", "nvidia-smi", "-q", "-x", timeout=90
    )

    def runner(_command: list[str], **_options: Any) -> Any:
        return type("R", (), {"returncode": 0, "stdout": xml, "stderr": ""})()

    by_gpu: dict[str, dict[str, Any]] = {}
    for sample in query_nvidia_temperature_limits(runner):
        key = sample.gpu_uuid or sample.gpu_index or "gpu"
        entry = by_gpu.setdefault(
            key,
            {
                "gpu_uuid": sample.gpu_uuid,
                "gpu_index": sample.gpu_index,
                "pci_bdf": sample.pci_bdf,
                "product": sample.labels.get("modelName"),
                "limits": {},
            },
        )
        entry["limits"][sample.canonical_name] = sample.value
    # The product parser omits the current T.Limit *reading* (<gpu_temp_tlimit>,
    # DCGM field 153): it caches the XML and reads the reading per tick instead.
    # The plan decision needs it -- a device with T.Limit thresholds grades the
    # core on that scale -- and the safety gate needs it to confirm the device
    # is well above 0 before arming. Parse it here from the same XML.
    for uuid, reading in _current_tlimit_readings(xml).items():
        if uuid in by_gpu:
            by_gpu[uuid]["limits"].setdefault("gpu_tlimit_c", reading)
    return [by_gpu[key] for key in sorted(by_gpu)]


def _current_tlimit_readings(xml: str) -> dict[str, float]:
    """``{uuid: <gpu_temp_tlimit> Celsius}`` from one ``nvidia-smi -q -x`` dump."""

    import xml.etree.ElementTree as ET

    readings: dict[str, float] = {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return readings
    for gpu in root.findall(".//gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        text = (gpu.findtext("temperature/gpu_temp_tlimit") or "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if uuid and match:
            readings[uuid] = float(match.group())
    return readings


def select_target_gpu(
    devices: list[dict[str, Any]], requested_uuid: str
) -> dict[str, Any]:
    if not devices:
        raise RegionalFixtureError("no GPU device limits were read from the node")
    if requested_uuid:
        for device in devices:
            if device.get("gpu_uuid") == requested_uuid:
                return device
        raise RegionalFixtureError(f"requested GPU {requested_uuid} is not on the node")
    with_uuid = [device for device in devices if device.get("gpu_uuid")]
    if not with_uuid:
        raise RegionalFixtureError("the node's GPUs report no UUID")
    return sorted(with_uuid, key=lambda item: str(item.get("gpu_index")))[0]


def build_device_plans(
    devices: list[dict[str, Any]],
    *,
    thresholds: GpuMetricsThresholds,
    confirmation_samples: int,
) -> dict[str, plans.DevicePlan]:
    result: dict[str, plans.DevicePlan] = {}
    for device in devices:
        uuid = device.get("gpu_uuid")
        if not uuid:
            continue
        result[uuid] = plans.build_device_plan(
            gpu_uuid=uuid,
            gpu_index=device.get("gpu_index"),
            product=device.get("product"),
            raw_limits=device.get("limits") or {},
            thresholds=thresholds,
            confirmation_samples=confirmation_samples,
        )
    return result


# --------------------------------------------------------------------------- #
# Read-only preflight (used by --plan now and re-run before --execute)
# --------------------------------------------------------------------------- #
def read_only_preflight(settings: EnvLiveSettings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    workloads = regional.business_workloads(settings.node)
    identity = regional.evidence_identity()
    predecessor = (
        predecessor_evidence(
            settings.predecessor_path, settings.predecessor_id, **identity
        )
        if settings.predecessor_id is not None and settings.predecessor_path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    errors: list[str] = []
    if not predecessor["valid"]:
        errors.append(f"{settings.predecessor_id} predecessor evidence is not PASS")
    if node["ready"] != "True":
        errors.append(f"{node['name']} is not Ready")
    if node["ownership_annotations"]:
        errors.append(f"{node['name']} has pre-existing workflow ownership")
    if workloads:
        errors.append(f"{node['name']} has a business workload")
    agent = state.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{node['name']} Node Agent is not ACTIVE")

    devices: list[dict[str, Any]] = []
    device_plans: dict[str, Any] = {}
    target: dict[str, Any] = {}
    thresholds = GpuMetricsThresholds.from_environment()
    confirmation_samples = 3
    try:
        devices = read_device_limits(regional, settings.node)
        built = build_device_plans(
            devices, thresholds=thresholds, confirmation_samples=3
        )
        device_plans = {uuid: device_plan_report(plan) for uuid, plan in built.items()}
        target = select_target_gpu(devices, settings.gpu_uuid)
        reading = (target.get("limits") or {}).get("gpu_tlimit_c")
        if reading is not None and reading <= SAFE_TLIMIT_MARGIN_C:
            errors.append(
                f"target GPU T.Limit reading {reading}C is not safely above 0; "
                "refuse to arm"
            )
    except RegionalFixtureError as exc:
        errors.append(f"device limit read failed: {exc}")

    result = {
        **identity,
        "node": node,
        "store": {
            "agent_state": agent.get("lifecycle_state"),
            "collector_statuses": state.get("collector_statuses"),
            "incident": state.get("incident"),
        },
        "workloads": workloads,
        "predecessor": predecessor,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "devices": devices,
        "device_plans": device_plans,
        "target_gpu": target,
        "confirmation_samples": confirmation_samples,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def device_plan_report(plan: plans.DevicePlan) -> dict[str, Any]:
    return {
        "gpu_index": plan.gpu_index,
        "product": plan.product,
        "limits": plan.limits,
        "tlimit_reading": plan.tlimit_reading,
        "runnable_variants": plan.runnable_variants(),
        "decisions": {
            name: {
                "runnable": decision.runnable,
                "scale": decision.scale,
                "threshold_source": decision.threshold_source,
                "critical_value": decision.critical_value,
                "warning_value": decision.warning_value,
                "reason": decision.reason,
            }
            for name, decision in plan.decisions.items()
        },
    }


# --------------------------------------------------------------------------- #
# Proxy controller (the live path; exercised in execute)
# --------------------------------------------------------------------------- #
class EnvReplayController:
    """Owns the host probe, the proxy and the collector repoint for one node."""

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
        case_dir: Path,
        active_deadline_seconds: int = 3600,
        host: Any | None = None,
    ) -> None:
        self.regional = regional
        self.node = node
        self.run_id = run_id
        self.case_dir = case_dir
        self._repointed = False
        self._armed = False
        # A caller (the GF-GENERAL-LIVE 024 composites, a test) may hand in the
        # host probe; production builds the node-pinned probe Pod itself.
        self.host = host or HostProbeFixture(
            HostProbeSettings(
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=image,
                case_id=case_id,
                run_id=run_id,
                probe_script=PROBE_SCRIPT,
                state_directory=case_dir / "host-probes",
                active_deadline_seconds=active_deadline_seconds,
            )
        )

    def create(self) -> None:
        if getattr(self.host, "created", False):
            return
        require_action_time(180)
        self.host.create()

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        finite_seconds(timeout)
        if arguments[0] in {
            "deploy-proxy",
            "repoint-collector",
            "arm",
            "restore-collector",
        }:
            require_action_time(timeout)
        return self.host.execute(*arguments, timeout=timeout)

    def preflight(self) -> dict[str, Any]:
        return self.host.execute("preflight", timeout=120)

    def install_source(self, name: str, path: Path) -> dict[str, Any]:
        content = path.read_bytes()
        return self.execute(
            "install-source",
            "--run-id",
            self.run_id,
            "--name",
            name,
            "--sha256",
            hashlib.sha256(content).hexdigest(),
            "--b64",
            base64.b64encode(content).decode("ascii"),
            timeout=120,
        )

    def deploy_proxy(
        self,
        plan_path: Path,
        gpu_uuid: str,
        *,
        port: int = 9411,
        max_seconds: int = 600,
        gpu_stop_c: float = 80.0,
        memory_stop_c: float = 85.0,
    ) -> dict[str, Any]:
        self.install_source(PLAN_FILE, plan_path)
        self.install_source(TOOL_FILE, REPLAY_TOOL)
        return self.execute(
            "deploy-proxy",
            "--run-id",
            self.run_id,
            "--gpu-uuid",
            gpu_uuid,
            "--port",
            str(port),
            "--max-seconds",
            str(max_seconds),
            "--gpu-stop-c",
            str(gpu_stop_c),
            "--memory-stop-c",
            str(memory_stop_c),
            timeout=120,
        )

    def repoint(
        self, *, port: int = 9411, restore_seconds: int = 600
    ) -> dict[str, Any]:
        receipt = self.execute(
            "repoint-collector",
            "--run-id",
            self.run_id,
            "--port",
            str(port),
            "--restore-seconds",
            str(restore_seconds),
            timeout=120,
        )
        self._repointed = True
        return receipt

    def status(self, *, port: int = 9411) -> dict[str, Any]:
        return self.host.execute(
            "status", "--run-id", self.run_id, "--port", str(port), timeout=60
        )

    def arm(self, *, port: int = 9411) -> dict[str, Any]:
        receipt = self.execute(
            "arm", "--run-id", self.run_id, "--port", str(port), timeout=60
        )
        self._armed = True
        return receipt

    def restore(self) -> dict[str, Any]:
        receipt = self.execute(
            "restore-collector", "--run-id", self.run_id, timeout=120
        )
        self._repointed = False
        return receipt

    def stop_proxy(self, *, port: int = 9411) -> dict[str, Any]:
        return self.host.execute(
            "stop-proxy", "--run-id", self.run_id, "--port", str(port), timeout=60
        )

    def cleanup(self) -> dict[str, Any]:
        """Restore the endpoint, stop the proxy, delete the probe -- in order.

        The proxy is stopped only after the collector endpoint is back on the
        real exporter, so the single-producer invariant holds throughout.
        """

        receipts: dict[str, Any] = {}
        if self._repointed:
            try:
                receipts["restore"] = self.restore()
            except Exception as exc:  # noqa: BLE001 - recorded; cleanup continues
                receipts["restore_error"] = f"{type(exc).__name__}: {exc}"
        try:
            receipts["stop_proxy"] = self.stop_proxy()
        except Exception as exc:  # noqa: BLE001
            receipts["stop_proxy_error"] = f"{type(exc).__name__}: {exc}"
        receipts["probe_residuals"] = self.host.cleanup()
        return receipts


def write_plan_file(case_dir: Path, name: str, plan: dict[str, Any]) -> Path:
    """Write one reviewed replay plan the controller ships to the node."""

    path = case_dir / name
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return path


def gpu_records(
    regional: RegionalLiveFixture,
    node: str,
    since: datetime,
    names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """The node's GPU telemetry records since ``since`` (read-only).

    The accepted GPU_METRICS batches with the samples of ``names``, the latest
    accepted reading per GPU and field, the node's GPU findings (active and
    activated since) and the GPU collector status -- the reads the
    GF-GENERAL-LIVE DCGM categories judge on (``general_live_gpu_store``).
    """

    wanted = sorted(set(names))
    if any(not item or "," in item for item in wanted):
        raise ValueError("GPU state filters must be non-empty, no comma")
    return regional.cpu_python(
        GPU_STATE_PROBE,
        regional.settings.cluster_id,
        node,
        since.isoformat(),
        ",".join(wanted),
        timeout=READ_TIMEOUT_SECONDS,
        attempts=READ_ATTEMPTS,
    )


def node_records(
    regional: RegionalLiveFixture,
    node: str,
    since: datetime,
) -> dict[str, Any]:
    """The incidents, workflows (with step executions), remote commands and
    notifications the control plane recorded for the node since ``since``.

    ``RegionalLiveFixture.store_snapshot`` cannot answer this: it is keyed on
    an XID event, and a GPU environment finding is a NODE_HEALTH incident
    (``inc-gpu-<finding id>``) that no XID event names, so the snapshot never
    returned the workflow an armed replay produced.
    """

    return regional.cpu_python(
        ACTIVITY_PROBE,
        regional.settings.cluster_id,
        node,
        since.isoformat(),
        "collector_observation_valid",
        "",
        "GPU_METRICS",
        timeout=READ_TIMEOUT_SECONDS,
        attempts=READ_ATTEMPTS,
    )


def node_state(regional: RegionalLiveFixture, node: str) -> dict[str, Any]:
    """The runner's view of the Node (``project_node_snapshot``), bounded."""

    return project_node_snapshot(
        json.loads(
            regional.kubectl(
                "gpu",
                "get",
                "node",
                node,
                "-o",
                "json",
                timeout=NODE_READ_TIMEOUT_SECONDS,
            )
        )
    )


def replay_tool_sha256() -> str:
    return hashlib.sha256(REPLAY_TOOL.read_bytes()).hexdigest()
