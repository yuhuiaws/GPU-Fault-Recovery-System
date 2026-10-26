"""Shared scaffolding of ``run_general_live.py``.

The parts every GF-GENERAL-LIVE variant needs and no verdict depends on: the
settings and command line, the read-only preflight, node-pinned host probes
(``host_probe_fixture.py``) for the node, storage, network and collector-window
probe scripts, the LIFO cleanup stack every mutation registers its undo on,
the evidence writer and the bounded wait loops.

Nothing here decides a verdict; the category modules do that on the documents
these functions return, and ``general_live_verdicts.py`` holds the shared
chain checks.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.collector_action_guard import (
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.general_live_registry import (
    APPROVALS,
    VariantSpec,
)
from scripts.e2e.regional.general_live_store import preflight_state
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import add_live_arguments
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
    required,
    settings_from_arguments,
)

PROBES = Path(__file__).with_name("probes")
NODE_PROBE = PROBES / "general_live_node_probe.py"
#: The node probe's second file: the pause commands, shipped through its
#: ``install-source`` (sha256-checked) once per run and node before the first pause.
PAUSE_MODULE = PROBES / "general_live_node_pause.py"
STORAGE_PROBE = PROBES / "general_live_storage_probe.py"
NET_PROBE = PROBES / "general_live_net_probe.py"
WINDOW_PROBE = PROBES / "collector_window_probe.py"
DCGM_PROBE = PROBES / "general_live_dcgm_probe.py"
DEPS_PROBE = PROBES / "general_live_deps_probe.py"
SHARED_PROBE = PROBES / "general_live_shared_probe.py"
#: The 005 test export server image comes from the command line or this
#: variable (an immutable digest, like the host probe image).
EXPORT_IMAGE_ENV = "GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE"
#: Mount propagation for each probe's ``/host`` hostPath. Only the storage
#: probe creates a test loop/dm filesystem that the installed host collector
#: must statvfs and the ``systemd-run`` I/O writer must write to; both run in
#: the host root mount namespace, so that Pod propagates its mounts out of its
#: own namespace (``Bidirectional``). Every other probe only reads, but it must
#: not keep a *private* copy of the host's mounts either: a Pod started with the
#: default private propagation after the storage probe mounted its test device
#: (the window probe, 2026-10-26) holds that mount for its lifetime, pins the
#: superblock the host unmount is meant to end, and keeps the mapping open, so
#: the host's mounts flow in and out of the read-only probes (``HostToContainer``).
PROBE_HOST_ROOT_PROPAGATION = {"storage": "Bidirectional"}
READ_ONLY_PROBE_HOST_ROOT_PROPAGATION = "HostToContainer"
HOST_UNIT = "gpu-fault-host-collector.service"
#: The channels a host variant needs fresh before it starts; the node's own
#: collector status is the product's record, not the runner's oracle.
REQUIRED_FRESH_CHANNELS = ("HOST_TELEMETRY", "GPU_INVENTORY", "GPU_METRICS")
FRESH_SUCCESS_SECONDS = 420
HYPERPOD_LABEL = "sagemaker.amazonaws.com/compute-type"
EKS_NODEGROUP_LABEL = "eks.amazonaws.com/nodegroup"
QUARANTINE_PREFIX = "gpu-fault.io/"


class VariantNotRun(Exception):
    """Raised before any mutation: the variant cannot be executed here."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class VariantBlocked(VariantNotRun):
    """A site prerequisite is missing (BLOCKED rather than NOT_RUN)."""


class VariantInconclusive(Exception):
    """The effectiveness oracle did not see the raw source change."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_identity(case_id: str, variant: str, attempt: int, run_dir: Path) -> str:
    """A run id unique per run directory, stable between plan and execute.

    Round run directories end in the same lane token every round; the digest
    of the resolved directory keeps two rounds from sharing on-node state.
    """

    number = case_id.rsplit("-", 1)[-1]
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    stem = variant.replace("_", "-")[:28].strip("-")
    value = f"gl{number}-{stem}-a{int(attempt)}-{digest}"
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", value) is None:
        raise RegionalFixtureError(f"cannot derive a safe run id from {variant!r}")
    return value


@dataclass(frozen=True)
class GeneralLiveSettings:
    regional: RegionalLiveSettings
    case_id: str
    variant: str
    node: str
    peer_node: str
    host_probe_image: str
    hyperpod_cluster: str
    site_file: Path | None
    approvals: tuple[str, ...]
    #: GF-GENERAL-LIVE-005: the test export server image and an optional node
    #: pin for its Pod (default: any node but the client nodes).
    export_image: str = ""
    export_node: str = ""
    #: GF-GENERAL-LIVE-030: the idle GPU node the variant declares as the warm
    #: spare for its duration; empty means the lane's ``--node-b``.
    spare_candidate: str = ""

    @property
    def key(self) -> str:
        return f"{self.case_id}.{self.variant}"

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_GENERAL_LIVE_VARIANT": self.key,
            "GPU_FAULT_GENERAL_LIVE_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_live_arguments(parser, confirmation="GENERALLIVE<NNN>_EXECUTE")
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--approve",
        action="append",
        default=[],
        choices=sorted(APPROVALS),
        help="approval token for a dangerous branch; bound into the plan",
    )
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--fault-node", default="", help="the variant's target node")
    parser.add_argument("--node-b", default="", help="second node of a pair variant")
    parser.add_argument("--host-probe-image", default="")
    parser.add_argument("--hyperpod-cluster", default="")
    parser.add_argument("--site-file", default="")
    parser.add_argument(
        "--export-image",
        default="",
        help=(
            "005 test export server image (immutable digest); default "
            f"${EXPORT_IMAGE_ENV}"
        ),
    )
    parser.add_argument(
        "--export-node",
        default="",
        help="pin the 005 test export server Pod to this node (default: not a client node)",
    )
    parser.add_argument(
        "--spare-candidate",
        default="",
        help=(
            "030: the idle GPU node to declare as the warm spare for the variant "
            "(default: --node-b, the lane's second node); released in cleanup"
        ),
    )


def configure(arguments: argparse.Namespace) -> GeneralLiveSettings:
    image = required(
        arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        "host probe image",
    )
    if "@sha256:" not in image:
        raise RegionalFixtureError("host probe image must use an immutable digest")
    site_file = Path(arguments.site_file).expanduser() if arguments.site_file else None
    return GeneralLiveSettings(
        regional=settings_from_arguments(arguments),
        case_id=str(arguments.case),
        variant=str(arguments.variant),
        node=required(arguments.fault_node, "target node (--fault-node)"),
        peer_node=str(arguments.node_b or ""),
        host_probe_image=image,
        hyperpod_cluster=str(arguments.hyperpod_cluster or ""),
        site_file=site_file,
        approvals=tuple(sorted(set(arguments.approve))),
        export_image=str(
            getattr(arguments, "export_image", "") or os.getenv(EXPORT_IMAGE_ENV, "")
        ),
        export_node=str(getattr(arguments, "export_node", "") or ""),
        spare_candidate=str(getattr(arguments, "spare_candidate", "") or ""),
    )


def node_provider(labels: dict[str, Any]) -> str | None:
    if labels.get(HYPERPOD_LABEL) == "hyperpod":
        return "hp"
    if labels.get(EKS_NODEGROUP_LABEL):
        return "eks"
    return None


def channel_errors(statuses: list[dict[str, Any]], now: datetime) -> list[str]:
    """Why the node's required collector channels are not fresh and clean."""

    by_channel = {str(item.get("collector")): item for item in statuses}
    errors = []
    for channel in REQUIRED_FRESH_CHANNELS:
        item = by_channel.get(channel)
        if item is None or not item.get("last_success_at"):
            errors.append(f"{channel} has no successful observation")
            continue
        success = datetime.fromisoformat(
            str(item["last_success_at"]).replace("Z", "+00:00")
        )
        age = (now - success).total_seconds()
        if age > FRESH_SUCCESS_SECONDS:
            errors.append(f"{channel} last succeeded {age:.0f}s ago")
        last_error = item.get("last_error_at")
        if (
            last_error
            and datetime.fromisoformat(str(last_error).replace("Z", "+00:00"))
            >= success
        ):
            errors.append(f"{channel} reports a current collection error")
    return errors


def hyperpod_node_recovery(regional: RegionalLiveFixture, cluster: str) -> str | None:
    """``NodeRecovery`` of the HyperPod cluster (read-only describe)."""

    output = regional.run(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            cluster,
            "--region",
            regional.settings.region,
            "--query",
            "NodeRecovery",
            "--output",
            "json",
        ],
        timeout=120,
    ).stdout
    value = json.loads(output or "null")
    return str(value) if value is not None else None


def read_only_preflight(
    settings: GeneralLiveSettings,
    spec: VariantSpec,
    regional: RegionalLiveFixture,
) -> dict[str, Any]:
    """What must hold before the variant may run; nothing here mutates."""

    identity = regional.evidence_identity()
    planned: str | None = None
    if spec.deferred:
        planned = f"NOT_RUN({spec.deferred.split(':', 1)[0]})"
        return {**identity, "planned_verdict": planned, "errors": []}
    missing = [item for item in spec.approvals if item not in settings.approvals]
    if missing:
        planned = "NOT_RUN(NOT_AUTHORIZED)"
    errors: list[str] = []
    node = regional.node_snapshot(settings.node)
    metadata = regional.node_metadata(settings.node)
    provider = node_provider(metadata.get("labels") or {})
    if provider not in spec.lanes:
        errors.append(
            f"{settings.node} is a {provider or 'unknown'} node; {spec.key} runs on "
            f"{'/'.join(spec.lanes)}"
        )
    if node["ready"] != "True":
        errors.append(f"{settings.node} is not Ready")
    if node["unschedulable"]:
        errors.append(f"{settings.node} is cordoned")
    if node["ownership_annotations"]:
        errors.append(f"{settings.node} has gpu-fault ownership annotations")
    held = [
        item
        for item in node["taints"]
        if str(item.get("key", "")).startswith(QUARANTINE_PREFIX)
    ]
    if held:
        errors.append(f"{settings.node} carries gpu-fault taints")
    workloads = regional.business_workloads(settings.node)
    if workloads:
        errors.append(f"{settings.node} runs a business workload")
    gpu_pods = [
        item for item in regional.gpu_workloads() if item["node"] == settings.node
    ]
    if gpu_pods:
        errors.append(f"{settings.node} has GPU-requesting Pods")
    store = preflight_state(regional, settings.regional.cluster_id, settings.node)
    agent = store.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{settings.node} Node Agent is not ACTIVE")
    if store.get("open_incidents"):
        errors.append(f"{settings.node} has open incidents")
    if store.get("active_workflows"):
        errors.append(f"{settings.node} has active workflows")
    errors.extend(
        channel_errors(list(store.get("collector_statuses") or []), utc_now())
    )
    recovery = None
    if provider == "hp":
        if not settings.hyperpod_cluster:
            errors.append("--hyperpod-cluster is required on HyperPod")
        else:
            recovery = hyperpod_node_recovery(regional, settings.hyperpod_cluster)
            if recovery not in {None, "None"}:
                errors.append(
                    f"HyperPod NodeRecovery is {recovery}; it would race the product"
                )
    if settings.peer_node and settings.peer_node == settings.node:
        errors.append("the pair node equals the target node")
    if spec.preflight is not None:
        # The variant's own read-only prerequisites, with the fixture so a hook
        # can read the cluster (030 proves its spare candidate idle).
        errors.extend(spec.preflight(settings, regional))
    return {
        **identity,
        "planned_verdict": planned,
        "missing_approvals": missing,
        "provider": provider,
        "node": node,
        "store": store,
        "node_recovery": recovery,
        "workloads": workloads,
        "gpu_pods": gpu_pods,
        "errors": errors,
    }


class CleanupStack:
    """LIFO undo actions; every failure is kept and fails the variant."""

    def __init__(self) -> None:
        self._actions: list[tuple[str, Callable[[], Any]]] = []
        self.results: list[dict[str, Any]] = []

    def push(self, name: str, action: Callable[[], Any]) -> None:
        self._actions.append((name, action))

    def discard(self, name: str) -> None:
        """Forget an undo the variant already performed itself."""

        for index in range(len(self._actions) - 1, -1, -1):
            if self._actions[index][0] == name:
                del self._actions[index]
                return

    def pending(self) -> list[str]:
        return [name for name, _ in self._actions]

    def run(self) -> list[dict[str, Any]]:
        while self._actions:
            name, action = self._actions.pop()
            try:
                value = action()
                self.results.append({"name": name, "ok": True, "result": value})
            except BaseException as exc:  # noqa: BLE001 - every undo is attempted
                self.results.append(
                    {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                if not isinstance(exc, Exception):
                    self._actions.clear()
                    raise
        return self.results

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.results if not item["ok"]]


class Evidence:
    """Files of one variant run under ``cases/<key>/``."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir
        case_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, document: Any) -> Path:
        path = self.case_dir / f"{name}.json"
        write_json_atomic(path, document)
        return path

    def append_oracle(self, records: list[dict[str, Any]], *, source: str) -> None:
        path = self.case_dir / "oracle.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"source": source, **record}, sort_keys=True, default=str
                    )
                    + "\n"
                )


class ProbeHandle:
    """One node-pinned host probe Pod for one probe script, created lazily."""

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        script: Path,
        node: str,
        image: str,
        key: str,
        run_id: str,
        case_dir: Path,
        deadline_seconds: int,
        host_root_propagation: str = "None",
    ) -> None:
        self.script = script
        self.fixture = HostProbeFixture(
            HostProbeSettings(
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=image,
                case_id=key,
                run_id=run_id,
                probe_script=script,
                state_directory=case_dir / "host-probes",
                active_deadline_seconds=max(600, min(7200, deadline_seconds)),
                host_root_propagation=host_root_propagation,
            )
        )
        self.created = False

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        finite_seconds(timeout)
        if not self.created:
            require_action_time(180)
            self.fixture.create()
            self.created = True
        return self.fixture.execute(*arguments, timeout=timeout)

    def cleanup(self) -> dict[str, bool]:
        if not self.created:
            return self.fixture.residuals()
        return self.fixture.cleanup()


@dataclass
class VariantContext:
    """What a category module's ``execute`` receives."""

    spec: VariantSpec
    settings: GeneralLiveSettings
    regional: RegionalLiveFixture
    case_dir: Path
    run_id: str
    attempt: int
    deadline: datetime
    evidence: Evidence
    cleanup: CleanupStack = field(default_factory=CleanupStack)
    probes: dict[str, ProbeHandle] = field(default_factory=dict)
    survey: dict[str, Any] = field(default_factory=dict)
    #: ``<file>@<node>`` of every second node-side file installed in this run.
    shipped: set[str] = field(default_factory=set)
    clock: Callable[[], datetime] = utc_now
    sleep: Callable[[float], None] = time.sleep

    def probe(self, name: str, *, node: str | None = None) -> ProbeHandle:
        """The probe Pod running ``name`` (node, storage, net, window, dcgm,
        deps, shared)."""

        script = {
            "node": NODE_PROBE,
            "storage": STORAGE_PROBE,
            "net": NET_PROBE,
            "window": WINDOW_PROBE,
            "dcgm": DCGM_PROBE,
            "deps": DEPS_PROBE,
            "shared": SHARED_PROBE,
        }[name]
        target = node or self.settings.node
        handle_key = f"{name}@{target}"
        handle = self.probes.get(handle_key)
        if handle is None:
            handle = ProbeHandle(
                self.regional,
                script=script,
                node=target,
                image=self.settings.host_probe_image,
                key=self.spec.key,
                run_id=self.run_id,
                case_dir=self.case_dir,
                deadline_seconds=self.spec.max_minutes * 60 + 1200,
                host_root_propagation=PROBE_HOST_ROOT_PROPAGATION.get(
                    name, READ_ONLY_PROBE_HOST_ROOT_PROPAGATION
                ),
            )
            self.probes[handle_key] = handle
        return handle

    def node(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return self.probe("node").execute(
            *arguments, "--run-id", self.run_id, timeout=timeout
        )

    def pause(
        self, *arguments: str, node: str | None = None, timeout: int = 180
    ) -> dict[str, Any]:
        """A producer-pause command on ``node`` (``unit-pause``,
        ``exporter-pause``, ``log-collector-enable``).

        Their arm side is the node probe's second file (a probe ships as one
        file): it is installed sha256-checked once per run and node before the
        first pause; the resumes, ``residue`` and ``cleanup`` need no shipping.
        """

        handle = self.probe("node", node=node)
        key = f"{PAUSE_MODULE.name}@{node or self.settings.node}"
        if key not in self.shipped:
            content = PAUSE_MODULE.read_bytes()
            handle.execute(
                "install-source",
                "--name",
                PAUSE_MODULE.name,
                "--sha256",
                hashlib.sha256(content).hexdigest(),
                "--b64",
                base64.b64encode(content).decode("ascii"),
                "--run-id",
                self.run_id,
                timeout=120,
            )
            self.shipped.add(key)
        return handle.execute(*arguments, "--run-id", self.run_id, timeout=timeout)

    def storage(self, *arguments: str, timeout: int = 300) -> dict[str, Any]:
        return self.probe("storage").execute(
            *arguments, "--run-id", self.run_id, timeout=timeout
        )

    def net(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return self.probe("net").execute(
            *arguments, "--run-id", self.run_id, timeout=timeout
        )

    def window(self, *arguments: str, timeout: int = 300) -> dict[str, Any]:
        return self.probe("window").execute(*arguments, timeout=timeout)

    def dcgm(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        """The DCGM field / inventory replay controller (``--run-id`` added).

        A failed call is re-raised with the subcommand in front: the host
        fixture withholds the probe's output, so the subcommand name is the
        runner's only record of which step raised.
        """

        try:
            return self.probe("dcgm").execute(
                *arguments, "--run-id", self.run_id, timeout=timeout
            )
        except HostProbeError as exc:
            subcommand = arguments[0] if arguments else "?"
            raise type(exc)(f"dcgm {subcommand}: {exc}") from exc

    def deps(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        """The temporary test package probe (``--run-id`` added)."""

        return self.probe("deps").execute(
            *arguments, "--run-id", self.run_id, timeout=timeout
        )

    def shared(
        self, *arguments: str, node: str | None = None, timeout: int = 180
    ) -> dict[str, Any]:
        """The 005 shared-mount probe on ``node`` (``--run-id`` added)."""

        return self.probe("shared", node=node).execute(
            *arguments, "--run-id", self.run_id, timeout=timeout
        )

    def collector_env(self) -> dict[str, Any]:
        return dict(self.survey.get("collector_env") or {})

    def setting(self, key: str, default: int) -> int:
        value = self.collector_env().get(key)
        try:
            return int(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    @property
    def host_interval(self) -> int:
        return self.setting("GPU_FAULT_HOST_INTERVAL_SECONDS", 15)

    @property
    def host_summary(self) -> int:
        return self.setting("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", 300)

    @property
    def log_summary(self) -> int:
        """``GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS``: the node log collector's
        periodic post -- a health summary when the poll read cleanly, its
        collection errors otherwise -- and, apart from a matching log line, the
        only time it posts at all (``collectors/logs/node.py`` ``collect_once``)."""

        return self.setting("GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS", 300)

    @property
    def log_interval(self) -> int:
        """``GPU_FAULT_LOG_INTERVAL_SECONDS``: the node log collector's poll."""

        return self.setting("GPU_FAULT_LOG_INTERVAL_SECONDS", 10)

    def remaining_seconds(self) -> float:
        return (self.deadline - self.clock()).total_seconds()

    def wait_until(
        self,
        accept: Callable[[], Any],
        *,
        timeout_seconds: float,
        poll_seconds: float,
        name: str,
    ) -> Any:
        """Poll ``accept`` until it returns a value that is not ``None``.

        The deadline runs on ``self.clock`` so a test with a stubbed ``sleep``
        and a virtual clock terminates after one or two polls instead of
        busy-spinning for the real timeout; in production the clock is the wall
        clock, which is the same deadline the operator's window imposes.
        """

        start = self.clock()
        budget = finite_seconds(timeout_seconds)
        finite_seconds(poll_seconds, label="poll interval")
        timeline: list[dict[str, Any]] = []
        while True:
            value = accept()
            now = self.clock()
            timeline.append(
                {"observed_at": now.isoformat(), "accepted": value is not None}
            )
            self.evidence.write(f"{name}-timeline", {"entries": timeline})
            if value is not None or (now - start).total_seconds() >= budget:
                return value
            self.sleep(poll_seconds)

    def hold(
        self, seconds: float, *, name: str, check: Callable[[], Any] | None = None
    ) -> list[Any]:
        """Sleep ``seconds`` in poll steps on ``self.clock``, calling ``check``."""

        start = self.clock()
        budget = finite_seconds(seconds)
        observations: list[Any] = []
        while (self.clock() - start).total_seconds() < budget:
            require_action_time(60)
            if check is not None:
                observations.append(check())
                self.evidence.write(f"{name}-hold", {"observations": observations})
            self.sleep(min(30.0, budget))
        return observations


def backing_root(survey: dict[str, Any]) -> str:
    """The allow-listed backing directory with the most free space."""

    best: tuple[float, str] | None = None
    for item in survey.get("backing") or []:
        reading = item.get("statvfs") or {}
        available = reading.get("available_bytes")
        used = reading.get("used_percent")
        if not isinstance(available, (int, float)) or not isinstance(
            used, (int, float)
        ):
            continue
        if used > 70 or available < 8 * 1024**3:
            continue
        mount = item.get("mount") or {}
        if str(mount.get("fstype")) in {"tmpfs", "overlay", "nfs", "nfs4", "fuse"}:
            continue
        if best is None or available > best[0]:
            best = (float(available), str(item["root"]))
    if best is None:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "no allow-listed backing directory with 8 GiB free"
        )
    return best[1]


def require_tools(survey: dict[str, Any], *names: str) -> None:
    tools = survey.get("tools") or {}
    missing = [name for name in names if not tools.get(name)]
    if missing:
        raise VariantNotRun("CAPABILITY_ABSENT", "the node lacks " + ", ".join(missing))


def require_module(
    survey: dict[str, Any], name: str, approvals: tuple[str, ...]
) -> bool:
    """Whether ``name`` is loaded; NOT_RUN if it would have to be loaded unapproved."""

    module = (survey.get("modules") or {}).get(name) or {}
    if module.get("loaded"):
        return False
    if not module.get("available"):
        raise VariantNotRun(
            "CAPABILITY_ABSENT", f"kernel module {name} is not available"
        )
    if "KERNEL_MODULE_LOAD" not in approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED",
            f"kernel module {name} is not loaded; needs KERNEL_MODULE_LOAD",
        )
    return True


__all__ = [
    "CleanupStack",
    "EXPORT_IMAGE_ENV",
    "Evidence",
    "GeneralLiveSettings",
    "HOST_UNIT",
    "ProbeHandle",
    "VariantBlocked",
    "VariantContext",
    "VariantInconclusive",
    "VariantNotRun",
    "add_arguments",
    "backing_root",
    "channel_errors",
    "configure",
    "hyperpod_node_recovery",
    "node_provider",
    "read_only_preflight",
    "require_module",
    "require_tools",
    "run_identity",
    "utc_now",
]
