"""GF-GENERAL-LIVE-014: Kubernetes device-plugin registration faults (native).

A device-plugin fault must be a *real* one: the spec (014) is explicit that
patching ``Node.status`` or faking allocatable is only a source replay, so these
are ``native``. The one native injection that does not remove the DaemonSet
from the node (the COLLECT-017 B/C fixture does, and restores it *before* the
product acts, so the product takes the ``already_healthy`` branch of
``adapters/kubernetes/node_operations`` and never deletes a Pod) is a SIGSTOP of
the plugin container's own process (``probes/general_live_device_plugin_probe.py``).

kubelet never notices a frozen plugin: its ``ListAndWatch`` stream has no
deadline and no keepalive (``pkg/kubelet/cm/devicemanager/plugin/v1beta1/
client.go``), the stopped process keeps the socket open, so the Pod stays
``Running`` and allocatable stays full (live 2026-10-27, HyperPod ``gpu_plugin``:
``nvidia.com/gpu`` 8 for four frozen minutes -- the harness's earlier assumption
that kubelet de-registers the resource was wrong). The product's host collector
does notice: it probes every plugin socket with a bounded
``GetDevicePluginOptions`` each tick (``collectors/host/device_plugin.py``) and
raises ``<gpu|efa>_device_plugin_unresponsive`` = 1 after two timed-out ticks;
the CRITICAL finding plans ``RESTART_{GPU,EFA}_DEVICE_PLUGIN`` stamped
``plugin_unresponsive``, so the restart deletes the Pod although the node
advertises every device (receipt ``restart_reason=DEVICE_PLUGIN_UNRESPONSIVE``)
and confirms only a Ready Pod of another UID (``already_healthy=false``). The
effectiveness oracle is the probe sample, never an allocatable drop.

Two product generations. :class:`ProductGeneration` is decided *before* the
freeze: the node's installed host-collector package carries the probe modules
(the probe's ``feature``) and the control plane's rule table knows the metric
(``LIVENESS_FEATURE_PROBE`` in an API Pod). Against the liveness-probe product
the variants assert the chain above; against the earlier product (allocatable-
only detection) the freeze is confirmed, allocatable is watched, and when it
does not drop the variant is ``INCONCLUSIVE`` with the mechanism cited -- a
product that lacks the feature by design is not FAILed.

Variants: ``gpu_plugin`` / ``efa_plugin`` (the full detection -> restart ->
validate chain with the receipts above, allocatable never below the baseline
while frozen, incident RECOVERED, no reboot, no isolation); ``one_sample`` (the
probe SIGCONTs the plugin the instant the host collector logs its *first*
timed-out tick, before the second: no finding, no action); ``replacement_uid``
(the GPU restart graded on the replacement identity: recorded, new, not
re-deleted); ``validation_failure`` -- NOT_RUN(CAPABILITY_ABSENT): the restart
of a hung plugin fails only when kubelet is itself blocked in the frozen
plugin's ``Allocate`` (``VALIDATION_FAILURE_DEFERRED``), which nothing the
harness may do to a live shared node produces. ``expected_unknown`` /
``mig_timeslicing`` / ``running_training`` stay deferred in
``general_live_inventory``. Every variant is registered on both clusters:
nothing in the freeze or in the product's probe and restart is
provider-specific -- the EFA plugin's DaemonSet name follows the target node's
provider (``general_live_device_plugin_targets``), never this process's
environment -- and the lane preflight requires ``NodeRecovery=None`` on
HyperPod.

Reusable entry points: the probe's ``freeze`` / ``resume`` / ``cleanup`` /
``feature`` / ``probe-log`` subcommands and
:class:`DevicePluginRegistrationFixture` (discover, ``generation``,
``freeze_pod``, ``allocatable``, ``plugin_pods_on_node``, ``probe_log``,
``restore``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gpu_fault.adapters.kubernetes.node_operations import (
    companion_pod,
    pod_terminating,
)
from scripts.e2e.regional.general_live_common import activity, run_survey
from scripts.e2e.regional.general_live_device_plugin_targets import (
    GPU_TARGET,
    PluginTarget,
    efa_target,
    plugin_binary_candidates,
    plugin_daemonset,
    target_cluster_provider,
)
from scripts.e2e.regional.general_live_fixture import (
    PROBES,
    ProbeHandle,
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    operations,
    parse_time,
    samples,
    settled,
    verdict,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CASE_014 = "GF-GENERAL-LIVE-014"

#: The approval the product's device-plugin pod-delete/re-register branch needs,
#: and the separate one a post-restart reboot escalation (``RESTART_NODE``)
#: would need -- the ``validation_failure`` path this phase cannot produce.
DEVICE_PLUGIN_RESTART = "DEVICE_PLUGIN_RESTART"
NODE_REBOOT = "NODE_REBOOT"

DEVICE_PLUGIN_PROBE = PROBES / "general_live_device_plugin_probe.py"

#: Where the product's own records show the deleted / replacement Pod UIDs:
#: ``node_operations`` writes them into the ``RESTART_*`` step's
#: ``details.node_results.<node>``.
NODE_RESULTS_KEY = "node_results"

#: The default plugin-restart restore timeout the product waits for the
#: replacement before failing the step (``node_operations._restart_device_plugin``
#: ``restart_timeout_seconds``).
RESTART_TIMEOUT_SECONDS = 180

#: The product's liveness-probe defaults (``device_plugin_liveness.py``): one
#: ``GetDevicePluginOptions`` is bounded by 5 s, two consecutive timed-out
#: ticks open the finding, and the sample then carries this ``failure_mode``.
LIVENESS_PROBE_TIMEOUT_SECONDS = 5
LIVENESS_REQUIRED_SAMPLES = 2
UNRESPONSIVE_FAILURE_MODE = "DEVICE_PLUGIN_UNRESPONSIVE"
#: The host collector's environment knob that disables the probe when empty.
SOCKET_DIR_ENV = "GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR"
#: Ingest and dispatch on top of the two ticks plus the timeout the detection
#: takes; the first-timeout window of ``one_sample`` on top of one tick plus the
#: timeout; the negative watch after the resume.
DETECTION_MARGIN_SECONDS = 60
FIRST_TIMEOUT_MARGIN_SECONDS = 20
NEGATIVE_MARGIN_SECONDS = 60

#: Read-only, in an API Pod: whether the control plane's rule table knows the
#: liveness metric (b204 ``device_plugin_liveness.kubernetes_device_plugin_rules``
#: feeds ``host_health.NodeHealthPolicy.METRIC_RULES``). A release without the
#: module answers ``present: false``.
LIVENESS_FEATURE_PROBE = r"""
import json
import sys

metric = sys.argv[1]
try:
    from gpu_fault.device_plugin_liveness import DEVICE_PLUGIN_UNRESPONSIVE_METRICS
    from gpu_fault.host_health import NodeHealthPolicy
except ImportError as exc:
    print(json.dumps({"present": False, "rule": False, "error": str(exc)[:200]}))
else:
    rule = NodeHealthPolicy.METRIC_RULES.get(metric)
    print(json.dumps({
        "present": metric in DEVICE_PLUGIN_UNRESPONSIVE_METRICS,
        "rule": rule is not None,
        "severity": getattr(rule[2], "value", None) if rule else None,
        "action": getattr(rule[3], "value", None) if rule else None,
    }))
"""

#: Why ``validation_failure`` is NOT_RUN on both product generations.
VALIDATION_FAILURE_DEFERRED = (
    "CAPABILITY_ABSENT: a hung plugin's RESTART_*_DEVICE_PLUGIN fails only when "
    "kubelet is itself blocked in the frozen plugin's Allocate "
    "(pkg/kubelet/cm/devicemanager/endpoint.go, context.Background(), no timeout) "
    "and so neither kills the frozen container nor starts the replacement; "
    "producing that needs a Pod requesting the resource admitted on the node while "
    "the plugin is frozen, which wedges kubelet's sync loop for every Pod on the "
    "node until the product's reboot -- not producible safely on a live shared "
    "node. A re-frozen replacement no longer fails the chain (Ready with the full "
    "allocatable it is confirmed; the next probe tick opens a new finding), and a "
    "replacement that never becomes Ready cannot be produced without excluding the "
    "DaemonSet (the COLLECT-017 path on a cluster-wide object the other lane shares)"
)


# --------------------------------------------------------------------------- #
# Product generation
# --------------------------------------------------------------------------- #
@dataclass
class ProductGeneration:
    """Which product the variant faces, decided before the freeze.

    ``node_modules`` is what the node's installed host-collector package
    carries (the probe's ``feature``); ``control_plane`` is the API Pod's answer
    to ``LIVENESS_FEATURE_PROBE``. Only a node that probes *and* a control
    plane that grades the metric is the liveness generation; a node whose unit
    sets ``GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR`` empty has the probe disabled.
    """

    target: PluginTarget
    node_modules: dict[str, bool]
    control_plane: dict[str, Any]
    device_plugin_env: dict[str, str] = field(default_factory=dict)
    node_evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def node_probe(self) -> bool:
        return bool(self.node_modules) and all(self.node_modules.values())

    @property
    def control_plane_rule(self) -> bool:
        return bool(self.control_plane.get("present")) and bool(
            self.control_plane.get("rule")
        )

    @property
    def disabled(self) -> bool:
        return (
            SOCKET_DIR_ENV in self.device_plugin_env
            and not self.device_plugin_env[SOCKET_DIR_ENV].strip()
        )

    @property
    def liveness(self) -> bool:
        return self.node_probe and self.control_plane_rule and not self.disabled

    def describe(self) -> str:
        venv = self.node_evidence.get("venv") or "the node runtime"
        if self.node_probe:
            node = f"the node's host collector carries the liveness probe ({venv})"
        else:
            missing = sorted(
                name for name, present in self.node_modules.items() if not present
            )
            node = (
                "the node's host collector predates the device-plugin liveness probe "
                f"(no {', '.join(missing) or 'probe modules'} under {venv})"
            )
        plane = (
            f"the control plane grades {self.target.liveness_metric}"
            if self.control_plane_rule
            else f"the control plane has no rule for {self.target.liveness_metric}"
        )
        text = f"{node}; {plane}"
        if self.disabled:
            text += f"; {SOCKET_DIR_ENV} is empty on the node (probe disabled)"
        return text

    def record(self) -> dict[str, Any]:
        return {
            "liveness": self.liveness,
            "node_probe": self.node_probe,
            "control_plane_rule": self.control_plane_rule,
            "disabled": self.disabled,
            "node_modules": self.node_modules,
            "device_plugin_env": self.device_plugin_env,
            "control_plane": self.control_plane,
            "node": self.node_evidence,
            "description": self.describe(),
        }


# --------------------------------------------------------------------------- #
# Reusable fixture
# --------------------------------------------------------------------------- #
@dataclass
class DevicePluginRegistrationFixture:
    """Freeze one device-plugin process on the node and restore it.

    No DaemonSet is excluded and ``Node.status`` is never patched: the only
    mutation is the SIGSTOP the probe applies to the plugin container's own
    process, deadman-backed. ``restore`` resumes it and proves the node clean --
    allocatable at the baseline and exactly one plugin Pod on the node.
    """

    ctx: VariantContext
    target: PluginTarget
    namespace: str = ""
    daemonset: str = ""
    daemonset_uid: str = ""
    baseline_allocatable: int = 0
    original_pod_uid: str = ""
    original_pod_name: str = ""
    pod_command_binaries: tuple[str, ...] = ()
    frozen_uids: set[str] = field(default_factory=set)
    frozen_since_epoch: float = 0.0
    _restore_pushed: bool = False

    @property
    def node(self) -> str:
        return self.ctx.settings.node

    def _gpu_json(self, *arguments: str, **options: Any) -> Any:
        return json.loads(self.ctx.regional.kubectl("gpu", *arguments, **options))

    def discover(self) -> dict[str, Any]:
        daemonsets = self._gpu_json(
            "get", "daemonset", "-o", "json", all_namespaces=True
        )
        # The one DaemonSet the target names, cross-checked against the chart's
        # identity label / image and the product's own restart selector; the
        # refusal names the expected DaemonSet and the ones actually present
        # (the plain-EKS efa_plugin BLOCKED of 2026-10-27 had asked for the
        # HyperPod name: the coordinator's environment, not the target node's
        # provider, had chosen it).
        item = plugin_daemonset(list(daemonsets.get("items", [])), self.target)
        self.namespace = str(item["metadata"]["namespace"])
        self.daemonset = str(item["metadata"]["name"])
        self.daemonset_uid = str(item["metadata"].get("uid") or "")
        pods = self.plugin_pods_on_node()
        if len(pods) != 1:
            raise VariantBlocked(
                "FIXTURE_NOT_READY",
                f"expected exactly one {self.daemonset} Pod on {self.node}, "
                f"found {len(pods)}",
            )
        self.original_pod_uid = str(pods[0]["metadata"].get("uid") or "")
        self.original_pod_name = str(pods[0]["metadata"].get("name") or "")
        # The process basename the probe will signal is provider-specific and
        # not guessable: the HyperPod EFA plugin's is none of the four static
        # names, so a pure allow-list refused it live. Carry the plugin
        # container's own command/image basenames from the discovered Pod spec
        # and hand them to the probe (pod-UID cgroup match stays the hard gate).
        self.pod_command_binaries = plugin_binary_candidates(pods[0])
        self.baseline_allocatable = self.allocatable()
        if self.baseline_allocatable < 1:
            raise VariantBlocked(
                "FIXTURE_NOT_READY",
                f"{self.node} advertises no {self.target.resource}; nothing to lose",
            )
        evidence = {
            "namespace": self.namespace,
            "daemonset": self.daemonset,
            "daemonset_uid": self.daemonset_uid,
            "daemonset_token": self.target.daemonset_token,
            "daemonset_name_source": self.target.name_source,
            "product_selector": self.target.selector,
            "resource": self.target.resource,
            "socket_name": self.target.socket_name,
            "baseline_allocatable": self.baseline_allocatable,
            "original_pod_uid": self.original_pod_uid,
            "original_pod_name": self.original_pod_name,
            "pod_command_binaries": list(self.pod_command_binaries),
        }
        self.ctx.evidence.write("plugin-discover", evidence)
        return evidence

    def generation(self) -> ProductGeneration:
        """Which product this run faces (node package + control-plane rule)."""

        node = self._probe().execute(
            "feature", "--run-id", self.ctx.run_id, timeout=120
        )
        plane = self.ctx.regional.cpu_python(
            LIVENESS_FEATURE_PROBE, self.target.liveness_metric, timeout=120
        )
        modules = node.get("modules") or {}
        value = ProductGeneration(
            target=self.target,
            node_modules={str(key): bool(flag) for key, flag in modules.items()},
            control_plane=dict(plane),
            device_plugin_env={
                str(key): str(item)
                for key, item in (node.get("device_plugin_env") or {}).items()
            },
            node_evidence=dict(node),
        )
        self.ctx.evidence.write("product-generation", value.record())
        if value.node_probe and not value.control_plane_rule:
            raise VariantInconclusive(
                "mixed product generations: " + value.describe() + "; the node's "
                "finding could not be graded by this control plane"
            )
        if value.disabled:
            raise VariantInconclusive(
                "the device-plugin liveness probe is disabled by configuration: "
                + value.describe()
            )
        return value

    def allocatable(self) -> int:
        node = self._gpu_json("get", "node", self.node, "-o", "json")
        raw = (
            (node.get("status") or {})
            .get("allocatable", {})
            .get(self.target.resource, 0)
        )
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return 0

    def plugin_pods_on_node(self) -> list[dict[str, Any]]:
        """Live, non-terminating, non-companion plugin Pods owned by the
        DaemonSet on the node (the product's own definition)."""

        value = self._gpu_json("get", "pod", "-o", "json", namespace=self.namespace)
        owned = []
        for pod in value.get("items", []):
            if (pod.get("spec") or {}).get("nodeName") != self.node:
                continue
            if self.daemonset_uid and not any(
                owner.get("uid") == self.daemonset_uid
                and owner.get("kind") == "DaemonSet"
                for owner in (pod.get("metadata") or {}).get("ownerReferences", [])
            ):
                continue
            if pod_terminating(pod) or companion_pod(pod):
                continue
            owned.append(pod)
        return owned

    def _probe(self) -> ProbeHandle:
        return self.ctx.host_probe("plugin", script=DEVICE_PLUGIN_PROBE, node=self.node)

    def freeze_pod(
        self,
        pod_uid: str,
        *,
        seconds: int,
        resume_after_first_timeout: bool = False,
        window_seconds: int = 0,
    ) -> dict[str, Any]:
        """SIGSTOP the plugin process of ``pod_uid`` (deadman-backed).

        With ``resume_after_first_timeout`` the probe tails the host collector's
        journal and SIGCONTs the process the instant the first timed-out probe
        tick is logged, never later than ``window_seconds`` (``one_sample``).
        """

        if not self._restore_pushed:
            self.ctx.cleanup.push("device-plugin-restore", self.restore)
            self._restore_pushed = True
        arguments = ["freeze", "--run-id", self.ctx.run_id, "--pod-uid", pod_uid]
        for binary in self.target.binaries:
            arguments += ["--binary", binary]
        for binary in self.pod_command_binaries:
            arguments += ["--pod-command-binary", binary]
        arguments += ["--seconds", str(int(seconds))]
        if resume_after_first_timeout:
            arguments += [
                "--resume-after-first-timeout",
                "--window-seconds",
                str(int(window_seconds)),
            ]
        before = time.time()
        result = self._probe().execute(*arguments, timeout=int(window_seconds) + 180)
        if not result.get("frozen"):
            raise VariantInconclusive(
                f"the probe froze no plugin process in pod {pod_uid}"
            )
        self.frozen_uids.add(pod_uid)
        self.frozen_since_epoch = float(result.get("frozen_since_epoch") or before)
        self.ctx.evidence.write(f"freeze-{pod_uid}", result)
        # The probe confirms the SIGSTOP took (the task reached the stopped
        # state) before returning; if it did not, a running plugin keeps
        # answering the product's probe and kubelet alike, so refuse now with
        # the reason rather than waiting out a detection window (the live
        # gpu_plugin run recorded stopped:false and then waited ~4 min before
        # a misleading INCONCLUSIVE).
        not_stopped = [entry for entry in result["frozen"] if not entry.get("stopped")]
        if not_stopped:
            raise VariantInconclusive(
                f"the probe SIGSTOPped the {self.target.kind} plugin process in pod "
                f"{pod_uid} but it did not enter the stopped state ({not_stopped}); "
                "a plugin that keeps running keeps answering the product's probe, "
                "so the hang this variant depends on cannot occur"
            )
        return result

    def resume_now(self) -> dict[str, Any]:
        """Resume the frozen process(es) at once."""

        result = self._probe().execute(
            "resume", "--run-id", self.ctx.run_id, timeout=120
        )
        self.ctx.evidence.write("resume", result)
        return result

    def probe_log(
        self, since_epoch: float, *, name: str = "probe-journal"
    ) -> dict[str, Any]:
        """The host collector's own probe journal since ``since_epoch``."""

        result = self._probe().execute(
            "probe-log",
            "--run-id",
            self.ctx.run_id,
            "--since-epoch",
            f"{since_epoch:.3f}",
            timeout=120,
        )
        self.ctx.evidence.write(name, result)
        return result

    def restore(self) -> dict[str, Any]:
        """Resume any still-frozen process and prove the node clean.

        Idempotent: on the success path the product has already deleted the
        frozen Pod, so the resume is a no-op and the DaemonSet's replacement is
        healthy; on any other path the SIGCONT is what restores the plugin. A
        node that is not at the baseline, or does not carry exactly one plugin
        Pod, fails the variant's cleanup.
        """

        if not self._restore_pushed:
            return {"skipped": "no freeze happened"}
        cleaned = self._probe().execute(
            "cleanup", "--run-id", self.ctx.run_id, "--kind", "all", timeout=120
        )
        if not cleaned.get("clean", False):
            raise RegionalFixtureError(
                f"device-plugin probe cleanup is not clean: {cleaned.get('residue')}"
            )
        allocatable = self.allocatable()
        pods = self.plugin_pods_on_node()
        if allocatable != self.baseline_allocatable:
            raise RegionalFixtureError(
                f"{self.target.resource} allocatable is {allocatable}, not the "
                f"baseline {self.baseline_allocatable} after restore"
            )
        if len(pods) != 1:
            raise RegionalFixtureError(
                f"{self.node} carries {len(pods)} {self.daemonset} Pods after "
                "restore, expected exactly one"
            )
        result = {
            "resumed": cleaned.get("processes"),
            "allocatable": allocatable,
            "plugin_pod_uid": str(pods[0]["metadata"].get("uid") or ""),
        }
        self.ctx.evidence.write("plugin-restore", result)
        return result


# --------------------------------------------------------------------------- #
# Oracles (read the product's own records)
# --------------------------------------------------------------------------- #
def detection_budget_seconds(ctx: VariantContext) -> int:
    """Two ticks (each stretched by the probe's own timeout) plus the timeout
    that times the second out, plus ingest and dispatch."""

    return int(
        2 * (ctx.host_interval + LIVENESS_PROBE_TIMEOUT_SECONDS)
        + LIVENESS_PROBE_TIMEOUT_SECONDS
        + DETECTION_MARGIN_SECONDS
    )


def first_timeout_window_seconds(ctx: VariantContext) -> int:
    """How long ``one_sample`` lets the plugin stay frozen for the first
    timed-out tick to be logged: one tick plus the probe timeout plus margin."""

    return int(
        ctx.host_interval
        + LIVENESS_PROBE_TIMEOUT_SECONDS
        + FIRST_TIMEOUT_MARGIN_SECONDS
    )


def negative_watch_seconds(ctx: VariantContext) -> int:
    """After the resume: the tick that must answer again, the tick after it,
    and the ingest a finding would have needed."""

    return int(
        2 * (ctx.host_interval + LIVENESS_PROBE_TIMEOUT_SECONDS)
        + NEGATIVE_MARGIN_SECONDS
    )


def _await_allocatable_drop(
    fixture: DevicePluginRegistrationFixture,
    generation: ProductGeneration,
    *,
    timeout_seconds: float,
) -> int:
    """The earlier product's only detector: wait for allocatable to fall.

    kubelet does not break the stream of a stopped-but-connected plugin, so on
    that product the freeze is confirmed yet nothing is ever detected; the
    variant is INCONCLUSIVE with the mechanism and the generation cited.
    """

    ctx = fixture.ctx

    def accept() -> int | None:
        value = fixture.allocatable()
        return value if value < fixture.baseline_allocatable else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=10,
        name="allocatable-drop",
    )
    if value is None:
        raise VariantInconclusive(
            f"the {fixture.target.kind} plugin process is confirmed stopped yet "
            f"kubelet still advertises {fixture.baseline_allocatable} "
            f"{fixture.target.resource}: a SIGSTOP'd device plugin keeps its "
            "ListAndWatch socket open, so kubelet does not break the stream or "
            "de-register the resource, and this product's allocatable-mismatch "
            "detector (collectors/cloud/kubernetes) never sees a shortfall to act "
            f"on; {generation.describe()} -- the device-plugin liveness probe "
            "(R13h) is what detects a hung-but-connected plugin"
        )
    return value


def _await_liveness_sample(
    fixture: DevicePluginRegistrationFixture,
    started: datetime,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Wait for the product's ``*_device_plugin_unresponsive`` = 1 sample.

    Every poll also reads the raw kubelet allocatable: those readings are the
    proof that the advertisement stayed full while the plugin was frozen and
    that the detection came from the probe, not from kubelet.
    """

    ctx = fixture.ctx
    target = fixture.target
    readings: list[dict[str, Any]] = []

    def accept() -> dict[str, Any] | None:
        readings.append(
            {"observed_at": utc_now().isoformat(), "allocatable": fixture.allocatable()}
        )
        current = activity(ctx, started, metrics=target.metrics)
        hits = [
            item
            for item in samples(current, target.liveness_metric)
            if item.get("value") == 1
        ]
        return {"activity": current, "sample": hits[0]} if hits else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=5,
        name="liveness-detection",
    )
    result = {
        "sample": value["sample"] if value else None,
        "allocatable_readings": readings,
        "budget_seconds": timeout_seconds,
    }
    ctx.evidence.write("liveness-detection", result)
    return result


def _liveness_sample_errors(
    sample: dict[str, Any] | None,
    target: PluginTarget,
    *,
    late: list[dict[str, Any]] = (),
    budget_seconds: float | None = None,
) -> list[str]:
    """What the probe's finding sample must say about the frozen plugin.

    ``late`` are the value-1 samples a later read found after the detection
    budget ran out: named in the error, never graded as the detection.
    """

    if sample is None:
        error = (
            f"no {target.liveness_metric}=1 sample within {budget_seconds:g} s "
            if budget_seconds is not None
            else f"no {target.liveness_metric}=1 sample "
        ) + (
            "although the plugin process was confirmed frozen and this node's host "
            "collector carries the probe"
        )
        if late:
            error += f"; a sample arrived only at {late[0].get('observed_at')}"
        return [error]
    labels = sample.get("labels") or {}
    errors = []
    if labels.get("probe_state") != "TIMEOUT":
        errors.append(f"probe_state is {labels.get('probe_state')!r}, expected TIMEOUT")
    if str(labels.get("transport_answered")).lower() != "false":
        errors.append(
            f"transport_answered is {labels.get('transport_answered')!r}, expected "
            "false (a stopped process sends no HTTP/2 frame at all)"
        )
    sockets = [item for item in str(labels.get("sockets") or "").split(",") if item]
    if target.socket_name not in sockets:
        errors.append(
            f"sockets {sockets} do not name the frozen plugin's socket "
            f"{target.socket_name}"
        )
    if labels.get("failure_mode") != UNRESPONSIVE_FAILURE_MODE:
        errors.append(
            f"failure_mode is {labels.get('failure_mode')!r}, expected "
            f"{UNRESPONSIVE_FAILURE_MODE}"
        )
    consecutive = labels.get("consecutive_timeout_samples")
    required = labels.get("required_consecutive_samples")
    if consecutive is not None and required is not None:
        try:
            if int(str(consecutive)) < int(str(required)):
                errors.append(
                    f"the sample is 1 at consecutive_timeout_samples={consecutive} "
                    f"below required_consecutive_samples={required}"
                )
        except ValueError:
            errors.append(
                f"unreadable consecutive/required labels {consecutive}/{required}"
            )
    return errors


def _frozen_allocatable_errors(
    readings: list[dict[str, Any]], fixture: DevicePluginRegistrationFixture
) -> list[str]:
    """kubelet kept advertising every device while the plugin was frozen."""

    low = [
        item
        for item in readings
        if isinstance(item.get("allocatable"), int)
        and item["allocatable"] < fixture.baseline_allocatable
    ]
    if not low:
        return []
    return [
        f"kubelet advertised {low[0]['allocatable']} {fixture.target.resource} "
        f"(baseline {fixture.baseline_allocatable}) at {low[0]['observed_at']} while "
        "the plugin was frozen and before the product's restart: the "
        "hung-but-connected premise did not hold on this node"
    ]


def _restart_executions(
    activity_doc: dict[str, Any], operation: str
) -> list[dict[str, Any]]:
    result = []
    for workflow in activity_doc.get("workflows") or []:
        for execution in workflow.get("step_executions") or []:
            if execution.get("operation") == operation:
                result.append(execution)
    return result


def _merge_receipt(
    receipt: dict[str, Any],
    activity_doc: dict[str, Any],
    target: PluginTarget,
    node: str,
) -> None:
    """Accumulate the ``RESTART_*`` step's per-node receipt across polls.

    The step execution's ``details`` is replaced on every attempt
    (``execution/step_bounds.record_attempt``): the delete pass carries
    ``deleted_pod_uid`` and -- for a hung plugin -- ``restart_reason``, the
    waiting passes ``waiting_for_replacement_pod``, the final pass
    ``replacement_pod_uids`` and ``already_healthy``. A single post-hoc read
    sees only the last, so the runner keeps the union of every poll --
    the product's own records, read live. ``delete_pass_seen`` says whether
    the delete pass itself was caught (its ``restart_reason`` can be judged
    only then).
    """

    for execution in _restart_executions(activity_doc, target.restart_operation):
        node_result = (
            (execution.get("details") or {}).get(NODE_RESULTS_KEY) or {}
        ).get(node) or {}
        for key in ("deleted_pod_uid", "deleted_pod", "restart_reason"):
            if node_result.get(key):
                receipt[key] = str(node_result[key])
        if node_result.get("deleted_pod_uid") and not any(
            key in node_result
            for key in ("waiting_for_replacement_pod", "replacement_pod_uids")
        ):
            receipt["delete_pass_seen"] = True
        if "already_healthy" in node_result:
            receipt["already_healthy"] = bool(node_result["already_healthy"])
        replacements = node_result.get("replacement_pod_uids")
        if replacements:
            receipt["replacement_pod_uids"] = [str(item) for item in replacements]
        if "allocatable" in node_result:
            receipt["final_allocatable"] = node_result.get("allocatable")
        if execution.get("status") == "FAILED":
            receipt["restart_failed"] = True
            receipt["restart_error"] = str(execution.get("error") or "")[:300]


# --------------------------------------------------------------------------- #
# gpu_plugin / efa_plugin / replacement_uid
# --------------------------------------------------------------------------- #
def _restart_variant(
    ctx: VariantContext, target: PluginTarget, *, grade_replacement: bool
) -> dict[str, Any]:
    run_survey(ctx)
    fixture = DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    generation = fixture.generation()
    started = utc_now()
    deadman = min(ctx.spec.max_minutes * 60, 1500)
    fixture.freeze_pod(fixture.original_pod_uid, seconds=deadman)
    frozen_at = utc_now()
    detection: dict[str, Any] | None = None
    if generation.liveness:
        detection = _await_liveness_sample(
            fixture, started, timeout_seconds=detection_budget_seconds(ctx)
        )
        finding_metric = target.liveness_metric
    else:
        _await_allocatable_drop(
            fixture, generation, timeout_seconds=ctx.host_interval * 8 + 120
        )
        finding_metric = target.mismatch_metric
    receipt: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics=target.metrics)
        _merge_receipt(receipt, current, target, ctx.settings.node)
        incidents = finding_incidents(current, finding_metric)
        if not incidents:
            return None
        ids = [str(item["incident_id"]) for item in incidents]
        return current if settled(current, ids) else None

    # A node that probes and never reported the hang has already failed the
    # variant; give the chain one more detection budget to show up late rather
    # than the whole window.
    undetected = detection is not None and detection["sample"] is None
    observed = ctx.wait_until(
        accept,
        timeout_seconds=(
            detection_budget_seconds(ctx) if undetected else ctx.spec.max_minutes * 60
        ),
        poll_seconds=5,
        name="restart-settle",
    )
    if observed is None:
        observed = activity(ctx, started, metrics=target.metrics)
        _merge_receipt(receipt, observed, target, ctx.settings.node)
    ctx.evidence.write("restart-receipt", receipt)
    journal = (
        fixture.probe_log(fixture.frozen_since_epoch) if generation.liveness else {}
    )

    # The replacement Pod on the node after the restart (census), and whether
    # the product deleted the original.
    final_pods = fixture.plugin_pods_on_node()
    final_uids = [str(pod["metadata"].get("uid") or "") for pod in final_pods]
    original_gone = fixture.original_pod_uid not in final_uids

    stages: dict[str, list[str]] = {}
    if detection is not None:
        stages["detected"] = _liveness_sample_errors(
            detection["sample"],
            target,
            late=[
                item
                for item in samples(observed, target.liveness_metric)
                if item.get("value") == 1
            ],
            budget_seconds=detection["budget_seconds"],
        )
    stages["restart_chain"] = chain_errors(
        observed,
        ChainExpectation(
            metric=finding_metric,
            present=True,
            operations=(
                target.restart_operation,
                "TRIGGER_HEALTH_SNAPSHOT",
                target.validate_operation,
            ),
            validation=target.validate_operation,
            incident_states=frozenset({"RECOVERED"}),
        ),
    )
    stages["deleted_original"] = _deleted_errors(
        receipt, fixture, original_gone, liveness=generation.liveness
    )
    stages["replacement_ready"] = _replacement_errors(
        receipt, fixture, final_pods, final_uids, liveness=generation.liveness
    )
    if detection is not None:
        stages["allocatable_full_while_frozen"] = _frozen_allocatable_errors(
            detection["allocatable_readings"], fixture
        )
    stages["allocatable_restored"] = _allocatable_errors(observed, fixture)
    stages["no_reboot"] = hardware_errors(observed)
    stages["not_isolated"] = _isolation_errors(ctx, observed)
    if grade_replacement:
        stages["no_redelete"] = _no_redelete_errors(receipt, fixture, final_uids)

    sample = detection["sample"] if detection else None
    seen = parse_time(sample.get("observed_at")) if sample else None
    notes = [
        f"froze the {target.kind} plugin process (pod {fixture.original_pod_uid}); "
        + (
            "the product's socket probe found it hung and the product deleted and "
            "replaced the Pod"
            if generation.liveness
            else "allocatable dropped and the product deleted and replaced the Pod"
        ),
        generation.describe(),
    ]
    if generation.liveness and not receipt.get("delete_pass_seen"):
        notes.append(
            "the delete pass's own receipt (restart_reason) was replaced before a "
            "poll caught it; the delete is proven by deleted_pod_uid in the later "
            "receipts and by the Pod census"
        )
    return _result(
        stages,
        notes=notes,
        extra={
            "generation": generation.record(),
            "restart_receipt": receipt,
            "liveness_sample": sample,
            "detection_seconds": (
                (seen - frozen_at).total_seconds() if seen is not None else None
            ),
            "allocatable_while_frozen": (
                detection["allocatable_readings"] if detection else []
            ),
            "probe_journal_timeouts": journal.get("timeouts") if journal else None,
            "original_pod_uid": fixture.original_pod_uid,
            "final_plugin_pod_uids": final_uids,
            "baseline_allocatable": fixture.baseline_allocatable,
        },
    )


def gpu_plugin(ctx: VariantContext) -> dict[str, Any]:
    """Freeze the GPU device plugin; the product restarts it, no reboot."""

    return _restart_variant(ctx, GPU_TARGET, grade_replacement=False)


def efa_plugin(ctx: VariantContext) -> dict[str, Any]:
    """Freeze the EFA device plugin; the product restarts it, no reboot.

    The plugin's DaemonSet name follows the target node's provider
    (``target_cluster_provider``), never this process's environment.
    """

    return _restart_variant(
        ctx, efa_target(target_cluster_provider(ctx)), grade_replacement=False
    )


def replacement_uid(ctx: VariantContext) -> dict[str, Any]:
    """The restart's replacement Pod UID is recorded, new, and not re-deleted."""

    return _restart_variant(ctx, GPU_TARGET, grade_replacement=True)


def _deleted_errors(
    receipt: dict[str, Any],
    fixture: DevicePluginRegistrationFixture,
    original_gone: bool,
    *,
    liveness: bool,
) -> list[str]:
    errors = []
    deleted = receipt.get("deleted_pod_uid")
    if deleted:
        if deleted != fixture.original_pod_uid:
            errors.append(
                f"the restart deleted pod {deleted}, not the frozen plugin pod "
                f"{fixture.original_pod_uid}"
            )
    elif not original_gone:
        # No receipt named the deleted Pod; the delete still has to be proven
        # on the product's effect: the frozen Pod is gone and the restart step
        # ran (the chain stage asserts the operation).
        errors.append(
            f"the frozen plugin pod {fixture.original_pod_uid} is still on the node; "
            "no delete receipt and no census proof of the delete"
        )
    if liveness and receipt.get("delete_pass_seen"):
        reason = receipt.get("restart_reason")
        if reason != UNRESPONSIVE_FAILURE_MODE:
            errors.append(
                f"the delete receipt carries restart_reason={reason!r}, expected "
                f"{UNRESPONSIVE_FAILURE_MODE}: the restart did not say why a Pod on a "
                "node advertising every device was deleted"
            )
    return errors


def _replacement_errors(
    receipt: dict[str, Any],
    fixture: DevicePluginRegistrationFixture,
    final_pods: list[dict[str, Any]],
    final_uids: list[str],
    *,
    liveness: bool,
) -> list[str]:
    replacements = receipt.get("replacement_pod_uids") or []
    errors = []
    if not replacements:
        errors.append("the restart step recorded no replacement_pod_uids")
    if fixture.original_pod_uid in replacements:
        errors.append(
            f"replacement_pod_uids names the deleted pod {fixture.original_pod_uid}"
        )
    if liveness:
        healthy = receipt.get("already_healthy")
        if healthy is True:
            errors.append(
                "the restart read the full allocatable as already_healthy and "
                "deleted nothing: a hung plugin advertises every device"
            )
        elif healthy is None:
            errors.append("no SUCCEEDED receipt carried already_healthy=false")
    if len(final_pods) != 1:
        errors.append(
            f"{len(final_pods)} plugin Pods on the node after the restart, expected one"
        )
        return errors
    replacement = final_pods[0]
    uid = str(replacement["metadata"].get("uid") or "")
    if uid == fixture.original_pod_uid:
        errors.append("the plugin Pod on the node is still the deleted one")
    if replacements and uid not in replacements:
        errors.append(
            f"the Ready plugin Pod {uid} is not in replacement_pod_uids {replacements}"
        )
    if not _pod_ready(replacement):
        errors.append(f"the replacement plugin Pod {uid} is not Ready")
    return errors


def _no_redelete_errors(
    receipt: dict[str, Any],
    fixture: DevicePluginRegistrationFixture,
    final_uids: list[str],
) -> list[str]:
    """The product reached the idempotent healthy branch, not a second delete."""

    errors = []
    if receipt.get("restart_failed"):
        errors.append("the restart step failed rather than confirming the replacement")
    replacements = receipt.get("replacement_pod_uids") or []
    # The final healthy branch returns replacement_pod_uids and clears its
    # annotation; a second delete would leave the node with a different,
    # newer Pod than the one the receipt named.
    if replacements and final_uids and set(final_uids) - set(replacements):
        errors.append(
            f"the plugin Pod on the node {final_uids} is not the recorded "
            f"replacement {replacements}: the product re-deleted the new Pod"
        )
    return errors


def _allocatable_errors(
    observed: dict[str, Any], fixture: DevicePluginRegistrationFixture
) -> list[str]:
    target = fixture.target
    counts = [
        item
        for item in samples(observed, target.allocatable_metric)
        if isinstance(item.get("value"), (int, float))
    ]
    if not counts:
        # Fall back to the live kubelet oracle: the collector may not have
        # delivered a healthy summary yet, but the resource is back.
        if fixture.allocatable() >= fixture.baseline_allocatable:
            return []
        return [
            f"no fresh {target.allocatable_metric} sample and allocatable still short"
        ]
    if counts[-1]["value"] < fixture.baseline_allocatable:
        return [
            f"the last {target.allocatable_metric} sample is {counts[-1]['value']:g}, "
            f"below the baseline {fixture.baseline_allocatable}"
        ]
    mismatch = [
        item
        for item in samples(observed, target.mismatch_metric)
        if isinstance(item.get("value"), (int, float))
    ]
    if mismatch and mismatch[-1]["value"] != 0:
        return [f"{target.mismatch_metric} is still {mismatch[-1]['value']:g}"]
    return []


# --------------------------------------------------------------------------- #
# one_sample
# --------------------------------------------------------------------------- #
def one_sample(ctx: VariantContext) -> dict[str, Any]:
    """One timed-out probe tick is not confirmed into an action (2-sample rule).

    The probe SIGCONTs the plugin the instant the host collector logs its
    first timed-out ``GetDevicePluginOptions`` tick (``consecutive_timeout_
    samples=1``) and before the second, so the confirmation never closes: no
    finding, no restart, no delete. Against the earlier product the freeze is
    confirmed and allocatable watched (``_await_allocatable_drop``).
    """

    target = GPU_TARGET
    run_survey(ctx)
    fixture = DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    generation = fixture.generation()
    started = utc_now()
    deadman = min(ctx.spec.max_minutes * 60, 1500)
    if not generation.liveness:
        fixture.freeze_pod(fixture.original_pod_uid, seconds=deadman)
        _await_allocatable_drop(
            fixture, generation, timeout_seconds=ctx.host_interval * 8 + 120
        )
        fixture.resume_now()
        raise VariantInconclusive(
            "allocatable dropped under a frozen plugin on a product without the "
            "liveness probe; this harness grades the single-sample window on the "
            "probe's timed-out tick, not on the allocatable rule -- rerun against "
            f"the probe generation; {generation.describe()}"
        )
    window = first_timeout_window_seconds(ctx)
    receipt = fixture.freeze_pod(
        fixture.original_pod_uid,
        seconds=deadman,
        resume_after_first_timeout=True,
        window_seconds=window,
    )
    first = receipt.get("first_timeout")
    if not isinstance(first, dict):
        raise VariantInconclusive(
            f"the host collector logged no timed-out GetDevicePluginOptions tick "
            f"within {window} s of the confirmed freeze (the plugin was resumed at "
            f"the window); {generation.describe()}"
        )
    if receipt.get("second_timeout_seen"):
        raise VariantInconclusive(
            "the second consecutive timed-out tick was logged before the probe "
            f"could resume the plugin ({receipt.get('timeouts')}): the single-"
            "sample window was not achieved"
        )

    def confirmed() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics=target.metrics)
        minted = finding_incidents(current, target.liveness_metric)
        edge = [
            item
            for item in samples(current, target.liveness_metric)
            if item.get("value") == 1
        ]
        return current if minted or edge else None

    early = ctx.wait_until(
        confirmed,
        timeout_seconds=negative_watch_seconds(ctx),
        poll_seconds=10,
        name="one-sample-negative",
    )
    observed = early or activity(ctx, started, metrics=target.metrics)
    journal = fixture.probe_log(fixture.frozen_since_epoch)
    timeouts = [
        item for item in (journal.get("timeouts") or []) if isinstance(item, dict)
    ]
    if len(timeouts) != 1 or int(timeouts[0].get("consecutive") or 0) != 1:
        raise VariantInconclusive(
            f"the host collector logged {len(timeouts)} timed-out ticks after the "
            f"freeze ({[item.get('consecutive') for item in timeouts]}), not the "
            "single first tick the window needs; a second tick timed out before the "
            "resume took effect"
        )
    stages = {
        "no_confirmed_sample": [
            f"a confirmed {target.liveness_metric} sample was recorded: "
            f"{item.get('observed_at')} {item.get('labels')}"
            for item in samples(observed, target.liveness_metric)
            if item.get("value") == 1
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=target.liveness_metric, present=False)
        ),
        "no_action": _no_plugin_restart_errors(observed, target),
        "not_isolated": _isolation_errors(ctx, observed),
    }
    return _result(
        stages,
        notes=[
            "the plugin was frozen only until the host collector logged its first "
            f"timed-out probe tick (consecutive {first.get('consecutive')} of "
            f"{first.get('required')}) and resumed on the node "
            f"{receipt.get('resume_latency_seconds')} s after that line; the "
            "2-sample rule left no finding",
            generation.describe(),
        ],
        extra={
            "generation": generation.record(),
            "first_timeout": first,
            "resume_latency_seconds": receipt.get("resume_latency_seconds"),
            "window_seconds": window,
            "probe_journal_timeouts": timeouts,
        },
    )


def _no_plugin_restart_errors(
    observed: dict[str, Any], target: PluginTarget
) -> list[str]:
    found = []
    for workflow in observed.get("workflows") or []:
        if target.restart_operation in operations(workflow):
            found.append(str(workflow.get("request_id")))
    return (
        [f"a {target.restart_operation} workflow was planned: {found}"] if found else []
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _pod_ready(pod: dict[str, Any]) -> bool:
    conditions = (pod.get("status") or {}).get("conditions") or []
    return any(
        str(item.get("type")) == "Ready" and str(item.get("status")).lower() == "true"
        for item in conditions
    )


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    isolation = isolation_operations(observed)
    errors = [f"isolation operations planned: {isolation}"] if isolation else []
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("the node is isolated after a device-plugin fault")
    return errors


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        "proof_scope": "native",
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _spec(
    variant: str,
    *,
    injection: str,
    expected: str,
    execute: Any,
    risk: str,
    max_minutes: int,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_014,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=_CLEANUP,
        max_minutes=max_minutes,
        approvals=(DEVICE_PLUGIN_RESTART,),
        execute=execute,
    )


_CLEANUP = (
    "resume the frozen plugin process (deadman-backed SIGCONT); prove allocatable "
    "at the baseline and exactly one plugin Pod on the node"
)
_FREEZE_EFFECT = (
    "kubelet keeps the Pod Running and the allocatable full (no ListAndWatch "
    "deadline); the product's host collector finds the socket unanswering"
)
_RESTART_EXPECTED = (
    "within ~2 host ticks + 5 s a {metric}=1 sample (probe_state=TIMEOUT, "
    "transport_answered=false, sockets naming {socket}); its CRITICAL finding plans "
    "{restart} with plugin_unresponsive: the frozen Pod is deleted (receipt "
    "restart_reason=DEVICE_PLUGIN_UNRESPONSIVE, deleted_pod_uid), the SUCCEEDED "
    "receipt names a Ready replacement of another UID with already_healthy=false, "
    "TRIGGER_HEALTH_SNAPSHOT + {validate} SUCCEEDED, allocatable never below the "
    "baseline, incident RECOVERED, no reboot, no isolation; a product without the "
    "probe is INCONCLUSIVE (allocatable never drops), not FAIL"
)


def _restart_expected(target: PluginTarget) -> str:
    return _RESTART_EXPECTED.format(
        metric=target.liveness_metric,
        socket=target.socket_name,
        restart=target.restart_operation,
        validate=target.validate_operation,
    )


VARIANTS = [
    _spec(
        "gpu_plugin",
        injection=(
            "SIGSTOP the nvidia-device-plugin container process on the node (its "
            "pod-UID + binary are the only ones the probe will signal); "
            + _FREEZE_EFFECT
        ),
        expected=_restart_expected(GPU_TARGET),
        execute=gpu_plugin,
        risk="H",
        max_minutes=30,
    ),
    _spec(
        "efa_plugin",
        injection=(
            "SIGSTOP the EFA device-plugin container process on the node (its "
            "DaemonSet named by the target node's provider); " + _FREEZE_EFFECT
        ),
        expected=_RESTART_EXPECTED.format(
            metric="efa_device_plugin_unresponsive",
            socket="aws-efa-device-plugin.sock",
            restart="RESTART_EFA_DEVICE_PLUGIN",
            validate="VALIDATE_FABRIC",
        ),
        execute=efa_plugin,
        risk="H",
        max_minutes=30,
    ),
    _spec(
        "one_sample",
        injection=(
            "SIGSTOP the GPU device-plugin process; the probe tails the host "
            "collector's journal and SIGCONTs the instant the first timed-out "
            "GetDevicePluginOptions tick is logged (consecutive_timeout_samples=1), "
            "before the second"
        ),
        expected=(
            "one timed-out tick is below required_consecutive_samples=2: no "
            "gpu_device_plugin_unresponsive=1 sample, no finding, no "
            "RESTART_GPU_DEVICE_PLUGIN, no delete, no isolation; the next tick "
            "answers again"
        ),
        execute=one_sample,
        risk="M",
        max_minutes=20,
    ),
    _spec(
        "replacement_uid",
        injection=(
            "SIGSTOP the GPU device-plugin process and let the product restart it, "
            "grading the replacement identity"
        ),
        expected=(
            "the restart records a new replacement Pod UID different from the deleted "
            "one, the Ready Pod on the node is that replacement, and the product does "
            "not re-delete it (idempotent healthy branch)"
        ),
        execute=replacement_uid,
        risk="H",
        max_minutes=30,
    ),
    deferred_variant(
        CASE_014,
        "validation_failure",
        proof_mode=NATIVE,
        reason=VALIDATION_FAILURE_DEFERRED,
        lanes=("eks", "hp"),
        expected=(
            "a RESTART_GPU_DEVICE_PLUGIN whose replacement never restores the "
            "resource FAILS at restart_timeout_seconds and escalates to RESTART_NODE "
            "(event id 'reboot-after-<restart>') under the separate NODE_REBOOT "
            "authorization"
        ),
    ),
]
