"""GF-GENERAL-LIVE-014: Kubernetes allocatable under-report (native).

A Kubernetes allocatable shortfall must come from a *real* device-plugin
registration fault: the spec (014) is explicit that patching ``Node.status`` or
faking allocatable is only a source replay, so there is no live-replay path --
these are ``native``. The one native injection that does not remove the
DaemonSet from the node (the COLLECT-017 B/C fixture does, and restores it
*before* the product acts, so the product takes the ``already_healthy`` branch
of ``adapters/kubernetes/node_operations`` and never deletes a Pod) is to
freeze the plugin container's own process: a SIGSTOP (``probes/
general_live_device_plugin_probe.py``) stops it answering kubelet's device-plugin
``ListAndWatch`` stream, kubelet de-registers the resource, and the node's
allocatable decays while the plugin Pod stays ``Running``. The product then has
to delete and replace the Pod itself (``RESTART_{GPU,EFA}_DEVICE_PLUGIN``) -- the
branch (``node_operations._restart_device_plugin_node`` ~297-330), the
``required_consecutive_samples=2`` confirmation (``collectors/cloud/kubernetes``)
and the new-Pod-UID verification COLLECT-017 never reaches.

Variants:

* ``gpu_plugin`` / ``efa_plugin`` -- freeze the GPU / EFA plugin, let the product
  restart it: ``RESTART_*_DEVICE_PLUGIN`` + ``TRIGGER_HEALTH_SNAPSHOT`` +
  ``VALIDATE_{GPU,FABRIC}`` all SUCCEEDED, the step receipt names a new Ready
  replacement Pod UID (``replacement_pod_uids``) different from the deleted one,
  allocatable back to baseline, incident RECOVERED, no reboot;
* ``one_sample`` -- freeze only until the product records one allocatable
  mismatch, resume before the second: the 2-sample rule leaves no finding and no
  action;
* ``replacement_uid`` -- the GPU restart, graded on the replacement identity: the
  new Pod UID is recorded, differs from the deleted one, and the product does
  not re-delete it once it is Ready;
* ``validation_failure`` -- keep the registration broken across the replacement
  (re-freeze each new Pod) so the restart's own allocatable snapshot never
  confirms: the ``RESTART_*_DEVICE_PLUGIN`` step fails at ``restart_timeout_seconds``
  and -- tagged ``failure_escalation_action=REBOOT_NODE`` by
  ``orchestration/families/health._resource_escalation`` -- escalates to
  ``RESTART_NODE`` under the separate ``NODE_REBOOT`` authorization.

``expected_unknown`` / ``mig_timeslicing`` (CAPABILITY_ABSENT) and
``running_training`` (FIXTURE_NOT_READY, lane N) stay deferred in
``general_live_inventory``.

All five native variants are registered on both clusters. Nothing in the
freeze is provider-specific: the plugin Pods are found by their kubelet
device-plugin socket mount, the EFA plugin's DaemonSet name comes from the
node's provider (``node_provider.efa_device_plugin_name``), the SIGSTOP targets
the plugin container's own process, and the product's restart/escalation runs
on both providers; HyperPod's health-monitoring agent watches GPU/EFA hardware,
not the plugin registration, and the lane preflight already requires
``NodeRecovery=None`` there. The EKS-first ordering of the first live runs was
sequencing, not a constraint.

The reusable entry points another category (e.g. a lane-N ``running_training``
on a managed attempt) calls are the probe's ``freeze`` / ``resume`` / ``cleanup``
subcommands and :class:`DevicePluginRegistrationFixture` (discover, ``freeze_pod``,
``allocatable``, ``plugin_pods_on_node``, ``restore``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gpu_fault.adapters.kubernetes.node_operations import (
    companion_pod,
    pod_terminating,
)
from gpu_fault.node_provider import efa_device_plugin_name
from gpu_fault.orchestration.escalation import escalation_origin
from scripts.e2e.regional.collector_device_plugin_fixture import companion_daemonset
from scripts.e2e.regional.general_live_common import activity, run_survey
from scripts.e2e.regional.general_live_fixture import (
    PROBES,
    ProbeHandle,
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    operations,
    samples,
    verdict,
    workflows_of,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CASE_014 = "GF-GENERAL-LIVE-014"

#: The approval the product's device-plugin pod-delete/re-register branch needs,
#: and the separate one its post-restart escalation (``RESTART_NODE``) needs.
DEVICE_PLUGIN_RESTART = "DEVICE_PLUGIN_RESTART"
NODE_REBOOT = "NODE_REBOOT"

DEVICE_PLUGIN_PROBE = PROBES / "general_live_device_plugin_probe.py"

#: Where the product's own records show the deleted / replacement Pod UIDs:
#: ``node_operations`` writes them into the ``RESTART_*`` step's
#: ``details.node_results.<node>``.
NODE_RESULTS_KEY = "node_results"

#: The default plugin-restart restore timeout the product waits for allocatable
#: to return before failing the step (``node_operations._restart_device_plugin``
#: ``restart_timeout_seconds``); the ``validation_failure`` hold must outlast it.
RESTART_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class PluginTarget:
    """One device plugin: how it is discovered and how the product grades it."""

    kind: str  # "gpu" | "efa"
    resource: str  # the kubelet resource name
    daemonset_token: str  # the substring that identifies its DaemonSet
    binaries: tuple[str, ...]  # candidate process basenames (probe allow-list)
    restart_operation: str
    validate_operation: str

    @property
    def mismatch_metric(self) -> str:
        return f"{self.kind}_kubernetes_allocatable_mismatch"

    @property
    def allocatable_metric(self) -> str:
        return f"{self.kind}_kubernetes_allocatable_count"

    @property
    def expected_metric(self) -> str:
        return f"{self.kind}_kubernetes_expected_count"

    @property
    def metrics(self) -> set[str]:
        return {
            self.mismatch_metric,
            self.allocatable_metric,
            self.expected_metric,
            f"{self.kind}_kubernetes_allocatable_missing_count",
        }


GPU_TARGET = PluginTarget(
    kind="gpu",
    resource="nvidia.com/gpu",
    daemonset_token="nvidia-device-plugin",
    binaries=("nvidia-device-plugin",),
    restart_operation="RESTART_GPU_DEVICE_PLUGIN",
    validate_operation="VALIDATE_GPU",
)


def efa_target() -> PluginTarget:
    """The EFA plugin target; its DaemonSet name label is provider-specific."""

    return PluginTarget(
        kind="efa",
        resource="vpc.amazonaws.com/efa",
        daemonset_token=efa_device_plugin_name(),
        binaries=("aws-efa-k8s-device-plugin", "k8s-device-plugin", "device-plugin"),
        restart_operation="RESTART_EFA_DEVICE_PLUGIN",
        validate_operation="VALIDATE_FABRIC",
    )


# --------------------------------------------------------------------------- #
# Reusable fixture
# --------------------------------------------------------------------------- #
@dataclass
class DevicePluginRegistrationFixture:
    """Freeze one device-plugin process on the node and restore it.

    No DaemonSet is excluded and ``Node.status`` is never patched: the only
    mutation is the SIGSTOP the probe applies to the plugin container's own
    process, deadman-backed. ``restore`` resumes it and proves the node clean --
    allocatable back to baseline and exactly one plugin Pod on the node.
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
    _restore_pushed: bool = False

    @property
    def node(self) -> str:
        return self.ctx.settings.node

    def _gpu_json(self, *arguments: str, **options: Any) -> Any:
        import json

        return json.loads(self.ctx.regional.kubectl("gpu", *arguments, **options))

    def discover(self) -> dict[str, Any]:
        daemonsets = self._gpu_json(
            "get", "daemonset", "-o", "json", all_namespaces=True
        )
        import json as _json

        matches = [
            item
            for item in daemonsets.get("items", [])
            if self.target.daemonset_token.lower()
            in _json.dumps(item, sort_keys=True).lower()
            and int((item.get("status") or {}).get("desiredNumberScheduled") or 0) > 0
            and not companion_daemonset(item)
        ]
        if len(matches) != 1:
            raise VariantBlocked(
                "FIXTURE_NOT_READY",
                f"expected one {self.target.daemonset_token} DaemonSet on the "
                f"cluster, found {len(matches)}",
            )
        item = matches[0]
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
            "resource": self.target.resource,
            "baseline_allocatable": self.baseline_allocatable,
            "original_pod_uid": self.original_pod_uid,
            "original_pod_name": self.original_pod_name,
            "pod_command_binaries": list(self.pod_command_binaries),
        }
        self.ctx.evidence.write("plugin-discover", evidence)
        return evidence

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

    def freeze_pod(self, pod_uid: str, *, seconds: int) -> dict[str, Any]:
        """SIGSTOP the plugin process of ``pod_uid`` (deadman-backed)."""

        if not self._restore_pushed:
            self.ctx.cleanup.push("device-plugin-restore", self.restore)
            self._restore_pushed = True
        arguments = ["freeze", "--run-id", self.ctx.run_id, "--pod-uid", pod_uid]
        for binary in self.target.binaries:
            arguments += ["--binary", binary]
        for binary in self.pod_command_binaries:
            arguments += ["--pod-command-binary", binary]
        arguments += ["--seconds", str(int(seconds))]
        result = self._probe().execute(*arguments, timeout=180)
        if not result.get("frozen"):
            raise VariantInconclusive(
                f"the probe froze no plugin process in pod {pod_uid}"
            )
        self.frozen_uids.add(pod_uid)
        self.ctx.evidence.write(f"freeze-{pod_uid}", result)
        # The probe confirms the SIGSTOP took (the task reached the stopped
        # state) before returning; if it did not, a running plugin keeps
        # answering kubelet and allocatable will never drop, so refuse now with
        # the reason rather than waiting out the whole allocatable-drop window
        # (the live gpu_plugin run recorded stopped:false and then waited ~4 min
        # before the misleading "allocatable did not drop" INCONCLUSIVE).
        not_stopped = [entry for entry in result["frozen"] if not entry.get("stopped")]
        if not_stopped:
            raise VariantInconclusive(
                f"the probe SIGSTOPped the {self.target.kind} plugin process in pod "
                f"{pod_uid} but it did not enter the stopped state ({not_stopped}); "
                "a plugin that keeps running keeps answering kubelet, so the "
                "allocatable drop this variant depends on cannot occur"
            )
        return result

    def resume_now(self) -> dict[str, Any]:
        """Resume the frozen process(es) at once (the single-sample window)."""

        result = self._probe().execute(
            "resume", "--run-id", self.ctx.run_id, timeout=120
        )
        self.ctx.evidence.write("resume", result)
        return result

    def restore(self) -> dict[str, Any]:
        """Resume any still-frozen process and prove the node clean.

        Idempotent: on the success path the product has already deleted the
        frozen Pod, so the resume is a no-op and the DaemonSet's replacement is
        healthy; on any other path the SIGCONT is what restores allocatable.
        A node that is not back to baseline, or does not carry exactly one
        plugin Pod, fails the variant's cleanup.
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
# Oracle and receipt helpers (read the product's own records)
# --------------------------------------------------------------------------- #
def _await_allocatable_drop(
    fixture: DevicePluginRegistrationFixture, *, timeout_seconds: float
) -> int:
    """Wait (raw kubelet oracle) until allocatable falls below baseline."""

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
            "de-register the resource, and the product's allocatable-mismatch "
            "detector (collectors/cloud/kubernetes) never sees a shortfall to act on"
        )
    return value


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
    """Accumulate the ``RESTART_*`` step's Pod-UID receipt across polls.

    The step execution's ``details`` is replaced on every attempt
    (``execution/step_bounds.record_attempt``): the delete attempt carries
    ``deleted_pod_uid`` and the final healthy attempt carries
    ``replacement_pod_uids``, so a single post-hoc read sees only the last. The
    runner captures each poll and keeps the union -- the product's own records,
    read live.
    """

    for execution in _restart_executions(activity_doc, target.restart_operation):
        node_result = (
            (execution.get("details") or {}).get(NODE_RESULTS_KEY) or {}
        ).get(node) or {}
        deleted = node_result.get("deleted_pod_uid")
        if deleted:
            receipt["deleted_pod_uid"] = str(deleted)
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
    started = utc_now()
    deadman = min(ctx.spec.max_minutes * 60, 1500)
    fixture.freeze_pod(fixture.original_pod_uid, seconds=deadman)
    _await_allocatable_drop(fixture, timeout_seconds=ctx.host_interval * 8 + 120)
    receipt: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics=target.metrics)
        _merge_receipt(receipt, current, target, ctx.settings.node)
        incidents = finding_incidents(current, target.mismatch_metric)
        if not incidents:
            return None
        ids = [str(item["incident_id"]) for item in incidents]
        from scripts.e2e.regional.general_live_verdicts import settled

        return current if settled(current, ids) else None

    observed = ctx.wait_until(
        accept,
        timeout_seconds=ctx.spec.max_minutes * 60,
        poll_seconds=10,
        name="restart-settle",
    )
    if observed is None:
        observed = activity(ctx, started, metrics=target.metrics)
        _merge_receipt(receipt, observed, target, ctx.settings.node)
    ctx.evidence.write("restart-receipt", receipt)

    # The replacement Pod on the node after the restart (census), and whether
    # the product deleted the original.
    final_pods = fixture.plugin_pods_on_node()
    final_uids = [str(pod["metadata"].get("uid") or "") for pod in final_pods]
    original_gone = fixture.original_pod_uid not in final_uids

    stages = {
        "restart_chain": chain_errors(
            observed,
            ChainExpectation(
                metric=target.mismatch_metric,
                present=True,
                operations=(
                    target.restart_operation,
                    "TRIGGER_HEALTH_SNAPSHOT",
                    target.validate_operation,
                ),
                validation=target.validate_operation,
                incident_states=frozenset({"RECOVERED"}),
            ),
        ),
        "deleted_original": _deleted_errors(receipt, fixture, original_gone),
        "replacement_ready": _replacement_errors(
            receipt, fixture, final_pods, final_uids
        ),
        "allocatable_restored": _allocatable_errors(observed, fixture),
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
    }
    if grade_replacement:
        stages["no_redelete"] = _no_redelete_errors(receipt, fixture, final_uids)
    return _result(
        stages,
        notes=[
            f"froze the {target.kind} plugin process (pod {fixture.original_pod_uid}); "
            "the product deleted and replaced the Pod and re-registered the resource",
            "deleted_pod_uid is captured from the live step receipt; the final "
            "step receipt keeps only replacement_pod_uids "
            "(node_operations drops the deleted UID), so the delete is also "
            "proven by the Pod census",
        ],
        extra={
            "restart_receipt": receipt,
            "original_pod_uid": fixture.original_pod_uid,
            "final_plugin_pod_uids": final_uids,
            "baseline_allocatable": fixture.baseline_allocatable,
        },
    )


def gpu_plugin(ctx: VariantContext) -> dict[str, Any]:
    """Freeze the GPU device plugin; the product restarts it, no reboot."""

    return _restart_variant(ctx, GPU_TARGET, grade_replacement=False)


def efa_plugin(ctx: VariantContext) -> dict[str, Any]:
    """Freeze the EFA device plugin; the product restarts it, no reboot."""

    return _restart_variant(ctx, efa_target(), grade_replacement=False)


def replacement_uid(ctx: VariantContext) -> dict[str, Any]:
    """The restart's replacement Pod UID is recorded, new, and not re-deleted."""

    return _restart_variant(ctx, GPU_TARGET, grade_replacement=True)


def _deleted_errors(
    receipt: dict[str, Any],
    fixture: DevicePluginRegistrationFixture,
    original_gone: bool,
) -> list[str]:
    deleted = receipt.get("deleted_pod_uid")
    if deleted:
        if deleted != fixture.original_pod_uid:
            return [
                f"the restart deleted pod {deleted}, not the frozen plugin pod "
                f"{fixture.original_pod_uid}"
            ]
        return []
    # The transient delete receipt was not caught; the delete still has to be
    # proven on the product's effect: the frozen Pod is gone and the restart
    # step ran (the chain stage asserts the operation).
    if not original_gone:
        return [
            f"the frozen plugin pod {fixture.original_pod_uid} is still on the node; "
            "no delete receipt and no census proof of the delete"
        ]
    return []


def _replacement_errors(
    receipt: dict[str, Any],
    fixture: DevicePluginRegistrationFixture,
    final_pods: list[dict[str, Any]],
    final_uids: list[str],
) -> list[str]:
    replacements = receipt.get("replacement_pod_uids") or []
    errors = []
    if not replacements:
        errors.append("the restart step recorded no replacement_pod_uids")
    if fixture.original_pod_uid in replacements:
        errors.append(
            f"replacement_pod_uids names the deleted pod {fixture.original_pod_uid}"
        )
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
    """One allocatable mismatch is not confirmed into an action (2-sample rule).

    The plugin is frozen only until the product records a single mismatch
    sample (``consecutive_mismatch_samples=1``) and resumed before the second,
    so the confirmation window never closes: no finding, no restart, no delete.
    """

    target = GPU_TARGET
    run_survey(ctx)
    fixture = DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    started = utc_now()
    fixture.freeze_pod(
        fixture.original_pod_uid, seconds=min(ctx.spec.max_minutes * 60, 1500)
    )
    _await_allocatable_drop(fixture, timeout_seconds=ctx.host_interval * 8 + 120)

    def one_mismatch() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics=target.metrics)
        for item in samples(current, target.mismatch_metric):
            consecutive = (item.get("labels") or {}).get("consecutive_mismatch_samples")
            if str(consecutive) == "1":
                return current
            if item.get("value") == 1 or str(consecutive) in {"2", "3"}:
                # The confirmation already closed before the runner could
                # resume: the single-sample window was not achieved.
                return current
        return None

    reached = ctx.wait_until(
        one_mismatch,
        timeout_seconds=ctx.host_interval * 12 + 120,
        poll_seconds=5,
        name="one-sample",
    )
    fixture.resume_now()
    if reached is None:
        raise VariantInconclusive(
            "the product never recorded an allocatable mismatch sample; the "
            "single-sample window could not be judged"
        )
    # ``reached`` is the document that caught the mismatch sample; grade the
    # single-sample assertions on it. The post-resume read proves the product
    # minted no finding and took no action (an incident, had one been minted,
    # would persist into it).
    mismatch_samples = [
        {
            "observed_at": item["observed_at"],
            "value": item["value"],
            "consecutive": (item.get("labels") or {}).get(
                "consecutive_mismatch_samples"
            ),
        }
        for item in samples(reached, target.mismatch_metric)
    ]
    confirmed = [
        item
        for item in mismatch_samples
        if item["value"] == 1 or str(item["consecutive"]) in {"2", "3"}
    ]
    if confirmed:
        raise VariantInconclusive(
            "the mismatch reached the 2-sample confirmation before the runner "
            f"resumed the plugin: {confirmed}"
        )
    seen_one = [item for item in mismatch_samples if str(item["consecutive"]) == "1"]
    if not seen_one:
        raise VariantInconclusive(
            "no single allocatable mismatch sample (consecutive=1) was recorded"
        )
    observed = activity(ctx, started, metrics=target.metrics)
    stages = {
        "single_sample_only": [
            f"a confirmed mismatch sample was recorded: {item}" for item in confirmed
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=target.mismatch_metric, present=False)
        )
        + chain_errors(
            reached, ChainExpectation(metric=target.mismatch_metric, present=False)
        ),
        "no_action": _no_plugin_restart_errors(observed, target)
        + _no_plugin_restart_errors(reached, target),
        "not_isolated": _isolation_errors(ctx, observed),
    }
    return _result(
        stages,
        notes=[
            "the plugin was frozen only until one mismatch sample "
            f"(consecutive_mismatch_samples=1) and resumed before the second; "
            f"the 2-sample rule left no finding ({len(seen_one)} single sample(s))",
        ],
        extra={"mismatch_samples": mismatch_samples[-4:]},
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
# validation_failure
# --------------------------------------------------------------------------- #
def validation_failure(ctx: VariantContext) -> dict[str, Any]:
    """A restart whose allocatable snapshot never confirms escalates to reboot.

    The registration is kept broken across the replacement (each new plugin Pod
    is re-frozen), so the ``RESTART_*_DEVICE_PLUGIN`` step's own post-restart
    allocatable snapshot never reaches the expected count: the step fails at
    ``restart_timeout_seconds`` and -- tagged
    ``failure_escalation_action=REBOOT_NODE`` -- escalates to ``RESTART_NODE``
    under the separate ``NODE_REBOOT`` authorization.
    """

    target = GPU_TARGET
    run_survey(ctx)
    fixture = DevicePluginRegistrationFixture(ctx, target)
    fixture.discover()
    started = utc_now()
    deadman = min(ctx.spec.max_minutes * 60, 1500)
    fixture.freeze_pod(fixture.original_pod_uid, seconds=deadman)
    _await_allocatable_drop(fixture, timeout_seconds=ctx.host_interval * 8 + 120)

    def escalated() -> dict[str, Any] | None:
        # Re-break each replacement Pod so the restart can never confirm.
        for pod in fixture.plugin_pods_on_node():
            uid = str(pod["metadata"].get("uid") or "")
            if uid and uid not in fixture.frozen_uids and _pod_running(pod):
                try:
                    fixture.freeze_pod(uid, seconds=deadman)
                except VariantInconclusive:
                    # The replacement is not yet schedulable/visible as a
                    # process; the next poll retries.
                    pass
        current = activity(ctx, started, metrics=target.metrics)
        return current if _reboot_escalation(current, ctx.settings.node) else None

    observed = ctx.wait_until(
        escalated,
        timeout_seconds=ctx.spec.max_minutes * 60,
        poll_seconds=10,
        name="validation-failure-escalation",
    )
    if observed is None:
        observed = activity(ctx, started, metrics=target.metrics)
    escalations = _reboot_escalation(observed, ctx.settings.node)
    restart_failed = any(
        execution.get("status") == "FAILED"
        for execution in _restart_executions(observed, target.restart_operation)
    )
    stages = {
        "restart_failed": (
            []
            if restart_failed
            else [f"no {target.restart_operation} step failed to confirm the restart"]
        ),
        "reboot_escalation": (
            []
            if escalations
            else [
                "no RESTART_NODE escalation (event id 'reboot-after-<restart>') "
                "followed the failed device-plugin restart"
            ]
        ),
        "separate_authorization": (
            []
            if NODE_REBOOT in ctx.spec.approvals
            else ["the reboot escalation is not gated on a separate NODE_REBOOT token"]
        ),
    }
    return _result(
        stages,
        notes=[
            "the registration was re-broken across the replacement so the "
            "restart's allocatable snapshot never confirmed; the product "
            "escalated to RESTART_NODE (separate NODE_REBOOT authorization)",
        ],
        extra={
            "reboot_escalations": [item["incident_id"] for item in escalations],
            "original_pod_uid": fixture.original_pod_uid,
            "frozen_pod_uids": sorted(fixture.frozen_uids),
        },
    )


def _reboot_escalation(activity_doc: dict[str, Any], node: str) -> list[dict[str, Any]]:
    """Incidents whose event id is a reboot escalation planning RESTART_NODE."""

    result = []
    for incident in activity_doc.get("incidents") or []:
        origin = escalation_origin(str(incident.get("event_id") or ""))
        if origin is None or origin[0] != "reboot":
            continue
        if node not in (incident.get("node_ids") or []):
            continue
        plans_reboot = any(
            "RESTART_NODE" in operations(workflow)
            for workflow in workflows_of(activity_doc, str(incident["incident_id"]))
        )
        if plans_reboot:
            result.append(incident)
    return result


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
#: Launcher/shim basenames that are never the plugin itself; a ``/bin/sh -c``
#: entrypoint must not widen the set of processes the probe may signal.
_BINARY_NOISE = frozenset(
    {"sh", "bash", "dash", "ash", "env", "tini", "dumb-init", "pause", "exec"}
)
_SAFE_BINARY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def plugin_binary_candidates(pod: dict[str, Any]) -> tuple[str, ...]:
    """Process basenames the plugin container might run, from the Pod spec.

    The probe's static allow-list is provider-blind; these are the discovered
    Pod's *own* declared command and image basenames, passed as
    ``--pod-command-binary`` so the probe matches the real plugin process (the
    HyperPod EFA plugin's basename is none of the static names) while still
    requiring the pod-UID cgroup match. Shim names are dropped.
    """

    candidates: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        base = value.rsplit("/", 1)[-1]
        if base and base not in _BINARY_NOISE and _SAFE_BINARY.fullmatch(base):
            if base not in candidates:
                candidates.append(base)

    spec = pod.get("spec") or {}
    for container in spec.get("containers") or []:
        command = container.get("command") or []
        if command:
            add(command[0])
        image = container.get("image")
        if isinstance(image, str):
            last = image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
            add(last)
    return tuple(candidates)


def _pod_ready(pod: dict[str, Any]) -> bool:
    conditions = (pod.get("status") or {}).get("conditions") or []
    return any(
        str(item.get("type")) == "Ready" and str(item.get("status")).lower() == "true"
        for item in conditions
    )


def _pod_running(pod: dict[str, Any]) -> bool:
    return str((pod.get("status") or {}).get("phase")) == "Running"


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    isolation = isolation_operations(observed)
    errors = [f"isolation operations planned: {isolation}"] if isolation else []
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("the node is isolated after a device-plugin registration fault")
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
    lanes: tuple[str, ...],
    risk: str,
    flags: str,
    max_minutes: int,
    approvals: tuple[str, ...],
    cleanup: str,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_014,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags=flags,
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        approvals=approvals,
        execute=execute,
    )


_CLEANUP = (
    "resume the frozen plugin process (deadman-backed SIGCONT); prove allocatable "
    "back to baseline and exactly one plugin Pod on the node"
)


VARIANTS = [
    _spec(
        "gpu_plugin",
        injection=(
            "SIGSTOP the nvidia-device-plugin container process on the node (its "
            "pod-UID + binary are the only ones the probe will signal); kubelet "
            "de-registers nvidia.com/gpu while the Pod stays Running"
        ),
        expected=(
            "RESTART_GPU_DEVICE_PLUGIN deletes the frozen Pod and waits for a new "
            "Ready one (replacement_pod_uids != deleted), TRIGGER_HEALTH_SNAPSHOT + "
            "VALIDATE_GPU SUCCEEDED, allocatable back to baseline, incident RECOVERED, "
            "no reboot, no isolation"
        ),
        execute=gpu_plugin,
        lanes=("eks", "hp"),
        risk="H",
        flags="S",
        max_minutes=30,
        approvals=(DEVICE_PLUGIN_RESTART,),
        cleanup=_CLEANUP,
    ),
    _spec(
        "efa_plugin",
        injection=(
            "SIGSTOP the EFA device-plugin container process on the node; kubelet "
            "de-registers vpc.amazonaws.com/efa while the Pod stays Running"
        ),
        expected=(
            "RESTART_EFA_DEVICE_PLUGIN deletes the frozen Pod and waits for a new "
            "Ready one, TRIGGER_HEALTH_SNAPSHOT + VALIDATE_FABRIC SUCCEEDED, "
            "allocatable back to baseline, incident RECOVERED, no reboot, no isolation"
        ),
        execute=efa_plugin,
        lanes=("eks", "hp"),
        risk="H",
        flags="S",
        max_minutes=30,
        approvals=(DEVICE_PLUGIN_RESTART,),
        cleanup=_CLEANUP,
    ),
    _spec(
        "one_sample",
        injection=(
            "SIGSTOP the GPU device-plugin process only until the product records "
            "one allocatable mismatch sample (consecutive_mismatch_samples=1), then "
            "SIGCONT before the second sample"
        ),
        expected=(
            "the single mismatch is below required_consecutive_samples=2: no "
            "allocatable_mismatch finding, no RESTART_GPU_DEVICE_PLUGIN, no delete, "
            "no isolation; allocatable restored"
        ),
        execute=one_sample,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        max_minutes=20,
        approvals=(DEVICE_PLUGIN_RESTART,),
        cleanup=_CLEANUP,
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
        lanes=("eks", "hp"),
        risk="H",
        flags="S",
        max_minutes=30,
        approvals=(DEVICE_PLUGIN_RESTART,),
        cleanup=_CLEANUP,
    ),
    _spec(
        "validation_failure",
        injection=(
            "SIGSTOP the GPU device-plugin process and re-SIGSTOP every replacement "
            "Pod so allocatable never returns to the expected count across the "
            "restart's confirmation window"
        ),
        expected=(
            "the RESTART_GPU_DEVICE_PLUGIN step fails (allocatable never confirmed) "
            "and escalates to RESTART_NODE (event id 'reboot-after-<restart>') under "
            "the separate NODE_REBOOT authorization"
        ),
        execute=validation_failure,
        lanes=("eks", "hp"),
        risk="H",
        flags="X",
        max_minutes=60,
        approvals=(DEVICE_PLUGIN_RESTART, NODE_REBOOT),
        cleanup=_CLEANUP,
    ),
]
