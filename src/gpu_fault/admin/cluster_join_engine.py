"""Cluster-scoped verify and sync-state for the join's commit path.

After the release engine has joined a cluster, the admin side used to run two
site-wide engine modes as subprocesses: ``verify`` (every admin health check,
57 s on 2026-09-12, 27 s of it the alert-rule validation) and ``sync-state``
(a full ``capture_previous`` of every cluster, 74 s, of which ``sync-state``
consumes exactly one field: ``live_runtime_image``). Both re-read a site the
join changed in one known place.

This module composes the engine's own primitives in-process (the precedent is
``rotate_token.build_release``) so the join pays for the cluster it added:

* :func:`verify_joined_clusters` runs the health checks that read the joined
  cluster and skips the six site-wide checks that demonstrably do not
  (``JOIN_VERIFY_SKIPPED_CHECKS`` says why, one by one). Its ``control_api``
  check honours the ``COLLECTORS_READY`` gate's deferral of slow collector
  kinds for the joined cluster, where the engine's whole-set rule would refuse
  a cluster whose DCGM summary has not landed yet.
* :func:`sync_cluster_release_state` captures the joined cluster's snapshot,
  merges it with the committed release state's identity for the clusters the
  join did not touch, evaluates the site-level invariants (live Agent set ==
  HyperPod node set per cluster, one runtime image everywhere) on that merged
  document and hands it to the engine's ``sync_release_state`` through the
  ``_capture_previous`` seam the engine already exposes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_readiness import evaluate_join_readiness
from gpu_fault.admin.rotate_token import build_release, site_process_environment
from gpu_fault.admin.site import RenderedSite
from gpu_fault_release import regional_admin_checks as checks
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_images import (
    previous_node_installer_image,
    require_consistent_images,
)
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_state import (
    capture_agent_identities,
    capture_gpu_cluster_snapshot,
    deployment_image,
)

# Site-wide checks the join verify leaves to ``gpu-fault-admin verify``. Each
# entry names what the check reads and why none of it is the joined cluster;
# a check is listed here only when that is demonstrable from its source.
JOIN_VERIFY_SKIPPED_CHECKS: dict[str, str] = {
    "email_notifications": (
        "check_email_notifications reads the site's notification settings, "
        "CPU Secrets and SES/SNS channel state; it takes no cluster input"
    ),
    "aurora_credential_refresh": (
        "check_aurora_refresh reads the refresher CronJob, Role, Jobs and the "
        "gpu-fault-aurora Secret on the CPU cluster; it takes no cluster input"
    ),
    "adot_self_metrics": (
        "adot_self_metrics_report reads the control-plane collector's Pods "
        "(app=gpu-fault-adot on the CPU cluster) and AMP series of job "
        "gpu-fault-adot-self; the joined cluster's data-plane collector writes "
        "job gpu-fault-dataplane, which the check never queries"
    ),
    "nlb_runtime": (
        "_check_nlb_runtime reads the NLB, its TLS listener, certificate and "
        "target health (CPU nodes); the join only added security-group ingress "
        "CIDRs, which the check does not read"
    ),
    "aurora": (
        "_check_aurora reads the RDS cluster and instance status; it takes no "
        "cluster input"
    ),
    "monitoring": (
        "_check_monitoring reads the AMP workspace status, the static rule "
        "namespace health.amp_rule_namespace, the Alertmanager definition and "
        "the SNS subscriptions, then runs scripts/verify-regional-alerting.py "
        "over them (about 27 s); the join's _apply_observability writes the "
        "per-cluster absence rule into gpu-fault-dataplane-expected, a "
        "namespace this check never describes"
    ),
}

# The checks the join verify keeps, with the joined-cluster input each reads.
JOIN_VERIFY_KEPT_CHECKS: dict[str, str] = {
    "regional_contexts": "validates every cluster's kube context and HyperPod",
    "cpu_secrets": "gpu-fault-node-action-keys carries the joined nodes' keys",
    "cpu_workloads": (
        "the join's _apply_observability may re-apply the CPU collector; the "
        "config-core profile version must match the candidate site"
    ),
    "runtime_profile": "the join runs ensure_runtime_profile",
    "read_only_verifiers": "runs verify_dataplane_executor on every GPU cluster",
    "runtime_component_identity": "execs into every runtime Pod, joined included",
    "control_api": (
        "registry membership, Agent coverage and collector readiness of the "
        "joined cluster (deferred slow kinds honoured, see relax_control_api_report)"
    ),
    "gpu_cluster:<cluster_id>": (
        "one per configured cluster: the candidate verify is the only verify a "
        "join runs over the existing clusters (see PRECHECKED)"
    ),
}

_CONTROL_API_PROBE = "control_api_inspect"


def relax_control_api_report(
    text: str,
    readiness: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Apply the join's readiness rule to the control-API probe output.

    ``readiness`` maps a joined cluster id to its ``COLLECTORS_READY`` evidence.
    For a cluster whose gate deferred slow kinds, the probe's whole-set
    ``collector_readiness.ready`` is replaced by the join's verdict over the
    same payload (fast kinds reported, slow kinds scheduled, fleet ready over
    the expected nodes). Clusters without deferred kinds, and every other field
    of the report, pass through untouched. Returns the rewritten document and
    what was relaxed, for the verify evidence.
    """

    document = json.loads(text)
    relaxed: dict[str, Any] = {}
    clusters = document.get("clusters") if isinstance(document, dict) else None
    if not isinstance(clusters, dict):
        return text, relaxed
    for cluster_id, evidence in readiness.items():
        if not evidence.get("deferred_kinds"):
            continue
        cluster = clusters.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        payload = cluster.get("collector_readiness")
        if not isinstance(payload, dict) or payload.get("ready") is True:
            continue
        verdict = evaluate_join_readiness(
            payload,
            expected_nodes=[
                str(item) for item in cluster.get("expected_node_ids") or []
            ],
            fleet=cluster.get("fleet_readiness"),
        )
        if not verdict.ready:
            continue
        payload["ready"] = True
        payload["join_deferred_kinds"] = verdict.evidence()["deferred_kinds"]
        # What the relaxation actually excused: the deferred kinds some node had
        # not reported yet when verify ran.
        relaxed[cluster_id] = sorted(
            kind
            for kind in verdict.deferred_kinds
            if any(
                not node["deferred"].get(kind, {}).get("ready", True)
                for node in verdict.nodes.values()
            )
        )
    if not relaxed:
        return text, relaxed
    return json.dumps(document, separators=(",", ":")), relaxed


class _ReadinessRunner:
    """The engine runner, with the control-API probe's readiness relaxed."""

    def __init__(
        self,
        inner: Any,
        readiness: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._inner = inner
        self._readiness = readiness
        self.relaxed: dict[str, Any] = {}

    def run(self, args: list[str], **kwargs: Any) -> str:
        output = str(self._inner.run(args, **kwargs) or "")
        if probe_source(_CONTROL_API_PROBE) not in args:
            return output
        rewritten, relaxed = relax_control_api_report(output, self._readiness)
        self.relaxed.update(relaxed)
        return rewritten

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ReleaseView:
    """The release with a different runner; every other attribute is shared."""

    def __init__(self, release: Any, runner: Any) -> None:
        object.__setattr__(self, "_release", release)
        object.__setattr__(self, "runner", runner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._release, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._release, name, value)


def join_verify_specifications(
    release: Any,
    readiness: Mapping[str, Mapping[str, Any]],
) -> tuple[list[tuple[str, Any]], _ReadinessRunner]:
    """The kept checks, in the engine's order, bound to ``release``."""

    runner = _ReadinessRunner(release.runner, readiness)
    view = _ReleaseView(release, runner)
    specifications: list[tuple[str, Any]] = [
        ("regional_contexts", lambda: checks._check_contexts(release)),
        ("cpu_secrets", lambda: checks.check_cpu_secrets(release)),
        ("cpu_workloads", lambda: checks._check_cpu_workloads(release)),
        ("runtime_profile", lambda: checks._verify_profile(release)),
        ("read_only_verifiers", lambda: checks.run_read_only_verifiers(release)),
        (
            "runtime_component_identity",
            lambda: checks._check_runtime_component_identity(release),
        ),
        ("control_api", lambda: checks._check_control_api(view)),
    ]
    specifications.extend(
        (
            f"gpu_cluster:{target.cluster_id}",
            lambda target=target: checks._check_gpu_cluster(release, target),
        )
        for target in release.config.clusters
    )
    return specifications, runner


def verify_joined_clusters(
    candidate: RenderedSite,
    *,
    readiness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the join-scoped health report against the candidate site.

    ``readiness`` maps each cluster this join is verifying to its
    ``COLLECTORS_READY`` evidence. Prints the report like the engine's
    ``verify`` mode and raises :class:`BootstrapError` when a check fails.
    """

    with site_process_environment(candidate):
        release = build_release(candidate)
        release._apply_health_baseline(release._load_state())
        specifications, runner = join_verify_specifications(release, readiness)
        with release._read_snapshot():
            release._prime_deployment_snapshot()
            with ThreadPoolExecutor(
                max_workers=min(8, len(specifications))
            ) as executor:
                futures = [
                    executor.submit(checks._check, name, function)
                    for name, function in specifications
                ]
                results = [future.result() for future in futures]
    report = checks._report("verify", release, results)
    report["scope"] = {
        "joined_clusters": sorted(readiness),
        "skipped_checks": dict(JOIN_VERIFY_SKIPPED_CHECKS),
        "relaxed_collector_readiness": runner.relaxed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("healthy"):
        failed = "; ".join(
            f"{item['name']}: {item['summary']}"
            for item in results
            if item["status"] == "FAIL"
        )
        raise BootstrapError("join candidate verify failed: " + failed)
    return report


def verify_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a verify report the join records in its ``VERIFIED`` evidence."""

    scope = cast(Mapping[str, Any], report.get("scope") or {})
    return {
        "summary": dict(report.get("summary") or {}),
        "checks": sorted(
            str(item.get("name"))
            for item in report.get("checks") or []
            if isinstance(item, Mapping)
        ),
        "skipped_checks": sorted(scope.get("skipped_checks") or {}),
        "relaxed_collector_readiness": dict(
            scope.get("relaxed_collector_readiness") or {}
        ),
    }


def _committed_runtime_image(live_state: Mapping[str, Any]) -> str:
    """The runtime image the committed release state says the site runs.

    After a rollback that PASSED the state still names the failed candidate and
    the live truth is the ``previous`` snapshot (the engine's ``_live_truth_state``
    rule); an adopted legacy image wins over both.
    """

    truth: Mapping[str, Any] = live_state
    previous = live_state.get("previous")
    rollback = live_state.get("rollback_result")
    if (
        str(live_state.get("phase") or "") == "rolled-back"
        and isinstance(previous, Mapping)
        and isinstance(rollback, Mapping)
        and rollback.get("status") == "PASSED"
    ):
        truth = previous
    return str(
        live_state.get("adopted_live_runtime_image")
        or truth.get("live_runtime_image")
        or truth.get("runtime_image")
        or ""
    ).strip()


def capture_joined_cluster_previous(release: Any, cluster_id: str) -> dict[str, Any]:
    """The merged ``previous`` document for a join's ``sync-state``.

    Fresh reads: the joined cluster's snapshot and runtime images, the CPU
    runtime images, every cluster's active Agent identities (one probe) and
    HyperPod node names (one ``get nodes`` each). Everything else the full
    capture would re-read is unchanged by the join and comes from the committed
    release state. The invariants ``_capture_previous`` raises are evaluated on
    the merged document:

    * one runtime image across the CPU deployments, the joined cluster's
      deployments and the image the committed state records for the clusters
      the join did not touch;
    * each cluster's live Agent set equals its HyperPod node set.
    """

    target = release._target(cluster_id)
    with release._read_snapshot():
        release._prime_deployment_snapshot()
        live_state = dict(release._load_state())
        snapshot, cluster_images, installer_image = capture_gpu_cluster_snapshot(
            release, target
        )
        images: dict[str, str | None] = {
            f"cpu/{deployment}": deployment_image(release, release._cpu(), deployment)
            for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        }
        images.update(cluster_images)
        committed = _committed_runtime_image(live_state)
        if committed:
            images["release-state/runtime_image"] = committed
        live_runtime_image = require_consistent_images("runtime", images)
        node_installer_image = previous_node_installer_image(
            capture_gpu=True,
            images={cluster_id: installer_image},
            recorded=str(live_state.get("node_installer_image") or ""),
            configured=str(getattr(release, "node_installer_image", "") or ""),
        )
        agent_identities = capture_agent_identities(release)
        clusters = list(release.config.clusters)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(clusters)))) as pool:
            node_sets = list(
                pool.map(
                    lambda item: set(release._target_node_names(item)),
                    clusters,
                )
            )
    for item, expected_nodes in zip(clusters, node_sets, strict=True):
        captured_nodes = set(agent_identities[item.cluster_id]["node_ids"])
        if expected_nodes != captured_nodes:
            raise ReleaseError(
                f"{item.cluster_id} active Agent set does not match HyperPod nodes"
            )
    return {
        "capture_scope": f"cluster:{cluster_id}",
        "live_runtime_image": live_runtime_image,
        "runtime_image": committed or live_runtime_image,
        "node_installer_image": node_installer_image,
        "agent_identities": agent_identities,
        "clusters": {cluster_id: snapshot},
        "agent_nodes": {
            item.cluster_id: len(agent_identities[item.cluster_id]["node_ids"])
            for item in clusters
        },
    }


def sync_cluster_release_state(
    site: RenderedSite,
    *,
    cluster_id: str,
) -> dict[str, Any]:
    """``sync-state`` for a join, capturing only the joined cluster.

    The engine's ``sync_release_state`` consumes ``_capture_previous()`` for
    ``live_runtime_image`` and writes the committed NOOP state (``previous``
    is ``None``); the seam is overridden on this release instance so the
    engine's save path, narration and history stay its own.
    """

    # ``rollout`` loads the whole engine; imported where it is needed, like
    # ``rotate_token.build_release`` does, so the admin CLI does not pay for it
    # at start-up.
    from gpu_fault_release.rollout import sync_release_state

    with site_process_environment(site):
        release = build_release(site)
        document = capture_joined_cluster_previous(release, cluster_id)
        release._capture_previous = lambda plan=None: document
        sync_release_state(release)
    return {
        "capture_scope": document["capture_scope"],
        "live_runtime_image": document["live_runtime_image"],
        "agent_nodes": document["agent_nodes"],
    }
