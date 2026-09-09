"""The per-GPU-cluster half of the observability component (F10 fix rounds 1-2).

Two things live here, both keyed on the same set: the clusters whose target
carries an ``adot_irsa_role_arn`` while the site has an AMP workspace, which is
exactly the set the observability digest already folds.

1. **Expected-collector rules.** The static ``GpuFaultDataplaneCollectorMissing``
   rule used to carry a global ``absent(up{job="gpu-fault-dataplane"} == 1)``
   half, which fires forever on a site with a workspace and no collector yet --
   every deployed site today -- and pages SNS every four hours for a component
   the site has deliberately not enabled. The absence half is rendered here
   instead, one ``absent(...)`` per expected cluster, and put by the installer
   under its own AMP rule-groups namespace so the static namespace stays
   byte-stable. No expected cluster means the namespace is deleted, so a rule
   for a cluster whose role was removed cannot linger. The per-cluster shape is
   also what closes the blind spot the old comment admitted: one dead collector
   beside a live one produces no series at all, and only a rule that names the
   cluster can see that. The rules are re-put wherever the expected set can
   move: the OBSERVABILITY node of an upgrade, and the engine's own bootstrap,
   ``join_cluster`` and ``remove_cluster`` (``apply_observability``); the admin
   bootstrap runs the installer bare and leaves the namespace alone.

2. **Rollback compensation.** The first wiring put every configured cluster's
   collector back by rendering the *candidate's* manifest with the previous
   image: the keep filters, relabels and scrape targets stayed the candidate's,
   collectors were created on clusters the candidate never touched, and a
   cluster whose role the candidate had just added kept its collector. The
   compensation is now a snapshot, taken beside the control-plane one: every
   GPU cluster in the candidate config has its declared collector objects read
   through that cluster's own kube context before the OBSERVABILITY node runs,
   and rollback restores each cluster from ITS snapshot -- objects applied back
   where a collector was live, the candidate's objects deleted where none was.
   Because the capture iterates the candidate config, a cluster the candidate
   ADDED is in the snapshot too (every object absent, so it reads ``removed``);
   the only cluster the snapshot never saw is one added to site.yaml between the
   transaction's opening and its rollback, and that one is scaled to zero. A
   cluster removed from site.yaml in that window refuses the rollback. The
   expected-rules namespace goes back too. The previous-image path survives
   only for a state captured before this snapshot existed, and the rollback
   record says which path ran.

   The data-plane half is its own rollback phase
   (``rollback-dataplane-observability-restored``), run AFTER the CPU control
   plane is restored: a GPU cluster whose API is unreachable is a common reason
   the release failed in the first place, and it must not keep the previous
   control plane from coming back. Inside the phase one cluster's failure does
   not stop the others or the expected-rules restore; the phase then fails once,
   naming every failed cluster, so the rollback is honestly FAILED and a resume
   re-runs the (idempotent) phase. The control-plane half stays in the earlier
   ``observability_restore`` phase, which validates the WHOLE snapshot -- both
   halves -- before it mutates anything.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release.regional_gpu_bootstrap import rollback_gpu_adot_collectors
from gpu_fault_release.regional_manifest_snapshot import (
    apply_snapshot_objects,
    capture_declared_objects,
    declared_manifest_objects,
    delete_absent_objects,
    restart_snapshot_deployments,
    snapshot_parts,
)
from gpu_fault_release.regional_observability_rollback import (
    capture_observability_snapshot,
)
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_rendering import (
    DATAPLANE_ADOT_DEPLOYMENT,
    DATAPLANE_ADOT_MANIFEST,
    dataplane_adot_skip_reason,
)

ROOT = Path(__file__).resolve().parents[2]
AMP_MONITORING_INSTALLER = ROOT / "deploy/observability/install-amp-monitoring.sh"
#: The rule-groups namespace the rendered per-cluster rules live in. Separate
#: from ``health.amp_rule_namespace`` (the static file) on purpose: the static
#: namespace is compared byte-for-byte against the checked-in file on every
#: install, and rendered rules in it would make every site read as drifted.
DATAPLANE_EXPECTED_RULE_NAMESPACE = "gpu-fault-dataplane-expected"
DATAPLANE_EXPECTED_RULE_GROUP = "gpu-fault-dataplane-expected"
DATAPLANE_COLLECTOR_ALERT = "GpuFaultDataplaneCollectorMissing"
DATAPLANE_COLLECTOR_RUNBOOK_URL = (
    "docs/管理员日常运维.md#gpufaultdataplanecollectormissing"
)
#: What the data-plane collector stamps on every series
#: (``deploy/dataplane/adot-dataplane.yaml``, relabel ``control_plane_cluster``);
#: ``absent()`` only carries the labels named in its selector, so the rendered
#: rule has to add it as a static label or the alert lands in its own
#: Alertmanager group. Pinned against the manifest by a test.
DATAPLANE_CONTROL_PLANE_CLUSTER = "gpu-fault-regional-control-plane"
DATAPLANE_ADOT_ROLLOUT_TIMEOUT = "300s"
MAX_CAPTURE_WORKERS = 8
#: The rollback phase that puts the per-cluster collectors and the rendered
#: rules back: its timing entry, its in-progress and its checkpoint phase name.
DATAPLANE_OBSERVABILITY_PHASE = "dataplane_observability_restore"
DATAPLANE_OBSERVABILITY_RESTORING = "rollback-dataplane-observability-restoring"
DATAPLANE_OBSERVABILITY_RESTORED = "rollback-dataplane-observability-restored"
#: AMP validates a rule namespace asynchronously and deletes it asynchronously:
#: a put on a CREATING/UPDATING/DELETING namespace is a ConflictException. The
#: restore waits for it to settle, polling this often, this many times (the
#: installer's own wait is 300 s; 60 x 5 s matches it).
EXPECTED_RULES_SETTLE_ATTEMPTS = 60
EXPECTED_RULES_SETTLE_SECONDS = 5.0
_SETTLING_STATUSES = frozenset({"CREATING", "UPDATING", "DELETING"})
_sleep: Callable[[float], None] = time.sleep
# The cluster id is interpolated into a PromQL string literal and a YAML
# scalar; anything outside this set is refused rather than escaped.
_CLUSTER_ID_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- expected-collector rules -------------------------------------------------------


def expected_collector_targets(
    release: Any, *, exclude: frozenset[str] = frozenset()
) -> tuple[ClusterTarget, ...]:
    """The clusters the release applies a collector to, in config order.

    ``exclude`` names clusters to leave out although the config still carries
    them: ``remove_cluster`` runs against a config that still names the cluster
    it is removing (that is how it finds the target), and the rule for that
    cluster has to go with its collector rather than fire until the next deploy.
    """

    return tuple(
        target
        for target in release.config.clusters
        if target.cluster_id not in exclude
        and dataplane_adot_skip_reason(release, target) is None
    )


def _expected_rule(target: ClusterTarget) -> dict[str, Any]:
    cluster_id = target.cluster_id
    if not _CLUSTER_ID_SAFE.fullmatch(cluster_id):
        raise ReleaseError(
            f"cluster_id {cluster_id!r} cannot be quoted into a PromQL selector"
        )
    return {
        "alert": DATAPLANE_COLLECTOR_ALERT,
        "expr": (
            f'absent(up{{job="gpu-fault-dataplane", gpu_cluster="{cluster_id}"}} == 1)'
        ),
        "for": "15m",
        "labels": {
            "severity": "warning",
            "gpu_cluster": cluster_id,
            "control_plane_cluster": DATAPLANE_CONTROL_PLANE_CLUSTER,
            "region": target.region,
        },
        "annotations": {
            "summary": (
                f"The data-plane metrics collector of GPU cluster {cluster_id} "
                "is not writing to AMP"
            ),
            "runbook_url": DATAPLANE_COLLECTOR_RUNBOOK_URL,
            "description": (
                f"No `up == 1` series from the data-plane ADOT collector of GPU "
                f"cluster {cluster_id} (job gpu-fault-dataplane) reached AMP for "
                "15 minutes, although the release config carries an "
                "adot_irsa_role_arn for it, so the collector was applied and is "
                "expected to write. While this fires the Completion Watcher "
                "alerts (GpuFaultCompletionActiveStateUnavailable, "
                f"GpuFaultCompletionOutboxAppendFailures) are blind for "
                f"{cluster_id} and a quiet dashboard is not evidence. Read "
                f"deployment/{DATAPLANE_ADOT_DEPLOYMENT} in gpu-fault-system on "
                f"{cluster_id} (IRSA token, sigv4 errors, the /health probe) and "
                "its Pod logs. If the role was removed on purpose, drop "
                "adot_irsa_role_arn from the cluster and re-run `gpu-fault-admin "
                "deploy --state-dir STATE_DIR`: the release scales the collector "
                "down and re-renders these rules without this one. This rule is "
                "rendered per expected cluster by the release, which is why one "
                "dead collector beside a live one is caught here although the "
                "static rule sees no series for it."
            ),
        },
    }


def render_dataplane_expected_rules(
    release: Any, *, exclude: frozenset[str] = frozenset()
) -> str | None:
    """The per-cluster absence rules as an AMP rule-groups document, or ``None``.

    ``None`` -- not an empty group -- when no cluster is expected to carry a
    collector: the installer then deletes the namespace, so nothing stale is
    left to fire.
    """

    targets = expected_collector_targets(release, exclude=exclude)
    if not targets:
        return None
    document = {
        "groups": [
            {
                "name": DATAPLANE_EXPECTED_RULE_GROUP,
                "rules": [_expected_rule(target) for target in targets],
            }
        ]
    }
    return str(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))


def dataplane_expected_rules_sha256(release: Any) -> str:
    """Digest input: the rendered rules text, so a template edit moves the
    observability digest and the next release re-puts the rules."""

    return hashlib.sha256(
        (render_dataplane_expected_rules(release) or "").encode("utf-8")
    ).hexdigest()


def amp_installer_arguments(
    release: Any, directory: Path, *, exclude: frozenset[str] = frozenset()
) -> list[str]:
    """The installer's positional arguments for this release's expected set.

    The rendered document is written into ``directory`` (a temporary directory
    the caller owns for the installer's lifetime) and handed over by path; an
    empty expected set is an explicit deletion request. Only the admin
    bootstrap (``bootstrap_services``) runs the installer with no arguments at
    all, which leaves the namespace alone.
    """

    rendered = render_dataplane_expected_rules(release, exclude=exclude)
    if rendered is None:
        return ["--no-dataplane-expected-rules"]
    path = directory / "dataplane-expected-rules.yaml"
    path.write_text(rendered, encoding="utf-8")
    return ["--dataplane-expected-rules", str(path)]


def run_amp_monitoring_installer(
    release: Any,
    environment: dict[str, str],
    *,
    exclude: frozenset[str] = frozenset(),
) -> None:
    """Run the AMP monitoring installer with this release's expected-rules hand-off.

    ``environment`` is the installer's configuration (built by the release from
    its config); the rendered rules file lives only for the installer's run.
    """

    with tempfile.TemporaryDirectory() as directory:
        arguments = amp_installer_arguments(release, Path(directory), exclude=exclude)
        release.runner.run(
            ["bash", str(AMP_MONITORING_INSTALLER), *arguments],
            env=environment,
        )


def amp_installer_environment(release: Any) -> dict[str, str]:
    """The installer's configuration, from the release config over the process env."""

    topic_name = release.config.health.sns_topic_arn.rsplit(":", 1)[-1]
    environment = {
        **os.environ,
        "AWS_REGION": release.config.aws_region,
        "CPU_EKS_CLUSTER": release.config.cpu_eks_arn.rsplit("/", 1)[-1],
        "CPU_KUBECONFIG": release.config.cpu_kubeconfig,
        "AMP_WORKSPACE_ID": release.config.health.amp_workspace_id,
        "SNS_TOPIC_NAME": topic_name,
        "NAMESPACE": release.config.namespace,
        "RULE_NAMESPACE": release.config.health.amp_rule_namespace,
        "GPU_FAULT_ADOT_IMAGE": release.adot_image,
        "GPU_FAULT_ENABLE_ADOT": "true",
        "GPU_FAULT_ENABLE_AMP": "true",
        "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION": str(
            release.config.health.require_confirmed_sns_subscription
        ).lower(),
    }
    if release.config.notifications.admin_email:
        environment["GPU_FAULT_ALERT_EMAIL"] = release.config.notifications.admin_email
    return environment


def apply_observability(
    release: Any, *, exclude_cluster_ids: frozenset[str] = frozenset()
) -> None:
    """The observability component's control-plane step (bound as
    ``RegionalRelease._apply_observability``).

    Runs the AMP monitoring installer -- the IAM writer role, the SNS topic and
    its policy, the static rule namespace, the Alertmanager definition and the
    control-plane collector, each short-circuited when already converged -- and
    hands it this release's rendered per-cluster expected-collector rules (one
    ``absent()`` per cluster with an IRSA role), or an explicit deletion when no
    cluster is expected to carry a collector. Re-put on every run: the upgrade
    runs this whenever the observability digest moves (the digest folds the
    rendered text, so a rule-template edit reaches AMP through the same node),
    and the engine's bootstrap, ``join_cluster`` and ``remove_cluster`` run it
    because each of them changes the expected set without a deploy.
    ``exclude_cluster_ids`` is ``remove_cluster``'s way of dropping the cluster
    its config still names.
    """

    run_amp_monitoring_installer(
        release, amp_installer_environment(release), exclude=exclude_cluster_ids
    )


def _amp_common(release: Any) -> list[str]:
    return [
        "--region",
        release.config.aws_region,
        "--workspace-id",
        str(release.config.health.amp_workspace_id),
    ]


def _describe_expected_rules(release: Any) -> list[str]:
    return [
        "aws",
        "amp",
        "describe-rule-groups-namespace",
        *_amp_common(release),
        "--name",
        DATAPLANE_EXPECTED_RULE_NAMESPACE,
    ]


def _describe_expected_rules_document(release: Any) -> dict[str, Any] | None:
    """The namespace's describe document, ``None`` when it does not exist.

    Only ``ResourceNotFoundException`` is absence. A throttle, a credentials
    error or an unreadable answer raises: a restore built on a guess would
    create over a namespace that exists (ConflictException) or skip deleting
    one that does.
    """

    code, stdout, stderr = release.runner.probe_output(
        [*_describe_expected_rules(release), "--output", "json"]
    )
    if code:
        if "ResourceNotFoundException" in stderr:
            return None
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: {stderr.strip() or code}"
        )
    try:
        document = json.loads(stdout)["ruleGroupsNamespace"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: unexpected describe output"
        ) from exc
    if not isinstance(document, dict):
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: unexpected describe output"
        )
    return document


def _expected_rules_status(document: dict[str, Any] | None) -> str | None:
    if document is None:
        return None
    status = document.get("status")
    return str(status.get("statusCode") or "") if isinstance(status, dict) else ""


def capture_dataplane_expected_rules(release: Any) -> dict[str, Any]:
    """Read the expected-rules namespace as it is before the release re-puts it.

    Absence is a legitimate previous state (no cluster had a role); an
    unreadable namespace is not, because a rollback built on a guess would put
    back the wrong rules or delete rules that were there.
    """

    document = _describe_expected_rules_document(release)
    if document is None:
        return {"present": False, "data_base64": None}
    data = document.get("data")
    if not isinstance(data, str) or not data:
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: unexpected describe output"
        )
    return {"present": True, "data_base64": data}


def _settled_expected_rules_exist(release: Any) -> bool:
    """Whether the namespace exists once AMP has stopped changing it.

    A namespace the candidate's installer just put or deleted may still be
    CREATING/UPDATING/DELETING; writing to it now is a ConflictException, so
    the restore waits for AMP to settle first and fails closed if it never
    does. A failed definition (``CREATION_FAILED``, ``UPDATE_FAILED``) exists
    and can be put over.
    """

    status = _expected_rules_status(_describe_expected_rules_document(release))
    for _attempt in range(EXPECTED_RULES_SETTLE_ATTEMPTS):
        if status not in _SETTLING_STATUSES:
            return status is not None
        _say(
            f"AMP rule namespace {DATAPLANE_EXPECTED_RULE_NAMESPACE} is {status}; "
            "waiting for it to settle before restoring it"
        )
        _sleep(EXPECTED_RULES_SETTLE_SECONDS)
        status = _expected_rules_status(_describe_expected_rules_document(release))
    raise ReleaseError(
        f"AMP rule namespace {DATAPLANE_EXPECTED_RULE_NAMESPACE} is still {status} "
        f"after {EXPECTED_RULES_SETTLE_ATTEMPTS * EXPECTED_RULES_SETTLE_SECONDS:.0f}s"
    )


def _expected_rules_parts(snapshot: object) -> tuple[bool, bytes]:
    invalid = ReleaseError("previous expected-collector rules snapshot is invalid")
    if not isinstance(snapshot, dict) or "present" not in snapshot:
        raise invalid
    present = bool(snapshot["present"])
    if not present:
        return False, b""
    encoded = snapshot.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        raise invalid
    try:
        return True, base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise invalid from exc


def restore_dataplane_expected_rules(release: Any, snapshot: object) -> str:
    """Put the previous expected-rules namespace back; returns what was done.

    ``restored`` (put or created from the captured bytes), ``deleted`` (the
    previous release had none and the candidate put one) or ``absent`` (the
    previous release had none and there is nothing to delete).
    """

    present, data = _expected_rules_parts(snapshot)
    exists = _settled_expected_rules_exist(release)
    if not present:
        if not exists:
            return "absent"
        release.runner.run(
            [
                "aws",
                "amp",
                "delete-rule-groups-namespace",
                *_amp_common(release),
                "--name",
                DATAPLANE_EXPECTED_RULE_NAMESPACE,
            ]
        )
        return "deleted"
    verb = "put-rule-groups-namespace" if exists else "create-rule-groups-namespace"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dataplane-expected-rules.yaml"
        path.write_bytes(data)
        release.runner.run(
            [
                "aws",
                "amp",
                verb,
                *_amp_common(release),
                "--name",
                DATAPLANE_EXPECTED_RULE_NAMESPACE,
                "--data",
                f"fileb://{path}",
            ]
        )
    return "restored"


# --- collector snapshot ------------------------------------------------------------


def declared_dataplane_adot_objects() -> tuple[dict[str, str], ...]:
    """The objects ``adot-dataplane.yaml`` applies, in manifest order."""

    return declared_manifest_objects(
        (ROOT / DATAPLANE_ADOT_MANIFEST).read_text(encoding="utf-8"),
        label="data-plane ADOT manifest",
    )


def capture_dataplane_adot_snapshot(
    release: Any, target: ClusterTarget
) -> dict[str, Any]:
    """Read one cluster's live collector objects through that cluster's context.

    Unlike the control-plane collector, a missing Deployment is not an error:
    a cluster without a role has no collector, and the snapshot has to say so
    (every declared object ``absent``) so a rollback deletes what the candidate
    adds instead of pretending it was there.
    """

    captured = capture_declared_objects(
        release,
        declared_dataplane_adot_objects(),
        label=f"live data-plane ADOT object on {target.cluster_id}",
        kubectl=release._gpu(target),
    )
    present = any(str(item.get("kind")) == "Deployment" for item in captured["objects"])
    return {"present": present, **captured}


def capture_dataplane_adot_state(release: Any) -> dict[str, Any]:
    """Every cluster's collector snapshot (config order) plus the rules namespace."""

    targets = list(release.config.clusters)
    capture = release._capture_dataplane_adot_snapshot
    if len(targets) < 2:
        snapshots = [(target, capture(target)) for target in targets]
    else:
        # Read-only and scoped to one cluster each, so they run concurrently;
        # the results are merged in config order to keep the snapshot stable.
        with ThreadPoolExecutor(
            max_workers=min(MAX_CAPTURE_WORKERS, len(targets))
        ) as pool:
            futures = [(target, pool.submit(capture, target)) for target in targets]
            snapshots = [(target, future.result()) for target, future in futures]
    return {
        "clusters": {target.cluster_id: snapshot for target, snapshot in snapshots},
        "expected_rules": release._capture_dataplane_expected_rules(),
    }


def capture_observability_snapshot_with_dataplane(release: Any) -> dict[str, Any]:
    """The observability snapshot: control-plane blobs plus the data-plane half.

    Bound as ``RegionalRelease._capture_observability_snapshot`` so the
    previous-state capture picks up both under the one ``observability`` key it
    already stores; a state written by an older engine simply lacks
    ``dataplane_adot`` and takes the previous-image fallback on rollback.
    """

    return {
        **capture_observability_snapshot(release),
        "dataplane_adot": capture_dataplane_adot_state(release),
    }


# --- rollback ----------------------------------------------------------------------


class DataplaneRollbackIncomplete(ReleaseError):
    """The data-plane observability phase finished with at least one failure.

    ``phase_details`` is the phase's record as far as it got (per-cluster
    outcome, expected-rules outcome); the phase runner folds it into the
    durable timing entry beside the error, so the state says which clusters
    were restored and which were not.
    """

    def __init__(self, message: str, *, phase_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.phase_details = phase_details


def phase_failure_details(error: Exception) -> dict[str, Any]:
    """What a rollback phase runner records for a phase that raised ``error``.

    Always the error itself; plus, for an error that knows how far its phase
    got (``phase_details``, as :class:`DataplaneRollbackIncomplete` carries),
    that record -- so the durable state says which clusters were restored and
    which were not, not just that the phase failed.
    """

    details = getattr(error, "phase_details", None)
    return {
        **(dict(details) if isinstance(details, dict) else {}),
        "error": f"{type(error).__name__}: {error}",
    }


@dataclass(frozen=True)
class _DataplanePlan:
    """The validated data-plane half of a previous-state snapshot.

    ``path`` is ``snapshot`` (per-cluster objects in ``parsed``, the rules blob
    in ``expected_rules``, every configured target in ``targets``) or
    ``previous-image`` (``image`` set; a state captured before the snapshot).
    """

    path: str
    image: str = ""
    targets: dict[str, ClusterTarget] = field(default_factory=dict)
    parsed: dict[str, tuple[str, list[Any], list[Any]]] = field(default_factory=dict)
    expected_rules: object = None


def validate_dataplane_adot_snapshot(
    release: Any, previous: dict[str, Any]
) -> _DataplanePlan:
    """Validate the data-plane half of the snapshot without touching anything.

    Run by BOTH observability rollback phases: the control-plane phase runs it
    first so a snapshot that cannot be put back whole is refused before any
    half of it mutates a cluster or an AMP definition, and the data-plane
    phase runs it again on its own (idempotent) resume.
    """

    observability = previous.get("observability")
    if not isinstance(observability, dict):
        raise ReleaseError("previous observability snapshot is unavailable")
    if "dataplane_adot" not in observability:
        image = str(previous.get("adot_image") or "")
        if not image:
            raise ReleaseError("previous ADOT image is unavailable for the collectors")
        return _DataplanePlan(path="previous-image", image=image)
    state = observability["dataplane_adot"]
    if not isinstance(state, dict) or not isinstance(state.get("clusters"), dict):
        raise ReleaseError("previous data-plane ADOT collector snapshot is invalid")
    targets = {target.cluster_id: target for target in release.config.clusters}
    unknown = sorted(set(state["clusters"]) - set(targets))
    if unknown:
        raise ReleaseError(
            "previous data-plane ADOT collector snapshot names clusters missing "
            "from the release config: " + ", ".join(unknown)
        )
    parsed = {
        cluster_id: snapshot_parts(
            snapshot,
            label=f"previous data-plane ADOT collector snapshot for {cluster_id}",
        )
        for cluster_id, snapshot in sorted(state["clusters"].items())
    }
    expected_rules = state.get("expected_rules")
    _expected_rules_parts(expected_rules)
    return _DataplanePlan(
        path="snapshot",
        targets=targets,
        parsed=parsed,
        expected_rules=expected_rules,
    )


def _rollback_from_previous_image(
    release: Any, previous: dict[str, Any], plan: _DataplanePlan
) -> dict[str, Any]:
    _say(
        "previous observability snapshot predates the per-cluster collector "
        "capture; every configured cluster's collector is re-rendered from the "
        "CANDIDATE manifest with the previous ADOT image (template edits are "
        "not undone on the data plane)"
    )
    rollback_gpu_adot_collectors(release, previous)
    return {
        "path": "previous-image",
        "image": plan.image,
        "clusters": {
            target.cluster_id: "candidate-manifest-with-previous-image"
            for target in release.config.clusters
        },
    }


def _restore_cluster_collector(
    release: Any, target: ClusterTarget, parts: tuple[str, list[Any], list[Any]]
) -> str:
    """One cluster's collector back from its own snapshot; returns the outcome.

    ``restored`` (objects applied, the collector restarted and waited for),
    ``restored-unchanged`` (every object applied back as ``unchanged``, so the
    running Pod already serves the previous configuration and is left alone,
    as the installer leaves the control-plane collector alone on the same
    evidence) or ``removed`` (the snapshot had no objects: the candidate's were
    deleted).
    """

    namespace, objects, absent = parts
    kubectl = release._gpu(target)
    changed = apply_snapshot_objects(release, objects, kubectl=kubectl)
    delete_absent_objects(release, namespace, absent, kubectl=kubectl)
    if not objects:
        return "removed"
    if not changed:
        return "restored-unchanged"
    restart_snapshot_deployments(
        release,
        namespace,
        objects,
        timeout=DATAPLANE_ADOT_ROLLOUT_TIMEOUT,
        kubectl=kubectl,
    )
    return "restored"


def _scale_unseen_collector_down(release: Any, target: ClusterTarget) -> str:
    release._scale_if_present(release._gpu(target), DATAPLANE_ADOT_DEPLOYMENT, 0)
    return "scaled-to-zero"


def restore_dataplane_adot_state(
    release: Any, previous: dict[str, Any]
) -> dict[str, Any]:
    """Put every GPU cluster's collector back the way its snapshot found it.

    Validates the whole snapshot before touching any cluster. Iterates the
    clusters the SNAPSHOT knows about, not the candidate config, and restores
    each from its own objects regardless of how far the candidate got on it:
    re-applying a cluster's own snapshot is idempotent, and reading progress
    to skip it would tie the compensation to the progress record and leave a
    partially applied cluster unrestored; the restart is skipped when the
    apply changed nothing. A configured cluster the snapshot never saw joined
    site.yaml after the transaction opened (the capture covers the whole
    candidate config), so it is scaled to zero.

    One cluster failing -- an unreachable API, a ``rollout status`` that times
    out -- does not stop the others or the expected-rules restore; each failure
    is recorded per cluster and the phase raises once at the end, naming them,
    so the rollback is FAILED and a resume re-runs this phase.
    """

    plan = validate_dataplane_adot_snapshot(release, previous)
    if plan.path == "previous-image":
        return _rollback_from_previous_image(release, previous, plan)
    record: dict[str, str] = {}
    failed: list[str] = []
    steps: list[tuple[str, Callable[[], str]]] = [
        (
            cluster_id,
            functools.partial(
                _restore_cluster_collector, release, plan.targets[cluster_id], parts
            ),
        )
        for cluster_id, parts in plan.parsed.items()
    ]
    steps.extend(
        (
            cluster_id,
            functools.partial(
                _scale_unseen_collector_down, release, plan.targets[cluster_id]
            ),
        )
        for cluster_id in sorted(set(plan.targets) - set(plan.parsed))
    )
    for cluster_id, step in steps:
        try:
            record[cluster_id] = step()
        except Exception as exc:  # noqa: BLE001 -- one cluster must not stop the rest
            record[cluster_id] = f"failed: {type(exc).__name__}: {exc}"
            failed.append(cluster_id)
        _say(f"{cluster_id}: data-plane ADOT collector {record[cluster_id]}")
    try:
        outcome = restore_dataplane_expected_rules(release, plan.expected_rules)
    except Exception as exc:  # noqa: BLE001 -- recorded beside the clusters
        outcome = f"failed: {type(exc).__name__}: {exc}"
        failed.append("expected-rules")
    details = {"path": "snapshot", "clusters": record, "expected_rules": outcome}
    if failed:
        reasons = "; ".join(
            f"{name} ({record[name] if name in record else outcome})" for name in failed
        )
        others = ", ".join(
            f"{name} {state}" for name, state in record.items() if name not in failed
        )
        raise DataplaneRollbackIncomplete(
            f"data-plane observability rollback failed on {len(failed)} of "
            f"{len(steps) + 1} steps: {reasons}"
            + (f" -- the rest was restored ({others})" if others else "")
            + "; resume the rollback to retry the failed ones",
            phase_details=details,
        )
    return details


def restore_control_plane_observability(
    release: Any, previous: dict[str, Any]
) -> dict[str, Any]:
    """The ``observability_restore`` phase: the control-plane half only.

    Validates the WHOLE observability snapshot first -- the data-plane half
    included, although that half is restored later in its own phase -- so a
    snapshot that cannot be put back whole is refused before any AMP definition
    or collector object is touched. Returns the record the phase runner
    persists.
    """

    plan = validate_dataplane_adot_snapshot(release, previous)
    restarted = release._restore_observability_snapshot(previous.get("observability"))
    record: dict[str, Any] = {
        "dataplane_adot": f"deferred to {DATAPLANE_OBSERVABILITY_PHASE} ({plan.path})"
    }
    if isinstance(restarted, bool):
        record["control_plane_collector"] = (
            "restored" if restarted else "restored-unchanged"
        )
    return record


def restore_dataplane_observability_phase(
    release: Any,
    previous: dict[str, Any],
    compensation: Any,
    run_phase: Callable[[str, str, str, Callable[[], object]], None],
) -> None:
    """Run the data-plane observability rollback phase through ``run_phase``.

    Placed by ``rollback_release`` after the CPU restore: the previous control
    plane comes back before any GPU cluster's collector is touched, so an
    unreachable GPU cluster cannot hold the control plane hostage. Gated like
    the control-plane half, on the compensation plan's OBSERVABILITY component.
    """

    if not compensation.restores_observability:
        return
    run_phase(
        DATAPLANE_OBSERVABILITY_PHASE,
        DATAPLANE_OBSERVABILITY_RESTORING,
        DATAPLANE_OBSERVABILITY_RESTORED,
        lambda: restore_dataplane_adot_state(release, previous),
    )
