"""The per-GPU-cluster half of the observability component (F10 fix round 1).

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
   cluster can see that.

2. **Rollback compensation.** The first wiring put every configured cluster's
   collector back by rendering the *candidate's* manifest with the previous
   image: the keep filters, relabels and scrape targets stayed the candidate's,
   collectors were created on clusters the candidate never touched, and a
   cluster whose role the candidate had just added kept its collector. The
   compensation is now a snapshot, taken beside the control-plane one: every
   GPU cluster's declared collector objects are read through that cluster's own
   kube context before the OBSERVABILITY node runs, and rollback restores each
   cluster from ITS snapshot -- objects applied back where a collector was live,
   the candidate's objects deleted where none was, a cluster the snapshot never
   saw scaled to zero -- and puts the expected-rules namespace back too. The
   previous-image path survives only for a state captured before this snapshot
   existed, and the rollback record says which path ran.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
# The cluster id is interpolated into a PromQL string literal and a YAML
# scalar; anything outside this set is refused rather than escaped.
_CLUSTER_ID_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- expected-collector rules -------------------------------------------------------


def expected_collector_targets(release: Any) -> tuple[ClusterTarget, ...]:
    """The clusters the release applies a collector to, in config order."""

    return tuple(
        target
        for target in release.config.clusters
        if dataplane_adot_skip_reason(release, target) is None
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


def render_dataplane_expected_rules(release: Any) -> str | None:
    """The per-cluster absence rules as an AMP rule-groups document, or ``None``.

    ``None`` -- not an empty group -- when no cluster is expected to carry a
    collector: the installer then deletes the namespace, so nothing stale is
    left to fire.
    """

    targets = expected_collector_targets(release)
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


def amp_installer_arguments(release: Any, directory: Path) -> list[str]:
    """The installer's positional arguments for this release's expected set.

    The rendered document is written into ``directory`` (a temporary directory
    the caller owns for the installer's lifetime) and handed over by path; an
    empty expected set is an explicit deletion request. The bootstrap runs the
    installer with no arguments at all, which leaves the namespace alone.
    """

    rendered = render_dataplane_expected_rules(release)
    if rendered is None:
        return ["--no-dataplane-expected-rules"]
    path = directory / "dataplane-expected-rules.yaml"
    path.write_text(rendered, encoding="utf-8")
    return ["--dataplane-expected-rules", str(path)]


def run_amp_monitoring_installer(release: Any, environment: dict[str, str]) -> None:
    """Run the AMP monitoring installer with this release's expected-rules hand-off.

    ``environment`` is the installer's configuration (built by the release from
    its config); the rendered rules file lives only for the installer's run.
    """

    with tempfile.TemporaryDirectory() as directory:
        arguments = amp_installer_arguments(release, Path(directory))
        release.runner.run(
            ["bash", str(AMP_MONITORING_INSTALLER), *arguments],
            env=environment,
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


def capture_dataplane_expected_rules(release: Any) -> dict[str, Any]:
    """Read the expected-rules namespace as it is before the release re-puts it.

    Absence is a legitimate previous state (no cluster had a role); an
    unreadable namespace is not, because a rollback built on a guess would put
    back the wrong rules or delete rules that were there.
    """

    code, stdout, stderr = release.runner.probe_output(
        [*_describe_expected_rules(release), "--output", "json"]
    )
    if code:
        if "ResourceNotFoundException" in stderr:
            return {"present": False, "data_base64": None}
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: {stderr.strip() or code}"
        )
    try:
        data = json.loads(stdout)["ruleGroupsNamespace"]["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ReleaseError(
            "cannot read the expected-collector rule namespace "
            f"{DATAPLANE_EXPECTED_RULE_NAMESPACE}: unexpected describe output"
        ) from exc
    return {"present": True, "data_base64": str(data)}


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
    exists = release.runner.probe(_describe_expected_rules(release))
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


def _rollback_from_previous_image(
    release: Any, previous: dict[str, Any]
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
        "image": str(previous.get("adot_image") or ""),
        "clusters": {
            target.cluster_id: "candidate-manifest-with-previous-image"
            for target in release.config.clusters
        },
    }


def restore_dataplane_adot_state(
    release: Any, previous: dict[str, Any]
) -> dict[str, Any]:
    """Put every GPU cluster's collector back the way its snapshot found it.

    Validates the whole snapshot before touching any cluster. Iterates the
    clusters the SNAPSHOT knows about, not the candidate config, and restores
    each from its own objects regardless of how far the candidate got on it:
    re-applying a cluster's own snapshot is idempotent and costs one collector
    restart, while reading progress to skip it would tie the compensation to
    the progress record and leave a partially applied cluster unrestored. A
    configured cluster the snapshot never saw is one the candidate created,
    so it is scaled to zero.
    """

    observability = previous.get("observability")
    if not isinstance(observability, dict):
        raise ReleaseError("previous observability snapshot is unavailable")
    if "dataplane_adot" not in observability:
        return _rollback_from_previous_image(release, previous)
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
    record: dict[str, str] = {}
    for cluster_id, (namespace, objects, absent) in parsed.items():
        kubectl = release._gpu(targets[cluster_id])
        apply_snapshot_objects(release, objects, kubectl=kubectl)
        delete_absent_objects(release, namespace, absent, kubectl=kubectl)
        restart_snapshot_deployments(
            release,
            namespace,
            objects,
            timeout=DATAPLANE_ADOT_ROLLOUT_TIMEOUT,
            kubectl=kubectl,
        )
        record[cluster_id] = "restored" if objects else "removed"
        _say(
            f"{cluster_id}: data-plane ADOT collector {record[cluster_id]} "
            "from the previous release's snapshot"
        )
    for cluster_id in sorted(set(targets) - set(parsed)):
        release._scale_if_present(
            release._gpu(targets[cluster_id]), DATAPLANE_ADOT_DEPLOYMENT, 0
        )
        record[cluster_id] = "scaled-to-zero"
        _say(
            f"{cluster_id}: data-plane ADOT collector scaled to zero (the cluster "
            "is not in the previous release's snapshot)"
        )
    outcome = restore_dataplane_expected_rules(release, expected_rules)
    return {"path": "snapshot", "clusters": record, "expected_rules": outcome}


def restore_observability_with_dataplane(
    release: Any, previous: dict[str, Any]
) -> dict[str, Any]:
    """The observability restore phase: control-plane snapshot, then the data plane.

    Returns the record the rollback's phase runner persists beside the phase
    timing, so the durable state says which path compensated the collectors.
    """

    release._restore_observability_snapshot(previous.get("observability"))
    return {"dataplane_adot": restore_dataplane_adot_state(release, previous)}
