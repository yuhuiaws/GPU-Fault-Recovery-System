"""Capture and restore the live ADOT collector so rollback can compensate it.

Automatic rollback used to refuse any release whose diff touched the ADOT
manifest or image. The reason was real: the collector is applied straight from
the candidate checkout, so a rollback that restored every other component would
leave the collector running the candidate's configuration while the transaction
reported ``rollback PASSED``. Refusing beat lying -- but the refusal's
granularity was the whole release, so a single metrics-filter edit forced the
operator to give up automatic rollback for the wheels, Agents and schema too.

Capturing the live objects before the observability component mutates them makes
the collector compensable, exactly like the AMP rule and Alertmanager blobs that
the same snapshot already carries. The captured objects are the previous
release's own applied state, so restoring them restores the previous manifest
*and* the previous image without needing the previous checkout.

The object capture itself is shared with the control-plane endpoint, which is
applied from the tree the same way; see ``regional_manifest_snapshot``.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from gpu_fault_release import repository_root
from gpu_fault_release.regional_manifest_snapshot import (
    apply_snapshot_objects,
    capture_declared_objects,
    declared_manifest_objects,
    delete_absent_objects,
    restart_snapshot_deployments,
    snapshot_parts,
)
from gpu_fault_release.regional_release_config import ReleaseError

ROOT = repository_root()
ADOT_MANIFEST = ROOT / "deploy/observability/adot-control-plane.yaml"
ADOT_ROLLOUT_TIMEOUT = "300s"


def declared_adot_objects(manifest_text: str) -> tuple[dict[str, str], ...]:
    """Return the objects the ADOT manifest applies, in manifest order.

    The installer applies this file as-is, so the file is the authority on which
    objects an upgrade may mutate -- and therefore on which objects a rollback
    has to put back.
    """

    return declared_manifest_objects(manifest_text, label="ADOT manifest")


def capture_adot_objects(release: Any) -> dict[str, Any]:
    """Read the live ADOT objects the observability component is about to mutate.

    An object the manifest declares but the cluster does not have yet is one the
    candidate adds; rollback removes it rather than pretending it was there.
    """

    captured = capture_declared_objects(
        release,
        declared_adot_objects(ADOT_MANIFEST.read_text(encoding="utf-8")),
        label="live ADOT object",
    )
    if not any(str(item.get("kind")) == "Deployment" for item in captured["objects"]):
        # The collector Deployment exists on every converged site -- the same
        # capture already fails closed when it cannot read its image. Missing
        # here means this snapshot is reading the wrong namespace, and a
        # rollback built on it would silently compensate nothing.
        raise ReleaseError(
            "live ADOT collector Deployment is missing from namespace "
            f"{captured['namespace']}"
        )
    return captured


def _snapshot_parts(snapshot: object) -> tuple[str, list[Any], list[Any]]:
    if not isinstance(snapshot, dict) or "adot" not in snapshot:
        raise ReleaseError(
            "previous observability snapshot has no ADOT collector objects"
        )
    namespace, objects, absent = snapshot_parts(
        snapshot["adot"],
        label="previous ADOT collector snapshot",
    )
    if not objects:
        raise ReleaseError("previous ADOT collector snapshot is empty")
    return namespace, objects, absent


def restore_adot_objects(release: Any, snapshot: object) -> None:
    """Put the captured collector back and prove it runs again.

    The restart is not cosmetic: the collector reads its pipeline from the
    ConfigMap it mounts, so restoring the previous ConfigMap without replacing
    the running Pod would leave the candidate's configuration serving from
    memory, which is the exact state this compensation exists to end.
    """

    namespace, objects, absent = _snapshot_parts(snapshot)
    apply_snapshot_objects(release, objects)
    delete_absent_objects(release, namespace, absent)
    restart_snapshot_deployments(
        release,
        namespace,
        objects,
        timeout=ADOT_ROLLOUT_TIMEOUT,
    )


def capture_observability_snapshot(release: Any) -> dict[str, Any]:
    common = [
        "--region",
        release.config.aws_region,
        "--workspace-id",
        release.config.health.amp_workspace_id,
    ]
    rules = json.loads(
        release.runner.run(
            [
                "aws",
                "amp",
                "describe-rule-groups-namespace",
                *common,
                "--name",
                release.config.health.amp_rule_namespace,
                "--output",
                "json",
            ],
            capture=True,
        )
    )["ruleGroupsNamespace"]
    alertmanager = json.loads(
        release.runner.run(
            [
                "aws",
                "amp",
                "describe-alert-manager-definition",
                *common,
                "--output",
                "json",
            ],
            capture=True,
        )
    )["alertManagerDefinition"]
    return {
        "rule_namespace": release.config.health.amp_rule_namespace,
        "rules_data_base64": str(rules["data"]),
        "alertmanager_data_base64": str(alertmanager["data"]),
        "adot": capture_adot_objects(release),
    }


def restore_observability_snapshot(
    release: Any,
    snapshot: object,
) -> None:
    if not isinstance(snapshot, dict):
        raise ReleaseError("previous observability snapshot is unavailable")
    try:
        rules = base64.b64decode(
            str(snapshot["rules_data_base64"]),
            validate=True,
        )
        alertmanager = base64.b64decode(
            str(snapshot["alertmanager_data_base64"]),
            validate=True,
        )
        rule_namespace = str(snapshot["rule_namespace"])
    except (KeyError, ValueError) as exc:
        raise ReleaseError("previous observability snapshot is invalid") from exc
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rules_path = root / "rules.yaml"
        alertmanager_path = root / "alertmanager.yaml"
        rules_path.write_bytes(rules)
        alertmanager_path.write_bytes(alertmanager)
        release.runner.run(
            [
                "aws",
                "amp",
                "put-rule-groups-namespace",
                "--region",
                release.config.aws_region,
                "--workspace-id",
                release.config.health.amp_workspace_id,
                "--name",
                rule_namespace,
                "--data",
                f"fileb://{rules_path}",
            ]
        )
        release.runner.run(
            [
                "aws",
                "amp",
                "put-alert-manager-definition",
                "--region",
                release.config.aws_region,
                "--workspace-id",
                release.config.health.amp_workspace_id,
                "--data",
                f"fileb://{alertmanager_path}",
            ]
        )
    restore_adot_objects(release, snapshot)
