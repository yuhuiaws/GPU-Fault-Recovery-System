"""Read-only convergence proof used by release planning and final validation."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import repository_root
from gpu_fault_release.regional_dataplane_observability import (
    DATAPLANE_EXPECTED_RULE_NAMESPACE,
    render_dataplane_expected_rules,
)
from gpu_fault_release.regional_observability_rollback import (
    AMP_FAILED_STATUSES,
    AMP_SETTLING_STATUSES,
    AmpDefinition,
    amp_definition_status,
    describe_amp_definition,
)
from gpu_fault_release.regional_release_config import ReleaseError

ROOT = repository_root()

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease


def _definition_matches(
    release: RegionalRelease, *, name: str | None, expected: str | None
) -> bool:
    workspace = release.config.health.amp_workspace_id
    if not workspace:
        raise ReleaseError("monitoring workspace identity is unavailable")
    arguments = [
        "aws",
        "amp",
        "describe-rule-groups-namespace"
        if name
        else "describe-alert-manager-definition",
        "--region",
        release.config.aws_region,
        "--workspace-id",
        workspace,
    ]
    if name:
        arguments.extend(["--name", name])
    document = describe_amp_definition(
        release,
        AmpDefinition(
            describe=tuple(arguments),
            root="ruleGroupsNamespace" if name else "alertManagerDefinition",
            label=name or "AMP Alertmanager",
        ),
    )
    if document is None:
        return expected is None
    status = amp_definition_status(document)
    if status not in {"ACTIVE", *AMP_FAILED_STATUSES, *AMP_SETTLING_STATUSES}:
        raise ReleaseError("cannot validate an unknown monitoring definition status")
    if expected is None or status != "ACTIVE":
        return False
    try:
        actual = base64.b64decode(document["data"], validate=True).decode("utf-8")
        return bool(yaml.safe_load(actual) == yaml.safe_load(expected))
    except (KeyError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ReleaseError("cannot validate the live monitoring definition") from exc


def observability_drift(release: RegionalRelease) -> bool:
    if release.runner.dry_run or not release.config.health.amp_workspace_id:
        return False
    topic = release.config.health.sns_topic_arn
    if not topic:
        raise ReleaseError("monitoring notification topic identity is unavailable")
    manifest = (ROOT / "deploy/observability/adot-control-plane.yaml").read_text()
    for marker, value in {
        "REPLACE_WITH_AMP_WORKSPACE_ID": release.config.health.amp_workspace_id,
        "REPLACE_WITH_AWS_REGION": release.config.aws_region,
        "REPLACE_WITH_ADOT_IMAGE": release.adot_image,
        "gpu-fault-system": release.config.namespace,
    }.items():
        manifest = manifest.replace(marker, value)
    # kubectl diff applies server defaults and field ownership without persisting
    # a mutation. Its output may contain live configuration; never print it.
    with tempfile.TemporaryDirectory(prefix="gpu-fault-monitoring-probe-") as directory:
        path = Path(directory) / "adot.yaml"
        path.write_text(manifest)
        path.chmod(0o600)
        code, _output, _error = release.runner.probe_output(
            release._cpu("diff", "-f", str(path)), timeout_seconds=60
        )
    if code not in {0, 1}:
        raise ReleaseError("cannot read ADOT drift; monitoring state is unknown")
    drift = code == 1
    deployment = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            "gpu-fault-adot",
            "--ignore-not-found",
        )
    )
    drift |= int((deployment.get("status") or {}).get("availableReplicas") or 0) < 1
    rules = (ROOT / "deploy/observability/amp-rules.yaml").read_text()
    manager = (ROOT / "deploy/observability/amp-alertmanager.yaml").read_text()
    manager = manager.replace("REPLACE_WITH_SNS_TOPIC_ARN", topic)
    manager = manager.replace("REPLACE_WITH_AWS_REGION", release.config.aws_region)
    for name, expected in (
        (release.config.health.amp_rule_namespace, rules),
        (None, manager),
        (DATAPLANE_EXPECTED_RULE_NAMESPACE, render_dataplane_expected_rules(release)),
    ):
        drift |= not _definition_matches(release, name=name, expected=expected)
    return drift
