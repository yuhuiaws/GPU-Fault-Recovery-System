from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from scripts.component_wheels import (
        APPLICATION_COMPONENT_NAMES,
        component_source_digest,
    )
    from scripts.deploy_host_identity import (
        build_identity as deploy_host_bundle_identity,
    )
    from scripts.release_identity import build_release_identity, file_set_identity
else:
    from component_wheels import APPLICATION_COMPONENT_NAMES, component_source_digest
    from deploy_host_identity import build_identity as deploy_host_bundle_identity
    from release_identity import build_release_identity, file_set_identity


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_HOST_ORCHESTRATION_INPUTS = (
    "src/gpu_fault_release/*.py",
    # The engine runs from the site's source snapshot: its shell entry points,
    # the in-Pod probe programs it ships and the control-plane tools it calls
    # are deploy-host orchestration too. Live 2026-09-13: a probe-only fix was
    # classified QUALITY_ONLY and the site kept running the old probe.
    "deploy/control-plane/regional/*.sh",
    "deploy/control-plane/regional/probes/*.py",
    "deploy/control-plane/tools/*.py",
    "deploy/control-plane/tools/*.sh",
    "scripts/ci_gate.py",
    "scripts/ci_gate_artifacts.py",
    "scripts/ci_candidate_receipt.py",
    "scripts/deploy_source_identity.py",
    "scripts/release_deploy.py",
    "scripts/release_deploy_evidence.py",
    "scripts/release_failure_recovery.py",
    "scripts/release_live_state.py",
    "scripts/run_release_gates.py",
    "scripts/run_static_gates.py",
    "scripts/resolve_ci_run.py",
    "scripts/setup-deploy-host.sh",
    "scripts/setup_deploy_host.py",
    "scripts/staging_deploy.py",
    "scripts/staging_gate_caches.py",
    "scripts/staging_live_evidence.py",
    "scripts/staging_state_hygiene.py",
)
APPLICATION_IDENTITY_PROTOCOL_INPUTS = (
    "config/release-identity.yaml",
    "scripts/build-release-artifacts.py",
    "scripts/component_artifacts.py",
    "scripts/component_wheels.py",
    "scripts/deploy_source_identity.py",
    "scripts/release_identity.py",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def application_identity(root: Path = ROOT) -> dict[str, Any]:
    release = build_release_identity(root)
    value: dict[str, Any] = {
        "schema_version": 1,
        "component_module_digests": {
            name: component_source_digest(name) for name in APPLICATION_COMPONENT_NAMES
        },
        "release_source_identity_sha256": release["sha256"],
        "delivery_components": {
            name: item["sha256"] for name, item in release["components"].items()
        },
        "manifest_inputs_sha256": release["manifest_inputs"]["sha256"],
        "node_template_inputs_sha256": release["node_template_inputs"]["sha256"],
        "rendered_manifests_sha256": release["rendered_manifests"]["sha256"],
        "renderer_inputs_sha256": release["renderer_inputs"]["sha256"],
        "runtime_image_inputs_sha256": release["runtime_image_inputs"]["sha256"],
        "identity_protocol": file_set_identity(
            root,
            APPLICATION_IDENTITY_PROTOCOL_INPUTS,
        ),
    }
    value["sha256"] = canonical_sha256(value)
    return value


def deploy_host_identity(root: Path = ROOT) -> dict[str, Any]:
    bundle = deploy_host_bundle_identity(root)
    orchestration = file_set_identity(root, DEPLOY_HOST_ORCHESTRATION_INPUTS)
    value: dict[str, Any] = {
        "schema_version": 1,
        "bundle": bundle,
        "orchestration": orchestration,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def build_identity(root: Path = ROOT) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "application": application_identity(root),
        "deploy_host": deploy_host_identity(root),
    }
    value["sha256"] = canonical_sha256(value)
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    options = parser.parse_args(arguments)
    try:
        value = build_identity(options.root.expanduser().resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"deploy-source-identity: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
