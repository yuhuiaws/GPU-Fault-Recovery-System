"""M-6: the RDS CA bundle is shipped, mounted and pointed at everywhere the
control-plane Postgres DSN is opened.

``gpu_fault.admin.bootstrap._aurora_dsn`` refuses to build the Aurora DSN
unless ``GPU_FAULT_RDS_CA_BUNDLE`` names the in-Pod path of an RDS CA bundle;
it then bakes ``sslmode=verify-full&sslrootcert=<that path>`` into the
``gpu-fault-aurora`` Secret. These tests pin the deploy-side wiring that makes
that path real:

* every Pod that consumes ``gpu-fault-aurora``'s ``postgres-url`` -- the
  control-plane Deployment, the Aurora migration Job and the credential
  refresh CronJob -- mounts the bundle read-only at the one canonical path and
  sets ``GPU_FAULT_RDS_CA_BUNDLE`` to it;
* the admin-CLI deploy environment exports the same path, so bootstrap bakes a
  path the Pods actually serve;
* the bundle is fetched from the official AWS truststore and admitted only if
  its SHA-256 matches the pinned value (fail closed), the same digest-gate
  convention the wheel and node-bundle ship with.

Nothing here talks to kubectl or the network: the manifests are parsed as
YAML and the fetch script is checked with ``bash -n`` and text assertions.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"

# One canonical in-Pod path, shared by every consumer and by the DSN bootstrap
# bakes. The bundle ConfigMap is mounted at the parent directory so the file
# lands here without a subPath.
CANONICAL_PATH = "/etc/gpu-fault/rds/ca-bundle.pem"
MOUNT_DIR = "/etc/gpu-fault/rds"
CONFIGMAP_NAME = "gpu-fault-rds-ca-bundle"
VOLUME_NAME = "rds-ca-bundle"
ENV_NAME = "GPU_FAULT_RDS_CA_BUNDLE"

CONTROL_PLANE = DEPLOY / "control-plane" / "base" / "control-plane-deployment.yaml"
CREDENTIAL_REFRESH = (
    DEPLOY / "control-plane" / "regional" / "aurora-credential-refresh.yaml"
)
FETCH_SCRIPT = DEPLOY / "control-plane" / "tools" / "apply-rds-ca-bundle.sh"
REGIONAL_ENV = DEPLOY / "control-plane" / "regional" / "regional-env.example.sh"
GENERATED = DEPLOY / "control-plane" / "regional" / "generated"

# Every Pod that opens the Aurora DSN. The credential-refresh CronJob reads the
# Secret through the kube API and opens the DSN with verify_dsn rather than
# taking postgres-url as an env, so it is listed explicitly. The rest consume
# gpu-fault-aurora/postgres-url directly -- and since bootstrap bakes
# sslmode=verify-full&sslrootcert=<canonical path> into that Secret, every one
# of them fails to connect unless the bundle is mounted at that path.
CONSUMER_MANIFESTS = (
    CONTROL_PLANE,
    CREDENTIAL_REFRESH,
    DEPLOY / "migrations" / "postgres-to-aurora-migration.yaml",
    DEPLOY / "migrations" / "postgres-schema-ensure-job.yaml",
    DEPLOY / "migrations" / "postgres-schema-preflight-job.yaml",
    DEPLOY / "migrations" / "postgres-index-build-job.yaml",
    DEPLOY / "migrations" / "postgres-counter-shards-finalize-job.yaml",
    DEPLOY / "migrations" / "postgres-counter-shards-rollback-job.yaml",
)


def _docs(path: Path) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _pod_specs(path: Path) -> list[dict[str, Any]]:
    """Every Pod template spec across the workload kinds in the manifest."""
    specs: list[dict[str, Any]] = []
    for doc in _docs(path):
        kind = doc.get("kind")
        if kind in ("Deployment", "Job"):
            specs.append(doc["spec"]["template"]["spec"])
        elif kind == "CronJob":
            specs.append(doc["spec"]["jobTemplate"]["spec"]["template"]["spec"])
    return specs


def test_every_consumer_manifest_has_a_pod_spec() -> None:
    # Guards the tests below: if a manifest stopped shipping a Pod the
    # assertions would pass vacuously.
    for manifest in CONSUMER_MANIFESTS:
        assert _pod_specs(manifest), f"{manifest.name} has no Pod spec"


def _volume(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    for volume in spec.get("volumes", []):
        if volume.get("name") == name:
            return volume
    return None


def test_bundle_is_mounted_read_only_at_the_canonical_path_in_every_consumer() -> None:
    for manifest in CONSUMER_MANIFESTS:
        for spec in _pod_specs(manifest):
            volume = _volume(spec, VOLUME_NAME)
            assert volume is not None, f"{manifest.name}: no {VOLUME_NAME} volume"
            config_map = volume.get("configMap") or {}
            assert config_map.get("name") == CONFIGMAP_NAME, (
                f"{manifest.name}: {VOLUME_NAME} must source ConfigMap {CONFIGMAP_NAME}"
            )
            # Fail closed: a missing bundle must wedge the Pod, never run it
            # with an unverified connection, so the volume is not optional.
            assert config_map.get("optional") is not True, (
                f"{manifest.name}: the CA bundle volume must not be optional"
            )

            container = spec["containers"][0]
            mounts = {m["name"]: m for m in container.get("volumeMounts", [])}
            assert VOLUME_NAME in mounts, (
                f"{manifest.name}: container does not mount {VOLUME_NAME}"
            )
            mount = mounts[VOLUME_NAME]
            assert mount.get("mountPath") == MOUNT_DIR, (
                f"{manifest.name}: {VOLUME_NAME} must mount at {MOUNT_DIR} so "
                f"the file lands at {CANONICAL_PATH}"
            )
            assert mount.get("readOnly") is True, (
                f"{manifest.name}: the CA bundle mount must be read-only"
            )


def test_env_points_at_the_canonical_path_in_every_consumer() -> None:
    for manifest in CONSUMER_MANIFESTS:
        for spec in _pod_specs(manifest):
            container = spec["containers"][0]
            env = {item["name"]: item for item in container.get("env", [])}
            assert ENV_NAME in env, f"{manifest.name}: {ENV_NAME} is not set"
            assert env[ENV_NAME].get("value") == CANONICAL_PATH, (
                f"{manifest.name}: {ENV_NAME} must equal {CANONICAL_PATH}"
            )


def test_admin_cli_deploy_environment_exports_the_same_canonical_path() -> None:
    """bootstrap bakes GPU_FAULT_RDS_CA_BUNDLE into the DSN, so the admin-CLI
    deploy environment must export the same path the Pods mount at."""

    text = REGIONAL_ENV.read_text(encoding="utf-8")
    match = re.search(rf"""export\s+{ENV_NAME}=['"]([^'"]+)['"]""", text)
    assert match, f"{REGIONAL_ENV.name} does not export {ENV_NAME}"
    assert match.group(1) == CANONICAL_PATH, (
        f"{REGIONAL_ENV.name} must export {ENV_NAME}={CANONICAL_PATH}"
    )


# --- the fetch script: official source, pinned digest, fail closed ---------


def test_fetch_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(FETCH_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_fetch_script_pulls_the_official_bundle_and_pins_its_digest() -> None:
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    assert "truststore.pki.rds.amazonaws.com/global/global-bundle.pem" in text, (
        "the bundle must come from the official AWS truststore"
    )
    pin = re.search(r"RDS_CA_BUNDLE_SHA256=\"([0-9a-f]{64})\"", text)
    assert pin, "the script must pin a lowercase SHA-256 for the bundle"
    # The digest gate must run against the pin and fail closed on mismatch.
    assert "sha256sum -c" in text, "the pin must be enforced with sha256sum -c"
    assert CONFIGMAP_NAME in text, (
        f"the script must create the {CONFIGMAP_NAME} ConfigMap"
    )


def test_fetch_script_digest_gate_rejects_a_tampered_bundle(tmp_path: Path) -> None:
    """A bundle whose bytes do not match the pin must abort with a non-zero
    exit, so a swapped CA never reaches the cluster."""

    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    pin = re.search(r'RDS_CA_BUNDLE_SHA256="([0-9a-f]{64})"', text).group(1)
    tampered = tmp_path / "bundle.pem"
    tampered.write_text("-----BEGIN CERTIFICATE-----\nnot the real bundle\n")

    ok = subprocess.run(
        [
            "bash",
            "-c",
            'printf "%s  %s\\n" "$1" "$2" | sha256sum -c - >/dev/null',
            "_",
            pin,
            str(tampered),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode != 0, "the digest gate must reject a tampered bundle"


# --- the rendered role-split manifests carry the same wiring ---------------


def test_rendered_role_deployments_mount_the_bundle() -> None:
    """The regional deploy applies regional/generated/, not the base. Re-render
    after editing the base or these will drift."""

    for name in (
        "gpu-fault-api-ha-ingress.yaml",
        "gpu-fault-control-worker.yaml",
        "gpu-fault-telemetry-spool-worker.yaml",
    ):
        manifest = GENERATED / name
        specs = [spec for spec in _pod_specs(manifest)]
        assert specs, f"{name}: no Pod spec"
        spec = specs[0]
        volume = _volume(spec, VOLUME_NAME)
        assert volume is not None, f"{name}: no {VOLUME_NAME} volume rendered"
        assert (volume.get("configMap") or {}).get("name") == CONFIGMAP_NAME
        container = spec["containers"][0]
        mounts = {m["name"]: m for m in container.get("volumeMounts", [])}
        assert mounts.get(VOLUME_NAME, {}).get("mountPath") == MOUNT_DIR


def _consumes_aurora_postgres_url(spec: dict[str, Any]) -> bool:
    for container in spec.get("containers", []):
        for item in container.get("env", []):
            ref = (item.get("valueFrom") or {}).get("secretKeyRef") or {}
            if ref.get("name") == "gpu-fault-aurora" and (
                ref.get("key") == "postgres-url"
            ):
                return True
    return False


def test_no_aurora_dsn_consumer_is_left_unwired() -> None:
    """Discovery guard: every deploy manifest whose Pod takes
    gpu-fault-aurora/postgres-url as an env must be in CONSUMER_MANIFESTS, so a
    new consumer cannot silently ship without the CA bundle and break its
    verify-full connection. The rendered role-split copies are excluded (they
    derive from the base) and legacy trees are out of scope here."""

    covered = {p.resolve() for p in CONSUMER_MANIFESTS}
    missing = []
    for manifest in DEPLOY.rglob("*.yaml"):
        rel = manifest.relative_to(DEPLOY).parts
        if rel[0] in ("hyperpod", "node") or "generated" in rel:
            continue
        try:
            specs = _pod_specs(manifest)
        except (KeyError, TypeError, yaml.YAMLError):
            # Non-workload documents (e.g. CloudFormation templates with custom
            # tags) are not Aurora consumers.
            continue
        if any(_consumes_aurora_postgres_url(spec) for spec in specs):
            if manifest.resolve() not in covered:
                missing.append(str(manifest.relative_to(ROOT)))
    assert not missing, (
        "these manifests open the Aurora DSN but are not wired for the CA "
        f"bundle: {missing}"
    )


def test_rendered_core_config_sets_the_env() -> None:
    """The literal env is externalised into the -config-core ConfigMap that the
    role Deployments consume with envFrom."""

    for name in (
        "gpu-fault-api-ha-config-core.yaml",
        "gpu-fault-control-worker-config-core.yaml",
        "gpu-fault-telemetry-spool-worker-config-core.yaml",
    ):
        docs = _docs(GENERATED / name)
        data = docs[0].get("data", {})
        assert data.get(ENV_NAME) == CANONICAL_PATH, (
            f"{name}: {ENV_NAME} must be {CANONICAL_PATH}"
        )
