"""M-6: the RDS CA bundle is shipped, mounted and pointed at everywhere the
control-plane Postgres DSN is opened.

Managed bootstrap explicitly supplies the canonical CA path; the legacy DSN
helper refuses absent CA input. Both bake
``sslmode=verify-full&sslrootcert=<that path>`` into the
``gpu-fault-aurora`` Secret. These tests pin the deploy-side wiring that makes
that path real:

* every Pod that consumes ``gpu-fault-aurora``'s ``postgres-url`` -- the
  control-plane Deployment, the Aurora migration Job and the credential
  refresh CronJob -- mounts the bundle read-only at the one canonical path and
  sets ``GPU_FAULT_RDS_CA_BUNDLE`` to it;
* the legacy regional environment exports the same path; the managed bootstrap
  behavior without that export is covered by the bootstrap preamble tests;
* the bundle installed is the copy vendored in the release, admitted only if
  its SHA-256 matches the pinned value (fail closed), the same digest-gate
  convention the wheel and node-bundle ship with. The deploy and ``config``
  paths never install a download: AWS republishes the bundle whenever an RDS
  region launches, and while the script installed the download the 2026-09-30
  and 2026-10-14 rotations failed every deploy and every ``config`` until a
  release shipped a new pin. The published bundle is compared with the pin
  after the install, for a warning only.

Nothing here talks to a cluster or the network: the manifests are parsed as
YAML, and the install script runs against a fake ``kubectl`` and ``curl``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin.rds_ca_bundle import (
    APPLY_SCRIPT,
    PUBLISHED_BUNDLE_URL,
    RDS_CA_BUNDLE_PATH,
    VENDORED_BUNDLE,
)
from scripts import deploy_source_identity
from scripts.release_identity import file_set_identity

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"

# One canonical in-Pod path, shared by every consumer and by the DSN bootstrap
# bakes. The bundle ConfigMap is mounted at the parent directory so the file
# lands here without a subPath.
CANONICAL_PATH = RDS_CA_BUNDLE_PATH
MOUNT_DIR = "/etc/gpu-fault/rds"
CONFIGMAP_NAME = "gpu-fault-rds-ca-bundle"
VOLUME_NAME = "rds-ca-bundle"
ENV_NAME = "GPU_FAULT_RDS_CA_BUNDLE"

CONTROL_PLANE = DEPLOY / "control-plane" / "base" / "control-plane-deployment.yaml"
CREDENTIAL_REFRESH = (
    DEPLOY / "control-plane" / "regional" / "aurora-credential-refresh.yaml"
)
FETCH_SCRIPT = ROOT / APPLY_SCRIPT
VENDORED = ROOT / VENDORED_BUNDLE
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


# --- the install script: vendored copy, pinned digest, fail closed ---------

# The Kubernetes API server refuses an object whose annotations exceed 256 KiB,
# and `kubectl apply` keeps the whole ConfigMap in last-applied-configuration.
ANNOTATIONS_LIMIT_BYTES = 262144
FAKE_KUBECTL = """#!/usr/bin/env bash
printf 'kubectl %s\\n' "$*" >> "${FAKE_LOG}"
case " $* " in
    *" create configmap "*)
        for argument in "$@"; do
            case "${argument}" in --from-file=*) cat "${argument#--from-file=*=}" ;; esac
        done
        ;;
    *" apply -f - "*) cat > "${FAKE_APPLIED}" ;;
esac
"""
FAKE_CURL = """#!/usr/bin/env bash
printf 'curl %s\\n' "$*" >> "${FAKE_LOG}"
if [[ -z "${FAKE_PUBLISHED:-}" ]]; then
    echo "curl: (6) Could not resolve host: truststore.pki.rds.amazonaws.com" >&2
    exit 6
fi
while (($#)); do
    [[ "$1" == -o ]] && cp "${FAKE_PUBLISHED}" "$2"
    shift
done
exit 0
"""


def _pin() -> str:
    match = re.search(
        r'^RDS_CA_BUNDLE_SHA256="([0-9a-f]{64})"$',
        FETCH_SCRIPT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "the script must pin a lowercase SHA-256 for the bundle"
    return match.group(1)


def _pem_blocks(data: bytes) -> list[bytes]:
    return re.findall(
        rb"-----BEGIN CERTIFICATE-----\n.*?-----END CERTIFICATE-----\n", data, re.S
    )


def _install(
    tmp_path: Path,
    *,
    script: Path = FETCH_SCRIPT,
    published: bytes | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], bytes | None, list[str]]:
    """Run the install script against a fake kubectl and curl.

    ``published`` is what the fake curl serves; ``None`` makes the download
    fail. Returns the run, the bytes `kubectl apply` received and the ordered
    kubectl/curl invocations.
    """

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    for name, body in (("kubectl", FAKE_KUBECTL), ("curl", FAKE_CURL)):
        (fake_bin / name).write_text(body, encoding="utf-8")
        (fake_bin / name).chmod(0o755)
    log = tmp_path / "invocations.log"
    applied = tmp_path / "applied.yaml"
    for stale in (log, applied):
        stale.unlink(missing_ok=True)
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "FAKE_LOG": str(log),
        "FAKE_APPLIED": str(applied),
        # A real kubectl could not reach a cluster even if the fake were skipped.
        "KUBECONFIG": str(tmp_path / "no-such-kubeconfig"),
        **(environment or {}),
    }
    if published is not None:
        served = tmp_path / "served.pem"
        served.write_bytes(published)
        env["FAKE_PUBLISHED"] = str(served)
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, applied.read_bytes() if applied.exists() else None, calls


def test_fetch_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(FETCH_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_vendored_bundle_is_the_pinned_bundle() -> None:
    data = VENDORED.read_bytes()

    assert hashlib.sha256(data).hexdigest() == _pin(), (
        f"{VENDORED_BUNDLE} must be byte-identical to the bundle the script pins"
    )
    assert len(_pem_blocks(data)) == data.count(b"-----BEGIN CERTIFICATE-----") > 0
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    assert "sha256sum -c" in text, "the pin must be enforced with sha256sum -c"
    assert CONFIGMAP_NAME in text, f"the script must create {CONFIGMAP_NAME}"
    assert f':-{PUBLISHED_BUNDLE_URL}}}"' in text, (
        "the drift warning compares the official AWS truststore bundle"
    )
    assert '"${SCRIPT_DIR}/rds-global-bundle.pem"' in text


def test_the_vendored_bundle_is_installed_and_the_download_never_is(
    tmp_path: Path,
) -> None:
    """The deploy/config path trusts the release, not the network: even when
    the published bundle differs, the bytes applied are the vendored copy,
    and the download happens only after the apply, for the warning."""

    vendored = VENDORED.read_bytes()
    published = b"".join(_pem_blocks(vendored)[:-1])

    result, applied, calls = _install(tmp_path, published=published)

    assert result.returncode == 0, result.stderr
    assert applied == vendored, "the ConfigMap must carry the vendored copy"
    assert [call.split()[0] for call in calls] == ["kubectl", "kubectl", "curl"], calls
    assert "--proto =https" in calls[2] and PUBLISHED_BUNDLE_URL in calls[2]
    assert "WARNING: the published AWS RDS CA bundle changed" in result.stderr
    assert hashlib.sha256(published).hexdigest() in result.stderr
    assert f"{_pin()} ({len(_pem_blocks(vendored))} certificates)" in result.stderr
    assert "gpu-fault-admin status" in result.stderr


def test_an_unchanged_published_bundle_prints_no_warning(tmp_path: Path) -> None:
    result, applied, _calls = _install(tmp_path, published=VENDORED.read_bytes())

    assert result.returncode == 0, result.stderr
    assert applied == VENDORED.read_bytes()
    assert "WARNING" not in result.stderr and "note:" not in result.stderr


def test_a_failed_download_never_fails_the_install(tmp_path: Path) -> None:
    result, applied, calls = _install(tmp_path, published=None)

    assert result.returncode == 0, result.stderr
    assert applied == VENDORED.read_bytes()
    assert calls[-1].startswith("curl "), calls
    assert "note: could not download" in result.stderr
    assert "Could not resolve host" in result.stderr


def _tampered_copy(tmp_path: Path) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    script = tools / FETCH_SCRIPT.name
    shutil.copy2(FETCH_SCRIPT, script)
    blocks = _pem_blocks(VENDORED.read_bytes())
    # Every byte a real CA, one of them swapped out: still a valid-looking
    # bundle, not the pinned one.
    (tools / VENDORED.name).write_bytes(b"".join([*blocks[1:], blocks[0]]))
    return script


def test_a_tampered_vendored_copy_fails_closed(tmp_path: Path) -> None:
    result, applied, calls = _install(
        tmp_path, script=_tampered_copy(tmp_path), published=VENDORED.read_bytes()
    )

    assert result.returncode == 1, result.stderr
    assert "does not match the pinned SHA-256; refusing" in result.stderr
    assert applied is None and calls == [], (
        "neither kubectl nor the download runs once the digest gate refuses"
    )


@pytest.mark.parametrize("identical", [True, False], ids=["identical", "other"])
def test_a_source_override_passes_the_same_digest_gate(
    tmp_path: Path, identical: bool
) -> None:
    override = tmp_path / "override.pem"
    blocks = _pem_blocks(VENDORED.read_bytes())
    override.write_bytes(b"".join(blocks if identical else blocks[:-1]))

    result, applied, _calls = _install(
        tmp_path,
        published=VENDORED.read_bytes(),
        environment={"GPU_FAULT_RDS_CA_BUNDLE_SOURCE": str(override)},
    )

    if identical:
        assert result.returncode == 0, result.stderr
        assert applied == VENDORED.read_bytes()
    else:
        assert result.returncode == 1, result.stderr
        assert applied is None
        assert f"source:   {override}" in result.stderr


def test_the_vendored_bundle_fits_the_last_applied_annotation() -> None:
    """`kubectl apply` stores the whole ConfigMap as an annotation; a bundle
    that outgrew the API server's 256 KiB annotation limit would fail every
    deploy and config again, so a rotation that crosses it fails here first.
    The fix then is server-side apply or create/replace in the script."""

    document = {
        "apiVersion": "v1",
        "data": {"ca-bundle.pem": VENDORED.read_text(encoding="ascii")},
        "kind": "ConfigMap",
        "metadata": {
            "annotations": {},
            "creationTimestamp": None,
            "name": CONFIGMAP_NAME,
            "namespace": "gpu-fault-system",
        },
    }
    annotation = json.dumps(document, separators=(",", ":")) + "\n"

    assert len(annotation.encode()) < ANNOTATIONS_LIMIT_BYTES, (
        f"last-applied-configuration would be {len(annotation.encode())} bytes"
    )


def test_rolling_the_vendored_bundle_moves_the_deploy_host_identity() -> None:
    """A release whose only change is a rolled copy must still ship: the copy
    is hashed with the script that installs it, so the deploy is not
    QUALITY_ONLY (which keeps the site's snapshot)."""

    orchestration = file_set_identity(
        ROOT, deploy_source_identity.DEPLOY_HOST_ORCHESTRATION_INPUTS
    )["files"]

    assert VENDORED_BUNDLE in orchestration and APPLY_SCRIPT in orchestration
    assert orchestration[VENDORED_BUNDLE]["sha256"] == _pin()


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
