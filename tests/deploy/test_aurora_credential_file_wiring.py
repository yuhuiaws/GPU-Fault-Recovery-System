"""CP-3 (H1-1 / G-4): the Aurora DSN reaches running Pods as a file, not only
as a frozen env.

RDS rotates the managed master password every 7 days. With the DSN only in
``GPU_FAULT_STORE_URL`` (an env from a ``secretKeyRef``, never hot-reloaded)
every pooled connection recycled by ``max_idle``/``max_lifetime`` after a
rotation reconnected with the old password; the process degraded to
PoolTimeout/503 until the credential-refresh CronJob rolled the Deployments.
Now every control-plane Pod mounts the ``gpu-fault-aurora`` Secret as files
under one canonical directory and points ``GPU_FAULT_STORE_URL_FILE`` at the
``postgres-url`` key; the pool re-reads it on every connect. The env stays for
one release as the start-up fallback.

These tests pin the manifest wiring, the refresher's manifest (Jobs must carry
the ``app`` label the preflight filters on; the header must not claim running
replicas are unaffected), the verifier's mount check and the E-1 drain gate.
Nothing here talks to kubectl.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"
BASE = DEPLOY / "control-plane" / "base" / "control-plane-deployment.yaml"
GENERATED = DEPLOY / "control-plane" / "regional" / "generated"
CREDENTIAL_REFRESH = (
    DEPLOY / "control-plane" / "regional" / "aurora-credential-refresh.yaml"
)
TOOLS = DEPLOY / "control-plane" / "tools"
APPLY_SCRIPT = TOOLS / "apply-control-plane-role-split.sh"
VERIFY_SCRIPT = TOOLS / "verify_control_plane_role_split.py"

SECRET_NAME = "gpu-fault-aurora"
VOLUME_NAME = "aurora-credentials"
MOUNT_DIR = "/etc/gpu-fault/aurora"
DSN_FILE = f"{MOUNT_DIR}/postgres-url"
STATUS_FILE = f"{MOUNT_DIR}/last-refresh-status.json"
FILE_ENV = "GPU_FAULT_STORE_URL_FILE"
URL_ENV = "GPU_FAULT_STORE_URL"

ROLE_MANIFESTS = (
    "gpu-fault-api-ha-ingress.yaml",
    "gpu-fault-control-worker.yaml",
    "gpu-fault-telemetry-spool-worker.yaml",
)


def _docs(path: Path) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _pod_spec(path: Path) -> dict[str, Any]:
    (deployment,) = [d for d in _docs(path) if d.get("kind") == "Deployment"]
    return deployment["spec"]["template"]["spec"]


def _volume(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    for volume in spec.get("volumes", []):
        if volume.get("name") == name:
            return volume
    return None


def _assert_secret_mounted(spec: dict[str, Any], label: str) -> None:
    volume = _volume(spec, VOLUME_NAME)
    assert volume is not None, f"{label}: no {VOLUME_NAME} volume"
    secret = volume.get("secret") or {}
    assert secret.get("secretName") == SECRET_NAME, (
        f"{label}: {VOLUME_NAME} must project Secret {SECRET_NAME}"
    )
    # Fail closed: a Pod without its DSN must wedge, not run on the env alone.
    assert secret.get("optional") is not True, f"{label}: the mount must be required"
    # The whole Secret is projected so last-refresh-status.json lands beside
    # postgres-url without an items[] list that would have to be kept in sync.
    assert "items" not in secret, f"{label}: project the whole Secret, not items[]"

    container = spec["containers"][0]
    mounts = {m["name"]: m for m in container.get("volumeMounts", [])}
    assert VOLUME_NAME in mounts, f"{label}: container does not mount {VOLUME_NAME}"
    mount = mounts[VOLUME_NAME]
    assert mount.get("mountPath") == MOUNT_DIR, (
        f"{label}: {VOLUME_NAME} must mount at {MOUNT_DIR} so the DSN is {DSN_FILE}"
    )
    assert mount.get("readOnly") is True, f"{label}: the Secret mount is read-only"
    # kubelet never updates a subPath mount: the whole point is propagation.
    assert "subPath" not in mount and "subPathExpr" not in mount, (
        f"{label}: a subPath mount would freeze the DSN at Pod start"
    )


def _env(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in spec["containers"][0].get("env", [])}


def test_the_base_deployment_mounts_the_aurora_secret_as_files() -> None:
    _assert_secret_mounted(_pod_spec(BASE), BASE.name)


def test_the_base_deployment_points_the_store_at_the_mounted_dsn() -> None:
    env = _env(_pod_spec(BASE))
    assert env[FILE_ENV].get("value") == DSN_FILE, (
        f"{FILE_ENV} must name the projected postgres-url key"
    )
    # The env keeps the Secret as start-up fallback for one release; settings,
    # the executor and admission still read it before the store opens.
    assert env[URL_ENV].get("valueFrom", {}).get("secretKeyRef") == {
        "name": SECRET_NAME,
        "key": "postgres-url",
    }


def test_the_rendered_role_deployments_carry_the_mount() -> None:
    """The regional deploy applies regional/generated/, not the base. Re-render
    after editing the base or these drift."""

    for name in ROLE_MANIFESTS:
        _assert_secret_mounted(_pod_spec(GENERATED / name), name)


def test_the_rendered_postgres_config_points_at_the_mounted_dsn() -> None:
    """Literal env is externalised into the -config-postgres ConfigMap."""

    for role in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ):
        (config,) = _docs(GENERATED / f"{role}-config-postgres.yaml")
        assert config["data"].get(FILE_ENV) == DSN_FILE, role


# --- the refresher's manifest --------------------------------------------------


def _cronjob() -> dict[str, Any]:
    return next(d for d in _docs(CREDENTIAL_REFRESH) if d.get("kind") == "CronJob")


def test_refresh_jobs_carry_the_label_the_preflight_filters_on() -> None:
    """H1-3: ``check_aurora_refresh`` lists Jobs by
    ``app=gpu-fault-aurora-credential-refresh``. The CronJob controller (and
    ``kubectl create job --from=cronjob``) copy ``jobTemplate.metadata.labels``
    onto the Job; Pod-template labels never reach the Job object, so the
    'latest Job failed' branch matched the empty set."""

    template = _cronjob()["spec"]["jobTemplate"]
    labels = (template.get("metadata") or {}).get("labels") or {}
    assert labels.get("app") == "gpu-fault-aurora-credential-refresh"
    pod_labels = template["spec"]["template"]["metadata"]["labels"]
    assert pod_labels.get("app") == "gpu-fault-aurora-credential-refresh"


def test_the_refresher_no_longer_claims_running_replicas_are_unaffected() -> None:
    text = CREDENTIAL_REFRESH.read_text(encoding="utf-8")
    assert "not disturb running replicas" not in text
    assert "does not disturb running replicas" not in text
    assert "GPU_FAULT_STORE_URL_FILE" in text or "mounted" in text, (
        "the header must explain that running Pods reload the mounted Secret"
    )


def test_the_refresher_keeps_the_secret_role_for_status_writes() -> None:
    """H1-2: the refresher writes last-refresh-status.json into the Secret on
    every run; ``patch`` on that one Secret stays in the Role."""

    role = next(d for d in _docs(CREDENTIAL_REFRESH) if d.get("kind") == "Role")
    secret_rule = next(
        rule
        for rule in role["rules"]
        if rule.get("apiGroups") == [""] and rule.get("resources") == ["secrets"]
    )
    assert secret_rule["resourceNames"] == [SECRET_NAME]
    assert {"get", "patch"}.issubset(secret_rule["verbs"])


# --- the verifier checks the mount is live -----------------------------------


def test_verify_script_compiles() -> None:
    # In-process compile: ``py_compile`` would drop a __pycache__ under deploy/,
    # which the deploy layout gate rejects.
    compile(VERIFY_SCRIPT.read_text(encoding="utf-8"), str(VERIFY_SCRIPT), "exec")


def test_verify_script_checks_every_role_mounts_the_aurora_secret() -> None:
    text = VERIFY_SCRIPT.read_text(encoding="utf-8")
    assert VOLUME_NAME in text and MOUNT_DIR in text and SECRET_NAME in text, (
        "the verifier must fail the deploy when a role Deployment lost the mount"
    )
    for role in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ):
        assert role in text, f"the mount check must cover {role}"
    assert "subPath" in text, "a subPath mount must be rejected: it never updates"


# --- E-1: the spool drain gate reads the tier that still has spool enabled ----


def test_apply_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(APPLY_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def _function_body(text: str, name: str) -> str:
    start = text.index(f"{name}() {{")
    end = text.index("\n}\n", start)
    return text[start:end]


def test_spool_drain_gate_reads_the_spool_worker_not_the_ingress() -> None:
    """E-1: the disable sequence rolls ingress to spool=false first, so an
    ingress ``/metrics`` reports depth 0 whatever the table holds; the gate
    passed on the first poll and the spool tier was scaled away with rows
    still queued. The gate must read a Pod whose spool is still enabled -- the
    spool-worker -- and fall back to counting the table when none is Running."""

    body = _function_body(
        APPLY_SCRIPT.read_text(encoding="utf-8"), "wait_for_spool_drain"
    )
    assert "app=gpu-fault-telemetry-spool-worker" in body
    assert "app=gpu-fault-api-ha" not in body, (
        "the ingress reports depth 0 once its own spool admission is off"
    )
    assert re.search(r"count\(\*\)\s+FROM\s+gpu_fault_telemetry_spool", body), (
        "with no Running spool-worker the gate must count the table itself"
    )
    assert "gpu_fault_telemetry_spool_depth" in body
    assert "gpu_fault_telemetry_spool_leased" in body


def test_status_file_path_is_the_one_the_metrics_contributor_reads() -> None:
    """Documents the contract handed to the metrics contributor: the refresher
    writes ``last-refresh-status.json`` into the Secret, the mount makes it a
    file at this path."""

    assert STATUS_FILE == "/etc/gpu-fault/aurora/last-refresh-status.json"
