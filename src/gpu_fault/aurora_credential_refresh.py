"""Keep the Aurora connection Secret in step with a rotated master password.

Aurora clusters created with ``--manage-master-user-password`` have RDS
rotate the master password on a schedule (7 days by default). Every
control-plane Pod mounts the ``gpu-fault-aurora`` Secret as files under
``/etc/gpu-fault/aurora`` and the connection pool reads ``postgres-url`` from
there on every connect (``GPU_FAULT_STORE_URL_FILE``, CP-3), so the only thing
a rotation needs is for the Secret to be brought up to date: kubelet rewrites
the projected file in every running Pod within its sync period and the next
reconnect authenticates with the new password. Nothing restarts.

Before that mount existed the DSN was a frozen ``env`` from a ``secretKeyRef``.
"Running replicas are unaffected" was never true: every connection the pool
recycled after a rotation (``max_idle`` 300 s, ``max_lifetime`` 3600 s)
reconnected with the old password, psycopg_pool retried for 300 s per slot and
then gave it up, and each replica degraded to ``PoolTimeout``/503 until this
job rolled it -- while every NEW Pod died in ``CrashLoopBackOff``.

This module reads ``AWSCURRENT`` from Secrets Manager, splices the password
into the existing DSN, and writes the Secret when the two actually differ.
It no longer rolls the Deployments; ``--restart-deployments`` (default off)
keeps that behaviour for a site whose Pods do not mount the Secret yet.

Three properties matter for running this unattended in production:

* **Validate before switching.** The candidate DSN is opened against the
  database first. A refresher that cannot tell a good password from a bad one
  is free to replace a working Secret with a broken one, turning a rotation
  into an outage of its own making.
* **Report every run.** The outcome -- ok or failed, when, why -- is written
  into the same Secret as ``last-refresh-status.json``. The mount turns it
  into ``/etc/gpu-fault/aurora/last-refresh-status.json`` in every Pod and the
  control plane exports its age as
  ``gpu_fault_aurora_credential_refresh_last_success_age_seconds`` (H1-2),
  so a refresher that has been failing for days is an alert, not a surprise
  at the next rotation.
* **Converge, do not repeat.** The rollout token is a digest of the candidate
  DSN (H1-4), so two refreshers racing (the hourly tick and a release
  preflight Job) agree on it; with restarts enabled a Deployment already
  carrying that token is left alone.

Discovered as ``GF-REGIONAL-BOOT-016`` 附带发现 2; reworked for CP-3.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

from gpu_fault.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "gpu-fault-system"
DEFAULT_SECRET_NAME = "gpu-fault-aurora"
DEFAULT_SECRET_KEY = "postgres-url"
DEFAULT_DEPLOYMENT = "gpu-fault-api-ha"
RESTART_ANNOTATION = "gpu-fault.aws/aurora-credential-refreshed-at"
# Written into the Secret on every run; the Deployment's aurora-credentials
# mount makes it /etc/gpu-fault/aurora/last-refresh-status.json in every Pod.
STATUS_KEY = "last-refresh-status.json"
RESTART_SWITCH_ENV = "GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS"

_DSN_IN_TEXT = re.compile(r"(postgres(?:ql)?://[^:/\s]+:)[^@\s]+@")


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one refresh attempt.

    ``rotated`` means the Secret changed, while ``restarted`` means at least
    one Deployment pod template changed. The ``reason`` is carried for
    logging so an operator can distinguish a converged no-op from a resumed
    rollout without correlating timestamps.
    """

    rotated: bool
    restarted: bool
    reason: str


def replace_password(dsn: str, password: str) -> str:
    """Return ``dsn`` with its password replaced, everything else intact.

    The endpoint, database name, username and query string (``sslmode``,
    ``options``, ...) are preserved verbatim rather than rebuilt, so a DSN
    that was hand-tuned after deployment survives a rotation.
    """
    parts = urlsplit(dsn)
    if not parts.hostname:
        raise ValueError("DSN has no host")
    if not parts.username:
        raise ValueError("DSN has no username")
    userinfo = f"{quote(parts.username, safe='')}:{quote(password, safe='')}"
    netloc = f"{userinfo}@{parts.hostname}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def rollout_token_for(dsn: str) -> str:
    """Deterministic token for ``dsn``: concurrent refreshers converge on it
    and it leaks nothing about the password (H1-4)."""

    return hashlib.sha256(dsn.encode()).hexdigest()[:16]


def _redact_text(text: str) -> str:
    """Blank the password in any DSN embedded in free text (error messages)."""

    return _DSN_IN_TEXT.sub(r"\1***@", text)


def _redact(dsn: str) -> str:
    """Render a DSN safe to log: keep shape, drop the password."""
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "<unparsable dsn>"
    if not parts.hostname:
        return "<unparsable dsn>"
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    return f"{parts.scheme}://{user}:***@{host}{parts.path}"


def read_current_password(secret_arn: str, region_name: str) -> str:
    """Fetch the ``AWSCURRENT`` password for an RDS-managed master secret.

    ``AWSCURRENT`` is requested explicitly. During a rotation Secrets Manager
    can hold an ``AWSPENDING`` version that the database has not accepted
    yet, and the default stage resolution is not something to leave implicit
    when the consequence of picking wrong is a crash loop.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("install gpu-fault-control-plane[hyperpod]") from exc
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_arn, VersionStage="AWSCURRENT")
    payload = json.loads(response["SecretString"])
    password = payload.get("password")
    if not password:
        raise RuntimeError(f"managed secret {secret_arn} has no password field")
    return password


def verify_dsn(dsn: str, connect_timeout: int = 10) -> None:
    """Open ``dsn`` once, so a bad candidate never reaches the Secret."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("install gpu-fault-control-plane[postgres]") from exc
    with psycopg.connect(dsn, connect_timeout=connect_timeout) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()


def _annotations(value: Any) -> dict[str, str]:
    metadata = getattr(value, "metadata", None)
    return dict(getattr(metadata, "annotations", None) or {})


def _reconcile_deployments(
    apps: Any,
    *,
    namespace: str,
    targets: tuple[str, ...],
    rollout_token: str,
) -> tuple[str, ...]:
    restarted = []
    for target in targets:
        deployment = apps.read_namespaced_deployment(target, namespace)
        template = getattr(getattr(deployment, "spec", None), "template", None)
        if _annotations(template).get(RESTART_ANNOTATION) == rollout_token:
            continue
        apps.patch_namespaced_deployment(
            target,
            namespace,
            {
                "spec": {
                    "template": {
                        "metadata": {"annotations": {RESTART_ANNOTATION: rollout_token}}
                    }
                }
            },
        )
        restarted.append(target)
        LOGGER.info("restarted deployment %s/%s", namespace, target)
    return tuple(restarted)


def refresh_once(
    core,
    apps,
    *,
    namespace: str,
    secret_name: str,
    secret_key: str,
    deployment: str,
    deployments: tuple[str, ...] | None = None,
    region_name: str,
    timestamp: str,
    expected_secret_arn: str | None = None,
    verify=None,
    fetch_password=None,
    restart_deployments: bool = False,
) -> RefreshResult:
    """Bring ``secret_name`` up to date with the rotated master password.

    ``verify`` and ``fetch_password`` are injected so the decision logic can
    be tested without AWS or a database. ``restart_deployments`` is the
    compatibility switch: running Pods reload the mounted Secret, so by
    default no Deployment is touched.
    """
    # Resolved at call time so a test (or an operator shim) can replace the
    # module's verifier and password source.
    verify = verify or verify_dsn
    fetch_password = fetch_password or read_current_password
    secret = core.read_namespaced_secret(secret_name, namespace)
    data = secret.data or {}
    if secret_key not in data:
        raise RuntimeError(f"{namespace}/{secret_name} has no {secret_key} key")
    current_dsn = base64.b64decode(data[secret_key]).decode()

    encoded_arn = data.get("master-secret-arn")
    if not encoded_arn:
        # Self-managed password: there is no rotation to track, and guessing
        # would be worse than doing nothing.
        return RefreshResult(
            rotated=False,
            restarted=False,
            reason="secret has no master-secret-arn; nothing to refresh",
        )
    stored_secret_arn = base64.b64decode(encoded_arn).decode().strip()
    secret_arn = expected_secret_arn or stored_secret_arn
    arn_changed = secret_arn != stored_secret_arn
    rollout_token = _annotations(secret).get(RESTART_ANNOTATION)
    if not restart_deployments:
        # Without restarts there is no rollout to resume; a token left by an
        # earlier restarting build must not make a converged Secret look busy.
        rollout_token = None

    candidate = replace_password(current_dsn, fetch_password(secret_arn, region_name))
    password_changed = candidate != current_dsn
    targets = deployments or (deployment,)
    if not password_changed and not arn_changed and rollout_token is None:
        return RefreshResult(
            rotated=False,
            restarted=False,
            reason="secret already carries the AWSCURRENT password",
        )

    # The password differs, so a rotation happened. Prove the new one works
    # before overwriting a Secret that, however stale, may still be the only
    # copy of a password something is running on.
    if password_changed:
        verify(candidate)
    LOGGER.info(
        "aurora credentials changed; updating %s/%s (%s)",
        namespace,
        secret_name,
        _redact(candidate),
    )
    updates = {"master-secret-arn": base64.b64encode(secret_arn.encode()).decode()}
    if password_changed:
        updates[secret_key] = base64.b64encode(candidate.encode()).decode()
        rollout_token = rollout_token_for(candidate)
    if password_changed or arn_changed:
        patch: dict[str, object] = {"data": updates}
        if password_changed:
            patch["metadata"] = {"annotations": {RESTART_ANNOTATION: rollout_token}}
        core.patch_namespaced_secret(
            secret_name,
            namespace,
            patch,
        )

    # Running Pods read the DSN from the mounted Secret file on every connect;
    # kubelet delivers the new value within its sync period. Rolling the
    # Deployments is only for a site that still injects the DSN as an env.
    restarted_targets = (
        _reconcile_deployments(
            apps,
            namespace=namespace,
            targets=targets,
            rollout_token=rollout_token,
        )
        if restart_deployments and rollout_token is not None
        else ()
    )
    rotated = password_changed or arn_changed
    restarted = bool(restarted_targets)
    if password_changed and restarted:
        reason = "credentials updated and deployments reconciled"
    elif password_changed:
        reason = "credentials updated; running replicas reload the mounted Secret"
    elif arn_changed and restarted:
        reason = "managed secret ARN updated and deployment rollout resumed"
    elif arn_changed:
        reason = "managed secret ARN updated"
    elif restarted:
        reason = "deployment rollout resumed for current credentials"
    else:
        reason = "secret and database consumers already carry current credentials"
    return RefreshResult(
        rotated=rotated,
        restarted=restarted,
        reason=reason,
    )


def write_refresh_status(
    core: Any,
    *,
    namespace: str,
    secret_name: str,
    status: str,
    finished_at: str,
    error: str | None,
    rotated: bool,
    restarted: bool,
    reason: str | None,
) -> None:
    """Record this run's outcome as ``STATUS_KEY`` in the Secret (H1-2).

    Only that one key is patched; ``postgres-url`` and ``master-secret-arn``
    are never touched here. Error text is redacted so a DSN quoted by a driver
    exception cannot land in the Secret's status.
    """

    payload = {
        "status": status,
        "finished_at": finished_at,
        "error": _redact_text(error) if error is not None else None,
        "rotated": rotated,
        "restarted": restarted,
        "reason": reason,
    }
    core.patch_namespaced_secret(
        secret_name,
        namespace,
        {
            "data": {
                STATUS_KEY: base64.b64encode(
                    json.dumps(payload, sort_keys=True).encode()
                ).decode()
            }
        },
    )


def load_kubernetes_clients() -> tuple[Any, Any]:
    from kubernetes import client, config

    config.load_incluster_config()
    return client.CoreV1Api(), client.AppsV1Api()


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gpu-fault-aurora-credential-refresh",
        description=(
            "Copy the AWSCURRENT Aurora master password into the gpu-fault-aurora "
            "Secret; running Pods reload it from their mount."
        ),
    )
    parser.add_argument(
        "--restart-deployments",
        action="store_true",
        default=(os.environ.get(RESTART_SWITCH_ENV, "false").strip().lower() == "true"),
        help=(
            "Also roll the database-consuming Deployments after a rotation "
            "(compatibility for Pods that do not mount the Secret; default off, "
            f"or {RESTART_SWITCH_ENV}=true)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    from datetime import datetime, timezone

    arguments = _parse_arguments(argv)
    configure_logging()
    region_name = (
        os.environ.get("GPU_FAULT_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or ""
    ).strip()
    if not region_name:
        raise RuntimeError("AWS_REGION is required")
    core, apps = load_kubernetes_clients()
    namespace = os.environ.get("GPU_FAULT_NAMESPACE", DEFAULT_NAMESPACE)
    secret_name = os.environ.get("GPU_FAULT_AURORA_SECRET", DEFAULT_SECRET_NAME)
    deployments = tuple(
        item.strip()
        for item in os.environ.get(
            "GPU_FAULT_AURORA_RESTART_DEPLOYMENTS",
            os.environ.get(
                "GPU_FAULT_CONTROL_PLANE_DEPLOYMENT",
                DEFAULT_DEPLOYMENT,
            ),
        ).split(",")
        if item.strip()
    )
    max_attempts = int(os.environ.get("GPU_FAULT_AURORA_REFRESH_MAX_ATTEMPTS", "3"))
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = refresh_once(
                core,
                apps,
                namespace=namespace,
                secret_name=secret_name,
                secret_key=os.environ.get(
                    "GPU_FAULT_AURORA_SECRET_KEY", DEFAULT_SECRET_KEY
                ),
                deployment=deployments[0],
                deployments=deployments,
                region_name=region_name,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                expected_secret_arn=(
                    os.environ.get("GPU_FAULT_AURORA_MASTER_SECRET_ARN") or None
                ),
                restart_deployments=arguments.restart_deployments,
            )
            break
        except Exception as exc:
            last_error = exc
            LOGGER.exception(
                "aurora credential refresh attempt %s/%s failed",
                attempt,
                max_attempts,
            )
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 30))
    else:
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            write_refresh_status(
                core,
                namespace=namespace,
                secret_name=secret_name,
                status="failed",
                finished_at=observed_at,
                error=f"{type(last_error).__name__}: {last_error}",
                rotated=False,
                restarted=False,
                reason=None,
            )
        except Exception:
            LOGGER.exception("failed to record the Aurora credential refresh status")
        try:
            core.create_namespaced_event(
                namespace,
                {
                    "apiVersion": "v1",
                    "kind": "Event",
                    "metadata": {
                        "generateName": "gpu-fault-aurora-refresh-",
                        "namespace": namespace,
                    },
                    "involvedObject": {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "name": secret_name,
                        "namespace": namespace,
                    },
                    "reason": "AuroraCredentialRefreshFailed",
                    "message": _redact_text(
                        f"{type(last_error).__name__}: {last_error}"
                    ),
                    "type": "Warning",
                    "source": {"component": "gpu-fault-aurora-refresh"},
                    "firstTimestamp": observed_at,
                    "lastTimestamp": observed_at,
                    "count": 1,
                },
            )
        except Exception:
            LOGGER.exception("failed to create Aurora credential refresh Warning Event")
        finally:
            raise RuntimeError(
                "aurora credential refresh failed after retries"
            ) from last_error
    write_refresh_status(
        core,
        namespace=namespace,
        secret_name=secret_name,
        status="ok",
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        error=None,
        rotated=result.rotated,
        restarted=result.restarted,
        reason=result.reason,
    )
    LOGGER.info(
        "aurora credential refresh: rotated=%s restarted=%s (%s)",
        result.rotated,
        result.restarted,
        result.reason,
    )


if __name__ == "__main__":
    main()
