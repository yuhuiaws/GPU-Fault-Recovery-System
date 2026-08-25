"""Keep the Aurora connection Secret in step with a rotated master password.

Aurora clusters created with ``--manage-master-user-password`` have RDS
rotate the master password on a schedule (7 days by default). The control
plane reads its DSN from the ``gpu-fault-aurora`` Secret exactly once, when
``PostgresStore.__init__`` builds the connection pool, so a rotation does not
disturb running replicas. It breaks the *next* restart instead: whatever
triggers it (a rollout, an eviction, a node cycle, a probe restart) the new
process authenticates with the stale password and dies, which surfaces as
every replica in ``CrashLoopBackOff`` hours or days after the rotation.

This module closes that gap. It reads ``AWSCURRENT`` from Secrets Manager,
splices the password into the existing DSN, and — only when the two actually
differ — writes the Secret and restarts the Deployment so the new password
reaches ``GPU_FAULT_STORE_URL`` (an ``env`` from ``secretKeyRef`` is not
hot-reloaded).

Two properties matter for running this unattended in production:

* **Validate before switching.** The candidate DSN is opened against the
  database first. A refresher that cannot tell a good password from a bad one
  is free to replace a working Secret with a broken one, turning a rotation
  into an outage of its own making.
* **Do nothing when nothing changed.** Restarting replicas is the only step
  here that can disrupt serving, so it happens only after a real write.

Discovered as ``GF-REGIONAL-BOOT-016`` 附带发现 2.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "gpu-fault-system"
DEFAULT_SECRET_NAME = "gpu-fault-aurora"
DEFAULT_SECRET_KEY = "postgres-url"
DEFAULT_DEPLOYMENT = "gpu-fault-api-ha"
RESTART_ANNOTATION = "gpu-fault.aws/aurora-credential-refreshed-at"


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one refresh attempt.

    ``rotated`` is the only field that implies a write happened; the
    ``reason`` is carried for logging so an operator reading CronJob output
    can tell "already current" apart from "no managed secret configured"
    without correlating timestamps.
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
    verify=verify_dsn,
    fetch_password=read_current_password,
) -> RefreshResult:
    """Bring ``secret_name`` up to date with the rotated master password.

    ``verify`` and ``fetch_password`` are injected so the decision logic can
    be tested without AWS or a database.
    """
    import base64

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

    candidate = replace_password(current_dsn, fetch_password(secret_arn, region_name))
    password_changed = candidate != current_dsn
    if not password_changed and not arn_changed:
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
    core.patch_namespaced_secret(
        secret_name,
        namespace,
        {"data": updates},
    )

    # GPU_FAULT_STORE_URL comes from a secretKeyRef, so the running replicas
    # keep the old value until their pods are replaced. maxUnavailable=1 plus
    # the PodDisruptionBudget keeps a quorum serving through the rollout.
    targets = deployments or (deployment,)
    if password_changed:
        for target in targets:
            apps.patch_namespaced_deployment(
                target,
                namespace,
                {
                    "spec": {
                        "template": {
                            "metadata": {"annotations": {RESTART_ANNOTATION: timestamp}}
                        }
                    }
                },
            )
            LOGGER.info("restarted deployment %s/%s", namespace, target)
    return RefreshResult(
        rotated=True,
        restarted=password_changed,
        reason=(
            "credentials updated and deployment restarted"
            if password_changed
            else "managed secret ARN updated"
        ),
    )


def main() -> None:
    from datetime import datetime, timezone

    from kubernetes import client, config

    logging.basicConfig(
        level=os.environ.get("GPU_FAULT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    region_name = (
        os.environ.get("GPU_FAULT_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or ""
    ).strip()
    if not region_name:
        raise RuntimeError("AWS_REGION is required")
    config.load_incluster_config()
    core = client.CoreV1Api()
    apps = client.AppsV1Api()
    namespace = os.environ.get("GPU_FAULT_NAMESPACE", DEFAULT_NAMESPACE)
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
                secret_name=os.environ.get(
                    "GPU_FAULT_AURORA_SECRET", DEFAULT_SECRET_NAME
                ),
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
        observed_at = datetime.now(timezone.utc).isoformat()
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
                        "name": os.environ.get(
                            "GPU_FAULT_AURORA_SECRET",
                            DEFAULT_SECRET_NAME,
                        ),
                        "namespace": namespace,
                    },
                    "reason": "AuroraCredentialRefreshFailed",
                    "message": (f"{type(last_error).__name__}: {last_error}"),
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
    LOGGER.info(
        "aurora credential refresh: rotated=%s restarted=%s (%s)",
        result.rotated,
        result.restarted,
        result.reason,
    )


if __name__ == "__main__":
    main()
