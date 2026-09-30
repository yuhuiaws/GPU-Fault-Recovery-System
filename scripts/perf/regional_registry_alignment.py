"""Perf registry hygiene: leave the regional registry Secret as it was found
and prove the control plane agrees.

* ``record_secret_baseline`` / ``baseline_serialization``: register() records
  the digest and the ``json.dumps`` shape of the pre-run ``clusters.json``
  (never the bytes -- they carry plaintext tokens); teardown rebuilds the
  exact bytes from the restored entries and that shape.
* ``verify_alignment``: the Secret's config digest must equal the durable
  head's and no Running api-ha replica may report
  ``regional_registry.secret_drift`` (``/healthz?verbose=1``). Run before a
  perf run touches the registry (a pre-existing drift is never attributed to
  the run) and after its teardown (a drift it caused fails its cleanup).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

HEALTHZ_CLIENT = r"""
import json
import sys
import urllib.error
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/healthz?verbose=1"
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        status, body = response.status, response.read()
except urllib.error.HTTPError as exc:
    status, body = exc.code, exc.read()
try:
    payload = json.loads(body)
except ValueError:
    payload = None
print(json.dumps({"http_status": status, "payload": payload}))
"""

# The Secret baseline the run records at register time: the digest of the
# pre-run ``clusters.json`` bytes and the ``json.dumps`` shape that produced
# them, never the bytes themselves (they carry plaintext tokens). Teardown
# rebuilds the bytes from the restored entries and this shape.
SECRET_BASELINE_FILE = "registry-secret-baseline.json"
# An api-ha replica compares the Secret it STARTED on with the durable head
# (``secret_drift``); a replica that started inside a run's window (HA-005
# rolls the control plane mid-run) holds the run's transient Secret digest and
# reports drift after the restore although Secret and head agree. Teardown
# reconciles exactly those replicas with a rolling restart of the Deployment.
API_DEPLOYMENT = "deployment/gpu-fault-api-ha"
ROLLOUT_TIMEOUT = "600s"
SECRET_RESTORE_FILE = "registry-secret-restore.json"
SERIALIZATIONS: tuple[dict[str, Any], ...] = tuple(
    {
        "indent": indent,
        "separators": list(separators),
        "sort_keys": sort_keys,
        "trailing_newline": newline,
    }
    for indent in (None, 1, 2, 4)
    for separators in ((", ", ": "), (",", ":"), (",", ": "))
    for sort_keys in (False, True)
    for newline in (False, True)
)


def serialize_entries(entries: list[dict[str, Any]], shape: dict[str, Any]) -> str:
    """``json.dumps`` of ``entries`` in one recorded shape."""

    text = json.dumps(
        entries,
        indent=shape.get("indent"),
        separators=tuple(shape["separators"]),
        sort_keys=bool(shape.get("sort_keys")),
    )
    return text + "\n" if shape.get("trailing_newline") else text


def detect_serialization(
    raw: bytes, entries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The ``json.dumps`` shape that reproduces ``raw`` from ``entries``, or
    None when no known shape does (nothing is guessed)."""

    for shape in SERIALIZATIONS:
        if serialize_entries(entries, shape).encode() == raw:
            return dict(shape)
    return None


def record_secret_baseline(
    artifacts: Path, raw: bytes, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write ``SECRET_BASELINE_FILE``: digest, length and shape of the pre-run
    ``clusters.json``; tokens never leave the Secret."""

    record = {
        "clusters_json_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
        "entry_count": len(entries),
        "serialization": detect_serialization(raw, entries),
    }
    path = artifacts / SECRET_BASELINE_FILE
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return record


def baseline_serialization(
    artifacts: Path, entries: list[dict[str, Any]]
) -> str | None:
    """The pre-run bytes as text when ``entries`` re-serialised in the recorded
    shape reproduce the recorded digest; None when the record is missing, the
    shape unknown, or the content has since changed."""

    path = artifacts / SECRET_BASELINE_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    shape = record.get("serialization")
    if not isinstance(shape, dict):
        return None
    text = serialize_entries(entries, shape)
    if hashlib.sha256(text.encode()).hexdigest() != record.get("clusters_json_sha256"):
        return None
    return text


def api_replica_registry_health(
    control: Callable[..., str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(replicas, skipped)``: ``regional_registry`` from ``/healthz?verbose=1``
    of every Running api-ha replica, and the Pods not Running (not asked)."""

    inventory = json.loads(
        control("get", "pod", "-l", "app=gpu-fault-api-ha", "-o", "json")
    )
    replicas: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in inventory.get("items", []):
        name = str(item["metadata"]["name"])
        phase = (item.get("status") or {}).get("phase")
        if phase != "Running":
            skipped.append({"pod": name, "phase": phase})
            continue
        containers = (item.get("spec") or {}).get("containers") or [{}]
        ports = containers[0].get("ports") or [{}]
        port = int(ports[0].get("containerPort") or 8080)
        output = control(
            "exec", "-i", name, "--", "python3", "-c", HEALTHZ_CLIENT, str(port)
        )
        answer = json.loads(output.splitlines()[-1])
        registry_state = (answer.get("payload") or {}).get("regional_registry") or {}
        replicas.append(
            {
                "pod": name,
                "phase": phase,
                "created_at": item["metadata"].get("creationTimestamp"),
                "http_status": answer.get("http_status"),
                "ready": registry_state.get("ready"),
                "secret_drift": registry_state.get("secret_drift"),
                "secret_config_sha256": registry_state.get("secret_config_sha256"),
            }
        )
    return replicas, skipped


def _config_field_differences(
    secret_side: list[Any], head_side: list[Any]
) -> dict[str, list[str]]:
    """Per cluster id, the config-digest fields on which the Secret and the
    durable head disagree (``only in ...`` when one side lacks the cluster)."""

    from gpu_fault.regional_registry import _CONFIG_DIGEST_FIELDS

    secret_by_id = {item.cluster_id: item for item in secret_side}
    head_by_id = {item.cluster_id: item for item in head_side}
    differences: dict[str, list[str]] = {}
    for cluster_id in sorted(secret_by_id.keys() | head_by_id.keys()):
        left, right = secret_by_id.get(cluster_id), head_by_id.get(cluster_id)
        if left is None:
            differences[cluster_id] = ["only in the durable head"]
        elif right is None:
            differences[cluster_id] = ["only in the Secret"]
        else:
            fields = [
                field
                for field in _CONFIG_DIGEST_FIELDS
                if getattr(left, field) != getattr(right, field)
            ]
            if fields:
                differences[cluster_id] = fields
    return differences


def verify_alignment(
    control: Callable[..., str],
    *,
    secret_document: Callable[[], tuple[dict[str, Any], bytes, list[dict[str, Any]]]],
    durable_revision: Callable[[], dict[str, Any]],
    artifacts: Path | None,
    phase: str,
    restart_window_start: str | None = None,
) -> dict[str, Any]:
    """Prove the Secret, the durable head and every api-ha replica agree.

    The Secret's config digest must equal the head's, and no Running replica
    may report ``regional_registry.secret_drift``. The report is written to
    ``registry-alignment-<phase>.json``; a disagreement raises with the
    differing fields and replicas named.

    ``restart_window_start`` (the preflight's ``checked_at``) enables the one
    tolerated remedy: when Secret and head agree and every drifting replica
    was created after that instant, the replicas started on the run's
    transient Secret; they are rolled (``kubectl rollout restart`` + ``rollout
    status``) and the check repeats. A replica that predates the window or a
    Secret/head disagreement is never restarted away.
    """

    from gpu_fault.regional import RegionalClusterRegistration
    from gpu_fault.regional_registry import (
        configured_regional_registrations,
        regional_registry_config_sha256,
    )

    _metadata, _raw, values = secret_document()
    secret_side = configured_regional_registrations(values)
    revision = durable_revision()
    head_side = [
        RegionalClusterRegistration.model_validate(item)
        for item in revision["registrations"]
    ]
    secret_digest = regional_registry_config_sha256(secret_side)
    head_digest = regional_registry_config_sha256(head_side)
    replicas, skipped = api_replica_registry_health(control)
    errors: list[str] = []
    differences: dict[str, list[str]] = {}
    if secret_digest != head_digest:
        differences = _config_field_differences(secret_side, head_side)
        errors.append(
            f"Secret config digest {secret_digest[:12]} != durable head "
            f"{head_digest[:12]} (generation {revision['generation']}): "
            + "; ".join(
                f"{cluster}: {', '.join(fields)}"
                for cluster, fields in differences.items()
            )
        )
    for replica in replicas:
        if replica.get("secret_drift") is not False or replica.get("ready") is not True:
            errors.append(
                f"{replica['pod']} reports secret_drift={replica.get('secret_drift')!r}"
                f" ready={replica.get('ready')!r}"
                f" (secret_config_sha256={replica.get('secret_config_sha256')})"
            )
    restarted: list[str] = []
    if errors and not differences and restart_window_start is not None:
        drifting = [item for item in replicas if item.get("secret_drift") is not False]
        if drifting and all(
            _started_after(item.get("created_at"), restart_window_start)
            for item in drifting
        ):
            restarted = [str(item["pod"]) for item in drifting]
            control("rollout", "restart", API_DEPLOYMENT)
            control("rollout", "status", API_DEPLOYMENT, f"--timeout={ROLLOUT_TIMEOUT}")
            replicas, skipped = api_replica_registry_health(control)
            errors = [
                f"{replica['pod']} reports secret_drift={replica.get('secret_drift')!r}"
                f" ready={replica.get('ready')!r} after the rolling restart"
                for replica in replicas
                if replica.get("secret_drift") is not False
                or replica.get("ready") is not True
            ]
    report = {
        "phase": phase,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "aligned": not errors,
        "secret_config_sha256": secret_digest,
        "durable_config_sha256": head_digest,
        "durable_generation": revision["generation"],
        "field_differences": differences,
        "replicas": replicas,
        "skipped_replicas": skipped,
        "restarted_replicas": restarted,
        "errors": errors,
    }
    if artifacts is not None:
        (artifacts / f"registry-alignment-{phase}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    if errors:
        raise RuntimeError(f"registry alignment ({phase}) failed: " + "; ".join(errors))
    return report


def _started_after(created_at: Any, window_start: str) -> bool:
    """Whether a Pod's ``creationTimestamp`` lies after the preflight instant."""

    if not isinstance(created_at, str) or not created_at:
        return False
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        start = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
    except ValueError:
        return False
    return created > start


def preflight_checked_at(artifacts: Path | None) -> str | None:
    """The instant the preflight alignment ran, from its report on disk."""

    if artifacts is None:
        return None
    path = artifacts / "registry-alignment-preflight.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text()).get("checked_at")
    except (ValueError, OSError):
        return None
    return value if isinstance(value, str) else None
