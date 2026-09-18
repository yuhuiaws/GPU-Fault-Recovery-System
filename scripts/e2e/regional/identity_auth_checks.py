"""Pure identity boundary, credential-shape and rotation evidence checks."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime
from typing import Any

from scripts.e2e.regional.identity_acceptance_common import (
    IdentityAcceptanceError,
    secret_digest,
)

AUTH015_PROOF_GAPS = {
    "installation_time_master_custody": (
        "Post-installation snapshots cannot prove that the master was never "
        "present during installation."
    ),
    "deployed_node_a_key_activation": (
        "Secret provisioning does not activate the new key in Node Runtime."
    ),
    "deployed_cross_node_command_and_result_signatures": (
        "Local heartbeat signature tests do not exercise deployed command "
        "or result-query signatures."
    ),
}


def verdict(checks: dict[str, Any]) -> str:
    """Only nonempty, literal boolean checks can pass."""

    return (
        "PASS" if checks and all(value is True for value in checks.values()) else "FAIL"
    )


def failure_details(
    *,
    checks: dict[str, Any],
    cleanup_errors: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {"checks": checks, "cleanup_errors": cleanup_errors, **extra}


def auth015_scan_checks(
    scans_before: dict[str, Any],
    scans_after: dict[str, Any],
    *,
    tests: dict[str, Any],
    rotation_returncode: int,
    nodes: tuple[str, str],
    before_keys: dict[str, str],
    after_keys: dict[str, str],
    before_agents: dict[str, Any],
    after_agents: dict[str, Any],
) -> dict[str, bool]:
    return {
        "host_scan_before_zero_matches": all(
            not item["master_matches"] for item in scans_before.values()
        ),
        "host_scan_after_zero_matches": all(
            valid_master_scan(item) for item in scans_after.values()
        ),
        "cross_node_signature_tests": tests.get("passed") is True,
        "rotation_command_succeeded": rotation_returncode == 0,
        "node_a_key_changed": before_keys.get(nodes[0]) != after_keys.get(nodes[0]),
        "node_b_key_unchanged": before_keys.get(nodes[1]) == after_keys.get(nodes[1]),
        "no_other_node_key_changed": all(
            before_keys.get(node) == after_keys.get(node)
            for node in before_keys
            if node != nodes[0]
        ),
        "node_b_agent_continues": (
            before_agents["agents"][nodes[1]]["generation"]
            == after_agents["agents"][nodes[1]]["generation"]
            and after_agents["agents"][nodes[1]]["lifecycle_state"] == "ACTIVE"
        ),
        "node_b_heartbeat_advanced": heartbeat_advanced(
            before_agents["agents"][nodes[1]], after_agents["agents"][nodes[1]]
        ),
    }


def valid_master_scan(item: dict[str, Any]) -> bool:
    return (
        isinstance(item.get("master_matches"), list)
        and not item["master_matches"]
        and type(item.get("values_scanned")) is int
        and item["values_scanned"] > 0
        and item.get("host_tmp_exists") is True
        and item.get("systemd_environment_exists") is True
    )


def validated_node_key_rotation(
    before_keys: dict[str, str],
    before_cpu_keys: dict[str, str],
    after_gpu: dict[str, Any],
    after_cpu: dict[str, Any],
    node: str,
) -> dict[str, str]:
    after_keys = node_key_digests(after_gpu)
    after_cpu_keys = node_key_digests(after_cpu)
    if (
        set(before_keys) != set(after_keys)
        or set(before_cpu_keys) != set(after_cpu_keys)
        or {key for key in before_keys if before_keys[key] != after_keys[key]} != {node}
        or {
            key
            for key in before_cpu_keys
            if before_cpu_keys[key] != after_cpu_keys[key]
        }
        != {node}
        or after_cpu_keys[node] != after_keys[node]
    ):
        raise IdentityAcceptanceError(
            "node-key provisioning changed an unexpected key set"
        )
    return after_keys


INSTALLER_SECRET = "gpu-fault-control-plane-active"


def execution_token_hits(
    secrets_document: dict[str, Any],
    pods_document: dict[str, Any],
    *,
    digests: set[str],
) -> list[dict[str, str]]:
    """Where the data plane holds the execution token, by name or by value.

    ``digests`` are the SHA-256 of the token as the API Pod holds it (raw and
    stripped); every Secret value and every literal Pod env value is digested
    and compared, so a token stored under an innocent key is found too. Only
    names and keys are returned, never values.
    """

    matches = []
    pattern = re.compile(r"execution[_.-]?token", re.IGNORECASE)

    from scripts.e2e.regional.credential_value_scan import (
        CredentialScanError,
        credential_value_digests,
    )

    def value_digests(raw: bytes, *, json_container: bool = False) -> set[str]:
        try:
            return credential_value_digests(raw, require_json=json_container)
        except CredentialScanError as exc:
            raise IdentityAcceptanceError(str(exc)) from None

    if any(
        not isinstance(document.get("items"), list)
        or (document.get("metadata") or {}).get("continue")
        for document in (secrets_document, pods_document)
    ):
        raise IdentityAcceptanceError("credential object inventory is incomplete")
    for item in secrets_document.get("items", []):
        name = str(item.get("metadata", {}).get("name", ""))
        for key, encoded in (item.get("data") or {}).items():
            try:
                raw = base64.b64decode(str(encoded), validate=True)
            except (ValueError, TypeError):
                raise IdentityAcceptanceError(
                    "credential Secret encoding is invalid"
                ) from None
            by_value = bool(
                value_digests(raw, json_container=str(key).endswith(".json")) & digests
            )
            by_name = pattern.search(str(key)) is not None
            if by_value or by_name:
                matches.append(
                    {
                        "kind": "Secret",
                        "name": name,
                        "key": str(key),
                        "match": "value" if by_value else "name",
                    }
                )
    for item in pods_document.get("items", []):
        name = str(item.get("metadata", {}).get("name", ""))
        spec = item.get("spec", {})
        containers = [
            *spec.get("initContainers", []),
            *spec.get("containers", []),
            *spec.get("ephemeralContainers", []),
        ]
        for container in containers:
            for entry in container.get("env", []):
                literal = entry.get("value")
                by_value = isinstance(literal, str) and bool(
                    value_digests(literal.encode()) & digests
                )
                by_name = pattern.search(str(entry.get("name", ""))) is not None
                if by_value or by_name:
                    matches.append(
                        {
                            "kind": "Pod",
                            "name": name,
                            "key": str(entry.get("name")),
                            "match": "value" if by_value else "name",
                        }
                    )
    return matches


def registry_token_digests(
    entries: list[dict[str, Any]],
    cluster_id: str,
) -> tuple[str | None, str | None, str | None]:
    """(token digest, retiring digest, deadline) for one cluster.

    The bootstrap Secret carries plaintext tokens and a durable revision carries
    digests; comparing digests lets a restore be checked on either kind of site
    without handling the plaintext again.
    """

    for item in entries:
        if item.get("cluster_id") != cluster_id:
            continue
        token = item.get("token")
        retiring = item.get("retiring_token")
        return (
            secret_digest(str(token)) if token else item.get("token_sha256"),
            (
                secret_digest(str(retiring))
                if retiring
                else item.get("retiring_token_sha256")
            ),
            item.get("token_rotation_expires_at"),
        )
    raise IdentityAcceptanceError("target cluster is absent from registry")


def update_registry_token(
    entries: list[dict[str, Any]],
    cluster_id: str,
    token: str,
    *,
    retiring_token: str | None = None,
    rotation_expires_at: str | None = None,
) -> list[dict[str, Any]]:
    """Return the registry with one cluster's token, and rotation slot, replaced.

    Both plaintext fields are handed to the control plane, which stores only the
    digests; the driver never writes a token into evidence.
    """

    result = [dict(item) for item in entries]
    for item in result:
        if item.get("cluster_id") != cluster_id:
            continue
        item["token"] = token
        item.pop("token_sha256", None)
        item.pop("retiring_token", None)
        item.pop("retiring_token_sha256", None)
        item.pop("token_rotation_expires_at", None)
        if retiring_token is not None:
            item["retiring_token"] = retiring_token
            item["token_rotation_expires_at"] = rotation_expires_at
        return result
    raise IdentityAcceptanceError("target cluster is absent from registry")


def commands_not_misterminated(
    before_status: dict[str, str],
    after_status: dict[str, str | None],
) -> bool:
    """Whether every command open before the rotation survived it.

    An empty baseline used to pass vacuously (``all`` over nothing), so the
    check said "no command was misterminated" on a site that had no command to
    misterminate. It now needs at least one open command before the rotation.
    A command that ran to SUCCEEDED during the window was not interrupted; one
    that vanished (the status probe lists open commands only) or FAILED was.
    """

    if not before_status:
        return False
    return all(
        after_status.get(command_id) in {"PENDING", "WAITING", "LEASED", "SUCCEEDED"}
        for command_id in before_status
    )


def certificate_alert_checks(
    alert: dict[str, Any] | None,
    *,
    threshold_days: int,
) -> dict[str, Any]:
    """Whether the expiry alert the deploy ships is armed on a GPU node.

    The only certificate-expiry alerting in ``deploy/`` is the per-node
    ``gpu-fault-certificate-check.timer`` running
    ``check-control-plane-certificate`` with
    ``GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS`` from ``collector.env``; there
    is no PrometheusRule for it. ``threshold >= 30`` read the site file, not the
    node, so it passed with the timer disabled. ``alert`` is the host probe's
    reading; ``None`` means no node was given and the check is not evaluated.
    """

    if alert is None:
        return {"expiry_threshold_configured": "NOT_EVALUATED"}
    seconds = alert.get("min_validity_seconds")
    configured = (
        isinstance(seconds, int)
        and seconds >= threshold_days * 86400
        and alert.get("timer_enabled") is True
        and alert.get("timer_active") is True
    )
    return {"expiry_threshold_configured": configured}


def world_open_rules(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ingress permissions open to every IPv4 or IPv6 source."""

    broad = []
    for group in groups:
        for permission in group.get("IpPermissions") or []:
            sources = [
                item.get("CidrIp")
                for item in permission.get("IpRanges") or []
                if item.get("CidrIp") == "0.0.0.0/0"
            ] + [
                item.get("CidrIpv6")
                for item in permission.get("Ipv6Ranges") or []
                if item.get("CidrIpv6") == "::/0"
            ]
            if sources:
                broad.append(
                    {
                        "group_id": group["GroupId"],
                        "from_port": permission.get("FromPort"),
                        "to_port": permission.get("ToPort"),
                        "sources": sources,
                    }
                )
    return broad


HIGH_RISK_ROUTE_BUCKETS = {
    "/v1/runtime-profiles": "execution-token",
    "/v1/advisory-notifications/{notification_id}/send": "execution-token",
    "/v1/fleet/agents": "dual-credential",
}


def high_risk_route_errors(routes: list[dict[str, Any]]) -> list[str]:
    """The three routes whose bucket the catalog names, checked, not counted."""

    buckets = {str(item["path"]): item.get("bucket") for item in routes}
    errors = []
    for path, expected in HIGH_RISK_ROUTE_BUCKETS.items():
        actual = buckets.get(path)
        if actual is None:
            errors.append(f"{path} is not in the route inventory")
        elif actual != expected:
            errors.append(f"{path} is {actual}, expected {expected}")
    return errors


def node_key_digests(secret: dict[str, Any]) -> dict[str, str]:
    if not isinstance(secret.get("data"), dict) or not secret["data"]:
        raise IdentityAcceptanceError("node-action key data is missing")
    return {
        key: hashlib.sha256(base64.b64decode(value, validate=True)).hexdigest()
        for key, value in secret["data"].items()
    }


def master_reference_scan(resources: dict[str, Any]) -> dict[str, Any]:
    """Which GPU-plane resources reference the fleet-master Secret.

    The check is only meaningful if an installer resource was in the scan:
    the installer Job is transient, and a scan of an idle namespace finds no
    reference because it finds no installer. ``installer_resources`` names the
    Jobs/Pods that looked like an installer so the caller can tell "clean" from
    "nothing to look at".
    """

    hits: list[dict[str, str]] = []
    installer_resources: list[str] = []
    if not isinstance(resources.get("items"), list):
        raise IdentityAcceptanceError("installer inventory is incomplete")
    for item in resources["items"]:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        kind = str(item.get("kind") or "")
        labels = metadata.get("labels") or {}
        if "installer" in name or any("installer" in str(v) for v in labels.values()):
            installer_resources.append(f"{kind}/{name}")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            reference = value.get("secretKeyRef")
            if isinstance(reference, dict) and (
                reference.get("name") == INSTALLER_SECRET
            ):
                hits.append(
                    {
                        "path": path,
                        "secret": str(reference.get("name") or ""),
                        "key": str(reference.get("key") or ""),
                    }
                )
            secret_volume = value.get("secret")
            if isinstance(secret_volume, dict) and (
                secret_volume.get("secretName", secret_volume.get("name"))
                == INSTALLER_SECRET
            ):
                hits.append(
                    {
                        "path": path,
                        "secret": INSTALLER_SECRET,
                        "key": "",
                    }
                )
            env_from = value.get("secretRef")
            if isinstance(env_from, dict) and env_from.get("name") == INSTALLER_SECRET:
                hits.append({"path": path, "secret": INSTALLER_SECRET, "key": ""})
            for key, child in value.items():
                visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(resources, "")
    return {"hits": hits, "installer_resources": sorted(installer_resources)}


def heartbeat_advanced(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Node B kept heartbeating: its last_heartbeat_at moved forward."""

    try:
        earlier = datetime.fromisoformat(str(before["last_heartbeat_at"]))
        later = datetime.fromisoformat(str(after["last_heartbeat_at"]))
    except (KeyError, ValueError, TypeError):
        return False
    return later > earlier
