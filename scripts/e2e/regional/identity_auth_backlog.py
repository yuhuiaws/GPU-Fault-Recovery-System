"""Run-owned no-action command backlog for the AUTH-008 isolation probe."""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from gpu_fault.admin.diagnostics import diagnostic_text
from scripts.e2e.regional import audit_auth_boundary
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.identity_acceptance_common import (
    ClusterTarget,
    IdentityAcceptanceError,
    IdentitySite,
)
from scripts.e2e.regional.identity_auth_checks import verdict
from scripts.e2e.regional.identity_auth_probes import (
    AUTH008_BACKLOG_PROBE,
    AUTH008_CLAIM_PROBE,
    AUTH008_EXECUTOR_IDENTITY_PROBE,
)


def auth008_claim(
    regional: Any,
    *,
    header_cluster: str,
    executor_id: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    payload = audit_auth_boundary.claim_payload(
        executor_id=executor_id,
        artifact_sha256=identity["artifact"],
        compatibility_digest=identity["compatibility"],
    )
    payload["executor_protocol_version"] = identity["protocol"]
    return cast(
        dict[str, Any],
        regional.executor_python(
            AUTH008_CLAIM_PROBE,
            json.dumps([header_cluster, payload]),
            timeout=60,
            attempts=1,
        ),
    )


def sanitized_error(exc: BaseException) -> str:
    """The exception text with credential-shaped tokens removed and bounded.

    ``RegionalFixtureError`` already carries redacted command output; this only
    guarantees that whatever text reaches ``auth008-details.json`` has bearer
    tokens, ``token=`` assignments, key material and control characters gone.
    """

    return diagnostic_text(str(exc), limit=1024)


def run_auth008(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    if primary.cluster_id == secondary.cluster_id:
        raise IdentityAcceptanceError("AUTH-008 requires distinct A/B clusters")
    regional_a, regional_b = site.regional(primary), site.regional(secondary)
    deployment = regional_a.evidence_identity()
    peer = regional_b.evidence_identity()
    if (
        not deployment.get("release_id")
        or peer.get("release_id") != deployment["release_id"]
    ):
        raise IdentityAcceptanceError("AUTH-008 A/B release identity is incomplete")
    # The CPU probe binds to these pins, not to release_id: the CPU Pods export
    # the pins (from gpu-fault-release-metadata) but not GPU_FAULT_RELEASE_ID.
    # Both fixtures read the same CPU namespace, so one read is the release.
    pins = regional_a.release_pins()
    identities = [
        regional.executor_python(AUTH008_EXECUTOR_IDENTITY_PROBE, attempts=1)
        for regional in (regional_a, regional_b)
    ]
    for target, identity in zip((primary, secondary), identities, strict=True):
        if (
            identity.get("cluster_id") != target.cluster_id
            or not identity.get("executor_id")
            or identity.get("probe_owner_configured") is not False
            or type(identity.get("protocol")) is not int
            or identity["protocol"] < 1
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key) or "")) is None
                for key in ("artifact", "compatibility")
            )
        ):
            raise IdentityAcceptanceError(
                "AUTH-008 executor identity is incomplete or unsafe"
            )
    nonce = secrets.token_hex(16)
    normal_id, positive_id = f"auth008-{nonce}/a", f"auth008-{nonce}/b"
    spoofed_id = identities[1]["executor_id"]
    receipt = {
        "nonce": nonce,
        "cluster_a": primary.cluster_id,
        "cluster_b": secondary.cluster_id,
        "release_id": deployment["release_id"],
        "required_pins": pins,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        "claimants": [normal_id, spoofed_id, positive_id],
    }
    receipt_path = case_dir / f"auth008-intent-{nonce}.json"
    write_json_atomic(receipt_path, receipt)
    result: dict[str, Any] = {
        "verdict": "FAIL",
        "checks": {},
        "entries": {},
        "cleanup_errors": [],
        "ownership_receipt": receipt_path.name,
    }

    def backlog(action: str) -> dict[str, Any]:
        return regional_a.cpu_python(
            AUTH008_BACKLOG_PROBE, json.dumps([action, receipt]), attempts=1
        )

    try:
        before = backlog("seed")
        result["candidate_before"] = before
        if before.get("pending_unleased") is not True:
            raise IdentityAcceptanceError("AUTH-008 has no eligible owned B candidate")
        for name, regional, header, executor_id, identity in (
            (
                "AUTH-008-A-normal",
                regional_a,
                primary.cluster_id,
                normal_id,
                identities[0],
            ),
            (
                "AUTH-008-A-fake-executor",
                regional_a,
                primary.cluster_id,
                spoofed_id,
                identities[0],
            ),
            (
                "AUTH-008-B-header-A-token",
                regional_a,
                secondary.cluster_id,
                normal_id,
                identities[0],
            ),
            (
                "AUTH-008-A-header-B-token",
                regional_b,
                primary.cluster_id,
                positive_id,
                identities[1],
            ),
        ):
            result["entries"][name] = auth008_claim(
                regional,
                header_cluster=header,
                executor_id=executor_id,
                identity=identity,
            )
        after = backlog("snapshot")
        result["candidate_after_negatives"] = after
        entry_errors = audit_auth_boundary.matrix_errors(
            result["entries"], cluster_a=primary.cluster_id
        )
        result["checks"].update(
            {
                "precise_claim_responses": not any(
                    name in entry_errors
                    for name in audit_auth_boundary.CASE_ENTRIES["GF-REGIONAL-AUTH-008"]
                ),
                "a_claims_lease_nothing": audit_auth_boundary.probe_claims_leased_nothing(
                    result["entries"]
                ),
                "owned_b_candidate_unchanged": (
                    after.get("pending_unleased") is True and before == after
                ),
            }
        )
        if verdict(result["checks"]) != "PASS":
            raise IdentityAcceptanceError("AUTH-008 negative claim isolation failed")
        positive = auth008_claim(
            regional_b,
            header_cluster=secondary.cluster_id,
            executor_id=positive_id,
            identity=identities[1],
        )
        result["positive_b_claim"] = positive
        positive_state = backlog("snapshot")
        result["candidate_after_positive"] = positive_state
        result["checks"]["b_can_claim_exact_candidate"] = positive == {
            "status": 200,
            "body": {
                "commands": [
                    {
                        "command_id": before["command_id"],
                        "cluster_id": secondary.cluster_id,
                        "workflow_request_id": before["workflow_id"],
                        "incident_id": before["incident_id"],
                        "status": "LEASED",
                    }
                ]
            },
        } and (
            positive_state.get("command_id") == before["command_id"]
            and positive_state.get("status") == "LEASED"
            and positive_state.get("lease_owner") == positive_id
        )
        result["checks"]["release_unchanged"] = (
            regional_a.evidence_identity() == deployment
            and regional_b.evidence_identity() == peer
            and regional_a.release_pins() == pins
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = sanitized_error(exc)
        result["checks"]["probe_completed"] = False
    finally:
        try:
            result["cleanup"] = backlog("cleanup")
            result["checks"]["owned_records_retired"] = (
                result["cleanup"].get("retired") is True
            )
        except Exception as exc:
            result["cleanup_errors"].append(
                f"{type(exc).__name__}: {sanitized_error(exc)}"
            )
            result["checks"]["owned_records_retired"] = False
        result["verdict"] = verdict(result["checks"])
        result["limitations"] = [
            "A no-action probe-owner B command is cancelled and its isolated workflow "
            "terminalized; audit records are retained. No physical adapter executes.",
            "Each request uses only its origin cluster's deployed token and CA; "
            "no registry, allowlist, workload or node configuration is changed.",
        ]
        write_json_atomic(case_dir / "auth008-details.json", result)
    return result
