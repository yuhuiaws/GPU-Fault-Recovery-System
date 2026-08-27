from __future__ import annotations

import json
import os
from pathlib import Path
import re
import ssl
from typing import Any
import urllib.error
import urllib.request

Response = tuple[int, dict[str, Any]]


def _post(payload: dict, *, token: str) -> Response:
    request = urllib.request.Request(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
        + "/v1/regional/executors/readiness",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
        },
        method="POST",
    )
    context = ssl.create_default_context(
        cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
    )
    try:
        with urllib.request.urlopen(
            request,
            context=context,
            timeout=20,
        ) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _reasons(payload: dict[str, Any]) -> list[str]:
    reasons = payload.get("reasons")
    assert isinstance(reasons, list), payload
    assert all(isinstance(reason, str) for reason in reasons), payload
    return reasons


def _evidence(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "ready": payload.get("ready"),
        "detail": payload.get("detail"),
        "reasons": payload.get("reasons"),
        "execution_owners": payload.get("execution_owners"),
        "unsupported_execution_owners": payload.get("unsupported_execution_owners"),
        "executor_artifact_sha256": payload.get("executor_artifact_sha256"),
        "last_successful_claim_age_seconds": payload.get(
            "last_successful_claim_age_seconds"
        ),
    }


def validate_readiness_matrix(
    *,
    valid: Response,
    wrong_token: Response,
    wrong_pin: Response,
    no_owner: Response,
    stale: Response,
    wrong_artifact: str,
    stale_age_seconds: int,
) -> dict[str, dict[str, Any]]:
    valid_status, valid_payload = valid
    assert valid_status == 200 and valid_payload.get("ready") is True, valid
    assert _reasons(valid_payload) == [], valid_payload
    assert valid_payload.get("unsupported_execution_owners") == [], valid_payload

    wrong_token_status, wrong_token_payload = wrong_token
    assert wrong_token_status == 403, wrong_token
    assert wrong_token_payload.get("detail") == (
        "regional cluster authentication failed"
    ), wrong_token_payload

    wrong_pin_status, wrong_pin_payload = wrong_pin
    assert wrong_pin_status == 503 and wrong_pin_payload.get("ready") is False, (
        wrong_pin
    )
    wrong_pin_reasons = _reasons(wrong_pin_payload)
    assert len(wrong_pin_reasons) == 1, wrong_pin_payload
    assert wrong_pin_reasons[0].startswith(
        "regional executor artifact mismatch: expected "
    ), wrong_pin_payload
    assert wrong_pin_reasons[0].endswith(f", got {wrong_artifact}"), wrong_pin_payload
    assert wrong_pin_payload.get("executor_artifact_sha256") == wrong_artifact, (
        wrong_pin_payload
    )

    no_owner_status, no_owner_payload = no_owner
    assert no_owner_status == 503 and no_owner_payload.get("ready") is False, no_owner
    no_owner_reasons = _reasons(no_owner_payload)
    expected_no_owner = (
        "executor advertised no execution owners, so it can claim nothing"
    )
    assert no_owner_reasons[0] == expected_no_owner, no_owner_payload
    assert all(
        reason == expected_no_owner
        or reason.startswith(
            "open backlog needs execution owners this executor does not advertise: "
        )
        for reason in no_owner_reasons
    ), no_owner_payload
    assert no_owner_payload.get("execution_owners") == [], no_owner_payload

    stale_status, stale_payload = stale
    assert stale_status == 503 and stale_payload.get("ready") is False, stale
    stale_reasons = _reasons(stale_payload)
    assert len(stale_reasons) == 1, stale_payload
    assert re.fullmatch(
        rf"last successful claim was {stale_age_seconds}s ago \(limit \d+s\)",
        stale_reasons[0],
    ), stale_payload
    assert float(stale_payload.get("last_successful_claim_age_seconds")) == float(
        stale_age_seconds
    ), stale_payload

    return {
        "valid": _evidence(*valid),
        "wrong_token": _evidence(*wrong_token),
        "wrong_pin": _evidence(*wrong_pin),
        "no_owner": _evidence(*no_owner),
        "stale_claim": _evidence(*stale),
    }


def main() -> None:
    token = os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"]
    artifact = os.environ["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"]
    digest = os.environ["GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"]
    state_path = Path(
        os.environ.get(
            "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
            "/tmp/executor-claim-state.json",
        )
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    owners = list(state["execution_owners"])
    base = {
        "executor_id": "regional-readiness-audit",
        "executor_protocol_version": 2,
        "executor_artifact_sha256": artifact,
        "executor_compatibility_digest": digest,
        "execution_owners": owners,
        "last_successful_claim_age_seconds": 0,
    }
    valid_status, valid = _post(base, token=token)
    wrong_token = token[:-1] + ("a" if token[-1] != "a" else "b")
    wrong_token_result = _post(base, token=wrong_token)
    wrong_artifact = "0" * 64 if artifact != "0" * 64 else "1" * 64
    wrong_pin_status, wrong_pin = _post(
        {
            **base,
            "executor_artifact_sha256": wrong_artifact,
        },
        token=token,
    )
    no_owner_status, no_owner = _post(
        {**base, "execution_owners": []},
        token=token,
    )
    stale_age_seconds = 86400
    stale_status, stale = _post(
        {**base, "last_successful_claim_age_seconds": stale_age_seconds},
        token=token,
    )

    evidence = validate_readiness_matrix(
        valid=(valid_status, valid),
        wrong_token=wrong_token_result,
        wrong_pin=(wrong_pin_status, wrong_pin),
        no_owner=(no_owner_status, no_owner),
        stale=(stale_status, stale),
        wrong_artifact=wrong_artifact,
        stale_age_seconds=stale_age_seconds,
    )
    print(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
