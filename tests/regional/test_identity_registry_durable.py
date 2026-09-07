"""The identity runners publish registry changes as durable revisions.

Once a control plane has published a registry revision, ``clusters.json`` is a
bootstrap copy nothing reads. AUTH-016 edited it, rolled the control plane and
watched the new token get 403 (live, 2026-09-07). The revision API stores
digests only, so plaintext ``token``/``retiring_token`` are digested here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.regional import cluster_token_sha256
from scripts.e2e.regional.identity_acceptance_common import durable_registry_payload


def _entry(**overrides: object) -> dict[str, object]:
    return {
        "cluster_id": "cluster-a",
        "region": "us-west-2",
        "hyperpod_cluster_name": "hp-a",
        "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/a",
        "token_sha256": "a" * 64,
        "retiring_token_sha256": None,
        "token_rotation_expires_at": None,
        "enabled": True,
        "allowed_namespaces": ["training"],
        "updated_at": "2026-09-02T05:20:31Z",
        **overrides,
    }


def test_plaintext_tokens_become_digests_and_refresh_updated_at() -> None:
    now = datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc)
    rotated = _entry(
        token="n" * 48,
        retiring_token="o" * 48,
        token_rotation_expires_at="2026-09-07T06:30:00Z",
    )
    rotated.pop("token_sha256")

    (payload,) = durable_registry_payload([rotated], now=now)

    assert "token" not in payload and "retiring_token" not in payload, (
        "plaintext must never reach the API body"
    )
    assert payload["token_sha256"] == cluster_token_sha256("n" * 48)
    assert payload["retiring_token_sha256"] == cluster_token_sha256("o" * 48)
    assert payload["token_rotation_expires_at"] == "2026-09-07T06:30:00Z"
    # The 7-day rotation window is measured from updated_at.
    assert payload["updated_at"] == "2026-09-07T06:00:00Z"


def test_entries_without_plaintext_pass_through_untouched() -> None:
    original = _entry()
    (payload,) = durable_registry_payload([original])
    assert payload == original, "a restore must republish exactly what was read"
    assert payload is not original, "the caller's list must not be mutated"
