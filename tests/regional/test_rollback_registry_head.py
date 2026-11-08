"""The rollback's comparison of the durable registry head with the Secret.

``registry_head_drift`` judges the head the control plane serves against the
CPU Secret the way ``durable_registrations`` does and names each cluster that
differs; ``verify_registry_head`` turns a difference into the rollback's
fail-closed reason. The engine cases are in ``test_manual_rollback_engine``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.rotate_token_journal import rotation_journal_path
from gpu_fault.regional import (
    RegionalClusterRegistration,
    regional_registry_content_sha256,
)
from gpu_fault_release import regional_release_online_registry as ONLINE
from gpu_fault_release import regional_release_registry as REGISTRY
from gpu_fault_release import regional_release_registry_head as HEAD
from gpu_fault_release.regional_release_config import ReleaseError

TOKEN, OLD_TOKEN = "1" * 64, "2" * 64


def _entry(cluster_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "region": "us-west-2",
        "hyperpod_cluster_name": f"hp-{cluster_id}",
        "eks_cluster_arn": f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}",
        "token_sha256": TOKEN,
        "allowed_namespaces": ["team-a"],
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/8"],
        **fields,
    }


def _site(
    monkeypatch: pytest.MonkeyPatch, head: list[dict[str, Any]], secret: list[dict]
) -> None:
    """The registry API over ``head`` (digests redacted, as served) and the
    Secret's ``secret`` entries, behind ``registry_head_drift``."""

    models = [RegionalClusterRegistration.model_validate(item) for item in head]
    status = {
        "generation": 7,
        "content_sha256": regional_registry_content_sha256(models),
        "cluster_states": {item.cluster_id: item.lifecycle_state for item in models},
        "required_member_ids": [],
        "acked_member_ids": [],
        "missing_member_ids": [],
        "active_member_ids": [],
        "members": [],
        "converged": True,
    }
    rows = []
    for item in models:
        value = item.model_dump(mode="json")
        for field in ("token_sha256", "retiring_token_sha256"):
            digest = value.pop(field) or ""
            value[f"{field}_present"] = bool(digest)
            value[f"{field}_length"] = len(digest)
        rows.append(value)

    def request(_release: Any, method: str, path: str, *_a: Any, **_k: Any) -> Any:
        assert method == "GET", method
        return rows if path == "/v1/regional/clusters" else status

    monkeypatch.setattr(ONLINE, "_request", request)
    monkeypatch.setattr(ONLINE, "current_registrations", lambda _r, _o: secret)


def test_the_head_matches_a_secret_that_registers_the_same_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = [_entry("gpu-a", runtime_profile_version="v1"), _entry("gpu-eks")]
    _site(monkeypatch, registry, registry)

    drift = ONLINE.registry_head_drift(object())

    assert drift == {
        "matches": True,
        "generation": 7,
        "content_sha256": drift["content_sha256"],
    }


def test_each_cluster_that_differs_is_named_and_one_that_agrees_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpu-a's pins differ, gpu-c is only in the Secret; gpu-eks -- parked or
    not, it has an entry -- is compared and agrees."""

    _site(
        monkeypatch,
        [_entry("gpu-a", runtime_profile_version="v2"), _entry("gpu-eks")],
        [
            _entry("gpu-a", runtime_profile_version="v1"),
            _entry("gpu-c"),
            _entry("gpu-eks"),
        ],
    )

    drift = ONLINE.registry_head_drift(object())

    assert drift["matches"] is False
    assert drift["clusters"] == ["gpu-a", "gpu-c"]
    assert drift["reason"] == "regional registry membership differs from its Secret"


def test_a_keep_window_head_matches_with_the_journal_s_old_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``--keep-window`` rotation leaves the old digest in the head as
    retiring; the Secret never carries it. Only the journal's digest -- proven
    by the revision's content digest -- reconstructs the head."""

    head = [
        _entry(
            "gpu-a",
            retiring_token_sha256=OLD_TOKEN,
            created_at="2999-01-01T00:00:00+00:00",
            updated_at="2999-01-01T00:00:00+00:00",
            token_rotation_expires_at="2999-01-02T00:00:00+00:00",
        )
    ]
    _site(monkeypatch, head, [_entry("gpu-a")])
    journal = rotation_journal_path(tmp_path, "gpu-a")
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"old_token_sha256": OLD_TOKEN}), encoding="utf-8")
    release = SimpleNamespace(
        config=SimpleNamespace(
            admin_state_dir=tmp_path, clusters=(SimpleNamespace(cluster_id="gpu-a"),)
        )
    )

    unknown = ONLINE.registry_head_drift(release)
    known = ONLINE.registry_head_drift(
        release, retiring_digests=HEAD.rotation_retiring_digests(release)
    )

    assert (unknown["matches"], unknown["clusters"]) == (False, ["gpu-a"])
    assert known["matches"] is True
    assert (
        HEAD.rotation_retiring_digests(
            SimpleNamespace(config=SimpleNamespace(admin_state_dir=None))
        )
        == {}
    )


def test_a_token_digest_the_redacted_head_cannot_attribute_names_no_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _site(monkeypatch, [_entry("gpu-a")], [_entry("gpu-a", token_sha256="3" * 64)])

    drift = ONLINE.registry_head_drift(object())

    assert (drift["matches"], drift["clusters"]) == (False, [])
    assert drift["reason"] == "regional registry snapshot identity differs"


class _Release:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.runner = SimpleNamespace(dry_run=dry_run)
        self.config = SimpleNamespace(admin_state_dir=Path("/srv/site"), clusters=())
        self.calls: list[str] = []

    def _publish_restored_registry(self) -> None:
        self.calls.append("publish")

    def _complete_registry_restore(self) -> None:
        self.calls.append("complete")


def test_a_dry_run_compares_nothing() -> None:
    release = _Release(dry_run=True)

    assert HEAD.verify_registry_head(release) == {}
    assert release.calls == []


@pytest.mark.parametrize("pending", [False, True])
def test_a_head_that_is_the_secret_passes_after_a_pending_restore_is_published(
    monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    release = _Release()
    monkeypatch.setattr(HEAD, "registry_restore_pending", lambda _r: pending)
    monkeypatch.setattr(
        HEAD,
        "registry_head_drift",
        lambda _r, **_k: {"matches": True, "generation": 9, "content_sha256": "c" * 64},
    )

    evidence = HEAD.verify_registry_head(release)

    assert release.calls == (["publish", "complete"] if pending else [])
    assert evidence == {
        "registry_head": {
            "matches_secret": True,
            "generation": 9,
            "content_sha256": "c" * 64,
            "republished": pending,
        }
    }


@pytest.mark.parametrize(
    ("clusters", "mark_error", "expected"),
    [
        pytest.param(
            ["gpu-a"],
            None,
            "Secret on gpu-a (why); the Secret keeps the restored registry as "
            "previous-clusters.json until a publish converges",
            id="marked",
        ),
        pytest.param(
            [],
            None,
            "on a credential digest the redacted head attributes to no cluster",
            id="unattributed",
        ),
        pytest.param(
            ["gpu-a"],
            "write failed",
            "the Secret could not keep the restore pending (write failed)",
            id="mark-failed",
        ),
    ],
)
def test_a_head_that_is_not_the_secret_fails_closed_with_the_exit(
    monkeypatch: pytest.MonkeyPatch,
    clusters: list[str],
    mark_error: str | None,
    expected: str,
) -> None:
    release = _Release()
    marked: list[bool] = []

    def keep(_release: Any) -> None:
        marked.append(True)
        if mark_error is not None:
            raise ReleaseError(mark_error)

    monkeypatch.setattr(HEAD, "registry_restore_pending", lambda _r: False)
    monkeypatch.setattr(
        HEAD,
        "registry_head_drift",
        lambda _r, **_k: {"matches": False, "reason": "why", "clusters": clusters},
    )
    monkeypatch.setattr(HEAD, "keep_registry_restore_pending", keep)

    with pytest.raises(ReleaseError) as raised:
        HEAD.verify_registry_head(release)

    reason = str(raised.value)
    assert reason.startswith("rollback registry head mismatch: "), reason
    assert expected in reason
    assert "rerun `gpu-fault-admin deploy --state-dir /srv/site`" in reason
    assert marked == [True]
    assert release.calls == []


def test_a_staged_backup_the_cpu_stages_did_not_restore_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, staged = [_entry("gpu-a")], [_entry("gpu-b")]
    monkeypatch.setattr(REGISTRY, "registry_payloads", lambda _r: (staged, registry))

    with pytest.raises(ReleaseError, match="CPU stages did not restore"):
        REGISTRY.registry_restore_pending(object())  # type: ignore[arg-type]

    monkeypatch.setattr(REGISTRY, "registry_payloads", lambda _r: (registry, registry))
    assert REGISTRY.registry_restore_pending(object()) is True  # type: ignore[arg-type]
    monkeypatch.setattr(REGISTRY, "registry_payloads", lambda _r: (registry, None))
    assert REGISTRY.registry_restore_pending(object()) is False  # type: ignore[arg-type]
