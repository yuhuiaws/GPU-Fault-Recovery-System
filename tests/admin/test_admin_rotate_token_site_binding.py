"""The rotation journal binds the site facts the rotation reads, not site.yaml's bytes.

A release rewrites ``spec.repositoryRoot``, ``spec.release.manifest`` and
``spec.runtimeProfile.source`` in ``site.yaml``; none of them is read by the
rotation, so a journal bound to the file's raw sha256 refused to resume (and
to roll back) after a deploy ran between two steps. The journal now also
records a canonical projection of the rotation-relevant facts and resumes on
that; a journal from before the projection is rebound only when live state
proves the rotation already converged onto the recorded new token.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpu_fault.admin import rotate_token as module
from gpu_fault.admin import rotate_token_binding as binding
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin.test_admin_rotate_token import OLD_TOKEN, Harness, _site

REWRITTEN_SITE_SHA256 = "9" * 64


def _pause_before_acceptance(harness: Harness) -> dict:
    harness.acceptance_error = BootstrapError("quiet proof unavailable")
    with pytest.raises(BootstrapError, match="quiet proof"):
        harness.rotate()
    harness.acceptance_error = None
    return harness.state


def _stop_at_final_publish(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> dict:
    publish = module.publish_current_revision

    def fail_final(*args, **kwargs):
        raise ReleaseError("final registry publish interrupted")

    monkeypatch.setattr(module, "publish_current_revision", fail_final)
    with pytest.raises(ReleaseError, match="final registry publish"):
        harness.rotate()
    monkeypatch.setattr(module, "publish_current_revision", publish)
    state = harness.state
    assert module.STEP_CONTROL_PLANE_ROLLED in state["steps"]
    assert module.STEP_RETIRING_DROPPED not in state["steps"]
    return state


def _release_only_rewrite(harness: Harness) -> None:
    """What a deploy does to site.yaml: bytes change, rotation facts do not."""

    harness.site.source_sha256 = REWRITTEN_SITE_SHA256
    harness.site.release_config["runtime_profile"] = {
        "version": "hyperpod-v2",
        "source": "/snapshots/new/runtime-profile.yaml",
    }
    harness.site.release_config["release"] = {"manifest": "/snapshots/new/release.json"}
    harness.site.repository_root = Path("/snapshots/new")


def _strip_binding(harness: Harness) -> None:
    """Turn the journal into one written before the site binding existed."""

    path = module.rotation_state_path(harness.site, "gpu-a")
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["site_binding_sha256"]
    del state["site_binding"]
    path.write_text(json.dumps(state), encoding="utf-8")


def _new_only_head(harness: Harness) -> list[dict]:
    """The durable head a deploy leaves after re-rendering the registry."""

    new_digest = harness.state["new_token_sha256"]
    return [
        {
            "cluster_id": "gpu-a",
            "token_sha256": new_digest,
            "retiring_token_sha256": None,
        },
        {
            "cluster_id": "gpu-b",
            "token_sha256": hashlib.sha256(b"b" * 64).hexdigest(),
            "retiring_token_sha256": None,
        },
    ]


def test_a_new_journal_records_the_rotation_relevant_site_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    state = _pause_before_acceptance(harness)

    projection = binding.rotation_site_binding(harness.site, "gpu-a")
    assert state["site_sha256"] == "c" * 64, "the raw digest stays for audit"
    assert state["site_binding"] == projection
    assert state["site_binding_sha256"] == binding.site_binding_sha256(projection)
    assert projection["site_name"] == "staging"
    assert projection["cluster_id"] == "gpu-a"
    assert projection["token_file"] == str(harness.token_file)
    assert projection["cpu_kubeconfig"] == harness.site.release_config["cpu_kubeconfig"]
    assert projection["namespace"] == "gpu-fault-system"
    assert projection["registry_secret"] == {
        "namespace": "gpu-fault-system",
        "name": "gpu-fault-regional-clusters",
    }
    assert [item["cluster_id"] for item in projection["clusters"]] == ["gpu-a", "gpu-b"]
    assert OLD_TOKEN not in json.dumps(state), "the journal never carries a token"
    assert state["new_token_sha256"] not in json.dumps(projection)


def test_the_binding_ignores_release_only_site_fields(tmp_path: Path) -> None:
    first = _site(tmp_path)
    before = binding.site_binding_sha256(binding.rotation_site_binding(first, "gpu-a"))
    first.release_config["runtime_profile"] = {"version": "other"}
    first.release_config["release"] = {"manifest": "/elsewhere/release.json"}
    first.repository_root = Path("/elsewhere")
    first.source_sha256 = REWRITTEN_SITE_SHA256
    unchanged = binding.site_binding_sha256(
        binding.rotation_site_binding(first, "gpu-a")
    )
    assert unchanged == before

    first.release_config["clusters"][1]["context"] = "gpu-b-other-context"
    changed = binding.site_binding_sha256(binding.rotation_site_binding(first, "gpu-a"))
    assert changed != before, "a cluster entry is a rotation-relevant fact"


def test_resume_proceeds_after_a_release_only_site_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    state = _pause_before_acceptance(harness)
    _release_only_rewrite(harness)
    before = len(harness.calls)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["rotation_id"] == state["rotation_id"]
    assert [call[0] for call in harness.calls[before:]] == [
        "acceptance",
        "write_registry",
        "restart_control_plane",
        "publish",
    ]
    final = harness.state
    assert final["site_sha256"] == "c" * 64, "the starting digest is kept for audit"
    assert "site_rebound" not in final, "a bound journal needs no rebind"
    assert final["warnings"] == []


def test_resume_fails_closed_when_a_rotation_relevant_fact_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause_before_acceptance(harness)
    _release_only_rewrite(harness)
    harness.site.release_config["cpu_kubeconfig"] = str(tmp_path / "other.kubeconfig")
    before = len(harness.calls)

    with pytest.raises(BootstrapError) as failure:
        harness.rotate()

    message = str(failure.value)
    assert "rotate-token site changed since the rotation started" in message
    assert "cpu_kubeconfig" in message, "the refusal names the binding that differs"
    assert len(harness.calls) == before
    assert module.STATUS_IN_PROGRESS == harness.state["status"]


def test_a_legacy_journal_rebinds_when_live_state_holds_only_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observed defect: deploy republished the registry mid-finish."""

    harness = Harness(tmp_path, monkeypatch)
    state = _stop_at_final_publish(harness, monkeypatch)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    harness.durable = _new_only_head(harness)
    before = len(harness.calls)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["rotation_id"] == state["rotation_id"]
    assert [call[0] for call in harness.calls[before:]] == [], (
        "the head already dropped the retiring digest: nothing is republished"
    )
    final = harness.state
    rebound = final["site_rebound"]
    assert rebound["from_site_sha256"] == "c" * 64
    assert rebound["to_site_sha256"] == REWRITTEN_SITE_SHA256
    assert rebound["at"] > state["started_at"]
    evidence = rebound["evidence"]
    assert any(module.STEP_TOKEN_FILE_WRITTEN in item for item in evidence), evidence
    assert any("token file" in item for item in evidence), evidence
    assert any("durable registry head" in item for item in evidence), evidence
    projection = binding.rotation_site_binding(harness.site, "gpu-a")
    assert final["site_binding"] == projection
    assert final["site_binding_sha256"] == binding.site_binding_sha256(projection)
    dropped = final["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert dropped["retiring_token_dropped"] is True
    assert dropped["already_dropped"] is True
    serialized = json.dumps(final)
    assert OLD_TOKEN not in serialized
    assert harness.token_file.read_text(encoding="utf-8") not in serialized


def test_a_legacy_journal_rebinds_on_the_overlap_head_and_then_drops_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live shape: the head is still the rotation's own overlap revision
    (token new, retiring old), the Secret holds the new token only."""

    harness = Harness(tmp_path, monkeypatch)
    state = _stop_at_final_publish(harness, monkeypatch)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    head = harness.publishes[-1]["registrations"]
    target = [item for item in head if item["cluster_id"] == "gpu-a"][0]
    assert target["retiring_token_sha256"] == state["old_token_sha256"]
    before = len(harness.calls)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert [call[0] for call in harness.calls[before:]] == ["publish"], (
        "the retiring digest is still in the head: the final step drops it"
    )
    final = harness.state
    evidence = final["site_rebound"]["evidence"]
    assert any("retiring digest still present" in item for item in evidence), evidence
    assert final["site_rebound"]["from_site_sha256"] == "c" * 64
    assert final["site_binding_sha256"] == binding.site_binding_sha256(
        binding.rotation_site_binding(harness.site, "gpu-a")
    )
    dropped = final["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert dropped["already_dropped"] is False
    published = [
        item
        for item in harness.publishes[-1]["registrations"]
        if item["cluster_id"] == "gpu-a"
    ][0]
    assert published["token_sha256"] == state["new_token_sha256"]
    assert "retiring_token_sha256" not in published
    assert OLD_TOKEN not in json.dumps(final)


def test_a_legacy_journal_fails_closed_when_the_retiring_digest_is_not_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    state = _stop_at_final_publish(harness, monkeypatch)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    harness.durable = _new_only_head(harness)
    harness.durable[0]["retiring_token_sha256"] = "7" * 64
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="site changed since the rotation"):
        harness.rotate()

    assert len(harness.calls) == before
    after = harness.state
    assert "site_rebound" not in after
    assert "site_binding_sha256" not in after
    assert after["status"] == module.STATUS_IN_PROGRESS
    assert after["steps"].keys() == state["steps"].keys()


def test_a_legacy_journal_fails_closed_when_the_head_token_is_not_the_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    state = _stop_at_final_publish(harness, monkeypatch)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    harness.durable = _new_only_head(harness)
    harness.durable[0]["token_sha256"] = state["old_token_sha256"]
    harness.durable[0]["retiring_token_sha256"] = None
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="not the recorded new digest"):
        harness.rotate()

    assert len(harness.calls) == before
    assert "site_rebound" not in harness.state


def test_a_legacy_journal_fails_closed_when_the_token_file_is_not_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _stop_at_final_publish(harness, monkeypatch)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    harness.durable = _new_only_head(harness)
    harness.token_file.write_text("z" * 64, encoding="utf-8")
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="site changed since the rotation"):
        harness.rotate()

    assert len(harness.calls) == before
    assert "site_rebound" not in harness.state


def test_a_legacy_journal_before_the_token_file_write_is_never_rebound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause_before_acceptance(harness)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    harness.durable = _new_only_head(harness)
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="site changed since the rotation"):
        harness.rotate()

    assert len(harness.calls) == before
    assert "site_rebound" not in harness.state


def test_rollback_never_rebinds_a_legacy_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause_before_acceptance(harness)
    _strip_binding(harness)
    _release_only_rewrite(harness)
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="site changed since the rotation"):
        harness.rotate(rollback=True)

    assert len(harness.calls) == before
    after = harness.state
    assert "site_rebound" not in after
    assert "rollback_started_at" not in after


def test_rollback_of_a_bound_journal_survives_a_release_only_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause_before_acceptance(harness)
    _release_only_rewrite(harness)

    summary = harness.rotate(rollback=True)

    assert summary["status"] == module.STATUS_ROLLED_BACK
    assert harness.token_file.read_text(encoding="utf-8") == OLD_TOKEN


def test_the_final_step_is_idempotent_once_the_retiring_digest_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _stop_at_final_publish(harness, monkeypatch)
    harness.durable = _new_only_head(harness)
    before = len(harness.calls)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert [call[0] for call in harness.calls[before:]] == []
    evidence = harness.state["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert evidence["retiring_token_dropped"] is True
    assert evidence["already_dropped"] is True
    assert evidence["generation"] == len(harness.publishes)


def test_the_final_step_still_publishes_while_the_head_carries_the_retiring_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _stop_at_final_publish(harness, monkeypatch)
    before = len(harness.calls)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert [call[0] for call in harness.calls[before:]] == ["publish"]
    evidence = harness.state["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert evidence["already_dropped"] is False
    published = harness.publishes[-1]["registrations"]
    target = [item for item in published if item["cluster_id"] == "gpu-a"][0]
    assert target["token_sha256"] == harness.state["new_token_sha256"]
    assert "retiring_token_sha256" not in target


def test_registry_head_evidence_reports_why_the_head_is_not_converged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    digest = "1" * 64
    harness.durable = [
        {"cluster_id": "gpu-a", "token_sha256": digest, "retiring_token_sha256": None}
    ]
    assert binding.registry_head_evidence(harness.release, "gpu-a", digest) == {
        "converged": True,
        "generation": 0,
        "content_sha256": "f" * 64,
        "token_sha256": digest,
        "retiring_token_present": False,
    }
    other = binding.registry_head_evidence(harness.release, "gpu-a", "2" * 64)
    assert other["converged"] is False
    assert "not the recorded new digest" in other["reason"]
    missing = binding.registry_head_evidence(harness.release, "gpu-z", digest)
    assert missing["converged"] is False
    assert "gpu-z" in missing["reason"]
    harness.durable[0]["retiring_token_sha256"] = "3" * 64
    overlap = binding.registry_head_evidence(harness.release, "gpu-a", digest)
    assert overlap["converged"] is False, "no candidate: the strict drop check"
    assert "Secret" in overlap["reason"]
    accepted = binding.registry_head_evidence(
        harness.release, "gpu-a", digest, retiring_digest="3" * 64
    )
    assert accepted["converged"] is True
    assert accepted["retiring_token_present"] is True
    wrong = binding.registry_head_evidence(
        harness.release, "gpu-a", digest, retiring_digest="4" * 64
    )
    assert wrong["converged"] is False
    assert "Secret" in wrong["reason"]


def _durable_api(monkeypatch: pytest.MonkeyPatch, head: list[dict]) -> None:
    """Stand in for the registry API and the Secret behind ``durable_registrations``:
    the head carries full digests, the API redacts them, the Secret knows one token."""

    from gpu_fault.regional import (
        RegionalClusterRegistration,
        regional_registry_content_sha256,
    )
    from gpu_fault_release import regional_release_online_registry as registry

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
    redacted = []
    for item in models:
        value = item.model_dump(mode="json")
        for field in ("token_sha256", "retiring_token_sha256"):
            digest = value.pop(field) or ""
            value[f"{field}_present"] = bool(digest)
            value[f"{field}_length"] = len(digest)
        redacted.append(value)

    def request(release, method, path, payload=None, *, response_list=False):
        assert method == "GET", method
        return redacted if path == "/v1/regional/clusters" else status

    secret = [
        {
            **item.model_dump(
                mode="json",
                exclude={
                    "lifecycle_state",
                    "created_at",
                    "updated_at",
                    "token_sha256",
                    "retiring_token_sha256",
                    "token_rotation_expires_at",
                },
            ),
            "token_sha256": item.token_sha256,
        }
        for item in models
    ]
    monkeypatch.setattr(registry, "_request", request)
    monkeypatch.setattr(registry, "current_registrations", lambda release, o: secret)


def test_durable_registrations_restores_a_retiring_digest_only_from_a_proven_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault_release import regional_release_online_registry as registry

    new_digest, old_digest = "1" * 64, "2" * 64
    _durable_api(
        monkeypatch,
        [
            {
                "cluster_id": "gpu-a",
                "region": "us-west-2",
                "hyperpod_cluster_name": "hp-a",
                "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
                "token_sha256": new_digest,
                "retiring_token_sha256": old_digest,
                "created_at": "2999-01-01T00:00:00+00:00",
                "updated_at": "2999-01-01T00:00:00+00:00",
                "token_rotation_expires_at": "2999-01-02T00:00:00+00:00",
                "allowed_namespaces": ["team-a"],
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/8"],
            }
        ],
    )

    with pytest.raises(ReleaseError, match="credential identity differs"):
        registry.durable_registrations(object())
    with pytest.raises(ReleaseError, match="snapshot identity differs"):
        registry.durable_registrations(object(), retiring_digests={"gpu-a": "9" * 64})

    status, durable = registry.durable_registrations(
        object(), retiring_digests={"gpu-a": old_digest}
    )

    assert status.generation == 7
    assert [item.cluster_id for item in durable] == ["gpu-a"]
    assert durable[0].token_sha256 == new_digest
    assert durable[0].retiring_token_sha256 == old_digest
