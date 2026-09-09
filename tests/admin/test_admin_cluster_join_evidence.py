from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


def _candidate_sites(tmp_path: Path):
    source_path = site_file(tmp_path)
    document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    base = document["spec"]["clusters"][0]
    for suffix in ("b", "c"):
        token = tmp_path / "secure" / f"token-{suffix}"
        token.write_text(suffix * 64, encoding="utf-8")
        token.chmod(0o600)
        cluster = copy.deepcopy(base)
        cluster.update(
            {
                "clusterId": f"gpu-{suffix}",
                "context": f"gpu-{suffix}-context",
                "hyperpodClusterName": f"hp-gpu-{suffix}",
                "eksClusterArn": (
                    f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{suffix}"
                ),
                "tokenFile": str(token),
            }
        )
        document["spec"]["clusters"].append(cluster)
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    candidate_path.chmod(0o600)
    return load_site(source_path), load_site(candidate_path)


def _snapshot(
    *, states: dict[str, str], generation: int = 2, state_sha256: str = "a" * 64
) -> dict:
    return {
        "live_release_state_sha256": state_sha256,
        "live_release_identity_sha256": "b" * 64,
        "registry_generation": generation,
        "registry_content_sha256": f"{generation:x}".rjust(64, "c"),
        "registry_cluster_states": states,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def test_verified_membership_evidence_rejects_runtime_drift() -> None:
    before = _snapshot(states={"gpu-a": "ACTIVE", "gpu-b": "PENDING"})
    after = {**before, "registry_generation": 3}

    with pytest.raises(BootstrapError, match="drifted during"):
        evidence.build_verified_membership_evidence(
            before,
            after,
            candidate_site_sha256="d" * 64,
            source_site_sha256="e" * 64,
            source_site_non_membership_sha256="f" * 64,
            candidate_cluster_ids=["gpu-a", "gpu-b"],
            cluster_id="gpu-b",
        )


def test_commit_validation_checks_the_evidence_without_a_live_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A join takes three runtime snapshots: before and after the verify, and
    the final one after activation. The commit gate between them is a check of
    the recorded evidence against the transaction files, not a fourth and fifth
    read of the control plane.
    """

    source, candidate = _candidate_sites(tmp_path)
    baseline = _snapshot(
        states={"gpu-a": "ACTIVE", "gpu-b": "PENDING", "gpu-c": "PENDING"}
    )
    verified_at = datetime.now(timezone.utc)
    record = evidence.build_verified_membership_evidence(
        baseline,
        baseline,
        candidate_site_sha256=candidate.source_sha256,
        source_site_sha256=source.source_sha256,
        source_site_non_membership_sha256=evidence.site_non_membership_sha256(
            source.source
        ),
        candidate_cluster_ids=["gpu-a", "gpu-b", "gpu-c"],
        cluster_id="gpu-c",
        batch_id="batch-a",
        verified_at=verified_at,
    )
    monkeypatch.setattr(
        evidence,
        "membership_runtime_snapshot",
        lambda _site: pytest.fail("the commit gate re-read the live runtime"),
    )
    state = {
        "source_site_sha256": source.source_sha256,
        "source_site_non_membership_sha256": (
            evidence.site_non_membership_sha256(source.source)
        ),
        "completed_steps": ["SITE_UPDATED", "RELEASE_STATE_UPDATED"],
    }

    evidence.validate_verified_membership(
        evidence=record,
        state=state,
        current_site=source,
        candidate_site=candidate,
        cluster_id="gpu-c",
        now=verified_at + timedelta(seconds=30),
    )

    with pytest.raises(BootstrapError, match="cluster set drifted"):
        evidence.validate_verified_membership(
            evidence={**record, "candidate_cluster_ids": ["gpu-a", "gpu-c"]},
            state=state,
            current_site=source,
            candidate_site=candidate,
            cluster_id="gpu-c",
            now=verified_at + timedelta(seconds=30),
        )


def test_stale_verification_is_cleared_for_a_re_verify(tmp_path: Path) -> None:
    """Expired evidence means "verify again", never "roll the data plane back"."""

    verified_at = datetime.now(timezone.utc)
    record = {"verified_at": verified_at.isoformat()}
    assert not evidence.verification_is_stale(
        record, now=verified_at + timedelta(seconds=30)
    ), "evidence inside the window was treated as stale"
    assert evidence.verification_is_stale(
        record,
        now=verified_at + timedelta(seconds=evidence.VERIFICATION_MAX_AGE_SECONDS + 1),
    ), "evidence past the window was treated as fresh"
    assert evidence.verification_is_stale({}), "missing evidence was treated as fresh"

    state_path = tmp_path / "state.json"
    state = {
        "completed_steps": ["JOINED", "VERIFIED"],
        "evidence": {"JOINED": {}, "VERIFIED": record},
    }
    evidence.clear_verified_step(state_path, state)

    assert state["completed_steps"] == ["JOINED"]
    assert "VERIFIED" not in state["evidence"]
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written["completed_steps"] == ["JOINED"], "the cleared step was not saved"


def test_join_verification_evidence_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, candidate = _candidate_sites(tmp_path)
    baseline = _snapshot(
        states={"gpu-a": "ACTIVE", "gpu-b": "PENDING", "gpu-c": "PENDING"}
    )
    verified_at = datetime.now(timezone.utc)
    record = evidence.build_verified_membership_evidence(
        baseline,
        baseline,
        candidate_site_sha256=candidate.source_sha256,
        source_site_sha256=source.source_sha256,
        source_site_non_membership_sha256=evidence.site_non_membership_sha256(
            source.source
        ),
        candidate_cluster_ids=["gpu-a", "gpu-b", "gpu-c"],
        cluster_id="gpu-b",
        verified_at=verified_at,
    )
    monkeypatch.setattr(evidence, "membership_runtime_snapshot", lambda _site: baseline)

    with pytest.raises(evidence.JoinVerificationExpired, match="expired"):
        evidence.validate_verified_membership(
            evidence=record,
            state={
                "source_site_sha256": source.source_sha256,
                "source_site_non_membership_sha256": (
                    evidence.site_non_membership_sha256(source.source)
                ),
                "completed_steps": [],
            },
            current_site=source,
            candidate_site=candidate,
            cluster_id="gpu-b",
            now=verified_at
            + timedelta(seconds=evidence.VERIFICATION_MAX_AGE_SECONDS + 1),
        )


@pytest.mark.parametrize(
    "cluster_id", ["with space", "..", "x" * 200, "", "tab\tinside"]
)
def test_a_malformed_joined_cluster_id_is_rejected(cluster_id: str) -> None:
    """M-13: the joined cluster id decides which cluster is activated.

    It should always arrive already validated from the site document, so a value
    that does not match the anchored identifier shape is refused before it keys
    the registry-state comparison the activation boundary rests on.
    """

    baseline = _snapshot(states={"gpu-a": "ACTIVE", "gpu-b": "PENDING"})

    with pytest.raises(BootstrapError, match="cluster identity is malformed"):
        evidence.build_verified_membership_evidence(
            baseline,
            baseline,
            candidate_site_sha256="d" * 64,
            source_site_sha256="e" * 64,
            source_site_non_membership_sha256="f" * 64,
            candidate_cluster_ids=["gpu-a", "gpu-b"],
            cluster_id=cluster_id,
        )
