from __future__ import annotations

import copy
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


def test_batch_commit_accepts_only_prior_candidate_activation(
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
        cluster_id="gpu-c",
        batch_id="batch-a",
        verified_at=verified_at,
    )
    current = _snapshot(
        states={"gpu-a": "ACTIVE", "gpu-b": "ACTIVE", "gpu-c": "PENDING"},
        generation=3,
        state_sha256="d" * 64,
    )
    monkeypatch.setattr(evidence, "membership_runtime_snapshot", lambda _site: current)
    state = {
        "source_site_sha256": source.source_sha256,
        "source_site_non_membership_sha256": (
            evidence.site_non_membership_sha256(source.source)
        ),
        "completed_steps": ["SITE_UPDATED", "RELEASE_STATE_UPDATED"],
    }

    observed = evidence.validate_verified_membership(
        evidence=record,
        state=state,
        current_site=source,
        candidate_site=candidate,
        cluster_id="gpu-c",
        now=verified_at + timedelta(seconds=30),
    )

    assert observed["registry_generation"] == 3


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

    with pytest.raises(BootstrapError, match="expired"):
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
