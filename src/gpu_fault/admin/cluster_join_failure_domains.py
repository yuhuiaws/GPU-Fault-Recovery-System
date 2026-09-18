"""One journaled publication for a verified batch, before any ACTIVE transition."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin import failure_domain_map as maps
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import (
    RUNTIME_MEMBERSHIP_FIELDS,
    join_activation_is_irreversible,
    membership_runtime_snapshot,
    validate_verified_membership,
)
from gpu_fault.admin.cluster_join_state import complete_step, step_done
from gpu_fault.admin.cluster_join_types import (
    JoinAttempt,
    JoinClusterRequest,
    JoinExecution,
)
from gpu_fault.failure_domains import failure_domain_map_sha256

PUBLICATION_STARTED = "FAILURE_DOMAINS_STARTED"
PUBLICATION_READY = "FAILURE_DOMAINS_READY"


def _execution(attempt: JoinAttempt) -> JoinExecution:
    if attempt.execution is None:
        raise BootstrapError("failure-domain publication lacks a prepared join")
    return attempt.execution


def publish_batch_failure_domains(attempts: list[JoinAttempt]) -> None:
    if not attempts:
        return
    candidate = _execution(attempts[0]).candidate
    before = membership_runtime_snapshot(candidate)
    for attempt in attempts:
        execution = _execution(attempt)
        evidence = attempt.state["evidence"]["VERIFIED"]
        validate_verified_membership(
            evidence=evidence,
            state=attempt.state,
            current_site=attempt.request.site,
            candidate_site=execution.candidate,
            cluster_id=execution.cluster_id,
            check_runtime=False,
            allow_expired=join_activation_is_irreversible(attempt.state),
        )
        expected = evidence.get("post_verification_runtime") or evidence
        if execution.candidate.source_sha256 != candidate.source_sha256:
            raise BootstrapError("batch map candidates differ")
        if any(
            before.get(key) != expected.get(key) for key in RUNTIME_MEMBERSHIP_FIELDS
        ):
            # An already-started activation may have committed its own revision
            # before losing the ACK. Its existing final verifier handles that
            # transition, but a new publication still covers the full candidate.
            if not (
                join_activation_is_irreversible(attempt.state)
                and before["live_release_identity_sha256"]
                == expected["live_release_identity_sha256"]
                and before["registry_cluster_states"]
                == {
                    **expected["registry_cluster_states"],
                    execution.cluster_id: "ACTIVE",
                }
            ):
                raise BootstrapError("batch membership drifted before map publication")
    result = maps.build_failure_domain_map(candidate)
    for attempt in attempts:
        execution = _execution(attempt)
        if result.node_uids.get(execution.cluster_id) != execution.local.get(
            "node_uids"
        ):
            raise BootstrapError("batch node identities changed before map publication")
    manifest = maps.failure_domain_configmap(
        result.mapping, namespace=str(candidate.release_config["namespace"])
    )
    record: dict[str, Any] = {
        "publication_id": uuid.uuid4().hex,
        "candidate_site_sha256": candidate.source_sha256,
        "cluster_ids": sorted(result.mapping),
        "node_uids": result.node_uids,
        "map_sha256": failure_domain_map_sha256(manifest),
    }
    for attempt in attempts:
        complete_step(attempt.state_path, attempt.state, PUBLICATION_STARTED, record)
    published = maps.apply_failure_domain_map(candidate, prepared=result)
    if not published.worker_uid:
        raise BootstrapError("batch map publication lacks its control-worker identity")
    receipt = maps.verify_failure_domain_publication(
        candidate, digest=record["map_sha256"], worker_uid=published.worker_uid
    )
    after = membership_runtime_snapshot(candidate)
    if any(before.get(key) != after.get(key) for key in RUNTIME_MEMBERSHIP_FIELDS):
        raise BootstrapError("batch membership drifted during map publication")
    for attempt in attempts:
        complete_step(
            attempt.state_path,
            attempt.state,
            PUBLICATION_READY,
            {
                **record,
                **receipt,
                "attempt": attempt.state["attempt"],
                "converged_at": datetime.now(UTC).isoformat(),
            },
        )


def require_join_failure_domains(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    evidence = state["evidence"]
    verified = evidence["VERIFIED"]
    if not verified.get("batch_id") and PUBLICATION_STARTED not in evidence:
        return
    if not step_done(state, PUBLICATION_READY):
        publish_batch_failure_domains(
            [JoinAttempt(request, state_dir, state_path, state, execution=execution)]
        )
    record = evidence.get(PUBLICATION_READY)
    if (
        not isinstance(record, dict)
        or record.get("attempt") != state["attempt"]
        or record.get("candidate_site_sha256") != execution.candidate.source_sha256
        or record.get("cluster_ids") != verified["candidate_cluster_ids"]
        or (record.get("node_uids") or {}).get(execution.cluster_id)
        != execution.local.get("node_uids")
    ):
        raise BootstrapError("batch map publication is not bound to this verified join")
    maps.verify_failure_domain_publication(
        execution.candidate,
        digest=record["map_sha256"],
        worker_uid=record["worker_uid"],
        configmap_uid=record["configmap_uid"],
    )
