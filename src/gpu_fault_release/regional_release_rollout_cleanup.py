"""Driving the candidate's fleet rollout records to rest during a rollback.

These two steps are the rollback's only contact with the fleet deployment
records the upgrade minted, and they exist because a record left non-terminal is
not inert: ``execution/fleet_preflight.py`` reads every non-terminal deployment
for a cluster as an in-flight rollout and holds all destructive remediation
behind it. They are kept beside each other, and apart from the phase
orchestration that calls them, because the pair is one answer -- a targeted
cancel plus a sweep for what the cancel cannot name -- and reading either half
alone gives the wrong impression of the coverage.

``regional_release_orchestration`` re-exports both, so ``rollout_regional_release``
and the command tests keep reaching them through the module that owns the
rollback flow.
"""

from __future__ import annotations

import sys
from typing import Any

from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_progress import RollbackCompensationPlan


def cleanup_candidate_rollout_state(
    self: Any,
    compensation: RollbackCompensationPlan,
) -> None:
    for target in self.config.clusters:
        components = compensation.for_cluster(target.cluster_id)
        if components.intersection(
            {ReleaseComponent.RECONCILER, ReleaseComponent.AGENT}
        ):
            self._cancel_active_installer_jobs(target)
        if ReleaseComponent.AGENT not in components:
            continue
        deployment_id = self._fleet_deployment_id(
            target,
            phase="upgrade",
            artifact_sha=self.node_wheel_sha,
            bundle_sha=self.bundle_sha,
            template_sha=self.node_template_sha,
            config_digest=self.config.agent_config_digest,
            runtime_profile_version=self.config.runtime_profile_version,
        )
        self._fleet_command(
            "cancel-if-present",
            {
                "deployment_id": deployment_id,
                "reason": (
                    f"release {self.release_id} was rolled back after verification"
                ),
            },
        )
    terminalize_stranded_rollouts(self)


def terminalize_stranded_rollouts(self: Any) -> tuple[str, ...]:
    """Drive every non-terminal rollout of this release to a terminal state.

    The per-cluster ``cancel-if-present`` above is not sufficient on its own,
    for two independent reasons, and a miss is silent in both:

    * It is keyed by ``_fleet_deployment_id``, a hash over the candidate
      artifact/bundle/template/config digests. Any of those moving between the
      upgrade minting the record and the rollback recomputing the id resolves
      the cancel to ``ABSENT``, which is treated as success.
    * It only runs for a cluster whose compensation plan contains ``AGENT``,
      and the plan is derived from recorded component progress. An upgrade that
      created the fleet deployment and then died before its progress write
      leaves a record no plan claims.

    A record left non-terminal is not inert. ``execution/fleet_preflight.py``
    reads every non-terminal deployment for the cluster as an in-flight rollout
    and holds all destructive remediation behind it. On 2026-09-03 a
    ``release-upgrade`` record stranded in ``PLANNED`` -- four nodes still
    ``PENDING``, ``updated_at == created_at``, so not one wave had started --
    silently fenced the cluster for 34 hours until it was found by hand.

    So the sweep is by release id, covers every phase except the rollback's
    own, and is unconditional: a rollback holds the release lock, so no other
    rollout of the same release can legitimately be in flight.
    """

    result = self._fleet_command(
        "terminalize-release-rollouts",
        {
            "release_id": self.release_id,
            "reason": (
                f"release {self.release_id} was rolled back; the rollout was "
                "superseded before reaching a terminal state"
            ),
        },
    )
    terminalized = tuple(str(item) for item in result.get("terminalized") or ())
    if terminalized:
        print(
            "release-rollback terminalized stranded fleet deployments: "
            + ", ".join(terminalized),
            file=sys.stderr,
        )
    return terminalized


def terminalize_stranded_cluster_rollouts(
    self: Any,
    *,
    cluster_id: str,
    keep_deployment_id: str,
    reason: str,
) -> tuple[str, ...]:
    """Drive every other non-terminal rollout of ``cluster_id`` to a terminal state.

    Called by the node-runtime rollout right before it creates (or resumes)
    ``keep_deployment_id``. The by-release sweep in
    ``terminalize_stranded_rollouts`` only runs during a rollback and only sees
    the release being rolled back, so a transaction abandoned without a
    rollback -- it died in a guard, or its resume was refused and a fresh
    transaction was started instead -- leaves its fleet deployment record
    ``IN_PROGRESS`` forever, and ``execution/fleet_preflight.py`` fences every
    destructive remediation for the cluster behind it. On 2026-09-07 two such
    records (one of a different release id) did exactly that for four hours
    after the successor had converged.

    The release holding the lock is the only legitimate rollout for the
    cluster, so anything non-terminal other than its own record is stranded.
    Fakes and older control planes may answer with nothing; that is "nothing
    terminalized", never an error on the rollout path.
    """

    result = self._fleet_command(
        "terminalize-cluster-rollouts",
        {
            "cluster_id": cluster_id,
            "keep_deployment_id": keep_deployment_id,
            "reason": reason,
        },
    )
    terminalized = (
        tuple(str(item) for item in result.get("terminalized") or ())
        if isinstance(result, dict)
        else ()
    )
    if terminalized:
        print(
            "release-upgrade terminalized stranded fleet deployments: "
            + ", ".join(terminalized),
            file=sys.stderr,
        )
    return terminalized
