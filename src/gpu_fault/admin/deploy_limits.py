"""Deployment pressure ceilings; safety proofs and authorization remain separate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentConcurrency:
    bootstrap_tasks: int = 8
    read_only_checks: int = 8
    candidate_clusters: int = 4
    metadata_cleanup: int = 8
    site_installers: int = 64
    cluster_installers: int = 32


DEPLOY_CONCURRENCY = DeploymentConcurrency()
