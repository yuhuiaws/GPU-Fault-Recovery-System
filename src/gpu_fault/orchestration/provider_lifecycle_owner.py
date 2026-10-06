"""Plan-time ownership of provider lifecycle steps on a registered cluster.

A RESTART_NODE or REPLACE_NODE step is claimed by exactly one adapter per
cluster: ``gpu-fault-hyperpod-adapter`` on HyperPod, ``gpu-fault-eks-adapter``
on a plain EKS node group (``PROVIDER_NODE_REBOOT_OWNERS``). The runtime
profile a finding names is what the workload or collector stamped on it, and
on a mixed site that can be the *site* profile rather than the cluster's own
(live, gpu-eks-cluster DESTR-003: a workload submitted under the site's
``hyperpod-v1`` carried that version into its observation and the synthetic
replacement finding; REPLACE_NODE compiled with the HyperPod adapter as owner,
no executor on the eks cluster claimed it, and the step failed on the
unclaimed deadline fifteen minutes later with the node quarantined by the
hand-off). The cluster's registration knows better: it names the provider and
the Runtime Profile the cluster's own Agents, Executor and Watcher are pinned
to. Planning reads both -- the pinned profile replaces the named one, and a
provider step whose owner the cluster's executor never advertises is refused
here, before dispatch, as a BLOCKED plan with the reason in words.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS, cluster_provider
from gpu_fault.operation_registry import OperationAdapter, operations_for_adapter
from gpu_fault.store import NotFoundError

if TYPE_CHECKING:
    from gpu_fault.models import WorkflowStepSpec
    from gpu_fault.regional import RegionalClusterRegistration

#: The steps a provider lifecycle adapter claims: node reboot and replacement.
PROVIDER_LIFECYCLE_OPERATIONS = operations_for_adapter(OperationAdapter.HYPERPOD)


def cluster_registration(
    store: Any, cluster_id: str
) -> RegionalClusterRegistration | None:
    """The cluster's registration, or ``None`` where none is recorded (a
    single-cluster or simulated deployment keeps planning as before)."""

    try:
        registration = store.get_regional_cluster(cluster_id)
    except (NotFoundError, KeyError, AttributeError):
        return None
    return registration  # type: ignore[no-any-return]


def pinned_profile_version(
    registration: RegionalClusterRegistration | None, named: str
) -> str:
    """The Runtime Profile the cluster is pinned to, else the named one.

    An empty registration pin means the cluster inherits the site profile,
    which is then whatever the finding named.
    """

    if registration is None:
        return named
    pinned = str(registration.runtime_profile_version or "").strip()
    return pinned or named


def expected_lifecycle_owner(registration: RegionalClusterRegistration) -> str:
    """The adapter owner the cluster's executor advertises for its provider."""

    return PROVIDER_NODE_REBOOT_OWNERS[cluster_provider(registration.provider).value]


def provider_owner_errors(
    steps: list[WorkflowStepSpec],
    registration: RegionalClusterRegistration | None,
) -> list[str]:
    """Why the compiled provider steps could never be claimed on this cluster.

    Empty without a registration (nothing known to check against) and when
    every RESTART_NODE/REPLACE_NODE names the provider's own adapter. A step
    owned by another provider's adapter -- the other side of a mixed site --
    is named with the owner that would have claimed it; the plan is refused
    now rather than timing out unclaimed and handing the node to a drain.
    """

    if registration is None:
        return []
    expected = expected_lifecycle_owner(registration)
    provider = cluster_provider(registration.provider).value
    return [
        f"{step.operation.value} is owned by {step.execution_owner}, which a "
        f"{provider} cluster's executor never advertises (it advertises "
        f"{expected}); the runtime profile the plan compiled against is not "
        f"cluster {registration.cluster_id}'s"
        for step in steps
        if step.operation in PROVIDER_LIFECYCLE_OPERATIONS
        and step.execution_owner != expected
    ]
