"""What uninstall proves before it reports success.

Deleting is the easy half. These are the checks that decide whether the deletion
actually happened: the Kubernetes sweep over every registered object, and the AWS
verification that a resource marked ``DELETE`` is gone while a resource marked
``PRESERVE`` is still there. A false "verified" here is how an uninstall reports a
clean run while leaving a live NLB, or reports success after deleting something a
customer asked to keep.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import uninstall as admin_uninstall
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall import (
    CPU_CLUSTER_BOUND_RESOURCE_KEYS,
    _delete_or_verify_cpu,
    _verify_gpu_clusters,
    _verify_resources,
    verify_installed_registry_cleanup,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from tests.admin.test_admin_site import site_file

NOT_FOUND = 'Error from server (NotFound): namespaces "gpu-fault-system" not found'


def _resource(
    key: str,
    resource_type: str,
    resource_id: str,
    *,
    policy: InstallationResourceDeletePolicy = (
        InstallationResourceDeletePolicy.DELETE
    ),
    ownership: InstallationResourceOwnership = InstallationResourceOwnership.CREATED,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=ownership,
        delete_policy=policy,
        created_at=now,
        updated_at=now,
    )


def _inventory() -> dict[str, Any]:
    return {
        "inventory_snapshot": {
            "cpu": {
                "resources": [
                    {"scope": "namespaced", "kind": "deployment", "name": "api-ha"},
                    {
                        "scope": "cluster",
                        "kind": "clusterrole",
                        "name": "gpu-fault-control",
                    },
                ]
            },
            "gpu": {
                "resources": [
                    {"scope": "namespaced", "kind": "daemonset", "name": "collector"}
                ]
            },
        }
    }


class Kubectl:
    """Answers ``kubectl get`` for the cleanup sweep.

    ``present`` is the set of command fragments that should still resolve, so a
    test can leave exactly one object behind.
    """

    def __init__(
        self, *, present: Sequence[str] = (), error: str | None = None
    ) -> None:
        self.present = tuple(present)
        self.error = error
        self.calls: list[list[str]] = []

    def __call__(
        self, arguments: Sequence[Any], **_keywords: Any
    ) -> subprocess.CompletedProcess:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        line = " ".join(argv)
        if self.error is not None:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=self.error)
        if any(fragment in line for fragment in self.present):
            return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=NOT_FOUND)

    def queries(self) -> list[str]:
        return [" ".join(argv[argv.index("get") :]) for argv in self.calls]


class Cleaner:
    """Stands in for ``ResourceCleaner`` during verification.

    Only ``exists`` matters here, plus a record of which resources were probed at
    all -- a resource that is verified without being probed is the failure mode
    these tests exist for.
    """

    def __init__(self, *, existing: Sequence[str] = ()) -> None:
        self.existing = tuple(existing)
        self.probed: list[str] = []
        self.deleted_cpu: list[tuple[str, str]] = []

    def exists(self, resource: InstallationResource) -> bool:
        self.probed.append(resource.resource_key)
        return resource.resource_key in self.existing

    def delete_cpu_cluster(
        self, hyperpod: InstallationResource, eks: InstallationResource
    ) -> None:
        self.deleted_cpu.append((hyperpod.resource_key, eks.resource_key))
        self.existing = tuple(
            key
            for key in self.existing
            if key not in {hyperpod.resource_key, eks.resource_key}
        )


def test_the_kubernetes_sweep_covers_every_object_on_every_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each registered object is checked once per cluster, plus registry and namespace.

    The GPU side is per cluster context: checking only the first one would let a
    second GPU cluster keep a running collector that still writes to a control
    plane that no longer exists.
    """

    site = load_site(site_file(tmp_path))
    kubectl = Kubectl()
    monkeypatch.setattr(admin_uninstall.subprocess, "run", kubectl)

    result = verify_installed_registry_cleanup(site, _inventory())

    contexts = [item["context"] for item in site.release_config["clusters"]]
    expected = 2 + len(contexts)
    assert result == {"registered_kubernetes_resources_verified_absent": expected}
    assert kubectl.queries().count("get namespace gpu-fault-system") == 1 + len(
        contexts
    )
    assert kubectl.queries().count(
        "get configmap gpu-fault-installed-resources"
    ) == 1 + len(contexts)
    namespaced = next(argv for argv in kubectl.calls if "deployment" in " ".join(argv))
    assert "-n" in namespaced, "a namespaced object was queried without its namespace"
    cluster_scoped = next(
        argv for argv in kubectl.calls if "clusterrole" in " ".join(argv)
    )
    assert "-n" not in cluster_scoped


@pytest.mark.parametrize(
    ("present", "message"),
    [
        ("deployment api-ha", "installed resource still exists"),
        ("configmap gpu-fault-installed-resources", "registry still exists"),
        ("namespace gpu-fault-system", "solution namespace still exists"),
    ],
)
def test_one_surviving_object_fails_the_whole_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: str, message: str
) -> None:
    """Anything left behind keeps reconciling against a half-deleted install.

    The collector and the control-plane Deployment both act on their own; a
    surviving namespace also blocks a later reinstall, so none of the three may be
    reported as clean.
    """

    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(admin_uninstall.subprocess, "run", Kubectl(present=[present]))

    with pytest.raises(BootstrapError, match=message):
        verify_installed_registry_cleanup(site, _inventory())


def test_an_unreadable_cluster_is_not_read_as_a_clean_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a NotFound answer proves absence.

    An expired token or a revoked role answers every query with an error; treating
    that as "gone" would verify an uninstall that never reached the cluster.
    """

    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        admin_uninstall.subprocess,
        "run",
        Kubectl(error="error: You must be logged in to the server (Unauthorized)"),
    )

    with pytest.raises(BootstrapError, match="verification failed for cpu:"):
        verify_installed_registry_cleanup(site, _inventory())


def test_verification_records_the_terminal_state_of_each_resource() -> None:
    """Deleted, detached and preserved are three different end states.

    The sealed registry is the record of what the account looks like afterwards, so
    a detached resource marked ``DELETED`` would make a later audit look for
    something that is still there.
    """

    deleted = _resource("aws/nlb", "nlb", "test-nlb")
    detached = _resource(
        "aws/route53/vpc-association/gpu-a",
        "route53_vpc_association",
        "vpc-gpu-a",
        policy=InstallationResourceDeletePolicy.DETACH,
    )
    preserved = _resource(
        "aws/ses/administrator-email-identity",
        "ses_email_identity",
        "ops@example.com",
        policy=InstallationResourceDeletePolicy.PRESERVE,
        ownership=InstallationResourceOwnership.EXTERNAL,
    )
    cleaner = Cleaner(existing=["aws/ses/administrator-email-identity"])

    verified = _verify_resources(
        cleaner, [deleted, detached, preserved], cpu_disposition="keep"
    )

    assert [item.status for item in verified] == [
        InstallationResourceStatus.DELETED,
        InstallationResourceStatus.DETACHED,
        InstallationResourceStatus.PRESERVED,
    ]
    assert cleaner.probed == [
        "aws/nlb",
        "aws/route53/vpc-association/gpu-a",
        "aws/ses/administrator-email-identity",
    ]


def test_a_reused_resource_is_sealed_as_owned_once_it_is_deleted() -> None:
    """A reused resource that this solution deleted is no longer reusable.

    Leaving it ``REUSED`` in the sealed registry would tell a later run that the
    resource is still there to adopt.
    """

    reused = _resource(
        "aws/sqs/notifications",
        "sqs_queue",
        "gpu-fault-notifications",
        ownership=InstallationResourceOwnership.REUSED,
    )

    verified = _verify_resources(Cleaner(), [reused], cpu_disposition="keep")

    assert verified[0].ownership is InstallationResourceOwnership.CREATED
    assert verified[0].delete_policy is InstallationResourceDeletePolicy.DELETE


@pytest.mark.parametrize(
    ("policy", "existing", "message"),
    [
        (
            InstallationResourceDeletePolicy.PRESERVE,
            (),
            "preserved resource is missing",
        ),
        (InstallationResourceDeletePolicy.DELETE, ("aws/nlb",), "still exists"),
    ],
)
def test_verification_fails_in_both_directions(
    policy: InstallationResourceDeletePolicy, existing: tuple[str, ...], message: str
) -> None:
    """A missing preserved resource is as bad as a surviving deleted one.

    Uninstall promises both halves: what it was told to delete is gone, and what it
    was told to keep is untouched.
    """

    resource = _resource(
        "aws/nlb",
        "nlb",
        "test-nlb",
        policy=policy,
        ownership=(
            InstallationResourceOwnership.EXTERNAL
            if policy is InstallationResourceDeletePolicy.PRESERVE
            else InstallationResourceOwnership.CREATED
        ),
    )

    with pytest.raises(BootstrapError, match=message):
        _verify_resources(
            Cleaner(existing=existing), [resource], cpu_disposition="keep"
        )


def test_resources_bound_to_a_deleted_cpu_cluster_are_not_probed() -> None:
    """Probing a resource inside a cluster that no longer exists always errors.

    The Pod Identity add-on goes away with the cluster, so once the CPU cluster is
    deleted the probe would fail on a ``ResourceNotFoundException`` for the cluster
    itself and turn a correct uninstall into a verification failure.
    """

    key = sorted(CPU_CLUSTER_BOUND_RESOURCE_KEYS)[0]
    addon = _resource(key, "eks_addon", "eks-pod-identity-agent")
    cleaner = Cleaner(existing=(key,))

    verified = _verify_resources(cleaner, [addon], cpu_disposition="delete")

    assert cleaner.probed == []
    assert verified[0].status is InstallationResourceStatus.DELETED


def _cpu_snapshot(*, resources: Sequence[InstallationResource]) -> Any:
    return InstallationResourceSnapshot(site_id="test-site", resources=list(resources))


def _cpu_pair() -> list[InstallationResource]:
    return [
        _resource(
            "cluster/cpu/eks",
            "cpu_eks",
            "control",
            policy=(InstallationResourceDeletePolicy.PRESERVE),
            ownership=InstallationResourceOwnership.EXTERNAL,
        ),
        _resource(
            "cluster/cpu/hyperpod",
            "cpu_hyperpod",
            "hp-control",
            policy=(InstallationResourceDeletePolicy.PRESERVE),
            ownership=InstallationResourceOwnership.EXTERNAL,
        ),
    ]


def test_deleting_the_cpu_cluster_deletes_hyperpod_before_eks_and_verifies_both() -> (
    None
):
    """The HyperPod cluster has to go first, and both must be gone afterwards.

    Deleting the EKS cluster first orphans the HyperPod cluster, which then cannot
    be deleted at all; verifying only one of the two would hide that.
    """

    snapshot = _cpu_snapshot(resources=_cpu_pair())
    cleaner = Cleaner(existing=("cluster/cpu/eks", "cluster/cpu/hyperpod"))

    _delete_or_verify_cpu(cleaner, snapshot, disposition="delete")

    assert cleaner.deleted_cpu == [("cluster/cpu/hyperpod", "cluster/cpu/eks")]
    assert set(cleaner.probed) == {"cluster/cpu/eks", "cluster/cpu/hyperpod"}


def test_keeping_the_cpu_cluster_requires_it_to_still_be_there() -> None:
    """``--keep`` is a promise the control plane cluster survives.

    If it is gone, uninstall deleted something it was told not to, and the run must
    not report success.
    """

    snapshot = _cpu_snapshot(resources=_cpu_pair())

    with pytest.raises(BootstrapError, match="CPU cluster resource is missing"):
        _delete_or_verify_cpu(Cleaner(), snapshot, disposition="keep")


def test_a_registry_missing_a_cpu_record_stops_the_uninstall() -> None:
    """Both CPU records are required to decide what to delete.

    A registry with only the EKS record would let ``--delete`` leave the HyperPod
    cluster running and billing.
    """

    snapshot = _cpu_snapshot(resources=_cpu_pair()[:1])

    with pytest.raises(BootstrapError, match="both CPU cluster records"):
        _delete_or_verify_cpu(Cleaner(), snapshot, disposition="delete")


def test_gpu_clusters_are_counted_and_required_to_survive() -> None:
    """GPU clusters are never deleted by uninstall.

    They hold the customer's training capacity; the count is what the report shows
    the operator as proof they were left alone.
    """

    resources = [
        _resource("cluster/gpu-a/eks", "gpu_eks", "gpu-a"),
        _resource("cluster/gpu-a/hyperpod", "gpu_hyperpod", "hp-gpu-a"),
    ]
    snapshot = _cpu_snapshot(resources=resources)
    keys = tuple(item.resource_key for item in resources)

    assert _verify_gpu_clusters(Cleaner(existing=keys), snapshot) == 2

    with pytest.raises(BootstrapError, match="was not preserved"):
        _verify_gpu_clusters(Cleaner(existing=keys[:1]), snapshot)


def test_a_registry_with_no_gpu_cluster_stops_the_uninstall() -> None:
    """An empty GPU list means the registry is not the one for this site.

    Continuing would verify nothing and report a clean uninstall for a site whose
    GPU clusters were never examined.
    """

    snapshot = _cpu_snapshot(resources=_cpu_pair())

    with pytest.raises(BootstrapError, match="contains no GPU clusters"):
        _verify_gpu_clusters(Cleaner(), snapshot)
