"""``cpu_secrets`` over a site whose GPU clusters have no node.

The check refused an empty CPU ``gpu-fault-node-action-keys`` whenever the site
named a GPU cluster (``requires_node_keys=bool(clusters)``): a site whose
clusters have no keyed node -- its last keyed cluster removed, a first cluster
joined without nodes -- failed ``preflight``, ``verify`` and ``status --full``
although no node has a key to mirror. The empty Secret now passes while every
GPU cluster's selector selects no node, parked or unavailable (the release
engine's zero-node proof), and fails as before when a cluster has a node, when
a node list cannot be read, and when the Secret is missing or the read does not
return it as itself.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from gpu_fault.admin.execution import ProofCache
from gpu_fault.cluster_scope import node_selector
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_cpu_rolling_update as CPU_ROLL
from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_secret_checks as SECRETS
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_state import read_snapshot
from gpu_fault_release.regional_validation_evidence import QUICK_VALIDATION_EVIDENCE_ENV
from tests.regional._parked_cluster_support import (
    EKS,
    HYPERPOD,
    ParkedSite,
    ProviderRunner,
)

NAMESPACE = "gpu-fault-system"
KEYS = "gpu-fault-node-action-keys"
NOT_FOUND = f'Error from server (NotFound): secrets "{KEYS}" not found'
#: ``gpu-a``'s instance group (``TargetCount``, ``CurrentCount``): parked, or
#: wanting the instance it lost.
HYPERPOD_GROUP = {"parked": (0, 0), "unavailable": (1, 0), "nodes": (1, 1)}
PREFLIGHT_CHECKS = (
    "_check_tools",
    "_check_local_inputs",
    "check_artifact_config_maps",
    "_check_aws_identity",
    "_check_contexts",
    "_check_cpu_capacity",
    "_check_load_balancer_controller",
    "_check_nlb_inputs",
    "_check_aurora",
    "check_aurora_refresh",
    "check_control_record_archive_bucket",
    "check_diagnostic_evidence_destination",
    "check_email_notifications",
    "_check_monitoring",
    "workflow_safety_snapshot",
)
HEALTH_CHECKS = (
    "_check_contexts",
    "_check_cpu_workloads",
    "check_email_notifications",
    "check_aurora_refresh",
    "check_adot_self_metrics",
    "_verify_profile",
    "run_read_only_verifiers",
    "_check_runtime_component_identity",
    "_check_control_api",
    "_check_nlb_runtime",
    "_check_aurora",
    "_check_monitoring",
)


def _encoded(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def secret(name: str, data: dict[str, str] | None, **metadata: Any) -> dict[str, Any]:
    """A Secret as ``kubectl get secret <name> -o json`` returns it."""

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "uid": f"{name}-uid",
            "resourceVersion": "7",
            **metadata,
        },
        "data": data,
    }


class Provider(ProviderRunner):
    """The parked proof's provider reads, ``gpu-a``'s HyperPod instance group too."""

    def __init__(self, base: ProviderRunner, *, group: tuple[int, int]) -> None:
        super().__init__(base.nodegroups, base.nodepools)
        self.group = group

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        if arguments[:3] != ["aws", "sagemaker", "describe-cluster"]:
            return super().run(arguments, **kwargs)
        self.calls.append(list(arguments))
        target, current = self.group
        return json.dumps(
            {
                "InstanceGroups": [
                    {
                        "InstanceGroupName": "gpu-spot-group",
                        "InstanceType": "ml.p5en.48xlarge",
                        "TargetCount": target,
                        "CurrentCount": current,
                        "Status": "InService",
                        "KubernetesConfig": {"CurrentLabels": {}},
                    }
                ]
            }
        )

    def count(self, *command: str) -> int:
        return sum(call[: len(command)] == list(command) for call in self.calls)


class ZeroNodeSite(ParkedSite):
    """``gpu-eks-cluster`` parked at zero nodes beside ``gpu-a`` (HyperPod),
    which has its node or none (parked or unavailable), and the three CPU
    Secrets, the node-key one empty."""

    def __init__(self, *, hyperpod: str = "parked") -> None:
        super().__init__()
        #: Per node-list read, whether a report snapshot was open.
        self.node_reads_in_snapshot: list[bool] = []
        self.provider = Provider(self.provider, group=HYPERPOD_GROUP[hyperpod])
        self.runner = self.provider  # type: ignore[assignment]
        if hyperpod != "nodes":
            selector = node_selector(self.config.clusters[0])
            self.documents[(HYPERPOD, "nodes", "-l " + selector)] = {"items": []}
        for document in (
            secret(
                "gpu-fault-aurora",
                {"postgres-url": "reference", "master-secret-arn": "reference"},
            ),
            secret(
                "gpu-fault-control-plane-active",
                {
                    name: _encoded(character * 32)
                    for name, character in (
                        ("execution-token", "a"),
                        ("processor-replay-secret", "b"),
                        ("node-action-secret", "c"),
                    )
                },
            ),
            secret(KEYS, None),
        ):
            self.set_secret(document)

    def set_secret(self, document: dict[str, Any] | Exception) -> None:
        name = KEYS if isinstance(document, Exception) else document["metadata"]["name"]
        self.documents[("cpu", "secret", name)] = document

    def node_reads(self) -> list[tuple[str, str, str]]:
        return [read for read in self.reads if read[1] == "nodes"]

    def _get_json(self, arguments: list[str]) -> dict[str, Any]:
        if "nodes" in arguments:
            self.node_reads_in_snapshot.append(
                getattr(self, "_json_read_cache", None) is not None
            )
        return super()._get_json(arguments)

    def _apply_health_baseline(self, _state: Any) -> None:
        return None

    def _read_snapshot(self) -> Any:
        # The release's own snapshot, as ``RegionalRelease`` binds it.
        return read_snapshot(self)


@pytest.fixture(autouse=True)
def _fresh_shape_proofs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SECRETS, "PROOFS", ProofCache())
    monkeypatch.delenv(QUICK_VALIDATION_EVIDENCE_ENV, raising=False)


def _line(*states: tuple[str, str]) -> str:
    return (
        f"{KEYS} is empty and needs no key: no GPU cluster has a node ("
        + ", ".join(f"{cluster}: {state}" for cluster, state in states)
        + ")"
    )


@pytest.mark.parametrize("hyperpod", ["parked", "unavailable"])
def test_an_empty_node_key_secret_passes_while_no_gpu_cluster_has_a_node(
    hyperpod: str,
) -> None:
    site = ZeroNodeSite(hyperpod=hyperpod)

    value = CHECKS.check_cpu_secrets(site)

    line = _line((HYPERPOD, hyperpod), (EKS, "parked"))
    assert value.status == "PASS"
    assert value.summary == f"{SECRETS.SUMMARY}; {line}"
    assert value.details[SECRETS.NODE_KEYS_NOT_NEEDED_KEY] == line
    assert value.details["node_action_key_count"] == 0
    assert site.node_reads_in_snapshot == [False, False], (
        "both node lists are read afresh: the check opens no snapshot of its own"
    )


def test_a_first_cluster_joined_without_nodes_passes_with_the_empty_secret() -> None:
    """The join's candidate site names only the joining cluster, parked."""

    site = ZeroNodeSite()
    site.config.clusters = (site.eks,)

    value = CHECKS.check_cpu_secrets(site)

    assert value.details[SECRETS.NODE_KEYS_NOT_NEEDED_KEY] == _line((EKS, "parked"))


def test_an_empty_node_key_secret_fails_while_a_gpu_cluster_has_a_node() -> None:
    site = ZeroNodeSite(hyperpod="nodes")

    with pytest.raises(ReleaseError) as raised:
        CHECKS.check_cpu_secrets(site)

    assert str(raised.value) == (
        f"{KEYS} is empty, but GPU cluster {HYPERPOD} selects a node (or its "
        "node list cannot be read)"
    )


def test_an_unreadable_node_list_proves_nothing() -> None:
    site = ZeroNodeSite()
    selector = node_selector(site.config.clusters[0])
    site.documents[(HYPERPOD, "nodes", "-l " + selector)] = ReleaseError(
        "the GPU API server is unreachable"
    )

    with pytest.raises(ReleaseError, match=f"GPU cluster {HYPERPOD} selects a node"):
        CHECKS.check_cpu_secrets(site)


def test_a_missing_node_key_secret_fails_even_when_no_gpu_cluster_has_a_node() -> None:
    site = ZeroNodeSite()
    site.set_secret(ReleaseError(NOT_FOUND))

    with pytest.raises(ReleaseError, match="NotFound"):
        CHECKS.check_cpu_secrets(site)

    assert site.node_reads() == [], "nothing relaxes a Secret that is not there"


@pytest.mark.parametrize(
    "document",
    [
        {},
        secret(KEYS, None, deletionTimestamp="2026-11-08T00:00:00Z"),
        secret(KEYS, {}, uid=""),
        secret(KEYS, {}, namespace="another-namespace"),
        secret("another-secret", {}),
        {**secret(KEYS, {}), "kind": "ConfigMap"},
    ],
    ids=[
        "no-document",
        "being-deleted",
        "no-uid",
        "other-namespace",
        "other-name",
        "not-a-secret",
    ],
)
def test_only_the_empty_secret_itself_is_relaxed(document: dict[str, Any]) -> None:
    site = ZeroNodeSite()
    site.documents[("cpu", "secret", KEYS)] = document

    with pytest.raises(ReleaseError, match=f"{KEYS} is empty and was not read back"):
        CHECKS.check_cpu_secrets(site)

    assert site.node_reads() == [], "no zero-node proof is asked for"


def test_a_node_key_secret_with_keys_asks_no_cluster_for_its_nodes() -> None:
    site = ZeroNodeSite(hyperpod="nodes")
    site.set_secret(secret(KEYS, {"node-a": _encoded("k" * 64)}))

    value = CHECKS.check_cpu_secrets(site)

    assert value.summary == SECRETS.SUMMARY
    assert SECRETS.NODE_KEYS_NOT_NEEDED_KEY not in value.details
    assert value.details["node_action_key_count"] == 1
    assert site.node_reads() == [], "no node list is read"
    assert site.provider.calls == [], "no provider is read"


def test_the_proof_joins_the_snapshot_the_caller_opened() -> None:
    """Asked twice inside one snapshot, the provider is read once per cluster."""

    site = ZeroNodeSite()

    with read_snapshot(site):
        first = CHECKS.check_cpu_secrets(site)
        second = CHECKS.check_cpu_secrets(site)

    assert first == second
    assert site.node_reads_in_snapshot == [True] * 4
    assert site.provider.count("aws", "eks", "list-nodegroups") == 1
    assert site.provider.count("aws", "sagemaker", "describe-cluster") == 1


# --- the reports -----------------------------------------------------------------


def _stub(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    for name in names:
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CheckValue(name)
        )


def _checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in report["checks"]}


@pytest.mark.parametrize(
    "joining", [False, True], ids=["every-cluster-parked", "first-join-parked"]
)
def test_the_preflight_passes_an_empty_node_key_secret_over_a_site_without_nodes(
    monkeypatch: pytest.MonkeyPatch, joining: bool
) -> None:
    site = ZeroNodeSite()
    if joining:
        site.config.clusters = (site.eks,)
    _stub(monkeypatch, PREFLIGHT_CHECKS)

    report = CHECKS.build_preflight_report(site)

    check = _checks(report)["cpu_secrets"]
    states = ((EKS, "parked"),) if joining else ((HYPERPOD, "parked"), (EKS, "parked"))
    assert check["status"] == "PASS", check
    assert check["details"][SECRETS.NODE_KEYS_NOT_NEEDED_KEY] == _line(*states)
    assert report["healthy"] is True, report["summary"]


def test_the_preflight_still_fails_an_empty_node_key_secret_beside_a_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = ZeroNodeSite(hyperpod="nodes")
    _stub(monkeypatch, PREFLIGHT_CHECKS)

    report = CHECKS.build_preflight_report(site)

    check = _checks(report)["cpu_secrets"]
    assert check["status"] == "FAIL"
    assert check["summary"].startswith(
        f"{KEYS} is empty, but GPU cluster {HYPERPOD}"
    ), check["summary"]
    assert report["healthy"] is False


def _zero_node_data_plane(release: Any, target: Any, *, mode: str) -> CheckValue:
    """``gpu_cluster:<id>`` of a cluster without nodes, by the real rule
    (``zero_ready_nodes``): the other reader of the same proof in ``status``."""

    section = PARKED.zero_ready_nodes(release, target, {"items": []}, [])
    return CheckValue(PARKED.data_plane_summary(target.cluster_id, section), section)


@pytest.mark.parametrize("hyperpod", ["parked", "unavailable"])
def test_status_full_reads_the_zero_node_proof_once_beside_the_cluster_checks(
    monkeypatch: pytest.MonkeyPatch, hyperpod: str
) -> None:
    """``status --full``: ``cpu_secrets`` passes the empty Secret on the same
    proof the cluster checks report, read once per cluster for the report; an
    unavailable cluster keeps ``status`` red through its own check only."""

    site = ZeroNodeSite(hyperpod=hyperpod)
    _stub(monkeypatch, HEALTH_CHECKS)
    monkeypatch.setattr(CHECKS, "_check_gpu_cluster", _zero_node_data_plane)
    monkeypatch.setattr(CPU_ROLL, "health_checks", lambda *_args: [])
    monkeypatch.setattr(
        ADMIN,
        "build_release_summary",
        lambda _release, **_kwargs: {"live_release": {}, "next_deploy": {}},
    )

    status = ADMIN.build_status(site, full=True)

    checks = _checks(status["health"])
    assert checks["cpu_secrets"]["status"] == "PASS", checks["cpu_secrets"]
    assert checks["cpu_secrets"]["details"][SECRETS.NODE_KEYS_NOT_NEEDED_KEY] == (
        _line((HYPERPOD, hyperpod), (EKS, "parked"))
    )
    assert {
        cluster: entry["state"]
        for cluster, entry in status["zero_node_clusters"].items()
    } == {HYPERPOD: hyperpod, EKS: "parked"}
    assert ADMIN.failing_checks(status) == (
        [] if hyperpod == "parked" else [f"gpu_cluster:{HYPERPOD}"]
    )
    assert status["healthy"] is (hyperpod == "parked")
    assert site.provider.count("aws", "eks", "list-nodegroups") == 1
    assert site.provider.count("aws", "sagemaker", "describe-cluster") == 1
