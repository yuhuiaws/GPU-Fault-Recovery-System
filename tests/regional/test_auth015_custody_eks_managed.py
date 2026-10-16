"""AUTH-015 judges a plain EKS cluster's custody binding by its provider scope.

Custody binds a plain EKS (eks-managed) cluster under its ``cluster_id`` (it has
no HyperPod name; ``bootstrap_common.node_key_scope_name``) and selects its
nodes with the cluster's node selector; the runner's live anchors compared the
target's empty HyperPod name with the binding and refused every such site.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from scripts.e2e.regional import auth015_custody as custody
from tests.regional._auth015_custody_support import RuntimeFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard

SELECTOR = f"{OWNERSHIP_LABEL}=custody-site"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    fixture = RuntimeFixture(tmp_path, monkeypatch)
    fixture.install_runtime_keys()
    try:
        yield fixture
    finally:
        fixture.close()


def _as_eks_managed(runtime):
    target = replace(
        runtime.site.target,
        hyperpod_cluster_name="",
        provider="eks-managed",
        node_selector=SELECTOR,
    )
    runtime.site.target = target
    for item in runtime.site.raw["nodes"]["items"]:
        item["metadata"]["labels"] = {OWNERSHIP_LABEL: "custody-site"}
    binding = runtime.provisioning.binding
    return target, binding.model_copy(
        update={
            "site": binding.site.model_copy(
                update={"hyperpod_cluster": target.cluster_id}
            )
        }
    )


def test_live_anchors_accept_a_plain_eks_binding_under_its_cluster_id(runtime):
    target, binding = _as_eks_managed(runtime)

    custody.live_anchors(runtime.site, target, binding)

    reads = [event for event in runtime.site.events if "nodes" in event]
    assert reads and all(SELECTOR in event for event in reads), (
        "the plain EKS nodes must be read through the cluster's own selector"
    )


def test_live_anchors_still_refuse_a_plain_eks_binding_of_another_scope(runtime):
    target, binding = _as_eks_managed(runtime)
    foreign = binding.model_copy(
        update={"site": binding.site.model_copy(update={"hyperpod_cluster": "other"})}
    )

    with pytest.raises(CustodyError, match="differs from the authorized custody"):
        custody.live_anchors(runtime.site, target, foreign)
