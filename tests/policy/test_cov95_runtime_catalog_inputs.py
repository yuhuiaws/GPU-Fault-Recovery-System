from __future__ import annotations

from copy import deepcopy
from importlib.resources import files

import pytest
import yaml

from gpu_fault.policy import GpuFaultPolicyEngine, load_sxid_policy, load_xid_policy
from gpu_fault.policy.models import NVIDIA_ALWAYS_FATAL_SXIDS, SxidClassification


@pytest.fixture(scope="module")
def catalogs():
    return load_xid_policy(), load_sxid_policy()


@pytest.fixture
def sxid_document():
    return yaml.safe_load(
        files("gpu_fault.data")
        .joinpath("nvidia-fabric-manager-sxid-2025-11-14.yaml")
        .read_text(encoding="utf-8")
    )


def test_explicit_sxid_catalog_preserves_fatality_and_normalizes_no_investigation(
    tmp_path, sxid_document
):
    path = tmp_path / "sxid.yaml"
    path.write_text(yaml.safe_dump(sxid_document), encoding="utf-8")
    policy = load_sxid_policy(path)
    rules = {rule.sxid: rule for rule in policy.rules}
    assert rules[12028].investigatory_action is None, rules[12028]
    assert rules[11004].official_action == "RESTART_VM", rules[11004]
    assert {
        rule.sxid
        for rule in policy.rules
        if rule.classification is SxidClassification.ALWAYS_FATAL
    } == NVIDIA_ALWAYS_FATAL_SXIDS, policy


@pytest.mark.parametrize("change", ["duplicate", "fatality"])
def test_explicit_sxid_catalog_rejects_ambiguous_or_downgraded_rules(
    tmp_path, sxid_document, change
):
    document = deepcopy(sxid_document)
    if change == "duplicate":
        document["spec"]["rules"].append(deepcopy(document["spec"]["rules"][0]))
        message = "duplicate 11004"
    else:
        group = next(
            item
            for item in document["spec"]["rules"]
            if item["classification"] == "ALWAYS_FATAL"
        )
        group["classification"] = "NON_FATAL"
        message = "Always-Fatal SXID set"
    path = tmp_path / "invalid-sxid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_sxid_policy(path)


@pytest.mark.parametrize("size", [0, -1])
def test_policy_engine_rejects_nonpositive_idempotency_cache(catalogs, size):
    with pytest.raises(ValueError, match="cache size must be positive"):
        GpuFaultPolicyEngine(*catalogs, decision_cache_size=size)


@pytest.mark.parametrize("family", ["XIDs", "SXIDs"])
def test_injected_policy_models_cannot_ambiguously_redefine_an_event(catalogs, family):
    xid, sxid = catalogs
    if family == "XIDs":
        xid = xid.model_copy(
            update={"catalog_rules": [*xid.catalog_rules, xid.catalog_rules[0]]}
        )
    else:
        sxid = sxid.model_copy(update={"rules": [*sxid.rules, sxid.rules[0]]})
    with pytest.raises(ValueError, match=f"duplicate {family}"):
        GpuFaultPolicyEngine(xid, sxid)
