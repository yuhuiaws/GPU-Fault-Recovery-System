from __future__ import annotations

import copy
import json

import pytest
import yaml

from gpu_fault.admin import cluster_removal_state as state_api
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


@pytest.fixture
def context(tmp_path):
    site = load_site(site_file(tmp_path))
    directory, path, state = state_api.load_removal_state(site, "gpu-a")
    return site, directory, path, state


def persist(context):
    context[2].write_text(json.dumps(context[3]))


def discovered(context):
    state = context[3]
    state["phase"] = "DISCOVERED"
    state["completed_steps"] = ["DISCOVERED"]
    state["evidence"]["DISCOVERED"] = {
        "target": copy.deepcopy(state["target"]),
        "provider_identity": {
            "eks_arn": state["target"]["eks_cluster_arn"],
            "hyperpod_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/example",
        },
        "namespace_uid": "ns-example",
        "registry_digest": "a" * 64,
        "token_sha256": "b" * 64,
    }
    persist(context)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", 1, "current identity binding"),
        ("completed_steps", None, "invalid checkpoints"),
        ("completed_steps", [None], "invalid checkpoints"),
        ("completed_steps", ["DISCOVERED", "DISCOVERED"], "invalid checkpoints"),
        ("completed_steps", ["unknown"], "invalid checkpoints"),
        ("completed_steps", ["AWS_DETACHED"], "incomplete safety barriers"),
        ("evidence", [], "invalid checkpoints"),
        ("target", None, "invalid checkpoints"),
        ("phase", None, "invalid phase"),
        ("phase", "unknown", "invalid phase"),
        ("phase", "DISCOVERED", "invalid phase"),
        ("phase", "COMPLETED", "invalid phase"),
        ("attempt_id", None, "attempt identity"),
        ("attempt_id", "short", "attempt identity"),
        ("attempt_id", "G" * 32, "attempt identity"),
    ],
)
def test_removal_journal_rejects_unknown_or_inconsistent_checkpoints(
    context, field, value, message
):
    context[3][field] = value
    persist(context)
    with pytest.raises(BootstrapError, match=message):
        state_api.read_removal_state(context[2])


@pytest.mark.parametrize(
    "field",
    ["target", "provider_identity", "namespace_uid", "registry_digest", "token_sha256"],
)
def test_discovery_checkpoint_requires_all_identity_proofs(context, field):
    discovered(context)
    context[3]["evidence"]["DISCOVERED"].pop(field)
    persist(context)
    with pytest.raises(BootstrapError, match="discovery evidence is incomplete"):
        state_api.read_removal_state(context[2])


@pytest.mark.parametrize("kind", ["missing", "bad-json", "not-object"])
def test_unreadable_journal_never_becomes_new_attempt(context, kind):
    if kind == "missing":
        context[2].unlink()
    else:
        context[2].write_text("not-json" if kind == "bad-json" else "[]")
    with pytest.raises(BootstrapError, match="cannot read|current identity"):
        state_api.read_removal_state(context[2])


@pytest.mark.parametrize(
    "completed,allowed",
    [
        ([], {"ACTIVE"}),
        (["DISCOVERED"], {"ACTIVE", "DRAINING"}),
        (["CONTROL_REGISTRY_DRAINING"], {"DRAINING"}),
        (["KUBERNETES_REMOVED"], {"DRAINING", "REVOKED", None}),
        (["CONTROL_REGISTRY_REMOVED"], {None}),
    ],
)
def test_lifecycle_allowances_follow_monotonic_removal_barriers(completed, allowed):
    assert (
        state_api.allowed_registry_lifecycles({"completed_steps": completed}) == allowed
    )


@pytest.mark.parametrize(
    "field,value",
    [("site_id", "foreign"), ("cluster_id", "foreign"), ("site_identity", {})],
)
def test_saved_site_identity_drift_is_refused(context, field, value):
    context[3][field] = value
    with pytest.raises(BootstrapError, match="conflicts"):
        state_api.validate_saved_site(context[0], "gpu-a", context[3])


@pytest.mark.parametrize("field", ["source_site_sha256", "source_release_sha256"])
def test_saved_source_digest_drift_cannot_authorize_removal(context, field):
    context[3][field] = "0" * 64
    with pytest.raises(BootstrapError, match="identity drifted"):
        state_api.validate_saved_site(context[0], "gpu-a", context[3])


@pytest.mark.parametrize(
    "kind", ["attempt-directory", "site-identity", "cluster-identity"]
)
def test_resume_requires_original_attempt_directory_and_identity(context, kind):
    site, directory, _path, state = context
    if kind == "attempt-directory":
        directory.rmdir()
    elif kind == "site-identity":
        state["site_identity"] = {}
    else:
        state["cluster_id"] = "foreign"
    persist(context)
    with pytest.raises(BootstrapError, match="missing|conflicts"):
        state_api.load_removal_state(site, "gpu-a")


@pytest.mark.parametrize("cluster", ["", "../outside", "white space", "missing"])
def test_invalid_or_unmanaged_cluster_has_no_removal_state(context, cluster):
    with pytest.raises(BootstrapError, match="identity is invalid|unknown cluster_id"):
        state_api.load_removal_state(context[0], cluster)


def test_absent_target_without_site_commit_proof_is_refused(context):
    site = context[0]
    document = yaml.safe_load(site.source.read_text())
    document["spec"]["clusters"] = []
    site.source.write_text(yaml.safe_dump(document))
    with pytest.raises(BootstrapError, match="absent without"):
        state_api.validate_saved_site(load_site(site.source), "gpu-a", context[3])


@pytest.mark.parametrize("raw", ["invalid: [yaml", "[]", "spec: {}"])
def test_site_document_binding_requires_parsable_cluster_inventory(context, raw):
    site = context[0]
    site.source.write_text(raw)
    with pytest.raises(BootstrapError, match="cannot bind"):
        state_api.site_documents(site, "gpu-a")


def test_ownership_loss_persists_and_refuses_reentry(context):
    _site, _directory, path, state = context
    with pytest.raises(ProcessSupervisionLost):
        with state_api.removal_command_ownership(path, state):
            raise ProcessSupervisionLost("example loss")
    assert state_api.read_removal_state(path)["phase"] == "SUPERVISION_LOST"
    with pytest.raises(ProcessSupervisionLost, match="automatic retry"):
        with state_api.removal_command_ownership(path, state):
            pytest.fail("lost supervision allowed reentry")


def test_ownership_loss_survives_failure_to_persist_checkpoint(context):
    _site, directory, _path, state = context
    with pytest.raises(ProcessSupervisionLost) as error:
        with state_api.removal_command_ownership(directory, state):
            raise ProcessSupervisionLost("example loss")
    assert state["phase"] == "SUPERVISION_LOST"
    assert "could not persist" in error.value.__notes__[0]
