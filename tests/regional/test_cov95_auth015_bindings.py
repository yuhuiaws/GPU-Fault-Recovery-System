from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional import auth015_bindings as bindings
from scripts.e2e.regional import auth015_deployed as deployed
from scripts.e2e.regional import auth015_release as release
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from tests.regional._cov95_auth015_live import API_TOKEN, LiveSite
from tests.regional._cov95_auth015_support import KEY_A, KEY_B
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def live(tmp_path, monkeypatch):
    site = LiveSite(tmp_path, monkeypatch)
    try:
        yield site
    finally:
        site.close()


def run_proof(live):
    return deployed.prove_deployed_protocol(
        live,
        live.target,
        nodes=("node-a", "node-b"),
        key_document=live.key_document,
        release_inputs=live.files.inputs(),
    )


def test_real_wire_probe_binds_release_pod_nodes_and_rejections_but_not_custody(live):
    proof = run_proof(live)
    assert proof["verdict"] == "PASS"
    assert proof["scope"] == "deployed_command_and_result_signature_rejection"
    assert proof["installation_custody_proved"] is False
    assert proof["rotated_key_activation_proved"] is False
    assert proof["identity"]["cpu_pod"]["uid"] == "pod-uid-a"
    assert {
        name: item["node_uid"] for name, item in proof["identity"]["nodes"].items()
    } == {"node-a": "uid-node-a", "node-b": "uid-node-b"}
    assert proof["key_source"]["uid"] == "cpu-key-uid"
    assert len(live.api_requests) == 6 and len(live.pair.requests) == 6
    assert [item["status"] for item in proof["protocol"]["observations"]] == [
        404,
        404,
        403,
        401,
        401,
        404,
    ]
    assert all(
        event[0] == "registry-read" or event[1] in {"get", "exec"}
        for event in live.events
    ), "the subproof must not mutate Kubernetes, Store, Secrets or nodes"
    for secret in (KEY_A, KEY_B, API_TOKEN):
        assert secret not in repr(proof) and secret not in repr(live.events)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("release_state", "release_id"), "foreign"),
        (("release_state", "phase"), "rolling"),
        (("release_state", "transaction_committed"), False),
        (("release_state", "release_delivery_sha256"), "0" * 64),
        (("release_state", "bundle_sha256"), "0" * 64),
        (("release_state", "runtime_image"), "registry/other:latest"),
        (("release_state", "node_template_sha256"), "bad"),
        (("release_state", "agent_config_digest"), "bad"),
        (("release_metadata", "namespace"), "other"),
        (("release_metadata", "name"), "other"),
        (("release_metadata", "uid"), ""),
        (("release_metadata", "deletionTimestamp"), "2026-01-01"),
        (("registration", "cluster_id"), "foreign"),
        (("registration", "enabled"), False),
        (("registration", "lifecycle_state"), "PENDING"),
        (("registration", "agent_endpoint_allowed_cidrs"), []),
        (("registration", "agent_endpoint_allowed_cidrs"), "10.0.1.0/24"),
        (("registration", "agent_endpoint_allowed_cidrs"), [None]),
        (("registration", "agent_endpoint_allowed_cidrs"), ["10.0.2.0/24"]),
    ],
)
def test_unbound_release_or_registration_stops_before_node_requests(live, path, value):
    live.raw[path[0]][path[1]] = value
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == [], (
        "unbound release or tenant identity forbids probing a node"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("module_digest", "0" * 64),
        ("deployment_mode", "single-cluster"),
        ("required_agent_artifact_sha256", "0" * 64),
        ("required_agent_compatibility_digest", "0" * 64),
        ("required_agent_protocol_version", True),
        ("required_agent_protocol_version", 999),
        ("required_agent_config_digest", "0" * 64),
        ("required_runtime_profile_version", "other"),
        ("required_node_action_key_version", True),
        ("required_node_action_key_version", 1),
    ],
)
def test_running_cpu_api_must_expose_the_signed_component_and_exact_required_pins(
    live, field, value
):
    live.raw["agent_snapshot"]["version"][field] = value
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", "foreign"),
        ("node_id", "foreign"),
        ("node_action_key_version", 1),
        ("agent_protocol_version", 999),
        ("artifact_sha256", "0" * 64),
        ("compatibility_digest", "0" * 64),
        ("installer_bundle_sha256", "0" * 64),
        ("installer_template_sha256", "0" * 64),
        ("config_digest", "0" * 64),
        ("runtime_profile_version", "other"),
        ("lifecycle_state", "FAILED"),
        ("boot_id", ""),
        ("agent_incarnation_id", None),
        ("node_instance_id", "other"),
        ("tls_certificate_pem", None),
        ("lease_expires_at", None),
        ("generation", True),
        ("endpoint", "http://10.0.1.2:9099"),
        ("endpoint", "https://10.0.1.99:9099"),
        ("allowed_operations", ["FREEZE_EVIDENCE"]),
    ],
)
def test_foreign_stale_or_unsafe_agent_cannot_become_a_signature_target(
    live, field, value
):
    live.raw["agent_snapshot"]["agents"]["node-b"][field] = value
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


@pytest.mark.parametrize(
    "defect", ["naive-seen", "naive-lease", "stale", "future", "short-lease"]
)
def test_agent_freshness_and_lease_budget_are_required(live, defect):
    now = datetime.now(timezone.utc)
    record = live.raw["agent_snapshot"]["agents"]["node-b"]
    field, value = {
        "naive-seen": ("last_seen_at", now.replace(tzinfo=None)),
        "naive-lease": ("lease_expires_at", now.replace(tzinfo=None)),
        "stale": ("last_seen_at", now - timedelta(seconds=61)),
        "future": ("last_seen_at", now + timedelta(seconds=60)),
        "short-lease": ("lease_expires_at", now + timedelta(seconds=29)),
    }[defect]
    record[field] = value.isoformat()
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


@pytest.mark.parametrize(
    "defect",
    [
        "pod-namespace",
        "pod-image",
        "pod-image-id",
        "pod-container-id",
        "pod-not-running",
        "pod-label",
        "node-label",
        "node-provider",
        "node-boot",
        "node-deleting",
        "node-not-ready",
        "node-uid",
        "duplicate-node",
        "missing-node",
        "continued-list",
    ],
)
def test_physical_and_container_inventory_is_not_inferred_from_names(live, defect):
    pod = live.raw["pod"]
    node = live.raw["nodes"]["items"][1]
    if defect == "pod-namespace":
        pod["metadata"]["namespace"] = "foreign"
    elif defect == "pod-image":
        pod["spec"]["containers"][0]["image"] = "other@sha256:" + "0" * 64
    elif defect == "pod-image-id":
        pod["status"]["containerStatuses"][0]["imageID"] = "other@sha256:" + "0" * 64
    elif defect == "pod-container-id":
        pod["status"]["containerStatuses"][0]["containerID"] = ""
    elif defect == "pod-not-running":
        pod["status"]["containerStatuses"][0]["state"] = {}
    elif defect == "pod-label":
        pod["metadata"]["labels"]["app"] = "other"
    elif defect == "node-label":
        node["metadata"]["labels"] = {}
    elif defect == "node-provider":
        node["spec"]["providerID"] = "other"
    elif defect == "node-boot":
        node["status"]["nodeInfo"]["bootID"] = "other"
    elif defect == "node-deleting":
        node["metadata"]["deletionTimestamp"] = "2026-01-01"
    elif defect == "node-not-ready":
        node["status"]["conditions"] = []
    elif defect == "node-uid":
        node["metadata"]["uid"] = ""
    elif defect == "duplicate-node":
        live.raw["nodes"]["items"][1] = live.raw["nodes"]["items"][0]
    elif defect == "missing-node":
        live.raw["nodes"]["items"].pop()
    else:
        live.raw["nodes"]["metadata"] = {"continue": "more"}
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


@pytest.mark.parametrize("field", ["uid", "resourceVersion", "data", "namespace"])
def test_unproven_cpu_key_source_is_not_used(live, field):
    if field == "data":
        live.current_key_document["data"]["node-a"] = "changed"
    elif field == "namespace":
        live.key_document["metadata"][field] = "foreign"
    else:
        live.current_key_document["metadata"][field] = "changed"
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


@pytest.mark.parametrize(
    "defect", ["pod", "node", "agent", "release", "key", "artifact"]
)
def test_identity_drift_after_rejections_prevents_a_passing_subproof(live, defect):
    def change():
        if defect == "pod":
            live.raw["pod"]["metadata"]["uid"] = "replacement-pod"
        elif defect == "node":
            live.raw["nodes"]["items"][1]["metadata"]["uid"] = "replacement-node"
        elif defect == "agent":
            live.raw["agent_snapshot"]["agents"]["node-b"]["generation"] += 1
        elif defect == "release":
            live.raw["release_metadata"]["uid"] = "replacement-state"
        elif defect == "key":
            live.current_key_document["metadata"]["resourceVersion"] = "2"
        else:
            live.files.bundle_path.write_text('{"changed":true}')

    live.after_capture = change
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert len(live.pair.requests) == 6, (
        "identity rechecks follow the real rejection sequence"
    )


def test_two_node_names_cannot_hide_one_physical_node(live):
    live.raw["nodes"]["items"][1]["metadata"]["uid"] = live.raw["nodes"]["items"][0][
        "metadata"
    ]["uid"]
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert live.pair.requests == []


def test_low_level_snapshot_binding_rejects_an_extra_agent(live):
    raw = copy.deepcopy(live.raw)
    raw["agent_snapshot"]["agents"]["node-c"] = raw["agent_snapshot"]["agents"][
        "node-a"
    ]
    verified = release.verify_release_inputs(live.files.inputs())
    with pytest.raises(Auth015ProofError):
        bindings.bind_snapshot(
            raw, release=verified, target=live.target, nodes=("node-a", "node-b")
        )


@pytest.mark.parametrize("defect", ["not-ready", "ambiguous", "changed-name"])
def test_cpu_pod_selection_refuses_missing_or_ambiguous_identity(
    live, monkeypatch, defect
):
    original = live.cpu
    if defect == "not-ready":
        live.raw["pod"]["status"]["phase"] = "Pending"
    elif defect == "changed-name":
        live.after_capture = lambda: live.raw["pod"]["metadata"].update(name="api-b")
    else:
        import json

        def cpu(*arguments, **kwargs):
            value = original(*arguments, **kwargs)
            if arguments[:2] == ("get", "pod"):
                document = json.loads(value)
                other = copy.deepcopy(document["items"][0])
                other["metadata"]["uid"] = "ambiguous-pod"
                document["items"].append(other)
                return json.dumps(document)
            return value

        monkeypatch.setattr(live, "cpu", cpu)
    with pytest.raises(Auth015ProofError):
        run_proof(live)
    assert len(live.pair.requests) == (6 if defect == "changed-name" else 0)


def test_same_key_uid_and_data_do_not_excuse_a_foreign_object_kind(live):
    live.current_key_document["kind"] = "ConfigMap"
    with pytest.raises(Auth015ProofError, match="key source identity changed"):
        run_proof(live)
    assert live.pair.requests == []


def test_cpu_probe_needs_only_shipped_models_and_embedded_stdlib_transport(live):
    live.restrict_cpu_imports = True
    proof = run_proof(live)
    assert proof["verdict"] == "PASS"
    assert len(live.api_requests) == 6
