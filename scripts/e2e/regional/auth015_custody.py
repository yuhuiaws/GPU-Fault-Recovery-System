"""Independent, read-only runtime witness for prospectively provisioned keys."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.node_key_custody import (
    key_digests,
    namespace_uid,
    witness_source_identity,
)
from gpu_fault.admin.node_key_custody_chain import verify_chain
from gpu_fault.admin.node_key_custody_crypto import read_regular, write_once
from gpu_fault.admin.node_key_custody_models import (
    Activated,
    Chain,
    CustodyBinding,
    CustodyError,
    ReleaseBinding,
    RuntimeNode,
    Transaction,
    canonical,
    statement_sha256,
)
from scripts.e2e.regional.auth015_bindings import BoundSnapshot, objects
from scripts.e2e.regional.auth015_custody_inputs import (
    Auth015CustodyInputs,
    custody_input_identity,
    retired_key,
    verify_custody_inputs,
)
from scripts.e2e.regional.auth015_deployed import (
    capture_snapshot,
    prove_deployed_protocol,
)
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from scripts.e2e.regional.auth015_release import (
    Auth015ReleaseInputs,
    VerifiedAuth015Release,
    verify_release_inputs,
)
from scripts.e2e.regional.identity_acceptance_common import (
    ClusterTarget,
    IdentityCaseFailure,
    IdentitySite,
    secret_document,
)

KEY_SECRET = "gpu-fault-node-action-keys"
ROTATION_MARKER = "gpu-fault.io/node-action-key-rotation"


def witness_identity() -> str:
    return witness_source_identity(Path(__file__).resolve().parents[3])


def live_anchors(
    site: IdentitySite, target: ClusterTarget, binding: CustodyBinding
) -> None:
    expected = binding.site
    if (
        site.config.get("site_name") != expected.site_name
        or site.config.get("cpu_eks_arn") != expected.cpu_eks_arn
        or site.region != expected.region
        or target.region != expected.region
        or target.eks_cluster_arn != expected.gpu_eks_arn
        or target.cluster_id != expected.cluster_id
        or target.hyperpod_cluster_name != expected.hyperpod_cluster
        or site.namespace != expected.namespace
    ):
        raise CustodyError("AUTH015 site differs from the authorized custody site")
    for plane in ("cpu", "gpu"):
        for name, field in (
            ("kube-system", plane + "_cluster_uid"),
            (site.namespace, plane + "_namespace_uid"),
        ):
            arguments = ("get", "namespace", name, "-o", "json")
            output = (
                site.cpu(*arguments) if plane == "cpu" else site.gpu(target, *arguments)
            )
            if namespace_uid(json.loads(output), name) != getattr(expected, field):
                raise CustodyError("AUTH015 custody cluster incarnation differs")
    inventory = json.loads(
        site.gpu(
            target,
            "get",
            "nodes",
            "-l",
            "sagemaker.amazonaws.com/cluster-name=" + target.hyperpod_cluster_name,
            "-o",
            "json",
        )
    )
    items = objects(inventory, "Node")
    nodes = {}
    for item in items:
        metadata = item["metadata"]
        if (
            item.get("apiVersion") != "v1"
            or item.get("kind") != "Node"
            or metadata.get("deletionTimestamp")
            or metadata["name"] in nodes
            or metadata.get("labels", {}).get("sagemaker.amazonaws.com/cluster-name")
            != target.hyperpod_cluster_name
        ):
            raise CustodyError("AUTH015 custody Node inventory is ambiguous")
        nodes[metadata["name"]] = metadata["uid"]
    if nodes != binding.nodes:
        raise CustodyError("AUTH015 custody Node UID inventory differs")


def live_keys(
    site: IdentitySite, target: ClusterTarget, head: Transaction
) -> dict[str, Any]:
    observed = {}
    for plane in ("gpu", "cpu"):
        expected = getattr(head.completed.statement, plane)
        document = secret_document(site, plane, target, KEY_SECRET)
        metadata = document["metadata"]
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") != "Secret"
            or document.get("type") != "Opaque"
            or document.get("stringData") is not None
            or metadata.get("name") != KEY_SECRET
            or metadata.get("namespace") != site.namespace
            or metadata.get("uid") != expected.uid
            or not metadata.get("resourceVersion")
            or metadata.get("deletionTimestamp")
            or ROTATION_MARKER in (metadata.get("annotations") or {})
        ):
            raise CustodyError("AUTH015 custody key source is unbound or pending")
        digests = key_digests(document["data"])
        wanted = {node: state.sha256 for node, state in expected.keys.items()}
        if (
            plane == "gpu"
            and set(digests) != set(wanted)
            or any(digests.get(node) != value for node, value in wanted.items())
        ):
            raise CustodyError(
                "AUTH015 deployed key bytes differ from the signed receipt"
            )
        observed[plane] = document
    return observed["cpu"]


def bind_runtime(
    snapshot: BoundSnapshot,
    release: VerifiedAuth015Release,
    chain: Chain,
    nodes: tuple[str, str],
) -> None:
    head = chain.transactions[-1]
    authorization = head.authorization.statement
    expected = authorization.binding
    if set(nodes) - set(expected.nodes) or len(set(nodes)) != 2:
        raise CustodyError("AUTH015 nodes differ from the custody selection")
    for name, state in snapshot.binding["nodes"].items():
        observed = ReleaseBinding(
            release_id=release.release_id,
            manifest_sha256=release.input_identity["manifest"],
            delivery_sha256=release.delivery_sha256,
            node_wheel_sha256=release.node_wheel_sha256,
            node_digest=release.node_digest,
            bundle_sha256=release.bundle_sha256,
            template_sha256=state["installer_template_sha256"],
            config_digest=state["config_digest"],
            runtime_profile_version=state["runtime_profile_version"],
        )
        if observed != expected.release or state["node_uid"] != expected.nodes[name]:
            raise CustodyError(
                "AUTH015 runtime release or Node identity differs from custody"
            )
        if snapshot.agents[name].last_seen_at <= head.completed.statement.observed_at:
            raise CustodyError(
                "AUTH015 requires fresh authenticated heartbeats after provisioning"
            )
    if authorization.purpose == "rotate":
        previous = chain.transactions[-2].activated
        if (
            authorization.rotate_node != nodes[0]
            or previous is None
            or set(previous.statement.nodes) != set(nodes)
            or runtime_nodes(snapshot)[nodes[1]] != previous.statement.nodes[nodes[1]]
        ):
            raise CustodyError("AUTH015 rotation or sibling runtime identity differs")


def runtime_nodes(snapshot: BoundSnapshot) -> dict[str, RuntimeNode]:
    return {
        name: RuntimeNode(**{field: value[field] for field in RuntimeNode.model_fields})
        for name, value in snapshot.binding["nodes"].items()
    }


def json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def run_custody_acceptance(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    release_inputs: Auth015ReleaseInputs,
    custody_inputs: Auth015CustodyInputs,
    case_dir: Path,
) -> dict[str, Any]:
    stage = "custody-inputs"
    try:
        identity = custody_input_identity(custody_inputs)
        chain, head, crypto = verify_custody_inputs(custody_inputs)
        if head.authorization.statement.witness_sha256 != witness_identity():
            raise CustodyError(
                "AUTH015 witness differs from the independently authorized code"
            )
        stage = "release"
        release = verify_release_inputs(release_inputs)
        if (
            crypto.public_identity(read_regular(release_inputs.public_key_path))
            in crypto.public_keys
        ):
            raise CustodyError(
                "custody authorities must be independent of the release key"
            )
        binding = head.authorization.statement.binding
        stage = "live-identity"
        live_anchors(site, target, binding)
        key_document = live_keys(site, target, head)
        before = capture_snapshot(site, target, nodes, release)
        bind_runtime(before, release, chain, nodes)
        stage = "activation"
        old = retired_key(custody_inputs, chain)
        cross_node = prove_deployed_protocol(
            site,
            target,
            nodes=nodes,
            key_document=key_document,
            release_inputs=release_inputs,
        )
        proof = (
            prove_deployed_protocol(
                site,
                target,
                nodes=(nodes[1], nodes[0]),
                key_document=key_document,
                release_inputs=release_inputs,
                retired_key=old,
            )
            if old is not None
            else cross_node
        )
        if (
            cross_node["verdict"] != "PASS"
            or cross_node["identity"] != before.binding
            or proof["verdict"] != "PASS"
            or proof["identity"] != before.binding
            or proof["supplied_retired_key_denied"] is not (old is not None)
        ):
            raise CustodyError(
                "AUTH015 activation challenge identity or result differs"
            )
        live_anchors(site, target, binding)
        stage = "activation-readback"
        if live_keys(site, target, head) != key_document:
            raise CustodyError(
                "AUTH015 key source changed during activation verification"
            )
        if custody_input_identity(custody_inputs) != identity:
            raise CustodyError("AUTH015 custody inputs changed during the challenge")
        protocol_evidence = {
            "cross_node": cross_node["protocol"],
            "rotated_node": proof["protocol"] if old is not None else None,
        }
        activated = Activated(
            completed_sha256=statement_sha256(head.completed.statement),
            binding_sha256=statement_sha256(binding),
            observed_at=datetime.now(timezone.utc),
            nodes=runtime_nodes(before),
            runtime_identity_sha256=json_sha256(proof["identity"]),
            protocol_sha256=json_sha256(protocol_evidence),
            retired_key_denied=old is not None,
        )
        witness = crypto.sign(activated, "witness")
        stage = "signed-receipt"
        updated = head.model_copy(update={"activated": witness})
        result_chain = Chain(transactions=[*chain.transactions[:-1], updated])
        verify_chain(result_chain, crypto, binding=binding, require_activation=True)
        path = case_dir / (
            "auth015-custody-chain-" + statement_sha256(witness) + ".json"
        )
        write_once(path, canonical(result_chain))
        checks = {
            "installation_time_master_custody": True,
            "deployed_cross_node_command_and_result_signatures": True,
            "deployed_node_a_key_activation": old is not None,
            "independent_signed_activation_receipt": True,
            "node_b_key_unchanged": old is not None,
            "fresh_signed_heartbeats_after_provisioning": True,
            "node_b_agent_continues": True,
            "no_node_or_secret_mutation": True,
        }
        gaps = (
            {}
            if old is not None
            else {
                "deployed_node_a_key_activation": (
                    "Initial activation recorded prospectively; an authorized rotation "
                    "and deployed activation of its next generation are still required."
                )
            }
        )
        return {
            "verdict": "PASS" if not gaps else "FAIL",
            "checks": checks,
            "not_evaluated": gaps,
            "custody_chain_sha256": statement_sha256(result_chain),
            "custody_chain_file": str(path),
            "installation_custody_proved": True,
            "rotated_key_activation_proved": old is not None,
            "activation_receipt": witness.model_dump(mode="json"),
            "deployed_protocol": cross_node,
            "activation_protocol": proof if old is not None else None,
            "protocol_evidence": protocol_evidence,
            "cleanup_errors": [],
            "limitations": [
                "Custody attests the authorized trusted provisioning path, not "
                "unobserved activity before or outside that path.",
                "This read-only witness does not install, rotate or restore keys. "
                "Unproven historical installations require new authorized evidence.",
            ],
        }
    except Exception as exc:
        # Never echo private Secret JSON, credential file bytes or transport errors.
        reason = (
            str(exc)
            if isinstance(exc, (CustodyError, Auth015ProofError))
            else type(exc).__name__
        )
        raise IdentityCaseFailure(
            "AUTH015 custody/activation proof failed closed: " + stage + ": " + reason,
            details={
                "checks": {},
                "cleanup_errors": [],
                "requires_new_authorized_evidence": True,
                "failure_stage": stage,
            },
        ) from None
