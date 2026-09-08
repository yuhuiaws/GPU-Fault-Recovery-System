"""Local proxies for the manual regional least-privilege acceptance cases.

``GF-REGIONAL-BLAST-002/003/004`` and ``GF-REGIONAL-NOTIFY-004`` are the cases
that ask what each identity in this system is *able* to do rather than what it
did. On a live fleet they are answered with ``kubectl auth can-i`` matrices, an
IAM policy review and a Secret sweep, and that stays the run of record: only the
cluster knows which role is actually attached, and only the account knows
whether someone widened it by hand.

What CI can own is the declaration those runs are compared against -- the policy
document this repository creates for each role, the ClusterRole it applies to
each data-plane ServiceAccount, and which manifest is allowed to mention which
credential. A live matrix that stops matching the declaration is a finding; a
declaration that quietly grew a write verb between releases is the same finding
one release earlier, and it is the only half that can be caught before a GPU
cluster is touched.

Each check carries a positive control, because "no forbidden permission found"
is also what a scanner that reads no files reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from gpu_fault.admin.bootstrap_services import (
    control_plane_policy_document,
    executor_policy_document,
)
from gpu_fault.regional_registry import sync_regional_cluster_registry
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_gpu_rollout as ROLLOUT
from gpu_fault_release import regional_release_iam as IAM
from tests._builders import build_store
from tests.regional._regional_support import TOKEN_A, TOKEN_B

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_MANIFEST = ROOT / "deploy/dataplane/cluster-action-executor.yaml"
WATCHER_MANIFEST = ROOT / "deploy/dataplane/completion-watcher.yaml"

REGION = "us-west-2"
ACCOUNT = "123456789012"
SENDER = "gpu-fault@example.com"
HYPERPOD_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-cluster-a"
EXECUTOR_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-cluster-a-executor"

#: Kubernetes verbs that change something. ``*`` is included because a rule that
#: grants it grants all of the others.
WRITE_VERBS = frozenset(
    {"create", "update", "patch", "delete", "deletecollection", "*"}
)

#: The resources a GPU recovery acts on. A control-plane identity holding a write
#: verb on any of them could mutate a cluster directly, which is the property the
#: regional split exists to remove.
GPU_RESOURCES = frozenset({"nodes", "pods", "jobs", "pytorchjobs", "jobsets", "*"})

#: Provider APIs that change node lifecycle. ``BatchReplaceClusterNodes`` heads
#: the list because the whole design forbids it everywhere, but the control plane
#: must not hold even the reboot call that its executors legitimately use.
NODE_MUTATION_ACTIONS = frozenset(
    {
        "sagemaker:batchreplaceclusternodes",
        "sagemaker:batchdeleteclusternodes",
        "sagemaker:rebootclusternodes",
        "sagemaker:batchrebootclusternodes",
        "sagemaker:updatecluster",
        "sagemaker:deletecluster",
        "sagemaker:updateclustersoftware",
    }
)

#: Everything the regional rollout applies to a GPU EKS or installs on a GPU
#: node. ``deploy/hyperpod/`` is deliberately absent: that is the single-cluster
#: topology, where the control plane runs *in* the GPU cluster and therefore does
#: hold the execution token there. Mixing it in would make this scan meaningless
#: for the regional split it is about.
DATA_PLANE_TREES = ("deploy/dataplane", "deploy/node", "deploy/systemd")
CONTROL_PLANE_TREES = ("deploy/control-plane",)

#: How the administrator execution token appears in a manifest: as the variable
#: the process reads, or as the Secret key it is stored under.
EXECUTION_TOKEN_MARKERS = ("GPU_FAULT_EXECUTION_TOKEN", "execution-token")


def documents(path: Path) -> list[dict[str, Any]]:
    return [
        item for item in yaml.safe_load_all(path.read_text(encoding="utf-8")) if item
    ]


def rbac_rules(trees: tuple[str, ...]) -> Iterator[tuple[Path, str, dict[str, Any]]]:
    """Every RBAC rule declared under ``trees``, with the file that declares it."""

    for tree in trees:
        for path in sorted((ROOT / tree).rglob("*.yaml")):
            for document in documents(path):
                if document.get("kind") not in {"Role", "ClusterRole"}:
                    continue
                for rule in document.get("rules") or []:
                    yield path, document["metadata"]["name"], rule


def gpu_write_rules(trees: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    """Rules in ``trees`` that grant a write verb on a GPU recovery resource."""

    found = []
    for path, name, rule in rbac_rules(trees):
        resources = {
            str(item).split("/", 1)[0].lower() for item in rule.get("resources") or []
        }
        verbs = {str(item).lower() for item in rule.get("verbs") or []}
        if resources & GPU_RESOURCES and verbs & WRITE_VERBS:
            found.append((path.name, name, sorted(verbs & WRITE_VERBS)))
    return found


def statement_actions(document: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for statement in document["Statement"]:
        raw = statement.get("Action", [])
        actions.extend(raw if isinstance(raw, list) else [raw])
    return actions


def text_files(trees: tuple[str, ...]) -> list[Path]:
    return [
        path
        for tree in trees
        for path in sorted((ROOT / tree).rglob("*"))
        if path.is_file() and path.suffix in {".yaml", ".yml", ".sh", ".json", ".py"}
    ]


def token_mentions(trees: tuple[str, ...]) -> list[str]:
    mentions = []
    for path in text_files(trees):
        content = path.read_text(encoding="utf-8", errors="ignore")
        for marker in EXECUTION_TOKEN_MARKERS:
            if marker in content:
                mentions.append(f"{path.relative_to(ROOT)}:{marker}")
    return mentions


def cluster_entry(cluster_id: str, token: str) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "region": REGION,
        "hyperpod_cluster_name": f"hp-{cluster_id}",
        "eks_cluster_arn": (f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}"),
        "token": token,
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
    }


def test_blast002_the_control_plane_declares_no_gpu_write_permission() -> None:
    """The code guard against central mutation is only as good as the role.

    ``ApplicationContext`` refuses to enable the HyperPod adapter in regional
    mode, but that is a switch in a process an operator can restart with a
    different value. What makes the split structural is that the control plane's
    role cannot reboot, replace or delete a node even if the code asked it to --
    and that its Kubernetes identity holds no write verb on nodes, pods or jobs
    anywhere in the control-plane manifests.
    """

    observe = control_plane_policy_document(
        region=REGION, account_id=ACCOUNT, email_sender=SENDER
    )
    without_email = control_plane_policy_document(region=REGION, account_id=ACCOUNT)
    actions = [item.lower() for item in statement_actions(observe)]
    email = next(
        item for item in observe["Statement"] if item.get("Sid") == "AdministratorEmail"
    )

    assert [item for item in actions if item.startswith("sagemaker:")] == [
        "sagemaker:describecluster",
        "sagemaker:listclusternodes",
        "sagemaker:describeclusternode",
    ]
    assert NODE_MUTATION_ACTIONS.isdisjoint(actions), sorted(
        NODE_MUTATION_ACTIONS.intersection(actions)
    )
    # Sending is bound to the identity *and* to the From address: the resource
    # alone still lets a compromised control plane send as any verified sender
    # in the account, which is how a spoofed "recovery complete" mail would look
    # legitimate to the operator reading it.
    assert email["Resource"] == (f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/{SENDER}")
    assert email["Condition"] == {"StringEquals": {"ses:FromAddress": SENDER}}
    # No configured sender means no mail permission at all, rather than a
    # permission waiting for a sender to be set later.
    assert [item.lower() for item in statement_actions(without_email)] == [
        item for item in actions if item.startswith("sagemaker:")
    ]

    # Positive control: the same extractor does see a mutation action when the
    # document has one, so the assertion above is about the policy, not the
    # extractor.
    executor_actions = [
        item.lower()
        for item in statement_actions(
            executor_policy_document(hyperpod_arn=HYPERPOD_ARN)
        )
    ]
    assert "sagemaker:batchrebootclusternodes" in executor_actions

    offending = gpu_write_rules(CONTROL_PLANE_TREES)
    scanned = list(rbac_rules(CONTROL_PLANE_TREES))
    data_plane_writes = gpu_write_rules(DATA_PLANE_TREES)

    assert offending == []
    # Both controls: the scan really reads the control-plane tree, and it really
    # recognises a GPU write verb when one is declared -- the data plane is
    # supposed to have them.
    assert scanned, CONTROL_PLANE_TREES
    assert data_plane_writes, DATA_PLANE_TREES


def cluster_role(path: Path, name: str) -> dict[str, Any]:
    return next(
        item
        for item in documents(path)
        if item["kind"] == "ClusterRole" and item["metadata"]["name"] == name
    )


def verb_matrix(role: dict[str, Any]) -> dict[str, set[str]]:
    return {
        resource: set(rule["verbs"])
        for rule in role["rules"]
        for resource in rule["resources"]
    }


def test_blast003_the_executor_rbac_and_iam_stay_minimal() -> None:
    """The executor is the one identity that may mutate, so its list is the fence.

    The ClusterRole holds only what is cluster-scoped by nature: patch on nodes
    for cordon and taint but not delete, and read/list/watch on workload kinds
    because spare activation lists Pods in every namespace. Every namespaced
    write -- pods delete/patch, jobs/pytorchjobs/jobsets create/patch -- lives
    in a Role the rollout renders per ``allowed_namespaces`` entry, so the API
    server enforces the boundary GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES declares.
    No access to secrets, configmaps or RBAC objects anywhere, which is what
    stops a compromised executor from reading another cluster's credentials or
    widening its own binding.
    """

    role = cluster_role(EXECUTOR_MANIFEST, "gpu-fault-cluster-executor")
    binding = next(
        item
        for item in documents(EXECUTOR_MANIFEST)
        if item["kind"] == "ClusterRoleBinding"
    )
    verbs = verb_matrix(role)
    node_rule = next(rule for rule in role["rules"] if rule["resources"] == ["nodes"])

    assert verbs == {
        "nodes": {"get", "list", "watch", "patch"},
        "pods": {"get", "list", "watch"},
        "jobs": {"get", "list", "watch"},
        "pytorchjobs": {"get", "list", "watch"},
        "jobsets": {"get", "list", "watch"},
    }
    assert {"secrets", "configmaps", "clusterroles", "clusterrolebindings"}.isdisjoint(
        verbs
    ), sorted(verbs)
    # The asymmetries are the point, so they are stated rather than left to be
    # read out of the matrix above: no node delete, and no cluster-wide write
    # on anything that lives in a namespace.
    assert "delete" not in verbs["nodes"], sorted(verbs["nodes"])
    assert all(
        verbs[resource].isdisjoint(WRITE_VERBS)
        for resource in ("pods", "jobs", "pytorchjobs", "jobsets")
    ), verbs
    # The fact the live report has to record: Kubernetes RBAC cannot scope patch
    # by label, and resourceNames does not apply to list or watch, so this rule
    # is cluster-wide by construction. The compensating control is not in RBAC --
    # it is that the executor only ever patches node IDs that arrived in a
    # control-plane command bound to a fencing token.
    assert "resourceNames" not in node_rule, sorted(node_rule)
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "gpu-fault-cluster-executor",
            "namespace": "gpu-fault-system",
        }
    ]

    # The write verbs did move rather than vanish: one Role per allowed
    # namespace, bound to the same ServiceAccount, and still no pod create.
    rendered = ROLLOUT.render_workload_namespace_rbac(
        ("training",), system_namespace="gpu-fault-system"
    )[inventory.GPU_EXECUTOR_DEPLOYMENT]
    workload_role = next(
        item
        for item in rendered
        if item["kind"] == "Role" and item["metadata"]["namespace"] == "training"
    )
    namespaced = verb_matrix(workload_role)
    assert namespaced == {
        "pods": {"patch", "delete"},
        "jobs": {"create", "patch"},
        "pytorchjobs": {"create", "patch"},
        "jobsets": {"create", "patch"},
    }
    assert "create" not in namespaced["pods"]
    assert {
        (item["metadata"]["namespace"], tuple(s["name"] for s in item["subjects"]))
        for item in rendered
        if item["kind"] == "RoleBinding"
    } == {
        ("training", ("gpu-fault-cluster-executor",)),
        ("kube-system", ("gpu-fault-cluster-executor",)),
    }

    policy = executor_policy_document(hyperpod_arn=HYPERPOD_ARN)
    IAM.validate_executor_iam_documents(EXECUTOR_ROLE_ARN, [policy])

    # The creator and the release-time validator have to agree: the release gate
    # rejects anything outside its allowlist, so an action added here without
    # being added there would fail every rollout, and an action removed there
    # would let the gate pass a role it no longer describes.
    assert {item.lower() for item in statement_actions(policy)} == set(
        IAM.EXECUTOR_SAGEMAKER_ACTIONS
    )
    # Scoped to one cluster ARN, not to ``cluster/*``: this is what keeps one
    # compromised executor from rebooting a different cluster's nodes.
    assert policy["Statement"][0]["Resource"] == HYPERPOD_ARN


def test_blast003_the_completion_watcher_rbac_stays_read_only_cluster_wide() -> None:
    """The watcher's one exec is scoped to the namespaces it may touch.

    It runs a single Pod watch over every namespace, which RBAC cannot narrow,
    so get/list/watch on Pods stay cluster-wide. Everything it *does* to a
    training Pod -- annotate it, suspend its owner, read its log, and the
    ``nvidia-smi --query-gpu=uuid`` exec behind GPU_FAULT_DISCOVER_POD_GPU_UUIDS
    -- is granted per allowed namespace. ``pods/exec`` in a ClusterRole is a
    shell into every Pod on the cluster, because resourceNames does not apply
    to subresources; that is the shape this test exists to keep out.
    """

    role = cluster_role(WATCHER_MANIFEST, "gpu-fault-completion-watcher")
    binding = next(
        item
        for item in documents(WATCHER_MANIFEST)
        if item["kind"] == "ClusterRoleBinding"
    )
    verbs = verb_matrix(role)

    assert verbs == {
        "pods": {"get", "list", "watch"},
        "jobs": {"get"},
        "pytorchjobs": {"get"},
        "jobsets": {"get"},
        "configmaps": {"get", "update", "patch"},
    }
    assert "pods/exec" not in verbs and "pods/log" not in verbs, sorted(verbs)
    configmap_rule = next(
        rule for rule in role["rules"] if rule["resources"] == ["configmaps"]
    )
    # The only cluster-wide write is its own outbox, pinned by name.
    assert configmap_rule["resourceNames"] == ["gpu-fault-completion-watcher-outbox"]
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "gpu-fault-completion-watcher",
            "namespace": "gpu-fault-system",
        }
    ]

    rendered = ROLLOUT.render_workload_namespace_rbac(
        ("training", "research"), system_namespace="gpu-fault-system"
    )[inventory.GPU_WATCHER_DEPLOYMENT]
    roles = {
        item["metadata"]["namespace"]: verb_matrix(item)
        for item in rendered
        if item["kind"] == "Role"
    }
    assert set(roles) == {"training", "research"}
    assert roles["training"] == {
        "pods": {"patch"},
        "pods/exec": {"get", "create"},
        "pods/log": {"get"},
        "jobs": {"patch"},
        "pytorchjobs": {"patch"},
        "jobsets": {"patch"},
    }
    # The watcher gets no Role in the device-plugin namespace: only the
    # executor restarts plugins.
    assert "kube-system" not in roles
    # Positive control: the extractor reports exec when a rule has it, so the
    # ClusterRole assertion above is about the manifest, not the extractor.
    assert "create" in roles["research"]["pods/exec"]


def test_blast004_the_execution_token_stays_on_the_control_plane() -> None:
    """One leaked administrator token would make every cluster interchangeable.

    The execution token authorises workflow execution for the whole region, so a
    copy of it on a GPU cluster turns that cluster into an operator of all the
    others. The data plane authenticates with a per-cluster token instead, and
    those are stored as digests that differ per cluster -- so one cluster's
    credential proves nothing about another's.
    """

    data_plane = token_mentions(DATA_PLANE_TREES)
    control_plane = token_mentions(CONTROL_PLANE_TREES)

    assert data_plane == []
    # Positive control: the marker really is how the token appears in a manifest,
    # so the empty result above is a property of the data plane and not of the
    # search string.
    assert control_plane, CONTROL_PLANE_TREES

    store = build_store()
    registrations = sync_regional_cluster_registry(
        store,
        [cluster_entry("cluster-a", TOKEN_A), cluster_entry("cluster-b", TOKEN_B)],
    )
    first, second = registrations
    dumped = json.dumps([item.model_dump(mode="json") for item in registrations])

    assert [item.cluster_id for item in registrations] == ["cluster-a", "cluster-b"]
    assert first.token_sha256 != second.token_sha256
    # The registry keeps digests only. A plaintext token in the durable registry
    # would be readable by anything that can read the control plane's database,
    # including a restored snapshot.
    assert TOKEN_A not in dumped
    assert TOKEN_B not in dumped
    assert first.matched_token_slot(TOKEN_A) == "current"
    assert first.matched_token_slot(TOKEN_B) is None
    assert second.matched_token_slot(TOKEN_A) is None


def test_notify004_no_data_plane_identity_can_send_mail() -> None:
    """Mail from the data plane would bypass deduplication, not just tidiness.

    Advisory notifications are deduplicated by key in the control plane, which is
    also where the audit trail of what was sent lives. An executor holding
    ``ses:SendEmail`` could page an operator with no incident record behind it,
    and repeated sends for one fault would be indistinguishable from repeated
    faults -- so the capability exists exactly once, on the CPU side.
    """

    policy = executor_policy_document(hyperpod_arn=HYPERPOD_ARN)
    with_ses = {
        "Version": policy["Version"],
        "Statement": [
            *policy["Statement"],
            {"Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*"},
        ],
    }
    deployment = next(
        item for item in documents(EXECUTOR_MANIFEST) if item["kind"] == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    declared = {item["name"] for item in container["env"]}
    observe = control_plane_policy_document(
        region=REGION, account_id=ACCOUNT, email_sender=SENDER
    )

    assert [
        item for item in statement_actions(policy) if item.lower().startswith("ses:")
    ] == []
    # The release gate is the thing that would catch a role widened by hand after
    # bootstrap, so it has to reject the widening rather than only describe it.
    with pytest.raises(IAM.ReleaseError, match="exceeds the regional"):
        IAM.validate_executor_iam_documents(EXECUTOR_ROLE_ARN, [with_ses])

    # The executor is also given no mail configuration to use, so an accidentally
    # widened role still has no sender or recipient to send to.
    assert [item for item in declared if "EMAIL" in item] == []
    assert "ses:SendEmail" in statement_actions(observe)
