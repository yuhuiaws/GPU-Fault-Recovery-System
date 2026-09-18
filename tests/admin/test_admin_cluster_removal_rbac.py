from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_rbac import (
    delete_recorded_workload_namespace_rbac,
    inspect_workload_namespace_rbac,
)
from gpu_fault.admin.site import load_site
from gpu_fault_release.regional_deployment_inventory import (
    GPU_EXECUTOR_DEPLOYMENT,
    GPU_WATCHER_DEPLOYMENT,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    EXECUTOR_SERVICE_ACCOUNT,
    WATCHER_SERVICE_ACCOUNT,
    WORKLOAD_NAMESPACE_RBAC_LABEL,
    render_workload_namespace_rbac,
)
from tests.admin.test_admin_site import site_file


def with_gpu_kubeconfig(path):
    document = yaml.safe_load(path.read_text())
    document["spec"]["gpuKubeconfig"] = str(path.parent / "gpu.kubeconfig")
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_site(path)


class RbacApi:
    def __init__(self, site, target):
        self.site, self.target = site, target
        self.objects = {}
        self.requests = []
        self.deleted = []
        self.before_delete = None
        self.before_get = None
        self.lost_ack = False
        self.proof = None
        self.checkpoints = []
        self.namespace = site.release_config["namespace"]
        self.kubectl = [
            "kubectl",
            "--kubeconfig",
            str(
                site.release_config.get("gpu_kubeconfig")
                or site.environment["KUBECONFIG"]
            ),
            "--context",
            target["context"],
        ]
        self.expected = [
            item
            for group in render_workload_namespace_rbac(
                target["allowed_namespaces"], system_namespace=self.namespace
            ).values()
            for item in group
        ]
        for namespace in {
            self.namespace,
            *(item["metadata"]["namespace"] for item in self.expected),
        }:
            self.put("Namespace", "", namespace, apiVersion="v1")
        self.objects[("Namespace", "", self.namespace)]["metadata"]["uid"] = target[
            "expected_namespace_uid"
        ]
        for account, deployment in (
            (EXECUTOR_SERVICE_ACCOUNT, GPU_EXECUTOR_DEPLOYMENT),
            (WATCHER_SERVICE_ACCOUNT, GPU_WATCHER_DEPLOYMENT),
        ):
            self.put("ServiceAccount", self.namespace, account, apiVersion="v1")
            if account == EXECUTOR_SERVICE_ACCOUNT:
                self.objects[("ServiceAccount", self.namespace, account)]["metadata"][
                    "annotations"
                ] = {"eks.amazonaws.com/role-arn": target["executor_irsa_role_arn"]}
            self.put(
                "Deployment",
                self.namespace,
                deployment,
                apiVersion="apps/v1",
                spec={"template": {"spec": {"serviceAccountName": account}}},
            )
        for document in self.expected:
            item = deepcopy(document)
            metadata = item["metadata"]
            key = item["kind"], metadata["namespace"], metadata["name"]
            metadata.update(uid="uid-" + "-".join(key), resourceVersion="1")
            self.objects[key] = item

    def put(self, kind, namespace, name, **fields):
        key = kind, namespace, name
        self.objects[key] = {
            "kind": kind,
            **fields,
            "metadata": {
                "name": name,
                "namespace": namespace,
                "uid": "uid-" + "-".join(key),
                "resourceVersion": "1",
            },
        }

    @staticmethod
    def key_name(key):
        kind, namespace, name = key
        return f"{namespace}/{kind}/{name}"

    def add_manual_role(self):
        namespace = self.expected[0]["metadata"]["namespace"]
        self.put(
            "Role",
            namespace,
            "operator-custom",
            apiVersion="rbac.authorization.k8s.io/v1",
            rules=[{"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get"]}],
        )
        key = "Role", namespace, "operator-custom"
        self.objects[key]["metadata"]["labels"] = {
            WORKLOAD_NAMESPACE_RBAC_LABEL: "true"
        }
        return key

    def result(self, arguments, value=None, *, code=0):
        return subprocess.CompletedProcess(
            arguments,
            code,
            "" if value is None else json.dumps(value),
            "" if not code else "modeled Kubernetes refusal",
        )

    def __call__(self, arguments, **kwargs):
        assert arguments[:5] == [
            "kubectl",
            "--kubeconfig",
            str(
                self.site.release_config.get("gpu_kubeconfig")
                or self.site.environment["KUBECONFIG"]
            ),
            "--context",
            self.target["context"],
        ]
        if "get" in arguments:
            index = arguments.index("get")
            kind = arguments[index + 1]
            if kind == "roles,rolebindings":
                return self.result(
                    arguments,
                    {
                        "kind": "List",
                        "items": [
                            deepcopy(item)
                            for key, item in self.objects.items()
                            if key[0] in {"Role", "RoleBinding"}
                        ],
                    },
                )
            kind = {
                "namespace": "Namespace",
                "role": "Role",
                "rolebinding": "RoleBinding",
                "serviceaccount": "ServiceAccount",
                "deployment": "Deployment",
            }[kind]
            namespace = (
                "" if kind == "Namespace" else arguments[arguments.index("-n") + 1]
            )
            key = kind, namespace, arguments[index + 2]
            if self.before_get:
                self.before_get(key)
            return self.result(arguments, self.objects.get(key))
        assert "delete" in arguments and "--raw" in arguments
        namespace, plural, name = [
            unquote(item)
            for item in arguments[arguments.index("--raw") + 1].split("/")[-3:]
        ]
        kind = {"roles": "Role", "rolebindings": "RoleBinding"}[plural]
        key = kind, namespace, name
        options = json.loads(kwargs["input_text"])
        self.requests.append((key, options))
        if self.before_delete:
            self.before_delete(key)
        current = self.objects.get(key)
        if current is None:
            return self.result(arguments, code=1)
        metadata = current["metadata"]
        if options["preconditions"] != {
            "uid": metadata["uid"],
            "resourceVersion": metadata["resourceVersion"],
        }:
            return self.result(arguments, code=1)
        assert options["propagationPolicy"] == "Foreground"
        self.objects.pop(key)
        self.deleted.append(key)
        return self.result(arguments, code=1 if self.lost_ack else 0)


@pytest.fixture
def scenario(tmp_path):
    site = with_gpu_kubeconfig(site_file(tmp_path))
    target = {
        **site.release_config["clusters"][0],
        "expected_namespace_uid": "namespace-original",
        "expected_hyperpod_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-original",
    }
    api = RbacApi(site, target)
    directory = tmp_path / "rbac"
    directory.mkdir()
    return site, target, api, directory


def execute(scenario, *, verify_only=False):
    site, target, api, _directory = scenario
    if api.proof is None:
        api.proof = inspect_workload_namespace_rbac(
            site.release_config, target, run=api, kubectl=api.kubectl
        )
    assert api.proof is not None
    if verify_only:
        assert set(api.proof["removed"]) == set(api.proof["resources"])
    return delete_recorded_workload_namespace_rbac(
        site.release_config,
        target,
        api.proof,
        run=api,
        kubectl=api.kubectl,
        checkpoint=lambda: api.checkpoints.append(deepcopy(api.proof)),
    )


@pytest.mark.parametrize("unlabelled_role", [False, True])
def test_empty_owned_listing_needs_no_deployment_identity_or_files(
    tmp_path, unlabelled_role
):
    calls = []
    items = (
        [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {
                    "name": "operator-custom",
                    "namespace": "training",
                    "uid": "foreign-role",
                    "resourceVersion": "1",
                    "labels": {"app": "operator"},
                },
                "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
            }
        ]
        if unlabelled_role
        else []
    )

    def run(arguments, **_kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"kind": "List", "items": items}), ""
        )

    assert (
        inspect_workload_namespace_rbac(
            {}, {}, run=run, kubectl=["kubectl", "--context", "gpu"]
        )
        is None
    )
    assert len(calls) == 1
    assert "roles,rolebindings" in calls[0]
    assert list(tmp_path.iterdir()) == []


def test_inspection_captures_namespace_uid_for_unbound_collector_target(scenario):
    site, _, api, _ = scenario
    target = dict(site.release_config["clusters"][0])
    assert "expected_namespace_uid" not in target
    proof = inspect_workload_namespace_rbac(
        site.release_config, target, run=api, kubectl=api.kubectl
    )
    assert proof is not None
    assert proof["binding"]["system_namespace_uid"] == "namespace-original"
    assert api.requests == []
    checkpoints = []
    removed = delete_recorded_workload_namespace_rbac(
        site.release_config,
        {**target, "expected_namespace_uid": "namespace-original"},
        proof,
        run=api,
        kubectl=api.kubectl,
        checkpoint=lambda: checkpoints.append(deepcopy(proof["removed"])),
    )
    assert set(removed) == set(proof["resources"])
    assert set(checkpoints[-1]) == set(proof["resources"])
    assert len(checkpoints) == len(api.expected)


@pytest.mark.parametrize(
    "code,output", [(1, '{"kind":"List","items":[]}'), (0, "{}"), (0, "not-json")]
)
def test_failed_or_malformed_listing_is_not_empty_inventory(code, output):
    with pytest.raises(BootstrapError):
        inspect_workload_namespace_rbac(
            {},
            {},
            run=lambda args, **_kw: subprocess.CompletedProcess(args, code, output, ""),
            kubectl=["kubectl", "--context", "gpu"],
        )


def test_labelled_candidates_require_full_approved_configuration(scenario):
    _, _, api, _ = scenario
    with pytest.raises(BootstrapError, match="allowlist"):
        inspect_workload_namespace_rbac({}, {}, run=api, kubectl=api.kubectl)
    assert api.deleted == []


def test_missing_original_is_checkpointed_only_after_successful_absence(scenario):
    site, target, api, _ = scenario
    api.proof = inspect_workload_namespace_rbac(
        site.release_config, target, run=api, kubectl=api.kubectl
    )
    original = next(key for key in api.objects if key[0] == "RoleBinding")
    api.objects.pop(original)
    execute(scenario)
    assert original not in api.deleted
    assert api.key_name(original) in api.proof["removed"]
    assert len(api.checkpoints) == len(api.expected)


def test_checkpoint_failure_stops_deletion_and_replay_does_not_repeat_it(scenario):
    site, target, api, _ = scenario
    proof = inspect_workload_namespace_rbac(
        site.release_config, target, run=api, kubectl=api.kubectl
    )
    assert proof is not None
    durable = deepcopy(proof)

    def failed_checkpoint():
        raise OSError("journal unavailable")

    with pytest.raises(OSError, match="journal unavailable"):
        delete_recorded_workload_namespace_rbac(
            site.release_config,
            target,
            proof,
            run=api,
            kubectl=api.kubectl,
            checkpoint=failed_checkpoint,
        )
    assert len(api.deleted) == 1
    api.proof = durable
    execute(scenario)
    assert len(api.deleted) == len(api.expected)
    assert len(set(api.deleted)) == len(api.expected)


def test_pruning_preserves_foreign_labelled_roles_and_deletes_bound_pairs(scenario):
    _, _, api, directory = scenario
    manual = api.add_manual_role()
    preserved = deepcopy(api.objects[manual])
    expected = {
        (item["kind"], item["metadata"]["namespace"], item["metadata"]["name"])
        for item in api.expected
    }

    def assert_prepared(_key):
        record = api.proof
        assert len(record["resources"]) == len(expected)
        assert record["anchors"]["service_accounts"]

    api.before_delete = assert_prepared
    result = execute(scenario)
    assert set(api.deleted) == expected
    assert {key[1] for key in api.deleted} == {
        "gpu-fault-system",
        "training",
        "kube-system",
    }
    assert set(result) == {api.key_name(key) for key in expected}
    assert api.objects[manual] == preserved
    kinds = [key[0] for key in api.deleted]
    assert kinds == sorted(kinds, key=lambda kind: kind != "RoleBinding")
    assert set(api.proof["removed"]) == set(api.proof["resources"])
    assert len(api.checkpoints) == len(expected)
    assert list(directory.iterdir()) == []
    count = len(api.requests)
    assert execute(scenario, verify_only=True) == result
    assert len(api.requests) == count


@pytest.mark.parametrize(
    "drift",
    [
        "rules",
        "subject",
        "role-ref",
        "namespace",
        "service-account",
        "deployment",
        "shared",
    ],
)
def test_all_ownership_checks_precede_any_rbac_deletion(scenario, drift):
    _, target, api, _ = scenario
    binding = next(key for key in api.objects if key[0] == "RoleBinding")
    role = "Role", binding[1], binding[2]
    if drift == "rules":
        api.objects[role]["rules"].append(
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get"]}
        )
    elif drift == "subject":
        api.objects[binding]["subjects"][0]["namespace"] = "foreign"
    elif drift == "role-ref":
        api.objects[binding]["roleRef"]["kind"] = "ClusterRole"
    elif drift == "namespace":
        api.objects[("Namespace", "", api.namespace)]["metadata"]["uid"] = "replacement"
    elif drift == "service-account":
        api.objects[("ServiceAccount", api.namespace, EXECUTOR_SERVICE_ACCOUNT)][
            "metadata"
        ]["annotations"]["eks.amazonaws.com/role-arn"] = "foreign-role"
    elif drift == "deployment":
        api.objects[("Deployment", api.namespace, GPU_EXECUTOR_DEPLOYMENT)]["spec"][
            "template"
        ]["spec"]["serviceAccountName"] = "foreign"
    else:
        shared = deepcopy(api.objects[binding])
        shared["metadata"].update(name="operator-shared", uid="foreign-binding")
        shared["metadata"].pop("labels")
        shared["subjects"] = [{"kind": "User", "name": "operator"}]
        api.objects[("RoleBinding", binding[1], "operator-shared")] = shared
    before = deepcopy(api.objects)
    with pytest.raises(BootstrapError):
        execute(scenario)
    assert api.requests == []
    assert api.objects == before
    assert target["expected_namespace_uid"] == "namespace-original"


def test_partial_pruning_resumes_with_original_uids(scenario):
    _, _, api, directory = scenario

    def fail_second(_key):
        if len(api.requests) == 2:
            raise BootstrapError("interrupted pruning")

    api.before_delete = fail_second
    with pytest.raises(BootstrapError, match="interrupted"):
        execute(scenario)
    record = deepcopy(api.checkpoints[-1])
    assert len(record["removed"]) == 1
    api.before_delete = None
    execute(scenario)
    assert len(api.deleted) == len(api.expected)
    assert len(set(api.deleted)) == len(api.expected)
    assert api.proof["resources"] == record["resources"]
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_delete_request_is_conditioned_on_uid_and_checked_version(scenario, field):
    _, _, api, _ = scenario

    def replace_at_delete(key):
        api.objects[key]["metadata"][field] = "changed"

    api.before_delete = replace_at_delete
    with pytest.raises(BootstrapError):
        execute(scenario)
    assert len(api.requests) == 1
    assert api.deleted == []
    assert set(api.requests[0][1]["preconditions"]) == {"uid", "resourceVersion"}


def test_lost_delete_ack_is_accepted_only_after_confirmed_absence(scenario):
    _, _, api, _ = scenario
    api.lost_ack = True
    assert len(execute(scenario)) == len(api.expected)
    assert len(api.deleted) == len(api.expected)


def test_completed_prune_never_deletes_a_replacement(scenario):
    _, _, api, _ = scenario
    execute(scenario)
    old = api.deleted[0]
    template = next(
        item
        for item in api.expected
        if (item["kind"], item["metadata"]["namespace"], item["metadata"]["name"])
        == old
    )
    replacement = deepcopy(template)
    replacement["metadata"].update(uid="replacement-uid", resourceVersion="1")
    api.objects[old] = replacement
    requests = len(api.requests)
    with pytest.raises(BootstrapError, match="reappeared"):
        execute(scenario, verify_only=True)
    assert len(api.requests) == requests
    assert api.objects[old] == replacement


def test_replaced_service_account_blocks_remaining_prune(scenario):
    _, _, api, _ = scenario
    reads = []

    def replace_on_recheck(key):
        if key == ("ServiceAccount", api.namespace, EXECUTOR_SERVICE_ACCOUNT):
            reads.append(True)
            if len(reads) == 2:
                api.objects[key]["metadata"]["uid"] = "replacement-sa"

    api.before_get = replace_on_recheck
    with pytest.raises(BootstrapError, match="ownership changed"):
        execute(scenario)
    assert api.requests == []


def test_recreated_foreign_binding_blocks_role_deletion(scenario):
    _, _, api, _ = scenario
    injected = []

    def inject_before_role(key):
        if key[0] != "Role" or injected:
            return
        binding_key = "RoleBinding", key[1], key[2]
        template = next(
            item
            for item in api.expected
            if (item["kind"], item["metadata"]["namespace"], item["metadata"]["name"])
            == binding_key
        )
        binding = deepcopy(template)
        binding["metadata"].update(uid="new-foreign-binding", resourceVersion="1")
        binding["subjects"] = [{"kind": "User", "name": "operator"}]
        api.objects[binding_key] = binding
        injected.append(binding_key)

    api.before_get = inject_before_role
    with pytest.raises(BootstrapError, match="foreign binding"):
        execute(scenario)
    assert injected, "the test must recreate a binding before Role deletion"
    assert all(key[0] == "RoleBinding" for key in api.deleted), (
        "a recreated foreign binding must stop cleanup before any Role is deleted"
    )
    assert api.objects[injected[0]]["metadata"]["uid"] == "new-foreign-binding"


def test_recorded_proof_cannot_omit_service_account_ownership(scenario):
    _, _, api, _ = scenario
    api.before_delete = lambda _key: (_ for _ in ()).throw(
        BootstrapError("pause before deletion")
    )
    with pytest.raises(BootstrapError, match="pause"):
        execute(scenario)
    api.proof["anchors"]["service_accounts"] = {}
    api.before_delete = None
    requests = len(api.requests)
    with pytest.raises(BootstrapError, match="proof is invalid"):
        execute(scenario)
    assert len(api.requests) == requests
    assert api.deleted == []


def test_label_does_not_expand_the_recorded_namespace_scope(scenario):
    _, _, api, _ = scenario
    template = deepcopy(api.expected[0])
    template["metadata"].update(
        namespace="unapproved", uid="outside-scope", resourceVersion="1"
    )
    key = template["kind"], "unapproved", template["metadata"]["name"]
    api.objects[key] = template
    with pytest.raises(BootstrapError, match="outside the recorded namespace"):
        execute(scenario)
    assert api.requests == []
    assert api.objects[key] == template


def exercise_real_pruner_in_removal(tmp_path: Path, monkeypatch) -> None:
    from tests.admin._cluster_removal_support import RemovalScenario

    outer = RemovalScenario(tmp_path, monkeypatch)
    site = with_gpu_kubeconfig(outer.path)
    target = {
        **site.release_config["clusters"][0],
        "expected_namespace_uid": outer.namespace_uid,
        "expected_hyperpod_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-a-id",
    }
    api = RbacApi(site, target)
    manual = api.add_manual_role()
    monkeypatch.setattr(removal, "run_command", api)
    accepted_claim = [True]
    executor_stopped = [False]

    def cleanup(request, selected, directory):
        assert "drain" in outer.calls
        assert api.deleted == []
        api.proof = inspect_workload_namespace_rbac(
            site.release_config, target, run=api, kubectl=api.kubectl
        )
        assert api.proof is not None
        assert len(api.proof["documents"]) == len(api.expected)
        assert manual in api.objects
        path = outer.cleanup(request, selected, directory)
        accepted_claim[0] = False
        executor_stopped[0] = True
        delete_recorded_workload_namespace_rbac(
            site.release_config,
            target,
            api.proof,
            run=api,
            kubectl=api.kubectl,
            checkpoint=lambda: api.checkpoints.append(deepcopy(api.proof)),
        )
        return path

    def assert_draining(_key):
        assert "drain" in outer.calls
        assert "cleanup" in outer.calls
        assert not accepted_claim[0]
        assert executor_stopped[0]

    api.before_delete = assert_draining
    monkeypatch.setattr(removal, "_run_kubernetes_cleanup", cleanup)
    result = removal.remove_cluster(outer.request())
    assert result["phase"] == "COMPLETED"
    assert outer.calls.count("cleanup") == 1
    assert len(api.deleted) == len(api.expected)
    assert manual in api.objects
