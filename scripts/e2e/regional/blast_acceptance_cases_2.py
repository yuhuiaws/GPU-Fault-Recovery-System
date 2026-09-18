from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release.regional_release_iam import EXECUTOR_SAGEMAKER_ACTIONS
from gpu_fault_release.regional_release_gpu_rollout import (
    render_workload_namespace_rbac,
)
from scripts.e2e.regional.blast_acceptance_base import (
    EXECUTION_TOKEN_NAME,
    FORBIDDEN_EXECUTOR_ACTIONS,
    ROOT,
    CheckError,
    ClusterTarget,
    action_pattern_matches,
    allow_statement_matches,
    resources_for,
    sha256_bytes,
    statement_not_actions,
    write_json,
)
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from scripts.e2e.regional.blast_rbac_scope import (
    RBAC_INVENTORY,
    bound_rules,
    unexpected_grants,
)
from scripts.e2e.regional.credential_value_scan import (
    CredentialScanError,
    credential_value_digests,
)
from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records

EXECUTOR_MANIFEST = ROOT / "deploy" / "dataplane" / "cluster-action-executor.yaml"
EXECUTOR_CLUSTER_ROLE = "gpu-fault-cluster-executor"


def expected_executor_role(
    manifest: Path = EXECUTOR_MANIFEST,
) -> dict[str, list[str]]:
    """The executor ClusterRole as the shipped manifest declares it.

    BLAST-003 compares the live role against the manifest; a hard-coded copy
    of the rules drifted silently whenever the manifest changed, and then the
    case judged the deployment against a role nobody ships.
    """

    documents = yaml.safe_load_all(manifest.read_text(encoding="utf-8"))
    for document in documents:
        if (
            isinstance(document, dict)
            and document.get("kind") == "ClusterRole"
            and (document.get("metadata") or {}).get("name") == EXECUTOR_CLUSTER_ROLE
        ):
            return BlastCasesTwo.normalized_role_rules(document)
    raise CheckError(f"{manifest} declares no ClusterRole {EXECUTOR_CLUSTER_ROLE}")


class BlastCasesTwo(BlastCasesOne):
    def expected_namespace_roles(
        self, target: ClusterTarget
    ) -> dict[str, dict[str, list[str]]]:
        entries = [
            item
            for item in self.config["clusters"]
            if item["cluster_id"] == target.cluster_id
        ]
        if len(entries) != 1 or not isinstance(
            entries[0].get("allowed_namespaces"), list
        ):
            raise CheckError("executor workload namespace policy is missing")
        namespaces = entries[0]["allowed_namespaces"]
        if any(not isinstance(name, str) or not name.strip() for name in namespaces):
            raise CheckError("executor workload namespace policy is invalid")
        documents = render_workload_namespace_rbac(
            namespaces, system_namespace=self.namespace
        )[EXECUTOR_CLUSTER_ROLE]
        result: dict[str, dict[str, list[str]]] = {}
        for document in documents:
            if document["kind"] != "Role":
                continue
            scope = result.setdefault(document["metadata"]["namespace"], {})
            for key, verbs in self.normalized_role_rules(document).items():
                scope[key] = sorted(set(scope.get(key, [])) | set(verbs))
        return result

    @staticmethod
    def normalized_role_rules(role: Mapping[str, Any]) -> dict[str, list[str]]:
        result: dict[str, set[str]] = {}
        for rule in role.get("rules") or []:
            if rule.get("resourceNames") or rule.get("nonResourceURLs"):
                raise CheckError("executor role contains an unsupported rule scope")
            api_groups = rule.get("apiGroups") or [""]
            resources = rule.get("resources") or []
            verbs = rule.get("verbs") or []
            for api_group in api_groups:
                for resource in resources:
                    key = f"{api_group or 'core'}:{resource}"
                    result.setdefault(key, set()).update(str(item) for item in verbs)
        return {key: sorted(value) for key, value in sorted(result.items())}

    def executor_namespace_permissions(
        self,
        target: ClusterTarget,
        *,
        service_account: str,
        verbs: tuple[str, ...],
        sensitive_denied: bool,
    ) -> tuple[dict[str, Any], bool]:
        namespace_scopes = {}
        expected = self.expected_namespace_roles(target)
        resource_keys = {
            "pods": "core:pods",
            "jobs": "batch:jobs",
            "pytorchjobs.kubeflow.org": "kubeflow.org:pytorchjobs",
            "jobsets.jobset.x-k8s.io": "jobset.x-k8s.io:jobsets",
        }
        for namespace in self.namespace_names(target):
            namespaced_sensitive = self.auth_can_i(
                kube_prefix=(
                    "--kubeconfig",
                    self.gpu_kubeconfig,
                    "--context",
                    target.context,
                ),
                service_account=service_account,
                verbs=verbs,
                resources=("secrets", "configmaps"),
                namespace=namespace,
            )
            sensitive_denied = sensitive_denied and not any(
                allowed
                for row in namespaced_sensitive.values()
                for allowed in row.values()
            )
            workload_writes = self.auth_can_i(
                kube_prefix=(
                    "--kubeconfig",
                    self.gpu_kubeconfig,
                    "--context",
                    target.context,
                ),
                service_account=service_account,
                verbs=("create", "patch", "update", "delete"),
                resources=(
                    "pods",
                    "jobs",
                    "pytorchjobs.kubeflow.org",
                    "jobsets.jobset.x-k8s.io",
                ),
                namespace=namespace,
            )
            writes_match = all(
                allowed
                == (
                    verb in expected.get(namespace, {}).get(resource_keys[resource], [])
                )
                for verb, row in workload_writes.items()
                for resource, allowed in row.items()
            )
            namespace_scopes[namespace] = {
                "sensitive": namespaced_sensitive,
                "workload_writes": workload_writes,
                "workload_writes_match_policy": writes_match,
            }
        if set(expected) - set(namespace_scopes):
            raise CheckError("an authorized workload or plugin namespace is missing")
        return namespace_scopes, sensitive_denied

    def executor_iam_scope(
        self, target: ClusterTarget, namespace_scopes: dict[str, Any]
    ) -> dict[str, Any]:
        service_account_doc = self.gpu_json(
            target,
            "-n",
            self.namespace,
            "get",
            "serviceaccount",
            "gpu-fault-cluster-executor",
            "-o",
            "json",
        )
        annotation_role = str(
            service_account_doc.get("metadata", {})
            .get("annotations", {})
            .get("eks.amazonaws.com/role-arn", "")
        )
        role_binding_matches = annotation_role == target.executor_role_arn
        statements, inventory = self.iam_role_policies(target.executor_role_arn)
        hyperpod = self.aws(
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            target.hyperpod_cluster_name,
        )
        hyperpod_arn = hyperpod.get("ClusterArn")
        if not isinstance(hyperpod_arn, str) or not hyperpod_arn:
            raise CheckError("executor IAM audit lacks the actual HyperPod ARN")
        resources_scoped = all(
            not item.get("NotResource") and set(resources_for(item)) == {hyperpod_arn}
            for item in statements
            if str(item.get("Effect", "")).lower() == "allow"
        )
        patterns = self.allowed_action_patterns(statements)
        unexpected_actions = [
            pattern
            for pattern in patterns
            if pattern.lower() not in EXECUTOR_SAGEMAKER_ACTIONS
        ]
        ses_actions = sorted(
            {
                pattern
                for pattern in patterns
                if action_pattern_matches(pattern, "ses:SendEmail")
                or pattern.lower().startswith("ses:")
                or pattern == "*"
            }
        )
        forbidden = [
            action
            for action in FORBIDDEN_EXECUTOR_ACTIONS
            if any(allow_statement_matches(item, action) for item in statements)
        ]
        broad_not_action = any(
            str(item.get("Effect", "")).lower() == "allow"
            and statement_not_actions(item)
            for item in statements
        )
        inventory.update(
            {
                "allowed_action_patterns": patterns,
                "ses_action_patterns": ses_actions,
                "forbidden_provider_mutations_present": forbidden,
                "broad_allow_not_action_present": broad_not_action,
                "service_account_role_matches_site": role_binding_matches,
                "unexpected_action_patterns": unexpected_actions,
                "resources_scoped_to_local_hyperpod": resources_scoped,
                "namespace_permissions": namespace_scopes,
            }
        )
        write_json(
            self.run_dir / f"BLAST-003-iam-{target.cluster_id}.json",
            inventory,
        )

        return {
            "role_binding_matches": role_binding_matches,
            "ses_actions": ses_actions,
            "forbidden": forbidden,
            "broad_not_action": broad_not_action,
            "unexpected_actions": unexpected_actions,
            "resources_scoped": resources_scoped,
        }

    def blast_003(self) -> None:
        case_id = "GF-REGIONAL-BLAST-003"
        target_results = []
        passed = True
        expected_role = expected_executor_role()
        expected_node_verbs = set(expected_role.get("core:nodes", []))
        expected_pod_verbs = set(expected_role.get("core:pods", []))
        # The spec's two named facts about the manifest, checked against the
        # manifest itself rather than assumed: nodes carry no delete, pods no
        # create.
        if (
            expected_node_verbs != {"get", "list", "watch", "patch"}
            or expected_pod_verbs != {"get", "list", "watch"}
            or any(
                not set(verbs) <= {"get", "list", "watch"}
                for key, verbs in expected_role.items()
                if key != "core:nodes"
            )
        ):
            raise CheckError(
                "shipped executor ClusterRole grants nodes/delete or pods/create"
            )
        for target in self.targets:
            pods = self.gpu_json(
                target,
                "-n",
                self.namespace,
                "get",
                "pod",
                "-l",
                "app=gpu-fault-cluster-executor",
                "-o",
                "json",
            ).get("items")
            pod_identity_matches = (
                bool(pods)
                and len(ready_pod_records({"items": pods})) == len(pods)
                and all(
                    (item.get("spec") or {}).get("serviceAccountName")
                    == "gpu-fault-cluster-executor"
                    for item in pods
                )
            )
            service_account = (
                f"system:serviceaccount:{self.namespace}:gpu-fault-cluster-executor"
            )
            verbs = ("get", "list", "watch", "create", "update", "patch", "delete")
            resources = (
                "nodes",
                "pods",
                "jobs",
                "secrets",
                "configmaps",
                "pytorchjobs.kubeflow.org",
                "jobsets.jobset.x-k8s.io",
                "clusterroles.rbac.authorization.k8s.io",
            )
            matrix = self.auth_can_i(
                kube_prefix=(
                    "--kubeconfig",
                    self.gpu_kubeconfig,
                    "--context",
                    target.context,
                ),
                service_account=service_account,
                verbs=verbs,
                resources=resources,
            )
            write_json(
                self.run_dir / f"BLAST-003-rbac-{target.cluster_id}.json",
                matrix,
            )
            sensitive_denied = all(
                not matrix[verb][resource]
                for verb in verbs
                for resource in (
                    "secrets",
                    "configmaps",
                    "clusterroles.rbac.authorization.k8s.io",
                )
            )
            namespace_scopes, sensitive_denied = self.executor_namespace_permissions(
                target,
                service_account=service_account,
                verbs=verbs,
                sensitive_denied=sensitive_denied,
            )
            node_expected = all(
                matrix[verb]["nodes"] == (verb in expected_node_verbs) for verb in verbs
            )
            pod_expected = all(
                matrix[verb]["pods"] == (verb in expected_pod_verbs) for verb in verbs
            )
            workload_expected = all(
                matrix[verb][resource] == (verb in expected_role.get(role_key, []))
                for resource, role_key in (
                    ("jobs", "batch:jobs"),
                    ("pytorchjobs.kubeflow.org", "kubeflow.org:pytorchjobs"),
                    ("jobsets.jobset.x-k8s.io", "jobset.x-k8s.io:jobsets"),
                )
                for verb in verbs
            ) and all(
                value["workload_writes_match_policy"]
                for value in namespace_scopes.values()
            )

            role = self.gpu_json(
                target,
                "get",
                "clusterrole",
                "gpu-fault-cluster-executor",
                "-o",
                "json",
            )
            normalized_role = self.normalized_role_rules(role)
            role_matches_manifest = normalized_role == expected_role
            binding_errors = unexpected_grants(
                bound_rules(
                    self.gpu_json(target, "get", RBAC_INVENTORY, "-A", "-o", "json"),
                    service_account,
                ),
                expected_cluster=expected_role,
                expected_namespaces=self.expected_namespace_roles(target),
            )
            write_json(
                self.run_dir / f"BLAST-003-role-{target.cluster_id}.json",
                {
                    "actual": normalized_role,
                    "expected": expected_role,
                    "expected_source": str(EXECUTOR_MANIFEST.relative_to(ROOT)),
                    "matches": role_matches_manifest,
                    "unexpected_bound_grants": binding_errors,
                },
            )

            iam = self.executor_iam_scope(target, namespace_scopes)

            target_passed = all(
                (
                    sensitive_denied,
                    pod_identity_matches,
                    node_expected,
                    pod_expected,
                    workload_expected,
                    role_matches_manifest,
                    not binding_errors,
                    iam["role_binding_matches"],
                    not iam["ses_actions"],
                    not iam["forbidden"],
                    not iam["broad_not_action"],
                    not iam["unexpected_actions"],
                    iam["resources_scoped"],
                )
            )
            passed = passed and target_passed
            target_results.append(
                {
                    "cluster_id": target.cluster_id,
                    "sensitive_resources_all_denied": sensitive_denied,
                    "executor_pods_use_audited_service_account": pod_identity_matches,
                    "nodes_exact_permissions": node_expected,
                    "pods_exact_permissions": pod_expected,
                    "workloads_exact_permissions": workload_expected,
                    "cluster_role_matches_manifest": role_matches_manifest,
                    "unexpected_bound_grants": binding_errors,
                    "service_account_role_matches_site": iam["role_binding_matches"],
                    "ses_action_patterns": iam["ses_actions"],
                    "forbidden_provider_mutations_present": iam["forbidden"],
                }
            )

        self.record_case(
            case_id,
            "PASS" if passed else "FAIL",
            checks={"clusters": target_results},
            limitations=[
                "Kubernetes RBAC cannot scope nodes/patch by label selector; "
                "the executor can patch any node in its local GPU EKS."
            ],
        )
        if not passed:
            raise CheckError(f"{case_id} failed")

    def execution_token_digest(self, pod: str) -> tuple[str, int]:
        probe = (
            "import hashlib,json,os;"
            "v=os.environ.get('GPU_FAULT_EXECUTION_TOKEN','').strip();"
            "print(json.dumps({'sha256':hashlib.sha256(v.encode()).hexdigest(),"
            "'length':len(v)}))"
        )
        payload = json.loads(
            self.cpu_text(
                "-n",
                self.namespace,
                "exec",
                pod,
                "--",
                component_python("cpu"),
                "-c",
                probe,
            )
        )
        length = int(payload["length"])
        if length < 32:
            raise CheckError("control-plane execution token is missing or too short")
        return str(payload["sha256"]), length

    @staticmethod
    def decode_secret_value(value: str) -> bytes:
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CheckError("invalid base64 in Kubernetes Secret data") from exc

    def scan_gpu_objects(
        self,
        target: ClusterTarget,
        *,
        execution_hash: str,
        foreign_cluster_hashes: set[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        payload = self.gpu_json(
            target,
            "get",
            "secret,configmap,pod",
            "-A",
            "-o",
            "json",
        )
        hits: list[dict[str, Any]] = []
        if not isinstance(payload.get("items"), list) or (
            payload.get("metadata") or {}
        ).get("continue"):
            raise CheckError("GPU credential object inventory is incomplete")
        foreign_hits: list[dict[str, Any]] = []
        name_hits: list[dict[str, Any]] = []
        cluster_token_hash = ""
        canonical_count = 0

        def record_value(location: dict[str, Any], raw: bytes) -> None:
            try:
                digests = credential_value_digests(
                    raw, require_json=str(location.get("key", "")).endswith(".json")
                )
            except CredentialScanError as exc:
                raise CheckError(str(exc)) from None
            if execution_hash in digests:
                hits.append(location)
            if digests & (foreign_cluster_hashes or set()):
                foreign_hits.append(location)

        for item in payload.get("items", []):
            kind = str(item.get("kind", ""))
            metadata = item.get("metadata", {})
            namespace = str(metadata.get("namespace", ""))
            name = str(metadata.get("name", ""))
            if EXECUTION_TOKEN_NAME.search(name):
                name_hits.append(
                    {
                        "kind": kind,
                        "namespace": namespace,
                        "name": name,
                        "location": "object-name",
                    }
                )
            if kind == "Secret":
                for key, encoded in (item.get("data") or {}).items():
                    if EXECUTION_TOKEN_NAME.search(str(key)):
                        name_hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "location": f"data-key:{key}",
                            }
                        )
                    raw = self.decode_secret_value(str(encoded))
                    record_value(
                        {
                            "kind": kind,
                            "namespace": namespace,
                            "name": name,
                            "key": key,
                        },
                        raw,
                    )
                    if (
                        namespace == self.namespace
                        and name == "gpu-fault-regional-connection"
                        and key == "cluster-token"
                    ):
                        canonical_count += 1
                        if len(raw.strip()) < 32:
                            raise CheckError(
                                f"cluster token is missing or too short for {target.cluster_id}"
                            )
                        cluster_token_hash = sha256_bytes(raw.strip())
            elif kind == "ConfigMap":
                for key, value in (item.get("data") or {}).items():
                    if EXECUTION_TOKEN_NAME.search(str(key)):
                        name_hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "location": f"data-key:{key}",
                            }
                        )
                    record_value(
                        {
                            "kind": kind,
                            "namespace": namespace,
                            "name": name,
                            "key": key,
                        },
                        str(value).encode(),
                    )
                for key, encoded in (item.get("binaryData") or {}).items():
                    if EXECUTION_TOKEN_NAME.search(str(key)):
                        name_hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "location": f"binary-data-key:{key}",
                            }
                        )
                    raw = self.decode_secret_value(str(encoded))
                    record_value(
                        {
                            "kind": kind,
                            "namespace": namespace,
                            "name": name,
                            "key": key,
                        },
                        raw,
                    )
            elif kind == "Pod":
                spec = item.get("spec") or {}
                containers = [
                    *(spec.get("initContainers") or []),
                    *(spec.get("containers") or []),
                    *(spec.get("ephemeralContainers") or []),
                ]
                for container in containers:
                    container_name = str(container.get("name", ""))
                    for env in container.get("env") or []:
                        env_name = str(env.get("name", ""))
                        if EXECUTION_TOKEN_NAME.search(env_name):
                            name_hits.append(
                                {
                                    "kind": kind,
                                    "namespace": namespace,
                                    "name": name,
                                    "location": (
                                        f"container:{container_name}:env:{env_name}"
                                    ),
                                }
                            )
                        if "value" in env:
                            record_value(
                                {
                                    "kind": kind,
                                    "namespace": namespace,
                                    "name": name,
                                    "key": (
                                        f"container:{container_name}:env:{env_name}"
                                    ),
                                },
                                str(env.get("value", "")).encode(),
                            )
                        value_from = env.get("valueFrom") or {}
                        for ref_kind in ("secretKeyRef", "configMapKeyRef"):
                            ref = value_from.get(ref_kind) or {}
                            if any(
                                EXECUTION_TOKEN_NAME.search(str(ref.get(field, "")))
                                for field in ("name", "key")
                            ):
                                name_hits.append(
                                    {
                                        "kind": kind,
                                        "namespace": namespace,
                                        "name": name,
                                        "location": (
                                            f"container:{container_name}:"
                                            f"{ref_kind}:{ref.get('name')}:"
                                            f"{ref.get('key')}"
                                        ),
                                    }
                                )
                    for env_from in container.get("envFrom") or []:
                        for ref_kind in ("secretRef", "configMapRef"):
                            ref = env_from.get(ref_kind) or {}
                            if EXECUTION_TOKEN_NAME.search(str(ref.get("name", ""))):
                                name_hits.append(
                                    {
                                        "kind": kind,
                                        "namespace": namespace,
                                        "name": name,
                                        "location": (
                                            f"container:{container_name}:"
                                            f"{ref_kind}:{ref.get('name')}"
                                        ),
                                    }
                                )

        if not cluster_token_hash or canonical_count != 1:
            raise CheckError(f"cluster token not found for {target.cluster_id}")
        return {
            "execution_token_value_hash_hits": hits,
            "execution_token_name_or_reference_hits": name_hits,
            "foreign_cluster_token_hash_hits": foreign_hits,
        }, cluster_token_hash

    def cluster_token_digest(self, target: ClusterTarget) -> str:
        document = self.gpu_json(
            target,
            "-n",
            self.namespace,
            "get",
            "secret",
            "gpu-fault-regional-connection",
            "-o",
            "json",
        )
        raw = self.decode_secret_value(document["data"]["cluster-token"]).strip()
        if len(raw) < 32:
            raise CheckError(
                f"cluster token is missing or too short for {target.cluster_id}"
            )
        return sha256_bytes(raw)

    def blast_004(self) -> None:
        case_id = "GF-REGIONAL-BLAST-004"
        cpu_pod = self.ready_cpu_pod()
        execution_hash, execution_length = self.execution_token_digest(cpu_pod)
        cluster_results = []
        cluster_hashes = []
        passed = True
        baseline_hashes = {
            target.cluster_id: self.cluster_token_digest(target)
            for target in self.targets
        }
        for target in self.targets:
            object_scan, cluster_hash = self.scan_gpu_objects(
                target,
                execution_hash=execution_hash,
                foreign_cluster_hashes={
                    value
                    for cluster, value in baseline_hashes.items()
                    if cluster != target.cluster_id
                },
            )
            executor_pod = self.ready_executor_pod(target)
            runtime = json.loads(
                self.gpu_text(
                    target,
                    "-n",
                    self.namespace,
                    "exec",
                    executor_pod,
                    "--",
                    component_python("gpu"),
                    "-c",
                    "import hashlib,json,os;"
                    "print(json.dumps({'cluster_id':os.environ.get('GPU_FAULT_CLUSTER_ID'),"
                    "'digests':{key:hashlib.sha256(value.strip().encode()).hexdigest() "
                    "for key,value in os.environ.items()}}))",
                )
            )
            if (
                not isinstance(runtime, dict)
                or runtime.get("cluster_id") != target.cluster_id
                or not isinstance(runtime.get("digests"), dict)
                or not runtime["digests"]
            ):
                raise CheckError(
                    "executor runtime credential scan is incomplete or foreign"
                )
            env_digests = runtime["digests"]
            runtime_name_hits = [
                item for item in env_digests if EXECUTION_TOKEN_NAME.search(str(item))
            ]
            forbidden_hashes = {execution_hash} | {
                digest
                for cluster, digest in baseline_hashes.items()
                if cluster != target.cluster_id
            }
            runtime_value_hits = [
                name
                for name, digest in env_digests.items()
                if digest in forbidden_hashes
            ]
            cluster_passed = all(
                (
                    not object_scan["execution_token_value_hash_hits"],
                    not object_scan["execution_token_name_or_reference_hits"],
                    not object_scan["foreign_cluster_token_hash_hits"],
                    not runtime_name_hits,
                    not runtime_value_hits,
                    env_digests.get("GPU_FAULT_CONTROL_PLANE_TOKEN") == cluster_hash,
                    cluster_hash != execution_hash,
                    cluster_hash == baseline_hashes[target.cluster_id],
                )
            )
            passed = passed and cluster_passed
            cluster_hashes.append(cluster_hash)
            result = {
                "cluster_id": target.cluster_id,
                **object_scan,
                "executor_runtime_execution_token_env_names": runtime_name_hits,
                "executor_runtime_foreign_credential_env_names": runtime_value_hits,
                "cluster_token_sha256": cluster_hash,
                "cluster_token_differs_from_execution_token": (
                    cluster_hash != execution_hash
                ),
            }
            cluster_results.append(result)
            write_json(
                self.run_dir / f"BLAST-004-scan-{target.cluster_id}.json",
                result,
            )

        cluster_tokens_unique = len(set(cluster_hashes)) == len(cluster_hashes)
        passed = passed and cluster_tokens_unique and len(cluster_hashes) >= 2
        limitations = []
        if len(cluster_hashes) < 2:
            limitations.append(
                "The current site has one registered physical GPU cluster, "
                "so there are no cross-cluster token pairs to compare. "
                "Execution-token versus cluster-token separation is verified."
            )
        self.record_case(
            case_id,
            "PASS" if passed else "FAIL",
            checks={
                "execution_token_length": execution_length,
                "execution_token_sha256": execution_hash,
                "clusters": cluster_results,
                "cluster_token_count": len(cluster_hashes),
                "cluster_tokens_pairwise_unique": cluster_tokens_unique,
            },
            limitations=limitations,
        )
        if not passed:
            raise CheckError(f"{case_id} failed")
