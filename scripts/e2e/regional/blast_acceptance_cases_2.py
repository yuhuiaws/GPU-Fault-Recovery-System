from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping

from scripts.e2e.regional.blast_acceptance_base import (
    EXECUTION_TOKEN_NAME,
    FORBIDDEN_EXECUTOR_ACTIONS,
    CheckError,
    ClusterTarget,
    action_pattern_matches,
    allow_statement_matches,
    sha256_bytes,
    sha256_text,
    statement_not_actions,
    write_json,
)
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne


class BlastCasesTwo(BlastCasesOne):
    @staticmethod
    def normalized_role_rules(role: Mapping[str, Any]) -> dict[str, list[str]]:
        result: dict[str, set[str]] = {}
        for rule in role.get("rules") or []:
            api_groups = rule.get("apiGroups") or [""]
            resources = rule.get("resources") or []
            verbs = rule.get("verbs") or []
            for api_group in api_groups:
                for resource in resources:
                    key = f"{api_group or 'core'}:{resource}"
                    result.setdefault(key, set()).update(str(item) for item in verbs)
        return {key: sorted(value) for key, value in sorted(result.items())}

    def blast_003(self) -> None:
        case_id = "GF-REGIONAL-BLAST-003"
        target_results = []
        passed = True
        expected_role = {
            "core:nodes": ["get", "list", "patch", "watch"],
            "core:pods": ["delete", "get", "list", "patch", "watch"],
            "batch:jobs": ["create", "get", "list", "patch", "watch"],
            "kubeflow.org:pytorchjobs": [
                "create",
                "get",
                "list",
                "patch",
                "watch",
            ],
            "jobset.x-k8s.io:jobsets": [
                "create",
                "get",
                "list",
                "patch",
                "watch",
            ],
        }
        for target in self.targets:
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
            node_expected = all(
                matrix[verb]["nodes"] == (verb in {"get", "list", "watch", "patch"})
                for verb in verbs
            )
            pod_expected = all(
                matrix[verb]["pods"]
                == (verb in {"get", "list", "watch", "patch", "delete"})
                for verb in verbs
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
            write_json(
                self.run_dir / f"BLAST-003-role-{target.cluster_id}.json",
                {
                    "actual": normalized_role,
                    "expected": expected_role,
                    "matches": role_matches_manifest,
                },
            )

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
            patterns = self.allowed_action_patterns(statements)
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
                }
            )
            write_json(
                self.run_dir / f"BLAST-003-iam-{target.cluster_id}.json",
                inventory,
            )

            target_passed = all(
                (
                    sensitive_denied,
                    node_expected,
                    pod_expected,
                    role_matches_manifest,
                    role_binding_matches,
                    not ses_actions,
                    not forbidden,
                    not broad_not_action,
                )
            )
            passed = passed and target_passed
            target_results.append(
                {
                    "cluster_id": target.cluster_id,
                    "sensitive_resources_all_denied": sensitive_denied,
                    "nodes_exact_permissions": node_expected,
                    "pods_exact_permissions": pod_expected,
                    "cluster_role_matches_manifest": role_matches_manifest,
                    "service_account_role_matches_site": role_binding_matches,
                    "ses_action_patterns": ses_actions,
                    "forbidden_provider_mutations_present": forbidden,
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
            "v=os.environ.get('GPU_FAULT_EXECUTION_TOKEN','');"
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
                "python",
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
        name_hits: list[dict[str, Any]] = []
        cluster_token_hash = ""
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
                    if sha256_bytes(raw) == execution_hash:
                        hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "key": key,
                            }
                        )
                    if (
                        namespace == self.namespace
                        and name == "gpu-fault-regional-connection"
                        and key == "cluster-token"
                    ):
                        cluster_token_hash = sha256_bytes(raw)
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
                    if sha256_text(str(value)) == execution_hash:
                        hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "key": key,
                            }
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
                    if sha256_bytes(raw) == execution_hash:
                        hits.append(
                            {
                                "kind": kind,
                                "namespace": namespace,
                                "name": name,
                                "key": key,
                            }
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
                        if (
                            "value" in env
                            and sha256_text(str(env.get("value", ""))) == execution_hash
                        ):
                            hits.append(
                                {
                                    "kind": kind,
                                    "namespace": namespace,
                                    "name": name,
                                    "key": (
                                        f"container:{container_name}:env:{env_name}"
                                    ),
                                }
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

        if not cluster_token_hash:
            raise CheckError(f"cluster token not found for {target.cluster_id}")
        return {
            "execution_token_value_hash_hits": hits,
            "execution_token_name_or_reference_hits": name_hits,
        }, cluster_token_hash

    def blast_004(self) -> None:
        case_id = "GF-REGIONAL-BLAST-004"
        cpu_pod = self.ready_cpu_pod()
        execution_hash, execution_length = self.execution_token_digest(cpu_pod)
        cluster_results = []
        cluster_hashes = []
        passed = True
        for target in self.targets:
            object_scan, cluster_hash = self.scan_gpu_objects(
                target, execution_hash=execution_hash
            )
            executor_pod = self.ready_executor_pod(target)
            env_names = json.loads(
                self.gpu_text(
                    target,
                    "-n",
                    self.namespace,
                    "exec",
                    executor_pod,
                    "--",
                    "python",
                    "-c",
                    "import json,os;print(json.dumps(sorted(os.environ)))",
                )
            )
            runtime_name_hits = [
                item for item in env_names if EXECUTION_TOKEN_NAME.search(str(item))
            ]
            cluster_passed = all(
                (
                    not object_scan["execution_token_value_hash_hits"],
                    not object_scan["execution_token_name_or_reference_hits"],
                    not runtime_name_hits,
                    cluster_hash != execution_hash,
                )
            )
            passed = passed and cluster_passed
            cluster_hashes.append(cluster_hash)
            result = {
                "cluster_id": target.cluster_id,
                **object_scan,
                "executor_runtime_execution_token_env_names": runtime_name_hits,
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
        passed = passed and cluster_tokens_unique
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
