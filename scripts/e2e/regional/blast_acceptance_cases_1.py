from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from scripts.e2e.regional.blast_acceptance_base import (
    FORBIDDEN_CONTROL_PLANE_ACTIONS,
    GPU_FAULT_PREFIX,
    BlastRunnerBase,
    CheckError,
    allow_statement_matches,
    arn_parts,
    as_list,
    command,
    parse_time,
    policy_statement_summary,
    resources_for,
    statement_actions,
    statement_not_actions,
    write_json,
    write_text,
)


class BlastCasesOne(BlastRunnerBase):
    @staticmethod
    def node_security_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for node in payload.get("items", []):
            metadata = node.get("metadata", {})
            spec = node.get("spec", {})
            labels = metadata.get("labels") or {}
            annotations = metadata.get("annotations") or {}
            taints = sorted(
                [
                    {
                        "key": item.get("key"),
                        "value": item.get("value"),
                        "effect": item.get("effect"),
                    }
                    for item in spec.get("taints") or []
                ],
                key=lambda item: (
                    str(item["key"]),
                    str(item["value"]),
                    str(item["effect"]),
                ),
            )
            result[str(metadata.get("name", ""))] = {
                "taints": taints,
                "unschedulable": bool(spec.get("unschedulable", False)),
                "gpu_fault_labels": {
                    key: value
                    for key, value in sorted(labels.items())
                    if key.startswith(GPU_FAULT_PREFIX)
                },
                "gpu_fault_annotations": {
                    key: value
                    for key, value in sorted(annotations.items())
                    if key.startswith(GPU_FAULT_PREFIX)
                },
            }
        return result

    @staticmethod
    def managed_field_relevant(entry: Mapping[str, Any], *, since: datetime) -> bool:
        timestamp = entry.get("time")
        if not timestamp:
            return False
        try:
            if parse_time(str(timestamp)) < since:
                return False
        except ValueError:
            return True
        encoded = json.dumps(entry.get("fieldsV1") or {}, sort_keys=True)
        return any(
            marker in encoded
            for marker in (
                '"f:taints"',
                '"f:unschedulable"',
                "f:gpu-fault.io/",
            )
        )

    def blast_001(self) -> None:
        case_id = "GF-REGIONAL-BLAST-001"
        execution_card = json.loads(
            (self.e2e_dir / "execution-card.json").read_text(encoding="utf-8")
        )
        window = execution_card.get("maintenance_window") or {}
        start_text = str(window.get("start", ""))
        end_text = str(window.get("end", ""))
        if not start_text:
            raise CheckError("E2E-001 execution card has no start time")
        start = parse_time(start_text)

        current_nodes = self.cpu_json("get", "nodes", "-o", "json")
        write_json(self.run_dir / "BLAST-001-cpu-nodes-after.json", current_nodes)
        current_snapshot = self.node_security_snapshot(current_nodes)

        baseline_nodes = json.loads(
            self.trusted_cpu_baseline.read_text(encoding="utf-8")
        )
        baseline_snapshot = self.node_security_snapshot(baseline_nodes)
        baseline_identical = baseline_snapshot == current_snapshot

        managed_field_hits = []
        for node in current_nodes.get("items", []):
            metadata = node.get("metadata", {})
            for entry in metadata.get("managedFields") or []:
                if self.managed_field_relevant(entry, since=start):
                    managed_field_hits.append(
                        {
                            "node": metadata.get("name"),
                            "manager": entry.get("manager"),
                            "operation": entry.get("operation"),
                            "time": entry.get("time"),
                        }
                    )

        jobs = self.cpu_json("get", "jobs", "-A", "-o", "json")
        gpu_fault_jobs = []
        gpu_fault_jobs_in_window = []
        for job in jobs.get("items", []):
            metadata = job.get("metadata", {})
            labels = metadata.get("labels") or {}
            item = {
                "namespace": metadata.get("namespace"),
                "name": metadata.get("name"),
                "creation_timestamp": metadata.get("creationTimestamp"),
                "gpu_fault_labels": {
                    key: value
                    for key, value in labels.items()
                    if key.startswith(GPU_FAULT_PREFIX)
                },
            }
            if item["gpu_fault_labels"]:
                gpu_fault_jobs.append(item)
                created = metadata.get("creationTimestamp")
                if created and parse_time(str(created)) >= start:
                    gpu_fault_jobs_in_window.append(item)

        events = self.cpu_json("get", "events", "-A", "-o", "json")
        eviction_events = []
        for event in events.get("items", []):
            reason = str(event.get("reason", ""))
            message = str(event.get("message", ""))
            if reason not in {"Evicted", "TaintManagerEviction"} and not any(
                term in message for term in ("Evicted", "TaintManagerEviction")
            ):
                continue
            metadata = event.get("metadata", {})
            event_time = (
                event.get("eventTime")
                or event.get("lastTimestamp")
                or event.get("firstTimestamp")
                or metadata.get("creationTimestamp")
            )
            if event_time and parse_time(str(event_time)) < start:
                continue
            eviction_events.append(
                {
                    "namespace": metadata.get("namespace"),
                    "name": metadata.get("name"),
                    "reason": reason,
                    "event_time": event_time,
                    "involved_kind": event.get("involvedObject", {}).get("kind"),
                    "involved_name": event.get("involvedObject", {}).get("name"),
                }
            )

        e2e_state = json.loads(
            (self.e2e_dir / "control-plane-current.json").read_text(encoding="utf-8")
        )
        workflow_operations = sorted(
            {
                str(step.get("operation"))
                for workflow in e2e_state.get("workflows", [])
                for group in (
                    workflow.get("official_steps") or [],
                    workflow.get("safety_steps") or [],
                    workflow.get("step_executions") or [],
                )
                for step in group
                if step.get("operation")
            }
        )
        required_operations_present = all(
            item in workflow_operations
            for item in (
                "MARK_UNSCHEDULABLE",
                "STOP_WORKLOADS",
                "RESTART_WORKLOAD",
            )
        )

        checks = {
            "e2e_evidence_dir": str(self.e2e_dir),
            "e2e_window_start": start_text,
            "e2e_window_end": end_text,
            "workflow_operations": workflow_operations,
            "required_operations_present": required_operations_present,
            "trusted_baseline": str(self.trusted_cpu_baseline),
            "trusted_baseline_identical": baseline_identical,
            "relevant_managed_field_writes_since_e2e_start": managed_field_hits,
            "gpu_fault_jobs_current": gpu_fault_jobs,
            "gpu_fault_jobs_created_since_e2e_start": gpu_fault_jobs_in_window,
            "eviction_events_since_e2e_start": eviction_events,
        }
        write_json(self.run_dir / "BLAST-001-analysis.json", checks)

        passed = all(
            (
                required_operations_present,
                baseline_identical,
                not managed_field_hits,
                not gpu_fault_jobs_in_window,
                not eviction_events,
            )
        )
        status = "PASS" if passed else "FAIL"
        limitations = [
            "The current E2E-001 run did not preserve an immediate CPU node "
            "before snapshot. The verdict combines the latest trusted full "
            "CPU node baseline with managedFields and event/job checks for "
            "the exact E2E window."
        ]
        self.record_case(
            case_id,
            status,
            checks={
                "cpu_nodes_identical_to_trusted_baseline": baseline_identical,
                "relevant_node_writes_in_window": len(managed_field_hits),
                "gpu_fault_jobs_created_in_window": len(gpu_fault_jobs_in_window),
                "eviction_events_in_window": len(eviction_events),
                "required_e2e_operations_present": required_operations_present,
            },
            limitations=limitations,
        )
        if not passed:
            raise CheckError(f"{case_id} failed")

    def auth_can_i(
        self,
        *,
        kube_prefix: Sequence[str],
        service_account: str,
        verbs: Sequence[str],
        resources: Sequence[str],
    ) -> dict[str, dict[str, bool]]:
        matrix: dict[str, dict[str, bool]] = {}
        for verb in verbs:
            matrix[verb] = {}
            for resource in resources:
                result = command(
                    (
                        "kubectl",
                        *kube_prefix,
                        "auth",
                        "can-i",
                        verb,
                        resource,
                        f"--as={service_account}",
                        "--all-namespaces",
                    ),
                    check=False,
                )
                answer = result.stdout.strip().lower()
                if result.returncode not in {0, 1} or answer not in {"yes", "no"}:
                    raise CheckError(
                        f"auth can-i returned {answer!r} for {verb} {resource}"
                    )
                matrix[verb][resource] = answer == "yes"
        return matrix

    def iam_role_policies(
        self, role_arn: str
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        role_name = role_arn.rsplit("/", 1)[-1]
        inline = self.aws("iam", "list-role-policies", "--role-name", role_name)
        attached = self.aws(
            "iam", "list-attached-role-policies", "--role-name", role_name
        )
        documents: list[Mapping[str, Any]] = []
        policy_inventory: list[dict[str, Any]] = []

        for policy_name in inline.get("PolicyNames", []):
            payload = self.aws(
                "iam",
                "get-role-policy",
                "--role-name",
                role_name,
                "--policy-name",
                str(policy_name),
            )
            document = payload.get("PolicyDocument") or {}
            documents.append(document)
            policy_inventory.append(
                {
                    "kind": "inline",
                    "name": policy_name,
                    "statements": [
                        policy_statement_summary(item)
                        for item in as_list(document.get("Statement"))
                    ],
                }
            )

        for policy in attached.get("AttachedPolicies", []):
            policy_arn = str(policy["PolicyArn"])
            metadata = self.aws("iam", "get-policy", "--policy-arn", policy_arn)
            version_id = str(metadata["Policy"]["DefaultVersionId"])
            payload = self.aws(
                "iam",
                "get-policy-version",
                "--policy-arn",
                policy_arn,
                "--version-id",
                version_id,
            )
            document = payload.get("PolicyVersion", {}).get("Document") or {}
            documents.append(document)
            policy_inventory.append(
                {
                    "kind": "attached",
                    "name": policy.get("PolicyName"),
                    "arn": policy_arn,
                    "version_id": version_id,
                    "statements": [
                        policy_statement_summary(item)
                        for item in as_list(document.get("Statement"))
                    ],
                }
            )

        statements = [
            item
            for document in documents
            for item in as_list(document.get("Statement"))
            if isinstance(item, Mapping)
        ]
        return statements, {
            "role_arn": role_arn,
            "role_name": role_name,
            "policies": policy_inventory,
        }

    def cpu_control_plane_role_arn(self) -> tuple[str, dict[str, Any]]:
        associations = self.aws(
            "eks",
            "list-pod-identity-associations",
            "--cluster-name",
            self.cpu_cluster_name,
            "--namespace",
            self.namespace,
            "--service-account",
            "gpu-fault-control-plane",
        )
        summaries = associations.get("associations") or []
        if len(summaries) != 1:
            raise CheckError(
                "expected exactly one control-plane Pod Identity association"
            )
        association_id = str(summaries[0]["associationId"])
        detail = self.aws(
            "eks",
            "describe-pod-identity-association",
            "--cluster-name",
            self.cpu_cluster_name,
            "--association-id",
            association_id,
        )
        association = detail.get("association") or {}
        role_arn = str(association.get("roleArn", ""))
        if not role_arn:
            raise CheckError("control-plane Pod Identity has no role ARN")
        evidence = {
            "association_id": association_id,
            "namespace": association.get("namespace"),
            "service_account": association.get("serviceAccount"),
            "role_arn": role_arn,
        }
        return role_arn, evidence

    @staticmethod
    def allowed_action_patterns(
        statements: Iterable[Mapping[str, Any]],
    ) -> list[str]:
        return sorted(
            {
                action
                for statement in statements
                if str(statement.get("Effect", "")).lower() == "allow"
                for action in statement_actions(statement)
            }
        )

    def blast_002(self) -> None:
        case_id = "GF-REGIONAL-BLAST-002"
        pod = self.ready_cpu_pod()
        filesystem_probe = self.cpu_text(
            "-n",
            self.namespace,
            "exec",
            pod,
            "--",
            "sh",
            "-c",
            (
                "printf 'serviceaccount:\\n'; "
                "find /var/run/secrets/kubernetes.io/serviceaccount "
                "-maxdepth 1 -mindepth 1 -printf '%f\\n' 2>/dev/null | sort; "
                "printf 'kubeconfig_env='; "
                "if [ -n \"${KUBECONFIG:-}\" ]; then printf 'set\\n'; "
                "else printf 'unset\\n'; fi; "
                "printf 'home_kube='; "
                "if [ -d \"${HOME:-/root}/.kube\" ]; then printf 'present\\n'; "
                "else printf 'absent\\n'; fi"
            ),
        )
        write_text(self.run_dir / "BLAST-002-pod-filesystem.txt", filesystem_probe)

        service_account = (
            f"system:serviceaccount:{self.namespace}:gpu-fault-control-plane"
        )
        matrix = self.auth_can_i(
            kube_prefix=("--kubeconfig", self.cpu_kubeconfig),
            service_account=service_account,
            verbs=("patch", "delete", "create", "update"),
            resources=("nodes", "pods", "jobs"),
        )
        write_json(self.run_dir / "BLAST-002-rbac.json", matrix)
        rbac_all_denied = not any(
            allowed for resources in matrix.values() for allowed in resources.values()
        )

        role_arn, pod_identity = self.cpu_control_plane_role_arn()
        write_json(self.run_dir / "BLAST-002-pod-identity.json", pod_identity)
        statements, inventory = self.iam_role_policies(role_arn)
        patterns = self.allowed_action_patterns(statements)
        forbidden = [
            action
            for action in FORBIDDEN_CONTROL_PLANE_ACTIONS
            if any(allow_statement_matches(item, action) for item in statements)
        ]
        broad_not_action = any(
            str(item.get("Effect", "")).lower() == "allow"
            and statement_not_actions(item)
            for item in statements
        )
        sagemaker_patterns = [
            item for item in patterns if item.lower().startswith("sagemaker:")
        ]
        sagemaker_read_only = bool(sagemaker_patterns) and all(
            item.lower().startswith(("sagemaker:describe", "sagemaker:list"))
            for item in sagemaker_patterns
        )
        ses_send_statements = [
            item
            for item in statements
            if allow_statement_matches(item, "ses:SendEmail")
        ]
        ses_scoped = bool(ses_send_statements) and all(
            resources_for(item)
            and all(
                resource != "*"
                and resource.startswith(
                    f"arn:aws:ses:{self.region}:{arn_parts(self.cpu_eks_arn)[1]}:"
                    "identity/"
                )
                for resource in resources_for(item)
            )
            and bool(item.get("Condition"))
            for item in ses_send_statements
        )
        inventory.update(
            {
                "allowed_action_patterns": patterns,
                "forbidden_hyperpod_mutations_present": forbidden,
                "broad_allow_not_action_present": broad_not_action,
                "sagemaker_read_only_describe_list_only": sagemaker_read_only,
                "ses_send_statement_count": len(ses_send_statements),
                "ses_send_scoped_to_identity_with_condition": ses_scoped,
            }
        )
        write_json(self.run_dir / "BLAST-002-iam.json", inventory)

        passed = all(
            (
                rbac_all_denied,
                not forbidden,
                not broad_not_action,
                sagemaker_read_only,
                ses_scoped,
                "home_kube=absent" in filesystem_probe,
                "kubeconfig_env=unset" in filesystem_probe,
            )
        )
        self.record_case(
            case_id,
            "PASS" if passed else "FAIL",
            checks={
                "control_plane_write_rbac_all_denied": rbac_all_denied,
                "home_kube_absent": "home_kube=absent" in filesystem_probe,
                "kubeconfig_env_unset": "kubeconfig_env=unset" in filesystem_probe,
                "forbidden_hyperpod_mutations_present": forbidden,
                "sagemaker_describe_list_only": sagemaker_read_only,
                "ses_send_scoped": ses_scoped,
            },
        )
        if not passed:
            raise CheckError(f"{case_id} failed")
