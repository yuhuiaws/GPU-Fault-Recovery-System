from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
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
    containment_source,
    default_e2e_dir,
    notification_channel,
    observed_in_windows,
    parse_time,
    policy_statement_summary,
    resources_for,
    sha256_bytes,
    statement_actions,
    statement_not_actions,
    write_json,
    write_text,
)
from scripts.e2e.regional.blast_rbac_scope import (
    RBAC_INVENTORY,
    bound_rules,
    unexpected_grants,
)

AUTHORIZATION_RESOURCE_GROUPS = {
    "nodes": "",
    "pods": "",
    "secrets": "",
    "configmaps": "",
    "jobs": "batch",
}


class BlastCasesOne(BlastRunnerBase):
    @staticmethod
    def node_security_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("items"), list) or not payload["items"]:
            raise CheckError("CPU node baseline is empty or malformed")
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
            name, uid = metadata.get("name"), metadata.get("uid")
            if not name or not uid or name in result:
                raise CheckError("CPU node identity is missing or duplicated")
            result[str(name)] = {
                "uid": uid,
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

    @staticmethod
    def managed_field_key(node: str, entry: Mapping[str, Any]) -> str:
        return json.dumps(
            {
                "node": node,
                "manager": entry.get("manager"),
                "operation": entry.get("operation"),
                "time": entry.get("time"),
                "fieldsV1": entry.get("fieldsV1") or {},
            },
            sort_keys=True,
        )

    @classmethod
    def managed_field_writes(
        cls,
        current_nodes: Mapping[str, Any],
        *,
        baseline_nodes: Mapping[str, Any],
        since: datetime,
    ) -> list[dict[str, Any]]:
        """Taint/cordon/gpu-fault managedFields entries new since the baseline.

        An entry is a hit when it is relevant, stamped at or after ``since``
        and absent from the baseline snapshot taken immediately before the
        fault window. The time filter alone is not enough: a manager that
        rewrites its entry keeps the *newest* time, so a pre-window write by
        the same manager showed up as in-window noise and a baseline entry
        with a clock skewed forward did the same.
        """

        known = {
            cls.managed_field_key(str(node.get("metadata", {}).get("name", "")), entry)
            for node in baseline_nodes.get("items", [])
            for entry in node.get("metadata", {}).get("managedFields") or []
        }
        hits = []
        for node in current_nodes.get("items", []):
            metadata = node.get("metadata", {})
            name = str(metadata.get("name", ""))
            for entry in metadata.get("managedFields") or []:
                if not cls.managed_field_relevant(entry, since=since):
                    continue
                if cls.managed_field_key(name, entry) in known:
                    continue
                hits.append(
                    {
                        "node": name,
                        "manager": entry.get("manager"),
                        "operation": entry.get("operation"),
                        "time": entry.get("time"),
                    }
                )
        return hits

    def blast_001(self) -> None:
        case_id = "GF-REGIONAL-BLAST-001"
        if self.e2e_dir.resolve() != default_e2e_dir(self.root_run_dir).resolve():
            raise CheckError("BLAST-001 workload evidence must belong to this run")
        if (
            self.trusted_cpu_baseline.resolve()
            != (self.e2e_dir / "cpu-nodes-before.json").resolve()
        ):
            raise CheckError("BLAST-001 baseline must be the producer's own snapshot")
        execution_card = json.loads(
            (self.e2e_dir / "execution-card.json").read_text(encoding="utf-8")
        )
        window = execution_card.get("maintenance_window") or {}
        start_text = str(window.get("start", ""))
        end_text = str(window.get("end", ""))
        if (
            execution_card.get("case_id") != "GF-REGIONAL-E2E-001"
            or execution_card.get("verdict") != "PASS"
            or execution_card.get("cleanup_complete") is not True
            or execution_card.get("release_id")
            != self.evidence_identity()["release_id"]
            or execution_card.get("cluster_id")
            not in {target.cluster_id for target in self.targets}
        ):
            raise CheckError(
                "E2E-001 handoff is not a successful, cleaned, deployment-bound run"
            )
        start, end = parse_time(start_text), parse_time(end_text)
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or not start < end <= datetime.now(timezone.utc)
        ):
            raise CheckError("E2E-001 observation window is invalid")
        state_path = self.e2e_dir / "control-plane-current.json"
        if execution_card.get("baseline_sha256") != sha256_bytes(
            self.trusted_cpu_baseline.read_bytes()
        ) or execution_card.get("state_sha256") != sha256_bytes(
            state_path.read_bytes()
        ):
            raise CheckError("E2E-001 baseline or workflow handoff changed")

        current_nodes = self.cpu_json("get", "nodes", "-o", "json")
        write_json(self.run_dir / "BLAST-001-cpu-nodes-after.json", current_nodes)
        current_snapshot = self.node_security_snapshot(current_nodes)

        baseline_nodes = json.loads(
            self.trusted_cpu_baseline.read_text(encoding="utf-8")
        )
        baseline_snapshot = self.node_security_snapshot(baseline_nodes)
        baseline_identical = baseline_snapshot == current_snapshot
        containment = containment_source(
            self.root_run_dir,
            predecessor=self.predecessor,
            release_id=str(execution_card["release_id"]),
            cluster_id=str(execution_card["cluster_id"]),
            cpu_snapshot=baseline_snapshot,
        )
        containment_start = parse_time(containment["window_start"])
        containment_end = parse_time(containment["window_end"])
        if end > containment_start:
            raise CheckError("DESTR-001 must follow E2E-001 in the same run")

        windows = ((start, end), (containment_start, containment_end))

        managed_field_hits = self.managed_field_writes(
            current_nodes,
            baseline_nodes=baseline_nodes,
            since=start,
        )
        managed_field_hits = [
            hit
            for hit in managed_field_hits
            if not hit.get("time") or observed_in_windows(str(hit["time"]), windows)
        ]

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
                if not created or observed_in_windows(str(created), windows):
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
            if event_time and not observed_in_windows(str(event_time), windows):
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

        e2e_state = json.loads(state_path.read_text(encoding="utf-8"))
        workload_operations = sorted(
            {
                str(step.get("operation"))
                for workflow in e2e_state.get("workflows", [])
                if workflow.get("status") == "SUCCEEDED"
                for step in workflow.get("step_executions") or []
                if step.get("operation") and step.get("status") == "SUCCEEDED"
            }
        )
        workflow_operations = sorted(
            set(workload_operations) | set(containment["operations"])
        )
        required_operations_present = {"STOP_WORKLOADS", "RESTART_WORKLOAD"} <= set(
            workload_operations
        ) and containment["operations"] == ["MARK_UNSCHEDULABLE"]

        checks = {
            "e2e_evidence_dir": str(self.e2e_dir),
            "e2e_window_start": start_text,
            "e2e_window_end": end_text,
            "workflow_operations": workflow_operations,
            "workload_operations": workload_operations,
            "containment_source": containment,
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
            "The CPU node baseline is the snapshot E2E-001 wrote immediately "
            "before its injection (cpu-nodes-before.json); managedFields writes are "
            "counted only when they are new relative to that baseline and "
            "inside either producer's window.",
            "E2E-001 proves workload operations; DESTR-001 supplies the successful "
            "MARK_UNSCHEDULABLE command and CPU before/after evidence. This audit "
            "does not trigger a reset or workload restart.",
            "Current managedFields and retained Jobs/events cannot prove the "
            "absence of transient changes already removed from API history.",
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
                "successful_containment_source": containment["case_id"],
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
        namespace: str | None = None,
    ) -> dict[str, dict[str, bool]]:
        matrix: dict[str, dict[str, bool]] = {verb: {} for verb in verbs}
        queries = [(verb, resource) for verb in verbs for resource in resources]
        if not queries:
            return matrix

        def review(verb: str, resource: str) -> bool:
            name, separator, group = resource.partition(".")
            if not separator:
                if name not in AUTHORIZATION_RESOURCE_GROUPS:
                    raise CheckError(
                        f"authorization API group is unknown for {resource}"
                    )
                group = AUTHORIZATION_RESOURCE_GROUPS[name]
            attributes = {
                "verb": verb,
                "resource": name,
                "group": group,
                "namespace": namespace or "",
            }
            request = {
                "apiVersion": "authorization.k8s.io/v1",
                "kind": "SelfSubjectAccessReview",
                "spec": {"resourceAttributes": attributes},
            }
            # A raw review preserves the API group even before its CRD is installed.
            result = command(
                (
                    "kubectl",
                    *kube_prefix,
                    f"--as={service_account}",
                    "create",
                    "--raw=/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
                    "-f",
                    "-",
                ),
                input_text=json.dumps(request),
                check=False,
            )
            if result.returncode:
                raise CheckError(f"authorization review failed for {verb} {resource}")
            try:
                response = json.loads(result.stdout)
                status = response["status"]
                echoed = response["spec"]["resourceAttributes"]
                valid = (
                    response["apiVersion"] == request["apiVersion"]
                    and response["kind"] == request["kind"]
                    and isinstance(status, dict)
                    and isinstance(echoed, dict)
                    and all(
                        echoed.get(key, "") == attributes.get(key, "")
                        for key in (
                            "verb",
                            "resource",
                            "group",
                            "namespace",
                            "name",
                            "subresource",
                        )
                    )
                    and type(status.get("allowed")) is bool
                    and type(status.get("denied", False)) is bool
                    and not (status["allowed"] and status.get("denied"))
                    and not status.get("evaluationError")
                )
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise CheckError(
                    f"authorization review is incomplete for {verb} {resource}"
                )
            return bool(status["allowed"])

        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as pool:
            futures = [
                pool.submit(copy_context().run, review, verb, resource)
                for verb, resource in queries
            ]
            try:
                for (verb, resource), future in zip(queries, futures, strict=True):
                    matrix[verb][resource] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return matrix

    def namespace_names(self, target: Any = None) -> list[str]:
        document = (
            self.cpu_json("get", "namespaces", "-o", "json")
            if target is None
            else self.gpu_json(target, "get", "namespaces", "-o", "json")
        )
        items = document.get("items")
        if not isinstance(items, list) or not items:
            raise CheckError("namespace permission inventory is missing")
        names = [(item.get("metadata") or {}).get("name") for item in items]
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise CheckError("namespace permission inventory is malformed")
        return sorted(str(name) for name in names)

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
        if (
            not role_arn
            or association.get("namespace") != self.namespace
            or association.get("serviceAccount") != "gpu-fault-control-plane"
            or association.get("clusterName") != self.cpu_cluster_name
        ):
            raise CheckError(
                "control-plane Pod Identity does not bind the audited ServiceAccount"
            )
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

    @staticmethod
    def sagemaker_patterns(patterns: Iterable[str]) -> list[str]:
        return [
            item
            for item in patterns
            if ":" in item
            and allow_statement_matches(
                {"Effect": "Allow", "Action": item.split(":", 1)[0] + ":*"},
                "sagemaker:DescribeCluster",
            )
        ]

    @staticmethod
    def sagemaker_read_only(sagemaker_patterns: Iterable[str]) -> bool:
        """ "SageMaker 权限只有 Describe 和 List", zero patterns included.

        A role with no SageMaker permission at all satisfies the requirement;
        requiring at least one pattern failed the stricter role. The pattern
        count is recorded next to this so the evidence still says which of the
        two shapes was seen.
        """

        return all(
            item.lower().startswith(("sagemaker:describe", "sagemaker:list"))
            for item in sagemaker_patterns
        )

    def blast_002(self) -> None:
        case_id = "GF-REGIONAL-BLAST-002"
        pod = self.ready_cpu_pod()
        pod_document = self.cpu_json(
            "-n", self.namespace, "get", "pod", pod, "-o", "json"
        )
        if (pod_document.get("spec") or {}).get(
            "serviceAccountName"
        ) != "gpu-fault-control-plane":
            raise CheckError("CPU Pod does not use the audited ServiceAccount")
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
        namespace_matrices = {
            namespace: self.auth_can_i(
                kube_prefix=("--kubeconfig", self.cpu_kubeconfig),
                service_account=service_account,
                verbs=("create", "patch", "update", "delete"),
                resources=("pods", "jobs", "pytorchjobs.kubeflow.org"),
                namespace=namespace,
            )
            for namespace in self.namespace_names()
        }
        write_json(
            self.run_dir / "BLAST-002-rbac.json",
            {"cluster": matrix, "namespaces": namespace_matrices},
        )
        rbac_all_denied = not any(
            allowed
            for scope in [matrix, *namespace_matrices.values()]
            for resources in scope.values()
            for allowed in resources.values()
        )
        binding_errors = unexpected_grants(
            bound_rules(
                self.cpu_json("get", RBAC_INVENTORY, "-A", "-o", "json"),
                service_account,
            ),
            expected_cluster={},
            expected_namespaces={},
            cpu=True,
        )
        write_json(
            self.run_dir / "BLAST-002-bound-rbac.json",
            {"unexpected_grants": binding_errors},
        )
        rbac_all_denied = rbac_all_denied and not binding_errors

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
        sagemaker_patterns = self.sagemaker_patterns(patterns)
        sagemaker_read_only = self.sagemaker_read_only(sagemaker_patterns)
        # Notifications are least-privilege *per active channel*, not SES-only.
        # control_plane_policy_document (bootstrap_services.py) renders exactly
        # one send permission from spec.notifications.channel: sns:Publish scoped
        # to the site topic ARN (channel "sns", the admin-CLI default), or
        # ses:SendEmail scoped to the verified identity with a FromAddress
        # condition (channel "ses"), or neither (channel "disabled"). So the
        # audit reads the deployed channel and requires that one permission to be
        # present and scoped and the other absent, instead of demanding SES on a
        # topic-only site (which would fail the correct SNS posture).
        notification = self.notification_config()
        channel = notification_channel(notification)
        sns_topic_arn = str(notification.get("GPU_FAULT_SNS_TOPIC_ARN") or "").strip()
        account_id = arn_parts(self.cpu_eks_arn)[1]
        sns_send_statements = [
            item for item in statements if allow_statement_matches(item, "sns:Publish")
        ]
        ses_send_statements = [
            item
            for item in statements
            if allow_statement_matches(item, "ses:SendEmail")
        ]
        sender = str((self.config.get("notifications") or {}).get("email_sender") or "")
        sender_arn = f"arn:aws:ses:{self.region}:{account_id}:identity/{sender}"
        sns_scoped = (
            sns_topic_arn.startswith(f"arn:aws:sns:{self.region}:{account_id}:")
            and not any(marker in sns_topic_arn for marker in ("*", "?"))
            and bool(sns_send_statements)
            and all(
                set(resources_for(item)) == {sns_topic_arn}
                for item in sns_send_statements
            )
        )
        ses_scoped = bool(ses_send_statements) and all(
            bool(sender)
            and set(resources_for(item)) == {sender_arn}
            and ((item.get("Condition") or {}).get("StringEquals") or {}).get(
                "ses:FromAddress"
            )
            == sender
            for item in ses_send_statements
        )
        if channel == "sns":
            notification_send_scoped = sns_scoped and not ses_send_statements
        elif channel == "ses":
            notification_send_scoped = ses_scoped and not sns_send_statements
        else:  # "disabled": the control plane persists to the outbox only.
            notification_send_scoped = (
                not sns_send_statements and not ses_send_statements
            )
        inventory.update(
            {
                "allowed_action_patterns": patterns,
                "forbidden_hyperpod_mutations_present": forbidden,
                "broad_allow_not_action_present": broad_not_action,
                "sagemaker_read_only_describe_list_only": sagemaker_read_only,
                "sagemaker_action_pattern_count": len(sagemaker_patterns),
                "sagemaker_action_patterns": sagemaker_patterns,
                "active_notification_channel": channel,
                "sns_publish_statement_count": len(sns_send_statements),
                "sns_publish_scoped_to_site_topic": sns_scoped,
                "ses_send_statement_count": len(ses_send_statements),
                "ses_send_scoped_to_identity_with_condition": ses_scoped,
                "notification_send_scoped_to_active_channel": notification_send_scoped,
            }
        )
        write_json(self.run_dir / "BLAST-002-iam.json", inventory)

        passed = all(
            (
                rbac_all_denied,
                not forbidden,
                not broad_not_action,
                sagemaker_read_only,
                notification_send_scoped,
                "home_kube=absent" in filesystem_probe,
                "kubeconfig_env=unset" in filesystem_probe,
            )
        )
        self.record_case(
            case_id,
            "PASS" if passed else "FAIL",
            checks={
                "control_plane_write_rbac_all_denied": rbac_all_denied,
                "unexpected_bound_grants": binding_errors,
                "home_kube_absent": "home_kube=absent" in filesystem_probe,
                "kubeconfig_env_unset": "kubeconfig_env=unset" in filesystem_probe,
                "forbidden_hyperpod_mutations_present": forbidden,
                "sagemaker_describe_list_only": sagemaker_read_only,
                "sagemaker_action_pattern_count": len(sagemaker_patterns),
                "active_notification_channel": channel,
                "notification_send_scoped": notification_send_scoped,
            },
        )
        if not passed:
            raise CheckError(f"{case_id} failed")
