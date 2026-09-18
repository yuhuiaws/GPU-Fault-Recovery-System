from __future__ import annotations

import secrets
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_commands import CommandResult, matches_not_found, wait_until
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.cluster_join_evidence import join_activation_is_irreversible
from gpu_fault.admin.cluster_join_context import (
    ensure_kube_context,
    restore_current_context,
)
from gpu_fault.admin.cluster_join_state import (
    JOIN_RELEASE_STEPS,
    complete_step,
    join_release_started,
    step_done,
)
from gpu_fault.admin import cluster_removal_kubernetes as kubernetes_cleanup
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.process_supervisor import ensure_supervision_safe
from gpu_fault.admin.site import (
    RenderedSite,
    effective_environment,
    load_site,
    materialized_release_config,
)

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest


def _cleanup_candidate(
    candidate: RenderedSite,
    *,
    cluster_id: str,
    state_dir: Path,
    attempt: int,
) -> None:
    with materialized_release_config(candidate) as config:
        result = join.run_driver(
            [
                str(
                    candidate.repository_root
                    / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
                ),
                "--config",
                str(config),
                "--scope",
                "gpu",
                "--cluster-id",
                cluster_id,
                "--mode",
                "clean",
                "--node-mode",
                "uninstall",
                "--state-file",
                str(state_dir / f"rollback-kubernetes-{attempt:03d}.json"),
                "--execute",
            ],
            cwd=candidate.repository_root,
            env=effective_environment(candidate),
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise BootstrapError(
            "GPU cleanup: " + diagnostic_text((result.stderr or "").strip())
        )


def _command(arguments: list[str], *, not_found: tuple[str, ...]) -> None:
    result = join.run_command(arguments)
    if result.returncode == 0:
        return
    if matches_not_found(
        CommandResult(result.stdout or "", result.stderr or "", result.returncode),
        not_found,
    ):
        return
    raise BootstrapError(
        f"rollback command failed ({result.returncode}): "
        f"{' '.join(arguments[:3])}: {diagnostic_text(result.stderr.strip())}"
    )


def _iam_role(role: object, *, label: str, errors: list[str]) -> None:
    if not isinstance(role, dict) or not role.get("role_arn"):
        return
    role_name = str(role["role_arn"]).rsplit("/", 1)[-1]
    policy = str(role.get("inline_policy_name") or "")
    if policy:
        try:
            _command(
                [
                    "aws",
                    "iam",
                    "delete-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-name",
                    policy,
                ],
                not_found=("NoSuchEntity",),
            )
        except BootstrapError as exc:
            errors.append(f"{label} policy rollback: {exc}")
    try:
        _command(
            ["aws", "iam", "delete-role", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
    except BootstrapError as exc:
        errors.append(f"{label} role rollback: {exc}")


def _network(network: dict[str, Any], site: RenderedSite) -> None:
    if network.get("pending_mutation"):
        raise BootstrapError("join network mutation outcome is unproven")
    region = str(site.release_config["aws_region"])
    for eip in network.get("created_ingress_eips", []):
        _command(
            [
                "aws",
                "ec2",
                "revoke-security-group-ingress",
                "--region",
                region,
                "--group-id",
                str(site.release_config["nlb"]["security_group"]),
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            not_found=("InvalidPermission.NotFound",),
        )
    if network.get("association_created"):
        _command(
            [
                "aws",
                "route53",
                "disassociate-vpc-from-hosted-zone",
                "--hosted-zone-id",
                str(network["hosted_zone_id"]),
                "--vpc",
                f"VPCRegion={region},VPCId={network['vpc_id']}",
            ],
            not_found=("VPCAssociationNotFound",),
        )
        join._wait_vpc_association_absent(
            hosted_zone_id=str(network["hosted_zone_id"]),
            region=region,
            vpc_id=str(network["vpc_id"]),
        )


def _owned_namespace(
    request: JoinClusterRequest,
    *,
    local: dict[str, Any],
    state: dict[str, Any],
    context: str,
    kubeconfig: Path,
) -> bool:
    if context and local.get("namespace_creation_started"):
        observed_uid = join.probe_join_namespace(
            CommandRunner(), site=request.site, kubeconfig=kubeconfig, context=context
        )
        if observed_uid is not None and observed_uid != local.get("namespace_uid"):
            raise join.JoinTargetIdentityError(
                "GPU join namespace changed before rollback"
            )
        return observed_uid is not None
    if context and step_done(state, "LOCAL_INPUTS_READY"):
        raise join.JoinTargetIdentityError(
            "GPU join has no recorded namespace ownership"
        )
    return False


def _node_ownership(
    request: JoinClusterRequest,
    state: dict[str, Any],
    *,
    target: dict[str, Any],
    local: dict[str, Any],
    kubectl: list[str],
) -> dict[str, str]:
    evidence = state.get("evidence") or {}
    keys_possible = step_done(state, "NODE_KEYS_STARTED") or bool(
        (evidence.get("PREREQUISITES_READY") or {}).get("node_keys")
    )
    if not keys_possible and not join_release_started(state):
        return {}
    proof = evidence.get("NODE_NAMES_VERIFIED") or {}
    nodes = local.get("nodes") or []
    uids = local.get("node_uids")
    digests = proof.get("expected_key_sha256")
    if (
        not isinstance(uids, dict)
        or set(uids) != set(nodes)
        or not isinstance(digests, dict)
        or set(digests) != set(nodes)
        or proof.get("cluster_id")
        != (evidence.get("DISCOVERED") or {}).get("cluster_id")
        or proof.get("eks_arn") != target.get("eks_arn")
        or proof.get("node_uids") != uids
        or any(
            not isinstance(value, str) or len(value) != 64 for value in digests.values()
        )
    ):
        raise BootstrapError("join rollback has no complete node/key ownership proof")
    kubernetes_cleanup.clear_installer_annotations(
        partial(join.run_command, environment=effective_environment(request.site)),
        kubectl,
        hyperpod_name=str(target["hyperpod_name"]),
        node_uids=uids,
        annotations=(),
    )
    return dict(digests)


def _delete_namespace(
    request: JoinClusterRequest, kubectl: list[str], expected_uid: str
) -> None:
    namespace = str(request.site.release_config["namespace"])
    runner = partial(join.run_command, environment=effective_environment(request.site))
    kubernetes_cleanup.request_namespace_deletion(
        runner, kubectl, namespace, expected_uid
    )

    def absent() -> bool:
        document = kubernetes_cleanup.namespace_document(runner, kubectl, namespace)
        if document is not None and document["metadata"]["uid"] != expected_uid:
            raise BootstrapError("join namespace changed during rollback")
        return document is None

    wait_until(absent, description="joined namespace deletion", timeout_seconds=600)


def rollback(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    if join_activation_is_irreversible(state):
        raise BootstrapError("join activation has started; rollback is forbidden")
    ensure_supervision_safe(allow_interrupted=True)
    resuming = state.get("phase") in {"ROLLBACK_STARTED", "ROLLBACK_FAILED"}
    state["phase"] = "ROLLBACK_STARTED"
    write_json_atomic(state_path, state)
    evidence = state.get("evidence") or {}
    discovery = evidence.get("DISCOVERED") or {}
    cluster_id = str(discovery.get("cluster_id") or "")
    local = (
        evidence.get("LOCAL_INPUTS_READY") or evidence.get("LOCAL_INPUTS_STARTED") or {}
    )
    if step_done(state, "REMOTE_ROLLBACK_COMPLETED"):
        _finish(request, state_dir=state_dir, state_path=state_path, state=state)
        return
    errors = []
    registry_joined = step_done(state, "JOINED")
    candidate_path = Path(
        str((evidence.get("CANDIDATE_READY") or {}).get("site_file") or "")
    )
    candidate: RenderedSite | None = None
    target = discovery.get("target") or {}
    context = str(target.get("context") or "") if local else ""
    kubeconfig = Path(
        str(local.get("gpu_kubeconfig") or join._gpu_kubeconfig(request.site))
    )
    if context and resuming:
        with join._KUBECONFIG_THREAD_LOCK:
            ensure_kube_context(request.site, kubeconfig, target)
    namespace_present = _owned_namespace(
        request, local=local, state=state, context=context, kubeconfig=kubeconfig
    )
    kubectl = ["kubectl", "--kubeconfig", str(kubeconfig), "--context", context]
    key_digests: dict[str, str] = {}
    if (
        any(step_done(state, step) for step in ("CANDIDATE_READY", *JOIN_RELEASE_STEPS))
        and not candidate_path.is_file()
    ):
        raise BootstrapError("GPU join rollback candidate is missing")
    if cluster_id and candidate_path.is_file():
        try:
            candidate = load_site(
                candidate_path, repository_root=request.site.repository_root
            )
            lifecycle = None
            if join_release_started(state):
                snapshot = join.membership_runtime_snapshot(candidate)
                states = snapshot.get("registry_cluster_states")
                if not isinstance(states, dict):
                    raise join.JoinTargetIdentityError(
                        "GPU join registry state is unknown"
                    )
                lifecycle = states.get(cluster_id)
                if lifecycle not in {None, "PENDING", "FAILED", "ROLLED_BACK"} or (
                    registry_joined and lifecycle is None
                ):
                    raise join.JoinTargetIdentityError(
                        "GPU join registry state does not permit rollback"
                    )
                registry_joined = lifecycle is not None
            key_digests = _node_ownership(
                request, state, target=target, local=local, kubectl=kubectl
            )
            if lifecycle in {"PENDING", "FAILED"}:
                join._run_rollout(candidate, "fail-cluster", cluster_id=cluster_id)
            if namespace_present:
                _cleanup_candidate(
                    candidate,
                    cluster_id=cluster_id,
                    state_dir=state_dir,
                    attempt=int(state.get("attempt") or 1),
                )
        except Exception as exc:
            errors.append("kubernetes/control rollback: " + diagnostic_text(str(exc)))
            state["phase"] = "ROLLBACK_FAILED"
            state["rollback_errors"] = errors
            write_json_atomic(state_path, state)
            raise BootstrapError("; ".join(errors)) from exc
    else:
        key_digests = _node_ownership(
            request, state, target=target, local=local, kubectl=kubectl
        )
    nodes = list(local.get("nodes") or [])
    if join_release_started(state):
        join._clear_installer_annotations(
            request.site,
            {
                **target,
                "expected_node_uids": dict(local.get("node_uids") or {}),
                "hyperpod_cluster_name": str(target.get("hyperpod_name") or ""),
            },
            nodes,
        )
    if key_digests:
        try:
            join._remove_node_action_keys(
                request.site, nodes, expected_key_sha256=key_digests
            )
        except Exception as exc:
            errors.append("node key rollback: " + diagnostic_text(str(exc)))
    if context and namespace_present:
        try:
            _delete_namespace(request, kubectl, str(local["namespace_uid"]))
        except Exception as exc:
            errors.append("namespace rollback: " + diagnostic_text(str(exc)))
    prerequisites = evidence.get("PREREQUISITES_READY") or {}
    network = prerequisites.get("network")
    if isinstance(network, dict):
        try:
            _network(network, request.site)
        except Exception as exc:
            errors.append("network rollback: " + diagnostic_text(str(exc)))
    for prerequisite, label in (
        ("executor_role", "Executor"),
        ("adot_writer_role", "ADOT writer"),
    ):
        _iam_role(prerequisites.get(prerequisite), label=label, errors=errors)
    if not errors and candidate is not None:
        try:
            from gpu_fault.admin.cluster_join_commit import rollback_membership

            execution = join.JoinExecution(
                target=join._identity(dict(target)),
                cluster_id=cluster_id,
                discovery=dict(discovery),
                local=dict(local),
                prerequisites=dict(prerequisites),
                candidate=candidate,
            )
            rollback_membership(request, execution=execution, joined=registry_joined)
        except Exception as exc:
            errors.append("membership rollback: " + diagnostic_text(str(exc)))
    if errors:
        state["phase"] = "ROLLBACK_FAILED"
        state["rollback_errors"] = errors
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(state_path, state)
        raise BootstrapError("; ".join(errors))
    complete_step(
        state_path,
        state,
        "REMOTE_ROLLBACK_COMPLETED",
        {"registry_present": registry_joined},
    )
    _finish(request, state_dir=state_dir, state_path=state_path, state=state)


def _retain_token(
    request: JoinClusterRequest,
    *,
    target: dict[str, Any],
    cluster_id: str,
    token_file: Path,
    state_path: Path,
    state: dict[str, Any],
) -> Path:
    retained = request.site.source.parent / "secure" / f"{cluster_id}.token"
    binding = {
        "path": str(retained),
        "cluster_id": cluster_id,
        "eks_arn": str(target["eks_arn"]),
        "hyperpod_arn": str(target["hyperpod_arn"]),
    }
    source = token_file
    if not source.is_file() and state.get("retained_cluster_token") == binding:
        source = retained
    if not source.is_file() or source.is_symlink():
        raise BootstrapError("registered join token is unavailable for retry")
    value = source.read_text(encoding="utf-8")
    if retained.exists() and (
        retained.is_symlink()
        or not secrets.compare_digest(retained.read_text(encoding="utf-8"), value)
    ):
        raise BootstrapError("retained join token conflicts with the registered token")
    join.write_secret(retained, value)
    state["retained_cluster_token"] = binding
    write_json_atomic(state_path, state)
    return retained


def _remove_context(
    request: JoinClusterRequest, local: dict[str, Any], context: str
) -> None:
    kubeconfig = str(local.get("gpu_kubeconfig") or join._gpu_kubeconfig(request.site))
    with join._KUBECONFIG_THREAD_LOCK:
        contexts = join.run_command(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "config",
                "get-contexts",
                "-o",
                "name",
            ],
            environment=effective_environment(request.site),
        )
        if contexts.returncode:
            raise BootstrapError("cannot inspect GPU kubeconfig during rollback")
        names = (contexts.stdout or "").splitlines()
        if context in names:
            result = join.run_command(
                [
                    "kubectl",
                    "--kubeconfig",
                    kubeconfig,
                    "config",
                    "delete-context",
                    context,
                ],
                environment=effective_environment(request.site),
            )
            if result.returncode:
                raise BootstrapError(
                    "kube context rollback: " + diagnostic_text(result.stderr.strip())
                )
        restore_current_context(
            request.site, Path(kubeconfig), deleted=context, remaining=names
        )


def _finish(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    evidence = state.get("evidence") or {}
    discovery = evidence.get("DISCOVERED") or {}
    target = discovery.get("target") or {}
    cluster_id = str(discovery.get("cluster_id") or "")
    local = (
        evidence.get("LOCAL_INPUTS_READY") or evidence.get("LOCAL_INPUTS_STARTED") or {}
    )
    context = str(target.get("context") or "") if local else ""
    try:
        token_file = Path(str(local.get("token_file") or ""))
        secure = state_dir / "secure"
        allowed_tokens = {
            (secure / f"{cluster_id}.token").resolve(),
            (request.site.source.parent / "secure" / f"{cluster_id}.token").resolve(),
        }
        if local and (
            token_file.resolve() not in allowed_tokens
            or Path(str(local.get("fleet_master_file") or "")).resolve()
            != (secure / "fleet-master").resolve()
        ):
            raise BootstrapError(
                "join rollback credential paths are outside its ownership"
            )
        retained_token: Path | None = None
        if (evidence.get("REMOTE_ROLLBACK_COMPLETED") or {}).get("registry_present"):
            retained_token = _retain_token(
                request,
                target=target,
                cluster_id=cluster_id,
                token_file=token_file,
                state_path=state_path,
                state=state,
            )
        if context:
            _remove_context(request, local, context)
        for path in {
            token_file,
            secure / f"{cluster_id}.token",
            Path(str(local.get("fleet_master_file") or "")),
            secure / "fleet-master",
        }:
            if local and path != retained_token and path.is_file():
                path.unlink()
    except Exception as exc:
        state["phase"] = "ROLLBACK_FAILED"
        state["rollback_errors"] = [diagnostic_text(str(exc))]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(state_path, state)
        raise
    state["phase"] = "ROLLED_BACK"
    state["completed_steps"] = [
        item for item in ("PRECHECKED", "DISCOVERED") if step_done(state, item)
    ]
    state["evidence"] = {
        key: value
        for key, value in evidence.items()
        if key in {"PRECHECKED", "DISCOVERED"}
    }
    state.pop("rollback_errors", None)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
