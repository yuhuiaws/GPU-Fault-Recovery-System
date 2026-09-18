"""Administrator preparation, read-only resume and serialized custody execution."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Literal

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.deadlines import deadline_environment, remaining_timeout
from gpu_fault.admin.node_key_custody import (
    ProvisionInputs,
    producer_identity,
    witness_source_identity,
)
from gpu_fault.admin.node_key_custody_admin_config import (
    AdminCustodyRegistration,
    CUSTODY_DIRECTORY,
    load_admin_custody,
)
from gpu_fault.admin.node_key_custody_admin_probe import (
    AdminNodeKeyContext,
    CustodyReadRunner,
    current_binding,
    custody_verifier,
    verify_completed_custody,
)
from gpu_fault.admin.node_key_custody_chain import authorize_now
from gpu_fault.admin.node_key_custody_crypto import (
    CustodyCrypto,
    parse,
    private_command_environment,
    private_directory,
    read_regular,
    write_once,
)
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    CustodyBinding,
    CustodyError,
    Digest,
    Signed,
    Statement,
    canonical,
    statement_sha256,
)
from gpu_fault.admin.node_key_proof import read_node_key_proof

if TYPE_CHECKING:
    from gpu_fault.admin.node_key_custody_activation import CustodyActivation
    from gpu_fault.admin.site import RenderedSite

_KEY_MAP_LOCKS: dict[tuple[Path, str], RLock] = {}
_LOCKS_GUARD = Lock()


class CustodyPreparationRequired(BootstrapError):
    """A normal, persistent pause; no key mutation or rollback is authorized."""


class CustodyReconciliationRequired(BootstrapError):
    """Do not turn incomplete or foreign custody into automatic compensation."""


def bootstrap_custody_profile(
    state_dir: Path, repository_root: Path, existing_site: dict[str, Any] | None
) -> str:
    profile = (existing_site or {}).get("spec", {}).get("runtimeProfile", {})
    version = str(profile.get("version") or "hyperpod-v1")
    if existing_site is None or load_admin_custody(state_dir) is None:
        return version
    from gpu_fault_release.regional_runtime_profile import runtime_profile_policy_digest

    source = profile.get("source")
    template = profile.get("templateSource") or source
    if not isinstance(source, str) or not source or not isinstance(template, str):
        raise CustodyPreparationRequired(
            "custody requires a known Runtime Profile binding"
        )
    if runtime_profile_policy_digest(
        repository_root / source
    ) != runtime_profile_policy_digest(repository_root / template):
        raise CustodyPreparationRequired(
            "Runtime Profile policy is changing; custody cannot preauthorize an unresolved profile"
        )
    return version


class CustodyPreparation(Statement):
    schema_version: Literal[1] = 1
    kind: Literal["node-key-custody-preparation"] = "node-key-custody-preparation"
    authorization: Literal[False] = False
    binding: CustodyBinding
    producer_sha256: Digest
    witness_sha256: Digest


def key_map_lock(context: AdminNodeKeyContext) -> RLock:
    with _LOCKS_GUARD:
        return _KEY_MAP_LOCKS.setdefault(
            (context.cpu_kubeconfig.resolve(), context.namespace), RLock()
        )


def write_preparation(context: AdminNodeKeyContext, binding: CustodyBinding) -> Path:
    prepared = CustodyPreparation(
        binding=binding,
        producer_sha256=producer_identity(
            context.repository_root / "deploy/node/provision_node_action_keys.py"
        ),
        witness_sha256=witness_source_identity(context.repository_root),
    )
    directory = context.state_dir / CUSTODY_DIRECTORY / "preparations"
    directory.mkdir(mode=0o700, exist_ok=True)
    private_directory(directory)
    path = directory / (statement_sha256(prepared) + ".json")
    if path.exists():
        if read_regular(path) != canonical(prepared):
            raise CustodyError("custody preparation content changed")
    else:
        write_once(path, canonical(prepared))
    return path


def _validated_request(
    registration: AdminCustodyRegistration,
    context: AdminNodeKeyContext,
    binding: CustodyBinding,
    runner: CommandRunner,
) -> tuple[Path, ProvisionInputs, Signed[Authorization], CustodyCrypto]:
    reference = registration.selection.clusters[context.cluster.eks_arn]
    if reference is None:
        raise CustodyPreparationRequired(
            "custody needs an independently authorized request"
        )
    path = Path(reference)
    request = parse(ProvisionInputs, read_regular(path))
    crypto = custody_verifier(runner, registration)
    authorization = parse(
        Signed[Authorization], read_regular(Path(request.authorization))
    )
    approved = crypto.verify(authorization, "approval")
    if (
        approved.binding != binding
        or approved.producer_sha256
        != producer_identity(
            context.repository_root / "deploy/node/provision_node_action_keys.py"
        )
        or approved.witness_sha256 != witness_source_identity(context.repository_root)
        or hashlib.sha256(read_regular(Path(request.release_manifest))).hexdigest()
        != binding.release.manifest_sha256
        or registration.requests.get(context.cluster.eks_arn) != request
        or registration.transactions.get(context.cluster.eks_arn)
        != approved.transaction_id
        or registration.authorizations.get(context.cluster.eks_arn) != authorization
    ):
        raise CustodyError(
            "administrator custody authorization differs from current prepared inputs"
        )
    return path, request, authorization, crypto


def _completed(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    binding: CustodyBinding,
    request: ProvisionInputs,
    authorization: Signed[Authorization],
    crypto: CustodyCrypto,
    *,
    activation: CustodyActivation | None = None,
) -> dict[str, str] | None:
    chain_file = Path(request.state_directory) / (
        authorization.statement.transaction_id + ".chain.json"
    )
    if not chain_file.exists():
        return None
    head = verify_completed_custody(runner, context, crypto, binding, chain_file)
    if head.authorization != authorization:
        raise CustodyError(
            "administrator custody completion belongs to another authorization"
        )
    if activation is not None:
        activation.io.bind_completed(head.completed.statement)
    return {
        "cluster_id": context.cluster_id,
        "custody_chain": str(chain_file),
        "custody_chain_sha256": hashlib.sha256(read_regular(chain_file)).hexdigest(),
        "custody_authorization_sha256": statement_sha256(authorization),
        "runtime_activation": "NOT_PROVED",
    }


def run_custody_provisioner(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    path: Path,
    registration: AdminCustodyRegistration,
    environment: dict[str, str],
    activation: CustodyActivation | None,
) -> None:
    guard = activation.key_writer() if activation is not None else nullcontext(None)
    with guard as binding:
        arguments = [
            str(context.repository_root / "deploy/node/provision-node-action-keys.sh"),
            "--custody-request",
            str(path),
            "--custody-trust-sha256",
            registration.trust_sha256,
            "--custody-input-sha256",
            registration.input_sha256[context.cluster.eks_arn + ":inputs"],
        ]
        if binding is not None:
            state_path, state_sha256 = binding
            arguments.extend(
                [
                    "--custody-activation-state",
                    str(state_path),
                    "--custody-activation-sha256",
                    state_sha256,
                ]
            )
        if context.on_write is not None:
            context.on_write()
        runner.run(
            arguments,
            env=deadline_environment(environment),
            cwd=context.repository_root,
            capture=True,
            sensitive=True,
            mutate=True,
            timeout_seconds=remaining_timeout(300),
        )


def provision_admin_custody(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    *,
    fleet_master_file: Path,
    probe_only: bool,
) -> dict[str, str] | None:
    try:
        registration = load_admin_custody(context.state_dir)
    except CustodyError as exc:
        raise CustodyReconciliationRequired(str(exc)) from None
    if (
        registration is None
        or context.cluster.eks_arn not in registration.selection.clusters
    ):
        return None
    with key_map_lock(context):
        activation = None
        try:
            binding = current_binding(runner, context, registration)
            if registration.selection.clusters[context.cluster.eks_arn] is None:
                if probe_only:
                    raise BootstrapMutationRequired("custody preparation is incomplete")
                path = write_preparation(context, binding)
                raise CustodyPreparationRequired(
                    f"node-key custody awaits independent authorization for {path}; "
                    "configure its signed request and resume the same deploy or join"
                )
            path, request, authorization, crypto = _validated_request(
                registration, context, binding, runner
            )
            if authorization.statement.purpose == "rotate":
                from gpu_fault.admin.node_key_custody_activation import (
                    CustodyActivation,
                    activation_io,
                )

                activation = CustodyActivation(
                    activation_io(runner, context, authorization.statement),
                    context,
                    authorization,
                )
            completed = _completed(
                runner,
                context,
                binding,
                request,
                authorization,
                crypto,
                activation=activation,
            )
            if completed is not None:
                load_admin_custody(context.state_dir)
                if activation is not None:
                    if probe_only:
                        completed.update(activation.probe())
                    else:
                        activation.prepare()
                        completed.update(activation.finish())
                return completed
            intent = Path(request.state_directory) / (
                authorization.statement.transaction_id + ".started.json"
            )
            if intent.exists():
                raise CustodyError(
                    "incomplete custody intent requires explicit reconciliation"
                )
            if probe_only:
                raise BootstrapMutationRequired("signed custody completion is missing")
            from datetime import datetime, timezone

            authorize_now(authorization.statement, datetime.now(timezone.utc))
            if authorization.statement.purpose == "install":
                gpu = read_node_key_proof(
                    CustodyReadRunner(runner), context.kubectl("gpu"), context.namespace
                )
                cpu = read_node_key_proof(
                    CustodyReadRunner(runner), context.kubectl("cpu"), context.namespace
                )
                if (
                    gpu is not None
                    or cpu is not None
                    and set(cpu.digests) & set(binding.nodes)
                ):
                    raise CustodyError(
                        "pre-existing keys cannot be assigned installation provenance"
                    )
            if activation is not None:
                activation.prepare()
            environment = {
                **private_command_environment(),
                "KUBECONFIG": str(context.gpu_kubeconfig),
                "GPU_FAULT_KUBECTL_CONTEXT": context.cluster.context,
                "GPU_FAULT_NAMESPACE": context.namespace,
                "GPU_FAULT_CLUSTER_ID": context.cluster_id,
                "GPU_FAULT_HYPERPOD_CLUSTER": context.cluster.hyperpod_name,
                "GPU_FAULT_FLEET_MASTER_FILE": str(fleet_master_file),
                "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(context.cpu_kubeconfig),
                "GPU_FAULT_CONTROL_PLANE_NAMESPACE": context.namespace,
                "GPU_FAULT_CONTROL_PLANE_CONTEXT": "",
                "GPU_FAULT_NODE_ACTION_KEYS_SECRET": "gpu-fault-node-action-keys",
                "GPU_FAULT_ROTATE_NODE_ACTION_KEY": authorization.statement.rotate_node
                or "",
                "GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON": json.dumps(binding.nodes),
            }
            run_custody_provisioner(
                runner, context, path, registration, environment, activation
            )
            if current_binding(runner, context, registration) != binding:
                raise CustodyError("custody bindings changed during provisioning")
            result = _completed(
                runner,
                context,
                binding,
                request,
                authorization,
                crypto,
                activation=activation,
            )
            if result is None:
                raise CustodyError("custody helper did not produce a signed completion")
            if activation is not None:
                result.update(activation.finish())
            load_admin_custody(context.state_dir)
            return result
        except BootstrapMutationRequired:
            raise
        except CustodyError as exc:
            raise CustodyReconciliationRequired(str(exc)) from None
        except BootstrapError as exc:
            if isinstance(exc, CustodyPreparationRequired):
                raise
            raise CustodyReconciliationRequired(
                "custody command failed; retain its intent and reconcile"
            ) from None
        except Exception:
            if activation is None:
                raise
            raise CustodyReconciliationRequired(
                "custody activation could not complete; retain its bound progress "
                "and reconcile before any compensation"
            ) from None


def site_custody_context(
    site: RenderedSite,
    cluster: ClusterIdentity,
    cluster_id: str,
    *,
    gpu_kubeconfig: Path | None = None,
) -> AdminNodeKeyContext:
    config = site.release_config
    return AdminNodeKeyContext(
        state_dir=site.source.parent,
        repository_root=site.repository_root,
        site_id=str(config["site_name"]),
        cpu_eks_arn=str(config["cpu_eks_arn"]),
        cpu_kubeconfig=Path(str(config["cpu_kubeconfig"])),
        gpu_kubeconfig=gpu_kubeconfig or Path(str(config["gpu_kubeconfig"])),
        namespace=str(config["namespace"]),
        cluster=cluster,
        cluster_id=cluster_id,
        release_manifest=Path(str(config["release"]["manifest"])),
        runtime_profile_version=str(config["runtime_profile"]["version"]),
        agent_config_digest=str(config["release"]["agent_config_digest"]),
    )
