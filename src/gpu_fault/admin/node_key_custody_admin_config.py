"""Explicit administrator opt-in and content binding for node-key custody."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.node_key_custody import (
    ProvisionInputs,
    provisioning_input_identity,
)
from gpu_fault.admin.node_key_custody_chain import (
    validate_authorization,
    verify_chain,
)
from gpu_fault.admin.node_key_custody_crypto import (
    CustodyCrypto,
    parse,
    private_directory,
    read_regular,
)
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Chain,
    CustodyError,
    Digest,
    Identity,
    Signed,
    Statement,
    Trust,
    canonical,
    statement_sha256,
)

CUSTODY_DIRECTORY = "node-key-custody"
REGISTRATION_FILE = "registration.json"
EksArn = Annotated[
    str,
    StringConstraints(
        pattern=r"^arn:aws(?:-us-gov|-cn)?:eks:[a-z0-9-]+:[0-9]{12}:cluster/[A-Za-z0-9][A-Za-z0-9_-]*$"
    ),
]


class AdminCustodySelection(Statement):
    schema_version: Literal[1] = 1
    trust: Identity
    allow_staging: bool = False
    clusters: Annotated[dict[EksArn, Identity | None], Field(min_length=1)]


class AdminCustodyRegistration(Statement):
    schema_version: Literal[1] = 1
    state_directory: Identity
    selection: AdminCustodySelection
    trust_sha256: Digest
    input_sha256: dict[str, Digest]
    requests: dict[str, ProvisionInputs]
    transactions: dict[str, Digest]
    authorizations: dict[str, Signed[Authorization]]

    @model_validator(mode="after")
    def bound_request_snapshots(self) -> AdminCustodyRegistration:
        selected = {
            arn for arn, path in self.selection.clusters.items() if path is not None
        }
        if (
            self.requests.keys() != selected
            or self.transactions.keys() != selected
            or self.authorizations.keys() != selected
            or any(
                self.authorizations[arn].statement.transaction_id
                != self.transactions[arn]
                or self.authorizations[arn].statement.binding.site.gpu_eks_arn != arn
                for arn in selected
            )
        ):
            raise ValueError("custody registration has inconsistent request snapshots")
        return self


def registration_path(state_dir: Path) -> Path:
    return state_dir / CUSTODY_DIRECTORY / REGISTRATION_FILE


def configured_custody(state_dir: Path) -> bool:
    """A damaged registration is not permission to return to legacy behavior."""
    directory = state_dir / CUSTODY_DIRECTORY
    try:
        directory.lstat()
    except FileNotFoundError:
        return False
    private_directory(directory)
    return True


def _digest(path: Path) -> str:
    return hashlib.sha256(read_regular(path)).hexdigest()


def selection_input_identity(
    selection: AdminCustodySelection, trust_sha256: str
) -> dict[str, str]:
    trust_path = Path(selection.trust)
    if _digest(trust_path) != trust_sha256:
        raise CustodyError("administrator custody trust differs from its external pin")
    trust = parse(Trust, read_regular(trust_path))
    result = {
        "selection": hashlib.sha256(canonical(selection)).hexdigest(),
        "trust": trust_sha256,
    }
    for role in ("approval", "provisioner", "witness"):
        result[f"public:{role}"] = _digest(
            trust_path.parent / getattr(trust, role).public_key
        )
    for arn, reference in selection.clusters.items():
        if reference is None:
            continue
        request_path = Path(reference)
        result[f"{arn}:request"] = _digest(request_path)
        result[f"{arn}:inputs"] = provisioning_input_identity(
            request_path, trust_sha256
        )
        request = parse(ProvisionInputs, read_regular(request_path))
        if Path(request.trust).absolute() != trust_path.absolute():
            raise CustodyError(
                "administrator custody request uses a different trust source"
            )
        for name in ("authorization", "release_manifest", "previous_chain"):
            path = getattr(request, name)
            if path is not None:
                result[f"{arn}:{name}"] = _digest(Path(path))
    return result


def load_admin_custody(state_dir: Path) -> AdminCustodyRegistration | None:
    if not configured_custody(state_dir):
        return None
    path = registration_path(state_dir)
    value = parse(AdminCustodyRegistration, read_regular(path, private=True))
    if Path(
        value.state_directory
    ) != state_dir.resolve() or value.input_sha256 != selection_input_identity(
        value.selection, value.trust_sha256
    ):
        raise CustodyError(
            "administrator custody inputs changed; explicit reconfiguration is required"
        )
    return value


def authorize_successor(
    previous: AdminCustodyRegistration,
    arn: str,
    request: ProvisionInputs,
    authorization: Authorization,
    verifier: CustodyCrypto,
) -> None:
    prior = previous.requests[arn]
    completed_file = Path(prior.state_directory) / (
        previous.transactions[arn] + ".chain.json"
    )
    if (
        authorization.purpose != "rotate"
        or authorization.transaction_id == previous.transactions[arn]
        or request.previous_chain is None
        or request.retired_key_file is None
    ):
        raise CustodyError(
            "changing a started request requires an authorized activated successor"
        )
    completed = parse(Chain, read_regular(completed_file))
    archived = verify_chain(completed, verifier, binding=authorization.binding)
    activated = parse(Chain, read_regular(Path(request.previous_chain)))
    predecessor = verify_chain(
        activated, verifier, binding=authorization.binding, require_activation=True
    )
    if (
        archived.authorization != previous.authorizations[arn]
        or activated.transactions[:-1] != completed.transactions[:-1]
        or predecessor.model_copy(update={"activated": archived.activated}) != archived
        or statement_sha256(predecessor) != authorization.previous_receipt_sha256
        or predecessor.activated is None
        or authorization.rotate_node not in predecessor.activated.statement.nodes
    ):
        raise CustodyError("custody successor differs from its enrolled predecessor")


def configure_admin_custody(
    state_dir: Path,
    config_file: Path,
    trust_sha256: str,
    *,
    crypto: CustodyCrypto | None = None,
) -> AdminCustodyRegistration:
    with administrator_operation_lock(state_dir):
        root = state_dir.resolve()
        if configured_custody(root) and not registration_path(root).exists():
            raise CustodyError(
                "administrator custody registration is missing; reconciliation required"
            )
        supplied = parse(AdminCustodySelection, read_regular(config_file, private=True))
        base = config_file.absolute().parent
        selection = supplied.model_copy(
            update={
                "trust": str((base / supplied.trust).absolute()),
                "clusters": {
                    arn: str((base / path).absolute()) if path is not None else None
                    for arn, path in supplied.clusters.items()
                },
            }
        )
        identity = selection_input_identity(selection, trust_sha256)
        verifier = crypto or CustodyCrypto(Path(selection.trust), trust_sha256)
        if verifier.trust_sha256 != trust_sha256:
            raise CustodyError(
                "administrator custody verifier uses a different trust pin"
            )
        requests = {}
        transactions = {}
        authorizations = {}
        for arn, reference in selection.clusters.items():
            if reference is None:
                continue
            request = parse(ProvisionInputs, read_regular(Path(reference)))
            envelope = parse(
                Signed[Authorization], read_regular(Path(request.authorization))
            )
            authorization = verifier.verify(envelope, "approval")
            validate_authorization(authorization)
            if authorization.binding.site.gpu_eks_arn != arn:
                raise CustodyError(
                    "administrator custody request belongs to another cluster"
                )
            requests[arn] = request
            transactions[arn] = authorization.transaction_id
            authorizations[arn] = envelope
        destination = registration_path(root)
        if destination.exists():
            previous = parse(
                AdminCustodyRegistration, read_regular(destination, private=True)
            )
            if (
                previous.state_directory != str(root)
                or previous.trust_sha256 != trust_sha256
                or not previous.selection.clusters.keys() <= selection.clusters.keys()
            ):
                raise CustodyError(
                    "administrator custody enrollment cannot remove a cluster or replace its trust"
                )
            for arn, reference in previous.selection.clusters.items():
                if reference is not None and (
                    selection.clusters[arn] != reference
                    or any(
                        identity.get(key) != value
                        for key, value in previous.input_sha256.items()
                        if key.startswith(arn + ":")
                    )
                ):
                    prior = previous.requests[arn]
                    intent = Path(prior.state_directory) / (
                        previous.transactions[arn] + ".started.json"
                    )
                    if selection.clusters[arn] is None:
                        raise CustodyError(
                            "an enrolled custody request cannot be removed"
                        )
                    if intent.exists():
                        authorize_successor(
                            previous,
                            arn,
                            requests[arn],
                            authorizations[arn].statement,
                            verifier,
                        )
        if selection_input_identity(selection, trust_sha256) != identity:
            raise CustodyError("administrator custody inputs changed while configuring")
        directory = destination.parent
        directory.mkdir(mode=0o700, exist_ok=True)
        private_directory(directory)
        registration = AdminCustodyRegistration(
            state_directory=str(root),
            selection=selection,
            trust_sha256=trust_sha256,
            input_sha256=identity,
            requests=requests,
            transactions=transactions,
            authorizations=authorizations,
        )
        # This is administrator intent, not proof; only independent receipts
        # authorize key provisioning or establish historical custody.
        write_json_atomic(destination, registration.model_dump(mode="json"))
        return registration
