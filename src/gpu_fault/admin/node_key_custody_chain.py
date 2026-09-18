"""Verify every authority, identity and causal edge, without trusting snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Chain,
    Completed,
    CustodyBinding,
    CustodyError,
    KeyState,
    Started,
    Transaction,
    statement_sha256,
)


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CustodyError("custody receipt requires a timezone-aware timestamp")
    return value


def validate_authorization(authorization: Authorization) -> None:
    binding = authorization.binding
    site = binding.site
    if (
        not timedelta(0)
        < aware(authorization.expires_at) - aware(authorization.not_before)
        <= timedelta(hours=24)
        or len(set(binding.nodes.values())) != len(binding.nodes)
        or site.cpu_cluster_uid == site.gpu_cluster_uid
        or site.cpu_namespace_uid == site.gpu_namespace_uid
        or site.cpu_eks_arn == site.gpu_eks_arn
        or any(
            not arn.startswith(
                ("arn:aws:eks:", "arn:aws-us-gov:eks:", "arn:aws-cn:eks:")
            )
            or f":eks:{site.region}:" not in arn
            or ":cluster/" not in arn
            for arn in (site.cpu_eks_arn, site.gpu_eks_arn)
        )
        or authorization.purpose == "install"
        and (
            authorization.previous_receipt_sha256 is not None
            or authorization.rotate_node is not None
        )
        or authorization.purpose == "rotate"
        and (
            authorization.previous_receipt_sha256 is None
            or authorization.rotate_node not in binding.nodes
        )
    ):
        raise CustodyError("custody authorization has inconsistent identity or purpose")


def authorize_now(authorization: Authorization, now: datetime) -> None:
    validate_authorization(authorization)
    if not authorization.not_before <= aware(now) < authorization.expires_at:
        raise CustodyError("custody authorization is outside its approved window")


def validate_transition(
    authorization: Authorization,
    started: Started,
    completed: Completed,
    previous: Transaction | None,
) -> None:
    binding = authorization.binding
    validate_authorization(authorization)
    if (
        started.authorization_sha256 != statement_sha256(authorization)
        or not authorization.not_before
        <= aware(started.observed_at)
        <= aware(completed.observed_at)
        <= authorization.expires_at
        or completed.started_sha256 != statement_sha256(started)
        or completed.gpu.keys != started.planned_keys
        or completed.cpu.keys != completed.gpu.keys
        or completed.gpu.uid == completed.cpu.uid
        or started.cpu_other_keys_sha256 != completed.cpu_other_keys_sha256
        or started.before_cpu_uid is not None
        and started.before_cpu_uid != completed.cpu.uid
        or set(completed.gpu.keys) != set(binding.nodes)
        or len({state.sha256 for state in completed.gpu.keys.values()})
        != len(completed.gpu.keys)
        or any(
            state.node_uid != binding.nodes[node]
            or state.sha256 == binding.master.sha256
            for node, state in completed.gpu.keys.items()
        )
    ):
        raise CustodyError("custody receipt chain has an invalid provisioning edge")
    if previous is None:
        if (
            authorization.purpose != "install"
            or started.before_gpu is not None
            or any(state.generation != 1 for state in completed.gpu.keys.values())
        ):
            raise CustodyError("installation custody requires prospective absent keys")
        return
    old = previous.completed.statement
    if (
        authorization.purpose != "rotate"
        or authorization.previous_receipt_sha256 != statement_sha256(previous)
        or previous.authorization.statement.binding != binding
        or previous.activated is None
        or aware(previous.activated.statement.observed_at) > started.observed_at
        or authorization.rotate_node not in previous.activated.statement.nodes
        or started.before_gpu is None
        or started.before_gpu.uid != old.gpu.uid
        or started.before_gpu.keys != old.gpu.keys
        or completed.gpu.uid != old.gpu.uid
        or completed.cpu.uid != old.cpu.uid
    ):
        raise CustodyError("rotation lacks its activated, identity-bound predecessor")
    for node, state in completed.gpu.keys.items():
        expected = old.gpu.keys[node]
        if node == authorization.rotate_node:
            if (
                state.generation != expected.generation + 1
                or state.sha256 == expected.sha256
            ):
                raise CustodyError(
                    "rotation did not advance exactly one key generation"
                )
        elif state != expected:
            raise CustodyError("rotation changed a sibling key")


def verify_chain(
    chain: Chain,
    crypto: CustodyCrypto,
    *,
    binding: CustodyBinding | None = None,
    require_activation: bool = False,
    now: datetime | None = None,
) -> Transaction:
    current = now or datetime.now(timezone.utc)
    previous: Transaction | None = None
    identifiers: set[str] = set()
    for transaction in chain.transactions:
        authorization = crypto.verify(transaction.authorization, "approval")
        started = crypto.verify(transaction.started, "provisioner")
        completed = crypto.verify(transaction.completed, "provisioner")
        if (
            authorization.transaction_id in identifiers
            or binding is not None
            and authorization.binding != binding
            or aware(completed.observed_at) > aware(current)
        ):
            raise CustodyError("custody chain is replayed, foreign or future-dated")
        identifiers.add(authorization.transaction_id)
        validate_transition(authorization, started, completed, previous)
        if transaction.activated is not None:
            activated = crypto.verify(transaction.activated, "witness")
            if (
                activated.completed_sha256 != statement_sha256(completed)
                or activated.binding_sha256 != statement_sha256(authorization.binding)
                or not completed.observed_at <= aware(activated.observed_at) <= current
                or not set(activated.nodes) <= set(authorization.binding.nodes)
                or any(
                    state.node_uid != authorization.binding.nodes[node]
                    or not state.endpoint.startswith("https://")
                    for node, state in activated.nodes.items()
                )
                or activated.retired_key_denied != (authorization.purpose == "rotate")
                or len({state.endpoint for state in activated.nodes.values()}) != 2
            ):
                raise CustodyError(
                    "custody activation receipt is unbound or incomplete"
                )
        elif require_activation or transaction is not chain.transactions[-1]:
            raise CustodyError("custody chain lacks an independent activation receipt")
        previous = transaction
    # Chain's strict nonempty schema supplies the final transaction.
    return chain.transactions[-1]


def planned_keys(
    binding: CustodyBinding,
    digests: dict[str, str],
    previous: Transaction | None,
    rotate_node: str | None,
) -> dict[str, KeyState]:
    if set(digests) != set(binding.nodes):
        raise CustodyError("custody key inventory differs from the authorized nodes")
    return {
        node: KeyState(
            node_uid=binding.nodes[node],
            generation=(
                previous.completed.statement.gpu.keys[node].generation
                + int(node == rotate_node)
                if previous is not None
                else 1
            ),
            sha256=digest,
        )
        for node, digest in digests.items()
    }
