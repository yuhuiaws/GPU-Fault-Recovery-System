"""The one form a node-key digest takes wherever a join or a removal reads one.

A node-key digest is the SHA-256 of a node's action-key bytes, written as
``hashlib``'s ``hexdigest`` by ``node_key_proof.node_key_digest`` and by the
join: exactly 64 lowercase hex characters (``gpu_fault.digests.SHA256_PATTERN``).
The join records the map it proved in ``NODE_NAMES_VERIFIED.expected_key_sha256``
and the removal its own in ``DISCOVERED.node_key_ownership``; the join retry,
the join rollback, the removal and the CPU key delete both of them end in read
such a map back from those journals.

Each reader used to carry its own check. The rollback's only counted 64
characters, while the removal and the CPU key delete wanted lowercase hex, so a
digest the rollback took (an uppercase one, or 64 characters of anything)
passed its ownership proof and the undo steps after it, and was refused only by
the key delete at the end. Every reader now refuses the same maps, where it
reads them, naming the nodes. A digest is never folded to lowercase: one the
system did not write proves nothing about the key it claims to describe.
"""

from __future__ import annotations

from collections.abc import Iterable

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.digests import SHA256_PATTERN

#: Nodes named in a refusal before the rest are counted.
NAMED_NODES = 8


def is_node_key_digest(value: object) -> bool:
    """``value`` is a node-key digest as the system writes one."""

    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _named(nodes: Iterable[object]) -> str:
    names = sorted(str(node) for node in nodes)
    shown = ", ".join(names[:NAMED_NODES])
    hidden = len(names) - NAMED_NODES
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def node_key_digests(
    value: object, nodes: Iterable[str], *, refusal: str
) -> dict[str, str]:
    """``value`` as the digest map of exactly ``nodes``, or ``refusal`` and why.

    The refusal names the nodes whose digest is missing, unexpected or not 64
    lowercase hex characters -- never a value, which need not be a digest at
    all.
    """

    if not isinstance(value, dict):
        raise BootstrapError(f"{refusal}: no node-key digest map")
    expected = set(nodes)
    missing, unexpected = expected - set(value), set(value) - expected
    if missing or unexpected:
        gaps = [
            *([f"no digest for {_named(missing)}"] if missing else []),
            *([f"a digest for unproved {_named(unexpected)}"] if unexpected else []),
        ]
        raise BootstrapError(
            f"{refusal}: the node-key digest map has {'; '.join(gaps)}"
        )
    malformed = [
        node for node, digest in value.items() if not is_node_key_digest(digest)
    ]
    if malformed:
        raise BootstrapError(
            f"{refusal}: the node-key digest of {_named(malformed)} is not a "
            "SHA-256 hex digest (64 lowercase hex characters)"
        )
    return {str(node): str(digest) for node, digest in value.items()}
