"""The one place a SHA-256 hex digest is recognised and normalised.

Every artifact pin, config digest, plan digest and approval record in this system is
a 64-character lowercase SHA-256 hex string, and every one of them is eventually
compared for equality: a node refuses an installer bundle whose digest does not match
its pin, a rollout refuses an agent whose config digest drifted, an approval refuses
a plan it did not sign. Two spellings of the same digest compare unequal, which turns
a fail-closed check into a mismatch nobody ordered -- so the pattern and the
normalisation live here instead of in the eight modules that each carried a private
copy of the same regex.
"""

from __future__ import annotations

import re

# `\A`/`\Z` rather than `^`/`$`: `$` also matches immediately before a trailing
# newline, so `^[0-9a-f]{64}$` accepts a digest with a stray newline under `.match()`
# -- which is how a digest read from a file or a command's stdout arrives. Every
# caller today uses `fullmatch`, where the two spellings agree; this keeps the pattern
# correct for the caller who reaches for `match` instead.
SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
DIGEST_VALUE_MESSAGE = "digest must be a SHA-256 hex value"


def normalized_sha256(value: str | None) -> str | None:
    """Return `value` lowercased once it is known to be a SHA-256 digest.

    `None` passes through, because most digest fields are optional pins and "no pin
    declared" is a different state from "pin is unusable". Mixed-case input is
    accepted and folded rather than rejected: this is the contract the three model
    validators that shared this body already had, and the digest is only ever used as
    an identifier, so case is not information.
    """

    if value is None:
        return None
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(DIGEST_VALUE_MESSAGE)
    return normalized
