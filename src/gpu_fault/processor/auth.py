from __future__ import annotations

import secrets


def processor_internal_token_valid(
    supplied: str | None,
    expected: str,
) -> bool:
    return bool(supplied and expected and secrets.compare_digest(supplied, expected))
