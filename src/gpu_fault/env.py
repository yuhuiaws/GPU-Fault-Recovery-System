"""Boolean environment switches, parsed one way everywhere.

Before this module every read site spelled its own rule: some accepted only
the literal ``true``, some ``1/true/yes``, one ``!= "false"``. An operator who
wrote ``GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR=1`` got the switch silently
off. Every switch now accepts the same tokens, and a value that is none of
them is a misconfiguration that raises instead of reading as "off".

``scripts/generate-env-reference.py`` reads the token sets below straight out
of this module's AST, so the inventory that start-up validation enforces
cannot drift from what :func:`env_bool` accepts.
"""

from __future__ import annotations

import os
from typing import Mapping

TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
BOOLEAN_TOKENS = TRUE_TOKENS | FALSE_TOKENS


def invalid_boolean_message(name: str) -> str:
    """The one refusal text; it never quotes the value, which may be a secret."""

    return f"{name} must be one of {'/'.join(sorted(BOOLEAN_TOKENS))}"


def env_bool(
    name: str,
    default: bool = False,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the switch ``name``; unset or blank means ``default``.

    Case and surrounding whitespace are ignored. Anything that is neither a
    true nor a false token raises ``ValueError`` naming the variable.
    """

    environment: Mapping[str, str] = os.environ if environ is None else environ
    raw = environment.get(name, "")
    token = raw.strip().lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    if not token:
        return default
    raise ValueError(invalid_boolean_message(name))


def parse_bool(raw: str, *, name: str) -> bool:
    """Parse text that was already read from somewhere other than the process.

    Blank is refused here: without a variable to be unset there is no default
    to fall back to. The compare itself lives in :func:`env_bool` alone, so the
    one-entry mapping keeps this a second entry point rather than a second rule.
    """

    if not raw.strip():
        raise ValueError(invalid_boolean_message(name))
    return env_bool(name, environ={name: raw})
