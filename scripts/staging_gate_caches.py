"""Where a deploy keeps what its gates remember, which is never the snapshot.

A deploy runs its static gates inside a freshly prepared source tree. The tool
caches are correctly absent from that tree -- they are ignored by Git, so they
are in neither the copied untracked set nor the payload digest -- and the cost is
that every deploy re-derives an identical answer from cold. So the caches live
here instead: in the state directory, outside the snapshot and outside the
repository, shared by every snapshot of the same content.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

TOOL_CACHE_DIRECTORY = "tool-caches"
PUBLIC_RELEASE_VERDICT = "public-release-verdict.json"
# mypy is the whole reason this exists: 17s of full analysis per deploy against
# 0.4s replayed from a cache it validates by content. Ruff is here for symmetry
# and is worth well under a second either way.
TOOL_CACHE_VARIABLES = (
    ("MYPY_CACHE_DIR", "mypy"),
    ("RUFF_CACHE_DIR", "ruff"),
)


def public_release_verdict_cache(state_dir: Path) -> Path:
    """Where the public-release gate records a pass, so the snapshot can reuse it.

    The gate binds the record to a digest of exactly the bytes it scanned, so the
    second call in a deploy -- the prepared snapshot, a copy of the tree the first
    call already cleared -- proves the content is identical rather than matching
    every pattern against it again.
    """

    return state_dir / PUBLIC_RELEASE_VERDICT


def tool_cache_environment(
    state_dir: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """``base`` with the tool caches pointed outside the snapshot.

    Anything already set wins, so a caller that needs an isolated run keeps it.
    """

    environment = {**(base if base is not None else os.environ)}
    caches = state_dir / TOOL_CACHE_DIRECTORY
    for name, subdirectory in TOOL_CACHE_VARIABLES:
        if environment.get(name, "").strip():
            continue
        path = caches / subdirectory
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        environment[name] = str(path)
    return environment
