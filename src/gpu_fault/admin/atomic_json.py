"""One atomic writer for the JSON documents operator tooling keeps on disk.

Admin state, release plans, approval records and acceptance evidence are all read
back by a later command -- often a later command that decides whether a destructive
action may proceed. A half-written document there is worse than a missing one: it
parses as invalid JSON at best, and at worst it parses as a plan nobody approved. So
every one of those writes goes through this function, which makes the update
all-or-nothing and the result unreadable to other users.

Six near-copies of this used to exist, differing in exactly the ways that matter:
whether the parent directory's mode was enforced or only requested, whether the
temporary file was created privately or at a predictable path, and whether the bytes
were fsynced before the rename. Each copy was the strictest of them in some axis and
the weakest in another; this is the strict union.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Replace `path` with `value` as sorted, indented JSON, or leave it as it was.

    The bytes are written to a private temporary file in the same directory, flushed
    to the device, and only then renamed onto `path` -- a rename within one directory
    is atomic, so a concurrent reader sees either the previous document or this one,
    never a prefix of it. The fsync is what makes that hold across a host that loses
    power mid-release rather than only across a process that exits.

    The parent is forced to ``0o700`` and the file to ``0o600`` before any content
    reaches it, because these documents carry cluster names, node IDs and approval
    identities. Enforcing the directory mode matters even when the caller created it:
    ``mkdir(mode=...)`` is a no-op on a directory that already exists, so an
    inherited world-readable state directory would otherwise stay that way.
    """

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        # Including KeyboardInterrupt and SystemExit: an operator pressing Ctrl-C
        # must not leave a `.tmp` behind for the next run to trip over.
        temporary.unlink(missing_ok=True)
        raise
