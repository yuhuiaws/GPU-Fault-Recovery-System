"""Append-only audit history of release state transitions (review H5).

`gpu-fault-regional-release-state` is one ConfigMap that every checkpoint
overwrites, so after a release nobody can answer "who ran which phase when,
against which plan". Every `save_state` now also appends one entry here:

* `gpu-fault-release-history` ConfigMap, key `history.ndjson`, bounded to the
  newest `HISTORY_MAX_ENTRIES` lines so it cannot outgrow the 1 MiB limit;
* a mirror at `$GPU_FAULT_RELEASE_HISTORY_DIR/history.ndjson` (the admin
  state directory, passed down by `gpu-fault-admin`), so the audit survives a
  namespace that was deleted.

Entries carry digests, never content: the state and plan documents live in
the state ConfigMap, and the command line is redacted before it is recorded.

The ConfigMap is read once per process, on the first checkpoint that needs it,
and appended to in memory from then on: the site operation lock guarantees one
mutating engine process at a time, so nothing else writes the document while
this process holds it, and every kubectl call from the deploy host costs about
1.2 s -- re-reading before each of a release's ~15 checkpoints paid that twice
(an existence probe and the read) for an answer this process had just written.
The one read tolerates absence itself (`--ignore-not-found`), so a first
release does not pay a separate existence probe either.
A failed history write is announced on stderr and does not fail the release --
the checkpoint that matters was already persisted, and blocking a rollback on
an audit ConfigMap would be a new outage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.operator_identity import (
    local_operator_identity,
    resolve_operator_identity,
)

HISTORY_CONFIG_MAP = "gpu-fault-release-history"
HISTORY_KEY = "history.ndjson"
HISTORY_MAX_ENTRIES = 200
HISTORY_DIR_ENV = "GPU_FAULT_RELEASE_HISTORY_DIR"
HISTORY_MIRROR_FILE = "history.ndjson"
# Any argument naming a credential is dropped whole, as is the value following
# a credential-named option; the query/fragment part of a path is dropped too.
_SECRET_OPTION = re.compile(r"(?i)(token|secret|password|credential|key)")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _redacted_command_line(argv: list[str]) -> str:
    words: list[str] = []
    skip_value = False
    for word in argv[1:]:
        if skip_value:
            words.append("<redacted>")
            skip_value = False
            continue
        if word.startswith("-") and "=" in word:
            option, _value = word.split("=", 1)
            if _SECRET_OPTION.search(option):
                words.append(f"{option}=<redacted>")
                continue
            words.append(word.split("?", 1)[0])
            continue
        if word.startswith("-") and _SECRET_OPTION.search(word):
            words.append(word)
            skip_value = True
            continue
        words.append(word.split("?", 1)[0] if "token=" in word.lower() else word)
    return " ".join(shlex.quote(word) for word in words)


def operator_identity(release: Any) -> str:
    """The AWS caller ARN when the host has credentials, else user@host.

    The same resolver `gpu-fault-admin` attributes its own writes with, so one
    operator has one name across the admin audit trail and this history.
    Resolved once per process: a release checkpoints dozens of times and an
    STS round trip per checkpoint would be the slowest thing in the log.
    """

    cached = getattr(release, "_operator_identity", None)
    if isinstance(cached, str) and cached:
        return cached
    identity = resolve_operator_identity(fallback=local_operator_identity())
    release._operator_identity = identity
    return identity


def build_history_entry(
    release: Any, *, phase: str, state_text: str, argv: list[str]
) -> dict[str, Any]:
    plan = release.state.get("execution_plan")
    return {
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        # The release the written state describes. For the release's own
        # transaction that is this process's candidate; for the one recorded
        # checkpoint a candidate writes on another release's behalf -- the
        # commit of a complete, uncommitted live release -- it is that live
        # release, which the entry must be attributed to (deploy #32,
        # 2026-09-09, logged its commit of 7194b5261380 under 72ebbaa66a9f).
        "release_id": str(release.state.get("release_id") or release.release_id),
        "phase": phase,
        "release_lifecycle": release.state.get("release_lifecycle"),
        "state_sha256": sha256_text(state_text),
        "plan_sha256": (
            sha256_text(json.dumps(plan, sort_keys=True)) if plan is not None else None
        ),
        "operator": operator_identity(release),
        "command": _redacted_command_line(argv),
    }


def _existing_entries(release: Any) -> list[str]:
    """The history lines as this process last knew them.

    Served from the release object after the first read; `record_release_history`
    updates that copy only once its apply has succeeded, so a failed write leaves
    the next checkpoint reading the ConfigMap again rather than trusting a line
    the cluster never saw. A dry run never gets here (nothing is recorded), so
    the first read stays lazy.
    """

    cached = getattr(release, "_release_history_lines", None)
    if cached is not None:
        return list(cached)
    # `--ignore-not-found`: a first release has no history yet, and that answer
    # is an empty document rather than a failed read (or a separate probe).
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            HISTORY_CONFIG_MAP,
            "--ignore-not-found",
        )
    )
    raw = (value.get("data") or {}).get(HISTORY_KEY) or ""
    return [line for line in str(raw).splitlines() if line.strip()]


def _mirror(entry_line: str) -> None:
    directory = os.environ.get(HISTORY_DIR_ENV, "").strip()
    if not directory:
        return
    path = Path(directory).expanduser() / HISTORY_MIRROR_FILE
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry_line + "\n")
    path.chmod(0o600)


def record_release_history(release: Any, *, phase: str, state_text: str) -> None:
    """Append one transition; never raise into the release."""

    # A release without a runner (dry-run builders, narration-only fakes)
    # has nothing to write with; ``never raise`` covers that too.
    runner = getattr(release, "runner", None)
    if runner is None or getattr(runner, "dry_run", False):
        return
    try:
        entry = build_history_entry(
            release, phase=phase, state_text=state_text, argv=list(sys.argv)
        )
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        lines = [*_existing_entries(release), line][-HISTORY_MAX_ENTRIES:]
        # One `kubectl apply` per checkpoint; the read happened once per process.
        release.runner.run(
            release._cpu("apply", "-f", "-"),
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": HISTORY_CONFIG_MAP,
                        "namespace": release.config.namespace,
                        "labels": {"gpu-fault.io/release-history": "true"},
                    },
                    "data": {HISTORY_KEY: "\n".join(lines) + "\n"},
                }
            ),
        )
        release._release_history_lines = lines
        _mirror(line)
    except Exception as exc:
        print(
            f"WARNING: release history entry for phase {phase} was not recorded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
