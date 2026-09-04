"""Stop the test suite from executing cluster-mutating binaries.

A release-orchestration unit test once reached a real ``kubectl`` because a
``Runner(dry_run=False)`` was constructed and one of its collaborators was left
unpatched. The test failed with ``command failed (1): kubectl`` only because no
cluster happened to be reachable; on a developer machine with a live kubeconfig
it would have talked to whatever context was current.

The guard denies a small list of binaries that mutate clusters, nodes or cloud
accounts. It deliberately does not block ``python``, ``git`` or ``shellcheck``:
several gates legitimately shell out to those, and blocking them would push
every caller into an opt-out that then hides the case this guard exists for.

A test that genuinely needs one of these binaries marks itself:

    @pytest.mark.allows_cluster_binaries("kubectl")
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterator, Sequence
from typing import Any

MARKER = "allows_cluster_binaries"

#: Binaries that change cluster, node or cloud state. ``ssh``/``scp`` are here
#: because node remediation reaches nodes over them.
DENIED_BINARIES = frozenset(
    {
        "aws",
        "docker",
        "eksctl",
        "helm",
        "kubeadm",
        "kubectl",
        "nsenter",
        "nvidia-smi",
        "scp",
        "ssh",
        "systemctl",
    }
)


class ClusterBinaryBlocked(AssertionError):
    """Raised when a test tries to execute a cluster-mutating binary."""


def _candidate_names(argument: Any) -> Iterator[str]:
    """Yield plausible binary names for one ``Popen`` argument list or string."""
    if isinstance(argument, (str, bytes, os.PathLike)):
        text = os.fsdecode(argument)
        # A shell command string can chain several binaries; check every word
        # that looks like a command rather than only the first.
        try:
            words = shlex.split(text)
        except ValueError:
            words = text.split()
        for word in words:
            yield os.path.basename(word)
        return
    if isinstance(argument, Sequence):
        for item in argument:
            if isinstance(item, (str, bytes, os.PathLike)):
                yield os.path.basename(os.fsdecode(item))


def denied_binary(popen_args: Any, allowed: frozenset[str]) -> str | None:
    """Return the denied binary ``popen_args`` would run, if any."""
    for name in _candidate_names(popen_args):
        if name in DENIED_BINARIES and name not in allowed:
            return name
    return None


def install(monkeypatch: Any, allowed: frozenset[str]) -> None:
    """Wrap ``subprocess.Popen`` so denied binaries raise instead of running.

    ``subprocess.run``, ``check_output`` and ``check_call`` all resolve ``Popen``
    from the module namespace at call time, so one patch covers them.
    """
    real_popen = subprocess.Popen

    def guarded_popen(popen_args: Any = (), *args: Any, **kwargs: Any) -> Any:
        name = denied_binary(popen_args, allowed)
        if name is not None:
            raise ClusterBinaryBlocked(
                f"test tried to execute {name!r}, which mutates cluster or cloud "
                f"state: {popen_args!r}. Patch the collaborator that runs it, or "
                f"mark the test with @pytest.mark.{MARKER}({name!r})."
            )
        return real_popen(popen_args, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
