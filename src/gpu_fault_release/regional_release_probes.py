"""Load the control-plane probe programs the release engine ships to Pods.

See ``probes/README.md`` for why the source is read as text instead of imported:
the engine has to carry the probe so that a new engine can probe an old Pod.

The read happens once per probe and is cached, so a missing or unreadable probe
fails on first use rather than in the middle of a wave, and so `command_log` can
echo a probe by name instead of by its whole body.
"""

from __future__ import annotations

from pathlib import Path

from gpu_fault_release import repository_root

ROOT = repository_root()
# The probes stay next to the manifests under deploy/: they are shipped to Pods
# as source text, not imported, so they are release inputs rather than package
# modules (``deploy/control-plane/regional/probes/README.md``).
PROBE_DIR = ROOT / "deploy/control-plane/regional/probes"

_SOURCES: dict[str, str] = {}
_NAMES: dict[str, str] = {}


def probe_source(name: str) -> str:
    cached = _SOURCES.get(name)
    if cached is not None:
        return cached
    path = PROBE_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"control-plane probe is missing: {path}")
    source = path.read_text(encoding="utf-8")
    _SOURCES[name] = source
    _NAMES[source] = name
    return source


def probe_label(source: str) -> str | None:
    """Name the probe a source string came from, for command echoing.

    ``Runner.run`` prints every argv it executes. A probe passed as ``-c`` put
    its whole body into the deploy log -- measured at 24% of the lines of an
    upgrade -- which buries the kubectl trace an operator is actually reading.
    Only probes already loaded are named; an unrecognised ``-c`` payload is
    echoed as before rather than silently relabelled.
    """

    name = _NAMES.get(source)
    return None if name is None else f"<probe:{name}>"


def echoed_arguments(arguments: list[str]) -> list[str]:
    """Replace a probe body in the command echo with the probe's name.

    Only the echo changes. ``args`` itself is untouched, so the bytes on the
    wire and the ``-c`` calling convention are exactly what they were -- which
    is the whole point of shipping the probe as source (``probes/README.md``).

    Without this, every probe put its full body on one ``+ python3 -c ...``
    line. The probe programs are commented, so that was 24% of the lines of an
    upgrade, and it pushed the kubectl trace an operator actually reads off the
    screen.
    """

    echoed = list(arguments)
    for index, argument in enumerate(echoed):
        if index == 0 or echoed[index - 1] != "-c":
            continue
        label = probe_label(argument)
        if label is not None:
            echoed[index] = label
    return echoed


def command_label(args: list[str], *, sensitive: bool = False) -> str:
    """The command exactly as the ``+`` echo has always shown it.

    Not shortened: an argument that is not a probe body is the release's real
    input -- a manifest, a selector, a node name -- and the trace an operator
    reads to reconstruct what ran has to keep it whole. Shortening belongs to the
    elapsed line, which is a pointer rather than a record.
    """

    if sensitive:
        return "<sensitive command>"
    return " ".join([Path(args[0]).name, *echoed_arguments(args[1:])])
