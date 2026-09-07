"""Load the control-plane probe programs the release engine ships to Pods.

See ``probes/README.md`` for why the source is read as text instead of imported:
the engine has to carry the probe so that a new engine can probe an old Pod.

The read happens once per probe and is cached, so a missing or unreadable probe
fails on first use rather than in the middle of a wave, and so `command_log` can
echo a probe by name instead of by its whole body.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
