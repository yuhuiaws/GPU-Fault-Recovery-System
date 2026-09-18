"""The NOTIFY008 CPU payload contains only its new portable probe modules."""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional.probes.notify008_protocol import ProbeError

PROBES = Path(__file__).with_name("probes")
REQUIRED = frozenset(
    {
        "notify008_identity.py",
        "notify008_postgres.py",
        "notify008_probe.py",
        "notify008_process.py",
        "notify008_protocol.py",
        "notify008_provider.py",
    }
)


def source_bundle() -> dict[str, str]:
    paths = {path.name: path for path in PROBES.glob("notify008_*.py")}
    if set(paths) != REQUIRED or any(path.is_symlink() for path in paths.values()):
        raise ProbeError("NOTIFY008 portable probe closure differs")
    return {name: paths[name].read_text(encoding="utf-8") for name in sorted(paths)}
