"""Deliver only the pinned acceptance code, never runtime code or credentials."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
COMMON = (
    "scripts/e2e/regional/late_ownership_contract.py",
    "scripts/e2e/regional/late_ownership_barrier.py",
)
ENTRIES = {
    "executor": "scripts/e2e/regional/probes/late_ownership_executor_probe.py",
    "node": "scripts/e2e/regional/probes/late_ownership_node_probe.py",
    "reset-interval": "scripts/e2e/regional/probes/destr015_physical_probe.py",
}


def stdin_loader(program: str) -> str:
    """A small exec argument; the measured source travels over bounded stdin."""
    encoded = program.encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        "import hashlib,sys\n"
        f"source=sys.stdin.read({len(encoded)})\n"
        f"if hashlib.sha256(source.encode('ascii')).hexdigest() != {digest!r}:\n"
        " raise SystemExit(125)\n"
        "exec(compile(source,'<late-ownership-probe>','exec'))\n"
    )


def probe_program(
    role: Literal["executor", "node", "reset-interval"],
) -> tuple[str, str]:
    files: tuple[str, ...] = (*COMMON, ENTRIES[role])
    if role in {"node", "reset-interval"}:
        files += (
            "scripts/e2e/regional/late_ownership_trace.py",
            "scripts/e2e/regional/probes/late_ownership_tracer_child.py",
        )
    if role == "reset-interval":
        files += ("scripts/e2e/regional/destr015_physical_evidence.py",)
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in files}
    raw = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    entry = ENTRIES[role].removesuffix(".py").replace("/", ".")
    program = f"""
import base64, hashlib, importlib, json, os, pathlib, sys, tempfile
raw = base64.b64decode({encoded!r}, validate=True)
if hashlib.sha256(raw).hexdigest() != {digest!r}:
    raise SystemExit(125)
sources = json.loads(raw)
if set(sources) != set({files!r}):
    raise SystemExit(125)
with tempfile.TemporaryDirectory(prefix="gpu-fault-late-ownership-") as temporary:
    root = pathlib.Path(temporary)
    for name, text in sources.items():
        target = root / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        target.chmod(0o600)
    for package in ("scripts", "scripts/e2e", "scripts/e2e/regional", "scripts/e2e/regional/probes"):
        (root / package / "__init__.py").write_text("", encoding="ascii")
    sys.path.insert(0, str(root))
    module = importlib.import_module({entry!r})
    raise SystemExit(module.main())
"""
    return program, digest
