"""Reject credentials and oversized binary dumps under artifacts/."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
MAX_FILE_BYTES = 5 * 1024 * 1024
SECRET_KIND = re.compile(
    rb"^[ \t-]*kind:[ \t]*Secret[ \t]*$",
    re.MULTILINE,
)
WHEEL_KEY = re.compile(rb"gpu_fault_control_plane-[^\r\n:]*\.whl[ \t]*:")
# A credential does not have to arrive as ``kind: Secret``. The registry
# baseline that every capacity run wrote carried the live regional cluster
# token as a plain ``"token": "<64 hex>"`` pair in JSON, which both the
# Secret-manifest rule and the wheel rule walk straight past. Match the
# key/value shape instead of the document kind so the rule holds for JSON,
# YAML and .env alike.
#
# The boundary excludes only alphanumerics, not ``_`` or ``-``: the keys that
# actually carry live credentials here are compound
# (``execution_token``, ``node_action_secret_key``, ``x-api-token``), and a
# ``\w`` boundary walked past every one of them. Redacted digests still stay
# out of the match because ``token_sha256`` has no ``:`` right after ``token``.
CREDENTIAL_KEY = re.compile(
    rb"(?<![A-Za-z0-9])"
    rb"(?P<key>[A-Za-z0-9_.-]*"
    rb"(?:token|password|passwd|secret_key|api_key|access_key|private_key))"
    rb"(?![A-Za-z0-9])"
    rb"[\"']?\s*[:=]\s*"
    rb"[\"']?(?P<value>[^\s\"',}\]]{24,})",
    re.IGNORECASE,
)
PLACEHOLDER_PREFIXES = (b"REPLACE_WITH", b"<", b"$", b"{{", b"CHANGEME", b"example")
HEX_TOKEN = re.compile(rb"\A[0-9a-fA-F]{32,}\Z")


def contains_wheel_binary_data(data: bytes) -> bool:
    lines = data.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped != b"binaryData:":
            continue
        indent = len(line) - len(stripped)
        for child in lines[index + 1 :]:
            child_stripped = child.lstrip()
            if not child_stripped:
                continue
            child_indent = len(child) - len(child_stripped)
            if child_indent <= indent:
                break
            if WHEEL_KEY.search(child_stripped):
                return True
    return False


def looks_like_credential(value: bytes) -> bool:
    """Tell a live credential from a placeholder or an opaque identifier.

    Deliberately narrow: the point is a rule that never fires on the
    evidence we do want kept (Secret *names*, run ids, timestamps,
    ``REPLACE_WITH_*`` templates), so the one shape it does catch stays
    actionable instead of being muted after the first false positive.
    """
    if value.startswith(PLACEHOLDER_PREFIXES):
        return False
    if len(set(value)) < 8:
        return False
    if HEX_TOKEN.match(value):
        return True
    classes = (
        any(byte.islower() for byte in value.decode("latin-1"))
        + any(byte.isupper() for byte in value.decode("latin-1"))
        + any(byte.isdigit() for byte in value.decode("latin-1"))
    )
    # A hyphenated identifier (``gpu-fault-node-action-secret``) is a name,
    # not a secret; a urlsafe token is not hyphen-delimited that way.
    return classes == 3 and b"-" not in value


def credential_values(data: bytes) -> list[str]:
    found = []
    for match in CREDENTIAL_KEY.finditer(data):
        value = match.group("value")
        if looks_like_credential(value):
            found.append(match.group("key").decode())
    return found


def is_exempt(path: Path, artifacts: Path) -> bool:
    """Only the tree's own README is exempt, not every file named README.md.

    ``path.name == "README.md"`` exempted a README at any depth, so a
    per-case ``artifacts/perf/<case>/README.md`` could carry a Secret or
    grow past the size ceiling without the guard ever reading it.
    """
    return path.relative_to(artifacts) == Path("README.md")


def check(artifacts: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(artifacts.rglob("*")):
        if not path.is_file() or is_exempt(path, artifacts):
            continue
        relative = path.relative_to(artifacts).as_posix()
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"{relative} is {size} bytes; maximum is {MAX_FILE_BYTES}")
            continue
        data = path.read_bytes()
        if SECRET_KIND.search(data):
            failures.append(f"{relative} contains a Kubernetes Secret manifest")
        if contains_wheel_binary_data(data):
            failures.append(f"{relative} contains wheel binaryData")
        for key in sorted(set(credential_values(data))):
            failures.append(
                f"{relative} carries a cleartext credential under key {key!r}; "
                "record a sha256 digest instead"
            )
    return failures


def content_files(artifacts: Path) -> list[Path]:
    return [
        path
        for path in artifacts.rglob("*")
        if path.is_file() and not is_exempt(path, artifacts)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS,
    )
    parser.add_argument("--require-content", action="store_true")
    args = parser.parse_args()

    content = content_files(args.artifacts_root)
    if not content:
        print("artifacts safety check: no local artifact files to scan")
        return 2 if args.require_content else 0
    failures = check(args.artifacts_root)
    if failures:
        for failure in failures:
            print(f"artifacts safety check failed: {failure}")
        return 1
    print(f"artifacts safety check passed: scanned {len(content)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
