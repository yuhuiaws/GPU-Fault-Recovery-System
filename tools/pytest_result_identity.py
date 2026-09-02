from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "git command failed while identifying pytest results: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed.stdout


def source_identity(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    digest.update(_git_bytes(root, "rev-parse", "HEAD").strip())
    digest.update(b"\0diff\0")
    digest.update(_git_bytes(root, "diff", "--binary", "HEAD"))
    untracked = [
        Path(os.fsdecode(value))
        for value in _git_bytes(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        if value
    ]
    for relative in sorted(untracked, key=lambda value: value.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"untracked pytest input leaves repository: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsupported untracked pytest input: {relative}")
        digest.update(b"\0untracked\0")
        digest.update(relative.as_posix().encode())
        digest.update(f"{path.stat().st_mode & 0o777:04o}".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
