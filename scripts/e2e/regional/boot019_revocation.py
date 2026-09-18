"""Private, run-bound credential custody until BOOT-019 finishes revocation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


class RevocationCredentials:
    def __init__(self, run_dir: Path, *, site_path: Path, join_arn: str) -> None:
        self.directory = run_dir / "secure"
        self.scope = hashlib.sha256(
            f"{run_dir.resolve()}\0{site_path.resolve()}\0{join_arn}".encode()
        ).hexdigest()

    def _path(self, cluster_id: str) -> Path:
        key = hashlib.sha256(f"{self.scope}\0{cluster_id}".encode()).hexdigest()
        return self.directory / f"{key}.revoked-token"

    def capture(self, cluster_id: str, token: bytes) -> dict[str, Any]:
        if not token.strip():
            raise ValueError("revocation credential is empty")
        path = self._path(cluster_id)
        expected = hashlib.sha256(token).hexdigest()
        if path.exists() or path.is_symlink():
            self.load(
                {
                    "cluster_id": cluster_id,
                    "token_sha256": expected,
                    "custody_scope": self.scope,
                }
            )
        else:
            if self.directory.is_symlink():
                raise ValueError("revocation credential directory is unsafe")
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.chmod(0o700)
            fd, name = tempfile.mkstemp(dir=self.directory, prefix=".revocation-")
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(token)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "cluster_id": cluster_id,
            "token_storage": "private-bound-file",
            "token_sha256": expected,
            "custody_scope": self.scope,
        }

    def load(self, capture: dict[str, Any]) -> bytes:
        if capture.get("custody_scope") != self.scope:
            raise ValueError("revocation credential belongs to another run")
        path = self._path(str(capture["cluster_id"]))
        if (
            self.directory.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or self.directory.stat().st_mode & 0o077
            or path.stat().st_mode & 0o077
        ):
            raise ValueError("protected revocation credential is unavailable")
        token = path.read_bytes()
        if hashlib.sha256(token).hexdigest() != capture.get("token_sha256"):
            raise ValueError("revocation credential digest changed")
        return token

    def remove(self, capture: dict[str, Any]) -> None:
        if not self._path(str(capture["cluster_id"])).exists():
            return
        self.load(capture)
        self._path(str(capture["cluster_id"])).unlink()
