"""Bounded node-local history; never combine attempts or process generations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any


class CommunicationHistory:
    def __init__(
        self, path: Path, *, max_bytes: int = 32 * 1024 * 1024, max_age: float = 1800
    ):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path, self.max_bytes, self.max_age = path, max_bytes, max_age
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sample (scope TEXT, epoch INTEGER, stamp REAL, payload TEXT, size INTEGER, PRIMARY KEY(scope,epoch))"
            )
        path.chmod(0o600)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=0.2)

    @staticmethod
    def scope(job: str, attempt: str, boot: str) -> str:
        return hashlib.sha256(json.dumps([job, attempt, boot]).encode()).hexdigest()

    def append(
        self, scope: str, epoch: int, stamp: float, document: dict[str, Any]
    ) -> None:
        raw = json.dumps(document, separators=(",", ":")).encode()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("raw communication sample exceeds budget")
        raw = zlib.compress(raw)
        size = len(raw)
        if size > min(self.max_bytes, 1_048_576):
            raise ValueError("communication sample exceeds history budget")
        with self.connect() as db:
            db.execute("DELETE FROM sample WHERE stamp < ?", (stamp - self.max_age,))
            db.execute(
                "INSERT OR REPLACE INTO sample VALUES (?,?,?,?,?)",
                (scope, epoch, stamp, raw, size),
            )
            while (
                db.execute("SELECT COALESCE(SUM(size),0) FROM sample").fetchone()[0]
                > self.max_bytes
            ):
                db.execute(
                    "DELETE FROM sample WHERE rowid=(SELECT rowid FROM sample ORDER BY stamp,scope LIMIT 1)"
                )
            db.execute(
                "DELETE FROM sample WHERE rowid NOT IN (SELECT rowid FROM sample ORDER BY stamp DESC LIMIT 2048)"
            )

    def read(
        self, scope: str, *, since: float, until: float, limit: int = 64
    ) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT epoch,stamp,payload FROM sample WHERE scope=? AND stamp>=? AND stamp<=? ORDER BY stamp DESC LIMIT ?",
                (scope, since, until, min(64, max(1, limit))),
            ).fetchall()
        result = []
        for epoch, stamp, raw in reversed(rows):
            decoder = zlib.decompressobj()
            value = decoder.decompress(raw, 8 * 1024 * 1024 + 1)
            if len(value) > 8 * 1024 * 1024 or not decoder.eof or decoder.unused_data:
                raise ValueError("invalid bounded history payload")
            result.append(
                {
                    "epoch": epoch,
                    "observed_at_epoch": stamp,
                    "document": json.loads(value),
                }
            )
        return result
