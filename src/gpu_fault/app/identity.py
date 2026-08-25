from __future__ import annotations

import os


def pod_process_owner() -> str | None:
    pod_uid = os.getenv("POD_UID", "").strip()
    return f"{pod_uid}:{os.getpid()}" if pod_uid else None
