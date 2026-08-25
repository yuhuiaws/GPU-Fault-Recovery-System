from datetime import datetime, timedelta, timezone
from enum import StrEnum


class RemoteCommandStatus(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    WAITING = "WAITING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


def lease_deadline(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)
