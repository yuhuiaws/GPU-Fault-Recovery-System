"""Detect site-specific tokens without embedding a live AWS account."""

from __future__ import annotations

import re

STATE_DIRECTORY = "/secure/gpu-fault-bootstrap"
NODE_NAME_PREFIX = "gpu-fault-gpu-1-"
AWS_ACCOUNT_NUMBER = re.compile(r"(?<![0-9A-Za-z])(?!0{12})[0-9]{12}(?![0-9A-Za-z])")


def site_topology_leaks(source: str) -> list[str]:
    leaks = [token for token in (STATE_DIRECTORY, NODE_NAME_PREFIX) if token in source]
    leaks.extend(f"AWS account {match}" for match in AWS_ACCOUNT_NUMBER.findall(source))
    return leaks
