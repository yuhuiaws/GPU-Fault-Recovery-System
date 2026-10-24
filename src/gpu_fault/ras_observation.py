"""NCCL 2.28 JSON RAS observations; counts describe submission, not completion."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import time
from typing import Any

MAX_RAS_BYTES = 8 * 1024 * 1024


def query_ras(host: str, port: int, *, timeout: float = 4) -> dict[str, Any]:
    if not ipaddress.ip_address(host).is_loopback or not 1 <= port <= 65535:
        raise ValueError("RAS queries require a namespace-local loopback endpoint")
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as connection:
        stream = connection.makefile("rb")
        try:
            for command, expected in (
                (b"CLIENT PROTOCOL 2\n", b"SERVER PROTOCOL 2\n"),
                (b"TIMEOUT 2\n", b"OK\n"),
                (b"SET FORMAT json\n", b"OK\n"),
            ):
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                connection.sendall(command)
                reply = bytearray()
                while len(reply) < 256 and not reply.endswith(b"\n"):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("RAS handshake deadline")
                    connection.settimeout(remaining)
                    block = stream.read1(1)
                    if not block:
                        break
                    reply.extend(block)
                if reply != expected:
                    raise ValueError("unsupported RAS protocol or JSON capability")
            connection.sendall(b"VERBOSE STATUS\n")
            chunks, size = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("RAS response deadline")
                connection.settimeout(remaining)
                block = stream.read1(min(65536, MAX_RAS_BYTES + 1 - size))
                if not block:
                    break
                size += len(block)
                if size > MAX_RAS_BYTES:
                    raise ValueError("RAS response exceeds byte budget")
                chunks.append(block)
            return parse_ras(b"".join(chunks))
        finally:
            stream.close()


def parse_ras(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_RAS_BYTES:
        raise ValueError("RAS response exceeds byte budget")
    document = json.loads(raw)
    if not isinstance(document, dict) or not re.fullmatch(
        r"2\.28\.[0-9]+(?:[-+][A-Za-z0-9._-]+)?", str(document.get("nccl_version"))
    ):
        raise ValueError("unsupported NCCL RAS JSON version")
    groups = document.get("communicators")
    if (
        not isinstance(groups, list)
        or len(groups) > 4096
        or document.get("communicators_count") != len(groups)
    ):
        raise ValueError("invalid RAS communicator inventory")
    result, identities = [], set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("invalid RAS communicator")
        identity = (group.get("hash"), group.get("secondary_hash"))
        if (
            not all(isinstance(item, str) and item for item in identity)
            or identity in identities
        ):
            raise ValueError("ambiguous RAS communicator identity")
        identities.add(identity)
        members, missing, size = (
            group.get("ranks"),
            group.get("missing_ranks"),
            group.get("size"),
        )
        if (
            type(size) is not int
            or not 0 < size <= 65536
            or not isinstance(members, list)
            or not isinstance(missing, list)
            or group.get("ranks_count") != len(members)
            or group.get("missing_ranks_count") != len(missing)
            or len(members) + len(missing) != size
        ):
            raise ValueError("incomplete RAS inventory accounting")
        seen = set()
        for member in [*members, *missing]:
            if not isinstance(member, dict):
                raise ValueError("invalid RAS member")
            rank = member.get("rank")
            if type(rank) is not int or not 0 <= rank < size or rank in seen:
                raise ValueError("ambiguous RAS local rank")
            seen.add(rank)
            if (
                not isinstance(member.get("host"), str)
                or type(member.get("pid")) is not int
            ):
                raise ValueError("RAS member identity unavailable")
        for member in members:
            counts = member.get("collective_counts")
            if (
                not isinstance(counts, dict)
                or not counts
                or any(type(value) is not int or value < 0 for value in counts.values())
                or not isinstance(member.get("status"), dict)
            ):
                raise ValueError("RAS counters/status unavailable")
        result.append(
            {
                "identity": list(identity),
                "member_count": size,
                "members": members,
                "missing_members": missing,
                "complete": not missing
                and all(row["pid"] > 0 and row["host"] != "unknown" for row in members),
                "rank_namespace": "COMMUNICATOR_LOCAL",
                "counter_semantics": "ENQUEUED_OPERATION_COUNTS",
            }
        )
    metadata = document.get("ras")
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("timeouts_count")) is not int
        or metadata["timeouts_count"] < 0
    ):
        raise ValueError("RAS timeout coverage unavailable")
    return {
        "version": "nccl-ras-summary/v1",
        "nccl_version": document["nccl_version"],
        "groups": result,
        "timeouts_count": metadata["timeouts_count"],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "completion_progress_available": False,
    }
