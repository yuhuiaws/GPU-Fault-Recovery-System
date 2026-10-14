"""Self-contained registry requests for current and pre-registry CPU runtimes."""

from __future__ import annotations

import json
import re

from gpu_fault.installation_resources import (
    IDENTITY_CONFLICT,
    IDENTITY_FIELDS,
    RESOURCE_KEY_PATTERN,
)

LEGACY_REGISTRY_MARKER = "GPU_FAULT_LEGACY_REGISTRY_API"
UNAVAILABLE_REGISTRY_MARKER = "GPU_FAULT_REGISTRY_UNAVAILABLE"
#: A registry API 409 becomes this marker, the reason and the refused keys and
#: fields on stdout (exit 65) -- validated, never the response text itself.
CONFLICT_REGISTRY_MARKER = "GPU_FAULT_REGISTRY_CONFLICT"
CONFLICT_EXIT = 65
CONFLICT_REASONS = (IDENTITY_CONFLICT,)
_FIELD_PATTERN = re.compile(r"^[a-z_]{1,32}$")

_HTTP_PREAMBLE = (
    r"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

KEY = re.compile("""
    + repr(RESOURCE_KEY_PATTERN.pattern)
    + """)
FIELD = re.compile("""
    + repr(_FIELD_PATTERN.pattern)
    + """)
REASONS = """
    + repr(CONFLICT_REASONS)
    + r"""

def conflict_report(exc):
    try:
        body = json.loads(exc.read(65536))
        report = {
            "reason": body["reason"],
            "conflicts": [
                {"resource_key": item["resource_key"], "fields": list(item["fields"])}
                for item in body["conflicts"]
            ],
        }
    except Exception:
        return None
    if report["reason"] not in REASONS or not report["conflicts"] or not all(
        isinstance(item["resource_key"], str)
        and KEY.fullmatch(item["resource_key"])
        and item["fields"]
        and all(isinstance(name, str) and FIELD.fullmatch(name) for name in item["fields"])
        for item in report["conflicts"]
    ):
        return None
    return report

def request_json(request):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            print("GPU_FAULT_REGISTRY_UNAVAILABLE")
            raise SystemExit(75) from None
        report = conflict_report(exc) if exc.code == 409 else None
        if report is not None:
            print("GPU_FAULT_REGISTRY_CONFLICT " + json.dumps(report, separators=(",", ":")))
            raise SystemExit(65) from None
        print("registry HTTP request failed: " + str(exc.code), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        print("registry request failed: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None

def require_route(path):
    value = request_json("http://127.0.0.1:8080/openapi.json")
    if not isinstance(value, dict) or not isinstance(value.get("paths"), dict):
        print("registry API description is invalid", file=sys.stderr)
        raise SystemExit(1)
    if path not in value["paths"]:
        print("GPU_FAULT_LEGACY_REGISTRY_API")
        raise SystemExit(44)
"""
)


def conflict_report(
    returncode: int, stdout: str | None
) -> tuple[str, list[tuple[str, list[str]]]] | None:
    """``(reason, [(key, fields), ...])`` of a script's 409 report, else ``None``.

    Only the exact protocol counts -- exit 65 and one stdout line: the marker,
    a space and the report the script validated, checked again here.
    """

    lines = (stdout or "").strip().splitlines()
    marker, _, document = (lines[0] if len(lines) == 1 else "").partition(" ")
    if returncode != CONFLICT_EXIT or marker != CONFLICT_REGISTRY_MARKER:
        return None
    try:
        report = json.loads(document)
        reason = report["reason"]
        conflicts = [
            (item["resource_key"], list(item["fields"])) for item in report["conflicts"]
        ]
    except (TypeError, KeyError, ValueError):
        return None
    if (
        reason not in CONFLICT_REASONS
        or not conflicts
        or not all(
            isinstance(key, str)
            and RESOURCE_KEY_PATTERN.fullmatch(key)
            and fields
            and all(
                isinstance(name, str) and _FIELD_PATTERN.fullmatch(name)
                for name in fields
            )
            for key, fields in conflicts
        )
    ):
        return None
    return reason, conflicts


SYNC_SCRIPT = (
    _HTTP_PREAMBLE
    + r"""
require_route("/v1/installation-resources/sync")
payload = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
if len(payload) > 8 * 1024 * 1024:
    print("registry snapshot exceeds input limit", file=sys.stderr)
    raise SystemExit(1)
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/installation-resources/sync",
    data=payload,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
print(json.dumps(request_json(request), separators=(",", ":")))
"""
)

FETCH_SCRIPT = (
    _HTTP_PREAMBLE
    + r"""
require_route("/v1/installation-resources")
site = urllib.parse.quote(os.environ["GPU_FAULT_INSTALLATION_SITE_ID"], safe="")
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/installation-resources?site_id=" + site,
    headers={
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
print(json.dumps(request_json(request), separators=(",", ":")))
"""
)

DIRECT_SYNC_SCRIPT = (
    r"""
import json
import os
import sys
from pathlib import Path

import psycopg

IDENTITY = """
    + repr(("site_id", "resource_key", *IDENTITY_FIELDS))
    + r'''

class IdentityConflict(ValueError):
    pass

def changed_fields(stored, resource):
    def value(row, name):
        if name == "dependencies":
            return row.get(name) or []
        return row.get(name)
    return [name for name in IDENTITY if value(stored, name) != value(resource, name)]

def synchronize():
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("snapshot exceeds input limit")
    resources = json.loads(raw)
    credential_file = os.environ.get("GPU_FAULT_STORE_URL_FILE", "").strip()
    dsn = (
        Path(credential_file).read_text(encoding="utf-8").strip()
        if credential_file
        else os.environ.get("GPU_FAULT_STORE_URL", "").strip()
    )
    if not dsn:
        raise ValueError("database credential is unavailable")
    with psycopg.connect(dsn, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '60s'")
            cursor.execute("SET LOCAL lock_timeout = '10s'")
            for resource in sorted(
                resources["resources"], key=lambda item: item["resource_key"]
            ):
                key = resource["site_id"] + "/" + resource["resource_key"]
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES ('installation_resource', %s, %s::jsonb)
                    ON CONFLICT(kind, key)
                    DO UPDATE SET payload=excluded.payload
                    WHERE (
                        gpu_fault_objects.payload->>'site_id',
                        gpu_fault_objects.payload->>'resource_key',
                        gpu_fault_objects.payload->>'provider',
                        gpu_fault_objects.payload->>'resource_type',
                        gpu_fault_objects.payload->>'resource_id',
                        gpu_fault_objects.payload->>'resource_arn',
                        gpu_fault_objects.payload->>'region',
                        gpu_fault_objects.payload->>'account_id',
                        gpu_fault_objects.payload->>'ownership',
                        gpu_fault_objects.payload->>'delete_policy',
                        COALESCE(gpu_fault_objects.payload->'dependencies', '[]'::jsonb)
                    ) IS NOT DISTINCT FROM (
                        excluded.payload->>'site_id',
                        excluded.payload->>'resource_key',
                        excluded.payload->>'provider',
                        excluded.payload->>'resource_type',
                        excluded.payload->>'resource_id',
                        excluded.payload->>'resource_arn',
                        excluded.payload->>'region',
                        excluded.payload->>'account_id',
                        excluded.payload->>'ownership',
                        excluded.payload->>'delete_policy',
                        COALESCE(excluded.payload->'dependencies', '[]'::jsonb)
                    )
                    RETURNING key
                    """,
                    (key, json.dumps(resource, separators=(",", ":"))),
                )
                if cursor.fetchone() is None:
                    cursor.execute(
                        "SELECT payload FROM gpu_fault_objects "
                        "WHERE kind='installation_resource' AND key=%s",
                        (key,),
                    )
                    raise IdentityConflict(
                        resource["resource_key"]
                        + " ("
                        + ", ".join(changed_fields(cursor.fetchone()[0], resource))
                        + ")"
                    )
    print(len(resources["resources"]))

try:
    synchronize()
except IdentityConflict as exc:
    # Composed from validated registry keys and field names only.
    print("registry database synchronization failed: "
          + "installation resource identity cannot change: " + str(exc),
          file=sys.stderr)
    raise SystemExit(1) from None
except Exception as exc:
    print("registry database synchronization failed: " + type(exc).__name__,
          file=sys.stderr)
    raise SystemExit(1) from None
'''
)
