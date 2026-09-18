"""Self-contained registry requests for current and pre-registry CPU runtimes."""

LEGACY_REGISTRY_MARKER = "GPU_FAULT_LEGACY_REGISTRY_API"
UNAVAILABLE_REGISTRY_MARKER = "GPU_FAULT_REGISTRY_UNAVAILABLE"

_HTTP_PREAMBLE = r"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

def request_json(request):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            print("GPU_FAULT_REGISTRY_UNAVAILABLE")
            raise SystemExit(75) from None
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

DIRECT_SYNC_SCRIPT = r'''
import json
import os
import sys
from pathlib import Path

import psycopg

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
                    raise ValueError("installation resource identity cannot change")
    print(len(resources["resources"]))

try:
    synchronize()
except Exception as exc:
    print("registry database synchronization failed: " + type(exc).__name__,
          file=sys.stderr)
    raise SystemExit(1) from None
'''
