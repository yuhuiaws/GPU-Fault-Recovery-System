"""Count firing critical alerts in the AMP Alertmanager.

Request on stdin: ``{"region": "...", "workspace_id": "..."}``.
Response: ``{"count": N, "alerts": [{"alertname": ..., "severity": ...}]}``.

Unlike its neighbours this one runs on the **deploy host**, not inside a Pod:
the query is SigV4-signed against the AMP workspace with the deploy host's own
credentials, and the control-plane Pods deliberately have no such IAM identity
(their ADOT role grants ``aps:RemoteWrite`` and nothing else). It lives here
anyway because the reason to move these out of string literals -- ruff, mypy,
readability, a name in the deploy log instead of 40 lines of body -- has nothing
to do with where they execute. `boto3` is a deploy-host dependency and is
available for the same reason.

``active=true&silenced=false&inhibited=false`` is what makes this a release gate
rather than a dashboard: an operator who silenced an alert during the release has
said it is not a release blocker, and re-reading it as one would make silences
unusable exactly when they are needed.

A query that never got an answer -- no credentials to sign it with, AMP
throttling or unreachable -- is not a count of zero and not a failed gate
either: the probe prints ``{"unavailable": "<reason>"}`` and the engine takes
the sample again (``regional_release_observation``). Any other failure, an
HTTP 403 included, still exits non-zero.
"""

import json
import sys
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError


def _unavailable(reason: str) -> None:
    print(json.dumps({"unavailable": reason}, sort_keys=True))


def main() -> None:
    request = json.load(sys.stdin)
    region = request["region"]
    workspace_id = request["workspace_id"]
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
        f"{workspace_id}/alertmanager/api/v2/alerts"
        "?active=true&silenced=false&inhibited=false"
    )
    try:
        credentials = boto3.Session(region_name=region).get_credentials()
        frozen = credentials.get_frozen_credentials() if credentials else None
    except BotoCoreError:
        frozen = None
    if frozen is None:
        _unavailable("no-credentials")
        return
    signed = AWSRequest(method="GET", url=url, headers={"Accept": "application/json"})
    SigV4Auth(frozen, "aps", region).add_auth(signed)
    http_request = urllib.request.Request(
        url,
        headers={key: str(value) for key, value in signed.headers.items()},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:
            alerts = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code not in (429, 503):
            raise
        _unavailable("throttled")
        return
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        _unavailable("network")
        return
    critical = [
        {
            "alertname": (item.get("labels") or {}).get("alertname"),
            "severity": (item.get("labels") or {}).get("severity"),
        }
        for item in alerts
        if (item.get("labels") or {}).get("severity") == "critical"
    ]
    print(json.dumps({"count": len(critical), "alerts": critical}, sort_keys=True))


main()
