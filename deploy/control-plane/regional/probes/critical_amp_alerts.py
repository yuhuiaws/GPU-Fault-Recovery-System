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
"""

import json
import sys
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def main() -> None:
    request = json.load(sys.stdin)
    region = request["region"]
    workspace_id = request["workspace_id"]
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
        f"{workspace_id}/alertmanager/api/v2/alerts"
        "?active=true&silenced=false&inhibited=false"
    )
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are unavailable for AMP alert query")
    signed = AWSRequest(method="GET", url=url, headers={"Accept": "application/json"})
    SigV4Auth(credentials.get_frozen_credentials(), "aps", region).add_auth(signed)
    http_request = urllib.request.Request(
        url,
        headers={key: str(value) for key, value in signed.headers.items()},
    )
    with urllib.request.urlopen(http_request, timeout=30) as response:
        alerts = json.loads(response.read())
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
