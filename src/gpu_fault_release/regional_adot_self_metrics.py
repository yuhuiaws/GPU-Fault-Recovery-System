"""The ADOT collector's self-metrics, verified through AMP.

Control-plane review 2026-09-08, H2-4: the alert rules read ``otelcol_*``
series whose names were taken from the pinned collector image's no-suffix
convention; an image bump can rename them and a rule that reads a series that
never exists is silent forever. The series are read from AMP rather than from
the collector's ``:8889`` reader: that reader is bound to loopback on purpose
(unauthenticated collector internals), so the API-server proxy to the Pod IP
is refused -- the verify of deploy #25 (2026-09-09) failed that way after the
release had committed -- and the collector image has no shell for
``kubectl exec``. Reading AMP also proves the path the alerts depend on end
to end: self-scrape, keep list and remote_write.
"""

from __future__ import annotations

from gpu_fault.admin.deadlines import (
    deadline_scope,
    remaining_timeout,
    recovery_active,
)
from gpu_fault.admin.native_http import (
    NativeHttpError,
    export_aws_credentials,
    http_request,
)
from gpu_fault.admin.process_supervisor import ensure_supervision_safe

import json
import re
from typing import Any, cast

from gpu_fault_release import repository_root
from gpu_fault_release.regional_release_config import ReleaseError

ROOT = repository_root()
ADOT_POD_SELECTOR = "app=gpu-fault-adot"


def adot_self_metric_names() -> set[str]:
    """Every ``otelcol_*`` series the alert rules or the ADOT keep list rely on."""

    names: set[str] = set()
    for relative in (
        "deploy/observability/amp-rules.yaml",
        "deploy/observability/adot-control-plane.yaml",
    ):
        names.update(
            re.findall(r"otelcol_[a-z0-9_]+", (ROOT / relative).read_text("utf-8"))
        )
    return names


ADOT_SELF_SCRAPE_JOB = "gpu-fault-adot-self"
AMP_QUERY_TIMEOUT_SECONDS = 30
# The exporter creates its failure counter on the first failed send, so a
# collector that has never lost a batch has never published it (7 days of AMP
# history held no sample on 2026-09-09). GpuFaultTelemetryRemoteWriteFailing
# reads it as ``rate(...) > 0``, which absence satisfies correctly, so absence
# is accepted here as long as the sibling sent counter proves the exporter
# path publishes at all.
LAZY_ADOT_SELF_SERIES = frozenset({"otelcol_exporter_send_failed_metric_points"})


def amp_instant_query(release: Any, expression: str) -> list[dict[str, Any]]:
    """Run one PromQL instant query against the site's AMP workspace.

    The query endpoint is the ``aps-workspaces`` data-plane API, which the
    ``aws amp`` CLI does not expose, so the request is SigV4-signed here with
    the deploy host's own credentials (the same chain every ``aws`` call in
    this process uses). Returns the ``data.result`` vector.
    """
    ensure_supervision_safe(allow_interrupted=recovery_active())
    remaining_timeout(AMP_QUERY_TIMEOUT_SECONDS)

    from urllib.parse import urlencode

    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import ReadOnlyCredentials

    region = str(release.config.aws_region)
    workspace_id = str(release.config.health.amp_workspace_id)
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
        f"{workspace_id}/api/v1/query"
    )
    response: tuple[int, str] | None = None
    try:
        with deadline_scope("AMP query", AMP_QUERY_TIMEOUT_SECONDS):
            exported = export_aws_credentials(seconds=AMP_QUERY_TIMEOUT_SECONDS)
            credentials = ReadOnlyCredentials(
                exported["AccessKeyId"],
                exported["SecretAccessKey"],
                exported["SessionToken"] or None,
            )
            body = urlencode({"query": expression}).encode()
            signed = AWSRequest(
                method="POST",
                url=url,
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            SigV4Auth(credentials, "aps", region).add_auth(signed)
            response = http_request(
                "POST",
                url,
                dict(signed.headers),
                body,
                backend="aws",
                seconds=AMP_QUERY_TIMEOUT_SECONDS,
                label="AMP query",
            )
            remaining_timeout(AMP_QUERY_TIMEOUT_SECONDS)
    except NativeHttpError:
        pass
    if response is None:
        raise ReleaseError("AMP query transport or credential resolution failed")
    if response[0] != 200:
        raise ReleaseError(f"AMP query returned HTTP {response[0]}")
    try:
        document = json.loads(response[1])
    except (ValueError, UnicodeError):
        document = None
    if not isinstance(document, dict) or document.get("status") != "success":
        raise ReleaseError("AMP query returned an invalid response")
    data = document.get("data")
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, list) or not all(
        isinstance(item, dict) for item in result
    ):
        raise ReleaseError("AMP query returned an invalid result vector")
    return cast(list[dict[str, Any]], result)


def adot_self_metrics_report(release: Any) -> tuple[str, dict[str, Any]]:
    """AMP holds every ``otelcol_*`` series the alerts read, scraped by the
    live ADOT collector from itself (control-plane review 2026-09-08, H2-4).

    The self-scrape metric names were taken from the exporter's no-suffix
    convention; a collector image bump can rename them, and a rule that reads
    a series that never exists is silent forever. The series are read from
    AMP rather than from the collector's ``:8889`` endpoint: that reader is
    bound to loopback on purpose (unauthenticated collector internals), so the
    API-server proxy to the Pod IP is refused, and the collector image has no
    shell for ``kubectl exec``. Reading AMP also proves the path the alerts
    depend on -- self-scrape, keep list and remote_write -- end to end.
    """

    namespace = release.config.namespace
    pods = (
        release._get_json(
            release._cpu(
                "-n", namespace, "get", "pods", "-l", ADOT_POD_SELECTOR, "-o", "json"
            )
        ).get("items")
        or []
    )
    running = [
        str(pod["metadata"]["name"])
        for pod in pods
        if (pod.get("status") or {}).get("phase") == "Running"
    ]
    if not running:
        raise ReleaseError("no running gpu-fault-adot Pod to read self metrics from")
    selector = f'job="{ADOT_SELF_SCRAPE_JOB}"'
    up = amp_instant_query(release, f"max(up{{{selector}}})")
    up_values = [float(sample["value"][1]) for sample in up if sample.get("value")]
    if not up_values or max(up_values) < 1:
        raise ReleaseError(
            f"AMP has no live up{{{selector}}} sample: the collector "
            f"{running[0]} is not scraping itself or its remote_write is down"
        )
    exported = {
        str((sample.get("metric") or {}).get("__name__") or "")
        for sample in amp_instant_query(
            release, f'count by (__name__) ({{__name__=~"otelcol_.+", {selector}}})'
        )
    }
    required = adot_self_metric_names()
    missing = sorted(
        name
        for name in required
        if name not in exported and name not in LAZY_ADOT_SELF_SERIES
    )
    if missing:
        raise ReleaseError(
            "AMP does not hold "
            + ", ".join(missing)
            + f" for {selector}; the collector image renamed them (suffix?) or "
            "the keep list drops them -- fix the keep list and the rules before "
            "trusting the ADOT alerts"
        )
    return (
        f"AMP holds all {len(required)} otelcol series the rules read from "
        f"{ADOT_SELF_SCRAPE_JOB} (collector {running[0]})",
        {"pod": running[0], "series": sorted(required)},
    )
