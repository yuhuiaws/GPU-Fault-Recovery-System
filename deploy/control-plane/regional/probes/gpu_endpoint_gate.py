"""DNS, TLS and cluster-token gate run from inside the GPU cluster.

Unlike every other probe here this one is not shipped through ``kubectl exec``:
``regional_gpu_bootstrap.verify_gpu_control_plane_endpoint`` puts it in the
``command`` of a short-lived Pod, because the question is whether a Pod *in that
cluster* can reach and authenticate to the control plane -- through that
cluster's DNS, with the CA and token from the connection Secret mounted into it.
Nothing the administrator's host can run answers that.

Request: ``CONTROL_PLANE_URL``, ``EXPECTED_HOSTNAME`` and ``PROBE_INCIDENT_ID``
from the environment, plus ``/tls/ca.crt``, ``/auth/cluster-token`` and
``/auth/cluster-id`` from the mounted Secret. The token is mounted rather than
passed through env so it stays out of the Pod spec and out of anything that
dumps env.
Response: one JSON object on stdout; any failure is an exception, so the Pod's
exit status carries the verdict.

Kept free of anything cluster-specific: everything that varies arrives through
the environment or the mounted Secret.
"""

import json
import os
import pathlib
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

base_url = os.environ["CONTROL_PLANE_URL"].rstrip("/")
expected_hostname = os.environ["EXPECTED_HOSTNAME"]
parsed = urllib.parse.urlsplit(base_url)
if parsed.scheme != "https" or parsed.hostname != expected_hostname:
    raise RuntimeError("control-plane URL does not use the expected HTTPS hostname")
# A join associates the GPU VPC with the private hosted zone moments before
# this Pod starts, and the VPC resolver keeps answering NXDOMAIN for a while
# (live 2026-09-12: still unresolved 50 s after the association). Keep asking
# until the name resolves or the wait runs out; the caller's Pod deadline is
# longer than this.
dns_deadline = time.monotonic() + float(
    os.environ.get("DNS_RESOLVE_WAIT_SECONDS", "240")
)


# Failures that mean "the endpoint is not there yet", bare or wrapped in
# URLError(reason=...): the name not resolving, the connection refused or
# timing out, or the TLS handshake timing out because a brand-new NLB still
# forwards to nothing (live 2026-09-13: "_ssl.c:981: The handshake operation
# timed out" 35 s after the NLB went active). A certificate that fails to
# verify is a real answer and is not in this set.
_NOT_YET_REACHABLE = (
    socket.gaierror,
    socket.timeout,
    TimeoutError,
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
)


def _not_yet_reachable(error: BaseException) -> bool:
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, _NOT_YET_REACHABLE):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, _NOT_YET_REACHABLE) and not isinstance(
        reason, ssl.SSLCertVerificationError
    )


def until_resolved(operation: Callable[[], Any]) -> Any:
    """Run ``operation`` until the endpoint stops being "not there yet".

    The name resolves through whichever CoreDNS replica answers, and right after
    the zone association one replica can still serve the cached NXDOMAIN while
    another already resolves (live 2026-09-13: the resolve loop passed and the
    very next request failed "Name or service not known"); a brand-new NLB
    accepts connections before it forwards them. Every request in this probe
    therefore retries on those failures until the wait runs out; anything else
    -- a rejected certificate, an HTTP error -- is raised at once.
    """

    while True:
        try:
            return operation()
        except (OSError, urllib.error.URLError) as error:
            if not _not_yet_reachable(error) or time.monotonic() >= dns_deadline:
                raise
            time.sleep(5)


addresses = until_resolved(
    lambda: sorted(
        {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    )
)
if not addresses:
    raise RuntimeError("control-plane DNS returned no addresses")
context = ssl.create_default_context(cafile="/tls/ca.crt")


def _status(target: Any) -> int:
    with urllib.request.urlopen(target, context=context, timeout=15) as response:
        return int(response.status)


status = until_resolved(lambda: _status(base_url + "/healthz"))
if status != 200:
    raise RuntimeError(f"healthz returned HTTP {status}")

# /healthz answers 200 for every caller, so it cannot see a token the registry
# does not hold. Claiming work is the first thing that asks, which is a whole
# phase later: by then the Secret is installed and every Executor, Watcher and
# Collector has been rolled onto it, and the reject surfaces only as a 403
# traceback inside Pod logs. Ask here, while the endpoint rollback still has
# nothing but the Secret to undo.
token = pathlib.Path("/auth/cluster-token").read_text().strip()
cluster = pathlib.Path("/auth/cluster-id").read_text().strip()
query = urllib.parse.urlencode({"incident_id": os.environ["PROBE_INCIDENT_ID"]})
probe = urllib.request.Request(
    base_url + "/v1/regional/executors/incident-ownership?" + query,
    headers={
        "Authorization": "Bearer " + token,
        "X-GPU-Fault-Cluster-ID": cluster,
    },
)
try:
    authenticated = until_resolved(lambda: _status(probe))
except urllib.error.HTTPError as error:
    if error.code in (401, 403):
        raise RuntimeError(
            "control plane rejected the cluster token in secret "
            "gpu-fault-regional-connection with HTTP %d: %s; the registry holds "
            "no matching digest for cluster %s" % (error.code, error.reason, cluster)
        ) from error
    raise
print(
    json.dumps(
        {
            "hostname": parsed.hostname,
            "addresses": addresses,
            "tls": "verified",
            "status": status,
            "cluster_token": "accepted",
            "authenticated_status": authenticated,
        },
        sort_keys=True,
    )
)
