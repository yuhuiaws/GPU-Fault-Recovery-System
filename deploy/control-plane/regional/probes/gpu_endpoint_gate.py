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
import urllib.error
import urllib.parse
import urllib.request

base_url = os.environ["CONTROL_PLANE_URL"].rstrip("/")
expected_hostname = os.environ["EXPECTED_HOSTNAME"]
parsed = urllib.parse.urlsplit(base_url)
if parsed.scheme != "https" or parsed.hostname != expected_hostname:
    raise RuntimeError("control-plane URL does not use the expected HTTPS hostname")
addresses = sorted(
    {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
)
if not addresses:
    raise RuntimeError("control-plane DNS returned no addresses")
context = ssl.create_default_context(cafile="/tls/ca.crt")
with urllib.request.urlopen(
    base_url + "/healthz", context=context, timeout=15
) as response:
    status = response.status
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
    with urllib.request.urlopen(probe, context=context, timeout=15) as response:
        authenticated = response.status
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
