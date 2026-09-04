"""Prove an Executor Pod can complete a TLS handshake to the control plane.

Request: two environment variables the Executor Pod already carries --
``GPU_FAULT_CONTROL_PLANE_CA_FILE`` and ``GPU_FAULT_CONTROL_PLANE_URL``.
Response: the ``/healthz`` body on stdout.

This runs inside the *Executor* Pod, not the control plane, and that is the
whole point: the administrator's own host can reach the NLB through a
kubeconfig, a proxy and its own trust store, none of which the Executor has. The
only trust store that answers the question is the CA file mounted into the Pod,
so the check has to be made from there with ``ssl.create_default_context``
rather than by disabling verification anywhere.
"""

import json
import os
import ssl
import urllib.request

context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
with urllib.request.urlopen(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/healthz",
    context=context,
    timeout=15,
) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
