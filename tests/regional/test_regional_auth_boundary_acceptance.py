"""Local proxies for the manual regional authentication acceptance cases.

``GF-REGIONAL-AUTH-*`` and ``GF-REGIONAL-ISO-003`` are classified
``read-only-signal-replay``: they prove denial semantics with curl from a GPU
data-plane pod and change nothing. That makes them the cheapest cases to run
live and, precisely because of that, the ones most likely to be run once and
then trusted forever -- while the boundary they check is the one a route
addition or a middleware reorder breaks silently.

Every test here answers one catalog case against the real application, the real
authorization registry and a real store, so the boundary regresses in CI rather
than in the next acceptance window. It is a proxy, not a replacement: the live
case additionally proves the network path, the private CA and the NLB, none of
which exist in this process. ``tools/regional_acceptance_plan.py`` reads these
node ids from ``testcases/fault-scenarios.yaml`` and runs them as the local
pre-acceptance stage.

The mechanical sweeps (AUTH-009/010/014) enumerate routes from the application
itself rather than from a list kept here, because a hand-written route list
answers "the routes we thought about" and the case asks about every route the
NLB actually serves.
"""

from __future__ import annotations

import asyncio
import re
import ssl
from pathlib import Path

import httpx
import pytest
import yaml

from gpu_fault.app import create_app
from gpu_fault.app.authorization import (
    AUTHORIZATION_BUCKETS,
    UNDOCUMENTED_PUBLIC_PATHS,
    ExplicitAuthorizationRegistry,
    iter_api_routes,
)
from gpu_fault.cluster_executor import (
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
)
from gpu_fault.fleet import (
    AgentHeartbeat,
    AgentTransitionRequest,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import WorkflowOperation
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    registration,
    terminal,
)

ROOT = Path(__file__).resolve().parents[2]


CLAIM = "/v1/regional/executors/claim"
COLLECTOR_HEALTH = "/v1/collector-events/collector-health"
EXECUTION_TOKEN = "e" * 32
CLUSTER_HEADERS = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
PATH_PARAMETER = re.compile(r"\{[^}]+\}")
# The middlewares deny before routing, so any syntactically valid segment
# reaches the same check; ``cluster-a`` is registered, which keeps a denial from
# being an accident of an unknown cluster id.
PLACEHOLDER = "cluster-a"
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DENIAL_BY_BUCKET = {
    "public": 200,
    "cluster-token": 401,
    "dual-credential": 403,
    "execution-token": 403,
    # Only loopback reads /metrics unauthenticated; a data-plane peer is handed
    # to the execution-token check, which it cannot pass.
    "metrics": 403,
}


def regional_context():
    context = build_context()
    context.regional_mode = True
    context.execution_token = EXECUTION_TOKEN
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


def route_matrix(app) -> list[tuple[str, str, str]]:
    """Every declared ``(path, method, bucket)`` the application serves."""

    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)
    matrix = []
    for route in iter_api_routes(app.routes):
        bucket = registry.inventory.get(route.path)
        if bucket is None:
            continue
        for method in sorted((route.methods or set()) - {"HEAD", "OPTIONS"}):
            matrix.append((route.path, method, bucket))
    return matrix


def data_plane_client(app) -> httpx.AsyncClient:
    """A client whose socket peer is a GPU data-plane pod, not loopback."""

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("10.0.1.5", 44444)),
        base_url="http://test",
    )


def test_auth001_a_protected_route_without_the_cluster_header_answers_401() -> None:
    context = regional_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                CLAIM,
                headers={"Authorization": f"Bearer {TOKEN_A}"},
                json={"executor_id": "executor-a"},
            )

            assert response.status_code == 401, response.text
            assert response.json() == {"detail": "X-GPU-Fault-Cluster-ID is required"}

    asyncio.run(scenario())


def test_auth002_absent_malformed_and_empty_bearer_stay_distinguishable() -> None:
    """401 means "you sent no credential", 403 means "it was wrong".

    An operator debugging a data-plane pod reads the status code first: 401
    sends them to the Secret mount, 403 to the registry revision. Collapsing
    the two costs an acceptance window of guessing.
    """

    context = regional_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            statuses = {}
            for label, authorization in (
                ("absent", None),
                ("basic", "Basic YWRtaW46YWRtaW4="),
                ("empty-bearer", "Bearer "),
            ):
                headers = {"X-GPU-Fault-Cluster-ID": "cluster-a"}
                if authorization is not None:
                    headers["Authorization"] = authorization
                response = await client.post(
                    CLAIM, headers=headers, json={"executor_id": "executor-a"}
                )
                statuses[label] = (response.status_code, response.json())

            assert statuses["absent"] == (
                401,
                {"detail": "regional cluster bearer token is required"},
            )
            assert statuses["basic"] == statuses["absent"], (
                "a non-Bearer scheme has to fail like a missing credential"
            )
            assert statuses["empty-bearer"] == (
                403,
                {"detail": "regional cluster authentication failed"},
            )

    asyncio.run(scenario())


def test_auth004_a_near_miss_token_is_denied_exactly_like_a_wrong_one() -> None:
    """The denial must not narrow the search space for the real token.

    A response that differed between a token sharing 31 of 32 characters and one
    sharing none would turn the endpoint into an oracle, which is also why the
    digest comparison is ``secrets.compare_digest``.
    """

    context = regional_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            answers = []
            for token in ("0" * 32, TOKEN_A[:-1] + "z", TOKEN_A[:8]):
                response = await client.post(
                    CLAIM,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-GPU-Fault-Cluster-ID": "cluster-a",
                    },
                    json={"executor_id": "executor-a"},
                )
                answers.append((response.status_code, response.text))

            assert [status for status, _ in answers] == [403, 403, 403]
            assert len({body for _, body in answers}) == 1, (
                "the denial body distinguishes how close the token was"
            )
            detail = answers[0][1]
            assert "32" not in detail and "length" not in detail, detail

    asyncio.run(scenario())


def test_auth005_a_payload_for_another_cluster_is_403_and_writes_nothing() -> None:
    context = regional_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            spoofed = await client.post(
                "/v1/attempts/terminal",
                headers=CLUSTER_HEADERS,
                json=terminal("cluster-b").model_dump(mode="json"),
            )
            collector = await client.post(
                COLLECTOR_HEALTH,
                headers=CLUSTER_HEADERS,
                json={"cluster_id": "cluster-b", "node_id": "node-b"},
            )

            mismatch = {
                "detail": (
                    "authenticated cluster does not match all payload cluster_id values"
                )
            }
            assert (spoofed.status_code, spoofed.json()) == (403, mismatch)
            assert (collector.status_code, collector.json()) == (403, mismatch)
            assert context.store.list_collector_statuses("cluster-b") == []

    asyncio.run(scenario())


def test_auth006_a_nested_heartbeat_cluster_id_is_checked_before_the_signature() -> (
    None
):
    """The agent heartbeat carries its cluster id one level down.

    Cross-checking only the top level would let cluster A register agents into
    cluster B's fleet. The 403 also has to arrive before signature verification,
    or the boundary would depend on a shared secret the caller may well hold.
    """

    context = regional_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/fleet/agents/heartbeat",
                headers=CLUSTER_HEADERS,
                json={
                    "heartbeat": {"cluster_id": "cluster-b", "node_id": "node-b"},
                    "signature": "not-a-valid-signature",
                },
            )

            assert response.status_code == 403, response.text
            assert response.json() == {
                "detail": (
                    "authenticated cluster does not match all payload cluster_id values"
                )
            }
            assert context.store.list_agents("cluster-b") == []

    asyncio.run(scenario())


def test_auth009_every_cluster_token_write_route_rejects_a_foreign_payload() -> None:
    """The cross-check is a middleware, so it must hold for every write route.

    The case names eleven ingest and status endpoints. Enumerating them from the
    application instead means a twelfth one is covered the day it is added.
    """

    context = regional_context()
    app = create_app(context)
    writes = [
        (path, method)
        for path, method, bucket in route_matrix(app)
        if bucket == "cluster-token" and method in {"POST", "PUT", "PATCH"}
    ]

    async def scenario() -> None:
        async with asgi_client(app) as client:
            answers = {}
            for path, method in writes:
                response = await client.request(
                    method,
                    PATH_PARAMETER.sub(PLACEHOLDER, path),
                    headers=CLUSTER_HEADERS,
                    json={"cluster_id": "cluster-b"},
                )
                answers[f"{method} {path}"] = (
                    response.status_code,
                    response.json().get("detail"),
                )

            assert len(answers) >= 11, answers
            mismatch = (
                403,
                "authenticated cluster does not match all payload cluster_id values",
            )
            assert {
                key: value for key, value in answers.items() if value != mismatch
            } == {}
            assert context.store.list_collector_statuses("cluster-b") == []
            assert context.store.list_raw_evidence("cluster-b") == []

    asyncio.run(scenario())


def test_auth010_every_route_answers_the_denial_its_bucket_declares() -> None:
    """A credential-less request has exactly one allowed answer per bucket.

    This is the matrix the case asks a human to walk with curl. Deriving it from
    the route table makes the answer complete by construction, and pins that no
    write route sits in ``public`` or ``metrics``, where the NLB would serve it
    to anything that can reach the VPC.
    """

    context = regional_context()
    app = create_app(context)
    matrix = route_matrix(app)

    async def scenario() -> None:
        async with data_plane_client(app) as client:
            unexpected = {}
            for path, method, bucket in matrix:
                response = await client.request(
                    method,
                    PATH_PARAMETER.sub(PLACEHOLDER, path),
                    json={} if method in {"POST", "PUT", "PATCH"} else None,
                )
                if response.status_code != DENIAL_BY_BUCKET[bucket]:
                    unexpected[f"{method} {path}"] = (bucket, response.status_code)

            assert len(matrix) >= 80, "the route sweep collapsed to a handful of routes"
            assert unexpected == {}

    asyncio.run(scenario())

    exposed = [
        (path, method, bucket)
        for path, method, bucket in matrix
        if method in WRITE_METHODS and bucket in {"public", "metrics"}
    ]

    assert exposed == []


def test_auth011_health_and_metrics_do_not_expose_the_regional_inventory() -> None:
    """``/metrics`` carries a ``cluster_id`` label for every registered cluster.

    Readable from a GPU data-plane pod, it is a region-wide cluster inventory
    handed to the least trusted network in the deployment; ``/healthz`` has to
    stay open for the load balancer, so the two cannot share a bucket.
    """

    context = regional_context()
    app = create_app(context)

    async def scenario() -> None:
        async with data_plane_client(app) as client:
            health = await client.get("/healthz")
            metrics = await client.get("/metrics")
            clusters = await client.get("/v1/regional/clusters")
            with_cluster_token = await client.get(
                "/v1/regional/clusters", headers=CLUSTER_HEADERS
            )

            assert health.status_code == 200, health.text
            assert metrics.status_code == 403, metrics.text
            assert clusters.status_code in {401, 403}, clusters.text
            assert with_cluster_token.status_code in {401, 403}, with_cluster_token.text
            leaked = [
                response.text
                for response in (health, metrics, clusters, with_cluster_token)
                if "cluster-b" in response.text
            ]
            assert leaked == []

    asyncio.run(scenario())


def fleet_agent(context, cluster_id: str, node_id: str) -> None:
    """Register one live agent so cross-cluster reads have something to leak."""

    secret = "fleet-secret-" + "x" * 32
    if context.fleet_registry is None:
        context.fleet_registry = FleetRegistry(context.store, secret, now=lambda: NOW)
    heartbeat = AgentHeartbeat(
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint=f"http://{node_id}:9099",
        agent_protocol_version=3,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="policy-v1",
        runtime_profile_version="profile-v1",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        observed_at=NOW,
    )
    context.fleet_registry.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, secret)
        )
    )


def any_ca_bundle() -> str:
    """A real PEM path, since the assertion is about the context, not the trust.

    ``ssl.create_default_context`` refuses to load a bundle that does not parse,
    so this cannot be a fabricated path.
    """

    bundle = ssl.get_default_verify_paths().cafile
    if not bundle or not Path(bundle).is_file():
        pytest.skip("no CA bundle on this host to build an SSL context from")
    return bundle


def test_auth013_the_executor_trusts_the_private_ca_without_replacing_aws_trust() -> (
    None
):
    """The private CA is added for one hostname, not swapped in globally.

    Setting ``SSL_CERT_FILE`` or ``REQUESTS_CA_BUNDLE`` to the private bundle
    would make the pod trust that CA for every TLS destination and stop trusting
    the public roots the AWS SDK needs -- so an operator "fixing" a handshake
    that way silently turns the STS and EKS calls into unverified ones. The live
    case additionally proves an empty bundle and a wrong servername both fail;
    only the manifest half can be proven here.
    """

    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item.get("value") for item in container["env"]}
    sourced = {
        item["name"] for item in container["env"] if item.get("valueFrom") is not None
    }
    ca_file = env["GPU_FAULT_CONTROL_PLANE_CA_FILE"]

    assert ca_file == "/etc/gpu-fault/tls/ca.crt"
    assert "SSL_CERT_FILE" not in env
    assert "REQUESTS_CA_BUNDLE" not in env
    assert "GPU_FAULT_CONTROL_PLANE_INSECURE" not in env
    # The endpoint is a Secret value, so the manifest cannot show the scheme;
    # what it can show is that no plaintext URL is baked into the pod spec.
    assert "GPU_FAULT_CONTROL_PLANE_URL" in sourced
    assert env["GPU_FAULT_CONTROL_PLANE_URL"] is None

    client = RegionalExecutorClient(
        "https://control-plane.example", "cluster-a", TOKEN_A, ca_file=any_ca_bundle()
    )

    # Hostname verification is what makes the private CA an identity check
    # rather than a decoration: without it any certificate that CA ever signed
    # would be accepted for the control-plane name.
    assert client.ssl_context.check_hostname is True
    assert client.ssl_context.verify_mode == ssl.CERT_REQUIRED


def test_auth014_the_internet_facing_surface_is_exactly_its_declared_buckets() -> None:
    """Every route on an internet-facing NLB, enumerated rather than sampled.

    The EIP allowlist is a second layer: it stops packets from outside the VPC,
    which this process cannot test. What it cannot do is stop a GPU data-plane
    pod that legitimately reaches the NLB, so each route's own bucket has to be
    the real boundary -- including the ones that only look like reads, such as
    the fleet agent listing, whose cluster-token form must stay pinned to the
    authenticated cluster instead of honouring a ``cluster_id`` query.
    """

    context = regional_context()
    fleet_agent(context, "cluster-b", "node-b")
    app = create_app(context)
    matrix = route_matrix(app)

    async def scenario() -> None:
        async with data_plane_client(app) as client:
            other_cluster = await client.get(
                "/v1/fleet/agents?cluster_id=cluster-b", headers=CLUSTER_HEADERS
            )
            own_scope = await client.get("/v1/fleet/agents", headers=CLUSTER_HEADERS)
            other_agent = await client.get(
                "/v1/fleet/agents/cluster-b/node-b", headers=CLUSTER_HEADERS
            )
            operator_only = {}
            for path in ("/v1/runtime-profiles", "/v1/advisory-notifications/n-1/send"):
                response = await client.post(path, headers=CLUSTER_HEADERS, json={})
                operator_only[path] = response.status_code
            public_reads = {}
            for path in sorted(UNDOCUMENTED_PUBLIC_PATHS):
                response = await client.get(path)
                public_reads[path] = response.status_code

            assert other_cluster.status_code == 403, other_cluster.text
            assert other_cluster.json() == {
                "detail": "authenticated cluster cannot read agents for another cluster"
            }
            assert own_scope.json() == [], (
                "an unqualified cluster-token listing escaped its own cluster"
            )
            assert other_agent.status_code == 403, other_agent.text
            # A cluster token is not an operator credential, however much of the
            # data plane it can otherwise write to.
            assert operator_only == {
                "/v1/runtime-profiles": 403,
                "/v1/advisory-notifications/n-1/send": 403,
            }
            # The OpenAPI surface is public by decision, and the decision is
            # recorded here: it is read-only and describes routes without
            # naming a cluster.
            assert set(public_reads.values()) == {200}, public_reads

    asyncio.run(scenario())

    undeclared = [
        (path, method, bucket)
        for path, method, bucket in matrix
        if bucket not in AUTHORIZATION_BUCKETS
    ]

    assert undeclared == []
    assert len(matrix) >= 80, "the route sweep collapsed to a handful of routes"


def test_iso003_the_fleet_proxy_refuses_cross_cluster_reads_and_transitions() -> None:
    """The data plane denies locally, before the request leaves the pod.

    The control plane would deny it too, but a proxy that sent the request at
    all would make every executor bug a cross-cluster attempt at the boundary,
    and the case is what proves the executor cannot even ask.
    """

    class RecordingClient:
        cluster_id = "cluster-a"

        def __init__(self) -> None:
            self.requests: list[tuple[str, object]] = []

        def _get(self, path):
            self.requests.append((path, None))
            return []

        def _post(self, path, payload):
            self.requests.append((path, payload))
            return {}

    client = RecordingClient()
    registry = RegionalFleetRegistry(client)

    with pytest.raises(
        ClusterExecutorError, match="cannot read agents for another cluster"
    ):
        registry.list_agents("cluster-b")

    with pytest.raises(
        ClusterExecutorError, match="cannot transition an agent in another cluster"
    ):
        registry.revoke_agent(
            "cluster-b",
            "node-b",
            AgentTransitionRequest(
                expected_generation=1,
                transition_id="iso003",
                reason="cross-cluster revoke attempt",
            ),
        )

    assert client.requests == []
