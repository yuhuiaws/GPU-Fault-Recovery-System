from __future__ import annotations

import asyncio

import httpx

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.cluster_executor import (
    RegionalExecutorClient,
    RegionalFleetRegistry,
    RegionalIncidentOwnershipProvider,
)
from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.hyperpod_spares import NotificationSink
from gpu_fault.regional import RemoteEvidenceCaptureRequest
from gpu_fault.telemetry import EvidenceKind
from tests._builders import asgi_client, build_context, copy_model
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    RecordingNotifier,
    ownership_fixture,
    registration,
    spare_alert,
)


def test_remote_hyperpod_submission_endpoints_enforce_cluster_binding() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    headers_a = {
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }
    record = HyperPodSubmissionRecord(
        cluster_name="hp-cluster-a",
        idempotency_key="workflow-1/restart-node/0",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            reserve = await client.post(
                "/v1/regional/executors/hyperpod-submissions/reserve",
                headers=headers_a,
                json={
                    "cluster_id": "cluster-a",
                    "record": record.model_dump(mode="json"),
                },
            )
            assert reserve.status_code == 200
            assert reserve.json()["reserved"] is True

            replay = await client.post(
                "/v1/regional/executors/hyperpod-submissions/reserve",
                headers=headers_a,
                json={
                    "cluster_id": "cluster-a",
                    "record": record.model_dump(mode="json"),
                },
            )
            assert replay.status_code == 200
            assert replay.json()["reserved"] is False

            # cluster-b holds a valid token, but its registration maps
            # to hp-cluster-b, so it cannot touch another cluster's
            # HyperPod submission record.
            cross_cluster = await client.post(
                "/v1/regional/executors/hyperpod-submissions/reserve",
                headers={
                    "Authorization": f"Bearer {TOKEN_B}",
                    "X-GPU-Fault-Cluster-ID": "cluster-b",
                },
                json={
                    "cluster_id": "cluster-b",
                    "record": record.model_dump(mode="json"),
                },
            )
            assert cross_cluster.status_code == 403

            read = await client.get(
                "/v1/regional/executors/hyperpod-submissions",
                headers=headers_a,
                params={
                    "cluster_name": "hp-cluster-a",
                    "idempotency_key": ("workflow-1/restart-node/0"),
                },
            )
            assert read.status_code == 200
            assert read.json()["state"] == "INTENDED"

            missing = await client.get(
                "/v1/regional/executors/hyperpod-submissions",
                headers=headers_a,
                params={
                    "cluster_name": "hp-cluster-a",
                    "idempotency_key": "never-reserved",
                },
            )
            assert missing.status_code == 200
            assert missing.json() is None

            outcome = await client.post(
                "/v1/regional/executors/hyperpod-submissions/outcome",
                headers=headers_a,
                json={
                    "cluster_id": "cluster-a",
                    "record": copy_model(record, state="SUBMITTED").model_dump(
                        mode="json"
                    ),
                },
            )
            assert outcome.status_code == 200
            assert outcome.json()["record"]["state"] == "SUBMITTED"

            # An outcome for a key nobody reserved would let an executor
            # invent a completed submission it never made.
            unreserved = await client.post(
                "/v1/regional/executors/hyperpod-submissions/outcome",
                headers=headers_a,
                json={
                    "cluster_id": "cluster-a",
                    "record": copy_model(
                        record, idempotency_key="never-reserved", state="SUBMITTED"
                    ).model_dump(mode="json"),
                },
            )
            assert unreserved.status_code == 409

            conflicting = await client.post(
                "/v1/regional/executors/hyperpod-submissions/outcome",
                headers=headers_a,
                json={
                    "cluster_id": "cluster-a",
                    "record": copy_model(
                        record,
                        requested_node_identifiers=["some-other-node"],
                        state="SUBMITTED",
                    ).model_dump(mode="json"),
                },
            )
            assert conflicting.status_code == 409

    asyncio.run(scenario())


def test_incident_ownership_route_answers_takeover_question() -> None:
    """The regional executor has no store, so this is its only answer.

    Without it KubernetesWorkflowAdapter._can_take_over_node_isolation
    always returned False in a regional deployment and a node annotated
    by a dead workflow could never be isolated again.
    """

    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    ownership_fixture(context.store)
    headers_a = {
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }
    path = "/v1/regional/executors/incident-ownership"

    async def scenario() -> None:
        async with asgi_client(context) as client:
            dead = await client.get(
                path, headers=headers_a, params={"incident_id": "inc-dead"}
            )
            assert dead.status_code == 200
            assert dead.json() == {
                "incident_id": "inc-dead",
                "known": True,
                "workflow_request_id": "workflow-dead",
                "workflow_status": "FAILED",
                "terminal": True,
                "incident_state": "ACTION_PENDING",
                "quarantine_hold": False,
            }

            live = await client.get(
                path, headers=headers_a, params={"incident_id": "inc-live"}
            )
            assert live.status_code == 200
            assert live.json()["terminal"] is False
            assert live.json()["workflow_status"] == "RUNNING"

            quarantined = await client.get(
                path, headers=headers_a, params={"incident_id": "inc-quarantined"}
            )
            assert quarantined.status_code == 200
            assert quarantined.json()["terminal"] is True
            assert quarantined.json()["incident_state"] == "QUARANTINED"
            assert quarantined.json()["quarantine_hold"] is True

            # Fails closed rather than 404: an unknown incident must
            # read as "not finished", never as a licence to take over.
            unknown = await client.get(
                path, headers=headers_a, params={"incident_id": "inc-never-existed"}
            )
            assert unknown.status_code == 200
            assert unknown.json()["known"] is False
            assert unknown.json()["terminal"] is False

            cross_cluster = await client.get(
                path, headers=headers_a, params={"incident_id": "inc-other-cluster"}
            )
            assert cross_cluster.status_code == 403

            anonymous = await client.get(path, params={"incident_id": "inc-dead"})
            assert anonymous.status_code in {401, 403}

    asyncio.run(scenario())


def test_regional_ownership_provider_reads_the_route() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    ownership_fixture(context.store)
    app = create_app(context)

    class AsgiClient(RegionalExecutorClient):
        """Routes the real client's _get through the ASGI app."""

        def _get(self, path: str) -> object:
            async def call() -> object:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url=self.base_url
                ) as client:
                    response = await client.get(
                        path,
                        headers={
                            "Authorization": f"Bearer {self.token}",
                            "X-GPU-Fault-Cluster-ID": self.cluster_id,
                        },
                    )
                    assert response.status_code == 200, response.text
                    return response.json()

            return asyncio.run(call())

    provider = RegionalIncidentOwnershipProvider(
        AsgiClient("http://test", "cluster-a", TOKEN_A)
    )

    assert provider.incident_workflow_is_terminal("inc-dead") is True
    assert provider.incident_workflow_is_terminal("inc-live") is False
    quarantined = provider.incident_ownership("inc-quarantined")
    assert quarantined.terminal is True
    assert quarantined.quarantine_hold is True
    assert provider.incident_workflow_is_terminal("inc-never-existed") is False


def test_hyperpod_submission_paths_require_a_cluster_token() -> None:
    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for path in (
                "/v1/regional/executors/hyperpod-submissions/reserve",
                "/v1/regional/executors/hyperpod-submissions/outcome",
            ):
                anonymous = await client.post(path, json={"cluster_id": "cluster-a"})
                assert anonymous.status_code in {401, 403}
            unauthenticated_read = await client.get(
                "/v1/regional/executors/hyperpod-submissions",
                params={"cluster_name": "hp-cluster-a", "idempotency_key": "k"},
            )
            assert unauthenticated_read.status_code in {401, 403}

    asyncio.run(scenario())


def test_regional_executor_can_page_operator_on_spare_shortage() -> None:
    """Warm spare is the only replacement path, so a shortage must page.

    The executor owns no store in the regional deployment, so the alert
    can only reach an operator through this route. Before it existed the
    coordinator's notification_sink probe failed and the alert was only
    logged inside the data-plane Pod.
    """

    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    alert = spare_alert("cluster-a")

    async def scenario() -> None:
        async with asgi_client(context) as client:
            headers = {
                "Authorization": f"Bearer {TOKEN_A}",
                "X-GPU-Fault-Cluster-ID": "cluster-a",
            }
            payload = {
                "cluster_id": "cluster-a",
                "notification": alert.model_dump(mode="json"),
            }
            first = await client.post(
                "/v1/regional/executors/advisory-notifications",
                headers=headers,
                json=payload,
            )
            assert first.status_code == 200
            saved_id = first.json()["notification_id"]
            assert saved_id == alert.notification_id

            # Retried step -- must dedupe on the key, not page twice.
            second = await client.post(
                "/v1/regional/executors/advisory-notifications",
                headers=headers,
                json={
                    "cluster_id": "cluster-a",
                    "notification": spare_alert("cluster-a").model_dump(mode="json"),
                },
            )
            assert second.status_code == 200
            assert second.json()["notification_id"] == saved_id

    asyncio.run(scenario())

    stored = context.store.get_notification(alert.notification_id)
    assert "hyperpod-spare-insufficient" in stored.deduplication_key
    assert len(notifier.notifications) == 1


def test_regional_advisory_notification_rejects_other_clusters() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    context.regional_mode = True
    for cluster_id, token in (("cluster-a", TOKEN_A), ("cluster-b", TOKEN_B)):
        context.store.save_regional_cluster(registration(cluster_id, token))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            headers = {
                "Authorization": f"Bearer {TOKEN_A}",
                "X-GPU-Fault-Cluster-ID": "cluster-a",
            }
            spoofed_envelope = await client.post(
                "/v1/regional/executors/advisory-notifications",
                headers=headers,
                json={
                    "cluster_id": "cluster-b",
                    "notification": spare_alert("cluster-b").model_dump(mode="json"),
                },
            )
            assert spoofed_envelope.status_code == 403

            # Envelope matches but the payload names another cluster --
            # would otherwise let one tenant write alerts onto another.
            spoofed_payload = await client.post(
                "/v1/regional/executors/advisory-notifications",
                headers=headers,
                json={
                    "cluster_id": "cluster-a",
                    "notification": spare_alert("cluster-b").model_dump(mode="json"),
                },
            )
            assert spoofed_payload.status_code == 403

            anonymous = await client.post(
                "/v1/regional/executors/advisory-notifications",
                json={
                    "cluster_id": "cluster-a",
                    "notification": spare_alert("cluster-a").model_dump(mode="json"),
                },
            )
            assert anonymous.status_code in {401, 403}

    asyncio.run(scenario())
    assert notifier.notifications == []


def test_regional_executor_can_persist_workload_log_evidence() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/regional/executors/evidence",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={
                    "cluster_id": "cluster-a",
                    "record_id": "workload-log/test",
                    "node_id": "node-a",
                    "kind": "WORKLOAD_LOG",
                    "observed_at": NOW.isoformat(),
                    "attempt_ids": ["attempt-a"],
                    "payload": {"tail": "training output"},
                },
            )
            assert response.status_code == 200

    asyncio.run(scenario())

    evidence = context.store.list_raw_evidence(
        "cluster-a", node_id="node-a", kind=EvidenceKind.WORKLOAD_LOG
    )
    assert len(evidence) == 1
    assert evidence[0].payload["tail"] == "training output"


def test_regional_registry_persists_evidence_over_api() -> None:
    class Client:
        cluster_id = "cluster-a"

        def __init__(self) -> None:
            self.requests = []

        def _post(self, path, payload):
            self.requests.append((path, payload))
            return {
                **payload,
                "ingested_at": NOW.isoformat(),
                "expires_at": (NOW.replace(day=29).isoformat()),
            }

    client = Client()
    registry = RegionalFleetRegistry(client)
    request = RemoteEvidenceCaptureRequest(
        cluster_id="cluster-a",
        record_id="workload-log/registry",
        node_id="node-a",
        kind=EvidenceKind.WORKLOAD_LOG,
        observed_at=NOW,
        attempt_ids=["attempt-a"],
        payload={"tail": "line"},
    )

    record = registry.capture_evidence(request)

    assert record.record_id == request.record_id
    assert client.requests[0][0] == ("/v1/regional/executors/evidence")


def test_regional_registry_persists_notification_over_api() -> None:
    class Client:
        cluster_id = "cluster-a"

        def __init__(self) -> None:
            self.requests = []

        def _post(self, path, payload):
            self.requests.append((path, payload))
            return payload["notification"]

    client = Client()
    registry = RegionalFleetRegistry(client)
    notification = spare_alert("cluster-a")

    saved = registry.save_notification_if_absent(notification)

    assert saved == notification
    assert client.requests == [
        (
            "/v1/regional/executors/advisory-notifications",
            {
                "cluster_id": "cluster-a",
                "notification": notification.model_dump(mode="json"),
            },
        )
    ]


def test_regional_registry_satisfies_the_notification_sink_protocol() -> None:
    """The coordinator probes with isinstance, so the shape must match.

    hyperpod_spares.HyperPodSpareCoordinator sets notification_sink to
    None when its store fails this check, and then downgrades every
    capacity alert to a log line. The probe is structural, so a method
    rename on either side silently reintroduces the defect.
    """

    class _Client:
        cluster_id = "cluster-a"

    registry = RegionalFleetRegistry(_Client())
    assert isinstance(registry, NotificationSink)
    assert registry.store is registry


def test_node_installer_can_read_back_its_own_gpu_metrics() -> None:
    """The install self-check reads this path with a per-cluster token.

    verify-gpu-fault-collector.sh finishes by reading the node's own
    latest sample back from the control plane, and a node only ever
    holds GPU_FAULT_CONTROL_PLANE_TOKEN. When this path fell into the
    regional default-deny bucket the read returned 403, so every
    installer Job failed its last check, node_installer_reconciler
    marked the node Failed, and it re-created the Job every
    GPU_FAULT_INSTALL_RETRY_SECONDS -- reinstalling forever on a GPU
    node that was in fact healthy.
    """

    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    path = "/v1/gpu-metrics/cluster-a/node-1/latest"

    async def scenario() -> None:
        async with asgi_client(context) as client:
            node = await client.get(
                path,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert node.status_code == 200, node.text

            operator = await client.get(
                path, headers={"X-GPU-Fault-Execution-Token": "operator-token"}
            )
            assert operator.status_code == 200, operator.text

            anonymous = await client.get(path)
            assert anonymous.status_code == 403

            wrong_token = await client.get(
                path,
                headers={
                    "Authorization": f"Bearer {TOKEN_B}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert wrong_token.status_code == 403

            other_cluster = await client.get(
                "/v1/gpu-metrics/cluster-b/node-1/latest",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert other_cluster.status_code == 403
            assert "another cluster" in other_cluster.text

    asyncio.run(scenario())


def fleet_rollout_fence_deployment(deployment_id: str, cluster_id: str):
    """A rollout that has not started a wave, which is what the fence holds on."""

    from gpu_fault.fleet_deployment import (
        DeploymentNode,
        DeploymentNodeStatus,
        DeploymentStatus,
        FleetDeployment,
    )

    node = DeploymentNode(
        node_id="node-a", status=DeploymentNodeStatus.PENDING, updated_at=NOW
    )
    return FleetDeployment(
        deployment_id=deployment_id,
        cluster_id=cluster_id,
        desired_agent_version="0.10.0",
        desired_artifact_sha256="a" * 64,
        desired_policy_version="catalog-a",
        desired_runtime_profile_version="profile-a",
        desired_config_digest="c" * 64,
        max_unavailable=1,
        waves=[["node-a"]],
        nodes=[node],
        status=DeploymentStatus.PLANNED,
        created_at=NOW,
        updated_at=NOW,
    )


def test_the_data_plane_can_read_its_own_fleet_rollout_fence() -> None:
    """The executor owns no store, and it is the last gate before a mutation.

    The control-plane fence runs before dispatch and cannot revoke a remote
    command that already exists, so if the executor cannot ask this question
    nothing checks it again for the life of the command. On 2026-09-04 the
    executor asked its own storeless proxy instead, got an AttributeError, and
    -- the fence failing closed -- held every destructive command forever.
    """

    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    context.store.save_fleet_deployment(
        fleet_rollout_fence_deployment("rollout-b-1", "cluster-b")
    )
    app = create_app(context)

    class AsgiClient(RegionalExecutorClient):
        def _get(self, path: str) -> object:
            async def call() -> object:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url=self.base_url
                ) as client:
                    response = await client.get(
                        path,
                        headers={
                            "Authorization": f"Bearer {self.token}",
                            "X-GPU-Fault-Cluster-ID": self.cluster_id,
                        },
                    )
                    assert response.status_code == 200, response.text
                    return response.json()

            return asyncio.run(call())

    registry_a = RegionalFleetRegistry(AsgiClient("http://test", "cluster-a", TOKEN_A))
    registry_b = RegionalFleetRegistry(AsgiClient("http://test", "cluster-b", TOKEN_B))

    # cluster-b's rollout is invisible to cluster-a and fences only cluster-b,
    # so one cluster's upgrade cannot stall another cluster's remediation.
    assert registry_a.fleet_rollout_fence_deployments("cluster-a") == []
    assert registry_b.fleet_rollout_fence_deployments("cluster-b") == ["rollout-b-1"]

    context.store.save_fleet_deployment(
        fleet_rollout_fence_deployment("rollout-a-1", "cluster-a")
    )
    assert registry_a.fleet_rollout_fence_deployments("cluster-a") == ["rollout-a-1"]


def test_the_fleet_rollout_fence_route_is_cluster_scoped() -> None:
    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_fleet_deployment(
        fleet_rollout_fence_deployment("rollout-a-1", "cluster-a")
    )
    path = "/v1/regional/executors/fleet-rollout-fence"

    async def scenario() -> None:
        async with asgi_client(context) as client:
            own = await client.get(
                path,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert own.status_code == 200
            assert own.json() == {
                "cluster_id": "cluster-a",
                "fencing_deployment_ids": ["rollout-a-1"],
            }

            # The cluster is the authenticated identity, never an argument, so
            # there is no query parameter to point at another cluster.
            spoofed = await client.get(
                path,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                params={"cluster_id": "cluster-b"},
            )
            assert spoofed.status_code == 200
            assert spoofed.json()["cluster_id"] == "cluster-a"

            anonymous = await client.get(path)
            assert anonymous.status_code in {401, 403}

            wrong_token = await client.get(
                path,
                headers={
                    "Authorization": "Bearer not-the-cluster-token",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert wrong_token.status_code in {401, 403}

    asyncio.run(scenario())
