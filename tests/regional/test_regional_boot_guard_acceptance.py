"""Local proxies for the manual regional startup-guard acceptance cases.

``GF-REGIONAL-BOOT-*`` all ask the same question in different words: when the
declared configuration is wrong, does the process refuse to serve, or does it
come up and quietly behave like something else? Every one of these guards is a
check in ``ControlPlaneSettings.from_mapping`` or
``ApplicationContext.from_environment`` -- so the failure they protect against
(a control plane that boots with no registry, with the central HyperPod adapter
enabled, or with a 31-character cluster token) is reproducible in this process
with no cluster at all.

The live case stays: only a real rollout proves the Pod never reaches Ready,
that the message lands in the log stream an operator actually reads, and that
CloudTrail recorded no mutation. What CI can own is the guard itself, because a
guard that stops firing looks exactly like a guard that was never needed.

The environment is always built from the shared regional fixture and then broken
in exactly one axis, so each test names one variable as the cause.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.app.context import ApplicationContext
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications.ses import SesEmailNotifier, SesNotificationConfig
from gpu_fault.regional import RemoteCommandStatus
from tests._builders import asgi_client, build_context, build_store
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    enqueue_remote_command,
    regional_environment,
    registration,
)

ROOT = Path(__file__).resolve().parents[2]
REGIONAL_PATCH = (
    ROOT / "deploy/control-plane/regional/regional-control-plane-patch.yaml"
)
AMP_RULES = ROOT / "deploy/observability/amp-rules.yaml"
CLUSTER_A = {
    "cluster_id": "cluster-a",
    "region": "us-west-2",
    "hyperpod_cluster_name": "hp-cluster-a",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a",
    "token": TOKEN_A,
    "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
}


def set_registry(monkeypatch, value: str) -> None:
    monkeypatch.setenv("GPU_FAULT_REGIONAL_CLUSTERS_JSON", value)


def cluster_entry(**overrides) -> dict:
    entry = dict(CLUSTER_A)
    entry.update(overrides)
    return entry


def api_environment() -> dict[str, dict]:
    """The declared env of the regional API container, by variable name."""

    patch = yaml.safe_load(REGIONAL_PATCH.read_text(encoding="utf-8"))
    containers = patch["spec"]["template"]["spec"]["containers"]
    api = next(item for item in containers if item["name"] == "api")
    return {item["name"]: item for item in api["env"]}


def test_boot001_regional_mode_without_a_registry_refuses_to_start(
    monkeypatch, tmp_path
) -> None:
    """A regional control plane with no registry has no cluster to authenticate.

    It would come up healthy, answer ``/healthz`` with 200, and reject every
    executor as an unregistered cluster -- which reads as a data-plane problem.
    Recovery would stop for the whole region while the control plane reported
    itself fine, so the absence has to stop startup instead.
    """

    regional_environment(monkeypatch, tmp_path)
    monkeypatch.delenv("GPU_FAULT_REGIONAL_CLUSTERS_JSON", raising=False)

    with pytest.raises(
        RuntimeError, match="regional mode requires GPU_FAULT_REGIONAL_CLUSTERS_JSON"
    ):
        ApplicationContext.from_environment()


def test_boot002_a_malformed_registry_fails_and_an_explicit_empty_one_boots(
    monkeypatch, tmp_path
) -> None:
    """Each malformed shape gets its own message, and ``[]`` is not malformed.

    A single "invalid registry" error would send an operator to re-read the whole
    JSON blob; naming the shape points at the edit. And an explicitly empty list
    is a legitimate state -- a region whose first cluster has not been registered
    yet -- so it has to boot, or the greenfield rollout order becomes impossible.
    """

    regional_environment(monkeypatch, tmp_path)
    failures = {}
    for label, raw, expected in (
        ("truncated", '[{"cluster_id": "cluster-a"', "is invalid"),
        ("mapping", "{}", "registry must be a list"),
        ("scalar-entry", "[1]", "entries must be objects"),
    ):
        set_registry(monkeypatch, raw)
        with pytest.raises(RuntimeError, match=expected) as raised:
            ApplicationContext.from_environment()
        failures[label] = str(raised.value)

    set_registry(monkeypatch, "[]")
    context = ApplicationContext.from_environment()

    assert sorted(failures) == ["mapping", "scalar-entry", "truncated"]
    assert len(set(failures.values())) == 3, failures
    assert context.regional_mode is True
    assert context.store.list_regional_clusters() == []


def test_boot003_regional_mode_and_the_single_cluster_variable_are_exclusive(
    monkeypatch, tmp_path
) -> None:
    """Two sources of truth for "which HyperPod cluster" is one too many.

    ``GPU_FAULT_HYPERPOD_CLUSTER`` is the single-cluster form. Left set in
    regional mode it names one cluster while the registry names the others, and
    whichever code path reads it would act on the control plane's own account
    instead of the target cluster's executor.
    """

    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-control-plane-itself")

    with pytest.raises(RuntimeError, match="GPU_FAULT_HYPERPOD_CLUSTER must be unset"):
        ApplicationContext.from_environment()


def test_boot005_regional_mode_refuses_the_central_hyperpod_adapter(
    monkeypatch, tmp_path
) -> None:
    """The central adapter would mutate clusters from the control plane's role.

    That is the whole property the regional split exists to hold: node actions
    are submitted by each cluster's own executor with that cluster's own
    credentials. An adapter switch left on would move them back to one central
    role, and no per-cluster credential could then limit the blast radius.
    """

    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "true")

    with pytest.raises(RuntimeError, match="delegate HyperPod"):
        ApplicationContext.from_environment()


def test_boot006_regional_mode_refuses_in_cluster_quick_diagnostics(
    monkeypatch, tmp_path
) -> None:
    """In-cluster diagnostics would run against the wrong EKS entirely.

    The adapter execs into DCGM pods in *its own* cluster. On a regional control
    plane that is the CPU cluster, which has no GPUs: the diagnosis would either
    fail or, worse, return a healthy verdict about nodes it never looked at.
    """

    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS", "true")

    with pytest.raises(RuntimeError, match="in-cluster quick diagnostics"):
        ApplicationContext.from_environment()


def test_boot007_a_cluster_token_shorter_than_32_characters_stops_startup(
    monkeypatch, tmp_path
) -> None:
    """The cluster token is the only thing standing between clusters.

    It is compared by digest, so a short token is not a formatting problem: it is
    a guessable credential that authenticates one cluster's executor to submit
    another cluster's node actions. 32 characters is the declared floor, and the
    boundary is what a rotation script gets wrong.
    """

    regional_environment(monkeypatch, tmp_path)
    set_registry(monkeypatch, json.dumps([cluster_entry(token="t" * 31)]))

    with pytest.raises(ValueError, match="at least 32 characters"):
        ApplicationContext.from_environment()

    set_registry(monkeypatch, json.dumps([cluster_entry(token="t" * 32)]))
    context = ApplicationContext.from_environment()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            health = await client.get("/healthz")
            assert health.status_code == 200, health.text

    asyncio.run(scenario())
    assert [item.cluster_id for item in context.store.list_regional_clusters()] == [
        "cluster-a"
    ]


def test_boot008_registry_field_integrity_and_disabled_cluster_semantics(
    monkeypatch, tmp_path
) -> None:
    """A registry entry is validated, and ``enabled: false`` means denied.

    A missing ``eks_cluster_arn`` and a misspelled key are the same accident --
    an operator hand-editing JSON -- but a lax model would treat the second as a
    field with a default and register a cluster nobody described. And a disabled
    cluster has to keep serving 403 to its own token: that is how a cluster is
    taken out of the fleet without deleting the audit trail of it.
    """

    regional_environment(monkeypatch, tmp_path)
    incomplete = cluster_entry()
    incomplete.pop("eks_cluster_arn")
    set_registry(monkeypatch, json.dumps([incomplete]))
    with pytest.raises(ValueError, match="eks_cluster_arn"):
        ApplicationContext.from_environment()

    set_registry(
        monkeypatch, json.dumps([cluster_entry(alowed_namespaces=["training"])])
    )
    with pytest.raises(ValueError, match="alowed_namespaces"):
        ApplicationContext.from_environment()

    set_registry(
        monkeypatch,
        json.dumps(
            [
                cluster_entry(),
                cluster_entry(
                    cluster_id="cluster-b",
                    hyperpod_cluster_name="hp-cluster-b",
                    eks_cluster_arn=(
                        "arn:aws:eks:us-west-2:123456789012:cluster/cluster-b"
                    ),
                    token=TOKEN_B,
                    enabled=False,
                ),
            ]
        ),
    )
    context = ApplicationContext.from_environment()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            answers = {}
            for cluster_id, token in (("cluster-a", TOKEN_A), ("cluster-b", TOKEN_B)):
                response = await client.post(
                    "/v1/regional/executors/claim",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-GPU-Fault-Cluster-ID": cluster_id,
                    },
                    json={"executor_id": "executor-a"},
                )
                answers[cluster_id] = response.status_code
            health = await client.get("/healthz")

            # The disabled cluster is still registered -- it is refused at
            # authentication, not erased from the registry.
            assert answers == {"cluster-a": 200, "cluster-b": 403}
            assert health.status_code == 200, health.text

    asyncio.run(scenario())
    assert {
        item.cluster_id: item.enabled for item in context.store.list_regional_clusters()
    } == {"cluster-a": True, "cluster-b": False}


def test_boot009_the_managed_observer_requires_the_agent_registry(
    monkeypatch, tmp_path
) -> None:
    """The observer reads agent state; without the registry there is none.

    Enabled alone it would watch for managed recovery that it cannot see, and
    report nothing rather than fail -- so a HyperPod-managed replacement running
    underneath this control plane would go unnoticed while both act on the same
    node.
    """

    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", "true")
    monkeypatch.setenv("GPU_FAULT_ENABLE_AGENT_REGISTRY", "false")

    with pytest.raises(
        RuntimeError, match="requires GPU_FAULT_ENABLE_AGENT_REGISTRY=true"
    ):
        ApplicationContext.from_environment()


def test_boot013_the_ses_client_region_comes_only_from_the_declared_environment(
    monkeypatch,
) -> None:
    """A missing region makes every notification fail while recovery looks fine.

    ``boto3`` does not infer a region from Pod Identity, so an API Pod without
    ``AWS_REGION`` raises ``NoRegionError`` on the first send -- after the
    recovery has already run. The variable therefore has to be declared in the
    manifest rather than set by hand, and the client has to be built from it.
    """

    monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", "gpu-fault@example.com")
    monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", "oncall@example.com")
    monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", "true")

    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    primary = SesNotificationConfig.from_environment()

    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    fallback = SesNotificationConfig.from_environment()

    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    unset = SesNotificationConfig.from_environment()

    monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", "false")
    disabled = SesNotificationConfig.from_environment()

    # The client is built with whatever the config carries and nothing else, so a
    # region the manifest never declared reaches boto3 as None.
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            client=lambda service, **kwargs: calls.append((service, kwargs))
            or SimpleNamespace()
        ),
    )
    SesEmailNotifier(primary)
    SesEmailNotifier(unset)

    declared = api_environment()

    assert primary.region_name == "us-east-2"
    assert fallback.region_name == "us-east-2"
    assert unset.region_name is None
    assert primary.execution_enabled is True
    assert disabled.execution_enabled is False
    assert calls == [
        ("sesv2", {"region_name": "us-east-2"}),
        ("sesv2", {"region_name": None}),
    ]
    assert declared["AWS_REGION"]["value"], declared["AWS_REGION"]
    assert declared["AWS_DEFAULT_REGION"]["value"], declared["AWS_DEFAULT_REGION"]


def test_boot014_the_dangerous_delivery_switch_combination_is_named_not_silent(
    monkeypatch,
) -> None:
    """Three individually sensible switches multiply into "nobody is paged".

    Async delivery writes the notification to the outbox; the dispatcher drains
    it. Email on, async on, dispatcher off means every notification is stored
    with no delivery and no error anywhere -- the exact state that made an
    unexecuted recovery indistinguishable from a healthy one. So the service has
    to resolve the product to a single no, and the manifest has to set the two
    switches together.
    """

    def service(*, allow_email: bool, async_delivery: bool, dispatcher: bool):
        monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", "gpu-fault@example.com")
        monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", "oncall@example.com")
        monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", str(allow_email).lower())
        monkeypatch.setenv(
            "GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY", str(async_delivery).lower()
        )
        monkeypatch.setenv(
            "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", str(dispatcher).lower()
        )
        notifier = SesEmailNotifier(
            SesNotificationConfig.from_environment(),
            client=SimpleNamespace(send_email=_never_called),
        )
        return AdvisoryNotificationService(build_store(), notifier)

    queued_nowhere = service(allow_email=True, async_delivery=True, dispatcher=False)
    drained = service(allow_email=True, async_delivery=True, dispatcher=True)
    inline = service(allow_email=True, async_delivery=False, dispatcher=False)
    no_email = service(allow_email=False, async_delivery=False, dispatcher=True)

    assert queued_nowhere.delivers_externally() is False
    assert drained.delivers_externally() is True
    assert inline.delivers_externally() is True
    assert no_email.delivers_externally() is False
    assert "NOT DELIVERED" in queued_nowhere.describe_delivery_mode()
    assert "does not drain it" in queued_nowhere.describe_delivery_mode()
    assert "NOT DELIVERED" not in drained.describe_delivery_mode()

    declared = api_environment()
    assert declared["GPU_FAULT_ALLOW_EMAIL"]["value"] == "true"
    assert declared["GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED"]["value"] == "true"


def _never_called(**_kwargs):
    raise AssertionError("no test may call SES")


def test_boot015_a_command_no_adapter_owns_is_skipped_silently_but_counted() -> None:
    """Owner filtering is a silent ``continue``, so metrics are the only witness.

    A ``runtime_profile`` that names an execution owner no executor implements
    produces commands nobody claims. The executor logs nothing -- it never sees
    them -- and the workflow simply waits, so the only signal that recovery has
    stopped is the unclaimed backlog gauge. That makes the metric, and the alert
    that reads it, load-bearing rather than decorative.
    """

    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    orphan = enqueue_remote_command(
        context.store,
        "remote-" + "e" * 24,
        owner="gpu-fault-adapter-nobody-implements",
        created_at=NOW,
    )

    claimed = context.store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=25,
        lease_seconds=60,
        execution_owners={"gpu-fault-kubernetes-adapter", "gpu-fault-node-agent"},
    )

    async def scenario() -> str:
        async with asgi_client(context) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    metrics = asyncio.run(scenario())
    stored = context.store.get_remote_command(orphan.command_id)
    rules = yaml.safe_load(AMP_RULES.read_text(encoding="utf-8"))

    assert claimed == []
    assert stored.status is RemoteCommandStatus.PENDING
    assert stored.lease_owner is None
    assert 'gpu_fault_remote_command_total{status="PENDING"} 1' in metrics
    assert "gpu_fault_remote_command_oldest_unclaimed_seconds" in metrics
    # The alert has to live on the path that is actually evaluated for this
    # deployment: AMP rules, not an unread PrometheusRule.
    assert "gpu_fault_remote_command_oldest_unclaimed_seconds" in json.dumps(rules), (
        AMP_RULES.name
    )
