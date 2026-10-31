"""GF-GENERAL-LIVE-031: which nodes and channels the silence scan leaves alone.

``disabled_unit`` matched unit names against advisory keys that name channels
and could never fail; ``expired_agent`` read an empty window and passed on an
Agent record that merely existed; ``new_success`` discarded its UNKNOWN wait.
The first tests here run the product's own silence scan
(``notify_silent_collectors``) over an in-memory store, read its records with
the harness's real probes (``ACTIVITY_PROBE``, ``AGENT_CURRENCY_PROBE``) and
grade them with the variants' graders: the disabled unit and the expired Agent
are excluded, and the same graders fail the records of a scan that did not
exclude them. The rest drive the variants over scripted records: each old
vacuous shape now ends NOT_RUN, INCONCLUSIVE or FAIL.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gpu_fault.app import ApplicationContext, notify_silent_collectors
from gpu_fault.collector_kinds import collector_silent_thresholds
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from scripts.e2e.regional import general_live_collectors as col
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    activity_read,
    make_context,
)

SPECS = {spec.variant: spec for spec in col.VARIANTS}
CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
DEPARTED = "ip-10-0-99-9.us-west-2.compute.internal"
#: The units the lane node's Agent reports, the log collector disabled as on
#: this site (``collector_requirements.RECORDED_UNITS`` names, no ``.service``).
ENABLED_UNITS = (
    "gpu-fault-host-collector",
    "gpu-fault-metrics-collector",
    "gpu-fault-kernel-collector",
    "gpu-fault-fabric-manager-collector",
)
LOG_UNIT = "gpu-fault-log-collector"
#: The variants' fake clock (``make_context``) starts here.
T0 = datetime(2026, 10, 24, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# the product's scan over an in-memory store
# --------------------------------------------------------------------------- #
def agent(node: str, *, now: datetime, lease: timedelta, log: str) -> AgentRecord:
    services = {
        unit: {"active": "active", "enabled": "enabled"} for unit in ENABLED_UNITS
    }
    services[LOG_UNIT] = {
        "active": "inactive" if log == "disabled" else "active",
        "enabled": log,
    }
    return AgentRecord(
        cluster_id=CLUSTER,
        node_id=node,
        endpoint="https://10.0.43.5:9443",
        agent_version="test",
        artifact_sha256="a" * 64,
        policy_version="test",
        runtime_profile_version="simulated-v1",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT],
        collector_services=services,
        first_seen_at=now - timedelta(hours=2),
        last_seen_at=now + min(lease, timedelta(0)),
        lease_expires_at=now + lease,
    )


def scanned_store(
    *, log: str = "disabled", departed_lease: timedelta = timedelta(hours=-1)
) -> tuple[InMemoryStore, datetime, list[str]]:
    """The lane node's current Agent (every enabled channel fresh), a departed
    node's Agent with ``departed_lease``, and two passes of the product's scan;
    the store, the moment before the scan and the notification ids it sent."""

    now = datetime.now(timezone.utc)
    store = InMemoryStore()
    store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id=CLUSTER,
            region="us-west-2",
            eks_cluster_arn=f"arn:aws:eks:us-west-2:1:cluster/{CLUSTER}",
            provider="eks-managed",
            token_sha256="a" * 64,
            agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        )
    )
    store.save_agent(agent(NODE, now=now, lease=timedelta(hours=1), log=log))
    store.save_agent(agent(DEPARTED, now=now, lease=departed_lease, log="enabled"))
    for kind in collector_silent_thresholds():
        if kind is CollectorKind.NODE_LOGS:
            continue
        store.save_collector_status(
            CollectorStatus(
                cluster_id=CLUSTER,
                node_id=NODE,
                collector=kind,
                observed_at=now,
                ingested_at=now,
                last_success_at=now,
                sample_count=1,
            )
        )
    context = ApplicationContext(store=store)
    sent: list[str] = []
    with mock.patch.object(context.advisory_notifications, "send", sent.append):
        for _ in range(2):
            notify_silent_collectors(
                context,
                observed_at=now,
                silent_after_seconds=collector_silent_thresholds(),
                alert_interval_seconds=3600,
            )
    return store, now - timedelta(seconds=1), sent


def read_activity(store: InMemoryStore, node: str, since: datetime) -> dict[str, Any]:
    arguments = [CLUSTER, node, since.isoformat(), col.OBSERVATION, "", ""]
    return activity_read(store, arguments)


def read_agents(store: InMemoryStore) -> dict[str, Any]:
    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    code = compile(col.AGENT_CURRENCY_PROBE, "<agent-currency-probe>", "exec")
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["agent-currency-probe", CLUSTER]),
        contextlib.redirect_stdout(output),
    ):
        exec(code, {"__name__": "__main__"})
    return json.loads(output.getvalue())


def test_the_products_scan_excludes_the_disabled_unit_and_the_graders_see_it() -> None:
    store, since, sent = scanned_store()
    observed = read_activity(store, NODE, since)
    excluded = col.excluded_channels(
        observed["agent"], [f"{LOG_UNIT}.service", "gpu-fault-host-collector.service"]
    )
    assert excluded == {f"{LOG_UNIT}.service": ["NODE_LOGS"]}, excluded
    others = col.observations_since(observed, since, exclude=["NODE_LOGS"])
    assert {item["device"] for item in others} >= {"HOST_TELEMETRY", "GPU_METRICS"}, (
        "the positive control: the scan re-recorded the node's other channels"
    )
    assert col.exclusion_errors(observed, since, ["NODE_LOGS"]) == {
        "not_observed": [],
        "no_advisory": [],
    }
    assert sent == [], "every required channel of the node is fresh: no advisory"


def test_the_graders_fail_a_scan_that_watches_the_unit_after_all() -> None:
    # The same node with the log collector reported enabled: the product's
    # scan requires the channel, finds no success and sends its advisory.
    store, since, _sent = scanned_store(log="enabled")
    observed = read_activity(store, NODE, since)
    assert col.excluded_channels(observed["agent"], [f"{LOG_UNIT}.service"]) == {}
    errors = col.exclusion_errors(observed, since, ["NODE_LOGS"])
    assert len(errors["not_observed"]) == 1, errors
    assert errors["no_advisory"] == [
        "a disabled collector produced a silent advisory: "
        + next(
            item["deduplication_key"]
            for item in observed["notifications"]
            if "/NODE_LOGS/" in item["deduplication_key"]
        )
    ], errors


def test_the_products_scan_skips_the_expired_agent_and_the_probe_names_it() -> None:
    store, since, sent = scanned_store()
    agents = read_agents(store)
    target = col.stale_agent(agents)
    assert target is not None and target["node_id"] == DEPARTED, agents
    assert {item["node_id"]: item["current"] for item in agents["agents"]} == {
        NODE: True,
        DEPARTED: False,
    }, agents
    expired = read_activity(store, DEPARTED, since)
    assert col.observations_since(expired, since) == [], expired["metric_latest"]
    assert expired["notifications"] == [] and sent == [], (expired, sent)
    # The control: the same departed node with a lease still running is
    # scanned, recorded UNKNOWN and announced -- what the variant would FAIL.
    store, since, sent = scanned_store(departed_lease=timedelta(minutes=5))
    assert col.stale_agent(read_agents(store)) is None
    current = read_activity(store, DEPARTED, since)
    assert col.observations_since(current, since), "a current Agent is scanned"
    keys = [item["deduplication_key"] for item in current["notifications"]]
    assert keys and all(key.startswith("collector-silent/") for key in keys), keys
    assert len(sent) == len(keys), (sent, keys)


# --------------------------------------------------------------------------- #
# the variants over scripted records
# --------------------------------------------------------------------------- #
def survey(units: dict[str, str]) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"systemctl": True, "systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "units": {
            f"{name}.service": {"UnitFileState": state, "ActiveState": "active"}
            for name, state in units.items()
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


SITE_UNITS = {"gpu-fault-host-collector": "enabled", LOG_UNIT: "disabled"}


def observation(channel: str, observed_at: datetime, state: str = "HEALTHY") -> dict:
    return {
        "name": col.OBSERVATION,
        "device": channel,
        "value": 1 if state == "HEALTHY" else 0,
        "labels": {"state": state, "reason": "FRESH_SUCCESS"},
        "observed_at": observed_at.isoformat(),
    }


def node_doc(
    *observations: dict[str, Any], log: str = "disabled", keys: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "metric_latest": list(observations),
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [
            {"deduplication_key": key, "notification_id": key} for key in keys
        ],
        "collector_statuses": [],
        "agent": {
            "lifecycle_state": "ACTIVE",
            "collector_services": {
                LOG_UNIT: {"active": "inactive", "enabled": log},
                "gpu-fault-host-collector": {"active": "active", "enabled": "enabled"},
            },
        },
    }


def run(
    variant: str,
    tmp_path: Path,
    *,
    nodes: dict[str, Any],
    agents: Any = None,
    units: dict[str, str] = SITE_UNITS,
) -> dict[str, Any]:
    """``execute`` with the activity of each node (``nodes[node]``: a document
    or a list consumed per read) and ``AGENT_CURRENCY_PROBE``'s answers."""

    reads: dict[str, int] = {}

    def answer(value: Any, key: str) -> dict[str, Any]:
        if isinstance(value, list):
            index = reads.get(key, 0)
            reads[key] = index + 1
            return dict(value[min(index, len(value) - 1)])
        return dict(value)

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if script is col.AGENT_CURRENCY_PROBE:
            return answer(agents, "agents")
        assert script is ACTIVITY_PROBE, "a node read runs the activity probe"
        return answer(nodes[arguments[1]], arguments[1])

    spec = SPECS[variant]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        probes={f"node@{NODE}": FakeProbe({"survey": survey(units)})},
    )
    return spec.execute(ctx)


FRESH = T0 + timedelta(minutes=2)
OLD = T0 - timedelta(hours=5)


def test_disabled_unit_passes_when_the_scan_ran_and_left_the_unit_alone(
    tmp_path: Path,
) -> None:
    doc = node_doc(observation("HOST_TELEMETRY", FRESH), observation("NODE_LOGS", OLD))
    result = run("disabled_unit", tmp_path, nodes={NODE: doc})
    assert result["verdict"] == "PASS", result["errors"]
    assert result["excluded_channels"] == {f"{LOG_UNIT}.service": ["NODE_LOGS"]}


@pytest.mark.parametrize(
    ("doc", "stage"),
    [
        (
            node_doc(
                observation("HOST_TELEMETRY", FRESH),
                observation("NODE_LOGS", FRESH, "UNKNOWN"),
            ),
            "not_observed",
        ),
        (
            node_doc(
                observation("HOST_TELEMETRY", FRESH),
                keys=(f"collector-silent/{CLUSTER}/{NODE}/NODE_LOGS/489",),
            ),
            "no_advisory",
        ),
    ],
    ids=["observed", "announced"],
)
def test_disabled_unit_fails_when_the_scan_watched_the_unit(
    tmp_path: Path, doc: dict[str, Any], stage: str
) -> None:
    result = run("disabled_unit", tmp_path, nodes={NODE: doc})
    assert result["verdict"] == "FAIL", result
    assert [name for name, errors in result["stages"].items() if errors] == [stage]


def test_disabled_unit_is_inconclusive_without_its_positive_controls(
    tmp_path: Path,
) -> None:
    # The old fixture's shape: no Agent record, nothing observed -- a PASS then.
    with pytest.raises(VariantInconclusive, match="does not report"):
        run("disabled_unit", tmp_path, nodes={NODE: {**node_doc(), "agent": {}}})
    with pytest.raises(VariantInconclusive, match="recorded no observation"):
        run(
            "disabled_unit",
            tmp_path / "b",
            nodes={NODE: node_doc(observation("HOST_TELEMETRY", OLD))},
        )


def test_disabled_unit_without_a_disabled_unit_is_not_run(tmp_path: Path) -> None:
    with pytest.raises(VariantNotRun) as raised:
        run(
            "disabled_unit",
            tmp_path,
            nodes={NODE: node_doc()},
            units={"gpu-fault-host-collector": "enabled"},
        )
    assert raised.value.reason == "CAPABILITY_ABSENT", raised.value


def agents_doc(*, departed_current: bool | None) -> dict[str, Any]:
    agents = [{"node_id": NODE, "current": True, "last_seen_at": T0.isoformat()}]
    if departed_current is not None:
        agents.append(
            {
                "node_id": DEPARTED,
                "current": departed_current,
                "last_seen_at": OLD.isoformat(),
                "lifecycle_state": "ACTIVE",
            }
        )
    return {"agents": agents}


def test_expired_agent_passes_when_the_scan_skipped_the_expired_node(
    tmp_path: Path,
) -> None:
    nodes = {
        NODE: node_doc(observation("HOST_TELEMETRY", FRESH)),
        DEPARTED: node_doc(observation("HOST_TELEMETRY", OLD, "UNKNOWN")),
    }
    result = run(
        "expired_agent",
        tmp_path,
        nodes=nodes,
        agents=agents_doc(departed_current=False),
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert result["expired_agent"]["node_id"] == DEPARTED, result


def test_expired_agent_fails_when_the_expired_node_was_scanned(tmp_path: Path) -> None:
    nodes = {
        NODE: node_doc(observation("HOST_TELEMETRY", FRESH)),
        DEPARTED: node_doc(
            observation("HOST_TELEMETRY", FRESH, "UNKNOWN"),
            keys=(f"collector-silent/{CLUSTER}/{DEPARTED}/HOST_TELEMETRY/489",),
        ),
    }
    result = run(
        "expired_agent",
        tmp_path,
        nodes=nodes,
        agents=agents_doc(departed_current=False),
    )
    assert result["verdict"] == "FAIL", result
    assert all(result["stages"].values()), result["stages"]


def test_expired_agent_without_a_non_current_agent_is_not_run(tmp_path: Path) -> None:
    # The old variant passed here: an Agent record existed, the window was empty.
    with pytest.raises(VariantNotRun) as raised:
        run(
            "expired_agent",
            tmp_path,
            nodes={NODE: node_doc()},
            agents=agents_doc(departed_current=None),
        )
    assert raised.value.reason == "FIXTURE_NOT_READY", raised.value


def test_expired_agent_is_inconclusive_when_the_agent_comes_back(
    tmp_path: Path,
) -> None:
    nodes = {NODE: node_doc(observation("HOST_TELEMETRY", FRESH)), DEPARTED: node_doc()}
    agents = [agents_doc(departed_current=False), agents_doc(departed_current=True)]
    with pytest.raises(VariantInconclusive, match="did not stay non-current"):
        run("expired_agent", tmp_path, nodes=nodes, agents=agents)
