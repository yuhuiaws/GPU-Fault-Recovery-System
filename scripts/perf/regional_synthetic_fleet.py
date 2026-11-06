"""A perf run's synthetic healthy fleet: production-sized perf clusters (§8.5).

Since the fixed remediation limits were retired (detailed design v2 §9.5), a
GPU cluster's remediation storm breaker holds the cluster's new destructive
work once at least max(3, ceil(20% x its GPU nodes)) distinct nodes of two or
more workload domains newly enter destructive remediation within 600 s. Its
GPU nodes are the cluster's fleet Agents that are ACTIVE or DRAINING
(``gpu_fault.execution.storm_admission.cluster_gpu_nodes``, read at the
cluster's first admissions and at most once a minute per process), lease or
no lease. The suites registered Agents for the model's own nodes only, so
every synthetic cluster sat at the floor of 3 and tripped on its second or
third workflow.

Every action-bearing suite therefore registers a healthy fleet in each of its
synthetic clusters, beside the model's nodes, so that each cluster registers
``--fleet-nodes-per-cluster`` GPU nodes in all (default 256, the formal
model's cluster size, threshold 52). Fewer than 5 x the model's nodes + 1 is
refused: that puts the threshold above the model's nodes. The fleet is

* registered before any workflow of the run can be claimed (the seed script's
  ``fleet-agents`` mode), with the identity the model's own Agents carry;
* owned by the run: node ids ``fleet-node-<run>-c<NNN>-n<NNNN>`` in the run's
  registered ``perf-cap-*`` clusters, purged with them by the teardown, whose
  record walk takes every record of those clusters;
* heartbeating for the whole run: one in-Pod session
  (``synthetic_fleet_pod.py``) renews every fleet Agent by compare-and-set,
  one pass every ``FLEET_HEARTBEAT_SECONDS`` against a ``FLEET_LEASE_SECONDS``
  lease, and stops before the teardown, so a purged Agent is never written
  back;
* never a workflow's node: the model's refreshes, telemetry and watches skip
  it, and it publishes no telemetry (its collectors read silent on the
  ``perf-cap-*`` clusters, whose alerts the drill sink keeps).

``fleet.json`` records the plan, the registration, the heartbeat sessions and
the census taken when the run ends: per cluster the breaker's own node count
and threshold, the fleet's size and currency, and the breaker record. An
ordinary step fails when the breaker judged a cluster against a threshold no
higher than the model's nodes (the fleet was not counted), when a cluster's
breaker tripped, or when the fleet shrank or its heartbeat lapsed; a threshold
merely other than planned is a note. A breaker still open at the end is
handed over and released (``abort``, then ``release``) before the teardown, so
no run leaves one open.
``fleet-cleanup.json`` is the census after the teardown: no Agent and no
breaker record of the run's clusters may be left.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.remediation_storm import STORM_WINDOW_SECONDS, storm_threshold
from scripts.e2e.regional.regional_live_fixture import component_python

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
    from . import regional_storm_breaker as storm
    from .regional_capacity_recovery import cleanup_signal_guard
    from .regional_validation_telemetry import ValidationTelemetryWatcher
else:
    import regional_capacity_registry as registry
    import regional_storm_breaker as storm
    from regional_capacity_recovery import cleanup_signal_guard
    from regional_validation_telemetry import ValidationTelemetryWatcher

PERF_DIR = Path(__file__).resolve().parent
SEED_SCRIPT = PERF_DIR / "seed_regional_action_workflows.py"
FLEET_POD_SCRIPT = PERF_DIR / "synthetic_fleet_pod.py"
DEFAULT_FLEET_NODES_PER_CLUSTER = 256
#: Registered nodes per cluster at least FLEET_FACTOR x the model's + 1, so
#: ceil(20%) of them exceeds the model's nodes.
FLEET_FACTOR = 5
FLEET_LEASE_SECONDS = 900
FLEET_HEARTBEAT_SECONDS = 300
FLEET_SEED_TIMEOUT_SECONDS = 1200
FLEET_ARTIFACT = "fleet.json"
FLEET_CLEANUP_ARTIFACT = "fleet-cleanup.json"
FLEET_AGENTS_MODE = "fleet-agents"
HEARTBEAT_MODE = "fleet-heartbeat-watch"
CENSUS_MODE = "fleet-census"
FLEET_BOOTSTRAP = (
    "import sys;stream=sys.stdin.buffer;size=int(stream.readline());"
    "source=stream.read(size).decode();"
    "exec(compile(source,'synthetic_fleet_pod.py','exec'),{'__name__':'__main__'})"
)

Control = Callable[..., str]


def minimum_nodes(model_nodes_per_cluster: int) -> int:
    return FLEET_FACTOR * model_nodes_per_cluster + 1


def add_fleet_argument(parser: Any) -> None:
    parser.add_argument(
        "--fleet-nodes-per-cluster",
        type=int,
        default=DEFAULT_FLEET_NODES_PER_CLUSTER,
        help=(
            "GPU nodes each synthetic cluster registers, the model's own "
            "included: the remediation storm breaker trips at 20%% of them "
            "(default %(default)s; at least 5 x the model's nodes + 1)"
        ),
    )


def fleet_plan(
    *,
    clusters: int,
    model_nodes_per_cluster: int,
    nodes_per_cluster: int,
    storm: bool = False,
) -> dict[str, Any]:
    """The fleet of every synthetic cluster and the threshold it gives.

    ``storm`` is the breaker step's: its model crosses the threshold on
    purpose, so only ``nodes_per_cluster >= model nodes`` is required.
    """

    floor = model_nodes_per_cluster if storm else minimum_nodes(model_nodes_per_cluster)
    if clusters < 1 or model_nodes_per_cluster < 1 or nodes_per_cluster < floor:
        raise ValueError(
            f"--fleet-nodes-per-cluster {nodes_per_cluster} is below {floor} = "
            + (
                "the breaker step's own workflow nodes"
                if storm
                else f"{FLEET_FACTOR} x {model_nodes_per_cluster} model nodes + 1"
            )
            + ": the model would reach the remediation storm breaker's threshold"
        )
    threshold = storm_threshold(nodes_per_cluster)
    fleet = nodes_per_cluster - model_nodes_per_cluster
    return {
        "clusters": clusters,
        "nodes_per_cluster": nodes_per_cluster,
        "model_nodes_per_cluster": model_nodes_per_cluster,
        "fleet_agents_per_cluster": fleet,
        "fleet_agents": clusters * fleet,
        "threshold_nodes": threshold,
        "model_margin_nodes": threshold - model_nodes_per_cluster,
        "storm_step": storm,
        "window_seconds": STORM_WINDOW_SECONDS,
        "lease_seconds": FLEET_LEASE_SECONDS,
        "heartbeat_seconds": FLEET_HEARTBEAT_SECONDS,
    }


def plan_line(plan: dict[str, Any]) -> str:
    position = (
        f"the model's {plan['model_nodes_per_cluster']} nodes cross it on purpose"
        if plan["storm_step"]
        else f"the model stays {plan['model_margin_nodes']} nodes below it"
    )
    return (
        f"synthetic fleet: {plan['clusters']} clusters x {plan['nodes_per_cluster']} "
        f"registered GPU nodes ({plan['model_nodes_per_cluster']} model + "
        f"{plan['fleet_agents_per_cluster']} healthy fleet; {plan['fleet_agents']} "
        f"fleet Agents); remediation storm breaker threshold "
        f"{plan['threshold_nodes']} nodes in {plan['window_seconds']} s, {position}"
    )


def register_fleet(
    *,
    run_id: str,
    plan: dict[str, Any],
    agent_identity: dict[str, object] | None,
    control_fn: Control | None = None,
) -> dict[str, Any]:
    """The seed script's ``fleet-agents`` mode in an api-ha Pod."""

    if not plan["fleet_agents_per_cluster"]:
        return {"agents_created": 0}
    kube = registry.control if control_fn is None else control_fn
    environment = [
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={plan['clusters']}",
        f"ACTION_SEED_MODE={FLEET_AGENTS_MODE}",
        f"ACTION_FLEET_AGENTS_PER_CLUSTER={plan['fleet_agents_per_cluster']}",
        f"ACTION_AGENT_LEASE_SECONDS={plan['lease_seconds']}",
    ]
    if agent_identity is not None:
        environment.append(
            "ACTION_AGENT_IDENTITY_JSON="
            + json.dumps(agent_identity, separators=(",", ":"), sort_keys=True)
        )
    output = kube(
        "exec",
        "-i",
        storm.api_pod(kube),
        "--",
        "env",
        *environment,
        component_python("cpu"),
        "-",
        stdin=SEED_SCRIPT.read_bytes(),
        timeout=FLEET_SEED_TIMEOUT_SECONDS,
    )
    document = storm.last_json(output)
    if document.get("agents_created") != plan["fleet_agents"]:
        raise RuntimeError(
            f"synthetic fleet registration created {document.get('agents_created')} "
            f"Agents, planned {plan['fleet_agents']}"
        )
    return document


def fleet_census(
    *, run_id: str, clusters: int, control_fn: Control | None = None
) -> dict[str, Any]:
    """``synthetic_fleet_pod.py`` census of the run's clusters in an api-ha Pod."""

    kube = registry.control if control_fn is None else control_fn
    output = kube(
        "exec",
        "-i",
        storm.api_pod(kube),
        "--",
        "env",
        f"FLEET_RUN_ID={run_id}",
        f"FLEET_CLUSTERS={clusters}",
        f"FLEET_MODE={CENSUS_MODE}",
        component_python("cpu"),
        "-",
        stdin=FLEET_POD_SCRIPT.read_bytes(),
        timeout=300,
    )
    document = storm.last_json(output)
    if document.get("mode") != CENSUS_MODE:
        raise RuntimeError("synthetic fleet census returned no census")
    return document


class FleetHeartbeat(ValidationTelemetryWatcher):
    """The in-Pod heartbeat session of the run's fleet (start, then stop)."""

    script = FLEET_POD_SCRIPT
    final_endings = frozenset({"deadline", "fleet-gone"})

    def __init__(
        self,
        *,
        run_id: str,
        clusters: int,
        deadline_seconds: float,
        lease_seconds: float = FLEET_LEASE_SECONDS,
        interval_seconds: float = FLEET_HEARTBEAT_SECONDS,
        **options: Any,
    ) -> None:
        options.setdefault("max_restarts", 10)
        options.setdefault("restart_pause_seconds", 5.0)
        super().__init__(
            run_id=run_id,
            clusters=clusters,
            deadline_seconds=deadline_seconds,
            mode=HEARTBEAT_MODE,
            **options,
        )
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds

    def _argv(self, pod: str) -> list[str]:
        return [
            "kubectl",
            "--kubeconfig",
            registry.CONTROL_KUBECONFIG,
            "-n",
            registry.CONTROL_NAMESPACE,
            "exec",
            "-i",
            pod,
            "--",
            "env",
            f"FLEET_RUN_ID={self.run_id}",
            f"FLEET_CLUSTERS={self.clusters}",
            f"FLEET_MODE={HEARTBEAT_MODE}",
            f"FLEET_LEASE_SECONDS={self.lease_seconds:g}",
            f"FLEET_INTERVAL_SECONDS={self.interval_seconds:g}",
            f"FLEET_DEADLINE_SECONDS={self.deadline_seconds:g}",
            component_python("cpu"),
            "-c",
            FLEET_BOOTSTRAP,
        ]

    def heartbeat_summary(self, *, stop_confirmed: bool) -> dict[str, Any]:
        sessions = list(self.sessions)
        return {
            "stop_confirmed": stop_confirmed,
            "sessions": sessions,
            "restarts": max(0, len(sessions) - 1),
            "passes": sum(int(item.get("passes", 0)) for item in sessions),
            "renewed": sum(int(item.get("renewed", 0)) for item in sessions),
            "not_renewed": sum(int(item.get("not_renewed", 0)) for item in sessions),
            "longest_renewal_gap_seconds": max(
                (
                    float(item.get("longest_renewal_gap_seconds", 0))
                    for item in sessions
                ),
                default=0.0,
            ),
            "lease_seconds": self.lease_seconds,
            "interval_seconds": self.interval_seconds,
        }


def fleet_errors(
    plan: dict[str, Any],
    census: dict[str, Any],
    heartbeat: dict[str, Any],
    *,
    expected_trips: frozenset[str],
) -> list[str]:
    """Why the fleet did not give the run the breaker it planned (module docstring)."""

    errors: list[str] = []
    if "clusters" not in census:
        return [f"synthetic fleet census failed: {census.get('error')}"]
    for cluster_id, item in sorted(census["clusters"].items()):
        if item["fleet_agents"] != plan["fleet_agents_per_cluster"]:
            errors.append(
                f"{cluster_id} has {item['fleet_agents']} fleet Agents, the run "
                f"registered {plan['fleet_agents_per_cluster']}"
            )
        if item["fleet_current"] != item["fleet_agents"]:
            errors.append(
                f"{cluster_id}: {item['fleet_agents'] - item['fleet_current']} fleet "
                "Agents are not ACTIVE with a live lease (the heartbeat lapsed)"
            )
        breaker = item.get("breaker")
        if not census.get("product_breaker") or breaker is None:
            continue
        model = plan["model_nodes_per_cluster"]
        if not plan["storm_step"] and breaker["threshold"] <= model:
            errors.append(
                f"the breaker last judged {cluster_id} against {breaker['threshold']} "
                f"nodes, no more than the model's {model}: the fleet was not counted"
            )
        if breaker["trip_id"] and cluster_id not in expected_trips:
            errors.append(
                f"the remediation storm breaker of {cluster_id} tripped during the "
                f"run ({breaker['trip_nodes']} nodes, threshold "
                f"{breaker['trip_threshold']}, trip {breaker['trip_id']})"
            )
    if not heartbeat.get("stop_confirmed"):
        errors.append("the fleet heartbeat session did not stop")
    if heartbeat.get("longest_renewal_gap_seconds", 0) >= plan["lease_seconds"]:
        errors.append("a fleet Agent went a whole lease without a heartbeat")
    last = (heartbeat.get("sessions") or [{}])[-1]
    if last.get("ended_by") != "stopped":
        errors.append(
            f"the fleet heartbeat session ended {last.get('ended_by') or 'lost'} "
            "before the run did"
        )
    return errors


def fleet_notes(plan: dict[str, Any], census: dict[str, Any]) -> list[str]:
    """Deviations that leave the model below the breaker: a threshold the
    breaker judged other than planned (say a model Agent the product retired),
    recorded rather than failed. The storm step judges its own exactly."""

    notes = []
    for cluster_id, item in sorted((census.get("clusters") or {}).items()):
        breaker = item.get("breaker")
        if breaker is not None and breaker["threshold"] != plan["threshold_nodes"]:
            notes.append(
                f"the breaker last judged {cluster_id} against {breaker['threshold']} "
                f"nodes, the plan's threshold is {plan['threshold_nodes']}"
            )
    return notes


def close_open_breakers(
    census: dict[str, Any], *, run_id: str, control_fn: Control | None = None
) -> list[dict[str, Any]]:
    """Hand over and release every breaker of the run's clusters still open."""

    exits: list[dict[str, Any]] = []
    for cluster_id, item in sorted((census.get("clusters") or {}).items()):
        breaker = item.get("breaker") or {}
        if not breaker.get("open"):
            continue
        record: dict[str, Any] = {"cluster_id": cluster_id, "state": breaker["state"]}
        options = {
            "reason": f"perf run {run_id} ends with its breaker open",
            "reference": f"PERF-STORM-{run_id}",
            "control_fn": control_fn,
        }
        try:
            if breaker["state"] == "TRIPPED":
                record["abort"] = storm.storm_operator_action(
                    cluster_id, "abort", **options
                )
            record["release"] = storm.storm_operator_action(
                cluster_id, "release", **options
            )
        except Exception as exc:  # noqa: BLE001 - the teardown purges it regardless
            record["error"] = f"{type(exc).__name__}: {exc}"
        exits.append(record)
    return exits


def write_document(artifacts: Path, name: str, document: dict[str, Any]) -> None:
    (artifacts / name).write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    )


@contextmanager
def synthetic_fleet(
    artifacts: Path,
    *,
    run_id: str,
    plan: dict[str, Any],
    agent_identity: dict[str, object] | None,
    deadline_seconds: float,
    expected_trips: frozenset[str] = frozenset(),
    control_fn: Control | None = None,
    heartbeat_options: dict[str, Any] | None = None,
    log: Callable[[str], None] = registry.log,
) -> Iterator[dict[str, Any]]:
    """Register the fleet and keep it heartbeating while the block runs.

    Leaving the block stops the heartbeat, takes the census, releases any
    breaker still open and writes ``fleet.json``; a fleet verdict that failed
    raises unless the block itself is already raising.
    """

    document: dict[str, Any] = {"run_id": run_id, "plan": plan}
    write_document(artifacts, FLEET_ARTIFACT, document)
    log(plan_line(plan))
    document["registration"] = register_fleet(
        run_id=run_id,
        plan=plan,
        agent_identity=agent_identity,
        control_fn=control_fn,
    )
    write_document(artifacts, FLEET_ARTIFACT, document)
    heartbeat = FleetHeartbeat(
        run_id=run_id,
        clusters=plan["clusters"],
        deadline_seconds=deadline_seconds,
        lease_seconds=plan["lease_seconds"],
        interval_seconds=plan["heartbeat_seconds"],
        control_fn=control_fn,
        **(heartbeat_options or {}),
    )
    heartbeat.start()
    raising = False
    try:
        yield document
    except BaseException:
        raising = True
        raise
    finally:
        # The run's cleanup starts here: a signal no longer interrupts it.
        with cleanup_signal_guard():
            stopped = heartbeat.stop()
            document["heartbeat"] = heartbeat.heartbeat_summary(
                stop_confirmed=bool(stopped.get("stop_confirmed"))
            )
            try:
                census = fleet_census(
                    run_id=run_id, clusters=plan["clusters"], control_fn=control_fn
                )
            except Exception as exc:  # noqa: BLE001 - recorded, fails the verdict
                census = {"error": f"{type(exc).__name__}: {exc}"}
            document["census"] = census
            document["breaker_exits"] = close_open_breakers(
                census, run_id=run_id, control_fn=control_fn
            )
            errors = fleet_errors(
                plan, census, document["heartbeat"], expected_trips=expected_trips
            )
            document["errors"] = errors
            document["notes"] = fleet_notes(plan, census)
            document["verdict"] = "FAIL" if errors else "PASS"
            write_document(artifacts, FLEET_ARTIFACT, document)
        if errors and not raising:
            raise RuntimeError("synthetic fleet: " + "; ".join(errors))


def verify_fleet_cleanup(
    artifacts: Path,
    *,
    run_id: str,
    clusters: int,
    control_fn: Control | None = None,
) -> dict[str, Any]:
    """After the teardown: no Agent and no breaker record of the run's clusters."""

    census = fleet_census(run_id=run_id, clusters=clusters, control_fn=control_fn)
    residue = {
        cluster_id: {"agents": item["agents"], "breaker": item["breaker"]}
        for cluster_id, item in sorted(census["clusters"].items())
        if item["agents"] or item["breaker"] is not None
    }
    document = {"run_id": run_id, "census": census, "residue": residue}
    write_document(artifacts, FLEET_CLEANUP_ARTIFACT, document)
    if residue:
        raise RuntimeError(
            "synthetic fleet cleanup left "
            + ", ".join(
                f"{cluster_id}: {item['agents']} Agents"
                + (" and a breaker record" if item["breaker"] is not None else "")
                for cluster_id, item in residue.items()
            )
        )
    return document
