from __future__ import annotations

import copy
import io
import json
import subprocess
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from botocore.credentials import Credentials

from scripts.e2e.regional import capacity_acceptance_base as base
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture


def test_constructor_binds_the_site_and_generates_isolated_registry(capacity) -> None:
    harness, api = capacity
    assert harness.cpu_kubeconfig.endswith("cpu-kubeconfig"), (
        "CPU access comes from the selected site"
    )
    assert harness.processor_mode == "active-active", (
        "the live processor mode is preserved"
    )
    assert len({item["cluster_id"] for item in harness.registrations}) == 20, (
        "load identities remain disjoint"
    )
    assert len(set(harness.tokens)) == 20, (
        "each synthetic cluster has a separate credential"
    )
    assert harness.runtime_image.endswith("a" * 64), (
        "probes use the current worker's pinned image"
    )
    assert harness.run_dir.stat().st_mode & 0o777 == 0o700, "run files are private"
    assert len(api.calls) == 3, (
        "construction only reads processor config, release pins and worker deployment"
    )
    assert harness.executor_pins == {
        "executor_artifact_sha256": "e" * 64,
        "executor_compatibility_digest": "f" * 64,
    }, "probe executors present the pins the release ConfigMap requires"
    assert harness.claim_identity() == {
        "executor_protocol_version": base.CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        **harness.executor_pins,
    }, "raw claims carry the current protocol version and both pins"


@pytest.mark.parametrize("mode", ["", "disabled", "active-passive"])
def test_unsupported_processor_mode_fails_before_resource_creation(
    capacity, mode
) -> None:
    harness, api = capacity
    api.processor_mode = mode
    with pytest.raises(base.CapError, match="supported probe mode"):
        harness.__init__(
            site_path=harness.site_path,
            run_dir=harness.root_run_dir,
            case_id=harness.case_id,
            predecessor={"valid": True},
        )
    assert not api.applied, "unknown processor mode cannot start a load generator"


def test_common_resource_artifacts_do_not_contain_credentials(capacity) -> None:
    harness, api = capacity
    harness.create_common_resources()
    assert [item["kind"] for item in api.applied] == ["Secret", "ConfigMap"], (
        "credentials and scripts have separate resources"
    )
    secret, scripts = api.applied
    assert set(secret["stringData"]) == {
        "clusters.json",
        "processor-replay-secret",
        "execution-token",
    }, "only the intended Secret carries credentials"
    assert len(scripts["data"]) == 4, "the probe receives all required script assets"
    output = (harness.run_dir / "synthetic-registry-summary.json").read_text(
        encoding="utf-8"
    )
    assert all(token not in output for token in harness.tokens), (
        "evidence keeps hashes, not raw credentials"
    )
    assert all(
        len(item["token_sha256"]) == 64 for item in json.loads(output)["clusters"]
    ), "registry summaries are verifiable"


def test_probe_environment_disables_every_hardware_and_email_action(capacity) -> None:
    harness, _ = capacity
    environment = {item["name"]: item for item in harness.base_environment("CAP001")}
    for key in (
        "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER",
        "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER",
        "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER",
        "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE",
        "GPU_FAULT_ALLOW_EMAIL",
        "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT",
    ):
        assert environment[key]["value"] == "false", (
            f"{key} must remain disabled for capacity traffic"
        )
    assert (
        environment["GPU_FAULT_BASE_STORE_URL"]["valueFrom"]["secretKeyRef"]["name"]
        == "gpu-fault-aurora"
    ), "database credentials remain a Secret reference"
    assert (
        environment["GPU_FAULT_EXECUTION_TOKEN"]["valueFrom"]["secretKeyRef"]["name"]
        == harness.secret_name
    ), "synthetic token is isolated from production"


def test_environment_overrides_replace_sources_without_duplicate_names(
    capacity,
) -> None:
    harness, _ = capacity
    values = [
        {"name": "EXAMPLE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}}
    ]
    harness.set_env(values, {"EXAMPLE": "new", "SECOND": "value"})
    assert values == [
        {"name": "EXAMPLE", "value": "new"},
        {"name": "SECOND", "value": "value"},
    ], "overrides cannot retain stale valueFrom entries"


@pytest.mark.parametrize("samples", [[], [1.0], [3.0, 1.0, 2.0]])
def test_percentile_handles_empty_and_unordered_samples(samples) -> None:
    expected = None if not samples else max(samples)
    assert base.percentile(samples, 1.0) == expected, (
        "p100 is the largest measured value or unavailable"
    )
    if samples:
        assert base.percentile(samples, 0) == min(samples), (
            "the lower bound selects the smallest sample"
        )


@pytest.mark.parametrize("bad", ['queue{unterminated="x} 1\n', "invalid metric 7"])
def test_metrics_parse_failure_is_explicit(bad) -> None:
    with pytest.raises(base.CapError, match="malformed"):
        base.CapHarnessBase.parse_metrics(bad)


def test_metric_selection_uses_all_requested_labels_and_the_maximum() -> None:
    samples = [
        ("queue", {"cluster": "a"}, 2.0),
        ("queue", {"cluster": "b"}, 100.0),
        ("queue", {"cluster": "a"}, 3.0),
        ("other", {"cluster": "a"}, 200.0),
    ]
    assert base.CapHarnessBase.metric_value(samples, "queue", cluster="a") == 3.0, (
        "another cluster cannot contaminate the selected metric"
    )


def test_metrics_http_status_is_checked_before_parsing(capacity, monkeypatch) -> None:
    harness, _ = capacity
    monkeypatch.setattr(
        base.httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            503, request=httpx.Request("GET", "http://probe.invalid")
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        harness.metrics("http://probe.invalid")
    monkeypatch.setattr(
        base.httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200, text="queue 4\n", request=httpx.Request("GET", "http://probe.invalid")
        ),
    )
    assert harness.metrics("http://probe.invalid") == [("queue", {}, 4.0)], (
        "healthy responses are parsed as Prometheus data"
    )


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_amp_query_is_signed_with_an_explicit_region_and_workspace(
    capacity, monkeypatch, method
) -> None:
    harness, _ = capacity
    credentials = Credentials("EXAMPLEACCESSKEY", "example-only-secret")
    monkeypatch.setattr(
        base.boto3,
        "Session",
        lambda: SimpleNamespace(get_credentials=lambda: credentials),
    )
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        assert timeout == 30, "AMP request time is bounded"
        return io.BytesIO(b'{"status":"success"}')

    monkeypatch.setattr(base.urllib.request, "urlopen", urlopen)
    result = harness.amp_request(
        method, "/api/v1/query", {"query": 'queue{cluster="a"}'}
    )
    request = requests[0]
    assert result == {"status": "success"}, "structured AMP response reaches the caller"
    assert request.full_url.startswith(
        "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/unit-workspace/"
    ), "the request cannot target another workspace"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert "AWS4-HMAC-SHA256" in headers["authorization"], (
        "query authorization must be signed"
    )
    assert (request.data is not None) is (method == "POST"), (
        "query body follows the signed method"
    )


def test_missing_amp_credentials_cannot_issue_a_request(capacity, monkeypatch) -> None:
    harness, _ = capacity
    monkeypatch.setattr(
        base.boto3, "Session", lambda: SimpleNamespace(get_credentials=lambda: None)
    )
    with pytest.raises(base.CapError, match="credentials unavailable"):
        harness.amp_request("GET", "/api/v1/alerts")


@pytest.mark.parametrize(
    "response",
    [
        {"status": "error"},
        {"status": "success"},
        {"status": "success", "data": {"alerts": [{"state": "unknown"}]}},
    ],
)
def test_unknown_alert_inventory_is_not_resolved(
    capacity, monkeypatch, response
) -> None:
    harness, _ = capacity
    monkeypatch.setattr(harness, "amp_request", lambda *args, **kwargs: response)
    with pytest.raises(base.CapError):
        harness.alert_states("UnitAlert")


def test_alert_selection_keeps_pending_and_firing_distinct(
    capacity, monkeypatch
) -> None:
    harness, _ = capacity
    alerts = [
        {"state": "firing", "labels": {"alertname": "UnitAlert"}},
        {"state": "pending", "labels": {"alertname": "UnitAlert"}},
        {"state": "firing", "labels": {"alertname": "OtherAlert"}},
    ]
    monkeypatch.setattr(
        harness,
        "amp_request",
        lambda *args, **kwargs: {"status": "success", "data": {"alerts": alerts}},
    )
    assert harness.alert_states("UnitAlert") == ["firing", "pending"], (
        "unrelated alerts cannot prove the target state"
    )


def test_connection_budget_counts_every_role_and_explicit_pool_override(
    capacity,
) -> None:
    harness, api = capacity
    api.deployments[0]["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "value": "10"}
    ]
    report = harness.connection_budget()
    assert report["theoretical_total"] == 66 and report["max_connections"] == 100, (
        "replicas, worker processes and effective pools all enter the budget"
    )
    assert report["budget_ratio"] == 0.66, "the denominator is the live database limit"


@pytest.mark.parametrize(
    "problem",
    [
        "missing-role",
        "bad-replicas",
        "bad-pool",
        "override-source",
        "zero-processes",
        "zero-limit",
    ],
)
def test_incomplete_connection_budget_is_rejected(capacity, problem) -> None:
    harness, api = capacity
    container = api.deployments[0]["spec"]["template"]["spec"]["containers"][0]
    if problem == "missing-role":
        api.deployments.pop(0)
    elif problem == "bad-replicas":
        api.deployments[0]["spec"]["replicas"] = True
    elif problem == "bad-pool":
        api.pool_size = "0"
    elif problem == "override-source":
        container["env"] = [
            {"name": "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "valueFrom": {}}
        ]
    elif problem == "zero-processes":
        container["args"] = ["uvicorn --workers 0"]
    else:
        api.exec_output = "0"
    with pytest.raises(base.CapError):
        harness.connection_budget()


def test_cloudwatch_samples_are_sorted_and_bound_to_aurora(
    capacity, monkeypatch
) -> None:
    harness, _ = capacity
    start = datetime(2026, 9, 12, tzinfo=timezone.utc)
    calls = []

    def statistics(**arguments):
        calls.append(arguments)
        return {
            "Datapoints": [
                {
                    "Timestamp": start + timedelta(minutes=1),
                    "Average": 2.0,
                    "Maximum": 3.0,
                },
                {"Timestamp": start, "Average": 1.0, "Maximum": 2.0},
            ]
        }

    monkeypatch.setattr(
        base.boto3,
        "client",
        lambda *args, **kwargs: SimpleNamespace(get_metric_statistics=statistics),
    )
    report = harness.cloudwatch_window(start, start + timedelta(minutes=2))
    assert len(report) == 2 and len(calls) == 2, "both CPU and connections are measured"
    assert all(
        call["Dimensions"]
        == [{"Name": "DBClusterIdentifier", "Value": "unit-database"}]
        for call in calls
    ), "a different database cannot provide the evidence"
    assert report["CPUUtilization"][0]["average"] == 1.0, (
        "sample order is temporal, not provider response order"
    )


@pytest.mark.parametrize(
    "problem", ["missing-list", "empty", "missing-role", "duplicate-role"]
)
def test_unknown_production_deployment_inventory_cannot_be_an_unchanged_baseline(
    capacity, monkeypatch, problem
) -> None:
    harness, api = capacity
    if problem == "missing-list":
        monkeypatch.setattr(harness, "kubectl_json", lambda *args: {})
    elif problem == "empty":
        api.deployments = []
    elif problem == "missing-role":
        api.deployments.pop()
    else:
        api.deployments.append(copy.deepcopy(api.deployments[0]))
    with pytest.raises(base.CapError):
        harness.production_baseline()


@pytest.mark.parametrize(
    "problem", ["empty-pods", "missing-statuses", "unready", "missing-restart-count"]
)
def test_production_pod_observation_must_be_complete_before_capacity_mutation(
    capacity, problem
) -> None:
    harness, api = capacity
    if problem == "empty-pods":
        api.pods = []
    elif problem == "missing-statuses":
        api.pods[0]["status"].pop("containerStatuses")
    elif problem == "unready":
        api.pods[0]["status"]["conditions"][0]["status"] = "False"
    else:
        api.pods[0]["status"]["containerStatuses"][0].pop("restartCount")
    with pytest.raises(base.CapError):
        harness.production_baseline()
    assert not api.applied, (
        "an unprovable baseline cannot authorize creation of probe resources"
    )


def test_init_container_restarts_are_part_of_the_production_baseline(capacity) -> None:
    harness, api = capacity
    api.pods[0]["spec"]["initContainers"] = [{"name": "prepare"}]
    api.pods[0]["status"]["initContainerStatuses"] = [
        {"name": "prepare", "restartCount": 0}
    ]
    before = harness.production_baseline()
    api.pods[0]["status"]["initContainerStatuses"][0]["restartCount"] = 1
    after = harness.production_baseline()
    assert before != after, (
        "an init-container restart is a production change even when main containers remain Ready"
    )


def test_cpu_commands_preserve_context_stdin_and_deadline(
    capacity, monkeypatch
) -> None:
    harness, _ = capacity
    calls = []

    def command(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, "observed", "")

    monkeypatch.setattr(base, "run_fixture_command", command)
    result = base.CapCoreHarness.kubectl(
        harness, "get", "pod", input_text="request", timeout=17
    )
    assert result.stdout == "observed", "command output remains available to parsers"
    arguments, options = calls[0]
    assert arguments[:5] == (
        "kubectl",
        "--kubeconfig",
        harness.cpu_kubeconfig,
        "-n",
        harness.namespace,
    ), "commands remain CPU-scoped"
    assert options["input_text"] == "request" and options["timeout"] == 17, (
        "stdin is not moved into argv and the deadline is preserved"
    )


def test_probe_control_and_cgroup_reads_use_selected_pod(capacity) -> None:
    harness, api = capacity
    api.exec_output = '{"released":true}'
    result = harness.probe_control(
        SimpleNamespace(pod="owned-probe"), "/__cap__/release", {"tag": "load"}
    )
    assert result == {"released": True}, "probe control response is decoded"
    arguments, options = api.calls[-1]
    assert arguments[2] == "owned-probe" and arguments[-2:] == (
        "/__cap__/release",
        '{"tag":"load"}',
    ), "the command uses only the selected probe identity"
    assert options["timeout"] == 45, "control requests are bounded"
    api.exec_output = "user_usec 7\nusage_usec 123\n"
    assert harness.pod_cpu_usage_usec("owned-probe") == 123, (
        "cgroup total usage is read instead of a partial counter"
    )
    api.exec_output = "user_usec 7\n"
    with pytest.raises(base.CapError, match="lacks usage_usec"):
        harness.pod_cpu_usage_usec("owned-probe")
    api.exec_output = "4\n"
    assert harness.isolated_db_connections("owned-probe") == 4, (
        "database connection count is scoped by the probe's own DSN"
    )


def test_ephemeral_port_is_reserved_only_on_loopback(monkeypatch) -> None:
    calls = []
    listener = SimpleNamespace(
        bind=lambda address: calls.append(address),
        getsockname=lambda: ("127.0.0.1", 43210),
    )
    monkeypatch.setattr(base.socket, "socket", lambda: nullcontext(listener))
    assert base.CapProbeHarness.reserve_port() == 43210, (
        "the kernel-assigned port is returned"
    )
    assert calls == [("127.0.0.1", 0)], "port reservation never binds a public address"


def test_disabled_spool_role_is_explicit_and_unrelated_workloads_are_ignored(
    capacity,
) -> None:
    harness, api = capacity
    api.deployments[2]["spec"]["replicas"] = 0
    api.deployments[2]["status"] = {}
    api.pods.pop(2)
    api.deployments.append({"metadata": {"name": "unrelated"}})
    api.pods.append({"metadata": {"name": "unrelated", "labels": {}}})
    snapshot = harness.production_baseline()
    spool = next(
        row for row in snapshot["deployments"] if row["name"].endswith("spool-worker")
    )
    assert spool["replicas"] == spool["ready"] == 0, (
        "disabled roles stay in the inventory with an explicit zero"
    )
    assert len(snapshot["pods"]) == 3, (
        "unrelated Pods cannot substitute for a missing managed Pod"
    )


def test_incomplete_pod_identity_is_a_controlled_baseline_failure(capacity) -> None:
    harness, api = capacity
    api.pods[0]["metadata"].pop("uid")
    with pytest.raises(base.CapError, match="Pod identity"):
        harness.production_baseline()
