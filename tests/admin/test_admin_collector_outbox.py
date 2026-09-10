"""``gpu-fault-admin collector-outbox``: the node-local ``gpu-fault-collector
outbox`` command carried to the node as a NODE_ACTION step.

Client-side validation refuses before anything leaves the host; the in-Pod
script gates on the agent heartbeat and builds the pair with the product
builder; the verb waits for the terminal state, interprets the node's answer
(or the refusal) and archives under ``<state-dir>/collector-outbox/<node>/``.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import collector_outbox as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests.admin.conftest import TEST_OPERATOR_ARN

NOW = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)
NODE = "ip-10-0-0-7.ec2.internal"


def _site(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        environment={},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "namespace": "gpu-fault-system",
            "runtime_profile": {"version": "hyperpod-v1"},
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
    )


def _request(tmp_path: Path, **overrides) -> module.CollectorOutboxRequest:
    values = dict(
        site=_site(tmp_path),
        cluster_id="gpu-a",
        node_id=NODE,
        collector="kernel",
        action="stats",
        reference="CHG-11",
    )
    values.update(overrides)
    return module.CollectorOutboxRequest(**values)


# --------------------------------------------------------------------------
# Client-side validation


def test_requeue_dead_requires_yes(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="pass --yes"):
        _request(tmp_path, action="requeue-dead")
    _request(tmp_path, action="requeue-dead", confirm=True)


def test_every_action_requires_a_reference(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="--reference"):
        _request(tmp_path, reference="")


def test_unknown_collector_action_path_and_cluster_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="unknown collector"):
        _request(tmp_path, collector="sqs-hma")
    with pytest.raises(BootstrapError, match="unknown outbox action"):
        _request(tmp_path, action="purge")
    with pytest.raises(BootstrapError, match="path filter"):
        _request(tmp_path, action="list", path="v1/no-slash")
    with pytest.raises(BootstrapError, match="not in the managed site"):
        _request(tmp_path, cluster_id="gpu-z")


def test_there_is_no_remote_force(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        _request(tmp_path, force=True)  # type: ignore[call-arg]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [
                "collector-outbox",
                "--state-dir",
                str(tmp_path),
                "--cluster-id",
                "gpu-a",
                "--node",
                NODE,
                "--collector",
                "kernel",
                "--action",
                "requeue-dead",
                "--yes",
                "--force",
                "--reference",
                "CHG-1",
            ]
        )


def test_control_plane_script_compiles() -> None:
    compile(module.COLLECTOR_OUTBOX_SCRIPT, "collector-outbox", "exec")


# --------------------------------------------------------------------------
# The in-Pod script against an in-memory control plane


def _agent(store, *, operations: list[WorkflowOperation]) -> AgentRecord:
    record = AgentRecord(
        cluster_id="gpu-a",
        node_id=NODE,
        endpoint="https://10.0.0.7:9099",
        agent_version="2026.9.10",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="hyperpod-v1",
        config_digest="b" * 64,
        allowed_operations=operations,
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    store.save_agent(record)
    return record


class InPod:
    """Runs ``COLLECTOR_OUTBOX_SCRIPT`` the way ``kubectl exec`` would."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        from gpu_fault.app import ApplicationContext
        from tests._builders import build_store

        self.store = build_store()
        self.context = ApplicationContext(store=self.store)
        self.woken: list[bool] = []
        monkeypatch.setattr(
            self.context.dispatcher, "wake", lambda: self.woken.append(True)
        )
        monkeypatch.setattr(
            ApplicationContext,
            "from_environment",
            classmethod(lambda cls: self.context),
        )
        self.monkeypatch = monkeypatch
        self.capsys = capsys

    def run(self, payload: dict) -> dict:
        self.monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(compile(module.COLLECTOR_OUTBOX_SCRIPT, "<collector-outbox>", "exec"), {})
        return json.loads(self.capsys.readouterr().out)


def _submit_payload(**overrides) -> dict:
    payload = {
        "mode": "submit",
        "cluster_id": "gpu-a",
        "node_id": NODE,
        "collector": "kernel",
        "action": "requeue-dead",
        "confirm": True,
        "path": None,
        "operator": TEST_OPERATOR_ARN,
        "reference": "CHG-11",
        "runtime_profile_version": "hyperpod-v1",
    }
    payload.update(overrides)
    return payload


def test_the_in_pod_submit_builds_the_pair_and_wakes_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pod = InPod(monkeypatch, capsys)
    _agent(
        pod.store,
        operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE,
        ],
    )

    result = pod.run(_submit_payload())

    workflow = pod.store.get_workflow(result["workflow"]["request_id"])
    assert workflow.request_id.startswith("workflow-collector-outbox-"), (
        workflow.request_id
    )
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.fencing_token == 1
    assert workflow.runtime_profile_version == "hyperpod-v1"
    step = workflow.official_steps[0]
    assert step.operation is WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
    assert step.node_ids == [NODE]
    assert step.parameters["collector"] == "kernel"
    assert step.parameters["action"] == "requeue-dead"
    assert step.parameters["confirm"] is True
    assert step.parameters["operator"] == TEST_OPERATOR_ARN
    incident = pod.store.get_incident(result["incident"]["incident_id"])
    assert incident.incident_id.startswith(
        f"inc-operator-outbox-{NODE}-requeue-dead-"
    ), incident.incident_id
    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.policy_source == "OPERATOR"
    assert incident.workflow_request_id == workflow.request_id
    assert result["agent_version"] == "2026.9.10"
    assert pod.woken == [True]

    status = pod.run({"mode": "status", "workflow_request_id": workflow.request_id})
    assert status["workflow"]["request_id"] == workflow.request_id
    assert status["incident"]["incident_id"] == incident.incident_id


def test_the_in_pod_submit_refuses_a_node_without_a_heartbeat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pod = InPod(monkeypatch, capsys)

    with pytest.raises(SystemExit, match="no agent heartbeat"):
        pod.run(_submit_payload())
    assert pod.woken == []


def test_the_in_pod_submit_refuses_an_agent_that_predates_the_operation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An agent whose heartbeat does not advertise the operation would answer
    the command with HTTP 422/403; the gate refuses before an incident exists
    and points at the node-local CLI."""

    pod = InPod(monkeypatch, capsys)
    _agent(pod.store, operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS])

    with pytest.raises(SystemExit) as info:
        pod.run(_submit_payload())

    message = str(info.value)
    assert "predates outbox maintenance" in message
    assert "2026.9.10" in message
    assert "node-local gpu-fault-collector outbox" in message
    assert pod.woken == []
    assert pod.store.list_workflows() == [], "no workflow was written"


# --------------------------------------------------------------------------
# The verb end to end, with the exec channel replaced


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = _site(tmp_path)
        self.calls: list[dict] = []
        self.polls_until_terminal = 1
        self.terminal_status = "SUCCEEDED"
        self.incident_state = "RECOVERED"
        self.execution: dict = {
            "step_index": 0,
            "operation": "COLLECTOR_OUTBOX_MAINTENANCE",
            "status": "SUCCEEDED",
            "phase": "official",
            "details": {
                "node_results": {
                    NODE: {
                        "collector": "kernel",
                        "action": "stats",
                        "stats": {
                            "depth": 3,
                            "replayable": 1,
                            "dead": 2,
                            "payload_truncated": 1,
                            "oldest_failed_at": "2026-09-10T08:01:00+00:00",
                        },
                        "lock_holder": None,
                    }
                }
            },
        }
        self.slept: list[float] = []
        self.ticks = 0.0

        def script(site, payload, *, script):
            assert script is module.COLLECTOR_OUTBOX_SCRIPT
            self.calls.append(payload)
            if payload["mode"] == "submit":
                return {
                    "incident": {
                        "incident_id": "inc-operator-outbox-x",
                        "state": "ACTION_PENDING",
                    },
                    "workflow": {
                        "request_id": "workflow-collector-outbox-1",
                        "incident_id": "inc-operator-outbox-x",
                        "status": "PENDING",
                        "step_executions": [],
                    },
                    "agent_version": "2026.9.10",
                }
            polls = sum(1 for item in self.calls if item["mode"] == "status")
            terminal = polls >= self.polls_until_terminal
            return {
                "incident": {
                    "incident_id": "inc-operator-outbox-x",
                    "state": self.incident_state if terminal else "ACTION_PENDING",
                },
                "workflow": {
                    "request_id": "workflow-collector-outbox-1",
                    "incident_id": "inc-operator-outbox-x",
                    "status": self.terminal_status if terminal else "RUNNING",
                    "step_executions": [self.execution] if terminal else [],
                },
            }

        monkeypatch.setattr(module, "run_control_plane_script", script)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.ticks += seconds

    def clock(self) -> float:
        return self.ticks

    def run(
        self, request: module.CollectorOutboxRequest
    ) -> module.CollectorOutboxResult:
        return module.run_collector_outbox(
            request, now=lambda: NOW, sleep=self.sleep, clock=self.clock
        )

    def evidence(self) -> list[Path]:
        return sorted(module.evidence_directory(self.site, NODE).glob("*.json"))


def test_stats_submits_waits_for_the_terminal_state_and_prints_the_node_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.polls_until_terminal = 3

    result = harness.run(_request(tmp_path, site=harness.site, path="/v1/kernel-logs"))

    submit = harness.calls[0]
    assert submit["mode"] == "submit"
    assert (submit["cluster_id"], submit["node_id"]) == ("gpu-a", NODE)
    assert (submit["collector"], submit["action"], submit["confirm"]) == (
        "kernel",
        "stats",
        False,
    )
    assert submit["path"] == "/v1/kernel-logs"
    assert submit["operator"] == TEST_OPERATOR_ARN
    assert submit["reference"] == "CHG-11"
    assert submit["runtime_profile_version"] == "hyperpod-v1"
    assert [item["mode"] for item in harness.calls[1:]] == ["status"] * 3
    assert harness.slept == [module.POLL_INTERVAL_SECONDS] * 2
    assert result.succeeded is True
    assert result.workflow_status == "SUCCEEDED"
    assert result.incident_state == "RECOVERED"
    assert result.node_result["stats"]["dead"] == 2
    assert result.agent_version == "2026.9.10"
    assert "depth 3, replayable 1, dead 2" in result.message
    evidence = harness.evidence()
    assert [path.name for path in evidence] == ["stats-20260910T090000Z.json"]
    archived = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert archived["status"] == "SUCCEEDED"
    assert archived["operator"] == TEST_OPERATOR_ARN
    assert archived["reference"] == "CHG-11"
    assert archived["result"]["stats"]["depth"] == 3
    assert archived["workflow_request_id"] == "workflow-collector-outbox-1"


def test_list_rows_are_metadata_and_requeue_reports_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.execution["details"]["node_results"][NODE] = {
        "action": "list",
        "records": [
            {
                "index": 0,
                "request_id": "event-1",
                "path": "/v1/kernel-logs",
                "status": "dead",
                "replayable": False,
                "payload_truncated": False,
                "error": "HTTP 422",
                "failed_at": "2026-09-10T08:01:00+00:00",
            }
        ],
    }
    listed = harness.run(_request(tmp_path, site=harness.site, action="list"))
    assert listed.node_result["records"][0]["request_id"] == "event-1"
    assert "1 record(s) (1 dead)" in listed.message
    assert "payloads never leave the node" in listed.message

    harness.execution["details"]["node_results"][NODE] = {
        "action": "requeue-dead",
        "requeued": 2,
        "skipped_payload_truncated": 1,
        "dead_before": 3,
        "dead_after": 1,
    }
    requeued = harness.run(
        _request(tmp_path, site=harness.site, action="requeue-dead", confirm=True)
    )
    assert harness.calls[-2]["confirm"] is True
    assert "requeued 2 dead record(s)" in requeued.message
    assert "1 left dead" in requeued.message
    assert "dead 3 -> 1" in requeued.message
    assert [path.name for path in harness.evidence()] == [
        "list-20260910T090000Z.json",
        "requeue-dead-20260910T090000Z.json",
    ]


def test_a_held_lock_is_reported_with_its_holder_and_the_node_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.terminal_status = "FAILED"
    harness.execution.update(
        {
            "status": "FAILED",
            "error": (
                f"node agent {NODE}: CollectorOutboxRefused: collector outbox lock "
                "is held: another process still holds the collector outbox lock"
            ),
            "details": {
                "node_results": {},
                "lock_unavailable": True,
                "lock_path": "/var/lib/gpu-fault/outbox/kernel.ndjson.lock",
                "lock_holder": "recorded holder pid 4242 (collector), alive, since x",
            },
        }
    )

    result = harness.run(
        _request(tmp_path, site=harness.site, action="requeue-dead", confirm=True)
    )

    assert result.succeeded is False
    assert result.node_result is None
    assert "recorded holder pid 4242 (collector)" in result.message
    assert "retry shortly" in result.message
    assert "--force` on the node" in result.message
    assert "there is no remote --force" in result.message
    archived = json.loads(harness.evidence()[0].read_text(encoding="utf-8"))
    assert archived["status"] == "FAILED"
    assert archived["step_details"]["lock_unavailable"] is True


def test_an_agent_that_rejected_the_command_reads_as_predating_the_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heartbeat gate is the first line; if a stale record let the command
    through, the transport's recorded HTTP 422 (the agent could not parse the
    operation) is read the same way -- no raw traceback, the node-local CLI."""

    harness = Harness(tmp_path, monkeypatch)
    harness.terminal_status = "FAILED"
    harness.execution.update(
        {
            "status": "FAILED",
            "error": f"node agent {NODE} rejected request: HTTP 422: [...]",
            "details": {
                "node_action_error_code": "HTTP_REJECTION",
                "http_status": 422,
                "node_action_retryable": False,
            },
        }
    )

    result = harness.run(_request(tmp_path, site=harness.site))

    assert result.succeeded is False
    assert "predates outbox maintenance" in result.message
    assert "HTTP 422" in result.message
    assert "gpu-fault-collector outbox --collector kernel stats" in result.message
    assert module.agent_predates_operation({"http_status": 403}) is True
    assert module.agent_predates_operation({"http_status": 500}) is False
    assert module.agent_predates_operation({"lock_unavailable": True}) is False


def test_a_workflow_that_does_not_end_in_time_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.polls_until_terminal = 10**6

    result = harness.run(_request(tmp_path, site=harness.site, wait_seconds=5))

    assert result.succeeded is False
    assert result.workflow_status == "RUNNING"
    assert "still RUNNING after 5s" in result.error
    assert "workflow-collector-outbox-1" in result.message
    assert sum(harness.slept) >= 5
    assert [path.name for path in harness.evidence()] == ["stats-20260910T090000Z.json"]


def test_a_control_plane_refusal_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    def refuse(site, payload, *, script):
        raise BootstrapError("collector-outbox: node x has no agent heartbeat")

    monkeypatch.setattr(module, "run_control_plane_script", refuse)

    with pytest.raises(BootstrapError, match="no agent heartbeat"):
        harness.run(_request(tmp_path, site=harness.site))
    assert harness.evidence() == []


# --------------------------------------------------------------------------
# CLI


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "collector-outbox",
        "--state-dir",
        str(tmp_path),
        "--cluster-id",
        "gpu-a",
        "--node",
        NODE,
        "--collector",
        "kernel",
        *extra,
    ]


def test_run_collector_outbox_command_builds_the_request_and_returns_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site = _site(tmp_path)
    captured: list[module.CollectorOutboxRequest] = []

    def fake_run(request):
        captured.append(request)
        return module.CollectorOutboxResult(
            request=request,
            operator=TEST_OPERATOR_ARN,
            incident_id="inc-x",
            workflow_request_id="wf-x",
            workflow_status="SUCCEEDED" if request.action == "list" else "FAILED",
            incident_state="RECOVERED",
            node_result={"records": []},
            message="done",
        )

    monkeypatch.setattr(module, "run_collector_outbox", fake_run)
    arguments = cli.parser().parse_args(
        _argv(
            tmp_path,
            "--action",
            "list",
            "--path",
            "/v1/kernel-logs",
            "--reference",
            "CHG-3",
            "--wait-seconds",
            "42",
        )
    )

    assert module.run_collector_outbox_command(arguments, site=site) == 0
    request = captured[0]
    assert (request.cluster_id, request.node_id, request.collector) == (
        "gpu-a",
        NODE,
        "kernel",
    )
    assert (request.action, request.path, request.reference) == (
        "list",
        "/v1/kernel-logs",
        "CHG-3",
    )
    assert request.confirm is False and request.wait_seconds == 42
    printed = json.loads(capsys.readouterr().out)
    assert printed["workflow_request_id"] == "wf-x"
    assert printed["result"] == {"records": []}

    failed = cli.parser().parse_args(
        _argv(tmp_path, "--action", "requeue-dead", "--yes", "--reference", "CHG-3")
    )
    assert module.run_collector_outbox_command(failed, site=site) == 1, (
        "a FAILED workflow is a non-zero exit"
    )
    assert captured[1].confirm is True


def test_cli_rejects_bad_vocabulary_requires_reference_and_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            _argv(tmp_path, "--action", "purge", "--reference", "CHG-1")
        )
    with pytest.raises(SystemExit):
        cli.parser().parse_args(_argv(tmp_path, "--action", "stats"))
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [
                *_argv(tmp_path, "--action", "stats", "--reference", "CHG-1")[:-6],
                "--collector",
                "sqs-hma",
                "--action",
                "stats",
                "--reference",
                "CHG-1",
            ]
        )
    seen: list[tuple] = []
    monkeypatch.setattr(cli, "load_site", lambda path, *, repository_root: site)
    monkeypatch.setattr(
        cli,
        "run_collector_outbox_command",
        lambda arguments, *, site: seen.append((arguments.action, site)) or 0,
    )
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--action", "stats", "--reference", "CHG-1")
    )

    assert cli.run(arguments) == 0
    assert seen == [("stats", site)]


def test_requeue_dead_without_yes_is_refused_by_the_verb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    monkeypatch.setattr(
        module, "run_collector_outbox", lambda request: pytest.fail("must not run")
    )
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--action", "requeue-dead", "--reference", "CHG-1")
    )

    with pytest.raises(BootstrapError, match="pass --yes"):
        module.run_collector_outbox_command(arguments, site=site)


def test_the_verb_is_on_the_public_table() -> None:
    choices = cli.parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    assert "collector-outbox" in choices
