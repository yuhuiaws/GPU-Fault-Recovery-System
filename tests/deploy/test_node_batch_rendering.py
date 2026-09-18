from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.execution import current_deadline, deadline_scope
from gpu_fault.node_installer_rendering import InstallerIdentity
from gpu_fault_release import regional_node_batch as batch
from tests.deploy.test_node_preflight_inputs import inputs, render


@pytest.fixture
def rendered_batch(tmp_path: Path, request: pytest.FixtureRequest):
    document = inputs()
    for name, instance, address in (
        ("node-b", "ml.p5.4xlarge", "10.0.0.2"),
        ("node-c", "ml.p6-b200.48xlarge", "10.0.0.3"),
    ):
        node = copy.deepcopy(document["nodes"]["node-a"])
        node["metadata"]["uid"] = f"uid-{name}"
        node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = instance
        node["status"]["addresses"][0]["address"] = address
        document["nodes"][name] = node
    for index in range(3, getattr(request, "param", 3)):
        node = copy.deepcopy(document["nodes"]["node-a"])
        node["metadata"]["uid"] = f"uid-{index}"
        node["status"]["addresses"][0]["address"] = f"10.0.1.{index}"
        document["nodes"][f"node-{index:03d}"] = node
    result = render(tmp_path, document)
    assert result.returncode == 0, result.stderr
    template = tmp_path / "template.yaml"
    template.write_text(result.stdout)
    template.chmod(0o600)
    output = tmp_path / "batch"
    output.mkdir(mode=0o700)
    scope = batch.NodeScope(
        "gpu-test",
        "gpu-context",
        "gpu-fault-system",
        "gpu-fault-node-action-keys",
        "gpu-fault-regional-connection",
    )
    identity = InstallerIdentity(
        scope.namespace,
        "b" * 64,
        "a" * 64,
        "c" * 64,
        "d" * 64,
        scope.node_action_keys_secret,
        840,
        "http://{node_ip}:9400/metrics",
        template_content_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
    )
    return tmp_path / "nodes.json", template, output, scope, identity


def environment(document):
    container = document["spec"]["template"]["spec"]["containers"][0]
    return {item["name"]: item.get("value") for item in container["env"]}


def test_batch_reads_one_snapshot_and_preserves_each_nodes_identity(
    rendered_batch, monkeypatch
):
    source, template, output, scope, identity = rendered_batch
    reads = []
    read = batch.read_private_snapshot

    def tracked(path, binding):
        reads.append(path)
        return read(path, binding)

    monkeypatch.setattr(batch, "read_private_snapshot", tracked)
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    assert reads == [source]
    assert len(nodes) == 3
    assert len({node.job_name for node in nodes}) == 3
    for node, expected_gpu, expected_efa in zip(
        nodes, (8, 1, 8), (32, 1, 8), strict=True
    ):
        normal = json.loads(node.install_manifest.read_text())
        preflight = json.loads(node.preflight_manifest.read_text())
        values = environment(normal)
        assert values["TARGET_NODE_NAME"] == node.node.name
        assert values["TARGET_NODE_UID"] == node.node.uid
        assert values["TARGET_NODE_IP"] == node.node.address
        assert values["EXPECTED_GPU_COUNT"] == str(expected_gpu)
        assert values["EXPECTED_EFA_DEVICE_COUNT"] == str(expected_efa)
        assert environment(preflight)["PREFLIGHT_ONLY"] == "true"
        for document, read_only in ((normal, False), (preflight, True)):
            spec = document["spec"]["template"]["spec"]
            assert spec["nodeName"] == node.node.name
            secret = next(
                item for item in spec["volumes"] if item["name"] == "node-secret"
            )
            assert secret["secret"]["items"] == [
                {"key": node.node.name, "path": "node-action-secret"}
            ]
            root = next(
                item
                for item in spec["containers"][0]["volumeMounts"]
                if item["name"] == "host-root"
            )
            assert root["readOnly"] is read_only
    assert (
        environment(yaml.safe_load(template.read_text()))["TARGET_NODE_NAME"]
        == "node-a"
    )


@pytest.mark.parametrize(
    "field", ["cluster_id", "namespace", "context", "connection_secret"]
)
def test_batch_rejects_cross_scope_inputs_before_emitting_jobs(rendered_batch, field):
    source, template, output, scope, identity = rendered_batch
    document = json.loads(source.read_text())
    document[field] = "other"
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="scope"):
        batch.prepare_node_batch(source, template, output, scope, identity)
    assert list(output.iterdir()) == []


def test_batch_rejects_a_public_snapshot(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    source.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        batch.prepare_node_batch(source, template, output, scope, identity)


@pytest.mark.parametrize("drift", ["missing-pin", "wrong-pin", "template-bytes"])
def test_batch_uses_the_shared_trusted_template_loader(rendered_batch, drift):
    source, template, output, scope, identity = rendered_batch
    if drift == "template-bytes":
        template.write_text(template.read_text() + "\n")
    else:
        identity = replace(
            identity,
            template_content_sha256=None if drift == "missing-pin" else "e" * 64,
        )

    with pytest.raises(RuntimeError, match="required|does not match"):
        batch.prepare_node_batch(source, template, output, scope, identity)

    assert list(output.iterdir()) == [], "untrusted template bytes emitted node Jobs"


@pytest.mark.parametrize("with_dependencies", [False, True])
def test_batch_cli_passes_the_node_agents_identity_fields(
    rendered_batch, monkeypatch, capsys, with_dependencies
):
    source, template, output, scope, identity = rendered_batch
    image = "registry.invalid/node-dependencies@sha256:" + "e" * 64
    expected = replace(
        identity,
        node_dependency_image=image if with_dependencies else "",
        node_wheelhouse_sha256="f" * 64 if with_dependencies else "",
        require_rollback_slot=with_dependencies,
    )
    values = {
        "inputs": str(source),
        "template": str(template),
        "output": str(output),
        "cluster-id": scope.cluster_id,
        "context": scope.context,
        "namespace": scope.namespace,
        "node-action-keys-secret": scope.node_action_keys_secret,
        "connection-secret": scope.connection_secret,
        "config-digest": identity.config_digest,
        "artifact-sha256": identity.artifact_sha256,
        "bundle-sha256": identity.bundle_sha256,
        "template-sha256": identity.template_sha256,
        "template-content-sha256": identity.template_content_sha256,
        "metrics-url-template": identity.metrics_url_template,
        "deadline-seconds": str(identity.deadline_seconds),
    }
    if with_dependencies:
        values.update(
            {
                "node-dependency-image": image,
                "node-wheelhouse-sha256": "f" * 64,
                "require-rollback-slot": "true",
            }
        )
    argv = ["node-batch"] + [
        argument for name, value in values.items() for argument in (f"--{name}", value)
    ]
    observed = []

    def prepare(inputs, template_path, directory, binding, installer):
        assert (inputs, template_path, directory, binding) == (
            source,
            template,
            output,
            scope,
        )
        observed.append(installer)
        return []

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(batch, "prepare_node_batch", prepare)

    batch.main()

    assert observed == [expected], "batch CLI dropped a node identity field"
    assert json.loads(capsys.readouterr().out) == {
        "node_count": 0,
        "status": "RENDERED",
    }


class Runner:
    def __init__(
        self,
        *,
        fail_dry_run=False,
        fail_dry_run_after=None,
        fail_wait=False,
        fail_create_ack=False,
        dry_run_hook=None,
        required_admissions=0,
    ):
        self.commands = []
        self.admission_batches = []
        self.admission_bytes = []
        self.admitted = []
        self.inflight_batches = 0
        self.maximum_batches = 0
        self.active = set()
        self.documents = {}
        self.maximum = 0
        self.fail_dry_run = fail_dry_run
        self.fail_dry_run_after = fail_dry_run_after
        self.fail_wait = fail_wait
        self.fail_create_ack = fail_create_ack
        self.dry_run_hook = dry_run_hook
        self.required_admissions = required_admissions
        self.lock = threading.Lock()

    def run(self, command, **kwargs):
        if "--dry-run=server" in command:
            assert command[-2:] == ["-f", "-"]
            payload = kwargs["input_text"]
            document = json.loads(payload)
            assert document["apiVersion"] == "v1"
            assert document["kind"] == "List"
            items = document["items"]
            assert items and all(item["kind"] == "Job" for item in items)
            with self.lock:
                self.commands.append(command)
                self.admission_batches.append(items)
                self.admission_bytes.append(len(payload.encode("utf-8")))
                self.inflight_batches += 1
                self.maximum_batches = max(self.maximum_batches, self.inflight_batches)
            try:
                if self.dry_run_hook:
                    self.dry_run_hook(items)
                with self.lock:
                    for item in items:
                        if self.fail_dry_run or self.fail_dry_run_after == len(
                            self.admitted
                        ):
                            raise RuntimeError("admission refused")
                        self.admitted.append(item)
            finally:
                with self.lock:
                    self.inflight_batches -= 1
            return ""
        with self.lock:
            self.commands.append(command)
            if "create" in command:
                assert self.inflight_batches == 0, (
                    "a host Job started while a dry-run response was in flight"
                )
                assert len(self.admitted) >= self.required_admissions, (
                    "a host Job was created before all objects passed admission"
                )
                path = Path(command[command.index("-f") + 1])
                document = json.loads(path.read_text())
                assert (
                    document["metadata"]["labels"]["gpu-fault.io/node-preflight"]
                    == "true"
                )
                name = document["metadata"]["name"]
                document["metadata"]["uid"] = f"job-{name}"
                self.documents[name] = document
                self.active.add(name)
                self.maximum = max(self.maximum, len(self.active))
                if self.fail_create_ack:
                    raise RuntimeError("create acknowledgement lost")
            if "delete" in command:
                name = command[command.index("--raw") + 1].rsplit("/", 1)[1]
                options = json.loads(kwargs["input_text"])
                assert (
                    options["preconditions"]["uid"]
                    == self.documents[name]["metadata"]["uid"]
                )
                self.active.remove(name)
        if command[0] == "bash":
            time.sleep(0.01)
            if self.fail_wait:
                raise RuntimeError("host preflight failed")
        return ""

    def probe_output(self, arguments, **_kwargs):
        name = arguments[arguments.index("get") + 2]
        with self.lock:
            self.commands.append(arguments)
            return (
                0,
                json.dumps(self.documents[name]) if name in self.active else "",
                "",
            )


def test_every_dry_run_finishes_before_bounded_host_jobs_start(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(required_admissions=2 * len(nodes))
    batch.run_node_preflights(nodes, scope, identity, runner, workers=2)
    first_create = next(
        i for i, command in enumerate(runner.commands) if "create" in command
    )
    assert first_create == 1
    expected = [
        json.loads(path.read_text())
        for node in nodes
        for path in (node.install_manifest, node.preflight_manifest)
    ]
    assert {item["metadata"]["name"]: item for item in runner.admitted} == {
        item["metadata"]["name"]: item for item in expected
    }
    assert [len(items) for items in runner.admission_batches] == [6]
    assert runner.maximum <= 2
    assert runner.active == set()
    assert sum("create" in command for command in runner.commands) == len(nodes)


@pytest.mark.parametrize("dry_run", [False, True])
def test_failed_preflight_does_not_leave_its_started_jobs(rendered_batch, dry_run):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(fail_dry_run=dry_run, fail_wait=not dry_run)
    with pytest.raises(RuntimeError, match="refused|failed"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=1)
    assert runner.active == set()
    creates = sum("create" in command for command in runner.commands)
    assert creates == (0 if dry_run else 1)


def test_lost_create_ack_is_compensated_with_a_uid_precondition(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(fail_create_ack=True)
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=1)
    assert runner.active == set()
    assert sum("--raw" in command for command in runner.commands) == 1


def test_duplicate_node_uid_is_refused_before_any_rendered_job(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    document = json.loads(source.read_text())
    document["nodes"]["node-b"]["metadata"]["uid"] = document["nodes"]["node-a"][
        "metadata"
    ]["uid"]
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate node UIDs"):
        batch.prepare_node_batch(source, template, output, scope, identity)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("rendered_batch", [17], indirect=True)
@pytest.mark.parametrize("workers", [1, 2, 4, 5, 8])
def test_admission_batches_reduce_commands_without_losing_objects(
    rendered_batch, workers
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(required_admissions=2 * len(nodes))

    batch.run_node_preflights(nodes, scope, identity, runner, workers=workers)

    node_limit = batch.MAX_ADMISSION_BATCH_NODES
    assert len(runner.admission_batches) == math.ceil(len(nodes) / node_limit)
    assert all(
        len(items) <= 2 * node_limit and len(items) % 2 == 0
        for items in runner.admission_batches
    ), "admission List lost a node pair or exceeded its node limit"
    expected = [
        json.loads(path.read_text())
        for node in nodes
        for path in (node.install_manifest, node.preflight_manifest)
    ]
    assert len(runner.admitted) == len(expected)
    assert {item["metadata"]["name"]: item for item in runner.admitted} == {
        item["metadata"]["name"]: item for item in expected
    }
    assert all(
        size <= batch.MAX_ADMISSION_BATCH_BYTES for size in runner.admission_bytes
    ), "admission List exceeded its byte budget"
    assert runner.maximum <= workers
    assert runner.maximum_batches <= 8
    assert len(runner.documents) == len(nodes)
    assert not runner.active, "node preflight jobs remained after cleanup"
    first_create = next(
        index for index, command in enumerate(runner.commands) if "create" in command
    )
    assert first_create == len(runner.admission_batches)


@pytest.mark.parametrize("rendered_batch", [32], indirect=True)
@pytest.mark.parametrize("accepted", [1, 2, 5])
def test_partial_list_admission_failure_never_creates_host_jobs(
    rendered_batch, accepted
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(fail_dry_run_after=accepted)

    with pytest.raises(RuntimeError, match="admission refused") as raised:
        batch.run_node_preflights(nodes, scope, identity, runner, workers=4)

    assert len(runner.admitted) == accepted
    assert 1 <= len(runner.admission_batches) <= 8
    assert all("--dry-run=server" in command for command in runner.commands), (
        "host mutation ran before admission completed"
    )
    assert not runner.documents, "failed admission created a host preflight Job"
    assert not runner.active, "failed admission left an active host preflight"
    labels = [
        ", ".join(item["spec"]["template"]["spec"]["nodeName"] for item in items[::2])
        for items in runner.admission_batches
    ]
    assert any(
        label in note
        for label in labels
        for note in getattr(raised.value, "__notes__", [])
    ), "batch failure did not identify affected nodes"


@pytest.mark.parametrize("rendered_batch", [64], indirect=True)
def test_later_list_failure_stops_remaining_admission_batches(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner(fail_dry_run_after=9)

    with pytest.raises(RuntimeError, match="admission refused"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=4)

    assert 2 <= len(runner.admission_batches) < math.ceil(len(nodes) / 4)
    assert runner.maximum_batches <= 8
    assert len(runner.admitted) == 9
    assert all("--dry-run=server" in command for command in runner.commands), (
        "host mutation ran after a batch admission failure"
    )
    assert not runner.documents, "later admission failure created a host Job"


@pytest.mark.parametrize("rendered_batch", [32], indirect=True)
@pytest.mark.parametrize("fail_first", [False, True])
def test_admission_lists_overlap_and_drain_before_host_jobs_or_failure(
    rendered_batch, fail_first
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    all_started = threading.Barrier(8, timeout=5)
    first_done = threading.Event()
    drain = threading.Event()
    blocked_finished = threading.Event()

    def admission(items):
        name = items[0]["spec"]["template"]["spec"]["nodeName"]
        all_started.wait()
        if name == nodes[0].node.name:
            first_done.set()
            if fail_first:
                raise RuntimeError("List admission refused")
            return
        assert name in {node.node.name for node in nodes[4::4]}, (
            "an admission batch used the wrong node scope"
        )
        try:
            assert drain.wait(5), "test did not release the blocked admission batch"
        finally:
            blocked_finished.set()

    runner = Runner(dry_run_hook=admission, required_admissions=2 * len(nodes))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            batch.run_node_preflights, nodes, scope, identity, runner, workers=8
        )
        try:
            assert first_done.wait(5), "first admission batch never finished"
            assert not blocked_finished.is_set(), (
                "blocked admission unexpectedly finished"
            )
            assert not future.done(), "preflight returned while admission was in flight"
            assert not runner.documents, (
                "host preflight started before the admission barrier"
            )
            assert sum(len(items) // 2 for items in runner.admission_batches) == 32
            assert runner.maximum_batches == 8
        finally:
            drain.set()
        if fail_first:
            with pytest.raises(RuntimeError, match="List admission refused"):
                future.result(timeout=5)
        else:
            future.result(timeout=5)

    assert blocked_finished.is_set(), (
        "preflight did not drain the in-flight admission batch"
    )
    assert len(runner.admission_batches) == 8
    assert runner.inflight_batches == 0
    if fail_first:
        assert all("--dry-run=server" in command for command in runner.commands), (
            "failed admission reached a host mutation"
        )
        assert not runner.documents, "failed admission created a host preflight Job"
    else:
        assert len(runner.documents) == len(nodes)
        assert not runner.active, "successful preflight left an active host Job"


@pytest.mark.parametrize("rendered_batch", [32], indirect=True)
def test_admission_batches_split_at_the_byte_limit(rendered_batch, monkeypatch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    pair_sizes = [
        len(
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": [
                        json.loads(path.read_text())
                        for path in (node.install_manifest, node.preflight_manifest)
                    ],
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
        for node in nodes
    ]
    limit = max(pair_sizes)
    monkeypatch.setattr(batch, "MAX_ADMISSION_BATCH_BYTES", limit)
    runner = Runner()

    batch.run_node_preflights(nodes, scope, identity, runner, workers=8)

    assert [len(items) for items in runner.admission_batches] == [2] * len(nodes)
    assert all(size <= limit for size in runner.admission_bytes), (
        "byte-limited batches exceeded their cap"
    )
    assert len(runner.admitted) == 2 * len(nodes)
    assert not runner.active, "byte-limited preflight left an active host Job"


def test_oversized_manifest_fails_before_any_admission_or_host_create(
    rendered_batch, monkeypatch
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    monkeypatch.setattr(batch, "MAX_ADMISSION_BATCH_BYTES", 100)
    runner = Runner()

    with pytest.raises(ValueError, match="size budget"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=8)

    assert not runner.commands, "oversized manifest reached Kubernetes admission"


def test_oversized_node_pair_fails_closed(rendered_batch, monkeypatch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    limit = max(
        path.stat().st_size
        for node in nodes
        for path in (node.install_manifest, node.preflight_manifest)
    )
    monkeypatch.setattr(batch, "MAX_ADMISSION_BATCH_BYTES", limit)
    runner = Runner()

    with pytest.raises(ValueError, match="batch size budget"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=8)

    assert not runner.commands, "oversized node pair reached Kubernetes admission"


@pytest.mark.parametrize("workers", [0, -1, 9])
def test_invalid_concurrency_never_starts_admission(rendered_batch, workers):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    runner = Runner()

    with pytest.raises(ValueError, match="concurrency"):
        batch.run_node_preflights(nodes, scope, identity, runner, workers=workers)

    assert not runner.commands, "invalid concurrency reached Kubernetes admission"


def test_admission_and_host_workers_keep_the_callers_deadline(rendered_batch):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    observed = []

    class DeadlineRunner(Runner):
        def run(self, command, **kwargs):
            if "--dry-run=server" in command or "create" in command:
                observed.append(current_deadline())
            return super().run(command, **kwargs)

    with deadline_scope("node-preflight-test", 30) as deadline:
        batch.run_node_preflights(nodes, scope, identity, DeadlineRunner(), workers=2)

    assert len(observed) == 4
    assert observed == [deadline] * len(observed)


@pytest.mark.parametrize("rendered_batch", [64], indirect=True)
@pytest.mark.parametrize("timeout", [False, True])
def test_failed_admission_drains_a_bounded_window_and_never_starts_hosts(
    rendered_batch, timeout, monkeypatch
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    all_started = threading.Barrier(8, timeout=5)
    failure_observed = threading.Event()
    drain = threading.Event()
    failure = (
        subprocess.TimeoutExpired(["kubectl", "apply", "--dry-run=server"], 60)
        if timeout
        else RuntimeError("one List item was refused")
    )
    wait = batch.wait

    def observe_failure(*args, **kwargs):
        done, pending = wait(*args, **kwargs)
        if any(future.exception() is failure for future in done):
            failure_observed.set()
        return done, pending

    monkeypatch.setattr(batch, "wait", observe_failure)

    def admission(items):
        name = items[0]["spec"]["template"]["spec"]["nodeName"]
        all_started.wait()
        if name == nodes[0].node.name:
            raise failure
        assert drain.wait(5), "test failed to release the in-flight admission window"

    runner = Runner(dry_run_hook=admission, required_admissions=2 * len(nodes))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            batch.run_node_preflights, nodes, scope, identity, runner, workers=1
        )
        try:
            assert failure_observed.wait(5), "the scheduler never observed the failure"
            assert not future.done(), "the failed window was not drained"
            assert len(runner.admission_batches) == 8
            assert runner.maximum_batches == 8
            assert not runner.documents, "a failed admission started a host Job"
        finally:
            drain.set()
        with pytest.raises(type(failure)) as raised:
            future.result(timeout=5)

    assert raised.value is failure, "admission failure lost its original cause"
    assert runner.inflight_batches == 0
    assert len(runner.admission_batches) == 8, (
        "admission submitted another window after a failure"
    )
    assert all("--dry-run=server" in command for command in runner.commands), (
        "a failed or timed-out admission reached host preflight"
    )


@pytest.mark.parametrize("rendered_batch", [64], indirect=True)
def test_successful_admission_uses_eight_cli_workers_without_growing_the_host_window(
    rendered_batch,
):
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    arrivals = threading.Barrier(8, timeout=5)
    runner = Runner(
        dry_run_hook=lambda _items: arrivals.wait(), required_admissions=2 * len(nodes)
    )

    batch.run_node_preflights(nodes, scope, identity, runner, workers=1)

    assert runner.maximum_batches == 8, "admission was serialized by the host budget"
    assert len(runner.admission_batches) == 16
    assert runner.maximum == 1, "admission concurrency enlarged the host window"
    assert len(runner.admitted) == 2 * len(nodes)
    assert len(runner.documents) == len(nodes)
    assert not runner.active, "the single-host window left an active preflight Job"
