"""Contract tests for the CRI Pod-log fixture controller and the 017 managed job.

The controller is driven over a ``FakeProbe`` (what it sends the node probe and
how it cleans up); the reviewed lines are checked against the product's own log
rules; the managed job's modes are rendered onto the real manifest; the event
id the verdict recomputes is the product's.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.log_recovery_context import log_event_id as product_log_event_id
from gpu_fault.log_rules import matching_log_rules
from scripts.e2e.regional import general_live_applog_fixture as helper
from scripts.e2e.regional import general_live_applog_job as job_module
from scripts.e2e.regional import general_live_applogs as mod
from scripts.e2e.regional.general_live_fixture import GeneralLiveSettings
from tests.regional._general_live_fakes import FakeProbe, FakeRegional, make_context

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
UID = "0f3b7c2e-4d5a-4b6c-8d9e-0a1b2c3d4e5f"
MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "e2e"
    / "regional"
    / "manifests"
    / "training"
    / "single-node-warm-spare-pytorchjob.yaml"
)


# --------------------------------------------------------------------------- #
# reviewed records
# --------------------------------------------------------------------------- #
def test_reviewed_line_defaults_and_validation() -> None:
    assert helper.reviewed_line("x") == {
        "message": "x",
        "stream": "stdout",
        "flag": "F",
    }
    assert helper.reviewed_line("x", stream="stderr", flag="P")["flag"] == "P"
    for kwargs in ({"stream": "file"}, {"flag": "Q"}):
        with pytest.raises(ValueError):
            helper.reviewed_line("x", **kwargs)
    with pytest.raises(ValueError):
        helper.reviewed_line("two\nlines")
    first = helper.records_sha256([helper.reviewed_line("x")])
    assert first == helper.records_sha256([dict(helper.reviewed_line("x"))])
    assert len(first) == 64


def test_reviewed_lines_match_the_products_rules_as_designed() -> None:
    oom = matching_log_rules(helper.CUDA_OOM_LINE)
    assert [(rule.action.value, rule.severity.value.upper()) for rule in oom] == [
        ("RUN_DIAGNOSTICS", "WARNING")
    ], "the CUDA OOM line is the MEMORY diagnostic rule"
    nccl = matching_log_rules(helper.NCCL_ERROR_LINE)
    assert [rule.action.value for rule in nccl] == ["NO_ACTION"]
    assert matching_log_rules(helper.PARTIAL_HEAD) == [], (
        "the P fragment alone matches nothing"
    )
    assert matching_log_rules(helper.PARTIAL_TAIL) == [], (
        "the F fragment alone matches nothing"
    )
    whole = matching_log_rules(helper.PARTIAL_HEAD + helper.PARTIAL_TAIL)
    assert [(rule.category.value, rule.action.value) for rule in whole] == [
        ("TRAINING", "NO_ACTION")
    ], "only the stitched line is a distributed-training evidence record"


def test_log_event_id_is_the_products() -> None:
    batch = SimpleNamespace(cluster_id="gpu-eks-cluster", node_id=NODE)
    entry = SimpleNamespace(entry_id="a" * 64)
    expected = product_log_event_id(batch, entry, mod.safe_component)
    assert mod.log_event_id("gpu-eks-cluster", NODE, "a" * 64) == expected
    assert expected.startswith(
        "log-gpu-eks-cluster-ip-10-0-43-5.us-west-2.compute.internal-"
    ), "the event id names the cluster and node"


# --------------------------------------------------------------------------- #
# the controller over a FakeProbe
# --------------------------------------------------------------------------- #
def _controller(
    tmp_path: Path, script: dict[str, Any]
) -> tuple[helper.CriPodLogFixture, FakeProbe]:
    fake = FakeProbe(script)

    class _Regional:
        class settings:
            gpu_kubeconfig = tmp_path / "kubeconfig"
            gpu_context = "ctx"
            namespace = "gpu-fault-system"

    controller = helper.CriPodLogFixture(
        _Regional(),
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-017.live_pid",
        run_id="gl17-lp-a1-deadbeef",
        case_dir=tmp_path / "case",
        host=fake,
    )
    return controller, fake


def _call(fake: FakeProbe, name: str) -> tuple[str, ...]:
    return next(item for item in fake.calls if item[0] == name)


def test_open_passes_the_window_arguments(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path, {"open-window": {"after": {}}})
    controller.open(
        container="pytorch",
        approved=True,
        restore_seconds=1800,
        initial_tail_bytes=65536,
    )
    call = _call(fake, "open-window")
    assert "--approved" in call and call[call.index("--container") + 1] == "pytorch"
    assert call[call.index("--restore-seconds") + 1] == "1800"
    assert call[call.index("--initial-tail-bytes") + 1] == "65536"
    assert controller.container == "pytorch" and controller.evidence()["opened"] is True
    controller2, fake2 = _controller(tmp_path / "b", {"open-window": {}})
    controller2.open(container="pytorch")
    assert "--approved" not in _call(fake2, "open-window")


def test_write_sends_the_digest_and_the_records(tmp_path: Path) -> None:
    controller, fake = _controller(
        tmp_path, {"write": {"stamps": ["2026-10-25T10:00:00.000000000Z"]}}
    )
    records = [helper.reviewed_line("col ", flag="P")]
    controller.write(UID, container="pytorch", records=records, target="rotated")
    call = _call(fake, "write")
    assert call[call.index("--sha256") + 1] == helper.records_sha256(records)
    sent = json.loads(base64.b64decode(call[call.index("--b64") + 1]))
    assert sent == records and call[call.index("--target") + 1] == "rotated"
    assert controller.evidence()["writes"][0]["stamps"] == [
        "2026-10-25T10:00:00.000000000Z"
    ]
    with pytest.raises(ValueError, match="at least one"):
        controller.write(UID, container="pytorch", records=[])


def test_create_log_inspect_rotate_restart_and_status(tmp_path: Path) -> None:
    status = {
        "collector": {
            "tracked": {"/p": {"offset": 0}},
            "pending": {"/p": {"cri": {"stdout": {"message": "col "}}}},
        }
    }
    controller, fake = _controller(
        tmp_path,
        {
            "create-log": {"file": {"path": "/p"}},
            "inspect": {"process": {"init_pid": 42}},
            "status": status,
        },
    )
    controller.create_log(
        UID, container="pytorch", namespace="ns", pod_name="pod", allow_create_dir=True
    )
    call = _call(fake, "create-log")
    assert "--allow-create-dir" in call and call[call.index("--namespace") + 1] == "ns"
    controller.create_log(UID, container="pytorch", namespace="ns", pod_name="pod")
    assert "--allow-create-dir" not in fake.calls[-1]
    controller.inspect(UID, container_id="containerd://" + "ab" * 32)
    assert _call(fake, "inspect")[-1] == "ab" * 32, "the runtime prefix is stripped"
    controller.rotate(UID, container="pytorch")
    controller.restart_collector()
    assert [item[0] for item in fake.calls[-2:]] == ["rotate", "restart-collector"]
    view = controller.status()
    assert controller.tracked(view, "/p") == {"offset": 0}
    assert controller.tracked(view, "/other") is None
    assert controller.pending_fragment(view, "/p") == {"message": "col "}
    assert controller.pending_fragment(view, "/p", stream="stderr") is None


def test_cleanup_closes_an_open_window_and_fails_on_residue(tmp_path: Path) -> None:
    controller, fake = _controller(
        tmp_path, {"open-window": {}, "close-window": {"clean": True}}
    )
    controller.open(container="pytorch", approved=True)
    receipts = controller.cleanup()
    assert receipts["close"]["clean"] is True and fake.calls[-1][0] == "close-window"
    dirty, _ = _controller(
        tmp_path / "dirty", {"open-window": {}, "close-window": {"clean": False}}
    )
    dirty.open(container="pytorch", approved=True)
    with pytest.raises(RuntimeError, match="did not close clean"):
        dirty.cleanup()
    closed, fake3 = _controller(tmp_path / "closed", {})
    closed.cleanup()
    assert all(item[0] != "close-window" for item in fake3.calls), "nothing to close"


def test_cri_pod_logs_uses_the_registered_fake_and_pushes_cleanup(
    tmp_path: Path,
) -> None:
    spec = next(item for item in mod.VARIANTS if item.variant == "live_pid")
    fake = FakeProbe({"status": {"collector": {"tracked": {"/p": {}}}}})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(),
        probes={f"cri-pod-logs@{NODE}": fake},
    )
    logs = helper.cri_pod_logs(ctx)
    assert logs.host is fake and ctx.cleanup.pending() == ["cri-pod-logs"]
    view = logs.wait_tracked(ctx, "/p", name="tracked")
    assert logs.tracked(view, "/p") == {}


# --------------------------------------------------------------------------- #
# the managed job's modes on the real manifest
# --------------------------------------------------------------------------- #
def _document() -> dict[str, Any]:
    value = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict), "the manifest is a mapping"
    return value


def _container(document: dict[str, Any]) -> dict[str, Any]:
    replica = document["spec"]["pytorchReplicaSpecs"]["Master"]
    return next(
        item
        for item in replica["template"]["spec"]["containers"]
        if item["name"] == job_module.TRAINING_CONTAINER
    )


def test_trainer_mode_leaves_the_manifest_command_alone() -> None:
    document = _document()
    before = json.dumps(_container(document), sort_keys=True)
    job_module.apply_mode(document, "trainer", first_attempt="j-a001")
    assert json.dumps(_container(document), sort_keys=True) == before


def test_hold_mode_runs_a_python_init_that_exits_on_the_runners_file() -> None:
    document = _document()
    job_module.apply_mode(document, "hold", first_attempt="j-a001")
    command = _container(document)["command"]
    assert command[:2] == ["/bin/bash", "-lc"]
    assert command[2].startswith("exec python3 - <<'PY'"), "python is the init process"
    assert job_module.HOLD_EXIT_FILE in command[2]
    assert f"time.monotonic() + {job_module.HOLD_MAX_SECONDS}" in command[2]
    assert "sys.exit(0)" in command[2]
    body = command[2].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(body, "hold.py", "exec")  # the rendered heredoc is valid Python
    assert 'print("HOLD pid=%d" % os.getpid(), flush=True)' in body


def test_fail_once_mode_fails_only_the_first_attempt_by_its_label() -> None:
    document = _document()
    original = _container(document)["command"][2]
    job_module.apply_mode(document, "fail_once", first_attempt="j-a001")
    container = _container(document)
    script = container["command"][2]
    assert script.startswith(
        f'if [ "${{{job_module.ATTEMPT_ENV}}}" = "j-a001" ]; then sleep '
        f"{job_module.FAIL_ONCE_SECONDS}; exit 1; fi\n"
    ), "the first attempt fails after the hold"
    assert script.endswith(original), "the restarted attempt runs the real trainer"
    downward = [
        item for item in container["env"] if item["name"] == job_module.ATTEMPT_ENV
    ]
    assert downward == [
        {
            "name": job_module.ATTEMPT_ENV,
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": f"metadata.labels['{job_module.ATTEMPT_LABEL}']"
                }
            },
        }
    ]
    with pytest.raises(ValueError, match="mode must be one of"):
        job_module.apply_mode(_document(), "explode", first_attempt="j-a001")


def test_apply_mode_refuses_a_manifest_without_the_training_container() -> None:
    document = _document()
    _container(document)["name"] = "sidecar"
    with pytest.raises(ValueError, match="has no pytorch container"):
        job_module.apply_mode(document, "hold", first_attempt="j-a001")


def test_container_status_reads_identity_and_state() -> None:
    pod = {
        "status": {
            "containerStatuses": [
                {
                    "name": "pytorch",
                    "containerID": "containerd://" + "ab" * 32,
                    "state": {"terminated": {"exitCode": 0}},
                    "restartCount": 0,
                }
            ]
        }
    }
    status = job_module.container_status(pod, "pytorch")
    assert status["container_id"] == "ab" * 32 and status["running"] is False
    assert status["terminated"] == {"exitCode": 0}
    assert job_module.container_status({}, "pytorch")["container_id"] is None


def test_managed_job_needs_a_site_file(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    spec = next(item for item in mod.VARIANTS if item.variant == "live_pid")
    ctx = make_context(spec, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    with pytest.raises(VariantNotRun, match="site-file"):
        job_module.ManagedApplogJob(ctx, name="gl17-lp-a1-x", mode="trainer")


@pytest.fixture
def private_umask() -> Iterator[None]:
    """The managed fixtures' journals require the runner's 0o077 umask."""

    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.usefixtures("private_umask")
def test_managed_job_renders_mode_labels_and_identity(tmp_path: Path) -> None:
    spec = next(item for item in mod.VARIANTS if item.variant == "old_pod")
    base = make_context(spec, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    (tmp_path / "site.yaml").write_text("site: test\n")
    base.settings = GeneralLiveSettings(
        **{**base.settings.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    job = job_module.ManagedApplogJob(
        base, name="gl17-op-a1-deadbeef", mode="fail_once", restart_budget=1
    )
    rendered = yaml.safe_load(
        (base.case_dir / "gl17-op-a1-deadbeef.source.yaml").read_text()
    )
    labels = rendered["metadata"]["labels"]
    assert (labels["gpu-fault.io/job-id"], labels[job_module.ATTEMPT_LABEL]) == (
        "gl17-op-a1-deadbeef",
        "gl17-op-a1-deadbeef-a001",
    )
    assert rendered["metadata"]["name"] != "gpu-fault-single-node-warm-spare"
    assert _container(rendered)["command"][2].startswith('if [ "${GFRS_ATTEMPT_ID}"'), (
        "fail_once is rendered onto the trainer"
    )
    assert (
        rendered["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
            "nodeName"
        ]
        == NODE
    )
    assert job.identity() == {
        "job_id": "gl17-op-a1-deadbeef",
        "attempt_id": "gl17-op-a1-deadbeef-a001",
        "mode": "fail_once",
        "node": NODE,
        "namespace": "gpu-fault-system",
        "container": "pytorch",
        "restart_budget": 1,
    }
    assert (
        job.fixture.settings.restart_budget == 1
        and job.fixture.settings.expected_pods == 1
    )
    with pytest.raises(ValueError, match="only the hold mode"):
        job.request_exit("pod")
    with pytest.raises(ValueError, match="positive"):
        job.annotate_host_pid("pod", 0)
