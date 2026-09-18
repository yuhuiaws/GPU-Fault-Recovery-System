from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault import node_installer_reconciler as installer
from tests.node_agent._cov95_installer_support import (
    isolated_installer_io as isolated_installer_io,
)
from tests.node_agent._cov95_installer_support import make_reconciler, node_document
from tests.node_agent.test_node_installer_reconciler import (
    NOW,
    ApiError,
    BatchApi,
    CoreApi,
    job,
    node,
)


@pytest.mark.parametrize(
    "options",
    [
        {"artifact_sha256": "invalid"},
        {"bundle_sha256": "invalid"},
        {"max_unavailable": 0},
        {"job_active_deadline_seconds": 59},
    ],
)
def test_constructor_rejects_invalid_installation_contract(options) -> None:
    with pytest.raises(ValueError, match="SHA-256|positive|deadline"):
        make_reconciler(**options)


@pytest.mark.parametrize("status", [None, "False", "Unknown"])
def test_unknown_or_unready_node_is_deferred_without_a_job_read(status) -> None:
    current = node_document()
    current["status"]["conditions"] = [{"type": "Ready", "status": status}]
    core, batch = CoreApi([current]), BatchApi()
    result = make_reconciler(core, batch).reconcile_once()
    assert result["not_ready"] == 1
    assert batch.reads == 0
    assert batch.created == []
    assert core.patches == []


def test_json_node_binding_tracks_current_uid_boot_and_internal_address() -> None:
    current = node_document()
    core, batch = CoreApi([current]), BatchApi()
    assert make_reconciler(core, batch).reconcile_once()["created"] == 1
    created = batch.created[0][1]
    pod = created["spec"]["template"]["spec"]
    environment = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}
    assert environment["TARGET_NODE_UID"] == current["metadata"]["uid"]
    assert environment["TARGET_NODE_IP"] == "10.0.1.25"
    assert (
        core.patches[0][1]["metadata"]["annotations"][
            installer.INSTALLER_BOOT_ID_ANNOTATION
        ]
        == "boot-current"
    )
    assert pod["volumes"][0]["secret"]["items"] == [
        {"key": "hyperpod-i-123", "path": "node-action-secret"}
    ]


def test_missing_internal_address_never_creates_an_installer() -> None:
    current = node_document()
    current["status"]["addresses"] = [{"type": "ExternalIP", "address": "192.0.2.1"}]
    core, batch = CoreApi([current]), BatchApi()
    assert make_reconciler(core, batch).reconcile_once()["unsupported"] == 1
    assert batch.created == []
    assert core.patches == []


@pytest.mark.parametrize("status", [403, 409])
def test_create_error_keeps_budget_accounting_and_later_nodes_correct(status) -> None:
    class RacingBatch(BatchApi):
        def create_namespaced_job(self, namespace, body, _request_timeout=None):
            if body["spec"]["template"]["spec"]["nodeName"] == "hyperpod-a":
                raise ApiError(status)
            super().create_namespaced_job(
                namespace, body, _request_timeout=_request_timeout
            )

    core = CoreApi([node(name="hyperpod-a"), node(name="hyperpod-b")])
    batch = RacingBatch()
    result = make_reconciler(core, batch).reconcile_once()
    if status == 409:
        assert result["created"] == 1
        assert result["deferred"] == 1
        assert batch.created == []
        assert [name for name, _ in core.patches] == ["hyperpod-a"]
    else:
        assert result["error"] == 1
        assert result["created"] == 1
        assert [
            body["spec"]["template"]["spec"]["nodeName"] for _, body in batch.created
        ] == (["hyperpod-b"])


def test_job_read_error_is_not_interpreted_as_absence() -> None:
    class ForbiddenBatch(BatchApi):
        def read_namespaced_job(self, name, namespace, _request_timeout=None):
            raise ApiError(403)

    batch = ForbiddenBatch()
    result = make_reconciler(CoreApi([node()]), batch).reconcile_once()
    assert result["error"] == 1
    assert batch.created == []


@pytest.mark.parametrize("maximum", ["0", "-1"])
def test_wave_rejects_nonpositive_budget_before_node_listing(maximum) -> None:
    core = CoreApi([], wave_data={"allowed-nodes": "*", "max-unavailable": maximum})
    batch = BatchApi()
    active = make_reconciler(
        core, batch, wave_config_map="gpu-fault-node-installer-wave"
    )
    with pytest.raises(RuntimeError, match="must be positive"):
        active.reconcile_once()
    assert core.calls == [("read_namespaced_config_map", installer.REQUEST_TIMEOUT)]
    assert batch.created == []


@pytest.mark.parametrize("stamp", [None, "not-a-time"])
def test_failed_job_without_a_known_failure_time_never_retries(stamp) -> None:
    failed = {
        "metadata": {"creation_timestamp": stamp},
        "status": {
            "conditions": [
                {"type": "Ignored", "status": "True"},
                {"type": "Failed", "status": "True", "lastTransitionTime": stamp},
            ]
        },
    }
    current = node(annotations={installer.INSTALLER_ATTEMPTS_ANNOTATION: "invalid"})
    core, batch = CoreApi([current]), BatchApi(failed)
    assert make_reconciler(core, batch).reconcile_once()["failed"] == 1
    assert batch.deleted == []
    assert (
        core.patches[0][1]["metadata"]["annotations"][
            installer.INSTALLER_ATTEMPTS_ANNOTATION
        ]
        is None
    )


def test_json_failure_timestamp_controls_retry_and_removes_stale_reason() -> None:
    failed = {
        "status": {
            "conditions": [{"type": "Failed", "status": "True"}],
            "completionTime": (NOW - timedelta(seconds=301))
            .replace(tzinfo=None)
            .isoformat(),
        }
    }
    core = CoreApi([node(annotations={installer.INSTALLER_REASON_ANNOTATION: "old"})])
    batch = BatchApi(failed)
    assert make_reconciler(core, batch).reconcile_once()["failed"] == 1
    assert len(batch.deleted) == 1
    marked = core.patches[-1][1]["metadata"]["annotations"]
    assert marked[installer.INSTALLER_STATE_ANNOTATION] == "Retrying"
    assert marked[installer.INSTALLER_REASON_ANNOTATION] is None


@pytest.mark.parametrize(
    "status",
    [
        {},
        {"containerStatuses": [{"state": {"running": {}}}]},
        {"initContainerStatuses": [{"state": {"waiting": {}}}]},
        {
            "containerStatuses": [
                {"state": {"waiting": {"reason": "ContainerCreating"}}}
            ]
        },
    ],
)
def test_unproven_never_started_condition_keeps_job_running(status) -> None:
    core = CoreApi(
        [node()], pods={"hyperpod-i-123": {"metadata": {}, "status": status}}
    )
    batch = BatchApi(
        {"status": {"startTime": (NOW - timedelta(seconds=300)).isoformat()}}
    )
    assert make_reconciler(core, batch).reconcile_once()["running"] == 1
    assert batch.deleted == []
    assert core.patches == []


def test_never_started_init_container_uses_bounded_reason_and_backoff() -> None:
    core = CoreApi(
        [node()],
        pods={
            "hyperpod-i-123": {
                "status": {
                    "initContainerStatuses": [
                        {"state": {"waiting": {"reason": "ErrImagePull"}}}
                    ]
                }
            }
        },
    )
    batch = BatchApi({"status": {"start_time": NOW - timedelta(seconds=300)}})
    assert make_reconciler(core, batch).reconcile_once()["unsupported"] == 1
    assert len(batch.deleted) == 1
    marked = core.patches[0][1]["metadata"]["annotations"]
    assert marked[installer.INSTALLER_REASON_ANNOTATION] == "ErrImagePull"
    assert marked[installer.INSTALLER_ATTEMPTS_ANNOTATION] == "1"
    assert marked[installer.INSTALLER_RETRY_AFTER_ANNOTATION] == (
        NOW + timedelta(seconds=300)
    ).isoformat().replace("+00:00", "Z")


def test_running_job_without_any_start_time_is_not_guessed_stuck() -> None:
    core, batch = CoreApi([node()]), BatchApi({"status": {}})
    assert make_reconciler(core, batch).reconcile_once()["running"] == 1
    assert "list_namespaced_pod" not in [name for name, _ in core.calls]


def test_unreadable_installer_pods_preserve_active_job(caplog) -> None:
    class ForbiddenPods(CoreApi):
        def list_namespaced_pod(self, namespace, **kwargs):
            raise ApiError(403)

    batch = BatchApi(job(age_seconds=300))
    assert (
        make_reconciler(ForbiddenPods([node()]), batch).reconcile_once()["running"] == 1
    )
    assert batch.deleted == []
    assert "could not read the Pods" in caplog.text


def test_failed_heartbeat_write_does_not_discard_pass_result(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        installer, "RECONCILER_HEARTBEAT_PATH", str(tmp_path / "missing" / "heartbeat")
    )
    assert make_reconciler().reconcile_once()["error"] == 0
    assert "could not write reconciler heartbeat" in caplog.text


@pytest.mark.parametrize("available", [True, False])
def test_agent_socket_probe_uses_only_injected_connection(monkeypatch, available):
    calls = []

    def connect(address, *, timeout):
        calls.append((address, timeout))
        if not available:
            raise OSError("fake refusal")
        return nullcontext(SimpleNamespace())

    monkeypatch.setattr(installer.socket, "create_connection", connect)
    assert installer.agent_answers("192.0.2.1", 9099, timeout_seconds=1.5) is available
    assert calls == [(("192.0.2.1", 9099), 1.5)]
