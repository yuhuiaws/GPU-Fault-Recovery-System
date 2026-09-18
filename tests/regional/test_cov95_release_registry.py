from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_online_registry as online
from gpu_fault_release import regional_release_registry as registry
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import ResourceRelease
from tests.regional.test_release_registry_safety import (
    RegistrySecret,
    cluster_target,
    registry_release,
)


def test_absent_registry_is_empty_only_after_successful_resource_probe() -> None:
    secret = RegistrySecret()
    release = registry_release(secret)
    assert registry.registry_payloads(release) == ([], None)
    assert registry.registry(release) == []
    assert secret.applies == 0


@pytest.mark.parametrize("payload", [{}, [None], "invalid"])
def test_registry_payload_requires_list_of_objects(payload: Any) -> None:
    secret = RegistrySecret()
    secret.data = {
        "clusters.json": base64.b64encode(json.dumps(payload).encode()).decode()
    }
    with pytest.raises(ReleaseError, match="clusters.json is invalid"):
        registry.registry_payloads(registry_release(secret))
    assert secret.applies == 0


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("token", "requires token_file"),
        ("cidrs", "CIDRs are invalid"),
        ("short", "token is too short"),
    ],
)
def test_registry_entry_requires_explicit_valid_local_identity(
    tmp_path: Path, fault: str, problem: str
) -> None:
    target = cluster_target(tmp_path, cidrs=("192.0.2.0/24",))
    if fault == "token":
        target = replace(target, token_file=None)
    elif fault == "cidrs":
        target = replace(target, agent_endpoint_allowed_cidrs=("invalid",))
    else:
        Path(target.token_file).write_text("example-short")
    with pytest.raises(ReleaseError, match=problem):
        registry.registry_entry(target)


def test_registry_write_readback_failure_cannot_report_persisted_state() -> None:
    class LostWrite(RegistrySecret):
        def run(self, arguments: list[str], **kwargs: Any) -> str:
            if "apply" in arguments:
                self.applies += 1
                return ""
            return super().run(arguments, **kwargs)

    secret = LostWrite({"clusters.json": [{"cluster_id": "original"}]})
    release = registry_release(secret)
    with pytest.raises(ReleaseError, match="write did not persist"):
        registry.write_registry(release, [{"cluster_id": "candidate"}])
    assert secret.payload("clusters.json") == [{"cluster_id": "original"}]
    assert secret.applies == 1
    secret.dry_run = True
    registry.write_registry(release, [{"cluster_id": "candidate"}])
    assert secret.payload("clusters.json") == [{"cluster_id": "original"}]


def test_registry_initialization_refuses_unfinished_backup_and_preserves_equal_state() -> (
    None
):
    writes = []
    release = ResourceRelease()
    with pytest.raises(ReleaseError, match="unfinished release backup"):
        registry.initialize_registry(
            release,
            load=lambda _release: ([], []),
            desired=lambda _release: [],
            write=lambda *_args: writes.append(True),
        )
    registry.initialize_registry(
        release,
        load=lambda _release: ([], None),
        desired=lambda _release: [],
        write=lambda *_args: writes.append(True),
    )
    assert writes == []
    secret = RegistrySecret({"clusters.json": []})
    registry.commit_registry_update(registry_release(secret))
    assert secret.applies == 0


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("backup", "unfinished release backup"),
        ("version", "no resourceVersion"),
        ("conflicts", "changed repeatedly"),
    ],
)
def test_registry_update_refuses_unfinished_backup_missing_cas_and_repeated_conflicts(
    tmp_path: Path, fault: str, problem: str
) -> None:
    target = cluster_target(tmp_path, cidrs=("192.0.2.0/24",))
    release = ResourceRelease()
    document = {
        "metadata": {"resourceVersion": "1"},
        "data": {"clusters.json": base64.b64encode(b"[]").decode()},
    }
    if fault == "backup":
        document["data"]["previous-clusters.json"] = base64.b64encode(b"[]").decode()
    elif fault == "version":
        document["metadata"].clear()
    release.documents[("cpu", "secret", registry.REGISTRY_SECRET)] = document

    def fail(_arguments: list[str], _kwargs: dict[str, Any]) -> str:
        raise ReleaseError("resourceVersion conflict")

    release.runner.handler = fail
    before = copy.deepcopy(document)
    with pytest.raises(ReleaseError, match=problem):
        registry.update_registry(release, target, remove=False)
    assert document == before
    assert len(release.runner.calls) == (8 if fault == "conflicts" else 0)
    assert all(kwargs["sensitive"] for _arguments, kwargs in release.runner.calls), (
        "registry CAS retries must retain private command handling"
    )


def test_removing_an_already_absent_registration_issues_no_mutation(
    tmp_path: Path,
) -> None:
    release = ResourceRelease()
    release.documents[("cpu", "secret", registry.REGISTRY_SECRET)] = {"data": {}}
    target = cluster_target(tmp_path, cidrs=("192.0.2.0/24",))
    registry.update_registry(release, target, remove=True)
    assert release.runner.calls == []


def test_current_registrations_hashes_tokens_and_keeps_existing_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "example-placeholder-" + "x" * 32
    source = [
        {"cluster_id": "gpu-b", "token": token},
        {"cluster_id": "gpu-a", "token_sha256": "a" * 64},
    ]
    monkeypatch.setattr(online, "registry", lambda _release: copy.deepcopy(source))
    result = online.current_registrations(ResourceRelease(), {"gpu-b": "DRAINING"})
    assert [item["cluster_id"] for item in result] == ["gpu-a", "gpu-b"]
    assert all("token" not in item for item in result), (
        "online registry payloads must contain digests, never token values"
    )
    assert result[1]["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert result[1]["lifecycle_state"] == "DRAINING"
    source[0].pop("token")
    with pytest.raises(ReleaseError, match="no token digest source"):
        online.current_registrations(ResourceRelease(), {})


@pytest.mark.parametrize(
    "registrations", [[], [{"cluster_id": "gpu-a", "token_sha256": "a" * 64}] * 2]
)
def test_join_requires_a_unique_registration_before_publishing(
    monkeypatch: pytest.MonkeyPatch, registrations: list[Any]
) -> None:
    monkeypatch.setattr(online, "registry", lambda _release: registrations)
    release = ResourceRelease()
    with pytest.raises(ReleaseError, match="no unique cluster"):
        online.prepare_join_registry(release, "gpu-a")
    assert release.runner.calls == []


@pytest.mark.parametrize(
    "operation,lifecycle",
    [
        (online.prepare_join_registry, "PENDING"),
        (online.activate_join_registry, "ACTIVE"),
        (online.fail_join_registry, "FAILED"),
        (online.rollback_join_registry, "ROLLED_BACK"),
    ],
)
def test_join_lifecycle_wrappers_publish_bound_registration_and_explicit_state(
    monkeypatch: pytest.MonkeyPatch, operation: Any, lifecycle: str
) -> None:
    monkeypatch.setattr(
        online,
        "registry",
        lambda _release: [{"cluster_id": "gpu-a", "token_sha256": "a" * 64}],
    )
    received = []

    def publish(_release: Any, **kwargs: Any) -> str:
        received.append((json.loads(kwargs["input_text"]), kwargs))
        return '{"generation":2,"converged":true}'

    monkeypatch.setattr(online, "exec_cpu_ingress_command", publish)
    operation(ResourceRelease(), "gpu-a")
    request, kwargs = received[0]
    assert request["path"] == "/v1/regional/registry/clusters/gpu-a/transition"
    assert request["payload"]["registration"]["cluster_id"] == "gpu-a"
    assert request["payload"]["lifecycle_state"] == lifecycle
    assert request["use_current_generation"] is False
    assert kwargs["sensitive"] is True
    assert kwargs["timeout_seconds"] == request["timeout_seconds"] + 60


def test_registry_revision_rejects_nonobject_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        online, "exec_cpu_ingress_command", lambda *_args, **_kwargs: "[]"
    )
    with pytest.raises(ReleaseError, match="client returned a non-object"):
        online.publish_registry_revision(
            ResourceRelease(),
            path="/v1/example",
            payload={},
            use_current_generation=True,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    "operation", [online.publish_staged_registry, online.publish_restored_registry]
)
def test_staged_or_restored_registry_dry_run_has_no_transport(operation: Any) -> None:
    release = ResourceRelease()
    release.runner.dry_run = True
    assert operation(release) == {}
    assert release.runner.calls == []


class DrainRelease(ResourceRelease):
    def __init__(self) -> None:
        super().__init__()
        self.idle = False

    def _target(self, cluster_id: str) -> Any:
        if cluster_id != "gpu-a":
            raise ReleaseError("unknown cluster")
        return self.config.clusters[0]

    def _remote_commands_are_idle(self) -> bool:
        return self.idle


def test_drain_requires_idle_queue_and_other_lifecycle_states_use_scoped_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = DrainRelease()
    calls = []
    monkeypatch.setattr(
        online,
        "publish_current_registry",
        lambda _release, **kwargs: calls.append(kwargs) or {},
    )
    with pytest.raises(ReleaseError, match="PENDING/LEASED/WAITING"):
        online.drain_registry_cluster(release, "gpu-a")
    assert calls == []
    release.idle = True
    online.drain_registry_cluster(release, "gpu-a")
    online.revoke_registry_cluster(release, "gpu-a")
    online.purge_registry_cluster(release, "gpu-a")
    assert calls == [
        {
            "reason": "remove gpu-a draining",
            "lifecycle_overrides": {"gpu-a": "DRAINING"},
        },
        {"reason": "remove gpu-a revoked", "lifecycle_overrides": {"gpu-a": "REVOKED"}},
        {"reason": "remove gpu-a purged"},
    ]
