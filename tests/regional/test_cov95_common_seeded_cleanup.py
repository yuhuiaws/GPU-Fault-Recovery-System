from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import seeded_command_fixture as seeded
from tests.regional._cov95_cap_cases import Clock
from tests.regional._cov95_common_seeded import RUN_ID, ResourceAPI, metadata, probe


@pytest.mark.parametrize(
    "kind,uri",
    [
        ("pod", "/api/v1/namespaces/selected/pods/unit"),
        ("clusterrole", "/apis/rbac.authorization.k8s.io/v1/clusterroles/unit"),
    ],
)
def test_delete_is_uid_version_and_namespace_scoped(kind: str, uri: str) -> None:
    api = ResourceAPI()
    api.resources[(kind, "unit")] = metadata(kind)
    seeded.delete_owned_resource(
        kind,
        "unit",
        RUN_ID,
        client=api,
        namespace="selected",
        expected_uid=f"uid-{kind}",
        require_uid=True,
    )
    assert api.resources == {}
    deletion = next(args for args, _kwargs in api.calls if args[0] == "delete")
    assert deletion == ("delete", "--raw", uri, "-f", "-")
    assert api.deletes == [
        {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": f"uid-{kind}", "resourceVersion": "1"},
            "propagationPolicy": "Foreground",
        }
    ]


@pytest.mark.parametrize(
    "fault", ["unsupported", "uid", "version", "label", "proof", "replacement"]
)
def test_delete_never_mutates_unbound_or_unsupported_resources(fault: str) -> None:
    api = ResourceAPI()
    item = metadata("pod")
    if fault == "uid":
        item.pop("uid")
    elif fault == "version":
        item.pop("resourceVersion")
    elif fault == "label":
        item["labels"][seeded.RUN_LABEL] = "foreign"
    api.resources[("pod", "unit")] = item
    with pytest.raises(seeded.SeededCommandError):
        seeded.delete_owned_resource(
            "node" if fault == "unsupported" else "pod",
            "unit",
            RUN_ID,
            client=api,
            require_uid=True,
            expected_uid=None
            if fault == "proof"
            else "other"
            if fault == "replacement"
            else "uid-pod",
        )
    assert api.deletes == []
    assert ("pod", "unit") in api.resources


@pytest.mark.parametrize("outcome", ["replaced", "stuck", "read-failed"])
def test_successful_delete_ack_is_not_a_confirmed_absence(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    clock = Clock()
    clock.sleep_scale = 120
    monkeypatch.setattr(seeded, "time", clock)
    item = metadata("pod")
    deletions = []

    def client(*args: str, **kwargs: Any) -> str:
        if args[0] == "delete":
            deletions.append(json.loads(kwargs["stdin"]))
            return ""
        assert args[0] == "get"
        if deletions:
            if outcome == "read-failed":
                raise TimeoutError("synthetic post-delete read failure")
            if outcome == "replaced":
                return json.dumps({**item, "uid": "replacement"})
        return json.dumps(item)

    with pytest.raises(
        (seeded.SeededCommandError, TimeoutError),
        match="replaced|not confirmed|read failure",
    ):
        seeded.delete_owned_resource(
            "pod",
            "unit",
            RUN_ID,
            client=client,
            expected_uid="uid-pod",
            require_uid=True,
        )
    assert len(deletions) == 1


@pytest.mark.parametrize(
    "failure", ["", "purge", "configmap", "token", "registry", "postflight"]
)
def test_cleanup_attempts_independent_work_but_never_overwrites_a_failure_with_zeroes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    selected = probe(tmp_path)
    events = []
    registry = []
    if failure == "postflight":
        registry = [{"cluster_id": "perf-cap-remaining", "synthetic_run_id": RUN_ID}]
    monkeypatch.setattr(seeded, "load_registry", lambda: registry)
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda _probe: {"count": 0})
    monkeypatch.setattr(seeded, "database_residuals", lambda _prefix: {"total": 0})
    seeded.write_json(
        tmp_path / "probe-resource-identities.json",
        {
            "run_id": RUN_ID,
            "resources": {
                f"pod/{selected.pod}": "uid-pod",
                f"configmap/{selected.configmap}": "uid-configmap",
            },
        },
    )
    seeded.write_json(
        tmp_path / "registry-token-proof.json", {"run_id": RUN_ID, "uid": "uid-secret"}
    )

    def delete(kind: str, _name: str, run_id: str, **kwargs: Any) -> None:
        events.append(kind)
        assert run_id == RUN_ID
        assert kwargs["require_uid"] is True
        assert kwargs["expected_uid"] == f"uid-{kind}"
        if kind == failure or kind == "secret" and failure == "token":
            raise OSError(f"synthetic {kind} deletion failure")

    def purge(value: dict[str, Any]) -> dict[str, Any]:
        assert events[0] == "pod"
        assert value["command_id"] == f"remote-{RUN_ID}"
        events.append("purge")
        if failure == "purge":
            raise OSError("synthetic purge failure")
        return {"remaining": [], "remaining_links": 0}

    def deregister(_directory: Path, run_id: str) -> None:
        events.append("registry")
        assert run_id == RUN_ID
        if failure == "registry":
            raise OSError("synthetic registry failure")

    monkeypatch.setattr(seeded, "delete_owned_resource", delete)
    monkeypatch.setattr(seeded, "deregister_synthetic_cluster", deregister)
    result = {"verdict": "PASS"}
    seeded.cleanup(
        selected,
        tmp_path,
        RUN_ID,
        result,
        seeded.seed_identity(RUN_ID),
        state={"probe_started": True, "registry_started": True},
        purge=purge,
    )
    assert events == ["pod", "purge", "configmap", "secret", "registry"]
    assert result["verdict"] == ("FAIL" if failure else "PASS")
    assert seeded.residual_free(result) is (not failure)
    if failure == "postflight":
        assert "postflight_error" in result
        assert result["registry_postflight"]["count"] == 1
    elif failure:
        assert "cleanup_error" in result
        assert result["registry_postflight"]["count"] == 0
        assert result["kubernetes_postflight"]["count"] == 0


def test_failed_claimant_stop_withholds_store_and_registry_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected = probe(tmp_path)
    events = []

    def stop(kind: str, *_args: Any, **_kwargs: Any) -> None:
        events.append(kind)
        if kind == "pod":
            raise TimeoutError("claimant stop is unknown")

    monkeypatch.setattr(seeded, "delete_owned_resource", stop)
    monkeypatch.setattr(
        seeded,
        "deregister_synthetic_cluster",
        lambda *_args: pytest.fail(
            "deregistration raced a potentially running claimant"
        ),
    )
    monkeypatch.setattr(seeded, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(seeded, "kubernetes_residuals", lambda _probe: {"count": 0})
    result = {"verdict": "PASS"}
    seeded.cleanup(
        selected,
        tmp_path,
        RUN_ID,
        result,
        seeded.seed_identity(RUN_ID),
        state={"probe_started": True, "registry_started": True},
        purge=lambda _seed: pytest.fail("purge raced a potentially running claimant"),
        database_probe=lambda _prefix: {"total": 0},
    )
    assert events == ["pod", "configmap"]
    assert result["verdict"] == "FAIL"
    assert seeded.residual_free(result) is False
