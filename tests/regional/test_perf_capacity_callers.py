from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as capacity
from tests.regional._perf_caller_support import CapacityWire, capacity_args


def test_capacity_main_registers_creates_and_cleans_one_receipted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    assert capacity.main(capacity_args("all", tmp_path)) == 0, (
        "the fake complete run should finish"
    )

    intent = json.loads((tmp_path / "registry-registration-intent.json").read_text())
    receipts = json.loads((tmp_path / "capacity-resources.json").read_text())
    assert intent["cluster_ids"] == ["perf-cap-000"], (
        "cleanup scope must be persisted before registration"
    )
    assert receipts["run_id"] == "run-a", (
        "resource receipts must retain the registration identity"
    )
    assert all(item["uid"] for item in receipts["resources"].values()), (
        "all creations must retain their UIDs"
    )
    assert set(receipts["resources"]) == {
        f"configmap/{capacity.SCRIPT_CONFIGMAP}",
        f"configmap/{capacity.TEMPLATE_CONFIGMAP}",
        f"configmap/{capacity.START_GATE_CONFIGMAP}",
        f"role/{capacity.START_GATE_ROLE}",
        f"rolebinding/{capacity.START_GATE_ROLE_BINDING}",
        f"job/{capacity.CASES['burst']['job']}",
    }, "every real main resource must be in the run receipt"
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "exact cleanup must precede registry revocation"
    )
    assert wire.objects == {}, "the fake complete run must not leave owned resources"
    assert wire.entries == [], (
        "the fake complete run must not leave synthetic registration"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "ok", (
        "cleanup is part of success"
    )


@pytest.mark.parametrize("failure", ["missing-sink", "transport"])
def test_capacity_amp_refusal_cannot_register_or_authorize_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    if failure == "missing-sink":
        wire.amp.config["route"]["routes"] = []
        message = "must not mail drill alerts"
    else:
        wire.amp.error = TimeoutError("AMP preflight unavailable")
        message = "AMP preflight unavailable"

    with pytest.raises((RuntimeError, TimeoutError), match=message) as caught:
        capacity.main(capacity_args("all", tmp_path))

    if wire.amp.error is not None:
        assert caught.value is wire.amp.error, "the original AMP failure must survive"
    assert wire.creations == [], "AMP refusal must precede any token or load creation"
    assert wire.events == [], "AMP refusal must not publish, inspect, or clean run data"
    assert wire.teardowns == [], "AMP refusal cannot invent registry cleanup authority"
    assert not (tmp_path / "registry-registration-intent.json").exists(), (
        "failed AMP preflight must not grant registration ownership"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "aborted", (
        "failed AMP preflight must invalidate the run"
    )


@pytest.mark.parametrize(
    "resource",
    [
        capacity.SCRIPT_CONFIGMAP,
        capacity.TEMPLATE_CONFIGMAP,
        capacity.START_GATE_ROLE,
        str(capacity.CASES["burst"]["job"]),
    ],
)
def test_capacity_lost_create_ack_keeps_uid_and_cleans_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_create = resource

    with pytest.raises(TimeoutError, match="create ACK lost"):
        capacity.main(capacity_args("all", tmp_path))

    creations = [
        item for item in wire.creations if item["metadata"]["name"] == resource
    ]
    assert len(creations) == 1, "lost create ACK must not replay a creation"
    key = f"{creations[0]['kind'].lower()}/{resource}"
    receipt = json.loads((tmp_path / "capacity-resources.json").read_text())
    assert receipt["resources"][key]["uid"] == creations[0]["metadata"]["uid"], (
        "lost ACK UID must be durable"
    )
    assert f"delete:{key}" in wire.events, (
        "the created resource must be stopped by its receipt"
    )
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "failed creation still needs exact cleanup"
    )
    assert wire.objects == {}, "lost ACK cleanup must leave no owned resource"


def test_capacity_register_failure_unwinds_even_when_keep_registration_was_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_publish = True

    with pytest.raises(TimeoutError, match="registration ACK lost"):
        capacity.main(capacity_args("register", tmp_path, "--keep-registration"))

    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "partial registration cannot be intentionally kept"
    )
    assert wire.objects == {}, (
        "token creation must be tracked through a registration ACK failure"
    )


def test_capacity_register_run_teardown_reuse_original_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"
    intent = (tmp_path / "registry-registration-intent.json").read_bytes()
    proof = (tmp_path / "registry-token-proof.json").read_bytes()
    original_run = (tmp_path / "run.json").read_bytes()
    monkeypatch.setattr(
        capacity,
        "artifact_dir",
        lambda *_args: pytest.fail("resume cannot invent new receipts"),
    )

    assert capacity.main(capacity_args("run", tmp_path, "--keep-registration")) == 0, (
        "the registered run should execute"
    )
    assert wire.teardowns == [], (
        "keeping registration must retain the token and registry"
    )
    assert set(wire.objects) == {f"secret/{registry.TOKEN_SECRET}"}, (
        "kept registration must not keep load producers"
    )
    assert capacity.main(capacity_args("teardown", tmp_path)) == 0, (
        "explicit teardown should consume the original intent"
    )
    assert (tmp_path / "registry-registration-intent.json").read_bytes() == intent, (
        "resume must not replace intent"
    )
    assert (tmp_path / "registry-token-proof.json").read_bytes() == proof, (
        "resume must retain the original token UID"
    )
    assert (tmp_path / "run.json").read_bytes() == original_run, (
        "resume must preserve registration target provenance"
    )
    assert len(wire.teardowns) == 1, (
        "only the explicit teardown should revoke this kept registration"
    )


@pytest.mark.parametrize("command", ["run", "purge", "teardown"])
@pytest.mark.parametrize("missing", ["--suite-id", "--run-dir"])
def test_capacity_resume_requires_original_identity_before_api_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, missing: str
) -> None:
    CapacityWire(tmp_path, monkeypatch)
    monkeypatch.setattr(
        capacity,
        "release_identity",
        lambda: pytest.fail("unbound resume must not read the cluster"),
    )
    args = capacity_args(command, tmp_path)
    index = args.index(missing)
    del args[index : index + 2]

    with pytest.raises(SystemExit, match="2"):
        capacity.main(args)


@pytest.mark.parametrize("remaining", [1, False, None, "0"])
def test_capacity_incomplete_exact_cleanup_blocks_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remaining: object
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"
    wire.remaining = remaining

    with pytest.raises(RuntimeError, match="exact run-owned cleanup left data"):
        capacity.main(capacity_args("run", tmp_path))

    assert wire.teardowns == [], "unknown or nonzero cleanup cannot revoke credentials"
    assert set(wire.objects) == {f"secret/{registry.TOKEN_SECRET}"}, (
        "load resources stop but token remains recoverable"
    )
    assert (tmp_path / "registry-token-proof.json").is_file(), (
        "failed cleanup must preserve its original token receipt"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "aborted", (
        "cleanup failure cannot report success"
    )


def test_capacity_observation_failure_stops_sampler_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_wait = True

    with pytest.raises(RuntimeError, match="load observation failed"):
        capacity.main(capacity_args("all", tmp_path))

    assert wire.events.index("sampler-joined") < wire.events.index("data-cleanup"), (
        "sampling must stop on the failure path"
    )
    assert len(wire.teardowns) == 1, (
        "observation failure must still complete the owned lifecycle"
    )


@pytest.mark.parametrize("change", ["run", "cluster", "token", "release"])
def test_capacity_resume_rejects_drift_before_creating_load_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"
    if change in {"run", "cluster"}:
        path = tmp_path / "registry-registration-intent.json"
        value = json.loads(path.read_text())
        value["run_id" if change == "run" else "cluster_ids"] = (
            "other-run" if change == "run" else ["perf-cap-001"]
        )
        path.write_text(json.dumps(value))
    elif change == "token":
        wire.objects[f"secret/{registry.TOKEN_SECRET}"]["metadata"]["uid"] = (
            "replacement"
        )
    else:
        wire.identity["release_id"] = "different-release"

    with pytest.raises(RuntimeError, match="intent|Secret changed|release differs"):
        capacity.main(capacity_args("run", tmp_path))

    assert len(wire.creations) == 1, (
        "only the original token may exist after rejected resume"
    )
    assert wire.teardowns == [], "identity drift cannot authorize shared teardown"
    assert "data-cleanup" not in wire.events, (
        "rejected identity cannot authorize data deletion"
    )


def test_capacity_refuses_to_overwrite_an_existing_artifact_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    (tmp_path / "run.json").write_text('{"existing":true}')

    with pytest.raises(RuntimeError, match="artifacts already exist"):
        capacity.main(capacity_args("all", tmp_path))

    assert wire.creations == [], "existing evidence cannot authorize a new registration"
    assert (tmp_path / "run.json").read_text() == '{"existing":true}', (
        "existing evidence must remain unchanged"
    )


@pytest.mark.parametrize("remaining", [0, 1, False])
def test_capacity_no_purge_verifies_exact_absence_before_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remaining: object
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"
    wire.remaining = remaining

    if type(remaining) is int and remaining == 0:
        assert capacity.main(capacity_args("teardown", tmp_path, "--no-purge")) == 0, (
            "verified absence may revoke registration"
        )
        assert wire.teardowns[0]["purge"] is False, (
            "no-purge must not request destructive data cleanup"
        )
    else:
        with pytest.raises(RuntimeError, match="exact run-owned cleanup is incomplete"):
            capacity.main(capacity_args("teardown", tmp_path, "--no-purge"))
        assert wire.teardowns == [], (
            "unverified absence must keep registration recoverable"
        )
        assert (
            json.loads((tmp_path / "status.json").read_text())["status"] == "aborted"
        ), "failed cleanup must invalidate prior success"
    assert "data-cleanup" not in wire.events, (
        "no-purge may inspect only the exact registered scope"
    )


def test_capacity_purge_uses_original_scope_without_revoking_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"

    assert capacity.main(capacity_args("purge", tmp_path)) == 0, (
        "exact purge should finish"
    )
    assert wire.events[-1] == "data-cleanup", (
        "purge must delegate only the exact owned data scope"
    )
    assert wire.teardowns == [], "purge is not an authorization to revoke registration"
    assert set(wire.objects) == {f"secret/{registry.TOKEN_SECRET}"}, (
        "purge must preserve the original token"
    )


def test_capacity_teardown_failure_preserves_receipts_for_checked_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    assert (
        capacity.main(capacity_args("register", tmp_path, "--keep-registration")) == 0
    ), "registration should succeed"
    original_intent = (tmp_path / "registry-registration-intent.json").read_bytes()

    def fail(**_kwargs):
        raise RuntimeError("shared teardown unavailable")

    monkeypatch.setattr(capacity, "teardown", fail)
    with pytest.raises(RuntimeError, match="shared teardown unavailable"):
        capacity.main(capacity_args("teardown", tmp_path))
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "aborted", (
        "failed teardown cannot preserve success"
    )
    monkeypatch.setattr(capacity, "teardown", wire.teardown)
    assert capacity.main(capacity_args("teardown", tmp_path)) == 0, (
        "same-run cleanup retry should finish"
    )
    assert (
        tmp_path / "registry-registration-intent.json"
    ).read_bytes() == original_intent, "retry must retain the original intent"


def test_capacity_never_predeletes_an_existing_same_name_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    job = str(capacity.CASES["burst"]["job"])
    foreign = {
        "kind": "Job",
        "metadata": {
            "name": job,
            "namespace": capacity.NAMESPACE,
            "uid": "foreign",
            "resourceVersion": "1",
            "labels": {registry.RUN_LABEL: "other-run"},
        },
    }
    wire.objects[f"job/{job}"] = foreign

    with pytest.raises(RuntimeError, match="confirmed unused name"):
        capacity.main(capacity_args("all", tmp_path))

    assert wire.objects[f"job/{job}"] == foreign, (
        "same-name foreign Job must stay untouched"
    )
    assert f"delete:job/{job}" not in wire.events, (
        "a previous Job name is not deletion authority"
    )
