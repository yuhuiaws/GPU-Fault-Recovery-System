from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_resources as resources
from scripts.e2e.regional import run_ha011_busy_cpu_takeover as runner
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.ha_cleanup import ProcessSupervisionLost
from scripts.e2e.regional.regional_commands import RegionalCommandTimeout
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureAbort
from tests.regional._cov95_ha011_support import (
    IMAGE_ID,
    Clock,
    deadline,
    install_kubernetes,
    settings_at,
)
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)
from tests.regional.test_cov95_ha011_resources import prepare


def plan(settings, root: Path) -> dict:
    case_dir = root / "cases" / contracts.CASE_ID
    preflight = runner.read_only_preflight(settings, case_dir)
    value = {"details": runner.plan_details(settings, preflight)}
    write_json_atomic(case_dir / "plan.json", value)
    return value


def test_configuration_uses_only_the_canonical_case_predecessor(
    monkeypatch, tmp_path: Path
) -> None:
    settings = settings_at(tmp_path)
    args = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(settings.cpu_kubeconfig),
            "--cpu-context",
            settings.cpu_context,
            "--cluster-id",
            settings.cluster_id,
            "--region",
            settings.region,
            "--isolation-id",
            settings.isolation_id,
            "--postgres-image",
            settings.postgres_image,
        ]
    )
    called = []

    def canonical(root, case_id, explicit):
        called.append((root, case_id, explicit))
        return "GF-REGIONAL-HA-010", settings.predecessor_path

    monkeypatch.setattr(runner, "predecessor_path", canonical)
    configured = runner.configure(args)
    assert called == [(tmp_path, contracts.CASE_ID, "")], (
        "only HA011's registered predecessor may configure the case"
    )
    assert configured.predecessor_case == "GF-REGIONAL-HA-010", (
        "the predecessor must come from the shared formal contract"
    )
    monkeypatch.setattr(runner, "predecessor_path", lambda *_args: (None, None))
    with pytest.raises(contracts.ProofError, match="registered"):
        runner.configure(args)


def test_main_delegates_to_the_shared_approval_driver_without_invoking_cli(
    monkeypatch,
) -> None:
    received = []
    monkeypatch.setattr(
        runner, "run_standard_case", lambda case: received.append(case) or 7
    )
    assert runner.main() == 7 and received == [runner.CASE], (
        "the new entrypoint must retain the existing approval/receipt/supervision API"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "predecessor",
        "existing-namespace",
        "missing-plan",
        "plan-drift",
        "expired-window",
    ],
)
def test_failed_admission_never_creates_or_arms_resources(
    monkeypatch, tmp_path: Path, capsys, failure: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    value = plan(settings, tmp_path)
    window = deadline()
    if failure == "predecessor":
        write_json_atomic(settings.predecessor_path, {"verdict": "FAIL"})
    elif failure == "existing-namespace":
        fake.put(
            {
                "kind": "Namespace",
                "metadata": {"name": settings.isolated_namespace, "uid": "not-ours"},
            }
        )
    elif failure == "missing-plan":
        (tmp_path / "cases" / contracts.CASE_ID / "plan.json").unlink()
    elif failure == "plan-drift":
        value["details"]["owned_process_crash_only"] = False
        write_json_atomic(tmp_path / "cases" / contracts.CASE_ID / "plan.json", value)
    else:
        from datetime import datetime, timezone

        window = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert runner.execute_case(settings, tmp_path, 1, window) == 1, (
        "failed admission must return failure"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "FAIL" and not fake.created and not fake.armed, (
        "no deployment mutation may follow a missing approval identity"
    )


@pytest.mark.parametrize("when", ["create", "cleanup"])
def test_supervision_loss_prevents_all_subsequent_remote_commands(
    monkeypatch, tmp_path: Path, capsys, when: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    receipt = tmp_path / "cases" / contracts.CASE_ID / f"{contracts.CASE_ID}.json"
    write_json_atomic(
        receipt, {"case_id": contracts.CASE_ID, "attempt": 0, "verdict": "PASS"}
    )
    failure_index = None

    def lose(args):
        nonlocal failure_index
        if args[0] == ("create" if when == "create" else "delete"):
            failure_index = len(fake.calls)
            raise ProcessSupervisionLost("controlled owned-command supervision loss")

    fake.before = lose
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "supervision loss must override any proof PASS"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["supervision_lost"] is True and result["verdict"] == "FAIL", (
        "loss of supervision must be explicit in the final failure receipt"
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["attempt"] == 1, (
        "supervision loss must not leave the preceding attempt's PASS current"
    )
    assert failure_index is not None and len(fake.calls) == failure_index, (
        "no reads, retries, or cleanup commands may follow supervision loss"
    )


def test_unapproved_probe_fields_are_not_written_to_evidence(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    original = fake.dispatch

    def extra(args, namespace, text):
        value = original(args, namespace, text)
        if args[0] == "logs":
            proof = json.loads(value)
            proof["unexpected"] = "PROTECTED_TEST_PAYLOAD"
            return json.dumps(proof)
        return value

    fake.dispatch = extra
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "unknown probe output must fail closed"
    )
    output = capsys.readouterr().out
    assert "PROTECTED_TEST_PAYLOAD" not in output, (
        "unapproved probe data must not leak into public evidence"
    )
    assert "probe" not in json.loads(output), (
        "invalid raw evidence must never be persisted"
    )
    assert fake.deleted, "an invalid probe still requires verified owned cleanup"


def test_cleanup_failure_overrides_successful_runtime_evidence(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    receipt = tmp_path / "cases" / contracts.CASE_ID / f"{contracts.CASE_ID}.json"
    write_json_atomic(
        receipt, {"case_id": contracts.CASE_ID, "attempt": 0, "verdict": "PASS"}
    )

    def refuse_delete(args):
        if args[0] == "delete":
            raise RegionalCommandTimeout(args, 1)

    fake.before = refuse_delete
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "unfinished cleanup must override a complete runtime proof"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["cleanup_error_type"] == "RegionalCommandTimeout", (
        "cleanup ambiguity must be retained"
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["verdict"] == "FAIL", (
        "cleanup failure must replace the preceding attempt's PASS"
    )
    assert fake.armed and not fake.deleted, (
        "a cleanup timeout must not be described as completed deletion"
    )


@pytest.mark.parametrize(
    "response", ["not-json", "[]", '{"kind":"Node","metadata":{"name":"other"}}']
)
def test_malformed_kubernetes_reads_never_become_absence(
    monkeypatch, tmp_path: Path, response: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    fake.dispatch = lambda *_args: response
    with pytest.raises(contracts.ProofError):
        resources.CpuKubernetes(settings).read("Node", "cpu-node", optional=True)


@pytest.mark.parametrize(
    "failure", ["restart", "exit", "database", "malformed-proof", "timeout"]
)
def test_collection_requires_surviving_database_and_successful_bound_runtime(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    lifetime.arm(IMAGE_ID)
    statuses = fake.object("Pod", contracts.POD_NAME)["status"]["containerStatuses"]
    if failure == "restart":
        statuses[0]["restartCount"] = 1
    elif failure == "exit":
        statuses[0]["state"]["terminated"]["exitCode"] = 1
    elif failure == "database":
        statuses[1]["state"] = {"terminated": {"exitCode": 0}}
    elif failure == "malformed-proof":
        fake.probe = []
    else:
        statuses[0]["state"] = {"running": {"startedAt": "observed"}}
        monkeypatch.setattr(resources, "time", Clock(step=100))
    with pytest.raises(contracts.ProofError):
        lifetime.collect(IMAGE_ID)


def test_lifecycle_empty_and_existing_namespace_states_are_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    assert lifetime.cleanup()["namespace_absent"] is True, (
        "unused fixtures have no remote cleanup"
    )
    with pytest.raises(contracts.ProofError, match="no owned"):
        lifetime.collect(IMAGE_ID)
    with pytest.raises(contracts.ProofError, match="no owned"):
        lifetime.read_probe_pod(scheduled=False)
    lifetime.start(objects)
    with pytest.raises(contracts.ProofError, match="existing"):
        lifetime.start(objects)
    lifetime.cleanup()
    assert lifetime.cleanup()["namespace_absent"] is True, (
        "cleanup retry must observe absence without deleting again"
    )
    assert fake.objects.get(("Namespace", None, settings.isolated_namespace)) is None, (
        "owned namespace must stay absent"
    )


@pytest.mark.parametrize("failure", ["uid", "timeout"])
def test_cleanup_wait_refuses_recreated_or_stuck_namespace(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    original = fake.dispatch

    def retain(args, namespace, text):
        if args[0] == "delete":
            retained = fake.object("Namespace", settings.isolated_namespace)
            retained["metadata"]["deletionTimestamp"] = "deleting"
            if failure == "uid":
                retained["metadata"]["uid"] = "recreated"
            return "{}"
        return original(args, namespace, text)

    fake.dispatch = retain
    monkeypatch.setattr(resources, "time", Clock(step=16))
    with pytest.raises(contracts.ProofError):
        lifetime.cleanup()
    assert sum(args[0] == "delete" for _, args in fake.calls) == 1, (
        "cleanup may not retry deletion by name after ambiguity"
    )


@pytest.mark.parametrize(
    "failure", ["already-scheduled", "missing-version", "lost-ack", "timeout"]
)
def test_scheduling_gate_cannot_be_released_without_owned_conditional_ack(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    pod = fake.object("Pod", contracts.POD_NAME)
    if failure == "already-scheduled":
        pod["spec"]["nodeName"] = "cpu-node"
    elif failure == "missing-version":
        pod["metadata"].pop("resourceVersion")
    else:
        original = fake.dispatch

        def lost(args, namespace, text):
            result = original(args, namespace, text)
            if args[0] == "patch":
                if failure == "timeout":
                    raise RegionalCommandTimeout(args, 1)
                return "not-the-owned-uid"
            return result

        fake.dispatch = lost
    with pytest.raises((contracts.ProofError, RegionalCommandTimeout)):
        lifetime.arm(IMAGE_ID)
    assert not fake.armed, (
        "an unconfirmed scheduling transition must not arm runtime or database"
    )
    assert sum(args[0] == "patch" for _, args in fake.calls) <= 1, (
        "activation must not retry an ambiguous mutation"
    )


def test_changed_bundle_between_preflight_and_create_is_rejected(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    original = runner.source_bundle
    calls = 0

    def drifting(root):
        nonlocal calls
        calls += 1
        return original(root) if calls == 1 else "different-bundle"

    monkeypatch.setattr(runner, "source_bundle", drifting)
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "post-preflight probe drift must invalidate activation"
    )
    assert not fake.created, "changed source must be rejected before namespace creation"
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL", (
        "source drift must remain a failure receipt"
    )


def test_business_identity_drift_after_probe_cannot_pass(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)

    def replace_business(args):
        if fake.armed and args[:2] == ["get", "Deployment"]:
            fake.object("Deployment", contracts.DEPLOYMENT, business=True)["metadata"][
                "uid"
            ] = "replaced"

    fake.before = replace_business
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "business release identity must remain bound throughout the proof"
    )
    assert fake.deleted, "post-proof drift still requires owned isolated cleanup"
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL", (
        "drift cannot retain the runtime PASS"
    )


@pytest.mark.parametrize("abort", [KeyboardInterrupt, lambda: RegionalFixtureAbort(2)])
@pytest.mark.parametrize("when", ["preflight", "arm"])
def test_aborted_retry_invalidates_old_pass_before_any_runtime_step(
    monkeypatch, tmp_path: Path, abort, when: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    receipt = tmp_path / "cases" / contracts.CASE_ID / f"{contracts.CASE_ID}.json"
    old = {
        "case_id": contracts.CASE_ID,
        "attempt": 1,
        "verdict": "PASS",
        "status": "COMPLETED",
    }
    write_json_atomic(receipt, old)

    def interrupt(args):
        if (when == "preflight" and args[:2] == ["get", "Namespace"]) or (
            when == "arm" and args[0] == "exec"
        ):
            raise abort()

    fake.before = interrupt
    with pytest.raises((KeyboardInterrupt, RegionalFixtureAbort)):
        runner.execute_case(settings, tmp_path, 2, deadline())
    current = json.loads(receipt.read_text(encoding="utf-8"))
    assert current["attempt"] == 2 and current["verdict"] != "PASS", (
        "an aborted retry must never leave the previous attempt's PASS as current"
    )
    assert current["status"] == "PREFLIGHT", (
        "the current-attempt non-PASS receipt must predate fallible work"
    )
    assert fake.deleted is (when == "arm"), (
        "abort cleanup must still obey the owned resource lifecycle"
    )
    previous = receipt.parent / current["previous_receipt"]
    assert json.loads(previous.read_text(encoding="utf-8"))["attempt"] == 1, (
        "invalidating current success must preserve the prior attempt as history"
    )
