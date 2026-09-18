from __future__ import annotations

import copy
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gpu_fault.admin import deadlines
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault_release import regional_release_probe_job as jobs
from gpu_fault_release import regional_release_probe_job_diagnostics as diagnostics
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._prerequisite_repair_support import repair_release

PRIVATE = "fixture-private-unlabelled-material"


class ProbeFailure:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.release = repair_release(monkeypatch)
        self.original_run = self.release.runner.run
        self.records: list[dict[str, Any]] = []
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.failure: BaseException | None = ReleaseError(
            "proof Job failed: BackoffLimitExceeded"
        )
        self.read_failure: tuple[str, BaseException] | None = None
        self.on_wait: Callable[[], None] = lambda: None
        self.on_logs: Callable[[], None] = lambda: None
        self.on_list: Callable[[], None] = lambda: None
        self.raw_listing: str | None = None
        self.more_pods = False
        self.current_pod: dict[str, Any] | None = None
        self.private_key_pem = (
            Ed25519PrivateKey.generate()
            .private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            .decode("ascii")
        )
        self.logs = "\n".join(
            (
                PRIVATE,
                json.dumps({"data": {"opaque": PRIVATE}}),
                self.private_key_pem,
                "Traceback (most recent call last):",
                f'  File "{PRIVATE}", line 1, in <module>',
                "ImportError: cannot import name 'refresh' from "
                "'gpu_fault.aurora_credential_refresh'",
            )
        )
        self.document = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "gpu-fault-store-proof-" + "a" * 16,
                "namespace": self.release.config.namespace,
                "annotations": {jobs.RUN_ANNOTATION: "a" * 32},
            },
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 300,
                "ttlSecondsAfterFinished": 300,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {"name": "proof", "image": "example/cpu@sha256:" + "b" * 64}
                        ],
                    }
                },
            },
        }
        self.pods = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "proof-pod",
                    "uid": "pod-uid",
                    "namespace": self.release.config.namespace,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": self.document["metadata"]["name"],
                            "uid": "job-0",
                            "controller": True,
                        }
                    ],
                },
                "spec": {"containers": [{"env": [{"value": PRIVATE}]}]},
                "status": {
                    "phase": "Failed",
                    "message": PRIVATE,
                    "containerStatuses": [
                        {
                            "name": "proof",
                            "state": {
                                "terminated": {
                                    "exitCode": 1,
                                    "signal": 0,
                                    "reason": "Error",
                                    "message": PRIVATE,
                                }
                            },
                        }
                    ],
                },
            }
        ]
        monkeypatch.setattr(self.release.runner, "run", self.run)

    def run(self, args: list[str], **kwargs: Any) -> str:
        if args[0] == "bash":
            self.release.runner.events.append("proof-wait")
            self.on_wait()
            if self.failure is not None:
                raise self.failure
            self.release.runner.jobs[args[3]]["status"] = {
                "conditions": [{"type": "Complete", "status": "True"}]
            }
            return ""
        stage = (
            "list"
            if "get" in args and "--raw" in args
            else "logs"
            if "logs" in args and any(arg.startswith("pod/") for arg in args)
            else "pod"
            if "get" in args and "pod" in args
            else ""
        )
        if not stage:
            return str(self.original_run(args, **kwargs))
        self.release.runner.events.append("diagnostic-" + stage)
        self.calls.append((args, kwargs))
        if self.read_failure is not None and self.read_failure[0] == stage:
            raise self.read_failure[1]
        if stage == "list":
            self.on_list()
            return (
                self.raw_listing
                if self.raw_listing is not None
                else json.dumps(
                    {
                        "kind": "PodList",
                        "metadata": {"continue": PRIVATE if self.more_pods else ""},
                        "items": self.pods,
                    }
                )
            )
        if stage == "pod":
            name = args[args.index("pod") + 1]
            return json.dumps(
                self.current_pod
                or next(pod for pod in self.pods if pod["metadata"]["name"] == name)
            )
        self.on_logs()
        return self.logs

    def execute(self) -> dict[str, Any]:
        return jobs.run_probe_job(
            self.release,
            self.document,
            lambda record: self.records.append(copy.deepcopy(record)),
            verify=lambda: {"verified": True},
        )


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> ProbeFailure:
    return ProbeFailure(monkeypatch)


def test_failed_job_emits_safe_diagnostics_before_uid_scoped_cleanup(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str]
) -> None:
    before = copy.deepcopy(probe.release.state)
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    output = capsys.readouterr()
    assert error.value is probe.failure, "diagnostics replaced the original failure"
    assert output.out == "", "diagnostics contaminated the machine-readable report"
    assert '"exitCode": 1' in output.err
    assert '"reason": "Error"' in output.err
    assert '"python_errors": ["ImportError"]' in output.err
    assert "<sensitive output redacted>" in output.err
    assert PRIVATE not in (
        output.err + str(error.value.__notes__) + json.dumps(probe.records)
    ), "raw logs or Pod fields escaped the diagnostic allowlist"
    rendered = output.err + str(error.value.__notes__) + json.dumps(probe.records)
    assert not any(
        line in rendered for line in probe.private_key_pem.splitlines()[1:-1]
    ), "private key material escaped the diagnostic allowlist"
    assert probe.release.state == before, "diagnostics modified release state"
    assert probe.release.runner.jobs == {}, "failed proof Job survived cleanup"
    assert probe.records[-1]["status"] == "REMOVED"
    assert set(probe.records[-1]) == {
        "name",
        "namespace",
        "run_id",
        "spec_sha256",
        "admitted_spec_sha256",
        "owner_uid",
        "uid",
        "status",
    }, "diagnostics persisted fields in the ownership journal"
    events = probe.release.runner.events
    assert events.index("diagnostic-logs") < events.index("job-delete")
    assert events.index("diagnostic-pod") < events.index("job-delete")
    for arguments, options in probe.calls:
        assert arguments[:3] == ["kubectl", "--context", "cpu"]
        assert options["capture"] is True and options["sensitive"] is True, (
            "a diagnostic command could forward raw captured failures"
        )
        assert 0 < options["timeout_seconds"] <= 5
        assert "--request-timeout=5s" in arguments
    list_arguments = probe.calls[0][0]
    query = parse_qs(urlsplit(list_arguments[list_arguments.index("--raw") + 1]).query)
    assert query == {
        "labelSelector": ["batch.kubernetes.io/controller-uid=job-0"],
        "limit": ["2"],
    }
    log_arguments = probe.calls[1][0]
    assert {"--tail=80", "--limit-bytes=8192", "pod/proof-pod"} <= set(log_arguments)
    assert "--follow" not in log_arguments and "--all-containers" not in log_arguments


def test_success_does_not_read_or_emit_failure_diagnostics(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str]
) -> None:
    probe.failure = None
    assert probe.execute() == {"verified": True}
    assert probe.calls == []
    assert capsys.readouterr().err == ""
    assert probe.records[-1]["status"] == "REMOVED"


def test_raw_pod_list_items_can_omit_type_metadata(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str]
) -> None:
    probe.current_pod = copy.deepcopy(probe.pods[0])
    probe.pods[0].pop("apiVersion")
    probe.pods[0].pop("kind")
    with pytest.raises(ReleaseError):
        probe.execute()
    assert '"python_errors": ["ImportError"]' in capsys.readouterr().err
    assert probe.records[-1]["status"] == "REMOVED"


@pytest.mark.parametrize("stage", ["list", "logs", "pod"])
@pytest.mark.parametrize("kind", [ReleaseError, OSError, ValueError])
def test_diagnostic_read_failure_keeps_original_and_cleans_job(
    probe: ProbeFailure,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    kind: type[Exception],
) -> None:
    probe.read_failure = stage, kind(PRIVATE)
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    output = capsys.readouterr().err
    assert error.value is probe.failure
    assert "unavailable" in output and kind.__name__ in output
    assert PRIVATE not in output + str(error.value.__notes__)
    if stage == "logs":
        assert '"exitCode": 1' in output, "log failure discarded readable exit status"
    assert probe.release.runner.jobs == {}
    assert probe.records[-1]["status"] == "REMOVED"


def test_cleanup_failure_is_not_hidden_by_successful_diagnostics(
    probe: ProbeFailure,
) -> None:
    probe.release.runner.fail_delete = True
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    assert error.value is probe.failure
    assert any("ImportError" in note for note in error.value.__notes__), (
        "successful failure diagnostics were discarded"
    )
    assert any(
        "cleanup also failed: ReleaseError" in note for note in error.value.__notes__
    ), "diagnostics hid the cleanup failure"
    assert probe.records[-1]["status"] == "RUNNING"
    assert len(probe.release.runner.jobs) == 1, "test did not preserve failed cleanup"


@pytest.mark.parametrize("field", ["uid", "run", "owner", "spec"])
@pytest.mark.parametrize("when", ["before-diagnostics", "during-logs"])
def test_job_drift_blocks_diagnostics_and_deletion(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str], field: str, when: str
) -> None:
    def mutate() -> None:
        job = next(iter(probe.release.runner.jobs.values()))
        if field == "uid":
            job["metadata"]["uid"] = "replacement"
        elif field == "run":
            job["metadata"]["annotations"][jobs.RUN_ANNOTATION] = "c" * 32
        elif field == "owner":
            job["metadata"]["ownerReferences"][0]["uid"] = "replacement"
        else:
            job["spec"]["backoffLimit"] = 1

    if when == "before-diagnostics":
        probe.on_wait = mutate
    else:
        probe.on_logs = mutate
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    assert error.value is probe.failure
    assert "ImportError" not in capsys.readouterr().err + str(error.value.__notes__)
    assert "job-delete" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "RUNNING"
    if when == "before-diagnostics":
        assert probe.calls == [], "an unowned Job authorized Pod diagnostics"


@pytest.mark.parametrize(
    "field",
    ["namespace", "owner-uid", "owner-name", "controller", "extra-owner", "kind"],
)
def test_unowned_pod_is_not_read_for_logs(probe: ProbeFailure, field: str) -> None:
    pod = probe.pods[0]
    metadata = pod["metadata"]
    if field == "namespace":
        metadata["namespace"] = "foreign"
    elif field == "kind":
        pod["kind"] = "Secret"
    elif field == "extra-owner":
        metadata["ownerReferences"].append(
            copy.deepcopy(metadata["ownerReferences"][0])
        )
    else:
        key = {"owner-uid": "uid", "owner-name": "name", "controller": "controller"}[
            field
        ]
        metadata["ownerReferences"][0][key] = (
            False if field == "controller" else "foreign"
        )
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    assert error.value is probe.failure
    assert "diagnostic-logs" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "REMOVED", "Pod refusal skipped Job cleanup"


@pytest.mark.parametrize("field", ["uid", "owner"])
def test_pod_drift_discards_collected_logs(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    probe.current_pod = copy.deepcopy(probe.pods[0])
    metadata = probe.current_pod["metadata"]
    if field == "uid":
        metadata["uid"] = "replacement"
    else:
        metadata["ownerReferences"][0]["uid"] = "replacement"
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    assert "diagnostic-logs" in probe.release.runner.events
    assert "ImportError" not in capsys.readouterr().err + str(error.value.__notes__)
    assert error.value is probe.failure
    assert probe.records[-1]["status"] == "REMOVED"


@pytest.mark.parametrize("stage", ["list", "logs", "pod"])
def test_diagnostic_deadline_failure_propagates_and_cleanup_still_runs(
    probe: ProbeFailure, stage: str
) -> None:
    failure = deadlines.DeploymentDeadlineExceeded("diagnostic deadline")
    probe.read_failure = stage, failure
    with pytest.raises(deadlines.DeploymentDeadlineExceeded) as error:
        probe.execute()
    assert error.value is failure, "a deadline violation became optional diagnostics"
    assert error.value.__context__ is probe.failure, "original Job failure was lost"
    assert probe.records[-1]["status"] == "REMOVED"
    assert probe.release.runner.jobs == {}


def test_supervision_loss_during_diagnostics_stops_all_further_commands(
    probe: ProbeFailure, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost = ProcessSupervisionLost("ownership unproved")
    poisoned = False

    def guard(*, allow_interrupted: bool = False) -> None:
        if poisoned:
            raise lost

    def poison() -> None:
        nonlocal poisoned
        poisoned = True
        raise lost

    monkeypatch.setattr(jobs, "ensure_supervision_safe", guard)
    monkeypatch.setattr(diagnostics, "ensure_supervision_safe", guard)
    probe.on_list = poison
    with pytest.raises(ProcessSupervisionLost) as error:
        probe.execute()
    assert error.value is lost
    assert [
        event
        for event in probe.release.runner.events
        if event.startswith("diagnostic-")
    ] == ["diagnostic-list"]
    assert "job-delete" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "RUNNING"


@pytest.mark.parametrize("when", ["wait", "list"])
def test_parent_deadline_is_not_extended_for_diagnostics(
    probe: ProbeFailure, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    now = 100.0
    monkeypatch.setattr(deadlines, "time", SimpleNamespace(monotonic=lambda: now))

    def expire() -> None:
        nonlocal now
        now += 11

    if when == "wait":
        probe.on_wait = expire
    else:
        probe.on_list = expire
    with deadlines.deadline_scope("parent", 10):
        with pytest.raises(deadlines.DeploymentDeadlineExceeded):
            probe.execute()
    assert "diagnostic-logs" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "REMOVED"
    assert probe.release.runner.jobs == {}
    if when == "wait":
        assert probe.calls == [], "diagnostics borrowed the cleanup reserve"


def test_diagnostics_have_their_own_twenty_second_cap(
    probe: ProbeFailure, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 100.0
    monkeypatch.setattr(deadlines, "time", SimpleNamespace(monotonic=lambda: now))

    def expire() -> None:
        nonlocal now
        now += 21

    probe.on_list = expire
    with deadlines.deadline_scope("parent", 100):
        with pytest.raises(
            deadlines.DeploymentDeadlineExceeded, match="Job diagnostics"
        ):
            probe.execute()
    assert "diagnostic-logs" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "REMOVED"


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt(), ProcessSupervisionLost("lost")]
)
def test_nonordinary_job_failure_does_not_start_diagnostics(
    probe: ProbeFailure, failure: BaseException
) -> None:
    probe.failure = failure
    with pytest.raises(type(failure)) as error:
        probe.execute()
    assert error.value is failure
    assert probe.calls == []


@pytest.mark.parametrize(
    "listing",
    [
        "not-json",
        "[]",
        '{"kind":"PodList","items":{}}',
        '{"kind":"Secret","items":[]}',
        '{"kind":"PodList","items":[],"extra":"' + "x" * 262144 + '"}',
    ],
    ids=["malformed", "nonobject", "invalid-items", "wrong-kind", "oversized"],
)
def test_invalid_pod_list_is_bounded_and_never_forwarded(
    probe: ProbeFailure, listing: str
) -> None:
    probe.raw_listing = listing
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    assert error.value is probe.failure
    assert "diagnostic-logs" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "REMOVED"


@pytest.mark.parametrize("count", [2, 3])
def test_duplicate_or_excess_pods_are_rejected_before_logs(
    probe: ProbeFailure, count: int
) -> None:
    probe.pods *= count
    with pytest.raises(ReleaseError):
        probe.execute()
    assert "diagnostic-logs" not in probe.release.runner.events
    assert probe.records[-1]["status"] == "REMOVED"


def test_unknown_status_and_log_values_are_not_echoed(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str]
) -> None:
    probe.logs = PRIVATE + "\nAccessDenied\n"
    status = probe.pods[0]["status"]
    status["phase"] = PRIVATE
    terminated = status["containerStatuses"][0]["state"]["terminated"]
    terminated.update(reason=PRIVATE, exitCode=PRIVATE, signal=True)
    with pytest.raises(ReleaseError):
        probe.execute()
    output = capsys.readouterr().err
    assert PRIVATE not in output
    assert '"phase": "Unknown"' in output and '"reason": "unknown"' in output
    assert "AccessDenied" in output
    assert "exitCode" not in output and "signal" not in output


@pytest.mark.parametrize("size", [8192, 8193])
def test_log_limit_never_forwards_untrusted_tail(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str], size: int
) -> None:
    probe.logs = "x" * (size - len(PRIVATE)) + PRIVATE
    with pytest.raises(ReleaseError) as error:
        probe.execute()
    output = capsys.readouterr().err
    assert PRIVATE not in output + str(error.value.__notes__)
    assert len(output) < 4096, "diagnostics exceeded the existing output budget"
    assert (
        '"limit_reached": true' if size == 8192 else '"unavailable": "ValueError"'
    ) in output
    assert probe.records[-1]["status"] == "REMOVED"


def test_container_sampling_uses_only_admitted_names_and_marks_omissions(
    probe: ProbeFailure, capsys: pytest.CaptureFixture[str]
) -> None:
    containers = probe.document["spec"]["template"]["spec"]["containers"]
    containers.extend(
        {"name": name, "image": "example/cpu@sha256:" + "b" * 64}
        for name in ("second", "third")
    )
    probe.pods[0]["status"]["containerStatuses"].append(
        {"name": "unadmitted", "state": {"waiting": {"reason": PRIVATE}}}
    )
    other = copy.deepcopy(probe.pods[0])
    other["metadata"].update(name="second-pod", uid="second-pod-uid")
    probe.pods.append(other)
    probe.more_pods = True
    with pytest.raises(ReleaseError):
        probe.execute()
    log_calls = [args for args, _options in probe.calls if "logs" in args]
    assert [args[args.index("-c") + 1] for args in log_calls] == [
        "proof",
        "second",
        "proof",
        "second",
    ]
    output = capsys.readouterr().err
    assert '"containers_omitted": 1' in output and '"more_pods": true' in output
    assert PRIVATE not in output
    assert probe.records[-1]["status"] == "REMOVED"
