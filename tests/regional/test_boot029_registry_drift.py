"""The BOOT-029 registry convergence check in process (``boot029_registry_drift``).

The driver contract (``test_admin_lifecycle_sequence_registry_drift``) runs the
check as a subprocess against a fake ``kubectl``; here its rules are pinned one
by one against a fake control plane and a fake clock: the BOOT-019 role
boundary between the baseline and the reading after the verb, the product
generation read from the keys ``/healthz`` reports, the per-Pod registry
findings, the perf gate's bounded wait reused for every CPU role, the receipt
that accumulates one verdict per verb, and the fail-closed paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.failure_domains import FAILURE_DOMAIN_MAP_ANNOTATION
from scripts.e2e.regional import boot029_registry_drift as drift
from scripts.perf import regional_registry_alignment as alignment

API = "gpu-fault-api-ha"
WORKER = "gpu-fault-control-worker"
SPOOL = "gpu-fault-telemetry-spool-worker"
HEAD = "a" * 64
STALE = "b" * 64
MOUNTED = {
    "ready": True,
    "secret_drift": False,
    "secret_config_sha256": HEAD,
    "secret_source": "mounted-file",
    "secret_startup_config_sha256": STALE,
    "secret_error": None,
    "durable_config_sha256": HEAD,
}
PRE_SECRET_SOURCE = {"ready": True, "secret_drift": True, "secret_config_sha256": STALE}


def _role(
    uid: str, pods: dict[str, tuple[str, int]], *, template: str = "t", fd: str = ""
) -> dict[str, Any]:
    return {
        "uid": uid,
        "template_sha256": template * 64,
        "failure_domain_map_sha256": fd or None,
        "pods": {
            pod_uid: {"pod": name, "phase": "Running", "restarts": restarts}
            for pod_uid, (name, restarts) in pods.items()
        },
    }


def _baseline() -> dict[str, dict[str, Any]]:
    return {
        API: _role("d-api", {"u-a0": ("api-0", 0), "u-a1": ("api-1", 1)}),
        WORKER: _role("d-w", {"u-w0": ("worker-0", 0)}, fd="1" * 64),
        SPOOL: _role("d-s", {"u-s0": ("spool-0", 0)}),
    }


def test_unchanged_roles_are_listed_and_nothing_is_found() -> None:
    findings, rolled, unchanged = drift.pod_changes(
        _baseline(), _baseline(), operation="join-cluster"
    )

    assert (findings, rolled, unchanged) == ([], [], [API, WORKER, SPOOL])


def test_the_worker_may_roll_for_a_new_failure_domain_map_alone() -> None:
    after = _baseline()
    after[WORKER] = _role("d-w", {"u-w9": ("worker-9", 0)}, fd="2" * 64)

    findings, rolled, unchanged = drift.pod_changes(
        _baseline(), after, operation="remove-cluster"
    )

    assert findings == [] and unchanged == [API, SPOOL], findings
    assert rolled == [
        {
            "deployment": WORKER,
            "failure_domain_map_sha256_12": ["1" * 12, "2" * 12],
            "pods_before": 1,
            "pods_after": 1,
            "pods_replaced": 1,
        }
    ]


def _changed(change: Callable[[dict[str, dict[str, Any]]], None]) -> list[str]:
    after = _baseline()
    change(after)
    findings, _rolled, _unchanged = drift.pod_changes(
        _baseline(), after, operation="remove-cluster"
    )
    return findings


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda roles: roles[API]["pods"].update(
                {"u-a1": {"pod": "api-1", "phase": "Running", "restarts": 2}}
            ),
            f"{API}: container restarts in api-1 (uid u-a1)",
        ),
        (
            lambda roles: roles[API].update(
                pods={"u-a0": roles[API]["pods"]["u-a0"]}
                | {"u-a9": {"pod": "api-9", "phase": "Pending", "restarts": 0}}
            ),
            f"{API}: Pod(s) api-1 (uid u-a1) replaced by api-9 (uid u-a9); "
            "remove-cluster must not need a restart",
        ),
        (
            lambda roles: roles[SPOOL].update(uid="d-s2"),
            f"{SPOOL}: the Deployment was replaced (uid d-s -> d-s2)",
        ),
        (
            lambda roles: roles[SPOOL].update(template_sha256="e" * 64),
            f"{SPOOL}: its Pod template changed (template tttttttttttt -> "
            "eeeeeeeeeeee): remove-cluster started a roll of the role",
        ),
        (
            lambda roles: roles[API].update(failure_domain_map_sha256="3" * 64),
            f"{API}: its Pod template's failure-domain map digest changed, which "
            "only the control-worker carries",
        ),
        (
            lambda roles: roles[WORKER].update(
                pods={"u-w9": {"pod": "worker-9", "phase": "Running", "restarts": 0}}
            ),
            f"{WORKER}: Pod(s) worker-0 (uid u-w0) replaced by worker-9 (uid u-w9); "
            "remove-cluster must not need a restart",
        ),
        (
            lambda roles: roles[WORKER].update(
                failure_domain_map_sha256="2" * 64, template_sha256="e" * 64
            ),
            f"{WORKER}: its Pod template changed (template tttttttttttt -> "
            "eeeeeeeeeeee, besides the failure-domain map): remove-cluster started "
            "a roll of the role",
        ),
        (lambda roles: roles.pop(SPOOL), f"{SPOOL} is missing from the reading"),
    ],
    ids=[
        "container-restart",
        "pod-replaced",
        "deployment-replaced",
        "template-changed",
        "map-outside-worker",
        "worker-roll-without-map",
        "worker-template-beyond-map",
        "role-missing",
    ],
)
def test_the_role_boundary_names_every_change_a_verb_must_not_make(
    change: Callable[[dict[str, dict[str, Any]]], None], expected: str
) -> None:
    assert _changed(change) == [expected]


def test_a_role_absent_from_the_baseline_is_named() -> None:
    before = _baseline()
    before.pop(API)

    findings, _rolled, unchanged = drift.pod_changes(
        before, _baseline(), operation="join-cluster"
    )

    assert findings == [f"{API} is missing from the baseline"]
    assert unchanged == [WORKER, SPOOL]


@pytest.mark.parametrize(
    ("states", "generation"),
    [
        ([MOUNTED, MOUNTED], drift.MOUNTED_GENERATION),
        ([PRE_SECRET_SOURCE, PRE_SECRET_SOURCE], drift.STARTUP_GENERATION),
        ([MOUNTED, PRE_SECRET_SOURCE], drift.MIXED_GENERATION),
        ([{}, {"ready": True}], drift.NO_ANSWER),
        ([{}, PRE_SECRET_SOURCE], drift.STARTUP_GENERATION),
    ],
    ids=["mounted", "pre-secret-source", "mixed", "no-answer", "silent-pod-ignored"],
)
def test_the_generation_comes_from_the_keys_the_pods_report(
    states: list[dict[str, Any]], generation: str
) -> None:
    replicas = [
        {
            "pod": f"pod-{index}",
            "secret_drift": state.get("secret_drift"),
            "secret_source_reported": "secret_source" in state,
        }
        for index, state in enumerate(states)
    ]

    assert drift.product_generation(replicas) == generation


def test_registry_findings_name_what_keeps_a_pod_from_converging() -> None:
    def replica(name: str, **fields: Any) -> dict[str, Any]:
        return {"pod": name, "secret_source_reported": True, **MOUNTED, **fields}

    findings = drift.registry_findings(
        [
            replica("clean"),
            {"pod": "silent", "http_status": 503, "secret_drift": None},
            replica("old", secret_source_reported=False, secret_source=None),
            replica("unmounted", secret_source="startup-environment"),
            replica("stuck", secret_drift=True, secret_config_sha256=STALE),
            replica(
                "unreadable",
                ready=False,
                secret_config_sha256=None,
                secret_error="registry Secret file is unreadable (PermissionError)"
                + "x" * 300,
            ),
        ],
        window=180.0,
    )

    assert findings[:4] == [
        "silent: /healthz answered no regional_registry (HTTP 503)",
        "old names no secret_source: it compares the registry Secret it started "
        "with, which only a restart refreshes",
        "unmounted compares the startup-environment copy of the registry Secret, "
        "not the mounted file: it drifts after every join or remove until restarted",
        "stuck reports secret_drift=true ready=true after the 180 s convergence "
        f"window (Secret {'b' * 12}, durable head {'a' * 12})",
    ]
    (unreadable,) = findings[4:]
    assert unreadable.startswith(
        "unreadable reports secret_drift=false ready=false after the 180 s "
        "convergence window (Secret unreported, durable head aaaaaaaaaaaa, "
        "registry Secret file is unreadable (PermissionError)xxx"
    ), unreadable
    assert len(unreadable) < 400, "the product's fixed-text error is cut short"


class _Plane:
    """kubectl for one CPU control plane: Deployments, Pods and scripted
    ``/healthz`` answers (``registry(pod, read)`` per exec, ``None`` for a Pod
    that answers no ``regional_registry``)."""

    def __init__(
        self,
        *,
        registry: Callable[[str, int], dict[str, Any] | None] | None = None,
        pods: list[dict[str, Any]] | None = None,
        worker_map: str = "1" * 64,
    ) -> None:
        self.registry = registry or (lambda pod, read: MOUNTED)
        self.pods = pods if pods is not None else _pods()
        self.worker_map = worker_map
        self.reads: dict[str, int] = {}
        self.calls: list[tuple[str, ...]] = []

    def deployment(self, name: str) -> dict[str, Any]:
        annotations = {"gpu-fault.io/artifact-sha256": "c" * 64}
        if name == WORKER:
            annotations[FAILURE_DOMAIN_MAP_ANNOTATION] = self.worker_map
        return {
            "metadata": {"name": name, "uid": f"d-{name}"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": name}, "annotations": annotations},
                    "spec": {"containers": [{"name": name}]},
                }
            },
        }

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        if args[:2] == ("get", "deployment"):
            names = [item for item in args[2:] if item not in {"-o", "json"}]
            return json.dumps({"items": [self.deployment(name) for name in names]})
        if args[:2] == ("get", "pod"):
            return json.dumps({"items": copy.deepcopy(self.pods)})
        assert args[0] == "exec", f"the check only reads: {args}"
        pod = args[2]
        self.reads[pod] = self.reads.get(pod, 0) + 1
        state = self.registry(pod, self.reads[pod])
        payload = {} if state is None else {"regional_registry": state}
        return json.dumps({"http_status": 200, "payload": payload})


def _pod(app: str, name: str, *, terminating: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": name,
        "uid": f"u-{name}",
        "labels": {"app": app},
    }
    if terminating:
        metadata["deletionTimestamp"] = "2030-01-01T00:00:00Z"
    return {
        "metadata": metadata,
        "status": {"phase": "Running", "containerStatuses": [{"restartCount": 0}]},
        "spec": {"containers": [{"ports": [{"name": "http", "containerPort": 8081}]}]},
    }


def _pods() -> list[dict[str, Any]]:
    return [_pod(API, "api-0"), _pod(WORKER, "worker-0"), _pod(SPOOL, "spool-0")]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _check(plane: _Plane, baseline: Any = None, **kwargs: Any) -> dict[str, Any]:
    clock = _Clock()
    return drift.check_operation(
        plane,
        baseline=baseline if baseline is not None else drift.cpu_inventory(plane),
        operation="join-cluster",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


def test_mounted_pods_that_converge_inside_the_window_pass() -> None:
    plane = _Plane(
        registry=lambda pod, read: MOUNTED
        if read > 2
        else {**MOUNTED, "secret_drift": True}
    )

    verdict = _check(plane)

    assert verdict["verdict"] == "PASS", verdict["findings"]
    assert verdict["waited_seconds"] == 2 * drift.MOUNTED_CONVERGENCE_POLL_SECONDS
    assert plane.reads == {"api-0": 3, "worker-0": 3, "spool-0": 3}
    assert verdict["unchanged_roles"] == [API, WORKER, SPOOL]
    assert verdict["reason"] == (
        "after join-cluster: all 3 CPU Pod(s) compare the mounted registry Secret "
        "and report no drift (20 s of the 180 s window); unchanged: "
        f"{API}, {WORKER}, {SPOOL}"
    )
    assert verdict["pods"][0] == {
        "pod": "api-0",
        "app": API,
        "uid": "u-api-0",
        "secret_source": "mounted-file",
        "secret_drift": False,
        "ready": True,
        "secret_config_sha256_12": "a" * 12,
        "durable_config_sha256_12": "a" * 12,
        "secret_error": None,
    }


def test_a_drift_past_the_window_fails_after_the_bounded_wait() -> None:
    plane = _Plane(registry=lambda pod, read: {**MOUNTED, "secret_drift": True})

    verdict = _check(plane, window_seconds=30.0, poll_seconds=10.0)

    assert verdict["verdict"] == "FAIL" and verdict["waited_seconds"] == 30.0
    assert plane.reads["api-0"] == 4, "the first read and three polls inside 30 s"
    assert verdict["reason"].startswith(
        "after join-cluster: api-0 reports secret_drift=true ready=true after the "
        "30 s convergence window"
    ), verdict["reason"]
    assert verdict["reason"].endswith("(+2 more; see the stage log)"), verdict


def test_an_older_product_is_recorded_without_a_wait_and_not_judged() -> None:
    plane = _Plane(registry=lambda pod, read: PRE_SECRET_SOURCE)

    verdict = _check(plane)

    assert verdict["verdict"] == "NOT_APPLICABLE", verdict
    assert verdict["product_generation"] == "pre-secret-source"
    assert verdict["waited_seconds"] == 0.0, "a start-up copy never converges alone"
    assert len(verdict["findings"]) == 6, verdict["findings"]
    assert (
        "not judged (6 finding(s) under the mounted-Secret rule)" in (verdict["reason"])
    ), verdict["reason"]


def test_pods_of_two_generations_fail() -> None:
    plane = _Plane(
        registry=lambda pod, read: PRE_SECRET_SOURCE if pod == "spool-0" else MOUNTED
    )

    verdict = _check(plane)

    assert verdict["verdict"] == "FAIL" and verdict["product_generation"] == "mixed"
    assert verdict["findings"][0] == (
        "CPU Pods of two product generations: api-0 names secret_source, "
        "worker-0 names secret_source, spool-0 lacks secret_source"
    )


def test_no_answering_pod_fails() -> None:
    plane = _Plane(registry=lambda pod, read: None)

    verdict = _check(plane)

    assert verdict["verdict"] == "FAIL" and verdict["product_generation"] == "no-answer"
    assert verdict["findings"][0] == (
        "no Running CPU Pod answered /healthz with a regional_registry"
    )


def test_a_terminating_pod_is_neither_asked_nor_counted() -> None:
    plane = _Plane(pods=[*_pods(), _pod(WORKER, "worker-old", terminating=True)])

    verdict = _check(plane)

    assert verdict["verdict"] == "PASS", verdict["findings"]
    assert "worker-old" not in plane.reads
    assert verdict["skipped_pods"] == [
        {"pod": "worker-old", "phase": "Running", "terminating": True}
    ]
    assert "u-worker-old" not in drift.cpu_inventory(plane)[WORKER]["pods"]


def test_the_inventory_digests_the_worker_template_without_its_map() -> None:
    before = drift.cpu_inventory(_Plane())
    after = drift.cpu_inventory(_Plane(worker_map="2" * 64))

    assert before[WORKER]["template_sha256"] == after[WORKER]["template_sha256"]
    assert before[WORKER]["failure_domain_map_sha256"] == "1" * 64
    assert after[WORKER]["failure_domain_map_sha256"] == "2" * 64
    assert before[API]["failure_domain_map_sha256"] is None


def test_the_inventory_refuses_a_missing_role() -> None:
    plane = _Plane()
    plane.deployment = lambda name: {"metadata": {"name": "other"}}

    with pytest.raises(drift.DriftCheckError, match="Deployment\\(s\\) missing"):
        drift.cpu_inventory(plane)


def test_the_perf_reader_asks_every_named_role_and_keeps_its_default() -> None:
    plane = _Plane(pods=[*_pods(), _pod(API, "api-old", terminating=True)])

    default, _ = alignment.api_replica_registry_health(plane)
    every, skipped = alignment.api_replica_registry_health(
        plane, apps=drift.CPU_RUNTIME_DEPLOYMENTS
    )

    assert plane.calls[0] == ("get", "pod", "-l", f"app={API}", "-o", "json")
    assert plane.calls[-4] == (
        "get",
        "pod",
        "-l",
        f"app in ({API},{WORKER},{SPOOL})",
        "-o",
        "json",
    )
    assert [item["pod"] for item in every] == ["api-0", "worker-0", "spool-0"]
    assert [item["app"] for item in every] == [API, WORKER, SPOOL]
    assert every[0]["uid"] == "u-api-0" and every[0]["secret_source_reported"] is True
    assert every[0]["durable_config_sha256"] == HEAD
    assert skipped == [{"pod": "api-old", "phase": "Running", "terminating": True}]
    assert len(default) == 3, "this fake answers every app; the selector is the test"
    with pytest.raises(ValueError, match="no CPU role"):
        alignment.api_replica_registry_health(plane, apps=())


def _site(tmp_path: Path, spec: Any) -> Path:
    (tmp_path / "site.yaml").write_text(json.dumps({"spec": spec}), encoding="utf-8")
    return tmp_path


def test_the_site_names_the_cpu_kubeconfig_and_namespace(tmp_path: Path) -> None:
    state = _site(
        tmp_path,
        {"namespace": "gpu-fault-system", "cpu": {"kubeconfig": "/s/cpu.kubeconfig"}},
    )

    assert drift.site_cpu_target(state) == ("/s/cpu.kubeconfig", "gpu-fault-system")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (None, "cannot be read"),
        ({"namespace": "ns", "cpu": {"kubeconfig": "cpu.kubeconfig"}}, "absolute"),
        ({"namespace": "ns"}, "absolute"),
        ({"cpu": {"kubeconfig": "/s/cpu.kubeconfig"}}, "spec.namespace"),
    ],
    ids=["no-site", "relative-kubeconfig", "no-cpu", "no-namespace"],
)
def test_the_site_target_is_never_guessed(
    tmp_path: Path, spec: Any, message: str
) -> None:
    state = tmp_path if spec is None else _site(tmp_path, spec)

    with pytest.raises(drift.DriftCheckError, match=message):
        drift.site_cpu_target(state)


def test_kubectl_reads_the_named_cluster_and_reports_its_failure() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        code = 1 if command[-1] == "fail" else 0
        return subprocess.CompletedProcess(command, code, "out", "x\nError: refused\n")

    control = drift.kubectl_control("/s/cpu.kubeconfig", "ns", run=run)

    assert control("get", "pod") == "out"
    command, options = calls[0]
    assert command == ["kubectl", "--kubeconfig", "/s/cpu.kubeconfig", "-n", "ns"] + [
        "get",
        "pod",
    ]
    assert options["stdin"] is subprocess.DEVNULL and options["timeout"] == 120
    with pytest.raises(drift.DriftCheckError, match="exited 1: Error: refused"):
        control("get", "pod", "fail")

    def slow(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 120)

    with pytest.raises(drift.DriftCheckError, match="exceeded 120 s"):
        drift.kubectl_control("/k", "ns", run=slow)("exec", "-i", "api-0")


def _command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plane: Any
) -> Callable[..., int]:
    state = _site(
        tmp_path,
        {"namespace": "gpu-fault-system", "cpu": {"kubeconfig": "/s/cpu.kubeconfig"}},
    )
    monkeypatch.setattr(drift, "kubectl_control", lambda kubeconfig, namespace: plane)

    def main(action: str, *extra: str) -> int:
        common = ["--state", str(state), "--operation", "remove-cluster"]
        if action == "baseline":
            return drift.main(
                ["baseline", *common, "--out", str(tmp_path / "base.json"), *extra]
            )
        return drift.main(
            [
                "check",
                *common,
                "--baseline",
                str(tmp_path / "base.json"),
                "--receipt",
                str(tmp_path / "receipt.json"),
                *extra,
            ]
        )

    return main


def test_the_command_line_accumulates_one_verdict_per_verb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _command_line(monkeypatch, tmp_path, _Plane())
    arn = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-fixture"

    assert main("baseline") == 0
    assert "registry baseline before remove-cluster: 3 CPU Pod(s) in 3 role(s)" in (
        capsys.readouterr().err
    )
    assert main("check", "--gpu-arn", arn) == 0
    first = capsys.readouterr()
    assert main("check") == 0

    assert first.out.startswith("after remove-cluster: all 3 CPU Pod(s)"), first.out
    assert "registry convergence #1: PASS -- after remove-cluster" in first.err
    assert "  api-0 uid=u-api-0 source=mounted-file drift=false ready=true" in (
        first.err
    ), first.err
    lines = (tmp_path / "receipt.json").read_text().splitlines()
    verdicts = drift.recorded_verdicts(tmp_path / "receipt.json")
    assert len(lines) == 2 and [item["sequence"] for item in verdicts] == [1, 2]
    assert verdicts[0]["gpu_arns_sha256_12"] == [
        hashlib.sha256(arn.encode()).hexdigest()[:12]
    ]
    assert verdicts[1]["gpu_arns_sha256_12"] == []
    baseline = json.loads((tmp_path / "base.json").read_text())
    assert baseline["schema_version"] == 1 and set(baseline["roles"]) == {
        API,
        WORKER,
        SPOOL,
    }


def test_a_failed_check_is_recorded_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plane = _Plane()
    main = _command_line(monkeypatch, tmp_path, plane)
    assert main("baseline") == 0
    plane.pods = [_pod(API, "api-9"), _pod(WORKER, "worker-0"), _pod(SPOOL, "spool-0")]
    capsys.readouterr()

    assert main("check") == 1

    out = capsys.readouterr().out
    assert out.startswith(f"after remove-cluster: {API}: Pod(s) api-0"), out
    (verdict,) = drift.recorded_verdicts(tmp_path / "receipt.json")
    assert verdict["verdict"] == "FAIL" and verdict["unchanged_roles"] == [
        WORKER,
        SPOOL,
    ]


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            lambda monkeypatch, tmp_path: monkeypatch.setenv(
                drift.WINDOW_ENV, "forever"
            ),
            "BOOT029_REGISTRY_WINDOW_SECONDS is not a number of seconds",
        ),
        (
            lambda monkeypatch, tmp_path: monkeypatch.setenv(drift.POLL_ENV, "-1"),
            "BOOT029_REGISTRY_POLL_SECONDS must be a finite, non-negative number",
        ),
        (
            lambda monkeypatch, tmp_path: (tmp_path / "base.json").write_text(
                json.dumps({"schema_version": 9})
            ),
            "the registry baseline has an unknown schema",
        ),
        (
            lambda monkeypatch, tmp_path: (tmp_path / "base.json").unlink(),
            "FileNotFoundError: [Errno 2] No such file or directory: '<path sha256:",
        ),
    ],
    ids=["window-not-a-number", "negative-poll", "baseline-schema", "no-baseline"],
)
def test_the_check_fails_closed_when_it_cannot_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setup: Callable[[pytest.MonkeyPatch, Path], Any],
    message: str,
) -> None:
    main = _command_line(monkeypatch, tmp_path, _Plane())
    assert main("baseline") == 0
    setup(monkeypatch, tmp_path)

    assert main("check") == 1

    (verdict,) = drift.recorded_verdicts(tmp_path / "receipt.json")
    assert verdict["verdict"] == "FAIL" and verdict["pods"] == []
    assert message in verdict["reason"], verdict["reason"]
    assert verdict["reason"].startswith("after remove-cluster: the check could not"), (
        verdict["reason"]
    )
    assert str(tmp_path) not in json.dumps(verdict), "paths only as digests"


def test_the_window_and_poll_overrides_are_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = _command_line(monkeypatch, tmp_path, _Plane())
    monkeypatch.setenv(drift.WINDOW_ENV, "0.5")
    monkeypatch.setenv(drift.POLL_ENV, "0")
    assert main("baseline") == 0

    assert main("check") == 0

    (verdict,) = drift.recorded_verdicts(tmp_path / "receipt.json")
    assert (verdict["window_seconds"], verdict["poll_seconds"]) == (0.5, 0.0)


def test_a_baseline_that_cannot_be_read_stops_before_the_verb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def unreachable(*args: str) -> str:
        raise drift.DriftCheckError("kubectl get deployment exited 1: refused")

    main = _command_line(monkeypatch, tmp_path, unreachable)

    assert main("baseline") == 1

    assert capsys.readouterr().out == (
        "the CPU Pods could not be read before remove-cluster, which did not run: "
        "kubectl get deployment exited 1: refused\n"
    )
    assert not (tmp_path / "base.json").exists(), "no baseline is left behind"
