"""admin-lifecycle-sequence.sh judges the CPU control plane after every join/remove.

``join-cluster`` and ``remove-cluster`` rewrite the registry Secret and publish
the durable head without restarting the CPU Deployments, and since batch 245
each CPU Pod compares the Secret as mounted, so the BOOT-019/024/025
checklists' "``secret_drift=false``, Pod UIDs unchanged" must hold after every
such verb without a restart. The driver reads the CPU cluster right before the
verb (Deployments and Pods) and right after it (every Pod's ``/healthz``,
within the perf registry gate's window, and the Pods again) through
``scripts/e2e/regional/boot029_registry_drift.py``; each verdict lands in the
stage receipt as ``registry_convergence`` and in the stage log.

The fakes are the BOOT-029 driver suite's own
(``test_admin_lifecycle_sequence_script``); the fake ``kubectl`` and its knobs
are ``_admin_lifecycle_cpu_plane``: by default the control-worker rolls for the
new failure-domain map digest after every verb and every Pod compares the
mounted Secret without drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.regional.test_admin_lifecycle_sequence_mixed_site import EKS_GPU_ARN, _mixed
from tests.regional.test_admin_lifecycle_sequence_script import (
    GPU_ARN,
    _admin_verbs,
    _install_fakes,
    _record,
    _run,
    _sha256,
    _tool_log,
)

WORKER = "gpu-fault-control-worker"
API = "gpu-fault-api-ha"
SPOOL = "gpu-fault-telemetry-spool-worker"
LONG_DIGEST = re.compile(r"[0-9a-f]{64}")


def _verdicts(tmp_path: Path, stage: str) -> list[dict]:
    return list(_record(tmp_path)["stages"][stage].get("registry_convergence", []))


def _stage_log(tmp_path: Path, stage: str) -> str:
    name = _record(tmp_path)["stages"][stage]["log_name"]
    return (tmp_path / "state" / "acceptance" / name).read_text(encoding="utf-8")


def _healthz_reads(tmp_path: Path, epoch: int) -> dict[str, int]:
    """How often the fake answered each Pod's ``/healthz`` in one epoch."""

    prefix = f"healthz-{epoch}-"
    return {
        path.name[len(prefix) :]: int(path.read_text())
        for path in tmp_path.glob(prefix + "*")
    }


def test_every_join_and_remove_is_judged_and_the_pods_stay(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    for stage, operation in (("2", "remove-cluster"), ("3", "join-cluster")):
        (verdict,) = _verdicts(tmp_path, stage)
        assert verdict["operation"] == operation and verdict["sequence"] == 1, verdict
        assert verdict["verdict"] == "PASS", verdict["findings"]
        assert verdict["product_generation"] == "secret-source", verdict
        assert verdict["gpu_arns_sha256_12"] == [_sha256(GPU_ARN)[:12]], verdict
        assert (verdict["window_seconds"], verdict["poll_seconds"]) == (180.0, 10.0), (
            "the perf registry gate's bound and poll interval by default"
        )
        assert verdict["unchanged_roles"] == [API, SPOOL], verdict
        (roll,) = verdict["rolled_by_failure_domain_map"]
        assert roll["deployment"] == WORKER and roll["pods_replaced"] == 2, roll
        assert {pod["app"] for pod in verdict["pods"]} == {API, WORKER, SPOOL}
        assert all(
            pod["secret_source"] == "mounted-file" and pod["secret_drift"] is False
            for pod in verdict["pods"]
        ), verdict["pods"]
        assert [item["terminating"] for item in verdict["skipped_pods"]] == [True], (
            "the worker Pod the roll replaced is terminating and is not asked"
        )
        assert not LONG_DIGEST.search(json.dumps(verdict)), (
            "the receipt holds digests cut to 12 hex digits only"
        )
        log = _stage_log(tmp_path, stage)
        assert f"registry baseline before {operation}: 5 CPU Pod(s)" in log, log
        assert f"registry convergence #1: PASS -- after {operation}: all 5" in log
        assert re.search(
            rf"  {API}-\S+ uid=\S+ source=mounted-file drift=false", log
        ), "one line per Pod in the stage log"
    for stage in ("1", "4", "5"):
        assert _verdicts(tmp_path, stage) == [], (
            stage,
            "a single-provider site joins nothing inside S1/S5 and S4 is no join",
        )
    lines = _tool_log(env)
    kubectl = [line.split()[1] for line in lines if line.startswith("kubectl ")]
    assert kubectl and set(kubectl) == {"get", "exec"}, (
        "the check only reads the CPU cluster"
    )
    removal = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("gpu-fault-admin remove-cluster ")
    )
    assert lines[removal - 1].startswith("kubectl get pod "), (
        "the baseline is read right before the verb"
    )
    after = [line for line in lines[removal + 1 :] if line.startswith("kubectl ")]
    assert after[0].startswith("kubectl get pod ") and after[1].startswith(
        "kubectl exec -i "
    ), (after[:2], "after the verb the Pods' /healthz is read")


def test_drift_that_clears_inside_the_window_passes(tmp_path: Path) -> None:
    env = {
        **_install_fakes(tmp_path),
        "FAKE_REGISTRY_DRIFT_READS": "2",
        "BOOT029_REGISTRY_POLL_SECONDS": "0",
    }

    result = _run(tmp_path, env, "--last-stage", "3")

    assert result.returncode == 0, result.stdout + result.stderr
    for stage, epoch in (("2", 1), ("3", 2)):
        (verdict,) = _verdicts(tmp_path, stage)
        assert verdict["verdict"] == "PASS", verdict["findings"]
        assert verdict["waited_seconds"] > 0, verdict
        assert all(pod["secret_drift"] is False for pod in verdict["pods"]), verdict
        reads = _healthz_reads(tmp_path, epoch)
        assert len(reads) == 5 and set(reads.values()) == {3}, (
            reads,
            "two drifting reads while kubelet re-projects the Secret, then clean",
        )


def test_drift_that_outlasts_the_window_fails_naming_the_pod(tmp_path: Path) -> None:
    env = {
        **_install_fakes(tmp_path),
        "FAKE_REGISTRY_DRIFT_READS": "always",
        "BOOT029_REGISTRY_WINDOW_SECONDS": "0.3",
        "BOOT029_REGISTRY_POLL_SECONDS": "0.05",
    }

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 2 (remove-cluster)" in result.stderr, result.stderr
    assert re.search(
        rf"FAIL: registry convergence after remove-cluster: {API}-\S+ reports "
        r"secret_drift=true ready=true after the 0\.3 s convergence window "
        r"\(Secret [0-9a-f]{12}, durable head [0-9a-f]{12}\)",
        result.stderr,
    ), result.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy", "remove-cluster"], (
        "nothing runs after the failed check"
    )
    record = _record(tmp_path)
    assert record["stages"]["2"]["status"] == "FAIL" and "3" not in record["stages"]
    (verdict,) = _verdicts(tmp_path, "2")
    assert verdict["verdict"] == "FAIL" and verdict["window_seconds"] == 0.3, verdict
    assert verdict["waited_seconds"] >= 0.3, verdict
    assert len(verdict["findings"]) == 5, verdict["findings"]
    assert min(_healthz_reads(tmp_path, 1).values()) >= 2, (
        "every Pod was read again inside the window before the check gave up"
    )


def test_a_cpu_pod_the_verb_replaced_fails_the_stage(tmp_path: Path) -> None:
    env = {**_install_fakes(tmp_path), "FAKE_CPU_POD_REPLACED": API}

    result = _run(tmp_path, env, "--last-stage", "3")

    assert result.returncode != 0, result.stdout
    assert re.search(
        rf"FAIL: registry convergence after remove-cluster: {API}: Pod\(s\) "
        rf"{API}-\S+ \(uid \S+\) replaced by {API}-\S+-r1 \(uid \S+\); "
        "remove-cluster must not need a restart",
        result.stderr,
    ), result.stderr
    (verdict,) = _verdicts(tmp_path, "2")
    assert verdict["verdict"] == "FAIL", verdict
    assert verdict["unchanged_roles"] == [SPOOL], verdict
    assert all(pod["secret_drift"] is False for pod in verdict["pods"]), (
        "the registry itself converged: the restart alone fails the check"
    )


def test_the_control_worker_rolls_only_for_a_new_failure_domain_map(
    tmp_path: Path,
) -> None:
    env = {**_install_fakes(tmp_path), "FAKE_WORKER_ROLL_WITHOUT_MAP": "1"}

    result = _run(tmp_path, env, "--last-stage", "3")

    assert result.returncode != 0, result.stdout
    assert f"FAIL: registry convergence after remove-cluster: {WORKER}: Pod(s) " in (
        result.stderr
    ), result.stderr
    (verdict,) = _verdicts(tmp_path, "2")
    assert verdict["rolled_by_failure_domain_map"] == [], (
        "a worker roll without a new map digest is no audited roll"
    )
    assert verdict["unchanged_roles"] == [API, SPOOL], verdict


def test_a_product_without_secret_source_is_recorded_not_judged(tmp_path: Path) -> None:
    env = {**_install_fakes(tmp_path), "FAKE_REGISTRY_GENERATION": "pre-secret-source"}

    result = _run(tmp_path, env, "--last-stage", "3")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL:" not in result.stderr, result.stderr
    for stage in ("2", "3"):
        (verdict,) = _verdicts(tmp_path, stage)
        assert verdict["verdict"] == "NOT_APPLICABLE", verdict
        assert verdict["product_generation"] == "pre-secret-source", verdict
        assert "predates the mounted registry Secret" in verdict["reason"], verdict
        assert any("secret_drift=true" in item for item in verdict["findings"]), (
            "what the mounted-Secret rule would have flagged is kept as evidence"
        )
        assert _record(tmp_path)["stages"][stage]["status"] == "PASS"
        assert "registry convergence #1: NOT_APPLICABLE" in _stage_log(tmp_path, stage)


def test_an_unreadable_cpu_plane_stops_the_stage_before_the_verb(
    tmp_path: Path,
) -> None:
    env = {**_install_fakes(tmp_path), "FAKE_KUBECTL_FAIL": "get"}

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert (
        "FAIL: the CPU Pods could not be read before remove-cluster, which did not "
        "run: kubectl get deployment gpu-fault-api-ha exited 1"
    ) in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy"], "remove-cluster never ran"
    assert _verdicts(tmp_path, "2") == [], "no verb ran, so nothing was judged"


def test_the_in_stage_joins_of_a_mixed_site_are_judged_too(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    result = _mixed(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    hyperpod, eks = _sha256(GPU_ARN)[:12], _sha256(EKS_GPU_ARN)[:12]
    expected = {
        "1": [("join-cluster", [eks])],
        "2": [("remove-cluster", [eks]), ("remove-cluster", [hyperpod])],
        "3": [("join-cluster", [hyperpod, eks])],
        "4": [],
        "5": [("join-cluster", [hyperpod])],
    }
    for stage, operations in expected.items():
        verdicts = _verdicts(tmp_path, stage)
        assert [
            (item["operation"], item["gpu_arns_sha256_12"]) for item in verdicts
        ] == operations, (stage, verdicts)
        assert [item["sequence"] for item in verdicts] == list(
            range(1, len(operations) + 1)
        ), (stage, "one verdict per verb, in order")
        assert {item["verdict"] for item in verdicts} <= {"PASS"}, verdicts
    lines = _tool_log(env)
    joins = [
        index
        for index, line in enumerate(lines)
        if line.startswith("gpu-fault-admin join-cluster ")
    ]
    assert lines[joins[0] - 1].startswith("kubectl get pod "), (
        "S1's baseline precedes the join that stops for the profile plan"
    )
    assert "--approve-profile-plan" in lines[joins[1]], lines[joins[1]]
    assert lines[joins[1] + 1].startswith("kubectl get pod "), (
        "the check follows the approved rerun, not the stopped attempt"
    )
