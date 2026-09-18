from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace
from urllib.response import addinfourl

import pytest

from scripts.e2e.regional import run_boot019_admin_lifecycle as runner
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from tests.regional.test_boot019_admin_lifecycle import FakeBackend


def backend(tmp_path, monkeypatch):
    token = tmp_path / "fixture-token"
    token.write_bytes(b"fixture-only-test-token")
    (tmp_path / "fixture-ca").write_bytes(b"fixture-trust-root")
    site = SimpleNamespace(
        release_config={
            "cpu_kubeconfig": str(tmp_path / "empty-kubeconfig"),
            "namespace": "fixture",
            "clusters": [
                {
                    "cluster_id": "joined",
                    "token_file": str(token),
                    "control_plane_url": "https://fixture.invalid",
                    "ca_file": str(tmp_path / "fixture-ca"),
                }
            ],
        }
    )
    monkeypatch.setattr(runner, "load_site", lambda *_a, **_k: site)
    value = runner.LiveAdminLifecycleBackend(
        site_path=tmp_path / "site.json",
        gpu_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/joined",
        cluster_id="joined",
        allowed_namespaces=("fixture",),
        join_state_dir=tmp_path / "join-state",
        run_dir=tmp_path,
    )
    return value, site


@pytest.mark.parametrize(
    ("fault", "phase"),
    [
        ("before-activation", "ROLLED_BACK"),
        ("after-activation", "FAILED_AFTER_ACTIVATION"),
    ],
)
def test_completed_fault_boundary_is_reused_without_repeating_join(
    fault, phase, tmp_path, monkeypatch
) -> None:
    value, _site = backend(tmp_path, monkeypatch)
    state = {"phase": phase, "evidence": {"DISCOVERED": {"cluster_id": "joined"}}}
    value.join_state_dir.mkdir()
    (value.join_state_dir / "state.json").write_text(json.dumps(state))

    def unexpected(_request):
        raise AssertionError("a completed injected boundary must not repeat join")

    monkeypatch.setattr(runner, "join_cluster", unexpected)
    assert value.join(fault) == {**state, "cluster_id": "joined"}, (
        "resume must return the durable activation-boundary record"
    )


def test_unknown_join_fault_is_rejected_before_any_site_read(
    tmp_path, monkeypatch
) -> None:
    value, _site = backend(tmp_path, monkeypatch)
    with pytest.raises(runner.AcceptanceCheckError, match="unknown join fault"):
        value.join("unrecognized")
    assert not value.join_state_dir.exists(), (
        "invalid injection must not create a journal"
    )


def test_plain_join_calls_the_public_request_without_injection(
    tmp_path, monkeypatch
) -> None:
    value, site = backend(tmp_path, monkeypatch)
    requests = []
    monkeypatch.setattr(
        runner,
        "join_cluster",
        lambda request: requests.append(request)
        or {"phase": "COMPLETED", "cluster_id": "joined"},
    )
    assert value.join() == {"phase": "COMPLETED", "cluster_id": "joined"}, (
        "uninjected join must preserve the backend's completion result"
    )
    assert requests[0].site is site, "the join must bind the configured managed site"
    assert requests[0].allowed_namespaces == ("fixture",), (
        "the namespace allowlist must not expand"
    )


def test_snapshot_refuses_nonobject_resource_readback(tmp_path, monkeypatch) -> None:
    value, _site = backend(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "RegionalLiveFixture",
        SimpleNamespace(
            run=lambda command, **_kwargs: subprocess.CompletedProcess(
                command, 0, "[]", ""
            )
        ),
    )
    with pytest.raises(runner.AcceptanceCheckError, match="JSON object"):
        value.snapshot()


def test_missing_capture_refuses_revocation_probe_before_transport(
    tmp_path, monkeypatch
) -> None:
    value, _site = backend(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="another run"):
        value.probe_revoked_token({"cluster_id": "joined"})


def test_unexpected_success_is_observable_and_sensitive_cleanup_clears_capture(
    tmp_path, monkeypatch
) -> None:
    value, _site = backend(tmp_path, monkeypatch)
    capture = value.capture_joined_token("joined")
    requests = []
    monkeypatch.setattr(
        runner.ssl, "create_default_context", lambda **_kwargs: object()
    )

    def request(message, **kwargs):
        requests.append((message, kwargs))
        return addinfourl(io.BytesIO(b"{}"), {}, message.full_url, 200)

    monkeypatch.setattr(runner.urllib.request, "urlopen", request)
    assert value.probe_revoked_token(capture) == {
        "status": 200,
        "detail": "unexpected success",
    }, (
        "a revoked credential unexpectedly accepted by the fake API must remain a failure observation"
    )
    assert requests[0][1]["timeout"] == 20, (
        "the revocation probe needs a finite timeout"
    )
    secure = tmp_path / "secure"
    secure.mkdir(exist_ok=True)
    legacy = secure / "old.revoked-token"
    legacy.write_text("fixture-only-legacy")
    kept = secure / "unrelated.txt"
    kept.write_text("keep")
    value.cleanup_sensitive_files()
    assert legacy.exists() and kept.exists(), (
        "cleanup cannot suffix-sweep unrelated credentials or evidence"
    )
    value.cleanup_sensitive_files(completed=True)
    with pytest.raises(ValueError, match="unavailable"):
        value.probe_revoked_token(capture)


def test_remove_and_uninstall_forward_only_guarded_public_requests(
    tmp_path, monkeypatch
) -> None:
    value, site = backend(tmp_path, monkeypatch)
    requests = []
    final = tmp_path / "final.json"
    final.write_text(
        json.dumps(
            {
                "resources": [
                    None,
                    {"resource_key": "cluster/joined/eks", "status": "PRESERVED"},
                ]
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "remove_cluster",
        lambda request: (requests.append(request) or {"phase": "COMPLETED"}),
    )
    monkeypatch.setattr(
        runner,
        "uninstall",
        lambda request: (requests.append(request) or {"final_registry": str(final)}),
    )
    assert value.remove("joined") == {"phase": "COMPLETED"}, (
        "preserve the removal result"
    )
    assert value.uninstall()["final_registry_statuses"] == {
        "cluster/joined/eks": "PRESERVED"
    }
    assert (
        requests[0].site is site
        and requests[0].confirmation == runner.REMOVE_CONFIRMATION
    )
    assert (
        requests[1].cpu_disposition == "keep" and requests[1].reset_database is False
    ), (
        "the isolated lifecycle must not reset a preserved database or remove CPU infrastructure"
    )


@pytest.mark.parametrize("document", [None, {}, {"resources": None}])
def test_final_registry_requires_explicit_snapshot_rows(document, tmp_path) -> None:
    if document is None:
        result = {}
    else:
        path = tmp_path / "final.json"
        path.write_text(json.dumps(document))
        result = {"final_registry": str(path)}
    with pytest.raises(
        runner.AcceptanceCheckError, match="final registry|resources list"
    ):
        runner.final_registry_statuses(result)


@pytest.mark.parametrize(
    "failure", ["same-site", "invalid-arn", "empty-baseline", "empty-protected"]
)
def test_epoch_isolation_needs_explicit_nonoverlapping_physical_targets(
    failure, tmp_path, monkeypatch
) -> None:
    def arn(name):
        return f"arn:aws:eks:us-west-2:000000000000:cluster/{name}"

    left, right = tmp_path / "disposable", tmp_path / "protected"
    disposable = {
        "cpu_eks_arn": arn("cpu-a"),
        "clusters": [{"eks_cluster_arn": arn("gpu-a")}],
    }
    protected = {
        "cpu_eks_arn": arn("cpu-b"),
        "clusters": [{"eks_cluster_arn": arn("gpu-b")}],
    }
    if failure == "empty-baseline":
        disposable["clusters"] = []
    elif failure == "empty-protected":
        protected["clusters"] = []
    monkeypatch.setattr(
        runner,
        "load_site",
        lambda path, **_k: SimpleNamespace(
            release_config=disposable if path == left else protected
        ),
    )
    with pytest.raises(
        runner.AcceptanceCheckError, match="protected|physical|disposable"
    ):
        runner.epoch_targets(
            left,
            left if failure == "same-site" else right,
            "not-an-arn" if failure == "invalid-arn" else arn("gpu-new"),
        )


def test_lifecycle_resume_uses_durable_activation_stages(tmp_path) -> None:
    value = FakeBackend()
    recorder = EvidenceRecorder(
        tmp_path / "case.json", case_id=runner.CASE_ID, inputs={"fixture": True}
    )
    recorder.stage("baseline", value.snapshot)
    recorder.stage(
        "join_failure_before_activation", lambda: value.join("before-activation")
    )
    recorder.stage(
        "join_failure_after_activation", lambda: value.join("after-activation")
    )
    result = runner.run_admin_lifecycle(value, recorder)
    assert result["verdict"] == "PASS", "durable activation stages must be resumable"
    assert value.calls.count("join:before-activation") == 1, (
        "resume must not repeat an injected join"
    )
    assert value.calls.count("join:after-activation") == 1, (
        "activation intent must not be replayed"
    )


def test_sensitive_cleanup_failure_prevents_completion(tmp_path) -> None:
    class CleanupFailure(FakeBackend):
        def cleanup_sensitive_files(self, *, completed=False):
            super().cleanup_sensitive_files(completed=completed)
            raise RuntimeError("fixture sensitive cleanup failed")

    recorder = EvidenceRecorder(
        tmp_path / "case.json", case_id=runner.CASE_ID, inputs={"fixture": True}
    )
    with pytest.raises(RuntimeError, match="sensitive cleanup failed"):
        runner.run_admin_lifecycle(CleanupFailure(), recorder)
    result = json.loads((tmp_path / "case.json").read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED", (
        "correct lifecycle observations cannot override an unproved sensitive cleanup"
    )
