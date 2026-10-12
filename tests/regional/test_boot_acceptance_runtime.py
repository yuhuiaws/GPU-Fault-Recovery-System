"""BOOT-011/012/015 runtime runner contracts from the 2026-09-07 review."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.site_cluster_profile import cluster_runtime_profile_version
from scripts.e2e.regional import boot_acceptance_runtime as runtime
from scripts.e2e.regional.acceptance_runner_common import runtime_profile_resolution
from tests.regional._cov95_boot_owner import OwnerFixture
from tests.regional._cov95_cap_cases import Clock


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["x"], returncode, stdout=stdout, stderr=stderr)


# -- item 7: BOOT-012 reads the CA contract, not the secretKeyRef one ----------


def test_ca_file_contract_matches_the_manifest_against_itself() -> None:
    manifest = runtime.manifest_deployment()

    contract = runtime.ca_file_contract(manifest, copy.deepcopy(manifest))

    assert contract["passed"] is True, contract
    assert contract["ca_file"] == "/etc/gpu-fault/tls/ca.crt"
    assert contract["manifest_mount"]["read_only"] is True
    assert contract["manifest_mount"]["secret_name"] == "gpu-fault-regional-connection"
    assert contract["forbidden_trust_env_present"] == []


def test_ca_file_contract_rejects_drift_and_global_trust_overrides() -> None:
    manifest = runtime.manifest_deployment()
    container = lambda doc: doc["spec"]["template"]["spec"]["containers"][0]  # noqa: E731

    moved = copy.deepcopy(manifest)
    for item in container(moved)["env"]:
        if item["name"] == runtime.CA_FILE_ENV:
            item["value"] = "/etc/ssl/certs/ca.crt"
    assert runtime.ca_file_contract(manifest, moved)["passed"] is False

    unmounted = copy.deepcopy(manifest)
    container(unmounted)["volumeMounts"] = [
        mount
        for mount in container(unmounted)["volumeMounts"]
        if mount["name"] != "control-plane-ca"
    ]
    assert runtime.ca_file_contract(manifest, unmounted)["passed"] is False

    overridden = copy.deepcopy(manifest)
    container(overridden)["env"].append(
        {"name": "SSL_CERT_FILE", "value": "/etc/gpu-fault/tls/ca.crt"}
    )
    contract = runtime.ca_file_contract(manifest, overridden)
    assert contract["forbidden_trust_env_present"] == ["SSL_CERT_FILE"]
    assert contract["passed"] is False


def test_readiness_matrix_verdict_requires_every_replica_and_status() -> None:
    matrix = {
        "valid": {"status": 200},
        "wrong_token": {"status": 403},
        "wrong_pin": {"status": 503},
        "no_owner": {"status": 503},
        "stale_claim": {"status": 503},
    }
    verdict = runtime.readiness_matrix_verdict(
        [
            {"pod": "a", "readiness_matrix": matrix},
            {"pod": "b", "readiness_matrix": matrix},
        ]
    )
    assert verdict["verdict"] == "PASS"
    assert verdict["checks"]["stale_claim_503"] is True

    partial = runtime.readiness_matrix_verdict(
        [
            {"pod": "a", "readiness_matrix": matrix},
            {"pod": "b", "readiness_matrix": None},
        ]
    )
    assert partial["verdict"] == "FAIL"
    assert partial["checks"]["matrix_recorded_per_replica"] is False

    wrong = runtime.readiness_matrix_verdict(
        [{"pod": "a", "readiness_matrix": {**matrix, "wrong_token": {"status": 200}}}]
    )
    assert wrong["checks"]["wrong_token_403"] is False
    assert wrong["verdict"] == "FAIL"


# -- item 7: BOOT-011 readiness timing and production lease owners -------------


def _pod(name: str, created: str, ready_at: str | None) -> dict[str, Any]:
    conditions = []
    if ready_at is not None:
        conditions.append(
            {"type": "Ready", "status": "True", "lastTransitionTime": ready_at}
        )
    return {
        "metadata": {"name": name, "creationTimestamp": created},
        "status": {"conditions": conditions},
    }


def test_ready_within_needs_replicas_and_a_ready_transition_inside_the_limit() -> None:
    fast = {"items": [_pod("a", "2026-09-07T10:00:00Z", "2026-09-07T10:01:30Z")]}
    assert runtime.ready_within(fast, replicas=1)["passed"] is True
    assert runtime.ready_within(fast, replicas=1)["ready_seconds"] == {"a": 90.0}

    slow = {"items": [_pod("a", "2026-09-07T10:00:00Z", "2026-09-07T10:04:00Z")]}
    assert runtime.ready_within(slow, replicas=1)["passed"] is False

    # Zero replicas with zero Ready Pods used to count as ready.
    assert runtime.ready_within({"items": []}, replicas=0)["passed"] is False

    short = {"items": [_pod("a", "2026-09-07T10:00:00Z", "2026-09-07T10:00:10Z")]}
    assert runtime.ready_within(short, replicas=2)["passed"] is False

    never = {"items": [_pod("a", "2026-09-07T10:00:00Z", None)]}
    assert runtime.ready_within(never, replicas=1)["passed"] is False


def test_foreign_lease_owners_recognise_the_isolated_executor_identity() -> None:
    owners = [
        "prod-cluster/executor-prod-1",
        "iso-cluster/executor-iso-abc",
        "executor-iso-def",
        "other/executor-iso-def",
    ]

    foreign = runtime.foreign_lease_owners(
        owners, isolated_cluster_id="iso-cluster", isolated_pods=["executor-iso-def"]
    )

    assert foreign == [
        "executor-iso-def",
        "iso-cluster/executor-iso-abc",
        "other/executor-iso-def",
    ]
    assert (
        runtime.foreign_lease_owners(
            ["prod-cluster/executor-prod-1"],
            isolated_cluster_id="iso-cluster",
            isolated_pods=["executor-iso-def"],
        )
        == []
    )


# -- item 2: BOOT-015 alert window and step-isolated cleanup -------------------


class _Regional:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def cpu_python(
        self, script: str, *arguments: str, **_kwargs: Any
    ) -> dict[str, Any]:
        for name, value in self.responses.items():
            if script == getattr(runtime, name):
                self.calls.append(name)
                if isinstance(value, BaseException):
                    raise value
                return dict(value)
        raise AssertionError(f"unexpected probe: {script[:60]!r}")

    def kubectl(self, *arguments: str, **_kwargs: Any) -> str:
        return ""

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "r1", "cluster_id": "cluster-a"}


class _Fixture:
    def __init__(
        self, regional: _Regional, *, exec_error: Exception | None = None
    ) -> None:
        self.regional = regional
        self.cluster_id = "cluster-a"
        self.config = {"runtime_profile": {"version": "profile-a"}}
        self.exec_error = exec_error
        self.exec_calls = 0

    def pods(self, plane: str, app: str) -> list[str]:
        return ["executor-a"]

    def pod_json(self, plane: str, pod: str, script: str, *args: str) -> dict[str, Any]:
        return {"owners": ["gpu-fault-kubernetes-adapter"]}

    def exec(self, *arguments: str, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.exec_calls += 1
        if self.exec_error is not None:
            raise self.exec_error
        return _completed(0)


def test_wait_amp_alert_honours_a_deadline_set_before_the_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "amp_request",
        lambda **kwargs: polls.append(kwargs["path"])
        or {"data": {"result": [], "alerts": []}},
    )
    monkeypatch.setattr(
        runtime.time, "sleep", lambda _s: pytest.fail("no sleep past the deadline")
    )
    fixture = type(
        "F",
        (),
        {
            "config": {"health": {"amp_workspace_id": "ws"}},
            "region": "us-west-2",
            "cluster_id": "cluster-a",
        },
    )()

    result = runtime.wait_amp_alert(
        fixture,  # type: ignore[arg-type]
        expect_firing=True,
        timeout_seconds=300,
        deadline=runtime.time.monotonic() - 1,
    )

    assert result["matched"] is False
    assert len(result["timeline"]) == 1, "one poll at an expired deadline, no more"
    assert polls == ["/api/v1/query", "/api/v1/alerts"]


def test_boot015_cleanup_records_each_step_and_verifies_deletion_by_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "wait_amp_alert", lambda *a, **k: {"matched": False, "timeline": []}
    )
    regional = _Regional(
        {
            "REMOTE_DELETE_PROBE": RuntimeError("exec channel dropped"),
            "REMOTE_BASELINE_PROBE": {"ids": ["remote-other"], "count": 1},
        }
    )
    fixture = _Fixture(regional)

    cleanup = runtime.boot015_cleanup(
        fixture,  # type: ignore[arg-type]
        command_id="remote-boot015-1-1",
        pods=["executor-a"],
        baseline_ids={"remote-other"},
    )

    assert "exec channel dropped" in cleanup["delete"]["error"]
    assert cleanup["synthetic_command_removed"] is True
    assert cleanup["commands_introduced"] == []
    assert cleanup["readiness_recovered"] == {"executor-a": True}
    # A removed command does not by itself prove the alert has resolved.
    assert cleanup["resolve_observed"] == {"matched": False, "timeline": []}
    assert cleanup["passed"] is False
    assert fixture.exec_calls == 1, "readiness ran although the delete step failed"


def test_boot015_cleanup_fails_only_when_the_injected_command_survives() -> None:
    regional = _Regional(
        {
            "REMOTE_DELETE_PROBE": {"remaining": 2},
            "REMOTE_BASELINE_PROBE": {
                "ids": ["remote-other", "remote-boot015-1-1"],
                "count": 2,
            },
        }
    )

    cleanup = runtime.boot015_cleanup(
        _Fixture(regional),  # type: ignore[arg-type]
        command_id="remote-boot015-1-1",
        pods=["executor-a"],
        baseline_ids={"remote-other"},
    )

    assert cleanup["synthetic_command_removed"] is False
    assert cleanup["commands_introduced"] == ["remote-boot015-1-1"]
    assert "resolve_observed" not in cleanup
    assert cleanup["passed"] is False


def test_boot015_writes_details_and_cleans_up_when_a_probe_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "run", lambda *a, **k: _completed(0))
    monkeypatch.setattr(
        runtime, "wait_amp_alert", lambda *a, **k: {"matched": True, "timeline": []}
    )
    regional = _Regional(
        {
            "PROFILE_OWNER_PROBE": {
                "profiles": [{"profile_version": "profile-a", "orphan_owners": []}]
            },
            "REMOTE_BASELINE_PROBE": {"ids": [], "count": 0},
            "REMOTE_INJECT_PROBE": {"command_id": "x", "status": "PENDING"},
            "REMOTE_DELETE_PROBE": {"remaining": 0},
        }
    )
    fixture = _Fixture(regional, exec_error=RuntimeError("readiness exec dropped"))

    with pytest.raises(RuntimeError, match="readiness exec dropped"):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)  # type: ignore[arg-type]

    details = json.loads(
        (tmp_path / "boot015-details.json").read_text(encoding="utf-8")
    )
    assert details["verdict"] == "FAIL"
    assert "REMOTE_INJECT_PROBE" in regional.calls
    assert regional.calls.index("REMOTE_DELETE_PROBE") > regional.calls.index(
        "REMOTE_INJECT_PROBE"
    ), "the injected command is deleted even though the case body raised"
    assert details["cleanup"]["synthetic_command_removed"] is True
    assert any(
        "requires bounded AMP alert resolution" in item
        for item in details["limitations"]
    ), "cleanup must require alert resolution"


def test_boot015_metric_gap_is_short_and_windows_are_named() -> None:
    """The gap outlives the worker's metric scan cache; the windows stay named.

    The remote-command gauges come from a scan cache shared for 60 s per
    process, so two scrapes 5 s apart could read one snapshot (live 2026-09-13:
    first scrape 0 pending, second 1 pending 622 s old). The samples are polled
    against a bounded deadline instead of taken blind.
    """

    assert runtime.METRIC_SAMPLE_GAP_SECONDS > runtime.METRIC_SCAN_TTL_SECONDS, (
        "the second sample must land after the scan cache expired"
    )
    assert (
        runtime.METRIC_SAMPLE_DEADLINE_SECONDS >= 2 * runtime.METRIC_SCAN_TTL_SECONDS
    ), "polling must outlive at least two cache lifetimes"
    assert runtime.AMP_FIRING_WINDOW_SECONDS == 300, "the alert window is unchanged"
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "time.sleep(30)" not in source
    assert '"pending_counted": second["pending"] >= 1' in source
    assert 'state["lease_owner"] is None' in source


def test_empty_ca_rejection_accepts_both_openssl_refusals() -> None:
    """An empty CA file may fail at handshake (OpenSSL < 3.5) or at load (3.5).

    Live 2026-09-13: the cold image build pulled OpenSSL 3.5, the executor client
    refused the empty PEM with X509: NO_CERTIFICATE_OR_CRL_FOUND before any TLS
    handshake, and BOOT-012 read that stronger refusal as "not rejected".
    """

    assert runtime.empty_ca_rejection(
        "ssl.SSLError: [X509: NO_CERTIFICATE_OR_CRL_FOUND]"
    ), "the trust store refusing an empty PEM is a rejection"
    assert runtime.empty_ca_rejection(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    ), "a handshake that fails verification is a rejection"
    assert not runtime.empty_ca_rejection("Traceback ... ConnectionRefusedError"), (
        "an unrelated failure must not pass for a CA rejection"
    )


# -- mixed-provider site: BOOT-015 probes the target cluster's effective profile ----

# The round-8 S5 site: the plain-EKS cluster was deployed first, so the site base
# is eks-v1 and the HyperPod cluster carries its own spec.clusters[].runtimeProfile.
MIXED_SITE: dict[str, Any] = {
    "runtime_profile": {"version": "eks-v1"},
    "clusters": [
        {"cluster_id": "hp-cluster", "runtime_profile": {"version": "hyperpod-v1"}},
        {"cluster_id": "gpu-eks-cluster"},
    ],
}


def test_runtime_profile_resolution_prefers_the_clusters_own_pin_over_the_site() -> (
    None
):
    pinned = runtime_profile_resolution(MIXED_SITE, "hp-cluster")
    assert pinned == {
        "cluster_id": "hp-cluster",
        "site_version": "eks-v1",
        "effective_version": "hyperpod-v1",
        "source": "cluster",
    }
    base = runtime_profile_resolution(MIXED_SITE, "gpu-eks-cluster")
    assert base == {
        "cluster_id": "gpu-eks-cluster",
        "site_version": "eks-v1",
        "effective_version": "eks-v1",
        "source": "site",
    }
    assert runtime_profile_resolution(MIXED_SITE, "unknown")["source"] == "site", (
        "a cluster the site does not list follows the site, as the product does"
    )
    for cluster_id in ("hp-cluster", "gpu-eks-cluster", "unknown"):
        assert runtime_profile_resolution(MIXED_SITE, cluster_id)[
            "effective_version"
        ] == cluster_runtime_profile_version(MIXED_SITE, cluster_id), (
            "the runner must resolve exactly as the product's own rule does"
        )


@pytest.mark.parametrize(
    "config",
    [
        {"clusters": []},
        {"runtime_profile": {"version": ""}},
        {"runtime_profile": {"version": 1}},
        {
            "runtime_profile": {"version": "eks-v1"},
            "clusters": [
                {"cluster_id": "hp-cluster", "runtime_profile": {"version": ""}}
            ],
        },
        {
            "runtime_profile": {"version": "eks-v1"},
            "clusters": [
                {"cluster_id": "hp-cluster", "runtime_profile": {"version": 2}}
            ],
        },
    ],
)
def test_runtime_profile_resolution_requires_exact_versions(
    config: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="exact Runtime Profile version"):
        runtime_profile_resolution(config, "hp-cluster")


class _MixedSiteOwnerFixture(OwnerFixture):
    """The cov95 PASS harness on the mixed site; the profile probe echoes the
    version it was asked for and records it."""

    def __init__(
        self, clock: Clock, cluster_id: str, *, orphans: list[str] | None = None
    ) -> None:
        super().__init__(clock)
        self.cluster_id = cluster_id
        self.config.update(copy.deepcopy(MIXED_SITE))
        self.orphans = orphans or []
        self.profile_requests: list[str] = []

    def cpu_python(self, script: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if script == runtime.PROFILE_OWNER_PROBE:
            self.profile_requests.append(args[0])
            return {
                "profiles": [
                    {"profile_version": args[0], "orphan_owners": self.orphans}
                ]
            }
        return super().cpu_python(script, *args, **kwargs)


def _pass_harness(monkeypatch: pytest.MonkeyPatch, fixture: OwnerFixture) -> None:
    monkeypatch.setattr(runtime, "time", fixture.clock)
    monkeypatch.setattr(runtime, "amp_request", fixture.amp_request)
    monkeypatch.setattr(runtime, "run", lambda args, **_k: _completed(0))


@pytest.mark.parametrize(
    ("cluster_id", "effective", "source"),
    [("hp-cluster", "hyperpod-v1", "cluster"), ("gpu-eks-cluster", "eks-v1", "site")],
)
def test_boot015_probes_the_target_clusters_effective_profile_and_records_both(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cluster_id: str,
    effective: str,
    source: str,
) -> None:
    """Live: the round-8 two-provider run passed BOOT-015 on the plain-EKS
    cluster and failed it in 22 s on the HyperPod cluster with "Runtime Profile
    remote owner coverage is incomplete" -- the check probed the site base
    (eks-v1) for a cluster pinned to hyperpod-v1, so every HyperPod owner was an
    orphan of a profile the cluster does not run."""

    fixture = _MixedSiteOwnerFixture(Clock(), cluster_id)
    _pass_harness(monkeypatch, fixture)

    result = runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)

    assert fixture.profile_requests == [effective], (
        "the owner probe must read the profile the target cluster runs"
    )
    assert result["verdict"] == "PASS", result["checks"]
    expected = {
        "cluster_id": cluster_id,
        "site_version": "eks-v1",
        "effective_version": effective,
        "source": source,
    }
    assert result["runtime_profile"] == expected
    details = json.loads((tmp_path / "boot015-details.json").read_text("utf-8"))
    assert details["runtime_profile"] == expected, (
        "the evidence names the site base beside the version actually judged"
    )


def test_boot015_coverage_failure_names_the_judged_and_site_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _MixedSiteOwnerFixture(
        Clock(), "hp-cluster", orphans=["gpu-fault-hyperpod-adapter"]
    )
    _pass_harness(monkeypatch, fixture)

    with pytest.raises(
        runtime.BootAcceptanceError,
        match=(
            r"coverage is incomplete for hyperpod-v1 \(the cluster profile of "
            r"hp-cluster; site base eks-v1\)"
        ),
    ):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)
    assert fixture.profile_requests == ["hyperpod-v1"]
    assert "insert" not in fixture.probes, "a failed precondition injects nothing"
    assert not (tmp_path / "boot015-details.json").exists(), (
        "a rejected precondition leaves no mutation evidence"
    )


@pytest.mark.parametrize("pin", ["", 3])
def test_boot015_refuses_an_inexact_cluster_profile_pin_before_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pin: Any
) -> None:
    fixture = _MixedSiteOwnerFixture(Clock(), "hp-cluster")
    fixture.config["clusters"][0]["runtime_profile"]["version"] = pin
    _pass_harness(monkeypatch, fixture)

    with pytest.raises(runtime.BootAcceptanceError, match="exact Runtime Profile"):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)
    assert fixture.profile_requests == [] and "insert" not in fixture.probes
