from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import release_state
from gpu_fault.admin.site import RenderedSite, SiteConfigError, load_site
from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_capacity_acceptance as runner
from scripts.e2e.regional.capacity_acceptance_base import CapCoreHarness, CapError
from scripts.e2e.regional.regional_case_contract import predecessor_path
from tests.admin.test_admin_site import site_file


@dataclass
class CapacityCLI:
    site_file: Path
    kubeconfig: Path
    predecessor: Path
    argv: list[str]
    case_id: str
    state: dict[str, Any] = field(
        default_factory=lambda: {"release_id": "unit-release"}
    )
    loads: list[RenderedSite] = field(default_factory=list)
    reads: list[list[str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    harness_arguments: dict[str, Any] = field(default_factory=dict)
    scrape_source: dict[str, Any] = field(
        default_factory=lambda: {
            "image": "unit-registry/collector@sha256:" + "b" * 64,
            "config_sha256": "c" * 64,
            "service_account_uid": "unit-collector-account",
        }
    )
    scrape_reads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def plan_path(self) -> Path:
        return self.site_file.parent / "cases" / self.case_id / "plan.json"

    def approve(self) -> None:
        confirmation = self.case_id.removeprefix("GF-REGIONAL-").replace("-", "")
        self.argv.extend(
            [
                "--execute",
                "--confirm",
                f"{confirmation}_EXECUTE",
                "--maintenance-window-end",
                (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            ]
        )


def capacity_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_id: str = "GF-REGIONAL-CAP-001",
) -> CapacityCLI:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["clusters"] = []
    path.write_text(yaml.safe_dump(document))
    site = load_site(path)
    kubeconfig = Path(site.release_config["cpu_kubeconfig"])
    kubeconfig.write_text("unit-only-private-cpu-kubeconfig\n")
    predecessor_id, predecessor = predecessor_path(tmp_path, case_id, "")
    assert predecessor_id is not None and predecessor is not None, (
        "CAP cases must retain their formal predecessor"
    )
    predecessor.parent.mkdir(parents=True, exist_ok=True)
    predecessor.write_text(
        json.dumps(
            {
                "case_id": predecessor_id,
                "verdict": "PASS",
                "status": "COMPLETED",
                "release_id": "unit-release",
                "cluster_id": (
                    site.release_config["cpu_eks_arn"]
                    if predecessor_id in runner.CASE_IDS
                    else "preceding-case-gpu-cluster"
                ),
            }
        )
    )
    argv = [
        "capacity",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(path),
        "--case",
        case_id,
    ]
    cli = CapacityCLI(path, kubeconfig, predecessor, argv, case_id)

    def read_site(source: Path) -> RenderedSite:
        loaded = load_site(source)
        cli.loads.append(loaded)
        return loaded

    def read_release(
        arguments: Sequence[str], **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        assert list(arguments[:4]) == [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
        ], "release identity must be read through the site's actual CPU connection"
        assert list(arguments[5:]) == [
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        ], "identity preflight may only read the existing release ConfigMap"
        cli.reads.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps({"data": {"state.json": json.dumps(cli.state)}}),
            "",
        )

    def harness(**kwargs: Any) -> SimpleNamespace:
        cli.events.append("constructed")
        cli.harness_arguments.update(kwargs)

        def run() -> int:
            cli.events.append("run")
            return 0

        return SimpleNamespace(run=run)

    def capture_scrape_source(**kwargs: Any) -> dict[str, Any]:
        cli.scrape_reads.append(kwargs)
        return dict(cli.scrape_source)

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _mask: None))
    monkeypatch.setattr(runner, "load_site", read_site)
    monkeypatch.setattr(runner, "CapHarness", harness)
    monkeypatch.setattr(runner, "capture_scrape_source", capture_scrape_source)
    monkeypatch.setattr(release_state, "run_command", read_release)
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SITE_PROFILE", raising=False)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    monkeypatch.setenv("GPU_FAULT_CONTROL_KUBECONFIG", str(tmp_path / "unused-cpu"))
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "unused-gpu"))
    return cli


@pytest.mark.parametrize("case_id", runner.CASE_IDS)
def test_capacity_plan_and_execution_bind_only_cpu_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case_id: str,
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path, case_id)
    cli.state["private_data"] = "unit-only-private-release-state"
    assert runner.main() == 0, "a CPU-only site must support every CAP probe"
    plan = json.loads(cli.plan_path.read_text())
    site = cli.loads[0]
    assert plan["connections"] == {
        "environment:GPU_FAULT_CONTROL_KUBECONFIG": {
            "path": str(cli.kubeconfig),
            "sha256": hashlib.sha256(cli.kubeconfig.read_bytes()).hexdigest(),
        }
    }, "the guard must hash actual CPU content, with no GPU connection requirement"
    assert (
        plan["details"]["site_sha256"]
        == hashlib.sha256(cli.site_file.read_bytes()).hexdigest()
    ), "the plan must bind the parsed site bytes"
    assert plan["details"]["site_identity"] == {
        key: site.release_config[key]
        for key in ("site_name", "aws_region", "cpu_eks_arn", "namespace")
    }, "CPU identity must include its site, Region, EKS ARN and namespace"
    assert plan["details"]["release_id"] == "unit-release", (
        "release identity must come from the live CPU state"
    )
    assert cli.events == [], (
        "planning must not construct a credential-generating harness"
    )
    output = cli.plan_path.read_text() + capsys.readouterr().out
    assert "unit-only-private" not in output, (
        "neither kubeconfig contents nor unrelated release state may be serialized"
    )
    cli.approve()
    assert runner.main() == 0, "unchanged approved CPU inputs should execute"
    assert cli.events == ["constructed", "run"], (
        "the mutation harness is constructed only after authorization"
    )
    assert len(cli.loads) == len(cli.reads) == 2, (
        "execution must reload both the site and the deployed release"
    )
    assert cli.harness_arguments["rendered_site"] is cli.loads[-1], (
        "the harness must consume the site snapshot that was just authorized"
    )
    assert cli.harness_arguments["evidence_identity"] == {
        "release_id": "unit-release",
        "cluster_id": site.release_config["cpu_eks_arn"],
    }, "CAP evidence must identify the release and CPU cluster for its successor"
    assert isinstance(cli.harness_arguments["maintenance_deadline"], datetime), (
        'test_capacity_plan_and_execution_bind_only_cpu_inputs: expected isinstance(cli.harness_arguments["maintenance_deadline"], datetime)'
    )
    if case_id == "GF-REGIONAL-CAP-002":
        assert plan["details"]["scrape_source_binding"] == cli.scrape_source
        assert cli.harness_arguments["scrape_source_binding"] == cli.scrape_source
        assert (
            cli.scrape_reads
            == [
                {
                    "kubeconfig": cli.kubeconfig,
                    "namespace": site.release_config["namespace"],
                    "region": site.release_config["aws_region"],
                    "workspace_id": site.release_config["health"]["amp_workspace_id"],
                }
            ]
            * 2
        ), "CAP002 must reread collector authority before execution"
    else:
        assert cli.scrape_reads == [], "other capacity cases do not need an AMP scraper"
        assert cli.harness_arguments["scrape_source_binding"] is None


@pytest.mark.parametrize("field", ["image", "config_sha256", "service_account_uid"])
def test_capacity_rejects_scraper_identity_drift_before_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path, "GF-REGIONAL-CAP-002")
    assert runner.main() == 0
    cli.scrape_source[field] = "changed-collector-identity"
    cli.approve()
    with pytest.raises(RuntimeError, match="drifted at details"):
        runner.main()
    assert cli.events == [], "collector drift cannot authorize any probe resources"


def test_capacity_collector_read_failure_cannot_authorize_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path, "GF-REGIONAL-CAP-002")
    assert runner.main() == 0
    cli.approve()

    def unavailable(**kwargs: Any) -> dict[str, Any]:
        raise CapError("collector authority could not be read")

    monkeypatch.setattr(runner, "capture_scrape_source", unavailable)
    with pytest.raises(CapError, match="collector authority"):
        runner.main()
    assert cli.events == [], "an unreadable collector cannot authorize mutation"


@pytest.mark.parametrize("case_id", runner.CASE_IDS)
@pytest.mark.parametrize(
    "changed", ["kubeconfig", "site", "cpu_identity", "release", "predecessor"]
)
def test_capacity_rejects_same_path_input_drift_before_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str, changed: str
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path, case_id)
    assert runner.main() == 0, "the original identity must pass preflight"
    if changed == "release":
        cli.state["release_id"] = "another-release"
    elif changed == "cpu_identity":
        document = yaml.safe_load(cli.site_file.read_text())
        document["spec"]["cpu"]["eksArn"] += "-another"
        cli.site_file.write_text(yaml.safe_dump(document))
    elif changed == "predecessor":
        document = json.loads(cli.predecessor.read_text())
        document["executed_at"] = "2026-09-16T00:00:00+00:00"
        cli.predecessor.write_text(json.dumps(document))
    else:
        path = cli.kubeconfig if changed == "kubeconfig" else cli.site_file
        path.write_text(path.read_text() + "# changed at the same path\n")
    cli.approve()
    with pytest.raises(RuntimeError, match="drifted at connections|drifted at details"):
        runner.main()
    assert cli.events == [], "input drift must fail before harness construction or run"


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("release_id", ["", "   ", None, True, 7, []])
def test_capacity_missing_or_invalid_release_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execute: bool, release_id: Any
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path)
    if execute:
        assert runner.main() == 0, "the approved plan starts with a complete identity"
        cli.approve()
    cli.state["release_id"] = release_id
    with pytest.raises(CapError, match="deployed release identity is missing"):
        runner.main()
    assert cli.events == [], "unknown release identity cannot reach the harness"
    if not execute:
        assert not cli.plan_path.exists(), "unknown identity cannot produce a plan"


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize(
    "missing", ["site", "cpu_kubeconfig", "cpu_identity", "release"]
)
def test_capacity_missing_identity_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execute: bool, missing: str
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path)
    if execute:
        assert runner.main() == 0, "the approved plan starts with complete inputs"
        cli.approve()
    if missing == "site":
        cli.site_file.unlink()
    elif missing == "cpu_kubeconfig":
        cli.kubeconfig.unlink()
    elif missing == "release":
        cli.state.pop("release_id")
    else:
        document = yaml.safe_load(cli.site_file.read_text())
        document["spec"]["cpu"].pop("eksArn")
        cli.site_file.write_text(yaml.safe_dump(document))
    with pytest.raises((CapError, SiteConfigError)):
        runner.main()
    assert cli.events == [], "missing identity inputs cannot reach the harness"
    if not execute:
        assert not cli.plan_path.exists(), "missing identity cannot produce a plan"


@pytest.mark.parametrize(
    ("case_id", "field"),
    [
        ("GF-REGIONAL-CAP-001", "release_id"),
        ("GF-REGIONAL-CAP-002", "release_id"),
        ("GF-REGIONAL-CAP-002", "cluster_id"),
    ],
)
def test_capacity_predecessor_must_match_the_release_and_cpu_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str, field: str
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path, case_id)
    document = json.loads(cli.predecessor.read_text())
    document[field] = "another-identity"
    cli.predecessor.write_text(json.dumps(document))
    assert runner.main() == 1, "foreign CAP evidence cannot pass preflight"
    plan = json.loads(cli.plan_path.read_text())
    assert plan["preflight_passed"] is False, (
        "a failed predecessor must remain blocking"
    )
    cli.approve()
    with pytest.raises(RuntimeError, match="did not pass its preflight"):
        runner.main()
    assert cli.events == [], "a foreign predecessor cannot authorize mutations"


@pytest.mark.parametrize("execute", [False, True])
def test_capacity_release_read_failure_does_not_authorize_or_serialize_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execute: bool
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path)
    if execute:
        assert runner.main() == 0, (
            "the approved plan starts with a successful release read"
        )
        cli.approve()
    monkeypatch.setattr(
        release_state,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 1, "unit-only-private-output", "unit-only-private-error"
        ),
    )
    with pytest.raises(SiteConfigError, match="cannot read live ConfigMap") as error:
        runner.main()
    assert "unit-only-private" not in str(error.value), (
        "read failures must not expose captured command output"
    )
    assert cli.events == [], "a failed release read cannot create a harness"
    if not execute:
        assert not cli.plan_path.exists(), "a failed release read cannot create a plan"


def test_capacity_harness_consumes_the_authorized_site_without_reloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = capacity_cli(monkeypatch, tmp_path)
    assert runner.main() == 0, "the original CPU site must pass preflight"
    cli.approve()
    monkeypatch.setattr(runner, "CapHarness", CapCoreHarness)

    def kubectl_json(_self: CapCoreHarness, *arguments: str) -> dict[str, Any]:
        if arguments == (
            "get",
            "configmap",
            "gpu-fault-control-worker-config-processor",
            "-o",
            "json",
        ):
            return {"data": {"GPU_FAULT_PROCESSOR_MODE": "direct"}}
        assert arguments == (
            "get",
            "deployment",
            "gpu-fault-control-worker",
            "-o",
            "json",
        ), "the constructor may only read the processor mode and worker Deployment"
        return {
            "spec": {"template": {"spec": {"containers": [{"image": "unit-image"}]}}}
        }

    def run(harness: CapCoreHarness) -> int:
        assert harness.site is cli.loads[-1], (
            "the authorized site must reach the harness"
        )
        assert harness.evidence_identity == {
            "release_id": "unit-release",
            "cluster_id": harness.config["cpu_eks_arn"],
        }, "the harness must retain its CPU evidence binding"
        cli.events.append("run")
        return 0

    monkeypatch.setattr(CapCoreHarness, "kubectl_json", kubectl_json)
    monkeypatch.setattr(CapCoreHarness, "run", run, raising=False)
    monkeypatch.setattr(
        "scripts.e2e.regional.capacity_acceptance_base.load_site",
        lambda _path: pytest.fail(
            "the authorized site must not be replaced by a reload"
        ),
    )
    assert runner.main() == 0, (
        "the real constructor must accept the authorized snapshot"
    )
    assert cli.events == ["run"], "the authorized handoff must reach the harness run"
