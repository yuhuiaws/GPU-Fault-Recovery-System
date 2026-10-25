"""Construct the managed-job wrappers and drive run_general_live.main()."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_utilization as util
from tests.regional._general_live_fakes import (
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


class _FakeWorkloadFixture:
    def __init__(self, regional, settings, *, state_path=None) -> None:
        self.regional = regional
        self.settings = settings


class _FakePrewarm:
    def __init__(self, regional, *, case_id, run_id, state_path=None) -> None:
        self.regional = regional


def _patch_workload(monkeypatch) -> None:
    import scripts.e2e.regional.managed_workload_fixture as mwf

    monkeypatch.setattr(mwf, "ImagePrewarmFixture", _FakePrewarm)
    monkeypatch.setattr(mwf, "ManagedWorkloadFixture", _FakeWorkloadFixture)


def _ctx(spec, tmp_path: Path):
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(
        **{**settings.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    (tmp_path / "site.yaml").write_text("site: test\n")
    return make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )


def test_managed_idle_job_renders_idle_and_requested_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_workload(monkeypatch)
    spec = next(s for s in util.VARIANTS if s.variant == "ambiguous_ownership")
    ctx = _ctx(spec, tmp_path)
    job = util.ManagedIdleJob(ctx, name="gl3-amb", gpu_count=4)
    document = yaml.safe_load((ctx.case_dir / "gl3-amb.source.yaml").read_text())
    containers = [
        container
        for replica in document["spec"]["pytorchReplicaSpecs"].values()
        for container in replica["template"]["spec"]["containers"]
    ]
    assert any(
        env.get("name") == "GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT"
        and env.get("value") == "0"
        for container in containers
        for env in container.get("env", [])
    ), "any( env.get('name') == 'GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT' and en"
    assert all(
        container["resources"]["requests"]["nvidia.com/gpu"] == "4"
        for container in containers
        if "nvidia.com/gpu" in container["resources"]["requests"]
    ), "all( container['resources']['requests']['nvidia.com/gpu'] == '4' for c"
    # The trainer and the EFA request follow the GPU count: eight torchrun
    # workers on four GPUs failed the Pod and the product stopped the attempt
    # (live 2026-10-24/25), and two 16-EFA jobs never fit one node.
    assert [
        (
            container["command"][-1].count("--nproc-per-node=4"),
            container["resources"]["limits"]["vpc.amazonaws.com/efa"],
            container["resources"]["requests"]["vpc.amazonaws.com/efa"],
        )
        for container in containers
    ] == [(1, "8", "8")], containers
    assert job.name == "gl3-amb"


def test_managed_idle_job_requires_site_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_workload(monkeypatch)
    spec = next(s for s in util.VARIANTS if s.variant == "gpu_idle")
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    with pytest.raises(util.VariantNotRun, match="site-file"):
        util.ManagedIdleJob(ctx, name="x", gpu_count=8)


@pytest.mark.parametrize(
    "mode,needle",
    [
        ("exit:7", "exit ${CODE:-1}"),
        ("stop", "exit 143"),
        ("hang", "trap '' TERM"),
        ("exit_once:3", "*-r-*)"),
    ],
)
def test_managed_fail_job_applies_mode(
    tmp_path: Path, mode: str, needle: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_workload(monkeypatch)
    spec = next(s for s in att.VARIANTS if s.variant == "nonzero_exit")
    ctx = _ctx(spec, tmp_path)
    job = att.ManagedFailJob(ctx, name="gl28x", mode=mode, restart_budget=1)
    document = yaml.safe_load((ctx.case_dir / "gl28x.source.yaml").read_text())
    containers = [
        container
        for replica in document["spec"]["pytorchReplicaSpecs"].values()
        for container in replica["template"]["spec"]["containers"]
    ]
    commands = [container["command"][-1] for container in containers]
    assert any(needle in command for command in commands), (
        "any(needle in command for command in commands)"
    )
    assert job.attempt_id == "gl28x-a001"
    env = {item["name"]: item for item in containers[0]["env"]}
    assert env["CODE"]["value"] == (mode.split(":")[1] if ":" in mode else "1")
    if mode.startswith("exit_once"):
        # The Pod reads its own attempt id: the product's restart copy carries
        # the -r- id on its template label, so only the first attempt exits.
        assert env[att.ATTEMPT_ID_ENV]["valueFrom"] == {
            "fieldRef": {"fieldPath": f"metadata.labels['{att.ATTEMPT_ID_LABEL}']"}
        }
        assert "sleep 30; exit ${CODE:-1}" in commands[0], commands[0]
    else:
        assert att.ATTEMPT_ID_ENV not in env


# --------------------------------------------------------------------------- #
# run_general_live.main()
# --------------------------------------------------------------------------- #
def _argv(tmp_path: Path, variant: str, *extra: str) -> list[str]:
    kc = tmp_path / "kc"
    kc.write_text("apiVersion: v1\n")
    return [
        "run_general_live.py",
        "--run-dir",
        str(tmp_path / "run"),
        "--case",
        "GF-GENERAL-LIVE-001",
        "--variant",
        variant,
        "--cpu-kubeconfig",
        str(kc),
        "--gpu-kubeconfig",
        str(kc),
        "--gpu-context",
        "ctx",
        "--cluster-id",
        "gpu-eks-cluster",
        "--region",
        "us-west-2",
        "--fault-node",
        NODE,
        "--host-probe-image",
        "public.ecr.aws/x@sha256:" + "0" * 64,
        *extra,
    ]


def _patch_main(
    monkeypatch: pytest.MonkeyPatch, *, preflight_errors=None, verdict="PASS"
):
    from scripts.e2e.regional import run_general_live as runner

    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: FakeRegional())
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda s, spec, r: {
            "errors": preflight_errors or [],
            "planned_verdict": None,
            "release_id": "r",
            "cluster_id": "c",
        },
    )
    monkeypatch.setattr(
        runner,
        "build_plan",
        lambda **kw: {
            "case_id": kw["case_id"],
            "preflight_passed": kw["preflight_passed"],
        },
    )
    monkeypatch.setattr(
        runner,
        "authorize_execution",
        lambda *a, **k: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        + __import__("datetime").timedelta(hours=1),
    )
    monkeypatch.setattr(runner, "execute_variant", lambda *a, **k: {"verdict": verdict})
    return runner


def test_main_plan_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _patch_main(monkeypatch)
    monkeypatch.setattr("sys.argv", _argv(tmp_path, "cpu_busy", "--plan"))
    assert runner.main() == 0


def test_main_plan_fails_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _patch_main(monkeypatch, preflight_errors=["x"])
    monkeypatch.setattr("sys.argv", _argv(tmp_path, "cpu_busy", "--plan"))
    assert runner.main() == 1


def test_main_execute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _patch_main(monkeypatch, verdict="PASS")
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            tmp_path,
            "cpu_busy",
            "--execute",
            "--confirm",
            "GENERALLIVE001_EXECUTE",
            "--maintenance-window-end",
            "2026-11-07T00:00:00Z",
        ),
    )
    assert runner.main() == 0


def test_main_execute_wrong_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _patch_main(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            tmp_path,
            "cpu_busy",
            "--execute",
            "--confirm",
            "WRONG",
            "--maintenance-window-end",
            "2026-11-07T00:00:00Z",
        ),
    )
    with pytest.raises(Exception, match="confirmation must be exactly"):
        runner.main()


def test_main_deferred_execute_records_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _patch_main(monkeypatch)
    # invalid_delta and 004 probe_error are implemented now; 008 bond_vf stays
    # CAPABILITY_ABSENT on this site, so --execute records NOT_RUN without acting.
    kc = tmp_path / "kc"
    kc.write_text("apiVersion: v1\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_general_live.py",
            "--run-dir",
            str(tmp_path / "run"),
            "--case",
            "GF-GENERAL-LIVE-008",
            "--variant",
            "bond_vf",
            "--cpu-kubeconfig",
            str(kc),
            "--gpu-kubeconfig",
            str(kc),
            "--gpu-context",
            "ctx",
            "--cluster-id",
            "gpu-eks-cluster",
            "--region",
            "us-west-2",
            "--fault-node",
            NODE,
            "--host-probe-image",
            "a@sha256:" + "0" * 64,
            "--execute",
            "--confirm",
            "GENERALLIVE008_EXECUTE",
            "--maintenance-window-end",
            "2026-11-07T00:00:00Z",
        ],
    )
    assert runner.main() == 0
    evidence = (
        tmp_path
        / "run"
        / "cases"
        / "GF-GENERAL-LIVE-008.bond_vf"
        / "GF-GENERAL-LIVE-008.bond_vf.json"
    )
    assert evidence.is_file(), "evidence.is_file()"
    import json

    assert json.loads(evidence.read_text())["verdict"] == "NOT_RUN"


def test_managed_fail_job_renders_two_ranks_with_their_own_modes_on_one_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """028.cleanup_timeout: a Master that exits non-zero and a Worker that
    ignores the stop, both pinned to the node, four GPUs and eight EFA devices
    each (two such Pods fill one p5en node), expected as two Pods."""

    _patch_workload(monkeypatch)
    spec = next(s for s in att.VARIANTS if s.variant == "cleanup_timeout")
    ctx = _ctx(spec, tmp_path)
    job = att.ManagedFailJob(
        ctx,
        name="gl28ct",
        mode={"Master": "exit:1", "Worker": "hang"},
        restart_budget=0,
        gpu_count=4,
        ranks=2,
    )
    document = yaml.safe_load((ctx.case_dir / "gl28ct.source.yaml").read_text())
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert list(replicas) == ["Master", "Worker"]
    rendered = {}
    for role, replica in replicas.items():
        (container,) = replica["template"]["spec"]["containers"]
        rendered[role] = (
            replica["replicas"],
            replica["template"]["spec"]["nodeName"],
            container["command"][-1],
            container["resources"]["limits"]["nvidia.com/gpu"],
            container["resources"]["requests"]["vpc.amazonaws.com/efa"],
        )
    assert rendered == {
        "Master": (1, NODE, "sleep 30; exit ${CODE:-1}", "4", "8"),
        "Worker": (1, NODE, "trap '' TERM; sleep 3600", "4", "8"),
    }, rendered
    assert job.fixture.settings.expected_pods == 2
    assert job.fixture.settings.expected_gpu_count == 4
    assert job.fixture.settings.restart_budget == 0


def test_managed_fail_job_refuses_a_mode_for_a_replica_it_does_not_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_workload(monkeypatch)
    spec = next(s for s in att.VARIANTS if s.variant == "cleanup_timeout")
    ctx = _ctx(spec, tmp_path)
    with pytest.raises(ValueError, match="Worker"):
        att.ManagedFailJob(
            ctx, name="gl28x", mode={"Master": "exit:1", "Worker": "hang"}
        )
