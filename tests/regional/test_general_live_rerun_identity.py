"""GF-GENERAL-LIVE managed jobs: a rerun by the same lane names a new job.

Live 2026-10-26 (HyperPod lane hpb): 017.old_pod 21:52:54Z, 017.cri_partial_rotation
21:54:07Z and 017.live_pid 21:55:12Z ended INCONCLUSIVE "harness fixture failed:
RegionalFixtureError: an existing fixture is cleanup-only". Each was a rerun,
released by a landing, of a variant the same lane had run that morning: the
managed-job identity (``gl17-op-a6-227ee78d``: case number, job suffix, attempt
and a digest of the case directory) is stable per lane and attempt, the case
directory still held that job's completed ``.ownership.json`` and
``.prewarm.json`` from the morning, and a journaled fixture is create-once --
right for a crashed run, wrong for a deliberate rerun after a clean cleanup.
Reruns the other lane picked up had another digest and passed, which made the
failure look sporadic.

The rule under test is ``fixture_ownership.rerun_identity``: the base identity,
or ``<base>-r<n>`` once the case directory holds closed custody of earlier runs
under it; an open (begun, not completed) journal keeps its identity so the
fixture still refuses it as cleanup-only. ``VariantContext.identity`` memoises
the rule per run, so one run names a job once. Below the fake kubectl
(``test_general_live_managed_job_cleanup.Cluster``) everything is the runner's
own: the variant, the wrappers, the prewarm, the workload fixture, its journals.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import env_live_scope as scope
from scripts.e2e.regional import fixture_ownership
from scripts.e2e.regional import general_live_attempts as attempts
from scripts.e2e.regional import general_live_nccl_roles as nccl_roles
from scripts.e2e.regional import host_probe_fixture as probe_module
from scripts.e2e.regional import run_general_live as runner
from scripts.e2e.regional.fixture_ownership import (
    FixtureOwnership,
    custody_closed,
    custody_journals,
    rerun_identity,
)
from scripts.e2e.regional.general_live_fixture import (
    HOST_PROBE_STATE,
    RUN_ID,
    Evidence,
    GeneralLiveSettings,
    VariantContext,
    host_probe_records,
    rerun_run_id,
    run_identity,
)
from scripts.e2e.regional.managed_workload_fixture import (
    NOTHING_CREATED,
    managed_workload_name,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from tests.admin.test_admin_site import site_file as write_site_file
from tests.regional._host_probe_support import ProbeApi, host_probe
from tests.regional.test_general_live_managed_job_cleanup import (
    CASE_VARIANTS,
    RESTART_OPERATION,
    SINGLE_NODE,
    WRAPPERS,
    Cluster,
    execute,
    install,
    journal,
    variant_context,
)
from tests.regional.test_general_live_managed_job_start import (
    NAMESPACE,
    NODE,
    REGISTRY,
    configured,
)

NONZERO_EXIT = "GF-GENERAL-LIVE-028.nonzero_exit"
CLEANUP_ONLY = "an existing fixture is cleanup-only"
ENV_CASE = "GF-ENV-LIVE-010"
ENV_RUN_ID = "gf-env-live-010-a7-1793022554"
#: The live 022 rerun: its base run id, and the refusal its record drew.
RECOVERY_EVIDENCE = "GF-GENERAL-LIVE-022.recovery_evidence"
RECOVERY_RUN_ID = "gl022-recovery-evidence-a6-43c8c8f3"
RECORD_MISMATCH = "host probe ownership record does not match this run"


@pytest.fixture(autouse=True)
def private_umask() -> Iterator[None]:
    """``run_general_live.main`` runs under 0o077; the journals require it."""

    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


# --------------------------------------------------------------------------- #
# journals as the rule reads them
# --------------------------------------------------------------------------- #
def journal_at(path: Path, *, begun: bool, completed: bool) -> Path:
    """A custody journal in one of the three states a run can leave."""

    ownership = FixtureOwnership(path, {"purpose": "test", "inputs": {}})
    if begun:
        ownership.begin()
    if completed:
        ownership.complete()
    return path


def closed_journals(directory: Path, identity: str) -> list[Path]:
    return [
        journal_at(directory / f"{identity}.{kind}.json", begun=True, completed=True)
        for kind in fixture_ownership.JOURNAL_KINDS
    ]


def later_run(key: str, settings: GeneralLiveSettings, run_dir: Path) -> VariantContext:
    """The context a later run of ``key`` builds over the same case directory
    (``execute_variant``'s, as ``variant_context`` builds the first run's)."""

    spec = REGISTRY[key]
    case_dir = run_dir / "cases" / spec.key
    return VariantContext(
        spec=spec,
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        run_id=run_identity(spec.case_id, spec.variant, 1, run_dir),
        attempt=1,
        deadline=datetime.now(timezone.utc) + timedelta(hours=3),
        evidence=Evidence(case_dir),
    )


def test_custody_closed_reads_the_journal_state(tmp_path: Path) -> None:
    never_begun = journal_at(
        tmp_path / "a.ownership.json", begun=False, completed=False
    )
    begun = journal_at(tmp_path / "b.ownership.json", begun=True, completed=False)
    completed = journal_at(tmp_path / "c.prewarm.json", begun=True, completed=True)
    assert custody_closed(completed) is True, "a completed cleanup closed the custody"
    assert custody_closed(never_begun) is True, (
        "begin precedes every intent and every intent its create: a journal "
        "never begun journaled nothing and holds nothing"
    )
    assert custody_closed(begun) is False, "begun and not completed is a crashed run"

    garbage = tmp_path / "d.ownership.json"
    garbage.write_text("{", encoding="utf-8")
    garbage.chmod(0o600)
    not_a_record = tmp_path / "e.ownership.json"
    not_a_record.write_text('{"owner": "x"}', encoding="utf-8")
    not_a_record.chmod(0o600)
    completed.chmod(0o644)  # no longer private: not read as a journal
    for path in (garbage, not_a_record, completed, tmp_path / "missing.ownership.json"):
        assert custody_closed(path) is False, f"{path.name} must stay cleanup-only"


def test_rerun_identity_is_the_base_until_closed_custody_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = "gl17-op-a6-227ee78d"
    assert rerun_identity(tmp_path / "absent", base) == base
    assert rerun_identity(tmp_path, base) == base, "no journal: the base identity"

    closed_journals(tmp_path, base)
    assert rerun_identity(tmp_path, base) == f"{base}-r2", (
        "closed custody of the base: the next free ordinal"
    )
    closed_journals(tmp_path, f"{base}-r2")
    assert rerun_identity(tmp_path, base) == f"{base}-r3"

    # One journal of the pair suffices, and a never-begun pair is closed too.
    journal_at(tmp_path / "gl3-idle-1.ownership.json", begun=True, completed=True)
    assert rerun_identity(tmp_path, "gl3-idle-1") == "gl3-idle-1-r2"
    for kind in fixture_ownership.JOURNAL_KINDS:
        journal_at(tmp_path / f"gl3-attA-1.{kind}.json", begun=False, completed=False)
    assert rerun_identity(tmp_path, "gl3-attA-1") == "gl3-attA-1-r2", (
        "a failed preparation leaves journals that prove nothing was created"
    )

    # Open custody keeps its identity: the fixture refuses it as cleanup-only.
    journal_at(
        tmp_path / "gl28nz-nz-a1-c5ab1930.prewarm.json", begun=True, completed=False
    )
    assert rerun_identity(tmp_path, "gl28nz-nz-a1-c5ab1930") == "gl28nz-nz-a1-c5ab1930"
    closed_journals(tmp_path, "gl11-fr-a2-00000000")
    journal_at(
        tmp_path / "gl11-fr-a2-00000000-r2.ownership.json", begun=True, completed=False
    )
    assert rerun_identity(tmp_path, "gl11-fr-a2-00000000") == "gl11-fr-a2-00000000-r2"
    # 011's second attempt journals under ``<identity>-a002``: open there, the
    # identity is still that run's and stays cleanup-only.
    closed_journals(tmp_path, "gl11-ac-a3-00000000")
    journal_at(
        tmp_path / "gl11-ac-a3-00000000-a002.ownership.json",
        begun=True,
        completed=False,
    )
    assert rerun_identity(tmp_path, "gl11-ac-a3-00000000") == "gl11-ac-a3-00000000"
    assert [
        path.name for path in custody_journals(tmp_path, "gl11-ac-a3-00000000")
    ] == [
        "gl11-ac-a3-00000000-a002.ownership.json",
        "gl11-ac-a3-00000000.ownership.json",
        "gl11-ac-a3-00000000.prewarm.json",
    ]
    assert custody_journals(tmp_path, "gl11-ac-a3-0000000") == [], "whole stems only"

    for bad in ("x" * 59, "-leading", "trailing-", "", "a b"):
        with pytest.raises(
            RegionalFixtureError, match="cannot derive a rerun identity"
        ):
            rerun_identity(tmp_path, bad)
    monkeypatch.setattr(fixture_ownership, "RERUN_LIMIT", 2)
    closed_journals(tmp_path, "gl27-nan-a1-abcdef01")
    closed_journals(tmp_path, "gl27-nan-a1-abcdef01-r2")
    with pytest.raises(RegionalFixtureError, match="has used every rerun identity"):
        rerun_identity(tmp_path, "gl27-nan-a1-abcdef01")


def test_plan_and_execute_compute_the_same_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan and execute are two processes minutes apart over one case directory;
    the derivation reads the journals and writes nothing, so both agree -- and
    within one run the context memoises it, so a job is named once."""

    cluster = Cluster()
    ctx = variant_context(NONZERO_EXIT, tmp_path, monkeypatch, cluster)
    base = run_identity("GF-GENERAL-LIVE-028", "nz", 1, ctx.case_dir).replace(
        "gl028", "gl28nz"
    )
    planned = rerun_identity(ctx.case_dir, base)
    assert planned == base == rerun_identity(ctx.case_dir, base) == ctx.identity(base)

    closed_journals(ctx.case_dir, base)  # this run's first attempt came and went
    assert ctx.identity(base) == base, "memoised: the same job for a second attempt"
    assert ctx.identities == {base: base}
    assert rerun_identity(ctx.case_dir, base) == f"{base}-r2", (
        "the next run -- planned and executed -- gets the next ordinal"
    )
    later = later_run(NONZERO_EXIT, ctx.settings, ctx.case_dir.parent.parent)
    assert later.identity(base) == f"{base}-r2" == later.identity(base)
    assert later.identity(f"{base}-r2") == f"{base}-r2", (
        "an identity this run derived is returned as it is (024 hands its "
        "ledger's job id on to start_job), never derived a second time"
    )
    assert later.identities == {base: f"{base}-r2"}
    assert later.case_dir == ctx.case_dir, "the same lane, the same case directory"


# --------------------------------------------------------------------------- #
# the live sequence through execute_variant: run, rerun, rerun
# --------------------------------------------------------------------------- #
def budgeted_restart(
    regional: Any, cluster_id: str, job_id: str, attempt_ids: list[str]
) -> dict[str, Any]:
    """The product's records of a failed attempt it stopped and restarted."""

    (attempt,) = attempt_ids
    entry = {
        "decision": {"status": "PLAN_CREATED"},
        "event": {"terminal_status": "FAILED"},
        "containment_incident": {"incident_id": f"incident-{job_id}"},
        "containment_workflows": [
            {"official_steps": [{"operation": "STOP_WORKLOADS"}]}
        ],
        "recovery": [
            {"workflow": {"official_steps": [{"operation": "RESTART_WORKLOAD"}]}}
        ],
    }
    budget = {"restart_count": 1, "reservation_ids": [RESTART_OPERATION]}
    return {"attempts": {attempt: entry}, "restart_budget": budget}


def test_a_rerun_of_a_managed_job_variant_runs_under_the_next_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """028.nonzero_exit three times in one case directory, as one lane reruns
    a variant within a campaign attempt: ``<base>``, ``<base>-r2``,
    ``<base>-r3``, each a PASS under its own job, the earlier journals kept."""

    cluster = Cluster()
    settings, run_dir = configured(tmp_path, monkeypatch, REGISTRY[NONZERO_EXIT])
    install(monkeypatch, cluster)
    monkeypatch.setattr(attempts, "attempt_records", budgeted_restart)
    case_dir = run_dir / "cases" / NONZERO_EXIT
    base = run_identity("GF-GENERAL-LIVE-028", "nz", 1, case_dir).replace(
        "gl028", "gl28nz"
    )
    first_journal: str | None = None

    for suffix in ("", "-r2", "-r3"):
        identity = f"{base}{suffix}"
        result = execute(NONZERO_EXIT, settings, run_dir)

        assert (result["verdict"], result.get("errors")) == ("PASS", []), result
        assert result["attempt"] == f"{identity}-a001", result
        assert result["identities"] == {base: identity}, result["identities"]
        for kind in fixture_ownership.JOURNAL_KINDS:
            assert journal(case_dir / f"{identity}.{kind}.json")["completed"] is True, (
                f"{identity}.{kind}.json closes with the run"
            )
        (cleanup,) = result["cleanup"]
        name = managed_workload_name(SINGLE_NODE, identity)
        assert f"pytorchjob/{name}" in cleanup["result"]["workload"]["deleted"], cleanup
        assert cluster.kinds() == {"deployment"}, cluster.objects
        text = (case_dir / f"{base}.ownership.json").read_text(encoding="utf-8")
        first_journal = first_journal or text
        assert text == first_journal, "the first run's journal is never reopened"

    created = sorted(
        name for kind, name in cluster.deleted_objects if kind == "pytorchjob"
    )
    assert created == sorted(
        managed_workload_name(SINGLE_NODE, f"{base}{suffix}")
        for suffix in ("", "-r2", "-r3")
    ), "three runs, three jobs, three workload names"
    assert rerun_identity(case_dir, base) == f"{base}-r4"


def test_a_crashed_runs_custody_still_refuses_as_cleanup_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that started its job and died: the rerun finds the open journals,
    keeps the identity and the fixture refuses -- the crashed custody is never
    skipped; its cleanup-only resumption removes what the crashed run left,
    and only the run after that starts fresh under ``-r2``."""

    cluster = Cluster()
    settings, run_dir = configured(tmp_path, monkeypatch, REGISTRY[NONZERO_EXIT])
    install(monkeypatch, cluster)
    monkeypatch.setattr(attempts, "attempt_records", budgeted_restart)
    case_dir = run_dir / "cases" / NONZERO_EXIT
    base = run_identity("GF-GENERAL-LIVE-028", "nz", 1, case_dir).replace(
        "gl028", "gl28nz"
    )
    crashed = later_run(NONZERO_EXIT, settings, run_dir)
    job = attempts.ManagedFailJob(crashed, name=crashed.identity(base), mode="exit:1")
    job.start()
    assert job.name == base and ("pytorchjob", job.fixture.name) in cluster.objects
    assert journal(case_dir / f"{base}.ownership.json")["started"] is True
    # ... and the process dies here: no cleanup, both journals open.

    refused = execute(NONZERO_EXIT, settings, run_dir)

    assert refused["verdict"] == "INCONCLUSIVE", refused
    assert refused["detail"] == (
        f"harness fixture failed: RegionalFixtureError: {CLEANUP_ONLY}"
    ), refused["detail"]
    assert refused["identities"] == {base: base}, "the open custody keeps its identity"
    (cleanup,) = refused["cleanup"]
    assert cleanup["ok"] is True, cleanup
    assert (
        f"pytorchjob/{job.fixture.name}" in cleanup["result"]["workload"]["deleted"]
    ), "cleanup-only: the reopened journal removes the crashed run's workload"
    assert cluster.kinds() == {"deployment"}, cluster.objects
    assert journal(case_dir / f"{base}.ownership.json")["completed"] is True

    fresh = execute(NONZERO_EXIT, settings, run_dir)

    assert (fresh["verdict"], fresh["attempt"]) == ("PASS", f"{base}-r2-a001"), fresh
    assert fresh["identities"] == {base: f"{base}-r2"}


# --------------------------------------------------------------------------- #
# every journaled fixture follows: the wrappers, the prewarm, the control task
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(WRAPPERS))
def test_every_wrapper_journals_under_its_identity(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule reads ``<identity>.ownership.json`` and ``<identity>.prewarm.json``
    (and ``<identity>-a<nnn>.*`` of a later attempt): that is where every
    wrapper keeps its two journals."""

    ctx = variant_context(CASE_VARIANTS[case], tmp_path, monkeypatch, Cluster())
    job = WRAPPERS[case](ctx, "gl-job-a")
    assert job.fixture.ownership is not None and job.prewarm.ownership is not None
    paths = {job.fixture.ownership.path, job.prewarm.ownership.path}
    assert paths == {
        ctx.case_dir / "gl-job-a.ownership.json",
        ctx.case_dir / "gl-job-a.prewarm.json",
    }, paths
    assert set(custody_journals(ctx.case_dir, "gl-job-a")) == paths
    if case == "GF-GENERAL-LIVE-011":
        second = nccl_roles.SecondAttemptNcclJob(
            ctx, name="gl-job-a", stall_mode="quiet"
        )
        assert second.fixture.ownership is not None
        assert second.prewarm.ownership is not None
        assert second.fixture.ownership.path == (
            ctx.case_dir / "gl-job-a-a002.ownership.json"
        )
        assert set(custody_journals(ctx.case_dir, "gl-job-a")) == paths | {
            second.fixture.ownership.path,
            second.prewarm.ownership.path,
        }
    # Constructed and never started: the journals prove nothing was created,
    # so a later run is free to move on.
    assert rerun_identity(ctx.case_dir, "gl-job-a") == "gl-job-a-r2"


def test_the_control_task_follows_the_rerun_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GF-ENV-LIVE-010's control task is a managed job under the same
    fixtures: a second task under one run id in one case directory, after the
    first was removed, is ``envctl-<10 hex>-r2`` -- its prewarm journal, its
    job id and its attempt id with it."""

    cluster = Cluster()
    install(monkeypatch, cluster)
    kube = tmp_path / "kube"
    kube.mkdir()
    for plane in ("cpu", "gpu"):
        (kube / plane).write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=kube / "cpu",
            gpu_kubeconfig=kube / "gpu",
            gpu_context="gpu-a",
            namespace=NAMESPACE,
            cluster_id="gpu-a",
            region="us-west-2",
        )
    )
    site_root = tmp_path / "site"
    site_root.mkdir()
    site = write_site_file(site_root)
    case_dir = tmp_path / "cases" / ENV_CASE
    base = scope.control_task_name(ENV_CASE, ENV_RUN_ID)

    def control_task() -> scope.ControlTask:
        return scope.ControlTask(
            regional,
            site_file=site,
            case_dir=case_dir,
            case_id=ENV_CASE,
            run_id=ENV_RUN_ID,
            node=NODE,
        )

    first = control_task()
    assert first.name == base, "the first run keeps the control task's name"
    first.start()
    assert first.fixture.workload()["metadata"]["labels"]["gpu-fault.io/job-id"] == base
    record = first.cleanup()
    assert record["prewarm"] == NOTHING_CREATED and cluster.kinds() == {"deployment"}

    second = control_task()

    assert (second.name, second.attempt_id) == (f"{base}-r2", f"{base}-r2-a001")
    assert second.fixture.settings.job_id == f"{base}-r2"
    assert second.prewarm.ownership is not None
    assert second.prewarm.ownership.path == case_dir / f"{base}-r2.prewarm.json"
    assert second.fixture.ownership is not None
    assert second.fixture.ownership.path == case_dir / f"{base}-r2.ownership.json"
    assert second.fixture.name != first.fixture.name, "its own workload name"
    assert second.fixture.settings.manifest == case_dir / f"{base}-r2.source.yaml"
    second.start()
    assert second.snapshot()["job_id"] == f"{base}-r2"
    second.cleanup()
    assert cluster.kinds() == {"deployment"}, cluster.objects
    assert rerun_identity(case_dir, base) == f"{base}-r3"


# --------------------------------------------------------------------------- #
# the run id: host-probe records name the run
# --------------------------------------------------------------------------- #
def probe_record(
    case_dir: Path,
    run_id: str,
    *,
    closed: bool,
    probe: str = "general_live_node_probe.py",
    directory: str = HOST_PROBE_STATE,
) -> Path:
    """A host-probe ownership record as ``host_probe_fixture`` leaves one: named
    after a digest of the run id and the probe, its scope naming the run."""

    state = case_dir / directory
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{run_id}\0{probe}".encode()).hexdigest()[:10]
    path = state / f"gpu-fault-host-probe-{digest}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": {"run_id": run_id, "node": NODE, "case_id": RECOVERY_EVIDENCE},
                "owner": "0" * 32,
                "node_uid": "node-uid",
                "resources": {},
                "script_may_exist": False,
                "closed": closed,
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(".lock").write_text("", encoding="utf-8")
    return path


def test_rerun_run_id_follows_the_host_probe_records(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / RECOVERY_EVIDENCE
    base = RECOVERY_RUN_ID
    assert rerun_run_id(case_dir, base) == base, "no case directory yet: the base"
    assert host_probe_records(case_dir) == []

    probe_record(case_dir, base, closed=True)
    assert host_probe_records(case_dir) == [(base, True)]
    assert rerun_run_id(case_dir, base) == f"{base}-r2", (
        "the earlier run's record is closed: a new run id, so new record names"
    )
    probe_record(case_dir, f"{base}-r2", closed=True)
    probe_record(
        case_dir, f"{base}-r2", closed=True, probe="general_live_dcgm_probe.py"
    )
    assert rerun_run_id(case_dir, base) == f"{base}-r3"
    # One record of a run still open keeps that run id: the fixture meets it.
    probe_record(case_dir, f"{base}-r3", closed=True)
    probe_record(
        case_dir, f"{base}-r3", closed=False, probe="general_live_dcgm_probe.py"
    )
    assert rerun_run_id(case_dir, base) == f"{base}-r3"
    # 030's spare-service records are run records too; other files are not.
    probe_record(case_dir, f"{base}-r4", closed=True, directory="spare-service")
    (case_dir / "probe-error.json").write_text('{"error": "x"}', encoding="utf-8")
    (case_dir / HOST_PROBE_STATE / "garbage.json").write_text("{", encoding="utf-8")
    (case_dir / HOST_PROBE_STATE / "list.json").write_text("[1]", encoding="utf-8")
    assert sorted(host_probe_records(case_dir)) == [
        (base, True),
        (f"{base}-r2", True),
        (f"{base}-r2", True),
        (f"{base}-r3", False),
        (f"{base}-r3", True),
        (f"{base}-r4", True),
    ]
    assert rerun_run_id(case_dir, base) == f"{base}-r3"

    assert RUN_ID.fullmatch(f"{base}-r99") is not None, "a probe's SAFE_RUN_ID"
    with pytest.raises(RegionalFixtureError, match="cannot derive a rerun identity"):
        rerun_run_id(case_dir, "GL022-Upper")
    assert rerun_run_id(case_dir, base) == rerun_run_id(case_dir, base), (
        "plan and execute read the same records and agree"
    )


def test_a_rerun_names_fresh_host_probe_records_and_an_open_one_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live 022 sequence with the real fixture over the in-memory API: a run
    whose record closed is followed by a rerun under a changed probe script --
    new run id, new record, no refusal; a run whose record stayed open keeps
    its run id, and the changed script meets the open record and is refused
    as before, until that record is closed by its own cleanup-only resumption."""

    api = ProbeApi()
    monkeypatch.setattr(probe_module, "run_fixture_command", api.run)
    case_dir = tmp_path / "cases" / RECOVERY_EVIDENCE
    first = host_probe(
        tmp_path,
        run_id=RECOVERY_RUN_ID,
        case_id=RECOVERY_EVIDENCE,
        node=NODE,
        state_directory=case_dir / HOST_PROBE_STATE,
    )
    first.create()
    assert first.execute("survey") == {"ok": True}
    assert not any(first.cleanup().values()), "the first run cleaned up"
    assert host_probe_records(case_dir) == [(RECOVERY_RUN_ID, True)]

    # A landing changed the probe script; the same lane reruns the variant.
    first.settings.probe_script.write_text("print('{\"changed\": true}')\n")
    rerun = rerun_run_id(case_dir, RECOVERY_RUN_ID)
    assert rerun == f"{RECOVERY_RUN_ID}-r2"
    second = probe_module.HostProbeFixture(replace(first.settings, run_id=rerun))
    assert (second.pod, second.state_path) != (first.pod, first.state_path)
    second.create()
    assert second.execute("survey") == {"ok": True}, "no record stood in the way"
    assert json.loads(second.state_path.read_text())["scope"]["run_id"] == rerun
    assert json.loads(first.state_path.read_text())["closed"] is True, "untouched"
    # ... and this run's cleanup fails (the process then ends): its record
    # stays open, with the Pod and ConfigMap it owns.
    api.cleanup_error = True
    with pytest.raises(probe_module.HostProbeError):  # the output is redacted
        second.cleanup()
    assert json.loads(second.state_path.read_text())["closed"] is False

    first.settings.probe_script.write_text("print('{\"changed\": 2}')\n")
    kept = rerun_run_id(case_dir, RECOVERY_RUN_ID)
    assert kept == rerun, "an open record keeps its run id"
    refused = probe_module.HostProbeFixture(replace(first.settings, run_id=kept))
    with pytest.raises(probe_module.HostProbeError, match=RECORD_MISMATCH):
        refused.create()
    assert json.loads(second.state_path.read_text())["closed"] is False
    assert set(api.objects) == {"pod", "configmap"}, "the refusal deleted nothing"
    # The cleanup-only resumption under the script that made the record closes
    # it; the rerun after that starts under -r3.
    first.settings.probe_script.write_text("print('{\"changed\": true}')\n")
    api.cleanup_error = False
    assert not any(probe_module.HostProbeFixture(second.settings).cleanup().values()), (
        "the resumed cleanup under the record's own script proves the node clean"
    )
    assert api.objects == {}, "the resumed cleanup removed the crashed run's objects"
    assert sorted(host_probe_records(case_dir)) == [
        (RECOVERY_RUN_ID, True),
        (rerun, True),
    ]
    assert rerun_run_id(case_dir, RECOVERY_RUN_ID) == f"{RECOVERY_RUN_ID}-r3"


def test_the_runner_names_the_run_after_the_case_directorys_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execute_variant`` (and the plan) derive the run id from the records on
    disk: closed records of the base run id give ``-r2``; the managed job's own
    identity follows its journals, not the run id."""

    cluster = Cluster()
    settings, run_dir = configured(tmp_path, monkeypatch, REGISTRY[NONZERO_EXIT])
    install(monkeypatch, cluster)
    monkeypatch.setattr(attempts, "attempt_records", budgeted_restart)
    spec = REGISTRY[NONZERO_EXIT]
    case_dir = run_dir / "cases" / NONZERO_EXIT
    base_run_id = run_identity(spec.case_id, spec.variant, 1, run_dir)
    job = run_identity(spec.case_id, "nz", 1, case_dir).replace("gl028", "gl28nz")
    probe_record(case_dir, base_run_id, closed=True)

    result = execute(NONZERO_EXIT, settings, run_dir)

    assert (result["verdict"], result["run_id"]) == ("PASS", f"{base_run_id}-r2")
    assert result["identities"] == {job: job}, "no journal yet: the job's base"
    planned = runner.plan_details(
        spec, settings, {"errors": []}, run_id=rerun_run_id(case_dir, base_run_id)
    )
    assert planned["run_id"] == f"{base_run_id}-r2", (
        "the plan names the run id the execute will use"
    )
    # A record of the rerun left open: the run id stays, the job moves on.
    probe_record(case_dir, f"{base_run_id}-r2", closed=False)

    again = execute(NONZERO_EXIT, settings, run_dir)

    assert (again["verdict"], again["run_id"]) == ("PASS", f"{base_run_id}-r2")
    assert again["identities"] == {job: f"{job}-r2"}
