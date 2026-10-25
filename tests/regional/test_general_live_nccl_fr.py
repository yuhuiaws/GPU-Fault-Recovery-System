"""GF-GENERAL-LIVE-011 flight-recorder forensics variants over fakes.

The seven live-replay variants of ``general_live_nccl_fr`` are driven with the
fake stall job of ``test_general_live_nccl`` and product records whose hung
triage decision and bundle binding come from the *real* product pipeline over
the *real* test responder (``test_general_live_fr_responder.Site``): the
responders answer the Node Agent's own FIFO requests, ``collect_second_flight``
pairs the dumps, ``classify_hung_signals`` decides, ``bind_hung_bundle`` binds.
So a PASS here means the shape, the responder, the product and the verdict
agree; the perturbation tests then show each verdict stage catching the product
behaviour it exists to catch. The job rendering of a replayed recorder and the
ledger reader are checked against the real renderer and a kubectl fake.
"""

from __future__ import annotations

import copy
import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.hung_summary_codec import encode_summary
from scripts.e2e.regional import general_live_fr_replay as replay
from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional import general_live_nccl_fr as nf
from scripts.e2e.regional import general_live_nccl_job as nj
from scripts.e2e.regional import general_live_nccl_pair as pair
from scripts.e2e.regional import general_live_nccl_roles as roles
from scripts.e2e.regional import managed_workload_fixture as mwf
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    build_registry,
    coverage_rows,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional.test_general_live_fr_responder import NODE, PEER, Site, with_writes
from tests.regional.test_general_live_nccl import (
    SINCE,
    FakeNcclJob,
    _patch_job,
    efa_probe,
    hung_records,
)
from tests.regional.test_general_live_nccl_job import (
    PODS,
    FakePrewarm,
    FakeWorkload,
    KubectlRegional,
    env,
    pods,
    render,
    script_of,
    trainer,
)

_patch_job = _patch_job  # the fake job replaces ManagedNcclJob in every test here
SPECS = {spec.variant: spec for spec in nf.VARIANTS}
NODES = [NODE, PEER]


# --------------------------------------------------------------------------- #
# the real product pipeline over the real responder, once per shape
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, site: Site, *, prepare: Any = None) -> None:
        self.site = site
        self.shape = site.shape
        # The product's own second-dump interval (``orchestration/hung_dag.py``).
        self.decision, self.signals, self.sampled = site.decide(
            prepare=prepare, interval=2.0
        )
        self.bundle = site.bundle(self.decision, self.signals, self.sampled)
        self.ledgers = live_ledgers(site)

    def records(self, since: str = SINCE) -> dict[str, Any]:
        """One stall wave's records with this pipeline's triage and bundle.

        Wave 1's execution holds the first node's result, wave 2's the peer's
        (or nothing when wave 1 sufficed) and the decision; the bundle step is
        the one ``bind_hung_bundle`` returned. ``since`` is the stall time the
        runner asks about, so the incident timing reads as the product's.
        """

        records = hung_records(since)
        workflow = records["workflows"]["wf"]
        steps, executions = workflow["official_steps"], workflow["step_executions"]
        triage = [
            i for i, s in enumerate(steps) if s["operation"] == "COLLECT_HUNG_TRIAGE"
        ]
        rows_by_node: dict[str, list[dict[str, Any]]] = {}
        for signal in self.signals:
            row = {k: v for k, v in signal.items() if k != "node_id"}
            rows_by_node.setdefault(str(signal["node_id"]), []).append(row)

        def stored(node: str) -> dict[str, Any]:
            return {
                node: encode_summary({"node_id": node, "ranks": rows_by_node[node]})
            }

        executions[triage[0]]["details"] = {"node_results": stored(NODE)}
        executions[triage[1]]["details"] = {
            "node_results": stored(PEER) if PEER in self.sampled else {},
            "hung_triage_decision": self.bundle["parameters"]["hung_triage_decision"],
        }
        bundle_index = next(
            i
            for i, s in enumerate(steps)
            if s["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE"
        )
        steps[bundle_index] = {
            "operation": "COLLECT_DIAGNOSTIC_BUNDLE",
            "node_ids": self.bundle["node_ids"],
            "parameters": self.bundle["parameters"],
        }
        return records


def live_ledgers(site: Site) -> dict[str, Any]:
    """The site's ledgers with the pipes and dumps under the live names.

    Offline the responders serve from a temporary directory; a live Pod's
    serve from ``/tmp`` as the job environment names them, which is what the
    runner's fixture oracle checks.
    """

    text = json.dumps(site.ledgers())
    text = text.replace(site.pipe_base, replay.PIPE_BASE).replace(
        site.dump_base, replay.DUMP_BASE
    )
    return json.loads(text)


PIPELINES: dict[str, Pipeline] = {}
SITES: list[Site] = []


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory: pytest.TempPathFactory) -> Any:
    def get(variant: str) -> Pipeline:
        if variant not in PIPELINES:
            shape = replay.SHAPES[variant]()
            site = Site(tmp_path_factory.mktemp(f"fr-{variant}"), shape)
            SITES.append(site)
            prepare = with_writes if variant == "fr_plus_process" else None
            PIPELINES[variant] = Pipeline(site, prepare=prepare)
        return PIPELINES[variant]

    yield get
    for site in SITES:
        site.close()
    SITES.clear()
    PIPELINES.clear()


def view_of(pipeline: Pipeline, records: dict[str, Any] | None = None) -> nf.Reading:
    records = records or pipeline.records()
    rows, unreadable = nf.sampled_rows(records)
    return nf.Reading(
        shape=pipeline.shape,
        nodes=NODES,
        rows=rows,
        unreadable=unreadable,
        decision=nf.decision_of(records),
        bundle=nf.bundle_step(records),
        ledgers=pipeline.ledgers,
    )


CHECKS: dict[str, dict[str, Any]] = {
    "few_candidates": {"candidates": lambda v: nf.candidate_errors(v, targeted=True)},
    "ambiguous_candidates": {
        "candidates": lambda v: nf.candidate_errors(v, targeted=False)
    },
    "different_process_groups": {"separate_groups": nf.separate_groups_errors},
    "fr_temporal_completion": {"watermarks": nf.watermark_errors},
    "fr_second_missing": {
        "fr_withheld": nf.withheld_errors,
        "forensics_gap": nf.second_missing_errors,
    },
    "fr_whole_group": {"whole_group": nf.whole_group_errors},
    "fr_plus_process": {"process_evidence": nf.process_evidence_errors},
}


# --------------------------------------------------------------------------- #
# registry and rendering
# --------------------------------------------------------------------------- #
def test_the_seven_variants_complete_the_011_registry_as_live_replay() -> None:
    registry = build_registry(
        [*nc.VARIANTS, *nf.VARIANTS, *pair.VARIANTS, *roles.VARIANTS],
        rows=coverage_rows(),
        categories=[11],
    )
    assert set(SPECS) == set(replay.SHAPES) == set(CHECKS)
    for spec in nf.VARIANTS:
        assert registry[spec.key] is spec
        assert spec.proof_mode == LIVE_REPLAY and spec.lanes == ("hp",)
        assert spec.implemented and not spec.deferred and spec.approvals == ()
        assert spec.expected_outcome == "PASS"
        assert "test dump responder" in spec.injection
    assert not any(spec.variant in SPECS for spec in nc.VARIANTS), (
        "the native module no longer defers them"
    )


def test_a_replayed_recorder_renders_the_responder_and_turns_pytorchs_off() -> None:
    shape = replay.few_candidates_shape()
    document = render(flight_recorder=shape)
    for role in ("Master", "Worker"):
        container = trainer(pods(document)[role])
        values = env(container)
        assert values["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "0", role
        assert "TORCH_NCCL_DUMP_ON_TIMEOUT" not in values, role
        assert values["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] == replay.PIPE_BASE
        assert values["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == replay.DUMP_BASE
        assert values[replay.SHAPE_ENV] == shape.shape_json(
            world_size=16, procs_per_node=8
        )
        assert (
            json.loads(values[replay.SHAPE_ENV])["ranks"]["15"]["groups"][1]["uid"]
            == "2"
        )
        command = container["command"][-1]
        trainer_end = command.index("\nPY\n") + 4
        responder_start = command.index(
            f"cat >{replay.RESPONDER_PATH} <<'{replay.HEREDOC}'\n"
        )
        launch = command.index(f"python3 -u {replay.RESPONDER_PATH} serve ")
        assert (
            trainer_end
            == responder_start
            < launch
            < command.index("\nexec torchrun", launch)
        )
        body = (
            command[responder_start:]
            .split("\n", 1)[1]
            .split(f"\n{replay.HEREDOC}\n")[0]
        )
        assert body + "\n" == replay.responder_source()
        compile(body, "responder", "exec")
        line = command[launch:].split("\n", 1)[0]
        for token in (
            f"--state-dir {replay.STATE_DIR}",
            f"--pipe-base {replay.PIPE_BASE}",
            f"--dump-base {replay.DUMP_BASE}",
            f"--shape-env {replay.SHAPE_ENV}",
            "--node-rank-env RANK --world-size-env WORLD_SIZE",
            "--procs-per-node 8",
            f"--max-seconds {replay.RESPONDER_MAX_SECONDS}",
            f">{replay.LOG_PATH} 2>&1 &",
        ):
            assert token in line, (token, line)
        assert command.rstrip().endswith(nj.TRAINER_PATH), (
            "torchrun still runs the rendered trainer last"
        )
    assert script_of(document) == nj.trainer_script(), "the trainer itself is unchanged"
    plain = env(trainer(pods(render(flight_recorder=True))["Master"]))
    assert (
        replay.SHAPE_ENV not in plain and "TORCH_NCCL_DEBUG_INFO_TEMP_FILE" not in plain
    )
    assert replay.HEREDOC not in trainer(pods(render())["Master"])["command"][-1]
    assert (
        replay.RESPONDER_MAX_SECONDS
        >= nj.STALL_MAX_SECONDS + nj.PG_TIMEOUT_MARGIN_SECONDS
    )
    assert replay.RESPONDER_MAX_SECONDS >= nj.TRAINER_STEPS * nj.STEP_SECONDS


def test_the_job_wrapper_carries_the_replay_and_reads_the_ledgers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mwf, "ManagedWorkloadFixture", FakeWorkload)
    monkeypatch.setattr(mwf, "ImagePrewarmFixture", FakePrewarm)
    spec = SPECS["fr_whole_group"]
    ledgers = {"job-master-0": {"version": "x", "ranks": {}}, "job-worker-0": None}

    class Regional(KubectlRegional):
        def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
            self.commands.append(arguments)
            pod = arguments[1]
            if ledgers[pod] is None:
                raise RuntimeError("command terminated with exit code 1")
            return json.dumps(ledgers[pod])

    regional = Regional()
    base = make_settings(spec, node=NODE, peer=PEER)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    FakeWorkload.current = [dict(item) for item in PODS]
    shape = replay.whole_group_shape()
    job = nj.ManagedNcclJob(ctx, name="gl11-frwg-a1-deadbeef", flight_recorder=shape)
    assert job.replay is shape and job.flight_recorder is True
    assert job.identity()["flight_recorder_replay"] == "fr_whole_group"
    rendered = yaml.safe_load(
        (ctx.case_dir / "gl11-frwg-a1-deadbeef.source.yaml").read_text()
    )
    assert env(trainer(pods(rendered)["Worker"]))["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "0"
    assert replay.HEREDOC in trainer(pods(rendered)["Worker"])["command"][-1]
    job.start()
    read = job.flight_ledgers()
    assert read["job-master-0"] == ledgers["job-master-0"]
    assert read["job-worker-0"]["error"].startswith("RuntimeError: "), (
        "an unreadable ledger is reported, not raised"
    )
    assert all(
        command[-1] == f"cat {replay.LEDGER_PATH}"
        and command[:5] == ("exec", command[1], "-c", "pytorch", "--")
        for command in regional.commands
    ), regional.commands
    assert (
        nj.ManagedNcclJob(
            ctx, name="gl11-fron-a1-deadbeef", flight_recorder=True
        ).replay
        is None
    )


# --------------------------------------------------------------------------- #
# the verdict stages over the product's own decision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", sorted(CHECKS))
def test_each_variants_checks_pass_on_the_products_decision_over_its_shape(
    pipelines: Any, variant: str
) -> None:
    view = view_of(pipelines(variant))
    assert view.decision is not None and view.bundle
    paired = nf.withheld_errors if variant == "fr_second_missing" else nf.paired_errors
    assert paired(view) == []
    for name, check in CHECKS[variant].items():
        assert check(view) == [], name
    assert replay.responder_errors(view.ledgers, NODES) == []
    assert (
        replay.request_errors(view.ledgers, NODES, sampled_nodes=view.sampled_nodes)
        == []
    )
    excerpt = nf.decision_excerpt(view)
    assert excerpt["sampled_nodes"] == view.sampled_nodes
    if variant == "few_candidates":
        assert excerpt["culprit_ranks"] == [0] and excerpt["bundle_nodes"] == [NODE]
        assert view.sampled_nodes == [NODE], "wave 2 skipped: sufficient local evidence"


def test_the_decision_is_read_from_the_bundle_step_then_the_last_triage_execution(
    pipelines: Any,
) -> None:
    pipeline = pipelines("fr_whole_group")
    records = pipeline.records()
    bound = nf.decision_of(records)
    assert bound == pipeline.bundle["parameters"]["hung_triage_decision"]
    steps = records["workflows"]["wf"]["official_steps"]
    bundle = next(s for s in steps if s["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE")
    bundle["parameters"] = {}
    assert nf.decision_of(records) == bound, "the last triage execution carries it too"
    for execution in records["workflows"]["wf"]["step_executions"]:
        execution["details"].pop("hung_triage_decision", None)
    assert nf.decision_of(records) is None
    view = view_of(pipeline, records)
    assert nf.whole_group_errors(view) == [
        "no hung_triage_decision on the bundle step or a triage execution"
    ]
    assert nf.bundle_step({"incidents": [], "workflows": {}}) == {}
    assert nf.sampled_rows({"incidents": [], "workflows": {}}) == (
        [],
        ["no COLLECT_HUNG_TRIAGE execution in the hung workflow"],
    )


def test_paired_errors_catch_a_missing_rank_an_unpaired_series_and_a_short_interval(
    pipelines: Any,
) -> None:
    pipeline = pipelines("fr_whole_group")
    view = view_of(pipeline)
    short = copy.deepcopy(view)
    short.rows[3]["flight_recorder_series"]["interval_seconds"] = 0.5
    assert nf.paired_errors(short) == [
        "rank 3: the two dumps are 0.5 s apart, under 2 s"
    ]
    unpaired = copy.deepcopy(view)
    unpaired.rows[9]["flight_recorder_series"]["status"] = "SECOND_UNAVAILABLE"
    assert nf.paired_errors(unpaired) == [
        "rank 9: FR series is SECOND_UNAVAILABLE, not PAIRED"
    ]
    partial = copy.deepcopy(view)
    del partial.rows[12]
    assert nf.paired_errors(partial) == [
        "ranks [12] of a sampled node were not sampled"
    ]
    late = copy.deepcopy(view)
    series = late.rows[0]["flight_recorder_series"]
    first = datetime.fromisoformat(series["first_observed_at"])
    series["second_requested_at"] = (first + timedelta(seconds=1)).isoformat()
    series["second_observed_at"] = (first + timedelta(seconds=1.2)).isoformat()
    series["interval_seconds"] = 2.0
    assert nf.paired_errors(late) == [
        "rank 0: the second dump was requested 1.0 s after the first"
    ]
    empty = copy.deepcopy(view)
    empty.rows = []
    assert nf.paired_errors(empty) == ["COLLECT_HUNG_TRIAGE sampled no ranked process"]


def test_candidate_errors_catch_a_missed_lead_a_wrong_scope_and_a_stray_control(
    pipelines: Any,
) -> None:
    view = view_of(pipelines("few_candidates"))
    wrong_lead = copy.deepcopy(view)
    wrong_lead.decision["culprit_ranks"] = [0, 1]
    wrong_lead.decision["candidate_count"] = 2
    errors = nf.candidate_errors(wrong_lead, targeted=True)
    assert errors[0] == "culprit ranks [0, 1] != the planted leads [0]", errors
    conservative = copy.deepcopy(view)
    conservative.bundle["parameters"].update(
        bundle_scope="bounded_triage_nodes",
        capture_process_state=False,
        max_processes=0,
    )
    errors = nf.candidate_errors(conservative, targeted=True)
    assert (
        "bundle scope is 'bounded_triage_nodes', expected 'identity_bound_candidates'"
        in errors
    )
    assert any("max_processes" in item for item in errors), errors
    two_controls = copy.deepcopy(view)
    two_controls.decision["control_ranks"] = [1, 2]
    errors = nf.candidate_errors(two_controls, targeted=True)
    assert "control ranks [1, 2] must be at most one non-culprit rank" in errors
    assert any(item.startswith("bundle targets") for item in errors), (
        "a bundle bound for two PIDs no longer matches three ranks"
    )
    not_eligible = copy.deepcopy(view)
    not_eligible.decision["targeted_capture_eligible"] = False
    assert "targeted_capture_eligible is False, expected True" in nf.candidate_errors(
        not_eligible, targeted=True
    )
    ambiguous = view_of(pipelines("ambiguous_candidates"))
    targeted = copy.deepcopy(ambiguous)
    targeted.bundle["parameters"].update(
        target_pids_by_node={NODE: [100]}, max_processes=1
    )
    errors = nf.candidate_errors(targeted, targeted=False)
    assert errors == [
        f"a conservative bundle still targets PIDs {{'{NODE}': {{100}}}}",
        "bundle max_processes is 1, not 0",
    ], errors
    # Asking for "ambiguous" with few leads is a shape error, not a product one.
    few_as_ambiguous = copy.deepcopy(view)
    few_as_ambiguous.decision["targeted_capture_eligible"] = False
    few_as_ambiguous.bundle["parameters"].update(
        bundle_scope="bounded_triage_nodes",
        capture_process_state=False,
        operator_escalation_required=True,
        target_pids_by_node={},
        max_processes=0,
    )
    assert "only 1 leads were planted; that is not ambiguous" in nf.candidate_errors(
        few_as_ambiguous, targeted=False
    )


def test_whole_group_and_separate_group_errors_catch_an_invented_culprit(
    pipelines: Any,
) -> None:
    view = view_of(pipelines("fr_whole_group"))
    blamed = copy.deepcopy(view)
    blamed.decision["culprit_ranks"] = [5]
    blamed.decision["candidate_count"] = 1
    errors = nf.whole_group_errors(blamed)
    assert "a whole-group stall was blamed on ranks [5]" in errors
    assert "candidate_count 1 for a whole-group stall" in errors
    unobserved = copy.deepcopy(view)
    for finding in unobserved.decision["evidence_branches"]["fr_temporal"][
        "process_groups"
    ]:
        if finding["pg_name"] == "1:tp":
            finding["group_stall_observed"] = False
            finding["time_reason"] = "DISJOINT_WINDOWS"
    unobserved.decision["stalled_process_groups"] = ["2:tp"]
    errors = nf.whole_group_errors(unobserved)
    assert errors[0].startswith(
        "1:tp: group_stall_observed is False (DISJOINT_WINDOWS"
    ), errors
    assert "1:tp is not in stalled_process_groups ['2:tp']" in errors
    separate = view_of(pipelines("different_process_groups"))
    mixed = copy.deepcopy(separate)
    for finding in mixed.decision["evidence_branches"]["fr_temporal"]["process_groups"]:
        if finding["pg_name"] == "2:tp":
            finding["observed_members"] = list(range(16))
    mixed.decision["culprit_ranks"] = list(range(8, 16))
    mixed.decision["candidate_count"] = 8
    errors = nf.separate_groups_errors(mixed)
    assert errors[0].startswith("2:tp: observed [0, 1, 2, 3, 4, 5, 6, 7, 8"), errors
    assert any("were blamed although groups only differ" in item for item in errors), (
        "mixed groups must not blame ranks"
    )
    no_branch = copy.deepcopy(separate)
    no_branch.decision["evidence_branches"]["fr_temporal"] = None
    assert nf.separate_groups_errors(no_branch) == [
        "the decision has no fr_temporal branch (no PAIRED series)"
    ]


def test_watermark_second_missing_and_process_errors_catch_their_cases(
    pipelines: Any,
) -> None:
    temporal = view_of(pipelines("fr_temporal_completion"))
    stuck = copy.deepcopy(temporal)
    groups = stuck.decision["evidence_branches"]["fr_temporal"]["process_groups"]
    default = next(g for g in groups if g["pg_name"] == "0:default_pg")
    default["watermark_deltas_by_rank"]["4"] = {
        "last_enqueued": 2,
        "last_started": 2,
        "last_completed": 0,
    }
    default["completion_advanced_ranks"] = [r for r in range(16) if r != 4]
    errors = nf.watermark_errors(stuck)
    assert errors[0].startswith("0:default_pg rank 4: watermark deltas"), errors
    assert any("completion_advanced_ranks" in item for item in errors), errors
    missing = view_of(pipelines("fr_second_missing"))
    paired = copy.deepcopy(missing)
    paired.rows[0]["flight_recorder_series"]["status"] = "PAIRED"
    assert nf.withheld_errors(paired) == [
        "rank 0: FR series is PAIRED, not SECOND_UNAVAILABLE"
    ]
    invented = copy.deepcopy(missing)
    invented.decision["classification"] = "PLAUSIBLE"
    invented.decision["stalled_process_groups"] = ["0:default_pg"]
    invented.decision["evidence_branches"]["fr_temporal"] = {"process_groups": []}
    errors = nf.second_missing_errors(invented)
    assert "an FR temporal verdict was formed without a paired series" in errors
    assert "stalled_process_groups ['0:default_pg'] from one dump" in errors
    assert "classification PLAUSIBLE without a comparable second dump" in errors
    process = view_of(pipelines("fr_plus_process"))
    silenced = copy.deepcopy(process)
    processes = silenced.decision["evidence_branches"]["process_observations"][
        "processes"
    ]
    for item in processes:
        item["behavior"]["observations"] = []
        item["observations"] = []
    assert nf.process_evidence_errors(silenced) == [
        "no sampled process kept HOST_WRITES_OBSERVED (held ranks rewrite 4 MiB/s)"
    ]
    dropped = copy.deepcopy(process)
    processes = dropped.decision["evidence_branches"]["process_observations"][
        "processes"
    ]
    processes[0]["observations"] = []
    errors = nf.process_evidence_errors(dropped)
    assert errors == [
        f"{NODE}/100: behavior observations ['CPU_ACTIVITY', 'HOST_WRITES_OBSERVED'] dropped"
    ], errors
    pending = copy.deepcopy(process)
    processes = pending.decision["evidence_branches"]["process_observations"][
        "processes"
    ]
    processes[1]["fr_completion_advanced_groups"] = []
    assert nf.process_evidence_errors(pending) == [
        f"{NODE}/101: FR completion did not read as advanced"
    ]


# --------------------------------------------------------------------------- #
# the variants over the fake job
# --------------------------------------------------------------------------- #
def fr_context(spec: Any, tmp_path: Path, records_for: Any) -> Any:
    """A context whose product records are ``records_for(<stall time>)``."""

    base = make_settings(spec, node=NODE, peer=PEER)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})
    calls: list[tuple[str, ...]] = []

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        calls.append(arguments)
        return copy.deepcopy(records_for(arguments[-1]))

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=settings,
    )
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]
    return ctx


@pytest.mark.parametrize("variant", sorted(CHECKS))
def test_every_variant_passes_on_the_products_handling_of_its_replay(
    pipelines: Any, tmp_path: Path, variant: str
) -> None:
    pipeline = pipelines(variant)
    FakeNcclJob.ledgers = pipeline.ledgers
    ctx = fr_context(SPECS[variant], tmp_path, pipeline.records)
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeNcclJob.instances[-1]
    assert job.replay is not None and job.replay.name == variant
    assert job.stall_mode == ("write" if variant == "fr_plus_process" else "quiet")
    assert job.stalled and job.released == 1
    assert result["proof_scope"] == "live-replay"
    assert result["replay"]["shape"] == variant
    assert result["decision"]["sampled_nodes"] == pipeline.sampled
    assert set(result["stages"]) >= {
        "chain",
        "executed",
        "escalated",
        "timing",
        "merged",
        "resumed",
        "replay_requests",
    }
    assert (
        ctx.evidence.case_dir / f"{job.name.split('-')[1]}-fr-ledgers.json"
    ).is_file(), "the FR variants add their own stages to the chain's"
    assert result["attempt"]["job_id"] == job.name


def test_a_missing_or_broken_responder_ledger_is_inconclusive(
    pipelines: Any, tmp_path: Path
) -> None:
    pipeline = pipelines("fr_whole_group")
    FakeNcclJob.ledgers = {"pod-0": pipeline.ledgers["pod-0"]}
    ctx = fr_context(SPECS["fr_whole_group"], tmp_path, pipeline.records)
    with pytest.raises(VariantInconclusive, match="responder did not hold") as info:
        SPECS["fr_whole_group"].execute(ctx)
    assert f"{PEER}: no responder ledger for its Pod" in str(info.value)
    assert FakeNcclJob.instances[-1].released == 1, (
        "the stall was released all the same"
    )


def test_a_product_that_never_asked_or_misjudged_fails(
    pipelines: Any, tmp_path: Path
) -> None:
    pipeline = pipelines("few_candidates")
    unasked = copy.deepcopy(pipeline.ledgers)
    for record in unasked["pod-0"]["ranks"].values():
        record["requests"] = 1
    FakeNcclJob.ledgers = unasked
    result = SPECS["few_candidates"].execute(
        fr_context(SPECS["few_candidates"], tmp_path / "unasked", pipeline.records)
    )
    assert result["verdict"] == "FAIL"
    assert result["stages"]["replay_requests"][0].startswith(
        f"{NODE}: rank 0 received 1 dump request(s)"
    ), result["stages"]["replay_requests"]
    FakeNcclJob.ledgers = pipeline.ledgers

    def conservative(since: str) -> dict[str, Any]:
        records = pipeline.records(since)
        bundle = next(
            s
            for s in records["workflows"]["wf"]["official_steps"]
            if s["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE"
        )
        bundle["parameters"]["bundle_scope"] = "bounded_triage_nodes"
        return records

    result = SPECS["few_candidates"].execute(
        fr_context(SPECS["few_candidates"], tmp_path / "scope", conservative)
    )
    assert result["verdict"] == "FAIL"
    assert any(
        "bundle scope is 'bounded_triage_nodes'" in item for item in result["errors"]
    ), "the wrong bundle scope is named"
    assert result["stages"]["fr_paired"] == [], "the series themselves were fine"
