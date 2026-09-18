from __future__ import annotations

from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from tests.regional._cov95_boot032_approval import approve, execute, restart
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


def test_initial_plan_binds_two_distinct_sites_and_all_kubeconfigs(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    binding = lifecycle.NativeBackend(world.settings).initial()
    assert (
        binding["inputs"]["target"]["site_id"]
        != binding["inputs"]["protected"]["site_id"]
    ), "the accepted site must never be the sacrificial site's identity"
    assert set(world.settings.environment()) >= {
        "BOOT032_TARGET_CPU_KUBECONFIG",
        "BOOT032_TARGET_GPU_KUBECONFIG",
        "BOOT032_PROTECTED_CPU_KUBECONFIG",
        "BOOT032_PROTECTED_GPU_KUBECONFIG",
    }, "shared approval must bind all four explicit kubeconfigs"


def test_checked_native_uninstall_pauses_resumes_and_replays_read_only(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "first checked invocation must require a fresh process after native cleanup proof"
    )
    state_path = settings.case_dir / "boot032-state.json"
    paused = journal.read_case(state_path)
    assert paused["phase"] == "RESTART_REQUIRED", "partial case must not claim PASS"
    assert harness.state()["phase"] == "REGISTRY_EXPORTED", (
        "controlled pause must precede native AWS deletion phases"
    )
    assert harness.events == ["cleanup"], "no native AWS delete may precede restart"
    assert harness.native_reads, "pause must run actual native absence verification"
    initial_proof = paused["pause"]["proof"]
    assert initial_proof["registry_sha256"] == harness.snapshot.digest(), (
        "pause must bind the exact original full resource registry"
    )
    result_path = settings.case_dir / f"{contract.CASE_ID}.json"
    assert contract.read_document(result_path)["verdict"] == "NOT_RUN", (
        "a clean pause is not a completed lifecycle verdict"
    )
    assert lifecycle.read_only_preflight(settings, settings.case_dir)["errors"] == [], (
        "read-only preflight must support the original paused transaction"
    )
    restart(world)
    assert execute(world, deadline) == 0, "same approved scope must resume successfully"
    state = harness.state()
    result = contract.read_document(result_path)
    assert state["phase"] == "COMPLETED" and result["verdict"] == "PASS", (
        "PASS requires both native completion and independently verified final state"
    )
    assert harness.exports == 1 and harness.syncs == 1, (
        "resume must reuse the original sealed registry and delete plan"
    )
    assert harness.events.index("delete:cpu") < harness.events.index("delete:aurora"), (
        "native retirement must verify CPU removal before deleting Aurora"
    )
    assert result["checks"]["fresh_process_resume"] is True, (
        "resume needs a new process"
    )
    assert result["checks"]["protected_site_unchanged"] is True, (
        "accepted-site invariants must be read back after all destructive work"
    )
    assert harness.existing == {
        "cluster/gpu/eks",
        "cluster/gpu/hyperpod",
        "aws/aurora/final-snapshot",
    }, "GPU clusters and original-database final snapshot must survive full retirement"
    final, cleanup = journal.final_receipts(
        settings, plan["details"]["binding"], initial_proof
    )
    assert len(final.resources) == len(harness.snapshot.resources) + 1, (
        "final registry must retain every original identity plus the retained snapshot"
    )
    assert set(cleanup["node_targets"]) == {
        "gpu:" + contract.cluster_specs(world.target)[1]["context"]
    }, "cleanup proof must cover every sacrificial GPU cluster"
    before = list(harness.events), harness.cleanup_calls
    harness.no_mutations = True
    assert execute(world, deadline) == 0, "completed retry must verify without mutation"
    assert (harness.events, harness.cleanup_calls) == before, (
        "completed native work must never be invoked again by the acceptance runner"
    )
    assert contract.read_document(result_path)["checks"][
        "completed_replay_read_only"
    ], "replay evidence must distinguish final verification from another teardown"
