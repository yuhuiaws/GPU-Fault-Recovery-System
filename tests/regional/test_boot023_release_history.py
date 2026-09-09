"""Contract tests for GF-REGIONAL-BOOT-023.

Every verdict is judged against synthetic evidence -- history ConfigMap
bodies, previous-snapshot ConfigMap items, in-Pod registry probe answers, the
release config parser's refusal -- once on the intended run and once per way
the run can be wrong. The rollback_compatible refusal is exercised against the
real ``regional_release_config`` parser with a synthetic manifest. Nothing
here touches a cluster; the runner's live phases only call these functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_config as CONFIG
from scripts.e2e.regional import boot023_verdicts as verdicts
from scripts.e2e.regional import run_boot023_release_history as boot023
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "rel-42"
DIGEST = "a" * 64


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_boot020() -> None:
    metadata = RegionalCaseMetadata(
        case_id=boot023.CASE_ID,
        title="",
        category="regional-deployability",
        level="staging",
        risk="live-service-action",
        automation="manual",
        procedure="docs/x.md#gf-regional-boot-023",
        predecessor=boot023.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "BOOT023_", prefix
    assert boot023.CONFIRMATION.startswith(prefix), boot023.CONFIRMATION
    assert boot023.PREDECESSOR_CASE_ID == "GF-REGIONAL-BOOT-020", (
        "the NOOP config chain BOOT-020 prepares is what this case consumes"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-BOOT-023", verdicts.CASE_ID


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = boot023.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            boot023.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--noop-config",
            "/tmp/noop.json",
        ]
    )
    assert execute.execute is True and execute.confirm == boot023.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = boot023.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_configure_refuses_a_missing_noop_config(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    arguments = boot023.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "ctx",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--noop-config",
            str(tmp_path / "absent.json"),
        ]
    )
    with pytest.raises(Exception, match="does not exist"):
        boot023.configure(arguments)


def test_the_runner_and_probe_are_executable_with_a_shebang() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_boot023_release_history.py",
        ROOT / "scripts/e2e/regional/probes/boot023_registry_probe.py",
    ):
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3", first


def test_the_scripts_carry_no_site_topology() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_boot023_release_history.py",
        ROOT / "scripts/e2e/regional/probes/boot023_registry_probe.py",
        ROOT / "scripts/e2e/regional/boot023_verdicts.py",
    ):
        source = path.read_text(encoding="utf-8")
        for needle in (
            "/secure/gpu-fault-bootstrap",
            "514385905925",
            "gpu-fault-gpu-1-",
        ):
            assert needle not in source, f"{path.name} embeds {needle}"


# --------------------------------------------------------------------------- #
# Release history (ARCH-H5)
# --------------------------------------------------------------------------- #
def _entry(**overrides: Any) -> dict[str, Any]:
    entry = {
        "timestamp": "2026-09-06T10:00:00Z",
        "release_id": RELEASE_ID,
        "phase": "complete",
        "release_lifecycle": "released",
        "state_sha256": DIGEST,
        "plan_sha256": None,
        "operator": "arn:aws:sts::000000000000:assumed-role/admin/op",
        "command": "gpu-fault-admin deploy --site site.yaml --cluster-token <redacted>",
    }
    entry.update(overrides)
    return entry


def _before() -> list[dict[str, Any]]:
    return [_entry(phase="cpu-staged", timestamp="2026-09-06T09:00:00Z")]


def test_parse_history_keeps_unparsable_lines_visible() -> None:
    parsed = verdicts.parse_history('{"a": 1}\n\nnot json\n[1]\n')
    assert parsed == [{"a": 1}, {"_unparsed": "not json"}, {"_unparsed": "[1]"}], parsed


def test_an_appended_well_formed_noop_entry_passes() -> None:
    before = _before()
    after = [*before, _entry()]
    assert verdicts.history_append_errors(before, after, release_id=RELEASE_ID) == []


def test_a_history_that_did_not_grow_or_was_rewritten_fails() -> None:
    before = _before()
    assert verdicts.history_append_errors(
        before, list(before), release_id=RELEASE_ID
    ), "no growth must fail"
    rewritten = [_entry(phase="rewritten", timestamp="2026-09-06T09:00:00Z"), _entry()]
    errors = verdicts.history_append_errors(before, rewritten, release_id=RELEASE_ID)
    assert any("append-only" in item for item in errors), _text(errors)


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"release_id": "rel-1"}, "not the live release"),
        ({"state_sha256": "zz"}, "state_sha256"),
        ({"plan_sha256": "nope"}, "plan_sha256"),
        ({"operator": ""}, "no operator"),
        ({"timestamp": "yesterday"}, "ISO-8601"),
        ({"phase": ""}, "no phase"),
        ({"command": "deploy --cluster-token abc123"}, "unredacted credential"),
        ({"command": "deploy https://x/?token=abc"}, "token query"),
        ({"command": ""}, "no command"),
    ],
)
def test_each_malformed_history_field_is_named(
    override: dict[str, Any], needle: str
) -> None:
    errors = verdicts.history_entry_errors(
        _entry(**override), release_id=RELEASE_ID, label="entry"
    )
    assert any(needle in item for item in errors), _text(errors)


def test_the_newest_entry_must_record_the_noop_phase() -> None:
    before = _before()
    after = [*before, _entry(phase="cpu-staged")]
    errors = verdicts.history_append_errors(before, after, release_id=RELEASE_ID)
    assert any("complete" in item for item in errors), _text(errors)


def test_a_missing_history_field_is_reported_once() -> None:
    entry = _entry()
    del entry["operator"]
    errors = verdicts.history_entry_errors(entry, release_id=RELEASE_ID, label="e")
    assert any("lacks fields ['operator']" in item for item in errors), _text(errors)


def test_the_history_bound_is_enforced() -> None:
    before = [_entry(phase="p")] * verdicts.HISTORY_MAX_ENTRIES
    after = [*before, _entry()]
    errors = verdicts.history_append_errors(before, after, release_id=RELEASE_ID)
    assert any("above the 200 bound" in item for item in errors), _text(errors)


def test_a_full_ring_that_dropped_its_oldest_entry_still_counts_as_growth() -> None:
    """The live history is a 200-entry ring: once full, an append drops the
    oldest entry and the count stays at 200. The first live run (2026-09-09)
    read that as "did not grow"; the appended entries are found by aligning
    the retained suffix of ``before`` with the prefix of ``after``."""

    before = [
        _entry(
            phase="p", timestamp=f"2026-09-06T{index // 60:02d}:{index % 60:02d}:00Z"
        )
        for index in range(verdicts.HISTORY_MAX_ENTRIES)
    ]
    newest = _entry()
    after = [*before[1:], newest]

    assert verdicts.appended_history_entries(before, after) == [newest]
    assert verdicts.history_append_errors(before, after, release_id=RELEASE_ID) == []

    two = [*before[2:], _entry(phase="cpu-staged"), newest]
    assert verdicts.appended_history_entries(before, two) == [
        _entry(phase="cpu-staged"),
        newest,
    ]


def test_entries_lost_before_the_ring_is_full_are_a_rewrite() -> None:
    before = _before() + [_entry(phase="q", timestamp="2026-09-06T09:30:00Z")]
    after = [before[1], _entry()]  # dropped the first entry while far below the bound
    assert verdicts.appended_history_entries(before, after) is None
    errors = verdicts.history_append_errors(before, after, release_id=RELEASE_ID)
    assert any("append-only" in item for item in errors), _text(errors)
    assert verdicts.appended_history_entries([], [_entry()]) == [_entry()]


def test_the_runner_reads_the_appended_entries_through_the_ring_aware_helper() -> None:
    source = (ROOT / "scripts/e2e/regional/run_boot023_release_history.py").read_text(
        encoding="utf-8"
    )
    assert "verdicts.appended_history_entries(history_before, history_after)" in source
    assert "history_after[len(history_before) :]" not in source


def test_the_noop_may_move_only_the_state_timestamp_of_the_runtime_identity() -> None:
    from scripts.e2e.regional.regional_live_fixture import identity_without_state_fields

    before = {
        "release_state": {
            "release_id": "r",
            "updated_at_epoch": 1,
            "phase": "complete",
        },
        "deployments": {"cpu": {"gpu-fault-api-ha": {"generation": 3}}},
    }
    after = {
        "release_state": {
            "release_id": "r",
            "updated_at_epoch": 2,
            "phase": "complete",
        },
        "deployments": {"cpu": {"gpu-fault-api-ha": {"generation": 3}}},
    }
    fields = ("updated_at_epoch",)
    assert identity_without_state_fields(
        before, fields
    ) == identity_without_state_fields(after, fields)
    assert identity_without_state_fields(before, ()) != identity_without_state_fields(
        after, ()
    ), "without the allowance the timestamp still counts as drift"
    moved = {**after, "deployments": {"cpu": {"gpu-fault-api-ha": {"generation": 4}}}}
    assert identity_without_state_fields(
        before, fields
    ) != identity_without_state_fields(moved, fields), (
        "a Deployment generation change is never allowed"
    )
    source = (ROOT / "scripts/e2e/regional/run_boot023_release_history.py").read_text(
        encoding="utf-8"
    )
    assert 'mutable_state_fields=("updated_at_epoch",)' in source


def test_the_mirror_must_carry_the_appended_tail() -> None:
    appended = [_entry()]
    line = json.dumps(appended[0], sort_keys=True, separators=(",", ":"))
    assert verdicts.mirror_errors(["older", line], appended) == []
    assert verdicts.mirror_errors([], appended), "an empty mirror must fail"
    other = json.dumps(_entry(phase="other"), sort_keys=True)
    errors = verdicts.mirror_errors([other], appended)
    assert any("differs" in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Previous snapshots
# --------------------------------------------------------------------------- #
def _snapshot_item(name: str, digest: str, created: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "creationTimestamp": created,
            "annotations": {verdicts.PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION: digest},
        }
    }


def test_snapshot_groups_group_chunks_by_digest_newest_first() -> None:
    items = [
        _snapshot_item("prev-a-0", "d1", "2026-09-06T08:00:00Z"),
        _snapshot_item("prev-a-1", "d1", "2026-09-06T08:00:01Z"),
        _snapshot_item("prev-b-0", "d2", "2026-09-06T09:00:00Z"),
        {"metadata": {"name": "legacy-0", "creationTimestamp": "2026-09-06T07:00:00Z"}},
    ]
    groups = verdicts.snapshot_groups(items)
    assert groups == [["prev-b-0"], ["prev-a-0", "prev-a-1"], ["legacy-0"]], groups


def test_snapshot_retention_requires_the_bound_and_no_noop_change() -> None:
    groups = [["a"], ["b"], ["c"]]
    assert verdicts.snapshot_retention_errors(groups, list(groups)) == []
    errors = verdicts.snapshot_retention_errors(groups, [*groups, ["d"]])
    assert any("above the 3 retained" in item for item in errors), _text(errors)
    assert any("changed the previous-snapshot" in item for item in errors), _text(
        errors
    )


# --------------------------------------------------------------------------- #
# Registry durable head vs Secret (ARCH-H3)
# --------------------------------------------------------------------------- #
def _probe(**overrides: Any) -> dict[str, Any]:
    probe = {
        "pod": "gpu-fault-api-ha-1",
        "secret_config_sha256": DIGEST,
        "durable_config_sha256": DIGEST,
        "head_generation": 7,
        "healthz": {
            "status": 200,
            "payload": {
                "status": "ok",
                "regional_registry": {
                    "ready": True,
                    "secret_drift": False,
                    "secret_config_sha256": DIGEST,
                },
            },
        },
        "livez": {"status": 200, "payload": {"status": "alive"}},
    }
    probe.update(overrides)
    return probe


def test_a_pod_whose_head_matches_its_secret_passes() -> None:
    assert verdicts.registry_probe_errors([_probe()]) == []


def test_no_probe_answer_fails_closed() -> None:
    assert verdicts.registry_probe_errors([]) == [
        "no CPU Pod answered the registry probe"
    ]


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"durable_config_sha256": "b" * 64}, "did not publish durably"),
        ({"secret_config_sha256": None}, "not a sha256"),
        ({"healthz": {"status": 503, "payload": {}}}, "/healthz returned 503"),
        (
            {
                "healthz": {
                    "status": 200,
                    "payload": {
                        "regional_registry": {"ready": True, "secret_drift": True}
                    },
                }
            },
            "secret_drift=True",
        ),
        (
            {
                "healthz": {
                    "status": 200,
                    "payload": {
                        "regional_registry": {"ready": False, "secret_drift": False}
                    },
                }
            },
            "not ready",
        ),
        (
            {
                "healthz": {
                    "status": 200,
                    "payload": {
                        "regional_registry": {
                            "ready": True,
                            "secret_drift": False,
                            "secret_config_sha256": "c" * 64,
                        }
                    },
                }
            },
            "differs from the probe",
        ),
        ({"livez": {"status": 503}}, "/livez returned 503"),
    ],
)
def test_each_registry_probe_failure_is_named(
    override: dict[str, Any], needle: str
) -> None:
    errors = verdicts.registry_probe_errors([_probe(**override)])
    assert any(needle in item for item in errors), _text(errors)


def test_the_head_generation_must_not_move_across_a_noop() -> None:
    before = [_probe()]
    assert verdicts.registry_stability_errors(before, [_probe()]) == []
    errors = verdicts.registry_stability_errors(before, [_probe(head_generation=8)])
    assert any("generation moved" in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Manifest rollback flag (ARCH-H2) against the real parser
# --------------------------------------------------------------------------- #
def _manifest(database: dict[str, object] | None) -> dict[str, Any]:
    components = {
        name: {"sha256": DIGEST}
        for name in (
            "collector",
            "cpu",
            "dcgm",
            "endpoint",
            "executor",
            "node",
            "observability",
            "schema",
            "watcher",
        )
    }
    delivery = {
        "schema_version": 1,
        "runtime_prebuilt": True,
        "components": components,
        "images": {
            name: {"reference": f"registry/{name}@sha256:" + DIGEST}
            for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
        },
        "node_template_inputs": {"sha256": "b" * 64},
    }
    delivery["sha256"] = CONFIG.canonical_sha256(delivery)
    manifest: dict[str, Any] = {
        "deployable": True,
        "delivery": delivery,
        "components": {"node_bundle": {"template_sha256": "b" * 64}},
    }
    if database is not None:
        manifest["database"] = database
    return manifest


def test_the_real_parser_refuses_the_flag_and_the_verdict_accepts_its_message() -> None:
    message = boot023.rollback_flag_rejection(
        _manifest({"rollback_compatible": True}), CONFIG
    )
    assert message, "the parser must refuse database.rollback_compatible: true"
    assert verdicts.rollback_flag_errors(message) == [], message


def test_a_manifest_without_the_flag_is_accepted_and_that_is_a_verdict_failure() -> (
    None
):
    message = boot023.rollback_flag_rejection(_manifest(None), CONFIG)
    assert message == "", message
    errors = verdicts.rollback_flag_errors(message)
    assert any("was accepted" in item for item in errors), _text(errors)


def test_a_refusal_that_does_not_explain_itself_fails() -> None:
    errors = verdicts.rollback_flag_errors(
        "release manifest declares rollback_compatible"
    )
    assert any("does not explain itself" in item for item in errors), _text(errors)


def test_manifest_with_rollback_flag_sets_the_claim_on_the_configs_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(None)), encoding="utf-8")
    config_path = tmp_path / "noop.json"
    config_path.write_text(
        json.dumps({"release": {"manifest": "manifest.json"}}), encoding="utf-8"
    )
    flagged = boot023.manifest_with_rollback_flag(config_path)
    assert flagged["database"] == {"rollback_compatible": True}, flagged["database"]
    assert flagged["deployable"] is True, "the rest of the manifest is untouched"
    (tmp_path / "bare.json").write_text(json.dumps({"release": {}}), encoding="utf-8")
    with pytest.raises(Exception, match="names no manifest"):
        boot023.manifest_with_rollback_flag(tmp_path / "bare.json")


# --------------------------------------------------------------------------- #
# Preflight and plan identity
# --------------------------------------------------------------------------- #
def _preflight(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "classification": {"kind": "NOOP", "changed": []},
        "release_id": RELEASE_ID,
        "state_phase": "complete",
        "probes": [_probe()],
        "rollback_flag_message": (
            "release manifest declares database.rollback_compatible: true, which "
            "the runtime cannot honour: exact schema ... --accept-schema-change"
        ),
        "predecessor_valid": True,
        "tests_passed": True,
        "history_before": _before(),
        "snapshot_groups_before": [["a"], ["b"]],
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def test_the_intended_preflight_is_clean() -> None:
    assert _preflight() == []


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"classification": {"kind": "FULL", "changed": ["schema"]}}, "not NOOP"),
        ({"release_id": ""}, "no release_id"),
        ({"state_phase": "cpu-staged"}, "needs a completed transaction"),
        ({"predecessor_valid": False}, "predecessor evidence is not PASS"),
        ({"tests_passed": False}, "focused regression"),
        ({"history_before": [{"_unparsed": "x"}]}, "unparsable"),
        ({"snapshot_groups_before": [["a"], ["b"], ["c"], ["d"]]}, "above the 3"),
        ({"rollback_flag_message": ""}, "was accepted"),
        ({"probes": []}, "no CPU Pod answered"),
    ],
)
def test_each_preflight_stop_is_named(override: dict[str, Any], needle: str) -> None:
    errors = _preflight(**override)
    assert any(needle in item for item in errors), _text(errors)


def _preflight_document() -> dict[str, Any]:
    return {
        "release_id": RELEASE_ID,
        "classification": {"kind": "NOOP", "changed": [], "phase": "complete"},
        "history": _before(),
        "snapshot_groups": [["a"]],
        "registry_probes": [_probe(), _probe(pod="w-1", head_generation=7)],
    }


def test_plan_identity_pins_release_classification_and_history_shape() -> None:
    identity = boot023.plan_identity(_preflight_document())
    assert identity == {
        "release_id": RELEASE_ID,
        "classification": "NOOP",
        "state_phase": "complete",
        "history_entries": 1,
        "snapshot_groups": 1,
        "head_generations": [7],
    }, identity


def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / boot023.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="plan drifted"):
        boot023.verify_plan_identity(case_dir, _preflight_document())


def test_plan_details_name_the_noop_hard_stop_and_no_rollout() -> None:
    settings = boot023.Settings.__new__(boot023.Settings)
    object.__setattr__(settings, "noop_config", Path("/tmp/noop.json"))
    document = {**_preflight_document(), "predecessor": {"valid": True}}
    details = boot023.plan_details(settings, document)
    assert details["risk"] == "live-service-action", details["risk"]
    conditions = "\n".join(details["stop_conditions"])
    assert "classify as NOOP" in conditions, conditions
    assert "rollback_compatible" in conditions, conditions
    assert details["rollback"][
        "no_deployment_secret_or_registry_revision_is_written"
    ], details["rollback"]
