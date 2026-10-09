"""``gpu-fault-admin store-maintenance``: plan is read-only, execute needs the
exact token and an approved reference, takes the site operation lock,
re-plans in the Pod and records before/after sizes as evidence."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import store_maintenance as module
from gpu_fault.admin.bootstrap import BootstrapError
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.admin.operation_lock import site_operation_lock

SERVER = {
    "version": "PostgreSQL 16.4 on x86_64",
    "version_num": 160004,
    "database": "gpu_fault",
    "reindex_concurrently_supported": True,
}


def _table(name: str, action: str, **sizes) -> dict:
    values = {
        "live_rows": 40,
        "dead_rows": 12,
        "heap_pages": 1188,
        "heap_bytes": 1188 * 8192,
        "index_bytes": 2 * 1048576,
        "toast_bytes": 0,
        "total_bytes": 1188 * 8192 + 2 * 1048576,
    }
    values.update(sizes)
    return {
        "table": name,
        "action": action,
        **values,
        "reloptions": ["fillfactor=50"],
        "toast_reloptions": [],
        "indexes": [{"name": f"{name}_pkey", "bytes": 1048576, "valid": True}],
    }


PLAN = {
    "mode": "plan",
    "server": SERVER,
    "tables": [
        _table("gpu_fault_processor_lanes", "VACUUM (FULL, ANALYZE)"),
        _table(
            "gpu_fault_gpu_metric_latest",
            "REINDEX INDEX CONCURRENTLY + VACUUM (ANALYZE)",
            live_rows=1216,
            heap_pages=23435,
            heap_bytes=23435 * 8192,
            index_bytes=124 * 1048576,
        ),
    ],
    "missing_tables": [],
    "plan_sha256": "a" * 64,
}
EXECUTED = {
    "mode": "execute",
    "status": "SUCCEEDED",
    "server": SERVER,
    "plan_sha256": "a" * 64,
    "actions": [
        {
            "table": "gpu_fault_processor_lanes",
            "action": "VACUUM (FULL, ANALYZE)",
            "status": "SUCCEEDED",
            "steps": [
                {
                    "statement": "VACUUM (FULL, ANALYZE) gpu_fault_processor_lanes",
                    "seconds": 0.2,
                }
            ],
            "before": {"heap_pages": 1188, "live_rows": 40},
            "after": {"heap_pages": 1, "live_rows": 40},
        }
    ],
    "errors": [],
}


def _site(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        environment={},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "namespace": "gpu-fault-system",
            "runtime_profile": {"version": "hyperpod-v1"},
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
    )


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["store-maintenance", "--state-dir", str(tmp_path), *extra]


class _Pod:
    """The exec channel, recorded: plan then execute, each with its payload."""

    def __init__(self, monkeypatch, *, executed: dict | None = None) -> None:
        self.calls: list[tuple[dict, str]] = []
        self.executed = EXECUTED if executed is None else executed

        def run_script(site, payload, *, script):
            self.calls.append((payload, script))
            return json.loads(
                json.dumps(PLAN if payload["mode"] == "plan" else self.executed)
            )

        monkeypatch.setattr(module, "run_control_plane_script", run_script)


def test_plan_mode_prints_the_review_and_takes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    pod = _Pod(monkeypatch)
    monkeypatch.setattr(
        module,
        "administrator_operation_lock",
        lambda _path: pytest.fail("plan must not take the site operation lock"),
    )
    arguments = cli.parser().parse_args(_argv(tmp_path, "--reference", "CHG-7"))

    assert module.run_store_maintenance_command(arguments, site=_site(tmp_path)) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "plan" and printed["reference"] == "CHG-7"
    assert [item["table"] for item in printed["tables"]] == [
        "gpu_fault_processor_lanes",
        "gpu_fault_gpu_metric_latest",
    ]
    assert [payload for payload, _ in pod.calls] == [{"mode": "plan"}]
    assert "def execute(" in pod.calls[0][1], "the Pod runs the maintenance module"
    assert not (tmp_path / module.STATE_ROOT).exists(), "a plan leaves no evidence"
    assert is_readonly_command(arguments) is True


def test_execute_requires_the_exact_token_and_an_approved_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = _Pod(monkeypatch)
    site = _site(tmp_path)
    wrong = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-7", "--confirm", "YES")
    )
    with pytest.raises(BootstrapError, match="requires --confirm STORE_MAINTENANCE"):
        module.run_store_maintenance_command(wrong, site=site)
    bad = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "bad ref!", "--confirm", module.CONFIRMATION)
    )
    with pytest.raises(BootstrapError, match="reference is invalid"):
        module.run_store_maintenance_command(bad, site=site)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(_argv(tmp_path, "--confirm", module.CONFIRMATION))
    assert pod.calls == [], "refusals precede every Pod exec"


def test_execute_takes_the_lock_replans_and_records_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    pod = _Pod(monkeypatch)
    original = _site(tmp_path)
    events: list[str] = []

    @contextmanager
    def lock(path):
        assert path == tmp_path
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    def reload(site):
        assert site is original and events == ["lock"]
        events.append("reload")
        return site

    monkeypatch.setattr(module, "administrator_operation_lock", lock)
    monkeypatch.setattr(module, "reload_site_for_mutation", reload)
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-7", "--confirm", module.CONFIRMATION)
    )

    assert module.run_store_maintenance_command(arguments, site=original) == 0

    assert events == ["lock", "reload", "unlock"]
    assert [payload for payload, _ in pod.calls] == [
        {"mode": "plan"},
        {"mode": "execute", "plan_sha256": "a" * 64},
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "SUCCEEDED" and printed["reference"] == "CHG-7"
    assert printed["operator"].endswith("test-operator"), (
        "the evidence names the operator identity"
    )
    assert printed["plan"] == PLAN["tables"]
    assert printed["actions"] == EXECUTED["actions"]
    evidence = Path(printed["evidence_path"])
    assert evidence.parent == tmp_path / module.STATE_ROOT
    assert evidence.name.endswith("-" + "a" * 16 + ".json"), (
        "the evidence file carries the plan digest prefix"
    )
    recorded = json.loads(evidence.read_text())
    assert recorded["actions"][0]["before"]["heap_pages"] == 1188
    assert recorded["actions"][0]["after"]["heap_pages"] == 1
    assert recorded["plan_sha256"] == "a" * 64
    assert is_readonly_command(arguments) is False


def test_a_failed_table_is_recorded_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    failed = json.loads(json.dumps(EXECUTED))
    failed["status"] = "FAILED"
    failed["actions"][0]["status"] = "FAILED"
    failed["errors"] = [
        {"table": "gpu_fault_processor_lanes", "error": "LockNotAvailable"}
    ]
    _Pod(monkeypatch, executed=failed)
    monkeypatch.setattr(module, "reload_site_for_mutation", lambda site: site)
    monkeypatch.setattr(module, "administrator_operation_lock", _no_lock)
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-7", "--confirm", module.CONFIRMATION)
    )

    assert module.run_store_maintenance_command(arguments, site=_site(tmp_path)) == 1

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "FAILED"
    assert printed["errors"] == failed["errors"]
    recorded = json.loads(Path(printed["evidence_path"]).read_text())
    assert recorded["status"] == "FAILED", "a failed run is evidence too"


@contextmanager
def _no_lock(_path):
    yield


def test_execute_refuses_while_another_mutation_holds_the_site_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = _Pod(monkeypatch)
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with site_operation_lock(tmp_path, wait=False):
            held.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(timeout=10), "the holder must take the site lock first"
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-7", "--confirm", module.CONFIRMATION)
    )
    try:
        with pytest.raises(BootstrapError, match="another administrator mutation"):
            module.run_store_maintenance_command(arguments, site=_site(tmp_path))
    finally:
        release.set()
        holder.join(timeout=10)
    assert pod.calls == [], "a held lock refuses before any Pod exec"


def test_unexpected_pod_results_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    monkeypatch.setattr(
        module, "run_control_plane_script", lambda *a, **k: {"tables": "no"}
    )
    with pytest.raises(BootstrapError, match="unexpected shape"):
        module.plan_store_maintenance(site)
    weird = json.loads(json.dumps(EXECUTED))
    weird["status"] = "DONE"
    _Pod(monkeypatch, executed=weird)
    with pytest.raises(BootstrapError, match="unexpected result"):
        module.run_store_maintenance(site, reference="CHG-7", identity=lambda: "op")


def test_plan_lines_render_pages_mebibytes_and_actions() -> None:
    lines = module.plan_lines(PLAN)
    assert lines[0].startswith("store: gpu_fault on PostgreSQL 16.4"), lines[0]
    assert "REINDEX INDEX CONCURRENTLY supported: yes" in lines[1]
    assert any(
        line.startswith("gpu_fault_gpu_metric_latest: 1216 live rows")
        and "23435 heap pages (183.1 MiB)" in line
        and "indexes 124.0 MiB" in line
        and line.endswith("-> REINDEX INDEX CONCURRENTLY + VACUUM (ANALYZE)")
        for line in lines
    ), lines


def test_cli_registers_the_verb_and_dispatches_with_the_loaded_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choices = cli.parser()._subparsers._group_actions[0].choices
    assert "store-maintenance" in choices
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["store-maintenance", "--help"])
    site = _site(tmp_path)
    seen: list[tuple] = []
    monkeypatch.setattr(cli, "load_site", lambda path, repository_root=None: site)
    monkeypatch.setattr(
        cli,
        "run_store_maintenance_command",
        lambda arguments, *, site: seen.append((arguments.reference, site)) or 0,
    )
    arguments = cli.parser().parse_args(_argv(tmp_path, "--reference", "CHG-7"))
    assert cli.run(arguments) == 0
    assert seen == [("CHG-7", site)]
