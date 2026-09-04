"""The operator mode that closes a retired generation before the deploy.

This mode exists because of an ordering trap. A retired generation is exactly
what the ``workflow_safety`` release preflight refuses to roll past, so the
release carrying the dispatcher's revocation sweep cannot deploy while one is
open -- the fix is blocked by the bug it fixes. The operator therefore has to
close the record first, against the image that is already running, whose
``gpu_fault.retired_generation`` does not exist yet.

So this mode alone ships the decision as *source* into the Pod, the way the
engine ships its control-plane probes. That makes two things worth pinning that
no other admin mode needs: the shipped text has to stay executable against the
previously deployed runtime, and it has to stay the module's own file rather than
a copy that can drift.

The rest is the same two-phase discipline as ``test_admin_workflow_reconcile.py``
-- plan to disk, digest-bound apply, re-plan and compare before writing, archive
what was approved -- with one difference: an open remote command is not a
blocker here, because cancelling it is the first half of the apply.
"""

from __future__ import annotations

import ast
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault import retired_generation
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError

WORKFLOW = "workflow-45c6b6b7-2cde-4a46-853e-11abeb55a142"
SUCCESSOR = "workflow-successor"


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": ("arn:aws:eks:us-west-2:123456789012:cluster/cpu-control"),
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _item(**overrides) -> dict:
    item = {
        "request_id": WORKFLOW,
        "incident_id": "incident-a",
        "cluster_id": "gpu-a",
        "node_ids": ["node-a", "node-b", "node-c"],
        "incident_state": "RECOVERED",
        "status": "RUNNING",
        "fencing_token": 1,
        "incident_fencing_token": 4,
        "successor_workflow_id": SUCCESSOR,
        "open_remote_commands": [],
        "unsettled_local_steps": [],
        "completed_destructive_operations": [],
        "pending_destructive_operations": ["STOP_WORKLOADS"],
        "eligible": True,
        "cancellable": True,
        "reasons": [],
    }
    item.update(overrides)
    return item


def _runtime_plan(*items: dict) -> dict:
    return {
        "schema_version": 1,
        "mode": "retired-generation-plan",
        "evaluated_at": "2026-09-04T11:17:00+00:00",
        "plan_sha256": "b" * 64,
        "items": list(items) or [_item()],
    }


def _fake_runner(monkeypatch: pytest.MonkeyPatch, *items: dict) -> list[dict]:
    """Record every payload the driver would have been handed."""

    calls: list[dict] = []

    def run(_site_value, payload, *, script=None):
        calls.append(payload)
        if payload["mode"] == "plan":
            return _runtime_plan(*items)
        return {
            "mode": "retired-generation-apply",
            "records_deleted": 0,
            "applied_workflow_ids": [WORKFLOW],
            "cancelled_remote_commands": [],
            "settled_plan_sha256": "b" * 64,
            "archive_eligible_incident_ids": [],
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    return calls


def test_the_shipped_script_is_the_module_itself_plus_a_driver() -> None:
    """No second copy of the decision to keep in step.

    A hand-copied excerpt would be a fork of the predicate that decides whether a
    destructive workflow may be terminalized, reviewed once and then left behind
    by every change to the real one. Shipping the file means the reviewed rules
    and the operated rules are the same bytes.
    """

    script = reconcile.retired_generation_script()
    source = Path(retired_generation.__file__).read_text(encoding="utf-8")

    assert script.startswith(source), (
        "the shipped script is no longer the module's own file, so the reviewed "
        "predicate and the operated one can now drift"
    )
    assert script[len(source) :] == reconcile.RETIRED_GENERATION_DRIVER


def test_the_shipped_script_compiles_and_calls_both_entry_points() -> None:
    script = reconcile.retired_generation_script()
    tree = ast.parse(script)
    compile(script, "<retired-generation>", "exec")

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    for entry_point in (
        "build_retired_generation_plan",
        "apply_retired_generation_plan",
    ):
        assert entry_point in called, f"the driver no longer calls {entry_point}"
        assert entry_point in defined, (
            f"{entry_point} is not defined in the shipped source, so the driver "
            "would fail against the deployed image"
        )


def test_the_shipped_source_imports_nothing_the_old_image_may_lack() -> None:
    """The constraint that makes the pre-deploy close possible at all.

    This runs against the image that is *already* deployed. Anything imported
    from a module that only the incoming release introduces makes the script fail
    on import -- and it fails at the one moment there is no other way to clear
    the release blocker. ``gpu_fault.workflow_resolution`` and
    ``gpu_fault.execution`` are the specific hazards: they import in the other
    direction, so an innocuous-looking convenience import here is also a cycle.

    The modules below are the reviewed allowlist, all of them long-standing. Add
    to it only after checking the deployed image actually has what you added --
    or put the import behind ``except ImportError`` like the one exception below.
    """

    allowed = {
        "__future__",
        "hashlib",
        "json",
        "sys",
        "datetime",
        "typing",
        "gpu_fault.models",
        "gpu_fault.operation_registry",
        "gpu_fault.remote_command_models",
        "gpu_fault.store",
        "gpu_fault.app",
    }
    tree = ast.parse(reconcile.retired_generation_script())

    def modules(node: ast.AST) -> set[str]:
        found = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                found.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.add(child.module)
        return found

    guarded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = {
            name.id
            for handler in node.handlers
            for name in ast.walk(handler.type)
            if isinstance(name, ast.Name)
        }
        if "ImportError" in caught:
            guarded.update(module for body in node.body for module in modules(body))

    imported = modules(tree)

    assert imported - guarded <= allowed, (
        "the shipped retired-generation decision imports modules outside the "
        f"reviewed allowlist: {sorted(imported - guarded - allowed)}"
    )
    assert "gpu_fault.execution" in guarded, (
        "the restart-reservation release reaches into gpu_fault.execution, whose "
        "shape in the deployed image is exactly what this script cannot assume; "
        "that import has to stay behind except ImportError"
    )


def test_plan_binds_the_site_and_the_runtime_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)

    plan = reconcile.plan_retired_generation_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(WORKFLOW,)
    )

    assert plan["mode"] == "retired-generation-plan"
    assert plan["runtime_plan_sha256"] == "b" * 64
    assert plan["site_identity"]["site_sha256"] == "a" * 64
    path = tmp_path / reconcile.RETIRED_GENERATION_PLAN_PATH
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == plan


def test_apply_cancels_and_revokes_without_deleting_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open remote command is work for the apply, not a reason to refuse.

    Cancelling a command belonging to a generation the incident has already moved
    past is correct under every reading -- there is no version of events in which
    that ``STOP_WORKLOADS`` should still reach the nodes. So an item blocked only
    by open commands is applied, and the runtime does the cancel-then-revoke in
    that order.
    """

    calls = _fake_runner(
        monkeypatch,
        _item(
            open_remote_commands=["command-stop-workloads"],
            eligible=False,
            cancellable=True,
            reasons=["workflow has open remote commands: command-stop-workloads"],
        ),
    )
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )

    result = reconcile.apply_retired_generation_reconcile(
        site,
        tmp_path,
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-6459c07ea279",
    )

    assert calls[-1]["mode"] == "apply"
    assert calls[-1]["reference"] == "pre-deploy-6459c07ea279"
    assert calls[-1]["plan_sha256"] == "b" * 64
    assert calls[-1]["workflow_ids"] == [WORKFLOW]
    assert result["records_deleted"] == 0, (
        "reconciliation terminalizes records; deleting one would destroy the "
        "audit trail an operator needs afterwards"
    )
    assert result["admin_plan_sha256"] == plan["plan_sha256"]
    archive = tmp_path / reconcile.RETIRED_GENERATION_HISTORY_PATH / plan["plan_sha256"]
    assert (archive / "plan.json").is_file(), "the approved plan was not archived"
    assert (archive / "applied.json").is_file(), "the apply result was not archived"
    assert not (tmp_path / reconcile.RETIRED_GENERATION_PLAN_PATH).exists(), (
        "a consumed plan must not stay on disk to be applied twice"
    )


@pytest.mark.parametrize(
    "name,reasons",
    [
        (
            "it already completed something destructive",
            ["workflow completed destructive operations: STOP_WORKLOADS"],
        ),
        (
            "an adapter action never reported back",
            ["workflow has unsettled adapter actions on steps: 1"],
        ),
        ("the incident moved again", ["workflow is no longer a retired generation"]),
    ],
)
def test_apply_refuses_an_item_that_needs_an_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, reasons: list[str]
) -> None:
    """Refused here as well as in the runtime, and that is deliberate.

    The runtime re-derives every condition inside its transaction, so this check
    is not what makes the apply safe. What it makes possible is the operator
    reading *why* from the plan they approved, instead of from a Pod's stderr
    after a partial pass.
    """

    _fake_runner(monkeypatch, _item(eligible=False, cancellable=False, reasons=reasons))
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )

    with pytest.raises(BootstrapError, match="ineligible records"):
        reconcile.apply_retired_generation_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-6459c07ea279",
        )

    assert (tmp_path / reconcile.RETIRED_GENERATION_PLAN_PATH).is_file(), (
        f"the plan must survive a refusal when {name}"
    )


def test_apply_refuses_a_plan_that_changed_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live record is being dispatched every eighty seconds while you read.

    Between plan and apply the incident can advance again, a command can settle,
    or a step can be handed to an adapter -- each of which changes what the apply
    would do. The digest is what makes the operator's review bind the write
    rather than merely precede it.
    """

    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    _fake_runner(monkeypatch, _item(unsettled_local_steps=[1], status="SAFETY_PENDING"))

    with pytest.raises(BootstrapError, match="plan changed before apply"):
        reconcile.apply_retired_generation_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-6459c07ea279",
        )


def test_apply_refuses_a_plan_from_the_other_reconcile_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two modes share a state directory and refuse different things.

    ``workflow-reconcile`` closes a BLOCKED record after a verified restore and
    checks node scheduling evidence; this one terminalizes an open record and
    cancels its commands. Applying one plan through the other's rules would run
    an unreviewed write, so the saved mode is checked rather than assumed.
    """

    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    crossed = dict(plan, mode="workflow-reconcile-plan")
    (tmp_path / reconcile.RETIRED_GENERATION_PLAN_PATH).write_text(
        json.dumps(crossed), encoding="utf-8"
    )

    with pytest.raises(BootstrapError, match="not a retired generation plan"):
        reconcile.apply_retired_generation_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-6459c07ea279",
        )


def test_apply_refuses_a_plan_rendered_for_another_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan names workflows, not a cluster, so the site has to be bound.

    Workflow IDs carry no region or account, and this command terminalizes
    destructive work. Without the identity check, a plan reviewed against staging
    could be applied against whatever site the current state directory points at.
    """

    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    other = _site(tmp_path)
    other.source_sha256 = "c" * 64

    with pytest.raises(BootstrapError, match="site identity changed"):
        reconcile.apply_retired_generation_reconcile(
            other,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-6459c07ea279",
        )


@pytest.mark.parametrize(
    "reference", ["", "  ", "x", "ref with spaces", "bad;semicolon", "a" * 200]
)
def test_apply_refuses_an_unusable_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    """The reference is the only thing tying the write to a human decision.

    It is stamped into ``preemption_reason`` and the incident's reasons, which is
    where the next reader learns that a remediation was closed on purpose rather
    than lost. An empty or shell-shaped one is rejected before anything runs.
    """

    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_retired_generation_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )

    with pytest.raises(BootstrapError, match="reference is invalid"):
        reconcile.apply_retired_generation_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference=reference,
        )


def test_apply_refuses_when_no_plan_was_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)

    with pytest.raises(BootstrapError, match="has no saved plan"):
        reconcile.apply_retired_generation_reconcile(
            _site(tmp_path),
            tmp_path,
            expected_plan_sha256="d" * 64,
            reference="pre-deploy-6459c07ea279",
        )
