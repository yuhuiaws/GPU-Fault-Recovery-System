from __future__ import annotations

import signal
import sqlite3
from types import SimpleNamespace

import pytest

from gpu_fault.admin import api_budget as budget
from tests.admin._cov95_admission_support import Clock, ProcessRecord, RecoveryProcesses
from tests.admin._cov95_api_support import ShimTransport
from tests.admin.test_api_budget_handoff_model import PARENT, seed_lender

OTHER = "b" * 32


@pytest.fixture
def context(monkeypatch):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "owned admission scope was not created"
        model = RecoveryProcesses()
        clock = Clock()
        with monkeypatch.context() as patch:
            patch.setattr(budget, "os", model.os)
            patch.setattr(budget, "signal", model.signal)
            patch.setattr(budget, "Path", model.path)
            yield SimpleNamespace(root=root, model=model, clock=clock, patch=patch)


def execute(context, sql, parameters=()):
    with sqlite3.connect(context.root / "budget.sqlite3") as database:
        return database.execute(sql, parameters).fetchall()


def insert_middle(context):
    seed_lender(context.root, 4)
    execute(
        context,
        """
        INSERT INTO leases(id,backend,weight,pid,pid_start,command_pid,
            command_start,parent_id,depth,state)
        VALUES(?,'kubectl',2,101,'11',102,'12',?,1,'active')
    """,
        (OTHER, PARENT),
    )
    context.model.pid = 103


def test_admission_cannot_start_without_its_own_process_identity(context):
    context.model.records[102].start = ""
    with pytest.raises(
        budget.ApiBudgetError, match="cannot establish process identity"
    ):
        with budget.api_slot("aws"):
            pytest.fail("identity-free process acquired API capacity")
    assert execute(context, "SELECT id FROM leases") == []


@pytest.mark.parametrize("unknown", ["start", "unreadable"])
def test_unknown_owner_keeps_capacity_charged_during_recovery(context, unknown):
    seed_lender(context.root, 4)
    execute(context, "UPDATE leases SET command_pid=NULL,command_start=NULL")
    if unknown == "start":
        execute(context, "UPDATE leases SET pid_start=''")
    else:
        context.model.errors[100] = PermissionError("modeled unreadable procfs")
    with budget.api_slot("http"):
        assert execute(
            context, "SELECT state,weight FROM leases WHERE id=?", (PARENT,)
        ) == [("active", 4)]
    assert context.model.signals == []


@pytest.mark.parametrize("failure", [FileNotFoundError(), ProcessLookupError()])
def test_disappeared_unbound_shim_releases_only_its_own_lease(context, failure):
    seed_lender(context.root, 4)
    execute(context, "UPDATE leases SET command_pid=NULL,command_start=NULL")
    context.model.errors[100] = failure
    with budget.api_slot("http"):
        assert execute(context, "SELECT id FROM leases WHERE id=?", (PARENT,)) == []
    assert context.model.signals == []


@pytest.mark.parametrize("invalid", ["cycle", "bound", "nearest", "backend-cycle"])
def test_invalid_ancestry_cannot_borrow_or_signal_a_lender(context, invalid):
    if invalid == "nearest":
        insert_middle(context)
    else:
        seed_lender(context.root, 4)
    backend = "aws"
    if invalid == "cycle":
        context.model.records[101].parent = 102
    elif invalid == "bound":
        context.model.records.update(
            {pid: ProcessRecord(pid + 1, str(pid)) for pid in range(1000, 1300)}
        )
        context.model.pid = 1000
    elif invalid == "backend-cycle":
        execute(context, "UPDATE leases SET parent_id=?", (PARENT,))
        backend = "http"
    with pytest.raises(budget.ApiBudgetError, match="ancestry|nesting exceeds"):
        with budget.api_slot(backend, parent_id=PARENT):
            pytest.fail("unbound ancestry acquired a lender reservation")
    assert context.model.signals == []


def test_cross_backend_child_borrows_nearest_matching_capacity_only(context):
    insert_middle(context)
    with budget.api_slot("aws", parent_id=OTHER):
        assert execute(
            context, "SELECT state,weight FROM leases WHERE id=?", (OTHER,)
        ) == [("active", 2)]
        assert execute(
            context, "SELECT state,weight FROM leases WHERE id=?", (PARENT,)
        ) == [("parked", 4)]
    assert context.model.signals == [signal.SIGSTOP, signal.SIGCONT]
    assert context.model.signal_targets == [101, 101]
    assert execute(context, "SELECT state FROM leases WHERE id=?", (PARENT,)) == [
        ("active",)
    ]


def test_handoff_without_pidfd_support_cannot_stop_a_process(context):
    seed_lender(context.root, 4)
    del context.model.os.pidfd_open
    with pytest.raises(budget.ApiBudgetError, match="cannot safely hand off"):
        with budget.api_slot("aws", parent_id=PARENT):
            pytest.fail("unverified process signalling admitted a helper")
    assert context.model.signals == []
    assert execute(context, "SELECT id FROM leases") == [(PARENT,)]


def test_cross_backend_loan_waits_for_lender_command_binding(context):
    insert_middle(context)
    execute(
        context,
        "UPDATE leases SET command_pid=NULL,command_start=NULL WHERE id=?",
        (PARENT,),
    )
    context.patch.setattr(budget, "time", context.clock)
    with pytest.raises(budget.ApiBudgetError, match="admission deadline expired"):
        with budget.api_slot("aws", parent_id=OTHER):
            pytest.fail("unbound lender authorized a nested helper")
    assert context.model.signals == []
    assert execute(context, "SELECT COUNT(*) FROM leases") == [(2,)]


def test_competing_helper_cannot_enter_an_already_lent_branch(context):
    seed_lender(context.root, 4)
    execute(context, "UPDATE leases SET state='parked',lent_to=?", (OTHER,))
    execute(
        context,
        """
        INSERT INTO leases(id,backend,weight,pid,pid_start,parent_id,depth,state)
        VALUES(?,'aws',4,999,'99',?,1,'active')
    """,
        (OTHER, PARENT),
    )
    context.model.records[101].state = "T"
    context.patch.setattr(budget, "time", context.clock)
    with pytest.raises(budget.ApiBudgetError, match="admission deadline expired"):
        with budget.api_slot("aws", parent_id=PARENT):
            pytest.fail("second helper entered an occupied lender branch")
    assert context.model.signals == []
    assert execute(
        context, "SELECT state,lent_to FROM leases WHERE id=?", (PARENT,)
    ) == [("parked", OTHER)]


def test_heavier_helper_keeps_its_loan_while_waiting_for_global_capacity(context):
    seed_lender(context.root, 4)
    execute(
        context,
        "INSERT INTO leases(id,backend,weight,pid,pid_start,depth,state) "
        "VALUES(?,'aws',4,999,'99',0,'active')",
        (OTHER,),
    )
    context.clock.advance = 0.25
    context.clock.on_sleep = lambda: context.model.errors.update(
        {999: FileNotFoundError("modeled competing command exited")}
    )
    context.patch.setattr(budget, "time", context.clock)
    with budget.api_slot("aws", weight=8, parent_id=PARENT):
        assert execute(
            context, "SELECT SUM(weight) FROM leases WHERE state='active'"
        ) == [(8,)]
        assert execute(context, "SELECT state FROM leases WHERE id=?", (PARENT,)) == [
            ("parked",)
        ]
    assert context.model.signals == [signal.SIGSTOP, signal.SIGCONT]


def test_durable_handoff_without_command_identity_blocks_new_admission(context):
    seed_lender(context.root, 4)
    execute(
        context,
        "UPDATE leases SET state='resuming',command_pid=NULL,command_start=NULL",
    )
    with pytest.raises(budget.ApiBudgetError, match="lost its command identity"):
        with budget.api_slot("http"):
            pytest.fail("incomplete recovery intent admitted unrelated work")
    assert context.model.signals == []


def test_pending_resume_keeps_capacity_and_reports_bounded_failure(context):
    seed_lender(context.root, 4)
    context.clock.advance = 0.5
    context.patch.setattr(budget, "time", context.clock)
    send = context.model.send_signal

    def unavailable(descriptor, signum):
        if signum == signal.SIGCONT:
            raise PermissionError("modeled CONT acknowledgement unavailable")
        send(descriptor, signum)

    context.model.signal.pidfd_send_signal = unavailable
    with pytest.raises(budget.ApiBudgetError, match="handoff is still pending"):
        with budget.api_slot("aws", parent_id=PARENT):
            pass
    rows = execute(
        context, "SELECT state,weight,lent_to FROM leases WHERE id=?", (PARENT,)
    )
    assert len(rows) == 1
    assert rows[0][:2] == ("resuming", 4)
    assert rows[0][2] is not None
    assert context.model.signals == [signal.SIGSTOP]
    assert context.model.descriptors == {}


def test_dead_middle_command_never_resumes_lender_with_live_descendant(context):
    seed_lender(context.root, 4)
    with budget.api_slot("aws", parent_id=PARENT) as identifier:
        execute(
            context,
            "UPDATE leases SET command_pid=103,command_start='13',state='parked',lent_to=? WHERE id=?",
            (OTHER, identifier),
        )
        execute(
            context,
            """
            INSERT INTO leases(id,backend,weight,pid,pid_start,parent_id,depth,state)
            VALUES(?,'aws',4,999,'99',?,2,'active')
        """,
            (OTHER, identifier),
        )
        context.model.records[103].state = "Z"
    assert context.model.signals == [signal.SIGSTOP, signal.SIGKILL]
    assert context.model.records[101].state == "Z"
    assert execute(context, "SELECT state,weight FROM leases WHERE id=?", (OTHER,)) == [
        ("active", 4)
    ]


@pytest.mark.parametrize(
    "invalid", [None, "owner", "child", "state", "already-bound", "still-running"]
)
def test_cli_preexec_binding_and_completion_require_exact_process_identity(
    context, tmp_path, invalid
):
    transport = ShimTransport()
    transport.pid = 103
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "aws").touch(mode=0o700)
    context.patch.setenv("PATH", str(tools))
    transport.os.getpid = context.model.os.getpid
    transport.os.getppid = context.model.os.getppid
    transport.install(context.patch, ["aws", "--version"])
    spawn = transport.spawn
    wait = transport.wait

    def bound(arguments, **options):
        if invalid == "owner":
            context.model.records[102].start = "different"
        elif invalid == "child":
            context.model.records[103].start = ""
        elif invalid == "state":
            execute(context, "UPDATE leases SET state='waiting'")
        elif invalid == "already-bound":
            execute(context, "UPDATE leases SET command_pid=103,command_start='13'")
        context.model.pid = 103
        try:
            options["preexec_fn"]()
        finally:
            context.model.pid = 102
            if invalid in {"owner", "child", "state", "already-bound"}:
                context.model.records[103].state = "Z"
        return spawn(arguments, **options)

    def finished(timeout=None):
        value = wait(timeout=timeout)
        if invalid != "still-running":
            context.model.records[103].state = "Z"
        return value

    transport.subprocess.Popen = bound
    transport.wait = finished
    if invalid is None:
        assert budget.main() == transport.result
        assert execute(context, "SELECT id FROM leases") == []
    else:
        if invalid == "still-running":
            context.model.records[103].state = "S"
        with pytest.raises(
            budget.ApiBudgetError,
            match="child identity differs|cannot release a running",
        ):
            budget.main()
    assert transport.signals == []
