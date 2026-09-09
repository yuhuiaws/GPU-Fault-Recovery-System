from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.config import AdminConfigError, admin_config_write_lock
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    membership_operation_lock,
)
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK,
    SITE_OPERATION_LOCK_FD_ENV,
    SiteOperationBusy,
    site_operation_lock,
)


def test_a_second_membership_operation_is_refused_and_told_who_holds_the_lock(
    tmp_path: Path,
) -> None:
    """A join started during a 40-minute deploy is told, not hung.

    The membership lock used to wait silently, so an operator who typed
    ``join-cluster`` while a deploy held the site saw nothing for the rest of the
    deploy. Refusing at once and naming the holder (pid and command) lets them
    decide whether to wait or to stop the other operation.
    """

    site = SimpleNamespace(source=tmp_path / "site.yaml")

    with membership_operation_lock(site):
        with pytest.raises(BootstrapError, match=f"pid {os.getpid()}") as refused:
            with membership_operation_lock(site):
                pass

    assert "administrator mutation is in progress" in str(refused.value)


def test_a_refused_operation_names_the_holder_recorded_in_the_lock_file(
    tmp_path: Path,
) -> None:
    """The holder is read from the lock file, so a refusal from another
    process still names the command that is running."""

    lock_file = tmp_path / SITE_OPERATION_LOCK
    with site_operation_lock(tmp_path, wait=False):
        holder = json.loads(lock_file.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["command"], "the lock file does not record the holder command"
        with pytest.raises(SiteOperationBusy) as refused:
            with site_operation_lock(tmp_path, wait=False):
                pass

    assert holder["command"] in str(refused.value)
    assert str(os.getpid()) in str(refused.value)


def test_a_second_thread_in_the_same_process_is_refused_too(tmp_path: Path) -> None:
    site = SimpleNamespace(source=tmp_path / "site.yaml")
    first_entered = threading.Event()
    release_first = threading.Event()
    outcome: list[BaseException | None] = []

    def first() -> None:
        with membership_operation_lock(site):
            first_entered.set()
            assert release_first.wait(timeout=2), "first lock holder was not released"

    def second() -> None:
        assert first_entered.wait(timeout=2), "first lock holder did not enter"
        try:
            with membership_operation_lock(site):
                outcome.append(None)
        except BootstrapError as exc:
            outcome.append(exc)
        finally:
            release_first.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(outcome) == 1 and isinstance(outcome[0], BootstrapError), (
        "the second membership operation waited instead of being refused"
    )


def test_membership_and_administrator_locks_are_one_lock(tmp_path: Path) -> None:
    site = SimpleNamespace(source=tmp_path / "site.yaml")

    with administrator_operation_lock(tmp_path):
        with pytest.raises(BootstrapError, match="administrator mutation"):
            with membership_operation_lock(site):
                pass


def test_membership_and_config_operations_share_one_site_lock(tmp_path) -> None:
    site = SimpleNamespace(source=tmp_path / "site.yaml")

    with membership_operation_lock(site):
        with pytest.raises(AdminConfigError, match="administrator mutation"):
            with admin_config_write_lock(tmp_path):
                pass


def test_site_operation_lock_accepts_verified_inherited_descriptor(
    tmp_path, monkeypatch
) -> None:
    with site_operation_lock(tmp_path, wait=False) as descriptor:
        monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
        with site_operation_lock(tmp_path, wait=False) as inherited:
            assert inherited == descriptor, (
                "nested site operation did not reuse the verified inherited lock"
            )
