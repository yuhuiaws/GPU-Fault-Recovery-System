from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from gpu_fault.admin import cli
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.config_file import initialize_desired_admin_config
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK,
    SITE_OPERATION_LOCK_FD_ENV,
    SiteOperationBusy,
    inherited_site_operation_lock_fd,
    site_operation_lock,
)
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_site import mock_live_release, site_file


def test_invalid_removal_confirmation_is_rejected_before_site_loading(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as rejected:
        cli.parser().parse_args(
            [
                "remove-cluster",
                "--state-dir",
                str(tmp_path / "missing-site"),
                "--gpu-cluster-arn",
                "arn:aws:eks:us-east-1:123456789012:cluster/gpu-test",
                "--confirm",
                "WRONG",
            ]
        )
    assert rejected.value.code == 2
    assert not (tmp_path / "missing-site").exists(), (
        "invalid confirmation must not create the site directory"
    )


def test_same_inode_descriptor_without_lock_cannot_bypass_busy_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with site_operation_lock(tmp_path, wait=False):
        descriptor = os.open(tmp_path / SITE_OPERATION_LOCK, os.O_RDWR)
        try:
            monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
            assert inherited_site_operation_lock_fd(tmp_path) is None
            with pytest.raises(SiteOperationBusy):
                with site_operation_lock(tmp_path, wait=False):
                    pytest.fail("an unrelated open descriptor bypassed the site lock")
        finally:
            os.close(descriptor)


def test_released_descriptor_is_not_an_inherited_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with site_operation_lock(tmp_path, wait=False) as descriptor:
        duplicate = os.dup(descriptor)
    try:
        monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(duplicate))
        assert inherited_site_operation_lock_fd(tmp_path) is None
        with site_operation_lock(tmp_path, wait=False) as acquired:
            assert acquired != duplicate
    finally:
        os.close(duplicate)


def test_duplicated_locked_description_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with site_operation_lock(tmp_path, wait=False) as descriptor:
        duplicate = os.dup(descriptor)
        try:
            monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(duplicate))
            assert inherited_site_operation_lock_fd(tmp_path) == duplicate
        finally:
            os.close(duplicate)


def test_inherited_descriptor_does_not_bypass_other_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def competing() -> None:
        with site_operation_lock(tmp_path, wait=False):
            pytest.fail("a second thread reused an in-flight inherited lock")

    with site_operation_lock(tmp_path, wait=False) as descriptor:
        monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(competing)
            with pytest.raises(SiteOperationBusy):
                future.result(timeout=5)


def test_lock_symlink_cannot_truncate_another_file(tmp_path: Path) -> None:
    preserved = tmp_path / "preserved.txt"
    preserved.write_text("existing local content", encoding="utf-8")
    (tmp_path / SITE_OPERATION_LOCK).symlink_to(preserved)
    with pytest.raises(OSError):
        with site_operation_lock(tmp_path, wait=False):
            pytest.fail("a lock symlink was accepted")
    assert preserved.read_text(encoding="utf-8") == "existing local content"


def test_lock_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / SITE_OPERATION_LOCK)
    with pytest.raises(SiteOperationBusy, match="not a regular file"):
        with site_operation_lock(tmp_path, wait=False):
            pytest.fail("a FIFO was accepted as an operation lock")


@pytest.mark.parametrize("identity", ["site", "region", "cpu"])
def test_config_rejects_identity_changed_while_acquiring_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    path = site_file(tmp_path)
    original_lock = administrator_operation_lock

    @contextmanager
    def changed_site(root: Path) -> Iterator[None]:
        with original_lock(root):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if identity == "site":
                document["metadata"]["name"] = "different-site"
            elif identity == "region":
                document["spec"]["awsRegion"] = "us-west-2"
            else:
                document["spec"]["cpu"]["eksArn"] = (
                    "arn:aws:eks:us-east-1:123456789012:cluster/other-control"
                )
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            yield

    monkeypatch.setattr(cli, "administrator_operation_lock", changed_site)
    monkeypatch.setattr(
        cli,
        "verify_prebuilt_release",
        lambda *_args, **_kwargs: pytest.fail("identity drift reached release proof"),
    )
    arguments = cli.parser().parse_args(["config", "--state-dir", str(tmp_path)])
    with pytest.raises(
        (BootstrapError, SiteConfigError), match="identity|Region|region"
    ):
        cli.run(arguments)
    assert not (tmp_path / "admin-config/pending.json").exists(), (
        "site identity drift must not write pending configuration"
    )


def test_config_uses_current_site_resources_after_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    mock_live_release(monkeypatch)
    candidate = tmp_path / "capacity.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"aurora": {"minAcu": 16, "maxAcu": 64}},
            }
        ),
        encoding="utf-8",
    )
    candidate.chmod(0o600)
    original_lock = administrator_operation_lock
    observed: list[str] = []

    @contextmanager
    def changed_site(root: Path) -> Iterator[None]:
        with original_lock(root):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            document["spec"]["health"]["auroraClusterId"] = "current-aurora"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            yield

    def request(**kwargs: object) -> dict[str, bool]:
        observed.append(str(kwargs["cluster_id"]))
        return {"modified": True, "scale_up": False}

    monkeypatch.setattr(cli, "administrator_operation_lock", changed_site)
    monkeypatch.setattr(cli, "request_aurora_capacity", request)
    monkeypatch.setattr(cli, "_run_automatic_release", lambda **_kwargs: 0)
    monkeypatch.setattr(cli, "_approver_identity", lambda: "test-operator")
    arguments = cli.parser().parse_args(
        ["config", "--state-dir", str(tmp_path), "--file", str(candidate)]
    )
    assert cli.run(arguments) == 0
    assert observed == ["current-aurora"]
