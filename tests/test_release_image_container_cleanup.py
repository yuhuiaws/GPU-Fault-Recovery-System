from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.verify_release_images import (
    CONTAINER_OWNER_LABEL,
    ImageValidationCleanupError,
    run_image_check,
)

FOREIGN_ID = "f" * 64


class FakeDocker:
    def __init__(
        self, command_result: Callable[[str, list[str]], tuple[int, str]] | None = None
    ) -> None:
        self.command_result = command_result
        self.calls: list[list[str]] = []
        self.containers: dict[str, dict[str, Any]] = {}
        self.creations: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.removal_attempts: list[str] = []
        self.removed: list[str] = []
        self.before_create: Callable[[dict], None] | None = None
        self.after_create: Callable[[dict], None] | None = None
        self.on_start: Callable[[dict], None] | None = None
        self.on_remove: Callable[[str], None] | None = None
        self.listing_failure = False
        self.keep_after_remove = False
        self.cidfile: Path | None = None

    def __call__(
        self, arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        assert arguments[0] == "docker"
        assert options["capture_output"] is True
        assert options["check"] is False
        assert options["start_new_session"] is True
        assert options["timeout"] > 0
        code, output = 0, ""
        if arguments[1] == "create":
            self.cidfile = Path(arguments[arguments.index("--cidfile") + 1])
            reference_index = arguments.index("--") + 1
            reference = arguments[reference_index]
            owner_label = arguments[arguments.index("--label") + 1]
            assert owner_label.startswith(CONTAINER_OWNER_LABEL + "="), (
                "validation containers must carry the expected ownership label"
            )
            item = {
                "id": f"{len(self.creations) + 1:064x}",
                "name": "/" + arguments[arguments.index("--name") + 1],
                "owner": owner_label.split("=", 1)[1],
                "reference": reference,
                "image": "sha256:" + hashlib.sha256(reference.encode()).hexdigest(),
                "command": arguments[reference_index + 1 :],
            }
            self.creations.append(item)
            if self.before_create is not None:
                self.before_create(item)
            self.containers[item["id"]] = item
            self.cidfile.write_text(item["id"])
            if self.after_create is not None:
                self.after_create(item)
            output = item["id"] + "\n"
        elif arguments[1:3] == ["container", "ls"]:
            if self.listing_failure:
                return subprocess.CompletedProcess(arguments, 1, "", "unavailable")
            selector = arguments[arguments.index("--filter") + 1]
            values = list(self.containers.values())
            if selector.startswith("id="):
                values = [item for item in values if item["id"] == selector[3:]]
            else:
                assert selector.startswith("name=^/") and selector.endswith("$")
                values = [item for item in values if item["name"] == selector[6:-1]]
            output = "".join(item["id"] + "\n" for item in values)
        elif arguments[1:3] == ["container", "inspect"]:
            item = self.containers.get(arguments[-1])
            code = 0 if item is not None else 1
            output = json.dumps(item) if item is not None else ""
        elif arguments[1:3] == ["start", "--attach"]:
            item = self.containers[arguments[-1]]
            self.started.append(item["id"])
            if self.on_start is not None:
                self.on_start(item)
            if self.command_result is not None:
                code, output = self.command_result(item["reference"], item["command"])
            else:
                output = "checked\n"
        elif arguments[1:4] == ["container", "rm", "--force"]:
            identifier = arguments[-1]
            self.removal_attempts.append(identifier)
            if self.on_remove is not None:
                self.on_remove(identifier)
            if not self.keep_after_remove and identifier in self.containers:
                self.containers.pop(identifier)
                self.removed.append(identifier)
        else:
            raise AssertionError(f"unexpected Docker operation: {arguments[1:3]}")
        return subprocess.CompletedProcess(arguments, code, output, "")


@pytest.fixture(autouse=True)
def private_temporary_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def test_success_records_identity_before_start_and_confirms_cleanup() -> None:
    docker = FakeDocker()

    def inspect_record(item):
        assert docker.cidfile is not None
        record = json.loads(docker.cidfile.with_name("ownership.json").read_text())
        assert record["container_id"] == item["id"]
        assert record["image_id"] == item["image"]
        assert record["owner"] == item["owner"]
        assert docker.cidfile.parent.stat().st_mode & 0o777 == 0o700

    docker.on_start = inspect_record
    result = run_image_check(
        "candidate-image", ["/bin/sh", "-c", "true"], runner=docker
    )
    assert result.returncode == 0
    assert result.stdout == "checked\n"
    assert len(docker.removed) == 1 and docker.containers == {}
    assert docker.cidfile is not None and not docker.cidfile.parent.exists()
    creation = docker.calls[0]
    assert {
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    } <= set(creation), "validation sandbox flags were lost"
    assert "--rm" not in creation, (
        "auto-removal must not discard identity before inspection"
    )
    assert all(
        "--volumes" in call for call in docker.calls if call[1:3] == ["container", "rm"]
    ), "explicit cleanup must retain auto-removal's anonymous-volume cleanup"
    assert docker.calls[-2][1:3] == docker.calls[-1][1:3] == ["container", "ls"]


def test_nonzero_command_is_cleaned_before_returning_its_status() -> None:
    docker = FakeDocker(lambda *_: (7, "invalid"))
    result = run_image_check("candidate-image", ["false"], runner=docker)
    assert (result.returncode, result.stdout) == (7, "invalid")
    assert docker.containers == {}
    assert len(docker.removed) == 1


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "transport"])
def test_execution_failure_cleans_the_owned_container_and_preserves_the_error(
    failure: str,
) -> None:
    docker = FakeDocker()
    error = {
        "timeout": subprocess.TimeoutExpired(["docker", "start"], 1),
        "interrupt": KeyboardInterrupt(),
        "transport": OSError("client connection lost"),
    }[failure]

    def fail(_item):
        raise error

    docker.on_start = fail
    with pytest.raises(type(error)):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.containers == {}, (
        "execution failure left the validation container running"
    )
    assert len(docker.removed) == 1
    assert docker.cidfile is not None and not docker.cidfile.parent.exists()


@pytest.mark.parametrize("has_cid", [False, True])
def test_create_ack_loss_requires_a_cid_before_cleanup(has_cid: bool) -> None:
    docker = FakeDocker()

    def lose_ack(_item):
        assert docker.cidfile is not None
        if not has_cid:
            docker.cidfile.unlink()
        raise subprocess.TimeoutExpired(["docker", "create"], 1)

    docker.after_create = lose_ack
    expected = subprocess.TimeoutExpired if has_cid else ImageValidationCleanupError
    with pytest.raises(expected):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.started == [], (
        "an unacknowledged creation must not execute the command"
    )
    if has_cid:
        assert docker.containers == {}
        assert len(docker.removed) == 1
    else:
        assert docker.removal_attempts == [], (
            "a name and label must not replace a lost CID"
        )
        assert len(docker.containers) == 1
        assert docker.cidfile is not None
        assert docker.cidfile.with_name("ownership.json").is_file(), (
            "a missing creation CID must retain its unresolved ownership record"
        )


def test_unknown_create_without_a_cid_does_not_claim_cleanup() -> None:
    docker = FakeDocker()

    def fail_before_ack(_item):
        raise subprocess.TimeoutExpired(["docker", "create"], 1)

    docker.before_create = fail_before_ack
    with pytest.raises(ImageValidationCleanupError, match="cleanup unconfirmed"):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.started == docker.removal_attempts == []
    assert docker.cidfile is not None
    assert docker.cidfile.with_name("ownership.json").is_file(), (
        "an uncertain daemon-side create needs a retained ownership record"
    )


def test_lost_cid_does_not_authorize_a_replacement_with_copied_labels() -> None:
    docker = FakeDocker()

    def replace_before_ack(item):
        assert docker.cidfile is not None
        docker.cidfile.unlink()
        docker.containers.pop(item["id"])
        docker.containers[FOREIGN_ID] = {**item, "id": FOREIGN_ID}
        raise subprocess.TimeoutExpired(["docker", "create"], 1)

    docker.after_create = replace_before_ack
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.started == docker.removal_attempts == []
    assert list(docker.containers) == [FOREIGN_ID], (
        "a copied name/label pair cannot prove the original container incarnation"
    )


@pytest.mark.parametrize("field", ["owner", "name", "image", "reference"])
def test_changed_container_identity_is_never_removed(field: str) -> None:
    docker = FakeDocker()

    def change_identity(item):
        item[field] = "sha256:" + "b" * 64 if field == "image" else "foreign"

    docker.on_start = change_identity
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.removal_attempts == [], (
        "identity drift authorized a destructive Docker command"
    )
    assert len(docker.containers) == 1


@pytest.mark.parametrize("keep_original", [False, True])
def test_replaced_name_never_authorizes_removing_the_replacement(
    keep_original: bool,
) -> None:
    docker = FakeDocker()

    def replace(item):
        replacement = {**item, "id": FOREIGN_ID, "owner": "another-run"}
        if keep_original:
            item["name"] += "-renamed"
        else:
            docker.containers.pop(item["id"])
        docker.containers[FOREIGN_ID] = replacement

    docker.on_start = replace
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.removal_attempts == []
    assert FOREIGN_ID in docker.containers


def test_name_replacement_after_inspection_cannot_change_the_removal_target() -> None:
    docker = FakeDocker()

    def replace_before_remove(identifier):
        item = docker.containers.pop(identifier)
        docker.containers[FOREIGN_ID] = {
            **item,
            "id": FOREIGN_ID,
            "owner": "another-run",
        }

    docker.on_remove = replace_before_remove
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.removal_attempts == [docker.creations[0]["id"]]
    assert docker.removed == []
    assert FOREIGN_ID in docker.containers, (
        "cleanup targeted the name instead of its immutable ID"
    )


@pytest.mark.parametrize("bad_cid", ["bad", "a" * 66, FOREIGN_ID])
def test_changed_or_malformed_cid_file_fails_closed(bad_cid: str) -> None:
    docker = FakeDocker()

    def change_cid(_item):
        assert docker.cidfile is not None
        docker.cidfile.write_text(bad_cid)

    docker.on_start = change_cid
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.removal_attempts == []


def test_cleanup_ack_loss_is_accepted_only_after_confirming_absence() -> None:
    docker = FakeDocker()

    def remove_then_timeout(identifier):
        docker.containers.pop(identifier)
        raise subprocess.TimeoutExpired(["docker", "rm"], 1)

    docker.on_remove = remove_then_timeout
    assert run_image_check("candidate-image", ["check"], runner=docker).returncode == 0
    assert docker.containers == {}
    assert docker.cidfile is not None and not docker.cidfile.parent.exists()


@pytest.mark.parametrize("failure", ["unavailable", "still-present"])
def test_cleanup_uncertainty_overrides_a_successful_validation(failure: str) -> None:
    docker = FakeDocker()
    if failure == "unavailable":
        docker.on_start = lambda _item: setattr(docker, "listing_failure", True)
    else:
        docker.keep_after_remove = True
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert (
        docker.cidfile is not None
        and docker.cidfile.with_name("ownership.json").exists()
    )
    assert len(docker.containers) == 1


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("during_cleanup", [False, True])
def test_interrupts_restore_handlers_and_do_not_abandon_cleanup(
    number: int, during_cleanup: bool
) -> None:
    docker = FakeDocker()
    previous = {
        value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt(_item):
        handler = signal.getsignal(number)
        assert callable(handler), "the validation command must handle cancellation"
        handler(number, None)

    if during_cleanup:
        docker.on_remove = interrupt
    else:
        docker.on_start = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.containers == {}
    assert {value: signal.getsignal(value) for value in previous} == previous


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_cannot_create_a_container(timeout: float) -> None:
    docker = FakeDocker()
    with pytest.raises(ValueError, match="positive and finite"):
        run_image_check("candidate-image", ["check"], runner=docker, timeout=timeout)
    assert docker.calls == []


def test_nonzero_create_still_removes_its_identified_container() -> None:
    docker = FakeDocker()

    def run(arguments, **options):
        result = docker(arguments, **options)
        if arguments[1] == "create":
            return subprocess.CompletedProcess(arguments, 125, "", "create failed")
        return result

    with pytest.raises(ValueError, match="container creation failed"):
        run_image_check("candidate-image", ["check"], runner=run)
    assert docker.started == []
    assert docker.containers == {}
    assert len(docker.removed) == 1


def test_create_name_collision_never_removes_the_foreign_container() -> None:
    docker = FakeDocker()

    def collision(item):
        docker.containers[FOREIGN_ID] = {
            **item,
            "id": FOREIGN_ID,
            "owner": "someone-else",
        }
        raise OSError("name already reserved")

    docker.before_create = collision
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.started == docker.removal_attempts == []
    assert list(docker.containers) == [FOREIGN_ID]


@pytest.mark.parametrize("cid_defect", ["missing", "empty"])
def test_successful_create_requires_a_matching_cid_file_before_start(
    cid_defect: str,
) -> None:
    docker = FakeDocker()

    def lose_cid(_item):
        assert docker.cidfile is not None
        if cid_defect == "missing":
            docker.cidfile.unlink()
        else:
            docker.cidfile.write_text("")

    docker.after_create = lose_cid
    with pytest.raises(ValueError, match="matching CID file"):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.started == []
    assert docker.containers == {}, "the acknowledged CID still binds safe cleanup"


def test_cid_symlink_is_rejected_without_reading_its_target(tmp_path) -> None:
    docker = FakeDocker()
    target = tmp_path / "foreign"
    target.write_text(FOREIGN_ID)

    def replace_cid(_item):
        assert docker.cidfile is not None
        docker.cidfile.unlink()
        docker.cidfile.symlink_to(target)

    docker.on_start = replace_cid
    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=docker)
    assert docker.removal_attempts == []
    assert target.read_text() == FOREIGN_ID


@pytest.mark.parametrize("output", ["not-json", "[]", "{}"])
def test_malformed_identity_cannot_authorize_start_or_removal(output: str) -> None:
    docker = FakeDocker()

    def run(arguments, **options):
        result = docker(arguments, **options)
        if arguments[1:3] == ["container", "inspect"]:
            return subprocess.CompletedProcess(arguments, 0, output, "")
        return result

    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=run)
    assert docker.started == docker.removal_attempts == []


@pytest.mark.parametrize("output", ["short-id", "a" * 64 + "\n" + FOREIGN_ID])
def test_ambiguous_container_listing_cannot_authorize_removal(output: str) -> None:
    docker = FakeDocker()

    def run(arguments, **options):
        result = docker(arguments, **options)
        if arguments[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(arguments, 0, output, "")
        return result

    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=run)
    assert docker.removal_attempts == []


def test_unknown_inspection_is_not_treated_as_absence() -> None:
    docker = FakeDocker()

    def run(arguments, **options):
        result = docker(arguments, **options)
        if arguments[1:3] == ["container", "inspect"] and docker.started:
            return subprocess.CompletedProcess(arguments, 1, "", "daemon unavailable")
        return result

    with pytest.raises(ImageValidationCleanupError):
        run_image_check("candidate-image", ["check"], runner=run)
    assert docker.removal_attempts == []
    assert len(docker.containers) == 1


def test_external_removal_of_the_original_id_is_a_confirmed_absence() -> None:
    docker = FakeDocker()
    docker.on_start = lambda item: docker.containers.pop(item["id"])
    assert run_image_check("candidate-image", ["check"], runner=docker).returncode == 0
    assert docker.removal_attempts == []
    assert docker.cidfile is not None and not docker.cidfile.parent.exists()
