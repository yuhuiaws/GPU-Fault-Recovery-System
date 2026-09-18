"""Create-only public submission identity, using only an in-memory transport."""

from __future__ import annotations

import copy
import importlib
import json
import stat
import subprocess
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault import training_submit_cli
from scripts.e2e.regional.managed_workload_fixture import OWNER_LABEL, RESOURCE_APIS
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from scripts.e2e.regional.workload_submission_identity import (
    CreateOnlySubmissionIdentity,
)

yaml = importlib.import_module("yaml")

NAMESPACE = "training"
CONTEXT = "gpu-context"
RESOURCE_NAME = "train"
OWNER = "a" * 32
SENSITIVE = "synthetic-kubeconfig-credential-not-for-output"
RBAC_API = "rbac.authorization.k8s.io"


def gpu_config() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": CONTEXT,
        "preferences": {},
        "contexts": [
            {
                "name": CONTEXT,
                "context": {
                    "cluster": "gpu",
                    "user": "deployment",
                    "namespace": NAMESPACE,
                },
            }
        ],
        "clusters": [
            {
                "name": "gpu",
                "cluster": {
                    "server": "https://gpu.invalid",
                    "certificate-authority-data": "c3ludGhldGljLWNh",
                    "tls-server-name": "gpu.invalid",
                },
            }
        ],
        "users": [{"name": "deployment", "user": {"token": SENSITIVE}}],
    }


class Kubernetes:
    """Model RBAC authorization at RegionalLiveFixture.run, never an executable."""

    def __init__(self, regional: RegionalLiveFixture) -> None:
        self.regional = regional
        self.config = gpu_config()
        self.config_output: str | None = None
        self.config_status = 0
        self.config_stderr = ""
        self.fail_transport = ""
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.deletes: list[tuple[str, str, dict[str, Any]]] = []
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.permission_answers: dict[str, tuple[int, str, str]] = {}
        self.group_rules: list[dict[str, Any]] = []
        self.create_failure = ""
        self.create_collision = ""
        self.ack_change: Callable[[dict[str, Any]], None] | None = None
        self.before_delete: Callable[[dict[str, Any]], None] | None = None
        self.after_delete: Callable[[dict[str, Any]], None] | None = None
        self.delete_failure = ""
        self.read_failure = ""
        self.workload: dict[str, Any] | None = None
        self.workload_requests: list[tuple[str, int]] = []
        self.apply_commands: list[list[str]] = []
        self.create_race = False

    def auth(self, kubeconfig: Path) -> dict[str, Any]:
        value = json.loads(kubeconfig.read_text(encoding="utf-8"))
        return cast(dict[str, Any], value["users"][0]["user"])

    def rules(self, kubeconfig: Path) -> list[dict[str, Any]]:
        auth = self.auth(kubeconfig)
        rules = copy.deepcopy(
            self.group_rules
            if "system:authenticated" in auth.get("as-groups", [])
            else []
        )
        for (kind, _name), binding in self.objects.items():
            if kind != "RoleBinding" or not any(
                subject["kind"] == "User" and subject["name"] == auth.get("as")
                for subject in binding["subjects"]
            ):
                continue
            role = self.objects.get(("Role", binding["roleRef"]["name"]))
            if role is not None:
                rules.extend(copy.deepcopy(role["rules"]))
        return rules

    def allowed(
        self, kubeconfig: Path, verb: str, resource: str, group: str, name: str = ""
    ) -> bool:
        return any(
            verb in rule["verbs"] or "*" in rule["verbs"]
            for rule in self.rules(kubeconfig)
            if set(rule["resources"]) & {resource, "*"}
            and set(rule["apiGroups"]) & {group, "*"}
            and (not rule.get("resourceNames") or name in rule["resourceNames"])
        )

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        assert command[0] == "kubectl"
        assert command[command.index("--context") + 1] == CONTEXT
        assert command[command.index("-n") + 1] == NAMESPACE
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 60
        kubeconfig = Path(command[command.index("--kubeconfig") + 1])
        arguments = command[command.index("-n") + 2 :]
        verb = arguments[0]
        original = kubeconfig == self.regional.settings.gpu_kubeconfig
        if self.fail_transport == verb:
            raise RuntimeError(SENSITIVE)
        stdout, stderr, code = "", "", 0
        if verb == "config":
            assert original, "minification must read the original GPU kubeconfig"
            assert arguments == [
                "config",
                "view",
                "--raw",
                "--minify",
                "--flatten",
                "-o",
                "json",
            ]
            stdout = (
                json.dumps(self.config)
                if self.config_output is None
                else self.config_output
            )
            code, stderr = self.config_status, self.config_stderr
        elif verb == "auth":
            assert not original, (
                "permission checks must use the temporary impersonated identity"
            )
            action, target = arguments[2:4]
            qualified, _separator, name = target.partition("/")
            resource, group = qualified.split(".", 1)
            answer = self.allowed(kubeconfig, action, resource, group, name)
            code, stdout, stderr = self.permission_answers.get(
                action, (0, "yes\n", "") if answer else (1, "no\n", "")
            )
        elif verb == "create":
            assert original, (
                "temporary RBAC must be created with the original deployment identity"
            )
            assert arguments == ["create", "-f", "-", "-o", "json"]
            document = json.loads(kwargs["input_text"])
            kind, metadata = document["kind"], document["metadata"]
            key = (kind, metadata["name"])
            if kind == self.create_collision:
                foreign = copy.deepcopy(document)
                foreign["metadata"].update(uid="foreign", resourceVersion="1")
                self.objects[key] = foreign
            if key in self.objects:
                code, stderr = 1, "AlreadyExists"
            else:
                self.created.append(copy.deepcopy(document))
                metadata.update(uid=f"{kind}-uid", resourceVersion="1")
                self.objects[key] = copy.deepcopy(document)
                if self.ack_change is not None:
                    self.ack_change(document)
                stdout = json.dumps(document)
                if kind == self.create_failure:
                    code, stdout, stderr = 1, "", SENSITIVE
        elif verb == "get":
            assert original, (
                "RBAC cleanup reads must use the original deployment identity"
            )
            assert arguments[1].endswith("." + RBAC_API), (
                "cleanup inventory must stay within the RBAC API group"
            )
            kind = {"roles": "Role", "rolebindings": "RoleBinding"}[
                arguments[1].split(".")[0]
            ]
            assert arguments[3:] == ["--ignore-not-found", "-o", "json"]
            if kind == self.read_failure:
                code, stderr = 1, SENSITIVE
            else:
                value = self.objects.get((kind, arguments[2]))
                stdout = json.dumps(value) if value is not None else ""
        elif verb == "delete":
            assert original, (
                "RBAC teardown must not depend on the create-only submission identity"
            )
            assert arguments[1] == "--raw", "cleanup must use API preconditions"
            plural, name = arguments[2].split("/")[-2:]
            kind = {"roles": "Role", "rolebindings": "RoleBinding"}[plural]
            assert f"/namespaces/{NAMESPACE}/" in arguments[2]
            assert arguments[3:] == ["-f", "-"]
            options = json.loads(kwargs["input_text"])
            self.deletes.append((kind, name, options))
            current = self.objects[(kind, name)]
            if self.before_delete is not None:
                self.before_delete(current)
            if options["preconditions"] != {
                key: current["metadata"][key] for key in ("uid", "resourceVersion")
            }:
                code, stderr = 1, "Conflict"
            else:
                del self.objects[(kind, name)]
                if self.after_delete is not None:
                    self.after_delete(current)
                if kind == self.delete_failure:
                    code, stderr = 1, SENSITIVE
                else:
                    stdout = '{"apiVersion":"v1","kind":"Status","status":"Success"}'
        else:
            raise AssertionError(f"unexpected transport verb: {verb}")
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    def apply(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Simulate ordinary apply's GET then POST/PATCH, without rewriting argv."""
        self.apply_commands.append(command)
        assert command[0] == "kubectl"
        assert "apply" in command
        assert command[command.index("--context") + 1] == CONTEXT
        kubeconfig = Path(command[command.index("--kubeconfig") + 1])
        document = yaml.safe_load(kwargs["input"])
        group = document["apiVersion"].split("/")[0]
        resource = RESOURCE_APIS[document["kind"].lower()][1]
        name = document["metadata"]["name"]
        exists = self.workload is not None
        assert self.allowed(kubeconfig, "get", resource, group, name), (
            "public apply must be allowed to read the exact workload name"
        )
        self.workload_requests.append(("GET", 200 if exists else 404))
        if not exists and self.create_race:
            self.workload = {"metadata": {"name": name, "uid": "foreign"}}
        if exists:
            assert not self.allowed(kubeconfig, "patch", resource, group, name), (
                "a same-name workload must not be patchable by the submission identity"
            )
            self.workload_requests.append(("PATCH", 403))
            return subprocess.CompletedProcess(command, 1, "", "Forbidden")
        assert self.allowed(kubeconfig, "create", resource, group), (
            "public apply must retain create permission on the selected resource"
        )
        if self.workload is not None:
            self.workload_requests.append(("POST", 409))
            return subprocess.CompletedProcess(command, 1, "", "AlreadyExists")
        self.workload_requests.append(("POST", 201))
        if "--dry-run=server" not in command:
            self.workload = document
        return subprocess.CompletedProcess(command, 0, "created", "")


def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str = "job"
) -> tuple[CreateOnlySubmissionIdentity, Kubernetes]:
    cpu, gpu = tmp_path / "cpu.kubeconfig", tmp_path / "gpu.kubeconfig"
    cpu.write_text("synthetic CPU config", encoding="utf-8")
    original = gpu_config()
    original["contexts"].append(
        {"name": "cpu-context", "context": {"cluster": "cpu", "user": "cpu"}}
    )
    original["clusters"].append(
        {"name": "cpu", "cluster": {"server": "https://cpu.invalid"}}
    )
    original["users"].append({"name": "cpu", "user": {"token": "synthetic-cpu"}})
    gpu.write_text(json.dumps(original), encoding="utf-8")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context=CONTEXT,
            namespace=NAMESPACE,
            cluster_id="synthetic-cluster",
            region="us-west-2",
        )
    )
    api = Kubernetes(regional)
    monkeypatch.setattr(regional, "run", api.run)
    identity = CreateOnlySubmissionIdentity(
        regional,
        directory=tmp_path,
        owner=OWNER,
        resource=resource,
        resource_name=RESOURCE_NAME,
    )
    return identity, api


@pytest.mark.parametrize("resource", ["job", "pytorchjob", "jobset"])
def test_exact_scoped_role_and_impersonated_permission_checks(
    resource: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch, resource)
    with identity as entered:
        assert entered is identity
        role, binding = api.created
        group = RESOURCE_APIS[resource][0].split("/")[1]
        plural = RESOURCE_APIS[resource][1]
        assert role["rules"] == [
            {"apiGroups": [group], "resources": [plural], "verbs": ["get", "create"]}
        ]
        assert role["metadata"]["namespace"] == NAMESPACE
        assert role["metadata"]["labels"] == {OWNER_LABEL: OWNER}
        assert binding["roleRef"] == {
            "apiGroup": RBAC_API,
            "kind": "Role",
            "name": role["metadata"]["name"],
        }
        assert binding["subjects"] == [
            {"kind": "User", "apiGroup": RBAC_API, "name": identity.subject}
        ]
        assert identity.subject.startswith(f"gpu-fault-acceptance-submit-{OWNER}-"), (
            "the impersonated user must remain bound to the fixture owner"
        )
        checks = [
            command[command.index("can-i") + 1 :]
            for command, _kwargs in api.calls
            if "can-i" in command
        ]
        assert checks == [
            [
                verb,
                f"{plural}.{group}" + (f"/{RESOURCE_NAME}" if verb != "create" else ""),
            ]
            for verb in ("get", "create", "patch", "update", "delete")
        ]
        assert all(
            "--raw" not in command
            for command, _kwargs in api.calls
            if "create" in command
        ), "authorization must not require enumerating an EKS authorizer's rules"
    assert api.objects == {}
    assert [kind for kind, _name, _options in api.deletes] == ["RoleBinding", "Role"]
    assert all(
        options["preconditions"] == {"uid": f"{kind}-uid", "resourceVersion": "1"}
        for kind, _name, options in api.deletes
    ), "every RBAC deletion must fence the acknowledged UID and current resourceVersion"
    assert not identity.kubeconfig.parent.exists(), (
        "normal teardown must remove the private credential directory"
    )


@pytest.mark.parametrize(
    "auth",
    [
        {"token": SENSITIVE},
        {
            "exec": {
                "apiVersion": "client.authentication.k8s.io/v1",
                "command": "aws",
                "args": ["eks", "get-token", "--cluster-name", "synthetic"],
                "env": [{"name": "SYNTHETIC_INPUT", "value": SENSITIVE}],
                "interactiveMode": "Never",
                "provideClusterInfo": True,
            }
        },
        {"client-certificate-data": "c3ludGhldGlj", "client-key-data": SENSITIVE},
        {"auth-provider": {"name": "synthetic", "config": {"credential": SENSITIVE}}},
    ],
    ids=["token", "exec", "client-certificate", "auth-provider"],
)
def test_private_minified_config_preserves_auth_tls_and_original(
    auth: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    original = api.regional.settings.gpu_kubeconfig.read_bytes()
    api.config["users"][0]["user"] = copy.deepcopy(auth)
    with identity:
        private = json.loads(identity.kubeconfig.read_text(encoding="utf-8"))
        assert stat.S_IMODE(identity.kubeconfig.stat().st_mode) == 0o600
        assert stat.S_IMODE(identity.kubeconfig.parent.stat().st_mode) == 0o700
        assert private["clusters"] == api.config["clusters"]
        assert private["contexts"] == api.config["contexts"]
        assert len(private["users"]) == 1
        assert private["users"][0]["user"] == {
            **auth,
            "as": identity.subject,
            "as-groups": ["system:authenticated"],
        }
        assert all(
            str(api.regional.settings.cpu_kubeconfig) not in command
            for command, _kwargs in api.calls
        ), "GPU-only submission must never use the CPU kubeconfig"
    assert api.regional.settings.gpu_kubeconfig.read_bytes() == original


def test_same_owner_has_unique_subjects_and_context_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, api = harness(tmp_path, monkeypatch)
    second = CreateOnlySubmissionIdentity(
        api.regional,
        directory=tmp_path,
        owner=OWNER,
        resource="job",
        resource_name=RESOURCE_NAME,
    )
    assert first.subject != second.subject
    assert first.kubeconfig != second.kubeconfig
    with first, second:
        assert len(api.objects) == 4
    assert api.objects == {}
    before = len(api.calls)
    with pytest.raises(RegionalFixtureError, match="cannot be reused"), first:
        pytest.fail("identity reuse must fail")
    assert len(api.calls) == before


@pytest.mark.parametrize(
    ("verb", "answer"),
    [
        ("get", (1, "no\n", "")),
        ("create", (1, "no\n", "")),
        ("patch", (0, "yes\n", "")),
        ("update", (0, "yes\n", "")),
        ("delete", (0, "yes\n", "")),
        ("get", (1, "yes\n", "")),
        ("patch", (0, "no\n", "")),
        ("patch", (1, "no\n", SENSITIVE)),
        ("patch", (2, "no\n", "")),
        ("patch", (1, "", "")),
        ("patch", (1, "no\nunknown\n", "")),
        ("get", (0, "YES\n", "")),
        ("get", (0, "yes\n", "warning: uncertain")),
    ],
    ids=[
        "get-denied",
        "create-denied",
        "patch-allowed",
        "update-allowed",
        "delete-allowed",
        "yes-wrong-exit",
        "no-wrong-exit",
        "transport-no",
        "bad-exit",
        "empty",
        "extra-output",
        "unknown-answer",
        "warning",
    ],
)
def test_only_authoritative_expected_permissions_allow_entry(
    verb: str,
    answer: tuple[int, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    api.permission_answers[verb] = answer
    with pytest.raises(RegionalFixtureError, match="permission") as failure, identity:
        pytest.fail("unproven privileges must not expose a usable identity")
    assert api.objects == {}
    assert not identity.kubeconfig.parent.exists(), (
        "failed authorization must still remove the private kubeconfig directory"
    )
    assert SENSITIVE not in "".join(traceback.format_exception(failure.value))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("verb", ["patch", "update", "delete"])
@pytest.mark.parametrize("scope", ["all-names", "target-name", "other-name"])
def test_group_mutation_grants_are_checked_for_the_exact_workload(
    verb: str, scope: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    api.group_rules.append(
        {
            "apiGroups": ["batch"],
            "resources": ["jobs"],
            "verbs": [verb],
            **(
                {
                    "resourceNames": [
                        RESOURCE_NAME if scope == "target-name" else "other-training"
                    ]
                }
                if scope != "all-names"
                else {}
            ),
        }
    )
    if scope == "other-name":
        with identity:
            assert not api.allowed(
                identity.kubeconfig, verb, "jobs", "batch", RESOURCE_NAME
            ), "a grant for another workload name must not authorize this target"
    else:
        with pytest.raises(RegionalFixtureError, match=f"permission {verb}"), identity:
            pytest.fail("a group grant must not allow mutation of the bound workload")
    assert api.objects == {}
    assert not identity.kubeconfig.exists(), (
        "leaving the identity context must remove its private kubeconfig"
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "-train",
        "train-",
        "train/job",
        "Train",
        "train\n",
        "train..job",
        "*.train",
        "x" * 254,
    ],
    ids=[
        "empty",
        "leading-dash",
        "trailing-dash",
        "path",
        "uppercase",
        "newline",
        "empty-label",
        "wildcard",
        "too-long",
    ],
)
def test_resource_name_is_validated_before_any_transport_or_file_creation(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity, api = harness(tmp_path, monkeypatch)
    before = set(tmp_path.iterdir())
    with pytest.raises(RegionalFixtureError, match="name"):
        CreateOnlySubmissionIdentity(
            api.regional,
            directory=tmp_path,
            owner=OWNER,
            resource="job",
            resource_name=name,
        )
    assert api.calls == []
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("field", ["as", "as-groups", "as-uid", "as-user-extra"])
@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_preexisting_impersonation_is_refused_not_replaced(
    field: str, empty: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    existing = {
        "as": "" if empty else SENSITIVE,
        "as-groups": [] if empty else ["system:masters"],
        "as-uid": "" if empty else SENSITIVE,
        "as-user-extra": {} if empty else {"extra": [SENSITIVE]},
    }
    api.config["users"][0]["user"][field] = existing[field]
    with pytest.raises(RegionalFixtureError, match="kubeconfig") as failure, identity:
        pytest.fail("inherited impersonation must not be silently replaced")
    assert api.created == []
    assert not identity.kubeconfig.parent.exists(), (
        "rejected inherited impersonation must not leave a private credential directory"
    )
    assert SENSITIVE not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize(
    "defect",
    [
        "transport",
        "stderr",
        "exit",
        "malformed",
        "multiple",
        "wrong-context",
        "wrong-user",
        "insecure",
        "http",
        "missing-ca",
    ],
)
def test_config_failure_never_exposes_credentials_or_creates_rbac(
    defect: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    if defect == "transport":
        api.fail_transport = "config"
    elif defect == "stderr":
        api.config_stderr = SENSITIVE
    elif defect == "exit":
        api.config_status = 1
    elif defect == "malformed":
        api.config_output = '{"users": "' + SENSITIVE
    elif defect == "multiple":
        api.config["users"].append({"name": "cpu", "user": {"token": SENSITIVE}})
    elif defect == "wrong-context":
        api.config["current-context"] = "cpu-context"
    elif defect == "wrong-user":
        api.config["contexts"][0]["context"]["user"] = "cpu"
    elif defect == "insecure":
        api.config["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    elif defect == "http":
        api.config["clusters"][0]["cluster"]["server"] = "http://gpu.invalid"
    else:
        del api.config["clusters"][0]["cluster"]["certificate-authority-data"]
    with pytest.raises(RegionalFixtureError, match="kubeconfig") as failure, identity:
        pytest.fail("invalid kubeconfig must fail before RBAC creation")
    assert api.created == []
    assert not identity.kubeconfig.parent.exists(), (
        "kubeconfig preparation failures must clean their temporary directory"
    )
    assert SENSITIVE not in "".join(traceback.format_exception(failure.value))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("ancestor", [False, True], ids=["directory", "ancestor"])
def test_output_directory_never_traverses_symlinks(
    ancestor: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity, api = harness(tmp_path, monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    (real / "case").mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    directory = link / "case" if ancestor else link
    identity = CreateOnlySubmissionIdentity(
        api.regional,
        directory=directory,
        owner=OWNER,
        resource="job",
        resource_name=RESOURCE_NAME,
    )
    with pytest.raises(RegionalFixtureError, match="kubeconfig"), identity:
        pytest.fail("symlinked output path must not be followed")
    assert api.calls == []
    assert list((real / "case").iterdir()) == []
    assert sorted(item.name for item in real.iterdir()) == ["case"]


def test_existing_private_directory_is_never_reused_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    identity.kubeconfig.parent.mkdir()
    identity.kubeconfig.write_text("foreign", encoding="utf-8")
    with pytest.raises(RegionalFixtureError, match="kubeconfig"), identity:
        pytest.fail("foreign private directory must not be adopted")
    assert identity.kubeconfig.read_text(encoding="utf-8") == "foreign"
    assert api.calls == []


@pytest.mark.parametrize("kind", ["Role", "RoleBinding"])
@pytest.mark.parametrize("collision", [False, True], ids=["lost-ack", "already-exists"])
def test_missing_creation_ack_never_authorizes_adoption_or_deletion(
    kind: str, collision: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    if collision:
        api.create_collision = kind
    else:
        api.create_failure = kind
    with (
        pytest.raises(RegionalFixtureError, match="cleanup uncertain.*ACK missing"),
        identity,
    ):
        pytest.fail("an unacknowledged creation must fail closed")
    assert len(api.objects) == 1
    assert next(iter(api.objects))[0] == kind
    assert all(deleted != kind for deleted, _name, _options in api.deletes), (
        "a missing creation ACK must never authorize deletion of that resource"
    )
    assert not identity.kubeconfig.parent.exists(), (
        "uncertain RBAC ownership must not prevent removal of local credentials"
    )
    assert not any(
        "get" in command and command[command.index("get") + 1].startswith(kind.lower())
        for command, _kwargs in api.calls
    ), "a later GET cannot substitute for the missing creation ACK"


@pytest.mark.parametrize(
    "field", ["namespace", "name", "owner", "uid", "resourceVersion"]
)
def test_create_ack_must_prove_its_identity(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)

    def change(document: dict[str, Any]) -> None:
        if field == "owner":
            document["metadata"]["labels"][OWNER_LABEL] = "foreign"
        else:
            document["metadata"][field] = (
                "" if field in {"uid", "resourceVersion"} else "foreign"
            )

    api.ack_change = change
    with (
        pytest.raises(RegionalFixtureError, match="cleanup uncertain.*ACK missing"),
        identity,
    ):
        pytest.fail("foreign or incomplete ACK cannot establish custody")
    assert api.deletes == []
    assert len(api.objects) == 1


@pytest.mark.parametrize("field", ["uid", "owner", "namespace"])
def test_cleanup_preserves_replaced_or_foreign_binding(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    with pytest.raises(RegionalFixtureError, match="cleanup uncertain"), identity:
        binding = next(
            value
            for (kind, _name), value in api.objects.items()
            if kind == "RoleBinding"
        )
        if field == "owner":
            binding["metadata"]["labels"][OWNER_LABEL] = "foreign"
        else:
            binding["metadata"][field] = "foreign"
    assert [kind for kind, _name, _options in api.deletes] == ["Role"]
    assert [kind for kind, _name in api.objects] == ["RoleBinding"]
    assert not identity.kubeconfig.parent.exists(), (
        "refusing to delete a foreign binding must still remove local credentials"
    )


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_cleanup_delete_carries_atomic_uid_and_version_preconditions(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)

    def replace(document: dict[str, Any]) -> None:
        document["metadata"][field] = "replacement"

    api.before_delete = replace
    with pytest.raises(RegionalFixtureError, match="cleanup uncertain"), identity:
        pass
    assert len(api.objects) == 2
    assert [kind for kind, _name, _options in api.deletes] == ["RoleBinding", "Role"]
    assert all(
        options["preconditions"] == {"uid": f"{kind}-uid", "resourceVersion": "1"}
        for kind, _name, options in api.deletes
    ), "cleanup must send the pre-race UID and resourceVersion as API preconditions"


@pytest.mark.parametrize("failure", ["delete", "read", "recreate"])
def test_cleanup_errors_and_post_delete_replacement_are_never_hidden(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    if failure == "delete":
        api.delete_failure = "RoleBinding"
    elif failure == "read":
        api.read_failure = "RoleBinding"
    else:

        def recreate(document: dict[str, Any]) -> None:
            if document["kind"] == "RoleBinding":
                document["metadata"]["uid"] = "replacement"
                api.objects[("RoleBinding", document["metadata"]["name"])] = document

        api.after_delete = recreate
    with (
        pytest.raises(RegionalFixtureError, match="cleanup uncertain") as error,
        identity,
    ):
        pass
    assert SENSITIVE not in "".join(traceback.format_exception(error.value))
    assert not identity.kubeconfig.parent.exists(), (
        "RBAC cleanup failures must not leave the private kubeconfig behind"
    )
    assert not any(kind == "Role" for kind, _name in api.objects), (
        "failure to clean a binding must not skip cleanup of the acknowledged Role"
    )
    assert sum(kind == "RoleBinding" for kind, _name, _options in api.deletes) <= 1


def test_body_exception_is_preserved_after_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="submission failed"), identity:
        raise ValueError("submission failed")
    assert api.objects == {}
    assert not identity.kubeconfig.parent.exists(), (
        "a submission exception must still trigger private credential cleanup"
    )


def test_cleanup_interruption_still_removes_private_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)

    def interrupted(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if "delete" in command:
            raise KeyboardInterrupt
        return api.run(command, **kwargs)

    with pytest.raises(KeyboardInterrupt) as error, identity:
        monkeypatch.setattr(api.regional, "run", interrupted)
    assert len(api.objects) == 2
    assert not identity.kubeconfig.parent.exists(), (
        "interrupted RBAC cleanup must still remove the private credentials"
    )
    assert "cleanup uncertain" in "".join(error.value.__notes__)


def test_cleanup_uses_current_version_only_for_the_acknowledged_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    with identity:
        for document in api.objects.values():
            document["metadata"]["resourceVersion"] = "2"
    assert api.objects == {}
    assert all(
        options["preconditions"] == {"uid": f"{kind}-uid", "resourceVersion": "2"}
        for kind, _name, options in api.deletes
    ), "cleanup must pair the acknowledged UID with the freshly read resourceVersion"


@pytest.mark.parametrize("symlink", [False, True], ids=["file", "symlink"])
def test_config_replacement_is_not_followed_or_removed(
    symlink: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign"
    foreign.write_text("untouched", encoding="utf-8")
    with pytest.raises(RegionalFixtureError, match="cleanup uncertain"), identity:
        identity.kubeconfig.unlink()
        if symlink:
            identity.kubeconfig.symlink_to(foreign)
        else:
            identity.kubeconfig.write_text("foreign", encoding="utf-8")
    assert identity.kubeconfig.is_symlink() is symlink
    assert identity.kubeconfig.read_text(encoding="utf-8") == (
        "untouched" if symlink else "foreign"
    )
    assert foreign.read_text(encoding="utf-8") == "untouched"
    assert api.objects == {}


def test_private_directory_replacement_is_never_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    moved = tmp_path / "owned-moved"
    with pytest.raises(RegionalFixtureError, match="cleanup uncertain"), identity:
        identity.kubeconfig.parent.rename(moved)
        identity.kubeconfig.parent.mkdir()
        identity.kubeconfig.write_text("foreign", encoding="utf-8")
    assert identity.kubeconfig.read_text(encoding="utf-8") == "foreign"
    assert list(moved.iterdir()) == []
    assert api.objects == {}


@pytest.mark.parametrize("race", ["before-apply", "after-get"])
def test_real_training_submit_apply_cannot_overwrite_a_racing_workload(
    race: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity, api = harness(tmp_path, monkeypatch)
    monkeypatch.delenv(training_submit_cli.SITE_ENV, raising=False)
    manifest = tmp_path / "workload.json"
    manifest.write_text(
        json.dumps(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "train"},
                "spec": {
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {"name": "trainer", "image": "synthetic:local"}
                            ],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with identity:
        assert api.workload is None
        if race == "before-apply":
            api.workload = {"metadata": {"name": "train", "uid": "foreign"}}
        else:
            api.create_race = True
        args = training_submit_cli.parser().parse_args(
            [
                str(manifest),
                "--job-id",
                "train",
                "--attempt-id",
                "train-a1",
                "--kubeconfig",
                str(identity.kubeconfig),
                "--context",
                CONTEXT,
                "--namespace",
                NAMESPACE,
                "--runtime-profile-version",
                "synthetic",
            ]
        )
        assert training_submit_cli.run(args, runner=api.apply) == 1
        assert api.workload == {"metadata": {"name": "train", "uid": "foreign"}}
        assert api.apply_commands[0][-3:] == ["apply", "-f", "-"]
        assert api.workload_requests == (
            [("GET", 200), ("PATCH", 403)]
            if race == "before-apply"
            else [("GET", 404), ("POST", 409)]
        )
    assert api.objects == {}
    assert api.workload is not None, (
        "identity cleanup must never delete training resources"
    )
    assert SENSITIVE not in capsys.readouterr().err
