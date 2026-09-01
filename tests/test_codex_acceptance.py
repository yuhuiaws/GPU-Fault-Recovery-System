from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import subprocess
from typing import cast
from unittest import mock

import pytest

from tools.codex_acceptance import (
    CodexAcceptanceBackend,
    CodexInvocationError,
    InvalidAcceptanceInput,
    InvalidCodexOutput,
    MandatoryOrderEdge,
    build_codex_environment,
)


def _manual_case(case_id: str) -> dict[str, object]:
    return {
        "id": case_id,
        "title": f"Manual case {case_id}",
        "category": "regional-acceptance",
        "level": "end-to-end",
        "risk": "read-only-signal-replay",
        "problem": "Validate the recorded regional behavior.",
        "injection": "Inspect existing evidence without changing the environment.",
        "expected": ["The existing evidence proves the expected behavior."],
        "automation": "manual",
        "procedure": f"docs/acceptance.md#{case_id.lower()}",
    }


def _regional_case(
    case_id: str,
    phase: str,
) -> dict[str, object]:
    return {
        "id": case_id,
        "title": f"Regional case {case_id}",
        "phase": phase,
        "risk": "non-destructive",
        "summary": f"Validate {case_id}.",
        "automation": "manual",
        "expected": [f"{case_id} completes."],
    }


def _analysis_result(
    case_id: str,
    *,
    status: str = "PASS",
    affected_dependents: Sequence[str] = (),
) -> dict[str, object]:
    if status == "PASS":
        failure_details: list[str] = []
        reproduction: list[str] = []
        evidence: list[dict[str, str]] = [
            {
                "source": "artifacts/read-only-report.json",
                "observation": "The expected event is present.",
            }
        ]
        blockers: list[str] = []
        human_actions: list[str] = []
    elif status == "FAIL":
        failure_details = ["The expected event is absent."]
        reproduction = ["Inspect artifacts/read-only-report.json."]
        evidence = [
            {
                "source": "artifacts/read-only-report.json",
                "observation": "No expected event was recorded.",
            }
        ]
        blockers = []
        human_actions = []
    elif status == "BLOCKED":
        failure_details = ["The referenced artifact is unavailable."]
        reproduction = ["Check whether artifacts/read-only-report.json exists."]
        evidence = []
        blockers = ["Required evidence is unavailable."]
        human_actions = []
    else:
        failure_details = []
        reproduction = ["Review the controlled maintenance checklist."]
        evidence = []
        blockers = []
        human_actions = ["Run the approved manual observation."]
    if status == "NEEDS_HUMAN":
        test_process = [
            {
                "step_id": "request-human-action",
                "kind": "requires-human",
                "description": "Record the required controlled human action.",
                "depends_on": [],
                "expected": "An authorized operator performs the observation.",
                "executor_ref": "human",
                "status": "PLANNED",
                "evidence": [],
            }
        ]
    else:
        test_process = [
            {
                "step_id": "inspect-evidence",
                "kind": "evidence-check",
                "description": "Inspect the existing read-only evidence.",
                "depends_on": [],
                "expected": "The evidence resolves the case expectation.",
                "executor_ref": "provided-observations",
                "status": status,
                "evidence": evidence,
            }
        ]
    return {
        "case_id": case_id,
        "status": status,
        "summary": f"{case_id} result",
        "failure_details": failure_details,
        "reproduction": reproduction,
        "evidence": evidence,
        "test_process": test_process,
        "affected_dependents": list(affected_dependents),
        "blockers": blockers,
        "human_actions": human_actions,
    }


def _analysis_payload(*results: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "results": [dict(result) for result in results],
    }


def _review_payload(
    case_id: str,
    *,
    status: str = "PASS",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "review": {
            "case_id": case_id,
            "status": status,
            "summary": "Independent evidence review completed.",
            "findings": ["The artifact matches the expected case identity."],
            "missing_evidence": (
                ["A required artifact is absent."] if status == "BLOCKED" else []
            ),
            "contradictions": (
                ["The artifact contradicts the expected state."]
                if status == "FAIL"
                else []
            ),
            "human_actions": (
                ["Obtain an operator attestation."] if status == "NEEDS_HUMAN" else []
            ),
        },
    }


def _dependency_payload(
    *,
    trusted: bool = False,
    first_depends_on: Sequence[str] = (),
    second_depends_on: Sequence[str] = ("case-a",),
    first_executor: str = "codex-read-only",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "trusted": trusted,
        "results": [
            {
                "case_id": "case-b",
                "depends_on": list(second_depends_on),
                "locks": [
                    {
                        "resource": "regional-evidence",
                        "mode": "shared",
                    }
                ],
                "executor": "human",
                "confidence": 0.75,
                "rationale": "The second phase follows the first phase.",
            },
            {
                "case_id": "case-a",
                "depends_on": list(first_depends_on),
                "locks": [
                    {
                        "resource": "regional-evidence",
                        "mode": "shared",
                    }
                ],
                "executor": first_executor,
                "confidence": 0.9,
                "rationale": "Static evidence can be inspected read-only.",
            },
        ],
    }


def _argument_after(command: Sequence[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _success_runner(
    payload: Mapping[str, object],
    capture: dict[str, object] | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        command: Sequence[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        schema_path = Path(_argument_after(arguments, "--output-schema"))
        output_path = Path(_argument_after(arguments, "--output-last-message"))
        if capture is not None:
            capture["command"] = arguments
            capture["kwargs"] = dict(kwargs)
            capture["schema"] = cast(
                object,
                json.loads(schema_path.read_text(encoding="utf-8")),
            )
        output_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    return run


def _backend(
    repository: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> CodexAcceptanceBackend:
    default_environment = {
        "HOME": "/home/tester",
        "CODEX_HOME": "/home/tester/.codex",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    return CodexAcceptanceBackend(
        repository,
        timeout_seconds=37,
        environment=default_environment if environment is None else environment,
    )


def test_manual_batch_uses_hardened_codex_exec_and_clean_environment(
    tmp_path: Path,
) -> None:
    source_environment = {
        "HOME": "/home/tester",
        "CODEX_HOME": "/secure/codex-home",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "AWS_ACCESS_KEY_ID": "sensitive",
        "AWS_SECRET_ACCESS_KEY": "sensitive",
        "AWS_SESSION_TOKEN": "sensitive",
        "KUBECONFIG": "/sensitive/kubeconfig",
        "GPU_FAULT_EXECUTION_TOKEN": "sensitive",
        "GPU_FAULT_CLUSTER_TOKEN": "sensitive",
        "GPU_FAULT_NODE_ACTION_KEY": "sensitive",
        "GPU_FAULT_STORE_URL": "postgresql://sensitive",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://sensitive",
        "DATABASE_URL": "postgresql://sensitive",
        "POSTGRES_DSN": "postgresql://sensitive",
        "OPENAI_API_KEY": "sensitive",
        "HTTPS_PROXY": "https://user:password@proxy.invalid",
        "UNRELATED_SECRET": "sensitive",
    }
    capture: dict[str, object] = {}
    backend = _backend(tmp_path, environment=source_environment)

    with mock.patch(
        "tools.codex_acceptance.subprocess.run",
        side_effect=_success_runner(
            _analysis_payload(_analysis_result("case-a")),
            capture,
        ),
    ) as run:
        results = backend.analyze_manual_cases([_manual_case("case-a")])

    assert results["case-a"].status == "PASS"
    run.assert_called_once()
    command = cast(list[str], capture["command"])
    kwargs = cast(dict[str, object], capture["kwargs"])
    environment = cast(dict[str, str], kwargs["env"])
    prompt = cast(str, kwargs["input"])
    schema = cast(dict[str, object], capture["schema"])

    assert command[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--strict-config" in command
    assert _argument_after(command, "--sandbox") == "read-only"
    assert _argument_after(command, "--enable") == "multi_agent"
    disabled_features = {
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--disable"
    }
    assert disabled_features == {
        "apps",
        "browser_use",
        "computer_use",
        "hooks",
        "image_generation",
        "plugins",
        "skill_mcp_dependency_install",
        "skill_search",
    }
    assert _argument_after(command, "--cd") == str(tmp_path.resolve())
    assert command[-1] == "-"
    assert [item for item in command if "dangerously" in item] == []
    assert 'web_search="disabled"' in command
    assert "allow_login_shell=false" in command
    assert 'shell_environment_policy.inherit="core"' in command
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["timeout"] == 37
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL

    assert environment == {
        "HOME": "/home/tester",
        "CODEX_HOME": "/secure/codex-home",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CODEX_NON_INTERACTIVE": "1",
        "NO_COLOR": "1",
    }
    assert "one separate read-only subagent task for every supplied case" in prompt
    assert "must not modify source, configuration, environment state" in prompt
    assert "must not propose or apply a patch" in prompt
    assert "must not perform a repair-and-retry loop" in prompt
    assert "continue all remaining cases" in prompt
    assert "only the main runner may aggregate" in prompt
    assert "failure_details, reproduction, evidence" in prompt
    assert "affected_dependents" in prompt

    assert schema["additionalProperties"] is False
    properties = cast(dict[str, object], schema["properties"])
    results_schema = cast(dict[str, object], properties["results"])
    item_schema = cast(dict[str, object], results_schema["items"])
    assert item_schema["additionalProperties"] is False
    required = cast(list[str], item_schema["required"])
    assert {
        "failure_details",
        "reproduction",
        "evidence",
        "affected_dependents",
    } <= set(required)


def test_environment_builder_uses_an_allowlist() -> None:
    environment = build_codex_environment(
        {
            "HOME": "/home/tester",
            "CODEX_HOME": "/home/tester/.codex",
            "PATH": "/bin",
            "SSL_CERT_FILE": "/etc/ssl/cert.pem",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "KUBECONFIG": "/secret",
            "GPU_FAULT_FLEET_MASTER": "secret",
            "GPU_FAULT_STORE_URL": "secret",
            "CUSTOM_DSN": "secret",
            "CUSTOM_TOKEN": "secret",
        }
    )

    assert environment == {
        "HOME": "/home/tester",
        "CODEX_HOME": "/home/tester/.codex",
        "PATH": "/bin",
        "SSL_CERT_FILE": "/etc/ssl/cert.pem",
        "CODEX_NON_INTERACTIVE": "1",
        "NO_COLOR": "1",
    }


def test_batch_maps_every_case_and_does_not_stop_after_failure(
    tmp_path: Path,
) -> None:
    payload = _analysis_payload(
        _analysis_result("case-b", status="BLOCKED"),
        _analysis_result(
            "case-a",
            status="FAIL",
            affected_dependents=("case-b",),
        ),
    )
    backend = _backend(tmp_path)

    with mock.patch(
        "tools.codex_acceptance.subprocess.run",
        side_effect=_success_runner(payload),
    ):
        results = backend.analyze_manual_cases(
            [_manual_case("case-a"), _manual_case("case-b")]
        )

    assert list(results) == ["case-a", "case-b"]
    assert results["case-a"].status == "FAIL"
    assert results["case-a"].failure_details == ("The expected event is absent.",)
    assert results["case-a"].reproduction == (
        "Inspect artifacts/read-only-report.json.",
    )
    assert results["case-a"].affected_dependents == ("case-b",)
    assert results["case-b"].status == "BLOCKED"
    assert results["case-b"].blockers == ("Required evidence is unavailable.",)


def test_evidence_review_is_an_independent_read_only_invocation(
    tmp_path: Path,
) -> None:
    capture: dict[str, object] = {}
    backend = _backend(tmp_path)

    with mock.patch(
        "tools.codex_acceptance.subprocess.run",
        side_effect=_success_runner(_review_payload("case-a"), capture),
    ):
        review = backend.review_evidence(
            _manual_case("case-a"),
            [
                {
                    "source": "artifacts/read-only-report.json",
                    "observation": "The expected event is present.",
                }
            ],
        )

    assert review.status == "PASS"
    prompt = cast(str, cast(dict[str, object], capture["kwargs"])["input"])
    schema = cast(dict[str, object], capture["schema"])
    assert "independent read-only evidence reviewer" in prompt
    assert "do not trust or reuse a prior agent verdict" in prompt
    assert "do not propose or apply a patch" in prompt
    assert "do not perform repair-and-retry behavior" in prompt
    properties = cast(dict[str, object], schema["properties"])
    review_schema = cast(dict[str, object], properties["review"])
    assert review_schema["additionalProperties"] is False


def test_dependency_proposal_is_structured_ordered_and_untrusted(
    tmp_path: Path,
) -> None:
    capture: dict[str, object] = {}
    backend = _backend(tmp_path)

    with mock.patch(
        "tools.codex_acceptance.subprocess.run",
        side_effect=_success_runner(_dependency_payload(), capture),
    ):
        proposal = backend.propose_dependencies(
            [
                _regional_case("case-a", "phase-1"),
                _regional_case("case-b", "phase-2"),
            ],
            [MandatoryOrderEdge(before="case-a", after="case-b")],
        )

    assert proposal.trusted is False
    assert [item.case_id for item in proposal.cases] == ["case-a", "case-b"]
    assert proposal.by_case()["case-b"].depends_on == ("case-a",)
    assert proposal.by_case()["case-b"].locks[0].resource == "regional-evidence"
    assert proposal.by_case()["case-a"].executor == "codex-read-only"
    assert proposal.by_case()["case-a"].confidence == 0.9

    command = cast(list[str], capture["command"])
    prompt = cast(str, cast(dict[str, object], capture["kwargs"])["input"])
    schema = cast(dict[str, object], capture["schema"])
    assert _argument_after(command, "--sandbox") == "read-only"
    assert "--output-schema" in command
    assert "separate read-only subagent tasks by phase" in prompt
    assert '"before":"case-a"' in prompt
    assert '"after":"case-b"' in prompt
    assert "UNTRUSTED PROPOSAL" in prompt
    assert "must never be executed" in prompt
    assert "do not propose or apply a patch" in prompt
    properties = cast(dict[str, object], schema["properties"])
    trusted_schema = cast(dict[str, object], properties["trusted"])
    assert trusted_schema == {"type": "boolean", "const": False}


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("timeout", CodexInvocationError),
        ("nonzero", CodexInvocationError),
        ("missing", InvalidCodexOutput),
        ("empty", InvalidCodexOutput),
        ("invalid-json", InvalidCodexOutput),
    ],
)  # type: ignore[untyped-decorator]
def test_subprocess_and_output_failures_are_fail_closed(
    tmp_path: Path,
    mode: str,
    expected_error: type[Exception],
) -> None:
    backend = _backend(tmp_path)

    def run(
        command: Sequence[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(arguments, 37)
        if mode == "nonzero":
            return subprocess.CompletedProcess(
                arguments,
                9,
                stdout="sensitive-token",
                stderr="sensitive-secret",
            )
        output_path = Path(_argument_after(arguments, "--output-last-message"))
        if mode == "empty":
            output_path.write_text("", encoding="utf-8")
        elif mode == "invalid-json":
            output_path.write_text("{not-json", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    with (
        mock.patch("tools.codex_acceptance.subprocess.run", side_effect=run),
        pytest.raises(expected_error) as error,
    ):
        backend.analyze_manual_cases([_manual_case("case-a")])

    assert "sensitive-token" not in str(error.value)
    assert "sensitive-secret" not in str(error.value)


@pytest.mark.parametrize(
    "mode",
    [
        "unsupported-status",
        "missing-required-field",
        "unknown-dependent",
        "fail-without-reproduction",
    ],
)  # type: ignore[untyped-decorator]
def test_batch_rejects_illegal_structured_results(
    tmp_path: Path,
    mode: str,
) -> None:
    result = _analysis_result("case-a")
    if mode == "unsupported-status":
        result["status"] = "NOT_RUN"
    elif mode == "missing-required-field":
        del result["failure_details"]
    elif mode == "unknown-dependent":
        result["affected_dependents"] = ["unknown-case"]
    else:
        result = _analysis_result("case-a", status="FAIL")
        result["reproduction"] = []
    backend = _backend(tmp_path)

    with (
        mock.patch(
            "tools.codex_acceptance.subprocess.run",
            side_effect=_success_runner(_analysis_payload(result)),
        ),
        pytest.raises(InvalidCodexOutput),
    ):
        backend.analyze_manual_cases([_manual_case("case-a")])


@pytest.mark.parametrize(
    "mode",
    [
        "trusted",
        "missing-mandatory-edge",
        "cycle",
        "unsupported-executor",
    ],
)  # type: ignore[untyped-decorator]
def test_dependency_proposal_rejects_unusable_output(
    tmp_path: Path,
    mode: str,
) -> None:
    if mode == "trusted":
        payload = _dependency_payload(trusted=True)
    elif mode == "missing-mandatory-edge":
        payload = _dependency_payload(second_depends_on=())
    elif mode == "cycle":
        payload = _dependency_payload(first_depends_on=("case-b",))
    else:
        payload = _dependency_payload(first_executor="shell-agent")
    backend = _backend(tmp_path)

    with (
        mock.patch(
            "tools.codex_acceptance.subprocess.run",
            side_effect=_success_runner(payload),
        ),
        pytest.raises(InvalidCodexOutput),
    ):
        backend.propose_dependencies(
            [
                _regional_case("case-a", "phase-1"),
                _regional_case("case-b", "phase-2"),
            ],
            [("case-a", "case-b")],
        )


def test_review_rejects_a_mismatched_case_id(tmp_path: Path) -> None:
    backend = _backend(tmp_path)

    with (
        mock.patch(
            "tools.codex_acceptance.subprocess.run",
            side_effect=_success_runner(_review_payload("other-case")),
        ),
        pytest.raises(InvalidCodexOutput),
    ):
        backend.review_evidence(_manual_case("case-a"), [])


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [_manual_case("case-a"), _manual_case("case-a")],
        [{**_manual_case("case-a"), "automation": "pytest"}],
    ],
)  # type: ignore[untyped-decorator]
def test_invalid_batch_input_never_invokes_codex(
    tmp_path: Path,
    cases: list[dict[str, object]],
) -> None:
    backend = _backend(tmp_path)

    with (
        mock.patch("tools.codex_acceptance.subprocess.run") as run,
        pytest.raises(InvalidAcceptanceInput),
    ):
        backend.analyze_manual_cases(cases)

    run.assert_not_called()


def test_invalid_mandatory_edge_never_invokes_codex(tmp_path: Path) -> None:
    backend = _backend(tmp_path)

    with (
        mock.patch("tools.codex_acceptance.subprocess.run") as run,
        pytest.raises(InvalidAcceptanceInput),
    ):
        backend.propose_dependencies(
            [_regional_case("case-a", "phase-1")],
            [("case-a", "unknown-case")],
        )

    run.assert_not_called()
