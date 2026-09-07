from __future__ import annotations

import pytest

from gpu_fault import env_validation
from gpu_fault.env import (
    BOOLEAN_TOKENS,
    FALSE_TOKENS,
    TRUE_TOKENS,
    env_bool,
    invalid_boolean_message,
    parse_bool,
)

NAME = "GPU_FAULT_TEST_SWITCH"


@pytest.mark.parametrize("raw", sorted(TRUE_TOKENS))
def test_every_true_token_enables(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NAME, raw)

    assert env_bool(NAME) is True


@pytest.mark.parametrize("raw", sorted(FALSE_TOKENS))
def test_every_false_token_disables(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NAME, raw)

    assert env_bool(NAME, True) is False


def test_token_sets_are_the_documented_ones() -> None:
    assert TRUE_TOKENS == frozenset({"1", "true", "yes", "on"})
    assert FALSE_TOKENS == frozenset({"0", "false", "no", "off"})
    assert BOOLEAN_TOKENS == TRUE_TOKENS | FALSE_TOKENS
    assert not (TRUE_TOKENS & FALSE_TOKENS)


def test_validation_shares_the_single_token_definition() -> None:
    """One frozenset, not two copies that could drift."""

    assert env_validation.BOOLEAN_TOKENS is BOOLEAN_TOKENS
    assert env_validation.TRUE_TOKENS is TRUE_TOKENS


@pytest.mark.parametrize("raw", ["TRUE", " Yes ", "\tON\n", "1 "])
def test_case_and_surrounding_whitespace_are_ignored(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(NAME, raw)

    assert env_bool(NAME) is True


@pytest.mark.parametrize("default", [True, False])
def test_unset_falls_back_to_the_default(
    default: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(NAME, raising=False)

    assert env_bool(NAME, default) is default


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_means_unset(
    default: bool, raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(NAME, raw)

    assert env_bool(NAME, default) is default


@pytest.mark.parametrize("raw", ["ture", "2", "enabled", "yes please"])
def test_anything_else_is_loud(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo used to read as "off"; a misconfiguration must not be silent."""

    monkeypatch.setenv(NAME, raw)

    with pytest.raises(ValueError) as failure:
        env_bool(NAME, True)

    message = str(failure.value)
    assert message == invalid_boolean_message(NAME)
    assert NAME in message
    for token in BOOLEAN_TOKENS:
        assert token in message
    assert raw not in message, "the message must never quote the value"


def test_explicit_mapping_is_read_instead_of_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NAME, "true")

    assert env_bool(NAME, True, environ={NAME: "no"}) is False
    assert env_bool(NAME, True, environ={}) is True


def test_parse_bool_handles_text_from_elsewhere() -> None:
    assert parse_bool("On", name=NAME) is True
    assert parse_bool(" 0 ", name=NAME) is False
    with pytest.raises(ValueError, match=NAME):
        parse_bool("", name=NAME)
    with pytest.raises(ValueError, match=NAME):
        parse_bool("maybe", name=NAME)
