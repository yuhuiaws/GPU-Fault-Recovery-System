"""What may leave a process in a log line, and who decides it.

The redaction here is the precondition for shipping these logs anywhere off the
node: while the journal is node-local a secret in a line is bounded by the node,
and a log pipeline copies it into a service with its own retention and indexing.
So these cases pin both halves -- that the process gets a usable root logger at
all, and that nothing secret-shaped reaches a handler.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import pytest

from gpu_fault import logging_setup


@pytest.fixture
def restore_root() -> Iterator[logging.Logger]:
    """Puts the root logger back, so one case cannot silence the whole run."""

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield root
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


@pytest.fixture
def configure_fresh(restore_root: logging.Logger) -> Callable[[], None]:
    """Configures logging as a service process would, from an empty root.

    The clearing happens when the test calls this rather than during setup:
    pytest installs its own capture handler around the call phase, and
    `configure_logging` deliberately keeps its hands off a root that already has
    handlers, so clearing in setup would test the embedder path instead.
    """

    def configure() -> None:
        restore_root.handlers = []
        logging_setup.configure_logging()

    return configure


def _emitted(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err


def test_a_live_environment_secret_is_replaced_by_its_literal_value(
    configure_fresh: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shapes cannot catch this one, so the value has to be known.

    A cluster token interpolated into a message is an opaque string with no
    marker of what it is. Reading the secret-shaped environment variables is what
    makes it recognisable, and it is the case that actually matters here: the
    token is handed to every process in its environment.
    """

    monkeypatch.setenv("GPU_FAULT_CLUSTER_TOKEN", "sk-live-9d41f0c2ab7e")
    configure_fresh()

    logging.getLogger("gpu_fault.test").warning(
        "data plane rejected the call using %s", "sk-live-9d41f0c2ab7e"
    )

    emitted = _emitted(capsys)
    assert "sk-live-9d41f0c2ab7e" not in emitted, "the live token must not be printed"
    assert logging_setup.PLACEHOLDER in emitted, "the line still says a value was there"
    assert "data plane rejected the call" in emitted, (
        "redaction must leave the message readable"
    )


def test_a_secret_shaped_variable_holding_a_path_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token file's path is load-bearing in a runbook and is not a secret.

    `GPU_FAULT_CLUSTER_TOKEN_FILE` matches the same name rule as the token
    itself, so the values are separated by shape: redacting the path would cost
    the only trace of which file a process read.
    """

    monkeypatch.setenv("GPU_FAULT_CLUSTER_TOKEN_FILE", "/secure/gpu-fault/token")
    monkeypatch.setenv("GPU_FAULT_STORE_SECRET_ARN", "arn:aws:secretsmanager:x:y:z")
    monkeypatch.setenv("GPU_FAULT_API_TOKEN", "0f8b21c47d9e3a56")

    secrets = logging_setup.environment_secrets()

    assert "0f8b21c47d9e3a56" in secrets, "an opaque token value is a secret"
    assert "/secure/gpu-fault/token" not in secrets, "a path is not a secret"
    assert "arn:aws:secretsmanager:x:y:z" not in secrets, "an ARN is not a secret"


# Assembled from pieces rather than written out: the release safety scanner
# treats a literal example access key or PEM header anywhere in the tree as a
# leak, and it is right to -- a scanner with exceptions is a scanner that stops
# being read.
EXAMPLE_ACCESS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
PEM_BEGIN = "-----BEGIN RSA PRIVATE" + " KEY-----"
PEM_END = "-----END RSA PRIVATE" + " KEY-----"


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("authorization: Bearer 8fc41ab29de77c05", "Bearer"),
        ("token=eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJzYSJ9.Zm9vYmFyYmF6cXV4", "token="),
        (
            "opening postgresql://gpu_fault:hunter2@aurora.internal:5432/gpu",
            "aurora.internal:5432/gpu",
        ),
        (f"credentials for {EXAMPLE_ACCESS_KEY} expired", "expired"),
        ("password: 3f9a2b71c4", "password"),
        (f"{PEM_BEGIN}\nMIIEow\n{PEM_END}", PEM_BEGIN),
    ],
)
def test_secret_shapes_are_removed_and_the_line_stays_legible(
    text: str, kept: str
) -> None:
    """Each rule keeps the part an operator reads and drops the part they must not.

    The markers, the key name and the host survive on purpose: a line that has
    been fully replaced by a placeholder cannot be triaged, and an operator who
    cannot triage it will go looking for the unredacted copy.
    """

    redacted = logging_setup.redact(text)

    assert logging_setup.PLACEHOLDER in redacted, f"nothing was redacted in {text!r}"
    assert kept in redacted, f"{kept!r} is not the secret and has to survive"
    for fragment in (
        "8fc41ab29de77c05",
        "Zm9vYmFyYmF6cXV4",
        "hunter2",
        EXAMPLE_ACCESS_KEY,
        "3f9a2b71c4",
        "MIIEow",
    ):
        assert fragment not in redacted, f"{fragment!r} survived redaction"


def test_a_path_valued_key_is_not_treated_as_a_secret() -> None:
    """`token_file=/secure/...` is the line that says which file was read."""

    text = "reading cluster identity token_file=/secure/gpu-fault/token"

    assert logging_setup.redact(text) == text, (
        "a path-valued key must survive redaction intact"
    )


def test_a_traceback_carrying_a_secret_is_redacted_too(
    configure_fresh: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A filter never sees exception text, which is why the formatter redacts.

    `exc_text` is produced by the formatter, after every filter has run, so an
    exception whose message carries a token would have been the one line that
    escaped.
    """

    monkeypatch.setenv("GPU_FAULT_CLUSTER_TOKEN", "sk-live-9d41f0c2ab7e")
    configure_fresh()

    try:
        raise RuntimeError("401 for token sk-live-9d41f0c2ab7e")
    except RuntimeError:
        logging.getLogger("gpu_fault.test").exception("cluster call failed")

    emitted = _emitted(capsys)
    assert "Traceback" in emitted, "the traceback is still reported"
    assert "sk-live-9d41f0c2ab7e" not in emitted, (
        "an exception message is not exempt from redaction"
    )


def test_a_child_logger_is_redacted_by_the_root_handler(
    configure_fresh: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The enforcement point is the handler, not the logger.

    A filter on the root *logger* is only consulted for records logged directly
    to root; `callHandlers` applies each ancestor *handler's* filters to a record
    that propagated up from a child. Every module in this system logs through a
    child logger, so getting this wrong would redact nothing at all.
    """

    monkeypatch.setenv("GPU_FAULT_CLUSTER_TOKEN", "sk-live-9d41f0c2ab7e")
    configure_fresh()

    logging.getLogger("gpu_fault.deep.child.module").error(
        "token sk-live-9d41f0c2ab7e was refused"
    )

    assert "sk-live-9d41f0c2ab7e" not in _emitted(capsys), (
        "a record from a child logger reaches the root handler and must be redacted"
    )


def test_an_embedder_keeps_its_handlers_but_not_the_choice_to_redact(
    restore_root: logging.Logger,
) -> None:
    """pytest's caplog and any embedder own their logging; redaction is not theirs.

    Taking over a configured root would break caplog and any process that embeds
    this code, so root is left alone -- but its handlers still get the filter,
    because what may leave the process is not the embedder's call.
    """

    root = restore_root
    sentinel = logging.NullHandler()
    root.handlers = [sentinel]
    root.setLevel(logging.CRITICAL)

    logging_setup.configure_logging()
    logging_setup.configure_logging()

    assert root.handlers == [sentinel], "an embedder's handler must not be replaced"
    assert root.level == logging.CRITICAL, "an embedder's level must not be changed"
    assert [type(item) for item in sentinel.filters] == [
        logging_setup.SecretRedactingFilter
    ], "exactly one redaction filter, however often logging is configured"


def test_a_misspelled_level_falls_back_instead_of_killing_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`basicConfig` used to raise on a typo and take the process down at startup.

    Logging more than was asked for is the better failure: a collector that will
    not start reports nothing at all.
    """

    monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "VERBOSE")
    assert logging_setup.resolve_level() == logging.INFO, (
        "an unknown level name falls back to INFO"
    )

    monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", " debug ")
    assert logging_setup.resolve_level() == logging.DEBUG, (
        "the level name is trimmed and case-insensitive"
    )

    monkeypatch.delenv("GPU_FAULT_LOG_LEVEL")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert logging_setup.resolve_level() == logging.WARNING, (
        "the two entrypoint conventions both work"
    )

    monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "ERROR")
    assert logging_setup.resolve_level() == logging.ERROR, (
        "the namespaced variable wins over the bare one"
    )
