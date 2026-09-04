"""One root logging configuration, and one place that redacts what it emits.

Six entrypoints had grown their own ``basicConfig`` call -- the API, the
collector CLI, the cluster executor (twice), the completion controller, the
Aurora credential refresher and the node installer reconciler -- and they had
already drifted: two read only ``LOG_LEVEL``, two read only
``GPU_FAULT_LOG_LEVEL``, and two of them left the logger name out of the format,
so the same line could not be attributed to a process. A typo in the level
crashed the process at startup rather than defaulting.

The defect all of them exist to fix is worth restating, because it is the reason
a misconfiguration could look like silence: uvicorn's default LOGGING_CONFIG
configures only the ``uvicorn*`` loggers -- it has no ``root`` entry -- and
nothing else in these processes ever called ``basicConfig``. Root therefore
stayed at WARNING with no handlers, so every ``LOGGER.info`` was discarded
outright and every warning or traceback fell through to
``logging.lastResort``, which prints a bare message with no timestamp, level or
logger name. On a live node in 2026-08 that made a collector with zero delivery
lines indistinguishable from a dead one.

Redaction lives here for one reason: it is the precondition for shipping these
logs anywhere off the node. As long as the journal is node-local, a secret in a
line is bounded by the node; a log pipeline copies it into a service with its
own retention, indexing and access control. So the rule is that nothing leaves a
process unredacted, and the enforcement point is the handler rather than a
logger: a filter on the root *logger* is only consulted for records logged
directly to root, while ``callHandlers`` applies every ancestor *handler's*
filters to records that propagated up from a child logger.

Two mechanisms, because neither alone is enough:

* the literal values of secret-shaped environment variables, which is what
  actually catches this system's live secrets (a cluster token interpolated into
  a message is otherwise just an opaque string);
* the shapes -- ``Bearer`` headers, service-account JWTs, PEM private key
  bodies, DSN userinfo, AWS access key ids, and ``key=value`` pairs whose key
  names a secret.

The last rule redacts the *value*, so a line that means to report the name of a
Kubernetes Secret should write it as ``name=`` or ``secret_name=`` rather than
``secret=``: the filter cannot tell a name from the thing it names, and it
resolves that in favour of not printing secrets.
"""

from __future__ import annotations

import logging
import os
import re

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
PLACEHOLDER = "[redacted]"

# A variable whose name promises a secret. `_FILE`/`_PATH` names match too, and
# their values are excluded by shape below rather than by name, because the same
# variable name is used for both a token and the path of the file holding it.
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|TOKENS|SECRET|SECRETS|PASSWORD|PASSWD|CREDENTIAL"
    r"|CREDENTIALS|PRIVATE_KEY|API_KEY|SESSION_KEY|SIGNING_KEY)(?:_|$)"
)

# Values that carry no secret and are load-bearing in a runbook: a path, a URL,
# an ARN. Redacting `--token-file /secure/...` out of the logs would cost the
# only trace of which file a process read, and gain nothing.
_NOT_A_SECRET_VALUE = re.compile(r"^(?:/|\./|\.\./|~|[a-z][a-z0-9+.-]*://|arn:)")

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # The body between the markers only: the markers stay so the line still says
    # that a key was there.
    (
        re.compile(
            r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)[\s\S]*?"
            r"(-----END [A-Z ]*PRIVATE KEY-----)"
        ),
        rf"\1{PLACEHOLDER}\2",
    ),
    # The credential after an HTTP auth scheme. The scheme name stays, because
    # which scheme was offered is half of why a 401 happened.
    (
        re.compile(r"(?i)\b(bearer|basic|digest|negotiate)\s+[A-Za-z0-9._~+/=-]{8,}"),
        rf"\1 {PLACEHOLDER}",
    ),
    # A service-account or OIDC token: three base64url segments, and the first
    # one always starts `eyJ` because it is a JSON header.
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"),
        PLACEHOLDER,
    ),
    # `scheme://user:password@host`, which is how a DSN leaks.
    (re.compile(r"(://[^\s:/@]+:)[^\s@/]+(@)"), rf"\1{PLACEHOLDER}\2"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), PLACEHOLDER),
    (
        re.compile(
            r"(?i)\b(token|password|passwd|secret|credential|api[_-]?key"
            r"|authorization)(\"?\s*[:=]\s*\"?)"
            # A path, URL or ARN is not the secret. Neither is an auth scheme
            # name: the rule above has already taken the credential after it, and
            # replacing the scheme too would say less while hiding nothing more.
            r"(?!/|\./|\.\./|~|[a-z][a-z0-9+.-]*://|arn:"
            r"|bearer\b|basic\b|digest\b|negotiate\b)([^\s\"',}&]{4,})"
        ),
        rf"\1\2{PLACEHOLDER}",
    ),
)


def environment_secrets() -> tuple[str, ...]:
    """The live secret values this process was handed in its environment.

    Read once, when logging is configured: a value that enters the environment
    later is covered by the shape rules rather than by its literal text. Longest
    first, so a secret that contains another one is replaced whole.
    """

    values = {
        value
        for name, value in os.environ.items()
        if _SECRET_NAME.search(name.upper())
        and len(value.strip()) >= 8
        and not _NOT_A_SECRET_VALUE.match(value.strip())
    }
    return tuple(sorted((value.strip() for value in values), key=len, reverse=True))


def redact(text: str, literals: tuple[str, ...] = ()) -> str:
    """Remove secret values and secret shapes from one line of log text."""

    for literal in literals:
        text = text.replace(literal, PLACEHOLDER)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Redacts a record's message in place, before any handler formats it.

    Mutating the record is deliberate: every handler on the process then emits
    the redacted text, including handlers this code did not install.
    """

    def __init__(self, literals: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.literals = environment_secrets() if literals is None else literals

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - a broken format string
            # Let logging report its own formatting error rather than swallowing
            # the record here.
            return True
        redacted = redact(message, self.literals)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    """Redacts the formatted record, which is what covers tracebacks.

    A filter never sees exception text: ``exc_text`` is produced by the
    formatter, after every filter has run. An exception carrying a token in its
    message would otherwise be the one line that escapes redaction.
    """

    def __init__(self, fmt: str = LOG_FORMAT) -> None:
        super().__init__(fmt)
        self.literals = environment_secrets()

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self.literals)


def resolve_level() -> int:
    """The configured level, or INFO when it is missing or misspelled.

    A typo used to raise out of ``basicConfig`` and take the process down at
    startup, which is a worse failure than logging more than was asked for.
    """

    name = (
        (os.getenv("GPU_FAULT_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO")
        .strip()
        .upper()
    )
    level = logging.getLevelNamesMapping().get(name)
    if level is None:
        return logging.INFO
    return level


def install_redaction(handler: logging.Handler) -> None:
    """Attaches the redaction filter to a handler, at most once."""

    if any(isinstance(item, SecretRedactingFilter) for item in handler.filters):
        return
    handler.addFilter(SecretRedactingFilter())


def configure_logging() -> None:
    """Give this process a root logging configuration that redacts.

    Root is left alone when it already has handlers, so pytest's caplog and any
    process that embeds this code keep control of their own logging -- but their
    handlers still get the redaction filter, because what may leave the process
    is not theirs to decide.
    """

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            install_redaction(handler)
        return
    logging.basicConfig(level=resolve_level(), format=LOG_FORMAT)
    for handler in root.handlers:
        handler.setFormatter(RedactingFormatter())
        install_redaction(handler)
