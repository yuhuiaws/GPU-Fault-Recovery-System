"""The local outbox audit must detect broken delivery, including false success."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_collector_outbox as audit
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401


def test_audit_rejects_unwritable_sink_that_falsely_returns_success(
    monkeypatch: Any,
) -> None:
    sink_type = audit.HttpEventSink
    original_transport = audit.collector_sinks.urlopen

    class BrokenSink:
        def post(self, *args: Any, **kwargs: Any) -> None:
            return None

    def sink(*args: Any, **kwargs: Any) -> Any:
        parent = Path(kwargs["outbox_path"]).parent
        return BrokenSink() if parent.is_file() else sink_type(*args, **kwargs)

    monkeypatch.setattr(audit, "HttpEventSink", sink)
    with pytest.raises(audit.OutboxAuditFailure, match="unexpectedly buffered"):
        audit.audit_outbox()
    assert audit.collector_sinks.urlopen is original_transport, (
        "audit must restore transport even on failure"
    )


def test_expect_without_observed_value_retains_exact_failure_message() -> None:
    with pytest.raises(audit.OutboxAuditFailure, match="^missing proof$"):
        audit.expect(False, "missing proof")
