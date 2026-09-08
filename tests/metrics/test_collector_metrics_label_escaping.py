"""M-25: Prometheus/OpenMetrics label-value escaping in collector metrics.

The exposition format requires three escapes in a label value -- backslash,
newline, and double-quote -- applied in that order. Escaping only the quote
(the previous behaviour) let a crafted node_id break out of its quoted string
and inject or truncate metric lines.
"""

from __future__ import annotations

from gpu_fault.app.collector_metrics import (
    _escape_label_value,
    _labels,
    aggregate_lines,
)


def test_escape_covers_backslash_newline_and_quote_in_order() -> None:
    # x  \  y  <newline>  z  "  w
    raw = 'x\\y\nz"w'
    assert _escape_label_value(raw) == 'x\\\\y\\nz\\"w'


def test_backslash_is_escaped_before_the_inserted_escapes() -> None:
    # A real newline becomes a two-char escape (backslash + n)...
    assert _escape_label_value("\n") == "\\n"
    # ...while a literal backslash-then-n becomes an escaped backslash + n, so
    # the two inputs stay unambiguous once escaped.
    assert _escape_label_value("\\n") == "\\\\n"
    assert _escape_label_value("\n") != _escape_label_value("\\n")


def test_label_string_is_well_formed_for_a_crafted_value() -> None:
    nasty = 'evil"} injected 1\ngpu_fault_pwned{a="b'
    labels = _labels("cluster-a", "gpu", "metrics", node_id=nasty)

    # No raw newline survives to split the exposition line.
    assert "\n" not in labels
    # The only quotes left are the delimiters plus the escaped ones; every
    # literal quote from the value is backslash-escaped.
    assert labels.count('\\"') == nasty.count('"')
    # The crafted metric name cannot start a fresh line.
    assert "\ngpu_fault_pwned" not in labels


def test_aggregate_lines_never_emit_a_raw_newline_from_labels() -> None:
    rows = [
        {
            "cluster_id": "cluster-a",
            "node_id": 'node\n"a',
            "collector": "gpu-metrics",
            "channel": "metrics",
            "last_success_age_seconds": 120.0,
            "silent": True,
            "erroring": False,
        }
    ]
    lines = aggregate_lines(rows, top_n=5)
    # Each returned element is a single exposition line: no embedded newline
    # from the label value can smuggle in an extra line.
    for line in lines:
        assert "\n" not in line
    # The top-node line carries the escaped node id, not a raw quote/newline.
    top = [
        line for line in lines if line.startswith("gpu_fault_collector_silent_top_node")
    ]
    assert top and '\\"' in top[0] and "\\n" in top[0]
