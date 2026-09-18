"""Execute the shipped Store-I/O probe with synthetic, local-only HTTP I/O."""

from __future__ import annotations

import builtins
import io
import json
import sys
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from gpu_fault_release import regional_release_probes as probes
from gpu_fault_release import regional_release_validation as validation
from gpu_fault_release.regional_release_runtime_identity import CONTROL_PLANE_PYTHON
from tests.regional._cov95_release_support import (
    ResourceRelease,
    deployment,
    json_response,
)

METRIC = "gpu_fault_store_io_rejections_total"
PROCESSES = "gpu_fault_metrics_aggregation_processes"
DEGRADED = "gpu_fault_metrics_aggregation_degraded"
REASONS = ("capacity", "deadline", "backend_unavailable")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MISSING = object()


def zero_view_report(
    *,
    process_count: int = 1,
    complete: bool = True,
    all_labeled: bool = True,
    all_zero: bool = True,
) -> dict[str, int | bool]:
    """A current report fixture; each flag can be refused independently."""
    return {
        "series_count": 3 * process_count,
        "process_count": process_count,
        "complete": complete,
        "all_labeled": all_labeled,
        "all_zero": all_zero,
    }


def _counter(
    slot: int | str = 0,
    reason: str = "capacity",
    value: str = "0",
    *,
    process_id: str | None = None,
) -> str:
    labels = {"reason": reason, "process": str(slot)}
    if process_id is not None:
        labels["process_id"] = process_id
    encoded = ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items())
    return f"{METRIC}{{{encoded}}} {value}"


def _view(slots: tuple[int, ...] = (0,), *, process_id: str | None = None) -> list[str]:
    return [
        f"{PROCESSES} {len(slots)}",
        f"{DEGRADED} 0",
        *(
            _counter(slot, reason, process_id=process_id)
            for slot in slots
            for reason in REASONS
        ),
    ]


class _Response:
    def __init__(self, payload: bytes, read_error: Exception | None) -> None:
        self.stream = io.BytesIO(payload)
        self.read_error = read_error
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self.read_error is not None:
            raise self.read_error
        return self.stream.read(size)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.closed = True
        self.stream.close()


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lines: list[str] | bytes,
    *,
    read_error: Exception | None = None,
    open_error: Exception | None = None,
) -> dict[str, Any]:
    source = compile(
        probes.probe_source("store_io_rejection_series"), "<probe>", "exec"
    )
    body = lines if isinstance(lines, bytes) else ("\n".join(lines) + "\n").encode()
    response = _Response(body, read_error)
    requests: list[tuple[str, int]] = []
    original_import = builtins.__import__

    def stdlib_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.partition(".")[0] in sys.stdlib_module_names, name
        return original_import(name, *args, **kwargs)

    def urlopen(url: str, *, timeout: int) -> _Response:
        requests.append((url, timeout))
        if open_error is not None:
            raise open_error
        return response

    with monkeypatch.context() as patch:
        patch.setattr(urllib.request, "urlopen", urlopen)
        patch.setattr(sys, "argv", ["-c", json.dumps({"metric": METRIC, "port": 8080})])
        patch.setattr(sys, "stdin", None)
        patch.setattr(builtins, "__import__", stdlib_import)
        exec(source, {"__name__": "__main__"})  # noqa: S102 - shipped probe program
    captured = capsys.readouterr()
    assert captured.err == "", "parse/read errors must never echo response data"
    assert requests == [("http://127.0.0.1:8080/metrics", 10)]
    if open_error is None:
        assert response.closed, (
            "close the response on success and on every read failure"
        )
        assert response.read_sizes == [MAX_RESPONSE_BYTES + 1], "bound the full read"
    report = json.loads(captured.out)
    assert isinstance(report, dict), "one JSON report, with no extra output"
    return report


def _release(report: object) -> ResourceRelease:
    release = ResourceRelease()
    for role in validation.CPU_METRIC_PORTS:
        release.documents[("cpu", "deployment", role)] = deployment(
            role, replicas=int(role == "gpu-fault-api-ha")
        )
    release.documents[("cpu", "pods", "")] = {
        "items": [{"metadata": {"name": "api-a"}}]
    }
    release.runner.handler = json_response(report)
    return release


@pytest.mark.parametrize(
    "slots", [(0,), (0, 1, 2, 3), (7,), (0, 2, 7, 15), tuple(range(16))]
)
@pytest.mark.parametrize("process_id", [None, 'worker, reason="other"\\path\nline'])
def test_complete_zero_view_and_consumer_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    slots: tuple[int, ...],
    process_id: str | None,
) -> None:
    report = _run_probe(monkeypatch, capsys, _view(slots, process_id=process_id))
    assert report == zero_view_report(process_count=len(slots))
    release = _release(report)

    result = validation.store_io_rejection_series_ready(release)

    assert result == {
        "ready": True,
        "pods": {"gpu-fault-api-ha/api-a": report},
        "errors": [],
    }
    arguments, kwargs = release.runner.calls[0]
    assert arguments[arguments.index("exec") : -2] == [
        "exec",
        "api-a",
        "--",
        CONTROL_PLANE_PYTHON,
        "-c",
    ]
    assert json.loads(arguments[-1]) == {"metric": METRIC, "port": 8080}
    assert kwargs == {"capture": True}, "no stdin/input_text or interactive exec"


@pytest.mark.parametrize("family", [PROCESSES, DEGRADED])
def test_missing_or_duplicate_merger_metadata_cannot_prove_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], family: str
) -> None:
    lines = _view()
    metadata = next(line for line in lines if line.startswith(family + " "))
    for payload in (
        [line for line in lines if line != metadata],
        [*lines, metadata],
        [*lines, f"{family} 2"],
        [*lines, f'{family}{{process="0"}} 0'],
    ):
        report = _run_probe(monkeypatch, capsys, payload)
        assert report["complete"] is False
        assert (
            validation.store_io_rejection_series_ready(_release(report))["ready"]
            is False
        )


@pytest.mark.parametrize(
    "count",
    ["0", "-1", "17", "1.5", "1.00000000000000001", "NaN", "Inf", "true", "", "1 123"],
)
def test_invalid_publication_count_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], count: str
) -> None:
    lines = _view()
    lines[0] = f"{PROCESSES} {count}"
    assert _run_probe(monkeypatch, capsys, lines)["complete"] is False


@pytest.mark.parametrize(
    "degraded", ["1", "2", "-1", "NaN", "Inf", "false", "", "0 123", "1e-9999"]
)
def test_stale_future_or_unknown_publications_cannot_prove_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], degraded: str
) -> None:
    lines = _view((0, 2, 7, 15))
    lines[1] = f"{DEGRADED} {degraded}"
    assert _run_probe(monkeypatch, capsys, lines)["complete"] is False


@pytest.mark.parametrize("slots,count", [((0,), 4), ((0, 1, 2), 4), ((0, 1, 2, 3), 3)])
def test_process_slots_must_match_the_publication_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    slots: tuple[int, ...],
    count: int,
) -> None:
    lines = _view(slots)
    lines[0] = f"{PROCESSES} {count}"
    report = _run_probe(monkeypatch, capsys, lines)
    assert report["complete"] is False
    assert report["all_labeled"] is report["all_zero"] is True


@pytest.mark.parametrize("slot", [0, 7])
@pytest.mark.parametrize("reason", REASONS)
def test_every_slot_needs_every_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    slot: int,
    reason: str,
) -> None:
    lines = _view((0, 7))
    lines.remove(_counter(slot, reason))
    report = _run_probe(monkeypatch, capsys, lines)
    assert report["complete"] is False
    assert report["series_count"] == 5
    assert report["all_labeled"] is report["all_zero"] is True


@pytest.mark.parametrize(
    "sample",
    [
        "",
        " 0",
        '{reason="capacity"} 0',
        '{process_id="worker",reason="capacity"} 0',
        '{process="0"} 0',
        '{process="0",not_reason="capacity"} 0',
        '{process="0",process_id="reason=\\"capacity\\""} 0',
        '{process="0",reason="unknown"} 0',
        '{process="0",reason="capacity",extra="unknown"} 0',
        '{process="0",process="0",reason="capacity"} 0',
        '{process="0",process="1",reason="capacity"} 0',
        '{process="0",reason="capacity",reason="deadline"} 0',
        '{process="0",reason="capacity",reason="capacity"} 0',
        '{process="0",reason="capacity",process_id="a",process_id="b"} 0',
        '{process="0",reason=capacity} 0',
        "{process='0',reason=\"capacity\"} 0",
        '{process="0",reason="capacity" trailing="value"} 0',
        '{process="0",reason="capacity",} 0',
        '{process="0",reason="capacity"}junk 0',
        '{process="0",reason="capacity} 0',
        '{process="0",reason="capacity",process_id="bad\\q"} 0',
        '{process="0",reason="capacity",process_id="bad\\t"} 0',
        '{process="\\u0030",reason="capacity"} 0',
    ],
)
def test_legacy_malformed_unknown_and_conflicting_labels_are_not_proof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sample: str
) -> None:
    lines = _view()
    lines[2] = METRIC + sample
    assert _run_probe(monkeypatch, capsys, lines)["complete"] is False


@pytest.mark.parametrize("slot", ["-1", "16", "01", "+1", "1.0", "nan", "", " 0"])
def test_only_canonical_bounded_process_slots_are_accepted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], slot: str
) -> None:
    lines = _view()
    lines[2] = _counter(slot)
    assert _run_probe(monkeypatch, capsys, lines)["complete"] is False


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("value,process_id", [("0", None), ("1", None), ("0", "other")])
def test_duplicate_slot_reason_is_rejected_even_with_different_process_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    value: str,
    process_id: str | None,
) -> None:
    lines = [*_view((0, 7)), _counter(7, reason, value, process_id=process_id)]
    assert _run_probe(monkeypatch, capsys, lines)["complete"] is False


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "nan",
        "+Inf",
        "-Inf",
        "Infinity",
        "1e309",
        "-1",
        "-1e-9999",
        "example-private-metric-value",
        "",
        "0 123",
        "0 # exemplar",
        "0.0.0",
        "0_0",
        "0x0",
        "true",
        '"0"',
    ],
)
def test_nonfinite_negative_and_malformed_counter_values_are_not_proof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    lines = _view()
    lines[2] = _counter(value=value)
    report = _run_probe(monkeypatch, capsys, lines)
    assert report == zero_view_report(
        process_count=0, complete=False, all_labeled=False, all_zero=False
    )
    assert "example-private-metric-value" not in json.dumps(report)


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("value", ["1", "0.5", "1e-9999"])
def test_any_nonzero_counter_refuses_settling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    value: str,
) -> None:
    lines = _view((0, 7))
    lines[lines.index(_counter(7, reason))] = _counter(7, reason, value)
    report = _run_probe(monkeypatch, capsys, lines)
    assert report == zero_view_report(process_count=2, all_zero=False)
    assert (
        validation.store_io_rejection_series_ready(_release(report))["ready"] is False
    )


@pytest.mark.parametrize("value", ["0", "+0", "-0", "0.0", ".0", "0.", "0e-9999"])
def test_finite_zero_number_forms_are_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    lines = _view()
    lines[0] = f"{PROCESSES} 1.0"
    lines[1] = f"{DEGRADED} {value}"
    lines[2] = _counter(value=value)
    assert _run_probe(monkeypatch, capsys, lines) == zero_view_report()


@pytest.mark.parametrize("lines", [[], [f"{PROCESSES} 1", f"{DEGRADED} 0"]])
def test_no_counter_samples_is_not_a_zero_view(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lines: list[str],
) -> None:
    report = _run_probe(monkeypatch, capsys, lines)
    assert report["complete"] is report["all_zero"] is report["all_labeled"] is False


def test_comments_other_families_and_quoted_braces_do_not_confuse_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = [
        f"# HELP {METRIC} Store I/O rejection counter",
        f"# TYPE {METRIC} counter",
        f"{METRIC}_unrelated 99",
        f"{PROCESSES}_unrelated NaN",
        *_view(process_id='worker } { , "quoted"'),
    ]
    lines[4:] = [line.replace(",", ", ").replace("=", " = ") for line in lines[4:]]
    assert _run_probe(monkeypatch, capsys, lines) == zero_view_report()


@pytest.mark.parametrize("extra", [0, 1])
def test_http_response_size_limit_never_accepts_a_truncated_zero_prefix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: int
) -> None:
    prefix = ("\n".join(_view()) + "\n#").encode()
    body = prefix + b"x" * (MAX_RESPONSE_BYTES + extra - len(prefix))
    report = _run_probe(monkeypatch, capsys, body)
    assert report["complete"] is (extra == 0)


@pytest.mark.parametrize("failure", ["read", "open", "decode"])
def test_http_failures_are_data_free_and_close_acquired_responses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    marker = "example-private-response-detail"
    body = ("\n".join(_view()) + "\n").encode()
    report = _run_probe(
        monkeypatch,
        capsys,
        body + (b"\xff" if failure == "decode" else b""),
        read_error=OSError(marker) if failure == "read" else None,
        open_error=urllib.error.URLError(marker) if failure == "open" else None,
    )
    assert report == zero_view_report(
        process_count=0, complete=False, all_labeled=False, all_zero=False
    )
    assert marker not in json.dumps(report)


def test_http_error_body_is_closed_without_disclosing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    body = io.BytesIO(b"example-private-http-error-body")
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8080/metrics",
        503,
        "example-private-http-error-detail",
        Message(),
        body,
    )
    report = _run_probe(monkeypatch, capsys, _view(), open_error=error)
    assert body.closed, "HTTPError owns a response even though urlopen raised"
    assert report == zero_view_report(
        process_count=0, complete=False, all_labeled=False, all_zero=False
    )


@pytest.mark.parametrize("field", ["complete", "all_labeled", "all_zero"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(MISSING, id="missing"),
        False,
        None,
        0,
        1,
        "true",
        "false",
        [],
        {},
        [True],
    ],
)
def test_consumer_requires_exact_true_for_every_proof_boolean(
    field: str, value: object
) -> None:
    report: dict[str, object] = dict(zero_view_report())
    if value is MISSING:
        report.pop(field)
    else:
        report[field] = value
    result = validation.store_io_rejection_series_ready(_release(report))
    assert result["ready"] is False
    assert len(result["errors"]) == 1, "each flag independently rejects the report"


@pytest.mark.parametrize("report", [None, [], True, "", 1])
def test_consumer_rejects_non_object_reports(report: object) -> None:
    result = validation.store_io_rejection_series_ready(_release(report))
    assert result == {
        "ready": False,
        "pods": {},
        "errors": [
            "gpu-fault-api-ha/api-a returned an invalid Store I/O metric report"
        ],
    }


def test_consumer_cannot_authorize_settling_without_any_pod_evidence() -> None:
    release = _release(zero_view_report())
    release.documents[("cpu", "deployment", "gpu-fault-api-ha")]["spec"]["replicas"] = 0
    assert validation.store_io_rejection_series_ready(release)["ready"] is False
    assert release.runner.calls == []


def test_consumer_refuses_an_old_report_even_when_observed_series_are_zero() -> None:
    report = {"series_count": 3, "all_labeled": True, "all_zero": True}
    result = validation.store_io_rejection_series_ready(_release(report))
    assert result["ready"] is False
    assert result["errors"] == [
        "gpu-fault-api-ha/api-a has incomplete Store I/O metric coverage"
    ]
