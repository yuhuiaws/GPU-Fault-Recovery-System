"""ISO-006 judges A's claim latency on claims its probe's own container did not stall.

Live (HyperPod as cluster A, four attempts on clean sites): every A claim sample
was a 13-29 ms fast mode plus, in half or more of them, one or two stalls of
20-75 ms -- in the baseline with cluster B still talking and under the cut
alike. The probe runs inside the executor container (CPU limit 500m); that
container's CFS counters showed about half of its active periods throttled for
38-48 ms each, cluster B's in-cut samples (DNS and a locally rejected connect,
never reaching the control plane) carried the same modes, and the control
plane's store-I/O admission and pool checkout waits stayed under 0.1 ms. The
median of 10 against 19 such samples moved -16, +9, +19 and +15 ms with nothing
degraded, and two attempts failed on it alone ("A claim p50 rose from 0.018s to
0.038s", "... from 0.026s to 0.040s").
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import run_iso006_cluster_offline as iso006
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A
from tests.regional.test_cov95_identity_audit import bound_api, probe

#: A's claim latencies (seconds) of the four HyperPod attempts: baseline (ten
#: samples, 30 s apart, B still talking) and window (B cut off).
LIVE_ATTEMPTS = {
    "mixed-site attempt 8 (PASS)": (
        [0.021, 0.0617, 0.0288, 0.0839, 0.0505, 0.0761, 0.0538, 0.0661, 0.0713, 0.02],
        [0.0681, 0.0215, 0.0417, 0.0424, 0.038, 0.0167, 0.0516, 0.0712, 0.0589]
        + [0.0357, 0.0626, 0.0209, 0.0186, 0.0203, 0.0591, 0.0202, 0.0182, 0.016]
        + [0.0554],
    ),
    "eks-primary attempt 2 (PASS by 0.6 ms)": (
        [0.0213, 0.0213, 0.0392, 0.0343, 0.0148, 0.0636, 0.0327, 0.0157, 0.0564]
        + [0.0567],
        [0.0421, 0.0174, 0.0472, 0.053, 0.0476, 0.0525, 0.0151, 0.0633, 0.0336]
        + [0.0207, 0.0541, 0.0201, 0.0301, 0.0476, 0.0209, 0.0647, 0.0233, 0.0907]
        + [0.0158],
    ),
    "both-providers attempt 1 (FAIL: p50 0.018 -> 0.038)": (
        [0.0652, 0.0169, 0.0272, 0.0209, 0.0733, 0.0158, 0.0146, 0.0182, 0.0481]
        + [0.0168],
        [0.0198, 0.0196, 0.0684, 0.0197, 0.0708, 0.0581, 0.0376, 0.0196, 0.0194]
        + [0.0696, 0.016, 0.0384, 0.0152, 0.0736, 0.0549, 0.0489, 0.0195, 0.0312]
        + [0.0928],
    ),
    "both-providers attempt 2 (FAIL: p50 0.026 -> 0.040)": (
        [0.0219, 0.0477, 0.0358, 0.0257, 0.0708, 0.0135, 0.0228, 0.0194, 0.0435]
        + [0.0414],
        [0.0654, 0.0373, 0.0379, 0.0589, 0.0201, 0.0669, 0.0361, 0.0197, 0.0443]
        + [0.0384, 0.0405, 0.0706, 0.0169, 0.0169, 0.0289, 0.0289, 0.0703, 0.0597]
        + [0.0573, 0.0598],
    ),
}
FAILED_ATTEMPTS = [name for name in LIVE_ATTEMPTS if "FAIL" in name]


def claims(*latencies: float, throttled: bool | None = None) -> list[dict[str, Any]]:
    sample: dict[str, Any] = {"status": 200, "command_count": 0}
    if throttled is not None:
        sample["client_cpu_throttled"] = throttled
    return [{**sample, "latency_seconds": value} for value in latencies]


def prefixes(errors: list[str]) -> list[str]:
    return [item.split(" rose ")[0] for item in errors]


@pytest.mark.parametrize("attempt", sorted(LIVE_ATTEMPTS))
def test_the_live_attempts_pass_on_the_fastest_claim_and_the_p95(attempt: str) -> None:
    baseline, window = (claims(*values) for values in LIVE_ATTEMPTS[attempt])

    assert iso006.latency_errors(baseline, window) == []
    assert iso006.median_is_judged(baseline, window) is False, (
        "samples without a throttle reading cannot carry a median judgment"
    )


@pytest.mark.parametrize("attempt", FAILED_ATTEMPTS)
@pytest.mark.parametrize(
    ("degrade", "expected"),
    [
        pytest.param(
            lambda index, value: value + 0.015,
            ["A claim base latency"],
            id="every-claim-15ms-slower",
        ),
        pytest.param(
            lambda index, value: value * 2,
            ["A claim base latency"],
            id="every-claim-twice-as-slow",
        ),
        pytest.param(
            lambda index, value: value + 0.150 if index % 5 == 0 else value,
            ["A claim p95"],
            id="one-claim-in-five-150ms-slower",
        ),
    ],
)
def test_a_degraded_control_plane_still_fails_on_the_live_background(
    attempt: str, degrade: Any, expected: list[str]
) -> None:
    baseline, window = LIVE_ATTEMPTS[attempt]
    degraded = claims(*(degrade(index, value) for index, value in enumerate(window)))

    assert prefixes(iso006.latency_errors(claims(*baseline), degraded)) == expected


def test_a_5xx_under_the_cut_fails_whatever_the_latency() -> None:
    baseline, window = LIVE_ATTEMPTS[FAILED_ATTEMPTS[0]]
    errors = iso006.latency_errors(
        claims(*baseline), [*claims(*window), {"status": 503, "latency_seconds": 0.02}]
    )

    assert errors == ["A claim returned 503 during the block"]


#: The fast mode of the live samples: what the claim path costs when the
#: probe's container is not throttled while the claim is timed.
FAST = [0.0146, 0.0158, 0.0168, 0.0169, 0.0182, 0.0196, 0.0201, 0.0209, 0.0219]
STALLED = [0.0481, 0.0652, 0.0708, 0.0733, 0.0928]


def test_claims_proven_unthrottled_judge_the_median_too() -> None:
    baseline = claims(*FAST, 0.0228, throttled=False)
    window = claims(*FAST, *FAST, 0.0199, throttled=False)
    assert iso006.median_is_judged(baseline, window) is True
    assert iso006.latency_errors(baseline, window) == []

    # Most claims, not all: the fastest stays, the p95 moves below its floor;
    # only the median of claims proven unthrottled can see it.
    most_slower = claims(
        *FAST[:3],
        *(value + 0.030 for value in [*FAST[3:], *FAST, 0.0199]),
        throttled=False,
    )
    assert prefixes(iso006.latency_errors(baseline, most_slower)) == ["A claim p50"]
    assert (
        iso006.latency_errors(
            claims(*FAST, 0.0228), claims(*FAST[:3], *(v + 0.030 for v in FAST))
        )
        == []
    ), "without readings the same shift is left to the fastest claim and p95"


def test_stalls_of_the_probe_container_are_left_out_not_judged() -> None:
    baseline = [
        *claims(*FAST[:7], throttled=False),
        *claims(*STALLED[:3], throttled=True),
    ]
    window = [
        *claims(*FAST, *FAST[:5], throttled=False),
        *claims(*STALLED, throttled=True),
    ]

    assert iso006.latency_errors(baseline, window) == []
    summary = iso006.latency_summary(window)
    assert summary["judged"] == 14 and summary["client_cpu_throttled"] == 5
    assert summary["p95_seconds"] < min(STALLED), "a stalled claim reached the p95"


@pytest.mark.parametrize("phase", ["baseline", "block"])
def test_a_phase_its_probe_container_mostly_stalled_is_not_judged(phase: str) -> None:
    clean = claims(*FAST, 0.0199, throttled=False)
    stalled = [
        *claims(*FAST[:4], throttled=False),
        *claims(*STALLED, 0.0611, throttled=True),
    ]
    baseline, window = (stalled, clean) if phase == "baseline" else (clean, stalled)

    assert iso006.latency_errors(baseline, window) == [
        f"A claim latency could not be judged during the {phase}: 6 of 10 claims "
        "were stalled by the probe container's own CPU quota"
    ]


def test_the_evidence_names_what_each_phase_rests_on() -> None:
    baseline, window = LIVE_ATTEMPTS[FAILED_ATTEMPTS[1]]
    result: dict[str, Any] = {
        "baseline_samples": claims(*baseline),
        "validation_limitations": list(iso006.VALIDATION_LIMITATIONS),
    }
    iso006.record_claim_latency(result, claims(*window))

    assert result["a_claim_latency"]["median_judged"] is False
    assert result["a_claim_latency"]["block"]["client_cpu_throttle_unread"] == 20
    assert result["a_claim_base_baseline_seconds"] == min(baseline)
    assert result["a_claim_base_window_seconds"] == min(window)
    assert result["validation_limitations"][-1] == iso006.MEDIAN_UNJUDGED
    assert iso006.MEDIAN_UNJUDGED not in iso006.VALIDATION_LIMITATIONS, (
        "the module constant is not mutated"
    )

    proven: dict[str, Any] = {"baseline_samples": claims(*FAST, throttled=False)}
    iso006.record_claim_latency(proven, claims(*FAST, throttled=False))
    assert proven["a_claim_latency"]["median_judged"] is True
    assert "validation_limitations" not in proven


def throttling_reader() -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(common.CLIENT_CPU_THROTTLING_SOURCE, "<cpu-stat>", "exec"), namespace)
    return namespace["client_cpu_throttling"]


def test_the_probe_reads_its_containers_cfs_counters_on_cgroup_v2_and_v1(
    tmp_path: Path,
) -> None:
    read = throttling_reader()
    v2 = tmp_path / "cpu.stat"
    v2.write_text(
        "usage_usec 9\nnr_periods 40\nnr_throttled 3\nthrottled_usec 150000\n",
        encoding="ascii",
    )
    v1 = tmp_path / "cpu-v1.stat"
    v1.write_text(
        "nr_periods 40\nnr_throttled 2\nthrottled_time 5000000\n", encoding="ascii"
    )
    broken = tmp_path / "broken.stat"
    broken.write_text("nr_throttled many\n", encoding="ascii")
    missing = tmp_path / "missing.stat"

    assert read(((str(v2), "throttled_usec", 1e6),)) == (3, 0.15)
    assert read(
        ((str(missing), "throttled_usec", 1e6), (str(v1), "throttled_time", 1e9))
    ) == (2, 0.005)
    assert read(((str(broken), "throttled_usec", 1e6),)) is None
    assert read(((str(missing), "throttled_usec", 1e6),)) is None


@pytest.mark.parametrize(
    ("stall", "expected"),
    [(True, (True, 0.08)), (False, (False, 0.0)), (None, (None, None))],
    ids=["stalled", "clean", "unreadable"],
)
def test_the_latency_probe_settles_then_flags_a_stall_of_its_own_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stall: bool | None,
    expected: tuple[bool | None, float | None],
) -> None:
    _context, bridge = bound_api(monkeypatch)
    for name, value in {
        "GPU_FAULT_CONTROL_PLANE_URL": "https://unit.invalid",
        "GPU_FAULT_CONTROL_PLANE_CA_FILE": str(tmp_path / "public-ca"),
        "GPU_FAULT_CONTROL_PLANE_TOKEN": TOKEN_A,
        "GPU_FAULT_CLUSTER_ID": "a",
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    stat = tmp_path / "cpu.stat"
    if stall is not None:
        stat.write_text(
            "nr_periods 9\nnr_throttled 3\nthrottled_usec 150000\n", encoding="ascii"
        )
    script = common.CLAIM_LATENCY_PROBE.replace(
        repr(common.CLIENT_CPU_STAT_SOURCES),
        repr(((str(stat), "throttled_usec", 1e6),)),
    )
    assert script != common.CLAIM_LATENCY_PROBE, "the probe names its cpu.stat sources"
    events: list[Any] = []

    def claim(request: urllib.request.Request, **kwargs: Any) -> Any:
        events.append("claim")
        if stall:
            stat.write_text(
                "nr_periods 9\nnr_throttled 4\nthrottled_usec 230000\n",
                encoding="ascii",
            )
        return bridge.urlopen(request, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", claim)
    monkeypatch.setattr(time, "sleep", lambda seconds: events.append(seconds))

    result = probe(script)

    assert result["status"] == 200 and result["command_count"] == 0
    assert events == [common.CLAIM_LATENCY_SETTLE_SECONDS, "claim"], (
        "the claim was timed before the probe settled"
    )
    throttled, seconds = expected
    assert result["client_cpu_throttled"] is throttled
    assert result["client_cpu_throttled_seconds"] == (
        None if seconds is None else pytest.approx(seconds)
    )


def test_the_window_samples_a_with_the_latency_probe() -> None:
    seen: list[str] = []

    def executor_python(script: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(script)
        return {"status": 200}

    fixture: Any = SimpleNamespace(executor_python=executor_python)
    common.claim_sample(fixture)

    assert seen == [common.CLAIM_LATENCY_PROBE]
