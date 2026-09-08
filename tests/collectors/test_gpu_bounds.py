"""The GPU collectors cannot be wedged by a child in uninterruptible sleep.

Task 11 bounded the host collector's ``nvidia-smi`` calls with
``BoundedProcessRunner``; the DCGM and nvidia-smi collectors and the discovery
helpers kept ``subprocess.run`` as their default. On a GPU that fell off the bus
``nvidia-smi`` goes D-state, ``run(timeout=15)`` kills it and then ``wait()``s
for ever, ``TimeoutExpired`` never arrives, every guard in ``collect_once`` is
bypassed and the metrics unit -- ``Type=simple``, no watchdog -- is dead but
not restarted on exactly the node that matters.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterator

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu import discovery

from ._support import (
    NOW,
    DcgmMetricsCollector,
    NvidiaSmiMetricsCollector,
    RecordingSink,
    context,
)


class _WedgedPopen:
    """A ``Popen`` whose child is in uninterruptible sleep in the driver.

    ``communicate`` times out at once, ``kill()`` is delivered but not acted
    on, and ``wait()`` blocks until the test releases it -- which is exactly why
    ``subprocess.run``'s own timeout handling (kill, then a *blocking* wait)
    never returned. It is also a context manager because ``subprocess.run``
    uses ``Popen`` as one, and its ``__exit__`` waits too.
    """

    def __init__(self, argv: list[str], released: threading.Event) -> None:
        self.args = list(argv)
        self.pid = 424242
        self.returncode = None
        self.stdout = None
        self.stderr = None
        self.stdin = None
        self.kills = 0
        self._released = released

    def __enter__(self) -> _WedgedPopen:
        return self

    def __exit__(self, *_args) -> None:
        self.wait()

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        raise subprocess.TimeoutExpired(self.args, timeout or 0)

    def kill(self) -> None:
        self.kills += 1

    def wait(self, timeout: float | None = None) -> int:
        if self._released.wait(timeout):
            self.returncode = -9
            return -9
        raise subprocess.TimeoutExpired(self.args, timeout or 0)


class _WedgedDriver:
    """Every ``Popen`` is a wedged child; all of them are released on teardown."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.children: list[_WedgedPopen] = []

    def __call__(self, argv, **_kwargs) -> _WedgedPopen:
        child = _WedgedPopen(argv, self.released)
        self.children.append(child)
        return child


@pytest.fixture
def wedged_driver(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[_WedgedDriver]:
    """``subprocess.Popen`` is the wedged driver for the whole test.

    Patched at the call site the runner actually uses -- ``subprocess.Popen``
    looked up when the call happens -- so a collector built with *no* runner
    argument reaches it, which is the configuration the installer ships.
    """

    driver = _WedgedDriver()
    monkeypatch.setattr(subprocess, "Popen", driver)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    try:
        yield driver
    finally:
        driver.released.set()


class _ExporterResponse:
    def __enter__(self) -> _ExporterResponse:
        return self

    def __exit__(self, *_args) -> bool:
        return False

    def read(self) -> bytes:
        return b'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 80\n'


def _call_with_deadline(target, *, seconds: float) -> tuple[bool, list]:
    """Run ``target`` on a thread; say whether it returned inside ``seconds``.

    A collector that hangs must fail the test rather than hang it: the red
    version of these tests is the hang itself.
    """

    outcome: list = []

    def run() -> None:
        try:
            outcome.append(("returned", target()))
        except BaseException as exc:  # noqa: BLE001 -- the verdict is the test's
            outcome.append(("raised", exc))

    thread = threading.Thread(target=run, name="gpu-collector-under-test", daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    return not thread.is_alive(), outcome


def test_dcgm_collect_once_returns_when_nvidia_smi_is_in_d_state(
    monkeypatch: pytest.MonkeyPatch, wedged_driver: _WedgedDriver
) -> None:
    """Inventory and temperature-limit probes both wedge; the scrape still ships.

    The exporter answers over HTTP and does not need the driver call to return,
    so the only thing between the node and a GPU_METRICS batch is the bound on
    the two ``nvidia-smi`` subprocesses ``collect_once`` runs first.
    """

    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen",
        lambda *_args, **_kwargs: _ExporterResponse(),
    )
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", now=lambda: NOW, interval_seconds=15
    )

    started = time.monotonic()
    returned, outcome = _call_with_deadline(collector.collect_once, seconds=8)
    elapsed = time.monotonic() - started

    assert returned, (
        "DcgmMetricsCollector.collect_once did not return within 8s with "
        "nvidia-smi in D-state: the default runner waits on a child that "
        f"ignores SIGKILL (children started: {len(wedged_driver.children)})"
    )
    assert outcome and outcome[0][0] == "returned", (
        f"collect_once raised instead of shipping the scrape: {outcome}"
    )
    assert wedged_driver.children, "collect_once never reached nvidia-smi at all"
    assert all(child.kills == 1 for child in wedged_driver.children), (
        "a wedged nvidia-smi child was never killed"
    )
    assert elapsed < 8, f"collect_once took {elapsed:.1f}s against a wedged driver"
    assert [path for path, _payload in sink.requests] == [GPU_METRICS_PATH], (
        f"the DCGM scrape was not delivered around the wedged probes: {sink.requests}"
    )


def test_nvidia_smi_collect_once_returns_when_nvidia_smi_is_in_d_state(
    wedged_driver: _WedgedDriver,
) -> None:
    """Fallback mode has nothing but nvidia-smi, so the round fails -- promptly.

    ``run()`` turns the failure into an error batch and backs off; what it
    cannot survive is ``collect_once`` never returning at all.
    """

    collector = NvidiaSmiMetricsCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW
    )

    started = time.monotonic()
    returned, outcome = _call_with_deadline(collector.collect_once, seconds=8)
    elapsed = time.monotonic() - started

    assert returned, (
        "NvidiaSmiMetricsCollector.collect_once did not return within 8s with "
        "nvidia-smi in D-state: the default runner waits on a child that "
        f"ignores SIGKILL (children started: {len(wedged_driver.children)})"
    )
    assert outcome and outcome[0][0] == "raised", (
        f"a round with no working nvidia-smi must fail, not ship: {outcome}"
    )
    assert isinstance(outcome[0][1], subprocess.TimeoutExpired), (
        f"the wedged merged query must surface as TimeoutExpired: {outcome}"
    )
    assert wedged_driver.children, "collect_once never reached nvidia-smi at all"
    assert all(child.kills == 1 for child in wedged_driver.children), (
        "a wedged nvidia-smi child was never killed"
    )
    assert elapsed < 8, f"collect_once took {elapsed:.1f}s against a wedged driver"


@pytest.mark.parametrize(
    "probe",
    [
        discovery.query_gpu_inventory,
        discovery.query_nvidia_temperature_limits,
        discovery.discover_gpu_product,
        discovery.discover_gpu_software_versions,
    ],
    ids=lambda probe: probe.__name__,
)
def test_discovery_helpers_default_to_a_bounded_runner(
    wedged_driver: _WedgedDriver, probe
) -> None:
    """Each discovery function's *default* runner must give up on a D-state child.

    ``context_from_environment`` passes a bounded runner explicitly (Task 11);
    these defaults are what the GPU collectors and every other direct caller
    get, and they were still ``subprocess.run``.
    """

    started = time.monotonic()
    returned, outcome = _call_with_deadline(probe, seconds=8)
    elapsed = time.monotonic() - started

    assert returned, (
        f"{probe.__name__} did not return within 8s with nvidia-smi in D-state: "
        "its default runner waits on a child that ignores SIGKILL"
    )
    assert outcome and outcome[0][0] == "raised", (
        f"{probe.__name__} must report the wedged driver as unavailable: {outcome}"
    )
    assert isinstance(outcome[0][1], discovery.CollectorError), (
        f"{probe.__name__} let the subprocess error escape as "
        f"{type(outcome[0][1]).__name__} instead of CollectorError"
    )
    assert wedged_driver.children and wedged_driver.children[0].kills == 1, (
        f"{probe.__name__}'s wedged child was never killed"
    )
    assert elapsed < 8, f"{probe.__name__} took {elapsed:.1f}s against a wedged driver"
