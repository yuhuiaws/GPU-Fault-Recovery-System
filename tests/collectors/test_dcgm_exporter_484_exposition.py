"""The collector reads what dcgm-exporter 4.8.4 serves.

4.8.4 (batch 135's pin) renders ``/metrics`` with the Prometheus ``expfmt``
encoder instead of the 4.5.2 text template: families sorted by name, each
with ``# HELP``/``# TYPE`` from the counters CSV, every value parsed to a float
and printed in Go's shortest ``'g'`` form -- so a large counter reads
``1.23456789012e+11`` -- and the host label renamed ``Hostname`` ->
``hostname`` (4.8.3, NVIDIA/dcgm-exporter#655). Field names stay the CSV's,
including the 22 deprecated aliases go-dcgm v1.4611.2 still resolves. The
collector must map every counters field from that text exactly as it did.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext

ROOT = Path(__file__).resolve().parents[2]
COUNTERS = ROOT / "deploy" / "dataplane" / "dcgm-counters.csv"
OBSERVED_AT = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
GPUS = (("0", "GPU-5d1c", "00000000:97:00.0"), ("1", "GPU-77ab", "00000000:A8:00.0"))
#: Values in the form the encoder prints them (Go strconv 'g', shortest).
VALUES = {
    "DCGM_FI_DEV_POWER_USAGE": "412.345",
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": "700",
    "DCGM_FI_DEV_FB_USED": "143290",
    "DCGM_FI_DEV_POWER_VIOLATION": "1.23456789012e+11",
    "DCGM_FI_DEV_THERMAL_VIOLATION": "0",
    "DCGM_FI_DEV_XID_ERRORS": "0",
    "DCGM_FI_DEV_FABRIC_MANAGER_STATUS": "3",
    "DCGM_FI_DEV_SM_CLOCK": "1.98e+03",
}


class _Sink:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, path: str, payload: dict) -> dict:
        self.payloads.append(payload)
        return {}


def _families() -> list[tuple[str, str, str]]:
    rows = []
    for line in COUNTERS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, kind, help_text = (field.strip() for field in line.split(","))
        rows.append((name, kind, help_text))
    return sorted(rows)


def exposition_484() -> str:
    lines = []
    for name, kind, help_text in _families():
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
        for index, uuid, bus in GPUS:
            labels = (
                f'gpu="{index}",UUID="{uuid}",pci_bus_id="{bus}",device="nvidia{index}",'
                'modelName="NVIDIA H200",hostname="ip-10-0-42-179.us-west-2.compute.internal"'
            )
            if name == "DCGM_FI_DEV_XID_ERRORS":
                labels += ',err_code="0",err_msg="No Error"'
            lines.append(f"{name}{{{labels}}} {VALUES.get(name, '1')}")
    return "\n".join(lines) + "\n"


def test_every_counters_field_maps_from_the_484_exposition() -> None:
    collector = DcgmMetricsCollector(
        _Sink(), CollectorContext(cluster_id="gpu-eks"), node_id="ip-10-0-42-179"
    )

    batch = collector.collect_text(exposition_484(), observed_at=OBSERVED_AT)

    names = [name for name, _kind, _help in _families()]
    assert set(names) <= set(DCGM_METRICS), "every field the CSV exports is mapped"
    expected = {DCGM_METRICS[name][0] for name in names}
    mapped = {sample.canonical_name for sample in batch.samples}
    assert expected <= mapped, (
        f"fields served but not collected: {sorted(expected - mapped)}"
    )
    assert batch.collection_errors == [], batch.collection_errors
    by_key = {
        (sample.gpu_uuid, sample.canonical_name): sample for sample in batch.samples
    }
    violation = by_key[("GPU-5d1c", DCGM_METRICS["DCGM_FI_DEV_POWER_VIOLATION"][0])]
    assert violation.value == 123456789.012, (
        "an exponent-form counter in nanoseconds is still read as microseconds"
    )
    sm_clock = by_key[("GPU-77ab", DCGM_METRICS["DCGM_FI_DEV_SM_CLOCK"][0])]
    assert sm_clock.value == 1980.0
    power = by_key[("GPU-5d1c", DCGM_METRICS["DCGM_FI_DEV_POWER_USAGE"][0])]
    assert power.value == 412.345
    sample = by_key[("GPU-5d1c", DCGM_METRICS["DCGM_FI_DEV_GPU_TEMP"][0])]
    assert (sample.gpu_index, sample.pci_bdf) == ("0", "00000000:97:00.0")
    assert sample.labels["hostname"].startswith("ip-10-0-42-179"), (
        "the renamed host label rides along; nothing reads Hostname"
    )
    assert "Hostname" not in sample.labels
