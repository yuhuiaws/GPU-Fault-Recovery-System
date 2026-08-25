from __future__ import annotations

from gpu_fault.diagnostics import KubernetesDcgmDiagnosticAdapter
from gpu_fault.models import DiagnosticRequest, TriageOutcome
from tests._builders import build_store


class CoreApi:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready

    def read_node(self, _node_id):
        return {
            "status": {
                "addresses": [{"type": "InternalIP", "address": "10.0.0.1"}],
                "conditions": [
                    {"type": "Ready", "status": "True" if self.ready else "False"},
                    {"type": "MemoryPressure", "status": "False"},
                    {"type": "DiskPressure", "status": "False"},
                    {"type": "PIDPressure", "status": "False"},
                    {"type": "NetworkUnavailable", "status": "False"},
                ],
            }
        }


def request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="diag-1",
        attempt_id="attempt-1",
        node_ids=["node-a"],
        checks=["gpu-enumeration", "dcgm-passive-health"],
    )


def test_healthy_node_and_dcgm_metrics_pass() -> None:
    adapter = KubernetesDcgmDiagnosticAdapter(
        build_store(),
        CoreApi(),
        fetcher=lambda _url, _timeout: (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 42\n'
            'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0"} 0\n'
            'DCGM_FI_DEV_RETIRED_PENDING{gpu="0"} 0\n'
        ),
    )

    adapter.submit(request())
    report = adapter.result("diag-1")

    assert report.findings[0].outcome is TriageOutcome.PASS


def test_pending_retired_page_proposes_gpu_reset() -> None:
    adapter = KubernetesDcgmDiagnosticAdapter(
        build_store(),
        CoreApi(),
        fetcher=lambda _url, _timeout: (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 42\nDCGM_FI_DEV_RETIRED_PENDING{gpu="0"} 1\n'
        ),
    )

    adapter.submit(request())
    finding = adapter.result("diag-1").findings[0]

    assert finding.outcome is TriageOutcome.FAIL
    assert finding.proposed_action.value == "RESET_GPU"
    assert "DCGM_FI_DEV_RETIRED_PENDING" in finding.failed_checks


def test_row_remap_failure_requires_drain_and_diagnostics() -> None:
    adapter = KubernetesDcgmDiagnosticAdapter(
        build_store(),
        CoreApi(),
        fetcher=lambda _url, _timeout: (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 42\n'
            'DCGM_FI_DEV_ROW_REMAP_FAILURE{gpu="0"} 1\n'
        ),
    )

    adapter.submit(request())
    finding = adapter.result("diag-1").findings[0]

    assert finding.outcome is TriageOutcome.FAIL
    assert finding.proposed_action.value == "DRAIN"
    assert "DCGM_FI_DEV_ROW_REMAP_FAILURE" in (finding.failed_checks)


def test_unreachable_dcgm_is_inconclusive() -> None:
    def fail(_url, _timeout):
        raise TimeoutError("scrape timed out")

    adapter = KubernetesDcgmDiagnosticAdapter(build_store(), CoreApi(), fetcher=fail)

    adapter.submit(request())
    finding = adapter.result("diag-1").findings[0]

    assert finding.outcome is TriageOutcome.INCONCLUSIVE
    assert finding.failed_checks == ["dcgm-passive-health"]
