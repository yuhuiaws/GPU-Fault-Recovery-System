from __future__ import annotations

import threading

from gpu_fault.admin.cluster_join_readonly import (
    cached_network_baseline,
    parallel_verify_and_discover,
)
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


def test_single_join_verification_and_discovery_overlap() -> None:
    verify_started = threading.Event()
    discover_started = threading.Event()

    def verify() -> None:
        verify_started.set()
        assert discover_started.wait(timeout=2), "join discovery did not start"

    def discover() -> str:
        discover_started.set()
        assert verify_started.wait(timeout=2), "baseline verification did not start"
        return "gpu-b"

    assert parallel_verify_and_discover(verify, discover) == "gpu-b"


def test_single_join_network_baseline_cache_is_site_bound(tmp_path) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    cache = tmp_path / "network-baseline.json"
    calls = []

    first = cached_network_baseline(
        cache, site, lambda: calls.append("discover") or [{"vpc_id": "vpc-a"}]
    )
    second = cached_network_baseline(
        cache, site, lambda: calls.append("unexpected") or []
    )

    assert first == second == [{"vpc_id": "vpc-a"}]
    assert calls == ["discover"]

    cache.touch()
    document = path.read_text(encoding="utf-8")
    path.write_text(
        document.replace("test-site", "test-site-updated"), encoding="utf-8"
    )
    path.chmod(0o600)
    changed = load_site(path)
    third = cached_network_baseline(
        cache, changed, lambda: calls.append("rediscover") or [{"vpc_id": "vpc-b"}]
    )

    assert third == [{"vpc_id": "vpc-b"}]
    assert calls == ["discover", "rediscover"]
