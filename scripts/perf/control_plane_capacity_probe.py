from __future__ import annotations

import argparse
import json
import math
import ssl
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(
    values: list[float],
    quantile: float,
) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def request_once(
    url: str,
    *,
    timeout: float,
    ssl_context: ssl.SSLContext,
) -> tuple[int | None, float, str | None]:
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gpu-fault-capacity-probe/1"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl_context,
        ) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    except (OSError, TimeoutError) as exc:
        return (
            None,
            time.perf_counter() - started,
            type(exc).__name__,
        )
    return status, time.perf_counter() - started, None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=("Run a bounded, read-only NLB/API/Aurora capacity probe.")
    )
    result.add_argument("--url", required=True)
    result.add_argument("--path", default="/metrics")
    result.add_argument("--requests", type=int, default=1000)
    result.add_argument("--concurrency", type=int, default=32)
    result.add_argument("--timeout-seconds", type=float, default=10)
    result.add_argument("--ca-certificate")
    result.add_argument("--max-error-rate", type=float, default=0)
    result.add_argument("--max-p95-seconds", type=float, default=5)
    result.add_argument("--min-requests-per-second", type=float, default=0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be positive")
    if not 0 <= args.max_error_rate <= 1:
        raise SystemExit("max error rate must be between 0 and 1")

    base = args.url.rstrip("/") + "/"
    url = urllib.parse.urljoin(base, args.path.lstrip("/"))
    ssl_context = ssl.create_default_context(cafile=args.ca_certificate)
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    transport_errors = 0
    transport_error_types: dict[str, int] = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix="capacity-probe",
    ) as executor:
        futures = [
            executor.submit(
                request_once,
                url,
                timeout=args.timeout_seconds,
                ssl_context=ssl_context,
            )
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            status, latency, transport_error = future.result()
            latencies.append(latency)
            if transport_error is not None:
                transport_errors += 1
                transport_error_types[transport_error] = (
                    transport_error_types.get(transport_error, 0) + 1
                )
                continue
            if status is None:
                raise RuntimeError(
                    "capacity probe returned neither status nor transport error"
                )
            statuses[status] = statuses.get(status, 0) + 1

    elapsed = time.perf_counter() - started
    http_errors = sum(count for status, count in statuses.items() if status >= 400)
    errors = http_errors + transport_errors
    error_rate = errors / args.requests
    requests_per_second = args.requests / elapsed
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    summary = {
        "url": url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 6),
        "requests_per_second": round(requests_per_second, 3),
        "status_counts": {str(key): statuses[key] for key in sorted(statuses)},
        "transport_errors": transport_errors,
        "transport_error_types": dict(sorted(transport_error_types.items())),
        "error_rate": round(error_rate, 6),
        "latency_seconds": {
            "mean": (round(statistics.fmean(latencies), 6) if latencies else None),
            "p50": round(p50, 6) if p50 is not None else None,
            "p95": round(p95, 6) if p95 is not None else None,
            "p99": round(p99, 6) if p99 is not None else None,
            "max": round(max(latencies), 6) if latencies else None,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    return int(
        error_rate > args.max_error_rate
        or p95 is None
        or p95 > args.max_p95_seconds
        or requests_per_second < args.min_requests_per_second
    )


if __name__ == "__main__":
    raise SystemExit(main())
