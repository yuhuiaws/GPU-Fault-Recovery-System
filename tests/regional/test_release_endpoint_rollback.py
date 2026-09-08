import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_endpoint_rollback as ENDPOINT
from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]

HOSTED_ZONE = "Z0EXAMPLE"
HOSTNAME = "control-plane.gpu-fault.internal"
PREVIOUS_TARGET = "gpu-fault-regional-previous.elb.us-west-2.amazonaws.com"
CANDIDATE_TARGET = "gpu-fault-regional-candidate.elb.us-west-2.amazonaws.com"


def _declared() -> tuple[dict[str, str], ...]:
    return ENDPOINT.declared_endpoint_objects(
        ENDPOINT.NLB_MANIFEST.read_text(encoding="utf-8")
    )


def _live_service() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": ENDPOINT.NLB_SERVICE,
            "namespace": "gpu-fault-system",
            "uid": "9f8e",
            "resourceVersion": "8123",
            "annotations": {
                "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": "arn:previous",
                "kubectl.kubernetes.io/last-applied-configuration": "{}",
            },
        },
        "spec": {"ports": [{"port": 443}]},
        "status": {"loadBalancer": {"ingress": [{"hostname": PREVIOUS_TARGET}]}},
    }


def _record(target: str) -> dict[str, Any]:
    return {
        "Name": f"{HOSTNAME}.",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": target}],
    }


class Release:
    """A release whose live endpoint state the test declares up front."""

    def __init__(
        self,
        *,
        service: dict[str, Any] | None,
        record: dict[str, Any] | None,
        nlb: dict[str, Any] | None = None,
        published: str = PREVIOUS_TARGET,
    ) -> None:
        self.config = SimpleNamespace(
            namespace="gpu-fault-system",
            aws_region="us-west-2",
            nlb={"name": "gpu-fault-regional"} if nlb is None else nlb,
            dns=SimpleNamespace(hosted_zone_id=HOSTED_ZONE, hostname=HOSTNAME),
        )
        self.service = service
        self.record = record
        self.published = published
        self.calls: list[list[str]] = []
        self.applied: list[Any] = []
        self.runner = SimpleNamespace(run=self._run, dry_run=False)

    @staticmethod
    def _cpu(*arguments: str) -> list[str]:
        return ["kubectl", "--kubeconfig", "/secure/cpu", *arguments]

    def _run(self, arguments: list[str], **_kwargs: Any) -> str:
        self.calls.append(list(arguments))
        if arguments[0] == "aws":
            return self._aws(arguments)
        if "apply" in arguments:
            self.applied.append(
                json.loads(
                    Path(arguments[arguments.index("-f") + 1]).read_text(
                        encoding="utf-8"
                    )
                )
            )
            return ""
        if "get" in arguments and "jsonpath" in " ".join(arguments):
            return self.published
        if "get" in arguments:
            return json.dumps(self.service) if self.service is not None else ""
        return ""

    def _aws(self, arguments: list[str]) -> str:
        if "list-resource-record-sets" in arguments:
            return json.dumps(
                {"ResourceRecordSets": [self.record] if self.record else []}
            )
        if "change-resource-record-sets" in arguments:
            return json.dumps({"ChangeInfo": {"Id": "/change/C1"}})
        if "get-change" in arguments:
            return json.dumps({"ChangeInfo": {"Status": "INSYNC"}})
        return "{}"

    def dns_actions(self) -> list[tuple[str, Any]]:
        actions: list[tuple[str, Any]] = []
        for arguments in self.calls:
            if "change-resource-record-sets" not in arguments:
                continue
            batch = json.loads(arguments[arguments.index("--change-batch") + 1])
            for change in batch["Changes"]:
                actions.append((change["Action"], change["ResourceRecordSet"]))
        return actions


def test_capture_reads_the_live_service_and_the_record_it_will_overwrite() -> None:
    release = Release(service=_live_service(), record=_record(PREVIOUS_TARGET))

    snapshot = ENDPOINT.capture_endpoint_snapshot(release)

    assert snapshot["configured"] is True
    assert snapshot["namespace"] == "gpu-fault-system"
    assert snapshot["absent"] == []
    assert [item["kind"] for item in snapshot["objects"]] == [
        item["kind"] for item in _declared()
    ]
    captured = snapshot["objects"][0]
    # The previous certificate ARN only ever existed in the previous checkout;
    # capturing the live object is what makes it recoverable at all.
    assert captured["metadata"]["annotations"] == {
        "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": "arn:previous"
    }
    for field in ("uid", "resourceVersion"):
        assert field not in captured["metadata"], field
    assert "status" not in captured
    assert snapshot["dns"] == {
        "hosted_zone_id": HOSTED_ZONE,
        "hostname": HOSTNAME,
        "record": _record(PREVIOUS_TARGET),
    }


def test_capture_skips_a_site_that_has_no_nlb_configured() -> None:
    # ``apply_control_plane_nlb`` returns before touching anything when ``nlb``
    # is empty, so there is nothing to compensate -- and a snapshot that
    # recorded the Service as absent would invite rollback into deleting a
    # Service this release never applied.
    release = Release(service=_live_service(), record=None, nlb={})

    assert ENDPOINT.capture_endpoint_snapshot(release) == {"configured": False}
    assert release.calls == []


def test_capture_records_a_candidate_created_service_for_deletion() -> None:
    release = Release(service=None, record=None)

    snapshot = ENDPOINT.capture_endpoint_snapshot(release)

    assert snapshot["objects"] == []
    assert snapshot["absent"] == [
        {"resource": item["resource"], "name": item["name"]} for item in _declared()
    ]
    assert snapshot["dns"]["record"] is None


def test_capture_fails_closed_without_a_hosted_zone() -> None:
    release = Release(service=_live_service(), record=None)
    release.config.dns = SimpleNamespace(hosted_zone_id="", hostname=HOSTNAME)

    with pytest.raises(MODULE.ReleaseError, match="dns.hosted_zone_id"):
        ENDPOINT.capture_endpoint_snapshot(release)


def _snapshot(
    *, objects: list[Any], absent: list[Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "configured": True,
        "namespace": "gpu-fault-system",
        "objects": objects,
        "absent": absent,
        "dns": {"hosted_zone_id": HOSTED_ZONE, "hostname": HOSTNAME, "record": record},
    }


def _previous_service() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": ENDPOINT.NLB_SERVICE, "namespace": "gpu-fault-system"},
        "spec": {"ports": [{"port": 443}]},
    }


def test_restore_puts_the_service_back_before_the_record() -> None:
    release = Release(service=_live_service(), record=_record(CANDIDATE_TARGET))

    ENDPOINT.restore_endpoint_snapshot(
        release,
        _snapshot(
            objects=[_previous_service()], absent=[], record=_record(PREVIOUS_TARGET)
        ),
    )

    verbs = [
        "apply"
        if "apply" in arguments
        else next((word for word in arguments if word.startswith("change-")), "read")
        for arguments in release.calls
    ]
    # The record must never name a load balancer whose configuration has not
    # been put back yet, so the Service is restored first.
    assert verbs[0] == "apply"
    assert "change-resource-record-sets" in verbs
    assert verbs.index("apply") < verbs.index("change-resource-record-sets")
    assert release.applied == [
        {"apiVersion": "v1", "kind": "List", "items": [_previous_service()]}
    ]
    assert release.dns_actions() == [("UPSERT", _record(PREVIOUS_TARGET))]


def test_restore_deletes_the_record_before_the_service_the_candidate_added() -> None:
    # Deleting the Service destroys the load balancer it published, so a record
    # still pointing at it would be left dangling.
    release = Release(service=None, record=_record(CANDIDATE_TARGET))
    absent = [
        {"resource": item["resource"], "name": item["name"]} for item in _declared()
    ]

    ENDPOINT.restore_endpoint_snapshot(
        release, _snapshot(objects=[], absent=absent, record=None)
    )

    order = [
        "delete-record" if "change-resource-record-sets" in arguments else word
        for arguments in release.calls
        for word in (next((item for item in arguments if item == "delete"), "read"),)
    ]
    assert "delete-record" in order and "delete" in order
    assert order.index("delete-record") < order.index("delete")
    # Route53 only deletes a record set it is given exactly, so the candidate's
    # own record is re-read rather than rebuilt from configuration that may
    # itself have been what changed.
    assert release.dns_actions() == [("DELETE", _record(CANDIDATE_TARGET))]
    assert release.applied == []


def test_restore_without_an_nlb_does_nothing() -> None:
    release = Release(service=_live_service(), record=_record(CANDIDATE_TARGET))

    ENDPOINT.restore_endpoint_snapshot(release, {"configured": False})

    assert release.calls == []


def test_restore_refuses_a_record_that_misses_the_restored_load_balancer() -> None:
    # A rollback that reported success here would leave every Agent resolving
    # the endpoint to an address the restored Service does not publish.
    release = Release(
        service=_live_service(),
        record=_record(CANDIDATE_TARGET),
        published=CANDIDATE_TARGET,
    )

    with pytest.raises(MODULE.ReleaseError, match="does not point at the restored"):
        ENDPOINT.restore_endpoint_snapshot(
            release,
            _snapshot(
                objects=[_previous_service()],
                absent=[],
                record=_record(PREVIOUS_TARGET),
            ),
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "snapshot",
    [
        {},
        {"namespace": "gpu-fault-system", "objects": [], "absent": []},
        _snapshot(objects=[_previous_service()], absent=[], record=None)
        | {"dns": {"hostname": HOSTNAME}},
        _snapshot(objects=[_previous_service()], absent=[], record=None)
        | {"namespace": ""},
    ],
)
def test_restore_refuses_a_snapshot_it_cannot_put_back(snapshot: object) -> None:
    release = Release(service=_live_service(), record=None)

    with pytest.raises(MODULE.ReleaseError, match="endpoint snapshot"):
        ENDPOINT.restore_endpoint_snapshot(release, snapshot)

    assert release.calls == [], "a snapshot that cannot be restored still mutated state"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class PublishingRelease(Release):
    """A Service whose hostname appears only after ``blank_reads`` empty reads.

    The bounded `kubectl wait` verdicts are scripted; a ``False`` consumes the
    wait's ``--timeout`` on the clock, as kubectl giving up at it would.
    """

    def __init__(self, clock: FakeClock, *, blank_reads: int, verdicts: list[bool]):
        super().__init__(service=_live_service(), record=_record(CANDIDATE_TARGET))
        self.clock = clock
        self.blank_reads = blank_reads
        self.verdicts = list(verdicts)
        self.waits: list[tuple[list[str], float]] = []
        self.runner = SimpleNamespace(run=self._run, probe=self._probe, dry_run=False)

    def _run(self, arguments: list[str], **kwargs: Any) -> str:
        if "get" in arguments and "jsonpath" in " ".join(arguments):
            self.calls.append(list(arguments))
            if self.blank_reads:
                self.blank_reads -= 1
                return ""
            return self.published
        return super()._run(arguments, **kwargs)

    def _probe(self, arguments: list[str], *, timeout_seconds=None) -> bool:
        self.waits.append((list(arguments), self.clock.now))
        met = self.verdicts.pop(0) if self.verdicts else False
        if not met:
            self.clock.now += _wait_timeout(arguments)
        return met


def _wait_timeout(arguments: list[str]) -> float:
    (flag,) = [value for value in arguments if value.startswith("--timeout=")]
    return float(flag.removeprefix("--timeout=").removesuffix("s"))


def _patch_clock(monkeypatch, clock: FakeClock) -> None:
    from gpu_fault_release import regional_dns as DNS
    from gpu_fault_release import regional_release_rollout_wait as WAIT

    for module in (ENDPOINT, DNS, WAIT):
        monkeypatch.setattr(module, "time", clock)


def _restored_previous_snapshot() -> dict[str, Any]:
    """The snapshot whose restore has to prove the record names the hostname
    the restored Service publishes."""

    return _snapshot(
        objects=[_previous_service()], absent=[], record=_record(PREVIOUS_TARGET)
    )


def _hostname_reads(release: Release) -> list[list[str]]:
    return [
        arguments
        for arguments in release.calls
        if "get" in arguments and "jsonpath" in " ".join(arguments)
    ]


def test_published_hostname_wakes_when_the_service_publishes_it(monkeypatch) -> None:
    """The 5s between reads is spent in `kubectl wait`, not a timer."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    release = PublishingRelease(clock, blank_reads=1, verdicts=[True])

    ENDPOINT.restore_endpoint_snapshot(release, _restored_previous_snapshot())

    # The record named the hostname the second read published, so the restore
    # accepted it; one blank read, one watch, one read that answered.
    assert release.dns_actions() == [("UPSERT", _record(PREVIOUS_TARGET))]
    assert len(_hostname_reads(release)) == 2
    assert [arguments[3:] for arguments, _ in release.waits] == [
        [
            "-n",
            "gpu-fault-system",
            "wait",
            "service/gpu-fault-api-nlb",
            "--for=jsonpath={.status.loadBalancer.ingress[0].hostname}",
            "--timeout=5s",
        ]
    ]
    assert release.waits[0][0][:3] == ["kubectl", "--kubeconfig", "/secure/cpu"]
    assert clock.sleeps == [] and clock.now == 0.0


def test_published_hostname_keeps_its_deadline_and_verdict(monkeypatch) -> None:
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    release = PublishingRelease(clock, blank_reads=10_000, verdicts=[])

    with pytest.raises(
        MODULE.ReleaseError, match="publishes no load balancer hostname"
    ):
        ENDPOINT.restore_endpoint_snapshot(release, _restored_previous_snapshot())

    assert clock.now == ENDPOINT.PUBLISHED_HOSTNAME_TIMEOUT_SECONDS
    assert all(
        _wait_timeout(arguments) <= ENDPOINT.PUBLISHED_HOSTNAME_TIMEOUT_SECONDS - at
        for arguments, at in release.waits
    ), "a watch was allowed to outlive the deadline"
    assert len(release.waits) == ENDPOINT.PUBLISHED_HOSTNAME_TIMEOUT_SECONDS // 5


def test_nlb_service_hostname_wait_wakes_on_the_same_watch(monkeypatch) -> None:
    from gpu_fault_release import regional_dns as DNS

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    release = PublishingRelease(clock, blank_reads=2, verdicts=[False, True])

    hostname = DNS.wait_service_hostname(release)

    assert hostname == PREVIOUS_TARGET
    assert [(arguments[5:], at) for arguments, at in release.waits] == [
        (
            [
                "wait",
                "service/gpu-fault-api-nlb",
                "--for=jsonpath={.status.loadBalancer.ingress[0].hostname}",
                "--timeout=5s",
            ],
            0.0,
        ),
        (
            [
                "wait",
                "service/gpu-fault-api-nlb",
                "--for=jsonpath={.status.loadBalancer.ingress[0].hostname}",
                "--timeout=5s",
            ],
            5.0,
        ),
    ]
    assert clock.sleeps == []


def test_nlb_service_hostname_wait_still_fails_at_600s(monkeypatch) -> None:
    from gpu_fault_release import regional_dns as DNS

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    release = PublishingRelease(clock, blank_reads=10_000, verdicts=[])

    with pytest.raises(
        MODULE.ReleaseError, match="did not publish a hostname within 600s"
    ):
        DNS.wait_service_hostname(release)

    assert clock.now == 600.0
    assert all(
        _wait_timeout(arguments) <= 600 - at for arguments, at in release.waits
    ), release.waits
