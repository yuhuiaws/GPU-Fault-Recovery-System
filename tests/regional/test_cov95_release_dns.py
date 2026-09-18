from __future__ import annotations

import copy
import json
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_dns as dns
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_checks import HOSTNAME, CheckedRelease
from tests.regional._cov95_release_support import Clock

NLB_HOSTNAME = "example-nlb.elb.amazonaws.com"


class DnsRelease(CheckedRelease):
    def __init__(self) -> None:
        super().__init__()
        self.config.dns = SimpleNamespace(hosted_zone_id="ZEXAMPLE", hostname=HOSTNAME)
        self.service_hostname = NLB_HOSTNAME
        self.expected_replicas = "3"
        self.aws.update(
            {
                ("route53", "get-hosted-zone"): {
                    "HostedZone": {
                        "Id": "/hostedzone/ZEXAMPLE",
                        "Config": {"PrivateZone": True},
                        "Name": "example.test.",
                    }
                },
                ("route53", "list-resource-record-sets"): {
                    "ResourceRecordSets": [
                        {
                            "Name": HOSTNAME + ".",
                            "Type": "CNAME",
                            "TTL": 60,
                            "ResourceRecords": [{"Value": NLB_HOSTNAME + "."}],
                        }
                    ]
                },
                ("route53", "change-resource-record-sets"): {
                    "ChangeInfo": {"Id": "/change/example"}
                },
                ("route53", "get-change"): {"ChangeInfo": {"Status": "INSYNC"}},
                ("route53", "wait"): {},
            }
        )
        self.aws["elbv2", "describe-load-balancers"]["LoadBalancers"][0]["DNSName"] = (
            NLB_HOSTNAME
        )

    def dispatch(self, arguments: list[str], kwargs: dict[str, Any]) -> str:
        if arguments[0] == "kubectl" and "get" in arguments:
            kind = arguments[arguments.index("get") + 1]
            if kind == "service":
                return self.service_hostname
            if kind == "deployment":
                return self.expected_replicas
        return super().dispatch(arguments, kwargs)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(dns.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dns.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        dns.socket,
        "getaddrinfo",
        lambda *_args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))
        ],
    )
    return clock


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("missing", "requires dns.hosted_zone_id"),
        ("zone-id", "exactly once"),
        ("public-zone", "private hosted zone"),
        ("hostname", "not inside hosted zone"),
        ("certificate-state", "expected ISSUED"),
        ("certificate-name", "does not cover"),
        ("expiry", "has no NotAfter"),
        ("expired", "certificate is expired"),
    ],
)
def test_dns_prerequisites_fail_on_identity_scope_and_certificate_errors(
    fault: str, problem: str
) -> None:
    release = DnsRelease()
    zone = release.aws["route53", "get-hosted-zone"]["HostedZone"]
    certificate = release.aws["acm", "describe-certificate"]["Certificate"]
    if fault == "missing":
        release.config.dns.hosted_zone_id = ""
    elif fault == "zone-id":
        zone["Id"] = "other"
    elif fault == "public-zone":
        zone["Config"]["PrivateZone"] = False
    elif fault == "hostname":
        release.config.dns.hostname = "outside.invalid"
    elif fault == "certificate-state":
        certificate["Status"] = "PENDING_VALIDATION"
    elif fault == "certificate-name":
        certificate["SubjectAlternativeNames"] = []
    elif fault == "expiry":
        certificate["NotAfter"] = None
    else:
        certificate["NotAfter"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ReleaseError, match=problem):
        dns.verify_control_plane_dns_prerequisites(release)
    assert all(
        "change-resource-record-sets" not in args
        for args, _kwargs in release.runner.calls
    ), "failed DNS prerequisites must not issue a Route53 mutation"


def test_dns_prerequisites_support_exact_and_single_label_wildcard_names() -> None:
    release = DnsRelease()
    dns.verify_control_plane_dns_prerequisites(release)
    release.config.dns.hostname = HOSTNAME.upper() + "."
    release.aws["acm", "describe-certificate"]["Certificate"][
        "SubjectAlternativeNames"
    ] = ["*.EXAMPLE.TEST."]
    dns.verify_control_plane_dns_prerequisites(release)
    release.runner.dry_run = True
    release.runner.calls.clear()
    dns.verify_control_plane_dns_prerequisites(release)
    assert release.runner.calls == []


def test_record_lookup_ignores_nonobjects_wrong_names_and_wrong_record_types() -> None:
    release = DnsRelease()
    expected = release.aws["route53", "list-resource-record-sets"][
        "ResourceRecordSets"
    ][0]
    release.aws["route53", "list-resource-record-sets"]["ResourceRecordSets"] = [
        None,
        {"Name": "other.test", "Type": "CNAME"},
        {"Name": HOSTNAME, "Type": "A"},
        expected,
    ]
    assert (
        dns.read_dns_record(release, hosted_zone_id="ZEXAMPLE", hostname=HOSTNAME)
        == expected
    )
    release.aws["route53", "list-resource-record-sets"]["ResourceRecordSets"].pop()
    assert (
        dns.read_dns_record(release, hosted_zone_id="ZEXAMPLE", hostname=HOSTNAME)
        is None
    )
    assert all("--region" not in args for args, _kwargs in release.runner.calls), (
        "global Route53 reads must not inherit a regional endpoint option"
    )
    assert all(kwargs["sensitive"] for _args, kwargs in release.runner.calls), (
        "DNS record reads must remain on the private transport path"
    )


@pytest.mark.parametrize(
    "fault,problem",
    [("id", "did not return a change ID"), ("pending", "expected INSYNC")],
)
def test_dns_change_requires_acknowledgement_and_insync(
    fault: str, problem: str
) -> None:
    release = DnsRelease()
    if fault == "id":
        release.aws["route53", "change-resource-record-sets"] = {}
    else:
        release.aws["route53", "get-change"]["ChangeInfo"]["Status"] = "PENDING"
    with pytest.raises(ReleaseError, match=problem):
        dns.submit_dns_change(
            release,
            [{"Action": "UPSERT", "ResourceRecordSet": {}}],
            hosted_zone_id="ZEXAMPLE",
        )
    assert len(release.runner.calls) == (1 if fault == "id" else 3)


def test_unchanged_dns_needs_no_change_and_changed_dns_waits_for_insync(
    clock: Clock,
) -> None:
    release = DnsRelease()
    dns.submit_dns_change(release, [], hosted_zone_id="ZEXAMPLE")
    assert release.runner.calls == []
    dns.ensure_control_plane_dns(release)
    assert all(
        "change-resource-record-sets" not in args
        for args, _kwargs in release.runner.calls
    ), "an already converged CNAME must not be rewritten"
    release.aws["route53", "list-resource-record-sets"]["ResourceRecordSets"] = []
    dns.ensure_cname_points_at(release, NLB_HOSTNAME)
    change = next(
        args
        for args, _kwargs in release.runner.calls
        if "change-resource-record-sets" in args
    )
    payload = json.loads(change[change.index("--change-batch") + 1])
    assert payload["Changes"][0]["ResourceRecordSet"]["ResourceRecords"] == [
        {"Value": NLB_HOSTNAME}
    ]
    assert any(
        "resource-record-sets-changed" in args for args, _kwargs in release.runner.calls
    ), "a changed CNAME must wait for Route53 acknowledgement"
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("nlb-absent", "within 900s"),
        ("nlb-starting", "within 900s"),
        ("listener", "configured TLS listener"),
        ("certificate", "configured TLS listener"),
        ("hostname", "does not match"),
        ("replicas", "expected NLB target count"),
        ("target-groups", "healthy targets"),
        ("target-health", "healthy targets"),
    ],
)
def test_dns_convergence_waits_are_bounded_and_fail_closed(
    clock: Clock, fault: str, problem: str
) -> None:
    release = DnsRelease()
    if fault == "nlb-absent":
        release.aws["elbv2", "describe-load-balancers"]["LoadBalancers"] = []
    elif fault == "nlb-starting":
        release.aws["elbv2", "describe-load-balancers"]["LoadBalancers"][0]["State"][
            "Code"
        ] = "provisioning"
    elif fault == "listener":
        release.aws["elbv2", "describe-listeners"]["Listeners"] = []
    elif fault == "certificate":
        release.aws["elbv2", "describe-listeners"]["Listeners"][0]["Certificates"] = []
    elif fault == "hostname":
        release.aws["elbv2", "describe-load-balancers"]["LoadBalancers"][0][
            "DNSName"
        ] = "different.invalid"
    elif fault == "replicas":
        release.expected_replicas = "unknown"
    elif fault == "target-groups":
        release.aws["elbv2", "describe-target-groups"]["TargetGroups"] = []
    else:
        release.aws["elbv2", "describe-target-health"]["TargetHealthDescriptions"][0][
            "TargetHealth"
        ]["State"] = "unhealthy"
    with pytest.raises(ReleaseError, match=problem):
        dns.ensure_control_plane_dns(release)
    assert clock.elapsed <= 900
    assert not any(
        "change-resource-record-sets" in args for args, _kwargs in release.runner.calls
    ), "failed NLB convergence must not publish its DNS record"


@pytest.mark.parametrize("returns_empty", [False, True])
def test_raw_dns_failure_or_empty_answer_cannot_pass_convergence(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, returns_empty: bool
) -> None:
    def lookup(*_args: Any) -> list[Any]:
        if returns_empty:
            return []
        raise socket.gaierror("fixture name resolution failed")

    monkeypatch.setattr(dns.socket, "getaddrinfo", lookup)
    with pytest.raises(ReleaseError, match="raw NLB DNS did not resolve"):
        dns.ensure_control_plane_dns(DnsRelease())
    assert clock.elapsed == 300


def test_service_hostname_deadline_uses_bounded_waits(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    release = DnsRelease()
    release.service_hostname = ""
    observed = []

    def wait(_release: Any, arguments: list[str], *, seconds: float) -> None:
        observed.append((arguments, seconds))
        clock.sleep(seconds)

    monkeypatch.setattr(dns, "bounded_kubectl_wait", wait)
    with pytest.raises(ReleaseError, match="within 600s"):
        dns.wait_service_hostname(release)
    assert clock.elapsed == 600
    assert all(0 < seconds <= 5 for _arguments, seconds in observed), (
        "hostname retries must use positive waits within the five-second bound"
    )


def test_disabled_nlb_and_dry_run_empty_hostname_do_not_issue_changes(
    clock: Clock,
) -> None:
    release = DnsRelease()
    release.config.nlb = {}
    dns.ensure_control_plane_dns(release)
    dns.prepare_control_plane_nlb(release)
    assert release.runner.calls == []
    release = DnsRelease()
    release.runner.dry_run = True
    release.service_hostname = ""
    dns.ensure_control_plane_dns(release)
    assert len(release.runner.calls) == 1
    assert clock.sleeps == []


def test_dry_run_nlb_and_target_waits_do_not_spin(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    release = DnsRelease()
    release.runner.dry_run = True
    release.aws["elbv2", "describe-load-balancers"]["LoadBalancers"] = []
    release.aws["elbv2", "describe-target-groups"]["TargetGroups"] = []
    monkeypatch.setattr(dns.socket, "getaddrinfo", lambda *_args: [])
    before = copy.deepcopy(release.aws)
    dns.ensure_control_plane_dns(release)
    assert release.aws == before
    assert clock.sleeps == []
