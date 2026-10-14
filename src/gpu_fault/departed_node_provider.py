"""The provider half of the departed-node proof, read by the control plane itself.

Two reads, both read-only and both answering "does the provider still have
this instance?" -- never "what is it doing": HyperPod's ``ListClusterNodes``
for the registered HyperPod cluster (``sagemaker:ListClusterNodes``, which the
control-plane role has held since the identity registry), and EC2's
``DescribeInstances`` for the instance ids the fleet recorded on an
``eks-managed`` cluster (``ec2:DescribeInstances``, the bootstrap's
``DepartedNodeInstanceReads`` statement). The EC2 read filters by instance id
instead of naming the ids, because naming one the account no longer knows
fails the whole call -- and "unknown to EC2" is exactly the answer wanted.

Every failure -- a throttle, a denied call, a timeout, an answer that does not
parse, more pages than ``PROVIDER_PAGE_LIMIT`` -- raises
``ProviderLookupError``: the caller turns it into "provider membership
unknown" and tries again next cycle. Nothing here reads as "absent" by default.

The interface is deliberately narrow (``ProviderMembership``) so the EC2
client and its IAM statement can be shared with other control-plane readers
without this module growing into a provider adapter.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

#: Pages of either listing read before the lookup is declared broken.
PROVIDER_PAGE_LIMIT = 100
#: Instance ids per ``DescribeInstances`` filter (EC2 caps filter values).
EC2_FILTER_BATCH = 200
#: Seconds a provider call may take; a slow provider is a failed lookup.
PROVIDER_CALL_TIMEOUT_SECONDS = 10


class ProviderLookupError(RuntimeError):
    """The provider could not answer; ``str()`` is safe to log and record."""


class ProviderMembership(Protocol):
    def hyperpod_instance_ids(self, cluster_name: str, region: str) -> frozenset[str]:
        """The EC2 instance ids HyperPod lists for ``cluster_name``."""
        ...

    def ec2_instance_states(
        self, region: str, instance_ids: Sequence[str]
    ) -> dict[str, str]:
        """``instance_id -> state name``; an id EC2 no longer knows is absent."""
        ...


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict) and error.get("Code"):
            return str(error["Code"])
    return type(exc).__name__


def _default_client_factory(service: str, region: str) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - the image ships boto3
        raise ProviderLookupError("boto3 is not installed") from exc
    return boto3.client(
        service,
        region_name=region,
        config=Config(
            connect_timeout=PROVIDER_CALL_TIMEOUT_SECONDS,
            read_timeout=PROVIDER_CALL_TIMEOUT_SECONDS,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


class Boto3ProviderMembership:
    """``ProviderMembership`` over boto3, one client per service and region."""

    def __init__(self, client_factory: Callable[[str, str], Any] | None = None) -> None:
        self._client_factory = client_factory or _default_client_factory
        self._clients: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    def _client(self, service: str, region: str) -> Any:
        if not region:
            raise ProviderLookupError(f"no AWS region for the {service} lookup")
        key = (service, region)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = self._client_factory(service, region)
            return self._clients[key]

    def hyperpod_instance_ids(self, cluster_name: str, region: str) -> frozenset[str]:
        if not cluster_name.strip():
            raise ProviderLookupError("the cluster has no HyperPod cluster name")
        client = self._client("sagemaker", region)
        instance_ids: set[str] = set()
        token: str | None = None
        for _page in range(PROVIDER_PAGE_LIMIT):
            request: dict[str, Any] = {"ClusterName": cluster_name, "MaxResults": 100}
            if token:
                request["NextToken"] = token
            try:
                response = client.list_cluster_nodes(**request)
            except Exception as exc:
                raise ProviderLookupError(
                    f"cannot list HyperPod nodes of {cluster_name}: {_error_code(exc)}"
                ) from exc
            summaries = (
                response.get("ClusterNodeSummaries")
                if isinstance(response, dict)
                else None
            )
            if not isinstance(summaries, list):
                raise ProviderLookupError(
                    f"HyperPod node inventory of {cluster_name} is invalid"
                )
            for row in summaries:
                value = row.get("InstanceId") if isinstance(row, dict) else None
                if isinstance(value, str) and value.strip():
                    instance_ids.add(value.strip())
            next_token = response.get("NextToken")
            if not next_token:
                return frozenset(instance_ids)
            if not isinstance(next_token, str) or next_token == token:
                raise ProviderLookupError(
                    f"HyperPod node inventory of {cluster_name} repeats its page token"
                )
            token = next_token
        raise ProviderLookupError(
            f"HyperPod node inventory of {cluster_name} exceeds "
            f"{PROVIDER_PAGE_LIMIT} pages"
        )

    def ec2_instance_states(
        self, region: str, instance_ids: Sequence[str]
    ) -> dict[str, str]:
        unique = sorted(
            {str(item).strip() for item in instance_ids if str(item).strip()}
        )
        if not unique:
            return {}
        client = self._client("ec2", region)
        states: dict[str, str] = {}
        for start in range(0, len(unique), EC2_FILTER_BATCH):
            batch = unique[start : start + EC2_FILTER_BATCH]
            token: str | None = None
            for _page in range(PROVIDER_PAGE_LIMIT):
                request: dict[str, Any] = {
                    "Filters": [{"Name": "instance-id", "Values": batch}]
                }
                if token:
                    request["NextToken"] = token
                try:
                    response = client.describe_instances(**request)
                except Exception as exc:
                    raise ProviderLookupError(
                        f"cannot describe EC2 instances in {region}: {_error_code(exc)}"
                    ) from exc
                reservations = (
                    response.get("Reservations") if isinstance(response, dict) else None
                )
                if not isinstance(reservations, list):
                    raise ProviderLookupError("EC2 instance description is invalid")
                for reservation in reservations:
                    instances = (
                        reservation.get("Instances")
                        if isinstance(reservation, dict)
                        else None
                    )
                    if not isinstance(instances, list):
                        raise ProviderLookupError("EC2 instance description is invalid")
                    for instance in instances:
                        identifier = (
                            instance.get("InstanceId")
                            if isinstance(instance, dict)
                            else None
                        )
                        state = (
                            (instance.get("State") or {}).get("Name")
                            if isinstance(instance, dict)
                            else None
                        )
                        if not isinstance(identifier, str) or not isinstance(
                            state, str
                        ):
                            raise ProviderLookupError(
                                "EC2 instance description is invalid"
                            )
                        states[identifier] = state.strip().lower()
                token = response.get("NextToken")
                if not token:
                    break
            else:
                raise ProviderLookupError(
                    f"EC2 instance description exceeds {PROVIDER_PAGE_LIMIT} pages"
                )
        return states
