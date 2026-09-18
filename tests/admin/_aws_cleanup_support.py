from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from typing import Any

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

ACCOUNT = "123456789012"
REGION = "us-east-1"
SITE = "test-site"
TAGS = [{"Key": "gpu-fault:site-id", "Value": SITE}]
TOPIC = f"arn:aws:sns:{REGION}:{ACCOUNT}:gpu-fault"
QUEUE = f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/gpu-fault"


def resource(
    kind: str,
    identifier: str,
    *,
    arn: str | None = None,
    attributes: dict[str, str] | None = None,
    policy: InstallationResourceDeletePolicy = InstallationResourceDeletePolicy.DELETE,
    ownership: InstallationResourceOwnership = InstallationResourceOwnership.CREATED,
) -> InstallationResource:
    return InstallationResource(
        site_id=SITE,
        resource_key=f"aws/{kind}/{hashlib.sha256(identifier.encode()).hexdigest()[:12]}",
        resource_type=kind,
        resource_id=identifier,
        resource_arn=arn,
        region=REGION,
        account_id=ACCOUNT,
        provider="kubernetes" if kind == "helm_release" else "aws",
        ownership=ownership,
        delete_policy=policy,
        attributes=attributes or {},
    )


class Aws:
    def __init__(
        self,
        responses: dict[tuple[str, str], Any | Callable[[list[str]], Any]]
        | None = None,
    ) -> None:
        self.responses = {
            ("sts", "get-caller-identity"): {"Account": ACCOUNT},
            **(responses or {}),
        }
        self.calls: list[list[str]] = []

    def __call__(
        self, arguments: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(arguments))
        key = (arguments[1], arguments[2])
        if key not in self.responses:
            raise AssertionError(f"unexpected cleanup operation {key}")
        response = self.responses[key]
        if callable(response):
            response = response(arguments)
        if isinstance(response, subprocess.CompletedProcess):
            return response
        return subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps(response), stderr=""
        )

    @property
    def mutations(self) -> list[list[str]]:
        return [
            call
            for call in self.calls
            if call[2].startswith(
                (
                    "delete-",
                    "modify-",
                    "set-",
                    "detach-",
                    "remove-",
                    "disassociate-",
                    "change-",
                )
            )
            or call[2] == "unsubscribe"
        ]


def absent(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["aws"], 254, stdout="", stderr=f"An error occurred ({code})"
    )
