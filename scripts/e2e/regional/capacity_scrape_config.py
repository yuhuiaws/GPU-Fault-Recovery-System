"""Strict, credential-free ADOT inputs for the CAP002 scrape companion."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

import yaml  # type: ignore[import-untyped]

from scripts.e2e.regional.regional_commands import RegionalFixtureError


class ScrapeCompanionError(RegionalFixtureError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScrapeCompanionError(message)


def name(value: Any, *, limit: int = 63) -> str:
    label = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    pattern = rf"{label}(?:\.{label})*" if limit > 63 else label
    require(
        isinstance(value, str)
        and 0 < len(value) <= limit
        and re.fullmatch(pattern, value) is not None
        and all(len(part) <= 63 for part in value.split(".")),
        "scrape resource name is invalid",
    )
    return str(value)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fields(
    value: Any, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and required <= value.keys()
        and value.keys() <= required | (optional or set()),
        "ADOT configuration has missing or unsupported fields",
    )
    return dict(value)


class _UniqueLoader(yaml.SafeLoader):  # type: ignore[misc]
    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            require(
                isinstance(key, str) and key not in result,
                "ADOT YAML keys must be unique strings",
            )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def seconds(value: Any) -> float:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9]+(?:ms|s|m)", value) is not None,
        "ADOT duration is invalid",
    )
    suffix = "ms" if value.endswith("ms") else value[-1]
    return int(value[: -len(suffix)]) * {"ms": 0.001, "s": 1, "m": 60}[suffix]


def pipeline(
    raw: Any, *, region: str, workspace_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return only validated exporter settings, never the production receiver."""
    require(
        isinstance(region, str)
        and re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]", region) is not None,
        "AMP Region is invalid",
    )
    name(workspace_id, limit=80)
    require(
        re.fullmatch(r"ws-[a-z0-9-]+", workspace_id) is not None,
        "AMP workspace ID is invalid",
    )
    require(
        isinstance(raw, str) and 0 < len(raw.encode()) <= 131072,
        "ADOT configuration is missing or too large",
    )
    try:
        config = yaml.load(raw, Loader=_UniqueLoader)
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):
        raise ScrapeCompanionError("ADOT YAML is invalid") from None
    config = fields(
        config, {"receivers", "processors", "exporters", "extensions", "service"}
    )
    exporter = fields(config["exporters"], {"prometheusremotewrite"})[
        "prometheusremotewrite"
    ]
    exporter = fields(
        exporter, {"endpoint", "add_metric_suffixes", "auth", "retry_on_failure"}
    )
    endpoint = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
        f"{workspace_id}/api/v1/remote_write"
    )
    require(
        exporter["endpoint"] == endpoint
        and exporter["add_metric_suffixes"] is False
        and exporter["auth"] == {"authenticator": "sigv4auth"},
        "ADOT exporter must use the bound AMP workspace and SigV4",
    )
    retry = fields(
        exporter["retry_on_failure"],
        {"enabled", "initial_interval", "max_interval", "max_elapsed_time"},
    )
    require(
        retry["enabled"] is True
        and 0
        < seconds(retry["initial_interval"])
        <= seconds(retry["max_interval"])
        <= seconds(retry["max_elapsed_time"])
        <= 120,
        "ADOT exporter retry bounds are invalid",
    )
    batch = fields(
        fields(config["processors"], {"batch"})["batch"],
        {"timeout", "send_batch_size", "send_batch_max_size"},
    )
    require(
        0 < seconds(batch["timeout"]) <= 10
        and type(batch["send_batch_size"]) is int
        and type(batch["send_batch_max_size"]) is int
        and 0 < batch["send_batch_size"] <= batch["send_batch_max_size"] <= 2000,
        "ADOT batch bounds are invalid",
    )
    extensions = fields(config["extensions"], {"sigv4auth", "health_check"})
    require(
        extensions["sigv4auth"] == {"region": region, "service": "aps"}
        and extensions["health_check"]
        == {"endpoint": "0.0.0.0:8090", "path": "/health"},
        "ADOT extensions differ from the supported AMP health pipeline",
    )
    service = fields(config["service"], {"extensions", "pipelines", "telemetry"})
    require(
        service["extensions"] == ["sigv4auth", "health_check"]
        and service["pipelines"]
        == {
            "metrics": {
                "receivers": ["prometheus"],
                "processors": ["batch"],
                "exporters": ["prometheusremotewrite"],
            }
        },
        "ADOT service pipeline is unsupported",
    )
    telemetry = fields(service["telemetry"], {"logs", "metrics"})
    logs = fields(telemetry["logs"], {"level"})
    require(logs["level"] in {"info", "warn", "error"}, "ADOT log level is unsafe")
    metrics = fields(telemetry["metrics"], {"level", "readers"})
    require(
        metrics["level"] in {"basic", "normal", "detailed"}
        and metrics["readers"]
        == [
            {"pull": {"exporter": {"prometheus": {"host": "127.0.0.1", "port": 8889}}}}
        ],
        "ADOT self telemetry must remain on loopback",
    )
    receiver = fields(
        fields(config["receivers"], {"prometheus"})["prometheus"], {"config"}
    )
    jobs = fields(receiver["config"], {"scrape_configs"})["scrape_configs"]
    require(
        isinstance(jobs, list) and all(isinstance(job, dict) for job in jobs),
        "ADOT scrape inventory is invalid",
    )
    selected = [job for job in jobs if job.get("job_name") == "gpu-fault-control-plane"]
    require(len(selected) == 1, "ADOT control-plane scrape job is ambiguous")
    rules = selected[0].get("relabel_configs")
    require(isinstance(rules, list), "ADOT target labels are unavailable")
    labels: dict[str, str] = {}
    for target in ("region", "control_plane_cluster"):
        matches = [
            rule
            for rule in rules
            if isinstance(rule, dict) and rule.get("target_label") == target
        ]
        require(len(matches) == 1, "ADOT target labels are ambiguous")
        rule = fields(matches[0], {"action", "target_label", "replacement"})
        replacement = rule["replacement"]
        require(
            rule["action"] == "replace"
            and isinstance(replacement, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", replacement)
            is not None,
            "ADOT target label is invalid",
        )
        labels[target] = replacement
    require(labels["region"] == region, "ADOT target Region differs")
    safe = {
        key: copy.deepcopy(config[key])
        for key in ("processors", "exporters", "extensions", "service")
    }
    return safe, labels


def workload_identity(
    pod: dict[str, Any], service_account: dict[str, Any], *, region: str
) -> dict[str, Any]:
    """Validate fresh projected IRSA/Pod Identity configuration, not token bytes."""
    spec = pod["spec"]
    container = spec["containers"][0]
    require(
        not container.get("envFrom"), "ADOT identity cannot use indirect environment"
    )
    entries = container.get("env", [])
    require(isinstance(entries, list), "ADOT environment is invalid")
    env: dict[str, str] = {}
    for entry in entries:
        entry = fields(entry, {"name", "value"})
        key, value = entry["name"], entry["value"]
        require(
            isinstance(key, str) and isinstance(value, str) and key not in env,
            "ADOT environment is duplicated or indirect",
        )
        env[key] = value
    common = {
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_STS_REGIONAL_ENDPOINTS",
        "GOGC",
        "GOMEMLIMIT",
    }
    for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        require(env.get(key, region) == region, "ADOT identity Region differs")
    require(
        env.get("AWS_STS_REGIONAL_ENDPOINTS", "regional") == "regional",
        "ADOT STS endpoint selection is unsupported",
    )
    require(
        re.fullmatch(r"[1-9][0-9]{0,2}", env.get("GOGC", "50")) is not None
        and re.fullmatch(r"[1-9][0-9]{0,2}MiB", env.get("GOMEMLIMIT", "220MiB"))
        is not None
        and int(env.get("GOMEMLIMIT", "220MiB")[:-3]) <= 220,
        "ADOT runtime memory settings are unsupported",
    )
    annotations = service_account["metadata"].get("annotations") or {}
    require(
        isinstance(annotations, dict), "ADOT ServiceAccount annotations are invalid"
    )
    role = annotations.get("eks.amazonaws.com/role-arn")
    if role is not None:
        require(
            isinstance(role, str)
            and re.fullmatch(
                r"arn:aws(?:-us-gov|-cn)?:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+",
                role,
            )
            is not None,
            "ADOT IRSA role reference is invalid",
        )
        mode, audience, path = "irsa", "sts.amazonaws.com", "token"
        mount_path = "/var/run/secrets/eks.amazonaws.com/serviceaccount"
        expected = {
            "AWS_ROLE_ARN": role,
            "AWS_WEB_IDENTITY_TOKEN_FILE": f"{mount_path}/{path}",
        }
    else:
        mode, audience, path = (
            "pod-identity",
            "pods.eks.amazonaws.com",
            "eks-pod-identity-token",
        )
        mount_path = "/var/run/secrets/pods.eks.amazonaws.com/serviceaccount"
        uri = env.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        require(
            uri
            in {
                "http://169.254.170.23/v1/credentials",
                "http://[fd00:ec2::23]/v1/credentials",
            },
            "ADOT must have an existing EKS workload identity",
        )
        expected = {
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": uri,
            "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": f"{mount_path}/{path}",
        }
    require(
        env.keys() <= common | expected.keys()
        and all(env.get(key) == value for key, value in expected.items()),
        "ADOT identity environment is missing or contains unapproved credentials",
    )
    mounts = container.get("volumeMounts", [])
    require(isinstance(mounts, list), "ADOT identity mount is unavailable")
    selected_mounts = [
        item
        for item in mounts
        if isinstance(item, dict) and item.get("mountPath") == mount_path
    ]
    require(len(selected_mounts) == 1, "ADOT identity mount is ambiguous")
    mount = fields(selected_mounts[0], {"name", "mountPath", "readOnly"})
    name(mount["name"])
    require(mount["readOnly"] is True, "ADOT identity mount is writable")
    volumes = spec.get("volumes", [])
    require(isinstance(volumes, list), "ADOT identity projection is unavailable")
    selected_volumes = [
        item
        for item in volumes
        if isinstance(item, dict) and item.get("name") == mount["name"]
    ]
    require(len(selected_volumes) == 1, "ADOT identity projection is ambiguous")
    volume = fields(selected_volumes[0], {"name", "projected"})
    projected = fields(volume["projected"], {"sources"}, {"defaultMode"})
    require(
        projected.get("defaultMode", 420) in {420, 292}
        and isinstance(projected["sources"], list)
        and len(projected["sources"]) == 1,
        "ADOT identity projection has unapproved sources",
    )
    token = fields(
        fields(projected["sources"][0], {"serviceAccountToken"})["serviceAccountToken"],
        {"audience", "expirationSeconds", "path"},
    )
    require(
        token["audience"] == audience
        and token["path"] == path
        and type(token["expirationSeconds"]) is int
        and 600 <= token["expirationSeconds"] <= 86400,
        "ADOT identity projection scope is invalid",
    )
    return {
        "mode": mode,
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
        "volume": copy.deepcopy(volume),
        "mount": copy.deepcopy(mount),
    }
