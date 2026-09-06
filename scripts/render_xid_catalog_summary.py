from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.policy.catalog_integrity import (
    validate_xid_catalog_document,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml"
DEFAULT_SOURCE_LABEL = "src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml"
SUMMARY_XIDS = (
    45,
    48,
    94,
    95,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    154,
    159,
    164,
    165,
)


class CatalogSummaryError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogSummaryError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CatalogSummaryError(f"{name} must be a sequence")
    return value


def load_catalog(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogSummaryError(f"cannot read catalog {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogSummaryError(f"invalid catalog YAML {path}: {exc}") from exc

    root = _mapping(document, "catalog document")
    if root.get("apiVersion") != "gpu-fault.io/v1alpha1":
        raise CatalogSummaryError("unexpected XID catalog apiVersion")
    if root.get("kind") != "NvidiaXidCatalogPolicy":
        raise CatalogSummaryError("unexpected XID catalog kind")
    try:
        validate_xid_catalog_document(root)
    except ValueError as exc:
        raise CatalogSummaryError(str(exc)) from exc

    metadata = _mapping(root.get("metadata"), "metadata")
    spec = _mapping(root.get("spec"), "spec")
    catalog = _mapping(spec.get("catalog"), "spec.catalog")
    defaults = _mapping(spec.get("defaults"), "spec.defaults")
    correlation = _mapping(spec.get("correlation"), "spec.correlation")
    nvlink5 = _mapping(spec.get("nvlink5"), "spec.nvlink5")
    _sequence(catalog.get("productFamilies"), "spec.catalog.productFamilies")
    _sequence(spec.get("catalogRules"), "spec.catalogRules")
    _sequence(nvlink5.get("decodeRules"), "spec.nvlink5.decodeRules")
    _mapping(spec.get("resolutionBuckets"), "spec.resolutionBuckets")

    required = {
        "metadata.name": metadata.get("name"),
        "metadata.generatedFrom": metadata.get("generatedFrom"),
        "metadata.sourceSha256": metadata.get("sourceSha256"),
        "spec.catalog.version": catalog.get("version"),
        "spec.catalog.coverage": catalog.get("coverage"),
        "spec.defaults.markerTtlSeconds": defaults.get("markerTtlSeconds"),
        "spec.correlation.companionWindowSeconds": correlation.get(
            "companionWindowSeconds"
        ),
        "spec.nvlink5.driverBoundary": nvlink5.get("driverBoundary"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise CatalogSummaryError(
            "catalog is missing required fields: " + ", ".join(missing)
        )
    return root


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return " ".join(str(value).split()).replace("|", "\\|")


def render_summary(
    document: dict[str, Any],
    *,
    source_label: str = DEFAULT_SOURCE_LABEL,
) -> str:
    metadata = _mapping(document["metadata"], "metadata")
    spec = _mapping(document["spec"], "spec")
    catalog = _mapping(spec["catalog"], "spec.catalog")
    defaults = _mapping(spec["defaults"], "spec.defaults")
    correlation = _mapping(spec["correlation"], "spec.correlation")
    rules = _sequence(spec["catalogRules"], "spec.catalogRules")
    nvlink5 = _mapping(spec["nvlink5"], "spec.nvlink5")
    decode_rules = _sequence(nvlink5["decodeRules"], "spec.nvlink5.decodeRules")
    resolution_buckets = _mapping(
        spec["resolutionBuckets"],
        "spec.resolutionBuckets",
    )

    rules_by_xid = {
        int(_mapping(rule, "catalog rule")["xid"]): _mapping(
            rule,
            "catalog rule",
        )
        for rule in rules
    }
    missing_xids = [str(xid) for xid in SUMMARY_XIDS if xid not in rules_by_xid]
    if missing_xids:
        raise CatalogSummaryError(
            "summary XIDs are absent from the catalog: " + ", ".join(missing_xids)
        )

    decode_by_xid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in decode_rules:
        rule = _mapping(item, "NVLink5 decode rule")
        decode_by_xid[int(rule["xid"])].append(rule)

    product_families = []
    for item in _sequence(
        catalog["productFamilies"],
        "spec.catalog.productFamilies",
    ):
        family = _mapping(item, "product family")
        prefixes = "/".join(str(value) for value in family["modelPrefixes"])
        product_families.append(f"{family['family']} ({prefixes})")

    lines = [
        "# 附录：固定版本 NVIDIA XID Catalog 摘要",
        "",
        (
            f"> 本附录在构建时直接从 `{source_label}` 生成。它是只读摘要，"
            "不是可编辑的运行时配置；完整规则以该固定版本 catalog 为准。"
        ),
        "",
        "## Catalog 身份",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| API / Kind | `{document['apiVersion']}` / `{document['kind']}` |",
        f"| 名称 | `{metadata['name']}` |",
        f"| Catalog 版本 | `{catalog['version']}` |",
        f"| 覆盖范围 | `{catalog['coverage']}` |",
        f"| 上游文件 | <{metadata['generatedFrom']}> |",
        f"| 上游 SHA-256 | `{metadata['sourceSha256']}` |",
        f"| 生成物 SHA-256 | `{metadata['generatedSha256']}` |",
        f"| 产品族 | {_cell(product_families)} |",
        f"| Catalog XID 规则 | {len(rules)} |",
        f"| NVLink5 解码规则 | {len(decode_rules)} |",
        f"| Resolution Bucket | {len(resolution_buckets)} |",
        f"| Marker TTL | {defaults['markerTtlSeconds']} 秒 |",
        (f"| Companion 关联窗口 | {correlation['companionWindowSeconds']} 秒 |"),
        f"| NVLink5 驱动边界 | R{nvlink5['driverBoundary']} |",
        "",
        "## 关键 XID 的官方字段",
        "",
        (
            "下表只压缩展示正式 catalog 中的字段，不重新解释或覆盖 NVIDIA "
            "Immediate Action。"
        ),
        "",
        "| XID | Mnemonic | 产品 | Immediate Action | Investigatory Action "
        "| XID 154 关联 |",
        "|---:|---|---|---|---|---|",
    ]
    for xid in SUMMARY_XIDS:
        rule = rules_by_xid[xid]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(xid),
                    _cell(rule.get("mnemonic")),
                    _cell(rule.get("products")),
                    _cell(rule.get("immediateAction")),
                    _cell(rule.get("investigatoryAction")),
                    _cell(rule.get("xid154Linkage")),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## XID 144–150 解码覆盖",
            "",
            (
                "运行时使用完整 32 位 V1/V2 `IntrInfo` pattern、`Error Status` "
                "以及可选的 `Action2`，不是按少量 subcode bit 猜测动作。"
            ),
            "",
            "| XID | 解码规则数 | Catalog 动作集合 |",
            "|---:|---:|---|",
        ]
    )
    for xid in range(144, 151):
        xid_rules = decode_by_xid[xid]
        actions = {
            str(rule["recoveryAction"])
            for rule in xid_rules
            if rule.get("recoveryAction")
        }
        actions.update(
            str(rule["action2"]) for rule in xid_rules if rule.get("action2")
        )
        lines.append(f"| {xid} | {len(xid_rules)} | {_cell(sorted(actions))} |")

    lines.extend(
        [
            "",
            "## 运行时边界",
            "",
            (
                "- 未知 XID、证据不足或尚无执行器时的 `QUARANTINE` 是控制面"
                " fail-closed 安全行为，不是 NVIDIA catalog 字段。"
            ),
            (
                "- 动作预算、排空超时、证据门禁和恢复验证由控制面配置及"
                " operation registry 管理，不从 XID catalog 读取。"
            ),
            (
                "- `GPU_FAULT_XID_POLICY_PATH` 只接受与上述正式 catalog "
                "schema 相同的完整、经审核文件。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Render a Markdown summary from the pinned NVIDIA XID catalog"
    )
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        document = load_catalog(args.catalog)
        summary = render_summary(
            document,
            source_label=args.source_label,
        )
    except CatalogSummaryError as exc:
        raise SystemExit(f"render-xid-catalog-summary: {exc}") from exc
    print(summary)


if __name__ == "__main__":
    main()
