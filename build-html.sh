#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

mermaid_runtime="html-assets/mermaid.min.js"
mermaid_version="11.16.0"
mermaid_sha256="74d7c46dabca328c2294733910a8aa1ed0c37451776e8d5295da38a2b758fb9b"
mermaid_url="https://cdn.jsdelivr.net/npm/mermaid@${mermaid_version}/dist/mermaid.min.js"
output="GPU_FAILURE_AUTOMATION_DESIGN.html"
python_bin="${PYTHON:-python3}"
xid_catalog_summary="$(
  mktemp "${TMPDIR:-/tmp}/gpu-fault-xid-summary.XXXXXX.md"
)"

cleanup() {
  rm -f "$xid_catalog_summary"
}
trap cleanup EXIT

if [[ ! -s "$mermaid_runtime" ]] ||
  [[ "$(sha256sum "$mermaid_runtime" | awk '{print $1}')" != "$mermaid_sha256" ]]; then
  command -v curl >/dev/null || {
    printf 'curl is required to fetch Mermaid %s\n' "$mermaid_version" >&2
    exit 1
  }
  mkdir -p "$(dirname "$mermaid_runtime")"
  mermaid_tmp="${mermaid_runtime}.tmp"
  curl -fsSL "$mermaid_url" -o "$mermaid_tmp"
  printf '%s  %s\n' "$mermaid_sha256" "$mermaid_tmp" | sha256sum -c -
  mv "$mermaid_tmp" "$mermaid_runtime"
fi

PYTHONPATH="${root_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$python_bin" scripts/render_xid_catalog_summary.py \
  --catalog src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml \
  >"$xid_catalog_summary"

pandoc \
  README.md \
  docs/README.md \
  docs/概要设计.md \
  docs/详细设计.md \
  docs/components/nvidia-policy.md \
  docs/管理员快速部署.md \
  docs/管理员Profile变更审批.md \
  docs/管理员日常运维.md \
  docs/安全与参数参考.md \
  docs/部署和运维手册.md \
  docs/部署和运维手册逐章解读.md \
  docs/开发者部署实现.md \
  docs/扩展指南.md \
  docs/管理员环境变量参考.md \
  docs/环境变量参考.md \
  docs/故障模拟测试手册.md \
  docs/故障类别与处置动作总表.md \
  docs/区域用例索引.md \
  docs/区域模式端到端验收测试用例.md \
  docs/性能压测验收方案.md \
  docs/validation-limitations.md \
  "$xid_catalog_summary" \
  --from=gfm \
  --to=html5 \
  --standalone \
  --embed-resources \
  --resource-path=.:docs \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --lua-filter=html-build.lua \
  --include-in-header=html-header.html \
  --variable=lang:zh-CN \
  --metadata=title:"GPU 故障处理设计、部署、运维与验收文档集" \
  --metadata=subtitle:"区域 CPU 控制面与 HyperPod EKS GPU 数据面分离方案" \
  --output="$output"

printf 'Generated %s\n' "$output"
