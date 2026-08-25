local document_ids = {
  ["README.md"] = "main-design",
  ["docs/README.md"] = "docs-index",
  ["概要设计.md"] = "overview-design",
  ["详细设计.md"] = "detailed-design",
  ["nvidia-policy.md"] = "nvidia-policy",
  ["部署和运维手册.md"] = "operations-manual",
  ["部署和运维手册逐章解读.md"] = "operations-manual-guide",
  ["扩展指南.md"] = "extension-guide",
  ["环境变量参考.md"] = "environment-reference",
  ["故障模拟测试手册.md"] = "fault-simulation-manual",
  ["区域用例索引.md"] = "regional-case-index",
  ["区域模式端到端验收测试用例.md"] = "regional-acceptance-cases",
  ["性能压测验收方案.md"] = "performance-acceptance",
  ["validation-limitations.md"] = "validation-limitations",
}

local title_ids = {
  ["GPU 多机多卡训练故障自动化处理设计"] = "main-design",
  ["文档索引"] = "docs-index",
  ["GPU 故障处理系统概要设计"] = "overview-design",
  ["GPU 故障处理系统详细设计"] = "detailed-design",
  ["NVIDIA 官方策略实现审计"] = "nvidia-policy",
  ["GPU 故障处理系统部署和运维手册"] = "operations-manual",
  ["GPU 故障处理系统《部署和运维手册》逐章解读"] = "operations-manual-guide",
  ["GPU Fault 扩展指南"] = "extension-guide",
  ["环境变量参考"] = "environment-reference",
  ["GPU 故障模拟测试手册"] = "fault-simulation-manual",
  ["区域分离部署用例索引（按执行顺序）"] = "regional-case-index",
  ["区域模式端到端验收测试用例"] = "regional-acceptance-cases",
  ["GPU故障恢复控制面性能压测验收方案"] = "performance-acceptance",
  ["Validation limitations"] = "validation-limitations",
  ["附录：固定版本 NVIDIA XID Catalog 摘要"] = "nvidia-xid-catalog-summary",
}

local function basename(path)
  return path:gsub("\\", "/"):match("([^/]+)$")
end

local function normalize_path(path)
  local result = path:gsub("\\", "/"):gsub("^%./", "")
  while result:match("^%.%./") do
    result = result:sub(4)
  end
  return result
end

local function document_id(path)
  local normalized = normalize_path(path)
  local direct = document_ids[normalized]
  if direct then
    return direct
  end
  if normalized:match("^docs/") or normalized:match("^security/") then
    return document_ids[basename(normalized)]
  end
  if not normalized:match("/") then
    return document_ids[normalized]
  end
  return nil
end

local function external_target(path)
  local normalized = normalize_path(path)
  if normalized:match("^history/")
      or normalized:match("^reference/")
      or normalized:match("^evidence/")
      or normalized:match("^security/") then
    return "docs/" .. normalized
  end
  return normalized
end

local function html_escape(value)
  return value
    :gsub("&", "&amp;")
    :gsub("<", "&lt;")
    :gsub(">", "&gt;")
end

function Header(element)
  local title = pandoc.utils.stringify(element.content)
  local id = title_ids[title]
  if id then
    element.identifier = id
    return element
  end

  local stripped = title:gsub("^%d+[%.%d]*%s+", "")
  if stripped ~= title then
    local parsed = pandoc.read("# " .. stripped, "markdown")
    if parsed.blocks[1] and parsed.blocks[1].t == "Header" then
      element.content = parsed.blocks[1].content
    end
  end
  return element
end

function Link(element)
  local file, fragment = element.target:match("^([^#]+%.md)(#.*)$")
  if file and fragment then
    if document_id(file) then
      element.target = fragment
    else
      element.target = external_target(file) .. fragment
    end
    return element
  end

  file = element.target:match("^([^#]+%.md)$")
  if file then
    local id = document_id(file)
    if id then
      element.target = "#" .. id
    else
      element.target = external_target(file)
    end
  end
  return element
end

function CodeBlock(element)
  if FORMAT:match("html") and element.classes:includes("mermaid") then
    return pandoc.RawBlock(
      "html",
      '<pre class="mermaid">' .. html_escape(element.text) .. "</pre>"
    )
  end
  return element
end
