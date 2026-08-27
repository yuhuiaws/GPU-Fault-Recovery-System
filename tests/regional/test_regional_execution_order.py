from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs/区域模式端到端验收测试用例.md"
ORDER = ROOT / "testcases/regional-execution-order.yaml"
CATALOG = ROOT / "testcases/fault-scenarios.yaml"
CASE_HEADING = re.compile(r"^### (GF-REGIONAL-[A-Z0-9-]+)(?:[：:].*)?$", re.MULTILINE)
CHAPTER_HEADING = re.compile(r"^## [\d.]+ ([A-Z0-9]+)[：:]", re.MULTILINE)
CASE_PREFIX_HEADING = re.compile(r"^### GF-REGIONAL-([A-Z0-9]+)-\d+", re.MULTILINE)
DOMAIN_TABLE_HEADING = "| 域 | 覆盖内容 |"
DOMAIN_ROW = re.compile(r"^\| `([A-Z0-9]+)` \|")
PHASE_SECTION_HEADING = "## 17. 建议执行顺序"
PHASE_TABLE_HEADING = "| 阶段 | 严格执行顺序 |"
PHASE_NUMBER = re.compile(r"第 ?\d+ ?阶段|阶段 ?\d+")
MAINTENANCE_WINDOW_VALUES = {"否", "建议", "是", "分级"}
# 判定口径的唯一写法。历史上同一个角色有四种标签——判定 / 通用判定 /
# 判定层要素 / 断言 / 期望——读者要先猜哪一段才是"过不过"的依据。
# 标签后必须紧跟 ：或（，否则「判定层要素」这种以「判定」开头、
# 说的却是"被测对象有哪些分支"的小标题也会被算成判定口径。
VERDICT_LABEL = re.compile(
    r"^(?:\s*[-*] 判定[（(：:]"
    r"|\*\*判定\*\*[（(：:]"
    r"|##### 判定)",
    re.MULTILINE,
)


def _phase_section(text: str) -> tuple[int, int]:
    start = text.index(PHASE_SECTION_HEADING)
    next_section = text.find("\n## ", start + len(PHASE_SECTION_HEADING))
    return start, len(text) if next_section < 0 else next_section


def _documented_phases(text: str) -> list[tuple[int, str, str]]:
    start, end = _phase_section(text)
    section = text[start:end]
    table = section[section.index(PHASE_TABLE_HEADING) :]
    result = []
    for line in table.splitlines():
        if not line.startswith("|"):
            break
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        head = columns[0].split(" ", 1)
        if not head[0].isdigit():
            continue
        window = columns[2].replace("*", "").split("（", 1)[0].strip()
        result.append((int(head[0]), head[1].strip(), window))
    return result


def _yaml_phases(value: dict) -> list[tuple[int, str, str]]:
    return [
        (phase["sequence"], phase["name"], phase["maintenance_window"])
        for phase in sorted(value["phases"], key=lambda item: item["sequence"])
    ]


def _ordered_cases(value: dict) -> list[str]:
    result = []
    phases = sorted(value["phases"], key=lambda item: item["sequence"])
    assert [item["sequence"] for item in phases] == list(range(len(phases)))
    for phase in phases:
        for entry in phase["entries"]:
            if "case" in entry:
                result.append(entry["case"])
                continue
            item = entry["range"]
            result.extend(
                f"GF-REGIONAL-{item['prefix']}-{number:03d}"
                for number in range(item["start"], item["end"] + 1)
            )
    return result


def test_regional_execution_order_covers_every_documented_case() -> None:
    document_cases = CASE_HEADING.findall(DOCUMENT.read_text(encoding="utf-8"))
    value = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    ordered = _ordered_cases(value)
    do_not_run = [item["case"] for item in value["do_not_run"]]
    indexed = [*ordered, *do_not_run]
    assert len(indexed) == len(set(indexed))
    assert set(indexed) == set(document_cases)
    assert len(indexed) == len(document_cases) == 153


def test_do_not_run_matches_regional_superseded_cases() -> None:
    order = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    cases = {
        case["id"]: case
        for case in catalog["test_cases"]
        if case["id"].startswith("GF-REGIONAL-")
    }
    retired = {item["case"] for item in order["do_not_run"]}
    superseded = {
        case_id
        for case_id, case in cases.items()
        if (case.get("evidence") or {}).get("verdict") == "SUPERSEDED"
    }

    assert retired == superseded == {"GF-REGIONAL-DESTR-004"}
    case = cases["GF-REGIONAL-DESTR-004"]
    assert case["automation"] == "manual"
    assert case["risk"] == "destructive"
    assert case["superseded_by"] == "GF-REGIONAL-DESTR-013"


def test_regional_execution_order_pins_special_dependencies() -> None:
    value = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    ordered = _ordered_cases(value)
    position = {case: index for index, case in enumerate(ordered)}
    assert position["GF-REGIONAL-BOOT-011"] < position["GF-REGIONAL-BOOT-001"]
    assert position["GF-REGIONAL-WORKLOAD-001"] < position["GF-REGIONAL-ISO-001"]
    assert position["GF-REGIONAL-WORKLOAD-001"] < position["GF-REGIONAL-CMD-015"]
    assert position["GF-REGIONAL-WORKLOAD-001"] < position["GF-REGIONAL-PREEMPT-012"]
    assert position["GF-REGIONAL-WORKLOAD-001"] < position["GF-REGIONAL-E2E-001"]
    assert position["GF-REGIONAL-E2E-001"] < position["GF-REGIONAL-BLAST-001"]
    assert position["GF-REGIONAL-E2E-001"] < position["GF-REGIONAL-NOTIFY-001"]
    assert position["GF-REGIONAL-DESTR-011"] < position["GF-REGIONAL-DESTR-001"]
    assert position["GF-REGIONAL-DESTR-009"] < position["GF-REGIONAL-DESTR-012"]
    assert position["GF-REGIONAL-HA-003"] < position["GF-REGIONAL-DESTR-013"]
    assert ordered[-1] == "GF-REGIONAL-COLLECT-015"


def test_every_case_is_defined_in_the_chapter_for_its_prefix() -> None:
    # 用例必须定义在自己前缀的章节里。前缀不是写死的映射表，是从章标题
    # `## N. PREFIX：…` 自己读出来的，所以新开一章自动纳入守卫。
    # 挡的是实际发生过的两类走位：六条新用例定义在「测试覆盖审计」附录里
    # （读者按 §4/§5/§8/§10/§13 找永远找不到），以及 WORKLOAD 两条夹在
    # DESTR-008 与 DESTR-009 中间，把破坏性章节劈成两半。
    chapter: str | None = None
    misplaced = []
    for index, line in enumerate(
        DOCUMENT.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith("## "):
            heading = CHAPTER_HEADING.match(line)
            chapter = heading.group(1) if heading else None
            continue
        case = CASE_PREFIX_HEADING.match(line)
        if case is not None and case.group(1) != chapter:
            misplaced.append(f"{index}: {line.strip()} 在 {chapter} 章下")

    assert misplaced == []


def _case_bodies(text: str) -> dict[str, str]:
    lines = text.splitlines()
    starts: list[tuple[int, str | None]] = []
    for index, line in enumerate(lines):
        heading = CASE_HEADING.match(line)
        if heading is not None:
            starts.append((index, heading.group(1)))
        elif line.startswith("### ") or line.startswith("## "):
            starts.append((index, None))
    bodies = {}
    for position, (index, case_id) in enumerate(starts):
        if case_id is None:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        bodies[case_id] = "\n".join(lines[index + 1 : end])
    return bodies


def test_every_runnable_case_states_its_verdict_under_one_label() -> None:
    # 每条要跑的用例必须有且只有一种"过不过"的标签：判定。
    # 执行人读一条用例时最需要的就是这一段，而它曾经有四种写法
    # （判定 / 每个场景的通用判定 / 判定层要素 / 光秃秃的"断言…"），
    # 其中 37 条根本搜不到「判定」二字。DO_NOT_RUN 的用例例外：
    # 它们只保留编号与作废理由，写判定反而像还能执行。
    text = DOCUMENT.read_text(encoding="utf-8")
    value = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    retired = {item["case"] for item in value["do_not_run"]}
    bodies = _case_bodies(text)
    missing = sorted(
        case_id
        for case_id, body in bodies.items()
        if case_id not in retired and VERDICT_LABEL.search(body) is None
    )

    assert missing == []
    assert all(VERDICT_LABEL.search(bodies[case_id]) is None for case_id in retired)


def test_domain_table_lists_exactly_the_domains_that_exist() -> None:
    # §1 的域表就是全手册的图例：读者拿到 `GF-REGIONAL-COLLECT-004`
    # 只能靠它知道 COLLECT 是什么、对应哪条上线门禁。三个集合必须相等：
    # 图例里写的域、真实存在的章节、真实存在的用例前缀。
    # 挡的是实际发生过的漂移：新开了 8.5/11.5/13.5 三章、补了 50 条用例，
    # 但域表还停在 11 行，PREEMPT/WORKLOAD/COLLECT 三个域没有任何图例。
    text = DOCUMENT.read_text(encoding="utf-8")
    table = text[text.index(DOMAIN_TABLE_HEADING) :]
    documented = []
    for line in table.splitlines():
        if not line.startswith("|"):
            break
        row = DOMAIN_ROW.match(line)
        if row is not None:
            documented.append(row.group(1))

    chapters = CHAPTER_HEADING.findall(text)
    prefixes = CASE_PREFIX_HEADING.findall(text)

    assert documented == sorted(set(documented), key=documented.index)
    assert set(documented) == set(chapters)
    assert set(documented) == set(prefixes)


def test_document_phase_table_matches_the_order_yaml() -> None:
    # 阶段号、阶段名与维护窗口只有一份来源（本 YAML）。§17 的表是它的
    # 人读投影，两边必须逐行相等——否则读者按表执行、脚本按 YAML 生成
    # 索引，两个结论会不一致（曾经就是：表里阶段 12 写「是」，YAML 没
    # 声明，生成的索引印成「否」，把维护窗口要求印丢了）。
    value = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    documented = _documented_phases(DOCUMENT.read_text(encoding="utf-8"))

    assert documented == _yaml_phases(value)
    assert {item[2] for item in documented} <= (MAINTENANCE_WINDOW_VALUES)


def test_phase_numbers_appear_only_in_the_phase_table() -> None:
    # 正文一律用阶段名，数字只出现在 §17 那张表里。理由是实测的：
    # DESTR 三个阶段从 11/12/13 变成 13/14/15 之后，正文 21 处编号
    # 全部指错了阶段（"阶段 11 内部 DESTR-011 最先跑"——而当时的
    # 阶段 11 已经是非破坏性 HA）。只要正文不写数字，这类漂移不可能发生。
    text = DOCUMENT.read_text(encoding="utf-8")
    start, end = _phase_section(text)
    offending = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not PHASE_NUMBER.search(line):
            continue
        offset = text.index(line)
        if start <= offset < end and line.startswith("|"):
            continue
        offending.append(f"{index}: {line.strip()}")

    assert offending == []
