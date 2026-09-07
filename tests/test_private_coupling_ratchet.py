from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check-test-private-coupling.py"
MODULE = lazy_script_module(MODULE_PATH)


def collect(source: str) -> Counter[str]:
    collector = MODULE.PrivateAccessCollector("tests/probe.py")
    collector.visit(ast.parse(source))
    return collector.accesses


def test_private_names_are_counted_and_self_is_not() -> None:
    accesses = collect(
        "def test_case(self, module):\n"
        "    self._own = 1\n"
        "    cls._own = 1\n"
        "    return module._helper() + module._helper()\n"
    )

    assert accesses == Counter({"tests/probe.py:module._helper": 2})


def test_namespace_bypass_dunders_are_counted() -> None:
    """``__globals__`` 这类 dunder 必须计入，否则它是绕过这道闸门的现成缺口。

    ``module._helper`` 会被拦，改写 ``module.helper.__globals__["dependency"]``
    却一直是免费的——而后者更糟：它把被调方自己的命名空间换掉，被测函数哪天不再
    去查那个名字了，用例照样绿。``tests/_script_loader.py`` 的
    ``__setattr__`` 转发就是为了把用例从这个写法上撬走的，只是当时没人守住。
    """
    accesses = collect(
        "def test_case(module):\n"
        '    module.run.__globals__["dependency"] = 1\n'
        "    handle.run.__globals__\n"
        "    module.decorated.__wrapped__()\n"
        "    module.compiled.__code__ = other\n"
    )

    assert accesses == Counter(
        {
            # 同一个耦合从两个 handle 看到，算在一条上。
            "tests/probe.py:run.__globals__": 2,
            "tests/probe.py:decorated.__wrapped__": 1,
            "tests/probe.py:compiled.__code__": 1,
        }
    )


def test_ordinary_dunders_stay_out_of_the_ratchet() -> None:
    """否则 ``__class__``/``__dict__``/``__name__`` 会把基线灌成噪声。"""
    accesses = collect(
        "def test_case(value):\n"
        "    return (value.__class__, value.__dict__, value.__name__)\n"
    )

    assert accesses == Counter()


def test_growth_slack_and_removal_are_all_reported() -> None:
    baseline = {"tests/probe.py:module._helper": 2}

    assert MODULE.failures(baseline, Counter({"tests/probe.py:module._helper": 3})) == [
        "private test coupling grew: tests/probe.py:module._helper 2 -> 3"
    ]
    assert MODULE.failures(baseline, Counter({"tests/probe.py:module._helper": 1})) == [
        "private coupling baseline has slack: tests/probe.py:module._helper "
        "2 -> 1; run --write-baseline"
    ]
    assert MODULE.failures(baseline, Counter()) == [
        "stale private coupling baseline entry: tests/probe.py:module._helper"
    ]
    assert MODULE.failures({}, Counter({"tests/probe.py:module._helper": 1})) == [
        "new private test coupling: tests/probe.py:module._helper (1)"
    ]


def test_the_loader_contract_is_the_only_baselined_globals_access() -> None:
    """``__globals__`` 只有一处合法：证明加载器给出的是同一个模块对象。

    这条断言是这道闸门的意义所在——多出任何一处 ``__globals__``，
    ratchet 会先红，得由人来解释为什么它不该改成
    ``monkeypatch.setattr(module, ...)``。
    """
    baseline = json.loads(MODULE.BASELINE.read_text(encoding="utf-8"))
    entries = {key for key in baseline if key.endswith(".__globals__")}

    assert entries == {
        "tests/test_script_loader_contract.py:file_set_identity.__globals__"
    }
