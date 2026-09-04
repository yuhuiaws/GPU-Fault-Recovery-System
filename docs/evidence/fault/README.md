# Curated fault evidence

公开仓库不保存真实环境故障报告。临时 runner 输出、全量 XID replay、调试 dump、
失败重跑和长期证据都进入 `artifacts/fault/`、CI artifact 或访问受控的私有证据库。

本目录只保留公开 evidence schema：

- `manifest.yaml`：公开报告清单，默认 `reports: []`；
- `index.json`：由 `scripts/build-fault-evidence-index.py` 生成；
- 本 README。

标准 verdict 只有：

- `PASS`
- `PASS_WITH_LIMITATIONS`
- `FAIL`
- `PARTIAL`
- `INVALID`

报告级 verdict 与 catalog 用例级 `evidence.verdict` 的合法关系如下：

| Report verdict | Allowed case verdicts |
|---|---|
| `PASS` / `PASS_WITH_LIMITATIONS` | 所有关联用例均为 `PASS` |
| `FAIL` | `BLOCKED` 或 `NOT_RUN` |
| `PARTIAL` | `PASS`、`BLOCKED` 或 `NOT_RUN` |
| `INVALID` | `BLOCKED`、`NOT_RUN` 或 `SUPERSEDED` |

私有证据进入长期存储前必须完成账号 ID、ARN、桶名、集群名、节点 ID、内部地址和
凭据脱敏。公开 catalog 只保留 verdict 和下面的摘要绑定，不保存报告路径、执行时间
或现场 notes。

## PASS 的摘要绑定

`PASS` 是唯一会被读者直接当作结论的 verdict，也是唯一无法由门禁复现的 verdict：
live 用例只在维护窗口对真实 GPU 执行一次。没有绑定时这样一条 `PASS` 是永久的——
它既能在用例文本被改写后继续为没人执行过的断言背书，也会随代码一路发布下去。
因此每条 `PASS` 必须带：

```yaml
evidence:
  verdict: PASS
  verified:
    case_digest: <sha256>
    components:
      control_plane: <sha256>
      executor: <sha256>
      node_runtime: <sha256>
```

- `case_digest` 是这条用例规范性字段的 sha256（不含 `evidence` 和 `execution`）。
  不匹配是硬失败：改写了断言就要重新执行，不能让旧结论跟着新文本走。
- `components` 是 `scripts/component_wheels.py` 的组件源码摘要。三个组件全部记录：
  live 用例都是端到端链路，信号从节点经 executor 到控制面。
- 只允许摘要。执行时间、集群名、节点 ID、操作人和报告路径仍然不进 catalog——
  摘要能标定一份代码，却不泄露它运行在哪个环境里，这也是本节能存在的唯一原因。

`scripts/build-fault-evidence-index.py` 把每条 `PASS` 分成三类：

| 状态 | 含义 | 处置 |
|---|---|---|
| `FRESH` | 绑定的组件摘要等于当前代码 | 无 |
| `STALE` | 组件摘要已漂移 | 下个维护窗口重新执行 |
| `UNBOUND` | `PASS` 早于本机制，没有组件摘要 | 下个维护窗口重新执行并补绑定 |

`STALE` 只被推导、不被存储，所以没人能靠在 YAML 里写一个词让漂移的结论继续有效；
它也刻意不是硬失败，否则每次发布都要先做真机执行才能过 CI，反而会逼出假证据。
`UNBOUND` 由 `tools/run_fault_test_cases.py` 的 `EVIDENCE_UNBOUND_PASS_CASES`
单向棘轮收口：名单只允许变短，名单里的用例也不允许事后补一个「今天」的摘要，
因为那等于把没有做过的复核写成证据。`index.json` 只记录哪些 `PASS` 有绑定，不记录
摘要本身——摘要会随任何源码改动变化，写进去会让 `--check` 永久 diff。

## Workflow

```bash
# 原始输出
python3 tools/run_fault_test_cases.py

# 公开树只生成空或已批准的脱敏索引
python3 scripts/build-fault-evidence-index.py

# CI/提交门禁
python3 scripts/build-fault-evidence-index.py --check
```

不得手工修改 `index.json`。
