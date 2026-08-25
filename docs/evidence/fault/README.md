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
凭据脱敏。公开 catalog 只保留 verdict，不保存报告路径、执行时间或现场 notes。

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
