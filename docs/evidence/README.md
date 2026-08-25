# Evidence boundary

本目录分为两种完全不同的证据面：

| Path | Versioned | Purpose | Authority |
|---|---|---|---|
| `fault/` | Yes | 公开、脱敏后的 fault evidence schema；`reports: []` 是正常状态 | `manifest.yaml` 与生成的 `index.json` |
| `perf/` | No | 私有本地性能证据，可能包含集群名、节点或实例标识 | 每个 run 的 `run.json`、`status.json` 与 `summary.json` |

原始 fault runner 输出只能进入 `artifacts/fault/`、CI artifact 或访问受控的
私有证据库，不得复制到本目录。`fault/` 只保存公开 schema，不保存真实报告、
执行时间或现场 notes。

## Performance files

`perf/` 被 `.gitignore` 排除，永不进入公开仓库。标准 run 目录中的文件由
`scripts/perf/regional_capacity_suite.py` 生成：

- `run.json`：用例、规模、release 与启动时间；
- `status.json`：`ok` 或 `aborted` 以及原因；
- `summary.json`：最终容量和延迟结论；
- `cgroup-*.json`、`postgres-*.json`、`aurora.json`：资源与数据库采样；
- `queue-drain.json`、`processor-priority-latency.json`：队列和优先级延迟。

`*-inflight*.json` 与相应 summary 由
`scripts/perf/capture_processor_inflight.py` 生成。

## Conclusion snapshots

以下私有 Markdown 是人工整理的结论快照，不是公开规格：

- `perf/regional-processor-concurrency-live-20260729.md`：记录聚合窗口和跨故障
  并发问题；产生于标准 run 布局建立之前，没有可一一对应的规范化 run 目录。
- `perf/regional-processor-concurrency-live-20260730.md`：记录 active-active claim、
  RESET_GPU 调度和节点侧 quiesce 结论；同样属于标准布局前的快照。
- `perf/spool-comparison-20260819.md`：比较 spool enabled/disabled 的 32/50 集群
  与单超大集群结果；对应 `perf/spool-{enabled,disabled}-*/` 和
  `perf/single-cluster-1000n-*/` run 目录。
- `perf/spool-comparison-20260825.md`：release `02d7428d56b3` 的 32/50
  集群 spool enabled/disabled 四组同步突发复测。

## Isolated artifacts

以下文件无法从内容安全地归入某个现有 run，因此保留在私有 `perf/` 根并显式登记：

- `cap003-claim-baseline-20260820.json`：多阶段 claim baseline；内容只有
  release、阶段时间与容量数据，没有 suite/run 关联字段。
- `handler-phase-canary-32c-256-inflight-summary.json`：只记录源 JSONL 名称和
  观测窗口；当前没有同名或同时间的规范化 run 目录。

`notification-fix-capacity-50c-1562-inflight-summary.json` 已归入
`perf/notification-fix-capacity-50c-1562-20260818T100733Z/inflight-summary.json`，
因为 case、并发和观测时间均与该 run 一致。
