# BOOT-001~010 一次性守卫探针夹具

`docs/区域模式端到端验收测试用例.md` §4.0 的可执行实现。

**为什么这些脚本在仓库里、而不是像早期那样临时写进 `/tmp`**：
2026-08-04 的执行中，机器在跑完 `BOOT-009` 之后重启，`/tmp` 被整体清空。
夹具脚本、派生清单、kubeconfig 全部丢失，而集群上的探针 Deployment
与临时 Secret **仍然留着**——探针在 `CrashLoopBackOff` 里空转了 8 小时、
重启 102 次。清理流程如果依赖只存在于 `/tmp` 的脚本，
就会在最需要它的时候不可用。所以：

- 判定与清理脚本一律放仓库，随代码走；
- 只有**派生产物**（`guard-probe-base.json`）和 kubeconfig 允许放 `/tmp`，
  因为它们都能用 `derive.sh` / `aws eks update-kubeconfig` 一条命令重建；
- `cleanup.sh` 从受检目标重新读资源并核对身份；读取失败不能作为资源不存在，
  必须先证明探针 Pod 消失，再清理独立数据库。不得把生产 DSN 挂载到派生探针。

## 文件

| 文件 | 作用 |
| --- | --- |
| `derive.sh` | 从生产 Deployment 派生探针基线清单到 `/tmp/guard-probe-base.json`，并做 6 项断言 |
| `mutate.py` | 单变量变形器（`del` / `set` / `sref`） |
| `registry.py` | 生成探针用的 registry JSON（token 长度、`enabled`、缺字段、拼错字段） |
| `reset.sh` | 删探针并**轮询到 Pod 数归零**（`delete --wait` 不等 Pod） |
| `assert.sh` | 轮询判定"不 Ready + 日志含指定文本"，label 下 >1 个 Pod 时拒绝判定 |
| `cleanup.sh` | 全组跑完后的受检清理；资源读取、删除和数据库清理错误均传播 |
| `../boot_guard_control.py` | 绑定运行目录、前置证据、目标及 schema 3 的计划与执行授权 |
| `../boot_guard_isolation.py` | 清除继承的数据库路由覆盖，先验证实际数据库身份再允许探针 DDL |

## 用法

从仓库根目录使用整组受检入口。先按站点准备并导出 `CPU_KUBECONFIG`、
`NAMESPACE`、`CPU_HYPERPOD_CLUSTER`、`AWS_REGION`、`RUN_DIR` 和 `PYTHON`；
不从示例 ARN、当前 context 或默认 Region 推断目标。

```bash
bash scripts/e2e/regional/run_regional_boot_guard_cases.sh \
  --run-dir "${RUN_DIR}" --plan
bash scripts/e2e/regional/run_regional_boot_guard_cases.sh \
  --run-dir "${RUN_DIR}" --execute --confirm BOOT_GUARD_EXECUTE \
  --maintenance-window-end "${MAINTENANCE_WINDOW_END}"
```

`RUN_DIR` 必须与 `--run-dir` 相同；从 7 或 8 续跑需要为同一
`BOOT_GUARD_START_CASE` 重新计划。单独的 derive/reset/mutate/assert 是编排内部步骤，
不能绕过目标、前置证据和维护窗口检查直接使用。

P2/P3 在 CPU Pod 内读取当前凭据，派生独立数据库及只读 DSN 挂载。
探针不继承 `GPU_FAULT_STORE_URL_FILE` 或 libpq 路由覆盖；实际数据库身份不符时，
必须在 schema ensure 前拒绝，不能仅凭新 DSN 字符串宣称隔离成功。
