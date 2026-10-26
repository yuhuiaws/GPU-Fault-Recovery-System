# GPU 环境故障真机验收

## 当前覆盖

现有高温、Power Brake 和普通功率限制用例经过真实 parser、API、policy、workflow
executor，但设备操作、Kubernetes 和工作负载边界为替身，不是已通过的真机验收。
本轮补充绝对设备温度、T.Limit、显存、绝对关机温度和热 warning 复合的软件执行用例。
以前的 `GF-REGIONAL-COLLECT-002` 主要证明功率异常投递，不能替代停训、隔离和人工
复核链路的真实回执。

以下 12 项独立登记为 `GF-ENV-LIVE-001..012`，当前全部 **manual / NOT_RUN**。
不能把本文、回放工具测试通过或以前版本的现场记录改写为本版本的 LIVE PASS。
判据以[环境策略复核](gpu-environment-policy-review.md)及当前源码为准。

其他通用故障的详细真机用例和32类完整映射见
[通用故障真机验收补充](general-fault-live-acceptance.md)。

| 编号 | 主验证目标 | 注入性质 |
|---|---|---|
| 001 | 数据源、能力与健康基线 | 真机原始观测 |
| 002 | 正常 power cap 不触发硬件恢复 | 真机受控功率限制及负载 |
| 003 | 两种温标的 warning 与仅软件工作限值 | 真机遥测回放 |
| 004 | GPU/显存持续严重温度到隔离、停训、人工复核 | 真机遥测回放 |
| 005 | 瞬时异常、清除、重复记录不凑成三次确认 | 真机遥测回放 |
| 006 | thermal duration/复合 warning 不越权 | 真机遥测回放 |
| 007 | Power Brake 与普通降频位区分 | 真机遥测回放 |
| 008 | N/A、字段缺失与新旧别名 | 真机原始观测及遥测回放 |
| 009 | 采集间断、collector 重启与来源切换 | 真机受控服务操作 |
| 010 | GPU、节点、任务作用域与通知 | 多 GPU 真机遥测回放 |
| 011 | 绝对关机温度越界不等待三次 | 真机遥测回放，禁止物理加热 |
| 012 | 真正供电/散热故障及修复后检查 | 自然故障或厂商批准的实验室验证 |

“真机遥测回放”只证明正常安装的 collector、真实网络/数据库/控制面、真实调度与
测试任务操作能够贯通；不证明传感器真的出现该故障，也不证明硬件故障已经修复。

## 共同前置条件

1. 使用独立测试集群/节点池和可丢弃训练任务；不得使用业务训练节点。
   保存审批、维护窗口、节点/GPU allowlist、目标 job/attempt/Pod UID、
   Runtime Profile 和允许动作。确认无法授权测试范围外的节点/任务。
2. 固定制品、源码、驱动、DCGM、exporter、GPU 型号、物理 UUID/BDF、boot ID。
   保存真正运行进程的配置，不仅是磁盘上的 env 文件。清点旧 THERMAL_COOLDOWN
   工作流并先完成审核；不得让其自动重新接纳污染本轮结论。
3. 从真实 `nvidia-smi -q -x` 和 exporter 读取原始数据。温度与 T.Limit 各自使用
   同刻度阈值；没有对应能力的变体标 NOT_RUN 并说明，不用编造的硬件限值代替。
   nvidia-smi 模式不等同完整 DCGM 能力，源切换必须单独记录。
4. 确认 Node Agent lease、completion watcher 任务归属、通知接收方与时钟同步。
   除测试任务外节点应为空；先运行正常采集至少三个周期，确认没有历史故障、
   未完成硬件动作或未知采集状态。计划动作必须只命中测试对象。
5. 所有阶段记录输入、HTTP/采集时间、batch ID、GPU 身份、finding、
   `confirmation_count`、incident/workflow ID、每一步 execution、真实 provider/
   Kubernetes 回执、任务退出状态和通知结果。202、计划列表、单条日志不是 PASS。
6. 样本有效性以控制面实际接受的 `observed_at`、来源与 boot 身份为准。
   3 个 HTTP GET 不等于控制面接受了 3 个有效样本；相邻严重样本超过 60 秒应重计。
   工作流等待聚合、队列或门禁时计入实测延迟，不修改策略来追求 PASS。
7. 建议单个回放窗口上限 600 秒，工作流执行上限采用实际 step deadline 加传输预算。
   到期、动作结果未知、作用域错误、真实温度越过预定安全线均立即中止并保留证据。
   禁止禁风扇、堵风道、拔电源、绕过驱动温控或为测试提高功率上限。

## 回放工具与接线

`tools/gpu_environment_replay.py` 使用真实 exporter 作为上游，**只改一个物理 GPU
已有的指定字段**。其他 GPU 和指标保持原始含义；不会制造缺失的 T.Limit/限值字段。
计划的 duration 使用内部 us，工具会转换成 exporter 原始 ns。
`null` 表示省略该 GPU 的该字段。工具不会修改 GPU、任务、systemd 或 collector 配置。

工具初始只透传。安装器探测、基线采集都不消耗测试步骤；单独 `/arm` 后才开始。
计划耗尽或到期后恢复新鲜上游透传，不无限重复最后一个故障值。真实上游出现
HW thermal/Power Brake、T.Limit≤0、GPU≥80 C 或显存≥85 C 时，永久中止本次覆盖，
透传真实读数；可将温度安全线调得更低，不能提高。这些安全线不是故障策略阈值。
安全监测不是独立硬件保护，仍须现场人员和带外停止手段。

在已批准的测试节点，使用与待验收制品对应的源码工具及安装了 collectors 依赖的
Python。先只验证计划，下面的命令不启动网络：

```bash
python -m tools.gpu_environment_replay \
  --plan testcases/gpu-environment/power-brake.json \
  --gpu-uuid "$GPU_UUID"
```

确认真实上游为本机 exporter，输出目录属于本次实验且审计文件尚不存在，再启动：

```bash
python -m tools.gpu_environment_replay \
  --plan testcases/gpu-environment/power-brake.json \
  --gpu-uuid "$GPU_UUID" \
  --upstream http://127.0.0.1:9400/metrics --port 9411 \
  --max-seconds 600 --serve --ack ISOLATED_GPU_TEST_NODE \
  --audit "$RUN_DIR/replay.jsonl"
```

通过正常单节点维护/安装流程，将**该测试节点现有且唯一的** metrics collector 的
`--dcgm-metrics-url` 配成 `http://127.0.0.1:9411/metrics`，不改真实 exporter 配置，
不并行启动第二个相同身份的 producer。预先准备恢复原 endpoint/配置摘要的独立
回滚手段。确认几轮透传进入真实控制面且任务归属正确后，再执行：

```bash
curl --fail http://127.0.0.1:9411/status
curl --fail -X POST -H 'X-GFRS-Replay-Ack: ISOLATED_GPU_TEST_NODE' \
  http://127.0.0.1:9411/arm
```

只有目标 collector 读取 `/metrics`；操作员查看 `/status`，避免消耗步骤。
同一实例不能再次 arm；重跑需新审计文件和新的干净测试任务。
审计保存每次源/输出 SHA、覆盖值和是否因真实安全读数中止；源抓取失败返回 503，
不得伪造健康批次，不跟随HTTP重定向或使用继承的HTTP代理。审计的 prepared 记录
不是控制面接收回执。工具不输出 LIVE PASS，结果必须由实际工作流证据判定。

模板 `tlimit-example.json` 的 -3/1/25 只适用于已验证 slowdown=-2 C、warning=3 C
且原始状态安全的对应设备。其他型号按真实同刻度 W/C 生成本轮审阅过的计划，
不要照搬示例数值。工具拒绝修改硬件限值元数据。

exporter 把 `DCGM_FI_DEV_THERMAL_VIOLATION`/`DCGM_FI_DEV_POWER_VIOLATION` 声明为
`counter`（`deploy/dataplane/dcgm-counters.csv`），`prometheus_client` 的文本解析器会把
这类族的样本改名为 `<name>_total`。工具与采集器解析器一样按裸名回退匹配
（`canonical_mapping`，对应 `collectors/gpu/dcgm.parse_dcgm_samples`），变换后的文本仍以
`_total` 名发布，由采集器的同一回退读回。2026-10-26 现网 GF-ENV-LIVE-006 `hw_thermal`
之前因缺少该回退，每次 armed 读取都被工具以 503 拒绝（审计行 "requested fields are
absent"），采集器每个周期记 `cannot scrape DCGM exporter`、代理 `reads` 停在 arm 时计数，
用例以 "consumed 0 of 7" INCONCLUSIVE。`/status` 现在带 `failed_reads`/`last_error`：
被拒绝的读取是有名字的计数，不再只是停住的 `reads`。

## 单项验收

### GF-ENV-LIVE-001

**来源与健康基线。** 读取 `nvidia-smi -q -x`、exporter 原文和控制面的最新指标，
逐 GPU 比对 UUID/BDF、核心/显存温度、T.Limit、原因位、功率及上限。
核对 `source_boot_id`、采集器模式、支持的新旧字段名及至少三次新采样。
开启回放代理但不 arm，验证透传值没有改变，原任务 step 继续前进。
通过条件：必需字段全部有对应读数/明确不支持记录，没有新增环境故障动作。
记录每项实采数/期望数；有遗漏不能因为 collector status 健康就通过。

### GF-ENV-LIVE-002

**真实 power cap。** 复用 `GF-REGIONAL-COLLECT-002` 的受控负载、单 GPU 降限、
先布 deadman 后修改、持久化还原记录及投递证据，另外跟踪完整诊断工作流。
该旧 runner 会恢复默认上限，因此本案例复用它的前提是测试前每张 GPU 上限均为
默认值；不是默认值时 BLOCKED，不能把用户原配置覆盖掉。
只在硬件允许范围内降低上限，不提高上限；负载使用已批准、有限时的工具。
通过条件：若形成本站性能 finding，实际只执行只读诊断；无隔离、STOP、reset、
reboot、RMA；原 attempt/Pod UID 与训练进度保持。若没有形成有效候选，记未触发，
不能算故障分支通过。清理须停止负载、逐卡确认原上限，并验证还原日志。

### GF-ENV-LIVE-003

**warning 与温标。** 绝对温标设备回放 W≤T<C；T.Limit 设备回放 C<T≤W。
W/C 来自本轮真实设备限值和配置。保留其他指标新鲜正常，连续送入至少三个周期，
确认对应温标产生 WARNING/只读诊断，T.Limit 生效时不重复制造绝对 fallback 故障。
仅有 GPU max-operating 的设备单独验证其不会被误作 HW slowdown。
通过条件：真实验证步骤能完成；无 STOP/隔离/重启，原任务持续推进。缺少某种
设备温标的变体明确 NOT_RUN，不在代理中伪造 nvidia-smi 限值。

### GF-ENV-LIVE-004

**持续严重温度。** 分别运行核心绝对温度、T.Limit 和显存三个变体，独立任务/状态。
仅改变目标 GPU 当前读数，越过 C 但不改设备阈值；前两个已接受样本不得 STOP，
第三个样本 `confirmation_count=3` 后进入 ENVIRONMENT_REVIEW。
通过条件：真实 cordon、正确任务 suspend/退出、quarantine taint、诊断包回执、
snapshot 和通知均存在，incident ESCALATED；没有 reset/reboot/RMA/任务重启。
随后恢复真实低温读数并观察超过旧 300 秒冷却时间，节点仍保持隔离。
所有温度都是遥测回放，不允许为了这一分支实际加热硬件。

### GF-ENV-LIVE-005

**瞬时、清除与重复。** 分别发送 C 越界一次后健康、越界两次后健康、两次
warning 后一次严重；每个变体至少再观察三个有效采集周期。
另外重放同一已接受 batch 的投递重试，使用本项目受检重试入口，不自行伪造身份。
通过条件：清除也穿过 edge filter，严重计数重新起算，重复 batch 不推进次数；
不发生 STOP/隔离。持续异常随后再连送三个有效样本，应仍能按004触发。
如果现场没有受检 batch 重试入口，该子变体留为 NOT_RUN。

### GF-ENV-LIVE-006

**duration/复合 warning。** 原始温度保持健康，使用 `thermal-duration.json`
连续增加 duration，单位转换记录为 us→ns；分别叠加 `0x20/0x40`，不得因此 DRAIN。
再在 warning 区间温度下重复该过程，验证 THERMAL_STRESS 仍只诊断，不高于组件。
通过条件：实际执行只读验证，没有压测、STOP、reset/reboot；停止增量并恢复
原因位/温度后能完成诊断。用例不要求硬件累计计数器真正归零。
warning 区间的重复由 GF-GENERAL-LIVE-024 `thermal_warning` 用同一回放工具与探针执行
（`env_live_plans.thermal_stress_plan`，见通用故障真机验收 phase 2 DCGM 节），
功耗复合由 024 `power_performance`（`power_limit_throttling_plan`，不服务 `0x80`）执行。
2026-10-26 现网（plain EKS `ip-10-0-43-5`，R13g）首次执行 `hw_thermal` 为 INCONCLUSIVE
"the collector consumed 0 of 7 planned replay reads"：repoint 证明已成立（首次透传读取
18.1 秒后出现，arm 前 `reads` 0→2），但工具对 counter 族（`_total` 改名）的字段匹配缺失，
arm 后每次读取都被 503 拒绝，见"回放工具与接线"。两条 duration 计划（`sw_thermal`、
`hw_thermal`）与 024 `thermal_warning`/`power_performance`（`thermal_violation_total_us`、
`power_violation_total_us`）同受影响，修复后重跑；判据不变。

### GF-ENV-LIVE-007

**Power Brake。** 使用 `power-brake.json`，先验证一次 `0x80` 后清除不隔离，
再验证三个有效 `0x80` 样本触发与004相同的人工复核链。
另以独立任务验证 `0x04/0x08/0x10/0x100` 不误作 brake；`0x84/0xC0/0x180`
仍按 `mask & 0x80` 识别。温度 warning 不能把已确认 brake 降级为热复合 warning。
通过条件：回执明确是供电检查，不执行 reboot 或自动恢复；只处理目标节点/任务。
禁止拔电源或修改PDU制造此信号；真实物理注入见012。

### GF-ENV-LIVE-008

**不可用字段与别名。** 回放省略目标原因位（null）、非整数 mask、DCGM blank 数值，
每种连续至少三次；同时保持原始 GPU 身份和其他字段。核对 collection error/UNKNOWN
而非虚假 brake，不能从无数据生成 STOP/隔离。
在支持旧/新字段名的实际 exporter 版本上分别运行001和007，不用两份 exporter
冒充同一个 producer。别名冲突的细粒度行为已有软件测试；当前代理不制造不存在的
别名，现场缺少对应版本/受检 fixture 时，该变体保留 NOT_RUN。
多GPU时须单独检查缺字段GPU，不能用其他GPU的同名字段或整个通道的健康状态代替；
若实现仅按批次字段集合判完整而遗漏该GPU，应记录缺陷并判该变体失败。

### GF-ENV-LIVE-009

**间断与来源切换。** 在两个严重样本后，用受检单节点维护方式暂停采集，
使实际接受样本间隔大于60秒，再恢复；应从1重新确认。
另在保持正常温度时重启 collector，验证新鲜度和只有一个 producer。需要验证
DCGM→nvidia-smi 时，通过正常安装流程切换，核对 source 真实变化和新旧字段解析，
不能靠代理伪造 source/boot 标签，它们会被 collector 重新盖章。
通过条件：缺测不当健康，来源变化不继承确认，不发生越权恢复。
跨 boot 必须结合另行批准的节点维护重启验证；本例不主动重启节点，未做则注明缺口。

### GF-ENV-LIVE-010

**作用域与并发。** 至少两张 GPU 的测试节点和一个对照节点。只改一张 GPU，
验证原始对照指标未改动、finding/诊断包 GPU 范围准确；隔离可以是节点级，
STOP 必须是受影响分布式训练任务的已归属范围，不能错误要求只停单个 rank。
同时报告同一故障时核对 workflow 合并、节点锁与通知去重，不重复 STOP。
另安排独立对照任务/节点确认无变更，不能把同一个分布式任务其他节点的合法停止
误判成越界。没有多任务隔离条件则该变体 NOT_RUN。

### GF-ENV-LIVE-011

**绝对关机温度例外。** 只在真实 XML 报告绝对 `gpu_shutdown_temperature_c`
且已完成001的节点上，通过代理把核心当前温度改为该限值+1。
一个已接受样本即可触发保护，不等待三次；实际环境动作按004验收。
T.Limit 的 shutdown offset 不是同一证据，只有该 offset 的设备不能替代本例。
绝对禁止物理加热到关机温度；本例是立即保护逻辑的真机链路回放，不是关机实验。

### GF-ENV-LIVE-012

**真实故障及修复。** 仅消费自然故障，或由设备厂商/机房给出审批、操作步骤、
限值和带外停止机制的实验室故障注入。GFRS不提供制造过热/电源故障的方法。
采集未经代理修改的 NVML/nvidia-smi/DCGM、BMC/供电域证据，确认故障持续后
实际隔离/停训/人工通知；由现场团队修复。未获得可控故障条件时保持 NOT_RUN。
修复后在另行批准的负载窗口观察同 GPU 的温度/余量/原因位以及机架同位置关联，
保留恢复报告并经受控人工流程重新接纳。空闲降温、bit短暂清零、重启后消失，
都不能单独宣告根因修复。不得把回放007或011的结果充当本例 PASS。

## 清理与结果

每项都必须完成清理检查；清理失败使整项 FAIL，不能只记录主体步骤通过。

1. 取消测试负载和回放，恢复原 collector endpoint、模式、配置摘要及唯一 producer，
   至少观察三个真实新批次；代理应在恢复 endpoint 后才关闭。
2. 功率实验核对逐 GPU 还原回执和上限。没有达到 default=原始值的前提不得复用
   COLLECT-002 的默认值还原。动作结果未知时先保持隔离并调查，不盲目再次写功率。
3. 对已隔离节点保留 taint/归属，完成审批、现场健康验证和已有受控恢复流程后才
   重新接纳；测试脚本不得直接删除 taint 或自动恢复训练。
4. 输出 `case_id`、变体、制品/配置摘要、时间、节点/GPU/任务身份、注入性质、
   原始与变换后输入摘要、采集覆盖率、工作流/步骤/外部回执、负向观察窗口、
   通知证据、清理结果与缺口。`live-telemetry-replay` 和 `physical-fault` 分开记录。
5. 任一必需变体未执行，整类不能宣称完整通过；登记缺口和 NOT_RUN/BLOCKED原因。

用例清单可以安全列出，**不会执行真机操作**：

```bash
.venv/bin/python tools/run_fault_test_cases.py --category gpu-environment-live --include-manual --list
```

## 验收运行器 `run_env_live.py`

`scripts/e2e/regional/run_env_live.py` 把本规格落成一个受保护的运行器，覆盖
GF-ENV-LIVE-001..011（012 始终 NOT_RUN）。默认只计划，不动集群。

```bash
# 只读计划：读节点与设备真实限值，决定每个变体能否在该硬件上运行，写 plan.json
.venv/bin/python scripts/e2e/regional/run_env_live.py \
  --site-profile "$PROFILE" --run-dir "$RUN_DIR" \
  --case GF-ENV-LIVE-004 --plan

# 真机执行（我稍后在维护窗口内运行）：需要逐案确认串，隔离/停训分支另需分支确认串
.venv/bin/python scripts/e2e/regional/run_env_live.py \
  --site-profile "$PROFILE" --run-dir "$RUN_DIR" \
  --case GF-ENV-LIVE-004 --execute --confirm ENVLIVE004_EXECUTE \
  --confirm-isolate ENVLIVE004_ISOLATE --confirm-stop ENVLIVE004_STOP \
  --maintenance-window-end "$WINDOW_END"
```

- **逐案确认串** `ENVLIVE<NNN>_EXECUTE`；会 cordon/quarantine 并停训的用例
  （004/007/010/011）另需 `--confirm-isolate ENVLIVE<NNN>_ISOLATE` 与
  `--confirm-stop ENVLIVE<NNN>_STOP`。本链从不重启训练任务。
- **010 的对照节点**：`--node-b <对照节点>`（站点 profile 的 `node-b` 自动填入）；
  对照任务需要 `--site-file`。例：
  `--case GF-ENV-LIVE-010 --node-b "$NODE_B" --execute --confirm ENVLIVE010_EXECUTE
  --confirm-isolate ENVLIVE010_ISOLATE --confirm-stop ENVLIVE010_STOP`。
- **设备能力判定**：运行器在只读 preflight 用产品自身的 XML 解析读该节点逐 GPU
  的同刻度限值（`scripts/e2e/regional/env_live_plans.py`），据此判定每个严重温度
  变体 RUN/NOT_RUN，并由真实限值（而非 `tlimit-example.json`）生成回放计划。
  p5en H200 现场只报 T.Limit 刻度（shutdown −8、slowdown −2、max-operating 0），
  因此绝对核心温度变体与 011（绝对关机温度）在该站 NOT_RUN，T.Limit 变体与
  按配置 fallback 的显存变体运行。
- **代理接线**：`scripts/e2e/regional/probes/env_replay_proxy.py` 作为特权 host
  探针，把 `tools/gpu_environment_replay.py` 原样投到节点（base64 + sha256 校验，
  `-I` 运行）用 `systemd-run` 瞬态单元 `--serve`，再**只**对
  `gpu-fault-metrics-collector.service` 下 drop-in 把 `GPU_FAULT_DCGM_METRICS_URL`
  指向 `http://127.0.0.1:9411/metrics`。不改真实 exporter、不改 `collector.env`、
  不起第二个同身份 producer。`systemd-run` deadman 与运行器 finally 两路恢复原
  endpoint；代理只在 endpoint 恢复后关闭。真实上游出现 HW thermal/Power Brake、
  T.Limit≤0、GPU≥80 C 或显存≥85 C 时工具永久中止并透传真实读数。
- **判据** 由 `scripts/e2e/regional/env_live_verdicts.py` 依据控制面真实
  incident/workflow/notification/证据记录给出；工具从不输出 LIVE PASS。
- **两种集群**：运行器不含任何 HyperPod 专有步骤，plain EKS 与 HyperPod 共用。
  唯一的差别在只读 preflight 读取逐 GPU 限值的方式：exporter 以 DaemonSet 运行的节点
  （HyperPod）在该节点现有的 `gpu-fault-dcgm-exporter` Pod 内执行 `nvidia-smi -q -x`；
  host 模式节点（plain EKS：exporter 是 `gpu-fault-dcgm-exporter.service` 容器，
  DaemonSet 无 Pod）没有可 exec 的 Pod，则由一个瞬态特权 host 探针（与执行阶段相同的
  `env_replay_proxy.py`，子命令 `preflight`）在节点上用产品解析器读取同一份 XML 与
  exporter 的 T.Limit 读数，读完即删除并校验无残留；preflight/plan 记录
  `device_limits_source`（`exporter-pod` / `host-probe`）。两条 EKS lane（eks/eksb）
  各自作用于本 lane 的节点；004/007/010 隔离节点，取全部锁（X）。
- **009 间断与来源切换**：计划 `core_tlimit/interruption` = 2 次健康读 → 2 次严重读 →
  暂停 → 1 次严重读 → ≥3 次健康读。arm 之后立刻由 `env_replay_proxy.py pause-collector`
  在节点上起一个瞬态等待单元：它以 0.25 秒轮询代理 `/status`，在 collector 恰好消费完
  第二个严重读（`pause_after_reads`，按代理绝对读数计）的瞬间对
  `gpu-fault-metrics-collector.service` 执行 `systemctl kill --signal=SIGSTOP`（与
  `general_live_node_probe.py unit-pause` 相同的暂停方式，但在节点上读同步地触发：
  由运行器按轮询时机发起的暂停可能晚一个采集周期，让第三个严重样本完成确认并隔离
  节点，而 009 不带隔离确认串）。暂停前先以 `systemd-run --on-active` 装好恢复
  deadman（`resume-collector`，暂停时长 + 120 秒）；运行器在暂停满
  `CONFIRMATION_MAX_GAP_SECONDS + 15` = 75 秒后 `resume-collector`，待计划读完、清除
  读数被接受且该字段 finding 失效后 `restart-collector`（drop-in 保留，一次
  `systemctl restart`，核对 endpoint 不变）。等待单元没有起来则该案 INCONCLUSIVE 并
  立即还原 endpoint（在严重读开始之前）；等待单元超时或暂停落在目标读之后同样记
  INCONCLUSIVE（此时产品对三个连续严重样本的反应不是本例证据）。判据
  （`scripts/e2e/regional/env_live_interruption.py`）：暂停前已接受 2 个严重样本
  （WARNING `confirmation 2/3`），暂停后第一个被接受的严重样本与上一个相隔 >60 秒且
  其 finding `confirmation_count` 为 1、无 CRITICAL、无隔离/禁止步骤、无
  RESTORE_SCHEDULING（不越权恢复）；间隔内没有任何被接受的读数，暂停期间已激活的
  finding 不得在没有读数的情况下消失（缺测不当健康）；清除读数穿过 edge filter；重启
  后出现带节点 boot id 的 `initial-baseline` 批次，所有读数只有一个 `source`，重启后的
  批次速率不超过单个 producer（只有一个 producer）。DCGM→nvidia-smi 的来源切换只能经
  正常安装流程（`GPU_FAULT_METRICS_MODE`）完成，跨 boot 需另行批准的节点维护重启：
  本运行器都不驱动，记为未覆盖缺口而不是 PASS。
- **010 作用域与并发**：需要 `--node-b`（站点 profile 的 `node-b`）指定对照节点、
  `--site-file`（提交对照任务）与至少两张 GPU 的目标节点，缺一则 NOT_RUN 并写明原因。
  执行前先对对照节点做与目标节点相同的洁净 preflight；再在对照节点上以
  `gpu-training-submit` 提交节点固定的单节点训练夹具（`env_live_scope.ControlTask`，
  以验收默认 GPU duty 运行——`acceptance_gpu_load.DEFAULT_GPU_DUTY_PERCENT`，30 %，
  通过 `set_gpu_duty_percent` 显式写入渲染清单并记入 `control_task_before` 回执
  （`gpu_duty_percent`）——镜像预热），等待其 Running 且心跳正常，记录 Pod UID 与对照
  节点状态。对照任务不能空闲：产品的 `LOW_GPU_UTILIZATION`/`LOW_CPU_UTILIZATION`
  （`active_workload_only` 规则）只对节点上唯一有活容器的受管 attempt 所持 GPU 计时，
  利用率 ≤5 % 持续 300 秒即开 incident，一个 GPU duty 0 的受管对照任务约 9 分钟后就会被
  产品按设计判为低利用率故障，使 control 阶段因 harness 自己的夹具而 FAIL。该 duty 不会
  与回放计划巧合：scope 判据只读目标节点的批次与 finding（目标节点其余 GPU 无负载），
  对照节点只读记录不读样本，而 010 的严重计划在目标 GPU 上投放的是超过设备自身
  slowdown/最高工作温度一度的温度值，对照 GPU 出现这种读数是真实热故障而非有界负载
  （平均功耗只是功率上限的一部分）的产物；
  然后在目标节点的一张 GPU 上 arm 与 004 相同的严重 T.Limit 计划。判据
  （`env_live_scope.ScopeJudge`，在 004 判据之上）：目标节点其他 GPU 的该字段无
  finding、其被接受样本不含回放值（原始对照指标未改动）；incident 的 GPU/节点作用域
  只含目标 GPU/节点；对照节点全程无 cordon、无 `gpu-fault.io/` taint、无归属注解、无新
  incident/workflow/notification（新 incident 若为 `low_gpu_utilization`/
  `low_cpu_utilization`，仍判 FAIL，但 control 阶段错误文本附加诊断——产品判的是对照
  任务自身的利用率，核对其 GPU duty——证据列出 `control_utilization_incidents`）；
  对照任务 Pod 的 UID 集合不变且始终 Running；同一故障
  只有一个带隔离步骤的工作流、评审 incident 最多一条 SENT 通知（其余 DUPLICATE）、
  STOP_WORKLOADS 最多执行一次。目标节点按 preflight 要求没有归属训练任务，因此受影响
  分布式任务的 STOP 范围未测量，记入证据。清理顺序：还原 endpoint → 停代理 → 删探针 →
  删除对照任务（`remove_managed_job`）。两节点运行在 plain EKS 上占用两条 EKS lane 的
  节点（campaign 列表标 X）。
- **特殊用例**：002 复用 `GF-REGIONAL-COLLECT-002` 的受控功率与 dcgmproftester
  负载及其 deadman 还原（本运行器对 002 记 BLOCKED 并指向该运行器）；011 只在设备
  报告绝对 `gpu_shutdown_temperature_c` 时运行（H200 为 NOT_RUN）；012 为真实物理/
  厂商实验室故障，始终 NOT_RUN。

### arm 之后的观察与判据

arm 只是实验开始：正常安装的 collector 每个采集周期经代理读取一次，edge filter
决定哪些读取被投递，控制面对投递的批次分级，节点健康路径再编排并执行工作流。
运行器不再在 arm 后立即取一次快照判定，而是有界轮询
（`scripts/e2e/regional/env_live_watch.py`），按用例期望判定
（`scripts/e2e/regional/env_live_expectations.py`、
`scripts/e2e/regional/env_live_judges.py`）：

- **读取**：代理 `/status` 的 `reads`/`next_values`/`armed`/`aborted`（不消耗步骤），
  直到计划读完；控制面已接受的 GPU_METRICS 批次（含被回放字段的样本）、各 GPU
  最新读数、GPU finding（含 `confirmation_count` 与 `threshold_source`）；该节点的
  NODE_HEALTH incident、工作流及其步骤执行、通知；以及 Node 对象。共享的
  `store_snapshot` 以 XID 事件为键，看不到 `inc-gpu-<finding>` 环境 incident，
  因此不再使用。每次读取有超时，失败只记一次轮询的读错误。
- **截止时间**：按节点 `collector.env` 的采集节奏定界（与 repoint 首次读取预算同一组
  项）：自 arm（001 为 repoint）起，计划读数 × 采集周期，加节奏抖动（读数 × 周期 ×
  `GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO`，默认 0.1）、启动错峰与随机启动抖动
  （`GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS` 15 秒 + `GPU_FAULT_COLLECTOR_STARTUP_
  JITTER_SECONDS` 20 秒，窗口内重启的 collector 首次抓取所需），加一个周期与 30 秒处理
  余量，再加用例项：严重确认类加旧冷却 300 秒（`COOLDOWN_SECONDS`）+ 一个周期 + 30 秒
  派发；warning/duration 加 max(3 × 周期 + 30 秒保持, VALIDATE_GPU 热告警宽限 120 秒 +
  30 秒)；瞬时、非 brake 位与不可用字段加 3 × 周期 + 30 秒保持；001 为启动错峰 15 秒 +
  启动抖动 20 秒 + 一个周期 + 3 个透传周期 + 其抖动 + 30 秒。各项写入记录
  `observation.window.terms`。上限是 arm + 代理 `max_seconds`、repoint + collector deadman
  与维护窗口结束三者最早者减 90 秒清理余量；必需工作流仍在其自身 execution
  deadline 内运行、或保持期未满时，等待可延长到该上限，绝不超过。清理顺序与
  deadman 不变。
- **残留检查、repoint 证明与零读取复核**：探针 `preflight` 列出节点上残留的上一轮运行——
  `/run/gpu-fault-env-replay` 下未关闭的窗口、仍加载在 collector 单元上的 repoint drop-in、
  systemd 仍列出的 `gpu-fault-env-*` 瞬态单元（代理、恢复 deadman、暂停等待器）、进程读取的
  endpoint 与 `collector.env` 不一致——任一存在即不部署、INCONCLUSIVE 并逐条点名
  （`env_live_fixture.node_leftovers`）。repoint 回执必须带完整证明（`repointed`、进程环境
  指向代理、首次透传读取已出现：`env_live_fixture.repoint_proof`，与 024 共用），arm 前再读
  一次 `status`：进程 endpoint 仍是代理、代理未 armed、`reads` 不低于证明时的计数
  （`require_proxy_read_before_arm`）。arm 后若到首次读取应到时刻（一个带抖动的周期 +
  启动错峰 + 启动抖动，`CollectorCadence.first_read_due_seconds`）代理仍计 0 次读取，运行器
  读回一次单元 `snapshot`，与 repoint 回执对比并点名原因（`env_live_watch.diagnose_zero_reads`）：
  `endpoint-reverted`、`dropin-gone`、`unit-restarted`/`unit-inactive`、`proxy-refused-reads`
  （`/status` `failed_reads`/`last_error`，503）或 `collector-silent`，写入
  `replay.zero_reads_diagnosis` 与 INCONCLUSIVE 文字。证据范围仍为 `live-telemetry-replay`；
  复核只说明没有证明什么，不替代任何正向证据。
- **判定**：计划未读完、代理因真实安全读数中止、collector 不再读代理、或没有任何
  已接受记录证明回放值进入控制面并被分级，为 INCONCLUSIVE，绝不 PASS；值已进入但
  预期反应缺失或错误（只读信号出现隔离/禁止步骤、GPU/节点作用域错误、规格要求的
  工作流缺失、已确认 finding 没有 ENVIRONMENT_REVIEW 链与回执），为 FAIL；只有正向
  证据才 PASS。清理失败仍使整项 FAIL。
- **各用例**：
  - 004/011 与 007 的持续/混合 mask：已接受的连续严重样本（≤60 秒间隔，011 为 1 个）
    → CRITICAL finding（同刻度阈值来源；007 为供电检查）→ 目标 GPU/节点作用域的
    ENVIRONMENT_REVIEW 工作流 SUCCEEDED 且 MARK_UNSCHEDULABLE、QUARANTINE、诊断包、
    snapshot、ESCALATE_SUPPORT 回执齐全 → incident ESCALATED、通知已发送、节点
    cordon 并带 `gpu-fault.io/quarantined` taint；确认前的 WARNING 工作流不得有隔离
    步骤；计划读完且降温读数被接受后再保持 315 秒，节点仍隔离。
  - 003：同刻度 WARNING finding，T.Limit 生效时没有绝对 fallback finding，只读诊断
    工作流（FREEZE_EVIDENCE、VALIDATE_GPU）SUCCEEDED，保持期内无隔离/STOP。
  - 005 与 007 瞬时 brake：确认计数不超过实际送出的严重样本数且从不 CRITICAL，
    清除读数被接受（穿过 edge filter）且 finding 失效。
  - 006：热原因位或 duration（或 THERMAL_STRESS）WARNING，只读诊断工作流完成。
  - 007 非 brake 位：该位被接受且没有任何该字段 finding。
  - 008：有 collection error/UNKNOWN 记录点名该字段；若已接受的批次缺该 GPU 的字段
    却无人报告 UNKNOWN（多 GPU 时按批次字段集合判完整），判 FAIL。
  - 009：见上节"009 间断与来源切换"；轮询多读一项节点侧暂停状态（`pause-status`），
    恢复与重启两个维护步骤由判据按记录触发。
  - 010：004 的全部判据，外加对照 GPU、对照节点、对照任务与并发合并/去重的检查；轮询
    多读一项对照节点的 Node、活动记录与对照任务 Pod。
  - 001：repoint 后 collector 的 `initial-baseline` 批次带目标 GPU 的必需字段、
    `source_boot_id` 与节点 boot 一致、静态值（功率上限、阈值）与 repoint 前相同、
    代理至少 3 次透传读取、没有新的 GPU 环境 incident；只有 preflight 数据为
    INCONCLUSIVE。
- **已知局限**：非 brake 位与不可用字段不是 edge filter 的投递边沿，只有健康汇总
  （默认 300 秒）恰好落在短计划内才会进入控制面，否则为 INCONCLUSIVE；preflight 只
  接受无业务负载的节点，因此通常没有归属测试任务，证据中写明任务进度/挂起未测量
  （STOP_WORKLOADS 只对 ACTIVE attempt 编排）；产品在 600 秒回放窗口内来不及完成的
  反应记为 INCONCLUSIVE。

单元测试（`tests/regional/test_env_live_*.py`、`tests/regional/test_env_replay_proxy_probe.py`）
用替身覆盖计划、判据、运行器分支、有界观察与代理控制器，不触及集群；它们是支持性
逻辑覆盖，不把任何用例改写为 LIVE PASS。
