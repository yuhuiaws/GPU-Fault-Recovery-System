from __future__ import annotations

import os as os
from threading import RLock as RLock
from typing import Any as Any
from typing import Protocol as Protocol

from pydantic import Field as Field
from pydantic import model_validator as model_validator

from gpu_fault.hyperpod import (
    HyperPodRecoveryAdvisory as HyperPodRecoveryAdvisory,
)
from gpu_fault.models import (
    AdvisoryNotification as AdvisoryNotification,
)
from gpu_fault.models import (
    NotificationResult as NotificationResult,
)
from gpu_fault.models import (
    NotificationStatus as NotificationStatus,
)
from gpu_fault.models import StrictModel as StrictModel


RESTART_GUARD_TEMPLATE_VERSION = "restart-guard-zh-v1"
RESTART_WORKLOAD_TEMPLATE_VERSION = "restart-workload-zh-v1"
RESTART_NODE_TEMPLATE_VERSION = "restart-node-zh-v1"
WARM_SPARE_REPLACEMENT_TEMPLATE_VERSION = "warm-spare-replacement-zh-v1"
RESTART_FABRIC_MANAGER_TEMPLATE_VERSION = "restart-fabric-manager-zh-v1"
GPU_RESET_TEMPLATE_VERSION = "gpu-reset-zh-v1"
FABRIC_RESET_TEMPLATE_VERSION = "fabric-reset-zh-v1"
NOT_APPLICABLE_TEMPLATE_VERSION = "xid-not-applicable-zh-v2"
XID_INVESTIGATORY_TEMPLATE_VERSION = "xid-investigatory-zh-v1"
SXID_EVENT_TEMPLATE_VERSION = "sxid-event-zh-v1"
HARDWARE_ESCALATION_TEMPLATE_VERSION = "hardware-escalation-zh-v1"
NVLINK74_SUPPORT_TEMPLATE_VERSION = "xid74-support-zh-v2"
NVLINK74_MECHANICAL_TEMPLATE_VERSION = "xid74-mechanical-zh-v1"
GPU_MECHANICAL_TEMPLATE_VERSION = "gpu-mechanical-zh-v1"
DCGM_DIAGNOSTIC_TEMPLATE_VERSION = "dcgm-diagnostic-zh-v2"
EFA_RDMA_EVENT_TEMPLATE_VERSION = "efa-rdma-event-zh-v1"
HARDWARE_INVENTORY_TEMPLATE_VERSION = "hardware-inventory-mismatch-zh-v1"
HOST_RESOURCE_EVENT_TEMPLATE_VERSION = "host-resource-event-zh-v1"

HARDWARE_INVENTORY_EMAIL_TEMPLATE = """\
GPU 节点硬件 Inventory 不匹配通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Event ID：{event_id}
- 事件时间：{observed_at}
- EC2/HyperPod 实例类型：{node_instance_type}
- 硬件类型：{resource_type}
- 严重级别：{severity}

二、Inventory 检查
- Metric：{metric_name}
- 期望数量：{expected_count}
- 当前 ACTIVE 数量：{observed_count}
- 当前发现数量：{discovered_count}
- 缺失数量：{missing_count}
- 超出数量：{excess_count}
- 连续异常采样：{consecutive_samples}
- 触发所需采样：{required_samples}

三、关联训练任务
- Workload 状态：{workload_state}
- 受影响 workload：{workloads}

四、系统处理
- 策略动作：{recommended_action}
- Workflow steps：{workflow_steps}
- Policy source：{policy_source}
- Policy version：{policy_version}

五、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

EFA_RDMA_EVENT_EMAIL_TEMPLATE = """\
GPU EFA/RDMA 监控事件通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Event ID：{event_id}
- 事件时间：{observed_at}
- 事件类型：{event_type}
- 严重级别：{severity}

二、监控信号
- Metric：{metric_name}
- Device/Port：{device}
- 当前值：{value}
- 流量状态：{traffic_signal}
- 流量 baseline：{baseline}
- 触发原因：{reason}

三、关联训练任务
- Workload 状态：{workload_state}
- Job ID：{job_id}
- Attempt ID：{attempt_id}
- 受影响 workload：{workloads}

四、系统处理
- 策略动作：{recommended_action}
- Policy source：{policy_source}
- Policy version：{policy_version}
- Workflow steps：{workflow_steps}

五、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

HOST_RESOURCE_EVENT_EMAIL_TEMPLATE = """\
GPU 节点资源隐患通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Event ID：{event_id}
- 事件时间：{observed_at}
- 严重级别：{severity}

二、监控信号
- Signal：{signal}
- Metric：{metric_name}
- Device/Mount：{device}
- 当前值：{value}
- 比较方式：{comparison}
- 配置阈值：{threshold_percent}%
- 持续时间阈值：{minimum_active_seconds} 秒
- 触发原因：{reason}
- 同批次关联指标：{related_metrics}

三、关联训练任务
- Workload 状态：{workload_state}
- 受影响 workload：{workloads}

四、系统处理
- 策略动作：{recommended_action}
- Workflow steps：{workflow_steps}
- Policy source：{policy_source}
- Policy version：{policy_version}

五、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

DCGM_DIAGNOSTIC_EMAIL_TEMPLATE = """\
GPU DCGM 快速诊断结果通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Event ID：{event_id}
- 受影响 workload：{workloads}

二、诊断结果
{node_results}

三、系统处理
- 控制面动作：{control_plane_action}
- PASS/WARN：进入温度冷却观察和 GPU validation。
- FAIL/INCONCLUSIVE：停止受影响任务、禁止调度并隔离节点，不自动重启训练。
- 仅配置类失败（dcgmErrorSeverity_t=CONFIG，如 persistence mode 未开）：
  按 WARN 处理，不隔离节点、不停止任务；请按下方指导修主机配置后复跑诊断。

四、下一步处理指导
{recommended_actions}

五、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

NVLINK74_SUPPORT_EMAIL_TEMPLATE = """\
GPU XID 74 NVLink 事件支持通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 节点：{node_ids}
- 受影响 workload：{workloads}
- Event ID：{event_id}
- Policy source：{policy_source}
- Official action：{official_action}
- 解码与处置原因：
{reasons}

二、系统处理
- 未执行 XID 74 单 GPU reset。
- 节点调度和训练任务状态以 Workflow 实际步骤记录为准。
- 已生成厂商支持记录：{ticket_id}

三、工单信息
- Ticket ID：{ticket_id}
- Workflow operation：ESCALATE_SUPPORT

邮件模板：{template_version}
"""

NVLINK74_MECHANICAL_EMAIL_TEMPLATE = """\
GPU XID 74 NVLink 机械检查待办

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 节点：{node_ids}
- NVLink：{link_id}
- PCI BDF：{pci_bdf}
- 寄存器计数：{occurrence_counts}

二、当前状态
- Workflow operation：CHECK_MECHANICALS
- 训练和调度状态以 Workflow 实际步骤记录为准。
- 系统不会自动确认物理检查已经完成。

三、确认字段
- Annotation：{annotation}
- Required value：{annotation_value}

邮件模板：{template_version}
"""

GPU_MECHANICAL_EMAIL_TEMPLATE = """\
GPU 机械检查待办

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 节点：{node_ids}
- XID：{xid}
- PCI BDF：{pci_bdf}

二、当前状态
- Workflow operation：CHECK_MECHANICALS
- 训练和调度状态以 Workflow 实际步骤记录为准。
- 系统不会自动确认物理检查已经完成。

三、确认字段
- Annotation：{annotation}
- Required value：{annotation_value}

邮件模板：{template_version}
"""

HARDWARE_ESCALATION_EMAIL_TEMPLATE = """\
GPU 节点自动恢复失败及硬件下线通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 故障节点：{node_ids}
- 受影响 workload：{workloads}
- 事件类型：{event_type}
- 原始事件：{event_id}
- 失败原因：
{reasons}

二、系统处理
- GPU reset、节点 reboot 和 warm-spare replacement 自动恢复链路已失败。
- 故障节点保持 unschedulable 和 quarantine，不会恢复调度。
- 系统已创建厂商支持工单记录：{ticket_id}

三、工单信息
- Ticket ID：{ticket_id}
- Policy source：{policy_source}
- Official action：{official_action}
- Workflow operation：ESCALATE_SUPPORT

邮件模板：{template_version}
"""

NOT_APPLICABLE_EMAIL_TEMPLATE = """\
GPU XID 策略不适用通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- XID：{xid}
- GPU 产品：{product}
- Driver branch：{driver_branch}
- CUDA 版本：{cuda_version}
- 事件时间：{observed_at}
- Event ID：{event_id}
- Workload 状态：{workload_state}
- 受影响 workload：{workloads}

二、策略判定
- Disposition：NOT_APPLICABLE
- NVIDIA Catalog 动作：{official_action}
- NVIDIA Investigatory Action：{investigatory_action}
- Policy source：{policy_source}
- Policy version：{policy_version}
- 判定原因：
{reasons}

三、系统处理
- 官方恢复动作：未执行
- Safety action：{safety_action}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Workflow 状态：{workflow_status}
- Safety steps：{safety_steps}
- Official steps：{official_steps}

四、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

XID_INVESTIGATORY_EMAIL_TEMPLATE = """\
GPU XID 调查动作通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- XID：{xid}
- GPU 产品：{product}
- 事件时间：{observed_at}
- Event ID：{event_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Workload 状态：{workload_state}
- 受影响 workload：{workloads}

二、策略判定
- Disposition：{disposition}
- NVIDIA Immediate Action：{official_action}
- 系统 Effective Action：{effective_action}
- NVIDIA Investigatory Action：{investigatory_action}
- Policy source：{policy_source}
- Policy version：{policy_version}
- 判定原因：
{reasons}

三、调查动作状态
- 本方案不会自动执行 Investigatory Action。
- 是否执行该调查动作由管理员决定。
- Immediate Action 和训练任务状态以 Workflow 实际记录为准。

四、证据信息
{evidence_refs}

邮件模板：{template_version}
"""

SXID_EVENT_EMAIL_TEMPLATE = """\
GPU NVSwitch SXID 事件通知

一、事件信息
- 集群：{cluster_id}
- 节点：{node_id}
- SXID：{sxid}
- GPU 产品：{product}
- 事件时间：{observed_at}
- Event ID：{event_id}
- Collector source：{event_source}
- Evidence：{evidence_ref}
- Workload 状态：{workload_state}
- 受影响 workload：{workloads}

二、NVSwitch 信息
- Classification：{classification}
- Classification source：{classification_source}
- Link scope：{link_scope}
- Link scope source：{link_scope_source}
- Switch：{switch_id}
- Port：{port}
- PCI BDF：{pci_bdf}
- Fabric partition：{fabric_partition}

三、策略判定
- Disposition：{disposition}
- NVIDIA Official Action：{official_action}
- 系统 Effective Action：{effective_action}
- Safety Action：{safety_action}
- NVIDIA Investigatory Action：{investigatory_action}
- Policy source：{policy_source}
- Policy version：{policy_version}
- 判定原因：
{reasons}

四、系统记录
- Incident：{incident_id}
- Workflow：{workflow_id}
- 后续恢复动作及其结果使用独立固定模板通知。

邮件模板：{template_version}
"""

GPU_COUNT_CHANGE_EMAIL_TEMPLATE = """\
GPU 训练任务重启已暂停，需要管理员处理

一、发生了什么
- 训练任务：{job_id}
- 失败运行：{attempt_id}
- 集群：{cluster_id}
- 受影响 workload：{workloads}
- 失败运行使用的 GPU 数量：{source_gpu_count}
- 当前模板准备使用的 GPU 数量：{target_gpu_count}
- 系统检测到重启前后 GPU 数量不一致或无法确认，因此已阻止自动重启。
- 当前没有创建新的训练 workload，也没有消耗 restart budget。

二、建议管理员做什么
方案 A：恢复原资源。将 workload 模板恢复为 {source_gpu_count} 张 GPU；系统再次检查到数量一致后才会继续重启。
方案 B：接受 {target_gpu_count} 张 GPU。先修改并核对以下配置：
1. world size 及数据并行、张量并行、流水线并行策略；
2. per-device batch size、gradient accumulation 和 global batch size；
3. learning rate 及其调度策略；
4. checkpoint 与新并行拓扑的兼容性；
5. 训练模板中的 expected critical ranks 和资源请求。
完成上述修改并审核无误后，执行以下精确审批命令。不要在修改训练参数前批准：
{approval_commands}

审批后，控制面会重新读取 workload 模板；只有实际 GPU 变化仍为 {approval_annotation} 时才会继续。

三、追踪信息
- Incident：{incident_id}
- 审批 annotation：gpu-fault.io/approve-gpu-count-change={approval_annotation}
- 邮件模板：{template_version}
"""

RESTART_BUDGET_EMAIL_TEMPLATE = """\
GPU 训练任务自动重启次数已耗尽，需要管理员处理

一、发生了什么
- 训练任务：{job_id}
- 最近失败运行：{attempt_id}
- 集群：{cluster_id}
- 已完成或预留的自动重启次数：{restart_count}
- 配置的最大自动重启次数：{restart_budget}
- 系统已停止继续自动重启，防止任务反复失败并占用 GPU 资源。

二、建议管理员做什么
1. 检查最近几次失败的 GPU、节点、训练日志和 checkpoint；
2. 判断故障是否已修复，以及当前资源和训练参数是否仍然适用；
3. 不要直接提高 restart budget 掩盖重复故障；
4. 确认可以继续后，使用新的 job ID 人工提交新训练任务。

三、追踪信息
- Incident：{incident_id}
- 邮件模板：{template_version}
"""

RESTART_WORKLOAD_EMAIL_TEMPLATE = """\
GPU 训练任务自动重启通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- 训练任务：{job_id}
- 受影响 workload：{workloads}

二、重启信息
- 系统动作：RESTART_WORKLOAD
- 原 attempt：{source_attempt_id}
- 新 attempt：{restart_attempt_id}
- 原 GPU 数量：{source_gpu_count}
- 重启 GPU 数量：{target_gpu_count}
- 本任务已使用自动重启次数：{restart_count}
- 本任务最大自动重启次数：{restart_budget}
- Kubernetes 重启操作：已成功提交

三、追踪信息
- Workflow：{workflow_id}
- Operation：{operation_id}
- 邮件模板：{template_version}
"""

RESTART_NODE_EMAIL_TEMPLATE = """\
GPU 节点自动重启通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 触发事件：{event_id}
- 故障类型：{event_type}
- 故障标识：{fault_identifier}
- Collector source：{event_source}
- Policy source：{policy_source}
- 策略动作：{official_action}
- Effective action：{effective_action}
- 重启原因：
{reasons}
- 受影响节点：{node_ids}

二、重启信息
- 系统动作：RESTART_NODE
- HyperPod 动作：BatchRebootClusterNodes
- HyperPod operation：{operation_id}
- 重启节点明细：
{node_observations}
- 重启确认方式：{confirmation_source}
- 重启状态：已确认完成

三、后续状态
- GPU 验证：由 workflow 后续 VALIDATE_GPU step 执行
- Fabric 验证：由 workflow 后续 VALIDATE_FABRIC step 执行
- 调度恢复：验证成功后由 RESTORE_SCHEDULING step 执行

邮件模板：{template_version}
"""

WARM_SPARE_REPLACEMENT_EMAIL_TEMPLATE = """\
GPU 节点 warm-spare 替换成功通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- Event ID：{event_id}
- Policy source：{policy_source}
- Official action：{official_action}
- Effective action：{effective_action}

二、替换结果
- 故障节点：{fault_nodes}
- 激活 warm spare：{spare_nodes}
- 节点重绑定：
{node_rebindings}
- 确认来源：{confirmation_source}
- Provider replacement API submitted：{provider_mutation_submitted}
- Operation ID：{operation_id}

三、系统状态
- REPLACE_NODE branch 已成功完成。
- 原故障节点保持隔离，后续训练使用重绑定后的 warm spare。
- GPU/Fabric validation、恢复调度和训练 restart 仍以 Workflow 最终状态为准。

四、处置原因
{reasons}

邮件模板：{template_version}
"""

RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE = """\
GPU Fabric Manager 自动重启通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 触发事件：{event_id}
- 故障类型：{event_type}
- Policy source：{policy_source}
- 策略动作：{official_action}
- 触发原因：
{reasons}
- 受影响 workload：{workloads}

二、重启信息
- 系统动作：RESTART_FABRIC_MANAGER
- Operation：{operation_id}
- 服务：nvidia-fabricmanager
- 节点重启明细：
{node_results}
- 重启状态：已确认完成

邮件模板：{template_version}
"""

GPU_RESET_EMAIL_TEMPLATE = """\
GPU 自动重置完成通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 触发事件：{event_id}
- 故障类型：{event_type}
- Policy source：{policy_source}
- 策略动作：{official_action}
- 触发原因：
{reasons}
- 受影响 workload：{workloads}

二、执行信息
- 系统动作：RESET_GPU
- Operation：{operation_id}
- 目标节点：{node_ids}
- 目标 GPU UUID：{gpu_uuids}
- 执行结果：
{node_results}
- Reset 状态：已执行成功

三、后续状态
- GPU 服务恢复：由 workflow 后续 RESTORE_GPU_SERVICES step 执行
- GPU 验证：由 workflow 后续 VALIDATE_GPU step 执行
- 调度恢复：验证成功后由 RESTORE_SCHEDULING step 执行
- 训练恢复：全部验证成功后由 RESTART_WORKLOAD step 执行

邮件模板：{template_version}
"""

FABRIC_RESET_EMAIL_TEMPLATE = """\
GPU/NVSwitch 自动重置通知

一、事件信息
- 集群：{cluster_id}
- Incident：{incident_id}
- Workflow：{workflow_id}
- 触发事件：{event_id}
- 故障类型：{event_type}
- SXID：{sxid}
- Policy source：{policy_source}
- 策略动作：{official_action}
- Fabric partition：{fabric_partition}
- 触发原因：
{reasons}
- 受影响 workload：{workloads}

二、执行信息
- 系统动作：RESET_ALL_GPUS_NVSWITCHES
- Operation：{operation_id}
- 执行节点明细：
{node_results}
- Reset 状态：已执行并通过本机 GPU inventory 复核

三、后续状态
- GPU 服务恢复：由 workflow 后续 RESTORE_GPU_SERVICES step 执行
- GPU 验证：由 workflow 后续 VALIDATE_GPU step 执行
- Fabric 验证：由 workflow 后续 VALIDATE_FABRIC step 执行
- 训练恢复：全部验证成功后由 RESTART_WORKLOAD step 执行

邮件模板：{template_version}
"""
