# Drawer 任务 20→10 语言阶段设计

## 1. 目标

在不改变现有 20 个专家控制阶段、IK 路径、物理门控和成功判定的前提下，将
`drawer_insert_close` 数据中的语言条件压缩为 10 个宏观阶段。现有原始 HDF5
不改写，转换时生成新的 LeRobotDataset；未来采集直接记录 10 阶段语言，同时
保留真实专家阶段名用于诊断。

本改动只发生在 `s4_smolvla_isaaclab`。固定版本的 `lerobot` 仓库只作为依赖调用，
不得修改。

## 2. 非目标

- 不减少或重排 20 个专家控制阶段。
- 不改变机器人轨迹、IK、手指动作、随机化或任务成功条件。
- 不修改现有 HDF5、`s4_drawer_insert_close_v0`、旧训练输出或旧 checkpoint。
- 不在本次工作中重新采集或训练。
- 不修改 `/home/zfy/smolVLA/lerobot`。

## 3. 语言阶段

| ID | Prompt | 来源控制阶段 | Rollout 结束门控来源 |
|---|---|---|---|
| `prepare_hands` | `Open both hands and prepare for the task.` | `initial_open_hands` | `initial_open_hands` |
| `approach_drawer_handle` | `Move the left hand onto the drawer handle.` | `left_approach_handle`, `left_approach_handle_fine`, `left_grasp_handle` | `left_grasp_handle` |
| `grasp_drawer_handle` | `Close the left hand around the drawer handle and hold it.` | `left_close_hand` | `left_close_hand` |
| `pull_drawer` | `Pull the drawer open with the left hand.` | `pull_drawer` | `pull_drawer` |
| `approach_can` | `Move the open right hand around the can and hold it steady.` | `right_pregrasp_can`, `right_grasp_can`, `right_settle_before_close` | `right_settle_before_close` |
| `grasp_can` | `Close the right hand around the can and hold it steady.` | `right_close_hand`, `right_hold_grasp` | `right_hold_grasp` |
| `lift_can` | `Lift the grasped can clear of the support surface.` | `right_lift_can` | `right_lift_can` |
| `place_can` | `Move the grasped can into the open drawer.` | `right_place_in_drawer` | `right_place_in_drawer` |
| `release_and_retreat` | `Release the can and move the open right hand clear of the drawer.` | `right_open_hand`, `right_lift_clear_drawer`, `right_retreat_clear_drawer` | `right_retreat_clear_drawer` |
| `close_drawer_and_home` | `Close the drawer and return both arms home.` | `right_retreat_and_start_close`, `left_open_hand`, `left_joint_transition_after_release`, `left_home` | `left_home` |

约束：ID 和 prompt 唯一；20 个控制阶段全部且只被映射一次；每组来源阶段在控制
顺序中连续；`rollout_gate_phase` 必须属于对应组。

## 4. 配置模型

在 `configs/tasks/drawer_insert_close.scripted.yaml` 增加顶层
`language_phases`。每项包含稳定 ID、prompt、`source_phases` 和
`rollout_gate_phase`。英文文本不再作为程序主键。

配置加载层提供纯函数，构建：

- 控制阶段名 → 语言阶段 ID；
- 旧控制阶段 task 文本 → 语言阶段 ID；
- 语言阶段 ID → prompt；
- prompt → 语言阶段 ID；
- 语言阶段 ID → Rollout 门控来源控制阶段。

加载时执行完整性和唯一性校验。

## 5. HDF5 与未来采集

HDF5 每帧保存：

- `obs/task_description`：10 阶段宏观 prompt；
- `obs/language_phase_id`：稳定语言阶段 ID；
- `obs/expert_phase_name`：真实的 20 阶段控制器阶段名。

专家控制器继续返回控制阶段名和原始 task。记录器利用配置映射写入宏观语言。
失败诊断、罐子首次位移定位等逻辑改用 `expert_phase_name`，避免多个控制阶段共享
prompt 后产生歧义。

旧 HDF5 不含新增字段。转换器必须兼容：通过旧的逐帧 `task_description` 查找原控制
阶段，再映射到新语言阶段。

## 6. 转换与新数据集

转换器对旧、新 HDF5 采用统一优先级：

1. 有 `language_phase_id` 时直接解析并校验；
2. 否则用旧 `task_description` 映射；
3. 无法映射时失败，不静默回退为全局任务文本。

LeRobot 每帧仍只写标准 `task` 字段，不修改 LeRobot 源码。转换输出的
`meta/s4_contract.json` 增加语言契约版本、有序阶段 ID、prompt、来源控制阶段和门控
来源。新数据集 ID 为 `s4_drawer_insert_close_v1_10phase`。

首次迁移通过现有 `--root-path` 指向旧 HDF5；未来采集写入新的 staging 目录。
原 `s4_drawer_insert_close_v0` 保留。

## 7. 训练

训练配置切换到：

- dataset：`s4_drawer_insert_close_v1_10phase`；
- output：`smolvla_drawer_insert_close_v1_10phase`。

状态、动作、相机、FPS 和 normalization 数值契约不变。新语言分布必须从基础模型
开始新训练，不 resume 旧 20 阶段 checkpoint。

## 8. Policy Server 与 Rollout

Policy Server继续从数据 parquet 的连续 `task_index` 运行段恢复常用顺序和中位帧数。
相邻控制阶段拥有相同 prompt 后会自然合并成 10 个运行段。Server 从
`meta/s4_contract.json` 将 prompt 解析为 `language_phase_id`，并在 schedule 中返回 ID。

Rollout 按 ID 解析门控配置，再取对应 `rollout_gate_phase` 的现有物理门控字段。
不得再用英文 prompt 直接匹配控制阶段。

阶段延长预算：

- `approach_drawer_handle`：恢复 20 个 20 Hz 策略帧；
- `pull_drawer`：恢复 20 个 20 Hz 策略帧；
- 其他阶段：默认 20 帧。

每个宏观阶段切换时保持现有行为：清空旧 Chunk、重置 policy、用新 prompt 预测
50 帧，并用 8 帧从上一阶段末端命令过渡。语言切换次数由 20 次降为 10 次。

## 9. 数据检查与兼容性

`dataset_check.py` 验证：

- contract 声明 10 阶段语言版本；
- tasks 恰好等于配置中的 10 个唯一 prompt；
- 每个 episode 的连续语言 ID 顺序一致；
- 旧 20 阶段文本和新 10 阶段文本不得混杂；
- episode、frame、图像、state 和 action 数量保持一致；
- checkpoint 与 dataset 的语言契约版本一致（新 checkpoint 记录该字段时）。

旧数据集仍可使用旧 checkpoint；新数据集只用于新训练和新 Rollout。转换默认不
覆盖已有目标，除非用户显式使用 `--overwrite`。

## 10. 测试与验证

采用测试驱动实现：

1. 配置映射完整性、唯一性和连续性；
2. 旧 HDF5 task 文本映射为 10 prompts；
3. 新 HDF5 保存宏观 ID、prompt 和专家阶段名；
4. 转换输出 10 个 task，帧数和数值字段不变；
5. Policy schedule 合并为 10 阶段并携带稳定 ID；
6. Rollout 门控按 ID 解析终点控制阶段；
7. 恢复80帧的延长阶段；
8. dataset check 拒绝未知、混合或乱序语言契约；
9. 相关 CPU 测试、Shell/Python 语法检查通过。

不以未运行的 Isaac Sim、完整转换、训练或 Rollout 作为已验证结果。大型旧 HDF5
的真实转换必须在代码测试完成后由用户明确执行，避免占用当前算力和磁盘。

## 11. 文档更新

同步更新数据采集、HDF5 schema、转换、检查、训练、Policy Server、在线 Rollout、
架构知识库和课程实现章节。文档明确区分 20 个专家控制阶段与 10 个语言阶段，
提供旧 HDF5 一次性转换、新采集、新训练和 Rollout 命令。
