# Drawer 语言阶段契约

`drawer_insert_close` 现在明确区分两层阶段：

- **20 个专家控制阶段**：负责 IK 目标、手指动作、物理门控、失败诊断和成功采集；
- **10 个语言宏阶段**：作为 SmolVLA 的逐帧任务文本，也是在线 Rollout 的阶段计划。

减少语言阶段不会删除或合并控制动作。映射的唯一配置源是
`configs/tasks/drawer_insert_close.scripted.yaml` 中的 `language_phases`。

| 语言 ID | 模型输入文本 | 包含的专家控制阶段 | Rollout 结束门控 |
|---|---|---|---|
| `prepare_hands` | Open both hands and prepare for the task. | `initial_open_hands` | `initial_open_hands` |
| `approach_drawer_handle` | Move the left hand onto the drawer handle. | 三个左手接近阶段 | `left_grasp_handle` |
| `grasp_drawer_handle` | Close the left hand around the drawer handle and hold it. | `left_close_hand` | `left_close_hand` |
| `pull_drawer` | Pull the drawer open with the left hand. | `pull_drawer` | `pull_drawer` |
| `approach_can` | Move the open right hand around the can and hold it steady. | 预抓取、精确接近、闭手前稳定 | `right_settle_before_close` |
| `grasp_can` | Close the right hand around the can and hold it steady. | 闭手、持握 | `right_hold_grasp` |
| `lift_can` | Lift the grasped can clear of the support surface. | `right_lift_can` | `right_lift_can` |
| `place_can` | Move the grasped can into the open drawer. | `right_place_in_drawer` | `right_place_in_drawer` |
| `release_and_retreat` | Release the can and move the open right hand clear of the drawer. | 松手、上抬、退出 | `right_retreat_clear_drawer` |
| `close_drawer_and_home` | Close the drawer and return both arms home. | 关抽屉、左手松开、过渡、回 Home | `left_home` |

## 数据兼容规则

未来采集的 HDF5 每帧保存：

- `obs/task_description`：10 阶段宏观文本；
- `obs/language_phase_id`：稳定语言阶段 ID；
- `obs/expert_phase_name`：实际执行的 20 阶段专家名。

旧 HDF5 不会被改写。转换器读取旧的 `obs/task_description`，按当前映射生成新的
10 阶段 LeRobotDataset。无法识别、字段长度不一致或多个字段互相矛盾时，转换会
直接报错，不会静默混合数据。

当前新数据集 ID 是 `s4_drawer_insert_close_v1_10phase`，训练输出是
`smolvla_drawer_insert_close_v1_10phase`。旧 `v0` 数据集、输出和 checkpoint 保留，
但不能 resume 到新语言契约，也不能与新数据集配对 Rollout。

转换后的 `meta/s4_contract.json` 保存完整有序语言契约。训练入口把该文件复制为
训练目录的 `s4_dataset_contract.json`；数据检查和 Rollout 会比较两者，防止误用
旧 checkpoint。

> 注意：语言契约变化改变了训练条件分布，因此必须从基础 SmolVLA 开始 fresh
> training。它不改变 26D state/action、三路相机、20 Hz 数据频率或 50 帧 Action Chunk。
