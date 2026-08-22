# 数据契约 `s4_bimanual_v1`

## 26D state/action

顺序固定为：

```text
[0:7]   left_arm:  shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
[7:13]  left_hand: thumb yaw/pitch, index, middle, ring, pinky
[13:20] right_arm: 同上
[20:26] right_hand: 同上
```

精确 joint names：

```text
left_shoulder_pitch_joint, left_shoulder_roll_joint,
left_shoulder_yaw_joint, left_elbow_joint, left_wrist_roll_joint,
left_wrist_pitch_joint, left_wrist_yaw_joint,
lh_thumb_cmc_yaw, lh_thumb_cmc_pitch, lh_index_mcp_pitch,
lh_middle_mcp_pitch, lh_ring_mcp_pitch, lh_pinky_mcp_pitch,
right_shoulder_pitch_joint, right_shoulder_roll_joint,
right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint,
right_wrist_pitch_joint, right_wrist_yaw_joint,
rh_thumb_cmc_yaw, rh_thumb_cmc_pitch, rh_index_mcp_pitch,
rh_middle_mcp_pitch, rh_ring_mcp_pitch, rh_pinky_mcp_pitch
```

action 是 absolute joint target，不是 delta。6D hand control 由
`s4_robot/control_mapping.py` 映射到 active joints，并用倍率扩展 exposed mimic
joints。SmolVLA pre/postprocessor 使用 checkpoint 内保存的 mean/std 完成
STATE/ACTION normalization，仿真侧不重复归一化。

机器可读 schema 版本、动作语义、20 Hz 数据频率和 120 Hz 控制频率定义在
`configs/tasks/drawer_insert_close.dataset.json`。兼容修改保持
`s4_bimanual_v1`；改变维度、顺序、相机 key 或动作语义时必须创建新版本并提供迁移说明。

## HDF5

- `data/demo_N/processed_actions`: `[T,26] float32`
- `states/articulation/robot/joint_position`: `[T,full_dof]`
- `obs/s4_active_joint_pos`: `[T,26]`
- `obs/task_description`: `[T] UTF-8`
- `obs/language_phase_id`: `[T] UTF-8`，稳定的 10 阶段语言 ID
- `obs/expert_phase_name`: `[T] UTF-8`，真实的 20 阶段控制器阶段名
- `obs/{chest_front,left_wrist,right_wrist}_rgb`: `[T,480,680,3] uint8 RGB`

## LeRobotDataset

feature keys 为 `observation.state`、`action` 和三路
`observation.images.*`；FPS=20。每帧还包含 timestamp、episode/frame/index 和
task_index。SmolVLA 内部 padding 上限 50/32 不改变真实 26D contract。

语言文本采用 `drawer_10phase_v1` 契约；20 个专家控制阶段仍保留用于控制与诊断。
完整映射见[语言阶段契约](LANGUAGE_PHASES.md)。
