# 真机 VLA 采集（real_vla）

独立于仿真 `data/`、`scripts/record_dataset.py` 和 IsaacLab rollout。只通过 adapter 使用现有 `hardware_teleop` 的 RobotBridge / Pink 遥操，不复制 ROS 驱动。

当前里程碑：采集 raw episode。LeRobot 转换、训练、推理只留接口。

## 冻结规格（v1）

- 任务：拉开抽屉（固定文本写在 episode metadata）
- Policy state/action：**8D** = 实测臂 7 + 逻辑夹爪 1（OPEN=0 / GRASP=1）
- 真机手指仍然发 **6D** HandsCmd，由 `gripper_adapter` 展开
- 相机：头部 D435i + 当前臂腕部 D405
- 控制保持现有 30 Hz；raw 按真实时间戳异步落盘
- 图片：每路一个 `.mkv`；低维：`trajectory.h5` 或 fallback `trajectory.npz`
- ABXY：A 开始 / B 结束并回 Home / X 保存 / Y 长按丢弃

## 启动

先确认相机左右腕序列号（各拍一张）：

```bash
cd /home/coral/qirobot_smolVLA/s4_smolvla_isaaclab
bash run.sh real-collect-cameras
```

对照 `/tmp/real_vla_camera_test/*.png`，必要时改 `real_vla/config/cameras.yaml` 里两个 D405 serial。

SDK 已在跑的前提下：

```bash
cd /home/coral/qirobot_smolVLA/s4_smolvla_isaaclab
sudo -E bash run.sh real-collect --arm-output --input-debug
```

Quest 仍打开 `https://192.168.110.35:8443`。

流程：自动 Home → A 开始采 → 右手 Grip 遥操、Trigger 抓握 → B 停止并回 Home → 看 QUALITY → X 保存或 Y 长按丢弃。

数据默认写到 `/home/coral/real_vla_data/session_*/episodes/`。

## 目录

```
real_vla/
  config/           collection / robot / cameras
  input/            Quest ABXY
  robot/            S4 adapter, gripper 1D→6D, HomeManager
  cameras/          RealSense capture threads
  collection/       状态机、异步 writer、quality、causal sync 报告
  data/             episode 读取；lerobot_export 为 NotImplemented
  scripts/          collect / camera_test / inspect / validate
```

普通 `bash run.sh teleop-hardware --arm-output` 行为不变。采集只通过 `TeleopHooks` 观察 tick 和最终下发关节。
