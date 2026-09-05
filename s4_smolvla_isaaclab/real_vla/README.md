# 真机 VLA 采集（real_vla）

独立于仿真 `data/`、`scripts/record_dataset.py` 和 IsaacLab rollout。只通过 adapter 使用现有 `hardware_teleop` 的 RobotBridge / Pink 遥操，不复制 ROS 驱动。

当前里程碑：采集 raw episode。LeRobot 转换、训练、推理只留接口。

## 冻结规格（v2）

- 任务：接近把手、抓紧、拉开、推回关闭、松手撤离并自动回 Home
- Policy state/action：**8D** = 实测臂 7 + 逻辑夹爪 1（OPEN=0 / GRASP=1）
- 真机手指仍然发 **6D** HandsCmd，由 `gripper_adapter` 展开
- 相机：头部 D435i + 当前臂腕部 D405
- 控制保持现有 30 Hz；raw 按真实时间戳异步落盘
- 图片：每路一个 `.mkv`；低维：`trajectory.h5` 或 fallback `trajectory.npz`
- ABXY：A 回 Home 后开始 / B 停止人工遥操并继续记录自动回 Home / X 保存 / Y 长按丢弃

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
sudo -E bash run.sh real-collect --arm-output
```

如果同时运行 `S2_homie_controller.launch.py` 控制腿部，必须使用 SDK 的融合入口：

```bash
sudo -E bash run.sh real-collect --arm-output --with-leg-deploy
```

腿部 deploy 已运行时必须使用 `--with-leg-deploy`；遗漏该参数会被安全检查拒绝。

此模式要求先启动腿部 deploy，并验证 `/lowcmd` 上唯一的既有发布者是
`/qi_topic_converter`；启动前还必须收到实际腿部策略帧。运行中该发布者消失、
策略帧超过 250 ms 未更新或出现第三个发布者时，遥操会立即停止输出。
不要把默认的 `/lowcmd_replay + mode_ctrl=4` 模式与腿部 deploy 同时运行。

正常采集会显示每秒刷新一次的彩色状态表，关键事件同时写入当前 session 的
`runtime.log`。需要完整 IK/bridge 滚屏日志时追加 `--input-debug`，旧调试输出
保持不变。

Quest 仍打开 `https://192.168.110.35:8443`。

流程：自动 Home → READY 阶段可用双手 Grip/Trigger 调整双臂双手 → A 再次确认 Home 后开始采集 → 采集中只允许右手 Grip 遥操、Trigger 抓握 → 拉开抽屉 → 推回关闭 → 松手并撤开 → B 停止人工遥操 → 系统继续记录自动回 Home → 看 QUALITY → X 保存或 Y 长按丢弃。

只有显式传入 `--arm-output` 才允许真机输出。READY 阶段开放双臂双手；A 被接受后只开放 `robot.yaml` 指定的活动臂和手，非活动侧回 Home 并保持张开。state/action 任一失效、控制 fault、低维丢样或视频帧数不一致都会令 episode 变为 `QUALITY INVALID`，invalid episode 不能按 X 保存。

每个 episode 同时保存采集配置快照、Git commit、dirty 标志、采集代码 SHA256，以及 `collection_phase`（`0=teleop`、`1=return_home`）。异常退出会在下次启动时跨 session 回收 pending episode。

数据默认写到 `/home/coral/real_vla_data/session_*/episodes/`。

批量检查所有已保存 episode（默认只读，不移动数据）：

```bash
sudo -E bash run.sh real-audit-dataset /home/coral/real_vla_data \
  --report /tmp/real_vla_audit.json
```

`INVALID`、读取失败的 `ERROR` 和未完成保存的 `PENDING` 会被视为不可用。
`WARN` 通常是未超过无效阈值的短时相机间隔，继续保留；`REVIEW` 需要人工查看视频确认任务是否完成。
审计启动时会先检查当前 Python 能否导入 OpenCV；缺少 `cv2` 时命令立即报依赖错误，
不会把所有视频误报为无效。若自动选择的硬件 Conda 环境没有 OpenCV，可显式使用
`sudo -E env S4_HW_TELEOP_RUNTIME=system bash run.sh real-audit-dataset ...`。
确认清单后，可恢复地隔离确定无效的数据和 pending 数据（不会删除）：

```bash
sudo -E bash run.sh real-audit-dataset /home/coral/real_vla_data \
  --quarantine-invalid --yes --report /tmp/real_vla_quarantine.json
```

## 目录

```
real_vla/
  config/           collection / robot / cameras
  input/            Quest ABXY
  robot/            S4 adapter, gripper 1D→6D, HomeManager
  cameras/          RealSense capture threads
  collection/       状态机、异步 writer、quality、causal sync 报告
  data/             episode 读取；lerobot_export 为 NotImplemented
  scripts/          collect / camera_test / inspect / validate / audit
```

普通 `bash run.sh teleop-hardware --arm-output` 行为不变。采集只通过 `TeleopHooks` 观察 tick 和最终下发关节。
