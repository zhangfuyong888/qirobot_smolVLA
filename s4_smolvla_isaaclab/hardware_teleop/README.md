# 真机 Quest 遥操作（hardware_teleop）

本目录是 **S4 双臂真机遥操作** 的独立模块。遥操电脑作为 **控制服务器**：

- 接收 Meta Quest 3 手柄输入（无视频流，controller-only）
- 在本地做 IK 解算（默认 RMPflow，可选 Pinocchio）
- 通过 ROS2 DDS **直接发布** `lowcmd`（手臂）和 `/handscmd`（灵巧手）

**不依赖** `qiling_s4`、MoveIt、数据采集或 VLA 训练代码。仿真遥操仍使用 `bash run.sh teleop`；真机遥操使用 `bash run.sh teleop-hardware`。

---

## 目录

1. [系统架构](#系统架构)
2. [目录结构](#目录结构)
3. [环境要求](#环境要求)
4. [一次性准备](#一次性准备)
5. [配置文件说明](#配置文件说明)
6. [启动与使用](#启动与使用)
7. [Quest 手柄操作](#quest-手柄操作)
8. [ROS2 话题与数据流](#ros2-话题与数据流)
9. [控制逻辑与安全机制](#控制逻辑与安全机制)
10. [IK 后端切换](#ik-后端切换)
11. [命令行参数](#命令行参数)
12. [首次联调 Checklist](#首次联调-checklist)
13. [常见问题](#常见问题)
14. [开发与测试](#开发与测试)

---



## 系统架构

```
┌─────────────────┐     WebSocket/HTTPS      ┌──────────────────────────────────┐
│  Meta Quest 3   │ ───────────────────────► │  hardware_teleop/main.py         │
│  (无视频流)      │   controller_frame      │  ├─ QuestWebServer (复用)         │
└─────────────────┘                         │  ├─ BimanualTeleopMapper (复用)  │
                                            │  ├─ headless Isaac + RMPflow IK  │
                                            │  └─ HardwareRobotBridge          │
                                            └───────────┬──────────┬───────────┘
                                                        │          │
                              ROS2 DDS (CycloneDDS)     │          │
                                                        ▼          ▼
                                            ┌─────────────────────────────┐
                                            │  真机 SDK                    │
                                            │  订阅: lowcmd, /handscmd     │
                                            │  发布: lowstate              │
                                            └─────────────────────────────┘
```



### 与仿真遥操的区别


| 项目       | 仿真 `run.sh teleop`          | 真机 `run.sh teleop-hardware` |
| -------- | --------------------------- | --------------------------- |
| Isaac 场景 | 完整任务场景 + 可选 viewport        | **仅加载机器人 URDF**（headless）   |
| 物理步进     | 120 Hz 仿真                   | 无真实物理，仅同步关节状态做 FK/IK        |
| 指令输出     | `set_joint_position_target` | `lowcmd` **+** `/handscmd`  |
| 状态输入     | 仿真关节                        | `lowstate`**（默认）**          |
| 控制频率     | ~120 Hz                     | **30 Hz**（与真机 MIT 控制一致）     |
| ROS2     | 不需要                         | **需要**（Humble + CycloneDDS） |




### 设计原则

- **自包含**：qi 消息定义 vendored 在 `ros_ws/`，lowstate 解析逻辑在 `vendored/`
- **隔离**：不修改 `record_dataset.py`、训练、rollout 等 VLA 链路
- **可切换 IK**：默认 RMPflow（与仿真手感一致）；真机效果不佳时可换 Pinocchio

---



## 目录结构

```
hardware_teleop/
├── README.md                 # 本文档
├── main.py                   # 真机遥操主入口
├── config_loader.py          # 加载 quest_hardware.yaml
├── joint_mapping.py          # 26D ↔ 14 臂关节、符号、步长限制
├── hand_mapping.py           # trigger 0..1 → 手部 uint16 0..255
├── config/
│   ├── quest_hardware.yaml   # 真机 ROS 话题、增益、手部、IK 后端
│   ├── ros_env.sh            # ROS2 + CycloneDDS 环境（常改：网卡名）
│   ├── ros_env.example.sh    # 模板
│   └── ros_env.local.sh      # 可选本地覆盖（gitignore，不提交）
├── scripts/
│   ├── source_ros_env.sh     # 统一 source 入口
│   └── build_ros_msgs.sh     # 编译 vendored qi 消息
├── ros/
│   ├── robot_bridge.py       # 订阅状态、发布 lowcmd/handscmd
│   └── env.py                # 本地 ROS 安装路径提示
├── ik/
│   ├── rmpflow_headless.py   # 默认 IK（headless Isaac + Lula）
│   └── pinocchio_backend.py  # 备用 IK
├── scene/
│   └── minimal.py            # 仅 spawn 机器人 articulation
├── vendored/                 # 从 qiling 逻辑复制，无运行时依赖
│   ├── joint_layout.py       # 真机 26 关节顺序
│   ├── joint_state_safety.py # 异常 joint state 过滤
│   └── lowstate_decode.py    # lowstate → 臂关节角
└── ros_ws/                   # 本地 colcon 工作区（build/install 已 gitignore）
    └── src/qi/msg/           # vendored LowCmd, LowState, HandsCmd 等
```

上层复用（不在本目录内，但遥操依赖）：

- `teleoperation/server.py` — Quest HTTPS/WSS 服务
- `teleoperation/mapping.py` — 离合、坐标映射、手部 trigger
- `teleoperation/webxr/index.html` — Quest 浏览器页面
- `configs/teleoperation/meta_quest3.yaml` — 映射/平滑/安全/workspace（通过 `quest_hardware.yaml` 引用）

---



## 环境要求



### 遥操电脑


| 组件        | 说明                                     |
| --------- | -------------------------------------- |
| OS        | Linux（与 Isaac Lab 环境一致）                |
| ROS2      | Humble，`/opt/ros/humble/setup.bash`    |
| RMW       | `rmw_cyclonedds_cpp`                   |
| Isaac Lab | `env_isaaclab` conda 环境（headless IK 用） |
| 网络        | 与机器人在同一局域网；DDS 绑定 **接机器人的网卡**          |




### 机器人侧


| 组件       | 说明                                               |
| -------- | ------------------------------------------------ |
| 上电 + SDK | 正常运行，持续发布 `lowstate`                             |
| 接收指令     | 订阅 `lowcmd`（手臂）、`/handscmd`（手）                   |
| **禁止并发** | 不要同时运行其他 `lowcmd` 发布者（如 `moveit_mit_arm_bridge`） |




### Quest

- Meta Quest 3，挂脖模式即可（**不需要**向头显推视频）
- 与遥操电脑同一 WiFi
- 浏览器访问 `https://<电脑局域网IP>:8443`

---



## 一次性准备

在项目根目录 `s4_smolvla_isaaclab/` 下执行：

### 1. 编译 vendored qi ROS 消息

```bash
bash run.sh teleop-hardware-build
# 等价于: bash hardware_teleop/scripts/build_ros_msgs.sh
```

成功后生成：`hardware_teleop/ros_ws/install/setup.bash`

### 2. 配置 ROS/DDS 环境

编辑网卡名（接机器人的那张网卡）：

```bash
# 若还没有配置文件，从模板复制：
cp hardware_teleop/config/ros_env.example.sh hardware_teleop/config/ros_env.sh

# 编辑这一行：
# HW_TELEOP_NETWORK_INTERFACE=enp47s0
```

验证：

```bash
source hardware_teleop/scripts/source_ros_env.sh
# 应看到: [HW-TELEOP][ENV] ready ros=/opt/ros/humble interface=enp47s0 ...
```



### 3. 生成 Quest HTTPS 证书

证书 IP 必须是 Quest 能访问到的 **电脑局域网 IP**（不是 `127.0.0.1`）：

```bash
bash run.sh teleop-cert --ip 192.168.110.63 --overwrite
```

证书路径：`.local/teleoperation/cert.pem` / `key.pem`

### 4. （可选）本地覆盖

若某些机器有额外环境变量，可创建（不提交 git）：

```bash
# hardware_teleop/config/ros_env.local.sh
export HW_TELEOP_NETWORK_INTERFACE=wlan0
```

---



## 配置文件说明



### `config/ros_env.sh` — ROS2 / DDS 环境


| 变量                             | 默认值                  | 说明               |
| ------------------------------ | -------------------- | ---------------- |
| `HW_TELEOP_ROS_DISTRO`         | `/opt/ros/humble`    | ROS2 安装路径        |
| `HW_TELEOP_NETWORK_INTERFACE`  | `enp47s0`            | CycloneDDS 绑定的网卡 |
| `HW_TELEOP_RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | RMW 实现           |


`bash run.sh teleop-hardware` 会自动 source；手动调试 topic 时在终端执行：

```bash
source hardware_teleop/scripts/source_ros_env.sh
```

查看可复制命令：

```bash
bash run.sh teleop-hardware-env
```



### `config/quest_hardware.yaml` — 真机控制参数

引用仿真映射配置：

```yaml
teleop_config: configs/teleoperation/meta_quest3.yaml
```



#### `hardware` 段


| 字段                        | 默认             | 说明                               |
| ------------------------- | -------------- | -------------------------------- |
| `control_rate_hz`         | `30.0`         | 主控制循环频率                          |
| `state_source`            | `lowstate`     | 状态来源：`lowstate` 或 `joint_states` |
| `lowstate_topic`          | `lowstate`     | SDK 状态话题                         |
| `lowcmd_topic`            | `lowcmd`       | 手臂 MIT 指令话题                      |
| `hands_cmd_topic`         | `/handscmd`    | 灵巧手指令话题                          |
| `body_dof`                | `26`           | 12 腿 + 7 左臂 + 7 右臂               |
| `arm_kp` / `arm_kd`       | `60.0` / `2.0` | lowcmd 手臂 PD 增益                  |
| `reversed_joint_names`    | 见 yaml         | 与真机符号约定一致的 3 个翻转关节               |
| `max_joint_step_rad`      | `0.065`        | 每周期最大关节变化（rad）                   |
| `initial_state_timeout_s` | `15.0`         | 等待首帧 lowstate 超时                 |
| `stale_command_hold`      | `true`         | Quest 输入 stale 时停止手臂运动           |
| `max_state_age_s`         | `0.5`          | lowstate 断流超过此时间则停止手臂运动          |


#### `startup` 段

启动后、等待 Quest 摇操之前，双臂会**缓慢插值**到任务 `home_pose`（来自 `load_task_control_profiles`），避免上电瞬间跳变。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `move_to_home` | `true` | 是否执行启动 homing |
| `duration_s` | `4.0` | homing 最长时长（秒） |
| `max_joint_step_rad` | `0.03` | homing 每周期最大关节步长（比遥操更慢） |
| `position_tolerance_rad` | `0.02` | 到达 home 的容差 |
| `check_lowcmd_publishers` | `true` | 创建本节点 lowcmd 发布者**之前**检查是否已有其他发布者 |

若检测到 `lowcmd` 上已有发布者（例如 `moveit_mit_arm_bridge`），进程会**拒绝启动**，避免多源指令冲突。

#### `gravity_compensation` 段

手臂 lowcmd 的 `motor.tau` 会叠加 Pinocchio 重力补偿（逻辑 vendored 自 qiling `moveit_mit_arm_bridge`）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否启用重力补偿 |
| `urdf_path` | `assets/my_robot/urdf/s4_40dof_merged.urdf` | 动力学 URDF |
| `scale` | `0.6` | 重力力矩缩放（与 qiling 真机默认一致） |
| `tau_limit` | `12.0` | 单关节力矩限幅（N·m） |
| `ramp_time` | `2.0` | 启动后重力补偿从 0 渐增到满量程的时间 |
| `source` | `current` | 用 `current`（实测关节）或 `target`（指令关节）算重力 |

Pinocchio 不可用时仅打印警告并继续（无重力补偿），不影响遥操主流程。


#### `hands` 段

仿真里 trigger 0..1 线性插值 open/close；真机映射为 uint16 0..255：


| 字段                                         | 说明          |
| ------------------------------------------ | ----------- |
| `left_open_uint16` / `left_close_uint16`   | 左手开/合 6 指目标 |
| `right_open_uint16` / `right_close_uint16` | 右手开/合       |
| `duration_ms`                              | 每个手指动作持续时间  |




#### `ik` 段


| 字段        | 可选值           | 说明                                     |
| --------- | ------------- | -------------------------------------- |
| `backend` | `rmpflow`（默认） | headless Isaac + Lula，与仿真 RMPflow 参数一致 |
|           | `pinocchio`   | Pinocchio DLS，便于真机侧重新调参                |




#### `scene` 段

headless Isaac 仅用于加载机器人做 FK/IK，**不加载 drawer 等任务资产**。

### `configs/teleoperation/meta_quest3.yaml` — 映射与手感

真机与仿真共用（通过 `teleop_config` 引用），常用项：


| 段                         | 作用                   |
| ------------------------- | -------------------- |
| `network.port`            | Quest 服务端口，默认 `8443` |
| `network.stale_timeout_s` | 超过 1 s 无有效帧 → 冻结双臂   |
| `mapping.position_scale`  | 手柄位移放大，默认 `2.2`      |
| `mapping.clutch`          | Grip 离合阈值            |
| `safety.workspace_*`      | TCP 工作空间限制           |
| `smoothing.arm_*`         | 手臂指令平滑               |


---



## 启动与使用



### 标准启动流程

**终端 1 — 启动真机遥操（自动 source ROS 环境 + 编译检查）：**

```bash
# 在 s4_smolvla_isaaclab 项目根目录执行
bash run.sh teleop-hardware
```

**终端 2 — 调试 ROS 话题（需先 source 环境）：**

```bash
# 在 s4_smolvla_isaaclab 项目根目录执行
source hardware_teleop/scripts/source_ros_env.sh

ros2 topic list
ros2 topic hz lowstate          # 机器人 SDK 发布，启动遥操前就应能看到
ros2 topic hz lowcmd            # 遥操启动且收到 lowstate 后 ~30Hz
ros2 topic echo /handscmd --once
```

**Quest — 连接手柄：**

1. Quest 连接与电脑相同 WiFi
2. 浏览器打开：`https://<电脑IP>:8443`
3. 进入 WebXR，允许手柄追踪
4. 日志出现 `[HW-TELEOP][INPUT] session=...` 表示已连接



### 启动成功时的典型日志

```
[HW-TELEOP][ENV] ready ros=/opt/ros/humble interface=enp47s0 rmw=rmw_cyclonedds_cpp
[HW-TELEOP] runtime=hardware ik=rmpflow control_rate_hz=30.0 state=lowstate ...
[HW-TELEOP] waiting for lowstate (timeout=15.0s)...
[HW-TELEOP][BOOT] minimal headless robot scene ready (no task assets)
[HW-TELEOP][RMPFLOW] ready: independent left/right policies ...
[HW-TELEOP] Meta Quest controller server ready
[HW-TELEOP] Quest URL: https://192.168.x.x:8443
```



### 短跑测试（不长期占用）

```bash
bash run.sh teleop-hardware --max-runtime-s 30
```

---



## Quest 手柄操作


| 输入                               | 功能                      |
| -------------------------------- | ----------------------- |
| **Grip / Squeeze**（`buttons[1]`） | 单臂离合：按住才动该侧手臂           |
| **松开 Grip**                      | 手臂停在当前姿态，不再跟踪旧目标        |
| **Trigger**                      | 手部开合 0→1（映射到 uint16 指令） |
| 左右手                              | **独立离合**，可只控一侧          |


头显挂脖即可，**不向 Quest 推送相机画面**（controller-only 模式）。

---



## ROS2 话题与数据流



### 话题一览


| 话题          | 类型            | 方向  | 发布者     | 何时可见               |
| ----------- | ------------- | --- | ------- | ------------------ |
| `lowstate`  | `qi/LowState` | 订阅  | 机器人 SDK | 机器人上电 + DDS 通      |
| `lowcmd`    | `qi/LowCmd`   | 发布  | 本模块     | 遥操启动且收到 lowstate 后 |
| `/handscmd` | `qi/HandsCmd` | 发布  | 本模块     | 同上                 |


> 注意：`lowcmd` 默认 **无** 前导 `/`；手部话题为 `/handscmd`。



### lowcmd 内容（26 电机）

- 索引 0–11：腿部，`mode=0`（不主动控腿）
- 索引 12–18：左臂 7 关节，`mode=1`，MIT PD（`q, kp, kd`）
- 索引 19–25：右臂 7 关节，同上

手臂关节名顺序见 `vendored/joint_layout.py` 中 `REAL_ROBOT_BODY_JOINT_ORDER`。

### 26D 控制向量布局

与仿真 / VLA 数据集一致（`s4_robot/control_mapping.py`）：


| 索引    | 内容                             |
| ----- | ------------------------------ |
| 0–6   | 左臂 7 关节（rad）                   |
| 7–12  | 左手 6 关节（rad，仿真侧；真机走 uint16 通道） |
| 13–19 | 右臂 7 关节                        |
| 20–25 | 右手 6 关节                        |




### 符号约定

以下 3 个关节在 lowstate 读入 / lowcmd 写出时做符号翻转（与真机 SDK 约定一致）：

- `left_wrist_roll_joint`
- `left_wrist_yaw_joint`
- `right_shoulder_yaw_joint`

---



## 控制逻辑与安全机制

**启动阶段**（Quest 连接前）：

1. 等待首帧 `lowstate`
2. 检查 `lowcmd` 是否已有其他发布者
3. 双臂按 `startup` 配置缓慢移动到任务 home pose
4. 进入主循环，等待 Grip 离合遥操

每 **30 Hz** 循环一次：

1. **spin ROS** — 处理 `lowstate` 回调，更新实测臂关节角
2. **同步 headless 机器人** — 将真机关节写入 Isaac articulation（仅 FK/IK 用）
3. **读取 Quest 帧** — `BimanualTeleopMapper` 输出 TCP 目标 + 手部 trigger
4. **IK** — 仅在 clutch 按下侧调用 RMPflow/Pinocchio
5. **组 26D 指令** — 未 clutch 侧保持当前实测关节；手部始终跟 trigger
6. **平滑 + 限幅** — `smooth_command` + `max_joint_step_rad`
7. **发布** — `lowcmd`（臂）+ `/handscmd`（手）



### 安全机制


| 机制                   | 说明                              |
| -------------------- | ------------------------------- |
| 离合门控                 | 未按 Grip 不运动该臂                   |
| stale 冻结             | Quest 断连 / 超 1 s 无有效帧 → 停止发新臂指令 |
| 关节步长限制               | 每周期 `max_joint_step_rad`        |
| workspace 限制         | mapper 内 TCP 边界 + 速度限制          |
| JointStateFrameGuard | 拒绝 NaN、异常全零 lowstate 帧          |
| 初始状态等待               | 未收到 lowstate 不进入控制循环            |
| lowcmd 冲突检测            | 启动前拒绝已有 lowcmd 发布者               |
| 启动 homing              | 缓慢过渡到 home pose 再等待摇操           |
| 重力补偿 ramp            | `motor.tau` 渐增，减轻上电冲击             |
| lowstate 断流             | 超过 `max_state_age_s` 停止手臂运动并 hold |




### 急停建议

本模块 **没有** 独立硬件急停 topic。建议：

- 松开双手 Grip + 关闭遥操进程
- 或切断机器人物理急停
- 确保无其他节点并发发布 `lowcmd`

---



## IK 后端切换



### 默认：RMPflow（推荐先试）

- headless Isaac 加载机器人 URDF，不渲染、不加载任务场景
- 使用与仿真相同的 Lula RMPflow 配置（`configs/teleoperation/rmpflow/`）
- 控制频率 30 Hz 下 `update_every_n_steps=1`

```bash
bash run.sh teleop-hardware
# 或在 quest_hardware.yaml 中 ik.backend: rmpflow
```



### 备用：Pinocchio

若真机上 RMPflow 跟踪手感不理想，可切换 DLS IK：

```bash
bash run.sh teleop-hardware --ik-backend pinocchio
```

或在 `quest_hardware.yaml` 中设置 `ik.backend: pinocchio`。

Pinocchio 参数来自 `meta_quest3.yaml` 的 `ik:` 段（`posture_gain`, `damping`, `max_joint_delta_rad` 等）。

---



## 命令行参数

通过 `bash run.sh teleop-hardware -- [参数]` 传递：


| 参数                               | 说明                                                        |
| -------------------------------- | --------------------------------------------------------- |
| `--hardware-config PATH`         | 硬件配置 yaml，默认 `hardware_teleop/config/quest_hardware.yaml` |
| `--ik-backend rmpflow|pinocchio` | 覆盖 yaml 中的 IK 后端                                          |
| `--host / --port`                | Quest 服务绑定地址（默认读 meta_quest3.yaml）                        |
| `--insecure-http`                | 仅桌面调试，Quest WebXR 不可用                                     |
| `--report-period-s 0.5`          | 状态日志周期                                                    |
| `--max-runtime-s N`              | N 秒后自动退出，0 为一直运行                                          |
| `--input-debug`                  | 打印详细输入/跟踪信息                                               |
| `--headless`                     | 默认已启用（Isaac 无窗口）                                          |


---



## 首次联调 Checklist

按顺序执行，避免一上来就动真机：

- [ ] **1. 机器人上电**，SDK 运行，无其他 `lowcmd` 发布者
- [ ] **2. 配置网卡** — 编辑 `config/ros_env.sh` 中 `HW_TELEOP_NETWORK_INTERFACE`
- [ ] **3. source 环境** — `source hardware_teleop/scripts/source_ros_env.sh`
- [ ] **4. 验证 lowstate** — `ros2 topic hz lowstate` 有稳定频率
- [ ] **5. 编译 qi 消息** — `bash run.sh teleop-hardware-build`（若未做过）
- [ ] **6. 证书** — `bash run.sh teleop-cert --ip <Quest可访问的IP>`
- [ ] **7. 短跑遥操** — `bash run.sh teleop-hardware --max-runtime-s 30`
- [ ] **8. 验证 lowcmd** — 另一终端 `ros2 topic hz lowcmd` ≈ 30 Hz
- [ ] **9. Quest 连接** — 打开 `https://<IP>:8443`，确认 session 日志
- [ ] **10. 单臂小范围** — 按住一侧 Grip，小位移试动，确认方向与离合
- [ ] **11. 双手 + Trigger** — 验证手部 `/handscmd` 随 trigger 变化

---



## 常见问题



### `ros2 topic list` 为空或没有 `lowstate`

- 检查 `HW_TELEOP_NETWORK_INTERFACE` 是否为 **接机器人** 的网卡（`ip link` 查看）
- 确认与机器人同网段
- 确认机器人 SDK 已发布 `lowstate`
- 两个终端都要 `source hardware_teleop/scripts/source_ros_env.sh`



### 遥操启动报 `timed out waiting for lowstate`

- 遥操电脑收不到 SDK 状态，先完成上一节排查
- 可适当增大 `initial_state_timeout_s`



### 遥操启动报 qi messages not built

```bash
bash run.sh teleop-hardware-build
```



### Quest 打不开页面

- 使用 **局域网 IP**，不是 `127.0.0.1`
- 证书 IP 与访问 IP 一致：`bash run.sh teleop-cert --ip <LAN_IP> --overwrite`
- Quest 与电脑同一 WiFi



### 手臂乱动 / 方向反了

- 检查 `reversed_joint_names` 是否与真机一致
- 确认无其他节点并发发 `lowcmd`
- 首次试动用小范围、低速度，单臂 clutch



### 想改用 `/joint_states` 而不是 `lowstate`

在 `quest_hardware.yaml` 中：

```yaml
hardware:
  state_source: joint_states
  joint_states_topic: /joint_states
```

需要外部节点发布含 14 臂关节名的 `sensor_msgs/JointState`。

### 与 VLA 训练/采集的关系

本模块 **完全独立**。修改此处不影响：

- `bash run.sh record`
- `bash run.sh train`
- `bash run.sh rollout`

仿真采集的数据与真机 lowcmd 控制不在同一链路，**不能混用**为同一训练集。

---



## 开发与测试



### 单元测试

```bash
# 在 s4_smolvla_isaaclab 项目根目录执行
python3 -m pytest tests/test_hardware_teleop_*.py -q
```

覆盖：配置加载、关节映射、手部 uint16 插值、lowstate 解析。

### 修改代码后建议检查

```bash
python3 -m pytest tests/test_hardware_teleop_*.py tests/test_teleoperation_*.py -q
python3 -m py_compile hardware_teleop/main.py hardware_teleop/ros/robot_bridge.py
```



### 相关入口（项目根）


| 命令                                  | 作用                |
| ----------------------------------- | ----------------- |
| `bash run.sh teleop-hardware`       | 启动真机遥操            |
| `bash run.sh teleop-hardware-build` | 编译 qi 消息          |
| `bash run.sh teleop-hardware-env`   | 打印 source 命令      |
| `bash run.sh teleop-cert`           | 生成 Quest HTTPS 证书 |
| `bash run.sh teleop`                | 仿真遥操（非本模块）        |


---



## 快速参考

```bash
# 一次性
bash run.sh teleop-hardware-build
bash run.sh teleop-cert --ip 192.168.x.x --overwrite
# 编辑 hardware_teleop/config/ros_env.sh 中的网卡名

# 每次 — 调试
source hardware_teleop/scripts/source_ros_env.sh
ros2 topic hz lowstate

# 每次 — 启动
bash run.sh teleop-hardware

# Quest
# https://192.168.x.x:8443
# Grip = 臂离合 | Trigger = 手
```
