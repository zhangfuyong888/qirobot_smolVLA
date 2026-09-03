# 真机 Quest 遥操作（hardware_teleop）

本目录是 **S4 双臂真机遥操作** 的独立模块。控制服务器既可运行在开发电脑，也可直接运行在机器人电脑：

- 接收 Meta Quest 3 手柄输入（无视频流，controller-only）
- 在本地用仓库内置 Pink + Pinocchio 做 FK/IK 解算
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
│  Meta Quest 3   │ ───────────────────────► │  hardware_teleop/pink_main.py    │
│  (无视频流)      │   controller_frame      │  ├─ QuestWebServer (复用)         │
└─────────────────┘                         │  ├─ BimanualTeleopMapper (复用)  │
                                            │  ├─ vendored Pink + Pinocchio IK │
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
| Isaac 场景 | 完整任务场景 + 可选 viewport        | **不启动 Isaac**                  |
| 物理步进     | 120 Hz 仿真                   | 无真实物理，直接用实测关节做 Pink FK/IK   |
| 指令输出     | `set_joint_position_target` | `lowcmd` **+** `/handscmd`  |
| 状态输入     | 仿真关节                        | `lowstate`**（默认）**          |
| 控制频率     | ~120 Hz                     | **30 Hz**（与真机 MIT 控制一致）     |
| ROS2     | 不需要                         | **需要**（Humble + CycloneDDS） |




### 设计原则

- **自包含**：qi 消息定义 vendored 在 `ros_ws/`，lowstate 解析逻辑在 `vendored/`
- **隔离**：不修改 `record_dataset.py`、训练、rollout 等 VLA 链路
- **同源 IK**：仿真和真机复用同一个 Pink solver、权重、TCP offset 与肘部 barrier
- **可回退**：旧 headless Isaac/RMPflow 入口保留为 `teleop-hardware-isaac`

---



## 目录结构

```
hardware_teleop/
├── README.md                 # 本文档
├── pink_main.py              # 纯 Pink 真机遥操主入口（无 Isaac）
├── main.py                   # 旧 headless Isaac/RMPflow 回退入口
├── environment.yml           # Python 3.10 轻量真机环境
├── requirements-system-runtime.txt # 真机系统 Python 的项目局部依赖版本
├── config_loader.py          # 加载 quest_hardware.yaml
├── joint_mapping.py          # 26D ↔ 14 臂关节、符号、步长限制
├── hand_mapping.py           # trigger 0..1 → 手部 uint16 0..255
├── config/
│   ├── quest_hardware.yaml   # 真机 ROS 话题、增益、手部、IK 后端
│   ├── ros_env.sh            # ROS2 + CycloneDDS 环境（常改：网卡名）
│   ├── ros_env.example.sh    # 模板
│   ├── ros_env.robot.example.sh # 已检查真机的 lo/Domain16/system-Python 模板
│   └── ros_env.local.sh      # 可选本地覆盖（gitignore，不提交）
├── scripts/
│   ├── source_ros_env.sh     # 统一 source 入口
│   ├── prepare_system_runtime.sh # 仅向项目 .local 安装系统运行依赖
│   └── build_ros_msgs.sh     # 编译 vendored qi 消息
├── ros/
│   ├── robot_bridge.py       # 订阅状态、发布 lowcmd/handscmd
│   └── env.py                # 本地 ROS 安装路径提示
├── ik/
│   ├── pink_backend.py       # 默认纯 Pink IK
│   ├── rmpflow_headless.py   # 旧 headless Isaac + Lula 回退
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
| OS        | Ubuntu 22.04 / Linux                     |
| ROS2      | Humble，`/opt/ros/humble/setup.bash`    |
| RMW       | 自动选择已安装的 CycloneDDS 或 Fast DDS；可显式配置 |
| Python    | 开发机可用 Conda；真机直接用 `/usr/bin/python3.10` + 项目局部 wheel |
| 网络        | 与机器人在同一局域网；DDS 绑定 **接机器人的网卡**          |




### 机器人侧


| 组件       | 说明                                               |
| -------- | ------------------------------------------------ |
| 上电 + SDK | 正常运行，持续发布 `lowstate`                             |
| 接收指令     | 订阅 `lowcmd`（手臂）、`/handscmd`（手）                   |
| 并发检查 | 正常拓扑必须恰有一个站立策略 `lowcmd` 源；多个源或旧 mode-5 遥操会被拒绝 |




### Quest

- Meta Quest 3，挂脖模式即可（**不需要**向头显推视频）
- 与遥操电脑同一 WiFi
- 浏览器访问 `https://<电脑局域网IP>:8443`

---



## 一次性准备

在项目根目录 `s4_smolvla_isaaclab/` 下执行：

### 0. 真机系统 Python 局部依赖（无 Conda、无 venv）

真机已检查为 Ubuntu 22.04 x86_64、Python 3.10、ROS Pinocchio 3.9.0。先查看计划，不写文件：

```bash
bash run.sh teleop-hardware-system-prepare --check
```

用户确认安装后，依赖只写入 gitignore 的 `.local/hardware_python`，不会写 `/usr` 或 `~/.local`：

```bash
bash run.sh teleop-hardware-system-prepare --install
```

固定安装 `scipy 1.15.2`、`aiohttp 3.14.3`、`qpsolvers 4.12.0`、`daqp 0.8.7`、`quadprog 0.1.13`。脚本刻意不安装 NumPy 和 Pinocchio，真机继续使用已有 NumPy 1.26.4 与 `/opt/ros/humble` 的 Pinocchio 3.9.0。

已检查的真机目前把上述五个包安装在 `~/.local`。机器人模板显式设置 `S4_HW_TELEOP_ALLOW_USER_SITE=1`，因此精确版本和 Pinocchio 来源检查通过后可以直接使用；项目局部目录仍是新安装时更隔离的首选。真机用户目录还存在 `pin/libpinocchio 4.1.0`，运行时必须看到 doctor 打印的实际 Pinocchio 路径位于 `/opt/ros/humble`。

真机当前 `pip check` 会报告 `cmeel-boost 1.90.0` 希望使用 NumPy 2，以及 PyNaCl 缺少 cffi；这是 `~/.local` 中其他 PyPI 包留下的环境冲突。不要为了消除此提示升级系统 NumPy、Pinocchio 或 cmeel，否则可能破坏 ROS ABI。遥操入口会对实际导入路径和精确版本做门禁；后续若需彻底隔离，使用项目局部安装目录而不是继续改全局用户 site。

### 真机 SDK 当前部署状态

只读核查确认真机没有 `/usr/bin/sn_loco_server` 和 `/etc/qi-sdk`，所以源码树中的 `debian/start_sn_loco.sh` 当前不能直接使用。已经编译且包含 mode-5 策略腿融合的 ELF 位于：

```text
/home/coral/nanshan_south/qi_sdk_internal/install/qi_sdk/bin/sn_loco_server
```

它的配置位于 `qi_sdk_internal/install/config/`，默认 `lo`、Domain 16，动态库检查无缺失。应继续使用机器人现有、已经验证的 SDK/站立控制启动流程；不要把 `debian/start_sn_loco.sh` 当成已安装服务。遥操和在线 doctor 会核查**实际运行进程**的二进制，不限定它必须安装在 `/usr/bin`。

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

直接在已检查的机器人电脑运行时，复制专用覆盖模板：

```bash
cp hardware_teleop/config/ros_env.robot.example.sh \
   hardware_teleop/config/ros_env.local.sh
```

该模板使用 `lo`、ROS Domain 16、CycloneDDS 和 `/usr/bin/python3`；Quest HTTPS 仍通过 `wlp44s0` 的 `192.168.110.35` 访问。

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
export HW_TELEOP_NETWORK_INTERFACE=wlp44s0
```

---



## 配置文件说明



### `config/ros_env.sh` — ROS2 / DDS 环境


| 变量                             | 默认值                  | 说明               |
| ------------------------------ | -------------------- | ---------------- |
| `HW_TELEOP_ROS_DISTRO`         | `/opt/ros/humble`    | ROS2 安装路径        |
| `HW_TELEOP_NETWORK_INTERFACE`  | `enp47s0`            | CycloneDDS 绑定的网卡 |
| `HW_TELEOP_RMW_IMPLEMENTATION` | `auto` | 优先 CycloneDDS，未安装时使用 Fast DDS；也可显式指定 |


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
| `arm_kp` / `arm_kd`       | `60.0` / `3.0` | 与已核查 qiling 桥一致的手臂 PD 增益       |
| `reversed_joint_names`    | 见 yaml         | 与真机符号约定一致的 3 个翻转关节               |
| `max_joint_step_rad`      | `0.003`        | 首测单周期关节限幅（30 Hz 下最大 0.09 rad/s） |
| `initial_state_timeout_s` | `15.0`         | 等待首帧 lowstate 超时                 |
| `stale_command_hold`      | `true`         | Quest 输入 stale 时停止手臂运动           |
| `max_state_age_s`         | `0.2`          | lowstate 断流超过此时间则停止发送 lowcmd       |
| `max_state_joint_jump_rad` | `0.35`       | 拒绝单帧关节位置突跳                         |
| `input_stale_timeout_s`   | `0.12`         | 真机 Quest 输入超时                         |
| `max_tcp_translation_speed_m_s` | `0.08` | 首测 TCP 目标平移速度上限                    |
| `max_tcp_rotation_speed_rad_s` | `0.30`  | 首测 TCP 目标旋转速度上限                    |
| `command_watchdog_timeout_s` | `0.10` | 主循环无心跳后触发策略目标回退 |
| `release_duration_s` | `10.0` | mode 5 平滑释放的最长时间 |


#### `startup` 段

默认启动不会自动移动双臂。完成 shadow 和单臂验证后，才可按需启用任务 `home_pose` 插值。

“启动 homing”不是电机找机械零点，而是收到可信 lowstate 和站立策略数据后，以受限关节步长把双臂从当前实测姿态插值到任务 `home_pose`。启用方式是复制一份 `quest_hardware.yaml`，将 `startup.move_to_home` 改为 `true`，再用 `--hardware-config` 指向该文件；`--skip-homing` 始终可在本次运行强制跳过。不要在未完成 shadow、方向核对和单臂小步长验证前启用。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `move_to_home` | `false` | 是否执行启动 homing；首次真机保持关闭 |
| `duration_s` | `4.0` | homing 最长时长（秒） |
| `max_joint_step_rad` | `0.01` | homing 每周期最大关节步长 |
| `position_tolerance_rad` | `0.02` | 到达 home 的容差 |
| `check_lowcmd_publishers` | `true` | 图中超过一个已有 lowcmd publisher 时拒绝启动 |
| `require_policy_lowcmd` | `true` | 发 mode 5 前必须实际收到有效策略腿命令 |
| `policy_min_valid_frames` | `90` | 启动所需连续有效非 mode-5 帧数 |
| `max_policy_age_s` | `0.2` | 策略断流后停止发送 mode-5 的门限 |
| `policy_stable_duration_s` | `3.0` | IMU、腿速和跟踪误差必须连续稳定的时间 |
| `require_sdk_mode5_merge` | `true` | 校验运行中的 SDK 二进制含策略腿融合实现 |
| `approved_sdk_sha256` | 见 yaml | 只允许已审查的 SDK 可执行文件摘要 |

真机站立时本来就应存在一个策略 `lowcmd` 源。程序不只看 ROS graph：还会订阅同一话题，确认连续收到 `mode_ctrl != 5`、26 电机齐全、12 个腿电机启用且数值有限的策略包。若启动前看到另一个 mode-5 包，则判定旧遥操仍在运行并拒绝启动。

#### `gravity_compensation` 段

手臂 lowcmd 的 `motor.tau` 会叠加 Pinocchio 重力补偿（逻辑 vendored 自 qiling `moveit_mit_arm_bridge`）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | 首次真机默认关闭；单臂确认方向和力矩后再启用 |
| `urdf_path` | `assets/my_robot/urdf/s4_40dof_merged.urdf` | 动力学 URDF |
| `scale` | `0.6` | 重力力矩缩放（与 qiling 真机默认一致） |
| `tau_limit` | `12.0` | 单关节力矩限幅（N·m） |
| `ramp_time` | `2.0` | 启动后重力补偿从 0 渐增到满量程的时间 |
| `source` | `current` | 用 `current`（实测关节）或 `target`（指令关节）算重力 |

重力补偿初始化失败时会打印警告并禁用前馈力矩；但 Pink 主流程本身仍要求 ROS Pinocchio 可用，doctor 不通过时禁止启动真机遥操。

重力补偿没有命令行开关。测试时应复制硬件 YAML，将 `gravity_compensation.enabled` 改为 `true`，启动时使用 `--enabled-arms left|right`、较小的 `--max-arm-step-rad` 和默认 ramp，只验证一条手臂的力矩方向、静态下垂和急停。`--shadow` 不创建命令 publisher，因此不能验证实际补偿力矩。


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
| `backend` | `pink`（默认） | 仓库内置 Pink + Pinocchio，无 Isaac/GPU 依赖 |
|           | `rmpflow` / `pinocchio` | 仅供旧 `teleop-hardware-isaac` 回退入口 |




#### `scene` 段

仅供旧 `teleop-hardware-isaac` 入口使用；纯 Pink 入口不会读取或创建 Isaac 场景。

### `configs/teleoperation/meta_quest3.yaml` — 映射与手感

真机与仿真共用（通过 `teleop_config` 引用），常用项：


| 段                         | 作用                   |
| ------------------------- | -------------------- |
| `network.port`            | Quest 服务端口，默认 `8443` |
| `network.stale_timeout_s` | 超过 1 s 无有效帧 → 冻结双臂   |
| `mapping.position_scale`  | 手柄位移放大，当前为 `2.0`      |
| `mapping.clutch`          | Grip 离合阈值            |
| `safety.workspace_*`      | TCP 工作空间限制           |
| `smoothing.arm_*`         | 手臂指令平滑               |


---



## 启动与使用



### 标准启动流程

先做静态 doctor；机器人 SDK/站立策略启动并开始发状态后，再做只读在线检查：

```bash
bash run.sh teleop-hardware-doctor --robot-profile --require-daqp
bash run.sh teleop-hardware-doctor --robot-profile --require-daqp --require-live-state
```

第二条不会创建任何 publisher，会打印 `/lowcmd` 现有 publisher，并核查运行中的 `sn_loco_server` 是否包含新版 mode-5 融合实现。真正启动时还会等待有效策略腿数据，不能只凭节点名放行。

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
ros2 topic hz lowcmd            # 启动遥操前就应看到站立策略；启动后为策略+30Hz mode-5 混合流
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
[HW-PINK] verified SDK mode5 merge: pid=... executable=...
[HW-PINK] runtime=hardware_no_isaac control_rate_hz=30.0 state=lowstate ...
[HW-PINK] waiting for lowstate (timeout=15.0s)...
[HW-TELEOP] standing-policy lowcmd ready: 3 valid frames, age=...s
[HW-PINK][IK] details={'backend': 'pink', 'runtime': 'hardware_no_isaac', ...}
[HW-PINK][FK] initial_tcp_L=(...) initial_tcp_R=(...)
[HW-PINK] Meta Quest controller server ready
[HW-PINK] Quest URL: https://192.168.x.x:8443
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
| `lowcmd`    | `qi/LowCmd`   | 订阅 + 发布 | 站立策略 + 本模块 | 策略持续发非 5，本模块发 mode 5 |
| `/handscmd` | `qi/HandsCmd` | 发布  | 本模块     | 同上                 |


> 注意：`lowcmd` 默认 **无** 前导 `/`；手部话题为 `/handscmd`。
> SDK 配置中的原生 DDS 名称 `rt/lowstate`、`rt/lowcmd`、`rt/handscmd` 是 ROS2 自动添加 `rt/` 前缀后的名称；项目 YAML 不能再写一次 `rt/`。



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

1. 在创建命令桥前，验证运行中的 SDK 二进制包含 mode-5 策略腿融合实现
2. 检查 graph 中没有多个未知 `lowcmd` 源
3. 等待并校验首批 `lowstate`
4. 等待连续、有效、新鲜的站立策略腿命令，证明 SDK 腿缓存已建立
5. 默认保持当前实测姿态，不自动 homing
6. 进入主循环，等待 Grip 离合遥操

每 **30 Hz** 循环一次：

1. **spin ROS** — 处理 `lowstate` 回调，更新实测臂关节角
2. **Pink FK** — 直接用符号转换后的 LA7+RA7 实测关节计算双臂 TCP
3. **读取 Quest 帧** — `BimanualTeleopMapper` 输出 TCP 目标 + 手部 trigger
4. **IK** — clutch 按下时调用双臂 Pink QP（含肘部 barrier 与关节限制）
5. **组 26D 指令** — 未 clutch 侧保持当前实测关节；手部始终跟 trigger
6. **平滑 + 限幅** — `smooth_command` + `max_joint_step_rad`
7. **发布** — `lowcmd`（臂）+ `/handscmd`（手）



### 安全机制


| 机制                   | 说明                              |
| -------------------- | ------------------------------- |
| 离合门控                 | 未按 Grip 不运动该臂                   |
| stale 冻结             | Quest 超 0.25 s 无有效帧 → 释放离合并回锚实测姿态 |
| 关节步长限制               | 最终每周期不超过 0.020 rad；首测建议 0.005 |
| workspace 限制         | mapper 内 TCP 边界 + 真机 0.5 m/s、1.5 rad/s 目标速度限制 |
| JointStateFrameGuard | 拒绝 NaN、异常全零和大于 0.35 rad 的单帧突跳 |
| 初始状态等待               | 未收到 lowstate 不进入控制循环            |
| 策略腿缓存门禁             | 无连续有效策略腿包时绝不发送 mode-5          |
| SDK 版本门禁              | 运行二进制缺少融合标记时拒绝命令输出          |
| 运行期 publisher 监控     | 外部 `lowcmd` 源增至两个以上时锁存故障并停止本模块输出 |
| 反馈/策略断流             | 超过 0.2 s 停止发送 lowcmd，让 SDK 回到策略控制 |
| 重启离合回锚              | Grip 松开或输入 stale 后，以实测关节重置步进基准 |
| 启动 homing              | 默认关闭，完成分阶段验证后才启用              |
| 重力补偿                  | 默认关闭，确认方向后再使用 ramp               |




### 急停建议

本模块 **没有** 独立硬件急停 topic。建议：

- 松开双手 Grip + 关闭遥操进程
- 或切断机器人物理急停
- 正常保留一个站立策略 `lowcmd` 源，同时确保旧遥操、MoveIt 和 replay 控制器已停止

---



## IK 后端切换



### 默认：Pink（无 Isaac）

- 直接读取真机 LA7+RA7 反馈做 Pinocchio FK
- 复用仿真 Pink 权重、TCP offset、关节限制和肘部 PositionBarrier
- QP 默认使用 `quadprog`；此前 `qpsolvers 4.12.0 + daqp 0.7.2` 的 API 不兼容已通过固定 `daqp 0.8.7` 解决，DAQP 保留为可验证备选，不会自动改变当前 solver

```bash
bash run.sh teleop-hardware
# 或在 quest_hardware.yaml 中 ik.backend: pink
```

只读 shadow（不会创建 lowcmd 或手部 publisher）：

```bash
bash run.sh teleop-hardware --shadow --skip-homing --input-debug
```

旧 Isaac/RMPflow 回退：

```bash
bash run.sh teleop-hardware-isaac
```

---



## 命令行参数

通过 `bash run.sh teleop-hardware [参数]` 直接传递：


| 参数                               | 说明                                                        |
| -------------------------------- | --------------------------------------------------------- |
| `--hardware-config PATH`         | 硬件配置 yaml，默认 `hardware_teleop/config/quest_hardware.yaml` |
| `--ik-backend pink`             | 覆盖 yaml 中的纯运行时 IK 后端                                    |
| `--host HOST` / `--port PORT`    | Quest 服务绑定地址（默认读 meta_quest3.yaml）                        |
| `--cert PATH` / `--key PATH`     | 覆盖 HTTPS 证书和私钥路径                                             |
| `--insecure-http`                | 仅桌面调试，Quest WebXR 不可用                                     |
| `--report-period-s 0.5`          | 状态日志周期                                                    |
| `--max-runtime-s N`              | N 秒后自动退出，0 为一直运行                                          |
| `--input-debug`                  | 打印详细输入/跟踪信息                                               |
| `--shadow`                       | 只读状态和计算 IK，不创建任何命令 publisher                         |
| `--arm-output`                   | 完成预检后显式允许真机 mode-5 输出                                  |
| `--enabled-arms left|right|both` | 分级联调时只允许指定手臂运动                                        |
| `--enable-hands`                 | 显式启用手部命令；默认关闭                                          |
| `--skip-homing`                  | 跳过启动回零                                                        |
| `--max-arm-step-rad N`           | 设置不大于 YAML 上限的更严格单周期关节步长                           |
| `--record-state-jsonl PATH`      | 记录 q14、FK、目标和命令，供离线 Pink replay                         |
| `--overwrite-state-log`          | 明确允许覆盖已存在的 state JSONL；默认拒绝                            |
| `--allow-existing-lowcmd-publishers` | 仅在人工确认 graph 中多个 publisher 身份后放宽“最多一个”检查；策略数据门禁仍生效 |
| `--allow-no-policy-lowcmd` | **危险**：绕过策略腿缓存门禁，仅限固定工装 |
| `--allow-unverified-sdk-mode5-merge` | **危险**：绕过 SDK 融合版本校验，不用于正常站立真机 |


---



## 首次联调 Checklist

按顺序执行，避免一上来就动真机：

- [ ] **1. 机器人上电**，SDK 与站立策略正常运行
- [ ] **2. 配置网卡** — 编辑 `config/ros_env.sh` 中 `HW_TELEOP_NETWORK_INTERFACE`
- [ ] **3. source 环境** — `source hardware_teleop/scripts/source_ros_env.sh`
- [ ] **4. 编译 qi 消息** — `bash run.sh teleop-hardware-build`（若未做过）
- [ ] **5. doctor 在线检查** — `--require-live-state` 有稳定 `/lowstate`，识别策略 publisher，并验证正在运行的 SDK 二进制
- [ ] **6. 证书** — `bash run.sh teleop-cert --ip <Quest可访问的IP>`
- [ ] **7. shadow 短跑** — `bash run.sh teleop-hardware --shadow --max-runtime-s 30`
- [ ] **8. 验证 lowcmd** — 启动遥操前已存在稳定站立策略流；启动后是策略流叠加 30 Hz mode-5，不能按总频率等于 30 Hz 判断
- [ ] **9. Quest 连接** — 打开 `https://<IP>:8443`，确认 session 日志
- [ ] **10. 单臂小范围** — `--arm-output --enabled-arms left --max-arm-step-rad 0.003`，确认方向与离合
- [ ] **11. 故障注入** — 断开 Quest、停止浏览器和 Ctrl-C，确认看门狗平滑释放 mode 5
- [ ] **12. 双手 + Trigger** — 单臂验证完成后再使用 `--enable-hands`

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
- 保留唯一站立策略源，并确认没有旧遥操、MoveIt 或 replay 控制器并发发 `lowcmd`
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
python3 -m py_compile hardware_teleop/pink_main.py hardware_teleop/ros/robot_bridge.py
```



### 相关入口（项目根）


| 命令                                  | 作用                |
| ----------------------------------- | ----------------- |
| `bash run.sh teleop-hardware`       | 启动真机遥操            |
| `bash run.sh teleop-hardware-isaac` | 旧 Isaac/RMPflow 回退   |
| `bash run.sh teleop-hardware-build` | 编译 qi 消息          |
| `bash run.sh teleop-hardware-system-prepare` | 检查/安装真机项目局部依赖 |
| `bash run.sh teleop-hardware-doctor` | 检查 Python、Pink/QP、ROS/qi 和可选在线状态 |
| `bash run.sh teleop-hardware-env`   | 打印 source 命令      |
| `bash run.sh teleop-cert`           | 生成 Quest HTTPS 证书 |
| `bash run.sh teleop`                | 仿真遥操（非本模块）        |


---



## 快速参考

```bash
# 一次性
bash run.sh teleop-hardware-system-prepare --check
# 用户确认后：
bash run.sh teleop-hardware-system-prepare --install
bash run.sh teleop-hardware-build
cp hardware_teleop/config/ros_env.robot.example.sh hardware_teleop/config/ros_env.local.sh
bash run.sh teleop-cert --ip 192.168.110.35 --overwrite

# 静态检查；SDK 启动后再加 --require-live-state
bash run.sh teleop-hardware-doctor --robot-profile --require-daqp

# 每次 — 调试
source hardware_teleop/scripts/source_ros_env.sh
ros2 topic hz lowstate

# 首次只读 shadow
bash run.sh teleop-hardware --shadow --skip-homing --input-debug

# 分级放开单臂小步长
bash run.sh teleop-hardware --arm-output --enabled-arms left --skip-homing --max-arm-step-rad 0.003

# Quest
# https://192.168.x.x:8443
# Grip = 臂离合 | Trigger = 手
```
