# S4 SmolVLA IsaacLab

这是一个基于 **Isaac Sim 5.1、IsaacLab、LeRobot 和 SmolVLA** 的双臂机器人仿真学习工程。当前主线任务 `drawer_insert_close` 使用左手拉开抽屉，右手抓取罐子并放入抽屉，随后关闭抽屉并返回 Home。

项目提供从专家数据采集到在线闭环评估的完整接口：

```text
IsaacLab 专家策略
→ HDF5 成功轨迹
→ LeRobotDataset
→ SmolVLA 训练
→ checkpoint
→ Policy Server
→ IsaacLab Rollout
```

> 当前活动版本是 `drawer_12phase_v4_serial_acquire`。预抓握、接触抓握和后续搬运动作使用清晰边界，左右臂主体运动串行执行。旧数据集或旧 checkpoint 不能仅凭 26D 维度相同就认为与当前版本兼容，必须通过 `dataset-check` 检查。

## 1. 当前技术契约

| 项目 | 当前值 |
|---|---|
| 活跃任务 | `drawer_insert_close` |
| Schema | `s4_bimanual_v1` |
| 语言契约 | `drawer_12phase_v4_serial_acquire` |
| 专家控制阶段 / 语言阶段 | 27 / 12 |
| State / Action | 26D / 26D |
| Action 语义 | `absolute_joint_target` |
| 关节顺序 | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 |
| 相机 | 胸前、左腕、右腕三路 RGB，480×680 |
| 控制频率 / 数据频率 | 120 Hz / 20 Hz |
| Action Chunk | 50 个策略帧 |
| 默认在线重规划 | 40 个策略帧 |
| 抽屉初始开度 | 固定 `0.00 m` |
| 主罐随机化 | 5×5 分层网格内连续随机 |
| 当前数据集 | `s4_drawer_insert_close_v4_12phase_serial_acquire` |

契约来源是：

- `configs/tasks/drawer_insert_close.dataset.json`
- `configs/tasks/drawer_insert_close.scripted.yaml`
- `configs/tasks/drawer_insert_close.smolvla.yaml`

## 2. 仓库与外部资源

推荐目录结构：

```text
workspace/
├── smolVLA/                         # 顶层 Git 仓库
│   ├── s4_smolvla_isaaclab/         # 本项目
│   └── lerobot/                      # 固定 commit 的 Git submodule
└── IsaacLab/                         # 外部 IsaacLab checkout
```

Git 仓库不包含以下大文件：

- Isaac Sim 和外部 IsaacLab；
- `local_assets/` 场景资产包；
- SmolVLM2 基础模型；
- HDF5、LeRobotDataset；
- 训练 checkpoint 和 Rollout 输出。

这些资源必须单独分发，并由 `.env` 指向实际路径。

## 3. 从克隆到首次 Rollout

以下步骤面向“已有训练数据集和 checkpoint，希望直接复现 Rollout”的使用者。完整环境导出、资产制作和自主训练见 [复现与部署](docs/REPRODUCTION.md)。

> **只做 Rollout 也必须安装两个环境。** `env_isaaclab` 运行仿真、相机和机器人控制；`smolvla` 运行本机 Policy Server，负责加载 checkpoint、读取 LeRobotDataset 的阶段信息并生成动作。当前 `run.sh rollout` 会从 `S4_SMOLVLA_PREFIX/bin/python` 启动这个子进程，因此缺少任意一个环境都无法完成当前在线 Rollout。只有未来实现远程 Policy Server 后，仿真工作站才可能只安装 `env_isaaclab`。

### 3.1 克隆代码和 LeRobot

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL> smolVLA
cd smolVLA
git submodule status
cd s4_smolvla_isaaclab
```

如果之前没有递归克隆，在顶层 `smolVLA/` 执行：

```bash
git submodule update --init --recursive
```

不要自行把 `lerobot/` 更新到最新分支；当前验证 commit 记录在 `environment/versions.md`。

### 3.2 准备外部 IsaacLab

安装与 Isaac Sim 5.1 匹配的 IsaacLab checkout，并保持下列入口存在：

```text
/path/to/IsaacLab/isaaclab.sh
```

当前已记录的 IsaacLab commit 和包版本见 `environment/versions.md`。

### 3.3 创建两个 Conda 环境

仿真环境：

```bash
conda env create -f environment/isaaclab.yml
conda activate env_isaaclab
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com

cd /path/to/IsaacLab
./isaaclab.sh --install none
```

SmolVLA 环境：

```bash
cd /path/to/smolVLA/s4_smolvla_isaaclab
conda env create -f environment/smolvla.yml
conda activate smolvla
python -m pip install -e /path/to/smolVLA/lerobot
```

`run.sh` 会自动选择环境：场景、采集和 Rollout simulator 使用 Python 3.11；转换、检查、训练和 Policy Server 使用 Python 3.12。

### 3.4 放置外部资源

将维护者提供的场景资产解压为：

```text
s4_smolvla_isaaclab/local_assets/isaac/5.1/
├── Isaac/...
└── manifest.json
```

将基础模型放到：

```text
s4_smolvla_isaaclab/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/
```

如果要直接 Rollout，还需要收到彼此匹配的：

```text
datasets/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire/
outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/
```

### 3.5 配置本机路径

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
S4_PROJECT_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab
ISAACLAB_ROOT=/path/to/IsaacLab
ISAAC_ASSET_ROOT=/path/to/isaacsim_assets/Assets/Isaac/5.1
S4_SCENE_ASSET_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab/local_assets/isaac/5.1
LEROBOT_ROOT=/path/to/smolVLA/lerobot
SMOLVLA_MODEL_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab/models
S4_DATA_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab/datasets
S4_OUTPUT_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab/outputs
S4_CACHE_ROOT=/path/to/smolVLA/s4_smolvla_isaaclab/.cache
S4_ISAACLAB_ENV=env_isaaclab
S4_SMOLVLA_ENV=smolvla
```

`.env` 是本机文件，不要提交到 Git。

### 3.6 逐层检查

没有数据集和 checkpoint 时：

```bash
bash run.sh doctor
bash run.sh list-tasks
bash run.sh activate-task drawer_insert_close
```

场景检查：

```bash
bash run.sh sim
```

确认环境材质、光照、机器人、抽屉、罐子和相机正常后退出。

已有数据集与 checkpoint 时，执行严格检查：

```bash
bash run.sh doctor --strict

bash run.sh dataset-check \
  datasets/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/<STEP>/pretrained_model
```

### 3.7 启动 Rollout

有渲染窗口的固定场景回归：

```bash
bash run.sh rollout \
  --deterministic \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/<STEP>/pretrained_model \
  --dataset-root datasets/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire \
  --chunk-replan-frames 40 \
  --chunk-overlap-blend-frames 5 \
  --phase-transition-blend-frames 8 \
  --phase-max-extension-frames 20 \
  --drawer-phase-max-extension-frames 80 \
  --policy-device cuda
```

20 轮随机主罐位置成功率：

```bash
bash run.sh rollout \
  --success-rate 20 \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/<STEP>/pretrained_model \
  --dataset-root datasets/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire \
  --chunk-replan-frames 40 \
  --chunk-overlap-blend-frames 5 \
  --phase-transition-blend-frames 8 \
  --phase-max-extension-frames 20 \
  --drawer-phase-max-extension-frames 80 \
  --policy-device cuda
```

加上 `--headless` 可隐藏窗口，但三路相机仍会渲染。输出写入 `outputs/eval/rollout_<timestamp>_.../`，包括视频、动作 CSV、诊断图和 `summary.json`。

## 4. 自主采集和训练

小规模有界面采集：

```bash
bash run.sh record --episodes 5
```

确认动作和画面正常后，可以使用一体化入口完成“采集 → HDF5 检查 → 转换 →
LeRobotDataset 检查”。有渲染窗口时不要传 `--headless`：

```bash
bash run.sh collect-convert \
  --episodes 200 \
  --random-seed 42 \
  --episode-timeout-s 300 \
  --reset-settle-s 2.0 \
  --record-every-n 6 \
  --max-failed-attempts 10
```

`collect-convert` 不接受末尾附加的数据集路径或 `--expected-episodes`；它会从活动任务配置
解析目标数据集，并在转换前后自动使用期望成功数检查。如果目标 LeRobotDataset 已存在，
只有在明确接受替换时才额外加入 `--overwrite`。无界面正式采集只需额外加入
`--headless`。

采集中的普通任务失败不会写入 `demo_*`，而是丢弃当前内存 episode、记录失败并重置：

- 抓罐相关失败先在同一精确点额外重试 3 次；
- 仍失败后在当前 5×5 网格单元内部重新随机点位；
- 非抓取阶段失败直接在当前格内重新采样；
- 当前格只有接受成功 episode 后才推进；
- `--max-failed-attempts 10` 允许累计 10 次失败，第 11 次失败才中断；已经提交的成功
  HDF5 episode 和失败日志仍会保留，但本次一体化命令不会继续转换。

完整的安全顺序是：

```text
采集成功 HDF5
→ 检查 HDF5
→ 转换 LeRobotDataset
→ 检查 LeRobotDataset
→ fresh training
→ checkpoint 契约检查
→ 固定场景 Rollout
→ 随机场景成功率
```

不要直接照抄一个带 `--overwrite` 的总命令。正式数据应使用独立输出路径，并在每个破坏性操作前确认目标。完整命令见 [完整流水线](docs/PIPELINE.md)。

## 5. 文档

工程文档只保留三份：

- [文档索引](docs/README.md)
- [复现与部署](docs/REPRODUCTION.md)
- [完整流水线、契约与诊断](docs/PIPELINE.md)

课程教程保留在 [docs/course/](docs/course/SMOLVLA_ADVANCED_TECHNICAL_COURSE.md)，按“原理 → 项目实现 → 部署”分为三章。

## 6. 重要安全边界

- `--episodes N` 表示目标成功 episode 总数，不是总尝试数。
- 失败尝试写入失败日志，不进入最终训练数据。
- `--max-failed-attempts N` 是全局失败安全预算，不决定网格点如何选择；超过预算才中断。
- Resume 必须使用同一个 HDF5，并保持采集契约不变。
- `--overwrite` 只应在明确接受替换目标时使用。
- State/action 顺序、相机 key、FPS、Action 语义或语言契约变化后，需要重新转换、重新训练并重新 Rollout。
- IK 可达不等于物理抓取成功；离线误差低不等于在线成功率高。
- 数据集和 checkpoint 必须通过契约检查，不能只看目录名或 tensor 维度。

## 7. 真实接口

随代码变化时，以当前命令和配置为准：

```bash
bash run.sh help
bash run.sh collect-convert --help
bash run.sh train --help
```

核心入口是 `run.sh`；外部 LeRobot 作为固定 submodule 使用，本项目不要求修改 LeRobot 源码。
