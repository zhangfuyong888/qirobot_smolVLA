# SmolVLA 高级技术教程：从 IsaacLab 仿真数据到 VLA 闭环控制

> 文档状态：本页是课程总索引。原单文件教程已按“原理 → 实现 → 部署”拆分为三个递进章节；工程事实以当前代码和配置为准。

## 课程定位

本教程以双臂灵巧手任务 `drawer_insert_close` 为贯穿案例，讲解如何把 Isaac Sim/IsaacLab 中的机器人操作任务转化为可供 Vision-Language-Action（VLA）模型学习的多模态时序数据，并完成脚本化专家控制、HDF5 采集、LeRobotDataset 转换、SmolVLA 训练、离线评估和在线闭环 Rollout。

课程不逐行解释源码，而是围绕三条主线展开：

1. **原理线**：模型为什么这样设计，视觉、语言、状态和动作怎样形成学习问题；
2. **实现线**：任务、专家、数据、模型和在线控制怎样通过稳定契约连接；
3. **部署线**：如何准备两个 Python 环境、外部仓库、资产、模型和运行目录，并验证整条链路。

当前项目只研究仿真 VLA，不展开仿真与真机联合数据训练。

## 适用读者与前置知识

本文面向已经掌握以下内容的读者：

- 机器人关节、连杆、自由度和执行器；
- 正运动学、逆运动学和坐标变换；
- 机械臂与灵巧手的基础控制；
- Python、Linux、Conda 和深度学习训练的基本使用。

教程只在 TCP、IK、关节目标、重力补偿和执行器跟踪处做必要衔接，不重新推导基础机器人学。

## 学习目标

完成三章后，读者应能够：

- 解释 VLA 和 SmolVLA 的输入、输出、Action Chunk 与 Flow Matching；
- 说明 Isaac Sim、IsaacLab、LeRobot 和 SmolVLA 的职责边界；
- 设计一个可采集、可学习、可自动评估的仿真操作任务；
- 理解脚本专家、随机化和数据质量之间的关系；
- 理解 HDF5 与 LeRobotDataset 的字段映射；
- 检查训练、离线评估和在线 Rollout 的接口是否一致；
- 在另一台工作站上配置项目路径、环境、资产和模型；
- 根据动作日志、任务阶段和物理状态定位成功率问题。

## 三章目录

| 顺序 | 章节 | 核心问题 | 建议读者 |
|---|---|---|---|
| 1 | [第一章：SmolVLA 原理与任务设计](01-principles.md) | SmolVLA 为什么能从图像、语言和状态预测动作？什么样的仿真任务适合学习？ | 第一次接触 VLA 或需要理解系统设计者 |
| 2 | [第二章：项目实现与端到端闭环](02-implementation.md) | 当前项目怎样完成专家控制、采集、转换、训练和 Rollout？ | 准备读代码、采集数据或优化成功率者 |
| 3 | [第三章：项目环境与完整部署](03-deployment.md) | 如何准备仓库、双环境、资产、模型、数据目录并完成部署验收？ | 需要复现、迁移或交付项目者 |

```mermaid
flowchart LR
    A[第一章<br/>原理与任务设计] --> B[第二章<br/>项目实现与闭环]
    B --> C[第三章<br/>环境与部署]
    C --> D[仿真任务]
    D --> E[专家数据]
    E --> F[LeRobotDataset]
    F --> G[SmolVLA 训练]
    G --> H[在线 Rollout]
    H --> I[成功率诊断]
    I -.优化反馈.-> D
```

## 推荐阅读路线

### 路线 A：系统学习

按第一章、第二章、第三章顺序阅读。该路线先建立模型与数据契约，再理解实现，最后部署。

### 路线 B：准备采集和训练

先阅读第二章中的“专家策略”“数据采集”“转换与检查”，再阅读第三章的环境验收和标准运行顺序；遇到 Action Chunk、Flow Matching 或归一化问题时回查第一章。

### 路线 C：迁移到新工作站

先完成第三章的仓库、环境、资产和模型准备，再按部署验收顺序执行。环境通过后，阅读第二章确认当前数据和 checkpoint 契约。

### 路线 D：优化 Rollout 成功率

直接阅读第二章的在线 Rollout、Raw/Fused/Command/Actual 和失败诊断，再回到第一章理解“可达不等于可抓”以及随机化覆盖原则。

## 当前技术基线

| 组件 | 当前记录 | 证据来源 |
|---|---|---|
| Isaac Sim | 5.1.0.0 | `environment/versions.md` |
| IsaacLab | 0.54.2，外部 checkout | `environment/versions.md` |
| LeRobot | 0.6.1，外部 submodule/check-out | 本地 LeRobot 源码与 `environment/versions.md` |
| 仿真环境 | Python 3.11，环境名 `env_isaaclab` | `environment/isaaclab.yml`、`run.sh` |
| 模型环境 | Python 3.12，环境名 `smolvla` | `environment/smolvla.yml`、`run.sh` |
| VLM 基座 | `SmolVLM2-500M-Video-Instruct` 本地目录 | 当前任务训练配置 |
| 活跃案例 | `drawer_insert_close` | `configs/tasks/` |

> 版本说明：`environment/versions.md` 是已验证工作站快照，不表示任意补丁版本都可互换。部署时应记录实际 commit、Python、CUDA、驱动和包版本。

## 证据标记

三章统一使用以下表述区分证据强度：

- **当前代码/配置**：可由当前实现直接证明；
- **测试契约**：测试代码定义了预期行为，但不代表本次已经执行；
- **官方资料**：来自 SmolVLA 论文、Hugging Face/LeRobot 或 NVIDIA 官方资料；
- **历史结果**：README 或旧实验留下的结果，只说明当时条件；
- **尚未验证**：有设计或实现依据，但缺少当前版本实验结果。

> 技术要点：IK 可达不等于物理抓取成功；低离线误差不等于在线闭环成功；训练完成也不等于数据、模型和 Rollout 契约一定匹配。

## 当前接口速照

下表用于防止阅读历史日志或旧文档时混淆当前实现。具体解释见第一章和第二章。

| 项目 | 当前实现 |
|---|---|
| 专家状态机 / 模型语言 | 27 个控制阶段 / 12 个语言宏阶段 |
| 主罐随机 | 默认启用，5×5 分层格内连续随机 |
| 抓取相关物体位移门控 | 20 mm |
| 同一位置重试 | 初始尝试之外额外重试 3 次 |
| 重试耗尽 | 留在同一格内重新采样，不跳过该格 |
| state/action | 26D，绝对关节目标 |
| 数据/控制频率 | 20 Hz / 120 Hz |
| Action Chunk | 50 个策略帧 |
| 在线重规划 | 默认每 40 个策略帧 |
| 训练保存频率 | 当前任务 YAML 默认 50000 steps |
| Headless | 隐藏 GUI，但三路相机仍需渲染 |
| 离线 preview | 当前只传入第一路视觉 feature |
| 最终成功条件 | 主要检查抽屉开度和罐子世界坐标 Z |

## 课程文件

```text
docs/course/
├── index.md                # 本索引
├── 01-principles.md        # 原理、架构、任务设计
├── 02-implementation.md    # 专家、数据、训练、Rollout
└── 03-deployment.md        # 环境、资产、配置、部署验收
```

## 官方参考资料

1. Mustafa Shukor et al. [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844), 2025.
2. Hugging Face LeRobot. [SmolVLA 官方文档](https://huggingface.co/docs/lerobot/smolvla).
3. Hugging Face LeRobot. [SmolVLA 官方实现](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla).
4. Hugging Face LeRobot. [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).
5. NVIDIA. [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/).
6. NVIDIA. [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/).

## 项目内部参考

- `README.md`、`run.sh`、`.env.example`：项目入口和路径配置；
- `configs/tasks/`：数据、专家和训练的真实配置；
- `docs/README.md`：精简后的工程文档索引；
- `docs/REPRODUCTION.md`：双环境、资产、模型与部署；
- `docs/PIPELINE.md`：采集、转换、训练、Rollout、契约与诊断。
