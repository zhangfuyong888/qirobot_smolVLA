# S4 SmolVLA 工程文档

工程文档被有意收敛为三份。真实接口以 `run.sh` 和当前任务配置为准，文档不再为每个命令、模块或历史问题单独建文件。

## 阅读入口

| 文档 | 内容 | 适合读者 |
|---|---|---|
| [README.md](../README.md) | 项目介绍、从克隆到首次 Rollout | 第一次使用项目的人 |
| [REPRODUCTION.md](REPRODUCTION.md) | 双 Conda 环境、版本导出、外部仓库、资产、模型、数据与 checkpoint 部署 | 负责安装、迁移和交付的人 |
| [PIPELINE.md](PIPELINE.md) | 采集、转换、检查、训练、Rollout、核心契约和故障定位 | 负责实验和成功率优化的人 |

课程教程位于 [course/](course/index.md)，它不是命令手册，而是按“原理 → 实现 → 部署”讲解整个系统。

## 当前活动基线

| 项目 | 当前配置 |
|---|---|
| Task | `drawer_insert_close` |
| Language contract | `drawer_12phase_v4_serial_acquire` |
| Dataset | `s4_drawer_insert_close_v4_12phase_serial_acquire` |
| State/action | 26D / 26D absolute joint target |
| Cameras | 胸前、左腕、右腕 RGB |
| Dataset/control rate | 20 Hz / 120 Hz |
| Expert/language phases | 27 / 12 |
| Drawer reset | 固定关闭，`0.00 m` |
| Can sampling | 5×5 分层网格内连续随机 |

配置的唯一事实来源：

```text
configs/tasks/drawer_insert_close.dataset.json
configs/tasks/drawer_insert_close.scripted.yaml
configs/tasks/drawer_insert_close.smolvla.yaml
```

## 统一命令入口

```bash
bash run.sh help
bash run.sh doctor
bash run.sh sim
bash run.sh record --episodes 5
bash run.sh collect-convert --help
bash run.sh dataset-check --help
bash run.sh train --help
```

`run.sh` 负责选择正确的 Python 环境。不要在同一解释器中同时导入 Isaac Sim 和当前 LeRobot/SmolVLA 依赖。

当前在线 Rollout 也需要两个环境：`env_isaaclab` 运行仿真，`smolvla` 运行本机 Policy Server。它不是只安装 IsaacLab 就能运行的单环境入口。

当前离线 `preview` 入口存在一个已确认的路径限制：必须从项目根目录使用
`PYTHONPATH="$PWD" bash run.sh preview ...`。其他统一入口不需要这个前缀；详情见
[PIPELINE.md](PIPELINE.md#10-离线预览)。

## 文档维护规则

更新项目时只维护：

1. 根 `README.md` 的首次使用路径；
2. `REPRODUCTION.md` 的安装、版本和资源交付；
3. `PIPELINE.md` 的接口、契约、命令与诊断；
4. `course/` 中受变更影响的教程内容。

不要重新建立按脚本、按问题或按历史实验拆分的 Markdown 知识库。历史实验可以保存在 Git 历史、结构化日志或实验输出中，不应混入当前操作说明。
