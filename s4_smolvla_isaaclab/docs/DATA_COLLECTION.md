# 数据采集

环境：`env_isaaclab`，由 `run.sh` 自动选择。

```bash
bash run.sh sim
bash run.sh record --episodes 10
bash run.sh record --episodes 200 --headless
```

物理/控制为 120 Hz，每 6 步采一帧，HDF5 为 20 Hz。`--headless` 关闭 GUI，
但 camera sensor 仍渲染三路 RGB；它不是“无相机”。每次 reset 后按 scripted
配置等待场景稳定。超时或 success criteria 失败的 episode 不写入文件，直到
达到请求的成功轮数。

## 采集随机化开关（与 rollout 对齐）

主开关在
[`drawer_insert_close.scripted.yaml`](../configs/tasks/drawer_insert_close.scripted.yaml)
的 `randomization`：

| 开关 | 当前默认 | 作用 |
|---|---|---|
| `can_xy.enabled` | **true** | 主抓取罐按 5×5 分层网格在验证范围内随机 XY |
| `distractor_cans.enabled` | **false** | **不生成**三个柜面 YCB 干扰物 |
| `drawer_initial_open.enabled` | true | 仍随机抽屉初始开度 `[0.00, 0.05]` m |

采集日志会打印：

```text
[RECORD] can_xy_randomization=True distractor_cans=False drawer_initial_open=True
```

写入 HDF5 / 转换后的 `meta/s4_contract.json` 会记录 `distractor_cans_enabled`
与当时的 `randomization` 快照。rollout 默认跟 contract：新采无干扰物数据不会
再塞回三个干扰罐；`can_xy.enabled=true` 时 `--success-rate` 会按同一网格/范围偏置主罐。

CLI 可临时覆盖 YAML（不必改文件）：

```bash
# 固定主罐、不要干扰物
bash run.sh record --episodes 10 --headless \
  --no-can-xy-randomization --no-distractor-cans

# 主罐随机 + 三个干扰物（旧配方）
bash run.sh record --episodes 10 --headless \
  --can-xy-randomization --distractor-cans
```

YAML 里仍保留已验证的 `can_xy` 范围与干扰物区域；干扰物仅在
`distractor_cans.enabled=true`（或 `--distractor-cans`）时生成。

在正式采集前可重复运行离线密集检查：

```bash
bash run.sh validate-workspace --grid 41 61
```

检查会覆盖每个采样点的预抓取、抓取和抬升目标，并拒绝 IK 不收敛、关节裕量
不足或雅可比条件过差的区域。它同时检查罐子底面与柜面边界的 5 mm 安全裕量。
验证器默认使用与采集相同的 DLS damping、posture gain 和最大关节步长。重力补偿、
执行器跟踪和接触属于 PhysX 动力学，不进入离线 IK；Isaac Sim 中的手指接触仍需用
小批量物理采集验证，离线 IK 不等价于碰撞仿真。

脚本化阶段严格区分手指和手臂动作：闭合命令完成并保持后才抬臂；释放时等待
实际手指张开、罐子位于抽屉内且线速度不超过 0.05 m/s，随后先竖直抬高手，
再向外退出，最后才关抽屉。

普通 `sim`、`teleop` 默认不生成干扰物。录制时是否生成由
`randomization.distractor_cans.enabled`（或 `--distractor-cans` /
`--no-distractor-cans`）决定。rollout 根据转换数据集的
`meta/s4_contract.json` 自动匹配，也可用同样的 CLI 覆盖。

场景光照是固定的，不参与随机化。预览、采集和 rollout 将任务区附近的仓库灯缩放
到原始强度的 18%，远处背景灯设为 55%，并共用低强度环境光、前侧柔光与 RTX
质量设置。这样可以提高环境亮度，同时避免白色机器人和桌面过曝，并保留颜色、
粗糙度和纹理细节。

默认输出：`${S4_DATA_ROOT}/staging/<dataset>/<task>_scripted.hdf5`。中断时已
flush 的完整 `demo_N` 通常可用，但必须运行：

当前采集仍执行 20 个精细专家控制阶段，但逐帧语言只使用 10 个宏阶段。HDF5
同时记录语言 ID、宏文本和专家阶段名，分别服务于训练与失败诊断。详见
[语言阶段契约](LANGUAGE_PHASES.md)。

采集中断后可在同一个 HDF5 上续采。`--episodes` 表示文件最终需要达到的成功
条数，而不是额外追加条数；已有 episode、随机数状态、网格游标、当前格内点和
当前格子状态都会恢复。抓取失败会在同一精确位置额外重试三次，然后在同一格换点；
非抓取失败直接在同一格换点。总体失败次数仍受 `--max-failed-attempts` 安全上限约束。
修复问题后可续采同一文件。示例：

```bash
bash run.sh record --output datasets/staging/s4_drawer_insert_close_v1_10phase/run.hdf5 \
  --episodes 20 --resume
```

```bash
bash run.sh dataset-check --hdf5
```

多进程采集使用 `bash run.sh record-parallel --num-episodes 100 --workers 2`。
每个 worker 是独立 Isaac Sim 进程；先从 2 个开始评估显存。

只执行“采集、校验、转换、再校验”而不训练：

```bash
bash run.sh collect-convert --episodes 200 --headless --overwrite
```

该命令不会调用训练脚本。默认把本轮 HDF5 写入带时间戳的独立目录；`--overwrite`
只用于替换目标 LeRobotDataset，不会删除刚采集的 HDF5。转换前会检查成功条数、
失败摘要和网格覆盖；默认只要存在跳过格子就停止，不会转换覆盖不完整的数据集。

每次失败会立即写入 HDF5 同目录下的 `*_failures.jsonl`，并在
`*_failure_summary.json` 中按阶段、类型、诊断归因和原因汇总。单次记录包含罐子
首次移动 5/10 mm 的阶段、最大 XY 位移、网格编号、TCP 目标/误差、右臂命令跟踪
误差、仿真实时率、抽屉开度、手指状态，以及重力补偿开关、尺度和实际补偿力统计。
阶段快照还记录罐子相对右手 TCP、五个右手末端指节及其中心的位置，可直接判断
闭手时是否包裹罐体中段。放置完成后，脚本会先确认右手完全松开再退出抽屉；左手
关抽屉后也会先张开、上抬并直线后退到把手外，再允许回 Home，避免回程勾开抽屉。
抬升失败还会区分没有抓牢或中途滑脱。进程异常退出
时已经落盘的失败记录仍保留。预抓取或抓取闭手前罐子位移超过 10 mm 会立即失败，
避免无意义地等待阶段跑满 1000 步。

执行带安全关卡的“采集、转换、检查、训练”完整流程：

```bash
bash run.sh collect-train --episodes 200 --headless \
  --overwrite-dataset --overwrite-training-output
```

任何采集未完成、失败报告不匹配、跳格、HDF5/LeRobot 契约错误都会在训练前停止。
失败尝试本身不会写入 HDF5；自动流程默认允许上述按阶段重试，并使用
`--max-failed-attempts 1000` 作为系统性故障的总安全上限。需要严格调试时仍可显式
传入 `--max-failed-attempts 0`，让第一次失败立即停止。

并行采集时，每个 worker 使用“基础随机种子 + worker id”，避免多个 HDF5 文件
重复同一套随机序列。单进程自动流程可以用 `--random-seed 42` 明确设置种子。
