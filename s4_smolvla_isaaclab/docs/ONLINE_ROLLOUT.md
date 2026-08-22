# 在线 Rollout

IsaacLab 主进程使用 `env_isaaclab`，policy server 使用 `smolvla`。

当前数据集使用 10 个语言宏阶段。Policy Server 从 `meta/s4_contract.json` 为
`tasks.parquet` 中的 prompt 恢复稳定 `language_phase_id`；Rollout 再按该 ID 找到
宏阶段末端的专家门控。20 个专家阶段不作为 20 次语言切换。默认每 40 个策略帧
重规划一次 50 帧 Action Chunk，并在重叠区融合；阶段切换仍使用 8 帧过渡。
`approach_drawer_handle` 和 `pull_drawer` 的门控扩展上限为 80 帧，其他宏阶段为 20 帧。

Rollout 启动前还会比较数据集 `s4_contract.json` 与训练目录
`s4_dataset_contract.json`；旧 `v0` checkpoint 不能与当前 10 阶段数据集混用。

## 输出目录约定

每次 rollout **只写一个子文件夹**（多轮随机测试也全部放在同一文件夹内）：

```text
outputs/eval/rollout_<YYYYMMDD_HHMMSS>_<det|randN>_ckpt<step>/
  rollout.avi                 # 单轮
  rollout_actions.csv
  rollout_actions.png
  ep001.avi                   # 多轮（全部在同一目录）
  ep001_actions.csv
  ep001_actions.png
  ...
  summary.json
```

- 不传路径时自动按时间戳命名，避免互相覆盖、避免 `outputs/eval/` 根目录堆文件。
- `--output-dir outputs/eval/my_rand20`：自定义本次运行目录。
- `--output-video outputs/eval/foo.avi`：兼容写法，实际写入 `outputs/eval/foo/`。

## 固定场景回归

```bash
bash run.sh rollout \
  --headless \
  --deterministic \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --policy-device cuda
```

或指定目录：

```bash
bash run.sh rollout \
  --headless \
  --deterministic \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --policy-device cuda \
  --output-dir outputs/eval/det_360k
```

`--deterministic` 展开为 `--no-randomize-task --seed 42`。`--seed` 无论是否随机化都默认且保持为 **42**。

## 随机化成功率评估

`--success-rate N` 打开任务随机化，但**每一项仍受 scripted YAML 的 `enabled`
开关约束**（与采集同一文件）：

| 变量 | YAML 开关 | 当前默认 |
|---|---|---|
| 罐子 XY 偏移 | `can_xy.enabled` | **true**（5×5 分层网格） |
| 抽屉初始开度 | `drawer_initial_open.enabled` | true（`[0.00, 0.05]` m） |
| 三个柜面干扰物 | `distractor_cans.enabled` + 数据集 `s4_contract.json` | **false**（不生成） |

因此在当前默认配置下，`--success-rate 20` 会随机**主罐 XY + 抽屉初开度**，
但不会放入三个干扰罐。这与新采集分布一致。

- 旧数据集若 `meta/s4_contract.json` 写有 `distractor_cans_enabled: true`，
  rollout 仍会按 contract 生成干扰物，以匹配当时训练视觉。
- 可用 `--distractor-cans` / `--no-distractor-cans` 显式覆盖 contract。
- 若要固定主罐评估，传 `--deterministic`，或 `--no-randomize-task`，或把 YAML
  `can_xy.enabled` 临时改成 `false`。
- `--deterministic` 关闭全部任务随机（罐/抽屉都固定）；种子仍为 **42**。

默认范围来自
[`drawer_insert_close.scripted.yaml`](../configs/tasks/drawer_insert_close.scripted.yaml)。
实验种子固定为 42；多轮共用同一 RNG 流，产物全部在同一 `output-dir`。

```bash
bash run.sh rollout \
  --headless \
  --success-rate 20 \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --policy-device cuda
```

自定义范围与目录（会重新打开罐 XY 随机）：

```bash
bash run.sh rollout \
  --headless \
  --episodes 20 \
  --randomize-task \
  --seed 42 \
  --can-x-range -0.05 0.05 \
  --can-y-range -0.05 0.05 \
  --drawer-open-range 0.0 0.05 \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --policy-device cuda \
  --output-dir outputs/eval/rand20_360k
```

结束日志会打印 `output_dir=...`、`can_xy_enabled=...`、`distractor_cans=...`
和 `success=K/N`。可用 `--no-save-videos` / `--no-save-diagnostics` 只写
`summary.json`。

成功条件见 scripted YAML 的 `success`：`drawer_open_abs_max` 与 `can_world_z`。

## 控制与诊断

```bash
bash run.sh diagnose outputs/eval/<run_dir>/ep001_actions.csv
# 或单轮
bash run.sh diagnose outputs/eval/<run_dir>/rollout_actions.csv
```

当前固定场景回归基线（仅作接口参考）：
`complete=True success=True drawer≈0.003m can_z≈1.023m sim≈23.6s`。
