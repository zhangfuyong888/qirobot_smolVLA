# 快速开始

## 1. 配置

```bash
cd s4_smolvla_isaaclab
cp .env.example .env
# 编辑 .env 后检查
bash run.sh doctor --strict
```

## 2. 场景与采集（`env_isaaclab`）

`run.sh` 自动选择环境，无需手工 activate：

```bash
bash run.sh activate-task drawer_insert_close
bash run.sh sim
bash run.sh record --episodes 10 --headless
```

成功标志是 recorder 只写入满足 success criteria 的 episode。

## 3. 转换、验证和训练（`smolvla`）

```bash
bash run.sh convert --overwrite
bash run.sh dataset-check
bash run.sh train
```

输出分别位于 `${S4_DATA_ROOT}/lerobot_data/<dataset>` 和
`${S4_OUTPUT_ROOT}/train/<run>`。

## 4. 评估

```bash
bash run.sh preview --num-frames 20 --device cuda
bash run.sh rollout \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --deterministic --policy-device cuda
# 随机化成功率（例如 20 轮，全部写入同一子目录）
bash run.sh rollout \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --success-rate 20 --policy-device cuda --headless
bash run.sh diagnose outputs/eval/<run_dir>/rollout_actions.csv
```

产物在 `outputs/eval/rollout_<timestamp>_<det|randN>_ckpt<step>/`。
