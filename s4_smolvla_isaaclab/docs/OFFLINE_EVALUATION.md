# 离线评估

```bash
bash run.sh preview \
  --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model \
  --num-frames 20 --device cuda
```

该命令从 LeRobotDataset 取真实 state、三路图像和 task text，经与训练一致的
preprocessor/policy/postprocessor 得到 action，与 expert action 计算 MAE/RMSE。
它检查 feature、normalization 和模型加载，但属于 teacher-forced 单步评估；低
MAE 不保证闭环 rollout 成功。输出 CSV 位于 `${S4_OUTPUT_ROOT}/eval/`。
