# SmolVLA 训练

环境：`smolvla`。

```bash
bash run.sh train
bash run.sh train --steps 500000 --batch-size 16 --save-freq 20000
bash run.sh train --resume --steps 500000
```

参数来自 active task 的 `.smolvla.yaml`。当前关键值：26D、三相机、
`chunk_size=50`、`n_obs_steps=1`、图像 padding resize 512x512、mean/std state
和 action normalization、冻结 vision encoder、训练 expert 和 state projection。

`--resume` 从 `checkpoints/last` 恢复 optimizer/scheduler/RNG，不等于只加载
pretrained_model。`--overwrite-output` 会删除训练输出，只有明确需要重新训练时
使用。checkpoint 含 model、pre/postprocessor stats 和 train config。

当前训练只接受 `s4_drawer_insert_close_v1_10phase` 和
`drawer_10phase_v1`。启动时会检查数据集 `meta/s4_contract.json`，并复制到训练
目录的 `s4_dataset_contract.json`。由于语言条件已经由 20 段改为 10 段，必须
fresh training；旧 `v0` checkpoint 不得用 `--resume` 接到当前输出目录。
