# 数据集验证

```bash
bash run.sh dataset-check
# 当前 10 阶段数据集与由它训练的 checkpoint 必须具有相同 sidecar 契约
bash run.sh dataset-check --checkpoint outputs/train/smolvla_drawer_insert_close_v1_10phase/checkpoints/<step>/pretrained_model
# 也可单独检查尚未转换的旧原始 HDF5；检查器会验证其文本能否映射到当前 10 阶段
bash run.sh dataset-check datasets/staging/s4_drawer_insert_close_v0 --hdf5
```

验证内容：26D shape、三相机 key/shape、20 Hz、Parquet frame/task 字段、每个
episode 的 frame/timestamp 单调性、NaN/Inf、视频存在与首帧解码，以及 checkpoint
input/output features。成功时输出两行 `[OK]`；任一契约不一致以非零状态退出。

视觉抽查可在 `smolvla` 环境安装 `rerun-sdk` 后使用 LeRobot 自带
`lerobot-dataset-viz`。自动验证不能替代对随机 episode 的视角、动作时序和任务
成功结果抽查。
