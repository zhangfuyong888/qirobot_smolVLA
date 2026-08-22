# 数据集转换

环境：`smolvla`。

```bash
bash run.sh convert
# 将已有 v0 原始 HDF5 转换成当前 v1_10phase LeRobotDataset
bash run.sh convert --root-path datasets/staging/s4_drawer_insert_close_v0 --overwrite
```

转换器读取 HDF5 `processed_actions`、active state、逐帧 task text 和三路 RGB，
调用外部 LeRobot `LeRobotDataset.create/add_frame/save_episode/finalize`。视频由
PyAV 编码；转换不会重新渲染相机，因此 MP4 视角完全来自 HDF5。

转换器会把旧 HDF5 的 20 段逐帧文本映射为 10 个语言宏阶段；未来 HDF5 则使用
`language_phase_id` 并与文本、专家阶段交叉校验。输出
`meta/s4_contract.json` 记录 `drawer_10phase_v1` 的完整有序定义。

`--overwrite` 只针对目标 LeRobotDataset；它不会删除 HDF5。不同任务或 schema
必须使用不同 repo/dataset ID，避免覆盖训练所依赖的数据统计量。
当前输出 ID 为 `s4_drawer_insert_close_v1_10phase`，不会覆盖旧 `v0` 数据集。
