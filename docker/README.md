# S4 SmolVLA 完整 Docker 发布镜像

此目录构建一个单一、离线可运行的镜像。镜像包含 `env_isaaclab`、`smolvla`、项目源码、IsaacLab 源码快照、场景资产、基础模型、现有数据集，以及构建时 `outputs/` 下的全部 checkpoint、评估日志和视频。发布镜像把 `350000/pretrained_model` 设为默认 Rollout checkpoint；显式传入 `--checkpoint` 时仍以命令行参数为准。

构建上下文会排除宿主机 `.env`，避免用户目录下的绝对路径覆盖容器路径。构建阶段还会把 checkpoint JSON 中由 LeRobot 保存的工作站路径重写为镜像内路径，并在残留路径时直接失败。ROS 的 `build/`、`install/` 和 `log/` 也会被排除，因为这些目录包含不可移植的宿主机软链接；ROS 源码仍然保留。

## 前置条件

- Linux x86_64、Docker Engine、NVIDIA 驱动和 NVIDIA Container Toolkit；
- Docker 内可运行 `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`；
- 至少 180GB 可用磁盘空间，建议 250GB 以上；完整资源、构建缓存和最终导出包会同时占用空间；
- 已确认本工作区资源是要发布的版本。构建不会移动外部 IsaacLab checkout。

## 构建

先准备本地 CUDA 基座。它也同时验证 Docker 能使用 NVIDIA GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker image tag nvidia/cuda:12.8.1-base-ubuntu22.04 s4-cuda-base:12.8.1
```

Dockerfile 有意使用本地的 `s4-cuda-base:12.8.1` 标签，以避免某些网络环境中 Docker 镜像加速器对构建元数据请求返回 401。

构建过程中 IsaacLab 会被标识为容器环境，因此自动跳过会触发交互式 Omniverse EULA 的 VS Code 设置步骤。Isaac Sim 5.1 的 pip 启动器实际读取 `OMNI_KIT_ACCEPT_EULA=Y`；镜像与 Compose 已设置该变量，同时保留 NVIDIA 容器常用的 `ACCEPT_EULA=Y` 与 `PRIVACY_CONSENT=Y`。使用镜像仍表示使用者已同意对应 NVIDIA 许可。

构建时不从 GitHub 下载 Miniforge：请先在主机下载一次到构建上下文中。这样可复用主机的网络/代理路径，避免 Docker BuildKit 内部下载因代理未继承而失败。文件不会提交到 Git。

```bash
cd "${HOME}/smolVLA"
mkdir -p docker/vendor
curl --fail --location --continue-at - --retry 12 --retry-all-errors --retry-delay 5 \
  --connect-timeout 30 \
  --output docker/vendor/Miniforge3-25.3.1-0-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh
echo '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768  docker/vendor/Miniforge3-25.3.1-0-Linux-x86_64.sh' | sha256sum -c -
```

当前工作区已经完成这一步，因此可直接继续后续构建。

IsaacLab 0.54.2 在 Isaac Sim 5.1 上需要 URDF Importer 2.4.31 及其 pip archive。发布镜像把这两个首次运行扩展放入本地 `extscache`，避免 Rollout 再访问在线 Kit registry。首次准备工作区时执行：

```bash
bash docker/prepare_kit_extensions.sh
```

`docker/vendor/kit-exts/` 被 Git 忽略，但会进入 Docker 构建上下文；构建脚本会在开始前验证两个扩展完整存在。

如果主机使用本地代理（例如 `127.0.0.1:7890`），请在同一个终端中保留
`HTTP_PROXY`、`HTTPS_PROXY` 等环境变量再执行构建。`build_full.sh` 会仅在构建期使用
主机网络并传递这些变量，使 Conda、PyPI 和 NVIDIA 包索引可访问；最终镜像不保留代理设置。

先把外部 IsaacLab checkout 复制为工作区快照。该命令不会覆盖已有快照：

```bash
cd "${HOME}/smolVLA"
bash docker/prepare_workspace.sh "${HOME}/IsaacLab"
```

然后构建完整镜像：

```bash
bash docker/build_full.sh s4-smolvla:full-v2
```

构建脚本会先确认场景、基础模型、至少一个 LeRobotDataset contract 和至少一个完整 checkpoint 都存在，然后在 `docker/release-manifest.env` 写入三个 Git commit、dirty 状态和 UTC 时间。该文件会复制进镜像的 `/workspace/release-manifest/versions.env`，但被 Git 忽略。

## 运行与验证

镜像内部已经包含构建时的 `datasets/` 和 `outputs/`。推荐通过 Compose 启动：首次创建命名卷时，Docker 会自动把镜像中已有的数据集和训练输出复制到卷内；之后采集、转换、训练和 Rollout 产生的修改会保存在卷中，即使容器使用 `--rm` 删除也不会丢失。首次初始化约 14GB 数据，可能需要等待且暂时没有进度输出。容器每次启动还会幂等更新卷内 checkpoint 和评估 JSON 的路径元数据，以兼容旧命名卷；不会改动模型权重、数据集或训练状态。

缓存和普通运行时文件写到宿主机 `docker/runtime/`。数据集和训练输出存放在 Docker 命名卷 `docker_s4-datasets` 与 `docker_s4-outputs` 中。

```bash
cd "${HOME}/smolVLA"
docker compose -f docker/compose.yaml run --rm s4-smolvla bash
```

容器内先校验：

```bash
s4-verify-runtime
```

验证会检查两个环境、CUDA、Isaac Sim、IsaacLab、LeRobot 源码、场景资产、基础模型、活动任务契约、内置数据集和 350000 checkpoint，并实际启动一次 Headless Rendering。只有此命令通过，才应开始训练或 Rollout。

然后保持项目原有入口：

```bash
cd /workspace/smolVLA/s4_smolvla_isaaclab
bash run.sh help
bash run.sh train --help
bash run.sh rollout --help
```

直接使用镜像内置的 350000 checkpoint 做 20 轮随机 Rollout：

```bash
bash run.sh rollout --headless --success-rate 20 --policy-device cuda
```

查看卷：

```bash
docker volume ls | grep 'docker_s4-'
docker volume inspect docker_s4-datasets docker_s4-outputs
```

> 不要随意执行 `docker compose down -v`，`-v` 会删除持久化的数据集和训练输出卷。镜像中原始副本仍可通过新建卷恢复，但容器运行后新增的数据会丢失。

Isaac Sim 运行需接受 NVIDIA EULA；镜像和 Compose 已设置 `OMNI_KIT_ACCEPT_EULA=Y`，Compose 也保留 `ACCEPT_EULA=Y` 和 `PRIVACY_CONSENT=Y`。发布镜像前，维护者仍需确认场景资产、基础模型和训练数据的再分发许可。

## 导出与导入

在构建机导出：

```bash
docker save s4-smolvla:full-v2 | zstd -T0 -19 -o s4-smolvla_full-v2.tar.zst
sha256sum s4-smolvla_full-v2.tar.zst > s4-smolvla_full-v2.tar.zst.sha256
```

在服务器验证并导入：

```bash
sha256sum -c s4-smolvla_full-v2.tar.zst.sha256
zstd -dc s4-smolvla_full-v2.tar.zst | docker load
```

训练时选择四张可见 GPU 的示例：

```bash
docker run --rm -it \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=64g \
  --network=host \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$PWD/docker/runtime:/workspace/runtime" \
  -v docker_s4-datasets:/workspace/smolVLA/s4_smolvla_isaaclab/datasets \
  -v docker_s4-outputs:/workspace/smolVLA/s4_smolvla_isaaclab/outputs \
  s4-smolvla:full-v2 bash
```

容器内：

```bash
cd /workspace/smolVLA/s4_smolvla_isaaclab
bash run.sh train --num-gpus 4 --gpu-ids 0,1,2,3 --batch-size 4 --num-workers 6 --master-port 29500
```
