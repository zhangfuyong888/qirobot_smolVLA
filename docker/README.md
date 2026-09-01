# S4 SmolVLA 完整 Docker 发布镜像

此目录构建一个单一、离线可运行的镜像。镜像包含 `env_isaaclab`、`smolvla`、项目源码、IsaacLab 源码快照、场景资产、基础模型、现有数据集，以及构建时 `outputs/` 下的全部 checkpoint、评估日志和视频。发布镜像把 `350000/pretrained_model` 设为默认 Rollout checkpoint；显式传入 `--checkpoint` 时仍以命令行参数为准。

构建上下文会排除宿主机 `.env`，避免用户目录下的绝对路径覆盖容器路径。构建阶段还会把 checkpoint JSON 中由 LeRobot 保存的工作站路径重写为镜像内路径，并在残留路径时直接失败。ROS 的 `build/`、`install/` 和 `log/` 也会被排除，因为这些目录包含不可移植的宿主机软链接；ROS 源码仍然保留。

从 `full-v3` 起，镜像在后部 runtime layer 安装 **Vulkan 加载器**（`libvulkan1`、`vulkan-tools`），并内置 headless **NVIDIA ICD**（`/etc/vulkan/icd.d/nvidia_icd_headless.json`，指向 `libEGL_nvidia.so.0`）。NVIDIA 驱动库（`libEGL_nvidia.so.0` 等）**不会** bake 进镜像，必须由 NVIDIA Container Toolkit 在容器启动时从宿主注入。运行时需 `--gpus all` 与 `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`（Compose 已默认设置）。

## Full-v3 host requirements

- x86_64 Linux（不支持 Jetson / ARM）
- NVIDIA RTX GPU 工作站或服务器
- NVIDIA 专有驱动（镜像与驱动版本无关；当前已在 RTX 4090 + driver 570.190 + CUDA 12.8 上实测）
- Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker GPU 支持：`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi` 应成功
- 启动容器时 `NVIDIA_DRIVER_CAPABILITIES` 必须包含 `graphics,compute,utility`
- Headless Vulkan 使用 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json`

若 `docker info` 的 `Runtimes` 中尚无 `nvidia`，管理员需执行：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtime
```

诊断命令（无需 Compose，直接验证 runtime）：

```bash
bash docker/run.sh --gpus 0 verify
```

`s4-verify-runtime` 四级检查：CUDA → `vulkaninfo`（NVIDIA，禁止 llvmpipe）→ Isaac headless renderer（Kit 日志 fail-fast）→ camera RGB frame。

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
bash docker/build_full.sh s4-smolvla:full-v3
```

构建脚本会先确认场景、基础模型、至少一个 LeRobotDataset contract 和至少一个完整 checkpoint 都存在，然后在 `docker/release-manifest.env` 写入三个 Git commit、dirty 状态和 UTC 时间。该文件会复制进镜像的 `/workspace/release-manifest/versions.env`，但被 Git 忽略。

## 运行与验证

镜像内部已经包含构建时的 `datasets/` 和 `outputs/`。推荐通过 **`docker/run.sh`** 启动，可统一指定宿主机 GPU：

```bash
cd "${HOME}/smolVLA"

# 单卡验证（物理 GPU 0）
bash docker/run.sh --gpus 0 verify

# 交互 shell（物理 GPU 3）
bash docker/run.sh --gpus 3 bash

# Compose + 命名卷（四卡训练前进入容器）
bash docker/run.sh --gpus 0,1,2,3 --compose bash
```

`--gpus` 接受宿主机 `nvidia-smi` 上的物理编号：`0`、`0,1,2,3` 或 `all`（默认）。容器内 GPU 始终从 `cuda:0` 重新编号；例如 `--gpus 4,5,6,7` 时，训练命令仍写 `--gpu-ids 0,1,2,3`。

也可用环境变量：

```bash
S4_GPUS=2 bash docker/run.sh verify
S4_GPUS=0,1,2,3 bash docker/run.sh --compose bash
```

Compose 直接启动（等价于 `docker/run.sh --compose`）：

```bash
S4_GPUS=0 docker compose -f docker/compose.yaml run --rm s4-smolvla s4-verify-runtime
```

镜像内部已经包含构建时的 `datasets/` 和 `outputs/`。Compose 模式下，首次创建命名卷时 Docker 会自动把镜像中已有的数据集和训练输出复制到卷内；之后采集、转换、训练和 Rollout 产生的修改会保存在卷中，即使容器使用 `--rm` 删除也不会丢失。首次初始化约 14GB 数据，可能需要等待且暂时没有进度输出。容器每次启动还会幂等更新卷内 checkpoint 和评估 JSON 的路径元数据，以兼容旧命名卷；不会改动模型权重、数据集或训练状态。

容器内先校验：

```bash
s4-verify-runtime
# 或宿主机直接：
bash docker/run.sh --gpus 0 verify
```

缓存和普通运行时文件写到宿主机 `docker/runtime/`。数据集和训练输出存放在 Docker 命名卷 `docker_s4-datasets` 与 `docker_s4-outputs` 中。

验证会检查两个环境、四级 runtime（CUDA、NVIDIA Vulkan、Isaac headless renderer、camera RGB）、IsaacLab、LeRobot 源码、场景资产、基础模型、活动任务契约、内置数据集和 350000 checkpoint。只有此命令通过，才应开始训练或 Rollout。

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
docker save s4-smolvla:full-v3 | zstd -T0 -19 -o s4-smolvla_full-v3.tar.zst
sha256sum s4-smolvla_full-v3.tar.zst > s4-smolvla_full-v3.tar.zst.sha256
```

在服务器验证并导入：

```bash
sha256sum -c s4-smolvla_full-v3.tar.zst.sha256
zstd -dc s4-smolvla_full-v3.tar.zst | docker load
```

服务器上推荐用 **`docker/run.sh`** 指定 GPU 并验证：

```bash
bash docker/run.sh --gpus 0 verify
```

或手动 `docker run`（等价）：

```bash
-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
-e NVIDIA_VISIBLE_DEVICES=0          # 多卡机器上指定空闲 GPU
-e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
```

导入后先跑：

```bash
bash docker/run.sh --gpus 0 verify
```

手动等价命令：

```bash
docker run --rm --gpus all --ipc=host --shm-size=64g --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /tmp/s4-runtime:/workspace/runtime \
  s4-smolvla:full-v3 s4-verify-runtime
```

预期成功输出应包含：

```text
[OK] CUDA PASS: NVIDIA GeForce RTX 4090 cuda 12.8
driverName = NVIDIA
[OK] NVIDIA Vulkan renderer
[OK] Isaac Sim headless renderer
[OK] Isaac Sim camera RGB frame (1, 128, 128, 3)
[OK] checkpoint tokenizer and processor pipeline
[OK] complete runtime verification passed
```

以下任一出现则验证必须非零退出：`llvmpipe`、`ERROR_INCOMPATIBLE_DRIVER`、`GPU Foundation is not initialized`、`vkCreateInstance failed`。

四卡训练示例（物理 GPU 0–3）：

```bash
bash docker/run.sh --gpus 0,1,2,3 --compose bash
```

容器内：

```bash
bash run.sh train --num-gpus 4 --gpu-ids 0,1,2,3 --batch-size 4 --num-workers 6 --master-port 29500
```

另一组四卡（物理 GPU 4–7）可开第二个容器：

```bash
bash docker/run.sh --gpus 4,5,6,7 --compose bash
# 容器内同样 --gpu-ids 0,1,2,3
```
