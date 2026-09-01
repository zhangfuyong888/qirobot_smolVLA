# S4 SmolVLA 完整 Docker 发布镜像（full-v4-r1）

此目录构建一个单一、离线可运行的镜像。镜像包含 `env_isaaclab`、`smolvla`、项目源码、IsaacLab 源码快照、场景资产、基础模型、现有数据集，以及构建时 `outputs/` 下的全部 checkpoint、评估日志和视频。发布镜像把 `350000/pretrained_model` 设为默认 Rollout checkpoint；显式传入 `--checkpoint` 时仍以命令行参数为准。

构建上下文会排除宿主机 `.env`，避免用户目录下的绝对路径覆盖容器路径。构建阶段还会把 checkpoint JSON 中由 LeRobot 保存的工作站路径重写为镜像内路径，并在残留路径时直接失败。ROS 的 `build/`、`install/` 和 `log/` 也会被排除，因为这些目录包含不可移植的宿主机软链接；ROS 源码仍然保留。

从 `full-v3` 起，镜像在后部 runtime layer 安装 **Vulkan 加载器**（`libvulkan1`、`vulkan-tools`），并内置 headless **NVIDIA ICD**（`/etc/vulkan/icd.d/nvidia_icd_headless.json`，指向 `libEGL_nvidia.so.0`）。NVIDIA 驱动库（`libEGL_nvidia.so.0` 等）**不会** bake 进镜像，必须由 NVIDIA Container Toolkit 在容器启动时从宿主注入。

`full-v4` 进一步加入：正式固定的 `accelerate==1.14.0`、删除训练输出前的只读运行时 preflight、Rollout/Training/Full 三种验证 profile、统一的 Isaac native 环境初始化，以及宿主机 contract 检查。`full-v4-r1` 修正 Camera smoke test 的长 `python -c` 启动问题，以及 Accelerate 多卡训练必须使用绝对 `lerobot-train` 路径的问题。GPU 选择统一通过 `docker/run.sh --gpus ...`，不要手工组合 `NVIDIA_VISIBLE_DEVICES`。

Isaac Camera smoke test 以独立 Python 文件运行，不再把完整 Kit 程序通过长 `python -c` 传入。该形式已在 RTX 4090 服务器上成功生成 `(1, 128, 128, 3)` RGB frame，并规避内联启动形式中已复现的 Kit `SIGSEGV 139`。

当前 `full-v4-r1` 已在 Ubuntu 22.04、NVIDIA driver 570.190、8×RTX 4090 服务器上完成以下实测：单卡 CUDA/PyTorch、EGL Vulkan、Isaac Sim 5.1 headless、Camera RGB、真实 Rollout、单卡 resume 训练、双卡 Accelerate/NCCL 启动与真实反向传播。该结果证明此服务器上的完整链路，不代表任意驱动/GPU 组合自动兼容；新宿主仍必须执行 preflight 和对应 profile。

## Full-v4-r1 host requirements

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

先在宿主机执行 contract 检查：

```bash
bash docker/host_preflight.sh --gpu 0
```

再执行容器 runtime 验证：

```bash
bash docker/run.sh --image s4-smolvla:full-v4-r1 --gpus 0 verify
```

`s4-verify-runtime` 默认运行 full profile：训练依赖/数据集 + CUDA → `vulkaninfo`（NVIDIA，禁止 llvmpipe）→ Isaac headless renderer（Kit 日志 fail-fast）→ camera RGB frame。

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
bash docker/build_full.sh s4-smolvla:full-v4-r1
```

构建脚本会先确认场景、基础模型、至少一个 LeRobotDataset contract 和至少一个完整 checkpoint 都存在，然后在 `docker/release-manifest.env` 写入三个 Git commit、dirty 状态和 UTC 时间。该文件会复制进镜像的 `/workspace/release-manifest/versions.env`，但被 Git 忽略。

## 运行与验证

镜像内部已经包含构建时的 `datasets/` 和 `outputs/`。推荐通过 **`docker/run.sh`** 启动，可统一指定宿主机 GPU：

```bash
cd "${HOME}/smolVLA"
export S4_IMAGE=s4-smolvla:full-v4-r1

# 单卡验证（物理 GPU 0）
bash docker/run.sh --gpus 0 verify

# 只验证训练环境与完整数据集
bash docker/run.sh --gpus 0 verify-train

# 两卡时还会实际启动 2 个 Accelerate/DDP rank 并执行一次 collective
bash docker/run.sh --gpus 0,1 verify-train

# 只验证 Rollout/Isaac/Vulkan/Camera
bash docker/run.sh --gpus 0 verify-rollout

# 交互 shell（物理 GPU 3）
bash docker/run.sh --gpus 3 bash

# Compose + 命名卷（四卡训练前进入容器）
bash docker/run.sh --gpus 0,1,2,3 --compose bash
```

当前 `docker/run.sh`/Compose 的兼容默认标签仍是 `s4-smolvla:full-v4`；运行 r1 时必须像上面一样设置 `S4_IMAGE`，或每次显式传 `--image s4-smolvla:full-v4-r1`。镜像标签只选择 artifact，不改变容器内代码。

`--gpus` 接受宿主机 `nvidia-smi` 上的物理编号：`0`、`0,1,2,3` 或 `all`（默认）。GPU 可见性在**创建容器时**固定，运行中的容器不能再增加 GPU。容器内 GPU 从 `cuda:0` 重新编号；例如 `--gpus 4,5` 时，容器内是 `0,1`，双卡训练仍写 `--gpu-ids 0,1`。最安全的策略是只向容器暴露本次任务真正使用的物理卡。

`S4_GPUS` 仅保留兼容性并会打印弃用提示；新命令统一使用 `--gpus`。不要直接用 Compose 环境变量模拟 GPU 隔离，因为环境变量不会替代 Docker 的 device request。

若目标机只有已导入镜像、没有宿主项目脚本，也可直接 `docker run`；此时必须自行设置 `--gpus`、`--ipc=host`、共享内存、graphics capability、EGL ICD，并为训练挂载持久化的数据集和输出卷。宿主脚本已经统一封装这些参数，因此正式使用仍优先同步同一 commit 的仓库并调用 `docker/run.sh`。

仅有镜像时的单卡持久化示例：

```bash
docker run --rm -it \
  --gpus '"device=0"' \
  --ipc=host --shm-size=64g --network=host \
  -e S4_DOCKER_SELECTED_GPU_COUNT=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v s4-datasets:/workspace/smolVLA/s4_smolvla_isaaclab/datasets \
  -v s4-outputs:/workspace/smolVLA/s4_smolvla_isaaclab/outputs \
  -v s4-runtime:/workspace/runtime \
  s4-smolvla:full-v4-r1 bash
```

直接启动的 `s4-*` 卷名和 Compose 自动生成的 `docker_s4-*` 卷名不是同一组数据。选定一种方式后不要在不知情的情况下切换卷名。

容器使用 `--network=host`，网络层可访问宿主局域网和互联网；训练入口仍默认设置 Hugging Face offline，Isaac 默认关闭在线扩展 registry，以固定模型和 Kit 依赖。网络可用不等于运行时会自动下载资源。

镜像内部已经包含构建时的 `datasets/` 和 `outputs/`。Compose 模式下，首次创建命名卷时 Docker 会自动把镜像中已有的数据集和训练输出复制到卷内；之后采集、转换、训练和 Rollout 产生的修改会保存在卷中，即使容器使用 `--rm` 删除也不会丢失。首次初始化约 14GB 数据，可能需要等待且暂时没有进度输出。容器每次启动还会幂等更新卷内 checkpoint 和评估 JSON 的路径元数据，以兼容旧命名卷；不会改动模型权重、数据集或训练状态。

容器内先校验：

```bash
s4-verify-runtime
# 或宿主机直接：
bash docker/run.sh --image s4-smolvla:full-v4-r1 --gpus 0 verify
```

缓存和普通运行时文件写到宿主机 `docker/runtime/`。数据集和训练输出存放在 Docker 命名卷 `docker_s4-datasets` 与 `docker_s4-outputs` 中。

完整验证会检查两个环境、Accelerate/LeRobot training dependency、完整数据集和视频解码、CUDA、NVIDIA Vulkan、Isaac headless renderer、camera RGB、基础模型和 350000 checkpoint。Camera 测试由独立的 `scripts/verify_isaac_camera.py` 执行。也可通过 `s4-verify-runtime --profile train|rollout` 分开定位问题。只有对应 profile 通过，才应开始训练或 Rollout。

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
docker save s4-smolvla:full-v4-r1 | zstd -T0 -19 -o s4-smolvla_full-v4-r1.tar.zst
sha256sum s4-smolvla_full-v4-r1.tar.zst > s4-smolvla_full-v4-r1.tar.zst.sha256
```

在服务器验证并导入：

```bash
sha256sum -c s4-smolvla_full-v4-r1.tar.zst.sha256
zstd -dc s4-smolvla_full-v4-r1.tar.zst | docker load
```

服务器上推荐用 **`docker/run.sh`** 指定 GPU 并验证：

```bash
export S4_IMAGE=s4-smolvla:full-v4-r1
bash docker/run.sh --gpus 0 verify
```

导入后先跑：

```bash
bash docker/run.sh --gpus 0 verify
```

不再提供手写 `docker run + NVIDIA_VISIBLE_DEVICES` 作为正式入口；`docker/run.sh` 会同时生成正确的 Docker GPU device request、graphics capability、EGL ICD、IPC 和缓存挂载。

预期成功输出应包含：

```text
[OK] CUDA PASS: NVIDIA GeForce RTX 4090 cuda 12.8
[TRAIN][PREFLIGHT] PASS ... 'accelerate': '1.14.0' ...
[OK] LeRobot training dependencies, launcher and dataset
driverName = NVIDIA
[OK] NVIDIA Vulkan renderer
[OK] Isaac Sim headless renderer
[OK] Isaac Sim camera RGB frame (1, 128, 128, 3)
[OK] checkpoint tokenizer and processor pipeline
[OK] full runtime verification passed
```

以下任一出现则验证必须非零退出：`llvmpipe`、`ERROR_INCOMPATIBLE_DRIVER`、`GPU Foundation is not initialized`、`vkCreateInstance failed`。

四卡训练示例（物理 GPU 0–3）：

```bash
bash docker/run.sh --gpus 0,1,2,3 --compose bash
```

容器内：

```bash
bash run.sh train --resume --steps 500000 --num-gpus 4 --gpu-ids 0,1,2,3 --batch-size 4 --num-workers 6 --master-port 29500
```

上例假设 `checkpoints/last` 的保存 step 小于 500000；`--steps` 是目标总步数，不是额外步数。Fresh training 必须使用配置中的新输出路径，不能通过省略 `--resume` 覆盖镜像/卷内已有 run。

`--batch-size` 是每个 rank 的 batch，全局 batch 为 `batch-size × num-gpus`。Resume 若改变 checkpoint 记录的 world size 或每 rank batch，LeRobot 会继续训练，但不能保证逐 rank 样本顺序完全一致；要求 sample-exact 时应保持原值。共享服务器开始训练前先用 `nvidia-smi` 检查目标物理卡显存，容器内 `cuda:0` 可能对应宿主机任意一张已选择卡。若 OOM 日志显示其他 PID 已占用几十 GiB，应该换空闲物理卡或协调进程，而不是首先调小 batch 或修改 allocator。

不要让两个训练容器同时写同一个 `S4_OUTPUT_ROOT`/命名卷中的同一 run。并发任务还必须使用不同的 `--master-port`。

另一组四卡（物理 GPU 4–7）可开第二个容器：

```bash
bash docker/run.sh --gpus 4,5,6,7 --compose bash
# 容器内同样 --gpu-ids 0,1,2,3
```
