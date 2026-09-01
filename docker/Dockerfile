#
# Complete, self-contained S4 SmolVLA image.
# It intentionally uses the same installation path as this workstation:
# Isaac Sim 5.1 Python packages in env_isaaclab, rather than the NGC Isaac Sim
# container or a standalone Isaac Sim archive.
# Deliberately use the local tag created from the verified CUDA runtime image.
# This avoids querying a registry mirror during the build.  See docker/README.md.
FROM s4-cuda-base:12.8.1

ARG DEBIAN_FRONTEND=noninteractive

ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:${PATH}
ENV S4_CONDA_ROOT=/opt/conda
ENV S4_ISAACLAB_ENV=env_isaaclab
ENV S4_SMOLVLA_ENV=smolvla
ENV S4_PROJECT_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab
ENV ISAACLAB_ROOT=/workspace/smolVLA/IsaacLab
ENV LEROBOT_ROOT=/workspace/smolVLA/lerobot
ENV S4_SCENE_ASSET_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab/local_assets/isaac/5.1
# Keep the legacy Isaac asset variable on the same bundled tree. Without this,
# run.sh would derive a nonexistent /root/isaacsim_assets path in the container.
ENV ISAAC_ASSET_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab/local_assets/isaac/5.1
ENV SMOLVLA_MODEL_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab/models
ENV S4_DATA_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab/datasets
ENV S4_OUTPUT_ROOT=/workspace/smolVLA/s4_smolvla_isaaclab/outputs
ENV S4_CACHE_ROOT=/workspace/runtime/cache
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxtst6 \
        pkg-config \
        rsync \
        tini \
        unzip \
        zstd \
    && rm -rf /var/lib/apt/lists/*

COPY docker/vendor/Miniforge3-25.3.1-0-Linux-x86_64.sh /tmp/miniforge.sh
RUN echo '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768  /tmp/miniforge.sh' | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda config --system --set channel_priority strict \
    && conda clean -a -y

# Build environments before copying the large datasets/models/output tree so that
# model or data changes do not invalidate the expensive dependency layers.
COPY s4_smolvla_isaaclab/environment/isaaclab.yml /tmp/environment/isaaclab.yml
COPY docker/pip-build-constraints.txt /tmp/isaaclab-pip-build-constraints.txt

RUN conda env create -f /tmp/environment/isaaclab.yml

# Keep the large Isaac Sim installation in its own layer.  If a later IsaacLab
# source dependency fails, Docker can reuse this completed download layer.
RUN conda run -n env_isaaclab python -m pip install --upgrade pip \
    && conda run -n env_isaaclab python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com \
    && conda clean -a -y

RUN conda run -n env_isaaclab python -m pip install "setuptools<81" "torchaudio==2.7.0" \
    && conda run -n env_isaaclab bash -lc 'export PIP_BUILD_CONSTRAINT=/tmp/isaaclab-pip-build-constraints.txt; python -m pip install "flatdict==4.0.1"' \
    && conda clean -a -y

# Keep SmolVLA's environment specification after all Isaac environment layers.
# A policy-only dependency change must not force Docker to reinstall Isaac Sim.
COPY s4_smolvla_isaaclab/environment/smolvla.yml /tmp/environment/smolvla.yml
RUN conda env create -f /tmp/environment/smolvla.yml \
    && conda run -n smolvla python -m pip install --upgrade pip \
    && conda run -n smolvla python -m pip uninstall -y torch torchvision torchaudio \
    && conda run -n smolvla python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0' \
    && conda clean -a -y

WORKDIR /workspace/smolVLA

# IsaacLab is a workspace-local release snapshot.  docker/prepare_workspace.sh
# copies it here from the verified external checkout without its .git history.
COPY IsaacLab /workspace/smolVLA/IsaacLab
# BuildKit's executor is not always detected by IsaacLab's is_docker() helper.
# Mark this one build layer as a container so IsaacLab skips VS Code setup,
# whose interactive Omniverse EULA prompt cannot be answered during docker build.
RUN touch /run/.containerenv \
    && conda run -n env_isaaclab bash -lc 'export PIP_BUILD_CONSTRAINT=/tmp/isaaclab-pip-build-constraints.txt; cd /workspace/smolVLA/IsaacLab && ./isaaclab.sh --install none' \
    && rm -f /run/.containerenv

# IsaacLab 0.54.2 requests URDF importer 2.4.31 while the Isaac Sim 5.1 pip
# bundle contains 2.4.30.  Vendor the two exact first-run extensions so rollout
# never needs the online Kit registry.
COPY docker/vendor/kit-exts/ /opt/conda/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/

COPY lerobot /workspace/smolVLA/lerobot
RUN conda run -n smolvla python -m pip install -e /workspace/smolVLA/lerobot

# This final copy intentionally includes the full project resource tree:
# local_assets/, models/, datasets/, and all current outputs/checkpoints.  The
# workstation .env and generated ROS build trees are excluded by .dockerignore.
COPY s4_smolvla_isaaclab /workspace/smolVLA/s4_smolvla_isaaclab

# LeRobot checkpoints store resolved training-workstation paths in JSON
# metadata. Rewrite only the image copy to the container roots, then fail the
# build if any /home/<user> or /Users/<user> artifact path remains.
COPY docker/sanitize_release_paths.py /usr/local/bin/s4-sanitize-release-paths
RUN chmod 0755 /usr/local/bin/s4-sanitize-release-paths \
    && /usr/local/bin/s4-sanitize-release-paths \
        --project-root "$S4_PROJECT_ROOT" \
        --lerobot-root "$LEROBOT_ROOT" \
        --isaaclab-root "$ISAACLAB_ROOT" \
        --data-root "$S4_DATA_ROOT" \
        --model-root "$SMOLVLA_MODEL_ROOT" \
        --output-root "$S4_OUTPUT_ROOT" \
    && /usr/local/bin/s4-sanitize-release-paths \
        --project-root "$S4_PROJECT_ROOT" \
        --lerobot-root "$LEROBOT_ROOT" \
        --isaaclab-root "$ISAACLAB_ROOT" \
        --data-root "$S4_DATA_ROOT" \
        --model-root "$SMOLVLA_MODEL_ROOT" \
        --output-root "$S4_OUTPUT_ROOT" \
        --check

# MaterialX in the Isaac Sim pip distribution links against libXt.  Install it
# in a late layer so this runtime-only correction does not invalidate the
# expensive Conda, Isaac Sim, IsaacLab, and LeRobot build cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxt6 \
    && rm -rf /var/lib/apt/lists/*

# Headless Isaac Sim needs the Vulkan loader and an EGL-backed NVIDIA ICD.
# libEGL_nvidia.so.0 is mounted at runtime by NVIDIA Container Toolkit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libvulkan1 \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY docker/vulkan/nvidia_icd_headless.json \
    /etc/vulkan/icd.d/nvidia_icd_headless.json
RUN test -r /etc/vulkan/icd.d/nvidia_icd_headless.json \
    && ldconfig -p | grep -q 'libvulkan\.so\.1'

ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# The release image is immediately usable for the validated 350000-step policy.
# This is a Docker-only default: an explicit --checkpoint still wins, and
# workstation behavior remains unchanged when this variable is unset.
ENV S4_ROLLOUT_CHECKPOINT=/workspace/smolVLA/s4_smolvla_isaaclab/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model
# Isaac Sim 5.1 installed from pip reads this variable before bootstrapping
# Kit. Runtime users remain responsible for accepting the NVIDIA license.
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV S4_KIT_OFFLINE=1
ENV S4_ISAAC_ENV_HELPER=/usr/local/lib/s4/isaac_env.sh

COPY docker/isaac_env.sh /usr/local/lib/s4/isaac_env.sh
COPY docker/entrypoint.sh /usr/local/bin/s4-entrypoint
COPY docker/verify_runtime.sh /usr/local/bin/s4-verify-runtime
COPY docker/release-manifest.env /workspace/release-manifest/source-versions.env
RUN chmod 0755 /usr/local/lib/s4/isaac_env.sh /usr/local/bin/s4-entrypoint /usr/local/bin/s4-verify-runtime \
    && mkdir -p /workspace/runtime/cache /workspace/release-manifest \
    && cp /workspace/release-manifest/source-versions.env /workspace/release-manifest/versions.env \
    && printf '%s\n' "isaacsim=5.1.0.0" >> /workspace/release-manifest/versions.env \
    && conda run -n env_isaaclab python -c "import importlib.metadata as m; assert m.version('isaacsim') == '5.1.0.0'; import isaaclab" \
    && conda run -n smolvla python -c "import importlib.metadata as m; import torch, lerobot; assert torch.version.cuda == '12.8', torch.version.cuda; assert m.version('accelerate') == '1.14.0'" \
    && cd /workspace/smolVLA/s4_smolvla_isaaclab \
    && bash run.sh doctor --strict

WORKDIR /workspace/smolVLA/s4_smolvla_isaaclab
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/s4-entrypoint"]
CMD ["bash"]
