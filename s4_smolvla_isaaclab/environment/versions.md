# Validated Environment Snapshot

Collected on 2026-08-07. This is the known working workstation snapshot, not
a claim that every patch release is interchangeable.

| Component | Version / revision |
|---|---|
| Ubuntu | 22.04.5 LTS |
| NVIDIA driver | 580.159.03 (last successful Isaac Sim log) |
| CUDA used by PyTorch | 12.8 |
| Isaac Sim | 5.1.0.0 |
| IsaacLab | 0.54.2, Git `37ddf626871758333d6ed89cf64ad702aef127d0` (checkout dirty) |
| `env_isaaclab` Python | 3.11.15 |
| `env_isaaclab` PyTorch | 2.7.0+cu128 |
| NumPy / h5py | 1.26.x / 3.16.0 |
| Pinocchio / Pink package | `pin` 2.7.0 / `pin-pink` 3.1.0 installed; teleoperation uses the vendored Pink source |
| LeRobot | 0.6.1, Git `3f2179f3b69708b6ad009b2e7685dd9d05269ee1` |
| `smolvla` Python | 3.12.13 |
| `smolvla` PyTorch | 2.7.0+cu128 |
| Transformers | 5.5.4 |
| PyAV / PyArrow / pandas | 15.1.0 / 25.0.0 / 3.0.5 |
| Project | Git `8a9745917aca78fcdf7ceee5c0badfdc717c8e1c` before normalization |

System `ffmpeg` was not on `PATH` during the audit. LeRobot conversion used
PyAV and the SVT-AV1 codec available to that Python environment. The provided
SmolVLA environment adds `ffmpeg` for explicit command-line inspection.

## Docker full-v4-r1 deployment validation

Validated on 2026-09-01 on Ubuntu 22.04 with NVIDIA driver 570.190 and an
8×RTX 4090 server. The release containers reported Python 3.11.16 in
`env_isaaclab`, Python 3.12.14 in `smolvla`, PyTorch 2.7.0+cu128 and CUDA
userspace 12.8. The `smolvla` environment includes Accelerate 1.14.0 as a
required dependency for both single- and multi-GPU training. The following runtime capabilities passed: NVIDIA EGL Vulkan,
Isaac Sim 5.1 headless renderer, a real `(1, 128, 128, 3)` RGB camera frame,
SmolVLA rollout, single-GPU resume, and two-rank Accelerate/NCCL training with
real forward/backward steps.

This Docker record is intentionally separate from the workstation snapshot
above. Host NVIDIA driver libraries are injected by NVIDIA Container Toolkit
and are not baked into the image.

## Real-robot LAN rollout dependencies

Validated on 2026-09-07 with user-site packages disabled:

| Process | Environment | Rollout-specific packages |
|---|---|---|
| Policy server | `smolvla`, Python 3.12 | OpenCV headless 4.13.0.92, msgpack 1.1.2, pyzmq 27.0.2 |
| Robot client | `s4_hardware_teleop`, Python 3.10 | NumPy 2.2.6, OpenCV headless 4.11.0.86, msgpack 1.1.2, pyzmq 27.0.2 |
| ROS/Python compatibility | `s4_hardware_teleop` | lark 1.1.1, typeguard 2.2.2, pytest 6.2.5 |

Both environments passed `pip check`. The final rollout-focused suites passed
41 tests in the robot environment and 22 tests in the host environment (the
host command disables unrelated system ROS pytest plugin auto-loading). The
project-local system-Python runtime also passed with ROS Pinocchio 3.9.0,
OpenCV headless 4.11.0.86, msgpack 1.1.2 and pyzmq 27.0.2. ROS2 Humble and the
locally built `qi` messages remain system/workspace dependencies and must be sourced with
`hardware_teleop/scripts/source_ros_env.sh` before starting a robot rollout.
