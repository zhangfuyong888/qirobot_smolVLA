# S4 real VLA stack

`real_vla/` remains the Python 3.10 ROS2 collection layer. This directory owns the
contract-driven path from saved raw episodes to LeRobot, SmolVLA training, checked
checkpoints, LAN inference, and shadow/live robot rollout.

The hard boundary is intentional:

- `common/` imports neither Torch/LeRobot nor ROS.
- `host/` runs in the Python 3.12 `smolvla` environment and never imports ROS.
- `robot/` runs in the Python 3.10 hardware environment and never imports Torch/LeRobot.
- Only the robot process can publish hardware commands. Live mode requires both
  `rollout.mode: live` in robot YAML and the CLI `--live` flag.

## Workflow

```bash
bash real_vla_stack/run.sh raw-check
bash real_vla_stack/run.sh convert
bash real_vla_stack/run.sh dataset-check
bash real_vla_stack/run.sh train --profile smoke
bash real_vla_stack/run.sh checkpoint-check
bash real_vla_stack/run.sh serve

# On the robot (shadow is the default):
bash real_vla_stack/run.sh rollout
```

After shadow logs confirm joint order, RGB images, normalization, gripper semantics,
network latency and action scale, set `rollout.mode: live` in the robot-specific YAML
and explicitly run `bash real_vla_stack/run.sh rollout --live`.

The dataset contract is written to `meta/s4_contract.json`; its SHA256 is copied into
training provenance and the deployment manifest. The LAN server and robot reject a
mismatched hash before actions enter the buffer. Images use causal latest-before
alignment, RGB uint8 HWC in the dataset, and JPEG over a ZeroMQ multipart LAN protocol.
Arm targets are interpolated from 20 Hz policy time to 30 Hz control time; the logical
gripper remains stepwise. A stale chunk is never repeated indefinitely.
