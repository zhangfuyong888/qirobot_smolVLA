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
bash real_vla_stack/run.sh analyze-dynamics
bash real_vla_stack/run.sh train --profile smoke
bash real_vla_stack/run.sh checkpoint-check --checkpoint /absolute/path/to/checkpoint
bash real_vla_stack/run.sh behavior-probe --checkpoint /absolute/path/to/checkpoint --samples 10
bash real_vla_stack/run.sh serve --checkpoint /absolute/path/to/checkpoint

# On the robot (shadow is the default; run.sh sources the ROS environment):
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

Production training is initialized from the full local `lerobot/smolvla_base`
snapshot configured by `model.pretrained_policy`. Startup fails closed unless the
snapshot contains its policy config, model weights, processors, normalization
state, Action Expert, and state/action projections. Pretrained mode does not
override architecture fields such as `max_state_dim`, `max_action_dim`, or VLM
layout. LeRobot rebuilds normalization from the drawer dataset. The `smoke`,
`overfit`, `baseline`, and `full` profiles are 300, 5k, 200k, and 400k steps;
long runs save every 50k steps.

The project-specific pretrained preflight lives in
`host/training/preflight.py`, not in the pinned `lerobot` submodule. Before an
actual training run it validates the snapshot metadata and performs a CPU-only
`SmolVLAPolicy.from_pretrained(..., strict=True)` load through LeRobot's public
API. This keeps the submodule clean while failing closed on missing or
unexpected model tensors.

## RTC rollout

完整的真机 rollout 架构、时间轴、RTC/安全参数和日志判读见
[docs/real_robot_rollout.md](docs/real_robot_rollout.md)。该文档是部署与调参的
主参考；本节保留 RTC 的关键约束。

RTC support follows the pinned LeRobot implementation. The deployment profile
enables it with `server.rtc.enabled: true`, a 10-step execution horizon, and
maximum guidance weight 10.

- `prev_chunk_left_over` is the unprocessed, normalized model-space output from
  `predict_action_chunk`, shaped `[T, 8]`; LeRobot pads it internally to the
  policy's `max_action_dim`.
- The server stores this raw chunk separately from the postprocessed physical
  joint targets sent to the robot. Robot-side radians and binary gripper values
  are never fed back as RTC guidance.
- The robot acknowledges the last chunk it actually accepted. A rejected chunk
  is never promoted to the server's RTC prefix state.
- If limiter lag or closed-gripper contact lag persists, the robot holds the
  last safe target and asks the server to clear stale RTC history. Motion only
  resumes after a fresh measured-state chunk passes the stricter resync gate.
- Prefix position is derived from monotonic observation timestamps. Delay is
  estimated from the rolling P95 of the last 30 end-to-end observation ages and
  converted with `ceil(age * dataset_fps)`, matching LeRobot's latency handling.
- LeRobot fully guides the delay prefix, tapers guidance through the execution
  horizon, and leaves the remainder of the new chunk unguided. The robot executes
  the postprocessed chunk on its original observation-time axis.

Robot-side chunk blend defaults to zero in the RTC-ready configuration. Hard
jump/tracking checks, hardware limits, and data-derived per-joint dynamics guards
remain active. Compare rollout logs with:

```bash
python scripts/compare_rollout_logs.py /path/to/rollout_A /path/to/rollout_B
```

## Runtime environments and command route

The host process uses `environment/smolvla.yml`. The robot process uses
`hardware_teleop/environment.yml` plus ROS2 Humble and the locally built `qi`
messages. Recreate or update those Conda environments from the corresponding
YAML files; do not rely on packages from the user's Python site directory.

The live rollout does **not** publish directly to the standing-controller topic
`/lowcmd`. It constructs `qi/msg/LowCmd` frames and publishes them to the reviewed
SDK arm-only route `/lowcmd_replay` with `mode_ctrl=4`; non-arm motors are disabled
in that message. The active hand is published separately on `/handscmd`. Direct
`/lowcmd` output is reserved for the separately reviewed leg-deploy route and is
not selected by `real_vla_stack/robot/rollout/main.py`.

Shadow mode still captures observations and calls the LAN policy server, but the
ROS command publishers are not created. Live output requires both
`rollout.mode: live` and the `--live` command-line flag.

For a final no-motion check of the exact live configuration, add
`--preflight-only` together with `--live`. It exits after SDK, feedback, camera,
network, inference and action-contract checks, before Home or policy commands.

## Runtime safety behavior

- One persistent network worker owns the ZeroMQ socket; the 30 Hz control loop
  never shares that socket across threads.
- Action time starts at observation capture, not response arrival. Delayed action
  steps are skipped, and a response older than `max_response_age_ms` is rejected.
  At `max_motion_chunk_age_ms` the controller freezes the last absolute target;
  it waits only until `max_chunk_age_ms` for recovery before aborting.
- Robot feedback age and command-publisher conflicts are checked continuously.
  Stale feedback or a new conflicting publisher causes immediate output
  relinquishment; the robot does not attempt an open-loop return-home move.
- The default fault policy does not return home: network, camera, policy, or
  action faults briefly hold only while feedback is fresh, then relinquish.
  Return-home remains enabled after normal episode completion.
- The measured arm must stay within `max_command_tracking_error_rad` of the last
  step-limited command. A stalled or badly lagging actuator relinquishes output.
- Live commands are paced from the previous actual publication, so a delayed
  camera or network iteration skips a control tick instead of sending catch-up
  bursts. A rollout-specific velocity/acceleration filter sits ahead of the
  hardware bridge's final per-message joint-step limiter.
- Policy requests use `replan_interval_steps`, independently of the 20 Hz action
  timeline. The current `execute_horizon=35` / `replan_interval_steps=10` profile
  retains 35 policy steps and requests a replacement after ten. A replacement
  chunk is delay-aligned and checked against current state; robot-side blend is
  applied only when `chunk_blend_duration_ms` is non-zero (the RTC profile sets it
  to zero).
- A single unsafe stochastic chunk is logged and discarded. The last fresh plan
  remains active; consecutive rejection beyond the configured limit, or expiry
  of that plan, still aborts and relinquishes command output.
- Before any live motion, a separate preflight session must complete one real
  camera/network/GPU inference and pass the non-execution action checks. The
  deployment session then starts at request zero and resets policy state.
- Both cameras receive the configured startup warm-up interval before the first
  policy observation so auto exposure and white balance have settled.
- When `start_from_home` is enabled, the pre-home policy chunk is never executed,
  so preflight checks its timing, shape, finite values, adjacent steps and
  gripper range but defers only the first-target/observation-state comparison.
  After deterministic homing, chunk step zero is checked against the state sent
  with its observation, while the delay-aligned execution target is separately
  checked against current measured state before any policy action is published.
- CUDA is fail-closed: a server configured with `device: cuda` does not silently
  fall back to CPU.
- Live startup verifies the running SDK executable against the configured
  approved SHA256 list and refuses to coexist with the leg-deploy
  `/qi_topic_converter` route on `/lowcmd`.
- Control events are written by a background logger. Shadow saves both camera
  JPEGs and state at the configured interval and records each candidate execution
  chunk in `events.jsonl`.

## Commissioning sequence

1. Build messages and validate the robot runtime:

   ```bash
   bash run.sh teleop-hardware-build
   bash run.sh teleop-hardware-system-prepare --check
   source hardware_teleop/scripts/source_ros_env.sh
   ros2 topic hz lowstate
   ros2 topic info /lowcmd_replay --verbose
   ```

2. On the inference host, validate and serve one explicit checkpoint:

   ```bash
   bash real_vla_stack/run.sh checkpoint-check --checkpoint /absolute/path/to/checkpoint
   bash real_vla_stack/run.sh serve --checkpoint /absolute/path/to/checkpoint
   ```

3. Run without `--live` for an RTC shadow pass (no motion is published):

   ```bash
   bash real_vla_stack/run.sh rollout --max-runtime-s 30
   ```

   Inspect `~/real_rollouts/rollout_*/events.jsonl` and the saved
   `observations/*.jpg`. There must be no `abort` or `policy_error`, camera order
   and color must be correct, and candidate joint/gripper values must be plausible.
   The host and robot must use the same checkout because RTC diagnostics use
   protocol version 4.

4. Clear the workspace, keep a person on the hardware emergency stop, set
   `rollout.mode: live`, and start with a five-second trial:

   ```bash
   bash real_vla_stack/run.sh rollout --live --max-runtime-s 5
   ```

   Increase to 10 seconds and then the configured episode duration only after the
   short run starts, executes, returns home, and relinquishes cleanly.
