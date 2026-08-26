#!/usr/bin/env python
"""JSON-lines SmolVLA policy server for IsaacLab rollout.

The IsaacLab environment uses Python 3.11 and should stay isolated. This
server runs under the `smolvla` environment, loads the policy once, then reads
observations from stdin and writes actions to stdout.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = Path(os.environ.get("S4_CACHE_ROOT", PROJECT_ROOT / ".cache")) / "huggingface"


def _set_local_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "transformers"))


def _resolve_checkpoint(path: str) -> Path:
    ckpt = Path(path).expanduser()
    if (ckpt / "pretrained_model").exists():
        ckpt = ckpt / "pretrained_model"
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"SmolVLA pretrained_model config.json not found: {ckpt}")
    return ckpt


def _image_array_from_payload(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(x) for x in payload["shape"])
    raw = base64.b64decode(payload["b64"])
    image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.copy()


def _annotate_phase_schedule(
    schedule: list[dict[str, Any]],
    dataset_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach stable language IDs when the converted dataset declares them."""
    raw_phases = dataset_contract.get("language_phases")
    if raw_phases is None:
        return schedule
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("Dataset s4_contract.json has an invalid language_phases list")
    by_prompt = {str(item.get("task", "")): dict(item) for item in raw_phases}
    if (
        not all(by_prompt)
        or not all(str(item.get("id", "")) for item in by_prompt.values())
        or len(by_prompt) != len(raw_phases)
    ):
        raise ValueError("Dataset language phase prompts and IDs must be non-empty and unique")
    annotated: list[dict[str, Any]] = []
    for item in schedule:
        prompt = str(item.get("task", ""))
        if prompt not in by_prompt:
            raise ValueError(f"Dataset task {prompt!r} is not present in language contract")
        contract_phase = by_prompt[prompt]
        active_groups = contract_phase.get("active_action_groups")
        if not isinstance(active_groups, list) or not active_groups:
            raise ValueError(f"Dataset language phase {prompt!r} has no active_action_groups")
        annotated.append(
            {
                **item,
                "language_phase_id": str(contract_phase["id"]),
                "active_action_groups": [str(group) for group in active_groups],
                "rollout_timeout": str(contract_phase.get("rollout_timeout", "fail")),
                "rollout_extension": str(contract_phase.get("rollout_extension", "default")),
            }
        )
    expected_ids = [str(item["id"]) for item in raw_phases]
    actual_ids = [str(item["language_phase_id"]) for item in annotated]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Dataset phase schedule does not match declared language order: "
            f"actual={actual_ids}, expected={expected_ids}"
        )
    return annotated


def _load_phase_schedule(dataset_root: Path) -> list[dict[str, Any]]:
    """Recover the common phase order and median duration from the dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not tasks_path.is_file() or not data_paths:
        raise FileNotFoundError(f"Incomplete LeRobot dataset at {dataset_root}")

    tasks_table = pq.read_table(tasks_path, columns=["task_index", "task"])
    task_by_index = {
        int(index): str(task)
        for index, task in zip(
            tasks_table.column("task_index").to_pylist(),
            tasks_table.column("task").to_pylist(),
            strict=True,
        )
    }
    frame_table = pa.concat_tables(
        [pq.read_table(path, columns=["episode_index", "frame_index", "task_index"]) for path in data_paths]
    )
    rows = sorted(
        zip(
            frame_table.column("episode_index").to_pylist(),
            frame_table.column("frame_index").to_pylist(),
            frame_table.column("task_index").to_pylist(),
            strict=True,
        ),
        key=lambda row: (int(row[0]), int(row[1])),
    )

    runs_by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    current_episode: int | None = None
    current_task: int | None = None
    current_length = 0
    for episode_raw, _, task_raw in rows:
        episode = int(episode_raw)
        task = int(task_raw)
        if episode != current_episode or task != current_task:
            if current_episode is not None and current_task is not None:
                runs_by_episode[current_episode].append((current_task, current_length))
            current_episode = episode
            current_task = task
            current_length = 1
        else:
            current_length += 1
    if current_episode is not None and current_task is not None:
        runs_by_episode[current_episode].append((current_task, current_length))
    if not runs_by_episode:
        raise RuntimeError(f"No phase rows found in {dataset_root}")

    orders = Counter(tuple(task for task, _ in runs) for runs in runs_by_episode.values())
    common_order, matching_episodes = orders.most_common(1)[0]
    durations: dict[tuple[int, int], list[int]] = defaultdict(list)
    for runs in runs_by_episode.values():
        if tuple(task for task, _ in runs) != common_order:
            continue
        for phase_index, (task, length) in enumerate(runs):
            durations[(phase_index, task)].append(length)

    schedule = []
    for phase_index, task_index in enumerate(common_order):
        frames = max(int(np.median(durations[(phase_index, task_index)])), 1)
        schedule.append(
            {
                "phase_index": phase_index,
                "task_index": task_index,
                "task": task_by_index[task_index],
                "frames": frames,
            }
        )
    contract_path = dataset_root / "meta" / "s4_contract.json"
    dataset_contract = (
        json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
    )
    schedule = _annotate_phase_schedule(schedule, dataset_contract)
    print(
        f"[SERVER] phase schedule episodes={matching_episodes}/{len(runs_by_episode)} "
        f"phases={len(schedule)} frames={sum(item['frames'] for item in schedule)}",
        file=sys.stderr,
        flush=True,
    )
    return schedule


def main() -> None:
    _set_local_hf_cache()
    parser = argparse.ArgumentParser(description="Serve SmolVLA actions over JSON lines.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[SERVER] python={sys.executable}", file=sys.stderr, flush=True)
    print(f"[SERVER] conda_prefix={os.environ.get('CONDA_PREFIX', '')}", file=sys.stderr, flush=True)

    from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[SERVER] CUDA unavailable; using CPU", file=sys.stderr, flush=True)
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = _resolve_checkpoint(args.checkpoint)
    print(f"[SERVER] loading {ckpt}", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        policy = SmolVLAPolicy.from_pretrained(str(ckpt), local_files_only=True)
    print("[SERVER] checkpoint loaded", file=sys.stderr, flush=True)
    print(f"[SERVER] moving policy to {device}", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        policy = policy.to(device)
    print(f"[SERVER] policy on {device}", file=sys.stderr, flush=True)
    policy.eval()
    policy.reset()
    image_keys = [k for k, v in policy.config.input_features.items() if v.type.name == "VISUAL"]
    if not image_keys:
        raise RuntimeError("Policy checkpoint has no visual input feature.")
    expected_image_shapes = {
        key: list(policy.config.input_features[key].shape)
        for key in image_keys
    }
    phase_schedule = _load_phase_schedule(Path(args.dataset_root).expanduser().resolve())
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    print(f"[SERVER] ready image_keys={image_keys}", file=sys.stderr, flush=True)
    print(
        json.dumps(
            {
                "status": "ready",
                "image_keys": image_keys,
                "image_shapes": expected_image_shapes,
                "state_dim": int(policy.config.input_features["observation.state"].shape[0]),
                "action_dim": int(policy.config.output_features["action"].shape[0]),
                "device": str(device),
                "phase_schedule": phase_schedule,
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "reset":
                policy.reset()
                print(json.dumps({"status": "reset"}), flush=True)
                continue
            state = np.asarray(request["state"], dtype=np.float32)
            task = str(request["task"])
            image_payloads = request.get("images", {})
            missing = [key for key in image_keys if key not in image_payloads]
            extra = [key for key in image_payloads if key not in image_keys]
            if missing or extra:
                raise ValueError(f"Visual feature mismatch: missing={missing} extra={extra}")
            observation = {"observation.state": state}
            for key in image_keys:
                image = _image_array_from_payload(image_payloads[key])
                expected_hwc = tuple(expected_image_shapes[key][1:]) + (expected_image_shapes[key][0],)
                if image.shape != expected_hwc:
                    raise ValueError(f"{key} shape={image.shape}, expected={expected_hwc}")
                observation[key] = image
            with torch.inference_mode():
                with contextlib.redirect_stdout(sys.stderr):
                    batch = prepare_observation_for_inference(
                        observation,
                        device,
                        task=task,
                        robot_type="S4-Bimanual",
                    )
                    batch = preprocessor(batch)
                    if request.get("mode") == "chunk":
                        action_chunk = policy.predict_action_chunk(batch)
                        action_chunk = postprocessor(action_chunk)
                        action_chunk = action_chunk.squeeze(0).detach().cpu().numpy().astype(float).tolist()
                        response = {"action_chunk": action_chunk}
                    else:
                        action = policy.select_action(batch)
                        action = postprocessor(action)
                        action = action.squeeze(0).detach().cpu().numpy().astype(float).tolist()
                        response = {"action": action}
            print(json.dumps(response), flush=True)
        except Exception as exc:  # Keep server alive long enough for IsaacLab to report the error.
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
