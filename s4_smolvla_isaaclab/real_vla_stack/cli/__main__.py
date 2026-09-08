from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from real_vla_stack.common.config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract-driven S4 real VLA pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("raw-check")
    convert = sub.add_parser("convert")
    convert.add_argument("--overwrite", action="store_true")
    sub.add_parser("dataset-check")
    sub.add_parser("analyze-dynamics")
    train = sub.add_parser("train")
    train.add_argument("--profile", choices=["smoke", "overfit", "baseline", "full"])
    train.add_argument("--dry-run", action="store_true")
    checkpoint = sub.add_parser("checkpoint-check")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    probe = sub.add_parser("behavior-probe")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--samples", type=int, default=10)
    probe.add_argument("--max-episodes", type=int)
    serve = sub.add_parser("serve")
    serve.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_pipeline_config(args.config)
    if args.command == "raw-check":
        from real_vla_stack.host.dataset.raw_validator import validate_raw_dataset

        _, _, report = validate_raw_dataset(cfg.host_path_value("raw_root"), cfg.contract)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "convert":
        from real_vla_stack.host.dataset.exporter import convert_raw_to_lerobot

        print(convert_raw_to_lerobot(cfg, overwrite=args.overwrite))
        return 0
    if args.command == "dataset-check":
        from real_vla_stack.host.dataset.lerobot_validator import validate_lerobot_dataset

        root = cfg.host_path_value("lerobot_root") / str(cfg.host["dataset"]["repo_id"])
        print(
            json.dumps(
                validate_lerobot_dataset(
                    root, cfg.contract, raw_root=cfg.host_path_value("raw_root")
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "analyze-dynamics":
        from real_vla_stack.host.dataset.analyze_action_dynamics import analyze_action_dynamics

        root = cfg.host_path_value("lerobot_root") / str(cfg.host["dataset"]["repo_id"])
        print(json.dumps(analyze_action_dynamics(root, fps=cfg.contract.dataset_fps), indent=2))
        return 0
    if args.command == "train":
        from real_vla_stack.host.training.launcher import launch_training

        command = launch_training(cfg, profile=args.profile, dry_run=args.dry_run)
        if args.dry_run:
            print(shlex.join(command))
        return 0
    if args.command == "behavior-probe":
        from real_vla_stack.host.training.behavior_probe import probe_checkpoint_behavior

        print(
            json.dumps(
                probe_checkpoint_behavior(
                    cfg,
                    args.checkpoint,
                    samples_per_observation=args.samples,
                    max_episodes=args.max_episodes,
                ),
                indent=2,
            )
        )
        return 0
    if args.command in {"checkpoint-check", "serve"}:
        from real_vla_stack.host.training.checkpoint import check_checkpoint, resolve_deployment_checkpoint

        model = args.checkpoint or resolve_deployment_checkpoint(cfg)
        manifest = check_checkpoint(cfg, model, run_inference=args.command == "checkpoint-check")
        if args.command == "checkpoint-check":
            print(json.dumps(manifest, indent=2))
            return 0
        from real_vla_stack.host.inference.policy_runner import PolicyRunner
        from real_vla_stack.host.inference.server import serve_policy

        server = cfg.host["server"]
        rtc = server.get("rtc", {})
        runner = PolicyRunner(
            model,
            cfg.contract,
            device=str(server["device"]),
            rtc_enabled=bool(rtc.get("enabled", False)),
            rtc_execution_horizon=int(rtc.get("execution_horizon", 10)),
            rtc_max_guidance_weight=float(rtc.get("max_guidance_weight", 10.0)),
        )
        serve_policy(runner, bind=str(server["bind"]), port=int(server["port"]))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
