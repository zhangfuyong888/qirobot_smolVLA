#!/usr/bin/env python3
"""Tiny collective used by the Docker training verification profile."""

from __future__ import annotations

import argparse

import torch
from accelerate import Accelerator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-processes", type=int, required=True)
    args = parser.parse_args()
    accelerator = Accelerator()
    if accelerator.num_processes != args.expected_processes:
        raise RuntimeError(
            f"Accelerate launched {accelerator.num_processes} processes, "
            f"expected {args.expected_processes}"
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError(f"Accelerate rank is not on CUDA: {accelerator.device}")
    rank = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered = accelerator.gather(rank)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        actual = sorted(int(value) for value in gathered.cpu().tolist())
        expected = list(range(args.expected_processes))
        if actual != expected:
            raise RuntimeError(f"Accelerate collective ranks={actual}, expected={expected}")
        print(
            f"[OK] Accelerate DDP launch processes={args.expected_processes} "
            f"ranks={actual}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
