#!/usr/bin/env python3
"""Summarize rollout raw/fused action jumps and command tracking."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


GROUPS = {"LA": slice(0, 7), "LH": slice(7, 13), "RA": slice(13, 20), "RH": slice(20, 26)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a rollout diagnostics CSV.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("Diagnostics CSV needs at least two rows")
    joint_names = [key.removeprefix("raw.") for key in rows[0] if key.startswith("raw.")]
    vector_prefixes = ["raw", "ensemble", "command", "actual"]
    if any(key.startswith("masked.") for key in rows[0]):
        vector_prefixes.insert(2, "masked")
    arrays = {
        prefix: np.asarray([[float(row[f"{prefix}.{joint}"]) for joint in joint_names] for row in rows])
        for prefix in vector_prefixes
    }
    for name, group in GROUPS.items():
        raw_jump = np.abs(np.diff(arrays["raw"][:, group], axis=0)).reshape(-1)
        fused_jump = np.abs(np.diff(arrays["ensemble"][:, group], axis=0)).reshape(-1)
        tracking = np.abs(arrays["command"][:, group] - arrays["actual"][:, group]).reshape(-1)
        masking = (
            np.abs(arrays["ensemble"][:, group] - arrays["masked"][:, group]).reshape(-1)
            if "masked" in arrays
            else np.zeros(1, dtype=np.float64)
        )
        print(
            f"{name}: raw_jump mean/p95={raw_jump.mean():.4f}/{np.quantile(raw_jump, .95):.4f} "
            f"fused={fused_jump.mean():.4f}/{np.quantile(fused_jump, .95):.4f} "
            f"mask={masking.mean():.4f}/{np.quantile(masking, .95):.4f} "
            f"tracking={tracking.mean():.4f}/{np.quantile(tracking, .95):.4f} rad"
        )


if __name__ == "__main__":
    main()
