#!/usr/bin/env python
"""Re-run quality checks on a saved episode directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_vla.config_loader import load_collection_config  # noqa: E402
from real_vla.data.validation import validate_saved_episode  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_collection_config(args.config)
    result, _, _ = validate_saved_episode(args.episode_dir, config)
    print(result.label)
    for note in result.notes:
        print(f"- {note}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
