from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class RolloutLogger:
    def __init__(self, root: Path, run: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        self.events = (self.root / "events.jsonl").open("a", encoding="utf-8")

    def event(self, kind: str, **payload: Any) -> None:
        self.events.write(json.dumps({"type": kind, **payload}, separators=(",", ":")) + "\n")
        self.events.flush()
        os.fsync(self.events.fileno())

    def close(self) -> None:
        self.events.close()
