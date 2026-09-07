from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any


class RolloutLogger:
    def __init__(self, root: Path, run: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        self.events = (self.root / "events.jsonl").open("a", encoding="utf-8")
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._writer_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._write_loop,
            name="rollout-log-writer",
            daemon=True,
        )
        self._thread.start()

    def event(self, kind: str, **payload: Any) -> None:
        if self._writer_error is not None:
            raise RuntimeError("rollout log writer failed") from self._writer_error
        self._queue.put(
            json.dumps(
                {"type": kind, "timestamp_ns": time.monotonic_ns(), **payload},
                separators=(",", ":"),
            )
            + "\n"
        )

    def _write_loop(self) -> None:
        next_sync = time.monotonic() + 0.5
        try:
            while True:
                line = self._queue.get()
                if line is None:
                    break
                self.events.write(line)
                if time.monotonic() >= next_sync:
                    self.events.flush()
                    os.fsync(self.events.fileno())
                    next_sync = time.monotonic() + 0.5
            self.events.flush()
            os.fsync(self.events.fileno())
        except BaseException as exc:
            self._writer_error = exc

    def save_observation(
        self,
        *,
        request_id: int,
        state: list[float],
        image_timestamps_ns: tuple[int, int],
        head_jpeg: bytes,
        wrist_jpeg: bytes,
    ) -> None:
        root = self.root / "observations"
        root.mkdir(exist_ok=True)
        stem = f"{int(request_id):06d}"
        (root / f"{stem}_head.jpg").write_bytes(head_jpeg)
        (root / f"{stem}_wrist.jpg").write_bytes(wrist_jpeg)
        self.event(
            "observation_snapshot",
            request_id=int(request_id),
            state=state,
            image_timestamps_ns=list(image_timestamps_ns),
            head_file=f"observations/{stem}_head.jpg",
            wrist_file=f"observations/{stem}_wrist.jpg",
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("rollout log writer did not stop")
        self.events.close()
        if self._writer_error is not None:
            raise RuntimeError("rollout log writer failed") from self._writer_error
