"""Small fault latch for real-robot teleoperation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


SDK_ARM_REPLAY_MARKERS = (b"rt/lowcmd_replay", b"Arms command - mode_ctrl")


def find_verified_arm_replay_sdk_process(
    proc_root: Path = Path("/proc"),
    *,
    process_names: tuple[str, ...] = ("sn_loco_server",),
    approved_sha256: tuple[str, ...] = (),
) -> tuple[int, Path]:
    """Return a running approved SDK binary with the arm-only replay handler."""
    approved = {value.lower() for value in approved_sha256}
    candidates: list[str] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if comm not in process_names:
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
            payload = executable.read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            candidates.append(f"pid={entry.name} comm={comm} executable=unreadable")
            continue
        if all(marker in payload for marker in SDK_ARM_REPLAY_MARKERS):
            digest = hashlib.sha256(payload).hexdigest()
            if approved and digest not in approved:
                candidates.append(
                    f"pid={entry.name} comm={comm} executable={executable} "
                    f"sha256={digest} not-approved"
                )
                continue
            return int(entry.name), executable
        candidates.append(
            f"pid={entry.name} comm={comm} executable={executable} arm-replay-marker=missing"
        )
    detail = "; ".join(candidates) if candidates else "no sn_loco_server process found"
    raise RuntimeError(
        "verified SDK arm-only replay support is required before command output: "
        + detail
    )


@dataclass
class TeleopFaultLatch:
    """Hold motion after a controller fault until both grips are released."""

    reason: str | None = None
    trip_count: int = 0

    @property
    def active(self) -> bool:
        return self.reason is not None

    def trip(self, reason: str) -> bool:
        """Latch a fault and return True only for a new fault transition."""
        message = str(reason).strip() or "unknown controller fault"
        if self.reason is not None:
            return False
        self.reason = message
        self.trip_count += 1
        return True

    def clear_if_released(
        self,
        *,
        left_clutch: bool,
        right_clutch: bool,
        state_feed_ok: bool,
    ) -> bool:
        """Clear only while both grips are released and feedback is healthy."""
        if self.reason is None or left_clutch or right_clutch or not state_feed_ok:
            return False
        self.reason = None
        return True
