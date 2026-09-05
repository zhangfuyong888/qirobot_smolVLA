"""Dependency-free ANSI dashboard for real-robot collection."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO

from hardware_teleop.hooks import TeleopStatus


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class CollectionDashboardStatus:
    state: str
    episode_id: int
    saved_episodes: int
    allowed_arms: str
    duration_s: float
    state_count: int
    action_count: int
    quality: str
    disk_free_gb: float
    session_name: str
    home_status: str
    camera_stats: Mapping[str, Mapping[str, float | int]]


class CollectionConsoleDashboard:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"

    def __init__(
        self,
        *,
        enabled: bool,
        event_log_path: Path,
        refresh_interval_s: float = 1.0,
        stream: TextIO | None = None,
        force_terminal: bool = False,
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = bool(enabled and (is_tty or force_terminal))
        self.color = self.enabled and "NO_COLOR" not in os.environ
        self.event_log_path = Path(event_log_path)
        self.refresh_interval_s = max(float(refresh_interval_s), 0.1)
        self._events: deque[tuple[str, str, str]] = deque(maxlen=5)
        self._lock = threading.Lock()
        self._started = False
        self._last_render_s = 0.0

    def add_event(self, message: str, level: str = "info") -> None:
        text = " ".join(str(message).split())
        if not text:
            return
        normalized = level.lower()
        if normalized not in {"info", "success", "warning", "error"}:
            normalized = "info"
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._events.append((stamp, normalized, text))
            try:
                self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.event_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"[{normalized}] {text}\n"
                    )
            except OSError:
                # Logging is observational and must never interrupt robot control.
                pass

    def render(
        self,
        teleop: TeleopStatus,
        collection: CollectionDashboardStatus,
        *,
        force: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        if not force and teleop.monotonic_s - self._last_render_s < self.refresh_interval_s:
            return True
        self._last_render_s = teleop.monotonic_s
        try:
            if not self._started:
                self.stream.write("\x1b[?1049h\x1b[?25l")
                self._started = True
            width = max(78, min(shutil.get_terminal_size((118, 32)).columns, 118))
            lines = self._build_lines(width, teleop, collection)
            self.stream.write("\x1b[H\x1b[2J" + "\n".join(lines) + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            self.enabled = False
            return False
        return True

    def close(self) -> None:
        if not self._started:
            return
        try:
            self.stream.write("\x1b[?25h\x1b[?1049l")
            self.stream.flush()
        except (OSError, ValueError):
            pass
        self._started = False

    def _build_lines(
        self,
        width: int,
        teleop: TeleopStatus,
        collection: CollectionDashboardStatus,
    ) -> list[str]:
        active = _active_label(teleop.left_active, teleop.right_active)
        output_ok = teleop.command_output_enabled and not teleop.output_relinquished
        quest_online = teleop.quest_clients > 0
        frame_age_ms = teleop.quest_frame_age_s * 1.0e3
        margin = teleop.minimum_joint_limit_margin_rad
        margin_text = "n/a" if not math.isfinite(margin) else f"{margin:.3f} rad"
        quality_level = (
            "error" if "INVALID" in collection.quality else
            "warning" if "WARN" in collection.quality else
            "success" if "PASS" in collection.quality else "info"
        )
        lines = [self._top(width, "S4 REAL-VLA COLLECTION")]
        lines.append(
            self._row(
                width,
                f" STATE {self._state(collection.state)}   "
                f"EP {self._value(f'{collection.episode_id:06d}', 'info')}   "
                f"SAVED {self._value(str(collection.saved_episodes), 'success')}   "
                f"SESSION {self._dim(collection.session_name)}",
            )
        )
        lines.append(self._divider(width, "CONTROL"))
        lines.append(
            self._row(
                width,
                f" ALLOWED {self._value(collection.allowed_arms, 'info')}   "
                f"MOVING {self._value(active, 'success' if active != 'NONE' else 'muted')}   "
                f"GRIP L/R {teleop.left_grip:.2f}/{teleop.right_grip:.2f}   "
                f"TRIGGER L/R {teleop.left_trigger:.2f}/{teleop.right_trigger:.2f}   "
                f"OUTPUT {self._value('ENABLED' if output_ok else 'STOPPED', 'success' if output_ok else 'error')}",
            )
        )
        regrip = []
        if teleop.left_requires_release:
            regrip.append("L")
        if teleop.right_requires_release:
            regrip.append("R")
        lines.append(
            self._row(
                width,
                f" CLUTCH {self._value('REGRIP ' + '/'.join(regrip), 'warning') if regrip else self._value('READY', 'success')}   "
                f"FAULT {self._value(teleop.fault_reason or 'NONE', 'error' if teleop.fault_reason else 'success')}",
            )
        )
        lines.append(self._divider(width, "QUEST"))
        tracking_ok = teleop.left_tracking and teleop.right_tracking
        lines.append(
            self._row(
                width,
                f" CONNECTION {self._value('ONLINE' if quest_online else 'OFFLINE', 'success' if quest_online else 'error')}   "
                f"INPUT {self._value('STALE' if teleop.input_stale else 'OK', 'error' if teleop.input_stale else 'success')}   "
                f"INPUT AGE {self._latency(frame_age_ms, 60.0, 120.0)}   "
                f"TRACK L/R {self._value(_yes_no(teleop.left_tracking) + '/' + _yes_no(teleop.right_tracking), 'success' if tracking_ok else 'error')}   "
                f"CAL {self._value('READY' if teleop.calibrated else 'REQUIRED', 'success' if teleop.calibrated else 'error')}",
            )
        )
        lines.append(
            self._row(
                width,
                f" BOUNDARY {self._value('SAFE' if teleop.boundary_safe else 'OUTSIDE', 'success' if teleop.boundary_safe else 'error')}",
            )
        )
        lines.append(self._divider(width, "ROBOT / SAFETY"))
        lines.append(
            self._row(
                width,
                f" LOOP {self._rate(teleop.loop_hz, teleop.target_hz)}   "
                f"LOWSTATE {self._latency(teleop.state_age_s * 1.0e3, 80.0, 200.0)}   "
                f"TRACK ERR {self._threshold(teleop.proximal_tracking_error_rad, 0.10, 0.18, 'rad')}   "
                f"TCP ERR L/R {self._threshold(max(teleop.left_tcp_error_m, teleop.right_tcp_error_m), 0.05, 0.10, 'm')}",
            )
        )
        margin_level = "success" if not math.isfinite(margin) or margin >= 0.10 else "warning" if margin >= 0.03 else "error"
        lines.append(
            self._row(
                width,
                f" JOINT MARGIN {self._value(margin_text, margin_level)}   "
                f"LIMIT-ACTIVE {self._value(str(teleop.joint_limit_active_joints), 'warning' if teleop.joint_limit_active_joints else 'success')}   "
                f"STATE-FEED {self._value('STALE' if teleop.state_feed_stale else 'OK', 'error' if teleop.state_feed_stale else 'success')}   "
                f"ARM-GRAPH {self._value('CONFLICT' if teleop.arm_graph_conflict else 'OK', 'error' if teleop.arm_graph_conflict else 'success')}",
            )
        )
        lines.append(self._divider(width, "DATASET"))
        lines.append(
            self._row(
                width,
                f" PHASE {self._state(collection.state)}   "
                f"DURATION {collection.duration_s:6.1f}s   "
                f"STATE/ACTION {collection.state_count}/{collection.action_count}   "
                f"QUALITY {self._value(collection.quality, quality_level)}   "
                f"DISK {self._disk(collection.disk_free_gb)}",
            )
        )
        for name, stats in collection.camera_stats.items():
            age = float(stats.get("last_frame_age_ms", math.inf))
            drops = int(stats.get("writer_drops", 0))
            reads = int(stats.get("read_failures", 0))
            captured = int(stats.get("episode_frames", 0))
            lines.append(
                self._row(
                    width,
                    f" {name.upper():<14} AGE {self._latency(age, 60.0, 100.0)}   "
                    f"EP-FRAMES {captured:<6d} DROPS {self._value(str(drops), 'error' if drops else 'success')}   "
                    f"READ-FAIL {self._value(str(reads), 'warning' if reads else 'success')}",
                )
            )
        if collection.home_status:
            lines.append(self._row(width, f" HOME {self._value(collection.home_status, 'warning')}"))
        lines.append(self._divider(width, "RECENT EVENTS"))
        with self._lock:
            events = list(self._events)
        if not events:
            lines.append(self._row(width, f" {self._dim('No events yet')}"))
        else:
            for stamp, level, message in events:
                lines.append(self._row(width, f" {stamp}  {self._value(message, level)}"))
        lines.append(self._bottom(width))
        lines.append(
            self._fit(
                "A HOME+RECORD   B END   X SAVE   HOLD Y DISCARD   Ctrl+C STOP",
                width,
            )
        )
        return lines

    def _state(self, state: str) -> str:
        level = "success" if state == "RECORDING" else "warning" if "HOMING" in state else "info"
        return self._value(state, level)

    def _rate(self, value: float, target: float) -> str:
        ratio = value / max(target, 1.0e-6)
        level = "success" if ratio >= 0.90 else "warning" if ratio >= 0.75 else "error"
        return self._value(f"{value:.1f}/{target:.1f} Hz", level)

    def _latency(self, value_ms: float, warn_ms: float, bad_ms: float) -> str:
        if not math.isfinite(value_ms):
            return self._value("n/a", "error")
        level = "success" if value_ms < warn_ms else "warning" if value_ms < bad_ms else "error"
        return self._value(f"{value_ms:.0f} ms", level)

    def _threshold(self, value: float, warn: float, bad: float, unit: str) -> str:
        level = "success" if value < warn else "warning" if value < bad else "error"
        return self._value(f"{value:.3f} {unit}", level)

    def _disk(self, value: float) -> str:
        level = "success" if value >= 10.0 else "warning" if value >= 2.0 else "error"
        return self._value(f"{value:.1f} GB", level)

    def _value(self, text: str, level: str) -> str:
        if not self.color:
            return str(text)
        colors = {
            "error": self.RED,
            "warning": self.YELLOW,
            "success": self.GREEN,
            "info": self.CYAN,
            "muted": self.DIM,
        }
        return f"{self.BOLD}{colors.get(level, '')}{text}{self.RESET}"

    def _dim(self, text: str) -> str:
        return f"{self.DIM}{text}{self.RESET}" if self.color else text

    def _row(self, width: int, text: str) -> str:
        content = self._fit(text, width - 2)
        padding = max(width - 2 - _display_width(content), 0)
        return f"│{content}{' ' * padding}│"

    def _top(self, width: int, title: str) -> str:
        return _border(width, "┌", "┐", title)

    def _divider(self, width: int, title: str) -> str:
        return _border(width, "├", "┤", title)

    def _bottom(self, width: int) -> str:
        return "└" + "─" * (width - 2) + "┘"

    def _fit(self, text: str, width: int) -> str:
        if _display_width(text) <= width:
            return text
        plain = _ANSI_RE.sub("", text)
        result = ""
        used = 0
        for char in plain:
            char_width = _char_width(char)
            if used + char_width > max(width - 1, 0):
                break
            result += char
            used += char_width
        return result + "…"


def _active_label(left: bool, right: bool) -> str:
    if left and right:
        return "BOTH"
    if left:
        return "LEFT"
    if right:
        return "RIGHT"
    return "NONE"


def _yes_no(value: bool) -> str:
    return "OK" if value else "LOST"


def _char_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in _ANSI_RE.sub("", text))


def _border(width: int, left: str, right: str, title: str) -> str:
    label = f" {title} "
    available = max(width - 2 - len(label), 0)
    before = available // 2
    after = available - before
    return left + "─" * before + label + "─" * after + right
