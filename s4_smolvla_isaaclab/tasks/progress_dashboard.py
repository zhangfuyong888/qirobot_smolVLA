"""Terminal progress dashboard for scripted task collection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata


@dataclass(frozen=True)
class DashboardSnapshot:
    episode: int
    episode_total: int
    attempt: int
    clock_time: str
    success_count: int
    failure_count: int
    phase_index: int
    phase_total: int
    phase_name: str
    step: int
    step_total: int
    elapsed_s: float
    timeout_s: float
    episode_sim_s: float
    collection_elapsed_s: float
    frames: int
    left_pos: float
    left_rot: float
    right_pos: float
    right_rot: float
    left_pos_limit: float
    left_rot_limit: float
    right_pos_limit: float
    right_rot_limit: float
    left_tcp_gate: bool
    right_tcp_gate: bool
    drawer_open_m: float
    drawer_open_min_m: float
    drawer_open_limit_m: float


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char) or char in {"\ufe0e", "\ufe0f"}:
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _fit(text: str, width: int) -> str:
    result = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        )
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    return "".join(result) + " " * max(width - used, 0)


def _indicator(value: float, limit: float) -> str:
    return "🟢" if math.isfinite(value) and value <= limit else "🔴"


def _metric(label: str, value: float, limit: float, unit: str, *, active: bool = True) -> str:
    indicator = _indicator(value, limit) if active else "—"
    return f"{label:<6} {value:6.3f} / {limit:6.3f} {unit:<3} {indicator}"


def _phase_progress(snapshot: DashboardSnapshot) -> float:
    phase_fraction = min(max(snapshot.step / max(snapshot.step_total, 1), 0.0), 1.0)
    return min(max((snapshot.phase_index - 1 + phase_fraction) / max(snapshot.phase_total, 1), 0.0), 1.0)


def _episode_progress(snapshot: DashboardSnapshot) -> float:
    return min(max(snapshot.episode / max(snapshot.episode_total, 1), 0.0), 1.0)


def _bar(progress: float, width: int) -> str:
    size = max(int(width), 10)
    filled = min(int(round(progress * size)), size)
    return "█" * filled + "░" * (size - filled)


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "gray": "\033[90m",
    "cyan": "\033[36;1m",
    "blue": "\033[34;1m",
    "yellow": "\033[33;1m",
    "red": "\033[31;1m",
    "green": "\033[32;1m",
    "magenta": "\033[35;1m",
}


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "--:--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _paint(text: str, color: str, enabled: bool) -> str:
    return f"{_ANSI[color]}{text}{_ANSI['reset']}" if enabled else text


def format_dashboard(
    snapshot: DashboardSnapshot,
    *,
    width: int = 78,
    bar_width: int = 24,
    color: bool = False,
) -> str:
    """Build a fixed-width Unicode panel with optional ANSI emphasis."""
    width = max(int(width), 72)
    inner = width - 2
    phase_progress = _phase_progress(snapshot)
    episode_progress = _episode_progress(snapshot)
    phase_bar = _bar(phase_progress, bar_width)
    episode_bar = _bar(episode_progress, bar_width)
    completed_attempts = snapshot.success_count + snapshot.failure_count
    success_rate = snapshot.success_count / completed_attempts if completed_attempts else 0.0
    eta_s = (
        snapshot.collection_elapsed_s / snapshot.success_count
        * max(snapshot.episode_total - snapshot.success_count, 0)
        if snapshot.success_count > 0
        else float("nan")
    )
    attempt_total = completed_attempts + 1
    left_pos_metric = _metric(
        "L-Pos", snapshot.left_pos, snapshot.left_pos_limit, "m", active=snapshot.left_tcp_gate
    )
    left_rot_metric = _metric(
        "L-Rot", snapshot.left_rot, snapshot.left_rot_limit, "rad", active=snapshot.left_tcp_gate
    )
    right_pos_metric = _metric(
        "R-Pos", snapshot.right_pos, snapshot.right_pos_limit, "m", active=snapshot.right_tcp_gate
    )
    right_rot_metric = _metric(
        "R-Rot", snapshot.right_rot, snapshot.right_rot_limit, "rad", active=snapshot.right_tcp_gate
    )

    def border(left: str, fill: str, right: str) -> str:
        return _paint(left + fill * inner + right, "gray", color)

    def row(content: str, highlights: tuple[tuple[str, str], ...] = ()) -> str:
        fitted = _fit(content, inner)
        if color:
            for fragment, shade in highlights:
                fitted = fitted.replace(fragment, _paint(fragment, shade, True), 1)
        return _paint("║", "gray", color) + fitted + _paint("║", "gray", color)

    task_text = (
        f" TASK [{phase_bar}] {phase_progress * 100:5.1f}%  "
        f"(Phase {snapshot.phase_index}/{snapshot.phase_total})"
    )
    data_text = (
        f" DATA [{episode_bar}] {episode_progress * 100:5.1f}%  "
        f"(Episode {snapshot.episode}/{snapshot.episode_total})"
    )
    stats_text = (
        f" SUCCESS {snapshot.success_count:03d}   FAIL {snapshot.failure_count:03d}   "
        f"ATTEMPTS {attempt_total:03d}   RATE {success_rate * 100:5.1f}%"
    )
    drawer_min_active = math.isfinite(snapshot.drawer_open_min_m)
    drawer_max_active = math.isfinite(snapshot.drawer_open_limit_m)
    drawer_gate_active = drawer_min_active or drawer_max_active
    drawer_gate_ok = math.isfinite(snapshot.drawer_open_m)
    if drawer_min_active:
        drawer_gate_ok = drawer_gate_ok and snapshot.drawer_open_m >= snapshot.drawer_open_min_m
    if drawer_max_active:
        drawer_gate_ok = drawer_gate_ok and snapshot.drawer_open_m <= snapshot.drawer_open_limit_m
    if drawer_min_active and drawer_max_active:
        drawer_requirement = (
            f"[{snapshot.drawer_open_min_m:6.3f}, {snapshot.drawer_open_limit_m:6.3f}]"
        )
    elif drawer_min_active:
        drawer_requirement = f">={snapshot.drawer_open_min_m:6.3f}"
    elif drawer_max_active:
        drawer_requirement = f"<={snapshot.drawer_open_limit_m:6.3f}"
    else:
        drawer_requirement = "monitor"
    drawer_text = (
        f"DRAWER {snapshot.drawer_open_m:6.3f} {drawer_requirement} m "
        f"{'🟢' if drawer_gate_ok else '🔴'}"
        if drawer_gate_active
        else f"DRAWER {snapshot.drawer_open_m:6.3f} m (monitor)"
    )
    gates_text = (
        f" GATES L-TCP={'ON ' if snapshot.left_tcp_gate else 'OFF'}  "
        f"R-TCP={'ON ' if snapshot.right_tcp_gate else 'OFF'}   {drawer_text}"
    )
    rows = [
        border("╔", "═", "╗"),
        row(
            f" EP{snapshot.episode:03d}/{snapshot.episode_total:03d} TRY{snapshot.attempt:02d}   "
            f"TIME {snapshot.clock_time}",
            ((f"EP{snapshot.episode:03d}/{snapshot.episode_total:03d}", "cyan"),
             (f"TRY{snapshot.attempt:02d}", "yellow"), (snapshot.clock_time, "gray")),
        ),
        row(
            f" PHASE {snapshot.phase_index:02d}/{snapshot.phase_total:02d}  {snapshot.phase_name}   "
            f"STEP {snapshot.step:04d}/{snapshot.step_total:04d}",
            ((f"PHASE {snapshot.phase_index:02d}/{snapshot.phase_total:02d}", "blue"),
             (snapshot.phase_name, "blue")),
        ),
        border("╠", "═", "╣"),
        row(task_text, ((f"[{phase_bar}]", "blue"), (f"{phase_progress * 100:5.1f}%", "blue"))),
        border("╠", "═", "╣"),
        row(
            " " + left_pos_metric + "   " + left_rot_metric,
            ((left_pos_metric, "gray" if not snapshot.left_tcp_gate else (
                "green" if snapshot.left_pos <= snapshot.left_pos_limit else "red"
            )),
             (left_rot_metric, "gray" if not snapshot.left_tcp_gate else (
                 "green" if snapshot.left_rot <= snapshot.left_rot_limit else "red"
             ))),
        ),
        row(
            " " + right_pos_metric + "   " + right_rot_metric,
            ((right_pos_metric, "gray" if not snapshot.right_tcp_gate else (
                "green" if snapshot.right_pos <= snapshot.right_pos_limit else "red"
            )),
             (right_rot_metric, "gray" if not snapshot.right_tcp_gate else (
                 "green" if snapshot.right_rot <= snapshot.right_rot_limit else "red"
             ))),
        ),
        row(
            gates_text,
            ((f"L-TCP={'ON ' if snapshot.left_tcp_gate else 'OFF'}",
              "yellow" if snapshot.left_tcp_gate else "gray"),
             (f"R-TCP={'ON ' if snapshot.right_tcp_gate else 'OFF'}",
              "yellow" if snapshot.right_tcp_gate else "gray"),
             (drawer_text, "green" if drawer_gate_ok else ("red" if drawer_gate_active else "gray"))),
        ),
        border("╠", "═", "╣"),
        row(
            f" EPISODE wall={snapshot.elapsed_s:6.1f}/{snapshot.timeout_s:.1f}s   "
            f"sim={snapshot.episode_sim_s:6.1f}s   frames={snapshot.frames:05d}",
            (("EPISODE", "yellow"),),
        ),
        row(
            f" TOTAL {_duration(snapshot.collection_elapsed_s)}   ETA {_duration(eta_s)}   "
            f"completed_attempts={completed_attempts:03d}",
            (("TOTAL", "cyan"), ("ETA", "magenta")),
        ),
        border("╠", "═", "╣"),
        row(data_text, ((f"[{episode_bar}]", "green"), (f"{episode_progress * 100:5.1f}%", "green"))),
        row(
            stats_text,
            ((f"SUCCESS {snapshot.success_count:03d}", "green"),
             (f"FAIL {snapshot.failure_count:03d}", "red"),
             (f"ATTEMPTS {attempt_total:03d}", "yellow")),
        ),
        border("╚", "═", "╝"),
    ]
    return "\n".join(rows)


def format_compact(snapshot: DashboardSnapshot, *, bar_width: int = 16) -> str:
    """Build a single line suitable for redirected logs and CI output."""
    phase_progress = _phase_progress(snapshot)
    episode_progress = _episode_progress(snapshot)
    phase_bar = _bar(phase_progress, bar_width)
    episode_bar = _bar(episode_progress, bar_width)
    metrics = (
        _metric("L-Pos", snapshot.left_pos, snapshot.left_pos_limit, "m", active=snapshot.left_tcp_gate),
        _metric("L-Rot", snapshot.left_rot, snapshot.left_rot_limit, "rad", active=snapshot.left_tcp_gate),
        _metric("R-Pos", snapshot.right_pos, snapshot.right_pos_limit, "m", active=snapshot.right_tcp_gate),
        _metric("R-Rot", snapshot.right_rot, snapshot.right_rot_limit, "rad", active=snapshot.right_tcp_gate),
    )
    return (
        f"TIME {snapshot.clock_time} | "
        f"EP{snapshot.episode:03d}/{snapshot.episode_total:03d} TRY{snapshot.attempt:02d} | "
        f"PHASE {snapshot.phase_index:02d}/{snapshot.phase_total:02d} {snapshot.phase_name} | "
        f"STEP {snapshot.step:04d}/{snapshot.step_total:04d} | "
        f"ELAP {snapshot.elapsed_s:5.1f}/{snapshot.timeout_s:5.1f}s | "
        + " | ".join(metrics)
        + f" | TASK [{phase_bar}] {phase_progress * 100:5.1f}%"
        + f" | WALL {snapshot.elapsed_s:.1f}/{snapshot.timeout_s:.1f}s"
        + f" SIM {snapshot.episode_sim_s:.1f}s FRAMES {snapshot.frames}"
        + f" TOTAL {_duration(snapshot.collection_elapsed_s)}"
        + f" | GATES L={int(snapshot.left_tcp_gate)} R={int(snapshot.right_tcp_gate)}"
        + (
            f" DRAWER {snapshot.drawer_open_m:.3f}>={snapshot.drawer_open_min_m:.3f}m"
            if math.isfinite(snapshot.drawer_open_min_m)
            and not math.isfinite(snapshot.drawer_open_limit_m)
            else f" DRAWER {snapshot.drawer_open_m:.3f}<={snapshot.drawer_open_limit_m:.3f}m"
            if math.isfinite(snapshot.drawer_open_limit_m)
            and not math.isfinite(snapshot.drawer_open_min_m)
            else f" DRAWER {snapshot.drawer_open_min_m:.3f}<={snapshot.drawer_open_m:.3f}<={snapshot.drawer_open_limit_m:.3f}m"
            if math.isfinite(snapshot.drawer_open_min_m)
            and math.isfinite(snapshot.drawer_open_limit_m)
            else f" DRAWER {snapshot.drawer_open_m:.3f}m"
        )
        + f" | DATA [{episode_bar}] {episode_progress * 100:5.1f}% "
        + f"(Episode {snapshot.episode}/{snapshot.episode_total}) "
        + f"SUCCESS {snapshot.success_count:03d} FAIL {snapshot.failure_count:03d} "
        + f"ATTEMPTS {snapshot.success_count + snapshot.failure_count + 1:03d}"
    )
