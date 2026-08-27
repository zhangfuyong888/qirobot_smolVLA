from __future__ import annotations

from tasks.progress_dashboard import DashboardSnapshot, format_compact, format_dashboard


def _snapshot(**overrides) -> DashboardSnapshot:
    values = {
        "episode": 1,
        "episode_total": 2,
        "attempt": 1,
        "clock_time": "12:00:00",
        "success_count": 0,
        "failure_count": 0,
        "phase_index": 1,
        "phase_total": 2,
        "phase_name": "test_phase",
        "step": 1,
        "step_total": 100,
        "elapsed_s": 1.0,
        "timeout_s": 10.0,
        "episode_sim_s": 1.0,
        "collection_elapsed_s": 1.0,
        "frames": 1,
        "left_pos": 1.0,
        "left_rot": 1.0,
        "right_pos": 1.0,
        "right_rot": 1.0,
        "left_pos_limit": 0.05,
        "left_rot_limit": 0.5,
        "right_pos_limit": 0.05,
        "right_rot_limit": 0.5,
        "left_tcp_gate": False,
        "right_tcp_gate": True,
        "drawer_open_m": 0.04,
        "drawer_open_min_m": 0.08,
        "drawer_open_limit_m": float("nan"),
    }
    values.update(overrides)
    return DashboardSnapshot(**values)


def test_disabled_tcp_metrics_are_neutral_instead_of_failed() -> None:
    text = format_dashboard(_snapshot(), color=False)
    left_row = next(line for line in text.splitlines() if "L-Pos" in line)
    right_row = next(line for line in text.splitlines() if "R-Pos" in line)
    assert "—" in left_row
    assert "🔴" not in left_row
    assert "🔴" in right_row


def test_drawer_minimum_gate_is_reported_in_full_and_compact_logs() -> None:
    snapshot = _snapshot()
    dashboard = format_dashboard(snapshot, color=False)
    compact = format_compact(snapshot)
    assert "DRAWER  0.040 >= 0.080 m 🔴" in dashboard
    assert "DRAWER 0.040>=0.080m" in compact


def test_disabled_tcp_metrics_receive_gray_ansi_highlight() -> None:
    text = format_dashboard(_snapshot(), color=True)
    assert "\033[90mL-Pos" in text
