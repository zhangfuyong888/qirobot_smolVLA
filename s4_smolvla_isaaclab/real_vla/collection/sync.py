"""Causal latest-before-t alignment used for quality reports, not online recording."""

from __future__ import annotations

import numpy as np


def latest_before(timestamps_ns: np.ndarray, t_ns: int) -> int | None:
    if timestamps_ns.size == 0:
        return None
    index = int(np.searchsorted(timestamps_ns, int(t_ns), side="right") - 1)
    if index < 0:
        return None
    return index


def age_ms(timestamps_ns: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    ages = np.full(grid_ns.shape, np.nan, dtype=np.float64)
    for i, t_ns in enumerate(grid_ns):
        index = latest_before(timestamps_ns, int(t_ns))
        if index is None:
            continue
        ages[i] = max(int(t_ns) - int(timestamps_ns[index]), 0) / 1.0e6
    return ages


def percentile_report(values_ms: np.ndarray) -> dict[str, float]:
    finite = values_ms[np.isfinite(values_ms)]
    if finite.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def alignment_report(
    *,
    t_start_ns: int,
    t_end_ns: int,
    dt_s: float = 0.05,
    head_ts: np.ndarray,
    wrist_ts: np.ndarray,
    state_ts: np.ndarray,
    action_ts: np.ndarray,
) -> dict[str, dict[str, float]]:
    if t_end_ns <= t_start_ns:
        return {}
    step = max(int(dt_s * 1.0e9), 1)
    grid = np.arange(t_start_ns, t_end_ns + 1, step, dtype=np.int64)
    return {
        "head_age_ms": percentile_report(age_ms(np.asarray(head_ts, dtype=np.int64), grid)),
        "wrist_age_ms": percentile_report(age_ms(np.asarray(wrist_ts, dtype=np.int64), grid)),
        "state_age_ms": percentile_report(age_ms(np.asarray(state_ts, dtype=np.int64), grid)),
        "action_age_ms": percentile_report(age_ms(np.asarray(action_ts, dtype=np.int64), grid)),
    }
