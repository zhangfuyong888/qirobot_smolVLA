from __future__ import annotations

from types import SimpleNamespace

import pytest

from real_vla_stack.host.inference.server import RequestSessionGuard


class _Runner:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _request(session: str, request_id: int):
    return SimpleNamespace(session_id=session, request_id=request_id)


def test_session_guard_resets_once_and_rejects_replay() -> None:
    runner = _Runner()
    guard = RequestSessionGuard()
    guard.accept(runner, _request("a", 0))
    guard.accept(runner, _request("a", 1))
    assert runner.resets == 1
    with pytest.raises(ValueError, match="stale"):
        guard.accept(runner, _request("a", 1))


def test_new_session_must_start_at_zero() -> None:
    runner = _Runner()
    guard = RequestSessionGuard()
    with pytest.raises(ValueError, match="start with request_id=0"):
        guard.accept(runner, _request("new", 4))
