"""Collection state machine.

STARTUP -> HOMING -> READY -> HOMING_TO_RECORD -> RECORDING -> RETURNING_HOME -> REVIEW
READY allows free teleop. A homes, then RECORDING starts from the home pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from real_vla.input.quest_buttons import QuestButtons


class CollectionState(str, Enum):
    STARTUP = "STARTUP"
    HOMING = "HOMING"
    READY = "READY"
    HOMING_TO_RECORD = "HOMING_TO_RECORD"
    RECORDING = "RECORDING"
    RETURNING_HOME = "RETURNING_HOME"
    REVIEW = "REVIEW"


class CollectionEvent(str, Enum):
    STARTUP_OK = "STARTUP_OK"
    HOME_DONE = "HOME_DONE"
    PRE_RECORD_HOME = "PRE_RECORD_HOME"
    START = "START"
    END = "END"
    WRITER_DONE = "WRITER_DONE"
    SAVE = "SAVE"
    DISCARD = "DISCARD"
    LOW_DISK = "LOW_DISK"
    ABORT_RECORDING = "ABORT_RECORDING"


@dataclass(frozen=True)
class Transition:
    state: CollectionState
    event: CollectionEvent | None
    message: str = ""


class CollectionStateMachine:
    def __init__(self) -> None:
        self.state = CollectionState.STARTUP
        self.writer_finalized = False
        self.robot_at_home = False
        self.review_ready = False

    def reset_home_flags(self) -> None:
        self.writer_finalized = False
        self.robot_at_home = False
        self.review_ready = False

    def on_startup_ok(self) -> Transition:
        return self._go(CollectionState.HOMING, CollectionEvent.STARTUP_OK, "startup complete")

    def on_home_arrived(self) -> Transition:
        if self.state == CollectionState.HOMING:
            return self._go(
                CollectionState.READY,
                CollectionEvent.HOME_DONE,
                "READY  Grip teleop  A: home then record",
            )
        if self.state == CollectionState.HOMING_TO_RECORD:
            return self._go(
                CollectionState.RECORDING,
                CollectionEvent.START,
                "RECORDING  B: END",
            )
        if self.state == CollectionState.RETURNING_HOME:
            self.robot_at_home = True
            return self._maybe_review("robot at home")
        return Transition(self.state, None)

    def on_writer_finalized(self) -> Transition:
        if self.state != CollectionState.RETURNING_HOME:
            return Transition(self.state, None)
        self.writer_finalized = True
        return self._maybe_review("episode writer finalized")

    def on_buttons(self, buttons: QuestButtons, *, disk_ok: bool = True) -> Transition:
        if self.state == CollectionState.READY and buttons.a_rising:
            if not disk_ok:
                return Transition(self.state, CollectionEvent.LOW_DISK, "A ignored: LOW DISK SPACE")
            self.reset_home_flags()
            return self._go(
                CollectionState.HOMING_TO_RECORD,
                CollectionEvent.PRE_RECORD_HOME,
                "A  homing then RECORDING",
            )
        if self.state == CollectionState.RECORDING and buttons.b_rising:
            self.reset_home_flags()
            return self._go(
                CollectionState.RETURNING_HOME,
                CollectionEvent.END,
                "END  returning home",
            )
        if self.state == CollectionState.REVIEW and buttons.x_rising:
            return self._go(
                CollectionState.READY,
                CollectionEvent.SAVE,
                "saved  READY  Grip teleop  A: home then record",
            )
        if self.state == CollectionState.REVIEW and buttons.y_held:
            return self._go(
                CollectionState.READY,
                CollectionEvent.DISCARD,
                "discarded  READY  Grip teleop  A: home then record",
            )
        return Transition(self.state, None)

    def abort_recording(self, reason: str) -> Transition:
        if self.state != CollectionState.RECORDING:
            return Transition(self.state, None)
        self.reset_home_flags()
        return self._go(
            CollectionState.RETURNING_HOME,
            CollectionEvent.ABORT_RECORDING,
            reason,
        )

    def _maybe_review(self, message: str) -> Transition:
        if self.robot_at_home and self.writer_finalized:
            self.review_ready = True
            return self._go(CollectionState.REVIEW, CollectionEvent.WRITER_DONE, message)
        return Transition(self.state, None, message)

    def _go(self, state: CollectionState, event: CollectionEvent, message: str) -> Transition:
        self.state = state
        return Transition(state, event, message)
