"""Quest ABXY edges. A/B live on the right controller, X/Y on the left."""

from __future__ import annotations

from dataclasses import dataclass

from teleoperation.protocol import ControllerFrame


@dataclass(frozen=True)
class QuestButtons:
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    a_rising: bool = False
    b_rising: bool = False
    x_rising: bool = False
    y_rising: bool = False
    y_held: bool = False


def _pressed(sample_buttons: tuple[float, ...], index: int, threshold: float) -> bool:
    if index < 0 or index >= len(sample_buttons):
        return False
    return float(sample_buttons[index]) >= threshold


class QuestButtonDecoder:
    def __init__(
        self,
        *,
        a_index: int = 4,
        b_index: int = 5,
        x_index: int = 4,
        y_index: int = 5,
        press_threshold: float = 0.5,
        discard_hold_s: float = 0.6,
    ) -> None:
        self.a_index = int(a_index)
        self.b_index = int(b_index)
        self.x_index = int(x_index)
        self.y_index = int(y_index)
        self.press_threshold = float(press_threshold)
        self.discard_hold_s = float(discard_hold_s)
        self._prev_a = False
        self._prev_b = False
        self._prev_x = False
        self._prev_y = False
        self._y_down_s: float | None = None
        self._y_hold_fired = False

    def update(self, frame: ControllerFrame | None, now_s: float) -> QuestButtons:
        right_buttons = () if frame is None or not frame.right.valid else frame.right.buttons
        left_buttons = () if frame is None or not frame.left.valid else frame.left.buttons
        a = _pressed(right_buttons, self.a_index, self.press_threshold)
        b = _pressed(right_buttons, self.b_index, self.press_threshold)
        x = _pressed(left_buttons, self.x_index, self.press_threshold)
        y = _pressed(left_buttons, self.y_index, self.press_threshold)

        y_held = False
        if y:
            if self._y_down_s is None:
                self._y_down_s = float(now_s)
            elif (
                not self._y_hold_fired
                and float(now_s) - self._y_down_s >= self.discard_hold_s
            ):
                y_held = True
                self._y_hold_fired = True
        else:
            self._y_down_s = None
            self._y_hold_fired = False

        edges = QuestButtons(
            a=a,
            b=b,
            x=x,
            y=y,
            a_rising=a and not self._prev_a,
            b_rising=b and not self._prev_b,
            x_rising=x and not self._prev_x,
            y_rising=y and not self._prev_y,
            y_held=y_held,
        )
        self._prev_a = a
        self._prev_b = b
        self._prev_x = x
        self._prev_y = y
        return edges
