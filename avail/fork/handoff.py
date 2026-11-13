"""Takeover: a person drives the arm, and the frame log marks which frames were theirs.

A :class:`TeleopSource` is whatever lets a person drive: a leader arm, a spacemouse, or the sim's
scripted hands. ``take_over(cell, request)`` runs until the person hands control back and returns a
:class:`Handback`.

:class:`HandoffLog` is the bookkeeping the cell's frame log reads. ``begin`` marks the first frame
under the person's control and ``end`` the last, so both ends of a takeover are assisted frames.
``tag(frame_index)`` says ``autonomous`` or ``assisted`` for any frame. Frame indices are those of
the cycle's frame log, so a new cycle starts a new log, or calls ``reset``::

    log = HandoffLog()
    log.begin(412, t)                      # the person has the arm from frame 412
    ...                                    # frames 412 on are tagged assisted while it is open
    span = log.end(1172, t)                # and gives it back after frame 1172
    Intervention(..., **span.fields())     # handoff_ts, handback_ts, handoff_frame, handback_frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..schemas.kinds import FRAME_TAGS

if TYPE_CHECKING:
    from .human import HumanRequest

__all__ = ["ASSISTED", "AUTONOMOUS", "HandoffLog", "Handback", "Takeover", "TeleopSource"]

AUTONOMOUS, ASSISTED = FRAME_TAGS


@dataclass(frozen=True)
class Handback:
    """How a takeover ended: whether the person finished the step, the seconds they had control, a note.

    ``before`` is the person's look at the step when they took the arm, before they changed anything:
    True when it was done already, False when it was not, None when they did not say. It is the label
    of the claim the robot raised, whose evidence describes that state. ``ok`` describes the state
    after their fix, which no evidence of the claim saw."""

    ok: bool
    seconds: float
    note: str = ""
    before: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise ValueError(f"ok is {self.ok!r}. Fix: say with True or False whether the person finished the step")
        if self.before is not None and not isinstance(self.before, bool):
            raise ValueError(f"before is {self.before!r}. Fix: say with True, False or None whether the step was "
                             "done when the person took over")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or \
                not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError(f"seconds is {self.seconds!r}. Fix: give a finite number of seconds, 0 or more")
        object.__setattr__(self, "seconds", float(self.seconds))
        if not isinstance(self.note, str):
            raise TypeError(f"note is {self.note!r}. Fix: give text")


@runtime_checkable
class TeleopSource(Protocol):
    """A way for a person to drive the cell until they hand it back."""

    def take_over(self, cell: Any, request: HumanRequest) -> Handback:
        """Give the person control for ``request``, wait until they hand back, and say how it ended."""
        ...


def _frame(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} is {value!r}. Fix: give the frame's index in the cycle's frame log, 0 or more")
    return value


def _time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} is {value!r}. Fix: give a finite time in seconds")
    return float(value)


@dataclass(frozen=True)
class Takeover:
    """1 stretch of frames under a person's control. Both ends are assisted frames. Open while it has no handback."""

    handoff_frame: int
    handoff_ts: float
    handback_frame: int | None = None
    handback_ts: float | None = None

    @property
    def is_open(self) -> bool:
        return self.handback_frame is None

    def covers(self, frame_index: int) -> bool:
        """True for a frame under the person's control. An open takeover covers every frame from its handoff on."""
        if frame_index < self.handoff_frame:
            return False
        return self.handback_frame is None or frame_index <= self.handback_frame

    def fields(self) -> dict[str, Any]:
        """The 4 takeover fields of an ``Intervention`` record."""
        return {"handoff_ts": self.handoff_ts, "handback_ts": self.handback_ts,
                "handoff_frame": self.handoff_frame, "handback_frame": self.handback_frame}


class HandoffLog:
    """Which frames of 1 cycle's frame log were under a person's control."""

    def __init__(self) -> None:
        self._closed: list[Takeover] = []
        self._open: Takeover | None = None

    @property
    def active(self) -> bool:
        """True while a person has control."""
        return self._open is not None

    @property
    def takeovers(self) -> list[Takeover]:
        """Every takeover of the cycle in order, the open one last."""
        return [*self._closed, *([self._open] if self._open is not None else [])]

    def last(self) -> Takeover | None:
        """The latest takeover, open or closed, or None."""
        found = self.takeovers
        return found[-1] if found else None

    def begin(self, frame_index: int, ts: float) -> Takeover:
        """The person takes control: ``frame_index`` is the first frame they drive."""
        frame_index, ts = _frame(frame_index, "frame_index"), _time(ts, "ts")
        if self._open is not None:
            raise RuntimeError(f"a takeover is already open since frame {self._open.handoff_frame}. Fix: end it first")
        if self._closed and frame_index <= self._closed[-1].handback_frame:
            raise ValueError(f"frame {frame_index} is not after the last handback, frame "
                             f"{self._closed[-1].handback_frame}. Fix: begin at a later frame, or reset for a new cycle")
        self._open = Takeover(frame_index, ts)
        return self._open

    def end(self, frame_index: int, ts: float) -> Takeover:
        """The person hands control back: ``frame_index`` is the last frame they drove."""
        frame_index, ts = _frame(frame_index, "frame_index"), _time(ts, "ts")
        if self._open is None:
            raise RuntimeError("no takeover is open. Fix: call begin when the person takes control")
        if frame_index < self._open.handoff_frame:
            raise ValueError(f"the handback frame {frame_index} comes before the handoff frame "
                             f"{self._open.handoff_frame}. Fix: give the last frame the person drove")
        if ts < self._open.handoff_ts:
            raise ValueError(f"the handback time {ts:.3f} comes before the handoff time {self._open.handoff_ts:.3f}. "
                             "Fix: take both times from the same clock")
        span = Takeover(self._open.handoff_frame, self._open.handoff_ts, frame_index, ts)
        self._closed.append(span)
        self._open = None
        return span

    def tag(self, frame_index: int) -> str:
        """``assisted`` for a frame under a person's control, else ``autonomous``."""
        frame_index = _frame(frame_index, "frame_index")
        return ASSISTED if any(span.covers(frame_index) for span in self.takeovers) else AUTONOMOUS

    def reset(self) -> None:
        """Forget every takeover, for the frame log of a new cycle. Refused while a person has control."""
        if self._open is not None:
            raise RuntimeError(f"a takeover is open since frame {self._open.handoff_frame}. Fix: end it first")
        self._closed.clear()
