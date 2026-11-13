"""instr.link: the link signal an instrumented station reports. It costs nothing and moves nothing.

Some stations can simply be asked. A switch reports that a port has a link, a charger reports
that it sees a load. This source reads that signal through ``station.read_signal('link')``. It
exists to show what an instrumented station gets for free, next to the sources that have to
work for the same answer.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Claim, Cost, Evidence
from ..station.base import Station
from .base import make_evidence, register

LINK_SIGNAL = "link"


class Link:
    """Reads the station's link signal and returns ``link_up``.

    Feature: ``link_up``, 1.0 when the station reports the link up and 0.0 when it reports it down.

    Score: ``link_up``.

    What the link says. The contacts mate: the plug is at full depth in the port the station
    watches. It says nothing about a latch. A plug that is fully in with its latch blocked
    reads link up, so for ``latched`` a link that is down rules the claim out and a link that is
    up does not settle it.

    The signal lags. In the simulated station the link comes up 0.2 s after the plug reaches
    depth, and a real switch takes longer. This source reads once and does not wait, so a read
    taken right after a probe has seated the plug can still say down. The caller should let the
    station wait before the read.

    A station without the signal raises KeyError from ``read_signal``. That becomes an evidence
    with no features and score None, as does a signal that carries no true or false ``link_up``.
    """

    source_id = "instr.link"
    kind = "instrumented"
    predicates = {"seated", "latched", "at_port", "released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        started = station.robot_state().ts
        reader = getattr(station, "read_signal", None)
        try:
            signal = reader(LINK_SIGNAL) if reader is not None else None
        except KeyError:                                # the station is not instrumented
            signal = None
        seconds = station.robot_state().ts - started
        link_up = signal.get("link_up") if isinstance(signal, dict) else None
        if not isinstance(link_up, (bool, np.bool_)):
            return make_evidence(self, claim, {}, seconds=seconds)
        value = 1.0 if link_up else 0.0
        score = value if claim.predicate.name in self.predicates else None
        return make_evidence(self, claim, {"link_up": value}, score=score, seconds=seconds)


LINK = register(Link())
