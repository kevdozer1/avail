"""An operator who is not perfect (the night of 2026-11-16, step 4). Description only, dev seeds only. Privileged.

``OracleHuman`` answers from truth and is wrong on 2% of confirms (``human.human_error``), with a log-normal answer delay
of median 20 s. This operator is the same person with 2 things changed:

- **wrong** on a yes or no answer with probability ``error``: it says yes to a bad claim or no to a good one, on a
  confirm and on the sign-off after a takeover. Its own draws, from its own stream, decide it. ``OracleHuman``'s own
  error on a confirm is set aside.
- **slower to answer** on a long-tailed delay: log-normal with median ``median_s`` and 95th percentile ``p95_s``
  (Kevin's brief: 8 s and 40 s, so a log spread of ln(5) / 1.645 = 0.978).

A run asks for it through the environment of a dev run: ``AVAIL_SIMOP_OPERATOR="error,median_s,p95_s"`` (for example
``0.05,8,40``). ``cable_shift.make_human`` reads that variable. A final run never sets it, and refuses it.
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

from .oracle_human import OracleHuman

ENV = "AVAIL_SIMOP_OPERATOR"
Z95 = 1.6448536269514722


def from_env() -> tuple[float, float, float] | None:
    text = os.environ.get(ENV, "").strip()
    if not text:
        return None
    error, median_s, p95_s = (float(v) for v in text.split(","))
    if not (0.0 <= error < 0.5 and 0.0 < median_s < p95_s):
        raise ValueError(f"{ENV}={text!r}: give error in [0, 0.5), and a median under the 95th percentile")
    return error, median_s, p95_s


class ImperfectHuman(OracleHuman):
    assistant_id = "imperfect_human"

    def __init__(self, cell, seed: int, error: float, median_s: float, p95_s: float):
        super().__init__(cell, seed)
        self.error, self.median_s = float(error), float(median_s)
        self.sigma = math.log(float(p95_s) / float(median_s)) / Z95
        self.own = np.random.default_rng([int(seed), 9393])

    def ask(self, request):
        answer = super().ask(request)
        delay = self.median_s * math.exp(self.sigma * float(self.own.normal()))
        changes = {"seconds_to_answer": delay}
        ctx = request.context
        if request.question == "confirm":
            done = bool(self.cell.truth_of(int(ctx["item"]), str(ctx["predicate"])))
            if self.own.random() < self.error:
                done = not done
            changes.update(answer="yes" if done else "no", truth_label=done)
        elif request.question == "takeover" and answer.truth_label is not None:
            ok = bool(answer.truth_label)
            if self.own.random() < self.error:
                ok = not ok
            changes["truth_label"] = ok
        return dataclasses.replace(answer, **changes)
