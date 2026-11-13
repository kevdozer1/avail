# The statistics in brief

Level 1: every threshold comes with a bound on how often it accepts a bad state, every reported rate comes with its n
and a 95% interval, and test seeds are read once.

## A threshold with a false-accept bound

Score `n` calibration states where the predicate is false and sort the scores. At level `alpha`, the accept threshold
is the score of rank `k = ceil((n + 1) * (1 - alpha))`. A new bad state drawn like the calibration ones clears it with
chance at most `alpha`. This is class-conditional split conformal calibration. A stage needs `n >= 1/alpha - 1` bad
states before it may accept at all: 49 at `alpha = 0.02`. A check cannot be calibrated without failure examples.

## A cascade

Stages run in order of cost, each with its own accept and reject threshold. By the union bound the cascade's
false-accept rate is at most the sum of the stages' alphas. A push or tug can repair a bad plug, which shrinks the
pool of bad states between stages, so the budget is shared in proportion to how many states are still bad at each
stage, with the shares taken from the fit split (`avail/witness/calibrate.py`).

## A certificate

The bound above is an average over calibration draws. A certificate is about this witness: on states it never saw,
count false accepts `x` in `m` bad states and report the one-sided 95% Clopper-Pearson upper bound. 0 in 1,124 gives
0.27%. Reports give both alpha and the certified bound with its `m`.

## A shift

A customer's bar is a defect rate per cycle: a cycle declared done with a bad item in it. To certify under 0.5% per
cycle at 95% takes 598 clean cycles, or at most 16 defects in 5,000 (`avail trials` and `avail/stats/planning.py`).
Calls for help and time per cycle are reported next to it, because a gate that asks about everything is safe and
useless.

## Intervals and seeds

- Rates: Wilson 95% intervals. Bounds on rare events: one-sided Clopper-Pearson.
- Splits: `fit` to choose features and stage order, `calibration` to set thresholds, `test` to report. Test ranges
  are fixed in `config/experiment.yaml`, and `avail/seeds.py` refuses a second read of any of them.
- Drift on a running shift: a 3-sigma p chart and a CUSUM on windowed rates (`avail/stats/spc.py`).
- Known-answer tests in `tests/test_stats.py` reproduce each number on this page.
