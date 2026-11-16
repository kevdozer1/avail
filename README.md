# AVAIL

![Calls for help per 100 cycles against cycles with a bad cable shipped, in simulation with a simulated operator: the skills alone shipped a bad cable in 662 of 1,000 cycles at 0 calls, E3 in 31 of 5,000 at 119 calls, E3b in 2 of 5,000 at 224 calls, and the code here in 0 of 500 dev cycles at 57 calls, against a 0.5% bar. A small panel shows calls per 100 cycles over 10 simulated days of recalibration](docs/img/frontier.png)

AVAIL is a verification and ask-for-help layer for agentic robot stacks. A step counts as done only when calibrated
evidence or a person says so. Each check's false-accept rate is bounded by class-conditional conformal calibration.
Uncertain or failed steps go to recovery actions and then to an operator. Every answer is recorded for recalibration.

Everything in this repository is simulation, and the operator is simulated too: it answers from simulator truth and
takes over with a scripted correction. In the figure a simulated dual-arm YAM picks 5 cables per cycle and plugs each
into its assigned `latch` port. A cycle ships a bad cable when it ends declared done while a cable is not seated,
latched and let go in its assigned port. Each point is 1 run with 95% intervals: 3 on test seeds read once, and the
point marked dev on 500 dev cycles of the code here, drawn at its one-sided bound since it had 0 bad cycles. The
dashed line is the bar set before the runs: under 0.5% of cycles at 95% confidence, which allows at most 16 bad cycles
in 5,000. The small panel is 10 simulated days of 200 dev cycles, each day's thresholds recalibrated on the operator's
answers from the days before.

| Sim, `latch`, test seeds read once | Cycles | Cycles with a bad cable shipped | Calls for help per 100 cycles | Time per cycle |
|---|---|---|---|---|
| The skills' own done flags, on E3b's seeds | 1,000 | 662 (66.2%, 63.2 to 69.1) | 0 | 87 s |
| E3: AVAIL with the simulated operator | 5,000 | 31 (0.62%, 0.44 to 0.88), bound 0.84% | 118.7 | 157 s |
| E3b: the same after its checks were fixed | 5,000 | 2 (0.04%, 0.01 to 0.15), bound 0.13% | 223.6 | 219 s |

Rates carry a Wilson 95% interval, and bounds are one-sided 95% Clopper-Pearson. E3 missed the bar. Every bad cable
it shipped passed at the release step, through a hand-set rule with no calibrated bound. Before E3b the 3 hand-set
checks became calibrated witnesses, and the release check learned to close the jaws on the released plug and pull it
lightly, which a latched plug holds. E3b met the bar on new seeds, at 223.6 calls for help per 100 cycles and 2.53
times the skills' cycle time. The 2 runs are not 2 tests of 1 system, and E3's failure stays on record. Both ran
before the recovery ladder in this code existed. In them the only recovery was 1 retry of a rejected step, and an
uncertain step went straight to the operator. [RESULTS.md](docs/RESULTS.md) has every table.

## Run the demo: 1 command

Needs [uv](https://docs.astral.sh/uv/) and git. No GPU, no API key.

```bash
git clone <this repo> avail
cd avail
uv run avail demo
```

On a fresh clone that command does the setup too. uv installs the locked dependencies, and the demo clones ASPIRE at
its pinned commit into `third_party/ASPIRE` for the dual-YAM station model (`scripts/setup_aspire.py`). Then it
simulates 36 plug states for the `latch` connector in easy mode and 36 in hard mode, runs every free check and the
calibrated cascade that ships in `avail/witness/shipped` on each, prints a small table per mode and opens an HTML
report, `avail_data/runs/demo/report.html`. On a fresh clone on 2026-11-16 the command took 8 min 16 s in all, on 1
worker: 1 min 14 s for uv to install 55 packages, about 10 s to clone ASPIRE, and 6 min 43 s for the demo itself.
The default is 4 workers. [RELEASE_CHECK.md](RELEASE_CHECK.md) has the commands and times.

The demo shows the insertion check, the step that decides whether 1 plug is in, on a station where the truth is known.
It does not run the cable loop of the figure. That loop's witnesses are fit and calibrated on tables from about 5,000
simulated cycles, which do not ship and take hours to make. [Add a task](docs/ADD_A_TASK.md) lists the commands.

`uv run pytest -q` runs the test suite, 3,093 tests in about 5 minutes. `-m "not slow"` leaves out 11 slow
tests, 3 of which start up to 8 simulation processes at once. Tests that need the zoo's or the cable loop's tables
skip and say so.

## Why

In simulation, agentic robotics systems grade themselves with privileged truth: ASPIRE keeps the programs its
benchmark scores as successes, RPG practices against simulator checkers, and EmbodiedSWE grades against hidden rubrics
and trains only on verified outcomes. On a real robot that grader is gone. AVAIL is a grader you can take to the real
robot: physical checks with an error rate measured against sim truth, a person when it cannot tell, and every
intervention logged as labeled data, all outside the agent.

## Use it

- [Use it with ASPIRE](docs/USE_WITH_ASPIRE.md): an MCP proxy between ASPIRE's agent and its robot tools that passes
  every tool through and adds `avail_verify`, `avail_probe`, `avail_ask_human` and `avail_done`.
- [Audit your success check](docs/AUDIT_YOUR_CHECK.md): give it the function that says "done" in your stack and get
  its pass rate on bad states, by failure type, with n and intervals.
- [Add a task](docs/ADD_A_TASK.md): 1 task file, 1 file per check, and a testbed module. The core does not change.
- [How it works](docs/HOW_IT_WORKS.md) and [the statistics in brief](docs/STATISTICS.md).

## More results

All rows are simulation, on test seeds read once. [RESULTS.md](docs/RESULTS.md) has every table and
[STATISTICS.md](docs/STATISTICS.md) the method.

| Measurement | n | Result |
|---|---|---|
| Sim: the skill's own done flag passes bad `latch` plugs | 1,170 | 80.1% (77.7 to 82.3) |
| Sim: the wrist camera check passes bad `latch` plugs | 1,170 | 15.1% (13.2 to 17.3) |
| Sim: a 3 mm tug passes bad `latch` near misses, hard mode | 200 | 2.0% (0.8 to 5.0) |
| Sim: a cheap VLM judge passes the same near misses | 200 | 17.5% (12.9 to 23.4) |
| Sim: AVAIL's cascade at alpha 2% accepts bad `latch` plugs, hard mode v2 | 1,111 | 0.5%, bound 1.06% |
| Sim: the same cascade sends `latch` claims to a person | 1,886 | 2.6% (2.0 to 3.4) |

![1 unlatched plug and what each grader said about it, then 100 cycles of the 5-cable loop under 4 graders](docs/img/fig.png)

The figure run: 100 cycles per grader on the same 100 new seeds, read once. The skill's own flag declared a bad cable
done in 64. A cheap VLM judge looking after every step held back all 100, because it said no to almost everything.
AVAIL alone declared none bad and finished 43, and AVAIL with the simulated operator finished 97 at 134 calls for help
per 100 cycles.

## What this does not show

- No real hardware yet. Every number is simulation. A real camera, connector and force estimate will be noisier.
- The operator is simulated. It answers from simulator truth, is wrong on 2% of confirms by design, and takes over
  with a scripted correction.
- E3b met the bar at a high cost: 223.6 calls for help per 100 cycles and 2.2 robots per operator. A run that asks
  for the same bar at half the calls, E4, is written down in `config/experiment.yaml` and has not run.
- The code here is 2 days newer than E3b. It adds the recovery ladder (a 2nd look, a regrip or a reinsert before a
  call), per-check alphas from an error budget and a rule for crossed thresholds, all chosen on dev and calibration
  seeds. None of them has a test-seed result. A 1st gate on 500 dev cycles, which allows at most 1 bad cycle, failed
  with 2 in its 1st 154, both after a recovery action that was then dropped. Without it, 500 new dev cycles had 0 bad
  cycles at 56.8 calls for help per 100 cycles and 4.9 robots per operator. 0 in 500 bounds the rate at 0.60%, so
  that is a dev number and cannot meet the bar alone.
- ASPIRE's real server was not run. The proxy was tested against a mock built from ASPIRE's own tool definitions.
- The closed-loop judge is a cheap model. A stronger judge may pass fewer bad plugs and was not run in closed loop.
- The bounds hold for states like these: this station and this mix of failures.

## Credits

- The GR00T 2 whitepaper, for naming verification as an open problem.
- FailBench, for measuring it.
- EmbodiedSWE, for the finding that coding agents hack visible graders up to 43% of the time, and for rubric tests.
- Walden and Russ Tedrake's CoRL 2026 keynote, for the loop and the customer's bar.
- ASPIRE by NVIDIA, for the dual-YAM station model, read from a clone at setup time. See [NOTICE](NOTICE).

## License

Apache-2.0. See [LICENSE](LICENSE).
