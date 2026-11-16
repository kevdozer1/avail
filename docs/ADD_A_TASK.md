# Add a task

Level 1: a new task is 1 task file, any checks it needs (1 file each), and a testbed module that runs it. The core, which claims, verifies, forks, asks a person, gates and records, does not change.

The worked example is the cable loop of S5: pick a cable from the rack, carry it, insert it into its assigned port, release it, then the next cable, 5 cables per cycle. It was added beside the plug task of S2 to S4 (`tasks/plug_insert.yaml`). The last section lists every place the core changed while it was added.

## What the core is

The core is the product: claim, evidence, calibrated check, fork with handoff, done gate, ledger and exports. It lives in `avail/core/`, `avail/schemas/`, `avail/ledger/`, `avail/stats/`, `avail/witness/`, `avail/fork/`, `avail/flywheel/`, `avail/shift/`, `avail/evidence/base.py` and `avail/evidence/fused.py`. It never imports MuJoCo, the plug station or anything plug-specific. `tests/test_core_boundary.py` checks that on every run of the test suite.

A task meets the core in 3 places:

| Place | What it is | For the cable loop |
|---|---|---|
| Task file | `tasks/<task>.yaml`: the items, the steps, the predicates with what "done" means in plain words, the checks that serve each predicate and how they decide, and the fork's settings | `tasks/cable_loop.yaml` |
| Checks | 1 file per check under `avail/evidence/`, each registering 1 evidence source | `jaw_hold.py`, `hover_pose.py`, `cable_tag.py`, `release_record.py`, plus the plug task's checks it reuses |
| Testbed | the module the task file names under `testbed:`. It builds the world, runs the robot's skills, and holds the truth the evaluator and the simulated person read | `avail/station/mj_plug/cable_shift.py` with `cable_cell.py`, `cable_scene.py`, `oracle_human.py`, `cable_teach.py` |

## Step 1. Write the task file

```yaml
task: cable_loop
title: Pick each cable from the tray, carry it to its assigned port, insert it and let go of it
testbed: avail.station.mj_plug.cable_shift
items: {kind: cable, count: 5}
steps:
  - {name: pick, claim: held}
  - {name: carry, claim: at_port}
  - {name: insert, claim: seated, claim_by_variant: {latch: latched}, judge: true}
  - {name: release, claim: released_stable}
predicates:
  held:
    done_means: The gripper holds the named cable, and no other cable, in its jaws.
    checks: [passive.jaw_hold, vision.cable_tag]
    decide:
      kind: rule
      accept_all: [{feature: passive.jaw_hold/in_band, above: 0.5}, {feature: vision.cable_tag/match, above: 0.5}]
      reject_any: [{feature: passive.jaw_hold/closed_empty, above: 0.5}, {feature: vision.cable_tag/mismatch, above: 0.5}]
  seated:
    done_means: The cable's plug is in its assigned port, its shoulder is flush with the socket face within the depth tolerance, and it sits straight.
    decide: {kind: witness, witness: "{variant}"}
fork:
  max_attempts: 2
  ask: {abstain: confirm, reject_last: choose, at_port_reject: guide_marks}
```

This is the task file as E3 ran it. After E3 (2026-11-13) `held`, `at_port` and `released_stable` were decided by
calibrated witnesses too (`decide: {kind: witness, witness: "{variant}"}`), and on 2026-11-16 `fork` gained a `ladder`
of recovery actions. `tasks/cable_loop.yaml` is the file as it stands now. The rule form above still works for a new
task.

- **Steps** run in order for each item. Each raises 1 claim. A variant (the plug station's connector type) can swap a step's predicate, as `latch` swaps `seated` for `latched`. `judge: true` marks the steps the E3 vlm arm's judge decides. That arm trusts the skill's own flag on the rest.
- **`done_means`** is the definition a person applies when asked, and the one every claim carries. The schemas take predicate definitions from the task files, so a new predicate needs no code.
- **`decide`** is either a rule witness or a calibrated witness. A rule witness accepts when every `accept_all` condition holds, rejects when any `reject_any` condition holds, and abstains otherwise. Its conditions name features as `<check>/<feature>`. A calibrated witness is a cascade fit and calibrated on labeled states (E2), read from the witness store. Through E3 the cable loop used rules for its 3 new predicates, and a cascade fit and calibrated on its own insertions (`cable_zoo.py`) at the customer bar's alpha for `seated` and `latched`. Since E3 the other 3 are calibrated witnesses as well (`check_zoo.py`).
- **`fork`** sets the attempts and what a person is asked: `confirm` after an abstain, `choose` after the last reject, and `guide_marks` after the last `at_port` reject. A key `<predicate>_reject` sets the question for that predicate's last reject.
- **`fork.ladder`** (2026-11-16) lists, per predicate, the testbed's recovery actions to try before the 1st call on a step, each followed by the same witness on a fresh claim. Only the witness can accept after one. `fork.ladder_after` says whether a predicate's ladder follows an abstain, a last reject or both. The cable loop's are `relook` for `held`, `regrip` for `at_port` and `reinsert` for `latched`.

## Step 2. Write the checks it needs, 1 file each

A check is an evidence source: `source_id`, `kind`, the `predicates` it serves, `estimate_cost(claim)` and `collect(station, claim, params)`, which returns `Evidence` with flat numeric features. It sees the robot only through a `StationView` (cameras, robot state, guarded motion) and what the testbed puts in `params`. It never sees truth. `avail/evidence/jaw_hold.py` is the smallest example:

```python
class JawHold:
    source_id = "passive.jaw_hold"
    kind = "passive_proprio"
    predicates = {"held", "at_port", "released_stable"}

    def estimate_cost(self, claim): return Cost.zero()

    def collect(self, station, claim, params):
        state = station.robot_state().arms[params["arm"]]
        ...
        return make_evidence(self, claim, features, score=score)

register(JawHold())
```

The registry looks for a check named `<kind>.<name>` in `avail/evidence/<name>.py`, so a new file is all a new check needs.

The cable loop needed 4:

| Check | Serves | Reads | What it found on dev renders |
|---|---|---|---|
| `passive.jaw_hold` | held, at_port, released_stable | the gripper's own reading against the taught part width | separates a held cable from empty jaws and open jaws |
| `passive.hover_pose` | at_port | the tool pose against the taught pre-insert pose of the named port | finds the wrong port. A cable cocked in the jaws puts the tool 3 to 7 mm off |
| `vision.cable_tag` | held, at_port | the color at the tool point in the wrist camera, as a direction in RGB, against a palette taught on dev seeds | tells the named cable from another and from none. It cannot see a cable cocked in the jaws from this mount, so it reports no angle |
| `passive.release_record` | released_stable | the release skill's own gripper reading before each lift | finds a release that lifted with the jaws shut and took the plug out with it |

The last 2 rows changed after the dev shift dev1, before any shift test seed was read. `vision.cable_tag` compared RGB distances, and a seed's lighting scales a rendered color, so it saw no cable on about 1 pick in 5 and called a person. `passive.release_record` is new: a stuck release lifted plugs out of their ports, and the rule that read only the jaws accepted the retry.

After E3, 3 more joined the witnesses: `passive.hover_offset` for `at_port` (the hover pose against the board's offset as the robot measured it), `passive.release_lift` for `released_stable` (the wrist's force as the jaws opened and as it lifted away) and `probe.release_tug` for `released_stable` (the jaws close on the released plug again and pull it below the latch's hold).

## Step 3. Write the testbed module

The task file names it under `testbed:`. The shift runner imports it by that name and calls:

| Function | Returns |
|---|---|
| `make_cell(seed, variant, profile)` | the cell the cycle driver runs: `run_skill`, `claim_params(item, step, result)`, `view`, `target_of(item)`, `now`, `begin_assist`, `end_assist`, `n_items`, `variant`, and `rung_begin` and `rung_end` for a task file with a `fork.ladder`. A person's marks reach `run_skill` on the next try. The cable cell's taught palette comes from `uv run python -m avail.station.mj_plug.cable_teach` |
| `make_witnesses(task, variant, alpha)` | 1 witness per predicate: rules from the task file, cascades from the witness store |
| `make_human(cell, seed)` | the person behind the fork. In sim, OracleHuman answers from truth and plays the teleop source of a takeover. Its `Handback` says whether the step was done when they took the arm (`before`) and after (`ok`) |
| `make_judge(session=...)` | the vlm arm's judge, charged to the spend session the runner names |
| `evaluate(cell)` | the evaluator's truth for every item at the end of a cycle |
| `frames_table(cell)`, `scene_events(cell)` | the cycle's frame log, every frame tagged `autonomous` or `assisted` |
| `evaluator(cell)`, `close(cell)` | the evaluator's truth per step for the shift report, and the cell's cleanup |
| `setup_files()` | the files besides config/experiment.yaml that set how the cycles go, the witnesses they load and the robot model among them. The E3 command records their hashes and checks them before each arm, and the shift runner before each cycle |
| `station_problem()` | why a final run could not build the world, or None, for files no command here makes, such as a robot model git does not track. The E3 command refuses to start while it returns a reason |
| `judge_problem()` | why the vlm arm's judge could not be called now (no key, no such model), or None. The E3 command asks it when the run holds the vlm arm, and refuses to start while it returns a reason |
| `prepare(task, variant, alpha)`, `prepare_problem(variants, alpha)` | what `avail shift prepare` builds (the witnesses, and for the cable loop the judge's reference picture), and why a final run would find any of it missing or stale, or None. The 2nd reads only, and the E3 command refuses to start while it returns a reason. With no `prepare`, the command builds the witnesses alone |

## Step 4. Run it

The cable loop's witnesses are built from tables under `AVAIL_DATA`, which do not ship. These 6 commands make the
tables on the seeds E3b's witnesses were made from. Together they simulate about 5,000 cycles, which takes hours. With
this snapshot's `check_witness` settings in `config/experiment.yaml`, the witnesses built from them are the ones of
2026-11-16, which differ from E3b's:

```powershell
uv run python -m avail.station.mj_plug.cable_zoo --variant latch --split cable_fit --seeds 24000:25000
uv run python -m avail.station.mj_plug.cable_zoo --variant latch --split cable_calibration --seeds 26000:27000
uv run python -m avail.station.mj_plug.check_zoo --variant latch --split check_fit
uv run python -m avail.station.mj_plug.check_zoo --variant latch --split check_calibration
uv run python -m avail.station.mj_plug.check_zoo --variant latch --split check_fit_drag
uv run python -m avail.station.mj_plug.check_zoo --variant latch --split check_calibration_drag
```

Then:

```powershell
uv run avail shift prepare --task cable_loop --variant latch
uv run avail shift run --task cable_loop --variant latch --arm avail_simop --seeds 20200:20300 --shift dev3 --workers 2
uv run avail shift report --shift dev3
uv run avail shift export --shift dev3
```

Use a new shift name and new dev seeds for each run. dev1's ledgers predate the fix to takeover labels, so do not export dev1. `report` writes the shift card from the ledgers alone. `export` writes every takeover as an intervention episode (LeRobot-style parquet with each frame tagged, and the handoff and handback marked) and every intervention's calibration label.

## Every place the core changed for the cable loop

The core was separated from the plug station on the night the cable loop was added (2026-11-12), and the cycle driver, the fork and the done gate were built that night too. So the changes fall in 3 groups. The goal for the 3rd group was none. It came out at 15.

**Separating the core.** Any 2nd task would have needed these.

| Change | Where |
|---|---|
| The robot types (Pose, Station, RobotState and the rest) moved out of the plug station, which still offers them | `avail/core/robot.py` |
| Tasks are files: items, steps, predicates with what done means, the checks, how each decides, the fork's settings | `avail/core/task.py`, `tasks/` |
| A claim carries its predicate's definition from the task file | `avail/schemas/` |
| The cascade takes its stages, the features it skips and the station id as arguments. The plug station's stages moved to the zoo, and the zoo's figures to the analysis | `avail/witness/cascade.py`, `avail/zoo/stages.py`, `avail/analysis/figures.py` |
| A check named `<kind>.<name>` is found in its own file | `avail/evidence/base.py` |
| A scan fails the test suite when the core imports the simulator or the testbed, or holds a plug word | `tests/test_core_boundary.py` |

**What any task with steps needs.** Built for the cable loop, with no word of it in them.

| Piece | Where |
|---|---|
| Rule witnesses, read from the task file | `avail/witness/rules.py` |
| The fork: retry, ask a person (confirm, choose, marks) or hand control to a teleop source, and back | `avail/fork/policy.py`, `human.py`, `handoff.py` |
| The done gate: an item is done when each step has an accepting verdict or a person's record | `avail/fork/gate.py` |
| The intervention record (a calibration label, a correction demo, a shift metric) and its export | `avail/flywheel/` |
| The cycle driver, the shift runner, the shift card, the tool server's core side | `avail/shift/` |

**What the cable loop then needed in the core.**

| # | Change | Why |
|---|---|---|
| 1 | The task file names its testbed module (`testbed:`). The shift runner imports it by that name | the core runs a testbed it never names |
| 2 | The done gate takes a takeover's record, with the person's label after the handback | the gate refused cables a person had seated |
| 3 | The intervention record is written when the intervention closes | its outcome is not known before, and the record has no open state |
| 4 | A 2nd question to a person on the same step becomes a takeover | a person who kept answering retry could loop |
| 5 | The `cycle_start` event lists the cycle's items | the done gate rebuilds a cycle from the ledger |
| 6 | The tool server's core side asks the cell which probes it has. `avail shift` lost its default variant | both held plug words, which the boundary scan caught |
| 7 | The cycle driver named 2 of the loop's steps: it sent the vlm arm's judge to `insert`, and gave the `carry` step's last reject its own question. Both are task file entries now: a step's `judge: true`, and `fork.ask`'s `<predicate>_reject`, which the fork policy already read. The claim's target and the skill's record went the same way: the cell gives `target_of(item)`, and `claim_params` takes the whole skill result | the plug-word scan could not see step names. A test now fails when a string in the core is a step, predicate or item kind of a task file |
| 8 | The shift card's text says items in place of cables, and it no longer states 0 safety events | it named the task, and nothing measured the safety line |
| 9 | A takeover's intervention record is labeled with what the person found when they took the arm. The sign-off after the handback stays on the Assist record, which the done gate reads | the label sat on the evidence of the state before the fix and said done, so each takeover that ended done taught calibration that a failed step was good |
| 10 | A takeover right after the person's own answer is the same call for help, with no 2nd wait | a choose answered with takeover counted as 2 calls and 2 waits |
| 11 | A cycle cut off by a stopped run reruns under a new id (`_r1`) | its records mixed with the rerun's |
| 12 | `avail shift export` writes every run of a shift into 1 tree, with its episodes numbered on, and refuses a folder with no ledger. Its frame rate is the mean frame gap | it read the shift folder as 1 run and exported nothing. A log on a fixed grid of ticks alternates 2 gaps, so the median gap overstated the rate |
| 13 | The E3 command plans the judge arm's cost, refuses before any seed is read when the variant's cap cannot cover it, warns while any variant's cap is short (this one's too when `--arm` leaves its judge arm out), says the caps are frozen once E3 has read a seed, refuses a variant outside `e3.variants`, a config other than the 1 E3 read its split under, and a start while the testbed's `station_problem`, `judge_problem` or `prepare_problem` finds a robot model missing, a judge it cannot call, or a witness or reference missing or stale, and runs the judge arm last. `avail shift prepare` builds what the testbed's `prepare` lists. Each variant's arm has its own spend session. The command records the hashes of the task file, the config, the testbed's setup files and the code at its 1st run (`avail/shift/setup_record.py`), checks them before each arm, and refuses an arm that finds 1 changed. The shift runner checks a final run's record before each cycle and before the cycle's end is written, refuses a final run that passes no record, and `avail shift run` refuses `--final-run E3` | a cap raised after E3's 1st read comes too late, a shared session would leave the 2nd variant's arm without money, the seed guard hashes only config/experiment.yaml, a file or the code edited while 1 arm ran would reach that arm's later cycles, and a final run builds no witness and no reference, so 1 made after the 1st arm would change the recorded setup |
| 14 | The shift runner hands a final run's judge that run's own spend session (`make_judge(session=...)`) | the judge's calls are charged to the run that makes them |
| 15 | A claim gets 1 calibration label: a takeover right after the person's own labeled answer on it adds none. An Assist record lists the answers the person was offered | a confirm answered no on the last try and the takeover after it labeled the same evidence twice, and the Assist listed the fork's menu |

Changes 2 to 5 were bugs in new core code that the cable loop ran 1st. Changes 9 to 15 came from 2 reviews of the core and the testbed, and none names the task. Later checks of the reports and the E3 command added to change 13. Changes 5 to 15 were made after the dev shift dev1 started, which ran from a frozen copy of the code. dev1's ledgers list a cycle's items as a count, so the done gate cannot rebuild a dev1 cycle from them. After change 7 the new driver gave the same cycles on dev seeds 20,003 and 20,004 in the avail arm: the same items done, the same forks, and the same 102.42 and 84.71 s of motion.

**After E3 (2026-11-13 to 2026-11-16).** 4 more changes, and none names the task: a stage whose 2 thresholds cross
may accept above its accept threshold (`crossing` in `avail/stats/conformal.py` and `avail/witness/cascade.py`), the
task file's `fork.ladder` and the cycle driver that runs it (`avail/core/task.py`, `avail/shift/driver.py`), a
bootstrap bound on a mean for E4's calls for help (`avail/stats/bootstrap.py`), and takeovers written as correction
demos in documented JSONL (`avail/flywheel/corrections.py`).

**Still in the core with the plug station's shape.**

- The core calls a variant `connector`: `Cascade.connector`, `WitnessScope.connector`, and the witness store's folders `<connector>/<predicate>`. For the cable loop the variant is a connector type, so it fits. A task with another kind of variant would store it under that name.
- `BUILTIN_MODULES` in `avail/evidence/base.py` names the plug station's 5 check modules, which load whenever a check is looked up.
- `avail/evidence/base.py` documents the plug station's params (`nominal`, `connector`, `insert`) as what a check may find.
