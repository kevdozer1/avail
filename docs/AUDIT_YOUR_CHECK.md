# Audit your success check

Level 1: give `avail audit-check` a function that says whether a robot step is done, and it tells you how often
that function says done on a plug that is not, by failure type, with n and a 95% interval.

A success check is whatever decides "done" in your stack: a skill's own flag, a camera classifier, a VLM prompt, a
force threshold. It is usually tested on the cases it was built from. The near-miss zoo tests it on the cases that
fool it: plugs stopped 0.5 to 4 mm short, at the mouth only, unlatched, in the wrong port, jammed at a tilt, or
dropped, each made by the simulated plug station with the truth kept apart. This is the measured version of
EmbodiedSWE's rubric tests. EmbodiedSWE grades coding agents against hidden rubrics and finds agents hacking visible
graders up to 43% of the time. Here the zoo plays the hidden rubric for a robot's success check, and the number
that comes out is a rate with a bound.

## Run it

```bash
uv run avail audit-check --check builtin:skill_flag --connector latch
uv run avail audit-check --check my_checks.py:plug_is_in --connector latch
```

`--check` is `builtin:<name>`, `path/to/file.py:function` or `package.module:function`. The function takes 1
argument, the state at the claim, and returns True for done. It sees what a robot would have, never the truth:

| Field | What it holds |
|---|---|
| `state.frames` | the claim-time RGB images, `{camera: HxWx3 uint8}` (wrist, left, top), loaded when read |
| `state.cameras` | each camera's intrinsics `K` and extrinsics `T` (camera to world) |
| `state.skill` | what the insertion skill reported: `flag` (its own done), `stopped_by`, `travel_mm`, `expected_travel_mm`, `force_at_stop_n` |
| `state.port` | the target port's nominal pose: `x`, `y`, `z` in m and `yaw` in rad |
| `state.features` | AVAIL's own checks on the same state, `{"<source>/<feature>": value}` |

The default split is the zoo's calibration split (2,000 states per connector, of which 1,891 are insertion claims on
`latch`). A test split is refused: it is read once, by its final run. A check that raises counts as not passing.

## Worked example: the skill's own flag and the camera check

```python
# my_checks.py
def skill_says_done(state):
    return state.skill["flag"]

def camera_sees_it_home(state):
    f = state.features
    return f.get("vision.gap@claim/plug_found", 0) >= 0.5 and f.get("vision.gap@claim/gap_mm_est", 99) <= 0.5
```

These 2 are also the builtins `builtin:skill_flag` and `builtin:camera`. On the `latch` zoo, calibration split, sim:

| Bad insertions, by failure type | Skill's own flag passes | Camera check passes |
|---|---|---|
| 0.5 to 1 mm short | 100.0% (145 of 145, 97.4 to 100.0) | 0.0% (0 of 145, 0.0 to 2.6) |
| 1 to 2 mm short | 100.0% (161 of 161, 97.7 to 100.0) | 0.0% (0 of 161, 0.0 to 2.3) |
| 2 to 4 mm short | 95.0% (152 of 160, 90.4 to 97.4) | 0.0% (0 of 160, 0.0 to 2.3) |
| mouth only | 0.0% (0 of 157, 0.0 to 2.4) | 0.0% (0 of 157, 0.0 to 2.4) |
| unlatched | 100.0% (159 of 159, 97.6 to 100.0) | 100.0% (159 of 159, 97.6 to 100.0) |
| wrong port | 100.0% (175 of 175, 97.9 to 100.0) | 0.0% (0 of 175, 0.0 to 2.1) |
| tilted jam | 96.7% (145 of 150, 92.4 to 98.6) | 26.0% (39 of 150, 19.6 to 33.6) |
| dropped | 31.3% (21 of 67, 21.5 to 43.2) | 0.0% (0 of 67, 0.0 to 5.4) |
| **all bad insertions** | **81.6% (958 of 1,174, 79.3 to 83.7)** | **16.9% (198 of 1,174, 14.8 to 19.1)** |
| good insertions | 100.0% (717 of 717, 99.5 to 100.0) | 99.6% (714 of 717, 98.8 to 99.9) |

Read it per row. The skill says done on 4 of every 5 bad plugs: it only knows when it missed the mouth. The camera
is right about short plugs and wrong about every unlatched one, because an unlatched plug looks seated. No single
check covers every row, which is why AVAIL fuses them and adds probes ([RESULTS.md](RESULTS.md)).

## What the numbers are, and are not

- Simulation: the plug station's renders and physics. No real camera or real connector was measured.
- The zoo's mix of failures, set in `config/experiment.yaml`. Another mix gives another overall rate. The per-type
  rows are the ones that carry over.
- A check tuned on these same states will look better here than it is. Tune on `fit`, audit on `calibration`.
