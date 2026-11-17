# Release check

Level 1: on 2026-11-16 the scan found no key, no private path, no text from the private notes and no private name in
243 text files, and on a fresh clone the 1-command demo ran with its setup in 8 min 16 s on 1 worker and 3,082 tests
passed.

## The scan (2026-11-16)

Run from the private working repo, because it reads the private notes it compares against:

```
uv run python scripts/release_scan.py <this folder>
```

| Check | What it looks for | Hits |
|---|---|---|
| Keys and tokens | 9 key shapes (OpenRouter, sk-, Anthropic, GitHub, Hugging Face, AWS, Google, private key blocks, Bearer values) and the exact value of the 1 key in the private `.env` | 0 |
| Paths | any path on the data drive and any path into a Windows, Linux or macOS user's home folder | 0 |
| Private notes | any run of 8 words shared with the private notes folder (23,618 runs) | 0 |
| Private names | the 9 names on the private list, case-insensitive | 0 |
| By hand | every line with `Jacob` that is not `Jacobian`, and every `Trevor` | 0 lines |

The scan before the commit of the new code read 243 text files and found 0 hits. The rescan after the fresh-clone
check, with this file and the docs in their final form, read the same 243 files and found 0 hits. `.git`, `.venv`,
`third_party` and `avail_data` were skipped. The 1 line the scan lists for a hand check is the table row above, which
names no person. A case-insensitive search for both names found that same line and no other.

## What changed since 2026-11-13

- The code in `avail/`, `config/`, `tasks/` and `tests/` is the working repo's as of 2026-11-16, copied file by file:
  27 files changed and 26 added. No file had to be dropped. New since the last snapshot: the calibrated witnesses
  for `held`, `at_port` and `released_stable` that E3b ran, the release pull (`probe.release_tug`), the recovery
  ladder, the error budget, and E4's plan and statistics. `pyproject.toml`, `uv.lock` and the setup script's pin
  did not change.
- `docs/img/frontier.png` and `frontier.json` are new. `fig.png` and `fig.svg` were redrawn with the simulated
  operator's label.
- Edits that exist only here: the README and the docs pages, `.env.example`, `.gitignore`, `tests/test_writing.py`,
  `avail/demo.py` (its pointers to `docs/RESULTS.md`, and since today it runs `scripts/setup_aspire.py` before it
  simulates anything, with `--no-setup` to skip that), and `scripts/setup_aspire.py`, which now prints its
  "Next" line only when it runs as a script.

## The E4 update (2026-11-16, evening)

- E4 ran on this code and passed both of its hypotheses. The README, `docs/RESULTS.md`, `docs/img/frontier.png`
  and `docs/img/frontier.json` now report it, and the opening line of `docs/HOW_IT_WORKS.md` now says where a retry
  and a recovery action come before a person.
- `avail/analysis/frontier.py` is the working repo's, which draws each flywheel day's interval and lets the y axis
  reach E4's interval. No other code changed, so the cable loop's code is the code E4 ran.
- The rescan after these edits read 243 text files and found 0 hits.

## What was left out

The private notes, the session briefs, the plan, the agent instructions, reports and drafts, archived worktrees, all
data, the internal design docs (`docs/01` to `docs/13`), the README draft they describe, and the internal scripts
(the release scan, the judge and catalogue scripts, and E4's launcher). Also left out: the figure run's cached
example and summary (`docs/img/fig.example.json` holds a data-drive path), the showcase stills and the report
figures. Code and config comments still name those design docs in 100 places, the session briefs in 9, the reports
in 10 and the agent instructions in 28. They are pointers. 1 function reads a file that was left out:
`avail/analysis/film_record.py` takes E3's task file from an archived worktree, so the showcase film cannot be
recorded from this snapshot.

## To check before pushing

- Citations: the README credits the GR00T 2 whitepaper, FailBench, EmbodiedSWE, ASPIRE, RPG, and Walden and Russ
  Tedrake's CoRL 2026 keynote by name only. Add a link or paper ID for each. The 43% hack rate is EmbodiedSWE's
  figure as recalled in the brief, so confirm it against the paper.
- `Kevin` appears 54 times in code comments and config, as the person whose decisions set the experiments.
- E4 ran on 2026-11-16, after the scan above. The README and the figure report it (see "The E4 update" above).

## The fresh clone (2026-11-16, 04:44 to 05:01)

`git clone` of this folder at commit `1149f1a` into a new empty folder on the data drive, with `AVAIL_DATA` set to a
new empty folder beside it, `PYTHONPATH` and the key empty, and `UV_LINK_MODE=copy`. The clone had no `.venv`,
`third_party`, `.env` or data.

| Step | Command | Result | Time |
|---|---|---|---|
| Clone | `git clone <this folder> avail` | exit 0 | 5.5 s |
| Demo, setup included | `uv run avail demo --workers 1 --no-open` | exit 0. uv made `.venv` and installed 55 packages in 1 min 14 s. The demo's setup cloned ASPIRE at `fa00f0e` and wrote `.env` in about 10 s. Then 36 states per mode in 6 min 43 s: 0 of 24 bad plugs accepted in each mode, 0 claims sent to a person, report written | 8 min 16 s (495.9 s) |
| Tests | `uv run pytest -q` with the 3 tests below deselected | exit 0: 3,082 passed, 8 skipped, 3 deselected | 5 min 16 s (316.2 s) |

- The demo ran on 1 worker because 12 other simulation processes were running on this machine and memory was
  tight. The default is 4 workers, and with 4 the demo itself took 81 s on 2026-11-13.
- The 3 deselected tests are slow tests that start several simulation processes at once:
  `tests/test_station.py::test_the_insertion_record_does_not_carry_the_shortfall` (8),
  `tests/test_station.py::test_a_tilted_jam_is_cocked_in_the_fingers_with_the_tool_straight` (6) and
  `tests/test_zoo.py::test_generate_writes_states_and_frames_and_resumes` (2). The other 8 slow tests ran.
- The 8 skips: the 4 tests of `tests/test_audit.py` need the zoo's calibration tables and 2 tests of
  `tests/test_cable_loop.py` need the cable loop's tables, both under `AVAIL_DATA`. `tests/test_writing.py` found no
  session report to check and no private notes folder to read its name list from.
- The data drive is exFAT, which records no file owner, so git refuses to work in a repo there. The check gave its
  own processes `safe.directory=*` through `GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_0` and `GIT_CONFIG_VALUE_0`, which
  changes no config file. The setup's clone of ASPIRE needs it on such a drive. A clone on NTFS, ext4 or APFS does
  not.
- After the clone only `README.md`, `RELEASE_CHECK.md`, `docs/ADD_A_TASK.md`, `docs/RESULTS.md` and
  `docs/STATISTICS.md` changed, in the next commit. No code changed.

On 2026-11-13 the same check from a copied folder passed: setup 10.6 s, 3,067 tests passed and 8 skipped in 4 min
15 s, and the demo in 81 s on 4 workers.
