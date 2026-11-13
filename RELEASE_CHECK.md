# Release check

Level 1: the scan found no key, no private path, no text from the private notes and no private name, and the test
suite passes from this folder.

## The scan (2026-11-13, 02:55)

Run from the private working repo, because it reads the private notes it compares against:

```
uv run python scripts/release_scan.py <this folder>
```

| Check | What it looks for | Hits |
|---|---|---|
| Keys and tokens | 9 key shapes (OpenRouter, sk-, Anthropic, GitHub, Hugging Face, AWS, Google, private key blocks, Bearer values) and the exact value of the 1 key in the private `.env` | 0 |
| Paths | any `D:` path and any user-home path (`C:\Users\...`, `/home/...`, `/Users/...`) | 0 |
| Private notes | any run of 8 words shared with the private notes folder (23,618 runs) | 0 |
| Private names | the 9 names on the private list, case-insensitive | 0 |
| By hand | every line with `Jacob` that is not `Jacobian`, and every `Trevor` | 0 lines |

209 text files were scanned. `.git`, `.venv`, `third_party` and `avail_data` were skipped. The scan found 3 hits on
its own run, all in `tests/test_paths_config.py`: a sample `D:` path in a parser test and the needles of the test that
bans such paths. Both were rewritten so the file holds no drive path, in the working repo too.

## What was left out

The private notes, the session briefs, the plan, the agent instructions, reports and drafts, archived worktrees, all
data, the internal design docs (`docs/01` to `docs/13`) and the internal scripts. Code comments still name those
design docs in 58 places and the session briefs in 9. They are pointers, and the code reads none of them.

## To check before pushing

- Citations: the README credits the GR00T 2 whitepaper, FailBench, EmbodiedSWE, ASPIRE, RPG, and Walden and Russ
  Tedrake's CoRL 2026 keynote by name only. Add a link or paper ID for each. The 43% hack rate is EmbodiedSWE's
  figure as recalled in the brief, so confirm it against the paper.
- `Kevin` appears 49 times in code comments and docs, as the person whose decisions set the experiments.

## The test suite here, without ASPIRE (2026-11-13, 02:50)

`uv run pytest -m "not slow"` with no `third_party/ASPIRE` and no data: 2,948 passed, 106 skipped, 11 deselected in
58 s. The skips are the tests that need the station model or the zoo.

<!-- FRESH SETUP -->
