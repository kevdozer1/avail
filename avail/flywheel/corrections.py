"""Every operator takeover of a shift as a correction demo, in documented JSONL (the night of 2026-11-16, step 5).

LeRobot is not installed on this machine, so the demos are written as JSONL beside the LeRobot-style tree that
``avail.flywheel.export.export_interventions`` writes (parquet episodes, ``meta/``, ``labels.jsonl``). 1 line per takeover:

- ``episode``, ``sim_or_real`` (always ``sim`` here), ``intervention_id``, ``cycle_id``, ``item``, ``step``
- ``claim``: the claim the takeover fixed: ``claim_id``, ``predicate``, ``verdict_id``, the ``trigger`` that opened the
  fork (for example ``reject at attempt 2``), and ``label_when_taken``, what the operator found when they took the arm
- ``outcome``: ``taken_over_done`` or ``taken_over_failed``, and ``fps``, the frame log's rate
- ``handoff_frame`` and ``handback_frame``: indexes into ``frames`` where the operator took and gave back the arm
- ``frames``: from 2 s before the handoff to the handback, each with ``t`` (s from the episode's start),
  ``observation.state`` (6 joint readings in rad, then the gripper from 0 closed to 1 open), ``action`` (the 6 joint
  targets and the gripper target the controller sent) and ``tag`` (``autonomous`` or ``assisted``)

In sim the operator is ``OracleHuman``: a scripted correction from the state the robot is in. The demos are its
motions, not a person's.
"""

from __future__ import annotations

import json
from pathlib import Path

README = __doc__


def export_corrections(shift_dir, out_dir) -> dict:
    import pandas as pd

    from ..ledger.reader import iter_events
    from .export import export_interventions

    shift_dir, out_dir = Path(shift_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = out_dir / "lerobot_style"
    info = export_interventions(shift_dir, tree)
    ivs = {}
    for ledger in sorted(shift_dir.glob("*/ledger.jsonl")):
        for e in iter_events(ledger, kinds=["intervention"]):
            ivs[e["payload"]["intervention_id"]] = e["payload"]
    episodes = [json.loads(line) for line in (tree / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    path = out_dir / "corrections.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ep in episodes:
            iv = ivs.get(ep["intervention_id"], {})
            df = pd.read_parquet(tree / ep["data_path"])
            frames = [{"t": float(t), "observation.state": [float(v) for v in s], "action": [float(v) for v in a],
                       "tag": str(tag)}
                      for t, s, a, tag in zip(df["timestamp"], df["observation.state"], df["action"], df["tag"])]
            fh.write(json.dumps({
                "episode": ep["episode_index"], "sim_or_real": info["sim_or_real"],
                "intervention_id": ep["intervention_id"], "cycle_id": ep["cycle_id"], "item": ep["item"],
                "step": ep["step"],
                "claim": {"claim_id": iv.get("claim_id"), "predicate": iv.get("predicate_name"),
                          "verdict_id": iv.get("verdict_id"), "trigger": iv.get("trigger"),
                          "label_when_taken": iv.get("truth_label")},
                "outcome": ep["outcome"], "fps": info["fps"], "handoff_frame": ep["handoff_episode_frame"],
                "handback_frame": ep["handback_episode_frame"], "frames": frames}) + "\n")
            n += 1
    (out_dir / "README.md").write_text(README.strip() + "\n", encoding="utf-8", newline="\n")
    return {"corrections": n, "path": str(path), "takeovers": info["takeovers"], "skipped": info["skipped"],
            "lerobot_style": str(tree)}
