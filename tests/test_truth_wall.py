"""The truth wall (CLAUDE.md rule 10): checks never read simulator truth.

This file has 2 jobs. It holds the real tree at 0 violations. And it proves the
scanner in ``avail/walls.py`` has teeth, so that a pass on the real tree means
something: every rule is shown to fire on a snippet, clean code is shown to
pass, and the same bad snippet is shown to pass outside the guarded set.

Snippets are only parsed. Nothing in them is imported or run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from avail import walls
from avail.walls import Violation, find_wall_violations, guarded_files, is_guarded, scan_source

T_IMPORT = walls.RULE_TRUTH_IMPORT
S_IMPORT = walls.RULE_SIM_IMPORT
T_ACCESS = walls.RULE_TRUTH_ACCESS
MJ = walls.RULE_MJ_HANDLE
INTERNALS = walls.RULE_STATION_INTERNALS
UNPARSEABLE = walls.RULE_UNPARSEABLE
GUARD_MISSING = walls.RULE_GUARD_MISSING

EVIDENCE = "avail/evidence/sneaky.py"
WITNESS = "avail/witness/sneaky.py"
AGENT = "avail/agent/sneaky.py"
POLICY = "avail/fork/policy.py"

# Files that may read truth or drive the simulator. The wall does not apply to them.
UNGUARDED = [
    "avail/station/mj_plug/inject.py",
    "avail/station/mj_plug/truth.py",
    "avail/station/mj_plug/__init__.py",
    "avail/fork/human.py",
    "avail/shift/runner.py",
    "tests/test_station.py",
]


@pytest.fixture(autouse=True)
def _data_root_in_tmp(monkeypatch, tmp_path_factory):
    """Nothing here writes data. If an import ever reaches for the data root, it lands in tmp."""
    monkeypatch.setenv("AVAIL_DATA", str(tmp_path_factory.getbasetemp() / "avail_data"))


def found(source: str, rel_path: str = EVIDENCE) -> set[tuple[int, str]]:
    """The (line, rule) pairs the scanner reports for a snippet."""
    return {(v.line, v.rule) for v in scan_source(textwrap.dedent(source), rel_path)}


# ============================================================================ the real tree
def test_real_tree_has_no_wall_violations():
    violations = find_wall_violations()
    assert violations == [], "truth wall broken:\n" + "\n".join(str(v) for v in violations)


def test_real_tree_scan_covers_every_guarded_root():
    """0 violations would mean nothing if the scan looked at 0 files."""
    files = guarded_files()
    assert "avail/fork/policy.py" in files
    for folder in ("avail/evidence", "avail/witness", "avail/agent"):
        inside = [name for name in files if name.startswith(folder + "/")]
        assert inside, f"the wall scans no file under {folder}"
    assert all(is_guarded(name) for name in files)
    assert all((walls.REPO_ROOT / name).is_file() for name in files)
    unguarded_on_disk = {"avail/fork/human.py", "avail/fork/gate.py", "avail/station/base.py", "avail/walls.py"}
    assert not unguarded_on_disk & set(files)


# ============================================================================ which files are guarded
@pytest.mark.parametrize("rel_path", [
    "avail/evidence/base.py",
    "avail/evidence/__init__.py",
    "avail/evidence/sub/deeper/probe.py",
    "avail/witness/calibrate.py",
    "avail/witness/__init__.py",
    "avail/agent/mcp_server.py",
    "avail/agent/skilldoc/helper.py",
    "avail/fork/policy.py",
    "avail\\evidence\\probes.py",
    "./avail/witness/store.py",
    "avail/station/../evidence/probes.py",
])
def test_guarded_paths(rel_path):
    assert is_guarded(rel_path) is True


@pytest.mark.parametrize("rel_path", UNGUARDED + [
    "avail/fork/gate.py",
    "avail/fork/__init__.py",
    "avail/fork/policy_v2.py",
    "avail/fork/policy.pyc",
    "avail/fork/policy/extra.py",
    "avail/flywheel/labels.py",
    "avail/walls.py",
    "avail/evidence.py",
    "avail/agent.py",
    "avail/evidence_extra/source.py",
    "avail/evidence/notes.md",
    "avail/evidence/probes",
    "other/avail/evidence/probes.py",
    "evidence/probes.py",
    "../avail/evidence/probes.py",
])
def test_unguarded_paths(rel_path):
    assert is_guarded(rel_path) is False


@pytest.mark.parametrize("rel_path", ["/repo/avail/evidence/probes.py", "C:/repo/avail/evidence/probes.py",
                                      "C:\\repo\\avail\\evidence\\probes.py"])
def test_an_absolute_path_is_refused_instead_of_silently_passing(rel_path):
    """An absolute path would never match the guarded set, so a bad file would look clean."""
    with pytest.raises(ValueError, match="relative to the repo root"):
        scan_source("import mujoco\n", rel_path)
    with pytest.raises(ValueError, match="relative to the repo root"):
        is_guarded(rel_path)


# ============================================================================ snippets that must be flagged
# (id, rel_path, source, the exact set of (line, rule) the scanner must report)
BAD: list[tuple[str, str, str, set[tuple[int, str]]]] = [
    # ---- rule (a): the truth module, import statements
    ("truth-import-absolute", EVIDENCE,
     "import avail.station.mj_plug.truth\n", {(1, T_IMPORT)}),
    ("truth-import-absolute-aliased", WITNESS,
     "import avail.station.mj_plug.truth as ground\n", {(1, T_IMPORT)}),
    ("truth-from-package", EVIDENCE,
     "from avail.station.mj_plug import truth\n", {(1, T_IMPORT)}),
    ("truth-from-package-aliased", AGENT,
     "from avail.station.mj_plug import truth as t\n", {(1, T_IMPORT)}),
    ("truth-from-module", EVIDENCE,
     "from avail.station.mj_plug.truth import truth\n", {(1, T_IMPORT)}),
    ("truth-from-module-aliased", WITNESS,
     "from avail.station.mj_plug.truth import truth as peek\n", {(1, T_IMPORT)}),
    ("truth-from-module-star", EVIDENCE,
     "from avail.station.mj_plug.truth import *\n", {(1, T_IMPORT)}),
    ("truth-relative-from-package", "avail/evidence/probes.py",
     "from ..station.mj_plug import truth\n", {(1, T_IMPORT)}),
    ("truth-relative-from-module", "avail/evidence/probes.py",
     "from ..station.mj_plug.truth import x\n", {(1, T_IMPORT)}),
    ("truth-relative-in-package-init", "avail/witness/__init__.py",
     "from ..station.mj_plug import truth as ground\n", {(1, T_IMPORT)}),
    ("truth-relative-from-subpackage", "avail/agent/skilldoc/helper.py",
     "from ...station.mj_plug.truth import truth\n", {(1, T_IMPORT)}),
    ("truth-relative-in-fork-policy", POLICY,
     "from ..station.mj_plug import truth as t\n", {(1, T_IMPORT)}),
    ("truth-import-inside-a-function", EVIDENCE,
     """\
     def collect(station, claim, params):
         from avail.station.mj_plug.truth import truth as peek
         return 1.0
     """, {(2, T_IMPORT)}),
    ("truth-import-inside-try", WITNESS,
     """\
     try:
         import avail.station.mj_plug.truth as ground
     except ImportError:
         ground = None
     """, {(2, T_IMPORT)}),
    ("truth-reexported-under-another-name", AGENT,
     "from avail.station import truth as peek\n", {(1, T_IMPORT)}),
    ("truth-reexported-relative", WITNESS,
     "from ..shift.evaluator import truth\n", {(1, T_IMPORT)}),
    # ---- rule (a): the truth module, named by a string
    ("truth-importlib-literal", EVIDENCE,
     """\
     import importlib
     mod = importlib.import_module("avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-importlib-module-aliased", EVIDENCE,
     """\
     import importlib as il
     mod = il.import_module("avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-import_module-bare", WITNESS,
     """\
     from importlib import import_module
     mod = import_module("avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-importlib-keyword", AGENT,
     """\
     import importlib
     mod = importlib.import_module(name="avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-importlib-concatenated-literal", EVIDENCE,
     """\
     import importlib
     mod = importlib.import_module("avail.station." + "mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-importlib-fstring-without-fields", EVIDENCE,
     """\
     import importlib
     mod = importlib.import_module(f"avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-dunder-import-positional-fromlist", AGENT,
     'pkg = __import__("avail.station.mj_plug", globals(), locals(), ["truth"])\n', {(1, T_IMPORT), (1, S_IMPORT)}),
    ("truth-importlib-relative-to-literal-package", EVIDENCE,
     """\
     import importlib
     mod = importlib.import_module(".truth", "avail.station.mj_plug")
     """, {(2, T_IMPORT)}),
    ("truth-importlib-relative-to-own-package", "avail/evidence/probes.py",
     """\
     import importlib
     mod = importlib.import_module("..station.mj_plug.truth", __package__)
     """, {(2, T_IMPORT)}),
    ("truth-dunder-import", EVIDENCE,
     'mod = __import__("avail.station.mj_plug.truth")\n', {(1, T_IMPORT)}),
    ("truth-dunder-import-fromlist", WITNESS,
     'pkg = __import__("avail.station.mj_plug", fromlist=["truth"])\n', {(1, T_IMPORT), (1, S_IMPORT)}),
    ("truth-dunder-import-with-level", "avail/evidence/probes.py",
     'mod = __import__("station.mj_plug.truth", globals(), locals(), [], 2)\n', {(1, T_IMPORT)}),
    ("truth-builtins-dunder-import", AGENT,
     """\
     import builtins
     mod = builtins.__import__("avail.station.mj_plug.truth")
     """, {(2, T_IMPORT)}),
    ("truth-sys-modules-lookup", EVIDENCE,
     """\
     import sys
     mod = sys.modules["avail.station.mj_plug.truth"]
     """, {(2, T_IMPORT)}),
    ("truth-entry-point-string", AGENT,
     'TRUTH_FN = "avail.station.mj_plug.truth:truth"\n', {(1, T_IMPORT)}),
    # ---- rule (b): mujoco and the sim backend
    ("sim-import-mujoco", EVIDENCE, "import mujoco\n", {(1, S_IMPORT)}),
    ("sim-import-mujoco-aliased", WITNESS, "import mujoco as mj\n", {(1, S_IMPORT)}),
    ("sim-from-mujoco", AGENT, "from mujoco import MjData\n", {(1, S_IMPORT)}),
    ("sim-import-mujoco-submodule", EVIDENCE, "import mujoco.viewer\n", {(1, S_IMPORT)}),
    ("sim-import-mujoco-among-others", EVIDENCE, "import json, mujoco, math\n", {(1, S_IMPORT)}),
    ("sim-from-mujoco-in-fork-policy", POLICY, "from mujoco import mj_step\n", {(1, S_IMPORT)}),
    ("sim-importlib-mujoco", EVIDENCE,
     """\
     import importlib
     mj = importlib.import_module("mujoco")
     """, {(2, S_IMPORT)}),
    ("sim-dunder-import-mujoco", WITNESS, 'mj = __import__("mujoco")\n', {(1, S_IMPORT)}),
    ("sim-importlib-mujoco-by-keyword", WITNESS,
     """\
     import importlib
     mj = importlib.import_module(name="mujoco")
     """, {(2, S_IMPORT)}),
    ("sim-importlib-mujoco-fstring-without-fields", EVIDENCE,
     """\
     import importlib
     mj = importlib.import_module(f"mujoco")
     """, {(2, S_IMPORT)}),
    ("sim-importlib-mujoco-joined-literals", EVIDENCE,
     """\
     import importlib
     mj = importlib.import_module("mu" + "jo" + "co")
     """, {(2, S_IMPORT)}),
    ("sim-import_module-aliased-mujoco", AGENT,
     """\
     from importlib import import_module as load
     mj = load("mujoco")
     """, {(2, S_IMPORT)}),
    ("sim-from-backend-class", AGENT,
     "from avail.station.mj_plug import MjPlugStation\n", {(1, S_IMPORT)}),
    ("sim-import-backend-package", EVIDENCE,
     "import avail.station.mj_plug\n", {(1, S_IMPORT)}),
    ("sim-from-station-import-backend", EVIDENCE,
     "from avail.station import mj_plug\n", {(1, S_IMPORT)}),
    ("sim-from-backend-star", WITNESS,
     "from avail.station.mj_plug import *\n", {(1, S_IMPORT)}),
    ("sim-from-backend-submodule", EVIDENCE,
     "from avail.station.mj_plug.inject import scripted_insert\n", {(1, S_IMPORT)}),
    ("sim-backend-module-that-only-starts-like-truth", EVIDENCE,
     "from avail.station.mj_plug.truth_table import rows\n", {(1, S_IMPORT)}),
    ("sim-relative-backend-aliased", "avail/agent/standin_s1.py",
     "from ..station import mj_plug as backend\n", {(1, S_IMPORT)}),
    ("sim-relative-backend-submodule", "avail/evidence/vision.py",
     "from ..station.mj_plug.scene import make_scene\n", {(1, S_IMPORT)}),
    ("sim-relative-backend-in-package-init", "avail/witness/__init__.py",
     "from ..station.mj_plug import MjPlugStation\n", {(1, S_IMPORT)}),
    ("sim-relative-backend-from-subpackage", "avail/agent/skilldoc/helper.py",
     "from ...station.mj_plug import MjPlugStation\n", {(1, S_IMPORT)}),
    ("sim-relative-backend-in-fork-policy", POLICY,
     "from ..station.mj_plug.inject import scripted_insert\n", {(1, S_IMPORT)}),
    ("sim-type-checking-block", AGENT,
     """\
     from typing import TYPE_CHECKING
     if TYPE_CHECKING:
         from avail.station.mj_plug import MjPlugStation
     """, {(3, S_IMPORT)}),
    ("sim-importlib-backend", EVIDENCE,
     """\
     import importlib
     scene = importlib.import_module("avail.station.mj_plug.scene")
     """, {(2, S_IMPORT)}),
    ("sim-importlib-relative-to-literal-package", EVIDENCE,
     """\
     import importlib
     scene = importlib.import_module(".scene", "avail.station.mj_plug")
     """, {(2, S_IMPORT)}),
    ("sim-importlib-relative-to-own-package", "avail/evidence/probes.py",
     """\
     import importlib
     scene = importlib.import_module("..station.mj_plug.scene", package=__package__)
     """, {(2, S_IMPORT)}),
    ("sim-dunder-import-backend-with-level", "avail/evidence/probes.py",
     'scene = __import__("station.mj_plug.scene", globals(), locals(), [], 2)\n', {(1, S_IMPORT)}),
    ("sim-dunder-import-backend-with-keyword-level", "avail/evidence/probes.py",
     'scene = __import__("station.mj_plug.scene", level=2)\n', {(1, S_IMPORT)}),
    ("sim-backend-path-in-a-table", AGENT,
     'STATIONS = {"sim": "avail.station.mj_plug"}\n', {(1, S_IMPORT)}),
    ("sim-backend-entry-point-string", AGENT,
     'FACTORY = "avail.station.mj_plug:MjPlugStation"\n', {(1, S_IMPORT)}),
    ("sim-backend-prefix-for-a-built-path", EVIDENCE,
     """\
     import importlib
     def load(name):
         return importlib.import_module("avail.station.mj_plug." + name)
     """, {(3, S_IMPORT)}),
    ("sim-backend-through-attribute", EVIDENCE,
     """\
     import avail.station
     inject = avail.station.mj_plug.inject
     """, {(2, S_IMPORT)}),
    ("sim-backend-through-attribute-of-the-station-package", "avail/agent/standin_s1.py",
     """\
     from .. import station
     scene = station.mj_plug.scene
     """, {(2, S_IMPORT)}),
    ("mixed-truth-and-backend-in-one-import", EVIDENCE,
     "from avail.station.mj_plug import truth, inject\n", {(1, T_IMPORT), (1, S_IMPORT)}),
    # ---- rule (c): anything named truth
    ("truth-call-bare-name", EVIDENCE,
     """\
     def collect(station, claim, params):
         return truth(station)
     """, {(2, T_ACCESS)}),
    ("truth-call-method-on-station", EVIDENCE,
     """\
     def collect(station, claim, params):
         return station.truth()
     """, {(2, T_ACCESS)}),
    ("truth-attribute-without-call", WITNESS,
     """\
     def label(x):
         return x.truth
     """, {(2, T_ACCESS)}),
    ("truth-attribute-on-self-station", AGENT,
     """\
     class Policy:
         def act(self):
             return self.station.truth()["seated"]
     """, {(3, T_ACCESS)}),
    ("truth-attribute-on-a-module", EVIDENCE,
     """\
     import avail.station as st
     value = st.truth(None)
     """, {(2, T_ACCESS)}),
    ("truth-attribute-stored", AGENT,
     """\
     class Agent:
         def __init__(self, fn):
             self.truth = fn
     """, {(3, T_ACCESS)}),
    ("truth-through-getattr", EVIDENCE,
     """\
     def collect(station, claim, params):
         return getattr(station, "truth")()
     """, {(2, T_ACCESS)}),
    ("truth-call-in-fork-policy", POLICY,
     """\
     def choose(verdict, options, station):
         return "retry" if truth(station)["seated"] else "ask_human"
     """, {(2, T_ACCESS)}),
    # ---- rule (c): MuJoCo handles on any object
    ("mj-private-data-on-station", EVIDENCE,
     """\
     def collect(station, claim, params):
         return station._mj_data.qpos[7]
     """, {(2, MJ)}),
    ("mj-private-model-on-any-object", WITNESS,
     """\
     def step(x):
         return x._mj_model.opt.timestep
     """, {(2, MJ)}),
    ("mj-public-handles-deep-in-a-chain", AGENT,
     """\
     class Agent:
         def depth(self):
             return self.env.mj_data.site_xpos[self.env.mj_model.site("plug_tip").id]
     """, {(3, MJ)}),
    ("mj-class-names-as-attributes", EVIDENCE,
     """\
     def fresh(backend, xml):
         model = backend.MjModel.from_xml_path(xml)
         return backend.MjData(model)
     """, {(2, MJ), (3, MJ)}),
    ("mj-through-getattr-with-default", EVIDENCE,
     """\
     def collect(station, claim, params):
         data = getattr(station, "_mj_data", None)
         return data
     """, {(2, MJ)}),
    ("mj-through-hasattr", WITNESS,
     """\
     def is_sim(station):
         return hasattr(station, "mj_model")
     """, {(2, MJ)}),
    ("mj-attribute-on-a-later-line-of-the-expression", EVIDENCE,
     """\
     def collect(station, claim, params):
         return (station
                 ._mj_data
                 .qpos)
     """, {(3, MJ)}),
    # ---- rule (c): the internals of a station
    ("station-data-read", EVIDENCE,
     """\
     def collect(station, claim, params):
         return station.data.qpos[7]
     """, {(2, INTERNALS)}),
    ("station-model-read", EVIDENCE,
     """\
     def collect(station, claim, params):
         return station.model.opt.timestep
     """, {(2, INTERNALS)}),
    ("station-private-sim-on-stn", WITNESS,
     """\
     def depth(stn):
         return stn._sim.plug_depth()
     """, {(2, INTERNALS)}),
    ("station-data-on-self-station", AGENT,
     """\
     class Agent:
         def depth(self):
             return self.station._data.qpos
     """, {(3, INTERNALS)}),
    ("station-model-on-self-private-station", AGENT,
     """\
     class Agent:
         def step(self):
             return self._station._model.opt.timestep
     """, {(3, INTERNALS)}),
    ("station-internals-written", EVIDENCE,
     """\
     def reset(station):
         station._sim = None
         del station.data
     """, {(2, INTERNALS), (3, INTERNALS)}),
    ("station-internals-through-getattr", EVIDENCE,
     """\
     def collect(station, claim, params):
         return getattr(station, "_sim")
     """, {(2, INTERNALS)}),
    ("station-internals-through-setattr", AGENT,
     """\
     class Agent:
         def swap(self, other):
             setattr(self._station, "_model", other)
     """, {(3, INTERNALS)}),
    ("station-held-by-another-object", AGENT,
     """\
     def depth(ctx):
         return ctx.station.data.qpos
     """, {(2, INTERNALS)}),
    ("station-internals-in-fork-policy", POLICY,
     """\
     def choose(verdict, options, station):
         return options[0] if station.data.ncon else "ask_human"
     """, {(2, INTERNALS)}),
    # ---- rule (c): the same reads spelled as class patterns in a match statement
    ("match-pattern-reads-a-mujoco-handle", EVIDENCE,
     """\
     def collect(station, claim, params):
         match station:
             case object(_mj_data=data):
                 return data
     """, {(3, MJ)}),
    ("match-pattern-reads-truth", WITNESS,
     """\
     def label(row):
         match row:
             case object(seed=seed, truth=value):
                 return value
     """, {(3, T_ACCESS)}),
    ("match-pattern-reads-station-internals", AGENT,
     """\
     class Agent:
         def layout(self):
             match self.station:
                 case object(station_id="so101"):
                     return None
                 case object(_lay=lay) | object(data=lay):
                     return lay
     """, {(6, INTERNALS)}),
    ("match-pattern-nested-inside-another-pattern", EVIDENCE,
     """\
     def depth(event):
         match event:
             case [object(mj_model=model), *rest]:
                 return model
     """, {(3, MJ)}),
    # ---- several at once, each on its own line
    ("a-source-that-peeks-3-ways", "avail/evidence/probes.py",
     '''\
     """A probe that cheats."""
     import numpy as np
     from ..station.mj_plug import truth

     def collect(station, claim, params):
         obs = station.observe()
         depth = truth(station)["seated"]["depth_mm"]
         q = station._mj_data.qpos
         return float(depth), np.asarray(q), obs
     ''', {(3, T_IMPORT), (7, T_ACCESS), (8, MJ)}),
]

BAD_PARAMS = [pytest.param(rel, src, want, id=case) for case, rel, src, want in BAD]


def test_there_are_at_least_12_distinct_bad_snippets():
    sources = [textwrap.dedent(src) for _, _, src, _ in BAD]
    assert len(set(sources)) == len(sources), "2 cases share a snippet"
    assert len(set(sources)) >= 12
    assert len({case for case, _, _, _ in BAD}) == len(BAD), "2 cases share an id"
    import_forms = [case for case, _, _, want in BAD if {rule for _, rule in want} & {T_IMPORT, S_IMPORT}]
    attribute_forms = [case for case, _, _, want in BAD if {rule for _, rule in want} & {T_ACCESS, MJ, INTERNALS}]
    assert len(import_forms) >= 12 and len(attribute_forms) >= 12


def test_every_wall_rule_has_a_snippet_that_trips_it():
    tripped = {rule for _, _, _, want in BAD for _, rule in want}
    assert tripped == set(walls.WALL_RULES)


@pytest.mark.parametrize("rel_path, source, expected", BAD_PARAMS)
def test_bad_snippet_is_flagged(rel_path, source, expected):
    violations = scan_source(textwrap.dedent(source), rel_path)
    assert {(v.line, v.rule) for v in violations} == expected
    for v in violations:
        assert isinstance(v, Violation)
        assert v.rel_path == rel_path
        assert walls.STATION_ADVICE in v.message, v.message


@pytest.mark.parametrize("rel_path, source, expected", BAD_PARAMS)
@pytest.mark.parametrize("unguarded", UNGUARDED)
def test_same_snippet_is_not_flagged_outside_the_guarded_set(unguarded, rel_path, source, expected):
    """``avail/station/mj_plug/inject.py`` may use truth. So may the evaluator and OracleHuman."""
    assert scan_source(textwrap.dedent(source), unguarded) == []


@pytest.mark.parametrize("guarded", [
    "avail/evidence/passive.py", "avail/evidence/__init__.py", "avail/evidence/sub/extra.py",
    "avail/witness/cascade.py", "avail/agent/mcp_server.py", "avail/agent/skilldoc/tool.py", POLICY,
])
def test_one_snippet_is_flagged_in_every_guarded_place(guarded):
    source = """\
    from avail.station.mj_plug.truth import truth

    def peek(station):
        return truth(station), station._mj_data, station.model
    """
    assert found(source, guarded) == {(1, T_IMPORT), (4, T_ACCESS), (4, MJ), (4, INTERNALS)}


# ---- the exact name lists in the rule, one case per name
@pytest.mark.parametrize("receiver", ["station", "stn", "self.station", "self._station"])
@pytest.mark.parametrize("attr", ["data", "model", "_data", "_model", "_sim"])
def test_each_station_internal_is_flagged_on_each_station_name(receiver, attr):
    source = f"def peek(self, station, stn):\n    return {receiver}.{attr}\n"
    assert found(source, WITNESS) == {(2, INTERNALS)}


@pytest.mark.parametrize("receiver", ["station", "stn", "self.station", "self._station"])
@pytest.mark.parametrize("attr", ["_lay", "_cfg", "_retention", "_true_fz", "_true_peak_push", "_spec", "_arms",
                                  "_tool_force", "_snapshot", "_restore", "_tick", "_x", "__mangled"])
def test_every_private_attribute_of_a_station_is_off_limits(receiver, attr):
    """The sim station keeps its layout, scene, retention and true forces behind names like these."""
    source = f"def peek(self, station, stn):\n    return {receiver}.{attr}\n"
    assert found(source, EVIDENCE) == {(2, INTERNALS)}


@pytest.mark.parametrize("receiver", ["ctx.station", "self._stn", "_station", "runner.env.stn", "cls._station",
                                      "self.__station", "self.__stn", "__station"])
def test_a_station_is_still_a_station_when_another_object_holds_it(receiver):
    """The rule names station, stn, self.station and self._station. Any owner and any leading underscores count."""
    source = f"def peek(self, ctx, runner, cls):\n    return {receiver}.data, {receiver}._sim\n"
    assert found(source, AGENT) == {(2, INTERNALS)}


@pytest.mark.parametrize("receiver", ["x", "station", "self.env", "self.station", "a.b.c", "make()"])
@pytest.mark.parametrize("attr", ["_mj_model", "_mj_data", "mj_model", "mj_data", "MjData", "MjModel"])
def test_each_mujoco_handle_is_flagged_on_any_object(receiver, attr):
    source = f"def peek(self, x, station, a, make):\n    return {receiver}.{attr}\n"
    assert found(source, AGENT) == {(2, MJ)}


@pytest.mark.parametrize("receiver", ["x", "station", "self", "self.station", "labels.frame", "make()"])
def test_an_attribute_named_truth_is_flagged_on_any_object(receiver):
    source = f"def peek(self, x, station, labels, make):\n    return {receiver}.truth\n"
    assert found(source, WITNESS) == {(2, T_ACCESS)}


@pytest.mark.parametrize("call", [
    'getattr(station, "{attr}")',
    'getattr(station, "{attr}", None)',
    'hasattr(station, "{attr}")',
    'setattr(station, "{attr}", 0)',
    'delattr(station, "{attr}")',
])
@pytest.mark.parametrize("attr, rule", [("truth", T_ACCESS), ("_mj_data", MJ), ("MjModel", MJ),
                                        ("data", INTERNALS), ("_sim", INTERNALS)])
def test_attribute_access_by_string_is_the_same_violation(call, attr, rule):
    source = f"def peek(station):\n    return {call.format(attr=attr)}\n"
    assert found(source) == {(2, rule)}


def test_messages_name_what_was_found():
    """Each snippet yields exactly 1 violation, and its message opens by naming the thing that was found."""
    def what(source, rel_path=EVIDENCE):
        (violation,) = scan_source(textwrap.dedent(source), rel_path)
        return violation.message.split(". ")[0]

    # an import is reported under the module it resolves to
    assert what("from ..station.mj_plug.scene import make_scene\n", "avail/evidence/vision.py") == (
        "imports avail.station.mj_plug.scene")
    assert what("from ...station.mj_plug import truth\n", "avail/agent/skilldoc/helper.py") == (
        "imports avail.station.mj_plug.truth")
    assert what("from avail.station.mj_plug.truth import truth as peek\n") == "imports avail.station.mj_plug.truth"
    assert what("from avail.station import mj_plug\n") == "imports avail.station.mj_plug"
    assert what("from avail.station.mj_plug import MjPlugStation, make_scene\n") == "imports avail.station.mj_plug"
    assert what("import mujoco as mj\n") == "imports mujoco"
    assert what("import json, mujoco.viewer, mujoco\n") == "imports mujoco, mujoco.viewer"
    # a dynamic import is reported once, under the module the string resolves to
    assert what('import importlib\nm = importlib.import_module("avail.station.mj_plug.scene")\n') == (
        "imports avail.station.mj_plug.scene through import_module()")
    assert what('import importlib\nm = importlib.import_module(".scene", "avail.station.mj_plug")\n') == (
        "imports avail.station.mj_plug.scene through import_module()")
    assert what('import importlib\nm = importlib.import_module(".scene", package="avail.station.mj_plug")\n') == (
        "imports avail.station.mj_plug.scene through import_module()")
    assert what('def f(name):\n    return __import__("avail.station.mj_plug", fromlist=[name])\n') == (
        "imports avail.station.mj_plug through __import__()")
    assert what('FACTORY = "avail.station.mj_plug.station:MjPlugStation"\n') == (
        "names avail.station.mj_plug.station:MjPlugStation in a string")
    # calls and attributes are reported with the expression as written
    assert what("def f(station):\n    return truth(station)\n") == "calls truth()"
    assert what("def f(self):\n    return self.station.truth\n") == "reaches for self.station.truth"
    assert what("def f(self):\n    return self._station._sim\n") == "touches self._station._sim"
    assert what("def f(x):\n    return x.env._mj_data\n") == "touches x.env._mj_data"
    assert what('def f(stn):\n    return getattr(stn, "data")\n') == "touches getattr(stn, 'data')"
    assert what("import avail\nx = avail.station.mj_plug\n") == "reaches the sim backend through avail.station.mj_plug"


def test_a_long_expression_is_shortened_in_the_message_and_keeps_its_tail():
    owner = "self." + ".".join(["a_rather_long_attribute_name"] * 12)
    (violation,) = scan_source(f"def f(self):\n    return {owner}.station.truth\n", AGENT)
    assert (violation.line, violation.rule) == (2, T_ACCESS)
    assert violation.message.startswith("reaches for ...")
    assert "a_rather_long_attribute_name.station.truth." in violation.message
    assert len(violation.message) < 400


def test_a_flagged_attribute_on_a_huge_expression_is_still_reported():
    """Printing the expression can fail when it is nested hundreds deep. The finding must survive that."""
    owner = " + ".join(["a"] * 700)
    (violation,) = scan_source(f"x = ({owner}).truth\n", EVIDENCE)
    assert (violation.line, violation.rule) == (1, T_ACCESS)
    assert len(violation.message) < 400


def test_a_long_chain_of_joined_literals_is_still_read_as_one_string():
    pieces = ["avail", ".station", ".mj_plug", ".scene"] + [""] * 600
    name = " + ".join(repr(piece) for piece in pieces)
    assert found(f"import importlib\nmod = importlib.import_module({name})\n") == {(2, S_IMPORT)}


def test_the_advice_names_the_station_protocol():
    for word in ("Station protocol", "avail/station/base.py", "observe()", "robot_state()", "move_", "set_gripper()"):
        assert word in walls.STATION_ADVICE


def test_rule_names_are_stable():
    """Reports and other tests refer to rules by these names."""
    assert walls.WALL_RULES == ("truth_import", "sim_import", "truth_access", "mj_handle", "station_internals")
    assert walls.RULES == walls.WALL_RULES + ("unparseable", "guard_missing")


def test_violation_reads_as_path_line_rule_message():
    (violation,) = scan_source("\nimport mujoco\n", EVIDENCE)
    assert (violation.rel_path, violation.line, violation.rule) == (EVIDENCE, 2, "sim_import")
    assert str(violation) == f"{EVIDENCE}:2: [sim_import] {violation.message}"
    with pytest.raises(AttributeError):
        violation.line = 99                                    # a finding cannot be edited after the fact


def test_violations_come_back_in_line_order_and_without_duplicates():
    # the deepest finding sits on line 1 and the shallowest on line 2, so no walk order is line order by luck
    source = "x = [a.truth for a in b] + [a.truth for a in b]\nimport mujoco\ny = station.data\n"
    violations = scan_source(source, "avail\\evidence\\probes.py")
    assert [(v.line, v.rule) for v in violations] == [(1, T_ACCESS), (2, S_IMPORT), (3, INTERNALS)]
    assert {v.rel_path for v in violations} == {"avail/evidence/probes.py"}     # forward slashes


# ============================================================================ snippets that must pass
CLEAN: list[tuple[str, str, str]] = [
    ("empty-file", EVIDENCE, ""),
    ("stub-with-only-a-docstring", POLICY, '"""The default fork policy.\n\nBuilt in session S5.\n"""\n'),
    ("witness-with-its-own-model-attribute", "avail/witness/calibrate.py",
     '''\
     """A fused scorer. ``model`` is the fitted scikit-learn estimator, unrelated to any station."""
     from sklearn.linear_model import LogisticRegression


     class FusedScorer:
         def __init__(self, model=None, data=None):
             self.model = model or LogisticRegression()
             self._model = self.model
             self.data = data
             self._data = data

         def fit(self, table):
             self.model.fit(table[["travel_mm", "disp_mm"]], table["label"])
             return self.model.coef_

         def score(self, features):
             return float(self.model.predict_proba([features])[0][1])


     def describe(witness, scorer, model, data):
         return {"stages": witness.model, "rows": len(scorer.data), "coef": scorer._model.coef_,
                 "n": data.shape, "kind": model.__class__.__name__, "cal": witness._data}
     '''),
    ("evidence-source-that-uses-the-station-protocol", "avail/evidence/passive.py",
     '''\
     """passive.pose_residual: the measured tool pose against the taught seated pose."""
     from __future__ import annotations

     from ..schemas import Claim, Evidence
     from ..station.base import Observation, RobotState, Station


     class PoseResidual:
         source_id = "passive.pose_residual"
         kind = "passive_proprio"
         predicates = {"seated", "at_port"}

         def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
             obs: Observation = station.observe()
             state: RobotState = station.robot_state()
             axial = state.arms["right"].tool_pose.pos[2] - params["taught_z"]
             return Evidence(claim_id=claim.claim_id, source_id=self.source_id, kind=self.kind,
                             features={"axial_mm": 1000.0 * axial, "n_frames": float(len(obs.frames))})
     '''),
    ("probe-that-moves-the-robot-through-the-protocol", "avail/evidence/probes.py",
     '''\
     from ..config import load_experiment
     from ..station.base import MotionResult, Station


     def tug(station: Station, connector: str) -> dict:
         cfg = load_experiment()
         before = station.robot_state()
         result: MotionResult = station.move_relative(
             "right", delta_m=(0.0, 0.0, 0.003), frame="tool", speed=0.003,
             max_force=cfg.tug_force_cap_n(connector), max_travel=0.003)
         frames = station.observe(cameras=["right_camera_d405"])
         station.move_to_pose("right", before.arms["right"].tool_pose, speed=0.01)
         station.set_gripper("right", 0.2, max_torque=0.5)
         station.stop()
         return {"station": station.station_id, "stopped_by": result.stopped_by, "frames": frames}
     '''),
    ("agent-that-holds-a-station-and-stays-on-the-protocol", "avail/agent/mcp_server.py",
     '''\
     class ToolServer:
         def __init__(self, station):
             self.station = station
             self._station = station

         def observe(self, cameras=None):
             return self.station.observe(cameras)

         def state(self):
             stn = self._station
             return stn.robot_state(), self._station.station_id, self.station.stop
     '''),
    ("source-that-uses-nominal-geometry-and-the-public-extras-of-the-sim-station", "avail/evidence/vision.py",
     '''\
     def gap(station, stn, port):
         geometry = station.nominal()
         station.zero_force_estimate("right")
         station.wait(0.1)
         frame = station.observe(cameras=["right_camera_d405"]).frames["right_camera_d405"]
         return (geometry.seated_tool_pose(port), station.connector, frame.rgb.shape,
                 station.__class__.__name__, stn.__doc__, type(station).__module__)
     '''),
    ("a-view-that-wraps-a-station-and-passes-the-protocol-through", "avail/evidence/base.py",
     '''\
     class StationView:
         """A station as an evidence source may see it: the Station protocol and nothing else."""

         __slots__ = ("__station",)

         def __init__(self, station):
             object.__setattr__(self, "_StationView__station", station)

         @property
         def station_id(self):
             return self.__station.station_id

         def observe(self, cameras=None):
             return self.__station.observe(cameras)

         def robot_state(self):
             return self.__station.robot_state()

         def move_relative(self, arm, delta_m, frame, speed, max_force, max_travel):
             return self.__station.move_relative(arm, delta_m, frame, speed, max_force, max_travel)

         def read_signal(self, name):
             reader = getattr(self.__station, "read_signal", None)
             return reader(name) if reader else None

         def __getattr__(self, name):
             raise AttributeError(f"an evidence source may not read {name!r} from a station")


     def load_builtin(registry):
         import importlib
         for module in ("passive", "probes", "vision"):
             importlib.import_module(f"{__package__}.{module}")
         return registry
     '''),
    ("match-statements-that-read-nothing-private", POLICY,
     '''\
     def choose(verdict, station, scorer):
         match verdict:
             case object(decision="abstain"):
                 return "ask_human"
             case object(decision="reject", stage_reached=stage) if stage < 2:
                 return "retry"
         match station:
             case object(station_id="so101"):
                 return "ask_human"
         match scorer:
             case object(model=model, data=data, _fit=fit):
                 return "abort"
     '''),
    ("fork-policy-that-reads-only-the-verdict", POLICY,
     '''\
     def choose(verdict, options, attempts_left):
         if verdict.decision == "abstain":
             return "ask_human"
         if attempts_left and "repair:push_seat" in options:
             return "repair:push_seat"
         return "retry" if attempts_left else "ask_human"
     '''),
    ("the-rule-described-in-comments-docstrings-and-prose", EVIDENCE,
     '''\
     """Sources never import avail.station.mj_plug.truth and never call truth(station).

     They also keep away from station._mj_data, station.data and mujoco.
     """
     # station._sim and station.model are off limits here, and so is `import mujoco`
     NOTE = "see avail.station.mj_plug.truth for the definition of seated"
     ALSO = "avail.station.mj_plug.truth is where the evaluator gets its labels"
     WHY = "truth(station) is for the evaluator. station.data is privileged."
     '''),
    ("labels-that-arrive-as-data", "avail/witness/calibrate.py",
     '''\
     def thresholds(table, alpha):
         negatives = table[table["truth"] == 0]
         labels = table["truth_label"]
         ground_truth = labels.to_numpy()
         return negatives["score"].quantile(1 - alpha), ground_truth, {"truth": int(labels.sum())}


     def split(scores, truth):
         """``truth`` here is a plain list of labels handed in by the caller. It is never called."""
         return [s for s, t in zip(scores, truth) if t], [s for s, t in zip(scores, truth) if not t]


     def record(assist, audit):
         return assist.truth_label, audit.truth_label, audit.truthful, assist.ground_truth
     '''),
    ("names-that-only-look-like-the-forbidden-ones", AGENT,
     '''\
     def summary(x, station, stations, workstation):
         return (x.mj_data_path, x.MjDataset, x._mj_models, x.mjdata, x.model_dump(), x.data_dir,
                 station.station_id, station.model_name, station.data_rate_hz, station.dataset,
                 stations.data, workstation.model, x.station_cfg.model, x.stn_notes.data, x.mj_plug_notes,
                 x.mj_plug, x.profiles.mj_plug.noise, stations.mj_plug)   # "mj_plug" is also the sim station's id
     '''),
    ("imports-of-the-protocol-and-the-other-backends", "avail/agent/proxy.py",
     '''\
     import avail.station.base
     from avail.station import base, cap_client
     from avail.station.base import Station
     from ..station import so101
     from ..station.base import MotionResult
     from ..station.cap_client import CapStation
     from ..fork.human import Human
     from . import standin_s1
     '''),
    ("imports-whose-names-only-look-like-the-forbidden-ones", EVIDENCE,
     '''\
     import mujoco_helpers
     import numpy.mujoco
     from avail.station.mj_plug_notes import text
     from avail.station.mj_plugs import socket
     from avail_tools.station.mj_plug import thing
     from .truthiness import score
     from ..flywheel.labels import truth_label
     '''),
    ("a-relative-import-that-resolves-elsewhere", "avail/evidence/vision.py",
     "from .station.mj_plug.scene import make_scene\n"),
    ("a-relative-import-from-a-subpackage-that-resolves-elsewhere", "avail/agent/skilldoc/helper.py",
     "from ..station.mj_plug import MjPlugStation\n"),
    ("dynamic-imports-of-allowed-modules", "avail/evidence/base.py",
     '''\
     import importlib
     from importlib import import_module as load


     def registry(names):
         found = [importlib.import_module("avail.evidence.passive"), load("json"), __import__("math")]
         found += [importlib.import_module(name) for name in names]
         found.append(importlib.import_module(".probes", __package__))
         found.append(importlib.import_module("..station.base", __package__))
         return found
     '''),
    ("strings-that-are-not-module-paths", AGENT,
     '''\
     SCOPE = {"station": "mj_plug", "engine": "mujoco", "label": "truth", "word": "avail.station"}
     HELP = "the sim station lives in avail.station.mj_plug and is built by the CLI"
     '''),
    ("a-function-other-than-getattr-given-a-forbidden-word", EVIDENCE,
     '''\
     def lookup(station, row, log, fetch, register):
         log.info("truth")
         fetch(station, "data")
         register(station, "truth", "_mj_data")
         return row.get("data"), row.get("_mj_data"), getattr(row, "model"), getattr(station, "station_id")
     '''),
    ("calls-with-odd-arguments-do-not-trip-or-crash-the-scan", EVIDENCE,
     '''\
     import importlib


     def odd(station, args, kw, name):
         return (getattr(*args), getattr(station, *args), getattr(station, name), getattr(station, 5, None),
                 hasattr(station, None), importlib.import_module(*args), importlib.import_module(5),
                 importlib.import_module(), importlib.import_module(b"mujoco"), __import__(**kw),
                 importlib.import_module("mu" % "joco"), importlib.import_module(f""), getattr(station, f""),
                 __import__(name, level=name), __import__("json", fromlist=[name, 5]),
                 __import__("json", fromlist=name), __import__("json", level="2"))
     '''),
]

CLEAN_PARAMS = [pytest.param(rel, src, id=case) for case, rel, src in CLEAN]


@pytest.mark.parametrize("rel_path, source", CLEAN_PARAMS)
def test_clean_snippet_passes(rel_path, source):
    violations = scan_source(textwrap.dedent(source), rel_path)
    assert violations == [], "\n".join(str(v) for v in violations)


@pytest.mark.parametrize("receiver", ["self", "witness", "scorer", "self.scorer", "frame", "result.fit",
                                      "stations", "workstation", "self.station_cfg", "station_state", "make()"])
@pytest.mark.parametrize("attr", ["data", "model", "_data", "_model", "_sim", "_lay", "_cfg", "_private"])
def test_data_and_model_are_fine_on_objects_that_are_not_stations(receiver, attr):
    source = (f"def use(self, witness, scorer, frame, result, stations, workstation, station_state, make):\n"
              f"    return {receiver}.{attr}\n")
    assert found(source, WITNESS) == set()


def test_scanning_never_runs_or_imports_the_code():
    source = "import a_package_that_exists_nowhere_at_all\nraise SystemExit(3)\n"
    assert scan_source(source, EVIDENCE) == []


# ============================================================================ a file that does not parse
def test_a_syntax_error_in_a_guarded_file_is_a_violation():
    source = "from ..station.mj_plug import truth\n\ndef broken(:\n    pass\n"
    violations = scan_source(source, EVIDENCE)
    assert [(v.rel_path, v.line, v.rule) for v in violations] == [(EVIDENCE, 3, UNPARSEABLE)]
    assert "cannot be parsed" in violations[0].message
    assert "Fix the file" in violations[0].message


@pytest.mark.parametrize("source, line", [
    ("def f(:\n    pass\n", 1),
    ("x = 1\ny = = 2\n", 2),
    ("def f():\nreturn 1\n", 2),
    ("x = (1,\n", 1),
    ("import os\x00\n", 0),
    ("x = " + "(" * 3000 + "1" + ")" * 3000 + "\n", 1),
    ("x = " + " + ".join(["1"] * 20000) + "\n", 0),
])
def test_every_kind_of_unparseable_guarded_file_is_reported(source, line):
    violations = scan_source(source, WITNESS)
    assert [(v.line, v.rule) for v in violations] == [(line, UNPARSEABLE)]


def test_a_syntax_error_outside_the_guarded_set_is_not_this_scanners_business():
    assert scan_source("def broken(:\n    pass\n", "avail/station/mj_plug/inject.py") == []


def test_a_byte_order_mark_is_not_a_syntax_error():
    assert found("\ufeffimport json\n") == set()
    assert found("\ufeffimport mujoco\n") == {(1, S_IMPORT)}


# ============================================================================ the walk over a tree
STUB = '"""A stub."""\n'


def make_tree(root, files=None):
    """A small repo under ``root`` with 1 clean file in each guarded place, plus ``files``.

    A value of None leaves that path out. Bytes are written as they are.
    """
    layout = {
        "avail/evidence/base.py": STUB,
        "avail/witness/calibrate.py": STUB,
        "avail/agent/mcp_server.py": STUB,
        "avail/fork/policy.py": STUB,
    }
    layout.update(files or {})
    for rel, text in layout.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return root


USES_TRUTH = "from avail.station.mj_plug.truth import truth\n\n\ndef label(station):\n    return truth(station)\n"


def test_tree_scan_of_a_clean_tree_is_empty(tmp_path):
    root = make_tree(tmp_path, {"avail/evidence/odd_folder.py/inner.py": STUB})   # a folder whose name ends in .py
    assert find_wall_violations(root) == []
    assert guarded_files(root) == [
        "avail/agent/mcp_server.py", "avail/evidence/base.py", "avail/evidence/odd_folder.py/inner.py",
        "avail/fork/policy.py", "avail/witness/calibrate.py",
    ]


def test_tree_scan_finds_planted_violations_and_only_those(tmp_path):
    root = make_tree(tmp_path, {
        # guarded, and bad
        "avail/evidence/sub/deep/peek.py": USES_TRUTH,
        "avail/agent/__init__.py": "def depth(stn):\n    return stn._sim\n",
        "avail/fork/policy.py": "import mujoco\n",
        "avail/witness/store.py": "def step(x):\n\n    return x.mj_model\n",
        # not guarded: these may read truth and drive the simulator
        "avail/station/mj_plug/inject.py": USES_TRUTH,
        "avail/station/mj_plug/truth.py": "import mujoco\n\n\ndef truth(station):\n    return station._mj_data\n",
        "avail/fork/human.py": USES_TRUTH,
        "avail/fork/gate.py": USES_TRUTH,
        "avail/shift/runner.py": USES_TRUTH,
        "avail/evidence.py": USES_TRUTH,
        "tests/test_station.py": USES_TRUTH,
        # guarded folder, but not Python
        "avail/evidence/notes.md": USES_TRUTH,
    })
    violations = find_wall_violations(root)
    assert [(v.rel_path, v.line, v.rule) for v in violations] == [
        ("avail/agent/__init__.py", 2, INTERNALS),
        ("avail/evidence/sub/deep/peek.py", 1, T_IMPORT),
        ("avail/evidence/sub/deep/peek.py", 5, T_ACCESS),
        ("avail/fork/policy.py", 1, S_IMPORT),
        ("avail/witness/store.py", 3, MJ),
    ]
    assert find_wall_violations(str(root)) == violations                       # a str root works too
    assert guarded_files(root) == [
        "avail/agent/__init__.py", "avail/agent/mcp_server.py", "avail/evidence/base.py",
        "avail/evidence/sub/deep/peek.py", "avail/fork/policy.py", "avail/witness/calibrate.py",
        "avail/witness/store.py",
    ]


def test_tree_scan_reports_a_guarded_file_with_a_syntax_error(tmp_path):
    root = make_tree(tmp_path, {"avail/agent/proxy.py": "import mujoco\n\ndef broken(:\n    pass\n"})
    violations = find_wall_violations(root)
    assert [(v.rel_path, v.line, v.rule) for v in violations] == [("avail/agent/proxy.py", 3, UNPARSEABLE)]


def test_tree_scan_reads_bytes_the_way_python_does(tmp_path):
    root = make_tree(tmp_path, {
        "avail/evidence/bom.py": b"\xef\xbb\xbfimport mujoco\n",                  # UTF-8 with a byte order mark
        "avail/evidence/latin.py": b"# -*- coding: latin-1 -*-\ncaf\xe9 = 1\nimport mujoco\n",   # a Latin-1 identifier
        "avail/evidence/crlf.py": b"import json\r\nimport mujoco\r\n",
        "avail/evidence/garbage.py": b"\xff\xfeimport mujoco\n",                  # not valid UTF-8
        "avail/evidence/clean_bom.py": b"\xef\xbb\xbfimport json\n",
    })
    assert [(v.rel_path, v.line, v.rule) for v in find_wall_violations(root)] == [
        ("avail/evidence/bom.py", 1, S_IMPORT),
        ("avail/evidence/crlf.py", 2, S_IMPORT),
        ("avail/evidence/garbage.py", 1, UNPARSEABLE),
        ("avail/evidence/latin.py", 3, S_IMPORT),
    ]


def test_a_guarded_file_that_cannot_be_read_is_a_violation(tmp_path, monkeypatch):
    root = make_tree(tmp_path, {"avail/evidence/locked.py": "import json\n"})
    read_bytes = Path.read_bytes

    def refuse_locked(self):
        if self.name == "locked.py":
            raise PermissionError("locked")
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", refuse_locked)
    violations = find_wall_violations(root)
    assert [(v.rel_path, v.line, v.rule) for v in violations] == [("avail/evidence/locked.py", 0, UNPARSEABLE)]
    assert "PermissionError" in violations[0].message


@pytest.mark.parametrize("missing, reported", [
    ({"avail/witness/calibrate.py": None}, "avail/witness"),
    ({"avail/evidence/base.py": None, "avail/evidence/notes.md": "not python"}, "avail/evidence"),
    ({"avail/agent/mcp_server.py": None, "avail/agent.py": STUB}, "avail/agent"),
    ({"avail/fork/policy.py": None, "avail/fork/policies.py": STUB}, "avail/fork/policy.py"),
])
def test_a_wall_that_guards_nothing_says_so(tmp_path, missing, reported):
    """If a guarded folder is renamed, the scan must not go quiet."""
    violations = find_wall_violations(make_tree(tmp_path, missing))
    assert [(v.rel_path, v.line, v.rule) for v in violations] == [(reported, 0, GUARD_MISSING)]
    assert "the wall guards nothing here" in violations[0].message
    assert "avail/walls.py" in violations[0].message


def test_an_empty_folder_reports_every_guarded_place(tmp_path):
    violations = find_wall_violations(tmp_path)
    assert [(v.rel_path, v.rule) for v in violations] == [
        ("avail/agent", GUARD_MISSING), ("avail/evidence", GUARD_MISSING),
        ("avail/fork/policy.py", GUARD_MISSING), ("avail/witness", GUARD_MISSING),
    ]
    assert guarded_files(tmp_path) == []


# ============================================================================ running it by hand
def test_main_prints_each_violation_and_exits_1(tmp_path, capsys):
    root = make_tree(tmp_path, {"avail/evidence/peek.py": USES_TRUTH})
    assert walls.main(["--root", str(root)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("avail/evidence/peek.py:1: [truth_import] imports avail.station.mj_plug.truth. ")
    assert lines[1].startswith("avail/evidence/peek.py:5: [truth_access] calls truth(). ")
    assert all(line.endswith("observe(), robot_state(), move_*, set_gripper().") for line in lines[:2])
    assert lines[2] == "truth wall: 2 violations. Guarded files scanned: 5."
    assert len(lines) == 3


def test_main_counts_a_single_violation_in_the_singular(tmp_path, capsys):
    root = make_tree(tmp_path, {"avail/witness/peek.py": "import mujoco\n"})
    assert walls.main(["--root", str(root)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "truth wall: 1 violation. Guarded files scanned: 5."


def test_main_exits_0_on_a_clean_tree(tmp_path, capsys):
    assert walls.main(["--root", str(make_tree(tmp_path))]) == 0
    assert capsys.readouterr().out.splitlines() == ["truth wall: 0 violations. Guarded files scanned: 4."]


def test_main_on_the_real_tree_agrees_with_the_scan(capsys):
    assert walls.main([]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"truth wall: 0 violations. Guarded files scanned: {len(guarded_files())}."]


def test_the_module_runs_by_hand(tmp_path):
    """``python -m avail.walls`` is the by-hand check the module docstring promises."""
    root = make_tree(tmp_path, {"avail/agent/peek.py": "def f(station):\n    return station._lay\n"})
    done = subprocess.run([sys.executable, "-m", "avail.walls", "--root", str(root)],
                          cwd=walls.REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("avail/agent/peek.py:2: [station_internals] touches station._lay. ")
    assert lines[-1] == "truth wall: 1 violation. Guarded files scanned: 5."
