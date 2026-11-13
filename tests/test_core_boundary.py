"""The core boundary (S5): the product never reaches into the plug testbed.

Kevin's direction for S5: the small core is the product, the plug station is its testbed, and the
2 stay apart. The core never imports MuJoCo or anything plug-specific. This file holds the tree to
that with a static scan, like the truth wall in ``avail/walls.py``: source is parsed with ``ast`` and
nothing is imported, so the scan also reads a half-built file.

The core: ``avail/core``, ``avail/schemas``, ``avail/ledger``, ``avail/stats``, ``avail/witness``,
``avail/fork``, ``avail/flywheel``, ``avail/shift``, and the 2 files ``avail/evidence/base.py`` and
``avail/evidence/fused.py``. Every ``.py`` file under those folders is scanned as it is on disk when
the test runs, so a file another session adds is scanned too.

A core file must not:

1. Import ``mujoco``, ``mink``, ``avail.station``, ``avail.zoo``, ``avail.analysis``, ``avail.demo``,
   ``avail.agent``, or an evidence source (an ``avail.evidence`` module other than ``base`` and ``fused``).
   A relative import counts, and so do ``importlib.import_module`` and ``__import__`` with a constant
   name, and a string that is exactly the dotted path of such a module. Rule ``forbidden_import``.
2. Import a part of avail that is not core. The core may use 3 pieces of infrastructure:
   ``avail.paths``, ``avail.seeds``, and from ``avail.config`` only ``load_experiment`` and ``env``.
   Rule ``outside_core``.
3. Hold a plug word in a name or in a string constant: fric, latch, prong, push_seat, mj_plug,
   NominalGeometry, plug_cycle, and the class names P0 and N1 to N9. Rule ``plug_word``.

Documentation may hold plug words: docstrings, any string that is a statement by itself, comments,
and the texts given as ``description=`` or ``help=``. A plug word counts as a whole word of a name or
of a string: ``mj_plug_hard``, ``MjPlugStation`` and ``"repair:push_seat"`` hold one, ``friction``
and ``latched`` hold none. The class names count in capitals only, so ``p0`` and ``n1`` are free.

``avail/router`` ships with the core and is held to rules 1 and 3. ``avail/config.py``,
``avail/paths.py`` and ``avail/seeds.py`` are held to rule 1, so the core cannot reach the testbed
through them. Plug words belong in the config by design.

The scan fails closed: a scanned file that does not parse is a finding, and so is a core folder with
no Python file. A static scan cannot see a module name built at run time, as when the shift runner
imports a task's testbed by the name its task file gives. That is how the core reaches a testbed
without naming one. ``test_importing_the_core_loads_no_testbed_module`` imports every core module in
a fresh interpreter and checks what got loaded.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avail.paths import REPO_ROOT

# ----------------------------------------------------------------------------- what is scanned
CORE_DIRS = ("avail/core", "avail/schemas", "avail/ledger", "avail/stats", "avail/witness", "avail/fork",
             "avail/flywheel", "avail/shift")
CORE_FILES = ("avail/evidence/base.py", "avail/evidence/fused.py")
SHARED_DIRS = ("avail/router",)
INFRA_FILES = ("avail/config.py", "avail/paths.py", "avail/seeds.py")

RULE_FORBIDDEN = "forbidden_import"
RULE_OUTSIDE = "outside_core"
RULE_WORD = "plug_word"
RULE_UNPARSEABLE = "unparseable"
RULE_MISSING = "missing"

CORE_RULES = frozenset({RULE_FORBIDDEN, RULE_OUTSIDE, RULE_WORD})
SHARED_RULES = frozenset({RULE_FORBIDDEN, RULE_WORD})
INFRA_RULES = frozenset({RULE_FORBIDDEN})

# ----------------------------------------------------------------------------- what is off limits
FORBIDDEN_TOP = ("mujoco", "mink")
FORBIDDEN_AVAIL = ("avail.station", "avail.zoo", "avail.analysis", "avail.demo", "avail.agent")
EVIDENCE_PACKAGE = "avail.evidence"
EVIDENCE_IN_CORE = ("avail.evidence.base", "avail.evidence.fused")
CORE_MODULES = tuple(d.replace("/", ".") for d in CORE_DIRS) + EVIDENCE_IN_CORE
INFRA_MODULES = ("avail.paths", "avail.seeds")
CONFIG_NAMES = ("load_experiment", "env")

PLUG_WORDS = ("fric", "latch", "prong", "push_seat", "mj_plug", "NominalGeometry", "plug_cycle")
PLUG_CLASSES = ("P0",) + tuple(f"N{k}" for k in range(1, 10))
DOC_KEYWORDS = ("description", "help")

# Findings that were read and are kept, each with its reason. An entry that matches nothing must go.
EXCEPTIONS: dict[tuple[str, str, str], str] = {
    ("avail/stats/planning.py", RULE_WORD, "P0"):
        "WaldSPRT.P0 names the hypothesis p <= p0 of Wald's sequential test, the statistics sense of p0. "
        "It is not the class of a seated plug.",
}

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_ATOM = re.compile(r"[A-Z]+[0-9]*(?![a-z])|[A-Z]?[a-z]+[0-9]*|[0-9]+")
_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?")
_IMPORTERS = frozenset({"import_module", "__import__"})


@dataclass(frozen=True, order=True)
class Finding:
    """One place where a scanned file crosses the core boundary. ``detail`` is the module or the word."""

    rel: str
    line: int
    rule: str
    detail: str
    message: str

    def __str__(self) -> str:
        return f"{self.rel}:{self.line}: [{self.rule}] {self.message}"


# ----------------------------------------------------------------------------- the rules
def _under(target: str, module: str) -> bool:
    return target == module or target.startswith(module + ".")


def import_rule(target: str) -> str | None:
    """The rule a core file breaks by importing the dotted path ``target``, or None when it may."""
    top = target.split(".")[0]
    if top in FORBIDDEN_TOP:
        return RULE_FORBIDDEN
    if top != "avail":
        return None                                           # the standard library and third-party packages
    if any(_under(target, module) for module in FORBIDDEN_AVAIL):
        return RULE_FORBIDDEN
    if _under(target, EVIDENCE_PACKAGE) and not any(_under(target, module) for module in EVIDENCE_IN_CORE):
        return RULE_FORBIDDEN
    if target == "avail" or any(_under(target, module) for module in CORE_MODULES + INFRA_MODULES):
        return None
    if target in {f"avail.config.{name}" for name in CONFIG_NAMES}:
        return None
    return RULE_OUTSIDE


def words(text: str) -> list[list[str]]:
    """The words of each name-like token in ``text``: ``MjPlugStation`` gives ``Mj``, ``Plug``, ``Station``."""
    return [[atom for part in token.split("_") for atom in _ATOM.findall(part)] for token in _TOKEN.findall(text)]


_PLUG_WORD_ATOMS = tuple((word, [atom.lower() for atom in words(word)[0]]) for word in PLUG_WORDS)


def plug_words_in(text: str) -> list[str]:
    """The plug words ``text`` holds as whole words. The class names count in capitals only."""
    found = set()
    for atoms in words(text):
        lower = [atom.lower() for atom in atoms]
        for word, target in _PLUG_WORD_ATOMS:
            if any(lower[i:i + len(target)] == target for i in range(len(lower) - len(target) + 1)):
                found.add(word)
        found.update(atom for atom in atoms if atom in PLUG_CLASSES)
    return sorted(found)


def rules_for(rel: str) -> frozenset[str]:
    """The rules a repo-relative path is held to. Empty when the scan does not read it."""
    if rel in CORE_FILES or any(rel.startswith(folder + "/") for folder in CORE_DIRS):
        return CORE_RULES
    if any(rel.startswith(folder + "/") for folder in SHARED_DIRS):
        return SHARED_RULES
    if rel in INFRA_FILES:
        return INFRA_RULES
    return frozenset()


def _const_str(node: ast.AST | None) -> str | None:
    """The string a constant expression spells: a literal, literals joined with ``+``, an f-string with no field."""
    pieces, pending = [], [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            pieces.append(current.value)
        elif isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            pending += [current.right, current.left]
        elif isinstance(current, ast.JoinedStr):
            pending += reversed(current.values)
        else:
            return None
    return "".join(pieces)


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ----------------------------------------------------------------------------- the scan of 1 file
class _Scan:
    def __init__(self, rel: str, rules: frozenset[str]) -> None:
        self.rel, self.rules = rel, rules
        self.package = rel[:-len(".py")].split("/")[:-1]
        self.found: set[Finding] = set()
        self.docs: set[int] = set()                    # string constants that are documentation
        self.consumed: set[int] = set()                # string constants already judged as an import
        self.importers = {name: name for name in _IMPORTERS}

    def run(self, tree: ast.AST) -> list[Finding]:
        nodes = list(ast.walk(tree))
        for node in nodes:
            if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                self.docs.update(id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant))
            elif isinstance(node, ast.keyword) and node.arg in DOC_KEYWORDS:
                self.docs.update(id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _IMPORTERS and alias.asname:
                        self.importers[alias.asname] = alias.name
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._judge(alias.name, node.lineno, alias.name)
                    self._name(alias.asname, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                self._import_from(node)
            elif isinstance(node, ast.Call):
                self._call(node)
        for node in nodes:
            self._names_of(node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in self.docs:
                self._string(node)
        return sorted(self.found)

    def _add(self, line: int, rule: str, detail: str, message: str) -> None:
        if rule in self.rules:
            self.found.add(Finding(self.rel, int(line), rule, detail, message))

    # ---- imports
    def _absolute(self, level: int, module: str, package: list[str] | None = None) -> str:
        if level <= 0:
            return module
        parts = self.package if package is None else package
        base = parts[:max(0, len(parts) - (level - 1))]
        return ".".join([*base, module] if module else base)

    def _judge(self, target: str, line: int, shown: str, how: str = "imports") -> None:
        rule = import_rule(target)
        if rule == RULE_FORBIDDEN:
            self._add(line, rule, shown, f"{how} {shown}. The core never imports the simulator or the testbed: no "
                      "mujoco, mink, station, zoo, analysis, demo, agent, and no evidence source but base and "
                      "fused. Hand the core what it needs as data, or through a protocol in avail/core.")
        elif rule == RULE_OUTSIDE:
            self._add(line, rule, shown, f"{how} {shown}, which is not core. The core may import the core "
                      "packages, avail.paths, avail.seeds, and load_experiment and env from avail.config.")

    def _import_from(self, node: ast.ImportFrom) -> None:
        base = self._absolute(node.level, node.module or "")
        for alias in node.names:
            target = base if alias.name == "*" else ".".join(part for part in (base, alias.name) if part)
            rule = import_rule(target)
            # name the module when all of it is off limits, and the name when only some of the module is
            whole = base not in ("", "avail", EVIDENCE_PACKAGE, "avail.config") and import_rule(base) == rule
            self._judge(target, node.lineno, base if whole else target)
            if alias.name != "*":
                self._name(alias.name, node.lineno)
            self._name(alias.asname, node.lineno)

    def _call(self, node: ast.Call) -> None:
        func = node.func
        importer = None
        if isinstance(func, ast.Name):
            importer = self.importers.get(func.id)
        elif isinstance(func, ast.Attribute) and func.attr in _IMPORTERS:
            importer = func.attr
        if importer is None:
            return
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

        def argument(position: int, name: str) -> ast.AST | None:
            return node.args[position] if len(node.args) > position else keywords.get(name)

        name_node = argument(0, "name")
        name = _const_str(name_node)
        if name is None:
            return                                    # a name built at run time: see the module docstring
        self.consumed.update(id(n) for n in ast.walk(name_node) if isinstance(n, ast.Constant))
        if importer == "import_module":
            package = _const_str(argument(1, "package"))
            level = len(name) - len(name.lstrip("."))
            targets = [self._absolute(level, name[level:], package.split(".") if package else None)]
        else:
            level_node = argument(4, "level")
            level = level_node.value if isinstance(level_node, ast.Constant) and isinstance(level_node.value, int) else 0
            targets = [self._absolute(level, name)]
            fromlist = argument(3, "fromlist")
            if isinstance(fromlist, (ast.List, ast.Tuple, ast.Set)):
                targets += [f"{targets[0]}.{item}" for item in (_const_str(e) for e in fromlist.elts) if item]
        for target in targets:
            self._judge(target, node.lineno, target, f"imports through {importer}()")

    # ---- names and strings
    def _name(self, name: str | None, line: int) -> None:
        if not name:
            return
        for word in plug_words_in(name):
            self._add(line, RULE_WORD, word, f"the name {_short(name)} holds the plug word {word}. Plug words belong "
                      "to the testbed: use a generic name, or move the code to the testbed.")

    def _names_of(self, node: ast.AST) -> None:
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            self._name(node.id, line)
        elif isinstance(node, ast.Attribute):
            self._name(node.attr, getattr(node, "end_lineno", None) or line)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._name(node.name, line)
        elif isinstance(node, ast.arg):
            self._name(node.arg, line)
        elif isinstance(node, ast.keyword):
            self._name(node.arg, line)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                self._name(name, line)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            self._name(node.name, line)
        elif isinstance(node, ast.MatchMapping):
            self._name(node.rest, line)
        elif isinstance(node, ast.MatchClass):
            for name in node.kwd_attrs:
                self._name(name, line)
        elif type(node).__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            self._name(getattr(node, "name", None), line)

    def _string(self, node: ast.Constant) -> None:
        text = node.value
        for word in plug_words_in(text):
            self._add(node.lineno, RULE_WORD, word, f"the string {_short(text)!r} holds the plug word {word}. Plug "
                      "words belong to the testbed: make the text generic, or move it to the testbed. A docstring, "
                      "a description= or a help= text may hold them.")
        if id(node) in self.consumed:
            return
        dotted = text[:-1] if text.endswith(".") else text
        if _DOTTED.fullmatch(dotted) and import_rule(dotted.replace(":", ".")) == RULE_FORBIDDEN:
            self._judge(dotted.replace(":", "."), node.lineno, dotted, "names in a string")


def scan_source(source: str | bytes, rel: str) -> list[Finding]:
    """The boundary findings in 1 file's source. ``rel`` is the path from the repo root with forward slashes."""
    rules = rules_for(rel)
    if not rules:
        return []
    if isinstance(source, str) and source.startswith("﻿"):
        source = source[1:]
    try:
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, ValueError, RecursionError) as exc:
        detail = getattr(exc, "msg", None) or str(exc) or type(exc).__name__
        return [Finding(rel, int(getattr(exc, "lineno", 0) or 0), RULE_UNPARSEABLE, "",
                        f"cannot be parsed ({detail}), so the boundary cannot be checked here. Fix the file.")]
    return _Scan(rel, rules).run(tree)


def scanned_files(root: Path = REPO_ROOT) -> dict[str, frozenset[str]]:
    """Every file the scan reads under ``root``, with the rules it is held to."""
    out: dict[str, frozenset[str]] = {}
    for folder in CORE_DIRS + SHARED_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                out[rel] = rules_for(rel)
    for name in CORE_FILES + INFRA_FILES:
        if (root / name).is_file():
            out[name] = rules_for(name)
    return out


def find_findings(root: Path = REPO_ROOT) -> list[Finding]:
    """Every finding under ``root``, the kept exceptions included. A missing core place is a finding too."""
    files = scanned_files(root)
    out: list[Finding] = []
    for folder in CORE_DIRS + SHARED_DIRS:
        if not any(rel.startswith(folder + "/") for rel in files):
            out.append(Finding(folder, 0, RULE_MISSING, folder, "holds no Python file, so the scan reads nothing "
                               "here. If the code moved, update the lists at the top of this test."))
    for name in CORE_FILES + INFRA_FILES:
        if name not in files:
            out.append(Finding(name, 0, RULE_MISSING, name, "does not exist, so the scan reads nothing here."))
    for rel in files:
        try:
            source = (root / rel).read_bytes()
        except OSError as exc:
            out.append(Finding(rel, 0, RULE_UNPARSEABLE, "", f"cannot be read ({type(exc).__name__})."))
            continue
        out.extend(scan_source(source, rel))
    return sorted(out)


def module_name(rel: str) -> str:
    """``avail.core.task`` for ``avail/core/task.py`` and ``avail.core`` for ``avail/core/__init__.py``."""
    parts = rel[:-len(".py")].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def kept(finding: Finding) -> bool:
    return (finding.rel, finding.rule, finding.detail) in EXCEPTIONS


# ============================================================================ the real tree
@pytest.fixture(scope="module")
def findings() -> list[Finding]:
    return find_findings()


@pytest.mark.parametrize("place", CORE_DIRS + CORE_FILES + SHARED_DIRS + INFRA_FILES)
def test_the_core_stays_clear_of_the_testbed(findings, place):
    here = [f for f in findings if (f.rel == place or f.rel.startswith(place + "/")) and not kept(f)]
    assert not here, f"{place} crosses the core boundary:\n" + "\n".join(str(f) for f in here)


def test_the_scan_reads_the_core():
    """0 findings would mean nothing if the scan read nothing."""
    files = scanned_files()
    for folder in CORE_DIRS + SHARED_DIRS:
        assert any(rel.startswith(folder + "/") for rel in files), folder
    for rel in ("avail/core/robot.py", "avail/core/task.py", "avail/witness/cascade.py", "avail/schemas/records.py",
                "avail/evidence/base.py", "avail/evidence/fused.py", "avail/router/core.py", "avail/config.py"):
        assert rel in files, rel
    assert files["avail/evidence/base.py"] == CORE_RULES and files["avail/router/core.py"] == SHARED_RULES
    assert files["avail/config.py"] == INFRA_RULES
    assert not any(rel.startswith(("avail/station/", "avail/zoo/", "avail/analysis/", "avail/agent/")) for rel in files)
    assert not any(rel.startswith("avail/evidence/") and rel not in CORE_FILES for rel in files)


def test_every_kept_exception_is_still_needed(findings):
    seen = {(f.rel, f.rule, f.detail) for f in findings}
    stale = [key for key in EXCEPTIONS if key not in seen]
    assert not stale, f"these exceptions match nothing any more and must go: {stale}"


def test_importing_the_core_loads_no_testbed_module(tmp_path):
    """Every core module, imported in a fresh interpreter, loads no simulator and no testbed module."""
    modules = sorted({module_name(rel) for rel, rules in scanned_files().items() if rules == CORE_RULES})
    code = textwrap.dedent("""
        import importlib, json, sys
        failed = {}
        for name in json.loads(sys.argv[1]):
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {exc}"
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("avail", "mujoco", "mink"))
        print(json.dumps({"failed": failed, "loaded": loaded}))
    """)
    env = {**os.environ, "AVAIL_DATA": str(tmp_path / "avail_data")}
    done = subprocess.run([sys.executable, "-c", code, json.dumps(modules)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert not result["failed"], f"core modules that do not import: {result['failed']}"
    testbed = [m for m in result["loaded"] if m != EVIDENCE_PACKAGE and import_rule(m) == RULE_FORBIDDEN]
    assert not testbed, f"importing the core loaded {testbed}"
    assert {"avail.core.robot", "avail.core.task", "avail.witness.cascade", "avail.evidence.base"} <= set(result["loaded"])


# ============================================================================ the core names no task's own words
def task_vocabulary() -> dict[str, str]:
    """Every step, predicate and item kind a task file names, with what it is in which task."""
    from avail.core.task import load_task, task_names

    out: dict[str, str] = {}
    for name in task_names():
        task = load_task(name)
        for step in task.steps:
            out.setdefault(step.name, f"a step of {name}")
        for predicate in task.predicates:
            out.setdefault(predicate, f"a predicate of {name}")
        out.setdefault(task.items.kind, f"the item kind of {name}")
    return out


def task_words_in(source: str, rel: str, vocabulary) -> list[tuple[int, str]]:
    """The string constants of ``source`` that are exactly a word of ``vocabulary``. Documentation may hold them."""
    tree = ast.parse(source, filename=rel)
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            docs.update(id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant))
        elif isinstance(node, ast.keyword) and node.arg in DOC_KEYWORDS:
            docs.update(id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant))
    return sorted((node.lineno, node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
                  and node.value in vocabulary)


def test_the_core_names_no_step_predicate_or_item_kind_of_a_task():
    """A task's steps, predicates and item kind live in its task file, and the core reads them from there.

    S5 found 2 in the cycle driver the plug-word scan could not see: it sent the vlm arm's judge to the
    step named insert, and gave the carry step's last reject its own question. Both are task file
    entries now (a step's ``judge``, and fork.ask's ``<predicate>_reject``)."""
    vocabulary = task_vocabulary()
    assert {"pick", "carry", "insert", "release", "held", "at_port", "seated", "cable"} <= set(vocabulary)
    found = [f"{rel}:{line}: the string {word!r} is {vocabulary[word]}"
             for rel, rules in scanned_files().items() if rules == CORE_RULES
             for line, word in task_words_in((REPO_ROOT / rel).read_text(encoding="utf-8"), rel, vocabulary)]
    assert not found, "the core names a task's own words. Read them from the task file:\n" + "\n".join(found)


def test_the_task_word_scan_has_teeth():
    vocabulary = {"insert": "a step", "held": "a predicate"}
    source = textwrap.dedent('''
        """The insert step claims held."""
        if step.name != "insert":
            pass
        parser.add_argument("--x", help="held")
        x = f"{name}_reject"
        ''')
    assert task_words_in(source, "avail/shift/snippet.py", vocabulary) == [(3, "insert")]


# ============================================================================ the scanner has teeth
def found(source: str, rel: str = "avail/witness/snippet.py") -> set[tuple[int, str, str]]:
    return {(f.line, f.rule, f.detail) for f in scan_source(textwrap.dedent(source), rel)}


@pytest.mark.parametrize("source, expected", [
    ("import mujoco\n", {(1, RULE_FORBIDDEN, "mujoco")}),
    ("import mink.tasks as ik\n", {(1, RULE_FORBIDDEN, "mink.tasks")}),
    ("from ..station.base import Pose, Station\n", {(1, RULE_FORBIDDEN, "avail.station.base")}),
    ("from ..evidence import passive\n", {(1, RULE_FORBIDDEN, "avail.evidence.passive")}),
    ("from ..evidence.probes import Tug\n", {(1, RULE_FORBIDDEN, "avail.evidence.probes")}),
    ("from ..zoo import store\n", {(1, RULE_FORBIDDEN, "avail.zoo")}),
    ("import avail.analysis.e2\n", {(1, RULE_FORBIDDEN, "avail.analysis.e2")}),
    ("from avail import demo\n", {(1, RULE_FORBIDDEN, "avail.demo")}),
    ("from .. import agent\n", {(1, RULE_FORBIDDEN, "avail.agent")}),
    ("import importlib\nm = importlib.import_module('avail.zoo.' + 'harness')\n", {(2, RULE_FORBIDDEN, "avail.zoo.harness")}),
    ("from importlib import import_module as load\nm = load('..station.cap_client', __package__)\n",
     {(2, RULE_FORBIDDEN, "avail.station.cap_client")}),
    ("m = __import__('mujoco')\n", {(1, RULE_FORBIDDEN, "mujoco")}),
    ("TARGET = 'avail.analysis.figures:CHECK_NAMES'\n", {(1, RULE_FORBIDDEN, "avail.analysis.figures:CHECK_NAMES")}),
    ("from .. import config\n", {(1, RULE_OUTSIDE, "avail.config")}),
    ("from ..config import ExperimentConfig\n", {(1, RULE_OUTSIDE, "avail.config.ExperimentConfig")}),
    ("import avail.router\n", {(1, RULE_OUTSIDE, "avail.router")}),
    ("from ..walls import scan_source\n", {(1, RULE_OUTSIDE, "avail.walls")}),
])
def test_an_import_of_the_testbed_or_outside_the_core_is_found(source, expected):
    assert found(source) == expected


def test_what_the_core_may_import_passes():
    source = """
        import json
        import numpy as np
        import avail.schemas
        import avail.evidence.fused
        from .. import paths, seeds
        from ..config import env, load_experiment
        from ..core.robot import Station
        from ..evidence import base, fused
        from ..evidence.base import get
        from ..schemas import Verdict
        from . import calibrate
        import importlib
        mod = importlib.import_module(f"{__package__}.{name}")        # built at run time: out of reach
        """
    assert found(source) == set()


def test_the_router_and_the_infrastructure_are_held_to_fewer_rules():
    assert found("from ..config import MissingCredentialError, load_secret\n", "avail/router/core.py") == set()
    assert found("import mujoco\n", "avail/router/core.py") == {(1, RULE_FORBIDDEN, "mujoco")}
    assert found("LATCH = 'latch'\n", "avail/router/core.py") == {(1, RULE_WORD, "latch")}
    assert found("CONNECTORS = ('fric', 'latch')\n", "avail/config.py") == set()
    assert found("from .station import base\n", "avail/config.py") == {(1, RULE_FORBIDDEN, "avail.station")}
    assert found("import mujoco\n", "avail/station/mj_plug/scene.py") == set()       # the testbed is not scanned


@pytest.mark.parametrize("source, line, word", [
    ("def fric_depth():\n    pass\n", 1, "fric"),
    ("class LatchCheck:\n    pass\n", 1, "latch"),
    ("def f(prong=1):\n    return prong\n", 1, "prong"),
    ("f(latch=1)\n", 1, "latch"),
    ("x = station.mj_plug_hard\n", 1, "mj_plug"),
    ("from typing import NominalGeometry\n", 1, "NominalGeometry"),
    ("N5 = 1\n", 1, "N5"),
    ("try:\n    pass\nexcept ValueError as latch_error:\n    pass\n", 3, "latch"),
    ("x = 'repair:push_seat'\n", 1, "push_seat"),
    ("x = f'{kind}:push_seat'\n", 1, "push_seat"),
    ("x = {'P0': 'seated'}\n", 1, "P0"),
    ("raise ValueError('the latch is open')\n", 1, "latch"),
    ("x = 'MjPlugStation'\n", 1, "mj_plug"),
    ("x = 'B/probe.push_seat/_y'\n", 1, "push_seat"),
    ("task = 'plug_cycle'\n", 1, "plug_cycle"),
    ("names = ['nominal_geometry']\n", 1, "NominalGeometry"),
])
def test_a_plug_word_in_a_name_or_a_string_is_found(source, line, word):
    assert (line, RULE_WORD, word) in found(source)


def test_documentation_may_hold_plug_words():
    source = '''
        """A module about the latch and the fric connector, N1 to N9 and P0."""

        from pydantic import Field

        X = 1
        """The attribute docstring of X names probe.push_seat."""


        class Spec:
            """A class docstring names mj_plug."""

            name: str = Field(description="the connector, for example latch")


        def f():
            """A function docstring names NominalGeometry."""
            # a comment names plug_cycle
            parser.add_argument("--x", help="the fric connector")
        '''
    assert found(source) == set()


@pytest.mark.parametrize("text", ["friction", "latched", "unlatched", "p0", "n1", "N10", "plugged", "a plug",
                                  "prongs", "fricative", "NominalGeo", "push seat", "seated", "P01"])
def test_a_word_that_only_looks_like_a_plug_word_is_free(text):
    assert plug_words_in(text) == [], text


def test_words_split_names_at_underscores_and_capitals():
    assert words("MjPlugStation") == [["Mj", "Plug", "Station"]]
    assert words("HTTPServer x_y2") == [["HTTP", "Server"], ["x", "y2"]]
    assert plug_words_in("witness_mj_plug_latch_latched") == ["latch", "mj_plug"]
    assert plug_words_in("cls_N1, P0Seated, LATCH_FORCE") == ["N1", "P0", "latch"]


def test_a_file_that_does_not_parse_is_a_finding():
    (finding,) = scan_source("def f(:\n", "avail/fork/policy.py")
    assert finding.rule == RULE_UNPARSEABLE and finding.line == 1


def test_a_core_folder_with_no_python_file_is_a_finding(tmp_path):
    for place in CORE_DIRS + SHARED_DIRS:
        (tmp_path / place).mkdir(parents=True)
        (tmp_path / place / "__init__.py").write_text('"""A package."""\n', encoding="utf-8")
    for name in CORE_FILES + INFRA_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("import json\n", encoding="utf-8")
    assert find_findings(tmp_path) == []
    (tmp_path / "avail/fork/__init__.py").unlink()
    (tmp_path / "avail/evidence/fused.py").unlink()
    assert {(f.rel, f.rule) for f in find_findings(tmp_path)} == {("avail/fork", RULE_MISSING),
                                                                  ("avail/evidence/fused.py", RULE_MISSING)}


# ============================================================================ the moves of S5
def test_the_robot_types_keep_their_old_address():
    from avail.core import robot
    from avail.evidence import base as evidence_base
    from avail.station import base as station_base

    for name in robot.__all__:
        assert getattr(station_base, name) is getattr(robot, name), name
    for name in ("CameraFrame", "MotionResult", "Observation", "Pose", "RobotState", "Station"):
        assert getattr(evidence_base, name) is getattr(robot, name), name
    assert hasattr(station_base, "NominalGeometry") and not hasattr(robot, "NominalGeometry")


def test_the_plug_cascade_and_the_zoo_figures_live_in_the_testbed():
    from avail.analysis import figures
    from avail.shift import report
    from avail.witness import cascade
    from avail.zoo import stages

    assert not hasattr(cascade, "PLUG_STAGES") and not hasattr(cascade, "SKIP_FEATURES")
    assert [d.name for d in stages.PLUG_STAGES] == ["passive", "push", "tug", "look"]
    assert stages.SKIP_FEATURES == ("port_index_est",) and stages.STATION_ID == "mj_plug"
    for name in ("CLASS_NAMES", "CHECK_NAMES", "fig_far_by_check", "fig_class_grid", "md_evidence_table", "_mpl"):
        assert hasattr(figures, name), name
        assert not hasattr(report, name), name


def _two_stage_table(n: int, seed: int) -> pd.DataFrame:
    """2 stages over 1 source each. ``A/src.a/idx`` is bookkeeping: an index that measures nothing."""
    r = np.random.default_rng(seed)
    y0 = r.random(n) < 0.5
    y1 = y0 | (r.random(n) < 0.7)
    return pd.DataFrame({
        "y0": y0, "A/src.a/x": np.where(y0, 1.0, 0.0) + r.normal(0, 1, n), "A/src.a/idx": r.integers(1, 9, n).astype(float),
        "A/src.a/_seconds": 0.0, "B/src.b/x": np.where(y1, 4.0, 0.0) + r.normal(0, 1, n), "B/src.b/_y": y1,
        "B/src.b/_seconds": 1.0,
    })


def test_the_cascade_takes_its_stages_skipped_features_and_station_as_arguments():
    from avail.witness.cascade import UNKNOWN_STATION, Cascade, StageDef, calibrate_cascade, fit_scorers

    defs = (StageDef("first", ("src.a",), "A", "y0"), StageDef("second", ("src.b",), "B", "B/src.b/_y"))
    fit, cal = _two_stage_table(3000, 1), _two_stage_table(2000, 2)
    with pytest.raises(TypeError):
        fit_scorers(fit, "v1", "placed")                              # no stage definitions: no default stages
    every, _, _ = fit_scorers(fit, "v1", "placed", defs)
    scorers, neg, pos = fit_scorers(fit, "v1", "placed", defs, ("idx",))
    assert "src.a/idx" in every[0].features and "src.a/idx" not in scorers[0].features
    with pytest.raises(TypeError):
        calibrate_cascade(scorers, neg, pos, cal, "v1", "placed", 0.05, 0.05, defs)      # no station id
    cascade = calibrate_cascade(scorers, neg, pos, cal, "v1", "placed", 0.05, 0.05, defs, station_id="cell_7",
                                skip_features=("idx",))
    assert cascade.station_id == "cell_7" and cascade.skip_features == ("idx",)
    assert cascade.to_witness().scope.station_id == "cell_7"
    stored = cascade.to_dict()
    again = Cascade.from_dict(stored)
    assert again.stage_defs == defs and again.skip_features == ("idx",) and again.station_id == "cell_7"
    assert again.decide_table(cal).equals(cascade.decide_table(cal))
    # a file written before S5 lists no skipped features, and one may lack a station id: same decisions
    old = {k: v for k, v in stored.items() if k not in ("skip_features", "station_id")}
    older = Cascade.from_dict(old)
    assert older.station_id == UNKNOWN_STATION and older.skip_features == ()
    assert older.decide_table(cal).equals(cascade.decide_table(cal))
    with pytest.raises(ValueError, match="stage definitions"):
        Cascade("placed", "v1", 0.05, 0.05, cascade.stages, station_id="cell_7", stage_defs=defs[:1])
    with pytest.raises(ValueError, match="station_id"):
        Cascade("placed", "v1", 0.05, 0.05, cascade.stages, station_id=" ", stage_defs=defs)
    assert math.isfinite(sum(s.alpha_k for s in cascade.stages))
