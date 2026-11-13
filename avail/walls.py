"""The truth wall: a static scan that keeps checks away from simulator truth.

CLAUDE.md rule 10 says evidence sources, witnesses, fork policies and agents see
camera images, proprioception and nominal geometry only. They never read
privileged simulator state. Only the evaluator, the audit sampler and
``OracleHuman`` may call ``truth()``.

This module enforces that rule on source text with the ``ast`` module. Nothing
is imported and nothing runs, so the scan works on stubs, on half-built modules
and on snippets inside a test.

Guarded files: every ``.py`` file under ``avail/evidence/``, ``avail/witness/``
and ``avail/agent/``, plus the single file ``avail/fork/policy.py``.

A guarded file must not:

1. Import the truth module ``avail.station.mj_plug.truth`` in any form:
   absolute, relative, aliased, or through ``importlib.import_module`` or
   ``__import__`` with a string that names it. Rule ``truth_import``.
2. Import ``mujoco``, or anything at all from ``avail.station.mj_plug``. The sim
   backend is reached only through the Station protocol in
   ``avail/station/base.py``. Rule ``sim_import``.
3. Call anything named ``truth`` or reach for an attribute named ``truth``, as
   in ``truth(station)``, ``station.truth()`` or ``x.truth``. Rule
   ``truth_access``.
4. Touch an attribute named ``_mj_model``, ``_mj_data``, ``mj_model``,
   ``mj_data``, ``MjData`` or ``MjModel`` on any object. Rule ``mj_handle``.
5. Touch ``data``, ``model`` or any private attribute on a station, meaning an
   object named ``station``, ``stn``, ``self.station`` or ``self._station``.
   Any owner of those names counts, with any leading underscores. Private
   means a name that starts with an underscore: ``_data``, ``_model``,
   ``_sim`` and every other one. A station has no private part in its
   protocol, and the sim station keeps its layout, its scene and its true
   forces behind such names. Rule ``station_internals``.

The wall fails closed. A guarded file that cannot be parsed is a violation
(rule ``unparseable``). A guarded folder that holds no Python files is a
violation too (rule ``guard_missing``), because a wall that scans nothing would
pass forever.

The same access in another spelling is the same violation. ``getattr(station,
"_mj_data")`` counts as ``station._mj_data``, and so does the class pattern
``case object(_mj_data=d)`` in a ``match station`` block. A string that is
exactly the name of the sim backend or of a module inside it counts as an
import of it, which covers ``sys.modules[...]`` and tables of dotted paths. So
does the attribute path ``avail.station.mj_plug``. Importing a name called
``truth`` from any module counts as importing truth, which covers a re-export.

What a static scan cannot see: it follows names and never values. After
``s = station`` the line ``s.data`` passes. So does a module path assembled at
run time, and so does anything behind ``exec``, ``eval``, ``vars()`` or
``__dict__``. There is no ignore comment. Guarded code still gets read by a
person.

Use it from a test with :func:`find_wall_violations`, on a snippet with
:func:`scan_source`, or by hand with ``python -m avail.walls``.
"""

from __future__ import annotations

import argparse
import ast
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import REPO_ROOT

__all__ = [
    "GUARDED_DIRS", "GUARDED_FILES", "RULES", "WALL_RULES", "STATION_ADVICE",
    "RULE_TRUTH_IMPORT", "RULE_SIM_IMPORT", "RULE_TRUTH_ACCESS", "RULE_MJ_HANDLE", "RULE_STATION_INTERNALS",
    "RULE_UNPARSEABLE", "RULE_GUARD_MISSING",
    "Violation", "scan_source", "find_wall_violations", "is_guarded", "guarded_files", "main",
]

# ----------------------------------------------------------------------------- what is guarded
GUARDED_DIRS: tuple[str, ...] = ("avail/evidence", "avail/witness", "avail/agent")
GUARDED_FILES: tuple[str, ...] = ("avail/fork/policy.py",)

# ----------------------------------------------------------------------------- what is off limits
SIM_BACKEND_PACKAGE = "avail.station.mj_plug"
SIM_ENGINE_PACKAGE = "mujoco"
TRUTH_NAME = "truth"
MJ_HANDLE_ATTRS = frozenset({"_mj_model", "_mj_data", "mj_model", "mj_data", "MjData", "MjModel"})
STATION_HANDLE_ATTRS = frozenset({"data", "model"})        # plus every private name, see _is_private
STATION_NAMES = frozenset({"station", "stn"})              # with any leading underscores, see _is_station

# ----------------------------------------------------------------------------- rule names
RULE_TRUTH_IMPORT = "truth_import"
RULE_SIM_IMPORT = "sim_import"
RULE_TRUTH_ACCESS = "truth_access"
RULE_MJ_HANDLE = "mj_handle"
RULE_STATION_INTERNALS = "station_internals"
RULE_UNPARSEABLE = "unparseable"
RULE_GUARD_MISSING = "guard_missing"

#: The 5 ways a guarded file can cross the wall.
WALL_RULES: tuple[str, ...] = (
    RULE_TRUTH_IMPORT, RULE_SIM_IMPORT, RULE_TRUTH_ACCESS, RULE_MJ_HANDLE, RULE_STATION_INTERNALS,
)
#: Every rule a :class:`Violation` can carry.
RULES: tuple[str, ...] = WALL_RULES + (RULE_UNPARSEABLE, RULE_GUARD_MISSING)

#: What guarded code does in place of the forbidden access. Every wall message ends with it.
STATION_ADVICE = (
    "Use the Station protocol in avail/station/base.py instead: "
    "observe(), robot_state(), move_*, set_gripper()."
)

_TRUTH_IS_PRIVATE = (
    "Only the evaluator, the audit sampler and OracleHuman may read ground truth. "
    "A label reaches guarded code as data."
)
_WHY = {
    RULE_TRUTH_IMPORT: _TRUTH_IS_PRIVATE,
    RULE_TRUTH_ACCESS: _TRUTH_IS_PRIVATE,
    RULE_SIM_IMPORT: "Guarded code reaches the simulator only through a Station that is handed to it.",
    RULE_MJ_HANDLE: "MuJoCo model and data objects are privileged simulator state.",
    RULE_STATION_INTERNALS: "The internals of a station are privileged simulator state.",
}

_BACKEND_PARTS = SIM_BACKEND_PACKAGE.split(".")
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
_ATTR_BUILTINS = frozenset({"getattr", "hasattr", "setattr", "delattr"})
_DOTTED_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?")
_DRIVE = re.compile(r"^[A-Za-z]:")
_MAX_SHOWN = 60
_BOM = chr(0xFEFF)


@dataclass(frozen=True, order=True)
class Violation:
    """One place where a guarded file crosses the truth wall.

    ``rel_path`` is relative to the repo root with forward slashes. ``line`` is
    1-based, and 0 when the finding is about a whole file or folder. ``rule`` is
    one of :data:`RULES`. ``message`` says what was found and what to do instead.
    """

    rel_path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.rel_path}:{self.line}: [{self.rule}] {self.message}"


# ----------------------------------------------------------------------------- paths
def _normalize(rel_path: str | Path) -> str:
    """A repo-relative path with forward slashes and no ``.`` or ``..`` steps."""
    text = str(rel_path).replace("\\", "/")
    if text.startswith("/") or _DRIVE.match(text):
        raise ValueError(
            f"rel_path {str(rel_path)!r} is absolute. Pass the path relative to the repo root, "
            "for example 'avail/evidence/probes.py'."
        )
    return posixpath.normpath(text)


def is_guarded(rel_path: str | Path) -> bool:
    """True when the wall applies to this repo-relative path."""
    rel = _normalize(rel_path)
    if not rel.endswith(".py"):
        return False
    return rel in GUARDED_FILES or any(rel.startswith(folder + "/") for folder in GUARDED_DIRS)


def _root(repo_root: str | Path | None) -> Path:
    return REPO_ROOT if repo_root is None else Path(repo_root)


def guarded_files(repo_root: str | Path | None = None) -> list[str]:
    """Every guarded file that exists under ``repo_root``, as sorted repo-relative paths."""
    root = _root(repo_root)
    found: set[str] = set()
    for folder in GUARDED_DIRS:                                 # a missing folder yields nothing
        found.update(p.relative_to(root).as_posix() for p in (root / folder).rglob("*.py") if p.is_file())
    for name in GUARDED_FILES:
        if (root / name).is_file():
            found.add(name)
    return sorted(found)


# ----------------------------------------------------------------------------- small ast helpers
def _terminal_name(node: ast.AST | None) -> str | None:
    """The last identifier of a name or a dotted expression: ``station`` for ``self.station``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_station(node: ast.AST | None) -> bool:
    """True when an expression is named like a station: ``station``, ``stn``, ``self._station``, ``self.__stn``."""
    name = _terminal_name(node)
    return name is not None and name.lstrip("_") in STATION_NAMES


def _is_private(attr: str | None) -> bool:
    """True for ``_name`` and ``__name``. False for a dunder such as ``__class__``."""
    return bool(attr) and attr.startswith("_") and not (attr.startswith("__") and attr.endswith("__"))


def _const_str(node: ast.AST | None) -> str | None:
    """The string a constant expression spells, or None when it depends on a run-time value.

    Handles a literal, literals joined with ``+`` and an f-string with no fields.
    """
    pieces: list[str] = []
    pending = [node]
    while pending:                                              # a loop, so a long chain cannot overflow the stack
        current = pending.pop()
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            pieces.append(current.value)
        elif isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            pending += [current.right, current.left]            # the left side is read next
        elif isinstance(current, ast.JoinedStr):
            pending += current.values                           # at most 1 piece when there are no fields
        else:
            return None                                         # a name, a field, a missing argument
    return "".join(pieces)


def _show(node: ast.AST) -> str:
    """Source text for an expression, shortened, for a message."""
    try:
        text = ast.unparse(node)
    except Exception:  # a pathological expression still gets reported, just without its text
        return "<expression>"
    return text if len(text) <= _MAX_SHOWN else "..." + text[-(_MAX_SHOWN - 3):]


def _import_rule(dotted: str) -> str | None:
    """The rule an import of this dotted path breaks, or None when the import is allowed."""
    parts = dotted.split(".")
    if TRUTH_NAME in parts:
        return RULE_TRUTH_IMPORT
    if parts[:len(_BACKEND_PARTS)] == _BACKEND_PARTS or parts[0] == SIM_ENGINE_PACKAGE:
        return RULE_SIM_IMPORT
    return None


def _backend_path(text: str) -> str | None:
    """The string, without a trailing dot, when it is exactly the sim backend or a name inside it.

    ``pkg.module:attr`` and a trailing dot are accepted, since both are ways to spell an import
    target. Prose that merely mentions the backend does not match and gives None.
    """
    if text.endswith("."):
        text = text[:-1]
    if not _DOTTED_PATH.fullmatch(text):
        return None
    return text if text.replace(":", ".").split(".")[:len(_BACKEND_PARTS)] == _BACKEND_PARTS else None


# ----------------------------------------------------------------------------- the scan of one file
class _Scan:
    """Walks one parsed guarded file and collects its violations."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.package = rel[: -len(".py")].split("/")[:-1]      # the package that holds the file
        self.found: list[Violation] = []
        self.consumed: set[int] = set()                         # string nodes already judged as imports
        self.importers = {name: name for name in _DYNAMIC_IMPORTERS}

    def run(self, tree: ast.AST) -> list[Violation]:
        nodes = list(ast.walk(tree))                            # breadth order: a call comes before its arguments
        for node in nodes:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _DYNAMIC_IMPORTERS and alias.asname:
                        self.importers[alias.asname] = alias.name
        for node in nodes:
            if isinstance(node, ast.Import):
                self._imports(node.lineno, [(alias.name, _import_rule(alias.name)) for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                self._import_from(node)
            elif isinstance(node, ast.Call):
                self._call(node)
            elif isinstance(node, ast.Attribute):
                self._attribute(node.value, node.attr, getattr(node, "end_lineno", None) or node.lineno)
            elif isinstance(node, ast.Match):
                self._match(node)
            elif isinstance(node, ast.Constant):
                self._string(node)
        return sorted(set(self.found))

    def _add(self, line: int, rule: str, what: str) -> None:
        self.found.append(Violation(self.rel, int(line), rule, f"{what} {_WHY[rule]} {STATION_ADVICE}"))

    # ---- imports
    def _relative(self, level: int, module: str, package: list[str] | None = None) -> str:
        """The absolute dotted path of a relative import made from this file."""
        if level <= 0:
            return module
        parts = self.package if package is None else package
        base = parts[: max(0, len(parts) - (level - 1))]
        return ".".join([*base, module] if module else base)

    def _imports(self, line: int, hits: list[tuple[str, str | None]]) -> None:
        for rule in (RULE_TRUTH_IMPORT, RULE_SIM_IMPORT):
            names = sorted({name for name, hit in hits if hit == rule})
            if names:
                self._add(line, rule, f"imports {', '.join(names)}.")

    def _import_from(self, node: ast.ImportFrom) -> None:
        base = self._relative(node.level, node.module or "")
        hits: list[tuple[str, str | None]] = []
        for alias in node.names:
            name = ".".join(part for part in (base, alias.name) if part)
            rule = _import_rule(name)
            # report the module when the module is the problem, and the name when only the name is
            hits.append((base if base and _import_rule(base) == rule else name, rule))
        self._imports(node.lineno, hits)

    def _dynamic_import(self, node: ast.Call, importer: str) -> None:
        """``import_module(name, package)`` or ``__import__(name, globals, locals, fromlist, level)``."""
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

        def argument(position: int, keyword: str) -> ast.AST | None:
            return node.args[position] if len(node.args) > position else keywords.get(keyword)

        name_node = argument(0, "name")
        name = _const_str(name_node)
        if name is None:
            return                                              # built at run time: out of reach for a static scan
        self._consume(name_node)
        if importer == "import_module":
            package_node = argument(1, "package")
            package = _const_str(package_node)
            if package is not None:
                self._consume(package_node)
            level = len(name) - len(name.lstrip("."))
            targets = [self._relative(level, name[level:], package.split(".") if package else None)]
        else:
            level_node = argument(4, "level")
            literal_level = isinstance(level_node, ast.Constant) and isinstance(level_node.value, int)
            targets = [self._relative(level_node.value if literal_level else 0, name)]
            fromlist = argument(3, "fromlist")
            if isinstance(fromlist, (ast.List, ast.Tuple, ast.Set)):
                items = [_const_str(item) for item in fromlist.elts]
                targets += [f"{targets[0]}.{item}" for item in items if item]
        for rule in (RULE_TRUTH_IMPORT, RULE_SIM_IMPORT):
            names = sorted({target for target in targets if _import_rule(target) == rule})
            if names:
                self._add(node.lineno, rule, f"imports {', '.join(names)} through {importer}().")

    def _consume(self, node: ast.AST | None) -> None:
        if node is not None:
            self.consumed.update(id(sub) for sub in ast.walk(node) if isinstance(sub, ast.Constant))

    def _string(self, node: ast.Constant) -> None:
        if id(node) in self.consumed or not isinstance(node.value, str):
            return
        named = _backend_path(node.value)
        if named:
            self._add(node.lineno, _import_rule(named.replace(":", ".")), f"names {named} in a string.")

    # ---- calls and attributes
    def _call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == TRUTH_NAME:
                self._add(func.lineno, RULE_TRUTH_ACCESS, "calls truth().")
            importer = self.importers.get(func.id)
            if importer:
                self._dynamic_import(node, importer)
            elif func.id in _ATTR_BUILTINS and len(node.args) >= 2:
                self._attribute(node.args[0], _const_str(node.args[1]), node.lineno, via=func.id)
        elif isinstance(func, ast.Attribute) and func.attr in _DYNAMIC_IMPORTERS:
            self._dynamic_import(node, func.attr)

    def _attribute(self, owner: ast.AST, attr: str | None, line: int, via: str | None = None) -> None:
        """Judge ``owner.attr``. ``via`` names the builtin when the access is spelled ``getattr(owner, "attr")``."""
        if attr == TRUTH_NAME:
            rule, verb = RULE_TRUTH_ACCESS, "reaches for"
        elif attr in MJ_HANDLE_ATTRS:
            rule, verb = RULE_MJ_HANDLE, "touches"
        elif (attr in STATION_HANDLE_ATTRS or _is_private(attr)) and _is_station(owner):
            rule, verb = RULE_STATION_INTERNALS, "touches"
        elif attr == _BACKEND_PARTS[-1] and _terminal_name(owner) == _BACKEND_PARTS[-2]:
            rule, verb = RULE_SIM_IMPORT, "reaches the sim backend through"     # station.mj_plug as attributes
        else:
            return
        shown = f"{via}({_show(owner)}, {attr!r})" if via else f"{_show(owner)}.{attr}"
        self._add(line, rule, f"{verb} {shown}.")

    def _match(self, node: ast.Match) -> None:
        """``case Cls(attr=...)`` reads ``attr`` from the subject of the match. Judge it like ``subject.attr``."""
        for case in node.cases:
            for pattern in ast.walk(case.pattern):
                if isinstance(pattern, ast.MatchClass):
                    for attr in pattern.kwd_attrs:
                        self._attribute(node.subject, attr, pattern.lineno)


def _unparseable(rel: str, line: int, detail: str) -> Violation:
    return Violation(
        rel, line, RULE_UNPARSEABLE,
        f"cannot be parsed ({detail}), so the wall cannot check it. Fix the file so that it parses. "
        "A guarded file the scanner cannot read counts as a violation.",
    )


def _scan(source: str | bytes, rel: str) -> list[Violation]:
    """Violations in one guarded file. ``rel`` is already normalized and guarded."""
    if isinstance(source, str) and source.startswith(_BOM):
        source = source[1:]                                     # a byte order mark is not a syntax error
    try:
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, ValueError, RecursionError) as exc:    # RecursionError: code nested too deeply to parse
        detail = getattr(exc, "msg", None) or str(exc) or type(exc).__name__
        return [_unparseable(rel, int(getattr(exc, "lineno", 0) or 0), detail)]
    return _Scan(rel).run(tree)


# ----------------------------------------------------------------------------- public entry points
def scan_source(source: str, rel_path: str) -> list[Violation]:
    """Truth wall violations in one piece of source text.

    ``rel_path`` is the file's path relative to the repo root, for example
    ``avail/evidence/probes.py``. It decides whether the wall applies and how
    relative imports resolve. Source outside the guarded set returns an empty
    list without being parsed. A guarded file with a syntax error returns 1
    ``unparseable`` violation. The result is sorted by line.
    """
    rel = _normalize(rel_path)
    if not is_guarded(rel):
        return []
    return _scan(source, rel)


def find_wall_violations(repo_root: str | Path | None = None) -> list[Violation]:
    """Every truth wall violation in the tree under ``repo_root`` (default: this repo).

    Scans each guarded file and also reports a guarded folder with no Python
    files and a guarded file that is absent. An empty list means the wall holds.
    """
    root = _root(repo_root)
    files = guarded_files(root)
    out: list[Violation] = []
    moved = "If the code moved, update GUARDED_DIRS and GUARDED_FILES in avail/walls.py to follow it."
    for folder in GUARDED_DIRS:
        if not any(name.startswith(folder + "/") for name in files):
            out.append(Violation(folder, 0, RULE_GUARD_MISSING,
                                 f"holds no Python files, so the wall guards nothing here. {moved}"))
    for name in GUARDED_FILES:
        if name not in files:
            out.append(Violation(name, 0, RULE_GUARD_MISSING,
                                 f"does not exist, so the wall guards nothing here. {moved}"))
    for rel in files:
        try:
            source = (root / rel).read_bytes()
        except OSError as exc:
            out.append(_unparseable(rel, 0, f"read failed with {type(exc).__name__}"))
        else:
            out.extend(_scan(source, rel))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    """Print every violation and a 1-line count. Exit code 1 when the wall is broken."""
    parser = argparse.ArgumentParser(prog="python -m avail.walls", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="repo root to scan (default: this repo)")
    args = parser.parse_args(argv)
    violations = find_wall_violations(args.root)
    for violation in violations:
        print(violation)
    n = len(violations)
    print(f"truth wall: {n} violation{'' if n == 1 else 's'}. Guarded files scanned: {len(guarded_files(args.root))}.",
          flush=True)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
