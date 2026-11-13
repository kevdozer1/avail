"""The static half of the done gate (CLAUDE.md rule 11).

Only evidence or a person marks a step done. At run time the record types refuse
an accept with no evidence and refuse ``accept`` anywhere in a fork. The rules here
read the source instead, so a code path that would let free text decide is caught
before it runs. ``tests/test_done_gate.py`` runs them on the whole package.

The rules:

1. Outside ``avail/witness/`` and ``avail/schemas/``, no code under ``avail/`` builds a
   verdict: no call to ``Verdict(...)`` or ``schemas.Verdict(...)``, no subclass of it,
   no ``model_construct``, and no ``Verdict.model_validate`` on a literal written in the
   code. Reading a logged verdict back with ``Verdict.model_validate(row)`` is fine anywhere.
2. Outside those 2 packages, nothing assigns to an attribute named ``decision`` (also
   through ``setattr`` or ``__dict__``) and nothing copies a record with a new decision
   (``model_copy(update={"decision": ...})``).
3. Under ``avail/fork/`` and ``avail/agent/``, no function takes a parameter that names
   a decision as free text: ``decision``, ``verdict_text``, ``answer_text`` and names
   like them, typed ``str`` or left untyped. A parameter typed ``Verdict`` or ``Decision``
   is fine.

This is a lint for honest code. It reads names and shapes, so code that hides what it
does (``getattr``, ``exec``, a dict built elsewhere) gets past it. The run-time checks
in :mod:`avail.schemas.records` do not depend on it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BUILDERS", "DECIDERS", "PACKAGE_DIR", "RULES", "GateViolation", "package_sources", "scan_source", "scan_tree",
]

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE = "avail"
BUILDERS = ("witness", "schemas")
"""The only packages under ``avail/`` that may build a verdict or set a decision."""
DECIDERS = ("fork", "agent")
"""The packages under ``avail/`` whose functions must not take a decision as free text."""

RULES: dict[str, str] = {
    "verdict_built": "Ask the witness for the verdict (avail.witness builds it from evidence). "
                     "To read a logged verdict use Verdict.model_validate(row)",
    "verdict_subclassed": "Use Verdict itself. A subclass could drop the check that an accept needs evidence",
    "decision_assigned": "A decision is set once, by the witness, when the Verdict is built. If this attribute is "
                         "something else, give it another name, such as outcome",
    "decision_updated": "A decision is set once, by the witness, when the Verdict is built. Do not copy a record "
                        "with a new one",
    "text_decision_param": "Take a Verdict, or type the parameter as Decision. Free text never becomes a decision, "
                           "and a decider picks a fork option",
    "unparsed": "Fix the syntax. A file the gate rules cannot read cannot be vouched for",
}
"""Each rule's name and how to fix a violation of it."""

_DECISION = "decision"
_CONSTRUCTORS = ("model_construct", "construct")
_VALIDATORS = ("model_validate", "model_validate_json", "model_validate_strings", "parse_obj", "parse_raw")
_COPIERS = ("model_copy", "copy", "replace", "__replace__")
_WORDS = "decision|verdict|answer"
_TEXT_DECISION_PARAM = re.compile(
    rf"(?:decision|verdict)s?"                                   # decision, verdict
    rf"|(?:{_WORDS})_(?:text|str|string|raw|json)"               # verdict_text, answer_text, decision_str
    rf"|(?:raw|text|free|model|llm|vlm|judge)_(?:{_WORDS})",     # raw_decision, model_answer, llm_verdict
    re.IGNORECASE,
)
_FREE_TEXT_TYPES = frozenset({"str", "Any", "object", "AnyStr", "Text", "bytes"})


@dataclass(frozen=True)
class GateViolation:
    """One place where the source breaks a gate rule."""

    path: str
    line: int
    rule: str
    what: str

    @property
    def fix(self) -> str:
        return RULES[self.rule]

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.what}. Fix: {self.fix}."


# ------------------------------------------------------------------ small ast helpers
def _is_text(node: ast.AST | None, text: str, any_case: bool = False) -> bool:
    """True when ``node`` is the string literal ``text``."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    return (node.value.strip().casefold() if any_case else node.value) == text


def _is_dict_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"


def _is_literal_mapping(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Dict) or _is_dict_call(node)


def _literal_sets_decision(node: ast.AST | None) -> bool:
    """True for a dict written in the code with a ``decision`` key: ``{"decision": ...}`` or ``dict(decision=...)``."""
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if _is_text(key, _DECISION) or (key is None and _literal_sets_decision(value)):  # None marks a ** spread
                return True
        return False
    if _is_dict_call(node):
        return _passes_decision(node) or any(_literal_sets_decision(arg) for arg in node.args)
    return False


def _passes_decision(call: ast.Call) -> bool:
    """True when a call passes ``decision=...``, by name or by spreading a dict written in the code."""
    return any(
        keyword.arg == _DECISION or (keyword.arg is None and _literal_sets_decision(keyword.value))
        for keyword in call.keywords
    )


def _is_instance_dict(node: ast.AST) -> bool:
    """``x.__dict__`` or ``vars(x)``: the place an attribute lives."""
    if isinstance(node, ast.Attribute) and node.attr == "__dict__":
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "vars"


def _is_free_text(annotation: ast.AST | None) -> bool:
    """True when a parameter's annotation lets free text in. No annotation counts as free text."""
    if annotation is None:
        return True
    if isinstance(annotation, ast.Constant):
        if annotation.value is None:
            return False
        if isinstance(annotation.value, str):  # a quoted annotation such as "str | None"
            try:
                return _is_free_text(ast.parse(annotation.value, mode="eval").body)
            except SyntaxError:
                return True
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id in _FREE_TEXT_TYPES
    if isinstance(annotation, ast.Attribute):
        return annotation.attr in _FREE_TEXT_TYPES
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_free_text(annotation.left) or _is_free_text(annotation.right)
    if isinstance(annotation, ast.Subscript):
        outer = annotation.value
        name = outer.id if isinstance(outer, ast.Name) else outer.attr if isinstance(outer, ast.Attribute) else ""
        if name == "Literal":
            return False
        inner = annotation.slice
        parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
        if name == "Annotated":
            parts = parts[:1]
        return any(_is_free_text(part) for part in parts)
    return False


def _verdict_names(tree: ast.AST) -> set[str]:
    """Every name this file binds to the Verdict class: the import, ``as`` names and plain aliases."""
    names = {"Verdict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "Verdict")

    def refers(node: ast.AST | None) -> bool:
        return (isinstance(node, ast.Name) and node.id in names) or (
            isinstance(node, ast.Attribute) and node.attr == "Verdict")

    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and refers(node.value):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and refers(node.value):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    grew = True
    return names


# ------------------------------------------------------------------ the scanner
class _Scanner(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.AST, check_builds: bool, check_params: bool) -> None:
        self.path = path
        self.check_builds = check_builds
        self.check_params = check_params
        self.verdict_names = _verdict_names(tree)
        self.found: list[GateViolation] = []

    def _add(self, node: ast.AST, rule: str, what: str) -> None:
        self.found.append(GateViolation(self.path, getattr(node, "lineno", 0), rule, what))

    def _is_verdict(self, node: ast.AST | None) -> bool:
        return (isinstance(node, ast.Name) and node.id in self.verdict_names) or (
            isinstance(node, ast.Attribute) and node.attr == "Verdict")

    # -- rule 1 and the call forms of rule 2
    def visit_Call(self, node: ast.Call) -> None:
        if self.check_builds:
            self._check_call(node)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call) -> None:
        func = node.func
        if self._is_verdict(func):
            self._add(node, "verdict_built", "builds a Verdict by calling Verdict(...)")
            return
        first = node.args[0] if node.args else None
        if isinstance(func, ast.Attribute):
            on_verdict = self._is_verdict(func.value)
            if func.attr in _CONSTRUCTORS and (on_verdict or _passes_decision(node)):
                self._add(node, "verdict_built", f"builds a record with {func.attr}, which skips validation")
            elif func.attr in _VALIDATORS and on_verdict and _is_literal_mapping(first):
                self._add(node, "verdict_built",
                          f"builds a Verdict from a literal with Verdict.{func.attr}, which is not reading a logged one")
            elif func.attr in _COPIERS and self._copies_with_decision(node):
                self._add(node, "decision_updated", f"copies a record with a new decision through {func.attr}")
            elif func.attr == "__setattr__" and any(_is_text(arg, _DECISION) for arg in node.args):
                self._add(node, "decision_assigned", "sets the attribute decision through __setattr__")
            elif func.attr == "update" and _is_instance_dict(func.value) and (
                    _passes_decision(node) or any(_literal_sets_decision(arg) for arg in node.args)):
                self._add(node, "decision_assigned", "sets the attribute decision through __dict__")
        elif isinstance(func, ast.Name):
            if func.id == "setattr" and len(node.args) >= 2 and _is_text(node.args[1], _DECISION):
                self._add(node, "decision_assigned", "sets the attribute decision through setattr")
            elif func.id == "replace" and _passes_decision(node):
                self._add(node, "decision_updated", "copies a record with a new decision through replace")
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name == "validate_record" and len(node.args) >= 2 and _is_literal_mapping(node.args[1]) and (
                _is_text(first, "verdict", any_case=True) or self._is_verdict(first)):
            self._add(node, "verdict_built",
                      "builds a Verdict from a literal with validate_record, which is not reading a logged one")

    @staticmethod
    def _copies_with_decision(node: ast.Call) -> bool:
        if _passes_decision(node):
            return True
        return any(keyword.arg == "update" and _literal_sets_decision(keyword.value) for keyword in node.keywords)

    # -- rule 2: assignment
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.check_builds and node.attr == _DECISION and isinstance(node.ctx, (ast.Store, ast.Del)):
            self._add(node, "decision_assigned", "assigns to an attribute named decision")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (self.check_builds and isinstance(node.ctx, (ast.Store, ast.Del)) and _is_text(node.slice, _DECISION)
                and _is_instance_dict(node.value)):
            self._add(node, "decision_assigned", "sets the attribute decision through __dict__")
        self.generic_visit(node)

    # -- rule 1: subclassing
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.check_builds and any(self._is_verdict(base) for base in node.bases):
            self._add(node, "verdict_subclassed", f"class {node.name} subclasses Verdict")
        self.generic_visit(node)

    # -- rule 3: parameters
    def _check_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, name: str) -> None:
        if not self.check_params:
            return
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        params += [extra for extra in (args.vararg, args.kwarg) if extra is not None]
        for param in params:
            if _TEXT_DECISION_PARAM.fullmatch(param.arg) and _is_free_text(param.annotation):
                typed = "untyped" if param.annotation is None else f"typed {ast.unparse(param.annotation)}"
                self._add(param, "text_decision_param",
                          f"{name} takes the parameter {param.arg} ({typed}), which reads as a decision in free text")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_params(node, f"function {node.name}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_params(node, f"function {node.name}")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._check_params(node, "a lambda")
        self.generic_visit(node)


def _subpackage(path: str) -> str:
    """``fork`` for ``avail/fork/policy.py``, and an empty string for a file directly under ``avail/``."""
    parts = path.replace("\\", "/").strip("/").split("/")
    if parts[0] != PACKAGE:
        raise ValueError(
            f"{path!r} is not a path under {PACKAGE}/. Fix: give the path relative to the repo root, "
            f"for example {PACKAGE}/fork/policy.py."
        )
    return parts[1] if len(parts) > 2 else ""


def scan_source(source: str, path: str) -> list[GateViolation]:
    """The gate violations in one file's source.

    ``path`` is the file's path relative to the repo root, for example ``avail/fork/policy.py``.
    It decides which rules apply: rules 1 and 2 everywhere except ``avail/witness/`` and
    ``avail/schemas/``, rule 3 under ``avail/fork/`` and ``avail/agent/``.
    """
    path = path.replace("\\", "/")
    subpackage = _subpackage(path)
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError) as exc:  # a null byte is a ValueError on some versions
        reason = exc.msg if isinstance(exc, SyntaxError) else f"{type(exc).__name__}: {exc}"
        return [GateViolation(path, getattr(exc, "lineno", None) or 0, "unparsed", f"the file does not parse ({reason})")]
    scanner = _Scanner(path, tree, check_builds=subpackage not in BUILDERS, check_params=subpackage in DECIDERS)
    scanner.visit(tree)
    return sorted(scanner.found, key=lambda found: (found.line, found.rule))


def package_sources(package_dir: Path | None = None) -> Iterator[tuple[str, Path]]:
    """``(path relative to the repo root, file)`` for every Python file of the package, in a stable order."""
    root = Path(package_dir) if package_dir is not None else PACKAGE_DIR
    for file in sorted(root.rglob("*.py")):
        yield f"{PACKAGE}/{file.relative_to(root).as_posix()}", file


def scan_tree(package_dir: Path | None = None) -> list[GateViolation]:
    """The gate violations in every Python file under ``avail/`` (or under ``package_dir``, read as ``avail/``)."""
    found: list[GateViolation] = []
    for path, file in package_sources(package_dir):
        try:
            source = file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            found.append(GateViolation(path, 0, "unparsed", f"the file cannot be read ({type(exc).__name__})"))
            continue
        found.extend(scan_source(source, path))
    return found
