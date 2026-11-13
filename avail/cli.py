"""The ``avail`` command line.

Each subcommand lives in its own module and is imported only when it runs, so
``avail trials`` never pays for MuJoCo. A command module exposes
``main(argv) -> int`` and builds its own argument parser.
"""

from __future__ import annotations

import importlib
import sys

# name -> (module, one-line help, session that builds it)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "smoke": ("avail.station.feasibility", "load, render and move the YAM station (the S0 feasibility spike)", "S0"),
    "trials": ("avail.stats.cli", "trial counts for comparing or certifying success rates", "S1"),
    "schemas": ("avail.schemas.export", "write the JSON Schema of every record type", "S1"),
    "models": ("avail.router.smoke", "the model table, and a live smoke call per model when a key is set", "S1"),
    "station": ("avail.station.mj_plug.cli", "the plug station: acceptance checks, oracle cycle, videos", "S2"),
    "zoo": ("avail.zoo.cli", "generate near-miss states and run the evidence sources over them", "S2"),
    "e1e2": ("avail.analysis.cli", "calibrate, then compute the evidence table (E1) and the bound (E2)", "S4"),
    "demo": ("avail.demo", "a small zoo, the free checks and an HTML report in under 2 minutes, no key", "S4"),
    "s4b": ("avail.analysis.s4b_cli", "hard mode: camera measurement and picture pairs on dev seeds", "S4b"),
    "s4c": ("avail.analysis.s4c_cli", "hard mode v2: v1 and v2 side by side, and what the click settles", "S4c"),
    "shift": ("avail.shift.cli", "run a task's cycles on its testbed, report the shift from the ledger, export interventions", "S5"),
    "proxy": ("avail.agent.proxy", "AVAIL in front of another MCP server, such as ASPIRE's CAP server", "S5c"),
    "audit-check": ("avail.audit", "how often your success check passes bad states, by failure type, on the zoo", "S5c"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: avail <command> [options]", "", "commands:"]
    for name, (_, help_text, session) in COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {help_text}")
    lines += ["", "Run 'avail <command> --help' for the options of one command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"avail: unknown command {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module_name, _, session = COMMANDS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("avail"):
            print(f"avail {name}: not built yet (session {session}).", file=sys.stderr)
            return 2
        raise
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
