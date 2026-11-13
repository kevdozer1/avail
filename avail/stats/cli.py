"""``avail trials``: how many trials a comparison or a certificate needs.

    avail trials --p0 0.95 --p1 0.99
    avail trials --certify 0.005
    avail trials --certify 0.005 --allowed-failures 16

Each answer is plain text. The count stands alone on its own line, so a script
can read it, and the formula follows with the numbers plugged in.
"""

from __future__ import annotations

import argparse
import math
import sys

from . import _validate as check
from .intervals import cp_upper, n_to_certify
from .planning import _two_proportion_terms

PROG = "avail trials"

_EXAMPLES = """examples:
  avail trials --p0 0.95 --p1 0.99
      trials per arm to tell a 95% success rate from 99%  (285)
  avail trials --certify 0.005
      clean trials that certify a failure rate of at most 0.5%  (598)
  avail trials --certify 0.005 --allowed-failures 16
      trials that certify 0.5% with up to 16 failures  (4857)
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Trial counts for comparing 2 success rates and for certifying a failure rate.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare = parser.add_argument_group("compare 2 success rates (two-sided two-proportion z-test)")
    compare.add_argument("--p0", type=float, metavar="RATE", help="the first success rate, for example 0.95")
    compare.add_argument("--p1", type=float, metavar="RATE", help="the second success rate, for example 0.99")
    compare.add_argument("--alpha", type=float, metavar="A", help="significance level (default 0.05)")
    compare.add_argument("--power", type=float, metavar="P", help="chance of detecting the difference (default 0.8)")
    certify = parser.add_argument_group("certify a failure rate (one-sided Clopper-Pearson upper bound)")
    certify.add_argument("--certify", type=float, metavar="P_MAX",
                         help="the failure rate to certify, for example 0.005")
    certify.add_argument("--allowed-failures", type=int, metavar="K",
                         help="failures the certificate may contain (default 0)")
    certify.add_argument("--conf", type=float, metavar="C", help="confidence of the certificate (default 0.95)")
    return parser


def _num(x: float) -> str:
    """A number as a person would type it, so 1 - 0.95 prints as 0.05."""
    return f"{x:.6g}"


def compare_lines(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> list[str]:
    """The printed answer for telling success rate ``p0`` from ``p1``."""
    t = _two_proportion_terms(PROG, p0, p1, alpha, power, names=("--p0", "--p1", "--alpha", "--power"))
    p0, p1, n = t["p1"], t["p2"], t["n"]
    z_a, z_b = f"{t['z_a']:.4f}", f"{t['z_b']:.4f}"
    return [
        f"Trials per arm to tell a success rate of {_num(p0)} from {_num(p1)}",
        f"Two-sided two-proportion z-test, alpha = {_num(t['alpha'])}, power = {_num(t['power'])}",
        "",
        str(n),
        "",
        "Formula",
        "  n = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p0 * (1 - p0) + p1 * (1 - p1)))^2 / (p0 - p1)^2",
        "  pbar = (p0 + p1) / 2, z_a = z(1 - alpha / 2), z_b = z(power), z = the standard normal quantile",
        "With the numbers",
        f"  pbar = ({_num(p0)} + {_num(p1)}) / 2 = {_num(t['pbar'])}",
        f"  z_a = z({_num(1.0 - t['alpha'] / 2.0)}) = {z_a}",
        f"  z_b = z({_num(t['power'])}) = {z_b}",
        f"  n = ({z_a} * sqrt(2 * {_num(t['pbar'])} * {_num(1.0 - t['pbar'])}) + {z_b} * "
        f"sqrt({_num(p0)} * {_num(1.0 - p0)} + {_num(p1)} * {_num(1.0 - p1)}))^2 / ({_num(p0)} - {_num(p1)})^2",
        f"    = {t['n_exact']:.2f}, rounded up to {n} per arm ({2 * n} in total)",
    ]


def certify_lines(p_max: float, allowed_failures: int = 0, conf: float = 0.95) -> list[str]:
    """The printed answer for certifying a failure rate of at most ``p_max``."""
    p_max = check.fraction(PROG, "--certify", p_max, example="0.005 for a 0.5% bar")
    k = check.count(PROG, "--allowed-failures", allowed_failures, 0,
                    "Pass the number of failures the certificate may contain, 0 or more.")
    conf = check.fraction(PROG, "--conf", conf, example="0.95")
    n = n_to_certify(p_max, k, conf)
    at_n = cp_upper(k, n, conf)
    lines = [
        f"Trials to certify a failure rate of at most {_num(p_max)} with {k} "
        f"{'failure' if k == 1 else 'failures'} allowed",
        f"One-sided Clopper-Pearson upper bound, conf = {_num(conf)}",
        "",
        str(n),
        "",
        "Formula",
    ]
    if k == 0:
        exact = math.log1p(-conf) / math.log1p(-p_max)
        lines += [
            "  n = the smallest count whose upper bound is at most p_max",
            "  with 0 failures the upper bound is 1 - (1 - conf)^(1 / n), so n = ln(1 - conf) / ln(1 - p_max), rounded up",
            "With the numbers",
            f"  n = ln({_num(1.0 - conf)}) / ln({_num(1.0 - p_max)}) = {exact:.2f}, rounded up to {n}",
        ]
    else:
        lines += [
            "  n = the smallest count with cp_upper(k, n, conf) <= p_max",
            "  cp_upper(k, n, conf) = the failure rate at which k or fewer failures in n trials has probability 1 - conf",
            "With the numbers",
        ]
    lines.append(f"  upper bound after {k} in {n}: {_num(at_n)}, at most {_num(p_max)}")
    if n - 1 >= max(k, 1):
        lines.append(f"  upper bound after {k} in {n - 1}: {_num(cp_upper(k, n - 1, conf))}, above {_num(p_max)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run ``avail trials``. Returns 0 on success and 2 on a bad command line."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:                   # argparse exits on --help and on a flag it cannot read
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)

    compare = args.p0 is not None or args.p1 is not None
    certify = args.certify is not None
    problem = None
    if not compare and not certify:
        problem = "nothing to compute. Pass --p0 and --p1 to compare 2 success rates, or --certify to plan a certificate."
    elif compare and (args.p0 is None or args.p1 is None):
        problem = "--p0 and --p1 go together. Pass both success rates, for example --p0 0.95 --p1 0.99."
    elif not compare and (args.alpha is not None or args.power is not None):
        problem = "--alpha and --power apply to a comparison. Pass --p0 and --p1 with them."
    elif not certify and (args.allowed_failures is not None or args.conf is not None):
        problem = "--allowed-failures and --conf apply to a certificate. Pass --certify with them."
    if problem is None:
        try:
            blocks = []
            if compare:
                blocks.append(compare_lines(
                    args.p0, args.p1,
                    0.05 if args.alpha is None else args.alpha,
                    0.8 if args.power is None else args.power,
                ))
            if certify:
                blocks.append(certify_lines(
                    args.certify,
                    0 if args.allowed_failures is None else args.allowed_failures,
                    0.95 if args.conf is None else args.conf,
                ))
        except ValueError as exc:
            problem = str(exc).removeprefix(f"{PROG}: ")
    if problem is not None:
        print(f"{PROG}: {problem}", file=sys.stderr)
        print(f"Run '{PROG} --help' for the options.", file=sys.stderr)
        return 2
    print("\n\n".join("\n".join(block) for block in blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
