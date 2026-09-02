import argparse

from momentum.backtest import run_backtest
from momentum.config import load_config
from momentum.reporting import generate_tearsheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="momentum")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config_path")
    run_parser.add_argument("--output", default="tearsheet.html")

    args = parser.parse_args(argv)

    config = load_config(args.config_path)
    returns, benchmark_returns = run_backtest(config)
    generate_tearsheet(returns, benchmark_returns, args.output)
    print(f"Tearsheet written to {args.output}")
    return 0
