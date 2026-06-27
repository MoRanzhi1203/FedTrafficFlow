"""Read-only runtime monitor for experiment 1 output directories."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only monitor for experiment 1 runs")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:  # pragma: no cover - platform-dependent fallback
        return f"command failed: {exc}"


def print_status(output_dir: Path) -> None:
    print("=" * 80, flush=True)
    print(f"[monitor] output_dir={output_dir}", flush=True)
    print(f"[monitor] exists={output_dir.exists()}", flush=True)
    if output_dir.exists():
        files = sorted(path.name for path in output_dir.iterdir())
        print(f"[monitor] files={files}", flush=True)
        convergence_path = output_dir / "convergence_history.csv"
        if convergence_path.exists():
            convergence_df = pd.read_csv(convergence_path)
            print("[monitor] convergence tail:", flush=True)
            print(convergence_df.tail(5).to_string(index=False), flush=True)
        print(f"[monitor] main_metrics_exists={(output_dir / 'main_metrics.csv').exists()}", flush=True)
        print(f"[monitor] prediction_samples_exists={(output_dir / 'prediction_samples.csv').exists()}", flush=True)
    print("[monitor] python processes:", flush=True)
    print(run_command(["powershell", "-NoProfile", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime, Path | Format-Table -AutoSize"]), flush=True)
    print("[monitor] gpu:", flush=True)
    print(run_command(["nvidia-smi"]), flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    while True:
        print_status(output_dir)
        if args.once:
            break
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
