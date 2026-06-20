from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge correlation-topology results into legacy summary CSVs.")
    parser.add_argument("--scenario_dir", required=True, type=Path)
    parser.add_argument("--legacy_summary_dir", required=True, type=Path)
    parser.add_argument("--correlation_summary_dir", required=True, type=Path)
    parser.add_argument("--output_summary_dir", required=True, type=Path)

    parser.add_argument("--new_method", default="correlation_topology_neighbor_fill", type=str)
    return parser.parse_args()


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def merge_one_csv(
    legacy_path: Path,
    correlation_path: Path,
    output_path: Path,
    old_method: str,
    new_method: str,
) -> None:
    legacy_df = pd.read_csv(legacy_path)
    correlation_df = pd.read_csv(correlation_path)

    if "method" not in legacy_df.columns:
        legacy_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    if "method" not in correlation_df.columns:
        raise RuntimeError(f"method column missing in correlation summary: {correlation_path}")

    merged = pd.concat(
        [
            legacy_df.loc[legacy_df["method"].astype(str) != old_method].copy(),
            correlation_df.loc[correlation_df["method"].astype(str) == new_method].copy(),
        ],
        ignore_index=True,
    )

    sort_keys = [name for name in ["missing_rate", "method", "chunk_index", "day_index", "group_dimension", "flow_group", "length_group"] if name in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.scenario_dir = ensure_absolute(project_root, args.scenario_dir)
    args.legacy_summary_dir = ensure_absolute(project_root, args.legacy_summary_dir)
    args.correlation_summary_dir = ensure_absolute(project_root, args.correlation_summary_dir)
    args.output_summary_dir = ensure_absolute(project_root, args.output_summary_dir)

    if not args.legacy_summary_dir.exists():
        raise FileNotFoundError(f"legacy summary dir not found: {args.legacy_summary_dir}")
    if not args.correlation_summary_dir.exists():
        raise FileNotFoundError(f"correlation summary dir not found: {args.correlation_summary_dir}")

    args.output_summary_dir.mkdir(parents=True, exist_ok=True)
    legacy_csv_paths = sorted(args.legacy_summary_dir.glob("*.csv"))
    if not legacy_csv_paths:
        raise FileNotFoundError(f"no legacy csv files found under {args.legacy_summary_dir}")

    for legacy_path in legacy_csv_paths:
        correlation_path = args.correlation_summary_dir / legacy_path.name
        if not correlation_path.exists():
            raise FileNotFoundError(f"correlation counterpart missing for {legacy_path.name}: {correlation_path}")
        merge_one_csv(
            legacy_path=legacy_path,
            correlation_path=correlation_path,
            output_path=args.output_summary_dir / legacy_path.name,
            old_method=args.old_method,
            new_method=args.new_method,
        )


if __name__ == "__main__":
    main()
