"""Read-only inventory export for full-cells region-client experiments."""

from __future__ import annotations

import argparse

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.result_writer import write_csv, write_text
from real_data_experiments.common.tensor_dataset import load_grid_tensor_bundle
from real_data_experiments.region_client_full_cells.rfc_partition import (
    build_full_cell_inventory,
    build_inventory_markdown,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the full-cell inventory.")
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="real_data_experiments/region_client_full_cells/full_cell_inventory.csv",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="real_data_experiments/region_client_full_cells/full_cell_inventory_zh.md",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = load_grid_tensor_bundle(args.tensor_path, args.regions_path)
    inventory_df = build_full_cell_inventory(bundle)
    output_csv = resolve_path(args.output_csv)
    output_report = resolve_path(args.output_report)
    write_csv(inventory_df, output_csv)
    write_text(build_inventory_markdown(inventory_df, str(output_csv)), output_report)
    print(f"[inventory_csv] {output_csv}")
    print(f"[inventory_report] {output_report}")
    print(f"[valid_cells] {int(inventory_df['is_valid_cell'].sum())}")


if __name__ == "__main__":
    main()

