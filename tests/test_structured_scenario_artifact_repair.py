from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis_scripts.analyze_structured_missingness_distribution import (
    normalize_mask_file_name,
    resolve_missing_root,
)
from analysis_scripts.repair_structured_scenario_artifacts import (
    repair_structured_scenario_artifacts,
    validate_manifest_distribution_consistency,
)


class StructuredScenarioArtifactRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_input_chunk(self, file_name: str, rows: int) -> None:
        df = pd.DataFrame(
            {
                "节点ID": list(range(rows)),
                "时间段": list(range(rows)),
                "路口车流量": [10 + idx for idx in range(rows)],
            }
        )
        df.to_parquet(self.input_dir / file_name, index=False)

    def _build_outage_scenario_dir(self, *, include_foreign_status: bool = False) -> Path:
        scenario_dir = self.root / "results" / "rdm_exp" / "scenarios" / "nso_mix" / "miss_set"
        manifests_dir = scenario_dir / "manifests"
        masks_dir = scenario_dir / "masks"
        missing_dir = scenario_dir / "miss_data"
        audits_dir = scenario_dir / "audits"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        missing_dir.mkdir(parents=True, exist_ok=True)
        audits_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "length_mode": "mixed_short_mid_long",
            "length_group_probs": [0.4, 0.4, 0.2],
            "short_length_range": [1, 4],
            "mid_length_range": [5, 12],
            "long_length_range": [13, 24],
            "tolerance": 0.001,
        }
        (scenario_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

        prepare_df = pd.DataFrame(
            [
                {"chunk_index": 0, "day_index": 0, "file_name": "node_flow_chunk_000.parquet", "row_count": 4, "target_non_null_count": 4},
                {"chunk_index": 1, "day_index": 1, "file_name": "node_flow_chunk_001.parquet", "row_count": 6, "target_non_null_count": 6},
            ]
        )
        prepare_df.to_csv(manifests_dir / "structured_prepare_chunk_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"node_id": [1, 2, 3]}).to_csv(manifests_dir / "node_index.csv", index=False, encoding="utf-8-sig")
        (manifests_dir / "global_time_index_summary.json").write_text(
            json.dumps({"chunk_count": 2, "period": 96}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        chunk_status_rows = []
        for rate, count_by_file in [(0.05, {"node_flow_chunk_000.parquet": 1, "node_flow_chunk_001.parquet": 1}), (0.1, {"node_flow_chunk_000.parquet": 2, "node_flow_chunk_001.parquet": 2})]:
            scenario_tag = f"mechanism_node_subset_temporal_outage__rate_{str(rate).replace('0.', '0p')}__mixed_short_mid_long__seed_42"
            short_name = f"nso_r{int(round(rate * 100)):02d}_mix_s42"
            (masks_dir / short_name).mkdir(parents=True, exist_ok=True)
            (missing_dir / short_name).mkdir(parents=True, exist_ok=True)
            for chunk_index, (file_name, missing_count) in enumerate(count_by_file.items()):
                row_count = int(prepare_df.loc[prepare_df["file_name"] == file_name, "row_count"].iloc[0])
                mask_df = pd.DataFrame(
                    {
                        "row_index": list(range(missing_count)),
                        "global_time_index": list(range(missing_count)),
                        "节点ID": [1] * missing_count,
                        "actual_length": [1] * missing_count,
                        "length_group": ["short"] * missing_count,
                    }
                )
                mask_df.to_parquet((masks_dir / short_name / file_name.replace(".parquet", "_mask.parquet")), index=False)
                pd.DataFrame({"节点ID": list(range(row_count)), "时间段": list(range(row_count)), "路口车流量": [None] * row_count}).to_parquet(
                    missing_dir / short_name / file_name,
                    index=False,
                )
                chunk_status_rows.append(
                    {
                        "mechanism": "node_subset_temporal_outage",
                        "missing_rate_target": rate,
                        "scenario_tag": scenario_tag,
                        "parameter_setting": "mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2",
                        "length_mode": "mixed_short_mid_long",
                        "chunk_index": chunk_index,
                        "day_index": chunk_index,
                        "file_name": file_name,
                        "mask_path": str(masks_dir / short_name / file_name.replace(".parquet", "_mask.parquet")),
                        "missing_dataset_path": str(missing_dir / short_name / file_name),
                        "mask_file_count": 1,
                        "missing_dataset_file_count": 1,
                        "row_count": row_count,
                        "target_non_null_count": row_count,
                        "observed_missing_count": missing_count,
                        "observed_missing_rate": missing_count / float(row_count),
                        "drops_entire_day": False,
                        "drops_all_nodes_at_same_time": False,
                        "uses_row_index_mask": True,
                        "modifies_only_target_col": True,
                        "point_topup_count": 0,
                    }
                )
        if include_foreign_status:
            chunk_status_rows.append(
                {
                    "mechanism": "node_temporal_block",
                    "missing_rate_target": 0.05,
                    "scenario_tag": "mechanism_node_temporal_block__rate_0p05__mixed_short_mid_long__seed_42",
                    "parameter_setting": "mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2",
                    "length_mode": "mixed_short_mid_long",
                    "chunk_index": 0,
                    "day_index": 0,
                    "file_name": "node_flow_chunk_000.parquet",
                    "mask_path": "foreign_mask_path",
                    "missing_dataset_path": "foreign_missing_dataset_path",
                    "mask_file_count": 1,
                    "missing_dataset_file_count": 1,
                    "row_count": 4,
                    "target_non_null_count": 4,
                    "observed_missing_count": 1,
                    "observed_missing_rate": 0.25,
                    "drops_entire_day": False,
                    "drops_all_nodes_at_same_time": False,
                    "uses_row_index_mask": True,
                    "modifies_only_target_col": True,
                    "point_topup_count": 0,
                }
            )
        pd.DataFrame(chunk_status_rows).to_csv(
            manifests_dir / "structured_missing_chunk_status.csv",
            index=False,
            encoding="utf-8-sig",
        )

        self._write_input_chunk("node_flow_chunk_000.parquet", 4)
        self._write_input_chunk("node_flow_chunk_001.parquet", 6)
        return scenario_dir

    def test_repair_rebuilds_nso_artifacts_from_local_outputs(self) -> None:
        scenario_dir = self._build_outage_scenario_dir()
        manifest_df, distribution_df, validation_df = repair_structured_scenario_artifacts(
            scenario_dir=scenario_dir,
            input_dir=self.input_dir,
            target_col="路口车流量",
            node_col="节点ID",
            period=96,
        )

        self.assertEqual(set(manifest_df["mechanism"].tolist()), {"node_subset_temporal_outage"})
        self.assertEqual(set(distribution_df["mechanism"].tolist()), {"node_subset_temporal_outage"})
        self.assertEqual(set(manifest_df["scenario_output_name"].tolist()), {"nso_r05_mix_s42", "nso_r10_mix_s42"})
        self.assertEqual(set(distribution_df["scenario"].tolist()), {"nso_r05_mix_s42", "nso_r10_mix_s42"})
        self.assertTrue(validation_df["is_consistent"].all())

        repaired_manifest = pd.read_csv(scenario_dir / "manifests" / "structured_missing_scenario_summary.csv")
        repaired_distribution = pd.read_csv(scenario_dir / "audits" / "structured_missingness_distribution_summary.csv")
        self.assertEqual(len(repaired_manifest), 2)
        self.assertEqual(len(repaired_distribution), 2)
        self.assertTrue((repaired_manifest["mechanism"] == "node_subset_temporal_outage").all())
        self.assertTrue((repaired_distribution["mechanism"] == "node_subset_temporal_outage").all())

    def test_repair_ignores_foreign_shared_status_rows(self) -> None:
        scenario_dir = self._build_outage_scenario_dir(include_foreign_status=True)
        manifest_df, distribution_df, validation_df = repair_structured_scenario_artifacts(
            scenario_dir=scenario_dir,
            input_dir=self.input_dir,
            target_col="路口车流量",
            node_col="节点ID",
            period=96,
        )

        self.assertEqual(set(manifest_df["scenario_output_name"].tolist()), {"nso_r05_mix_s42", "nso_r10_mix_s42"})
        self.assertEqual(set(distribution_df["scenario"].tolist()), {"nso_r05_mix_s42", "nso_r10_mix_s42"})
        self.assertTrue((manifest_df["mechanism"] == "node_subset_temporal_outage").all())
        self.assertTrue(validation_df["is_consistent"].all())

    def test_validate_manifest_distribution_consistency_rejects_count_mismatch(self) -> None:
        manifest_df = pd.DataFrame(
            [
                {
                    "scenario_output_name": "nso_r05_mix_s42",
                    "scenario_tag": "mechanism_node_subset_temporal_outage__rate_0p05__mixed_short_mid_long__seed_42",
                    "mechanism": "node_subset_temporal_outage",
                    "missing_rate_target": 0.05,
                    "observed_missing_count": 10,
                    "observed_missing_rate": 0.05,
                }
            ]
        )
        distribution_df = pd.DataFrame(
            [
                {
                    "scenario": "nso_r05_mix_s42",
                    "mechanism": "node_subset_temporal_outage",
                    "missing_rate_target": 0.05,
                    "observed_missing_count": 9,
                    "observed_missing_rate": 0.05,
                }
            ]
        )
        with self.assertRaises(RuntimeError):
            validate_manifest_distribution_consistency(manifest_df, distribution_df)

    def test_resolve_missing_root_supports_short_and_legacy_names(self) -> None:
        scenario_dir = self.root / "scenario"
        (scenario_dir / "miss_data").mkdir(parents=True, exist_ok=True)
        self.assertEqual(resolve_missing_root(scenario_dir), scenario_dir / "miss_data")

        legacy_dir = self.root / "legacy"
        (legacy_dir / "missing_datasets").mkdir(parents=True, exist_ok=True)
        self.assertEqual(resolve_missing_root(legacy_dir), legacy_dir / "missing_datasets")

    def test_normalize_mask_file_name_restores_chunk_name(self) -> None:
        self.assertEqual(normalize_mask_file_name("node_flow_chunk_000_mask.parquet"), "node_flow_chunk_000.parquet")
        self.assertEqual(normalize_mask_file_name("node_flow_chunk_000.parquet"), "node_flow_chunk_000.parquet")


if __name__ == "__main__":
    unittest.main()
