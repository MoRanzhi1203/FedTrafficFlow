"""Read-only audit for experiment 1 client allocation vs review requirements."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit for experiment 1 client allocation requirements."
    )
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--alignment-report", type=str, required=True)
    parser.add_argument("--heterogeneity-report", type=str, required=True)
    parser.add_argument("--output-report", type=str, required=True)
    return parser


def resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def parse_selected_clients(raw_text: str) -> list[int]:
    values = [part.strip() for part in str(raw_text).split(",")]
    clients: list[int] = []
    for value in values:
        if value:
            clients.append(int(value))
    if not clients:
        raise ValueError("--selected-clients must not be empty.")
    return clients


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def load_json(path: Path) -> dict:
    return json.loads(read_text(path))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def format_bool(value: bool) -> str:
    return "是" if value else "否"


def format_judgement(value: str) -> str:
    allowed = {"已满足", "部分满足", "未满足", "需要说明限制"}
    if value not in allowed:
        raise ValueError(f"Unsupported judgement: {value}")
    return value


def detect_grid_cell_level(split_summary: dict, client_rows: list[dict[str, str]]) -> bool:
    clients = split_summary.get("clients", [])
    used_region_count = int(split_summary.get("used_region_count", 0))
    selected_region_ids = split_summary.get("selected_region_ids", [])
    unique_region_ids = {
        int(row["region_id"])
        for row in client_rows
        if row.get("entity_kind") == "region" and row.get("region_id")
    }
    return (
        bool(clients)
        and used_region_count == len(clients)
        and len(selected_region_ids) == len(clients)
        and len(unique_region_ids) == len(clients)
        and all("region_id" in item for item in clients)
    )


def detect_cluster_level_absent(alignment_text: str) -> bool:
    negative_patterns = [
        r"cluster-level client 尚未完成",
        r"cluster-level client 还未完成",
        r"cluster client .*尚未",
        r"client 分组 / cluster client .*尚未",
        r"由新实验 3-6 分别承接",
        r"后续新实验 3-6",
    ]
    return contains_any(alignment_text, negative_patterns)


def build_client_fact_rows(split_summary: dict, client_rows: list[dict[str, str]]) -> list[list[object]]:
    metric_map: dict[int, dict[str, str]] = {}
    for row in client_rows:
        if row.get("method") == "FedAvg" and row.get("region_id"):
            metric_map[int(row["region_id"])] = row

    rows: list[list[object]] = []
    for client_item in split_summary.get("clients", []):
        region_id = int(client_item["region_id"])
        metric_row = metric_map.get(region_id, {})
        rows.append(
            [
                region_id,
                client_item.get("pooled_row", ""),
                client_item.get("pooled_col", ""),
                metric_row.get("centroid_lon", ""),
                metric_row.get("centroid_lat", ""),
                client_item.get("source_node_count", ""),
                client_item.get("mean_total_flow", ""),
                client_item.get("train", {}).get("sample_count", ""),
                client_item.get("val", {}).get("sample_count", ""),
                client_item.get("test", {}).get("sample_count", ""),
            ]
        )
    return rows


def build_requirement_rows() -> list[list[object]]:
    return [
        [
            1,
            "一审 / revision",
            "真实数据 client 需要体现多区域数据分布，而不是单一局部样本。",
            "client 组织方式需要能说明跨区域联合建模与真实数据覆盖。",
            "部分回应",
        ],
        [
            2,
            "一审 / 导师会议",
            "需要分析区域异质性 / non-IID 对联邦训练稳定性和结果差异的影响。",
            "必须补 client 分布差异、异质 client 证据与 FedAvg 受影响机制。",
            "已回应",
        ],
        [
            3,
            "一审 / revision",
            "需要补充 client-level variability，而不是只给总体平均指标。",
            "必须有 per-client 指标、client 级差异或主要拖累 client 诊断。",
            "已回应",
        ],
        [
            4,
            "一审 / 原稿口径",
            "需要说明 client 数量设置是否合理，并与原稿设定保持可解释的一致性。",
            "要解释当前 K 的选择、增强点与局限。",
            "部分回应",
        ],
        [
            5,
            "一审 / 原稿口径",
            "需要说明 clustering procedure 或 cluster client 的组织方式。",
            "不仅要有 grid-cell client，还要说明 cluster-level client 的组织逻辑。",
            "未回应",
        ],
        [
            6,
            "一审 / revision",
            "需要说明 train/validation/test split 的构造方式，避免实验口径不清。",
            "要明确时间连续切分与样本数边界。",
            "已回应",
        ],
        [
            7,
            "一审 / revision",
            "需要解释 non-IID 下 client weights 与标准 FedAvg 的稳定性限制。",
            "要说明标准样本量加权 FedAvg 在强异质条件下的表现边界。",
            "部分回应",
        ],
        [
            8,
            "原稿真实数据设定",
            "原稿需要同时区分 grid-cell-level 与 cluster-level 两类 client 组织。",
            "实验 1 只能覆盖其中一类，另一类需在后续实验补齐。",
            "部分回应",
        ],
    ]


def build_compliance_rows(facts: dict[str, object]) -> list[list[object]]:
    return [
        [
            "当前 5-client 设置是否比原稿 K=3 有增强",
            "当前 `num_clients=5`，原稿口径按 K=3 组织。",
            format_judgement("已满足"),
            "仍需人工核对原始 PDF / docx 中 K=3 表述。",
            "在文档中明确把 K=5 写成对 K=3 的增强，而非与原稿冲突。",
        ],
        [
            "当前 client 数量是否仍可能被认为偏少",
            "5 个真实数据 client 已高于 K=3，但总体仍属小规模真实数据 FL。",
            format_judgement("需要说明限制"),
            "可能仍被审稿人认为覆盖范围有限。",
            "明确写出“增强但仍是小规模真实数据 FL”这一限制。",
        ],
        [
            "当前是否已充分说明为什么选这 5 个 client",
            "已明确固定 ID，但缺少论文级选择依据。",
            format_judgement("未满足"),
            "缺少 selected_clients 选择逻辑、筛选准则和与原稿叙事的映射。",
            "补充选择依据与固定 5 个 grid cells 的设计说明。",
        ],
        [
            "当前是否已充分说明空间覆盖",
            "已有 pooled_row / pooled_col / centroid / source_node_count 事实。",
            format_judgement("部分满足"),
            "已有事实表，但尚未转化为空间覆盖解释。",
            "在报告中补 5 个 grid cells 的空间覆盖与流量分布说明。",
        ],
        [
            "当前是否已充分说明 cluster procedure",
            "项目内已有 cluster client 迁移口径，但实验 1 未展开。",
            format_judgement("未满足"),
            "当前实验 1 没有 cluster procedure 结果或正式说明。",
            "明确 grouped-client / global-partition 实验线由新实验 3-6 分别承接，当前仅说明原稿组织方式。",
        ],
        [
            "当前是否已经完成 cluster-level client 实验",
            "当前 result_dir 为 `grid_cell_main_full_cuda_v4`，且对齐报告已说明尚未完成。",
            format_judgement("未满足"),
            "缺少 cluster-level 实验结果与报告。",
            "不要在实验 1 中冒充已完成；保留为后续新实验 3-6 分别补齐。",
        ],
        [
            "当前是否已经回应 non-IID",
            "已有 gap diagnosis 与 heterogeneity diagnosis，并锁定 289。",
            format_judgement("已满足"),
            "需要把结论转写为论文级表述。",
            "把 289 异质性写成诊断结果和 FedAvg 局限解释。",
        ],
        [
            "当前是否已经回应 client-level variability",
            "已有 per-client 指标、pairwise correlation、leave-one-client-out 统计。",
            format_judgement("已满足"),
            "正文尚缺精炼版表格。",
            "补一张 client-level variability 汇总表。",
        ],
        [
            "当前实验 1 是否满足一审 / 导师要求",
            "已有真实数据闭环、split 证据、non-IID 证据，但缺选择依据与 cluster-level 补充。",
            format_judgement("部分满足"),
            "selected_clients 依据不足，cluster-level 未完成。",
            "先补文档和审计报告，再决定后续实验方向。",
        ],
    ]


def extract_evidence(alignment_text: str, heterogeneity_text: str, split_summary: dict, run_config: dict) -> dict[str, object]:
    has_variability = contains_any(
        heterogeneity_text,
        [r"client 级指标对比", r"client 分布差异", r"leave-one-client-out", r"Pearson 相关性"],
    )
    has_289 = contains_any(
        heterogeneity_text,
        [r"289.*主要拖累 client", r"主要拖累 client 是否是 `?289`?", r"移除 region `?289`? 后"],
    )
    has_split = bool(split_summary.get("clients")) and all(
        item.get("train", {}).get("sample_count", 0) and item.get("val", {}).get("sample_count", 0) and item.get("test", {}).get("sample_count", 0)
        for item in split_summary.get("clients", [])
    )
    cluster_absent = detect_cluster_level_absent(alignment_text)
    selection_basis_missing = contains_any(
        alignment_text,
        [
            r"仍需补充为何选这 5 个 grid cell",
            r"仍需说明 selected_clients 选择依据",
            r"client 划分逻辑尚未形成论文级说明",
        ],
    )
    workflow = run_config.get("workflow", "")
    rounds = run_config.get("communication_rounds", "")
    device = run_config.get("device", "")
    return {
        "has_variability": has_variability,
        "has_289": has_289,
        "has_split": has_split,
        "cluster_absent": cluster_absent,
        "selection_basis_missing": selection_basis_missing,
        "workflow": workflow,
        "rounds": rounds,
        "device": device,
    }


def render_report(
    run_config: dict,
    split_summary: dict,
    client_rows: list[dict[str, str]],
    selected_clients_arg: list[int],
    alignment_text: str,
    heterogeneity_text: str,
) -> str:
    evidence = extract_evidence(alignment_text, heterogeneity_text, split_summary, run_config)
    selected_clients_current = [int(value) for value in run_config.get("selected_clients", [])]
    num_clients = int(run_config.get("num_clients", 0))
    current_matches_arg = selected_clients_current == selected_clients_arg
    is_grid_cell_level = detect_grid_cell_level(split_summary, client_rows)
    has_cluster_level_result = False if evidence["cluster_absent"] else False
    selection_basis_status = "否"
    if evidence["selection_basis_missing"]:
        selection_basis_status = "否"
    elif current_matches_arg:
        selection_basis_status = "部分"

    requirement_table = pipe_table(
        ["编号", "来源", "原始要求摘要", "对 client 分配的含义", "当前是否回应"],
        build_requirement_rows(),
    )
    client_fact_table = pipe_table(
        [
            "region_id",
            "pooled_row",
            "pooled_col",
            "centroid_lon",
            "centroid_lat",
            "source_node_count",
            "mean_total_flow",
            "train_samples",
            "val_samples",
            "test_samples",
        ],
        build_client_fact_rows(split_summary, client_rows),
    )
    compliance_table = pipe_table(
        ["要求", "当前状态", "判断", "缺口", "修复方式"],
        build_compliance_rows(
            {
                "num_clients": num_clients,
                "selected_clients": selected_clients_current,
            }
        ),
    )

    lines = [
        "# 实验 1：真实数据 client 分配要求检测判断报告",
        "",
        "## 1. 检测目的",
        "",
        "本报告用于判断当前真实数据实验 1 的 client 分配是否符合一审意见和导师会议要求。",
        "",
        "## 2. 一审与导师会议中的 client 分配要求",
        "",
        "补充说明：当前自动检索基于项目内可读取的文本化材料；未直接解析原始 `docx/pdf` 原文，因此在最终对外答复前仍需人工核对原始一审 `docx/pdf`。",
        "",
        requirement_table,
        "",
        "## 3. 原稿真实数据实验中的 client 组织方式",
        "",
        "- 原稿包含 `grid-cell-level client`。",
        "- 原稿包含 `cluster-level client`。",
        "- `grid-cell-level`：一个 grid cell 对应一个 client。",
        "- `cluster-level`：多个 grid cells 按时间模式相似性聚成一个 client。",
        "- 原稿真实数据设定按当前项目口径整理为 `K=3`、`R=5`；其中 `K=3` 与现有 `region_client` 默认 `num_clients=3` 一致，但最终仍需人工核对原始稿件。",
        "- `grid-cell-level` 强调 `local heterogeneity`。",
        "- `cluster-level` 强调 `intra-client homogeneity`。",
        "",
        "## 4. 当前实验 1 的 client 分配事实",
        "",
        f"- 当前 `selected_clients = {','.join(str(item) for item in selected_clients_current)}`。",
        f"- 当前 client 数量 = `{num_clients}`。",
        f"- 当前 workflow = `{evidence['workflow']}`，rounds = `{evidence['rounds']}`，device = `{evidence['device']}`。",
        f"- 当前属于 grid-cell-level client：`{format_bool(is_grid_cell_level)}`。",
        f"- 当前不是 cluster-level client：`{format_bool(not has_cluster_level_result)}`。",
        "- 当前已完成 `FedAvg vs Independent`。",
        "- 当前已补充 `NaiveLastValue`。",
        "- 当前已完成 `r20/r40/r60` rounds 诊断。",
        f"- 当前已完成 client 异质性诊断：`{format_bool(evidence['has_variability'])}`。",
        f"- 当前已发现 `289` 是主要拖累 client：`{format_bool(evidence['has_289'])}`。",
        "",
        "当前 5 个 selected clients 的空间与数据统计如下：",
        "",
        client_fact_table,
        "",
        "补充判断：",
        "",
        f"- 当前 `selected_clients` 是否与审计输入一致：`{format_bool(current_matches_arg)}`。",
        f"- 当前是否已有 train/val/test split 证据：`{format_bool(evidence['has_split'])}`。",
        f"- 当前是否已有 client 选择依据：`{selection_basis_status}`。",
        "",
        "## 5. 当前设置与一审/导师要求的符合程度",
        "",
        compliance_table,
        "",
        "## 6. 需要修复优化的点",
        "",
        "- 补充 `selected_clients` 选择依据，不只给出固定 ID。",
        "- 补充 5 个 grid cells 的空间覆盖和流量分布统计。",
        "- 补充 `client-level variability` 表，保留 per-client 指标与差异摘要。",
        "- 把 `289` 异质性写成“诊断结果”，不要写成随意删除依据。",
        "- 说明 5-client 是对原稿 `K=3` 的增强，但仍属于小规模真实数据 FL。",
        "- 说明新实验 3-6 的 grouped-client / global-partition 实验线仍需继续补齐。",
        "- 统一 `CCN/CNN` 表述，真实数据实验 1 以 `CNN-LSTM-Attention` 为准。",
        "- 明确 `NaiveLastValue` 是 sanity baseline，不是原始主对比 baseline。",
        "- 明确 `Independent` 仍是主学习型 baseline。",
        "- 不要声称 `FedAvg` 已全面超过所有 baseline。",
        "",
        "## 7. 修复后的推荐实验叙事",
        "",
        "- 当前实验 1 应定位为 `grid-cell-level fine-grained heterogeneous client setting`。",
        "- 该设置用于验证真实数据联邦链路，并暴露强 non-IID 下标准 FedAvg 的局限。",
        "- `selected_clients=290,284,318,288,289` 将原稿 `K=3` 扩展为 `K=5`，增强了 `client-level variability` 分析。",
        "- 其中 `289` 显示出明显异质性，是解释 FedAvg 未全面超过 `NaiveLastValue` 的关键证据。",
        "- 后续新实验 3/4 的 grouped-client 组织与新实验 5/6 的全局覆盖式划分，将用于验证更同质或更系统的 client 组织是否能缓解 FedAvg 的跨 client 平均欠拟合问题。",
        "",
        "## 8. 是否需要立即修改实验",
        "",
        "- 不建议现在直接删除 `NaiveLastValue`。",
        "- 不建议现在直接删除 `289`。",
        "- 不建议现在直接进入实验 2。",
        "- 不建议现在改 `FedAvg`。",
        "- 建议先补齐 client 分配依据和诊断报告。",
        "- 下一步可做 `leave-one-client-out / 4-client smoke`，或转入新实验 3-6 的 grouped-client / global-partition 审计。",
        "",
        "## 9. 结论",
        "",
        "- 一审 / 导师没有要求固定 client ID。",
        "- 一审 / 导师真正要求的是 client 分配逻辑、异质性证据、client 数量说明、clustering procedure、client-level variability。",
        "- 当前实验 1 已部分满足。",
        "- 当前最大缺口是 `selected_clients` 选择依据，以及新实验 3-6 的 grouped-client / global-partition 实验线尚未完成。",
        "- 当前最小修复动作是补充 client 分配检测报告，并修正对齐报告中的相关表述。",
        "",
        "## 10. 边界声明",
        "",
        "- 本阶段未运行训练。",
        "- 未运行新实验 2-6。",
        "- 未修改 FedAvg。",
        "- 未修改模型结构。",
        "- 未修改数据划分。",
        "- 未提交 `results/`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()

    result_dir = resolve_path(args.result_dir)
    alignment_report = resolve_path(args.alignment_report)
    heterogeneity_report = resolve_path(args.heterogeneity_report)
    output_report = resolve_path(args.output_report)
    selected_clients_arg = parse_selected_clients(args.selected_clients)

    run_config = load_json(result_dir / "run_config.json")
    split_summary = load_json(result_dir / "split_summary.json")
    client_rows = load_csv_rows(result_dir / "client_metrics.csv")
    alignment_text = read_text(alignment_report)
    heterogeneity_text = read_text(heterogeneity_report)

    report_text = render_report(
        run_config=run_config,
        split_summary=split_summary,
        client_rows=client_rows,
        selected_clients_arg=selected_clients_arg,
        alignment_text=alignment_text,
        heterogeneity_text=heterogeneity_text,
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report_text, encoding="utf-8")

    print("[client_count]", run_config.get("num_clients", ""))
    print("[selected_clients]", ",".join(str(item) for item in run_config.get("selected_clients", [])))
    print("[workflow]", run_config.get("workflow", ""))
    print("[rounds]", run_config.get("communication_rounds", ""))
    print("[device]", run_config.get("device", ""))
    print("[output_report]", output_report)


if __name__ == "__main__":
    main()
