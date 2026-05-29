from fed_simulation.config import namespace_to_config
from fed_simulation.framework import build_common_parser, run_experiment_suite
from fed_simulation.models import GCN_MODEL_BUILDERS, GCN_MODEL_LABELS


def main() -> None:
    parser = build_common_parser(
        description="GCN + BiLSTM + Attention 联邦交通流预测（优化重构版）",
        default_output_dir="results/gcn_fed_review_enhanced",
    )
    args = parser.parse_args()
    config = namespace_to_config(args, model_name="gcn", default_output_dir="results/gcn_fed_review_enhanced")
    run_experiment_suite(config, model_name="gcn", model_builders=GCN_MODEL_BUILDERS, model_labels=GCN_MODEL_LABELS)


if __name__ == "__main__":
    main()
