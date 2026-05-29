from fed_simulation.config import namespace_to_config
from fed_simulation.framework import build_common_parser, run_experiment_suite
from fed_simulation.models import CNN_MODEL_BUILDERS, CNN_MODEL_LABELS


def main() -> None:
    parser = build_common_parser(
        description="CNN + BiLSTM + Attention 联邦交通流预测（优化重构版）",
        default_output_dir="results/cnn_fed_review_enhanced",
    )
    args = parser.parse_args()
    config = namespace_to_config(args, model_name="cnn", default_output_dir="results/cnn_fed_review_enhanced")
    run_experiment_suite(config, model_name="cnn", model_builders=CNN_MODEL_BUILDERS, model_labels=CNN_MODEL_LABELS)


if __name__ == "__main__":
    main()
