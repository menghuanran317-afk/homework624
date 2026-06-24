import argparse

from lstm import LSTMModel
from train_utils import run_experiments


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/processed/train_with_calendar.csv")
    parser.add_argument("--test_path", type=str, default="data/processed/test_with_calendar.csv")

    # 长期预测推荐使用这个
    parser.add_argument("--full_data_path", type=str, default="")

    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--output_len", type=int, default=90)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--figure_dir", type=str, default="figures")

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    return parser.parse_args()


def main():
    args = parse_args()

    def build_model(input_dim):
        return LSTMModel(
            input_dim=input_dim,
            output_len=args.output_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    run_experiments(args, "LSTM", build_model)


if __name__ == "__main__":
    main()