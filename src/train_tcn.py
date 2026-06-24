import argparse

from tcn import TCNModel
from train_utils import run_experiments


def parse_channels(channel_str):
    return [int(x) for x in channel_str.split(",")]


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

    parser.add_argument("--channels", type=str, default="64,64,128")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--figure_dir", type=str, default="figures")

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    args = parser.parse_args()
    args.channels = parse_channels(args.channels)

    return args


def main():
    args = parse_args()

    def build_model(input_dim):
        return TCNModel(
            input_dim=input_dim,
            output_len=args.output_len,
            channels=args.channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout
        )

    run_experiments(args, "TCN", build_model)


if __name__ == "__main__":
    main()