import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import load_data, PowerDataset, FEATURE_COLS, TARGET_COL
from tcn import TCNModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_target(data, scaler):
    target_index = FEATURE_COLS.index(TARGET_COL)
    mean = scaler.mean_[target_index]
    std = scaler.scale_[target_index]
    return data * std + mean


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, scaler):
    model.eval()

    preds = []
    trues = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_inv = inverse_target(preds, scaler)
    trues_inv = inverse_target(trues, scaler)

    mse = mean_squared_error(trues_inv.flatten(), preds_inv.flatten())
    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())

    return total_loss / len(loader), mse, mae, preds_inv, trues_inv


def plot_prediction(preds, trues, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(trues[0], label="Ground Truth")
    plt.plot(preds[0], label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Global Active Power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_experiment(args, seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, scaler = load_data(
        args.train_path,
        args.test_path,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    train_dataset = PowerDataset(X_train, y_train)
    test_dataset = PowerDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TCNModel(
        input_dim=X_train.shape[-1],
        output_len=args.output_len,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_mse = float("inf")
    best_mae = float("inf")
    best_preds = None
    best_trues = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, mse, mae, preds, trues = evaluate(
            model,
            test_loader,
            criterion,
            device,
            scaler,
        )

        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_preds = preds
            best_trues = trues

            os.makedirs(args.save_dir, exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    f"tcn_output{args.output_len}_seed{seed}.pth",
                ),
            )

        print(
            f"Seed {seed} | "
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Test Loss {test_loss:.6f} | "
            f"MSE {mse:.4f} | "
            f"MAE {mae:.4f}"
        )

    plot_prediction(
        best_preds,
        best_trues,
        save_path=os.path.join(
            args.figure_dir,
            f"tcn_output{args.output_len}_seed{seed}.png",
        ),
        title=f"TCN Prediction vs Ground Truth, Output={args.output_len}, Seed={seed}",
    )

    return best_mse, best_mae


def parse_channels(channel_str):
    return [int(x) for x in channel_str.split(",")]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train_with_calendar.csv",
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="data/processed/test_with_calendar.csv",
    )

    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--output_len", type=int, default=90)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--channels", type=str, default="64,64,128")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--figure_dir", type=str, default="figures")

    args = parser.parse_args()
    args.channels = parse_channels(args.channels)

    seeds = [1, 2, 3, 4, 5]
    results = []

    for seed in seeds:
        mse, mae = run_experiment(args, seed)
        results.append([mse, mae])

    results = np.array(results)

    print("\n========== Final Results ==========")
    print("Model: TCN")
    print(f"Input Length: {args.input_len}")
    print(f"Output Length: {args.output_len}")
    print(f"MSE Mean: {results[:, 0].mean():.4f}")
    print(f"MSE Std : {results[:, 0].std():.4f}")
    print(f"MAE Mean: {results[:, 1].mean():.4f}")
    print(f"MAE Std : {results[:, 1].std():.4f}")


if __name__ == "__main__":
    main()