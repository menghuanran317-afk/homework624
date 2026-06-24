import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []

        for i in range(len(num_channels)):
            dilation = 2 ** i

            in_channels = (
                num_inputs
                if i == 0
                else num_channels[i - 1]
            )

            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_len,
        channels=[64, 64, 128],
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        self.tcn = TemporalConvNet(
            input_dim,
            channels,
            kernel_size,
            dropout
        )

        self.fc = nn.Linear(
            channels[-1],
            output_len
        )

    def forward(self, x):

        # [B,L,C] -> [B,C,L]
        x = x.permute(0, 2, 1)

        x = self.tcn(x)

        # 最后时间步
        x = x[:, :, -1]

        x = self.fc(x)

        return x