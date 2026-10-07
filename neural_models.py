import torch
import torch.nn as nn


class MLPDirect(nn.Module):
    """
    论文式 (19)-(20): 单隐层 MLP，tanh 隐层 + tanh 输出(直接输出仓位 in [-1,1])。
    dropout 作用于输入和隐层（论文: "applied to the inputs and hidden state for the MLP"）。
    n_hidden_layers=2 可恢复你原来的两隐层结构（参数更多，更容易过拟合）。
    """

    def __init__(
        self,
        n_features=8,
        lookback=5,
        hidden_size=20,
        dropout=0.0,
        n_hidden_layers=1,
    ):
        super().__init__()

        in_dim = n_features * lookback

        layers = [nn.Dropout(dropout)]
        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size

        layers += [nn.Linear(in_dim, 1), nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


class LockedDropout(nn.Module):
    """
    同一条序列的所有时间步共用同一个 dropout mask（variational dropout，
    Gal & Ghahramani 2016 的输入 dropout 部分）。x: (batch, time, features)
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


class LSTMDirect(nn.Module):
    """
    LSTM + 输入 dropout(锁定 mask) + 输出 dropout。
    注意：论文还对循环状态做 dropout；nn.LSTM 不支持，这里没有实现（输入+输出 dropout 已能起到主要正则作用）。
    """

    def __init__(
        self,
        n_features=8,
        hidden_size=20,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.in_drop = LockedDropout(dropout)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out_drop = nn.Dropout(dropout)

        self.output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.in_drop(x)
        output, _ = self.lstm(x)
        h = self.out_drop(output[:, -1, :])
        return self.output(h).squeeze(-1)
