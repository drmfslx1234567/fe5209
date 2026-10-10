# neural_loss.py

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from neural_config import (
    TARGET_ANNUAL_VOL,
    TRADING_DAYS,
)


TARGET_DAILY_VOL = TARGET_ANNUAL_VOL / (TRADING_DAYS ** 0.5)


def strategy_returns(position, vol, fwd_ret):
    """
    Per-asset strategy contribution:

        X_t
        * sigma_target_daily / sigma_t
        * r_(t,t+1)
    """

    leverage = TARGET_DAILY_VOL / torch.clamp(vol, min=1e-8)

    return position * leverage * fwd_ret


def sharpe_loss(position, vol, fwd_ret):
    """
    Negative annualised Sharpe.

    Optimizer minimizes loss, therefore:

        loss = -Sharpe
    """

    r = strategy_returns(
        position,
        vol,
        fwd_ret,
    )

    mean = r.mean()
    std = r.std(unbiased=False)

    sharpe = (
        mean / (std + 1e-8)
        * (TRADING_DAYS ** 0.5)
    )

    return -sharpe


def mse_return_loss(prediction, vol, fwd_ret):
    """
    Optional regression baseline.

    Target = volatility-normalised next return.

    This is NOT the direct Sharpe model.
    """

    target = fwd_ret / torch.clamp(vol, min=1e-8)

    return torch.mean((prediction - target) ** 2)