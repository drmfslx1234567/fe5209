# neural_backtest.py

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from neural_config import (
    TARGET_ANNUAL_VOL,
    TRADING_DAYS,
)


TARGET_DAILY_VOL = TARGET_ANNUAL_VOL / np.sqrt(TRADING_DAYS)


def predict_arrays(models, dataset, device, batch_size=4096):
    """
    models 可以是单个模型，也可以是模型列表（集成：对各模型输出的仓位取平均）。
    返回 (position, vol, fwd_ret) 三个 numpy 数组。
    """

    from torch.utils.data import DataLoader

    if not isinstance(models, (list, tuple)):
        models = [models]

    for m in models:
        m.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pos, vol, fwd = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            p = torch.stack([m(x) for m in models]).mean(dim=0)
            pos.append(p.cpu())
            vol.append(batch["vol"])
            fwd.append(batch["fwd_ret"])

    return (
        torch.cat(pos).numpy(),
        torch.cat(vol).numpy(),
        torch.cat(fwd).numpy(),
    )


def predict_dataset(models, dataset, device, batch_size=4096):

    position, vol, fwd = predict_arrays(models, dataset, device, batch_size)

    weight = position * TARGET_DAILY_VOL / vol

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dataset.dates),
            "Ticker": dataset.tickers,
            "signal": position,
            "vol_daily": vol,
            "fwd_ret_1d": fwd,
            "weight": weight,
            "contrib": weight * fwd,
        }
    )


def build_daily_portfolio(predictions):

    daily = (
        predictions
        .groupby("Date")
        .agg(
            port_ret=("contrib", "mean"),
            n_assets=("contrib", "count"),
        )
        .sort_index()
    )

    return daily


def rescale_to_target_vol(
    returns,
    span=60,
    target=TARGET_ANNUAL_VOL,
):

    realised_vol = (
        returns
        .ewm(span=span, adjust=False)
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    scale = (target / realised_vol).shift(1)

    return returns * scale


def annualised_return(r):
    return r.mean() * TRADING_DAYS


def annualised_vol(r):
    return r.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(r):

    wealth = (1 + r).cumprod()

    running_max = wealth.cummax()

    dd = wealth / running_max - 1

    return -dd.min()


def downside_deviation(r):

    downside = r[r < 0]

    if len(downside) == 0:
        return np.nan

    return downside.std() * np.sqrt(TRADING_DAYS)


def summary_table(r):

    r = r.dropna()

    ret = annualised_return(r)
    vol = annualised_vol(r)
    downside = downside_deviation(r)
    mdd = max_drawdown(r)

    return {
        "E[Return]": ret,
        "Vol.": vol,
        "Downside Deviation": downside,
        "MDD": mdd,
        "Sharpe": ret / vol if vol > 0 else np.nan,
        "Sortino": ret / downside if downside > 0 else np.nan,
        "Calmar": ret / mdd if mdd > 0 else np.nan,
        "% +ve Returns": (r > 0).mean(),
        "Ave.P/Ave.L": (
            r[r > 0].mean() / abs(r[r < 0].mean())
            if (r < 0).any()
            else np.nan
        ),
    }