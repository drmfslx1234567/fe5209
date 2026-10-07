import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from neural_config import FEATURES


def load_model_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])

    required = (
        ["Date", "Ticker", "vol_daily", "fwd_ret_1d", "model_available"]
        + FEATURES
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["model_available"] == True].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=FEATURES + ["vol_daily", "fwd_ret_1d"])

    df = df[df["vol_daily"] > 0]

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} rows | "
        f"{df['Ticker'].nunique()} tickers | "
        f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    )
    return df


def make_expanding_splits(
    df,
    first_test_year=None,
    test_years=5,
):
    """
    Expanding-window cross validation.

    Example:

        train <= 2010
        test  = 2011-2015

        train <= 2015
        test  = 2016-2020

        ...
    """

    min_year = df["Date"].dt.year.min()
    max_year = df["Date"].dt.year.max()

    if first_test_year is None:
        first_test_year = min_year + 5

    splits = []

    test_start = first_test_year

    while test_start <= max_year:

        test_end = min(test_start + test_years - 1, max_year)

        train = df[df["Date"].dt.year < test_start].copy()

        test = df[
            (df["Date"].dt.year >= test_start)
            & (df["Date"].dt.year <= test_end)
        ].copy()

        if len(train) > 0 and len(test) > 0:
            splits.append(
                {
                    "train": train,
                    "test": test,
                    "train_end": test_start - 1,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        test_start += test_years

    return splits


def chronological_train_validation_split(df, validation_fraction=0.10):
    """
    Last 10% of dates = validation.

    IMPORTANT:
    split by DATE, not random rows.
    """

    dates = np.array(sorted(df["Date"].unique()))

    split_idx = int(len(dates) * (1.0 - validation_fraction))

    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    train = df[df["Date"].isin(train_dates)].copy()
    val = df[df["Date"].isin(val_dates)].copy()

    return train, val


class SequenceDataset(Dataset):
    """
    懒加载版本：不再预先物化 N x lookback x F 的大张量（LSTM 63 步时会爆内存）。
    关键改动：df 传入「含历史的完整数据」，只保留目标日在 [start_date, end_date] 内的样本，
    这样 val/test 的前 lookback-1 天不会因为窗口不足而被丢掉。
    """

    def __init__(self, df, lookback, feature_cols=FEATURES,
                 start_date=None, end_date=None):

        self.lookback = lookback

        feats, vols, rets = [], [], []
        rows, dates, tickers = [], [], []
        offset = 0

        for ticker, g in df.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)

            feats.append(g[feature_cols].to_numpy(np.float32))
            vols.append(g["vol_daily"].to_numpy(np.float32))
            rets.append(g["fwd_ret_1d"].to_numpy(np.float32))

            d = g["Date"].to_numpy()
            mask = np.ones(len(g), dtype=bool)
            mask[: lookback - 1] = False
            if start_date is not None:
                mask &= d >= np.datetime64(start_date)
            if end_date is not None:
                mask &= d <= np.datetime64(end_date)

            idx = np.where(mask)[0]
            rows.append(idx + offset)
            dates.append(d[idx])
            tickers.append(np.full(len(idx), ticker, dtype=object))
            offset += len(g)

        self.F = torch.from_numpy(np.concatenate(feats))
        self.vol = torch.from_numpy(np.concatenate(vols))
        self.fwd = torch.from_numpy(np.concatenate(rets))
        self.rows = np.concatenate(rows)
        self.dates = np.concatenate(dates)
        self.tickers = np.concatenate(tickers)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "x": self.F[r - self.lookback + 1: r + 1],
            "vol": self.vol[r],
            "fwd_ret": self.fwd[r],
            "idx": idx,
        }


def make_fold_datasets(df, split, lookback, validation_fraction=0.10):
    """
    一个 fold 的 train / val / test。
    train/val 按日期切（最后 10% 日期为 val），test 为 [test_start, test_end] 年。
    三者都从「<= test_end 的完整历史」构造窗口，窗口只回看过去。
    """
    hist = df[df["Date"].dt.year <= split["test_end"]]

    train_all = split["train"]
    dates = np.array(sorted(train_all["Date"].unique()))
    cut = int(len(dates) * (1 - validation_fraction))
    train_end, val_start = dates[cut - 1], dates[cut]
    val_end = dates[-1]

    test_start = np.datetime64(f"{split['test_start']}-01-01")
    test_end = np.datetime64(f"{split['test_end']}-12-31")

    train_ds = SequenceDataset(hist, lookback, end_date=train_end)
    val_ds = SequenceDataset(hist, lookback, start_date=val_start, end_date=val_end)
    test_ds = SequenceDataset(hist, lookback, start_date=test_start, end_date=test_end)
    return train_ds, val_ds, test_ds
