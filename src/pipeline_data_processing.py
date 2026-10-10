# pipeline_data_processing.py —— 特征工程
# 读取 data/raw/ohlcv_data_by_date_adjusted.xlsx 的 Close sheet，只保留 src/universe.py 中的资产池，
# 输出 data/processed/full_feature_panel.csv 和 model_ready_dataset.csv。
# 运行方式（在项目根目录下）:  python src/pipeline_data_processing.py

from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_RAW, DATA_PROCESSED
from universe import UNIVERSE


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = DATA_RAW / "ohlcv_data_by_date_adjusted.xlsx"
OUTPUT_DIR = DATA_PROCESSED

VOL_SPAN = 60
MIN_VOL_OBS = 20

HORIZONS = {
    "1d": 1,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

MACD_PAIRS = [
    (8, 24),
    (16, 48),
    (32, 96),
]

WINSOR_HALFLIFE = 252
WINSOR_N_STD = 5


# ============================================================
# 1. Load Adjusted Close
# ============================================================

def load_close(file_path):
    """
    Load adjusted Close prices from the 'Close' sheet.

    Raw Yahoo Finance data were downloaded with auto_adjust=True.
    No forward filling is performed.
    """

    close = pd.read_excel(
        file_path,
        sheet_name="Close",
        index_col=0,
    )

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    # Keep only the project universe (src/universe.py).
    missing = [t for t in UNIVERSE if t not in close.columns]
    if missing:
        raise ValueError(f"Tickers not found in price file: {missing}")
    close = close[UNIVERSE]
    # Drop dates on which none of these tickers traded.
    close = close.dropna(how="all")

    return close


# ============================================================
# 2. Daily Simple Returns
# ============================================================

def compute_daily_returns(close):
    """
    Compute simple daily returns independently for each ETF:

        ret_t = P_t / P_(t-1) - 1

    Each ETF uses its own valid trading observations.
    Missing market days are NOT treated as 0% returns.
    """

    returns = pd.DataFrame(
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for ticker in close.columns:

        price = close[ticker].dropna()

        ret = price.pct_change(
            fill_method=None
        )

        returns.loc[ret.index, ticker] = ret

    return returns


# ============================================================
# 3. EWM Volatility
# ============================================================

def compute_ewm_volatility(
    returns,
    span=VOL_SPAN,
    min_periods=MIN_VOL_OBS,
):
    """
    Compute daily exponentially weighted volatility.

    Paper parameter:
        EWM span = 60
    """

    vol_daily = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    for ticker in returns.columns:

        r = returns[ticker].dropna()

        vol = (
            r.ewm(
                span=span,
                adjust=False,
                min_periods=min_periods,
            )
            .std()
        )

        vol_daily.loc[vol.index, ticker] = vol

    return vol_daily


# ============================================================
# 4. Multi-Horizon Momentum Returns
# ============================================================

def compute_momentum_returns(
    close,
    horizons=HORIZONS,
):
    """
    Compute cumulative simple returns:

        1D  = 1 trading observation
        1M  = 21
        3M  = 63
        6M  = 126
        1Y  = 252

        return_h(t) = P_t / P_(t-h) - 1
    """

    momentum_returns = {}

    for name, horizon in horizons.items():

        result = pd.DataFrame(
            index=close.index,
            columns=close.columns,
            dtype=float,
        )

        for ticker in close.columns:

            price = close[ticker].dropna()

            ret_h = (
                price / price.shift(horizon) - 1
            )

            result.loc[
                ret_h.index,
                ticker
            ] = ret_h

        momentum_returns[name] = result

    return momentum_returns


# ============================================================
# 5. Volatility-Normalised Momentum Returns
# ============================================================

def normalize_momentum_returns(
    momentum_returns,
    vol_daily,
    horizons=HORIZONS,
):
    """
    Paper-style volatility normalisation:

        norm_ret_h =
            return_h / (vol_daily * sqrt(h))
    """

    return {
        f"norm_ret_{name}":
            momentum_returns[name]
            / (
                vol_daily
                * np.sqrt(horizon)
            )

        for name, horizon
        in horizons.items()
    }


# ============================================================
# 6. Normalised MACD Features
# ============================================================

def compute_macd_features(
    close,
    macd_pairs=MACD_PAIRS,
):
    """
    Compute paper-style normalised MACD.

    For each (S, L):

        MACD = EWMA_S(P) - EWMA_L(P)

    Paper timescale definition implies:

        alpha = 1 / timescale

    First normalisation:

        q = MACD / rolling_std_63(P)

    Second normalisation:

        Y = q / rolling_std_252(q)
    """

    macd_features = {}

    for short, long in macd_pairs:

        result = pd.DataFrame(
            index=close.index,
            columns=close.columns,
            dtype=float,
        )

        for ticker in close.columns:

            price = close[ticker].dropna()

            # IMPORTANT:
            # Paper timescale -> alpha = 1 / S
            # Do NOT replace with ewm(span=S).
            ema_short = price.ewm(
                alpha=1 / short,
                adjust=False,
            ).mean()

            ema_long = price.ewm(
                alpha=1 / long,
                adjust=False,
            ).mean()

            raw_macd = (
                ema_short - ema_long
            )

            price_std_63 = (
                price
                .rolling(63)
                .std()
            )

            q = (
                raw_macd
                / price_std_63
            )

            q_std_252 = (
                q
                .rolling(252)
                .std()
            )

            y = (
                q
                / q_std_252
            )

            result.loc[
                y.index,
                ticker
            ] = y

        macd_features[
            f"macd_{short}_{long}"
        ] = result

    return macd_features


# ============================================================
# 7. EWM Winsorisation
# ============================================================

def winsorize_ewm(
    df,
    halflife=WINSOR_HALFLIFE,
    n_std=WINSOR_N_STD,
):
    """
    Apply causal EWM winsorisation independently
    for each ETF.

    Bounds:

        EWM mean +/- 5 * EWM std

    EWM half-life:

        252 trading observations
    """

    output = pd.DataFrame(
        index=df.index,
        columns=df.columns,
        dtype=float,
    )

    for ticker in df.columns:

        x = df[ticker].dropna()

        mean = x.ewm(
            halflife=halflife,
            adjust=False,
        ).mean()

        std = x.ewm(
            halflife=halflife,
            adjust=False,
        ).std()

        lower = mean - n_std * std
        upper = mean + n_std * std

        output.loc[
            x.index,
            ticker
        ] = x.clip(
            lower=lower,
            upper=upper,
        )

    return output


def clean_features(features):
    """
    Winsorise all model features.
    """

    return {
        name: winsorize_ewm(df)
        for name, df in features.items()
    }


# ============================================================
# 8. Forward Returns
# ============================================================

def compute_forward_returns(ret_1d):
    """
    Construct next-trading-observation return:

        fwd_ret_1d(t)
            = return from t to next ETF trading observation

    Each ETF is shifted independently on its own
    valid trading calendar.
    """

    forward = pd.DataFrame(
        index=ret_1d.index,
        columns=ret_1d.columns,
        dtype=float,
    )

    for ticker in ret_1d.columns:

        r = ret_1d[ticker].dropna()

        forward.loc[
            r.index,
            ticker
        ] = r.shift(-1)

    return forward


# ============================================================
# 9. Build Long-Format Panel
# ============================================================

def build_panel(
    close,
    momentum_returns,
    vol_daily,
    features_clean,
    fwd_ret_1d,
):
    """
    Combine prices, raw returns, volatility,
    model features and forward returns into
    a Date x Ticker long-format panel.
    """

    # ----------------------------------------
    # Model features
    # ----------------------------------------

    panel = pd.concat(
        {
            name: df.stack()
            for name, df
            in features_clean.items()
        },
        axis=1,
    )

    panel.index.names = [
        "Date",
        "Ticker",
    ]

    # ----------------------------------------
    # Basic variables
    # ----------------------------------------

    panel["Close"] = close.stack()

    panel["ret_1d"] = (
        momentum_returns["1d"]
        .stack()
    )

    panel["ret_1m"] = (
        momentum_returns["1m"]
        .stack()
    )

    panel["ret_3m"] = (
        momentum_returns["3m"]
        .stack()
    )

    panel["ret_6m"] = (
        momentum_returns["6m"]
        .stack()
    )

    panel["ret_1y"] = (
        momentum_returns["1y"]
        .stack()
    )

    panel["vol_daily"] = (
        vol_daily.stack()
    )

    panel["fwd_ret_1d"] = (
        fwd_ret_1d.stack()
    )

    # ----------------------------------------
    # Availability flags
    # ----------------------------------------

    feature_cols = list(
        features_clean.keys()
    )

    required = (
        feature_cols
        + [
            "Close",
            "ret_1d",
            "vol_daily",
        ]
    )

    panel["feature_available"] = (
        panel[required]
        .notna()
        .all(axis=1)
    )

    panel["model_available"] = (
        panel["feature_available"]
        & panel["fwd_ret_1d"].notna()
    )

    # ----------------------------------------
    # Convert MultiIndex -> columns
    # ----------------------------------------

    panel = (
        panel
        .reset_index()
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    return panel


# ============================================================
# 10. Quality Control
# ============================================================

def run_quality_checks(
    panel,
    feature_cols,
):
    """
    Final QC for model-ready observations.

    Raises an error if:
        - feature NaN exists
        - feature Inf exists
        - forward return is missing
    """

    model_data = panel.loc[
        panel["model_available"]
    ].copy()

    feature_nan = (
        model_data[feature_cols]
        .isna()
        .sum()
        .sum()
    )

    feature_inf = np.isinf(
        model_data[feature_cols]
        .to_numpy()
    ).sum()

    forward_nan = (
        model_data["fwd_ret_1d"]
        .isna()
        .sum()
    )

    print("\n===== FINAL DATA QC =====")

    print(
        "Tickers:",
        model_data["Ticker"].nunique()
    )

    print(
        "Model-ready rows:",
        len(model_data)
    )

    print(
        "Feature NaN:",
        feature_nan
    )

    print(
        "Feature Inf:",
        feature_inf
    )

    print(
        "Forward Return NaN:",
        forward_nan
    )

    # Hard validation
    if feature_nan != 0:
        raise ValueError(
            "Model features contain NaN."
        )

    if feature_inf != 0:
        raise ValueError(
            "Model features contain Inf."
        )

    if forward_nan != 0:
        raise ValueError(
            "Forward returns contain NaN."
        )

    return model_data


# ============================================================
# 11. Export
# ============================================================

def export_data(
    panel,
    model_data,
    output_dir=OUTPUT_DIR,
):
    """
    Export:

    1. full_feature_panel.csv
       Complete audit dataset.

    2. model_ready_dataset.csv
       Only model_available == True observations.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        output_dir
        / "full_feature_panel.csv"
    )

    model_path = (
        output_dir
        / "model_ready_dataset.csv"
    )

    panel.to_csv(
        full_path,
        index=False,
    )

    model_data.to_csv(
        model_path,
        index=False,
    )

    print("\n===== FILES SAVED =====")
    print(full_path)
    print(model_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print(
        "===== ETF DATA PIPELINE START ====="
    )

    # ----------------------------------------
    # Step 1: Load adjusted prices
    # ----------------------------------------

    print("\n[1/8] Loading adjusted Close...")

    close = load_close(
        INPUT_FILE
    )

    print(
        f"Dates: {len(close):,}"
    )

    print(
        f"ETFs: {len(close.columns)}"
    )

    # ----------------------------------------
    # Step 2: Daily returns
    # ----------------------------------------

    print("\n[2/8] Computing daily returns...")

    returns = compute_daily_returns(
        close
    )

    # ----------------------------------------
    # Step 3: EWM volatility
    # ----------------------------------------

    print("\n[3/8] Computing EWM volatility...")

    vol_daily = compute_ewm_volatility(
        returns
    )

    # ----------------------------------------
    # Step 4: Multi-horizon returns
    # ----------------------------------------

    print(
        "\n[4/8] Computing momentum returns..."
    )

    momentum_returns = (
        compute_momentum_returns(
            close
        )
    )

    # Sanity check:
    # 1D momentum return must equal Step 2 daily return.
    if not np.allclose(
        momentum_returns["1d"].to_numpy(),
        returns.to_numpy(),
        equal_nan=True,
    ):
        raise ValueError(
            "1D momentum return does not match daily return."
        )

    # ----------------------------------------
    # Step 5: Normalised momentum
    # ----------------------------------------

    print(
        "\n[5/8] Normalising momentum returns..."
    )

    norm_features = (
        normalize_momentum_returns(
            momentum_returns,
            vol_daily,
        )
    )

    # ----------------------------------------
    # Step 6: MACD
    # ----------------------------------------

    print(
        "\n[6/8] Computing MACD features..."
    )

    macd_features = (
        compute_macd_features(
            close
        )
    )

    # Combine the 8 paper-style model features.
    features = {
        **norm_features,
        **macd_features,
    }

    # Apply paper-style EWM winsorisation.
    features_clean = (
        clean_features(
            features
        )
    )

    # ----------------------------------------
    # Step 7: Forward return + panel
    # ----------------------------------------

    print(
        "\n[7/8] Building model panel..."
    )

    fwd_ret_1d = (
        compute_forward_returns(
            returns
        )
    )

    panel = build_panel(
        close=close,
        momentum_returns=momentum_returns,
        vol_daily=vol_daily,
        features_clean=features_clean,
        fwd_ret_1d=fwd_ret_1d,
    )

    # ----------------------------------------
    # Step 8: Final QC + export
    # ----------------------------------------

    print(
        "\n[8/8] Running final QC..."
    )

    feature_cols = list(
        features_clean.keys()
    )

    model_data = run_quality_checks(
        panel,
        feature_cols,
    )

    export_data(
        panel,
        model_data,
    )

    print(
        "\n===== ETF DATA PIPELINE COMPLETE ====="
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
