# neural_search.py
# 随机搜索 + Top-K 配置 + 多随机种子集成（论文 Section IV-B / Appendix B 的做法，加上集成防过拟合）

import random

import numpy as np
import pandas as pd
import torch

from neural_config import (
    SEED,
    SEARCH_SPACE,
    SEARCH_MAX_EPOCHS,
    SEARCH_PATIENCE,
    MAX_EPOCHS,
    PATIENCE,
    TOP_K,
    N_SEEDS,
    TARGET_ANNUAL_VOL,
    TRADING_DAYS,
)
from train_utils import train_model
from neural_backtest import predict_arrays


TARGET_DAILY_VOL = TARGET_ANNUAL_VOL / np.sqrt(TRADING_DAYS)


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def _fit_one(build_model, cfg, seed, train_ds, val_ds, device, max_epochs, patience):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg)

    return train_model(
        model,
        train_ds,
        val_ds,
        device=device,
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        max_epochs=max_epochs,
        patience=patience,
        max_grad_norm=cfg["max_grad_norm"],
        weight_decay=cfg["weight_decay"],
        verbose=False,
    )


def _annual_sharpe(position, vol, fwd):
    lev = TARGET_DAILY_VOL / np.clip(vol, 1e-8, None)
    r = position * lev * fwd
    return float(r.mean() / (r.std() + 1e-8) * np.sqrt(TRADING_DAYS))


def _row(tag, fold, stage, rank, seed, cfg, info):
    return {
        "model": tag,
        "fold": fold,
        "stage": stage,
        "rank": rank,
        "seed": seed,
        "val_sharpe": info["best_val_sharpe"],
        "train_sharpe": info["train_sharpe_at_best"],
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
        **cfg,
    }


def fit_ensemble(build_model, train_ds, val_ds, device, fold, tag, search_iter):
    """
    1) 随机搜索 search_iter 组超参数（缩短训练），按验证集 Sharpe 排序
    2) 取 Top-K 配置，每组用 N_SEEDS 个种子完整训练
    3) 返回所有最终模型（预测时对仓位取平均）和搜索日志

    只使用验证集，测试集全程不参与。
    """

    rng = random.Random(SEED + 1000 * fold)

    log = []
    trials = []

    # ---------------- 1. random search

    for it in range(search_iter):

        cfg = sample_config(rng)
        seed = SEED + 100 * fold + it

        _, info = _fit_one(
            build_model, cfg, seed, train_ds, val_ds, device,
            SEARCH_MAX_EPOCHS, SEARCH_PATIENCE,
        )

        trials.append((cfg, info))
        log.append(_row(tag, fold, "search", 0, seed, cfg, info))

        print(
            f"[{tag} fold {fold}] search {it + 1}/{search_iter} "
            f"val_sharpe={info['best_val_sharpe']:+.3f} "
            f"train_sharpe={info['train_sharpe_at_best']:+.3f} "
            f"epoch={info['best_epoch']} | {cfg}"
        )

    def score(t):
        v = t[1]["best_val_sharpe"]
        return -np.inf if not np.isfinite(v) else v

    trials.sort(key=score, reverse=True)
    top = trials[:TOP_K]

    # ---------------- 2. top-K configs x N_SEEDS seeds

    members = []

    for rank, (cfg, _) in enumerate(top, 1):

        for s in range(N_SEEDS):

            seed = SEED + 100000 * fold + 1000 * rank + s

            model, info = _fit_one(
                build_model, cfg, seed, train_ds, val_ds, device,
                MAX_EPOCHS, PATIENCE,
            )

            members.append(model)
            log.append(_row(tag, fold, "final", rank, seed, cfg, info))

            print(
                f"[{tag} fold {fold}] final rank{rank} seed{s} "
                f"val_sharpe={info['best_val_sharpe']:+.3f} "
                f"epoch={info['best_epoch']}"
            )

    # ---------------- 3. ensemble diagnostics

    pos, vol, fwd = predict_arrays(members, val_ds, device)

    print(
        f"[{tag} fold {fold}] ensemble of {len(members)} models: "
        f"val Sharpe = {_annual_sharpe(pos, vol, fwd):+.3f} "
        f"(验证集同时用于选模型，这个数偏乐观，仅作参考)"
    )

    return members, pd.DataFrame(log)
