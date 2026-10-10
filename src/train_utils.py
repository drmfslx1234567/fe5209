# train_utils.py

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_loss import sharpe_loss


def train_model(
    model,
    train_dataset,
    val_dataset,
    device,
    batch_size=1024,
    learning_rate=1e-3,
    max_epochs=100,
    patience=25,
    max_grad_norm=1.0,
    weight_decay=0.0,
    verbose=True,
):
    """
    返回 (model, info)。model 已恢复为验证 Sharpe 最好那一轮的权重。
    info: best_val_sharpe / best_epoch / train_sharpe_at_best / epochs_run
    """

    model = model.to(device)

    # AdamW: weight_decay=0 时等同于论文的 Adam
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        drop_last=False,
    )

    best_val = np.inf
    best_state = None
    best_epoch = 0
    train_at_best = np.nan

    epochs_without_improvement = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):

        # ---------------- train

        model.train()

        train_losses = []

        for batch in train_loader:

            x = batch["x"].to(device)
            vol = batch["vol"].to(device)
            fwd = batch["fwd_ret"].to(device)

            optimizer.zero_grad()

            position = model(x)

            loss = sharpe_loss(position, vol, fwd)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

            optimizer.step()

            train_losses.append(loss.item())

        # ---------------- validation（整个验证集一次性算 Sharpe）

        model.eval()

        pos_all, vol_all, fwd_all = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                pos_all.append(model(batch["x"].to(device)))
                vol_all.append(batch["vol"].to(device))
                fwd_all.append(batch["fwd_ret"].to(device))

        val_loss = sharpe_loss(
            torch.cat(pos_all), torch.cat(vol_all), torch.cat(fwd_all)
        ).item()

        train_loss = float(np.mean(train_losses))

        if verbose:
            print(
                f"epoch={epoch:03d} "
                f"train={train_loss:.5f} "
                f"val={val_loss:.5f}"
            )

        # ---------------- early stopping

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch
            train_at_best = train_loss

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            if verbose:
                print(f"Early stopping at epoch {epoch}")

            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_val_sharpe": -best_val,
        "best_epoch": best_epoch,
        "train_sharpe_at_best": -train_at_best,
        "epochs_run": epoch,
    }

    return model, info
