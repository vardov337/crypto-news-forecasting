"""LSTM на PyTorch: окна наблюдений, стандартизованная цель, ранняя остановка, несколько зёрен (задача 7.4).

Вход — последовательность признаков строк t − L + 1, …, t (L — lookback), выход — прогноз
доходности следующего часа. Два слоя LSTM, второй вдвое меньше первого, dropout между
ними и перед выходным слоем. Признаки и цель стандартизуются по обучающему окну фолда.
Окно, в котором есть хотя бы одна непригодная строка (пропуск биржи), не используется.
Ранняя остановка — по последним 10% обучающих окон; берутся веса лучшей эпохи.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptonews.features import TARGET
from cryptonews.validation import Window


def complete_windows(valid: np.ndarray, length: int) -> np.ndarray:
    """ok[t] — все строки t − length + 1, …, t пригодны."""
    bad = np.concatenate([[0], np.cumsum(~valid.astype(bool))])
    ok = np.zeros(len(valid), dtype=bool)
    if len(valid) >= length:
        ok[length - 1:] = (bad[length:] - bad[:len(valid) - length + 1]) == 0
    return ok


def build_net(n_features: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.LSTM(n_features, hidden, batch_first=True)
            self.second = nn.LSTM(hidden, max(1, hidden // 2), batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(max(1, hidden // 2), 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.first(x)
            out, _ = self.second(self.drop(out))
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    return Net()


class LSTMModel:
    name, stochastic, needs_grid = "lstm", True, True

    def __init__(self, dropout: float, learning_rate: float, batch_size: int, max_epochs: int, patience: int,
                 scale_target: bool, early_stopping_fraction: float, compute_device: str | None = None):
        self.dropout, self.lr, self.batch = float(dropout), float(learning_rate), int(batch_size)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.scale_target, self.fraction = bool(scale_target), float(early_stopping_fraction)
        self.device = compute_device

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        length, hidden = int(params["lookback"]), int(params["hidden_size"])
        columns = list(columns)
        index = table.index
        valid = table["valid"].to_numpy(bool)
        X = table[columns].to_numpy(np.float64)
        y = table[TARGET].to_numpy(np.float64)

        fit_mask = valid & (index < window.train_end)
        mu, sd = X[fit_mask].mean(axis=0), X[fit_mask].std(axis=0)
        sd[~(sd > 0)] = 1.0
        y_mu, y_sd = (y[fit_mask].mean(), y[fit_mask].std()) if self.scale_target else (0.0, 1.0)
        Xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
        ys = np.nan_to_num((y - y_mu) / y_sd).astype(np.float32)

        ok = complete_windows(valid, length)
        train_rows = np.flatnonzero(ok & (index < window.train_end))
        test_rows = np.flatnonzero(ok & (index >= window.test_start) & (index < window.test_end))
        n_val = max(1, int(round(len(train_rows) * self.fraction)))
        fit_rows, val_rows = train_rows[:-n_val], train_rows[-n_val:]

        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        X_t = torch.tensor(Xs, device=device)
        y_t = torch.tensor(ys, device=device)
        offsets = torch.arange(-length + 1, 1, device=device)

        def windows(rows: np.ndarray):
            r = torch.as_tensor(rows, device=device, dtype=torch.long)
            return X_t[r[:, None] + offsets], y_t[r]

        net = build_net(len(columns), hidden, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(int(seed))

        def predict(rows: np.ndarray) -> np.ndarray:
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 4096):
                    xb, _ = windows(rows[i:i + 4096])
                    out.append(net(xb).float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

        best, best_state, wait, epochs = np.inf, None, 0, 0
        for epoch in range(self.max_epochs):
            net.train()
            order = fit_rows[torch.randperm(len(fit_rows), generator=generator).numpy()]
            for i in range(0, len(order), self.batch):
                xb, yb = windows(order[i:i + self.batch])
                optimizer.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                optimizer.step()
            epochs = epoch + 1
            val_loss = float(np.mean((predict(val_rows) - ys[val_rows]) ** 2))
            if val_loss < best - 1e-12:
                best, wait = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        pred = predict(test_rows).astype(float) * y_sd + y_mu
        return (pd.Series(pred, index=index[test_rows], name="y_pred"),
                {"device": device, "epochs": epochs, "best_val_mse_scaled": float(best),
                 "train_windows": int(len(fit_rows)), "val_windows": int(len(val_rows))})
