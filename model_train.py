"""
Step 3 — Model Selection for A-share stock price prediction.

Trains and evaluates four models on the feature matrix from Step 2:
  Baselines : Linear Regression, Random Forest
  Deep learning : LSTM, Transformer  (sequence prediction)

Evaluation metrics: RMSE, MAPE, directional accuracy (DA).

Install:
    pip install scikit-learn torch pandas numpy

Usage:
    from model_train import ModelSuite
    suite = ModelEngine(sequence_len=20)
    X, y, dates = suite.prepare(features_df, target="ret_1d_fwd", horizon=1)
    results = suite.run_all(X, y, dates)
    print(results.sort_values("rmse"))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ====================================================================== #
#  Metrics
# ====================================================================== #
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of days where sign(actual) == sign(predicted)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")
    actual_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(actual_dir == pred_dir) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_acc": directional_accuracy(y_true, y_pred),
    }


# ====================================================================== #
#  Data preparation: tabular + sequence views
# ====================================================================== #
@dataclass
class ModelEngine:
    """Prepares data, trains baselines + DL models, and evaluates them."""

    sequence_len: int = 20           # lookback window for sequence models
    target: str = "ret_1d_fwd"       # forward return to predict
    test_size: float = 0.2
    random_state: int = 42

    # populated after prepare()
    feature_cols: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def make_target(self, df: pd.DataFrame, horizon: int = 1,
                    price_col: str = "close") -> pd.DataFrame:
        """Add a forward-return target column."""
        df = df.copy()
        df[self.target] = df[price_col].pct_change(horizon).shift(-horizon)
        return df

    # ------------------------------------------------------------------ #
    def prepare(self, df: pd.DataFrame, horizon: int = 1,
                dropna: bool = True):
        """
        Build tabular X/y and a sequence view for DL models.

        Returns
        -------
        X_tab  : (n, f) ndarray   — for Linear/RF
        X_seq  : (n, seq_len, f)  — for LSTM/Transformer
        y      : (n,)             — forward returns
        dates  : DatetimeIndex aligned to the last row of each sample
        """
        df = self.make_target(df, horizon=horizon)

        # exclude target + non-feature columns
        exclude = {self.target, "symbol", "event_type"}  # event_type_code kept
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype.kind in "iufb"
        ]
        self.feature_cols = feature_cols

        feats = df[feature_cols].astype(float)
        y = df[self.target].astype(float)

        if dropna:
            valid = y.notna() & feats.notna().all(axis=1)
            feats, y = feats.loc[valid], y.loc[valid]

        # ----- tabular view -----
        X_tab = feats.values
        # ----- sequence view: sliding windows -----
        L = self.sequence_len
        X_seq, y_seq, dates = [], [], []
        vals = feats.values
        idx = feats.index
        for i in range(L, len(vals)):
            X_seq.append(vals[i - L:i])
            y_seq.append(y.iloc[i])
            dates.append(idx[i])
        X_seq = np.asarray(X_seq, dtype=np.float32)
        y_seq = np.asarray(y_seq, dtype=np.float32)
        dates = pd.DatetimeIndex(dates)
        return X_tab[L:], y_seq, X_seq, dates

    # ------------------------------------------------------------------ #
    def split_indices(self, n: int):
        """Chronological train/test split (no shuffling — time series)."""
        n_test = int(n * self.test_size)
        return np.arange(0, n - n_test), np.arange(n - n_test, n)

    # ================================================================== #
    #  Baselines
    # ================================================================== #
    def train_linear(self, X_train, y_train, X_test):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        model = LinearRegression().fit(scaler.transform(X_train), y_train)
        return model.predict(scaler.transform(X_test))

    def train_random_forest(self, X_train, y_train, X_test):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1,
            random_state=self.random_state,
        ).fit(X_train, y_train)
        return model.predict(X_test)

    # ================================================================== #
    #  Deep learning: LSTM & Transformer
    # ================================================================== #
    def _torch(self):
        import torch
        torch.manual_seed(self.random_state)
        return torch

    def train_lstm(self, X_train, y_train, X_test, epochs=30, lr=1e-3, batch=64):
        torch = self._torch()
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        f = X_train.shape[2]

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(f, 64, num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return self._train_seq(LSTMNet, X_train, y_train, X_test,
                               epochs, lr, batch, device)

    def train_transformer(self, X_train, y_train, X_test, epochs=30, lr=1e-3, batch=64):
        torch = self._torch()
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        f = X_train.shape[2]
        L = X_train.shape[1]

        class TransformerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(f, 64)
                self.pos = nn.Parameter(torch.randn(1, L, 64) * 0.02)
                enc = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128,
                                                 dropout=0.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, num_layers=2)
                self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                x = self.proj(x) + self.pos
                x = self.encoder(x)
                return self.head(x[:, -1, :]).squeeze(-1)

        return self._train_seq(TransformerNet, X_train, y_train, X_test,
                               epochs, lr, batch, device)

    def _train_seq(self, NetClass, X_train, y_train, X_test,
                   epochs, lr, batch, device):
        """Shared training loop for sequence models."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
        yt = torch.tensor(y_train, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)

        model = NetClass().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32,
                                       device=device)).cpu().numpy()
        return preds

    # ================================================================== #
    #  Orchestration
    # ================================================================== #
    def run_all(self, X_tab, y, X_seq, dates) -> pd.DataFrame:
        """Train every model and return a metrics comparison table."""
        tr, te = self.split_indices(len(y))
        results = {}

        print("→ Linear Regression")
        results["linear"] = evaluate(
            y[te], self.train_linear(X_tab[tr], y[tr], X_tab[te]))

        print("→ Random Forest")
        results["random_forest"] = evaluate(
            y[te], self.train_random_forest(X_tab[tr], y[tr], X_tab[te]))

        print("→ LSTM")
        results["lstm"] = evaluate(
            y[te], self.train_lstm(X_seq[tr], y[tr], X_seq[te]))

        print("→ Transformer")
        results["transformer"] = evaluate(
            y[te], self.train_transformer(X_seq[tr], y[tr], X_seq[te]))

        return pd.DataFrame(results).T.sort_values("rmse")


# ---------------------------------------------------------------------- #
#  Self-test on the Step-2 feature matrix (or synthetic if absent)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    path = "data_cache/features.parquet"
    if os.path.exists(path):
        feats = pd.read_parquet(path)
    else:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-02", "2024-06-28")
        n = len(idx)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
        feats = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n),
                              "ma5": close, "ma20": close, "rsi": 50.0,
                              "macd": 0.0, "bb_width": 0.02}, index=idx)
        feats.index.name = "date"

    suite = ModelEngine(sequence_len=20, test_size=0.2)
    X_tab, y, X_seq, dates = suite.prepare(feats)
    print(f"tabular: {X_tab.shape}  sequence: {X_seq.shape}  samples: {len(y)}")

    results = suite.run_all(X_tab, y, X_seq, dates)
    print("\n=== Model comparison ===")
    print(results.round(4))
