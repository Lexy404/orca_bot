import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from config import NUM_FEATURES, SEQ_LEN, MIN_LIVE_CANDLES


def compute_engineered_features(df: pd.DataFrame) -> np.ndarray:
    """
    Computes an 11-dimensional feature matrix from raw OHLCV data:
      0: open (raw, normalized per window in __getitem__)
      1: high (raw, normalized per window in __getitem__)
      2: low (raw, normalized per window in __getitem__)
      3: close (raw, normalized per window in __getitem__)
      4: volume (raw, log-normalized per window in __getitem__)
      5: log_return (log(close_t / close_{t-1}))
      6: hl_range_pct ((high - low) / close) — instantaneous intrabar range
      7: realized_vol_6h (rolling 6-hour std of log_return)
      8: natr_14 (14-period EMA of gap-aware True Range normalized by close)
      9: chaikin_vol (Chaikin Volatility: 10-period ROC of 10-period EMA of (high - low))
      10: clv (Close Location Value: ((close - low) - (high - close)) / (high - low) in [-1, 1])
    """
    df_feat = df.copy()
    close = df_feat["close"].astype(np.float64).clip(lower=1e-8)
    high = df_feat["high"].astype(np.float64)
    low = df_feat["low"].astype(np.float64)
    prev_close = close.shift(1)

    # 1. Log returns & instantaneous intrabar range
    log_ret = np.log(close / prev_close).fillna(0.0)
    hl_span = (high - low).clip(lower=0.0)
    hl_range = (hl_span / close).fillna(0.0)
    rvol_6h = log_ret.rolling(window=6, min_periods=1).std().fillna(0.0)

    # 2. Gap-aware True Range & Normalized ATR (NATR-14)
    tr1 = hl_span
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(hl_span)
    atr_14 = true_range.ewm(span=14, adjust=False).mean()
    natr_14 = (atr_14 / close).fillna(0.0)

    # 3. Chaikin Volatility (10-period ROC of 10-period EMA of High-Low range)
    hl_ema_10 = hl_span.ewm(span=10, adjust=False).mean()
    hl_ema_lag_10 = hl_ema_10.shift(10)
    chaikin_vol = ((hl_ema_10 - hl_ema_lag_10) / hl_ema_lag_10.replace(0.0, np.nan)).fillna(0.0)
    chaikin_vol = chaikin_vol.clip(lower=-5.0, upper=5.0)

    # 4. Close Location Value (CLV) — directional intrabar buying/selling pressure in [-1.0, +1.0]
    clv = (((close - low) - (high - close)) / hl_span.replace(0.0, np.nan)).fillna(0.0)
    clv = clv.clip(lower=-1.0, upper=1.0)

    df_feat["log_return"] = log_ret
    df_feat["hl_range_pct"] = hl_range
    df_feat["realized_vol_6h"] = rvol_6h
    df_feat["natr_14"] = natr_14
    df_feat["chaikin_vol"] = chaikin_vol
    df_feat["clv"] = clv

    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
        "hl_range_pct",
        "realized_vol_6h",
        "natr_14",
        "chaikin_vol",
        "clv",
    ]
    return df_feat[cols].to_numpy(dtype=np.float32)


def normalize_window(x_raw: np.ndarray, base_price: float) -> np.ndarray:
    """In-place window normalization shared by the offline dataset and the live path."""
    # 1. Price features as relative deviations around 0 (scaled by 10x)
    x_raw[:, 0:4] = (x_raw[:, 0:4] / base_price - 1.0) * 10.0

    # 2. Volume log-normalization relative to window mean
    x_raw[:, 4] = np.log1p(np.maximum(x_raw[:, 4], 0.0))
    vol_mean = x_raw[:, 4].mean()
    if vol_mean > 0:
        x_raw[:, 4] = (x_raw[:, 4] / vol_mean) - 1.0

    # 3. Percentage volatility/return features (indices 5..8) scaled by 100x
    x_raw[:, 5:9] = x_raw[:, 5:9] * 100.0

    # 4. Chaikin Volatility (9) and CLV (10) are already O(1)
    return x_raw


def build_single_window_features(candles: list[dict]) -> tuple[np.ndarray, list[int], list[int], float]:
    """
    Transforms a list of closed 1-hour candle dicts (at least MIN_LIVE_CANDLES; the leading candles are used for
    EMA/ROC warm-up and the latest SEQ_LEN form the model window)
    into a normalized (72, 11) feature array, hour/day index lists, and current spot price.
    """
    if len(candles) < MIN_LIVE_CANDLES:
        raise ValueError(f"Expected at least {MIN_LIVE_CANDLES} candles, got {len(candles)}")

    df = pd.DataFrame(candles)
    for col in ["time", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing candle field: {col}")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)
    df["time"] = df["time"].astype(np.int64)

    if not np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("Candles contain NaN/Inf values")
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    if (
        (np.minimum.reduce([o, h, l, c]) <= 0).any()
        or (v < 0).any()
        or (h < np.maximum(o, c)).any()
        or (l > np.minimum(o, c)).any()
        or (h < l).any()
    ):
        raise ValueError("Candles violate OHLCV sanity (need low <= open,close <= high, prices > 0, volume >= 0)")
    steps = np.diff(df["time"].to_numpy())
    if (steps != 3600).any():
        raise ValueError("Candles must be strictly consecutive 1h candles (time in seconds, ascending, no gaps)")

    raw_matrix = compute_engineered_features(df)
    x_raw = raw_matrix[-SEQ_LEN:].copy()
    df_window = df.iloc[-SEQ_LEN:]

    base_price = float(x_raw[-1, 3])
    if base_price <= 0:
        raise ValueError(f"Invalid base_price encountered: {base_price}")

    normalize_window(x_raw, base_price)

    dt_series = pd.to_datetime(df_window["time"], unit="s", utc=True)
    hours = dt_series.dt.hour.tolist()
    days = dt_series.dt.day_of_week.tolist()

    return x_raw.astype(np.float32), hours, days, base_price


class CryptoTimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for Solana CLMM liquidity range and trend prediction.
    Input X: (history_len, 11) normalized OHLCV + NATR-14 + Chaikin Volatility + CLV indicators.
    Target Y: [target_high_coef, target_low_coef, target_trend_coef] over forecast_len hours.
    """

    def __init__(self, csv_file: str, history_len: int = SEQ_LEN, forecast_len: int = 1):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Dataset CSV not found at path: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.history_len = history_len
        self.forecast_len = forecast_len

        self.data = compute_engineered_features(self.df)

        # A window (history + forecast candles) is valid only if its candles are consecutive 1h bars.
        ts = self.df["timestamp"].to_numpy(dtype=np.int64)
        gap = (np.diff(ts) != 3600).astype(np.int64)
        gap_cum = np.concatenate([[0], np.cumsum(gap)])
        span = history_len + forecast_len
        n = max(0, len(ts) - span + 1)
        self.valid_mask = np.array(
            [gap_cum[i + span - 1] - gap_cum[i] == 0 for i in range(n)], dtype=bool
        )

        self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=True)
        self.hours = self.df["datetime"].dt.hour.to_numpy(dtype=np.int64)
        self.days_of_week = self.df["datetime"].dt.day_of_week.to_numpy(dtype=np.int64)

    def valid_indices(self, index_range: range) -> list[int]:
        """Indices in index_range whose window contains no time gap."""
        return [i for i in index_range if self.valid_mask[i]]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.history_len - self.forecast_len + 1)

    def __getitem__(self, idx: int):
        x_start = idx
        x_end = idx + self.history_len
        x_raw = self.data[x_start:x_end].copy()

        base_price = float(x_raw[-1, 3])

        # Extract future target window before in-place normalization
        y_start = x_end
        y_end = x_end + self.forecast_len
        y_raw = self.data[y_start:y_end]

        future_highs = y_raw[:, 1]
        future_lows = y_raw[:, 2]

        target_high = float(future_highs.max()) / base_price
        target_low = float(future_lows.min()) / base_price
        target_trend = float(y_raw[-1, 3]) / base_price

        normalize_window(x_raw, base_price)

        target = torch.tensor([target_high, target_low, target_trend], dtype=torch.float32)
        window_hours = torch.tensor(self.hours[x_start:x_end], dtype=torch.long)
        window_days = torch.tensor(self.days_of_week[x_start:x_end], dtype=torch.long)

        return torch.tensor(x_raw, dtype=torch.float32), target, window_hours, window_days


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "sol_usdc_data.csv")
    dataset = CryptoTimeSeriesDataset(csv_path)
    print(f"Successfully constructed {len(dataset)} sliding windows for training.")

    sample_x, sample_y, sample_hours, sample_days = dataset[20]
    print("\n--- Tensor Dimension Report ---")
    print(f"X (72h history, {NUM_FEATURES} features): {sample_x.shape}")
    print(f"Y (Target [High, Low, Trend]): {sample_y.shape} -> Values: {sample_y.tolist()}")
    print(f"Hours (Time embeddings): {sample_hours.shape}")
    print(f"Days of week (Time embeddings): {sample_days.shape}")