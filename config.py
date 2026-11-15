"""Shared constants for model, dataset, training, backtest and live inference.

Single source of truth: change here, not in individual scripts.
"""

NUM_FEATURES = 11
SEQ_LEN = 72
ZMQ_PORT_DEFAULT = 5555
SEED = 42  # used by pretrain.py / train.py for reproducibility

# Model output parameterization (must be identical at training and serving time)
MIN_SPREAD = 0.0015        # pred_high - pred_low >= MIN_SPREAD (relative to price)
LOW_OFFSET_SCALE = 0.015   # pred_low   = 1 - softplus(raw) * LOW_OFFSET_SCALE
SPREAD_SCALE = 0.030       # pred_high  = pred_low + MIN_SPREAD + softplus(raw) * SPREAD_SCALE
MAX_TREND_DEV = 0.10       # pred_trend = 1 + tanh(raw) * MAX_TREND_DEV

# Live inference needs extra history so EMA / ROC indicators are warmed up the same way
# as in training (where they see the full history). Parity with the offline pipeline is
# verified by test_feature_parity.py.
WARMUP_CANDLES = 78
MIN_LIVE_CANDLES = SEQ_LEN + WARMUP_CANDLES
MAX_LIVE_CANDLES = 2000
MAX_PRICE_DEVIATION = 0.05   # live: reject current_price farther than this from the last close
MAX_CANDLE_AGE_SEC = 3 * 3600  # live: reject if the last candle's open time is older than this

# Chronological split of window indices: [train | embargo | val | embargo | test]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EMBARGO_GAP = SEQ_LEN + 1  # input window + 1 forecast step


def split_ranges(n_windows: int) -> dict[str, range]:
    """Index ranges of train / val / test windows, separated by EMBARGO_GAP windows."""
    train_end = int(TRAIN_FRAC * n_windows)
    val_start = train_end + EMBARGO_GAP
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n_windows)
    test_start = val_end + EMBARGO_GAP
    return {
        "train": range(0, train_end),
        "val": range(val_start, val_end),
        "test": range(test_start, n_windows),
    }


# Regime policy thresholds (train monitoring, backtest and live must agree)
CRITICAL_DROP_THRESHOLD = 0.994   # pred_trend < this  -> SKIPPED_CASH
MODERATE_DROP_THRESHOLD = 0.9975  # pred_trend < this  -> SKEW_RANGE_DOWN
SKEW_LOW_SHIFT = 0.30             # pred_low  -= SKEW_LOW_SHIFT * spread
SKEW_HIGH_SHIFT = 0.15            # pred_high -= SKEW_HIGH_SHIFT * spread
SKEW_MIN_HIGH_MULT = 1.004        # pred_high >= price * this


def skewed_bounds(price: float, raw_low: float, raw_high: float) -> tuple[float, float]:
    """Range for the SKEW_RANGE_DOWN regime (shared by backtest and live)."""
    spread = raw_high - raw_low
    pa = raw_low - SKEW_LOW_SHIFT * spread
    pb = max(raw_high - SKEW_HIGH_SHIFT * spread, price * SKEW_MIN_HIGH_MULT)
    return pa, pb


def validate_bounds(pa: float, pb: float, price: float) -> None:
    """CLMM guard shared by backtest and live: raises ValueError for unusable ranges."""
    if not (price > 0 and pa > 0 and pa < pb):
        raise ValueError(f"Invalid CLMM bounds: price={price}, pa={pa}, pb={pb}")


# Training loss (train.py). The crash threshold is tied to the regime policy on purpose.
LOSS_Q_LOW = 0.10
LOSS_Q_HIGH = 0.90
LOSS_Q_TREND = 0.42
LOSS_IL_PENALTY = 2.0        # extra weight when the realized low pierces pred_low
LOSS_WIDTH_PENALTY = 0.08
LOSS_TREND_WEIGHT = 1.5
LOSS_CRASH_PENALTY = 2.0
LOSS_DIRECTION_WEIGHT = 0.20
LOSS_CRASH_TARGET = CRITICAL_DROP_THRESHOLD  # target_trend below this counts as a crash
