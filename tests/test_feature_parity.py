"""Read-only integrity checks: feature parity, splits, quantile ordering, live handler, loss and CLMM math.
Run from the project root: python tests/test_feature_parity.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))

from config import NUM_FEATURES, SEQ_LEN, MIN_LIVE_CANDLES, EMBARGO_GAP, split_ranges, validate_bounds, skewed_bounds
from dataset import CryptoTimeSeriesDataset, build_single_window_features
from model import OrcaCryptoTransformer
from live_predict import handle_request
from backtest_historical import compute_clm_position, evaluate_clm_hour
from train import OrcaAsymmetricQuantileLoss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "sol_usdc_data.csv")


def candles_from(df, start, end):
    return [
        dict(time=int(df.timestamp[j]), open=float(df.open[j]), high=float(df.high[j]),
             low=float(df.low[j]), close=float(df.close[j]), volume=float(df.volume[j]))
        for j in range(start, end)
    ]


def test_feature_parity():
    df = pd.read_csv(CSV)
    ds = CryptoTimeSeriesDataset(CSV)
    ts = df.timestamp.to_numpy()
    checked = 0
    for k in np.linspace(MIN_LIVE_CANDLES, len(ds) - 1, 40, dtype=int):
        start = k + SEQ_LEN - MIN_LIVE_CANDLES  # exactly MIN_LIVE_CANDLES candles ending at window end
        if (np.diff(ts[start:k + SEQ_LEN]) != 3600).any():
            continue  # live path requires contiguous candles
        x, _, h, d = ds[int(k)]
        xl, hl, dl, _ = build_single_window_features(candles_from(df, start, k + SEQ_LEN))
        assert xl.shape == (SEQ_LEN, NUM_FEATURES)
        np.testing.assert_allclose(xl, x.numpy(), atol=1e-3, err_msg=f"window {k}")
        assert hl == h.tolist() and dl == d.tolist()
        checked += 1
    assert checked >= 20, f"too few windows checked: {checked}"


def test_split_embargo():
    n = 17000
    s = split_ranges(n)
    # window i covers candles [i, i + SEQ_LEN] (inclusive of the target candle)
    assert s["val"].start - (s["train"].stop - 1) >= SEQ_LEN + 1
    # real dataset: last candle touched by a train window (index i covers candles i..i+SEQ_LEN) < first val candle
    ds = CryptoTimeSeriesDataset(CSV)
    r = split_ranges(len(ds))
    tr, va, te = ds.valid_indices(r["train"]), ds.valid_indices(r["val"]), ds.valid_indices(r["test"])
    assert tr and va and te
    assert max(tr) + SEQ_LEN < min(va) and max(va) + SEQ_LEN < min(te)
    assert s["test"].start - (s["val"].stop - 1) >= SEQ_LEN + 1
    assert EMBARGO_GAP >= SEQ_LEN + 1


def test_quantiles_non_crossing():
    m = OrcaCryptoTransformer(num_features=NUM_FEATURES).eval()
    x = torch.randn(512, SEQ_LEN, NUM_FEATURES) * 10
    h = torch.randint(0, 24, (512, SEQ_LEN))
    d = torch.randint(0, 7, (512, SEQ_LEN))
    out = m(x, h, d)
    assert out.shape == (512, 3)
    assert (out[:, 0] - out[:, 1] >= m.min_spread - 1e-6).all()


def test_live_handler_errors_are_json():
    m = OrcaCryptoTransformer(num_features=NUM_FEATURES).eval()
    df = pd.read_csv(CSV)
    good = candles_from(df, len(df) - MIN_LIVE_CANDLES, len(df))
    price = float(df.close.iloc[-1])
    ts = df.timestamp.to_numpy()
    assert (np.diff(ts[-MIN_LIVE_CANDLES:]) == 3600).all(), "tail of CSV has a gap; pick another slice"

    now = float(ts[-1]) + 3600  # fixed "now" right after the last closed candle
    ok = handle_request(m, "cpu", json.dumps({"current_price": price, "candles": good}), now=now)
    assert ok["status"] == "success", ok
    assert ok["regime"] == "SKIPPED_CASH" or ok["pred_high"] > ok["pred_low"], ok
    json.dumps(ok, allow_nan=False)

    bad_inputs = [
        "not json", "[]", "null", json.dumps({}),
        json.dumps({"candles": good}),                                   # no price
        json.dumps({"current_price": 0, "candles": good}),
        json.dumps({"current_price": price * 2, "candles": good}),
        json.dumps({"current_price": price, "candles": good[:50]}),
        json.dumps({"current_price": price, "candles": "x" * 500}),
        json.dumps({"current_price": price, "candles": [{"time": 1}] * MIN_LIVE_CANDLES}),
        json.dumps({"current_price": price, "candles": good[:-1] + [dict(good[-1], close=float("nan"))]}),
    ]
    bad_ohlc = good[:-1] + [dict(good[-1], high=good[-1]["low"] * 0.9)]
    bad_inputs += [json.dumps({"current_price": price, "candles": bad_ohlc})]
    for msg in bad_inputs:
        r = handle_request(m, "cpu", msg, now=now)
        assert r["status"] == "error", (msg[:60], r)
        json.dumps(r, allow_nan=False)

    stale = handle_request(m, "cpu", json.dumps({"current_price": price, "candles": good}), now=now + 10 * 3600)
    assert stale["status"] == "error" and "Stale" in stale["message"], stale


def test_clmm_math():
    # one-sided allocation
    _, x0, y0 = compute_clm_position(90.0, 100.0, 110.0, 1000.0)   # price below pa -> all SOL
    assert y0 == 0.0 and abs(x0 * 90.0 - 1000.0) < 1e-6
    _, x0, y0 = compute_clm_position(120.0, 100.0, 110.0, 1000.0)  # price above pb -> all USDC
    assert x0 == 0.0 and abs(y0 - 1000.0) < 1e-9
    l, x0, y0 = compute_clm_position(105.0, 100.0, 110.0, 1000.0)
    assert abs(x0 * 105.0 + y0 - 1000.0) < 1e-6
    # unchanged price -> no impermanent loss; fees are non-negative and zero without volume
    _, _, v1, hodl, il, fee = evaluate_clm_hour(105.0, 100.0, 110.0, l, x0, y0, 0.0, 106.0, 104.0, 4.3e6)
    assert abs(il) < 1e-6 and fee == 0.0 and abs(v1 - 1000.0) < 1e-6
    # guards
    for bad in [(100.0, 100.0, 100.0), (110.0, 100.0, 105.0), (-1.0, 10.0, 5.0), (100.0, 90.0, 0.0)]:
        try:
            validate_bounds(*bad[:2], bad[2])
        except ValueError:
            continue
        raise AssertionError(f"validate_bounds accepted {bad}")
    pa, pb = skewed_bounds(100.0, 99.0, 101.0)
    assert 0 < pa < pb and pb >= 100.0 * 1.004


def test_loss_sane():
    loss = OrcaAsymmetricQuantileLoss()
    t = torch.tensor([[1.01, 0.99, 1.0]])
    perfect = loss(t, t)
    worse_low = loss(torch.tensor([[1.01, 1.02, 1.0]]), t)   # pred_low above realized low (pierced)
    assert torch.isfinite(perfect) and worse_low > perfect


if __name__ == "__main__":
    for t in (test_feature_parity, test_split_embargo, test_quantiles_non_crossing, test_live_handler_errors_are_json,
              test_clmm_math, test_loss_sane):
        t()
        print("OK", t.__name__)
