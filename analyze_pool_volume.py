"""Measured pool volume from on-chain snapshots vs Bybit SOLUSDC volume.

Read-only analysis: reads data/pool_snapshots.csv, downloads public Bybit hourly candles
(never touches data/sol_usdc_data.csv), prints a per-interval table and totals.

Volume derivation (consecutive snapshots a -> b):
    LP fees   = (fee_growth_b - fee_growth_a) / 2^64 * L_raw   (A in SOL lamports, B in USDC micro-units)
                with L_raw = mean of the two snapshots (approximation: L changes inside the hour)
    volume    = (fees_A_sol * price + fees_B_usdc) / (fee_rate * (1 - protocol_share))
Bybit candles are spread uniformly inside their hour and integrated over each snapshot interval.

Usage: python analyze_pool_volume.py [--since "2026-11-14 23:00"] [--out volume_analysis.csv]
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import requests

SNAPSHOTS = "data/pool_snapshots.csv"
BYBIT_URL = "https://api.bybit.com/v5/market/kline"
Q64 = 2 ** 64
MAX_GAP_MIN = 90  # intervals longer than this are skipped (missed snapshots)


def pool_volume(snap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, len(snap)):
        a, b = snap.iloc[i - 1], snap.iloc[i]
        dt_min = (b.ts_utc - a.ts_utc).total_seconds() / 60
        if dt_min > MAX_GAP_MIN:
            print(f"skip gap {a.ts_utc} -> {b.ts_utc} ({dt_min:.0f} min)", file=sys.stderr)
            continue
        liq = (a.liquidity_raw + b.liquidity_raw) / 2
        fee_a = (b.fee_growth_global_a - a.fee_growth_global_a) / Q64 * liq / 1e9  # SOL
        fee_b = (b.fee_growth_global_b - a.fee_growth_global_b) / Q64 * liq / 1e6  # USDC
        px = (a.price + b.price) / 2
        eff_fee = (b.fee_rate / 1e6) * (1 - b.protocol_fee_rate / 1e4)
        vol_usd = (fee_a * px + fee_b) / eff_fee
        rows.append({"start": a.ts_utc, "end": b.ts_utc, "dt_min": dt_min, "price": px,
                     "fee_sol": fee_a, "fee_usdc": fee_b, "pool_usd": vol_usd,
                     "pool_sol": vol_usd / px})
    return pd.DataFrame(rows)


def fetch_bybit(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Closed or partial hourly SOLUSDC spot candles covering [start, end]."""
    out, cur = [], int(start.floor("h").timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        r = requests.get(BYBIT_URL, timeout=20, params={
            "category": "spot", "symbol": "SOLUSDC", "interval": "60",
            "start": cur, "end": end_ms, "limit": 1000})
        r.raise_for_status()
        lst = r.json()["result"]["list"]
        if not lst:
            break
        out += lst
        cur = max(int(x[0]) for x in lst) + 3_600_000
        time.sleep(0.2)
    c = pd.DataFrame(out, columns=["t", "o", "h", "l", "c", "vol", "turnover"]).astype(float)
    c = c.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    c["cstart"] = pd.to_datetime(c.t, unit="ms")
    c["cend"] = c.cstart + pd.Timedelta(hours=1)
    return c


def bybit_in_interval(c: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp):
    ov = (c.cend.clip(upper=b) - c.cstart.clip(lower=a)).dt.total_seconds().clip(lower=0) / 3600
    return (c.vol * ov).sum(), (c.turnover * ov).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-14 23:00",
                    help="UTC; default skips the manual install-time rows")
    ap.add_argument("--out", default=None, help="optional CSV with the per-interval table")
    args = ap.parse_args()

    snap = pd.read_csv(SNAPSHOTS, parse_dates=["ts_utc"])
    snap = snap[snap.ts_utc >= pd.Timestamp(args.since)].reset_index(drop=True)
    if len(snap) < 2:
        sys.exit("not enough snapshots after --since")
    r = pool_volume(snap)
    c = fetch_bybit(snap.ts_utc.iloc[0], snap.ts_utc.iloc[-1])
    r[["bybit_sol", "bybit_usd"]] = [bybit_in_interval(c, a, b) for a, b in zip(r.start, r.end)]
    r["ratio"] = r.pool_usd / r.bybit_usd

    pd.set_option("display.width", 220)
    cols = ["start", "end", "price", "fee_sol", "fee_usdc", "pool_usd", "bybit_usd", "ratio"]
    print(r[cols].to_string(float_format=lambda x: f"{x:,.2f}"))
    days = r.dt_min.sum() / 1440
    print(f"\nintervals: {len(r)}, covered: {r.dt_min.sum() / 60:.1f} h")
    print(f"pool   USD {r.pool_usd.sum() / 1e6:.2f}M  (~{r.pool_usd.sum() / days / 1e6:.0f}M/day)   "
          f"Bybit USD {r.bybit_usd.sum() / 1e6:.2f}M (~{r.bybit_usd.sum() / days / 1e6:.1f}M/day)")
    print(f"ratio pool/Bybit: sum {r.pool_usd.sum() / r.bybit_usd.sum():.1f}x, "
          f"hourly mean {r.ratio.mean():.1f}x, median {r.ratio.median():.1f}x, "
          f"min {r.ratio.min():.1f}x, max {r.ratio.max():.1f}x")
    if args.out:
        r.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
